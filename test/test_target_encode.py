"""Tests for the target-mean kernels: the three formulas, the target column, and the artifact.

The last two tests cover all six target encoders: an integer target above 2^53, and a float target at the magnitude
bound.
"""

import json
import math
import pickle
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest

from fast_feature_transform import (
    CatBoostEncoder,
    EngineError,
    JamesSteinEncoder,
    LeaveOneOutEncoder,
    MEstimateEncoder,
    Pipeline,
    Select,
    TargetEncoder,
    Transformer,
    WOEEncoder,
    _core,
)
from helpers import read_rows, write_parquet

PRIOR: float = 0.625


@pytest.fixture
def clicks(tmp_path: Path) -> list[str]:
    """One file with four row groups: four categories against a 0/1 target.

    Args:
        tmp_path: pytest-provided temporary directory.

    Returns:
        Paths of the written Parquet files.

    """
    table: pa.Table = pa.table(
        {
            "city": ["ams", "ams", "ams", "ber", "ber", "cph", "dub", "dub"],
            "clicked": [1, 0, 1, 1, 0, 0, 1, 1],
        }
    )
    return [write_parquet(tmp_path / "clicks.parquet", table, row_group_size=2)]


def test_target_encode_blends_the_category_mean_with_the_prior(clicks: list[str], tmp_path: Path) -> None:
    encoder: TargetEncoder = TargetEncoder(target="clicked", columns=["city"], min_samples_leaf=2, smoothing=1.0).fit(
        clicks
    )
    assert encoder.prior_ == pytest.approx(PRIOR)

    def blend(n: float, s: float) -> float:
        weight: float = 1.0 / (1.0 + math.exp(-(n - 2.0) / 1.0))
        return PRIOR * (1 - weight) + (s / n) * weight

    assert encoder.values_["city"] == pytest.approx(
        {"ams": blend(3, 2), "ber": blend(2, 1), "cph": blend(1, 0), "dub": blend(2, 2)}
    )
    assert encoder.null_values_ == {"city": None}

    rows: list[dict[str, Any]] = read_rows(encoder.transform_parquet(clicks, output_dir=tmp_path / "out"))
    assert rows[0]["city"] == pytest.approx(blend(3, 2))
    assert rows[0]["clicked"] == 1  # the target passes through
    assert encoder.transform_dict({"city": "cph", "clicked": 0}) == {
        "city": pytest.approx(blend(1, 0)),
        "clicked": 0,
    }
    # An unseen category and a null unseen at fit are both worth the prior.
    assert encoder.transform_dict({"city": "osl"}) == {"city": pytest.approx(PRIOR)}
    assert encoder.transform_dict({"city": None}) == {"city": pytest.approx(PRIOR)}


def test_m_estimate_shrinks_toward_the_prior_and_gives_up_on_an_id_like_column(
    clicks: list[str], tmp_path: Path
) -> None:
    encoder: MEstimateEncoder = MEstimateEncoder(target="clicked", columns=["city"], m=2.0).fit(clicks)
    assert encoder.values_["city"] == pytest.approx(
        {
            "ams": (2 + PRIOR * 2) / (3 + 2),
            "ber": (1 + PRIOR * 2) / (2 + 2),
            "cph": (0 + PRIOR * 2) / (1 + 2),
            "dub": (2 + PRIOR * 2) / (2 + 2),
        }
    )
    # m = 0 is the plain category mean.
    plain: MEstimateEncoder = MEstimateEncoder(target="clicked", columns=["city"], m=0.0).fit(clicks)
    assert plain.values_["city"]["ams"] == pytest.approx(2 / 3)

    ids: str = write_parquet(
        tmp_path / "ids.parquet",
        pa.table({"city": ["ams", "ber", "cph", "dub"], "clicked": [1, 0, 1, 1]}),
    )
    id_like: MEstimateEncoder = MEstimateEncoder(target="clicked", columns=["city"]).fit([ids])
    assert id_like.values_["city"] == {"ams": 0.75, "ber": 0.75, "cph": 0.75, "dub": 0.75}
    assert id_like.transform_dict({"city": "ams"}) == {"city": 0.75}


def test_woe_encode_needs_a_binary_target_and_centres_on_zero(clicks: list[str], tmp_path: Path) -> None:
    encoder: WOEEncoder = WOEEncoder(target="clicked", columns=["city"]).fit(clicks)

    def woe(n: float, s: float, r: float = 1.0) -> float:
        return math.log(((s + r) / (5 + 2 * r)) / (((n - s) + r) / ((8 - 5) + 2 * r)))

    assert encoder.values_["city"] == pytest.approx(
        {"ams": woe(3, 2), "ber": woe(2, 1), "cph": 0.0, "dub": woe(2, 2)}  # cph was seen once
    )
    assert encoder.transform_dict({"city": "osl"}) == {"city": 0.0}  # unknown is 0, not the prior
    assert encoder.transform_dict({"city": None}) == {"city": 0.0}
    tuned: WOEEncoder = WOEEncoder(target="clicked", columns=["city"], regularization=0.25).fit(clicks)
    assert tuned.values_["city"]["ams"] == pytest.approx(woe(3, 2, 0.25))

    for target, message in (
        ([1, 1, 1, 1, 1, 1, 1, 1], "only one distinct value"),
        ([0, 1, 2, 0, 1, 2, 0, 1], "more than two distinct values"),
        ([1, 2, 1, 2, 1, 2, 1, 2], "no value 0 was found"),
        ([0, -1, 0, -1, 0, -1, 0, -1], "no value 1 was found"),
    ):
        path: str = write_parquet(
            tmp_path / "woe.parquet",
            pa.table({"city": ["ams", "ber", "cph", "dub"] * 2, "clicked": target}),
        )
        with pytest.raises(EngineError, match=message):
            WOEEncoder(target="clicked", columns=["city"]).fit([path])


def test_a_null_category_is_learned_and_a_null_target_teaches_nothing(tmp_path: Path) -> None:
    path: str = write_parquet(
        tmp_path / "gaps.parquet",
        pa.table({"city": ["ams", "ams", None, None, "ber"], "clicked": [1.0, 0.0, 1.0, 1.0, None]}),
    )
    encoder: MEstimateEncoder = MEstimateEncoder(target="clicked", columns=["city"], m=0.0).fit([path])
    # The ber row carried no target, so it never reached a table: prior is 3/4.
    assert encoder.prior_ == pytest.approx(0.75)
    assert encoder.values_["city"] == {"ams": 0.5}
    assert encoder.null_values_ == {"city": 1.0}
    assert encoder.transform_dict({"city": None}) == {"city": 1.0}
    assert encoder.transform_dict({"city": "ber"}) == {"city": pytest.approx(0.75)}

    as_null: MEstimateEncoder = MEstimateEncoder(
        target="clicked", columns=["city"], m=0.0, handle_missing="return_nan"
    ).fit([path])
    assert as_null.null_values_ == {"city": None}
    assert as_null.transform_dict({"city": None}) == {"city": None}

    strict: MEstimateEncoder = MEstimateEncoder(target="clicked", columns=["city"], m=0.0, handle_unknown="error").fit(
        [path]
    )
    with pytest.raises(EngineError, match="category that was not seen at fit"):
        strict.transform_dict({"city": "osl"})
    # A null is not an unknown category: handle_unknown='error' does not fire on it.
    assert strict.transform_dict({"city": None}) == {"city": 1.0}

    with pytest.raises(EngineError, match="has nulls and handle_missing is 'error'"):
        MEstimateEncoder(target="clicked", columns=["city"], handle_missing="error").fit([path])
    clean: str = write_parquet(tmp_path / "clean.parquet", pa.table({"city": ["ams", "ber"], "clicked": [1.0, 0.0]}))
    picky: MEstimateEncoder = MEstimateEncoder(target="clicked", columns=["city"], handle_missing="error").fit([clean])
    with pytest.raises(EngineError, match="has a null and handle_missing is 'error'"):
        picky.transform_dict({"city": None})


def test_unfitted_columns_default_to_every_utf8_column_but_never_the_target(tmp_path: Path) -> None:
    path: str = write_parquet(
        tmp_path / "wide.parquet",
        pa.table({"city": ["ams", "ber"], "device": ["ios", "web"], "clicked": [1, 0]}),
    )
    encoder: TargetEncoder = TargetEncoder(target="clicked").fit([path])
    assert sorted(encoder.values_) == ["city", "device"]
    numbers: str = write_parquet(tmp_path / "numbers.parquet", pa.table({"n": [1, 2], "clicked": [1, 0]}))
    with pytest.raises(EngineError, match="no utf8 columns to encode"):
        TargetEncoder(target="clicked").fit([numbers])


def test_the_target_is_read_at_fit_only_and_can_be_dropped(clicks: list[str], tmp_path: Path) -> None:
    pipeline: Pipeline = Pipeline(
        [("encode", TargetEncoder(target="clicked", columns=["city"])), ("keep", Select(columns=["city"]))]
    ).fit(clicks)
    assert pipeline.transform_dict({"city": "ams", "clicked": 1}) == {"city": pytest.approx(0.6314360527118139)}
    # A serving file without the label still transforms.
    serving: str = write_parquet(tmp_path / "serving.parquet", pa.table({"city": ["ams", "ber"]}))
    rows: list[dict[str, Any]] = read_rows(pipeline.transform_parquet([serving], output_dir=tmp_path / "out"))
    assert sorted(rows[0]) == ["city"]


def test_target_errors_name_the_target_column(clicks: list[str], tmp_path: Path) -> None:
    with pytest.raises(EngineError, match="target column 'missing' not found in input"):
        TargetEncoder(target="missing", columns=["city"]).fit(clicks)
    strings: str = write_parquet(tmp_path / "strings.parquet", pa.table({"city": ["ams"], "clicked": ["yes"]}))
    with pytest.raises(EngineError, match="is utf8, expected an integer or float column"):
        TargetEncoder(target="clicked", columns=["city"]).fit([strings])
    numeric: str = write_parquet(tmp_path / "numeric.parquet", pa.table({"seats": [4], "clicked": [1]}))
    with pytest.raises(EngineError, match="is i64, expected utf8"):
        TargetEncoder(target="clicked", columns=["seats"]).fit([numeric])
    empty: str = write_parquet(
        tmp_path / "empty.parquet", pa.table({"city": ["ams"], "clicked": pa.array([None], pa.int64())})
    )
    with pytest.raises(EngineError, match="has no non-null values to learn from"):
        TargetEncoder(target="clicked", columns=["city"]).fit([empty])
    nan: str = write_parquet(tmp_path / "nan.parquet", pa.table({"city": ["ams"], "clicked": [float("nan")]}))
    with pytest.raises(EngineError, match="is not finite"):
        TargetEncoder(target="clicked", columns=["city"]).fit([nan])


def test_the_fitted_table_is_independent_of_the_split_and_survives_the_artifact(
    clicks: list[str], tmp_path: Path
) -> None:
    floats: str = write_parquet(
        tmp_path / "floats.parquet",
        pa.table({"city": ["ams", "ams", "ber", "ber"], "clicked": [1e100, 1.0, -1e100, 2.0]}),
        row_group_size=1,
    )
    for encoder in (
        TargetEncoder(target="clicked", columns=["city"]),
        MEstimateEncoder(target="clicked", columns=["city"]),
        WOEEncoder(target="clicked", columns=["city"]),
    ):
        states: set[str] = set()
        for threads in (1, 4):
            encoder.fit(clicks, threads=threads)
            states.add(encoder._require_handle().stage_state(type(encoder).__name__.lower()))
        assert len(states) == 1, type(encoder).__name__
    exact: set[str] = set()
    for threads in (1, 4):
        summed: MEstimateEncoder = MEstimateEncoder(target="clicked", columns=["city"], m=0.0).fit(
            [floats], threads=threads
        )
        exact.add(summed._require_handle().stage_state("mestimateencoder"))
    assert len(exact) == 1

    pipeline: Pipeline = Pipeline([("blend", WOEEncoder(target="clicked", columns=["city"]))]).fit(clicks)
    pipeline.save(tmp_path / "artifact")
    manifest: dict[str, Any] = json.loads((tmp_path / "artifact" / "manifest.json").read_text())
    state: dict[str, Any] = manifest["ops"][0]["state"]
    assert sorted(state["values"]["city"]) == ["ams", "ber", "cph", "dub"]
    assert state["null_values"] == {"city": None}
    assert state["prior"] == pytest.approx(PRIOR)
    record: dict[str, Any] = {"city": "ams", "clicked": 1}
    assert Pipeline.load(tmp_path / "artifact").transform_dict(record) == pipeline.transform_dict(record)
    assert pickle.loads(pickle.dumps(pipeline)).transform_dict(record) == pipeline.transform_dict(record)

    manifest["ops"][0]["state"]["values"]["city"]["ams"] = "high"
    with pytest.raises(EngineError, match="must be a finite number"):
        _core.FittedPipeline.load_json(json.dumps(manifest))

    # Every shell's params() must rebuild the shell: Pipeline.load calls
    # shell_cls(**op["params"]).
    for index, shell in enumerate(
        (
            TargetEncoder(target="clicked", columns=["city"], min_samples_leaf=3, smoothing=2.5),
            MEstimateEncoder(target="clicked", columns=["city"], m=4.0, handle_unknown="return_nan"),
            WOEEncoder(target="clicked", columns=["city"], regularization=0.5, handle_missing="return_nan"),
        )
    ):
        fitted: Pipeline = Pipeline([("blend", shell)]).fit(clicks)
        fitted.save(tmp_path / f"round-{index}")
        restored: Pipeline = Pipeline.load(tmp_path / f"round-{index}")
        rebuilt = restored["blend"]
        assert isinstance(rebuilt, Transformer)
        assert rebuilt.params() == shell.params()
        assert restored.transform_dict(record) == fitted.transform_dict(record)


def test_the_shells_validate_their_arguments() -> None:
    bad_option: Any = "sometimes"
    text: Any = "3"
    number: Any = 7
    for shell in (TargetEncoder, MEstimateEncoder, WOEEncoder):
        with pytest.raises(TypeError, match="target must be a column name"):
            shell(target=number)
        with pytest.raises(ValueError, match="is also the target"):
            shell(target="clicked", columns=["city", "clicked"])
        with pytest.raises(TypeError, match="not a single string"):
            shell(target="clicked", columns=text)
        with pytest.raises(ValueError, match="at least one column"):
            shell(target="clicked", columns=[])
        with pytest.raises(ValueError, match="handle_unknown must be"):
            shell(target="clicked", handle_unknown=bad_option)
        with pytest.raises(ValueError, match="handle_missing must be"):
            shell(target="clicked", handle_missing=bad_option)
    with pytest.raises(TypeError, match="min_samples_leaf must be an integer"):
        TargetEncoder(target="clicked", min_samples_leaf=text)
    # REGRESSION: 2**63 used to reach the kernel wrapped to -2**63, which
    # turned the blend weight from 0 to 1.
    with pytest.raises(ValueError, match=r"min_samples_leaf must be an integer from -2\*\*63 to 2\*\*63 - 1"):
        TargetEncoder(target="clicked", min_samples_leaf=2**63)
    with pytest.raises(TypeError, match="smoothing must be a number"):
        TargetEncoder(target="clicked", smoothing=text)
    with pytest.raises(ValueError, match="smoothing must be a finite number > 0"):
        TargetEncoder(target="clicked", smoothing=0)
    with pytest.raises(TypeError, match="m must be a number"):
        MEstimateEncoder(target="clicked", m=text)
    with pytest.raises(ValueError, match="m must be a finite number >= 0"):
        MEstimateEncoder(target="clicked", m=-1.0)
    with pytest.raises(TypeError, match="regularization must be a number"):
        WOEEncoder(target="clicked", regularization=text)
    with pytest.raises(ValueError, match="regularization must be a finite number > 0"):
        WOEEncoder(target="clicked", regularization=0.0)


BIG: int = 2**62

# Each category's integer targets. An f64 holds every second integer above 2^53 and every 1,024th above 2^62, so a
# sum of these values as floats loses what an exact integer sum keeps. The large values cancel across the column, so
# the prior is small and every category's own sum shows in its value.
INTEGER_TARGETS: dict[str | None, list[int]] = {
    "cancel": [BIG + 1, -BIG, BIG + 1, -BIG, 3],  # sums to 5; to 3 once each value is rounded to an f64
    "round": [BIG + 511] * 9,  # each value rounds down by 511, and their exact sum rounds up by 3,593
    "top": [2**63 - 1, 2**63 - 1, BIG + 3],  # sums past 2^64
    "bottom": [-(2**63)] * 7 + [5],  # sums below -2^65, cancelling the large sums above
    "single": [7],
    None: [2**53 + 1, -(2**53), 2**53 + 1, -(2**53)],  # sums to 2; to 0 as floats
}

# Categories whose values are all one integer that no f64 holds: each value rounds up to an f64 and each category's
# sum rounds down. Each category's variance is 0, which JamesSteinEncoder finds only when it sums the values and their
# squares exactly.
CONSTANT_TARGETS: dict[str | None, list[int]] = {f"level{step}": [BIG + step * 2**20 + 700] * 3 for step in range(4)}

# Categories near 2^53. 2^53 + 1 is the one integer above 2^53 whose f64 is 2^53, and squared as that f64 it loses
# 2^54 + 1: the edge category's value under JamesSteinEncoder's independent model then moves by 4,953,902.
EDGE_TARGETS: dict[str | None, list[int]] = {
    "edge": [2**53 + 1, 2**53 + 1, 2**53 + 2**28],
    "up": [2**53 + 2**28] * 3,
    "down": [2**53 - 2**28] * 3,
    "mid": [2**53 + 2] * 3,
}

# Categories near -(2^62 + 2^11), whose square is 2^124 + 2^74 + 2^22. The kernel cuts an integer sum or square at bits
# 22 and 75 into pieces that an f64 holds, every sum here is negative, and 5 of the 12 squares have bit 74 set. The
# category means lie 2^40 to 2^43 apart and each category spans 2^40 to 2^42, so dropping bit 74 from each square
# moves every value of JamesSteinEncoder's independent and pooled models.
CUT = -(BIG + 2**11)
CUT_TARGETS: dict[str | None, list[int]] = {
    "a": [CUT, CUT - 3 * 2**39, CUT + 5 * 2**39 + 1],
    "b": [CUT - 2**42, CUT - 2**42 + 2**40 + 11, CUT - 2**42 - 3 * 2**38],
    "c": [CUT + 2**41 + 7, CUT + 2**41 - 2**40, CUT + 2**41 + 2**39 + 3],
    "d": [CUT - 3 * 2**41 + 5, CUT - 3 * 2**41 - 2**40, CUT - 3 * 2**41 + 2**39 + 9],
}

# WOEEncoder needs a target of two values, one at most 0 and one at least 1. With these two a category of as many
# rows of each sums to its count of the larger one, and to 0 once each value is rounded to an f64.
WOE_TARGETS: dict[str | None, list[int]] = {
    "two": [BIG + 1, -BIG] * 2,
    "one": [BIG + 1, -BIG],
    "three": [BIG + 1, -BIG] * 3,
    "high": [BIG + 1],
    "low": [-BIG],
    None: [BIG + 1, -BIG],
}

# The column and the target of every fixture below, as an encoder's arguments.
ENCODED: dict[str, Any] = {"columns": ["c"], "target": "y"}


def _target_table(targets: Mapping[str | None, Sequence[float]], kind: pa.DataType) -> pa.Table:
    """Build the (category, target) rows of a fixture, with the categories interleaved.

    Args:
        targets: Each category's target values; the key None is the null category.
        kind: The Arrow type of the target column.

    Returns:
        A table of the string column ``c`` and the target column ``y``.

    """
    rows: list[tuple[str | None, float]] = [(key, value) for key, values in targets.items() for value in values]
    rows = rows[::3] + rows[1::3] + rows[2::3]
    return pa.table(
        {"c": pa.array([key for key, _ in rows], pa.string()), "y": pa.array([value for _, value in rows], kind)}
    )


def _fitted_by_formula(  # noqa: C901
    encoder: Transformer, targets: Mapping[str | None, Sequence[float]]
) -> tuple[float, dict[str | None, float]]:
    """Compute what an encoder fits, from exact sums and the formula its docstring gives.

    Every sum is an exact fraction, rounded to a float once, where the kernel rounds it. The arithmetic on the
    rounded sums is the kernel's, in the kernel's order.

    Args:
        encoder: The encoder, which is read for its class and its parameters only.
        targets: Each category's target values; the key None is the null category.

    Returns:
        The prior, and each category's value.

    """
    counts: dict[str | None, int] = {key: len(values) for key, values in targets.items()}
    exact: dict[str | None, Fraction] = {
        key: sum((Fraction(v) for v in values), Fraction(0)) for key, values in targets.items()
    }
    squares: dict[str | None, Fraction] = {
        key: sum((Fraction(v) * Fraction(v) for v in values), Fraction(0)) for key, values in targets.items()
    }
    sums: dict[str | None, float] = {key: float(value) for key, value in exact.items()}
    rows: int = sum(counts.values())
    total: float = float(sum(exact.values(), Fraction(0)))
    prior: float = total / rows
    id_like: bool = len(targets) == rows

    def variance(n: int, s: Fraction, q: Fraction) -> float:
        return 0.0 if n < 2 else max(0.0, float(n * q - s * s) / (float(n) * (n - 1)))

    def value(key: str | None) -> float:  # noqa: C901
        n: float = float(counts[key])
        s: float = sums[key]
        if isinstance(encoder, TargetEncoder):
            weight: float = 1.0 / (1.0 + math.exp(-((n - float(encoder.min_samples_leaf)) / encoder.smoothing)))
            return prior * (1 - weight) + (s / n) * weight
        if isinstance(encoder, MEstimateEncoder):
            return prior if id_like else (s + prior * encoder.m) / (n + encoder.m)
        if isinstance(encoder, WOEEncoder):
            r: float = encoder.regularization
            return 0.0 if n == 1 else math.log(((s + r) / (total + 2 * r)) / (((n - s) + r) / ((rows - total) + 2 * r)))
        if isinstance(encoder, LeaveOneOutEncoder):
            return s / n if n > 1 else prior
        if isinstance(encoder, CatBoostEncoder):
            return (s + prior * encoder.a) / (n + encoder.a) if n > 1 else prior
        assert isinstance(encoder, JamesSteinEncoder)
        if id_like:
            return prior
        mean: float = s / n
        categories: float = float(len(targets))
        target_variance: float = variance(rows, sum(exact.values(), Fraction(0)), sum(squares.values(), Fraction(0)))
        if encoder.model == "independent":
            weight = 1.0
            if categories >= 2:
                own: float = variance(counts[key], exact[key], squares[key])
                both: float = target_variance + own
                shrinkage: float = (0.0 if both == 0.0 else own / both) * (categories - 3) / (categories - 1)
                weight = min(max(1.0 - shrinkage, 0.0), 1.0)
            return weight * mean + (1 - weight) * prior
        if encoder.model == "pooled":
            errors: list[float] = [sums[other] / counts[other] - prior for other in targets]
            sse: float = math.fsum(error * error for error in errors)
            pooled: float = 0.0
            if sse > 0:
                pooled = min(max((categories - 3) * (target_variance / (rows / categories)) / sse, 0.0), 1.0)
            return prior + (1 - pooled) * (mean - prior)
        weight = n / (n + float(rows))
        return weight * mean + (1 - weight) * prior

    return prior, {key: value(key) for key in targets}


@pytest.mark.parametrize(
    ("shell", "options", "targets"),
    [
        pytest.param(TargetEncoder, {}, INTEGER_TARGETS, id="target"),
        pytest.param(TargetEncoder, {"min_samples_leaf": 3, "smoothing": 0.5}, INTEGER_TARGETS, id="target-leaf-3"),
        pytest.param(MEstimateEncoder, {}, INTEGER_TARGETS, id="m-estimate"),
        pytest.param(MEstimateEncoder, {"m": 0.0}, INTEGER_TARGETS, id="m-estimate-m-0"),
        pytest.param(WOEEncoder, {}, WOE_TARGETS, id="woe"),
        pytest.param(WOEEncoder, {"regularization": 0.25}, WOE_TARGETS, id="woe-regularization-0.25"),
        pytest.param(JamesSteinEncoder, {}, INTEGER_TARGETS, id="james-stein-independent"),
        pytest.param(JamesSteinEncoder, {"model": "pooled"}, INTEGER_TARGETS, id="james-stein-pooled"),
        pytest.param(JamesSteinEncoder, {"model": "beta"}, INTEGER_TARGETS, id="james-stein-beta"),
        pytest.param(JamesSteinEncoder, {}, CONSTANT_TARGETS, id="james-stein-independent-constant"),
        pytest.param(JamesSteinEncoder, {}, EDGE_TARGETS, id="james-stein-independent-edge"),
        pytest.param(JamesSteinEncoder, {"model": "pooled"}, CONSTANT_TARGETS, id="james-stein-pooled-constant"),
        pytest.param(JamesSteinEncoder, {"model": "pooled"}, EDGE_TARGETS, id="james-stein-pooled-edge"),
        pytest.param(JamesSteinEncoder, {}, CUT_TARGETS, id="james-stein-independent-cut"),
        pytest.param(JamesSteinEncoder, {"model": "pooled"}, CUT_TARGETS, id="james-stein-pooled-cut"),
        pytest.param(LeaveOneOutEncoder, {}, INTEGER_TARGETS, id="leave-one-out"),
        pytest.param(CatBoostEncoder, {}, INTEGER_TARGETS, id="cat-boost"),
        pytest.param(CatBoostEncoder, {"a": 0.0}, INTEGER_TARGETS, id="cat-boost-a-0"),
    ],
)
def test_an_integer_target_above_2_to_the_53_fits_by_the_formula(
    tmp_path: Path, shell: type[Transformer], options: dict[str, Any], targets: dict[str | None, list[int]]
) -> None:
    """Every encoder sums an integer target whole, so its prior and values are the formula's on the exact sums.

    Args:
        tmp_path: pytest temporary directory.
        shell: The encoder class.
        options: The encoder's own parameters.
        targets: Each category's integer targets.

    """
    path: str = write_parquet(tmp_path / "integers.parquet", _target_table(targets, pa.int64()), row_group_size=4)
    for threads in (1, 4):
        encoder: Any = shell(**ENCODED, **options).fit([path], threads=threads)
        prior, values = _fitted_by_formula(encoder, targets)
        state: dict[str, Any] = encoder._state()
        assert state["prior"] == prior
        fitted: dict[str | None, float] = dict(state["values"]["c"]) | {None: state["null_values"]["c"]}
        if None not in targets:
            assert fitted.pop(None) is None
        assert fitted == values


@pytest.mark.parametrize(
    ("shell", "power", "holder"),
    [
        (TargetEncoder, 900, "a sum"),
        (MEstimateEncoder, 900, "a sum"),
        (WOEEncoder, 900, "a sum"),
        (JamesSteinEncoder, 450, "the squared sums"),
        (LeaveOneOutEncoder, 900, "a sum"),
        (CatBoostEncoder, 900, "a sum"),
    ],
)
def test_a_float_target_fits_below_its_magnitude_bound_and_is_refused_from_it(
    tmp_path: Path, shell: type[Transformer], power: int, holder: str
) -> None:
    """A float target fits up to the largest f64 below the encoder's bound, and the bound itself is refused.

    Every encoder takes a target below 2^900 in magnitude, except JamesSteinEncoder, which sums the target's squares
    and takes one below 2^450.

    Args:
        tmp_path: pytest temporary directory.
        shell: The encoder class.
        power: The bound is 2 to this power.
        holder: What the message says a value from the bound on is more than.

    """
    bound: float = 2.0**power
    inside: float = math.nextafter(bound, 0.0)

    def fit(targets: Mapping[str | None, Sequence[float]]) -> Any:
        path: str = write_parquet(tmp_path / "floats.parquet", _target_table(targets, pa.float64()), row_group_size=2)
        return shell(**ENCODED).fit([path], threads=1)

    # WOEEncoder needs a target of two values, one at most 0 and one at least 1.
    fitting: list[dict[str | None, list[float]]] = [{"a": [inside, 0.0], "b": [inside, 0.0]}]
    if shell is not WOEEncoder:
        # The exact sum of every target is 1.5, though two of them are as large as the encoder takes.
        fitting = [{"a": [value, 1.0], "b": [-value, 0.5]} for value in (inside, -inside)]
    for targets in fitting:
        encoder: Any = fit(targets)
        prior, values = _fitted_by_formula(encoder, targets)
        state: dict[str, Any] = encoder._state()
        assert state["prior"] == prior
        assert state["values"]["c"] == values

    for value in (bound, -bound, math.inf if power == 900 else 1e200):
        with pytest.raises(EngineError) as refused:
            fit({"a": [value, 0.0], "b": [1.0, 0.0]})
        problem: str = f"is 2^{power} or more in magnitude, more than {holder} can hold (scale the target first)"
        if math.isinf(value):
            problem = "is not finite; a target cannot be NaN or infinity"
        assert str(refused.value) == f"{shell.__name__.lower()}: target column 'y': value {value!r} {problem}"
    if power == 900:
        # Nothing is refused between 2^450, where JamesSteinEncoder stops, and 2^900.
        for value in (2.0**450, 1e200):
            assert fit({"a": [value, 0.0], "b": [value, 0.0]})._state()["prior"] == value / 2
