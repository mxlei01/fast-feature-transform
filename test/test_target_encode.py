"""Tests for the target-mean kernels: the three formulas, the target column, and the artifact."""

import json
import math
import pickle
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fast_feature_transform import (
    EngineError,
    MEstimateEncoder,
    Pipeline,
    Select,
    TargetEncoder,
    Transformer,
    WOEEncoder,
    _core,
)

PRIOR: float = 0.625


def _write(path: Path, table: pa.Table, **kwargs: Any) -> str:
    """Write a Parquet file and return its path as a string.

    Args:
        path: Destination file.
        table: Table to write.
        **kwargs: Passed through to pyarrow's writer.

    Returns:
        The file path.

    """
    pq.write_table(table, path, **kwargs)
    return str(path)


def _rows(parts: list[Path]) -> list[dict[str, Any]]:
    """Read output parts back as records.

    Args:
        parts: Part files in order.

    Returns:
        The rows as dicts.

    """
    return pa.concat_tables([pq.read_table(part) for part in parts]).to_pylist()


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
    return [_write(tmp_path / "clicks.parquet", table, row_group_size=2)]


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

    rows: list[dict[str, Any]] = _rows(encoder.transform_parquet(clicks, output_dir=tmp_path / "out"))
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

    ids: str = _write(
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
        path: str = _write(
            tmp_path / "woe.parquet",
            pa.table({"city": ["ams", "ber", "cph", "dub"] * 2, "clicked": target}),
        )
        with pytest.raises(EngineError, match=message):
            WOEEncoder(target="clicked", columns=["city"]).fit([path])


def test_a_null_category_is_learned_and_a_null_target_teaches_nothing(tmp_path: Path) -> None:
    path: str = _write(
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
    clean: str = _write(tmp_path / "clean.parquet", pa.table({"city": ["ams", "ber"], "clicked": [1.0, 0.0]}))
    picky: MEstimateEncoder = MEstimateEncoder(target="clicked", columns=["city"], handle_missing="error").fit([clean])
    with pytest.raises(EngineError, match="has a null and handle_missing is 'error'"):
        picky.transform_dict({"city": None})


def test_unfitted_columns_default_to_every_utf8_column_but_never_the_target(tmp_path: Path) -> None:
    path: str = _write(
        tmp_path / "wide.parquet",
        pa.table({"city": ["ams", "ber"], "device": ["ios", "web"], "clicked": [1, 0]}),
    )
    encoder: TargetEncoder = TargetEncoder(target="clicked").fit([path])
    assert sorted(encoder.values_) == ["city", "device"]
    numbers: str = _write(tmp_path / "numbers.parquet", pa.table({"n": [1, 2], "clicked": [1, 0]}))
    with pytest.raises(EngineError, match="no utf8 columns to encode"):
        TargetEncoder(target="clicked").fit([numbers])


def test_the_target_is_read_at_fit_only_and_can_be_dropped(clicks: list[str], tmp_path: Path) -> None:
    pipeline: Pipeline = Pipeline(
        [("encode", TargetEncoder(target="clicked", columns=["city"])), ("keep", Select(columns=["city"]))]
    ).fit(clicks)
    assert pipeline.transform_dict({"city": "ams", "clicked": 1}) == {"city": pytest.approx(0.6314360527118139)}
    # A serving file without the label still transforms.
    serving: str = _write(tmp_path / "serving.parquet", pa.table({"city": ["ams", "ber"]}))
    rows: list[dict[str, Any]] = _rows(pipeline.transform_parquet([serving], output_dir=tmp_path / "out"))
    assert sorted(rows[0]) == ["city"]


def test_target_errors_name_the_target_column(clicks: list[str], tmp_path: Path) -> None:
    with pytest.raises(EngineError, match="target column 'missing' not found in input"):
        TargetEncoder(target="missing", columns=["city"]).fit(clicks)
    strings: str = _write(tmp_path / "strings.parquet", pa.table({"city": ["ams"], "clicked": ["yes"]}))
    with pytest.raises(EngineError, match="is utf8, expected an integer or float column"):
        TargetEncoder(target="clicked", columns=["city"]).fit([strings])
    numeric: str = _write(tmp_path / "numeric.parquet", pa.table({"seats": [4], "clicked": [1]}))
    with pytest.raises(EngineError, match="is i64, expected utf8"):
        TargetEncoder(target="clicked", columns=["seats"]).fit([numeric])
    empty: str = _write(
        tmp_path / "empty.parquet", pa.table({"city": ["ams"], "clicked": pa.array([None], pa.int64())})
    )
    with pytest.raises(EngineError, match="has no non-null values to learn from"):
        TargetEncoder(target="clicked", columns=["city"]).fit([empty])
    nan: str = _write(tmp_path / "nan.parquet", pa.table({"city": ["ams"], "clicked": [float("nan")]}))
    with pytest.raises(EngineError, match="is not finite"):
        TargetEncoder(target="clicked", columns=["city"]).fit([nan])


def test_the_fitted_table_is_independent_of_the_split_and_survives_the_artifact(
    clicks: list[str], tmp_path: Path
) -> None:
    floats: str = _write(
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
