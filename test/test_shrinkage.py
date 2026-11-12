"""Tests for the shrinkage target encoders: the models, the target binding, nulls and unknowns, and the artifact."""

import json
import pickle
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest

from fast_feature_transform import (
    CatBoostEncoder,
    EngineError,
    JamesSteinEncoder,
    LeaveOneOutEncoder,
    Pipeline,
    Select,
    _core,
)
from helpers import read_rows, write_parquet

# device = ios web ios · tv ios ·, clicks = 1 4 3 6 5 2 7:
# ios sums to 6 over 3 rows, web and tv have one row each, the null category
# sums to 13 over 2 rows, and the prior is 28 / 7 = 4.
PRIOR: float = 4.0
IOS_MEAN: float = 2.0
NULL_MEAN: float = 6.5
GLOBAL_VAR: float = 28 / 6  # var(clicks), ddof=1


@pytest.fixture
def events(tmp_path: Path) -> list[str]:
    """One file with four row groups, holding repeats, singletons, and a null category.

    Args:
        tmp_path: pytest-provided temporary directory.

    Returns:
        Paths of the written Parquet files.

    """
    table: pa.Table = pa.table(
        {
            "device": ["ios", "web", "ios", None, "tv", "ios", None],
            "clicks": [1.0, 4.0, 3.0, 6.0, 5.0, 2.0, 7.0],
        }
    )
    return [write_parquet(tmp_path / "events.parquet", table, row_group_size=2)]


def test_leave_one_out_takes_the_category_mean_and_collapses_a_singleton(events: list[str], tmp_path: Path) -> None:
    encoder: LeaveOneOutEncoder = LeaveOneOutEncoder(columns=["device"], target="clicks").fit(events)
    assert encoder.prior_ == pytest.approx(PRIOR)
    # web and tv have one row each, so they collapse to the prior rather than
    # leak that row's own target.
    assert encoder.values_ == {"device": {"ios": IOS_MEAN, "tv": PRIOR, "web": PRIOR}}
    rows: list[dict[str, Any]] = read_rows(encoder.transform_parquet(events, output_dir=tmp_path / "out"))
    assert rows[0] == {"device": pytest.approx(IOS_MEAN), "clicks": 1.0}  # the target passes through
    assert rows[3]["device"] == pytest.approx(NULL_MEAN)
    assert encoder.transform_dict({"device": "ios"}) == {"device": pytest.approx(IOS_MEAN)}
    assert encoder.transform_dict({"device": "nobody"}) == {"device": pytest.approx(PRIOR)}


def test_cat_boost_smooths_toward_the_prior(events: list[str]) -> None:
    encoder: CatBoostEncoder = CatBoostEncoder(columns=["device"], target="clicks").fit(events)
    assert encoder.values_["device"]["ios"] == pytest.approx((6 + PRIOR) / (3 + 1))
    assert encoder.values_["device"]["web"] == pytest.approx(PRIOR)  # a singleton takes the raw prior
    assert encoder.transform_dict({"device": None}) == {"device": pytest.approx((13 + PRIOR) / (2 + 1))}

    smoothed: CatBoostEncoder = CatBoostEncoder(columns=["device"], target="clicks", a=2.5).fit(events)
    assert smoothed.values_["device"]["ios"] == pytest.approx((6 + PRIOR * 2.5) / (3 + 2.5))
    none: CatBoostEncoder = CatBoostEncoder(columns=["device"], target="clicks", a=0).fit(events)
    assert none.values_["device"]["ios"] == pytest.approx(IOS_MEAN)


def test_james_stein_shrinks_by_the_model(events: list[str]) -> None:
    independent: JamesSteinEncoder = JamesSteinEncoder(columns=["device"], target="clicks").fit(events)
    # ios: within-category variance 1, four categories, so B = 1/(gvar + 1) * (4 - 3)/(4 - 1).
    weight: float = 1 - (1 / (GLOBAL_VAR + 1)) / 3
    assert independent.values_["device"]["ios"] == pytest.approx(weight * IOS_MEAN + (1 - weight) * PRIOR)
    # web has one row, so no variance and no shrinkage: it keeps its own mean,
    # 4, which is also the prior.
    assert independent.values_["device"]["web"] == pytest.approx(PRIOR)

    beta: JamesSteinEncoder = JamesSteinEncoder(columns=["device"], target="clicks", model="beta").fit(events)
    assert beta.values_["device"]["ios"] == pytest.approx(3 / 10 * IOS_MEAN + 7 / 10 * PRIOR)

    pooled: JamesSteinEncoder = JamesSteinEncoder(columns=["device"], target="clicks", model="pooled").fit(events)
    errors: float = sum((mean - PRIOR) ** 2 for mean in (IOS_MEAN, PRIOR, 5.0, NULL_MEAN))  # ios, web, tv, null
    shrinkage: float = (4 - 3) * (GLOBAL_VAR / (7 / 4)) / errors
    assert pooled.values_["device"]["ios"] == pytest.approx(PRIOR + (1 - shrinkage) * (IOS_MEAN - PRIOR))

    # As many categories as rows means the column identifies the row.
    identifiers: pa.Table = pa.table({"device": ["a", "b", "c", "d"], "clicks": [1.0, 2.0, 3.0, 6.0]})
    path: str = write_parquet(Path(events[0]).parent / "ids.parquet", identifiers)
    id_like: JamesSteinEncoder = JamesSteinEncoder(columns=["device"], target="clicks").fit([path])
    assert id_like.values_["device"] == {"a": 3.0, "b": 3.0, "c": 3.0, "d": 3.0}


def test_shrinkage_null_and_unknown_options(events: list[str], tmp_path: Path) -> None:
    keep_null: LeaveOneOutEncoder = LeaveOneOutEncoder(
        columns=["device"], target="clicks", handle_missing="return_nan"
    ).fit(events)
    assert keep_null.transform_dict({"device": None}) == {"device": None}
    # The null category still shaped the statistics it was left out of.
    assert keep_null.prior_ == pytest.approx(PRIOR)

    as_null: LeaveOneOutEncoder = LeaveOneOutEncoder(
        columns=["device"], target="clicks", handle_unknown="return_nan"
    ).fit(events)
    assert as_null.transform_dict({"device": "nobody"}) == {"device": None}
    assert read_rows(as_null.transform_parquet(events, output_dir=tmp_path / "out"))[0]["device"] == pytest.approx(
        IOS_MEAN
    )

    strict: CatBoostEncoder = CatBoostEncoder(columns=["device"], target="clicks", handle_unknown="error").fit(events)
    with pytest.raises(EngineError, match="category that was not seen at fit"):
        strict.transform_dict({"device": "nobody"})

    with pytest.raises(EngineError, match="has nulls and handle_missing is 'error'"):
        JamesSteinEncoder(columns=["device"], target="clicks", handle_missing="error").fit(events)

    # A row whose target is null is not there at all, so its category is unknown.
    partial: pa.Table = pa.table({"device": ["ios", "ios", "tv"], "clicks": [1.0, 3.0, None]})
    path: str = write_parquet(tmp_path / "partial.parquet", partial)
    sparse: LeaveOneOutEncoder = LeaveOneOutEncoder(columns=["device"], target="clicks").fit([path])
    assert sparse.values_ == {"device": {"ios": 2.0}}
    assert sparse.prior_ == pytest.approx(2.0)


# REGRESSION: a mis-joined label column — every row null after a left join
# that missed — used to fit, save, load and serve a constant 0.0 for every
# category, with nothing anywhere saying so. Under handle_unknown="error" it
# was stranger still: the encoder reported a category it definitely saw as
# unseen, pointing the operator at the wrong thing. target_encode always had
# this guard; the shrinkage and quantile families did not.
def test_shrinkage_refuses_a_target_with_no_non_null_values(tmp_path: Path) -> None:
    path: str = write_parquet(
        tmp_path / "unlabelled.parquet",
        pa.table({"device": ["ios", "web"], "clicks": pa.array([None, None], pa.float64())}),
    )
    for cls in (JamesSteinEncoder, LeaveOneOutEncoder, CatBoostEncoder):
        with pytest.raises(EngineError, match="target column 'clicks' has no non-null values to learn from"):
            cls(columns=["device"], target="clicks").fit([path])


# REGRESSION: adding the same amount to every target moves every value by that
# amount. The variances used to come from sums rounded before the subtraction,
# which cancel when the target sits far from zero: at 1e9 the independent and
# pooled models gave the raw category means.
@pytest.mark.parametrize("model", ["independent", "pooled"])
@pytest.mark.parametrize("step", [0.0, 0.1])
def test_james_stein_values_move_with_the_target(tmp_path: Path, model: str, step: float) -> None:
    # A step adds a fraction to each target, which leaves the sums with several partials, so the exact variance has
    # cross terms to form.
    color: list[str] = ["amber", "amber", "blue", "blue", "blue", "cyan", "delta", "delta"]
    clicks: list[float] = [1.0, 3.0, 2.0, 8.0, 4.0, 5.0, 0.0, 6.0]
    fitted: dict[float, dict[str, float]] = {}
    for offset in (0.0, 1e9):
        values: list[float] = [value + offset + step * (row + 1) for row, value in enumerate(clicks)]
        path: str = write_parquet(tmp_path / f"offset-{offset:g}.parquet", pa.table({"color": color, "clicks": values}))
        fitted[offset] = JamesSteinEncoder(columns=["color"], target="clicks", model=model).fit([path]).values_["color"]
    raw_amber: float = (clicks[0] + step + clicks[1] + 2 * step) / 2
    assert fitted[0.0]["amber"] != pytest.approx(raw_amber)  # shrunk toward the prior, not the raw mean
    for key, value in fitted[0.0].items():
        assert fitted[1e9][key] - 1e9 == pytest.approx(value, rel=0, abs=1e-6)


def test_shrinkage_binds_the_target_and_lets_serving_drop_it(events: list[str], tmp_path: Path) -> None:
    # columns=None encodes every utf8 column; the target is numeric, so it is
    # never one of them, and a trailing Select drops it from the output.
    pipeline: Pipeline = Pipeline(
        [
            ("encode", CatBoostEncoder(target="clicks")),
            ("drop", Select(columns=["device"])),
        ]
    ).fit(events)
    assert pipeline.transform_dict({"device": "ios", "clicks": 1.0}) == {"device": pytest.approx(2.5)}

    # A fitted encoder reads only its columns, so a file without the label transforms.
    unlabelled: str = write_parquet(tmp_path / "serve.parquet", pa.table({"device": ["ios", "web"]}))
    encoder: LeaveOneOutEncoder = LeaveOneOutEncoder(columns=["device"], target="clicks").fit(events)
    served: list[dict[str, Any]] = read_rows(encoder.transform_parquet([unlabelled], output_dir=tmp_path / "served"))
    assert served == [{"device": pytest.approx(IOS_MEAN)}, {"device": pytest.approx(PRIOR)}]


def test_shrinkage_is_independent_of_the_split_and_survives_the_artifact(events: list[str], tmp_path: Path) -> None:
    states: set[str] = set()
    for threads in (1, 4):
        encoder: JamesSteinEncoder = JamesSteinEncoder(columns=["device"], target="clicks").fit(events, threads=threads)
        states.add(encoder._require_handle().stage_state("jamessteinencoder"))
    assert len(states) == 1

    pipeline: Pipeline = Pipeline([("shrink", CatBoostEncoder(columns=["device"], target="clicks"))]).fit(events)
    pipeline.save(tmp_path / "artifact")
    manifest: dict[str, Any] = json.loads((tmp_path / "artifact" / "manifest.json").read_text())
    state: dict[str, Any] = manifest["ops"][0]["state"]
    assert state["prior"] == pytest.approx(PRIOR)
    assert state["values"]["device"]["ios"] == pytest.approx(2.5)
    assert state["null_values"]["device"] == pytest.approx((13 + PRIOR) / 3)
    record: dict[str, Any] = {"device": "ios", "clicks": 1.0}
    assert Pipeline.load(tmp_path / "artifact").transform_dict(record) == pipeline.transform_dict(record)
    assert pickle.loads(pickle.dumps(pipeline)).transform_dict(record) == pipeline.transform_dict(record)

    # The model is a parameter, so it has to survive the rebuild on load too.
    shrunk: Pipeline = Pipeline(
        [("shrink", JamesSteinEncoder(columns=["device"], target="clicks", model="pooled"))]
    ).fit(events)
    shrunk.save(tmp_path / "pooled")
    assert Pipeline.load(tmp_path / "pooled").transform_dict(record) == shrunk.transform_dict(record)

    manifest["ops"][0]["state"]["values"]["device"]["ios"] = "two"
    with pytest.raises(EngineError, match="must be a finite number"):
        _core.FittedPipeline.load_json(json.dumps(manifest))


def test_shrinkage_refuses_a_target_it_cannot_read(events: list[str], tmp_path: Path) -> None:
    with pytest.raises(EngineError, match="target column 'missing' not found in input"):
        LeaveOneOutEncoder(columns=["device"], target="missing").fit(events)
    labels: str = write_parquet(tmp_path / "labels.parquet", pa.table({"device": ["ios"], "label": ["yes"]}))
    with pytest.raises(EngineError, match="target column 'label' is utf8, expected an integer or float column"):
        LeaveOneOutEncoder(columns=["device"], target="label").fit([labels])

    numbers: str = write_parquet(tmp_path / "numbers.parquet", pa.table({"n": [1, 2], "clicks": [1.0, 2.0]}))
    with pytest.raises(EngineError, match="no utf8 columns to encode"):
        JamesSteinEncoder(target="clicks").fit([numbers])


def test_shrinkage_validates_its_arguments() -> None:
    bad_option: Any = "sometimes"
    number: Any = 3
    text: Any = "1"
    bare: Any = "device"
    with pytest.raises(TypeError, match="target must be a column name"):
        LeaveOneOutEncoder(columns=["device"], target=number)
    # REGRESSION: the target among the columns is refused here, as the
    # target-mean encoders refuse it; these three used to refuse it at fit.
    for shell in (JamesSteinEncoder, LeaveOneOutEncoder, CatBoostEncoder):
        with pytest.raises(ValueError, match="column 'y' is also the target"):
            shell(columns=["device", "y"], target="y")
    with pytest.raises(ValueError, match="handle_unknown must be"):
        CatBoostEncoder(target="y", handle_unknown=bad_option)
    with pytest.raises(ValueError, match="handle_missing must be"):
        CatBoostEncoder(target="y", handle_missing=bad_option)
    with pytest.raises(TypeError, match="columns must be a list of column names"):
        JamesSteinEncoder(columns=bare, target="y")
    with pytest.raises(ValueError, match="model must be one of"):
        JamesSteinEncoder(target="y", model=bad_option)
    with pytest.raises(ValueError, match="not portable"):
        JamesSteinEncoder(target="y", model="binary")
    with pytest.raises(TypeError, match="a must be a number"):
        CatBoostEncoder(target="y", a=text)
    with pytest.raises(ValueError, match="a must be a finite number >= 0"):
        CatBoostEncoder(target="y", a=-1)
    assert CatBoostEncoder(target="y").params() == {
        "columns": None,
        "target": "y",
        "a": 1.0,
        "handle_unknown": "value",
        "handle_missing": "value",
    }


@pytest.mark.parametrize(
    ("targets", "devices"),
    [
        (
            [
                "0x1.0759c9adb04ddp-515",
                "0x1.4b8e24878adebp-514",
                "-0x1.64223bf27ff24p-514",
                "0x1.45f8a613e17f4p-513",
                "0x1.3f589a1be05f3p-511",
                "0x1.a492359d2f346p-514",
                "0x1.fb94454ba0f16p-513",
                "0x1.8d30c99cac762p-512",
                "0x1.f2aef4affa37cp-516",
                "-0x1.2519373039992p-512",
                "-0x1.380021070d57fp-514",
                "0x1.a3822ae3cedd7p-512",
            ],
            ["k0", "k4", "k0", "k1", "k1", "k2", "k0", "k0", "k4", "k3", "k4", "k0"],
        ),
        (
            [
                "0x1.2982d5355ea91p-517",
                "-0x1.68a85a28a9d15p-515",
                "0x1.47e7a4707229bp-513",
                "0x1.6133e9e68afc8p-515",
                "0x1.4024db6f22feep-516",
                "-0x1.7954e5f6aa07ap-514",
                "-0x1.d1deb4bb08c46p-517",
                "-0x1.209f7b7872f71p-513",
                "0x1.d414906000bf9p-516",
                "0x1.013937f0c2050p-514",
                "-0x1.ded07c84fb602p-515",
                "-0x1.31c8e5aa803a6p-515",
            ],
            ["k0", "k2", "k3", "k4", "k4", "k0", "k2", "k1", "k3", "k0", "k3", "k3"],
        ),
    ],
    ids=["partials split by the merge", "terms split by the merge"],
)
def test_james_stein_fits_targets_near_1e_155_alike_for_every_split_and_thread_count(
    tmp_path: Path, targets: list[str], devices: list[str]
) -> None:
    """Targets whose squares are subnormal fit one table, whatever the row groups and the thread count.

    REGRESSION: a category's variance reads the terms of its exact sums, which
    depend on the sums' values alone. It used to read the partials, which two
    workers can leave split differently than one, and near 1e-155 the products
    of two partials underflow and round: with the first targets the value of
    category k0 differed in its last bit between one thread and two or four.
    The second targets do the same to a reader of the doubles the sums hold now.

    Args:
        tmp_path: pytest temporary directory.
        targets: The target column, as hexadecimal floats.
        devices: The category of each row.

    """
    table: pa.Table = pa.table({"device": devices, "clicks": [float.fromhex(text) for text in targets]})
    states: set[str] = set()
    for rows_per_group in (2, 4, 6, 12):
        path: str = write_parquet(tmp_path / f"groups{rows_per_group}.parquet", table, row_group_size=rows_per_group)
        for threads in (1, 2, 4):
            for _ in range(3):  # which worker takes which row group varies from run to run
                encoder: JamesSteinEncoder = JamesSteinEncoder(columns=["device"], target="clicks")
                encoder.fit([path], threads=threads)
                states.add(encoder._require_handle().stage_state("jamessteinencoder"))
    assert len(states) == 1  # byte-identical, not merely close
