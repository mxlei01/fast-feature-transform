"""Tests for the quantile_encode and summary_encode kernels: smoothing, target rules, and the artifact."""

import json
import pickle
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fast_feature_transform import EngineError, Pipeline, QuantileEncoder, Select, SummaryEncoder, _core
from helpers import read_rows, write_parquet

PRIOR: float = 3.5  # the median of prices 1..6


@pytest.fixture
def events(tmp_path: Path) -> list[str]:
    """One file with three row groups, holding repeats, a null, and a float target.

    Args:
        tmp_path: pytest-provided temporary directory.

    Returns:
        Paths of the written Parquet files.

    """
    table: pa.Table = pa.table(
        {"device": ["ios", "web", "ios", None, "tv", "ios"], "price": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}
    )
    return [write_parquet(tmp_path / "events.parquet", table, row_group_size=2)]


def test_quantile_encode_smooths_each_category_toward_the_prior(events: list[str], tmp_path: Path) -> None:
    encoder: QuantileEncoder = QuantileEncoder(target="price", columns=["device"]).fit(events)
    assert encoder.prior_ == pytest.approx(PRIOR)
    assert encoder.categories_ == {"device": ["ios", "tv", "web"]}
    # ios holds 1, 3, 6: median 3 over 3 values, smoothed with m=1 -> (3*3 + 3.5)/4.
    assert encoder.encodings_["device"] == pytest.approx([3.125, 4.25, 2.75])

    table: pa.Table = pa.concat_tables(
        [pq.read_table(part) for part in encoder.transform_parquet(events, output_dir=tmp_path / "out")]
    )
    assert table.schema.field("device").type == pa.float64()
    assert table.column("device").to_pylist() == pytest.approx([3.125, 2.75, 3.125, 3.75, 4.25, 3.125])
    assert encoder.transform_dict({"device": "ios", "price": 1.0}) == {"device": 3.125, "price": 1.0}
    assert encoder.transform_dict({"device": None}) == {"device": 3.75}  # the null was a category at fit
    assert encoder.transform_dict({"device": "nobody"}) == {"device": PRIOR}

    # m = 0 is the raw quantile, a large m is the prior.
    raw: QuantileEncoder = QuantileEncoder(target="price", columns=["device"], m=0).fit(events)
    assert raw.encodings_["device"] == pytest.approx([3.0, 5.0, 2.0])
    shrunk: QuantileEncoder = QuantileEncoder(target="price", columns=["device"], m=1e6).fit(events)
    assert shrunk.encodings_["device"] == pytest.approx([PRIOR, PRIOR, PRIOR], abs=1e-4)


def test_quantile_encode_interpolates_like_numpy(tmp_path: Path) -> None:
    # Four values, so the quartiles fall between two of them: h = q*(n-1).
    path: str = write_parquet(tmp_path / "even.parquet", pa.table({"c": ["a"] * 4, "y": [1.0, 2.0, 3.0, 4.0]}))
    for quantile, expected in ((0.0, 1.0), (0.25, 1.75), (0.5, 2.5), (0.9, 3.7), (1.0, 4.0)):
        encoder: QuantileEncoder = QuantileEncoder(target="y", columns=["c"], quantile=quantile, m=0).fit([path])
        assert encoder.encodings_["c"] == pytest.approx([expected])
        assert encoder.prior_ == pytest.approx(expected)


def test_summary_encode_replaces_each_column_with_one_per_quantile(events: list[str], tmp_path: Path) -> None:
    encoder: SummaryEncoder = SummaryEncoder(target="price", columns=["device"]).fit(events)
    assert encoder.priors_ == pytest.approx([2.25, 4.75])
    assert sorted(encoder.encodings_) == ["device_25", "device_75"]
    # ios holds 1, 3, 6, whose 25% is 2; tv and web hold one value each.
    assert encoder.encodings_["device_25"] == pytest.approx([2.0625, 3.625, 2.125])

    rows: list[dict[str, Any]] = read_rows(encoder.transform_parquet(events, output_dir=tmp_path / "out"))
    assert list(rows[0]) == ["device_25", "device_75", "price"]  # the generated columns take device's place
    assert encoder.transform_dict({"device": "ios", "price": 1.0}) == rows[0]
    assert encoder.transform_dict({"device": "nobody"}) == {"device_25": 2.25, "device_75": 4.75}

    three: SummaryEncoder = SummaryEncoder(target="price", columns=["device"], quantiles=[0.1, 0.5, 0.9], m=2.0).fit(
        events
    )
    assert sorted(three.encodings_) == ["device_10", "device_50", "device_90"]
    assert list(three.transform_dict({"device": "web"})) == ["device_10", "device_50", "device_90"]


def test_quantile_encode_null_and_unknown_options(events: list[str], tmp_path: Path) -> None:
    keep_null: QuantileEncoder = QuantileEncoder(target="price", columns=["device"], handle_missing="return_nan").fit(
        events
    )
    assert keep_null.transform_dict({"device": None}) == {"device": None}
    assert keep_null.categories_ == {"device": ["ios", "tv", "web"]}

    as_null: QuantileEncoder = QuantileEncoder(target="price", columns=["device"], handle_unknown="return_nan").fit(
        events
    )
    assert as_null.transform_dict({"device": "nobody"}) == {"device": None}
    assert as_null.transform_dict({"device": None}) == {"device": 3.75}  # nulls answer to handle_missing alone
    assert read_rows(as_null.transform_parquet(events, output_dir=tmp_path / "out"))[0]["device"] == pytest.approx(
        3.125
    )

    strict: QuantileEncoder = QuantileEncoder(target="price", columns=["device"], handle_unknown="error").fit(events)
    with pytest.raises(EngineError, match="category that was not seen at fit"):
        strict.transform_dict({"device": "nobody"})

    plain: str = write_parquet(tmp_path / "plain.parquet", pa.table({"device": ["ios", "web"], "price": [1.0, 2.0]}))
    fussy: QuantileEncoder = QuantileEncoder(target="price", columns=["device"], handle_missing="error").fit([plain])
    with pytest.raises(EngineError, match="has a null and handle_missing is 'error'"):
        fussy.transform_dict({"device": None})
    with pytest.raises(EngineError, match="has nulls and handle_missing is 'error'"):
        QuantileEncoder(target="price", columns=["device"], handle_missing="error").fit(events)


# REGRESSION: a mis-joined label column — every row null after a left join
# that missed — used to fit, save, load and serve a constant 0.0 for every
# category, with nothing anywhere saying so. target_encode always had this
# guard; the quantile and shrinkage families did not.
def test_quantile_encode_refuses_a_target_with_no_non_null_values(tmp_path: Path) -> None:
    path: str = write_parquet(
        tmp_path / "unlabelled.parquet",
        pa.table({"device": ["ios", "web"], "price": pa.array([None, None], pa.float64())}),
    )
    for cls in (QuantileEncoder, SummaryEncoder):
        with pytest.raises(EngineError, match="target column 'price' has no non-null values to learn from"):
            cls(target="price", columns=["device"]).fit([path])


def test_quantile_encode_target_rules(events: list[str], tmp_path: Path) -> None:
    with pytest.raises(EngineError, match="target column 'nope' not found in input"):
        QuantileEncoder(target="nope", columns=["device"]).fit(events)
    text: str = write_parquet(tmp_path / "text.parquet", pa.table({"device": ["ios"], "other": ["a"], "price": [1.0]}))
    with pytest.raises(EngineError, match="target column 'other' is utf8, expected an integer or float column"):
        QuantileEncoder(target="other", columns=["device"]).fit([text])
    with pytest.raises(ValueError, match="cannot encode its target"):
        QuantileEncoder(target="price", columns=["price"])

    # A row whose target is null contributes to nothing, so its category is unseen.
    holes: str = write_parquet(tmp_path / "holes.parquet", pa.table({"device": ["ios", "web"], "price": [2.0, None]}))
    sparse: QuantileEncoder = QuantileEncoder(target="price", columns=["device"]).fit([holes])
    assert sparse.categories_ == {"device": ["ios"]}
    assert sparse.transform_dict({"device": "web"}) == {"device": 2.0}

    # Once fitted the target is no longer read, so a serving file without it transforms.
    encoder: QuantileEncoder = QuantileEncoder(target="price", columns=["device"]).fit(events)
    serving: str = write_parquet(tmp_path / "serving.parquet", pa.table({"device": ["ios", "tv"]}))
    assert [row["device"] for row in read_rows(encoder.transform_parquet([serving], output_dir=tmp_path / "out"))] == (
        pytest.approx([3.125, 4.25])
    )
    assert encoder.transform_dict({"device": "ios"}) == {"device": 3.125}


def test_quantile_encode_is_independent_of_the_split_and_survives_the_artifact(
    events: list[str], tmp_path: Path
) -> None:
    for shell in (QuantileEncoder, SummaryEncoder):
        states: set[str] = set()
        for threads in (1, 4):
            encoder = shell(target="price", columns=["device"]).fit(events, threads=threads)
            states.add(encoder._require_handle().stage_state(shell.__name__.lower()))
        assert len(states) == 1

    pipeline: Pipeline = Pipeline(
        [("encode", SummaryEncoder(target="price", columns=["device"])), ("drop", Select(columns=["device_25"]))]
    ).fit(events)
    pipeline.save(tmp_path / "artifact")
    manifest: dict[str, Any] = json.loads((tmp_path / "artifact" / "manifest.json").read_text())
    state: dict[str, Any] = manifest["ops"][0]["state"]
    assert state["categories"] == {"device": ["ios", "tv", "web"]}
    assert state["nulls"]["device_25"] == pytest.approx(3.125)
    record: dict[str, Any] = {"device": "ios", "price": 1.0}
    assert Pipeline.load(tmp_path / "artifact").transform_dict(record) == pipeline.transform_dict(record)
    assert pickle.loads(pickle.dumps(pipeline)).transform_dict(record) == pipeline.transform_dict(record)
    assert set(pipeline.transform_dict(record)) == {"device_25"}  # a trailing Select drops the target

    manifest["ops"][0]["state"]["encodings"]["device_25"][0] = "nope"
    with pytest.raises(EngineError, match="must hold finite numbers"):
        _core.FittedPipeline.load_json(json.dumps(manifest))


def test_quantile_encode_validates_its_arguments(events: list[str], tmp_path: Path) -> None:
    text: Any = "0.5"
    missing: Any = None
    with pytest.raises(TypeError, match="target must be a column name"):
        QuantileEncoder(target=missing, columns=["device"])
    for shell in (QuantileEncoder, SummaryEncoder):  # REGRESSION: both used to refuse this only at fit
        with pytest.raises(ValueError, match="column 'price' is also the target"):
            shell(target="price", columns=["device", "price"])
    with pytest.raises(TypeError, match="quantile must be a number"):
        QuantileEncoder(target="price", quantile=text)
    with pytest.raises(ValueError, match="quantile must be a number between 0 and 1"):
        QuantileEncoder(target="price", quantile=1.5)
    with pytest.raises(TypeError, match="m must be a number"):
        QuantileEncoder(target="price", m=text)
    with pytest.raises(ValueError, match="m must be a finite number >= 0"):
        QuantileEncoder(target="price", m=-1)
    with pytest.raises(ValueError, match="handle_unknown must be"):
        QuantileEncoder(target="price", handle_unknown="sometimes")
    with pytest.raises(ValueError, match="handle_missing must be"):
        QuantileEncoder(target="price", handle_missing="sometimes")
    with pytest.raises(TypeError, match="quantiles must be a non-empty list"):
        SummaryEncoder(target="price", quantiles=[])
    with pytest.raises(ValueError, match="round to the same percentile"):
        SummaryEncoder(target="price", quantiles=[0.12, 0.125])
    numbers: str = write_parquet(tmp_path / "numbers.parquet", pa.table({"price": [1.0, 2.0]}))
    with pytest.raises(EngineError, match="no utf8 columns to encode"):
        QuantileEncoder(target="price").fit([numbers])
    # REGRESSION: a target spanning past f64 is refused at fit. Its median used
    # to overflow into a prior that saved as null, so the pipeline would not
    # load.
    wide: str = write_parquet(
        tmp_path / "wide.parquet",
        pa.table({"device": pa.array([None, None], pa.string()), "price": [-1.7e308, 1.7e308]}),
    )
    with pytest.raises(EngineError, match="target column 'price' spans past the range of f64"):
        QuantileEncoder(target="price", columns=["device"], handle_missing="return_nan").fit([wide])
    clash: str = write_parquet(
        tmp_path / "clash.parquet", pa.table({"device": ["ios"], "device_25": [1.0], "price": [1.0]})
    )
    with pytest.raises(EngineError, match="output column 'device_25' already exists in input"):
        SummaryEncoder(target="price", columns=["device"]).fit([clash])
