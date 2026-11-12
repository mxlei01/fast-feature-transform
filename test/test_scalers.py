"""Tests for the scikit-learn scalers: the statistics, both transform paths, the artifact, and every refusal."""

import json
import math
import pickle
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fast_feature_transform import (
    Binarizer,
    Cast,
    EngineError,
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    Pipeline,
    StandardScaler,
    Transformer,
    _core,
)
from helpers import read_rows, write_parquet

ROWS: int = 8


def _records(table: pa.Table) -> list[dict[str, Any]]:
    """The source table as records, for driving the record path.

    Args:
        table: The input table.

    Returns:
        One dict per row.

    """
    return table.to_pylist()


@pytest.fixture
def prices(tmp_path: Path) -> tuple[str, pa.Table]:
    """One file with four row groups: a plain column, a column with a null, and a constant one.

    Args:
        tmp_path: pytest-provided temporary directory.

    Returns:
        The file path and the table that was written.

    """
    table: pa.Table = pa.table(
        {
            "price": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "gap": [-4.0, None, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0],
            "flat": [3.0] * ROWS,
        }
    )
    return write_parquet(tmp_path / "prices.parquet", table, row_group_size=2), table


def _agrees(transformer: Transformer, table: pa.Table, parts: list[Path]) -> None:
    """Assert the record path answers exactly what the batch path did.

    Args:
        transformer: The fitted transformer.
        table: The source table the batch path read.
        parts: The part files it wrote.

    """
    for record, row in zip(_records(table), read_rows(parts), strict=True):
        assert transformer.transform_dict(record) == row


def test_standard_scaler_fits_population_statistics_and_passes_nulls_through(
    prices: tuple[str, pa.Table], tmp_path: Path
) -> None:
    path, table = prices
    scaler: StandardScaler = StandardScaler(columns=["price", "gap", "flat"]).fit([path])
    assert scaler.mean_["price"] == 4.5
    assert scaler.var_["price"] == 5.25  # ddof=0, not the sample variance
    assert scaler.scale_["price"] == math.sqrt(5.25)
    assert scaler.n_samples_seen_ == {"price": ROWS, "gap": ROWS - 1, "flat": ROWS}
    # A constant column scales by 1 rather than by 0, so it comes out zeros.
    assert scaler.var_["flat"] == 0.0
    assert scaler.scale_["flat"] == 1.0

    parts: list[Path] = scaler.transform_parquet([path], output_dir=tmp_path / "out")
    rows: list[dict[str, Any]] = read_rows(parts)
    assert [row["flat"] for row in rows] == [0.0] * ROWS
    assert rows[0]["price"] == (1.0 - 4.5) / math.sqrt(5.25)
    assert rows[1]["gap"] is None  # a null teaches nothing and comes out a null
    _agrees(scaler, table, parts)


def test_standard_scaler_flags_leave_out_the_centring_or_the_division(prices: tuple[str, pa.Table]) -> None:
    path, _ = prices
    centred: StandardScaler = StandardScaler(columns=["price"], with_std=False).fit([path])
    assert centred.transform_dict({"price": 5.0}) == {"price": 0.5}
    divided: StandardScaler = StandardScaler(columns=["price"], with_mean=False).fit([path])
    assert divided.transform_dict({"price": 5.0}) == {"price": 5.0 / math.sqrt(5.25)}
    neither: StandardScaler = StandardScaler(columns=["price"], with_mean=False, with_std=False).fit([path])
    assert neither.transform_dict({"price": 5.0}) == {"price": 5.0}
    # Both statistics are recorded whichever flags are set.
    assert neither.mean_["price"] == 4.5


def test_min_max_scaler_maps_the_fitted_range_and_clips_on_request(
    prices: tuple[str, pa.Table], tmp_path: Path
) -> None:
    path, table = prices
    scaler: MinMaxScaler = MinMaxScaler(columns=["price", "gap", "flat"], feature_range=(-1.0, 1.0)).fit([path])
    assert scaler.data_min_ == {"price": 1.0, "gap": -4.0, "flat": 3.0}
    assert scaler.data_max_ == {"price": 8.0, "gap": 10.0, "flat": 3.0}
    assert scaler.data_range_ == {"price": 7.0, "gap": 14.0, "flat": 0.0}
    assert scaler.scale_["gap"] == 2.0 / 14.0
    assert scaler.min_["gap"] == -1.0 - (-4.0) * (2.0 / 14.0)
    assert scaler.n_samples_seen_ == ROWS  # rows, nulls included

    parts: list[Path] = scaler.transform_parquet([path], output_dir=tmp_path / "out")
    rows: list[dict[str, Any]] = read_rows(parts)
    # x * scale + min lands on the ends of the range up to one rounding, which
    # is exactly where scikit-learn's own two operations land.
    assert rows[0]["price"] == pytest.approx(-1.0)
    assert rows[-1]["price"] == pytest.approx(1.0)
    assert [row["flat"] for row in rows] == [-1.0] * ROWS  # a constant column lands on the low end
    _agrees(scaler, table, parts)

    # Only a value outside the fitted range can leave feature_range.
    assert scaler.transform_dict({"price": 15.0, "gap": 0.0, "flat": 3.0})["price"] == pytest.approx(3.0)
    clipped: MinMaxScaler = MinMaxScaler(columns=["price"], feature_range=(-1.0, 1.0), clip=True).fit([path])
    assert clipped.transform_dict({"price": 15.0}) == {"price": 1.0}
    assert clipped.transform_dict({"price": -15.0}) == {"price": -1.0}


def test_max_abs_scaler_divides_by_the_largest_magnitude(prices: tuple[str, pa.Table], tmp_path: Path) -> None:
    path, table = prices
    scaler: MaxAbsScaler = MaxAbsScaler(columns=["price", "gap", "flat"]).fit([path])
    assert scaler.max_abs_ == {"price": 8.0, "gap": 10.0, "flat": 3.0}
    assert scaler.scale_ == scaler.max_abs_
    assert scaler.n_samples_seen_ == ROWS

    parts: list[Path] = scaler.transform_parquet([path], output_dir=tmp_path / "out")
    rows: list[dict[str, Any]] = read_rows(parts)
    assert rows[0]["price"] == 0.125
    assert rows[0]["gap"] == -0.4
    _agrees(scaler, table, parts)

    clipped: MaxAbsScaler = MaxAbsScaler(columns=["price"], clip=True).fit([path])
    assert clipped.transform_dict({"price": 80.0}) == {"price": 1.0}
    assert clipped.transform_dict({"price": -80.0}) == {"price": -1.0}


def test_min_max_scaler_refuses_a_scale_of_zero(tmp_path: Path) -> None:
    """A fit whose scale would round to 0 is refused.

    REGRESSION: a column spanning past the range of f64 used to fit a scale of
    0, mapping every value to the low end, and its saved pipeline then failed
    to load.
    """
    wide: str = write_parquet(tmp_path / "wide.parquet", pa.table({"a": [-1e308, 0.0, 1e308]}))
    with pytest.raises(EngineError, match=r"^minmaxscaler: column 'a' spans past the range of f64$"):
        MinMaxScaler(columns=["a"]).fit([wide])
    tall: str = write_parquet(tmp_path / "tall.parquet", pa.table({"a": [0.0, 1e308]}))
    narrow: MinMaxScaler = MinMaxScaler(columns=["a"], feature_range=(1.0, 1.0 + 2.0**-52))
    with pytest.raises(EngineError, match=r"fits a scale that rounds to 0; widen 'feature_range'$"):
        narrow.fit([tall])


def test_standard_scaler_fits_a_tiny_constant_column_that_saves_and_loads(tmp_path: Path) -> None:
    """A constant column of tiny values fits variance 0, as scikit-learn does, and its saved pipeline loads.

    REGRESSION: below about 1e-146 the exact sums round their residuals, and
    1e-158 used to fit a variance of -5e-324, which a saved pipeline refuses.
    """
    path: str = write_parquet(tmp_path / "tiny.parquet", pa.table({"x": [1e-158] * 4}))
    pipeline: Pipeline = Pipeline([("scale", StandardScaler(columns=["x"]))]).fit([path])
    scaler = pipeline["scale"]
    assert isinstance(scaler, StandardScaler)
    assert scaler.var_ == {"x": 0.0}
    pipeline.save(tmp_path / "saved")
    assert Pipeline.load(tmp_path / "saved").transform_dict({"x": 1e-158}) == {"x": 0.0}


def test_max_abs_scaler_leaves_an_all_zero_column_alone(tmp_path: Path) -> None:
    path: str = write_parquet(tmp_path / "zeros.parquet", pa.table({"a": [0.0, 0.0, 0.0]}))
    scaler: MaxAbsScaler = MaxAbsScaler(columns=["a"]).fit([path])
    assert scaler.max_abs_["a"] == 0.0
    assert scaler.scale_["a"] == 1.0  # the near-constant rule, so 0 / 1 rather than 0 / 0
    assert scaler.transform_dict({"a": 0.0}) == {"a": 0.0}


def test_scalers_take_every_numeric_column_when_columns_is_none(tmp_path: Path) -> None:
    path: str = write_parquet(tmp_path / "mixed.parquet", pa.table({"a": [1.0, 3.0], "n": [2, 6], "c": ["x", "y"]}))
    scaler: MinMaxScaler = MinMaxScaler().fit([path])
    assert set(scaler.data_min_) == {"a", "n"}  # the utf8 column is not a numeric one
    assert scaler.transform_dict({"a": 1.0, "n": 6, "c": "x"}) == {"a": 0.0, "n": 1.0, "c": "x"}
    # An integer column is read as f64 and comes out f64, as scikit-learn's cast does.
    parts: list[Path] = scaler.transform_parquet([path], output_dir=tmp_path / "out")
    assert pq.read_table(parts[0]).schema.field("n").type == pa.float64()


def test_binarizer_is_strictly_above_the_threshold_and_emits_i64(tmp_path: Path) -> None:
    table: pa.Table = pa.table({"score": [-1.0, -0.5, 0.0, 0.5, 1.0], "keep": [1.0] * 5})
    path: str = write_parquet(tmp_path / "scores.parquet", table)
    binarizer: Binarizer = Binarizer(columns=["score"], threshold=0.5).fit([path])
    parts: list[Path] = binarizer.transform_parquet([path], output_dir=tmp_path / "out")
    assert pq.read_table(parts[0]).schema.field("score").type == pa.int64()
    assert [row["score"] for row in read_rows(parts)] == [0, 0, 0, 0, 1]  # exactly on the threshold is a 0
    _agrees(binarizer, table, parts)
    assert Binarizer(columns=["score"]).fit([path]).transform_dict({"score": 0.0, "keep": 1.0}) == {
        "score": 0,
        "keep": 1.0,
    }


def test_normalizer_divides_each_row_by_its_own_norm(tmp_path: Path) -> None:
    table: pa.Table = pa.table({"x": [3.0, 0.0, 1.0], "y": [4.0, 0.0, -1.0], "z": [9.0, 9.0, 9.0]})
    path: str = write_parquet(tmp_path / "vectors.parquet", table, row_group_size=2)
    l2: Normalizer = Normalizer(columns=["x", "y"]).fit([path])
    parts: list[Path] = l2.transform_parquet([path], output_dir=tmp_path / "out")
    rows: list[dict[str, Any]] = read_rows(parts)
    assert rows[0] == {"x": 0.6, "y": 0.8, "z": 9.0}  # z is not listed, so it is not touched
    assert rows[1] == {"x": 0.0, "y": 0.0, "z": 9.0}  # a zero row is left alone, not divided
    assert rows[2]["x"] == pytest.approx(1 / math.sqrt(2))
    _agrees(l2, table, parts)

    assert Normalizer(columns=["x", "y"], norm="l1").fit([path]).transform_dict({"x": 3.0, "y": -1.0})["x"] == 0.75
    assert Normalizer(columns=["x", "y"], norm="max").fit([path]).transform_dict({"x": 2.0, "y": -4.0})["y"] == -1.0
    # The norm is over the listed vector, so it does not depend on the order
    # the columns were listed in.
    forward: Normalizer = Normalizer(columns=["x", "y", "z"]).fit([path])
    backward: Normalizer = Normalizer(columns=["z", "y", "x"]).fit([path])
    record: dict[str, Any] = {"x": 1.0, "y": 2.0, "z": 3.0}
    assert forward.transform_dict(record) == backward.transform_dict(record)


@pytest.mark.parametrize(
    ("factory", "stage"),
    [
        (lambda: StandardScaler(columns=["price", "gap"]), "standardscaler"),
        (lambda: MinMaxScaler(columns=["price", "gap"]), "minmaxscaler"),
        (lambda: MaxAbsScaler(columns=["price", "gap"]), "maxabsscaler"),
    ],
)
def test_the_fitted_table_never_depends_on_the_thread_count(
    factory: Any, stage: str, prices: tuple[str, pa.Table], tmp_path: Path
) -> None:
    path, _ = prices
    states: set[str] = set()
    outputs: set[str] = set()
    for threads in (1, 4):
        scaler: Transformer = factory().fit([path], threads=threads)
        states.add(scaler._require_handle().stage_state(stage))
        outputs.add(json.dumps(read_rows(scaler.transform_parquet([path], output_dir=tmp_path / f"out{threads}"))))
    assert len(states) == 1  # byte-identical, not merely close
    assert len(outputs) == 1


def test_scalers_survive_the_artifact_and_the_pickle(prices: tuple[str, pa.Table], tmp_path: Path) -> None:
    path, _ = prices
    pipeline: Pipeline = Pipeline(
        [
            ("standard", StandardScaler(columns=["price"])),
            ("range", MinMaxScaler(columns=["gap"], feature_range=(-1.0, 1.0), clip=True)),
            ("magnitude", MaxAbsScaler(columns=["flat"])),
            ("flag", Binarizer(columns=["price"], threshold=0.0)),
            ("unit", Normalizer(columns=["gap", "flat"], norm="l1")),
        ]
    ).fit([path])
    record: dict[str, Any] = {"price": 4.0, "gap": 2.0, "flat": 3.0}
    expected: dict[str, Any] = pipeline.transform_dict(record)
    pipeline.save(tmp_path / "artifact")
    assert Pipeline.load(tmp_path / "artifact").transform_dict(record) == expected
    assert pickle.loads(pickle.dumps(pipeline)).transform_dict(record) == expected

    manifest: dict[str, Any] = json.loads((tmp_path / "artifact" / "manifest.json").read_text())
    ops: list[dict[str, Any]] = manifest["ops"]
    assert [op["kernel"] for op in ops] == [
        "standard_scale",
        "min_max_scale",
        "max_abs_scale",
        "binarize",
        "normalize",
    ]
    assert ops[1]["params"] == {"columns": ["gap"], "feature_range": [-1.0, 1.0], "clip": True}
    assert ops[3]["state"] == {}  # binarize learns nothing
    assert ops[4]["params"] == {"columns": ["gap", "flat"], "norm": "l1"}

    # A hand-edited manifest is refused rather than silently defaulted.
    broken: dict[str, Any] = json.loads(json.dumps(manifest))
    broken["ops"][0]["state"]["scales"]["price"] = 0.0
    with pytest.raises(EngineError, match=r"scales\['price'\] must be a positive number"):
        _core.FittedPipeline.load_json(json.dumps(broken))
    for scale in (0.0, -1.0):
        # min_max_scale fits (high - low) / handle_zeros(range), always strictly
        # positive: a zero collapses the column onto its offset and a negative
        # one reverses it, so the loader refuses both.
        broken = json.loads(json.dumps(manifest))
        broken["ops"][1]["state"]["scales"]["gap"] = scale
        with pytest.raises(EngineError, match=r"scales\['gap'\] must be a positive number"):
            _core.FittedPipeline.load_json(json.dumps(broken))
    broken = json.loads(json.dumps(manifest))
    broken["ops"][2]["state"]["max_abs"] = {"other": 1.0}
    with pytest.raises(EngineError, match=r"max_abs\['flat'\] must be a finite number"):
        _core.FittedPipeline.load_json(json.dumps(broken))
    broken = json.loads(json.dumps(manifest))
    del broken["ops"][2]["state"]["max_abs"]["flat"]
    with pytest.raises(EngineError, match="state 'columns' does not name the same columns"):
        _core.FittedPipeline.load_json(json.dumps(broken))
    broken = json.loads(json.dumps(manifest))
    broken["ops"][3]["state"] = {"scales": {"price": 1.0}}
    with pytest.raises(EngineError, match="binarize learns nothing"):
        _core.FittedPipeline.load_json(json.dumps(broken))


def test_scalers_refuse_columns_they_cannot_read(prices: tuple[str, pa.Table], tmp_path: Path) -> None:
    path, _ = prices
    text: str = write_parquet(tmp_path / "text.parquet", pa.table({"c": ["x", "y"]}))
    with pytest.raises(EngineError, match="column 'c' is utf8, expected an integer or float column"):
        StandardScaler(columns=["c"]).fit([text])
    with pytest.raises(EngineError, match="no integer or float columns to scale"):
        MaxAbsScaler().fit([text])
    with pytest.raises(EngineError, match="column 'nobody' not found in input"):
        MinMaxScaler(columns=["nobody"]).fit([path])
    # Cast makes the column readable; no other step parses strings as numbers.
    pipeline: Pipeline = Pipeline([("cast", Cast({"c": "f64"})), ("scale", MinMaxScaler(columns=["c"]))])
    numbers: str = write_parquet(tmp_path / "numbers.parquet", pa.table({"c": ["1", "3"]}))
    assert pipeline.fit([numbers]).transform_dict({"c": "3"}) == {"c": 1.0}

    empty: str = write_parquet(tmp_path / "empty.parquet", pa.table({"a": pa.array([None, None], pa.float64())}))
    with pytest.raises(EngineError, match="column 'a' has no non-null values to learn from"):
        StandardScaler(columns=["a"]).fit([empty])


def test_scalers_refuse_a_nan_or_an_infinity_in_the_data(tmp_path: Path) -> None:
    # A missing number has one spelling here — a null — so the other one is an
    # error rather than a second, silently different, kind of missing.
    nans: str = write_parquet(tmp_path / "nan.parquet", pa.table({"a": [1.0, float("nan")]}))
    with pytest.raises(EngineError, match="is not finite; write a missing value as a null"):
        StandardScaler(columns=["a"]).fit([nans])
    infinities: str = write_parquet(tmp_path / "inf.parquet", pa.table({"a": [1.0, float("inf")]}))
    with pytest.raises(EngineError, match="is not finite; write a missing value as a null"):
        MinMaxScaler(columns=["a"]).fit([infinities])
    good: str = write_parquet(tmp_path / "good.parquet", pa.table({"a": [1.0, 2.0]}))
    scaler: MaxAbsScaler = MaxAbsScaler(columns=["a"]).fit([good])
    with pytest.raises(EngineError, match="is not finite"):
        scaler.transform_dict({"a": float("nan")})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")], ids=["nan", "inf", "-inf"])
def test_scalers_refuse_a_nan_or_an_infinity_at_transform_through_both_paths(tmp_path: Path, value: float) -> None:
    # The fit-time refusal above is a different call site from this one: a
    # transform reaches finite_number from inside the batch visitor and from
    # the record path, and a scaler that refused only what it was fitted on
    # would let a NaN through on the data that actually matters.
    good: str = write_parquet(tmp_path / "good.parquet", pa.table({"a": [1.0, 2.0, 3.0, 4.0]}))
    hostile: str = write_parquet(tmp_path / "hostile.parquet", pa.table({"a": [1.0, value, 2.0]}))
    message: str = "is not finite; write a missing value as a null"
    for scaler in (
        StandardScaler(columns=["a"]).fit([good]),
        MinMaxScaler(columns=["a"]).fit([good]),
        MaxAbsScaler(columns=["a"]).fit([good]),
    ):
        with pytest.raises(EngineError, match=message):
            scaler.transform_parquet([hostile], output_dir=tmp_path / "out")
        with pytest.raises(EngineError, match=message):
            scaler.transform_dict({"a": value})


def test_standard_scaler_treats_a_near_constant_column_as_constant(tmp_path: Path) -> None:
    # scikit-learn's _is_constant_feature is a relative test,
    # var <= n*eps*var + (n*mean*eps)^2, so a column whose spread is real but
    # tiny beside its magnitude counts as constant and scales by 1, even though
    # its variance is not zero. Pinned here because the obvious reading
    # (variance > 0, therefore divide by its root) is the wrong one and would
    # diverge from scikit-learn on exactly this shape of data.
    values: list[float] = [1e8, 1e8 + 1e-7, 1e8, 1e8 + 1e-7]
    path: str = write_parquet(tmp_path / "near.parquet", pa.table({"a": values}))
    scaler: StandardScaler = StandardScaler(columns=["a"]).fit([path])
    assert scaler.var_["a"] > 0.0
    assert scaler.scale_["a"] == 1.0
    assert scaler.mean_["a"] == 100000000.00000006

    # The values scikit-learn 1.9 produces for this column, exactly.
    expected: list[float] = [-5.960464477539063e-08, 4.470348358154297e-08] * 2
    parts: list[Path] = scaler.transform_parquet([path], output_dir=tmp_path / "out")
    assert [row["a"] for row in read_rows(parts)] == expected
    for value, row in zip(expected, values, strict=True):
        assert scaler.transform_dict({"a": row})["a"] == value


def test_binarizer_and_normalizer_refuse_a_null(tmp_path: Path) -> None:
    path: str = write_parquet(tmp_path / "gaps.parquet", pa.table({"a": [1.0, None], "b": [2.0, 3.0]}))
    binarizer: Binarizer = Binarizer(columns=["a"]).fit([path])
    with pytest.raises(EngineError, match="column 'a' has a null, which binarize has no answer for"):
        binarizer.transform_parquet([path], output_dir=tmp_path / "out")
    with pytest.raises(EngineError, match="column 'a' has a null, which binarize has no answer for"):
        binarizer.transform_dict({"a": None})

    normalizer: Normalizer = Normalizer(columns=["a", "b"]).fit([path])
    with pytest.raises(EngineError, match="column 'a' has a null"):
        normalizer.transform_parquet([path], output_dir=tmp_path / "out2")
    with pytest.raises(EngineError, match="column 'a' has a null"):
        normalizer.transform_dict({"a": None, "b": 1.0})
    with pytest.raises(EngineError, match="key 'b' missing from record"):
        normalizer.transform_dict({"a": 1.0})


def test_scaler_arguments_are_validated_before_the_engine_runs() -> None:
    bare_string: Any = "price"
    text: Any = "0.5"
    not_a_pair: Any = (0.0, 1.0, 2.0)
    with pytest.raises(TypeError, match="not a single string"):
        StandardScaler(columns=bare_string)
    with pytest.raises(ValueError, match="at least one column"):
        MinMaxScaler(columns=[])
    with pytest.raises(TypeError, match="with_mean must be True or False"):
        StandardScaler(with_mean=text)
    with pytest.raises(TypeError, match="with_std must be True or False"):
        StandardScaler(with_std=text)
    with pytest.raises(TypeError, match=r"feature_range must be a \(low, high\) pair"):
        MinMaxScaler(feature_range=not_a_pair)
    with pytest.raises(TypeError, match="feature_range must be a number"):
        MinMaxScaler(feature_range=(text, 1.0))
    with pytest.raises(ValueError, match="feature_range must be a finite number"):
        MinMaxScaler(feature_range=(0.0, float("inf")))
    with pytest.raises(ValueError, match="feature_range must be .low, high. with low < high"):
        MinMaxScaler(feature_range=(1.0, 1.0))
    with pytest.raises(TypeError, match="clip must be True or False"):
        MaxAbsScaler(clip=text)
    with pytest.raises(TypeError, match="threshold must be a number"):
        Binarizer(columns=["a"], threshold=text)
    with pytest.raises(ValueError, match="threshold must be a finite number"):
        Binarizer(columns=["a"], threshold=float("nan"))
    with pytest.raises(ValueError, match="norm must be one of"):
        Normalizer(columns=["a"], norm="l3")
    # The two stateless kernels have no columns=None spelling at all.
    with pytest.raises(TypeError, match="missing 1 required positional argument"):
        Binarizer()  # ty: ignore[missing-argument]
    with pytest.raises(TypeError, match="missing 1 required positional argument"):
        Normalizer()  # ty: ignore[missing-argument]


def test_scaler_reprs_round_trip_through_their_params() -> None:
    assert repr(StandardScaler(columns=["a"])) == "StandardScaler(columns=['a'], with_mean=True, with_std=True)"
    assert repr(MinMaxScaler()) == "MinMaxScaler(columns=None, feature_range=[0.0, 1.0], clip=False)"
    assert repr(MaxAbsScaler()) == "MaxAbsScaler(columns=None, clip=False)"
    assert repr(Binarizer(columns=["a"])) == "Binarizer(columns=['a'], threshold=0.0)"
    assert repr(Normalizer(columns=["a", "b"])) == "Normalizer(columns=['a', 'b'], norm='l2')"


def test_standard_scaler_centres_on_the_exactly_rounded_mean(tmp_path: Path) -> None:
    """The fitted mean is the exact mean rounded to f64, which is not always scikit-learn's.

    ``StandardScaler`` sums with Shewchuk partials, so ``mean_`` is the
    exact mean of the column rounded to f64. scikit-learn's
    ``_incremental_mean_and_var`` accumulates chunk by chunk and can land one
    ulp of the mean away from it. With ``with_std=False`` nothing divides that
    gap away, so it appears undivided in the output and scales with the column's
    magnitude: on the column below ours is ``1.0000000000000002e16`` against
    scikit-learn's ``1e16``, and every output value differs by 2.0.

    Ours is the more accurate answer, and this test pins it.

    Args:
        tmp_path: pytest temporary directory.

    """
    values: list[float] = [1e16, 1.0000000000000004e16, 1e16, 1e16, 1.0000000000000004e16]
    source: str = write_parquet(tmp_path / "large.parquet", pa.table({"price": values}))
    scaler: StandardScaler = StandardScaler(columns=["price"], with_mean=True, with_std=False).fit([source])
    assert scaler.mean_ == {"price": 1.0000000000000002e16}
    assert scaler.mean_["price"] == math.fsum(values) / len(values)
    rows: list[dict[str, Any]] = read_rows(scaler.transform_parquet([source], output_dir=tmp_path / "out"))
    assert [row["price"] for row in rows] == [-2.0, 2.0, -2.0, -2.0, 2.0]
    for value, row in zip(values, rows, strict=True):
        assert scaler.transform_dict({"price": value}) == row


def test_standard_scaler_fits_values_near_1e_155_alike_for_every_split_and_thread_count(tmp_path: Path) -> None:
    """Values whose squares are subnormal fit one state, whatever the row groups and the thread count.

    REGRESSION: the variance reads the terms of the exact sum, which depend on
    the sum's value alone. It used to read the partials, which two workers can
    leave split differently than one, and near 1e-155 the products with them
    underflow and round: these eight values fitted a variance one subnormal
    step larger with two or four threads than with one.

    Args:
        tmp_path: pytest temporary directory.

    """
    values: list[float] = [
        float.fromhex(text)
        for text in (
            "-0x1.f6ce40edf5c76p-515",
            "0x1.73575e9aed87fp-513",
            "0x1.e72ae1a4c72c6p-514",
            "0x1.a123fefa800d6p-513",
            "0x1.bf71b1e54a35fp-513",
            "0x1.49aafe6faea33p-512",
            "0x1.275f886390578p-514",
            "0x1.042bc2ea82348p-511",
        )
    ]
    table: pa.Table = pa.table({"x": values})
    states: set[str] = set()
    for rows_per_group in (1, 2, 4, 8):
        path: str = write_parquet(tmp_path / f"groups{rows_per_group}.parquet", table, row_group_size=rows_per_group)
        for threads in (1, 2, 4):
            for _ in range(3):  # which worker takes which row group varies from run to run
                scaler: StandardScaler = StandardScaler(columns=["x"]).fit([path], threads=threads)
                states.add(scaler._require_handle().stage_state("standardscaler"))
    assert len(states) == 1  # byte-identical, not merely close
    assert json.loads(states.pop())["variances"] == {"x": float.fromhex("0x0.19ce25531acb2p-1022")}


@pytest.mark.parametrize(
    ("norm", "row"),
    [
        # The largest f64 and three values of half its last step: the sum is
        # past the range, yet no single addition to the largest overflows.
        ("l1", [1.7976931348623157e308, 2.0**969, 2.0**969, 2.0**969]),
        ("l2", [math.nextafter(2.0**512, 0.0), 2.0**485, 2.0**485, 2.0**485]),
    ],
    ids=["l1", "l2"],
)
def test_normalizer_refuses_a_row_whose_norm_leaves_the_range_through_both_paths(
    tmp_path: Path, norm: str, row: list[float]
) -> None:
    """A row whose exact norm is past f64's range is an error, never a row of NaN.

    Args:
        tmp_path: pytest temporary directory.
        norm: The norm to divide by.
        row: One row whose norm leaves the range.

    """
    names: list[str] = ["a", "b", "c", "d"]
    table: pa.Table = pa.table({name: [1.0, value] for name, value in zip(names, row, strict=True)})
    path: str = write_parquet(tmp_path / "top.parquet", table)
    normalizer: Normalizer = Normalizer(columns=names, norm=norm).fit([path])
    message: str = rf"a row's {norm} norm leaves the range of f64$"
    with pytest.raises(EngineError, match=message):
        normalizer.transform_dict(dict(zip(names, row, strict=True)))
    with pytest.raises(EngineError, match=message):
        normalizer.transform_parquet([path], output_dir=tmp_path / "out")


def test_normalizer_divides_a_row_whose_norm_is_the_largest_f64(tmp_path: Path) -> None:
    """A row whose exact l1 norm rounds to the largest f64 is normalized, on both paths."""
    largest: float = 1.7976931348623157e308
    row: dict[str, float] = {
        "a": 2.0**1023,
        "b": 2.0**971 - 2.0**969 - 2.0**960,
        "c": 2.0**1023 - 2.0**972 + 2.0**970,
    }
    assert math.fsum(row.values()) == largest
    path: str = write_parquet(tmp_path / "top.parquet", pa.table({name: [value] for name, value in row.items()}))
    normalizer: Normalizer = Normalizer(columns=list(row), norm="l1").fit([path])
    expected: dict[str, float] = {name: value / largest for name, value in row.items()}
    assert normalizer.transform_dict(row) == expected
    assert read_rows(normalizer.transform_parquet([path], output_dir=tmp_path / "out")) == [expected]
