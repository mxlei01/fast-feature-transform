"""Tests for the invariants the engine design rests on: pass scheduling, parity, merges, concurrency, column order."""

import subprocess  # nosec B404
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import fast_feature_transform
from fast_feature_transform import (
    Binarizer,
    BinaryEncoder,
    Cast,
    CountEncoder,
    HashingEncoder,
    JamesSteinEncoder,
    KBinsDiscretizer,
    LabelEncoder,
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    OneHotEncoder,
    OrdinalEncoder,
    Pipeline,
    PolynomialFeatures,
    PowerTransformer,
    QuantileEncoder,
    QuantileTransformer,
    RobustScaler,
    Select,
    SplineTransformer,
    StandardScaler,
    SumEncoder,
    TargetEncoder,
)
from helpers import write_parquet

THREADS: int = 16
CALLS_PER_THREAD: int = 2_000


def _encoded(parts: list[Path], column: str) -> list[int]:
    """Concatenate one encoded column across output parts.

    Args:
        parts: Part files in order.
        column: Column to read.

    Returns:
        The column's values in row order.

    """
    return pa.concat_tables([pq.read_table(part) for part in parts]).column(column).to_pylist()


def test_multi_pass_fit_schedules_a_dependent_encoder_in_a_later_pass(tmp_path: Path) -> None:
    path: str = write_parquet(
        tmp_path / "t.parquet",
        pa.table({"a": ["x", "y", "x", None], "b": ["p", "q", "r", "p"], "c": ["ignored"] * 4}),
        row_group_size=2,
    )
    pipeline: Pipeline = Pipeline(
        [("e1", LabelEncoder(columns=["a"])), ("s", Select(columns=["a", "b"])), ("e2", LabelEncoder())]
    ).fit([path], threads=2)
    first, second = pipeline["e1"], pipeline["e2"]
    assert isinstance(first, LabelEncoder)
    assert isinstance(second, LabelEncoder)
    assert first.label_maps_ == {"a": {"x": 1, "y": 2}}
    assert second.label_maps_ == {"b": {"p": 1, "q": 2, "r": 3}}  # only b is still utf8 after e1
    parts: list[Path] = pipeline.transform_parquet([path], output_dir=tmp_path / "out")
    rows: list[dict[str, object]] = pa.concat_tables([pq.read_table(part) for part in parts]).to_pylist()
    source: list[dict[str, object]] = pq.read_table(path).to_pylist()
    assert rows == [pipeline.transform_dict(dict(row)) | {} for row in [{k: r[k] for k in ("a", "b")} for r in source]]


def test_unknown_and_null_categories_encode_to_zero_in_batch_mode(tmp_path: Path) -> None:
    train: str = write_parquet(tmp_path / "train.parquet", pa.table({"device": ["a", "b", "a"]}))
    serve: str = write_parquet(tmp_path / "serve.parquet", pa.table({"device": ["zzz", "a", None, "b", "qq"]}))
    encoder: LabelEncoder = LabelEncoder(columns=["device"]).fit([train])
    parts: list[Path] = encoder.transform_parquet([serve], output_dir=tmp_path / "out")
    batch: list[int] = _encoded(parts, "device")
    assert batch == [0, 1, 0, 2, 0]
    assert batch == [encoder.transform_dict({"device": value})["device"] for value in ["zzz", "a", None, "b", "qq"]]


# REGRESSION: handle_missing="error" is the strict option, so it has to mean
# the same thing in every port — and that meaning is category_encoders': a
# null is refused wherever one turns up, at transform as well as at fit. It
# used to hold only for the fit data in 13 of the 19 encoders that take the
# option, so the same pipeline let a serving null through as -2 or as the
# prior. The list is every encoder that takes handle_missing.
_HANDLE_MISSING_ENCODERS: list[str] = [
    "CountEncoder",
    "OrdinalEncoder",
    "OneHotEncoder",
    "RankHotEncoder",
    "BaseNEncoder",
    "BinaryEncoder",
    "GrayEncoder",
    "BackwardDifferenceEncoder",
    "HelmertEncoder",
    "SumEncoder",
    "PolynomialEncoder",
    "TargetEncoder",
    "MEstimateEncoder",
    "WOEEncoder",
    "JamesSteinEncoder",
    "LeaveOneOutEncoder",
    "CatBoostEncoder",
    "QuantileEncoder",
    "SummaryEncoder",
]
_SUPERVISED: frozenset[str] = frozenset(
    {
        "TargetEncoder",
        "MEstimateEncoder",
        "WOEEncoder",
        "JamesSteinEncoder",
        "LeaveOneOutEncoder",
        "CatBoostEncoder",
        "QuantileEncoder",
        "SummaryEncoder",
    }
)


@pytest.mark.parametrize("name", _HANDLE_MISSING_ENCODERS)
def test_handle_missing_error_refuses_a_null_at_transform_in_every_encoder(name: str, tmp_path: Path) -> None:
    train: str = write_parquet(
        tmp_path / "train.parquet", pa.table({"device": ["a", "a", "b", "b"], "label": [1.0, 0.0, 1.0, 0.0]})
    )
    serve: str = write_parquet(tmp_path / "serve.parquet", pa.table({"device": ["a", None], "label": [1.0, 0.0]}))
    extra: dict[str, object] = {"target": "label"} if name in _SUPERVISED else {}
    encoder = getattr(fast_feature_transform, name)(columns=["device"], handle_missing="error", **extra)
    encoder.fit([train])
    # The fit data has no nulls, so the fit-time check passes; the serving one
    # must fail, and identically through both paths.
    with pytest.raises(Exception, match="column 'device' has a null and handle_missing is 'error'"):
        encoder.transform_parquet([serve], output_dir=tmp_path / f"out-{name}")
    with pytest.raises(Exception, match="column 'device' has a null and handle_missing is 'error'"):
        encoder.transform_dict({"device": None, "label": 1.0})
    # A record without a null still encodes, so the option is a refusal and
    # not a broken encoder.
    assert encoder.transform_dict({"device": "a", "label": 1.0})


# REGRESSION: a null counts toward handle_missing="error" even on a row whose
# target is null. The target-mean and shrinkage encoders used to skip such a
# row before counting its null, so the fit accepted it and the fitted encoder
# then refused its own training file at transform. Two-row row groups and two
# threads give the count more than one worker state to merge.
@pytest.mark.parametrize("name", sorted(_SUPERVISED))
@pytest.mark.parametrize("row_group_size", [None, 2])
def test_handle_missing_error_refuses_at_fit_a_null_whose_target_is_null(
    name: str, row_group_size: int | None, tmp_path: Path
) -> None:
    train: str = write_parquet(
        tmp_path / "train.parquet",
        pa.table({"device": ["a", "a", "b", "b", None], "label": [1.0, 0.0, 1.0, 0.0, None]}),
        row_group_size=row_group_size,
    )
    encoder = getattr(fast_feature_transform, name)(columns=["device"], target="label", handle_missing="error")
    with pytest.raises(Exception, match="column 'device' has nulls and handle_missing is 'error'"):
        encoder.fit([train], threads=2)


def test_large_string_nulls_and_empty_strings(tmp_path: Path) -> None:
    values: list[str | None] = ["q", None, "", "p", "q", None]
    path: str = write_parquet(tmp_path / "large.parquet", pa.table({"c": pa.array(values, type=pa.large_string())}))
    encoder: LabelEncoder = LabelEncoder(columns=["c"]).fit([path])
    assert encoder.label_maps_["c"] == {"": 1, "p": 2, "q": 3}
    parts: list[Path] = encoder.transform_parquet([path], output_dir=tmp_path / "out")
    assert _encoded(parts, "c") == [3, 0, 1, 2, 3, 0]
    assert [encoder.transform_dict({"c": value})["c"] for value in values] == [3, 0, 1, 2, 3, 0]


def test_all_null_column_yields_an_empty_map_that_round_trips(tmp_path: Path) -> None:
    path: str = write_parquet(
        tmp_path / "nulls.parquet", pa.table({"c": pa.array([None, None, None], type=pa.string())})
    )
    encoder: LabelEncoder = LabelEncoder().fit([path])
    assert encoder.label_maps_ == {"c": {}}
    assert encoder.vocab_size("c") == 1
    parts: list[Path] = encoder.transform_parquet([path], output_dir=tmp_path / "out")
    assert _encoded(parts, "c") == [0, 0, 0]
    pipeline: Pipeline = Pipeline([("e", LabelEncoder())]).fit([path])
    pipeline.save(tmp_path / "artifact")
    assert Pipeline.load(tmp_path / "artifact").transform_dict({"c": "anything"}) == {"c": 0}


def test_single_threaded_and_parallel_fits_agree_across_worker_boundaries(tmp_path: Path) -> None:
    # 40 row groups of 50 rows; every category appears exactly twice, in different row groups,
    # so applying min_count=2 per worker, before the merge, would drop every category whose
    # two rows went to different workers.
    values: list[str] = [f"cat_{i:04d}" for i in range(1000)] * 2
    path: str = write_parquet(tmp_path / "straddle.parquet", pa.table({"c": values}), row_group_size=50)
    serial: LabelEncoder = LabelEncoder(columns=["c"], min_count=2).fit([path], threads=1)
    parallel: LabelEncoder = LabelEncoder(columns=["c"], min_count=2).fit([path], threads=8)
    assert len(serial.label_maps_["c"]) == 1000
    assert serial.label_maps_ == parallel.label_maps_


def test_transform_dict_is_safe_under_free_threaded_concurrency(tmp_path: Path) -> None:
    path: str = write_parquet(
        tmp_path / "t.parquet", pa.table({"device": ["a", "b", "c", "a"], "price": [1.0, 2.0, 3.0, 4.0]})
    )
    pipeline: Pipeline = Pipeline([("s", Select(columns=["device"])), ("e", LabelEncoder())]).fit([path])
    standalone: LabelEncoder = LabelEncoder(columns=["device"]).fit([path])
    # An expanding encoder too: its record path builds a whole new row, and it
    # has to size that row from what the fit froze rather than from `outputs`,
    # which resolve() rewrites under a mutex while this may be running.
    expanding: OneHotEncoder = OneHotEncoder(columns=["device"]).fit([path])
    records: list[dict[str, object]] = [{"device": value, "price": 1.0} for value in ("a", "b", "c", "zzz", None)]
    expected: list[dict[str, object]] = [pipeline.transform_dict(record) for record in records]
    expected_standalone: list[dict[str, object]] = [standalone.transform_dict(record) for record in records]
    expected_expanding: list[dict[str, object]] = [expanding.transform_dict(record) for record in records]

    def worker(seed: int) -> int:
        mismatches: int = 0
        for call in range(CALLS_PER_THREAD):
            index: int = (seed + call) % len(records)
            if pipeline.transform_dict(records[index]) != expected[index]:
                mismatches += 1
            if standalone.transform_dict(records[index]) != expected_standalone[index]:
                mismatches += 1
            if expanding.transform_dict(records[index]) != expected_expanding[index]:
                mismatches += 1
        return mismatches

    with ThreadPoolExecutor(max_workers=THREADS) as pool:
        assert sum(pool.map(worker, range(THREADS))) == 0


@settings(max_examples=60, deadline=None)
@given(
    values=st.lists(st.one_of(st.none(), st.text(max_size=8)), max_size=120),
    row_group_size=st.integers(min_value=1, max_value=40),
    min_count=st.integers(min_value=1, max_value=4),
)
def test_label_encoder_properties(
    tmp_path_factory: pytest.TempPathFactory, values: list[str | None], row_group_size: int, min_count: int
) -> None:
    tmp_path: Path = tmp_path_factory.mktemp("prop")
    path: str = write_parquet(
        tmp_path / "p.parquet", pa.table({"c": pa.array(values, type=pa.string())}), row_group_size=row_group_size
    )
    serial: LabelEncoder = LabelEncoder(columns=["c"], min_count=min_count).fit([path], threads=1)
    parallel: LabelEncoder = LabelEncoder(columns=["c"], min_count=min_count).fit([path], threads=8)
    label_map: dict[str, int] = serial.label_maps_["c"]
    assert label_map == parallel.label_maps_["c"]

    counts: dict[str, int] = {}
    for value in values:
        if value is not None:
            counts[value] = counts.get(value, 0) + 1
    kept: list[str] = sorted((value for value, count in counts.items() if count >= min_count), key=str.encode)
    assert label_map == {value: index + 1 for index, value in enumerate(kept)}

    parts: list[Path] = parallel.transform_parquet([path], output_dir=tmp_path / "out")
    batch: list[int] = _encoded(parts, "c")
    assert batch == [parallel.transform_dict({"c": value})["c"] for value in values]
    assert all(0 <= index <= len(kept) for index in batch)


# The expanding steps and the columns each one consumes. Every one of them
# drops the columns it consumes and appends what it generates after the
# columns that stay; SummaryEncoder is the one exception in either ported
# package and substitutes its quantile columns in the source column's position.
_EXPANDING: list[tuple[str, dict[str, object], list[str]]] = [
    ("OneHotEncoder", {"columns": ["x"]}, ["x"]),
    ("RankHotEncoder", {"columns": ["x"]}, ["x"]),
    ("BaseNEncoder", {"columns": ["x"]}, ["x"]),
    ("BinaryEncoder", {"columns": ["x"]}, ["x"]),
    ("GrayEncoder", {"columns": ["x"]}, ["x"]),
    ("BackwardDifferenceEncoder", {"columns": ["x"]}, ["x"]),
    ("HelmertEncoder", {"columns": ["x"]}, ["x"]),
    ("SumEncoder", {"columns": ["x"]}, ["x"]),
    ("PolynomialEncoder", {"columns": ["x"]}, ["x"]),
    ("HashingEncoder", {"columns": ["x"], "n_components": 2}, ["x"]),
    ("GroupAgg", {"key": "pre", "aggs": {"y": ["mean"]}}, ["y"]),
    ("PolynomialFeatures", {"columns": ["n"], "degree": 2}, ["n"]),
    ("SplineTransformer", {"columns": ["n"], "n_knots": 3, "degree": 1}, ["n"]),
    ("KBinsDiscretizer", {"columns": ["n"], "n_bins": 2, "encode": "onehot"}, ["n"]),
]


@pytest.fixture
def wide(tmp_path: Path) -> tuple[list[str], dict[str, object]]:
    """A frame whose encoded column sits between two that pass through.

    Args:
        tmp_path: pytest-provided temporary directory.

    Returns:
        The written Parquet paths, and one record from the same data.

    """
    table: pa.Table = pa.table(
        {
            "pre": ["p", "p", "q", "q"],
            "x": ["a", "b", "c", "a"],
            "post": ["s", "t", "s", "t"],
            "n": [1.0, 2.0, 3.0, 4.0],
            "y": [1.0, 2.0, 3.0, 4.0],
        }
    )
    return [write_parquet(tmp_path / "wide.parquet", table)], {"pre": "p", "x": "a", "post": "s", "n": 1.0, "y": 1.0}


# category_encoders puts an expanding encoder's generated columns in the
# source column's place — [pre, x, post] becomes [pre, x_1, x_2, post] —
# except HashingEncoder, which puts them first. We deliberately append them
# after the columns that stay, [pre, post, x_1, x_2], so the generated block
# is a function of the step's fitted state, not of the incoming column order.
# It is also load-bearing: parquet_to_memmap lays fields out in output order,
# so a memmap's dtype follows it.
@pytest.mark.parametrize(("name", "params", "consumed"), _EXPANDING, ids=[case[0] for case in _EXPANDING])
def test_every_expanding_step_appends_its_generated_columns(
    name: str, params: dict[str, object], consumed: list[str], wide: tuple[list[str], dict[str, object]], tmp_path: Path
) -> None:
    source, record = wide
    step = getattr(fast_feature_transform, name)(**params)
    step.fit(source)
    columns: list[str] = pq.read_table(step.transform_parquet(source, output_dir=tmp_path / name)[0]).column_names
    stays: list[str] = [column for column in ["pre", "x", "post", "n", "y"] if column not in consumed]
    assert columns[: len(stays)] == stays, f"{name} did not keep the untouched columns first, in input order"
    assert len(columns) > len(stays), f"{name} generated nothing"
    # The record path lays the same keys down in the same order.
    assert list(step.transform_dict(dict(record))) == columns


def test_the_appended_block_is_the_shape_the_reference_describes(
    wide: tuple[list[str], dict[str, object]], tmp_path: Path
) -> None:
    source, record = wide
    hot: OneHotEncoder = OneHotEncoder(columns=["x"]).fit(source)
    # category_encoders would give [pre, x_1, x_2, x_3, post, n, y] here.
    expected: list[str] = ["pre", "post", "n", "y", "x_1", "x_2", "x_3"]
    assert pq.read_table(hot.transform_parquet(source, output_dir=tmp_path / "hot")[0]).column_names == expected
    assert list(hot.transform_dict(dict(record))) == expected


def test_summary_encoder_substitutes_in_place(wide: tuple[list[str], dict[str, object]], tmp_path: Path) -> None:
    source, record = wide
    step = fast_feature_transform.SummaryEncoder(columns=["x"], target="y")
    step.fit(source)
    columns: list[str] = pq.read_table(step.transform_parquet(source, output_dir=tmp_path / "summary")[0]).column_names
    assert columns == ["pre", "x_25", "x_75", "post", "n", "y"]
    assert list(step.transform_dict(dict(record))) == columns


def _reordered(record: dict[str, object]) -> list[dict[str, object]]:
    """Return a record's fields with the keys reversed and in three shuffled orders.

    Args:
        record: The record.

    Returns:
        Four records holding the fields of ``record``.

    """
    items: list[tuple[str, object]] = list(record.items())
    orders: list[dict[str, object]] = [dict(reversed(items))]
    for seed in range(3):
        order: np.ndarray = np.random.default_rng(seed).permutation(len(items))
        orders.append(dict(items[index] for index in order))
    return orders


def _batch_rows(pipeline: Pipeline, path: str, output_dir: Path) -> list[dict[str, object]]:
    """Transform a file through the batch path and read the rows back.

    Args:
        pipeline: A fitted pipeline.
        path: The Parquet file to transform.
        output_dir: Directory for the output parts.

    Returns:
        The transformed rows, in the file's row order.

    """
    parts: list[Path] = pipeline.transform_parquet([path], output_dir=output_dir)
    return pa.concat_tables([pq.read_table(part) for part in parts]).to_pylist()


# A step finds the fields it reads by name, and it tries the field after the
# one it found last before it searches the record. Each place a kernel does
# that is reached by a step below. Their columns sit between other fields,
# some are listed out of the table's order, and some are read by two steps.
def test_a_record_gives_its_batch_row_whatever_order_its_keys_are_in(tmp_path: Path) -> None:
    rows: int = 48
    rng: np.random.Generator = np.random.default_rng(20)
    columns: dict[str, object] = {}
    for index in range(16):
        columns[f"n{index:02d}"] = rng.normal(size=rows)
        if index < 10:
            columns[f"t{index:02d}"] = rng.choice(["a", "b", "c", "d"], size=rows)
    columns["y"] = rng.normal(size=rows)
    table: pa.Table = pa.table(columns)
    path: str = write_parquet(tmp_path / "wide.parquet", table)
    pipeline: Pipeline = Pipeline(
        [
            ("cast", Cast({"n00": "f32", "n01": "f64"})),
            ("standard", StandardScaler(columns=["n01", "n02", "n03"])),
            ("min_max", MinMaxScaler(columns=["n05", "n04"])),
            ("max_abs", MaxAbsScaler(columns=["n06"])),
            ("robust", RobustScaler(columns=["n07", "n08"])),
            ("quantile", QuantileTransformer(columns=["n09"], n_quantiles=10)),
            ("power", PowerTransformer(columns=["n10"])),
            ("binarize", Binarizer(columns=["n11"], threshold=0.5)),
            ("normalize", Normalizer(columns=["n03", "n01", "n02"])),
            ("bins", KBinsDiscretizer(columns=["n12"], n_bins=3, encode="ordinal", strategy="uniform")),
            ("hot_bins", KBinsDiscretizer(columns=["n13"], n_bins=3, encode="onehot", strategy="uniform")),
            ("polynomial", PolynomialFeatures(columns=["n15", "n14"], degree=2)),
            ("spline", SplineTransformer(columns=["n04"], n_knots=3, degree=1)),
            ("ordinal", OrdinalEncoder(columns=["t00", "t01"])),
            ("count", CountEncoder(columns=["t02"])),
            ("target", TargetEncoder(target="y", columns=["t03"])),
            ("james_stein", JamesSteinEncoder(columns=["t04"], target="y")),
            ("quantile_encode", QuantileEncoder(target="y", columns=["t05"])),
            ("one_hot", OneHotEncoder(columns=["t06"])),
            ("binary", BinaryEncoder(columns=["t07"])),
            ("sum", SumEncoder(columns=["t08"])),
            ("hashing", HashingEncoder(columns=["t09"], n_components=4)),
        ]
    ).fit([path])
    expected: list[dict[str, object]] = _batch_rows(pipeline, path, tmp_path / "out")
    source: list[dict[str, object]] = table.to_pylist()
    for row in range(0, rows, 8):
        kept: dict[str, object] = pipeline.transform_dict(dict(source[row]))
        assert kept == expected[row]
        assert list(kept) == list(expected[row])  # in the table's key order, the batch path's column order too
        for record in _reordered(source[row]):
            assert pipeline.transform_dict(record) == expected[row]


# From 128 fields on, a step that reads 16 columns or more finds them through
# an index of the record and a step that reads fewer searches it, so a wide
# record has both kinds of step to get right.
def test_a_record_of_128_fields_or_more_gives_its_batch_row_whatever_order_its_keys_are_in(tmp_path: Path) -> None:
    rng: np.random.Generator = np.random.default_rng(21)
    names: list[str] = [f"w{index:03d}" for index in range(140)]
    table: pa.Table = pa.table({name: rng.normal(size=8) for name in names})
    path: str = write_parquet(tmp_path / "wider.parquet", table)
    pipeline: Pipeline = Pipeline(
        [
            ("indexed", StandardScaler(columns=names[100:120])),
            ("searched", MinMaxScaler(columns=names[112:104:-1] + names[130:136])),
        ]
    ).fit([path])
    expected: list[dict[str, object]] = _batch_rows(pipeline, path, tmp_path / "out")
    source: list[dict[str, object]] = table.to_pylist()
    for row in (0, 7):
        assert pipeline.transform_dict(dict(source[row])) == expected[row]
        for record in _reordered(source[row]):
            assert pipeline.transform_dict(record) == expected[row]


def test_tracked_stub_matches_the_generated_one(tmp_path: Path) -> None:
    # Regenerate with: python -m nanobind.stubgen -m fast_feature_transform._core -o fast_feature_transform/_core.pyi,
    # then ruff check --select PYI009 --fix and ruff format on it (the same normalization applied here).
    repo: Path = Path(__file__).resolve().parent.parent
    generated: Path = tmp_path / "_core.pyi"
    stubgen: list[str] = [
        sys.executable,
        "-m",
        "nanobind.stubgen",
        "-m",
        "fast_feature_transform._core",
        "-o",
        str(generated),
    ]
    ruff: list[str] = [sys.executable, "-m", "ruff"]
    config: list[str] = ["--config", str(repo / "pyproject.toml"), "--quiet"]
    subprocess.run(stubgen, check=True, capture_output=True)  # nosec B603
    subprocess.run([*ruff, "check", *config, "--select", "PYI009", "--fix", str(generated)], check=True)  # nosec B603
    subprocess.run([*ruff, "format", *config, str(generated)], check=True)  # nosec B603
    assert generated.read_text() == (repo / "fast_feature_transform" / "_core.pyi").read_text()
