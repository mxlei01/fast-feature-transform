"""The measurement rules of the two benchmark scripts and the module they share.

``benchmark/transformers.py`` needs category_encoders and pandas, which are
deliberately not dependencies of this project, so the module is loaded here
with those two stubbed out. Nothing tested below touches either of them. The
tests cover how the scripts measure: the Parquet settings they charge upstream
for, how a one-column file is read, how repeated runs are summarized, and that
the case order and the row-group layout are not fixed.

Both scripts measure through ``benchmark/measure.py``, and the
``test_both_scripts_*`` tests keep them there: the two once reported different
methodologies.
"""

import importlib.util
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fast_feature_transform import Select

BENCHMARK_DIR: Path = Path(__file__).resolve().parent.parent / "benchmark"


def _load_benchmark(script: str) -> ModuleType:
    """Import one benchmark script with its two optional dependencies stubbed.

    Args:
        script: File name of the script inside ``benchmark/``.

    Returns:
        The loaded module.

    """
    spec = importlib.util.spec_from_file_location(f"benchmark_{Path(script).stem}", BENCHMARK_DIR / script)
    assert spec is not None
    assert spec.loader is not None
    module: ModuleType = importlib.util.module_from_spec(spec)
    stubs: dict[str, ModuleType] = {name: ModuleType(name) for name in ("pandas", "category_encoders")}
    # The scripts import ``measure`` the way they do when run directly, with
    # their own directory on sys.path.
    if str(BENCHMARK_DIR) not in sys.path:
        sys.path.insert(0, str(BENCHMARK_DIR))
    # Only the two stubbed names are put back afterwards: restoring the whole
    # of sys.modules would drop everything the import pulled in, and a C
    # extension cannot be loaded twice in one process.
    saved: dict[str, ModuleType | None] = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        spec.loader.exec_module(module)
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
    return module


BENCH: ModuleType = _load_benchmark("transformers.py")
LABEL: ModuleType = _load_benchmark("label_encoder.py")
MEASURE: ModuleType = importlib.import_module("measure")


def _column_facts(path: Path) -> list[tuple[Any, ...]]:
    """Describe how every column of a Parquet file was encoded.

    Args:
        path: The Parquet file to describe.

    Returns:
        One tuple per column per row group: name, compression, encodings, and
        whether a dictionary page was written.

    """
    metadata = pq.ParquetFile(path).metadata
    facts: list[tuple[Any, ...]] = []
    for group in range(metadata.num_row_groups):
        for index in range(metadata.num_columns):
            column = metadata.row_group(group).column(index)
            facts.append(
                (
                    column.path_in_schema,
                    column.compression,
                    tuple(sorted(column.encodings)),
                    column.dictionary_page_offset is not None,
                )
            )
    return facts


def _stored_bytes(path: Path, column: str) -> tuple[int, int]:
    """Return the stored size of a column's dictionary page and of its whole chunk in a file's first row group.

    Args:
        path: The Parquet file to describe.
        column: The column's name.

    Returns:
        The bytes from the dictionary page to the first data page, which follows it, and the bytes of every page of
        the chunk, as compressed in the file.

    """
    group = pq.ParquetFile(path).metadata.row_group(0)
    chunk = next(group.column(i) for i in range(group.num_columns) if group.column(i).path_in_schema == column)
    return chunk.data_page_offset - chunk.dictionary_page_offset, chunk.total_compressed_size


def _row_groups(path: Path) -> list[int]:
    """List the row counts of a Parquet file's row groups, in order.

    Args:
        path: The Parquet file to describe.

    Returns:
        One row count per row group.

    """
    metadata = pq.ParquetFile(path).metadata
    return [metadata.row_group(group).num_rows for group in range(metadata.num_row_groups)]


# Each part holds more rows than one batch, so the sink writes several row
# groups per part: 65,536-row batches for the narrow output and 21,845 for
# the wide one, whose batches shrink to fit their wider rows. A part no
# larger than a batch is one row group however the write-back lays it out.
@pytest.mark.parametrize(("columns", "rows_per_part"), [(2, 150_000), (32, 50_000)])
def test_write_back_matches_our_sinks_parquet_settings(tmp_path, columns, rows_per_part):
    names: list[str] = [f"c{i}" for i in range(columns)]
    values: np.ndarray = np.random.default_rng(0).normal(100.0, 25.0, (2 * rows_per_part, columns))
    source: Path = tmp_path / "in.parquet"
    pq.write_table(pa.table({name: values[:, i] for i, name in enumerate(names)}), source, row_group_size=rows_per_part)

    fitted = Select(columns=names).fit([str(source)])
    ours: list[Path] = sorted(
        Path(part) for part in fitted.transform_parquet([str(source)], output_dir=tmp_path / "ours")
    )
    MEASURE.write_back(values, tmp_path / "theirs", MEASURE.sink_layout(ours))
    theirs: list[Path] = sorted((tmp_path / "theirs").glob("*.parquet"))

    assert [path.name for path in ours] == [path.name for path in theirs]
    for ours_part, theirs_part in zip(ours, theirs, strict=True):
        ours_meta = pq.ParquetFile(ours_part).metadata
        theirs_meta = pq.ParquetFile(theirs_part).metadata
        assert len(_row_groups(ours_part)) > 1
        assert _row_groups(theirs_part) == _row_groups(ours_part)
        assert _column_facts(theirs_part) == _column_facts(ours_part)
        assert theirs_meta.format_version == ours_meta.format_version
        assert theirs_meta.metadata == ours_meta.metadata
        assert abs(theirs_part.stat().st_size - ours_part.stat().st_size) < 0.01 * ours_part.stat().st_size


def test_write_parts_refuses_a_layout_that_does_not_cover_the_table(tmp_path):
    with pytest.raises(ValueError, match="the layout covers 4 rows, the table has 6"):
        MEASURE.write_parts(pa.table({"x": np.arange(6)}), tmp_path, [(2, 2), (2, 2)])


def test_dictionary_columns_follows_the_sinks_float_rule():
    schema = pa.schema(
        [("f64", pa.float64()), ("f32", pa.float32()), ("tag", pa.string()), ("n", pa.int64()), ("flag", pa.bool_())]
    )
    assert MEASURE.dictionary_columns(schema) == ["tag", "n", "flag"]


def test_dictionary_page_limit_follows_the_sinks_rule() -> None:
    lowered: pa.Schema = pa.schema(
        [
            ("i8", pa.int8()),
            ("i16", pa.int16()),
            ("i32", pa.int32()),
            ("i64", pa.int64()),
            ("u8", pa.uint8()),
            ("u16", pa.uint16()),
            ("u32", pa.uint32()),
            ("u64", pa.uint64()),
            ("f16", pa.float16()),
            ("f32", pa.float32()),
            ("f64", pa.float64()),
            ("flag", pa.bool_()),
            ("date32", pa.date32()),
            ("date64", pa.date64()),
            ("time32", pa.time32("ms")),
            ("time64", pa.time64("us")),
            ("timestamp", pa.timestamp("us")),
            ("duration", pa.duration("us")),
            ("text", pa.dictionary(pa.int32(), pa.string())),
            ("large_text", pa.dictionary(pa.int32(), pa.large_string())),
            ("bytes", pa.dictionary(pa.int8(), pa.binary())),
            ("large_bytes", pa.dictionary(pa.int32(), pa.large_binary())),
        ]
    )
    assert MEASURE.dictionary_page_limit(lowered) == 248 * 1024
    for name in lowered.names:
        assert MEASURE.dictionary_page_limit(pa.schema([lowered.field(name)])) == 248 * 1024, name
    kept: list[pa.DataType] = [
        pa.string(),
        pa.large_string(),
        pa.binary(),
        pa.binary(16),
        pa.decimal128(38, 4),
        pa.list_(pa.int64()),
        pa.list_(pa.float32(), 8),
        pa.struct([("n", pa.int64())]),
        pa.dictionary(pa.int32(), pa.int64()),
    ]
    # One such column keeps Arrow's limit wherever it stands, last or first.
    for kind in kept:
        other: pa.Field = pa.field("other", kind)
        assert MEASURE.dictionary_page_limit(lowered.append(other)) is None, kind
        assert MEASURE.dictionary_page_limit(lowered.insert(0, other)) is None, kind


# The sink is the reference: one table goes through it and through the
# write-back, and its column of distinct integers has to come out of both the
# same size. Beside numbers that is 248 KiB of dictionary and the rest of the
# rows whole; beside a plain string, a decimal or a list it is every value in
# the dictionary, because one such column keeps Arrow's 1 MiB for the part.
@pytest.mark.parametrize("beside", ["numbers", "plain string", "decimal", "list"])
def test_write_parts_lowers_the_dictionary_limit_when_the_sink_does(tmp_path: Path, beside: str) -> None:
    rows: int = 65_536
    rng = np.random.default_rng(4)
    ids: np.ndarray = rng.integers(-(2**62), 2**62, rows, dtype=np.int64)
    assert len(np.unique(ids)) == rows
    small: np.ndarray = np.arange(rows) % 7
    options: dict[str, Any] = {}
    if beside == "numbers":
        columns: dict[str, Any] = {"x": rng.normal(size=rows), "flag": small > 3, "n": small.astype(np.int32)}
    elif beside == "plain string":
        # Delta-encoded strings reach the sink as plain strings, as the write-back's are.
        columns = {"other": pa.array([f"s{value}" for value in small])}
        options = {"use_dictionary": ["id"], "column_encoding": {"other": "DELTA_BYTE_ARRAY"}}
    elif beside == "decimal":
        columns = {"other": pa.array(small).cast(pa.decimal128(38, 2))}
    else:
        columns = {"other": pa.ListArray.from_arrays(np.arange(rows + 1, dtype=np.int32), pa.array(small))}
    table: pa.Table = pa.table({"id": ids, **columns})
    source: Path = tmp_path / "in.parquet"
    pq.write_table(table, source, **options)

    ours: list[Path] = (
        Select(columns=table.column_names)
        .fit([str(source)])
        .transform_parquet([str(source)], output_dir=tmp_path / "ours")
    )
    MEASURE.write_parts(table, tmp_path / "theirs", MEASURE.sink_layout(ours))
    theirs: list[Path] = sorted((tmp_path / "theirs").glob("*.parquet"))

    assert [path.name for path in theirs] == [path.name for path in ours] == ["part-00000.parquet"]
    assert _row_groups(ours[0]) == _row_groups(theirs[0]) == [rows]
    ours_dictionary, ours_chunk = _stored_bytes(ours[0], "id")
    theirs_dictionary, theirs_chunk = _stored_bytes(theirs[0], "id")
    assert (ours_dictionary < 256 * 1024) == (beside == "numbers")
    assert (ours_dictionary >= 8 * rows) == (beside != "numbers")
    assert abs(theirs_dictionary - ours_dictionary) < 0.01 * ours_dictionary
    assert abs(theirs_chunk - ours_chunk) < 0.01 * ours_chunk


def test_write_parts_opens_each_part_through_a_buffer_as_large_as_the_sinks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The sink writes each part through a 64 KiB buffer. pyarrow writes a path without one, so write_parts opens
    # every part itself. The buffer changes the write calls the file sees and not its bytes, so the test watches how
    # each part is opened, and reads the table back.
    opened: list[tuple[str, int | None]] = []
    open_stream = pa.output_stream

    def watched(source: Any, *args: Any, **kwargs: Any) -> Any:
        opened.append((Path(source).name, kwargs.get("buffer_size")))
        return open_stream(source, *args, **kwargs)

    monkeypatch.setattr(pa, "output_stream", watched)
    table: pa.Table = pa.table({"x": np.arange(6)})
    MEASURE.write_parts(table, tmp_path, [(4, 2), (2, 2)])
    assert opened == [("part-00000.parquet", 64 * 1024), ("part-00001.parquet", 64 * 1024)]
    assert pa.concat_tables([pq.read_table(tmp_path / name) for name, _ in opened]).equals(table)


def test_read_matrix_does_not_copy_a_one_column_file(tmp_path):
    path: Path = tmp_path / "price.parquet"
    values: np.ndarray = np.random.default_rng(1).normal(100.0, 25.0, 5_000)
    pq.write_table(pa.table({"price": values}), path)
    matrix: np.ndarray = BENCH._read_matrix(path)
    assert matrix.shape == (5_000, 1)
    assert not matrix.flags.owndata
    assert np.array_equal(matrix[:, 0], values)


def test_read_matrix_stacks_several_columns(tmp_path):
    path: Path = tmp_path / "two.parquet"
    values: np.ndarray = np.random.default_rng(2).normal(0.0, 1.0, (5_000, 2))
    pq.write_table(pa.table({"a": values[:, 0], "b": values[:, 1]}), path)
    matrix: np.ndarray = BENCH._read_matrix(path)
    assert matrix.shape == (5_000, 2)
    assert np.array_equal(matrix, values)


def test_subsampled_quantile_fit_is_timed_with_the_read_inside(tmp_path, monkeypatch):
    path: Path = tmp_path / "price.parquet"
    pq.write_table(pa.table({"price": np.random.default_rng(3).normal(100.0, 25.0, 20_000)}), path)
    read: Any = BENCH._read_matrix
    calls: list[Path] = []

    def slow_read(target: Path) -> np.ndarray:
        calls.append(target)
        time.sleep(0.02)
        return read(target)

    monkeypatch.setattr(BENCH, "_read_matrix", slow_read)
    monkeypatch.setattr(MEASURE, "REPEATS", 3)
    stat = BENCH._measure_subsampled(path)
    assert calls == [path, path, path]
    assert stat.median >= 0.02


def test_repeats_are_enough_to_take_a_median_of():
    assert MEASURE.REPEATS >= 5


def test_repeat_reports_the_median_not_the_fastest_run(monkeypatch):
    monkeypatch.setattr(MEASURE, "REPEATS", 5)
    delays: list[float] = [0.0, 0.02, 0.02, 0.02, 0.02]
    pending: Any = iter(delays)
    stat = MEASURE.repeat(lambda: time.sleep(next(pending)))
    assert len(stat.samples) == len(delays)
    assert stat.low < 0.01
    assert stat.median >= 0.015
    assert stat.high >= stat.median


def test_summarise_reports_the_middle_and_the_spread():
    stat = MEASURE.summarise([3.0, 1.0, 100.0, 2.0, 4.0])
    assert stat.median == 3.0
    assert stat.q1 == 2.0
    assert stat.q3 == 4.0
    assert stat.iqr == 2.0
    assert stat.low == 1.0
    assert stat.high == 100.0


def test_run_order_is_a_shuffle_of_every_case():
    declared: list[str] = [case.name for case in BENCH.CASES]
    orders: set[tuple[str, ...]] = {
        tuple(case.name for case in BENCH._run_order(np.random.default_rng(seed))) for seed in range(20)
    }
    assert all(sorted(order) == sorted(declared) for order in orders)
    assert len(orders) > 1
    assert any(list(order) != declared for order in orders)


def test_generate_honours_the_row_group_layout(tmp_path):
    dataset = BENCH.generate(tmp_path, 1_000, 250)
    assert dataset.row_groups == 4
    for path in dataset.paths.values():
        assert pq.ParquetFile(path).num_row_groups == 4


def test_the_sweep_measures_more_than_one_layout():
    assert len(set(BENCH.SWEEP_ROW_GROUPS)) > 1
    assert min(BENCH.SWEEP_ROW_GROUPS) == 1


def test_both_scripts_summarise_their_runs_the_same_way():
    # The two once reported the best of three and the median of seven; sharing
    # the functions is what stops that from coming back.
    assert BENCH.repeat is MEASURE.repeat
    assert LABEL.repeat is MEASURE.repeat
    assert BENCH.REPEATS == MEASURE.REPEATS
    assert LABEL.REPEATS == MEASURE.REPEATS
    assert not hasattr(LABEL, "_best_of")


def test_both_scripts_write_upstream_output_through_the_same_sink_settings():
    assert BENCH.write_back is MEASURE.write_back
    assert LABEL.write_parts is MEASURE.write_parts
    assert BENCH.sink_layout is MEASURE.sink_layout
    assert LABEL.sink_layout is MEASURE.sink_layout
    assert LABEL.ROW_GROUP_SIZE == MEASURE.ROW_GROUP_SIZE


def test_as_table_takes_a_single_column_result():
    table = MEASURE.as_table(np.arange(6, dtype=np.int64))
    assert table.num_columns == 1
    assert table.num_rows == 6


def test_label_encoder_measures_in_a_shuffled_order():
    order: dict[str, Any] = {name: (lambda: None) for name in ("a", "b", "c", "d", "e")}
    orders: set[tuple[str, ...]] = set()
    for _ in range(20):
        stats, drawn = LABEL._measure(order, np.random.default_rng())
        assert sorted(stats) == sorted(order)
        assert sorted(drawn) == sorted(order)
        orders.add(tuple(drawn))
    assert len(orders) > 1
