"""The Parquet settings a timing comparison writes another library's output with.

``benchmark/measure.py`` writes that output the way this package's sink
writes its own, and the tests compare the two files: compression, encodings,
dictionary limit, write buffer, parts and row groups.
"""

import importlib
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fast_feature_transform import Select

BENCHMARK_DIR: Path = Path(__file__).resolve().parent.parent / "benchmark"

# The module is imported the way a script beside it imports it, with its
# directory on sys.path.
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))
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


def test_as_table_takes_a_single_column_result():
    table = MEASURE.as_table(np.arange(6, dtype=np.int64))
    assert table.num_columns == 1
    assert table.num_rows == 6
