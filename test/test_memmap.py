"""Tests for the memmap sink: layout, ordering, parity with Parquet output, and refusals."""

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fast_feature_transform import Cast, EngineError, LabelEncoder, Pipeline, Select, parquet_to_memmap


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


def _dtype(index: dict[str, Any]) -> np.dtype:
    """Build the numpy structured dtype an index describes.

    Args:
        index: The parsed index.json.

    Returns:
        The record dtype.

    """
    fields: list[Any] = []
    for column in index["columns"]:
        if column["size"] == 1:
            fields.append((column["name"], column["dtype"]))
        else:
            fields.append((column["name"], column["dtype"], (column["size"],)))
    return np.dtype(fields)


def _read(index_path: Path) -> np.ndarray:
    """Open the memmap an index describes, checking the index against numpy's layout.

    Args:
        index_path: Path of index.json.

    Returns:
        The records as a numpy memmap (an empty array for zero rows, which mmap cannot open).

    """
    index: dict[str, Any] = json.loads(index_path.read_text())
    assert index["format"] == "fast-feature-transform-memmap"
    assert index["byte_order"] == "little"
    dtype: np.dtype = _dtype(index)
    assert dtype.itemsize == index["itemsize"]
    fields = dtype.fields
    assert fields is not None
    assert [fields[c["name"]][1] for c in index["columns"]] == [c["offset"] for c in index["columns"]]
    if index["rows"] == 0:
        return np.empty(0, dtype)
    return np.memmap(index_path.parent / index["file"], dtype=dtype, mode="r", shape=(index["rows"],))


def test_memmap_of_transform_parquet_output_matches_the_parts(tmp_path: Path) -> None:
    files: list[str] = []
    for i in range(2):
        table: pa.Table = pa.table(
            {
                "device": ["ios", "web", "ios", "tv"],
                "price": [0.1 * (i + 1), 2.5, 3.0, 4.25],
                "count": pa.array([7 * i, -3, 12, 0], pa.int32()),
                "flag": [True, False, True, False],
            }
        )
        files.append(_write(tmp_path / f"in-{i}.parquet", table, row_group_size=3))
    pipeline: Pipeline = Pipeline(
        [("types", Cast({"price": "f32"})), ("encode", LabelEncoder(columns=["device"]))]
    ).fit(files)

    parts: list[Path] = pipeline.transform_parquet(files, output_dir=tmp_path / "parquet")
    records: np.ndarray = _read(parquet_to_memmap(parts, output_dir=tmp_path / "mmap"))
    expected: pa.Table = pa.concat_tables([pq.read_table(part) for part in parts])
    assert records.dtype.names == ("device", "price", "count", "flag")
    assert records.dtype == np.dtype([("device", "int64"), ("price", "float32"), ("count", "int32"), ("flag", "bool")])
    assert len(records) == 8
    for name in records.dtype.names:
        np.testing.assert_array_equal(records[name], expected.column(name).to_numpy())


def test_memmap_keeps_row_order_across_files_and_row_groups(tmp_path: Path) -> None:
    files: list[str] = [
        _write(
            tmp_path / f"f{i}.parquet",
            pa.table({"id": pa.array(range(i * 10, i * 10 + 10), pa.int64())}),
            row_group_size=3,
        )
        for i in range(3)
    ]
    records: np.ndarray = _read(parquet_to_memmap(files, output_dir=tmp_path / "out"))
    np.testing.assert_array_equal(records["id"], np.arange(30))


def test_memmap_writes_large_row_groups_in_chunks(tmp_path: Path) -> None:
    # One row group of 9,000,000 int64 rows packs to 68.7 MiB, more than one 64 MiB chunk.
    ids: np.ndarray = np.arange(9_000_000, dtype=np.int64)
    path: str = _write(tmp_path / "big.parquet", pa.table({"id": pa.array(ids)}), row_group_size=len(ids))
    assert pq.ParquetFile(path).num_row_groups == 1
    records: np.ndarray = _read(parquet_to_memmap([path], output_dir=tmp_path / "out"))
    np.testing.assert_array_equal(records["id"], ids)


def test_memmap_stores_fixed_size_lists_as_multi_element_fields(tmp_path: Path) -> None:
    values: np.ndarray = np.arange(12, dtype=np.float32).reshape(3, 4)
    embedding: pa.Array = pa.FixedSizeListArray.from_arrays(pa.array(values.ravel(), pa.float32()), 4)
    single: pa.Array = pa.FixedSizeListArray.from_arrays(pa.array([7, 8, 9], pa.int64()), 1)
    table: pa.Table = pa.table({"id": pa.array([1, 2, 3], pa.int16()), "emb": embedding, "one": single})
    path: str = _write(tmp_path / "emb.parquet", table)
    index_path: Path = parquet_to_memmap([path], output_dir=tmp_path / "out")
    index: dict[str, Any] = json.loads(index_path.read_text())
    assert index["columns"] == [
        {"name": "id", "dtype": "int16", "size": 1, "offset": 0},
        {"name": "emb", "dtype": "float32", "size": 4, "offset": 2},
        {"name": "one", "dtype": "int64", "size": 1, "offset": 18},
    ]
    records: np.ndarray = _read(index_path)
    np.testing.assert_array_equal(records["emb"], values)
    np.testing.assert_array_equal(records["id"], [1, 2, 3])
    np.testing.assert_array_equal(records["one"], [7, 8, 9])

    holes: pa.Array = pa.FixedSizeListArray.from_arrays(pa.array([1.0, 2.0, 3.0, None], pa.float32()), 2)
    path = _write(tmp_path / "holes.parquet", pa.table({"emb": holes}))
    with pytest.raises(EngineError, match="column 'emb' has a null at row 1"):
        parquet_to_memmap([path], output_dir=tmp_path / "holes")


def test_memmap_stores_plain_lists_of_one_length_as_vectors(tmp_path: Path) -> None:
    # DuckDB, pandas and transform_parquet's parts store a vector as a plain list.
    values: np.ndarray = np.arange(24, dtype=np.float32).reshape(6, 4)
    plain: pa.Array = pa.array(list(values), pa.list_(pa.float32()))
    files: list[str] = [
        _write(tmp_path / "a.parquet", pa.table({"emb": plain.slice(0, 4)}), row_group_size=3),
        _write(tmp_path / "b.parquet", pa.table({"emb": plain.slice(4)})),
    ]
    index_path: Path = parquet_to_memmap(files, output_dir=tmp_path / "out")
    assert json.loads(index_path.read_text())["columns"] == [
        {"name": "emb", "dtype": "float32", "size": 4, "offset": 0}
    ]
    np.testing.assert_array_equal(_read(index_path)["emb"], values)
    direct: Path = (
        Pipeline([("keep", Select(columns=["emb"]))]).fit(files).transform_memmap(files, output_dir=tmp_path / "direct")
    )
    assert (direct.parent / "data.mmap").read_bytes() == (index_path.parent / "data.mmap").read_bytes()

    large: str = _write(tmp_path / "large.parquet", pa.table({"emb": plain.cast(pa.large_list(pa.float32()))}))
    np.testing.assert_array_equal(_read(parquet_to_memmap([large], output_dir=tmp_path / "large"))["emb"], values)


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        # Twelve values over four rows look like three a row in the footer, so the rows themselves decide.
        (
            [[1.0, 2.0], [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]],
            "row 0 holds a list of 2, and a vector field needs 3 values",
        ),
        ([[1.0, 2.0], [1.0, 2.0, 3.0]], "the same number in every row, with no null or empty row"),
        ([[1.0, 2.0, 3.0], None, [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], "with no null or empty row"),
        ([[1.0, 2.0, 3.0], [], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], "with no null or empty row"),
        ([[1.0, None, 3.0], [1.0, 2.0, 3.0]], "column 'emb' has a null at row 0"),
        # Every row empty looks like one value a row in the footer.
        ([[], [], []], "row 0 holds an empty list, and a vector field cannot hold an empty row"),
    ],
    ids=["ragged_rows_with_an_even_total", "ragged_rows", "null_row", "empty_row", "null_value", "every_row_empty"],
)
def test_memmap_refuses_plain_lists_that_are_not_vectors(tmp_path: Path, rows: list[Any], message: str) -> None:
    path: str = _write(tmp_path / "emb.parquet", pa.table({"emb": pa.array(rows, pa.list_(pa.float32()))}))
    with pytest.raises(EngineError, match=message):
        parquet_to_memmap([path], output_dir=tmp_path / "out")
    assert not (tmp_path / "out" / "data.mmap").exists()


def test_memmap_refuses_files_whose_vectors_differ_in_length(tmp_path: Path) -> None:
    files: list[str] = [
        _write(tmp_path / f"{size}.parquet", pa.table({"emb": pa.array([[1.0] * size] * 2, pa.list_(pa.float32()))}))
        for size in (3, 4)
    ]
    with pytest.raises(EngineError, match="the same number in every row, with no null or empty row"):
        parquet_to_memmap(files, output_dir=tmp_path / "out")


def test_memmap_names_lists_it_cannot_hold(tmp_path: Path) -> None:
    # Two strings in every row, but no field holds strings.
    strings: str = _write(tmp_path / "strings.parquet", pa.table({"tags": [["a", "b"], ["c", "d"]]}))
    with pytest.raises(EngineError, match="column 'tags' is list<.*: a list column is written as a vector only when"):
        parquet_to_memmap([strings], output_dir=tmp_path / "strings")

    # Parts made from no rows show no vector width, though the source's fixed-size list does.
    empty: pa.Table = pa.table({"emb": pa.array([], pa.list_(pa.float32(), 4))})
    source: str = _write(tmp_path / "empty.parquet", empty)
    pipeline: Pipeline = Pipeline([("keep", Select(columns=["emb"]))]).fit([source])
    direct: Path = pipeline.transform_memmap([source], output_dir=tmp_path / "direct")
    assert json.loads(direct.read_text())["columns"] == [{"name": "emb", "dtype": "float32", "size": 4, "offset": 0}]
    parts: list[Path] = pipeline.transform_parquet([source], output_dir=tmp_path / "parts")
    with pytest.raises(EngineError, match="the input has no rows to show how many values its vectors hold"):
        parquet_to_memmap(parts, output_dir=tmp_path / "from-parts")


def test_memmap_of_zero_rows_is_an_empty_file(tmp_path: Path) -> None:
    path: str = _write(tmp_path / "empty.parquet", pa.table({"n": pa.array([], pa.int64())}))
    index_path: Path = parquet_to_memmap([path], output_dir=tmp_path / "out")
    assert json.loads(index_path.read_text())["rows"] == 0
    assert (tmp_path / "out" / "data.mmap").stat().st_size == 0
    assert len(_read(index_path)) == 0


def test_memmap_refuses_strings_and_nulls_and_leaves_nothing_behind(tmp_path: Path) -> None:
    text: str = _write(tmp_path / "text.parquet", pa.table({"device": ["a"], "n": [1]}))
    with pytest.raises(EngineError, match="column 'device' is utf8; cast or encode strings"):
        parquet_to_memmap([text], output_dir=tmp_path / "out")
    assert not (tmp_path / "out" / "data.mmap").exists()

    holes: str = _write(tmp_path / "holes.parquet", pa.table({"n": pa.array([1, None, 3], pa.int64())}))
    with pytest.raises(EngineError, match="column 'n' has a null at row 1"):
        parquet_to_memmap([holes], output_dir=tmp_path / "out")
    assert sorted(os.listdir(tmp_path / "out")) == []

    # A null is reported with its global row, not its row-group-local one.
    two: str = _write(
        tmp_path / "two.parquet", pa.table({"n": pa.array([1, 2, 3, None], pa.int64())}), row_group_size=2
    )
    with pytest.raises(EngineError, match="has a null at row 3"):
        parquet_to_memmap([two], output_dir=tmp_path / "out")


def test_memmap_replaces_a_previous_output_only_when_the_run_succeeds(tmp_path: Path) -> None:
    big: str = _write(tmp_path / "big.parquet", pa.table({"n": pa.array([1, 2, 3], pa.int64())}))
    small: str = _write(tmp_path / "small.parquet", pa.table({"n": pa.array([9], pa.int64())}))
    holes: str = _write(tmp_path / "holes.parquet", pa.table({"n": pa.array([1, None], pa.int64())}))
    out: Path = tmp_path / "out"
    old: np.ndarray = _read(parquet_to_memmap([big], output_dir=out))
    assert old["n"].tolist() == [1, 2, 3]

    records: np.ndarray = _read(parquet_to_memmap([small], output_dir=out, threads=1))
    assert records["n"].tolist() == [9]
    assert (out / "data.mmap").stat().st_size == 8
    # The new file was renamed into place: the old mapping still reads the old rows.
    assert old["n"].tolist() == [1, 2, 3]

    with pytest.raises(EngineError, match="has a null at row 1"):
        parquet_to_memmap([holes], output_dir=out)
    assert sorted(os.listdir(out)) == ["data.mmap", "index.json"]
    assert _read(out / "index.json")["n"].tolist() == [9]


def test_memmap_matches_every_numeric_width_and_bool(tmp_path: Path) -> None:
    table: pa.Table = pa.table(
        {
            "i8": pa.array([-128, 127], pa.int8()),
            "u16": pa.array([0, 65535], pa.uint16()),
            "u64": pa.array([0, 2**64 - 1], pa.uint64()),
            "f64": pa.array([0.1, -1e300], pa.float64()),
            "b": pa.array([True, False], pa.bool_()),
        }
    )
    path: str = _write(tmp_path / "widths.parquet", table)
    records: np.ndarray = _read(parquet_to_memmap([path], output_dir=tmp_path / "out"))
    for name in table.column_names:
        np.testing.assert_array_equal(records[name], table.column(name).to_numpy())
