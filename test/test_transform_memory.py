"""Tests for batched transforms: exact values, part order, batch sizing, and cleanup after a failure."""

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fast_feature_transform import (
    EngineError,
    HashingEncoder,
    LabelEncoder,
    MinMaxScaler,
    Pipeline,
    Select,
    StandardScaler,
    parquet_to_memmap,
)


def _assert_bits(left: pa.Table, right: pa.Table) -> None:
    assert left.schema == right.schema
    assert left.equals(right)
    for before, after in zip(left.columns, right.columns, strict=True):
        if pa.types.is_floating(before.type):
            valid = before.is_valid().to_numpy()
            assert before.to_numpy()[valid].tobytes() == after.to_numpy()[valid].tobytes()


@pytest.mark.parametrize("threads", [1, 4])
def test_chunked_outputs_keep_part_order_and_exact_memmap_bytes(tmp_path: Path, threads: int) -> None:
    rows = 150003
    index = np.arange(rows, dtype=np.int64)
    table = pa.table(
        {
            "x": (index % 1021 - 510).astype(np.float64) / 32,
            "key": pa.array(["a", "b", "new"]).take(pa.array(index % 3)),
            "flag": index % 7 == 0,
        }
    )
    training = tmp_path / "fit.parquet"
    pq.write_table(table.slice(0, 97), training)
    pipeline = Pipeline(
        [
            ("encode", LabelEncoder(columns=["key"])),
            ("scale", StandardScaler(columns=["x"])),
            ("range", MinMaxScaler(columns=["x"], clip=True)),
        ]
    ).fit([training], threads=threads)
    files = [tmp_path / f"input-{i}.parquet" for i in range(3)]
    for path, values in zip(files, [table.slice(0, 0), table, table.slice(41, 17)], strict=True):
        pq.write_table(values, path, row_group_size=rows)
    parts = pipeline.transform_parquet(files, output_dir=tmp_path / "parts")
    assert len(parts) == sum(pq.ParquetFile(path).num_row_groups for path in files)
    assert pq.ParquetFile(parts[1]).num_row_groups > 1
    actual = pq.read_table(parts)
    assert [(field.name, field.type) for field in actual.schema] == [
        ("x", pa.float64()),
        ("key", pa.int64()),
        ("flag", pa.bool_()),
    ]
    # The record path is the oracle: it sees one row at a time, so no batching can touch it.
    rows = pa.concat_tables([table, table.slice(41, 17)]).to_pylist()
    expected = pa.Table.from_pylist([pipeline.transform_dict(row) for row in rows], schema=actual.schema)
    _assert_bits(expected, actual)
    direct = pipeline.transform_memmap(files, output_dir=tmp_path / "direct")
    converted = parquet_to_memmap(parts, output_dir=tmp_path / "converted", threads=threads)
    assert json.loads(direct.read_text()) == json.loads(converted.read_text())
    assert (direct.parent / "data.mmap").read_bytes() == (converted.parent / "data.mmap").read_bytes()
    # The index owns the physical field ordering; compare each declared field.
    description = json.loads(direct.read_text())
    dtype = np.dtype([(field["name"], field["dtype"]) for field in description["columns"]])
    records = np.fromfile(direct.parent / "data.mmap", dtype=dtype)
    for name in expected.column_names:
        assert records[name].tobytes() == expected[name].to_numpy().tobytes()


def test_wide_intermediate_and_projected_input_use_appropriate_batches(tmp_path: Path) -> None:
    keys = [f"key-{i:03d}" for i in range(256)]
    index = np.arange(10001, dtype=np.int64)
    table = pa.table({"key": pa.array(keys).take(pa.array(index % len(keys)))})
    source = tmp_path / "wide.parquet"
    pq.write_table(table, source, row_group_size=len(index))
    pipeline = Pipeline(
        [("encode", HashingEncoder(columns=["key"], n_components=256)), ("select", Select(columns=["col_0"]))]
    ).fit([source], threads=1)
    parts = pipeline.transform_parquet([source], output_dir=tmp_path / "wide-output")
    assert len(parts) == 1
    assert pq.ParquetFile(parts[0]).num_row_groups > 1
    result = pq.read_table(parts)
    expected = np.array([pipeline.transform_dict({"key": key})["col_0"] for key in keys], dtype=np.int64)
    np.testing.assert_array_equal(result["col_0"].to_numpy(), expected[index % 256])

    # Hundreds of unread source columns must not force tiny output batches.
    numeric = pa.table({f"x{i}": index for i in range(256)})
    projected_source = tmp_path / "projected.parquet"
    pq.write_table(numeric, projected_source, row_group_size=len(index))
    narrow = Select(columns=["x0"]).fit([projected_source], threads=1)
    projected = narrow.transform_parquet([projected_source], output_dir=tmp_path / "projected-output")
    assert pq.ParquetFile(projected[0]).num_row_groups == 1
    assert pq.read_table(projected).equals(numeric.select(["x0"]))


def test_vector_elements_count_like_scalars_and_binary_values_by_their_bytes(tmp_path: Path) -> None:
    # A vector's element counts 8 bytes like a scalar column, however narrow,
    # so 768 of them batch like 768 float64 columns or a 6,144-byte binary
    # value, whether they arrive as a fixed-size list or as the plain list
    # Parquet stores, whose length comes from the footers. Such a batch holds
    # 910 rows, and a row group takes batches until it holds 4,096 rows, five
    # of these, so 5,000 rows are two row groups.
    rows, dim = 5000, 768
    values = np.random.default_rng(0).standard_normal((rows, dim), dtype=np.float32)
    flat = pa.array(values.reshape(-1))
    offsets = pa.array(np.arange(0, rows * dim + 1, dim, dtype=np.int32))
    holes = pa.array([i % 100 == 0 for i in range(rows)])
    tables = {
        "fixed": pa.table({"v": pa.FixedSizeListArray.from_arrays(flat, dim)}),
        "narrow": pa.table({"v": pa.FixedSizeListArray.from_arrays(pa.array(np.zeros(rows * dim, np.int8)), dim)}),
        "plain": pa.table({"v": pa.ListArray.from_arrays(offsets, flat)}),
        "binary": pa.table(
            {"v": pa.Array.from_buffers(pa.binary(8 * dim), rows, [None, pa.py_buffer(values.repeat(2))])}
        ),
        "scalars": pa.table({f"x{i}": values[:, i].astype(np.float64) for i in range(dim)}),
        # A null row in every hundred: no single width, the same average.
        "nulls": pa.table({"v": pa.ListArray.from_arrays(offsets, flat, mask=holes)}),
        # Strings are no vector: a list of them stays one variable-length value.
        "tokens": pa.table({"v": [[f"word{(i + k) % 5000}" for k in range(100)] for i in range(rows)]}),
    }
    largest = {}
    for name, table in tables.items():
        source = tmp_path / f"{name}.parquet"
        pq.write_table(table, source, row_group_size=rows)
        selected = Select(columns=table.column_names).fit([source], threads=1)
        parts = selected.transform_parquet([source], output_dir=tmp_path / name)
        metadata = pq.ParquetFile(parts[0]).metadata
        assert metadata.num_rows == rows
        largest[name] = max(metadata.row_group(group).num_rows for group in range(metadata.num_row_groups))
        if name in ("fixed", "plain"):
            np.testing.assert_array_equal(
                pq.read_table(parts)["v"].combine_chunks().flatten().to_numpy(), values.ravel()
            )
    assert largest.pop("tokens") == rows
    assert largest.pop("nulls") < rows
    assert largest["fixed"] < rows
    assert largest == dict.fromkeys(largest, largest["scalars"])


def test_a_plain_list_keeps_its_footer_width_through_later_stages(tmp_path: Path) -> None:
    # Hashing a key widens the batch while the vector passes through: the
    # plain list counts as wide there as the fixed-size list's tag does. A
    # batch holds 682 rows and a row group six batches, so 4,500 rows are two
    # row groups.
    rows, dim = 4500, 768
    flat = pa.array(np.random.default_rng(0).standard_normal(rows * dim, dtype=np.float32))
    vectors = {
        "fixed": pa.FixedSizeListArray.from_arrays(flat, dim),
        "plain": pa.ListArray.from_arrays(pa.array(np.arange(0, rows * dim + 1, dim, dtype=np.int32)), flat),
    }
    largest = {}
    for name, vector in vectors.items():
        source = tmp_path / f"{name}.parquet"
        table = pa.table({"key": [f"k{i % 7}" for i in range(rows)], "v": vector})
        pq.write_table(table, source, row_group_size=rows)
        pipeline = Pipeline([("hash", HashingEncoder(columns=["key"], n_components=256))]).fit([source], threads=1)
        metadata = pq.ParquetFile(pipeline.transform_parquet([source], output_dir=tmp_path / name)[0]).metadata
        largest[name] = max(metadata.row_group(group).num_rows for group in range(metadata.num_row_groups))
    assert largest["plain"] == largest["fixed"] < rows


def test_late_transform_failures_clean_output_and_keep_previous_memmap(tmp_path: Path) -> None:
    table = pa.table({"x": np.arange(150003, dtype=np.float64)})
    source = tmp_path / "source.parquet"
    pq.write_table(table, source, row_group_size=table.num_rows)
    pipeline = Pipeline([("scale", StandardScaler(columns=["x"]))]).fit([source], threads=1)
    directory = tmp_path / "packed"
    index = pipeline.transform_memmap([source], output_dir=directory)
    previous = index.read_bytes(), (directory / "data.mmap").read_bytes()
    bad = tmp_path / "bad.parquet"
    for failure in (float("nan"), None):
        broken = pa.concat_tables(
            [table.slice(0, table.num_rows - 1), pa.table({"x": pa.array([failure], pa.float64())})]
        )
        pq.write_table(broken, bad, row_group_size=table.num_rows)
        with pytest.raises(EngineError, match="not finite|cannot hold nulls"):
            pipeline.transform_memmap([bad], output_dir=directory)
        assert (index.read_bytes(), (directory / "data.mmap").read_bytes()) == previous
        assert not list(directory.glob("*.tmp"))
    pq.write_table(pa.concat_tables([table, pa.table({"x": [float("nan")]})]), bad, row_group_size=table.num_rows + 1)
    output = tmp_path / "parts"
    pipeline.transform_parquet([source], output_dir=output)
    with pytest.raises(EngineError, match="not finite"):
        pipeline.transform_parquet([bad], output_dir=output)
    assert not list(output.glob("part-*.parquet*"))


def test_split_row_group_keeps_bits_nulls_and_fixed_list_values(tmp_path: Path) -> None:
    rows = 10001
    values = np.arange(rows * 128, dtype=np.int16)
    vectors = pa.FixedSizeListArray.from_arrays(pa.array(values), 128)
    numbers = np.tile([-0.0, 0.0, 1.25, -2.5], rows // 4 + 1)[:rows]
    keys = pa.array(["a", "b", "c", "d"]).take(pa.array(np.arange(rows) % 4)).dictionary_encode()
    table = pa.table({"vector": vectors, "x": pa.array(numbers, mask=np.arange(rows) % 11 == 0), "key": keys}).slice(
        3, rows - 7
    )
    source = tmp_path / "fit.parquet"
    pq.write_table(table.slice(0, 64), source)
    pipeline = Pipeline(
        [
            ("scale", StandardScaler(columns=["x"], with_mean=False)),
            ("expand", HashingEncoder(columns=["key"], n_components=256)),
            ("select", Select(columns=["vector", "x"])),
        ]
    ).fit([source], threads=1)
    whole = tmp_path / "whole.parquet"
    pq.write_table(table, whole, row_group_size=table.num_rows)  # one row group, read in several batches
    parts = pipeline.transform_parquet([whole], output_dir=tmp_path / "out")
    assert pq.ParquetFile(parts[0]).num_row_groups > 1
    result = pq.read_table(parts)
    # Parquet stores the fixed-size list as a plain list<int16>.
    assert result["vector"].cast(table["vector"].type).equals(table["vector"])
    # Every row against the record path: _assert_bits checks the null mask and each float's bits, -0.0 included.
    records = zip(table["x"].to_pylist(), table["key"].to_pylist(), strict=True)
    expected = [pipeline.transform_dict({"vector": None, "x": x, "key": key})["x"] for x, key in records]
    _assert_bits(pa.table({"x": pa.array(expected, pa.float64())}), result.select(["x"]))
    assert result["x"].is_null().equals(table["x"].is_null())


def test_wide_fit_prefix_keeps_exact_statistics_across_partitions(tmp_path: Path) -> None:
    rows = 10001
    index = np.arange(rows, dtype=np.int64)
    table = pa.table({"key": pa.array(["a", "b", "c", "d"]).take(pa.array(index % 4))})
    large = tmp_path / "large.parquet"
    small = tmp_path / "small.parquet"
    pq.write_table(table, large, row_group_size=rows)
    pq.write_table(table, small, row_group_size=127)

    def make_pipeline() -> Pipeline:
        return Pipeline(
            [
                ("expand", HashingEncoder(columns=["key"], n_components=256)),
                ("scale", StandardScaler()),
                ("select", Select(columns=["col_0"])),
            ]
        )

    serial = make_pipeline().fit([large], threads=1)
    parallel = make_pipeline().fit([small], threads=4)
    serial_scale, parallel_scale = serial["scale"], parallel["scale"]
    assert isinstance(serial_scale, StandardScaler)
    assert isinstance(parallel_scale, StandardScaler)
    assert serial_scale._state() == parallel_scale._state()
    _assert_bits(
        pq.read_table(serial.transform_parquet([large], output_dir=tmp_path / "serial")),
        pq.read_table(parallel.transform_parquet([large], output_dir=tmp_path / "parallel")),
    )
