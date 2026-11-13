"""Runtime boundaries: direct training output, error reporting, output directories, and concurrency."""

import itertools
import json
import pickle
import re
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fast_feature_transform import (
    CountEncoder,
    EngineError,
    HelmertEncoder,
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    Pipeline,
    RankHotEncoder,
    Select,
    StandardScaler,
    parquet_to_memmap,
)


def _fit(tmp_path: Path, table: pa.Table, steps: list[Any], *, threads: int = 4) -> Pipeline:
    path = tmp_path / "fit.parquet"
    pq.write_table(table, path, row_group_size=2)
    return Pipeline(steps).fit([path], threads=threads)


def test_vocab_size_does_not_materialize_state(tmp_path: Path) -> None:
    table = pa.table({"key": ["b", "a", "b"]})
    pipeline = _fit(tmp_path, table, [("encode", LabelEncoder(columns=["key"]))])
    encoder = pipeline["encode"]
    assert isinstance(encoder, LabelEncoder)
    assert encoder._state_cache is None
    assert encoder.vocab_size("key") == 3
    assert encoder._state_cache is None
    restored = pickle.loads(pickle.dumps(pipeline))
    restored_encoder = restored["encode"]
    assert isinstance(restored_encoder, LabelEncoder)
    assert restored_encoder.vocab_size("key") == 3
    assert restored_encoder._state_cache is None
    with pytest.raises(KeyError, match="absent"):
        encoder.vocab_size("absent")


def test_vocab_size_follows_a_refit_and_survives_a_pickle(tmp_path: Path) -> None:
    source = tmp_path / "source.parquet"
    pq.write_table(pa.table({"key": ["a", "b"], "other": ["a", "a"]}), source)
    encoder = LabelEncoder().fit([source])
    assert encoder.vocab_size("key") == 3
    assert encoder.vocab_size("other") == 2
    assert encoder._state_cache is None
    restored = pickle.loads(pickle.dumps(encoder))
    assert restored.vocab_size("key") == 3
    pq.write_table(pa.table({"key": ["c"], "other": ["c"]}), source)
    encoder.fit([source])
    assert encoder.vocab_size("key") == 2
    assert encoder._state_cache is None
    assert restored.vocab_size("key") == 3


@pytest.mark.parametrize("column", [None, 0, ("key",), [], {}])
def test_vocab_size_preserves_dictionary_key_errors(tmp_path: Path, column: Any) -> None:
    encoder = LabelEncoder(columns=["key"])
    with pytest.raises(RuntimeError, match="not fitted"):
        encoder.vocab_size(column)
    pipeline = _fit(tmp_path, pa.table({"key": ["a", "b"]}), [("encode", encoder)])
    assert pipeline["encode"] is encoder
    error = TypeError if isinstance(column, (list, dict)) else KeyError
    with pytest.raises(error):
        encoder.vocab_size(column)


@pytest.mark.parametrize("threads", [1, 4])
def test_direct_memmap_matches_parquet_round_trip_bytes(tmp_path: Path, threads: int) -> None:
    table = pa.table(
        {
            "z": ["b", "a", "b", "c", "a"],
            "x": [-0.0, 0.0, 1.25, -2.5, 3.75],
            "flag": [True, False, True, True, False],
        }
    )
    pipeline = _fit(
        tmp_path,
        table,
        [
            ("encode", LabelEncoder(columns=["z"])),
            ("scale", StandardScaler(columns=["x"], with_mean=False)),
        ],
        threads=threads,
    )
    source = [tmp_path / "fit.parquet"]
    parts = pipeline.transform_parquet(source, output_dir=tmp_path / "parts")
    reference = parquet_to_memmap(parts, output_dir=tmp_path / "reference", threads=threads)
    direct = pipeline.transform_memmap(source, output_dir=tmp_path / "direct")
    assert json.loads(direct.read_text()) == json.loads(reference.read_text())
    assert (direct.parent / "data.mmap").read_bytes() == (reference.parent / "data.mmap").read_bytes()


def test_parts_holding_a_vector_convert_to_the_direct_memmap(tmp_path: Path) -> None:
    # Parquet stores a fixed-size list as a plain list, so the parts read back
    # as list<int16>; their footers still show two values a row, which makes
    # the column a two-element field again.
    table = pa.table({"vector": pa.array([[1, 2], [3, 4], [5, 6]], pa.list_(pa.int16(), 2))})
    pipeline = _fit(tmp_path, table, [("select", Select(columns=["vector"]))])
    source = [tmp_path / "fit.parquet"]
    parts = pipeline.transform_parquet(source, output_dir=tmp_path / "parts")
    assert pq.read_schema(parts[0]).field("vector").type == pa.list_(pa.int16())
    reference = parquet_to_memmap(parts, output_dir=tmp_path / "reference")
    direct = pipeline.transform_memmap(source, output_dir=tmp_path / "direct")
    assert json.loads(direct.read_text()) == json.loads(reference.read_text())
    assert (direct.parent / "data.mmap").read_bytes() == (reference.parent / "data.mmap").read_bytes()


def test_direct_memmap_failure_preserves_previous_output(tmp_path: Path) -> None:
    pipeline = _fit(tmp_path, pa.table({"x": [1.0, 2.0, 3.0]}), [("scale", StandardScaler(columns=["x"]))])
    directory = tmp_path / "direct"
    index = pipeline.transform_memmap([tmp_path / "fit.parquet"], output_dir=directory)
    previous = (index.read_bytes(), (directory / "data.mmap").read_bytes())
    bad = tmp_path / "bad.parquet"
    pq.write_table(pa.table({"x": [None, 2.0, 3.0]}), bad, row_group_size=1)
    with pytest.raises(EngineError, match="cannot hold nulls"):
        pipeline.transform_memmap([bad], output_dir=directory)
    assert (index.read_bytes(), (directory / "data.mmap").read_bytes()) == previous
    assert not list(directory.glob("*.tmp"))


@pytest.mark.parametrize("row_group_size", [None, 2])
def test_dictionary_keys_nulls_and_signed_zeros_match_records(tmp_path: Path, row_group_size: int | None) -> None:
    fit = pa.table({"x": [-2.0, -1.0, 1.0, 2.0], "y": [-2.0, -1.0, 1.0, 2.0], "key": ["a", "b", "a", "b"]})
    pipeline = _fit(
        tmp_path,
        fit,
        [
            ("encode", LabelEncoder(columns=["key"])),
            # y is only divided, which keeps the sign of a zero; MinMaxScaler's offset would erase it from x.
            ("scale", StandardScaler(columns=["x", "y"], with_mean=False)),
            ("range", MinMaxScaler(columns=["x"], clip=True)),
        ],
    )
    values = np.array([99.0, -0.0, 0.0, np.nextafter(0.0, 1.0), -3.0, 3.0, 88.0])
    x = pa.array(values, mask=np.array([False, False, False, False, True, False, False])).slice(1, 5)
    keys = pa.array(["unused", "a", "new", None, "b", "a", "unused"]).dictionary_encode().slice(1, 5)
    table = pa.table({"x": x, "y": x, "key": keys})
    expected = [pipeline.transform_dict(row) for row in table.to_pylist()]
    source = tmp_path / "input.parquet"
    pq.write_table(table, source, row_group_size=row_group_size)  # the engine reads "key" back as a dictionary
    result = pq.read_table(pipeline.transform_parquet([source], output_dir=tmp_path / "out"))
    assert result.to_pylist() == expected
    for column in ("x", "y"):
        valid = [i for i, row in enumerate(expected) if row[column] is not None]
        expected_bits = np.array([expected[i][column] for i in valid], dtype=np.float64).view(np.uint64)
        got_bits = np.array([result[column][i].as_py() for i in valid], dtype=np.float64).view(np.uint64)
        np.testing.assert_array_equal(got_bits, expected_bits)
    assert np.signbit(result["y"][0].as_py())  # also fails if both paths lose the sign the same way


def test_one_handle_supports_concurrent_parquet_transforms(tmp_path: Path) -> None:
    table = pa.table({"x": np.arange(128, dtype=np.float64) / 16}).slice(3, 117)
    pipeline = _fit(
        tmp_path,
        table,
        [
            ("scale", StandardScaler(columns=["x"])),
            ("range", MinMaxScaler(columns=["x"])),
        ],
    )
    source = [tmp_path / "fit.parquet"]
    expected = pq.read_table(pipeline.transform_parquet(source, output_dir=tmp_path / "expected"))
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(
            executor.map(
                lambda run: pq.read_table(pipeline.transform_parquet(source, output_dir=tmp_path / f"out-{run}")),
                range(16),
            )
        )
    assert all(result.equals(expected) for result in results)


# REGRESSION: with several threads, the first failing row group in source
# order decides the error, as with one thread, even when it fails last: here
# one large group whose only null comes at its end, ahead of small groups that
# fail on their first row. Whichever row group failed first used to decide it.
def test_parquet_transform_reports_the_first_failing_row_group(tmp_path: Path) -> None:
    pq.write_table(pa.table({"c": ["a", "b"], "d": ["x", "y"]}), tmp_path / "fit.parquet")
    first, later = tmp_path / "first.parquet", tmp_path / "later.parquet"
    rows = 300_000
    pq.write_table(pa.table({"c": ["a"] * (rows - 1) + [None], "d": ["x"] * rows}), first, row_group_size=rows)
    pq.write_table(pa.table({"c": ["a"] * 4000, "d": ["zz"] * 4000}), later, row_group_size=250)
    model = OneHotEncoder(columns=["c", "d"], handle_unknown="error", handle_missing="error")
    model.fit([tmp_path / "fit.parquet"], threads=4)
    for _ in range(5):
        with pytest.raises(EngineError, match="column 'c' has a null"):
            model.transform_parquet([first, later], output_dir=tmp_path / "out")


# A generated name holding a NUL byte must stay whole: 'c_\x00x' cut at the NUL
# would become 'c_', a duplicate of the empty category's column.
@pytest.mark.parametrize("encoder", [OneHotEncoder, RankHotEncoder])
def test_generated_names_that_hold_a_nul_byte_stay_whole(tmp_path: Path, encoder: Any) -> None:
    categories = ["", "\x00x", "y"]
    pq.write_table(pa.table({"c": categories}), tmp_path / "fit.parquet")
    model = encoder(columns=["c"], use_cat_names=True).fit([tmp_path / "fit.parquet"])
    expected = ["c_", "c_\x00x", "c_y"]
    assert list(model.transform_dict({"c": "y"})) == expected
    (part,) = model.transform_parquet([tmp_path / "fit.parquet"], output_dir=tmp_path / "out")
    assert pq.read_table(part).column_names == expected


# An input column's name holding a NUL byte must stay whole: "c\x00x" cut at the
# NUL would become "c", so a pipeline over "c" would read it, one fitted on
# "c\x00x" would refuse it, and the two names together would collide.
def test_input_names_that_hold_a_nul_byte_stay_whole(tmp_path: Path) -> None:
    pq.write_table(pa.table({"c": ["a", "b"], "c\x00x": ["b", "a"]}), tmp_path / "fit.parquet")
    model = OrdinalEncoder(columns=["c", "c\x00x"]).fit([tmp_path / "fit.parquet"])
    table = pa.table({"c": ["a", "b"], "c\x00x": ["a", "b"]})
    pq.write_table(table, tmp_path / "input.parquet")
    expected = [model.transform_dict(row) for row in table.to_pylist()]
    parts = model.transform_parquet([tmp_path / "input.parquet"], output_dir=tmp_path / "out")
    assert pq.read_table(parts).to_pylist() == expected
    only_c = OrdinalEncoder(columns=["c"]).fit([tmp_path / "fit.parquet"])
    pq.write_table(pa.table({"c\x00x": ["b"]}), tmp_path / "nul-only.parquet")
    with pytest.raises(EngineError, match="column 'c' not found in input"):
        only_c.transform_parquet([tmp_path / "nul-only.parquet"], output_dir=tmp_path / "refused")


# REGRESSION: a pipeline that emits no columns used to fail transform_parquet
# only at its first part file, after transforming, and transform_memmap with a
# message about its input.
def test_file_transforms_refuse_a_result_without_columns_up_front(tmp_path: Path) -> None:
    pq.write_table(pa.table({"c": ["a", "a", "a"]}), tmp_path / "one.parquet")
    model = HelmertEncoder(columns=["c"]).fit([tmp_path / "one.parquet"])
    assert model.transform_dict({"c": "a"}) == {}
    out = tmp_path / "out"
    out.mkdir()
    (out / "part-00000.parquet").write_bytes(b"left alone")
    with pytest.raises(EngineError, match="emits no columns, so its rows cannot be written to Parquet"):
        model.transform_parquet([tmp_path / "one.parquet"], output_dir=out)
    assert (out / "part-00000.parquet").read_bytes() == b"left alone"
    with pytest.raises(EngineError, match="emits no columns, and a memmap record needs at least one field"):
        model.transform_memmap([tmp_path / "one.parquet"], output_dir=tmp_path / "memmap")
    # Without rows there is nothing to lose: Parquet still writes the empty part,
    # while a memmap needs a field whatever the row count.
    pq.write_table(pa.table({"c": pa.array([], pa.string())}), tmp_path / "empty.parquet")
    (part,) = model.transform_parquet([tmp_path / "empty.parquet"], output_dir=tmp_path / "empty-out")
    assert (pq.read_metadata(part).num_columns, pq.read_metadata(part).num_rows) == (0, 0)
    with pytest.raises(EngineError, match="emits no columns, and a memmap record needs at least one field"):
        model.transform_memmap([tmp_path / "empty.parquet"], output_dir=tmp_path / "empty-memmap")


# REGRESSION: Arrow's Parquet reader decodes dictionary indices without
# checking them against the dictionary, so a corrupt file used to crash fits
# and transforms, or encode rows as unseen. Here rows 16 on point one past a
# three-entry dictionary, written into an uncompressed data page.
@pytest.mark.parametrize("encoder", [OrdinalEncoder, CountEncoder])
def test_parquet_dictionary_indices_outside_the_dictionary_are_refused(tmp_path: Path, encoder: Any) -> None:
    clean, corrupt = tmp_path / "clean.parquet", tmp_path / "corrupt.parquet"
    table = pa.table({"c": (["a"] * 16 + ["b"] * 16 + ["c"] * 16) * 2})
    pq.write_table(table, clean, compression="NONE", write_statistics=False, data_page_version="1.0")
    # Bit width 2, then runs of 16 (header 0x20) of the indices 0, 1, 2, 0, 1, 2;
    # rewritten as 16 rows of index 0, then 80 of index 3.
    runs = bytes([0x02, 0x20, 0x00, 0x20, 0x01, 0x20, 0x02, 0x20, 0x00, 0x20, 0x01, 0x20, 0x02])
    data = clean.read_bytes()
    assert data.count(runs) == 1
    corrupt.write_bytes(data.replace(runs, bytes([0x02, 0x20, 0x00, 0xA0, 0x01, 0x03]) + runs[6:]))
    message = "column 'c' has a dictionary index outside its dictionary"
    with pytest.raises(EngineError, match=message):
        encoder(columns=["c"]).fit([corrupt])
    model = encoder(columns=["c"]).fit([clean])
    with pytest.raises(EngineError, match=message):
        model.transform_parquet([corrupt], output_dir=tmp_path / "out")
    with pytest.raises(EngineError, match=message):
        model.transform_memmap([corrupt], output_dir=tmp_path / "memmap")


_NESTED_ENTRIES = ["aa", "bb", *(f"c{i}" for i in range(10)), "zz"]


def _nested_strings(kind: str) -> tuple[pa.Array, int]:
    """A column of `kind` whose strings use all 13 of _NESTED_ENTRIES, with nulls among them and among its rows.

    A map's keys and rows hold no null.

    Args:
        kind: "list", "large_list", "fixed_size_list", "struct", "map keys", "map values", "list of structs" or
            "fixed_size_list of structs".

    Returns:
        The column, and the number of the Parquet leaf that holds the strings.
    """
    words = [_NESTED_ENTRIES[i % 13] for i in range(195)]
    strings = pa.array([None if i % 7 == 3 else word for i, word in enumerate(words)])
    offsets = pa.array(range(0, 196, 3), pa.int32())
    rows = pa.array([row % 5 == 2 for row in range(65)])
    records = pa.StructArray.from_arrays(
        [strings, pa.array(range(195))], ["name", "n"], mask=pa.array([row % 11 == 5 for row in range(195)])
    )
    if kind == "list":
        return pa.ListArray.from_arrays(offsets, strings, mask=rows), 0
    if kind == "large_list":
        return pa.LargeListArray.from_arrays(offsets.cast(pa.int64()), strings, mask=rows), 0
    if kind == "fixed_size_list":
        return pa.FixedSizeListArray.from_arrays(strings, 3, mask=rows), 0
    if kind == "struct":
        return records, 0
    if kind == "list of structs":
        return pa.ListArray.from_arrays(offsets, records, mask=rows), 0
    if kind == "fixed_size_list of structs":
        return pa.FixedSizeListArray.from_arrays(records, 3, mask=rows), 0
    return pa.MapArray.from_arrays(offsets, pa.array(words), strings), 0 if kind == "map keys" else 1


# REGRESSION: strings inside a list, large list, struct or map are read as a
# dictionary in a child array, and their indices are checked as a top-level
# column's are; strings inside a fixed-size list are read plain, Arrow's
# decoder checks their indices, and the column it fails on is named. A file
# whose nested indices lay outside the dictionary used to crash
# transform_parquet, or, inside a fixed-size list with null rows, to pass
# through with wrong values. Here the dictionary page of the strings declares
# 12 entries instead of 13, so the rows that use the last one point past the
# dictionary the reader keeps.
@pytest.mark.parametrize(
    "kind",
    [
        "list",
        "large_list",
        "fixed_size_list",
        "struct",
        "map keys",
        "map values",
        "list of structs",
        "fixed_size_list of structs",
    ],
)
def test_parquet_dictionary_indices_outside_a_nested_dictionary_are_refused(tmp_path: Path, kind: str) -> None:
    column, leaf = _nested_strings(kind)
    table = pa.table({"c": column, "x": pa.array([float(row) for row in range(len(column))])})
    clean, corrupt = tmp_path / "clean.parquet", tmp_path / "corrupt.parquet"
    pq.write_table(table, clean, compression="NONE", write_statistics=False)
    chunk = pq.ParquetFile(clean).metadata.row_group(0).column(leaf)
    # In Thrift's compact encoding the dictionary page header (field 7 of the
    # page header, a struct: 0x4C) opens with num_values (field 1, an i32:
    # 0x15) as a zigzag varint: 13 is the byte 26, 12 the byte 24.
    pages = slice(chunk.dictionary_page_offset, chunk.data_page_offset)
    data = bytearray(clean.read_bytes())
    assert bytes(data[pages]).count(b"\x4c\x15\x1a") == 1
    data[pages.start + bytes(data[pages]).index(b"\x4c\x15\x1a") + 2] = 24
    corrupt.write_bytes(bytes(data))
    with pytest.raises(pa.ArrowInvalid, match="Index not in dictionary bounds"):
        pq.read_table(corrupt)

    model = Pipeline([("s", StandardScaler(columns=["x"]))]).fit([clean])
    (part,) = model.transform_parquet([clean], output_dir=tmp_path / "clean-out")
    assert pq.read_table(part).column("c").to_pylist() == table.column("c").to_pylist()
    # Strings inside a fixed-size list are read plain: Arrow's decoder refuses
    # the index, and the column that fails on its own is named.
    problem = (
        "column 'c' cannot be decoded: Invalid: Index not in dictionary bounds"
        if kind.startswith("fixed_size_list")
        else "column 'c' has a dictionary index outside its dictionary: Index 12 out of bounds"
    )
    message = f"{corrupt}: failed to read row group 0: {problem}"
    with pytest.raises(EngineError, match=re.escape(message)):
        model.transform_parquet([corrupt], output_dir=tmp_path / "out")
    assert list((tmp_path / "out").iterdir()) == []
    # No step takes a nested column, so only a step that keeps it reads it at
    # fit; the memmap writers refuse such a column before reading a row.
    with pytest.raises(EngineError, match=re.escape(message)):
        Pipeline([("keep", Select(["c", "x"])), ("s", StandardScaler(columns=["x"]))]).fit([corrupt])
    with pytest.raises(EngineError, match="which cannot be written to a memmap"):
        model.transform_memmap([corrupt], output_dir=tmp_path / "memmap")
    with pytest.raises(EngineError, match="which cannot be written to a memmap"):
        parquet_to_memmap([corrupt], output_dir=tmp_path / "converted")


# REGRESSION: EngineError was raised from a C string, so a message ended at a
# NUL byte in a name, and one holding a byte that is not UTF-8, here a Parquet
# column name, surfaced as UnicodeDecodeError instead.
def test_engine_error_messages_keep_every_byte(tmp_path: Path) -> None:
    pq.write_table(pa.table({"x": [1.0]}), tmp_path / "plain.parquet")
    with pytest.raises(EngineError) as nul:
        Pipeline([("s", Select(["a\x00b"]))]).fit([tmp_path / "plain.parquet"])
    assert str(nul.value) == "s: select column 'a\x00b' not found in input"
    # Written as "zq", then renamed "z\xff" in the footer.
    good, bad = tmp_path / "good.parquet", tmp_path / "bad.parquet"
    pq.write_table(pa.table({"zq": [1.0]}), good, store_schema=False)
    data = good.read_bytes()
    assert data.count(b"zq") == 2
    bad.write_bytes(data.replace(b"zq", b"z\xff"))
    with pytest.raises(EngineError) as invalid:
        Pipeline([("s", Select(["zq"]))]).fit([bad, good])
    assert str(invalid.value) == f"{good}: column 'z\\xff' is missing (present in {bad})"


def test_empty_input_and_standalone_transformer(tmp_path: Path) -> None:
    table = pa.table({"x": [1.0, 2.0, 3.0]})
    pipeline = _fit(tmp_path, table, [("scale", StandardScaler(columns=["x"]))])
    pq.write_table(table.slice(0, 0), tmp_path / "empty.parquet")
    (part,) = pipeline.transform_parquet([tmp_path / "empty.parquet"], output_dir=tmp_path / "empty-out")
    result = pq.read_table(part)
    assert result.num_rows == 0
    assert result.column_names == ["x"]
    assert result.schema.field("x").type == pa.float64()
    standalone = StandardScaler(columns=["x"]).fit([tmp_path / "fit.parquet"])
    source = [tmp_path / "fit.parquet"]
    alone = pq.read_table(standalone.transform_parquet(source, output_dir=tmp_path / "alone"))
    assert alone.equals(pq.read_table(pipeline.transform_parquet(source, output_dir=tmp_path / "piped")))
    direct = standalone.transform_memmap(source, output_dir=tmp_path / "standalone")
    assert json.loads(direct.read_text())["rows"] == 3


def test_adjacent_scalers_preserve_intermediate_error_stage(tmp_path: Path) -> None:
    pipeline = _fit(
        tmp_path,
        pa.table({"x": [-0.5, 0.5]}),
        [
            ("first", StandardScaler(columns=["x"], with_mean=False)),
            ("second", StandardScaler(columns=["x"], with_mean=False)),
        ],
    )
    source = tmp_path / "overflow.parquet"
    pq.write_table(pa.table({"x": [0.0, np.finfo(np.float64).max]}), source)
    with pytest.raises(EngineError, match="second.*not finite"):
        pipeline.transform_parquet([source], output_dir=tmp_path / "out")


def test_concurrent_same_handle_uses_each_input_schema_snapshot(tmp_path: Path) -> None:
    pipeline = _fit(tmp_path, pa.table({"x": [1.0, 2.0, 3.0]}), [("scale", StandardScaler(columns=["x"]))])
    sources = []
    tables = []
    for index in range(8):
        table = pa.table({f"passthrough_{index}": [index] * 9, "x": [1.0, 2.0, 3.0] * 3})
        path = tmp_path / f"input-{index}.parquet"
        pq.write_table(table, path, row_group_size=2)
        sources.append(path)
        tables.append(table)

    def transform(index: int) -> None:
        parts = pipeline.transform_parquet([sources[index]], output_dir=tmp_path / f"output-{index}")
        parquet = pa.concat_tables([pq.read_table(part) for part in parts])
        assert parquet.to_pylist() == [pipeline.transform_dict(row) for row in tables[index].to_pylist()]
        assert parquet.column_names == [f"passthrough_{index}", "x"]

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(transform, range(8)))


def _directory_alias(tmp_path: Path, name: str, alias_kind: str) -> tuple[Path, Path]:
    directory = tmp_path / name
    directory.mkdir()
    if alias_kind == "case":
        alias = tmp_path / name.lower()
        if not alias.exists():
            pytest.skip("filesystem is case-sensitive")
        assert alias.samefile(directory)
    else:
        alias = tmp_path / (name + "-alias")
        alias.symlink_to(directory, target_is_directory=True)
    return directory, alias


@pytest.mark.parametrize("kind", ["parquet", "memmap"])
@pytest.mark.parametrize("alias_kind", ["symlink", "case"])
def test_same_output_directory_aliases_publish_one_complete_run(
    tmp_path: Path,
    kind: str,
    alias_kind: str,
) -> None:
    directory, alias = _directory_alias(tmp_path, "Output", alias_kind)
    pipeline = _fit(tmp_path, pa.table({"x": [1.0, 2.0]}), [("scale", StandardScaler(columns=["x"]))])
    artifact = tmp_path / "artifact"
    pipeline.save(artifact)
    other = Pipeline.load(artifact, threads=4)
    first = tmp_path / "first.parquet"
    second = tmp_path / "second.parquet"
    pq.write_table(pa.table({"x": [1.0] * 13}), first, row_group_size=2)
    pq.write_table(pa.table({"x": [2.0] * 5}), second, row_group_size=2)
    with ThreadPoolExecutor(max_workers=2) as executor:
        if kind == "parquet":
            a = executor.submit(pipeline.transform_parquet, [first], output_dir=directory)
            b = executor.submit(other.transform_parquet, [second], output_dir=alias)
        else:
            a = executor.submit(pipeline.transform_memmap, [first], output_dir=directory)
            b = executor.submit(other.transform_memmap, [second], output_dir=alias)
        a.result()
        b.result()
    assert not list(directory.glob("*.tmp"))
    if kind == "parquet":
        parts = sorted(directory.glob("part-*.parquet"))
        values = pa.concat_tables([pq.read_table(path) for path in parts])["x"].to_pylist()
        assert values in ([-1.0] * 13, [1.0] * 5)
        assert len(parts) == (7 if len(values) == 13 else 3)
    else:
        index = json.loads((directory / "index.json").read_text())
        values = np.fromfile(directory / "data.mmap", dtype=np.float64).tolist()
        assert values in ([-1.0] * 13, [1.0] * 5)
        assert index["rows"] == len(values)


@pytest.mark.parametrize("alias_kind", ["symlink", "case"])
def test_concurrent_artifact_saves_publish_one_complete_manifest(tmp_path: Path, alias_kind: str) -> None:
    directory, alias = _directory_alias(tmp_path, "Artifact", alias_kind)
    pipelines = []
    expected = []
    for prefix in ("a", "b"):
        source = tmp_path / f"{prefix}.parquet"
        pq.write_table(pa.table({"key": [f"{prefix}-{i:05d}" for i in range(2048)]}), source)
        pipeline = Pipeline([("encode", LabelEncoder(columns=["key"]))]).fit([source], threads=1)
        assert pipeline._handle is not None
        expected.append(pipeline._handle.manifest_json())
        pipelines.append(pipeline)
    start = Barrier(2)

    def save(index: int) -> None:
        destination = directory if index == 0 else alias
        try:
            for _ in range(8):
                start.wait()
                pipelines[index].save(destination)
        except BaseException:
            start.abort()
            raise

    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(save, range(2)))
    assert (directory / "manifest.json").read_text() in expected
    assert not (directory / "manifest.json.tmp").exists()
    restored = Pipeline.load(directory)
    encoder = restored["encode"]
    assert isinstance(encoder, LabelEncoder)
    assert encoder.vocab_size("key") == 2049
    assert restored.transform_dict({"key": "a-00000"}) in ({"key": 0}, {"key": 1})


@pytest.fixture(scope="module")
def large_keys(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Write 200,000 distinct keys, which take tens of milliseconds to fit or encode.

    Args:
        tmp_path_factory: pytest's session-wide temporary directory factory.

    Returns:
        The Parquet file.

    """
    source = tmp_path_factory.mktemp("large") / "keys.parquet"
    pq.write_table(pa.table({"key": [f"key-{i:07d}" for i in range(200_000)]}), source)
    return source


@pytest.fixture(scope="module")
def large_artifact(large_keys: Path) -> Path:
    """Save a pipeline whose artifact takes tens of milliseconds to write or parse.

    Args:
        large_keys: The keys the pipeline is fitted on.

    Returns:
        The artifact directory.

    """
    Pipeline([("encode", LabelEncoder(columns=["key"]))]).fit([large_keys]).save(large_keys.parent / "artifact")
    return large_keys.parent / "artifact"


@pytest.fixture(scope="module")
def large_numbers(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Write 2,000,000 i64 zeros, a file of about 9 kB whose memmap holds 16 MB.

    Args:
        tmp_path_factory: pytest's session-wide temporary directory factory.

    Returns:
        The Parquet file.

    """
    source = tmp_path_factory.mktemp("numbers") / "numbers.parquet"
    pq.write_table(pa.table({"code": np.zeros(2_000_000, dtype=np.int64)}), source)
    return source


def _longest_pause_during(call: Callable[[], object]) -> tuple[float, float]:
    """Run a call while another thread wakes every millisecond, and find the longest time that thread slept in it.

    Holding the GIL for the whole call keeps the other thread asleep for all of
    it; releasing it lets that thread wake about every millisecond.

    Args:
        call: The call to measure.

    Returns:
        The longest stretch of the call in which the other thread did not wake, and the call's duration, both in
        milliseconds.

    """
    wakeups: list[float] = []
    stop = threading.Event()

    def tick() -> None:
        while not stop.is_set():
            time.sleep(0.001)
            wakeups.append(time.perf_counter())

    ticker = threading.Thread(target=tick)
    ticker.start()
    try:
        time.sleep(0.02)  # let the ticker reach its loop
        start = time.perf_counter()
        call()
        end = time.perf_counter()
    finally:
        stop.set()
        ticker.join()
    inside: list[float] = [start, *(wakeup for wakeup in wakeups if start < wakeup < end), end]
    return max(later - earlier for earlier, later in itertools.pairwise(inside)) * 1e3, (end - start) * 1e3


@pytest.mark.parametrize("operation", ["save", "load", "pickle.dumps", "pickle.loads", "stages_state"])
def test_artifact_calls_let_other_threads_run(large_artifact: Path, tmp_path: Path, operation: str) -> None:
    # On the standard build these calls stopped every other thread for their
    # whole duration: 1.2 s for a 2,000,000-category save. A free-threaded
    # build passes trivially; the Standard-CPython CI job is where it can fail.
    pipeline = Pipeline.load(large_artifact)  # fresh: no manifest cached on its handle yet
    payload = pickle.dumps(Pipeline.load(large_artifact))
    calls: dict[str, Callable[[], object]] = {
        "save": lambda: pipeline.save(tmp_path / "copy"),
        "load": lambda: Pipeline.load(large_artifact),
        "pickle.dumps": lambda: pickle.dumps(pipeline),
        "pickle.loads": lambda: pickle.loads(payload).transform_dict({"key": "key-0000001"}),
        "stages_state": pipeline.stages_state,
    }
    longest, duration = _longest_pause_during(calls[operation])
    # Holding the GIL keeps the other thread asleep for the whole call.
    # Releasing it lets that thread wake every 1.3 ms at most on an idle
    # machine (measured on an Apple M4). A loaded machine stretches the
    # sleeps: a macOS release runner woke only 22 times in a 622 ms save, one
    # per 28 ms. So the bound is half the call, which a call that holds the
    # GIL cannot meet however fast or slow the machine is.
    assert longest < duration / 2, f"{operation}: another thread slept {longest:.0f} ms of a {duration:.0f} ms call"


@pytest.mark.parametrize(
    "operation", ["fit", "transform_parquet", "transform_memmap", "parquet_to_memmap", "stage_state"]
)
def test_batch_calls_let_other_threads_run(
    large_keys: Path, large_artifact: Path, large_numbers: Path, tmp_path: Path, operation: str
) -> None:
    # The check and the bound of the test above, for the calls that read or
    # write data. A free-threaded build passes trivially here too, unless
    # PYTHON_GIL=1 turns its GIL on. Each call reads its input three times, on
    # one thread so that more cores do not shorten it: 37 to 52 ms, against 36
    # to 44 ms for save and stage_state (Apple M4). One read takes 16 to 29 ms,
    # and a late wakeup on a loaded machine is a larger share of a shorter
    # call. stage_state is measured on the handle: a fitted attribute also
    # parses the state with json.loads, which holds the GIL, 27 ms against
    # stage_state's 38 ms for these 200,000 keys (Apple M4), too close to the
    # bound.
    pipeline = Pipeline.load(large_artifact, threads=1)
    handle = pipeline._handle
    assert handle is not None
    keys = [large_keys] * 3
    numbers = [large_numbers] * 3
    calls: dict[str, Callable[[], object]] = {
        "fit": lambda: Pipeline([("encode", LabelEncoder(columns=["key"]))]).fit(keys, threads=1),
        "transform_parquet": lambda: pipeline.transform_parquet(keys, output_dir=tmp_path / "parquet"),
        "transform_memmap": lambda: pipeline.transform_memmap(keys, output_dir=tmp_path / "memmap"),
        "parquet_to_memmap": lambda: parquet_to_memmap(numbers, output_dir=tmp_path / "memmap", threads=1),
        "stage_state": lambda: handle.stage_state("encode"),
    }
    longest, duration = _longest_pause_during(calls[operation])
    assert longest < duration / 2, f"{operation}: another thread slept {longest:.0f} ms of a {duration:.0f} ms call"


@pytest.mark.parametrize("kind", ["parquet", "memmap", "convert"])
def test_case_alias_output_directory_cannot_overwrite_its_input(tmp_path: Path, kind: str) -> None:
    directory, alias = _directory_alias(tmp_path, "Output", "case")
    table = pa.table({"x": [1.0, 2.0]})
    pipeline = _fit(tmp_path, table, [("scale", StandardScaler(columns=["x"]))])
    source = directory / "part-00000.parquet"
    pq.write_table(table, source)
    before = source.read_bytes()
    transform = {
        "parquet": pipeline.transform_parquet,
        "memmap": pipeline.transform_memmap,
        "convert": parquet_to_memmap,
    }[kind]
    with pytest.raises(EngineError, match="inside the output directory"):
        transform([source], output_dir=alias)
    assert source.read_bytes() == before
