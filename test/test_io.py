"""Tests for the Parquet source and sink: empty row groups, page encodings, projection, output contract."""

import base64
import json
import os
import resource
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pyarrow.parquet import encryption

from fast_feature_transform import EngineError
from fast_feature_transform.pipeline import Pipeline
from fast_feature_transform.transformers import (
    GroupAgg,
    HashingEncoder,
    LabelEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    Select,
    StandardScaler,
)
from helpers import write_parquet

# A full batch of a narrow output, which the sink writes as one row group.
_BATCH_ROWS: int = 65_536


def _distinct_integers(count: int, seed: int) -> np.ndarray:
    """Draw distinct int64 values from the whole range, which Snappy cannot shrink.

    Args:
        count: How many values to draw.
        seed: Seed of the generator.

    Returns:
        The values, 8 bytes each in a Parquet page.

    """
    values: np.ndarray = np.random.default_rng(seed).integers(-(2**62), 2**62, count, dtype=np.int64)
    assert len(np.unique(values)) == count
    return values


# The characters of _distinct_words: with 64 of them, words drawn at random repeat too little for Snappy to shorten.
_ALPHABET: np.ndarray = np.frombuffer(b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_", dtype="S1")


def _distinct_words(count: int, seed: int) -> np.ndarray:
    """Draw distinct strings of 16 random characters, in which Snappy finds nothing to shorten.

    Args:
        count: How many strings to draw.
        seed: Seed of the generator.

    Returns:
        The strings. A Parquet page stores each as a 4-byte length and its 16 characters.

    """
    letters: np.ndarray = np.random.default_rng(seed).choice(_ALPHABET, size=(count, 16))
    words: np.ndarray = letters.view("S16").ravel().astype(str)
    assert len(np.unique(words)) == count
    return words


def _write_strings(path: Path, table: pa.Table, column: str, storage: str) -> str:
    """Write one row group whose string column is stored in a given way, and check that it is.

    Args:
        path: Destination file.
        table: Table to write.
        column: The string column the storage applies to.
        storage: "no dictionary" stores every value in plain pages. "dictionary given up" makes the writer check its
            dictionary page after every 64 values and drop the dictionary once the page holds 4 KiB, so the values
            after that are in plain pages. "dictionary" keeps every value in the dictionary page.

    Returns:
        The file path.

    """
    options: dict[str, Any] = {
        "no dictionary": {"use_dictionary": False},
        "dictionary given up": {"dictionary_pagesize_limit": 4 * 1024, "write_batch_size": 64},
        "dictionary": {"dictionary_pagesize_limit": 64 * 1024 * 1024},
    }[storage]
    source: str = write_parquet(path, table, row_group_size=table.num_rows, **options)
    assert pq.ParquetFile(source).metadata.num_row_groups == 1
    chunk = _first_chunk(Path(source), column)
    dictionary, data = _stored_bytes(chunk)
    if storage == "no dictionary":
        assert not chunk.has_dictionary_page
    elif storage == "dictionary given up":
        assert 0 < dictionary < data
    else:
        assert dictionary > data
    return source


def _first_chunk(part: Path, column: str) -> Any:
    """Return the metadata of a column's chunk in a part's first row group.

    Args:
        part: The Parquet file to describe.
        column: The column's name.

    Returns:
        pyarrow's metadata of that column chunk.

    """
    group = pq.ParquetFile(part).metadata.row_group(0)
    return next(group.column(i) for i in range(group.num_columns) if group.column(i).path_in_schema == column)


def _stored_bytes(chunk: Any) -> tuple[int, int]:
    """Split a column chunk's stored size into its dictionary page and its data pages.

    Args:
        chunk: pyarrow's metadata of the column chunk.

    Returns:
        The bytes of the dictionary page and the bytes of the data pages, as compressed in the file with their
        headers. The dictionary page comes first, so it ends where the first data page starts.

    """
    dictionary: int = chunk.data_page_offset - chunk.dictionary_page_offset if chunk.has_dictionary_page else 0
    return dictionary, chunk.total_compressed_size - dictionary


def test_empty_row_groups_and_empty_files_are_tolerated(tmp_path: Path) -> None:
    schema: pa.Schema = pa.schema([("device", pa.string()), ("price", pa.float64())])
    holes: Path = tmp_path / "holes.parquet"
    with pq.ParquetWriter(holes, schema) as writer:
        writer.write_table(pa.table({"device": ["a", "b"], "price": [1.0, 2.0]}, schema=schema))
        writer.write_table(schema.empty_table())
        writer.write_table(pa.table({"device": ["c", "a"], "price": [3.0, 4.0]}, schema=schema))
    empty: str = write_parquet(tmp_path / "empty.parquet", schema.empty_table())
    row_groups: int = pq.ParquetFile(holes).num_row_groups + pq.ParquetFile(empty).num_row_groups

    encoder: LabelEncoder = LabelEncoder(columns=["device"]).fit([str(holes), empty])
    assert encoder.label_maps_["device"] == {"a": 1, "b": 2, "c": 3}
    parts: list[Path] = encoder.transform_parquet([str(holes), empty], output_dir=tmp_path / "out")
    assert len(parts) == row_groups
    table: pa.Table = pa.concat_tables([pq.read_table(part) for part in parts])
    assert table.column("device").to_pylist() == [1, 2, 3, 1]


@pytest.mark.parametrize("encoding", ["DELTA_BYTE_ARRAY", "DELTA_LENGTH_BYTE_ARRAY", "PLAIN"])
def test_string_page_encodings_are_all_readable(tmp_path: Path, encoding: str) -> None:
    path: str = write_parquet(
        tmp_path / "encoded.parquet",
        pa.table({"device": ["a", "b", "a"]}),
        use_dictionary=False,
        column_encoding={"device": encoding},
    )
    encoder: LabelEncoder = LabelEncoder(columns=["device"]).fit([path])
    assert encoder.label_maps_["device"] == {"a": 1, "b": 2}
    parts: list[Path] = encoder.transform_parquet([path], output_dir=tmp_path / "out")
    assert pq.read_table(parts[0]).column("device").to_pylist() == [1, 2, 1]


def test_delta_encoded_strings_take_the_plain_paths_and_match_records(tmp_path: Path) -> None:
    # The engine reads string columns as dictionaries unless their pages are delta-encoded, as some writers make
    # them; those arrive as plain strings. HashingEncoder then remembers each value's bucket, up to a limit that 300
    # distinct values in 1,000 rows exceed, and GroupAgg looks plain keys up directly. Both must answer as records do.
    values = [None if row % 17 == 0 else f"v{row * 7 % 300}" for row in range(1000)]
    table = pa.table({"key": values, "amount": [float(row % 23) for row in range(1000)]})
    path = write_parquet(
        tmp_path / "delta.parquet", table, use_dictionary=False, column_encoding={"key": "DELTA_BYTE_ARRAY"}
    )
    assert "DELTA_BYTE_ARRAY" in pq.ParquetFile(path).metadata.row_group(0).column(0).encodings
    for step in (
        HashingEncoder(columns=["key"], n_components=16),
        GroupAgg(key="key", aggs={"amount": ["sum", "mean"]}),
    ):
        model = step.fit([path])
        expected = [model.transform_dict(row) for row in table.to_pylist()]
        parts = model.transform_parquet([path], output_dir=tmp_path / type(step).__name__)
        assert pq.read_table(parts).to_pylist() == expected, type(step).__name__


def test_transform_parquet_replaces_stale_parts(tmp_path: Path) -> None:
    big: str = write_parquet(tmp_path / "big.parquet", pa.table({"device": ["a", "b", "a", "b"]}), row_group_size=2)
    small: str = write_parquet(tmp_path / "small.parquet", pa.table({"device": ["a"]}))
    encoder: LabelEncoder = LabelEncoder(columns=["device"]).fit([big])
    out: Path = tmp_path / "out"
    assert len(encoder.transform_parquet([big], output_dir=out)) == 2
    parts: list[Path] = encoder.transform_parquet([small], output_dir=out)
    assert sorted(os.listdir(out)) == [part.name for part in parts] == ["part-00000.parquet"]
    assert pq.read_table(str(out)).num_rows == 1


def test_transform_parquet_refuses_a_source_inside_the_output_dir(tmp_path: Path) -> None:
    source: str = write_parquet(tmp_path / "part-00000.parquet", pa.table({"device": ["a", "b"]}))
    encoder: LabelEncoder = LabelEncoder(columns=["device"]).fit([source])
    with pytest.raises(Exception, match="inside the output directory"):
        encoder.transform_parquet([source], output_dir=tmp_path)
    assert pq.read_table(source).column("device").to_pylist() == ["a", "b"]


def test_schema_mismatch_at_transform_fails_before_touching_the_output_dir(tmp_path: Path) -> None:
    good: str = write_parquet(tmp_path / "good.parquet", pa.table({"device": ["a", "b"]}))
    bad: str = write_parquet(tmp_path / "bad.parquet", pa.table({"device": [1, 2]}))
    encoder: LabelEncoder = LabelEncoder(columns=["device"]).fit([good])
    out: Path = tmp_path / "out"
    with pytest.raises(Exception, match="expected utf8"):
        encoder.transform_parquet([bad], output_dir=out)
    assert os.listdir(out) == []
    # A failure before the run starts leaves an earlier run's output alone.
    previous: list[Path] = encoder.transform_parquet([good], output_dir=out)
    with pytest.raises(Exception, match="expected utf8"):
        encoder.transform_parquet([bad], output_dir=out)
    assert sorted(os.listdir(out)) == [part.name for part in previous]


def test_output_parts_are_compressed_and_floats_skip_dictionary_encoding(tmp_path: Path) -> None:
    table: pa.Table = pa.table(
        {
            "device": ["a", "b"],
            "price": [1.0, 2.0],
            "emb": pa.array([[1.0, 2.0], [3.0, 4.0]], pa.list_(pa.float32())),
            "pair": pa.array([{"x": 1.5, "n": 1}, {"x": 2.5, "n": 2}]),
            "half": pa.array([[1.0], [2.0]], pa.list_(pa.float16(), 1)),
        }
    )
    source: str = write_parquet(tmp_path / "src.parquet", table)
    parts: list[Path] = (
        Select(columns=table.column_names).fit([source]).transform_parquet([source], output_dir=tmp_path / "out")
    )
    metadata = pq.ParquetFile(parts[0]).metadata
    row_group = metadata.row_group(0)
    leaves: dict[str, Any] = {
        row_group.column(i).path_in_schema: row_group.column(i) for i in range(row_group.num_columns)
    }
    assert list(leaves) == ["device", "price", "emb.list.element", "pair.x", "pair.n", "half.list.element"]
    assert {leaf.compression for leaf in leaves.values()} == {"SNAPPY"}
    # Floats skip the dictionary at any depth; other values keep it.
    dictionary = {"PLAIN_DICTIONARY", "RLE_DICTIONARY"}
    assert {name for name, leaf in leaves.items() if dictionary & set(leaf.encodings)} == {"device", "pair.n"}


def _row_group_rows(part: Path) -> list[int]:
    """List the row counts of a part's row groups, in order.

    Args:
        part: The Parquet file to describe.

    Returns:
        One row count per row group.

    """
    metadata = pq.ParquetFile(part).metadata
    return [metadata.row_group(group).num_rows for group in range(metadata.num_row_groups)]


@pytest.mark.parametrize(
    ("columns", "rows", "row_groups"),
    [
        (300, 6_000, [2 * 2_330, 1_340]),  # a batch of 2,330 rows: two reach 4,096
        (900, 5_000, [6 * 776, 344]),  # a batch of 776 rows: six of them
    ],
)
def test_a_wide_output_gathers_its_small_batches_into_row_groups(
    tmp_path: Path, columns: int, rows: int, row_groups: list[int]
) -> None:
    # REGRESSION: a batch of fewer than 4,096 rows shares a row group with the
    # ones after it, until the row group holds 4,096 rows or six batches. Each
    # batch used to be a row group, and a batch holds fewer rows the wider the
    # output is (5,592,405 bytes at 8 bytes a number), so a very wide output
    # was hundreds of row groups that held little but their own headers.
    values: np.ndarray = np.arange(rows * columns, dtype=np.float64).reshape(rows, columns)
    table: pa.Table = pa.table({f"x{i}": values[:, i] for i in range(columns)})
    source: str = write_parquet(tmp_path / "wide.parquet", table, row_group_size=rows)
    parts: list[Path] = (
        Select(columns=table.column_names).fit([source]).transform_parquet([source], output_dir=tmp_path / "wide")
    )
    assert _row_group_rows(parts[0]) == row_groups
    assert pq.read_table(parts).equals(table)


def test_strings_read_from_plain_pages_share_one_dictionary_in_a_row_group(tmp_path: Path) -> None:
    # A string column stored in plain pages is read with a dictionary that
    # grows from batch to batch, and Arrow's writer keeps a row group's
    # dictionary only while every batch brings an equal one. Each word here
    # fills 1,000 rows, so every batch brings words the batch before it did
    # not have. The sink gives the two batches of a row group one dictionary,
    # and the column chunk of the first row group is 110 bytes: five words
    # and runs of one index. Without that it is 25,399 bytes, the 2,299 rows
    # of the second batch written as words.
    rows, columns = 6_000, 300
    words: np.ndarray = np.array([f"word-{i:02d}" for i in range(20)])
    numbers: np.ndarray = np.arange(rows * columns, dtype=np.float64).reshape(rows, columns)
    table: pa.Table = pa.table(
        {"s": words[np.arange(rows) // 1_000], **{f"x{i}": numbers[:, i] for i in range(columns)}}
    )
    source: str = write_parquet(tmp_path / "plain.parquet", table, row_group_size=rows, use_dictionary=False)
    parts: list[Path] = (
        Select(columns=table.column_names).fit([source]).transform_parquet([source], output_dir=tmp_path / "out")
    )
    assert _row_group_rows(parts[0]) == [2 * 2_299, rows - 2 * 2_299]
    assert _first_chunk(parts[0], "s").total_uncompressed_size < 5_000
    assert pq.read_table(parts).equals(table)


def test_strings_in_a_list_read_from_plain_pages_share_one_dictionary_in_a_row_group(tmp_path: Path) -> None:
    # The strings of a list column are read with a growing dictionary too, one
    # level below the column, and the sink gives the two batches of a row group
    # one dictionary there as well. Each row holds two strings and each word
    # fills 1,000 rows, so the column chunk of the first row group is 1,287
    # bytes. Without the one dictionary it is 51,867 bytes, the 4,598
    # strings of the second batch written as words.
    rows, columns = 6_000, 300
    words: np.ndarray = np.array([f"word-{i:02d}" for i in range(20)])
    numbers: np.ndarray = np.arange(rows * columns, dtype=np.float64).reshape(rows, columns)
    tags: pa.Array = pa.ListArray.from_arrays(
        pa.array(np.arange(0, 2 * rows + 1, 2, dtype=np.int32)), pa.array(words[np.arange(2 * rows) // 2_000])
    )
    table: pa.Table = pa.table({"tags": tags, **{f"x{i}": numbers[:, i] for i in range(columns)}})
    source: str = write_parquet(tmp_path / "plain.parquet", table, row_group_size=rows, use_dictionary=False)
    parts: list[Path] = (
        Select(columns=table.column_names).fit([source]).transform_parquet([source], output_dir=tmp_path / "out")
    )
    assert _row_group_rows(parts[0]) == [2 * 2_299, rows - 2 * 2_299]
    assert _first_chunk(parts[0], "tags.list.element").total_uncompressed_size < 10_000
    assert pq.read_table(parts).equals(table)


@pytest.mark.parametrize(
    ("columns", "row_groups"),
    [
        (2, [_BATCH_ROWS, _BATCH_ROWS, 18_928]),
        (20, [34_952, 34_952, 34_952, 34_952, 10_192]),
        (32, [21_845, 21_845, 21_845, 21_845, 21_845, 21_845, 18_930]),
    ],
)
def test_a_batch_of_4096_rows_or_more_is_a_row_group_of_its_own(
    tmp_path: Path, columns: int, row_groups: list[int]
) -> None:
    rows: int = 150_000
    table: pa.Table = pa.table({f"x{i}": np.arange(rows, dtype=np.float64) + i for i in range(columns)})
    source: str = write_parquet(tmp_path / "source.parquet", table, row_group_size=rows)
    parts: list[Path] = (
        Select(columns=table.column_names).fit([source]).transform_parquet([source], output_dir=tmp_path / "out")
    )
    assert _row_group_rows(parts[0]) == row_groups
    assert pq.read_table(parts).equals(table)


def test_a_numbers_only_output_ends_a_dictionary_of_distinct_integers_at_248_kib(tmp_path: Path) -> None:
    # Arrow's writer hashes an integer column into a dictionary until the dictionary page reaches a limit, 1 MiB by
    # default, which the 65,536 rows of a row group never do. When every output column is a number the sink lowers
    # the limit to 248 KiB, 31,744 values of 8 bytes. `id` is distinct in every row: its dictionary stops there, short
    # of 32,768 entries, and the rows after it are stored whole. `key` holds 30,000 distinct values, 240,000 bytes
    # of dictionary: it stays below the limit and every row is an index of at most two bytes, as under 1 MiB.
    rng = np.random.default_rng(0)
    pool: np.ndarray = _distinct_integers(30_000, 2)
    keys: np.ndarray = np.concatenate([pool, rng.choice(pool, _BATCH_ROWS - len(pool))])
    rng.shuffle(keys)
    table: pa.Table = pa.table(
        {"id": _distinct_integers(_BATCH_ROWS, 1), "key": keys, "x": rng.normal(size=_BATCH_ROWS)}
    )
    source: str = write_parquet(tmp_path / "numbers.parquet", table)
    parts: list[Path] = (
        Select(columns=table.column_names).fit([source]).transform_parquet([source], output_dir=tmp_path / "out")
    )
    assert pq.ParquetFile(parts[0]).metadata.row_group(0).num_rows == _BATCH_ROWS
    assert pq.read_table(parts[0]).equals(table)

    id_dictionary, id_data = _stored_bytes(_first_chunk(parts[0], "id"))
    assert 248 * 1024 <= id_dictionary < 256 * 1024
    assert id_data >= 8 * (_BATCH_ROWS - 32_768)
    key_dictionary, key_data = _stored_bytes(_first_chunk(parts[0], "key"))
    assert 8 * 30_000 <= key_dictionary < 8 * 30_000 + 1024
    assert key_data < 2 * _BATCH_ROWS + 1024


@pytest.mark.parametrize("kind", ["plain string", "decimal", "list"])
def test_an_output_with_a_plain_string_decimal_or_list_column_keeps_arrows_dictionary_limit(
    tmp_path: Path, kind: str
) -> None:
    # The limit is one setting for a whole part, and a dictionary of strings, decimals or list elements can be worth
    # more than 248 KiB, so one such column keeps Arrow's 1 MiB for every column: the distinct `id` beside it has
    # all 65,536 values in its dictionary page and every row is an index.
    small: np.ndarray = np.arange(_BATCH_ROWS) % 7
    options: dict[str, Any] = {}
    if kind == "plain string":
        # Delta-encoded strings reach the sink as plain strings, not as a dictionary.
        other: pa.Array = pa.array([f"s{value}" for value in small])
        options = {"use_dictionary": ["id"], "column_encoding": {"other": "DELTA_BYTE_ARRAY"}}
    elif kind == "decimal":
        other = pa.array(small).cast(pa.decimal128(38, 2))
    else:
        other = pa.ListArray.from_arrays(np.arange(_BATCH_ROWS + 1, dtype=np.int32), pa.array(small))
    table: pa.Table = pa.table({"id": _distinct_integers(_BATCH_ROWS, 3), "other": other})
    source: str = write_parquet(tmp_path / "mixed.parquet", table, **options)
    parts: list[Path] = (
        Select(columns=table.column_names).fit([source]).transform_parquet([source], output_dir=tmp_path / "out")
    )
    assert pq.ParquetFile(parts[0]).metadata.row_group(0).num_rows == _BATCH_ROWS
    assert pq.read_table(parts[0]).equals(table)

    id_dictionary, id_data = _stored_bytes(_first_chunk(parts[0], "id"))
    assert id_dictionary >= 8 * _BATCH_ROWS
    assert id_data < 2 * _BATCH_ROWS + 1024


def test_strings_read_as_a_dictionary_keep_it_past_the_lowered_limit(tmp_path: Path) -> None:
    # A string column stored with a dictionary reaches the sink as that dictionary, so it does not keep the limit at
    # 1 MiB: the distinct `id` beside it stops at 248 KiB. Arrow writes such a column as the dictionary it already
    # is, whatever the limit: 20,000 words of 24 characters, 560,000 bytes with their lengths, stay in the
    # dictionary page, and every row is an index of at most two bytes.
    rng = np.random.default_rng(4)
    words: list[str] = [f"{index:05d}-{value:018d}" for index, value in enumerate(rng.integers(0, 10**18, 20_000))]
    table: pa.Table = pa.table(
        {
            "id": _distinct_integers(_BATCH_ROWS, 5),
            "word": pa.array(words).take(pa.array(np.arange(_BATCH_ROWS) % 20_000)),
        }
    )
    source: str = write_parquet(tmp_path / "words.parquet", table)
    parts: list[Path] = (
        Select(columns=table.column_names).fit([source]).transform_parquet([source], output_dir=tmp_path / "out")
    )
    assert pq.ParquetFile(parts[0]).metadata.row_group(0).num_rows == _BATCH_ROWS
    assert pq.read_table(parts[0]).equals(table)

    id_dictionary, _ = _stored_bytes(_first_chunk(parts[0], "id"))
    assert 248 * 1024 <= id_dictionary < 256 * 1024
    word = _first_chunk(parts[0], "word")
    assert word.total_uncompressed_size > 20_000 * (4 + 24) > 248 * 1024
    assert _stored_bytes(word)[1] < 2 * _BATCH_ROWS + 1024


@pytest.mark.parametrize("storage", ["no dictionary", "dictionary given up"])
def test_strings_stored_in_plain_pages_pass_through_at_about_the_inputs_size(tmp_path: Path, storage: str) -> None:
    # REGRESSION: a batch keeps only the dictionary entries its rows use. A string column is read as a dictionary,
    # and when its pages are plain the reader builds one dictionary for the row group as it reads, so each batch
    # arrives with every entry of the batches before it. The three batches of this row group (65,536 distinct
    # values, 65,536 nulls, 3,928 more distinct values) are written with 65,536, 0 and 3,928 entries; they used to
    # be written with 65,536, 65,536 and 69,464.
    words: np.ndarray = _distinct_words(_BATCH_ROWS + 3_928, 6)
    column: pa.Array = pa.concat_arrays(
        [pa.array(words[:_BATCH_ROWS]), pa.nulls(_BATCH_ROWS, pa.string()), pa.array(words[_BATCH_ROWS:])]
    )
    table: pa.Table = pa.table({"id": column})
    source: str = _write_strings(tmp_path / "ids.parquet", table, "id", storage)
    parts: list[Path] = Select(columns=["id"]).fit([source]).transform_parquet([source], output_dir=tmp_path / "out")
    assert _row_group_rows(parts[0]) == [_BATCH_ROWS, _BATCH_ROWS, 3_928]
    assert pq.read_table(parts[0]).equals(table)
    # The output holds each value once, as the input does, and adds an index of 2 bytes for each of the 69,464 rows
    # that hold one: 1.11 times the input. It used to hold 200,536 values, 3.0 times the input.
    assert parts[0].stat().st_size <= 1.2 * Path(source).stat().st_size


def test_ordinal_and_label_codes_do_not_depend_on_how_a_row_group_stores_its_strings(tmp_path: Path) -> None:
    # 70,000 rows draw from 60,000 words, and one row in fifty is null. In one row group the second batch holds 4,464
    # rows and arrives with the dictionary of the whole row group, about 41,000 entries, whether the reader built it
    # from plain pages or the file stores it: the batch keeps the entries its rows use, and the encoders must code
    # them as they code a batch that arrives with exactly its own dictionary. The last file is that reference: its
    # row groups of 20,000 rows are one batch each. Both columns hold the same strings.
    rows: int = _BATCH_ROWS + 4_464
    rng = np.random.default_rng(8)
    words: np.ndarray = np.sort(_distinct_words(60_000, 7))
    drawn: np.ndarray = rng.integers(0, 60_000, rows)
    nulls: np.ndarray = rng.random(rows) < 0.02
    column: pa.Array = pa.array(words[drawn], mask=nulls)
    table: pa.Table = pa.table({"ordinal": column, "label": column})
    sources: dict[str, str] = {
        storage: _write_strings(tmp_path / f"{index}.parquet", table, "label", storage)
        for index, storage in enumerate(("no dictionary", "dictionary given up", "dictionary"))
    }
    sources["a dictionary per batch"] = write_parquet(tmp_path / "small.parquet", table, row_group_size=20_000)

    # The categories are the words some row holds, in byte order, which is the order of `words`. OrdinalEncoder
    # counts them from 1 and gives a null seen at fit the code after the last; LabelEncoder counts them from 1 and
    # gives a null 0.
    categories: np.ndarray = np.unique(drawn[~nulls])
    rank: np.ndarray = np.searchsorted(categories, drawn) + 1
    expected: dict[str, np.ndarray] = {
        "ordinal": np.where(nulls, len(categories) + 1, rank),
        "label": np.where(nulls, 0, rank),
    }
    for storage, source in sources.items():
        pipeline: Pipeline = Pipeline(
            [("ordinal", OrdinalEncoder(columns=["ordinal"])), ("label", LabelEncoder(columns=["label"]))]
        ).fit([source])
        coded: pa.Table = pq.read_table(pipeline.transform_parquet([source], output_dir=tmp_path / f"out-{storage}"))
        for name, codes in expected.items():
            np.testing.assert_array_equal(coded.column(name).to_numpy(), codes, err_msg=f"{storage}: {name}")
        if storage != "no dictionary":
            continue
        # The memmap sink receives the same batches.
        index_path: Path = pipeline.transform_memmap([source], output_dir=tmp_path / "memmap")
        index: dict[str, Any] = json.loads(index_path.read_text())
        assert [(field["name"], field["dtype"]) for field in index["columns"]] == [
            ("ordinal", "int64"),
            ("label", "int64"),
        ]
        records: np.ndarray = np.memmap(
            index_path.parent / index["file"], dtype=[("ordinal", "int64"), ("label", "int64")], mode="r"
        )
        for name, codes in expected.items():
            np.testing.assert_array_equal(records[name], codes, err_msg=f"memmap: {name}")


def test_one_hot_columns_do_not_depend_on_how_a_row_group_stores_its_strings(tmp_path: Path) -> None:
    # A one-hot of 900 categories makes a row 900 columns wide, so a batch holds 776 rows, fewer than the 900
    # entries of the row group's dictionary: every batch but the first of a row group of 1,000 rows keeps only the
    # entries its rows use, and so does the first when the file stores the dictionary. Each of the 900 words is in
    # the first 900 rows, in a shuffled order. The two batches, of 776 and 224 rows, share one row group of the
    # output.
    words: np.ndarray = np.sort(_distinct_words(900, 9))
    rng = np.random.default_rng(10)
    choice: np.ndarray = np.concatenate([rng.permutation(900), rng.integers(0, 900, 100)])
    table: pa.Table = pa.table({"c": words[choice]})
    # Column c_k is 1 in the rows that hold the k-th category in byte order.
    expected: np.ndarray = np.zeros((1_000, 900), dtype=np.int64)
    expected[np.arange(1_000), choice] = 1
    for storage in ("no dictionary", "dictionary given up", "dictionary"):
        source: str = _write_strings(tmp_path / f"{storage}.parquet", table, "c", storage)
        encoder: OneHotEncoder = OneHotEncoder(columns=["c"]).fit([source])
        parts: list[Path] = encoder.transform_parquet([source], output_dir=tmp_path / f"out-{storage}")
        assert _row_group_rows(parts[0]) == [1_000]
        hot: pa.Table = pq.read_table(parts[0])
        assert hot.column_names == [f"c_{k}" for k in range(1, 901)]
        np.testing.assert_array_equal(np.column_stack([column.to_numpy() for column in hot.columns]), expected, storage)


def test_a_write_the_disk_refuses_fails_the_transform_and_leaves_no_part(tmp_path: Path) -> None:
    # The sink writes each part through a 64 KiB buffer, so a part smaller than that reaches the disk only when it is
    # closed. A file size limit of 16 bytes stages a disk that refuses the write: the transform raises, naming the
    # part, and the output directory holds neither the part nor its temporary file. Python ignores the signal that
    # comes with the refusal.
    source: str = write_parquet(tmp_path / "in.parquet", pa.table({"x": [1.0, 2.0, 3.0]}))
    scaler: StandardScaler = StandardScaler(columns=["x"]).fit([source])
    out: Path = tmp_path / "out"
    soft, hard = resource.getrlimit(resource.RLIMIT_FSIZE)
    resource.setrlimit(resource.RLIMIT_FSIZE, (16, hard))
    try:
        with pytest.raises(EngineError, match=r"part-00000\.parquet: .*File too large"):
            scaler.transform_parquet([source], output_dir=out)
        left: list[str] = os.listdir(out)
    finally:
        resource.setrlimit(resource.RLIMIT_FSIZE, (soft, hard))
    assert left == []
    parts: list[Path] = scaler.transform_parquet([source], output_dir=out)
    assert pq.read_table(parts[0]).num_rows == 3


@pytest.mark.parametrize("order", [(0, 1), (1, 0), (0, 1, 2)], ids=["dictionary first", "delta first", "three"])
def test_files_that_store_one_nested_string_column_differently_are_read_together(
    tmp_path: Path, order: tuple[int, ...]
) -> None:
    # REGRESSION: a list of strings is one column however a file's writer
    # encoded it. Its type used to record whether the strings were read as a
    # dictionary, which follows from the encoding, so a file with dictionary
    # pages and one with delta pages were refused as holding different types.
    tags: list[list[str] | None] = [["a", "b"], ["c"], [], ["a"], None] * 40
    table: pa.Table = pa.table(
        {
            "tags": pa.array(tags, pa.list_(pa.string())),
            "pair": pa.array([{"name": "n1", "v": 1}, {"name": "n2", "v": 2}] * 100),
            "word": ["x", "y", "z", "x", "y"] * 40,
            "x": np.arange(200, dtype=np.float64),
        }
    )
    delta: dict[str, str] = {
        "tags.list.element": "DELTA_BYTE_ARRAY",
        "pair.name": "DELTA_BYTE_ARRAY",
        "pair.v": "PLAIN",
        "word": "DELTA_BYTE_ARRAY",
        "x": "PLAIN",
    }
    files: list[str] = [
        write_parquet(tmp_path / "dictionary.parquet", table, use_dictionary=True),
        write_parquet(tmp_path / "delta.parquet", table, use_dictionary=False, column_encoding=delta),
        # Dictionary pages that give way to plain ones partway.
        write_parquet(tmp_path / "fallback.parquet", table, use_dictionary=True, dictionary_pagesize_limit=4),
    ]
    sources: list[str] = [files[i] for i in order]
    pipeline: Pipeline = Pipeline([("scale", StandardScaler(columns=["x"], with_mean=False, with_std=False))])
    parts: list[Path] = pipeline.fit(sources).transform_parquet(sources, output_dir=tmp_path / "out")
    assert len(parts) == len(sources)
    # pyarrow reads the parts back as one table, each column of one type.
    assert pq.read_table(tmp_path / "out").equals(pa.concat_tables([table] * len(sources)))


def test_files_whose_nested_columns_differ_in_type_are_still_refused(tmp_path: Path) -> None:
    strings: str = write_parquet(tmp_path / "strings.parquet", pa.table({"tags": pa.array([["a"], ["b"]])}))
    numbers: str = write_parquet(tmp_path / "numbers.parquet", pa.table({"tags": pa.array([[1], [2]])}))
    large: str = write_parquet(
        tmp_path / "large.parquet", pa.table({"tags": pa.array([["a"], ["b"]], pa.large_list(pa.string()))})
    )
    # The message names the type, without the dictionary the strings are read as.
    message: str = (
        r"numbers\.parquet: column 'tags' is list<element: int64> but .*strings\.parquet has list<element: string>$"
    )
    with pytest.raises(EngineError, match=message):
        Select(columns=["tags"]).fit([strings, numbers])
    with pytest.raises(
        EngineError, match=r"column 'tags' is large_list<element: string> but .* has list<element: string>$"
    ):
        Select(columns=["tags"]).fit([strings, large])


def test_schema_mismatch_messages_name_the_column(tmp_path: Path) -> None:
    first: str = write_parquet(tmp_path / "a.parquet", pa.table({"device": ["a"], "price": [1.0]}))
    fewer: str = write_parquet(tmp_path / "b.parquet", pa.table({"device": ["a"]}))
    reordered: str = write_parquet(tmp_path / "c.parquet", pa.table({"price": [1.0], "device": ["a"]}))
    with pytest.raises(Exception, match="'price' is missing"):
        Select(columns=["device"]).fit([first, fewer])
    with pytest.raises(Exception, match="same order"):
        Select(columns=["device"]).fit([first, reordered])


def test_duplicate_column_names_are_rejected(tmp_path: Path) -> None:
    table: pa.Table = pa.Table.from_arrays([pa.array(["a"]), pa.array(["b"])], names=["device", "device"])
    path: str = write_parquet(tmp_path / "dup.parquet", table)
    with pytest.raises(Exception, match="duplicate column name 'device'"):
        Select(columns=["device"]).fit([path])


def test_missing_and_directory_inputs_are_named(tmp_path: Path) -> None:
    with pytest.raises(Exception, match="no such file"):
        Select(columns=["device"]).fit([str(tmp_path / "nope.parquet")])
    with pytest.raises(Exception, match="is a directory"):
        Select(columns=["device"]).fit([str(tmp_path)])


class _PlainKms(encryption.KmsClient):
    """A key client that keeps keys in the file itself, which is all a test needs."""

    def __init__(self, config: encryption.KmsConnectionConfig) -> None:
        super().__init__()

    def wrap_key(self, key_bytes: bytes, master_key_identifier: str) -> bytes:
        return base64.b64encode(key_bytes)

    def unwrap_key(self, wrapped_key: bytes, master_key_identifier: str) -> bytes:
        return base64.b64decode(wrapped_key)


def test_encrypted_files_are_refused_with_the_file_named(tmp_path: Path) -> None:
    table: pa.Table = pa.table({"device": ["a", "b", "a"], "price": [1.0, 2.0, 3.0]})
    factory = encryption.CryptoFactory(_PlainKms)
    paths: dict[str, str] = {}
    for name, plaintext_footer in (("footer", False), ("column", True)):
        configuration = encryption.EncryptionConfiguration(
            footer_key="footer",
            column_keys={"key": ["device"]},
            plaintext_footer=plaintext_footer,
            double_wrapping=False,
        )
        properties = factory.file_encryption_properties(encryption.KmsConnectionConfig(), configuration)
        paths[name] = str(tmp_path / f"{name}.parquet")
        with pq.ParquetWriter(paths[name], table.schema, encryption_properties=properties) as writer:
            writer.write_table(table)

    with pytest.raises(EngineError, match=r"footer\.parquet: is a Parquet file with an encrypted footer"):
        StandardScaler(columns=["price"]).fit([paths["footer"]])
    # With a plaintext footer only the encrypted column is unreadable: a step that reads another column fits, but the
    # column fails whenever it is decoded, when a step reads it or when it passes through a transform.
    scaler = StandardScaler(columns=["price"]).fit([paths["column"]])
    assert scaler.mean_ == {"price": 2.0}
    with pytest.raises(EngineError, match=r"column\.parquet: failed to read row group 0"):
        scaler.transform_parquet([paths["column"]], output_dir=tmp_path / "out")
    with pytest.raises(EngineError, match=r"column\.parquet: failed to read row group 0"):
        LabelEncoder(columns=["device"]).fit([paths["column"]])
    # A file that only starts like an encrypted one is not Parquet at all.
    csv: Path = tmp_path / "parents.csv"
    csv.write_bytes(b"PARENT_ID,CHILD_ID\n1,2\n")
    with pytest.raises(EngineError, match=r"parents\.csv: unrecognized format \(expected Parquet\)"):
        StandardScaler(columns=["price"]).fit([str(csv)])


def test_untouched_columns_pass_through_and_selected_columns_are_projected(tmp_path: Path) -> None:
    table: pa.Table = pa.table({"device": ["a"], "tags": pa.array([[1, 2]], pa.list_(pa.int64())), "price": [1.0]})
    path: str = write_parquet(tmp_path / "wide.parquet", table)
    encoded: list[Path] = (
        LabelEncoder(columns=["device"]).fit([path]).transform_parquet([path], output_dir=tmp_path / "all")
    )
    assert pq.read_table(encoded[0]).column_names == ["device", "tags", "price"]
    selected: list[Path] = (
        Pipeline([("s", Select(columns=["price"]))]).fit([path]).transform_parquet([path], output_dir=tmp_path / "some")
    )
    assert pq.read_table(selected[0]).column_names == ["price"]
