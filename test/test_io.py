"""Tests for the Parquet source and sink: empty row groups, page encodings, projection, output contract."""

import base64
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
from fast_feature_transform.transformers import GroupAgg, HashingEncoder, LabelEncoder, Select, StandardScaler


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
    empty: str = _write(tmp_path / "empty.parquet", schema.empty_table())
    row_groups: int = pq.ParquetFile(holes).num_row_groups + pq.ParquetFile(empty).num_row_groups

    encoder: LabelEncoder = LabelEncoder(columns=["device"]).fit([str(holes), empty])
    assert encoder.label_maps_["device"] == {"a": 1, "b": 2, "c": 3}
    parts: list[Path] = encoder.transform_parquet([str(holes), empty], output_dir=tmp_path / "out")
    assert len(parts) == row_groups
    table: pa.Table = pa.concat_tables([pq.read_table(part) for part in parts])
    assert table.column("device").to_pylist() == [1, 2, 3, 1]


@pytest.mark.parametrize("encoding", ["DELTA_BYTE_ARRAY", "DELTA_LENGTH_BYTE_ARRAY", "PLAIN"])
def test_string_page_encodings_are_all_readable(tmp_path: Path, encoding: str) -> None:
    path: str = _write(
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
    path = _write(tmp_path / "delta.parquet", table, use_dictionary=False, column_encoding={"key": "DELTA_BYTE_ARRAY"})
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
    big: str = _write(tmp_path / "big.parquet", pa.table({"device": ["a", "b", "a", "b"]}), row_group_size=2)
    small: str = _write(tmp_path / "small.parquet", pa.table({"device": ["a"]}))
    encoder: LabelEncoder = LabelEncoder(columns=["device"]).fit([big])
    out: Path = tmp_path / "out"
    assert len(encoder.transform_parquet([big], output_dir=out)) == 2
    parts: list[Path] = encoder.transform_parquet([small], output_dir=out)
    assert sorted(os.listdir(out)) == [part.name for part in parts] == ["part-00000.parquet"]
    assert pq.read_table(str(out)).num_rows == 1


def test_transform_parquet_refuses_a_source_inside_the_output_dir(tmp_path: Path) -> None:
    source: str = _write(tmp_path / "part-00000.parquet", pa.table({"device": ["a", "b"]}))
    encoder: LabelEncoder = LabelEncoder(columns=["device"]).fit([source])
    with pytest.raises(Exception, match="inside the output directory"):
        encoder.transform_parquet([source], output_dir=tmp_path)
    assert pq.read_table(source).column("device").to_pylist() == ["a", "b"]


def test_schema_mismatch_at_transform_fails_before_touching_the_output_dir(tmp_path: Path) -> None:
    good: str = _write(tmp_path / "good.parquet", pa.table({"device": ["a", "b"]}))
    bad: str = _write(tmp_path / "bad.parquet", pa.table({"device": [1, 2]}))
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
    source: str = _write(tmp_path / "src.parquet", table)
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
    source: str = _write(tmp_path / "numbers.parquet", table)
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
    source: str = _write(tmp_path / "mixed.parquet", table, **options)
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
    source: str = _write(tmp_path / "words.parquet", table)
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


def test_a_write_the_disk_refuses_fails_the_transform_and_leaves_no_part(tmp_path: Path) -> None:
    # The sink writes each part through a 64 KiB buffer, so a part smaller than that reaches the disk only when it is
    # closed. A file size limit of 16 bytes stages a disk that refuses the write: the transform raises, naming the
    # part, and the output directory holds neither the part nor its temporary file. Python ignores the signal that
    # comes with the refusal.
    source: str = _write(tmp_path / "in.parquet", pa.table({"x": [1.0, 2.0, 3.0]}))
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


def test_schema_mismatch_messages_name_the_column(tmp_path: Path) -> None:
    first: str = _write(tmp_path / "a.parquet", pa.table({"device": ["a"], "price": [1.0]}))
    fewer: str = _write(tmp_path / "b.parquet", pa.table({"device": ["a"]}))
    reordered: str = _write(tmp_path / "c.parquet", pa.table({"price": [1.0], "device": ["a"]}))
    with pytest.raises(Exception, match="'price' is missing"):
        Select(columns=["device"]).fit([first, fewer])
    with pytest.raises(Exception, match="same order"):
        Select(columns=["device"]).fit([first, reordered])


def test_duplicate_column_names_are_rejected(tmp_path: Path) -> None:
    table: pa.Table = pa.Table.from_arrays([pa.array(["a"]), pa.array(["b"])], names=["device", "device"])
    path: str = _write(tmp_path / "dup.parquet", table)
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
    path: str = _write(tmp_path / "wide.parquet", table)
    encoded: list[Path] = (
        LabelEncoder(columns=["device"]).fit([path]).transform_parquet([path], output_dir=tmp_path / "all")
    )
    assert pq.read_table(encoded[0]).column_names == ["device", "tags", "price"]
    selected: list[Path] = (
        Pipeline([("s", Select(columns=["price"]))]).fit([path]).transform_parquet([path], output_dir=tmp_path / "some")
    )
    assert pq.read_table(selected[0]).column_names == ["price"]
