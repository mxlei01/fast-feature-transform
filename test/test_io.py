"""Tests for the Parquet source and sink: empty row groups, page encodings, projection, output contract."""

import base64
import os
from pathlib import Path
from typing import Any

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
