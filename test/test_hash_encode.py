"""Tests for the hash_encode kernel: md5 buckets, the shared block, the empty artifact, and the error paths."""

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fast_feature_transform import Cast, EngineError, HashingEncoder, Pipeline, _core
from helpers import read_rows, write_parquet

COMPONENTS: int = 4


def _bucket(value: str, components: int = COMPONENTS) -> int:
    """The bucket a value falls in, straight from the definition upstream uses.

    Args:
        value: The value being hashed.
        components: The block width.

    Returns:
        The index of the counter that value increments.

    """
    digest: bytes = hashlib.md5(value.encode(), usedforsecurity=False).digest()
    return int.from_bytes(digest, "big") % components


@pytest.fixture
def events(tmp_path: Path) -> list[str]:
    """One file with three row groups, holding repeats and a null in each string column.

    Args:
        tmp_path: pytest-provided temporary directory.

    Returns:
        Paths of the written Parquet files.

    """
    table: pa.Table = pa.table(
        {
            "device": ["ios", "web", "ios", None, "tv", "ios"],
            "source": ["ads", "ads", "seo", "seo", None, "ads"],
            "price": [1.0] * 6,
        }
    )
    return [write_parquet(tmp_path / "events.parquet", table, row_group_size=2)]


def test_hash_encode_replaces_the_column_with_a_block_of_md5_counters(events: list[str], tmp_path: Path) -> None:
    encoder: HashingEncoder = HashingEncoder(columns=["device"], n_components=COMPONENTS).fit(events)
    assert encoder.output_columns == ["col_0", "col_1", "col_2", "col_3"]
    parts: list[Path] = encoder.transform_parquet(events, output_dir=tmp_path / "out")
    table: pa.Table = pa.concat_tables([pq.read_table(part) for part in parts])
    # The hashed column is gone; everything else passes through beside the block.
    assert set(table.schema.names) == {"source", "price", "col_0", "col_1", "col_2", "col_3"}
    assert table.schema.field("col_0").type == pa.int64()

    devices: list[str | None] = ["ios", "web", "ios", None, "tv", "ios"]
    sources: list[str | None] = ["ads", "ads", "seo", "seo", None, "ads"]
    for row, device in zip(table.to_pylist(), devices, strict=True):
        counters: list[int] = [row[name] for name in encoder.output_columns]
        assert sum(counters) == (0 if device is None else 1)
        if device is not None:
            assert counters[_bucket(device)] == 1

    # The record path answers exactly what the batch path did.
    for row, device, source in zip(table.to_pylist(), devices, sources, strict=True):
        assert encoder.transform_dict({"device": device, "source": source, "price": 1.0}) == row


def test_hash_encode_feeds_every_column_into_the_one_block(events: list[str], tmp_path: Path) -> None:
    encoder: HashingEncoder = HashingEncoder(columns=["device", "source"], n_components=COMPONENTS).fit(events)
    rows: list[dict[str, Any]] = read_rows(encoder.transform_parquet(events, output_dir=tmp_path / "out"))
    assert set(rows[0]) == {"price", "col_0", "col_1", "col_2", "col_3"}
    # Row 0 is ("ios", "ads"): two values, so two increments in one block.
    assert sum(rows[0][name] for name in encoder.output_columns) == 2
    assert rows[0][f"col_{_bucket('ios')}"] + rows[0][f"col_{_bucket('ads')}"] == 2
    # Row 3 is (None, "seo") and row 4 is ("tv", None): a null counts nowhere.
    assert sum(rows[3][name] for name in encoder.output_columns) == 1
    assert sum(rows[4][name] for name in encoder.output_columns) == 1
    assert encoder.transform_dict({"device": None, "source": None, "price": 1.0}) == {
        "price": 1.0,
        "col_0": 0,
        "col_1": 0,
        "col_2": 0,
        "col_3": 0,
    }


def test_hash_encode_hashes_a_cast_integer_as_its_digits(tmp_path: Path) -> None:
    path: str = write_parquet(tmp_path / "ids.parquet", pa.table({"user_id": [7, 42, 7]}))
    pipeline: Pipeline = Pipeline(
        [("cast", Cast({"user_id": "utf8"})), ("hash", HashingEncoder(columns=["user_id"], n_components=COMPONENTS))]
    ).fit([path])
    rows: list[dict[str, Any]] = read_rows(pipeline.transform_parquet([path], output_dir=tmp_path / "out"))
    assert rows[0][f"col_{_bucket('7')}"] == 1  # str(7), which is what category_encoders hashes
    assert rows[1][f"col_{_bucket('42')}"] == 1


def test_hash_encode_is_independent_of_the_split_and_survives_the_artifact(events: list[str], tmp_path: Path) -> None:
    outputs: set[str] = set()
    states: set[str] = set()
    for threads in (1, 4):
        encoder: HashingEncoder = HashingEncoder(columns=["device"], n_components=COMPONENTS).fit(
            events, threads=threads
        )
        states.add(encoder._require_handle().stage_state("hashingencoder"))
        outputs.add(json.dumps(read_rows(encoder.transform_parquet(events, output_dir=tmp_path / f"out{threads}"))))
    assert states == {"{}"}  # nothing is learned, so there is nothing a split could change
    assert len(outputs) == 1

    pipeline: Pipeline = Pipeline([("hash", HashingEncoder(columns=["device"], n_components=COMPONENTS))]).fit(events)
    pipeline.save(tmp_path / "artifact")
    manifest: dict[str, Any] = json.loads((tmp_path / "artifact" / "manifest.json").read_text())
    assert manifest["ops"][0]["state"] == {}
    assert manifest["ops"][0]["params"] == {"columns": ["device"], "n_components": 4, "hash_method": "md5"}
    record: dict[str, Any] = {"device": "ios", "price": 1.0}
    assert Pipeline.load(tmp_path / "artifact").transform_dict(record) == pipeline.transform_dict(record)
    assert pickle.loads(pickle.dumps(pipeline)).transform_dict(record) == pipeline.transform_dict(record)

    manifest["ops"][0]["state"] = {"counts": {"device": 1}}
    with pytest.raises(EngineError, match="hash_encode learns nothing"):
        _core.FittedPipeline.load_json(json.dumps(manifest))


def test_hash_encode_refuses_columns_it_cannot_hash_and_a_second_block(events: list[str], tmp_path: Path) -> None:
    with pytest.raises(EngineError, match="column 'price' is f64, expected utf8"):
        HashingEncoder(columns=["price"]).fit(events)
    with pytest.raises(EngineError, match="column 'nobody' not found in input"):
        HashingEncoder(columns=["nobody"]).fit(events)
    two: Pipeline = Pipeline([("a", HashingEncoder(columns=["device"])), ("b", HashingEncoder(columns=["source"]))])
    with pytest.raises(EngineError, match="two hash encoders cannot share one pipeline"):
        two.fit(events)
    # A column named like a bucket is fine as long as it is the hashed one.
    path: str = write_parquet(tmp_path / "named.parquet", pa.table({"col_0": ["ios", "web"]}))
    assert HashingEncoder(columns=["col_0"], n_components=1).fit([path]).transform_dict({"col_0": "ios"}) == {
        "col_0": 1
    }


def test_hash_encode_validates_its_arguments() -> None:
    bare_string: Any = "device"
    text: Any = "8"
    with pytest.raises(TypeError, match="not a single string"):
        HashingEncoder(columns=bare_string)
    with pytest.raises(ValueError, match="at least one column"):
        HashingEncoder(columns=[])
    with pytest.raises(TypeError, match="n_components must be an integer"):
        HashingEncoder(columns=["device"], n_components=text)
    with pytest.raises(ValueError, match="n_components must be between 1 and 4096"):
        HashingEncoder(columns=["device"], n_components=0)
    with pytest.raises(ValueError, match="n_components must be between 1 and 4096"):
        HashingEncoder(columns=["device"], n_components=4097)
    with pytest.raises(ValueError, match="hash_method must be 'md5'"):
        HashingEncoder(columns=["device"], hash_method="sha256")
