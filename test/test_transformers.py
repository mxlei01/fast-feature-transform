"""Tests for the LabelEncoder and Select transformers and their C++ kernels."""

import pickle
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fast_feature_transform.pipeline import Pipeline
from fast_feature_transform.transformers import LabelEncoder, Select

DEVICE_VALUES: list[str | None] = ["mob", "desk", "mob", "tab", "mob", "desk", None, "mob"]
SOURCE_VALUES: list[str] = ["a", "b", "a", "a", "c", "b", "a", "a"]
PRICE_VALUES: list[float] = [12.5, 8.0, 3.0, 4.5, 9.9, 1.0, 2.0, 5.0]

MOB_INDEX: int = 2  # byte-sorted vocab: desk=1, mob=2, tab=3
DESK_INDEX: int = 1
TAB_INDEX: int = 3
DEVICE_MAP: dict[str, int] = {"desk": DESK_INDEX, "mob": MOB_INDEX, "tab": TAB_INDEX}


@pytest.fixture
def train_files(tmp_path: Path) -> list[str]:
    """Write two small Parquet files with multiple row groups.

    Args:
        tmp_path: pytest-provided temporary directory.

    Returns:
        Paths of the written Parquet files.

    """
    table: pa.Table = pa.table({"device": DEVICE_VALUES, "source": SOURCE_VALUES, "price": PRICE_VALUES})
    first: Path = tmp_path / "train_0.parquet"
    second: Path = tmp_path / "train_1.parquet"
    pq.write_table(table, first, row_group_size=3)
    pq.write_table(table, second, row_group_size=3)
    return [str(first), str(second)]


def test_label_encoder_fit_builds_sorted_one_based_maps(train_files: list[str]) -> None:
    encoder: LabelEncoder = LabelEncoder(columns=["device"]).fit(train_files)
    assert encoder.label_maps_["device"] == DEVICE_MAP
    assert encoder.vocab_size("device") == len(DEVICE_MAP) + 1


def test_label_encoder_columns_none_encodes_all_utf8(train_files: list[str]) -> None:
    encoder: LabelEncoder = LabelEncoder().fit(train_files)
    assert set(encoder.label_maps_) == {"device", "source"}


def test_label_encoder_min_count_drops_rare_categories(train_files: list[str]) -> None:
    encoder: LabelEncoder = LabelEncoder(columns=["device"], min_count=4).fit(train_files)
    assert encoder.label_maps_["device"] == {"desk": 1, "mob": 2}


def test_label_encoder_transform_dict_known_unknown_null(train_files: list[str]) -> None:
    encoder: LabelEncoder = LabelEncoder(columns=["device"]).fit(train_files)
    encoded = encoder.transform({"device": "mob", "price": 1.0})
    assert encoded == {"device": MOB_INDEX, "price": 1.0}
    assert encoder.transform({"device": "spaceship", "price": 1.0})["device"] == 0
    assert encoder.transform({"device": None, "price": 1.0})["device"] == 0


def test_label_encoder_transform_dict_missing_key_raises(train_files: list[str]) -> None:
    encoder: LabelEncoder = LabelEncoder(columns=["device"]).fit(train_files)
    with pytest.raises(Exception, match="missing"):
        encoder.transform({"price": 1.0})


def test_label_encoder_transform_dict_non_string_raises(train_files: list[str]) -> None:
    encoder: LabelEncoder = LabelEncoder(columns=["device"]).fit(train_files)
    with pytest.raises(Exception, match="string or null"):
        encoder.transform({"device": 2})


def test_label_encoder_batch_output_matches_row_path_positionally(train_files: list[str], tmp_path: Path) -> None:
    encoder: LabelEncoder = LabelEncoder(columns=["device"]).fit(train_files)
    parts: list[Path] = encoder.transform(train_files, output_dir=str(tmp_path / "out"))
    table: pa.Table = pa.concat_tables([pq.read_table(part) for part in parts])
    batch_encoded: list[int] = table.column("device").to_pylist()
    row_encoded: list[int] = [encoder.transform({"device": value})["device"] for value in DEVICE_VALUES] * 2
    assert batch_encoded == row_encoded
    assert table.column("device").type == pa.int64()


def test_label_encoder_handles_large_string_columns(tmp_path: Path) -> None:
    table: pa.Table = pa.table({"city": pa.array(["nyc", "sf", "nyc", "la"], type=pa.large_string())})
    path: Path = tmp_path / "large.parquet"
    pq.write_table(table, path)
    encoder: LabelEncoder = LabelEncoder(columns=["city"]).fit([str(path)])
    assert encoder.label_maps_["city"] == {"la": 1, "nyc": 2, "sf": 3}
    parts: list[Path] = encoder.transform([str(path)], output_dir=str(tmp_path / "out"))
    assert pq.read_table(parts[0]).column("city").to_pylist() == [2, 3, 2, 1]


def test_label_encoder_unicode_categories_sort_by_bytes(tmp_path: Path) -> None:
    values: list[str] = ["é", "é", "", "z", "😀"]
    path: Path = tmp_path / "unicode.parquet"
    pq.write_table(pa.table({"c": values}), path)
    encoder: LabelEncoder = LabelEncoder(columns=["c"]).fit([str(path)])
    assert encoder.label_maps_["c"] == {"": 1, "z": 2, "é": 3, "😀": 4}
    assert encoder.transform({"c": ""})["c"] == 1


def test_label_encoder_rejects_non_utf8_column(train_files: list[str]) -> None:
    with pytest.raises(Exception, match="expected utf8"):
        LabelEncoder(columns=["price"]).fit(train_files)


def test_mismatched_file_schemas_fail_loudly(tmp_path: Path) -> None:
    good: Path = tmp_path / "good.parquet"
    bad: Path = tmp_path / "bad.parquet"
    pq.write_table(pa.table({"device": ["mob", "desk"]}), good)
    pq.write_table(pa.table({"device": [0, 1]}), bad)
    with pytest.raises(Exception, match="is i64 but"):
        LabelEncoder(columns=["device"]).fit([str(good), str(bad)])


def test_threads_do_not_change_the_fit(train_files: list[str]) -> None:
    single: LabelEncoder = LabelEncoder(columns=["device", "source"]).fit(train_files, threads=1)
    many: Pipeline = Pipeline([("encode", LabelEncoder(columns=["device", "source"]))]).fit(train_files, threads=16)
    encoder = many["encode"]
    assert isinstance(encoder, LabelEncoder)
    assert encoder.label_maps_ == single.label_maps_


def test_select_projects_and_orders_columns(train_files: list[str], tmp_path: Path) -> None:
    selector: Select = Select(columns=["price", "device"]).fit(train_files)
    parts: list[Path] = selector.transform(train_files, output_dir=str(tmp_path / "out"))
    table: pa.Table = pq.read_table(parts[0])
    assert table.column_names == ["price", "device"]
    assert selector.transform({"device": "mob", "source": "a", "price": 2.0}) == {"price": 2.0, "device": "mob"}


def test_select_missing_column_fails_at_fit(train_files: list[str]) -> None:
    with pytest.raises(Exception, match="not found"):
        Select(columns=["nope"]).fit(train_files)


def test_select_duplicate_column_fails_at_fit(train_files: list[str]) -> None:
    with pytest.raises(Exception, match="listed more than once"):
        Select(columns=["device", "device"]).fit(train_files)


def test_standalone_fitted_transformer_pickle_round_trip(train_files: list[str]) -> None:
    encoder: LabelEncoder = LabelEncoder(columns=["device"]).fit(train_files)
    restored: LabelEncoder = pickle.loads(pickle.dumps(encoder))
    assert restored.label_maps_ == encoder.label_maps_
    assert restored.transform({"device": "mob"})["device"] == MOB_INDEX


def test_pipeline_fitted_transformer_refuses_standalone_transform(train_files: list[str]) -> None:
    pipeline: Pipeline = Pipeline(
        [("select", Select(columns=["device"])), ("encode", LabelEncoder(columns=["device"]))]
    ).fit(train_files)
    encoder = pipeline["encode"]
    assert isinstance(encoder, LabelEncoder)
    with pytest.raises(RuntimeError, match="fitted as part of a Pipeline"):
        encoder.transform({"device": "mob"})
    # The refusal must not corrupt the encoder's binding to the pipeline.
    assert encoder.label_maps_["device"]["mob"] == MOB_INDEX
