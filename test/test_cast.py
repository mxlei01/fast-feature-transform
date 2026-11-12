"""Tests for the cast kernel: the closed conversion table, exact output types, and batch/record parity."""

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fast_feature_transform import Cast, EngineError, LabelEncoder, Pipeline
from helpers import read_rows, write_parquet

TWO_POW_53: int = 2**53


def test_a_numeric_id_must_be_cast_before_it_can_be_encoded(tmp_path: Path) -> None:
    path: str = write_parquet(tmp_path / "ids.parquet", pa.table({"user_id": pa.array([7, 12, 7], pa.int32())}))
    with pytest.raises(EngineError, match="is i32, expected utf8"):
        LabelEncoder(columns=["user_id"]).fit([path])
    pipeline: Pipeline = Pipeline(
        [("types", Cast({"user_id": "utf8"})), ("encode", LabelEncoder(columns=["user_id"]))]
    ).fit([path])
    encoder = pipeline["encode"]
    assert isinstance(encoder, LabelEncoder)
    assert encoder.label_maps_["user_id"] == {"12": 1, "7": 2}  # digits, sorted as strings
    assert pipeline.transform_dict({"user_id": 12}) == {"user_id": 1}
    assert read_rows(pipeline.transform_parquet([path], output_dir=tmp_path / "out")) == [
        {"user_id": 2},
        {"user_id": 1},
        {"user_id": 2},
    ]
    # 12.0 is not "12": Cast never turns a float into a string.
    with pytest.raises(EngineError, match="cannot cast a float to utf8"):
        pipeline.transform_dict({"user_id": 12.0})


def test_cast_output_types_are_exact_and_agree_with_the_record_path(tmp_path: Path) -> None:
    table: pa.Table = pa.table(
        {
            "price": pa.array([0.1, 2.5, None], pa.float64()),
            "count": pa.array([32767, -5, None], pa.int64()),
            "flag": pa.array([True, False, None], pa.bool_()),
            "text_id": pa.array(["12", "+7", None], pa.string()),
            "text_price": pa.array(["1.5", "-2e3", None], pa.string()),
        }
    )
    path: str = write_parquet(tmp_path / "mixed.parquet", table)
    cast: Cast = Cast({"price": "f32", "count": "i16", "flag": "i64", "text_id": "i64", "text_price": "f64"}).fit(
        [path]
    )
    parts: list[Path] = cast.transform_parquet([path], output_dir=tmp_path / "out")
    output: pa.Table = pq.read_table(parts[0])
    assert output.schema.types == [pa.float32(), pa.int16(), pa.int64(), pa.int64(), pa.float64()]
    rows: list[dict[str, Any]] = output.to_pylist()
    assert rows[2] == {"price": None, "count": None, "flag": None, "text_id": None, "text_price": None}
    for source, batch_row in zip(table.to_pylist()[:2], rows[:2], strict=True):
        assert cast.transform_dict(source) == batch_row
    assert rows[0]["price"] == np.float32(0.1)  # rounded through single precision on both paths
    assert rows[0]["price"] != 0.1


def test_cast_parses_strings_strictly(tmp_path: Path) -> None:
    path: str = write_parquet(tmp_path / "text.parquet", pa.table({"n": ["12"]}))
    cast: Cast = Cast({"n": "i64"}).fit([path])
    assert cast.transform_dict({"n": "+7"}) == {"n": 7}
    assert cast.transform_dict({"n": "-0"}) == {"n": 0}
    for bad in ("12x", "", " 12", "1.0", "0x1f", "+-5", "++5", "+", "-"):
        with pytest.raises(EngineError, match="is not an integer"):
            cast.transform_dict({"n": bad})
    with pytest.raises(EngineError, match="does not fit in a 64-bit integer"):
        cast.transform_dict({"n": "9223372036854775808"})
    as_float: Cast = Cast({"n": "f64"}).fit([path])
    assert as_float.transform_dict({"n": "-2e3"}) == {"n": -2000.0}
    assert as_float.transform_dict({"n": ".5"}) == {"n": 0.5}
    for bad in ("two", "inf", "-inf", "Infinity", "nan", "NaN", "nan(123)", "+-0.5", "1e", " 1"):
        with pytest.raises(EngineError, match="is not a number"):
            as_float.transform_dict({"n": bad})
    with pytest.raises(EngineError, match="is out of range for f64"):
        as_float.transform_dict({"n": "1e400"})


def test_unsigned_64_bit_values_keep_their_type_or_become_digits(tmp_path: Path) -> None:
    big: int = 2**63 + 5
    path: str = write_parquet(tmp_path / "u64.parquet", pa.table({"n": pa.array([big, 7], pa.uint64())}))
    kept: list[Path] = Cast({"n": "u64"}).fit([path]).transform_parquet([path], output_dir=tmp_path / "kept")
    assert pq.read_table(kept[0]).column("n").to_pylist() == [big, 7]
    assert pq.read_table(kept[0]).schema.types == [pa.uint64()]
    digits: list[Path] = Cast({"n": "utf8"}).fit([path]).transform_parquet([path], output_dir=tmp_path / "digits")
    assert pq.read_table(digits[0]).column("n").to_pylist() == [str(big), "7"]
    with pytest.raises(EngineError, match="does not fit in a 64-bit integer"):
        Cast({"n": "i64"}).fit([path]).transform_parquet([path], output_dir=tmp_path / "narrow")


def test_cast_refuses_lossy_or_unlisted_conversions(tmp_path: Path) -> None:
    path: str = write_parquet(
        tmp_path / "t.parquet", pa.table({"n": pa.array([1], pa.int64()), "x": [0.5], "f": [True], "s": ["a"]})
    )
    for tag, value, message in (
        ("i8", 300, "does not fit in i8"),
        ("u8", -1, "does not fit in u8"),
        ("i64", 1.5, "is not a whole number"),
        ("f64", TWO_POW_53 + 1, "where f64 no longer holds every integer exactly"),
        ("f32", 2**24 + 1, "where f32 no longer holds every integer exactly"),
        ("f32", 1e39, "does not fit in f32"),
    ):
        cast: Cast = Cast({"n": tag}).fit([path])
        with pytest.raises(EngineError, match=message):
            cast.transform_dict({"n": value})
    with pytest.raises(EngineError, match="cannot cast a float to utf8 \\(a float has no single spelling"):
        Cast({"x": "utf8"}).fit([path])
    with pytest.raises(EngineError, match="cannot cast a bool to utf8"):
        Cast({"f": "utf8"}).fit([path])
    with pytest.raises(ValueError, match="unsupported type 'bool'"):
        Cast({"s": "bool"})
    with pytest.raises(ValueError, match="unsupported type 'int'"):
        Cast({"n": "int"})
    with pytest.raises(TypeError, match="non-empty mapping"):
        Cast({})
    with pytest.raises(EngineError, match="'nope' not found in input"):
        Cast({"nope": "i64"}).fit([path])


# Every refusal the cast table has for a value: the target, the type and the
# value it refuses, and its text after the place it names. The record path
# builds its place, "<stage>: key '<name>'", only for a refused value, so each
# text is held to the batch path's, which names the column instead of the key.
_REFUSALS: list[tuple[str, pa.DataType, Any, str]] = [
    ("i8", pa.int64(), 300, "value 300 does not fit in i8"),
    ("u8", pa.int64(), -1, "value -1 does not fit in u8"),
    ("i64", pa.float64(), 1.5, "value 1.5 is not a whole number"),
    ("i64", pa.float64(), float("nan"), "value nan is not a whole number"),
    ("i64", pa.float64(), 1e19, "value 1e+19 does not fit in a 64-bit integer"),
    ("i64", pa.string(), "12x", "'12x' is not an integer"),
    ("i64", pa.string(), "1\x002", "'1\x002' is not an integer"),
    ("i64", pa.string(), "9223372036854775808", "'9223372036854775808' does not fit in a 64-bit integer"),
    (
        "f64",
        pa.int64(),
        TWO_POW_53 + 1,
        "value 9007199254740993 is beyond 2^53 in magnitude, where f64 no longer holds every integer exactly",
    ),
    (
        "f32",
        pa.int64(),
        2**24 + 1,
        "value 16777217 is beyond 2^24 in magnitude, where f32 no longer holds every integer exactly",
    ),
    ("f32", pa.float64(), 1e39, "value 1e+39 does not fit in f32"),
    ("f64", pa.string(), "two", "'two' is not a number"),
    ("f64", pa.string(), "1e400", "'1e400' is out of range for f64"),
    (
        "utf8",
        pa.float64(),
        2.0,
        "cannot cast a float to utf8 (a float has no single spelling; cast to an integer first)",
    ),
    ("utf8", pa.bool_(), True, "cannot cast a bool to utf8"),
    ("f64", pa.bool_(), True, "cannot cast a bool to f64"),
]


@pytest.mark.parametrize(
    ("target", "source", "value", "text"), _REFUSALS, ids=[f"{case[0]} from {case[2]!r}" for case in _REFUSALS]
)
def test_a_refused_record_value_names_its_stage_and_key_and_reads_as_the_batch_path_does(
    target: str, source: pa.DataType, value: Any, text: str, tmp_path: Path
) -> None:
    accepted: str = write_parquet(
        tmp_path / "accepted.parquet", pa.table({"kept": pa.array([1], pa.int64()), "n": pa.array([1], pa.int64())})
    )
    refused: str = write_parquet(
        tmp_path / "refused.parquet", pa.table({"kept": pa.array([1], pa.int64()), "n": pa.array([value], source)})
    )
    # Cast walks its columns in sorted order, so the refused one is its second:
    # the key a message names is the refused value's, not the first one's.
    pipeline: Pipeline = Pipeline([("types", Cast({"kept": "i64", "n": target}))]).fit([accepted])
    with pytest.raises(EngineError) as record_error:
        pipeline.transform_dict({"n": value, "kept": 1})
    assert str(record_error.value) == f"types: key 'n': {text}"
    with pytest.raises(EngineError) as batch_error:
        pipeline.transform_parquet([refused], output_dir=tmp_path / "out")
    assert str(batch_error.value) == f"types: column 'n': {text}"


def test_a_refused_record_value_spells_its_stage_and_key_in_full(tmp_path: Path) -> None:
    key: str = "prix d'été"
    path: str = write_parquet(tmp_path / "t.parquet", pa.table({key: pa.array([1], pa.int64())}))
    nested: Pipeline = Pipeline([("outer", Pipeline([("it's", Cast({key: "i8"}))]))]).fit([path])
    with pytest.raises(EngineError) as nested_error:
        nested.transform_dict({key: 300})
    assert str(nested_error.value) == f"outer.it's: key '{key}': value 300 does not fit in i8"
    alone: Cast = Cast({key: "i8"}).fit([path])
    with pytest.raises(EngineError) as alone_error:
        alone.transform_dict({key: -300})
    assert str(alone_error.value) == f"cast: key '{key}': value -300 does not fit in i8"


def test_cast_names_the_refused_key_among_several_and_refuses_no_null(tmp_path: Path) -> None:
    columns: list[str] = ["a", "b", "c"]
    path: str = write_parquet(
        tmp_path / "t.parquet", pa.table({column: pa.array([1], pa.int64()) for column in columns})
    )
    pipeline: Pipeline = Pipeline([("types", Cast(dict.fromkeys(columns, "i8")))]).fit([path])
    for refused in columns:
        # The record's keys are not in the order Cast walks its columns in.
        record: dict[str, Any] = {"c": 1, "a": 2, "b": 3} | {refused: 300}
        with pytest.raises(EngineError) as error:
            pipeline.transform_dict(record)
        assert str(error.value) == f"types: key '{refused}': value 300 does not fit in i8"
    with pytest.raises(EngineError) as missing:
        pipeline.transform_dict({"c": 1, "a": 2})
    assert str(missing.value) == "types: key 'b' missing from record"
    for target in ("utf8", "i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64", "f32", "f64"):
        assert Cast({"a": target}).fit([path]).transform_dict({"a": None, "b": 7}) == {"a": None, "b": 7}


def test_cast_survives_artifacts_and_pickle(tmp_path: Path) -> None:
    path: str = write_parquet(
        tmp_path / "t.parquet", pa.table({"user_id": pa.array([3, 4], pa.int32()), "p": [1.0, 2.0]})
    )
    pipeline: Pipeline = Pipeline(
        [("types", Cast({"user_id": "utf8", "p": "f32"})), ("encode", LabelEncoder(columns=["user_id"]))]
    ).fit([path])
    pipeline.save(tmp_path / "artifact")
    loaded: Pipeline = Pipeline.load(tmp_path / "artifact")
    cast = loaded["types"]
    assert isinstance(cast, Cast)
    assert cast.types == {"user_id": "utf8", "p": "f32"}
    record: dict[str, Any] = {"user_id": 4, "p": 0.1}
    assert loaded.transform_dict(record) == pipeline.transform_dict(record) == {"user_id": 2, "p": np.float32(0.1)}
    assert pickle.loads(pickle.dumps(pipeline)).transform_dict(record) == pipeline.transform_dict(record)


@settings(max_examples=60, deadline=None)
@given(
    values=st.lists(st.one_of(st.none(), st.integers(min_value=-(2**31), max_value=2**31 - 1)), max_size=50),
    reals=st.lists(st.one_of(st.none(), st.floats(allow_nan=False, allow_infinity=False, width=32)), max_size=50),
)
def test_cast_properties(
    tmp_path_factory: pytest.TempPathFactory, values: list[int | None], reals: list[float | None]
) -> None:
    tmp_path: Path = tmp_path_factory.mktemp("cast")
    length: int = min(len(values), len(reals))
    ints: list[int | None] = values[:length]
    floats: list[float | None] = reals[:length]
    table: pa.Table = pa.table({"n": pa.array(ints, pa.int32()), "x": pa.array(floats, pa.float64())})
    path: str = write_parquet(tmp_path / "p.parquet", table)
    cast: Cast = Cast({"n": "utf8", "x": "f32"}).fit([path])
    rows: list[dict[str, Any]] = read_rows(cast.transform_parquet([path], output_dir=tmp_path / "out"))
    for n, x, row in zip(ints, floats, rows, strict=True):
        assert row == cast.transform_dict({"n": n, "x": x})
        assert row["n"] == (None if n is None else str(n))
        assert row["x"] == (None if x is None else np.float32(x))
