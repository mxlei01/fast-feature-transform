"""Tests for datetimes: a record's datetime and a timestamp column both become int64 epoch microseconds."""

# Naive datetimes are the subject here: they count wall-clock time as if it were UTC.
# ruff: noqa: DTZ001

import json
import subprocess  # nosec B404
import sys
from datetime import UTC, date, datetime, time, timedelta, timezone, tzinfo
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fast_feature_transform import EngineError, GroupAgg, Pipeline, StandardScaler, parquet_to_memmap

EPOCH: datetime = datetime(1970, 1, 1)
EPOCH_UTC: datetime = datetime(1970, 1, 1, tzinfo=UTC)
ONE_US: timedelta = timedelta(microseconds=1)


class Stamp(datetime):
    """A datetime subclass, as pandas.Timestamp is one."""


class PandasLike(datetime):
    """A pandas.Timestamp stand-in: its fields may hold a placeholder year, and its asm8 holds the true count."""

    true_count: np.datetime64 = np.datetime64("NaT", "ns")

    @property
    def asm8(self) -> np.datetime64:
        """The true stored count, which pandas keeps beside the datetime fields.

        Returns:
            The count as a numpy datetime64.

        """
        return self.true_count


def _pandas_like(true_count: np.datetime64) -> PandasLike:
    """Make a stand-in whose fields read 1972-01-01, as pandas writes them for 0000-01-01 and 10000-01-01.

    Args:
        true_count: What its asm8 returns.

    Returns:
        The stand-in.

    """
    stamp: PandasLike = PandasLike(1972, 1, 1)
    stamp.true_count = true_count
    return stamp


class Floating(tzinfo):
    """A time zone that gives no offset, which leaves a datetime naive."""

    def utcoffset(self, dt: datetime | None) -> timedelta | None:
        """Decline to give an offset.

        Args:
            dt: The datetime asking.

        Returns:
            None.

        """
        return None


def _scaler(tmp_path: Path) -> Pipeline:
    """Fit a pipeline that reads only "x", so every other record key passes through.

    Args:
        tmp_path: pytest-provided temporary directory.

    Returns:
        The fitted pipeline.

    """
    path: Path = tmp_path / "x.parquet"
    pq.write_table(pa.table({"x": [1.0, 3.0]}), path)
    return Pipeline([("scale", StandardScaler(columns=["x"]))]).fit([path])


def _through(pipeline: Pipeline, value: Any) -> Any:
    """Pass one value through the record path under a key no step reads.

    Args:
        pipeline: A pipeline from _scaler.
        value: The value to send.

    Returns:
        What came back under that key.

    """
    return pipeline.transform_dict({"x": 1.0, "when": value})["when"]


def test_a_naive_datetime_counts_its_wall_clock_as_utc(tmp_path: Path) -> None:
    pipeline: Pipeline = _scaler(tmp_path)
    values: list[datetime] = [
        EPOCH,
        datetime(2024, 1, 5, 8, 30, 0, 123_456),
        datetime(1969, 12, 31, 23, 59, 59, 999_999),
        datetime(1, 1, 1),
        datetime(9999, 12, 31, 23, 59, 59, 999_999),
    ]
    # Arrow's timestamp[us] storage is the oracle: what the batch path reads.
    stored: list[int] = pa.array(values, pa.timestamp("us")).cast(pa.int64()).to_pylist()
    assert stored[1] == 1_704_443_400_123_456
    assert [_through(pipeline, value) for value in values] == stored
    assert [_through(pipeline, Stamp.fromisoformat(value.isoformat())) for value in values] == stored
    assert _through(pipeline, datetime(2024, 1, 5, 8, 30, tzinfo=Floating())) == 1_704_443_400_000_000


def test_an_aware_datetime_counts_its_utc_instant(tmp_path: Path) -> None:
    pipeline: Pipeline = _scaler(tmp_path)
    offsets: list[timedelta] = [
        timedelta(0),
        timedelta(hours=-5),
        timedelta(hours=5, minutes=30),
        timedelta(hours=-23, minutes=-59, seconds=-59, microseconds=-999_999),
        timedelta(hours=23, minutes=59, seconds=59, microseconds=999_999),
    ]
    for offset in offsets:
        value: datetime = datetime(2024, 1, 5, 8, 30, 0, 250_000, tzinfo=timezone(offset))
        expected: int = (value - EPOCH_UTC) // ONE_US
        if offset.microseconds == 0:  # pyarrow drops an offset's microseconds; Python's arithmetic does not
            assert expected == pa.array([value], pa.timestamp("us", tz="UTC")).cast(pa.int64())[0].as_py()
        assert _through(pipeline, value) == expected, offset


def test_nat_becomes_null_and_numpy_datetimes_convert_exactly_with_floor(tmp_path: Path) -> None:
    pipeline: Pipeline = _scaler(tmp_path)
    assert _through(pipeline, np.datetime64("NaT", "us")) is None
    assert _through(pipeline, np.datetime64("NaT", "ns")) is None
    lowest: int = -(2**63) + 1  # the lowest count that is not NaT
    cases: list[tuple[np.datetime64, int]] = [
        (np.datetime64("2024-01-05T08:30:00.123456789", "ns"), 1_704_443_400_123_456),
        (np.datetime64(-1, "ns"), -1),
        (np.datetime64(-1_500, "ns"), -2),
        (np.datetime64(1_500, "ns"), 1),
        (np.datetime64("2024-01-05T08:30:00.123456", "us"), 1_704_443_400_123_456),
        (np.datetime64(-1, "ms"), -1_000),
        (np.datetime64(-1, "s"), -1_000_000),
        (np.datetime64(-1, "m"), -60_000_000),
        (np.datetime64(5, "h"), 18_000_000_000),
        # numpy's own cast to microseconds overflows on these, though they fit.
        (np.datetime64(lowest, "ns"), lowest // 1_000),
        (np.datetime64(lowest, "as"), lowest // 10**12),
        (np.array([lowest], "M8[7ns]")[0], lowest * 7 // 1_000),  # a multiplied unit
        (np.array([2**62], "M8[3ns]")[0], 2**62 * 3 // 1_000),
    ]
    for value, expected in cases:
        assert _through(pipeline, value) == expected, value


def test_a_pandas_timestamp_converts_from_its_true_count(tmp_path: Path) -> None:
    pipeline: Pipeline = _scaler(tmp_path)
    # Outside years 1 to 9999 pandas fills the year field with 1970 or 1972; asm8 is the true count.
    assert _through(pipeline, _pandas_like(np.datetime64("10000-01-01T00:00:00", "s"))) == 253_402_300_800_000_000
    assert _through(pipeline, _pandas_like(np.datetime64("0000-01-01T00:00:00", "s"))) == -62_167_219_200_000_000
    assert _through(pipeline, _pandas_like(np.datetime64("2024-01-05T08:30:00.123456789", "ns"))) == (
        1_704_443_400_123_456
    )
    assert _through(pipeline, _pandas_like(np.datetime64("NaT", "ns"))) is None  # pandas.NaT


def test_datetimes_outside_int64_microseconds_dates_and_other_time_types_are_refused(tmp_path: Path) -> None:
    pipeline: Pipeline = _scaler(tmp_path)
    for far in (np.datetime64(10**10, "h"), np.datetime64(-(10**10), "h"), np.datetime64(2**62, "s")):
        with pytest.raises(EngineError, match="'when' is outside the range of int64 microseconds"):
            _through(pipeline, far)
    # numpy's day units and coarser are dates, refused like datetime.date; the batch path keeps them as date32.
    for unit in ("D", "W", "M", "Y"):
        with pytest.raises(EngineError, match=f"'when' is a numpy datetime64 in '{unit}' units, which is a date"):
            _through(pipeline, np.datetime64(1, unit))
    for value, name in (
        (date(2024, 1, 5), "date"),
        (time(8, 30), "time"),
        (timedelta(days=1), "timedelta"),
        (np.timedelta64(1, "s"), "timedelta64"),
    ):
        with pytest.raises(EngineError, match=rf"must be str, int, float, bool, datetime, or None \(got {name}\)"):
            _through(pipeline, value)


# Each case sends a value that the record path reads through attributes it
# looks up: a value of a type named datetime64, as numpy's is, or a
# datetime's asm8, as a pandas Timestamp has one. A lookup that failed used
# to end the process, so the cases run in a child process, which prints what
# came back under the value's key.
ATTRIBUTE_LOOKUP_CHILD: str = """
import datetime
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from fast_feature_transform import Pipeline, StandardScaler


class Count:
    def item(self):
        return 5


def astype(self, kind):
    return Count()


def named(**members):
    return type("datetime64", (), members)()


def raising(error):
    def get(self):
        raise error

    return property(get)


def stamp(asm8):
    return type("Stamp", (datetime.datetime,), {"asm8": asm8})(2024, 1, 5, 8, 30)


cases = [
    ("no dtype", named(astype=astype)),
    ("no astype", named(dtype=np.dtype("M8[s]"))),
    ("dtype raises", named(astype=astype, dtype=raising(ValueError("no dtype")))),
    ("astype raises", named(astype=raising(ValueError("no astype")), dtype=np.dtype("M8[s]"))),
    ("both", named(astype=astype, dtype=np.dtype("M8[s]"))),
    ("asm8 without dtype", stamp(property(lambda self: named(astype=astype)))),
    ("asm8 raises", stamp(raising(ValueError("no asm8")))),
    ("asm8 absent", stamp(raising(AttributeError("asm8")))),
]
pq.write_table(pa.table({"x": [1.0, 3.0]}), sys.argv[1])
pipeline = Pipeline([("scale", StandardScaler(columns=["x"]))]).fit([sys.argv[1]], threads=1)
for label, value in cases:
    try:
        answer = pipeline.transform_dict({"x": 1.0, "when": value})["when"]
    except Exception as error:
        answer = f"{type(error).__name__}: {error}"
    print(f"{label}: {answer}")
"""


def test_a_failed_attribute_lookup_raises_the_values_own_error(tmp_path: Path) -> None:
    # REGRESSION: the record path fetches each attribute it calls with before
    # the call. A value of a type named datetime64 that lacks astype or dtype
    # is not numpy's, and gets the EngineError of a type that is not
    # supported; any other failed lookup raises the value's own error, and an
    # asm8 that raises AttributeError is taken as absent. nanobind used to
    # convert value.attr("dtype") as a call's argument, where its failure
    # ended the process, and an asm8 that raised anything else was read as
    # absent.
    unsupported: str = "EngineError: record value for 'when' must be str, int, float, bool, datetime, or None"
    expected: list[str] = [
        f"no dtype: {unsupported} (got datetime64)",
        f"no astype: {unsupported} (got datetime64)",
        "dtype raises: ValueError: no dtype",
        "astype raises: ValueError: no astype",
        "both: 5000000",  # 5 seconds
        "asm8 without dtype: AttributeError: 'datetime64' object has no attribute 'dtype'",
        "asm8 raises: ValueError: no asm8",
        f"asm8 absent: {(datetime(2024, 1, 5, 8, 30) - EPOCH) // ONE_US}",  # read from its fields
    ]
    command: list[str] = [sys.executable, "-c", ATTRIBUTE_LOOKUP_CHILD, str(tmp_path / "x.parquet")]
    child = subprocess.run(command, capture_output=True, text=True, cwd=tmp_path, timeout=60, check=False)  # nosec B603
    assert (child.returncode, child.stdout.splitlines()) == (0, expected), child.stderr


def test_timestamp_columns_read_as_epoch_microseconds_and_agree_with_records(tmp_path: Path) -> None:
    pipeline: Pipeline = _scaler(tmp_path)
    naive: list[datetime | None] = [datetime(2024, 1, 5, 8, 30, 0, 1), None, datetime(1969, 12, 31, 23, 59, 59)]
    plus_five = timezone(timedelta(hours=5))
    aware: list[datetime | None] = [
        value.replace(microsecond=0, tzinfo=plus_five) if value else None for value in naive
    ]
    nanos: list[int | None] = [1_704_443_400_000_001_999, None, -1_500]
    table: pa.Table = pa.table(
        {
            "x": [1.0, 2.0, 3.0],
            "us": pa.array(naive, pa.timestamp("us")),
            "ms": pa.array([0, -1, None], pa.timestamp("ms", tz="UTC")),
            "s": pa.array(aware, pa.timestamp("s", tz="+05:00")),
            "ns": pa.array(nanos, pa.timestamp("ns")),
        }
    )
    source: Path = tmp_path / "stamps.parquet"
    pq.write_table(table, source)  # Parquet has no seconds unit: "s" is written, and read, as milliseconds
    out: pa.Table = pq.read_table(pipeline.transform_parquet([source], output_dir=tmp_path / "parts"))
    for name in ("us", "ms", "s", "ns"):
        assert out.schema.field(name).type == pa.int64(), name
    assert out.column("us").to_pylist() == [(value - EPOCH) // ONE_US if value else None for value in naive]
    assert out.column("ms").to_pylist() == [0, -1_000, None]
    assert out.column("s").to_pylist() == [(value - EPOCH_UTC) // ONE_US if value else None for value in aware]
    assert out.column("ns").to_pylist() == [1_704_443_400_000_001, None, -2]

    # The record path agrees: Python datetimes for whole microseconds, numpy
    # datetime64[ns] (NaT for null) for the column holding nanoseconds.
    rows: list[dict[str, Any]] = table.drop_columns(["ns"]).to_pylist()
    stamps: np.ndarray = table.column("ns").to_numpy(zero_copy_only=False)
    expected: list[dict[str, Any]] = out.to_pylist()
    for row, stamp, want in zip(rows, stamps, expected, strict=True):
        assert pipeline.transform_dict({**row, "ns": stamp}) == want


def test_parquet_transforms_and_memmaps_take_timestamp_columns_as_int64(tmp_path: Path) -> None:
    pickup: list[datetime] = [datetime(2026, 8, 1, 0, 5), datetime(2026, 8, 31, 23, 59, 59)]
    source: Path = tmp_path / "trips.parquet"
    pq.write_table(pa.table({"x": [1.0, 2.0], "pickup": pa.array(pickup, pa.timestamp("us"))}), source)
    micros: list[int] = [(value - EPOCH) // ONE_US for value in pickup]
    pipeline: Pipeline = Pipeline([("scale", StandardScaler(columns=["x"]))]).fit([source])

    part: pa.Table = pq.read_table(pipeline.transform_parquet([source], output_dir=tmp_path / "parts")[0])
    assert part.schema.field("pickup").type == pa.int64()
    assert part.column("pickup").to_pylist() == micros

    for index in (
        pipeline.transform_memmap([source], output_dir=tmp_path / "mmap"),
        Path(parquet_to_memmap([str(source)], output_dir=str(tmp_path / "direct"))),
    ):
        fields: dict[str, Any] = {field["name"]: field for field in json.loads(index.read_text())["columns"]}
        assert fields["pickup"]["dtype"] == "int64"


def test_steps_read_a_timestamp_column_as_epoch_microseconds(tmp_path: Path) -> None:
    source: Path = tmp_path / "events.parquet"
    stamps: pa.Array = pa.array([1, None, 3, -2], pa.timestamp("ms"))
    pq.write_table(pa.table({"user": ["a", "b", "a", "b"], "ts": stamps}), source)
    step: GroupAgg = GroupAgg(key="user", aggs={"ts": ["min", "max"]}).fit([source])
    assert step.values_ == {"ts_min": [1_000, -2_000], "ts_max": [3_000, -2_000]}
    scaler: StandardScaler = StandardScaler(columns=["ts"]).fit([source])
    assert scaler.mean_ == {"ts": pytest.approx(2_000 / 3)}


def test_int96_timestamps_read_at_microseconds_outside_1677_to_2262(tmp_path: Path) -> None:
    # Legacy writers store INT96, which Arrow decodes as nanoseconds by default and so wraps for these dates.
    stamps: list[datetime] = [
        datetime(9999, 12, 31),
        datetime(1, 1, 1),
        datetime(1500, 6, 1, 12),
        datetime(3000, 1, 1, 0, 0, 0, 1),
    ]
    source: Path = tmp_path / "legacy.parquet"
    table: pa.Table = pa.table({"x": [1.0, 2.0, 3.0, 4.0], "t": pa.array(stamps, pa.timestamp("us"))})
    pq.write_table(table, source, use_deprecated_int96_timestamps=True)
    expected: list[int] = [(value - EPOCH) // ONE_US for value in stamps]
    pipeline: Pipeline = Pipeline([("scale", StandardScaler(columns=["x"]))]).fit([source])
    part: pa.Table = pq.read_table(pipeline.transform_parquet([source], output_dir=tmp_path / "parts")[0])
    assert part.column("t").to_pylist() == expected
    index: dict[str, Any] = json.loads(pipeline.transform_memmap([source], output_dir=tmp_path / "mmap").read_text())
    dtype: np.dtype = np.dtype([(column["name"], column["dtype"]) for column in index["columns"]])
    records: np.ndarray = np.fromfile(tmp_path / "mmap" / index["file"], dtype=dtype)
    assert records["t"].tolist() == expected


def test_files_must_agree_on_integers_naive_and_zoned_timestamps(tmp_path: Path) -> None:
    def write(name: str, column: pa.Array) -> Path:
        path: Path = tmp_path / name
        pq.write_table(pa.table({"t": column}), path)
        return path

    epochs: Path = write("epochs.parquet", pa.array([1_700_000_000], pa.int64()))
    naive: Path = write("naive.parquet", pa.array([1_700_000_000], pa.timestamp("s")))
    utc: Path = write("utc.parquet", pa.array([1_700_000_000_000_000], pa.timestamp("us", tz="UTC")))
    with pytest.raises(EngineError, match="'t' is a timestamp without a time zone but .*epochs.parquet has i64"):
        StandardScaler(columns=["t"]).fit([epochs, naive])
    with pytest.raises(EngineError, match="'t' is a timestamp with a time zone but .*has a timestamp without"):
        parquet_to_memmap([str(naive), str(utc)], output_dir=str(tmp_path / "mixed"))
    scaler: StandardScaler = StandardScaler(columns=["t"]).fit(
        [naive, write("micros.parquet", pa.array([0], pa.timestamp("us")))]
    )
    assert scaler.mean_ == {"t": 1_700_000_000_000_000 / 2}


def test_a_timestamp_outside_the_microsecond_range_is_refused_in_a_file(tmp_path: Path) -> None:
    pipeline: Pipeline = _scaler(tmp_path)
    source: Path = tmp_path / "far.parquet"
    # Parquet has no seconds unit, so 10**14 seconds is stored as 10**17 milliseconds: past int64 microseconds.
    pq.write_table(pa.table({"x": [1.0], "t": pa.array([10**14], pa.timestamp("s"))}), source)
    with pytest.raises(
        EngineError, match="column 't' holds timestamp\\[ms\\] value 100000000000000000, outside the range"
    ):
        pipeline.transform_parquet([source], output_dir=tmp_path / "out")


def test_a_group_agg_saves_a_timestamp_column_as_an_integer(tmp_path: Path) -> None:
    source: Path = tmp_path / "events.parquet"
    pq.write_table(pa.table({"user": ["a", "b", "a"], "ts": pa.array([1, None, 3], pa.timestamp("ms"))}), source)
    fresh: Pipeline = Pipeline([("agg", GroupAgg(key="user", aggs={"ts": ["count"]}))]).fit([source])
    fresh.save(tmp_path / "artifact")
    manifest: dict[str, Any] = json.loads((tmp_path / "artifact" / "manifest.json").read_text())
    assert manifest["ops"][0]["state"]["column_types"] == {"ts": "i64"}
    loaded: Pipeline = Pipeline.load(tmp_path / "artifact")
    assert loaded.transform_dict({"user": "a"}) == fresh.transform_dict({"user": "a"}) == {"user": "a", "ts_count": 2}
    loaded_parts: list[Path] = loaded.transform_parquet([source], output_dir=tmp_path / "loaded")
    fresh_parts: list[Path] = fresh.transform_parquet([source], output_dir=tmp_path / "fresh")
    assert pq.read_table(loaded_parts).to_pylist() == pq.read_table(fresh_parts).to_pylist()
