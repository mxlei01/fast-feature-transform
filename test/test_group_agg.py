"""Tests for the group_agg kernel: aggregate semantics, explicit output types, batch/record parity, and the artifact."""

import json
import math
import pickle
import statistics
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pytest

from fast_feature_transform import Cast, EngineError, GroupAgg, LabelEncoder, Pipeline, Select, _core
from helpers import read_table, write_parquet

ALL: list[str] = ["count", "sum", "mean", "min", "max", "median"]


@pytest.fixture
def events(tmp_path: Path) -> list[str]:
    """Two files with two row groups each: string keys, a float and an integer column, nulls in all three.

    Args:
        tmp_path: pytest-provided temporary directory.

    Returns:
        Paths of the written Parquet files.

    """
    first: pa.Table = pa.table(
        {
            "user": ["a", "b", "a", "c"],
            "price": pa.array([1.5, 2.5, None, 4.0], pa.float64()),
            "clicks": pa.array([1, 2, 3, None], pa.int32()),
        }
    )
    second: pa.Table = pa.table(
        {
            "user": ["b", None, "a", "d"],
            "price": pa.array([0.5, 9.0, 2.0, None], pa.float64()),
            "clicks": pa.array([5, 7, 1, None], pa.int32()),
        }
    )
    return [
        write_parquet(tmp_path / "events-0.parquet", first, row_group_size=2),
        write_parquet(tmp_path / "events-1.parquet", second, row_group_size=2),
    ]


def test_group_agg_computes_each_aggregate_with_explicit_types(events: list[str], tmp_path: Path) -> None:
    step: GroupAgg = GroupAgg(key="user", aggs={"price": ALL, "clicks": ALL}).fit(events)
    assert step.keys_ == ["a", "b", "c", "d"]
    values: dict[str, list[Any]] = step.values_
    # a: price [1.5, 2.0], clicks [1, 3, 1]; b: price [2.5, 0.5], clicks [2, 5]; c: price [4.0], no clicks;
    # d: nothing but the key; the null-key row contributes nothing.
    assert values["price_count"] == [2, 2, 1, 0]
    assert values["price_sum"] == [3.5, 3.0, 4.0, 0.0]
    assert values["price_mean"] == [1.75, 1.5, 4.0, 0.0]
    assert values["price_min"] == [1.5, 0.5, 4.0, 0.0]
    assert values["price_max"] == [2.0, 2.5, 4.0, 0.0]
    assert values["price_median"] == [1.75, 1.5, 4.0, 0.0]
    assert values["clicks_count"] == [3, 2, 0, 0]
    assert values["clicks_sum"] == [5, 7, 0, 0]
    assert values["clicks_mean"] == [5 / 3, 3.5, 0.0, 0.0]
    assert values["clicks_min"] == [1, 2, 0, 0]
    assert values["clicks_max"] == [3, 5, 0, 0]
    assert values["clicks_median"] == [1.0, 3.5, 0.0, 0.0]

    table: pa.Table = read_table(step.transform_parquet(events, output_dir=tmp_path / "out"))
    # The aggregated columns are replaced by their aggregates, appended after the rest in column-name order
    # and then the listed order; the key stays.
    assert table.column_names == ["user"] + [f"clicks_{a}" for a in ALL] + [f"price_{a}" for a in ALL]
    types: dict[str, pa.DataType] = {field.name: field.type for field in table.schema}
    assert types["clicks_count"] == pa.int64()
    assert types["clicks_sum"] == pa.int64()
    assert types["clicks_mean"] == pa.float64()
    assert types["clicks_min"] == pa.int32()  # min and max keep the column's type
    assert types["clicks_max"] == pa.int32()
    assert types["clicks_median"] == pa.float64()
    assert types["price_sum"] == pa.float64()
    assert types["price_min"] == pa.float64()
    rows: list[dict[str, Any]] = table.to_pylist()
    expected_a: dict[str, Any] = {
        "clicks_count": 3,
        "clicks_sum": 5,
        "clicks_mean": 5 / 3,
        "clicks_min": 1,
        "clicks_max": 3,
        "clicks_median": 1.0,
        "price_count": 2,
        "price_sum": 3.5,
        "price_mean": 1.75,
        "price_min": 1.5,
        "price_max": 2.0,
        "price_median": 1.75,
    }
    assert rows[0] == {"user": "a", **expected_a}
    assert step.transform_dict({"user": "a", "price": 1.5, "clicks": 1}) == rows[0]
    # A null key reads as unknown: zeros of the right type, nothing else touched.
    zeros: dict[str, Any] = {name: type(value)() for name, value in expected_a.items()}  # 0 or 0.0 per column type
    assert rows[5] == {"user": None, **zeros}
    assert step.transform_dict({"user": None}) == {"user": None, **zeros}
    assert step.transform_dict({"user": "nobody"}) == {"user": "nobody", **zeros}
    for name, value in step.transform_dict({"user": "nobody"}).items():
        if name != "user":
            assert type(value) is type(zeros[name])


def test_group_agg_integer_keys_and_key_type_checks(tmp_path: Path) -> None:
    table: pa.Table = pa.table({"uid": pa.array([7, 12, 7], pa.int32()), "n": pa.array([1.0, 2.0, 5.0], pa.float64())})
    path: str = write_parquet(tmp_path / "ints.parquet", table)
    step: GroupAgg = GroupAgg(key="uid", aggs={"n": ["sum"]}).fit([path])
    assert step.keys_ == [7, 12]
    assert step.values_ == {"n_sum": [6.0, 2.0]}
    assert step.transform_dict({"uid": 7}) == {"uid": 7, "n_sum": 6.0}
    assert step.transform_dict({"uid": 8}) == {"uid": 8, "n_sum": 0.0}
    with pytest.raises(EngineError, match="key 'uid' must be an integer or null"):
        step.transform_dict({"uid": "7"})
    with pytest.raises(EngineError, match="key 'uid' must be an integer or null"):
        step.transform_dict({"uid": True})
    with pytest.raises(EngineError, match="key 'uid' missing from record"):
        step.transform_dict({"n": 1.0})

    strings: str = write_parquet(tmp_path / "strings.parquet", pa.table({"uid": ["7", "12"], "n": [1.0, 2.0]}))
    with pytest.raises(EngineError, match="key column 'uid' is utf8, but the fitted table has integer keys"):
        step.transform_parquet([strings], output_dir=tmp_path / "out")
    fitted_on_strings: GroupAgg = GroupAgg(key="uid", aggs={"n": ["sum"]}).fit([strings])
    with pytest.raises(EngineError, match="key 'uid' must be a string or null"):
        fitted_on_strings.transform_dict({"uid": 7})
    with pytest.raises(EngineError, match="key column 'uid' is i32, but the fitted table has utf8 keys"):
        fitted_on_strings.transform_parquet([path], output_dir=tmp_path / "out")

    # An encoded key is an integer key; the aggregate column may also be the key.
    pipeline: Pipeline = Pipeline(
        [("encode", LabelEncoder(columns=["uid"])), ("stats", GroupAgg(key="uid", aggs={"uid": ["count"]}))]
    ).fit([strings])
    assert pipeline.transform_dict({"uid": "7"}) == {"uid": 2, "uid_count": 1}


def test_group_agg_validates_params_columns_and_output_names(tmp_path: Path) -> None:
    empty_key: Any = ""
    bare: Any = "mean"
    no_aggs: Any = {}
    with pytest.raises(TypeError, match="key must be a column name"):
        GroupAgg(key=empty_key, aggs={"n": ["sum"]})
    with pytest.raises(TypeError, match="non-empty mapping"):
        GroupAgg(key="k", aggs=no_aggs)
    with pytest.raises(TypeError, match="must be a non-empty list"):
        GroupAgg(key="k", aggs={"n": bare})
    with pytest.raises(ValueError, match="unknown aggregate 'avg'"):
        GroupAgg(key="k", aggs={"n": ["avg"]})
    with pytest.raises(ValueError, match="more than once"):
        GroupAgg(key="k", aggs={"n": ["sum", "sum"]})
    assert repr(GroupAgg(key="k", aggs={"n": ("sum", "max")})) == "GroupAgg(key='k', aggs={'n': ['sum', 'max']})"

    table: pa.Table = pa.table(
        {
            "user": ["a", "b"],
            "device": ["ios", "web"],
            "price": [1.0, 2.0],
            "price_mean": [0.0, 0.0],
            "score": [0.5, 0.25],
        }
    )
    path: str = write_parquet(tmp_path / "mixed.parquet", table)
    with pytest.raises(EngineError, match="key column 'nope' not found in input"):
        GroupAgg(key="nope", aggs={"price": ["sum"]}).fit([path])
    with pytest.raises(EngineError, match="key column 'score' is f64, expected utf8 or an integer column"):
        GroupAgg(key="score", aggs={"price": ["sum"]}).fit([path])
    with pytest.raises(EngineError, match="column 'nope' not found in input"):
        GroupAgg(key="user", aggs={"nope": ["sum"]}).fit([path])
    with pytest.raises(EngineError, match="column 'device' is utf8; only count applies to a string column"):
        GroupAgg(key="user", aggs={"device": ["count", "mean"]}).fit([path])
    with pytest.raises(EngineError, match="output column 'price_mean' already exists in input"):
        GroupAgg(key="user", aggs={"price": ["mean"]}).fit([path])
    counted: GroupAgg = GroupAgg(key="user", aggs={"device": ["count"]}).fit([path])
    assert counted.values_ == {"device_count": [1, 1]}
    with pytest.raises(EngineError, match="key 'device_count' already present in record"):
        counted.transform_dict({"user": "a", "device_count": 5})


def test_group_agg_errors_on_integer_overflow_and_non_finite_floats(tmp_path: Path) -> None:
    big: pa.Table = pa.table({"k": ["x", "x"], "n": pa.array([2**62, 2**62], pa.int64())})
    path: str = write_parquet(tmp_path / "big.parquet", big)
    with pytest.raises(EngineError, match="the sum overflows a 64-bit integer"):
        GroupAgg(key="k", aggs={"n": ["sum"]}).fit([path])
    # Only the sum overflows a 64-bit integer: max and mean of the same column do not.
    assert GroupAgg(key="k", aggs={"n": ["max", "mean"]}).fit([path]).values_ == {"n_max": [2**62], "n_mean": [2.0**62]}

    nan: str = write_parquet(tmp_path / "nan.parquet", pa.table({"k": ["x"], "f": pa.array([math.nan], pa.float64())}))
    with pytest.raises(EngineError, match="is not finite; group_agg cannot aggregate NaN or infinity"):
        GroupAgg(key="k", aggs={"f": ["max"]}).fit([nan])


def test_group_agg_keeps_narrow_types_through_cast_and_the_artifact(tmp_path: Path) -> None:
    table: pa.Table = pa.table({"k": ["a", "a", "b"], "price": [1.25, 2.5, 8.0], "n": pa.array([1, 2, 3], pa.uint8())})
    path: str = write_parquet(tmp_path / "narrow.parquet", table)
    pipeline: Pipeline = Pipeline(
        [
            ("types", Cast({"price": "f32"})),
            ("stats", GroupAgg(key="k", aggs={"price": ["min", "sum"], "n": ["max", "sum"]})),
        ]
    ).fit([path])
    out: pa.Table = read_table(pipeline.transform_parquet([path], output_dir=tmp_path / "out"))
    types: dict[str, pa.DataType] = {field.name: field.type for field in out.schema}
    assert types["price_min"] == pa.float32()
    assert types["price_sum"] == pa.float64()
    assert types["n_max"] == pa.uint8()
    assert types["n_sum"] == pa.int64()
    assert out.to_pylist()[2] == {"k": "b", "n_max": 3, "n_sum": 3, "price_min": 8.0, "price_sum": 8.0}

    pipeline.save(tmp_path / "artifact")
    manifest: dict[str, Any] = json.loads((tmp_path / "artifact" / "manifest.json").read_text())
    state: dict[str, Any] = manifest["ops"][1]["state"]
    assert state["key_type"] == "utf8"
    assert state["keys"] == ["a", "b"]
    assert state["column_types"] == {"n": "u8", "price": "f32"}
    assert state["values"] == {"n_max": [2, 3], "n_sum": [3, 3], "price_min": [1.25, 8.0], "price_sum": [3.75, 8.0]}
    loaded: Pipeline = Pipeline.load(tmp_path / "artifact")
    record: dict[str, Any] = {"k": "a", "price": 1.25, "n": 1}
    assert loaded.transform_dict(record) == pipeline.transform_dict(record)
    assert pickle.loads(pickle.dumps(pipeline)).transform_dict(record) == pipeline.transform_dict(record)
    stats = loaded["stats"]
    assert isinstance(stats, GroupAgg)
    assert stats.keys_ == ["a", "b"]

    # A hand-edited table is checked before it is trusted.
    manifest["ops"][1]["state"]["values"]["n_max"] = [2, 300]
    with pytest.raises(EngineError, match="value 300 does not fit in u8"):
        _core.FittedPipeline.load_json(json.dumps(manifest))
    manifest["ops"][1]["state"]["values"]["n_max"] = [2, 3]
    manifest["ops"][1]["state"]["values"]["price_min"] = [2.0**24 + 1, 8.0]  # not a float32 value
    with pytest.raises(EngineError, match=r"values\['price_min'\] must hold f32 values"):
        _core.FittedPipeline.load_json(json.dumps(manifest))
    manifest["ops"][1]["state"]["values"]["price_min"] = [1.25, 8.0]
    manifest["ops"][1]["state"]["keys"] = ["b", "a"]
    with pytest.raises(EngineError, match="keys must be sorted and unique"):
        _core.FittedPipeline.load_json(json.dumps(manifest))


def test_group_agg_outputs_feed_later_steps_in_batch_and_record_mode(tmp_path: Path) -> None:
    path: str = write_parquet(tmp_path / "sales.parquet", pa.table({"user": ["a", "b", "a"], "price": [1.0, 2.0, 5.0]}))
    pipeline: Pipeline = Pipeline(
        [
            ("stats", GroupAgg(key="user", aggs={"price": ["sum", "count"]})),
            ("types", Cast({"price_sum": "f32"})),
            ("keep", Select(columns=["user", "price_sum", "price_count"])),
            ("again", GroupAgg(key="price_count", aggs={"price_sum": ["max"]})),
        ]
    ).fit([path])
    rows: list[dict[str, Any]] = read_table(pipeline.transform_parquet([path], output_dir=tmp_path / "out")).to_pylist()
    # stats replaces price with price_sum and price_count; again replaces price_sum with price_sum_max.
    assert rows == [
        {"user": "a", "price_count": 2, "price_sum_max": 6.0},
        {"user": "b", "price_count": 1, "price_sum_max": 2.0},
        {"user": "a", "price_count": 2, "price_sum_max": 6.0},
    ]
    assert pipeline.transform_dict({"user": "b", "price": 2.0}) == rows[1]
    pipeline.save(tmp_path / "artifact")
    loaded: Pipeline = Pipeline.load(tmp_path / "artifact")
    assert read_table(loaded.transform_parquet([path], output_dir=tmp_path / "again")).to_pylist() == rows


def test_group_agg_sums_are_exact_and_independent_of_the_split(tmp_path: Path) -> None:
    # Three magnitude tiers with cancellation: a merely compensated sum loses the small tier for some splits.
    values: list[float] = [1e40, 1.0, -1e40, 1e20, 1.0, -1e20, 1.0, 1.0]
    path: str = write_parquet(tmp_path / "tiers.parquet", pa.table({"k": ["x"] * 8, "v": values}), row_group_size=2)
    states: set[str] = set()
    for threads in (1, 2, 4):
        step: GroupAgg = GroupAgg(key="k", aggs={"v": ["sum", "mean"]}).fit([path], threads=threads)
        assert step.values_ == {"v_mean": [0.5], "v_sum": [4.0]}
        states.add(step._require_handle().stage_state("groupagg"))
    assert len(states) == 1

    # An integer sum that overflows 64 bits on the way but not at the end never depends on the split either,
    # and the mean of integers comes from the exact sum.
    transient: pa.Table = pa.table({"k": ["x"] * 3, "n": pa.array([2**63 - 1, -2, 1], pa.int64())})
    path = write_parquet(tmp_path / "transient.parquet", transient, row_group_size=1)
    for threads in (1, 3):
        assert GroupAgg(key="k", aggs={"n": ["sum"]}).fit([path], threads=threads).values_ == {"n_sum": [2**63 - 2]}
    wide: pa.Table = pa.table({"k": ["x"] * 3, "n": pa.array([2**62 + 1, -(2**62) + 1, 3], pa.int64())})
    assert GroupAgg(key="k", aggs={"n": ["sum", "mean"]}).fit(
        [write_parquet(tmp_path / "wide.parquet", wide)]
    ).values_ == {
        "n_mean": [5 / 3],
        "n_sum": [5],
    }

    # A median midpoint neither cancels nor loses the last bit of a subnormal.
    pair: pa.Table = pa.table({"k": ["x", "x", "y", "y"], "v": [-1e9, 1e9 + 1e-7, 5e-324, 5e-324]})
    assert GroupAgg(key="k", aggs={"v": ["median"]}).fit([write_parquet(tmp_path / "pair.parquet", pair)]).values_ == {
        "v_median": [statistics.median([-1e9, 1e9 + 1e-7]), 5e-324]
    }
    # A summed value of 2^900 or more in magnitude is refused up front, so no split can make a sum overflow.
    huge: pa.Table = pa.table({"k": ["x", "x", "x"], "v": [1.7e308, 1.7e308, -1.7e308]})
    for threads in (1, 2):
        with pytest.raises(EngineError, match=r"column 'v': value 1.7e\+308 is 2\^900 or more in magnitude"):
            GroupAgg(key="k", aggs={"v": ["sum"]}).fit(
                [write_parquet(tmp_path / "huge.parquet", huge, row_group_size=1)], threads=threads
            )
    assert GroupAgg(key="k", aggs={"v": ["max"]}).fit([write_parquet(tmp_path / "huge.parquet", huge)]).values_ == {
        "v_max": [1.7e308]
    }
    # Signed zeros collapse to one zero, so min, max, and median cannot depend on which worker saw which row.
    zeros: pa.Table = pa.table({"k": ["x", "x", "x"], "v": [0.0, -0.0, 0.0]})
    path = write_parquet(tmp_path / "zeros.parquet", zeros, row_group_size=1)
    for threads in (1, 3):
        table_: dict[str, list[Any]] = (
            GroupAgg(key="k", aggs={"v": ["min", "max", "median"]}).fit([path], threads=threads).values_
        )
        assert table_ == {"v_max": [0.0], "v_median": [0.0], "v_min": [0.0]}
        assert all(math.copysign(1.0, column[0]) == 1.0 for column in table_.values())


def test_group_agg_serves_from_the_key_alone_counts_any_column_and_checks_keys(tmp_path: Path) -> None:
    table: pa.Table = pa.table(
        {
            "user": ["a", "b", "a"],
            "price": [1.0, 2.0, 5.0],
            "day": pa.array([1, None, 3], pa.date32()),
        }
    )
    path: str = write_parquet(tmp_path / "events.parquet", table)
    step: GroupAgg = GroupAgg(key="user", aggs={"price": ["sum"], "day": ["count"]}).fit([path])
    assert step.values_ == {"price_sum": [6.0, 2.0], "day_count": [2, 0]}
    # A count-only column keeps its own type in the state (date32[day] here), which reloads.
    assert pickle.loads(pickle.dumps(step))._state()["column_types"]["day"] == "date32[day]"
    with pytest.raises(EngineError, match=r"column 'day' is date32\[day\]; only count applies"):
        GroupAgg(key="user", aggs={"day": ["max"]}).fit([path])

    # At transform only the key is read: a key-only file works, like a key-only record.
    keys_only: str = write_parquet(tmp_path / "keys.parquet", pa.table({"user": ["b", "zzz"]}))
    assert read_table(step.transform_parquet([keys_only], output_dir=tmp_path / "out")).to_pylist() == [
        {"user": "b", "price_sum": 2.0, "day_count": 0},
        {"user": "zzz", "price_sum": 0.0, "day_count": 0},
    ]

    # Keys must be valid UTF-8, because the table is JSON.
    bad: pa.Table = pa.table(
        {"user": pa.array([b"\xff\xfe", b"ok"], pa.binary()).cast(pa.string(), safe=False), "price": [1.0, 2.0]}
    )
    with pytest.raises(EngineError, match="not valid UTF-8"):
        GroupAgg(key="user", aggs={"price": ["sum"]}).fit([write_parquet(tmp_path / "bad.parquet", bad)])


def test_group_agg_matches_python_and_is_independent_of_threads(tmp_path: Path) -> None:
    rng: np.random.Generator = np.random.default_rng(7)
    keys: list[str] = [f"k{i}" for i in range(40)]
    rows: int = 6000
    users: list[str] = [keys[i] for i in rng.integers(0, len(keys), rows)]
    # Magnitudes spread over many orders so naive summation would depend on the order of addition.
    scales: np.ndarray = np.array([1e-9, 1.0, 1e6, -1e6, 3.25, 1e12])
    amounts: list[float] = (scales[rng.integers(0, len(scales), rows)] * rng.random(rows)).tolist()
    counts: list[int] = rng.integers(-1000, 1001, rows).tolist()
    table: pa.Table = pa.table({"user": users, "amount": amounts, "n": counts})
    path: str = write_parquet(tmp_path / "random.parquet", table, row_group_size=97)

    one: GroupAgg = GroupAgg(key="user", aggs={"amount": ALL, "n": ALL}).fit([path], threads=1)
    many: GroupAgg = GroupAgg(key="user", aggs={"amount": ALL, "n": ALL}).fit([path], threads=8)
    handle_one, handle_many = one._require_handle(), many._require_handle()
    assert handle_one.stage_state("groupagg") == handle_many.stage_state("groupagg")

    by_key: dict[str, tuple[list[float], list[int]]] = {key: ([], []) for key in keys}
    for user, amount, n in zip(users, amounts, counts, strict=True):
        by_key[user][0].append(amount)
        by_key[user][1].append(n)
    assert one.keys_ == sorted(keys)
    values: dict[str, list[Any]] = one.values_
    for i, key in enumerate(one.keys_):
        amount_values, n_values = by_key[key]
        assert values["amount_count"][i] == len(amount_values)
        assert values["amount_sum"][i] == pytest.approx(math.fsum(amount_values), rel=1e-12)
        assert values["amount_mean"][i] == pytest.approx(math.fsum(amount_values) / len(amount_values), rel=1e-12)
        assert values["amount_min"][i] == min(amount_values)
        assert values["amount_max"][i] == max(amount_values)
        assert values["amount_median"][i] == pytest.approx(statistics.median(amount_values), rel=1e-12)
        assert values["n_sum"][i] == sum(n_values)
        assert values["n_mean"][i] == pytest.approx(sum(n_values) / len(n_values), rel=1e-12)
        assert values["n_min"][i] == min(n_values)
        assert values["n_max"][i] == max(n_values)
        assert values["n_median"][i] == statistics.median(n_values)

    # The batch path and the record path agree row for row.
    out: list[dict[str, Any]] = read_table(one.transform_parquet([path], output_dir=tmp_path / "out")).to_pylist()
    for row in out[:50]:  # the record's own amount and n are replaced whatever they hold
        assert one.transform_dict({"user": row["user"], "amount": 1.0, "n": 2}) == row
