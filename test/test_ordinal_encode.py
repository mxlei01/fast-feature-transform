"""Tests for the ordinal substrate: ordinal_encode, one_hot_encode and rank_hot_encode."""

import json
import pickle
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest

from fast_feature_transform import (
    CountEncoder,
    EngineError,
    OneHotEncoder,
    OrdinalEncoder,
    Pipeline,
    RankHotEncoder,
    _core,
)
from helpers import read_table, write_parquet


@pytest.fixture
def colors(tmp_path: Path) -> list[str]:
    """One file with three row groups, categories out of order, and a null.

    Args:
        tmp_path: pytest-provided temporary directory.

    Returns:
        Paths of the written Parquet files.

    """
    table: pa.Table = pa.table({"color": ["cyan", "amber", None, "blue", "amber", "cyan"], "size": ["l"] * 6})
    return [write_parquet(tmp_path / "colors.parquet", table, row_group_size=2)]


def test_ordinal_encode_numbers_sorted_categories_with_the_null_last(colors: list[str], tmp_path: Path) -> None:
    encoder: OrdinalEncoder = OrdinalEncoder(columns=["color"]).fit(colors)
    assert encoder.categories_ == {"color": ["amber", "blue", "cyan"]}
    rows: list[dict[str, Any]] = read_table(encoder.transform_parquet(colors, output_dir=tmp_path / "out")).to_pylist()
    assert [row["color"] for row in rows] == [3, 1, 4, 2, 1, 3]  # the null was seen at fit, so it takes the last code
    assert encoder.transform_dict({"color": "amber", "size": "l"}) == {"color": 1, "size": "l"}
    assert encoder.transform_dict({"color": "zulu"}) == {"color": -1}

    from_zero: OrdinalEncoder = OrdinalEncoder(columns=["color"], index_start=0).fit(colors)
    assert from_zero.transform_dict({"color": "amber"}) == {"color": 0}
    # REGRESSION: an index_start past int64 is refused; 2**63 used to reach the
    # kernel wrapped to -2**63, which gave negative codes.
    with pytest.raises(ValueError, match=r"index_start must be an integer from -2\*\*63 to 2\*\*63 - 1"):
        OrdinalEncoder(columns=["color"], index_start=2**63)

    plain: str = write_parquet(tmp_path / "plain.parquet", pa.table({"color": ["amber", "blue"]}))
    unseen_null: OrdinalEncoder = OrdinalEncoder(columns=["color"]).fit([plain])
    assert unseen_null.transform_dict({"color": None}) == {"color": -2}  # never a fitted category


def test_ordinal_encode_null_and_unknown_options(colors: list[str], tmp_path: Path) -> None:
    as_null: OrdinalEncoder = OrdinalEncoder(columns=["color"], handle_missing="return_nan").fit(colors)
    assert as_null.transform_dict({"color": None}) == {"color": None}
    assert as_null.categories_ == {"color": ["amber", "blue", "cyan"]}

    unknown_null: OrdinalEncoder = OrdinalEncoder(columns=["color"], handle_unknown="return_nan").fit(colors)
    assert unknown_null.transform_dict({"color": "zulu"}) == {"color": None}

    strict: OrdinalEncoder = OrdinalEncoder(columns=["color"], handle_unknown="error").fit(colors)
    with pytest.raises(EngineError, match="category that was not seen at fit"):
        strict.transform_dict({"color": "zulu"})
    with pytest.raises(EngineError, match="has nulls and handle_missing is 'error'"):
        OrdinalEncoder(columns=["color"], handle_missing="error").fit(colors)


def test_one_hot_and_rank_hot_expand_the_column(colors: list[str], tmp_path: Path) -> None:
    hot: OneHotEncoder = OneHotEncoder(columns=["color"]).fit(colors)
    table: pa.Table = read_table(hot.transform_parquet(colors, output_dir=tmp_path / "hot"))
    # The source column is replaced; the null category is sorted last and gets its own column.
    assert table.column_names == ["size", "color_1", "color_2", "color_3", "color_4"]
    assert table.to_pylist()[0] == {"size": "l", "color_1": 0, "color_2": 0, "color_3": 1, "color_4": 0}
    assert table.to_pylist()[2] == {"size": "l", "color_1": 0, "color_2": 0, "color_3": 0, "color_4": 1}
    assert hot.transform_dict({"color": "amber", "size": "l"}) == table.to_pylist()[1]
    assert hot.transform_dict({"color": "zulu"}) == {"color_1": 0, "color_2": 0, "color_3": 0, "color_4": 0}

    named: OneHotEncoder = OneHotEncoder(columns=["color"], use_cat_names=True).fit(colors)
    assert read_table(named.transform_parquet(colors, output_dir=tmp_path / "named")).column_names == [
        "size",
        "color_amber",
        "color_blue",
        "color_cyan",
        "color_nan",
    ]

    ranked: RankHotEncoder = RankHotEncoder(columns=["color"]).fit(colors)
    ranks: list[dict[str, Any]] = read_table(ranked.transform_parquet(colors, output_dir=tmp_path / "rank")).to_pylist()
    assert ranks[1] == {"size": "l", "color_1": 1, "color_2": 0, "color_3": 0, "color_4": 0}  # amber, rank 1
    assert ranks[0] == {"size": "l", "color_1": 1, "color_2": 1, "color_3": 1, "color_4": 0}  # cyan, rank 3
    assert ranked.transform_dict({"color": "cyan", "size": "l"}) == ranks[0]
    with pytest.raises(ValueError, match="handle_unknown must be one of"):
        RankHotEncoder(columns=["color"], handle_unknown="indicator")

    indicator: OneHotEncoder = OneHotEncoder(columns=["color"], handle_unknown="indicator").fit(colors)
    assert indicator.transform_dict({"color": "zulu"})["color_-1"] == 1
    assert indicator.transform_dict({"color": "amber"})["color_-1"] == 0


# REGRESSION: category_encoders' RankHotEncoder has no mapping row for a null
# it never fitted, so it falls through to the same default row an unseen
# category takes and handle_unknown decides it. Ours answered all zeros
# whatever handle_unknown said, so under "return_nan" a serving null came out
# as zeros instead of nulls. OneHotEncoder does have that row, and is right to
# keep giving zeros.
def test_rank_hot_hands_a_null_it_never_fitted_to_handle_unknown(tmp_path: Path) -> None:
    plain: list[str] = [write_parquet(tmp_path / "plain.parquet", pa.table({"color": ["amber", "blue"]}))]
    late: list[str] = [write_parquet(tmp_path / "late.parquet", pa.table({"color": ["amber", None]}))]

    ranked: RankHotEncoder = RankHotEncoder(columns=["color"], handle_unknown="return_nan").fit(plain)
    rows: list[dict[str, Any]] = read_table(ranked.transform_parquet(late, output_dir=tmp_path / "rank")).to_pylist()
    assert rows == [{"color_1": 1, "color_2": 0}, {"color_1": None, "color_2": None}]
    assert ranked.transform_dict({"color": None}) == rows[1]

    # 'error' refuses an unseen category, never a null, and 'value' zeroes it.
    strict: RankHotEncoder = RankHotEncoder(columns=["color"], handle_unknown="error").fit(plain)
    assert strict.transform_dict({"color": None}) == {"color_1": 0, "color_2": 0}
    with pytest.raises(EngineError, match="category that was not seen at fit"):
        strict.transform_dict({"color": "zulu"})
    assert RankHotEncoder(columns=["color"]).fit(plain).transform_dict({"color": None}) == {"color_1": 0, "color_2": 0}

    hot: OneHotEncoder = OneHotEncoder(columns=["color"], handle_unknown="return_nan").fit(plain)
    assert hot.transform_dict({"color": None}) == {"color_1": 0, "color_2": 0}


# REGRESSION: category_encoders' ordinal pre-encoder keeps a slot for the
# missing value, and under handle_missing="return_nan" RankHotEncoder leaves
# that slot in as a level of its own: an extra trailing column named after
# the -2 code, which nothing ever lights up. We dropped it, so our frame was a
# column narrower than the library's. OneHotEncoder drops it upstream too.
def test_rank_hot_keeps_the_placeholder_column_under_return_nan(colors: list[str], tmp_path: Path) -> None:
    ranked: RankHotEncoder = RankHotEncoder(columns=["color"], handle_missing="return_nan").fit(colors)
    table: pa.Table = read_table(ranked.transform_parquet(colors, output_dir=tmp_path / "rank"))
    assert table.column_names == ["size", "color_1", "color_2", "color_3", "color_-2"]
    rows: list[dict[str, Any]] = table.to_pylist()
    assert rows[1] == {"size": "l", "color_1": 1, "color_2": 0, "color_3": 0, "color_-2": 0}  # amber
    assert rows[0] == {"size": "l", "color_1": 1, "color_2": 1, "color_3": 1, "color_-2": 0}  # cyan
    assert rows[2] == dict.fromkeys(("color_1", "color_2", "color_3", "color_-2")) | {"size": "l"}  # the null
    assert ranked.transform_dict({"color": "amber", "size": "l"}) == rows[1]
    assert ranked.transform_dict({"color": None, "size": "l"}) == rows[2]

    named: RankHotEncoder = RankHotEncoder(columns=["color"], handle_missing="return_nan", use_cat_names=True)
    assert read_table(named.fit(colors).transform_parquet(colors, output_dir=tmp_path / "named")).column_names == [
        "size",
        "color_amber",
        "color_blue",
        "color_cyan",
        "color_nan",
    ]

    # The column list is rebuilt from the loaded table, not stored, so the
    # artifact and the pickle have to produce it again.
    pipeline: Pipeline = Pipeline([("rank", RankHotEncoder(columns=["color"], handle_missing="return_nan"))])
    pipeline.fit(colors)
    pipeline.save(tmp_path / "artifact")
    record: dict[str, Any] = {"color": "amber", "size": "l"}
    assert Pipeline.load(tmp_path / "artifact").transform_dict(record) == rows[1]
    assert pickle.loads(pickle.dumps(pipeline)).transform_dict(record) == rows[1]

    # handle_missing='error' refits the pre-encoder strictly: no slot, no column.
    plain: list[str] = [write_parquet(tmp_path / "plain.parquet", pa.table({"color": ["amber", "blue"]}))]
    assert RankHotEncoder(columns=["color"], handle_missing="error").fit(plain).transform_dict({"color": "amber"}) == {
        "color_1": 1,
        "color_2": 0,
    }
    hot: OneHotEncoder = OneHotEncoder(columns=["color"], handle_missing="return_nan").fit(colors)
    assert hot.transform_dict({"color": "amber", "size": "l"}) == {
        "size": "l",
        "color_1": 1,
        "color_2": 0,
        "color_3": 0,
    }


# REGRESSION: the "#" that disambiguates a repeated output name was scoped to
# one column, so encoding "a" (holding "b_c") beside "a_b" (holding "c")
# emitted "a_b_c" twice. The batch path wrote a Parquet part with a duplicate
# column name, unreadable and found only when somebody later opened it, while
# the record path silently returned one feature fewer, so a model trained from
# the parts and a model served from records had different feature counts.
# category_encoders' counter spans every encoded column; ours does now too.
def test_one_hot_disambiguates_a_name_that_repeats_across_columns(tmp_path: Path) -> None:
    path: str = write_parquet(tmp_path / "collide.parquet", pa.table({"a": ["b_c", "d"], "a_b": ["c", "e"]}))
    hot: OneHotEncoder = OneHotEncoder(columns=["a", "a_b"], use_cat_names=True).fit([path])
    table: pa.Table = read_table(hot.transform_parquet([path], output_dir=tmp_path / "out"))
    assert table.column_names == ["a_b_c", "a_d", "a_b_c#", "a_b_e"]
    rows: list[dict[str, Any]] = table.to_pylist()
    assert rows[0] == {"a_b_c": 1, "a_d": 0, "a_b_c#": 1, "a_b_e": 0}
    # The record path emits the same four keys, not three.
    assert hot.transform_dict({"a": "b_c", "a_b": "c"}) == rows[0]


# REGRESSION: the batch path refuses a pass-through column named like a
# generated one at resolve; the record path used to keep both and let the
# generated value win, so the same data was a hard error through
# transform_parquet and a silent value substitution through transform_dict.
def test_one_hot_refuses_a_record_key_that_collides_with_a_generated_column(colors: list[str]) -> None:
    hot: OneHotEncoder = OneHotEncoder(columns=["color"]).fit(colors)
    with pytest.raises(EngineError, match="key 'color_1' already present in record"):
        hot.transform_dict({"color": "amber", "color_1": 99})


def test_a_later_step_sees_the_schema_the_expansion_produced(colors: list[str], tmp_path: Path) -> None:
    # count_encode picks its columns from the schema at fit time, so it must
    # see one_hot's output: color is gone by then, size is still utf8.
    pipeline: Pipeline = Pipeline([("hot", OneHotEncoder(columns=["color"])), ("counts", CountEncoder())]).fit(colors)
    counts = pipeline["counts"]
    assert isinstance(counts, CountEncoder)
    assert counts.counts_ == {"size": {"l": 6}}
    row: dict[str, Any] = pipeline.transform_dict({"color": "amber", "size": "l"})
    assert row == {"size": 6, "color_1": 1, "color_2": 0, "color_3": 0, "color_4": 0}
    assert read_table(pipeline.transform_parquet(colors, output_dir=tmp_path / "out")).to_pylist()[1] == row


def test_the_substrate_is_independent_of_the_split_and_survives_the_artifact(colors: list[str], tmp_path: Path) -> None:
    for cls in (OrdinalEncoder, OneHotEncoder, RankHotEncoder):
        states: set[str] = set()
        for threads in (1, 4):
            encoder = cls(columns=["color"]).fit(colors, threads=threads)
            states.add(encoder._require_handle().stage_state(cls.__name__.lower()))
        assert len(states) == 1, cls.__name__

    pipeline: Pipeline = Pipeline([("hot", OneHotEncoder(columns=["color"]))]).fit(colors)
    pipeline.save(tmp_path / "artifact")
    manifest: dict[str, Any] = json.loads((tmp_path / "artifact" / "manifest.json").read_text())
    assert manifest["ops"][0]["state"] == {
        "columns": ["color"],
        "categories": {"color": ["amber", "blue", "cyan"]},
        "has_null": {"color": True},
    }
    record: dict[str, Any] = {"color": "blue", "size": "l"}
    assert Pipeline.load(tmp_path / "artifact").transform_dict(record) == pipeline.transform_dict(record)
    assert pickle.loads(pickle.dumps(pipeline)).transform_dict(record) == pipeline.transform_dict(record)

    manifest["ops"][0]["state"]["categories"]["color"] = ["cyan", "amber"]
    with pytest.raises(EngineError, match="must be sorted and unique"):
        _core.FittedPipeline.load_json(json.dumps(manifest))
