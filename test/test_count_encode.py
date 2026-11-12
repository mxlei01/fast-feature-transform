"""Tests for the count_encode kernel: tallies, null and unknown handling, group folding, and the artifact."""

import json
import math
import pickle
import re
import time
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fast_feature_transform import CountEncoder, EngineError, Pipeline, _core
from helpers import read_rows, write_parquet


@pytest.fixture
def events(tmp_path: Path) -> list[str]:
    """One file with three row groups, holding repeats and a null.

    Args:
        tmp_path: pytest-provided temporary directory.

    Returns:
        Paths of the written Parquet files.

    """
    table: pa.Table = pa.table({"device": ["ios", "web", "ios", None, "tv", "ios"], "price": [1.0] * 6})
    return [write_parquet(tmp_path / "events.parquet", table, row_group_size=2)]


def test_count_encode_tallies_categories_and_keeps_the_column_type_explicit(events: list[str], tmp_path: Path) -> None:
    encoder: CountEncoder = CountEncoder(columns=["device"]).fit(events)
    assert encoder.counts_ == {"device": {"ios": 3, "tv": 1, "web": 1}}
    table: pa.Table = pa.concat_tables(
        [pq.read_table(part) for part in encoder.transform_parquet(events, output_dir=tmp_path / "out")]
    )
    assert table.schema.field("device").type == pa.int64()
    assert table.column("device").to_pylist() == [3, 1, 3, 1, 1, 3]  # the fit saw one null, so a null encodes as 1
    assert encoder.transform_dict({"device": "ios", "price": 1.0}) == {"device": 3, "price": 1.0}
    assert encoder.transform_dict({"device": None}) == {"device": 1}
    assert encoder.transform_dict({"device": "nobody"}) == {"device": 0}

    ratios: CountEncoder = CountEncoder(columns=["device"], normalize=True).fit(events)
    rows: list[dict[str, Any]] = read_rows(ratios.transform_parquet(events, output_dir=tmp_path / "ratio"))
    assert rows[0]["device"] == pytest.approx(0.5)
    assert ratios.transform_dict({"device": "nobody"}) == {"device": 0.0}


def test_count_encode_null_and_unknown_options(events: list[str], tmp_path: Path) -> None:
    keep_null: CountEncoder = CountEncoder(columns=["device"], handle_missing="return_nan").fit(events)
    assert keep_null.transform_dict({"device": None}) == {"device": None}
    assert keep_null.counts_ == {"device": {"ios": 3, "tv": 1, "web": 1}}

    filled: CountEncoder = CountEncoder(columns=["device"], handle_unknown=-1).fit(events)
    assert filled.transform_dict({"device": "nobody"}) == {"device": -1}

    as_null: CountEncoder = CountEncoder(columns=["device"], handle_unknown="return_nan").fit(events)
    assert as_null.transform_dict({"device": "nobody"}) == {"device": None}
    assert read_rows(as_null.transform_parquet(events, output_dir=tmp_path / "out"))[0]["device"] == 3

    strict: CountEncoder = CountEncoder(columns=["device"], handle_unknown="error").fit(events)
    with pytest.raises(EngineError, match="category that was not seen at fit"):
        strict.transform_dict({"device": "nobody"})

    nulls: str = write_parquet(tmp_path / "nulls.parquet", pa.table({"device": ["ios", None]}))
    with pytest.raises(EngineError, match="has nulls and handle_missing is 'error'"):
        CountEncoder(columns=["device"], handle_missing="error").fit([nulls])


def test_count_encode_folds_small_groups(tmp_path: Path) -> None:
    table: pa.Table = pa.table({"device": ["ios"] * 6 + ["web", "tv", "mob", "mob"]})
    path: str = write_parquet(tmp_path / "rare.parquet", table)
    folded: CountEncoder = CountEncoder(columns=["device"], min_group_size=3).fit([path])
    assert folded.counts_ == {"device": {"ios": 6, "mob_tv_web": 4}}
    assert folded.transform_dict({"device": "web"}) == {"device": 4}
    assert folded.transform_dict({"device": "ios"}) == {"device": 6}

    named: CountEncoder = CountEncoder(columns=["device"], min_group_size=0.3, min_group_name="other").fit([path])
    assert named.counts_["device"] == {"ios": 6, "other": 4}
    # A threshold nothing falls under leaves every category alone.
    intact: CountEncoder = CountEncoder(columns=["device"], min_group_size=1).fit([path])
    assert intact.counts_["device"] == {"ios": 6, "mob": 2, "tv": 1, "web": 1}


# REGRESSION: the default group name is the members joined with "_", so a
# column holding "x", "y" and "x_y" produced a group named like a category it
# did not fold. Both landed in the same table and the loser was dropped
# silently, so "x_y" served the wrong number — baked into the manifest, so it
# survived save and load. category_encoders overwrites in the same situation
# and loses the real count instead; we refuse rather than pick a winner.
def test_count_encode_refuses_a_group_named_like_a_surviving_category(tmp_path: Path) -> None:
    path: str = write_parquet(
        tmp_path / "collide.parquet",
        pa.table({"device": ["x", "x", "y", "y"] + ["x_y"] * 3 + ["keep"] * 10}),
    )
    with pytest.raises(EngineError, match="the folded group is named 'x_y', which is already a category"):
        CountEncoder(columns=["device"], min_group_size=3).fit([path])
    with pytest.raises(EngineError, match="the folded group is named 'keep', which is already a category"):
        CountEncoder(columns=["device"], min_group_size=3, min_group_name="keep").fit([path])


# REGRESSION: the default group name skipped the separator after an empty
# member, so "" and "x" folded into "x" where category_encoders' "_".join
# makes "_x". counts_ then disagreed with upstream's mapping, a surviving "_x"
# kept its own count instead of being refused as a collision, and a category
# named like the shortened name was refused where category_encoders fits it:
# "" and the null folded into "nan" beside a literal "nan".
def test_count_encode_keeps_the_separator_after_an_empty_member(tmp_path: Path) -> None:
    path: str = write_parquet(tmp_path / "empty.parquet", pa.table({"device": ["", "x"] + ["keep"] * 3}))
    folded: CountEncoder = CountEncoder(columns=["device"], min_group_size=2).fit([path])
    assert folded.counts_ == {"device": {"_x": 2, "keep": 3}}
    assert folded.transform_dict({"device": ""}) == {"device": 2}

    collide: str = write_parquet(
        tmp_path / "collide.parquet", pa.table({"device": ["", "x"] + ["_x"] * 2 + ["keep"] * 3})
    )
    with pytest.raises(EngineError, match="the folded group is named '_x', which is already a category"):
        CountEncoder(columns=["device"], min_group_size=2).fit([collide])

    text: str = write_parquet(tmp_path / "text.parquet", pa.table({"device": ["", None] + ["nan"] * 3 + ["keep"] * 3}))
    beside: CountEncoder = CountEncoder(columns=["device"], min_group_size=2).fit([text])
    assert beside.counts_ == {"device": {"_nan": 2, "keep": 3, "nan": 3}}
    assert beside.transform_dict({"device": None}) == {"device": 2}
    assert beside.transform_dict({"device": "nan"}) == {"device": 3}


# REGRESSION: normalize divides a count by the fitted row count, but an
# integer handle_unknown is the literal number the user asked to mark unknowns
# with. It used to be divided too, which for a large fit set is
# indistinguishable from a real small count. Both paths went through the same
# lookup, so the batch and record answers agreed with each other and were both
# wrong; the parity fixture had no case combining the three.
def test_count_encode_does_not_normalize_an_integer_unknown_fill(tmp_path: Path) -> None:
    train: str = write_parquet(tmp_path / "train.parquet", pa.table({"device": ["ios", "ios", "web", "web"]}))
    serve: str = write_parquet(tmp_path / "serve.parquet", pa.table({"device": ["ios", "nobody", None]}))
    encoder: CountEncoder = CountEncoder(columns=["device"], normalize=True, handle_unknown=7).fit([train])
    values: list[Any] = [
        row["device"] for row in read_rows(encoder.transform_parquet([serve], output_dir=tmp_path / "out"))
    ]
    assert values == [0.5, 7.0, 7.0]
    assert values == [encoder.transform_dict({"device": value})["device"] for value in ["ios", "nobody", None]]


def test_count_encode_is_independent_of_the_split_and_survives_the_artifact(events: list[str], tmp_path: Path) -> None:
    states: set[str] = set()
    for threads in (1, 4):
        encoder: CountEncoder = CountEncoder(columns=["device"]).fit(events, threads=threads)
        states.add(encoder._require_handle().stage_state("countencoder"))
    assert len(states) == 1

    pipeline: Pipeline = Pipeline([("counts", CountEncoder(columns=["device"]))]).fit(events)
    pipeline.save(tmp_path / "artifact")
    manifest: dict[str, Any] = json.loads((tmp_path / "artifact" / "manifest.json").read_text())
    state: dict[str, Any] = manifest["ops"][0]["state"]
    assert state["counts"] == {"device": {"ios": 3, "tv": 1, "web": 1}}
    assert state["null_counts"] == {"device": 1}
    assert state["rows"] == 6
    record: dict[str, Any] = {"device": "ios", "price": 1.0}
    assert Pipeline.load(tmp_path / "artifact").transform_dict(record) == pipeline.transform_dict(record)
    assert pickle.loads(pickle.dumps(pipeline)).transform_dict(record) == pipeline.transform_dict(record)

    manifest["ops"][0]["state"]["counts"]["device"]["ios"] = -1
    with pytest.raises(EngineError, match="must be a count"):
        _core.FittedPipeline.load_json(json.dumps(manifest))


def test_count_encode_validates_its_arguments(events: list[str]) -> None:
    bad_option: Any = "sometimes"
    text: Any = "3"
    with pytest.raises(ValueError, match="handle_unknown must be"):
        CountEncoder(columns=["device"], handle_unknown=bad_option)
    with pytest.raises(ValueError, match="handle_missing must be"):
        CountEncoder(columns=["device"], handle_missing=bad_option)
    # The kernel takes a bool or 'force' and nothing equal to one: 1, 1.0 and
    # numpy's True used to pass here and fail at fit.
    options: tuple[Any, ...] = (bad_option, 1, 1.0, np.True_)
    for option in options:
        with pytest.raises(ValueError, match="combine_min_nan_groups must be"):
            CountEncoder(columns=["device"], combine_min_nan_groups=option)
    with pytest.raises(ValueError, match="combine_min_nan_groups='force' needs handle_missing 'value' or 'error'"):
        CountEncoder(columns=["device"], combine_min_nan_groups="force", handle_missing="return_nan")
    # An integer past int64 is refused here: the kernels read one as an int64.
    with pytest.raises(ValueError, match=r"handle_unknown must be an integer from -2\*\*63 to 2\*\*63 - 1"):
        CountEncoder(columns=["device"], handle_unknown=2**63)
    with pytest.raises(ValueError, match=r"min_group_size must be an integer from -2\*\*63 to 2\*\*63 - 1"):
        CountEncoder(columns=["device"], min_group_size=2**63)
    with pytest.raises(TypeError, match="min_group_size must be a number"):
        CountEncoder(columns=["device"], min_group_size=text)
    for size in (0, math.nan, math.inf):
        with pytest.raises(ValueError, match="min_group_size must be a finite number > 0"):
            CountEncoder(columns=["device"], min_group_size=size)
    with pytest.raises(TypeError, match="normalize must be"):
        CountEncoder(columns=["device"], normalize=text)
    numbers: pa.Table = pa.table({"n": [1, 2]})
    with pytest.raises(EngineError, match="no utf8 columns to count-encode"):
        CountEncoder().fit([write_parquet(Path(events[0]).parent / "numbers.parquet", numbers)])


# The fit data of the folding tests below. With a threshold of 3 the one null
# of "c" is below it and the three of "d" are not, so combine_min_nan_groups
# True, False and "force" fold the two columns in three different ways.
_FOLD_DATA: dict[str, list[str | None]] = {
    "c": ["a", "b", "", "keep", "keep", "keep", "keep", None, "keep"],
    "d": ["x", "y", None, "z", "z", "z", None, None, "z"],
}
_FOLD_PROBES: dict[str, list[str | None]] = {
    "c": ["a", "b", "", "keep", None, "unseen", "_a_b", "rare"],
    "d": ["x", "y", "z", None, "unseen", "x_y", "a", "rare"],
}


class _Fold(NamedTuple):
    """One way of folding ``_FOLD_DATA``: the state it fits and the count each of ``_FOLD_PROBES`` answers with."""

    options: dict[str, Any]
    counts: dict[str, dict[str, int]]
    null_counts: dict[str, int]
    groups: dict[str, dict[str, list[str]]]  # as saved now: a group's name and its members
    groups_0_0_1: dict[str, dict[str, str]]  # as version 0.0.1 saved them: a member and its group's name
    c: list[int]
    d: list[int]


_DEFAULT_NAME: _Fold = _Fold(
    options={"min_group_size": 3},
    counts={"c": {"_a_b_nan": 4, "keep": 5}, "d": {"x_y": 2, "z": 4}},
    null_counts={"c": 4, "d": 3},
    groups={"c": {"_a_b_nan": ["", "a", "b"]}, "d": {"x_y": ["x", "y"]}},
    groups_0_0_1={"c": {"": "_a_b_nan", "a": "_a_b_nan", "b": "_a_b_nan"}, "d": {"x": "x_y", "y": "x_y"}},
    c=[4, 4, 4, 5, 4, 0, 0, 0],
    d=[2, 2, 4, 3, 0, 2, 0, 0],
)
_FORCED: _Fold = _Fold(
    options={"min_group_size": 3, "combine_min_nan_groups": "force"},
    counts={"c": {"_a_b_nan": 4, "keep": 5}, "d": {"nan_x_y": 5, "z": 4}},
    null_counts={"c": 4, "d": 5},
    groups={"c": {"_a_b_nan": ["", "a", "b"]}, "d": {"nan_x_y": ["x", "y"]}},
    groups_0_0_1={"c": {"": "_a_b_nan", "a": "_a_b_nan", "b": "_a_b_nan"}, "d": {"x": "nan_x_y", "y": "nan_x_y"}},
    c=[4, 4, 4, 5, 4, 0, 0, 0],
    d=[5, 5, 4, 5, 0, 0, 0, 0],
)
_FOLDS: list[_Fold] = [
    _DEFAULT_NAME,
    _Fold(
        options={"min_group_size": 3, "combine_min_nan_groups": False},
        counts={"c": {"_a_b": 3, "keep": 5}, "d": {"x_y": 2, "z": 4}},
        null_counts={"c": 1, "d": 3},
        groups={"c": {"_a_b": ["", "a", "b"]}, "d": {"x_y": ["x", "y"]}},
        groups_0_0_1={"c": {"": "_a_b", "a": "_a_b", "b": "_a_b"}, "d": {"x": "x_y", "y": "x_y"}},
        c=[3, 3, 3, 5, 1, 0, 3, 0],
        d=[2, 2, 4, 3, 0, 2, 0, 0],
    ),
    _FORCED,
    _Fold(
        options={"min_group_size": 3, "min_group_name": "rare"},
        counts={"c": {"keep": 5, "rare": 4}, "d": {"rare": 2, "z": 4}},
        null_counts={"c": 4, "d": 3},
        groups={"c": {"rare": ["", "a", "b"]}, "d": {"rare": ["x", "y"]}},
        groups_0_0_1={"c": {"": "rare", "a": "rare", "b": "rare"}, "d": {"x": "rare", "y": "rare"}},
        c=[4, 4, 4, 5, 4, 0, 0, 4],
        d=[2, 2, 4, 3, 0, 0, 0, 2],
    ),
    # A group named like one of its members: "a" is a member and the group.
    _Fold(
        options={"min_group_size": 3, "min_group_name": "a"},
        counts={"c": {"a": 4, "keep": 5}, "d": {"a": 2, "z": 4}},
        null_counts={"c": 4, "d": 3},
        groups={"c": {"a": ["", "a", "b"]}, "d": {"a": ["x", "y"]}},
        groups_0_0_1={"c": {"": "a", "a": "a", "b": "a"}, "d": {"x": "a", "y": "a"}},
        c=[4, 4, 4, 5, 4, 0, 0, 0],
        d=[2, 2, 4, 3, 0, 0, 2, 0],
    ),
    _DEFAULT_NAME._replace(options={"min_group_size": 3, "normalize": True}),
    _FORCED._replace(options={"min_group_size": 0.3, "normalize": True, "combine_min_nan_groups": "force"}),
]
_FOLD_IDS: list[str] = [json.dumps(fold.options) for fold in _FOLDS]


def _fold_files(tmp_path: Path) -> tuple[str, str]:
    """Write ``_FOLD_DATA`` and ``_FOLD_PROBES`` as Parquet files.

    Args:
        tmp_path: pytest-provided temporary directory.

    Returns:
        The path of the fit data, in three row groups, and the path of the probes.

    """
    train: str = write_parquet(tmp_path / "train.parquet", pa.table(_FOLD_DATA), row_group_size=4)
    return train, write_parquet(tmp_path / "probes.parquet", pa.table(_FOLD_PROBES))


def _fold_state(fold: _Fold, groups: dict[str, Any]) -> dict[str, Any]:
    """Build the state a fold fits, with its groups in one of the two layouts.

    Args:
        fold: The way of folding.
        groups: ``fold.groups`` or ``fold.groups_0_0_1``.

    Returns:
        The state of the count_encode stage.

    """
    return {"columns": ["c", "d"], "counts": fold.counts, "groups": groups, "null_counts": fold.null_counts, "rows": 9}


def _assert_fold_answers(fitted: Pipeline | CountEncoder, fold: _Fold, probes: str, output_dir: Path) -> None:
    """Check that the probes encode to the fold's counts on the batch path and on the record path.

    Args:
        fitted: A fitted pipeline or encoder.
        fold: The way of folding it was fitted with.
        probes: The Parquet file of ``_FOLD_PROBES``.
        output_dir: Where the batch path writes its parts.

    """
    expected: list[dict[str, Any]] = [{"c": c, "d": d} for c, d in zip(fold.c, fold.d, strict=True)]
    if fold.options.get("normalize"):  # divides a count by the 9 fitted rows
        expected = [{"c": row["c"] / 9, "d": row["d"] / 9} for row in expected]
    assert read_rows(fitted.transform_parquet([probes], output_dir=output_dir)) == expected
    records: list[dict[str, Any]] = [
        fitted.transform_dict({"c": c, "d": d}) for c, d in zip(_FOLD_PROBES["c"], _FOLD_PROBES["d"], strict=True)
    ]
    assert records == expected


@pytest.mark.parametrize("fold", _FOLDS, ids=_FOLD_IDS)
def test_count_encode_saves_each_group_once_and_survives_save_load_and_pickle(fold: _Fold, tmp_path: Path) -> None:
    train, probes = _fold_files(tmp_path)
    pipeline: Pipeline = Pipeline([("count", CountEncoder(columns=["c", "d"], **fold.options))]).fit([train])
    pipeline.save(tmp_path / "saved")
    manifest: dict[str, Any] = json.loads((tmp_path / "saved" / "manifest.json").read_text())
    assert manifest["ops"][0]["state"] == _fold_state(fold, fold.groups)
    assert json.loads(pipeline.stages_state()) == {"count": _fold_state(fold, fold.groups)}
    encoder: CountEncoder = CountEncoder(columns=["c", "d"], **fold.options).fit([train])
    assert encoder.counts_ == fold.counts

    fitted: dict[str, Pipeline | CountEncoder] = {
        "fitted": pipeline,
        "loaded": Pipeline.load(tmp_path / "saved"),
        "unpickled": pickle.loads(pickle.dumps(pipeline)),
        "unpickled encoder": pickle.loads(pickle.dumps(encoder)),
    }
    for name, candidate in fitted.items():
        _assert_fold_answers(candidate, fold, probes, tmp_path / name)


# A pipeline that version 0.0.1 saved keeps loading. That version saved a
# member and its group's name where a group's name and its members are saved
# now. Each manifest here is what it wrote for the fit, but for the time in
# created_at: its library_version, schema_version and params, and the groups
# in its layout.
@pytest.mark.parametrize("fold", _FOLDS, ids=_FOLD_IDS)
def test_count_encode_loads_a_pipeline_saved_by_version_0_0_1(fold: _Fold, tmp_path: Path) -> None:
    train, probes = _fold_files(tmp_path)
    params: dict[str, Any] = {
        "columns": ["c", "d"],
        "normalize": False,
        "min_group_size": None,
        "min_group_name": None,
        "combine_min_nan_groups": True,
        "handle_unknown": "value",
        "handle_missing": "value",
    }
    manifest: dict[str, Any] = {
        "created_at": "1999-01-01T00:00:00Z",
        "library_version": "0.0.1",
        "ops": [
            {
                "kernel": "count_encode",
                "params": params | fold.options,
                "stage": "count",
                "state": _fold_state(fold, fold.groups_0_0_1),
            }
        ],
        "schema_version": "0.3.0",
    }
    (tmp_path / "saved").mkdir()
    (tmp_path / "saved" / "manifest.json").write_text(json.dumps(manifest))
    loaded: Pipeline = Pipeline.load(tmp_path / "saved")
    _assert_fold_answers(loaded, fold, probes, tmp_path / "loaded")
    _assert_fold_answers(pickle.loads(pickle.dumps(loaded)), fold, probes, tmp_path / "unpickled")

    # It is the pipeline a fit gives now, and saving it again writes the
    # present layout.
    fitted: Pipeline = Pipeline([("count", CountEncoder(columns=["c", "d"], **fold.options))]).fit([train])
    assert loaded.stages_state() == fitted.stages_state()
    loaded.save(tmp_path / "again")
    again: dict[str, Any] = json.loads((tmp_path / "again" / "manifest.json").read_text())
    assert again["ops"][0]["state"] == _fold_state(fold, fold.groups)
    assert again["library_version"] == "0.0.1"


# REGRESSION: the saved pipeline holds a group's name once, with its members,
# so it grows linearly with the folded categories. It used to hold the name
# once for each member, and the default name joins every member: 2,000 folded
# categories of 8 characters saved 36 MB.
def test_count_encode_saved_pipeline_grows_linearly_with_the_folded_categories(tmp_path: Path) -> None:
    sizes: dict[int, int] = {}
    for rare in (1000, 2000, 4000):
        table: pa.Table = pa.table({"c": [f"cat{index:05d}" for index in range(rare)] + ["big"] * 5000})
        path: str = write_parquet(tmp_path / f"{rare}.parquet", table)
        pipeline: Pipeline = Pipeline([("count", CountEncoder(columns=["c"], min_group_size=2))]).fit([path])
        assert pipeline.transform_dict({"c": "cat00999"}) == {"c": rare}
        pipeline.save(tmp_path / str(rare))
        sizes[rare] = (tmp_path / str(rare) / "manifest.json").stat().st_size
        assert len(pickle.dumps(pipeline)) < sizes[rare] + 1000
        assert len(pipeline.stages_state()) < sizes[rare]
    assert sizes[2000] < 60_000
    # A category's 8 characters are saved three times: twice with a "_" in the
    # group's name, which is a key of the counts and a key of the groups, and
    # once between quotes and before a comma in the list of members.
    each: int = 2 * (8 + 1) + (8 + 3)
    assert sizes[2000] - sizes[1000] == 1000 * each
    assert sizes[4000] - sizes[2000] == 2000 * each


# Loading looks a group's name up once and not once for each member, so the
# load is linear in the members. The default name is as long as the members
# together: with a lookup for each of 128,000 members of 9 characters the load
# took 7.5 s of CPU time, and it takes 18 ms. The bound of one second is 50
# times that.
def test_count_encode_loads_a_large_group_in_time_linear_in_its_members(tmp_path: Path) -> None:
    members: int = 128_000
    table: pa.Table = pa.table({"c": [f"cat{index:06d}" for index in range(members)] + ["big"] * 5})
    path: str = write_parquet(tmp_path / "train.parquet", table)
    Pipeline([("count", CountEncoder(columns=["c"], min_group_size=2))]).fit([path]).save(tmp_path / "saved")
    start: float = time.process_time()
    loaded: Pipeline = Pipeline.load(tmp_path / "saved")
    seconds: float = time.process_time() - start
    assert loaded.transform_dict({"c": "cat127999"}) == {"c": members}
    assert seconds < 1.0, f"the load took {seconds:.2f} s of CPU time"


def test_count_encode_refuses_malformed_groups(tmp_path: Path) -> None:
    train, _ = _fold_files(tmp_path)
    pipeline: Pipeline = Pipeline([("count", CountEncoder(columns=["c", "d"], min_group_size=3))]).fit([train])
    pipeline.save(tmp_path / "saved")
    text: str = (tmp_path / "saved" / "manifest.json").read_text()
    assert json.loads(text)["ops"][0]["state"]["groups"] == _DEFAULT_NAME.groups
    twice: str = "count: groups['c'] folds 'a' more than once"
    no_list: str = "count: groups['c']['_a_b_nan'] must be a non-empty list of member names"
    malformed: list[tuple[Any, str]] = [
        # A member in two groups, and twice in one.
        ({"c": {"_a_b_nan": ["", "a"], "keep": ["a", "b"]}}, twice),
        ({"c": {"_a_b_nan": ["", "a", "b", "a"]}}, twice),
        # One entry in each layout names "a".
        ({"c": {"_a_b_nan": ["", "a", "b"], "a": "keep"}}, twice),
        # A member that is not a string, and a group without a list of members.
        ({"c": {"_a_b_nan": ["", "a", 7]}}, no_list),
        ({"c": {"_a_b_nan": ["", ["a"]]}}, no_list),
        ({"c": {"_a_b_nan": []}}, no_list),
        ({"c": {"_a_b_nan": {"a": "b"}}}, no_list),
        ({"c": {"_a_b_nan": None}}, no_list),
        ({"c": {"_a_b_nan": 3}}, no_list),
        # A column that is not encoded, and groups that are not an object.
        ({"e": {"x_y": ["x", "y"]}}, "count: groups names column 'e', which this encoder does not encode"),
        ({"c": ["", "a", "b"]}, "count: groups['c'] must be an object"),
        (["c"], "count: 'groups' must be an object"),
        # A group that is not a category, and a member that still is one.
        ({"c": {"gone": ["a"]}}, "count: groups['c'] folds 'a' into 'gone', which is not a category of counts['c']"),
        ({"c": {"_a_b_nan": ["keep"]}}, "count: groups['c'] folds 'keep', which is still a category of counts['c']"),
    ]
    for groups, message in malformed:
        manifest: dict[str, Any] = json.loads(text)
        manifest["ops"][0]["state"]["groups"] = groups
        with pytest.raises(EngineError, match=f"^{re.escape(message)}$"):
            _core.FittedPipeline.load_json(json.dumps(manifest))


# A fit folds one group for a column and saves its members sorted. A manifest
# written by hand can hold several groups, their members in any order: it
# loads, and saving it again sorts the members.
def test_count_encode_loads_several_groups_with_members_in_any_order(tmp_path: Path) -> None:
    state: dict[str, Any] = {
        "columns": ["c"],
        "counts": {"c": {"keep": 5, "low": 7, "rare": 3}},
        "groups": {"c": {"rare": ["b", "", "a"], "low": ["m", "l"]}},
        "null_counts": {"c": None},
        "rows": 15,
    }
    manifest: dict[str, Any] = {
        "ops": [{"kernel": "count_encode", "params": {"columns": ["c"]}, "stage": "count", "state": state}],
        "schema_version": "0.3.0",
    }
    (tmp_path / "saved").mkdir()
    (tmp_path / "saved" / "manifest.json").write_text(json.dumps(manifest))
    loaded: Pipeline = Pipeline.load(tmp_path / "saved")
    values: list[str] = ["a", "b", "", "l", "m", "keep", "rare", "low", "unseen"]
    counts: list[int] = [3, 3, 3, 7, 7, 5, 3, 7, 0]
    assert [loaded.transform_dict({"c": value})["c"] for value in values] == counts
    probes: str = write_parquet(tmp_path / "probes.parquet", pa.table({"c": values}))
    assert read_rows(loaded.transform_parquet([probes], output_dir=tmp_path / "out")) == [
        {"c": count} for count in counts
    ]
    state["groups"] = {"c": {"low": ["l", "m"], "rare": ["", "a", "b"]}}
    assert json.loads(loaded.stages_state()) == {"count": state}


def test_count_encode_groups_do_not_depend_on_the_split_or_the_row_order(tmp_path: Path) -> None:
    table: pa.Table = pa.table(_FOLD_DATA)
    states: set[str] = set()
    for name, rows in (("forward", table), ("backward", table.take(list(range(table.num_rows - 1, -1, -1))))):
        for row_group_size in (2, 4, 9):
            path: str = write_parquet(tmp_path / f"{name}{row_group_size}.parquet", rows, row_group_size=row_group_size)
            for threads in (1, 4):
                encoder: CountEncoder = CountEncoder(columns=["c", "d"], min_group_size=3).fit([path], threads=threads)
                states.add(encoder._require_handle().stage_state("countencoder"))
    assert len(states) == 1
    assert json.loads(states.pop()) == _fold_state(_DEFAULT_NAME, _DEFAULT_NAME.groups)
