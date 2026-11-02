"""Tests for the count_encode kernel: tallies, null and unknown handling, group folding, and the artifact."""

import json
import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fast_feature_transform import CountEncoder, EngineError, Pipeline, _core


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


def _rows(parts: list[Path]) -> list[dict[str, Any]]:
    """Read output parts back as records.

    Args:
        parts: Part files in order.

    Returns:
        The rows as dicts.

    """
    return pa.concat_tables([pq.read_table(part) for part in parts]).to_pylist()


@pytest.fixture
def events(tmp_path: Path) -> list[str]:
    """One file with three row groups, holding repeats and a null.

    Args:
        tmp_path: pytest-provided temporary directory.

    Returns:
        Paths of the written Parquet files.

    """
    table: pa.Table = pa.table({"device": ["ios", "web", "ios", None, "tv", "ios"], "price": [1.0] * 6})
    return [_write(tmp_path / "events.parquet", table, row_group_size=2)]


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
    rows: list[dict[str, Any]] = _rows(ratios.transform_parquet(events, output_dir=tmp_path / "ratio"))
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
    assert _rows(as_null.transform_parquet(events, output_dir=tmp_path / "out"))[0]["device"] == 3

    strict: CountEncoder = CountEncoder(columns=["device"], handle_unknown="error").fit(events)
    with pytest.raises(EngineError, match="category that was not seen at fit"):
        strict.transform_dict({"device": "nobody"})

    nulls: str = _write(tmp_path / "nulls.parquet", pa.table({"device": ["ios", None]}))
    with pytest.raises(EngineError, match="has nulls and handle_missing is 'error'"):
        CountEncoder(columns=["device"], handle_missing="error").fit([nulls])


def test_count_encode_folds_small_groups(tmp_path: Path) -> None:
    table: pa.Table = pa.table({"device": ["ios"] * 6 + ["web", "tv", "mob", "mob"]})
    path: str = _write(tmp_path / "rare.parquet", table)
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
    path: str = _write(
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
    path: str = _write(tmp_path / "empty.parquet", pa.table({"device": ["", "x"] + ["keep"] * 3}))
    folded: CountEncoder = CountEncoder(columns=["device"], min_group_size=2).fit([path])
    assert folded.counts_ == {"device": {"_x": 2, "keep": 3}}
    assert folded.transform_dict({"device": ""}) == {"device": 2}

    collide: str = _write(tmp_path / "collide.parquet", pa.table({"device": ["", "x"] + ["_x"] * 2 + ["keep"] * 3}))
    with pytest.raises(EngineError, match="the folded group is named '_x', which is already a category"):
        CountEncoder(columns=["device"], min_group_size=2).fit([collide])

    text: str = _write(tmp_path / "text.parquet", pa.table({"device": ["", None] + ["nan"] * 3 + ["keep"] * 3}))
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
    train: str = _write(tmp_path / "train.parquet", pa.table({"device": ["ios", "ios", "web", "web"]}))
    serve: str = _write(tmp_path / "serve.parquet", pa.table({"device": ["ios", "nobody", None]}))
    encoder: CountEncoder = CountEncoder(columns=["device"], normalize=True, handle_unknown=7).fit([train])
    values: list[Any] = [
        row["device"] for row in _rows(encoder.transform_parquet([serve], output_dir=tmp_path / "out"))
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
        CountEncoder().fit([_write(Path(events[0]).parent / "numbers.parquet", numbers)])
