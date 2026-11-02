"""Tests for the positional encoders: basen_encode, binary_encode and gray_encode."""

import itertools
import json
import pickle
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fast_feature_transform import (
    BaseNEncoder,
    BinaryEncoder,
    CountEncoder,
    EngineError,
    GrayEncoder,
    Pipeline,
    _core,
)


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


def _table(parts: list[Path]) -> pa.Table:
    """Read output parts back as one table.

    Args:
        parts: Part files in order.

    Returns:
        The concatenated table.

    """
    return pa.concat_tables([pq.read_table(part) for part in parts])


@pytest.fixture
def colors(tmp_path: Path) -> list[str]:
    """One file with three row groups, categories out of order, and a null.

    Args:
        tmp_path: pytest-provided temporary directory.

    Returns:
        Paths of the written Parquet files.

    """
    table: pa.Table = pa.table({"color": ["cyan", "amber", None, "blue", "amber", "cyan"], "size": ["l"] * 6})
    return [_write(tmp_path / "colors.parquet", table, row_group_size=2)]


def test_binary_encode_spells_the_rank_of_every_level(colors: list[str], tmp_path: Path) -> None:
    encoder: BinaryEncoder = BinaryEncoder(columns=["color"]).fit(colors)
    assert encoder.categories_ == {"color": ["amber", "blue", "cyan"]}
    table: pa.Table = _table(encoder.transform_parquet(colors, output_dir=tmp_path / "out"))
    # Three categories plus the fitted null make four levels, and the word 0
    # is reserved for an unknown, so the rank needs three binary digits.
    assert table.column_names == ["size", "color_0", "color_1", "color_2"]
    rows: list[dict[str, Any]] = table.to_pylist()
    assert rows[1] == {"size": "l", "color_0": 0, "color_1": 0, "color_2": 1}  # amber, rank 1
    assert rows[3] == {"size": "l", "color_0": 0, "color_1": 1, "color_2": 0}  # blue, rank 2
    assert rows[0] == {"size": "l", "color_0": 0, "color_1": 1, "color_2": 1}  # cyan, rank 3
    assert rows[2] == {"size": "l", "color_0": 1, "color_1": 0, "color_2": 0}  # the null sorts last, rank 4
    assert encoder.transform_dict({"color": "amber", "size": "l"}) == rows[1]
    # An unknown takes the code word the ranks left free.
    assert encoder.transform_dict({"color": "zulu"}) == {"color_0": 0, "color_1": 0, "color_2": 0}

    plain: str = _write(tmp_path / "plain.parquet", pa.table({"color": ["amber", "blue"]}))
    unseen_null: BinaryEncoder = BinaryEncoder(columns=["color"]).fit([plain])
    assert unseen_null.transform_dict({"color": None}) == {"color_0": 0, "color_1": 0}  # never a fitted level


def test_basen_encode_writes_the_rank_in_its_base(colors: list[str], tmp_path: Path) -> None:
    base3: BaseNEncoder = BaseNEncoder(columns=["color"], base=3).fit(colors)
    rows: list[dict[str, Any]] = _table(base3.transform_parquet(colors, output_dir=tmp_path / "three")).to_pylist()
    assert list(rows[0]) == ["size", "color_0", "color_1"]  # four levels need two base-3 digits
    assert [rows[1]["color_0"], rows[1]["color_1"]] == [0, 1]  # amber, rank 1
    assert [rows[0]["color_0"], rows[0]["color_1"]] == [1, 0]  # cyan, rank 3
    assert [rows[2]["color_0"], rows[2]["color_1"]] == [1, 1]  # the null, rank 4
    assert base3.transform_dict({"color": "cyan", "size": "l"}) == rows[0]

    # base 1 is category_encoders' documented special case: a single 1 at
    # position r, so the leading column is always zero.
    one: BaseNEncoder = BaseNEncoder(columns=["color"], base=1).fit(colors)
    spread: dict[str, Any] = one.transform_dict({"color": "blue"})
    assert spread == {"color_0": 0, "color_1": 0, "color_2": 1, "color_3": 0, "color_4": 0}
    assert one.transform_dict({"color": "zulu"}) == dict.fromkeys(spread, 0)

    # A base above the level count is ordinal encoding in one column.
    wide: BaseNEncoder = BaseNEncoder(columns=["color"], base=10).fit(colors)
    assert wide.transform_dict({"color": "cyan"}) == {"color_0": 3}


def test_gray_encode_changes_one_digit_between_neighbouring_ranks(tmp_path: Path) -> None:
    # Deliberately unsorted: category_encoders sorts the categories for Gray,
    # so this encoder agrees with it on any fit data.
    source: str = _write(tmp_path / "gray.parquet", pa.table({"c": ["e", "c", "a", "d", "b"]}))
    encoder: GrayEncoder = GrayEncoder(columns=["c"]).fit([source])
    assert encoder.categories_ == {"c": ["a", "b", "c", "d", "e"]}
    words: list[list[int]] = [
        [encoder.transform_dict({"c": value})[f"c_{digit}"] for digit in range(3)] for value in "abcde"
    ]
    assert words == [[0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 1, 1]]
    neighbours = itertools.pairwise(words)
    assert all(sum(a != b for a, b in zip(one, other, strict=True)) == 1 for one, other in neighbours)


def test_the_null_and_unknown_options(colors: list[str], tmp_path: Path) -> None:
    as_null: BinaryEncoder = BinaryEncoder(columns=["color"], handle_missing="return_nan").fit(colors)
    assert as_null.categories_ == {"color": ["amber", "blue", "cyan"]}
    assert as_null.transform_dict({"color": None}) == {"color_0": None, "color_1": None, "color_2": None}
    # Upstream sizes the width from the level list of its ordinal pre-encoder,
    # which always keeps a slot for the null, so 'return_nan' still spends the
    # digit that a fourth level would have needed.
    assert as_null.transform_dict({"color": "amber"}) == {"color_0": 0, "color_1": 0, "color_2": 1}

    unknown_null: BinaryEncoder = BinaryEncoder(columns=["color"], handle_unknown="return_nan").fit(colors)
    assert unknown_null.transform_dict({"color": "zulu"}) == {"color_0": None, "color_1": None, "color_2": None}

    # 'indicator' makes the unknown a real extra level: it takes the rank
    # after the fitted ones, here 5, and the width is sized to carry it.
    indicator: BinaryEncoder = BinaryEncoder(columns=["color"], handle_unknown="indicator").fit(colors)
    assert indicator.transform_dict({"color": "zulu"}) == {"color_0": 1, "color_1": 0, "color_2": 1}
    assert indicator.transform_dict({"color": "amber"}) == {"color_0": 0, "color_1": 0, "color_2": 1}
    # Under Gray the indicator keeps its plain binary word, as upstream leaves it.
    gray_indicator: GrayEncoder = GrayEncoder(columns=["color"], handle_unknown="indicator").fit(colors)
    assert gray_indicator.transform_dict({"color": "zulu"}) == {"color_0": 1, "color_1": 0, "color_2": 1}
    assert gray_indicator.transform_dict({"color": "cyan"}) == {"color_0": 0, "color_1": 1, "color_2": 0}

    strict: BinaryEncoder = BinaryEncoder(columns=["color"], handle_unknown="error").fit(colors)
    with pytest.raises(EngineError, match="category that was not seen at fit"):
        strict.transform_dict({"color": "zulu"})
    with pytest.raises(EngineError, match="has nulls and handle_missing is 'error'"):
        BinaryEncoder(columns=["color"], handle_missing="error").fit(colors)


def test_a_later_step_sees_the_schema_the_expansion_produced(colors: list[str], tmp_path: Path) -> None:
    # count_encode picks its columns from the schema at fit time, so it must
    # see the digits: color is gone by then, size is still utf8.
    pipeline: Pipeline = Pipeline([("bits", BinaryEncoder(columns=["color"])), ("counts", CountEncoder())]).fit(colors)
    counts = pipeline["counts"]
    assert isinstance(counts, CountEncoder)
    assert counts.counts_ == {"size": {"l": 6}}
    row: dict[str, Any] = pipeline.transform_dict({"color": "amber", "size": "l"})
    assert row == {"size": 6, "color_0": 0, "color_1": 0, "color_2": 1}
    assert _table(pipeline.transform_parquet(colors, output_dir=tmp_path / "out")).to_pylist()[1] == row


def test_the_encoders_survive_the_split_and_the_artifact(colors: list[str], tmp_path: Path) -> None:
    for cls in (BaseNEncoder, BinaryEncoder, GrayEncoder):
        states: set[str] = set()
        for threads in (1, 4):
            encoder = cls(columns=["color"]).fit(colors, threads=threads)
            states.add(encoder._require_handle().stage_state(cls.__name__.lower()))
        assert len(states) == 1, cls.__name__

    pipeline: Pipeline = Pipeline([("bits", GrayEncoder(columns=["color"]))]).fit(colors)
    pipeline.save(tmp_path / "artifact")
    manifest: dict[str, Any] = json.loads((tmp_path / "artifact" / "manifest.json").read_text())
    assert manifest["ops"][0]["kernel"] == "gray_encode"
    assert manifest["ops"][0]["params"] == {"columns": ["color"], "handle_unknown": "value", "handle_missing": "value"}
    assert manifest["ops"][0]["state"] == {
        "columns": ["color"],
        "categories": {"color": ["amber", "blue", "cyan"]},
        "has_null": {"color": True},
    }
    record: dict[str, Any] = {"color": "blue", "size": "l"}
    assert Pipeline.load(tmp_path / "artifact").transform_dict(record) == pipeline.transform_dict(record)
    assert pickle.loads(pickle.dumps(pipeline)).transform_dict(record) == pipeline.transform_dict(record)

    # base is the one param the three do not share, and it round-trips
    # through the artifact, which is what rebuilds the shell.
    based: Pipeline = Pipeline([("bits", BaseNEncoder(columns=["color"], base=7))]).fit(colors)
    based.save(tmp_path / "based")
    reloaded = Pipeline.load(tmp_path / "based")["bits"]
    assert isinstance(reloaded, BaseNEncoder)
    assert reloaded.base == 7
    assert reloaded.params() == json.loads((tmp_path / "based" / "manifest.json").read_text())["ops"][0]["params"]
    standalone: BaseNEncoder = BaseNEncoder(columns=["color"], base=7).fit(colors)
    assert pickle.loads(pickle.dumps(standalone)).transform_dict(record) == standalone.transform_dict(record)

    manifest["ops"][0]["state"]["categories"]["color"] = ["cyan", "amber"]
    with pytest.raises(EngineError, match="must be sorted and unique"):
        _core.FittedPipeline.load_json(json.dumps(manifest))


def test_the_encoders_refuse_columns_they_cannot_encode(colors: list[str], tmp_path: Path) -> None:
    numbers: str = _write(tmp_path / "numbers.parquet", pa.table({"price": [1.0, 2.0]}))
    with pytest.raises(EngineError, match="is f64, expected utf8"):
        BinaryEncoder(columns=["price"]).fit([numbers])
    with pytest.raises(EngineError, match="not found in input"):
        BinaryEncoder(columns=["missing"]).fit(colors)
    with pytest.raises(EngineError, match="no utf8 columns to binary-encode"):
        BinaryEncoder().fit([numbers])
    # The digits would land on a column that is already there.
    clash: str = _write(tmp_path / "clash.parquet", pa.table({"color": ["amber", "blue"], "color_0": ["x", "y"]}))
    with pytest.raises(EngineError, match="output column 'color_0' already exists in input"):
        BinaryEncoder(columns=["color"]).fit([clash])


def test_the_shells_validate_their_arguments() -> None:
    bare_string: Any = "color"
    fraction: Any = 2.5
    flag: Any = True
    with pytest.raises(TypeError, match="not a single string"):
        BaseNEncoder(columns=bare_string)
    with pytest.raises(ValueError, match="at least one column"):
        GrayEncoder(columns=[])
    with pytest.raises(TypeError, match="base must be an integer"):
        BaseNEncoder(base=fraction)
    with pytest.raises(TypeError, match="base must be an integer"):
        BaseNEncoder(base=flag)
    with pytest.raises(ValueError, match="base must be 1 or more"):
        BaseNEncoder(base=0)
    with pytest.raises(ValueError, match="handle_unknown must be one of"):
        BinaryEncoder(handle_unknown="sometimes")
    with pytest.raises(ValueError, match="handle_missing must be one of"):
        GrayEncoder(handle_missing="sometimes")
    # Both fix the base at 2, as in category_encoders: it is not an argument.
    for cls in (BinaryEncoder, GrayEncoder):
        fixed_base: Any = cls
        with pytest.raises(TypeError, match="base"):
            fixed_base(base=2)
    assert repr(BaseNEncoder(columns=["c"], base=3)) == (
        "BaseNEncoder(columns=['c'], handle_unknown='value', handle_missing='value', base=3)"
    )
