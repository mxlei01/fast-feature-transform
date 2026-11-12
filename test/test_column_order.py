"""Regression tests: a fitted column list must survive an artifact in fit order.

With ``columns=None`` a step resolves its column list from the input schema,
in schema order. That order is part of the fitted result: an expanding encoder
names its generated columns walking it, and the ``#`` that disambiguates two
colliding generated names is appended walking it too. Rebuilding the list from
the keys of the fitted JSON object gives alphabetical order instead, so a
save/load or a pickle would move the generated block — and, where two names
collide, hand a feature name another feature's value.

Every case here uses a schema whose order is not its alphabetical order, so a
step that rebuilt its list from object keys fails.
"""

import json
import pickle
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fast_feature_transform import EngineError, _core
from fast_feature_transform.pipeline import Pipeline
from fast_feature_transform.transformers import (
    BackwardDifferenceEncoder,
    BaseNEncoder,
    Binarizer,
    BinaryEncoder,
    Cast,
    CatBoostEncoder,
    CountEncoder,
    GrayEncoder,
    GroupAgg,
    HashingEncoder,
    HelmertEncoder,
    JamesSteinEncoder,
    KBinsDiscretizer,
    LabelEncoder,
    LeaveOneOutEncoder,
    MaxAbsScaler,
    MEstimateEncoder,
    MinMaxScaler,
    Normalizer,
    OneHotEncoder,
    OrdinalEncoder,
    PolynomialEncoder,
    PolynomialFeatures,
    PowerTransformer,
    QuantileEncoder,
    QuantileTransformer,
    RankHotEncoder,
    RobustScaler,
    Select,
    SplineTransformer,
    StandardScaler,
    SumEncoder,
    SummaryEncoder,
    TargetEncoder,
    Transformer,
    WOEEncoder,
)

# Column order is deliberately not alphabetical, in both families: a step that
# derived its list from sorted object keys would serve [a_text, z_text] and
# [a_num, z_num] instead.
_TEXT: dict[str, list[str]] = {
    "z_text": ["u", "v", "u", "v"],
    "a_text": ["p", "q", "q", "p"],
}
_NUMBERS: dict[str, list[float]] = {
    "z_num": [1.0, 2.0, 3.0, 4.0],
    "a_num": [10.0, 20.0, 30.0, 40.0],
}
_TARGET: dict[str, list[float]] = {"y": [0.0, 1.0, 1.0, 0.0]}

_RECORD: dict[str, Any] = {"z_text": "u", "a_text": "q", "z_num": 2.0, "a_num": 30.0, "y": 1.0}
_SCHEMA: list[str] = [*_TEXT, *_NUMBERS, *_TARGET]

# Every transformer whose `columns` may be None, so its list is resolved from
# the schema at fit and must come back out of the artifact in that order.
_DERIVING: list[Transformer] = [
    LabelEncoder(),
    CountEncoder(),
    OrdinalEncoder(),
    OneHotEncoder(),
    OneHotEncoder(use_cat_names=True),
    RankHotEncoder(),
    BaseNEncoder(base=3),
    BinaryEncoder(),
    GrayEncoder(),
    BackwardDifferenceEncoder(),
    HelmertEncoder(),
    SumEncoder(),
    PolynomialEncoder(),
    TargetEncoder(target="y"),
    MEstimateEncoder(target="y"),
    WOEEncoder(target="y"),
    JamesSteinEncoder(target="y"),
    LeaveOneOutEncoder(target="y"),
    CatBoostEncoder(target="y"),
    QuantileEncoder(target="y"),
    SummaryEncoder(target="y"),
    StandardScaler(),
    MinMaxScaler(),
    MaxAbsScaler(),
    RobustScaler(),
    QuantileTransformer(n_quantiles=3),
    KBinsDiscretizer(n_bins=2, encode="ordinal", strategy="uniform"),
    KBinsDiscretizer(n_bins=2, encode="onehot", strategy="uniform"),
]

# The steps that take every number when given no columns, and steps whose
# generated columns, all numbers, are known only once they are fitted.
_NUMBERS_FROM_SCHEMA: dict[str, Callable[[], Transformer]] = {
    "StandardScaler": StandardScaler,
    "MinMaxScaler": MinMaxScaler,
    "MaxAbsScaler": MaxAbsScaler,
    "RobustScaler": RobustScaler,
    "QuantileTransformer": lambda: QuantileTransformer(n_quantiles=3),
    "KBinsDiscretizer": lambda: KBinsDiscretizer(n_bins=2, encode="ordinal", strategy="uniform"),
}
_EXPANDING: dict[str, Callable[[], Transformer]] = {
    "OneHotEncoder": lambda: OneHotEncoder(columns=["z_text"]),
    "BinaryEncoder": lambda: BinaryEncoder(columns=["z_text"]),
    "HelmertEncoder": lambda: HelmertEncoder(columns=["z_text"]),
    "KBinsDiscretizer": lambda: KBinsDiscretizer(columns=["z_num"], n_bins=2, encode="onehot", strategy="uniform"),
}


@pytest.fixture
def mixed(tmp_path: Path) -> str:
    """Write a Parquet file whose column order is not its alphabetical order.

    Args:
        tmp_path: pytest-provided temporary directory.

    Returns:
        The file path.

    """
    path: Path = tmp_path / "mixed.parquet"
    pq.write_table(pa.table({**_TEXT, **_NUMBERS, **_TARGET}), path)
    return str(path)


@pytest.fixture
def colliding(tmp_path: Path) -> str:
    """Write a Parquet file whose generated one-hot names collide.

    ``z_b`` holds the category ``c`` and ``z`` holds ``b_c``, so both generate
    ``z_b_c``; the second gets a ``#`` appended, and which column that is
    depends entirely on the order the two are walked in.

    Args:
        tmp_path: pytest-provided temporary directory.

    Returns:
        The file path.

    """
    path: Path = tmp_path / "collide.parquet"
    pq.write_table(pa.table({"z_b": ["c", "q"], "z": ["b_c", "w"]}), path)
    return str(path)


def _round_trips(pipeline: Pipeline, tmp_path: Path, name: str) -> list[Pipeline]:
    """Return the pipeline as it comes back from an artifact and from a pickle.

    Args:
        pipeline: A fitted pipeline.
        tmp_path: pytest temporary directory to save the artifact under.
        name: A directory name unique within tmp_path.

    Returns:
        The reloaded pipeline and the unpickled one.

    """
    pipeline.save(tmp_path / name)
    return [Pipeline.load(tmp_path / name), pickle.loads(pickle.dumps(pipeline))]


def test_a_collision_suffix_keeps_its_value_across_an_artifact_and_a_pickle(colliding: str, tmp_path: Path) -> None:
    """Colliding names keep their values; alphabetical order would flip ``z_b_c`` 1 -> 0 and ``z_b_c#`` 0 -> 1.

    Args:
        colliding: The colliding-names Parquet fixture.
        tmp_path: pytest temporary directory.

    """
    pipeline: Pipeline = Pipeline([("hot", OneHotEncoder(use_cat_names=True))]).fit([colliding])
    record: dict[str, Any] = {"z_b": "c", "z": "w"}
    fitted: dict[str, Any] = pipeline.transform_dict(dict(record))
    assert fitted == {"z_b_c": 1, "z_b_q": 0, "z_b_c#": 0, "z_w": 1}
    for restored in _round_trips(pipeline, tmp_path, "collide"):
        assert restored.transform_dict(dict(record)) == fitted
        assert list(restored.transform_dict(dict(record))) == list(fitted)


@pytest.mark.parametrize("expanding", _EXPANDING)
@pytest.mark.parametrize("numbers", _NUMBERS_FROM_SCHEMA)
def test_a_step_given_no_columns_takes_the_columns_known_when_the_fit_starts(
    numbers: str, expanding: str, mixed: str
) -> None:
    # REGRESSION: a step given no columns never takes the columns an expanding
    # step generates at fit. It used to take them when it fitted a pass later
    # than the expanding step, which a step in front of both decides.
    def steps() -> list[tuple[str, Transformer]]:
        return [("expand", _EXPANDING[expanding]()), ("step", _NUMBERS_FROM_SCHEMA[numbers]())]

    alone: Pipeline = Pipeline(steps()).fit([mixed])
    # The scaler in front leaves a_num as it is. The last step reads what it
    # writes, so it fits one pass later, after the expanding step is fitted.
    wait: Transformer = StandardScaler(columns=["a_num"], with_mean=False, with_std=False)
    waiting: Pipeline = Pipeline([("wait", wait), *steps()]).fit([mixed])

    state: dict[str, Any] = json.loads(alone.stages_state())["step"]
    consumed: str = "z_num" if expanding == "KBinsDiscretizer" else "z_text"
    assert state["columns"] == [name for name in [*_NUMBERS, *_TARGET] if name != consumed]
    assert json.loads(waiting.stages_state())["step"] == state
    assert waiting.transform_dict(_RECORD) == alone.transform_dict(_RECORD)


@pytest.mark.parametrize("transformer", _DERIVING, ids=lambda t: type(t).__name__)
def test_a_derived_column_list_survives_an_artifact_and_a_pickle_in_fit_order(
    transformer: Transformer, mixed: str, tmp_path: Path
) -> None:
    """Every ``columns=None`` step serves the same names, in the same order, after a round trip.

    Args:
        transformer: A transformer whose ``columns`` may be None.
        mixed: The Parquet fixture with a non-alphabetical column order.
        tmp_path: pytest temporary directory.

    """
    name: str = type(transformer).__name__
    pipeline: Pipeline = Pipeline([("step", transformer)]).fit([mixed])
    fitted: list[tuple[str, Any]] = list(pipeline.transform_dict(dict(_RECORD)).items())
    # The fit order is the schema order, which is not the alphabetical one.
    assert fitted != sorted(fitted), name
    for restored in _round_trips(pipeline, tmp_path, "artifact"):
        assert list(restored.transform_dict(dict(_RECORD)).items()) == fitted, name
    # An in-place step serves the same record whatever order it walks, so the
    # manifest is asserted directly too: the list it records is the resolved
    # one, never a sorted one.
    manifest: dict[str, Any] = json.loads((tmp_path / "artifact" / "manifest.json").read_text())
    recorded: list[str] = manifest["ops"][0]["state"]["columns"]
    assert recorded == [column for column in _SCHEMA if column in set(recorded)], name
    assert recorded != sorted(recorded), name


def test_the_manifest_records_the_resolved_order_not_a_sorted_one(mixed: str, tmp_path: Path) -> None:
    """The fitted state names its columns as an ordered list, in schema order.

    Args:
        mixed: The Parquet fixture with a non-alphabetical column order.
        tmp_path: pytest temporary directory.

    """
    pipeline: Pipeline = Pipeline(
        [("hot", OneHotEncoder()), ("scale", StandardScaler())],
    ).fit([mixed])
    pipeline.save(tmp_path / "artifact")
    manifest: dict[str, Any] = json.loads((tmp_path / "artifact" / "manifest.json").read_text())
    assert manifest["ops"][0]["state"]["columns"] == ["z_text", "a_text"]
    assert manifest["ops"][1]["state"]["columns"] == ["z_num", "a_num", "y"]


def test_a_reordered_column_list_is_a_different_artifact(mixed: str, tmp_path: Path) -> None:
    """Reordering the state's column list changes the output's column order, so it cannot be ignored.

    Args:
        mixed: The Parquet fixture with a non-alphabetical column order.
        tmp_path: pytest temporary directory.

    """
    pipeline: Pipeline = Pipeline([("hot", OneHotEncoder())]).fit([mixed])
    pipeline.save(tmp_path / "artifact")
    manifest: dict[str, Any] = json.loads((tmp_path / "artifact" / "manifest.json").read_text())
    manifest["ops"][0]["state"]["columns"] = ["a_text", "z_text"]
    swapped: Any = _core.FittedPipeline.load_json(json.dumps(manifest))
    assert list(swapped.transform_row(dict(_RECORD))) != list(pipeline.transform_dict(dict(_RECORD)))


def test_an_artifact_without_a_column_order_is_refused(mixed: str, tmp_path: Path) -> None:
    """A manifest without the fitted column order cannot have one reconstructed, so it is refused.

    Args:
        mixed: The Parquet fixture with a non-alphabetical column order.
        tmp_path: pytest temporary directory.

    """
    pipeline: Pipeline = Pipeline([("hot", OneHotEncoder())]).fit([mixed])
    pipeline.save(tmp_path / "artifact")
    manifest: dict[str, Any] = json.loads((tmp_path / "artifact" / "manifest.json").read_text())
    assert manifest["schema_version"] == "0.3.0"

    stripped: dict[str, Any] = json.loads(json.dumps(manifest))
    del stripped["ops"][0]["state"]["columns"]
    with pytest.raises(EngineError, match="state must hold a 'columns' list"):
        _core.FittedPipeline.load_json(json.dumps(stripped))


# The other half of the sweep: every kernel that requires its column list,
# as `columns` or as the keys of `types` (cast) or `aggs` (group_agg).
# None of these derives a column list from its fitted state, so none could
# carry the defect — but the sweep is what says so, rather than a reading of
# the source. group_agg is the one to watch: nlohmann sorts JSON object keys,
# so its aggregate block is ordered by sorted column name both at fit and at
# load, which is stable but is not the order `aggs` was written in.
_EXPLICIT: list[Transformer] = [
    GroupAgg(key="z_text", aggs={"z_num": ["mean", "sum"], "a_num": ["max"]}),
    HashingEncoder(columns=["z_text", "a_text"], n_components=3),
    PolynomialFeatures(columns=["z_num", "a_num"], degree=2),
    SplineTransformer(columns=["z_num", "a_num"], n_knots=3, degree=1),
    PowerTransformer(columns=["z_num", "a_num"]),
    Binarizer(columns=["z_num", "a_num"], threshold=2.0),
    Normalizer(columns=["z_num", "a_num"]),
    Cast(types={"z_num": "f64", "a_num": "f64"}),
    Select(columns=["z_num", "a_num", "z_text"]),
]


@pytest.mark.parametrize("transformer", _EXPLICIT, ids=lambda t: type(t).__name__)
def test_a_required_column_list_survives_an_artifact_and_a_pickle_too(
    transformer: Transformer, mixed: str, tmp_path: Path
) -> None:
    """A step whose column list comes from its params serves the same output after a round trip.

    Args:
        transformer: A transformer whose ``columns`` (or equivalent) is required.
        mixed: The Parquet fixture with a non-alphabetical column order.
        tmp_path: pytest temporary directory.

    """
    name: str = type(transformer).__name__
    pipeline: Pipeline = Pipeline([("step", transformer)]).fit([mixed])
    fitted: list[tuple[str, Any]] = list(pipeline.transform_dict(dict(_RECORD)).items())
    for restored in _round_trips(pipeline, tmp_path, "artifact"):
        assert list(restored.transform_dict(dict(_RECORD)).items()) == fitted, name
