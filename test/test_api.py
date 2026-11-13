"""Tests for the Python API surface: validation, nesting rules, pickling, and ergonomics."""

import copy
import enum
import importlib.metadata
import inspect
import json
import pickle
import subprocess  # nosec B404
import sys
import tempfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar, override

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import fast_feature_transform
from fast_feature_transform import (
    BinaryEncoder,
    Cast,
    CatBoostEncoder,
    CountEncoder,
    EngineError,
    GroupAgg,
    HashingEncoder,
    HelmertEncoder,
    JamesSteinEncoder,
    KBinsDiscretizer,
    LabelEncoder,
    Normalizer,
    OneHotEncoder,
    OrdinalEncoder,
    Pipeline,
    PowerTransformer,
    QuantileEncoder,
    QuantileTransformer,
    Select,
    SplineTransformer,
    StandardScaler,
    SummaryEncoder,
    TargetEncoder,
    Transformer,
    WOEEncoder,
    _core,
    parquet_to_memmap,
)
from fast_feature_transform.transformers import _output_directory
from helpers import NAME_KIND_IDS, NAME_KINDS

# LabelEncoder numbers the sorted categories from 1: desk 1, mob 2, tab 3.
MOB_INDEX: int = 2


@pytest.fixture
def train_files(tmp_path: Path) -> list[str]:
    """Write one Parquet file with two row groups.

    Args:
        tmp_path: pytest-provided temporary directory.

    Returns:
        Paths of the written Parquet files.

    """
    table: pa.Table = pa.table(
        {"device": ["mob", "desk", "mob", "tab"], "source": ["a", "b", "a", "a"], "price": [1.0, 2.0, 3.0, 4.0]}
    )
    path: Path = tmp_path / "train.parquet"
    pq.write_table(table, path, row_group_size=2)
    return [str(path)]


def _nested() -> Pipeline:
    return Pipeline(
        [
            ("features", Pipeline([("select", Select(columns=["device", "source"]))])),
            ("encode", LabelEncoder(columns=["device", "source"])),
        ]
    )


def test_package_exports_the_public_api() -> None:
    assert fast_feature_transform.__all__ == [
        "BackwardDifferenceEncoder",
        "BaseNEncoder",
        "Binarizer",
        "BinaryEncoder",
        "Cast",
        "CatBoostEncoder",
        "CountEncoder",
        "EngineError",
        "GrayEncoder",
        "GroupAgg",
        "HashingEncoder",
        "HelmertEncoder",
        "JamesSteinEncoder",
        "KBinsDiscretizer",
        "LabelEncoder",
        "LeaveOneOutEncoder",
        "MEstimateEncoder",
        "MaxAbsScaler",
        "MinMaxScaler",
        "Normalizer",
        "OneHotEncoder",
        "OrdinalEncoder",
        "Pipeline",
        "PolynomialEncoder",
        "PolynomialFeatures",
        "PowerTransformer",
        "QuantileEncoder",
        "QuantileTransformer",
        "RankHotEncoder",
        "RobustScaler",
        "Select",
        "SplineTransformer",
        "StandardScaler",
        "SumEncoder",
        "SummaryEncoder",
        "TargetEncoder",
        "Transformer",
        "WOEEncoder",
        "__version__",
        "parquet_to_memmap",
    ]
    assert issubclass(Select, Transformer)
    assert issubclass(EngineError, RuntimeError)
    assert fast_feature_transform.__version__


def test_the_version_is_the_installed_one_and_comes_from_the_extension(tmp_path: Path) -> None:
    assert type(fast_feature_transform.__version__) is str
    assert fast_feature_transform.__version__ == _core.library_version()
    assert fast_feature_transform.__version__ == importlib.metadata.version("fast-feature-transform")
    # REGRESSION: importing the package does not import importlib.metadata. It
    # used to ask the installed metadata for the version, and that import and
    # its search of the installed distributions took about a third of this one
    # with no cached bytecode and about half with the bytecode cached. A child
    # process shows it, because this process has imported the metadata.
    # Where the interpreter's own start imports it, there is nothing to show.
    script: str = (
        "import sys\n"
        "before = 'importlib.metadata' in sys.modules\n"
        "import fast_feature_transform\n"
        "print(before, 'importlib.metadata' in sys.modules)\n"
    )
    command: list[str] = [sys.executable, "-c", script]
    child = subprocess.run(command, check=True, capture_output=True, text=True, cwd=tmp_path)  # nosec B603
    assert child.stdout.split() in (["False", "False"], ["True", "True"])


@pytest.mark.parametrize("call", ["pipeline", "transformer", "memmap"])
def test_a_failed_batch_call_removes_the_temporary_directory_it_made(
    call: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # REGRESSION: a call given no output_dir makes a temporary directory, and
    # removes it again when it fails. It used to leave the empty directory.
    temporary: Path = tmp_path / "temporary"
    temporary.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(temporary))
    numbers: Path = tmp_path / "numbers.parquet"
    pq.write_table(pa.table({"x": [1.0, 2.0, 3.0]}), numbers)
    calls: dict[str, Any] = {
        "pipeline": Pipeline([("select", Select(columns=["x"]))]).fit([numbers]).transform_parquet,
        "transformer": Select(columns=["x"]).fit([numbers]).transform_parquet,
        "memmap": parquet_to_memmap,
    }
    with pytest.raises(EngineError):
        calls[call]([tmp_path / "missing.parquet"])
    assert list(temporary.iterdir()) == []
    with pytest.raises(TypeError, match="not a single path"):
        calls[call](str(numbers))
    assert list(temporary.iterdir()) == []

    # A call that succeeds keeps the directory, which holds what it wrote.
    written = calls[call]([numbers])
    (made,) = temporary.iterdir()
    assert made.name.startswith("fast_feature_transform_")
    assert {path.parent for path in ([written] if call == "memmap" else written)} == {made}


def _interrupted_call(made: list[Path], *, wrote: bool) -> None:
    with _output_directory(None) as directory:
        made.append(Path(directory))
        if wrote:
            (made[-1] / "part-00000.parquet").write_bytes(b"rows")
        raise KeyboardInterrupt


def test_an_interrupt_removes_the_temporary_directory_only_while_it_is_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An interrupt, which is not an Exception, removes the directory too. One
    # that lands while the engine runs is raised when the engine returns, after
    # it has written its output, and rmdir leaves that output where it is.
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    made: list[Path] = []
    with pytest.raises(KeyboardInterrupt):
        _interrupted_call(made, wrote=False)
    assert not made[0].exists()
    with pytest.raises(KeyboardInterrupt):
        _interrupted_call(made, wrote=True)
    assert [path.name for path in made[1].iterdir()] == ["part-00000.parquet"]


def test_nested_pipeline_refuses_to_transform_on_its_own(train_files: list[str]) -> None:
    outer: Pipeline = _nested().fit(train_files)
    features = outer["features"]
    assert isinstance(features, Pipeline)
    with pytest.raises(RuntimeError, match="'features' is a step of a larger pipeline; call this on the root pipeline"):
        features.transform_dict({"device": "mob", "source": "a", "price": 1.0})
    with pytest.raises(RuntimeError, match="call this on the root pipeline"):
        features.transform_parquet(train_files)
    assert outer.transform_dict({"device": "mob", "source": "a", "price": 1.0}) == {"device": MOB_INDEX, "source": 1}


def test_stages_json_lists_every_step_in_execution_order(train_files: list[str]) -> None:
    pipeline: Pipeline = _nested().fit(train_files)
    assert json.loads(pipeline.stages_json()) == [
        {"stage": "features.select", "kernel": "select", "params": {"columns": ["device", "source"]}},
        {"stage": "encode", "kernel": "label_encode", "params": {"columns": ["device", "source"], "min_count": 1}},
    ]


def test_stages_state_maps_every_stage_to_what_it_learned_in_order(train_files: list[str]) -> None:
    pipeline: Pipeline = _nested().fit(train_files)
    states: dict[str, Any] = json.loads(pipeline.stages_state())
    assert list(states) == ["features.select", "encode"]
    assert states["features.select"] == {}
    encoder = pipeline["encode"]
    assert isinstance(encoder, LabelEncoder)
    assert states["encode"]["label_maps"] == encoder.label_maps_
    assert states["encode"]["label_maps"]["device"] == {"desk": 1, "mob": MOB_INDEX, "tab": 3}
    # Each value is exactly the state the artifact stores for that stage.
    manifest: dict[str, Any] = json.loads(pipeline._require_handle().manifest_json())
    assert states == {op["stage"]: op["state"] for op in manifest["ops"]}


def test_stages_json_and_stages_state_survive_save_load_and_pickle(train_files: list[str], tmp_path: Path) -> None:
    pipeline: Pipeline = _nested().fit(train_files)
    pipeline.save(tmp_path / "artifact")
    loaded: Pipeline = Pipeline.load(tmp_path / "artifact")
    assert loaded.stages_json() == pipeline.stages_json()
    assert loaded.stages_state() == pipeline.stages_state()
    # Each method is the first call on its own unpickled copy, since an
    # unpickled pipeline loads its artifact on first use.
    payload: bytes = pickle.dumps(pipeline)
    for method in ("stages_json", "stages_state"):
        restored: Pipeline = pickle.loads(payload)
        assert restored._handle is None
        assert getattr(restored, method)() == getattr(pipeline, method)()


def test_stages_json_and_stages_state_refuse_unfitted_and_nested_pipelines(train_files: list[str]) -> None:
    with pytest.raises(RuntimeError, match="not fitted"):
        _nested().stages_json()
    with pytest.raises(RuntimeError, match="not fitted"):
        _nested().stages_state()
    features = _nested().fit(train_files)["features"]
    assert isinstance(features, Pipeline)
    with pytest.raises(RuntimeError, match="call this on the root pipeline"):
        features.stages_json()
    with pytest.raises(RuntimeError, match="call this on the root pipeline"):
        features.stages_state()


def test_transform_parquet_rejects_a_bare_path_and_non_paths(train_files: list[str]) -> None:
    pipeline: Pipeline = _nested().fit(train_files)
    encoder: LabelEncoder = LabelEncoder(columns=["device"]).fit(train_files)
    bare_path: Any = Path(train_files[0])
    records: Any = [{"device": "mob"}]
    for target in (pipeline, encoder):
        with pytest.raises(TypeError, match="wrap it in a list"):
            target.transform_parquet(bare_path)
        with pytest.raises(TypeError, match="element"):
            target.transform_parquet(records)
    with pytest.raises(TypeError, match="wrap it in a list"):
        pipeline.fit(bare_path)


def test_constructor_arguments_are_validated() -> None:
    bare_string: Any = "device"
    numbers: Any = [1, 2]
    text_count: Any = "2"
    a_class: Any = Select
    not_a_pair: Any = [Select(columns=["a"])]
    with pytest.raises(TypeError, match="not a single string"):
        Select(columns=bare_string)
    with pytest.raises(ValueError, match="at least one column"):
        Select(columns=[])
    with pytest.raises(TypeError, match="list of column names"):
        LabelEncoder(columns=numbers)
    with pytest.raises(TypeError, match="min_count must be an integer"):
        LabelEncoder(min_count=text_count)
    with pytest.raises(ValueError, match="min_count must be >= 1"):
        LabelEncoder(min_count=0)
    # REGRESSION: the kernels read an integer argument as an int64, and 2**63
    # used to reach them, where it was read wrapped or refused only at fit.
    with pytest.raises(ValueError, match=r"LabelEncoder: min_count must be an integer from -2\*\*63 to 2\*\*63 - 1"):
        LabelEncoder(min_count=2**63)
    with pytest.raises(TypeError, match="must be a Transformer or Pipeline instance"):
        Pipeline([("s", a_class)])
    with pytest.raises(TypeError, match="pair"):
        Pipeline(not_a_pair)


# Options that take a string from a fixed set: at least one of every
# constructor that checks one with _one_of, both of the target encoders', and
# the two checked with messages of their own.
_STRING_OPTIONS: list[tuple[type[Transformer], dict[str, Any], str]] = [
    (OrdinalEncoder, {"handle_unknown": "return_nan"}, "handle_unknown"),
    (OrdinalEncoder, {"handle_missing": "error"}, "handle_missing"),
    (OneHotEncoder, {"handle_unknown": "indicator"}, "handle_unknown"),
    (BinaryEncoder, {"handle_missing": "return_nan"}, "handle_missing"),
    (HelmertEncoder, {"handle_unknown": "error"}, "handle_unknown"),
    (CountEncoder, {"handle_unknown": "return_nan"}, "handle_unknown"),
    (CountEncoder, {"handle_missing": "error"}, "handle_missing"),
    (TargetEncoder, {"target": "y", "handle_unknown": "error"}, "handle_unknown"),
    (WOEEncoder, {"target": "y", "handle_missing": "return_nan"}, "handle_missing"),
    (CatBoostEncoder, {"target": "y", "handle_unknown": "error"}, "handle_unknown"),
    (JamesSteinEncoder, {"target": "y", "handle_missing": "return_nan"}, "handle_missing"),
    (JamesSteinEncoder, {"target": "y", "model": "pooled"}, "model"),
    (QuantileEncoder, {"target": "y", "handle_unknown": "return_nan"}, "handle_unknown"),
    (SummaryEncoder, {"target": "y", "handle_missing": "error"}, "handle_missing"),
    (Normalizer, {"columns": ["x"], "norm": "l1"}, "norm"),
    (QuantileTransformer, {"output_distribution": "normal"}, "output_distribution"),
    (KBinsDiscretizer, {"encode": "ordinal"}, "encode"),
    (KBinsDiscretizer, {"strategy": "uniform"}, "strategy"),
    (SplineTransformer, {"columns": ["x"], "extrapolation": "linear"}, "extrapolation"),
    (PowerTransformer, {"columns": ["x"], "method": "box-cox"}, "method"),
    (HashingEncoder, {"columns": ["c"], "hash_method": "md5"}, "hash_method"),
    (CountEncoder, {"combine_min_nan_groups": "force"}, "combine_min_nan_groups"),
]


@pytest.mark.parametrize(
    ("transformer", "arguments", "option"),
    _STRING_OPTIONS,
    ids=[f"{transformer.__name__}-{option}" for transformer, _, option in _STRING_OPTIONS],
)
def test_an_option_given_as_an_enum_of_strings_is_taken_as_its_value(
    transformer: type[Transformer], arguments: dict[str, Any], option: str
) -> None:
    # REGRESSION: an option given as a str subclass is stored as the plain
    # string it equals. An Enum with a str mixin equals its value and prints
    # as "Member.NAME": the encoders that printed it were refused at fit, and
    # those that kept the member itself were not.
    member: Any = enum.Enum("Member", {"NAME": arguments[option]}, type=str).NAME
    plain: Transformer = transformer(**arguments)
    given: Transformer = transformer(**{**arguments, option: member})
    assert type(getattr(given, option)) is str
    assert getattr(given, option) == arguments[option]
    assert given.params() == plain.params()
    assert json.dumps(given.params()) == json.dumps(plain.params())


def test_type_tags_and_aggregates_given_as_an_enum_of_strings_are_taken_as_their_values() -> None:
    tag: Any = enum.Enum("Tag", {"NAME": "f64"}, type=str).NAME
    aggregate: Any = enum.Enum("Aggregate", {"NAME": "mean"}, type=str).NAME
    cast: Cast = Cast({"x": tag})
    assert [type(value) for value in cast.types.values()] == [str]
    assert cast.params() == Cast({"x": "f64"}).params()
    group: GroupAgg = GroupAgg(key="k", aggs={"x": [aggregate]})
    assert [type(value) for value in group.aggs["x"]] == [str]
    assert group.params() == GroupAgg(key="k", aggs={"x": ["mean"]}).params()


def test_an_encoder_whose_option_is_an_enum_fits_like_one_given_the_string(tmp_path: Path) -> None:
    source: Path = tmp_path / "train.parquet"
    pq.write_table(pa.table({"c": ["a", "b", "a", "b"], "y": [0.0, 1.0, 1.0, 0.0]}), source)
    member: Any = enum.Enum("Member", {"NAME": "return_nan"}, type=str).NAME
    pairs: list[tuple[Transformer, Transformer]] = [
        (
            OrdinalEncoder(columns=["c"], handle_unknown=member),
            OrdinalEncoder(columns=["c"], handle_unknown="return_nan"),
        ),
        (
            QuantileEncoder(target="y", columns=["c"], handle_unknown=member),
            QuantileEncoder(target="y", columns=["c"], handle_unknown="return_nan"),
        ),
    ]
    for given, plain in pairs:
        given.fit([source])
        plain.fit([source])
        for record in ({"c": "a", "y": 0.0}, {"c": "never seen", "y": 0.0}):
            assert repr(given.transform_dict(dict(record))) == repr(plain.transform_dict(dict(record)))


def _subclassed_strings(value: Any) -> list[Any]:
    """Find the strings whose type is a subclass of str in nested dicts, lists and tuples.

    Args:
        value: The value to search, dict keys included.

    Returns:
        Every such string, in the order found.

    """
    if isinstance(value, str):
        return [] if type(value) is str else [value]
    if isinstance(value, dict):
        return [found for pair in value.items() for found in _subclassed_strings(pair)]
    if isinstance(value, (list, tuple)):
        return [found for item in value for found in _subclassed_strings(item)]
    return []


# Every transformer class the package exports.
_TRANSFORMERS: list[type[Transformer]] = sorted(
    (
        each
        for each in vars(fast_feature_transform).values()
        if isinstance(each, type) and issubclass(each, Transformer) and each is not Transformer
    ),
    key=lambda each: each.__name__,
)


def _named_arguments(transformer: type[Transformer], name: Callable[[str], Any]) -> dict[str, Any]:
    """Build constructor arguments that pass a transformer every kind of name it takes.

    Args:
        transformer: The transformer class.
        name: Makes each name from its value.

    Returns:
        The arguments.

    """
    if transformer is Cast:
        return {"types": {name("price"): "f32", name("device"): "utf8"}}
    if transformer is GroupAgg:
        return {"key": name("device"), "aggs": {name("price"): ["mean", "max"], name("source"): ["count"]}}
    arguments: dict[str, Any] = {"columns": [name("device"), name("source")]}
    parameters = inspect.signature(transformer).parameters
    if "target" in parameters:
        arguments["target"] = name("price")
    if "min_group_name" in parameters:
        arguments["min_group_name"] = name("rare")
    return arguments


@pytest.mark.parametrize("name", NAME_KINDS, ids=NAME_KIND_IDS)
def test_every_name_a_transformer_takes_is_stored_as_the_plain_string(name: Callable[[str], Any]) -> None:
    # REGRESSION: a column name, a target, the GroupAgg key, min_group_name
    # and the keys of Cast and GroupAgg given as a str subclass are stored as
    # the plain strings they equal. They used to be kept as given, so repr
    # and params() showed an Enum member, and a pickle loaded only where the
    # caller's class could be imported.
    for transformer in _TRANSFORMERS:
        given: Transformer = transformer(**_named_arguments(transformer, name))
        plain: Transformer = transformer(**_named_arguments(transformer, str))
        assert _subclassed_strings(vars(given)) == [], transformer.__name__
        assert repr(given) == repr(plain)


def _named_pipeline(name: Callable[[str], Any]) -> Pipeline:
    """Build a pipeline nested three deep whose step names ``name`` makes.

    Args:
        name: Makes each step name from its value.

    Returns:
        The unfitted pipeline.

    """
    inner: Pipeline = Pipeline([(name("label"), LabelEncoder(columns=["device"]))])
    outer: Pipeline = Pipeline([(name("inner"), inner), (name("scale"), StandardScaler(columns=["price"]))])
    return Pipeline([(name("outer"), outer), (name("target"), TargetEncoder(target="price", columns=["source"]))])


def _fitted_attributes(pipeline: Pipeline, prefix: str = "") -> dict[str, Any]:
    """Read every fitted attribute of every transformer in a pipeline.

    Args:
        pipeline: A fitted pipeline.
        prefix: The dotted step names above this pipeline.

    Returns:
        Each attribute's value, by dotted step name and attribute name.

    """
    found: dict[str, Any] = {}
    for step_name, step in pipeline.steps:
        if isinstance(step, Pipeline):
            found.update(_fitted_attributes(step, f"{prefix}{step_name}."))
            continue
        for attribute in dir(type(step)):
            if attribute.endswith("_") and isinstance(getattr(type(step), attribute), property):
                found[f"{prefix}{step_name}.{attribute}"] = getattr(step, attribute)
    assert [type(step_name) for step_name, _ in pipeline.steps] == [str] * len(pipeline)
    return found


@pytest.mark.parametrize("name", NAME_KINDS, ids=NAME_KIND_IDS)
def test_a_step_name_given_as_a_str_subclass_is_stored_as_the_plain_string(
    name: Callable[[str], Any], train_files: list[str]
) -> None:
    # REGRESSION: a step name given as a str subclass is stored as the plain
    # string it equals. A nested step's stage name is built by formatting the
    # step names, and an Enum with a str mixin formats as "EnumName.outer",
    # so every fitted attribute of a nested step raised EngineError "no stage
    # named 'EnumName.outer.EnumName.inner.EnumName.label' in this pipeline".
    plain: Pipeline = _named_pipeline(str).fit(train_files)
    expected: dict[str, Any] = _fitted_attributes(plain)
    assert list(expected) == [
        "outer.inner.label.label_maps_",
        "outer.scale.mean_",
        "outer.scale.n_samples_seen_",
        "outer.scale.scale_",
        "outer.scale.var_",
        "target.null_values_",
        "target.prior_",
        "target.values_",
    ]
    given: Pipeline = _named_pipeline(name).fit(train_files)
    assert given.stages_json() == plain.stages_json()
    for pipeline in (given, pickle.loads(pickle.dumps(given))):
        assert _fitted_attributes(pipeline) == expected
        assert repr(pipeline) == repr(plain)
        outer = pipeline[name("outer")]
        assert isinstance(outer, Pipeline)
        inner = outer[name("inner")]
        assert isinstance(inner, Pipeline)
        encoder = inner[name("label")]
        assert isinstance(encoder, LabelEncoder)
        assert encoder.vocab_size("device") == len(expected["outer.inner.label.label_maps_"]["device"]) + 1
        with pytest.raises(RuntimeError, match=r"^Pipeline 'outer' is a step of a larger pipeline"):
            outer.stages_json()
        with pytest.raises(RuntimeError, match=r"^Pipeline 'outer\.inner' is a step of a larger pipeline"):
            inner.stages_json()
    # An error that shows a step name shows the plain string.
    with pytest.raises(ValueError, match=r"^duplicate step name 'label'$"):
        Pipeline([("label", Select(columns=["device"])), (name("label"), Select(columns=["device"]))])
    with pytest.raises(ValueError, match=r"^invalid step name 'a\.b'"):
        Pipeline([(name("a.b"), Select(columns=["device"]))])
    a_class: Any = Select
    with pytest.raises(TypeError, match=r"^step 'label' must be a Transformer or Pipeline instance"):
        Pipeline([(name("label"), a_class)])


# Loads each pickled case in an interpreter that cannot import the modules
# named after the file of cases, and prints what each loaded object answers.
_LOAD_IN_A_CHILD: str = """
import importlib.util
import json
import pickle
import sys

assert [importlib.util.find_spec(module) for module in sys.argv[2:]] == [None] * len(sys.argv[2:])
answers = []
with open(sys.argv[1], "rb") as cases:
    for payload, record in pickle.load(cases):
        try:
            loaded = pickle.loads(payload)
        except Exception as error:
            answers.append([type(error).__name__, str(error)])
            continue
        answers.append([repr(loaded), None if record is None else repr(loaded.transform_dict(record))])
print(json.dumps(answers))
"""


def test_a_pickle_of_names_given_as_str_subclasses_loads_where_their_class_is_not_defined(
    train_files: list[str], tmp_path: Path
) -> None:
    # REGRESSION: a pickle of a transformer or pipeline given its names as a
    # str subclass holds only plain strings. It used to hold the subclass, so
    # it loaded only in an interpreter that could import the caller's class.
    record: dict[str, Any] = {"device": "desk", "source": "a", "price": 1.0}
    cases: list[tuple[bytes, dict[str, Any] | None]] = []
    expected: list[list[str | None]] = []
    fitted: list[Callable[[Callable[[str], Any]], Transformer | Pipeline]] = [
        _named_pipeline,
        lambda name: TargetEncoder(target=name("price"), columns=[name("source")]),
        lambda name: GroupAgg(key=name("device"), aggs={name("price"): ["mean"]}),
        lambda name: CountEncoder(columns=[name("device")], min_group_size=2, min_group_name=name("rare")),
        lambda name: Cast({name("price"): "f32"}),
    ]
    for name in NAME_KINDS:
        for transformer in _TRANSFORMERS:
            cases.append((pickle.dumps(transformer(**_named_arguments(transformer, name))), None))
            expected.append([repr(transformer(**_named_arguments(transformer, str))), None])
        for make in fitted:
            cases.append((pickle.dumps(make(name).fit(train_files)), record))
            plain: Transformer | Pipeline = make(str).fit(train_files)
            expected.append([repr(plain), repr(plain.transform_dict(record))])
    path: Path = tmp_path / "cases.pickle"
    path.write_bytes(pickle.dumps(cases))
    modules: list[str] = sorted({kind.__module__ for kind in NAME_KINDS})
    command: list[str] = [sys.executable, "-c", _LOAD_IN_A_CHILD, str(path), *modules]
    child = subprocess.run(command, check=True, capture_output=True, text=True, cwd=tmp_path)  # nosec B603
    assert json.loads(child.stdout) == expected


def test_threads_are_validated_and_accepted_everywhere(train_files: list[str], tmp_path: Path) -> None:
    for threads in (-1, 2**32):
        with pytest.raises(ValueError, match=r"threads must be an integer from 0 to 4294967295 \(0 uses all cores\)"):
            _nested().fit(train_files, threads=threads)
    encoder: LabelEncoder = LabelEncoder(columns=["device"]).fit(train_files, threads=1)
    assert encoder.label_maps_["device"]["mob"] == MOB_INDEX
    pipeline: Pipeline = _nested().fit(train_files, threads=2)
    pipeline.save(tmp_path / "artifact")
    for threads in (-1, 2**32):
        with pytest.raises(ValueError, match="threads must be an integer from 0 to 4294967295"):
            Pipeline.load(tmp_path / "artifact", threads=threads)
        with pytest.raises(ValueError, match="threads must be an integer from 0 to 4294967295"):
            fast_feature_transform.parquet_to_memmap(train_files, threads=threads)
    assert Pipeline.load(tmp_path / "artifact", threads=1).transform_dict({"device": "tab", "source": "b"}) == {
        "device": 3,
        "source": 2,
    }


def test_fit_refuses_a_temp_dir_that_does_not_exist(train_files: list[str], tmp_path: Path) -> None:
    """A temp_dir that does not exist is refused before the fit starts.

    REGRESSION: the engine opens temp_dir only when the fit first spills, so a
    typo used to pass on small data and fail on data big enough to spill.
    """
    with pytest.raises(ValueError, match="temp_dir must be an existing directory"):
        _nested().fit(train_files, temp_dir=tmp_path / "missing")
    assert _nested().fit(train_files, temp_dir=tmp_path).transform_dict({"device": "tab", "source": "b"})


def test_pipeline_indexing_length_and_repr(train_files: list[str]) -> None:
    pipeline: Pipeline = _nested()
    assert len(pipeline) == 2
    assert pipeline[0] is pipeline["features"]
    assert pipeline[-1] is pipeline["encode"]
    with pytest.raises(KeyError):
        pipeline["nope"]
    with pytest.raises(IndexError):
        pipeline[5]
    assert repr(Select(columns=["a"])) == "Select(columns=['a'])"
    assert repr(LabelEncoder()) == "LabelEncoder(columns=None, min_count=1)"
    assert repr(pipeline).startswith("Pipeline(steps=[('features', Pipeline(steps=[('select', Select(")


def test_nested_pipeline_pickle_stores_the_manifest_once(train_files: list[str]) -> None:
    pipeline: Pipeline = _nested().fit(train_files)
    payload: bytes = pickle.dumps(pipeline)
    assert payload.count(b'"schema_version"') == 1
    restored: Pipeline = pickle.loads(payload)
    assert restored.transform_dict({"device": "mob", "source": "a"}) == {"device": MOB_INDEX, "source": 1}
    encoder = restored["encode"]
    assert isinstance(encoder, LabelEncoder)
    assert encoder.label_maps_["device"]["mob"] == MOB_INDEX
    # The nested pipeline restored inside the root shares the root's handle.
    features = restored["features"]
    assert isinstance(features, Pipeline)
    assert features._handle is restored._handle


def test_ownership_is_known_without_parsing_the_manifest(train_files: list[str]) -> None:
    standalone: LabelEncoder = LabelEncoder(columns=["device"]).fit(train_files)
    assert standalone._require_binding()[2] is True
    bound = _nested().fit(train_files)["encode"]
    assert isinstance(bound, LabelEncoder)
    assert bound._require_binding()[2] is False
    restored: LabelEncoder = pickle.loads(pickle.dumps(standalone))
    assert restored._require_binding()[2] is True
    assert restored.transform_dict({"device": "mob"}) == {"device": MOB_INDEX}


class _Interleaved(LabelEncoder):
    """A LabelEncoder that calls ``hook`` before each attribute write and ``after`` after it.

    Another thread can run between any two attribute writes, and between a
    write and the code that follows it. The hooks run there in the writing
    thread, so the interleaving is forced rather than hoped for. They are
    class attributes, so copies and pickles carry none of them, and both
    are off while either runs.
    """

    hook: ClassVar[Callable[[str, Any], None] | None] = None
    after: ClassVar[Callable[[str, Any], None] | None] = None

    @override
    def __setattr__(self, name: str, value: Any) -> None:
        cls = type(self)
        hook, after = cls.hook, cls.after
        cls.hook = cls.after = None
        try:
            if hook is not None:
                hook(name, value)
            super().__setattr__(name, value)
            if after is not None:
                after(name, value)
        finally:
            cls.hook, cls.after = hook, after


def _outcome(read: Callable[[], object]) -> object:
    try:
        return read()
    except (RuntimeError, KeyError) as error:
        return f"{type(error).__name__}: {error}"


@pytest.mark.parametrize(
    ("first", "then"),
    [("alone", "alone"), ("alone", "held"), ("held", "alone"), ("held", "held"), ("alone", "beside")],
    ids=[
        "standalone refit",
        "standalone to a pipeline step",
        "pipeline step to standalone",
        "pipeline refit",
        "standalone to a pipeline whose other step has its stage name",
    ],
)
def test_a_thread_sees_the_old_fit_or_the_new_one_between_any_two_writes_of_a_refit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, first: str, then: str
) -> None:
    old: Path = tmp_path / "old.parquet"
    new: Path = tmp_path / "new.parquet"
    pq.write_table(pa.table({"u": ["a", "b"], "v": ["p", "p"]}), old)
    pq.write_table(pa.table({"u": ["b", "c"], "v": ["q", "r"]}), new)
    record: dict[str, object] = {"u": "b", "v": "q"}
    encoder: _Interleaved = _Interleaved(columns=["u"])
    owners: dict[str, Pipeline | _Interleaved] = {
        "alone": encoder,
        "held": Pipeline([("enc", encoder)]),
        # Standalone, the stage name is the class name: "_interleaved".
        "beside": Pipeline([("enc", encoder), ("_interleaved", LabelEncoder(columns=["v"]))]),
    }
    owners[first].fit([old])

    def read() -> list[object]:
        twin: LabelEncoder = copy.copy(encoder)
        deep: LabelEncoder = copy.deepcopy(encoder)
        restored: LabelEncoder = pickle.loads(pickle.dumps(encoder))
        return [
            _outcome(lambda: encoder.transform_dict(record)),
            _outcome(lambda: twin.transform_dict(record)),
            _outcome(lambda: deep.transform_dict(record)),
            _outcome(lambda: restored.transform_dict(record)),
            _outcome(lambda: encoder.label_maps_),
            _outcome(lambda: encoder.vocab_size("u")),
        ]

    before: list[object] = read()
    seen: list[list[object]] = []
    monkeypatch.setattr(_Interleaved, "hook", lambda name, value: seen.append(read()))
    owners[then].fit([new])
    monkeypatch.setattr(_Interleaved, "hook", None)
    assert encoder._state_cache is None  # the refit released the state the probes read
    after: list[object] = read()
    assert before[4] == {"u": {"a": 1, "b": 2}}
    assert after[4] == {"u": {"b": 1, "c": 2}}
    assert seen
    for answers in seen:
        for answer, old_answer, new_answer in zip(answers, before, after, strict=True):
            assert answer in (old_answer, new_answer)


@pytest.mark.parametrize("held", [False, True], ids=["standalone", "held by a pipeline"])
@pytest.mark.parametrize("when", ["while the state is read", "before the read stores it", "after the read stores it"])
def test_a_fitted_attribute_read_that_a_refit_overtakes_leaves_no_stale_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, held: bool, when: str
) -> None:
    old: Path = tmp_path / "old.parquet"
    new: Path = tmp_path / "new.parquet"
    pq.write_table(pa.table({"u": ["a", "b"]}), old)
    pq.write_table(pa.table({"u": ["b", "c"]}), new)
    encoder: _Interleaved = _Interleaved(columns=["u"])
    owner: Pipeline | _Interleaved = Pipeline([("enc", encoder)]) if held else encoder
    owner.fit([old])
    refits: list[str] = []

    def refit() -> None:
        if not refits:
            refits.append(when)
            owner.fit([new])

    if when == "while the state is read":
        # The C++ core lets other threads run while it reads the state.
        handle, stage, standalone = encoder._require_binding()

        def stage_state(name: str) -> str:
            refit()
            return handle.stage_state(name)

        reading: Any = SimpleNamespace(stage_state=stage_state)
        encoder._bind(reading, stage, standalone=standalone)
    else:

        def refit_at_the_store(name: str, value: Any) -> None:
            if name == "_state_cache" and value is not None:
                refit()

        monkeypatch.setattr(
            _Interleaved, "hook" if when == "before the read stores it" else "after", refit_at_the_store
        )
    # The read began before the refit, so it answers for the old fit; the
    # next one answers for the new fit.
    assert encoder.label_maps_ == {"u": {"a": 1, "b": 2}}
    assert refits
    assert encoder.label_maps_ == {"u": {"b": 1, "c": 2}}


def test_a_fitted_attribute_read_that_binding_the_next_step_overtakes_leaves_no_stale_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # One transformer can be two steps of a pipeline: the fit binds it to the
    # first step, then to the second, on the same handle.
    path: Path = tmp_path / "u.parquet"
    pq.write_table(pa.table({"u": ["a", "b"]}), path)
    encoder: _Interleaved = _Interleaved(columns=["u"])
    Pipeline([("first", encoder), ("text", Cast({"u": "utf8"})), ("second", encoder)]).fit([path])
    handle: _core.FittedPipeline = encoder._require_handle()
    encoder._bind(handle, "first", standalone=False)  # as the fit leaves it between the two steps
    binds: list[str] = []

    def bind_the_second_step_before_the_read_stores(name: str, value: Any) -> None:
        if name == "_state_cache" and value is not None and not binds:
            binds.append(name)
            encoder._bind(handle, "second", standalone=False)

    monkeypatch.setattr(_Interleaved, "hook", bind_the_second_step_before_the_read_stores)
    assert encoder.label_maps_ == {"u": {"a": 1, "b": 2}}
    assert binds
    assert encoder.label_maps_ == {"u": {"1": 1, "2": 2}}


def test_generator_steps_are_materialized_before_validation() -> None:
    pipeline: Pipeline = Pipeline((name, Select(columns=["a"])) for name in ("s", "t"))
    assert len(pipeline) == 2
    assert [name for name, _ in pipeline.steps] == ["s", "t"]


def test_unpickled_root_binds_on_any_public_path(train_files: list[str]) -> None:
    restored: Pipeline = pickle.loads(pickle.dumps(_nested().fit(train_files)))
    encoder = restored.steps[1][1]  # the public attribute, not __getitem__
    assert isinstance(encoder, LabelEncoder)
    assert encoder.label_maps_["device"]["mob"] == MOB_INDEX
    features = restored.steps[0][1]
    assert isinstance(features, Pipeline)
    assert features._handle is restored._handle


def test_unpickled_root_binds_safely_under_concurrent_first_use(train_files: list[str]) -> None:
    payload: bytes = pickle.dumps(_nested().fit(train_files))
    record: dict[str, object] = {"device": "mob", "source": "a"}
    for _ in range(20):
        restored: Pipeline = pickle.loads(payload)

        def probe(index: int, restored: Pipeline = restored) -> object:
            if index % 2 == 0:
                return restored.transform_dict(record)["device"]
            encoder = restored["encode"]
            assert isinstance(encoder, LabelEncoder)
            return encoder.vocab_size("device")

        with ThreadPoolExecutor(max_workers=16) as pool:
            results = list(pool.map(probe, range(16)))
        assert results == [MOB_INDEX, 4] * 8
        features = restored["features"]
        assert isinstance(features, Pipeline)
        assert features._handle is restored._handle


def test_pickle_after_refit_serializes_the_current_fit(train_files: list[str], tmp_path: Path) -> None:
    outer: Pipeline = _nested().fit(train_files)
    inner = outer["features"]
    assert isinstance(inner, Pipeline)
    pickle.dumps((inner, outer))  # pickles the nested pipeline before the root
    other: Path = tmp_path / "other.parquet"
    pq.write_table(pa.table({"device": ["zzz", "desk"], "source": ["a", "b"]}), other)
    outer.fit([str(other)])
    restored: Pipeline = pickle.loads(pickle.dumps(inner))
    assert isinstance(restored["select"], Select)  # binds the view to its artifact
    handle = restored._handle
    assert handle is not None
    assert json.loads(handle.manifest_json())["ops"][1]["state"]["label_maps"]["device"] == {"desk": 1, "zzz": 2}


def test_shallow_copy_shares_the_handle_and_leaves_the_original_bound(train_files: list[str]) -> None:
    pipeline: Pipeline = _nested().fit(train_files)
    clone: Pipeline = copy.copy(pipeline)
    assert clone.transform_dict({"device": "mob", "source": "a"}) == {"device": MOB_INDEX, "source": 1}
    assert clone._handle is pipeline._handle
    encoder = pipeline["encode"]
    assert isinstance(encoder, LabelEncoder)
    assert encoder._require_handle() is pipeline._handle


def test_numpy_integers_are_accepted_for_counts_and_threads(train_files: list[str]) -> None:
    count: Any = np.int64(2)
    threads: Any = np.int32(1)
    encoder: LabelEncoder = LabelEncoder(columns=["device"], min_count=count).fit(train_files, threads=threads)
    assert encoder.min_count == 2
    assert encoder.label_maps_["device"] == {"mob": 1}


def test_fitted_state_is_read_once_per_binding(train_files: list[str]) -> None:
    encoder: LabelEncoder = LabelEncoder(columns=["device"]).fit(train_files)
    first = encoder.label_maps_
    assert encoder.label_maps_ is first
    assert encoder.vocab_size("device") == len(first["device"]) + 1
    encoder.fit(train_files)  # a new binding invalidates the cached view
    assert encoder.label_maps_ is not first
    assert encoder.label_maps_ == first
