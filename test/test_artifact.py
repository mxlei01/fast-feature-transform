"""Tests for artifact integrity: provenance, validation on load, and how engine errors surface."""

import collections
import datetime
import enum
import functools
import importlib.metadata
import inspect
import itertools
import json
import math
import pickle
import re
import subprocess  # nosec B404
import sys
import types
import weakref
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import fast_feature_transform
from fast_feature_transform import (
    BackwardDifferenceEncoder,
    BaseNEncoder,
    BinaryEncoder,
    Cast,
    CatBoostEncoder,
    CountEncoder,
    EngineError,
    GrayEncoder,
    HelmertEncoder,
    JamesSteinEncoder,
    LeaveOneOutEncoder,
    MEstimateEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    PolynomialEncoder,
    PowerTransformer,
    QuantileEncoder,
    RankHotEncoder,
    SumEncoder,
    SummaryEncoder,
    TargetEncoder,
    WOEEncoder,
    _core,
)
from fast_feature_transform.pipeline import _SHELLS, Pipeline
from fast_feature_transform.transformers import LabelEncoder, Select, StandardScaler, Transformer
from helpers import write_parquet

INT64_MAX: int = 2**63 - 1


@pytest.fixture
def source(tmp_path: Path) -> str:
    """Write a small Parquet file with one utf8 column and one float column.

    Args:
        tmp_path: pytest-provided temporary directory.

    Returns:
        The file path.

    """
    path: Path = tmp_path / "train.parquet"
    pq.write_table(pa.table({"device": ["ios", "web", "ios"], "price": [1.0, 2.0, 3.0]}), path)
    return str(path)


def _manifest(pipeline: Pipeline) -> dict[str, Any]:
    """Return a fitted pipeline's manifest as a dict.

    Args:
        pipeline: A fitted pipeline.

    Returns:
        The parsed manifest.

    """
    return json.loads(pipeline._require_handle().manifest_json())


def test_engine_error_is_a_runtime_error_exported_from_the_package(source: str) -> None:
    assert issubclass(EngineError, RuntimeError)
    with pytest.raises(EngineError, match="not found in input"):
        LabelEncoder(columns=["nope"]).fit([source])


def test_manifest_records_the_package_version(source: str) -> None:
    manifest = _manifest(Pipeline([("e", LabelEncoder(columns=["device"]))]).fit([source]))
    assert manifest["library_version"] == importlib.metadata.version("fast-feature-transform")
    assert manifest["library_version"] == _core.library_version()


def test_provenance_survives_load_save_and_pickle(source: str, tmp_path: Path) -> None:
    pipeline: Pipeline = Pipeline([("e", LabelEncoder(columns=["device"]))]).fit([source])
    manifest = _manifest(pipeline)
    manifest["library_version"] = "9.9.9"
    manifest["created_at"] = "1999-01-01T00:00:00Z"
    artifact: Path = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "manifest.json").write_text(json.dumps(manifest))

    loaded: Pipeline = Pipeline.load(artifact)
    loaded.save(tmp_path / "copy")
    copied = json.loads((tmp_path / "copy" / "manifest.json").read_text())
    assert (copied["library_version"], copied["created_at"]) == ("9.9.9", "1999-01-01T00:00:00Z")
    restored: Pipeline = pickle.loads(pickle.dumps(loaded))
    assert _manifest(restored)["library_version"] == "9.9.9"


def test_unknown_schema_version_is_rejected_on_load(source: str) -> None:
    manifest = _manifest(Pipeline([("e", LabelEncoder(columns=["device"]))]).fit([source]))
    manifest["schema_version"] = "42.0.0"
    with pytest.raises(EngineError, match="schema_version '42.0.0' is not supported"):
        _core.FittedPipeline.load_json(json.dumps(manifest))


def test_malformed_manifests_are_engine_errors_with_context() -> None:
    with pytest.raises(EngineError, match="invalid manifest"):
        _core.FittedPipeline.load_json("not json")
    with pytest.raises(EngineError, match="'ops' must be a list"):
        _core.FittedPipeline.load_json(json.dumps({"schema_version": "0.3.0"}))
    with pytest.raises(EngineError, match="op 'sel'"):
        _core.FittedPipeline.load_json(
            json.dumps({"schema_version": "0.3.0", "ops": [{"stage": "sel", "kernel": "select", "state": {}}]})
        )
    with pytest.raises(EngineError, match="select: column 'a' listed more than once"):
        _core.FittedPipeline.load_json(
            json.dumps(
                {
                    "schema_version": "0.3.0",
                    "ops": [{"stage": "sel", "kernel": "select", "params": {"columns": ["a", "a"]}, "state": {}}],
                }
            )
        )


def test_loaded_label_maps_are_validated(source: str) -> None:
    manifest = _manifest(Pipeline([("e", LabelEncoder(columns=["device"]))]).fit([source]))
    manifest["ops"][0]["state"]["label_maps"]["device"] = {"ios": 1, "web": 500}
    with pytest.raises(EngineError, match="exactly 1..2"):
        _core.FittedPipeline.load_json(json.dumps(manifest))


def test_select_and_cast_refuse_a_state_a_manifest_gives_them(source: str) -> None:
    # REGRESSION: select and cast learn nothing, so a manifest that gives
    # either a state other than an empty object is refused, as it is for the
    # other steps that learn nothing. Such a state used to load and be ignored.
    pipeline: Pipeline = Pipeline([("s", Select(columns=["device", "price"])), ("c", Cast({"price": "f32"}))])
    manifest: dict[str, Any] = _manifest(pipeline.fit([source]))
    assert [op["state"] for op in manifest["ops"]] == [{}, {}]
    record: dict[str, Any] = {"device": "ios", "price": 1.5}
    assert _core.FittedPipeline.load_json(json.dumps(manifest)).transform_row(record) == record
    for index, stage, kernel in ((0, "s", "select"), (1, "c", "cast")):
        for state in ({"columns": ["price"]}, [], None):
            edited: dict[str, Any] = json.loads(json.dumps(manifest))
            edited["ops"][index]["state"] = state
            refusal: str = f"{stage}: {kernel} learns nothing, so its 'state' must be an empty object"
            with pytest.raises(EngineError, match=f"^{re.escape(refusal)}$"):
                _core.FittedPipeline.load_json(json.dumps(edited))


def test_loaded_explicit_columns_keep_their_order(tmp_path: Path) -> None:
    path: Path = tmp_path / "two.parquet"
    pq.write_table(pa.table({"device": ["a"], "country": ["b"]}), path)
    pipeline: Pipeline = Pipeline([("e", LabelEncoder(columns=["device", "country"]))]).fit([str(path)])
    pipeline.save(tmp_path / "artifact")
    loaded: Pipeline = Pipeline.load(tmp_path / "artifact")
    encoder = loaded["e"]
    assert isinstance(encoder, LabelEncoder)
    assert encoder.columns == ["device", "country"]
    loaded.save(tmp_path / "again")
    original = json.loads((tmp_path / "artifact" / "manifest.json").read_text())
    again = json.loads((tmp_path / "again" / "manifest.json").read_text())
    assert original == again


def test_non_utf8_category_fails_at_fit_not_at_save(tmp_path: Path) -> None:
    values: pa.Array = pa.array([b"\xff\xfe", b"ok"], type=pa.binary()).cast(pa.string(), safe=False)
    path: Path = tmp_path / "bad.parquet"
    pq.write_table(pa.table({"device": values}), path)
    with pytest.raises(EngineError, match="not valid UTF-8"):
        LabelEncoder(columns=["device"]).fit([str(path)])


def test_kernel_params_are_validated_by_the_core_too() -> None:
    def manifest(kernel: str, params: dict[str, Any]) -> str:
        return json.dumps({"schema_version": "0.3.0", "ops": [{"stage": "s", "kernel": kernel, "params": params}]})

    with pytest.raises(EngineError, match=r"'min_count' must be an integer from 1 to 2\^63 - 1"):
        _core.FittedPipeline.load_json(manifest("label_encode", {"columns": ["device"], "min_count": 0}))
    with pytest.raises(EngineError, match="'columns' must be a non-empty list"):
        _core.FittedPipeline.load_json(manifest("label_encode", {"columns": [], "min_count": 1}))
    with pytest.raises(EngineError, match="'columns' must be a non-empty list"):
        _core.FittedPipeline.load_json(manifest("select", {"columns": []}))


class _MisspelledScaler(StandardScaler):
    """A StandardScaler whose plan spells ``with_mean`` as ``with_means``."""

    def params(self) -> dict[str, Any]:
        """Kernel parameters with one name misspelled.

        Returns:
            StandardScaler's parameters, with ``with_mean`` renamed ``with_means``.

        """
        named: dict[str, Any] = super().params()
        named["with_means"] = named.pop("with_mean")
        return named


MISSPELLED: str = "standard_scale: unknown parameter 'with_means' (accepted: columns, with_mean, with_std)"
# The fixture's prices are 1, 2 and 3: their mean is 2 and their variance 2/3,
# so the mean itself scales to this without centering and to 0 with it.
UNCENTERED_MEAN: float = 2.0 / math.sqrt(2.0 / 3.0)


# REGRESSION: a parameter name the kernel does not know is an error. The core
# used to read the names it knew and pass over every other one, so a plan or a
# manifest that spelled with_mean as with_means fitted or loaded without an
# error and centered the column, as the default with_mean=True does.
def test_a_plan_that_misspells_a_parameter_name_is_refused_at_fit(source: str) -> None:
    with pytest.raises(EngineError) as refused:
        Pipeline([("scale", _MisspelledScaler(columns=["price"], with_mean=False))]).fit([source])
    assert str(refused.value) == MISSPELLED
    with pytest.raises(EngineError) as refused:
        _MisspelledScaler(columns=["price"], with_mean=False).fit([source])
    assert str(refused.value) == MISSPELLED
    # Under its own name the parameter is read: the column is not centered.
    scaler: StandardScaler = StandardScaler(columns=["price"], with_mean=False).fit([source])
    assert scaler.transform_dict({"price": 2.0}) == {"price": UNCENTERED_MEAN}


def test_a_manifest_that_misspells_a_parameter_name_is_refused_at_load(source: str, tmp_path: Path) -> None:
    pipeline: Pipeline = Pipeline([("scale", StandardScaler(columns=["price"], with_mean=False))]).fit([source])
    pipeline.save(tmp_path / "saved")
    assert Pipeline.load(tmp_path / "saved").transform_dict({"price": 2.0}) == {"price": UNCENTERED_MEAN}

    path: Path = tmp_path / "saved" / "manifest.json"
    manifest: dict[str, Any] = json.loads(path.read_text())
    params: dict[str, Any] = manifest["ops"][0]["params"]
    assert params == {"columns": ["price"], "with_mean": False, "with_std": True}
    params["with_means"] = params.pop("with_mean")
    path.write_text(json.dumps(manifest))
    with pytest.raises(EngineError) as refused:
        Pipeline.load(tmp_path / "saved")
    assert str(refused.value) == MISSPELLED
    with pytest.raises(EngineError) as refused:
        _core.FittedPipeline.load_json(json.dumps(manifest))
    assert str(refused.value) == MISSPELLED


# 'c' holds nulls and two categories seen once, which min_group_size=2 folds
# into one group.
_FRAME: dict[str, list[Any]] = {
    "c": ["a", "a", "b", "b", "b", "r", None, None, "a", "q"],
    "y": [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
    "x": [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 9.0, 12.0],
}
_ENCODE: dict[str, Any] = {"columns": ["c"]}
_TARGET: dict[str, Any] = {"columns": ["c"], "target": "y"}

# The entries of a fitted state that every version writes and that a kernel
# used to read as optional, each with a step that writes it.
_ALWAYS_WRITTEN: list[tuple[type[Transformer], dict[str, Any], str]] = [
    (LabelEncoder, _ENCODE, "vocab_sizes"),
    (CountEncoder, _ENCODE | {"min_group_size": 2}, "null_counts"),
    (CountEncoder, _ENCODE | {"min_group_size": 2}, "groups"),
    *[
        (encoder, _ENCODE, "has_null")
        for encoder in (
            OrdinalEncoder,
            OneHotEncoder,
            RankHotEncoder,
            BaseNEncoder,
            BinaryEncoder,
            GrayEncoder,
            BackwardDifferenceEncoder,
            HelmertEncoder,
            SumEncoder,
            PolynomialEncoder,
        )
    ],
    *[
        (encoder, _TARGET, "null_values")
        for encoder in (
            TargetEncoder,
            MEstimateEncoder,
            WOEEncoder,
            JamesSteinEncoder,
            LeaveOneOutEncoder,
            CatBoostEncoder,
        )
    ],
    *[
        (PowerTransformer, {"columns": ["x"], "standardize": standardize}, entry)
        for standardize in (True, False)
        for entry in ("means", "scales")
    ],
]


def _fitted_manifest(shell: type[Transformer], arguments: dict[str, Any], tmp_path: Path) -> dict[str, Any]:
    """Fit one step named 'step' on the frame above and return its manifest.

    Args:
        shell: The transformer class.
        arguments: Its constructor's arguments.
        tmp_path: Directory for the training file.

    Returns:
        The parsed manifest, which loads as saved.

    """
    path: str = write_parquet(tmp_path / "train.parquet", pa.table(_FRAME))
    manifest: dict[str, Any] = _manifest(Pipeline([("step", shell(**arguments))]).fit([path]))
    _core.FittedPipeline.load_json(json.dumps(manifest))
    return manifest


def _refusal(manifest: dict[str, Any], edit: Callable[[dict[str, Any]], object]) -> str:
    """Load a copy of a manifest whose one state an edit changed, and return the error.

    Args:
        manifest: A manifest of one op.
        edit: Changes the op's state in place.

    Returns:
        The message of the EngineError that loading raises.

    """
    edited: dict[str, Any] = json.loads(json.dumps(manifest))
    edit(edited["ops"][0]["state"])
    with pytest.raises(EngineError) as refused:
        _core.FittedPipeline.load_json(json.dumps(edited))
    return str(refused.value)


# REGRESSION: an entry of a fitted state that every saved pipeline holds is
# required at load, so a manifest that drops it or misspells its key fails and
# names the step and the entry. Such a manifest used to load. Without
# has_null, null_values, null_counts or groups, a null or a folded category then
# got another answer; without vocab_sizes, or without the null means and scales
# of a step that does not standardize, no answer changed.
@pytest.mark.parametrize(
    ("shell", "arguments", "entry"),
    _ALWAYS_WRITTEN,
    ids=[
        f"{shell.__name__}-{entry}" + ("" if arguments.get("standardize", True) else "-unstandardized")
        for shell, arguments, entry in _ALWAYS_WRITTEN
    ],
)
def test_a_manifest_without_a_state_entry_every_version_writes_is_refused(
    shell: type[Transformer], arguments: dict[str, Any], entry: str, tmp_path: Path
) -> None:
    manifest: dict[str, Any] = _fitted_manifest(shell, arguments, tmp_path)
    if shell is PowerTransformer:
        # Without standardize, 'means' and 'scales' are saved as null, and
        # null is accepted in place of the object for these two alone.
        assert (manifest["ops"][0]["state"][entry] is None) == (not arguments["standardize"])
        missing: str = f"step: state must hold a '{entry}' object or null"
    else:
        # The frame gives the entry something to say about 'c': a null
        # category, a folded group or a vocabulary size.
        saved: Any = manifest["ops"][0]["state"][entry]["c"]
        assert saved is not None
        assert saved is not False
        missing = f"step: state must hold a '{entry}' object"
        assert _refusal(manifest, lambda state: state.update({entry: None})) == f"step: '{entry}' must be an object"
    assert _refusal(manifest, lambda state: state.pop(entry)) == missing
    assert _refusal(manifest, lambda state: state.update({entry + "s": state.pop(entry)})) == missing


# REGRESSION: the quantile encoders write a null encoding for every output,
# null when its column had no null category, and a manifest that lacks one is
# refused. Dropping every entry of a column used to load, and a null then
# encoded to the prior.
@pytest.mark.parametrize("shell", [QuantileEncoder, SummaryEncoder])
def test_a_manifest_without_the_null_encoding_of_an_output_is_refused(shell: type[Transformer], tmp_path: Path) -> None:
    manifest: dict[str, Any] = _fitted_manifest(shell, _TARGET, tmp_path)
    outputs: list[str] = list(manifest["ops"][0]["state"]["nulls"])
    for dropped in [[output] for output in outputs] + [outputs]:

        def drop(state: dict[str, Any], dropped: list[str] = dropped) -> None:
            for output in dropped:
                del state["nulls"][output]

        assert _refusal(manifest, drop) == f"step: nulls has no entry for column '{dropped[0]}'"


# What each constructor requires, by argument name.
_REQUIRED: dict[str, Any] = {
    "columns": ["x"],
    "types": {"x": "f64"},
    "key": "k",
    "aggs": {"x": ["sum"]},
    "target": "y",
}


# For every class, the names params() sends, the names its kernel reports and
# the constructor's arguments are one set. The core refuses a name the kernel
# does not report, so a name that only params() sends fails every fit.
# Pipeline.load builds each step from the names the kernel reports, so a name
# the constructor does not take fails every load.
def test_every_transformer_sends_the_names_its_kernel_reports_and_takes_them_back() -> None:
    exported: list[Any] = [getattr(fast_feature_transform, name) for name in fast_feature_transform.__all__]
    classes: set[Any] = {each for each in exported if isinstance(each, type) and issubclass(each, Transformer)}
    assert set(_SHELLS.values()) == classes - {Transformer}
    for kernel, shell in _SHELLS.items():
        arguments: list[inspect.Parameter] = list(inspect.signature(shell).parameters.values())
        required: list[str] = [each.name for each in arguments if each.default is each.empty]
        step: Transformer = shell(**{name: _REQUIRED[name] for name in required})
        sent: dict[str, Any] = step.params()
        assert set(sent) == {each.name for each in arguments}, kernel
        # The kernel names what it accepts when it refuses a name.
        op: dict[str, Any] = {"stage": "s", "kernel": kernel, "params": sent | {"no_such_parameter": 1}, "state": {}}
        with pytest.raises(EngineError) as refused:
            _core.FittedPipeline.load_json(json.dumps({"schema_version": "0.3.0", "ops": [op]}))
        accepted: str = ", ".join(sorted(sent))
        assert str(refused.value) == f"{kernel}: unknown parameter 'no_such_parameter' (accepted: {accepted})"


def test_transform_dict_rejects_unconvertible_values_by_name(source: str) -> None:
    pipeline: Pipeline = Pipeline([("e", LabelEncoder(columns=["device"]))]).fit([source])
    with pytest.raises(EngineError, match="'big' does not fit in a 64-bit integer"):
        pipeline.transform_dict({"device": "ios", "big": INT64_MAX + 1})
    non_str_key: dict[Any, Any] = {"device": "ios", 1: 1}
    with pytest.raises(EngineError, match="record keys must be str"):
        pipeline.transform_dict(non_str_key)
    with pytest.raises(EngineError, match="record keys must be str"):
        pipeline.transform_dict({"device": "ios", "\ud800": 1})
    with pytest.raises(EngineError, match="'device' is not encodable as UTF-8"):
        pipeline.transform_dict({"device": "\ud800"})
    with pytest.raises(EngineError, match="'blob' must be str, int, float, bool, datetime, or None \\(got bytes\\)"):
        pipeline.transform_dict({"device": "ios", "blob": b"x"})


def test_transform_dict_refuses_a_record_that_is_not_a_dict(source: str) -> None:
    # REGRESSION: a record that is not a dict raises a TypeError that names
    # transform_dict and the record's type. The TypeError used to name the
    # internal transform_row and its handle type.
    pipeline: Pipeline = Pipeline([("s", Select(columns=["device"]))]).fit([source])
    standalone: Select = Select(columns=["device"]).fit([source])
    records: list[tuple[Any, str]] = [
        ([("device", "ios")], "list"),
        (None, "NoneType"),
        (types.MappingProxyType({"device": "ios"}), "mappingproxy"),
        (collections.UserDict(device="ios"), "UserDict"),
    ]
    for record, name in records:
        for owner in (pipeline, standalone):
            with pytest.raises(TypeError) as refused:
                owner.transform_dict(record)
            assert str(refused.value) == f"transform_dict: X must be a dict, got {name}"
    # A subclass of dict is a dict.
    assert pipeline.transform_dict(collections.OrderedDict(device="ios")) == {"device": "ios"}
    assert standalone.transform_dict(collections.OrderedDict(device="ios")) == {"device": "ios"}


class _OwnHash(str):
    """A str whose hash is not its text's, so a dict keeps it beside a str of the same text."""

    __slots__ = ()

    def __hash__(self) -> int:
        """Hash the text, then flip the lowest bit.

        Returns:
            A hash that differs from the text's.

        """
        return hash(str.__str__(self)) ^ 1


class _OwnEquality(str):
    """A str that equals only its own kind, so a dict keeps it beside a str of the same text."""

    __slots__ = ()
    __hash__ = str.__hash__

    def __eq__(self, other: object) -> bool:
        """Compare as text, with another _OwnEquality only.

        Args:
            other: The object compared with.

        Returns:
            Whether other is an _OwnEquality of the same text.

        """
        return type(other) is _OwnEquality and str.__eq__(self, other)


class _InheritedHash(_OwnHash):
    """An _OwnHash subclass that defines no hash: its type takes the hash of _OwnHash, not of str."""

    __slots__ = ()


class _Bare(str):
    """A str subclass that keeps the hash and comparison of str."""

    __slots__ = ()


class _Texts(enum.StrEnum):
    """The key texts of the test below, as StrEnum members."""

    A = "a"
    ACUTE = "é"
    DEVICE = "device"


class _MixedTexts(str, enum.Enum):
    """The key texts of the test below, as members of an Enum with a str mixin."""

    A = "a"
    ACUTE = "é"
    DEVICE = "device"


def test_transform_dict_refuses_a_record_whose_keys_repeat_a_text(source: str) -> None:
    # REGRESSION: a record whose keys hold one text twice is an error that
    # names the text. A step used to read one of the two fields, which one
    # depending on the order of the keys, the width of the record and the
    # step, and the output dict kept the value of the later field.
    pipeline: Pipeline = Pipeline([("s", Select(columns=["device"]))]).fit([source])
    # Each record holds "device" and then one to three keys, of every text and
    # kind in every order; one text is not ASCII. A str, a numpy.str_, a
    # StrEnum member, a member of an Enum with a str mixin and a _Bare of one
    # text are one key of a dict, since their types use the hash and
    # comparison of str. An _OwnHash, an _InheritedHash and an _OwnEquality
    # are each a key of their own beside those.
    kinds: list[type[str]] = [str, np.str_, _Texts, _MixedTexts, _Bare, _OwnHash, _InheritedHash, _OwnEquality]
    keys: list[str] = [kind(text) for text in ("a", "é", "device") for kind in kinds]
    refused_records: int = 0
    for count in (1, 2, 3):
        for chosen in itertools.product(keys, repeat=count):
            record: dict[str, Any] = {"device": "ios"}
            for index, key in enumerate(chosen):
                record[key] = f"v{index}"
            texts: list[str] = [str.__str__(key) for key in record]
            repeated: set[str] = {text for text in texts if texts.count(text) > 1}
            if not repeated:
                plain: dict[str, Any] = {str.__str__(key): value for key, value in record.items()}
                assert pipeline.transform_dict(record) == pipeline.transform_dict(plain), chosen
                continue
            refused_records += 1
            with pytest.raises(EngineError) as refused:
                pipeline.transform_dict(record)
            assert str(refused.value) in {f"record holds the key '{text}' twice" for text in repeated}, chosen
    assert refused_records > 0


class _ChangesOnIndex:
    """An integer-like value whose __index__ changes its record."""

    def __init__(self, change: Callable[[], Any]) -> None:
        """Hold the change.

        Args:
            change: Changes the record this value is in.

        """
        self.change: Callable[[], Any] = change

    def __index__(self) -> int:
        """Change the record, then answer.

        Returns:
            1.

        """
        self.change()
        return 1


@pytest.mark.parametrize(
    "value",
    [
        0.5,
        np.int64(1),
        np.float32(0.5),
        np.bool_(True),
        np.array(3),
        datetime.datetime(2021, 6, 1),  # noqa: DTZ001
        datetime.datetime(2021, 6, 1, tzinfo=datetime.UTC),
        datetime.datetime(2021, 6, 1, tzinfo=datetime.timezone(datetime.timedelta(hours=2))),
        np.datetime64("2021-06-01T00:00", "us"),
        _ChangesOnIndex(lambda: None),
    ],
    ids=[
        "float",
        "int64",
        "float32",
        "numpy_bool",
        "numpy_array",
        "datetime",
        "datetime_utc",
        "datetime_offset",
        "datetime64",
        "python_class",
    ],
)
def test_transform_dict_does_not_check_keys_and_values_that_cannot_repeat_a_key(source: str, value: Any) -> None:
    # The names are checked only for a key whose type has its own hash or
    # comparison. A value does not set off the check, whatever its type:
    # Python code that reading it runs cannot repeat a key, since such a value
    # is read only after every field of the record has been read. Records of
    # StrEnum members and numpy.str_ values do not pay for the check. A key
    # whose class changes after it is stored breaks the rule for dict keys:
    # the record then holds one text twice, and the check, which does not run
    # for such a record, does not see it.
    pipeline: Pipeline = Pipeline([("s", Select(columns=["device"]))]).fit([source])
    key: str = _OwnHash("device")
    record: dict[str, Any] = {"device": "ios", key: "android", "v": value}
    object.__setattr__(key, "__class__", _Bare)
    assert [type(each) for each in record] == [str, _Bare, str]
    assert list(pipeline.transform_dict(record)) == ["device"]


class _ChangesOnAsm8(datetime.datetime):
    """A datetime that changes its record when asked for a pandas Timestamp's asm8, and then has none."""

    change: Callable[[], Any]

    def __getattr__(self, name: str) -> Any:
        """Change the record when asked for asm8, then refuse the attribute.

        Args:
            name: The attribute asked for.

        Raises:
            AttributeError: Always.

        """
        if name == "asm8":
            self.change()
        raise AttributeError(name)


class _ChangesOnUtcoffset(datetime.tzinfo):
    """A tzinfo whose utcoffset changes the record of its datetime."""

    def __init__(self, change: Callable[[], Any]) -> None:
        """Hold the change.

        Args:
            change: Changes the record that holds a datetime in this zone.

        """
        self.change: Callable[[], Any] = change

    def utcoffset(self, dt: datetime.datetime | None, /) -> datetime.timedelta:
        """Change the record, then answer.

        Args:
            dt: The datetime asked about.

        Returns:
            One hour.

        """
        self.change()
        return datetime.timedelta(hours=1)


class _ChangesFirst:
    """A function that changes a record and then answers as another function."""

    def __init__(self, change: Callable[[], Any], call: Callable[..., Any]) -> None:
        """Hold the change and the function.

        Args:
            change: Changes the record.
            call: The function that answers.

        """
        self.change: Callable[[], Any] = change
        self.call: Callable[..., Any] = call

    def __call__(self, *args: Any) -> Any:
        """Change the record, then answer.

        Args:
            *args: Passed to the function.

        Returns:
            What the function returns.

        """
        self.change()
        return self.call(*args)


def _rebuild(record: dict[str, Any]) -> None:
    """Add 40 keys to a record of five, enough to rebuild its dict's table.

    Args:
        record: The record.

    """
    for index in range(40):
        record[f"g{index}"] = index


def _take_out_a_and_rebuild(record: dict[str, Any]) -> None:
    """Take the key "a" out of a record, then rebuild its table.

    Args:
        record: The record.

    """
    del record["a"]
    _rebuild(record)


def _move_a(record: dict[str, Any]) -> None:
    """Take the key "a" out of a record and store it again, last.

    Args:
        record: The record.

    """
    record["a"] = record.pop("a")


def _move_a_and_rebuild(record: dict[str, Any]) -> None:
    """Take the key "a" out of a record, rebuild its table, and store "a" again, last.

    Args:
        record: The record.

    """
    a: Any = record.pop("a")
    _rebuild(record)
    record["a"] = a


# Each changes a record that holds "a", "device", "b", "c" and "price", in
# that order, while "b" is read.
_CHANGES: dict[str, Callable[[dict[str, Any]], Any]] = {
    "take_out_a_read_key": lambda record: record.pop("a"),
    "take_out_a_later_key": lambda record: record.pop("c"),
    "add_a_key": lambda record: record.update(added=1.0),
    "rebuild_the_table": _rebuild,
    "take_out_a_read_key_and_rebuild": _take_out_a_and_rebuild,
    "move_a_read_key": _move_a,
    "move_a_read_key_and_rebuild": _move_a_and_rebuild,
    "store_another_value": lambda record: record.update(c=99.0),
    "clear": lambda record: record.clear(),
}

# 2021-06-01 00:00 in epoch microseconds.
_MIDNIGHT: int = (
    datetime.datetime(2021, 6, 1, tzinfo=datetime.UTC) - datetime.datetime(1970, 1, 1, tzinfo=datetime.UTC)
) // datetime.timedelta(microseconds=1)


def _changing_value(
    route: str, change: Callable[[], Any], monkeypatch: pytest.MonkeyPatch, datetime_data: Callable[..., Any]
) -> tuple[Any, int, Any]:
    """Make a value whose reading runs a change once, through one route.

    Args:
        route: How reading the value runs the change.
        change: The change.
        monkeypatch: Replaces numpy.datetime_data for the route through it.
        datetime_data: numpy's own datetime_data.

    Returns:
        The value, what the record path reads it as, and an object to keep
        alive while the value is read.

    """
    if route == "weakref_proxy":
        referent: _ChangesOnIndex = _ChangesOnIndex(change)
        return weakref.proxy(referent), 1, referent
    if route == "named_numpy":
        # The C name of a class made in Python is its __name__.
        named: Any = type("numpy.int64", (_ChangesOnIndex,), {"__module__": "numpy"})(change)
        return named, 1, named
    if route == "tzinfo":
        zoned: datetime.datetime = datetime.datetime(2021, 6, 1, tzinfo=_ChangesOnUtcoffset(change))
        return zoned, _MIDNIGHT - 3_600_000_000, zoned
    if route == "asm8":
        stamp: _ChangesOnAsm8 = _ChangesOnAsm8(2021, 6, 1)
        stamp.change = change
        return stamp, _MIDNIGHT, stamp
    if route == "datetime_data":
        monkeypatch.setattr(np, "datetime_data", _ChangesFirst(change, datetime_data))
        moment: np.datetime64 = np.datetime64("2021-06-01T00:00", "us")
        return moment, _MIDNIGHT, moment
    value: _ChangesOnIndex = _ChangesOnIndex(change)
    return value, 1, value


@pytest.mark.parametrize("route", ["index", "weakref_proxy", "named_numpy", "tzinfo", "asm8", "datetime_data"])
def test_transform_dict_reads_a_record_as_it_was_when_a_value_changes_it(
    source: str, route: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # REGRESSION: a value whose own code, run while the record is read,
    # changes the record does not change what is read: the record is read as
    # it was when the call began. The walk over the dict used to read what
    # the record held at each step. It read a key that was taken out and
    # stored again a second time, and refused the record; it read a key added
    # and a value stored after the call began; it missed a key taken out
    # before it was read; and when a key was taken out and enough keys added
    # to rebuild the dict's table, it skipped a later key that the record
    # held throughout. The code is Python code that reading the value
    # reaches: a method of the value's class, also when the class carries
    # numpy's name and module; the __index__ of the object that a weakref
    # proxy refers to; a tzinfo's utcoffset; the lookup of a pandas
    # Timestamp's asm8 on a datetime; and numpy.datetime_data, which reading a
    # datetime64 calls.
    pipeline: Pipeline = Pipeline([("s", StandardScaler(columns=["price"]))]).fit([source])
    datetime_data: Callable[..., Any] = np.datetime_data
    for name, change in _CHANGES.items():
        record: dict[str, Any] = {}
        value, read_as, _alive = _changing_value(route, functools.partial(change, record), monkeypatch, datetime_data)
        record.update({"a": 1.0, "device": "ios", "b": value, "c": 3.0, "price": 2.0})
        items: list[tuple[str, Any]] = list(record.items())
        out: dict[str, Any] = pipeline.transform_dict(record)
        # The price at fit has mean 2.0, so the scaler reads 2.0 as 0.0.
        assert list(out.items()) == [("a", 1.0), ("device", "ios"), ("b", read_as), ("c", 3.0), ("price", 0.0)], name
        assert list(record.items()) != items, name  # the change ran


def test_manifest_with_invalid_bytes_is_an_engine_error(tmp_path: Path) -> None:
    artifact: Path = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "manifest.json").write_bytes(b'{"schema_version":"0.3.0","ops":[{"stage":"i\xffs"}]}')
    with pytest.raises(EngineError, match="invalid manifest"):
        Pipeline.load(artifact)
    with pytest.raises(EngineError, match="'library_version' must be a string"):
        _core.FittedPipeline.load_json(json.dumps({"schema_version": "0.3.0", "library_version": 5, "ops": []}))


def test_transform_dict_rejects_lookalike_numerics_by_type(source: str) -> None:
    pipeline: Pipeline = Pipeline([("s", Select(columns=["device", "price"]))]).fit([source])
    for value, name in ((np.complex128(1 + 2j), "complex128"), (np.array(3.5), "ndarray"), (Decimal("1.5"), "Decimal")):
        with pytest.raises(EngineError, match=f"\\(got {name}\\)"):
            pipeline.transform_dict({"device": "ios", "price": value})
    assert pipeline.transform_dict({"device": "ios", "price": np.array(3)}) == {"device": "ios", "price": 3}


class bool_:
    """A value that carries the name of numpy's bool, and refuses a truth value."""

    def __bool__(self) -> bool:
        """Refuse a truth value.

        Raises:
            ValueError: Always.

        """
        raise ValueError("no truth value")


class _NameIsNotText(type):
    """A metaclass that gives its classes a __name__ that is not a str."""

    @property
    def __name__(self) -> Any:
        """Answer a number.

        Returns:
            5.

        """
        return 5


class _Count:
    """What a lookalike datetime64's astype returns: its item is 5."""

    def item(self) -> int:
        """Answer the count.

        Returns:
            5.

        """
        return 5


def _astype(_value: Any, _kind: str) -> _Count:
    """Answer as numpy's astype does, with an object whose item is 5.

    Args:
        _value: The value.
        _kind: The type asked for.

    Returns:
        A _Count.

    """
    return _Count()


def test_transform_dict_reads_a_numpy_value_only_from_numpy_types(source: str) -> None:
    # REGRESSION: a value is read as numpy's bool, datetime64 or floating
    # type only when its type is numpy's own. Any other type is not
    # supported: one that carries numpy's name, also with numpy as its module
    # or as part of its name, and a subclass of a numpy type. The record path
    # used to go by the type's __name__ alone: a class named bool_ without
    # __bool__ was read as True, one named float32 through its __float__,
    # and one named datetime64 through its astype and dtype, whose results
    # could raise RuntimeError("std::bad_cast"). A type whose metaclass gives
    # a __name__ that is not a str raised that RuntimeError too.
    pipeline: Pipeline = Pipeline([("s", StandardScaler(columns=["price"]))]).fit([source])
    lookalikes: list[tuple[Any, str]] = [
        (type("bool_", (), {})(), "bool_"),
        (bool_(), "bool_"),
        (type("bool", (), {"__bool__": lambda _self: False})(), "bool"),
        (type("bool_", (), {"__module__": "numpy"})(), "bool_"),
        (type("numpy.bool", (), {})(), "numpy.bool"),  # the C name of a class made in Python is its __name__
        (type("datetime64", (), {"astype": _astype, "dtype": np.dtype("M8[s]")})(), "datetime64"),
        (type("float32", (), {"__float__": lambda _self: 1.25})(), "float32"),
        (type("float16", (), {"__float__": lambda _self: 1.25})(), "float16"),
        (type("longdouble", (), {"__float__": lambda _self: 1.25})(), "longdouble"),
        (type("float128", (), {"__float__": lambda _self: 1.25})(), "float128"),
        (type("float32", (np.float32,), {})(1.25), "float32"),
        (_NameIsNotText("Odd", (), {})(), "Odd"),
    ]
    unsupported: str = "record value for 'v' must be str, int, float, bool, datetime, or None"
    for value, name in lookalikes:
        with pytest.raises(EngineError) as refused:
            pipeline.transform_dict({"price": 2.0, "v": value})
        assert str(refused.value) == f"{unsupported} (got {name})"
    # numpy's own values, read as numpy reads them.
    owns: list[tuple[Any, Any]] = [
        (np.bool_(False), False),
        (np.bool_(True), True),
        (np.float32(1.25), 1.25),
        (np.float16(2.5), 2.5),
        (np.longdouble(0.5), 0.5),
    ]
    for value, expected in owns:
        out: Any = pipeline.transform_dict({"price": 2.0, "v": value})["v"]
        assert (type(out), out) == (type(expected), expected)
    assert pipeline.transform_dict({"price": 2.0, "v": np.datetime64(5, "s")})["v"] == 5_000_000


# A process that has not imported numpy sends values whose types carry
# numpy's names, and prints whether numpy is imported after.
NO_NUMPY_CHILD: str = """
import sys

from fast_feature_transform import EngineError, Pipeline, StandardScaler


class Count:
    def item(self):
        return 5


class datetime64:
    dtype = "M8[s]"

    def astype(self, kind):
        return Count()


pipeline = Pipeline([("s", StandardScaler(columns=["price"]))]).fit([sys.argv[1]], threads=1)
for value in (datetime64(), type("bool_", (), {})(), type("float32", (), {"__float__": lambda self: 1.0})()):
    try:
        print(pipeline.transform_dict({"price": 2.0, "v": value})["v"])
    except EngineError as error:
        print(error)
print("numpy" in sys.modules)
"""


def test_transform_dict_does_not_import_numpy(source: str, tmp_path: Path) -> None:
    # REGRESSION: numpy is not a dependency, and the record path does not
    # import it. A value of a type named datetime64 that had astype and dtype
    # used to import numpy, to call its datetime_data.
    command: list[str] = [sys.executable, "-c", NO_NUMPY_CHILD, source]
    child = subprocess.run(command, capture_output=True, text=True, cwd=tmp_path, timeout=60, check=False)  # nosec B603
    unsupported: str = "record value for 'v' must be str, int, float, bool, datetime, or None"
    expected: list[str] = [f"{unsupported} (got datetime64)", f"{unsupported} (got bool_)"]
    expected += [f"{unsupported} (got float32)", "False"]
    assert (child.returncode, child.stdout.splitlines()) == (0, expected), child.stderr


class _NoIndex:
    """An integer-like value whose __index__ fails with an error other than TypeError."""

    def __index__(self) -> int:
        """Refuse an integer.

        Raises:
            ValueError: Always.

        """
        raise ValueError("no integer value")


def test_transform_dict_raises_the_error_a_value_raises_while_it_converts(source: str) -> None:
    # REGRESSION: an error from a value's own __index__ reaches the caller.
    # Any error from __index__ used to become the EngineError for an
    # unsupported type.
    pipeline: Pipeline = Pipeline([("s", Select(columns=["device"]))]).fit([source])
    with pytest.raises(ValueError, match="no integer value"):
        pipeline.transform_dict({"device": "ios", "n": _NoIndex()})
    # A value whose reading runs Python code is read after every other field,
    # so the refusal of a later field comes first.
    with pytest.raises(EngineError, match="'big' does not fit in a 64-bit integer"):
        pipeline.transform_dict({"device": "ios", "n": _NoIndex(), "big": INT64_MAX + 1})


def test_transform_dict_refuses_the_first_value_it_reads_of_two_it_refuses(source: str) -> None:
    # The walk over a record reads, in record order, every value that runs no
    # Python code to read, numpy values among them. It then reads the other
    # values in record order. Of two values that are refused, the one read
    # first is the one refused.
    pipeline: Pipeline = Pipeline([("s", Select(columns=["device"]))]).fit([source])
    with pytest.raises(EngineError, match="^record value for 'a' does not fit in a 64-bit integer$"):
        pipeline.transform_dict({"device": "ios", "a": np.uint64(2**64 - 1), "b": INT64_MAX + 1})
    with pytest.raises(EngineError, match="^record value for 'b' does not fit in a 64-bit integer$"):
        pipeline.transform_dict({"device": "ios", "b": INT64_MAX + 1, "a": np.uint64(2**64 - 1)})
    # Neither _NoIndex nor Decimal is read in the walk.
    with pytest.raises(ValueError, match="^no integer value$"):
        pipeline.transform_dict({"device": "ios", "a": _NoIndex(), "b": Decimal(2)})
    unsupported: str = r"must be str, int, float, bool, datetime, or None \(got Decimal\)"
    with pytest.raises(EngineError, match=f"^record value for 'b' {unsupported}$"):
        pipeline.transform_dict({"device": "ios", "b": Decimal(2), "a": _NoIndex()})


class _LookupRaises:
    """A value whose lookup of an attribute it lacks raises KeyError, as a __getattr__ that reads a dict does."""

    def __getattr__(self, name: str) -> Any:
        """Look the name up in an empty dict.

        Args:
            name: The attribute's name.

        Raises:
            KeyError: Always.

        """
        raise KeyError(name)


def test_transform_dict_raises_the_error_a_value_raises_while_an_attribute_is_looked_up(source: str) -> None:
    # REGRESSION: a value's lookup of an attribute that raises anything but
    # AttributeError raises that error, as Python's hasattr does. The record
    # path used to hand the error to sys.unraisablehook and go on as if the
    # value lacked the attribute, here __index__, so the value got the
    # EngineError for an unsupported type.
    pipeline: Pipeline = Pipeline([("s", Select(columns=["device"]))]).fit([source])
    with pytest.raises(KeyError, match="__index__"):
        pipeline.transform_dict({"device": "ios", "n": _LookupRaises()})


def test_transform_dict_accepts_numpy_scalars(source: str) -> None:
    pipeline: Pipeline = Pipeline([("s", Select(columns=["device", "n", "x", "flag"]))]).fit(
        [_with_extra_columns(source)]
    )
    out = pipeline.transform_dict(
        {"device": np.str_("ios"), "n": np.int32(7), "x": np.float32(1.5), "flag": np.bool_(True)}
    )
    assert out == {"device": "ios", "n": 7, "x": 1.5, "flag": True}
    assert isinstance(out["n"], int)
    assert isinstance(out["flag"], bool)
    with pytest.raises(EngineError, match="'n' does not fit in a 64-bit integer"):
        pipeline.transform_dict({"device": "ios", "n": np.uint64(2**64 - 1), "x": 1.0, "flag": False})


def _with_extra_columns(source: str) -> str:
    """Rewrite the fixture with numeric and boolean columns added.

    Args:
        source: Path of the fixture Parquet file.

    Returns:
        Path of the rewritten file.

    """
    table: pa.Table = pq.read_table(source)
    table = table.append_column("n", pa.array([1, 2, 3])).append_column("x", pa.array([0.5, 1.5, 2.5]))
    table = table.append_column("flag", pa.array([True, False, True]))
    pq.write_table(table, source)
    return source
