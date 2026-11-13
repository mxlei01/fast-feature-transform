"""Tests for artifact integrity: provenance, validation on load, and how engine errors surface."""

import datetime
import importlib.metadata
import inspect
import itertools
import json
import math
import pickle
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import fast_feature_transform
from fast_feature_transform import (
    BackwardDifferenceEncoder,
    BaseNEncoder,
    BinaryEncoder,
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


def test_transform_dict_refuses_a_record_whose_keys_repeat_a_text(source: str) -> None:
    # REGRESSION: a record whose keys hold one text twice is an error that
    # names the text. A step used to read one of the two fields, which one
    # depending on the order of the keys, the width of the record and the
    # step, and the output dict kept the value of the later field.
    pipeline: Pipeline = Pipeline([("s", Select(columns=["device"]))]).fit([source])
    # Each record holds "device" and then one to three keys, of every text and
    # kind in every order; one text is not ASCII. A str and a numpy.str_ of
    # one text are one key of a dict; each of the other kinds is a key of its
    # own beside a str.
    kinds: list[type[str]] = [str, np.str_, _OwnHash, _OwnEquality]
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


def _move_device(record: dict[str, Any], grow: int) -> None:
    """Take the key "device" out of a record and store it again, last.

    Args:
        record: The record.
        grow: How many keys to add in between; enough of them rebuild the
            dict's table.

    """
    device: Any = record.pop("device")
    for index in range(grow):
        record[f"g{index}"] = index
    record["device"] = device


class _MovesOnIndex:
    """An integer-like value whose __index__ moves the key "device" of its record."""

    def __init__(self, record: dict[str, Any], grow: int) -> None:
        """Hold the record to change.

        Args:
            record: The record this value is in.
            grow: Passed to _move_device.

        """
        self.record: dict[str, Any] = record
        self.grow: int = grow

    def __index__(self) -> int:
        """Move the key, then answer.

        Returns:
            1.

        """
        _move_device(self.record, self.grow)
        return 1


class _MovesOnAsm8(datetime.datetime):
    """A datetime that moves the key "device" of the record in `record` when asked for a pandas Timestamp's asm8."""

    record: ClassVar[dict[str, Any]] = {}

    def __getattr__(self, name: str) -> Any:
        """Move the key, then refuse the attribute.

        Args:
            name: The attribute asked for.

        Raises:
            AttributeError: Always.

        """
        if name == "asm8":
            _move_device(type(self).record, 0)
        raise AttributeError(name)


@pytest.mark.parametrize("mover", ["index", "index_grown", "asm8"])
def test_transform_dict_refuses_a_record_whose_value_moves_a_key_while_it_is_read(source: str, mover: str) -> None:
    # REGRESSION: a value whose own code, run while the record is read, takes
    # a key that was read out of the record and stores it again is an error
    # that names the key. The read used to reach the key a second time, at
    # its new place, and gave the steps a record that held it twice.
    pipeline: Pipeline = Pipeline([("s", Select(columns=["device"]))]).fit([source])
    record: dict[str, Any] = {"device": "ios"}
    if mover == "asm8":
        _MovesOnAsm8.record = record
        record["t"] = _MovesOnAsm8(2021, 6, 1)
    else:
        record["n"] = _MovesOnIndex(record, 40 if mover == "index_grown" else 0)
    with pytest.raises(EngineError) as refused:
        pipeline.transform_dict(record)
    assert str(refused.value) == "record holds the key 'device' twice"
    # The key moved: the record holds it once, last.
    assert list(record)[-1] == "device"


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
    """A value that carries the name of numpy's bool, which the record path tests by its truth."""

    def __bool__(self) -> bool:
        """Refuse a truth value.

        Raises:
            ValueError: Always.

        """
        raise ValueError("no truth value")


class _NoIndex:
    """An integer-like value whose __index__ fails with an error other than TypeError."""

    def __index__(self) -> int:
        """Refuse an integer.

        Raises:
            ValueError: Always.

        """
        raise ValueError("no integer value")


def test_transform_dict_raises_the_error_a_value_raises_while_it_converts(source: str) -> None:
    # REGRESSION: an error from a value's own __bool__ or __index__ reaches
    # the caller. A raising __bool__ used to leave its error set, which
    # surfaced as a SystemError, and any error from __index__ used to become
    # the EngineError for an unsupported type.
    pipeline: Pipeline = Pipeline([("s", Select(columns=["device"]))]).fit([source])
    with pytest.raises(ValueError, match="no truth value"):
        pipeline.transform_dict({"device": "ios", "flag": bool_()})
    with pytest.raises(ValueError, match="no integer value"):
        pipeline.transform_dict({"device": "ios", "n": _NoIndex()})


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
