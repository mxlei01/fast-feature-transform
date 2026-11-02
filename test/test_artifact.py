"""Tests for artifact integrity: provenance, validation on load, and how engine errors surface."""

import importlib.metadata
import json
import pickle
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fast_feature_transform import EngineError, _core
from fast_feature_transform.pipeline import Pipeline
from fast_feature_transform.transformers import LabelEncoder, Select

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


def test_transform_dict_rejects_unconvertible_values_by_name(source: str) -> None:
    pipeline: Pipeline = Pipeline([("e", LabelEncoder(columns=["device"]))]).fit([source])
    with pytest.raises(EngineError, match="'big' does not fit in a 64-bit integer"):
        pipeline.transform_dict({"device": "ios", "big": INT64_MAX + 1})
    non_str_key: dict[Any, Any] = {"device": "ios", 1: 1}
    with pytest.raises(EngineError, match="record keys must be str"):
        pipeline.transform_dict(non_str_key)
    with pytest.raises(EngineError, match="'device' is not encodable as UTF-8"):
        pipeline.transform_dict({"device": "\ud800"})
    with pytest.raises(EngineError, match="'blob' must be str, int, float, bool, datetime, or None \\(got bytes\\)"):
        pipeline.transform_dict({"device": "ios", "blob": b"x"})


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
