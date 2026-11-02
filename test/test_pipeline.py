"""Tests for Pipeline composition, artifact save/load, and pickle delegation."""

import importlib.metadata
import json
import pickle
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fast_feature_transform import _core
from fast_feature_transform.pipeline import Pipeline
from fast_feature_transform.transformers import LabelEncoder, Select

MOB_INDEX: int = 2
NUM_ROWS: int = 8


@pytest.fixture
def train_files(tmp_path: Path) -> list[str]:
    """Write one Parquet file with several row groups.

    Args:
        tmp_path: pytest-provided temporary directory.

    Returns:
        Paths of the written Parquet files.

    """
    table: pa.Table = pa.table(
        {
            "device": ["mob", "desk", "mob", "tab", "mob", "desk", None, "mob"],
            "source": ["a", "b", "a", "a", "c", "b", "a", "a"],
            "price": [12.5, 8.0, 3.0, 4.5, 9.9, 1.0, 2.0, 5.0],
        }
    )
    path: Path = tmp_path / "train.parquet"
    pq.write_table(table, path, row_group_size=2)
    return [str(path)]


def _build_pipeline() -> Pipeline:
    """Build the pipeline these tests share: select then encode, inside a nested pipeline.

    Returns:
        An unfitted two-level pipeline.

    """
    return Pipeline(
        [
            (
                "features",
                Pipeline(
                    [
                        ("select", Select(columns=["device", "source"])),
                        ("encode", LabelEncoder(columns=["device", "source"])),
                    ]
                ),
            ),
        ]
    )


def test_transform_dict_and_parquet_match_transform(train_files: list[str], tmp_path: Path) -> None:
    pipeline: Pipeline = _build_pipeline().fit(train_files)
    record = {"device": "mob", "source": "a", "price": 1.0}
    assert pipeline.transform_dict(record) == pipeline.transform(record)
    via_explicit = pipeline.transform_parquet(train_files, output_dir=str(tmp_path / "a"))
    via_dispatch = pipeline.transform(train_files, output_dir=str(tmp_path / "b"))
    assert [p.name for p in via_explicit] == [p.name for p in via_dispatch]


def test_pipeline_fit_transform_dict(train_files: list[str]) -> None:
    pipeline: Pipeline = _build_pipeline().fit(train_files, threads=4)
    row = pipeline.transform({"device": "mob", "source": "a", "price": 1.0})
    assert row == {"device": MOB_INDEX, "source": 1}


def test_pipeline_transform_batch_writes_ordered_parts(train_files: list[str], tmp_path: Path) -> None:
    pipeline: Pipeline = _build_pipeline().fit(train_files)
    parts: list[Path] = pipeline.transform(train_files, output_dir=str(tmp_path / "out"))
    assert [part.name for part in parts] == sorted(part.name for part in parts)
    table: pa.Table = pa.concat_tables([pq.read_table(part) for part in parts])
    assert table.num_rows == NUM_ROWS
    assert table.column_names == ["device", "source"]


def test_pipeline_step_lookup_reaches_fitted_state(train_files: list[str]) -> None:
    pipeline: Pipeline = _build_pipeline().fit(train_files)
    features = pipeline["features"]
    assert isinstance(features, Pipeline)
    encoder = features["encode"]
    assert isinstance(encoder, LabelEncoder)
    assert encoder.label_maps_["device"]["mob"] == MOB_INDEX


def test_pipeline_save_load_round_trip(train_files: list[str], tmp_path: Path) -> None:
    pipeline: Pipeline = _build_pipeline().fit(train_files)
    artifact: Path = tmp_path / "artifact"
    pipeline.save(artifact)
    assert (artifact / "manifest.json").exists()
    restored: Pipeline = Pipeline.load(artifact)
    original = pipeline.transform({"device": "mob", "source": "a", "price": 1.0})
    assert restored.transform({"device": "mob", "source": "a", "price": 1.0}) == original


def test_pipeline_load_rebuilds_nested_structure(train_files: list[str], tmp_path: Path) -> None:
    pipeline: Pipeline = _build_pipeline().fit(train_files)
    artifact: Path = tmp_path / "artifact"
    pipeline.save(artifact)
    restored: Pipeline = Pipeline.load(artifact)
    features = restored["features"]
    assert isinstance(features, Pipeline)
    encoder = features["encode"]
    assert isinstance(encoder, LabelEncoder)
    assert encoder.label_maps_["device"]["mob"] == MOB_INDEX
    assert encoder.min_count == 1


def test_pipeline_pickle_round_trip(train_files: list[str]) -> None:
    pipeline: Pipeline = _build_pipeline().fit(train_files)
    restored: Pipeline = pickle.loads(pickle.dumps(pipeline))
    assert restored.transform({"device": "mob", "source": "a", "price": 1.0})["device"] == MOB_INDEX
    features = restored["features"]
    assert isinstance(features, Pipeline)
    encoder = features["encode"]
    assert isinstance(encoder, LabelEncoder)
    assert encoder.label_maps_["device"]["mob"] == MOB_INDEX


def test_nested_pipeline_pickled_alone_keeps_stage_bindings(train_files: list[str]) -> None:
    pipeline: Pipeline = _build_pipeline().fit(train_files)
    features = pipeline["features"]
    assert isinstance(features, Pipeline)
    restored: Pipeline = pickle.loads(pickle.dumps(features))
    encoder = restored["encode"]
    assert isinstance(encoder, LabelEncoder)
    assert encoder.label_maps_["device"]["mob"] == MOB_INDEX


def test_manifest_names_kernels_and_stages(train_files: list[str]) -> None:
    pipeline: Pipeline = _build_pipeline().fit(train_files)
    manifest = json.loads(pipeline._require_handle().manifest_json())
    assert manifest["schema_version"] == "0.3.0"
    # The fitting release, exactly as the package reports it (dev suffix included).
    assert (
        manifest["library_version"] == importlib.metadata.version("fast-feature-transform") == _core.library_version()
    )
    assert [op["stage"] for op in manifest["ops"]] == ["features.select", "features.encode"]
    assert [op["kernel"] for op in manifest["ops"]] == ["select", "label_encode"]


def test_unknown_kernel_fails_loudly() -> None:
    manifest: str = json.dumps(
        {"schema_version": "0.3.0", "ops": [{"stage": "s", "kernel": "flux_capacitor", "params": {}, "state": {}}]}
    )
    with pytest.raises(Exception, match="unknown kernel"):
        _core.FittedPipeline.load_json(manifest)


def test_unfitted_pipeline_raises() -> None:
    with pytest.raises(RuntimeError, match="not fitted"):
        _build_pipeline().transform({"device": "mob"})


def test_fit_rejects_a_bare_path(train_files: list[str]) -> None:
    with pytest.raises(TypeError, match="wrap it in a list"):
        _build_pipeline().fit(train_files[0])


def test_duplicate_step_names_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate step name"):
        Pipeline([("enc", LabelEncoder(columns=["a"])), ("enc", LabelEncoder(columns=["b"]))])


def test_dotted_step_names_rejected() -> None:
    with pytest.raises(ValueError, match="must not contain"):
        Pipeline([("a.b", Select(columns=["a"]))])


def test_non_parquet_input_rejected_by_magic_bytes(tmp_path: Path) -> None:
    fake: Path = tmp_path / "fake.parquet"
    fake.write_text("not parquet at all")
    with pytest.raises(Exception, match="not a Parquet file"):
        _build_pipeline().fit([str(fake)])


def test_fit_transform_writes_final_output(train_files: list[str], tmp_path: Path) -> None:
    parts: list[Path] = _build_pipeline().fit_transform(train_files, output_dir=str(tmp_path / "out"))
    table: pa.Table = pa.concat_tables([pq.read_table(part) for part in parts])
    assert table.num_rows == NUM_ROWS


# REGRESSION: the steps setter replaced the Python step list without touching
# the fitted C++ handle, so a fitted pipeline whose steps had been reassigned
# reported the new steps from repr, __getitem__ and steps while transform_dict
# and transform_parquet went on running the plan that was compiled. Mutating
# the returned list in place lied the same way. A fitted pipeline is immutable:
# the setter refuses, and the getter hands out a copy.
def test_a_fitted_pipeline_refuses_to_have_its_steps_replaced(train_files: list[str], tmp_path: Path) -> None:
    pipeline: Pipeline = _build_pipeline().fit(train_files)
    before: dict[str, object] = pipeline.transform_dict({"device": "mob", "source": "a", "price": 1.0})

    with pytest.raises(RuntimeError, match="already fitted"):
        pipeline.steps = [("select", Select(columns=["source"]))]
    # An unpickled pipeline is fitted before its manifest is loaded, and
    # refusing must not load it just to refuse.
    restored: Pipeline = pickle.loads(pickle.dumps(pipeline))
    assert restored._manifest is not None
    with pytest.raises(RuntimeError, match="already fitted"):
        restored.steps = [("select", Select(columns=["source"]))]
    assert restored._manifest is not None
    # A nested step is fitted too, so the root cannot be hollowed out through it.
    inner = pipeline["features"]
    assert isinstance(inner, Pipeline)
    with pytest.raises(RuntimeError, match="already fitted"):
        inner.steps = [("select", Select(columns=["source"]))]

    # The list handed out is a copy, so appending to it changes nothing at all
    # rather than changing what the pipeline claims to be.
    pipeline.steps.append(("extra", Select(columns=["device"])))
    assert [name for name, _ in pipeline.steps] == ["features"]
    assert repr(pipeline) == repr(_build_pipeline().fit(train_files))
    assert pipeline.transform_dict({"device": "mob", "source": "a", "price": 1.0}) == before
    parts: list[Path] = pipeline.transform_parquet(train_files, output_dir=tmp_path / "out")
    assert pa.concat_tables([pq.read_table(part) for part in parts]).num_rows == NUM_ROWS


def test_an_unfitted_pipeline_still_takes_new_steps(train_files: list[str]) -> None:
    pipeline: Pipeline = Pipeline([("select", Select(columns=["device"]))])
    pipeline.steps = [("select", Select(columns=["source"]))]
    assert repr(pipeline) == "Pipeline(steps=[('select', Select(columns=['source']))])"
    assert pipeline.fit(train_files).transform_dict({"device": "mob", "source": "a"}) == {"source": "a"}
    # The setter validates what it is given, which .append never did.
    with pytest.raises(ValueError, match="duplicate step name"):
        Pipeline([]).steps = [("s", Select(columns=["a"])), ("s", Select(columns=["b"]))]
