"""Tests for exact fits: one result for any row-group split, bounded fit batches, and temporary-file cleanup."""

import os
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fast_feature_transform import (
    EngineError,
    GroupAgg,
    KBinsDiscretizer,
    Pipeline,
    PowerTransformer,
    QuantileEncoder,
    QuantileTransformer,
    RobustScaler,
    SplineTransformer,
    StandardScaler,
    SummaryEncoder,
    Transformer,
)


def _cases() -> dict[str, Callable[[], Transformer]]:
    return {
        "robust": lambda: RobustScaler(columns=["x", "positive"]),
        "quantile": lambda: QuantileTransformer(columns=["x"], n_quantiles=37),
        "bins_quantile": lambda: KBinsDiscretizer(columns=["x"], n_bins=7, strategy="quantile"),
        "bins_kmeans": lambda: KBinsDiscretizer(columns=["x"], n_bins=7, strategy="kmeans"),
        "spline": lambda: SplineTransformer(columns=["x"], n_knots=7, knots="quantile"),
        "power_yeo": lambda: PowerTransformer(columns=["x"]),
        "power_box": lambda: PowerTransformer(columns=["positive"], method="box-cox"),
        "group_median": lambda: GroupAgg(key="key", aggs={"x": ["median", "mean", "count"], "i": ["median", "sum"]}),
        "quantile_encoder": lambda: QuantileEncoder(columns=["key"], target="y"),
        "summary_encoder": lambda: SummaryEncoder(columns=["key"], target="y", quantiles=[0.1, 0.5, 0.9]),
        "standard_control": lambda: StandardScaler(columns=["x", "positive"]),
    }


def _fixture(rows: int = 2049) -> pa.Table:
    index = np.arange(rows, dtype=np.int64)
    x = ((index * 37) % 1021 - 510).astype(np.float64) / 32
    x[0] = -0.0
    keys = [None if i % 29 == 0 else "common" if i % 5 else f"key-{i % 19}" for i in range(rows)]
    return pa.table(
        {
            "key": keys,
            "x": x,
            "positive": np.abs(x) + 0.125,
            "i": index % 101 - 50,
            "y": pa.array(((index * 17) % 211 - 105).astype(np.float64) / 8, mask=index % 31 == 0),
        }
    )


def _assert_bits(left: pa.Table, right: pa.Table) -> None:
    assert left.schema == right.schema
    assert left.equals(right)
    for before, after in zip(left.columns, right.columns, strict=True):
        if pa.types.is_floating(before.type):
            valid = before.is_valid().to_numpy()
            a = before.to_numpy()[valid]
            b = after.to_numpy()[valid]
            assert a.tobytes() == b.tobytes()


@pytest.mark.parametrize("case", list(_cases()))
@pytest.mark.parametrize("threads", [1, 4])
def test_fit_matches_across_input_partitions(tmp_path: Path, case: str, threads: int) -> None:
    table = _fixture()
    path = tmp_path / "input.parquet"
    pq.write_table(table, path, row_group_size=311)
    factory = _cases()[case]
    reference_path = tmp_path / "single-group.parquet"
    pq.write_table(table, reference_path, row_group_size=table.num_rows)
    reference = factory().fit([reference_path], threads=1)
    spill_parent = tmp_path / "scratch"
    spill_parent.mkdir()
    actual = factory().fit([path], threads=threads, temp_dir=spill_parent)
    assert actual._state() == reference._state()
    probe = tmp_path / "probe.parquet"
    pq.write_table(table.slice(0, 53), probe)
    _assert_bits(
        pq.read_table(reference.transform_parquet([probe], output_dir=tmp_path / "reference-out")),
        pq.read_table(actual.transform_parquet([probe], output_dir=tmp_path / "actual-out")),
    )
    assert list(spill_parent.iterdir()) == []


def test_single_large_row_group_uses_fit_batches_and_cleans_failed_refit(tmp_path: Path) -> None:
    table = _fixture(131073)
    path = tmp_path / "large-group.parquet"
    pq.write_table(table, path, row_group_size=table.num_rows, use_dictionary=False)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    pipeline = Pipeline([("scale", RobustScaler(columns=["x"]))]).fit([path], threads=1, temp_dir=scratch)
    before = pipeline.transform_dict({"x": 2.5})
    bad_values = table["x"].to_numpy().copy()
    bad_values[-1] = float("nan")
    bad = table.set_column(table.schema.get_field_index("x"), "x", pa.array(bad_values))
    pq.write_table(bad, path)
    with pytest.raises(EngineError, match="not finite"):
        pipeline.fit([path], threads=1, temp_dir=scratch)
    assert pipeline.transform_dict({"x": 2.5}) == before
    assert list(scratch.iterdir()) == []


@pytest.mark.parametrize("fit_transform", [False, True])
def test_fit_temp_dir_validation(fit_transform: bool) -> None:
    pipeline = Pipeline([("scale", RobustScaler(columns=["x"]))])
    operation = pipeline.fit_transform if fit_transform else pipeline.fit
    with pytest.raises(TypeError, match="temp_dir"):
        operation(["never-opened.parquet"], temp_dir=b"bytes")  # ty: ignore[invalid-argument-type]


def test_fit_transform_with_temp_directory_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "input.parquet"
    pq.write_table(pa.table({"x": [1.0, 2.0, 4.0, 8.0]}), path)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    pipeline = Pipeline([("scale", RobustScaler(columns=["x"]))])
    parts = pipeline.fit_transform([path], threads=1, temp_dir=scratch, output_dir=tmp_path / "output")
    assert pq.read_table(parts).num_rows == 4
    assert list(scratch.iterdir()) == []
    artifact = tmp_path / "model"
    pipeline.save(artifact)
    assert str(scratch) not in (artifact / "manifest.json").read_text()
    restored = Pipeline.load(artifact)
    assert restored.transform_dict({"x": 3.0}) == pipeline.transform_dict({"x": 3.0})


@pytest.mark.scale
def test_automatic_spill_and_failure_cleanup_preserve_fitted_state(tmp_path: Path) -> None:
    path = tmp_path / "large.parquet"
    chunk = pa.table({"x": np.tile([1.0, 3.0], 32768)})
    with pq.ParquetWriter(path, chunk.schema) as writer:
        for _ in range(80):
            writer.write_table(chunk)
    reference_path = tmp_path / "reference.parquet"
    pq.write_table(chunk.slice(0, 16), reference_path)
    scaler = RobustScaler(columns=["x"]).fit([reference_path], threads=1)
    expected = scaler._state()

    # A directory that exists, so the fit starts, but that the spill cannot
    # write into. (A path that is not a directory is refused before the fit.)
    # Root writes into a read-only directory anyway, so it skips this part.
    if os.geteuid() != 0:
        unavailable = tmp_path / "read-only"
        unavailable.mkdir(mode=0o555)
        with pytest.raises(EngineError, match="fit spill"):
            scaler.fit([path], threads=1, temp_dir=unavailable)
        assert scaler._state() == expected
        assert list(unavailable.iterdir()) == []

    scratch = tmp_path / "scratch"
    scratch.mkdir()
    scaler.fit([path], threads=1, temp_dir=scratch)
    assert scaler._state() == expected
    assert list(scratch.iterdir()) == []

    invalid = tmp_path / "invalid.parquet"
    pq.write_table(pa.table({"x": [float("nan")]}), invalid)
    with pytest.raises(EngineError, match="not finite"):
        scaler.fit([path, invalid], threads=1, temp_dir=scratch)
    assert scaler._state() == expected
    assert list(scratch.iterdir()) == []
