"""The measurement rules of the two benchmark scripts and the module they share.

``benchmark/transformers.py`` needs category_encoders and pandas, which are
deliberately not dependencies of this project, so the module is loaded here
with those two stubbed out. Nothing tested below touches either of them. The
tests cover how the scripts measure: the Parquet settings they charge upstream
for, how a one-column file is read, how repeated runs are summarized, and that
the case order and the row-group layout are not fixed.

Both scripts measure through ``benchmark/measure.py``, and the
``test_both_scripts_*`` tests keep them there: the two once reported different
methodologies.
"""

import importlib.util
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fast_feature_transform import Select

BENCHMARK_DIR: Path = Path(__file__).resolve().parent.parent / "benchmark"


def _load_benchmark(script: str) -> ModuleType:
    """Import one benchmark script with its two optional dependencies stubbed.

    Args:
        script: File name of the script inside ``benchmark/``.

    Returns:
        The loaded module.

    """
    spec = importlib.util.spec_from_file_location(f"benchmark_{Path(script).stem}", BENCHMARK_DIR / script)
    assert spec is not None
    assert spec.loader is not None
    module: ModuleType = importlib.util.module_from_spec(spec)
    stubs: dict[str, ModuleType] = {name: ModuleType(name) for name in ("pandas", "category_encoders")}
    # The scripts import ``measure`` the way they do when run directly, with
    # their own directory on sys.path.
    if str(BENCHMARK_DIR) not in sys.path:
        sys.path.insert(0, str(BENCHMARK_DIR))
    # Only the two stubbed names are put back afterwards: restoring the whole
    # of sys.modules would drop everything the import pulled in, and a C
    # extension cannot be loaded twice in one process.
    saved: dict[str, ModuleType | None] = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        spec.loader.exec_module(module)
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
    return module


BENCH: ModuleType = _load_benchmark("transformers.py")
LABEL: ModuleType = _load_benchmark("label_encoder.py")
MEASURE: ModuleType = importlib.import_module("measure")


def _column_facts(path: Path) -> list[tuple[Any, ...]]:
    """Describe how every column of a Parquet file was encoded.

    Args:
        path: The Parquet file to describe.

    Returns:
        One tuple per column per row group: name, compression, encodings, and
        whether a dictionary page was written.

    """
    metadata = pq.ParquetFile(path).metadata
    facts: list[tuple[Any, ...]] = []
    for group in range(metadata.num_row_groups):
        for index in range(metadata.num_columns):
            column = metadata.row_group(group).column(index)
            facts.append(
                (
                    column.path_in_schema,
                    column.compression,
                    tuple(sorted(column.encodings)),
                    column.dictionary_page_offset is not None,
                )
            )
    return facts


def _row_groups(path: Path) -> list[int]:
    """List the row counts of a Parquet file's row groups, in order.

    Args:
        path: The Parquet file to describe.

    Returns:
        One row count per row group.

    """
    metadata = pq.ParquetFile(path).metadata
    return [metadata.row_group(group).num_rows for group in range(metadata.num_row_groups)]


# Each part holds more rows than one batch, so the sink writes several row
# groups per part: 65,536-row batches for the narrow output and 21,845 for
# the wide one, whose batches shrink to fit their wider rows. A part no
# larger than a batch is one row group however the write-back lays it out.
@pytest.mark.parametrize(("columns", "rows_per_part"), [(2, 150_000), (32, 50_000)])
def test_write_back_matches_our_sinks_parquet_settings(tmp_path, columns, rows_per_part):
    names: list[str] = [f"c{i}" for i in range(columns)]
    values: np.ndarray = np.random.default_rng(0).normal(100.0, 25.0, (2 * rows_per_part, columns))
    source: Path = tmp_path / "in.parquet"
    pq.write_table(pa.table({name: values[:, i] for i, name in enumerate(names)}), source, row_group_size=rows_per_part)

    fitted = Select(columns=names).fit([str(source)])
    ours: list[Path] = sorted(
        Path(part) for part in fitted.transform_parquet([str(source)], output_dir=tmp_path / "ours")
    )
    MEASURE.write_back(values, tmp_path / "theirs", MEASURE.sink_layout(ours))
    theirs: list[Path] = sorted((tmp_path / "theirs").glob("*.parquet"))

    assert [path.name for path in ours] == [path.name for path in theirs]
    for ours_part, theirs_part in zip(ours, theirs, strict=True):
        ours_meta = pq.ParquetFile(ours_part).metadata
        theirs_meta = pq.ParquetFile(theirs_part).metadata
        assert len(_row_groups(ours_part)) > 1
        assert _row_groups(theirs_part) == _row_groups(ours_part)
        assert _column_facts(theirs_part) == _column_facts(ours_part)
        assert theirs_meta.format_version == ours_meta.format_version
        assert theirs_meta.metadata == ours_meta.metadata
        assert abs(theirs_part.stat().st_size - ours_part.stat().st_size) < 0.01 * ours_part.stat().st_size


def test_write_parts_refuses_a_layout_that_does_not_cover_the_table(tmp_path):
    with pytest.raises(ValueError, match="the layout covers 4 rows, the table has 6"):
        MEASURE.write_parts(pa.table({"x": np.arange(6)}), tmp_path, [(2, 2), (2, 2)])


def test_dictionary_columns_follows_the_sinks_float_rule():
    schema = pa.schema(
        [("f64", pa.float64()), ("f32", pa.float32()), ("tag", pa.string()), ("n", pa.int64()), ("flag", pa.bool_())]
    )
    assert MEASURE.dictionary_columns(schema) == ["tag", "n", "flag"]


def test_read_matrix_does_not_copy_a_one_column_file(tmp_path):
    path: Path = tmp_path / "price.parquet"
    values: np.ndarray = np.random.default_rng(1).normal(100.0, 25.0, 5_000)
    pq.write_table(pa.table({"price": values}), path)
    matrix: np.ndarray = BENCH._read_matrix(path)
    assert matrix.shape == (5_000, 1)
    assert not matrix.flags.owndata
    assert np.array_equal(matrix[:, 0], values)


def test_read_matrix_stacks_several_columns(tmp_path):
    path: Path = tmp_path / "two.parquet"
    values: np.ndarray = np.random.default_rng(2).normal(0.0, 1.0, (5_000, 2))
    pq.write_table(pa.table({"a": values[:, 0], "b": values[:, 1]}), path)
    matrix: np.ndarray = BENCH._read_matrix(path)
    assert matrix.shape == (5_000, 2)
    assert np.array_equal(matrix, values)


def test_subsampled_quantile_fit_is_timed_with_the_read_inside(tmp_path, monkeypatch):
    path: Path = tmp_path / "price.parquet"
    pq.write_table(pa.table({"price": np.random.default_rng(3).normal(100.0, 25.0, 20_000)}), path)
    read: Any = BENCH._read_matrix
    calls: list[Path] = []

    def slow_read(target: Path) -> np.ndarray:
        calls.append(target)
        time.sleep(0.02)
        return read(target)

    monkeypatch.setattr(BENCH, "_read_matrix", slow_read)
    monkeypatch.setattr(MEASURE, "REPEATS", 3)
    stat = BENCH._measure_subsampled(path)
    assert calls == [path, path, path]
    assert stat.median >= 0.02


def test_repeats_are_enough_to_take_a_median_of():
    assert MEASURE.REPEATS >= 5


def test_repeat_reports_the_median_not_the_fastest_run(monkeypatch):
    monkeypatch.setattr(MEASURE, "REPEATS", 5)
    delays: list[float] = [0.0, 0.02, 0.02, 0.02, 0.02]
    pending: Any = iter(delays)
    stat = MEASURE.repeat(lambda: time.sleep(next(pending)))
    assert len(stat.samples) == len(delays)
    assert stat.low < 0.01
    assert stat.median >= 0.015
    assert stat.high >= stat.median


def test_summarise_reports_the_middle_and_the_spread():
    stat = MEASURE.summarise([3.0, 1.0, 100.0, 2.0, 4.0])
    assert stat.median == 3.0
    assert stat.q1 == 2.0
    assert stat.q3 == 4.0
    assert stat.iqr == 2.0
    assert stat.low == 1.0
    assert stat.high == 100.0


def test_run_order_is_a_shuffle_of_every_case():
    declared: list[str] = [case.name for case in BENCH.CASES]
    orders: set[tuple[str, ...]] = {
        tuple(case.name for case in BENCH._run_order(np.random.default_rng(seed))) for seed in range(20)
    }
    assert all(sorted(order) == sorted(declared) for order in orders)
    assert len(orders) > 1
    assert any(list(order) != declared for order in orders)


def test_generate_honours_the_row_group_layout(tmp_path):
    dataset = BENCH.generate(tmp_path, 1_000, 250)
    assert dataset.row_groups == 4
    for path in dataset.paths.values():
        assert pq.ParquetFile(path).num_row_groups == 4


def test_the_sweep_measures_more_than_one_layout():
    assert len(set(BENCH.SWEEP_ROW_GROUPS)) > 1
    assert min(BENCH.SWEEP_ROW_GROUPS) == 1


def test_both_scripts_summarise_their_runs_the_same_way():
    # The two once reported the best of three and the median of seven; sharing
    # the functions is what stops that from coming back.
    assert BENCH.repeat is MEASURE.repeat
    assert LABEL.repeat is MEASURE.repeat
    assert BENCH.REPEATS == MEASURE.REPEATS
    assert LABEL.REPEATS == MEASURE.REPEATS
    assert not hasattr(LABEL, "_best_of")


def test_both_scripts_write_upstream_output_through_the_same_sink_settings():
    assert BENCH.write_back is MEASURE.write_back
    assert LABEL.write_parts is MEASURE.write_parts
    assert BENCH.sink_layout is MEASURE.sink_layout
    assert LABEL.sink_layout is MEASURE.sink_layout
    assert LABEL.ROW_GROUP_SIZE == MEASURE.ROW_GROUP_SIZE


def test_as_table_takes_a_single_column_result():
    table = MEASURE.as_table(np.arange(6, dtype=np.int64))
    assert table.num_columns == 1
    assert table.num_rows == 6


def test_label_encoder_measures_in_a_shuffled_order():
    order: dict[str, Any] = {name: (lambda: None) for name in ("a", "b", "c", "d", "e")}
    orders: set[tuple[str, ...]] = set()
    for _ in range(20):
        stats, drawn = LABEL._measure(order, np.random.default_rng())
        assert sorted(stats) == sorted(order)
        assert sorted(drawn) == sorted(order)
        orders.add(tuple(drawn))
    assert len(orders) > 1
