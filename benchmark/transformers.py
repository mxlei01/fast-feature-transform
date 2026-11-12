"""Benchmark: fast-feature-transform transformers vs category_encoders and scikit-learn.

Five transformers, one per shape, so the table says something about the design
rather than about one kernel: a plain tally (``CountEncoder``), a supervised
table (``TargetEncoder``), an expanding encoder (``OneHotEncoder``), a
streaming numeric statistic (``StandardScaler``), and a buffered one
(``QuantileTransformer``). Four measurements: batch fit and transform over
Parquet on disk at 1M and 10M rows, the same batch transform with upstream's
output written to Parquet the way ours is, single-record serving latency, and
peak RSS during the fit. A row-group sweep sits beside them, because how the
input is laid out decides how many workers ours gets to use.

Where each side is favored, stated up front rather than buried:

- We do strictly more work in the batch column. Ours streams Parquet in and
  Parquet out, its row groups shared among up to one worker per core, and
  pays every byte of disk I/O in both directions. The upstream libraries
  work on an in-memory frame or array: the Parquet read is counted into their
  fit, and their transform writes nothing. Parquet is the only batch input
  ours takes, so the ``matched work`` table reports the mirror: upstream's
  transform with the same Parquet write added, and then with the Parquet
  read added as well.
- The write-back matches our sink setting for setting. Same Snappy
  compression, same Parquet 2.6 format, no stored Arrow or pandas schema,
  dictionary encoding disabled for floating-point columns and left on for
  everything else, the same dictionary page limit and the same 64 KiB write
  buffer in front of each file, which is what ``PartWriter`` in
  ``src/fastfeaturetransform/io.cpp`` does, and the same parts and row groups.
  The sink writes a row group per batch and sizes its batches by how wide the
  pipeline's rows are, so that layout is read back from the parts our own
  transform wrote rather than restated here. pyarrow's defaults would charge
  upstream for dictionary-encoding float columns that our own writer never
  dictionary-encodes: 174.4 ms against 69.5 ms on the 10M-row StandardScaler
  output, about 105 ms of work our sink does not do. Our sink also deletes the
  output directory's existing part files when a run starts and writes each part
  to a temporary name and renames it; the write-back does neither.
- Ours can use every core. One worker thread per row group, capped at the
  core count; pandas, NumPy and scikit-learn are single-threaded here. The
  ``ours 1 thread`` column is the same measurement with ``threads=1``, so the
  per-core comparison sits next to the default one.
- Row-group layout is a condition, not a constant. The main tables use
  500,000-row row groups (2 at 1M rows, 20 at 10M), which is a layout that
  suits us. The sweep at the end re-measures the 10M-row files at 1, 2, 10
  and 20 row groups so the reader can see how much of the result is the
  layout.
- Upstream writes only the column it encodes. Each transformer gets a
  Parquet file holding exactly the columns it needs, generated once per size
  from the same arrays, and both sides read all of it; ours then rewrites
  every column of that file. ``TargetEncoder`` is the case where that bites:
  our output carries the encoded ``device`` and the ``clicked`` target it was
  fitted against, upstream's carries ``device`` alone. A single-column file
  reaches NumPy with no copy beyond pyarrow's, as in upstream's own code.
- QuantileTransformer's ``subsample`` is the one tuned parameter. scikit-learn
  subsamples to 10,000 rows by default, which is much faster and much smaller
  but computes a different, sampled answer; we have no ``subsample``, so the
  comparison runs upstream with ``subsample=None``, the setting where both
  sides compute the same quantiles. The default-subsample fit is measured too
  and reported separately, labeled, because it answers a different question;
  it is timed exactly like the fit beside it, Parquet read included.
- Serving is one record in, one record out. Ours is ``transform_dict``;
  upstream is a one-row frame or array, built and transformed, which is what
  serving costs with those libraries: the construction of the one-row input
  is inside the timed call, as it is in production. The amortized vectorized
  number (a whole-batch transform divided by its rows) is reported in its own
  column because it answers a different question.

Both sides compute the same values — that is what the parity suite is for,
not this script — with one visible difference: our one-hot columns are in
sorted category order and upstream's are in order of first appearance, which is
the same 32 columns of work either way.

Fairness rules: every batch timing is the median of REPEATS runs with the
interquartile range beside it, never the best run, because the fastest run
hides both variance and the occasional multi-hundred-millisecond outlier; warm
cache; the same data for both sides; the transformers run in a random order at
each size, drawn afresh every run and printed, so cache and thermal drift do
not always land on the same one; one process, except for the peak-RSS runs,
which each start a fresh interpreter because ``ru_maxrss`` is a high-water
mark that never comes back down. A size an upstream library cannot do is
recorded as a limit rather than dropped.

Run with:
    uv run --with category_encoders==2.10.0 --with pandas python benchmark/transformers.py
"""

import os
import platform
import resource
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import sklearn
from sklearn.preprocessing import QuantileTransformer as SkQuantileTransformer
from sklearn.preprocessing import StandardScaler as SkStandardScaler

from fast_feature_transform import CountEncoder, OneHotEncoder, QuantileTransformer, StandardScaler, TargetEncoder
from fast_feature_transform.transformers import Transformer
from measure import (
    REPEATS,
    ROW_GROUP_SIZE,
    Stat,
    milliseconds,
    p99,
    repeat,
    sink_layout,
    write_back,
)

try:
    import category_encoders as ce  # ty: ignore[unresolved-import]
    import pandas as pd  # ty: ignore[unresolved-import]
except ImportError:
    print(
        "This benchmark needs category_encoders and pandas, which are deliberately not\n"
        "dependencies of this project (the parity fixtures are checked in, so the test\n"
        "run needs neither). Run it with:\n\n"
        "  uv run --with category_encoders==2.10.0 --with pandas python benchmark/transformers.py\n\n"
        "or `make transformer-benchmark`.",
        file=sys.stderr,
    )
    raise SystemExit(1) from None

SIZES: tuple[int, ...] = (1_000_000, 10_000_000)
SERVING_ROWS: int = 1_000_000
SWEEP_ROWS: int = 10_000_000
SWEEP_ROW_GROUPS: tuple[int, ...] = (1, 2, 10, 20)
SERVING_SAMPLES: int = 5000
WARMUP_SAMPLES: int = 1000
DEVICE_CLASSES: int = 1000
CHANNEL_CLASSES: int = 32
N_QUANTILES: int = 1000
SEED: int = 7


@dataclass(frozen=True)
class Dataset:
    """One generated size and layout: the Parquet files on disk and the vocabularies behind them.

    Attributes:
        rows: Row count of every file.
        directory: Directory holding the files and the transform outputs.
        paths: Parquet file per dataset key ("device", "target", "channel", "price").
        row_groups: Row groups per file; ours runs one worker per row group, up to one per core.
        devices: The 1,000-category vocabulary of the "device" column.
        channels: The 32-category vocabulary of the "channel" column.

    """

    rows: int
    directory: Path
    paths: dict[str, Path]
    row_groups: int
    devices: np.ndarray
    channels: np.ndarray


@dataclass(frozen=True)
class Case:
    """One transformer measured on both sides.

    Attributes:
        name: Our transformer's class name.
        upstream: The upstream class it was ported from.
        dataset: Which generated Parquet file it reads.
        build: Builds an unfitted instance of ours.
        read: Reads that Parquet file into upstream's input (timed inside its fit).
        fit: Fits the upstream estimator on that input.
        transform: Runs the upstream estimator's batch transform.
        record: Builds one serving record from the dataset's vocabularies.
        serve: Transforms one record the way upstream serving has to.

    """

    name: str
    upstream: str
    dataset: str
    build: Callable[[], Transformer]
    read: Callable[[Path], Any]
    fit: Callable[[Any], Any]
    transform: Callable[[Any, Any], Any]
    record: Callable[[Dataset, np.random.Generator], dict[str, Any]]
    serve: Callable[[Any, dict[str, Any]], Any]


@dataclass
class BatchResult:
    """Batch and memory numbers for one transformer at one size.

    Attributes:
        case: The transformer's name.
        rows: Rows processed.
        fit_ours: Our fit, all cores.
        fit_ours_1t: Our fit with threads=1.
        fit_upstream: Upstream's fit including its Parquet read, or None if it could not.
        transform_ours: Our Parquet-in, Parquet-out transform, all cores.
        transform_ours_1t: The same with threads=1.
        transform_upstream: Upstream's in-memory transform, or None.
        transform_upstream_written: Upstream's transform plus the Parquet write our sink does, or None.
        transform_upstream_full: Upstream's Parquet read, transform and Parquet write, or None.
        rss_ours: Peak RSS above the interpreter baseline for our fit, MiB.
        rss_upstream: The same for upstream's read-and-fit, MiB.
        note: A limit an upstream library hit, if any.

    """

    case: str
    rows: int
    fit_ours: Stat
    fit_ours_1t: Stat
    fit_upstream: Stat | None
    transform_ours: Stat
    transform_ours_1t: Stat
    transform_upstream: Stat | None
    transform_upstream_written: Stat | None
    transform_upstream_full: Stat | None
    rss_ours: float
    rss_upstream: float
    note: str


@dataclass(frozen=True)
class SweepResult:
    """Our own fit and transform over one row-group layout of the same values.

    Attributes:
        case: The transformer's name.
        row_groups: Row groups in each input file.
        workers: Workers ours runs: one per row group, capped at the core count.
        fit_ours: Our fit at that layout.
        transform_ours: Our transform at that layout.

    """

    case: str
    row_groups: int
    workers: int
    fit_ours: Stat
    transform_ours: Stat


@dataclass(frozen=True)
class ServingResult:
    """Single-record latencies for one transformer, in microseconds.

    Attributes:
        case: The transformer's name.
        ours_p50: Median ``transform_dict`` call.
        ours_p99: 99th percentile ``transform_dict`` call.
        upstream_p50: Median one-row upstream transform.
        upstream_p99: 99th percentile one-row upstream transform.
        upstream_amortised: Upstream's 1M-row vectorized transform divided by 1M, which answers a different question.

    """

    case: str
    ours_p50: float
    ours_p99: float
    upstream_p50: float
    upstream_p99: float
    upstream_amortised: float


def _read_frame(path: Path) -> pd.DataFrame:
    """Read a Parquet file into a pandas frame, the way category_encoders is used.

    Args:
        path: Parquet file to read.

    Returns:
        The whole file as a frame.

    """
    return pq.read_table(path).to_pandas()


def _read_matrix(path: Path) -> np.ndarray:
    """Read a Parquet file into the 2-D float array scikit-learn's scalers take.

    A one-column file (every file a scaler case reads) becomes a view of the
    array pyarrow returns, which is what upstream's own code would get;
    ``np.column_stack`` would copy the whole column first and charge upstream
    for work its API does not require.

    Args:
        path: Parquet file to read.

    Returns:
        The file's columns as one 2-D array.

    """
    table: pa.Table = pq.read_table(path)
    columns: list[np.ndarray] = [table.column(name).to_numpy(zero_copy_only=False) for name in table.schema.names]
    if len(columns) == 1:
        return columns[0].reshape(-1, 1)
    return np.column_stack(columns)


def _ce_count_fit(frame: pd.DataFrame) -> Any:
    """Fit category_encoders.CountEncoder on the device column.

    Args:
        frame: The device frame.

    Returns:
        The fitted encoder.

    """
    return ce.CountEncoder(cols=["device"]).fit(frame)


def _ce_target_fit(frame: pd.DataFrame) -> Any:
    """Fit category_encoders.TargetEncoder on the device column against the target.

    Args:
        frame: The device-and-target frame.

    Returns:
        The fitted encoder.

    """
    return ce.TargetEncoder(cols=["device"]).fit(frame[["device"]], frame["clicked"])


def _ce_onehot_fit(frame: pd.DataFrame) -> Any:
    """Fit category_encoders.OneHotEncoder on the channel column.

    Args:
        frame: The channel frame.

    Returns:
        The fitted encoder.

    """
    return ce.OneHotEncoder(cols=["channel"]).fit(frame)


def _sk_standard_fit(matrix: np.ndarray) -> Any:
    """Fit scikit-learn's StandardScaler.

    Args:
        matrix: The 2-D price array.

    Returns:
        The fitted scaler.

    """
    return SkStandardScaler().fit(matrix)


def _sk_quantile_fit(matrix: np.ndarray) -> Any:
    """Fit scikit-learn's QuantileTransformer over every row, as ours does.

    ``subsample=None`` is the setting where both sides compute the same
    quantiles; upstream's default samples 10,000 rows and answers a different
    question, which ``_measure_subsampled`` measures separately.

    Args:
        matrix: The 2-D price array.

    Returns:
        The fitted transformer.

    """
    return SkQuantileTransformer(n_quantiles=N_QUANTILES, subsample=None).fit(matrix)


def _sk_quantile_fit_subsampled(matrix: np.ndarray) -> Any:
    """Fit scikit-learn's QuantileTransformer at its default subsample of 10,000 rows.

    Args:
        matrix: The 2-D price array.

    Returns:
        The fitted transformer, whose quantiles come from a sample.

    """
    return SkQuantileTransformer(n_quantiles=N_QUANTILES).fit(matrix)


def _ce_transform(encoder: Any, frame: pd.DataFrame) -> Any:
    """Transform the whole frame with a fitted category_encoders encoder.

    Args:
        encoder: The fitted encoder.
        frame: The frame it was fitted on.

    Returns:
        The encoded frame.

    """
    return encoder.transform(frame)


def _ce_transform_device(encoder: Any, frame: pd.DataFrame) -> Any:
    """Transform the device column only, which is what TargetEncoder was fitted on.

    Args:
        encoder: The fitted encoder.
        frame: The device-and-target frame.

    Returns:
        The encoded frame.

    """
    return encoder.transform(frame[["device"]])


def _sk_transform(estimator: Any, matrix: np.ndarray) -> Any:
    """Transform the whole array with a fitted scikit-learn estimator.

    Args:
        estimator: The fitted estimator.
        matrix: The array it was fitted on.

    Returns:
        The transformed array.

    """
    return estimator.transform(matrix)


def _record_device(dataset: Dataset, rng: np.random.Generator) -> dict[str, Any]:
    """Build one serving record holding a device category.

    Args:
        dataset: The dataset whose vocabulary to draw from.
        rng: Random source.

    Returns:
        A one-key record.

    """
    return {"device": str(dataset.devices[rng.integers(0, DEVICE_CLASSES)])}


def _record_target(dataset: Dataset, rng: np.random.Generator) -> dict[str, Any]:
    """Build one serving record for TargetEncoder, which passes the target through.

    Args:
        dataset: The dataset whose vocabulary to draw from.
        rng: Random source.

    Returns:
        A record holding the category and the target column.

    """
    return {"device": str(dataset.devices[rng.integers(0, DEVICE_CLASSES)]), "clicked": float(rng.integers(0, 2))}


def _record_channel(dataset: Dataset, rng: np.random.Generator) -> dict[str, Any]:
    """Build one serving record holding a channel category.

    Args:
        dataset: The dataset whose vocabulary to draw from.
        rng: Random source.

    Returns:
        A one-key record.

    """
    return {"channel": str(dataset.channels[rng.integers(0, CHANNEL_CLASSES)])}


def _record_price(dataset: Dataset, rng: np.random.Generator) -> dict[str, Any]:
    """Build one serving record holding a price.

    Args:
        dataset: Unused; present for one record signature across cases.
        rng: Random source.

    Returns:
        A one-key record.

    """
    del dataset
    return {"price": float(rng.normal(100.0, 25.0))}


def _ce_serve_device(encoder: Any, record: dict[str, Any]) -> Any:
    """Serve one record through category_encoders: build a one-row frame and transform it.

    Args:
        encoder: The fitted encoder.
        record: The record to encode.

    Returns:
        The one-row encoded frame.

    """
    return encoder.transform(pd.DataFrame({"device": [record["device"]]}))


def _ce_serve_channel(encoder: Any, record: dict[str, Any]) -> Any:
    """Serve one record through category_encoders.OneHotEncoder.

    Args:
        encoder: The fitted encoder.
        record: The record to encode.

    Returns:
        The one-row encoded frame.

    """
    return encoder.transform(pd.DataFrame({"channel": [record["channel"]]}))


def _sk_serve_price(estimator: Any, record: dict[str, Any]) -> Any:
    """Serve one record through scikit-learn: build a 1x1 array and transform it.

    Args:
        estimator: The fitted estimator.
        record: The record to transform.

    Returns:
        The one-row transformed array.

    """
    return estimator.transform(np.array([[record["price"]]]))


CASES: tuple[Case, ...] = (
    Case(
        name="CountEncoder",
        upstream="category_encoders.CountEncoder",
        dataset="device",
        build=lambda: CountEncoder(columns=["device"]),
        read=_read_frame,
        fit=_ce_count_fit,
        transform=_ce_transform,
        record=_record_device,
        serve=_ce_serve_device,
    ),
    Case(
        name="TargetEncoder",
        upstream="category_encoders.TargetEncoder",
        dataset="target",
        build=lambda: TargetEncoder(target="clicked", columns=["device"]),
        read=_read_frame,
        fit=_ce_target_fit,
        transform=_ce_transform_device,
        record=_record_target,
        serve=_ce_serve_device,
    ),
    Case(
        name="OneHotEncoder",
        upstream="category_encoders.OneHotEncoder",
        dataset="channel",
        build=lambda: OneHotEncoder(columns=["channel"]),
        read=_read_frame,
        fit=_ce_onehot_fit,
        transform=_ce_transform,
        record=_record_channel,
        serve=_ce_serve_channel,
    ),
    Case(
        name="StandardScaler",
        upstream="sklearn.preprocessing.StandardScaler",
        dataset="price",
        build=lambda: StandardScaler(columns=["price"]),
        read=_read_matrix,
        fit=_sk_standard_fit,
        transform=_sk_transform,
        record=_record_price,
        serve=_sk_serve_price,
    ),
    Case(
        name="QuantileTransformer",
        upstream="sklearn.preprocessing.QuantileTransformer",
        dataset="price",
        build=lambda: QuantileTransformer(columns=["price"], n_quantiles=N_QUANTILES),
        read=_read_matrix,
        fit=_sk_quantile_fit,
        transform=_sk_transform,
        record=_record_price,
        serve=_sk_serve_price,
    ),
)

BY_NAME: dict[str, Case] = {case.name: case for case in CASES}


def _time_fit(read: Callable[[Path], Any], fit: Callable[[Any], Any], path: Path) -> Stat:
    """Time ``fit(read(path))``: upstream's fit with its Parquet read inside the clock.

    Every upstream fit in this script is measured through here, so the
    subsampled QuantileTransformer fit and the exact one beside it are the same
    measurement of two settings rather than two different measurements.

    Args:
        read: How this case reads its Parquet file.
        fit: The upstream fit to time.
        path: The Parquet file to read.

    Returns:
        The summary across repeats.

    """
    return repeat(lambda: fit(read(path)))


def _run_order(rng: np.random.Generator) -> list[Case]:
    """Return the cases in a random order.

    A fixed order lets cache state and thermal drift settle on the same
    transformers every run, and a fixed seed only changes which order that is,
    so the draw is unseeded and the order it produced is printed with the
    results.

    Args:
        rng: Random source.

    Returns:
        Every case, once, in a random order.

    """
    return [CASES[index] for index in rng.permutation(len(CASES))]


def _peak_rss_mib() -> float:
    """Return this process's peak resident set size in mebibytes.

    ``ru_maxrss`` is bytes on macOS and kibibytes on Linux.

    Returns:
        The high-water mark in MiB.

    """
    peak: int = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 2**20 if sys.platform == "darwin" else peak / 2**10


def generate(root: Path, rows: int, row_group_size: int = ROW_GROUP_SIZE) -> Dataset:
    """Write one Parquet file per transformer input shape, all from the same arrays.

    Every file is generated once per size and layout and reused by every
    measurement, so both sides see identical bytes. Each transformer gets
    exactly the columns it needs, so upstream is never charged for reading a
    column it ignores. The row-group size is a parameter because it decides
    how many workers ours gets, which the sweep at the end of the run measures.

    Args:
        root: Directory to create the dataset's own directory under.
        rows: Row count to generate.
        row_group_size: Rows per row group in every generated file.

    Returns:
        The dataset description.

    """
    directory: Path = root / f"{rows}-{row_group_size}"
    directory.mkdir(parents=True, exist_ok=True)
    rng: np.random.Generator = np.random.default_rng(SEED)
    devices: np.ndarray = np.array([f"cat_{i:04d}" for i in range(DEVICE_CLASSES)])
    channels: np.ndarray = np.array([f"ch_{i:02d}" for i in range(CHANNEL_CLASSES)])
    device: np.ndarray = devices[rng.integers(0, DEVICE_CLASSES, rows)]
    channel: np.ndarray = channels[rng.integers(0, CHANNEL_CLASSES, rows)]
    price: np.ndarray = rng.normal(100.0, 25.0, rows)
    clicked: np.ndarray = rng.integers(0, 2, rows).astype(np.float64)
    tables: dict[str, pa.Table] = {
        "device": pa.table({"device": device}),
        "target": pa.table({"device": device, "clicked": clicked}),
        "channel": pa.table({"channel": channel}),
        "price": pa.table({"price": price}),
    }
    paths: dict[str, Path] = {}
    for key, table in tables.items():
        paths[key] = directory / f"{key}.parquet"
        pq.write_table(table, paths[key], row_group_size=row_group_size)
    row_groups: int = pq.ParquetFile(paths["device"]).num_row_groups
    return Dataset(rows, directory, paths, row_groups, devices, channels)


def measure_batch(dataset: Dataset, case: Case) -> BatchResult:
    """Time both sides' fit and transform at one size.

    Ours streams the Parquet file in and part files out; upstream reads the
    file into memory (counted into its fit) and transforms it in memory. The
    two write-back numbers repeat upstream's transform with the Parquet write
    ours already pays for, and then with the Parquet read as well, which is
    the whole of what ``transform_parquet`` does.

    Args:
        dataset: The generated size.
        case: The transformer to measure.

    Returns:
        Everything but the RSS fields, which measure_rss fills in.

    """
    path: Path = dataset.paths[case.dataset]
    sources: list[str] = [str(path)]
    out_dir: Path = dataset.directory / f"out-{case.name}"
    fit_ours: Stat = repeat(lambda: case.build().fit(sources))
    fit_ours_1t: Stat = repeat(lambda: case.build().fit(sources, threads=1))
    ours: Transformer = case.build().fit(sources)
    ours_1t: Transformer = case.build().fit(sources, threads=1)
    transform_ours: Stat = repeat(lambda: ours.transform_parquet(sources, output_dir=out_dir))
    transform_ours_1t: Stat = repeat(lambda: ours_1t.transform_parquet(sources, output_dir=out_dir))
    # Upstream's output is written in the parts and row groups ours was.
    layout: list[tuple[int, int]] = sink_layout(ours.transform_parquet(sources, output_dir=out_dir))

    note: str = ""
    fit_upstream: Stat | None = None
    transform_upstream: Stat | None = None
    written: Stat | None = None
    full: Stat | None = None
    out: Path = dataset.directory / f"upstream-{case.name}"
    try:
        fit_upstream = _time_fit(case.read, case.fit, path)
        data: Any = case.read(path)
        fitted: Any = case.fit(data)
        transform_upstream = repeat(lambda: case.transform(fitted, data))
        written = repeat(lambda: write_back(case.transform(fitted, data), out, layout))
        full = repeat(lambda: write_back(case.transform(fitted, case.read(path)), out, layout))
    except MemoryError:
        note = f"{case.upstream} ran out of memory at {dataset.rows:,} rows"
    return BatchResult(
        case=case.name,
        rows=dataset.rows,
        fit_ours=fit_ours,
        fit_ours_1t=fit_ours_1t,
        fit_upstream=fit_upstream,
        transform_ours=transform_ours,
        transform_ours_1t=transform_ours_1t,
        transform_upstream=transform_upstream,
        transform_upstream_written=written,
        transform_upstream_full=full,
        rss_ours=0.0,
        rss_upstream=0.0,
        note=note,
    )


def measure_sweep(dataset: Dataset, case: Case) -> SweepResult:
    """Time our own fit and transform over one row-group layout.

    Only our side moves with the layout: upstream reads the whole file into
    memory either way. Ours runs one worker per row group, up to the core
    count, so this is also the parallelism sweep.

    Args:
        dataset: The generated size at this layout.
        case: The transformer to measure.

    Returns:
        Our two timings at this layout.

    """
    sources: list[str] = [str(dataset.paths[case.dataset])]
    out_dir: Path = dataset.directory / f"out-{case.name}"
    fit_ours: Stat = repeat(lambda: case.build().fit(sources))
    ours: Transformer = case.build().fit(sources)
    transform_ours: Stat = repeat(lambda: ours.transform_parquet(sources, output_dir=out_dir))
    return SweepResult(
        case=case.name,
        row_groups=dataset.row_groups,
        workers=min(dataset.row_groups, os.cpu_count() or 1),
        fit_ours=fit_ours,
        transform_ours=transform_ours,
    )


def measure_serving(dataset: Dataset, case: Case, amortised: float) -> ServingResult:
    """Time one record in, one record out, on both sides.

    Ours is ``transform_dict``. Upstream's honest equivalent is building the
    one-row frame or array a serving path would build and transforming it;
    that construction is inside the timed call.

    Args:
        dataset: The size to fit both sides on.
        case: The transformer to measure.
        amortised: Upstream's vectorized transform over this dataset divided
            by its rows, in microseconds, carried through for the table.

    Returns:
        The per-call latencies in microseconds.

    """
    path: Path = dataset.paths[case.dataset]
    ours: Transformer = case.build().fit([str(path)])
    fitted: Any = case.fit(case.read(path))
    rng: np.random.Generator = np.random.default_rng(SEED)
    records: list[dict[str, Any]] = [case.record(dataset, rng) for _ in range(SERVING_SAMPLES)]

    def timed(fn: Callable[[dict[str, Any]], object]) -> list[float]:
        """Replay the first WARMUP_SAMPLES records untimed, then time every record once.

        Args:
            fn: The single-record transform to time.

        Returns:
            Per-call wall times in microseconds.

        """
        for record in records[:WARMUP_SAMPLES]:
            fn(record)
        latencies: list[float] = []
        for record in records:
            start: int = time.perf_counter_ns()
            fn(record)
            latencies.append((time.perf_counter_ns() - start) / 1e3)
        return latencies

    ours_latencies: list[float] = timed(ours.transform_dict)
    upstream_latencies: list[float] = timed(lambda record: case.serve(fitted, record))
    return ServingResult(
        case=case.name,
        ours_p50=statistics.median(ours_latencies),
        ours_p99=p99(ours_latencies),
        upstream_p50=statistics.median(upstream_latencies),
        upstream_p99=p99(upstream_latencies),
        upstream_amortised=amortised,
    )


def measure_rss(dataset: Dataset, case: Case) -> tuple[float, float]:
    """Measure each side's peak RSS for the fit, one fresh subprocess each.

    ``ru_maxrss`` is a high-water mark that never comes down, so the two fits
    cannot share a process. Each child reports its peak above its own
    post-import baseline; both children import the same modules, so the
    baseline cancels and what is left is what the fit cost.

    Args:
        dataset: The generated size.
        case: The transformer to measure.

    Returns:
        Our peak and upstream's peak, in MiB above the baseline.

    """
    return (
        _rss_child(case.name, "ours", dataset.directory),
        _rss_child(case.name, "upstream", dataset.directory),
    )


def _rss_child(name: str, side: str, directory: Path) -> float:
    """Run one fit in a fresh interpreter and return its peak RSS above baseline.

    Args:
        name: The case name.
        side: "ours" or "upstream".
        directory: The dataset directory the child reads from.

    Returns:
        Peak RSS above the child's post-import baseline, in MiB.

    """
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--rss", name, side, str(directory)],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip().split()[-1])


def _run_rss_child(name: str, side: str, directory: Path) -> None:
    """Fit one side once and print its peak RSS above this interpreter's baseline.

    Args:
        name: The case name.
        side: "ours" or "upstream".
        directory: The dataset directory to read from.

    """
    case: Case = BY_NAME[name]
    path: Path = directory / f"{case.dataset}.parquet"
    baseline: float = _peak_rss_mib()
    if side == "ours":
        case.build().fit([str(path)])
    else:
        case.fit(case.read(path))
    print(f"{baseline:.1f} {_peak_rss_mib() - baseline:.1f}")


def _cpu_name() -> str:
    """Return a readable CPU name, falling back to the architecture.

    Returns:
        The vendor's brand string where the platform exposes one.

    """
    try:
        if sys.platform == "darwin":
            probe = subprocess.run(
                ["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True,
            )
            return probe.stdout.strip()
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError, subprocess.SubprocessError:
        pass
    return platform.processor() or platform.machine()


def _machine() -> str:
    """Describe the machine the numbers were measured on.

    Returns:
        One line naming the CPU, core count, memory, OS, Python, and the
        category_encoders and scikit-learn versions.

    """
    cores: int = os.cpu_count() or 0
    try:
        memory: float = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    except AttributeError, ValueError, OSError:
        memory = 0.0
    return (
        f"{_cpu_name()}, {cores} cores, {memory:.0f} GiB RAM, "
        f"{platform.system()} {platform.release()}, Python {platform.python_version()}, "
        f"category_encoders {ce.__version__}, scikit-learn {sklearn.__version__}"
    )


def _times(value: float) -> str:
    """Format a times-faster figure, keeping a decimal where the factor is small.

    Args:
        value: How many times faster ours is; below 1 means upstream won.

    Returns:
        The formatted factor.

    """
    return f"{value:,.1f}x" if value < 100 else f"{value:,.0f}x"


def _ratio(ours: Stat, upstream: Stat | None) -> str:
    """Format upstream's median divided by ours, the times-faster figure.

    Args:
        ours: Our measurement.
        upstream: Upstream's measurement, or None if it could not run.

    Returns:
        The ratio, or an em dash.

    """
    return "—" if upstream is None else _times(upstream.median / ours.median)


def _ms(stat: Stat | None) -> str:
    """Format a measurement as its median in milliseconds with the interquartile range beside it.

    Args:
        stat: The measurement, or None.

    Returns:
        ``median (IQR)`` in milliseconds, or an em dash.

    """
    return "—" if stat is None else milliseconds(stat)


def _rows(rows: int) -> str:
    """Format a row count as 1M or 10M.

    Args:
        rows: The row count.

    Returns:
        The short label.

    """
    return f"{rows / 1_000_000:g}M"


def print_batch_table(results: list[BatchResult]) -> None:
    """Print the batch table: fit and transform per transformer per size, with peak RSS.

    Args:
        results: Every batch measurement, in run order.

    """
    print(f"\n### Batch (median of {REPEATS}, IQR in parentheses, warm cache)\n")
    print(
        "| transformer | mode | rows | ours ms (IQR) | ours 1 thread | upstream | ratio | "
        "peak RSS ours | peak RSS upstream |"
    )
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for result in results:
        print(
            f"| {result.case} | fit | {_rows(result.rows)} | {_ms(result.fit_ours)} | "
            f"{_ms(result.fit_ours_1t)} | {_ms(result.fit_upstream)} | "
            f"{_ratio(result.fit_ours, result.fit_upstream)} | {result.rss_ours:,.0f} MiB | "
            f"{result.rss_upstream:,.0f} MiB |"
        )
        print(
            f"| {result.case} | transform | {_rows(result.rows)} | {_ms(result.transform_ours)} | "
            f"{_ms(result.transform_ours_1t)} | {_ms(result.transform_upstream)} | "
            f"{_ratio(result.transform_ours, result.transform_upstream)} | — | — |"
        )
    for result in results:
        if result.note:
            print(f"\n{result.note}")


def print_spread_table(results: list[BatchResult]) -> None:
    """Print the fastest, median and slowest run and the IQR of each of our own measurements.

    Args:
        results: Every batch measurement, in run order.

    """
    print(f"\n### Spread of our own batch runs ({REPEATS} repeats each)\n")
    print("| transformer | mode | rows | fastest | median | slowest | IQR |")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for result in results:
        for mode, stat in (("fit", result.fit_ours), ("transform", result.transform_ours)):
            print(
                f"| {result.case} | {mode} | {_rows(result.rows)} | {stat.low * 1e3:,.1f} ms | "
                f"{stat.median * 1e3:,.1f} ms | {stat.high * 1e3:,.1f} ms | {stat.iqr * 1e3:,.1f} ms |"
            )


def print_io_table(results: list[BatchResult]) -> None:
    """Print what the batch transform column hides: ours reads and writes Parquet, upstream does neither.

    Args:
        results: Every batch measurement, in run order.

    """
    print("\n### Matched work: the same transform with our sink's Parquet write added\n")
    print(
        "| transformer | rows | ours (Parquet in, Parquet out) | upstream (in memory) | "
        "upstream + Parquet out | upstream + Parquet in and out |"
    )
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for result in results:
        print(
            f"| {result.case} | {_rows(result.rows)} | {_ms(result.transform_ours)} | "
            f"{_ms(result.transform_upstream)} | {_ms(result.transform_upstream_written)} "
            f"({_ratio(result.transform_ours, result.transform_upstream_written)}) | "
            f"{_ms(result.transform_upstream_full)} "
            f"({_ratio(result.transform_ours, result.transform_upstream_full)}) |"
        )


def print_sweep_table(results: list[SweepResult]) -> None:
    """Print how much of our batch number is the row-group layout of the input.

    Args:
        results: Every sweep measurement, in run order.

    """
    print(f"\n### Row-group sweep: the same {SWEEP_ROWS:,} values laid out {len(SWEEP_ROW_GROUPS)} ways\n")
    print("| transformer | row groups | workers | ours fit | ours transform |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for result in results:
        print(
            f"| {result.case} | {result.row_groups} | {result.workers} | {_ms(result.fit_ours)} | "
            f"{_ms(result.transform_ours)} |"
        )


def print_serving_table(results: list[ServingResult]) -> None:
    """Print the serving table: per-call latency, and the amortized number beside it.

    Args:
        results: Every serving measurement.

    """
    print(f"\n### Serving, one record per call ({SERVING_SAMPLES:,} calls after {WARMUP_SAMPLES:,} warm-up)\n")
    print("| transformer | ours p50 | ours p99 | upstream p50 | upstream p99 | ratio at p50 | upstream amortized |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for result in results:
        print(
            f"| {result.case} | {result.ours_p50:.2f} us | {result.ours_p99:.2f} us | "
            f"{result.upstream_p50:,.1f} us | {result.upstream_p99:,.1f} us | "
            f"{_times(result.upstream_p50 / result.ours_p50)} | {result.upstream_amortised:.3g} us |"
        )


def print_quantile_note(subsampled: dict[int, Stat], results: list[BatchResult]) -> None:
    """Print the one place a parameter was chosen rather than left at its default.

    Both upstream columns are the same measurement (read the Parquet file,
    then fit) with one setting changed, so they can be compared to each other.

    Args:
        subsampled: Upstream's default-subsample fit per size.
        results: The batch results, for the matching exact fits.

    """
    print("\n### QuantileTransformer: what upstream's default would have measured instead\n")
    print("| rows | ours (every row) | upstream subsample=None (same answer) | upstream default subsample=10,000 |")
    print("| ---: | ---: | ---: | ---: |")
    for result in results:
        if result.case == "QuantileTransformer":
            print(
                f"| {_rows(result.rows)} | {_ms(result.fit_ours)} | {_ms(result.fit_upstream)} | "
                f"{_ms(subsampled[result.rows])} |"
            )


def _measure_subsampled(path: Path) -> Stat:
    """Time scikit-learn's QuantileTransformer at its default subsample.

    Timed through ``_time_fit`` like every other upstream fit, Parquet read
    included, so it lines up with the ``subsample=None`` fit printed beside it.

    Args:
        path: The price Parquet file.

    Returns:
        The summary across repeats.

    """
    return _time_fit(_read_matrix, _sk_quantile_fit_subsampled, path)


def _run_sizes(root: Path) -> tuple[list[BatchResult], list[ServingResult], dict[int, Stat]]:
    """Measure every case at every size, in a random order per size.

    Args:
        root: Directory to generate the datasets under.

    Returns:
        The batch results, the serving results, and the subsampled quantile fits.

    """
    results: list[BatchResult] = []
    serving: list[ServingResult] = []
    subsampled: dict[int, Stat] = {}
    order_rng: np.random.Generator = np.random.default_rng()
    for rows in SIZES:
        dataset: Dataset = generate(root, rows)
        order: list[Case] = _run_order(order_rng)
        workers: int = min(dataset.row_groups, os.cpu_count() or 1)
        print(
            f"\n{rows:,} rows, {dataset.row_groups} row groups per file, {workers} worker threads for ours", flush=True
        )
        print(f"  order: {', '.join(case.name for case in order)}", flush=True)
        for case in order:
            result: BatchResult = measure_batch(dataset, case)
            result.rss_ours, result.rss_upstream = measure_rss(dataset, case)
            results.append(result)
            print(f"  {case.name} done", flush=True)
        subsampled[rows] = _measure_subsampled(dataset.paths["price"])
        if rows == SERVING_ROWS:
            for case in order:
                batch: BatchResult = next(r for r in results if r.case == case.name and r.rows == rows)
                upstream: Stat | None = batch.transform_upstream
                amortised: float = 0.0 if upstream is None else upstream.median / rows * 1e6
                serving.append(measure_serving(dataset, case, amortised))
                print(f"  {case.name} serving done", flush=True)
        shutil.rmtree(dataset.directory, ignore_errors=True)
    results.sort(key=lambda r: (r.rows, [c.name for c in CASES].index(r.case)))
    serving.sort(key=lambda r: [c.name for c in CASES].index(r.case))
    return results, serving, subsampled


def _run_sweep(root: Path) -> list[SweepResult]:
    """Re-measure our side over the same values at four row-group layouts.

    Args:
        root: Directory to generate the datasets under.

    Returns:
        Our fit and transform per case per layout.

    """
    sweep: list[SweepResult] = []
    order_rng: np.random.Generator = np.random.default_rng()
    for groups in SWEEP_ROW_GROUPS:
        dataset: Dataset = generate(root, SWEEP_ROWS, SWEEP_ROWS // groups)
        print(f"\nsweep: {dataset.row_groups} row groups per file", flush=True)
        for case in _run_order(order_rng):
            sweep.append(measure_sweep(dataset, case))
            print(f"  {case.name} done", flush=True)
        shutil.rmtree(dataset.directory, ignore_errors=True)
    sweep.sort(key=lambda r: ([c.name for c in CASES].index(r.case), r.row_groups))
    return sweep


def main() -> None:
    """Run every measurement and print the tables."""
    print(f"machine: {_machine()}")
    print(f"method: median of {REPEATS} runs, IQR beside it, a fresh random case order per phase")
    started: float = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="fast_feature_transform_bench_") as tmp:
        root: Path = Path(tmp)
        results, serving, subsampled = _run_sizes(root)
        sweep: list[SweepResult] = _run_sweep(root)
    print_batch_table(results)
    print_spread_table(results)
    print_io_table(results)
    print_sweep_table(sweep)
    print_serving_table(serving)
    print_quantile_note(subsampled, results)
    print(f"\nwhole run: {time.perf_counter() - started:,.0f} s")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--rss":
        _run_rss_child(sys.argv[2], sys.argv[3], Path(sys.argv[4]))
    else:
        main()
