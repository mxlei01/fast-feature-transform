"""Benchmark: fast-feature-transform LabelEncoder vs scikit-learn.

Every batch timing here is the median of ``REPEATS`` runs with the
interquartile range beside it, measured through ``benchmark/measure.py`` — the
same module ``benchmark/transformers.py`` measures through, so the two
scripts' numbers are comparable.

Batch: fast-feature-transform streams Parquet in and Parquet out with one worker
thread per row group, up to one per core, and pays every byte of disk I/O in
both directions. scikit-learn works on an in-memory NumPy array, with the
Parquet read counted into its fit. Because those are not the same work,
upstream's transform is timed three ways: as scikit-learn performs it,
returning the array in memory; with our sink's Parquet write added; and with
the Parquet read added as well, which is the only row where both sides do the
same work.

Serving: one-record ``transform_dict`` calls are timed against a per-row
scikit-learn transform with unseen-category masking, and the table reports the
median and 99th percentile per call.

Run with: make label-encoder-benchmark.
"""

import os
import statistics
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.preprocessing import LabelEncoder as SkLabelEncoder

from fast_feature_transform import LabelEncoder, Pipeline
from measure import REPEATS, ROW_GROUP_SIZE, Stat, milliseconds, p99, repeat, sink_layout, write_parts

NUM_CLASSES: int = 1000
SERVING_SAMPLES: int = 5000
WARMUP_SAMPLES: int = 1000
BATCH_ROWS: tuple[int, ...] = (1_000_000, 10_000_000)


@dataclass(frozen=True)
class BatchResult:
    """One batch size measured on both sides.

    Attributes:
        rows: Rows in the generated file.
        row_groups: Row groups in it; ours runs one worker per row group, up to one per core.
        order: The order the measurements were taken in, as drawn for this run.
        stats: Measurement name -> its summary.

    """

    rows: int
    row_groups: int
    order: tuple[str, ...]
    stats: dict[str, Stat]

    @property
    def comparisons(self) -> tuple[tuple[str, Stat, Stat], ...]:
        """Return the rows of the printed table: a label and the two sides it compares.

        Returns:
            Label, our summary, upstream's summary.

        """
        return (
            ("fit", self.stats["ours fit"], self.stats["upstream fit"]),
            ("transform, upstream in memory", self.stats["ours transform"], self.stats["upstream transform"]),
            ("transform, upstream to Parquet", self.stats["ours transform"], self.stats["upstream transform written"]),
            (
                "transform, upstream Parquet in and out",
                self.stats["ours transform"],
                self.stats["upstream transform read and written"],
            ),
        )


@dataclass(frozen=True)
class ServingResult:
    """One-record latencies from both sides, in microseconds.

    Attributes:
        ours: Every timed call through ``transform_dict``.
        upstream: Every timed per-row scikit-learn call.

    """

    ours: list[float]
    upstream: list[float]


def _read(path: Path) -> np.ndarray:
    """Read the encoded column out of a Parquet file as the NumPy array upstream works on.

    Args:
        path: The Parquet file to read.

    Returns:
        The "device" column as an array of Python objects.

    """
    return pq.read_table(path).column("device").to_numpy(zero_copy_only=False)


def _classes() -> np.ndarray:
    """Return the fixed category vocabulary.

    Returns:
        Array of NUM_CLASSES category strings.

    """
    return np.array([f"cat_{i:04d}" for i in range(NUM_CLASSES)])


def _measure(order: dict[str, Callable[[], object]], rng: np.random.Generator) -> tuple[dict[str, Stat], list[str]]:
    """Time each named measurement, in an order drawn afresh for this run.

    A fixed order lets cache state and thermal drift settle on the same side
    of the comparison every time, and a fixed seed only changes which order
    that is, so the draw is unseeded and the order it produced is printed with
    the results.

    Args:
        order: Measurement name -> the zero-argument callable to time.
        rng: Random source for the draw.

    Returns:
        The summary per name, and the order they were measured in.

    """
    names: list[str] = [list(order)[index] for index in rng.permutation(len(order))]
    return {name: repeat(order[name]) for name in names}, names


def batch(rows: int) -> BatchResult:
    """Benchmark fit and transform over a Parquet file of the given size.

    Args:
        rows: Number of rows to generate and process.

    Returns:
        Every measurement taken at this size.

    """
    rng = np.random.default_rng(7)
    classes: np.ndarray = _classes()
    values: np.ndarray = classes[rng.integers(0, NUM_CLASSES, rows)]
    with tempfile.TemporaryDirectory(prefix="fast_feature_transform_bench_") as tmp_name:
        tmp: Path = Path(tmp_name)
        source: Path = tmp / "data.parquet"
        pq.write_table(pa.table({"device": values}), source, row_group_size=ROW_GROUP_SIZE)
        row_groups: int = pq.ParquetFile(source).num_row_groups

        encoder: LabelEncoder = LabelEncoder(columns=["device"]).fit([str(source)])
        sk = SkLabelEncoder()
        array: np.ndarray = _read(source)
        sk.fit(array)
        # Our sink names the encoded column, picks its type and lays out its
        # parts and row groups; the matched write-back has to write that
        # column that way, not one of its own invention.
        parts: list[Path] = encoder.transform_parquet([str(source)], output_dir=tmp / "schema")
        column: str = pq.ParquetFile(parts[0]).schema_arrow.names[0]
        layout: list[tuple[int, int]] = sink_layout(parts)

        stats, order = _measure(
            {
                "ours fit": lambda: LabelEncoder(columns=["device"]).fit([str(source)]),
                "ours transform": lambda: encoder.transform_parquet([str(source)], output_dir=tmp / "out"),
                "upstream fit": lambda: sk.fit(_read(source)),
                "upstream transform": lambda: sk.transform(array),
                "upstream transform written": lambda: write_parts(
                    pa.table({column: sk.transform(array)}), tmp / "upstream_out", layout
                ),
                "upstream transform read and written": lambda: write_parts(
                    pa.table({column: sk.transform(_read(source))}), tmp / "upstream_full", layout
                ),
            },
            np.random.default_rng(),
        )

    return BatchResult(rows=rows, row_groups=row_groups, order=tuple(order), stats=stats)


def print_batch(result: BatchResult) -> None:
    """Print one batch result as a Markdown table.

    Args:
        result: The measurements to print.

    """
    workers: int = min(result.row_groups, os.cpu_count() or 1)
    print(f"\n#### {result.rows:,} rows, {result.row_groups} row groups, {workers} worker threads\n")
    print(f"measured in this order: {', '.join(result.order)}\n")
    print("| measurement | ours | upstream | ratio |")
    print("| --- | ---: | ---: | ---: |")
    for label, mine, theirs in result.comparisons:
        print(f"| {label} | {milliseconds(mine)} | {milliseconds(theirs)} | {theirs.median / mine.median:.1f}x |")


def _per_call_latencies(fn: Callable[[dict[str, str]], object], records: list[dict[str, str]]) -> list[float]:
    """Time fn once per record after a warm-up and return microseconds per call.

    Args:
        fn: The single-record transform to time.
        records: Records to feed; the first WARMUP_SAMPLES are replayed untimed first.

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


def serving() -> ServingResult:
    """Benchmark single-record transform_dict against per-row scikit-learn.

    Returns:
        Every timed call from both sides.

    """
    rng = np.random.default_rng(7)
    classes: np.ndarray = _classes()
    with tempfile.TemporaryDirectory(prefix="fast_feature_transform_bench_") as tmp_name:
        source: Path = Path(tmp_name) / "data.parquet"
        pq.write_table(pa.table({"device": classes[rng.integers(0, NUM_CLASSES, 100_000)]}), source)
        pipeline: Pipeline = Pipeline([("encode", LabelEncoder(columns=["device"]))]).fit([str(source)])
        sk = SkLabelEncoder()
        sk.fit(_read(source))

    records: list[dict[str, str]] = [
        {"device": str(classes[rng.integers(0, NUM_CLASSES)])} for _ in range(SERVING_SAMPLES)
    ]

    def sk_transform_row(record: dict[str, str]) -> np.ndarray:
        array = np.array([record["device"]])
        known = np.isin(array, sk.classes_)
        out = np.zeros(1, dtype=np.int64)
        out[known] = sk.transform(array[known]) + 1
        return out

    return ServingResult(
        ours=_per_call_latencies(pipeline.transform_dict, records),
        upstream=_per_call_latencies(sk_transform_row, records),
    )


def print_serving(result: ServingResult) -> None:
    """Print the serving latencies as a Markdown table.

    Args:
        result: The measurements to print.

    """
    ours_p50: float = statistics.median(result.ours)
    upstream_p50: float = statistics.median(result.upstream)
    print(f"\n#### Serving, one record per call ({SERVING_SAMPLES:,} calls after {WARMUP_SAMPLES:,} warm-up)\n")
    print("| | p50 | p99 |")
    print("| --- | ---: | ---: |")
    print(f"| ours | {ours_p50:.2f} us | {p99(result.ours):.2f} us |")
    print(f"| scikit-learn | {upstream_p50:,.2f} us | {p99(result.upstream):,.2f} us |")
    print(f"\n{upstream_p50 / ours_p50:,.0f}x at p50.")


if __name__ == "__main__":
    print(f"LabelEncoder vs scikit-learn, median of {REPEATS} runs with the IQR in parentheses, warm cache.")
    for size in BATCH_ROWS:
        print_batch(batch(size))
    print_serving(serving())
