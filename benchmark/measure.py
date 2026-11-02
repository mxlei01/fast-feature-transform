"""How both benchmark scripts measure, so that the two cannot report different methodologies.

The scripts are ``label_encoder.py`` and ``transformers.py``. Each batch
timing is the median of ``REPEATS`` runs with the interquartile range beside
it, and upstream's output is written to Parquet the way our sink writes ours.
"""

import statistics
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

REPEATS: int = 7
ROW_GROUP_SIZE: int = 500_000


@dataclass(frozen=True)
class Stat:
    """One measurement repeated REPEATS times, summarized by its middle rather than its best.

    Attributes:
        samples: Every wall time collected, in seconds, sorted.
        median: The middle sample.
        q1: First quartile.
        q3: Third quartile.
        low: Fastest sample.
        high: Slowest sample.

    """

    samples: tuple[float, ...]
    median: float
    q1: float
    q3: float
    low: float
    high: float

    @property
    def iqr(self) -> float:
        """Return the interquartile range, the spread the tables print.

        Returns:
            ``q3 - q1`` in seconds.

        """
        return self.q3 - self.q1


def summarise(samples: Iterable[float]) -> Stat:
    """Summarize repeated wall times by their median and interquartile range.

    Args:
        samples: The collected times, in seconds.

    Returns:
        The summary the tables print.

    """
    values: list[float] = sorted(samples)
    quartiles: list[float] = statistics.quantiles(values, n=4, method="inclusive")
    return Stat(
        samples=tuple(values),
        median=statistics.median(values),
        q1=quartiles[0],
        q3=quartiles[2],
        low=values[0],
        high=values[-1],
    )


def repeat(fn: Callable[[], object]) -> Stat:
    """Run fn REPEATS times and summarize the wall times.

    The median, not the minimum: the fastest of a few warm runs reads as a
    tighter number than the workload delivers, and it hides the occasional
    outlier that a serving or batch schedule pays.

    Args:
        fn: Zero-argument callable to time.

    Returns:
        The summary across repeats.

    """
    samples: list[float] = []
    for _ in range(REPEATS):
        start: float = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return summarise(samples)


def p99(latencies: list[float]) -> float:
    """Return the 99th percentile of the given samples.

    Args:
        latencies: Sample values.

    Returns:
        The value below which 99% of samples fall.

    """
    return sorted(latencies)[int(0.99 * (len(latencies) - 1))]


def milliseconds(stat: Stat) -> str:
    """Format a stat as ``median (IQR)`` in milliseconds.

    Args:
        stat: The summary to format.

    Returns:
        The two numbers, comma-grouped, one decimal each.

    """
    return f"{stat.median * 1e3:,.1f} ({stat.iqr * 1e3:,.1f})"


def dictionary_columns(schema: pa.Schema) -> list[str]:
    """Return the columns our sink leaves dictionary-encoded: every one that is not a float.

    ``PartWriter`` in ``src/fastfeaturetransform/io.cpp``, which writes every part
    our batch sink produces, disables dictionary encoding for floating-point
    columns and leaves Parquet's default on for the rest, so a matched
    write-back has to do the same. Charging upstream for
    dictionary-encoding a float column ours writes plain is the difference
    between a matched comparison and a flattering one.

    Args:
        schema: The output schema about to be written.

    Returns:
        The column names dictionary encoding stays on for.

    """
    return [field.name for field in schema if not pa.types.is_floating(field.type)]


def as_table(output: Any) -> pa.Table:
    """Convert an upstream transform's output into the Arrow table a Parquet write needs.

    Args:
        output: A pandas frame, or a NumPy array of one or two dimensions.

    Returns:
        The same values as an Arrow table, carrying no pandas or Arrow schema
        metadata, because our sink writes none.

    """
    if isinstance(output, np.ndarray):
        values: np.ndarray = output if output.ndim == 2 else output.reshape(-1, 1)
        return pa.table({f"c{i}": values[:, i] for i in range(values.shape[1])})
    return pa.Table.from_pandas(output, preserve_index=False).replace_schema_metadata(None)


def sink_layout(parts: Iterable[str | Path]) -> list[tuple[int, int]]:
    """Read back how our sink split a transform's output into parts and row groups.

    The sink writes one part per input row group and each part as one row
    group per batch, and the batch shrinks as the pipeline's rows widen, so the
    layout depends on the transformer. Reading it from the files ours wrote
    keeps the write-back matched when that rule changes; a copy of the rule
    here would drift from the sink's.

    Args:
        parts: The part files a ``transform_parquet`` call returned, in order.

    Returns:
        Each part's row count and the size of its largest row group, which is
        the batch size: every row group but a part's last is a full batch.

    """
    layout: list[tuple[int, int]] = []
    for part in parts:
        metadata = pq.ParquetFile(part).metadata
        largest: int = max((metadata.row_group(i).num_rows for i in range(metadata.num_row_groups)), default=0)
        layout.append((metadata.num_rows, max(largest, 1)))
    return layout


def write_parts(table: pa.Table, directory: Path, layout: Sequence[tuple[int, int]]) -> None:
    """Write an Arrow table the way our C++ sink wrote the same transform's output.

    It writes the parts and row groups ``layout`` describes (see
    ``sink_layout``) with the settings ``PartWriter`` uses: Snappy
    compression, Parquet format 2.6, no stored Arrow schema, and dictionary
    encoding off for floats and on for everything else. Our sink also deletes
    the directory's existing part files when a run starts and writes each part
    under a temporary name and renames it; this does neither, since both cost
    a few file operations per part, not per row.

    Args:
        table: The values to write.
        directory: Directory to write the part files into.
        layout: Each part's row count and row-group size, from ``sink_layout``.

    Raises:
        ValueError: If the layout's parts do not add up to the table's rows.

    """
    covered: int = sum(rows for rows, _ in layout)
    if covered != table.num_rows:
        raise ValueError(f"the layout covers {covered:,} rows, the table has {table.num_rows:,}")
    directory.mkdir(parents=True, exist_ok=True)
    dictionary: list[str] = dictionary_columns(table.schema)
    start: int = 0
    for index, (rows, row_group_rows) in enumerate(layout):
        pq.write_table(
            table.slice(start, rows),
            directory / f"part-{index:05d}.parquet",
            row_group_size=row_group_rows,
            compression="snappy",
            use_dictionary=dictionary,
            version="2.6",
            store_schema=False,
        )
        start += rows


def write_back(output: Any, directory: Path, layout: Sequence[tuple[int, int]]) -> None:
    """Write an upstream transform's output the way our sink wrote ours.

    Args:
        output: A pandas frame, or a NumPy array of one or two dimensions.
        directory: Directory to write the part files into.
        layout: Each part's row count and row-group size, from ``sink_layout``.

    """
    write_parts(as_table(output), directory, layout)
