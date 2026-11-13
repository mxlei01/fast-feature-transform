"""Write another library's output to Parquet the way this package's sink writes its own.

A timing comparison that ends in Parquet has to charge the other library for
the same file: the same compression, encodings, dictionary limit, write buffer,
parts and row groups. ``write_back`` writes a transform's output that way, and
``sink_layout`` reads the parts and row groups to copy from files the sink
wrote.
"""

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

LOW_DICTIONARY_LIMIT: int = 248 * 1024
# The sink writes each part through a buffer of this size (``kPartBufferBytes``
# in ``src/fastfeaturetransform/io.cpp``). pyarrow opens a path without one
# and hands the file every page header and every page body in a write call of
# its own, so the write-back opens its parts through the same buffer.
PART_BUFFER_BYTES: int = 64 * 1024


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


def dictionary_page_limit(schema: pa.Schema) -> int | None:
    """Return the dictionary page limit the sink writes a table of this schema with.

    ``PartWriter`` lowers Arrow's limit of 1 MiB to 248 KiB when every column
    is an integer, a float, a bool, a date, a time, a timestamp, a duration,
    or a dictionary of strings or bytes, and keeps Arrow's limit when any
    column is something else (``suits_low_dictionary_limit`` in
    ``src/fastfeaturetransform/io.cpp``). A matched write-back has to do the
    same: at 1 MiB pyarrow hashes every value of a column of distinct
    integers into a dictionary, work the sink stops at 248 KiB.

    Args:
        schema: The output schema about to be written.

    Returns:
        The limit in bytes, or None to keep Arrow's.

    """

    def suits(kind: pa.DataType) -> bool:
        if pa.types.is_dictionary(kind):
            values: pa.DataType = kind.value_type
            return (
                pa.types.is_string(values)
                or pa.types.is_large_string(values)
                or pa.types.is_binary(values)
                or pa.types.is_large_binary(values)
            )
        return (
            pa.types.is_integer(kind)
            or pa.types.is_floating(kind)
            or pa.types.is_boolean(kind)
            or pa.types.is_date(kind)
            or pa.types.is_time(kind)
            or pa.types.is_timestamp(kind)
            or pa.types.is_duration(kind)
        )

    return LOW_DICTIONARY_LIMIT if all(suits(field.type) for field in schema) else None


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


def sink_layout(parts: Iterable[str | Path]) -> list[list[int]]:
    """Read back how our sink split a transform's output into parts and row groups.

    The sink writes one part per input row group and gathers small batches
    into row groups by their rows, their number and their bytes, and the
    batch shrinks as the pipeline's rows widen, so the layout depends on the
    transformer. Reading every row group's size from the files ours wrote
    keeps the write-back matched when that rule changes; a copy of the rule
    here would drift from the sink's.

    Args:
        parts: The part files a ``transform_parquet`` call returned, in order.

    Returns:
        The row count of every row group of each part, in order.

    """
    layout: list[list[int]] = []
    for part in parts:
        metadata = pq.ParquetFile(part).metadata
        layout.append([metadata.row_group(i).num_rows for i in range(metadata.num_row_groups)])
    return layout


def write_parts(table: pa.Table, directory: Path, layout: Sequence[Sequence[int]]) -> None:
    """Write an Arrow table the way our C++ sink wrote the same transform's output.

    It writes the parts and row groups ``layout`` describes (see
    ``sink_layout``) with the settings ``PartWriter`` uses: Snappy
    compression, Parquet format 2.6, no stored Arrow schema, dictionary
    encoding off for floats and on for everything else, the sink's
    dictionary page limit (see ``dictionary_page_limit``), and the sink's
    write buffer of ``PART_BUFFER_BYTES`` in front of each file. The sink also
    deletes the directory's existing part files when a run starts and writes
    each part under a temporary name and renames it; this does neither, since
    both cost a few file operations per part, not per row.

    Args:
        table: The values to write.
        directory: Directory to write the part files into.
        layout: The row count of every row group of each part, from ``sink_layout``.

    Raises:
        ValueError: If the layout's row groups do not add up to the table's rows.

    """
    covered: int = sum(sum(row_groups) for row_groups in layout)
    if covered != table.num_rows:
        raise ValueError(f"the layout covers {covered:,} rows, the table has {table.num_rows:,}")
    directory.mkdir(parents=True, exist_ok=True)
    dictionary: list[str] = dictionary_columns(table.schema)
    limit: int | None = dictionary_page_limit(table.schema)
    start: int = 0
    for index, row_groups in enumerate(layout):
        with (
            pa.output_stream(directory / f"part-{index:05d}.parquet", buffer_size=PART_BUFFER_BYTES) as part,
            pq.ParquetWriter(
                part,
                table.schema,
                compression="snappy",
                use_dictionary=dictionary,
                dictionary_pagesize_limit=limit,
                version="2.6",
                store_schema=False,
            ) as writer,
        ):
            for rows in row_groups:
                # A part without rows is one row group without rows, as the
                # sink writes it.
                writer.write_table(table.slice(start, rows), row_group_size=max(rows, 1))
                start += rows


def write_back(output: Any, directory: Path, layout: Sequence[Sequence[int]]) -> None:
    """Write an upstream transform's output the way our sink wrote ours.

    Args:
        output: A pandas frame, or a NumPy array of one or two dimensions.
        directory: Directory to write the part files into.
        layout: The row count of every row group of each part, from ``sink_layout``.

    """
    write_parts(as_table(output), directory, layout)
