"""Helpers that many test files share."""

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


def write_parquet(path: Path, table: pa.Table, **kwargs: Any) -> str:
    """Write a Parquet file and return its path as a string.

    Args:
        path: Destination file.
        table: Table to write.
        **kwargs: Passed through to pyarrow's writer.

    Returns:
        The file path.

    """
    pq.write_table(table, path, **kwargs)
    return str(path)


def read_table(parts: list[Path]) -> pa.Table:
    """Read output parts back as one table.

    Args:
        parts: Part files in order.

    Returns:
        The concatenated table.

    """
    return pa.concat_tables([pq.read_table(part) for part in parts])


def read_rows(parts: list[Path]) -> list[dict[str, Any]]:
    """Read output parts back as records.

    Args:
        parts: Part files in order.

    Returns:
        The rows as dicts.

    """
    return read_table(parts).to_pylist()
