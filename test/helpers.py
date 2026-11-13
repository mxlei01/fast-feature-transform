"""Helpers that many test files share."""

import enum
from collections.abc import Callable
from pathlib import Path
from typing import Any, override

import pyarrow as pa
import pyarrow.parquet as pq

# Every name the tests give as a str subclass. An Enum with a str mixin
# formats as "EnumName.member", a StrEnum formats as its value, and
# ShoutedName formats in upper case; each equals the plain string. An
# interpreter started outside test/ cannot import this module, so it does not
# define these classes.
_SUBCLASS_NAMES: tuple[str, ...] = (
    "device",
    "source",
    "price",
    "height",
    "rare",
    "a.b",
    "outer",
    "inner",
    "label",
    "scale",
    "target",
)
EnumName: Any = enum.Enum("EnumName", {name: name for name in _SUBCLASS_NAMES}, type=str)
StrEnumName: Any = enum.StrEnum("StrEnumName", {name: name for name in _SUBCLASS_NAMES})


class ShoutedName(str):
    """A str subclass that formats as its value in upper case."""

    @override
    def __str__(self) -> str:
        return self.upper()


# Each makes a name, given its value, as one kind of str subclass.
NAME_KINDS: list[Callable[[str], Any]] = [EnumName, StrEnumName, ShoutedName]
NAME_KIND_IDS: list[str] = ["enum_with_str_mixin", "str_enum", "str_subclass_with_own_str"]


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
