"""Transformers: Python declarations of the C++ kernels.

Transformers hold kernel parameters only; fit and transform execute in the
C++ core. After fit, fitted state lives in C++ and the properties here are
read-only views of it.
"""

import contextlib
import itertools
import json
import math
import numbers
import operator
import os
import tempfile
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, ClassVar, Self, overload, override

from fast_feature_transform import _core


def _cached_manifest(handle: _core.FittedPipeline) -> str:
    """Return a fitted handle's manifest JSON, serialized once per handle.

    Fitted state never changes, so the string is cached on the handle and
    every pipeline or transformer bound to it pickles the very same object,
    which pickle then stores once.

    Args:
        handle: The fitted C++ pipeline.

    Returns:
        The manifest JSON.

    """
    cached: str | None = getattr(handle, "_manifest_cache", None)
    if cached is None:
        cached = handle.manifest_json()
        setattr(handle, "_manifest_cache", cached)  # noqa: B010
    return cached


def _integer(value: Any, what: str) -> int:
    """Accept an integer from -2**63 to 2**63 - 1, as a plain int.

    An integer is a value that ``operator.index`` takes, such as an int or a
    numpy integer, but not a bool or a numpy bool. The kernels read every
    integer option as an int64, so a larger one is refused here rather than
    read wrapped or refused at fit.

    Args:
        value: The value to check.
        what: What the value is, for the error message.

    Returns:
        The value as an int.

    Raises:
        TypeError: If the value is a bool or not an integer.
        ValueError: If the value is outside -2**63 to 2**63 - 1.

    """
    if isinstance(value, bool):
        raise TypeError(f"{what} must be an integer, got {value!r}")
    try:
        number: int = operator.index(value)
    except TypeError:
        raise TypeError(f"{what} must be an integer, got {value!r}") from None
    if not -(2**63) <= number < 2**63:
        raise ValueError(f"{what} must be an integer from -2**63 to 2**63 - 1, got {value!r}")
    return number


def _real(value: Any, what: str) -> float:
    """Accept a number within the range of f64, as the nearest float.

    A number is an integer, as ``_integer`` defines one, or an instance of
    ``numbers.Real`` that is not a ``numbers.Integral``: a float, a
    ``fractions.Fraction`` or a numpy float. A bool, a numpy bool or a numpy
    timedelta64 is not a number. The kernels read every number option as an
    f64, so a number whose nearest f64 is infinite or 0 though the number is
    neither, such as ``10**400`` or ``Fraction(1, 10**400)``, is outside the
    range of f64 and refused here rather than read as that infinity or 0.

    Args:
        value: The value to check.
        what: What the value is, for the error message.

    Returns:
        The nearest float to the value.

    Raises:
        TypeError: If the value is a bool or not a number.
        ValueError: If the value is outside the range of f64.

    """
    # int and float are named first because a check against numbers.Real
    # takes several times as long for them.
    real: Any = (
        value
        if isinstance(value, (int, float))
        or (isinstance(value, numbers.Real) and not isinstance(value, numbers.Integral))
        else None
    )
    if real is None:
        # Any other integer, such as a numpy integer or a numpy integer array
        # of no dimensions, is read through operator.index, as _integer reads
        # it. numpy registers timedelta64 as a numbers.Integral, and
        # operator.index refuses it.
        with contextlib.suppress(TypeError):
            real = operator.index(value)
    if real is None or isinstance(real, bool):
        raise TypeError(f"{what} must be a number, got {value!r}")
    try:
        number: float = float(real)
    except OverflowError:
        number = math.inf
    if (math.isinf(number) or number == 0.0) and number != real:
        raise ValueError(f"{what} is outside the range of f64, got {value!r}")
    return number


def _source_paths(X: object) -> list[str]:
    """Normalize a batch input to a list of Parquet paths.

    Args:
        X: A list (or tuple) of str or path-like Parquet file paths.

    Returns:
        The paths as strings, in order.

    Raises:
        TypeError: If X is a single path, or holds anything but paths.

    """
    if isinstance(X, (str, os.PathLike)):
        raise TypeError("X must be a list of Parquet paths, not a single path; wrap it in a list")
    if not isinstance(X, (list, tuple)):
        raise TypeError(f"X must be a list of Parquet paths, got {type(X).__name__}")
    paths: list[str] = []
    for item in X:
        if not isinstance(item, (str, os.PathLike)):
            raise TypeError(f"X must be a list of Parquet paths; element {item!r} is a {type(item).__name__}")
        paths.append(os.fspath(item))
    return paths


def _directory_path(path: object, what: str) -> str:
    """Turn a directory argument into the str the engine takes, through os.fspath.

    str() turns None, bytes and an os.PathLike other than pathlib.Path into
    text that names another directory. The operating system ends a path at
    its first NUL character, so a path that holds one names a place other
    than the one the caller wrote, and an empty path names none.

    Args:
        path: The caller's value.
        what: The argument's name, for the error message.

    Returns:
        The path as a str.

    Raises:
        TypeError: If path is neither a str nor an os.PathLike that gives a str.
        ValueError: If path is empty or holds a NUL character.

    """
    text: object = os.fspath(path) if isinstance(path, (str, bytes, os.PathLike)) else None
    if isinstance(text, bytes):
        raise TypeError(f"{what} must be a str path, got bytes; decode it with os.fsdecode")
    if not isinstance(text, str):
        raise TypeError(f"{what} must be a str or os.PathLike, got {type(path).__name__}")
    if not text:
        raise ValueError(f"{what} must not be empty; pass '.' for the current directory")
    if "\0" in text:
        raise ValueError(f"{what} must not hold a NUL character, got {text!r}")
    return text


@contextlib.contextmanager
def _output_directory(output_dir: str | Path | None) -> Iterator[str]:
    """Give a batch call the directory it writes into.

    Args:
        output_dir: The caller's directory, or None for a new temporary one.

    Yields:
        The directory's path. A temporary directory made here is removed
        again if the call raises and leaves it empty.

    Raises:
        TypeError: If output_dir is not None and is neither a str nor an
            os.PathLike that gives a str.
        ValueError: If output_dir is empty or holds a NUL character.

    """
    if output_dir is not None:
        yield _directory_path(output_dir, "output_dir")
        return
    created: str = tempfile.mkdtemp(prefix="fast_feature_transform_")
    try:
        yield created
    except BaseException:
        # rmdir removes the directory only while it is empty. A call that
        # fails in the engine leaves no file in it, and an interrupt raised
        # once the engine has written its output leaves that output in place.
        with contextlib.suppress(OSError):
            os.rmdir(created)
        raise


def _check_threads(threads: int) -> int:
    """Validate a worker-thread count.

    Args:
        threads: Requested worker threads; 0 means all cores.

    Returns:
        The validated count.

    Raises:
        ValueError: If threads is not an integer from 0 to 2**32 - 1.

    """
    message: str = f"threads must be an integer from 0 to {2**32 - 1} (0 uses all cores), got {threads!r}"
    try:
        value: int = _integer(threads, "threads")
    except TypeError, ValueError:
        raise ValueError(message) from None
    if not 0 <= value < 2**32:
        raise ValueError(message)
    return value


def _fit_engine(threads: int, temp_dir: str | Path | None) -> _core.Engine:
    """Construct the native engine with an optional spill directory.

    Args:
        threads: Worker count, or zero for all cores.
        temp_dir: Parent directory for private spill files; None or an empty
            string uses the system temporary directory.

    Returns:
        The native engine.

    Raises:
        TypeError: If temp_dir is not None and is neither a str nor an
            os.PathLike that gives a str.
        ValueError: If threads is invalid, or temp_dir holds a NUL character
            or is not an existing directory. The engine opens
            temp_dir only when the fit first spills, so a typo would
            otherwise pass on small data.

    """
    directory: str = ""
    # An empty string has always meant the system temporary directory, as
    # None does, so it is not refused here as an empty output directory is.
    if temp_dir is not None and temp_dir != "":
        directory = _directory_path(temp_dir, "temp_dir")
        if not os.path.isdir(directory):
            raise ValueError(f"temp_dir must be an existing directory, got {directory!r}")
    return _core.Engine(_check_threads(threads), directory)


def _target(target: object, what: str) -> str:
    """Validate a supervised encoder's target column name.

    Args:
        target: The value passed for a ``target`` argument.
        what: The transformer name, for the error message.

    Returns:
        The target column name, as a plain ``str``.

    Raises:
        TypeError: If target is not a non-empty string.

    """
    if not isinstance(target, str) or not target:
        raise TypeError(f"{what}: target must be a column name, got {target!r}")
    # A name given as a str subclass is stored as the plain string it equals,
    # as _column_list stores a column name.
    return str.__str__(target)


def _target_outside(target: str, columns: list[str] | None, what: str) -> None:
    """Refuse a target that is also one of the columns to encode.

    Args:
        target: The target column name.
        columns: The columns to encode, or None for every utf8 column.
        what: The transformer name, for the error message.

    Raises:
        ValueError: If columns holds the target.

    """
    if columns is not None and target in columns:
        raise ValueError(f"{what}: column {target!r} is also the target; a target encoder cannot encode its target")


def _column_list(columns: object, what: str) -> list[str]:
    """Validate a list of column names.

    Args:
        columns: The value passed for a ``columns`` argument.
        what: The transformer name, for the error message.

    Returns:
        The column names as a new list of plain ``str``.

    Raises:
        TypeError: If columns is a bare string or holds non-strings.
        ValueError: If columns is empty.

    """
    if isinstance(columns, str):
        raise TypeError(f"{what}: columns must be a list of column names, not a single string")
    if not isinstance(columns, (list, tuple)):
        raise TypeError(f"{what}: columns must be a list of column names, got {columns!r}")
    try:
        # str.__str__ refuses anything but a str and returns a plain str
        # unchanged. A str subclass, such as an Enum with a str mixin, comes
        # back as the plain string it equals, which output_columns formats and
        # a pickle stores; the member formats as "ClassName.MEMBER" and
        # pickles as a reference to its class.
        names: list[str] = list(map(str.__str__, columns))
    except TypeError:
        raise TypeError(f"{what}: columns must be a list of column names, got {columns!r}") from None
    if not names:
        raise ValueError(f"{what}: columns must name at least one column")
    return names


def _optional_columns(columns: object, what: str) -> list[str] | None:
    """Validate a ``columns`` argument that can be None.

    Args:
        columns: The value passed for ``columns``; None leaves the choice to the fit.
        what: The transformer name, for the error message.

    Returns:
        The column names, or None.

    Raises:
        TypeError: If columns is a bare string or holds non-strings.
        ValueError: If columns is empty.

    """
    return None if columns is None else _column_list(columns, what)


# A transformer's view of one fit: the fitted C++ pipeline, the transformer's
# dotted stage name in it, and whether the transformer owns that pipeline
# outright (standalone) or is one step of a Pipeline.
_Binding = tuple[_core.FittedPipeline, str, bool]


class Transformer:
    """Base transformer: declares a kernel and its parameters; execution lives in C++.

    Attributes:
        kernel_id: The kernel this transformer compiles to.

    """

    kernel_id: ClassVar[str]

    # None until the first fit. A fit publishes the whole binding in one
    # assignment and each reader takes everything it uses from one read of
    # it, so a thread that uses this transformer while another fits it again
    # sees the old fit or the new one, never the handle of one fit with the
    # stage name or the ownership of the other.
    _binding: _Binding | None = None
    # The fitted state, with the binding it was read from. A read that a
    # refit overtakes stores the old fit's state after the refit cleared the
    # cache; the next read finds that binding replaced and reads again.
    _state_cache: tuple[_Binding, dict[str, Any]] | None = None

    def params(self) -> dict[str, Any]:
        """Kernel parameters, serialized into the plan and the manifest.

        Returns:
            The declarative parameter mapping for this kernel.

        """
        raise NotImplementedError

    def fit(
        self,
        X: Sequence[str | Path],
        y: Any = None,
        *,
        threads: int = 0,
        temp_dir: str | Path | None = None,
    ) -> Self:
        """Fit this transformer standalone by wrapping it in a one-stage pipeline.

        A transformer is a view of one fit at a time: fitting it again, or
        fitting a Pipeline that holds it, rebinds it to the newest fit in one
        step, so a thread that uses it meanwhile sees the old fit or the new
        one.

        Fit reads Parquet in batches. Steps that need every value (exact
        medians, quantiles, k-means and power fits) keep those values within a
        64 MiB budget per fit and move them to temporary files when they
        outgrow it, trading disk space and I/O time for memory. The budget
        covers those values only: fitted tables, batches and Parquet decoder
        buffers take memory too. Nothing is sampled.

        Args:
            X: List of Parquet file paths.
            y: Ignored; present to match scikit-learn's ``fit(X, y)``.
            threads: Worker threads for the engine; 0 uses all cores.
            temp_dir: Existing parent directory for private temporary spill files;
                defaults to the system temporary directory. The files are
                deleted when the fit ends or fails.

        Returns:
            self, fitted.

        Raises:
            EngineError: If the input cannot be read or a column is missing or mistyped.
            TypeError: If X is a single path instead of a list of paths, or
                temp_dir is not None and is neither a str nor an os.PathLike
                that gives a str.
            ValueError: If threads is not an integer from 0 to 2**32 - 1, or
                temp_dir holds a NUL character or is not an existing
                directory.

        """
        from fast_feature_transform.pipeline import Pipeline

        # REGRESSION: a standalone fit binds this transformer once, as
        # standalone. It used to fit a one-step Pipeline, which bound it as a
        # pipeline step first, so a thread that used it before it was marked
        # standalone was refused, and a copy taken then was unfitted.
        stage: str = type(self).__name__.lower()
        plan: str = json.dumps(Pipeline([(stage, self)])._to_tree())
        self._bind(_fit_engine(threads, temp_dir).fit(plan, _source_paths(X)), stage, standalone=True)
        return self

    def transform_dict(self, X: dict[str, Any]) -> dict[str, Any]:
        """Transform a single record through this standalone-fitted transformer.

        Args:
            X: One record as a flat dict of str, int, float, bool, datetime,
                or None values (numpy scalars are converted, and a datetime
                becomes i64 microseconds since 1970-01-01).

        Returns:
            The transformed record.

        Raises:
            EngineError: If a value cannot be converted, a required key is missing, or the record holds one key twice.
            RuntimeError: If unfitted, or fitted as part of a Pipeline.

        """
        # The record path: a record costs a couple of hundred nanoseconds, and
        # the two frames _standalone_handle adds are a large share of that. A
        # transformer that is not ready still goes through it, for the error.
        binding: _Binding | None = self._binding
        if binding is not None and binding[2]:
            return binding[0].transform_row(X)
        return self._standalone_handle().transform_row(X)

    def transform_parquet(self, X: Sequence[str | Path], *, output_dir: str | Path | None = None) -> list[Path]:
        """Transform Parquet files through this standalone-fitted transformer.

        Args:
            X: List of Parquet file paths.
            output_dir: Directory for output part files; a temporary directory
                is created when omitted, and removed again if the run fails.
                Existing ``part-NNNNN.parquet`` files in it are replaced once
                the run starts, and a run that fails midway leaves none.

        Returns:
            The paths of the part files written, one per input row group, in
            source order.

        Raises:
            EngineError: If the input cannot be read, does not match the fitted schema, or
                has rows while the pipeline emits no columns, which Parquet output
                cannot hold; ``transform_dict`` returns them.
            RuntimeError: If unfitted, or fitted as part of a Pipeline.
            TypeError: If X is a single path instead of a list of paths, or
                output_dir is not None and is neither a str nor an os.PathLike
                that gives a str.
            ValueError: If output_dir is empty or holds a NUL character.

        """
        handle: _core.FittedPipeline = self._standalone_handle()
        with _output_directory(output_dir) as out_dir:
            return [Path(path) for path in handle.transform_batch(_source_paths(X), out_dir)]

    def transform_memmap(self, X: Sequence[str | Path], *, output_dir: str | Path) -> Path:
        """Transform Parquet files into one memmap: ``data.mmap`` plus ``index.json``.

        Args:
            X: Input Parquet files.
            output_dir: Directory for ``data.mmap`` and ``index.json``.

        Returns:
            The path of ``index.json``. The memmap has the layout that
            parquet_to_memmap writes.

        Raises:
            EngineError: If the input cannot be read or does not match the
                fitted schema, or the output has no columns or holds a string
                column, a null, a list column whose rows differ in length or
                are empty, or a type a memmap cannot store.
            RuntimeError: If this transformer does not own a fitted pipeline.
            TypeError: If X is a single path instead of a list of paths, or
                output_dir is neither a str nor an os.PathLike that gives a
                str.
            ValueError: If output_dir is empty or holds a NUL character.

        """
        return Path(
            self._standalone_handle().transform_memmap(_source_paths(X), _directory_path(output_dir, "output_dir"))
        )

    @overload
    def transform(self, X: dict[str, Any]) -> dict[str, Any]: ...

    @overload
    def transform(self, X: Sequence[str | Path], *, output_dir: str | Path | None = None) -> list[Path]: ...

    def transform(
        self, X: dict[str, Any] | Sequence[str | Path], *, output_dir: str | Path | None = None
    ) -> dict[str, Any] | list[Path]:
        """Transform a record dict or a list of Parquet paths through this transformer.

        Dispatches on input type: a dict goes to transform_dict, and a list or
        tuple of paths to transform_parquet.

        Args:
            X: A single record dict, or a list of Parquet file paths.
            output_dir: Directory for batch output part files; a temporary
                directory is created when omitted.

        Returns:
            A transformed record dict, or the list of written part-file paths.

        Raises:
            EngineError: If the engine rejects the input.
            RuntimeError: If unfitted, or fitted as part of a Pipeline.
            TypeError: If X is neither a dict nor a list of paths, or
                output_dir is not None and is neither a str nor an os.PathLike
                that gives a str.
            ValueError: If output_dir is empty or holds a NUL character.

        """
        if isinstance(X, dict):
            return self.transform_dict(X)
        if isinstance(X, (list, tuple)):
            return self.transform_parquet(X, output_dir=output_dir)
        raise TypeError(f"unsupported input type: {type(X).__name__}; expected a dict or a list of Parquet paths")

    def __repr__(self) -> str:
        """Render the transformer as its constructor call.

        Returns:
            ``ClassName(param=value, ...)``.

        """
        arguments: str = ", ".join(f"{key}={value!r}" for key, value in self.params().items())
        return f"{type(self).__name__}({arguments})"

    def _standalone_handle(self) -> _core.FittedPipeline:
        """Return the handle when this transformer owns it exclusively.

        Returns:
            The fitted C++ pipeline handle.

        Raises:
            RuntimeError: If unfitted, or fitted as part of a Pipeline.

        """
        handle, _, standalone = self._require_binding()
        if not standalone:
            raise RuntimeError(
                f"{type(self).__name__} was fitted as part of a Pipeline; call transform on that pipeline"
            )
        return handle

    def __getstate__(self) -> dict[str, Any]:
        """Pickle the transformer, carrying fitted state only when it owns its handle.

        A standalone-fitted transformer embeds its manifest so the round trip
        stays fitted; a pipeline-bound one pickles unfitted (its owning
        Pipeline's pickle carries the manifest and rebinds on restore).

        Returns:
            The instance dict without the C++ handle, plus the manifest when
            this transformer owns its handle.

        """
        transient: tuple[str, ...] = ("_binding", "_state_cache")
        state: dict[str, Any] = {key: value for key, value in self.__dict__.items() if key not in transient}
        binding: _Binding | None = self._binding
        if binding is not None and binding[2]:
            state["_manifest"] = _cached_manifest(binding[0])
            state["_stage_name"] = binding[1]
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore the transformer, reloading fitted state when the pickle carried it.

        Args:
            state: State dict produced by __getstate__.

        """
        manifest: str | None = state.pop("_manifest", None)
        stage: str | None = state.pop("_stage_name", None)
        self.__dict__.update(state)
        if manifest is not None and stage is not None:
            self._bind(_core.FittedPipeline.load_json(manifest), stage, standalone=True)

    def _bind(self, handle: _core.FittedPipeline, stage: str, *, standalone: bool) -> None:
        """Attach the fitted C++ pipeline handle this transformer belongs to.

        Args:
            handle: The fitted C++ pipeline.
            stage: This transformer's dotted stage name in the plan.
            standalone: Whether this transformer owns the handle outright.

        """
        self._binding = (handle, stage, standalone)
        self._state_cache = None  # releases the old fit's state

    def _require_binding(self) -> _Binding:
        """Return the binding, read once, or raise if this transformer is unfitted.

        Returns:
            The fitted C++ pipeline, this transformer's stage name in it, and
            whether this transformer owns it outright.

        Raises:
            RuntimeError: If the transformer has not been fitted.

        """
        binding: _Binding | None = self._binding
        if binding is None:
            raise RuntimeError(f"{type(self).__name__} is not fitted")
        return binding

    def _require_handle(self) -> _core.FittedPipeline:
        """Return the fitted handle or raise if this transformer is unfitted.

        Returns:
            The fitted C++ pipeline handle.

        Raises:
            RuntimeError: If the transformer has not been fitted.

        """
        return self._require_binding()[0]

    def _state(self) -> dict[str, Any]:
        """Read this transformer's fitted state from the C++ core (once per binding).

        Returns:
            The fitted state as a plain dict.

        """
        cached: tuple[_Binding, dict[str, Any]] | None = self._state_cache
        if cached is not None and cached[0] is self._binding:
            return cached[1]
        binding: _Binding = self._require_binding()
        handle, stage, _ = binding
        state: dict[str, Any] = json.loads(handle.stage_state(stage))
        self._state_cache = (binding, state)
        return state


class Select(Transformer):
    """Column projection: keep exactly the given columns, in order."""

    kernel_id: ClassVar[str] = "select"

    def __init__(self, columns: list[str]) -> None:
        """Initialize the projection.

        Args:
            columns: Columns to keep, in output order.

        Raises:
            TypeError: If columns is a bare string or holds non-strings.
            ValueError: If columns is empty.

        """
        self.columns: list[str] = _column_list(columns, "Select")

    @override
    def params(self) -> dict[str, Any]:
        return {"columns": self.columns}


class LabelEncoder(Transformer):
    """Categorical utf8 -> i64 index; index 0 is reserved for unknown.

    Fit counts each category over Parquet batches, with memory proportional
    to the observed vocabulary. Known categories sort by byte order and take
    indices 1..n; unknown categories, nulls, and categories below
    ``min_count`` map to 0 (the ``torch.nn.Embedding(padding_idx=0)``
    convention). Both string layouts (``string`` and ``large_string``) are
    supported; categories must be valid UTF-8.
    """

    kernel_id: ClassVar[str] = "label_encode"

    def __init__(self, columns: list[str] | None = None, min_count: int = 1) -> None:
        """Initialize the encoder.

        Args:
            columns: Columns to encode; None encodes every utf8 column found
                at fit.
            min_count: Categories seen fewer than this many times are dropped
                (they encode to 0).

        Raises:
            TypeError: If columns is a bare string or min_count is not an integer.
            ValueError: If columns is empty, or min_count is below 1 or above
                2**63 - 1.

        """
        self.columns: list[str] | None = _optional_columns(columns, "LabelEncoder")
        self.min_count: int = _integer(min_count, "LabelEncoder: min_count")
        if self.min_count < 1:
            raise ValueError(f"LabelEncoder: min_count must be >= 1, got {min_count}")

    @override
    def params(self) -> dict[str, Any]:
        return {"columns": self.columns, "min_count": self.min_count}

    @property
    def label_maps_(self) -> dict[str, dict[str, int]]:
        """Fitted category -> index maps, materialized from the C++ core.

        Returns:
            Mapping of column name to its category -> index map.

        """
        return self._state()["label_maps"]

    def vocab_size(self, column: str) -> int:
        """Number of embedding rows needed for a column (len(map) + 1 for unknown).

        Args:
            column: The encoded column name.

        Returns:
            The vocabulary size including the reserved unknown index 0.

        Raises:
            KeyError: If this encoder does not encode the column.
            RuntimeError: If the encoder is not fitted.

        """
        handle, stage, _ = self._require_binding()
        if not isinstance(column, str):
            # Preserve Python dict key semantics for unusual invalid arguments.
            return self._state()["vocab_sizes"][column]
        try:
            return handle.vocab_size(stage, column)
        except _core.EngineError:
            # A column this encoder does not encode raises KeyError, as its
            # docstring says, from a lookup in the materialized state.
            return self._state()["vocab_sizes"][column]


TYPE_TAGS: frozenset[str] = frozenset({"utf8", "i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64", "f32", "f64"})


class Cast(Transformer):
    """Explicit per-column type conversion.

    Apart from reading datetimes as epoch microseconds, the engine never
    converts a value on its own, so a column whose type does not suit a
    downstream step is converted here, visibly, from a closed table: integers
    to any integer width and floats to any float width (an out-of-range value
    is an error, never a wrap); integers to strings as decimal digits; strings
    to integers or floats, parsed strictly; integers to floats only up to
    2^53 in magnitude for f64 and 2^24 for f32, where every integer is exact;
    floats to integers only when whole and in range; bools to integers.
    Floats are never cast to strings, because ``1`` and ``1.0`` have no
    single spelling. Nulls pass through. On the record path the value's own
    type is the source; a conversion outside the table is an error naming
    the key.
    """

    kernel_id: ClassVar[str] = "cast"

    def __init__(self, types: dict[str, str]) -> None:
        """Initialize the cast.

        Args:
            types: Mapping of column name to target type tag, one of
                ``utf8``, ``i8``, ``i16``, ``i32``, ``i64``, ``u8``, ``u16``,
                ``u32``, ``u64``, ``f32``, ``f64``.

        Raises:
            TypeError: If types is not a non-empty mapping of names to tags.
            ValueError: If a tag is not one of the supported targets.

        """
        if not isinstance(types, dict) or not types:
            raise TypeError("Cast: types must be a non-empty mapping of column name to type tag")
        plain: bool = True
        for column, tag in types.items():
            if type(column) is not str or type(tag) is not str:
                if not isinstance(column, str) or not isinstance(tag, str):
                    raise TypeError(f"Cast: types must map column names to type tags, got {column!r}: {tag!r}")
                plain = False
            if tag not in TYPE_TAGS:
                raise ValueError(
                    f"Cast: unsupported type {tag!r} for column {column!r}; choose one of {sorted(TYPE_TAGS)}"
                )
        # A name or tag given as a str subclass, such as an Enum with a str
        # mixin, is stored as the plain string it equals, as _column_list
        # stores a name and _one_of an option. A mapping of plain strings is
        # copied whole rather than pair by pair, which costs a call per pair.
        self.types: dict[str, str] = (
            dict(types) if plain else {str.__str__(column): str.__str__(tag) for column, tag in types.items()}
        )

    @override
    def params(self) -> dict[str, Any]:
        return {"types": self.types}


def _one_of(value: object, allowed: frozenset[str] | set[str], what: str, option: str) -> str:
    """Validate a string option against the values a kernel accepts.

    Args:
        value: The value passed for the option.
        allowed: The accepted values.
        what: The transformer name, for the error message.
        option: The option name, for the error message.

    Returns:
        The value, as a plain ``str``.

    Raises:
        ValueError: If the value is not one of the accepted ones.

    """
    if value not in allowed:
        raise ValueError(f"{what}: {option} must be one of {sorted(allowed)}, got {value!r}")
    # A str subclass, such as an Enum with a str mixin, equals an accepted
    # value and prints as something else. The kernel takes the value itself.
    return str.__str__(value) if isinstance(value, str) else str(value)


_HANDLE_UNKNOWN: frozenset[str] = frozenset({"value", "return_nan", "error"})
_HANDLE_MISSING: frozenset[str] = frozenset({"value", "return_nan", "error"})

AGGREGATES: frozenset[str] = frozenset({"count", "sum", "mean", "min", "max", "median"})


class GroupAgg(Transformer):
    """Per-key aggregates, fitted once over the training data and looked up at transform.

    ``GroupAgg(key="user_id", aggs={"price": ["mean", "max"], "clicks": ["sum"]})``
    streams the data once, keeping the requested aggregates of each column
    for every key; transform then replaces each aggregated column with one
    column per aggregate, named ``<column>_<agg>`` and appended after the
    remaining columns, looked up by the row's key (the key column stays). The
    key is a utf8 or integer column. ``count`` counts a column's non-null
    values and applies to any column; ``sum``, ``mean``, ``min``, ``max``,
    and ``median`` need an integer or float column and skip its nulls. Types
    stay explicit: ``count`` is i64, ``sum`` is i64 for integer columns (an
    overflow is an error) and f64 for floats, ``mean`` and ``median`` are
    f64, and ``min`` and ``max`` keep the column's type. An unknown or null
    key, and a key whose values were all null, yield 0 for every aggregate,
    like LabelEncoder's index 0. The streaming aggregates keep memory
    proportional to the number of keys; ``median`` keeps every value per key
    and moves them to temporary files when they outgrow the fit's memory
    budget.
    """

    kernel_id: ClassVar[str] = "group_agg"

    def __init__(self, key: str, aggs: dict[str, list[str] | tuple[str, ...]]) -> None:
        """Initialize the aggregation.

        Args:
            key: The column to group by.
            aggs: Mapping of column name to the aggregates to compute for it,
                each one of ``count``, ``sum``, ``mean``, ``min``, ``max``,
                ``median``.

        Raises:
            TypeError: If key is not a column name, or aggs is not a non-empty
                mapping of column names to non-empty lists.
            ValueError: If an aggregate is unknown or listed twice for a column.

        """
        if not isinstance(key, str) or not key:
            raise TypeError(f"GroupAgg: key must be a column name, got {key!r}")
        if not isinstance(aggs, dict) or not aggs:
            raise TypeError("GroupAgg: aggs must be a non-empty mapping of column name to a list of aggregates")
        validated: dict[str, list[str]] = {}
        for column, names in aggs.items():
            if not isinstance(column, str) or not column:
                raise TypeError(f"GroupAgg: aggs keys must be column names, got {column!r}")
            if isinstance(names, str) or not isinstance(names, (list, tuple)) or not names:
                raise TypeError(f"GroupAgg: aggregates for column {column!r} must be a non-empty list, got {names!r}")
            for name in names:
                if not isinstance(name, str) or name not in AGGREGATES:
                    raise ValueError(
                        f"GroupAgg: unknown aggregate {name!r} for column {column!r}; choose from {sorted(AGGREGATES)}"
                    )
            if len(set(names)) != len(names):
                raise ValueError(f"GroupAgg: column {column!r} lists an aggregate more than once")
            # A name or aggregate given as a str subclass, such as an Enum
            # with a str mixin, is stored as the plain string it equals.
            validated[str.__str__(column)] = [str.__str__(name) for name in names]
        self.key: str = str.__str__(key)
        self.aggs: dict[str, list[str]] = validated

    @override
    def params(self) -> dict[str, Any]:
        return {"key": self.key, "aggs": self.aggs}

    @property
    def keys_(self) -> list[Any]:
        """Fitted keys, sorted (strings or integers, matching the key column).

        Returns:
            The keys in table order.

        """
        return self._state()["keys"]

    @property
    def values_(self) -> dict[str, list[Any]]:
        """Fitted aggregates: output column name -> one value per key, in ``keys_`` order.

        Returns:
            Mapping of ``<column>_<agg>`` to its values.

        """
        return self._state()["values"]


def _group_size(value: object) -> float:
    """Validate CountEncoder's min_group_size: an integer count or a share of rows.

    An integer, as ``_integer`` defines one, is a count. Any other number,
    as ``_real`` defines one, is a share of the fitted rows.

    Args:
        value: The value passed for ``min_group_size``.

    Returns:
        The count as an int, or the share as a float.

    Raises:
        TypeError: If the value is a bool or not a number.
        ValueError: If the value is not finite, not above 0, an integer above
            2**63 - 1, or outside the range of f64.

    """
    size: float
    try:
        size = _integer(value, "CountEncoder: min_group_size")
    except TypeError:
        size = _real(value, "CountEncoder: min_group_size")
    if not math.isfinite(size) or size <= 0:
        raise ValueError(f"CountEncoder: min_group_size must be a finite number > 0, got {value!r}")
    return size


class CountEncoder(Transformer):
    """Categorical utf8 -> how often that category was seen at fit.

    The simplest of the category_encoders family and the one whose fitted
    table is a plain tally, so it is exact and never depends on how the rows
    were split across workers. ``normalize=True`` divides each count by the
    fitted row count, making the output f64 instead of i64. A null is a
    countable category under ``handle_missing="value"``, a null under
    ``"return_nan"``, and an error under ``"error"`` — refused at transform
    as well as at fit, as category_encoders does. An unseen category yields 0
    under ``handle_unknown="value"`` (a null when ``handle_missing`` is
    ``"return_nan"``), that integer when an integer is given, a null under
    ``"return_nan"``, and an error under ``"error"``; an integer is the
    literal number you asked to see, so ``normalize=True`` scales a real
    count and leaves that number alone. A ``min_group_size`` group whose name
    is already a category that survived the fold is a fit-time error rather
    than a silent overwrite.
    """

    kernel_id: ClassVar[str] = "count_encode"

    def __init__(
        self,
        columns: list[str] | None = None,
        *,
        normalize: bool = False,
        min_group_size: float | None = None,
        min_group_name: str | None = None,
        combine_min_nan_groups: bool | str = True,
        handle_unknown: str | int = "value",
        handle_missing: str = "value",
    ) -> None:
        """Initialize the encoder.

        Args:
            columns: Columns to encode; None encodes every utf8 column found
                at fit.
            normalize: Divide each count by the fitted row count.
            min_group_size: Fold the categories below this threshold into one
                group whose count is their sum, when that group gets two or
                more members and some non-null category stays out of it. An
                integer is a count, and any other number is a fraction of the
                fitted rows.
            min_group_name: Name for that group; the members joined with "_"
                in sorted order when omitted.
            combine_min_nan_groups: Whether the null category joins the folded
                group when its count is below min_group_size too; "force"
                folds it whenever there were nulls, and False never does.
            handle_unknown: What an unseen category becomes: "value" (0, or a
                null when handle_missing is "return_nan"), "return_nan",
                "error", or an integer to fill with.
            handle_missing: What a null becomes: "value" (a category of its
                own), "return_nan", or "error" (refused at transform as well
                as at fit).

        Raises:
            TypeError: If columns is a bare string or an option has the wrong type.
            ValueError: If an option is not one of the accepted values, an
                integer handle_unknown or min_group_size is outside -2**63 to
                2**63 - 1, min_group_size is not a finite number above 0 or is
                outside the range of f64, or combine_min_nan_groups is "force"
                with handle_missing "return_nan".

        """
        self.columns: list[str] | None = _optional_columns(columns, "CountEncoder")
        _flag(normalize, "CountEncoder", "normalize")
        size: float | None = None if min_group_size is None else _group_size(min_group_size)
        if min_group_name is not None and not isinstance(min_group_name, str):
            raise TypeError(f"CountEncoder: min_group_name must be a string, got {min_group_name!r}")
        if not isinstance(combine_min_nan_groups, bool) and combine_min_nan_groups != "force":
            raise ValueError(
                f"CountEncoder: combine_min_nan_groups must be True, False or 'force', got {combine_min_nan_groups!r}"
            )
        unknown: str | int
        if isinstance(handle_unknown, str) and handle_unknown in _HANDLE_UNKNOWN:
            unknown = _one_of(handle_unknown, _HANDLE_UNKNOWN, "CountEncoder", "handle_unknown")
        else:
            try:
                unknown = _integer(handle_unknown, "CountEncoder: handle_unknown")
            except TypeError:
                raise ValueError(
                    f"CountEncoder: handle_unknown must be an integer or one of {sorted(_HANDLE_UNKNOWN)}, "
                    f"got {handle_unknown!r}"
                ) from None
        handle_missing = _one_of(handle_missing, _HANDLE_MISSING, "CountEncoder", "handle_missing")
        if combine_min_nan_groups == "force" and handle_missing == "return_nan":
            raise ValueError("CountEncoder: combine_min_nan_groups='force' needs handle_missing 'value' or 'error'")
        self.normalize: bool = normalize
        self.min_group_size: float | None = size
        self.min_group_name: str | None = None if min_group_name is None else str.__str__(min_group_name)
        self.combine_min_nan_groups: bool | str = (
            combine_min_nan_groups if isinstance(combine_min_nan_groups, bool) else "force"
        )
        self.handle_unknown: str | int = unknown
        self.handle_missing: str = handle_missing

    @override
    def params(self) -> dict[str, Any]:
        return {
            "columns": self.columns,
            "normalize": self.normalize,
            "min_group_size": self.min_group_size,
            "min_group_name": self.min_group_name,
            "combine_min_nan_groups": self.combine_min_nan_groups,
            "handle_unknown": self.handle_unknown,
            "handle_missing": self.handle_missing,
        }

    @property
    def counts_(self) -> dict[str, dict[str, int]]:
        """Fitted category -> count tables, materialized from the C++ core.

        Returns:
            Mapping of column name to its category -> count table.

        """
        return self._state()["counts"]


class OrdinalEncoder(Transformer):
    """Categorical utf8 -> an integer code, in place.

    Categories sort and take the codes ``index_start`` through
    ``index_start + n - 1``. Under ``handle_missing="value"``, a null seen at
    fit is a category of its own and takes the code after them, and a null
    that was not seen at fit encodes to ``-2``, as category_encoders does. An
    unseen category encodes to ``-1`` under ``handle_unknown="value"``, a
    null under ``"return_nan"``, and an error under ``"error"``. An
    ``index_start`` so large that the highest code would not fit in an i64 is
    a fit-time error, never a wrap into a negative embedding index.
    """

    kernel_id: ClassVar[str] = "ordinal_encode"

    def __init__(
        self,
        columns: list[str] | None = None,
        *,
        index_start: int = 1,
        handle_unknown: str = "value",
        handle_missing: str = "value",
    ) -> None:
        """Initialize the encoder.

        Args:
            columns: Columns to encode; None encodes every utf8 column found
                at fit.
            index_start: The code the first category takes.
            handle_unknown: What an unseen category becomes: "value" (-1),
                "return_nan", or "error".
            handle_missing: What a null becomes: "value" (a category of its
                own), "return_nan", or "error" (refused at transform as well
                as at fit).

        Raises:
            TypeError: If columns is a bare string or index_start is not an integer.
            ValueError: If index_start is outside -2**63 to 2**63 - 1, or an
                option is not one of the accepted values.

        """
        self.columns: list[str] | None = _optional_columns(columns, "OrdinalEncoder")
        self.index_start: int = _integer(index_start, "OrdinalEncoder: index_start")
        self.handle_unknown: str = _one_of(handle_unknown, _HANDLE_UNKNOWN, "OrdinalEncoder", "handle_unknown")
        self.handle_missing: str = _one_of(handle_missing, _HANDLE_MISSING, "OrdinalEncoder", "handle_missing")

    @override
    def params(self) -> dict[str, Any]:
        return {
            "columns": self.columns,
            "index_start": self.index_start,
            "handle_unknown": self.handle_unknown,
            "handle_missing": self.handle_missing,
        }

    @property
    def categories_(self) -> dict[str, list[str]]:
        """Fitted categories per column, sorted.

        Returns:
            Mapping of column name to its sorted category list.

        """
        return self._state()["categories"]


class _Expanding(Transformer):
    """Shared base for the encoders that replace a column with one column per category."""

    def __init__(
        self,
        columns: list[str] | None = None,
        *,
        use_cat_names: bool = False,
        handle_unknown: str = "value",
        handle_missing: str = "value",
    ) -> None:
        """Initialize the encoder.

        Args:
            columns: Columns to encode; None encodes every utf8 column found
                at fit.
            use_cat_names: Name the output columns after the categories
                rather than after the codes.
            handle_unknown: What an unseen category becomes: "value" (all
                zeros), "return_nan", "error", or "indicator" for
                OneHotEncoder, which adds a ``<column>_-1`` column.
            handle_missing: What a null becomes: "value" (a category of its
                own), "return_nan", or "error" (refused at transform as well
                as at fit).

        Raises:
            TypeError: If columns is a bare string or use_cat_names is not a bool.
            ValueError: If an option is not one of the accepted values.

        """
        what: str = type(self).__name__
        allowed: frozenset[str] = _HANDLE_UNKNOWN | ({"indicator"} if self.kernel_id == "one_hot_encode" else set())
        self.columns: list[str] | None = _optional_columns(columns, what)
        self.use_cat_names: bool = _flag(use_cat_names, what, "use_cat_names")
        self.handle_unknown: str = _one_of(handle_unknown, allowed, what, "handle_unknown")
        self.handle_missing: str = _one_of(handle_missing, _HANDLE_MISSING, what, "handle_missing")

    @override
    def params(self) -> dict[str, Any]:
        return {
            "columns": self.columns,
            "use_cat_names": self.use_cat_names,
            "handle_unknown": self.handle_unknown,
            "handle_missing": self.handle_missing,
        }

    @property
    def categories_(self) -> dict[str, list[str]]:
        """Fitted categories per column, sorted.

        Returns:
            Mapping of column name to its sorted category list.

        """
        return self._state()["categories"]


class OneHotEncoder(_Expanding):
    """Categorical utf8 -> one 0/1 column per fitted category, replacing the source column.

    Columns are named ``<column>_<code>`` with the code the encoder would
    have assigned, or ``<column>_<category>`` with ``use_cat_names=True``; a
    name that would repeat gets ``#`` appended, counted across every encoded
    column and not only within one, as category_encoders counts it. A
    null seen at fit under ``handle_missing="value"`` is a category of its
    own, sorted last, and gets its own column. An unseen category is all
    zeros under ``handle_unknown="value"``, all nulls under
    ``"return_nan"``, an error under ``"error"``, and a 1 in an extra
    ``<column>_-1`` column under ``"indicator"``. Under
    ``handle_missing="value"``, a null that was never a fitted category is
    all zeros whatever ``handle_unknown`` says. How many columns there are is
    known only after the fit, so a later step in the same pipeline cannot
    name them, and one given no columns does not take them. The generated
    block is appended after the columns that stay rather than substituted in
    the encoded column's place.
    """

    kernel_id: ClassVar[str] = "one_hot_encode"


class RankHotEncoder(_Expanding):
    """Categorical utf8 -> thermometer coding: a 1 in every column up to the category's rank.

    The same layout as OneHotEncoder, but a category at rank ``r`` lights up
    every column through ``r`` rather than only its own, which suits an
    ordered category. ``handle_unknown="indicator"`` is not available here,
    matching category_encoders. Two things also follow category_encoders and
    differ from OneHotEncoder: under ``handle_missing="value"``, a null that
    was never a fitted category takes ``handle_unknown``'s answer rather than
    always being zeros (all nulls under ``"return_nan"``, all zeros
    otherwise, never an error), and ``handle_missing="return_nan"`` emits one
    trailing ``<column>_-2`` column (``<column>_nan`` with
    ``use_cat_names=True``) that nothing ever lights up, for the slot
    category_encoders' ordinal pre-encoder keeps there.
    """

    kernel_id: ClassVar[str] = "rank_hot_encode"


_POSITIONAL_UNKNOWN: frozenset[str] = _HANDLE_UNKNOWN | {"indicator"}


class _Positional(Transformer):
    """Shared base for the encoders that spell a category's rank out digit by digit."""

    def __init__(
        self,
        columns: list[str] | None = None,
        *,
        handle_unknown: str = "value",
        handle_missing: str = "value",
    ) -> None:
        """Initialize the encoder.

        Args:
            columns: Columns to encode; None encodes every utf8 column found
                at fit.
            handle_unknown: What an unseen category becomes: "value" (the
                all-zero code word), "return_nan", "error", or "indicator"
                (an extra level's binary code word, which under GrayEncoder
                can equal a category's, as in category_encoders).
            handle_missing: What a null becomes: "value" (a level of its
                own), "return_nan", or "error" (refused at transform as well
                as at fit).

        Raises:
            TypeError: If columns is a bare string or holds non-strings.
            ValueError: If an option is not one of the accepted values.

        """
        what: str = type(self).__name__
        self.columns: list[str] | None = _optional_columns(columns, what)
        self.handle_unknown: str = _one_of(handle_unknown, _POSITIONAL_UNKNOWN, what, "handle_unknown")
        self.handle_missing: str = _one_of(handle_missing, _HANDLE_MISSING, what, "handle_missing")

    @override
    def params(self) -> dict[str, Any]:
        return {
            "columns": self.columns,
            "handle_unknown": self.handle_unknown,
            "handle_missing": self.handle_missing,
        }

    @property
    def categories_(self) -> dict[str, list[str]]:
        """Fitted categories per column, sorted.

        Returns:
            Mapping of column name to its sorted category list.

        """
        return self._state()["categories"]


class BaseNEncoder(_Positional):
    """Categorical utf8 -> the base-N spelling of the category's rank, one column per digit.

    The category at 1-based rank ``r`` among the sorted levels takes the
    ``digits``-wide base-``base`` representation of ``r``, most significant
    digit first, in columns ``<column>_0`` through ``<column>_{digits - 1}``,
    which are appended after the columns that stay; the source column is
    dropped. ``digits`` is ``ceil(log(levels + 1, base))``, computed with
    integer arithmetic, so the code word ``0`` stays free for an unknown.

    ``base=1`` is category_encoders' documented special case: a rank becomes
    a single 1 at position ``r`` of ``levels + 1`` columns, which is one-hot
    with a leading column that is always zero. ``base=2`` is BinaryEncoder,
    and a base above the number of levels writes the rank itself in a single
    column, which is ordinal encoding.

    How many columns the encoder produces is known only after the fit, so a
    later step in the same pipeline cannot name them, and one given no columns
    does not take them.
    """

    kernel_id: ClassVar[str] = "basen_encode"

    def __init__(
        self,
        columns: list[str] | None = None,
        *,
        base: int = 2,
        handle_unknown: str = "value",
        handle_missing: str = "value",
    ) -> None:
        """Initialize the encoder.

        Args:
            columns: Columns to encode; None encodes every utf8 column found
                at fit.
            base: The base the rank is written in; 1 is the one-hot special
                case, 2 is binary.
            handle_unknown: What an unseen category becomes: "value" (the
                all-zero code word), "return_nan", "error", or "indicator",
                which gives it a code word of its own.
            handle_missing: What a null becomes: "value" (a level of its
                own), "return_nan", or "error" (refused at transform as well
                as at fit).

        Raises:
            TypeError: If columns is a bare string or base is not an integer.
            ValueError: If base is below 1 or above 2**63 - 1, or an option is
                not one of the accepted values.

        """
        super().__init__(columns, handle_unknown=handle_unknown, handle_missing=handle_missing)
        self.base: int = _integer(base, "BaseNEncoder: base")
        if self.base < 1:
            raise ValueError(f"BaseNEncoder: base must be 1 or more, got {base!r}")

    @override
    def params(self) -> dict[str, Any]:
        return {**super().params(), "base": self.base}


class BinaryEncoder(_Positional):
    """Categorical utf8 -> the binary spelling of the category's rank, one column per bit.

    BaseNEncoder with the base fixed at 2, as category_encoders fixes it:
    four levels give three columns and the code words ``001``, ``010``,
    ``011`` and ``100``. There is no ``base`` parameter.
    """

    kernel_id: ClassVar[str] = "binary_encode"


class GrayEncoder(_Positional):
    """Categorical utf8 -> the reflected binary (Gray) spelling of the category's rank.

    The same width as BinaryEncoder, but the category at rank ``r`` takes the
    binary expansion of ``r ^ (r >> 1)``, so consecutive ranks differ in
    exactly one column, which matters only for an ordered category. Five
    levels give the code words ``001``, ``011``, ``010``, ``110`` and
    ``111``.

    category_encoders sorts the categories for this encoder, as it does for
    RankHotEncoder, so unlike BaseNEncoder and BinaryEncoder this one matches
    category_encoders on any fit data, not only data whose categories first
    appear in sorted order.
    """

    kernel_id: ClassVar[str] = "gray_encode"


class _Contrast(Transformer):
    """Shared base for the four contrast coders, which differ only in their matrix."""

    def __init__(
        self,
        columns: list[str] | None = None,
        *,
        handle_unknown: str = "value",
        handle_missing: str = "value",
    ) -> None:
        """Initialize the encoder.

        Args:
            columns: Columns to encode; None encodes every utf8 column found
                at fit.
            handle_unknown: What an unseen category becomes: "value" (a row of
                zeros), "return_nan", "error", or "indicator", which makes it
                a real extra level and so adds one more output column (a
                column with a single level still emits none, as in
                category_encoders).
            handle_missing: What a null becomes: "value" (a level of its own,
                sorted last), "return_nan", or "error" (refused at transform
                as well as at fit). The last two still keep a placeholder
                level for the missing value that no value can reach, as
                category_encoders does, so they emit one column more than
                the categories alone would.

        Raises:
            TypeError: If columns is a bare string or holds non-strings.
            ValueError: If an option is not one of the accepted values.

        """
        what: str = type(self).__name__
        self.columns: list[str] | None = _optional_columns(columns, what)
        self.handle_unknown: str = _one_of(handle_unknown, _HANDLE_UNKNOWN | {"indicator"}, what, "handle_unknown")
        self.handle_missing: str = _one_of(handle_missing, _HANDLE_MISSING, what, "handle_missing")

    @override
    def params(self) -> dict[str, Any]:
        return {
            "columns": self.columns,
            "handle_unknown": self.handle_unknown,
            "handle_missing": self.handle_missing,
        }

    @property
    def categories_(self) -> dict[str, list[str]]:
        """Fitted categories per column, sorted.

        Returns:
            Mapping of column name to its sorted category list.

        """
        return self._state()["categories"]


class BackwardDifferenceEncoder(_Contrast):
    """Categorical utf8 -> backward difference contrast coding, in f64 columns.

    A column of n fitted levels becomes n - 1 columns named ``<column>_0``
    through ``<column>_{n-2}``, and the level at 0-based index i emits row i
    of the n x (n - 1) matrix ``M[i][j] = (j + 1 - n) / n`` when ``i <= j``
    and ``(j + 1) / n`` otherwise: column j contrasts level j + 1 against the
    level before it. There is no intercept column, so a column with a single
    level emits nothing at all — unless ``handle_missing`` is ``"return_nan"``
    or ``"error"``, whose placeholder level gives it a second one to contrast
    against. How many columns there are is known only after the fit, so a
    later step in the same pipeline cannot name them, and one given no columns
    does not take them. The generated block is appended after the columns that
    stay rather than substituted in the encoded column's place.
    """

    kernel_id: ClassVar[str] = "backward_difference_encode"


class HelmertEncoder(_Contrast):
    """Categorical utf8 -> Helmert contrast coding, in f64 columns.

    The same layout as BackwardDifferenceEncoder, with the matrix
    ``M[i][j] = -1`` when ``i <= j``, ``i`` when ``i == j + 1``, and 0
    otherwise: column j contrasts level j + 1 against the mean of the levels
    before it. Every entry is a small integer held exactly in an f64.
    """

    kernel_id: ClassVar[str] = "helmert_encode"


class SumEncoder(_Contrast):
    """Categorical utf8 -> sum (deviation) contrast coding, in f64 columns.

    The same layout as BackwardDifferenceEncoder, with the identity on every
    level but the last, whose row is all -1: each column contrasts one level
    against the grand mean and the last level is the reference the others are
    read against. Because levels sort here, that reference is the last
    category in byte order, or the fitted null under
    ``handle_missing="value"``, which sorts after every real category. Under
    ``handle_missing="return_nan"`` or ``"error"`` it is the placeholder
    level, which no value reaches, and under ``handle_unknown="indicator"``
    it is the unknown level, whatever ``handle_missing`` says.
    """

    kernel_id: ClassVar[str] = "sum_encode"


class PolynomialEncoder(_Contrast):
    """Categorical utf8 -> orthogonal polynomial contrast coding, in f64 columns.

    The same layout as BackwardDifferenceEncoder: column j is the degree
    j + 1 polynomial over the sorted level positions, with the constant
    column omitted. The fit uses Arnoldi iteration on the level positions,
    centered and scaled into [-1, 1], with two orthogonalization passes at
    each degree, avoiding large powers and stabilizing the high-degree basis.
    Building a matrix for n effective levels takes O(n^3) time and O(n^2)
    memory; transformation uses that matrix as a lookup table.

    A saved pipeline stores the fitted matrix verbatim and uses it as stored.
    At high cardinality the values differ from category_encoders': its
    patsy/LAPACK QR basis accumulates high-degree error (about 1e-5 at 32
    levels), and the stable basis does not reproduce that error.
    """

    kernel_id: ClassVar[str] = "polynomial_encode"


MAX_COMPONENTS: int = 4096


class HashingEncoder(Transformer):
    """The hashing trick: utf8 columns collapse into one shared block of counters.

    The one encoder that learns nothing. A value's bucket is
    ``int.from_bytes(md5(value), "big") % n_components``, a pure function of
    its bytes, so the encoder never grows a vocabulary and a value it never
    saw is no different from any other. Every encoded column feeds the same
    block: the encoded columns are dropped and ``col_0`` through
    ``col_{n_components - 1}`` appended after the columns that stay, all i64,
    so a row with k non-null encoded values sums to k across the block. A null
    contributes nothing to any counter.

    Only utf8 columns can be hashed, and the bytes hashed are the value's own.
    ``Cast`` a number to utf8 first: that spells an integer as decimal digits,
    exactly what category_encoders' ``str(value)`` gives, while a float has
    no single spelling and cannot be cast to utf8 at all.

    The output names are category_encoders' and are fixed, so two
    HashingEncoders in one pipeline would collide; that is a fit-time error
    rather than a silent overwrite. category_encoders' ``max_process``,
    ``max_sample`` and ``process_creation_method`` are not ported: they are
    multiprocessing options, and the engine already runs up to one worker
    per Parquet row group.
    """

    kernel_id: ClassVar[str] = "hash_encode"

    def __init__(self, columns: list[str], *, n_components: int = 8, hash_method: str = "md5") -> None:
        """Initialize the encoder.

        Args:
            columns: Columns to hash; every one of them feeds the same block
                of counters and none of them survives to the output.
            n_components: Number of buckets, and so of output columns, from 1
                to 4096. The collision rate is roughly ``1 / n_components``.
            hash_method: The digest to hash with; only ``"md5"``,
                category_encoders' default, is implemented.

        Raises:
            TypeError: If columns is a bare string or n_components is not an integer.
            ValueError: If columns is empty, n_components is out of range, or
                hash_method is anything but "md5".

        """
        self.columns: list[str] = _column_list(columns, "HashingEncoder")
        self.n_components: int = _integer(n_components, "HashingEncoder: n_components")
        if not 1 <= self.n_components <= MAX_COMPONENTS:
            raise ValueError(f"HashingEncoder: n_components must be between 1 and {MAX_COMPONENTS}, got {n_components}")
        if hash_method != "md5":
            raise ValueError(f"HashingEncoder: hash_method must be 'md5', got {hash_method!r}")
        self.hash_method: str = "md5"

    @override
    def params(self) -> dict[str, Any]:
        return {"columns": self.columns, "n_components": self.n_components, "hash_method": self.hash_method}

    @property
    def output_columns(self) -> list[str]:
        """The block this encoder writes, ``col_0`` through ``col_{n_components - 1}``.

        Known from the parameters alone, before any fit, because the encoder
        holds no state.

        Returns:
            The output column names, in block order.

        """
        return [f"col_{index}" for index in range(self.n_components)]


class _TargetBlend(Transformer):
    """Shared base for the target-mean family: a categorical column learned from a target.

    The target is named in the parameters rather than passed to ``fit``, so a
    saved pipeline rebuilds from its manifest alone and a serving file that
    carries no label still transforms. It is an ordinary integer or float
    column of the input, and it passes through to the output — add a trailing
    ``Select`` to drop it. A row whose target is null teaches nothing; a NaN
    or infinite target is an error.

    Attributes:
        columns: Columns to encode, or None for every utf8 column at fit.
        target: The target column.
        handle_unknown: What an unseen category becomes.
        handle_missing: What a null becomes; "error" is refused at
            transform as well as at fit.

    """

    def __init__(
        self,
        target: str,
        columns: list[str] | None,
        handle_unknown: str,
        handle_missing: str,
    ) -> None:
        """Validate the parameters every kernel in the family shares.

        Args:
            target: The integer or float column to learn from.
            columns: Columns to encode; None encodes every utf8 column found
                at fit.
            handle_unknown: What an unseen category becomes: "value",
                "return_nan", or "error".
            handle_missing: What a null becomes: "value", "return_nan", or
                "error" (refused at transform as well as at fit).

        Raises:
            TypeError: If target is not a column name or columns is a bare string.
            ValueError: If columns is empty, holds the target, or an option is
                not one of the accepted values.

        """
        what: str = type(self).__name__
        self.target: str = _target(target, what)
        self.columns: list[str] | None = _optional_columns(columns, what)
        _target_outside(self.target, self.columns, what)
        self.handle_unknown: str = _one_of(handle_unknown, _HANDLE_UNKNOWN, what, "handle_unknown")
        self.handle_missing: str = _one_of(handle_missing, _HANDLE_MISSING, what, "handle_missing")

    def _shared_params(self) -> dict[str, Any]:
        """Parameters every kernel in the family takes.

        Returns:
            The declarative parameter mapping shared by the family.

        """
        return {
            "columns": self.columns,
            "target": self.target,
            "handle_unknown": self.handle_unknown,
            "handle_missing": self.handle_missing,
        }

    @property
    def prior_(self) -> float:
        """The fit-wide target mean: the sum of the target over its non-null count.

        Returns:
            The prior.

        """
        return self._state()["prior"]

    @property
    def values_(self) -> dict[str, dict[str, float]]:
        """Fitted category -> encoded value tables, materialized from the C++ core.

        Returns:
            Mapping of column name to its category -> value table.

        """
        return self._state()["values"]

    @property
    def null_values_(self) -> dict[str, float | None]:
        """The value of each column's null category, or None where it has none.

        Returns:
            Mapping of column name to the null category's value.

        """
        return self._state()["null_values"]


def _positive(value: Any, what: str) -> float:
    """Validate a strictly positive smoothing-style parameter.

    Args:
        value: The value to check.
        what: What the value is, for the error message.

    Returns:
        The value as a float.

    Raises:
        TypeError: If the value is a bool or not a number.
        ValueError: If the value is not positive, not finite, or outside the
            range of f64.

    """
    number: float = _real(value, what)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{what} must be a finite number > 0, got {value!r}")
    return number


class TargetEncoder(_TargetBlend):
    """Categorical utf8 -> a blend of the category's target mean and the fit-wide prior.

    ``TargetEncoder(target="clicked", columns=["city"])`` learns, for each
    category, how many rows carried it (``n``) and what their target summed to
    (``s``), and encodes it as ``prior * (1 - w) + (s / n) * w``, where
    ``prior`` is the fit-wide target mean and
    ``w = 1 / (1 + exp(-(n - min_samples_leaf) / smoothing))`` is an S-curve
    that trusts a category the more rows it has. The output is f64. A null is
    a category of its own under ``handle_missing="value"`` (a null unseen at
    fit takes the prior), a null under ``"return_nan"``, and an error under
    ``"error"``. An unseen category takes the prior under
    ``handle_unknown="value"``, a null under ``"return_nan"``, and is an error
    under ``"error"``. The ``hierarchy`` parameter of category_encoders is not
    supported: it needs extra columns describing a tree of categories.
    """

    kernel_id: ClassVar[str] = "target_encode"

    def __init__(
        self,
        target: str,
        columns: list[str] | None = None,
        *,
        min_samples_leaf: int = 20,
        smoothing: float = 10.0,
        handle_unknown: str = "value",
        handle_missing: str = "value",
    ) -> None:
        """Initialize the encoder.

        Args:
            target: The integer or float column to learn from; it passes
                through to the output.
            columns: Columns to encode; None encodes every utf8 column found
                at fit.
            min_samples_leaf: The category size at which the S-curve reaches
                0.5, so a smaller category leans on the prior.
            smoothing: How flat that S-curve is; must be positive and finite.
            handle_unknown: What an unseen category becomes: "value" (the
                prior), "return_nan", or "error".
            handle_missing: What a null becomes: "value" (a category of its
                own), "return_nan", or "error" (refused at transform as well
                as at fit).

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If columns is empty or holds the target,
                min_samples_leaf is outside -2**63 to 2**63 - 1, smoothing
                is not a finite number above 0 or is outside the range of
                f64, or an option is not one of the accepted values.

        """
        super().__init__(target, columns, handle_unknown, handle_missing)
        self.min_samples_leaf: int = _integer(min_samples_leaf, "TargetEncoder: min_samples_leaf")
        self.smoothing: float = _positive(smoothing, "TargetEncoder: smoothing")

    @override
    def params(self) -> dict[str, Any]:
        return self._shared_params() | {"min_samples_leaf": self.min_samples_leaf, "smoothing": self.smoothing}


class MEstimateEncoder(_TargetBlend):
    """Categorical utf8 -> the m-probability estimate of its target mean.

    ``MEstimateEncoder(target="clicked", columns=["city"])`` is the
    one-parameter cousin of TargetEncoder: a category carrying ``n`` rows that
    sum to ``s`` encodes as ``(s + prior * m) / (n + m)``, which is the
    category mean pulled toward the fit-wide prior by ``m`` imagined prior
    rows. A column with as many categories as fit rows is id-like and teaches
    nothing, so the whole column becomes the prior. The output is f64, and
    nulls and unseen categories follow ``handle_missing`` and
    ``handle_unknown`` as they do for TargetEncoder. The ``randomized`` and
    ``sigma`` parameters of category_encoders are not supported: they
    multiply the training-time output by Gaussian noise and are off by
    default.
    """

    kernel_id: ClassVar[str] = "m_estimate_encode"

    def __init__(
        self,
        target: str,
        columns: list[str] | None = None,
        *,
        m: float = 1.0,
        handle_unknown: str = "value",
        handle_missing: str = "value",
    ) -> None:
        """Initialize the encoder.

        Args:
            target: The integer or float column to learn from; it passes
                through to the output.
            columns: Columns to encode; None encodes every utf8 column found
                at fit.
            m: How many imagined prior rows pull a category toward the prior;
                must be finite and not negative.
            handle_unknown: What an unseen category becomes: "value" (the
                prior), "return_nan", or "error".
            handle_missing: What a null becomes: "value" (a category of its
                own), "return_nan", or "error" (refused at transform as well
                as at fit).

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If columns is empty or holds the target, m is negative,
                not finite, or outside the range of f64, or an option is not
                one of the accepted values.

        """
        super().__init__(target, columns, handle_unknown, handle_missing)
        self.m: float = _smoothing(m, "MEstimateEncoder: m")

    @override
    def params(self) -> dict[str, Any]:
        return self._shared_params() | {"m": self.m}


class WOEEncoder(_TargetBlend):
    """Categorical utf8 -> the weight of evidence of its binary target.

    ``WOEEncoder(target="clicked", columns=["city"])`` encodes a category
    carrying ``n`` rows that sum to ``s`` as
    ``ln(((s + r) / (S + 2r)) / (((n - s) + r) / ((N - S) + 2r)))``, the log
    odds of the category against the fit-wide odds, where ``S`` and ``N`` are
    the target's sum and row count and ``r`` is ``regularization`` (which
    keeps the ratio away from zero). The target must be binary: exactly two
    distinct values, one of them 0 or less and the other 1 or more, or the fit
    is an error. A category seen exactly once encodes as 0, and so does an
    unseen category: the scale is centered on 0, not on the prior. The output
    is f64. The ``randomized`` and ``sigma`` parameters of category_encoders
    are not supported: they multiply the training-time output by Gaussian
    noise and are off by default.
    """

    kernel_id: ClassVar[str] = "woe_encode"

    def __init__(
        self,
        target: str,
        columns: list[str] | None = None,
        *,
        regularization: float = 1.0,
        handle_unknown: str = "value",
        handle_missing: str = "value",
    ) -> None:
        """Initialize the encoder.

        Args:
            target: The binary integer or float column to learn from; it
                passes through to the output.
            columns: Columns to encode; None encodes every utf8 column found
                at fit.
            regularization: Added to both halves of the odds ratio so neither
                can be zero; must be positive and finite.
            handle_unknown: What an unseen category becomes: "value" (0),
                "return_nan", or "error".
            handle_missing: What a null becomes: "value" (a category of its
                own), "return_nan", or "error" (refused at transform as well
                as at fit).

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If columns is empty or holds the target, regularization
                is not a finite number above 0 or is outside the range of f64,
                or an option is not one of the accepted values.

        """
        super().__init__(target, columns, handle_unknown, handle_missing)
        self.regularization: float = _positive(regularization, "WOEEncoder: regularization")

    @override
    def params(self) -> dict[str, Any]:
        return self._shared_params() | {"regularization": self.regularization}


class _ShrinkageEncoder(Transformer):
    """Shared base for the target encoders that shrink a category mean toward the prior.

    The target column is named in the parameters, not handed to ``fit``: a
    saved pipeline rebuilds from its manifest alone, and ``Pipeline.fit``'s
    ``y`` stays ignored. While unfitted the step reads the target as well as
    its columns; once fitted it reads only its columns, so a serving file
    without the label still transforms. The target itself passes through to
    the output — add a trailing ``Select`` to drop it.
    """

    def __init__(
        self,
        columns: list[str] | None,
        target: str,
        handle_unknown: str,
        handle_missing: str,
    ) -> None:
        """Validate the arguments every shrinkage encoder shares.

        Args:
            columns: Columns to encode; None encodes every utf8 column found
                at fit.
            target: The integer or float column to take the statistics from.
            handle_unknown: What an unseen category becomes: "value" (the
                prior), "return_nan", or "error".
            handle_missing: What a null becomes: "value" (a category of its
                own), "return_nan", or "error" (refused at transform as well
                as at fit).

        Raises:
            TypeError: If columns is a bare string or target is not a column name.
            ValueError: If columns is empty, holds the target, or an option is
                not one of the accepted values.

        """
        what: str = type(self).__name__
        self.columns: list[str] | None = _optional_columns(columns, what)
        self.target: str = _target(target, what)
        _target_outside(self.target, self.columns, what)
        self.handle_unknown: str = _one_of(handle_unknown, _HANDLE_UNKNOWN, what, "handle_unknown")
        self.handle_missing: str = _one_of(handle_missing, _HANDLE_MISSING, what, "handle_missing")

    @override
    def params(self) -> dict[str, Any]:
        return {
            "columns": self.columns,
            "target": self.target,
            "handle_unknown": self.handle_unknown,
            "handle_missing": self.handle_missing,
        }

    @property
    def prior_(self) -> float:
        """The mean of the target's non-null values, the value every category is shrunk toward.

        Returns:
            The prior.

        """
        return self._state()["prior"]

    @property
    def values_(self) -> dict[str, dict[str, float]]:
        """Fitted category -> encoded value tables, materialized from the C++ core.

        Returns:
            Mapping of column name to its category -> value table.

        """
        return self._state()["values"]


JAMES_STEIN_MODELS: frozenset[str] = frozenset({"independent", "pooled", "beta"})


class JamesSteinEncoder(_ShrinkageEncoder):
    """Categorical utf8 -> the target mean of its category, shrunk toward the global mean.

    The weight is closed-form rather than cross-validated, and ``model``
    chooses how it is computed. With ``m_c`` the category's target mean,
    ``n_c`` its row count, ``K`` the number of categories, ``N`` the fitted
    rows, and ``prior`` the target's global mean:

    * ``independent`` (the default) weighs each category on its own variance:
      ``B_c = i_var_c / (var(y) + i_var_c) * (K - 3) / (K - 1)``, where
      ``i_var_c`` is the within-category sample variance (0 for a category of
      one row), the weight is ``w_c = clip(1 - B_c, 0, 1)`` and the value is
      ``w_c * m_c + (1 - w_c) * prior``.
    * ``pooled`` assumes one variance for every category:
      ``variance = var(y) / (N / K)``, ``SSE = sum (m_c - prior)^2``,
      ``B = clip((K - 3) * variance / SSE, 0, 1)``, and the value is
      ``prior + (1 - B) * (m_c - prior)``.
    * ``beta`` weighs a category by its size alone: ``w_c = n_c / (n_c + N)``.

    A column with as many categories as rows identifies the row, so every
    category collapses to the prior, as category_encoders does.
    ``model="binary"`` is refused: category_encoders reads its estimate out of
    a module global that its optimizer happens to leave behind, so the result
    is not reproducible from the published formula. A constant target leaves
    the ``independent`` weight at 0/0, where category_encoders answers NaN and
    this package answers the category mean, which is the prior.
    category_encoders' ``randomized`` and ``sigma`` control Gaussian noise
    added at training time only, so they are not parameters here.
    """

    kernel_id: ClassVar[str] = "james_stein_encode"

    def __init__(
        self,
        columns: list[str] | None = None,
        *,
        target: str,
        model: str = "independent",
        handle_unknown: str = "value",
        handle_missing: str = "value",
    ) -> None:
        """Initialize the encoder.

        Args:
            columns: Columns to encode; None encodes every utf8 column found
                at fit.
            target: The integer or float column to take the statistics from.
            model: One of ``independent``, ``pooled``, ``beta``.
            handle_unknown: What an unseen category becomes: "value" (the
                prior), "return_nan", or "error".
            handle_missing: What a null becomes: "value" (a category of its
                own), "return_nan", or "error" (refused at transform as well
                as at fit).

        Raises:
            TypeError: If columns is a bare string or target is not a column name.
            ValueError: If columns is empty or holds the target, or model or an
                option is not one of the accepted values.

        """
        super().__init__(columns, target, handle_unknown, handle_missing)
        if model == "binary":
            raise ValueError(
                "JamesSteinEncoder: model='binary' reads its estimate from a module global that category_encoders' "
                "optimizer leaves behind, so it is not reproducible from the formula and not portable; "
                f"choose one of {sorted(JAMES_STEIN_MODELS)}"
            )
        self.model: str = _one_of(model, JAMES_STEIN_MODELS, "JamesSteinEncoder", "model")

    @override
    def params(self) -> dict[str, Any]:
        return {**super().params(), "model": self.model}


class LeaveOneOutEncoder(_ShrinkageEncoder):
    """Categorical utf8 -> the target mean of its category, with a singleton collapsing to the prior.

    The value is ``sum_c / n_c`` when the category has more than one row and
    the target's global mean otherwise, so a category of one row cannot leak
    that row's own target. This is category_encoders' inference form (its
    ``transform`` called without a target). Its ``fit_transform`` instead
    leaves each training row's own target out of that row's value, which is
    a function of the row rather than of the fitted state; transform here is
    a pure function of the fitted state, so those training-time values are
    not reproduced. Its ``sigma`` and ``random_state`` control Gaussian noise
    added at training time only, so they are not parameters here.
    """

    kernel_id: ClassVar[str] = "leave_one_out_encode"

    def __init__(
        self,
        columns: list[str] | None = None,
        *,
        target: str,
        handle_unknown: str = "value",
        handle_missing: str = "value",
    ) -> None:
        """Initialize the encoder.

        Args:
            columns: Columns to encode; None encodes every utf8 column found
                at fit.
            target: The integer or float column to take the statistics from.
            handle_unknown: What an unseen category becomes: "value" (the
                prior), "return_nan", or "error".
            handle_missing: What a null becomes: "value" (a category of its
                own), "return_nan", or "error" (refused at transform as well
                as at fit).

        Raises:
            TypeError: If columns is a bare string or target is not a column name.
            ValueError: If columns is empty or holds the target, or an option is
                not one of the accepted values.

        """
        super().__init__(columns, target, handle_unknown, handle_missing)


class CatBoostEncoder(_ShrinkageEncoder):
    """Categorical utf8 -> the target mean of its category, smoothed toward the prior.

    The value is ``(sum_c + prior * a) / (n_c + a)`` when the category has
    more than one row (``a`` extra observations worth of prior, 1 by default)
    and the raw prior otherwise, so a singleton is replaced rather than
    smoothed. This is category_encoders' inference form (its ``transform``
    called without a target). Its ``fit_transform`` instead walks the
    training rows in order and gives each row the running statistic of the
    rows before it, which depends on row order; fits here do not, so those
    training-time values are not reproduced. Its ``sigma`` and
    ``random_state`` control Gaussian noise added at training time only, so
    there is nothing to reproduce at inference and they are not parameters
    here.
    """

    kernel_id: ClassVar[str] = "cat_boost_encode"

    def __init__(
        self,
        columns: list[str] | None = None,
        *,
        target: str,
        a: float = 1.0,
        handle_unknown: str = "value",
        handle_missing: str = "value",
    ) -> None:
        """Initialize the encoder.

        Args:
            columns: Columns to encode; None encodes every utf8 column found
                at fit.
            target: The integer or float column to take the statistics from.
            a: Additive smoothing, the number of prior-valued observations
                each category of more than one row is credited with.
            handle_unknown: What an unseen category becomes: "value" (the
                prior), "return_nan", or "error".
            handle_missing: What a null becomes: "value" (a category of its
                own), "return_nan", or "error" (refused at transform as well
                as at fit).

        Raises:
            TypeError: If columns is a bare string, target is not a column
                name, or a is not a number.
            ValueError: If columns is empty or holds the target, a is negative,
                not finite, or outside the range of f64, or an option is not
                one of the accepted values.

        """
        super().__init__(columns, target, handle_unknown, handle_missing)
        self.a: float = _smoothing(a, "CatBoostEncoder: a")

    @override
    def params(self) -> dict[str, Any]:
        return {**super().params(), "a": self.a}


def _quantile(quantile: object, what: str, key: str) -> float:
    """Validate one quantile.

    Args:
        quantile: The value passed for a quantile.
        what: The transformer name, for the error message.
        key: The argument name, for the error message.

    Returns:
        The quantile as a float.

    Raises:
        TypeError: If the quantile is a bool or not a number.
        ValueError: If the quantile is outside [0, 1] or outside the range of
            f64.

    """
    number: float = _real(quantile, f"{what}: {key}")
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{what}: {key} must be a number between 0 and 1, got {quantile!r}")
    return number


def _smoothing(value: object, what: str) -> float:
    """Validate a smoothing weight that may be 0, such as an m or an a.

    Args:
        value: The value to check.
        what: What the value is, for the error message.

    Returns:
        The weight as a float.

    Raises:
        TypeError: If the value is a bool or not a number.
        ValueError: If the value is negative, not finite, or outside the
            range of f64.

    """
    number: float = _real(value, what)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{what} must be a finite number >= 0, got {value!r}")
    return number


class QuantileEncoder(Transformer):
    """Categorical utf8 -> a quantile of the target, shrunk toward the global quantile.

    ``QuantileEncoder(target="price", columns=["city"], quantile=0.5)`` fits
    the quantile of ``price`` over every row (the prior) and over the rows of
    each category, then encodes a category holding ``n`` target values whose
    quantile is ``q`` as ``(n * q + prior * m) / (n + m)``: ``m=0`` is the raw
    quantile and a large ``m`` is the prior. The quantile is numpy's
    ``"linear"`` method, so it interpolates between the two values around
    ``quantile * (n - 1)``. The target is named here rather than passed to
    ``fit``, so a saved pipeline rebuilds the step exactly; it must be an
    integer or float column, it cannot be one of ``columns``, and it passes
    through to the output (add a trailing ``Select`` to drop it). A row whose
    target is null contributes to nothing. A null is a category of its own
    under ``handle_missing="value"`` (the prior when the fit saw none), a null
    under ``"return_nan"``, and an error under ``"error"`` (at transform as
    well as at fit); an unseen category yields the prior under
    ``handle_unknown="value"``, a null under ``"return_nan"``, and an error
    under ``"error"``. The encoded column becomes f64 in place. An exact
    quantile needs every target value, so the fit keeps them all, for each
    encoded column and for the prior, and moves them to temporary files when
    they outgrow the fit's memory budget.
    """

    kernel_id: ClassVar[str] = "quantile_encode"

    def __init__(
        self,
        target: str,
        columns: list[str] | None = None,
        *,
        quantile: float = 0.5,
        m: float = 1.0,
        handle_unknown: str = "value",
        handle_missing: str = "value",
    ) -> None:
        """Initialize the encoder.

        Args:
            target: The integer or float column whose quantile is encoded.
            columns: Columns to encode; None encodes every utf8 column found
                at fit, except the target.
            quantile: The quantile to take, between 0 and 1; 0.5 is the median.
            m: The m-probability smoothing weight; 0 disables smoothing.
            handle_unknown: What an unseen category becomes: "value" (the
                prior), "return_nan", or "error".
            handle_missing: What a null becomes: "value" (a category of its
                own), "return_nan", or "error" (refused at transform as well
                as at fit).

        Raises:
            TypeError: If target, columns, quantile, or m has the wrong type.
            ValueError: If columns is empty or holds the target, quantile is
                outside [0, 1], m is negative or not finite, quantile or m is
                outside the range of f64, or a handling option is not one of
                the accepted values.

        """
        self.target: str = _target(target, "QuantileEncoder")
        self.columns: list[str] | None = _optional_columns(columns, "QuantileEncoder")
        _target_outside(self.target, self.columns, "QuantileEncoder")
        self.quantile: float = _quantile(quantile, "QuantileEncoder", "quantile")
        self.m: float = _smoothing(m, "QuantileEncoder: m")
        self.handle_unknown: str = _one_of(handle_unknown, _HANDLE_UNKNOWN, "QuantileEncoder", "handle_unknown")
        self.handle_missing: str = _one_of(handle_missing, _HANDLE_MISSING, "QuantileEncoder", "handle_missing")

    @override
    def params(self) -> dict[str, Any]:
        return {
            "columns": self.columns,
            "target": self.target,
            "quantile": self.quantile,
            "m": self.m,
            "handle_unknown": self.handle_unknown,
            "handle_missing": self.handle_missing,
        }

    @property
    def prior_(self) -> float:
        """The fitted quantile of the target's non-null values.

        Returns:
            The prior.

        """
        return self._state()["priors"][0]

    @property
    def categories_(self) -> dict[str, list[str]]:
        """Fitted categories per encoded column, sorted.

        Returns:
            Mapping of column name to its categories.

        """
        return self._state()["categories"]

    @property
    def encodings_(self) -> dict[str, list[float]]:
        """Fitted values per encoded column, aligned with ``categories_``.

        Returns:
            Mapping of column name to one encoded value per category.

        """
        return self._state()["encodings"]


class SummaryEncoder(Transformer):
    """QuantileEncoder at several quantiles at once, one output column each.

    ``SummaryEncoder(target="price", columns=["city"], quantiles=(0.25, 0.75))``
    fits one QuantileEncoder per quantile over the same data and replaces
    each encoded column, in its own position, with one f64 column per
    quantile named ``<column>_<round(quantile * 100)>``, in the order the
    quantiles were given. Everything else — the smoothing, the target rules,
    the null and unknown handling, and how the target values are kept — is
    QuantileEncoder's. Two quantiles that round to the same percentile would
    name the same column and are rejected.
    """

    kernel_id: ClassVar[str] = "summary_encode"

    def __init__(
        self,
        target: str,
        columns: list[str] | None = None,
        *,
        quantiles: Sequence[float] = (0.25, 0.75),
        m: float = 1.0,
        handle_unknown: str = "value",
        handle_missing: str = "value",
    ) -> None:
        """Initialize the encoder.

        Args:
            target: The integer or float column whose quantiles are encoded.
            columns: Columns to encode; None encodes every utf8 column found
                at fit, except the target.
            quantiles: The quantiles to take, each between 0 and 1.
            m: The m-probability smoothing weight; 0 disables smoothing.
            handle_unknown: What an unseen category becomes: "value" (the
                prior), "return_nan", or "error".
            handle_missing: What a null becomes: "value" (a category of its
                own), "return_nan", or "error" (refused at transform as well
                as at fit).

        Raises:
            TypeError: If target, columns, quantiles, or m has the wrong type,
                or quantiles is empty.
            ValueError: If columns is empty or holds the target, quantiles
                holds a value outside [0, 1] or two that round to the same
                percentile, m is negative or not finite, a quantile or m is
                outside the range of f64, or a handling option is not one of
                the accepted values.

        """
        self.target: str = _target(target, "SummaryEncoder")
        self.columns: list[str] | None = _optional_columns(columns, "SummaryEncoder")
        _target_outside(self.target, self.columns, "SummaryEncoder")
        if isinstance(quantiles, str) or not isinstance(quantiles, (list, tuple)) or not quantiles:
            raise TypeError(f"SummaryEncoder: quantiles must be a non-empty list of numbers, got {quantiles!r}")
        self.quantiles: list[float] = [_quantile(value, "SummaryEncoder", "quantiles") for value in quantiles]
        percentiles: list[int] = [round(value * 100) for value in self.quantiles]
        if len(set(percentiles)) != len(percentiles):
            raise ValueError(f"SummaryEncoder: two quantiles round to the same percentile, got {self.quantiles!r}")
        self.m: float = _smoothing(m, "SummaryEncoder: m")
        self.handle_unknown: str = _one_of(handle_unknown, _HANDLE_UNKNOWN, "SummaryEncoder", "handle_unknown")
        self.handle_missing: str = _one_of(handle_missing, _HANDLE_MISSING, "SummaryEncoder", "handle_missing")

    @override
    def params(self) -> dict[str, Any]:
        return {
            "columns": self.columns,
            "target": self.target,
            "quantiles": self.quantiles,
            "m": self.m,
            "handle_unknown": self.handle_unknown,
            "handle_missing": self.handle_missing,
        }

    @property
    def priors_(self) -> list[float]:
        """The fitted quantiles of the target's non-null values, in ``quantiles`` order.

        Returns:
            One prior per quantile.

        """
        return self._state()["priors"]

    @property
    def categories_(self) -> dict[str, list[str]]:
        """Fitted categories per encoded column, sorted.

        Returns:
            Mapping of column name to its categories.

        """
        return self._state()["categories"]

    @property
    def encodings_(self) -> dict[str, list[float]]:
        """Fitted values per generated column, aligned with its source column's ``categories_``.

        Returns:
            Mapping of ``<column>_<percentile>`` to one encoded value per category.

        """
        return self._state()["encodings"]


def _flag(value: object, what: str, key: str) -> bool:
    """Validate a boolean option.

    Args:
        value: The value passed for the option.
        what: The transformer name, for the error message.
        key: The option name, for the error message.

    Returns:
        The value.

    Raises:
        TypeError: If the value is not a bool.

    """
    if not isinstance(value, bool):
        raise TypeError(f"{what}: {key} must be True or False, got {value!r}")
    return value


def _finite(value: object, what: str, key: str) -> float:
    """Validate a finite number.

    Args:
        value: The value passed for the option.
        what: The transformer name, for the error message.
        key: The option name, for the error message.

    Returns:
        The value as a float.

    Raises:
        TypeError: If the value is a bool or not a number.
        ValueError: If the value is not finite or outside the range of f64.

    """
    number: float = _real(value, f"{what}: {key}")
    if not math.isfinite(number):
        raise ValueError(f"{what}: {key} must be a finite number, got {value!r}")
    return number


class _Scaler(Transformer):
    """Shared base for the three fitted scalers: a numeric column list and its row count.

    ``columns=None`` scales every integer or float column that reaches the
    step and is known when the fit starts, chosen once at fit and saved with
    the pipeline. A null teaches nothing and comes out a null, which is what
    scikit-learn's NaN does; a NaN or infinity in the data itself is an
    error, because a missing number has one spelling here. Every scaled
    column becomes f64 whatever width it arrived as.
    """

    def __init__(self, columns: list[str] | None = None) -> None:
        """Initialize the scaler.

        Args:
            columns: Columns to scale; None scales every integer or float
                column found at fit.

        Raises:
            TypeError: If columns is a bare string or holds non-strings.
            ValueError: If columns is empty.

        """
        self.columns: list[str] | None = _optional_columns(columns, type(self).__name__)

    @property
    def scale_(self) -> dict[str, float]:
        """The divisor (or multiplier) fitted for each column.

        Returns:
            Mapping of column name to its fitted scale.

        """
        return self._state()["scales"]


class StandardScaler(_Scaler):
    """Numeric columns -> ``(x - mean) / sqrt(var)``, the scikit-learn scaler.

    ``StandardScaler(columns=["price"])`` fits the mean and the population
    variance (``ddof=0``) of ``price`` over its non-null values, then encodes
    ``(x - mean_) / scale_``, subtracting and then dividing, in that order.
    ``scale_`` is ``sqrt(var_)`` except for a column whose variance is
    indistinguishable from zero, which scales by 1 instead, so a centered
    constant column comes out 0 rather than NaN; that is scikit-learn's own
    bound, ``var <= n * eps * var + (n * mean * eps) ** 2``, taken on the
    variance before the square root. ``with_mean=False`` leaves out the
    centering and ``with_std=False`` the division; both statistics are
    recorded either way, because they cost one pass whichever flags are set.

    The sums are exact partials and the variance comes from the exact
    ``sum((x - mean) ** 2)``, divided by n, so the fitted table never
    depends on how the rows were split across workers. scikit-learn
    accumulates both incrementally instead, so the two agree by accident
    rather than by construction: ``mean_`` can differ from scikit-learn's by
    up to one ulp of the mean, and with ``with_std=False`` that difference
    reaches the output undivided and grows with the column's magnitude. On
    ``[1e16, 1.0000000000000004e16, 1e16, 1e16, 1.0000000000000004e16]``
    this package centers on ``1.0000000000000002e16``, the correctly rounded
    and so more accurate mean, where scikit-learn centers on ``1e16``, and
    every output value differs by 2.0. Every parity case matches
    scikit-learn bit for bit.

    The squares of a column's values are summed exactly as well, and a
    column whose squares sum to ``2**1024 - 2**970`` or more, half a step
    past the largest f64 (about 1.8e308), is an error: n values with a root
    mean square above about ``1.34e154 / sqrt(n)``. A sum within half a step
    of the largest f64, on either side, rounds to it, and such a column can
    be an error in one order or split of its rows and fit in another. With
    several threads the split, and so the outcome, can change from one fit
    of the same files to the next. scikit-learn fits these columns when
    their squared deviations from the mean sum within f64. No other column
    is an error for its magnitude.
    """

    kernel_id: ClassVar[str] = "standard_scale"

    def __init__(self, columns: list[str] | None = None, *, with_mean: bool = True, with_std: bool = True) -> None:
        """Initialize the scaler.

        Args:
            columns: Columns to scale; None scales every integer or float
                column found at fit.
            with_mean: Subtract the fitted mean.
            with_std: Divide by the fitted scale.

        Raises:
            TypeError: If columns is a bare string, or a flag is not a bool.
            ValueError: If columns is empty.

        """
        super().__init__(columns)
        self.with_mean: bool = _flag(with_mean, "StandardScaler", "with_mean")
        self.with_std: bool = _flag(with_std, "StandardScaler", "with_std")

    @override
    def params(self) -> dict[str, Any]:
        return {"columns": self.columns, "with_mean": self.with_mean, "with_std": self.with_std}

    @property
    def mean_(self) -> dict[str, float]:
        """The fitted mean of each column's non-null values.

        Returns:
            Mapping of column name to its mean.

        """
        return self._state()["means"]

    @property
    def var_(self) -> dict[str, float]:
        """The fitted population variance (``ddof=0``) of each column's non-null values.

        Returns:
            Mapping of column name to its variance.

        """
        return self._state()["variances"]

    @property
    def n_samples_seen_(self) -> dict[str, int]:
        """How many non-null values each column contributed, as scikit-learn counts them.

        Returns:
            Mapping of column name to its non-null count.

        """
        return self._state()["counts"]


class MinMaxScaler(_Scaler):
    """Numeric columns -> ``x * scale_ + min_``, mapping the fitted range onto ``feature_range``.

    ``MinMaxScaler(columns=["price"], feature_range=(0, 1))`` fits
    ``data_min_`` and ``data_max_``, then
    ``scale_ = (high - low) / (data_max_ - data_min_)`` and
    ``min_ = low - data_min_ * scale_``, so the smallest fitted value lands on
    ``low`` and the largest on ``high``, up to rounding, as in scikit-learn. A
    column whose range is below ``10 * f64`` epsilon divides by 1 instead, so
    a constant column comes out at ``low``, up to that rounding, rather than
    NaN. ``clip=True`` clamps the result back into ``feature_range``, which a
    value outside the fitted range, or a fitted extreme by that rounding, can
    leave. A scale of 0 is a fit-time error: one from a column that spans
    past the range of f64, or from a ``feature_range`` so narrow beside the
    column's range that the scale rounds to 0, would map every value to
    ``low``, as scikit-learn does. The minimum and maximum are exact under
    any split, so the fitted table is bit-identical however the rows were
    divided across workers.
    """

    kernel_id: ClassVar[str] = "min_max_scale"

    def __init__(
        self,
        columns: list[str] | None = None,
        *,
        feature_range: Sequence[float] = (0.0, 1.0),
        clip: bool = False,
    ) -> None:
        """Initialize the scaler.

        Args:
            columns: Columns to scale; None scales every integer or float
                column found at fit.
            feature_range: The (low, high) range the fitted range maps onto.
            clip: Clamp transformed values into feature_range.

        Raises:
            TypeError: If columns is a bare string, feature_range is not a
                pair of numbers, or clip is not a bool.
            ValueError: If columns is empty, feature_range is not two finite
                numbers with the first below the second, or a bound is
                outside the range of f64.

        """
        super().__init__(columns)
        if isinstance(feature_range, str) or not isinstance(feature_range, (list, tuple)) or len(feature_range) != 2:
            raise TypeError(f"MinMaxScaler: feature_range must be a (low, high) pair, got {feature_range!r}")
        low: float = _finite(feature_range[0], "MinMaxScaler", "feature_range")
        high: float = _finite(feature_range[1], "MinMaxScaler", "feature_range")
        if low >= high:
            raise ValueError(f"MinMaxScaler: feature_range must be (low, high) with low < high, got {feature_range!r}")
        self.feature_range: tuple[float, float] = (low, high)
        self.clip: bool = _flag(clip, "MinMaxScaler", "clip")

    @override
    def params(self) -> dict[str, Any]:
        return {"columns": self.columns, "feature_range": list(self.feature_range), "clip": self.clip}

    @property
    def data_min_(self) -> dict[str, float]:
        """The smallest non-null value fitted per column.

        Returns:
            Mapping of column name to its minimum.

        """
        return self._state()["data_min"]

    @property
    def data_max_(self) -> dict[str, float]:
        """The largest non-null value fitted per column.

        Returns:
            Mapping of column name to its maximum.

        """
        return self._state()["data_max"]

    @property
    def data_range_(self) -> dict[str, float]:
        """The fitted spread per column, ``data_max_ - data_min_``.

        Returns:
            Mapping of column name to its range.

        """
        highs: dict[str, float] = self.data_max_
        return {column: highs[column] - low for column, low in self.data_min_.items()}

    @property
    def min_(self) -> dict[str, float]:
        """The offset added after scaling, ``low - data_min_ * scale_``.

        Returns:
            Mapping of column name to its offset.

        """
        return self._state()["mins"]

    @property
    def n_samples_seen_(self) -> int:
        """How many rows the fit read, nulls included, as scikit-learn counts them.

        Returns:
            The fitted row count.

        """
        return self._state()["rows"]


class MaxAbsScaler(_Scaler):
    """Numeric columns -> ``x / max_abs_``, the scaler that leaves a zero a zero.

    ``MaxAbsScaler(columns=["price"])`` fits the largest absolute non-null
    value of ``price`` and divides by it, so the fitted data lands in
    ``[-1, 1]`` without shifting anything — the scaler to reach for when zero
    means "absent". A column whose largest absolute value is below
    ``10 * f64`` epsilon divides by 1 instead, so an all-zero column comes
    out zeros. ``clip=True`` clamps the result to ``[-1, 1]``, which only a
    value of greater magnitude than any seen at fit can leave. The maximum is
    exact under any split, so the fitted table is bit-identical however the
    rows were divided across workers.
    """

    kernel_id: ClassVar[str] = "max_abs_scale"

    def __init__(self, columns: list[str] | None = None, *, clip: bool = False) -> None:
        """Initialize the scaler.

        Args:
            columns: Columns to scale; None scales every integer or float
                column found at fit.
            clip: Clamp transformed values to [-1, 1].

        Raises:
            TypeError: If columns is a bare string or clip is not a bool.
            ValueError: If columns is empty.

        """
        super().__init__(columns)
        self.clip: bool = _flag(clip, "MaxAbsScaler", "clip")

    @override
    def params(self) -> dict[str, Any]:
        return {"columns": self.columns, "clip": self.clip}

    @property
    def max_abs_(self) -> dict[str, float]:
        """The largest absolute non-null value fitted per column.

        Returns:
            Mapping of column name to its maximum magnitude.

        """
        return self._state()["max_abs"]

    @property
    def n_samples_seen_(self) -> int:
        """How many rows the fit read, nulls included, as scikit-learn counts them.

        Returns:
            The fitted row count.

        """
        return self._state()["rows"]


class Binarizer(Transformer):
    """Numeric columns -> 0 or 1 by a threshold. Stateless.

    ``Binarizer(columns=["clicks"], threshold=0.0)`` writes 1 where the value
    is strictly above the threshold and 0 everywhere else, so a value sitting
    exactly on the threshold is a 0. A null is an error rather than a 0 or
    a 1: scikit-learn's ``Binarizer`` refuses a NaN outright instead of
    guessing, and so does this package, on the record and batch paths alike.
    scikit-learn keeps the input dtype; here the output type is declared, so
    the column becomes i64 whatever came in.

    ``columns`` is required because nothing is learned: a saved pipeline that
    did not name them would binarize a different set of columns for every
    file it was handed.
    """

    kernel_id: ClassVar[str] = "binarize"

    def __init__(self, columns: list[str], *, threshold: float = 0.0) -> None:
        """Initialize the binarizer.

        Args:
            columns: Columns to binarize; every one of them becomes i64.
            threshold: Values strictly above this become 1, the rest 0.

        Raises:
            TypeError: If columns is a bare string or threshold is not a number.
            ValueError: If columns is empty, or threshold is not finite or is
                outside the range of f64.

        """
        self.columns: list[str] = _column_list(columns, "Binarizer")
        self.threshold: float = _finite(threshold, "Binarizer", "threshold")

    @override
    def params(self) -> dict[str, Any]:
        return {"columns": self.columns, "threshold": self.threshold}


NORMS: frozenset[str] = frozenset({"l1", "l2", "max"})


class Normalizer(Transformer):
    """Each row divided by its own norm across the listed columns. Stateless.

    ``Normalizer(columns=["x", "y", "z"], norm="l2")`` divides every row by
    ``sqrt(x**2 + y**2 + z**2)``; ``norm="l1"`` divides by ``|x| + |y| + |z|``
    and ``norm="max"`` by the largest of the three magnitudes. A row whose
    norm is below ``10 * f64`` epsilon is left alone rather than divided, so
    an all-zero row stays all zeros.

    Each output depends on the other listed columns of the same row, which is
    why ``columns`` is a required, ordered parameter with no ``None``
    spelling: the saved pipeline has to name the whole vector, or it would
    normalize by a different set of columns for every file it was handed. A
    file missing one of them is an error rather than a quietly different
    answer. A null is an error too, as scikit-learn's NaN is: a norm over a
    missing number has no value. Every listed column becomes f64.

    The l1 and l2 sums are exact partials, so the norm does not depend on
    the order the columns were listed in, except at the top of the range. A
    row whose sum of absolute values (l1) or of squares rounded to f64 (l2)
    is ``2**1024 - 2**970`` or more, half a step past the largest f64, is
    an error in every order. A sum less than half a step past the largest
    f64 rounds to it, and such a row of three or more columns can be an
    error in one column order and normalized in another.
    """

    kernel_id: ClassVar[str] = "normalize"

    def __init__(self, columns: list[str], *, norm: str = "l2") -> None:
        """Initialize the normalizer.

        Args:
            columns: The ordered vector each row is normalized over; every
                one of them becomes f64.
            norm: The norm to divide by: "l1", "l2", or "max".

        Raises:
            TypeError: If columns is a bare string or holds non-strings.
            ValueError: If columns is empty or norm is not one of the
                accepted values.

        """
        self.columns: list[str] = _column_list(columns, "Normalizer")
        self.norm: str = _one_of(norm, NORMS, "Normalizer", "norm")

    @override
    def params(self) -> dict[str, Any]:
        return {"columns": self.columns, "norm": self.norm}


_OUTPUT_DISTRIBUTIONS: frozenset[str] = frozenset({"uniform", "normal"})
_KBINS_ENCODINGS: frozenset[str] = frozenset({"onehot", "onehot-dense", "ordinal"})
_KBINS_STRATEGIES: frozenset[str] = frozenset({"uniform", "quantile", "kmeans"})
# Nine of the methods np.percentile accepts (all but lower, higher, midpoint
# and nearest), the ones KBinsDiscretizer(strategy="quantile") passes straight
# through.
_QUANTILE_METHODS: frozenset[str] = frozenset(
    {
        "inverted_cdf",
        "averaged_inverted_cdf",
        "closest_observation",
        "interpolated_inverted_cdf",
        "hazen",
        "weibull",
        "linear",
        "median_unbiased",
        "normal_unbiased",
    }
)


def _percentile_range(value: object, what: str) -> tuple[float, float]:
    """Validate a (q_min, q_max) percentile pair.

    Args:
        value: The value passed for ``quantile_range``.
        what: The transformer name, for the error message.

    Returns:
        The pair as floats.

    Raises:
        TypeError: If the value is not a pair of numbers.
        ValueError: If the pair is not ``0 <= q_min <= q_max <= 100``, or a
            bound is outside the range of f64.

    """
    if isinstance(value, str) or not isinstance(value, (list, tuple)) or len(value) != 2:
        raise TypeError(f"{what}: quantile_range must be two percentiles, got {value!r}")
    low: float = _real(value[0], f"{what}: quantile_range")
    high: float = _real(value[1], f"{what}: quantile_range")
    if not 0.0 <= low <= high <= 100.0:
        raise ValueError(f"{what}: quantile_range must satisfy 0 <= q_min <= q_max <= 100, got {value!r}")
    return low, high


class RobustScaler(Transformer):
    """Numeric columns -> ``(x - median) / (q_max - q_min)``, in place.

    ``RobustScaler(columns=["price"])`` fits the median of ``price`` and the
    width of its interquartile range, then transforms every value to
    ``(x - center_) / scale_``. Both statistics are taken over the column's
    non-null fit values, so an outlier moves them far less than it moves a
    mean and a standard deviation. A scale below ``10 * f64`` epsilon becomes
    ``1``, so a constant column is centered and not divided (scikit-learn's
    ``_handle_zeros_in_scale``). ``unit_variance=True`` additionally divides
    the scale by ``ppf(q_max / 100) - ppf(q_min / 100)``, the width of the
    same range on a standard normal, so a normally distributed column comes
    out with unit variance; it needs ``0 < q_min < q_max < 100``, because the
    other ranges make that divisor zero or infinite, and refuses a ``q_min``
    so small that ``q_min / 100`` rounds to 0 for the same reason. Two
    percentiles so close that their normal quantiles are equal make it 0
    too, which the fit refuses.
    ``with_centering=False`` and ``with_scaling=False`` leave the statistic
    they name unfitted, and the saved pipeline records it as null rather than
    as a neutral number. A null is ignored at fit and passes through at
    transform, which is what scikit-learn's NaN does; a value that is NaN or
    infinity is an error. The column becomes f64 in place, and an integer
    column is widened to f64, so a magnitude past 2^53 rounds. An exact
    median and quantile need every value of a column, so the fit keeps them
    all and moves them to temporary files when they outgrow the fit's memory
    budget; with centering and scaling both off, it keeps none.
    """

    kernel_id: ClassVar[str] = "robust_scale"

    def __init__(
        self,
        columns: list[str] | None = None,
        *,
        with_centering: bool = True,
        with_scaling: bool = True,
        quantile_range: Sequence[float] = (25.0, 75.0),
        unit_variance: bool = False,
    ) -> None:
        """Initialize the scaler.

        Args:
            columns: Columns to scale; None scales every integer or float
                column found at fit.
            with_centering: Subtract the median.
            with_scaling: Divide by the quantile range.
            quantile_range: The (q_min, q_max) percentiles whose width is the
                scale, with ``0 <= q_min <= q_max <= 100``.
            unit_variance: Divide the scale by the width of the same range on
                a standard normal.

        Raises:
            TypeError: If columns, a flag, or a quantile bound has the wrong type.
            ValueError: If columns is empty, the quantile range is not two
                ordered percentiles, a bound is outside the range of f64, or
                unit_variance is set and the range is not
                ``0 < q_min < q_max < 100`` or ``q_min / 100`` rounds to 0.

        """
        self.columns: list[str] | None = _optional_columns(columns, "RobustScaler")
        self.with_centering: bool = _flag(with_centering, "RobustScaler", "with_centering")
        self.with_scaling: bool = _flag(with_scaling, "RobustScaler", "with_scaling")
        self.unit_variance: bool = _flag(unit_variance, "RobustScaler", "unit_variance")
        self.quantile_range: tuple[float, float] = _percentile_range(quantile_range, "RobustScaler")
        q_min, q_max = self.quantile_range
        if self.unit_variance and not 0.0 < q_min < q_max < 100.0:
            raise ValueError(
                f"RobustScaler: unit_variance needs quantile_range strictly inside 0..100 with q_min < q_max, "
                f"got {tuple(quantile_range)!r}"
            )
        if self.unit_variance and q_min / 100.0 == 0.0:
            raise ValueError(f"RobustScaler: unit_variance needs q_min / 100 above 0, and {q_min!r} / 100 rounds to 0")

    @override
    def params(self) -> dict[str, Any]:
        return {
            "columns": self.columns,
            "with_centering": self.with_centering,
            "with_scaling": self.with_scaling,
            "quantile_range": list(self.quantile_range),
            "unit_variance": self.unit_variance,
        }

    @property
    def center_(self) -> dict[str, float | None]:
        """The fitted median per column, or None where ``with_centering`` is False.

        Returns:
            Mapping of column name to its center.

        """
        return self._state()["center"]

    @property
    def scale_(self) -> dict[str, float | None]:
        """The fitted quantile width per column, or None where ``with_scaling`` is False.

        Returns:
            Mapping of column name to its scale.

        """
        return self._state()["scale"]


MAX_QUANTILES: int = 100_000


class QuantileTransformer(Transformer):
    """Numeric columns -> their rank in the fitted distribution, in place.

    ``QuantileTransformer(columns=["price"])`` fits ``n_quantiles`` values per
    column (the percentiles of its non-null fit values at
    ``linspace(0, 100, n_quantiles)``) against the shared grid
    ``references_ = linspace(0, 1, n_quantiles)``, and ``n_quantiles`` is
    capped at the fit row count, nulls included, as scikit-learn caps it.
    Transform interpolates a value into that grid forwards and backwards and
    averages the two, so a run of equal quantiles answers with the middle of
    its range rather than one end. With ``output_distribution="uniform"`` the
    result is in ``[0, 1]`` up to rounding: the fitted minimum maps to
    exactly 0, the maximum to exactly 1, and a value outside the fitted range
    is clipped to them, which makes this the one scaler that always flattens
    outliers rather than rescaling them. With ``"normal"`` the rank is then
    mapped through the inverse normal CDF and clipped to ``ppf(1e-7 - eps)``
    below and ``ppf(1 - (1e-7 - eps))`` above so a bound cannot map to
    infinity, and the pinning to 0 and 1 applies within ``1e-7`` of the
    fitted bounds (scikit-learn's ``BOUNDS_THRESHOLD``). A null is ignored at
    fit and passes through at transform, which is what scikit-learn's NaN
    does; a value that is NaN or infinity is an error. The column becomes f64
    in place, and an integer column is widened to f64. The grid needs every
    value of a column, so the fit keeps them all, never a subsample, and
    moves them to temporary files when they outgrow the fit's memory budget.
    """

    kernel_id: ClassVar[str] = "quantile_transform"

    def __init__(
        self,
        columns: list[str] | None = None,
        *,
        n_quantiles: int = 1000,
        output_distribution: str = "uniform",
    ) -> None:
        """Initialize the transformer.

        Args:
            columns: Columns to transform; None transforms every integer or
                float column found at fit.
            n_quantiles: How many points the fitted grid holds (1 to
                100,000), capped at the fit row count.
            output_distribution: "uniform" for a rank in [0, 1], or "normal"
                for that rank mapped through the inverse normal CDF.

        Raises:
            TypeError: If columns or n_quantiles has the wrong type.
            ValueError: If columns is empty, n_quantiles is not between 1 and
                100,000, or output_distribution is not one of the accepted
                values.

        """
        self.columns: list[str] | None = _optional_columns(columns, "QuantileTransformer")
        self.n_quantiles: int = _integer(n_quantiles, "QuantileTransformer: n_quantiles")
        if not 1 <= self.n_quantiles <= MAX_QUANTILES:
            raise ValueError(
                f"QuantileTransformer: n_quantiles must be between 1 and {MAX_QUANTILES}, got {n_quantiles!r}"
            )
        self.output_distribution: str = _one_of(
            output_distribution, _OUTPUT_DISTRIBUTIONS, "QuantileTransformer", "output_distribution"
        )

    @override
    def params(self) -> dict[str, Any]:
        return {
            "columns": self.columns,
            "n_quantiles": self.n_quantiles,
            "output_distribution": self.output_distribution,
        }

    @property
    def n_quantiles_(self) -> int:
        """How many points the fitted grid actually holds.

        Returns:
            ``min(n_quantiles, fit rows)``, at least 1.

        """
        return len(self._state()["references"])

    @property
    def references_(self) -> list[float]:
        """The shared reference grid, ``linspace(0, 1, n_quantiles_)``.

        Returns:
            The reference quantiles.

        """
        return self._state()["references"]

    @property
    def quantiles_(self) -> dict[str, list[float]]:
        """The fitted values per column, aligned with ``references_``.

        Returns:
            Mapping of column name to its quantile grid.

        """
        return self._state()["quantiles"]


MAX_OUTPUTS: int = 4096


class KBinsDiscretizer(Transformer):
    """Numeric columns -> the bin each value falls in.

    ``KBinsDiscretizer(columns=["price"], n_bins=4)`` fits the bin edges of
    each column and replaces its values with the bin they land in.
    ``strategy="uniform"`` spaces ``n_bins + 1`` edges evenly from the
    column's minimum to its maximum, ``"quantile"`` puts them at the
    percentiles of ``linspace(0, 100, n_bins + 1)`` under ``quantile_method``,
    and ``"kmeans"`` runs 1-D Lloyd's algorithm from a uniform grid and puts
    an edge midway between each pair of sorted cluster centers. Transform
    answers how many interior edges a value is at or past, so a value below
    the first edge is bin 0 and one past the last is the top bin. The
    ``"quantile"`` and ``"kmeans"`` strategies then drop every edge within
    ``1e-8`` of the one before it, as scikit-learn does, so a column can end
    up with fewer bins than ``n_bins`` asked for. A constant column is one
    bin holding everything. ``encode="ordinal"`` replaces the column with an
    i64 bin index in place. ``"onehot"`` and ``"onehot-dense"`` mean the same
    thing here, because the output is always dense columns: both drop the
    column and append one i64 column per bin, named ``<column>_<bin>``, after
    the columns that stay, so how many columns there are is known only after
    the fit and a later step in the same pipeline cannot name them, and one
    given no columns does not take them. A null is an error, at fit and at
    transform, because scikit-learn refuses one outright; so is a value that is
    NaN or infinity. An integer column is widened to f64, so a magnitude past
    2^53 rounds. ``"quantile"`` and ``"kmeans"`` keep every value and move them
    to temporary files when they outgrow the fit's memory budget, while
    ``"uniform"`` needs only the extremes and keeps nothing. Spilled k-means
    makes sequential passes and gives the same edges, bit for bit, as a fit
    held in memory.

    ``"kmeans"`` follows scikit-learn's Lloyd iteration down to its
    empty-cluster rules, which decide whole bin layouts on a column with few
    distinct values: a relocated point keeps the label the assignment gave
    it, a cluster still empty afterwards is placed at the biggest cluster's
    slot as that slot stands, and the centers returned are the means the
    stopping iteration computed. What this package cannot follow is
    scikit-learn's arbitrary choice among points that tie on distance from
    their center, which scikit-learn makes over the rows in file order and
    this package over the sorted values.

    ``strategy="uniform"`` over a column spanning a few ulps returns bin edges
    that repeat, as ``np.linspace`` and scikit-learn do; the saved pipeline
    keeps them, and the bin between two equal edges is empty.
    """

    kernel_id: ClassVar[str] = "k_bins_discretize"

    def __init__(
        self,
        columns: list[str] | None = None,
        *,
        n_bins: int = 5,
        encode: str = "onehot",
        strategy: str = "quantile",
        quantile_method: str = "averaged_inverted_cdf",
    ) -> None:
        """Initialize the discretizer.

        Args:
            columns: Columns to bin; None bins every integer or float column
                found at fit.
            n_bins: How many bins to ask for, from 2 to 4096.
            encode: "ordinal" for one bin-index column, or "onehot" /
                "onehot-dense" for one column per bin.
            strategy: "uniform", "quantile", or "kmeans".
            quantile_method: The numpy quantile method ``strategy="quantile"``
                uses, one of the nine scikit-learn accepts.

        Raises:
            TypeError: If columns or n_bins has the wrong type.
            ValueError: If columns is empty, n_bins is not between 2 and
                4096, or an option is not one of the accepted values.

        """
        self.columns: list[str] | None = _optional_columns(columns, "KBinsDiscretizer")
        self.n_bins: int = _integer(n_bins, "KBinsDiscretizer: n_bins")
        if not 2 <= self.n_bins <= MAX_OUTPUTS:
            raise ValueError(f"KBinsDiscretizer: n_bins must be between 2 and {MAX_OUTPUTS}, got {n_bins!r}")
        self.encode: str = _one_of(encode, _KBINS_ENCODINGS, "KBinsDiscretizer", "encode")
        self.strategy: str = _one_of(strategy, _KBINS_STRATEGIES, "KBinsDiscretizer", "strategy")
        self.quantile_method: str = _one_of(quantile_method, _QUANTILE_METHODS, "KBinsDiscretizer", "quantile_method")

    @override
    def params(self) -> dict[str, Any]:
        return {
            "columns": self.columns,
            "n_bins": self.n_bins,
            "encode": self.encode,
            "strategy": self.strategy,
            "quantile_method": self.quantile_method,
        }

    @property
    def bin_edges_(self) -> dict[str, list[float]]:
        """The fitted bin edges per column, ascending.

        Returns:
            Mapping of column name to its edges, outer bounds included.

        """
        return self._state()["bin_edges"]

    @property
    def n_bins_(self) -> dict[str, int]:
        """How many bins each column actually has after the too-small ones were dropped.

        Returns:
            Mapping of column name to its bin count.

        """
        return {name: len(edges) - 1 for name, edges in self.bin_edges_.items()}


def _monomial_name(columns: Sequence[str], monomial: tuple[int, ...]) -> str:
    """Name one monomial the way the kernel names it.

    Args:
        columns: The source column names, in the order the expansion uses.
        monomial: The column indices the monomial multiplies, ascending.

    Returns:
        ``a*b`` or ``a^2``, or ``1`` for the empty monomial.

    """
    if not monomial:
        return "1"
    parts: list[str] = []
    for index, group in itertools.groupby(monomial):
        exponent: int = len(list(group))
        parts.append(columns[index] if exponent == 1 else f"{columns[index]}^{exponent}")
    return "*".join(parts)


def _monomial_names(
    columns: Sequence[str], degree: int | tuple[int, int], interaction_only: bool, include_bias: bool
) -> list[str]:
    """Every generated column name, in scikit-learn's order.

    Args:
        columns: The source column names, in the order the expansion uses.
        degree: The maximum degree, or a (min, max) pair.
        interaction_only: Whether to drop monomials with an exponent above 1.
        include_bias: Whether to prepend the constant column.

    Returns:
        The generated names, in output order.

    Raises:
        ValueError: If the expansion would generate more than MAX_OUTPUTS columns.

    """
    minimum, maximum = (0, degree) if isinstance(degree, int) else degree
    pick = itertools.combinations if interaction_only else itertools.combinations_with_replacement
    names: list[str] = ["1"] if include_bias else []
    for total in range(max(1, minimum), maximum + 1):
        if interaction_only and total > len(columns):
            break
        for monomial in pick(range(len(columns)), total):
            names.append(_monomial_name(columns, monomial))
            if len(names) > MAX_OUTPUTS:
                raise ValueError(
                    f"PolynomialFeatures: {len(columns)} columns of degree {maximum} would generate more than "
                    f"the {MAX_OUTPUTS} columns allowed"
                )
    return names


class PolynomialFeatures(Transformer):
    """Numeric columns -> one column per monomial of them, replacing them.

    ``PolynomialFeatures(columns=["price", "height"])`` drops the two columns
    and appends ``1``, ``price``, ``height``, ``price^2``, ``price*height``
    and ``height^2`` after the columns that stay: every monomial of total
    degree at most ``degree`` over the listed columns, in scikit-learn's
    order (the bias first, then degree by degree, and within a degree by the
    order ``columns`` gives). ``degree=(min, max)`` keeps only the degrees in
    that range (the bias survives either way), and ``interaction_only=True``
    drops every monomial with an exponent above 1, leaving the products of
    distinct columns.

    The width is a function of the parameters alone, so this transformer
    never fits: it declares its outputs before any data is read, and a later
    step in the same pipeline can name them. scikit-learn writes a product as
    ``price height``; a space is a poor column name, so this package writes
    ``price*height``, and a source column whose name holds ``*``, ``^`` or a
    space is refused rather than generating a name nothing could parse. The
    bias column is named ``1``, as scikit-learn names it.

    Every factor is multiplied in scikit-learn's own order, so the products
    agree bit for bit. A null factor makes the monomial null; the bias has no
    factors and stays 1. Outputs are f64 whatever the inputs are; an integer
    column is read exactly or refused, and ``sparse_output`` and ``order``
    have no counterpart because the output here is always dense columns.

    A product that overflows to an infinity is an error naming the monomial,
    in both transform paths, rather than a column the next numeric step would
    refuse to read: ``a^2`` for ``a = 1e200`` raises where scikit-learn writes
    ``inf``.
    """

    kernel_id: ClassVar[str] = "polynomial_features"

    def __init__(
        self,
        columns: list[str],
        *,
        degree: int | tuple[int, int] = 2,
        interaction_only: bool = False,
        include_bias: bool = True,
    ) -> None:
        """Initialize the expansion.

        Args:
            columns: Columns to expand, in the order the monomials use; none
                of them survives to the output.
            degree: The maximum total degree, or a ``(min, max)`` pair of
                non-negative integers with ``min <= max``.
            interaction_only: Keep only products of distinct columns.
            include_bias: Prepend a constant column of ones named ``1``.

        Raises:
            TypeError: If columns is a bare string, degree is not an integer
                or a pair of them, or a flag is not a bool.
            ValueError: If columns is empty, a column name would make a
                generated name ambiguous, degree is negative or inverted,
                degree 0 is asked for without a bias, or the expansion would
                be wider than ``MAX_OUTPUTS`` (4096) columns.

        """
        self.columns: list[str] = _column_list(columns, "PolynomialFeatures")
        for column in self.columns:
            if set(column) & {"*", "^", " "}:
                raise ValueError(
                    f"PolynomialFeatures: column {column!r} holds one of '*', '^' or a space, "
                    "which would make a generated name ambiguous"
                )
        self.degree: int | tuple[int, int] = _degree(degree)
        self.interaction_only: bool = _flag(interaction_only, "PolynomialFeatures", "interaction_only")
        self.include_bias: bool = _flag(include_bias, "PolynomialFeatures", "include_bias")
        maximum: int = self.degree if isinstance(self.degree, int) else self.degree[1]
        if maximum == 0 and not include_bias:
            raise ValueError("PolynomialFeatures: degree 0 with include_bias False would generate no columns")
        names: list[str] = _monomial_names(self.columns, self.degree, self.interaction_only, self.include_bias)
        if include_bias and "1" in names[1:]:
            raise ValueError(
                "PolynomialFeatures: column '1' has the bias column's name; rename it or set include_bias=False"
            )

    @override
    def params(self) -> dict[str, Any]:
        return {
            "columns": self.columns,
            "degree": self.degree if isinstance(self.degree, int) else list(self.degree),
            "interaction_only": self.interaction_only,
            "include_bias": self.include_bias,
        }

    @property
    def output_columns(self) -> list[str]:
        """The columns this expansion writes, in output order.

        Known from the parameters alone, before any fit, because this
        transformer learns nothing.

        Returns:
            The generated column names.

        """
        return _monomial_names(self.columns, self.degree, self.interaction_only, self.include_bias)


def _degree(degree: object) -> int | tuple[int, int]:
    """Validate a polynomial degree: an integer, or a (min, max) pair.

    Args:
        degree: The value passed for ``degree``.

    Returns:
        The degree as an int or a pair of ints.

    Raises:
        TypeError: If the value is neither an integer nor a pair of them.
        ValueError: If a bound is negative or min exceeds max.

    """
    if isinstance(degree, (list, tuple)):
        if len(degree) != 2:
            raise TypeError(f"PolynomialFeatures: degree must be an integer or a (min, max) pair, got {degree!r}")
        low: int = _integer(degree[0], "PolynomialFeatures: degree")
        high: int = _integer(degree[1], "PolynomialFeatures: degree")
        if low < 0 or high < 0:
            raise ValueError(f"PolynomialFeatures: degree must not be negative, got {degree!r}")
        if low > high:
            raise ValueError(f"PolynomialFeatures: degree must have min <= max, got {degree!r}")
        return (low, high)
    value: int = _integer(degree, "PolynomialFeatures: degree")
    if value < 0:
        raise ValueError(f"PolynomialFeatures: degree must not be negative, got {degree!r}")
    return value


_EXTRAPOLATION: frozenset[str] = frozenset({"error", "constant", "linear", "continue", "periodic"})
_SPLINE_MISSING: frozenset[str] = frozenset({"error", "zeros"})


class SplineTransformer(Transformer):
    """Numeric columns -> a B-spline basis per column, from fitted knots.

    ``SplineTransformer(columns=["hour"], n_knots=5, degree=3)`` drops
    ``hour`` and appends ``hour_sp_0`` through ``hour_sp_6`` after the
    columns that stay: the ``n_knots + degree - 1`` B-spline basis functions
    of that degree over knots fitted to the column. ``knots="uniform"``
    places them at ``linspace(min, max, n_knots)``, which needs only the
    column's minimum and maximum; ``knots="quantile"`` places them at the
    percentiles of the column's non-null values, which the fit keeps and
    moves to temporary files when they outgrow the fit's memory budget.
    ``degree`` further knots are added beyond each end, spaced like the two
    knots at that end (under ``extrapolation="periodic"``, like the knots at
    the other end, so the two ends meet), and the basis is the Cox-de Boor
    recursion over that vector.

    A generated basis value that is not finite is an error naming the
    generated column, in both transform paths, rather than an ``inf`` the next
    numeric step would refuse to read. ``"continue"`` reaches one far outside
    the fitted range, and a knot vector spaced by a denormal reaches one
    inside it.

    ``extrapolation`` says what happens outside the fitted range:
    ``"error"`` refuses the value, ``"constant"`` holds the boundary basis
    values, ``"linear"`` continues them along the boundary derivative,
    ``"continue"`` evaluates the boundary polynomial pieces on, and
    ``"periodic"`` wraps the column onto the interval between the first and
    last knot (which needs ``degree < n_knots`` and gives ``n_knots - 1``
    columns). ``include_bias=False`` drops the last basis function per
    column. ``handle_missing="error"`` refuses a null, at transform as well
    as at fit; ``"zeros"`` encodes it as zeros in every generated column.

    The width comes from the parameters alone even though the knot positions
    are fitted, so a later step in the same pipeline can name the outputs.
    ``sparse_output`` and ``order`` have no counterpart because the output
    here is always dense columns, and ``knots`` as an explicit array is not
    accepted: knots here are always fitted.
    """

    kernel_id: ClassVar[str] = "spline_transform"

    def __init__(
        self,
        columns: list[str],
        *,
        n_knots: int = 5,
        degree: int = 3,
        knots: str = "uniform",
        extrapolation: str = "constant",
        include_bias: bool = True,
        handle_missing: str = "error",
    ) -> None:
        """Initialize the transformer.

        Args:
            columns: Columns to expand; none of them survives to the output.
            n_knots: How many knots to fit per column, from 2 to 4096.
            degree: The polynomial degree of the basis, from 0 to 4096.
            knots: Where to put them: "uniform" (min to max) or "quantile".
            extrapolation: What a value outside the fitted range becomes:
                "error", "constant", "linear", "continue" or "periodic".
            include_bias: Keep the last basis function per column.
            handle_missing: What a null becomes: "error" or "zeros".

        Raises:
            TypeError: If columns is a bare string, a count is not an
                integer, or include_bias is not a bool.
            ValueError: If columns is empty, a count is out of range, an
                option is not one of the accepted values, periodic
                extrapolation is asked for with degree at or above n_knots,
                the combination generates no columns at all, or the
                expansion would be wider than ``MAX_OUTPUTS`` (4096) columns.

        """
        self.columns: list[str] = _column_list(columns, "SplineTransformer")
        self.n_knots: int = _integer(n_knots, "SplineTransformer: n_knots")
        if not 2 <= self.n_knots <= MAX_OUTPUTS:
            raise ValueError(f"SplineTransformer: n_knots must be between 2 and {MAX_OUTPUTS}, got {n_knots!r}")
        self.degree: int = _integer(degree, "SplineTransformer: degree")
        if not 0 <= self.degree <= MAX_OUTPUTS:
            raise ValueError(f"SplineTransformer: degree must be between 0 and {MAX_OUTPUTS}, got {degree!r}")
        self.knots: str = _one_of(knots, frozenset({"uniform", "quantile"}), "SplineTransformer", "knots")
        self.extrapolation: str = _one_of(extrapolation, _EXTRAPOLATION, "SplineTransformer", "extrapolation")
        self.include_bias: bool = _flag(include_bias, "SplineTransformer", "include_bias")
        self.handle_missing: str = _one_of(handle_missing, _SPLINE_MISSING, "SplineTransformer", "handle_missing")
        if self.extrapolation == "periodic" and self.n_knots <= self.degree:
            raise ValueError(
                "SplineTransformer: periodic extrapolation needs degree below n_knots, got "
                f"n_knots {self.n_knots} and degree {self.degree}"
            )
        if self.splines < 1:
            raise ValueError(
                f"SplineTransformer: n_knots {self.n_knots}, degree {self.degree}, extrapolation "
                f"{self.extrapolation!r} and include_bias {self.include_bias} generate no columns at all"
            )
        if self.splines * len(self.columns) > MAX_OUTPUTS:
            raise ValueError(
                f"SplineTransformer: {len(self.columns)} columns of {self.splines} splines would generate more "
                f"than the {MAX_OUTPUTS} columns allowed"
            )

    @override
    def params(self) -> dict[str, Any]:
        return {
            "columns": self.columns,
            "n_knots": self.n_knots,
            "degree": self.degree,
            "knots": self.knots,
            "extrapolation": self.extrapolation,
            "include_bias": self.include_bias,
            "handle_missing": self.handle_missing,
        }

    @property
    def splines(self) -> int:
        """How many columns each input column becomes.

        Returns:
            ``n_knots + degree - 1`` (``n_knots - 1`` when periodic), one
            fewer without a bias.

        """
        total: int = self.n_knots - 1 if self.extrapolation == "periodic" else self.n_knots + self.degree - 1
        return total - (0 if self.include_bias else 1)

    @property
    def output_columns(self) -> list[str]:
        """The columns this transformer writes, in output order.

        Known from the parameters alone, before any fit; only the knot
        positions are fitted.

        Returns:
            The generated column names.

        """
        return [f"{column}_sp_{index}" for column in self.columns for index in range(self.splines)]

    @property
    def knots_(self) -> dict[str, list[float]]:
        """The fitted base knots per column, ascending.

        Returns:
            Mapping of column name to its ``n_knots`` knot positions.

        """
        return self._state()["knots"]


class PowerTransformer(Transformer):
    """Numeric columns -> the power of them that is most Gaussian, in place.

    ``PowerTransformer(columns=["price"])`` fits one ``lambda`` per column by
    maximizing the profile log-likelihood of the transformed column, then
    applies it: Yeo-Johnson takes any value, Box-Cox needs strictly positive
    data and refuses anything else, at transform as well as at fit. Box-Cox
    also refuses a constant column, whose likelihood has no maximum;
    Yeo-Johnson leaves one at ``lambda = 1``, the identity, as scikit-learn
    does.

    ``standardize=True`` then centers and scales each transformed column by
    its own mean and standard deviation. A null teaches nothing and passes
    through as a null, which is how scikit-learn treats a NaN.

    The solver is scipy's: Box-Cox brackets ``(-2, 2)`` and runs Brent's
    method, Yeo-Johnson runs Brent's method bounded by the interval scipy
    derives from the column's magnitude, both on scipy's own objective and at
    its own tolerance. Box-Cox then caps lambda as scipy does, so that no
    fitted value transforms to more than about 1.8e304 (the largest f64 divided
    by 10,000); scipy warns when it caps, and this package does not. The fit
    keeps every value, a log of each and a work array, and the solver rereads
    the last two on every iteration. Each moves to a temporary file when it
    outgrows the fit's memory budget, which makes the fit slower but leaves
    its result the same, bit for bit.

    Every sum is exact, so the fitted lambda does not depend on how the rows
    were split — but it can differ from scikit-learn's, because the last bits
    of the objective decide where a flat optimum lands. On the parity tests'
    well-conditioned columns the transformed values differ by at most 1.71e-6,
    and the tests allow 1e-5. Elsewhere nothing bounds the difference: on a
    near-constant column the likelihood is so flat that ``lambdas_`` can
    differ from scikit-learn's by 100 or more.
    """

    kernel_id: ClassVar[str] = "power_transform"

    def __init__(self, columns: list[str], *, method: str = "yeo-johnson", standardize: bool = True) -> None:
        """Initialize the transformer.

        Args:
            columns: Columns to transform, in place.
            method: "yeo-johnson" (any value) or "box-cox" (strictly positive).
            standardize: Center and scale each transformed column.

        Raises:
            TypeError: If columns is a bare string or standardize is not a bool.
            ValueError: If columns is empty or method is not one of the
                accepted values.

        """
        self.columns: list[str] = _column_list(columns, "PowerTransformer")
        self.method: str = _one_of(method, frozenset({"yeo-johnson", "box-cox"}), "PowerTransformer", "method")
        self.standardize: bool = _flag(standardize, "PowerTransformer", "standardize")

    @override
    def params(self) -> dict[str, Any]:
        return {"columns": self.columns, "method": self.method, "standardize": self.standardize}

    @property
    def lambdas_(self) -> dict[str, float]:
        """The fitted power per column.

        Returns:
            Mapping of column name to its lambda.

        """
        return self._state()["lambdas"]

    @property
    def means_(self) -> dict[str, float] | None:
        """The mean of each transformed column, or None without standardize.

        Returns:
            Mapping of column name to the center subtracted at transform.

        """
        return self._state()["means"]

    @property
    def scales_(self) -> dict[str, float] | None:
        """The standard deviation of each transformed column, or None without standardize.

        Returns:
            Mapping of column name to the scale divided out at transform.

        """
        return self._state()["scales"]
