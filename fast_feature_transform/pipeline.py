"""The Pipeline: composition in Python, execution in C++.

``fit`` hands the step tree to the C++ engine in one call. The engine
flattens it and fits it in streaming passes over the files: a step that
reads a column written by a step still fitting waits for a later pass, and
each pass recomputes the output of the steps already fitted rather than
storing it. ``transform`` dispatches on input type: a dict takes the record
path, and a list of Parquet paths takes the batch path.
"""

import json
import operator
import threading
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Self, SupportsIndex, overload

from fast_feature_transform import _core
from fast_feature_transform.transformers import (
    BackwardDifferenceEncoder,
    BaseNEncoder,
    Binarizer,
    BinaryEncoder,
    Cast,
    CatBoostEncoder,
    CountEncoder,
    GrayEncoder,
    GroupAgg,
    HashingEncoder,
    HelmertEncoder,
    JamesSteinEncoder,
    KBinsDiscretizer,
    LabelEncoder,
    LeaveOneOutEncoder,
    MaxAbsScaler,
    MEstimateEncoder,
    MinMaxScaler,
    Normalizer,
    OneHotEncoder,
    OrdinalEncoder,
    PolynomialEncoder,
    PolynomialFeatures,
    PowerTransformer,
    QuantileEncoder,
    QuantileTransformer,
    RankHotEncoder,
    RobustScaler,
    Select,
    SplineTransformer,
    StandardScaler,
    SumEncoder,
    SummaryEncoder,
    TargetEncoder,
    Transformer,
    WOEEncoder,
    _cached_manifest,
    _check_threads,
    _directory_path,
    _fit_engine,
    _output_directory,
    _ParquetPaths,
    _PathT,
    _source_paths,
    _StrPath,
)

TransformInput = dict[str, Any] | _ParquetPaths[_PathT]
TransformOutput = dict[str, Any] | list[Path]

Step = tuple[str, "Transformer | Pipeline"]

# A pipeline's view of one fit: the fitted C++ pipeline, this pipeline's
# dotted stage-name prefix in it ("" for a root), and the manifest of an
# unpickled pipeline that is not loaded yet. An unfitted pipeline holds
# (None, "", None), an unpickled one (None, prefix, manifest) and a bound one
# (handle, prefix, None). A bind publishes the whole view in one assignment
# and each reader takes everything it uses from one read of it, so a thread
# that uses this pipeline while another fits it, or fits a pipeline that
# holds it, sees the old fit or the new one, never the handle of one with
# the prefix of the other.
_PipelineBinding = tuple[_core.FittedPipeline | None, str, str | None]

# One fit's reading of a step tree: each pipeline in the tree, by its id,
# with the step list the fit read from it. The fit compiles these lists and
# binds these lists.
_Reading = dict[int, tuple["Pipeline", list[Step]]]

_SHELLS: dict[str, type[Transformer]] = {
    "select": Select,
    "cast": Cast,
    "label_encode": LabelEncoder,
    "group_agg": GroupAgg,
    "count_encode": CountEncoder,
    "ordinal_encode": OrdinalEncoder,
    "one_hot_encode": OneHotEncoder,
    "rank_hot_encode": RankHotEncoder,
    "hash_encode": HashingEncoder,
    "target_encode": TargetEncoder,
    "m_estimate_encode": MEstimateEncoder,
    "woe_encode": WOEEncoder,
    "james_stein_encode": JamesSteinEncoder,
    "leave_one_out_encode": LeaveOneOutEncoder,
    "cat_boost_encode": CatBoostEncoder,
    "quantile_encode": QuantileEncoder,
    "summary_encode": SummaryEncoder,
    "basen_encode": BaseNEncoder,
    "binary_encode": BinaryEncoder,
    "gray_encode": GrayEncoder,
    "backward_difference_encode": BackwardDifferenceEncoder,
    "helmert_encode": HelmertEncoder,
    "sum_encode": SumEncoder,
    "polynomial_encode": PolynomialEncoder,
    "standard_scale": StandardScaler,
    "min_max_scale": MinMaxScaler,
    "max_abs_scale": MaxAbsScaler,
    "binarize": Binarizer,
    "normalize": Normalizer,
    "robust_scale": RobustScaler,
    "quantile_transform": QuantileTransformer,
    "k_bins_discretize": KBinsDiscretizer,
    "polynomial_features": PolynomialFeatures,
    "spline_transform": SplineTransformer,
    "power_transform": PowerTransformer,
}


def parquet_to_memmap(X: _ParquetPaths[_PathT], *, output_dir: _StrPath | None = None, threads: int = 0) -> Path:
    """Convert Parquet files into one memmap: ``data.mmap`` plus ``index.json``.

    Meant for ``transform_parquet``'s output. The file holds row-major
    packed records, one field per column (the layout of a numpy structured
    dtype), preallocated from the Parquet row counts and written in
    parallel; ``index.json`` records the row count, record size, byte order,
    and each field's name, numpy dtype, size, and byte offset. Every column
    becomes a field, so strings must already be encoded and nulls filled. A
    vector column becomes a field of ``size`` elements, whether it is stored
    as a fixed-size list or, as Parquet writers usually store one, as a plain
    list holding the same number of values in every row.

    Args:
        X: A list or tuple of Parquet file paths.
        output_dir: Directory for ``data.mmap`` and ``index.json``; a temporary
            directory is created when omitted and removed with its files if
            the call raises. A previous memmap there is replaced only after
            every row is written, so a run that fails before then leaves it
            untouched.
        threads: Worker threads; 0 uses all cores.

    Returns:
        The path of ``index.json``.

    Raises:
        EngineError: If the input cannot be read or holds a string column, a
            null, a list column whose rows differ in length or are empty, or
            a type a memmap cannot store.
        TypeError: If X is a single path instead of a list of paths or holds
            an os.PathLike that gives bytes, or output_dir is not None and is
            neither a str nor an os.PathLike that gives a str.
        ValueError: If threads is not an integer from 0 to 2**32 - 1,
            output_dir is empty or holds a NUL character, or a path in X or
            output_dir cannot be encoded as UTF-8.

    """
    sources: list[str] = _source_paths(X)
    count: int = _check_threads(threads)
    with _output_directory(output_dir) as out_dir:
        return Path(_core.parquet_to_memmap(sources, out_dir, count))


def _validated_steps(steps: Iterable[Step]) -> list[Step]:
    """Materialize and validate a step list.

    Args:
        steps: Ordered (name, transformer-or-pipeline) pairs.

    Returns:
        The steps as a new list, each name a plain ``str``.

    Raises:
        TypeError: If a step is not a (name, Transformer or Pipeline instance) pair.
        ValueError: On an empty, dotted, or duplicated step name.

    """
    materialized: list[Step] = list(steps)  # a one-shot iterable must be consumed exactly once
    seen: set[str] = set()
    plain: bool = True
    for step in materialized:
        if not (isinstance(step, tuple) and len(step) == 2 and isinstance(step[0], str)):
            raise TypeError(f"each step must be a (name, transformer) pair, got {step!r}")
        name, transformer = step
        if type(name) is not str:
            # A name given as a str subclass, such as an Enum with a str
            # mixin, is stored as the plain string it equals. A nested step's
            # stage name is built by formatting the step names, and such a
            # member formats as "ClassName.MEMBER", which names no stage.
            name = str.__str__(name)
            plain = False
        if not isinstance(transformer, (Transformer, Pipeline)):
            raise TypeError(f"step {name!r} must be a Transformer or Pipeline instance, got {transformer!r}")
        if not name or "." in name:
            raise ValueError(f"invalid step name {name!r}: must be non-empty and must not contain '.'")
        if name in seen:
            raise ValueError(f"duplicate step name {name!r}")
        seen.add(name)
    # Steps whose names are all plain strings are kept as given rather than
    # rebuilt pair by pair, which costs a new pair for every step.
    if plain:
        return materialized
    return [(str.__str__(name), transformer) for name, transformer in materialized]


class Pipeline:
    """An ordered list of named transformer steps, executed by the C++ engine."""

    def __init__(self, steps: Iterable[Step]) -> None:
        """Initialize the pipeline.

        Args:
            steps: Ordered (name, transformer-or-pipeline) pairs. Names must
                be unique, non-empty, and must not contain ``.`` (dots are the
                nesting separator in stage names).

        Raises:
            TypeError: If a step is not a (name, Transformer or Pipeline instance) pair.
            ValueError: On an empty, dotted, or duplicated step name.

        """
        self._steps: list[Step] = _validated_steps(steps)
        self._binding: _PipelineBinding = (None, "", None)
        self._bind_lock: threading.Lock = threading.Lock()
        # The fits that have read the steps and not yet returned or raised;
        # the steps setter refuses while there is one. Changed under
        # _bind_lock, as _steps is.
        self._fitting: int = 0

    @property
    def steps(self) -> list[Step]:
        """The (name, step) pairs, fitted when the pipeline is.

        A new list every time, holding the same step objects: the fitted plan
        lives in C++ and cannot be edited from Python, so a list that reached
        back into the pipeline would let ``steps.append(...)`` change what the
        pipeline says it does without changing what it runs. Change the steps
        by assigning to this property, which validates them, and which a
        pipeline that is fitted or being fitted refuses.

        Returns:
            The step list.

        """
        self._ensure_bound()
        return list(self._steps)

    @steps.setter
    def steps(self, steps: Iterable[Step]) -> None:
        """Replace the steps of a pipeline that is neither fitted nor being fitted.

        A fit reads the steps of its pipeline and of every pipeline nested in
        it once, as it starts, and compiles and binds what it read. An
        assignment on another thread either lands before that read, and the
        fit runs the new steps, or raises.

        Args:
            steps: Ordered (name, transformer-or-pipeline) pairs.

        Raises:
            RuntimeError: If this pipeline is fitted, or is being fitted: a
                fit of this pipeline, or of a pipeline that holds it, has read
                the steps of this pipeline and has not yet returned or raised.
                The fitted plan and every transformer bound to it belong to
                the steps that were compiled; swapping the Python list would
                leave the pipeline describing one plan and executing another.
                Build a new Pipeline and fit that.
            TypeError: If a step is not a (name, Transformer or Pipeline instance) pair.
            ValueError: On an empty, dotted, or duplicated step name.

        """
        # Checked before the new steps are read, so that a fitted pipeline
        # refuses whatever it is given, and again under the lock that a fit
        # reads the steps under, so that the assignment raises when a fit read
        # them in between.
        self._refuse_new_steps()
        validated: list[Step] = _validated_steps(steps)
        with self._bind_lock:
            self._refuse_new_steps()
            self._steps = validated

    def fit(
        self,
        X: _ParquetPaths[_PathT],
        y: Any = None,
        *,
        threads: int = 0,
        temp_dir: _StrPath | None = None,
    ) -> Self:
        """Fit on the given Parquet files.

        A pipeline fits all its steps, nested ones included, in one engine
        call. Fitting again, or fitting a pipeline that holds this object,
        replaces its fitted state at once: a thread that uses it meanwhile
        sees the old fit or the new one.

        Fit reads Parquet in batches. Steps that need every value (exact
        medians, quantiles, k-means and power fits) keep those values within a
        64 MiB budget per fit and move them to temporary files when they
        outgrow it, trading disk space and I/O time for memory. The budget
        covers those values only: fitted tables, batches and Parquet decoder
        buffers take memory too. Nothing is sampled.

        Args:
            X: A list or tuple of Parquet file paths.
            y: Ignored; present to match scikit-learn's ``fit(X, y)``.
            threads: Worker threads for the engine; 0 uses all cores.
            temp_dir: Existing parent directory for private temporary spill files;
                defaults to the system temporary directory. The files are
                deleted when the fit ends or fails.

        Returns:
            self, fitted.

        Raises:
            EngineError: If the input cannot be read or a column is missing or mistyped.
            TypeError: If X is a single path instead of a list of paths or
                holds an os.PathLike that gives bytes, or temp_dir is not None
                and is neither a str nor an os.PathLike that gives a str.
            ValueError: If threads is not an integer from 0 to 2**32 - 1,
                temp_dir holds a NUL character or is not an existing
                directory, or a path in X or temp_dir cannot be encoded as
                UTF-8.

        """
        # REGRESSION: the fit compiles and binds one reading of the steps, and
        # every pipeline in the tree refuses new steps from that reading until
        # the fit returns or raises. The steps used to be read again for the
        # bind, after the engine returned, and the setter refused only a
        # fitted pipeline: steps assigned in between were bound to the plan
        # compiled from the old ones, and the pipeline listed one plan and ran
        # another.
        reading: _Reading = {}
        try:
            plan: str = json.dumps(self._to_tree(reading))
            engine: _core.Engine = _fit_engine(threads, temp_dir)
            self._adopt(engine.fit(plan, _source_paths(X)), reading)
        finally:
            for pipeline, _ in reading.values():
                with pipeline._bind_lock:
                    pipeline._fitting -= 1
        return self

    def transform_dict(self, X: dict[str, Any]) -> dict[str, Any]:
        """Transform a single record through the record path.

        The record path takes no binary value: a ``bytes`` value raises
        EngineError, even for a binary column that the batch path passes
        through, so a record leaves out the keys of such columns.

        Args:
            X: One record as a flat dict of str, int, float, bool, datetime,
                or None values (numpy scalars are converted, and a datetime
                becomes i64 microseconds since 1970-01-01).

        Returns:
            The transformed record.

        Raises:
            EngineError: If a value cannot be converted, a required key is missing, or the record holds one key twice.
            RuntimeError: If unfitted, or fitted as a step of a pipeline.
            TypeError: If X is not a dict.

        """
        # The record path: a record costs a couple of hundred nanoseconds, and
        # the two frames _require_handle adds are a large share of that.
        # Anything but a bound root still goes through it, to load a deferred
        # artifact or to raise.
        handle, prefix, _ = self._binding
        if handle is None or prefix:
            handle = self._require_handle()
        return handle.transform_row(X)

    def transform_parquet(self, X: _ParquetPaths[_PathT], *, output_dir: _StrPath | None = None) -> list[Path]:
        """Transform Parquet files through the streaming batch path.

        Args:
            X: A list or tuple of Parquet file paths.
            output_dir: Directory for output part files; a temporary directory
                is created when omitted and removed with its files if the
                call raises. Once the run starts, the files named
                ``part-<number>.parquet`` whose number has five or more
                digits, and the same names ending in ``.tmp``, are deleted,
                and a run that fails midway leaves none. Each part number has
                the width of the last part number, with at least five digits.
                Names are compared with their case: on a volume that ignores
                case, a file whose name differs from a part's only in case is
                not deleted, and a run that writes that part writes over it.

        Returns:
            The paths of the part files written, one per input row group, in
            source order.

        Raises:
            EngineError: If the input cannot be read, does not match the fitted schema, or
                has rows while the pipeline emits no columns, which Parquet output
                cannot hold; ``transform_dict`` returns them.
            RuntimeError: If unfitted, or fitted as a step of a pipeline.
            TypeError: If X is a single path instead of a list of paths or
                holds an os.PathLike that gives bytes, or output_dir is not
                None and is neither a str nor an os.PathLike that gives a str.
            ValueError: If output_dir is empty or holds a NUL character, or a
                path in X or output_dir cannot be encoded as UTF-8.

        """
        handle: _core.FittedPipeline = self._require_handle()
        sources: list[str] = _source_paths(X)
        with _output_directory(output_dir) as out_dir:
            return [Path(path) for path in handle.transform_batch(sources, out_dir)]

    def transform_memmap(self, X: _ParquetPaths[_PathT], *, output_dir: _StrPath) -> Path:
        """Transform Parquet files into one memmap: ``data.mmap`` plus ``index.json``.

        The records go from the transformed batches into the memmap without
        intermediate Parquet files.

        Args:
            X: Input Parquet files, in their desired row order.
            output_dir: Directory for ``data.mmap`` and ``index.json``.

        Returns:
            The path of ``index.json``. The memmap has the layout that
            parquet_to_memmap writes.

        Raises:
            EngineError: If the input cannot be read or does not match the
                fitted schema, or the output has no columns or holds a string
                column, a null, a list column whose rows differ in length or
                are empty, or a type a memmap cannot store.
            RuntimeError: If unfitted, or fitted as a step of a pipeline.
            TypeError: If X is a single path instead of a list of paths or
                holds an os.PathLike that gives bytes, or output_dir is
                neither a str nor an os.PathLike that gives a str.
            ValueError: If output_dir is empty or holds a NUL character, or a
                path in X or output_dir cannot be encoded as UTF-8.

        """
        return Path(
            self._require_handle().transform_memmap(_source_paths(X), _directory_path(output_dir, "output_dir"))
        )

    @overload
    def transform(self, X: dict[str, Any]) -> dict[str, Any]: ...

    @overload
    def transform(self, X: _ParquetPaths[_PathT], *, output_dir: _StrPath | None = None) -> list[Path]: ...

    def transform(self, X: TransformInput[_PathT], *, output_dir: _StrPath | None = None) -> TransformOutput:
        """Transform a record dict or a list of Parquet paths.

        Dispatches on input type: a dict goes to transform_dict, and a list or
        tuple of paths to transform_parquet.

        Args:
            X: A record dict, or a list or tuple of Parquet file paths.
            output_dir: Directory for batch output part files; a temporary
                directory is created when omitted and removed with its files
                if the call raises.

        Returns:
            A transformed record dict, or the list of written part-file paths.

        Raises:
            EngineError: If the engine rejects the input.
            RuntimeError: If unfitted, or fitted as a step of a pipeline.
            TypeError: If X is neither a dict nor a list of paths, X holds an
                os.PathLike that gives bytes, or output_dir is not None and is
                neither a str nor an os.PathLike that gives a str.
            ValueError: If output_dir is empty or holds a NUL character, or a
                path in X or output_dir cannot be encoded as UTF-8.

        """
        if isinstance(X, dict):
            return self.transform_dict(X)
        if isinstance(X, (list, tuple)):
            return self.transform_parquet(X, output_dir=output_dir)
        raise TypeError(f"unsupported input type: {type(X).__name__}; expected a dict or a list of Parquet paths")

    def fit_transform(
        self,
        X: _ParquetPaths[_PathT],
        y: Any = None,
        *,
        threads: int = 0,
        temp_dir: _StrPath | None = None,
        output_dir: _StrPath | None = None,
    ) -> list[Path]:
        """Fit the pipeline, then transform the same files with transform_parquet.

        Args:
            X: A list or tuple of Parquet file paths.
            y: Ignored; present to match scikit-learn's ``fit_transform(X, y)``.
            threads: Worker threads for the engine; 0 uses all cores.
            temp_dir: Existing parent directory for private temporary spill files;
                defaults to the system temporary directory. The files are
                deleted when the fit ends or fails.
            output_dir: Directory for output part files; a temporary directory
                is created when omitted and removed with its files if the
                call raises.

        Returns:
            The list of written part-file paths.

        Raises:
            EngineError: If fit or transform_parquet raises it.
            TypeError: If X is a single path instead of a list of paths or
                holds an os.PathLike that gives bytes, or temp_dir or
                output_dir is not None and is neither a str nor an os.PathLike
                that gives a str.
            ValueError: If threads is not an integer from 0 to 2**32 - 1,
                output_dir is empty, temp_dir or output_dir holds a NUL
                character, temp_dir is not an existing directory, or a path in
                X, temp_dir or output_dir cannot be encoded as UTF-8. Both
                directories are checked before the fit starts.

        """
        if output_dir is not None:
            output_dir = _directory_path(output_dir, "output_dir")
        return self.fit(X, y, threads=threads, temp_dir=temp_dir).transform_parquet(X, output_dir=output_dir)

    def save(self, path: _StrPath) -> None:
        """Write the fitted pipeline to ``<path>/manifest.json``, which ``Pipeline.load`` reads.

        Args:
            path: Directory for ``manifest.json``, created if missing. A
                ``manifest.json`` already there is replaced.

        Raises:
            EngineError: If the directory or manifest cannot be written.
            RuntimeError: If unfitted, or if this pipeline is a step of a larger one.
            TypeError: If path is neither a str nor an os.PathLike that gives a str.
            ValueError: If path is empty, holds a NUL character, or cannot be
                encoded as UTF-8.

        """
        self._require_handle().save(_directory_path(path, "path"))

    def stages_json(self) -> str:
        """Describe the fitted steps in execution order, as JSON text.

        One object per stage, ``{"stage": ..., "kernel": ..., "params": ...}``,
        where ``stage`` is the dotted name of the step in the nested tree
        (``"features.encode"``) and ``params`` are the step's parameters.
        Fitted state is left out, so this stays cheap however large the
        vocabularies are.

        Returns:
            A JSON array.

        Raises:
            RuntimeError: If unfitted, or if this pipeline is a step of a larger one.

        """
        return self._require_handle().stages_json()

    def stages_state(self) -> str:
        """Return what every step learned in fit, as JSON text.

        One JSON object maps each dotted stage name to the fitted state that
        ``save`` writes for it: the same values fitted attributes such as
        ``label_maps_`` read. A step that learns nothing maps to ``{}``. The
        keys are written in execution order, which ``json.loads`` keeps; a
        parser that does not keep object order can take it from
        ``stages_json``. The text grows with the vocabularies.

        Returns:
            A JSON object.

        Raises:
            RuntimeError: If unfitted, or if this pipeline is a step of a larger one.

        """
        return self._require_handle().stages_state()

    @classmethod
    def load(cls, path: _StrPath, *, threads: int = 0) -> Pipeline:
        """Load a fitted pipeline from the directory ``save`` wrote.

        The nested step structure is rebuilt from the manifest's dotted stage
        names, so ``pipeline["outer"]["inner"]`` access works after loading.

        Args:
            path: The directory holding ``manifest.json``.
            threads: Worker threads for batch transforms; 0 uses all cores.

        Returns:
            A fitted pipeline with the original nesting shape.

        Raises:
            EngineError: If the manifest is missing or invalid, or its layout
                (``schema_version``) is one this build does not read.
            TypeError: If path is neither a str nor an os.PathLike that gives a str.
            ValueError: If path is empty, holds a NUL character or cannot be
                encoded as UTF-8, or threads is not an integer from 0 to
                2**32 - 1.

        """
        return cls._from_handle(_core.FittedPipeline.load(_directory_path(path, "path"), _check_threads(threads)))

    def __getitem__(self, key: SupportsIndex | str) -> Transformer | Pipeline:
        """Look up a step by name or position.

        Args:
            key: The step name given at construction, or its index as a list
                reads one: an int, a numpy integer, or any other value that
                ``operator.index`` takes. A bool is such a value, so ``True``
                is index 1.

        Returns:
            The step's transformer or nested pipeline.

        Raises:
            IndexError: If the index is out of range.
            KeyError: If no step has the given name, or the key is neither a
                str nor a value that ``operator.index`` takes.

        """
        steps: list[Step] = self.steps
        if not isinstance(key, str):
            try:
                index: int = operator.index(key)
            except TypeError:
                raise KeyError(key) from None
            return steps[index][1]
        for step_name, step in steps:
            if step_name == key:
                return step
        raise KeyError(key)

    def __len__(self) -> int:
        """Number of steps.

        Returns:
            The step count.

        """
        return len(self._steps)

    def __repr__(self) -> str:
        """Render the pipeline as its constructor call.

        Returns:
            ``Pipeline(steps=[...])``.

        """
        return f"Pipeline(steps={self._steps!r})"

    def __copy__(self) -> Pipeline:
        """Shallow copy: shares the steps and the fitted handle, with its own bind lock.

        Returns:
            A pipeline over the same steps and artifact.

        """
        clone: Pipeline = type(self).__new__(type(self))
        clone.__dict__.update(self.__dict__)
        clone._bind_lock = threading.Lock()
        # The copy is not being fitted: a fit of the original takes its count
        # back from the original only.
        clone._fitting = 0
        return clone

    def __getstate__(self) -> dict[str, Any]:
        """Pickle as the artifact: the manifest is the only serialized form.

        The manifest string is serialized once per fitted handle and shared by
        every pipeline bound to it, so pickle stores one copy however deep the
        nesting. A nested pipeline pickled on its own restores as a view over
        the whole fitted artifact, with its stage prefix preserved.

        Returns:
            State dict holding the steps, manifest JSON, and stage prefix.

        """
        # REGRESSION: the manifest and the prefix come from one read of the
        # binding. The handle, the prefix and the manifest of an unpickled
        # pipeline used to be read one after another, so a bind on another
        # thread between two of the reads left the pickle with neither the
        # handle nor the manifest, and it restored unfitted, or with the
        # manifest of one fit and the prefix of the other.
        handle, prefix, manifest = self._binding
        if handle is not None:
            manifest = _cached_manifest(handle)
        return {"steps": self._steps, "manifest": manifest, "prefix": prefix}

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore from the pickled artifact manifest.

        The artifact is loaded lazily, on first use, and a root binds every
        nested step at once, so nested pipelines inside it never load their
        own copy.

        Args:
            state: State dict produced by __getstate__.

        """
        self._steps = state["steps"]
        self._binding = (None, state.get("prefix", ""), state["manifest"])
        self._bind_lock = threading.Lock()
        self._fitting = 0

    def _ensure_bound(self) -> _PipelineBinding:
        """Load a deferred (unpickled) saved pipeline and bind every step to it, once.

        Returns:
            The binding, read once after any load, so it holds no manifest.

        """
        binding: _PipelineBinding = self._binding
        if binding[2] is None:  # published last by _bind_stages, so a bound tree is fully bound
            return binding
        with self._bind_lock:
            binding = self._binding
            if binding[2] is not None:
                binding = self._bind_stages(_core.FittedPipeline.load_json(binding[2]), prefix=binding[1], reading=None)
            return binding

    def _refuse_new_steps(self) -> None:
        """Raise if this pipeline is fitted or being fitted.

        Raises:
            RuntimeError: If it is.

        """
        # Checked without _ensure_bound: an unpickled pipeline is fitted even
        # before its manifest is loaded, and refusing must not load it just to
        # refuse.
        handle, _, manifest = self._binding
        if handle is not None or manifest is not None:
            raise RuntimeError(
                "Pipeline is already fitted, so its steps cannot be reassigned; "
                "build a new Pipeline with the steps you want and fit that"
            )
        if self._fitting:
            raise RuntimeError(
                "Pipeline is being fitted, so its steps cannot be reassigned; "
                "build a new Pipeline with the steps you want and fit that"
            )

    def _to_tree(self, reading: _Reading) -> dict[str, Any]:
        """Read the step tree for a fit and serialize it for the C++ plan compiler.

        Each pipeline in the tree is read once, under its bind lock, and
        counts the fit in ``_fitting`` from then on, which refuses new steps;
        the fit takes the count back when it returns or raises.

        Args:
            reading: Collects each pipeline read, with its steps.

        Returns:
            The nested plan tree as JSON-compatible data.

        """
        with self._bind_lock:
            if id(self) not in reading:
                reading[id(self)] = (self, self._steps)
                self._fitting += 1
        serialized_steps: list[list[Any]] = []
        for name, step in reading[id(self)][1]:
            if isinstance(step, Pipeline):
                serialized_steps.append([name, step._to_tree(reading)])
            else:
                serialized_steps.append([name, {"kernel": step.kernel_id, "params": step.params()}])
        return {"type": "pipeline", "steps": serialized_steps}

    def _adopt(self, handle: _core.FittedPipeline, reading: _Reading | None) -> None:
        """Take ownership of a fitted C++ handle and bind every stage's shell.

        Args:
            handle: The fitted C++ pipeline.
            reading: The steps the plan was compiled from, or None to bind
                the steps of a loaded pipeline, which cannot change.

        """
        with self._bind_lock:
            self._bind_stages(handle, prefix="", reading=reading)

    def _bind_stages(self, handle: _core.FittedPipeline, prefix: str, reading: _Reading | None) -> _PipelineBinding:
        """Recursively give each transformer, then this pipeline, its fitted-state view.

        Args:
            handle: The fitted C++ pipeline.
            prefix: Dotted stage-name prefix for this nesting level.
            reading: The steps the plan was compiled from, or None to bind
                the steps of an unpickled or loaded pipeline, which cannot
                change.

        Returns:
            The binding it published.

        """
        steps: list[Step] = self._steps if reading is None else reading[id(self)][1]
        for name, step in steps:
            stage: str = name if prefix == "" else f"{prefix}.{name}"
            if isinstance(step, Pipeline):
                step._bind_stages(handle, prefix=stage, reading=reading)
            else:
                step._bind(handle, stage, standalone=False)
        # REGRESSION: the handle, the prefix and the cleared manifest are
        # published in one assignment, after every step: nothing observes a
        # half-bound tree. They used to be three attributes written one after
        # another, so a thread that used, copied or pickled a pipeline that
        # was a root and a step of another between two of the writes got the
        # handle of one fit with the prefix of the other, and the step
        # answered for the whole larger pipeline.
        binding: _PipelineBinding = (handle, prefix, None)
        self._binding = binding
        return binding

    def _require_handle(self) -> _core.FittedPipeline:
        """Return the fitted handle or raise if this pipeline cannot run on its own.

        Returns:
            The fitted C++ pipeline handle.

        Raises:
            RuntimeError: If the pipeline has not been fitted, or is a nested
                step of a larger pipeline (its stages cannot run in isolation).

        """
        binding: _PipelineBinding = self._binding
        if binding[2] is not None and not binding[1]:
            # A nested pipeline is refused without loading what it would refuse.
            binding = self._ensure_bound()
        handle, prefix, _ = binding
        if prefix:
            raise RuntimeError(f"Pipeline {prefix!r} is a step of a larger pipeline; call this on the root pipeline")
        if handle is None:
            raise RuntimeError("Pipeline is not fitted")
        return handle

    @classmethod
    def _from_handle(cls, handle: _core.FittedPipeline) -> Pipeline:
        """Rebuild the nested declarative tree from a loaded handle's stages.

        Args:
            handle: A fitted C++ pipeline loaded from an artifact.

        Returns:
            A fitted Pipeline whose nesting mirrors the dotted stage names.

        """
        pipeline: Pipeline = cls(cls._steps_from_ops(json.loads(handle.stages_json())))
        pipeline._adopt(handle, None)
        return pipeline

    @classmethod
    def _steps_from_ops(cls, ops: list[dict[str, Any]]) -> list[Step]:
        """Group flat manifest ops back into a nested step tree by stage prefix.

        Args:
            ops: Op entries whose "stage" fields are dotted names.

        Returns:
            Steps with nested Pipelines rebuilt (flatten order is depth-first,
            so ops sharing a prefix are always consecutive).

        Raises:
            ValueError: If an op names a kernel this package has no transformer for.

        """
        steps: list[Step] = []
        index: int = 0
        while index < len(ops):
            head, dot, _ = ops[index]["stage"].partition(".")
            if not dot:
                op = ops[index]
                shell_cls = _SHELLS.get(op["kernel"])
                if shell_cls is None:
                    raise ValueError(
                        f"the saved pipeline uses kernel {op['kernel']!r}, which has no transformer class here"
                    )
                steps.append((head, shell_cls(**op["params"])))
                index += 1
                continue
            nested: list[dict[str, Any]] = []
            while index < len(ops) and ops[index]["stage"].partition(".")[0] == head and "." in ops[index]["stage"]:
                inner = dict(ops[index])
                inner["stage"] = ops[index]["stage"].partition(".")[2]
                nested.append(inner)
                index += 1
            steps.append((head, cls(cls._steps_from_ops(nested))))
        return steps
