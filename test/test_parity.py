"""Value-for-value parity with category_encoders and scikit-learn, from fixtures the real libraries generated.

Regenerate with scripts/regen_parity_fixtures.py and
scripts/regen_sklearn_fixtures.py; see their docstrings for the pinned
commands. Types are compared numerically, not by dtype: pandas widens a column
to float64 as soon as one unknown appears, while our types are declared at fit
and stay put. A supervised category_encoders case names its target column,
which upstream takes as a separate ``y`` and we keep as an ordinary column, so
our output holds exactly that one column more. A case carrying ``raised``
instead of ``expected`` is one the library refuses at transform: we have to
refuse it through both paths rather than encode anything.

The scikit-learn cases work the other way round on columns: scikit-learn
transforms whatever frame it is handed, so its fixture was generated on the
columns the case names and our output carries the rest of the dataset through
untouched. Every expected column has to be there and to agree value for value;
a column only we have has to be one of the dataset's, never one we invented.

Every case with ``expected`` values is held to four things, whichever library
it came from. Values are compared for equality of the doubles unless the case
declares a tolerance (``_same``); the output schema is compared as ordered
sequences with nothing extra (``_assert_ordered_schema``); every record has to
come back from transform_dict as its batch row, keys in the same order; and
the whole case is transformed a second and a third time, through an artifact
save/load and through a pickle, because a fitted step that comes back meaning
something else is a defect nothing else here would catch.
"""

import json
import math
import pickle
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import fast_feature_transform
from fast_feature_transform import Transformer
from fast_feature_transform.pipeline import Pipeline

FIXTURE: dict[str, Any] = json.loads((Path(__file__).parent / "fixtures" / "category_encoders.json").read_text())
CASES: list[dict[str, Any]] = FIXTURE["cases"]

SKLEARN: dict[str, Any] = json.loads((Path(__file__).parent / "fixtures" / "sklearn_preprocessing.json").read_text())
SKLEARN_CASES: list[dict[str, Any]] = SKLEARN["cases"]


def _column(values: list[Any]) -> pa.Array:
    """Build one fixture column, keeping numbers numeric.

    Args:
        values: The column's values, nulls included.

    Returns:
        The values as an Arrow array of their natural type.

    """
    present: list[Any] = [value for value in values if value is not None]
    if present and all(isinstance(value, int) and not isinstance(value, bool) for value in present):
        return pa.array(values, pa.int64())
    if present and all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in present):
        return pa.array(values, pa.float64())
    return pa.array(values, pa.string())


def _write(path: Path, dataset: dict[str, list[Any]]) -> str:
    """Write a fixture dataset as Parquet.

    Each column takes its type from its values (see ``_column``): integers
    become int64, other numbers float64, and anything else utf8.

    Args:
        path: Destination file.
        dataset: Column name -> values.

    Returns:
        The file path.

    """
    pq.write_table(pa.table({name: _column(values) for name, values in dataset.items()}), path)
    return str(path)


def _same(ours: Any, theirs: Any, tolerance: float | None = None, relative: float | None = None) -> bool:
    """Compare one encoded value with the library's, ignoring integer/float width.

    Args:
        ours: The value our engine produced.
        theirs: The value the library produced.
        tolerance: Absolute slack the case allows. Few cases set one:
            PolynomialEncoder, because patsy builds the polynomial basis with
            LAPACK's QR and we use Arnoldi iteration; one CountEncoder
            normalize case, whose grouped value upstream is the sum of its
            members' ratios; and the scikit-learn cases that go through an
            inverse normal CDF, Lloyd's algorithm, a lambda fitted by maximum
            likelihood, or a B-spline basis of degree 2 or more, which scipy
            evaluates with a fused multiply-add. The rest compare exactly.
        relative: Slack relative to the library's value, for the cases whose
            values are too large for an absolute one: a Box-Cox lambda capped
            as scipy caps it, which transforms values to 1e304 and more. An
            infinite value of the library's has to be matched exactly.

    Returns:
        Whether they agree.

    """
    if ours is None or theirs is None:
        return ours is None and theirs is None
    if isinstance(ours, str) or isinstance(theirs, str):
        return bool(ours == theirs)
    if tolerance is not None:
        return abs(float(ours) - float(theirs)) <= tolerance
    if relative is not None:
        mine, ref = float(ours), float(theirs)
        return mine == ref or (math.isfinite(ref) and abs(mine - ref) <= relative * abs(ref))
    return float(ours) == float(theirs)


def _assert_ordered_schema(ours: list[str], source: list[str], expected: list[str], where: str) -> None:
    """Assert our output columns are the case's, in order, with nothing extra.

    Two ordered sequences pin the frame: the columns a step passes through keep
    the order the input had them in, and the columns a step generates keep the
    order the library named them in.

    Where the generated block sits relative to the pass-through columns is the
    one thing left free, because that is a deliberate difference: an expanding
    encoder appends its block after the columns that survive the step, while
    upstream expands each column where it stood.

    Args:
        ours: Our output column names, in output order.
        source: The transformed dataset's column names, in input order.
        expected: The library's output column names, in its order.
        where: The case name, for the assertion message.

    """
    generated: list[str] = [name for name in expected if name not in source]
    survivors: list[str] = [name for name in source if name in set(ours)]
    assert set(ours) == set(generated) | set(survivors), f"{where}: output columns differ"
    assert [name for name in ours if name in set(generated)] == generated, f"{where}: generated columns out of order"
    assert [name for name in ours if name in set(survivors)] == survivors, f"{where}: pass-through columns out of order"


def _round_trips(transformer: Transformer, tmp_path: Path) -> list[Any]:
    """Return the fitted transformer as it comes back from an artifact and from a pickle.

    Args:
        transformer: A standalone-fitted transformer.
        tmp_path: pytest temporary directory to save the artifact under.

    Returns:
        The reloaded pipeline and the unpickled transformer, both fitted.

    """
    transformer._require_handle().save(str(tmp_path / "artifact"))
    return [Pipeline.load(tmp_path / "artifact"), pickle.loads(pickle.dumps(transformer))]


def _rows(transformer: Any, sources: list[str], output_dir: Path) -> list[dict[str, Any]]:
    """Transform Parquet files and read the parts back as records.

    Args:
        transformer: Anything with a transform_parquet method.
        sources: Parquet paths to transform.
        output_dir: Where the part files go.

    Returns:
        Every output row, in order.

    """
    parts: list[Path] = transformer.transform_parquet(sources, output_dir=output_dir)
    return pa.concat_tables([pq.read_table(part) for part in parts]).to_pylist()


@pytest.mark.parametrize("case", CASES, ids=[case["name"] for case in CASES])
def test_matches_category_encoders(case: dict[str, Any], tmp_path: Path) -> None:
    fit_path: str = _write(tmp_path / "fit.parquet", FIXTURE["datasets"][case["fit"]])
    transform_name: str = case.get("transform", case["fit"])
    transform_path: str = _write(tmp_path / "transform.parquet", FIXTURE["datasets"][transform_name])

    params: dict[str, Any] = dict(case["params"])
    if case.get("target") is not None:
        params["target"] = case["target"]
    transformer: Transformer = getattr(fast_feature_transform, case["encoder"])(**params)
    transformer.fit([fit_path])

    source: dict[str, list[Any]] = FIXTURE["datasets"][transform_name]
    if "raised" in case:
        # The library refuses this transform outright, so both of our paths
        # must too — a guarantee that held only for the fit data would be none.
        with pytest.raises(Exception, match="handle_missing is 'error'"):
            transformer.transform_parquet([transform_path], output_dir=tmp_path / "out")
        records: list[dict[str, Any]] = [
            {name: values[index] for name, values in source.items()}
            for index in range(len(next(iter(source.values()))))
        ]
        assert any(value is None for record in records for value in record.values())
        for record in records:
            if any(value is None for value in record.values()):
                with pytest.raises(Exception, match="handle_missing is 'error'"):
                    transformer.transform_dict(record)
        return

    rows: list[dict[str, Any]] = _rows(transformer, [transform_path], tmp_path / "out")

    expected: dict[str, list[Any]] = case["expected"]
    tolerance: float | None = case.get("tolerance")
    # A supervised encoder names its target in the params, so the target column
    # is in our output and never in the library's; it passes through in place,
    # so the ordered check counts it among the source columns.
    target: set[str] = {case["target"]} if "target" in case else set()
    assert set(rows[0]) - target == set(expected), "output columns differ"
    _assert_ordered_schema(list(rows[0]), list(source), list(expected), case["name"])
    for column, values in expected.items():
        got: list[Any] = [row[column] for row in rows]
        assert len(got) == len(values)
        for index, (mine, theirs) in enumerate(zip(got, values, strict=True)):
            assert _same(mine, theirs, tolerance), f"{column}[{index}]: ours {mine!r} != category_encoders {theirs!r}"

    # The record path has to agree with the batch path it was fitted beside.
    records: list[dict[str, Any]] = [
        {name: values[index] for name, values in source.items()} for index in range(len(rows))
    ]
    for record, row in zip(records, rows, strict=True):
        assert transformer.transform_dict(record) == row
        assert list(transformer.transform_dict(record)) == list(row)

    # And the artifact has to serve what the fit did — the same columns, in the
    # same order, holding the same values, through both paths.
    for restored in _round_trips(transformer, tmp_path):
        again: list[dict[str, Any]] = _rows(restored, [transform_path], tmp_path / "round-trip")
        assert [list(row) for row in again] == [list(row) for row in rows]
        assert again == rows
        for record, row in zip(records, rows, strict=True):
            assert restored.transform_dict(record) == row


@pytest.mark.parametrize("case", SKLEARN_CASES, ids=[case["name"] for case in SKLEARN_CASES])
def test_matches_sklearn(case: dict[str, Any], tmp_path: Path) -> None:
    fit_path: str = _write(tmp_path / "fit.parquet", SKLEARN["datasets"][case["fit"]])
    transform_name: str = case.get("transform", case["fit"])
    transform_path: str = _write(tmp_path / "transform.parquet", SKLEARN["datasets"][transform_name])

    transformer: Transformer = getattr(fast_feature_transform, case["transformer"])(**case["params"])
    transformer.fit([fit_path])

    rows: list[dict[str, Any]] = _rows(transformer, [transform_path], tmp_path / "out")

    expected: dict[str, list[Any]] = case["expected"]
    tolerance: float | None = case.get("tolerance")
    relative: float | None = case.get("relative_tolerance")
    source: dict[str, list[Any]] = SKLEARN["datasets"][transform_name]
    # scikit-learn transforms the frame it was handed, so a column the case did
    # not name is ours alone and passes through; every column it did name has
    # to be here, and a column that is neither is one we invented.
    assert set(expected) <= set(rows[0]), "an expected column is missing from our output"
    _assert_ordered_schema(list(rows[0]), list(source), list(expected), case["name"])
    for column, values in expected.items():
        got: list[Any] = [row[column] for row in rows]
        assert len(got) == len(values)
        for index, (mine, theirs) in enumerate(zip(got, values, strict=True)):
            assert _same(mine, theirs, tolerance, relative), (
                f"{column}[{index}]: ours {mine!r} != scikit-learn {theirs!r}"
            )

    # The record path has to agree with the batch path it was fitted beside.
    records: list[dict[str, Any]] = [
        {name: values[index] for name, values in source.items()} for index in range(len(rows))
    ]
    for record, row in zip(records, rows, strict=True):
        assert transformer.transform_dict(record) == row
        assert list(transformer.transform_dict(record)) == list(row)

    # And the artifact has to serve what the fit did — the same columns, in the
    # same order, holding the same values, through both paths.
    for restored in _round_trips(transformer, tmp_path):
        again: list[dict[str, Any]] = _rows(restored, [transform_path], tmp_path / "round-trip")
        assert [list(row) for row in again] == [list(row) for row in rows]
        assert again == rows
        for record, row in zip(records, rows, strict=True):
            assert restored.transform_dict(record) == row


# The largest deviation from scikit-learn 1.9.0 measured over the ten
# PowerTransformer fixture cases with an absolute tolerance, at
# power/box-cox-raw height[3]. The committed tolerance sits about six times
# above it. The PowerTransformer docstring and
# scripts/regen_sklearn_fixtures.py quote both numbers; this test keeps them
# honest. It is a measurement over this data, never a bound on the transformer:
# see the script's POWER_TOLERANCE comment for the input classes where the
# deviation is unbounded.
POWER_MEASURED_DEVIATION: float = 1.71e-6
POWER_COMMITTED_TOLERANCE: float = 1e-5


def test_power_transformer_stays_inside_the_measured_sklearn_deviation(tmp_path: Path) -> None:
    """Every PowerTransformer case with an absolute tolerance stays inside the measured deviation.

    The ten cases carry a tolerance of 1e-5, about six times the largest
    deviation measured over them; off the fixture it bounds nothing. This
    walks the shipped fixture, measures what our solver actually produces
    against what scikit-learn recorded, and fails if a case's tolerance
    changes or if the largest deviation grows past POWER_MEASURED_DEVIATION
    or falls below a tenth of it.

    Args:
        tmp_path: pytest temporary directory.

    """
    # The capped Box-Cox cases compare relative to their values instead.
    cases: list[dict[str, Any]] = [
        case for case in SKLEARN_CASES if case["name"].startswith("power/") and "relative_tolerance" not in case
    ]
    assert len(cases) == 10
    worst: float = 0.0
    for case in cases:
        assert case["tolerance"] == POWER_COMMITTED_TOLERANCE, case["name"]
        fit_path: str = _write(tmp_path / "fit.parquet", SKLEARN["datasets"][case["fit"]])
        transform_name: str = case.get("transform", case["fit"])
        transform_path: str = _write(tmp_path / "transform.parquet", SKLEARN["datasets"][transform_name])
        transformer: Transformer = getattr(fast_feature_transform, case["transformer"])(**case["params"])
        transformer.fit([fit_path])
        parts: list[Path] = transformer.transform_parquet([transform_path], output_dir=tmp_path / case["name"])
        rows: list[dict[str, Any]] = pa.concat_tables([pq.read_table(part) for part in parts]).to_pylist()
        for column, values in case["expected"].items():
            for index, theirs in enumerate(values):
                mine: Any = rows[index][column]
                if mine is None or theirs is None:
                    assert mine is None
                    assert theirs is None
                    continue
                worst = max(worst, abs(float(mine) - float(theirs)))
    assert worst <= POWER_MEASURED_DEVIATION, f"the deviation from scikit-learn grew to {worst!r}"
    assert worst > POWER_MEASURED_DEVIATION / 10.0, (
        f"the deviation fell to {worst!r}; re-measure it and tighten the documented figures"
    )


# The largest deviation from scikit-learn 1.9.0 measured over the eleven
# SplineTransformer fixture cases of degree 2 or more, at
# spline/extrapolation-continue height_sp_1[0], where the value is about -300.
# The committed tolerance sits about nine times above it.
# scripts/regen_sklearn_fixtures.py quotes every figure; this test keeps them
# honest.
SPLINE_MEASURED_DEVIATION: float = 5.7e-14
SPLINE_COMMITTED_TOLERANCE: float = 5e-13
# Degree 0 and 1 have no accumulation step in the Cox-de Boor recursion, so
# there is nothing for scipy's fused multiply-add to round differently: those
# cases are compared exactly and carry no tolerance at all.
SPLINE_EXACT_CASES: frozenset[str] = frozenset(
    {"spline/degree-0", "spline/degree-1", "spline/extrapolation-linear-degree-1"}
)


def test_spline_transformer_stays_inside_the_measured_sklearn_deviation(tmp_path: Path) -> None:
    """Every SplineTransformer case stays inside the measured deviation.

    The eleven cases of degree 2 or more carry a tolerance of 5e-13, about
    nine times the largest deviation measured over them, and the three of
    degree 0 or 1 carry none. This walks the shipped fixture, measures what
    our basis actually produces against what scikit-learn recorded, and fails
    if a case's tolerance changes, if an exact case deviates at all, or if the
    largest deviation grows past SPLINE_MEASURED_DEVIATION or falls below a
    tenth of it.

    Args:
        tmp_path: pytest temporary directory.

    """
    cases: list[dict[str, Any]] = [case for case in SKLEARN_CASES if case["name"].startswith("spline/")]
    assert len(cases) == 14
    worst: float = 0.0
    for case in cases:
        exact: bool = case["name"] in SPLINE_EXACT_CASES
        assert case.get("tolerance") == (None if exact else SPLINE_COMMITTED_TOLERANCE), case["name"]
        fit_path: str = _write(tmp_path / "fit.parquet", SKLEARN["datasets"][case["fit"]])
        transform_name: str = case.get("transform", case["fit"])
        transform_path: str = _write(tmp_path / "transform.parquet", SKLEARN["datasets"][transform_name])
        transformer: Transformer = getattr(fast_feature_transform, case["transformer"])(**case["params"])
        transformer.fit([fit_path])
        rows: list[dict[str, Any]] = _rows(transformer, [transform_path], tmp_path / case["name"])
        for column, values in case["expected"].items():
            for index, theirs in enumerate(values):
                mine: Any = rows[index][column]
                if mine is None or theirs is None:
                    assert mine is None
                    assert theirs is None
                    continue
                deviation: float = abs(float(mine) - float(theirs))
                assert not (exact and deviation > 0.0), f"{case['name']} {column}[{index}] is no longer exact"
                worst = max(worst, deviation)
    assert worst <= SPLINE_MEASURED_DEVIATION, f"the deviation from scikit-learn grew to {worst!r}"
    assert worst > SPLINE_MEASURED_DEVIATION / 10.0, (
        f"the deviation fell to {worst!r}; re-measure it and tighten the documented figures"
    )
