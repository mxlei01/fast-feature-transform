"""Tests for the four contrast coders: backward difference, Helmert, sum, and polynomial."""

import itertools
import json
import pickle
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fast_feature_transform import (
    BackwardDifferenceEncoder,
    CountEncoder,
    EngineError,
    HelmertEncoder,
    Pipeline,
    PolynomialEncoder,
    SumEncoder,
    _core,
)

CODERS = (BackwardDifferenceEncoder, HelmertEncoder, SumEncoder, PolynomialEncoder)

# patsy's Poly() for four levels, which is what the parity fixture checks too.
POLY_4: list[list[float]] = [
    [-0.6708203932499369, 0.5, -0.223606797749979],
    [-0.22360679774997896, -0.5, 0.6708203932499368],
    [0.22360679774997902, -0.5, -0.6708203932499369],
    [0.670820393249937, 0.5, 0.2236067977499791],
]


def _write(path: Path, table: pa.Table, **kwargs: Any) -> str:
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


def _table(parts: list[Path]) -> pa.Table:
    """Read output parts back as one table.

    Args:
        parts: Part files in order.

    Returns:
        The concatenated table.

    """
    return pa.concat_tables([pq.read_table(part) for part in parts])


@pytest.fixture
def colors(tmp_path: Path) -> list[str]:
    """Four categories over three row groups, in no particular order.

    Args:
        tmp_path: pytest-provided temporary directory.

    Returns:
        Paths of the written Parquet files.

    """
    table: pa.Table = pa.table(
        {"color": ["delta", "amber", "blue", "cyan", "amber", "delta"], "size": ["l"] * 6},
    )
    return [_write(tmp_path / "colors.parquet", table, row_group_size=2)]


@pytest.fixture
def with_null(tmp_path: Path) -> list[str]:
    """Three categories plus a null, over three row groups.

    Args:
        tmp_path: pytest-provided temporary directory.

    Returns:
        Paths of the written Parquet files.

    """
    table: pa.Table = pa.table({"color": ["cyan", "amber", None, "blue", "amber", "cyan"], "size": ["l"] * 6})
    return [_write(tmp_path / "nulls.parquet", table, row_group_size=2)]


def test_each_coder_replaces_the_column_with_its_own_matrix(colors: list[str], tmp_path: Path) -> None:
    difference: BackwardDifferenceEncoder = BackwardDifferenceEncoder(columns=["color"]).fit(colors)
    assert difference.categories_ == {"color": ["amber", "blue", "cyan", "delta"]}
    table: pa.Table = _table(difference.transform_parquet(colors, output_dir=tmp_path / "diff"))
    # Four levels, so three f64 columns named after their position, and the
    # source column is gone.
    assert table.column_names == ["size", "color_0", "color_1", "color_2"]
    assert table.schema.field("color_0").type == pa.float64()
    rows: list[dict[str, Any]] = table.to_pylist()
    assert rows[1] == {"size": "l", "color_0": -0.75, "color_1": -0.5, "color_2": -0.25}  # amber, level 0
    assert rows[0] == {"size": "l", "color_0": 0.25, "color_1": 0.5, "color_2": 0.75}  # delta, level 3
    assert difference.transform_dict({"color": "amber", "size": "l"}) == rows[1]

    helmert: HelmertEncoder = HelmertEncoder(columns=["color"]).fit(colors)
    assert [helmert.transform_dict({"color": name}) for name in ("amber", "blue", "cyan", "delta")] == [
        {"color_0": -1.0, "color_1": -1.0, "color_2": -1.0},
        {"color_0": 1.0, "color_1": -1.0, "color_2": -1.0},
        {"color_0": 0.0, "color_1": 2.0, "color_2": -1.0},
        {"color_0": 0.0, "color_1": 0.0, "color_2": 3.0},
    ]

    # The last level in sorted order is sum coding's reference level.
    total: SumEncoder = SumEncoder(columns=["color"]).fit(colors)
    assert total.transform_dict({"color": "amber"}) == {"color_0": 1.0, "color_1": 0.0, "color_2": 0.0}
    assert total.transform_dict({"color": "delta"}) == {"color_0": -1.0, "color_1": -1.0, "color_2": -1.0}

    polynomial: PolynomialEncoder = PolynomialEncoder(columns=["color"]).fit(colors)
    for level, expected in zip(("amber", "blue", "cyan", "delta"), POLY_4, strict=True):
        row: dict[str, Any] = polynomial.transform_dict({"color": level})
        assert [row["color_0"], row["color_1"], row["color_2"]] == pytest.approx(expected, abs=1e-12)


def test_a_fitted_null_is_a_level_of_its_own_and_sorts_last(with_null: list[str], tmp_path: Path) -> None:
    total: SumEncoder = SumEncoder(columns=["color"]).fit(with_null)
    assert total.categories_ == {"color": ["amber", "blue", "cyan"]}
    # Three categories plus the null level: four levels, so three columns, and
    # the null is the reference level sum coding leaves out.
    rows: list[dict[str, Any]] = _table(total.transform_parquet(with_null, output_dir=tmp_path / "out")).to_pylist()
    assert rows[1] == {"size": "l", "color_0": 1.0, "color_1": 0.0, "color_2": 0.0}  # amber
    assert rows[2] == {"size": "l", "color_0": -1.0, "color_1": -1.0, "color_2": -1.0}  # the fitted null
    assert total.transform_dict({"color": None, "size": "l"}) == rows[2]

    # REGRESSION: a null under 'return_nan' is a row of nulls and not a level,
    # but category_encoders' ordinal pre-encoder still holds a slot for it,
    # and that slot is a level of its own that nothing can reach. So the
    # matrix stays four levels wide and three columns come out. This used to
    # drop a column, leaving one fewer than category_encoders makes.
    dropped: SumEncoder = SumEncoder(columns=["color"], handle_missing="return_nan").fit(with_null)
    assert dropped.transform_dict({"color": None}) == {"color_0": None, "color_1": None, "color_2": None}
    assert dropped.transform_dict({"color": "amber"}) == {"color_0": 1.0, "color_1": 0.0, "color_2": 0.0}
    assert dropped.transform_dict({"color": "cyan"}) == {"color_0": 0.0, "color_1": 0.0, "color_2": 1.0}
    # An unseen category is still the zero row, over the wider matrix.
    assert dropped.transform_dict({"color": "zulu"}) == {"color_0": 0.0, "color_1": 0.0, "color_2": 0.0}


def test_unknown_categories_follow_handle_unknown(colors: list[str], tmp_path: Path) -> None:
    zeros: HelmertEncoder = HelmertEncoder(columns=["color"]).fit(colors)
    assert zeros.transform_dict({"color": "zulu"}) == {"color_0": 0.0, "color_1": 0.0, "color_2": 0.0}
    # A null that was never a fitted category is a row of zeros too.
    assert zeros.transform_dict({"color": None}) == {"color_0": 0.0, "color_1": 0.0, "color_2": 0.0}

    nulls: HelmertEncoder = HelmertEncoder(columns=["color"], handle_unknown="return_nan").fit(colors)
    assert nulls.transform_dict({"color": "zulu"}) == {"color_0": None, "color_1": None, "color_2": None}

    # 'indicator' makes the unknown a real fifth level, so a fourth column appears.
    flagged: HelmertEncoder = HelmertEncoder(columns=["color"], handle_unknown="indicator").fit(colors)
    table: pa.Table = _table(flagged.transform_parquet(colors, output_dir=tmp_path / "flag"))
    assert table.column_names == ["size", "color_0", "color_1", "color_2", "color_3"]
    assert flagged.transform_dict({"color": "zulu"}) == {
        "color_0": 0.0,
        "color_1": 0.0,
        "color_2": 0.0,
        "color_3": 4.0,
    }
    assert flagged.transform_dict({"color": "amber"}) == {
        "color_0": -1.0,
        "color_1": -1.0,
        "color_2": -1.0,
        "color_3": -1.0,
    }

    strict: HelmertEncoder = HelmertEncoder(columns=["color"], handle_unknown="error").fit(colors)
    with pytest.raises(EngineError, match="category that was not seen at fit"):
        strict.transform_dict({"color": "zulu"})


def test_a_single_level_emits_no_columns_at_all(tmp_path: Path) -> None:
    only_one: str = _write(tmp_path / "one.parquet", pa.table({"color": ["amber"] * 4, "size": ["l", "m", "s", "l"]}))
    for cls in CODERS:
        # Even with the indicator level asked for: upstream drops the column
        # before it ever appends one.
        encoder = cls(columns=["color"], handle_unknown="indicator").fit([only_one])
        parts: list[Path] = encoder.transform_parquet([only_one], output_dir=tmp_path / cls.__name__)
        assert _table(parts).column_names == ["size"]
        assert encoder.transform_dict({"color": "amber", "size": "l"}) == {"size": "l"}

    # The column is still read, so 'error' still refuses an unseen category.
    strict: SumEncoder = SumEncoder(columns=["color"], handle_unknown="error").fit([only_one])
    with pytest.raises(EngineError, match="category that was not seen at fit"):
        strict.transform_dict({"color": "zulu", "size": "l"})


def test_a_later_step_sees_the_schema_the_expansion_produced(colors: list[str], tmp_path: Path) -> None:
    # count_encode picks its columns from the schema at fit time, so it must
    # see what the coder emitted: color is gone by then, size is still utf8.
    pipeline: Pipeline = Pipeline([("sum", SumEncoder(columns=["color"])), ("counts", CountEncoder())]).fit(colors)
    counts = pipeline["counts"]
    assert isinstance(counts, CountEncoder)
    assert counts.counts_ == {"size": {"l": 6}}
    row: dict[str, Any] = pipeline.transform_dict({"color": "amber", "size": "l"})
    assert row == {"size": 6, "color_0": 1.0, "color_1": 0.0, "color_2": 0.0}
    assert _table(pipeline.transform_parquet(colors, output_dir=tmp_path / "out")).to_pylist()[1] == row


def test_the_fitted_state_survives_threads_the_artifact_and_a_pickle(with_null: list[str], tmp_path: Path) -> None:
    for cls in CODERS:
        states: set[str] = set()
        for threads in (1, 4):
            encoder = cls(columns=["color"]).fit(with_null, threads=threads)
            states.add(encoder._require_handle().stage_state(cls.__name__.lower()))
        assert len(states) == 1, cls.__name__

    pipeline: Pipeline = Pipeline([("poly", PolynomialEncoder(columns=["color"]))]).fit(with_null)
    pipeline.save(tmp_path / "artifact")
    manifest: dict[str, Any] = json.loads((tmp_path / "artifact" / "manifest.json").read_text())
    assert manifest["ops"][0]["kernel"] == "polynomial_encode"
    state: dict[str, Any] = manifest["ops"][0]["state"]
    basis: dict[str, Any] = state.pop("polynomial_basis")
    assert state == {
        "columns": ["color"],
        "categories": {"color": ["amber", "blue", "cyan"]},
        "has_null": {"color": True},
    }
    assert basis["version"] == 1
    np.testing.assert_allclose(basis["matrices"]["color"], POLY_4, rtol=0, atol=1e-12)
    state["polynomial_basis"] = basis
    record: dict[str, Any] = {"color": "blue", "size": "l"}
    assert Pipeline.load(tmp_path / "artifact").transform_dict(record) == pipeline.transform_dict(record)
    assert pickle.loads(pickle.dumps(pipeline)).transform_dict(record) == pipeline.transform_dict(record)

    manifest["ops"][0]["state"]["categories"]["color"] = ["cyan", "amber"]
    with pytest.raises(EngineError, match="must be sorted and unique"):
        _core.FittedPipeline.load_json(json.dumps(manifest))


@pytest.mark.parametrize(
    ("categories", "fitted_null", "missing", "unknown"),
    [
        (32, False, "value", "value"),
        (31, True, "value", "value"),
        (31, False, "return_nan", "value"),
        (31, False, "error", "value"),
        (30, False, "return_nan", "indicator"),
    ],
    ids=["categories", "fitted-null", "missing-placeholder", "strict-placeholder", "unknown-indicator"],
)
def test_polynomial_basis_is_stable_for_32_effective_levels(
    tmp_path: Path, categories: int, fitted_null: bool, missing: str, unknown: str
) -> None:
    values: list[str | None] = [f"c{index:03}" for index in range(categories)]
    if fitted_null:
        values.append(None)
    source: str = _write(tmp_path / "levels.parquet", pa.table({"color": values}), row_group_size=7)
    encoder: PolynomialEncoder = PolynomialEncoder(
        columns=["color"], handle_missing=missing, handle_unknown=unknown
    ).fit([source])
    state: dict[str, Any] = json.loads(encoder._require_handle().stage_state("polynomialencoder"))
    matrix = np.asarray(state["polynomial_basis"]["matrices"]["color"])
    assert matrix.shape == (32, 31)
    assert np.isfinite(matrix).all()
    # The full basis is orthonormal, including the omitted constant column.
    complete = np.column_stack((np.full(32, 1 / np.sqrt(32)), matrix))
    np.testing.assert_allclose(complete.T @ complete, np.eye(32), rtol=0, atol=2e-13)

    inferred: list[str | None] = [*values, "unseen"]
    if missing != "error" and None not in inferred:
        inferred.append(None)
    table: pa.Table = pa.table({"color": inferred})
    expected: list[dict[str, Any]] = [encoder.transform_dict(record) for record in table.to_pylist()]
    # Strings reach the kernels as Parquet's dictionaries, or as plain strings when the pages are delta-encoded.
    plain: dict[str, Any] = {"use_dictionary": False, "column_encoding": {"color": "DELTA_BYTE_ARRAY"}}
    for encoding, options in (("dictionary", {}), ("delta", plain)):
        probe: str = _write(tmp_path / f"inferred-{encoding}.parquet", table, row_group_size=7, **options)
        output: pa.Table = _table(encoder.transform_parquet([probe], output_dir=tmp_path / f"out-{encoding}"))
        assert output.to_pylist() == expected, encoding
    names: list[str] = [f"color_{index}" for index in range(31)]
    assert output.column_names == names
    for index in range(categories):
        assert list(expected[index].values()) == matrix[index].tolist()
    if fitted_null:
        assert list(encoder.transform_dict({"color": None}).values()) == matrix[categories].tolist()
    if unknown == "indicator":
        assert list(encoder.transform_dict({"color": "unseen"}).values()) == matrix[-1].tolist()
    else:
        assert encoder.transform_dict({"color": "unseen"}) == dict.fromkeys(names, 0.0)
    if missing == "return_nan":
        assert encoder.transform_dict({"color": None}) == dict.fromkeys(names)
    elif missing == "error":
        with pytest.raises(EngineError, match="null"):
            encoder.transform_dict({"color": None})


def test_polynomial_artifacts_preserve_stored_values_and_column_order(tmp_path: Path) -> None:
    table: pa.Table = pa.table(
        {
            "a": [f"a{index:02}" for index in range(32)],
            "z": [f"z{31 - index:02}" for index in range(32)],
        }
    )
    source: str = _write(tmp_path / "two.parquet", table, row_group_size=5)
    runs: Iterator[int] = itertools.count()

    def batch(model: Pipeline) -> list[dict[str, Any]]:
        return _table(model.transform_parquet([source], output_dir=tmp_path / f"out-{next(runs)}")).to_pylist()

    fitted: Pipeline = Pipeline([("poly", PolynomialEncoder(columns=["z", "a"]))]).fit([source])
    original: list[dict[str, Any]] = batch(fitted)
    artifact: Path = tmp_path / "artifact"
    fitted.save(artifact)
    assert batch(Pipeline.load(artifact)) == original
    assert batch(pickle.loads(pickle.dumps(fitted))) == original

    manifest: dict[str, Any] = json.loads((artifact / "manifest.json").read_text())
    assert manifest["schema_version"] == "0.3.0"
    state: dict[str, Any] = manifest["ops"][0]["state"]
    assert state["columns"] == ["z", "a"]
    # Fitted coefficients are model state. Loading must not reconstruct or
    # normalize a matrix, even if its values differ from today's fitter.
    state["polynomial_basis"]["matrices"]["z"][0][-1] = 0.125
    state["polynomial_basis"]["matrices"]["a"][0][0] = -0.25
    (artifact / "manifest.json").write_text(json.dumps(manifest))
    loaded: Pipeline = Pipeline.load(artifact)
    changed: list[dict[str, Any]] = batch(loaded)
    original[-1]["z_30"] = 0.125
    original[0]["a_0"] = -0.25
    assert changed == original
    assert list(changed[0]) == [f"{column}_{index}" for column in ("z", "a") for index in range(31)]

    for restored in (loaded, pickle.loads(pickle.dumps(loaded))):
        for record, expected in zip(table.to_pylist(), changed, strict=True):
            assert restored.transform_dict(record) == expected
    loaded.save(tmp_path / "resaved")
    assert json.loads((tmp_path / "resaved" / "manifest.json").read_text()) == manifest
    assert batch(Pipeline.load(tmp_path / "resaved")) == changed


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        ((), None),
        (("version",), True),
        (("version",), 1.0),
        (("version",), 2),
        (("matrices",), []),
        (("matrices",), {}),
        (("matrices",), {"color": POLY_4, "extra": POLY_4}),
        (("matrices", "color"), POLY_4[:-1]),
        (("matrices", "color", 0), POLY_4[0][:-1]),
        (("matrices", "color", 0, 0), None),
        (("matrices", "color", 0, 0), True),
        (("matrices", "color", 0, 0), "0.5"),
    ],
    ids=[
        "null-basis",
        "boolean-version",
        "float-version",
        "unknown-version",
        "non-object-matrices",
        "missing-column",
        "extra-column",
        "wrong-row-count",
        "wrong-column-count",
        "null-value",
        "boolean-value",
        "string-value",
    ],
)
def test_polynomial_artifacts_reject_invalid_stored_matrices(
    colors: list[str], path: tuple[str | int, ...], replacement: Any
) -> None:
    pipeline: Pipeline = Pipeline([("poly", PolynomialEncoder(columns=["color"]))]).fit(colors)
    manifest: dict[str, Any] = json.loads(pipeline._require_handle().manifest_json())
    parent: Any = manifest["ops"][0]["state"]
    keys: tuple[str | int, ...] = ("polynomial_basis", *path)
    for key in keys[:-1]:
        parent = parent[key]
    parent[keys[-1]] = replacement
    with pytest.raises(EngineError, match="polynomial"):
        _core.FittedPipeline.load_json(json.dumps(manifest))


def test_a_polynomial_artifact_needs_its_stored_basis(colors: list[str], tmp_path: Path) -> None:
    fitted: Pipeline = Pipeline([("poly", PolynomialEncoder(columns=["color"]))]).fit(colors)
    manifest: dict[str, Any] = json.loads(fitted._require_handle().manifest_json())
    del manifest["ops"][0]["state"]["polynomial_basis"]
    with pytest.raises(EngineError, match="'polynomial_basis'"):
        _core.FittedPipeline.load_json(json.dumps(manifest))
    # Any other schema_version is refused before an op is loaded.
    manifest["schema_version"] = "0.2.0"
    artifact: Path = tmp_path / "other-version"
    artifact.mkdir()
    (artifact / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(EngineError, match="schema_version '0.2.0'.*refit the pipeline"):
        Pipeline.load(artifact)


def test_polynomial_artifact_preserves_an_empty_single_level_matrix(tmp_path: Path) -> None:
    table: pa.Table = pa.table({"color": ["only"] * 3, "size": [1, 2, 3]})
    source: str = _write(tmp_path / "single.parquet", table)
    fitted: Pipeline = Pipeline([("poly", PolynomialEncoder(columns=["color"], handle_unknown="indicator"))]).fit(
        [source]
    )
    manifest: dict[str, Any] = json.loads(fitted._require_handle().manifest_json())
    assert manifest["ops"][0]["state"]["polynomial_basis"] == {"version": 1, "matrices": {"color": []}}
    artifact: Path = tmp_path / "single-artifact"
    fitted.save(artifact)
    parts: list[Path] = Pipeline.load(artifact).transform_parquet([source], output_dir=tmp_path / "out")
    assert _table(parts).equals(table.select(["size"]))


def test_the_shells_validate_their_arguments() -> None:
    bare_string: Any = "color"
    numbers: Any = [1]
    for cls in CODERS:
        with pytest.raises(TypeError, match="columns must be a list of column names, not a single string"):
            cls(columns=bare_string)
        with pytest.raises(TypeError, match="columns must be a list of column names"):
            cls(columns=numbers)
        with pytest.raises(ValueError, match="columns must name at least one column"):
            cls(columns=[])
        with pytest.raises(ValueError, match="handle_unknown must be one of"):
            cls(columns=["color"], handle_unknown="maybe")
        with pytest.raises(ValueError, match="handle_missing must be one of"):
            cls(columns=["color"], handle_missing="indicator")
        assert cls(columns=["color"]).params() == {
            "columns": ["color"],
            "handle_unknown": "value",
            "handle_missing": "value",
        }
        assert repr(cls()) == f"{cls.__name__}(columns=None, handle_unknown='value', handle_missing='value')"


def test_the_engine_names_what_it_refuses(colors: list[str], with_null: list[str], tmp_path: Path) -> None:
    with pytest.raises(EngineError, match="column 'nope' not found in input"):
        SumEncoder(columns=["nope"]).fit(colors)
    with pytest.raises(EngineError, match="has nulls and handle_missing is 'error'"):
        SumEncoder(columns=["color"], handle_missing="error").fit(with_null)

    # No coercion: a non-utf8 column has to be cast first.
    numbers: str = _write(tmp_path / "numbers.parquet", pa.table({"color": [1, 2, 3], "size": ["l", "m", "s"]}))
    with pytest.raises(EngineError, match="column 'color' is i64, expected utf8"):
        HelmertEncoder(columns=["color"]).fit([numbers])
    only_numbers: str = _write(tmp_path / "only.parquet", pa.table({"n": [1, 2, 3]}))
    with pytest.raises(EngineError, match="no utf8 columns to helmert-encode"):
        HelmertEncoder().fit([only_numbers])

    clash: str = _write(
        tmp_path / "clash.parquet",
        pa.table({"color": ["amber", "blue", "cyan"], "color_0": ["x", "y", "z"]}),
    )
    with pytest.raises(EngineError, match="output column 'color_0' already exists in input"):
        PolynomialEncoder(columns=["color"]).fit([clash])

    unfitted: SumEncoder = SumEncoder(columns=["color"])
    with pytest.raises(RuntimeError, match="not fitted"):
        unfitted.transform_dict({"color": "amber"})
    fitted: SumEncoder = SumEncoder(columns=["color"]).fit(colors)
    with pytest.raises(EngineError, match="key 'color' missing from record"):
        fitted.transform_dict({"size": "l"})
    with pytest.raises(EngineError, match="key 'color' must be a string or null"):
        fitted.transform_dict({"color": 3})
