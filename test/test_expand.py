"""Tests for the polynomial_features, spline_transform and power_transform kernels."""

import json
import math
import pickle
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fast_feature_transform import (
    Cast,
    EngineError,
    Pipeline,
    PolynomialFeatures,
    PowerTransformer,
    Select,
    SplineTransformer,
    StandardScaler,
    _core,
)
from helpers import NAME_KIND_IDS, NAME_KINDS, read_rows, write_parquet


@pytest.fixture
def numbers(tmp_path: Path) -> list[str]:
    """One file with three row groups of two numeric columns and a passthrough.

    Args:
        tmp_path: pytest-provided temporary directory.

    Returns:
        Paths of the written Parquet files.

    """
    table: pa.Table = pa.table(
        {
            "price": [1.0, 2.0, 4.0, 8.0, 3.0, 5.0],
            "height": [10.0, 12.0, 9.0, 15.0, 11.0, 13.0],
            "device": ["ios", "web", "ios", "tv", "web", "ios"],
        }
    )
    return [write_parquet(tmp_path / "numbers.parquet", table, row_group_size=2)]


def test_polynomial_features_expands_across_columns(numbers: list[str], tmp_path: Path) -> None:
    expand: PolynomialFeatures = PolynomialFeatures(columns=["price", "height"])
    assert expand.output_columns == ["1", "price", "height", "price^2", "price*height", "height^2"]
    expand.fit(numbers)
    rows: list[dict[str, Any]] = read_rows(expand.transform_parquet(numbers, output_dir=tmp_path / "out"))
    # The source columns leave; the passthrough stays and the monomials follow.
    assert list(rows[0]) == ["device", *expand.output_columns]
    assert rows[0] == {
        "device": "ios",
        "1": 1.0,
        "price": 1.0,
        "height": 10.0,
        "price^2": 1.0,
        "price*height": 10.0,
        "height^2": 100.0,
    }
    for row_index, row in enumerate(rows):
        record: dict[str, Any] = {"price": row["price"], "height": row["height"], "device": row["device"]}
        assert expand.transform_dict(record) == row, row_index

    # An integer column is read exactly, and the outputs are f64 either way.
    integers: str = write_parquet(tmp_path / "ints.parquet", pa.table({"price": [2, 3], "height": [5, 7]}))
    both: PolynomialFeatures = PolynomialFeatures(columns=["price", "height"]).fit([integers])
    table: pa.Table = pa.concat_tables(
        [pq.read_table(part) for part in both.transform_parquet([integers], output_dir=tmp_path / "ints-out")]
    )
    assert table.schema.field("price*height").type == pa.float64()
    assert table.column("price*height").to_pylist() == [10.0, 21.0]


def test_polynomial_features_honours_degree_interaction_and_bias(numbers: list[str], tmp_path: Path) -> None:
    cubed: PolynomialFeatures = PolynomialFeatures(columns=["price", "height"], degree=3, include_bias=False)
    assert cubed.output_columns == [
        "price",
        "height",
        "price^2",
        "price*height",
        "height^2",
        "price^3",
        "price^2*height",
        "price*height^2",
        "height^3",
    ]
    assert cubed.fit(numbers).transform_dict({"price": 2.0, "height": 3.0, "device": "ios"})["price^2*height"] == 12.0

    interactions: PolynomialFeatures = PolynomialFeatures(columns=["price", "height"], degree=2, interaction_only=True)
    assert interactions.output_columns == ["1", "price", "height", "price*height"]
    ranged: PolynomialFeatures = PolynomialFeatures(columns=["price", "height"], degree=(2, 3))
    assert ranged.output_columns == [
        "1",
        "price^2",
        "price*height",
        "height^2",
        "price^3",
        "price^2*height",
        "price*height^2",
        "height^3",
    ]
    bias_only: PolynomialFeatures = PolynomialFeatures(columns=["price"], degree=0)
    assert bias_only.fit(numbers).transform_dict({"price": 9.0, "height": 1.0, "device": "ios"}) == {
        "height": 1.0,
        "device": "ios",
        "1": 1.0,
    }

    # A null factor makes its monomials null; the bias uses no factor at all.
    holes: str = write_parquet(tmp_path / "holes.parquet", pa.table({"price": [None, 2.0], "height": [3.0, None]}))
    nulls: PolynomialFeatures = PolynomialFeatures(columns=["price", "height"]).fit([holes])
    assert read_rows(nulls.transform_parquet([holes], output_dir=tmp_path / "holes-out"))[0] == {
        "1": 1.0,
        "price": None,
        "height": 3.0,
        "price^2": None,
        "price*height": None,
        "height^2": 9.0,
    }
    assert nulls.transform_dict({"price": None, "height": 3.0}) == {
        "1": 1.0,
        "price": None,
        "height": 3.0,
        "price^2": None,
        "price*height": None,
        "height^2": 9.0,
    }


def test_polynomial_features_validates_its_arguments(numbers: list[str], tmp_path: Path) -> None:
    text: Any = "2"
    not_a_flag: Any = 1
    three: Any = (1, 2, 3)
    with pytest.raises(ValueError, match="columns must name at least one column"):
        PolynomialFeatures(columns=[])
    with pytest.raises(ValueError, match="would make a generated name ambiguous"):
        PolynomialFeatures(columns=["unit price"])
    with pytest.raises(ValueError, match="would make a generated name ambiguous"):
        PolynomialFeatures(columns=["a^2"])
    # REGRESSION: a column named 1 meets the bias column, also named 1. The
    # clash used to pass here and fail at fit. A column listed twice is still
    # refused at fit, where the message names it.
    with pytest.raises(ValueError, match="column '1' has the bias column's name"):
        PolynomialFeatures(columns=["1", "x"])
    assert PolynomialFeatures(columns=["x", "x"]).columns == ["x", "x"]
    assert PolynomialFeatures(columns=["1", "x"], include_bias=False).output_columns == ["1", "x", "1^2", "1*x", "x^2"]
    with pytest.raises(TypeError, match="degree must be an integer"):
        PolynomialFeatures(columns=["a"], degree=text)
    with pytest.raises(TypeError, match="degree must be an integer or a .min, max. pair"):
        PolynomialFeatures(columns=["a"], degree=three)
    with pytest.raises(ValueError, match="degree must not be negative"):
        PolynomialFeatures(columns=["a"], degree=-1)
    with pytest.raises(ValueError, match="degree must have min <= max"):
        PolynomialFeatures(columns=["a"], degree=(3, 2))
    with pytest.raises(ValueError, match="degree 0 with include_bias False"):
        PolynomialFeatures(columns=["a"], degree=0, include_bias=False)
    with pytest.raises(TypeError, match="interaction_only must be True or False"):
        PolynomialFeatures(columns=["a"], interaction_only=not_a_flag)
    with pytest.raises(TypeError, match="include_bias must be True or False"):
        PolynomialFeatures(columns=["a"], include_bias=not_a_flag)
    with pytest.raises(ValueError, match="more than the 4096 columns allowed"):
        PolynomialFeatures(columns=["a", "b", "c", "d"], degree=20)

    with pytest.raises(EngineError, match="is utf8, expected an integer or float column"):
        PolynomialFeatures(columns=["device"]).fit(numbers)
    with pytest.raises(EngineError, match="column 'nope' not found in input"):
        PolynomialFeatures(columns=["nope"]).fit(numbers)
    clash: str = write_parquet(tmp_path / "clash.parquet", pa.table({"price": [1.0], "price^2": [1.0]}))
    with pytest.raises(EngineError, match="output column 'price\\^2' already exists in input"):
        PolynomialFeatures(columns=["price"]).fit([clash])
    expand: PolynomialFeatures = PolynomialFeatures(columns=["price", "height"]).fit(numbers)
    with pytest.raises(EngineError, match="key 'price\\*height' already present in record"):
        expand.transform_dict({"price": 1.0, "height": 2.0, "price*height": 3.0})
    with pytest.raises(EngineError, match="must be a number or null"):
        expand.transform_dict({"price": "cheap", "height": 2.0})
    infinite: str = write_parquet(tmp_path / "inf.parquet", pa.table({"price": [math.inf], "height": [1.0]}))
    with pytest.raises(EngineError, match="is not finite"):
        expand.transform_parquet([infinite], output_dir=tmp_path / "inf-out")


@pytest.mark.parametrize("name", NAME_KINDS, ids=NAME_KIND_IDS)
def test_output_columns_name_the_columns_written_when_a_column_name_is_a_str_subclass(
    name: Callable[[str], Any], numbers: list[str], tmp_path: Path
) -> None:
    # REGRESSION: a column name given as a str subclass is stored as the
    # plain string it equals, so output_columns names the columns written. An
    # Enum with a str mixin formats as "EnumName.price", and output_columns
    # used to hold "EnumName.price^2" and "EnumName.price_sp_0" where price^2
    # and price_sp_0 were written, so a later step that named them failed.
    def expansions(make: Callable[[str], Any]) -> list[PolynomialFeatures | SplineTransformer]:
        return [
            PolynomialFeatures(columns=[make("price"), make("height")], degree=3),
            SplineTransformer(columns=[make("price"), make("height")], n_knots=3, degree=2),
        ]

    for given, plain in zip(expansions(name), expansions(str), strict=True):
        assert given.output_columns == plain.output_columns
        given.fit(numbers)
        assert list(given.transform_dict({"price": 2.0, "height": 11.0})) == given.output_columns
        rows: list[dict[str, Any]] = read_rows(
            given.transform_parquet(numbers, output_dir=tmp_path / type(given).__name__)
        )
        assert list(rows[0]) == ["device", *given.output_columns]
    # A later step in the same pipeline can name them.
    for expansion in expansions(name):
        chained: Pipeline = Pipeline([("expand", expansion), ("pick", Select(columns=expansion.output_columns))])
        chained.fit(numbers)
        assert list(chained.transform_dict({"price": 2.0, "height": 11.0})) == expansion.output_columns


def test_spline_transform_fits_knots_and_evaluates_the_basis(numbers: list[str], tmp_path: Path) -> None:
    spline: SplineTransformer = SplineTransformer(columns=["price"], n_knots=3, degree=1)
    assert spline.splines == 3
    assert spline.output_columns == ["price_sp_0", "price_sp_1", "price_sp_2"]
    spline.fit(numbers)
    # price runs 1..8, so the uniform knots are 1, 4.5 and 8.
    assert spline.knots_ == {"price": [1.0, 4.5, 8.0]}
    rows: list[dict[str, Any]] = read_rows(spline.transform_parquet(numbers, output_dir=tmp_path / "out"))
    assert list(rows[0]) == ["height", "device", "price_sp_0", "price_sp_1", "price_sp_2"]
    assert rows[0]["price_sp_0"] == pytest.approx(1.0)  # the first knot lights the first basis alone
    assert rows[2]["price_sp_1"] == pytest.approx((4.0 - 1.0) / 3.5)  # 4.0 is 6/7 of the way to the middle knot
    for index, row in enumerate(rows):
        source: dict[str, Any] = {
            "price": [1.0, 2.0, 4.0, 8.0, 3.0, 5.0][index],
            "height": row["height"],
            "device": row["device"],
        }
        assert spline.transform_dict(source) == row

    # quantile knots take the percentiles of the whole column instead.
    quantile: SplineTransformer = SplineTransformer(columns=["price"], n_knots=3, degree=1, knots="quantile").fit(
        numbers
    )
    assert quantile.knots_ == {"price": [1.0, 3.5, 8.0]}

    # Dropping the bias drops the last basis function per column.
    lean: SplineTransformer = SplineTransformer(columns=["price"], n_knots=3, degree=1, include_bias=False)
    assert lean.output_columns == ["price_sp_0", "price_sp_1"]
    assert set(lean.fit(numbers).transform_dict({"price": 1.0, "height": 1.0, "device": "ios"})) == {
        "height",
        "device",
        "price_sp_0",
        "price_sp_1",
    }


def test_spline_transform_extrapolation_modes(numbers: list[str], tmp_path: Path) -> None:
    def basis(mode: str, value: float, **kwargs: Any) -> list[float]:
        spline: SplineTransformer = SplineTransformer(
            columns=["price"], n_knots=3, degree=1, extrapolation=mode, **kwargs
        ).fit(numbers)
        row: dict[str, Any] = spline.transform_dict({"price": value, "height": 1.0, "device": "ios"})
        return [row[name] for name in spline.output_columns]

    assert basis("constant", 100.0) == pytest.approx([0.0, 0.0, 1.0])
    assert basis("constant", -100.0) == pytest.approx([1.0, 0.0, 0.0])
    # Linear continues each boundary basis along its own slope.
    assert basis("linear", 9.0)[2] == pytest.approx(1.0 + (9.0 - 8.0) / 3.5)
    # 'continue' evaluates the boundary polynomial pieces on, unclamped.
    assert basis("continue", 9.0)[2] == pytest.approx(1.0 + (9.0 - 8.0) / 3.5)
    # 'periodic' wraps onto [1, 8): 9.0 reads as 2.0.
    assert basis("periodic", 9.0) == pytest.approx(basis("periodic", 2.0))

    strict: SplineTransformer = SplineTransformer(columns=["price"], n_knots=3, extrapolation="error").fit(numbers)
    with pytest.raises(EngineError, match="beyond the limits of the knots and extrapolation is 'error'"):
        strict.transform_dict({"price": 9.0, "height": 1.0, "device": "ios"})
    outside: str = write_parquet(tmp_path / "outside.parquet", pa.table({"price": [9.0], "height": [1.0]}))
    with pytest.raises(EngineError, match="beyond the limits of the knots"):
        strict.transform_parquet([outside], output_dir=tmp_path / "outside-out")


def test_spline_transform_missing_and_fit_refusals(numbers: list[str], tmp_path: Path) -> None:
    holes: str = write_parquet(tmp_path / "holes.parquet", pa.table({"price": [1.0, None, 8.0]}))
    with pytest.raises(EngineError, match="has nulls and handle_missing is 'error'"):
        SplineTransformer(columns=["price"], n_knots=3).fit([holes])
    zeros: SplineTransformer = SplineTransformer(columns=["price"], n_knots=3, handle_missing="zeros").fit([holes])
    assert zeros.knots_ == {"price": [1.0, 4.5, 8.0]}  # the null teaches nothing
    row: dict[str, Any] = zeros.transform_dict({"price": None})
    assert list(row.values()) == [0.0] * zeros.splines
    assert read_rows(zeros.transform_parquet([holes], output_dir=tmp_path / "holes-out"))[1] == row

    single: str = write_parquet(tmp_path / "single.parquet", pa.table({"price": [3.0]}))
    with pytest.raises(EngineError, match="fewer than two non-null values"):
        SplineTransformer(columns=["price"], n_knots=3).fit([single])
    with pytest.raises(EngineError, match="is utf8, expected an integer or float column"):
        SplineTransformer(columns=["device"]).fit(numbers)
    clash: str = write_parquet(tmp_path / "clash.parquet", pa.table({"price": [1.0, 2.0], "price_sp_0": [1.0, 2.0]}))
    with pytest.raises(EngineError, match="output column 'price_sp_0' already exists in input"):
        SplineTransformer(columns=["price"], n_knots=3).fit([clash])
    spline: SplineTransformer = SplineTransformer(columns=["price"], n_knots=3).fit(numbers)
    with pytest.raises(EngineError, match="key 'price_sp_0' already present in record"):
        spline.transform_dict({"price": 1.0, "price_sp_0": 2.0})


def test_spline_transform_validates_its_arguments() -> None:
    not_a_flag: Any = 1
    with pytest.raises(ValueError, match="n_knots must be between 2 and 4096"):
        SplineTransformer(columns=["a"], n_knots=1)
    with pytest.raises(ValueError, match="n_knots must be between 2 and 4096"):
        SplineTransformer(columns=["a"], n_knots=5000)
    with pytest.raises(ValueError, match="degree must be between 0 and 4096"):
        SplineTransformer(columns=["a"], degree=-1)
    with pytest.raises(ValueError, match="knots must be one of"):
        SplineTransformer(columns=["a"], knots="median")
    with pytest.raises(ValueError, match="extrapolation must be one of"):
        SplineTransformer(columns=["a"], extrapolation="wrap")
    with pytest.raises(ValueError, match="handle_missing must be one of"):
        SplineTransformer(columns=["a"], handle_missing="value")
    with pytest.raises(TypeError, match="include_bias must be True or False"):
        SplineTransformer(columns=["a"], include_bias=not_a_flag)
    with pytest.raises(ValueError, match="periodic extrapolation needs degree below n_knots"):
        SplineTransformer(columns=["a"], n_knots=3, degree=3, extrapolation="periodic")
    with pytest.raises(ValueError, match="more than the 4096 columns allowed"):
        SplineTransformer(columns=["a", "b"], n_knots=3000)


def test_power_transform_fits_one_lambda_per_column(numbers: list[str], tmp_path: Path) -> None:
    power: PowerTransformer = PowerTransformer(columns=["price"], method="box-cox", standardize=False).fit(numbers)
    lambda_: float = power.lambdas_["price"]
    assert 0.0 < lambda_ < 1.0  # price is right-skewed, so the fitted power pulls the tail in
    assert power.means_ is None
    assert power.scales_ is None
    rows: list[dict[str, Any]] = read_rows(power.transform_parquet(numbers, output_dir=tmp_path / "out"))
    assert list(rows[0]) == ["price", "height", "device"]  # rewritten in place
    assert rows[0]["price"] == pytest.approx(0.0, abs=1e-12)  # box-cox(1) is 0 for every lambda
    # Any other value is scipy's boxcox of it at the fitted lambda.
    assert rows[3]["price"] == pytest.approx(math.expm1(lambda_ * math.log(8.0)) / lambda_)
    for index, row in enumerate(rows):
        source: dict[str, Any] = {
            "price": [1.0, 2.0, 4.0, 8.0, 3.0, 5.0][index],
            "height": row["height"],
            "device": row["device"],
        }
        assert power.transform_dict(source) == row

    # standardize centers and scales the transformed column.
    scaled: PowerTransformer = PowerTransformer(columns=["price"], method="box-cox").fit(numbers)
    assert scaled.means_ is not None
    assert scaled.scales_ is not None
    values: list[float] = [
        row["price"] for row in read_rows(scaled.transform_parquet(numbers, output_dir=tmp_path / "s"))
    ]
    assert sum(values) / len(values) == pytest.approx(0.0, abs=1e-12)

    # A null teaches nothing and passes through as a null.
    holes: str = write_parquet(tmp_path / "holes.parquet", pa.table({"price": [1.0, None, 4.0, 8.0]}))
    sparse: PowerTransformer = PowerTransformer(columns=["price"], standardize=False).fit([holes])
    assert sparse.transform_dict({"price": None}) == {"price": None}
    assert read_rows(sparse.transform_parquet([holes], output_dir=tmp_path / "holes-out"))[1] == {"price": None}


def test_power_transform_method_rules(numbers: list[str], tmp_path: Path) -> None:
    signed: str = write_parquet(tmp_path / "signed.parquet", pa.table({"price": [-2.0, 0.0, 1.5, 4.0]}))
    with pytest.raises(EngineError, match="is not strictly positive, which box-cox requires"):
        PowerTransformer(columns=["price"], method="box-cox").fit([signed])
    # Yeo-Johnson takes any value.
    yeo: PowerTransformer = PowerTransformer(columns=["price"], standardize=False).fit([signed])
    assert yeo.transform_dict({"price": -2.0})["price"] < 0.0

    flat: str = write_parquet(tmp_path / "flat.parquet", pa.table({"price": [2.5, 2.5, 2.5]}))
    with pytest.raises(EngineError, match="is constant, which box-cox cannot fit a lambda for"):
        PowerTransformer(columns=["price"], method="box-cox").fit([flat])
    constant: PowerTransformer = PowerTransformer(columns=["price"], standardize=False).fit([flat])
    assert constant.lambdas_["price"] == 1.0  # the identity, as scikit-learn leaves it

    positive: PowerTransformer = PowerTransformer(columns=["price"], method="box-cox", standardize=False).fit(numbers)
    with pytest.raises(EngineError, match="is not strictly positive, which box-cox requires"):
        positive.transform_dict({"price": 0.0, "height": 1.0, "device": "ios"})
    nonpositive: str = write_parquet(tmp_path / "zero.parquet", pa.table({"price": [0.0], "height": [1.0]}))
    with pytest.raises(EngineError, match="is not strictly positive"):
        positive.transform_parquet([nonpositive], output_dir=tmp_path / "zero-out")

    short: str = write_parquet(tmp_path / "short.parquet", pa.table({"price": [3.0]}))
    with pytest.raises(EngineError, match="fewer than two non-null values"):
        PowerTransformer(columns=["price"]).fit([short])
    with pytest.raises(EngineError, match="is utf8, expected an integer or float column"):
        PowerTransformer(columns=["device"]).fit(numbers)


def test_power_transform_validates_its_arguments() -> None:
    not_a_flag: Any = 1
    with pytest.raises(ValueError, match="columns must name at least one column"):
        PowerTransformer(columns=[])
    with pytest.raises(ValueError, match="method must be one of"):
        PowerTransformer(columns=["a"], method="log")
    with pytest.raises(TypeError, match="standardize must be True or False"):
        PowerTransformer(columns=["a"], standardize=not_a_flag)


def test_expand_kernels_are_independent_of_the_split_and_survive_the_artifact(
    numbers: list[str], tmp_path: Path
) -> None:
    for shell in (
        SplineTransformer(columns=["price"], n_knots=3, knots="quantile"),
        PowerTransformer(columns=["price"], method="box-cox"),
    ):
        states: set[str] = set()
        for threads in (1, 4):
            shell.fit(numbers, threads=threads)
            states.add(shell._require_handle().stage_state(type(shell).__name__.lower()))
        assert len(states) == 1

    pipeline: Pipeline = Pipeline(
        [
            ("cast", Cast(types={"price": "f64"})),
            ("power", PowerTransformer(columns=["price"], method="box-cox")),
            ("spline", SplineTransformer(columns=["height"], n_knots=3, degree=1)),
            ("expand", PolynomialFeatures(columns=["price"], degree=(1, 2), include_bias=False)),
            ("keep", Select(columns=["price^2", "height_sp_0"])),
        ]
    ).fit(numbers)
    record: dict[str, Any] = {"price": 4.0, "height": 12.0, "device": "ios"}
    expected: dict[str, Any] = pipeline.transform_dict(record)
    assert set(expected) == {"price^2", "height_sp_0"}
    pipeline.save(tmp_path / "artifact")
    manifest: dict[str, Any] = json.loads((tmp_path / "artifact" / "manifest.json").read_text())
    assert manifest["ops"][2]["state"]["knots"] == {"height": [9.0, 12.0, 15.0]}
    assert manifest["ops"][3]["state"] == {}  # polynomial_features learns nothing
    assert manifest["ops"][3]["params"]["degree"] == [1, 2]  # a degree range round-trips as a pair
    assert Pipeline.load(tmp_path / "artifact").transform_dict(record) == expected
    assert pickle.loads(pickle.dumps(pipeline)).transform_dict(record) == expected

    manifest["ops"][2]["state"]["knots"]["height"] = [9.0, 15.0, 12.0]
    with pytest.raises(EngineError, match="must be sorted"):
        _core.FittedPipeline.load_json(json.dumps(manifest))
    manifest["ops"][2]["state"]["knots"]["height"] = [9.0, 12.0, 15.0]
    manifest["ops"][1]["state"]["scales"]["price"] = 0.0
    with pytest.raises(EngineError, match="must be positive"):
        _core.FittedPipeline.load_json(json.dumps(manifest))


def test_the_expanders_never_hand_a_non_finite_value_downstream(tmp_path: Path) -> None:
    """A generated column the next step could not read is an error at the step that made it.

    ``a^2`` for ``a = 1e200`` overflows to infinity, and a spline basis
    continued far past its knots does the same. Every kernel here refuses a
    non-finite number when it reads one, so an expander that emitted one built
    a pipeline that fitted, wrote its manifest, and then failed at the
    following step or, worse, wrote the infinity into the output. Both
    transform paths refuse it where it is produced, naming the generated
    column.

    Args:
        tmp_path: pytest temporary directory.

    """
    wide: str = write_parquet(tmp_path / "wide.parquet", pa.table({"a": [1e200, 2.0]}))
    poly: PolynomialFeatures = PolynomialFeatures(columns=["a"], degree=2).fit([wide])
    with pytest.raises(EngineError, match=r"column 'a\^2' generates the value inf, which is not finite"):
        poly.transform_parquet([wide], output_dir=tmp_path / "poly")
    with pytest.raises(EngineError, match=r"column 'a\^2' generates the value inf, which is not finite"):
        poly.transform_dict({"a": 1e200})
    # The pipeline that used to fail one step late now fails at the source.
    with pytest.raises(EngineError, match="generates the value inf"):
        Pipeline([("pf", PolynomialFeatures(columns=["a"], degree=2)), ("ss", StandardScaler(columns=["a^2"]))]).fit(
            [wide]
        )

    small: str = write_parquet(tmp_path / "small.parquet", pa.table({"a": [0.0, 1.0, 2.0, 3.0]}))
    far: str = write_parquet(tmp_path / "far.parquet", pa.table({"a": [1e300]}))
    spline: SplineTransformer = SplineTransformer(columns=["a"], n_knots=3, degree=2, extrapolation="continue").fit(
        [small]
    )
    with pytest.raises(EngineError, match="generates the value"):
        spline.transform_parquet([far], output_dir=tmp_path / "spline")
    with pytest.raises(EngineError, match="generates the value"):
        spline.transform_dict({"a": 1e300})
    # Values inside the fitted range transform without error, all finite.
    rows: list[dict[str, Any]] = read_rows(spline.transform_parquet([small], output_dir=tmp_path / "ok"))
    assert all(math.isfinite(value) for row in rows for value in row.values())


def test_spline_transformer_refuses_a_width_that_generates_nothing() -> None:
    """Parameters that produce no output columns are refused at construction.

    ``n_knots=2, degree=1, extrapolation="periodic", include_bias=False`` has
    ``splines == 0``. The constructor accepted it, ``fit()`` succeeded and wrote
    a manifest, and ``transform_parquet`` then raised "cannot write 1 rows with
    no columns" — at serving, on an artifact that had validated cleanly.
    """
    with pytest.raises(ValueError, match="generate no columns at all"):
        SplineTransformer(columns=["a"], n_knots=2, degree=1, extrapolation="periodic", include_bias=False)
    with pytest.raises(ValueError, match="generate no columns at all"):
        SplineTransformer(columns=["a"], n_knots=2, degree=0, include_bias=False)


def test_spline_knots_do_not_depend_on_the_row_group_split(tmp_path: Path) -> None:
    """A -0.0 among the zeros must not reach the knot table.

    ``min()`` and ``max()`` return their first argument on a tie and ``sort`` is
    not stable, so the sign of zero used to depend on which worker claimed which
    row group: two fits of the same file gave different manifest bytes, which
    breaks artifact hashing and any "refit is a no-op" check. The transform
    answers were always the same, which is why only the bytes tell.

    Args:
        tmp_path: pytest temporary directory.

    """
    values: list[float] = [-0.0, 0.0, -0.0, 1.0, 0.0, 2.0, -0.0, 3.0]
    states: set[str] = set()
    for row_group_size in (1, 2, 3, 8):
        source: str = write_parquet(
            tmp_path / f"zeros-{row_group_size}.parquet",
            pa.table({"a": values}),
            row_group_size=row_group_size,
        )
        for threads in (1, 4):
            fitted: SplineTransformer = SplineTransformer(columns=["a"], n_knots=3, degree=2, knots="quantile").fit(
                [source], threads=threads
            )
            states.add(json.dumps(fitted._state(), sort_keys=True))
    assert len(states) == 1, states
    assert "-0.0" not in states.pop()
