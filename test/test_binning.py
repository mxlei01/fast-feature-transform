"""Tests for the robust_scale, quantile_transform and k_bins_discretize kernels: fits, nulls, and the artifact."""

import json
import math
import pickle
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fast_feature_transform import (
    Cast,
    EngineError,
    KBinsDiscretizer,
    Pipeline,
    QuantileTransformer,
    RobustScaler,
    Select,
    _core,
)


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


def _rows(parts: list[Path]) -> list[dict[str, Any]]:
    """Read output parts back as records.

    Args:
        parts: Part files in order.

    Returns:
        The rows as dicts.

    """
    return pa.concat_tables([pq.read_table(part) for part in parts]).to_pylist()


@pytest.fixture
def prices(tmp_path: Path) -> list[str]:
    """One file of eight prices over four row groups, so the merge has work to do.

    Args:
        tmp_path: pytest-provided temporary directory.

    Returns:
        Paths of the written Parquet files.

    """
    table: pa.Table = pa.table(
        {"price": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], "tag": ["a", "b", "c", "d", "e", "f", "g", "h"]}
    )
    return [_write(tmp_path / "prices.parquet", table, row_group_size=2)]


def test_robust_scaler_centres_on_the_median_and_scales_by_the_quantile_range(
    prices: list[str], tmp_path: Path
) -> None:
    scaler: RobustScaler = RobustScaler(columns=["price"]).fit(prices)
    assert scaler.center_ == {"price": pytest.approx(4.5)}  # the median of 1..8
    assert scaler.scale_ == {"price": pytest.approx(3.5)}  # 6.25 - 2.75

    table: pa.Table = pa.concat_tables(
        [pq.read_table(part) for part in scaler.transform_parquet(prices, output_dir=tmp_path / "out")]
    )
    assert table.schema.field("price").type == pa.float64()
    assert table.column("price").to_pylist()[0] == pytest.approx(-1.0)
    assert scaler.transform_dict({"price": 1.0, "tag": "a"}) == {"price": pytest.approx(-1.0), "tag": "a"}
    assert scaler.transform_dict({"price": 4.5}) == {"price": pytest.approx(0.0)}

    # The flags leave a statistic unfitted, and the artifact says so.
    plain: RobustScaler = RobustScaler(columns=["price"], with_centering=False, with_scaling=False).fit(prices)
    assert plain.center_ == {"price": None}
    assert plain.scale_ == {"price": None}
    assert plain.transform_dict({"price": 2.0}) == {"price": 2.0}

    # A wider range is a wider scale; unit_variance divides it by the same
    # range on a standard normal.
    wide: RobustScaler = RobustScaler(columns=["price"], quantile_range=(10.0, 90.0)).fit(prices)
    assert wide.scale_["price"] == pytest.approx(5.6)
    unit: RobustScaler = RobustScaler(columns=["price"], unit_variance=True).fit(prices)
    assert unit.scale_["price"] == pytest.approx(3.5 / 1.3489795003921634)


def test_robust_scaler_unit_variance_reads_a_tiny_q_min_as_scikit_learn_does(tmp_path: Path) -> None:
    """A q_min whose q_min / 100 is below the smallest normal double fits scikit-learn's scale.

    REGRESSION: the inverse normal CDF used to answer NaN below about 6e-311,
    so q_min = 1e-318 failed the fit with "its values are too large".
    """
    powers: str = _write(tmp_path / "powers.parquet", pa.table({"price": [1.0, 2.0, 4.0, 8.0, 16.0]}))
    scaler: RobustScaler = RobustScaler(columns=["price"], quantile_range=(1e-318, 75.0), unit_variance=True)
    assert scaler.fit([powers]).scale_["price"] == pytest.approx(0.17974705181433237, rel=1e-12)


def test_robust_scaler_ignores_nulls_and_maps_a_zero_scale_to_one(tmp_path: Path) -> None:
    holes: str = _write(tmp_path / "holes.parquet", pa.table({"price": [1.0, None, 3.0, None, 5.0, 9.0]}))
    scaler: RobustScaler = RobustScaler(columns=["price"]).fit([holes])
    assert scaler.center_["price"] == pytest.approx(4.0)  # the median of 1, 3, 5, 9
    assert scaler.transform_dict({"price": None}) == {"price": None}
    assert [row["price"] for row in _rows(scaler.transform_parquet([holes], output_dir=tmp_path / "out"))][1] is None

    flat: str = _write(tmp_path / "flat.parquet", pa.table({"price": [3.0, 3.0, 3.0]}))
    constant: RobustScaler = RobustScaler(columns=["price"]).fit([flat])
    assert constant.scale_ == {"price": pytest.approx(1.0)}  # the near-zero rule, not a division by zero
    assert constant.transform_dict({"price": 5.0}) == {"price": pytest.approx(2.0)}

    # Every numeric column, integers included: they widen to f64.
    integers: str = _write(tmp_path / "ints.parquet", pa.table({"n": [1, 2, 3, 4], "tag": ["a", "b", "c", "d"]}))
    every: RobustScaler = RobustScaler().fit([integers])
    assert set(every.center_) == {"n"}
    assert every.transform_dict({"n": 1, "tag": "a"}) == {"n": pytest.approx(-1.0), "tag": "a"}


def test_quantile_transformer_maps_into_the_fitted_grid(prices: list[str], tmp_path: Path) -> None:
    transformer: QuantileTransformer = QuantileTransformer(columns=["price"], n_quantiles=3).fit(prices)
    assert transformer.n_quantiles_ == 3
    assert transformer.references_ == pytest.approx([0.0, 0.5, 1.0])
    assert transformer.quantiles_ == {"price": pytest.approx([1.0, 4.5, 8.0])}

    table: pa.Table = pa.concat_tables(
        [pq.read_table(part) for part in transformer.transform_parquet(prices, output_dir=tmp_path / "out")]
    )
    assert table.schema.field("price").type == pa.float64()
    assert table.column("price").to_pylist()[0] == pytest.approx(0.0)
    assert transformer.transform_dict({"price": 8.0, "tag": "h"}) == {"price": pytest.approx(1.0), "tag": "h"}
    # Outside the fitted range the rank is clipped, which is what makes this
    # the one scaler that always flattens an outlier instead of rescaling it.
    assert transformer.transform_dict({"price": 1000.0}) == {"price": pytest.approx(1.0)}
    assert transformer.transform_dict({"price": -1000.0}) == {"price": pytest.approx(0.0)}
    assert transformer.transform_dict({"price": None}) == {"price": None}

    # n_quantiles is capped at the fit row count, as upstream caps it.
    capped: QuantileTransformer = QuantileTransformer(columns=["price"], n_quantiles=1000).fit(prices)
    assert capped.n_quantiles_ == 8

    # output_distribution="normal" maps the rank through the inverse normal
    # CDF and clips the bounds so they cannot reach infinity.
    normal: QuantileTransformer = QuantileTransformer(
        columns=["price"], n_quantiles=3, output_distribution="normal"
    ).fit(prices)
    assert normal.transform_dict({"price": 4.5})["price"] == pytest.approx(0.0)
    assert normal.transform_dict({"price": 8.0})["price"] == pytest.approx(5.199337582605575)
    assert normal.transform_dict({"price": 1.0})["price"] == pytest.approx(-5.199337582605575)


def test_quantile_transformer_averages_the_two_interpolations_over_ties(tmp_path: Path) -> None:
    # The grid is 0, 1, 1, 1, 2 over the references 0, 0.25, 0.5, 0.75, 1. For
    # a 1, forward interpolation alone gives 0.75, the last tie's reference,
    # and backward alone 0.25, the first's; averaged, 0.5. The expected values
    # are scikit-learn 1.9.0's.
    ties: str = _write(tmp_path / "ties.parquet", pa.table({"price": [0.0, 1.0, 1.0, 1.0, 2.0]}))
    transformer: QuantileTransformer = QuantileTransformer(columns=["price"], n_quantiles=5).fit([ties])
    assert transformer.quantiles_ == {"price": [0.0, 1.0, 1.0, 1.0, 2.0]}
    expected: dict[float, float] = {1.0: 0.5, 0.5: 0.125, 1.5: 0.875, 0.0: 0.0, 2.0: 1.0}
    assert {value: transformer.transform_dict({"price": value})["price"] for value in expected} == expected


def test_quantile_transformer_clips_a_rank_below_the_smallest_normal_double(tmp_path: Path) -> None:
    """A rank too small for a normal double still maps to the lower clip, as in scikit-learn.

    REGRESSION: over 0 and 2**1023, the value 2**-20 ranks at 2**-1044, where
    the inverse normal CDF used to answer NaN, which the clip let through.
    """
    wide: str = _write(tmp_path / "wide.parquet", pa.table({"price": [0.0, 2.0**1023]}))
    normal: QuantileTransformer = QuantileTransformer(
        columns=["price"], n_quantiles=2, output_distribution="normal"
    ).fit([wide])
    clip: float = normal.transform_dict({"price": 0.0})["price"]
    assert clip == pytest.approx(-5.199337582605575, rel=1e-12)
    for value in (2.0**-20, 2.0**-10):
        assert normal.transform_dict({"price": value})["price"] == clip


def test_k_bins_discretizer_bins_every_strategy(prices: list[str], tmp_path: Path) -> None:
    uniform: KBinsDiscretizer = KBinsDiscretizer(columns=["price"], n_bins=4, encode="ordinal", strategy="uniform").fit(
        prices
    )
    assert uniform.bin_edges_ == {"price": pytest.approx([1.0, 2.75, 4.5, 6.25, 8.0])}
    assert uniform.n_bins_ == {"price": 4}
    table: pa.Table = pa.concat_tables(
        [pq.read_table(part) for part in uniform.transform_parquet(prices, output_dir=tmp_path / "out")]
    )
    assert table.schema.field("price").type == pa.int64()
    assert table.column("price").to_pylist() == [0, 0, 1, 1, 2, 2, 3, 3]
    assert uniform.transform_dict({"price": -100.0, "tag": "a"}) == {"price": 0, "tag": "a"}
    assert uniform.transform_dict({"price": 100.0, "tag": "a"}) == {"price": 3, "tag": "a"}

    quantile: KBinsDiscretizer = KBinsDiscretizer(columns=["price"], n_bins=4, encode="ordinal").fit(prices)
    assert quantile.n_bins_ == {"price": 4}
    kmeans: KBinsDiscretizer = KBinsDiscretizer(columns=["price"], n_bins=2, encode="ordinal", strategy="kmeans").fit(
        prices
    )
    assert kmeans.bin_edges_["price"][1] == pytest.approx(4.5)


def test_k_bins_discretizer_expands_to_one_column_per_bin(prices: list[str], tmp_path: Path) -> None:
    for encode in ("onehot", "onehot-dense"):  # dense output either way: sparse_output is dropped
        binner: KBinsDiscretizer = KBinsDiscretizer(columns=["price"], n_bins=2, encode=encode, strategy="uniform").fit(
            prices
        )
        rows: list[dict[str, Any]] = _rows(binner.transform_parquet(prices, output_dir=tmp_path / f"out-{encode}"))
        assert set(rows[0]) == {"tag", "price_0", "price_1"}
        assert [row["price_0"] for row in rows] == [1, 1, 1, 1, 0, 0, 0, 0]
        assert binner.transform_dict({"price": 8.0, "tag": "h"}) == {"tag": "h", "price_0": 0, "price_1": 1}
        # A record key named like a generated column would be dropped silently.
        with pytest.raises(EngineError, match="key 'price_0' already present in record"):
            binner.transform_dict({"price": 1.0, "price_0": 9})


def test_k_bins_discretizer_drops_collapsed_bins_and_bins_a_constant_column(tmp_path: Path) -> None:
    # The low percentiles all land on 1, so upstream drops those edges.
    ties: str = _write(tmp_path / "ties.parquet", pa.table({"price": [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 5.0, 9.0]}))
    binner: KBinsDiscretizer = KBinsDiscretizer(columns=["price"], n_bins=5, encode="ordinal").fit([ties])
    assert binner.n_bins_["price"] < 5

    flat: str = _write(tmp_path / "flat.parquet", pa.table({"price": [3.0, 3.0, 3.0]}))
    constant: KBinsDiscretizer = KBinsDiscretizer(columns=["price"], n_bins=4, encode="onehot").fit([flat])
    assert constant.n_bins_ == {"price": 1}
    assert constant.transform_dict({"price": 99.0}) == {"price_0": 1}

    # Four more of numpy's quantile methods fit the tied column too.
    for method in ("inverted_cdf", "hazen", "linear", "normal_unbiased"):
        chosen: KBinsDiscretizer = KBinsDiscretizer(
            columns=["price"], n_bins=3, encode="ordinal", quantile_method=method
        ).fit([ties])
        assert chosen.n_bins_["price"] >= 1


def test_k_bins_discretizer_refuses_a_null(tmp_path: Path) -> None:
    holes: str = _write(tmp_path / "holes.parquet", pa.table({"price": [1.0, None, 3.0]}))
    with pytest.raises(EngineError, match="column 'price' has nulls, which k_bins_discretize refuses"):
        KBinsDiscretizer(columns=["price"], strategy="uniform").fit([holes])
    whole: str = _write(tmp_path / "whole.parquet", pa.table({"price": [1.0, 2.0, 3.0, 4.0]}))
    binner: KBinsDiscretizer = KBinsDiscretizer(columns=["price"], n_bins=2, encode="ordinal").fit([whole])
    with pytest.raises(EngineError, match="is null, which k_bins_discretize refuses"):
        binner.transform_dict({"price": None})
    with pytest.raises(EngineError, match="has a null, which k_bins_discretize refuses"):
        binner.transform_parquet([holes], output_dir=tmp_path / "out")


def test_binning_refuses_a_value_that_is_not_finite(tmp_path: Path) -> None:
    # A null is how a missing value is spelled here (it is what scikit-learn's
    # NaN maps onto), so a float NaN or infinity is an error everywhere.
    broken: str = _write(tmp_path / "broken.parquet", pa.table({"price": [1.0, float("nan"), 3.0]}))
    whole: str = _write(tmp_path / "whole.parquet", pa.table({"price": [1.0, 2.0, 3.0, 4.0]}))
    # KBinsDiscretizer refuses a null too, so only the other two suggest one.
    for shell, message in (
        (RobustScaler(columns=["price"]), "is not finite; write a missing value as a null"),
        (QuantileTransformer(columns=["price"], n_quantiles=3), "is not finite; write a missing value as a null"),
        (KBinsDiscretizer(columns=["price"], n_bins=2, encode="ordinal"), r"is not finite$"),
    ):
        with pytest.raises(EngineError, match=message):
            type(shell)(**shell.params()).fit([broken])
        fitted = type(shell)(**shell.params()).fit([whole])
        with pytest.raises(EngineError, match=message):
            fitted.transform_parquet([broken], output_dir=tmp_path / "out")
        with pytest.raises(EngineError, match=message):
            fitted.transform_dict({"price": float("inf")})

    holes: str = _write(
        tmp_path / "holes.parquet", pa.table({"price": [None, None]}, schema=pa.schema([("price", pa.float64())]))
    )
    for scaler in (RobustScaler(columns=["price"]), QuantileTransformer(columns=["price"])):
        with pytest.raises(EngineError, match="has no non-null values to learn from"):
            type(scaler)(**scaler.params()).fit([holes])


def test_binning_is_independent_of_the_split_and_survives_the_artifact(prices: list[str], tmp_path: Path) -> None:
    shells: list[Any] = [
        RobustScaler(columns=["price"]),
        QuantileTransformer(columns=["price"], n_quantiles=4),
        KBinsDiscretizer(columns=["price"], n_bins=3, encode="ordinal"),
        KBinsDiscretizer(columns=["price"], n_bins=3, encode="ordinal", strategy="kmeans"),
        KBinsDiscretizer(columns=["price"], n_bins=3, encode="ordinal", strategy="uniform"),
    ]
    for shell in shells:
        states: set[str] = set()
        for threads in (1, 4):
            fitted = type(shell)(**shell.params()).fit(prices, threads=threads)
            states.add(fitted._require_handle().stage_state(type(shell).__name__.lower()))
        assert len(states) == 1, f"{type(shell).__name__} depends on the thread count"

    # encode="ordinal" keeps the column in place, so a later step can still
    # name it; the one-hot width is learned and cannot be named in the same
    # pipeline.
    pipeline: Pipeline = Pipeline(
        [
            ("bin", KBinsDiscretizer(columns=["price"], n_bins=2, encode="ordinal", strategy="uniform")),
            ("keep", Select(["price"])),
        ]
    ).fit(prices)
    pipeline.save(tmp_path / "artifact")
    manifest: dict[str, Any] = json.loads((tmp_path / "artifact" / "manifest.json").read_text())
    assert manifest["ops"][0]["state"]["bin_edges"] == {"price": pytest.approx([1.0, 4.5, 8.0])}
    record: dict[str, Any] = {"price": 1.0, "tag": "a"}
    assert Pipeline.load(tmp_path / "artifact").transform_dict(record) == pipeline.transform_dict(record)
    assert pickle.loads(pickle.dumps(pipeline)).transform_dict(record) == pipeline.transform_dict(record)
    assert pipeline.transform_dict(record) == {"price": 0}

    # A transform reads only the interior edges, so only those must not
    # decrease; a fit can write an outer edge out of order.
    manifest["ops"][0]["params"]["n_bins"] = 3
    manifest["ops"][0]["state"]["bin_edges"]["price"] = [1.0, 6.0, 4.5, 8.0]
    with pytest.raises(EngineError, match="must not decrease between its outer edges"):
        _core.FittedPipeline.load_json(json.dumps(manifest))


def test_binning_validates_its_arguments(prices: list[str], tmp_path: Path) -> None:
    text: Any = "nope"
    bare: Any = "price"
    short: Any = (25.0,)
    with pytest.raises(TypeError, match="columns must be a list of column names"):
        RobustScaler(columns=bare)
    with pytest.raises(ValueError, match="columns must name at least one column"):
        RobustScaler(columns=[])
    with pytest.raises(TypeError, match="with_centering must be True or False"):
        RobustScaler(with_centering=text)
    with pytest.raises(TypeError, match="quantile_range must be two percentiles"):
        RobustScaler(quantile_range=short)
    with pytest.raises(ValueError, match="quantile_range must satisfy"):
        RobustScaler(quantile_range=(75.0, 25.0))
    with pytest.raises(ValueError, match="unit_variance needs quantile_range strictly inside"):
        RobustScaler(quantile_range=(0.0, 100.0), unit_variance=True)
    with pytest.raises(ValueError, match=r"needs q_min / 100 above 0, and 2\.4e-322 / 100 rounds to 0"):
        RobustScaler(quantile_range=(2.4e-322, 75.0), unit_variance=True)

    with pytest.raises(TypeError, match="n_quantiles must be an integer"):
        QuantileTransformer(n_quantiles=text)
    for count in (0, 100_001):
        with pytest.raises(ValueError, match="n_quantiles must be between 1 and 100000"):
            QuantileTransformer(n_quantiles=count)
    with pytest.raises(ValueError, match="output_distribution must be one of"):
        QuantileTransformer(output_distribution="poisson")

    for count in (1, 4097):
        with pytest.raises(ValueError, match="n_bins must be between 2 and 4096"):
            KBinsDiscretizer(n_bins=count)
    with pytest.raises(ValueError, match="encode must be one of"):
        KBinsDiscretizer(encode="sparse")
    with pytest.raises(ValueError, match="strategy must be one of"):
        KBinsDiscretizer(strategy="gaussian")
    with pytest.raises(ValueError, match="quantile_method must be one of"):
        KBinsDiscretizer(quantile_method="nearest")

    strings: str = _write(tmp_path / "strings.parquet", pa.table({"tag": ["a", "b"]}))
    with pytest.raises(EngineError, match="no integer or float columns to scale"):
        RobustScaler().fit([strings])
    with pytest.raises(EngineError, match="is utf8, expected an integer or float column"):
        RobustScaler(columns=["tag"]).fit([strings])
    clash: str = _write(tmp_path / "clash.parquet", pa.table({"price": [1.0, 2.0], "price_0": [1.0, 2.0]}))
    with pytest.raises(EngineError, match="output column 'price_0' already exists in input"):
        KBinsDiscretizer(columns=["price"], n_bins=2, strategy="uniform").fit([clash])

    # A string column is an error, not a parse: Cast is the one step that
    # changes a type.
    numbers: str = _write(tmp_path / "numbers.parquet", pa.table({"price": ["1.0", "2.0", "3.0", "4.0"]}))
    cast: Pipeline = Pipeline([("cast", Cast(types={"price": "f64"})), ("scale", RobustScaler(columns=["price"]))]).fit(
        [numbers]
    )
    assert cast.transform_dict({"price": "1.0"})["price"] == pytest.approx(-1.0)


def test_kmeans_reproduces_upstream_empty_cluster_relocation(tmp_path: Path) -> None:
    """A kmeans fit that empties a cluster bins the way scikit-learn 1.9.0 does.

    ``[0, 0, 0, 0, 1, 5, 0, 0]`` at ``n_bins=5`` leaves three clusters empty on
    the first Lloyd pass. Upstream's ``_relocate_empty_clusters_dense`` keeps
    ``labels`` const while it moves a point's weight, skips relocation when
    every distance is zero, and ``_average_centers`` puts a cluster that is
    still empty at the biggest cluster's slot as that slot stands — the
    unaveraged sum when the biggest cluster comes later. All three together are
    what makes the edges ``[0, 0, 0.5, 3, 5]`` and the bins
    ``[1, 1, 1, 1, 2, 3, 1, 1]``; missing any one binned every row differently.

    Args:
        tmp_path: pytest temporary directory.

    """
    values: list[float] = [0.0, 0.0, 0.0, 0.0, 1.0, 5.0, 0.0, 0.0]
    source: str = _write(tmp_path / "empty.parquet", pa.table({"price": values}))
    binner: KBinsDiscretizer = KBinsDiscretizer(columns=["price"], n_bins=5, encode="ordinal", strategy="kmeans").fit(
        [source]
    )
    assert binner.bin_edges_ == {"price": [0.0, 0.0, 0.5, 3.0, 5.0]}
    rows: list[dict[str, Any]] = _rows(binner.transform_parquet([source], output_dir=tmp_path / "out"))
    assert [row["price"] for row in rows] == [1, 1, 1, 1, 2, 3, 1, 1]
    for value, row in zip(values, rows, strict=True):
        assert binner.transform_dict({"price": value}) == row


def test_kmeans_stops_at_the_first_pass_whose_centers_barely_move(tmp_path: Path) -> None:
    """A kmeans fit stops once its centers move less than the tolerance, as scikit-learn 1.9.0 does.

    On the cubes of 0 to 299 at ``n_bins=3`` that is the ninth Lloyd pass, three
    before the labels stop changing, and the edges are that pass's. The squared
    center shifts are summed afresh in every pass: a sum carried over from the
    earlier passes never falls under the tolerance, and the fit then runs on to
    other edges.

    Args:
        tmp_path: pytest temporary directory.

    """
    source: str = _write(tmp_path / "cubes.parquet", pa.table({"price": [float(i**3) for i in range(300)]}))
    binner: KBinsDiscretizer = KBinsDiscretizer(columns=["price"], n_bins=3, encode="ordinal", strategy="kmeans").fit(
        [source]
    )
    assert binner.bin_edges_ == {"price": [0.0, 5868953.75, 15616841.25, 26730899.0]}


def test_every_fitted_bin_table_loads_back(tmp_path: Path) -> None:
    """Round-trip every fitted bin table: fit -> state() -> save() -> load() holds for every column shape a fit accepts.

    ``linspace`` over a span of a few ulps returns edges that repeat, which is
    what ``strategy="uniform"`` stores (upstream stores them too). ``load_state``
    demanded a strictly increasing table, so ``save()`` succeeded and ``load()``
    threw: the failure surfaced at serving, on an artifact that had validated at
    fit.

    Args:
        tmp_path: pytest temporary directory.

    """
    columns: dict[str, list[float]] = {
        "two-ulp": [1e16, math.nextafter(1e16, 2e16), 1e16, math.nextafter(1e16, 2e16)],
        "adjacent": [1.0, math.nextafter(1.0, 2.0), 1.0, math.nextafter(1.0, 2.0)],
        "denormal": [0.0, 5e-324, 0.0, 5e-324],
        "ordinary": [1.0, 2.0, 3.0, 4.0],
        "constant": [3.0, 3.0, 3.0, 3.0],
    }
    for what, values in columns.items():
        source: str = _write(tmp_path / f"{what}.parquet", pa.table({"price": values}))
        for strategy in ("uniform", "quantile", "kmeans"):
            pipeline: Pipeline = Pipeline(
                [("bin", KBinsDiscretizer(columns=["price"], n_bins=3, encode="ordinal", strategy=strategy))]
            )
            try:
                fitted: Pipeline = pipeline.fit([source])
            except EngineError:
                continue  # a fit that refuses outright leaves no artifact to reload
            directory: Path = tmp_path / f"artifact-{what}-{strategy}"
            fitted.save(directory)
            record: dict[str, Any] = {"price": values[0]}
            assert Pipeline.load(directory).transform_dict(record) == fitted.transform_dict(record), (what, strategy)
            assert pickle.loads(pickle.dumps(fitted)).transform_dict(record) == fitted.transform_dict(record)


def test_a_fitted_table_naming_no_columns_is_refused(prices: list[str], tmp_path: Path) -> None:
    """A hand-edited manifest with an empty fitted table, or an empty or missing column list, is refused at load.

    With ``columns=None`` the state's ``columns`` list names the fitted
    columns, and the fitted table must name the same ones. An empty table used
    to load as a fitted step owning nothing: transform passed every row
    through, and for ``encode="onehot"`` the whole generated block was missing
    from the output, with no error anywhere.

    Args:
        prices: The shared Parquet fixture.
        tmp_path: pytest temporary directory.

    """
    tables: dict[str, str] = {
        "bin_edges": "KBinsDiscretizer",
        "quantiles": "QuantileTransformer",
        "center": "RobustScaler",
    }
    for key, name in tables.items():
        transformer: Any = {
            "KBinsDiscretizer": KBinsDiscretizer(n_bins=2, encode="onehot", strategy="uniform"),
            "QuantileTransformer": QuantileTransformer(n_quantiles=2),
            "RobustScaler": RobustScaler(),
        }[name]
        fitted: Pipeline = Pipeline([("step", transformer)]).fit(prices)
        fitted.save(tmp_path / f"artifact-{key}")
        manifest: dict[str, Any] = json.loads((tmp_path / f"artifact-{key}" / "manifest.json").read_text())
        emptied_table: dict[str, Any] = json.loads(json.dumps(manifest))
        emptied_table["ops"][0]["state"][key] = {}
        with pytest.raises(EngineError, match="state 'columns' does not name the same columns"):
            _core.FittedPipeline.load_json(json.dumps(emptied_table))
        emptied_list: dict[str, Any] = json.loads(json.dumps(manifest))
        emptied_list["ops"][0]["state"] = {**emptied_list["ops"][0]["state"], key: {}, "columns": []}
        with pytest.raises(EngineError, match="state must hold a 'columns' list"):
            _core.FittedPipeline.load_json(json.dumps(emptied_list))
        dropped_list: dict[str, Any] = json.loads(json.dumps(manifest))
        del dropped_list["ops"][0]["state"]["columns"]
        with pytest.raises(EngineError, match="state must hold a 'columns' list"):
            _core.FittedPipeline.load_json(json.dumps(dropped_list))
