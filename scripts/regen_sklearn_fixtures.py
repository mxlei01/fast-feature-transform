"""Regenerate the scikit-learn parity fixture.

Runs the real library once per case and records what it produced, so the
normal test run can compare against it without calling scikit-learn or
installing pandas. Run it with the versions pinned (scipy fits
PowerTransformer's lambdas), as make sklearn-fixtures does:

    uv run --isolated --no-project --python 3.14t \
        --with scikit-learn==1.9.0 --with scipy==1.18.1 --with pandas \
        python scripts/regen_sklearn_fixtures.py

Cases name our transformer and our parameter spelling; UPSTREAM maps both to
sklearn.preprocessing, dropping any parameter it maps to None. ``columns`` is
always one of those: scikit-learn transforms every column of the frame it is
handed, so run_case narrows the frame to the columns a case names and leaves
the rest as pass-through columns that our output carries and the library
never sees.

A dataset holds numeric columns only, and ``None`` is a null; pandas reads it
as the NaN the scalers ignore at fit and pass through at transform. Dataset
names are prefixed by the family that reads them (``scale-``, ``bin-``,
``expand-``), because each of the three families needs its own "nulls" frame.

Every dataset is small on purpose. The transformers upstream subsamples with
an RNG only do so above their ``subsample`` threshold (10,000 rows for
QuantileTransformer, 200,000 for KBinsDiscretizer); ours have no
``subsample`` or ``random_state`` and fit exactly on every row, so every case
has to sit below those thresholds, where upstream would not have sampled
either, or the two sides would be computing different things rather than the
same thing two ways.

A case may carry a "tolerance", an absolute difference test_parity allows
instead of demanding the exact value. Four kinds of case need one: the inverse
normal CDF behind ``unit_variance`` and ``output_distribution="normal"``;
``strategy="kmeans"``, whose Lloyd iteration sums in a different order from
scikit-learn's; PowerTransformer's lambda, the maximizer of a flat likelihood
found by Brent's method; and the SplineTransformer cases of degree 2 or more,
whose basis scipy evaluates with a fused multiply-add. The Box-Cox cases whose
lambda is capped carry a "relative_tolerance" instead, since their values
reach 1e304 and more. Every other case is compared for equality of the
IEEE-754 doubles, not near-equality.

An absolute tolerance is a slack over the data below, not a bound on the
transformer. The comment on each constant says why it is needed and, for
PowerTransformer and SplineTransformer, what was measured and where it stops
holding.
"""

import json
import math
from pathlib import Path
from typing import Any

VERSION = "1.9.0"

DATASETS: dict[str, dict[str, list[Any]]] = {
    # Dyadic values throughout, so a mean, a sum of squares and a range are
    # all exact in binary: scikit-learn's incrementally accumulated variance
    # and our exact one then agree bit for bit and every case compares exactly.
    "scale-numbers": {
        "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "b": [-4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0],
        "c": [0.5, 0.25, 0.75, 1.0, 0.125, 0.375, 0.625, 0.875],
    },
    # An integer column reaches scikit-learn as int64 and is cast to float64
    # there; ours is i64 in Parquet and widened the same way.
    "scale-integers": {
        "a": [1, 2, 3, 4, 5, 6, 7, 8],
        "b": [-4, -2, 0, 2, 4, 6, 8, 10],
    },
    # Nulls in both columns and in different rows, so the per-column non-null
    # count differs between them.
    "scale-nulls": {
        "a": [1.0, 2.0, None, 4.0, 5.0, None, 7.0, 8.0],
        "b": [-4.0, None, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0],
    },
    # The three ways _handle_zeros_in_scale fires — zero variance, zero range,
    # zero maximum magnitude — beside a column it leaves alone.
    "scale-constant": {
        "a": [3.0, 3.0, 3.0, 3.0],
        "b": [0.0, 0.0, 0.0, 0.0],
        "c": [1.0, 2.0, 3.0, 4.0],
    },
    # Both signs, and a column whose largest magnitude is its smallest value:
    # max|x| is not max(x).
    "scale-signed": {
        "a": [1.0, -8.0, 2.0, -0.5],
        "b": [-3.0, 3.0, 0.0, 1.5],
    },
    # Values outside every fitted range of "scale-numbers", which is the only
    # thing clip can reach; the same three columns, so a scaler fitted there
    # transforms this.
    "scale-outside": {
        "a": [-4.0, 0.5, 8.0, 12.0],
        "b": [-8.0, 0.0, 10.0, 24.0],
        "c": [-1.0, 0.0, 1.0, 2.0],
    },
    # Rows for a norm: a 3-4-5 triple whose l2 norm is exact, an all-zero row
    # (the one _handle_zeros_in_scale leaves alone), and two mixed-sign rows.
    "scale-vectors": {
        "x": [3.0, 0.0, 1.0, -2.0],
        "y": [4.0, 0.0, -1.0, 2.0],
        "z": [0.0, 0.0, 1.0, 1.0],
    },
    "scale-vectors-integers": {
        "x": [3, 0, 1, -2],
        "y": [4, 0, -1, 2],
        "z": [0, 0, 1, 1],
    },
    # Values sitting exactly on the thresholds the binarize cases use, which
    # is where "strictly greater" is the whole of the semantics.
    "scale-thresholds": {
        "a": [-1.0, -0.5, 0.0, 0.5, 1.0],
        "b": [0.0, 0.25, 0.5, 0.75, 1.0],
    },
    "scale-thresholds-integers": {
        "a": [-2, -1, 0, 1, 2],
        "b": [0, 1, 2, 3, 4],
    },
    # Spread out, no ties, no nulls: the ordinary case for every transformer.
    "bin-spread": {
        "x": [-4.0, -2.5, -1.0, 0.0, 0.5, 1.5, 2.0, 3.25, 4.0, 5.5, 6.0, 7.75, 8.0, 9.5, 11.0, 14.0],
        "y": [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5],
    },
    # Values well outside what "bin-spread" holds: the clipping at both ends.
    "bin-outliers": {
        "x": [-100.0, -4.0, 0.5, 14.0, 100.0],
        "y": [-50.0, 2.0, 5.0, 9.5, 50.0],
    },
    # Nulls, which the scalers ignore at fit and pass through at transform
    # (upstream's NaN). KBinsDiscretizer refuses them, so it never fits here.
    "bin-nulls": {
        "x": [1.0, None, 3.0, 4.0, None, 6.0, 7.0, 9.0],
        "y": [2.0, 4.0, None, 8.0, 10.0, 12.0, 14.0, 16.0],
    },
    # Long runs of one value: what QuantileTransformer's forward/backward
    # averaging is for, and what makes KBinsDiscretizer drop a bin.
    "bin-ties": {
        "x": [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 5.0, 5.0, 9.0],
        "y": [0.0, 0.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 8.0, 8.0],
    },
    # One column that never varies: a zero scale, and a single bin.
    "bin-constant": {
        "x": [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
        "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    },
    # Integer columns, which we widen to f64 exactly as upstream does.
    "bin-integers": {
        "x": [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3],
        "y": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160],
    },
    # Three groups with a wide gap between them, so Lloyd's algorithm reaches
    # the same clustering however the sums were ordered.
    "bin-clusters": {
        "x": [0.0, 0.1, 0.2, 0.3, 5.0, 5.1, 5.2, 5.3, 10.0, 10.1, 10.2, 10.3],
        "y": [1.0, 1.5, 2.0, 2.5, 20.0, 20.5, 21.0, 21.5, 40.0, 40.5, 41.0, 41.5],
    },
    # Far fewer distinct values than n_bins, so Lloyd's algorithm empties a
    # cluster on its first pass and has to relocate: a center that is still
    # empty afterwards is placed at the biggest cluster's slot (the unaveraged
    # sum of it when that cluster comes later), the relocated point keeps the
    # label the assignment gave it, and the fitted edges repeat. Without all
    # three of those, 'x' bins as [0,0,0,0,1,2,0,0] instead of [1,1,1,1,2,3,1,1].
    "bin-kmeans-empty": {
        "x": [0.0, 0.0, 0.0, 0.0, 1.0, 5.0, 0.0, 0.0],
        "y": [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 20.0],
    },
    # Two adjacent doubles at 1e16, 2.0 apart: linspace returns edges that
    # repeat, which is what upstream's uniform strategy stores too, so the
    # fitted table has to survive its own load_state.
    "bin-two-ulp": {
        "x": [1e16, 1.0000000000000002e16, 1e16, 1.0000000000000002e16],
        "y": [1.0, 2.0, 3.0, 4.0],
    },
    # A quantile grid spanning 315 orders of magnitude. It is the one shape
    # that tells a fused multiply-add apart from a separate multiply and add
    # in the interpolation: the backward pass of QuantileTransformer answers
    # 7.76e-17 fused and 0.0 split, which moves the rank from -3.88e-17 to
    # 4.5e-285. numpy's compiled np.interp fuses, and so do we.
    "bin-wide": {
        "x": [-9007199254740992.0, 0.1, 0.5, 1.0, 1e-300, 1e300],
        "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    },
    # Strictly positive, so the same frame fits Box-Cox as well as everything
    # else; float throughout, so nothing depends on a dtype.
    "expand": {
        "price": [1.2, 3.4, 0.7, 8.1, 2.5, 4.9, 6.3, 1.8],
        "height": [10.0, 12.5, 9.25, 15.0, 11.75, 13.5, 14.25, 10.5],
    },
    "expand-nulls": {
        "price": [1.2, 3.4, None, 8.1, 2.5, 4.9, 6.3, 1.8],
        "height": [10.0, 12.5, 9.25, None, 11.75, 13.5, 14.25, 10.5],
    },
    # The same columns, well outside the fitted range at both ends: what the
    # four extrapolating spline modes are read against.
    "expand-wide": {
        "price": [-4.0, 12.0, 0.5, 20.0, -1.0, 7.0, 3.0, 9.0],
        "height": [-2.0, 30.0, 5.0, 18.0, 1.0, 22.0, 11.0, 25.0],
    },
    "expand-three": {
        "price": [1.2, 3.4, 0.7, 8.1, 2.5, 4.9, 6.3, 1.8],
        "height": [10.0, 12.5, 9.25, 15.0, 11.75, 13.5, 14.25, 10.5],
        "weight": [2.0, -1.5, 0.25, 4.75, -3.0, 1.25, 0.5, -2.25],
    },
    # Integers, to exercise the other side of the numeric reader: scikit-learn
    # widens them to float64 and so do we.
    "expand-integers": {
        "price": [1, 3, 7, 2, 5, 4, 6, 8],
        "height": [10, 12, 9, 15, 11, 13, 14, 20],
    },
    # One column: with linear extrapolation below degree 2, scikit-learn 1.9.0
    # gets the second and later columns of a call wrong or raises IndexError,
    # so only a single-column frame is comparable.
    "expand-single": {"price": [1.2, 3.4, 0.7, 8.1, 2.5, 4.9, 6.3, 1.8]},
    "expand-single-wide": {"price": [-4.0, 12.0, 0.5, 20.0, -1.0, 7.0, 3.0, 9.0]},
    # Ties, so the quantile knots repeat and the recursion meets equal knots.
    "expand-ties": {
        "price": [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 5.0, 5.0],
        "height": [3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 9.0, 9.0],
    },
    # Negative and mixed-sign values: yeo-johnson's other two likelihood
    # branches, which Box-Cox refuses outright.
    "expand-signed": {
        "price": [-3.5, 1.2, 0.0, -0.75, 4.5, 2.25, -2.0, 6.0],
        "height": [-1.0, -2.5, -4.0, -0.5, -3.25, -1.75, -5.0, -2.0],
    },
    # A constant column beside a varying one: yeo-johnson leaves the constant
    # at lambda 1, the identity.
    "expand-constant": {
        "price": [2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5],
        "height": [10.0, 12.5, 9.25, 15.0, 11.75, 13.5, 14.25, 10.5],
    },
    # Each column sits tightly around one magnitude, so box-cox's likelihood
    # alone picks a lambda (about 33.7 for price, -35.5 for height) that
    # transforms its most extreme value past f64's largest. scipy caps the
    # lambda so that value lands on f64's largest over 10,000 instead.
    "expand-capped": {
        "price": [9.0e29, 9.6e29, 9.8e29, 9.9e29, 9.95e29, 1.0e30, 1.0e30, 1.0e30],
        "height": [1.0e-30, 1.0e-30, 1.0e-30, 1.005e-30, 1.01e-30, 1.02e-30, 1.04e-30, 1.1e-30],
    },
    # Near 1 the capped lambda passes 10,000 (14,555 for price, -13,844 for
    # height), so the power under the most extreme value overflows f64 and
    # scipy's boxcox goes through logarithms to stay finite.
    "expand-capped-near-one": {
        "price": [1.0499, 1.04994, 1.04996, 1.04998, 1.04999, 1.05, 1.05, 1.05],
        "height": [0.95, 0.95, 0.95, 0.950005, 0.95001, 0.95002, 0.95004, 0.9501],
    },
    # Past expand-capped's range: the power overflows for the first two values
    # of each column, while their quotient by lambda still fits.
    "expand-capped-wide": {
        "price": [2.2e30, 2.0e30, 1.5e30, 1.2e30, 1.0e30, 9.0e29, 5.0e29, 1.0e29],
        "height": [4.5e-31, 5.0e-31, 7.0e-31, 9.0e-31, 1.0e-30, 1.1e-30, 2.0e-30, 1.0e-29],
    },
}

# our transformer -> (upstream class, {our param: upstream param}); a
# parameter mapped to None has no upstream counterpart and is dropped.
# `columns` is always one of those: scikit-learn transforms whatever frame it
# is handed, so run_case narrows the frame instead.
UPSTREAM: dict[str, tuple[str, dict[str, str | None]]] = {
    "StandardScaler": ("StandardScaler", {"columns": None}),
    "MinMaxScaler": ("MinMaxScaler", {"columns": None}),
    "MaxAbsScaler": ("MaxAbsScaler", {"columns": None}),
    "Binarizer": ("Binarizer", {"columns": None}),
    "Normalizer": ("Normalizer", {"columns": None}),
    "RobustScaler": ("RobustScaler", {"columns": None}),
    "QuantileTransformer": ("QuantileTransformer", {"columns": None}),
    "KBinsDiscretizer": ("KBinsDiscretizer", {"columns": None}),
    "PolynomialFeatures": ("PolynomialFeatures", {"columns": None}),
    "SplineTransformer": ("SplineTransformer", {"columns": None}),
    "PowerTransformer": ("PowerTransformer", {"columns": None}),
}

# The inverse normal CDF: Acklam's rational approximation refined by two
# Halley steps agrees with scipy's Cephes ndtri to about one ulp, not to the
# bit, so the three cases that call it compare with a little slack.
NORMAL_TOLERANCE: float = 1e-12
# Lloyd's algorithm: our cluster sums are exact and order-independent while
# scikit-learn's accumulate in row order across OpenMP chunks, so a center can
# differ in its last bits even when the clustering is identical. What the
# tolerance cannot cover is a different clustering, which is why the
# empty-cluster rules are reproduced exactly rather than slackened.
KMEANS_TOLERANCE: float = 1e-9
# PowerTransformer needs by far the loosest tolerance of the four: its
# lambda maximizes a profile log-likelihood over a flat optimum, found by the
# same Brent iteration scipy runs, and the objective differs in its last bits
# because our sums are exact partials and numpy's are pairwise. On
# power/box-cox's 'height' column the two objectives differ by 2.7e-15 at the
# same lambda, and the maximum of ours sits 1.5e-7 away from the maximum of
# theirs; the solver finds each one's maximum, so the gap is the objective's,
# not the iteration's, and closing it would mean summing in numpy's row order
# — which would make the fit depend on the row-group split.
#
# This tolerance is derived from a measurement over these ten cases only.
# Against scikit-learn 1.9.0 the largest deviation is 1.71e-6 on a transformed
# value (power/box-cox-raw, height[3]) and 4.1e-7 on a lambda
# (power/yeo-johnson-nulls, height), so the tolerance below sits about six
# times above what was measured.
#
# It is not a bound on the transformer. Over about 1,200 fuzzed fits the
# deviation is unbounded both ways: 3.9e+133 absolute on a column of magnitude
# 1e133 (the relative deviation stays near 1e-4 there, so no absolute slack
# can cover it), and relative 2.0, with opposite signs, on a near-constant
# column, whose likelihood is flat enough that the two implementations pick
# lambdas 122 apart. The datasets above deliberately avoid both classes: every
# case carrying this tolerance is fitted on a well-conditioned column of order
# 1 to 15. The capped Box-Cox cases fit such columns on purpose and carry
# CAPPED_TOLERANCE instead.
POWER_TOLERANCE: float = 1e-5
# A capped Box-Cox lambda maximizes nothing: it is a closed form in the
# column's most extreme value, which ours computes with scipy's own Lambert W
# iteration. Its values reach 1e304 and more, though, where an absolute slack
# means nothing, and the libm calls under them may round differently on another
# platform: an ulp in the log moves the output by 1e-13 to 2e-13 relative,
# since the exponent is about 700, though an ulp in the exp costs only an ulp.
# From a power of 709.78 up, scipy's ARM64 build also fuses the exponent's
# multiply and subtract where ours rounds both. So the capped cases compare
# relative to the value, with this slack. It is not a measurement: on ARM64
# macOS, where the fixture was made, the largest deviation is 1.14e-13, at
# power/box-cox-capped-near-one height[5].
CAPPED_TOLERANCE: float = 1e-12
# The B-spline basis. Our Cox-de Boor recursion is scipy's `_deBoor_D` step for
# step, over a knot vector that matches scipy's to the bit, but scipy's binary
# is compiled with the default fp-contract, so its `h[n-1] += w*(xb - x)` is
# emitted as one fused multiply-add and rounds once where ours rounds twice.
# This build fixes `-ffp-contract=off` on each of our translation units (`make
# cpp-flags` is the gate), so a fitted artifact answers the same on every
# compiler and machine; matching scipy here would mean chasing a number that
# changes with how scipy itself was built, so the cases carry a slack instead.
# Degree 0 and 1 have no accumulation step at all and stay exact.
#
# This tolerance is derived from a measurement over these eleven cases only.
# The largest deviation is 5.7e-14 absolute, at spline/extrapolation-continue
# height_sp_1[0], where the value itself is about -300; the largest relative
# one is 3.7e-16 (three ulp) at spline/extrapolation-periodic price_sp_3[0].
# The tolerance below sits about nine times above the absolute measurement. It
# is not a bound on the transformer: off the fixture the basis has been
# measured at up to 16 ulp (degree 5 with extrapolation="linear").
SPLINE_TOLERANCE: float = 5e-13

CASES: list[dict[str, Any]] = [
    {
        "name": "standard/plain",
        "transformer": "StandardScaler",
        "params": {"columns": ["a", "b", "c"]},
        "fit": "scale-numbers",
    },
    {
        # columns=None scales every numeric column that reaches the step,
        # which is the whole frame scikit-learn scales.
        "name": "standard/every-column",
        "transformer": "StandardScaler",
        "params": {},
        "fit": "scale-numbers",
    },
    {
        # 'c' is not listed, so it passes through untouched and the library
        # never sees it.
        "name": "standard/one-column",
        "transformer": "StandardScaler",
        "params": {"columns": ["a"]},
        "fit": "scale-numbers",
    },
    {
        "name": "standard/nulls",
        "transformer": "StandardScaler",
        "params": {"columns": ["a", "b"]},
        "fit": "scale-nulls",
    },
    {
        "name": "standard/constant",
        "transformer": "StandardScaler",
        "params": {"columns": ["a", "b", "c"]},
        "fit": "scale-constant",
    },
    {
        "name": "standard/no-mean",
        "transformer": "StandardScaler",
        "params": {"columns": ["a", "b"], "with_mean": False},
        "fit": "scale-numbers",
    },
    {
        "name": "standard/no-std",
        "transformer": "StandardScaler",
        "params": {"columns": ["a", "b"], "with_std": False},
        "fit": "scale-numbers",
    },
    {
        "name": "standard/neither",
        "transformer": "StandardScaler",
        "params": {"columns": ["a", "b"], "with_mean": False, "with_std": False},
        "fit": "scale-numbers",
    },
    {
        "name": "standard/integers",
        "transformer": "StandardScaler",
        "params": {"columns": ["a", "b"]},
        "fit": "scale-integers",
    },
    {
        "name": "standard/outside-the-fitted-range",
        "transformer": "StandardScaler",
        "params": {"columns": ["a", "b", "c"]},
        "fit": "scale-numbers",
        "transform": "scale-outside",
    },
    {
        "name": "min-max/plain",
        "transformer": "MinMaxScaler",
        "params": {"columns": ["a", "b", "c"]},
        "fit": "scale-numbers",
    },
    {
        "name": "min-max/every-column",
        "transformer": "MinMaxScaler",
        "params": {},
        "fit": "scale-numbers",
    },
    {
        "name": "min-max/feature-range",
        "transformer": "MinMaxScaler",
        "params": {"columns": ["a", "b", "c"], "feature_range": (-1.0, 3.0)},
        "fit": "scale-numbers",
    },
    {
        "name": "min-max/nulls",
        "transformer": "MinMaxScaler",
        "params": {"columns": ["a", "b"]},
        "fit": "scale-nulls",
    },
    {
        "name": "min-max/constant",
        "transformer": "MinMaxScaler",
        "params": {"columns": ["a", "b", "c"], "feature_range": (2.0, 5.0)},
        "fit": "scale-constant",
    },
    {
        "name": "min-max/integers",
        "transformer": "MinMaxScaler",
        "params": {"columns": ["a", "b"]},
        "fit": "scale-integers",
    },
    {
        "name": "min-max/unclipped",
        "transformer": "MinMaxScaler",
        "params": {"columns": ["a", "b", "c"]},
        "fit": "scale-numbers",
        "transform": "scale-outside",
    },
    {
        "name": "min-max/clipped",
        "transformer": "MinMaxScaler",
        "params": {"columns": ["a", "b", "c"], "clip": True},
        "fit": "scale-numbers",
        "transform": "scale-outside",
    },
    {
        "name": "min-max/clipped-feature-range",
        "transformer": "MinMaxScaler",
        "params": {"columns": ["a", "b", "c"], "feature_range": (-1.0, 3.0), "clip": True},
        "fit": "scale-numbers",
        "transform": "scale-outside",
    },
    {
        "name": "max-abs/plain",
        "transformer": "MaxAbsScaler",
        "params": {"columns": ["a", "b", "c"]},
        "fit": "scale-numbers",
    },
    {
        "name": "max-abs/every-column",
        "transformer": "MaxAbsScaler",
        "params": {},
        "fit": "scale-signed",
    },
    {
        "name": "max-abs/signed",
        "transformer": "MaxAbsScaler",
        "params": {"columns": ["a", "b"]},
        "fit": "scale-signed",
    },
    {
        "name": "max-abs/zero-column",
        "transformer": "MaxAbsScaler",
        "params": {"columns": ["a", "b", "c"]},
        "fit": "scale-constant",
    },
    {
        "name": "max-abs/nulls",
        "transformer": "MaxAbsScaler",
        "params": {"columns": ["a", "b"]},
        "fit": "scale-nulls",
    },
    {
        "name": "max-abs/integers",
        "transformer": "MaxAbsScaler",
        "params": {"columns": ["a", "b"]},
        "fit": "scale-integers",
    },
    {
        "name": "max-abs/unclipped",
        "transformer": "MaxAbsScaler",
        "params": {"columns": ["a", "b", "c"]},
        "fit": "scale-numbers",
        "transform": "scale-outside",
    },
    {
        "name": "max-abs/clipped",
        "transformer": "MaxAbsScaler",
        "params": {"columns": ["a", "b", "c"], "clip": True},
        "fit": "scale-numbers",
        "transform": "scale-outside",
    },
    {
        "name": "binarize/default-threshold",
        "transformer": "Binarizer",
        "params": {"columns": ["a", "b"]},
        "fit": "scale-thresholds",
    },
    {
        # 0.5 is a value of both columns, so this is the case that pins
        # "strictly above" rather than "at least".
        "name": "binarize/on-the-threshold",
        "transformer": "Binarizer",
        "params": {"columns": ["a", "b"], "threshold": 0.5},
        "fit": "scale-thresholds",
    },
    {
        "name": "binarize/negative-threshold",
        "transformer": "Binarizer",
        "params": {"columns": ["a", "b"], "threshold": -0.5},
        "fit": "scale-thresholds",
    },
    {
        "name": "binarize/one-column",
        "transformer": "Binarizer",
        "params": {"columns": ["b"], "threshold": 0.25},
        "fit": "scale-thresholds",
    },
    {
        "name": "binarize/integers",
        "transformer": "Binarizer",
        "params": {"columns": ["a", "b"], "threshold": 1.0},
        "fit": "scale-thresholds-integers",
    },
    {
        "name": "normalize/l2",
        "transformer": "Normalizer",
        "params": {"columns": ["x", "y", "z"]},
        "fit": "scale-vectors",
    },
    {
        "name": "normalize/l1",
        "transformer": "Normalizer",
        "params": {"columns": ["x", "y", "z"], "norm": "l1"},
        "fit": "scale-vectors",
    },
    {
        "name": "normalize/max",
        "transformer": "Normalizer",
        "params": {"columns": ["x", "y", "z"], "norm": "max"},
        "fit": "scale-vectors",
    },
    {
        # Two of three columns: the norm is over the listed vector only, and
        # 'z' passes through untouched.
        "name": "normalize/subset",
        "transformer": "Normalizer",
        "params": {"columns": ["x", "y"]},
        "fit": "scale-vectors",
    },
    {
        "name": "normalize/integers",
        "transformer": "Normalizer",
        "params": {"columns": ["x", "y", "z"]},
        "fit": "scale-vectors-integers",
    },
    {
        "name": "normalize/integers-l1",
        "transformer": "Normalizer",
        "params": {"columns": ["x", "y", "z"], "norm": "l1"},
        "fit": "scale-vectors-integers",
    },
    {
        "name": "robust/plain",
        "transformer": "RobustScaler",
        "params": {"columns": ["x", "y"]},
        "fit": "bin-spread",
    },
    {
        "name": "robust/every-column",
        "transformer": "RobustScaler",
        "params": {},
        "fit": "bin-spread",
    },
    {
        "name": "robust/nulls",
        "transformer": "RobustScaler",
        "params": {"columns": ["x", "y"]},
        "fit": "bin-nulls",
    },
    {
        "name": "robust/no-centering",
        "transformer": "RobustScaler",
        "params": {"columns": ["x", "y"], "with_centering": False},
        "fit": "bin-spread",
    },
    {
        "name": "robust/no-scaling",
        "transformer": "RobustScaler",
        "params": {"columns": ["x", "y"], "with_scaling": False},
        "fit": "bin-spread",
    },
    {
        "name": "robust/quantile-range",
        "transformer": "RobustScaler",
        "params": {"columns": ["x", "y"], "quantile_range": [10.0, 90.0]},
        "fit": "bin-spread",
    },
    {
        "name": "robust/unit-variance",
        "transformer": "RobustScaler",
        "params": {"columns": ["x", "y"], "unit_variance": True},
        "fit": "bin-spread",
        "tolerance": NORMAL_TOLERANCE,
    },
    {
        "name": "robust/constant-column",
        "transformer": "RobustScaler",
        "params": {"columns": ["x", "y"]},
        "fit": "bin-constant",
    },
    {
        "name": "robust/integers",
        "transformer": "RobustScaler",
        "params": {"columns": ["x", "y"]},
        "fit": "bin-integers",
    },
    {
        "name": "robust/outliers",
        "transformer": "RobustScaler",
        "params": {"columns": ["x", "y"]},
        "fit": "bin-spread",
        "transform": "bin-outliers",
    },
    {
        "name": "quantile-transform/plain",
        "transformer": "QuantileTransformer",
        "params": {"columns": ["x", "y"]},
        "fit": "bin-spread",
    },
    {
        "name": "quantile-transform/every-column",
        "transformer": "QuantileTransformer",
        "params": {"n_quantiles": 5},
        "fit": "bin-spread",
    },
    {
        "name": "quantile-transform/small-grid",
        "transformer": "QuantileTransformer",
        "params": {"columns": ["x", "y"], "n_quantiles": 3},
        "fit": "bin-spread",
    },
    {
        "name": "quantile-transform/ties",
        "transformer": "QuantileTransformer",
        "params": {"columns": ["x", "y"], "n_quantiles": 6},
        "fit": "bin-ties",
    },
    {
        "name": "quantile-transform/nulls",
        "transformer": "QuantileTransformer",
        "params": {"columns": ["x", "y"], "n_quantiles": 4},
        "fit": "bin-nulls",
    },
    {
        "name": "quantile-transform/constant-column",
        "transformer": "QuantileTransformer",
        "params": {"columns": ["x", "y"], "n_quantiles": 4},
        "fit": "bin-constant",
    },
    {
        "name": "quantile-transform/outliers",
        "transformer": "QuantileTransformer",
        "params": {"columns": ["x", "y"], "n_quantiles": 8},
        "fit": "bin-spread",
        "transform": "bin-outliers",
    },
    {
        "name": "quantile-transform/integers",
        "transformer": "QuantileTransformer",
        "params": {"columns": ["x", "y"], "n_quantiles": 5},
        "fit": "bin-integers",
    },
    {
        "name": "quantile-transform/normal",
        "transformer": "QuantileTransformer",
        "params": {"columns": ["x", "y"], "n_quantiles": 8, "output_distribution": "normal"},
        "fit": "bin-spread",
        "tolerance": NORMAL_TOLERANCE,
    },
    {
        "name": "quantile-transform/normal-outliers",
        "transformer": "QuantileTransformer",
        "params": {"columns": ["x", "y"], "n_quantiles": 8, "output_distribution": "normal"},
        "fit": "bin-spread",
        "transform": "bin-outliers",
        "tolerance": NORMAL_TOLERANCE,
    },
    {
        "name": "kbins/uniform-ordinal",
        "transformer": "KBinsDiscretizer",
        "params": {"columns": ["x", "y"], "n_bins": 4, "encode": "ordinal", "strategy": "uniform"},
        "fit": "bin-spread",
    },
    {
        "name": "kbins/uniform-onehot",
        "transformer": "KBinsDiscretizer",
        "params": {"columns": ["x", "y"], "n_bins": 3, "encode": "onehot", "strategy": "uniform"},
        "fit": "bin-spread",
    },
    {
        "name": "kbins/uniform-onehot-dense",
        "transformer": "KBinsDiscretizer",
        "params": {"columns": ["x", "y"], "n_bins": 3, "encode": "onehot-dense", "strategy": "uniform"},
        "fit": "bin-spread",
    },
    {
        "name": "kbins/quantile-ordinal",
        "transformer": "KBinsDiscretizer",
        "params": {"columns": ["x", "y"], "n_bins": 4, "encode": "ordinal"},
        "fit": "bin-spread",
    },
    {
        "name": "kbins/quantile-linear",
        "transformer": "KBinsDiscretizer",
        "params": {"columns": ["x", "y"], "n_bins": 4, "encode": "ordinal", "quantile_method": "linear"},
        "fit": "bin-spread",
    },
    {
        "name": "kbins/quantile-hazen",
        "transformer": "KBinsDiscretizer",
        "params": {"columns": ["x", "y"], "n_bins": 5, "encode": "ordinal", "quantile_method": "hazen"},
        "fit": "bin-spread",
    },
    {
        "name": "kbins/quantile-inverted-cdf",
        "transformer": "KBinsDiscretizer",
        "params": {"columns": ["x", "y"], "n_bins": 5, "encode": "ordinal", "quantile_method": "inverted_cdf"},
        "fit": "bin-spread",
    },
    {
        "name": "kbins/quantile-onehot",
        "transformer": "KBinsDiscretizer",
        "params": {"columns": ["x", "y"], "n_bins": 3, "encode": "onehot"},
        "fit": "bin-spread",
    },
    # Runs of equal values collapse edges, so a column ends up with fewer
    # bins than n_bins asked for.
    {
        "name": "kbins/quantile-dropped-bins",
        "transformer": "KBinsDiscretizer",
        "params": {"columns": ["x", "y"], "n_bins": 5, "encode": "ordinal"},
        "fit": "bin-ties",
    },
    {
        "name": "kbins/quantile-dropped-bins-onehot",
        "transformer": "KBinsDiscretizer",
        "params": {"columns": ["x", "y"], "n_bins": 5, "encode": "onehot"},
        "fit": "bin-ties",
    },
    {
        "name": "kbins/constant-column",
        "transformer": "KBinsDiscretizer",
        "params": {"columns": ["x", "y"], "n_bins": 3, "encode": "onehot", "strategy": "uniform"},
        "fit": "bin-constant",
    },
    {
        "name": "kbins/integers",
        "transformer": "KBinsDiscretizer",
        "params": {"columns": ["x", "y"], "n_bins": 4, "encode": "ordinal", "strategy": "uniform"},
        "fit": "bin-integers",
    },
    {
        "name": "kbins/outliers",
        "transformer": "KBinsDiscretizer",
        "params": {"columns": ["x", "y"], "n_bins": 4, "encode": "ordinal", "strategy": "uniform"},
        "fit": "bin-spread",
        "transform": "bin-outliers",
    },
    {
        "name": "kbins/kmeans-ordinal",
        "transformer": "KBinsDiscretizer",
        "params": {"columns": ["x", "y"], "n_bins": 3, "encode": "ordinal", "strategy": "kmeans"},
        "fit": "bin-clusters",
        "tolerance": KMEANS_TOLERANCE,
    },
    {
        "name": "kbins/kmeans-onehot",
        "transformer": "KBinsDiscretizer",
        "params": {"columns": ["x", "y"], "n_bins": 3, "encode": "onehot", "strategy": "kmeans"},
        "fit": "bin-clusters",
        "tolerance": KMEANS_TOLERANCE,
    },
    {
        "name": "kbins/kmeans-spread",
        "transformer": "KBinsDiscretizer",
        "params": {"columns": ["x", "y"], "n_bins": 4, "encode": "ordinal", "strategy": "kmeans"},
        "fit": "bin-spread",
        "tolerance": KMEANS_TOLERANCE,
    },
    {
        # The empty-cluster path: every rule in _relocate_empty_clusters_dense
        # and _average_centers has to hold, or every row lands in a different
        # bin and the fitted bin count differs.
        "name": "kbins/kmeans-empty-cluster",
        "transformer": "KBinsDiscretizer",
        "params": {"columns": ["x", "y"], "n_bins": 5, "encode": "ordinal", "strategy": "kmeans"},
        "fit": "bin-kmeans-empty",
        "tolerance": KMEANS_TOLERANCE,
    },
    {
        "name": "kbins/kmeans-empty-cluster-onehot",
        "transformer": "KBinsDiscretizer",
        "params": {"columns": ["x", "y"], "n_bins": 5, "encode": "onehot", "strategy": "kmeans"},
        "fit": "bin-kmeans-empty",
        "tolerance": KMEANS_TOLERANCE,
    },
    {
        # linspace between two adjacent doubles repeats an edge; upstream
        # keeps those for strategy="uniform" and so do we, so the bins and the
        # edge count both have to match.
        "name": "kbins/uniform-two-ulp-span",
        "transformer": "KBinsDiscretizer",
        "params": {"columns": ["x", "y"], "n_bins": 5, "encode": "ordinal", "strategy": "uniform"},
        "fit": "bin-two-ulp",
    },
    {
        # The grid wide enough to tell the fused interpolation apart from the
        # split one; see the "bin-wide" dataset.
        "name": "quantile-transform/wide-grid",
        "transformer": "QuantileTransformer",
        "params": {"columns": ["x", "y"], "n_quantiles": 2},
        "fit": "bin-wide",
    },
    {
        "name": "polynomial/default",
        "transformer": "PolynomialFeatures",
        "params": {"columns": ["price", "height"]},
        "fit": "expand",
    },
    {
        "name": "polynomial/degree-3",
        "transformer": "PolynomialFeatures",
        "params": {"columns": ["price", "height"], "degree": 3},
        "fit": "expand",
    },
    {
        "name": "polynomial/three-columns",
        "transformer": "PolynomialFeatures",
        "params": {"columns": ["price", "height", "weight"], "degree": 3},
        "fit": "expand-three",
    },
    {
        "name": "polynomial/interaction-only",
        "transformer": "PolynomialFeatures",
        "params": {"columns": ["price", "height", "weight"], "degree": 3, "interaction_only": True},
        "fit": "expand-three",
    },
    {
        "name": "polynomial/no-bias",
        "transformer": "PolynomialFeatures",
        "params": {"columns": ["price", "height"], "degree": 2, "include_bias": False},
        "fit": "expand",
    },
    {
        "name": "polynomial/degree-range",
        "transformer": "PolynomialFeatures",
        "params": {"columns": ["price", "height", "weight"], "degree": [2, 3]},
        "fit": "expand-three",
    },
    {
        "name": "polynomial/degree-zero",
        "transformer": "PolynomialFeatures",
        "params": {"columns": ["price", "height"], "degree": 0},
        "fit": "expand",
    },
    {
        "name": "polynomial/integers",
        "transformer": "PolynomialFeatures",
        "params": {"columns": ["price", "height"], "degree": 2},
        "fit": "expand-integers",
    },
    {
        "name": "polynomial/unseen-rows",
        "transformer": "PolynomialFeatures",
        "params": {"columns": ["price", "height"], "degree": 2},
        "fit": "expand",
        "transform": "expand-wide",
    },
    {
        "name": "spline/uniform",
        "transformer": "SplineTransformer",
        "params": {"columns": ["price", "height"]},
        "fit": "expand",
        "tolerance": SPLINE_TOLERANCE,
    },
    {
        "name": "spline/quantile",
        "transformer": "SplineTransformer",
        "params": {"columns": ["price", "height"], "knots": "quantile"},
        "fit": "expand",
        "tolerance": SPLINE_TOLERANCE,
    },
    {
        "name": "spline/quantile-ties",
        "transformer": "SplineTransformer",
        "params": {"columns": ["price", "height"], "knots": "quantile", "n_knots": 6},
        "fit": "expand-ties",
        "tolerance": SPLINE_TOLERANCE,
    },
    {
        "name": "spline/no-bias",
        "transformer": "SplineTransformer",
        "params": {"columns": ["price", "height"], "include_bias": False},
        "fit": "expand",
        "tolerance": SPLINE_TOLERANCE,
    },
    {
        "name": "spline/degree-1",
        "transformer": "SplineTransformer",
        "params": {"columns": ["price", "height"], "degree": 1, "n_knots": 4},
        "fit": "expand",
    },
    {
        "name": "spline/degree-0",
        "transformer": "SplineTransformer",
        "params": {"columns": ["price", "height"], "degree": 0, "extrapolation": "continue"},
        "fit": "expand",
        "transform": "expand-wide",
    },
    {
        "name": "spline/extrapolation-error",
        "transformer": "SplineTransformer",
        "params": {"columns": ["price", "height"], "extrapolation": "error"},
        "fit": "expand",
        "tolerance": SPLINE_TOLERANCE,
    },
    {
        "name": "spline/extrapolation-constant",
        "transformer": "SplineTransformer",
        "params": {"columns": ["price", "height"], "extrapolation": "constant"},
        "fit": "expand",
        "transform": "expand-wide",
        "tolerance": SPLINE_TOLERANCE,
    },
    {
        # Linear extrapolation below degree 2 is a single-column case: there,
        # scikit-learn 1.9.0 gets the second and later columns of a call wrong
        # or raises IndexError.
        "name": "spline/extrapolation-linear",
        "transformer": "SplineTransformer",
        "params": {"columns": ["price", "height"], "extrapolation": "linear"},
        "fit": "expand",
        "transform": "expand-wide",
        "tolerance": SPLINE_TOLERANCE,
    },
    {
        "name": "spline/extrapolation-linear-degree-1",
        "transformer": "SplineTransformer",
        "params": {"columns": ["price"], "extrapolation": "linear", "degree": 1},
        "fit": "expand-single",
        "transform": "expand-single-wide",
    },
    {
        "name": "spline/extrapolation-continue",
        "transformer": "SplineTransformer",
        "params": {"columns": ["price", "height"], "extrapolation": "continue"},
        "fit": "expand",
        "transform": "expand-wide",
        "tolerance": SPLINE_TOLERANCE,
    },
    {
        "name": "spline/extrapolation-periodic",
        "transformer": "SplineTransformer",
        "params": {"columns": ["price", "height"], "extrapolation": "periodic"},
        "fit": "expand",
        "transform": "expand-wide",
        "tolerance": SPLINE_TOLERANCE,
    },
    {
        "name": "spline/missing-zeros",
        "transformer": "SplineTransformer",
        "params": {"columns": ["price", "height"], "handle_missing": "zeros"},
        "fit": "expand-nulls",
        "tolerance": SPLINE_TOLERANCE,
    },
    {
        "name": "spline/integers",
        "transformer": "SplineTransformer",
        "params": {"columns": ["price", "height"], "n_knots": 4},
        "fit": "expand-integers",
        "tolerance": SPLINE_TOLERANCE,
    },
    {
        "name": "power/yeo-johnson",
        "transformer": "PowerTransformer",
        "params": {"columns": ["price", "height"]},
        "fit": "expand",
        "tolerance": POWER_TOLERANCE,
    },
    {
        "name": "power/yeo-johnson-raw",
        "transformer": "PowerTransformer",
        "params": {"columns": ["price", "height"], "standardize": False},
        "fit": "expand",
        "tolerance": POWER_TOLERANCE,
    },
    {
        "name": "power/yeo-johnson-signed",
        "transformer": "PowerTransformer",
        "params": {"columns": ["price", "height"]},
        "fit": "expand-signed",
        "tolerance": POWER_TOLERANCE,
    },
    {
        "name": "power/yeo-johnson-constant",
        "transformer": "PowerTransformer",
        "params": {"columns": ["price", "height"]},
        "fit": "expand-constant",
        "tolerance": POWER_TOLERANCE,
    },
    {
        "name": "power/yeo-johnson-nulls",
        "transformer": "PowerTransformer",
        "params": {"columns": ["price", "height"]},
        "fit": "expand-nulls",
        "tolerance": POWER_TOLERANCE,
    },
    {
        "name": "power/box-cox",
        "transformer": "PowerTransformer",
        "params": {"columns": ["price", "height"], "method": "box-cox"},
        "fit": "expand",
        "tolerance": POWER_TOLERANCE,
    },
    {
        "name": "power/box-cox-raw",
        "transformer": "PowerTransformer",
        "params": {"columns": ["price", "height"], "method": "box-cox", "standardize": False},
        "fit": "expand",
        "tolerance": POWER_TOLERANCE,
    },
    {
        "name": "power/box-cox-nulls",
        "transformer": "PowerTransformer",
        "params": {"columns": ["price", "height"], "method": "box-cox"},
        "fit": "expand-nulls",
        "tolerance": POWER_TOLERANCE,
    },
    {
        "name": "power/integers",
        "transformer": "PowerTransformer",
        "params": {"columns": ["price", "height"], "method": "box-cox"},
        "fit": "expand-integers",
        "tolerance": POWER_TOLERANCE,
    },
    {
        "name": "power/unseen-rows",
        "transformer": "PowerTransformer",
        "params": {"columns": ["price", "height"]},
        "fit": "expand",
        "transform": "expand-wide",
        "tolerance": POWER_TOLERANCE,
    },
    {
        "name": "power/box-cox-capped",
        "transformer": "PowerTransformer",
        "params": {"columns": ["price", "height"], "method": "box-cox", "standardize": False},
        "fit": "expand-capped",
        "relative_tolerance": CAPPED_TOLERANCE,
    },
    {
        "name": "power/box-cox-capped-near-one",
        "transformer": "PowerTransformer",
        "params": {"columns": ["price", "height"], "method": "box-cox", "standardize": False},
        "fit": "expand-capped-near-one",
        "relative_tolerance": CAPPED_TOLERANCE,
    },
    {
        "name": "power/box-cox-capped-unseen",
        "transformer": "PowerTransformer",
        "params": {"columns": ["price", "height"], "method": "box-cox", "standardize": False},
        "fit": "expand-capped",
        "transform": "expand-capped-wide",
        "relative_tolerance": CAPPED_TOLERANCE,
    },
]


def our_name(column: str) -> str:
    """Return our spelling of one generated column name.

    scikit-learn writes a polynomial product as ``price height``; a space is a
    poor column name, so ours is ``price*height``, and a source column whose
    name holds ``*``, ``^`` or a space is refused. No other transformer here
    puts a space in a name, so the rename is unconditional.

    Args:
        column: The name scikit-learn generated.

    Returns:
        The name our kernel generates for the same column.

    """
    return column.replace(" ", "*")


def run_case(case: dict[str, Any]) -> dict[str, list[Any]]:
    """Fit and transform one case with the real library.

    Args:
        case: The case description.

    Returns:
        The transformed columns as plain lists, keyed by our column names.

    """
    import numpy as np
    import pandas as pd
    import sklearn.preprocessing

    upstream_class, renames = UPSTREAM[case["transformer"]]
    params = {
        renames.get(key, key): value for key, value in case["params"].items() if renames.get(key, key) is not None
    }
    if "quantile_range" in params:  # upstream insists on a tuple, and JSON has no tuples
        params["quantile_range"] = tuple(params["quantile_range"])
    transformer = getattr(sklearn.preprocessing, upstream_class)(**params)
    # scikit-learn transforms every column of the frame it is given, so the
    # frame is narrowed to the columns the case names; the rest are
    # pass-through columns of ours that the library never sees.
    columns = case["params"].get("columns") or list(DATASETS[case["fit"]])
    fit_frame = pd.DataFrame(DATASETS[case["fit"]])[columns]
    transform_frame = pd.DataFrame(DATASETS[case.get("transform", case["fit"])])[columns]
    transformer.fit(fit_frame)
    out = transformer.transform(transform_frame)
    if hasattr(out, "toarray"):  # encode="onehot" hands back a sparse matrix
        out = out.toarray()
    out = np.asarray(out)
    names = [our_name(str(name)) for name in transformer.get_feature_names_out()]
    if case["transformer"] == "KBinsDiscretizer" and case["params"].get("encode", "onehot") != "ordinal":
        # KBinsDiscretizer fits its internal OneHotEncoder on a float array, so
        # upstream names a generated column '<column>_0.0'; we spell the bin
        # index as an integer.
        names = [name.removesuffix(".0") for name in names]
    encoded: dict[str, list[Any]] = {}
    for index, name in enumerate(names):
        values = out[:, index].tolist()
        encoded[name] = [None if isinstance(v, float) and math.isnan(v) else v for v in values]
    return encoded


def main() -> None:
    """Write the fixture next to the tests."""
    cases = []
    for case in CASES:
        entry = dict(case)
        entry["expected"] = run_case(case)
        cases.append(entry)
    fixture = {"scikit_learn": VERSION, "datasets": DATASETS, "cases": cases}
    path = Path(__file__).resolve().parent.parent / "test" / "fixtures" / "sklearn_preprocessing.json"
    path.write_text(json.dumps(fixture, indent=2, sort_keys=False) + "\n")
    print(f"wrote {len(cases)} cases to {path}")


if __name__ == "__main__":
    main()
