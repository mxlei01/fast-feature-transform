"""Regenerate the category_encoders parity fixture.

Runs the real library once per case and records what it produced, so the
normal test run can compare against it without installing category_encoders,
pandas or statsmodels. Run it with the version pinned, as make parity-fixtures
does:

    uv run --isolated --no-project --python 3.14t \
        --with category_encoders==2.10.0 --with pandas \
        python scripts/regen_parity_fixtures.py

Cases name our transformer and our parameter spelling; UPSTREAM maps both to
category_encoders, dropping any parameter it maps to None. Datasets list
categories in sorted order wherever the encoder is sensitive to category
order, because we assign codes by sorted order and the library assigns them
by order of first appearance. GrayEncoder and RankHotEncoder are the
exceptions: the library sorts the categories for those two, so their cases may
fit on the deliberately unsorted "shuffled" dataset. A supervised case names
its target column with a top-level "target" key: we take the target from the
frame, the library takes it as a separate y. A case may carry a "tolerance",
an absolute difference test_parity allows instead of demanding the exact
value. Only the PolynomialEncoder cases and count/normalize-min-group-sum need
one: patsy factorizes the polynomial basis with LAPACK and we do not; upstream
sums a group's member ratios and we divide its count by the rows.

A case may instead carry "raises": True, meaning the library refuses that
transform. The fixture then records the message it raised rather than an
encoded frame, and test_parity demands that we refuse it too. Only
handle_missing="error" needs this: it is a transform-time check upstream, not
only a fit-time one.
"""

import json
import math
from pathlib import Path
from typing import Any

VERSION = "2.10.0"

DATASETS: dict[str, dict[str, list[Any]]] = {
    "plain": {
        "color": ["amber", "amber", "amber", "blue", "blue", "cyan", "delta", "delta"],
        "size": ["l", "m", "s", "l", "m", "s", "l", "m"],
    },
    "nulls": {
        "color": ["amber", "amber", None, "blue", None, "cyan"],
        "size": ["l", "m", "s", None, "l", "m"],
    },
    "rare": {
        "color": ["amber"] * 6 + ["blue", "cyan", "delta", "delta"],
        "size": ["l"] * 5 + ["m"] * 5,
    },
    # One small category and a null as rare as it: the null is the second
    # member a fold needs, unless combine_min_nan_groups keeps it apart.
    "rare-null": {
        "color": ["amber"] * 4 + ["blue"] * 3 + ["cyan", None],
        "size": ["l"] * 5 + ["m"] * 4,
    },
    # Two small categories and a null too common to join them but by force.
    "rare-nulls": {
        "color": ["amber"] * 4 + ["blue", "cyan"] + [None] * 3,
        "size": ["l"] * 5 + ["m"] * 4,
    },
    # Counts that sit on a fractional threshold under normalize: 7 / 100 is
    # not below 0.07, though 7 is below 0.07 * 100 = 7.000000000000001.
    "ratio-edge": {
        "color": ["amber"] * 86 + ["blue"] * 7 + ["cyan"] * 7,
        "size": ["l"] * 50 + ["m"] * 50,
    },
    "ratio-edge-null": {
        "color": ["amber"] * 86 + ["blue"] * 7 + [None] * 7,
        "size": ["l"] * 50 + ["m"] * 50,
    },
    # A group of a category and the null, whose normalized value upstream sums
    # from its members' ratios: 0.2 + 0.1 is 0.30000000000000004, 3 / 10 is not.
    "group-sum": {
        "color": ["amber"] * 7 + ["blue"] + [None] * 2,
        "size": ["l"] * 5 + ["m"] * 5,
    },
    "unseen": {
        "color": ["amber", "zulu", "blue", None],
        "size": ["l", "m", "s", "l"],
    },
    # Categories that do not first appear in sorted order: only GrayEncoder
    # and RankHotEncoder, which the library sorts for, can fit on this.
    "shuffled": {
        "color": ["echo", "cyan", "amber", "delta", "blue", "echo", "cyan"],
        "size": ["s", "m", "l", "s", "m", "l", "s"],
    },
    # One category, so every contrast coder emits no columns at all.
    "single": {
        "color": ["amber", "amber", "amber", "amber"],
        "size": ["l", "m", "s", "l"],
    },
    # No nulls anywhere: pandas stores a missing string as NaN, and the hashing
    # encoder hashes str(value), so it hashes "nan" rather than skipping it.
    "hashed": {
        "color": ["amber", "blue", "cyan", "delta", "echo", "amber", "zulu", ""],
        "size": ["l", "m", "s", "l", "m", "s", "l", "m"],
    },
    # The target-mean family needs a numeric target column in the frame.
    "blend": {
        "color": ["amber", "amber", "amber", "blue", "blue", "cyan", "delta", "delta"],
        "size": ["l", "m", "s", "l", "m", "s", "l", "m"],
        "y": [1, 0, 1, 1, 0, 0, 1, 1],
    },
    "blend-float": {
        "color": ["amber", "amber", "amber", "blue", "blue", "cyan", "delta", "delta"],
        "size": ["l", "m", "s", "l", "m", "s", "l", "m"],
        "y": [2.5, -1.25, 0.5, 3.0, 1.75, -0.5, 0.25, 4.0],
    },
    "blend-binary-float": {
        "color": ["amber", "amber", "amber", "blue", "blue", "cyan", "delta", "delta"],
        "size": ["l", "m", "s", "l", "m", "s", "l", "m"],
        "y": [1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
    },
    "blend-nulls": {
        "color": ["amber", "amber", None, "blue", None, "cyan", "blue", None],
        "size": ["l", "m", "s", "l", "m", "s", "l", "m"],
        "y": [1, 0, 1, 1, 0, 0, 1, 1],
    },
    "blend-unseen": {
        "color": ["amber", "zulu", "blue", None],
        "size": ["l", "m", "s", "l"],
        "y": [1, 0, 1, 0],
    },
    # Every row its own category: m_estimate's id-like guard.
    "blend-ids": {
        "color": ["amber", "blue", "cyan", "delta"],
        "size": ["l", "m", "s", "l"],
        "y": [1, 0, 1, 1],
    },
    # Supervised cases name a target column; run_case hands it to the library
    # as y and drops it from X, while our engine reads it from the frame.
    "labelled": {
        "color": ["amber", "amber", "blue", "blue", "blue", "cyan", "delta", "delta"],
        "size": ["l", "m", "s", "l", "m", "s", "l", "m"],
        "y": [1.0, 3.0, 2.0, 8.0, 4.0, 5.0, 0.0, 6.0],
    },
    "labelled-nulls": {
        "color": ["amber", "amber", None, "blue", "blue", None, "cyan", "delta"],
        "size": ["l", "m", "s", "l", "m", "s", "l", "m"],
        "y": [1.0, 3.0, 2.0, 8.0, 4.0, 5.0, 0.0, 6.0],
    },
    # An integer target, to exercise the other side of the target reader.
    "labelled-binary": {
        "color": ["amber", "amber", "blue", "blue", "blue", "cyan", "delta", "delta"],
        "size": ["l", "m", "s", "l", "m", "s", "l", "m"],
        "y": [1, 0, 1, 1, 0, 1, 0, 0],
    },
    "labelled-unseen": {
        "color": ["amber", "zulu", "blue", None],
        "size": ["l", "m", "s", "l"],
        "y": [1.0, 2.0, 3.0, 4.0],
    },
    # The supervised datasets carry the target as an ordinary column.
    "target_plain": {
        "color": ["amber", "amber", "blue", "blue", "blue", "cyan", "delta", "delta"],
        "size": ["l", "m", "s", "l", "m", "s", "l", "m"],
        "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    },
    # Group sizes 3, 4, 1, 2: both an odd and an even count to interpolate over.
    "target_spread": {
        "color": ["amber"] * 3 + ["blue"] * 4 + ["cyan"] + ["delta"] * 2,
        "size": ["l", "m", "s", "l", "m", "s", "l", "m", "s", "l"],
        "y": [2.0, 9.0, 4.0, 1.0, 8.0, 3.0, 6.0, 5.0, 7.0, 0.5],
    },
    "target_nulls": {
        "color": ["amber", "amber", None, "blue", None, "cyan"],
        "size": ["l", "m", "s", "l", "m", "s"],
        "y": [1.5, 2.5, 3.5, 4.5, 5.5, 6.5],
    },
    "target_unseen": {
        "color": ["amber", "zulu", "blue", None],
        "size": ["l", "m", "s", "l"],
        "y": [1.0, 2.0, 3.0, 4.0],
    },
    "target_integer": {
        "color": ["amber", "amber", "blue", "blue", "blue", "cyan"],
        "size": ["l", "m", "s", "l", "m", "s"],
        "y": [10, 20, 30, 40, 50, 60],
    },
    # Two columns whose generated one-hot names collide: 'a' holding 'b_c'
    # and 'a_b' holding 'c' both want 'a_b_c'. The library disambiguates with
    # '#' across every encoded column, not within one.
    "colliding-names": {
        "a": ["b_c", "d", "b_c"],
        "a_b": ["c", "e", "c"],
    },
    # The same collision, on a frame whose column order is not its
    # alphabetical order: 'z_b' comes first and holds 'c', 'z' comes second
    # and holds 'b_c', so which of the two owns 'z_b_c' and which is pushed to
    # 'z_b_c#' is decided by the order the encoder walks its columns in.
    # Fitted with cols=None, that order is the frame's; reconstructing it
    # alphabetically from a saved artifact would silently swap the two values,
    # which the round trip in test_parity catches.
    # Each column's first-appearance order is its sorted order, as everywhere
    # else here, so the only thing under test is the column order. The two
    # 'z_b_c' columns hold different values, so swapping them is visible.
    "colliding-names-unsorted": {
        "z_b": ["c", "c", "q", "q"],
        "z": ["b_c", "w", "b_c", "w"],
    },
    # Nulls that were not there at fit: what handle_missing="error" refuses at
    # transform. The categories are the ones "plain" and "labelled" fit on.
    "late-nulls": {
        "color": ["amber", None, "blue"],
        "size": ["l", "m", "s"],
    },
    "labelled-late-nulls": {
        "color": ["amber", None, "blue"],
        "size": ["l", "m", "s"],
        "y": [1.0, 2.0, 3.0],
    },
}

# our transformer -> (upstream class, {our param: upstream param}); a
# parameter mapped to None has no upstream counterpart and is dropped.
UPSTREAM: dict[str, tuple[str, dict[str, str | None]]] = {
    "CountEncoder": ("CountEncoder", {"columns": "cols"}),
    "HashingEncoder": ("HashingEncoder", {"columns": "cols"}),
    "OrdinalEncoder": ("OrdinalEncoder", {"columns": "cols"}),
    "OneHotEncoder": ("OneHotEncoder", {"columns": "cols"}),
    "RankHotEncoder": ("RankHotEncoder", {"columns": "cols"}),
    "TargetEncoder": ("TargetEncoder", {"columns": "cols", "target": None}),
    "MEstimateEncoder": ("MEstimateEncoder", {"columns": "cols", "target": None}),
    "WOEEncoder": ("WOEEncoder", {"columns": "cols", "target": None}),
    "JamesSteinEncoder": ("JamesSteinEncoder", {"columns": "cols"}),
    "LeaveOneOutEncoder": ("LeaveOneOutEncoder", {"columns": "cols"}),
    "CatBoostEncoder": ("CatBoostEncoder", {"columns": "cols"}),
    "QuantileEncoder": ("QuantileEncoder", {"columns": "cols"}),
    "SummaryEncoder": ("SummaryEncoder", {"columns": "cols"}),
    "BaseNEncoder": ("BaseNEncoder", {"columns": "cols"}),
    "BinaryEncoder": ("BinaryEncoder", {"columns": "cols"}),
    "GrayEncoder": ("GrayEncoder", {"columns": "cols"}),
    "BackwardDifferenceEncoder": ("BackwardDifferenceEncoder", {"columns": "cols"}),
    "HelmertEncoder": ("HelmertEncoder", {"columns": "cols"}),
    "SumEncoder": ("SumEncoder", {"columns": "cols"}),
    "PolynomialEncoder": ("PolynomialEncoder", {"columns": "cols"}),
}

# Our PolynomialEncoder uses Arnoldi iteration with reorthogonalization; patsy
# factorizes a Vandermonde matrix with the local LAPACK's QR. The small cases
# here agree within this tolerance, but upstream's rounding error grows with
# the degree, so it is not an agreement bound for every cardinality: at 32
# categories the two differ by about 1e-5, and by more than 0.5 from 40 to 128.
POLYNOMIAL_TOLERANCE: float = 1e-12

CASES: list[dict[str, Any]] = [
    {"name": "count/plain", "encoder": "CountEncoder", "params": {"columns": ["color"]}, "fit": "plain"},
    {
        "name": "count/two-columns",
        "encoder": "CountEncoder",
        "params": {"columns": ["color", "size"]},
        "fit": "plain",
    },
    {
        "name": "count/normalize",
        "encoder": "CountEncoder",
        "params": {"columns": ["color"], "normalize": True},
        "fit": "plain",
    },
    {"name": "count/nulls", "encoder": "CountEncoder", "params": {"columns": ["color"]}, "fit": "nulls"},
    {
        "name": "count/nulls-return-nan",
        "encoder": "CountEncoder",
        "params": {"columns": ["color"], "handle_missing": "return_nan"},
        "fit": "nulls",
    },
    {
        "name": "count/unseen",
        "encoder": "CountEncoder",
        "params": {"columns": ["color"]},
        "fit": "plain",
        "transform": "unseen",
    },
    {
        "name": "count/unseen-fill",
        "encoder": "CountEncoder",
        "params": {"columns": ["color"], "handle_unknown": -1},
        "fit": "plain",
        "transform": "unseen",
    },
    {
        "name": "count/unseen-return-nan",
        "encoder": "CountEncoder",
        "params": {"columns": ["color"], "handle_unknown": "return_nan"},
        "fit": "plain",
        "transform": "unseen",
    },
    {
        # normalize divides a count by the fit rows, but an integer
        # handle_unknown is the literal number asked for and is not divided.
        "name": "count/normalize-unseen-fill",
        "encoder": "CountEncoder",
        "params": {"columns": ["color"], "normalize": True, "handle_unknown": 7},
        "fit": "plain",
        "transform": "unseen",
    },
    {
        "name": "count/missing-error-at-transform",
        "encoder": "CountEncoder",
        "params": {"columns": ["color"], "handle_missing": "error"},
        "fit": "plain",
        "transform": "late-nulls",
        "raises": True,
    },
    {
        "name": "count/min-group-size",
        "encoder": "CountEncoder",
        "params": {"columns": ["color"], "min_group_size": 3},
        "fit": "rare",
    },
    {
        "name": "count/min-group-name",
        "encoder": "CountEncoder",
        "params": {"columns": ["color"], "min_group_size": 3, "min_group_name": "other"},
        "fit": "rare",
    },
    {
        "name": "count/min-group-null",
        "encoder": "CountEncoder",
        "params": {"columns": ["color"], "min_group_size": 2},
        "fit": "rare-null",
    },
    {
        "name": "count/min-group-null-apart",
        "encoder": "CountEncoder",
        "params": {"columns": ["color"], "min_group_size": 2, "combine_min_nan_groups": False},
        "fit": "rare-null",
    },
    {
        "name": "count/min-group-null-forced",
        "encoder": "CountEncoder",
        "params": {"columns": ["color"], "min_group_size": 2, "combine_min_nan_groups": "force"},
        "fit": "rare-null",
    },
    {
        "name": "count/min-group-common-null",
        "encoder": "CountEncoder",
        "params": {"columns": ["color"], "min_group_size": 2},
        "fit": "rare-nulls",
    },
    {
        "name": "count/min-group-common-null-apart",
        "encoder": "CountEncoder",
        "params": {"columns": ["color"], "min_group_size": 2, "combine_min_nan_groups": False},
        "fit": "rare-nulls",
    },
    {
        "name": "count/min-group-common-null-forced",
        "encoder": "CountEncoder",
        "params": {"columns": ["color"], "min_group_size": 2, "combine_min_nan_groups": "force"},
        "fit": "rare-nulls",
    },
    {
        "name": "count/normalize-min-group-ratio",
        "encoder": "CountEncoder",
        "params": {"columns": ["color"], "normalize": True, "min_group_size": 0.07},
        "fit": "ratio-edge",
    },
    {
        "name": "count/normalize-min-group-ratio-null",
        "encoder": "CountEncoder",
        "params": {"columns": ["color"], "normalize": True, "min_group_size": 0.07},
        "fit": "ratio-edge-null",
    },
    {
        # A group's value is its integer total over the rows here, and the
        # float sum of its members' ratios upstream, in an order an
        # order-independent fit does not keep: they can differ in the last bit.
        "name": "count/normalize-min-group-sum",
        "encoder": "CountEncoder",
        "params": {"columns": ["color"], "normalize": True, "min_group_size": 3},
        "fit": "group-sum",
        "tolerance": 1e-15,
    },
    {"name": "ordinal/plain", "encoder": "OrdinalEncoder", "params": {"columns": ["color"]}, "fit": "plain"},
    {"name": "ordinal/nulls", "encoder": "OrdinalEncoder", "params": {"columns": ["color"]}, "fit": "nulls"},
    {
        "name": "ordinal/index-start-0",
        "encoder": "OrdinalEncoder",
        "params": {"columns": ["color"], "index_start": 0},
        "fit": "plain",
    },
    {
        "name": "ordinal/unseen",
        "encoder": "OrdinalEncoder",
        "params": {"columns": ["color"]},
        "fit": "plain",
        "transform": "unseen",
    },
    {
        "name": "ordinal/unseen-return-nan",
        "encoder": "OrdinalEncoder",
        "params": {"columns": ["color"], "handle_unknown": "return_nan"},
        "fit": "plain",
        "transform": "unseen",
    },
    {
        "name": "ordinal/missing-error-at-transform",
        "encoder": "OrdinalEncoder",
        "params": {"columns": ["color"], "handle_missing": "error"},
        "fit": "plain",
        "transform": "late-nulls",
        "raises": True,
    },
    {"name": "one-hot/plain", "encoder": "OneHotEncoder", "params": {"columns": ["color"]}, "fit": "plain"},
    {
        # The '#' that disambiguates a repeated name spans every encoded
        # column: 'a' holding 'b_c' and 'a_b' holding 'c' both want 'a_b_c'.
        "name": "one-hot/colliding-names",
        "encoder": "OneHotEncoder",
        "params": {"columns": ["a", "a_b"], "use_cat_names": True},
        "fit": "colliding-names",
    },
    {
        # cols=None: the encoder resolves its own column list, in frame order,
        # and the '#' is assigned walking it. The fitted order therefore has
        # to survive an artifact and a pickle unchanged.
        "name": "one-hot/colliding-names-derived",
        "encoder": "OneHotEncoder",
        "params": {"columns": None, "use_cat_names": True},
        "fit": "colliding-names-unsorted",
    },
    {
        "name": "one-hot/cat-names",
        "encoder": "OneHotEncoder",
        "params": {"columns": ["color"], "use_cat_names": True},
        "fit": "plain",
    },
    {"name": "one-hot/nulls", "encoder": "OneHotEncoder", "params": {"columns": ["color"]}, "fit": "nulls"},
    {
        "name": "one-hot/unseen",
        "encoder": "OneHotEncoder",
        "params": {"columns": ["color"]},
        "fit": "plain",
        "transform": "unseen",
    },
    {
        "name": "one-hot/indicator",
        "encoder": "OneHotEncoder",
        "params": {"columns": ["color"], "handle_unknown": "indicator"},
        "fit": "plain",
        "transform": "unseen",
    },
    {
        # The pair of cases that separate one_hot from rank_hot: one_hot has a
        # mapping row for a null it never fitted, so handle_unknown never sees
        # one, and it drops the pre-encoder's placeholder column outright.
        "name": "one-hot/unseen-return-nan",
        "encoder": "OneHotEncoder",
        "params": {"columns": ["color"], "handle_unknown": "return_nan"},
        "fit": "plain",
        "transform": "unseen",
    },
    {
        "name": "one-hot/missing-return-nan",
        "encoder": "OneHotEncoder",
        "params": {"columns": ["color"], "handle_missing": "return_nan"},
        "fit": "plain",
        "transform": "unseen",
    },
    {
        "name": "one-hot/nulls-return-nan",
        "encoder": "OneHotEncoder",
        "params": {"columns": ["color"], "handle_missing": "return_nan"},
        "fit": "nulls",
    },
    {"name": "rank-hot/plain", "encoder": "RankHotEncoder", "params": {"columns": ["color"]}, "fit": "plain"},
    {
        "name": "rank-hot/cat-names",
        "encoder": "RankHotEncoder",
        "params": {"columns": ["color"], "use_cat_names": True},
        "fit": "plain",
    },
    {
        "name": "rank-hot/unseen",
        "encoder": "RankHotEncoder",
        "params": {"columns": ["color"]},
        "fit": "plain",
        "transform": "unseen",
    },
    {"name": "rank-hot/nulls", "encoder": "RankHotEncoder", "params": {"columns": ["color"]}, "fit": "nulls"},
    {
        # A null that was never a fitted category takes the same
        # default row an unseen category takes, so this one is all NaN.
        "name": "rank-hot/unseen-return-nan",
        "encoder": "RankHotEncoder",
        "params": {"columns": ["color"], "handle_unknown": "return_nan"},
        "fit": "plain",
        "transform": "unseen",
    },
    {
        # The pre-encoder's slot for the missing value survives
        # 'return_nan' as a level of its own, so a trailing 'color_-2' column
        # comes out that nothing ever lights up.
        "name": "rank-hot/missing-return-nan",
        "encoder": "RankHotEncoder",
        "params": {"columns": ["color"], "handle_missing": "return_nan"},
        "fit": "plain",
        "transform": "unseen",
    },
    {
        "name": "rank-hot/nulls-return-nan",
        "encoder": "RankHotEncoder",
        "params": {"columns": ["color"], "handle_missing": "return_nan"},
        "fit": "nulls",
    },
    {
        "name": "rank-hot/cat-names-return-nan",
        "encoder": "RankHotEncoder",
        "params": {"columns": ["color"], "handle_missing": "return_nan", "use_cat_names": True},
        "fit": "nulls",
    },
    {
        # 'error' re-fits the pre-encoder strictly, so there is no slot to keep
        # and the width matches the default.
        "name": "rank-hot/missing-error",
        "encoder": "RankHotEncoder",
        "params": {"columns": ["color"], "handle_missing": "error"},
        "fit": "plain",
    },
    {
        "name": "backward-difference/plain",
        "encoder": "BackwardDifferenceEncoder",
        "params": {"columns": ["color"]},
        "fit": "plain",
    },
    {
        "name": "backward-difference/nulls",
        "encoder": "BackwardDifferenceEncoder",
        "params": {"columns": ["color"]},
        "fit": "nulls",
    },
    {
        "name": "backward-difference/unseen",
        "encoder": "BackwardDifferenceEncoder",
        "params": {"columns": ["color"]},
        "fit": "plain",
        "transform": "unseen",
    },
    {
        # The contrast coders fit their pre-encoder with
        # handle_missing='value' whatever their own option says, so its slot
        # for the missing value is a level of its own under 'return_nan' and
        # 'error' alike — one more output column, and a different matrix.
        "name": "backward-difference/nulls-return-nan",
        "encoder": "BackwardDifferenceEncoder",
        "params": {"columns": ["color"], "handle_missing": "return_nan"},
        "fit": "nulls",
    },
    {
        "name": "backward-difference/missing-error",
        "encoder": "BackwardDifferenceEncoder",
        "params": {"columns": ["color"], "handle_missing": "error"},
        "fit": "plain",
    },
    {"name": "helmert/plain", "encoder": "HelmertEncoder", "params": {"columns": ["color"]}, "fit": "plain"},
    {
        "name": "helmert/two-columns",
        "encoder": "HelmertEncoder",
        "params": {"columns": ["color", "size"]},
        "fit": "plain",
    },
    {
        "name": "helmert/unseen",
        "encoder": "HelmertEncoder",
        "params": {"columns": ["color"]},
        "fit": "plain",
        "transform": "unseen",
    },
    {
        "name": "helmert/indicator",
        "encoder": "HelmertEncoder",
        "params": {"columns": ["color"], "handle_unknown": "indicator"},
        "fit": "plain",
        "transform": "unseen",
    },
    {"name": "sum/plain", "encoder": "SumEncoder", "params": {"columns": ["color"]}, "fit": "plain"},
    {"name": "sum/nulls", "encoder": "SumEncoder", "params": {"columns": ["color"]}, "fit": "nulls"},
    {
        "name": "sum/unseen-return-nan",
        "encoder": "SumEncoder",
        "params": {"columns": ["color"], "handle_unknown": "return_nan"},
        "fit": "plain",
        "transform": "unseen",
    },
    {
        "name": "sum/missing-return-nan",
        "encoder": "SumEncoder",
        "params": {"columns": ["color"], "handle_missing": "return_nan"},
        "fit": "plain",
        "transform": "unseen",
    },
    {
        # The indicator level still sorts after the placeholder one.
        "name": "sum/indicator-return-nan",
        "encoder": "SumEncoder",
        "params": {"columns": ["color"], "handle_missing": "return_nan", "handle_unknown": "indicator"},
        "fit": "plain",
        "transform": "unseen",
    },
    {"name": "sum/single-level", "encoder": "SumEncoder", "params": {"columns": ["color"]}, "fit": "single"},
    {
        # One category has nothing to contrast against and emits nothing — but
        # the placeholder level is a second one, so this emits a column.
        "name": "sum/single-level-return-nan",
        "encoder": "SumEncoder",
        "params": {"columns": ["color"], "handle_missing": "return_nan"},
        "fit": "single",
    },
    {
        "name": "helmert/missing-error",
        "encoder": "HelmertEncoder",
        "params": {"columns": ["color"], "handle_missing": "error"},
        "fit": "plain",
    },
    {
        "name": "polynomial/plain",
        "encoder": "PolynomialEncoder",
        "params": {"columns": ["color"]},
        "fit": "plain",
        "tolerance": POLYNOMIAL_TOLERANCE,
    },
    {
        "name": "polynomial/nulls",
        "encoder": "PolynomialEncoder",
        "params": {"columns": ["color"]},
        "fit": "nulls",
        "tolerance": POLYNOMIAL_TOLERANCE,
    },
    {
        "name": "polynomial/two-columns",
        "encoder": "PolynomialEncoder",
        "params": {"columns": ["color", "size"]},
        "fit": "plain",
        "tolerance": POLYNOMIAL_TOLERANCE,
    },
    {
        "name": "polynomial/indicator",
        "encoder": "PolynomialEncoder",
        "params": {"columns": ["color"], "handle_unknown": "indicator"},
        "fit": "plain",
        "transform": "unseen",
        "tolerance": POLYNOMIAL_TOLERANCE,
    },
    {
        "name": "polynomial/missing-return-nan",
        "encoder": "PolynomialEncoder",
        "params": {"columns": ["color"], "handle_missing": "return_nan"},
        "fit": "plain",
        "transform": "unseen",
        "tolerance": POLYNOMIAL_TOLERANCE,
    },
    {"name": "hash/plain", "encoder": "HashingEncoder", "params": {"columns": ["color"]}, "fit": "plain"},
    {
        "name": "hash/two-columns",
        "encoder": "HashingEncoder",
        "params": {"columns": ["color", "size"], "n_components": 4},
        "fit": "plain",
    },
    {
        "name": "hash/wide",
        "encoder": "HashingEncoder",
        "params": {"columns": ["color"], "n_components": 16},
        "fit": "hashed",
    },
    {
        "name": "hash/single-bucket",
        "encoder": "HashingEncoder",
        "params": {"columns": ["color"], "n_components": 1},
        "fit": "hashed",
    },
    {
        "name": "hash/unseen",
        "encoder": "HashingEncoder",
        "params": {"columns": ["color"], "n_components": 6},
        "fit": "plain",
        "transform": "hashed",
    },
    # --- the target-mean family: the target is a column of the frame ---
    {
        "name": "target/default",
        "encoder": "TargetEncoder",
        "params": {"columns": ["color"], "target": "y"},
        "fit": "blend",
        "target": "y",
    },
    {
        "name": "target/tuned",
        "encoder": "TargetEncoder",
        "params": {"columns": ["color"], "target": "y", "min_samples_leaf": 2, "smoothing": 1.5},
        "fit": "blend",
        "target": "y",
    },
    {
        "name": "target/two-columns",
        "encoder": "TargetEncoder",
        "params": {"columns": ["color", "size"], "target": "y", "min_samples_leaf": 2, "smoothing": 1.0},
        "fit": "blend",
        "target": "y",
    },
    {
        "name": "target/float-target",
        "encoder": "TargetEncoder",
        "params": {"columns": ["color"], "target": "y", "min_samples_leaf": 2, "smoothing": 1.0},
        "fit": "blend-float",
        "target": "y",
    },
    {
        "name": "target/nulls",
        "encoder": "TargetEncoder",
        "params": {"columns": ["color"], "target": "y", "min_samples_leaf": 2, "smoothing": 1.0},
        "fit": "blend-nulls",
        "target": "y",
    },
    {
        "name": "target/nulls-return-nan",
        "encoder": "TargetEncoder",
        "params": {"columns": ["color"], "target": "y", "handle_missing": "return_nan"},
        "fit": "blend-nulls",
        "target": "y",
    },
    {
        "name": "target/unseen",
        "encoder": "TargetEncoder",
        "params": {"columns": ["color"], "target": "y", "min_samples_leaf": 2, "smoothing": 1.0},
        "fit": "blend",
        "transform": "blend-unseen",
        "target": "y",
    },
    {
        "name": "target/unseen-return-nan",
        "encoder": "TargetEncoder",
        "params": {"columns": ["color"], "target": "y", "handle_unknown": "return_nan"},
        "fit": "blend",
        "transform": "blend-unseen",
        "target": "y",
    },
    {
        "name": "m-estimate/default",
        "encoder": "MEstimateEncoder",
        "params": {"columns": ["color"], "target": "y"},
        "fit": "blend",
        "target": "y",
    },
    {
        "name": "m-estimate/m-large",
        "encoder": "MEstimateEncoder",
        "params": {"columns": ["color", "size"], "target": "y", "m": 12.5},
        "fit": "blend",
        "target": "y",
    },
    {
        "name": "m-estimate/m-zero",
        "encoder": "MEstimateEncoder",
        "params": {"columns": ["color"], "target": "y", "m": 0.0},
        "fit": "blend",
        "target": "y",
    },
    {
        "name": "m-estimate/float-target",
        "encoder": "MEstimateEncoder",
        "params": {"columns": ["color"], "target": "y", "m": 3.0},
        "fit": "blend-float",
        "target": "y",
    },
    {
        "name": "m-estimate/nulls",
        "encoder": "MEstimateEncoder",
        "params": {"columns": ["color"], "target": "y"},
        "fit": "blend-nulls",
        "target": "y",
    },
    {
        "name": "m-estimate/unseen",
        "encoder": "MEstimateEncoder",
        "params": {"columns": ["color"], "target": "y"},
        "fit": "blend",
        "transform": "blend-unseen",
        "target": "y",
    },
    {
        "name": "m-estimate/id-like",
        "encoder": "MEstimateEncoder",
        "params": {"columns": ["color"], "target": "y"},
        "fit": "blend-ids",
        "target": "y",
    },
    {
        "name": "woe/default",
        "encoder": "WOEEncoder",
        "params": {"columns": ["color"], "target": "y"},
        "fit": "blend",
        "target": "y",
    },
    {
        "name": "woe/regularization",
        "encoder": "WOEEncoder",
        "params": {"columns": ["color", "size"], "target": "y", "regularization": 0.25},
        "fit": "blend",
        "target": "y",
    },
    {
        "name": "woe/float-target",
        "encoder": "WOEEncoder",
        "params": {"columns": ["color"], "target": "y"},
        "fit": "blend-binary-float",
        "target": "y",
    },
    {
        "name": "woe/nulls",
        "encoder": "WOEEncoder",
        "params": {"columns": ["color"], "target": "y"},
        "fit": "blend-nulls",
        "target": "y",
    },
    {
        "name": "woe/nulls-return-nan",
        "encoder": "WOEEncoder",
        "params": {"columns": ["color"], "target": "y", "handle_missing": "return_nan"},
        "fit": "blend-nulls",
        "target": "y",
    },
    {
        "name": "woe/unseen",
        "encoder": "WOEEncoder",
        "params": {"columns": ["color"], "target": "y"},
        "fit": "blend",
        "transform": "blend-unseen",
        "target": "y",
    },
    {
        "name": "james-stein/independent",
        "encoder": "JamesSteinEncoder",
        "params": {"columns": ["color"], "target": "y"},
        "fit": "labelled",
        "target": "y",
    },
    {
        "name": "james-stein/pooled",
        "encoder": "JamesSteinEncoder",
        "params": {"columns": ["color"], "target": "y", "model": "pooled"},
        "fit": "labelled",
        "target": "y",
    },
    {
        "name": "james-stein/beta",
        "encoder": "JamesSteinEncoder",
        "params": {"columns": ["color"], "target": "y", "model": "beta"},
        "fit": "labelled",
        "target": "y",
    },
    {
        "name": "james-stein/two-columns",
        "encoder": "JamesSteinEncoder",
        "params": {"columns": ["color", "size"], "target": "y"},
        "fit": "labelled",
        "target": "y",
    },
    {
        "name": "james-stein/integer-target",
        "encoder": "JamesSteinEncoder",
        "params": {"columns": ["color"], "target": "y"},
        "fit": "labelled-binary",
        "target": "y",
    },
    {
        "name": "james-stein/nulls",
        "encoder": "JamesSteinEncoder",
        "params": {"columns": ["color"], "target": "y"},
        "fit": "labelled-nulls",
        "target": "y",
    },
    {
        "name": "james-stein/nulls-return-nan",
        "encoder": "JamesSteinEncoder",
        "params": {"columns": ["color"], "target": "y", "handle_missing": "return_nan"},
        "fit": "labelled-nulls",
        "target": "y",
    },
    {
        "name": "james-stein/unseen",
        "encoder": "JamesSteinEncoder",
        "params": {"columns": ["color"], "target": "y"},
        "fit": "labelled",
        "transform": "labelled-unseen",
        "target": "y",
    },
    {
        "name": "james-stein/unseen-return-nan",
        "encoder": "JamesSteinEncoder",
        "params": {"columns": ["color"], "target": "y", "handle_unknown": "return_nan"},
        "fit": "labelled",
        "transform": "labelled-unseen",
        "target": "y",
    },
    # The leave-one-out family is compared against upstream's transform without
    # a target: its inference branch, the only form our frozen state can be a
    # pure function of. run_case never passes y to transform, which is what
    # makes these cases meaningful.
    {
        "name": "leave-one-out/plain",
        "encoder": "LeaveOneOutEncoder",
        "params": {"columns": ["color"], "target": "y"},
        "fit": "labelled",
        "target": "y",
    },
    {
        "name": "leave-one-out/integer-target",
        "encoder": "LeaveOneOutEncoder",
        "params": {"columns": ["color", "size"], "target": "y"},
        "fit": "labelled-binary",
        "target": "y",
    },
    {
        "name": "leave-one-out/nulls",
        "encoder": "LeaveOneOutEncoder",
        "params": {"columns": ["color"], "target": "y"},
        "fit": "labelled-nulls",
        "target": "y",
    },
    {
        "name": "leave-one-out/nulls-return-nan",
        "encoder": "LeaveOneOutEncoder",
        "params": {"columns": ["color"], "target": "y", "handle_missing": "return_nan"},
        "fit": "labelled-nulls",
        "target": "y",
    },
    {
        "name": "leave-one-out/unseen",
        "encoder": "LeaveOneOutEncoder",
        "params": {"columns": ["color"], "target": "y"},
        "fit": "labelled",
        "transform": "labelled-unseen",
        "target": "y",
    },
    {
        "name": "leave-one-out/unseen-return-nan",
        "encoder": "LeaveOneOutEncoder",
        "params": {"columns": ["color"], "target": "y", "handle_unknown": "return_nan"},
        "fit": "labelled",
        "transform": "labelled-unseen",
        "target": "y",
    },
    {
        "name": "cat-boost/plain",
        "encoder": "CatBoostEncoder",
        "params": {"columns": ["color"], "target": "y"},
        "fit": "labelled",
        "target": "y",
    },
    {
        "name": "cat-boost/smoothing",
        "encoder": "CatBoostEncoder",
        "params": {"columns": ["color"], "target": "y", "a": 2.5},
        "fit": "labelled",
        "target": "y",
    },
    {
        "name": "cat-boost/integer-target",
        "encoder": "CatBoostEncoder",
        "params": {"columns": ["color", "size"], "target": "y"},
        "fit": "labelled-binary",
        "target": "y",
    },
    {
        "name": "cat-boost/nulls",
        "encoder": "CatBoostEncoder",
        "params": {"columns": ["color"], "target": "y"},
        "fit": "labelled-nulls",
        "target": "y",
    },
    {
        "name": "cat-boost/nulls-return-nan",
        "encoder": "CatBoostEncoder",
        "params": {"columns": ["color"], "target": "y", "handle_missing": "return_nan"},
        "fit": "labelled-nulls",
        "target": "y",
    },
    {
        "name": "cat-boost/unseen",
        "encoder": "CatBoostEncoder",
        "params": {"columns": ["color"], "target": "y"},
        "fit": "labelled",
        "transform": "labelled-unseen",
        "target": "y",
    },
    {
        "name": "cat-boost/unseen-return-nan",
        "encoder": "CatBoostEncoder",
        "params": {"columns": ["color"], "target": "y", "handle_unknown": "return_nan"},
        "fit": "labelled",
        "transform": "labelled-unseen",
        "target": "y",
    },
    {
        "name": "quantile/median",
        "encoder": "QuantileEncoder",
        "params": {"columns": ["color"]},
        "target": "y",
        "fit": "target_plain",
    },
    {
        "name": "quantile/two-columns",
        "encoder": "QuantileEncoder",
        "params": {"columns": ["color", "size"]},
        "target": "y",
        "fit": "target_plain",
    },
    {
        "name": "quantile/odd-and-even-groups",
        "encoder": "QuantileEncoder",
        "params": {"columns": ["color"]},
        "target": "y",
        "fit": "target_spread",
    },
    {
        "name": "quantile/q25",
        "encoder": "QuantileEncoder",
        "params": {"columns": ["color"], "quantile": 0.25},
        "target": "y",
        "fit": "target_spread",
    },
    {
        "name": "quantile/q90-smoothed",
        "encoder": "QuantileEncoder",
        "params": {"columns": ["color"], "quantile": 0.9, "m": 2.5},
        "target": "y",
        "fit": "target_spread",
    },
    {
        "name": "quantile/m-zero",
        "encoder": "QuantileEncoder",
        "params": {"columns": ["color"], "m": 0},
        "target": "y",
        "fit": "target_spread",
    },
    {
        "name": "quantile/integer-target",
        "encoder": "QuantileEncoder",
        "params": {"columns": ["color"], "quantile": 0.25},
        "target": "y",
        "fit": "target_integer",
    },
    {
        "name": "quantile/nulls",
        "encoder": "QuantileEncoder",
        "params": {"columns": ["color"]},
        "target": "y",
        "fit": "target_nulls",
    },
    {
        "name": "quantile/nulls-return-nan",
        "encoder": "QuantileEncoder",
        "params": {"columns": ["color"], "handle_missing": "return_nan"},
        "target": "y",
        "fit": "target_nulls",
    },
    {
        "name": "quantile/unseen",
        "encoder": "QuantileEncoder",
        "params": {"columns": ["color"]},
        "target": "y",
        "fit": "target_plain",
        "transform": "target_unseen",
    },
    {
        "name": "quantile/unseen-return-nan",
        "encoder": "QuantileEncoder",
        "params": {"columns": ["color"], "handle_unknown": "return_nan"},
        "target": "y",
        "fit": "target_plain",
        "transform": "target_unseen",
    },
    {
        "name": "summary/default-quantiles",
        "encoder": "SummaryEncoder",
        "params": {"columns": ["color"]},
        "target": "y",
        "fit": "target_plain",
    },
    {
        "name": "summary/two-columns",
        "encoder": "SummaryEncoder",
        "params": {"columns": ["color", "size"]},
        "target": "y",
        "fit": "target_plain",
    },
    {
        "name": "summary/three-quantiles",
        "encoder": "SummaryEncoder",
        "params": {"columns": ["color"], "quantiles": [0.1, 0.5, 0.9], "m": 2.0},
        "target": "y",
        "fit": "target_spread",
    },
    {
        "name": "summary/nulls",
        "encoder": "SummaryEncoder",
        "params": {"columns": ["color"]},
        "target": "y",
        "fit": "target_nulls",
    },
    {
        "name": "summary/unseen",
        "encoder": "SummaryEncoder",
        "params": {"columns": ["color"], "quantiles": [0.25, 0.5, 0.75]},
        "target": "y",
        "fit": "target_plain",
        "transform": "target_unseen",
    },
    {"name": "basen/plain", "encoder": "BaseNEncoder", "params": {"columns": ["color"]}, "fit": "plain"},
    {
        "name": "basen/base-1",
        "encoder": "BaseNEncoder",
        "params": {"columns": ["color"], "base": 1},
        "fit": "plain",
    },
    {
        "name": "basen/base-3",
        "encoder": "BaseNEncoder",
        "params": {"columns": ["color"], "base": 3},
        "fit": "plain",
    },
    {
        "name": "basen/base-10",
        "encoder": "BaseNEncoder",
        "params": {"columns": ["color"], "base": 10},
        "fit": "plain",
    },
    {"name": "basen/nulls", "encoder": "BaseNEncoder", "params": {"columns": ["color"]}, "fit": "nulls"},
    {
        "name": "basen/nulls-return-nan",
        "encoder": "BaseNEncoder",
        "params": {"columns": ["color"], "handle_missing": "return_nan"},
        "fit": "nulls",
    },
    {
        "name": "basen/two-columns",
        "encoder": "BaseNEncoder",
        "params": {"columns": ["color", "size"], "base": 3},
        "fit": "plain",
    },
    {
        "name": "basen/unseen",
        "encoder": "BaseNEncoder",
        "params": {"columns": ["color"]},
        "fit": "plain",
        "transform": "unseen",
    },
    {
        "name": "basen/unseen-return-nan",
        "encoder": "BaseNEncoder",
        "params": {"columns": ["color"], "handle_unknown": "return_nan"},
        "fit": "plain",
        "transform": "unseen",
    },
    {
        "name": "basen/indicator",
        "encoder": "BaseNEncoder",
        "params": {"columns": ["color"], "handle_unknown": "indicator"},
        "fit": "plain",
        "transform": "unseen",
    },
    {
        # Three categories, where the pre-encoder's slot for the missing value
        # is the level that crosses the power of two: two digits under
        # handle_missing='value' and three under the others. "color" has four
        # categories and hides the difference, which is why these use "size".
        "name": "basen/size-missing-return-nan",
        "encoder": "BaseNEncoder",
        "params": {"columns": ["size"], "handle_missing": "return_nan"},
        "fit": "plain",
    },
    {
        "name": "basen/size-missing-error",
        "encoder": "BaseNEncoder",
        "params": {"columns": ["size"], "handle_missing": "error"},
        "fit": "plain",
    },
    {
        "name": "basen/single-level-return-nan",
        "encoder": "BaseNEncoder",
        "params": {"columns": ["color"], "handle_missing": "return_nan"},
        "fit": "single",
    },
    {
        "name": "james-stein/missing-error-at-transform",
        "encoder": "JamesSteinEncoder",
        "params": {"columns": ["color"], "target": "y", "handle_missing": "error"},
        "fit": "labelled",
        "transform": "labelled-late-nulls",
        "target": "y",
        "raises": True,
    },
    {"name": "binary/plain", "encoder": "BinaryEncoder", "params": {"columns": ["color"]}, "fit": "plain"},
    {
        "name": "binary/missing-error-at-transform",
        "encoder": "BinaryEncoder",
        "params": {"columns": ["color"], "handle_missing": "error"},
        "fit": "plain",
        "transform": "late-nulls",
        "raises": True,
    },
    {"name": "binary/nulls", "encoder": "BinaryEncoder", "params": {"columns": ["color"]}, "fit": "nulls"},
    {
        "name": "binary/two-columns",
        "encoder": "BinaryEncoder",
        "params": {"columns": ["color", "size"]},
        "fit": "plain",
    },
    {
        "name": "binary/unseen",
        "encoder": "BinaryEncoder",
        "params": {"columns": ["color"]},
        "fit": "plain",
        "transform": "unseen",
    },
    {
        "name": "binary/indicator",
        "encoder": "BinaryEncoder",
        "params": {"columns": ["color"], "handle_unknown": "indicator"},
        "fit": "plain",
        "transform": "unseen",
    },
    {
        "name": "binary/size-missing-return-nan",
        "encoder": "BinaryEncoder",
        "params": {"columns": ["size"], "handle_missing": "return_nan"},
        "fit": "plain",
    },
    {"name": "gray/plain", "encoder": "GrayEncoder", "params": {"columns": ["color"]}, "fit": "plain"},
    # The library sorts the categories for Gray, so this one holds on data
    # whose categories do not first appear in sorted order.
    {"name": "gray/unsorted", "encoder": "GrayEncoder", "params": {"columns": ["color", "size"]}, "fit": "shuffled"},
    {"name": "gray/nulls", "encoder": "GrayEncoder", "params": {"columns": ["color"]}, "fit": "nulls"},
    {
        "name": "gray/nulls-return-nan",
        "encoder": "GrayEncoder",
        "params": {"columns": ["color"], "handle_missing": "return_nan"},
        "fit": "nulls",
    },
    {
        "name": "gray/unseen",
        "encoder": "GrayEncoder",
        "params": {"columns": ["color"]},
        "fit": "plain",
        "transform": "unseen",
    },
    {
        "name": "gray/indicator",
        "encoder": "GrayEncoder",
        "params": {"columns": ["color"], "handle_unknown": "indicator"},
        "fit": "shuffled",
        "transform": "unseen",
    },
]


def run_case(case: dict[str, Any]) -> dict[str, list[Any]]:
    """Fit and transform one case with the real library.

    Args:
        case: The case description.

    Returns:
        The encoded columns as plain lists.

    """
    import category_encoders
    import pandas as pd

    upstream_class, renames = UPSTREAM[case["encoder"]]
    target = case.get("target")
    params = {
        renames.get(key, key): value
        for key, value in case["params"].items()
        if key != "target" and renames.get(key, key) is not None
    }
    encoder = getattr(category_encoders, upstream_class)(**params)
    fit_frame = pd.DataFrame(DATASETS[case["fit"]])
    transform_frame = pd.DataFrame(DATASETS[case.get("transform", case["fit"])])
    if target is None:
        encoder.fit(fit_frame)
        out = encoder.transform(transform_frame)
    else:
        # Supervised: we name the target column, upstream takes it as y. The
        # transform runs without y on purpose: that is the inference branch,
        # the only one a frozen state can reproduce, and the one in which the
        # leave-one-out family differs from its fit_transform.
        encoder.fit(fit_frame.drop(columns=[target]), fit_frame[target])
        out = encoder.transform(transform_frame.drop(columns=[target]))
    encoded: dict[str, list[Any]] = {}
    for column in out.columns:
        values = out[column].tolist()
        encoded[str(column)] = [None if isinstance(v, float) and math.isnan(v) else v for v in values]
    return encoded


def main() -> None:
    """Write the fixture next to the tests."""
    cases = []
    for case in CASES:
        entry = dict(case)
        if case.get("raises"):
            # Record what the library refused, and refuse to record a case
            # that no longer raises: the fixture must never drift quietly.
            try:
                run_case(case)
            except ValueError as error:
                entry["raised"] = str(error)
            else:
                raise SystemExit(f"{case['name']}: expected category_encoders to raise, but it did not")
        else:
            entry["expected"] = run_case(case)
        cases.append(entry)
    fixture = {"category_encoders": VERSION, "datasets": DATASETS, "cases": cases}
    path = Path(__file__).resolve().parent.parent / "test" / "fixtures" / "category_encoders.json"
    path.write_text(json.dumps(fixture, indent=2, sort_keys=False) + "\n")
    print(f"wrote {len(cases)} cases to {path}")


if __name__ == "__main__":
    main()
