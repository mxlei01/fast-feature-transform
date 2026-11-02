// Explicit C++ tests for the contrast coders: the four matrices, the levels
// that handle_missing and handle_unknown add, the accumulator-merge contract,
// batch-vs-record parity, and load_state.

#include <arrow/array.h>

#include <algorithm>
#include <bit>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <cmath>
#include <limits>
#include <sstream>

#include "fastfeaturetransform/kernel.hpp"
#include "fastfeaturetransform/test_support.hpp"

namespace fft = fastfeaturetransform;
using Catch::Matchers::ContainsSubstring;
using fft::testing::make_batch;
using fft::testing::utf8_column;

namespace {

// Fit the way the engine does: resolve, run the pass, then resolve the same
// input schema again, because what the encoder emits is known only afterwards.
std::unique_ptr<fft::Kernel> fit_kernel(const std::string& id, const nlohmann::json& params, fft::Schema& schema,
                                        const std::vector<fft::Batch>& batches) {
  auto kernel = fft::make_kernel(id, params);
  kernel->stage = "enc";
  const fft::Schema input = schema;
  kernel->resolve(schema);
  std::vector<std::unique_ptr<fft::AccState>> states;
  for (const fft::Batch& batch : batches) {
    states.push_back(kernel->make_state());
    kernel->consume(*states.back(), batch);
  }
  kernel->finalize(std::move(states));
  schema = input;
  kernel->resolve(schema);
  return kernel;
}

// Every column of one transformed batch from index `first` on, as rows of
// doubles; a null becomes NaN so a row compares as a whole.
std::vector<std::vector<double>> coded(const fft::Batch& batch, std::size_t first) {
  std::vector<std::vector<double>> rows;
  for (int64_t row = 0; row < batch.rows; ++row) {
    std::vector<double> values;
    for (std::size_t col = first; col < batch.cols.size(); ++col) {
      const auto& doubles = static_cast<const arrow::DoubleArray&>(*batch.cols[col]);
      values.push_back(doubles.IsNull(row) ? std::nan("") : doubles.Value(row));
    }
    rows.push_back(std::move(values));
  }
  return rows;
}

bool same_row(const std::vector<double>& ours, const std::vector<double>& theirs) {
  if (ours.size() != theirs.size()) return false;
  for (std::size_t i = 0; i < ours.size(); ++i) {
    if (std::isnan(ours[i]) != std::isnan(theirs[i])) return false;
    if (!std::isnan(ours[i]) && std::fabs(ours[i] - theirs[i]) > 1e-12) return false;
  }
  return true;
}

std::vector<std::optional<std::string>> numbered_levels(std::size_t count) {
  std::vector<std::optional<std::string>> values;
  for (std::size_t i = 0; i < count; ++i) values.emplace_back("c" + std::to_string(1000 + i));
  return values;
}

}  // namespace

TEST_CASE("the contrast coders validate their params") {
  REQUIRE_THROWS_WITH((fft::make_kernel("helmert_encode", {{"handle_missing", "sometimes"}})),
                      ContainsSubstring("'handle_missing' must be 'value', 'return_nan' or 'error'"));
  REQUIRE_THROWS_WITH((fft::make_kernel("sum_encode", {{"handle_unknown", "maybe"}})),
                      ContainsSubstring("'handle_unknown' must be 'value', 'return_nan', 'indicator' or 'error'"));
  REQUIRE_THROWS_WITH((fft::make_kernel("polynomial_encode", {{"columns", nlohmann::json::array()}})),
                      ContainsSubstring("'columns' must be a non-empty list"));
  REQUIRE_THROWS_WITH((fft::make_kernel("backward_difference_encode", {{"columns", {"c", "c"}}})),
                      ContainsSubstring("column 'c' listed more than once"));
  // All four take the indicator level, unlike rank_hot_encode.
  for (const char* id : {"backward_difference_encode", "helmert_encode", "sum_encode", "polynomial_encode"}) {
    REQUIRE_NOTHROW((fft::make_kernel(id, {{"handle_unknown", "indicator"}})));
    REQUIRE(fft::make_kernel(id, nlohmann::json::object())->id() == id);
  }
}

TEST_CASE("helmert_encode rewrites the schema and puts the fitted null last") {
  fft::Schema schema{{{"c", "utf8"}, {"n", "i64"}}};
  const fft::Batch first = make_batch({"c"}, {utf8_column({"c", "a"})});
  const fft::Batch second = make_batch({"c"}, {utf8_column({"b", std::nullopt, "a"})});
  auto kernel = fit_kernel("helmert_encode", {{"columns", {"c"}}}, schema, {first, second});

  // Four levels — a, b, c and the null, which sorts last — so three columns.
  REQUIRE(schema.cols == fft::Schema{{{"n", "i64"}, {"c_0", "f64"}, {"c_1", "f64"}, {"c_2", "f64"}}}.cols);
  REQUIRE(kernel->outputs == std::vector<std::string>{"c_0", "c_1", "c_2"});
  REQUIRE(kernel->state().at("categories").at("c") == nlohmann::json({"a", "b", "c"}));
  REQUIRE(kernel->state().at("has_null").at("c") == true);

  // Whichever order the workers merged in, the fitted state is the same.
  fft::Schema other{{{"c", "utf8"}, {"n", "i64"}}};
  REQUIRE(fit_kernel("helmert_encode", {{"columns", {"c"}}}, other, {second, first})->state() == kernel->state());

  const std::vector<std::optional<std::string>> values{"a", "b", "c", std::nullopt, "zz"};
  const fft::Batch out = kernel->transform(make_batch({"c"}, {utf8_column(values)}));
  const std::vector<std::vector<double>> rows = coded(out, 0);
  REQUIRE(rows[0] == std::vector<double>{-1.0, -1.0, -1.0});
  REQUIRE(rows[1] == std::vector<double>{1.0, -1.0, -1.0});
  REQUIRE(rows[2] == std::vector<double>{0.0, 2.0, -1.0});
  REQUIRE(rows[3] == std::vector<double>{0.0, 0.0, 3.0});  // the null level
  REQUIRE(rows[4] == std::vector<double>{0.0, 0.0, 0.0});  // unknown under 'value'

  // Batch and record agree value for value.
  for (std::size_t i = 0; i < values.size(); ++i) {
    fft::RowValue value = values[i].has_value() ? fft::RowValue(*values[i]) : fft::RowValue(std::monostate{});
    const fft::Row row = kernel->transform_row({{"c", value}});
    REQUIRE(row.size() == 3);
    std::vector<double> theirs;
    for (const auto& [key, held] : row) theirs.push_back(std::get<double>(held));
    REQUIRE(same_row(rows[i], theirs));
  }
}

TEST_CASE("each coder builds its own matrix") {
  auto fit_four = [](const std::string& id, const nlohmann::json& params) {
    fft::Schema schema{{{"c", "utf8"}}};
    return fit_kernel(id, params, schema, {make_batch({"c"}, {utf8_column({"a", "b", "c", "d"})})});
  };
  const std::vector<std::optional<std::string>> levels{"a", "b", "c", "d"};
  auto rows_of = [&](const std::unique_ptr<fft::Kernel>& kernel) {
    return coded(kernel->transform(make_batch({"c"}, {utf8_column(levels)})), 0);
  };

  const std::vector<std::vector<double>> difference = rows_of(fit_four("backward_difference_encode", {}));
  REQUIRE(difference[0] == std::vector<double>{-0.75, -0.5, -0.25});
  REQUIRE(difference[1] == std::vector<double>{0.25, -0.5, -0.25});
  REQUIRE(difference[2] == std::vector<double>{0.25, 0.5, -0.25});
  REQUIRE(difference[3] == std::vector<double>{0.25, 0.5, 0.75});

  const std::vector<std::vector<double>> sum = rows_of(fit_four("sum_encode", {}));
  REQUIRE(sum[0] == std::vector<double>{1.0, 0.0, 0.0});
  REQUIRE(sum[2] == std::vector<double>{0.0, 0.0, 1.0});
  REQUIRE(sum[3] == std::vector<double>{-1.0, -1.0, -1.0});  // the last level is the reference

  // patsy's Poly() for four levels, to the tolerance the parity fixture uses.
  const std::vector<std::vector<double>> poly = rows_of(fit_four("polynomial_encode", {}));
  REQUIRE(same_row(poly[0], {-0.6708203932499369, 0.5, -0.223606797749979}));
  REQUIRE(same_row(poly[1], {-0.22360679774997896, -0.5, 0.6708203932499368}));
  REQUIRE(same_row(poly[2], {0.22360679774997902, -0.5, -0.6708203932499369}));
  REQUIRE(same_row(poly[3], {0.670820393249937, 0.5, 0.2236067977499791}));
}

TEST_CASE("polynomial contrasts remain an accurate ordered basis at higher cardinality") {
  // Raw powers followed by one Gram-Schmidt pass lose orthogonality already
  // at 32 levels, and overflow their squared norm at 128. These checks use
  // discrete-polynomial identities rather than another floating-point QR.
  for (const std::size_t count : {32U, 64U, 128U}) {
    INFO("levels = " << count);
    const auto values = numbered_levels(count);
    const fft::Batch batch = make_batch({"c"}, {utf8_column(values)});
    fft::Schema schema{{{"c", "utf8"}}};
    auto kernel = fit_kernel("polynomial_encode", {{"columns", {"c"}}}, schema, {batch});
    const auto rows = coded(kernel->transform(batch), 0);
    REQUIRE(rows.size() == count);
    for (const auto& row : rows) {
      REQUIRE(row.size() == count - 1);
      REQUIRE(std::all_of(row.begin(), row.end(), [](double value) { return std::isfinite(value); }));
    }

    const long double n = static_cast<long double>(count);
    const long double constant = 1.0L / std::sqrt(n);
    auto value = [&](std::size_t row, std::size_t degree) -> long double {
      return degree == 0 ? constant : rows[row][degree - 1];
    };
    long double gram_error = 0.0L;
    for (std::size_t first = 0; first < count; ++first) {
      for (std::size_t second = first; second < count; ++second) {
        long double dot = 0.0L;
        for (std::size_t i = 0; i < count; ++i) dot += value(i, first) * value(i, second);
        gram_error = std::max(gram_error, std::fabs(dot - (first == second ? 1.0L : 0.0L)));
      }
    }
    REQUIRE(gram_error < 5e-13L);

    const long double linear_norm = std::sqrt(n * (n * n - 1.0L) / 12.0L);
    const long double quadratic_norm = std::sqrt(n * (n * n - 1.0L) * (n * n - 4.0L) / 180.0L);
    long double low_degree_error = 0.0L;
    long double symmetry_error = 0.0L;
    for (std::size_t i = 0; i < count; ++i) {
      const long double x = static_cast<long double>(i) - (n - 1.0L) / 2.0L;
      low_degree_error = std::max(low_degree_error, std::fabs(value(i, 1) - x / linear_norm));
      low_degree_error =
          std::max(low_degree_error, std::fabs(value(i, 2) - (x * x - (n * n - 1.0L) / 12.0L) / quadratic_norm));
      for (std::size_t degree = 1; degree < count; ++degree) {
        const long double sign = degree % 2 == 0 ? 1.0L : -1.0L;
        symmetry_error = std::max(symmetry_error, std::fabs(value(i, degree) - sign * value(count - 1 - i, degree)));
      }
    }
    REQUIRE(low_degree_error < 5e-13L);
    REQUIRE(symmetry_error < 5e-13L);

    // The last degree has a closed form: alternating binomial coefficients,
    // with the final entry positive, normalized to unit length.
    std::vector<long double> highest(count, 1.0L);
    long double highest_norm = 1.0L;
    for (std::size_t i = 1; i < count; ++i) {
      highest[i] = highest[i - 1] * static_cast<long double>(count - i) / static_cast<long double>(i);
      highest_norm += highest[i] * highest[i];
    }
    highest_norm = std::sqrt(highest_norm);
    long double highest_error = 0.0L;
    for (std::size_t i = 0; i < count; ++i) {
      const long double sign = (count - 1 + i) % 2 == 0 ? 1.0L : -1.0L;
      highest_error = std::max(highest_error, std::fabs(value(i, count - 1) - sign * highest[i] / highest_norm));
    }
    REQUIRE(highest_error < 5e-13L);

    // Every degree above the constant obeys the independent three-term
    // recurrence for orthonormal polynomials on equally spaced, centered
    // integer scores.
    auto coefficient = [n](std::size_t degree) {
      const long double k = static_cast<long double>(degree);
      return std::sqrt(k * k * (n * n - k * k) / (4.0L * (4.0L * k * k - 1.0L)));
    };
    long double recurrence_error = 0.0L;
    for (std::size_t degree = 1; degree < count; ++degree) {
      for (std::size_t i = 0; i < count; ++i) {
        const long double x = static_cast<long double>(i) - (n - 1.0L) / 2.0L;
        const long double previous = degree == 1 ? 0.0L : coefficient(degree - 1) * value(i, degree - 2);
        const long double residual = coefficient(degree) * value(i, degree) - x * value(i, degree - 1) + previous;
        recurrence_error = std::max(recurrence_error, std::fabs(residual));
      }
    }
    REQUIRE(recurrence_error < 5e-12L);

    for (std::size_t i = 0; i < count; ++i) {
      const auto record = kernel->transform_row({{"c", *values[i]}});
      REQUIRE(record.size() == count - 1);
      for (std::size_t col = 0; col + 1 < count; ++col) REQUIRE(std::get<double>(record[col].second) == rows[i][col]);
    }
  }
}

TEST_CASE("the level count follows handle_missing and handle_unknown") {
  auto fit_three = [](const nlohmann::json& params) {
    fft::Schema schema{{{"c", "utf8"}}};
    return fit_kernel("sum_encode", params, schema, {make_batch({"c"}, {utf8_column({"a", "b", std::nullopt})})});
  };
  const std::vector<std::optional<std::string>> values{"a", std::nullopt, "zz"};
  auto rows_of = [&](const std::unique_ptr<fft::Kernel>& kernel) {
    return coded(kernel->transform(make_batch({"c"}, {utf8_column(values)})), 0);
  };

  // handle_missing='value': the null is a level, so three levels, two columns.
  const std::vector<std::vector<double>> valued = rows_of(fit_three({{"columns", {"c"}}}));
  REQUIRE(valued[0] == std::vector<double>{1.0, 0.0});
  REQUIRE(valued[1] == std::vector<double>{-1.0, -1.0});  // the null is the reference level
  REQUIRE(valued[2] == std::vector<double>{0.0, 0.0});

  // REGRESSION: handle_missing='return_nan' drops the null level but keeps the
  // pre-encoder's placeholder for it, so the matrix is still three levels wide
  // and two columns come out, not the one this used to emit. Nothing reaches
  // the placeholder row: a null is a row of nulls.
  auto nan_kernel = fit_three({{"columns", {"c"}}, {"handle_missing", "return_nan"}});
  REQUIRE(nan_kernel->outputs == std::vector<std::string>{"c_0", "c_1"});
  const std::vector<std::vector<double>> nan_missing = rows_of(nan_kernel);
  REQUIRE(nan_missing[0] == std::vector<double>{1.0, 0.0});
  REQUIRE(std::isnan(nan_missing[1][0]));
  REQUIRE(std::isnan(nan_missing[1][1]));
  REQUIRE(nan_missing[2] == std::vector<double>{0.0, 0.0});

  // handle_missing='error' keeps the placeholder too — the pre-encoder holds
  // the slot whatever the option — so the width matches 'return_nan'.
  fft::Schema clean_schema{{{"c", "utf8"}}};
  auto strict = fit_kernel("sum_encode", {{"columns", {"c"}}, {"handle_missing", "error"}}, clean_schema,
                           {make_batch({"c"}, {utf8_column({"a", "b"})})});
  REQUIRE(strict->outputs == std::vector<std::string>{"c_0", "c_1"});

  // handle_unknown='indicator' adds a real fourth level after the null one.
  auto indicator = fit_three({{"columns", {"c"}}, {"handle_unknown", "indicator"}});
  REQUIRE(indicator->outputs == std::vector<std::string>{"c_0", "c_1", "c_2"});
  const std::vector<std::vector<double>> flagged = rows_of(indicator);
  REQUIRE(flagged[0] == std::vector<double>{1.0, 0.0, 0.0});
  REQUIRE(flagged[1] == std::vector<double>{0.0, 0.0, 1.0});     // the null level
  REQUIRE(flagged[2] == std::vector<double>{-1.0, -1.0, -1.0});  // the indicator level, last

  // The indicator level still sorts after the placeholder one.
  auto both = fit_three({{"columns", {"c"}}, {"handle_missing", "return_nan"}, {"handle_unknown", "indicator"}});
  REQUIRE(both->outputs == std::vector<std::string>{"c_0", "c_1", "c_2"});
  const std::vector<std::vector<double>> flagged_nan = rows_of(both);
  REQUIRE(flagged_nan[0] == std::vector<double>{1.0, 0.0, 0.0});
  REQUIRE(std::isnan(flagged_nan[1][0]));
  REQUIRE(flagged_nan[2] == std::vector<double>{-1.0, -1.0, -1.0});

  // handle_unknown='return_nan' nulls the whole row; 'error' refuses it.
  const std::vector<std::vector<double>> nan_unknown =
      rows_of(fit_three({{"columns", {"c"}}, {"handle_unknown", "return_nan"}}));
  REQUIRE(std::isnan(nan_unknown[2][0]));
  REQUIRE_THROWS_WITH(rows_of(fit_three({{"columns", {"c"}}, {"handle_unknown", "error"}})),
                      ContainsSubstring("has a category that was not seen at fit"));
  REQUIRE_THROWS_WITH(fit_three({{"columns", {"c"}}, {"handle_missing", "error"}}),
                      ContainsSubstring("has nulls and handle_missing is 'error'"));
}

TEST_CASE("a single level emits no columns at all") {
  fft::Schema schema{{{"c", "utf8"}, {"n", "i64"}}};
  auto kernel = fit_kernel("polynomial_encode", {{"columns", {"c"}}, {"handle_unknown", "indicator"}}, schema,
                           {make_batch({"c"}, {utf8_column({"a", "a"})})});
  REQUIRE(schema.cols == fft::Schema{{{"n", "i64"}}}.cols);
  REQUIRE(kernel->outputs.empty());
  const fft::Batch out = kernel->transform(make_batch({"c"}, {utf8_column({"a", "zz"})}));
  REQUIRE(out.names.empty());
  // Mostly null, so written row at a time.
  REQUIRE(kernel->transform(make_batch({"c"}, {utf8_column({"a", std::nullopt, std::nullopt})})).names.empty());
  REQUIRE(kernel->transform_row({{"c", std::string("a")}}).empty());

  // The column is still read, so handle_unknown='error' still refuses, either way.
  fft::Schema strict_schema{{{"c", "utf8"}}};
  auto strict = fit_kernel("polynomial_encode", {{"columns", {"c"}}, {"handle_unknown", "error"}}, strict_schema,
                           {make_batch({"c"}, {utf8_column({"a", "a"})})});
  REQUIRE_THROWS_WITH(strict->transform(make_batch({"c"}, {utf8_column({"zz"})})),
                      ContainsSubstring("has a category that was not seen at fit"));
  REQUIRE_THROWS_WITH(strict->transform(make_batch({"c"}, {utf8_column({std::nullopt, "zz", std::nullopt})})),
                      ContainsSubstring("has a category that was not seen at fit"));
}

// REGRESSION: the placeholder level counts toward the two levels a contrast
// needs, so one category under handle_missing='return_nan' contrasts against
// it and emits a column. It used to emit nothing at all.
TEST_CASE("a single level still contrasts against the placeholder level") {
  const std::vector<std::pair<const char*, double>> first_row{{"backward_difference_encode", -0.5},
                                                              {"helmert_encode", -1.0},
                                                              {"sum_encode", 1.0},
                                                              {"polynomial_encode", -0.7071067811865475}};
  for (const auto& [id, expected] : first_row) {
    fft::Schema schema{{{"c", "utf8"}}};
    auto kernel = fit_kernel(id, {{"columns", {"c"}}, {"handle_missing", "return_nan"}}, schema,
                             {make_batch({"c"}, {utf8_column({"a", "a"})})});
    REQUIRE(kernel->outputs == std::vector<std::string>{"c_0"});
    const std::vector<std::vector<double>> rows =
        coded(kernel->transform(make_batch({"c"}, {utf8_column({"a", "zz", std::nullopt})})), 0);
    REQUIRE(std::fabs(rows[0][0] - expected) < 1e-12);
    REQUIRE(rows[1][0] == 0.0);  // an unknown under handle_unknown='value'
    REQUIRE(std::isnan(rows[2][0]));
    REQUIRE(std::get<double>(kernel->transform_row({{"c", std::string("a")}})[0].second) == rows[0][0]);
  }
}

// A dense batch is gathered column at a time, which writes the zero row and
// the slot under a null explicitly; they must stay +0.0 bit for bit. The next
// test covers the other ways a batch is written.
TEST_CASE("the contrast coders write +0.0 for the zero row and under a null") {
  for (const char* id : {"backward_difference_encode", "helmert_encode", "sum_encode", "polynomial_encode"}) {
    CAPTURE(id);
    fft::Schema schema{{{"c", "utf8"}}};
    auto kernel = fit_kernel(id, {{"columns", {"c"}}, {"handle_missing", "return_nan"}}, schema,
                             {make_batch({"c"}, {utf8_column({"a", "b", "c"})})});
    const fft::Batch out = kernel->transform(make_batch({"c"}, {utf8_column({"a", "zz", std::nullopt, "c"})}));
    REQUIRE(out.cols.size() == 3);
    for (const auto& column : out.cols) {
      const auto& doubles = static_cast<const arrow::DoubleArray&>(*column);
      REQUIRE(doubles.null_count() == 1);
      REQUIRE(doubles.IsNull(2));
      REQUIRE(std::bit_cast<uint64_t>(doubles.Value(1)) == 0);  // an unknown under handle_unknown='value'
      REQUIRE(std::bit_cast<uint64_t>(doubles.Value(2)) == 0);  // the slot under the null
    }
  }
}

// A batch is written row at a time when nulls are more than half of it (up to
// 10 columns) or leave few enough coded rows for its width (wider), column at a
// time otherwise, and a column-at-a-time batch of 4 or more columns whose rows
// turn out mostly zero rows is zero-filled. Every way must give the record
// path's answer for every row, and +0.0 under a null. Under
// handle_missing='value' a null is the zero row instead of a null row. The
// shift moves which rows are coded; with 999, the first and last rows are
// coded rows in every shape.
TEST_CASE("the contrast coders' row-at-a-time and column-at-a-time writes agree with the record path") {
  struct Shape {
    int levels;   // fitted categories; the output width under 'return_nan', one fewer under 'value'
    int nulls;    // of 1,000 rows
    int unknown;  // never fitted: the zero row under handle_unknown='value'
  };
  const std::vector<Shape> shapes{
      {4, 600, 0},    // narrow, more than half null: row at a time
      {4, 600, 150},  // the same with unknowns: zero rows row at a time
      {4, 400, 450},  // narrow, under half null: column at a time
      {4, 0, 980},    // narrow, 98% unknown: zero-filled at 4 columns, gathered at 3
      {4, 0, 0},      // narrow, dense
      {16, 950, 0},   // wide, 95% null: row at a time
      {16, 950, 30},  // the same with unknowns
      {16, 860, 60},  // wide, 86% null and 6% unknown: column at a time, zero-filled
      {16, 0, 960},   // wide, 96% unknown: column at a time, zero-filled
      {16, 300, 0},   // wide, 30% null: gathered
      {32, 960, 0},   // widest, 96% null: row at a time
      {32, 900, 0},   // widest, 90% null: column at a time, zero-filled
  };
  for (const char* id : {"backward_difference_encode", "helmert_encode", "sum_encode", "polynomial_encode"}) {
    for (const char* missing : {"return_nan", "value"}) {
      for (const Shape& shape : shapes) {
        std::vector<std::optional<std::string>> fitted;
        for (int level = 0; level < shape.levels; ++level) fitted.emplace_back("v" + std::to_string(level));
        fft::Schema schema{{{"c", "utf8"}}};
        auto kernel = fit_kernel(id, {{"columns", {"c"}}, {"handle_missing", missing}}, schema,
                                 {make_batch({"c"}, {utf8_column(fitted)})});
        for (int shift : {0, 501, 999}) {
          CAPTURE(id, missing, shape.levels, shape.nulls, shape.unknown, shift);
          std::vector<std::optional<std::string>> values;
          for (int i = 0; i < 1000; ++i) {
            // (7 * i + shift) % 1000 visits every residue once, which spreads each kind through the column.
            const int slot = (7 * i + shift) % 1000;
            if (slot < shape.nulls) {
              values.emplace_back(std::nullopt);
            } else if (slot < shape.nulls + shape.unknown) {
              values.emplace_back("zz");
            } else {
              values.emplace_back("v" + std::to_string(i % shape.levels));
            }
          }
          const fft::Batch out = kernel->transform(make_batch({"c"}, {utf8_column(values)}));
          for (std::size_t row = 0; row < values.size(); ++row) {
            const auto& value = values[row];
            const fft::Row record =
                kernel->transform_row({{"c", value ? fft::RowValue(*value) : fft::RowValue(std::monostate{})}});
            REQUIRE(record.size() == out.cols.size());
            for (std::size_t c = 0; c < out.cols.size(); ++c) {
              const auto& doubles = static_cast<const arrow::DoubleArray&>(*out.cols[c]);
              const auto slot = static_cast<int64_t>(row);
              if (std::holds_alternative<std::monostate>(record[c].second)) {
                REQUIRE(doubles.IsNull(slot));
                REQUIRE(std::bit_cast<uint64_t>(doubles.Value(slot)) == 0);
              } else {
                REQUIRE(doubles.IsValid(slot));
                REQUIRE(std::bit_cast<uint64_t>(doubles.Value(slot)) ==
                        std::bit_cast<uint64_t>(std::get<double>(record[c].second)));
              }
            }
          }
        }
      }
    }
  }
}

// REGRESSION: the batch path refuses a passthrough column named like a
// generated one at resolve; the record path used to keep both and let the
// generated value win, so the same record was a hard error through Parquet and
// a silent substitution through transform_dict.
TEST_CASE("the contrast coders refuse a record key that collides with a generated column") {
  fft::Schema schema{{{"c", "utf8"}}};
  auto kernel =
      fit_kernel("helmert_encode", {{"columns", {"c"}}}, schema, {make_batch({"c"}, {utf8_column({"a", "b"})})});
  fft::Schema resolved{{{"c", "utf8"}, {"c_0", "f64"}}};
  REQUIRE_THROWS_WITH(kernel->resolve(resolved), ContainsSubstring("output column 'c_0' already exists in input"));
  REQUIRE_THROWS_WITH(kernel->transform_row({{"c", std::string("a")}, {"c_0", 9.0}}),
                      ContainsSubstring("key 'c_0' already present in record"));
}

// REGRESSION: 'error' is the strict option, so it refuses a null wherever one
// turns up. It used to encode a null at transform as a row of zeros.
TEST_CASE("the contrast coders refuse a null at transform under handle_missing error") {
  for (const char* id : {"backward_difference_encode", "helmert_encode", "sum_encode", "polynomial_encode"}) {
    fft::Schema schema{{{"c", "utf8"}}};
    auto kernel = fit_kernel(id, {{"columns", {"c"}}, {"handle_missing", "error"}}, schema,
                             {make_batch({"c"}, {utf8_column({"a", "b"})})});
    REQUIRE_THROWS_WITH(kernel->transform(make_batch({"c"}, {utf8_column({"a", std::nullopt})})),
                        ContainsSubstring("column 'c' has a null and handle_missing is 'error'"));
    REQUIRE_THROWS_WITH(kernel->transform_row({{"c", fft::RowValue(std::monostate{})}}),
                        ContainsSubstring("column 'c' has a null and handle_missing is 'error'"));
  }
}

// REGRESSION: the encoded columns are erased before the collision check, as
// hash_encode does, so a generated name may reuse an encoded column's. The
// check used to run first, so encoding 'a' beside 'a_0' was a fit-time error
// with no way out but renaming a column.
TEST_CASE("the contrast coders may generate a name an encoded column had") {
  fft::Schema schema{{{"a", "utf8"}, {"a_0", "utf8"}}};
  auto kernel = fit_kernel("sum_encode", {{"columns", {"a", "a_0"}}}, schema,
                           {make_batch({"a", "a_0"}, {utf8_column({"x", "y"}), utf8_column({"p", "q"})})});
  REQUIRE(kernel->outputs == std::vector<std::string>{"a_0", "a_0_0"});
}

TEST_CASE("the contrast coders' load_state rejects a malformed table") {
  auto fresh = [] {
    auto kernel = fft::make_kernel("sum_encode", {{"columns", {"c"}}});
    kernel->stage = "enc";
    return kernel;
  };
  const nlohmann::json good{{"columns", {"c"}}, {"categories", {{"c", {"a", "b", "c"}}}}, {"has_null", {{"c", false}}}};
  REQUIRE_NOTHROW(fresh()->load_state(good));
  auto broken = [&](auto mutate, const std::string& message) {
    nlohmann::json state = good;
    mutate(state);
    REQUIRE_THROWS_WITH(fresh()->load_state(state), ContainsSubstring(message));
  };
  broken([](nlohmann::json& state) { state.erase("categories"); }, "must hold a 'categories' object");
  broken([](nlohmann::json& state) { state["categories"] = "abc"; }, "must hold a 'categories' object");
  broken([](nlohmann::json& state) { state["categories"]["c"] = {"c", "a"}; }, "must be sorted and unique");
  broken([](nlohmann::json& state) { state["categories"]["c"] = {"a", "a"}; }, "must be sorted and unique");
  broken([](nlohmann::json& state) { state["categories"]["c"] = {1, 2}; }, "must hold strings");
  broken([](nlohmann::json& state) { state["categories"]["c"] = nlohmann::json::array(); }, "is empty");
  broken([](nlohmann::json& state) { state["categories"] = {{"d", {"a", "b", "c"}}}; },
         "categories['c'] must be a list");
  // REGRESSION: the state's own ordered column list is what the step reloads,
  // so a list that does not describe the fitted table is a malformed manifest
  // rather than something to reconcile.
  broken([](nlohmann::json& state) { state["categories"].erase("c"); },
         "state 'columns' does not name the same columns as the fitted table");
  broken([](nlohmann::json& state) { state.erase("columns"); }, "state must hold a 'columns' list");
  broken([](nlohmann::json& state) { state["has_null"]["c"] = "yes"; }, "must be true or false");

  // A loaded table rebuilds the matrix, so the codes survive the artifact.
  auto loaded = fresh();
  loaded->load_state(good);
  const fft::Row row = loaded->transform_row({{"c", std::string("c")}});
  REQUIRE(row.size() == 2);
  REQUIRE(std::get<double>(row[0].second) == -1.0);
  REQUIRE(std::get<double>(row[1].second) == -1.0);
}

TEST_CASE("polynomial state preserves the fitted matrix and selected column order verbatim") {
  const nlohmann::json params{{"columns", {"z", "a"}}};
  fft::Schema schema{{{"a", "utf8"}, {"z", "utf8"}}};
  auto fitted =
      fit_kernel("polynomial_encode", params, schema,
                 {make_batch({"a", "z"}, {utf8_column({"x", "y", "z", "x"}), utf8_column({"a", "b", "c", "d"})})});
  nlohmann::json state = fitted->state();
  REQUIRE(state.at("columns") == nlohmann::json({"z", "a"}));
  REQUIRE(state.at("polynomial_basis").at("version") == 1);
  REQUIRE(state.at("polynomial_basis").at("matrices").size() == 2);
  REQUIRE(state.at("polynomial_basis").at("matrices").at("z").size() == 4);
  REQUIRE(state.at("polynomial_basis").at("matrices").at("a").size() == 3);

  // Loading is a replay of stored coefficients, with no normalization or
  // basis reconstruction. A finite edited coefficient makes that observable.
  state["polynomial_basis"]["matrices"]["z"][1][2] = 0.125;
  auto loaded = fft::make_kernel("polynomial_encode", params);
  loaded->stage = "enc";
  loaded->load_state(state);
  REQUIRE(loaded->state() == state);
  fft::Schema loaded_schema{{{"a", "utf8"}, {"z", "utf8"}}};
  loaded->resolve(loaded_schema);
  REQUIRE(loaded->outputs == std::vector<std::string>{"z_0", "z_1", "z_2", "a_0", "a_1"});
  REQUIRE(loaded_schema.cols == schema.cols);

  const auto record = loaded->transform_row({{"a", std::string("y")}, {"z", std::string("b")}});
  const auto batch = loaded->transform(make_batch({"a", "z"}, {utf8_column({"y"}), utf8_column({"b"})}));
  const auto rows = coded(batch, 0);
  REQUIRE(record.size() == 5);
  REQUIRE(rows.size() == 1);
  REQUIRE(std::get<double>(record[2].second) == 0.125);
  for (std::size_t i = 0; i < record.size(); ++i) REQUIRE(std::get<double>(record[i].second) == rows[0][i]);
}

TEST_CASE("polynomial state rejects malformed persisted bases") {
  auto fresh = [] {
    auto kernel = fft::make_kernel("polynomial_encode", {{"columns", {"c"}}});
    kernel->stage = "enc";
    return kernel;
  };
  fft::Schema schema{{{"c", "utf8"}}};
  const nlohmann::json good = fit_kernel("polynomial_encode", {{"columns", {"c"}}}, schema,
                                         {make_batch({"c"}, {utf8_column({"a", "b", "c", "d"})})})
                                  ->state();
  REQUIRE_NOTHROW(fresh()->load_state(good));
  auto broken = [&](auto mutate) {
    nlohmann::json state = good;
    mutate(state);
    REQUIRE_THROWS(fresh()->load_state(state));
  };
  broken([](nlohmann::json& state) { state.erase("polynomial_basis"); });
  broken([](nlohmann::json& state) { state["polynomial_basis"] = nullptr; });
  broken([](nlohmann::json& state) { state["polynomial_basis"] = nlohmann::json::array(); });
  broken([](nlohmann::json& state) { state["polynomial_basis"] = "invalid"; });
  broken([](nlohmann::json& state) { state["polynomial_basis"].erase("version"); });
  for (const auto& invalid : std::vector<nlohmann::json>{2, 0, -1, 1.0, "1", true, nullptr}) {
    INFO("basis version = " << invalid);
    broken([&](nlohmann::json& state) { state["polynomial_basis"]["version"] = invalid; });
  }
  broken([](nlohmann::json& state) { state["polynomial_basis"].erase("matrices"); });
  broken([](nlohmann::json& state) { state["polynomial_basis"]["matrices"] = nlohmann::json::array(); });
  broken([](nlohmann::json& state) { state["polynomial_basis"]["matrices"].erase("c"); });
  broken([](nlohmann::json& state) {
    state["polynomial_basis"]["matrices"]["extra"] = state["polynomial_basis"]["matrices"]["c"];
  });
  broken([](nlohmann::json& state) { state["polynomial_basis"]["matrices"]["c"] = nullptr; });
  broken([](nlohmann::json& state) { state["polynomial_basis"]["matrices"]["c"] = nlohmann::json::object(); });
  broken([](nlohmann::json& state) { state["polynomial_basis"]["matrices"]["c"].erase(0); });
  broken([](nlohmann::json& state) {
    state["polynomial_basis"]["matrices"]["c"].push_back(state["polynomial_basis"]["matrices"]["c"][0]);
  });
  broken([](nlohmann::json& state) { state["polynomial_basis"]["matrices"]["c"][0] = nullptr; });
  broken([](nlohmann::json& state) { state["polynomial_basis"]["matrices"]["c"][0].erase(0); });
  broken([](nlohmann::json& state) { state["polynomial_basis"]["matrices"]["c"][0].push_back(0.0); });
  for (const auto& invalid : std::vector<nlohmann::json>{
           "0.1", true, nullptr, nlohmann::json::array(), std::numeric_limits<double>::quiet_NaN(),
           std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity()}) {
    broken([&](nlohmann::json& state) { state["polynomial_basis"]["matrices"]["c"][0][0] = invalid; });
  }
}

TEST_CASE("polynomial state retains an empty matrix when one level emits no columns") {
  const nlohmann::json params{{"columns", {"c"}}, {"handle_unknown", "indicator"}};
  fft::Schema schema{{{"c", "utf8"}}};
  const auto fitted = fit_kernel("polynomial_encode", params, schema, {make_batch({"c"}, {utf8_column({"a"})})});
  nlohmann::json state = fitted->state();
  REQUIRE(state.at("polynomial_basis").at("matrices").at("c") == nlohmann::json::array());
  auto loaded = fft::make_kernel("polynomial_encode", params);
  loaded->stage = "enc";
  REQUIRE_NOTHROW(loaded->load_state(state));
  REQUIRE(loaded->state() == state);
  REQUIRE(loaded->transform_row({{"c", std::string("unknown")}}).empty());
  std::ostringstream stream;
  loaded->write_state(stream);
  REQUIRE(stream.str() == state.dump());
  state["polynomial_basis"]["matrices"]["c"] = nlohmann::json::array({nlohmann::json::array()});
  REQUIRE_THROWS(loaded->load_state(state));
}

TEST_CASE("polynomial state without a stored basis is refused, not rebuilt") {
  const nlohmann::json state{
      {"columns", {"c"}}, {"categories", {{"c", {"a", "b", "c", "d"}}}}, {"has_null", {{"c", false}}}};
  auto loaded = fft::make_kernel("polynomial_encode", {{"columns", {"c"}}});
  loaded->stage = "enc";
  REQUIRE_THROWS_WITH(loaded->load_state(state), ContainsSubstring("enc: state must hold a 'polynomial_basis' object"));
}

TEST_CASE("the other contrast coders stream exactly their state") {
  for (const char* id : {"backward_difference_encode", "helmert_encode", "sum_encode"}) {
    INFO(id);
    fft::Schema schema{{{"c", "utf8"}}};
    const auto kernel =
        fit_kernel(id, {{"columns", {"c"}}}, schema, {make_batch({"c"}, {utf8_column({"a\"", std::nullopt, "b"})})});
    std::ostringstream stream;
    kernel->write_state(stream);
    REQUIRE(stream.str() == kernel->state().dump());
  }
}

TEST_CASE("polynomial state streams exact JSON with escaped names and empty matrices") {
  const std::string first = "z\"\\\n";
  const std::string second = "a\t雪";
  const std::string empty("m\0key", 5);
  const nlohmann::json params{{"columns", {first, second, empty}}};
  fft::Schema schema{{{second, "utf8"}, {empty, "utf8"}, {first, "utf8"}}};
  auto fitted = fit_kernel("polynomial_encode", params, schema,
                           {make_batch({first, second, empty}, {utf8_column({"quote\"", "slash\\", "line\n", "雪🙂"}),
                                                                utf8_column({"x", "y", std::nullopt, "x"}),
                                                                utf8_column({"same", "same", "same", "same"})})});
  const nlohmann::json state = fitted->state();
  REQUIRE(state.at("polynomial_basis").at("matrices").at(empty) == nlohmann::json::array());
  auto require_exact_stream = [](const fft::Kernel& kernel) {
    std::ostringstream stream;
    stream << std::hex << std::showbase << std::scientific;
    stream.precision(1);
    kernel.write_state(stream);
    REQUIRE(stream.str() == kernel.state().dump());
    REQUIRE(nlohmann::json::parse(stream.str()) == kernel.state());
  };
  require_exact_stream(*fitted);
  auto loaded = fft::make_kernel("polynomial_encode", params);
  loaded->load_state(state);
  require_exact_stream(*loaded);
}
