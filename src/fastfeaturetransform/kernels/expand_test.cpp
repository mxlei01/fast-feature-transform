// Explicit C++ tests for polynomial_features, spline_transform and
// power_transform: the params, the schema rewrite, the accumulator merge in
// both orders, the two transform paths agreeing value for value, and
// load_state refusing a hand-edited manifest.

#include <arrow/array.h>

#include <algorithm>
#include <bit>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <cmath>
#include <limits>
#include <string_view>

#include "fastfeaturetransform/kernel.hpp"
#include "fastfeaturetransform/spill.hpp"
#include "fastfeaturetransform/test_support.hpp"

namespace fft = fastfeaturetransform;
using Catch::Approx;
using Catch::Matchers::ContainsSubstring;
using fft::testing::f64_column;
using fft::testing::make_batch;
using fft::testing::numeric_column;

namespace {

// A resolved kernel over the given schema.
std::unique_ptr<fft::Kernel> resolved(const std::string& id, const nlohmann::json& params, fft::Schema schema) {
  auto kernel = fft::make_kernel(id, params);
  kernel->stage = "expand";
  kernel->resolve(schema);
  return kernel;
}

// Fit a kernel with one worker state per batch.
std::unique_ptr<fft::Kernel> fitted(const std::string& id, const nlohmann::json& params, fft::Schema schema,
                                    const std::vector<fft::Batch>& batches) {
  auto kernel = fft::make_kernel(id, params);
  kernel->stage = "expand";
  kernel->resolve(schema);
  std::vector<std::unique_ptr<fft::AccState>> states;
  for (const fft::Batch& batch : batches) {
    states.push_back(kernel->make_state());
    kernel->consume(*states.back(), batch);
  }
  kernel->finalize(std::move(states));
  return kernel;
}

// One f64 column of a transformed batch, by name.
std::vector<double> column_of(const fft::Batch& batch, const std::string& name) {
  const int idx = batch.find(name);
  REQUIRE(idx >= 0);
  const auto& array = static_cast<const arrow::DoubleArray&>(*batch.cols[static_cast<std::size_t>(idx)]);
  std::vector<double> values;
  for (int64_t row = 0; row < array.length(); ++row) values.push_back(array.Value(row));
  return values;
}

// Every generated value of one row, batch path and record path, compared.
void paths_agree(const fft::Kernel& kernel, const fft::Batch& out, const fft::Row& record, int64_t row) {
  fft::Row got = kernel.transform_row(record);
  REQUIRE(got.size() == out.names.size());
  for (std::size_t i = 0; i < out.names.size(); ++i) {
    REQUIRE(got[i].first == out.names[i]);
    const auto& array = static_cast<const arrow::DoubleArray&>(*out.cols[i]);
    if (array.IsNull(row)) {
      REQUIRE(std::holds_alternative<std::monostate>(got[i].second));
    } else {
      REQUIRE(std::get<double>(got[i].second) == array.Value(row));
    }
  }
}

const fft::Schema kPair{{{"a", "f64"}, {"b", "f64"}}};

std::unique_ptr<fft::Kernel> fitted_with_budget(const std::string& id, const nlohmann::json& params,
                                                const fft::Schema& schema, const std::vector<fft::Batch>& batches,
                                                std::size_t bytes, const std::filesystem::path& temp_dir,
                                                bool expect_spill) {
  fft::FitMemoryOptions options;
  options.memory_limit = bytes;
  options.temp_dir = temp_dir.string();
  auto memory = std::make_shared<fft::FitMemory>(options);
  fft::FitMemoryScope scope(memory);
  auto kernel = resolved(id, params, schema);
  std::vector<std::unique_ptr<fft::AccState>> states;
  for (const fft::Batch& batch : batches) {
    states.push_back(kernel->make_state());
    kernel->consume(*states.back(), batch);
  }
  if (expect_spill) {
    REQUIRE(memory->spill_files_created() > 0);
    bool found_file = false;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(temp_dir)) {
      found_file = found_file || entry.is_regular_file();
    }
    REQUIRE(found_file);
  }
  kernel->finalize(std::move(states));
  REQUIRE(memory->peak_bytes_in_memory() <= bytes);
  REQUIRE(memory->bytes_in_memory() == 0);
  return kernel;
}

void batch_bits_equal(const fft::Batch& left, const fft::Batch& right) {
  REQUIRE(left.names == right.names);
  REQUIRE(left.rows == right.rows);
  REQUIRE(left.cols.size() == right.cols.size());
  for (std::size_t column = 0; column < left.cols.size(); ++column) {
    const auto& expected = static_cast<const arrow::DoubleArray&>(*left.cols[column]);
    const auto& actual = static_cast<const arrow::DoubleArray&>(*right.cols[column]);
    REQUIRE(actual.length() == expected.length());
    for (int64_t row = 0; row < expected.length(); ++row) {
      REQUIRE(actual.IsNull(row) == expected.IsNull(row));
      if (expected.IsValid(row)) {
        REQUIRE(std::bit_cast<uint64_t>(actual.Value(row)) == std::bit_cast<uint64_t>(expected.Value(row)));
      }
    }
  }
}

}  // namespace

TEST_CASE("spline quantile fitting spills exact ranks and releases temporary files") {
  fft::testing::TempDir temp;
  std::vector<std::optional<double>> values;
  for (int i = 0; i < 4099; ++i) {
    if (i % 31 == 0)
      values.push_back(std::nullopt);
    else
      values.push_back(i % 5 == 0 ? -0.0 : static_cast<double>((i * 7919) % 4099) / 64.0);
  }
  const auto source = numeric_column<arrow::DoubleType>(values);
  const fft::Batch whole = make_batch({"x"}, {source});
  const std::vector<fft::Batch> parts{make_batch({"x"}, {source->Slice(2050)}),
                                      make_batch({"x"}, {source->Slice(0, 2050)})};
  const fft::Schema schema{{{"x", "f64"}}};
  for (int knots : {2, 5, 17}) {
    CAPTURE(knots);
    const nlohmann::json params{
        {"columns", {"x"}}, {"n_knots", knots}, {"knots", "quantile"}, {"handle_missing", "zeros"}};
    const auto resident = fitted_with_budget("spline_transform", params, schema, {whole}, 1U << 20, temp.path, false);
    const auto spilled = fitted_with_budget("spline_transform", params, schema, parts, 128, temp.path, true);
    REQUIRE(resident->state().dump() == spilled->state().dump());
    batch_bits_equal(resident->transform(whole), spilled->transform(whole));
    REQUIRE(std::filesystem::is_empty(temp.path));
  }
}

TEST_CASE("power fitting spills raw logs and objective scratch without changing bits") {
  fft::testing::TempDir temp;
  const fft::Schema schema{{{"x", "f64"}}};
  for (const std::string profile : {"box-cox", "positive", "negative", "mixed"}) {
    std::vector<std::optional<double>> values;
    for (int i = 0; i < 257; ++i) {
      if (i % 31 == 0) {
        values.push_back(std::nullopt);
        continue;
      }
      const double magnitude = 0.125 + static_cast<double>((i * 7919) % 257) / 32.0;
      values.push_back(profile == "negative" || (profile == "mixed" && i % 2 == 0) ? -magnitude : magnitude);
    }
    if (profile == "mixed") {
      values[1] = -0.0;
      values[2] = 0.0;
      values[3] = -std::numeric_limits<double>::denorm_min();
      values[4] = std::numeric_limits<double>::denorm_min();
      values[5] = -0x1p-500;
      values[6] = 0x1p-500;
    }
    const auto source = numeric_column<arrow::DoubleType>(values);
    const fft::Batch whole = make_batch({"x"}, {source});
    const std::vector<fft::Batch> parts{make_batch({"x"}, {source->Slice(129)}),
                                        make_batch({"x"}, {source->Slice(0, 129)})};
    for (bool standardize : {false, true}) {
      CAPTURE(profile, standardize);
      const nlohmann::json params{{"columns", {"x"}},
                                  {"method", profile == "box-cox" ? "box-cox" : "yeo-johnson"},
                                  {"standardize", standardize}};
      const auto resident = fitted_with_budget("power_transform", params, schema, {whole}, 1U << 20, temp.path, false);
      const auto spilled = fitted_with_budget("power_transform", params, schema, parts, 128, temp.path, true);
      REQUIRE(resident->state().dump() == spilled->state().dump());
      batch_bits_equal(resident->transform(whole), spilled->transform(whole));
      REQUIRE(std::filesystem::is_empty(temp.path));
    }
  }
}

TEST_CASE("expansion fit spill files are removed after validation failures") {
  fft::testing::TempDir temp;
  const fft::Schema schema{{{"x", "f64"}}};
  const nlohmann::json box{{"columns", {"x"}}, {"method", "box-cox"}};
  std::vector<double> constant(257, 3.0);
  REQUIRE_THROWS_WITH(fitted_with_budget("power_transform", box, schema, {make_batch({"x"}, {f64_column(constant)})},
                                         128, temp.path, true),
                      ContainsSubstring("is constant, which box-cox cannot fit a lambda for"));
  REQUIRE(std::filesystem::is_empty(temp.path));
  constant.push_back(std::numeric_limits<double>::quiet_NaN());
  REQUIRE_THROWS_WITH(fitted_with_budget("power_transform", box, schema, {make_batch({"x"}, {f64_column(constant)})},
                                         128, temp.path, false),
                      ContainsSubstring("is not finite"));
  REQUIRE(std::filesystem::is_empty(temp.path));
  std::vector<std::optional<double>> nullable(257, 1.0);
  nullable.push_back(std::nullopt);
  REQUIRE_THROWS_WITH(
      fitted_with_budget("spline_transform", {{"columns", {"x"}}, {"knots", "quantile"}}, schema,
                         {make_batch({"x"}, {numeric_column<arrow::DoubleType>(nullable)})}, 128, temp.path, true),
      ContainsSubstring("has nulls and handle_missing is 'error'"));
  REQUIRE(std::filesystem::is_empty(temp.path));
}

TEST_CASE("spline sparse outputs preserve record bits at knots and extrapolation boundaries") {
  std::vector<double> knots;
  for (int i = 0; i < 40; ++i) knots.push_back(static_cast<double>(i / 2) / 4.0);
  for (int degree : {0, 1, 3}) {
    for (const char* mode : {"constant", "linear", "continue", "periodic", "error"}) {
      for (bool bias : {false, true}) {
        CAPTURE(degree, mode, bias);
        const nlohmann::json params{{"columns", {"x"}},      {"n_knots", knots.size()}, {"degree", degree},
                                    {"extrapolation", mode}, {"include_bias", bias},    {"handle_missing", "zeros"}};
        auto kernel = fft::make_kernel("spline_transform", params);
        kernel->stage = "expand";
        kernel->load_state({{"knots", {{"x", knots}}}});
        std::vector<std::optional<double>> values{std::nullopt, -0.0, 0.0};
        for (double knot : knots) {
          if (knot > knots.front()) values.push_back(std::nextafter(knot, -std::numeric_limits<double>::infinity()));
          values.push_back(knot);
          if (knot < knots.back()) values.push_back(std::nextafter(knot, std::numeric_limits<double>::infinity()));
        }
        if (std::string_view(mode) != "error") {
          values.push_back(-1.0);
          values.push_back(6.0);
        }
        const auto source = numeric_column<arrow::DoubleType>(values);
        const fft::Batch output = kernel->transform(make_batch({"x"}, {source}));
        for (std::size_t row = 0; row < values.size(); ++row) {
          const fft::Row expected = kernel->transform_row(
              {{"x", values[row] ? fft::RowValue(*values[row]) : fft::RowValue(std::monostate{})}});
          REQUIRE(expected.size() == output.cols.size());
          for (std::size_t c = 0; c < output.cols.size(); ++c) {
            const auto& actual = static_cast<const arrow::DoubleArray&>(*output.cols[c]);
            REQUIRE(actual.null_count() == 0);
            REQUIRE(std::bit_cast<uint64_t>(actual.Value(static_cast<int64_t>(row))) ==
                    std::bit_cast<uint64_t>(std::get<double>(expected[c].second)));
          }
        }
      }
    }
  }
}

TEST_CASE("spline_transform finds a value's knot interval alike in its scan and its search") {
  // At degree 0 each output column is the indicator of one knot interval, so
  // the output shows the interval found: the one that starts at the last knot
  // at or below the value, never at the final knot. Six knots take the scan
  // and forty the binary search, which starts past 32 intervals.
  for (const int count : {6, 40}) {
    CAPTURE(count);
    std::vector<double> knots;
    for (int i = 0; i < count; ++i) knots.push_back(static_cast<double>(i / 2) / 4.0);
    auto kernel = fft::make_kernel("spline_transform", {{"columns", {"x"}}, {"n_knots", count}, {"degree", 0}});
    kernel->stage = "expand";
    kernel->load_state({{"knots", {{"x", knots}}}});
    std::vector<double> values;
    for (const double knot : knots) {
      if (knot > knots.front()) values.push_back(std::nextafter(knot, knots.front()));
      values.push_back(knot);
      if (knot < knots.back()) values.push_back(std::nextafter(knot, knots.back()));
    }
    const fft::Batch output = kernel->transform(make_batch({"x"}, {f64_column(values)}));
    REQUIRE(output.cols.size() == knots.size() - 1);
    for (std::size_t row = 0; row < values.size(); ++row) {
      const auto interval =
          static_cast<std::size_t>(std::upper_bound(knots.begin() + 1, knots.end() - 1, values[row]) - knots.begin()) -
          1;
      for (std::size_t column = 0; column < output.cols.size(); ++column) {
        CAPTURE(values[row], column);
        const auto& actual = static_cast<const arrow::DoubleArray&>(*output.cols[column]);
        REQUIRE(actual.Value(static_cast<int64_t>(row)) == (column == interval ? 1.0 : 0.0));
      }
    }
  }
}

TEST_CASE("Yeo-Johnson cached-log arithmetic retains lambda branches and signed zeros") {
  constexpr double epsilon = std::numeric_limits<double>::epsilon();
  const std::vector<double> values{-0.0, 0.0, -0x1p-500, 0x1p-500, -1.0, 1.0, -0x1p100, 0x1p100};
  for (double lambda : {-1.0, -epsilon / 2, 0.0, epsilon / 2, epsilon, 1.0, std::nextafter(2.0, 0.0), 2.0, 3.0}) {
    CAPTURE(lambda);
    auto kernel = fft::make_kernel("power_transform", {{"columns", {"x"}}, {"standardize", false}});
    kernel->stage = "expand";
    kernel->load_state({{"lambdas", {{"x", lambda}}}, {"means", nullptr}, {"scales", nullptr}});
    const fft::Batch output = kernel->transform(make_batch({"x"}, {f64_column(values)}));
    const auto& actual = static_cast<const arrow::DoubleArray&>(*output.cols[0]);
    for (std::size_t row = 0; row < values.size(); ++row) {
      const double value = values[row];
      double expected = 0.0;
      if (value >= 0.0) {
        expected = std::fabs(lambda) < epsilon ? std::log1p(value) : std::expm1(lambda * std::log1p(value)) / lambda;
      } else {
        expected = std::fabs(lambda - 2.0) > epsilon ? -std::expm1((2.0 - lambda) * std::log1p(-value)) / (2.0 - lambda)
                                                     : -std::log1p(-value);
      }
      REQUIRE(std::bit_cast<uint64_t>(actual.Value(static_cast<int64_t>(row))) == std::bit_cast<uint64_t>(expected));
    }
  }
}

TEST_CASE("spline quantile selection matches sorted percentile knots across partitions") {
  std::vector<double> values;
  for (int i = 0; i < 4099; ++i) values.push_back(i % 5 == 0 ? -0.0 : static_cast<double>((i * 7919) % 4099) / 64.0);
  auto sorted = values;
  for (double& value : sorted)
    if (value == 0.0) value = 0.0;
  std::sort(sorted.begin(), sorted.end());
  const nlohmann::json params{{"columns", {"x"}}, {"n_knots", 5}, {"knots", "quantile"}};
  const fft::Schema schema{{{"x", "f64"}}};
  const auto source = f64_column(values);
  const auto whole = fitted("spline_transform", params, schema, {make_batch({"x"}, {source})});
  const auto split = fitted("spline_transform", params, schema,
                            {make_batch({"x"}, {source->Slice(2000)}), make_batch({"x"}, {source->Slice(0, 2000)})});
  REQUIRE(whole->state().dump() == split->state().dump());
  for (std::size_t i = 0; i < 5; ++i) {
    const double position = static_cast<double>(i) / 4.0 * static_cast<double>(sorted.size() - 1);
    const auto lower = static_cast<std::size_t>(position);
    double expected = sorted[lower];
    if (lower + 1 < sorted.size()) {
      const double weight = position - static_cast<double>(lower);
      const double gap = sorted[lower + 1] - sorted[lower];
      expected = weight >= 0.5 ? sorted[lower + 1] - gap * (1.0 - weight) : sorted[lower] + gap * weight;
    }
    REQUIRE(std::bit_cast<uint64_t>(whole->state().at("knots").at("x").at(i).get<double>()) ==
            std::bit_cast<uint64_t>(expected));
  }
}

TEST_CASE("polynomial suffix reuse keeps multiplication order and null factors") {
  const std::vector<double> probes{-0.0, 0.0, 0x1.0000000000001p0, 0x1.fffffffffffffp-1, 1.0 / 3.0, -7.0, 1e-50, 1e50};
  for (const nlohmann::json degree : {nlohmann::json(4), nlohmann::json::array({3, 4})}) {
    for (bool nullable : {false, true}) {
      auto kernel = resolved("polynomial_features", {{"columns", {"a", "b"}}, {"degree", degree}}, kPair);
      std::vector<std::optional<double>> left;
      std::vector<std::optional<double>> right;
      for (std::size_t row = 0; row < 67; ++row) {
        left.push_back(nullable && row % 11 == 0 ? std::optional<double>{} : std::optional<double>{probes[row % 8]});
        right.push_back(nullable && row % 7 == 0 ? std::optional<double>{}
                                                 : std::optional<double>{probes[(row * 3 + 1) % 8]});
      }
      const fft::Batch output = kernel->transform(
          make_batch({"a", "b"}, {numeric_column<arrow::DoubleType>(left), numeric_column<arrow::DoubleType>(right)}));
      for (std::size_t row = 0; row < left.size(); ++row) {
        const fft::Row expected =
            kernel->transform_row({{"a", left[row] ? fft::RowValue(*left[row]) : fft::RowValue(std::monostate{})},
                                   {"b", right[row] ? fft::RowValue(*right[row]) : fft::RowValue(std::monostate{})}});
        for (std::size_t c = 0; c < expected.size(); ++c) {
          const auto& actual = static_cast<const arrow::DoubleArray&>(*output.cols[c]);
          REQUIRE(actual.IsNull(static_cast<int64_t>(row)) ==
                  std::holds_alternative<std::monostate>(expected[c].second));
          if (actual.IsValid(static_cast<int64_t>(row))) {
            REQUIRE(std::bit_cast<uint64_t>(actual.Value(static_cast<int64_t>(row))) ==
                    std::bit_cast<uint64_t>(std::get<double>(expected[c].second)));
          }
        }
      }
    }
  }
}

TEST_CASE("polynomial_features validates its params") {
  REQUIRE_THROWS_WITH((fft::make_kernel("polynomial_features", nlohmann::json::object())),
                      ContainsSubstring("'columns' must be a non-empty list"));
  REQUIRE_THROWS_WITH((fft::make_kernel("polynomial_features", {{"columns", {"a b"}}})),
                      ContainsSubstring("would make a generated name ambiguous"));
  REQUIRE_THROWS_WITH((fft::make_kernel("polynomial_features", {{"columns", {"a^2"}}})),
                      ContainsSubstring("would make a generated name ambiguous"));
  REQUIRE_THROWS_WITH((fft::make_kernel("polynomial_features", {{"columns", {"a"}}, {"degree", -1}})),
                      ContainsSubstring("'degree' must be an integer from 0 to 2^63 - 1"));
  REQUIRE_THROWS_WITH((fft::make_kernel("polynomial_features", {{"columns", {"a"}}, {"degree", {3, 2}}})),
                      ContainsSubstring("'degree' must be an integer from 0 to 2^63 - 1"));
  REQUIRE_THROWS_WITH((fft::make_kernel("polynomial_features", {{"columns", {"a"}}, {"degree", 1.5}})),
                      ContainsSubstring("'degree' must be an integer from 0 to 2^63 - 1"));
  REQUIRE_THROWS_WITH(
      (fft::make_kernel("polynomial_features", {{"columns", {"a"}}, {"degree", 0}, {"include_bias", false}})),
      ContainsSubstring("degree 0 with include_bias false would generate no columns"));
  REQUIRE_THROWS_WITH((fft::make_kernel("polynomial_features", {{"columns", {"a"}}, {"interaction_only", 1}})),
                      ContainsSubstring("'interaction_only' must be true or false"));
  // Four columns at degree 20 make 10,626 monomials; the cap stops it here.
  REQUIRE_THROWS_WITH((fft::make_kernel("polynomial_features", {{"columns", {"a", "b", "c", "d"}}, {"degree", 20}})),
                      ContainsSubstring("more than the 4096 columns allowed"));
  // Nothing is learned, so no pass ever has to run for it.
  REQUIRE_FALSE(fft::make_kernel("polynomial_features", {{"columns", {"a"}}})->needs_fit());
  REQUIRE(fft::make_kernel("polynomial_features", {{"columns", {"a"}}})->state().empty());
  REQUIRE(fft::make_kernel("polynomial_features", {{"columns", {"a"}}})->params().at("degree") == 2);
}

TEST_CASE("polynomial_features declares its outputs without fitting") {
  auto kernel = fft::make_kernel("polynomial_features", {{"columns", {"a", "b"}}});
  kernel->stage = "expand";
  fft::Schema schema{{{"a", "f64"}, {"keep", "i64"}, {"b", "i32"}}};
  kernel->resolve(schema);
  // The source columns leave and the monomials arrive in sklearn's order.
  REQUIRE(
      schema.cols ==
      fft::Schema{
          {{"keep", "i64"}, {"1", "f64"}, {"a", "f64"}, {"b", "f64"}, {"a^2", "f64"}, {"a*b", "f64"}, {"b^2", "f64"}}}
          .cols);
  REQUIRE(kernel->outputs == std::vector<std::string>{"1", "a", "b", "a^2", "a*b", "b^2"});

  auto text = fft::make_kernel("polynomial_features", {{"columns", {"a"}}});
  text->stage = "expand";
  fft::Schema strings{{{"a", "utf8"}}};
  REQUIRE_THROWS_WITH(text->resolve(strings),
                      ContainsSubstring("is utf8, expected an integer or float column (cast it first)"));

  auto missing = fft::make_kernel("polynomial_features", {{"columns", {"gone"}}});
  missing->stage = "expand";
  fft::Schema absent{{{"a", "f64"}}};
  REQUIRE_THROWS_WITH(missing->resolve(absent), ContainsSubstring("column 'gone' not found in input"));

  // A surviving column named like a monomial would be silently dropped.
  auto collides = fft::make_kernel("polynomial_features", {{"columns", {"a"}}});
  collides->stage = "expand";
  fft::Schema taken{{{"a", "f64"}, {"a^2", "f64"}}};
  REQUIRE_THROWS_WITH(collides->resolve(taken), ContainsSubstring("output column 'a^2' already exists in input"));
}

TEST_CASE("polynomial_features multiplies in sklearn's order and both paths agree") {
  auto kernel = resolved("polynomial_features", {{"columns", {"a", "b"}}}, kPair);
  fft::Batch out = kernel->transform(make_batch({"a", "b"}, {f64_column({2.0, 3.0}), f64_column({5.0, 7.0})}));
  REQUIRE(out.names == std::vector<std::string>{"1", "a", "b", "a^2", "a*b", "b^2"});
  REQUIRE(column_of(out, "1") == std::vector<double>{1.0, 1.0});
  REQUIRE(column_of(out, "a^2") == std::vector<double>{4.0, 9.0});
  REQUIRE(column_of(out, "a*b") == std::vector<double>{10.0, 21.0});
  REQUIRE(column_of(out, "b^2") == std::vector<double>{25.0, 49.0});
  paths_agree(*kernel, out, {{"a", 2.0}, {"b", 5.0}}, 0);
  paths_agree(*kernel, out, {{"a", 3.0}, {"b", 7.0}}, 1);

  // A null factor makes its monomials null; the bias uses no factor at all.
  fft::Batch nulls = kernel->transform(make_batch(
      {"a", "b"}, {numeric_column<arrow::DoubleType>({std::nullopt}), numeric_column<arrow::DoubleType>({4.0})}));
  REQUIRE(static_cast<const arrow::DoubleArray&>(*nulls.cols[0]).Value(0) == 1.0);   // the bias
  REQUIRE(static_cast<const arrow::DoubleArray&>(*nulls.cols[1]).IsNull(0));         // a
  REQUIRE(static_cast<const arrow::DoubleArray&>(*nulls.cols[2]).Value(0) == 4.0);   // b
  REQUIRE(static_cast<const arrow::DoubleArray&>(*nulls.cols[4]).IsNull(0));         // a*b
  REQUIRE(static_cast<const arrow::DoubleArray&>(*nulls.cols[5]).Value(0) == 16.0);  // b^2
  paths_agree(*kernel, nulls, {{"a", std::monostate{}}, {"b", 4.0}}, 0);

  REQUIRE_THROWS_WITH(kernel->transform_row({{"a", std::string("x")}, {"b", 1.0}}),
                      ContainsSubstring("must be a number or null"));
  REQUIRE_THROWS_WITH(kernel->transform_row({{"a", 1.0}}), ContainsSubstring("key 'b' missing from record"));
  // A record key named like a generated column would be dropped silently.
  REQUIRE_THROWS_WITH(kernel->transform_row({{"a", 1.0}, {"b", 2.0}, {"a*b", 9.0}}),
                      ContainsSubstring("key 'a*b' already present in record"));
}

TEST_CASE("polynomial_features honours interaction_only and a degree range") {
  auto interactions =
      resolved("polynomial_features", {{"columns", {"a", "b"}}, {"degree", 2}, {"interaction_only", true}}, kPair);
  REQUIRE(interactions->outputs == std::vector<std::string>{"1", "a", "b", "a*b"});
  auto ranged =
      resolved("polynomial_features", {{"columns", {"a", "b"}}, {"degree", {2, 3}}, {"include_bias", false}}, kPair);
  REQUIRE(ranged->outputs == std::vector<std::string>{"a^2", "a*b", "b^2", "a^3", "a^2*b", "a*b^2", "b^3"});
  fft::Batch out = ranged->transform(make_batch({"a", "b"}, {f64_column({2.0}), f64_column({5.0})}));
  REQUIRE(column_of(out, "a^2*b") == std::vector<double>{20.0});
  REQUIRE(column_of(out, "a*b^2") == std::vector<double>{50.0});
}

TEST_CASE("polynomial_features load_state rejects a table a hand-edited manifest put there") {
  auto fresh = [] {
    auto kernel = fft::make_kernel("polynomial_features", {{"columns", {"a"}}});
    kernel->stage = "expand";
    return kernel;
  };
  REQUIRE_NOTHROW(fresh()->load_state(nlohmann::json::object()));
  REQUIRE_THROWS_WITH(fresh()->load_state({{"powers", {1, 2}}}),
                      ContainsSubstring("polynomial_features learns nothing, so its 'state' must be an empty object"));
  REQUIRE_THROWS_WITH(fresh()->load_state(nlohmann::json::array()), ContainsSubstring("must be an empty object"));
}

TEST_CASE("spline_transform validates its params") {
  REQUIRE_THROWS_WITH((fft::make_kernel("spline_transform", {{"columns", {"a"}}, {"n_knots", 1}})),
                      ContainsSubstring("'n_knots' must be an integer between 2 and 4096"));
  REQUIRE_THROWS_WITH((fft::make_kernel("spline_transform", {{"columns", {"a"}}, {"degree", -1}})),
                      ContainsSubstring("'degree' must be an integer between 0 and 4096"));
  REQUIRE_THROWS_WITH((fft::make_kernel("spline_transform", {{"columns", {"a"}}, {"knots", "median"}})),
                      ContainsSubstring("'knots' must be 'uniform' or 'quantile'"));
  REQUIRE_THROWS_WITH((fft::make_kernel("spline_transform", {{"columns", {"a"}}, {"extrapolation", "wrap"}})),
                      ContainsSubstring("'extrapolation' must be 'error', 'constant', 'linear'"));
  REQUIRE_THROWS_WITH((fft::make_kernel("spline_transform", {{"columns", {"a"}}, {"handle_missing", "value"}})),
                      ContainsSubstring("'handle_missing' must be 'error' or 'zeros'"));
  REQUIRE_THROWS_WITH(
      (fft::make_kernel("spline_transform",
                        {{"columns", {"a"}}, {"n_knots", 3}, {"degree", 3}, {"extrapolation", "periodic"}})),
      ContainsSubstring("'periodic' needs degree below n_knots"));
  REQUIRE_THROWS_WITH((fft::make_kernel("spline_transform", {{"columns", {"a", "b"}}, {"n_knots", 3000}})),
                      ContainsSubstring("more than the 4096 columns allowed"));
  REQUIRE_THROWS_WITH((fft::make_kernel("spline_transform", {{"columns", {"a"}}, {"n_knots", 5000}})),
                      ContainsSubstring("'n_knots' must be an integer between 2 and 4096"));
  REQUIRE(fft::make_kernel("spline_transform", {{"columns", {"a"}}})->needs_fit());
  REQUIRE(fft::make_kernel("spline_transform", {{"columns", {"a"}}})->params().at("extrapolation") == "constant");
}

TEST_CASE("spline_transform declares its width before the knots are fitted") {
  auto kernel = fft::make_kernel("spline_transform", {{"columns", {"a"}}, {"n_knots", 3}, {"degree", 1}});
  kernel->stage = "expand";
  fft::Schema schema{{{"a", "f64"}, {"keep", "i64"}}};
  kernel->resolve(schema);
  REQUIRE(schema.cols == fft::Schema{{{"keep", "i64"}, {"a_sp_0", "f64"}, {"a_sp_1", "f64"}, {"a_sp_2", "f64"}}}.cols);

  auto no_bias = fft::make_kernel("spline_transform",
                                  {{"columns", {"a"}}, {"n_knots", 3}, {"degree", 1}, {"include_bias", false}});
  no_bias->stage = "expand";
  REQUIRE(no_bias->outputs == std::vector<std::string>{"a_sp_0", "a_sp_1"});

  auto text = fft::make_kernel("spline_transform", {{"columns", {"a"}}});
  text->stage = "expand";
  fft::Schema strings{{{"a", "utf8"}}};
  REQUIRE_THROWS_WITH(text->resolve(strings),
                      ContainsSubstring("is utf8, expected an integer or float column (cast it first)"));
}

TEST_CASE("spline_transform merges min and max in either order and both paths agree") {
  // The knots of [0, 1, 2, 3, 4, 5] at n_knots=3 are 0, 2.5 and 5, whatever
  // the split, and at degree 1 the basis is the hat functions over them.
  const nlohmann::json params{{"columns", {"a"}}, {"n_knots", 3}, {"degree", 1}};
  const fft::Schema schema{{{"a", "f64"}}};
  const fft::Batch left = make_batch({"a"}, {f64_column({0.0, 1.0, 2.0})});
  const fft::Batch right = make_batch({"a"}, {f64_column({3.0, 4.0, 5.0})});
  auto kernel = fitted("spline_transform", params, schema, {left, right});
  const nlohmann::json state = kernel->state();
  REQUIRE(state.at("knots").at("a") == nlohmann::json({0.0, 2.5, 5.0}));
  // Only the merged min and max may decide the knots: this is the monoid contract.
  REQUIRE(fitted("spline_transform", params, schema, {right, left})->state() == state);

  // extrapolation defaults to 'constant', which holds the boundary values.
  fft::Batch out = kernel->transform(make_batch({"a"}, {f64_column({-1.0, 7.0, 2.0})}));
  REQUIRE(out.names == std::vector<std::string>{"a_sp_0", "a_sp_1", "a_sp_2"});
  // Below the range every basis but the first is 0, above it every one but
  // the last, and 2.0 sits four fifths of the way from 0 to the knot at 2.5.
  const std::vector<std::vector<double>> want{{1.0, 0.0, 0.2}, {0.0, 0.0, 0.8}, {0.0, 1.0, 0.0}};
  for (std::size_t j = 0; j < 3; ++j) {
    const std::vector<double> got = column_of(out, "a_sp_" + std::to_string(j));
    for (std::size_t row = 0; row < 3; ++row) REQUIRE(got[row] == Approx(want[j][row]).margin(1e-15));
  }
  paths_agree(*kernel, out, {{"a", -1.0}}, 0);
  paths_agree(*kernel, out, {{"a", 7.0}}, 1);
  paths_agree(*kernel, out, {{"a", 2.0}}, 2);
  REQUIRE_THROWS_WITH(kernel->transform_row({{"a", std::string("x")}}), ContainsSubstring("must be a number or null"));
  REQUIRE_THROWS_WITH(kernel->transform_row({{"b", 1.0}}), ContainsSubstring("key 'a' missing from record"));
  REQUIRE_THROWS_WITH(kernel->transform_row({{"a", 1.0}, {"a_sp_1", 2.0}}),
                      ContainsSubstring("key 'a_sp_1' already present in record"));
}

TEST_CASE("spline_transform buffers for quantile knots and merges to the same table") {
  const nlohmann::json params{{"columns", {"a"}}, {"n_knots", 3}, {"degree", 1}, {"knots", "quantile"}};
  const fft::Schema schema{{{"a", "f64"}}};
  const fft::Batch left = make_batch({"a"}, {f64_column({0.0, 1.0, 8.0})});
  const fft::Batch right = make_batch({"a"}, {f64_column({3.0, 4.0, 5.0})});
  auto kernel = fitted("spline_transform", params, schema, {left, right});
  // The 0th, 50th and 100th percentiles of [0, 1, 3, 4, 5, 8] are 0, 3.5, 8.
  REQUIRE(kernel->state().at("knots").at("a") == nlohmann::json({0.0, 3.5, 8.0}));
  REQUIRE(fitted("spline_transform", params, schema, {right, left})->state() == kernel->state());
}

TEST_CASE("spline_transform refuses what it cannot fit or extrapolate") {
  const fft::Schema schema{{{"a", "f64"}}};
  REQUIRE_THROWS_WITH(fitted("spline_transform", {{"columns", {"a"}}, {"n_knots", 3}}, schema,
                             {make_batch({"a"}, {f64_column({1.0})})}),
                      ContainsSubstring("fewer than two non-null values"));
  REQUIRE_THROWS_WITH(fitted("spline_transform", {{"columns", {"a"}}, {"n_knots", 3}}, schema,
                             {make_batch({"a"}, {numeric_column<arrow::DoubleType>({1.0, 2.0, std::nullopt})})}),
                      ContainsSubstring("has nulls and handle_missing is 'error'"));
  // Over three subnormal units, linspace's step rounds up past the maximum and
  // the last knot falls below the one before. scipy's BSpline refuses such
  // knots, and so does the fit, rather than save a table load_state refuses.
  for (const int degree : {0, 3}) {
    REQUIRE_THROWS_WITH(fitted("spline_transform", {{"columns", {"a"}}, {"n_knots", 6}, {"degree", degree}}, schema,
                               {make_batch({"a"}, {f64_column({5e-324, 2e-323})})}),
                        ContainsSubstring("column 'a' spans too little to place 6 knots in order"));
  }
  // REGRESSION: uniform knots under about 5.6e-309 apart are refused. de
  // Boor's divisions by their gaps overflow, so the fit used to save knots
  // with which every row, the fitted ones included, transformed to inf or NaN.
  REQUIRE_THROWS_WITH(fitted("spline_transform", {{"columns", {"a"}}}, schema,
                             {make_batch({"a"}, {f64_column({0.0, 1e-310, 2e-310})})}),
                      ContainsSubstring("column 'a' spans too little to place 5 knots at least about 5.6e-309 apart"));
  // Quantile knots with one such gap among wide ones still fit, as in sklearn:
  // only the rows beside that gap fail, at transform.
  auto quantiles = fitted("spline_transform", {{"columns", {"a"}}, {"knots", "quantile"}}, schema,
                          {make_batch({"a"}, {f64_column({0.0, 1e-310, 1.0, 2.0, 3.0})})});
  REQUIRE(std::isfinite(column_of(quantiles->transform(make_batch({"a"}, {f64_column({2.5})})), "a_sp_0")[0]));
  // A degree-0 basis divides by nothing, so those knots serve it, as in
  // sklearn: each value lands in the indicator of its knot interval.
  const std::vector<double> values{0.0, 1e-310, 5e-311, 2e-310};
  auto indicator = fitted("spline_transform", {{"columns", {"a"}}, {"n_knots", 3}, {"degree", 0}}, schema,
                          {make_batch({"a"}, {f64_column(values)})});
  const fft::Batch basis = indicator->transform(make_batch({"a"}, {f64_column(values)}));
  REQUIRE(column_of(basis, "a_sp_0") == std::vector<double>{1.0, 0.0, 1.0, 0.0});
  REQUIRE(column_of(basis, "a_sp_1") == std::vector<double>{0.0, 1.0, 0.0, 1.0});

  auto zeros = fitted("spline_transform", {{"columns", {"a"}}, {"n_knots", 3}, {"handle_missing", "zeros"}}, schema,
                      {make_batch({"a"}, {numeric_column<arrow::DoubleType>({1.0, 2.0, std::nullopt})})});
  fft::Batch out = zeros->transform(make_batch({"a"}, {numeric_column<arrow::DoubleType>({std::nullopt})}));
  for (const std::shared_ptr<arrow::Array>& column : out.cols) {
    REQUIRE_FALSE(column->IsNull(0));
    REQUIRE(static_cast<const arrow::DoubleArray&>(*column).Value(0) == 0.0);
  }
  paths_agree(*zeros, out, {{"a", std::monostate{}}}, 0);

  auto strict = fitted("spline_transform", {{"columns", {"a"}}, {"n_knots", 3}, {"extrapolation", "error"}}, schema,
                       {make_batch({"a"}, {f64_column({0.0, 5.0})})});
  REQUIRE_NOTHROW(strict->transform(make_batch({"a"}, {f64_column({0.0, 5.0})})));
  REQUIRE_THROWS_WITH(strict->transform(make_batch({"a"}, {f64_column({6.0})})),
                      ContainsSubstring("beyond the limits of the knots and extrapolation is 'error'"));
  REQUIRE_THROWS_WITH(strict->transform_row({{"a", -1.0}}),
                      ContainsSubstring("beyond the limits of the knots and extrapolation is 'error'"));
}

TEST_CASE("spline_transform load_state rejects a malformed table") {
  auto fresh = [] {
    auto kernel = fft::make_kernel("spline_transform", {{"columns", {"a"}}, {"n_knots", 3}, {"degree", 1}});
    kernel->stage = "expand";
    return kernel;
  };
  const nlohmann::json good{{"knots", {{"a", {0.0, 2.5, 5.0}}}}};
  REQUIRE_NOTHROW(fresh()->load_state(good));
  auto broken = [&](auto mutate, const std::string& message) {
    nlohmann::json state = good;
    mutate(state);
    REQUIRE_THROWS_WITH(fresh()->load_state(state), ContainsSubstring(message));
  };
  broken([](nlohmann::json& state) { state.erase("knots"); }, "state must hold a 'knots' object");
  broken([](nlohmann::json& state) { state["knots"] = nlohmann::json::array(); }, "state must hold a 'knots' object");
  broken([](nlohmann::json& state) { state["knots"]["a"] = {0.0, 5.0}; }, "must be a list of 3 knots");
  broken([](nlohmann::json& state) { state["knots"] = {{"b", {0.0, 2.5, 5.0}}}; }, "must be a list of 3 knots");
  broken([](nlohmann::json& state) { state["knots"]["a"] = {0.0, "two", 5.0}; }, "must hold finite numbers");
  broken([](nlohmann::json& state) { state["knots"]["a"] = {0.0, 5.0, 2.5}; }, "must be sorted");
}

TEST_CASE("power_transform validates its params") {
  REQUIRE_THROWS_WITH((fft::make_kernel("power_transform", nlohmann::json::object())),
                      ContainsSubstring("'columns' must be a non-empty list"));
  REQUIRE_THROWS_WITH((fft::make_kernel("power_transform", {{"columns", {"a"}}, {"method", "log"}})),
                      ContainsSubstring("'method' must be 'yeo-johnson' or 'box-cox'"));
  REQUIRE_THROWS_WITH((fft::make_kernel("power_transform", {{"columns", {"a"}}, {"standardize", 1}})),
                      ContainsSubstring("'standardize' must be true or false"));
  REQUIRE(fft::make_kernel("power_transform", {{"columns", {"a"}}})->needs_fit());
  REQUIRE(fft::make_kernel("power_transform", {{"columns", {"a"}}})->params().at("method") == "yeo-johnson");
}

TEST_CASE("power_transform rewrites the column in place") {
  auto kernel = fft::make_kernel("power_transform", {{"columns", {"a"}}});
  kernel->stage = "expand";
  fft::Schema schema{{{"a", "i64"}, {"keep", "utf8"}}};
  kernel->resolve(schema);
  REQUIRE(schema.cols == fft::Schema{{{"a", "f64"}, {"keep", "utf8"}}}.cols);

  auto text = fft::make_kernel("power_transform", {{"columns", {"keep"}}});
  text->stage = "expand";
  fft::Schema strings{{{"keep", "utf8"}}};
  REQUIRE_THROWS_WITH(text->resolve(strings),
                      ContainsSubstring("is utf8, expected an integer or float column (cast it first)"));
}

TEST_CASE("power_transform solves the same lambda however the rows were split") {
  const nlohmann::json params{{"columns", {"a"}}, {"method", "box-cox"}, {"standardize", false}};
  const fft::Schema schema{{{"a", "f64"}}};
  const fft::Batch left = make_batch({"a"}, {f64_column({1.0, 2.0})});
  const fft::Batch right = make_batch({"a"}, {f64_column({4.0, 8.0})});
  auto kernel = fitted("power_transform", params, schema, {left, right});
  // A geometric column is symmetric in log space, so the likelihood is maximal
  // and flat at lambda 0 and Brent stops wherever its tolerance lets it: ours
  // a few 1e-9 above, scipy's a few 1e-6 below. That is exactly the deviation
  // the parity fixture carries a tolerance for.
  REQUIRE(kernel->state().at("lambdas").at("a").get<double>() == Approx(0.0).margin(1e-5));
  REQUIRE(kernel->state().at("means").is_null());
  // Only the merged buffer may decide the lambda: exact sums, so the split
  // cannot move it by a single bit.
  REQUIRE(fitted("power_transform", params, schema, {right, left})->state() == kernel->state());

  fft::Batch out = kernel->transform(make_batch({"a"}, {numeric_column<arrow::DoubleType>({1.0, 8.0, std::nullopt})}));
  REQUIRE(column_of(out, "a")[0] == Approx(0.0).margin(1e-9));
  REQUIRE(column_of(out, "a")[1] == Approx(2.0794415417).margin(1e-4));  // log(8), the lambda-0 limit
  REQUIRE(static_cast<const arrow::DoubleArray&>(*out.cols[0]).IsNull(2));
  paths_agree(*kernel, out, {{"a", 1.0}}, 0);
  paths_agree(*kernel, out, {{"a", 8.0}}, 1);
  paths_agree(*kernel, out, {{"a", std::monostate{}}}, 2);
}

TEST_CASE("power_transform standardizes with the mean and scale of the transformed column") {
  const fft::Schema schema{{{"a", "f64"}}};
  auto kernel = fitted("power_transform", {{"columns", {"a"}}, {"method", "box-cox"}}, schema,
                       {make_batch({"a"}, {f64_column({1.0, 2.0, 4.0, 8.0})})});
  // The lambda is 0 to within the solver's tolerance, so the transformed
  // column is log2(x) * log(2): mean log(8)/2 and standard deviation
  // sqrt(5/4) * log(2).
  const nlohmann::json state = kernel->state();
  REQUIRE(state.at("means").at("a").get<double>() == Approx(1.0397207708).margin(1e-4));
  REQUIRE(state.at("scales").at("a").get<double>() == Approx(0.7749620027).margin(1e-4));
  fft::Batch out = kernel->transform(make_batch({"a"}, {f64_column({1.0, 8.0})}));
  REQUIRE(column_of(out, "a")[0] == Approx(-1.3416407865).margin(1e-4));
  REQUIRE(column_of(out, "a")[1] == Approx(1.3416407865).margin(1e-4));
}

TEST_CASE("power_transform standardizes a constant yeo-johnson column by its raw values") {
  // REGRESSION: sklearn leaves a column its constant check catches
  // untransformed when it fits the standardizer, and so does this fit: the
  // mean is the raw value and the scale 1. The fit used to take both from
  // expm1(log1p(x)), which is not exactly x: at 1.7e15 the mean moved by
  // 5.75, and on a nearly constant column its rounding noise was divided by a
  // scale of 1.8e-6, so values a few ulps apart came out near -0.75 and 1.29
  // where sklearn's are within 2.1e-6 of 0.
  const fft::Schema schema{{{"a", "f64"}}};
  auto constant = fitted("power_transform", {{"columns", {"a"}}}, schema,
                         {make_batch({"a"}, {f64_column({1.7e15, 1.7e15, 1.7e15, 1.7e15, 1.7e15})})});
  REQUIRE(constant->state().at("lambdas").at("a") == 1.0);
  REQUIRE(constant->state().at("means").at("a") == 1.7e15);
  REQUIRE(constant->state().at("scales").at("a") == 1.0);
  std::vector<double> near;
  for (int i = 0; i < 5; ++i) near.push_back(1e9 + 4.0 * i * std::ldexp(1.0, -23));  // 4 ulps of 1e9 apart
  auto nearly = fitted("power_transform", {{"columns", {"a"}}}, schema, {make_batch({"a"}, {f64_column(near)})});
  REQUIRE(nearly->state().at("scales").at("a") == 1.0);
  for (const double value : column_of(nearly->transform(make_batch({"a"}, {f64_column(near)})), "a")) {
    REQUIRE(std::fabs(value) < 1e-5);
  }
}

TEST_CASE("power_transform keeps lambda 1 on a constant column with nulls") {
  // A null teaches nothing, so these values are a constant column and keep
  // Yeo-Johnson's identity lambda. sklearn 1.9.0 checks for a constant column
  // before it drops the NaN, so its check misses it and it fits 17.58.
  const fft::Schema schema{{{"a", "f64"}}};
  auto kernel = fitted("power_transform", {{"columns", {"a"}}, {"standardize", false}}, schema,
                       {make_batch({"a"}, {numeric_column<arrow::DoubleType>({5.0, 5.0, std::nullopt, 5.0})})});
  REQUIRE(kernel->state().at("lambdas").at("a") == 1.0);
  fft::Batch out = kernel->transform(make_batch({"a"}, {f64_column({5.0, 7.0})}));
  REQUIRE(column_of(out, "a")[0] == Approx(5.0).epsilon(1e-15));  // expm1(log1p(x)), x up to rounding
  REQUIRE(column_of(out, "a")[1] == Approx(7.0).epsilon(1e-15));
}

// scipy caps a box-cox lambda so that the column's most extreme value
// transforms to at most f64's largest over 10,000. Uncapped, Brent finds
// about 33.7 for 'high', whose largest values then transform to infinity, and
// about -35.5 for 'low'.
TEST_CASE("power_transform caps a box-cox lambda as scipy does") {
  const fft::Schema schema{{{"high", "f64"}, {"low", "f64"}}};
  const fft::Batch batch = make_batch(
      {"high", "low"}, {f64_column({9.0e29, 9.6e29, 9.8e29, 9.9e29, 9.95e29, 1.0e30, 1.0e30, 1.0e30}),
                        f64_column({1.0e-30, 1.0e-30, 1.0e-30, 1.005e-30, 1.01e-30, 1.02e-30, 1.04e-30, 1.1e-30})});
  auto kernel = fitted("power_transform", {{"columns", {"high", "low"}}, {"method", "box-cox"}, {"standardize", false}},
                       schema, {batch});
  // scipy 1.18.1's lambdas: a value above 1 caps a positive lambda, one below
  // 1 a negative one.
  const nlohmann::json lambdas = kernel->state().at("lambdas");
  REQUIRE(lambdas.at("high").get<double>() == Approx(10.175408914365082).epsilon(1e-14));
  REQUIRE(lambdas.at("low").get<double>() == Approx(-10.175408914365082).epsilon(1e-14));
  constexpr double kLimit = 1.7976931348623157e308 / 10000.0;
  fft::Batch out = kernel->transform(batch);
  REQUIRE(column_of(out, "high")[7] == Approx(kLimit).epsilon(1e-12));
  REQUIRE(column_of(out, "low")[0] == Approx(-kLimit).epsilon(1e-12));
  paths_agree(*kernel, out, {{"high", 9.0e29}, {"low", 1.0e-30}}, 0);
  paths_agree(*kernel, out, {{"high", 1.0e30}, {"low", 1.1e-30}}, 7);
}

// Near 1, a capped lambda passes 10,000, and the power under the extreme value
// overflows f64 although the transformed value is only about 1.8e304, so the
// capped column is finite only if box-cox goes through logarithms there, as
// scipy's boxcox does. Nearer still, Brent's lambda of about 601,663 leaves the
// value at 7.8e302: scipy does not cap it, and a cap check whose power
// overflows must not cap it either.
TEST_CASE("power_transform keeps a box-cox column finite when lambda passes 10,000") {
  const fft::Schema schema{{{"high", "f64"}, {"low", "f64"}}};
  const fft::Batch batch =
      make_batch({"high", "low"}, {f64_column({1.0499, 1.04994, 1.04996, 1.04998, 1.04999, 1.05, 1.05, 1.05}),
                                   f64_column({0.95, 0.95, 0.95, 0.950005, 0.95001, 0.95002, 0.95004, 0.9501})});
  auto kernel = fitted("power_transform", {{"columns", {"high", "low"}}, {"method", "box-cox"}, {"standardize", false}},
                       schema, {batch});
  // scipy 1.18.1's capped lambdas.
  const nlohmann::json lambdas = kernel->state().at("lambdas");
  REQUIRE(lambdas.at("high").get<double>() == Approx(14555.353497031323).epsilon(1e-14));
  REQUIRE(lambdas.at("low").get<double>() == Approx(-13844.07052225851).epsilon(1e-14));
  constexpr double kLimit = 1.7976931348623157e308 / 10000.0;
  fft::Batch out = kernel->transform(batch);
  REQUIRE(column_of(out, "high")[7] == Approx(kLimit).epsilon(1e-12));
  REQUIRE(column_of(out, "low")[0] == Approx(-kLimit).epsilon(1e-12));
  paths_agree(*kernel, out, {{"high", 1.05}, {"low", 0.9501}}, 7);

  const fft::Schema one{{{"a", "f64"}}};
  auto window = fitted("power_transform", {{"columns", {"a"}}, {"method", "box-cox"}, {"standardize", false}}, one,
                       {make_batch({"a"}, {f64_column({1.001182, 1.001182, 1.001182, 1.001182, 1.001180998818,
                                                       1.001180998818, 1.001179997636, 1.00117699409})})});
  // scikit-learn keeps 601663.1835375819; capped, it would be 604319.84.
  REQUIRE(window->state().at("lambdas").at("a").get<double>() == Approx(601663.1835375819).epsilon(1e-6));
  REQUIRE(std::isfinite(column_of(window->transform(make_batch({"a"}, {f64_column({1.001182})})), "a")[0]));
}

// scipy's boxcox gives log(x) when |lambda| < 1e-19, and once lambda * log(x)
// reaches 709.78 it computes x^lambda / lambda through logarithms, so that a
// value whose power overflows f64 stays finite while the quotient fits.
TEST_CASE("power_transform evaluates box-cox as scipy's boxcox does") {
  auto loaded = [](double lambda) {
    auto kernel =
        fft::make_kernel("power_transform", {{"columns", {"a"}}, {"method", "box-cox"}, {"standardize", false}});
    kernel->stage = "expand";
    kernel->load_state({{"lambdas", {{"a", lambda}}}, {"means", nullptr}, {"scales", nullptr}});
    return kernel;
  };
  auto value = [](const fft::Kernel& kernel, double x) {
    return std::get<double>(kernel.transform_row({{"a", x}})[0].second);
  };
  // expm1(1e-20 * log(1.5)) / 1e-20 is an ulp away from log(1.5).
  REQUIRE(value(*loaded(1e-20), 1.5) == std::log(1.5));
  // scipy 1.18.1's values at the lambda capping 1e30 (and 1e-30): past the
  // fitted range the power overflows but the value does not, until it does.
  const auto high = loaded(10.175408914365082);
  const auto low = loaded(-10.175408914365082);
  REQUIRE(value(*high, 2.0e30) == Approx(2.0788294528972172e307).epsilon(1e-12));
  REQUIRE(value(*low, 5.0e-31) == Approx(-2.0788294528972172e307).epsilon(1e-12));
  REQUIRE(value(*high, 1.0e31) == std::numeric_limits<double>::infinity());
  REQUIRE(value(*low, 1.0e-31) == -std::numeric_limits<double>::infinity());
}

TEST_CASE("power_transform refuses what neither method can fit") {
  const fft::Schema schema{{{"a", "f64"}}};
  REQUIRE_THROWS_WITH(fitted("power_transform", {{"columns", {"a"}}, {"method", "box-cox"}}, schema,
                             {make_batch({"a"}, {f64_column({1.0, -2.0})})}),
                      ContainsSubstring("is not strictly positive, which box-cox requires"));
  REQUIRE_THROWS_WITH(fitted("power_transform", {{"columns", {"a"}}, {"method", "box-cox"}}, schema,
                             {make_batch({"a"}, {f64_column({3.0, 3.0})})}),
                      ContainsSubstring("is constant, which box-cox cannot fit a lambda for"));
  REQUIRE_THROWS_WITH(fitted("power_transform", {{"columns", {"a"}}}, schema,
                             {make_batch({"a"}, {numeric_column<arrow::DoubleType>({1.0, std::nullopt})})}),
                      ContainsSubstring("fewer than two non-null values"));
  // A constant column keeps lambda 1 under yeo-johnson, the identity.
  auto flat = fitted("power_transform", {{"columns", {"a"}}, {"standardize", false}}, schema,
                     {make_batch({"a"}, {f64_column({3.0, 3.0, 3.0})})});
  REQUIRE(flat->state().at("lambdas").at("a").get<double>() == 1.0);
  // A value box-cox cannot take is refused at transform too, in both paths.
  auto fitted_box = fitted("power_transform", {{"columns", {"a"}}, {"method", "box-cox"}}, schema,
                           {make_batch({"a"}, {f64_column({1.0, 2.0, 4.0})})});
  REQUIRE_THROWS_WITH(fitted_box->transform(make_batch({"a"}, {f64_column({0.0})})),
                      ContainsSubstring("is not strictly positive, which box-cox requires"));
  REQUIRE_THROWS_WITH(fitted_box->transform_row({{"a", -1.0}}),
                      ContainsSubstring("is not strictly positive, which box-cox requires"));
}

TEST_CASE("power_transform load_state rejects a malformed table") {
  auto fresh = [](bool standardize) {
    auto kernel = fft::make_kernel("power_transform", {{"columns", {"a"}}, {"standardize", standardize}});
    kernel->stage = "expand";
    return kernel;
  };
  const nlohmann::json good{{"lambdas", {{"a", 0.5}}}, {"means", {{"a", 1.0}}}, {"scales", {{"a", 2.0}}}};
  REQUIRE_NOTHROW(fresh(true)->load_state(good));
  REQUIRE_NOTHROW(fresh(false)->load_state({{"lambdas", {{"a", 0.5}}}, {"means", nullptr}, {"scales", nullptr}}));
  auto broken = [&](bool standardize, auto mutate, const std::string& message) {
    nlohmann::json state = good;
    mutate(state);
    REQUIRE_THROWS_WITH(fresh(standardize)->load_state(state), ContainsSubstring(message));
  };
  broken(true, [](nlohmann::json& state) { state.erase("lambdas"); }, "state must hold a 'lambdas' object");
  broken(true, [](nlohmann::json& state) { state["lambdas"] = {{"b", 0.5}}; }, "lambdas has no entry for column 'a'");
  broken(true, [](nlohmann::json& state) { state["lambdas"]["a"] = "half"; }, "must hold finite numbers");
  broken(true, [](nlohmann::json& state) { state["scales"]["a"] = 0.0; }, "must be positive");
  broken(
      true, [](nlohmann::json& state) { state["means"] = nullptr; },
      "must be present exactly when standardize is true");
  broken(false, [](nlohmann::json& state) { (void)state; }, "must be present exactly when standardize is true");
  broken(true, [](nlohmann::json& state) { state["means"] = {{"b", 1.0}}; }, "means has no entry for column 'a'");
}

// REGRESSION: an expander never writes a value its own readers refuse. a^2 for
// a = 1e200 overflows to inf, and a spline basis continued far past its knots
// does the same. Every kernel here refuses a non-finite number when it reads
// one, so an expander that emitted one would build a column the next step
// cannot read: the pipeline fitted, wrote its manifest, and only failed at the
// step after. Both transform paths refuse it at the producing step instead,
// naming the generated column.
TEST_CASE("the expanders refuse to generate a value that is not finite") {
  auto poly = resolved("polynomial_features", {{"columns", {"a"}}, {"degree", 2}}, fft::Schema{{{"a", "f64"}}});
  REQUIRE_THROWS_WITH(poly->transform(make_batch({"a"}, {f64_column({1e200, 2.0})})),
                      ContainsSubstring("column 'a^2' generates the value inf, which is not finite"));
  REQUIRE_THROWS_WITH(poly->transform_row({{"a", 1e200}}),
                      ContainsSubstring("column 'a^2' generates the value inf, which is not finite"));
  // An ordinary row through the same kernel is untouched.
  REQUIRE(column_of(poly->transform(make_batch({"a"}, {f64_column({3.0})})), "a^2") == std::vector<double>{9.0});

  auto spline =
      fitted("spline_transform", {{"columns", {"a"}}, {"n_knots", 3}, {"degree", 2}, {"extrapolation", "continue"}},
             fft::Schema{{{"a", "f64"}}}, {make_batch({"a"}, {f64_column({0.0, 1.0, 2.0, 3.0})})});
  REQUIRE_THROWS_WITH(spline->transform(make_batch({"a"}, {f64_column({1e300})})),
                      ContainsSubstring("generates the value"));
  REQUIRE_THROWS_WITH(spline->transform_row({{"a", 1e300}}), ContainsSubstring("generates the value"));

  // The same guard covers a hand-edited manifest whose knots are a denormal
  // apart, where the basis divides by a span that underflows.
  auto loaded = fft::make_kernel("spline_transform", {{"columns", {"a"}}, {"n_knots", 4}, {"degree", 2}});
  loaded->stage = "expand";
  loaded->load_state({{"knots", {{"a", {1e-310, 3.3333333333333e-310, 5.6666666666666e-310, 8e-310}}}}});
  REQUIRE_THROWS_WITH(loaded->transform(make_batch({"a"}, {f64_column({1e-310})})),
                      ContainsSubstring("generates the value"));
  REQUIRE_THROWS_WITH(loaded->transform_row({{"a", 1e-310}}), ContainsSubstring("generates the value"));
}

TEST_CASE("spline_transform refuses a width that generates nothing") {
  REQUIRE_THROWS_WITH(
      (fft::make_kernel(
          "spline_transform",
          {{"columns", {"a"}}, {"n_knots", 2}, {"degree", 1}, {"extrapolation", "periodic"}, {"include_bias", false}})),
      ContainsSubstring("generate no columns at all"));
  REQUIRE_THROWS_WITH((fft::make_kernel("spline_transform",
                                        {{"columns", {"a"}}, {"n_knots", 2}, {"degree", 0}, {"include_bias", false}})),
                      ContainsSubstring("generate no columns at all"));
}

TEST_CASE("spline_transform's fitted knots do not depend on the row split") {
  // REGRESSION: a zero knot is +0.0 for any mix of signed zeros and any split.
  // min() and max() return their first argument on a tie and std::sort is not
  // stable, so a -0.0 among the zeros used to put the sign of zero into the
  // knot table and make two fits of the same file differ in their bytes.
  const nlohmann::json params{{"columns", {"a"}}, {"n_knots", 3}, {"degree", 2}, {"knots", "quantile"}};
  const fft::Schema schema{{{"a", "f64"}}};
  const nlohmann::json expected =
      fitted("spline_transform", params, schema, {make_batch({"a"}, {f64_column({0.0, 0.0, 1.0, 2.0})})})->state();
  REQUIRE(expected.at("knots").at("a").at(0).get<double>() == 0.0);
  REQUIRE(!std::signbit(expected.at("knots").at("a").at(0).get<double>()));
  const std::vector<std::vector<fft::Batch>> splits = {
      {make_batch({"a"}, {f64_column({-0.0, 0.0, 1.0, 2.0})})},
      {make_batch({"a"}, {f64_column({0.0, -0.0, 1.0, 2.0})})},
      {make_batch({"a"}, {f64_column({-0.0, 2.0})}), make_batch({"a"}, {f64_column({0.0, 1.0})})},
      {make_batch({"a"}, {f64_column({1.0, 0.0})}), make_batch({"a"}, {f64_column({2.0, -0.0})})},
  };
  for (const std::vector<fft::Batch>& batches : splits) {
    REQUIRE(fitted("spline_transform", params, schema, batches)->state().dump() == expected.dump());
  }
}

TEST_CASE("the expanders refuse an integer past 2^53 rather than rounding it") {
  // The scalers round such an integer; the expanders refuse it through
  // expand.cpp's exact_double, which both transform paths call.
  constexpr int64_t kBig = (int64_t{1} << 53) + 1;
  auto poly = resolved("polynomial_features", {{"columns", {"a"}}, {"degree", 2}}, fft::Schema{{{"a", "i64"}}});
  REQUIRE_THROWS_WITH(poly->transform(make_batch({"a"}, {numeric_column<arrow::Int64Type>({kBig})})),
                      ContainsSubstring("value 9007199254740993 is beyond 2^53 in magnitude"));
  REQUIRE_THROWS_WITH(poly->transform_row({{"a", kBig}}),
                      ContainsSubstring("value 9007199254740993 is beyond 2^53 in magnitude"));
}

TEST_CASE("polynomial suffix reuse distinguishes a source named 1 from the bias") {
  auto kernel = resolved("polynomial_features", {{"columns", {"1"}}, {"degree", {2, 2}}}, fft::Schema{{{"1", "f64"}}});
  const auto output = kernel->transform(make_batch({"1"}, {f64_column({3.0, -2.0, 0.5})}));
  REQUIRE(column_of(output, "1") == std::vector<double>{1.0, 1.0, 1.0});
  REQUIRE(column_of(output, "1^2") == std::vector<double>{9.0, 4.0, 0.25});
  for (const double value : {3.0, -2.0, 0.5}) {
    auto row = kernel->transform_row({{"1", value}});
    REQUIRE(std::get<double>(*fft::row_find(row, "1^2")) == value * value);
  }
}
