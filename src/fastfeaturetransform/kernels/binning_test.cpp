// Explicit C++ tests for robust_scale, quantile_transform and
// k_bins_discretize: the params, the schema rewrite, the buffered
// accumulator merging to the same table in either order, the two transform
// paths agreeing value for value, and load_state.

#include <arrow/array.h>

#include <algorithm>
#include <bit>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <cmath>
#include <exception>
#include <limits>

#include "fastfeaturetransform/kernel.hpp"
#include "fastfeaturetransform/spill.hpp"
#include "fastfeaturetransform/test_support.hpp"

namespace fft = fastfeaturetransform;
using Catch::Approx;
using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::Equals;
using fft::testing::f64_column;
using fft::testing::make_batch;
using fft::testing::numeric_column;
using fft::testing::utf8_column;

namespace {

// Fit a kernel with one worker state per batch.
std::unique_ptr<fft::Kernel> fit_binning(const std::string& id, const nlohmann::json& params, fft::Schema schema,
                                         const std::vector<fft::Batch>& batches) {
  auto kernel = fft::make_kernel(id, params);
  kernel->stage = "bin";
  fft::Schema planned = schema;
  kernel->resolve(planned);
  std::vector<std::unique_ptr<fft::AccState>> states;
  for (const fft::Batch& batch : batches) {
    states.push_back(kernel->make_state());
    kernel->consume(*states.back(), batch);
  }
  kernel->finalize(std::move(states));
  // The engine resolves again against the same input schema after each fit
  // pass, which is what names a learned width.
  kernel->resolve(schema);
  return kernel;
}

std::unique_ptr<fft::Kernel> fit_binning_with_memory(const std::string& id, const nlohmann::json& params,
                                                     const fft::Schema& schema, const std::vector<fft::Batch>& batches,
                                                     std::size_t limit, bool expect_spill) {
  fft::testing::TempDir temporary;
  fft::FitMemoryOptions options;
  options.memory_limit = limit;
  options.temp_dir = temporary.path.string();
  auto memory = std::make_shared<fft::FitMemory>(options);
  std::unique_ptr<fft::Kernel> kernel;
  std::exception_ptr failure;
  {
    fft::FitMemoryScope scope(memory);
    try {
      kernel = fit_binning(id, params, schema, batches);
    } catch (...) {
      failure = std::current_exception();
    }
    REQUIRE(memory->peak_bytes_in_memory() <= limit);
    REQUIRE((memory->spill_files_created() > 0) == expect_spill);
  }
  memory.reset();
  REQUIRE(std::filesystem::is_empty(temporary.path));
  if (failure) std::rethrow_exception(failure);
  return kernel;
}

// A k_bins_discretize kernel loaded with the given edges, one list per column,
// and resolved against the given schema.
std::unique_ptr<fft::Kernel> loaded_bins(const std::string& encode, const nlohmann::json& edges, std::size_t n_bins,
                                         fft::Schema schema) {
  std::vector<std::string> columns;
  for (const auto& item : edges.items()) columns.push_back(item.key());
  auto kernel = fft::make_kernel(
      "k_bins_discretize", {{"columns", columns}, {"n_bins", n_bins}, {"encode", encode}, {"strategy", "uniform"}});
  kernel->stage = "bins";
  kernel->load_state({{"columns", columns}, {"bin_edges", edges}});
  kernel->resolve(schema);
  return kernel;
}

// The bin of every row of one binned column, read back from either encoding.
// A one-hot row must hold exactly one 1, and 0 in every other column.
std::vector<int64_t> read_bins(const fft::Batch& out, const std::string& name, std::size_t width, bool ordinal) {
  std::vector<int64_t> found(static_cast<std::size_t>(out.rows), -1);
  for (std::size_t bin = 0; bin < (ordinal ? 1 : width); ++bin) {
    const int idx = out.find(ordinal ? name : name + "_" + std::to_string(bin));
    REQUIRE(idx >= 0);
    const auto& column = static_cast<const arrow::Int64Array&>(*out.cols[static_cast<std::size_t>(idx)]);
    for (std::size_t row = 0; row < found.size(); ++row) {
      const int64_t value = column.Value(static_cast<int64_t>(row));
      if (ordinal) {
        found[row] = value;
        continue;
      }
      CAPTURE(row, bin, value);
      REQUIRE((value == 0 || value == 1));
      if (value == 1) {
        REQUIRE(found[row] == -1);
        found[row] = static_cast<int64_t>(bin);
      }
    }
  }
  return found;
}

// The same for one record, whose binned values follow the order of its
// output columns.
int64_t read_bin(const fft::Row& record, bool ordinal) {
  if (ordinal) return std::get<int64_t>(record.at(0).second);
  int64_t found = -1;
  for (std::size_t bin = 0; bin < record.size(); ++bin) {
    const int64_t value = std::get<int64_t>(record[bin].second);
    REQUIRE((value == 0 || value == 1));
    if (value == 1) {
      REQUIRE(found == -1);
      found = static_cast<int64_t>(bin);
    }
  }
  return found;
}

void require_binning_bits(const fft::Batch& actual, const fft::Batch& expected) {
  REQUIRE(actual.names == expected.names);
  REQUIRE(actual.rows == expected.rows);
  REQUIRE(actual.cols.size() == expected.cols.size());
  for (std::size_t column = 0; column < actual.cols.size(); ++column) {
    REQUIRE(actual.cols[column]->ValidateFull().ok());
    REQUIRE(actual.cols[column]->Equals(expected.cols[column]));
    if (actual.cols[column]->type_id() != arrow::Type::DOUBLE) continue;
    const auto& got = static_cast<const arrow::DoubleArray&>(*actual.cols[column]);
    const auto& want = static_cast<const arrow::DoubleArray&>(*expected.cols[column]);
    for (int64_t row = 0; row < got.length(); ++row) {
      if (!got.IsNull(row))
        REQUIRE(std::bit_cast<uint64_t>(got.Value(row)) == std::bit_cast<uint64_t>(want.Value(row)));
    }
  }
}

const fft::Schema kSchema{{{"x", "f64"}, {"tag", "utf8"}}};

// 1..8 split over two batches, so the merge has something to fold.
fft::Batch left() {
  return make_batch({"x", "tag"}, {f64_column({1.0, 2.0, 3.0, 4.0}), utf8_column({"a", "b", "c", "d"})});
}
fft::Batch right() {
  return make_batch({"x", "tag"}, {f64_column({5.0, 6.0, 7.0, 8.0}), utf8_column({"e", "f", "g", "h"})});
}

// Every value of an f64 column, nulls as nullopt.
std::vector<std::optional<double>> doubles(const arrow::Array& array) {
  const auto& typed = static_cast<const arrow::DoubleArray&>(array);
  std::vector<std::optional<double>> out;
  for (int64_t row = 0; row < typed.length(); ++row) {
    out.push_back(typed.IsNull(row) ? std::optional<double>{} : typed.Value(row));
  }
  return out;
}

}  // namespace

TEST_CASE("robust_scale validates its params") {
  REQUIRE_THROWS_WITH((fft::make_kernel("robust_scale", {{"with_centering", "yes"}})),
                      ContainsSubstring("'with_centering' must be true or false"));
  REQUIRE_THROWS_WITH((fft::make_kernel("robust_scale", {{"quantile_range", {1.0}}})),
                      ContainsSubstring("'quantile_range' must be two numbers"));
  REQUIRE_THROWS_WITH((fft::make_kernel("robust_scale", {{"quantile_range", {75.0, 25.0}}})),
                      ContainsSubstring("'quantile_range' must be two numbers"));
  REQUIRE_THROWS_WITH((fft::make_kernel("robust_scale", {{"quantile_range", {-1.0, 25.0}}})),
                      ContainsSubstring("'quantile_range' must be two numbers"));
  REQUIRE_THROWS_WITH((fft::make_kernel("robust_scale", {{"unit_variance", true}, {"quantile_range", {0.0, 100.0}}})),
                      ContainsSubstring("unit_variance needs 'quantile_range' strictly inside 0..100"));
  // Inside 0..100 the divisor can still fail: 2.4e-322 / 100 rounds to 0,
  // and two percentiles one ulp apart near 1e-300 share a normal quantile.
  REQUIRE_THROWS_WITH(
      (fft::make_kernel("robust_scale", {{"unit_variance", true}, {"quantile_range", {2.4e-322, 75.0}}})),
      Equals("robust_scale: unit_variance divides by ppf(q_max / 100) - ppf(q_min / 100), which is inf for this "
             "'quantile_range'; it must be finite and above 0"));
  const nlohmann::json touching{{"unit_variance", true}, {"quantile_range", {1e-300, std::nextafter(1e-300, 1.0)}}};
  REQUIRE_THROWS_WITH(fft::make_kernel("robust_scale", touching),
                      ContainsSubstring("which is 0 for this 'quantile_range'"));
  REQUIRE(fft::make_kernel("robust_scale", {})->needs_fit());
  REQUIRE(fft::make_kernel("robust_scale", {})->params().at("quantile_range") == nlohmann::json({25.0, 75.0}));
}

TEST_CASE("robust_scale rewrites the schema and refuses a column it cannot read") {
  auto kernel = fft::make_kernel("robust_scale", {{"columns", {"x"}}});
  kernel->stage = "bin";
  fft::Schema schema = kSchema;
  kernel->resolve(schema);
  REQUIRE(schema.cols == fft::Schema{{{"x", "f64"}, {"tag", "utf8"}}}.cols);

  auto integers = fft::make_kernel("robust_scale", {});
  integers->stage = "bin";
  fft::Schema mixed{{{"n", "i32"}, {"tag", "utf8"}}};
  integers->resolve(mixed);  // columns=null takes every numeric column
  REQUIRE(integers->inputs == std::vector<std::string>{"n"});
  REQUIRE(mixed.cols == fft::Schema{{{"n", "f64"}, {"tag", "utf8"}}}.cols);

  auto text = fft::make_kernel("robust_scale", {{"columns", {"tag"}}});
  text->stage = "bin";
  fft::Schema strings = kSchema;
  REQUIRE_THROWS_WITH(text->resolve(strings), ContainsSubstring("is utf8, expected an integer or float column"));

  auto nothing = fft::make_kernel("robust_scale", {});
  nothing->stage = "bin";
  fft::Schema all_text{{{"tag", "utf8"}}};
  REQUIRE_THROWS_WITH(nothing->resolve(all_text), ContainsSubstring("no integer or float columns to scale"));
}

TEST_CASE("robust_scale merges worker buffers and both paths agree") {
  const nlohmann::json params{{"columns", {"x"}}};
  auto kernel = fit_binning("robust_scale", params, kSchema, {left(), right()});
  const nlohmann::json state = kernel->state();
  REQUIRE(state.at("center").at("x").get<double>() == Approx(4.5));  // median of 1..8
  REQUIRE(state.at("scale").at("x").get<double>() == Approx(3.5));   // 6.25 - 2.75
  REQUIRE(fit_binning("robust_scale", params, kSchema, {right(), left()})->state() == state);

  const std::vector<std::optional<double>> values{1.0, 4.5, 8.0, std::nullopt};
  fft::Batch out = kernel->transform(make_batch({"x"}, {numeric_column<arrow::DoubleType>(values)}));
  const std::vector<std::optional<double>> scaled = doubles(*out.cols[0]);
  REQUIRE(scaled[0] == Approx(-1.0));
  REQUIRE(scaled[1] == Approx(0.0));
  REQUIRE(!scaled[3].has_value());  // a null passes straight through
  for (std::size_t i = 0; i < values.size(); ++i) {
    fft::RowValue value = values[i].has_value() ? fft::RowValue(*values[i]) : fft::RowValue(std::monostate{});
    fft::Row row = kernel->transform_row({{"x", value}});
    if (scaled[i].has_value()) {
      REQUIRE(std::get<double>(row[0].second) == *scaled[i]);
    } else {
      REQUIRE(std::holds_alternative<std::monostate>(row[0].second));
    }
  }
  REQUIRE_THROWS_WITH(kernel->transform_row({{"x", std::string("1")}}), ContainsSubstring("must be a number or null"));
  REQUIRE_THROWS_WITH(kernel->transform_row({{"tag", 1.0}}), ContainsSubstring("missing from record"));
}

TEST_CASE("robust_scale answers a constant column and a switched-off statistic") {
  const fft::Batch flat = make_batch({"x"}, {f64_column({3.0, 3.0, 3.0})});
  auto kernel = fit_binning("robust_scale", {{"columns", {"x"}}}, fft::Schema{{{"x", "f64"}}}, {flat});
  REQUIRE(kernel->state().at("scale").at("x").get<double>() == Approx(1.0));  // the near-zero rule

  auto plain = fit_binning("robust_scale", {{"columns", {"x"}}, {"with_centering", false}, {"with_scaling", false}},
                           fft::Schema{{{"x", "f64"}}}, {left()});
  REQUIRE(plain->state().at("center").at("x").is_null());
  REQUIRE(plain->state().at("scale").at("x").is_null());
  fft::Batch out = plain->transform(make_batch({"x"}, {f64_column({2.0})}));
  REQUIRE(doubles(*out.cols[0])[0] == Approx(2.0));  // an identity, not a division by nothing
}

TEST_CASE("robust_scale's unit_variance answers a q_min below the smallest normal double") {
  // REGRESSION: the inverse normal CDF answers every p in (0, 1). Below about
  // 6e-311 its Halley steps used to overflow into NaN, so q_min = 1e-318
  // failed the fit where sklearn fits a scale of 0.17974705181433237.
  const fft::Batch batch = make_batch({"x"}, {f64_column({1.0, 2.0, 4.0, 8.0, 16.0})});
  auto kernel =
      fit_binning("robust_scale", {{"columns", {"x"}}, {"quantile_range", {1e-318, 75.0}}, {"unit_variance", true}},
                  fft::Schema{{{"x", "f64"}}}, {batch});
  REQUIRE(kernel->state().at("scale").at("x").get<double>() == Approx(0.17974705181433237).epsilon(1e-12));
}

TEST_CASE("robust_scale's unit_variance keeps its relative accuracy on a tight central range") {
  // REGRESSION: near p = 0.5 the inverse normal CDF refines on erf's residual,
  // which keeps x's relative accuracy. Its erfc residual used to cancel there,
  // which put this scale 1.8e-11 from sklearn's, past the documented 1e-12.
  std::vector<double> values;
  for (int i = 0; i <= 1000; ++i) values.push_back(-3.0 + i * (6.0 / 1000.0));  // np.linspace(-3, 3, 1001)
  auto kernel =
      fit_binning("robust_scale", {{"columns", {"x"}}, {"quantile_range", {49.9999, 50.0001}}, {"unit_variance", true}},
                  fft::Schema{{{"x", "f64"}}}, {make_batch({"x"}, {f64_column(values)})});
  REQUIRE(kernel->state().at("scale").at("x").get<double>() == Approx(2.393653682416722).epsilon(1e-12));
}

TEST_CASE("robust_scale load_state rejects a malformed table") {
  auto fresh = [](const nlohmann::json& params) {
    auto kernel = fft::make_kernel("robust_scale", params);
    kernel->stage = "bin";
    return kernel;
  };
  const nlohmann::json params{{"columns", {"x"}}};
  const nlohmann::json good{{"columns", {"x"}}, {"center", {{"x", 4.5}}}, {"scale", {{"x", 3.5}}}};
  REQUIRE_NOTHROW(fresh(params)->load_state(good));
  auto broken = [&](auto mutate, const std::string& message) {
    nlohmann::json state = good;
    mutate(state);
    REQUIRE_THROWS_WITH(fresh(params)->load_state(state), ContainsSubstring(message));
  };
  broken([](nlohmann::json& state) { state.erase("scale"); }, "state must hold a 'center' object");
  broken([](nlohmann::json& state) { state["center"] = {{"y", 1.0}}; }, "center has no entry for column 'x'");
  broken([](nlohmann::json& state) { state["scale"]["x"] = "wide"; }, "scale['x'] must be a finite number");
  broken([](nlohmann::json& state) { state["scale"]["x"] = 0.0; }, "scale['x'] must be a positive number");
  broken([](nlohmann::json& state) { state["scale"]["x"] = -2.0; }, "scale['x'] must be a positive number");
  broken([](nlohmann::json& state) { state["center"]["x"] = nullptr; }, "center['x'] must be a finite number");
  // A statistic the params say was never fitted must be null, not a number no
  // transform would ever use.
  REQUIRE_THROWS_WITH(fresh({{"columns", {"x"}}, {"with_scaling", false}})->load_state(good),
                      ContainsSubstring("scale['x'] must be null when this scaler does not fit it"));
}

TEST_CASE("quantile_transform validates its params") {
  REQUIRE_THROWS_WITH((fft::make_kernel("quantile_transform", {{"n_quantiles", 0}})),
                      ContainsSubstring("'n_quantiles' must be an integer between 1 and"));
  REQUIRE_THROWS_WITH((fft::make_kernel("quantile_transform", {{"output_distribution", "poisson"}})),
                      ContainsSubstring("'output_distribution' must be 'uniform' or 'normal'"));
  REQUIRE(fft::make_kernel("quantile_transform", {})->params().at("n_quantiles") == 1000);
}

TEST_CASE("quantile_transform merges worker buffers and both paths agree") {
  const nlohmann::json params{{"columns", {"x"}}, {"n_quantiles", 3}};
  auto kernel = fit_binning("quantile_transform", params, kSchema, {left(), right()});
  const nlohmann::json state = kernel->state();
  REQUIRE(state.at("references") == nlohmann::json({0.0, 0.5, 1.0}));
  REQUIRE(state.at("quantiles").at("x") == nlohmann::json({1.0, 4.5, 8.0}));
  REQUIRE(fit_binning("quantile_transform", params, kSchema, {right(), left()})->state() == state);

  const std::vector<std::optional<double>> values{1.0, 4.5, 8.0, 100.0, -100.0, std::nullopt};
  fft::Batch out = kernel->transform(make_batch({"x"}, {numeric_column<arrow::DoubleType>(values)}));
  const std::vector<std::optional<double>> ranks = doubles(*out.cols[0]);
  REQUIRE(ranks[0] == Approx(0.0));  // the fitted minimum pins to exactly 0
  REQUIRE(ranks[1] == Approx(0.5));
  REQUIRE(ranks[2] == Approx(1.0));  // and the maximum to exactly 1
  REQUIRE(ranks[3] == Approx(1.0));  // past the fitted range, clipped
  REQUIRE(ranks[4] == Approx(0.0));
  REQUIRE(!ranks[5].has_value());
  for (std::size_t i = 0; i < values.size(); ++i) {
    fft::RowValue value = values[i].has_value() ? fft::RowValue(*values[i]) : fft::RowValue(std::monostate{});
    fft::Row row = kernel->transform_row({{"x", value}});
    if (ranks[i].has_value()) {
      REQUIRE(std::get<double>(row[0].second) == *ranks[i]);
    } else {
      REQUIRE(std::holds_alternative<std::monostate>(row[0].second));
    }
  }
}

TEST_CASE("quantile_transform averages the two interpolations over a run of ties") {
  // The grid is 0, 1, 1, 1, 2 over the references 0, 0.25, 0.5, 0.75, 1. A
  // 1 interpolated forwards alone answers 0.75, the last tie's reference, and
  // backwards alone 0.25, the first's; upstream averages the two to 0.5, and
  // so do we. The run sits inside the grid, so no bound pins the answer.
  const fft::Batch batch = make_batch({"x"}, {f64_column({0.0, 1.0, 1.0, 1.0, 2.0})});
  auto kernel =
      fit_binning("quantile_transform", {{"columns", {"x"}}, {"n_quantiles", 5}}, fft::Schema{{{"x", "f64"}}}, {batch});
  REQUIRE(kernel->state().at("quantiles").at("x") == nlohmann::json({0.0, 1.0, 1.0, 1.0, 2.0}));
  const std::vector<double> values{1.0, 0.5, 1.5, 0.0, 2.0};
  const std::vector<std::optional<double>> expected{0.5, 0.125, 0.875, 0.0, 1.0};  // sklearn 1.9.0's
  fft::Batch out = kernel->transform(make_batch({"x"}, {f64_column(values)}));
  REQUIRE(doubles(*out.cols[0]) == expected);
  for (std::size_t i = 0; i < values.size(); ++i) {
    REQUIRE(std::get<double>(kernel->transform_row({{"x", values[i]}})[0].second) == *expected[i]);
  }
}

TEST_CASE("k_bins_discretize's kmeans scales a column far from its mean by a power of two") {
  // REGRESSION: a column whose values reach 2^500 from its mean runs as the
  // column divided by the power of two that brings them just under 2^500,
  // which is exact, so its edges are that column's times the same power.
  // Unscaled, the squared centers overflowed past about 1.3e154, every
  // distance became infinite, and the fit returned wrong edges without an
  // error.
  const nlohmann::json params{{"columns", {"x"}}, {"n_bins", 3}, {"encode", "ordinal"}, {"strategy", "kmeans"}};
  const fft::Schema schema{{{"x", "f64"}}};
  const auto edges = [&](const std::vector<double>& values, int bins) {
    nlohmann::json asked = params;
    asked["n_bins"] = bins;
    const nlohmann::json fitted =
        fit_binning("k_bins_discretize", asked, schema, {make_batch({"x"}, {f64_column(values)})})->state();
    return fitted.at("bin_edges").at("x").get<std::vector<double>>();
  };
  const auto scaled = [](std::vector<double> values, int exponent) {
    for (double& value : values) value = std::ldexp(value, exponent);
    return values;
  };
  // A column of ordinary spread, and one of 2^600 beside -17, -8 and 9,
  // whose small values would underflow if the scaling went further.
  const std::vector<double> narrow{-2.0, -1.5, -1.0, -0.75, 0.0, 0.5, 1.0, 2.5, 3.0, 4.0};
  REQUIRE(edges(scaled(narrow, 520), 3) == scaled(edges(narrow, 3), 520));
  const std::vector<double> wide{-std::ldexp(1.0, 600), -17.0, -8.0, 9.0, std::ldexp(1.0, 600)};
  REQUIRE(edges(wide, 4) == scaled(edges(scaled(wide, -101), 4), 101));
  // Near f64's largest the initial midpoints overflow their sum, and a span
  // past f64 is refused rather than fit to a degenerate model.
  const std::vector<double> top{0.0, 1.63e308, 3e304, 1e304};
  REQUIRE(edges(top, 3) == scaled(edges(scaled(top, -2), 3), 2));
  REQUIRE_THROWS_WITH(edges({-1e308, -5e307, 0.0, 5e307, 1e308}, 3),
                      ContainsSubstring("column 'x' spans past the range of f64"));
}

TEST_CASE("quantile_transform interpolates between quantiles too close for numpy's slope") {
  // REGRESSION: a rank between two quantiles closer than the reference step
  // over f64's largest is a number in [0, 1]. numpy's slope overflows there,
  // so both interpolations used to come out inf and their average NaN, as
  // sklearn's still does.
  const fft::Batch batch = make_batch({"x"}, {f64_column({0.0, 1e-323, 2e-323, 3e-323})});
  auto kernel =
      fit_binning("quantile_transform", {{"columns", {"x"}}, {"n_quantiles", 4}}, fft::Schema{{{"x", "f64"}}}, {batch});
  fft::Batch out = kernel->transform(make_batch({"x"}, {f64_column({5e-324, 1.5e-323})}));
  const std::vector<std::optional<double>> ranks = doubles(*out.cols[0]);
  REQUIRE(ranks[0] == Approx(1.0 / 6.0));  // halfway from 0 to 1e-323, whose ranks are 0 and 1/3
  REQUIRE(ranks[1] == Approx(0.5));
}

TEST_CASE("quantile_transform clips a rank below the smallest normal double") {
  // REGRESSION: a rank below about 6e-311 maps to the lower clip, as in
  // sklearn. The inverse normal CDF used to answer NaN there, and the clip
  // let the NaN through. Over 0 and 2^1023, 2^-20 ranks at 2^-1044.
  const fft::Batch batch = make_batch({"x"}, {f64_column({0.0, std::ldexp(1.0, 1023)})});
  auto kernel =
      fit_binning("quantile_transform", {{"columns", {"x"}}, {"n_quantiles", 2}, {"output_distribution", "normal"}},
                  fft::Schema{{{"x", "f64"}}}, {batch});
  fft::Batch out =
      kernel->transform(make_batch({"x"}, {f64_column({0.0, std::ldexp(1.0, -20), std::ldexp(1.0, -10)})}));
  const std::vector<std::optional<double>> outputs = doubles(*out.cols[0]);
  REQUIRE(outputs[0] == Approx(-5.199337582605575).epsilon(1e-12));  // the fitted minimum, at the clip
  REQUIRE(outputs[1] == outputs[0]);
  REQUIRE(outputs[2] == outputs[0]);
  REQUIRE(std::get<double>(kernel->transform_row({{"x", std::ldexp(1.0, -20)}})[0].second) == *outputs[0]);
}

TEST_CASE("quantile_transform load_state rejects a malformed table") {
  auto fresh = [] {
    auto kernel = fft::make_kernel("quantile_transform", {{"columns", {"x"}}, {"n_quantiles", 3}});
    kernel->stage = "bin";
    return kernel;
  };
  const nlohmann::json good{
      {"columns", {"x"}}, {"references", {0.0, 0.5, 1.0}}, {"quantiles", {{"x", {1.0, 4.5, 8.0}}}}};
  REQUIRE_NOTHROW(fresh()->load_state(good));
  auto broken = [&](auto mutate, const std::string& message) {
    nlohmann::json state = good;
    mutate(state);
    REQUIRE_THROWS_WITH(fresh()->load_state(state), ContainsSubstring(message));
  };
  broken([](nlohmann::json& state) { state.erase("quantiles"); }, "state must hold a 'quantiles' object");
  broken([](nlohmann::json& state) { state.erase("references"); }, "state must hold a non-empty 'references' list");
  broken([](nlohmann::json& state) { state["references"] = {0.0, 0.6, 1.0}; }, "points evenly spaced from 0 to 1");
  broken([](nlohmann::json& state) { state["quantiles"]["x"] = {1.0, 4.5}; }, "one value per reference");
  broken([](nlohmann::json& state) { state["quantiles"]["x"] = {8.0, 4.5, 1.0}; }, "must not decrease");
  broken([](nlohmann::json& state) { state["quantiles"]["x"] = {1.0, 4.5, "far"}; }, "must hold finite numbers");
}

TEST_CASE("k_bins_discretize validates its params") {
  REQUIRE_THROWS_WITH((fft::make_kernel("k_bins_discretize", {{"n_bins", 1}})),
                      ContainsSubstring("'n_bins' must be an integer between 2 and"));
  REQUIRE_THROWS_WITH((fft::make_kernel("k_bins_discretize", {{"encode", "sparse"}})),
                      ContainsSubstring("'encode' must be 'onehot', 'onehot-dense' or 'ordinal'"));
  REQUIRE_THROWS_WITH((fft::make_kernel("k_bins_discretize", {{"strategy", "gaussian"}})),
                      ContainsSubstring("'strategy' must be 'uniform', 'quantile' or 'kmeans'"));
  REQUIRE_THROWS_WITH((fft::make_kernel("k_bins_discretize", {{"quantile_method", "nearest"}})),
                      ContainsSubstring("'quantile_method' must be one of"));
  REQUIRE(fft::make_kernel("k_bins_discretize", {})->params().at("strategy") == "quantile");
  REQUIRE(fft::make_kernel("k_bins_discretize", {})->params().at("quantile_method") == "averaged_inverted_cdf");
}

TEST_CASE("k_bins_discretize rewrites the schema for both encodings") {
  auto ordinal = fft::make_kernel("k_bins_discretize", {{"columns", {"x"}}, {"encode", "ordinal"}});
  ordinal->stage = "bin";
  fft::Schema schema = kSchema;
  ordinal->resolve(schema);
  REQUIRE(schema.cols == fft::Schema{{{"x", "i64"}, {"tag", "utf8"}}}.cols);

  // The width is learned, so an unfitted one-hot step drops its column and
  // declares nothing; the engine resolves again once the fit is in.
  auto expanding = fft::make_kernel("k_bins_discretize", {{"columns", {"x"}}, {"n_bins", 2}, {"strategy", "uniform"}});
  expanding->stage = "bin";
  fft::Schema wide = kSchema;
  expanding->resolve(wide);
  REQUIRE(wide.cols == fft::Schema{{{"tag", "utf8"}}}.cols);
  REQUIRE(expanding->outputs.empty());
}

TEST_CASE("k_bins_discretize merges worker buffers and both paths agree") {
  const nlohmann::json params{{"columns", {"x"}}, {"n_bins", 4}, {"encode", "ordinal"}, {"strategy", "uniform"}};
  auto kernel = fit_binning("k_bins_discretize", params, kSchema, {left(), right()});
  REQUIRE(kernel->state().at("bin_edges").at("x") == nlohmann::json({1.0, 2.75, 4.5, 6.25, 8.0}));
  REQUIRE(fit_binning("k_bins_discretize", params, kSchema, {right(), left()})->state() ==
          kernel->state());  // a min/max monoid: no buffer, and no dependence on the split

  const std::vector<double> values{-5.0, 1.0, 3.0, 5.0, 8.0, 99.0};
  fft::Batch out = kernel->transform(make_batch({"x"}, {f64_column(values)}));
  const auto& bins = static_cast<const arrow::Int64Array&>(*out.cols[0]);
  REQUIRE(bins.Value(0) == 0);  // below the first edge
  REQUIRE(bins.Value(1) == 0);
  REQUIRE(bins.Value(2) == 1);
  REQUIRE(bins.Value(3) == 2);
  REQUIRE(bins.Value(4) == 3);  // the top edge is not a bin of its own
  REQUIRE(bins.Value(5) == 3);
  for (std::size_t i = 0; i < values.size(); ++i) {
    fft::Row row = kernel->transform_row({{"x", values[i]}});
    REQUIRE(std::get<int64_t>(row[0].second) == bins.Value(static_cast<int64_t>(i)));
  }
  REQUIRE_THROWS_WITH(kernel->transform_row({{"x", std::monostate{}}}),
                      ContainsSubstring("is null, which k_bins_discretize refuses"));
}

TEST_CASE("k_bins_discretize expands to one column per bin in both paths") {
  const nlohmann::json params{{"columns", {"x"}}, {"n_bins", 2}, {"encode", "onehot"}, {"strategy", "uniform"}};
  auto kernel = fit_binning("k_bins_discretize", params, kSchema, {left(), right()});
  REQUIRE(kernel->outputs == std::vector<std::string>{"x_0", "x_1"});

  fft::Batch out = kernel->transform(make_batch({"x", "tag"}, {f64_column({1.0, 8.0}), utf8_column({"a", "b"})}));
  REQUIRE(out.names == std::vector<std::string>{"tag", "x_0", "x_1"});
  const auto& low = static_cast<const arrow::Int64Array&>(*out.cols[1]);
  const auto& high = static_cast<const arrow::Int64Array&>(*out.cols[2]);
  REQUIRE(low.Value(0) == 1);
  REQUIRE(high.Value(0) == 0);
  REQUIRE(low.Value(1) == 0);
  REQUIRE(high.Value(1) == 1);
  for (int64_t row = 0; row < 2; ++row) {
    fft::Row record = kernel->transform_row({{"x", row == 0 ? 1.0 : 8.0}, {"tag", std::string("a")}});
    REQUIRE(record.size() == 3);
    REQUIRE(record[0].first == "tag");
    REQUIRE(std::get<int64_t>(record[1].second) == low.Value(row));
    REQUIRE(std::get<int64_t>(record[2].second) == high.Value(row));
  }
  // A passthrough key named like a generated column would be dropped
  // silently, so the record path refuses it.
  REQUIRE_THROWS_WITH(kernel->transform_row({{"x", 1.0}, {"x_0", 1.0}}),
                      ContainsSubstring("key 'x_0' already present in record"));
}

// A bin is the number of interior edges at or below the value: upstream's
// searchsorted(edges[1:-1], x, side="right"), here std::upper_bound. Repeated
// edges, an edge at zero with values of either sign of zero, values on every
// edge, and values beyond both ends must bin the same in a float64 column
// without nulls (the dense path), in the same column at an offset, in an
// integer column (row by row), and record by record. The second edge set has
// 20 interior edges, past the 16 the dense path counts, so there the bin is
// searched for.
TEST_CASE("k_bins_discretize bins on and between its edges alike in every path") {
  const std::vector<std::vector<double>> edge_sets{{-3.0, -1.0, 0.0, 0.0, 2.5, 3.0}, [] {
                                                     std::vector<double> many;
                                                     for (int edge = 0; edge <= 21; ++edge)
                                                       many.push_back(edge * 0.5 - 5.0);
                                                     return many;
                                                   }()};
  for (const std::vector<double>& edges : edge_sets) {
    std::vector<double> values{-0.0, 0.0, -1e-300, 1e-300, -100.0, 100.0, 7.0, -7.0};
    for (const double edge : edges) {
      values.push_back(edge);
      values.push_back(std::nextafter(edge, -1e9));
      values.push_back(std::nextafter(edge, 1e9));
      values.push_back(std::round(edge));
    }
    std::vector<int64_t> expected;
    for (const double value : values) {
      expected.push_back(std::upper_bound(edges.begin() + 1, edges.end() - 1, value) - (edges.begin() + 1));
    }
    for (const std::string encode : {"ordinal", "onehot-dense"}) {
      CAPTURE(edges.size(), encode);
      const bool ordinal = encode == "ordinal";
      auto kernel = loaded_bins(encode, {{"x", edges}}, edges.size() - 1, fft::Schema{{{"x", "f64"}}});
      const auto bins = [&](const fft::Batch& batch) {
        return read_bins(kernel->transform(batch), "x", edges.size() - 1, ordinal);
      };
      std::vector<std::optional<double>> padded{999.0};  // sliced off, so the values start at an offset
      padded.insert(padded.end(), values.begin(), values.end());
      const auto sliced = numeric_column<arrow::DoubleType>(padded)->Slice(1, static_cast<int64_t>(values.size()));
      REQUIRE(bins(make_batch({"x"}, {f64_column(values)})) == expected);
      REQUIRE(bins(make_batch({"x"}, {sliced})) == expected);
      for (std::size_t row = 0; row < values.size(); ++row) {
        CAPTURE(values[row]);
        REQUIRE(read_bin(kernel->transform_row({{"x", values[row]}}), ordinal) == expected[row]);
      }
      std::vector<std::optional<int64_t>> integers;
      std::vector<int64_t> integer_bins;
      for (int value = -6; value <= 6; ++value) {
        integers.emplace_back(value);
        integer_bins.push_back(std::upper_bound(edges.begin() + 1, edges.end() - 1, static_cast<double>(value)) -
                               (edges.begin() + 1));
      }
      REQUIRE(bins(make_batch({"x"}, {numeric_column<arrow::Int64Type>(integers)})) == integer_bins);
      // A value that is not finite is refused at the row it is on, whichever path reads it.
      std::vector<double> bad = values;
      bad[3] = std::numeric_limits<double>::quiet_NaN();
      REQUIRE_THROWS_WITH(kernel->transform(make_batch({"x"}, {f64_column(bad)})),
                          ContainsSubstring("bins: column 'x': value nan is not finite"));
    }
  }
}

// The dense path takes a float64 column without nulls whose values are all
// finite, against at most 16 interior edges. Anything else is read row by
// row, which refuses the first null or value that is not finite.
TEST_CASE("k_bins_discretize refuses an infinity and a null in a float64 column") {
  const double inf = std::numeric_limits<double>::infinity();
  // k_bins_discretize refuses a null too, so its message does not suggest one.
  const std::string refusal = " is not finite";
  for (const std::string encode : {"ordinal", "onehot"}) {
    CAPTURE(encode);
    auto kernel = loaded_bins(encode, {{"x", {0.0, 1.0, 2.0, 3.0}}}, 3, fft::Schema{{{"x", "f64"}}});
    REQUIRE_THROWS_WITH(kernel->transform(make_batch({"x"}, {f64_column({1.0, inf, 2.0})})),
                        Equals("bins: column 'x': value inf" + refusal));
    REQUIRE_THROWS_WITH(kernel->transform(make_batch({"x"}, {f64_column({1.0, 2.0, -inf})})),
                        Equals("bins: column 'x': value -inf" + refusal));
    REQUIRE_THROWS_WITH(kernel->transform(make_batch({"x"}, {f64_column({-inf, std::nan(""), inf})})),
                        Equals("bins: column 'x': value -inf" + refusal));
    REQUIRE_THROWS_WITH(
        kernel->transform(make_batch({"x"}, {numeric_column<arrow::DoubleType>({1.0, std::nullopt, 2.0})})),
        Equals("bins: column 'x' has a null, which k_bins_discretize refuses"));
  }
}

// float32 goes row by row, and each value is binned as its widening to f64.
TEST_CASE("k_bins_discretize bins a float32 column by its f64 widening") {
  const std::vector<double> edges{-3.0, -1.0, 0.0, 0.0, 2.5, 3.0};
  std::vector<std::optional<float>> values;
  std::vector<int64_t> expected;
  for (const float value :
       {-7.0f, -1.0f, std::nextafter(-1.0f, -2.0f), -0.0f, 0.0f, 2.5f, std::nextafter(2.5f, 0.0f), 100.0f}) {
    values.emplace_back(value);
    expected.push_back(std::upper_bound(edges.begin() + 1, edges.end() - 1, static_cast<double>(value)) -
                       (edges.begin() + 1));
  }
  for (const std::string encode : {"ordinal", "onehot"}) {
    CAPTURE(encode);
    auto kernel = loaded_bins(encode, {{"x", edges}}, edges.size() - 1, fft::Schema{{{"x", "f32"}}});
    const fft::Batch out = kernel->transform(make_batch({"x"}, {numeric_column<arrow::FloatType>(values)}));
    REQUIRE(read_bins(out, "x", edges.size() - 1, encode == "ordinal") == expected);
  }
}

// n_bins can be up to 4096, so every bin index must survive the trip through
// whatever the transform holds the bins in.
TEST_CASE("k_bins_discretize bins past 255 bins in both paths") {
  std::vector<double> edges;
  for (int edge = 0; edge <= 300; ++edge) edges.push_back(static_cast<double>(edge));
  std::vector<double> values;
  std::vector<int64_t> expected;
  for (int bin = 0; bin < 300; ++bin) {
    values.push_back(bin + 0.5);
    expected.push_back(bin);
  }
  for (const std::string encode : {"ordinal", "onehot"}) {
    CAPTURE(encode);
    const bool ordinal = encode == "ordinal";
    auto kernel = loaded_bins(encode, {{"x", edges}}, 300, fft::Schema{{{"x", "f64"}}});
    REQUIRE(read_bins(kernel->transform(make_batch({"x"}, {f64_column(values)})), "x", 300, ordinal) == expected);
    REQUIRE(read_bin(kernel->transform_row({{"x", 299.5}}), ordinal) == 299);
  }
}

// The dense path counts a block of rows at a time, so a long column runs
// through several blocks and ends partway into one. Its values cycle with a
// period of 97, which does not divide the 2048-row block, so each block starts
// elsewhere in the cycle.
TEST_CASE("k_bins_discretize bins a long column alike in every block") {
  const std::vector<double> edges{-3.0, -1.0, 0.0, 0.0, 2.5, 3.0};
  std::vector<double> values;
  std::vector<int64_t> expected;
  for (int row = 0; row < 5000; ++row) {
    const double value = std::ldexp(static_cast<double>(row % 97) - 48.0, -3);
    values.push_back(value);
    expected.push_back(std::upper_bound(edges.begin() + 1, edges.end() - 1, value) - (edges.begin() + 1));
  }
  for (const std::string encode : {"ordinal", "onehot"}) {
    CAPTURE(encode);
    auto kernel = loaded_bins(encode, {{"x", edges}}, edges.size() - 1, fft::Schema{{{"x", "f64"}}});
    REQUIRE(read_bins(kernel->transform(make_batch({"x"}, {f64_column(values)})), "x", edges.size() - 1,
                      encode == "ordinal") == expected);
  }
}

// Each column is binned by its own edges.
TEST_CASE("k_bins_discretize bins each column by its own edges") {
  for (const std::string encode : {"ordinal", "onehot"}) {
    CAPTURE(encode);
    const bool ordinal = encode == "ordinal";
    auto kernel = loaded_bins(encode, {{"x", {0.0, 1.0, 2.0}}, {"y", {0.0, 10.0, 20.0}}}, 2,
                              fft::Schema{{{"x", "f64"}, {"y", "f64"}}});
    const fft::Batch out = kernel->transform(make_batch({"x", "y"}, {f64_column({0.5, 1.5}), f64_column({5.0, 15.0})}));
    REQUIRE(read_bins(out, "x", 2, ordinal) == std::vector<int64_t>{0, 1});
    REQUIRE(read_bins(out, "y", 2, ordinal) == std::vector<int64_t>{0, 1});
  }
}

TEST_CASE("k_bins_discretize drops bins that collapse and bins a constant column") {
  // Four 1s, three 2s, then 5 and 9: the low percentiles all land on 1, so
  // upstream drops those edges and the column keeps fewer bins than asked.
  const fft::Batch ties = make_batch({"x"}, {f64_column({1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 5.0, 9.0})});
  auto kernel = fit_binning("k_bins_discretize", {{"columns", {"x"}}, {"n_bins", 5}, {"encode", "ordinal"}},
                            fft::Schema{{{"x", "f64"}}}, {ties});
  REQUIRE(kernel->state().at("bin_edges").at("x").size() < 6);

  const fft::Batch flat = make_batch({"x"}, {f64_column({3.0, 3.0, 3.0})});
  auto constant = fit_binning("k_bins_discretize", {{"columns", {"x"}}, {"n_bins", 4}, {"encode", "onehot"}},
                              fft::Schema{{{"x", "f64"}}}, {flat});
  REQUIRE(constant->outputs == std::vector<std::string>{"x_0"});  // one bin holding everything
  fft::Batch out = constant->transform(make_batch({"x"}, {f64_column({-1.0, 3.0, 99.0})}));
  const auto& only = static_cast<const arrow::Int64Array&>(*out.cols[0]);
  REQUIRE(only.Value(0) == 1);
  REQUIRE(only.Value(2) == 1);
}

TEST_CASE("k_bins_discretize refuses a null and a value that is not finite at fit") {
  const fft::Batch nulls = make_batch({"x"}, {numeric_column<arrow::DoubleType>({1.0, std::nullopt, 3.0})});
  REQUIRE_THROWS_WITH(fit_binning("k_bins_discretize", {{"columns", {"x"}}, {"strategy", "uniform"}},
                                  fft::Schema{{{"x", "f64"}}}, {nulls}),
                      ContainsSubstring("column 'x' has nulls, which k_bins_discretize refuses"));
  const fft::Batch infinite = make_batch({"x"}, {f64_column({1.0, std::numeric_limits<double>::infinity()})});
  REQUIRE_THROWS_WITH(fit_binning("robust_scale", {{"columns", {"x"}}}, fft::Schema{{{"x", "f64"}}}, {infinite}),
                      ContainsSubstring("is not finite; write a missing value as a null"));

  // And at transform, through both paths: a NaN that reached the quantile
  // interpolation would compare false against every bound it holds.
  const double nan = std::numeric_limits<double>::quiet_NaN();
  for (const char* id : {"robust_scale", "quantile_transform", "k_bins_discretize"}) {
    nlohmann::json params{{"columns", {"x"}}};
    if (std::string(id) == "k_bins_discretize") params["encode"] = "ordinal";
    auto kernel = fit_binning(id, params, fft::Schema{{{"x", "f64"}}}, {left(), right()});
    REQUIRE_THROWS_WITH(kernel->transform(make_batch({"x"}, {f64_column({nan})})),
                        ContainsSubstring("column 'x': value nan is not finite"));
    REQUIRE_THROWS_WITH(kernel->transform_row({{"x", nan}}), ContainsSubstring("key 'x': value nan is not finite"));
  }
}

TEST_CASE("a scaler refuses a column with nothing to learn from") {
  const fft::Batch empty = make_batch({"x"}, {numeric_column<arrow::DoubleType>({std::nullopt, std::nullopt})});
  for (const char* id : {"robust_scale", "quantile_transform"}) {
    REQUIRE_THROWS_WITH(fit_binning(id, {{"columns", {"x"}}}, fft::Schema{{{"x", "f64"}}}, {empty}),
                        ContainsSubstring("column 'x' has no non-null values to learn from"));
  }
}

TEST_CASE("k_bins_discretize load_state rejects a malformed table") {
  auto fresh = [] {
    auto kernel = fft::make_kernel("k_bins_discretize", {{"columns", {"x"}}, {"encode", "ordinal"}});
    kernel->stage = "bin";
    return kernel;
  };
  const nlohmann::json good{{"columns", {"x"}}, {"bin_edges", {{"x", {0.0, 1.0, 2.0}}}}};
  REQUIRE_NOTHROW(fresh()->load_state(good));
  auto broken = [&](auto mutate, const std::string& message) {
    nlohmann::json state = good;
    mutate(state);
    REQUIRE_THROWS_WITH(fresh()->load_state(state), ContainsSubstring(message));
  };
  broken([](nlohmann::json& state) { state.erase("bin_edges"); }, "state must hold a 'bin_edges' object");
  broken([](nlohmann::json& state) { state["bin_edges"]["x"] = {1.0}; }, "must hold at least two edges");
  broken([](nlohmann::json& state) { state["bin_edges"]["x"] = {0.0, 2.0, 1.0, 3.0}; },
         "must not decrease between its outer edges");
  // The outer edges bound the fit data and a transform never reads them, so
  // either may sit out of order, as a fit can write it.
  REQUIRE_NOTHROW(fresh()->load_state({{"columns", {"x"}}, {"bin_edges", {{"x", {1.5, 1.0, 2.0, 0.5}}}}}));
  broken([](nlohmann::json& state) { state["bin_edges"]["x"] = {0.0, 1.0, "two"}; }, "must hold finite numbers");
  broken([](nlohmann::json& state) { state["bin_edges"]["x"] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0}; },
         "holds 7 edges, more than the 6 that 'n_bins' allows");
  // Repeated interior edges are what linspace gives over a span of a few ulps
  // and what upstream stores for them, so they load: the bin between two equal
  // edges is empty.
  REQUIRE_NOTHROW(fresh()->load_state({{"columns", {"x"}}, {"bin_edges", {{"x", {0.0, 1.0, 1.0}}}}}));
  broken([](nlohmann::json& state) { state["bin_edges"] = {{"y", {0.0, 1.0}}}; },
         "bin_edges['x'] must be a non-empty list of numbers");
  broken([](nlohmann::json& state) { state["bin_edges"] = nlohmann::json::object(); },
         "state 'columns' does not name the same columns as the fitted table");
  broken([](nlohmann::json& state) { state.erase("columns"); }, "state must hold a 'columns' list");
}

TEST_CASE("kmeans bins a well-separated column into its clusters") {
  const fft::Batch batch = make_batch({"x"}, {f64_column({0.0, 0.1, 0.2, 5.0, 5.1, 5.2, 10.0, 10.1, 10.2})});
  auto kernel = fit_binning("k_bins_discretize",
                            {{"columns", {"x"}}, {"n_bins", 3}, {"encode", "ordinal"}, {"strategy", "kmeans"}},
                            fft::Schema{{{"x", "f64"}}}, {batch});
  const nlohmann::json edges = kernel->state().at("bin_edges").at("x");
  REQUIRE(edges.size() == 4);
  REQUIRE(edges.at(1).get<double>() == Approx(2.6));  // midway between the 0.1 and 5.1 centers
  REQUIRE(edges.at(2).get<double>() == Approx(7.6));
  fft::Batch out = kernel->transform(make_batch({"x"}, {f64_column({0.0, 5.1, 10.2})}));
  const auto& bins = static_cast<const arrow::Int64Array&>(*out.cols[0]);
  REQUIRE(bins.Value(0) == 0);
  REQUIRE(bins.Value(1) == 1);
  REQUIRE(bins.Value(2) == 2);
  // Buffered, so the table must not depend on how the rows were split.
  const fft::Batch first = make_batch({"x"}, {f64_column({10.1, 0.2, 5.2, 0.0})});
  const fft::Batch second = make_batch({"x"}, {f64_column({5.0, 10.2, 0.1, 10.0, 5.1})});
  REQUIRE(fit_binning("k_bins_discretize",
                      {{"columns", {"x"}}, {"n_bins", 3}, {"encode", "ordinal"}, {"strategy", "kmeans"}},
                      fft::Schema{{{"x", "f64"}}}, {first, second})
              ->state() == kernel->state());
  REQUIRE_THROWS_WITH(fit_binning("k_bins_discretize",
                                  {{"columns", {"x"}}, {"n_bins", 4}, {"encode", "ordinal"}, {"strategy", "kmeans"}},
                                  fft::Schema{{{"x", "f64"}}}, {make_batch({"x"}, {f64_column({1.0, 2.0, 3.0})})}),
                      ContainsSubstring("fewer than the 4 clusters strategy='kmeans' needs"));
}

// REGRESSION: k-means follows the empty-cluster rules of upstream's Lloyd
// iteration. scikit-learn 1.9.0 bins [0, 0, 0, 0, 1, 5, 0, 0] at n_bins=5 as
// [1, 1, 1, 1, 2, 3, 1, 1] with bin_edges_ [0, 0, 0.5, 3, 5]. Getting there
// needs all three of _relocate_empty_clusters_dense's and _average_centers'
// rules: a cluster still empty after relocation takes the biggest cluster's
// slot as it stands (the unaveraged sum when that cluster comes later), the
// relocated point keeps the label the assignment gave it, and the centers
// returned are the means the stopping iteration computed. Miss any one and
// every row lands in a different bin.
TEST_CASE("kmeans reproduces upstream's empty-cluster relocation") {
  const fft::Batch batch = make_batch({"x"}, {f64_column({0.0, 0.0, 0.0, 0.0, 1.0, 5.0, 0.0, 0.0})});
  auto kernel = fit_binning("k_bins_discretize",
                            {{"columns", {"x"}}, {"n_bins", 5}, {"encode", "ordinal"}, {"strategy", "kmeans"}},
                            fft::Schema{{{"x", "f64"}}}, {batch});
  REQUIRE(kernel->state().at("bin_edges").at("x") == nlohmann::json({0.0, 0.0, 0.5, 3.0, 5.0}));
  fft::Batch out = kernel->transform(make_batch({"x"}, {f64_column({0.0, 0.0, 0.0, 0.0, 1.0, 5.0, 0.0, 0.0})}));
  const auto& bins = static_cast<const arrow::Int64Array&>(*out.cols[0]);
  const std::vector<int64_t> got(bins.raw_values(), bins.raw_values() + bins.length());
  REQUIRE(got == std::vector<int64_t>{1, 1, 1, 1, 2, 3, 1, 1});
  // The record path answers the same, and so does a different row split.
  REQUIRE(kernel->transform_row({{"x", 5.0}}) == fft::Row{{"x", int64_t{3}}});
  REQUIRE(fit_binning("k_bins_discretize",
                      {{"columns", {"x"}}, {"n_bins", 5}, {"encode", "ordinal"}, {"strategy", "kmeans"}},
                      fft::Schema{{{"x", "f64"}}},
                      {make_batch({"x"}, {f64_column({5.0, 0.0, 0.0})}),
                       make_batch({"x"}, {f64_column({0.0, 1.0, 0.0, 0.0, 0.0})})})
              ->state() == kernel->state());
}

// REGRESSION: a fit survives its own load_state. linspace over a span of a few
// ulps returns edges that repeat, which is what upstream's strategy="uniform"
// stores too. finalize() kept them and load_state() refused them, so save()
// succeeded and load() threw: the failure surfaced at serving, on an artifact
// that had validated at fit. A fit can also write an outer edge out of order:
// a k-means cluster left empty takes a center below the data, and linspace
// over a subnormal span rounds past the maximum, which then overwrites the
// last edge.
TEST_CASE("every fitted bin table loads back") {
  const double base = 1e16;
  const double next = std::nextafter(base, 2e16);
  struct Column {
    const char* what;
    std::vector<double> values;
  };
  const std::vector<Column> columns{
      {"two ulps apart", {base, next, base, next}},
      {"adjacent at one", {1.0, std::nextafter(1.0, 2.0)}},
      {"a denormal span", {0.0, std::numeric_limits<double>::denorm_min()}},
      {"ordinary", {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}},
      {"constant", {3.0, 3.0, 3.0}},
      {"an empty k-means cluster", {-3.0, 10.0, 10.0, -3.0, 5.0, 1.0}},
      {"three subnormal ulps", {0.0, 1.5e-323}},
  };
  std::size_t out_of_order = 0;  // fits whose edge list is not sorted end to end, which still load
  for (const Column& column : columns) {
    INFO(column.what);
    for (const char* strategy : {"uniform", "quantile", "kmeans"}) {
      for (const int bins : {2, 5, 6}) {
        const nlohmann::json params{
            {"columns", {"x"}}, {"n_bins", bins}, {"encode", "ordinal"}, {"strategy", strategy}};
        std::unique_ptr<fft::Kernel> fitted;
        try {
          fitted = fit_binning("k_bins_discretize", params, fft::Schema{{{"x", "f64"}}},
                               {make_batch({"x"}, {f64_column(column.values)})});
        } catch (const fft::Error&) {
          continue;  // a fit that refuses outright leaves no artifact to reload
        }
        const nlohmann::json state = fitted->state();
        auto reloaded = fft::make_kernel("k_bins_discretize", params);
        reloaded->stage = "bin";
        INFO(state.dump());
        REQUIRE_NOTHROW(reloaded->load_state(state));
        REQUIRE(reloaded->state() == state);
        const auto edges = state.at("bin_edges").at("x").get<std::vector<double>>();
        if (!std::is_sorted(edges.begin(), edges.end())) ++out_of_order;
      }
    }
  }
  REQUIRE(out_of_order > 0);
}

// REGRESSION: a fitted table that names no columns is malformed. With
// columns=null the column list comes from the state's 'columns' list. A
// hand-edited empty table used to load as a fitted kernel owning nothing:
// fitted = true, transform a silent no-op, and for encode="onehot" the whole
// generated feature block gone from the output without a word.
TEST_CASE("a fitted table with no columns is refused") {
  const std::vector<std::pair<const char*, nlohmann::json>> cases{
      {"k_bins_discretize", {{"bin_edges", nlohmann::json::object()}}},
      {"quantile_transform", {{"quantiles", nlohmann::json::object()}, {"references", {0.0, 1.0}}}},
      {"robust_scale", {{"center", nlohmann::json::object()}, {"scale", nlohmann::json::object()}}},
  };
  for (const auto& [id, state] : cases) {
    INFO(id);
    auto kernel = fft::make_kernel(id, nlohmann::json::object());
    kernel->stage = "bin";
    // Neither an absent column list nor an empty one may load as a fitted
    // step owning nothing, which is a silent no-op on every served row.
    REQUIRE_THROWS_WITH(kernel->load_state(state), ContainsSubstring("state must hold a 'columns' list"));
    nlohmann::json empty = state;
    empty["columns"] = nlohmann::json::array();
    REQUIRE_THROWS_WITH(kernel->load_state(empty), ContainsSubstring("state must hold a 'columns' list"));
  }
}

TEST_CASE("k_bins_discretize caps the columns the whole step generates") {
  // 4 columns of 2000 bins each is inside the per-column cap and far past the
  // total one; only the total catches it.
  std::vector<double> values;
  for (int i = 0; i < 4000; ++i) values.push_back(static_cast<double>(i));
  nlohmann::json edges = nlohmann::json::object();
  for (const char* name : {"a", "b", "c", "d"}) {
    std::vector<double> column;
    for (int i = 0; i <= 2000; ++i) column.push_back(static_cast<double>(i));
    edges[name] = column;
  }
  auto kernel = fft::make_kernel("k_bins_discretize", {{"n_bins", 2000}, {"encode", "onehot"}});
  kernel->stage = "bin";
  REQUIRE_THROWS_WITH(kernel->load_state({{"columns", {"a", "b", "c", "d"}}, {"bin_edges", edges}}),
                      ContainsSubstring("more than the 4096 allowed"));
}

TEST_CASE("the binning tier rounds an integer past 2^53 rather than refusing it") {
  // numeric.hpp offers two widening policies and this family takes the
  // rounding one, the way sklearn's own float64 cast does. Both paths.
  constexpr int64_t kBig = (int64_t{1} << 53) + 1;
  auto kernel = fit_binning("robust_scale", {{"columns", {"x"}}}, fft::Schema{{{"x", "i64"}}},
                            {make_batch({"x"}, {numeric_column<arrow::Int64Type>({0, 1, 2, 3})})});
  fft::Batch out = kernel->transform(make_batch({"x"}, {numeric_column<arrow::Int64Type>({kBig})}));
  const double batch_value = *doubles(*out.cols[0])[0];
  fft::Row row = kernel->transform_row({{"x", kBig}});
  REQUIRE(std::get<double>(row[0].second) == batch_value);
  // 2^53 + 1 is not representable, so it lands on 2^53 before anything scales.
  REQUIRE(batch_value == Approx((9007199254740992.0 - 1.5) / 1.5));
}

TEST_CASE("buffered numeric selections equal full-sort statistics across worker partitions") {
  std::vector<double> values;
  for (int i = 0; i < 8193; ++i) {
    const double value = static_cast<double>((i * 7919) % 8193 - 4096) / 16.0;
    values.push_back(i % 5 == 0 ? (i % 2 == 0 ? -0.0 : 0.0) : value);
  }
  std::vector<double> sorted = values;
  for (double& value : sorted)
    if (value == 0.0) value = 0.0;
  std::sort(sorted.begin(), sorted.end());
  const auto quantile = [&](double q) {
    const double index = static_cast<double>(sorted.size() - 1) * q;
    const auto lower = static_cast<std::size_t>(std::floor(index));
    if (lower == sorted.size() - 1) return sorted.back();
    const double weight = index - static_cast<double>(lower);
    const double left = sorted[lower];
    const double right = sorted[lower + 1];
    const double span = right - left;
    return weight >= 0.5 ? right - span * (1.0 - weight) : left + span * weight;
  };
  const auto input = f64_column(values);
  const fft::Schema schema{{{"x", "f64"}}};
  const std::vector<fft::Batch> whole{make_batch({"x"}, {input})};
  const std::vector<fft::Batch> parts{make_batch({"x"}, {input->Slice(6000)}),
                                      make_batch({"x"}, {input->Slice(0, 3000)}),
                                      make_batch({"x"}, {input->Slice(3000, 3000)})};
  for (bool centering : {false, true}) {
    for (bool scaling : {false, true}) {
      const nlohmann::json params{{"columns", {"x"}},
                                  {"with_centering", centering},
                                  {"with_scaling", scaling},
                                  {"quantile_range", {13.37, 86.63}}};
      const auto kernel = fit_binning("robust_scale", params, schema, parts);
      REQUIRE(kernel->state().dump() == fit_binning("robust_scale", params, schema, whole)->state().dump());
      if (centering) {
        REQUIRE(std::bit_cast<uint64_t>(kernel->state().at("center").at("x").get<double>()) ==
                std::bit_cast<uint64_t>(sorted[sorted.size() / 2]));
      }
      if (scaling) {
        const double expected = quantile(86.63 / 100.0) - quantile(13.37 / 100.0);
        REQUIRE(std::bit_cast<uint64_t>(kernel->state().at("scale").at("x").get<double>()) ==
                std::bit_cast<uint64_t>(expected));
      }
    }
  }
  // Two or four quantiles select at most eight distinct ranks; eight and larger
  // grids take the measured full-sort fallback. Both must match this oracle.
  for (int width : {2, 4, 8, 513}) {
    const nlohmann::json params{{"columns", {"x"}}, {"n_quantiles", width}};
    const auto kernel = fit_binning("quantile_transform", params, schema, parts);
    const nlohmann::json state = kernel->state();
    REQUIRE(state.dump() == fit_binning("quantile_transform", params, schema, whole)->state().dump());
    for (std::size_t i = 0; i < state.at("references").size(); ++i) {
      const double reference = state.at("references").at(i).get<double>();
      REQUIRE(std::bit_cast<uint64_t>(state.at("quantiles").at("x").at(i).get<double>()) ==
              std::bit_cast<uint64_t>(quantile(reference * 100.0 / 100.0)));
    }
  }
}

TEST_CASE("spilled robust and quantile fits preserve exact state across columns and partitions") {
  std::vector<std::optional<double>> x;
  std::vector<std::optional<double>> y;
  for (int i = 0; i < 4097; ++i) {
    const double value = static_cast<double>((i * 7919) % 4097 - 2048) / 16.0;
    x.emplace_back(i % 13 == 0 ? std::nullopt : std::optional<double>{i % 5 == 0 ? (i % 2 == 0 ? -0.0 : 0.0) : value});
    y.emplace_back(i % 17 == 0 ? std::nullopt : std::optional<double>{value * 3.0 + 0.125});
  }
  const auto first = numeric_column<arrow::DoubleType>(x);
  const auto second = numeric_column<arrow::DoubleType>(y);
  const fft::Schema schema{{{"x", "f64"}, {"y", "f64"}}};
  const fft::Batch input = make_batch({"x", "y"}, {first, second});
  const std::vector<fft::Batch> parts{make_batch({"x", "y"}, {first->Slice(3000), second->Slice(3000)}),
                                      make_batch({"x", "y"}, {first->Slice(0, 1500), second->Slice(0, 1500)}),
                                      make_batch({"x", "y"}, {first->Slice(1500, 1500), second->Slice(1500, 1500)})};
  constexpr std::size_t resident_limit = 8 * 1024 * 1024;
  for (bool centering : {false, true}) {
    for (bool scaling : {false, true}) {
      CAPTURE(centering, scaling);
      const nlohmann::json params{{"columns", {"x", "y"}},
                                  {"with_centering", centering},
                                  {"with_scaling", scaling},
                                  {"unit_variance", scaling},
                                  {"quantile_range", {13.37, 86.63}}};
      const auto resident = fit_binning_with_memory("robust_scale", params, schema, {input}, resident_limit, false);
      const auto spilled = fit_binning_with_memory("robust_scale", params, schema, parts, 1024, centering || scaling);
      REQUIRE(spilled->state().dump() == resident->state().dump());
      require_binning_bits(spilled->transform(input), resident->transform(input));
    }
  }
  for (int width : {2, 4, 31}) {
    for (const std::string distribution : {"uniform", "normal"}) {
      CAPTURE(width, distribution);
      const nlohmann::json params{
          {"columns", {"x", "y"}}, {"n_quantiles", width}, {"output_distribution", distribution}};
      const auto resident =
          fit_binning_with_memory("quantile_transform", params, schema, {input}, resident_limit, false);
      const auto spilled = fit_binning_with_memory("quantile_transform", params, schema, parts, 1024, true);
      REQUIRE(spilled->state().dump() == resident->state().dump());
      require_binning_bits(spilled->transform(input), resident->transform(input));
    }
  }
}

TEST_CASE("spilled KBins quantiles preserve every quantile method") {
  std::vector<double> values;
  for (int i = 0; i < 1025; ++i)
    values.push_back(i % 7 == 0 ? (i % 2 == 0 ? -0.0 : 0.0) : static_cast<double>((i * 127) % 1025 - 512) / 8.0);
  const auto array = f64_column(values);
  const fft::Schema schema{{{"x", "f64"}}};
  const fft::Batch input = make_batch({"x"}, {array});
  const std::vector<fft::Batch> parts{make_batch({"x"}, {array->Slice(513)}),
                                      make_batch({"x"}, {array->Slice(0, 513)})};
  for (const std::string method :
       {"inverted_cdf", "averaged_inverted_cdf", "closest_observation", "interpolated_inverted_cdf", "hazen", "weibull",
        "linear", "median_unbiased", "normal_unbiased"}) {
    CAPTURE(method);
    const nlohmann::json params{{"columns", {"x"}},
                                {"n_bins", 11},
                                {"encode", "ordinal"},
                                {"strategy", "quantile"},
                                {"quantile_method", method}};
    const auto resident = fit_binning_with_memory("k_bins_discretize", params, schema, {input}, 1024 * 1024, false);
    const auto spilled = fit_binning_with_memory("k_bins_discretize", params, schema, parts, 256, true);
    REQUIRE(spilled->state().dump() == resident->state().dump());
    require_binning_bits(spilled->transform(input), resident->transform(input));
  }
}

TEST_CASE("spilled kmeans preserves relocation ties repeated labels and constant columns") {
  struct Case {
    const char* name;
    std::vector<double> values;
    int bins;
    nlohmann::json expected = nullptr;
  };
  std::vector<Case> cases{{"empty relocation", {0.0, 0.0, 0.0, 0.0, 1.0, 5.0, 0.0, 0.0}, 5, {0.0, 0.0, 0.5, 3.0, 5.0}},
                          {"distance ties", {-9.0, -9.0, -9.0, 0.0, 0.0, 0.0, 9.0, 9.0, 9.0}, 7},
                          {"more clusters than distinct values", {0.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 10.0}, 8},
                          {"constant", std::vector<double>(16, 3.0), 4, {3.0, 3.0}},
                          {"assignment ties", {-8.0, -4.0, 0.0, 4.0, 8.0}, 2, {-8.0, 1.0, 8.0}},
                          {"ordinary", {}, 13}};
  for (int i = 0; i < 513; ++i) cases.back().values.push_back(static_cast<double>((i * 127) % 513 - 256) / 16.0);
  const fft::Schema schema{{{"x", "f64"}}};
  for (const auto& example : cases) {
    CAPTURE(example.name);
    const nlohmann::json params{
        {"columns", {"x"}}, {"n_bins", example.bins}, {"encode", "ordinal"}, {"strategy", "kmeans"}};
    const auto array = f64_column(example.values);
    const auto middle = static_cast<int64_t>(example.values.size() / 2);
    const fft::Batch input = make_batch({"x"}, {array});
    const std::vector<fft::Batch> parts{make_batch({"x"}, {array->Slice(middle)}),
                                        make_batch({"x"}, {array->Slice(0, middle)})};
    const auto resident = fit_binning_with_memory("k_bins_discretize", params, schema, {input}, 1024 * 1024, false);
    const auto spilled = fit_binning_with_memory("k_bins_discretize", params, schema, parts, 0, true);
    REQUIRE(spilled->state().dump() == resident->state().dump());
    if (!example.expected.is_null()) REQUIRE(spilled->state().at("bin_edges").at("x") == example.expected);
    require_binning_bits(spilled->transform(input), resident->transform(input));
  }
}

TEST_CASE("spilled binning failures retain validation and remove temporary data") {
  const fft::Schema schema{{{"x", "f64"}}};
  const auto prefix = make_batch({"x"}, {f64_column({1.0, 2.0, 3.0, 4.0})});
  const auto invalid = make_batch({"x"}, {f64_column({std::numeric_limits<double>::infinity()})});
  const auto missing = make_batch({"x"}, {numeric_column<arrow::DoubleType>({std::nullopt})});
  for (const std::string id : {"robust_scale", "quantile_transform", "k_bins_discretize"}) {
    CAPTURE(id);
    const nlohmann::json params{{"columns", {"x"}}};
    REQUIRE_THROWS_WITH(fit_binning_with_memory(id, params, schema, {prefix, invalid}, 0, true),
                        ContainsSubstring("is not finite"));
  }
  for (const std::string strategy : {"quantile", "kmeans"}) {
    const nlohmann::json params{{"columns", {"x"}}, {"n_bins", 2}, {"encode", "ordinal"}, {"strategy", strategy}};
    REQUIRE_THROWS_WITH(fit_binning_with_memory("k_bins_discretize", params, schema, {prefix, missing}, 0, true),
                        ContainsSubstring("has nulls"));
  }
  REQUIRE_THROWS_WITH(
      fit_binning_with_memory("k_bins_discretize", {{"columns", {"x"}}, {"n_bins", 5}, {"strategy", "kmeans"}}, schema,
                              {prefix}, 0, true),
      ContainsSubstring("fewer than the 5 clusters"));
}

TEST_CASE("disabled RobustScaler retains fit validation without buffering") {
  const nlohmann::json params{{"columns", {"x"}}, {"with_centering", false}, {"with_scaling", false}};
  const fft::Schema schema{{{"x", "f64"}}};
  REQUIRE_THROWS_WITH(fit_binning("robust_scale", params, schema,
                                  {make_batch({"x"}, {numeric_column<arrow::DoubleType>({std::nullopt})})}),
                      ContainsSubstring("has no non-null values"));
  REQUIRE_THROWS_WITH(fit_binning("robust_scale", params, schema,
                                  {make_batch({"x"}, {f64_column({std::numeric_limits<double>::infinity()})})}),
                      ContainsSubstring("is not finite"));
}

TEST_CASE("paired quantile interpolation retains mirrored searches at duplicate and extreme bounds") {
  const double maximum = std::numeric_limits<double>::max();
  const auto interpolate = [](double x, const std::vector<double>& grid, const std::vector<double>& references) {
    if (grid.size() == 1 || x < grid.front()) return references.front();
    if (x > grid.back()) return references.back();
    const auto upper = std::upper_bound(grid.begin(), grid.end(), x);
    const std::size_t lower = static_cast<std::size_t>(upper - grid.begin()) - 1;
    if (lower + 1 == grid.size() || grid[lower] == x) return references[lower];
    const double slope = (references[lower + 1] - references[lower]) / (grid[lower + 1] - grid[lower]);
    double result = std::fma(slope, x - grid[lower], references[lower]);
    if (std::isnan(result)) {
      result = std::fma(slope, x - grid[lower + 1], references[lower + 1]);
      if (std::isnan(result) && references[lower] == references[lower + 1]) result = references[lower];
    }
    return result;
  };
  // A grid of the default 1,000 points in tied runs: its search halves a
  // window of an odd count at several depths.
  std::vector<double> long_grid;
  for (int i = 0; i < 1000; ++i) long_grid.push_back(std::ldexp(static_cast<double>(i / 3 - 166), -2));
  const std::vector<std::vector<double>> grids{
      {-maximum, maximum}, {-maximum, -maximum, -1.0, -0.0, 0.0, 1.0, 1.0, maximum, maximum}, {0.0}, long_grid};
  const std::vector<double> probes{-maximum,
                                   -maximum / 2,
                                   -1.0,
                                   std::nextafter(-1.0, 0.0),
                                   -0.0,
                                   0.0,
                                   std::nextafter(0.0, 1.0),
                                   std::nextafter(1.0, 0.0),
                                   1.0,
                                   maximum / 2,
                                   maximum};
  for (const auto& grid : grids) {
    std::vector<double> values = probes;
    for (const double point : grid) {
      values.push_back(point);
      values.push_back(std::nextafter(point, -maximum));
      values.push_back(std::nextafter(point, maximum));
    }
    std::vector<double> references(grid.size(), 0.0);
    if (grid.size() > 1) {
      const double step = 1.0 / static_cast<double>(grid.size() - 1);
      for (std::size_t i = 0; i < grid.size(); ++i) references[i] = static_cast<double>(i) * step;
      references.back() = 1.0;
    }
    std::vector<double> mirrored_grid;
    std::vector<double> mirrored_references;
    for (auto it = grid.rbegin(); it != grid.rend(); ++it) mirrored_grid.push_back(-*it);
    for (auto it = references.rbegin(); it != references.rend(); ++it) mirrored_references.push_back(-*it);
    auto kernel = fft::make_kernel("quantile_transform", {{"columns", {"x"}}, {"n_quantiles", grid.size()}});
    kernel->stage = "bin";
    kernel->load_state({{"columns", {"x"}}, {"references", references}, {"quantiles", {{"x", grid}}}});
    const fft::Batch output = kernel->transform(make_batch({"x"}, {f64_column(values)}));
    const auto& actual = static_cast<const arrow::DoubleArray&>(*output.cols[0]);
    for (std::size_t row = 0; row < values.size(); ++row) {
      const double value = values[row];
      double expected =
          0.5 * (interpolate(value, grid, references) - interpolate(-value, mirrored_grid, mirrored_references));
      if (value == grid.back()) expected = 1.0;
      if (value == grid.front()) expected = 0.0;
      REQUIRE(std::bit_cast<uint64_t>(actual.Value(static_cast<int64_t>(row))) == std::bit_cast<uint64_t>(expected));
    }
  }
}
