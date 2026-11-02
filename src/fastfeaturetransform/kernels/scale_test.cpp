// Explicit C++ tests for the five sklearn scalers: the params, the schema
// rewrite, the monoid merge (in both orders, byte for byte), the two
// transform paths against each other, and load_state.

#include <arrow/array.h>
#include <arrow/buffer.h>

#include <algorithm>
#include <array>
#include <bit>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <cmath>
#include <limits>
#include <string_view>
#include <type_traits>

#include "fastfeaturetransform/kernel.hpp"
#include "fastfeaturetransform/test_support.hpp"

namespace fft = fastfeaturetransform;
using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::Equals;
using fft::testing::make_batch;
using fft::testing::numeric_column;

namespace {

using Doubles = std::vector<std::optional<double>>;

std::shared_ptr<arrow::Array> doubles(const Doubles& values) { return numeric_column<arrow::DoubleType>(values); }

// Fit in the given batch order, normally with one worker state per batch.
std::unique_ptr<fft::Kernel> fit_scaler(const std::string& id, const nlohmann::json& params, fft::Schema schema,
                                        const std::vector<fft::Batch>& batches, bool one_state_per_batch = true) {
  auto kernel = fft::make_kernel(id, params);
  kernel->stage = "scale";
  kernel->resolve(schema);
  std::vector<std::unique_ptr<fft::AccState>> states;
  for (const fft::Batch& batch : batches) {
    if (one_state_per_batch || states.empty()) states.push_back(kernel->make_state());
    kernel->consume(*states.back(), batch);
  }
  kernel->finalize(std::move(states));
  return kernel;
}

// One transformed f64 column, nulls as nullopt.
Doubles column_of(const fft::Batch& batch, const std::string& name) {
  const int idx = batch.find(name);
  REQUIRE(idx >= 0);
  const auto& column = static_cast<const arrow::DoubleArray&>(*batch.cols[static_cast<std::size_t>(idx)]);
  Doubles values;
  for (int64_t row = 0; row < column.length(); ++row) {
    values.push_back(column.IsNull(row) ? std::optional<double>{} : std::optional<double>{column.Value(row)});
  }
  return values;
}

// One transformed i64 column (binarize's output).
std::vector<int64_t> integers_of(const fft::Batch& batch, const std::string& name) {
  const int idx = batch.find(name);
  REQUIRE(idx >= 0);
  const auto& column = static_cast<const arrow::Int64Array&>(*batch.cols[static_cast<std::size_t>(idx)]);
  std::vector<int64_t> values;
  for (int64_t row = 0; row < column.length(); ++row) values.push_back(column.Value(row));
  return values;
}

// Every value of a one-column batch through the record path, which must
// answer exactly what the batch path did.
void record_agrees(const fft::Kernel& kernel, const std::string& name, const Doubles& source, const Doubles& scaled) {
  for (std::size_t i = 0; i < source.size(); ++i) {
    const fft::RowValue value = source[i].has_value() ? fft::RowValue(*source[i]) : fft::RowValue(std::monostate{});
    const fft::Row row = kernel.transform_row({{name, value}});
    REQUIRE(row.size() == 1);
    if (scaled[i].has_value()) {
      REQUIRE(std::get<double>(row[0].second) == *scaled[i]);
    } else {
      REQUIRE(std::holds_alternative<std::monostate>(row[0].second));
    }
  }
}

template <typename ArrowType>
void check_standard_fit_blocks() {
  using Scalar = typename ArrowType::c_type;
  constexpr bool wide = std::is_same_v<Scalar, double>;
  const fft::Schema schema{{{"a", wide ? "f64" : "f32"}}};
  const nlohmann::json params{{"columns", {"a"}}};
  const Scalar one = 1;
  const Scalar large = std::ldexp(one, wide ? 400 : 100);
  const Scalar small = std::ldexp(one, wide ? -200 : -60);
  const Scalar tiny_square = std::ldexp(one, wide ? -537 : -70);
  const Scalar next = std::nextafter(one, Scalar{2});
  const Scalar denormal = std::numeric_limits<Scalar>::denorm_min();
  const std::vector<std::pair<std::string, std::vector<Scalar>>> samples{
      {"cancellation", {large, one, -large, small, -small, -one, next, -next, denormal, -denormal}},
      {"near constant", {one, next, std::nextafter(next, Scalar{2}), std::nextafter(one, Scalar{0})}},
      {"tiny squares", {tiny_square, -tiny_square, std::nextafter(tiny_square, one), denormal, -denormal}},
      {"signed zeros", {Scalar{0}, -Scalar{0}}}};
  for (const auto& [label, probes] : samples) {
    for (int64_t length : {1, 63, 64, 65, 129, 257}) {
      CAPTURE(wide, label, length);
      // Nonfinite guards surround an offset slice. The nullable reference
      // holds the same values in the same order plus a null, which forces the
      // scalar visitor without teaching it an extra value.
      std::vector<Scalar> backing(static_cast<std::size_t>(length + 7), std::numeric_limits<Scalar>::quiet_NaN());
      std::vector<std::optional<Scalar>> reference_values;
      for (int64_t row = 0; row < length; ++row) {
        const Scalar value = probes[static_cast<std::size_t>(row) % probes.size()];
        backing[static_cast<std::size_t>(row + 3)] = value;
        reference_values.push_back(value);
      }
      reference_values.push_back(std::nullopt);
      const auto storage = arrow::Buffer::FromVector(std::move(backing));
      const std::vector<uint8_t> before(storage->data(), storage->data() + storage->size());
      const auto parent = std::make_shared<arrow::NumericArray<ArrowType>>(length + 7, storage, nullptr, 0);
      const auto source = parent->Slice(3, length);
      const fft::Batch dense = make_batch({"a"}, {source});
      const auto reference = fit_scaler("standard_scale", params, schema,
                                        {make_batch({"a"}, {numeric_column<ArrowType>(reference_values)})});
      const auto expected = reference->state();
      const auto expected_output = column_of(reference->transform(dense), "a");

      std::vector<fft::Batch> parts;
      const std::array<int64_t, 5> widths{1, 62, 1, 64, 3};
      for (int64_t first = 0; first < length;) {
        const int64_t count = std::min(widths[parts.size() % widths.size()], length - first);
        parts.push_back(make_batch({"a"}, {source->Slice(first, count)}));
        first += count;
      }
      const auto check = [&](const std::vector<fft::Batch>& batches, bool one_state_per_batch) {
        const auto fitted = fit_scaler("standard_scale", params, schema, batches, one_state_per_batch);
        const auto state = fitted->state();
        REQUIRE(state.dump() == expected.dump());
        for (const char* statistic : {"means", "variances", "scales"}) {
          REQUIRE(std::bit_cast<uint64_t>(state.at(statistic).at("a").template get<double>()) ==
                  std::bit_cast<uint64_t>(expected.at(statistic).at("a").template get<double>()));
        }
        const auto output = column_of(fitted->transform(dense), "a");
        for (std::size_t row = 0; row < output.size(); ++row) {
          REQUIRE(std::bit_cast<uint64_t>(*output[row]) == std::bit_cast<uint64_t>(*expected_output[row]));
        }
      };
      check({dense}, true);
      check(parts, false);  // the same accumulator carries across small batches
      check(parts, true);
      std::reverse(parts.begin(), parts.end());
      check(parts, true);
      REQUIRE(std::vector<uint8_t>(storage->data(), storage->data() + storage->size()) == before);
    }
  }
}

// A value no scaler takes, spelled as the message writes it: spelled out, not
// computed, so that a change in the formatting shows. x86 makes the NaN with
// its sign bit set for 0/0, so Parquet written there carries it.
struct NonFinite {
  double value;
  const char* spelled;
};

std::vector<NonFinite> non_finite_values() {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  return {{nan, "nan"}, {std::copysign(nan, -1.0), "-nan"}, {inf, "inf"}, {-inf, "-inf"}};
}

// binarize and normalize refuse a null too, so only the other scalers suggest
// writing a missing value as one.
std::string refusal(const std::string& id, const std::string& column, const char* spelled) {
  const bool null_passes = !id.starts_with("binarize") && !id.starts_with("normalize");
  return "scale: column '" + column + "': value " + spelled +
         (null_passes ? " is not finite; write a missing value as a null" : " is not finite");
}

std::unique_ptr<fft::Kernel> loaded_scaler(const std::string& id, const nlohmann::json& params,
                                           const nlohmann::json& state) {
  auto kernel = fft::make_kernel(id, params);
  kernel->stage = "scale";
  kernel->load_state(state);
  return kernel;
}

std::unique_ptr<fft::Kernel> resolved_scaler(const std::string& id, const nlohmann::json& params) {
  auto kernel = fft::make_kernel(id, params);
  kernel->stage = "scale";
  fft::Schema schema{{{"x", "f64"}, {"y", "f64"}}};
  kernel->resolve(schema);
  return kernel;
}

// Every scaler in this file, each reading column 'y' (normalize reads x, then y).
std::vector<std::pair<std::string, std::unique_ptr<fft::Kernel>>> every_scaler() {
  std::vector<std::pair<std::string, std::unique_ptr<fft::Kernel>>> out;
  out.emplace_back("standard_scale", loaded_scaler("standard_scale", {{"columns", {"y"}}},
                                                   {{"columns", {"y"}},
                                                    {"means", {{"y", 0.1}}},
                                                    {"variances", {{"y", 9.0}}},
                                                    {"scales", {{"y", 3.0}}},
                                                    {"counts", {{"y", 32}}}}));
  out.emplace_back("min_max_scale", loaded_scaler("min_max_scale", {{"columns", {"y"}}, {"feature_range", {-2.0, 3.0}}},
                                                  {{"columns", {"y"}},
                                                   {"data_min", {{"y", 0.0}}},
                                                   {"data_max", {{"y", 5.0}}},
                                                   {"scales", {{"y", 1.0}}},
                                                   {"mins", {{"y", -2.0}}},
                                                   {"rows", 32}}));
  out.emplace_back(
      "max_abs_scale",
      loaded_scaler("max_abs_scale", {{"columns", {"y"}}},
                    {{"columns", {"y"}}, {"max_abs", {{"y", 3.0}}}, {"scales", {{"y", 3.0}}}, {"rows", 32}}));
  out.emplace_back("binarize", resolved_scaler("binarize", {{"columns", {"y"}}, {"threshold", 0.5}}));
  for (const char* norm : {"l1", "l2", "max"}) {
    out.emplace_back(std::string("normalize/") + norm,
                     resolved_scaler("normalize", {{"columns", {"x", "y"}}, {"norm", norm}}));
  }
  return out;
}

}  // namespace

TEST_CASE("the scalers validate their params") {
  REQUIRE_THROWS_WITH((fft::make_kernel("standard_scale", {{"with_mean", "yes"}})),
                      ContainsSubstring("'with_mean' must be true or false"));
  REQUIRE_THROWS_WITH((fft::make_kernel("standard_scale", {{"with_std", 1}})),
                      ContainsSubstring("'with_std' must be true or false"));
  REQUIRE_THROWS_WITH((fft::make_kernel("standard_scale", {{"columns", {"a", "a"}}})),
                      ContainsSubstring("listed more than once"));

  const std::string range = "'feature_range' must be two finite numbers";
  REQUIRE_THROWS_WITH((fft::make_kernel("min_max_scale", {{"feature_range", 1}})), ContainsSubstring(range));
  REQUIRE_THROWS_WITH((fft::make_kernel("min_max_scale", {{"feature_range", {0, 1, 2}}})), ContainsSubstring(range));
  REQUIRE_THROWS_WITH((fft::make_kernel("min_max_scale", {{"feature_range", {1, 0}}})), ContainsSubstring(range));
  REQUIRE_THROWS_WITH((fft::make_kernel("min_max_scale", {{"feature_range", {0, 0}}})), ContainsSubstring(range));
  REQUIRE_THROWS_WITH((fft::make_kernel("min_max_scale", {{"clip", "yes"}})),
                      ContainsSubstring("'clip' must be true or false"));
  REQUIRE_THROWS_WITH((fft::make_kernel("max_abs_scale", {{"clip", 1}})),
                      ContainsSubstring("'clip' must be true or false"));

  // The two stateless kernels learn nothing, so they must name their columns:
  // an artifact that did not would touch a different set for every file.
  REQUIRE_THROWS_WITH((fft::make_kernel("binarize", nlohmann::json::object())),
                      ContainsSubstring("'columns' must be a non-empty list"));
  REQUIRE_THROWS_WITH((fft::make_kernel("binarize", {{"columns", {"a"}}, {"threshold", "0"}})),
                      ContainsSubstring("'threshold' must be a finite number"));
  REQUIRE_THROWS_WITH((fft::make_kernel("normalize", nlohmann::json::object())),
                      ContainsSubstring("'columns' must be a non-empty list"));
  REQUIRE_THROWS_WITH((fft::make_kernel("normalize", {{"columns", {"a"}}, {"norm", "l3"}})),
                      ContainsSubstring("'norm' must be 'l1', 'l2' or 'max'"));

  REQUIRE(fft::make_kernel("standard_scale", nlohmann::json::object())->needs_fit());
  REQUIRE(fft::make_kernel("min_max_scale", nlohmann::json::object())->needs_fit());
  REQUIRE(fft::make_kernel("max_abs_scale", nlohmann::json::object())->needs_fit());
  REQUIRE_FALSE(fft::make_kernel("binarize", {{"columns", {"a"}}})->needs_fit());
  REQUIRE_FALSE(fft::make_kernel("normalize", {{"columns", {"a"}}})->needs_fit());
  REQUIRE(fft::make_kernel("normalize", {{"columns", {"a"}}})->params().at("norm") == "l2");
  REQUIRE(fft::make_kernel("min_max_scale", nlohmann::json::object())->params().at("feature_range") ==
          nlohmann::json::array({0.0, 1.0}));
}

TEST_CASE("the scalers rewrite the column type and refuse a column they cannot read") {
  auto standard = fft::make_kernel("standard_scale", nlohmann::json::object());
  standard->stage = "scale";
  fft::Schema schema{{{"a", "f32"}, {"n", "i16"}, {"c", "utf8"}}};
  standard->resolve(schema);
  // columns=null takes every integer or float column, and every one of them
  // comes out f64 whatever width it arrived as.
  REQUIRE(schema.cols == fft::Schema{{{"a", "f64"}, {"n", "f64"}, {"c", "utf8"}}}.cols);
  REQUIRE(standard->inputs == std::vector<std::string>{"a", "n"});

  auto binarizer = fft::make_kernel("binarize", {{"columns", {"a"}}});
  binarizer->stage = "scale";
  fft::Schema binary{{{"a", "f64"}, {"c", "utf8"}}};
  binarizer->resolve(binary);
  REQUIRE(binary.cols == fft::Schema{{{"a", "i64"}, {"c", "utf8"}}}.cols);

  auto text = fft::make_kernel("min_max_scale", {{"columns", {"c"}}});
  text->stage = "scale";
  fft::Schema strings{{{"c", "utf8"}}};
  REQUIRE_THROWS_WITH(text->resolve(strings),
                      ContainsSubstring("column 'c' is utf8, expected an integer or float column (cast it first)"));

  auto missing = fft::make_kernel("max_abs_scale", {{"columns", {"gone"}}});
  missing->stage = "scale";
  fft::Schema absent{{{"a", "f64"}}};
  REQUIRE_THROWS_WITH(missing->resolve(absent), ContainsSubstring("column 'gone' not found in input"));

  auto nothing = fft::make_kernel("standard_scale", nlohmann::json::object());
  nothing->stage = "scale";
  fft::Schema categorical{{{"c", "utf8"}}};
  REQUIRE_THROWS_WITH(nothing->resolve(categorical), ContainsSubstring("no integer or float columns to scale"));
}

TEST_CASE("standard_scale merges worker sums in either order and both paths agree") {
  const nlohmann::json params{{"columns", {"a"}}};
  const fft::Schema schema{{{"a", "f64"}}};
  const fft::Batch first = make_batch({"a"}, {doubles({1.0, 2.0, std::nullopt})});
  const fft::Batch second = make_batch({"a"}, {doubles({3.0, 4.0, 5.0})});
  auto kernel = fit_scaler("standard_scale", params, schema, {first, second});
  const nlohmann::json state = kernel->state();
  REQUIRE(state.at("means").at("a") == 3.0);
  REQUIRE(state.at("variances").at("a") == 2.0);  // population variance of 1..5
  REQUIRE(state.at("counts").at("a") == 5);       // the null is not counted
  // Only the merged accumulators may decide the table: this is the monoid
  // contract, and the artifact is byte-identical under either split.
  REQUIRE(fit_scaler("standard_scale", params, schema, {second, first})->state().dump() == state.dump());

  const Doubles values{1.0, 3.0, 5.0, std::nullopt};
  const fft::Batch out = kernel->transform(make_batch({"a"}, {doubles(values)}));
  const Doubles scaled = column_of(out, "a");
  const double scale = std::sqrt(2.0);
  REQUIRE(scaled[0] == (1.0 - 3.0) / scale);
  REQUIRE(scaled[1] == 0.0);
  REQUIRE(scaled[3] == std::nullopt);  // a null passes through as a null
  record_agrees(*kernel, "a", values, scaled);

  REQUIRE_THROWS_WITH(kernel->transform_row({{"a", std::string("x")}}), ContainsSubstring("must be a number or null"));
  REQUIRE_THROWS_WITH(kernel->transform_row({{"b", 1.0}}), ContainsSubstring("key 'a' missing from record"));
  // An integer record value is a number, and reads as the same one.
  REQUIRE(std::get<double>(kernel->transform_row({{"a", int64_t{1}}})[0].second) == *scaled[0]);
}

TEST_CASE("standard_scale answers a constant column with zeros, not NaN") {
  const fft::Schema schema{{{"a", "f64"}, {"b", "f64"}}};
  const fft::Batch batch = make_batch({"a", "b"}, {doubles({7.0, 7.0, 7.0}), doubles({0.0, 0.0, 0.0})});
  auto kernel = fit_scaler("standard_scale", {{"columns", {"a", "b"}}}, schema, {batch});
  REQUIRE(kernel->state().at("scales").at("a") == 1.0);  // sklearn's near-constant rule, not sqrt(0)
  REQUIRE(kernel->state().at("scales").at("b") == 1.0);
  const fft::Batch out = kernel->transform(batch);
  REQUIRE(column_of(out, "a") == Doubles{0.0, 0.0, 0.0});
  REQUIRE(column_of(out, "b") == Doubles{0.0, 0.0, 0.0});

  // A column of nothing but nulls has no statistics to learn.
  REQUIRE_THROWS_WITH(fit_scaler("standard_scale", {{"columns", {"a"}}}, fft::Schema{{{"a", "f64"}}},
                                 {make_batch({"a"}, {doubles({std::nullopt, std::nullopt})})}),
                      ContainsSubstring("column 'a' has no non-null values to learn from"));

  // REGRESSION: a constant column of tiny values fits variance 0 and scale 1,
  // as sklearn does. Below about 1e-146 the exact sums round their
  // residuals, and 1e-158 used to fit -5e-324, which load_state refuses,
  // while 9e-159 fit +5e-324 and scaled by its square root, 2.2e-162.
  for (const double value : {1e-158, 9e-159}) {
    CAPTURE(value);
    auto tiny = fit_scaler("standard_scale", {{"columns", {"a"}}}, fft::Schema{{{"a", "f64"}}},
                           {make_batch({"a"}, {doubles({value, value, value, value})})});
    REQUIRE(tiny->state().at("variances").at("a") == 0.0);
    REQUIRE(tiny->state().at("scales").at("a") == 1.0);
    REQUIRE_NOTHROW(loaded_scaler("standard_scale", tiny->params(), tiny->state()));
  }
  // A tiny column that is not constant keeps the variance its sums give,
  // which here is sklearn's to the bit: 4 subnormal steps.
  auto spread = fit_scaler("standard_scale", {{"columns", {"a"}}}, fft::Schema{{{"a", "f64"}}},
                           {make_batch({"a"}, {doubles({0.0, 9e-162})})});
  REQUIRE(spread->state().at("variances").at("a") == std::ldexp(4.0, -1074));
  REQUIRE(spread->state().at("scales").at("a") == 4.445517498970155e-162);
}

TEST_CASE("standard_scale leaves out the centring or the division on request") {
  const fft::Schema schema{{{"a", "f64"}}};
  const fft::Batch batch = make_batch({"a"}, {doubles({1.0, 2.0, 3.0, 4.0, 5.0})});
  const fft::Batch probe = make_batch({"a"}, {doubles({5.0})});
  auto centred = fit_scaler("standard_scale", {{"columns", {"a"}}, {"with_std", false}}, schema, {batch});
  REQUIRE(column_of(centred->transform(probe), "a") == Doubles{2.0});
  auto divided = fit_scaler("standard_scale", {{"columns", {"a"}}, {"with_mean", false}}, schema, {batch});
  REQUIRE(column_of(divided->transform(probe), "a") == Doubles{5.0 / std::sqrt(2.0)});
  auto neither =
      fit_scaler("standard_scale", {{"columns", {"a"}}, {"with_mean", false}, {"with_std", false}}, schema, {batch});
  REQUIRE(column_of(neither->transform(probe), "a") == Doubles{5.0});
  // The statistics are recorded whichever flags are set.
  REQUIRE(neither->state().at("means").at("a") == 3.0);
}

TEST_CASE("standard_scale block products preserve scalar fit bits across slices and worker partitions") {
  check_standard_fit_blocks<arrow::DoubleType>();
  check_standard_fit_blocks<arrow::FloatType>();
}

TEST_CASE("standard_scale block precomputation preserves first invalid and overflowing row errors") {
  const nlohmann::json params{{"columns", {"a"}}};
  const fft::Schema schema{{{"a", "f64"}}};
  for (int64_t first_bad : {0, 63, 64, 65, 128}) {
    for (bool overflow_first : {false, true}) {
      CAPTURE(first_bad, overflow_first);
      Doubles values(131, 0.0);
      values[static_cast<std::size_t>(first_bad)] = overflow_first ? 1e200 : std::numeric_limits<double>::infinity();
      values[static_cast<std::size_t>(first_bad + 1)] =
          overflow_first ? std::numeric_limits<double>::quiet_NaN() : 1e200;
      const std::string error = overflow_first ? "sums past the range of f64" : "is not finite";
      const auto dense = doubles(values);
      values.push_back(std::nullopt);
      const auto scalar = doubles(values);
      for (const auto& input : {dense, scalar}) {
        REQUIRE_THROWS_WITH(fit_scaler("standard_scale", params, schema, {make_batch({"a"}, {input})}),
                            ContainsSubstring("scale: column 'a'") && ContainsSubstring(error));
      }
    }
  }

  // Each square is finite; adding the fourth overflows immediately before
  // the next block's NaN. Precomputing that NaN must not change the error.
  Doubles values(66, 0.0);
  constexpr double magnitude = 0x1p511;
  values[60] = magnitude;
  values[61] = -magnitude;
  values[62] = magnitude;
  values[63] = -magnitude;
  values[64] = std::numeric_limits<double>::quiet_NaN();
  const auto dense = doubles(values);
  values.push_back(std::nullopt);
  for (const auto& input : {dense, doubles(values)}) {
    REQUIRE_THROWS_WITH(fit_scaler("standard_scale", params, schema, {make_batch({"a"}, {input})}),
                        ContainsSubstring("scale: column 'a' sums past the range of f64"));
  }
}

TEST_CASE("standard_scale f32 block validation includes boundaries and tails") {
  const nlohmann::json params{{"columns", {"a"}}};
  const fft::Schema schema{{{"a", "f32"}}};
  for (int64_t position : {0, 63, 64, 65, 128}) {
    for (float invalid : {std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
      CAPTURE(position, invalid);
      std::vector<std::optional<float>> values(129, 0.0F);
      values[static_cast<std::size_t>(position)] = invalid;
      const auto dense = numeric_column<arrow::FloatType>(values);
      values.push_back(std::nullopt);
      for (const auto& input : {dense, numeric_column<arrow::FloatType>(values)}) {
        REQUIRE_THROWS_WITH(fit_scaler("standard_scale", params, schema, {make_batch({"a"}, {input})}),
                            ContainsSubstring("scale: column 'a': value") && ContainsSubstring("is not finite"));
      }
    }
  }
}

TEST_CASE("min_max_scale maps the fitted range onto the feature range and clips on request") {
  const nlohmann::json params{{"columns", {"a"}}, {"feature_range", {-1.0, 1.0}}};
  const fft::Schema schema{{{"a", "f64"}}};
  const fft::Batch first = make_batch({"a"}, {doubles({2.0, std::nullopt})});
  const fft::Batch second = make_batch({"a"}, {doubles({6.0, 4.0})});
  auto kernel = fit_scaler("min_max_scale", params, schema, {first, second});
  const nlohmann::json state = kernel->state();
  REQUIRE(state.at("data_min").at("a") == 2.0);
  REQUIRE(state.at("data_max").at("a") == 6.0);
  REQUIRE(state.at("scales").at("a") == 0.5);
  REQUIRE(state.at("mins").at("a") == -2.0);
  REQUIRE(state.at("rows") == 4);  // rows, not values: the null is one of them
  REQUIRE(fit_scaler("min_max_scale", params, schema, {second, first})->state().dump() == state.dump());

  const Doubles values{2.0, 4.0, 6.0, 10.0, std::nullopt};
  const Doubles scaled = column_of(kernel->transform(make_batch({"a"}, {doubles(values)})), "a");
  REQUIRE(scaled == Doubles{-1.0, 0.0, 1.0, 3.0, std::nullopt});  // 10 is outside the fitted range
  record_agrees(*kernel, "a", values, scaled);

  nlohmann::json clipped_params = params;
  clipped_params["clip"] = true;
  auto clipped = fit_scaler("min_max_scale", clipped_params, schema, {first, second});
  const Doubles clamped = column_of(clipped->transform(make_batch({"a"}, {doubles(values)})), "a");
  REQUIRE(clamped == Doubles{-1.0, 0.0, 1.0, 1.0, std::nullopt});
  record_agrees(*clipped, "a", values, clamped);

  // A constant column divides by 1, so it lands on the low end of the range.
  auto flat = fit_scaler("min_max_scale", params, schema, {make_batch({"a"}, {doubles({5.0, 5.0})})});
  REQUIRE(flat->state().at("scales").at("a") == 2.0);  // (1 - -1) / 1
  REQUIRE(column_of(flat->transform(make_batch({"a"}, {doubles({5.0})})), "a") == Doubles{-1.0});
}

TEST_CASE("min_max_scale refuses a fit whose scale would be 0") {
  // REGRESSION: a fit refuses a scale of 0. A column spanning past the range
  // of f64 used to fit one, mapping every value to the low end, and its saved
  // pipeline then failed to load.
  const fft::Schema schema{{{"a", "f64"}}};
  const fft::Batch wide = make_batch({"a"}, {doubles({-1e308, 0.0, 1e308})});
  REQUIRE_THROWS_WITH(fit_scaler("min_max_scale", {{"columns", {"a"}}}, schema, {wide}),
                      Equals("scale: column 'a' spans past the range of f64"));
  // A finite span of 1e308 under a feature range 2^-52 wide: the scale,
  // about 2e-324, rounds to 0.
  const fft::Batch tall = make_batch({"a"}, {doubles({0.0, 1e308})});
  const nlohmann::json narrow{{"columns", {"a"}}, {"feature_range", {1.0, 1.0 + std::ldexp(1.0, -52)}}};
  REQUIRE_THROWS_WITH(fit_scaler("min_max_scale", narrow, schema, {tall}),
                      Equals("scale: column 'a' fits a scale that rounds to 0; widen 'feature_range'"));
}

TEST_CASE("max_abs_scale divides by the largest magnitude and clips on request") {
  const nlohmann::json params{{"columns", {"a"}}};
  const fft::Schema schema{{{"a", "f64"}}};
  const fft::Batch first = make_batch({"a"}, {doubles({1.0, -8.0})});
  const fft::Batch second = make_batch({"a"}, {doubles({2.0, std::nullopt})});
  auto kernel = fit_scaler("max_abs_scale", params, schema, {first, second});
  const nlohmann::json state = kernel->state();
  REQUIRE(state.at("max_abs").at("a") == 8.0);  // the magnitude of the smallest value
  REQUIRE(state.at("scales").at("a") == 8.0);
  REQUIRE(state.at("rows") == 4);
  REQUIRE(fit_scaler("max_abs_scale", params, schema, {second, first})->state().dump() == state.dump());

  const Doubles values{-8.0, 0.0, 4.0, 16.0, std::nullopt};
  const Doubles scaled = column_of(kernel->transform(make_batch({"a"}, {doubles(values)})), "a");
  REQUIRE(scaled == Doubles{-1.0, 0.0, 0.5, 2.0, std::nullopt});
  record_agrees(*kernel, "a", values, scaled);

  auto clipped = fit_scaler("max_abs_scale", {{"columns", {"a"}}, {"clip", true}}, schema, {first, second});
  const Doubles clamped = column_of(clipped->transform(make_batch({"a"}, {doubles(values)})), "a");
  REQUIRE(clamped == Doubles{-1.0, 0.0, 0.5, 1.0, std::nullopt});
  record_agrees(*clipped, "a", values, clamped);

  // An all-zero column divides by 1, so it stays all zeros rather than NaN.
  auto zeros = fit_scaler("max_abs_scale", params, schema, {make_batch({"a"}, {doubles({0.0, 0.0})})});
  REQUIRE(zeros->state().at("scales").at("a") == 1.0);
  REQUIRE(column_of(zeros->transform(make_batch({"a"}, {doubles({0.0})})), "a") == Doubles{0.0});
}

TEST_CASE("binarize is strictly above the threshold and refuses a null") {
  auto kernel = fft::make_kernel("binarize", {{"columns", {"a"}}, {"threshold", 0.5}});
  kernel->stage = "scale";
  fft::Schema schema{{{"a", "f64"}}};
  kernel->resolve(schema);
  const Doubles values{0.0, 0.5, 0.5000000000000001, 1.0, -1.0};
  const fft::Batch out = kernel->transform(make_batch({"a"}, {doubles(values)}));
  const std::vector<int64_t> flags = integers_of(out, "a");
  REQUIRE(flags == std::vector<int64_t>{0, 0, 1, 1, 0});  // exactly on the threshold is a 0
  for (std::size_t i = 0; i < values.size(); ++i) {
    REQUIRE(std::get<int64_t>(kernel->transform_row({{"a", *values[i]}})[0].second) == flags[i]);
  }
  // sklearn refuses a NaN here rather than guessing, so a null is an error on
  // both paths rather than a silent 0 or 1.
  REQUIRE_THROWS_WITH(kernel->transform(make_batch({"a"}, {doubles({0.0, std::nullopt})})),
                      ContainsSubstring("column 'a' has a null, which binarize has no answer for"));
  REQUIRE_THROWS_WITH(kernel->transform_row({{"a", fft::RowValue(std::monostate{})}}),
                      ContainsSubstring("column 'a' has a null, which binarize has no answer for"));
  REQUIRE_THROWS_WITH(kernel->transform_row({{"b", 1.0}}), ContainsSubstring("key 'a' missing from record"));
  // Nothing is learned, so nothing a split could change.
  REQUIRE(kernel->state().empty());
}

TEST_CASE("normalize divides a row by its norm over the listed columns") {
  auto kernel = fft::make_kernel("normalize", {{"columns", {"x", "y"}}});
  kernel->stage = "scale";
  fft::Schema schema{{{"x", "f64"}, {"y", "f64"}, {"z", "f64"}}};
  kernel->resolve(schema);
  REQUIRE(schema.cols == fft::Schema{{{"x", "f64"}, {"y", "f64"}, {"z", "f64"}}}.cols);

  const fft::Batch out = kernel->transform(
      make_batch({"x", "y", "z"}, {doubles({3.0, 0.0, -2.0}), doubles({4.0, 0.0, 2.0}), doubles({9.0, 9.0, 9.0})}));
  REQUIRE(column_of(out, "x") == Doubles{0.6, 0.0, -2.0 / std::sqrt(8.0)});
  REQUIRE(column_of(out, "y") == Doubles{0.8, 0.0, 2.0 / std::sqrt(8.0)});
  REQUIRE(column_of(out, "z") == Doubles{9.0, 9.0, 9.0});  // not listed, so not touched

  // The record path answers exactly what the batch path did.
  const fft::Row row = kernel->transform_row({{"x", 3.0}, {"y", 4.0}, {"z", 9.0}});
  REQUIRE(std::get<double>(row[0].second) == 0.6);
  REQUIRE(std::get<double>(row[1].second) == 0.8);
  REQUIRE(std::get<double>(row[2].second) == 9.0);
  const fft::Row zero = kernel->transform_row({{"x", 0.0}, {"y", 0.0}});
  REQUIRE(std::get<double>(zero[0].second) == 0.0);  // a zero row is left alone, not divided

  auto l1 = fft::make_kernel("normalize", {{"columns", {"x", "y"}}, {"norm", "l1"}});
  l1->stage = "scale";
  fft::Schema ones{{{"x", "f64"}, {"y", "f64"}}};
  l1->resolve(ones);
  REQUIRE(std::get<double>(l1->transform_row({{"x", 3.0}, {"y", -1.0}})[0].second) == 0.75);
  auto largest = fft::make_kernel("normalize", {{"columns", {"x", "y"}}, {"norm", "max"}});
  largest->stage = "scale";
  fft::Schema maxima{{{"x", "f64"}, {"y", "f64"}}};
  largest->resolve(maxima);
  REQUIRE(std::get<double>(largest->transform_row({{"x", 2.0}, {"y", -4.0}})[1].second) == -1.0);

  // Every listed column has to be there, and none of them may be null.
  REQUIRE_THROWS_WITH(kernel->transform_row({{"x", 1.0}}), ContainsSubstring("key 'y' missing from record"));
  REQUIRE_THROWS_WITH(kernel->transform_row({{"x", 1.0}, {"y", fft::RowValue(std::monostate{})}}),
                      ContainsSubstring("column 'y' has a null"));
  REQUIRE_THROWS_WITH(kernel->transform(make_batch({"x", "y"}, {doubles({1.0}), doubles({std::nullopt})})),
                      ContainsSubstring("column 'y' has a null"));
  REQUIRE(kernel->state().empty());
}

TEST_CASE("the scalers' load_state rejects a malformed table") {
  auto fresh = [](const std::string& id, const nlohmann::json& params) {
    auto kernel = fft::make_kernel(id, params);
    kernel->stage = "scale";
    return kernel;
  };
  const nlohmann::json standard{{"columns", {"a"}},
                                {"means", {{"a", 3.0}}},
                                {"variances", {{"a", 2.0}}},
                                {"scales", {{"a", 1.5}}},
                                {"counts", {{"a", 5}}}};
  REQUIRE_NOTHROW(fresh("standard_scale", {{"columns", {"a"}}})->load_state(standard));
  auto broken = [&](const std::string& id, const nlohmann::json& params, const nlohmann::json& good, auto mutate,
                    const std::string& message) {
    nlohmann::json state = good;
    mutate(state);
    REQUIRE_THROWS_WITH(fresh(id, params)->load_state(state), ContainsSubstring(message));
  };
  const nlohmann::json one{{"columns", {"a"}}};
  broken(
      "standard_scale", one, standard, [](nlohmann::json& s) { s.erase("means"); }, "state must hold a 'means' object");
  broken(
      "standard_scale", one, standard, [](nlohmann::json& s) { s["means"]["a"] = "three"; },
      "means['a'] must be a finite number");
  broken(
      "standard_scale", one, standard, [](nlohmann::json& s) { s["variances"]["a"] = -1.0; },
      "variances['a'] must not be negative");
  // A scale of zero would divide every row of the column by zero; the fit can
  // never produce one, so only a hand-edited manifest can carry it.
  broken(
      "standard_scale", one, standard, [](nlohmann::json& s) { s["scales"]["a"] = 0.0; },
      "scales['a'] must be a positive number");
  broken(
      "standard_scale", one, standard, [](nlohmann::json& s) { s["counts"]["a"] = 0; },
      "counts['a'] must be an integer >= 1");
  broken(
      "standard_scale", one, standard, [](nlohmann::json& s) { s.erase("counts"); },
      "state must hold a 'counts' object");

  const nlohmann::json range{{"columns", {"a"}},        {"data_min", {{"a", 1.0}}}, {"data_max", {{"a", 5.0}}},
                             {"scales", {{"a", 0.25}}}, {"mins", {{"a", -0.25}}},   {"rows", 4}};
  REQUIRE_NOTHROW(fresh("min_max_scale", one)->load_state(range));
  broken(
      "min_max_scale", one, range, [](nlohmann::json& s) { s["data_min"]["a"] = 9.0; },
      "data_min['a'] must not be above data_max['a']");
  broken("min_max_scale", one, range, [](nlohmann::json& s) { s.erase("mins"); }, "state must hold a 'mins' object");
  broken("min_max_scale", one, range, [](nlohmann::json& s) { s["rows"] = -1; }, "'rows' must be an integer >= 0");
  // The fit computes (high - low) / handle_zeros(data_max - data_min), which is
  // always strictly positive; its three siblings refuse the same edit, and a
  // zero here would collapse the column onto its offset while a negative one
  // would reverse it — both silently, for every row.
  broken(
      "min_max_scale", one, range, [](nlohmann::json& s) { s["scales"]["a"] = 0.0; },
      "scales['a'] must be a positive number");
  broken(
      "min_max_scale", one, range, [](nlohmann::json& s) { s["scales"]["a"] = -1.0; },
      "scales['a'] must be a positive number");

  const nlohmann::json magnitude{{"columns", {"a"}}, {"max_abs", {{"a", 8.0}}}, {"scales", {{"a", 8.0}}}, {"rows", 4}};
  REQUIRE_NOTHROW(fresh("max_abs_scale", one)->load_state(magnitude));
  broken(
      "max_abs_scale", one, magnitude, [](nlohmann::json& s) { s["max_abs"]["a"] = -1.0; },
      "max_abs['a'] must not be negative");
  broken(
      "max_abs_scale", one, magnitude, [](nlohmann::json& s) { s["scales"]["a"] = 0.0; },
      "scales['a'] must be a positive number");

  // A loaded scaler that named no columns takes them, in order, from the
  // 'columns' list its own state carries — never from its table's key order,
  // which comes back alphabetical whatever the fit resolved.
  auto derived = fresh("max_abs_scale", nlohmann::json::object());
  REQUIRE_NOTHROW(derived->load_state(magnitude));
  REQUIRE(derived->inputs == std::vector<std::string>{"a"});
  REQUIRE(std::get<double>(derived->transform_row({{"a", 4.0}})[0].second) == 0.5);
  auto pair = fresh("max_abs_scale", nlohmann::json::object());
  REQUIRE_NOTHROW(pair->load_state({{"columns", {"z", "a"}},
                                    {"max_abs", {{"a", 8.0}, {"z", 8.0}}},
                                    {"scales", {{"a", 8.0}, {"z", 8.0}}},
                                    {"rows", 4}}));
  REQUIRE(pair->inputs == std::vector<std::string>{"z", "a"});
  broken(
      "max_abs_scale", one, magnitude, [](nlohmann::json& s) { s.erase("columns"); },
      "state must hold a 'columns' list");

  // The two stateless kernels learn nothing, so a table for them is an error.
  REQUIRE_NOTHROW(fresh("binarize", one)->load_state(nlohmann::json::object()));
  REQUIRE_THROWS_WITH(fresh("binarize", one)->load_state({{"scales", {{"a", 1.0}}}}),
                      ContainsSubstring("binarize learns nothing"));
  REQUIRE_THROWS_WITH(fresh("normalize", one)->load_state(nlohmann::json::array()),
                      ContainsSubstring("normalize learns nothing"));
}

TEST_CASE("the scalers round an integer past 2^53 rather than refusing it") {
  // numeric.hpp's four-argument for_each_numeric is the rounding policy, the
  // one sklearn's own float64 cast takes; expand.cpp passes its own `widen` and
  // refuses instead. Dropping the argument at an expand.cpp call site would
  // start rounding ids silently, so both policies are pinned on both paths:
  // the rounding here, the refusal in expand_test.cpp.
  constexpr int64_t kBig = (int64_t{1} << 53) + 1;
  const fft::Schema schema{{{"a", "i64"}}};
  auto kernel = fit_scaler("max_abs_scale", {{"columns", {"a"}}}, schema,
                           {make_batch({"a"}, {numeric_column<arrow::Int64Type>({kBig})})});
  REQUIRE(kernel->state().at("max_abs").at("a").get<double>() == 9007199254740992.0);
  fft::Batch out = kernel->transform(make_batch({"a"}, {numeric_column<arrow::Int64Type>({kBig})}));
  const auto& scaled = static_cast<const arrow::DoubleArray&>(*out.cols[0]);
  REQUIRE(scaled.Value(0) == 1.0);
  REQUIRE(std::get<double>(kernel->transform_row({{"a", kBig}})[0].second) == 1.0);
}

TEST_CASE("normalizer scratch and direct columns preserve bits across changing row norms") {
  for (const char* norm : {"l1", "l2", "max"}) {
    for (std::size_t width : {std::size_t{1}, std::size_t{4}, std::size_t{32}}) {
      CAPTURE(norm, width);
      fft::Batch input;
      input.rows = 129;
      fft::Schema schema;
      std::vector<Doubles> source(width);
      for (std::size_t c = 0; c < width; ++c) {
        const std::string name = "x" + std::to_string(c);
        input.names.push_back(name);
        schema.cols.emplace_back(name, "f64");
        for (int64_t row = 0; row < input.rows; ++row) {
          const int exponent = static_cast<int>((row + static_cast<int64_t>(c)) % 17) * 40 - 320;
          const double value =
              row % 7 == 0 ? (c % 2 == 0 ? -0.0 : 0.0) : std::ldexp(c % 2 == 0 ? 1.25 : -1.75, exponent);
          source[c].push_back(value);
        }
        input.cols.push_back(doubles(source[c]));
      }
      auto kernel = fft::make_kernel("normalize", {{"columns", input.names}, {"norm", norm}});
      kernel->stage = "scale";
      kernel->resolve(schema);
      const fft::Batch alias = input;
      const fft::Batch output = kernel->transform(input);
      for (int64_t row = 0; row < input.rows; ++row) {
        const auto slot = static_cast<std::size_t>(row);
        fft::Row record;
        for (std::size_t c = 0; c < width; ++c) record.emplace_back(input.names[c], *source[c][slot]);
        const fft::Row expected = kernel->transform_row(std::move(record));
        for (std::size_t c = 0; c < width; ++c) {
          const auto& actual = static_cast<const arrow::DoubleArray&>(*output.cols[c]);
          const auto& original = static_cast<const arrow::DoubleArray&>(*alias.cols[c]);
          REQUIRE(std::bit_cast<uint64_t>(actual.Value(row)) ==
                  std::bit_cast<uint64_t>(std::get<double>(expected[c].second)));
          REQUIRE(std::bit_cast<uint64_t>(original.Value(row)) == std::bit_cast<uint64_t>(*source[c][slot]));
        }
      }
    }
  }
  auto kernel = fft::make_kernel("normalize", {{"columns", {"a", "b"}}});
  kernel->stage = "scale";
  // The first column's late null wins over the second column's early NaN,
  // and all input validation precedes overflowing row-norm arithmetic.
  const double largest = std::numeric_limits<double>::max();
  REQUIRE_THROWS_WITH(
      kernel->transform(make_batch(
          {"a", "b"}, {doubles({largest, std::nullopt}), doubles({std::numeric_limits<double>::quiet_NaN(), 1})})),
      ContainsSubstring("column 'a' has a null"));
}

// Rows of one or two values skip the exact partials: IEEE addition already
// rounds a two-term sum correctly. Two more columns of zeros send the same row
// through the partials without changing its norm, so the two must agree bit
// for bit and refuse the same overflows.
TEST_CASE("normalize answers a one- or two-value row as the exact partials do") {
  const std::vector<double> specials{
      0.0,       -0.0, 5e-324, -1e-300, 1e-160,   1.5e-154, 0.1,
      1.0 / 3.0, -1.0, 3.0,    1e154,   -1.4e154, 1e300,    std::numeric_limits<double>::max()};
  std::vector<double> x;
  std::vector<double> y;
  for (double first : specials) {
    for (double second : specials) {
      x.push_back(first);
      y.push_back(second);
    }
  }
  for (int i = 0; i < 64; ++i) {
    x.push_back(std::ldexp(1.0 + i / 64.0, (i * 37) % 600 - 300));
    y.push_back(std::ldexp(-1.5 + i / 128.0, (i * 53) % 600 - 300));
  }
  auto normalizer = [](const std::vector<std::string>& columns, const char* norm) {
    auto kernel = fft::make_kernel("normalize", {{"columns", columns}, {"norm", norm}});
    kernel->stage = "scale";
    return kernel;
  };
  for (const char* norm : {"l1", "l2"}) {
    for (std::size_t width : {std::size_t{1}, std::size_t{2}}) {
      CAPTURE(norm, width);
      const std::vector<std::string> names =
          width == 1 ? std::vector<std::string>{"x"} : std::vector<std::string>{"x", "y"};
      std::vector<std::string> padded = names;
      padded.insert(padded.end(), {"zero", "nought"});
      auto direct = normalizer(names, norm);
      auto partials = normalizer(padded, norm);
      for (std::size_t row = 0; row < x.size(); ++row) {
        CAPTURE(x[row], y[row]);
        const fft::Batch pair = make_batch(
            names, width == 1 ? std::vector{doubles({x[row]})} : std::vector{doubles({x[row]}), doubles({y[row]})});
        fft::Batch padding = pair;
        padding.names.insert(padding.names.end(), {"zero", "nought"});
        padding.cols.insert(padding.cols.end(), {doubles({0.0}), doubles({0.0})});
        fft::Batch expected;
        try {
          expected = partials->transform(padding);
        } catch (const fft::Error&) {
          REQUIRE_THROWS_WITH(direct->transform(pair), ContainsSubstring("norm leaves the range of f64"));
          continue;
        }
        const fft::Batch actual = direct->transform(pair);
        for (const std::string& name : names) {
          const auto& got =
              static_cast<const arrow::DoubleArray&>(*actual.cols[static_cast<std::size_t>(actual.find(name))]);
          const auto& want =
              static_cast<const arrow::DoubleArray&>(*expected.cols[static_cast<std::size_t>(expected.find(name))]);
          REQUIRE(std::bit_cast<uint64_t>(got.Value(0)) == std::bit_cast<uint64_t>(want.Value(0)));
        }
      }
    }
  }
}

TEST_CASE("dense extrema retain canonical zero and partition-independent fitted state") {
  const fft::Schema schema{{{"a", "f64"}}};
  for (const char* id : {"min_max_scale", "max_abs_scale"}) {
    for (bool only_zero : {false, true}) {
      CAPTURE(id, only_zero);
      Doubles values;
      for (int row = 0; row < 259; ++row) {
        values.push_back(only_zero || row % 3 == 0 ? (row % 2 == 0 ? -0.0 : 0.0) : static_cast<double>(row % 17 - 8));
      }
      const auto source = doubles(values);
      const auto dense = fit_scaler(id, {{"columns", {"a"}}}, schema, {make_batch({"a"}, {source})});
      const auto split =
          fit_scaler(id, {{"columns", {"a"}}}, schema,
                     {make_batch({"a"}, {source->Slice(127)}), make_batch({"a"}, {source->Slice(0, 127)})});
      REQUIRE(dense->state().dump() == split->state().dump());
      const char* key = std::string_view(id) == "min_max_scale" ? "data_min" : "max_abs";
      if (only_zero) REQUIRE(std::bit_cast<uint64_t>(dense->state().at(key).at("a").get<double>()) == uint64_t{0});
    }
  }
}

// The whole message, on every path that reads a value: each scaler's batch
// transform over dense and nullable f64 and dense f32, the record path, and
// the fits. The bad value sits in normalize's second column.
TEST_CASE("every scaler's batch transform refuses a value that is not finite with the whole message") {
  for (const auto& [id, kernel] : every_scaler()) {
    for (const NonFinite& bad : non_finite_values()) {
      CAPTURE(id, bad.spelled);
      const auto narrow = static_cast<float>(bad.value);
      const std::vector<std::pair<std::string, fft::Batch>> inputs{
          {"dense f64", make_batch({"x", "y"}, {numeric_column<arrow::DoubleType>({1.0, 2.0, 3.0}),
                                                numeric_column<arrow::DoubleType>({0.5, bad.value, 0.25})})},
          // The null comes after the bad value, so binarize and normalize
          // reach the value before they reach the null.
          {"nullable f64", make_batch({"x", "y"}, {numeric_column<arrow::DoubleType>({1.0, 2.0, 3.0}),
                                                   numeric_column<arrow::DoubleType>({0.5, bad.value, {}})})},
          {"dense f32", make_batch({"x", "y"}, {numeric_column<arrow::FloatType>({1.0F, 2.0F, 3.0F}),
                                                numeric_column<arrow::FloatType>({0.5F, narrow, 0.25F})})}};
      for (const auto& [shape, batch] : inputs) {
        CAPTURE(shape);
        REQUIRE_THROWS_WITH(kernel->transform(batch), Equals(refusal(id, "y", bad.spelled)));
      }
    }
  }
}

TEST_CASE("every scaler's record path refuses a value that is not finite with the whole message") {
  for (const auto& [id, kernel] : every_scaler()) {
    for (const NonFinite& bad : non_finite_values()) {
      CAPTURE(id, bad.spelled);
      REQUIRE_THROWS_WITH(kernel->transform_row({{"x", 1.0}, {"y", bad.value}}), Equals(refusal(id, "y", bad.spelled)));
    }
  }
}

TEST_CASE("every fitted scaler refuses a value that is not finite with the whole message") {
  for (const char* id : {"standard_scale", "min_max_scale", "max_abs_scale"}) {
    for (const NonFinite& bad : non_finite_values()) {
      CAPTURE(id, bad.spelled);
      const auto narrow = static_cast<float>(bad.value);
      const std::vector<std::pair<std::string, std::shared_ptr<arrow::Array>>> columns{
          {"dense f64", numeric_column<arrow::DoubleType>({0.5, bad.value, 0.25})},
          {"nullable f64", numeric_column<arrow::DoubleType>({0.5, bad.value, {}})},
          {"dense f32", numeric_column<arrow::FloatType>({0.5F, narrow, 0.25F})}};
      for (const auto& [shape, column] : columns) {
        CAPTURE(shape);
        auto kernel = fft::make_kernel(id, {{"columns", {"y"}}});
        kernel->stage = "scale";
        fft::Schema schema{{{"y", column->type_id() == arrow::Type::FLOAT ? "f32" : "f64"}}};
        kernel->resolve(schema);
        auto state = kernel->make_state();
        REQUIRE_THROWS_WITH(kernel->consume(*state, make_batch({"y"}, {column})),
                            Equals(refusal(id, "y", bad.spelled)));
      }
    }
  }
}
