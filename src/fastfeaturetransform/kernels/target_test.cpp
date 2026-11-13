// Explicit C++ tests for the target-mean family: the params, the target
// checks and schema rewrite, the accumulator-merge (monoid) contract, the
// three formulas, and load_state.

#include <arrow/array.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <cmath>
#include <limits>
#include <utility>

#include "fastfeaturetransform/kernel.hpp"
#include "fastfeaturetransform/kernels/dictionary_test_support.hpp"
#include "fastfeaturetransform/test_support.hpp"

namespace fft = fastfeaturetransform;
using Catch::Approx;
using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::Equals;
using fft::testing::make_batch;
using fft::testing::numeric_column;
using fft::testing::utf8_column;

namespace {

// Fit one of the target-mean kernels with one worker state per batch.
std::unique_ptr<fft::Kernel> fit_blend(const std::string& id, const nlohmann::json& params, fft::Schema schema,
                                       const std::vector<fft::Batch>& batches) {
  return fft::testing::fit_kernel_per_batch(id, params, std::move(schema), batches, "blend");
}

// c: a a b b b c, with a 0/1 target — the shared fixture of these tests.
fft::Batch head() {
  return make_batch({"c", "y"}, {utf8_column({"a", "a", "b"}), numeric_column<arrow::Int64Type>({1, 0, 1})});
}
fft::Batch tail() {
  return make_batch({"c", "y"}, {utf8_column({"b", "b", "c"}), numeric_column<arrow::Int64Type>({1, 0, 0})});
}

const fft::Schema kSchema{{{"c", "utf8"}, {"y", "i64"}}};

}  // namespace

TEST_CASE("the target-mean kernels validate their params") {
  const nlohmann::json target{{"target", "y"}};
  REQUIRE_THROWS_WITH((fft::make_kernel("target_encode", nlohmann::json::object())),
                      ContainsSubstring("'target' must be a column name"));
  REQUIRE_THROWS_WITH((fft::make_kernel("target_encode", {{"target", "y"}, {"smoothing", 0}})),
                      ContainsSubstring("'smoothing' must be a positive number"));
  REQUIRE_THROWS_WITH((fft::make_kernel("target_encode", {{"target", "y"}, {"min_samples_leaf", 1.5}})),
                      ContainsSubstring("'min_samples_leaf' must be an integer"));
  // REGRESSION: an integer past int64 is refused. 2^63 used to be read
  // wrapped to -2^63, which turned the blend weight from 0 to 1.
  REQUIRE_THROWS_WITH((fft::make_kernel("target_encode", {{"target", "y"}, {"min_samples_leaf", uint64_t{1} << 63}})),
                      ContainsSubstring("'min_samples_leaf' must be an integer"));
  REQUIRE_THROWS_WITH((fft::make_kernel("m_estimate_encode", {{"target", "y"}, {"m", -1}})),
                      ContainsSubstring("'m' must be a finite number >= 0"));
  REQUIRE_THROWS_WITH((fft::make_kernel("woe_encode", {{"target", "y"}, {"regularization", 0}})),
                      ContainsSubstring("'regularization' must be a positive number"));
  REQUIRE_THROWS_WITH((fft::make_kernel("woe_encode", {{"target", "y"}, {"handle_missing", "sometimes"}})),
                      ContainsSubstring("'handle_missing' must be"));
  REQUIRE_THROWS_WITH((fft::make_kernel("woe_encode", {{"target", "y"}, {"handle_unknown", "sometimes"}})),
                      ContainsSubstring("'handle_unknown' must be"));
  for (const std::string& id : {"target_encode", "m_estimate_encode", "woe_encode"}) {
    REQUIRE(fft::make_kernel(id, target)->needs_fit());
    REQUIRE(fft::make_kernel(id, target)->params().at("handle_unknown") == "value");
  }
}

TEST_CASE("target_encode checks its target and rewrites the column type") {
  auto kernel = fft::make_kernel("target_encode", {{"columns", {"c"}}, {"target", "y"}});
  kernel->stage = "blend";
  fft::Schema schema{{{"c", "utf8"}, {"y", "i64"}}};
  kernel->resolve(schema);
  REQUIRE(schema.cols == fft::Schema{{{"c", "f64"}, {"y", "i64"}}}.cols);
  // Unfitted, the target is read too; a fitted kernel never reads it again.
  REQUIRE(kernel->inputs == std::vector<std::string>{"c", "y"});

  auto missing = fft::make_kernel("target_encode", {{"columns", {"c"}}, {"target", "y"}});
  missing->stage = "blend";
  fft::Schema without{{{"c", "utf8"}}};
  REQUIRE_THROWS_WITH(missing->resolve(without), ContainsSubstring("target column 'y' not found in input"));

  auto text = fft::make_kernel("target_encode", {{"columns", {"c"}}, {"target", "y"}});
  text->stage = "blend";
  fft::Schema strings{{{"c", "utf8"}, {"y", "utf8"}}};
  REQUIRE_THROWS_WITH(text->resolve(strings),
                      ContainsSubstring("target column 'y' is utf8, expected an integer or float column"));

  auto itself = fft::make_kernel("target_encode", {{"columns", {"y"}}, {"target", "y"}});
  itself->stage = "blend";
  fft::Schema numbers{{{"y", "i64"}}};
  REQUIRE_THROWS_WITH(itself->resolve(numbers), ContainsSubstring("is also the target"));

  auto numeric = fft::make_kernel("target_encode", {{"columns", {"n"}}, {"target", "y"}});
  numeric->stage = "blend";
  fft::Schema mixed{{{"n", "i64"}, {"y", "i64"}}};
  REQUIRE_THROWS_WITH(numeric->resolve(mixed), ContainsSubstring("is i64, expected utf8"));
}

TEST_CASE("target_encode merges worker accumulators and both paths agree") {
  const nlohmann::json params{{"columns", {"c"}}, {"target", "y"}, {"min_samples_leaf", 2}, {"smoothing", 1.0}};
  auto kernel = fit_blend("target_encode", params, kSchema, {head(), tail()});
  const nlohmann::json state = kernel->state();
  // prior 3/6; a: n=2 s=1, b: n=3 s=2, c: n=1 s=0.
  const double prior = 0.5;
  auto blend = [&](double n, double s) {
    const double weight = 1.0 / (1.0 + std::exp(-(n - 2.0) / 1.0));
    return prior * (1 - weight) + (s / n) * weight;
  };
  REQUIRE(state.at("prior").get<double>() == Approx(prior));
  REQUIRE(state.at("values").at("c").at("a").get<double>() == Approx(blend(2, 1)));
  REQUIRE(state.at("values").at("c").at("b").get<double>() == Approx(blend(3, 2)));
  REQUIRE(state.at("values").at("c").at("c").get<double>() == Approx(blend(1, 0)));
  REQUIRE(state.at("null_values").at("c").is_null());
  // Only the merged accumulators may decide a value: this is the monoid contract.
  REQUIRE(fit_blend("target_encode", params, kSchema, {tail(), head()})->state() == state);
  REQUIRE(kernel->inputs == std::vector<std::string>{"c"});

  const std::vector<std::optional<std::string>> values{"a", "b", "zzz", std::nullopt};
  fft::Batch out = kernel->transform(make_batch({"c"}, {utf8_column(values)}));
  const auto& encoded = static_cast<const arrow::DoubleArray&>(*out.cols[0]);
  REQUIRE(encoded.Value(0) == Approx(blend(2, 1)));
  REQUIRE(encoded.Value(2) == Approx(prior));  // unseen
  REQUIRE(encoded.Value(3) == Approx(prior));  // no null was seen at fit
  for (std::size_t i = 0; i < values.size(); ++i) {
    fft::RowValue value = values[i].has_value() ? fft::RowValue(*values[i]) : fft::RowValue(std::monostate{});
    fft::Row row = kernel->transform_row({{"c", value}});
    REQUIRE(std::get<double>(row[0].second) == encoded.Value(static_cast<int64_t>(i)));
  }
  REQUIRE_THROWS_WITH(kernel->transform_row({{"c", int64_t{1}}}), ContainsSubstring("must be a string or null"));
  REQUIRE_THROWS_WITH(kernel->transform_row({{"n", int64_t{1}}}), ContainsSubstring("missing from record"));
}

TEST_CASE("m_estimate_encode blends toward the prior and gives up on an id-like column") {
  auto kernel =
      fit_blend("m_estimate_encode", {{"columns", {"c"}}, {"target", "y"}, {"m", 2.0}}, kSchema, {head(), tail()});
  const nlohmann::json values = kernel->state().at("values").at("c");
  REQUIRE(values.at("a").get<double>() == Approx((1 + 0.5 * 2) / (2 + 2)));
  REQUIRE(values.at("b").get<double>() == Approx((2 + 0.5 * 2) / (3 + 2)));
  REQUIRE(values.at("c").get<double>() == Approx((0 + 0.5 * 2) / (1 + 2)));

  const fft::Batch ids =
      make_batch({"c", "y"}, {utf8_column({"a", "b", "c", "d"}), numeric_column<arrow::Int64Type>({1, 0, 1, 1})});
  auto id_like = fit_blend("m_estimate_encode", {{"columns", {"c"}}, {"target", "y"}}, kSchema, {ids});
  const nlohmann::json flat = id_like->state().at("values").at("c");
  for (const std::string& key : {"a", "b", "c", "d"}) REQUIRE(flat.at(key).get<double>() == Approx(0.75));
}

TEST_CASE("woe_encode needs a binary target and zeroes a category seen once") {
  auto kernel = fit_blend("woe_encode", {{"columns", {"c"}}, {"target", "y"}}, kSchema, {head(), tail()});
  const nlohmann::json values = kernel->state().at("values").at("c");
  // S = 3, N = 6: a is n=2 s=1, b is n=3 s=2, c is n=1 (the singleton guard).
  REQUIRE(values.at("a").get<double>() == Approx(std::log(((1.0 + 1) / (3 + 2)) / (((2 - 1.0) + 1) / ((6 - 3) + 2)))));
  REQUIRE(values.at("b").get<double>() == Approx(std::log(((2.0 + 1) / (3 + 2)) / (((3 - 2.0) + 1) / ((6 - 3) + 2)))));
  REQUIRE(values.at("c").get<double>() == 0.0);
  // Unknown is 0 for woe_encode, not the prior.
  REQUIRE(std::get<double>(kernel->transform_row({{"c", std::string("zzz")}})[0].second) == 0.0);

  const fft::Batch flat =
      make_batch({"c", "y"}, {utf8_column({"a", "b", "c"}), numeric_column<arrow::Int64Type>({1, 1, 1})});
  REQUIRE_THROWS_WITH(fit_blend("woe_encode", {{"columns", {"c"}}, {"target", "y"}}, kSchema, {flat}),
                      ContainsSubstring("must be binary, but it holds only one distinct value"));
  const fft::Batch three =
      make_batch({"c", "y"}, {utf8_column({"a", "b", "c"}), numeric_column<arrow::Int64Type>({0, 1, 2})});
  REQUIRE_THROWS_WITH(fit_blend("woe_encode", {{"columns", {"c"}}, {"target", "y"}}, kSchema, {three}),
                      ContainsSubstring("more than two distinct values"));
  const fft::Batch twos =
      make_batch({"c", "y"}, {utf8_column({"a", "b", "c"}), numeric_column<arrow::Int64Type>({1, 2, 2})});
  REQUIRE_THROWS_WITH(fit_blend("woe_encode", {{"columns", {"c"}}, {"target", "y"}}, kSchema, {twos}),
                      ContainsSubstring("no value 0 was found"));

  // With a 0/1 target only the regularization can make a weight of evidence
  // infinite: 5e-324 / 2 rounds to 0, so 'a', which holds no 1s, gets log 0.
  const fft::Batch split =
      make_batch({"c", "y"}, {utf8_column({"a", "a", "b", "b"}), numeric_column<arrow::Int64Type>({0, 0, 1, 1})});
  REQUIRE_THROWS_WITH(
      fit_blend("woe_encode", {{"columns", {"c"}}, {"target", "y"}, {"regularization", 5e-324}}, kSchema, {split}),
      ContainsSubstring("column 'c' has a category whose weight of evidence is not finite; check 'regularization'"));
  // upstream's binary check also takes -1 and 2, which can make a share negative.
  const fft::Batch wide =
      make_batch({"c", "y"}, {utf8_column({"a", "a", "b", "b"}), numeric_column<arrow::Int64Type>({-1, -1, 2, 2})});
  REQUIRE_THROWS_WITH(fit_blend("woe_encode", {{"columns", {"c"}}, {"target", "y"}}, kSchema, {wide}),
                      ContainsSubstring("weight of evidence is not finite; woe_encode needs a target of 0s and 1s"));
}

TEST_CASE("the target-mean kernels keep a null category and skip a null target") {
  const fft::Batch batch = make_batch({"c", "y"}, {utf8_column({"a", "a", std::nullopt, std::nullopt, "b"}),
                                                   numeric_column<arrow::Int64Type>({1, 0, 1, 1, std::nullopt})});
  const nlohmann::json params{{"columns", {"c"}}, {"target", "y"}, {"m", 0.0}};
  auto kernel = fit_blend("m_estimate_encode", params, kSchema, {batch});
  // The b row taught nothing: prior is 3/4 and b never reached the table.
  REQUIRE(kernel->state().at("prior").get<double>() == Approx(0.75));
  REQUIRE(kernel->state().at("values").at("c") == nlohmann::json({{"a", 0.5}}));
  REQUIRE(kernel->state().at("null_values").at("c").get<double>() == Approx(1.0));
  REQUIRE(std::get<double>(kernel->transform_row({{"c", fft::RowValue(std::monostate{})}})[0].second) == 1.0);

  auto as_null = fit_blend("m_estimate_encode", {{"columns", {"c"}}, {"target", "y"}, {"handle_missing", "return_nan"}},
                           kSchema, {batch});
  REQUIRE(as_null->state().at("null_values").at("c").is_null());
  REQUIRE(std::holds_alternative<std::monostate>(
      as_null->transform_row({{"c", fft::RowValue(std::monostate{})}})[0].second));

  REQUIRE_THROWS_WITH(fit_blend("m_estimate_encode", {{"columns", {"c"}}, {"target", "y"}, {"handle_missing", "error"}},
                                kSchema, {batch}),
                      ContainsSubstring("has nulls and handle_missing is 'error'"));
  const fft::Batch empty =
      make_batch({"c", "y"}, {utf8_column({"a"}), numeric_column<arrow::Int64Type>({std::nullopt})});
  REQUIRE_THROWS_WITH(fit_blend("target_encode", {{"columns", {"c"}}, {"target", "y"}}, kSchema, {empty}),
                      ContainsSubstring("has no non-null values to learn from"));
}

TEST_CASE("a float target sums exactly however the batches were split") {
  const fft::Schema schema{{{"c", "utf8"}, {"y", "f64"}}};
  const fft::Batch a =
      make_batch({"c", "y"}, {utf8_column({"a", "a"}), numeric_column<arrow::DoubleType>({1e100, 1.0})});
  const fft::Batch b =
      make_batch({"c", "y"}, {utf8_column({"a", "b"}), numeric_column<arrow::DoubleType>({-1e100, 2.0})});
  const nlohmann::json params{{"columns", {"c"}}, {"target", "y"}, {"m", 0.0}};
  auto kernel = fit_blend("m_estimate_encode", params, schema, {a, b});
  REQUIRE(kernel->state().at("values").at("c").at("a").get<double>() == Approx(1.0 / 3.0));
  REQUIRE(fit_blend("m_estimate_encode", params, schema, {b, a})->state() == kernel->state());

  const fft::Batch huge =
      make_batch({"c", "y"}, {utf8_column({"a"}), numeric_column<arrow::DoubleType>({1e300 * 1e300})});
  REQUIRE_THROWS_WITH(fit_blend("target_encode", {{"columns", {"c"}}, {"target", "y"}}, schema, {huge}),
                      ContainsSubstring("is not finite"));
}

// REGRESSION: a large enough smoothing parameter sends the encoded value to
// infinity. finalize used to accept it, though load_state called it invalid
// and state() wrote it as JSON null, so an hours-long fit produced an
// artifact that only proved unloadable afterwards, and transform handed a
// serving caller an inf in a column the schema calls f64.
TEST_CASE("the target-mean family refuses an encoding that is not finite") {
  const fft::Schema schema{{{"c", "utf8"}, {"y", "f64"}}};
  const fft::Batch batch =
      make_batch({"c", "y"}, {utf8_column({"a", "a", "b"}), numeric_column<arrow::DoubleType>({10.0, 20.0, 30.0})});
  // The target is ordinary, but prior * m overflows f64 on its own, and so
  // then does (s + prior * m) / (n + m).
  REQUIRE_THROWS_WITH(
      fit_blend("m_estimate_encode", {{"columns", {"c"}}, {"target", "y"}, {"m", 1e308}}, schema, {batch}),
      ContainsSubstring("column 'c' encodes to a value that is not finite; check 'm'"));
  // A table holding such a value is refused at load, even one written by hand.
  auto loading = fft::make_kernel("target_encode", {{"columns", {"c"}}, {"target", "y"}});
  loading->stage = "blend";
  REQUIRE_THROWS_WITH(loading->load_state({{"columns", {"c"}},
                                           {"prior", 0.5},
                                           {"values", {{"c", {{"x", std::numeric_limits<double>::infinity()}}}}},
                                           {"null_values", {{"c", nullptr}}}}),
                      ContainsSubstring("values['c']['x'] must be a finite number"));
}

// A row whose target is null teaches nothing, even when its category is the
// null one: the fit is the same without it.
TEST_CASE("every target encoder skips a null whose target is null") {
  const auto batch = [](bool targetless) {
    std::vector<std::optional<std::string>> keys{"a", "a", "b", "b", std::nullopt, std::nullopt};
    std::vector<std::optional<int64_t>> targets{1, 0, 1, 0, 1, 1};
    if (targetless) {
      keys.emplace_back(std::nullopt);
      targets.emplace_back(std::nullopt);
    }
    return make_batch({"c", "y"}, {utf8_column(keys), numeric_column<arrow::Int64Type>(targets)});
  };
  const nlohmann::json params{{"columns", {"c"}}, {"target", "y"}};
  for (const char* id : {"target_encode", "m_estimate_encode", "woe_encode", "james_stein_encode",
                         "leave_one_out_encode", "cat_boost_encode"}) {
    CAPTURE(id);
    REQUIRE(fit_blend(id, params, kSchema, {batch(true)})->state() ==
            fit_blend(id, params, kSchema, {batch(false)})->state());
  }
}

// The fit computes the null category's value only when it keeps it, under
// handle_missing='value', so a value that would not be finite fails the fit
// under that option alone.
TEST_CASE("a target encoder encodes the null category only where it keeps it") {
  // cat_boost_encode: a and b are singletons, which take the prior, 25, and
  // the null category's (70 + 25 * 1e308) / (2 + 1e308) is not finite.
  const fft::Batch boost = make_batch({"c", "y"}, {utf8_column({"a", "b", std::nullopt, std::nullopt}),
                                                   numeric_column<arrow::Int64Type>({10, 20, 30, 40})});
  // woe_encode: the null category holds no 1s, and 5e-324 / 2 rounds to 0.
  const fft::Batch woe = make_batch({"c", "y"}, {utf8_column({"a", "b", "c", "c", std::nullopt, std::nullopt}),
                                                 numeric_column<arrow::Int64Type>({1, 0, 1, 0, 0, 0})});
  struct Case {
    const char* id;
    nlohmann::json param;
    fft::Batch batch;
    const char* refused;
  };
  const Case cases[] = {
      {"cat_boost_encode", {{"a", 1e308}}, boost, "encodes to a value that is not finite; check 'a'"},
      {"woe_encode", {{"regularization", 5e-324}}, woe, "weight of evidence is not finite; check 'regularization'"}};
  for (const Case& entry : cases) {
    CAPTURE(entry.id);
    nlohmann::json params{{"columns", {"c"}}, {"target", "y"}};
    params.update(entry.param);
    REQUIRE_THROWS_WITH(fit_blend(entry.id, params, kSchema, {entry.batch}), ContainsSubstring(entry.refused));
    params["handle_missing"] = "return_nan";
    auto kernel = fit_blend(entry.id, params, kSchema, {entry.batch});
    REQUIRE(kernel->state().at("null_values").at("c").is_null());
    REQUIRE(std::holds_alternative<std::monostate>(
        kernel->transform_row({{"c", fft::RowValue(std::monostate{})}})[0].second));
  }
}

// REGRESSION: 'error' is the strict option, so it refuses a null at transform
// as well as at fit.
TEST_CASE("the target-mean family refuses a null at transform under handle_missing error") {
  const fft::Schema schema{{{"c", "utf8"}, {"y", "f64"}}};
  const fft::Batch batch =
      make_batch({"c", "y"}, {utf8_column({"a", "b"}), numeric_column<arrow::DoubleType>({1.0, 0.0})});
  for (const char* id : {"target_encode", "m_estimate_encode", "woe_encode"}) {
    auto kernel = fit_blend(id, {{"columns", {"c"}}, {"target", "y"}, {"handle_missing", "error"}}, schema, {batch});
    REQUIRE_THROWS_WITH(kernel->transform(make_batch({"c"}, {utf8_column({"a", std::nullopt})})),
                        ContainsSubstring("column 'c' has a null and handle_missing is 'error'"));
    REQUIRE_THROWS_WITH(kernel->transform_row({{"c", fft::RowValue(std::monostate{})}}),
                        ContainsSubstring("column 'c' has a null and handle_missing is 'error'"));
  }
}

// REGRESSION: a null counts toward handle_missing='error' even on a row whose
// target is null. Such a row used to be skipped before its null was counted,
// so the fit accepted the null and the fitted encoder then refused its own
// training data at transform.
TEST_CASE("the target-mean family under handle_missing error refuses a null whose target is null") {
  const fft::Schema schema{{{"c", "utf8"}, {"y", "f64"}}};
  // The null arrives in the second batch, so the merged count is the one
  // that decides.
  const std::vector<fft::Batch> batches{
      make_batch({"c", "y"}, {utf8_column({"a", "b"}), numeric_column<arrow::DoubleType>({1.0, 0.0})}),
      make_batch({"c", "y"}, {utf8_column({std::nullopt}), numeric_column<arrow::DoubleType>({std::nullopt})})};
  for (const char* id : {"target_encode", "m_estimate_encode", "woe_encode"}) {
    REQUIRE_THROWS_WITH(
        fit_blend(id, {{"columns", {"c"}}, {"target", "y"}, {"handle_missing", "error"}}, schema, batches),
        ContainsSubstring("column 'c' has nulls and handle_missing is 'error'"));
    REQUIRE_NOTHROW(fit_blend(id, {{"columns", {"c"}}, {"target", "y"}}, schema, batches));
  }
}

TEST_CASE("target_encode load_state rejects a malformed table") {
  auto fresh = [] {
    auto kernel = fft::make_kernel("target_encode", {{"columns", {"c"}}, {"target", "y"}});
    kernel->stage = "blend";
    return kernel;
  };
  const nlohmann::json good{
      {"columns", {"c"}}, {"prior", 0.5}, {"values", {{"c", {{"x", 0.25}}}}}, {"null_values", {{"c", nullptr}}}};
  REQUIRE_NOTHROW(fresh()->load_state(good));
  auto broken = [&](auto mutate, const std::string& message) {
    nlohmann::json state = good;
    mutate(state);
    REQUIRE_THROWS_WITH(fresh()->load_state(state), ContainsSubstring(message));
  };
  broken([](nlohmann::json& state) { state.erase("prior"); }, "must hold a 'values' object");
  broken([](nlohmann::json& state) { state["prior"] = "half"; }, "must hold a 'values' object");
  broken([](nlohmann::json& state) { state["values"]["c"] = 5; }, "must be an object");
  broken([](nlohmann::json& state) { state["values"]["c"]["x"] = "quarter"; }, "must be a finite number");
  broken([](nlohmann::json& state) { state["values"] = {{"d", {{"x", 0.25}}}}; }, "must be an object");
  broken([](nlohmann::json& state) { state["values"].erase("c"); },
         "state 'columns' does not name the same columns as the fitted table");
  broken([](nlohmann::json& state) { state.erase("columns"); }, "state must hold a 'columns' list");
  broken([](nlohmann::json& state) { state["null_values"]["c"] = "half"; }, "must be a finite number or null");
  // REGRESSION: the null category is the entry a hand-edited manifest is
  // likeliest to mangle, and a wrong shape used to read as a legitimate
  // 'no null category' rather than as an error, so a serving path silently
  // started mapping missing values onto the unknown value.
  broken([](nlohmann::json& state) { state["null_values"] = 0.5; }, "'null_values' must be an object");
  broken([](nlohmann::json& state) { state["null_values"].erase("c"); }, "null_values has no entry for column 'c'");
  // REGRESSION: null_values is required, as every saved pipeline holds it. A
  // state without it, with the key misspelled or with null in its place used
  // to load with no null category, so a null encoded to the prior.
  broken([](nlohmann::json& state) { state.erase("null_values"); }, "blend: state must hold a 'null_values' object");
  broken(
      [](nlohmann::json& state) {
        state["null_value"] = state.at("null_values");
        state.erase("null_values");
      },
      "blend: state must hold a 'null_values' object");
  broken([](nlohmann::json& state) { state["null_values"] = nullptr; }, "blend: 'null_values' must be an object");

  // A loaded encoder reads its column alone, so a serving file may omit the target.
  auto loaded = fresh();
  loaded->load_state(good);
  REQUIRE(loaded->inputs == std::vector<std::string>{"c"});
  fft::Schema schema{{{"c", "utf8"}}};
  REQUIRE_NOTHROW(loaded->resolve(schema));
  REQUIRE(*schema.find("c") == "f64");
}

TEST_CASE("target-mean dictionary fits exclude null-target rows and preserve exact artifacts") {
  const std::vector<std::optional<int64_t>> indices{0, 1, 4, 3, std::nullopt, 2, 0, 1, 2, 4, 0, 1, 3, 2, 0, 1};
  const auto first = fft::testing::dictionary_fixture<arrow::Int16Type>(
      {"a", "b", "a", std::nullopt, "targetless", "unused"}, indices, true, 3, 2);
  const auto second = fft::testing::dictionary_fixture<arrow::Int64Type>(
      {"c", "a", "c", std::nullopt, "targetless", "unused-new"}, indices);
  for (const std::string& id : {"target_encode", "m_estimate_encode", "woe_encode"}) {
    for (bool integer : {false, true}) {
      CAPTURE(id, integer);
      const auto target =
          integer
              ? numeric_column<arrow::Int64Type>({0, 1, std::nullopt, 1, 0, 1, 0, 1, 1, std::nullopt, 0, 0, 1, 0, 1, 0})
              : numeric_column<arrow::DoubleType>({0.0, 1.0, std::nullopt, 1.0, -0.0, 1.0, 0.0, 1.0, 1.0, std::nullopt,
                                                   0.0, 0.0, 1.0, 0.0, 1.0, 0.0});
      const nlohmann::json params{{"columns", {"c"}}, {"target", "y"}, {"handle_unknown", "error"}};
      const fft::Schema schema{{{"c", "utf8"}, {"y", integer ? "i64" : "f64"}}};
      auto dictionary = fit_blend(
          id, params, schema,
          {make_batch({"c", "y"}, {first.encoded, target}), make_batch({"c", "y"}, {second.encoded, target})});
      auto plain =
          fit_blend(id, params, schema,
                    {make_batch({"c", "y"}, {first.plain, target}), make_batch({"c", "y"}, {second.plain, target})});
      REQUIRE(dictionary->state().dump() == plain->state().dump());
      REQUIRE_FALSE(dictionary->state().at("values").at("c").contains("targetless"));
      const auto serving =
          fft::testing::dictionary_fixture({"a", "b", "c", "unseen", std::nullopt}, {0, 1, 2, 4, 0, 1, 2, 4});
      auto loaded = fft::make_kernel(id, params);
      loaded->load_state(dictionary->state());
      REQUIRE(loaded->state().dump() == dictionary->state().dump());
      REQUIRE(loaded->transform(make_batch({"c"}, {serving.encoded}))
                  .cols[0]
                  ->Equals(plain->transform(make_batch({"c"}, {serving.plain})).cols[0]));
      const auto bad = fft::testing::dictionary_fixture({"a", "unseen", std::nullopt}, {2, 1, 0, 2, 1});
      REQUIRE_THROWS_WITH(loaded->transform(make_batch({"c"}, {bad.encoded})),
                          ContainsSubstring("was not seen at fit"));
    }
  }
}

// An f64 holds every second integer above 2^53 and every 1,024th above 2^62,
// so these targets sum to small numbers only while every value is whole.
TEST_CASE("every target encoder sums an integer target above 2^53 whole") {
  const int64_t big = int64_t{1} << 62;
  // a: 2 over four rows, or 0 once big + 1 is rounded to big. b: 4 over
  // three rows, or 3. The fit: 6 over seven rows, or 3.
  const fft::Batch first = make_batch(
      {"c", "y"},
      {utf8_column({"a", "a", "b", "a"}), numeric_column<arrow::Int64Type>({big + 1, -big, big + 1, big + 1})});
  const fft::Batch second =
      make_batch({"c", "y"}, {utf8_column({"a", "b", "b"}), numeric_column<arrow::Int64Type>({-big, -big, 3})});
  // Under these params each kernel's value is the category's plain mean:
  // target_encode's weight rounds to 1 once (n - min_samples_leaf) / smoothing
  // passes 53 ln 2, about 36.7, and james_stein_encode keeps the mean of each
  // of two categories.
  const std::pair<const char*, nlohmann::json> cases[] = {
      {"target_encode", {{"min_samples_leaf", -100}, {"smoothing", 1.0}}},
      {"m_estimate_encode", {{"m", 0.0}}},
      {"james_stein_encode", nlohmann::json::object()},
      {"leave_one_out_encode", nlohmann::json::object()},
      {"cat_boost_encode", {{"a", 0.0}}}};
  for (const auto& [id, extra] : cases) {
    CAPTURE(id);
    nlohmann::json params{{"columns", {"c"}}, {"target", "y"}};
    params.update(extra);
    auto kernel = fit_blend(id, params, kSchema, {first, second});
    const nlohmann::json state = kernel->state();
    REQUIRE(state.at("prior").get<double>() == 6.0 / 7.0);
    REQUIRE(state.at("values").at("c").at("a").get<double>() == 0.5);
    REQUIRE(state.at("values").at("c").at("b").get<double>() == 4.0 / 3.0);
    REQUIRE(fit_blend(id, params, kSchema, {second, first})->state() == state);
  }

  // woe_encode needs a target of two values, one at most 0 and one at least
  // 1. A category of as many rows of each sums to its count of the larger
  // one: its odds are then the fit's, and its weight of evidence is 0.
  const fft::Batch binary =
      make_batch({"c", "y"}, {utf8_column({"a", "a", "b", "a", "b", "a"}),
                              numeric_column<arrow::Int64Type>({big + 1, -big, big + 1, big + 1, -big, -big})});
  auto woe = fit_blend("woe_encode", {{"columns", {"c"}}, {"target", "y"}}, kSchema, {binary});
  REQUIRE(woe->state().at("prior").get<double>() == 0.5);
  REQUIRE(woe->state().at("values").at("c") == nlohmann::json({{"a", 0.0}, {"b", 0.0}}));

  // Three rows of the largest i64 sum past what 64 bits hold.
  const int64_t most = std::numeric_limits<int64_t>::max();
  const fft::Batch wide =
      make_batch({"c", "y"}, {utf8_column({"a", "a", "a"}), numeric_column<arrow::Int64Type>({most, most, most})});
  for (const char* id :
       {"target_encode", "m_estimate_encode", "james_stein_encode", "leave_one_out_encode", "cat_boost_encode"}) {
    CAPTURE(id);
    auto kernel = fit_blend(id, {{"columns", {"c"}}, {"target", "y"}}, kSchema, {wide});
    REQUIRE(kernel->state().at("prior").get<double>() == 0x1p63);
  }
}

TEST_CASE("a float target fits below its kernel's magnitude bound and is refused from it") {
  struct Bound {
    const char* id;
    double bound;
    const char* refused;  // the bound as the message prints it, and the rest of the message
  };
  const char* sum =
      "8.452712498170644e+270 is 2^900 or more in magnitude, more than a sum can hold (scale the target first)";
  const char* squares =
      "2.9073548971824276e+135 is 2^450 or more in magnitude, more than the squared sums can hold (scale the target "
      "first)";
  // james_stein_encode keeps the sum of the target's squares, so its bound is
  // the square root of the others'.
  const Bound bounds[] = {
      {"target_encode", 0x1p900, sum},        {"m_estimate_encode", 0x1p900, sum},
      {"woe_encode", 0x1p900, sum},           {"james_stein_encode", 0x1p450, squares},
      {"leave_one_out_encode", 0x1p900, sum}, {"cat_boost_encode", 0x1p900, sum},
  };
  const fft::Schema schema{{{"c", "utf8"}, {"y", "f64"}}};
  const nlohmann::json params{{"columns", {"c"}}, {"target", "y"}};
  // Two values, 0 and one that is at least 1 or refused, so that woe_encode
  // reads the same batches as the others.
  auto batch = [](double value) {
    return make_batch({"c", "y"},
                      {utf8_column({"a", "a", "b", "b"}), numeric_column<arrow::DoubleType>({value, 0.0, value, 0.0})});
  };
  for (const Bound& entry : bounds) {
    CAPTURE(entry.id);
    const double inside = std::nextafter(entry.bound, 0.0);
    REQUIRE(fit_blend(entry.id, params, schema, {batch(inside)})->state().at("prior").get<double>() == inside / 2);
    REQUIRE_THROWS_WITH(fit_blend(entry.id, params, schema, {batch(entry.bound)}),
                        Equals(std::string("blend: target column 'y': value ") + entry.refused));
    REQUIRE_THROWS_WITH(fit_blend(entry.id, params, schema, {batch(-entry.bound)}),
                        Equals(std::string("blend: target column 'y': value -") + entry.refused));
    // Nothing is refused between 2^450, where james_stein_encode stops, and
    // 2^900.
    if (entry.bound == 0x1p900) REQUIRE_NOTHROW(fit_blend(entry.id, params, schema, {batch(0x1p450)}));
  }
}
