// Explicit C++ tests for the shrinkage encoders: the params, the target
// binding and schema rewrite, the accumulator-merge (monoid) contract, the
// per-model values, and load_state.

#include <arrow/array.h>

#include <algorithm>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "fastfeaturetransform/kernel.hpp"
#include "fastfeaturetransform/kernels/dictionary_test_support.hpp"
#include "fastfeaturetransform/test_support.hpp"

namespace fft = fastfeaturetransform;
using Catch::Approx;
using Catch::Matchers::ContainsSubstring;
using fft::testing::f64_column;
using fft::testing::make_batch;
using fft::testing::utf8_column;

namespace {

// Fit a shrinkage kernel with one worker state per batch.
std::unique_ptr<fft::Kernel> fit_kernel(const std::string& id, const nlohmann::json& params, fft::Schema schema,
                                        const std::vector<fft::Batch>& batches) {
  return fft::testing::fit_kernel_per_batch(id, params, std::move(schema), batches, "shrink");
}

const fft::Schema kSchema{{{"c", "utf8"}, {"y", "f64"}}};

// c = a a b b b, y = 1 3 2 8 4; split into two batches.
std::vector<fft::Batch> sample_batches() {
  return {make_batch({"c", "y"}, {utf8_column({"a", "a"}), f64_column({1.0, 3.0})}),
          make_batch({"c", "y"}, {utf8_column({"b", "b", "b"}), f64_column({2.0, 8.0, 4.0})})};
}

double value_of(const std::unique_ptr<fft::Kernel>& kernel, const std::string& category) {
  return kernel->state().at("values").at("c").at(category).get<double>();
}

}  // namespace

TEST_CASE("the shrinkage encoders validate their params") {
  const nlohmann::json target{{"target", "y"}};
  REQUIRE_THROWS_WITH(fft::make_kernel("james_stein_encode", nlohmann::json::object()),
                      ContainsSubstring("'target' must be a column name"));
  REQUIRE_THROWS_WITH(fft::make_kernel("leave_one_out_encode", {{"target", ""}}),
                      ContainsSubstring("'target' must be a column name"));
  REQUIRE_THROWS_WITH(fft::make_kernel("cat_boost_encode", {{"target", "y"}, {"handle_unknown", "sometimes"}}),
                      ContainsSubstring("'handle_unknown' must be"));
  REQUIRE_THROWS_WITH(fft::make_kernel("cat_boost_encode", {{"target", "y"}, {"handle_missing", "sometimes"}}),
                      ContainsSubstring("'handle_missing' must be"));
  REQUIRE_THROWS_WITH(fft::make_kernel("cat_boost_encode", {{"target", "y"}, {"a", -1}}),
                      ContainsSubstring("'a' must be a finite number >= 0"));
  REQUIRE_THROWS_WITH(fft::make_kernel("james_stein_encode", {{"target", "y"}, {"model", "shrunk"}}),
                      ContainsSubstring("'model' must be 'independent', 'pooled' or 'beta'"));
  // binary is refused on purpose: it is not reproducible from the formula.
  REQUIRE_THROWS_WITH(fft::make_kernel("james_stein_encode", {{"target", "y"}, {"model", "binary"}}),
                      ContainsSubstring("not portable"));
  // A kernel reads its own param before handle_unknown and handle_missing, as
  // every target encoder does, so with both wrong the message names its own.
  REQUIRE_THROWS_WITH(
      fft::make_kernel("cat_boost_encode", {{"target", "y"}, {"a", -1}, {"handle_unknown", "sometimes"}}),
      ContainsSubstring("'a' must be a finite number >= 0"));
  REQUIRE_THROWS_WITH(
      fft::make_kernel("james_stein_encode", {{"target", "y"}, {"model", "shrunk"}, {"handle_missing", "sometimes"}}),
      ContainsSubstring("'model' must be 'independent', 'pooled' or 'beta'"));
  REQUIRE(fft::make_kernel("james_stein_encode", target)->needs_fit());
  REQUIRE(fft::make_kernel("cat_boost_encode", target)->params().at("a") == 1.0);
  REQUIRE(fft::make_kernel("james_stein_encode", target)->params().at("model") == "independent");
}

TEST_CASE("a shrinkage encoder binds its target and rewrites the column type") {
  auto kernel = fft::make_kernel("leave_one_out_encode", {{"columns", {"c"}}, {"target", "y"}});
  kernel->stage = "shrink";
  fft::Schema schema = kSchema;
  kernel->resolve(schema);
  REQUIRE(schema.cols == fft::Schema{{{"c", "f64"}, {"y", "f64"}}}.cols);
  // Unfitted, the target has to be read as well, or the source never decodes it.
  REQUIRE(kernel->inputs == std::vector<std::string>{"c", "y"});
  REQUIRE(kernel->outputs == std::vector<std::string>{"c"});

  auto missing = fft::make_kernel("leave_one_out_encode", {{"columns", {"c"}}, {"target", "label"}});
  missing->stage = "shrink";
  fft::Schema without = kSchema;
  REQUIRE_THROWS_WITH(missing->resolve(without), ContainsSubstring("target column 'label' not found in input"));

  auto text = fft::make_kernel("cat_boost_encode", {{"columns", {"c"}}, {"target", "t"}});
  text->stage = "shrink";
  fft::Schema strings{{{"c", "utf8"}, {"t", "utf8"}}};
  REQUIRE_THROWS_WITH(text->resolve(strings),
                      ContainsSubstring("target column 't' is utf8, expected an integer or float column"));

  auto itself = fft::make_kernel("james_stein_encode", {{"columns", {"c", "y"}}, {"target", "y"}});
  itself->stage = "shrink";
  fft::Schema both = kSchema;
  REQUIRE_THROWS_WITH(itself->resolve(both), ContainsSubstring("is also the target"));

  auto numeric = fft::make_kernel("james_stein_encode", {{"columns", {"n"}}, {"target", "y"}});
  numeric->stage = "shrink";
  fft::Schema numbers{{{"n", "i64"}, {"y", "f64"}}};
  REQUIRE_THROWS_WITH(numeric->resolve(numbers), ContainsSubstring("is i64, expected utf8"));
}

TEST_CASE("a shrinkage encoder merges worker states in either order and both paths agree") {
  const nlohmann::json params{{"columns", {"c"}}, {"target", "y"}};
  const std::vector<fft::Batch> batches = sample_batches();
  auto kernel = fit_kernel("leave_one_out_encode", params, kSchema, batches);
  const nlohmann::json state = kernel->state();
  REQUIRE(state.at("prior").get<double>() == Approx(3.6));  // (1 + 3 + 2 + 8 + 4) / 5
  REQUIRE(state.at("values").at("c").at("a").get<double>() == Approx(2.0));
  REQUIRE(state.at("values").at("c").at("b").get<double>() == Approx(14.0 / 3));
  REQUIRE(state.at("null_values").at("c").is_null());
  // Only the merged accumulators may decide a value: this is the monoid contract.
  REQUIRE(fit_kernel("leave_one_out_encode", params, kSchema, {batches[1], batches[0]})->state() == state);
  // Once fitted the target is no longer read, so a serving file can omit it.
  REQUIRE(kernel->inputs == std::vector<std::string>{"c"});

  const std::vector<std::optional<std::string>> values{"a", "b", "zzz", std::nullopt};
  fft::Batch out = kernel->transform(make_batch({"c"}, {utf8_column(values)}));
  const auto& encoded = static_cast<const arrow::DoubleArray&>(*out.cols[0]);
  REQUIRE(encoded.Value(0) == Approx(2.0));
  REQUIRE(encoded.Value(2) == Approx(3.6));  // unseen -> the prior
  REQUIRE(encoded.Value(3) == Approx(3.6));  // a null the fit never saw -> the prior
  for (std::size_t i = 0; i < values.size(); ++i) {
    fft::RowValue value = values[i].has_value() ? fft::RowValue(*values[i]) : fft::RowValue(std::monostate{});
    fft::Row row = kernel->transform_row({{"c", value}});
    REQUIRE(std::get<double>(row[0].second) == encoded.Value(static_cast<int64_t>(i)));
  }
  REQUIRE_THROWS_WITH(kernel->transform_row({{"c", int64_t{1}}}), ContainsSubstring("must be a string or null"));
  REQUIRE_THROWS_WITH(kernel->transform_row({{"n", int64_t{1}}}), ContainsSubstring("missing from record"));
}

TEST_CASE("james_stein_encode fits targets near 1e-155 alike however the workers split them") {
  // REGRESSION: a category's variance multiplies the terms of its exact
  // sums, which depend on the sums' values alone. It used to multiply the
  // partials, which two workers can leave split differently than one, and
  // near 1e-155 the products of two partials underflow and round: with the
  // first targets the value of k0 differed in its last bit between one
  // worker and several. The second targets do the same to a reader of the
  // doubles the sums hold now.
  struct Sample {
    std::vector<double> targets;
    std::vector<std::optional<std::string>> keys;
  };
  const std::vector<Sample> samples{
      {{0x1.0759c9adb04ddp-515, 0x1.4b8e24878adebp-514, -0x1.64223bf27ff24p-514, 0x1.45f8a613e17f4p-513,
        0x1.3f589a1be05f3p-511, 0x1.a492359d2f346p-514, 0x1.fb94454ba0f16p-513, 0x1.8d30c99cac762p-512,
        0x1.f2aef4affa37cp-516, -0x1.2519373039992p-512, -0x1.380021070d57fp-514, 0x1.a3822ae3cedd7p-512},
       {"k0", "k4", "k0", "k1", "k1", "k2", "k0", "k0", "k4", "k3", "k4", "k0"}},
      {{0x1.2982d5355ea91p-517, -0x1.68a85a28a9d15p-515, 0x1.47e7a4707229bp-513, 0x1.6133e9e68afc8p-515,
        0x1.4024db6f22feep-516, -0x1.7954e5f6aa07ap-514, -0x1.d1deb4bb08c46p-517, -0x1.209f7b7872f71p-513,
        0x1.d414906000bf9p-516, 0x1.013937f0c2050p-514, -0x1.ded07c84fb602p-515, -0x1.31c8e5aa803a6p-515},
       {"k0", "k2", "k3", "k4", "k4", "k0", "k2", "k1", "k3", "k0", "k3", "k3"}}};
  const nlohmann::json params{{"columns", {"c"}}, {"target", "y"}};
  for (const Sample& sample : samples) {
    const std::size_t count = sample.targets.size();
    const auto rows = [&](std::size_t from, std::size_t to) {
      const auto first = static_cast<std::ptrdiff_t>(from);
      const auto last = static_cast<std::ptrdiff_t>(to);
      const std::vector<std::optional<std::string>> keys(sample.keys.begin() + first, sample.keys.begin() + last);
      const std::vector<double> targets(sample.targets.begin() + first, sample.targets.begin() + last);
      return make_batch({"c", "y"}, {utf8_column(keys), f64_column(targets)});
    };
    const std::string whole = fit_kernel("james_stein_encode", params, kSchema, {rows(0, count)})->state().dump();
    // Workers of two, three, four and six rows each, in file order and
    // reversed.
    for (const std::size_t size : {std::size_t{2}, std::size_t{3}, std::size_t{4}, std::size_t{6}}) {
      CAPTURE(sample.targets.front(), size);
      std::vector<fft::Batch> batches;
      for (std::size_t first = 0; first < count; first += size) batches.push_back(rows(first, first + size));
      REQUIRE(fit_kernel("james_stein_encode", params, kSchema, batches)->state().dump() == whole);
      std::reverse(batches.begin(), batches.end());
      REQUIRE(fit_kernel("james_stein_encode", params, kSchema, batches)->state().dump() == whole);
    }
    // Two workers, cut at every row, in both orders.
    for (std::size_t cut = 1; cut < count; ++cut) {
      CAPTURE(sample.targets.front(), cut);
      const std::vector<fft::Batch> forward{rows(0, cut), rows(cut, count)};
      const std::vector<fft::Batch> backward{rows(cut, count), rows(0, cut)};
      REQUIRE(fit_kernel("james_stein_encode", params, kSchema, forward)->state().dump() == whole);
      REQUIRE(fit_kernel("james_stein_encode", params, kSchema, backward)->state().dump() == whole);
    }
  }
}

TEST_CASE("each shrinkage model shrinks a category toward the prior") {
  const nlohmann::json params{{"columns", {"c"}}, {"target", "y"}};
  const std::vector<fft::Batch> batches = sample_batches();
  // prior 3.6, var(y) = 7.3, var(a) = 2, var(b) = 9.333..., K = 2 so the
  // (K - 3)/(K - 1) factor is negative and the weight clips to 1.
  auto independent = fit_kernel("james_stein_encode", params, kSchema, batches);
  REQUIRE(value_of(independent, "a") == Approx(2.0));
  REQUIRE(value_of(independent, "b") == Approx(14.0 / 3));

  nlohmann::json beta = params;
  beta["model"] = "beta";
  auto shrunk = fit_kernel("james_stein_encode", beta, kSchema, batches);
  REQUIRE(value_of(shrunk, "a") == Approx(2.0 / 7 * 2.0 + 5.0 / 7 * 3.6));
  REQUIRE(value_of(shrunk, "b") == Approx(3.0 / 8 * (14.0 / 3) + 5.0 / 8 * 3.6));

  nlohmann::json pooled = params;
  pooled["model"] = "pooled";
  auto shared = fit_kernel("james_stein_encode", pooled, kSchema, batches);
  // K - 3 is negative, so B clips to 0 and every category keeps its own mean.
  REQUIRE(value_of(shared, "a") == Approx(2.0));

  // CatBoost smooths with `a` prior-valued observations; a singleton is
  // replaced by the raw prior instead.
  nlohmann::json smoothed = params;
  smoothed["a"] = 2.0;
  auto boosted = fit_kernel("cat_boost_encode", smoothed, kSchema, batches);
  REQUIRE(value_of(boosted, "a") == Approx((4.0 + 3.6 * 2) / (2 + 2)));
  const fft::Batch single = make_batch({"c", "y"}, {utf8_column({"a", "a", "z"}), f64_column({1.0, 3.0, 9.0})});
  auto singleton = fit_kernel("cat_boost_encode", params, kSchema, {single});
  REQUIRE(value_of(singleton, "z") == Approx(13.0 / 3));  // the prior, not (9 + prior)/2
  REQUIRE(fit_kernel("leave_one_out_encode", params, kSchema, {single})
              ->state()
              .at("values")
              .at("c")
              .at("z")
              .get<double>() == Approx(13.0 / 3));
}

// REGRESSION: the variances are formed from the exact sums before they are
// rounded, so moving every target by the same amount moves every value by that
// amount. The sums used to be rounded first, and near 1e9 they cancelled to
// noise: the independent and pooled models then gave the raw category means
// (amber 2.0 where the independent model shrinks it to 2.1187 and the pooled
// one to 2.9637).
TEST_CASE("james_stein_encode gives the same values wherever the target sits") {
  // Two batches, so the merged sums are the ones that decide. A step adds a
  // fraction to each target, which leaves the sums with several partials, so
  // the exact variance has cross terms to form.
  auto fit_at = [](const std::string& model, double offset, double step) {
    auto y = [&](double value, int row) { return value + offset + step * (row + 1); };
    const std::vector<fft::Batch> batches{
        make_batch({"c", "y"}, {utf8_column({"amber", "amber", "blue"}), f64_column({y(1, 0), y(3, 1), y(2, 2)})}),
        make_batch({"c", "y"}, {utf8_column({"blue", "blue", "cyan", "delta", "delta"}),
                                f64_column({y(8, 3), y(4, 4), y(5, 5), y(0, 6), y(6, 7)})})};
    return fit_kernel("james_stein_encode", {{"columns", {"c"}}, {"target", "y"}, {"model", model}}, kSchema, batches);
  };
  const std::vector<std::pair<std::string, double>> shrunk_amber{{"independent", 2.1187214611872145},
                                                                 {"pooled", 2.963728323699422}};
  for (const auto& [model, amber] : shrunk_amber) {
    REQUIRE(value_of(fit_at(model, 0.0, 0.0), "amber") == Approx(amber));  // shrunk, not the raw mean 2.0
    for (double step : {0.0, 0.1}) {
      auto near = fit_at(model, 0.0, step);
      for (double offset : {1e6, 1e9}) {
        auto far = fit_at(model, offset, step);
        for (const char* category : {"amber", "blue", "cyan", "delta"}) {
          REQUIRE(value_of(far, category) - offset == Approx(value_of(near, category)).epsilon(0).margin(1e-6));
        }
      }
    }
  }
}

// REGRESSION: a null counts toward handle_missing='error' even on a row whose
// target is null. Such a row used to be skipped before its null was counted,
// so the fit accepted the null and the fitted encoder then refused its own
// training data at transform.
TEST_CASE("a shrinkage encoder under handle_missing error refuses a null whose target is null") {
  // The null arrives in the second batch, so the merged count is the one
  // that decides.
  const std::vector<fft::Batch> batches{
      make_batch({"c", "y"}, {utf8_column({"a", "b"}), f64_column({1.0, 2.0})}),
      make_batch({"c", "y"},
                 {utf8_column({std::nullopt}), fft::testing::numeric_column<arrow::DoubleType>({std::nullopt})})};
  for (const char* id : {"james_stein_encode", "leave_one_out_encode", "cat_boost_encode"}) {
    REQUIRE_THROWS_WITH(
        fit_kernel(id, {{"columns", {"c"}}, {"target", "y"}, {"handle_missing", "error"}}, kSchema, batches),
        ContainsSubstring("column 'c' has nulls and handle_missing is 'error'"));
    REQUIRE_NOTHROW(fit_kernel(id, {{"columns", {"c"}}, {"target", "y"}}, kSchema, batches));
  }
}

TEST_CASE("a shrinkage encoder counts the null category and skips a null target") {
  const nlohmann::json params{{"columns", {"c"}}, {"target", "y"}};
  const fft::Batch batch = make_batch(
      {"c", "y"}, {utf8_column({"a", "a", std::nullopt, std::nullopt, "b"}), f64_column({1.0, 3.0, 5.0, 9.0, 4.0})});
  auto kernel = fit_kernel("leave_one_out_encode", params, kSchema, {batch});
  REQUIRE(kernel->state().at("null_values").at("c").get<double>() == Approx(7.0));  // (5 + 9) / 2
  REQUIRE(std::get<double>(kernel->transform_row({{"c", fft::RowValue(std::monostate{})}})[0].second) == Approx(7.0));

  nlohmann::json keep_null = params;
  keep_null["handle_missing"] = "return_nan";
  auto nulls = fit_kernel("leave_one_out_encode", keep_null, kSchema, {batch});
  REQUIRE(nulls->state().at("null_values").at("c").is_null());
  REQUIRE(
      std::holds_alternative<std::monostate>(nulls->transform_row({{"c", fft::RowValue(std::monostate{})}})[0].second));
  // The null category still shaped the statistics: the prior is unchanged.
  REQUIRE(nulls->state().at("prior") == kernel->state().at("prior"));

  nlohmann::json strict = params;
  strict["handle_missing"] = "error";
  REQUIRE_THROWS_WITH(fit_kernel("leave_one_out_encode", strict, kSchema, {batch}),
                      ContainsSubstring("has nulls and handle_missing is 'error'"));

  // A row without a target is not there at all, so its category is unknown.
  const fft::Batch partial = make_batch(
      {"c", "y"},
      {utf8_column({"a", "a", "z"}), fft::testing::numeric_column<arrow::DoubleType>({1.0, 3.0, std::nullopt})});
  auto skipped = fit_kernel("leave_one_out_encode", params, kSchema, {partial});
  REQUIRE(skipped->state().at("prior").get<double>() == Approx(2.0));
  REQUIRE_FALSE(skipped->state().at("values").at("c").contains("z"));

  nlohmann::json unseen = params;
  unseen["handle_unknown"] = "error";
  auto refuses = fit_kernel("leave_one_out_encode", unseen, kSchema, {batch});
  REQUIRE_THROWS_WITH(refuses->transform_row({{"c", std::string("zzz")}}),
                      ContainsSubstring("category that was not seen at fit"));
}

// REGRESSION: the fit refuses a target with no non-null values. A mis-joined
// label column (all nulls after a left join that missed) used to fit, save,
// load and serve a constant 0.0 for every category, with nothing anywhere
// saying so. Worse, handle_unknown='error' then reported a category from the
// fit data as unseen, so the message pointed the operator at the wrong thing.
TEST_CASE("a shrinkage encoder refuses a target with no non-null values") {
  const fft::Batch unlabelled = make_batch(
      {"c", "y"},
      {utf8_column({"a", "b"}), fft::testing::numeric_column<arrow::DoubleType>({std::nullopt, std::nullopt})});
  for (const char* id : {"james_stein_encode", "leave_one_out_encode", "cat_boost_encode"}) {
    REQUIRE_THROWS_WITH(fit_kernel(id, {{"columns", {"c"}}, {"target", "y"}}, kSchema, {unlabelled}),
                        ContainsSubstring("target column 'y' has no non-null values to learn from"));
  }
}

// REGRESSION: a large enough smoothing constant sends the encoded value to
// infinity. finalize used to accept it, though load_state called it invalid
// and state() wrote it as JSON null, so the artifact was corrupt in a way
// indistinguishable from 'this category had no value'.
TEST_CASE("cat_boost_encode refuses an encoding that is not finite") {
  // The target is ordinary, but prior * a overflows f64 on its own, and so
  // then does (sum + prior * a) / (n + a).
  const fft::Batch batch = make_batch({"c", "y"}, {utf8_column({"a", "a", "b"}), f64_column({10.0, 20.0, 30.0})});
  REQUIRE_THROWS_WITH(
      fit_kernel("cat_boost_encode", {{"columns", {"c"}}, {"target", "y"}, {"a", 1e308}}, kSchema, {batch}),
      ContainsSubstring("column 'c' encodes to a value that is not finite; check 'a'"));
}

// REGRESSION: 'error' is the strict option, so it refuses a null wherever one
// turns up. It used to let a serving null through as the prior.
TEST_CASE("a shrinkage encoder refuses a null at transform under handle_missing error") {
  const fft::Batch batch = make_batch({"c", "y"}, {utf8_column({"a", "b"}), f64_column({1.0, 2.0})});
  for (const char* id : {"james_stein_encode", "leave_one_out_encode", "cat_boost_encode"}) {
    auto kernel = fit_kernel(id, {{"columns", {"c"}}, {"target", "y"}, {"handle_missing", "error"}}, kSchema, {batch});
    REQUIRE_THROWS_WITH(kernel->transform(make_batch({"c"}, {utf8_column({"a", std::nullopt})})),
                        ContainsSubstring("column 'c' has a null and handle_missing is 'error'"));
    REQUIRE_THROWS_WITH(kernel->transform_row({{"c", fft::RowValue(std::monostate{})}}),
                        ContainsSubstring("column 'c' has a null and handle_missing is 'error'"));
  }
}

TEST_CASE("shrinkage load_state rejects a malformed table") {
  auto fresh = [] {
    auto kernel = fft::make_kernel("cat_boost_encode", {{"columns", {"c"}}, {"target", "y"}});
    kernel->stage = "shrink";
    return kernel;
  };
  const nlohmann::json good{
      {"columns", {"c"}}, {"prior", 3.5}, {"values", {{"c", {{"x", 2.0}}}}}, {"null_values", {{"c", nullptr}}}};
  REQUIRE_NOTHROW(fresh()->load_state(good));
  auto broken = [&](auto mutate, const std::string& message) {
    nlohmann::json state = good;
    mutate(state);
    REQUIRE_THROWS_WITH(fresh()->load_state(state), ContainsSubstring(message));
  };
  broken([](nlohmann::json& state) { state.erase("prior"); }, "must hold a numeric 'prior'");
  broken([](nlohmann::json& state) { state["prior"] = "three"; }, "must hold a numeric 'prior'");
  broken([](nlohmann::json& state) { state["values"]["c"] = 5; }, "must be an object");
  broken([](nlohmann::json& state) { state["values"]["c"]["x"] = "two"; }, "must be a finite number");
  broken([](nlohmann::json& state) { state["values"] = {{"d", {{"x", 2.0}}}}; }, "must be an object");
  broken([](nlohmann::json& state) { state["values"].erase("c"); },
         "state 'columns' does not name the same columns as the fitted table");
  broken([](nlohmann::json& state) { state.erase("columns"); }, "state must hold a 'columns' list");
  broken([](nlohmann::json& state) { state["null_values"]["c"] = "two"; }, "must be a finite number or null");
  // REGRESSION: the null category is the entry a hand-edited manifest is
  // likeliest to mangle, and a wrong shape used to read as a legitimate
  // 'no null category' rather than as an error.
  broken([](nlohmann::json& state) { state["null_values"] = 2.0; }, "'null_values' must be an object");
  broken([](nlohmann::json& state) { state["null_values"].erase("c"); }, "null_values has no entry for column 'c'");
}

TEST_CASE("shrinkage dictionary fits preserve target exclusion exact sums and loaded outputs") {
  const std::vector<std::optional<int64_t>> indices{0, 1, 4, 3, std::nullopt, 2, 0, 1, 2, 4, 0, 1, 3, 2, 0, 1};
  const auto first = fft::testing::dictionary_fixture<arrow::Int8Type>(
      {"a", "b", "a", std::nullopt, "targetless", "unused"}, indices, true, 3, 2);
  const auto second = fft::testing::dictionary_fixture<arrow::Int32Type>(
      {"c", "a", "c", std::nullopt, "targetless", "unused-new"}, indices);
  const auto target = fft::testing::numeric_column<arrow::DoubleType>(
      {1e16, 1.0, std::nullopt, 2.0, -0.0, -1e16, 3.0, -1.0, 5.0, std::nullopt, 2.0, 0.0, 1.0, -3.0, 1.0, 0.0});
  for (const std::string& id : {"james_stein_encode", "leave_one_out_encode", "cat_boost_encode"}) {
    CAPTURE(id);
    const nlohmann::json params{{"columns", {"c"}}, {"target", "y"}, {"handle_unknown", "error"}};
    auto dictionary =
        fit_kernel(id, params, kSchema,
                   {make_batch({"c", "y"}, {first.encoded, target}), make_batch({"c", "y"}, {second.encoded, target})});
    auto plain =
        fit_kernel(id, params, kSchema,
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
  }
}
