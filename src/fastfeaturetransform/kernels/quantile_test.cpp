// Explicit C++ tests for quantile_encode and summary_encode: the params, the
// schema rewrite, the accumulator-merge (monoid) contract, the two transform
// paths agreeing, and load_state.

#include <arrow/array.h>

#include <algorithm>
#include <bit>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "fastfeaturetransform/kernel.hpp"
#include "fastfeaturetransform/kernels/dictionary_test_support.hpp"
#include "fastfeaturetransform/spill.hpp"
#include "fastfeaturetransform/test_support.hpp"

namespace fft = fastfeaturetransform;
using Catch::Approx;
using Catch::Matchers::ContainsSubstring;
using fft::testing::f64_column;
using fft::testing::make_batch;
using fft::testing::utf8_column;

namespace {

// Fit a kernel with one worker state per batch.
std::unique_ptr<fft::Kernel> fit_quantile(const std::string& id, const nlohmann::json& params, fft::Schema schema,
                                          const std::vector<fft::Batch>& batches) {
  auto kernel = fft::make_kernel(id, params);
  kernel->stage = "encode";
  kernel->resolve(schema);
  std::vector<std::unique_ptr<fft::AccState>> states;
  for (const fft::Batch& batch : batches) {
    states.push_back(kernel->make_state());
    kernel->consume(*states.back(), batch);
  }
  kernel->finalize(std::move(states));
  return kernel;
}

const fft::Schema kSchema{{{"color", "utf8"}, {"y", "f64"}}};

// 'a' -> [1, 2], 'b' -> [3, 4, 5], null -> [6], 'c' -> [7]; prior = median(1..7) = 4.
fft::Batch left() {
  return make_batch({"color", "y"}, {utf8_column({"a", "a", "b", "b"}), f64_column({1.0, 2.0, 3.0, 4.0})});
}
fft::Batch right() {
  return make_batch({"color", "y"}, {utf8_column({"b", std::nullopt, "c"}), f64_column({5.0, 6.0, 7.0})});
}

}  // namespace

TEST_CASE("supervised quantiles spill grouped values and priors exactly") {
  std::vector<std::optional<std::string>> keys;
  std::vector<std::optional<double>> targets;
  for (int i = 0; i < 4097; ++i) {
    keys.emplace_back(i % 17 == 0 ? std::nullopt : std::optional<std::string>{"key-" + std::to_string(i % 5)});
    targets.emplace_back(i % 29 == 0 ? std::nullopt
                                     : std::optional<double>{static_cast<double>((i * 13) % 701 - 350) / 32.0});
  }
  const fft::Batch all =
      make_batch({"color", "y"}, {utf8_column(keys), fft::testing::numeric_column<arrow::DoubleType>(targets)});
  for (const std::string id : {"quantile_encode", "summary_encode"}) {
    nlohmann::json params{{"columns", {"color"}}, {"target", "y"}, {"m", 2.0}};
    if (id == "summary_encode") params["quantiles"] = {0.01, 0.5, 0.99};
    const auto expected = fit_quantile(id, params, kSchema, {all})->state();
    fft::testing::TempDir temporary;
    {
      auto memory = std::make_shared<fft::FitMemory>(fft::FitMemoryOptions{512, temporary.path.string()});
      fft::FitMemoryScope scope(memory);
      std::vector<fft::Batch> batches;
      for (int64_t offset = 0; offset < all.rows; offset += 257) {
        const int64_t count = std::min(int64_t{257}, all.rows - offset);
        batches.push_back(
            make_batch(all.names, {all.cols[0]->Slice(offset, count), all.cols[1]->Slice(offset, count)}));
      }
      std::reverse(batches.begin(), batches.end());
      const auto actual = fit_quantile(id, params, kSchema, batches);
      REQUIRE(actual->state() == expected);
      REQUIRE(memory->spill_files_created() > 0);
      REQUIRE(memory->peak_bytes_in_memory() <= 512);
      REQUIRE(memory->bytes_in_memory() == 0);
      REQUIRE(memory->open_files() == 0);
    }
    REQUIRE(std::filesystem::is_empty(temporary.path));
  }
}

TEST_CASE("quantile_encode validates its params") {
  REQUIRE_THROWS_WITH((fft::make_kernel("quantile_encode", {{"columns", {"c"}}})),
                      ContainsSubstring("'target' must be a column name"));
  REQUIRE_THROWS_WITH((fft::make_kernel("quantile_encode", {{"target", "y"}, {"quantile", 1.5}})),
                      ContainsSubstring("'quantile' must be a number between 0 and 1"));
  REQUIRE_THROWS_WITH((fft::make_kernel("quantile_encode", {{"target", "y"}, {"m", -1}})),
                      ContainsSubstring("'m' must be a finite number >= 0"));
  REQUIRE_THROWS_WITH((fft::make_kernel("quantile_encode", {{"target", "y"}, {"handle_unknown", "sometimes"}})),
                      ContainsSubstring("'handle_unknown' must be"));
  REQUIRE_THROWS_WITH((fft::make_kernel("quantile_encode", {{"target", "y"}, {"handle_missing", "sometimes"}})),
                      ContainsSubstring("'handle_missing' must be"));
  REQUIRE_THROWS_WITH((fft::make_kernel("summary_encode", {{"target", "y"}, {"quantiles", nlohmann::json::array()}})),
                      ContainsSubstring("'quantiles' must be a non-empty list"));
  REQUIRE_THROWS_WITH((fft::make_kernel("summary_encode", {{"target", "y"}, {"quantiles", {0.12, 0.125}}})),
                      ContainsSubstring("round to the same percentile"));
  REQUIRE(fft::make_kernel("quantile_encode", {{"target", "y"}})->needs_fit());
  REQUIRE(fft::make_kernel("summary_encode", {{"target", "y"}})->params().at("quantiles") ==
          nlohmann::json({0.25, 0.75}));
}

TEST_CASE("quantile_encode rewrites the schema and checks the target") {
  auto kernel = fft::make_kernel("quantile_encode", {{"columns", {"color"}}, {"target", "y"}});
  kernel->stage = "encode";
  fft::Schema schema = kSchema;
  kernel->resolve(schema);
  REQUIRE(schema.cols == fft::Schema{{{"color", "f64"}, {"y", "f64"}}}.cols);
  REQUIRE(kernel->inputs == std::vector<std::string>{"color", "y"});  // the target is read while unfitted

  auto summary =
      fft::make_kernel("summary_encode", {{"columns", {"color"}}, {"target", "y"}, {"quantiles", {0.5, 0.9}}});
  summary->stage = "encode";
  fft::Schema wide{{{"color", "utf8"}, {"other", "i64"}, {"y", "f64"}}};
  summary->resolve(wide);
  // The generated columns take the source column's place; everything else stays.
  REQUIRE(wide.cols == fft::Schema{{{"color_50", "f64"}, {"color_90", "f64"}, {"other", "i64"}, {"y", "f64"}}}.cols);

  auto missing = fft::make_kernel("quantile_encode", {{"columns", {"color"}}, {"target", "nope"}});
  missing->stage = "encode";
  fft::Schema fresh = kSchema;
  REQUIRE_THROWS_WITH(missing->resolve(fresh), ContainsSubstring("target column 'nope' not found in input"));

  auto text = fft::make_kernel("quantile_encode", {{"columns", {"color"}}, {"target", "size"}});
  text->stage = "encode";
  fft::Schema strings{{{"color", "utf8"}, {"size", "utf8"}}};
  REQUIRE_THROWS_WITH(text->resolve(strings), ContainsSubstring("is utf8, expected an integer or float column"));

  auto itself = fft::make_kernel("quantile_encode", {{"columns", {"y"}}, {"target", "y"}});
  itself->stage = "encode";
  fft::Schema same = kSchema;
  REQUIRE_THROWS_WITH(itself->resolve(same), ContainsSubstring("cannot encode its target"));
}

TEST_CASE("quantile_encode merges worker buffers and both paths agree") {
  const nlohmann::json params{{"columns", {"color"}}, {"target", "y"}};
  auto kernel = fit_quantile("quantile_encode", params, kSchema, {left(), right()});
  const nlohmann::json state = kernel->state();
  REQUIRE(state.at("priors").at(0).get<double>() == Approx(4.0));
  REQUIRE(state.at("categories").at("color") == nlohmann::json({"a", "b", "c"}));
  // 'a': median 1.5 over 2 values -> (2*1.5 + 4)/3; 'b': median 4 over 3 -> (3*4 + 4)/4; 'c': (7 + 4)/2.
  REQUIRE(state.at("encodings").at("color").at(0).get<double>() == Approx(7.0 / 3.0));
  REQUIRE(state.at("encodings").at("color").at(1).get<double>() == Approx(4.0));
  REQUIRE(state.at("encodings").at("color").at(2).get<double>() == Approx(5.5));
  REQUIRE(state.at("nulls").at("color").get<double>() == Approx(5.0));  // the null category saw 6 alone
  // Only the merged buffers may decide a value: this is the monoid contract.
  REQUIRE(fit_quantile("quantile_encode", params, kSchema, {right(), left()})->state() == state);
  // And the fitted encoder no longer reads the target.
  REQUIRE(kernel->inputs == std::vector<std::string>{"color"});

  const std::vector<std::optional<std::string>> values{"a", "b", "zzz", std::nullopt};
  fft::Batch out = kernel->transform(make_batch({"color"}, {utf8_column(values)}));
  const auto& encoded = static_cast<const arrow::DoubleArray&>(*out.cols[0]);
  REQUIRE(encoded.Value(2) == Approx(4.0));  // unseen -> the prior
  REQUIRE(encoded.Value(3) == Approx(5.0));  // the null was a category at fit
  for (std::size_t i = 0; i < values.size(); ++i) {
    fft::RowValue value = values[i].has_value() ? fft::RowValue(*values[i]) : fft::RowValue(std::monostate{});
    fft::Row row = kernel->transform_row({{"color", value}});
    REQUIRE(std::get<double>(row[0].second) == encoded.Value(static_cast<int64_t>(i)));
  }
  REQUIRE_THROWS_WITH(kernel->transform_row({{"color", int64_t{1}}}), ContainsSubstring("must be a string or null"));
  REQUIRE_THROWS_WITH(kernel->transform_row({{"y", 1.0}}), ContainsSubstring("missing from record"));
}

TEST_CASE("summary_encode replaces a column with one per quantile in both paths") {
  const nlohmann::json params{{"columns", {"color"}}, {"target", "y"}, {"quantiles", {0.25, 0.75}}, {"m", 0}};
  auto kernel = fit_quantile("summary_encode", params, kSchema, {left(), right()});
  REQUIRE(kernel->outputs == std::vector<std::string>{"color_25", "color_75"});
  const nlohmann::json state = kernel->state();
  REQUIRE(fit_quantile("summary_encode", params, kSchema, {right(), left()})->state() == state);
  // m = 0 is the raw quantile: 'b' holds [3, 4, 5], so 25% is 3.5 and 75% is 4.5.
  REQUIRE(state.at("encodings").at("color_25").at(1).get<double>() == Approx(3.5));
  REQUIRE(state.at("encodings").at("color_75").at(1).get<double>() == Approx(4.5));

  fft::Batch out =
      kernel->transform(make_batch({"color", "y"}, {utf8_column({"a", "b", "zzz"}), f64_column({1.0, 2.0, 3.0})}));
  REQUIRE(out.names == std::vector<std::string>{"color_25", "color_75", "y"});
  const auto& low = static_cast<const arrow::DoubleArray&>(*out.cols[0]);
  const auto& high = static_cast<const arrow::DoubleArray&>(*out.cols[1]);
  for (int64_t row = 0; row < 3; ++row) {
    fft::Row record = kernel->transform_row({{"color", fft::RowValue(std::string(row == 2   ? "zzz"
                                                                                 : row == 1 ? "b"
                                                                                            : "a"))},
                                             {"y", fft::RowValue(1.0)}});
    REQUIRE(record.size() == 3);
    REQUIRE(record[0].first == "color_25");
    REQUIRE(record[1].first == "color_75");
    REQUIRE(std::get<double>(record[0].second) == low.Value(row));
    REQUIRE(std::get<double>(record[1].second) == high.Value(row));
  }
  REQUIRE_THROWS_WITH(kernel->transform_row({{"color", std::string("a")}, {"color_25", 1.0}}),
                      ContainsSubstring("already present in record"));
}

// REGRESSION: the fit refuses a target with no non-null values. A mis-joined
// label column (all nulls after a left join that missed) used to fit, save,
// load and serve a constant 0.0 for every category, with nothing anywhere
// saying so.
TEST_CASE("a quantile encoder refuses a target with no non-null values") {
  const fft::Batch unlabelled = make_batch(
      {"color", "y"},
      {utf8_column({"a", "b"}), fft::testing::numeric_column<arrow::DoubleType>({std::nullopt, std::nullopt})});
  for (const char* id : {"quantile_encode", "summary_encode"}) {
    REQUIRE_THROWS_WITH(fit_quantile(id, {{"columns", {"color"}}, {"target", "y"}}, kSchema, {unlabelled}),
                        ContainsSubstring("target column 'y' has no non-null values to learn from"));
  }
}

// REGRESSION: a large enough m sends prior * m, and the whole quotient, out of
// f64. finalize used to accept it, though load_state called it invalid and
// state() wrote it as JSON null.
TEST_CASE("quantile_encode refuses an encoding that is not finite") {
  // The target is ordinary, but prior * m overflows f64 on its own, and so
  // then does (n * q + prior * m) / (n + m).
  const fft::Batch batch = make_batch({"color", "y"}, {utf8_column({"a", "b"}), f64_column({10.0, 20.0})});
  REQUIRE_THROWS_WITH(
      fit_quantile("quantile_encode", {{"columns", {"color"}}, {"target", "y"}, {"m", 1e308}}, kSchema, {batch}),
      ContainsSubstring("column 'color' encodes to a value that is not finite; check 'm'"));
}

TEST_CASE("a quantile encoder refuses a target that spans past the range of f64") {
  // REGRESSION: a quantile between two target values more than f64's largest
  // apart is a fit-time error naming the target. numpy's lerp overflows there,
  // and the fit used to keep such a prior, which saved as JSON null, or to
  // blame 'm' when a category's quantile overflowed.
  const fft::Batch unseen =
      make_batch({"color", "y"}, {utf8_column({std::nullopt, std::nullopt}), f64_column({-1.7e308, 1.7e308})});
  for (const char* id : {"quantile_encode", "summary_encode"}) {
    REQUIRE_THROWS_WITH(fit_quantile(id, {{"columns", {"color"}}, {"target", "y"}, {"handle_missing", "return_nan"}},
                                     kSchema, {unseen}),
                        ContainsSubstring("encode: target column 'y' spans past the range of f64"));
  }
  const fft::Batch seen =
      make_batch({"color", "y"}, {utf8_column({"a", "a", "b"}), f64_column({-1.7e308, 1.7e308, 0.0})});
  REQUIRE_THROWS_WITH(
      fit_quantile("quantile_encode", {{"columns", {"color"}}, {"target", "y"}, {"m", 0.0}}, kSchema, {seen}),
      ContainsSubstring("encode: target column 'y' spans past the range of f64"));
  // Two rows of 1e308: the quantile is finite, but n times it is not, and no
  // m fixes that, so the message names the target rather than 'm'.
  const fft::Batch large = make_batch({"color", "y"}, {utf8_column({"a", "a"}), f64_column({1e308, 1e308})});
  REQUIRE_THROWS_WITH(
      fit_quantile("quantile_encode", {{"columns", {"color"}}, {"target", "y"}}, kSchema, {large}),
      ContainsSubstring("target column 'y' is too large to smooth: 2 times its quantile is past the range of f64"));
}

TEST_CASE("quantile_encode load_state rejects a malformed table") {
  auto fresh = [] {
    auto kernel = fft::make_kernel("quantile_encode", {{"columns", {"color"}}, {"target", "y"}});
    kernel->stage = "encode";
    return kernel;
  };
  const nlohmann::json good{{"columns", {"color"}},
                            {"priors", {4.0}},
                            {"categories", {{"color", {"a", "b"}}}},
                            {"encodings", {{"color", {1.0, 2.0}}}},
                            {"nulls", {{"color", nullptr}}}};
  REQUIRE_NOTHROW(fresh()->load_state(good));
  auto broken = [&](auto mutate, const std::string& message) {
    nlohmann::json state = good;
    mutate(state);
    REQUIRE_THROWS_WITH(fresh()->load_state(state), ContainsSubstring(message));
  };
  broken([](nlohmann::json& state) { state.erase("nulls"); }, "state must hold 'priors'");
  broken([](nlohmann::json& state) { state["priors"] = nlohmann::json::array(); }, "one value per quantile");
  broken([](nlohmann::json& state) { state["priors"] = {"four"}; }, "must hold finite numbers");
  broken([](nlohmann::json& state) { state["categories"]["color"] = {"b", "a"}; }, "must be sorted and unique");
  broken([](nlohmann::json& state) { state["categories"]["color"] = 5; }, "must be a list of category names");
  broken([](nlohmann::json& state) { state["encodings"]["color"] = {1.0}; }, "one value per category");
  broken([](nlohmann::json& state) { state["encodings"]["color"] = {1.0, "two"}; }, "must hold finite numbers");
}

TEST_CASE("quantile dictionary fits exclude null targets while still enforcing missing errors") {
  const std::vector<std::optional<int64_t>> indices{0, 1, 4, 3, std::nullopt, 2, 0, 1, 2, 4, 0, 1, 3, 2, 0, 1};
  const auto first = fft::testing::dictionary_fixture<arrow::Int8Type>(
      {"a", "b", "a", std::nullopt, "targetless", "unused"}, indices, true, 3, 2);
  const auto second = fft::testing::dictionary_fixture<arrow::Int64Type>(
      {"c", "a", "c", std::nullopt, "targetless", "unused-new"}, indices);
  const auto target = fft::testing::numeric_column<arrow::DoubleType>(
      {1.0, 3.0, std::nullopt, 2.0, -0.0, -1.0, 3.0, -1.0, 5.0, std::nullopt, 2.0, 0.0, 1.0, -3.0, 1.0, 0.0});
  for (const std::string& id : {"quantile_encode", "summary_encode"}) {
    CAPTURE(id);
    const nlohmann::json params{{"columns", {"color"}}, {"target", "y"}, {"handle_unknown", "error"}};
    auto dictionary = fit_quantile(
        id, params, kSchema,
        {make_batch({"color", "y"}, {first.encoded, target}), make_batch({"color", "y"}, {second.encoded, target})});
    auto plain = fit_quantile(
        id, params, kSchema,
        {make_batch({"color", "y"}, {first.plain, target}), make_batch({"color", "y"}, {second.plain, target})});
    REQUIRE(dictionary->state().dump() == plain->state().dump());
    REQUIRE(dictionary->state().at("categories").at("color") == nlohmann::json({"a", "b", "c"}));
    const auto serving =
        fft::testing::dictionary_fixture({"a", "b", "c", "unseen", std::nullopt}, {0, 1, 2, 4, 0, 1, 2, 4});
    auto loaded = fft::make_kernel(id, params);
    loaded->load_state(dictionary->state());
    const auto actual = loaded->transform(make_batch({"color"}, {serving.encoded}));
    const auto expected = plain->transform(make_batch({"color"}, {serving.plain}));
    REQUIRE(actual.names == expected.names);
    for (std::size_t column = 0; column < actual.cols.size(); ++column) {
      REQUIRE(actual.cols[column]->ValidateFull().ok());
      REQUIRE(actual.cols[column]->Equals(expected.cols[column]));
    }
    const auto missing = fft::testing::dictionary_fixture({"a", std::nullopt, "unused"}, {0, 1, 0, 0});
    const auto labels = fft::testing::numeric_column<arrow::DoubleType>({1.0, std::nullopt, 2.0, 3.0});
    REQUIRE_THROWS_WITH(fit_quantile(id, {{"columns", {"color"}}, {"target", "y"}, {"handle_missing", "error"}},
                                     kSchema, {make_batch({"color", "y"}, {missing.encoded, labels})}),
                        ContainsSubstring("has nulls and handle_missing is 'error'"));
  }
}

TEST_CASE("quantile rank selection agrees bit for bit with sorted interpolation across worker splits") {
  // Use a complete sort as the independent reference. The buffers range from
  // 21 values to 2049, so both the whole sort and selection run, with ties,
  // signed zeros and tails.
  const auto sorted_quantile = [](std::vector<double> values, double q) {
    std::sort(values.begin(), values.end());
    const double index = q * static_cast<double>(values.size() - 1);
    if (index <= 0.0) return values.front();
    if (index >= static_cast<double>(values.size() - 1)) return values.back();
    const auto lower = static_cast<std::size_t>(index);
    const double fraction = index - static_cast<double>(lower);
    const double difference = values[lower + 1] - values[lower];
    return fraction >= 0.5 ? values[lower + 1] - difference * (1.0 - fraction) : values[lower] + difference * fraction;
  };
  for (int64_t rows : {65, 127, 2049}) {
    std::vector<std::optional<std::string>> categories;
    std::vector<double> target;
    std::vector<std::vector<double>> buffers(3);
    for (int64_t i = 0; i < rows; ++i) {
      const auto category = static_cast<std::size_t>(i % 3);
      categories.push_back(category == 2 ? std::nullopt : std::optional<std::string>{category == 0 ? "a" : "b"});
      double value = static_cast<double>((i * 7919) % 4099 - 2049) / 10.0;
      if (i % 17 == 0) value = -0.0;
      target.push_back(value);
      buffers[category].push_back(value == 0.0 ? 0.0 : value);
    }
    std::vector<double> canonical = target;
    for (double& value : canonical) {
      if (value == 0.0) value = 0.0;
    }
    const auto input = make_batch({"color", "y"}, {utf8_column(categories), f64_column(target)});
    const int64_t split = rows / 2;
    const auto head = make_batch({"color", "y"}, {input.cols[0]->Slice(0, split), input.cols[1]->Slice(0, split)});
    const auto tail = make_batch({"color", "y"}, {input.cols[0]->Slice(split), input.cols[1]->Slice(split)});
    for (const std::vector<double>& quantiles :
         {std::vector<double>{0.0}, std::vector<double>{0.5}, std::vector<double>{1.0},
          std::vector<double>{0.0, 0.01, 0.25, 0.5, 0.7501, 0.99, 1.0}}) {
      const std::string id = quantiles.size() == 1 ? "quantile_encode" : "summary_encode";
      nlohmann::json params{{"columns", {"color"}}, {"target", "y"}, {"m", 0.75}};
      if (quantiles.size() == 1)
        params["quantile"] = quantiles.front();
      else
        params["quantiles"] = quantiles;
      CAPTURE(rows, quantiles);
      auto kernel = fit_quantile(id, params, kSchema, {input});
      REQUIRE(kernel->state().dump() == fit_quantile(id, params, kSchema, {tail, head})->state().dump());
      const auto actual = kernel->transform(make_batch({"color"}, {utf8_column({"a", "b", std::nullopt})}));
      for (std::size_t q = 0; q < quantiles.size(); ++q) {
        const double prior = sorted_quantile(canonical, quantiles[q]);
        REQUIRE(std::bit_cast<uint64_t>(kernel->state().at("priors").at(q).get<double>()) ==
                std::bit_cast<uint64_t>(prior));
        const auto& output = static_cast<const arrow::DoubleArray&>(*actual.cols[q]);
        for (std::size_t category = 0; category < buffers.size(); ++category) {
          const double n = static_cast<double>(buffers[category].size());
          const double expected = (n * sorted_quantile(buffers[category], quantiles[q]) + prior * 0.75) / (n + 0.75);
          REQUIRE(std::bit_cast<uint64_t>(output.Value(static_cast<int64_t>(category))) ==
                  std::bit_cast<uint64_t>(expected));
        }
      }
    }
  }
}
