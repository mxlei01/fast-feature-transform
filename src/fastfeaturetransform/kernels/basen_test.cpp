// Explicit C++ tests for the positional encoders: the digit widths, the code
// words of all three spellings, the data-dependent output columns, the
// accumulator-merge contract, and load_state.

#include <arrow/array.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "fastfeaturetransform/kernel.hpp"
#include "fastfeaturetransform/test_support.hpp"

namespace fft = fastfeaturetransform;
using Catch::Matchers::ContainsSubstring;
using fft::testing::make_batch;
using fft::testing::utf8_column;

namespace {

std::unique_ptr<fft::Kernel> fit_kernel(const std::string& id, const nlohmann::json& params, fft::Schema& schema,
                                        const std::vector<fft::Batch>& batches) {
  auto kernel = fft::make_kernel(id, params);
  kernel->stage = "enc";
  kernel->resolve(schema);
  std::vector<std::unique_ptr<fft::AccState>> states;
  for (const fft::Batch& batch : batches) {
    states.push_back(kernel->make_state());
    kernel->consume(*states.back(), batch);
  }
  kernel->finalize(std::move(states));
  return kernel;
}

// The digits of one row of a transformed batch, in output order.
std::vector<std::optional<int64_t>> word_at(const fft::Batch& batch, int64_t row) {
  std::vector<std::optional<int64_t>> word;
  for (const auto& column : batch.cols) {
    const auto& digits = static_cast<const arrow::Int64Array&>(*column);
    word.push_back(digits.IsNull(row) ? std::optional<int64_t>{} : digits.Value(row));
  }
  return word;
}

}  // namespace

TEST_CASE("the positional encoders validate their params") {
  REQUIRE_THROWS_WITH((fft::make_kernel("basen_encode", {{"base", 2.5}})),
                      ContainsSubstring("'base' must be an integer"));
  REQUIRE_THROWS_WITH((fft::make_kernel("basen_encode", {{"base", 0}})), ContainsSubstring("'base' must be 1 or more"));
  REQUIRE_THROWS_WITH((fft::make_kernel("basen_encode", {{"handle_missing", "sometimes"}})),
                      ContainsSubstring("'handle_missing' must be"));
  REQUIRE_THROWS_WITH((fft::make_kernel("gray_encode", {{"handle_unknown", "maybe"}})),
                      ContainsSubstring("'handle_unknown' must be"));
  // All three take 'indicator', as category_encoders' BaseNEncoder does.
  for (const char* id : {"basen_encode", "binary_encode", "gray_encode"}) {
    REQUIRE_NOTHROW((fft::make_kernel(id, {{"handle_unknown", "indicator"}})));
  }
  // Only base-N has a base: upstream fixes it at 2 for the other two.
  REQUIRE(fft::make_kernel("basen_encode", {{"base", 3}})->params().at("base") == 3);
  REQUIRE_FALSE(fft::make_kernel("binary_encode", nlohmann::json::object())->params().contains("base"));
  REQUIRE_FALSE(fft::make_kernel("gray_encode", {{"base", 3}})->params().contains("base"));
}

TEST_CASE("binary_encode declares one column per digit only once it has fitted") {
  auto kernel = fft::make_kernel("binary_encode", {{"columns", {"c"}}});
  kernel->stage = "enc";
  fft::Schema before{{{"c", "utf8"}, {"n", "i64"}}};
  kernel->resolve(before);
  // Unfitted: the column is consumed and nothing is emitted yet, because the
  // width follows from the category count; the engine resolves again after
  // the fit pass.
  REQUIRE(before.cols == fft::Schema{{{"n", "i64"}}}.cols);
  REQUIRE(kernel->outputs.empty());

  auto state = kernel->make_state();
  kernel->consume(*state, make_batch({"c"}, {utf8_column({"b", "a", "d", "c"})}));
  std::vector<std::unique_ptr<fft::AccState>> states;
  states.push_back(std::move(state));
  kernel->finalize(std::move(states));

  fft::Schema after{{{"c", "utf8"}, {"n", "i64"}}};
  kernel->resolve(after);
  // Four categories need three digits: the width is sized for n + 1 levels,
  // because the code word 0 is reserved for an unknown.
  REQUIRE(after.cols == fft::Schema{{{"n", "i64"}, {"c_0", "i64"}, {"c_1", "i64"}, {"c_2", "i64"}}}.cols);
  REQUIRE(kernel->outputs == std::vector<std::string>{"c_0", "c_1", "c_2"});

  fft::Batch out = kernel->transform(make_batch({"c"}, {utf8_column({"a", "b", "c", "d", "zz"})}));
  REQUIRE(out.names == std::vector<std::string>{"c_0", "c_1", "c_2"});
  REQUIRE(word_at(out, 0) == std::vector<std::optional<int64_t>>{0, 0, 1});
  REQUIRE(word_at(out, 1) == std::vector<std::optional<int64_t>>{0, 1, 0});
  REQUIRE(word_at(out, 2) == std::vector<std::optional<int64_t>>{0, 1, 1});
  REQUIRE(word_at(out, 3) == std::vector<std::optional<int64_t>>{1, 0, 0});
  REQUIRE(word_at(out, 4) == std::vector<std::optional<int64_t>>{0, 0, 0});  // unknown: the free code word

  // Row for row, the record path agrees with the batch path.
  const std::vector<std::string> values{"a", "b", "c", "d", "zz"};
  for (std::size_t i = 0; i < values.size(); ++i) {
    fft::Row row = kernel->transform_row({{"c", values[i]}});
    REQUIRE(row.size() == 3);
    for (std::size_t digit = 0; digit < 3; ++digit) {
      REQUIRE(row[digit].first == "c_" + std::to_string(digit));
      REQUIRE(std::get<int64_t>(row[digit].second) == word_at(out, static_cast<int64_t>(i))[digit]);
    }
  }
  REQUIRE_THROWS_WITH(kernel->transform_row({{"n", int64_t{1}}}), ContainsSubstring("missing from record"));
}

TEST_CASE("the fitted state does not depend on the order the accumulators merged in") {
  const fft::Batch first = make_batch({"c"}, {utf8_column({"delta", "amber"})});
  const fft::Batch second = make_batch({"c"}, {utf8_column({"blue", std::nullopt, "amber"})});
  fft::Schema forward{{{"c", "utf8"}}};
  fft::Schema backward{{{"c", "utf8"}}};
  auto one = fit_kernel("basen_encode", {{"columns", {"c"}}, {"base", 3}}, forward, {first, second});
  auto other = fit_kernel("basen_encode", {{"columns", {"c"}}, {"base", 3}}, backward, {second, first});
  REQUIRE(one->state() == other->state());
  REQUIRE(one->state().at("categories").at("c") == nlohmann::json({"amber", "blue", "delta"}));
  REQUIRE(one->state().at("has_null").at("c") == true);
  fft::Schema resolved{{{"c", "utf8"}}};
  one->resolve(resolved);
  // Four levels (the null is the last of them) in base 3: two digits.
  REQUIRE(one->outputs == std::vector<std::string>{"c_0", "c_1"});
  fft::Batch out = one->transform(make_batch({"c"}, {utf8_column({"amber", "blue", "delta", std::nullopt, "zz"})}));
  REQUIRE(word_at(out, 0) == std::vector<std::optional<int64_t>>{0, 1});
  REQUIRE(word_at(out, 1) == std::vector<std::optional<int64_t>>{0, 2});
  REQUIRE(word_at(out, 2) == std::vector<std::optional<int64_t>>{1, 0});
  REQUIRE(word_at(out, 3) == std::vector<std::optional<int64_t>>{1, 1});  // the fitted null ranks last
  REQUIRE(word_at(out, 4) == std::vector<std::optional<int64_t>>{0, 0});
}

TEST_CASE("gray_encode spells a rank so that neighbours differ in one digit") {
  fft::Schema schema{{{"c", "utf8"}}};
  auto kernel = fit_kernel("gray_encode", {{"columns", {"c"}}}, schema,
                           {make_batch({"c"}, {utf8_column({"e", "c", "a", "d", "b"})})});
  fft::Schema resolved{{{"c", "utf8"}}};
  kernel->resolve(resolved);
  REQUIRE(kernel->outputs == std::vector<std::string>{"c_0", "c_1", "c_2"});
  // Ranks 1..5 spell 1, 3, 2, 6, 7; every neighboring pair differs in
  // exactly one column.
  fft::Batch out = kernel->transform(make_batch({"c"}, {utf8_column({"a", "b", "c", "d", "e"})}));
  REQUIRE(word_at(out, 0) == std::vector<std::optional<int64_t>>{0, 0, 1});
  REQUIRE(word_at(out, 1) == std::vector<std::optional<int64_t>>{0, 1, 1});
  REQUIRE(word_at(out, 2) == std::vector<std::optional<int64_t>>{0, 1, 0});
  REQUIRE(word_at(out, 3) == std::vector<std::optional<int64_t>>{1, 1, 0});
  REQUIRE(word_at(out, 4) == std::vector<std::optional<int64_t>>{1, 1, 1});
}

TEST_CASE("the unknown and null options change the width and the words") {
  const fft::Batch batch = make_batch({"c"}, {utf8_column({"a", "b", "c"})});
  const fft::Batch input = make_batch({"c"}, {utf8_column({"a", "zz", std::nullopt})});

  // 'indicator' makes the unknown a real extra level, so it takes rank 4 and
  // the width is sized for one level more.
  fft::Schema indicator_schema{{{"c", "utf8"}}};
  auto indicator =
      fit_kernel("binary_encode", {{"columns", {"c"}}, {"handle_unknown", "indicator"}}, indicator_schema, {batch});
  fft::Batch out = indicator->transform(input);
  REQUIRE(out.names == std::vector<std::string>{"c_0", "c_1", "c_2"});
  REQUIRE(word_at(out, 1) == std::vector<std::optional<int64_t>>{1, 0, 0});
  REQUIRE(word_at(out, 2) == std::vector<std::optional<int64_t>>{0, 0, 0});  // a null never fitted: the free word

  // Under gray_encode the indicator keeps its plain binary word: upstream
  // rewrites only the fitted levels when it converts the mapping to Gray.
  fft::Schema gray_schema{{{"c", "utf8"}}};
  auto gray = fit_kernel("gray_encode", {{"columns", {"c"}}, {"handle_unknown", "indicator"}}, gray_schema, {batch});
  REQUIRE(word_at(gray->transform(input), 1) == std::vector<std::optional<int64_t>>{1, 0, 0});

  // 'return_nan' fills the whole row with nulls, for an unknown and, under
  // handle_missing, for a null.
  fft::Schema nan_schema{{{"c", "utf8"}}};
  auto as_null = fit_kernel("binary_encode",
                            {{"columns", {"c"}}, {"handle_unknown", "return_nan"}, {"handle_missing", "return_nan"}},
                            nan_schema, {batch});
  fft::Batch nulls = as_null->transform(input);
  REQUIRE(word_at(nulls, 1) == std::vector<std::optional<int64_t>>{{}, {}, {}});
  REQUIRE(word_at(nulls, 2) == std::vector<std::optional<int64_t>>{{}, {}, {}});

  fft::Schema strict_schema{{{"c", "utf8"}}};
  auto strict = fit_kernel("binary_encode", {{"columns", {"c"}}, {"handle_unknown", "error"}}, strict_schema, {batch});
  REQUIRE_THROWS_WITH(strict->transform(input), ContainsSubstring("not seen at fit"));

  // base 1 is upstream's one-hot special case: a single 1 at position r of
  // levels + 1 columns, so the leading column is always zero.
  fft::Schema one_schema{{{"c", "utf8"}}};
  auto one_hot = fit_kernel("basen_encode", {{"columns", {"c"}}, {"base", 1}}, one_schema, {batch});
  fft::Batch spread = one_hot->transform(make_batch({"c"}, {utf8_column({"a", "b", "c", "zz"})}));
  REQUIRE(spread.names == std::vector<std::string>{"c_0", "c_1", "c_2", "c_3"});
  REQUIRE(word_at(spread, 0) == std::vector<std::optional<int64_t>>{0, 1, 0, 0});
  REQUIRE(word_at(spread, 2) == std::vector<std::optional<int64_t>>{0, 0, 0, 1});
  REQUIRE(word_at(spread, 3) == std::vector<std::optional<int64_t>>{0, 0, 0, 0});
}

// REGRESSION: the batch path refuses a passthrough column named like a
// generated digit at resolve; the record path used to keep both and let the
// digit win, so the same record was a hard error through Parquet and a silent
// substitution through transform_dict.
TEST_CASE("the positional encoders refuse a record key that collides with a generated column") {
  fft::Schema schema{{{"c", "utf8"}}};
  auto kernel =
      fit_kernel("binary_encode", {{"columns", {"c"}}}, schema, {make_batch({"c"}, {utf8_column({"a", "b"})})});
  fft::Schema resolved{{{"c", "utf8"}, {"c_0", "i64"}}};
  REQUIRE_THROWS_WITH(kernel->resolve(resolved), ContainsSubstring("output column 'c_0' already exists in input"));
  REQUIRE_THROWS_WITH(kernel->transform_row({{"c", std::string("a")}, {"c_0", int64_t{9}}}),
                      ContainsSubstring("key 'c_0' already present in record"));
}

// REGRESSION: 'error' is the strict option, so it refuses a null wherever one
// turns up. It used to encode a null at transform as the all-zero word.
TEST_CASE("the positional encoders refuse a null at transform under handle_missing error") {
  for (const char* id : {"basen_encode", "binary_encode", "gray_encode"}) {
    fft::Schema schema{{{"c", "utf8"}}};
    auto kernel = fit_kernel(id, {{"columns", {"c"}}, {"handle_missing", "error"}}, schema,
                             {make_batch({"c"}, {utf8_column({"a", "b"})})});
    REQUIRE_THROWS_WITH(kernel->transform(make_batch({"c"}, {utf8_column({"a", std::nullopt})})),
                        ContainsSubstring("column 'c' has a null and handle_missing is 'error'"));
    REQUIRE_THROWS_WITH(kernel->transform_row({{"c", fft::RowValue(std::monostate{})}}),
                        ContainsSubstring("column 'c' has a null and handle_missing is 'error'"));
  }
}

// REGRESSION: base 1 spends one column per level, so the learned width follows
// the cardinality and a column that grows in production would allocate one
// row-length vector per category per batch. The fit refuses instead.
TEST_CASE("basen_encode caps how many columns it will generate") {
  std::vector<std::optional<std::string>> many;
  many.reserve(4096);
  for (int i = 0; i < 4096; ++i) many.emplace_back(std::to_string(i));
  fft::Schema schema{{{"c", "utf8"}}};
  REQUIRE_THROWS_WITH(
      fit_kernel("basen_encode", {{"columns", {"c"}}, {"base", 1}}, schema, {make_batch({"c"}, {utf8_column(many)})}),
      ContainsSubstring("more than the 4096 allowed"));
}

// REGRESSION: the encoded columns are erased before the collision check, as
// hash_encode does, so a generated name may reuse an encoded column's. The
// check used to run first, so encoding 'a' beside 'a_0' was a fit-time error
// with no way out but renaming a column.
TEST_CASE("the positional encoders may generate a name an encoded column had") {
  fft::Schema schema{{{"a", "utf8"}, {"a_0", "utf8"}}};
  auto kernel = fit_kernel("binary_encode", {{"columns", {"a", "a_0"}}}, schema,
                           {make_batch({"a", "a_0"}, {utf8_column({"x", "y"}), utf8_column({"p", "q"})})});
  fft::Schema resolved{{{"a", "utf8"}, {"a_0", "utf8"}}};
  REQUIRE_NOTHROW(kernel->resolve(resolved));
  REQUIRE(kernel->outputs == std::vector<std::string>{"a_0", "a_1", "a_0_0", "a_0_1"});
}

TEST_CASE("the positional encoders' load_state rejects a malformed table") {
  auto fresh = [] {
    auto kernel = fft::make_kernel("gray_encode", {{"columns", {"c"}}});
    kernel->stage = "enc";
    return kernel;
  };
  const nlohmann::json good{{"columns", {"c"}}, {"categories", {{"c", {"a", "b"}}}}, {"has_null", {{"c", false}}}};
  REQUIRE_NOTHROW(fresh()->load_state(good));
  auto broken = [&](auto mutate, const std::string& message) {
    nlohmann::json state = good;
    mutate(state);
    REQUIRE_THROWS_WITH(fresh()->load_state(state), ContainsSubstring(message));
  };
  broken([](nlohmann::json& state) { state.erase("categories"); }, "must hold a 'categories' object");
  broken([](nlohmann::json& state) { state["categories"]["c"] = {"b", "a"}; }, "must be sorted and unique");
  broken([](nlohmann::json& state) { state["categories"]["c"] = {2, 3}; }, "must hold strings");
  broken([](nlohmann::json& state) { state["categories"]["c"] = nlohmann::json::array(); }, "is empty");
  broken([](nlohmann::json& state) { state["has_null"]["c"] = 1; }, "must be true or false");
}
