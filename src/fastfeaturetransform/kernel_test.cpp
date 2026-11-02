// Explicit C++ tests for the kernel logic: the catalog, select, cast,
// label_encode, and group_agg, including the accumulator-merge (monoid)
// contract and the value-for-value agreement of the batch and record paths
// that the whole design rests on.

#include "fastfeaturetransform/kernel.hpp"

#include <arrow/array.h>

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <limits>

#include "fastfeaturetransform/kernels/dictionary_test_support.hpp"
#include "fastfeaturetransform/spill.hpp"
#include "fastfeaturetransform/test_support.hpp"

namespace fft = fastfeaturetransform;
using Catch::Matchers::ContainsSubstring;
using fft::testing::bool_column;
using fft::testing::f64_column;
using fft::testing::large_utf8_column;
using fft::testing::make_batch;
using fft::testing::numeric_column;
using fft::testing::utf8_column;

namespace {

std::pair<std::shared_ptr<arrow::Array>, std::shared_ptr<arrow::Array>> oversized_dictionary_slice(bool large) {
  std::vector<std::optional<std::string>> entries(4096, "unused");
  entries[0] = std::string("\xff");  // outside the slice: unused invalid entries must remain ignored
  entries[1] = "alpha";
  entries[3] = std::nullopt;
  entries[7] = "";
  entries[2048] = "β";
  entries[3071] = "alpha";  // distinct entries may contain the same category
  entries[4095] = "unknown";
  auto dictionary = large ? large_utf8_column(entries) : utf8_column(entries);
  auto indices = numeric_column<arrow::Int16Type>({0, 3071, 7, 2048, 3, std::nullopt, 1, 2048, 3071, 0});
  auto encoded =
      *arrow::DictionaryArray::FromArrays(arrow::dictionary(indices->type(), dictionary->type()), indices, dictionary);
  const std::vector<std::optional<std::string>> values{"alpha",      "",      "β", std::nullopt,
                                                       std::nullopt, "alpha", "β", "alpha"};
  return {encoded->Slice(1, 8), large ? large_utf8_column(values) : utf8_column(values)};
}

}  // namespace

TEST_CASE("make_kernel rejects unknown ids loudly") {
  REQUIRE_THROWS_AS(fft::make_kernel("flux_capacitor", {}), fft::Error);
}

TEST_CASE("select projects and reorders columns; row and batch agree") {
  auto kernel = fft::make_kernel("select", {{"columns", {"b", "a"}}});
  kernel->stage = "sel";
  fft::Schema schema{{{"a", "utf8"}, {"b", "f64"}, {"c", "f64"}}};
  kernel->resolve(schema);
  REQUIRE(schema.cols == fft::Schema{{{"b", "f64"}, {"a", "utf8"}}}.cols);

  fft::Batch out =
      kernel->transform(make_batch({"a", "b", "c"}, {utf8_column({"x"}), f64_column({1.5}), f64_column({9.0})}));
  REQUIRE(out.names == std::vector<std::string>{"b", "a"});

  fft::Row row = kernel->transform_row({{"a", std::string("x")}, {"b", 1.5}, {"c", 9.0}});
  REQUIRE(row.size() == 2);
  REQUIRE(row[0].first == "b");
  REQUIRE(std::get<double>(row[0].second) == 1.5);
}

TEST_CASE("select rejects missing columns at resolve and bad params at construction") {
  fft::Schema schema{{{"a", "utf8"}}};
  auto missing = fft::make_kernel("select", {{"columns", {"nope"}}});
  REQUIRE_THROWS_AS(missing->resolve(schema), fft::Error);
  // Extra parentheses keep the braces' commas out of the macro's argument list.
  REQUIRE_THROWS_WITH((fft::make_kernel("select", {{"columns", {"a", "a"}}})), ContainsSubstring("more than once"));
  REQUIRE_THROWS_WITH((fft::make_kernel("select", {{"columns", nlohmann::json::array()}})),
                      ContainsSubstring("non-empty list"));
  REQUIRE_THROWS_WITH((fft::make_kernel("select", {{"columns", "a"}})), ContainsSubstring("non-empty list"));
  REQUIRE_THROWS_WITH(fft::make_kernel("select", nlohmann::json::object()), ContainsSubstring("non-empty list"));
}

TEST_CASE("label_encode validates its params") {
  REQUIRE_THROWS_WITH((fft::make_kernel("label_encode", {{"columns", {"a"}}, {"min_count", 0}})),
                      ContainsSubstring("'min_count' must be an integer from 1 to 2^63 - 1"));
  REQUIRE_THROWS_WITH((fft::make_kernel("label_encode", {{"columns", {"a"}}, {"min_count", 1.5}})),
                      ContainsSubstring("'min_count' must be an integer from 1 to 2^63 - 1"));
  REQUIRE_THROWS_WITH((fft::make_kernel("label_encode", {{"columns", nlohmann::json::array()}})),
                      ContainsSubstring("non-empty list"));
  REQUIRE_NOTHROW((fft::make_kernel("label_encode", {{"columns", nullptr}})));
}

TEST_CASE("label_encode load_state validates the label maps and keeps explicit column order") {
  auto good = fft::make_kernel("label_encode", {{"columns", {"b", "a"}}, {"min_count", 1}});
  good->stage = "enc";
  good->load_state({{"columns", {"b", "a"}}, {"label_maps", {{"a", {{"x", 1}}}, {"b", {{"y", 2}, {"z", 1}}}}}});
  REQUIRE(good->params().at("columns") == nlohmann::json({"b", "a"}));
  REQUIRE(good->inputs == std::vector<std::string>{"b", "a"});

  auto bad = fft::make_kernel("label_encode", {{"columns", {"a"}}, {"min_count", 1}});
  bad->stage = "enc";
  auto load = [&](nlohmann::json state) {
    if (!state.empty() && !state.contains("columns")) state["columns"] = {"a"};
    bad->load_state(state);
  };
  REQUIRE_THROWS_WITH((load({{"label_maps", {{"a", {{"x", 0}}}}}})), ContainsSubstring("exactly 1..1"));
  REQUIRE_THROWS_WITH((load({{"label_maps", {{"a", {{"x", 1}, {"y", 500}}}}}})), ContainsSubstring("exactly 1..2"));
  REQUIRE_THROWS_WITH((load({{"label_maps", {{"a", {{"x", 1}, {"y", 1}}}}}})), ContainsSubstring("no duplicates"));
  REQUIRE_THROWS_WITH((load({{"label_maps", {{"a", {{"x", 1.5}}}}}})), ContainsSubstring("exactly 1..1"));
  REQUIRE_THROWS_WITH((load({{"label_maps", {{"other", {{"x", 1}}}}}})),
                      ContainsSubstring("label_maps has no entry for column 'a'"));
  REQUIRE_THROWS_WITH((load({{"label_maps", {{"a", 5}}}})), ContainsSubstring("must be an object"));
  REQUIRE_THROWS_WITH((load({{"label_maps", {{"a", {{"x", 1}}}}}, {"vocab_sizes", {{"a", 7}}}})),
                      ContainsSubstring("vocab_sizes disagrees"));
  REQUIRE_THROWS_WITH(load(nlohmann::json::object()), ContainsSubstring("label_maps"));
  // REGRESSION: the state's column order is the authority. Params that name a
  // different list, or a different order, make a manifest that describes two
  // different steps rather than one to reconcile silently.
  REQUIRE_THROWS_WITH((bad->load_state({{"columns", {"b"}}, {"label_maps", {{"b", {{"x", 1}}}}}})),
                      ContainsSubstring("state 'columns' does not match the 'columns' in params"));
  REQUIRE_THROWS_WITH(
      (good->load_state({{"columns", {"a", "b"}}, {"label_maps", {{"a", {{"x", 1}}}, {"b", {{"y", 1}}}}}})),
      ContainsSubstring("state 'columns' does not match the 'columns' in params"));
  REQUIRE_THROWS_WITH((bad->load_state({{"label_maps", {{"a", {{"x", 1}}}}}})),
                      ContainsSubstring("state must hold a 'columns' list"));
}

TEST_CASE("label_encode rejects categories that are not valid UTF-8 at fit") {
  auto kernel = fft::make_kernel("label_encode", {{"columns", {"c"}}});
  kernel->stage = "enc";
  fft::Schema schema{{{"c", "utf8"}}};
  kernel->resolve(schema);
  auto state = kernel->make_state();
  kernel->consume(*state, make_batch({"c"}, {utf8_column({std::string("\xff\xfe"), "ok"})}));
  std::vector<std::unique_ptr<fft::AccState>> states;
  states.push_back(std::move(state));
  REQUIRE_THROWS_WITH(kernel->finalize(std::move(states)), ContainsSubstring("not valid UTF-8"));
}

TEST_CASE("a null dictionary entry encodes like a null value") {
  auto kernel = fft::make_kernel("label_encode", {{"columns", {"c"}}});
  kernel->stage = "enc";
  fft::Schema schema{{{"c", "utf8"}}};
  kernel->resolve(schema);
  // dictionary = ["", null]; rows point at the empty string, the null entry, and are null themselves
  auto dictionary = utf8_column({std::string(""), std::nullopt});
  arrow::Int32Builder index_builder;
  (void)index_builder.Append(0);
  (void)index_builder.Append(1);
  (void)index_builder.AppendNull();
  std::shared_ptr<arrow::Array> indices;
  (void)index_builder.Finish(&indices);
  auto dict_array =
      *arrow::DictionaryArray::FromArrays(arrow::dictionary(arrow::int32(), arrow::utf8()), indices, dictionary);

  auto state = kernel->make_state();
  kernel->consume(*state, make_batch({"c"}, {dict_array}));
  std::vector<std::unique_ptr<fft::AccState>> states;
  states.push_back(std::move(state));
  kernel->finalize(std::move(states));
  REQUIRE(kernel->state().at("label_maps").at("c") == nlohmann::json({{"", 1}}));

  fft::Batch encoded = kernel->transform(make_batch({"c"}, {dict_array}));
  const auto& values = static_cast<const arrow::Int64Array&>(*encoded.cols[0]);
  REQUIRE(values.Value(0) == 1);
  REQUIRE(values.Value(1) == 0);
  REQUIRE(values.Value(2) == 0);
}

TEST_CASE("label_encode merges worker states into byte-sorted 1-based maps") {
  auto kernel = fft::make_kernel("label_encode", {{"columns", {"device"}}, {"min_count", 2}});
  kernel->stage = "enc";
  fft::Schema schema{{{"device", "utf8"}}};
  kernel->resolve(schema);
  REQUIRE(*schema.find("device") == "i64");

  // Two worker states, each holding every category once, below min_count 2:
  // only the merged counts may decide inclusion. This is the monoid contract.
  auto state_a = kernel->make_state();
  auto state_b = kernel->make_state();
  kernel->consume(*state_a, make_batch({"device"}, {utf8_column({"mob", "desk", "tab"})}));
  kernel->consume(*state_b, make_batch({"device"}, {utf8_column({"mob", "desk", "tab", std::nullopt})}));
  std::vector<std::unique_ptr<fft::AccState>> states;
  states.push_back(std::move(state_a));
  states.push_back(std::move(state_b));
  kernel->finalize(std::move(states));

  nlohmann::json fitted = kernel->state();
  REQUIRE(fitted.at("label_maps").at("device") == nlohmann::json({{"desk", 1}, {"mob", 2}, {"tab", 3}}));
  REQUIRE(fitted.at("vocab_sizes").at("device") == 4);
}

TEST_CASE("label_encode batch and row paths agree value-for-value") {
  auto kernel = fft::make_kernel("label_encode", {{"columns", {"device"}}});
  kernel->stage = "enc";
  fft::Schema schema{{{"device", "utf8"}}};
  kernel->resolve(schema);
  auto state = kernel->make_state();
  kernel->consume(*state, make_batch({"device"}, {utf8_column({"mob", "desk"})}));
  std::vector<std::unique_ptr<fft::AccState>> states;
  states.push_back(std::move(state));
  kernel->finalize(std::move(states));

  std::vector<std::optional<std::string>> inputs{"mob", "desk", "spaceship", std::nullopt};
  fft::Batch batch_out = kernel->transform(make_batch({"device"}, {utf8_column(inputs)}));
  const auto& encoded = static_cast<const arrow::Int64Array&>(*batch_out.cols[0]);

  for (std::size_t i = 0; i < inputs.size(); ++i) {
    fft::RowValue value = inputs[i].has_value() ? fft::RowValue(*inputs[i]) : fft::RowValue(std::monostate{});
    fft::Row row_out = kernel->transform_row({{"device", value}});
    REQUIRE(encoded.Value(static_cast<int64_t>(i)) == std::get<int64_t>(row_out[0].second));
  }
}

TEST_CASE("label_encode fitted-state round-trip preserves behavior") {
  auto original = fft::make_kernel("label_encode", {{"columns", {"device"}}});
  original->stage = "enc";
  fft::Schema schema{{{"device", "utf8"}}};
  original->resolve(schema);
  auto state = original->make_state();
  original->consume(*state, make_batch({"device"}, {utf8_column({"mob", "desk"})}));
  std::vector<std::unique_ptr<fft::AccState>> states;
  states.push_back(std::move(state));
  original->finalize(std::move(states));

  auto reloaded = fft::make_kernel("label_encode", {{"columns", nullptr}, {"min_count", 1}});
  reloaded->stage = "enc";
  reloaded->load_state(original->state());
  REQUIRE(reloaded->fitted);
  fft::Row out = reloaded->transform_row({{"device", std::string("mob")}});
  REQUIRE(std::get<int64_t>(out[0].second) == 2);
}

TEST_CASE("label_encode dictionary fast path agrees with the dense path") {
  auto kernel = fft::make_kernel("label_encode", {{"columns", {"device"}}});
  kernel->stage = "enc";
  fft::Schema schema{{{"device", "utf8"}}};
  kernel->resolve(schema);

  // dictionary = ["mob", "desk"], rows = [mob, desk, mob, null, desk]
  auto dictionary = utf8_column({"mob", "desk"});
  arrow::Int32Builder index_builder;
  for (int value : {0, 1, 0}) (void)index_builder.Append(value);
  (void)index_builder.AppendNull();
  (void)index_builder.Append(1);
  std::shared_ptr<arrow::Array> indices;
  (void)index_builder.Finish(&indices);
  auto dict_array =
      *arrow::DictionaryArray::FromArrays(arrow::dictionary(arrow::int32(), arrow::utf8()), indices, dictionary);

  auto state = kernel->make_state();
  kernel->consume(*state, make_batch({"device"}, {dict_array}));
  std::vector<std::unique_ptr<fft::AccState>> states;
  states.push_back(std::move(state));
  kernel->finalize(std::move(states));
  REQUIRE(kernel->state().at("label_maps").at("device") == nlohmann::json({{"desk", 1}, {"mob", 2}}));

  fft::Batch encoded = kernel->transform(make_batch({"device"}, {dict_array}));
  const auto& values = static_cast<const arrow::Int64Array&>(*encoded.cols[0]);
  std::vector<int64_t> expect{2, 1, 2, 0, 1};
  for (std::size_t i = 0; i < expect.size(); ++i) REQUIRE(values.Value(static_cast<int64_t>(i)) == expect[i]);
}

TEST_CASE("label_encode fits short slices without a dictionary-sized cache") {
  for (bool large : {false, true}) {
    const auto [dictionary, plain] = oversized_dictionary_slice(large);
    auto fit = [](const std::shared_ptr<arrow::Array>& array) {
      auto kernel = fft::make_kernel("label_encode", {{"columns", {"c"}}, {"min_count", 2}});
      kernel->stage = "enc";
      fft::Schema schema{{{"c", "utf8"}}};
      kernel->resolve(schema);
      auto state = kernel->make_state();
      kernel->consume(*state, make_batch({"c"}, {array}));
      std::vector<std::unique_ptr<fft::AccState>> states;
      states.push_back(std::move(state));
      kernel->finalize(std::move(states));
      return kernel;
    };
    const auto reference = fit(plain);
    const auto actual = fit(dictionary);
    REQUIRE(actual->state() == reference->state());
    REQUIRE(actual->state().at("label_maps").at("c") == nlohmann::json({{"alpha", 1}, {"β", 2}}));
    REQUIRE(actual->transform(make_batch({"c"}, {dictionary}))
                .cols[0]
                ->Equals(reference->transform(make_batch({"c"}, {plain})).cols[0]));
  }
}

TEST_CASE("label_encode handles large_string columns, with nulls and empty strings, and rejects non-utf8") {
  auto kernel = fft::make_kernel("label_encode", {{"columns", {"city"}}});
  kernel->stage = "enc";
  fft::Schema schema{{{"city", "utf8"}}};
  kernel->resolve(schema);
  std::vector<std::optional<std::string>> values{"nyc", std::nullopt, "", "sf", "nyc", std::nullopt};
  auto state = kernel->make_state();
  kernel->consume(*state, make_batch({"city"}, {large_utf8_column(values)}));
  std::vector<std::unique_ptr<fft::AccState>> states;
  states.push_back(std::move(state));
  kernel->finalize(std::move(states));
  REQUIRE(kernel->state().at("label_maps").at("city") == nlohmann::json({{"", 1}, {"nyc", 2}, {"sf", 3}}));

  fft::Batch encoded = kernel->transform(make_batch({"city"}, {large_utf8_column(values)}));
  const auto& out = static_cast<const arrow::Int64Array&>(*encoded.cols[0]);
  std::vector<int64_t> expect{2, 0, 1, 3, 2, 0};
  for (std::size_t i = 0; i < expect.size(); ++i) {
    REQUIRE(out.Value(static_cast<int64_t>(i)) == expect[i]);
    fft::RowValue value = values[i].has_value() ? fft::RowValue(*values[i]) : fft::RowValue(std::monostate{});
    REQUIRE(std::get<int64_t>(kernel->transform_row({{"city", value}})[0].second) == expect[i]);
  }

  REQUIRE_THROWS_AS(kernel->transform(make_batch({"city"}, {f64_column({1.0})})), fft::Error);
}

namespace {

std::unique_ptr<fft::Kernel> cast_kernel(const nlohmann::json& types) {
  auto kernel = fft::make_kernel("cast", {{"types", types}});
  kernel->stage = "types";
  return kernel;
}

}  // namespace

TEST_CASE("cast converts across the closed table and the batch path agrees with the record path") {
  auto kernel = cast_kernel(
      {{"id", "utf8"}, {"price", "f32"}, {"count", "i16"}, {"flag", "i64"}, {"text_id", "i64"}, {"text_price", "f64"}});
  fft::Schema schema{{{"id", "i32"},
                      {"price", "f64"},
                      {"count", "i64"},
                      {"flag", "bool"},
                      {"text_id", "utf8"},
                      {"text_price", "utf8"},
                      {"other", "f64"}}};
  kernel->resolve(schema);
  REQUIRE(schema.cols == fft::Schema{{{"id", "utf8"},
                                      {"price", "f32"},
                                      {"count", "i16"},
                                      {"flag", "i64"},
                                      {"text_id", "i64"},
                                      {"text_price", "f64"},
                                      {"other", "f64"}}}
                             .cols);

  fft::Batch out = kernel->transform(
      make_batch({"id", "price", "count", "flag", "text_id", "text_price"},
                 {numeric_column<arrow::Int32Type>({7, -3, std::nullopt}),
                  numeric_column<arrow::DoubleType>({0.1, 2.5, std::nullopt}),
                  numeric_column<arrow::Int64Type>({32767, -5, std::nullopt}), bool_column({true, false, std::nullopt}),
                  utf8_column({"12", "+7", std::nullopt}), utf8_column({"1.5", "-2e3", std::nullopt})}));
  REQUIRE(out.cols[0]->type_id() == arrow::Type::STRING);
  REQUIRE(out.cols[1]->type_id() == arrow::Type::FLOAT);
  REQUIRE(out.cols[2]->type_id() == arrow::Type::INT16);
  REQUIRE(out.cols[3]->type_id() == arrow::Type::INT64);
  REQUIRE(out.cols[4]->type_id() == arrow::Type::INT64);
  REQUIRE(out.cols[5]->type_id() == arrow::Type::DOUBLE);
  const auto& id = static_cast<const arrow::StringArray&>(*out.cols[0]);
  const auto& price = static_cast<const arrow::FloatArray&>(*out.cols[1]);
  const auto& count = static_cast<const arrow::Int16Array&>(*out.cols[2]);
  const auto& flag = static_cast<const arrow::Int64Array&>(*out.cols[3]);
  const auto& text_id = static_cast<const arrow::Int64Array&>(*out.cols[4]);
  const auto& text_price = static_cast<const arrow::DoubleArray&>(*out.cols[5]);
  REQUIRE(id.GetString(0) == "7");
  REQUIRE(id.GetString(1) == "-3");
  REQUIRE(price.Value(0) == static_cast<float>(0.1));
  REQUIRE(count.Value(0) == 32767);
  REQUIRE(count.Value(1) == -5);
  REQUIRE(flag.Value(0) == 1);
  REQUIRE(flag.Value(1) == 0);
  REQUIRE(text_id.Value(0) == 12);
  REQUIRE(text_id.Value(1) == 7);
  REQUIRE(text_price.Value(0) == 1.5);
  REQUIRE(text_price.Value(1) == -2000.0);
  for (const auto& column : out.cols) REQUIRE(column->IsNull(2));

  fft::Row row = kernel->transform_row({{"id", int64_t{7}},
                                        {"price", 0.1},
                                        {"count", int64_t{32767}},
                                        {"flag", true},
                                        {"text_id", std::string("12")},
                                        {"text_price", std::string("1.5")},
                                        {"other", 9.0}});
  REQUIRE(std::get<std::string>(row[0].second) == "7");
  REQUIRE(std::get<double>(row[1].second) == static_cast<double>(static_cast<float>(0.1)));
  REQUIRE(std::get<int64_t>(row[2].second) == 32767);
  REQUIRE(std::get<int64_t>(row[3].second) == 1);
  REQUIRE(std::get<int64_t>(row[4].second) == 12);
  REQUIRE(std::get<double>(row[5].second) == 1.5);
  REQUIRE(std::get<double>(row[6].second) == 9.0);  // untouched
  fft::Row nulls = kernel->transform_row({{"id", std::monostate{}},
                                          {"price", std::monostate{}},
                                          {"count", std::monostate{}},
                                          {"flag", std::monostate{}},
                                          {"text_id", std::monostate{}},
                                          {"text_price", std::monostate{}}});
  for (const auto& [name, value] : nulls) REQUIRE(std::holds_alternative<std::monostate>(value));
}

TEST_CASE("cast refuses conversions outside the table at resolve") {
  const fft::Schema schema{{{"x", "f64"}, {"b", "bool"}, {"s", "utf8"}}};
  auto resolve = [&](const nlohmann::json& types) {
    fft::Schema fresh = schema;
    cast_kernel(types)->resolve(fresh);
  };
  REQUIRE_THROWS_WITH((resolve({{"x", "utf8"}})),
                      ContainsSubstring("cannot cast a float to utf8 (a float has no single spelling"));
  REQUIRE_THROWS_WITH((resolve({{"b", "utf8"}})), ContainsSubstring("cannot cast a bool to utf8"));
  REQUIRE_THROWS_WITH((resolve({{"b", "f64"}})), ContainsSubstring("cannot cast a bool to f64"));
  REQUIRE_THROWS_WITH((resolve({{"nope", "i64"}})), ContainsSubstring("'nope' not found"));
  REQUIRE_THROWS_WITH((cast_kernel({{"s", "bool"}})), ContainsSubstring("unsupported cast target 'bool'"));
  REQUIRE_THROWS_WITH((cast_kernel({{"s", "int"}})), ContainsSubstring("unsupported cast target 'int'"));
  REQUIRE_THROWS_WITH((fft::make_kernel("cast", {{"types", nlohmann::json::object()}})),
                      ContainsSubstring("non-empty mapping"));
  REQUIRE_THROWS_WITH((fft::make_kernel("cast", {{"types", {{"s", 5}}}})), ContainsSubstring("type tag string"));
  REQUIRE_NOTHROW((resolve({{"s", "utf8"}})));  // a same-type cast is a no-op
}

TEST_CASE("cast refuses values that do not fit, on both paths") {
  fft::Schema schema{{{"n", "i64"}, {"x", "f64"}, {"s", "utf8"}}};
  auto expect_both = [&](const nlohmann::json& types, const std::string& column,
                         const std::shared_ptr<arrow::Array>& batch_column, fft::RowValue record_value,
                         const std::string& message) {
    auto kernel = cast_kernel(types);
    fft::Schema fresh = schema;  // resolve() rewrites the type it casts
    kernel->resolve(fresh);
    REQUIRE_THROWS_WITH(kernel->transform(make_batch({column}, {batch_column})), ContainsSubstring(message));
    REQUIRE_THROWS_WITH(kernel->transform_row({{column, record_value}}), ContainsSubstring(message));
  };
  expect_both({{"n", "i8"}}, "n", numeric_column<arrow::Int64Type>({300}), int64_t{300}, "does not fit in i8");
  expect_both({{"n", "u8"}}, "n", numeric_column<arrow::Int64Type>({-1}), int64_t{-1}, "does not fit in u8");
  expect_both({{"x", "i64"}}, "x", numeric_column<arrow::DoubleType>({1.5}), 1.5, "is not a whole number");
  expect_both({{"x", "i64"}}, "x", numeric_column<arrow::DoubleType>({1e19}), 1e19, "does not fit in a 64-bit integer");
  expect_both({{"n", "f64"}}, "n", numeric_column<arrow::Int64Type>({(int64_t{1} << 53) + 1}),
              int64_t{(int64_t{1} << 53) + 1},
              "is beyond 2^53 in magnitude, where f64 no longer holds every integer exactly");
  expect_both({{"n", "f32"}}, "n", numeric_column<arrow::Int64Type>({(int64_t{1} << 24) + 1}),
              int64_t{(int64_t{1} << 24) + 1},
              "is beyond 2^24 in magnitude, where f32 no longer holds every integer exactly");
  expect_both({{"x", "f32"}}, "x", numeric_column<arrow::DoubleType>({1e39}), 1e39, "does not fit in f32");
  expect_both({{"s", "i64"}}, "s", utf8_column({"12x"}), std::string("12x"), "is not an integer");
  expect_both({{"s", "i64"}}, "s", utf8_column({""}), std::string(""), "is not an integer");
  expect_both({{"s", "i64"}}, "s", utf8_column({"+-5"}), std::string("+-5"), "is not an integer");
  expect_both({{"s", "i64"}}, "s", utf8_column({"9223372036854775808"}), std::string("9223372036854775808"),
              "does not fit in a 64-bit integer");
  expect_both({{"s", "f64"}}, "s", utf8_column({"two"}), std::string("two"), "is not a number");
  expect_both({{"s", "f64"}}, "s", utf8_column({"inf"}), std::string("inf"), "is not a number");
  expect_both({{"s", "f64"}}, "s", utf8_column({"nan"}), std::string("nan"), "is not a number");
  expect_both({{"s", "f64"}}, "s", utf8_column({"1e400"}), std::string("1e400"), "is out of range for f64");

  // A record value of a type the table does not list for the target.
  auto to_text = cast_kernel({{"n", "utf8"}});
  fft::Schema text_schema = schema;
  to_text->resolve(text_schema);
  REQUIRE_THROWS_WITH(to_text->transform_row({{"n", 2.0}}), ContainsSubstring("cannot cast a float to utf8"));
  // An unsigned 64-bit value beyond int64 is an error rather than a wrap.
  auto widen = cast_kernel({{"n", "i64"}});
  fft::Schema widen_schema = schema;
  widen->resolve(widen_schema);
  REQUIRE_THROWS_WITH(widen->transform(make_batch({"n"}, {numeric_column<arrow::UInt64Type>({uint64_t{1} << 63})})),
                      ContainsSubstring("does not fit in a 64-bit integer"));
  REQUIRE_NOTHROW(widen->transform(make_batch({"n"}, {numeric_column<arrow::UInt64Type>({5})})));
  // ... but the same value keeps its type as u64 and becomes digits as utf8.
  fft::Schema unsigned_schema{{{"n", "u64"}}};
  auto keep = cast_kernel({{"n", "u64"}});
  keep->resolve(unsigned_schema);
  fft::Batch kept = keep->transform(make_batch({"n"}, {numeric_column<arrow::UInt64Type>({uint64_t{1} << 63})}));
  REQUIRE(static_cast<const arrow::UInt64Array&>(*kept.cols[0]).Value(0) == (uint64_t{1} << 63));
  fft::Schema digits_schema{{{"n", "u64"}}};
  auto digits = cast_kernel({{"n", "utf8"}});
  digits->resolve(digits_schema);
  fft::Batch text = digits->transform(make_batch({"n"}, {numeric_column<arrow::UInt64Type>({uint64_t{1} << 63})}));
  REQUIRE(static_cast<const arrow::StringArray&>(*text.cols[0]).GetString(0) == "9223372036854775808");

  // A dictionary whose values are not strings is not a string column.
  arrow::Int32Builder index_builder;
  (void)index_builder.Append(0);
  std::shared_ptr<arrow::Array> indices;
  (void)index_builder.Finish(&indices);
  auto int_dictionary = *arrow::DictionaryArray::FromArrays(arrow::dictionary(arrow::int32(), arrow::int64()), indices,
                                                            numeric_column<arrow::Int64Type>({9}));
  auto to_utf8 = cast_kernel({{"s", "utf8"}});
  fft::Schema utf8_schema = schema;
  to_utf8->resolve(utf8_schema);
  REQUIRE_THROWS_WITH(to_utf8->transform(make_batch({"s"}, {int_dictionary})), ContainsSubstring("cannot read"));
}

namespace {

const nlohmann::json kEveryAgg = nlohmann::json::array({"count", "sum", "mean", "min", "max", "median"});

// Fit a group_agg kernel with one worker state per batch.
std::unique_ptr<fft::Kernel> fit_group_agg(const nlohmann::json& params, fft::Schema schema,
                                           const std::vector<fft::Batch>& batches) {
  auto kernel = fft::make_kernel("group_agg", params);
  kernel->stage = "agg";
  kernel->resolve(schema);
  std::vector<std::unique_ptr<fft::AccState>> states;
  for (const fft::Batch& batch : batches) {
    states.push_back(kernel->make_state());
    kernel->consume(*states.back(), batch);
  }
  kernel->finalize(std::move(states));
  return kernel;
}

}  // namespace

TEST_CASE("group medians spill exact values under one shared worker budget") {
  std::vector<std::optional<std::string>> keys;
  std::vector<std::optional<double>> values;
  for (int i = 0; i < 4097; ++i) {
    keys.emplace_back(i % 97 == 0 ? std::nullopt : std::optional<std::string>{"key-" + std::to_string(i % 7)});
    values.emplace_back(i % 19 == 0 ? std::nullopt
                                    : std::optional<double>{static_cast<double>((i * 37) % 1021 - 510) / 16.0});
  }
  const fft::Batch all = make_batch({"key", "x"}, {utf8_column(keys), numeric_column<arrow::DoubleType>(values)});
  const fft::Schema schema{{{"key", "utf8"}, {"x", "f64"}}};
  const nlohmann::json params{{"key", "key"}, {"aggs", {{"x", {"median", "mean", "count"}}}}};
  const auto expected = fit_group_agg(params, schema, {all})->state();
  fft::testing::TempDir temporary;
  {
    auto memory = std::make_shared<fft::FitMemory>(fft::FitMemoryOptions{512, temporary.path.string()});
    fft::FitMemoryScope scope(memory);
    std::vector<fft::Batch> batches;
    for (int64_t offset = 0; offset < all.rows; offset += 331) {
      const int64_t count = std::min(int64_t{331}, all.rows - offset);
      batches.push_back(make_batch(all.names, {all.cols[0]->Slice(offset, count), all.cols[1]->Slice(offset, count)}));
    }
    std::reverse(batches.begin(), batches.end());
    const auto actual = fit_group_agg(params, schema, batches);
    REQUIRE(actual->state() == expected);
    REQUIRE(memory->spill_files_created() > 0);
    REQUIRE(memory->peak_bytes_in_memory() <= 512);
    REQUIRE(memory->bytes_in_memory() == 0);
    REQUIRE(memory->open_files() == 0);
  }
  REQUIRE(std::filesystem::is_empty(temporary.path));
}

TEST_CASE("group_agg validates its params") {
  REQUIRE_THROWS_WITH((fft::make_kernel("group_agg", {{"aggs", {{"n", {"sum"}}}}})),
                      ContainsSubstring("'key' must be a column name"));
  REQUIRE_THROWS_WITH((fft::make_kernel("group_agg", {{"key", "k"}})), ContainsSubstring("non-empty mapping"));
  REQUIRE_THROWS_WITH((fft::make_kernel("group_agg", {{"key", "k"}, {"aggs", nlohmann::json::object()}})),
                      ContainsSubstring("non-empty mapping"));
  REQUIRE_THROWS_WITH((fft::make_kernel("group_agg", {{"key", "k"}, {"aggs", {{"n", "sum"}}}})),
                      ContainsSubstring("non-empty list of names"));
  REQUIRE_THROWS_WITH((fft::make_kernel("group_agg", {{"key", "k"}, {"aggs", {{"n", {"avg"}}}}})),
                      ContainsSubstring("unknown aggregate 'avg'"));
  REQUIRE_THROWS_WITH((fft::make_kernel("group_agg", {{"key", "k"}, {"aggs", {{"n", {"sum", "sum"}}}}})),
                      ContainsSubstring("more than once"));
  auto kernel = fft::make_kernel("group_agg", {{"key", "k"}, {"aggs", {{"n", {"max", "sum"}}}}});
  REQUIRE(kernel->params() == nlohmann::json({{"key", "k"}, {"aggs", {{"n", {"max", "sum"}}}}}));
  REQUIRE(kernel->needs_fit());
}

TEST_CASE("group_agg resolves explicit output types and refuses what it cannot aggregate") {
  auto kernel = fft::make_kernel("group_agg",
                                 {{"key", "k"}, {"aggs", {{"n", kEveryAgg}, {"x", {"sum", "min"}}, {"s", {"count"}}}}});
  kernel->stage = "agg";
  REQUIRE(kernel->inputs == std::vector<std::string>{"k", "n", "s", "x"});
  REQUIRE(kernel->outputs == std::vector<std::string>{"n_count", "n_sum", "n_mean", "n_min", "n_max", "n_median",
                                                      "s_count", "x_sum", "x_min"});
  fft::Schema schema{{{"k", "utf8"}, {"n", "i32"}, {"x", "f64"}, {"s", "utf8"}}};
  kernel->resolve(schema);
  // The aggregated columns leave the schema; their aggregates follow the key.
  const fft::Schema expected{{{"k", "utf8"},
                              {"n_count", "i64"},
                              {"n_sum", "i64"},
                              {"n_mean", "f64"},
                              {"n_min", "i32"},
                              {"n_max", "i32"},
                              {"n_median", "f64"},
                              {"s_count", "i64"},
                              {"x_sum", "f64"},
                              {"x_min", "f64"}}};
  REQUIRE(schema.cols == expected.cols);

  auto refuse = [](const nlohmann::json& params, fft::Schema bad, const std::string& message) {
    auto refused = fft::make_kernel("group_agg", params);
    refused->stage = "agg";
    REQUIRE_THROWS_WITH(refused->resolve(bad), ContainsSubstring(message));
  };
  refuse({{"key", "x"}, {"aggs", {{"n", {"sum"}}}}}, fft::Schema{{{"x", "f64"}, {"n", "i32"}}},
         "expected utf8 or an integer column");
  refuse({{"key", "zz"}, {"aggs", {{"n", {"sum"}}}}}, fft::Schema{{{"k", "utf8"}, {"n", "i32"}}},
         "key column 'zz' not found");
  refuse({{"key", "k"}, {"aggs", {{"b", {"mean"}}}}}, fft::Schema{{{"k", "utf8"}, {"b", "bool"}}},
         "only count applies to a bool column");
  refuse({{"key", "k"}, {"aggs", {{"t", {"max"}}}}}, fft::Schema{{{"k", "utf8"}, {"t", "date32[day]"}}},
         "only count applies to a column of that type");
  // count needs only the null bitmap, so it applies to any column.
  auto counted = fft::make_kernel("group_agg", {{"key", "k"}, {"aggs", {{"t", {"count"}}}}});
  counted->stage = "agg";
  fft::Schema any{{{"k", "utf8"}, {"t", "date32[day]"}}};
  counted->resolve(any);
  REQUIRE(*any.find("t_count") == "i64");
  refuse({{"key", "k"}, {"aggs", {{"n", {"sum"}}}}}, fft::Schema{{{"k", "utf8"}, {"n", "i32"}, {"n_sum", "f64"}}},
         "already exists in input");
}

TEST_CASE("group_agg merges worker states into a key-sorted table, in any order, and both paths agree") {
  const nlohmann::json params{{"key", "k"}, {"aggs", {{"n", kEveryAgg}, {"x", {"sum", "min"}}}}};
  const fft::Schema schema{{{"k", "utf8"}, {"n", "i32"}, {"x", "f64"}}};
  const fft::Batch a =
      make_batch({"k", "n", "x"}, {utf8_column({"b", "a", "b"}), numeric_column<arrow::Int32Type>({1, 5, 3}),
                                   numeric_column<arrow::DoubleType>({0.5, std::nullopt, 1.5})});
  const fft::Batch b = make_batch(
      {"k", "n", "x"}, {utf8_column({"a", std::nullopt, "c"}), numeric_column<arrow::Int32Type>({2, 9, std::nullopt}),
                        numeric_column<arrow::DoubleType>({2.5, 1.0, 4.0})});
  auto kernel = fit_group_agg(params, schema, {a, b});
  const nlohmann::json state = kernel->state();
  REQUIRE(state.at("key_type") == "utf8");
  REQUIRE(state.at("keys") == nlohmann::json({"a", "b", "c"}));
  REQUIRE(state.at("column_types") == nlohmann::json({{"n", "i32"}, {"x", "f64"}}));
  const nlohmann::json& values = state.at("values");
  REQUIRE(values.at("n_count") == nlohmann::json({2, 2, 0}));
  REQUIRE(values.at("n_sum") == nlohmann::json({7, 4, 0}));
  REQUIRE(values.at("n_mean") == nlohmann::json({3.5, 2.0, 0.0}));
  REQUIRE(values.at("n_min") == nlohmann::json({2, 1, 0}));
  REQUIRE(values.at("n_max") == nlohmann::json({5, 3, 0}));
  REQUIRE(values.at("n_median") == nlohmann::json({3.5, 2.0, 0.0}));
  REQUIRE(values.at("x_sum") == nlohmann::json({2.5, 2.0, 4.0}));
  REQUIRE(values.at("x_min") == nlohmann::json({2.5, 0.5, 4.0}));
  // The monoid contract: workers merge in any order to the same table.
  REQUIRE(fit_group_agg(params, schema, {b, a})->state() == state);

  // Transform reads only the key; outputs follow the inputs and keep their types.
  const std::vector<std::optional<std::string>> keys{"c", "zz", std::nullopt, "a"};
  fft::Batch out = kernel->transform(make_batch({"k"}, {utf8_column(keys)}));
  REQUIRE(out.names ==
          std::vector<std::string>{"k", "n_count", "n_sum", "n_mean", "n_min", "n_max", "n_median", "x_sum", "x_min"});
  const auto& n_sum = static_cast<const arrow::Int64Array&>(*out.cols[static_cast<std::size_t>(out.find("n_sum"))]);
  const auto& n_min = static_cast<const arrow::Int32Array&>(*out.cols[static_cast<std::size_t>(out.find("n_min"))]);
  const auto& x_min = static_cast<const arrow::DoubleArray&>(*out.cols[static_cast<std::size_t>(out.find("x_min"))]);
  REQUIRE(n_min.type_id() == arrow::Type::INT32);
  REQUIRE(n_sum.Value(0) == 0);
  REQUIRE(n_sum.Value(3) == 7);
  REQUIRE(n_min.Value(3) == 2);
  REQUIRE(x_min.Value(0) == 4.0);
  REQUIRE(x_min.Value(1) == 0.0);
  for (std::size_t i = 0; i < keys.size(); ++i) {
    fft::RowValue key = keys[i].has_value() ? fft::RowValue(*keys[i]) : fft::RowValue(std::monostate{});
    fft::Row row = kernel->transform_row({{"k", key}});
    REQUIRE(row.size() == out.names.size());
    REQUIRE(std::get<int64_t>(*fft::row_find(row, "n_sum")) == n_sum.Value(static_cast<int64_t>(i)));
    REQUIRE(std::get<int64_t>(*fft::row_find(row, "n_min")) == n_min.Value(static_cast<int64_t>(i)));
    REQUIRE(std::get<double>(*fft::row_find(row, "x_min")) == x_min.Value(static_cast<int64_t>(i)));
  }
  REQUIRE_THROWS_WITH(kernel->transform_row({{"k", int64_t{1}}}), ContainsSubstring("must be a string or null"));
  REQUIRE_THROWS_WITH(kernel->transform_row({{"n", int64_t{1}}}), ContainsSubstring("key 'k' missing from record"));
  REQUIRE_THROWS_WITH(kernel->transform_row({{"k", std::string("a")}, {"n_sum", int64_t{1}}}),
                      ContainsSubstring("already present in record"));
}

TEST_CASE("group_agg takes integer and dictionary keys, and the fitted state round-trips") {
  const nlohmann::json params{{"key", "k"}, {"aggs", {{"n", {"sum"}}}}};
  auto ints = fit_group_agg(params, fft::Schema{{{"k", "i64"}, {"n", "i64"}}},
                            {make_batch({"k", "n"}, {numeric_column<arrow::Int64Type>({7, 12, 7}),
                                                     numeric_column<arrow::Int64Type>({1, 2, 5})})});
  REQUIRE(ints->state().at("key_type") == "i64");
  REQUIRE(ints->state().at("keys") == nlohmann::json({7, 12}));
  auto reloaded = fft::make_kernel("group_agg", params);
  reloaded->stage = "agg";
  reloaded->load_state(ints->state());
  REQUIRE(reloaded->fitted);
  REQUIRE(reloaded->state() == ints->state());
  fft::Row known = reloaded->transform_row({{"k", int64_t{7}}});
  REQUIRE(std::get<int64_t>(*fft::row_find(known, "n_sum")) == 6);
  fft::Row unknown = reloaded->transform_row({{"k", int64_t{8}}});
  REQUIRE(std::get<int64_t>(*fft::row_find(unknown, "n_sum")) == 0);
  REQUIRE_THROWS_WITH(reloaded->transform_row({{"k", std::string("7")}}),
                      ContainsSubstring("must be an integer or null"));
  fft::Schema strings{{{"k", "utf8"}, {"n", "i64"}}};
  REQUIRE_THROWS_WITH(reloaded->resolve(strings), ContainsSubstring("fitted table has integer keys"));
  fft::Schema narrower{{{"k", "i32"}, {"n", "i64"}}};
  REQUIRE_NOTHROW(reloaded->resolve(narrower));  // any integer width is an integer key
  // A fitted table reads only the key: the aggregated columns may be absent or retyped.
  REQUIRE(reloaded->inputs == std::vector<std::string>{"k"});
  fft::Schema key_only{{{"k", "i64"}}};
  reloaded->resolve(key_only);
  REQUIRE(key_only.cols == fft::Schema{{{"k", "i64"}, {"n_sum", "i64"}}}.cols);
  fft::Schema retyped{{{"k", "i64"}, {"n", "f64"}}};
  REQUIRE_NOTHROW(reloaded->resolve(retyped));

  // Dictionary-encoded keys: an entry no row uses never becomes a key.
  arrow::Int32Builder index_builder;
  (void)index_builder.AppendValues({1, 0, 1});
  std::shared_ptr<arrow::Array> indices;
  (void)index_builder.Finish(&indices);
  auto dictionary = *arrow::DictionaryArray::FromArrays(arrow::dictionary(arrow::int32(), arrow::utf8()), indices,
                                                        utf8_column({"b", "a", "unused"}));
  auto strings_kernel =
      fit_group_agg(params, fft::Schema{{{"k", "utf8"}, {"n", "i64"}}},
                    {make_batch({"k", "n"}, {dictionary, numeric_column<arrow::Int64Type>({1, 2, 5})})});
  REQUIRE(strings_kernel->state().at("keys") == nlohmann::json({"a", "b"}));
  REQUIRE(strings_kernel->state().at("values").at("n_sum") == nlohmann::json({6, 2}));
  fft::Batch looked_up = strings_kernel->transform(make_batch({"k"}, {dictionary}));
  const auto& sums = static_cast<const arrow::Int64Array&>(*looked_up.cols[1]);
  REQUIRE(sums.Value(0) == 6);
  REQUIRE(sums.Value(1) == 2);
}

TEST_CASE("group_agg fits and looks up short slices of much larger dictionaries") {
  const nlohmann::json params{{"key", "k"}, {"aggs", {{"x", kEveryAgg}}}};
  const fft::Schema schema{{{"k", "utf8"}, {"x", "f64"}}};
  const auto values = f64_column({1.0, 2.0, 3.0, std::numeric_limits<double>::quiet_NaN(),
                                  std::numeric_limits<double>::quiet_NaN(), 5.0, 7.0, 11.0});
  for (bool large : {false, true}) {
    const auto [dictionary, plain] = oversized_dictionary_slice(large);
    const auto reference = fit_group_agg(params, schema, {make_batch({"k", "x"}, {plain, values})});
    const auto actual = fit_group_agg(params, schema, {make_batch({"k", "x"}, {dictionary, values})});
    REQUIRE(actual->state() == reference->state());
    REQUIRE(actual->state().at("keys") == nlohmann::json({"", "alpha", "β"}));
    const auto& entries = static_cast<const arrow::DictionaryArray&>(*dictionary).dictionary();
    auto indices = numeric_column<arrow::Int32Type>({4095, 3071, 3, std::nullopt, 7, 2048});
    auto query =
        *arrow::DictionaryArray::FromArrays(arrow::dictionary(indices->type(), entries->type()), indices, entries);
    const fft::Batch expected = reference->transform(
        make_batch({"k"}, {utf8_column({"unknown", "alpha", std::nullopt, std::nullopt, "", "β"})}));
    const fft::Batch output = actual->transform(make_batch({"k"}, {query}));
    REQUIRE(output.names == expected.names);
    REQUIRE(output.cols[0] == query);
    for (std::size_t column = 1; column < output.cols.size(); ++column) {
      REQUIRE(output.cols[column]->Equals(expected.cols[column]));
    }
  }
}

TEST_CASE("oversized dictionary fallbacks retain type and used-key UTF-8 errors") {
  const auto dictionary = oversized_dictionary_slice(false).first;
  const auto& entries = static_cast<const arrow::DictionaryArray&>(*dictionary).dictionary();
  auto indices = numeric_column<arrow::Int32Type>({0, 0});
  auto invalid_utf8 =
      *arrow::DictionaryArray::FromArrays(arrow::dictionary(indices->type(), entries->type()), indices, entries);
  auto numbers = numeric_column<arrow::Int64Type>(std::vector<std::optional<int64_t>>(4096, 1));
  auto nonstring =
      *arrow::DictionaryArray::FromArrays(arrow::dictionary(indices->type(), numbers->type()), indices, numbers);
  for (const std::string id : {"label_encode", "group_agg"}) {
    const nlohmann::json params = id == "label_encode" ? nlohmann::json{{"columns", {"k"}}}
                                                       : nlohmann::json{{"key", "k"}, {"aggs", {{"x", {"count"}}}}};
    auto kernel = fft::make_kernel(id, params);
    kernel->stage = "stage";
    fft::Schema schema{{{"k", "utf8"}, {"x", "i64"}}};
    kernel->resolve(schema);
    auto state = kernel->make_state();
    const auto values = numeric_column<arrow::Int64Type>({1, 2});
    REQUIRE_THROWS_WITH(kernel->consume(*state, make_batch({"k", "x"}, {nonstring, values})),
                        ContainsSubstring("dictionary is int64, expected utf8"));
    kernel->consume(*state, make_batch({"k", "x"}, {invalid_utf8, values}));
    std::vector<std::unique_ptr<fft::AccState>> states;
    states.push_back(std::move(state));
    REQUIRE_THROWS_WITH(kernel->finalize(std::move(states)), ContainsSubstring("not valid UTF-8"));
  }
}

namespace {
std::unique_ptr<fft::Kernel> fit_on(const std::string& id, const nlohmann::json& params, const fft::Batch& batch) {
  auto kernel = fft::make_kernel(id, params);
  kernel->stage = "stage";
  fft::Schema schema{{{"k", "utf8"}, {"x", "i64"}}};
  kernel->resolve(schema);
  std::vector<std::unique_ptr<fft::AccState>> states;
  states.push_back(kernel->make_state());
  kernel->consume(*states.back(), batch);
  kernel->finalize(std::move(states));
  return kernel;
}

// Fit label_encode and group_agg twice, on `rows` as dictionary indices of one
// width and on the same strings unencoded, and require the same state and
// output. The first two rows are sliced off and each null index holds garbage.
// Both kernels have dictionary loops of their own, at fit and at transform.
template <typename IndexType>
void require_label_and_group_agg_match_plain(const std::vector<std::optional<std::string>>& entries,
                                             const std::vector<std::optional<int64_t>>& rows,
                                             const nlohmann::json& label_params = {{"columns", {"k"}}}) {
  std::vector<std::optional<std::string>> logical;
  std::vector<std::optional<int64_t>> numbers;
  for (std::size_t row = 2; row < rows.size(); ++row) {
    logical.push_back(rows[row] ? entries[static_cast<std::size_t>(*rows[row])] : std::nullopt);
    numbers.emplace_back(static_cast<int64_t>(row));
  }
  auto encoded = arrow::DictionaryArray::FromArrays(fft::testing::indices_with_garbage<IndexType>(rows)->Slice(2),
                                                    utf8_column(entries));
  REQUIRE(encoded.ok());
  const auto x = numeric_column<arrow::Int64Type>(numbers);
  const fft::Batch dictionary = make_batch({"k", "x"}, {*encoded, x});
  const fft::Batch plain = make_batch({"k", "x"}, {utf8_column(logical), x});
  const std::vector<std::pair<std::string, nlohmann::json>> kernels{
      {"label_encode", label_params}, {"group_agg", {{"key", "k"}, {"aggs", {{"x", {"count", "sum"}}}}}}};
  for (const auto& [id, params] : kernels) {
    CAPTURE(id);
    const auto reference = fit_on(id, params, plain);
    const auto actual = fit_on(id, params, dictionary);
    REQUIRE(actual->state() == reference->state());
    const fft::Batch expected = reference->transform(plain);
    const fft::Batch output = actual->transform(dictionary);
    REQUIRE(output.names == expected.names);
    for (std::size_t col = 0; col < output.cols.size(); ++col) {
      if (id == "group_agg" && output.names[col] == "k") continue;  // the key passes through as given
      REQUIRE(output.cols[col]->Equals(expected.cols[col]));
    }
  }
}
}  // namespace

// A dictionary is usually far smaller than its column, so a category can
// first appear past the dictionary's size: here "b" at row 3 and "c" at row 5
// of a three-entry dictionary, each twice. A fit that counted only as far as
// the dictionary's size, or marked entries instead of counting rows, would
// drop them under min_count 2, and the first would also leave group_agg
// without their keys.
TEST_CASE("label_encode and group_agg count every row past the dictionary's size") {
  const std::vector<std::optional<std::string>> entries{"a", "b", "c"};
  const std::vector<std::optional<int64_t>> rows{0, 0, 0, 0, 0, 1, std::nullopt, 2, 1, 2, 0};
  require_label_and_group_agg_match_plain<arrow::Int32Type>(entries, rows, {{"columns", {"k"}}, {"min_count", 2}});
}

TEST_CASE("label_encode and group_agg read dictionary indices of every width") {
  // int32_indices copies indices of every width but int32. A slot the copy
  // skipped would read 0, whose "b" also sits at entry 3, so the first row
  // reads entry 1 and the last entry 4: a skipped end changes the result.
  const std::vector<std::optional<std::string>> entries{"b", "a", std::nullopt, "b", "unseen", "unused"};
  std::vector<std::optional<int64_t>> rows{0, 0};
  for (int64_t row = 0; row < 70; ++row) {
    const int64_t index = (row + 1) % 6;
    rows.push_back(index == 5 || row % 11 == 10 ? std::nullopt : std::optional<int64_t>(index));
  }
  require_label_and_group_agg_match_plain<arrow::Int8Type>(entries, rows);
  require_label_and_group_agg_match_plain<arrow::UInt8Type>(entries, rows);
  require_label_and_group_agg_match_plain<arrow::Int16Type>(entries, rows);
  require_label_and_group_agg_match_plain<arrow::UInt16Type>(entries, rows);
  require_label_and_group_agg_match_plain<arrow::Int32Type>(entries, rows);
  require_label_and_group_agg_match_plain<arrow::UInt32Type>(entries, rows);
  require_label_and_group_agg_match_plain<arrow::Int64Type>(entries, rows);
  require_label_and_group_agg_match_plain<arrow::UInt64Type>(entries, rows);
  // Unsigned widths past their signed range: every entry once, scattered.
  for (const int64_t size : {256, 40000}) {
    std::vector<std::optional<std::string>> many;
    std::vector<std::optional<int64_t>> scattered{0, 0};
    for (int64_t i = 0; i < size; ++i) {
      many.emplace_back("k" + std::to_string(i));
      scattered.emplace_back(i * 7919 % size);  // 7919 is prime, so every entry comes up once
    }
    if (size == 256) {
      require_label_and_group_agg_match_plain<arrow::UInt8Type>(many, scattered);
    } else {
      require_label_and_group_agg_match_plain<arrow::UInt16Type>(many, scattered);
    }
  }
}

TEST_CASE("group_agg load_state rejects a malformed table") {
  const nlohmann::json params{{"key", "k"}, {"aggs", {{"n", {"sum", "max"}}}}};
  auto fresh = [&] {
    auto kernel = fft::make_kernel("group_agg", params);
    kernel->stage = "agg";
    return kernel;
  };
  const nlohmann::json good{{"key_type", "utf8"},
                            {"keys", {"a", "b"}},
                            {"column_types", {{"n", "i32"}}},
                            {"values", {{"n_sum", {3, 4}}, {"n_max", {2, 4}}}}};
  REQUIRE_NOTHROW(fresh()->load_state(good));
  auto broken = [&](auto mutate, const std::string& message) {
    nlohmann::json state = good;
    mutate(state);
    REQUIRE_THROWS_WITH(fresh()->load_state(state), ContainsSubstring(message));
  };
  broken([](nlohmann::json& state) { state.erase("keys"); }, "state must hold");
  broken([](nlohmann::json& state) { state["key_type"] = "f64"; }, "key_type must be");
  broken([](nlohmann::json& state) { state["keys"] = {"b", "a"}; }, "sorted and unique");
  broken([](nlohmann::json& state) { state["keys"] = {"a", "a"}; }, "sorted and unique");
  broken([](nlohmann::json& state) { state["keys"] = {1, 2}; }, "keys must all be strings");
  broken([](nlohmann::json& state) { state["values"]["n_sum"] = {3}; }, "one entry per key");
  broken([](nlohmann::json& state) { state["values"]["n_sum"] = {3, 4.5}; }, "must hold 64-bit integers");
  broken([](nlohmann::json& state) { state["values"]["n_max"] = {2, 1000000000000}; }, "does not fit in i32");
  broken([](nlohmann::json& state) { state["column_types"]["n"] = "utf8"; }, "only count applies");
  broken([](nlohmann::json& state) { state["column_types"].erase("n"); }, "has no type");
}

TEST_CASE("group_agg refuses an integer sum that overflows and non-finite floats") {
  const nlohmann::json params{{"key", "k"}, {"aggs", {{"n", {"sum"}}}}};
  const fft::Schema schema{{{"k", "utf8"}, {"n", "i64"}}};
  const int64_t huge = std::numeric_limits<int64_t>::max();
  REQUIRE_THROWS_WITH(
      (fit_group_agg(params, schema,
                     {make_batch({"k", "n"}, {utf8_column({"x", "x"}), numeric_column<arrow::Int64Type>({huge, 1})})})),
      ContainsSubstring("overflows a 64-bit integer"));
  // ... also when the overflow only appears while merging workers.
  REQUIRE_THROWS_WITH(
      (fit_group_agg(params, schema,
                     {make_batch({"k", "n"}, {utf8_column({"x"}), numeric_column<arrow::Int64Type>({huge})}),
                      make_batch({"k", "n"}, {utf8_column({"x"}), numeric_column<arrow::Int64Type>({1})})})),
      ContainsSubstring("overflows a 64-bit integer"));
  // Only the requested aggregates are accumulated: max alone never sums.
  REQUIRE_NOTHROW((
      fit_group_agg({{"key", "k"}, {"aggs", {{"n", {"max"}}}}}, schema,
                    {make_batch({"k", "n"}, {utf8_column({"x", "x"}), numeric_column<arrow::Int64Type>({huge, 1})})})));
  const fft::Schema floats{{{"k", "utf8"}, {"x", "f64"}}};
  REQUIRE_THROWS_WITH(
      (fit_group_agg({{"key", "k"}, {"aggs", {{"x", {"mean"}}}}}, floats,
                     {make_batch({"k", "x"}, {utf8_column({"x"}), numeric_column<arrow::DoubleType>(
                                                                      {std::numeric_limits<double>::infinity()})})})),
      ContainsSubstring("cannot aggregate NaN or infinity"));
}
