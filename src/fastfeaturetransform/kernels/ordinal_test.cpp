// Explicit C++ tests for the ordinal substrate: the codes, the expansion's
// data-dependent output columns, the accumulator-merge contract, and
// load_state.

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

}  // namespace

TEST_CASE("the ordinal substrate validates its params") {
  REQUIRE_THROWS_WITH((fft::make_kernel("ordinal_encode", {{"handle_missing", "sometimes"}})),
                      ContainsSubstring("'handle_missing' must be"));
  REQUIRE_THROWS_WITH((fft::make_kernel("ordinal_encode", {{"handle_unknown", "indicator"}})),
                      ContainsSubstring("'handle_unknown' must be"));
  REQUIRE_THROWS_WITH((fft::make_kernel("ordinal_encode", {{"index_start", 1.5}})),
                      ContainsSubstring("'index_start' must be an integer"));
  // REGRESSION: an integer past int64 is refused. JSON holds 2^63 as an
  // unsigned number, which used to be read wrapped to -2^63, so the codes
  // came out negative.
  REQUIRE_THROWS_WITH((fft::make_kernel("ordinal_encode", {{"index_start", uint64_t{1} << 63}})),
                      ContainsSubstring("'index_start' must be an integer"));
  REQUIRE_NOTHROW((fft::make_kernel("one_hot_encode", {{"handle_unknown", "indicator"}})));
  REQUIRE_THROWS_WITH((fft::make_kernel("rank_hot_encode", {{"handle_unknown", "indicator"}})),
                      ContainsSubstring("'handle_unknown' must be"));
  REQUIRE_THROWS_WITH((fft::make_kernel("one_hot_encode", {{"use_cat_names", 1}})),
                      ContainsSubstring("'use_cat_names' must be"));
}

TEST_CASE("ordinal_encode numbers sorted categories and puts the null last") {
  fft::Schema schema{{{"c", "utf8"}, {"n", "i64"}}};
  const fft::Batch a = make_batch({"c"}, {utf8_column({"c", "a"})});
  const fft::Batch b = make_batch({"c"}, {utf8_column({"b", std::nullopt, "a"})});
  auto kernel = fit_kernel("ordinal_encode", {{"columns", {"c"}}}, schema, {a, b});
  REQUIRE(schema.cols == fft::Schema{{{"c", "i64"}, {"n", "i64"}}}.cols);
  REQUIRE(kernel->state().at("categories").at("c") == nlohmann::json({"a", "b", "c"}));
  REQUIRE(kernel->state().at("has_null").at("c") == true);
  // Whatever order the workers merged in, the codes are the same.
  fft::Schema other{{{"c", "utf8"}, {"n", "i64"}}};
  REQUIRE(fit_kernel("ordinal_encode", {{"columns", {"c"}}}, other, {b, a})->state() == kernel->state());

  const std::vector<std::optional<std::string>> values{"a", "c", "zz", std::nullopt};
  fft::Batch out = kernel->transform(make_batch({"c"}, {utf8_column(values)}));
  const auto& codes = static_cast<const arrow::Int64Array&>(*out.cols[0]);
  REQUIRE(codes.Value(0) == 1);
  REQUIRE(codes.Value(1) == 3);
  REQUIRE(codes.Value(2) == -1);
  REQUIRE(codes.Value(3) == 4);
  for (std::size_t i = 0; i < values.size(); ++i) {
    fft::RowValue value = values[i].has_value() ? fft::RowValue(*values[i]) : fft::RowValue(std::monostate{});
    fft::Row row = kernel->transform_row({{"c", value}});
    REQUIRE(std::get<int64_t>(row[0].second) == codes.Value(static_cast<int64_t>(i)));
  }
}

TEST_CASE("one_hot_encode declares its columns only once it has fitted") {
  auto kernel = fft::make_kernel("one_hot_encode", {{"columns", {"c"}}});
  kernel->stage = "enc";
  fft::Schema before{{{"c", "utf8"}, {"n", "i64"}}};
  kernel->resolve(before);
  // Unfitted: the column is consumed and nothing is emitted yet, which is
  // why the engine resolves again after the fit pass.
  REQUIRE(before.cols == fft::Schema{{{"n", "i64"}}}.cols);
  REQUIRE(kernel->outputs.empty());

  auto state = kernel->make_state();
  kernel->consume(*state, make_batch({"c"}, {utf8_column({"b", "a", "b"})}));
  std::vector<std::unique_ptr<fft::AccState>> states;
  states.push_back(std::move(state));
  kernel->finalize(std::move(states));

  fft::Schema after{{{"c", "utf8"}, {"n", "i64"}}};
  kernel->resolve(after);
  REQUIRE(after.cols == fft::Schema{{{"n", "i64"}, {"c_1", "i64"}, {"c_2", "i64"}}}.cols);
  REQUIRE(kernel->outputs == std::vector<std::string>{"c_1", "c_2"});

  fft::Batch out = kernel->transform(make_batch(
      {"c", "n"}, {utf8_column({"a", "b", "zz"}), fft::testing::numeric_column<arrow::Int64Type>({1, 2, 3})}));
  REQUIRE(out.names == std::vector<std::string>{"n", "c_1", "c_2"});
  const auto& first = static_cast<const arrow::Int64Array&>(*out.cols[1]);
  const auto& second = static_cast<const arrow::Int64Array&>(*out.cols[2]);
  REQUIRE(first.Value(0) == 1);
  REQUIRE(second.Value(1) == 1);
  REQUIRE(first.Value(2) == 0);  // unknown lights nothing
  REQUIRE(second.Value(2) == 0);
  fft::Row row = kernel->transform_row({{"c", std::string("b")}, {"n", int64_t{2}}});
  REQUIRE(row.size() == 3);
  REQUIRE(row[0].first == "n");
  REQUIRE(std::get<int64_t>(*fft::row_find(row, "c_2")) == 1);
  REQUIRE_THROWS_WITH(kernel->transform_row({{"n", int64_t{1}}}), ContainsSubstring("missing from record"));
}

TEST_CASE("rank_hot_encode lights every column up to the category's rank") {
  fft::Schema schema{{{"c", "utf8"}}};
  auto kernel =
      fit_kernel("rank_hot_encode", {{"columns", {"c"}}}, schema, {make_batch({"c"}, {utf8_column({"c", "a", "b"})})});
  fft::Schema resolved{{{"c", "utf8"}}};
  kernel->resolve(resolved);
  REQUIRE(resolved.cols == fft::Schema{{{"c_1", "i64"}, {"c_2", "i64"}, {"c_3", "i64"}}}.cols);
  fft::Batch out = kernel->transform(make_batch({"c"}, {utf8_column({"a", "b", "c"})}));
  const auto& third = static_cast<const arrow::Int64Array&>(*out.cols[2]);
  const auto& second = static_cast<const arrow::Int64Array&>(*out.cols[1]);
  REQUIRE(second.Value(0) == 0);
  REQUIRE(second.Value(1) == 1);
  REQUIRE(third.Value(2) == 1);
}

// The batch path records each row's rank and then fills one column at a time;
// the record path lights its columns as it goes. Every row and column has to
// agree, including the 0 under a null and for an unknown.
TEST_CASE("rank_hot_encode's batch path agrees with the record path row by row") {
  std::vector<std::optional<std::string>> values;
  for (int i = 0; i < 300; ++i) {
    if (i % 7 == 0) {
      values.emplace_back(std::nullopt);
    } else if (i % 11 == 0) {
      values.emplace_back("zz");  // never fitted
    } else {
      values.emplace_back(std::string(1, static_cast<char>('a' + i % 4)));
    }
  }
  for (const char* unknown : {"value", "return_nan"}) {
    fft::Schema schema{{{"c", "utf8"}}};
    auto kernel = fit_kernel("rank_hot_encode", {{"columns", {"c"}}, {"handle_unknown", unknown}}, schema,
                             {make_batch({"c"}, {utf8_column({"a", "b", "c", "d"})})});
    for (int64_t offset : {int64_t{0}, int64_t{3}}) {
      CAPTURE(unknown, offset);
      const fft::Batch out = kernel->transform(make_batch({"c"}, {utf8_column(values)->Slice(offset)}));
      for (std::size_t row = 0; row + static_cast<std::size_t>(offset) < values.size(); ++row) {
        const std::optional<std::string>& value = values[row + static_cast<std::size_t>(offset)];
        const fft::Row record =
            kernel->transform_row({{"c", value ? fft::RowValue(*value) : fft::RowValue(std::monostate{})}});
        REQUIRE(record.size() == out.cols.size());
        for (std::size_t c = 0; c < out.cols.size(); ++c) {
          const auto& integers = static_cast<const arrow::Int64Array&>(*out.cols[c]);
          const auto slot = static_cast<int64_t>(row);
          if (std::holds_alternative<std::monostate>(record[c].second)) {
            REQUIRE(integers.IsNull(slot));
            REQUIRE(integers.Value(slot) == 0);
          } else {
            REQUIRE(integers.IsValid(slot));
            REQUIRE(integers.Value(slot) == std::get<int64_t>(record[c].second));
          }
        }
      }
    }
  }
}

// REGRESSION: upstream's RankHotEncoder reads an unfitted null out of the same
// default row an unseen category takes, so handle_unknown decides it; ours
// used to answer zeros whatever handle_unknown said. one_hot_encode has a
// mapping row for the null and is right to keep giving it zeros.
TEST_CASE("rank_hot_encode hands a null it never fitted to handle_unknown") {
  auto null_row = [](const char* id, const nlohmann::json& params) {
    fft::Schema schema{{{"c", "utf8"}}};
    auto kernel = fit_kernel(id, params, schema, {make_batch({"c"}, {utf8_column({"a", "b"})})});
    const fft::Batch out = kernel->transform(make_batch({"c"}, {utf8_column({std::nullopt})}));
    std::vector<std::optional<int64_t>> values;
    for (const auto& column : out.cols) {
      const auto& integers = static_cast<const arrow::Int64Array&>(*column);
      values.push_back(integers.IsNull(0) ? std::nullopt : std::optional<int64_t>{integers.Value(0)});
    }
    // The record path has to answer the same way.
    const fft::Row row = kernel->transform_row({{"c", fft::RowValue(std::monostate{})}});
    REQUIRE(row.size() == values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
      REQUIRE(std::holds_alternative<std::monostate>(row[i].second) == !values[i].has_value());
      if (values[i].has_value()) REQUIRE(std::get<int64_t>(row[i].second) == *values[i]);
    }
    return values;
  };
  const std::vector<std::optional<int64_t>> nulls{std::nullopt, std::nullopt};
  const std::vector<std::optional<int64_t>> zeros{0, 0};

  REQUIRE(null_row("rank_hot_encode", {{"columns", {"c"}}, {"handle_unknown", "return_nan"}}) == nulls);
  // 'error' refuses an unseen category, never an unfitted null.
  REQUIRE(null_row("rank_hot_encode", {{"columns", {"c"}}, {"handle_unknown", "error"}}) == zeros);
  REQUIRE(null_row("rank_hot_encode", {{"columns", {"c"}}}) == zeros);
  // one_hot_encode has a mapping row for the null, so it keeps giving zeros.
  REQUIRE(null_row("one_hot_encode", {{"columns", {"c"}}, {"handle_unknown", "return_nan"}}) == zeros);
}

// REGRESSION: upstream keeps its pre-encoder's slot for the missing value as a
// level under handle_missing='return_nan', so rank_hot_encode emits a trailing
// '<column>_-2' column that nothing lights up. We used to drop it.
TEST_CASE("rank_hot_encode keeps the placeholder column under handle_missing return_nan") {
  auto emitted = [](const char* id, const nlohmann::json& params,
                    const std::vector<std::optional<std::string>>& values) {
    fft::Schema schema{{{"c", "utf8"}}};
    auto kernel = fit_kernel(id, params, schema, {make_batch({"c"}, {utf8_column(values)})});
    fft::Schema resolved{{{"c", "utf8"}}};
    kernel->resolve(resolved);
    return kernel;
  };
  auto ranked =
      emitted("rank_hot_encode", {{"columns", {"c"}}, {"handle_missing", "return_nan"}}, {"a", "b", std::nullopt});
  REQUIRE(ranked->outputs == std::vector<std::string>{"c_1", "c_2", "c_-2"});
  const fft::Batch out = ranked->transform(make_batch({"c"}, {utf8_column({"a", "b", std::nullopt})}));
  REQUIRE(out.names == std::vector<std::string>{"c_1", "c_2", "c_-2"});
  const auto& placeholder = static_cast<const arrow::Int64Array&>(*out.cols[2]);
  REQUIRE(placeholder.Value(0) == 0);  // nothing ever lights the placeholder up
  REQUIRE(placeholder.Value(1) == 0);
  REQUIRE(placeholder.IsNull(2));  // a null is a row of nulls, that column included
  fft::Row row = ranked->transform_row({{"c", std::string("b")}});
  REQUIRE(row.size() == 3);
  REQUIRE(std::get<int64_t>(*fft::row_find(row, "c_2")) == 1);
  REQUIRE(std::get<int64_t>(*fft::row_find(row, "c_-2")) == 0);

  // use_cat_names spells it the way upstream spells a NaN category.
  auto named = emitted("rank_hot_encode",
                       {{"columns", {"c"}}, {"handle_missing", "return_nan"}, {"use_cat_names", true}}, {"a", "b"});
  REQUIRE(named->outputs == std::vector<std::string>{"c_a", "c_b", "c_nan"});

  // handle_missing='error' re-fits the pre-encoder strictly, so no slot and no
  // column; one_hot_encode drops the column under every option.
  auto strict = emitted("rank_hot_encode", {{"columns", {"c"}}, {"handle_missing", "error"}}, {"a", "b"});
  REQUIRE(strict->outputs == std::vector<std::string>{"c_1", "c_2"});
  auto hot =
      emitted("one_hot_encode", {{"columns", {"c"}}, {"handle_missing", "return_nan"}}, {"a", "b", std::nullopt});
  REQUIRE(hot->outputs == std::vector<std::string>{"c_1", "c_2"});
}

// REGRESSION: the '#' that disambiguates a repeated output name spans every
// encoded column, as upstream's counter does. It used to be scoped to one
// column, so encoding 'a' (holding 'b_c') beside 'a_b' (holding 'c') emitted
// 'a_b_c' twice: an unreadable Parquet part on the batch path, and one feature
// fewer on the record path.
TEST_CASE("one_hot_encode disambiguates an output name that repeats across columns") {
  fft::Schema schema{{{"a", "utf8"}, {"a_b", "utf8"}}};
  auto kernel = fit_kernel("one_hot_encode", {{"columns", {"a", "a_b"}}, {"use_cat_names", true}}, schema,
                           {make_batch({"a", "a_b"}, {utf8_column({"b_c", "d"}), utf8_column({"c", "e"})})});
  fft::Schema resolved{{{"a", "utf8"}, {"a_b", "utf8"}}};
  kernel->resolve(resolved);
  REQUIRE(kernel->outputs == std::vector<std::string>{"a_b_c", "a_d", "a_b_c#", "a_b_e"});

  fft::Batch out = kernel->transform(make_batch({"a", "a_b"}, {utf8_column({"b_c", "d"}), utf8_column({"c", "e"})}));
  REQUIRE(out.names == std::vector<std::string>{"a_b_c", "a_d", "a_b_c#", "a_b_e"});
  REQUIRE(static_cast<const arrow::Int64Array&>(*out.cols[0]).Value(0) == 1);
  REQUIRE(static_cast<const arrow::Int64Array&>(*out.cols[2]).Value(0) == 1);
  // The record path emits the same four keys, not three.
  fft::Row row = kernel->transform_row({{"a", std::string("b_c")}, {"a_b", std::string("c")}});
  REQUIRE(row.size() == 4);
  REQUIRE(std::get<int64_t>(*fft::row_find(row, "a_b_c")) == 1);
  REQUIRE(std::get<int64_t>(*fft::row_find(row, "a_b_c#")) == 1);
}

// REGRESSION: the batch path refuses a passthrough column named like a
// generated one at resolve; the record path used to keep both and let the
// generated value win, so the same record was a hard error through Parquet and
// a silent substitution through transform_dict.
TEST_CASE("one_hot_encode refuses a record key that collides with a generated column") {
  fft::Schema schema{{{"c", "utf8"}}};
  auto kernel =
      fit_kernel("one_hot_encode", {{"columns", {"c"}}}, schema, {make_batch({"c"}, {utf8_column({"a", "b"})})});
  fft::Schema resolved{{{"c", "utf8"}, {"c_1", "i64"}}};
  REQUIRE_THROWS_WITH(kernel->resolve(resolved), ContainsSubstring("output column 'c_1' already exists in input"));
  REQUIRE_THROWS_WITH(kernel->transform_row({{"c", std::string("a")}, {"c_1", int64_t{9}}}),
                      ContainsSubstring("key 'c_1' already present in record"));
}

// REGRESSION: 'error' is the strict option, so it refuses a null wherever one
// turns up, not only in the fit data.
TEST_CASE("the ordinal substrate refuses a null at transform under handle_missing error") {
  for (const char* id : {"ordinal_encode", "one_hot_encode", "rank_hot_encode"}) {
    fft::Schema schema{{{"c", "utf8"}}};
    auto kernel = fit_kernel(id, {{"columns", {"c"}}, {"handle_missing", "error"}}, schema,
                             {make_batch({"c"}, {utf8_column({"a", "b"})})});
    REQUIRE_THROWS_WITH(kernel->transform(make_batch({"c"}, {utf8_column({"a", std::nullopt})})),
                        ContainsSubstring("column 'c' has a null and handle_missing is 'error'"));
    REQUIRE_THROWS_WITH(kernel->transform_row({{"c", fft::RowValue(std::monostate{})}}),
                        ContainsSubstring("column 'c' has a null and handle_missing is 'error'"));
  }
}

// REGRESSION: index_start + the level count used to wrap into a negative code,
// which is undefined behavior and a negative index into an embedding table.
TEST_CASE("ordinal_encode refuses codes that do not fit in a 64-bit integer") {
  fft::Schema schema{{{"c", "utf8"}}};
  const nlohmann::json params{{"columns", {"c"}}, {"index_start", INT64_MAX}};
  REQUIRE_THROWS_WITH(fit_kernel("ordinal_encode", params, schema, {make_batch({"c"}, {utf8_column({"a", "b"})})}),
                      ContainsSubstring("highest code does not fit in a 64-bit integer"));
  // The same manifest, hand-edited, is refused at load rather than served.
  auto loading = fft::make_kernel("ordinal_encode", params);
  loading->stage = "enc";
  REQUIRE_THROWS_WITH(
      loading->load_state({{"columns", {"c"}}, {"categories", {{"c", {"a", "b"}}}}, {"has_null", {{"c", false}}}}),
      ContainsSubstring("highest code does not fit in a 64-bit integer"));
  // One category exactly fills the range, so it is allowed.
  fft::Schema room{{{"c", "utf8"}}};
  REQUIRE_NOTHROW(fit_kernel("ordinal_encode", params, room, {make_batch({"c"}, {utf8_column({"a"})})}));
}

// REGRESSION: the width of an expanding encoder is learned, so a column that
// was low-cardinality in the fit data and is not in production would allocate
// a row-length vector per category per batch. The fit refuses instead.
TEST_CASE("one_hot_encode caps how many columns it will generate") {
  std::vector<std::optional<std::string>> many;
  many.reserve(4097);
  for (int i = 0; i < 4097; ++i) many.emplace_back(std::to_string(i));
  fft::Schema schema{{{"c", "utf8"}}};
  REQUIRE_THROWS_WITH(
      fit_kernel("one_hot_encode", {{"columns", {"c"}}}, schema, {make_batch({"c"}, {utf8_column(many)})}),
      ContainsSubstring("more than the 4096 allowed"));
}

// REGRESSION: the encoded columns are erased before the collision check, as
// hash_encode does, so a generated name may reuse an encoded column's. The
// check used to run first, so encoding 'a' beside 'a_1' was a fit-time error
// with no way out but renaming a column.
TEST_CASE("one_hot_encode may generate a name an encoded column had") {
  fft::Schema schema{{{"a", "utf8"}, {"a_1", "utf8"}}};
  auto kernel = fit_kernel("one_hot_encode", {{"columns", {"a", "a_1"}}}, schema,
                           {make_batch({"a", "a_1"}, {utf8_column({"x", "y"}), utf8_column({"p", "q"})})});
  fft::Schema resolved{{{"a", "utf8"}, {"a_1", "utf8"}}};
  REQUIRE_NOTHROW(kernel->resolve(resolved));
  REQUIRE(kernel->outputs == std::vector<std::string>{"a_1", "a_2", "a_1_1", "a_1_2"});
}

TEST_CASE("the ordinal substrate's load_state rejects a malformed table") {
  auto fresh = [] {
    auto kernel = fft::make_kernel("one_hot_encode", {{"columns", {"c"}}});
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
  broken([](nlohmann::json& state) { state["categories"]["c"] = {"a", "a"}; }, "must be sorted and unique");
  broken([](nlohmann::json& state) { state["categories"]["c"] = {1, 2}; }, "must hold strings");
  broken([](nlohmann::json& state) { state["categories"]["c"] = nlohmann::json::array(); }, "is empty");
  broken([](nlohmann::json& state) { state["has_null"]["c"] = "yes"; }, "must be true or false");
  // REGRESSION: the null category is the entry a hand-edited manifest is
  // likeliest to mangle, and a wrong shape used to be read as a legitimate
  // 'no null category' rather than as an error.
  broken([](nlohmann::json& state) { state["has_null"] = true; }, "'has_null' must be an object");
  broken([](nlohmann::json& state) { state["has_null"].erase("c"); }, "has_null has no entry for column 'c'");
}
