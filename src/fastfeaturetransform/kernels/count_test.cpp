// Explicit C++ tests for count_encode: the params, the schema rewrite, the
// accumulator-merge (monoid) contract, group folding, and load_state.

#include <arrow/array.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <limits>
#include <sstream>

#include "fastfeaturetransform/kernel.hpp"
#include "fastfeaturetransform/kernels/dictionary_test_support.hpp"
#include "fastfeaturetransform/test_support.hpp"

namespace fft = fastfeaturetransform;
using Catch::Matchers::ContainsSubstring;
using fft::testing::make_batch;
using fft::testing::utf8_column;

namespace {

// Fit a count_encode kernel with one worker state per batch.
std::unique_ptr<fft::Kernel> fit_count(const nlohmann::json& params, fft::Schema schema,
                                       const std::vector<fft::Batch>& batches) {
  return fft::testing::fit_kernel_per_batch("count_encode", params, std::move(schema), batches, "counts");
}

// What `kernel` answers for every row of `batch`, whose columns are all
// utf8, after checking that the record path answers what the batch path does.
std::vector<fft::Row> both_paths(const fft::Kernel& kernel, const fft::Batch& batch) {
  const fft::Batch out = kernel.transform(batch);
  std::vector<fft::Row> rows;
  for (int64_t row = 0; row < batch.rows; ++row) {
    fft::Row record;
    fft::Row answers;
    for (std::size_t i = 0; i < batch.names.size(); ++i) {
      const auto& text = static_cast<const arrow::StringArray&>(*batch.cols[i]);
      record.emplace_back(batch.names[i],
                          text.IsNull(row) ? fft::RowValue(std::monostate{}) : fft::RowValue(text.GetString(row)));
      const arrow::Array& column = *out.cols[i];
      fft::RowValue answer = std::monostate{};
      if (column.type_id() == arrow::Type::DOUBLE) {
        if (!column.IsNull(row)) answer = static_cast<const arrow::DoubleArray&>(column).Value(row);
      } else {
        REQUIRE(column.type_id() == arrow::Type::INT64);
        if (!column.IsNull(row)) answer = static_cast<const arrow::Int64Array&>(column).Value(row);
      }
      answers.emplace_back(batch.names[i], answer);
    }
    REQUIRE(kernel.transform_row(record) == answers);
    rows.push_back(std::move(answers));
  }
  return rows;
}

}  // namespace

TEST_CASE("count_encode validates its params") {
  REQUIRE_THROWS_WITH((fft::make_kernel("count_encode", {{"handle_missing", "sometimes"}})),
                      ContainsSubstring("'handle_missing' must be"));
  REQUIRE_THROWS_WITH((fft::make_kernel("count_encode", {{"handle_unknown", "sometimes"}})),
                      ContainsSubstring("'handle_unknown' must be"));
  // REGRESSION: an integer fill past int64 is refused. 2^63 used to be read
  // wrapped to -2^63, which an unseen category then received.
  REQUIRE_THROWS_WITH((fft::make_kernel("count_encode", {{"handle_unknown", uint64_t{1} << 63}})),
                      ContainsSubstring("'handle_unknown' must be"));
  REQUIRE_THROWS_WITH((fft::make_kernel("count_encode", {{"normalize", "yes"}})),
                      ContainsSubstring("'normalize' must be"));
  for (const double size : {0.0, std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()}) {
    REQUIRE_THROWS_WITH((fft::make_kernel("count_encode", {{"min_group_size", size}})),
                        ContainsSubstring("'min_group_size' must be a finite number > 0"));
  }
  // REGRESSION: an integer min_group_size round-trips as itself. Near 2^63
  // its double rounds up to 2^63, and params() used to cast that back to
  // int64, which on x86-64 gives -2^63, a value load_state refuses.
  const int64_t largest = std::numeric_limits<int64_t>::max();
  REQUIRE(fft::make_kernel("count_encode", {{"min_group_size", largest}})->params().at("min_group_size") == largest);
  REQUIRE_THROWS_WITH((fft::make_kernel("count_encode", {{"min_group_size", uint64_t{1} << 63}})),
                      ContainsSubstring("an integer one at most 2^63 - 1"));
  REQUIRE_THROWS_WITH((fft::make_kernel("count_encode", {{"combine_min_nan_groups", "maybe"}})),
                      ContainsSubstring("must be true, false or 'force'"));
  REQUIRE_THROWS_WITH(
      (fft::make_kernel("count_encode", {{"combine_min_nan_groups", "force"}, {"handle_missing", "return_nan"}})),
      ContainsSubstring("needs handle_missing"));
  REQUIRE(fft::make_kernel("count_encode", nlohmann::json::object())->needs_fit());
}

TEST_CASE("count_encode rewrites the column type and refuses a non-utf8 column") {
  auto kernel = fft::make_kernel("count_encode", {{"columns", {"c"}}});
  kernel->stage = "counts";
  fft::Schema schema{{{"c", "utf8"}, {"n", "i64"}}};
  kernel->resolve(schema);
  REQUIRE(schema.cols == fft::Schema{{{"c", "i64"}, {"n", "i64"}}}.cols);

  auto ratios = fft::make_kernel("count_encode", {{"columns", {"c"}}, {"normalize", true}});
  ratios->stage = "counts";
  fft::Schema fresh{{{"c", "utf8"}}};
  ratios->resolve(fresh);
  REQUIRE(*fresh.find("c") == "f64");

  auto numeric = fft::make_kernel("count_encode", {{"columns", {"n"}}});
  numeric->stage = "counts";
  fft::Schema numbers{{{"n", "i64"}}};
  REQUIRE_THROWS_WITH(numeric->resolve(numbers), ContainsSubstring("is i64, expected utf8"));
}

TEST_CASE("count_encode merges worker tallies and both paths agree") {
  const nlohmann::json params{{"columns", {"c"}}};
  const fft::Schema schema{{{"c", "utf8"}}};
  const fft::Batch a = make_batch({"c"}, {utf8_column({"x", "y", std::nullopt})});
  const fft::Batch b = make_batch({"c"}, {utf8_column({"x", "x", "z"})});
  auto kernel = fit_count(params, schema, {a, b});
  const nlohmann::json state = kernel->state();
  REQUIRE(state.at("counts").at("c") == nlohmann::json({{"x", 3}, {"y", 1}, {"z", 1}}));
  REQUIRE(state.at("null_counts").at("c") == 1);
  REQUIRE(state.at("rows") == 6);
  // Only the merged tallies may decide a count: this is the monoid contract.
  REQUIRE(fit_count(params, schema, {b, a})->state() == state);

  const std::vector<std::optional<std::string>> values{"x", "y", "zzz", std::nullopt};
  fft::Batch out = kernel->transform(make_batch({"c"}, {utf8_column(values)}));
  const auto& counts = static_cast<const arrow::Int64Array&>(*out.cols[0]);
  REQUIRE(counts.Value(0) == 3);
  REQUIRE(counts.Value(2) == 0);  // unseen
  REQUIRE(counts.Value(3) == 1);  // the null was a category at fit
  for (std::size_t i = 0; i < values.size(); ++i) {
    fft::RowValue value = values[i].has_value() ? fft::RowValue(*values[i]) : fft::RowValue(std::monostate{});
    fft::Row row = kernel->transform_row({{"c", value}});
    REQUIRE(std::get<int64_t>(row[0].second) == counts.Value(static_cast<int64_t>(i)));
  }
  REQUIRE_THROWS_WITH(kernel->transform_row({{"c", int64_t{1}}}), ContainsSubstring("must be a string or null"));
  REQUIRE_THROWS_WITH(kernel->transform_row({{"n", int64_t{1}}}), ContainsSubstring("missing from record"));
}

TEST_CASE("count_encode folds categories under the threshold into one group") {
  const fft::Schema schema{{{"c", "utf8"}}};
  const fft::Batch batch = make_batch({"c"}, {utf8_column({"a", "a", "a", "a", "b", "c", "d", "d"})});
  auto folded = fit_count({{"columns", {"c"}}, {"min_group_size", 3}}, schema, {batch});
  REQUIRE(folded->state().at("counts").at("c") == nlohmann::json({{"a", 4}, {"b_c_d", 4}}));
  fft::Row row = folded->transform_row({{"c", std::string("b")}});
  REQUIRE(std::get<int64_t>(row[0].second) == 4);

  // An empty member keeps its separator, as in upstream's '_'.join: "" and 'x'
  // make '_x', and "" and the null make '_nan'.
  const fft::Batch empty = make_batch({"c"}, {utf8_column({"", "x", "keep", "keep", "keep"})});
  auto with_x = fit_count({{"columns", {"c"}}, {"min_group_size", 2}}, schema, {empty});
  REQUIRE(with_x->state().at("counts").at("c") == nlohmann::json({{"_x", 2}, {"keep", 3}}));
  const fft::Batch empty_null = make_batch({"c"}, {utf8_column({"", std::nullopt, "keep", "keep", "keep"})});
  auto with_null = fit_count({{"columns", {"c"}}, {"min_group_size", 2}}, schema, {empty_null});
  REQUIRE(with_null->state().at("groups").at("c") == nlohmann::json({{"_nan", nlohmann::json::array({""})}}));

  // A threshold that nothing falls under, and one that everything falls
  // under, both leave the tally alone.
  auto intact = fit_count({{"columns", {"c"}}, {"min_group_size", 1}}, schema, {batch});
  REQUIRE(intact->state().at("counts").at("c") == nlohmann::json({{"a", 4}, {"b", 1}, {"c", 1}, {"d", 2}}));
  auto all_small = fit_count({{"columns", {"c"}}, {"min_group_size", 99}}, schema, {batch});
  REQUIRE(all_small->state().at("counts").at("c") == nlohmann::json({{"a", 4}, {"b", 1}, {"c", 1}, {"d", 2}}));
}

// REGRESSION: the fit refuses a folded group named like a surviving category.
// The name used to reach the table twice, and emplace() silently kept
// whichever entry the sort put first, so 'x_y' lost the count of the rows that
// actually carried it. The default group name is the members joined with '_',
// so no user error is needed: any column holding 'x', 'y' and 'x_y' is enough.
TEST_CASE("count_encode never folds a group over a category of the same name") {
  const fft::Schema schema{{{"c", "utf8"}}};
  // x and y fall under the threshold and join into 'x_y', which is already a
  // category with a count of its own.
  const fft::Batch collides =
      make_batch({"c"}, {utf8_column({"x", "x", "y", "y", "x_y", "x_y", "x_y", "z", "z", "z", "z"})});
  REQUIRE_THROWS_WITH(fit_count({{"columns", {"c"}}, {"min_group_size", 3}}, schema, {collides}),
                      ContainsSubstring("the folded group is named 'x_y', which is already a category"));

  // The same for a name the user chose.
  const fft::Batch named = make_batch({"c"}, {utf8_column({"a", "b", "keep", "keep", "keep", "keep"})});
  REQUIRE_THROWS_WITH(
      fit_count({{"columns", {"c"}}, {"min_group_size", 3}, {"min_group_name", "keep"}}, schema, {named}),
      ContainsSubstring("the folded group is named 'keep', which is already a category"));

  // A group name that is one of the folded members is not a collision: the
  // member is gone from the tally and the group takes its key, which is what
  // upstream's mapping does too.
  auto folded = fit_count({{"columns", {"c"}}, {"min_group_size", 3}, {"min_group_name", "a"}}, schema, {named});
  REQUIRE(folded->state().at("counts").at("c") == nlohmann::json({{"a", 2}, {"keep", 4}}));
  REQUIRE(std::get<int64_t>(folded->transform_row({{"c", std::string("b")}})[0].second) == 2);

  // Such a table loads back, as does one whose group a lone member shares
  // with the null.
  const fft::Batch lone = make_batch({"c"}, {utf8_column({"a", std::nullopt, "keep", "keep", "keep", "keep"})});
  auto joined =
      fit_count({{"columns", {"c"}}, {"min_group_size", 3}, {"combine_min_nan_groups", "force"}}, schema, {lone});
  REQUIRE(joined->state().at("groups").at("c") == nlohmann::json({{"a_nan", nlohmann::json::array({"a"})}}));
  for (const auto* fitted : {folded.get(), joined.get()}) {
    const nlohmann::json state = fitted->state();
    auto loaded = fft::make_kernel("count_encode", fitted->params());
    loaded->stage = "counts";
    INFO(state.dump());
    REQUIRE_NOTHROW(loaded->load_state(state));
    REQUIRE(loaded->state() == state);
    for (const char* value : {"a", "b", "keep", "unseen"}) {
      REQUIRE(loaded->transform_row({{"c", std::string(value)}}) == fitted->transform_row({{"c", std::string(value)}}));
    }
  }
}

// Upstream folds the null category into the small group when its count is
// below the threshold too (combine_min_nan_groups=true) and whenever there
// were nulls ('force'): the group's count includes it, a null answers that
// count, and the default name spells it 'nan'. Under false the null stays
// apart and is not a member, so a lone small category does not fold.
TEST_CASE("count_encode folds the null into the small group as upstream does") {
  const fft::Schema schema{{{"c", "utf8"}}};
  const fft::Batch lone = make_batch({"c"}, {utf8_column({"a", "a", "a", "a", "b", "b", "b", "c", std::nullopt})});
  const fft::Batch common =
      make_batch({"c"}, {utf8_column({"a", "a", "a", "a", "b", "c", std::nullopt, std::nullopt, std::nullopt})});
  struct Case {
    const fft::Batch* batch;
    nlohmann::json combine;
    nlohmann::json counts;
    int64_t c_answer;
    int64_t null_answer;
  };
  const std::vector<Case> cases{
      {&lone, true, {{"a", 4}, {"b", 3}, {"c_nan", 2}}, 2, 2},
      {&lone, "force", {{"a", 4}, {"b", 3}, {"c_nan", 2}}, 2, 2},
      {&lone, false, {{"a", 4}, {"b", 3}, {"c", 1}}, 1, 1},
      {&common, true, {{"a", 4}, {"b_c", 2}}, 2, 3},
      {&common, "force", {{"a", 4}, {"b_c_nan", 5}}, 5, 5},
      {&common, false, {{"a", 4}, {"b_c", 2}}, 2, 3},
  };
  const auto answer = [](const fft::Kernel& kernel, fft::RowValue value) {
    return std::get<int64_t>(kernel.transform_row({{"c", std::move(value)}})[0].second);
  };
  for (const Case& c : cases) {
    CAPTURE(c.combine, c.counts);
    auto kernel = fit_count({{"columns", {"c"}}, {"min_group_size", 2}, {"combine_min_nan_groups", c.combine}}, schema,
                            {*c.batch});
    REQUIRE(kernel->state().at("counts").at("c") == c.counts);
    REQUIRE(answer(*kernel, std::string("c")) == c.c_answer);
    REQUIRE(answer(*kernel, std::monostate{}) == c.null_answer);
  }
}

// A count that sits on a fractional threshold: 7 is below 0.07 * 100 =
// 7.000000000000001, but 7 / 100 is not below 0.07. Upstream compares counts
// without normalize and ratios with it, and so must the fit.
TEST_CASE("count_encode compares counts, or ratios under normalize, with min_group_size as upstream does") {
  const fft::Schema schema{{{"c", "utf8"}}};
  std::vector<std::optional<std::string>> values(86, std::string("keep"));
  values.insert(values.end(), 7, std::string("a"));
  values.insert(values.end(), 7, std::string("b"));
  const fft::Batch batch = make_batch({"c"}, {utf8_column(values)});
  auto counts = fit_count({{"columns", {"c"}}, {"min_group_size", 0.07}}, schema, {batch});
  REQUIRE(counts->state().at("counts").at("c") == nlohmann::json({{"a_b", 14}, {"keep", 86}}));
  auto ratios = fit_count({{"columns", {"c"}}, {"min_group_size", 0.07}, {"normalize", true}}, schema, {batch});
  REQUIRE(ratios->state().at("counts").at("c") == nlohmann::json({{"a", 7}, {"b", 7}, {"keep", 86}}));
  REQUIRE(std::get<double>(ratios->transform_row({{"c", std::string("a")}})[0].second) == 0.07);
}

// REGRESSION: normalize divides a category's count by the fitted rows, but an
// integer handle_unknown is the literal number the user asked to see. It used
// to be divided too, which for a large fit set is indistinguishable from a
// real small count.
TEST_CASE("count_encode does not normalize an integer handle_unknown fill") {
  const fft::Schema schema{{{"c", "utf8"}}};
  const fft::Batch batch = make_batch({"c"}, {utf8_column({"a", "a", "b", "b"})});
  auto kernel = fit_count({{"columns", {"c"}}, {"normalize", true}, {"handle_unknown", 7}}, schema, {batch});

  fft::Batch out = kernel->transform(make_batch({"c"}, {utf8_column({"a", "zz", std::nullopt})}));
  const auto& values = static_cast<const arrow::DoubleArray&>(*out.cols[0]);
  REQUIRE(values.Value(0) == 0.5);  // a real count, scaled by the 4 fit rows
  REQUIRE(values.Value(1) == 7.0);  // the fill, as asked for
  REQUIRE(values.Value(2) == 7.0);  // a null the fit never saw takes it too
  // The record path has to answer the same, which is how the bug hid: both
  // went through the same look_up and both divided.
  REQUIRE(std::get<double>(kernel->transform_row({{"c", std::string("a")}})[0].second) == 0.5);
  REQUIRE(std::get<double>(kernel->transform_row({{"c", std::string("zz")}})[0].second) == 7.0);
  REQUIRE(std::get<double>(kernel->transform_row({{"c", fft::RowValue(std::monostate{})}})[0].second) == 7.0);
}

// REGRESSION: 'error' is the strict option, so it refuses a null wherever one
// turns up. It used to hold for the fit data only, which is the one case a
// user picking it does not need protecting from.
TEST_CASE("count_encode refuses a null at transform under handle_missing error") {
  const fft::Schema schema{{{"c", "utf8"}}};
  auto kernel = fit_count({{"columns", {"c"}}, {"handle_missing", "error"}}, schema,
                          {make_batch({"c"}, {utf8_column({"a", "b"})})});
  REQUIRE_THROWS_WITH(kernel->transform(make_batch({"c"}, {utf8_column({"a", std::nullopt})})),
                      ContainsSubstring("column 'c' has a null and handle_missing is 'error'"));
  REQUIRE_THROWS_WITH(kernel->transform_row({{"c", fft::RowValue(std::monostate{})}}),
                      ContainsSubstring("column 'c' has a null and handle_missing is 'error'"));
}

TEST_CASE("count_encode load_state rejects a malformed table") {
  auto fresh = [] {
    auto kernel = fft::make_kernel("count_encode", {{"columns", {"c"}}});
    kernel->stage = "counts";
    return kernel;
  };
  const nlohmann::json good{
      {"columns", {"c"}}, {"counts", {{"c", {{"x", 2}}}}}, {"null_counts", {{"c", nullptr}}}, {"rows", 2}};
  REQUIRE_NOTHROW(fresh()->load_state(good));
  auto broken = [&](auto mutate, const std::string& message) {
    nlohmann::json state = good;
    mutate(state);
    REQUIRE_THROWS_WITH(fresh()->load_state(state), ContainsSubstring(message));
  };
  broken([](nlohmann::json& state) { state.erase("rows"); }, "must hold a 'counts' object");
  broken([](nlohmann::json& state) { state["rows"] = -1; }, "must not be negative");
  broken([](nlohmann::json& state) { state["counts"]["c"] = 5; }, "must be an object");
  broken([](nlohmann::json& state) { state["counts"]["c"]["x"] = "two"; }, "must be a count");
  broken([](nlohmann::json& state) { state["counts"] = {{"d", {{"x", 2}}}}; }, "must be an object");
  broken([](nlohmann::json& state) { state["counts"].erase("c"); },
         "state 'columns' does not name the same columns as the fitted table");
  broken([](nlohmann::json& state) { state.erase("columns"); }, "state must hold a 'columns' list");
  // REGRESSION: load_state refuses a mangled null_counts or groups, which used
  // to be read as a legitimate default: a typo'd null count deleted the null
  // category, a negative one became a negative 'count', and a group pointing
  // at nothing turned a well-fitted category into an unknown.
  broken([](nlohmann::json& state) { state["null_counts"] = 3; }, "'null_counts' must be an object");
  broken([](nlohmann::json& state) { state["null_counts"].erase("c"); }, "null_counts has no entry for column 'c'");
  broken([](nlohmann::json& state) { state["null_counts"]["c"] = "1"; }, "null_counts['c'] must be a count or null");
  broken([](nlohmann::json& state) { state["null_counts"]["c"] = -1; }, "null_counts['c'] must be a count or null");
  broken([](nlohmann::json& state) { state["groups"] = {{"other", {{"a", "x"}}}}; },
         "groups names column 'other', which this encoder does not encode");
  broken([](nlohmann::json& state) { state["groups"] = {{"c", 5}}; }, "groups['c'] must be an object");
  broken([](nlohmann::json& state) { state["groups"] = {{"c", {{"a", "gone"}}}}; },
         "folds 'a' into 'gone', which is not a category");
  broken(
      [](nlohmann::json& state) {
        state["counts"]["c"]["g"] = 3;
        state["groups"] = {{"c", {{"x", "g"}}}};
      },
      "folds 'x', which is still a category");
  // The groups above are in the layout version 0.0.1 saved, a member and its
  // group's name. The present layout, a group's name and the list of its
  // members, is held to the same rules, and a member folds once.
  auto broken_groups = [&](const char* groups, const std::string& message) {
    broken(
        [&](nlohmann::json& state) {
          state["counts"]["c"]["g"] = 3;
          state["groups"] = nlohmann::json::parse(groups);
        },
        message);
  };
  broken_groups(R"({"other":{"x":["a"]}})", "groups names column 'other', which this encoder does not encode");
  broken_groups(R"({"c":["a"]})", "groups['c'] must be an object");
  for (const char* members : {"5", "null", "true", R"({"a":"x"})", "[]", R"(["a",7])", R"(["a",null])", R"([["a"]])"}) {
    CAPTURE(members);
    broken_groups((std::string(R"({"c":{"x":)") + members + "}}").c_str(),
                  "groups['c']['x'] must be a non-empty list of member names");
  }
  broken_groups(R"({"c":{"gone":["a"]}})", "folds 'a' into 'gone', which is not a category of counts['c']");
  broken_groups(R"({"c":{"g":["a","x"]}})", "folds 'x', which is still a category of counts['c']");
  broken_groups(R"({"c":{"x":["a","b","a"]}})", "groups['c'] folds 'a' more than once");
  broken_groups(R"({"c":{"g":["a","b"],"x":["b"]}})", "groups['c'] folds 'b' more than once");
  // One entry in each layout names 'a', so it would answer for two groups.
  broken_groups(R"({"c":{"a":"g","x":["a"]}})", "groups['c'] folds 'a' more than once");
  // A well-formed groups entry loads in either layout: 'a' folds into the
  // fitted 'x'.
  for (const char* groups : {R"({"c":{"a":"x"}})", R"({"c":{"x":["a"]}})"}) {
    CAPTURE(groups);
    nlohmann::json grouped = good;
    grouped["groups"] = nlohmann::json::parse(groups);
    auto loaded = fresh();
    REQUIRE_NOTHROW(loaded->load_state(grouped));
    REQUIRE(std::get<int64_t>(loaded->transform_row({{"c", std::string("a")}})[0].second) == 2);
    REQUIRE(loaded->state().at("groups") == nlohmann::json::parse(R"({"c":{"x":["a"]}})"));
  }
}

TEST_CASE("count_encode round-trips its rare groups, and loads several groups for one column") {
  std::vector<std::optional<std::string>> values(5, "keep");
  std::vector<std::string> members;
  for (int i = 0; i < 128; ++i) members.push_back("rare_" + std::to_string(i));
  std::sort(members.begin(), members.end());
  std::string group;
  for (const auto& member : members) {
    values.emplace_back(member);
    if (!group.empty()) group += '_';
    group += member;
  }
  for (bool normalize : {false, true}) {
    const nlohmann::json params{{"columns", {"c"}}, {"min_group_size", 3}, {"normalize", normalize}};
    auto fitted = fit_count(params, fft::Schema{{{"c", "utf8"}}}, {make_batch({"c"}, {utf8_column(values)})});
    const auto state = fitted->state();
    std::ostringstream streamed;
    fitted->write_state(streamed);
    REQUIRE(streamed.str() == state.dump());
    REQUIRE(state.at("counts").at("c") == nlohmann::json({{"keep", 5}, {group, 128}}));
    REQUIRE(state.at("groups") == nlohmann::json({{"c", {{group, members}}}}));
    auto loaded = fft::make_kernel("count_encode", params);
    loaded->load_state(state);
    REQUIRE(loaded->state() == state);
    const auto input = make_batch({"c"}, {utf8_column({members.front(), members.back(), group, "keep", "unseen"})});
    // A member answers its group's count, over the 133 fitted rows under
    // normalize.
    std::vector<fft::Row> expected;
    for (const int64_t count : {128, 128, 128, 5, 0}) {
      expected.push_back({{"c", normalize ? fft::RowValue(static_cast<double>(count) / 133.0) : fft::RowValue(count)}});
    }
    REQUIRE(both_paths(*fitted, input) == expected);
    REQUIRE(both_paths(*loaded, input) == expected);
  }

  // Several groups for one column load from either layout, the members in any
  // order, and are saved in the present layout with the members sorted. The
  // members of the two groups alternate, so in the layout of version 0.0.1
  // 'c' joins group_a after group_b was made: an entry joins the group its
  // name finds, which is not always the group made last.
  const nlohmann::json saved = nlohmann::json::parse(
      R"({"columns":["c"],"counts":{"c":{"group_a":3,"group_b":7,"keep":5}},)"
      R"("groups":{"c":{"group_a":["a","c"],"group_b":["b","d"]}},"null_counts":{"c":null},"rows":15})");
  for (const char* groups :
       {R"({"c":{"group_a":["a","c"],"group_b":["b","d"]}})", R"({"c":{"group_a":["c","a"],"group_b":["d","b"]}})",
        R"({"c":{"a":"group_a","b":"group_b","c":"group_a","d":"group_b"}})"}) {
    CAPTURE(groups);
    nlohmann::json state = saved;
    state["groups"] = nlohmann::json::parse(groups);
    auto loaded = fft::make_kernel("count_encode", {{"columns", {"c"}}});
    loaded->load_state(state);
    REQUIRE(loaded->state() == saved);
    std::ostringstream streamed;
    loaded->write_state(streamed);
    REQUIRE(streamed.str() == saved.dump());
    const auto result = loaded->transform(make_batch({"c"}, {utf8_column({"a", "b", "c", "d", "group_b", "keep"})}));
    REQUIRE(result.cols[0]->Equals(fft::testing::numeric_column<arrow::Int64Type>({3, 7, 3, 7, 7, 5})));
  }
}

// REGRESSION: the state holds a group's name once, with the list of its
// members, so it grows linearly with them. It used to hold the name once for
// each member, and the default name joins every member: 2,000 members of 8
// bytes each made a state of 36 MB.
TEST_CASE("count_encode saves a group's name once, so its state grows linearly with the members") {
  const auto state_size = [](int members) {
    std::vector<std::optional<std::string>> values(3, "keep");
    for (int i = 0; i < members; ++i) values.emplace_back("cat" + std::to_string(10000 + i));
    auto fitted = fit_count({{"columns", {"c"}}, {"min_group_size", 2}}, fft::Schema{{{"c", "utf8"}}},
                            {make_batch({"c"}, {utf8_column(values)})});
    std::ostringstream streamed;
    fitted->write_state(streamed);
    REQUIRE(streamed.str() == fitted->state().dump());
    return streamed.str().size();
  };
  // A member's 8 bytes are saved three times: twice with a '_' in the group's
  // name, which is a key of the counts and a key of the groups, and once
  // between quotes and before a comma in the list of members.
  const std::size_t each = 2 * (8 + 1) + (8 + 3);
  const std::size_t thousand = state_size(1000);
  REQUIRE(state_size(2000) == thousand + 1000 * each);
  REQUIRE(state_size(4000) == thousand + 3000 * each);
  REQUIRE(state_size(2000) < 60'000);
}

// A pipeline that version 0.0.1 saved keeps loading. That version saved a
// member and its group's name where a group's name and its members are saved
// now; each state below is what it saved for the fit beside it. Loaded, the
// state is the fitted kernel's: the same state in the present layout, and the
// same answers on the batch path and the record path.
TEST_CASE("count_encode loads the groups version 0.0.1 saved, however the fit folded them") {
  const fft::Schema schema{{{"c", "utf8"}, {"d", "utf8"}}};
  // With a threshold of 3 the one null of 'c' is below it and the three of
  // 'd' are not, so combine_min_nan_groups true, false and 'force' fold the
  // two columns in three different ways.
  const fft::Batch fit =
      make_batch({"c", "d"}, {utf8_column({"a", "b", "", "keep", "keep", "keep", "keep", std::nullopt, "keep"}),
                              utf8_column({"x", "y", std::nullopt, "z", "z", "z", std::nullopt, std::nullopt, "z"})});
  const fft::Batch probes =
      make_batch({"c", "d"}, {utf8_column({"a", "b", "", "keep", std::nullopt, "unseen", "_a_b", "rare"}),
                              utf8_column({"x", "y", "z", std::nullopt, "unseen", "x_y", "a", "rare"})});
  struct Case {
    nlohmann::json options;
    const char* saved;              // the state version 0.0.1 saved
    const char* groups;             // its groups in the present layout
    std::vector<int64_t> c_counts;  // the count each probe answers with
    std::vector<int64_t> d_counts;
  };
  const std::vector<Case> cases{
      {{{"min_group_size", 3}},
       R"({"columns":["c","d"],"counts":{"c":{"_a_b_nan":4,"keep":5},"d":{"x_y":2,"z":4}},)"
       R"("groups":{"c":{"":"_a_b_nan","a":"_a_b_nan","b":"_a_b_nan"},"d":{"x":"x_y","y":"x_y"}},)"
       R"("null_counts":{"c":4,"d":3},"rows":9})",
       R"({"c":{"_a_b_nan":["","a","b"]},"d":{"x_y":["x","y"]}})",
       {4, 4, 4, 5, 4, 0, 0, 0},
       {2, 2, 4, 3, 0, 2, 0, 0}},
      {{{"min_group_size", 3}, {"combine_min_nan_groups", false}},
       R"({"columns":["c","d"],"counts":{"c":{"_a_b":3,"keep":5},"d":{"x_y":2,"z":4}},)"
       R"("groups":{"c":{"":"_a_b","a":"_a_b","b":"_a_b"},"d":{"x":"x_y","y":"x_y"}},)"
       R"("null_counts":{"c":1,"d":3},"rows":9})",
       R"({"c":{"_a_b":["","a","b"]},"d":{"x_y":["x","y"]}})",
       {3, 3, 3, 5, 1, 0, 3, 0},
       {2, 2, 4, 3, 0, 2, 0, 0}},
      {{{"min_group_size", 3}, {"combine_min_nan_groups", "force"}},
       R"({"columns":["c","d"],"counts":{"c":{"_a_b_nan":4,"keep":5},"d":{"nan_x_y":5,"z":4}},)"
       R"("groups":{"c":{"":"_a_b_nan","a":"_a_b_nan","b":"_a_b_nan"},"d":{"x":"nan_x_y","y":"nan_x_y"}},)"
       R"("null_counts":{"c":4,"d":5},"rows":9})",
       R"({"c":{"_a_b_nan":["","a","b"]},"d":{"nan_x_y":["x","y"]}})",
       {4, 4, 4, 5, 4, 0, 0, 0},
       {5, 5, 4, 5, 0, 0, 0, 0}},
      {{{"min_group_size", 3}, {"min_group_name", "rare"}},
       R"({"columns":["c","d"],"counts":{"c":{"keep":5,"rare":4},"d":{"rare":2,"z":4}},)"
       R"("groups":{"c":{"":"rare","a":"rare","b":"rare"},"d":{"x":"rare","y":"rare"}},)"
       R"("null_counts":{"c":4,"d":3},"rows":9})",
       R"({"c":{"rare":["","a","b"]},"d":{"rare":["x","y"]}})",
       {4, 4, 4, 5, 4, 0, 0, 4},
       {2, 2, 4, 3, 0, 0, 0, 2}},
      {{{"min_group_size", 3}, {"normalize", true}},
       R"({"columns":["c","d"],"counts":{"c":{"_a_b_nan":4,"keep":5},"d":{"x_y":2,"z":4}},)"
       R"("groups":{"c":{"":"_a_b_nan","a":"_a_b_nan","b":"_a_b_nan"},"d":{"x":"x_y","y":"x_y"}},)"
       R"("null_counts":{"c":4,"d":3},"rows":9})",
       R"({"c":{"_a_b_nan":["","a","b"]},"d":{"x_y":["x","y"]}})",
       {4, 4, 4, 5, 4, 0, 0, 0},
       {2, 2, 4, 3, 0, 2, 0, 0}},
      // A group named like one of its members: 'a' is a member and the group.
      {{{"min_group_size", 3}, {"min_group_name", "a"}},
       R"({"columns":["c","d"],"counts":{"c":{"a":4,"keep":5},"d":{"a":2,"z":4}},)"
       R"("groups":{"c":{"":"a","a":"a","b":"a"},"d":{"x":"a","y":"a"}},)"
       R"("null_counts":{"c":4,"d":3},"rows":9})",
       R"({"c":{"a":["","a","b"]},"d":{"a":["x","y"]}})",
       {4, 4, 4, 5, 4, 0, 0, 0},
       {2, 2, 4, 3, 0, 0, 2, 0}},
      {{{"min_group_size", 0.3}, {"normalize", true}, {"combine_min_nan_groups", "force"}},
       R"({"columns":["c","d"],"counts":{"c":{"_a_b_nan":4,"keep":5},"d":{"nan_x_y":5,"z":4}},)"
       R"("groups":{"c":{"":"_a_b_nan","a":"_a_b_nan","b":"_a_b_nan"},"d":{"x":"nan_x_y","y":"nan_x_y"}},)"
       R"("null_counts":{"c":4,"d":5},"rows":9})",
       R"({"c":{"_a_b_nan":["","a","b"]},"d":{"nan_x_y":["x","y"]}})",
       {4, 4, 4, 5, 4, 0, 0, 0},
       {5, 5, 4, 5, 0, 0, 0, 0}},
  };
  for (const Case& item : cases) {
    CAPTURE(item.options);
    nlohmann::json params = item.options;
    params["columns"] = {"c", "d"};
    // normalize divides a count by the 9 fitted rows.
    const bool normalize = params.value("normalize", false);
    const auto answer = [&](int64_t count) {
      return normalize ? fft::RowValue(static_cast<double>(count) / 9.0) : fft::RowValue(count);
    };
    std::vector<fft::Row> expected;
    for (std::size_t i = 0; i < item.c_counts.size(); ++i) {
      expected.push_back({{"c", answer(item.c_counts[i])}, {"d", answer(item.d_counts[i])}});
    }
    const nlohmann::json older = nlohmann::json::parse(item.saved);
    const nlohmann::json present = [&] {
      nlohmann::json state = older;
      state["groups"] = nlohmann::json::parse(item.groups);
      return state;
    }();

    auto fitted = fit_count(params, schema, {fit});
    REQUIRE(fitted->state() == present);
    REQUIRE(both_paths(*fitted, probes) == expected);
    // The members are saved sorted however the rows were split.
    const auto slice = [&](int64_t offset, int64_t length) {
      return make_batch({"c", "d"}, {fit.cols[0]->Slice(offset, length), fit.cols[1]->Slice(offset, length)});
    };
    REQUIRE(fit_count(params, schema, {slice(4, 5), slice(0, 4)})->state() == present);
    for (const nlohmann::json* state : {&older, &present}) {
      auto loaded = fft::make_kernel("count_encode", params);
      loaded->stage = "counts";
      loaded->load_state(*state);
      REQUIRE(loaded->state() == present);
      REQUIRE(both_paths(*loaded, probes) == expected);
    }
  }
}

TEST_CASE("count_encode dictionary fits and transforms preserve counts policies and artifacts") {
  const auto first = fft::testing::dictionary_fixture<arrow::Int8Type>(
      {"a", "b", "a", std::nullopt, "unused"}, {0, 2, 1, std::nullopt, 3, 0, 1, 0, 2}, true, 3, 2);
  const auto second = fft::testing::dictionary_fixture<arrow::Int16Type>({"b", "a", "b", std::nullopt, "unused-new"},
                                                                         {0, 2, 1, 3, std::nullopt, 0, 1, 0, 2});
  for (bool normalize : {false, true}) {
    for (const std::string& missing : {"value", "return_nan"}) {
      const nlohmann::json params{
          {"columns", {"c"}}, {"normalize", normalize}, {"handle_missing", missing}, {"handle_unknown", 7}};
      auto fit = [&](bool dictionary) {
        auto kernel = fft::make_kernel("count_encode", params);
        fft::Schema schema{{{"c", "utf8"}}};
        kernel->resolve(schema);
        auto state = kernel->make_state();
        for (const auto* fixture : {&first, &second}) {
          kernel->consume(*state, make_batch({"c"}, {dictionary ? fixture->encoded : fixture->plain}));
        }
        std::vector<std::unique_ptr<fft::AccState>> states;
        states.push_back(std::move(state));
        kernel->finalize(std::move(states));
        return kernel;
      };
      auto dictionary = fit(true);
      auto plain = fit(false);
      REQUIRE(dictionary->state() == plain->state());
      for (const auto* fixture : {&first, &second, &first}) {
        REQUIRE(dictionary->transform(make_batch({"c"}, {fixture->encoded}))
                    .cols[0]
                    ->Equals(plain->transform(make_batch({"c"}, {fixture->plain})).cols[0]));
      }
      const auto unknown = fft::testing::dictionary_fixture({"new", std::nullopt, "unused"}, {0, 1, 0, 1});
      REQUIRE(dictionary->transform(make_batch({"c"}, {unknown.encoded}))
                  .cols[0]
                  ->Equals(plain->transform(make_batch({"c"}, {unknown.plain})).cols[0]));
    }
  }
}

TEST_CASE("count_encode counts the null rows of a batch that shares a larger dictionary") {
  // Ten entries, the first of them null, under five rows: the fit reads such
  // a batch row by row instead of tallying its dictionary. One row points at
  // the null entry and one is null itself, its index slot holding 0, the
  // null entry's index, so the column has two nulls.
  const auto fixture = fft::testing::dictionary_fixture({std::nullopt, "a", "b", "c", "d", "e", "f", "g", "h", "i"},
                                                        {1, std::nullopt, 0, 2, 1});
  const fft::Schema schema{{{"c", "utf8"}}};
  const auto dictionary = fit_count({{"columns", {"c"}}}, schema, {make_batch({"c"}, {fixture.encoded})});
  const auto plain = fit_count({{"columns", {"c"}}}, schema, {make_batch({"c"}, {fixture.plain})});
  REQUIRE(dictionary->state() == plain->state());
  REQUIRE(dictionary->state().at("counts").at("c") == nlohmann::json({{"a", 2}, {"b", 1}}));
  REQUIRE(dictionary->state().at("null_counts").at("c") == 2);
}

TEST_CASE("count_encode dictionary fits keep exact counts through map growth and revisits") {
  constexpr int64_t categories = 1024;
  std::vector<std::optional<std::string>> words;
  for (int64_t i = 0; i < categories; ++i) words.emplace_back("category-" + std::to_string(i));
  words.emplace_back(std::nullopt);
  words.push_back(words.front());  // a second dictionary entry for the first category
  words.emplace_back("unused");
  std::vector<std::optional<int64_t>> indices{0};
  for (int64_t i = 1; i < categories; ++i) {
    indices.insert(indices.end(), {i, 0, i / 2});
    if (i % 17 == 0) indices.emplace_back(categories);
    if (i % 19 == 0) indices.emplace_back(std::nullopt);
  }
  indices.emplace_back(categories + 1);
  for (int64_t i = categories; i > 0; --i) indices.emplace_back(i - 1);
  const auto fixture = fft::testing::dictionary_fixture<arrow::Int16Type>(words, indices, true, 3, 2);
  for (bool normalize : {false, true}) {
    CAPTURE(normalize);
    auto fit = [&](const std::shared_ptr<arrow::Array>& input) {
      auto kernel = fft::make_kernel("count_encode", {{"columns", {"c"}}, {"normalize", normalize}});
      fft::Schema schema{{{"c", "utf8"}}};
      kernel->resolve(schema);
      auto state = kernel->make_state();
      // Seed a different dense order before resolving dictionary entries.
      kernel->consume(*state, make_batch({"c"}, {utf8_column({"before-fit", "category-0", std::nullopt})}));
      kernel->consume(*state, make_batch({"c"}, {input}));
      std::vector<std::unique_ptr<fft::AccState>> states;
      states.push_back(std::move(state));
      kernel->finalize(std::move(states));
      return kernel;
    };
    auto dictionary = fit(fixture.encoded);
    auto plain = fit(fixture.plain);
    const auto state = dictionary->state();
    REQUIRE(state.dump() == plain->state().dump());
    REQUIRE(state.at("counts").at("c").size() == static_cast<std::size_t>(categories + 1));
    REQUIRE(state.at("counts").at("c").at("category-0") == categories + 4);
    REQUIRE_FALSE(state.at("counts").at("c").contains("unused"));
    REQUIRE(state.at("null_counts").at("c") == 1 + (categories - 1) / 17 + (categories - 1) / 19);
    const auto actual = dictionary->transform(make_batch({"c"}, {fixture.encoded}));
    REQUIRE(actual.cols[0]->ValidateFull().ok());
    REQUIRE(actual.cols[0]->Equals(plain->transform(make_batch({"c"}, {fixture.plain})).cols[0]));
  }
}

TEST_CASE("count_encode streams its exact state with escaped keys, groups and column order") {
  const std::string first = "z\"\\";
  const std::string second = "a\t雪";
  const std::string group_a = "group\"\\\n雪";
  const std::string group_b = "group\t🙂";
  const std::string member_with_nul("nul\0key", 7);
  const nlohmann::json saved{{"columns", {first, second}},
                             {"rows", 31},
                             {"counts", {{first, {{group_a, 11}, {group_b, 15}, {"keep", 5}}}, {second, {{"x", 31}}}}},
                             {"null_counts", {{first, 0}, {second, nullptr}}},
                             {"groups",
                              {{first, {{"", group_a}, {member_with_nul, group_a}, {"quote\"\\雪", group_b}}},
                               {second, nlohmann::json::object()}}}};
  auto loaded = fft::make_kernel("count_encode", {{"columns", {first, second}}});
  loaded->load_state(saved);
  // An explicitly empty groups column is left out of state().
  REQUIRE(loaded->state().at("groups") == nlohmann::json({{first,
                                                           {{group_a, nlohmann::json::array({"", member_with_nul})},
                                                            {group_b, nlohmann::json::array({"quote\"\\雪"})}}}}));
  std::ostringstream stream;
  stream << std::hex << std::showbase;
  loaded->write_state(stream);
  REQUIRE(stream.str() == loaded->state().dump());
  REQUIRE(nlohmann::json::parse(stream.str()) == loaded->state());

  auto fitted =
      fit_count({{"columns", {first, second}}, {"min_group_size", 3}, {"min_group_name", group_a}},
                fft::Schema{{{first, "utf8"}, {second, "utf8"}}},
                {make_batch({first, second}, {utf8_column({"keep", "keep", "keep", "keep", "", member_with_nul, "雪"}),
                                              utf8_column({"x", "x", "x", "x", "x", "x", std::nullopt})})});
  std::ostringstream fitted_stream;
  fitted->write_state(fitted_stream);
  REQUIRE(fitted_stream.str() == fitted->state().dump());
}
