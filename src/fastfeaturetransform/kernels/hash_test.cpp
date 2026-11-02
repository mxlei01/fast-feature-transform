// Explicit C++ tests for hash_encode: the params, the schema rewrite, the
// md5 buckets themselves (against hashlib's digests), the split-independence
// a stateless kernel gets for free, and load_state.

#include <arrow/array.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "fastfeaturetransform/kernel.hpp"
#include "fastfeaturetransform/kernels/dictionary_test_support.hpp"
#include "fastfeaturetransform/test_support.hpp"

namespace fft = fastfeaturetransform;
using Catch::Matchers::ContainsSubstring;
using fft::testing::make_batch;
using fft::testing::utf8_column;

namespace {

// A resolved hash_encode; the schema defaults to one utf8 column 'c'.
std::unique_ptr<fft::Kernel> hasher(const nlohmann::json& params, fft::Schema schema = fft::Schema{{{"c", "utf8"}}}) {
  auto kernel = fft::make_kernel("hash_encode", params);
  kernel->stage = "hash";
  kernel->resolve(schema);
  return kernel;
}

// The counters of one transformed batch, bucket by bucket.
std::vector<std::vector<int64_t>> counters(const fft::Batch& batch, std::size_t components) {
  std::vector<std::vector<int64_t>> out;
  for (std::size_t i = 0; i < components; ++i) {
    const int idx = batch.find("col_" + std::to_string(i));
    REQUIRE(idx >= 0);
    const auto& column = static_cast<const arrow::Int64Array&>(*batch.cols[static_cast<std::size_t>(idx)]);
    std::vector<int64_t> values;
    for (int64_t row = 0; row < column.length(); ++row) values.push_back(column.Value(row));
    out.push_back(std::move(values));
  }
  return out;
}

}  // namespace

TEST_CASE("hash_encode validates its params") {
  REQUIRE_THROWS_WITH((fft::make_kernel("hash_encode", nlohmann::json::object())),
                      ContainsSubstring("'columns' must be a non-empty list"));
  REQUIRE_THROWS_WITH((fft::make_kernel("hash_encode", {{"columns", {"c"}}, {"n_components", 0}})),
                      ContainsSubstring("'n_components' must be an integer between 1 and 4096"));
  REQUIRE_THROWS_WITH((fft::make_kernel("hash_encode", {{"columns", {"c"}}, {"n_components", 4097}})),
                      ContainsSubstring("'n_components' must be an integer between 1 and 4096"));
  REQUIRE_THROWS_WITH((fft::make_kernel("hash_encode", {{"columns", {"c"}}, {"n_components", 1.5}})),
                      ContainsSubstring("'n_components' must be an integer between 1 and 4096"));
  REQUIRE_THROWS_WITH((fft::make_kernel("hash_encode", {{"columns", {"c"}}, {"hash_method", "sha256"}})),
                      ContainsSubstring("'hash_method' must be 'md5'"));
  REQUIRE_THROWS_WITH((fft::make_kernel("hash_encode", {{"columns", {"c", "c"}}})),
                      ContainsSubstring("listed more than once"));
  // Nothing is learned, so no pass ever has to run for it.
  REQUIRE_FALSE(fft::make_kernel("hash_encode", {{"columns", {"c"}}})->needs_fit());
  REQUIRE(fft::make_kernel("hash_encode", {{"columns", {"c"}}})->state().empty());
  REQUIRE(fft::make_kernel("hash_encode", {{"columns", {"c"}}})->params().at("n_components") == 8);
}

TEST_CASE("hash_encode replaces the encoded columns with one block and refuses a collision") {
  auto kernel = fft::make_kernel("hash_encode", {{"columns", {"c", "d"}}, {"n_components", 3}});
  kernel->stage = "hash";
  fft::Schema schema{{{"c", "utf8"}, {"n", "i64"}, {"d", "utf8"}}};
  kernel->resolve(schema);
  REQUIRE(schema.cols == fft::Schema{{{"n", "i64"}, {"col_0", "i64"}, {"col_1", "i64"}, {"col_2", "i64"}}}.cols);
  REQUIRE(kernel->outputs == std::vector<std::string>{"col_0", "col_1", "col_2"});

  auto numeric = fft::make_kernel("hash_encode", {{"columns", {"n"}}});
  numeric->stage = "hash";
  fft::Schema numbers{{{"n", "i64"}}};
  REQUIRE_THROWS_WITH(numeric->resolve(numbers), ContainsSubstring("is i64, expected utf8 (cast it to utf8 first)"));

  auto missing = fft::make_kernel("hash_encode", {{"columns", {"gone"}}});
  missing->stage = "hash";
  fft::Schema absent{{{"c", "utf8"}}};
  REQUIRE_THROWS_WITH(missing->resolve(absent), ContainsSubstring("column 'gone' not found in input"));

  // A second hash encoder in the same pipeline would overwrite the first.
  auto second = fft::make_kernel("hash_encode", {{"columns", {"e"}}, {"n_components", 3}});
  second->stage = "hash2";
  fft::Schema taken = schema;
  taken.cols.emplace_back("e", "utf8");
  REQUIRE_THROWS_WITH(second->resolve(taken), ContainsSubstring("two hash encoders cannot share one pipeline"));
}

TEST_CASE("hash_encode counts md5 buckets and both paths agree") {
  // The expected buckets come straight from hashlib: md5(b'amber') is
  // 59548977279905234b7ed3b1710837f2, and that digest read big-endian is
  // 9 mod 13. 'blue' lands in the same bucket, 'cyan' in 3, and the empty
  // string (d41d8cd9...427e) in 6.
  auto kernel = hasher({{"columns", {"c"}}, {"n_components", 13}});
  const std::vector<std::optional<std::string>> values{"amber", "blue", "cyan", std::nullopt, ""};
  fft::Batch out = kernel->transform(make_batch({"c"}, {utf8_column(values)}));
  REQUIRE(out.find("c") < 0);  // the encoded column is gone
  const std::vector<std::vector<int64_t>> block = counters(out, 13);
  REQUIRE(block[9][0] == 1);  // 'amber'
  REQUIRE(block[9][1] == 1);  // 'blue', which collides with 'amber' at 13 buckets
  REQUIRE(block[3][2] == 1);  // 'cyan'
  REQUIRE(block[6][4] == 1);  // the empty string still hashes
  for (std::size_t bucket = 0; bucket < 13; ++bucket) REQUIRE(block[bucket][3] == 0);  // a null counts nowhere

  // Every row sums to its number of non-null encoded values, and the record
  // path answers exactly what the batch path did.
  for (std::size_t row = 0; row < values.size(); ++row) {
    int64_t total = 0;
    fft::RowValue value = values[row].has_value() ? fft::RowValue(*values[row]) : fft::RowValue(std::monostate{});
    fft::Row record = kernel->transform_row({{"c", value}});
    REQUIRE(record.size() == 13);
    for (std::size_t bucket = 0; bucket < 13; ++bucket) {
      REQUIRE(record[bucket].first == "col_" + std::to_string(bucket));
      REQUIRE(std::get<int64_t>(record[bucket].second) == block[bucket][row]);
      total += block[bucket][row];
    }
    REQUIRE(total == (values[row].has_value() ? 1 : 0));
  }
  REQUIRE_THROWS_WITH(kernel->transform_row({{"c", int64_t{1}}}), ContainsSubstring("must be a string or null"));
  REQUIRE_THROWS_WITH(kernel->transform_row({{"n", int64_t{1}}}), ContainsSubstring("missing from record"));
  REQUIRE_THROWS_WITH(kernel->transform_row({{"c", std::string("amber")}, {"col_0", int64_t{7}}}),
                      ContainsSubstring("key 'col_0' already present in record"));
}

TEST_CASE("hash_encode adds every encoded column into the one shared block") {
  auto kernel = hasher({{"columns", {"c", "d"}}, {"n_components", 4}}, fft::Schema{{{"c", "utf8"}, {"d", "utf8"}}});
  fft::Batch out = kernel->transform(
      make_batch({"c", "d"}, {utf8_column({"amber", "amber"}), utf8_column({"amber", std::nullopt})}));
  const std::vector<std::vector<int64_t>> block = counters(out, 4);
  int64_t first = 0;
  int64_t second = 0;
  for (std::size_t bucket = 0; bucket < 4; ++bucket) {
    first += block[bucket][0];
    second += block[bucket][1];
  }
  REQUIRE(first == 2);   // both values hashed, into the same bucket
  REQUIRE(second == 1);  // the null contributed nothing
}

TEST_CASE("hash_encode gives the same rows however the batch was split") {
  auto kernel = hasher({{"columns", {"c"}}, {"n_components", 5}});
  const std::vector<std::optional<std::string>> values{"a", "bb", "ccc", "dddd", "eeeee", "ffffff"};
  const std::vector<std::vector<int64_t>> whole =
      counters(kernel->transform(make_batch({"c"}, {utf8_column(values)})), 5);
  const std::vector<std::optional<std::string>> head{values.begin(), values.begin() + 2};
  const std::vector<std::optional<std::string>> tail{values.begin() + 2, values.end()};
  const std::vector<std::vector<int64_t>> a = counters(kernel->transform(make_batch({"c"}, {utf8_column(head)})), 5);
  const std::vector<std::vector<int64_t>> b = counters(kernel->transform(make_batch({"c"}, {utf8_column(tail)})), 5);
  for (std::size_t bucket = 0; bucket < 5; ++bucket) {
    std::vector<int64_t> joined = a[bucket];
    joined.insert(joined.end(), b[bucket].begin(), b[bucket].end());
    REQUIRE(joined == whole[bucket]);
  }
}

TEST_CASE("hash_encode remembered plain values agree with the record path, past the table limit too") {
  // A plain column hashes each distinct value once and answers its repeats
  // from a table of at most one entry per eight rows, so the distinct column
  // outgrows it partway through and goes back to hashing every value. The
  // record path remembers nothing, so it is the reference for every row.
  auto kernel = hasher({{"columns", {"c"}}, {"n_components", 13}});
  std::vector<std::optional<std::string>> repeated;
  std::vector<std::optional<std::string>> distinct;
  for (int i = 0; i < 4000; ++i) {
    const bool null = i % 11 == 0;
    repeated.push_back(null ? std::nullopt : std::optional<std::string>("v" + std::to_string(i % 37)));
    distinct.push_back(null ? std::nullopt : std::optional<std::string>("d" + std::to_string(i)));
  }
  for (const auto* values : {&repeated, &distinct}) {
    for (bool large : {false, true}) {
      for (std::size_t offset : {0U, 3U}) {
        CAPTURE(values == &distinct, large, offset);
        const auto column = large ? fft::testing::large_utf8_column(*values) : utf8_column(*values);
        const std::vector<std::vector<int64_t>> block =
            counters(kernel->transform(make_batch({"c"}, {column->Slice(static_cast<int64_t>(offset))})), 13);
        for (std::size_t row = 0; row + offset < values->size(); ++row) {
          const std::optional<std::string>& value = (*values)[row + offset];
          const fft::Row record =
              kernel->transform_row({{"c", value ? fft::RowValue(*value) : fft::RowValue(std::monostate{})}});
          std::vector<int64_t> expected;
          std::vector<int64_t> actual;
          for (std::size_t bucket = 0; bucket < 13; ++bucket) {
            expected.push_back(std::get<int64_t>(record[bucket].second));
            actual.push_back(block[bucket][row]);
          }
          REQUIRE(actual == expected);
        }
      }
    }
  }
}

TEST_CASE("hash_encode load_state rejects a table a hand-edited manifest put there") {
  auto fresh = [] {
    auto kernel = fft::make_kernel("hash_encode", {{"columns", {"c"}}});
    kernel->stage = "hash";
    return kernel;
  };
  REQUIRE_NOTHROW(fresh()->load_state(nlohmann::json::object()));
  REQUIRE_THROWS_WITH(fresh()->load_state({{"counts", {{"c", 1}}}}),
                      ContainsSubstring("hash_encode learns nothing, so its 'state' must be an empty object"));
  REQUIRE_THROWS_WITH(fresh()->load_state(nlohmann::json::array()), ContainsSubstring("must be an empty object"));
}

namespace {

template <typename IndexType>
void check_hash_dictionaries(bool large, std::size_t components, int64_t offset) {
  const std::vector<std::optional<int64_t>> indices{2, 1, std::nullopt, 4, 3, 5, 0, 2, 1, 5, 3, 0, 4, std::nullopt, 0};
  const auto original = fft::testing::dictionary_fixture<IndexType>(
      {"amber", "blue", "amber", "", std::nullopt, "cyan", "unused", "also-unused"}, indices, large, offset, offset);
  const auto changed = fft::testing::dictionary_fixture<IndexType>(
      {"cyan", "amber", "cyan", "", std::nullopt, "blue", "unused-new", "also-unused-new"}, indices, large, offset,
      offset);
  auto kernel =
      hasher({{"columns", {"c", "d"}}, {"n_components", components}}, fft::Schema{{{"c", "utf8"}, {"d", "utf8"}}});
  for (const auto* first : {&original, &changed, &original}) {
    const auto before = first->encoded->ToString();
    const auto expected = kernel->transform(make_batch({"c", "d"}, {first->plain, changed.plain}));
    const auto actual = kernel->transform(make_batch({"c", "d"}, {first->encoded, changed.encoded}));
    REQUIRE(actual.names == expected.names);
    REQUIRE(actual.rows == expected.rows);
    for (std::size_t bucket = 0; bucket < components; ++bucket) {
      REQUIRE(actual.cols[bucket]->ValidateFull().ok());
      REQUIRE(actual.cols[bucket]->Equals(expected.cols[bucket]));
      REQUIRE(actual.cols[bucket]->null_count() == 0);
    }
    for (std::size_t row = 0; row < first->decoded.size(); ++row) {
      const auto value = [](const std::optional<std::string>& text) -> fft::RowValue {
        return text.has_value() ? fft::RowValue{*text} : fft::RowValue{std::monostate{}};
      };
      const auto record =
          kernel->transform_row({{"c", value(first->decoded[row])}, {"d", value(changed.decoded[row])}});
      int64_t sum = 0;
      for (std::size_t bucket = 0; bucket < components; ++bucket) {
        const auto& array = static_cast<const arrow::Int64Array&>(*actual.cols[bucket]);
        REQUIRE(std::get<int64_t>(record[bucket].second) == array.Value(static_cast<int64_t>(row)));
        sum += array.Value(static_cast<int64_t>(row));
      }
      REQUIRE(sum == static_cast<int64_t>(first->decoded[row].has_value() + changed.decoded[row].has_value()));
    }
    REQUIRE(first->encoded->ToString() == before);
  }
}

}  // namespace

TEST_CASE("hash_encode dictionary caches preserve index widths slices nulls and per-column scope") {
  for (bool large : {false, true}) {
    for (std::size_t components : {1U, 8U, 13U}) {
      for (int64_t offset : {0, 3}) {
        CAPTURE(large, components, offset);
        check_hash_dictionaries<arrow::Int8Type>(large, components, offset);
        check_hash_dictionaries<arrow::Int16Type>(large, components, offset);
        check_hash_dictionaries<arrow::Int32Type>(large, components, offset);
        check_hash_dictionaries<arrow::Int64Type>(large, components, offset);
      }
    }
  }
}

TEST_CASE("hash_encode dictionary fallback and empty arrays retain validation") {
  auto kernel = hasher({{"columns", {"c"}}, {"n_components", 13}});
  for (bool large : {false, true}) {
    std::vector<std::optional<std::string>> words(128, "unused");
    words[5] = "amber";
    words[17] = std::nullopt;
    const auto oversized = fft::testing::dictionary_fixture(words, {5, 17, std::nullopt, 5}, large, 2, 1);
    const auto actual = kernel->transform(make_batch({"c"}, {oversized.encoded}));
    const auto expected = kernel->transform(make_batch({"c"}, {oversized.plain}));
    for (std::size_t i = 0; i < actual.cols.size(); ++i) REQUIRE(actual.cols[i]->Equals(expected.cols[i]));
    for (const std::vector<std::optional<int64_t>>& rows :
         {std::vector<std::optional<int64_t>>{}, std::vector<std::optional<int64_t>>{std::nullopt, std::nullopt}}) {
      const auto empty = fft::testing::dictionary_fixture({}, rows, large);
      const auto output = kernel->transform(make_batch({"c"}, {empty.encoded}));
      REQUIRE(output.rows == static_cast<int64_t>(rows.size()));
      for (const auto& column : output.cols) REQUIRE(column->ValidateFull().ok());
      const auto wrong =
          arrow::DictionaryArray::FromArrays(fft::testing::numeric_column<arrow::Int32Type>(
                                                 std::vector<std::optional<int32_t>>(rows.size(), std::nullopt)),
                                             fft::testing::numeric_column<arrow::Int64Type>({1}));
      REQUIRE(wrong.ok());
      REQUIRE_THROWS_WITH(kernel->transform(make_batch({"c"}, {*wrong})),
                          ContainsSubstring("dictionary is int64, expected utf8"));
    }
  }
}

TEST_CASE("hash_encode power-of-two buckets use the low bits of the big-endian digest") {
  // hashlib.md5(b'amber') = 59548977279905234b7ed3b1710837f2;
  // hashlib.md5(b'')      = d41d8cd98f00b204e9800998ecf8427e.
  for (std::size_t components : {1U, 8U, 256U, 4096U}) {
    auto kernel = hasher({{"columns", {"c"}}, {"n_components", components}});
    for (const auto& [text, tail] : std::vector<std::pair<std::string, std::size_t>>{{"amber", 0x37f2}, {"", 0x427e}}) {
      const auto row = kernel->transform_row({{"c", text}});
      for (std::size_t bucket = 0; bucket < components; ++bucket) {
        REQUIRE(std::get<int64_t>(row[bucket].second) == (bucket == tail % components ? 1 : 0));
      }
    }
  }
}
