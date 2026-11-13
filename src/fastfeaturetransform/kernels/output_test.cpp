// Explicit C++ tests for the lookups and outputs the categorical encoders
// share: dictionary input of every index width, sliced input, the perfect hash
// over short categories and its fallbacks, and the nulls that handle_unknown
// and handle_missing produce.

#include <arrow/array.h>

#include <algorithm>
#include <bit>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <random>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "fastfeaturetransform/kernel.hpp"
#include "fastfeaturetransform/kernels/categories.hpp"
#include "fastfeaturetransform/kernels/dictionary_test_support.hpp"
#include "fastfeaturetransform/kernels/util.hpp"
#include "fastfeaturetransform/test_support.hpp"

namespace fft = fastfeaturetransform;
using Catch::Matchers::ContainsSubstring;

namespace {
// label_encode takes neither option: it encodes a null and an unknown category
// as 0 under every policy.
std::unique_ptr<fft::Kernel> fitted_encoder(const std::string& id, const std::string& policy) {
  nlohmann::json params = {{"columns", {"c"}}};
  if (id != "label_encode") {
    params["handle_missing"] = policy;
    params["handle_unknown"] = policy;
  }
  auto kernel = fft::make_kernel(id, params);
  kernel->stage = "enc";
  fft::Schema schema{{{"c", "utf8"}}};
  kernel->resolve(schema);
  auto state = kernel->make_state();
  kernel->consume(*state, fft::testing::make_batch({"c"}, {fft::testing::utf8_column({"c", "a", "b"})}));
  std::vector<std::unique_ptr<fft::AccState>> states;
  states.push_back(std::move(state));
  kernel->finalize(std::move(states));
  return kernel;
}

void require_scalar_values(const fft::Kernel& kernel, const fft::Batch& batch,
                           const std::vector<std::optional<std::string>>& values) {
  for (const auto& column : batch.cols) REQUIRE(column->ValidateFull().ok());
  for (std::size_t row = 0; row < values.size(); ++row) {
    fft::RowValue value = values[row] ? fft::RowValue(*values[row]) : fft::RowValue(std::monostate{});
    const fft::Row scalar = kernel.transform_row({{"c", std::move(value)}});
    REQUIRE(scalar.size() == batch.cols.size());
    for (std::size_t col = 0; col < scalar.size(); ++col) {
      REQUIRE(scalar[col].first == batch.names[col]);
      const auto index = static_cast<int64_t>(row);
      REQUIRE(batch.cols[col]->IsNull(index) == std::holds_alternative<std::monostate>(scalar[col].second));
      if (batch.cols[col]->IsNull(index)) continue;
      if (batch.cols[col]->type_id() == arrow::Type::INT64) {
        REQUIRE(static_cast<const arrow::Int64Array&>(*batch.cols[col]).Value(index) ==
                std::get<int64_t>(scalar[col].second));
      } else {
        REQUIRE(std::bit_cast<uint64_t>(static_cast<const arrow::DoubleArray&>(*batch.cols[col]).Value(index)) ==
                std::bit_cast<uint64_t>(std::get<double>(scalar[col].second)));
      }
    }
  }
}
}  // namespace

TEST_CASE("categorical direct outputs preserve sliced dictionaries and scalar values") {
  const std::vector<std::string> ids{
      "label_encode",   "ordinal_encode", "one_hot_encode",   "rank_hot_encode",
      "basen_encode",   "binary_encode",  "gray_encode",      "backward_difference_encode",
      "helmert_encode", "sum_encode",     "polynomial_encode"};
  for (const auto& id : ids) {
    for (const std::string policy : {"value", "return_nan"}) {
      CAPTURE(id, policy);
      auto kernel = fitted_encoder(id, policy);
      const auto original_state = kernel->state();
      for (const bool large : {false, true}) {
        for (const int64_t length : {0, 1, 7, 8, 9, 63, 64, 65, 257}) {
          CAPTURE(large, length);
          // Duplicate and unused entries, a null dictionary entry, a null
          // index, and nonzero offsets in both the indices and dictionary.
          const std::vector<std::optional<std::string>> entries{"padding", "b",      "a",      std::nullopt,
                                                                "b",       "unseen", "unused", "padding"};
          auto values_array = large ? fft::testing::large_utf8_column(entries) : fft::testing::utf8_column(entries);
          auto dictionary = values_array->Slice(1, 6);
          std::vector<std::optional<int32_t>> indices{0, 0, 0};
          std::vector<std::optional<std::string>> logical;
          for (int64_t row = 0; row < length; ++row) {
            const int32_t index = static_cast<int32_t>(row % 6);
            indices.push_back(index == 5 ? std::nullopt : std::optional<int32_t>(index));
            logical.push_back(index == 5 ? std::nullopt : entries[static_cast<std::size_t>(index) + 1]);
          }
          indices.push_back(0);
          auto index_array = fft::testing::numeric_column<arrow::Int32Type>(indices);
          auto encoded = arrow::DictionaryArray::FromArrays(index_array->Slice(3, length), dictionary);
          REQUIRE(encoded.ok());
          const fft::Batch actual = kernel->transform(fft::testing::make_batch({"c"}, {*encoded}));
          const fft::Batch expected =
              kernel->transform(fft::testing::make_batch({"c"}, {fft::testing::utf8_column(logical)}));
          REQUIRE(actual.names == expected.names);
          REQUIRE(actual.rows == length);
          for (std::size_t col = 0; col < actual.cols.size(); ++col)
            REQUIRE(actual.cols[col]->Equals(expected.cols[col]));
          require_scalar_values(*kernel, actual, logical);
          REQUIRE(index_array->Equals(fft::testing::numeric_column<arrow::Int32Type>(indices)));
          REQUIRE(values_array->Equals(large ? fft::testing::large_utf8_column(entries)
                                             : fft::testing::utf8_column(entries)));
          REQUIRE(kernel->state() == original_state);
        }
      }
    }
  }
}

namespace {
template <typename IndexType>
void require_index_width_matches_plain(const fft::Kernel& kernel) {
  const std::vector<std::optional<std::string>> entries{"b", "a", std::nullopt, "b", "unseen", "unused"};
  std::vector<std::optional<int64_t>> rows{0, 0};  // sliced off below
  std::vector<std::optional<std::string>> logical;
  for (int64_t row = 0; row < 70; ++row) {
    // The first row reads entry 1 and the last entry 4: an index the copy
    // skipped would read 0, whose "b" also sits at entry 3.
    const int64_t index = (row + 1) % 6;
    const bool null = index == 5 || row % 11 == 10;
    rows.push_back(null ? std::nullopt : std::optional<int64_t>(index));
    logical.push_back(null ? std::nullopt : entries[static_cast<std::size_t>(index)]);
  }
  auto encoded = arrow::DictionaryArray::FromArrays(fft::testing::indices_with_garbage<IndexType>(rows)->Slice(2),
                                                    fft::testing::utf8_column(entries));
  REQUIRE(encoded.ok());
  const fft::Batch actual = kernel.transform(fft::testing::make_batch({"c"}, {*encoded}));
  const fft::Batch expected = kernel.transform(fft::testing::make_batch({"c"}, {fft::testing::utf8_column(logical)}));
  REQUIRE(actual.names == expected.names);
  for (std::size_t col = 0; col < actual.cols.size(); ++col) REQUIRE(actual.cols[col]->Equals(expected.cols[col]));
}

// Every entry of a dictionary wider than the signed range of its unsigned
// index type, each used once in a scattered order, against an encoder fitted
// on all of them, so that each index has an output of its own.
template <typename IndexType>
void require_unsigned_indices_match_plain(int64_t entries) {
  std::vector<std::optional<std::string>> values;
  for (int64_t i = 0; i < entries; ++i) values.emplace_back("k" + std::to_string(i));
  auto kernel = fft::make_kernel("ordinal_encode", {{"columns", {"c"}}});
  kernel->stage = "enc";
  fft::Schema schema{{{"c", "utf8"}}};
  kernel->resolve(schema);
  auto state = kernel->make_state();
  kernel->consume(*state, fft::testing::make_batch({"c"}, {fft::testing::utf8_column(values)}));
  std::vector<std::unique_ptr<fft::AccState>> states;
  states.push_back(std::move(state));
  kernel->finalize(std::move(states));
  std::vector<std::optional<int64_t>> rows;
  std::vector<std::optional<std::string>> logical;
  for (int64_t row = 0; row < entries; ++row) {
    const int64_t index = (row * 7919) % entries;  // 7919 is prime, so every entry comes up once
    rows.emplace_back(index);
    logical.push_back(values[static_cast<std::size_t>(index)]);
  }
  auto encoded = arrow::DictionaryArray::FromArrays(fft::testing::indices_with_garbage<IndexType>(rows),
                                                    fft::testing::utf8_column(values));
  REQUIRE(encoded.ok());
  const fft::Batch actual = kernel->transform(fft::testing::make_batch({"c"}, {*encoded}));
  const fft::Batch expected = kernel->transform(fft::testing::make_batch({"c"}, {fft::testing::utf8_column(logical)}));
  REQUIRE(actual.names == expected.names);
  for (std::size_t col = 0; col < actual.cols.size(); ++col) REQUIRE(actual.cols[col]->Equals(expected.cols[col]));
}
}  // namespace

// The dictionary loop reads int32 indices in place and copies any other width
// once; every width, sliced and with garbage under its nulls, must give what
// the same strings give unencoded, for every encoder that fits without a
// target and shares the loop.
TEST_CASE("categorical dictionary indices of every width agree with plain strings") {
  for (const std::string id :
       {"ordinal_encode", "one_hot_encode", "rank_hot_encode", "basen_encode", "binary_encode", "gray_encode",
        "backward_difference_encode", "helmert_encode", "sum_encode", "polynomial_encode", "count_encode"}) {
    for (const std::string policy : {"value", "return_nan"}) {
      CAPTURE(id, policy);
      const auto kernel = fitted_encoder(id, policy);
      require_index_width_matches_plain<arrow::Int8Type>(*kernel);
      require_index_width_matches_plain<arrow::UInt8Type>(*kernel);
      require_index_width_matches_plain<arrow::Int16Type>(*kernel);
      require_index_width_matches_plain<arrow::UInt16Type>(*kernel);
      require_index_width_matches_plain<arrow::Int32Type>(*kernel);
      require_index_width_matches_plain<arrow::UInt32Type>(*kernel);
      require_index_width_matches_plain<arrow::Int64Type>(*kernel);
      require_index_width_matches_plain<arrow::UInt64Type>(*kernel);
    }
  }
}

TEST_CASE("categorical dictionary unsigned indices past the signed range agree with plain strings") {
  require_unsigned_indices_match_plain<arrow::UInt8Type>(256);
  require_unsigned_indices_match_plain<arrow::UInt16Type>(40000);
}

TEST_CASE("categorical dictionary caches ignore unused entries and expire after each call") {
  for (const std::string id :
       {"ordinal_encode", "one_hot_encode", "rank_hot_encode", "basen_encode", "binary_encode", "gray_encode",
        "backward_difference_encode", "helmert_encode", "sum_encode", "polynomial_encode"}) {
    CAPTURE(id);
    auto kernel = fitted_encoder(id, "error");
    auto indices = fft::testing::numeric_column<arrow::Int8Type>({0, 0, 0, 0, 0, 0});
    for (const std::string category : {"a", "b", "c"}) {
      auto dictionary = fft::testing::utf8_column({category, "unseen", std::nullopt});
      auto array = arrow::DictionaryArray::FromArrays(indices, dictionary);
      REQUIRE(array.ok());
      const auto batch = kernel->transform(fft::testing::make_batch({"c"}, {*array}));
      require_scalar_values(*kernel, batch, std::vector<std::optional<std::string>>(6, category));
    }
    auto dictionary = fft::testing::utf8_column({"a", "unseen", std::nullopt});
    auto bad = arrow::DictionaryArray::FromArrays(fft::testing::numeric_column<arrow::Int8Type>({0, 1, 2}), dictionary);
    REQUIRE(bad.ok());
    REQUIRE_THROWS_WITH(kernel->transform(fft::testing::make_batch({"c"}, {*bad})),
                        ContainsSubstring("not seen at fit"));
  }
}

TEST_CASE("derived category rank indices preserve ordering across collisions and copies") {
  fft::Categories table;
  for (int i = 0; i < 4096; ++i) table.sorted.push_back("long-category-prefix-" + std::to_string(i));
  std::sort(table.sorted.begin(), table.sorted.end());
  table.has_null = true;
  table.index();
  auto copied = table;
  auto moved = std::move(copied);
  table = {};
  for (std::size_t i = 0; i < moved.sorted.size(); ++i) {
    REQUIRE(moved.rank(moved.sorted[i]) == static_cast<int64_t>(i));
    REQUIRE(moved.rank(moved.sorted[i] + "-unknown") == -1);
  }
  REQUIRE(moved.levels() == 4097);
  REQUIRE(moved.rank("") == -1);
}

namespace {
// Looks a value up as values sit inside an Arrow array, followed by a byte
// that is not NUL, so that a read past its end changes the answer, and again
// from a copy of exactly its size, where AddressSanitizer sees such a read.
// Returns an empty string when both answers are the sorted list's, and
// otherwise what went wrong.
std::string rank_mismatch(const fft::Categories& table, const std::string& value) {
  const auto found = std::lower_bound(table.sorted.begin(), table.sorted.end(), value);
  const int64_t expected = found != table.sorted.end() && *found == value ? found - table.sorted.begin() : -1;
  const std::string followed = value + '\xff';
  const int64_t in_buffer = table.rank(std::string_view(followed.data(), value.size()));
  const auto exact = std::make_unique<char[]>(value.size());
  std::copy(value.begin(), value.end(), exact.get());
  const int64_t alone = table.rank(std::string_view(exact.get(), value.size()));
  if (in_buffer == expected && alone == expected) return {};
  return "'" + value + "' ranks " + std::to_string(in_buffer) + " and " + std::to_string(alone) + ", expected " +
         std::to_string(expected) + "; ";
}

std::string rank_mismatches(const fft::Categories& table, const std::vector<std::string>& queries) {
  std::string mismatches;
  for (const std::string& query : queries) {
    if (mismatches.size() < 2000) mismatches += rank_mismatch(table, query);
  }
  return mismatches;
}

// The keys themselves, each with one byte changed, their prefixes and two
// extensions.
std::vector<std::string> near_queries(const std::vector<std::string>& keys) {
  std::vector<std::string> queries(keys.begin(), keys.end());
  for (const std::string& key : keys) {
    for (std::size_t i = 0; i < key.size(); ++i) {
      std::string changed = key;
      changed[i] = static_cast<char>(changed[i] ^ (1 << (i % 8)));
      queries.push_back(changed);
      queries.push_back(key.substr(0, i));
    }
    queries.push_back(key + 'a');
    queries.push_back(key + '\0');
  }
  return queries;
}
}  // namespace

// The perfect hash over short categories must answer exactly as the sorted
// list does: every category, every one-byte change of one, its prefixes and
// extensions, and random strings, for every key length from empty to past the
// 16 bytes it handles, and for vocabularies on both sides of its 64.
TEST_CASE("short category lookups agree with the sorted list for every length") {
  std::mt19937_64 rng(97);
  const auto random_string = [&](std::size_t size) {
    std::string out(size, '\0');
    for (char& c : out) c = static_cast<char>(rng() % 4 == 0 ? rng() % 256 : 'a' + rng() % 3);
    return out;
  };
  for (const std::size_t count : {1, 2, 3, 15, 16, 17, 32, 63, 64, 65}) {
    for (const std::size_t longest : {0, 3, 7, 8, 16, 17, 20}) {
      CAPTURE(count, longest);
      std::set<std::string> keys;
      for (int attempt = 0; keys.size() < count && attempt < 10000; ++attempt) {
        keys.insert(random_string(static_cast<std::size_t>(rng() % (longest + 1))));
      }
      fft::Categories table;
      table.sorted.assign(keys.begin(), keys.end());
      table.index();
      const bool all_short = std::all_of(keys.begin(), keys.end(), [](const std::string& k) { return k.size() <= 16; });
      REQUIRE(table.short_keys.empty() == (!all_short || keys.size() > 64));
      std::vector<std::string> queries = near_queries(table.sorted);
      for (int i = 0; i < 200; ++i) queries.push_back(random_string(static_cast<std::size_t>(rng() % 22)));
      const fft::Categories copied = table;
      REQUIRE(rank_mismatches(table, queries).empty());
      REQUIRE(rank_mismatches(copied, queries).empty());
    }
  }
}

// Keys a weak hash cannot separate under any seed: a shared first word with
// the last word and the size differing in step ("1000" and "10000"), runs of
// one byte, NUL runs from the empty key up, sequential codes that differ only
// in their last bytes, and single bytes. Each vocabulary gets the short table
// and answers as the sorted list does.
TEST_CASE("short category lookups get a table for structured vocabularies") {
  const auto padded = [](int value, std::size_t width) {
    const std::string digits = std::to_string(value);
    return std::string(width > digits.size() ? width - digits.size() : 0, '0') + digits;
  };
  std::vector<std::vector<std::string>> vocabularies;
  std::vector<std::string> tens;
  for (std::size_t digits = 1; digits <= 16; ++digits) {
    for (const char lead : {'1', '2', '5'}) tens.push_back(lead + std::string(digits - 1, '0'));
  }
  vocabularies.push_back(tens);
  for (const char byte : {'0', 'a', '\0', '\xff'}) {
    std::vector<std::string> runs;
    for (std::size_t size = 0; size <= 16; ++size) runs.emplace_back(size, byte);
    vocabularies.push_back(runs);
  }
  for (const auto& [prefix, width] : std::vector<std::pair<std::string, std::size_t>>{
           {"", 1}, {"c", 3}, {"item", 4}, {"category_", 2}, {"abcdefghijklm", 3}}) {
    std::vector<std::string> codes;
    for (int i = 0; i < 64; ++i) codes.push_back(prefix + padded(i, width));
    vocabularies.push_back(codes);
  }
  std::vector<std::string> bytes;
  for (int byte = 0; byte < 256; byte += 4) bytes.emplace_back(1, static_cast<char>(byte));
  vocabularies.push_back(bytes);
  for (std::vector<std::string>& keys : vocabularies) {
    std::sort(keys.begin(), keys.end());
    CAPTURE(keys);
    fft::Categories table;
    table.sorted = keys;
    table.index();
    REQUIRE_FALSE(table.short_keys.empty());
    REQUIRE(rank_mismatches(table, near_queries(keys)).empty());
  }
}

// Near misses of one key: its size and one of its two words (the first or last
// byte changed), or both words and another size (a repeated byte, or trailing
// NULs below four bytes). A one-key table has sixteen slots, so each near miss
// lands on the key's own slot about one time in sixteen, where only the full
// comparison turns it away.
TEST_CASE("short category lookups turn away near misses of a key") {
  const auto require_near = [](const std::string& key, const std::vector<std::string>& near) {
    fft::Categories table;
    table.sorted = {key};
    table.index();
    REQUIRE_FALSE(table.short_keys.empty());
    CAPTURE(key);
    std::vector<std::string> queries = near;
    queries.push_back(key);
    REQUIRE(rank_mismatches(table, queries).empty());
  };
  for (std::size_t size = 1; size <= 16; ++size) {
    for (int lead = 0; lead < 256; ++lead) {
      std::string key(size, 'k');
      key[0] = static_cast<char>(lead);
      std::vector<std::string> near;
      for (int byte = 0; byte < 256; byte += 8) {
        std::string head = key, tail = key;
        head[0] = static_cast<char>(byte + 1);
        tail[size - 1] = static_cast<char>(byte + 3);
        near.push_back(head);
        near.push_back(tail);
      }
      require_near(key, near);
    }
  }
  for (int byte = 0; byte < 256; ++byte) {
    for (std::size_t size = 0; size <= 16; ++size) {
      const std::string key(size, static_cast<char>(byte));
      std::vector<std::string> near;
      for (std::size_t other = 0; other <= 17; ++other) near.emplace_back(other, static_cast<char>(byte));
      for (std::size_t pad = 1; size + pad <= 3; ++pad) near.push_back(key + std::string(pad, '\0'));
      require_near(key, near);
    }
  }
}

// Two equal keys share a slot under every seed, the one way to make every seed
// fail. The lookup then falls back as an ineligible vocabulary does: to the
// binary search below 16 categories and to the probing table from 16.
TEST_CASE("short category lookups fall back when no seed separates the keys") {
  for (const std::size_t count : {2, 15, 16, 40}) {
    CAPTURE(count);
    fft::Categories table;
    for (std::size_t i = 0; i + 1 < count; ++i) table.sorted.push_back("k" + std::to_string(100 + i));
    table.sorted.push_back(table.sorted.back());
    table.index();
    REQUIRE(table.short_keys.empty());
    REQUIRE(table.rank_slots.empty() == (count < 16));
    for (const std::string& key : table.sorted) {
      const int64_t rank = table.rank(key);
      REQUIRE(rank >= 0);
      REQUIRE(table.sorted[static_cast<std::size_t>(rank)] == key);
    }
    REQUIRE(table.rank("k1") == -1);
    REQUIRE(table.rank("") == -1);
  }
}

namespace {
// A fitted state with `prefix` in front of every object key and string value
// that is one of `words`.
nlohmann::json prefixed(const nlohmann::json& state, const std::set<std::string>& words, const std::string& prefix) {
  if (state.is_string()) {
    const auto& text = state.get_ref<const std::string&>();
    return words.contains(text) ? nlohmann::json(prefix + text) : state;
  }
  if (state.is_array()) {
    nlohmann::json out = nlohmann::json::array();
    for (const auto& item : state) out.push_back(prefixed(item, words, prefix));
    return out;
  }
  if (!state.is_object()) return state;
  nlohmann::json out = nlohmann::json::object();
  for (const auto& [key, value] : state.items()) {
    out[words.contains(key) ? prefix + key : key] = prefixed(value, words, prefix);
  }
  return out;
}
}  // namespace

// The string-keyed encoders read a perfect hash when their categories are
// short and few, and their hash map otherwise. One fitted table must answer
// alike either way. Fitted on short categories, an encoder indexes them for
// the hash (nothing here checks that it then reads the hash). Loaded back
// with every category behind a 16-byte prefix, which makes the keys too long
// for the hash, it takes the map, and each probe behind the same prefix must
// encode as the probe did. Probes are the categories, near misses of them,
// unseen strings and a null, plain and dictionary-encoded.
TEST_CASE("string-keyed encoders answer alike through the perfect hash and their map") {
  const std::string prefix(16, 'p');
  std::vector<std::string> categories{"", std::string("\0", 1), "a", "ab", "abc", "abcd"};
  for (int i = 0; i < 30; ++i) categories.push_back("k" + std::to_string(i * 37));
  const std::set<std::string> words(categories.begin(), categories.end());
  std::vector<std::optional<std::string>> rows;
  std::vector<std::optional<int64_t>> targets;
  std::vector<std::optional<double>> numbers;
  for (int row = 0; row < 600; ++row) {
    rows.push_back(categories[static_cast<std::size_t>(row * 7 % static_cast<int>(categories.size()))]);
    targets.emplace_back(row % 3 == 0 ? 1 : 0);
    numbers.emplace_back(row % 11 * 0.5);
  }
  rows[5] = std::nullopt;
  std::vector<std::optional<std::string>> probes(categories.begin(), categories.end());
  for (const std::string& category : categories) {
    probes.push_back(category + "x");
    if (!category.empty()) probes.push_back(category.substr(1));
  }
  probes.push_back("unseen");
  probes.push_back(std::nullopt);
  const auto behind = [&](const std::vector<std::optional<std::string>>& values) {
    std::vector<std::optional<std::string>> out;
    for (const auto& value : values) out.push_back(value ? std::optional(prefix + *value) : std::nullopt);
    return out;
  };
  const fft::Batch training = fft::testing::make_batch(
      {"c", "target", "x"}, {fft::testing::utf8_column(rows), fft::testing::numeric_column<arrow::Int64Type>(targets),
                             fft::testing::numeric_column<arrow::DoubleType>(numbers)});
  const nlohmann::json supervised{{"columns", {"c"}}, {"target", "target"}};
  const std::vector<std::pair<std::string, nlohmann::json>> kernels{
      {"count_encode", {{"columns", {"c"}}}},
      {"target_encode", supervised},
      {"m_estimate_encode", supervised},
      {"woe_encode", supervised},
      {"james_stein_encode", supervised},
      {"leave_one_out_encode", supervised},
      {"cat_boost_encode", supervised},
      {"quantile_encode", supervised},
      {"summary_encode", {{"columns", {"c"}}, {"target", "target"}, {"quantiles", {0.25, 0.75}}}},
      {"label_encode", {{"columns", {"c"}}}},
      {"group_agg", {{"key", "c"}, {"aggs", {{"x", {"count", "sum", "max"}}}}}}};
  for (const auto& [id, params] : kernels) {
    CAPTURE(id);
    const auto make = [&] {
      auto kernel = fft::make_kernel(id, params);
      kernel->stage = "enc";
      return kernel;
    };
    auto hashed = make();
    fft::Schema schema{{{"c", "utf8"}, {"target", "i64"}, {"x", "f64"}}};
    hashed->resolve(schema);
    auto state = hashed->make_state();
    hashed->consume(*state, training);
    std::vector<std::unique_ptr<fft::AccState>> states;
    states.push_back(std::move(state));
    hashed->finalize(std::move(states));
    auto mapped = make();
    mapped->load_state(prefixed(hashed->state(), words, prefix));
    fft::Schema input{{{"c", "utf8"}}};
    fft::Schema same = input;
    hashed->resolve(input);
    mapped->resolve(same);
    for (const bool dictionary : {false, true}) {
      CAPTURE(dictionary);
      const auto column = [&](const std::vector<std::optional<std::string>>& values) -> std::shared_ptr<arrow::Array> {
        if (!dictionary) return fft::testing::utf8_column(values);
        // Each distinct value once in the dictionary, in order of first use.
        std::vector<std::optional<std::string>> entries;
        std::vector<std::optional<int64_t>> indices;
        for (const auto& value : values) {
          if (!value) {
            indices.emplace_back(std::nullopt);
            continue;
          }
          auto at = std::find(entries.begin(), entries.end(), value);
          if (at == entries.end()) at = entries.insert(entries.end(), value);
          indices.emplace_back(at - entries.begin());
        }
        return fft::testing::dictionary_fixture(entries, indices).encoded;
      };
      const fft::Batch expected = hashed->transform(fft::testing::make_batch({"c"}, {column(probes)}));
      const fft::Batch actual = mapped->transform(fft::testing::make_batch({"c"}, {column(behind(probes))}));
      REQUIRE(actual.names == expected.names);
      for (std::size_t col = 0; col < actual.cols.size(); ++col) {
        if (actual.names[col] == "c" && id == "group_agg") continue;  // the key passes through as given
        REQUIRE(actual.cols[col]->Equals(expected.cols[col]));
      }
    }
  }
}

namespace {
// A kernel fitted on one batch with one worker state.
std::unique_ptr<fft::Kernel> fit_one(const std::string& id, const nlohmann::json& params, fft::Schema schema,
                                     const fft::Batch& batch) {
  auto kernel = fft::make_kernel(id, params);
  kernel->stage = "enc";
  kernel->resolve(schema);
  std::vector<std::unique_ptr<fft::AccState>> states;
  states.push_back(kernel->make_state());
  kernel->consume(*states.back(), batch);
  kernel->finalize(std::move(states));
  return kernel;
}
}  // namespace

// Each column keeps a perfect hash of its own, so two columns of different
// short vocabularies must each be coded through their own.
TEST_CASE("label_encode codes each column through its own vocabulary") {
  auto kernel = fit_one("label_encode", {{"columns", {"a", "b"}}}, fft::Schema{{{"a", "utf8"}, {"b", "utf8"}}},
                        fft::testing::make_batch({"a", "b"}, {fft::testing::utf8_column({"x", "y", "x"}),
                                                              fft::testing::utf8_column({"p", "q", "r"})}));
  const fft::Batch out = kernel->transform(fft::testing::make_batch(
      {"a", "b"}, {fft::testing::utf8_column({"y", "x", "x"}), fft::testing::utf8_column({"r", "q", "p"})}));
  REQUIRE(out.cols[0]->Equals(fft::testing::numeric_column<arrow::Int64Type>({2, 1, 1})));
  REQUIRE(out.cols[1]->Equals(fft::testing::numeric_column<arrow::Int64Type>({3, 2, 1})));
}

// The same for the shrinkage encoders. Their record path reads the same
// tables, so the reference is the map: the fitted state loaded back with
// every category behind a 17-byte prefix, too long for the hash. Each
// category encodes apart from the prior and from the others in its column,
// so a lookup in the wrong column's table shows.
TEST_CASE("shrinkage encoders look each column up in its own table") {
  const fft::Batch training = fft::testing::make_batch(
      {"a", "b", "y"}, {fft::testing::utf8_column({"x", "x", "x", "w", "w", "w", "x", "w", "w"}),
                        fft::testing::utf8_column({"p", "p", "q", "q", "q", "r", "r", "r", "r"}),
                        fft::testing::numeric_column<arrow::Int64Type>({1, 1, 1, 0, 0, 0, 1, 0, 1})});
  const std::string prefix(17, '_');
  const std::set<std::string> words{"x", "w", "p", "q", "r"};
  const std::vector<std::string> as{"w", "x", "x", "w"};
  const std::vector<std::string> bs{"r", "q", "p", "r"};
  const auto probe = [&](const std::string& before) {
    std::vector<std::optional<std::string>> a;
    std::vector<std::optional<std::string>> b;
    for (std::size_t row = 0; row < as.size(); ++row) {
      a.emplace_back(before + as[row]);
      b.emplace_back(before + bs[row]);
    }
    return fft::testing::make_batch({"a", "b"}, {fft::testing::utf8_column(a), fft::testing::utf8_column(b)});
  };
  for (const std::string id : {"james_stein_encode", "leave_one_out_encode", "cat_boost_encode"}) {
    CAPTURE(id);
    const nlohmann::json params{{"columns", {"a", "b"}}, {"target", "y"}};
    auto kernel = fit_one(id, params, fft::Schema{{{"a", "utf8"}, {"b", "utf8"}, {"y", "i64"}}}, training);
    auto mapped = fft::make_kernel(id, params);
    mapped->stage = "enc";
    mapped->load_state(prefixed(kernel->state(), words, prefix));
    const fft::Batch out = kernel->transform(probe(""));
    const fft::Batch expected = mapped->transform(probe(prefix));
    REQUIRE(out.cols.size() == expected.cols.size());
    for (std::size_t col = 0; col < out.cols.size(); ++col) {
      const auto& actual = static_cast<const arrow::DoubleArray&>(*out.cols[col]);
      const auto& reference = static_cast<const arrow::DoubleArray&>(*expected.cols[col]);
      for (int64_t row = 0; row < actual.length(); ++row) {
        CAPTURE(col, row);
        REQUIRE(std::bit_cast<uint64_t>(actual.Value(row)) == std::bit_cast<uint64_t>(reference.Value(row)));
      }
    }
  }
}

// An unseen category follows handle_unknown whichever lookup misses it: the
// hash for one-byte categories, the map for twenty-byte ones.
TEST_CASE("quantile encoders apply handle_unknown on both lookups") {
  for (const std::string id : {"quantile_encode", "summary_encode"}) {
    for (const std::size_t width : {std::size_t{1}, std::size_t{20}}) {
      CAPTURE(id, width);
      const std::string a(width, 'a');
      const std::string b(width, 'b');
      const fft::Batch training =
          fft::testing::make_batch({"c", "y"}, {fft::testing::utf8_column({a, b, a, b}),
                                                fft::testing::numeric_column<arrow::DoubleType>({1.0, 2.0, 3.0, 4.0})});
      const fft::Schema schema{{{"c", "utf8"}, {"y", "f64"}}};
      const fft::Batch probe = fft::testing::make_batch({"c"}, {fft::testing::utf8_column({a, "zz"})});
      auto strict = fit_one(id, {{"columns", {"c"}}, {"target", "y"}, {"handle_unknown", "error"}}, schema, training);
      REQUIRE_THROWS_WITH(strict->transform(probe), ContainsSubstring("not seen at fit"));
      auto as_null =
          fit_one(id, {{"columns", {"c"}}, {"target", "y"}, {"handle_unknown", "return_nan"}}, schema, training);
      const fft::Batch out = as_null->transform(probe);
      for (const auto& column : out.cols) {
        REQUIRE_FALSE(column->IsNull(0));
        REQUIRE(column->IsNull(1));
      }
    }
  }
}

// An index rebuilt over keys it cannot take is empty, not the one before.
TEST_CASE("a short key index rebuilt over keys it cannot take is empty") {
  std::vector<std::string> many;
  for (int i = 0; i < 65; ++i) many.push_back("k" + std::to_string(i));
  const std::vector<std::vector<std::string>> unindexable{{std::string(17, 'a')}, many, {"same", "same"}};
  for (const auto& keys : unindexable) {
    CAPTURE(keys.size());
    fft::ShortKeys index;
    index.index(std::vector<std::string>{"a", "b"});
    REQUIRE_FALSE(index.empty());
    index.index(keys);
    REQUIRE(index.empty());
  }
}

// A map the index cannot take leaves no values behind, so a large or long
// vocabulary costs nothing beside its map: one with more keys than the index
// takes is not even copied, and one with a key too long for it keeps none of
// the values it read.
TEST_CASE("short values keep no copy of a map too large to index") {
  std::vector<std::pair<std::string, int64_t>> entries;
  for (int i = 0; i < 65; ++i) entries.emplace_back("k" + std::to_string(i), i);
  fft::ShortValues<int64_t> values;
  values.index(entries);
  REQUIRE(values.keys.empty());
  REQUIRE(values.values.capacity() == 0);
  const std::vector<std::pair<std::string, int64_t>> few(entries.begin(), entries.begin() + 3);
  values.index(few);
  REQUIRE_FALSE(values.keys.empty());
  REQUIRE(values.values == std::vector<int64_t>{0, 1, 2});
  values.index(entries);
  REQUIRE(values.keys.empty());
  REQUIRE(values.values.capacity() == 0);
  const std::vector<std::pair<std::string, int64_t>> long_key{{"a", 0}, {std::string(17, 'b'), 1}};
  values.index(long_key);
  REQUIRE(values.keys.empty());
  REQUIRE(values.values.capacity() == 0);
}

TEST_CASE("wide record bindings retain caller key order and fitted validation order") {
  std::vector<std::string> columns;
  fft::Schema schema;
  fft::Batch fit;
  fit.rows = 2;
  for (int i = 0; i < 150; ++i) {
    const std::string name = "c" + std::to_string(i);
    columns.push_back(name);
    schema.cols.emplace_back(name, "utf8");
    fit.names.push_back(name);
    fit.cols.push_back(fft::testing::utf8_column({"a", "b"}));
  }
  auto kernel = fft::make_kernel("ordinal_encode", {{"columns", columns}});
  kernel->stage = "wide";
  kernel->resolve(schema);
  auto state = kernel->make_state();
  kernel->consume(*state, fit);
  std::vector<std::unique_ptr<fft::AccState>> states;
  states.push_back(std::move(state));
  kernel->finalize(std::move(states));
  fft::Row input{{"passthrough", int64_t{99}}};
  for (int i = 0; i < 150; ++i) {
    const int index = (37 * i) % 150;
    input.emplace_back(columns[static_cast<std::size_t>(index)], std::string(index % 2 ? "a" : "b"));
  }
  const auto result = kernel->transform_row(input);
  REQUIRE(result.front() == input.front());
  for (std::size_t i = 1; i < input.size(); ++i) {
    REQUIRE(result[i].first == input[i].first);
    REQUIRE(std::get<int64_t>(result[i].second) == (std::get<std::string>(input[i].second) == "a" ? 1 : 2));
  }
  *fft::row_find(input, "c30") = int64_t{1};
  std::erase_if(input, [](const auto& field) { return field.first == "c73"; });
  REQUIRE_THROWS_WITH(kernel->transform_row(input), ContainsSubstring("key 'c30' must be a string or null"));
}

namespace {

// A field name that counts how often it is read as text. A record lookup
// reads a name once for each comparison it makes with it, so the count is
// the number of fields a lookup tried.
struct CountedName {
  std::string text;
  static inline std::size_t reads = 0;
  operator std::string_view() const {  // NOLINT(google-explicit-constructor): read wherever a name is
    ++reads;
    return text;
  }
};
using CountedRecord = std::vector<std::pair<CountedName, fft::RowValue>>;
using Counts = std::vector<std::size_t>;

// Fields f0, f1, ... in that order, each holding its own position.
CountedRecord counted_record(std::size_t fields) {
  CountedRecord record;
  for (std::size_t i = 0; i < fields; ++i) {
    record.emplace_back(CountedName{"f" + std::to_string(i)}, static_cast<int64_t>(i));
  }
  return record;
}

// Looks `names` up in this order through one lookup and returns how many
// fields each of them compared. Every answer is checked against a plain
// search of the record, which reads no name through the counter.
Counts comparisons(CountedRecord& record, const std::vector<std::string>& names) {
  fft::RecordLookup lookup(record, names.size());
  Counts counts;
  for (const std::string& name : names) {
    const auto field =
        std::find_if(record.begin(), record.end(), [&](const auto& entry) { return entry.first.text == name; });
    CountedName::reads = 0;
    const fft::RowValue* found = lookup.find(name);
    counts.push_back(CountedName::reads);
    REQUIRE(found == (field == record.end() ? nullptr : &field->second));
  }
  return counts;
}

// find() moves the cursor, so a lookup that is const cannot find.
template <typename Lookup>
concept Finds = requires(Lookup lookup, const std::string& name) { lookup.find(name); };
static_assert(Finds<fft::RecordLookup<fft::Row>&>);
static_assert(!Finds<const fft::RecordLookup<fft::Row>&>);

}  // namespace

TEST_CASE("a record lookup compares one name for each field read in the record's order") {
  CountedRecord record = counted_record(6);
  REQUIRE(comparisons(record, {"f0", "f1", "f2", "f3", "f4", "f5"}) == Counts{1, 1, 1, 1, 1, 1});
  // A run that begins inside the record pays one scan to reach it.
  REQUIRE(comparisons(record, {"f3", "f4", "f5"}) == Counts{4, 1, 1});
  // A gap pays the field tried first and then the scan.
  REQUIRE(comparisons(record, {"f0", "f2", "f3", "f5"}) == Counts{1, 4, 1, 7});
}

TEST_CASE("a record lookup pays one comparison more than a scan at most in any other order") {
  CountedRecord record = counted_record(6);
  // Reversed: the first two lookups have no field to try first, one before
  // any hit and one after a hit on the last field.
  REQUIRE(comparisons(record, {"f5", "f4", "f3", "f2", "f1", "f0"}) == Counts{6, 5, 5, 4, 3, 2});
  REQUIRE(comparisons(record, {"f3", "f0", "f5", "f1", "f4", "f2"}) == Counts{4, 2, 7, 2, 6, 4});
  // The last field, then the first: the scan finds it at once, and the
  // field after it is tried first again.
  REQUIRE(comparisons(record, {"f5", "f0", "f1"}) == Counts{6, 1, 1});
  // A name asked twice is found by the scan the second time.
  REQUIRE(comparisons(record, {"f2", "f2", "f3"}) == Counts{3, 4, 1});
  // A miss between two hits leaves the field to try where it was.
  REQUIRE(comparisons(record, {"f0", "absent", "f1"}) == Counts{1, 7, 1});
}

// A step of one column builds a lookup for one name. Trying a field before
// any hit would cost it a second comparison of the first field whenever its
// column is another one.
TEST_CASE("a record lookup's first name costs the scan alone") {
  CountedRecord record = counted_record(6);
  for (std::size_t i = 0; i < record.size(); ++i) {
    CAPTURE(i);
    REQUIRE(comparisons(record, {record[i].first.text}) == Counts{i + 1});
  }
  REQUIRE(comparisons(record, {"absent"}) == Counts{6});
}

TEST_CASE("a record lookup answers as a plain search does for every sequence of names") {
  CountedRecord record = counted_record(5);
  const std::vector<std::string> names{"f0", "f1", "f2", "f3", "f4", "absent"};
  // Every sequence of one to four lookups, repeats and misses included.
  for (std::size_t length = 1; length <= 4; ++length) {
    std::size_t sequences = 1;
    for (std::size_t i = 0; i < length; ++i) sequences *= names.size();
    for (std::size_t code = 0; code < sequences; ++code) {
      std::vector<std::size_t> picks;
      for (std::size_t rest = code; picks.size() < length; rest /= names.size()) picks.push_back(rest % names.size());
      std::vector<std::string> sequence;
      for (const std::size_t pick : picks) sequence.push_back(names[pick]);
      const Counts counts = comparisons(record, sequence);
      std::optional<std::size_t> hit;  // the field the previous hit was on
      for (std::size_t i = 0; i < length; ++i) {
        CAPTURE(code, i);
        // What a search from the first field compares: every field up to
        // the name's own, or all five for a name the record lacks.
        const std::size_t scan = std::min(picks[i] + 1, record.size());
        // A field is tried first when one follows the previous hit.
        const bool tries = hit.has_value() && *hit + 1 < record.size();
        if (tries && picks[i] == *hit + 1) {
          REQUIRE(counts[i] == 1);
        } else {
          REQUIRE(counts[i] == scan + (tries ? 1 : 0));
        }
        if (picks[i] < record.size()) hit = picks[i];
      }
    }
  }
}

TEST_CASE("a record lookup finds each field of a record whatever order they are asked in") {
  fft::Row record{{"", std::monostate{}},         {"ab", int64_t{1}},
                  {"price", std::string("p")},    {"prize", true},
                  {"a_name_of_twenty_one!", 5.0}, {"a_name_of_twenty_one?", 6.0}};
  std::vector<std::size_t> order(record.size());
  for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
  do {
    fft::RecordLookup lookup(record, order.size());
    for (const std::size_t index : order) REQUIRE(lookup.find(record[index].first) == &record[index].second);
  } while (std::next_permutation(order.begin(), order.end()));
  // A step writes its result through what it finds.
  fft::RecordLookup lookup(record, 3);
  REQUIRE(lookup.find("Price") == nullptr);
  *lookup.find("price") = int64_t{3};
  *lookup.find("prize") = std::monostate{};
  REQUIRE(std::get<int64_t>(record[2].second) == 3);
  REQUIRE(std::holds_alternative<std::monostate>(record[3].second));
}

TEST_CASE("a record lookup tells names of one size apart by any of their bytes") {
  for (std::size_t size = 1; size <= 40; ++size) {
    CAPTURE(size);
    // The name itself, then one field for each byte another name differs in.
    const std::string name(size, 'n');
    fft::Row record{{name, int64_t{-1}}};
    for (std::size_t byte = 0; byte < size; ++byte) {
      std::string other = name;
      other[byte] = 'm';
      record.emplace_back(other, static_cast<int64_t>(byte));
    }
    // In the record's order each name after the first is the field tried
    // first. Last field first, each is compared with the field after the
    // previous hit, which is another name, and then scanned for.
    fft::RecordLookup forward(record, record.size());
    for (std::size_t i = 0; i < record.size(); ++i) REQUIRE(forward.find(record[i].first) == &record[i].second);
    fft::RecordLookup backward(record, record.size());
    for (std::size_t i = record.size(); i-- > 0;) REQUIRE(backward.find(record[i].first) == &record[i].second);
    std::string absent = name;
    absent[size / 2] = 'o';
    REQUIRE(backward.find(absent) == nullptr);
    REQUIRE(backward.find(name + "n") == nullptr);
  }
}

TEST_CASE("a record lookup indexes a record of 128 fields that is read 16 times") {
  std::vector<std::string> names;
  for (std::size_t i = 0; i < 16; ++i) names.push_back("f" + std::to_string(127 - 5 * i));
  // Through the index a lookup compares no name of the record, in any order.
  CountedRecord wide = counted_record(128);
  REQUIRE(comparisons(wide, names) == Counts(16, 0));
  // One lookup or one field fewer keeps the linear lookup, which scans to
  // the last field for the first name.
  REQUIRE(comparisons(wide, std::vector<std::string>(names.begin(), names.begin() + 15)).front() == 128);
  CountedRecord narrower = counted_record(127);
  names.front() = "f126";
  REQUIRE(comparisons(narrower, names).front() == 127);
}

TEST_CASE("a record lookup answers with one field of a name the record holds twice") {
  fft::Row record{{"a", int64_t{1}}, {"b", int64_t{2}}, {"a", int64_t{3}}};
  // The scan answers with the first of the two.
  fft::RecordLookup scan(record, 1);
  REQUIRE(scan.find("a") == &record[0].second);
  // The field after the previous hit is tried first, and here it is the
  // second of the two.
  fft::RecordLookup cursor(record, 2);
  REQUIRE(cursor.find("b") == &record[1].second);
  REQUIRE(cursor.find("a") == &record[2].second);
  // The index keeps the first, whatever was found before.
  for (int i = 0; i < 126; ++i) record.emplace_back("f" + std::to_string(i), int64_t{0});
  fft::RecordLookup indexed(record, 16);
  REQUIRE(indexed.find("b") == &record[1].second);
  REQUIRE(indexed.find("a") == &record[0].second);
}

TEST_CASE("cached BaseN words retain binary indicator spelling for Gray codes") {
  for (const std::string id : {"basen_encode", "binary_encode", "gray_encode"}) {
    for (const int base : {1, 3}) {
      CAPTURE(id, base);
      nlohmann::json params{{"columns", {"c"}}, {"handle_unknown", "indicator"}};
      if (id == "basen_encode") params["base"] = base;
      auto kernel = fft::make_kernel(id, params);
      kernel->stage = "enc";
      kernel->load_state({{"columns", {"c"}}, {"categories", {{"c", {"a", "b", "c"}}}}, {"has_null", {{"c", false}}}});
      const std::vector<std::optional<std::string>> logical{"a", "unknown", std::nullopt, "c",          "b", "unknown",
                                                            "c", "unknown", "a",          std::nullopt, "b", "unknown"};
      const auto plain = kernel->transform(fft::testing::make_batch({"c"}, {fft::testing::utf8_column(logical)}));
      require_scalar_values(*kernel, plain, logical);
      const auto dictionary = arrow::DictionaryArray::FromArrays(
          fft::testing::numeric_column<arrow::Int8Type>({0, 3, 4, 2, 1, 3, 2, 3, 0, 4, 1, 3}),
          fft::testing::utf8_column({"a", "b", "c", "unknown", std::nullopt}));
      REQUIRE(dictionary.ok());
      const auto encoded = kernel->transform(fft::testing::make_batch({"c"}, {*dictionary}));
      require_scalar_values(*kernel, encoded, logical);
      if (id == "gray_encode") {
        // Three fitted levels plus the indicator cross a width boundary.
        // The unknown rank 4 is binary 100, not Gray 110.
        REQUIRE(encoded.cols.size() == 3);
        REQUIRE(static_cast<const arrow::Int64Array&>(*encoded.cols[0]).Value(1) == 1);
        REQUIRE(static_cast<const arrow::Int64Array&>(*encoded.cols[1]).Value(1) == 0);
        REQUIRE(static_cast<const arrow::Int64Array&>(*encoded.cols[2]).Value(1) == 0);
      }
    }
  }
}
