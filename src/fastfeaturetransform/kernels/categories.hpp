// The categorical substrate every category_encoders port is built on: the
// handle_missing / handle_unknown options and their parsers, the guards a
// fitted table has to pass, the per-column table of fitted categories, and
// the accumulator that collects them.
//
// The options live here and nowhere else, so handle_missing='error' means the
// same thing in every kernel. Only count_encode extends them, with an integer
// fill, and it wraps the enum below rather than redeclaring it.
//
// category_encoders numbers categories by order of first appearance in the
// fit frame. We sort them instead, so a fitted artifact never depends on how
// the rows were split across workers.
// A null seen at fit under handle_missing='value' is a category of its own
// and sorts after every real one, which is where upstream puts it too.
#pragma once

#include <arrow/util/utf8.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "fastfeaturetransform/kernel.hpp"
#include "fastfeaturetransform/kernels/util.hpp"

namespace fastfeaturetransform {

// Resolve repeated dictionary entries once, in the order rows first use them.
// Uncached arrays use the ordinary visitor; cached arrays read each index once
// and fetch a string only for its first accepted use. Unused entries never run
// resolve (which can reject an unknown category).
// A dense cache only pays for dictionaries no larger than the batch, and its
// allocation is capped independently of batch size. Results must be stable
// values or indices, not pointers into a container that grows while resolving.
template <typename Accept, typename Resolve, typename Emit>
void for_each_encoded_utf8_if(const arrow::Array& array, const std::string& stage, const std::string& name,
                              Accept&& accept, Resolve&& resolve, Emit&& emit) {
  using Result = std::decay_t<std::invoke_result_t<Resolve, bool, std::string_view>>;
  constexpr std::size_t max_cache_bytes = 8 * 1024 * 1024;
  const auto visit = [&](int64_t row, bool is_null, std::string_view value) {
    if (accept(row, is_null)) emit(row, resolve(is_null, value));
  };
  if (array.type_id() != arrow::Type::DICTIONARY) {
    for_each_utf8(array, stage, name, visit);
    return;
  }
  const auto& dictionary = static_cast<const arrow::DictionaryArray&>(array);
  const arrow::Array& values = *dictionary.dictionary();
  const int64_t entries = values.length();
  if (entries <= 0 || entries > array.length() ||
      static_cast<uint64_t>(entries) > max_cache_bytes / sizeof(std::optional<Result>)) {
    for_each_utf8_value(array, stage, name, visit);
    return;
  }
  if (values.type_id() != arrow::Type::STRING && values.type_id() != arrow::Type::LARGE_STRING) {
    throw Error(stage + ": column '" + name + "' dictionary is " + values.type()->ToString() + ", expected utf8");
  }
  std::vector<std::optional<Result>> cache(static_cast<std::size_t>(entries));
  // Nulls are asked for once per array, as in for_each_utf8. The bound on
  // entries above keeps every valid index far inside int32, and indices were
  // checked against the dictionary as each Parquet batch was read.
  const bool row_nulls = dictionary.null_count() != 0;
  const bool entry_nulls = values.null_count() != 0;
  std::vector<int32_t> copied;
  const int32_t* indices = int32_indices(dictionary, copied);
  for (int64_t row = 0; row < dictionary.length(); ++row) {
    if (row_nulls && dictionary.IsNull(row)) {
      if (accept(row, true)) emit(row, resolve(true, std::string_view{}));
      continue;
    }
    const int64_t entry = indices[row];
    const bool is_null = entry_nulls && values.IsNull(entry);
    if (!accept(row, is_null)) continue;
    if (is_null) {
      emit(row, resolve(true, std::string_view{}));
      continue;
    }
    auto& cached = cache[static_cast<std::size_t>(entry)];
    if (!cached.has_value()) {
      const std::string_view value = values.type_id() == arrow::Type::STRING
                                         ? static_cast<const arrow::StringArray&>(values).GetView(entry)
                                         : static_cast<const arrow::LargeStringArray&>(values).GetView(entry);
      cached.emplace(resolve(false, value));
    }
    emit(row, *cached);
  }
}

template <typename Resolve, typename Emit>
void for_each_encoded_utf8(const arrow::Array& array, const std::string& stage, const std::string& name,
                           Resolve&& resolve, Emit&& emit) {
  for_each_encoded_utf8_if(
      array, stage, name, [](int64_t, bool) { return true; }, std::forward<Resolve>(resolve), std::forward<Emit>(emit));
}

// Fit accumulators use stable integer slots so a cached dictionary resolution
// survives vector growth. Lookups take a string_view, so only the first
// occurrence of a category allocates a string.
template <typename Value>
struct CategoryAccumulator {
  StringIndex slots;
  std::vector<Value> values;

  std::size_t slot(std::string_view key) {
    auto found = slots.find(key);
    if (found != slots.end()) return found->second;
    const std::size_t index = values.size();
    slots.emplace(std::string(key), index);
    values.emplace_back();
    return index;
  }

  Value& at(std::string_view key) { return values[slot(key)]; }
  const Value& at(std::string_view key) const { return values[slots.find(key)->second]; }
};

// What a null becomes: a category of its own, a null, or an error — at fit
// and at transform alike, which is what category_encoders does.
enum class Missing { Value, ReturnNan, Error };
// What an unseen category becomes.
enum class Unknown { Value, ReturnNan, Error, Indicator };

inline Missing missing_from(const nlohmann::json& params, const std::string& what) {
  const std::string text = params.contains("handle_missing") && !params.at("handle_missing").is_null()
                               ? params.at("handle_missing").get<std::string>()
                               : "value";
  if (text == "value") return Missing::Value;
  if (text == "return_nan") return Missing::ReturnNan;
  if (text == "error") return Missing::Error;
  throw Error(what + ": 'handle_missing' must be 'value', 'return_nan' or 'error'");
}

inline Unknown unknown_from(const nlohmann::json& params, const std::string& what, bool allow_indicator) {
  const std::string text = params.contains("handle_unknown") && !params.at("handle_unknown").is_null()
                               ? params.at("handle_unknown").get<std::string>()
                               : "value";
  if (text == "value") return Unknown::Value;
  if (text == "return_nan") return Unknown::ReturnNan;
  if (text == "error") return Unknown::Error;
  if (text == "indicator" && allow_indicator) return Unknown::Indicator;
  throw Error(what + ": 'handle_unknown' must be 'value', 'return_nan'" + (allow_indicator ? ", 'indicator'" : "") +
              " or 'error'");
}

inline const char* missing_name(Missing missing) {
  switch (missing) {
    case Missing::Value:
      return "value";
    case Missing::ReturnNan:
      return "return_nan";
    case Missing::Error:
      break;
  }
  return "error";
}

inline const char* unknown_name(Unknown unknown) {
  switch (unknown) {
    case Unknown::Value:
      return "value";
    case Unknown::ReturnNan:
      return "return_nan";
    case Unknown::Indicator:
      return "indicator";
    case Unknown::Error:
      break;
  }
  return "error";
}

// handle_missing='error' is the strict option, so it refuses a null wherever
// one turns up: at fit through merge_categories, and here on every lookup,
// batch and record alike. category_encoders raises the same way from its own
// transform, and a guarantee that held only for the fit data would be no
// guarantee at all. Every kernel calls this from the is_null branch of its
// lookup, which is why the branch reads the same in all of them.
inline void refuse_null(Missing missing, const std::string& stage, const std::string& name) {
  if (missing == Missing::Error) {
    throw Error(stage + ": column '" + name + "' has a null and handle_missing is 'error'");
  }
}

// category_encoders builds every expanding encoder on an ordinal pre-encoder
// whose fitted mapping carries a slot for the missing value, unless the
// pre-encoder itself was fit with handle_missing='error'. Under the encoder's
// handle_missing='value' that slot is either a real level (a null was seen at
// fit) or is filtered back out; under every other option it stays in the
// mapping as a level of its own, sorted last, that no value can ever reach —
// a null is answered with nulls, or refused, before the mapping is consulted.
// It still widens what the encoder generates, and the generated width is part
// of the output schema, so every expanding encoder that keeps it counts it.
//
// Which options keep the slot depends on how the encoder fits that
// pre-encoder. basen_encode and the contrast coders always fit theirs with
// handle_missing='value', so their slot survives 'return_nan' and 'error'
// alike, which is what this returns. one_hot_encode and rank_hot_encode fit
// theirs with their own option, so for those two the slot only survives
// 'return_nan' — and one_hot_encode then drops the slot's column explicitly,
// leaving rank_hot_encode as the only expanding encoder that names it.
inline std::size_t missing_placeholder(Missing missing) { return missing == Missing::Value ? 0 : 1; }

// More generated columns than this would be more than any table wants, and
// each one costs a row-length vector per batch per worker. The width of an
// expanding encoder is learned, so nothing in the plan warns of a column that
// was low-cardinality in the fit data and is not in production: the fit
// refuses instead of the batch allocator. hash_encode caps n_components at the
// same number for the same reason.
constexpr std::size_t kMaxOutputs = 4096;

inline void check_output_width(std::size_t width, std::size_t levels, const std::string& stage,
                               const std::string& name) {
  if (width > kMaxOutputs) {
    throw Error(stage + ": column '" + name + "' has " + std::to_string(levels) + " levels, which would generate " +
                std::to_string(width) + " columns, more than the " + std::to_string(kMaxOutputs) + " allowed");
  }
}

// Two encoded columns whose generated names collide would drop one of them
// silently, so the fit refuses. Names are disambiguated where upstream
// disambiguates them; this catches whatever is left.
inline void reject_duplicate_outputs(const std::vector<std::string>& outputs, const std::string& stage) {
  std::vector<std::string> sorted = outputs;
  std::sort(sorted.begin(), sorted.end());
  auto dup = std::adjacent_find(sorted.begin(), sorted.end());
  if (dup != sorted.end()) throw Error(stage + ": output column '" + *dup + "' is generated twice");
}

// A number a kernel computed into its fitted table. load_state refuses a
// value that is not finite and state() refuses to write one, so an encoding
// that blows up fails here, at fit, naming the param to check, rather than
// when the pipeline is saved.
inline double finite_encoding(double value, const std::string& stage, const std::string& name, const char* param) {
  if (!std::isfinite(value)) {
    throw Error(stage + ": column '" + name + "' encodes to a value that is not finite; check '" + std::string(param) +
                "'");
  }
  return value;
}

// The last place a number that is not finite could still reach an artifact.
// nlohmann writes one as JSON null, which reads back as 'this category had no
// value', so a table that somehow held one would save as a corrupt manifest
// rather than fail.
inline double serializable(double value, const std::string& stage, const std::string& name) {
  if (!std::isfinite(value)) {
    throw Error(stage + ": column '" + name + "' holds a value that is not finite and cannot be serialized");
  }
  return value;
}

// An optional per-column object in a manifest, such as 'has_null',
// 'null_values' or 'null_counts'. Absent is the legitimate spelling of
// 'nothing to say', but present and the wrong shape is an error rather than a
// default: a hand-edited manifest must never quietly lose the null category.
// Returns null when the key is absent or null, and otherwise an object
// holding an entry for every column.
inline const nlohmann::json* optional_per_column_object(const nlohmann::json& state, const std::string& key,
                                                        const std::vector<std::string>& columns,
                                                        const std::string& stage) {
  if (!state.contains(key) || state.at(key).is_null()) return nullptr;
  const nlohmann::json& object = state.at(key);
  if (!object.is_object()) throw Error(stage + ": '" + key + "' must be an object");
  for (const std::string& name : columns) {
    if (!object.contains(name)) throw Error(stage + ": " + key + " has no entry for column '" + name + "'");
  }
  return &object;
}

// One column's fitted categories, sorted, plus whether a null was among them.
struct Categories {
  std::vector<std::string> sorted;
  bool has_null = false;
  // Up to 64 categories of at most 16 bytes are found through a perfect hash
  // (util.hpp). The probing table below sent about one lookup in five of the
  // benchmark's 32 categories through another category's slot first, and
  // resolving plain strings took about five times as long that way
  // (measured with benchmark/profile_kernels.py before the hash).
  ShortKeys short_keys;
  std::vector<std::size_t> rank_slots;

  // Otherwise a probing table of positions, not another copy of the keys or
  // views into movable strings. Below 16 categories the binary search is
  // cheap enough, and above 524,288 the table would pass 8 MiB, so those fall
  // back to the binary search too.
  void index() {
    rank_slots.clear();
    short_keys.index(sorted);
    constexpr std::size_t max_slots = 8 * 1024 * 1024 / sizeof(std::size_t);
    if (!short_keys.empty() || sorted.size() < 16 || sorted.size() > max_slots / 2) return;
    std::size_t capacity = 1;
    while (capacity < sorted.size() * 2) capacity *= 2;
    rank_slots.assign(capacity, sorted.size());
    for (std::size_t i = 0; i < sorted.size(); ++i) {
      std::size_t slot = TransparentStringHash{}(sorted[i]) & (capacity - 1);
      while (rank_slots[slot] != sorted.size()) slot = (slot + 1) & (capacity - 1);
      rank_slots[slot] = i;
    }
  }

  // Position of a category in the sorted list, or -1 when unseen.
  int64_t rank(std::string_view value) const {
    if (short_keys.empty()) return rank_long(value);
    return short_keys.find(value);
  }
  // Every category, plus the null one when it was seen.
  std::size_t levels() const { return sorted.size() + (has_null ? 1 : 0); }

 private:
  int64_t rank_long(std::string_view value) const {
    if (!rank_slots.empty()) {
      const std::size_t mask = rank_slots.size() - 1;
      std::size_t slot = TransparentStringHash{}(value)&mask;
      while (rank_slots[slot] != sorted.size()) {
        const std::size_t index = rank_slots[slot];
        if (TransparentStringEqual{}(sorted[index], value)) return static_cast<int64_t>(index);
        slot = (slot + 1) & mask;
      }
      return -1;
    }
    auto found = std::lower_bound(sorted.begin(), sorted.end(), value);
    if (found == sorted.end() || *found != value) return -1;
    return static_cast<int64_t>(found - sorted.begin());
  }
};

// The fit state of label_encode, count_encode and the kernels
// merge_categories serves (ordinal, one-hot, rank-hot, the positional
// encoders and the contrast coders): per column, how many rows held each
// category and how many were null, and how many rows were read.
// label_encode reads the counts, count_encode all three, and the others only
// which categories appeared and whether a null did.
struct SeenState final : AccState {
  std::unordered_map<std::string, StringCounts> counts;
  std::unordered_map<std::string, int64_t> nulls;
  int64_t rows = 0;
};

// Add one batch to a worker's state. A dictionary column whose dictionary
// holds no more entries than the batch has rows, and at most 2^20, is
// tallied by index, and then each entry its rows use is hashed once. Any
// other column is read row by row: a small batch can share a much larger
// row-group dictionary, whose counts would cost more than its rows.
//
// static, so that each file including this compiles a copy of its own,
// whose row loop has the lambda inlined and keeps the counts' map in a
// register. One copy shared by every file is compiled apart from the
// kernels, and its row loop reaches the map through the lambda on every
// row: a fit of 1,000,000 plain strings of 1,000 categories, 30% of them
// null, took 2% to 6% longer that way (one thread, Apple M4).
static inline void consume_categories(const std::vector<std::string>& columns, const std::string& stage,
                                      SeenState& state, const Batch& batch) {
  state.rows += batch.rows;
  for (const std::string& name : columns) {
    const int idx = batch.find(name);
    if (idx < 0) throw Error(stage + ": column '" + name + "' missing from batch");
    StringCounts& counts = state.counts[name];
    int64_t& nulls = state.nulls[name];
    const arrow::Array& array = *batch.cols[static_cast<std::size_t>(idx)];
    constexpr std::size_t max_cache_bytes = 8 * 1024 * 1024;
    const auto* dictionary =
        array.type_id() == arrow::Type::DICTIONARY ? &static_cast<const arrow::DictionaryArray&>(array) : nullptr;
    const int64_t entries = dictionary != nullptr ? dictionary->dictionary()->length() : 0;
    if (entries > 0 && entries <= array.length() &&
        static_cast<uint64_t>(entries) <= max_cache_bytes / sizeof(int64_t)) {
      std::vector<int64_t> index_counts(static_cast<std::size_t>(entries), 0);
      std::vector<int32_t> copied;
      const int32_t* indices = int32_indices(*dictionary, copied);
      const RowNulls null_at(*dictionary);
      const int64_t length = dictionary->length();
      for (int64_t row = 0; row < length; ++row) {
        if (!null_at(row)) ++index_counts[static_cast<std::size_t>(indices[row])];
      }
      // The rows null_at skipped, which is what the array counts as null.
      nulls += dictionary->null_count();
      const arrow::Array& values = *dictionary->dictionary();
      for_each_dictionary_entry(values, stage, name, [&](int64_t entry, bool is_null, std::string_view value) {
        const int64_t count = index_counts[static_cast<std::size_t>(entry)];
        if (count == 0) return;
        if (is_null) {
          nulls += count;
          return;
        }
        auto found = counts.find(value);
        if (found != counts.end()) {
          found->second += count;
        } else {
          counts.emplace(std::string(value), count);
        }
      });
    } else {
      // The null rows are counted before the loop, which then does nothing
      // for them: a test per null row made a label_encode fit of 1,000,000
      // plain strings, 95% of them null, about 3% slower (one thread, Apple
      // M4). A dictionary column's rows are null too where they point at a
      // null entry, which the column does not count.
      nulls += array.null_count();
      if (dictionary != nullptr && dictionary->dictionary()->null_count() != 0) {
        const arrow::Array& values = *dictionary->dictionary();
        for (int64_t row = 0; row < dictionary->length(); ++row) {
          if (dictionary->IsValid(row) && values.IsNull(dictionary->GetValueIndex(row))) ++nulls;
        }
      }
      const auto count_row = [&](int64_t, bool is_null, std::string_view value) {
        if (is_null) return;
        auto found = counts.find(value);
        if (found != counts.end()) {
          ++found->second;
        } else {
          counts.emplace(std::string(value), 1);
        }
      };
      for_each_utf8_value(array, stage, name, count_row);
    }
  }
}

// Fold every worker into the first, freeing each as it is consumed, so the
// peak stays near one copy of each column's categories.
inline std::unique_ptr<SeenState> merge_seen(std::vector<std::unique_ptr<AccState>>& states) {
  std::unique_ptr<SeenState> merged;
  for (auto& raw : states) {
    if (merged == nullptr) {
      merged.reset(static_cast<SeenState*>(raw.release()));
      continue;
    }
    auto& from = static_cast<SeenState&>(*raw);
    merged->rows += from.rows;
    for (auto& [name, counts] : from.counts) {
      StringCounts& into = merged->counts[name];
      if (into.empty()) {
        into = std::move(counts);
        continue;
      }
      for (auto& [key, count] : counts) into.try_emplace(std::move(key), 0).first->second += count;
    }
    for (const auto& [name, count] : from.nulls) merged->nulls[name] += count;
    raw.reset();
  }
  if (merged == nullptr) merged = std::make_unique<SeenState>();
  return merged;
}

inline std::unordered_map<std::string, Categories> merge_categories(const std::vector<std::string>& columns,
                                                                    const std::string& stage,
                                                                    std::vector<std::unique_ptr<AccState>>& states,
                                                                    Missing missing) {
  arrow::util::InitializeUTF8();  // idempotent; ValidateUTF8 needs its tables
  const std::unique_ptr<SeenState> merged = merge_seen(states);

  std::unordered_map<std::string, Categories> tables;
  for (const std::string& name : columns) {
    const int64_t nulls = merged->nulls[name];
    if (missing == Missing::Error && nulls > 0) {
      throw Error(stage + ": column '" + name + "' has nulls and handle_missing is 'error'");
    }
    Categories categories;
    for (auto& [key, count] : merged->counts[name]) {
      if (!arrow::util::ValidateUTF8(key)) {
        throw Error(stage + ": column '" + name +
                    "' has a category that is not valid UTF-8; category keys must be UTF-8 to serialize");
      }
      categories.sorted.push_back(key);
    }
    merged->counts.erase(name);
    std::sort(categories.sorted.begin(), categories.sorted.end());
    categories.index();
    categories.has_null = missing == Missing::Value && nulls > 0;
    if (categories.sorted.empty() && !categories.has_null) {
      throw Error(stage + ": column '" + name + "' has no categories to encode");
    }
    tables.emplace(name, std::move(categories));
  }
  return tables;
}

inline nlohmann::json categories_state(const std::vector<std::string>& columns,
                                       const std::unordered_map<std::string, Categories>& tables) {
  nlohmann::json categories = nlohmann::json::object();
  nlohmann::json nulls = nlohmann::json::object();
  for (const std::string& name : columns) {
    categories[name] = tables.at(name).sorted;
    nulls[name] = tables.at(name).has_null;
  }
  // 'columns' is the resolved order, not a set: see load_fitted_columns.
  return {{"columns", columns}, {"categories", categories}, {"has_null", nulls}};
}

inline std::unordered_map<std::string, Categories> load_categories(const nlohmann::json& state,
                                                                   const std::string& stage,
                                                                   std::vector<std::string>& columns,
                                                                   bool explicit_columns) {
  if (!state.is_object() || !state.contains("categories") || !state.at("categories").is_object()) {
    throw Error(stage + ": state must hold a 'categories' object");
  }
  const nlohmann::json& categories = state.at("categories");
  load_fitted_columns(state, stage, columns, explicit_columns, &categories);
  const nlohmann::json* has_null = optional_per_column_object(state, "has_null", columns, stage);
  std::unordered_map<std::string, Categories> tables;
  for (const std::string& name : columns) {
    if (!categories.contains(name) || !categories.at(name).is_array()) {
      throw Error(stage + ": categories['" + name + "'] must be a list");
    }
    Categories table;
    for (const nlohmann::json& value : categories.at(name)) {
      if (!value.is_string()) throw Error(stage + ": categories['" + name + "'] must hold strings");
      table.sorted.push_back(value.get<std::string>());
    }
    if (!std::is_sorted(table.sorted.begin(), table.sorted.end()) ||
        std::adjacent_find(table.sorted.begin(), table.sorted.end()) != table.sorted.end()) {
      throw Error(stage + ": categories['" + name + "'] must be sorted and unique");
    }
    if (has_null != nullptr) {
      const nlohmann::json& flag = has_null->at(name);
      if (!flag.is_boolean()) throw Error(stage + ": has_null['" + name + "'] must be true or false");
      table.has_null = flag.get<bool>();
    }
    if (table.sorted.empty() && !table.has_null) throw Error(stage + ": categories['" + name + "'] is empty");
    table.index();
    tables.emplace(name, std::move(table));
  }
  return tables;
}

// The utf8 columns a `columns: null` encoder takes, or the listed ones,
// checked against the incoming schema. `target`, when given, is the label
// column of a supervised encoder: it is numeric, so it never joins a derived
// list, and naming it explicitly is an error. Every category_encoders port
// comes through here, so the three target families cannot drift apart from
// each other or from the unsupervised ones.
//
// The list is derived once, at the first resolve of a fit, from the columns
// known then. A fit builds its kernels anew, so the list is empty until that
// resolve, and a derivation that finds no column is an error.
// REGRESSION: the engine resolves again after every pass, and a step that
// fits in a later pass keeps the list it has. It used to derive the list
// again, as numeric.hpp's bind_numeric did. The columns an expanding encoder
// had generated by then are numbers, so they joined a numeric step's list,
// and whether they did depended on which pass the step fitted in.
inline void bind_columns(const Schema& schema, const std::string& stage, std::vector<std::string>& columns,
                         bool explicit_columns, bool fitted, const std::string& what,
                         const std::string* target = nullptr) {
  if (!fitted && !explicit_columns && columns.empty()) {
    for (const auto& [name, type] : schema.cols) {
      if (type == "utf8") columns.push_back(name);
    }
    if (columns.empty()) throw Error(stage + ": no utf8 columns to " + what);
  }
  for (const std::string& name : columns) {
    if (target != nullptr && name == *target) {
      throw Error(stage + ": column '" + name + "' is also the target; a target encoder cannot encode its target");
    }
    const std::string* type = schema.find(name);
    if (type == nullptr) throw Error(stage + ": column '" + name + "' not found in input");
    if (*type != "utf8") {
      throw Error(stage + ": column '" + name + "' is " + *type + ", expected utf8 (cast it to utf8 first)");
    }
  }
}

}  // namespace fastfeaturetransform
