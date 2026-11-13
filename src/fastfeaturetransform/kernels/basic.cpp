// Three of the four core steps: select (projection), label_encode
// (categorical utf8 -> int64 index) and cast (explicit type conversion).
// Kernel behavior may evolve across releases; every artifact records the
// library_version that fitted it, and the deployment rule is to serve a model
// with the release that fitted it.

#include <arrow/util/utf8.h>

#include <algorithm>
#include <unordered_map>

#include "fastfeaturetransform/kernels/cast_table.hpp"
#include "fastfeaturetransform/kernels/categories.hpp"
#include "fastfeaturetransform/kernels/kernels.hpp"
#include "fastfeaturetransform/kernels/output.hpp"
#include "fastfeaturetransform/kernels/util.hpp"

namespace fastfeaturetransform {
namespace {

// ---------------------------------------------------------------------------
// select — column projection, in the given order. Stateless.
// Semantics: a missing requested column is an error at fit and at transform;
// the record path requires every requested key to be present.
// ---------------------------------------------------------------------------
class Select final : public Kernel {
 public:
  explicit Select(const nlohmann::json& params) : columns_(column_list(params, "columns", "select")) {
    for (std::size_t i = 0; i < columns_.size(); ++i) position_.emplace(columns_[i], i);
    // Known without a schema, so an artifact-loaded select declares what it
    // reads before any resolve() runs.
    inputs = columns_;
    outputs = columns_;
  }

  std::string id() const override { return "select"; }

  nlohmann::json params() const override { return {{"columns", columns_}}; }

  void resolve(Schema& schema) override {
    Schema out;
    for (const std::string& name : columns_) {
      const std::string* type = schema.find(name);
      if (type == nullptr) throw Error(stage + ": select column '" + name + "' not found in input");
      out.cols.emplace_back(name, *type);
    }
    schema = std::move(out);
  }

  Batch transform(Batch batch) const override {
    Batch out;
    out.rows = batch.rows;
    for (const std::string& name : columns_) {
      int idx = batch.find(name);
      if (idx < 0) throw Error(stage + ": select column '" + name + "' missing from batch");
      out.names.push_back(name);
      out.cols.push_back(batch.cols[static_cast<std::size_t>(idx)]);
    }
    return out;
  }

  Row transform_row(Row row) const override {
    // One pass over the record, placing each selected key at its output slot.
    Row out(columns_.size());
    std::vector<bool> found(columns_.size(), false);
    for (auto& [key, value] : row) {
      auto slot = position_.find(key);
      if (slot == position_.end()) continue;
      out[slot->second] = {key, std::move(value)};
      found[slot->second] = true;
    }
    for (std::size_t i = 0; i < columns_.size(); ++i) {
      if (!found[i]) throw Error(stage + ": select key '" + columns_[i] + "' missing from record");
    }
    return out;
  }

 private:
  std::vector<std::string> columns_;
  StringIndex position_;
};

// ---------------------------------------------------------------------------
// label_encode — categorical utf8 -> int64 index, in place. Semantics:
//   * category keys are byte-exact and must be valid UTF-8 (checked at fit,
//     because the artifact is JSON)
//   * known categories sort by byte order and take indices 1..n
//   * index 0 is reserved: unknown category, null, and categories dropped by
//     min_count all map to 0 (torch nn.Embedding padding_idx=0 convention)
//   * columns=null resolves to every utf8 column in the incoming schema
//   * on the record path: a missing key is an error; a non-string non-null
//     value is an error; null maps to 0
// ---------------------------------------------------------------------------
class LabelEncode final : public Kernel {
 public:
  explicit LabelEncode(const nlohmann::json& params) {
    if (params.contains("columns") && !params.at("columns").is_null()) {
      columns_ = column_list(params, "columns", "label_encode");
      explicit_columns_ = true;
      inputs = columns_;
      outputs = columns_;
    }
    if (params.contains("min_count")) {
      const nlohmann::json& min_count = params.at("min_count");
      if (!is_int64(min_count) || min_count.get<int64_t>() < 1) {
        throw Error("label_encode: 'min_count' must be an integer from 1 to 2^63 - 1");
      }
      min_count_ = min_count.get<int64_t>();
    }
  }

  std::string id() const override { return "label_encode"; }

  nlohmann::json params() const override {
    nlohmann::json out{{"min_count", min_count_}};
    out["columns"] = explicit_columns_ ? nlohmann::json(columns_) : nlohmann::json();
    return out;
  }

  void resolve(Schema& schema) override {
    bind_columns(schema, stage, columns_, explicit_columns_, fitted, "label-encode");
    for (auto& [name, type] : schema.cols) {
      if (std::find(columns_.begin(), columns_.end(), name) != columns_.end()) type = "i64";
    }
    inputs = columns_;
    outputs = columns_;
  }

  bool needs_fit() const override { return true; }

  std::unique_ptr<AccState> make_state() const override { return std::make_unique<SeenState>(); }

  void consume(AccState& state, const Batch& batch) const override {
    consume_categories(columns_, stage, static_cast<SeenState&>(state), batch);
  }

  void finalize(std::vector<std::unique_ptr<AccState>> states) override {
    arrow::util::InitializeUTF8();  // idempotent; ValidateUTF8 needs its tables
    const std::unique_ptr<SeenState> merged = merge_seen(states);
    maps_.clear();
    for (const std::string& name : columns_) {
      std::vector<std::string> keys;
      for (auto& [key, count] : merged->counts[name]) {
        if (count >= min_count_) keys.push_back(std::move(key));
      }
      merged->counts.erase(name);
      std::sort(keys.begin(), keys.end());
      auto& map = maps_[name];
      map.reserve(keys.size());
      for (std::size_t i = 0; i < keys.size(); ++i) {
        if (!arrow::util::ValidateUTF8(keys[i])) {
          throw Error(stage + ": column '" + name +
                      "' has a category that is not valid UTF-8; label_encode keys must be UTF-8 to serialize");
        }
        map.try_emplace(std::move(keys[i]), static_cast<int64_t>(i) + 1);
      }
    }
    index_maps();
    fitted = true;
  }

  nlohmann::json state() const override {
    nlohmann::json label_maps = nlohmann::json::object();
    nlohmann::json vocab_sizes = nlohmann::json::object();
    for (const std::string& name : columns_) {
      const auto& map = maps_.at(name);
      nlohmann::json entries = nlohmann::json::object();
      for (const auto& [key, index] : map) entries[key] = index;
      label_maps[name] = std::move(entries);
      vocab_sizes[name] = map.size() + 1;
    }
    return {{"columns", columns_}, {"label_maps", label_maps}, {"vocab_sizes", vocab_sizes}};
  }

  std::optional<int64_t> vocab_size(const std::string& column) const override {
    const auto found = maps_.find(column);
    if (found == maps_.end()) throw Error(stage + ": column '" + column + "' has no fitted vocabulary");
    return static_cast<int64_t>(found->second.size()) + 1;
  }

  // On artifact load resolve() never runs, so the state is the authority on
  // which columns this encoder owns and in what order; explicit params must
  // agree with it exactly. Indices must be exactly 1..n so vocab sizes and
  // the reserved 0 hold for hand-edited or foreign manifests too.
  void load_state(const nlohmann::json& state) override {
    if (!state.is_object() || !state.contains("label_maps") || !state.at("label_maps").is_object()) {
      throw Error(stage + ": state must hold a 'label_maps' object");
    }
    const nlohmann::json& label_maps = state.at("label_maps");
    load_fitted_columns(state, stage, columns_, explicit_columns_, &label_maps);

    maps_.clear();
    for (const std::string& name : columns_) {
      if (!label_maps.contains(name)) throw Error(stage + ": label_maps has no entry for column '" + name + "'");
      const nlohmann::json& entries = label_maps.at(name);
      if (!entries.is_object()) throw Error(stage + ": label_maps['" + name + "'] must be an object");
      auto& map = maps_[name];
      map.reserve(entries.size());
      std::vector<bool> seen(entries.size() + 1, false);
      for (const auto& [key, index] : entries.items()) {
        const int64_t n = static_cast<int64_t>(entries.size());
        if (!index.is_number_integer() || index.get<int64_t>() < 1 || index.get<int64_t>() > n ||
            seen[static_cast<std::size_t>(index.get<int64_t>())]) {
          throw Error(stage + ": column '" + name + "' label map indices must be exactly 1.." + std::to_string(n) +
                      " with no duplicates");
        }
        seen[static_cast<std::size_t>(index.get<int64_t>())] = true;
        map.emplace(key, index.get<int64_t>());
      }
    }
    // vocab_sizes repeats what label_maps holds, so it is only checked
    // against them, and it is required like every entry state() writes.
    const nlohmann::json& vocab_sizes = *per_column_object(state, "vocab_sizes", columns_, stage);
    for (const std::string& name : columns_) {
      if (vocab_sizes.at(name) != maps_.at(name).size() + 1) {
        throw Error(stage + ": column '" + name + "' vocab_sizes disagrees with label_maps");
      }
    }
    index_maps();
    inputs = columns_;
    outputs = columns_;
    fitted = true;
  }

  Batch transform(Batch batch) const override {
    for (const std::string& name : columns_) {
      int idx = batch.find(name);
      if (idx < 0) throw Error(stage + ": column '" + name + "' missing from batch");
      const auto& array = *batch.cols[static_cast<std::size_t>(idx)];
      const auto& map = maps_.at(name);
      const ShortValues<int64_t>& fast = short_maps_.at(name);
      PrimitiveOutput<arrow::Int64Type> output(array.length(), stage);
      if (array.type_id() == arrow::Type::DICTIONARY) {
        const auto& dictionary = static_cast<const arrow::DictionaryArray&>(array);
        const auto& entries = *dictionary.dictionary();
        constexpr int64_t max_entries = 8 * 1024 * 1024 / sizeof(int64_t);
        if (entries.length() <= array.length() && entries.length() <= max_entries) {
          // Label lookup cannot reject an unused category, so every dictionary
          // entry is coded up front, saving a cache-presence check per row;
          // the bound keeps a large, mostly unused dictionary off this path.
          std::vector<int64_t> codes(static_cast<std::size_t>(entries.length()), 0);
          for_each_dictionary_entry(entries, stage, name, [&](int64_t entry, bool is_null, std::string_view value) {
            if (is_null) return;
            if (const int64_t* found = fast.find(map, value)) codes[static_cast<std::size_t>(entry)] = *found;
          });
          std::vector<int32_t> copied;
          const int32_t* indices = int32_indices(dictionary, copied);
          const RowNulls null_at(dictionary);
          const int64_t length = dictionary.length();
          int64_t* out = output.mutable_values();
          for (int64_t row = 0; row < length; ++row) {
            out[row] = null_at(row) ? 0 : codes[static_cast<std::size_t>(indices[row])];
          }
          batch.cols[static_cast<std::size_t>(idx)] = output.finish();
          continue;
        }
      }
      for_each_encoded_utf8(
          array, stage, name,
          [&](bool is_null, std::string_view value) -> int64_t {
            if (is_null) return 0;
            const int64_t* found = fast.find(map, value);
            return found == nullptr ? 0 : *found;
          },
          [&](int64_t row, int64_t code) { output.mutable_values()[row] = code; });
      batch.cols[static_cast<std::size_t>(idx)] = output.finish();
    }
    return batch;
  }

  Row transform_row(Row row) const override {
    // One pass over the record: a hash lookup per key, whatever its width.
    std::size_t encoded = 0;
    for (auto& [key, value] : row) {
      auto map = maps_.find(key);
      if (map == maps_.end()) continue;
      ++encoded;
      if (std::holds_alternative<std::monostate>(value)) {
        value = static_cast<int64_t>(0);
      } else if (const std::string* text = std::get_if<std::string>(&value)) {
        auto found = map->second.find(*text);
        value = found == map->second.end() ? static_cast<int64_t>(0) : found->second;
      } else {
        throw Error(stage + ": key '" + key + "' must be a string or null");
      }
    }
    if (encoded != columns_.size()) {
      for (const std::string& name : columns_) {
        if (row_find(row, name) == nullptr) throw Error(stage + ": key '" + name + "' missing from record");
      }
    }
    return row;
  }

 private:
  std::vector<std::string> columns_;
  bool explicit_columns_ = false;
  // Each column's codes again, through a perfect hash when its keys allow.
  // The record path keeps its single lookup in maps_.
  void index_maps() {
    short_maps_.clear();
    for (const auto& [name, map] : maps_) short_maps_[name].index(map);
  }

  int64_t min_count_ = 1;
  std::unordered_map<std::string, StringCounts> maps_;
  std::unordered_map<std::string, ShortValues<int64_t>> short_maps_;
};

// ---------------------------------------------------------------------------
// cast — explicit, per-column type conversion. Stateless and row-wise.
// params: {"types": {column: target tag}} with targets from kCastTargets.
// Semantics: the cast table in cast_table.hpp, applied value by value on both
// paths; nulls pass through; the schema flowing out carries the target tags
// so downstream kernels resolve against the converted types.
// ---------------------------------------------------------------------------
class Cast final : public Kernel {
 public:
  explicit Cast(const nlohmann::json& params) {
    if (!params.contains("types") || !params.at("types").is_object() || params.at("types").empty()) {
      throw Error("cast: 'types' must be a non-empty mapping of column name to type tag");
    }
    for (const auto& [name, tag] : params.at("types").items()) {
      if (!tag.is_string()) throw Error("cast: the type for column '" + name + "' must be a type tag string");
      columns_.push_back(name);
      targets_.push_back(cast_target(tag.get<std::string>(), "cast: column '" + name + "'"));
    }
    inputs = columns_;
    outputs = columns_;
  }

  std::string id() const override { return "cast"; }

  nlohmann::json params() const override {
    nlohmann::json types = nlohmann::json::object();
    for (std::size_t i = 0; i < columns_.size(); ++i) types[columns_[i]] = targets_[i].tag;
    return {{"types", types}};
  }

  void resolve(Schema& schema) override {
    for (std::size_t i = 0; i < columns_.size(); ++i) {
      const std::string& name = columns_[i];
      const std::string where = stage + ": column '" + name + "'";
      const std::string* type = schema.find(name);
      if (type == nullptr) throw Error(where + " not found in input");
      const std::optional<Family> from = family_of_tag(*type);
      if (!from.has_value()) throw Error(where + " is " + *type + ", which cast cannot read");
      if (!cast_allowed(*from, targets_[i].family)) throw Error(cast_refusal(where, *from, targets_[i]));
      for (auto& [column, tag] : schema.cols) {
        if (column == name) tag = targets_[i].tag;
      }
    }
  }

  Batch transform(Batch batch) const override {
    for (std::size_t i = 0; i < columns_.size(); ++i) {
      int idx = batch.find(columns_[i]);
      if (idx < 0) throw Error(stage + ": column '" + columns_[i] + "' missing from batch");
      auto& column = batch.cols[static_cast<std::size_t>(idx)];
      column = cast_array(column, targets_[i], stage, columns_[i]);
    }
    return batch;
  }

  Row transform_row(Row row) const override {
    RecordLookup lookup(row, columns_.size());
    // Every message of the cast table begins with the place it is handed.
    // The table gets an empty one here, and "<stage>: key '<name>'" goes in
    // front only of a refusal, so a value that converts builds no message.
    // A refusal is thrown a second time for it, from the message rather than
    // what(), which ends at a NUL byte of the value.
    const std::string no_place;
    for (std::size_t i = 0; i < columns_.size(); ++i) {
      RowValue* value = lookup.find(columns_[i]);
      if (value == nullptr) throw Error(stage + ": key '" + columns_[i] + "' missing from record");
      try {
        *value = cast_value(std::move(*value), targets_[i], no_place);
      } catch (const Error& refusal) {
        throw Error(stage + ": key '" + columns_[i] + "'" + refusal.message);
      }
    }
    return row;
  }

 private:
  std::vector<std::string> columns_;
  std::vector<CastTarget> targets_;
};

}  // namespace

std::unique_ptr<Kernel> make_select(const nlohmann::json& params) { return std::make_unique<Select>(params); }

std::unique_ptr<Kernel> make_cast(const nlohmann::json& params) { return std::make_unique<Cast>(params); }

std::unique_ptr<Kernel> make_label_encode(const nlohmann::json& params) {
  return std::make_unique<LabelEncode>(params);
}

}  // namespace fastfeaturetransform
