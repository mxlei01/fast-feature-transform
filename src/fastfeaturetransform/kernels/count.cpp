// count_encode — replace a categorical column with how often each of its
// categories was seen at fit. The simplest of the category_encoders family:
// the fitted table is a plain per-category tally, so it is exact and does not
// depend on how the rows were split across workers.

#include <arrow/util/utf8.h>

#include <algorithm>
#include <cmath>
#include <ostream>
#include <unordered_map>

#include "fastfeaturetransform/kernels/cast_table.hpp"
#include "fastfeaturetransform/kernels/categories.hpp"
#include "fastfeaturetransform/kernels/kernels.hpp"
#include "fastfeaturetransform/kernels/output.hpp"
#include "fastfeaturetransform/kernels/util.hpp"

namespace fastfeaturetransform {
namespace {

// count_encode is the one encoder whose handle_unknown may also be an integer
// to fill with, so it wraps the shared option instead of redeclaring it. The
// fill is already in output units (the user asked for that literal number),
// which is why look_up has to say whether a value is one.
struct CountUnknown {
  Unknown kind = Unknown::Value;
  bool is_fill = false;
  int64_t fill = 0;
};

CountUnknown unknown_from_count(const nlohmann::json& params, const std::string& what) {
  CountUnknown unknown;
  if (!params.contains("handle_unknown") || params.at("handle_unknown").is_null()) return unknown;
  const nlohmann::json& value = params.at("handle_unknown");
  if (is_int64(value)) {
    unknown.is_fill = true;
    unknown.fill = value.get<int64_t>();
    return unknown;
  }
  const std::string message =
      what + ": 'handle_unknown' must be 'value', 'return_nan', 'error' or an integer from -2^63 to 2^63 - 1";
  if (!value.is_string()) throw Error(message);
  const std::string text = value.get<std::string>();
  if (text == "value") return unknown;
  if (text == "return_nan") {
    unknown.kind = Unknown::ReturnNan;
    return unknown;
  }
  if (text == "error") {
    unknown.kind = Unknown::Error;
    return unknown;
  }
  throw Error(message);
}

// ---------------------------------------------------------------------------
// count_encode — categorical utf8 -> its fit-time count, in place.
// params: {"columns": [...] | null, "normalize": bool,
//          "min_group_size": null | int | float, "min_group_name": null | str,
//          "combine_min_nan_groups": bool | "force",
//          "handle_unknown": "value" | "return_nan" | "error" | int,
//          "handle_missing": "value" | "return_nan" | "error"}
// Semantics:
//   * every category's count is exact and order-independent; normalize=true
//     divides by the fitted row count, so the output is f64 instead of i64
//   * a null is a countable category under handle_missing='value' (one not
//     seen at fit counts as unseen), a null under 'return_nan', and an error
//     under 'error', at transform as well as at fit
//   * an unseen category yields 0 under handle_unknown='value' (a null when
//     handle_missing is 'return_nan', as upstream does), that integer under
//     an integer, a null under 'return_nan', and an error under 'error'; an
//     integer fill is the literal number asked for, so normalize=true does
//     not divide it
//   * min_group_size folds every category below the threshold into one
//     group whose count is their sum, named min_group_name or the members
//     joined with '_' in sorted order, but only when the group gets more
//     than one member and some non-null category stays out of it; a group
//     name that is already a surviving category is a fit-time error rather
//     than a silent overwrite
//   * the null category joins that group when its count is below the
//     threshold too under combine_min_nan_groups=true, whenever there were
//     nulls under 'force', and never under false; joined, it counts as a
//     member, its count adds to the group's, the default group name spells
//     it 'nan', and a null answers the group's count
//   * columns=null encodes every utf8 column that reaches the step
// ---------------------------------------------------------------------------
class CountEncode final : public Kernel {
 public:
  explicit CountEncode(const nlohmann::json& params) {
    if (params.contains("columns") && !params.at("columns").is_null()) {
      columns_ = column_list(params, "columns", "count_encode");
      explicit_columns_ = true;
      inputs = columns_;
      outputs = columns_;
    }
    if (params.contains("normalize") && !params.at("normalize").is_null()) {
      if (!params.at("normalize").is_boolean()) throw Error("count_encode: 'normalize' must be true or false");
      normalize_ = params.at("normalize").get<bool>();
    }
    if (params.contains("min_group_size") && !params.at("min_group_size").is_null()) {
      const nlohmann::json& size = params.at("min_group_size");
      if (!size.is_number() || !std::isfinite(size.get<double>()) || size.get<double>() <= 0 ||
          (size.is_number_integer() && !is_int64(size))) {
        throw Error("count_encode: 'min_group_size' must be a finite number > 0, and an integer one at most 2^63 - 1");
      }
      min_group_size_ = size.get<double>();
      min_group_is_ratio_ = !size.is_number_integer();
      if (!min_group_is_ratio_) min_group_count_ = size.get<int64_t>();
    }
    if (params.contains("min_group_name") && !params.at("min_group_name").is_null()) {
      if (!params.at("min_group_name").is_string()) throw Error("count_encode: 'min_group_name' must be a string");
      min_group_name_ = params.at("min_group_name").get<std::string>();
    }
    if (params.contains("combine_min_nan_groups") && !params.at("combine_min_nan_groups").is_null()) {
      const nlohmann::json& combine = params.at("combine_min_nan_groups");
      if (combine.is_boolean()) {
        combine_nan_ = combine.get<bool>() ? CombineNan::Yes : CombineNan::No;
      } else if (combine.is_string() && combine.get<std::string>() == "force") {
        combine_nan_ = CombineNan::Force;
      } else {
        throw Error("count_encode: 'combine_min_nan_groups' must be true, false or 'force'");
      }
    }
    unknown_ = unknown_from_count(params, "count_encode");
    missing_ = missing_from(params, "count_encode");
    if (combine_nan_ == CombineNan::Force && missing_ == Missing::ReturnNan) {
      throw Error("count_encode: combine_min_nan_groups='force' needs handle_missing 'value' or 'error'");
    }
  }

  std::string id() const override { return "count_encode"; }

  nlohmann::json params() const override {
    nlohmann::json out;
    out["columns"] = explicit_columns_ ? nlohmann::json(columns_) : nlohmann::json();
    out["normalize"] = normalize_;
    out["min_group_size"] = min_group_size_ < 0   ? nlohmann::json()
                            : min_group_is_ratio_ ? nlohmann::json(min_group_size_)
                                                  : nlohmann::json(min_group_count_);
    out["min_group_name"] = min_group_name_.has_value() ? nlohmann::json(*min_group_name_) : nlohmann::json();
    out["combine_min_nan_groups"] =
        combine_nan_ == CombineNan::Force ? nlohmann::json("force") : nlohmann::json(combine_nan_ == CombineNan::Yes);
    out["handle_unknown"] =
        unknown_.is_fill ? nlohmann::json(unknown_.fill) : nlohmann::json(unknown_name(unknown_.kind));
    out["handle_missing"] = missing_name(missing_);
    return out;
  }

  void resolve(Schema& schema) override {
    // A fitted encoder's columns come from its state and are never re-derived.
    bind_columns(schema, stage, columns_, explicit_columns_, fitted, "count-encode");
    const std::string tag = normalize_ ? "f64" : "i64";
    for (auto& [name, type] : schema.cols) {
      if (std::find(columns_.begin(), columns_.end(), name) != columns_.end()) type = tag;
    }
    inputs = columns_;
    outputs = columns_;
  }

  bool needs_fit() const override { return true; }

  std::unique_ptr<AccState> make_state() const override { return std::make_unique<SeenState>(); }

  void consume(AccState& raw, const Batch& batch) const override {
    consume_categories(columns_, stage, static_cast<SeenState&>(raw), batch);
  }

  void finalize(std::vector<std::unique_ptr<AccState>> states) override {
    arrow::util::InitializeUTF8();  // idempotent; ValidateUTF8 needs its tables
    const std::unique_ptr<SeenState> merged = merge_seen(states);
    rows_ = merged->rows;
    tables_.clear();
    for (const std::string& name : columns_) {
      const int64_t nulls = merged->nulls[name];
      if (missing_ == Missing::Error && nulls > 0) {
        throw Error(stage + ": column '" + name + "' has nulls and handle_missing is 'error'");
      }
      // Sorted, so the fitted table is the same however the rows were split.
      std::vector<std::pair<std::string, int64_t>> tally;
      auto& counts = merged->counts[name];
      tally.reserve(counts.size());
      for (auto& [key, count] : counts) {
        if (!arrow::util::ValidateUTF8(key)) {
          throw Error(stage + ": column '" + name +
                      "' has a category that is not valid UTF-8; count_encode keys must be UTF-8 to serialize");
        }
        tally.emplace_back(key, count);
      }
      merged->counts.erase(name);
      std::sort(tally.begin(), tally.end());
      // What a null answers: its own count, or its group's once it joins one.
      int64_t null_count = missing_ == Missing::Value ? nulls : 0;
      combine_small(name, tally, null_count);
      Table& table = tables_[name];
      table.null_count = null_count;
      table.has_null_category = missing_ == Missing::Value && nulls > 0;
      for (auto& [key, count] : tally) {
        // emplace() keeps the first of a duplicate pair and drops the second
        // silently, so a colliding group name would corrupt a whole category.
        // Nothing may reach the table twice.
        if (table.counts.contains(key)) {
          throw Error(stage + ": column '" + name + "' tallies the category '" + key + "' twice");
        }
        table.counts.emplace(std::move(key), count);
      }
    }
    index_tables();
    fitted = true;
  }

  nlohmann::json state() const override {
    nlohmann::json counts = nlohmann::json::object();
    nlohmann::json null_counts = nlohmann::json::object();
    nlohmann::json groups = nlohmann::json::object();
    for (const std::string& name : columns_) {
      const Table& table = tables_.at(name);
      nlohmann::json entries = nlohmann::json::object();
      for (const auto& [key, count] : table.counts) entries[key] = count;
      counts[name] = std::move(entries);
      null_counts[name] = table.has_null_category ? nlohmann::json(table.null_count) : nlohmann::json();
      if (!table.members.empty()) groups[name] = groups_of(table);
    }
    return {
        {"columns", columns_}, {"counts", counts}, {"null_counts", null_counts}, {"groups", groups}, {"rows", rows_}};
  }

  // Write exactly what state() builds, one table at a time, without building
  // the whole state first.
  // nlohmann objects sort keys; matching that order and using its string
  // escaping makes this byte-for-byte identical to state().dump().
  void write_state(std::ostream& stream) const override {
    auto columns = columns_;
    std::sort(columns.begin(), columns.end());
    stream << "{\"columns\":" << nlohmann::json(columns_).dump() << ",\"counts\":{";
    bool first_column = true;
    for (const std::string& name : columns) {
      if (!first_column) stream << ',';
      first_column = false;
      stream << nlohmann::json(name).dump() << ":{";
      bool first = true;
      for (const auto* entry : sorted_entries(tables_.at(name).counts)) {
        if (!first) stream << ',';
        first = false;
        stream << nlohmann::json(entry->first).dump() << ':' << std::to_string(entry->second);
      }
      stream << '}';
    }
    stream << "},\"groups\":{";
    first_column = true;
    for (const std::string& name : columns) {
      const Table& table = tables_.at(name);
      if (table.members.empty()) continue;
      if (!first_column) stream << ',';
      first_column = false;
      stream << nlohmann::json(name).dump() << ':' << groups_of(table).dump();
    }
    stream << "},\"null_counts\":{";
    first_column = true;
    for (const std::string& name : columns) {
      if (!first_column) stream << ',';
      first_column = false;
      const Table& table = tables_.at(name);
      stream << nlohmann::json(name).dump() << ':'
             << (table.has_null_category ? std::to_string(table.null_count) : "null");
    }
    stream << "},\"rows\":" << std::to_string(rows_) << '}';
  }

  void load_state(const nlohmann::json& state) override {
    if (!state.is_object() || !state.contains("counts") || !state.at("counts").is_object() || !state.contains("rows") ||
        !state.at("rows").is_number_integer()) {
      throw Error(stage + ": state must hold a 'counts' object and an integer 'rows'");
    }
    rows_ = state.at("rows").get<int64_t>();
    if (rows_ < 0) throw Error(stage + ": 'rows' must not be negative");
    const nlohmann::json& counts = state.at("counts");
    load_fitted_columns(state, stage, columns_, explicit_columns_, &counts);
    // Both sub-objects are required in the shape state() writes: a missing or
    // wrong one would delete the null category or quietly turn a folded
    // category into an unknown. 'groups' names only the columns that fold a
    // category, so it holds no entry for the others.
    const nlohmann::json& null_counts = *per_column_object(state, "null_counts", columns_, stage);
    const nlohmann::json& groups = *per_column_object(state, "groups", {}, stage);
    for (const auto& [name, members] : groups.items()) {
      if (std::find(columns_.begin(), columns_.end(), name) == columns_.end()) {
        throw Error(stage + ": groups names column '" + name + "', which this encoder does not encode");
      }
      if (!members.is_object()) throw Error(stage + ": groups['" + name + "'] must be an object");
    }
    tables_.clear();
    for (const std::string& name : columns_) {
      if (!counts.contains(name) || !counts.at(name).is_object()) {
        throw Error(stage + ": counts['" + name + "'] must be an object");
      }
      Table& table = tables_[name];
      for (const auto& [key, count] : counts.at(name).items()) {
        if (!count.is_number_integer() || count.get<int64_t>() < 0) {
          throw Error(stage + ": counts['" + name + "']['" + key + "'] must be a count");
        }
        table.counts.emplace(key, count.get<int64_t>());
      }
      if (!null_counts.at(name).is_null()) {
        const nlohmann::json& count = null_counts.at(name);
        if (!count.is_number_integer() || count.get<int64_t>() < 0) {
          throw Error(stage + ": null_counts['" + name + "'] must be a count or null");
        }
        table.has_null_category = true;
        table.null_count = count.get<int64_t>();
      }
      if (!groups.contains(name)) continue;
      StringIndex group_ids;
      for (const auto& [key, value] : groups.at(name).items()) {
        // An entry is a group's name and the list of its members. Version
        // 0.0.1 saved a member and its group's name instead, and a string in
        // the list's place tells that layout apart.
        const bool listed =
            value.is_array() && !value.empty() &&
            std::all_of(value.begin(), value.end(), [](const nlohmann::json& member) { return member.is_string(); });
        if (!listed && !value.is_string()) {
          throw Error(stage + ": groups['" + name + "']['" + key + "'] must be a non-empty list of member names");
        }
        const std::string& into = listed ? key : value.get_ref<const std::string&>();
        const std::size_t size = listed ? value.size() : 1;
        // The group's name is looked up with the entry's first member and not
        // with each of them: the default name is as long as its members
        // together, and a lookup for each of 64,000 members of 8 bytes made a
        // load of 9 ms take 1.7 s (Apple M4, one thread).
        std::size_t group = 0;
        for (std::size_t i = 0; i < size; ++i) {
          const std::string& member = listed ? value[i].get_ref<const std::string&>() : key;
          // A member is folded away and a group is what it folds into, so the
          // one must be absent from the counts and the other present. Either
          // way round, a wrong entry would answer a fitted category as unseen.
          // A member can share its key with its own group, as a min_group_name
          // naming a folded category gives it: that key answers the group's
          // count either way.
          if (member != into && table.counts.contains(member)) {
            throw Error(stage + ": groups['" + name + "'] folds '" + member +
                        "', which is still a category of counts['" + name + "']");
          }
          if (i == 0) {
            auto [id, inserted] = group_ids.try_emplace(into, table.groups.size());
            if (inserted) {
              const auto count = table.counts.find(into);
              if (count == table.counts.end()) {
                throw Error(stage + ": groups['" + name + "'] folds '" + member + "' into '" + into +
                            "', which is not a category of counts['" + name + "']");
              }
              table.groups.push_back({into, count->second});
            }
            group = id->second;
          }
          // A member answers with one group's count, so it folds once.
          if (!table.members.emplace(member, group).second) {
            throw Error(stage + ": groups['" + name + "'] folds '" + member + "' more than once");
          }
        }
      }
    }
    index_tables();
    inputs = columns_;
    outputs = columns_;
    fitted = true;
  }

  Batch transform(Batch batch) const override {
    for (const std::string& name : columns_) {
      const int idx = batch.find(name);
      if (idx < 0) throw Error(stage + ": column '" + name + "' missing from batch");
      const arrow::Array& array = *batch.cols[static_cast<std::size_t>(idx)];
      const std::string where = stage + ": column '" + name + "'";
      const Table& table = tables_.at(name);
      if (normalize_) {
        PrimitiveOutput<arrow::DoubleType> values(array.length(), where);
        for_each_encoded_utf8(
            array, stage, name,
            [&](bool is_null, std::string_view value) -> std::optional<double> {
              const auto count = look_up(table, name, is_null, value);
              return count.has_value() ? std::optional<double>{as_double(*count)} : std::nullopt;
            },
            [&](int64_t row, const std::optional<double>& value) {
              if (value.has_value())
                values.mutable_values()[row] = *value;
              else
                values.set_null(row);
            });
        batch.cols[static_cast<std::size_t>(idx)] = values.finish();
      } else {
        PrimitiveOutput<arrow::Int64Type> values(array.length(), where);
        for_each_encoded_utf8(
            array, stage, name,
            [&](bool is_null, std::string_view value) { return look_up(table, name, is_null, value); },
            [&](int64_t row, const std::optional<Looked>& value) {
              if (value.has_value())
                values.mutable_values()[row] = value->value;
              else
                values.set_null(row);
            });
        batch.cols[static_cast<std::size_t>(idx)] = values.finish();
      }
    }
    return batch;
  }

  Row transform_row(Row row) const override {
    RecordLookup lookup(row, columns_.size());
    for (const std::string& name : columns_) {
      RowValue* value = lookup.find(name);
      if (value == nullptr) throw Error(stage + ": key '" + name + "' missing from record");
      const bool is_null = std::holds_alternative<std::monostate>(*value);
      const std::string* text = std::get_if<std::string>(value);
      if (!is_null && text == nullptr) throw Error(stage + ": key '" + name + "' must be a string or null");
      const std::optional<Looked> count = look_up(name, is_null, is_null ? std::string_view{} : *text);
      if (!count.has_value()) {
        *value = std::monostate{};
      } else if (normalize_) {
        *value = as_double(*count);
      } else {
        *value = count->value;
      }
    }
    return row;
  }

 private:
  template <typename Map>
  static std::vector<const typename Map::value_type*> sorted_entries(const Map& map) {
    std::vector<const typename Map::value_type*> entries;
    entries.reserve(map.size());
    for (const auto& entry : map) entries.push_back(&entry);
    std::sort(entries.begin(), entries.end(),
              [](const auto* left, const auto* right) { return left->first < right->first; });
    return entries;
  }

  enum class CombineNan { Yes, No, Force };

  struct Table {
    struct Group {
      std::string name;
      int64_t count;
    };
    StringCounts counts;
    StringIndex members;
    std::vector<Group> groups;
    // Every count a key answers with, the folded members' included, when the
    // keys allow a perfect hash. A member can share its key with the group it
    // folds into, which has the same count; it is indexed once.
    ShortValues<int64_t> short_counts;
    int64_t null_count = 0;
    bool has_null_category = false;
  };

  // One column's groups as the state holds them: each group's name with the
  // list of its members, sorted, so that the state is the same however the
  // rows were split and whichever layout the table was loaded from.
  // REGRESSION: the state holds a group's name once. It used to hold the name
  // once for each member, and the default name joins every member, so the
  // state grew with the square of the folded categories: a pipeline folding
  // 2,000 categories of 8 bytes each saved 36 MB, and now saves 58 KB.
  static nlohmann::json groups_of(const Table& table) {
    std::vector<std::vector<std::string>> members(table.groups.size());
    for (const auto& [member, group] : table.members) members[group].push_back(member);
    nlohmann::json groups = nlohmann::json::object();
    for (std::size_t group = 0; group < members.size(); ++group) {
      std::sort(members[group].begin(), members[group].end());
      groups[table.groups[group].name] = std::move(members[group]);
    }
    return groups;
  }

  void index_tables() {
    for (auto& [name, table] : tables_) {
      table.short_counts = {};
      // Only a member named like its group drops out, one per group at most,
      // so a table larger than this cannot be indexed and is not copied.
      if (table.counts.size() + table.members.size() > ShortKeys::kMaxKeys + table.groups.size()) continue;
      std::vector<std::pair<std::string_view, int64_t>> entries;
      for (const auto& [key, count] : table.counts) entries.emplace_back(key, count);
      for (const auto& [member, group] : table.members) {
        if (!table.counts.contains(member)) entries.emplace_back(member, table.groups[group].count);
      }
      table.short_counts.index(entries);
    }
  }

  double normalized(int64_t count) const {
    return rows_ == 0 ? 0.0 : static_cast<double>(count) / static_cast<double>(rows_);
  }

  // What one value encodes to. A count is in fit units and normalize=true
  // divides it; an integer handle_unknown is already the number the user
  // asked to see, so it passes through both paths untouched.
  struct Looked {
    int64_t value = 0;
    bool is_fill = false;
  };

  // The f64 the normalize=true schema promises for one looked-up value.
  double as_double(const Looked& looked) const {
    return looked.is_fill ? static_cast<double>(looked.value) : normalized(looked.value);
  }

  // The encoded value for one input value, or nullopt when the output is null.
  std::optional<Looked> look_up(const std::string& name, bool is_null, std::string_view value) const {
    return look_up(tables_.at(name), name, is_null, value);
  }

  std::optional<Looked> look_up(const Table& table, const std::string& name, bool is_null,
                                std::string_view value) const {
    if (is_null) {
      if (missing_ == Missing::ReturnNan) return std::nullopt;
      refuse_null(missing_, stage, name);
      if (table.has_null_category) return Looked{table.null_count, false};
      return unknown_value(name);
    }
    if (!table.short_counts.keys.empty()) {
      const int64_t at = table.short_counts.keys.find(value);
      if (at >= 0) return Looked{table.short_counts.values[static_cast<std::size_t>(at)], false};
      return unknown_value(name);
    }
    const auto member = table.members.find(value);
    if (member != table.members.end()) return Looked{table.groups[member->second].count, false};
    auto found = table.counts.find(value);
    if (found != table.counts.end()) return Looked{found->second, false};
    return unknown_value(name);
  }

  std::optional<Looked> unknown_value(const std::string& name) const {
    if (unknown_.is_fill) return Looked{unknown_.fill, true};
    switch (unknown_.kind) {
      case Unknown::Value:
        // With handle_missing='return_nan' upstream skips the column's
        // replace(np.nan, 0), so an unseen category is a null too.
        return missing_ == Missing::ReturnNan ? std::optional<Looked>{} : std::optional<Looked>{Looked{0, false}};
      case Unknown::ReturnNan:
        return std::nullopt;
      case Unknown::Indicator:
      case Unknown::Error:
        break;
    }
    throw Error(stage + ": column '" + name + "' has a category that was not seen at fit");
  }

  // Fold every category below the threshold into one, exactly when upstream
  // does: the group gets more than one member, their total is positive, and
  // some non-null category stays out. The null category joins them when its
  // count is below the threshold too (combine_min_nan_groups=true) or
  // whenever there were nulls ('force'), and never under false. Then it is
  // one of the members, its count adds to theirs, the default name spells it
  // 'nan' as upstream's str() does, and a null answers the group's count,
  // which null_count becomes.
  void combine_small(const std::string& name, std::vector<std::pair<std::string, int64_t>>& tally,
                     int64_t& null_count) {
    if (min_group_size_ < 0) return;
    // Upstream compares what it encodes with: a count against a size, a
    // fractional one times the rows; under normalize a ratio, count / rows,
    // against the size, an integer one divided by the rows. With rounding,
    // the two can disagree where a count sits on the threshold.
    const auto rows = static_cast<double>(rows_);
    const auto below = [&](int64_t count) {
      if (normalize_) {
        return static_cast<double>(count) / rows < (min_group_is_ratio_ ? min_group_size_ : min_group_size_ / rows);
      }
      return static_cast<double>(count) < (min_group_is_ratio_ ? min_group_size_ * rows : min_group_size_);
    };
    std::vector<std::size_t> small;
    for (std::size_t i = 0; i < tally.size(); ++i) {
      if (below(tally[i].second)) small.push_back(i);
    }
    const bool null_joins =
        null_count > 0 && (combine_nan_ == CombineNan::Force || (combine_nan_ == CombineNan::Yes && below(null_count)));
    int64_t total = null_joins ? null_count : 0;
    for (std::size_t i : small) total += tally[i].second;
    const std::size_t members = small.size() + (null_joins ? 1 : 0);
    if (total <= 0 || members < 2 || small.size() == tally.size()) return;

    std::vector<std::string> names;
    for (std::size_t i : small) names.push_back(tally[i].first);
    std::string group = min_group_name_.has_value() ? *min_group_name_ : join_sorted(names, null_joins);
    std::vector<std::pair<std::string, int64_t>> kept;
    kept.reserve(tally.size() - small.size() + 1);
    std::size_t next_small = 0;
    for (std::size_t i = 0; i < tally.size(); ++i) {
      if (next_small < small.size() && small[next_small] == i)
        ++next_small;
      else
        kept.push_back(std::move(tally[i]));
    }
    // A group named like a category that survived the fold would claim that
    // category's key. Upstream's mapper[group] = total silently overwrites the
    // real count, so every row carrying the category reads as the group total;
    // the default group name is the members joined with '_', so a column
    // holding 'x', 'y' and 'x_y' reaches this with no user error at all.
    for (const auto& [key, count] : kept) {
      if (key == group) {
        throw Error(stage + ": column '" + name + "': the folded group is named '" + group +
                    "', which is already a category of the column; set min_group_name to a name no category uses");
      }
    }
    Table& table = tables_[name];
    table.members.reserve(names.size());
    table.groups.push_back({group, total});
    for (std::string& member : names) table.members.emplace(std::move(member), 0);
    kept.emplace_back(std::move(group), total);
    std::sort(kept.begin(), kept.end());
    tally = std::move(kept);
    if (null_joins) null_count = total;
  }

  // The members joined with '_' in sorted order, the null among them as 'nan'.
  // As in upstream's '_'.join, an empty member keeps its separator: "" and 'x'
  // make '_x'.
  static std::string join_sorted(std::vector<std::string> names, bool with_null) {
    // The selected members already follow the sorted tally's order.
    if (with_null) names.insert(std::upper_bound(names.begin(), names.end(), std::string("nan")), "nan");
    std::string joined;
    for (std::size_t i = 0; i < names.size(); ++i) {
      if (i > 0) joined += '_';
      joined += names[i];
    }
    return joined;
  }

  std::vector<std::string> columns_;
  bool explicit_columns_ = false;
  bool normalize_ = false;
  double min_group_size_ = -1;  // < 0 = not set
  bool min_group_is_ratio_ = false;
  // An integer min_group_size as given: near 2^63 its double rounds up to
  // 2^63, which an int64 cannot hold, so params() writes this instead.
  int64_t min_group_count_ = 0;
  std::optional<std::string> min_group_name_;
  CombineNan combine_nan_ = CombineNan::Yes;
  CountUnknown unknown_;
  Missing missing_ = Missing::Value;
  int64_t rows_ = 0;
  std::unordered_map<std::string, Table> tables_;
};

}  // namespace

std::unique_ptr<Kernel> make_count_encode(const nlohmann::json& params) {
  return std::make_unique<CountEncode>(params);
}

}  // namespace fastfeaturetransform
