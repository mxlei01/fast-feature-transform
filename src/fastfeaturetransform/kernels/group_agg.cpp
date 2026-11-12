// group_agg: per-key aggregates, fitted once over the training data and
// looked up at transform. It replaces a column with several new ones, and its
// median is a buffered (O(rows)) aggregate.

#include <arrow/util/utf8.h>

#include <algorithm>
#include <cmath>
#include <format>
#include <limits>
#include <optional>
#include <unordered_map>

#include "fastfeaturetransform/kernels/cast_table.hpp"
#include "fastfeaturetransform/kernels/kernels.hpp"
#include "fastfeaturetransform/kernels/output.hpp"
#include "fastfeaturetransform/kernels/util.hpp"
#include "fastfeaturetransform/spill.hpp"

namespace fastfeaturetransform {
namespace {

// ---------------------------------------------------------------------------
// group_agg — per-key aggregates, fitted once and looked up at transform.
// params: {"key": column, "aggs": {column: [agg, ...]}} with aggs from kAggNames.
// Semantics:
//   * the key is a utf8 or integer column; a null key contributes nothing at
//     fit and reads as unknown at transform
//   * count counts a column's non-null values per key and works on any
//     column; sum, mean, min, max, and median need an integer or float
//     column and skip its nulls
//   * types stay explicit: count is i64; sum is i64 for integer columns (an
//     overflow is an error) and f64 for floats (a summed value of 2^900 or
//     more in magnitude is an error, so the sum itself never overflows); mean
//     is the exact sum in f64 divided by the count, median is computed in
//     f64; min and max keep the column's type; NaN or infinity in a column
//     with any aggregate but count is an error, because the fitted table is
//     JSON, and -0.0 reads as 0.0
//   * each aggregated column is replaced by one output per agg, named
//     <column>_<agg> and appended after the remaining columns in
//     column-name order and then the listed agg order (as label_encode
//     replaces a column's values, this replaces the column); the key stays
//   * an unknown or null key, and a key whose values were all null, yield 0
//     for every aggregate (the label_encode index-0 convention)
//   * transform reads only the key: a fitted kernel lists just the key as
//     its input, so the aggregated columns may be absent at transform
//   * the streaming aggregates keep O(keys) memory per worker; exact median
//     values spill under the shared fit budget and use external sorting
//   * integer sums are 128-bit and float sums exact (Shewchuk partials), so
//     the table is the same however row groups were split across workers
//   * on the record path: a missing key is an error; the key value must
//     match the fitted key type (string or integer) or be null
// ---------------------------------------------------------------------------
enum class Agg { Count, Sum, Mean, Min, Max, Median };
enum class KeyKind { Utf8, Integer };

const char* kAggNames = "count, sum, mean, min, max, median";

const char* agg_name(Agg agg) {
  switch (agg) {
    case Agg::Count:
      return "count";
    case Agg::Sum:
      return "sum";
    case Agg::Mean:
      return "mean";
    case Agg::Min:
      return "min";
    case Agg::Max:
      return "max";
    case Agg::Median:
      return "median";
  }
  return "";
}

Agg agg_from_name(const std::string& name, const std::string& what) {
  static const std::unordered_map<std::string, Agg> kAggs{
      {"count", Agg::Count}, {"sum", Agg::Sum}, {"mean", Agg::Mean},
      {"min", Agg::Min},     {"max", Agg::Max}, {"median", Agg::Median},
  };
  auto found = kAggs.find(name);
  if (found == kAggs.end()) {
    throw Error(what + ": unknown aggregate '" + name + "' (choose from " + kAggNames + ")");
  }
  return found->second;
}

const char* key_kind_name(KeyKind kind) { return kind == KeyKind::Utf8 ? "utf8" : "integer"; }

KeyKind key_kind_of_tag(const std::string& tag, const std::string& where) {
  if (tag == "utf8") return KeyKind::Utf8;
  if (family_of_tag(tag) == Family::Integer) return KeyKind::Integer;
  throw Error(where + " is " + tag + ", expected utf8 or an integer column");
}

__extension__ typedef __int128 Int128;  // NOLINT(modernize-use-using): __extension__ needs the typedef form

// One (key, column) accumulator. The integer members serve integer columns
// (the 128-bit sum cannot overflow before 2^64 values; the 64-bit range is
// checked once at finalize) and the float ones float columns.
struct Acc {
  Int128 isum = 0;
  int64_t count = 0;
  int64_t imin = std::numeric_limits<int64_t>::max();
  int64_t imax = std::numeric_limits<int64_t>::min();
  double fmin = std::numeric_limits<double>::infinity();
  double fmax = -std::numeric_limits<double>::infinity();
  ExactSum fsum;
};

// The median of a buffer (reordered in place); the midpoint of the two
// middle values for an even count, as pandas does. (lower + upper) / 2
// rounds once (the addition is exact under cancellation and in the
// subnormal range); the halves are the fallback when the addition overflows.
double median_of(SpillVector<double>& values) {
  if (values.empty()) return 0.0;
  const std::size_t middle = values.size() / 2;
  double upper;
  double lower = 0.0;
  if (values.spilled()) {
    values.sort();
    upper = values[middle];
    if (values.size() % 2 == 0) lower = values[middle - 1];
  } else {
    const auto memory = values.mutable_memory();
    auto at = memory.begin() + static_cast<std::ptrdiff_t>(middle);
    std::nth_element(memory.begin(), at, memory.end());
    upper = *at;
    if (values.size() % 2 == 0) lower = *std::max_element(memory.begin(), at);
  }
  if (values.size() % 2 == 1) return upper;
  const double midpoint = (lower + upper) / 2;
  return std::isfinite(midpoint) ? midpoint : lower / 2 + upper / 2;
}

using IntIndex = ankerl::unordered_dense::map<int64_t, std::size_t>;

class GroupAgg final : public Kernel {
 public:
  explicit GroupAgg(const nlohmann::json& params) {
    if (!params.contains("key") || !params.at("key").is_string() || params.at("key").get<std::string>().empty()) {
      throw Error("group_agg: 'key' must be a column name");
    }
    key_ = params.at("key").get<std::string>();
    if (!params.contains("aggs") || !params.at("aggs").is_object() || params.at("aggs").empty()) {
      throw Error("group_agg: 'aggs' must be a non-empty mapping of column name to a list of aggregates");
    }
    for (const auto& [name, list] : params.at("aggs").items()) {
      const std::string what = "group_agg: column '" + name + "'";
      if (name.empty()) throw Error("group_agg: 'aggs' has an empty column name");
      if (!list.is_array() || list.empty()) throw Error(what + ": aggregates must be a non-empty list of names");
      Column column;
      column.name = name;
      for (const auto& item : list) {
        if (!item.is_string()) throw Error(what + ": aggregates must be a non-empty list of names");
        const Agg agg = agg_from_name(item.get<std::string>(), what);
        if (std::find(column.aggs.begin(), column.aggs.end(), agg) != column.aggs.end()) {
          throw Error(what + ": aggregate '" + agg_name(agg) + "' listed more than once");
        }
        column.aggs.push_back(agg);
      }
      auto wants = [&](Agg agg) { return std::find(column.aggs.begin(), column.aggs.end(), agg) != column.aggs.end(); };
      column.count_only = column.aggs.size() == 1 && column.aggs[0] == Agg::Count;
      column.sums = wants(Agg::Sum) || wants(Agg::Mean);
      column.minmax = wants(Agg::Min) || wants(Agg::Max);
      if (wants(Agg::Median)) column.median = static_cast<int>(medians_++);
      columns_.push_back(std::move(column));
    }
    // Fit reads the key and every aggregated column; once fitted, transform
    // reads the key alone (finalize and load_state narrow the inputs).
    inputs.push_back(key_);
    for (const Column& column : columns_) {
      if (column.name != key_) inputs.push_back(column.name);
      for (Agg agg : column.aggs) outputs.push_back(column.name + "_" + agg_name(agg));
    }
  }

  std::string id() const override { return "group_agg"; }

  nlohmann::json params() const override {
    nlohmann::json aggs = nlohmann::json::object();
    for (const Column& column : columns_) {
      nlohmann::json names = nlohmann::json::array();
      for (Agg agg : column.aggs) names.push_back(agg_name(agg));
      aggs[column.name] = std::move(names);
    }
    return {{"key", key_}, {"aggs", aggs}};
  }

  void resolve(Schema& schema) override {
    const std::string* key_type = schema.find(key_);
    if (key_type == nullptr) throw Error(stage + ": key column '" + key_ + "' not found in input");
    const KeyKind kind = key_kind_of_tag(*key_type, stage + ": key column '" + key_ + "'");
    if (fitted && kind != key_kind_) {
      throw Error(stage + ": key column '" + key_ + "' is " + *key_type + ", but the fitted table has " +
                  key_kind_name(key_kind_) + " keys");
    }
    if (!fitted) {
      key_kind_ = kind;
      // The aggregated columns are read at fit only; a fitted table needs
      // just the key, whatever else the input holds.
      for (Column& column : columns_) {
        const std::string where = stage + ": column '" + column.name + "'";
        const std::string* type = schema.find(column.name);
        if (type == nullptr) throw Error(where + " not found in input");
        require_numeric(column, *type, where);
        column.tag = *type;
        column.family = family_of_tag(*type).value_or(Family::Utf8);
      }
    }
    for (const std::string& name : outputs) {
      if (schema.find(name) != nullptr) throw Error(stage + ": output column '" + name + "' already exists in input");
    }
    // Each aggregated column is replaced by its aggregates; the key stays.
    std::erase_if(schema.cols, [&](const auto& col) { return replaced(col.first); });
    for (const Column& column : columns_) {
      for (Agg agg : column.aggs) schema.cols.emplace_back(column.name + "_" + agg_name(agg), output_tag(column, agg));
    }
  }

  bool needs_fit() const override { return true; }

  struct State final : AccState {
    StringIndex str_slots;
    IntIndex int_slots;
    std::size_t slots = 0;
    std::size_t columns = 0;
    std::size_t medians = 0;
    std::vector<Acc> accs;                     // slots * columns
    std::vector<SpillVector<double>> buffers;  // slots * medians

    std::size_t add_slot() {
      ++slots;
      accs.resize(slots * columns);
      buffers.resize(slots * medians);
      return slots - 1;
    }
  };

  std::unique_ptr<AccState> make_state() const override {
    auto state = std::make_unique<State>();
    state->columns = columns_.size();
    state->medians = medians_;
    return state;
  }

  void consume(AccState& raw, const Batch& batch) const override {
    auto& state = static_cast<State&>(raw);
    const std::vector<int64_t> slots = key_slots(
        column_array(batch, stage, key_),
        [&](std::string_view key) -> int64_t {
          auto found = state.str_slots.find(key);
          if (found != state.str_slots.end()) return static_cast<int64_t>(found->second);
          const std::size_t slot = state.add_slot();
          state.str_slots.emplace(std::string(key), slot);
          return static_cast<int64_t>(slot);
        },
        [&](int64_t key) -> int64_t {
          auto [entry, inserted] = state.int_slots.try_emplace(key, 0);
          if (inserted) entry->second = state.add_slot();
          return static_cast<int64_t>(entry->second);
        });

    for (std::size_t c = 0; c < columns_.size(); ++c) {
      const Column& column = columns_[c];
      const arrow::Array& array = column_array(batch, stage, column.name);
      const std::string where = stage + ": column '" + column.name + "'";
      auto acc_at = [&](int64_t row) -> Acc* {
        const int64_t slot = slots[static_cast<std::size_t>(row)];
        return slot < 0 ? nullptr : &state.accs[static_cast<std::size_t>(slot) * columns_.size() + c];
      };
      auto buffer_at = [&](int64_t row) -> SpillVector<double>& {
        const auto slot = static_cast<std::size_t>(slots[static_cast<std::size_t>(row)]);
        return state.buffers[slot * medians_ + static_cast<std::size_t>(column.median)];
      };
      if (column.count_only) {  // the null bitmap is all it needs, whatever the type
        for (int64_t row = 0; row < array.length(); ++row) {
          if (array.IsNull(row)) continue;
          if (Acc* acc = acc_at(row); acc != nullptr) ++acc->count;
        }
      } else if (column.family == Family::Integer) {
        for_each_integer(array, stage, column.name, [&](int64_t row, bool is_null, int64_t value) {
          if (is_null) return;
          Acc* acc = acc_at(row);
          if (acc == nullptr) return;
          ++acc->count;
          if (column.sums) acc->isum += value;
          if (column.minmax) {
            acc->imin = std::min(acc->imin, value);
            acc->imax = std::max(acc->imax, value);
          }
          if (column.median >= 0) buffer_at(row).push_back(static_cast<double>(value));
        });
      } else {
        for_each_floating(array, stage, column.name, [&](int64_t row, bool is_null, double value) {
          if (is_null) return;
          Acc* acc = acc_at(row);
          if (acc == nullptr) return;
          if (!std::isfinite(value)) {
            throw Error(where + ": value " + std::format("{}", value) +
                        " is not finite; group_agg cannot aggregate NaN or infinity");
          }
          if (value == 0.0) value = 0.0;  // one zero: -0.0 would make min, max, and median depend on the split
          ++acc->count;
          if (column.sums) {
            // Bounding each value keeps every running sum inside f64 whatever
            // the split, so a sum can never fail on one thread count and
            // pass on another.
            if (std::fabs(value) >= kMaxSummand) {
              throw Error(where + ": value " + std::format("{}", value) +
                          " is 2^900 or more in magnitude, more than a sum can hold (scale the column first)");
            }
            if (!acc->fsum.add(value)) throw Error(overflow_f64(column));
          }
          if (column.minmax) {
            acc->fmin = std::min(acc->fmin, value);
            acc->fmax = std::max(acc->fmax, value);
          }
          if (column.median >= 0) buffer_at(row).push_back(value);
        });
      }
    }
  }

  void finalize(std::vector<std::unique_ptr<AccState>> states) override {
    // Fold every worker into the first one, freeing each as it is consumed.
    std::unique_ptr<State> merged;
    for (auto& raw : states) {
      if (raw == nullptr) continue;
      if (merged == nullptr) {
        merged.reset(static_cast<State*>(raw.release()));
        continue;
      }
      merge_into(*merged, static_cast<State&>(*raw));
      raw.reset();
    }
    if (merged == nullptr) merged = std::make_unique<State>();

    // Keys in sorted order: the table never depends on the thread count.
    const std::size_t n = merged->slots;
    std::vector<std::size_t> order(n);
    str_keys_.clear();
    int_keys_.clear();
    if (key_kind_ == KeyKind::Utf8) {
      std::vector<std::string> keys(n);
      for (auto& [key, slot] : merged->str_slots) keys[slot] = std::move(key);
      merged->str_slots.clear();
      for (std::size_t i = 0; i < n; ++i) order[i] = i;
      std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return keys[a] < keys[b]; });
      str_keys_.reserve(n);
      for (std::size_t slot : order) str_keys_.push_back(std::move(keys[slot]));
      arrow::util::InitializeUTF8();  // idempotent; ValidateUTF8 needs its tables
      for (const std::string& key : str_keys_) {
        if (!arrow::util::ValidateUTF8(key)) {
          throw Error(stage + ": key column '" + key_ +
                      "' has a key that is not valid UTF-8; group_agg keys must be UTF-8 to serialize");
        }
      }
    } else {
      std::vector<int64_t> keys(n);
      for (const auto& [key, slot] : merged->int_slots) keys[slot] = key;
      merged->int_slots.clear();
      for (std::size_t i = 0; i < n; ++i) order[i] = i;
      std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return keys[a] < keys[b]; });
      int_keys_.reserve(n);
      for (std::size_t slot : order) int_keys_.push_back(keys[slot]);
    }

    features_.clear();
    for (std::size_t c = 0; c < columns_.size(); ++c) {
      const Column& column = columns_[c];
      for (Agg agg : column.aggs) {
        Feature feature = make_feature(column, agg);
        for (std::size_t slot : order) {
          const Acc& acc = merged->accs[slot * columns_.size() + c];
          SpillVector<double>* buffer =
              column.median >= 0 ? &merged->buffers[slot * medians_ + static_cast<std::size_t>(column.median)]
                                 : nullptr;
          push_value(feature, column, agg, acc, buffer);
          if (agg == Agg::Median && buffer != nullptr) buffer->release();
        }
        features_.push_back(std::move(feature));
      }
    }
    merged.reset();
    index_keys();
    inputs = {key_};
    fitted = true;
  }

  nlohmann::json state() const override {
    nlohmann::json column_types = nlohmann::json::object();
    for (const Column& column : columns_) column_types[column.name] = column.tag;
    nlohmann::json values = nlohmann::json::object();
    for (const Feature& feature : features_) {
      values[feature.name] = feature.floating ? nlohmann::json(feature.doubles) : nlohmann::json(feature.ints);
    }
    return {{"key_type", key_kind_ == KeyKind::Utf8 ? "utf8" : "i64"},
            {"keys", key_kind_ == KeyKind::Utf8 ? nlohmann::json(str_keys_) : nlohmann::json(int_keys_)},
            {"column_types", column_types},
            {"values", values}};
  }

  // resolve() never runs on load, so the state carries the key type and each
  // aggregated column's type; every list is checked against the key count
  // and the feature's type before the table is trusted.
  void load_state(const nlohmann::json& state) override {
    const std::string bad = stage + ": state must hold 'key_type', sorted 'keys', 'column_types', and 'values'";
    if (!state.is_object() || !state.contains("key_type") || !state.at("key_type").is_string() ||
        !state.contains("keys") || !state.at("keys").is_array() || !state.contains("column_types") ||
        !state.at("column_types").is_object() || !state.contains("values") || !state.at("values").is_object()) {
      throw Error(bad);
    }
    const std::string key_type = state.at("key_type").get<std::string>();
    if (key_type != "utf8" && key_type != "i64") throw Error(stage + ": key_type must be 'utf8' or 'i64'");
    key_kind_ = key_type == "utf8" ? KeyKind::Utf8 : KeyKind::Integer;
    str_keys_.clear();
    int_keys_.clear();
    for (const nlohmann::json& key : state.at("keys")) {
      if (key_kind_ == KeyKind::Utf8) {
        if (!key.is_string()) throw Error(stage + ": keys must all be strings");
        str_keys_.push_back(key.get<std::string>());
      } else {
        if (!key.is_number_integer() || (key.is_number_unsigned() && key.get<uint64_t>() > INT64_MAX)) {
          throw Error(stage + ": keys must all be 64-bit integers");
        }
        int_keys_.push_back(key.get<int64_t>());
      }
    }
    const std::size_t n = std::max(str_keys_.size(), int_keys_.size());
    const bool sorted = key_kind_ == KeyKind::Utf8
                            ? std::is_sorted(str_keys_.begin(), str_keys_.end()) &&
                                  std::adjacent_find(str_keys_.begin(), str_keys_.end()) == str_keys_.end()
                            : std::is_sorted(int_keys_.begin(), int_keys_.end()) &&
                                  std::adjacent_find(int_keys_.begin(), int_keys_.end()) == int_keys_.end();
    if (!sorted) throw Error(stage + ": keys must be sorted and unique");

    const nlohmann::json& column_types = state.at("column_types");
    for (Column& column : columns_) {
      const std::string where = stage + ": column '" + column.name + "'";
      if (!column_types.contains(column.name) || !column_types.at(column.name).is_string()) {
        throw Error(where + " has no type in column_types");
      }
      const std::string tag = column_types.at(column.name).get<std::string>();
      require_numeric(column, tag, where);
      column.tag = tag;
      column.family = family_of_tag(tag).value_or(Family::Utf8);
    }

    features_.clear();
    const nlohmann::json& values = state.at("values");
    for (const Column& column : columns_) {
      for (Agg agg : column.aggs) {
        Feature feature = make_feature(column, agg);
        const std::string where = stage + ": values['" + feature.name + "']";
        if (!values.contains(feature.name) || !values.at(feature.name).is_array() ||
            values.at(feature.name).size() != n) {
          throw Error(where + " must be a list with one entry per key");
        }
        for (const nlohmann::json& value : values.at(feature.name)) {
          if (feature.floating) {
            if (!value.is_number() || !std::isfinite(value.get<double>())) {
              throw Error(where + " must hold finite numbers");
            }
            const double number = value.get<double>();
            if (feature.tag == "f32" && floating_in_width(number, cast_target("f32", where), where) != number) {
              throw Error(where + " must hold f32 values");
            }
            feature.doubles.push_back(number);
          } else {
            if (!value.is_number_integer() || (value.is_number_unsigned() && value.get<uint64_t>() > INT64_MAX)) {
              throw Error(where + " must hold 64-bit integers");
            }
            feature.ints.push_back(integer_in_range(value.get<int64_t>(), cast_target(feature.tag, where), where));
          }
        }
        features_.push_back(std::move(feature));
      }
    }
    index_keys();
    inputs = {key_};
    fitted = true;
  }

  Batch transform(Batch batch) const override {
    const std::vector<int64_t> slots = key_slots(
        column_array(batch, stage, key_),
        [&](std::string_view key) -> int64_t {
          if (!str_short_.empty()) return str_short_.find(key);
          auto found = str_index_.find(key);
          return found == str_index_.end() ? -1 : static_cast<int64_t>(found->second);
        },
        [&](int64_t key) -> int64_t {
          auto found = int_index_.find(key);
          return found == int_index_.end() ? -1 : static_cast<int64_t>(found->second);
        });
    drop_replaced(batch);
    const auto rows = static_cast<int64_t>(slots.size());
    for (const Feature& feature : features_) {
      const std::string where = stage + ": column '" + feature.name + "'";
      // Gathered as i64 or f64, then cast to the feature's own type, which
      // only the min or max of a narrower column changes. The cast cannot
      // fail: the fit and load_state hold every value to that type.
      std::shared_ptr<arrow::Array> array;
      if (feature.floating) {
        PrimitiveOutput<arrow::DoubleType> values(rows, where);
        double* out = values.mutable_values();
        for (int64_t row = 0; row < rows; ++row) {
          const int64_t slot = slots[static_cast<std::size_t>(row)];
          out[row] = slot < 0 ? 0.0 : feature.doubles[static_cast<std::size_t>(slot)];
        }
        array = values.finish();
      } else {
        PrimitiveOutput<arrow::Int64Type> values(rows, where);
        int64_t* out = values.mutable_values();
        for (int64_t row = 0; row < rows; ++row) {
          const int64_t slot = slots[static_cast<std::size_t>(row)];
          out[row] = slot < 0 ? 0 : feature.ints[static_cast<std::size_t>(slot)];
        }
        array = values.finish();
      }
      batch.names.push_back(feature.name);
      batch.cols.push_back(cast_array(array, cast_target(feature.tag, where), stage, feature.name));
    }
    return batch;
  }

  Row transform_row(Row row) const override {
    const RowValue* key = row_find(row, key_);
    if (key == nullptr) throw Error(stage + ": key '" + key_ + "' missing from record");
    int64_t slot = -1;
    if (std::holds_alternative<std::monostate>(*key)) {
      slot = -1;
    } else if (key_kind_ == KeyKind::Utf8) {
      const std::string* text = std::get_if<std::string>(key);
      if (text == nullptr) throw Error(stage + ": key '" + key_ + "' must be a string or null");
      auto found = str_index_.find(*text);
      if (found != str_index_.end()) slot = static_cast<int64_t>(found->second);
    } else {
      const int64_t* number = std::get_if<int64_t>(key);
      if (number == nullptr) throw Error(stage + ": key '" + key_ + "' must be an integer or null");
      auto found = int_index_.find(*number);
      if (found != int_index_.end()) slot = static_cast<int64_t>(found->second);
    }
    // One pass over the record for collisions, then append: the record path
    // stays linear in the record plus the outputs.
    for (const auto& [name, value] : row) {
      if (feature_index_.contains(name)) throw Error(stage + ": key '" + name + "' already present in record");
    }
    std::erase_if(row, [&](const auto& entry) { return replaced(entry.first); });
    row.reserve(row.size() + features_.size());
    for (const Feature& feature : features_) {
      if (feature.floating) {
        row.emplace_back(feature.name, slot < 0 ? 0.0 : feature.doubles[static_cast<std::size_t>(slot)]);
      } else {
        row.emplace_back(feature.name, slot < 0 ? int64_t{0} : feature.ints[static_cast<std::size_t>(slot)]);
      }
    }
    return row;
  }

 private:
  struct Column {
    std::string name;
    std::vector<Agg> aggs;
    std::string tag;  // the input type, fixed at fit or load
    Family family = Family::Utf8;
    bool count_only = false;  // only count: the null bitmap suffices, whatever the type
    bool sums = false;        // keep the running sum (for sum or mean)
    bool minmax = false;      // keep min and max
    int median = -1;          // index among the buffered columns, or -1
  };

  // One output column: its values in key order, integer or floating.
  struct Feature {
    std::string name;
    std::string tag;
    bool floating = false;
    std::vector<int64_t> ints;
    std::vector<double> doubles;
  };

  // Anything but count needs an integer or float column.
  static void require_numeric(const Column& column, const std::string& tag, const std::string& where) {
    if (column.count_only) return;
    const std::optional<Family> family = family_of_tag(tag);
    if (family == Family::Integer || family == Family::Floating) return;
    const std::string kind =
        family.has_value() ? std::string(family_name(*family)) + " column" : std::string("a column of that type");
    throw Error(where + " is " + tag + "; only count applies to " + kind + " (cast it to a number first)");
  }

  static std::string output_tag(const Column& column, Agg agg) {
    switch (agg) {
      case Agg::Count:
        return "i64";
      case Agg::Sum:
        return column.family == Family::Integer ? "i64" : "f64";
      case Agg::Mean:
      case Agg::Median:
        return "f64";
      case Agg::Min:
      case Agg::Max:
        return column.tag;
    }
    return "f64";
  }

  static Feature make_feature(const Column& column, Agg agg) {
    Feature feature;
    feature.name = column.name + "_" + agg_name(agg);
    feature.tag = output_tag(column, agg);
    feature.floating = feature.tag == "f32" || feature.tag == "f64";
    return feature;
  }

  void push_value(Feature& feature, const Column& column, Agg agg, const Acc& acc, SpillVector<double>* buffer) const {
    const bool integer = column.family == Family::Integer;
    auto float_sum = [&] {
      const double sum = acc.fsum.value();
      if (!std::isfinite(sum)) throw Error(overflow_f64(column));
      return sum;
    };
    switch (agg) {
      case Agg::Count:
        feature.ints.push_back(acc.count);
        return;
      case Agg::Sum:
        if (integer) {
          if (acc.isum > std::numeric_limits<int64_t>::max() || acc.isum < std::numeric_limits<int64_t>::min()) {
            throw Error(overflow(column));
          }
          feature.ints.push_back(static_cast<int64_t>(acc.isum));
        } else {
          feature.doubles.push_back(float_sum());
        }
        return;
      case Agg::Mean:
        if (acc.count == 0) {
          feature.doubles.push_back(0.0);
        } else if (integer) {
          feature.doubles.push_back(static_cast<double>(acc.isum) / static_cast<double>(acc.count));
        } else {
          feature.doubles.push_back(float_sum() / static_cast<double>(acc.count));
        }
        return;
      case Agg::Min:
        if (integer) {
          feature.ints.push_back(acc.count == 0 ? 0 : acc.imin);
        } else {
          feature.doubles.push_back(acc.count == 0 ? 0.0 : acc.fmin);
        }
        return;
      case Agg::Max:
        if (integer) {
          feature.ints.push_back(acc.count == 0 ? 0 : acc.imax);
        } else {
          feature.doubles.push_back(acc.count == 0 ? 0.0 : acc.fmax);
        }
        return;
      case Agg::Median:
        feature.doubles.push_back(buffer == nullptr ? 0.0 : median_of(*buffer));
        return;
    }
  }

  // An aggregated column other than the key is dropped and its aggregates
  // are appended instead, whether or not the input carried it.
  bool replaced(const std::string& name) const {
    if (name == key_) return false;
    return std::any_of(columns_.begin(), columns_.end(), [&](const Column& column) { return column.name == name; });
  }

  void drop_replaced(Batch& batch) const {
    Batch kept;
    kept.rows = batch.rows;
    for (std::size_t i = 0; i < batch.names.size(); ++i) {
      if (replaced(batch.names[i])) continue;
      kept.names.push_back(std::move(batch.names[i]));
      kept.cols.push_back(std::move(batch.cols[i]));
    }
    batch = std::move(kept);
  }

  std::string overflow(const Column& column) const {
    return stage + ": column '" + column.name + "': the sum overflows a 64-bit integer (cast the column to f64 first)";
  }

  std::string overflow_f64(const Column& column) const {
    return stage + ": column '" + column.name + "': the sum overflows f64";
  }

  // The slot of every row's key, or -1 for a null key (and for whatever the
  // slot callbacks answer -1 to). Utf8 keys arrive plain or dictionary-encoded;
  // a dictionary no larger than the batch is hashed once per entry in use.
  template <typename StringSlot, typename IntSlot>
  std::vector<int64_t> key_slots(const arrow::Array& keys, StringSlot string_slot, IntSlot int_slot) const {
    std::vector<int64_t> slots(static_cast<std::size_t>(keys.length()), -1);
    if (key_kind_ == KeyKind::Integer) {
      for_each_integer(keys, stage, key_, [&](int64_t row, bool is_null, int64_t value) {
        if (!is_null) slots[static_cast<std::size_t>(row)] = int_slot(value);
      });
      return slots;
    }
    if (keys.type_id() != arrow::Type::DICTIONARY) {
      for_each_utf8(keys, stage, key_, [&](int64_t row, bool is_null, std::string_view value) {
        if (!is_null) slots[static_cast<std::size_t>(row)] = string_slot(value);
      });
      return slots;
    }
    const auto& dictionary_array = static_cast<const arrow::DictionaryArray&>(keys);
    const arrow::Array& values = *dictionary_array.dictionary();
    constexpr std::size_t max_cache_bytes = 8 * 1024 * 1024;
    const auto entries = values.length();
    // Count a full byte for each packed `used` flag so this bound includes
    // both dense caches without relying on vector<bool>'s storage layout.
    if (entries <= 0 || entries > keys.length() ||
        static_cast<uint64_t>(entries) > max_cache_bytes / (sizeof(int64_t) + 1)) {
      for_each_utf8_value(keys, stage, key_, [&](int64_t row, bool is_null, std::string_view value) {
        if (!is_null) slots[static_cast<std::size_t>(row)] = string_slot(value);
      });
      return slots;
    }
    std::vector<bool> used(static_cast<std::size_t>(values.length()), false);
    std::vector<int32_t> copied;
    const int32_t* indices = int32_indices(dictionary_array, copied);
    const RowNulls null_at(dictionary_array);
    const int64_t length = dictionary_array.length();
    for (int64_t row = 0; row < length; ++row) {
      if (!null_at(row)) used[static_cast<std::size_t>(indices[row])] = true;
    }
    std::vector<int64_t> entry_slots(static_cast<std::size_t>(values.length()), -1);
    for_each_dictionary_entry(values, stage, key_, [&](int64_t entry, bool is_null, std::string_view value) {
      if (!is_null && used[static_cast<std::size_t>(entry)])
        entry_slots[static_cast<std::size_t>(entry)] = string_slot(value);
    });
    for (int64_t row = 0; row < length; ++row) {
      if (!null_at(row)) slots[static_cast<std::size_t>(row)] = entry_slots[static_cast<std::size_t>(indices[row])];
    }
    return slots;
  }

  void merge_into(State& into, State& from) const {
    std::vector<std::size_t> target(from.slots);
    if (key_kind_ == KeyKind::Utf8) {
      for (auto& [key, slot] : from.str_slots) {
        auto found = into.str_slots.find(key);
        if (found == into.str_slots.end()) {
          const std::size_t fresh = into.add_slot();
          found = into.str_slots.emplace(std::move(key), fresh).first;
        }
        target[slot] = found->second;
      }
      from.str_slots.clear();
    } else {
      for (const auto& [key, slot] : from.int_slots) {
        auto [entry, inserted] = into.int_slots.try_emplace(key, 0);
        if (inserted) entry->second = into.add_slot();
        target[slot] = entry->second;
      }
      from.int_slots.clear();
    }
    for (std::size_t slot = 0; slot < from.slots; ++slot) {
      for (std::size_t c = 0; c < columns_.size(); ++c) {
        Acc& a = into.accs[target[slot] * columns_.size() + c];
        const Acc& b = from.accs[slot * columns_.size() + c];
        a.count += b.count;
        a.isum += b.isum;
        if (!a.fsum.merge(b.fsum)) throw Error(overflow_f64(columns_[c]));
        a.imin = std::min(a.imin, b.imin);
        a.imax = std::max(a.imax, b.imax);
        a.fmin = std::min(a.fmin, b.fmin);
        a.fmax = std::max(a.fmax, b.fmax);
      }
      for (std::size_t m = 0; m < medians_; ++m) {
        SpillVector<double>& dst = into.buffers[target[slot] * medians_ + m];
        SpillVector<double>& src = from.buffers[slot * medians_ + m];
        dst.append(std::move(src));
      }
    }
  }

  void index_keys() {
    str_index_.clear();
    int_index_.clear();
    str_index_.reserve(str_keys_.size());
    int_index_.reserve(int_keys_.size());
    for (std::size_t i = 0; i < str_keys_.size(); ++i) str_index_.emplace(str_keys_[i], i);
    str_short_.index(str_keys_);
    for (std::size_t i = 0; i < int_keys_.size(); ++i) int_index_.emplace(int_keys_[i], i);
    feature_index_.clear();
    for (std::size_t i = 0; i < features_.size(); ++i) feature_index_.emplace(features_[i].name, i);
  }

  std::string key_;
  std::vector<Column> columns_;
  std::size_t medians_ = 0;
  KeyKind key_kind_ = KeyKind::Utf8;
  std::vector<std::string> str_keys_;  // fitted keys, sorted
  std::vector<int64_t> int_keys_;
  StringIndex str_index_;  // key -> position in the feature lists
  ShortKeys str_short_;    // the same, when the keys allow a perfect hash
  IntIndex int_index_;
  std::vector<Feature> features_;
  StringIndex feature_index_;  // output name -> position in features_
};

}  // namespace

std::unique_ptr<Kernel> make_group_agg(const nlohmann::json& params) { return std::make_unique<GroupAgg>(params); }

}  // namespace fastfeaturetransform
