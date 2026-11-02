// quantile_encode and summary_encode: a category becomes a quantile of its
// target values, shrunk toward the same quantile of the whole target. An
// exact quantile needs every target value of a category, so both keep those
// values in buffers that spill to temporary files and sort externally once
// spilled; resident memory grows with distinct categories, not with rows.

#include <arrow/util/utf8.h>

#include <algorithm>
#include <cmath>
#include <format>
#include <optional>
#include <unordered_map>

#include "fastfeaturetransform/kernels/cast_table.hpp"
#include "fastfeaturetransform/kernels/categories.hpp"
#include "fastfeaturetransform/kernels/kernels.hpp"
#include "fastfeaturetransform/kernels/output.hpp"
#include "fastfeaturetransform/kernels/util.hpp"
#include "fastfeaturetransform/spill.hpp"

namespace fastfeaturetransform {
namespace {

double quantile_value(const nlohmann::json& value, const std::string& message) {
  if (!value.is_number()) throw Error(message);
  const double quantile = value.get<double>();
  if (!std::isfinite(quantile) || quantile < 0.0 || quantile > 1.0) throw Error(message);
  return quantile;
}

// The percentile that names a generated column, as upstream spells it:
// Python's round(), which breaks a tie toward the even digit — exactly what
// nearbyint does under the default rounding mode.
std::string percentile_name(double quantile) {
  return std::to_string(static_cast<int64_t>(std::nearbyint(quantile * 100.0)));
}

// numpy's default method="linear", which is what upstream calls: with the
// values sorted ascending and h = quantile*(n-1), the answer interpolates
// between v[floor(h)] and v[floor(h)+1]. numpy lerps from whichever end is
// nearer (its _lerp), reproduced here so the last bit agrees too.
double quantile_of(const SpillVector<double>& sorted, double quantile) {
  if (sorted.empty()) return 0.0;
  const auto last = static_cast<double>(sorted.size() - 1);
  const double index = quantile * last;
  if (index <= 0.0) return sorted.front();
  if (index >= last) return sorted.back();
  const auto lower = static_cast<std::size_t>(index);  // index is positive and finite, so this floors
  const double weight = index - static_cast<double>(lower);
  const double left = sorted[lower];
  const double right = sorted[lower + 1];
  const double span = right - left;
  return weight >= 0.5 ? right - span * (1.0 - weight) : left + span * weight;
}

// Only the two neighbors of each requested quantile need their sorted
// positions, so select those ranks recursively: quantile_of reads the same
// values there as after a full sort. A spilled, tiny or densely requested
// buffer is sorted whole.
void prepare_quantiles(SpillVector<double>& values, const std::vector<double>& quantiles) {
  if (values.empty()) return;
  if (values.spilled()) {
    values.sort();
    return;
  }
  const auto memory = values.mutable_memory();
  std::vector<std::size_t> positions;
  const auto last = static_cast<double>(values.size() - 1);
  for (double quantile : quantiles) {
    const double index = quantile * last;
    if (index <= 0.0)
      positions.push_back(0);
    else if (index >= last)
      positions.push_back(values.size() - 1);
    else {
      const auto lower = static_cast<std::size_t>(index);
      positions.push_back(lower);
      positions.push_back(lower + 1);
    }
  }
  std::sort(positions.begin(), positions.end());
  positions.erase(std::unique(positions.begin(), positions.end()), positions.end());
  if (values.size() <= 64 || positions.size() >= values.size() / 8) {
    std::sort(memory.begin(), memory.end());
    return;
  }
  auto select = [&](auto&& self, std::size_t first, std::size_t end, std::size_t lower, std::size_t upper) -> void {
    if (first == end) return;
    const std::size_t middle = first + (end - first) / 2;
    const std::size_t position = positions[middle];
    std::nth_element(memory.begin() + static_cast<std::ptrdiff_t>(lower),
                     memory.begin() + static_cast<std::ptrdiff_t>(position),
                     memory.begin() + static_cast<std::ptrdiff_t>(upper));
    self(self, first, middle, lower, position);
    self(self, middle + 1, end, position + 1, upper);
  };
  select(select, 0, positions.size(), 0, values.size());
}

// The m-probability estimate: a category's own quantile pulled toward the
// global one, with weight m. Every recorded category has at least one target
// value, so the denominator is positive even when m is 0.
double smoothed(double category_quantile, std::size_t count, double prior, double m) {
  const auto n = static_cast<double>(count);
  return (n * category_quantile + prior * m) / (n + m);
}

void append_values(SpillVector<double>& into, SpillVector<double>& from) { into.append(std::move(from)); }

// category -> every target value seen for it, in arrival order.
using ValueBuffers = CategoryAccumulator<SpillVector<double>>;

// ---------------------------------------------------------------------------
// quantile_encode — categorical utf8 -> a smoothed quantile of the target.
// params: {"columns": [...] | null, "target": column, "quantile": float,
//          "m": float, "handle_unknown": "value" | "return_nan" | "error",
//          "handle_missing": "value" | "return_nan" | "error"}
// Semantics:
//   * prior is the quantile of the target over every fit row; a category
//     whose n target values have quantile q encodes to
//     (n*q + prior*m)/(n + m), so m = 0 is the raw quantile and a large m
//     is the prior
//   * the quantile is numpy's "linear" method: over the sorted values and
//     h = quantile*(n-1), it interpolates between v[floor(h)] and the value
//     after it
//   * the target is named in the params, not passed to fit; it must be an
//     integer or float column, must not be encoded itself, and passes
//     through to the output (add a trailing select to drop it)
//   * a row whose target is null contributes to nothing: not the prior, not
//     its category, not the null category
//   * a null answers to handle_missing alone: a category of its own under
//     'value' (the prior when the fit saw no null), a null under
//     'return_nan', and an error under 'error' — at fit if the column holds
//     one, at transform if one arrives later
//   * an unseen category yields the prior under handle_unknown='value', a
//     null under 'return_nan', and an error under 'error'
//   * the encoded column becomes f64 in place, keeping its name; columns=null
//     encodes every utf8 column that reaches the step
//   * a target value that is NaN or infinity is an error, and -0.0 reads as
//     0.0, because the fitted table is JSON and the sorted buffer must not
//     depend on the split
//   * target values spill to temporary files under the fit's shared memory
//     budget, so memory grows with distinct categories, not with rows; the
//     fitted table is independent of the split
// ---------------------------------------------------------------------------
// summary_encode — the same encoder run at several quantiles at once.
// params: as quantile_encode, but "quantiles": [float, ...] in place of
//         "quantile".
// Semantics: everything above, and
//   * each encoded column is replaced, in its own position, by one f64
//     column per quantile named <column>_<round(quantile*100)>, in the order
//     the quantiles were given
//   * two quantiles that round to the same percentile are a param error, as
//     they would name the same column
// ---------------------------------------------------------------------------
class QuantileEncode final : public Kernel {
 public:
  QuantileEncode(const nlohmann::json& params, bool summary) : summary_(summary) {
    const std::string what = summary_ ? "summary_encode" : "quantile_encode";
    if (!params.contains("target") || !params.at("target").is_string() ||
        params.at("target").get<std::string>().empty()) {
      throw Error(what + ": 'target' must be a column name");
    }
    target_ = params.at("target").get<std::string>();
    if (params.contains("columns") && !params.at("columns").is_null()) {
      columns_ = column_list(params, "columns", what);
      explicit_columns_ = true;
    }
    if (summary_) {
      nlohmann::json list = nlohmann::json::array({0.25, 0.75});
      if (params.contains("quantiles") && !params.at("quantiles").is_null()) list = params.at("quantiles");
      const std::string message = what + ": 'quantiles' must be a non-empty list of numbers between 0 and 1";
      if (!list.is_array() || list.empty()) throw Error(message);
      for (const nlohmann::json& item : list) quantiles_.push_back(quantile_value(item, message));
    } else {
      const std::string message = what + ": 'quantile' must be a number between 0 and 1";
      quantiles_.push_back(params.contains("quantile") && !params.at("quantile").is_null()
                               ? quantile_value(params.at("quantile"), message)
                               : 0.5);
    }
    for (double quantile : quantiles_) percentiles_.push_back(percentile_name(quantile));
    std::vector<std::string> sorted = percentiles_;
    std::sort(sorted.begin(), sorted.end());
    if (auto dup = std::adjacent_find(sorted.begin(), sorted.end()); dup != sorted.end()) {
      throw Error(what + ": two quantiles round to the same percentile '" + *dup + "'");
    }
    if (params.contains("m") && !params.at("m").is_null()) {
      const nlohmann::json& m = params.at("m");
      if (!m.is_number() || !std::isfinite(m.get<double>()) || m.get<double>() < 0.0) {
        throw Error(what + ": 'm' must be a finite number >= 0");
      }
      m_ = m.get<double>();
    }
    unknown_ = unknown_from(params, what, false);
    missing_ = missing_from(params, what);
    if (explicit_columns_) name_outputs();
  }

  std::string id() const override { return summary_ ? "summary_encode" : "quantile_encode"; }

  nlohmann::json params() const override {
    nlohmann::json out;
    out["columns"] = explicit_columns_ ? nlohmann::json(columns_) : nlohmann::json();
    out["target"] = target_;
    if (summary_) {
      out["quantiles"] = quantiles_;
    } else {
      out["quantile"] = quantiles_.front();
    }
    out["m"] = m_;
    out["handle_unknown"] = unknown_name(unknown_);
    out["handle_missing"] = missing_name(missing_);
    return out;
  }

  void resolve(Schema& schema) override {
    // A fitted encoder's columns come from its state and are never re-derived;
    // it no longer reads the target either, so a serving file without the
    // label still transforms.
    if (!fitted) {
      const std::string* target_type = schema.find(target_);
      if (target_type == nullptr) throw Error(stage + ": target column '" + target_ + "' not found in input");
      const std::optional<Family> family = family_of_tag(*target_type);
      if (family != Family::Integer && family != Family::Floating) {
        throw Error(stage + ": target column '" + target_ + "' is " + *target_type +
                    ", expected an integer or float column (cast it first)");
      }
      target_family_ = *family;
    }
    // The target is an integer or a float, checked just above, so it can
    // never join a derived utf8 list; naming it explicitly is the error.
    bind_columns(schema, stage, columns_, explicit_columns_, fitted, "encode", &target_);
    name_outputs();
    if (!summary_) {  // the column keeps its name and place, only its type changes
      for (auto& [name, type] : schema.cols) {
        if (encoded(name)) type = "f64";
      }
    } else {
      for (const std::string& name : outputs) {
        if (schema.find(name) != nullptr) throw Error(stage + ": output column '" + name + "' already exists in input");
      }
      Schema out;
      for (const auto& [name, type] : schema.cols) {
        if (!encoded(name)) {
          out.cols.emplace_back(name, type);
          continue;
        }
        for (std::size_t q = 0; q < quantiles_.size(); ++q) out.cols.emplace_back(output_name(name, q), "f64");
      }
      schema = std::move(out);
    }
    if (!fitted) inputs.push_back(target_);
  }

  bool needs_fit() const override { return true; }

  struct State final : AccState {
    SpillVector<double> target;                                         // every non-null target value seen
    std::unordered_map<std::string, ValueBuffers> buckets;              // column -> category -> its target values
    std::unordered_map<std::string, SpillVector<double>> null_buckets;  // column -> the null category's values
    std::unordered_map<std::string, int64_t> null_rows;                 // column -> nulls seen, for handle_missing
  };

  std::unique_ptr<AccState> make_state() const override { return std::make_unique<State>(); }

  void consume(AccState& raw, const Batch& batch) const override {
    auto& state = static_cast<State&>(raw);
    const arrow::Array& target = column_array(batch, stage, target_);
    const auto rows = static_cast<std::size_t>(target.length());
    std::vector<double> values(rows, 0.0);
    std::vector<bool> present(rows, false);
    auto keep = [&](int64_t row, bool is_null, double value) {
      if (is_null) return;
      if (!std::isfinite(value)) {
        throw Error(stage + ": target column '" + target_ + "': value " + std::format("{}", value) +
                    " is not finite; a quantile cannot be taken over NaN or infinity");
      }
      if (value == 0.0) value = 0.0;  // one zero: -0.0 would make the sorted buffer depend on the split
      values[static_cast<std::size_t>(row)] = value;
      present[static_cast<std::size_t>(row)] = true;
    };
    if (target_family_ == Family::Integer) {
      for_each_integer(target, stage, target_, [&](int64_t row, bool is_null, int64_t value) {
        keep(row, is_null, static_cast<double>(value));
      });
    } else {
      for_each_floating(target, stage, target_, keep);
    }
    for (std::size_t i = 0; i < rows; ++i) {
      if (present[i]) state.target.push_back(values[i]);
    }

    for (const std::string& name : columns_) {
      ValueBuffers& buckets = state.buckets[name];
      SpillVector<double>& null_bucket = state.null_buckets[name];
      int64_t& null_rows = state.null_rows[name];
      for_each_encoded_utf8_if(
          column_array(batch, stage, name), stage, name,
          [&](int64_t row, bool is_null) {
            // Every null counts toward handle_missing='error', even one with
            // a null target.
            if (is_null) ++null_rows;
            return present[static_cast<std::size_t>(row)];
          },
          [&](bool is_null, std::string_view value) -> std::optional<std::size_t> {
            return is_null ? std::nullopt : std::optional<std::size_t>{buckets.slot(value)};
          },
          [&](int64_t row, const std::optional<std::size_t>& category) {
            auto& buffer = category.has_value() ? buckets.values[*category] : null_bucket;
            buffer.push_back(values[static_cast<std::size_t>(row)]);
          });
    }
  }

  void finalize(std::vector<std::unique_ptr<AccState>> states) override {
    arrow::util::InitializeUTF8();  // idempotent; ValidateUTF8 needs its tables
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

    // Without a target there is nothing to learn: every prior and every
    // category would come out zero, and a mis-joined label column would fit,
    // save, load and serve a constant with nothing anywhere saying so.
    if (merged->target.empty()) {
      throw Error(stage + ": target column '" + target_ + "' has no non-null values to learn from");
    }
    // Selecting the same order statistics is independent of worker splits.
    prepare_quantiles(merged->target, quantiles_);
    priors_.clear();
    for (double quantile : quantiles_) priors_.push_back(target_quantile(merged->target, quantile));
    merged->target.release();

    tables_.clear();
    for (const std::string& name : columns_) {
      if (missing_ == Missing::Error && merged->null_rows[name] > 0) {
        throw Error(stage + ": column '" + name + "' has nulls and handle_missing is 'error'");
      }
      ValueBuffers& buckets = merged->buckets[name];
      Table& table = tables_[name];
      table.categories.reserve(buckets.slots.size());
      for (const auto& [key, slot] : buckets.slots) {
        if (!arrow::util::ValidateUTF8(key)) {
          throw Error(stage + ": column '" + name + "' has a category that is not valid UTF-8; " + id() +
                      " keys must be UTF-8 to serialize");
        }
        table.categories.push_back(key);
      }
      std::sort(table.categories.begin(), table.categories.end());  // sorted: the table never depends on the split
      table.values.assign(table.categories.size() * quantiles_.size(), 0.0);
      for (std::size_t i = 0; i < table.categories.size(); ++i) {
        SpillVector<double>& buffer = buckets.at(table.categories[i]);
        prepare_quantiles(buffer, quantiles_);
        for (std::size_t q = 0; q < quantiles_.size(); ++q) {
          table.values[i * quantiles_.size() + q] = encoding(buffer, quantiles_[q], priors_[q], name);
        }
        buffer.release();
      }
      SpillVector<double>& null_bucket = merged->null_buckets[name];
      if (missing_ == Missing::Value && !null_bucket.empty()) {
        prepare_quantiles(null_bucket, quantiles_);
        table.has_null = true;
        for (std::size_t q = 0; q < quantiles_.size(); ++q) {
          table.null_values.push_back(encoding(null_bucket, quantiles_[q], priors_[q], name));
        }
      }
    }
    merged.reset();
    index_tables();
    inputs = columns_;  // the target was read at fit only
    fitted = true;
  }

  nlohmann::json state() const override {
    nlohmann::json categories = nlohmann::json::object();
    nlohmann::json encodings = nlohmann::json::object();
    nlohmann::json nulls = nlohmann::json::object();
    for (const std::string& name : columns_) {
      const Table& table = tables_.at(name);
      categories[name] = table.categories;
      for (std::size_t q = 0; q < quantiles_.size(); ++q) {
        std::vector<double> column_values;
        column_values.reserve(table.categories.size());
        for (std::size_t i = 0; i < table.categories.size(); ++i) {
          column_values.push_back(serializable(table.values[i * quantiles_.size() + q], stage, name));
        }
        encodings[output_name(name, q)] = std::move(column_values);
        nulls[output_name(name, q)] =
            table.has_null ? nlohmann::json(serializable(table.null_values[q], stage, name)) : nlohmann::json();
      }
    }
    return {{"columns", columns_},
            {"priors", priors_},
            {"categories", categories},
            {"encodings", encodings},
            {"nulls", nulls}};
  }

  // resolve() never runs on load, so the state carries the whole table: one
  // prior per quantile, the sorted categories of each column, and one value
  // list per output column, all checked before they are trusted.
  void load_state(const nlohmann::json& state) override {
    const std::string bad = stage + ": state must hold 'priors', 'categories', 'encodings' and 'nulls'";
    if (!state.is_object() || !state.contains("priors") || !state.at("priors").is_array() ||
        !state.contains("categories") || !state.at("categories").is_object() || !state.contains("encodings") ||
        !state.at("encodings").is_object() || !state.contains("nulls") || !state.at("nulls").is_object()) {
      throw Error(bad);
    }
    if (state.at("priors").size() != quantiles_.size()) {
      throw Error(stage + ": 'priors' must hold one value per quantile");
    }
    priors_.clear();
    for (const nlohmann::json& prior : state.at("priors")) priors_.push_back(number(prior, stage + ": 'priors'"));

    const nlohmann::json& categories = state.at("categories");
    load_fitted_columns(state, stage, columns_, explicit_columns_, &categories);
    tables_.clear();
    for (const std::string& name : columns_) {
      if (!categories.contains(name) || !categories.at(name).is_array()) {
        throw Error(stage + ": categories['" + name + "'] must be a list of category names");
      }
      Table& table = tables_[name];
      for (const nlohmann::json& key : categories.at(name)) {
        if (!key.is_string()) throw Error(stage + ": categories['" + name + "'] must be a list of category names");
        table.categories.push_back(key.get<std::string>());
      }
      if (!std::is_sorted(table.categories.begin(), table.categories.end()) ||
          std::adjacent_find(table.categories.begin(), table.categories.end()) != table.categories.end()) {
        throw Error(stage + ": categories['" + name + "'] must be sorted and unique");
      }
      table.values.assign(table.categories.size() * quantiles_.size(), 0.0);
      for (std::size_t q = 0; q < quantiles_.size(); ++q) {
        const std::string output = output_name(name, q);
        const std::string where = stage + ": encodings['" + output + "']";
        const nlohmann::json& encodings = state.at("encodings");
        if (!encodings.contains(output) || !encodings.at(output).is_array() ||
            encodings.at(output).size() != table.categories.size()) {
          throw Error(where + " must be a list with one value per category");
        }
        for (std::size_t i = 0; i < table.categories.size(); ++i) {
          table.values[i * quantiles_.size() + q] = number(encodings.at(output).at(i), where);
        }
        const nlohmann::json& nulls = state.at("nulls");
        if (!nulls.contains(output) || nulls.at(output).is_null()) continue;
        table.has_null = true;
        table.null_values.push_back(number(nulls.at(output), stage + ": nulls['" + output + "']"));
      }
      if (table.has_null && table.null_values.size() != quantiles_.size()) {
        throw Error(stage + ": column '" + name + "' must have a null encoding for every quantile or none");
      }
    }
    index_tables();
    name_outputs();
    fitted = true;
  }

  Batch transform(Batch batch) const override {
    for (const std::string& name : columns_) {
      if (batch.find(name) < 0) throw Error(stage + ": column '" + name + "' missing from batch");
    }
    Batch out;
    out.rows = batch.rows;
    out.names.reserve(batch.names.size());
    out.cols.reserve(batch.cols.size());
    for (std::size_t i = 0; i < batch.names.size(); ++i) {
      const std::string& name = batch.names[i];
      if (!encoded(name)) {
        out.names.push_back(std::move(batch.names[i]));
        out.cols.push_back(std::move(batch.cols[i]));
        continue;
      }
      const arrow::Array& array = *batch.cols[i];
      const Table& table = tables_.at(name);
      auto validity = std::make_shared<OutputValidity>(array.length(), stage + ": column '" + name + "'");
      std::vector<PrimitiveOutput<arrow::DoubleType>> values;
      values.reserve(quantiles_.size());
      for (const std::string& output : table.outputs) {
        values.emplace_back(array.length(), stage + ": column '" + output + "'", false, validity);
      }
      for_each_encoded_utf8(
          array, stage, name,
          [&](bool is_null, std::string_view value) { return look_up(table, name, is_null, value); },
          [&](int64_t row, const double* found) {
            if (found == nullptr) {
              for (auto& output : values) output.set_null(row);
              return;
            }
            for (std::size_t q = 0; q < quantiles_.size(); ++q) values[q].mutable_values()[row] = found[q];
          });
      for (std::size_t q = 0; q < quantiles_.size(); ++q) {
        out.cols.push_back(values[q].finish());
        out.names.push_back(table.outputs[q]);
      }
    }
    return out;
  }

  Row transform_row(Row row) const override {
    {
      const RecordLookup lookup(row, columns_.size());
      for (const std::string& name : columns_) {
        if (lookup.find(name) == nullptr) throw Error(stage + ": key '" + name + "' missing from record");
      }
    }
    Row out;
    // Sized from what the fit froze, never from `outputs`: resolve() rewrites
    // that member under a mutex while this may be running on another thread.
    out.reserve(row.size() + columns_.size() * quantiles_.size());
    for (auto& [key, value] : row) {
      if (!encoded(key)) {
        if (output_index_.contains(key)) throw Error(stage + ": key '" + key + "' already present in record");
        out.emplace_back(std::move(key), std::move(value));
        continue;
      }
      const bool is_null = std::holds_alternative<std::monostate>(value);
      const std::string* text = std::get_if<std::string>(&value);
      if (!is_null && text == nullptr) throw Error(stage + ": key '" + key + "' must be a string or null");
      const Table& table = tables_.at(key);
      const double* found = look_up(table, key, is_null, is_null ? std::string_view{} : *text);
      const auto& names = table.outputs;
      for (std::size_t q = 0; q < quantiles_.size(); ++q) {
        out.emplace_back(names[q], found == nullptr ? RowValue(std::monostate{}) : RowValue(found[q]));
      }
    }
    return out;
  }

 private:
  // One encoded column: its sorted categories and, per category, one encoded
  // value per quantile (row-major by category, so a lookup is contiguous).
  struct Table {
    std::vector<std::string> categories;
    StringIndex index;      // category -> its position in categories
    ShortKeys short_index;  // the same, when the categories allow a perfect hash
    std::vector<double> values;
    std::vector<double> null_values;  // one per quantile, when the null category was seen
    std::vector<std::string> outputs;
    bool has_null = false;
  };

  static double number(const nlohmann::json& value, const std::string& where) {
    if (!value.is_number() || !std::isfinite(value.get<double>())) throw Error(where + " must hold finite numbers");
    return value.get<double>();
  }

  bool encoded(const std::string& name) const {
    return std::find(columns_.begin(), columns_.end(), name) != columns_.end();
  }

  std::string output_name(const std::string& column, std::size_t quantile) const {
    return summary_ ? column + "_" + percentiles_[quantile] : column;
  }

  void name_outputs() {
    inputs = columns_;
    outputs.clear();
    for (const std::string& name : columns_) {
      for (std::size_t q = 0; q < quantiles_.size(); ++q) outputs.push_back(output_name(name, q));
    }
    reject_duplicate_outputs(outputs, stage);
  }

  void index_tables() {
    output_index_.clear();
    for (const std::string& name : columns_) {
      Table& table = tables_.at(name);
      table.index.clear();
      table.index.reserve(table.categories.size());
      for (std::size_t i = 0; i < table.categories.size(); ++i) table.index.emplace(table.categories[i], i);
      table.short_index.index(table.categories);
      table.outputs.clear();
      for (std::size_t q = 0; q < quantiles_.size(); ++q) {
        table.outputs.push_back(output_name(name, q));
        output_index_.emplace(table.outputs.back(), q);
      }
    }
  }

  // The encoded values of one input value (one per quantile), or nullptr when
  // the output is a null.
  const double* look_up(const std::string& name, bool is_null, std::string_view value) const {
    return look_up(tables_.at(name), name, is_null, value);
  }

  const double* look_up(const Table& table, const std::string& name, bool is_null, std::string_view value) const {
    // A null answers to handle_missing alone: a null the fit never saw is the
    // prior, not an unknown category, exactly as upstream's -2 slot is.
    if (is_null) {
      if (missing_ == Missing::ReturnNan) return nullptr;
      refuse_null(missing_, stage, name);
      return table.has_null ? table.null_values.data() : priors_.data();
    }
    if (!table.short_index.empty()) {
      const int64_t found = table.short_index.find(value);
      if (found >= 0) return &table.values[static_cast<std::size_t>(found) * quantiles_.size()];
      return unknown_value(name);
    }
    auto found = table.index.find(value);
    if (found != table.index.end()) return &table.values[found->second * quantiles_.size()];
    return unknown_value(name);
  }

  // A quantile of the target. numpy's lerp between two neighbors more than
  // f64's largest apart overflows, and upstream carries that into the
  // encoding; the fit refuses it here and names the target.
  double target_quantile(const SpillVector<double>& sorted, double quantile) const {
    const double value = quantile_of(sorted, quantile);
    if (!std::isfinite(value)) throw Error(stage + ": target column '" + target_ + "' spans past the range of f64");
    return value;
  }

  // One bucket's encoding, category_encoders' (n * quantile + prior * m) /
  // (n + m). Either product can overflow f64: n * quantile when the target is
  // near f64's largest, and prior * m when m is large. state() refuses to
  // write a value that is not finite and load_state to read one, so the fit
  // refuses it here and names which product overflowed.
  double encoding(const SpillVector<double>& values, double quantile, double prior, const std::string& name) const {
    const double own = target_quantile(values, quantile);
    const double value = smoothed(own, values.size(), prior, m_);
    if (!std::isfinite(value) && std::isinf(static_cast<double>(values.size()) * own)) {
      throw Error(stage + ": target column '" + target_ + "' is too large to smooth: " + std::to_string(values.size()) +
                  " times its quantile is past the range of f64");
    }
    return finite_encoding(value, stage, name, "m");
  }

  const double* unknown_value(const std::string& name) const {
    switch (unknown_) {
      case Unknown::Value:
        return priors_.data();
      case Unknown::ReturnNan:
        return nullptr;
      case Unknown::Indicator:
      case Unknown::Error:
        break;
    }
    throw Error(stage + ": column '" + name + "' has a category that was not seen at fit");
  }

  void merge_into(State& into, State& from) const {
    append_values(into.target, from.target);
    for (auto& [name, buckets] : from.buckets) {
      ValueBuffers& target = into.buckets[name];
      for (auto& [key, slot] : buckets.slots) {
        auto& buffer = buckets.values[slot];
        auto found = target.slots.find(key);
        if (found == target.slots.end()) {
          target.slots.emplace(std::move(key), target.values.size());
          target.values.push_back(std::move(buffer));
        } else {
          append_values(target.values[found->second], buffer);
        }
      }
    }
    for (auto& [name, buffer] : from.null_buckets) append_values(into.null_buckets[name], buffer);
    for (const auto& [name, count] : from.null_rows) into.null_rows[name] += count;
  }

  bool summary_ = false;
  std::string target_;
  std::vector<std::string> columns_;
  bool explicit_columns_ = false;
  std::vector<double> quantiles_;
  std::vector<std::string> percentiles_;  // quantiles_ as the digits that name a summary column
  double m_ = 1.0;
  Unknown unknown_ = Unknown::Value;
  Missing missing_ = Missing::Value;
  Family target_family_ = Family::Floating;
  std::vector<double> priors_;  // one per quantile
  std::unordered_map<std::string, Table> tables_;
  StringIndex output_index_;  // output name -> its quantile, for collisions on the record path
};

}  // namespace

std::unique_ptr<Kernel> make_quantile_encode(const nlohmann::json& params) {
  return std::make_unique<QuantileEncode>(params, false);
}

std::unique_ptr<Kernel> make_summary_encode(const nlohmann::json& params) {
  return std::make_unique<QuantileEncode>(params, true);
}

}  // namespace fastfeaturetransform
