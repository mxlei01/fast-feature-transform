// The encoders built straight on the categorical substrate: every category
// seen at fit, sorted, then either numbered (ordinal_encode) or expanded
// into one column per category (one_hot_encode, rank_hot_encode). The
// substrate itself — the fitted category table, the accumulator, and the
// handle_missing / handle_unknown options — lives in kernels/categories.hpp.

#include <algorithm>
#include <limits>
#include <unordered_map>

#include "fastfeaturetransform/kernels/cast_table.hpp"
#include "fastfeaturetransform/kernels/categories.hpp"
#include "fastfeaturetransform/kernels/kernels.hpp"
#include "fastfeaturetransform/kernels/output.hpp"
#include "fastfeaturetransform/kernels/util.hpp"

namespace fastfeaturetransform {
namespace {

// ---------------------------------------------------------------------------
// ordinal_encode — categorical utf8 -> an integer code, in place.
// params: {"columns": [...] | null, "index_start": int,
//          "handle_unknown": "value" | "return_nan" | "error",
//          "handle_missing": "value" | "return_nan" | "error"}
// Semantics:
//   * categories sort and take codes index_start .. index_start + n - 1
//   * under handle_missing='value' a null seen at fit is a category of its
//     own and takes the code after them, and a null that was not seen at
//     fit encodes to -2, as upstream does; 'return_nan' encodes a null as a
//     null, and 'error' refuses one at transform as well as at fit
//   * an unseen category encodes to -1 under handle_unknown='value', a null
//     under 'return_nan', and an error under 'error'
//   * the highest code must fit in an i64, which fit and load check once the
//     level count is known rather than wrapping into a negative index
//   * columns=null encodes every utf8 column that reaches the step
// ---------------------------------------------------------------------------
class OrdinalEncode final : public Kernel {
 public:
  explicit OrdinalEncode(const nlohmann::json& params) {
    if (params.contains("columns") && !params.at("columns").is_null()) {
      columns_ = column_list(params, "columns", "ordinal_encode");
      explicit_columns_ = true;
      inputs = columns_;
      outputs = columns_;
    }
    if (params.contains("index_start") && !params.at("index_start").is_null()) {
      if (!is_int64(params.at("index_start"))) {
        throw Error("ordinal_encode: 'index_start' must be an integer from -2^63 to 2^63 - 1");
      }
      index_start_ = params.at("index_start").get<int64_t>();
    }
    unknown_ = unknown_from(params, "ordinal_encode", false);
    missing_ = missing_from(params, "ordinal_encode");
  }

  std::string id() const override { return "ordinal_encode"; }

  nlohmann::json params() const override {
    return {{"columns", explicit_columns_ ? nlohmann::json(columns_) : nlohmann::json()},
            {"index_start", index_start_},
            {"handle_unknown", unknown_name(unknown_)},
            {"handle_missing", missing_name(missing_)}};
  }

  void resolve(Schema& schema) override {
    bind_columns(schema, stage, columns_, explicit_columns_, fitted, "ordinal-encode");
    for (auto& [name, type] : schema.cols) {
      if (std::find(columns_.begin(), columns_.end(), name) != columns_.end()) type = "i64";
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
    tables_ = merge_categories(columns_, stage, states, missing_);
    check_codes();
    fitted = true;
  }

  nlohmann::json state() const override { return categories_state(columns_, tables_); }

  void load_state(const nlohmann::json& state) override {
    tables_ = load_categories(state, stage, columns_, explicit_columns_);
    check_codes();
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
      PrimitiveOutput<arrow::Int64Type> codes(array.length(), where);
      const Categories& table = tables_.at(name);
      for_each_encoded_utf8(
          array, stage, name, [&](bool is_null, std::string_view value) { return encode(name, table, is_null, value); },
          [&](int64_t row, const std::optional<int64_t>& code) {
            if (code.has_value()) {
              codes.mutable_values()[row] = *code;
            } else {
              codes.set_null(row);
            }
          });
      batch.cols[static_cast<std::size_t>(idx)] = codes.finish();
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
      const std::optional<int64_t> code = encode(name, tables_.at(name), is_null, is_null ? std::string_view{} : *text);
      if (code.has_value()) {
        *value = *code;
      } else {
        *value = std::monostate{};
      }
    }
    return row;
  }

 private:
  // The codes run index_start_ .. index_start_ + levels - 1, so only the top
  // can leave int64. Checked once the level count is known — at fit and at
  // load — rather than per value: encode() is the hot path, and a wrapped
  // code would be a negative index into an embedding table.
  void check_codes() const {
    for (const std::string& name : columns_) {
      const Categories& table = tables_.at(name);
      // The fitted null, when there is one, sorts after every category and
      // takes the last code; a table always holds at least one level.
      const Int128 highest = static_cast<Int128>(index_start_) + static_cast<Int128>(table.levels()) - 1;
      if (highest > static_cast<Int128>(std::numeric_limits<int64_t>::max())) {
        throw Error(stage + ": column '" + name + "' has " + std::to_string(table.levels()) +
                    " levels, whose highest code does not fit in a 64-bit integer from index_start " +
                    std::to_string(index_start_));
      }
    }
  }

  std::optional<int64_t> encode(const std::string& name, const Categories& table, bool is_null,
                                std::string_view value) const {
    if (is_null) {
      if (missing_ == Missing::ReturnNan) return std::nullopt;
      refuse_null(missing_, stage, name);
      if (table.has_null) return index_start_ + static_cast<int64_t>(table.sorted.size());
      return -2;  // a null that was never a fitted category, as upstream encodes it
    }
    const int64_t rank = table.rank(value);
    if (rank >= 0) return index_start_ + rank;
    switch (unknown_) {
      case Unknown::Value:
        return -1;
      case Unknown::ReturnNan:
        return std::nullopt;
      case Unknown::Indicator:
      case Unknown::Error:
        break;
    }
    throw Error(stage + ": column '" + name + "' has a category that was not seen at fit");
  }

  std::vector<std::string> columns_;
  bool explicit_columns_ = false;
  int64_t index_start_ = 1;
  Unknown unknown_ = Unknown::Value;
  Missing missing_ = Missing::Value;
  std::unordered_map<std::string, Categories> tables_;
};

// ---------------------------------------------------------------------------
// one_hot_encode / rank_hot_encode — categorical utf8 -> one integer column
// per fitted category, replacing the source column.
// params: {"columns": [...] | null, "use_cat_names": bool,
//          "handle_unknown": "value" | "return_nan" | "error"
//                            (one_hot also accepts "indicator"),
//          "handle_missing": "value" | "return_nan" | "error"}
// Semantics:
//   * categories sort; one_hot puts a 1 in the matching column, rank_hot
//     puts a 1 in every column up to and including it (thermometer coding)
//   * a null seen at fit under handle_missing='value' is a category of its
//     own, sorted last, and gets its own column; 'return_nan' makes a null a
//     row of nulls, and 'error' refuses one at transform as well as at fit
//   * under 'value', a null that was never a fitted category is all zeros,
//     except that rank_hot nulls it under handle_unknown='return_nan', which
//     is where upstream's two encoders part company
//   * under rank_hot + handle_missing='return_nan' the pre-encoder's slot for
//     the missing value survives as a level of its own and gets a trailing
//     '<column>_-2' column ('<column>_nan' with use_cat_names) that is never
//     lit; see missing_placeholder in kernels/categories.hpp. one_hot drops
//     that column, as upstream does
//   * an unseen category is all zeros under handle_unknown='value', all
//     nulls under 'return_nan', an error under 'error', and for one_hot a 1
//     in the extra '<column>_-1' column under 'indicator'
//   * output columns are named '<column>_<code>' with the code the encoder
//     would have assigned, or '<column>_<category>' with use_cat_names; a
//     name that would repeat gets '#' appended, across every encoded column
//     and not only within one, as upstream does
//   * how many columns there are is known only after the fit, so a later
//     step cannot name them in the same pipeline
// ---------------------------------------------------------------------------
class Expand final : public Kernel {
 public:
  Expand(const nlohmann::json& params, bool thermometer)
      : thermometer_(thermometer), what_(thermometer ? "rank_hot_encode" : "one_hot_encode") {
    if (params.contains("columns") && !params.at("columns").is_null()) {
      columns_ = column_list(params, "columns", what_);
      explicit_columns_ = true;
      inputs = columns_;
    }
    if (params.contains("use_cat_names") && !params.at("use_cat_names").is_null()) {
      if (!params.at("use_cat_names").is_boolean()) throw Error(what_ + ": 'use_cat_names' must be true or false");
      use_cat_names_ = params.at("use_cat_names").get<bool>();
    }
    unknown_ = unknown_from(params, what_.c_str(), !thermometer);
    missing_ = missing_from(params, what_.c_str());
  }

  std::string id() const override { return thermometer_ ? "rank_hot_encode" : "one_hot_encode"; }

  nlohmann::json params() const override {
    return {{"columns", explicit_columns_ ? nlohmann::json(columns_) : nlohmann::json()},
            {"use_cat_names", use_cat_names_},
            {"handle_unknown", unknown_name(unknown_)},
            {"handle_missing", missing_name(missing_)}};
  }

  void resolve(Schema& schema) override {
    bind_columns(schema, stage, columns_, explicit_columns_, fitted,
                 thermometer_ ? "rank-hot-encode" : "one-hot-encode");
    // Until the fit has run there are no categories, so the encoder consumes
    // its columns and emits nothing; the engine resolves again afterwards.
    outputs.clear();
    if (fitted) {
      for (const std::string& name : columns_) {
        for (const std::string& output : names_.at(name)) outputs.push_back(output);
      }
    }
    // The encoded columns leave first, so a generated name only collides with
    // a column that survives the step (hash_encode erases in the same order,
    // and encoding 'a' beside 'a_1' has to mean the same thing here).
    std::erase_if(schema.cols, [&](const auto& col) {
      return std::find(columns_.begin(), columns_.end(), col.first) != columns_.end();
    });
    reject_duplicate_outputs(outputs, stage);
    for (const std::string& name : outputs) {
      if (schema.find(name) != nullptr) throw Error(stage + ": output column '" + name + "' already exists in input");
    }
    for (const std::string& name : outputs) schema.cols.emplace_back(name, "i64");
    inputs = columns_;
  }

  bool needs_fit() const override { return true; }
  std::unique_ptr<AccState> make_state() const override { return std::make_unique<SeenState>(); }

  void consume(AccState& raw, const Batch& batch) const override {
    consume_categories(columns_, stage, static_cast<SeenState&>(raw), batch);
  }

  void finalize(std::vector<std::unique_ptr<AccState>> states) override {
    tables_ = merge_categories(columns_, stage, states, missing_);
    name_outputs();
    fitted = true;
  }

  nlohmann::json state() const override { return categories_state(columns_, tables_); }

  void load_state(const nlohmann::json& state) override {
    tables_ = load_categories(state, stage, columns_, explicit_columns_);
    name_outputs();
    inputs = columns_;
    fitted = true;
  }

  Batch transform(Batch batch) const override {
    Batch out;
    out.rows = batch.rows;
    for (std::size_t i = 0; i < batch.names.size(); ++i) {
      if (std::find(columns_.begin(), columns_.end(), batch.names[i]) != columns_.end()) continue;
      out.names.push_back(std::move(batch.names[i]));
      out.cols.push_back(std::move(batch.cols[i]));
    }
    for (const std::string& name : columns_) {
      const int idx = batch.find(name);
      if (idx < 0) throw Error(stage + ": column '" + name + "' missing from batch");
      const arrow::Array& array = *batch.cols[static_cast<std::size_t>(idx)];
      const std::vector<std::string>& names = names_.at(name);
      const Categories& table = tables_.at(name);
      const std::string where = stage + ": column '" + name + "'";
      auto validity = std::make_shared<OutputValidity>(array.length(), where);
      std::vector<PrimitiveOutput<arrow::Int64Type>> columns;
      columns.reserve(names.size());
      // Rank-hot fills every column itself, below; one-hot sets one per row.
      for (std::size_t c = 0; c < names.size(); ++c) {
        columns.emplace_back(array.length(), where, !thermometer_, validity);
      }
      // Rank-hot lights every column up to a row's rank, and storing a row
      // across that many columns at once makes the stores evict each other
      // from the cache, as in the contrast coders. So it records each row's
      // rank and then fills one column at a time. -1 is a null or an unknown
      // under 'value', which lights nothing.
      std::vector<int32_t> rank_of(thermometer_ ? static_cast<std::size_t>(array.length()) : 0, -1);
      for_each_encoded_utf8(
          array, stage, name,
          [&](bool is_null, std::string_view value) { return position(name, table, is_null, value); },
          [&](int64_t row, const std::optional<int64_t>& hot) {
            if (!hot.has_value()) {
              validity->set_null(row);
              return;
            }
            if (*hot < 0) return;  // an unknown under 'value': every column stays 0
            if (thermometer_) {
              rank_of[static_cast<std::size_t>(row)] = static_cast<int32_t>(*hot);
            } else {
              columns[static_cast<std::size_t>(*hot)].mutable_values()[row] = 1;
            }
          });
      if (thermometer_) {
        for (std::size_t c = 0; c < names.size(); ++c) {
          int64_t* values = columns[c].mutable_values();
          const auto rank = static_cast<int32_t>(c);
          for (std::size_t row = 0; row < rank_of.size(); ++row) values[row] = rank_of[row] >= rank ? 1 : 0;
        }
      }
      for (std::size_t c = 0; c < names.size(); ++c) {
        out.names.push_back(names[c]);
        out.cols.push_back(columns[c].finish());
      }
    }
    return out;
  }

  Row transform_row(Row row) const override {
    RecordLookup lookup(row, columns_.size());
    Row out;
    // Sized from what the fit froze, never from `outputs`: resolve() rewrites
    // that member under a mutex while this may be running on another thread.
    out.reserve(row.size() + generated_);
    for (auto& [key, value] : row) {
      if (std::find(columns_.begin(), columns_.end(), key) != columns_.end()) continue;
      // A passthrough field named like a generated column would be dropped
      // silently; the batch path refuses the same shape at resolve.
      if (output_index_.contains(key)) throw Error(stage + ": key '" + key + "' already present in record");
      out.emplace_back(key, std::move(value));
    }
    for (const std::string& name : columns_) {
      const RowValue* value = lookup.find(name);
      if (value == nullptr) throw Error(stage + ": key '" + name + "' missing from record");
      const bool is_null = std::holds_alternative<std::monostate>(*value);
      const std::string* text = std::get_if<std::string>(value);
      if (!is_null && text == nullptr) throw Error(stage + ": key '" + name + "' must be a string or null");
      const std::optional<int64_t> hot =
          position(name, tables_.at(name), is_null, is_null ? std::string_view{} : *text);
      const std::vector<std::string>& names = names_.at(name);
      for (std::size_t c = 0; c < names.size(); ++c) {
        if (!hot.has_value()) {
          out.emplace_back(names[c], std::monostate{});
        } else if (*hot < 0) {
          out.emplace_back(names[c], int64_t{0});
        } else {
          const auto index = static_cast<std::size_t>(*hot);
          const bool set = thermometer_ ? c <= index : c == index;
          out.emplace_back(names[c], set ? int64_t{1} : int64_t{0});
        }
      }
    }
    return out;
  }

 private:
  // Which output column a value lights up: an index, -1 for an unknown that
  // lights up nothing, or nullopt for a null row.
  std::optional<int64_t> position(const std::string& name, const Categories& table, bool is_null,
                                  std::string_view value) const {
    if (is_null) {
      if (missing_ == Missing::ReturnNan) return std::nullopt;
      refuse_null(missing_, stage, name);
      if (table.has_null) return static_cast<int64_t>(table.sorted.size());
      // A null that was never a fitted category. one_hot reads it out of a
      // mapping row that is zeros whatever handle_unknown says; rank_hot has
      // no row for it and falls through to the same default an unseen
      // category takes, so 'return_nan' nulls the row here and 'value' zeroes
      // it. 'error' does not refuse it (upstream only refuses an unseen
      // category there, never an unfitted null), and 'indicator' is not an
      // option rank_hot takes at all.
      if (thermometer_ && unknown_ == Unknown::ReturnNan) return std::nullopt;
      return -1;  // nothing lights up, not even the indicator
    }
    const int64_t rank = table.rank(value);
    if (rank >= 0) return rank;
    return unknown_position(name, table);
  }

  // Only an unseen category takes this path; an unseen null is handled above.
  std::optional<int64_t> unknown_position(const std::string& name, const Categories& table) const {
    switch (unknown_) {
      case Unknown::Value:
        return -1;
      case Unknown::ReturnNan:
        return std::nullopt;
      case Unknown::Indicator:
        return static_cast<int64_t>(table.levels());
      case Unknown::Error:
        break;
    }
    throw Error(stage + ": column '" + name + "' has a category that was not seen at fit");
  }

  // '<column>_<code>' by default, '<column>_<category>' with use_cat_names;
  // a name that would repeat gets '#' appended, as upstream does. `seen`
  // spans every encoded column, because upstream's does: encoding 'a' whose
  // category is 'b_c' beside 'a_b' whose category is 'c' gives 'a_b_c' and
  // 'a_b_c#', not the same name twice.
  void name_outputs() {
    names_.clear();
    output_index_.clear();
    generated_ = 0;
    std::unordered_map<std::string, int> seen;
    // These two encoders fit their pre-encoder with their own handle_missing,
    // so its slot for the missing value only survives 'return_nan' (see
    // missing_placeholder in kernels/categories.hpp), and one_hot then drops
    // the slot's column outright — as upstream does.
    const std::size_t placeholder = thermometer_ && missing_ == Missing::ReturnNan ? 1 : 0;
    for (const std::string& name : columns_) {
      const Categories& table = tables_.at(name);
      check_output_width(table.levels() + placeholder + (unknown_ == Unknown::Indicator ? 1 : 0), table.levels(), stage,
                         name);
      std::vector<std::string> generated;
      auto push = [&](const std::string& suffix) {
        std::string candidate = name + "_" + suffix;
        while (seen.contains(candidate)) candidate += "#";
        seen.emplace(candidate, 1);
        output_index_.emplace(candidate, generated_++);
        generated.push_back(std::move(candidate));
      };
      for (std::size_t i = 0; i < table.sorted.size(); ++i) {
        push(use_cat_names_ ? table.sorted[i] : std::to_string(static_cast<int64_t>(i) + 1));
      }
      if (table.has_null) push(use_cat_names_ ? "nan" : std::to_string(table.sorted.size() + 1));
      // The placeholder column, named for the -2 code the pre-encoder gives a
      // null it never saw. Nothing ever lights it up: every fitted category
      // ranks below it, and a null is a row of nulls under this option.
      if (placeholder > 0) push(use_cat_names_ ? "nan" : "-2");
      if (unknown_ == Unknown::Indicator) push("-1");
      names_.emplace(name, std::move(generated));
    }
  }

  bool thermometer_ = false;
  std::string what_;
  std::vector<std::string> columns_;
  bool explicit_columns_ = false;
  bool use_cat_names_ = false;
  Unknown unknown_ = Unknown::Value;
  Missing missing_ = Missing::Value;
  std::unordered_map<std::string, Categories> tables_;
  std::unordered_map<std::string, std::vector<std::string>> names_;
  // Both frozen by the fit, so the record path never reads a resolve-owned
  // member: the generated names, and how many of them there are.
  StringIndex output_index_;
  std::size_t generated_ = 0;
};

}  // namespace

std::unique_ptr<Kernel> make_ordinal_encode(const nlohmann::json& params) {
  return std::make_unique<OrdinalEncode>(params);
}

std::unique_ptr<Kernel> make_one_hot_encode(const nlohmann::json& params) {
  return std::make_unique<Expand>(params, false);
}

std::unique_ptr<Kernel> make_rank_hot_encode(const nlohmann::json& params) {
  return std::make_unique<Expand>(params, true);
}

}  // namespace fastfeaturetransform
