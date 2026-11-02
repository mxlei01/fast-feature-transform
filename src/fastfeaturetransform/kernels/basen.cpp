// The positional encoders: a category's rank written out digit by digit, one
// output column per digit. base-N (basen_encode), its base-2 special case
// (binary_encode), and the reflected binary code (gray_encode), all sharing
// the fitted category table in kernels/categories.hpp.
//
// The code word of a category is the base-b spelling of its 1-based rank, so
// the word 0 is free and is what an unknown gets. That is why the width is
// sized for one level more than there are levels.

#include <algorithm>
#include <unordered_map>

#include "fastfeaturetransform/kernels/cast_table.hpp"
#include "fastfeaturetransform/kernels/categories.hpp"
#include "fastfeaturetransform/kernels/kernels.hpp"
#include "fastfeaturetransform/kernels/output.hpp"
#include "fastfeaturetransform/kernels/util.hpp"

namespace fastfeaturetransform {
namespace {

// ceil(log(n, base)) computed with integer arithmetic only (upstream's
// _ceillogint), and deliberately not std::log: log(125)/log(5) is
// 3.0000000000000004 in binary floating point, so its ceiling would be 4, a
// digit too many. Equivalently: how many base-`base` digits n - 1 needs.
// `base` must be 2 or more and `n` 1 or more, which the callers guarantee.
int64_t ceil_log_int(int64_t n, int64_t base) {
  int64_t left = n - 1;
  int64_t digits = 0;
  while (left > 0) {
    ++digits;
    left /= base;
  }
  return digits;
}

// Which of the three encoders a kernel is: the base and the spelling of a
// rank are all that differ between them.
enum class Flavor { BaseN, Binary, Gray };

// ---------------------------------------------------------------------------
// basen_encode / binary_encode / gray_encode — categorical utf8 -> `digits`
// integer columns holding the base-b spelling of the category's rank, most
// significant digit first, replacing the source column.
// params: {"columns": [...] | null, "base": int (basen_encode only),
//          "handle_unknown": "value" | "return_nan" | "indicator" | "error",
//          "handle_missing": "value" | "return_nan" | "error"}
// Semantics:
//   * categories sort; the category at 1-based rank r takes the `digits`-wide
//     base-b spelling of r, and gray_encode the spelling of r ^ (r >> 1)
//   * digits = ceil(log(levels + 1, base)) by integer arithmetic, so the
//     word 0 stays free for an unknown; base=1 is upstream's special case,
//     where a rank is a 1 at position r of levels + 1 columns (one-hot with
//     a leading column that is always zero)
//   * a null seen at fit under handle_missing='value' is a level of its own,
//     sorted last, with its own code word; a null that was not a fitted
//     level is the all-zero word, and every null is a row of nulls under
//     'return_nan'; handle_missing='error' refuses a null at transform as
//     well as at fit
//   * under handle_missing='return_nan' or 'error' the pre-encoder's slot for
//     the missing value is a level of its own that no value reaches, so it
//     can cost a digit; see missing_placeholder in kernels/categories.hpp
//   * an unseen category is the all-zero word under handle_unknown='value',
//     a row of nulls under 'return_nan', an error under 'error', and a real
//     extra level (the rank after every other level) under 'indicator'
//   * output columns are named '<column>_0' .. '<column>_{digits - 1}'; how
//     many there are is known only after the fit, so a later step in the
//     same pipeline cannot name them
// ---------------------------------------------------------------------------
class BaseN final : public Kernel {
 public:
  BaseN(const nlohmann::json& params, Flavor flavor) : flavor_(flavor), what_(kernel_id(flavor)) {
    if (params.contains("columns") && !params.at("columns").is_null()) {
      columns_ = column_list(params, "columns", what_);
      explicit_columns_ = true;
      inputs = columns_;
    }
    if (flavor_ == Flavor::BaseN && params.contains("base") && !params.at("base").is_null()) {
      if (!is_int64(params.at("base"))) throw Error(what_ + ": 'base' must be an integer from 1 to 2^63 - 1");
      base_ = params.at("base").get<int64_t>();
      if (base_ < 1) throw Error(what_ + ": 'base' must be 1 or more");
    }
    unknown_ = unknown_from(params, what_.c_str(), true);
    missing_ = missing_from(params, what_.c_str());
  }

  std::string id() const override { return what_; }

  nlohmann::json params() const override {
    nlohmann::json out{{"columns", explicit_columns_ ? nlohmann::json(columns_) : nlohmann::json()},
                       {"handle_unknown", unknown_name(unknown_)},
                       {"handle_missing", missing_name(missing_)}};
    // Only base-N has a base; upstream fixes it at 2 for the other two, so
    // spelling it in their params would invite a plan that contradicts them.
    if (flavor_ == Flavor::BaseN) out["base"] = base_;
    return out;
  }

  void resolve(Schema& schema) override {
    bind_columns(schema, stage, columns_, explicit_columns_, fitted, bind_verb(flavor_));
    // Until the fit has run there are no levels and so no width: the encoder
    // consumes its columns and emits nothing, and the engine resolves again
    // once the fit pass is done.
    outputs.clear();
    if (fitted) {
      for (const std::string& name : columns_) {
        for (const std::string& output : names_.at(name)) outputs.push_back(output);
      }
    }
    // The encoded columns leave first, so a generated name only collides with
    // a column that survives the step; hash_encode erases in the same order,
    // and encoding 'a' beside 'a_0' has to mean the same thing.
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
    lay_out();
    fitted = true;
  }

  nlohmann::json state() const override { return categories_state(columns_, tables_); }

  void load_state(const nlohmann::json& state) override {
    tables_ = load_categories(state, stage, columns_, explicit_columns_);
    lay_out();
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
      const Layout& layout = layouts_.at(name);
      const std::string where = stage + ": column '" + name + "'";
      auto validity = std::make_shared<OutputValidity>(array.length(), where);
      std::vector<PrimitiveOutput<arrow::Int64Type>> columns;
      columns.reserve(names.size());
      for (std::size_t c = 0; c < names.size(); ++c) columns.emplace_back(array.length(), where, true, validity);
      std::vector<int64_t> word(names.size(), 0);
      // A word depends on its fitted rank, including the indicator's special
      // non-Gray spelling. Cache referenced words only when the table fits in
      // 1 MiB and can be reused within this batch; base=1 must not make a
      // quadratic cache for a large vocabulary.
      const std::size_t slots = std::max(table.levels() + 1, static_cast<std::size_t>(layout.indicator_rank) + 1);
      const bool cache_words = !names.empty() && slots <= static_cast<std::size_t>(array.length()) &&
                               slots <= (1024 * 1024 / sizeof(int64_t)) / names.size();
      std::vector<int64_t> words(cache_words ? slots * names.size() : 0);
      std::vector<uint8_t> ready(cache_words ? slots : 0, 0);
      for_each_encoded_utf8(
          array, stage, name,
          [&](bool is_null, std::string_view value) { return word_for(name, table, layout, is_null, value); },
          [&](int64_t row, const std::optional<Word>& code) {
            if (!code.has_value()) {
              validity->set_null(row);
              return;
            }
            const int64_t* coded = nullptr;
            if (cache_words) {
              const auto rank = static_cast<std::size_t>(code->rank);
              int64_t* cached = words.data() + rank * names.size();
              if (!ready[rank]) {
                spell(*code, names.size(), word);
                std::copy(word.begin(), word.end(), cached);
                ready[rank] = 1;
              }
              coded = cached;
            } else {
              spell(*code, names.size(), word);
              coded = word.data();
            }
            for (std::size_t c = 0; c < names.size(); ++c) columns[c].mutable_values()[row] = coded[c];
          });
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
    std::vector<int64_t> word;
    for (const std::string& name : columns_) {
      const RowValue* value = lookup.find(name);
      if (value == nullptr) throw Error(stage + ": key '" + name + "' missing from record");
      const bool is_null = std::holds_alternative<std::monostate>(*value);
      const std::string* text = std::get_if<std::string>(value);
      if (!is_null && text == nullptr) throw Error(stage + ": key '" + name + "' must be a string or null");
      const std::optional<Word> code =
          word_for(name, tables_.at(name), layouts_.at(name), is_null, is_null ? std::string_view{} : *text);
      const std::vector<std::string>& names = names_.at(name);
      if (code.has_value()) spell(*code, names.size(), word);
      for (std::size_t c = 0; c < names.size(); ++c) {
        if (code.has_value()) {
          out.emplace_back(names[c], word[c]);
        } else {
          out.emplace_back(names[c], std::monostate{});
        }
      }
    }
    return out;
  }

 private:
  // Where one column's levels sit: the rank the fitted null takes (0 when
  // there is none), the rank an unknown takes under 'indicator' (0 when it
  // does not), and how many digits wide every code word is.
  struct Layout {
    int64_t null_rank = 0;
    int64_t indicator_rank = 0;
    std::size_t digits = 1;
  };

  // The code word one value takes: the rank whose spelling it gets (0 is the
  // all-zero word, shared by every base) and whether that rank is spelled in
  // Gray. The indicator level never is: upstream converts only the fitted
  // levels of a binary mapping to Gray and leaves the unknown's row alone,
  // so under gray_encode + 'indicator' the unknown keeps its binary word.
  struct Word {
    int64_t rank = 0;
    bool gray = false;
  };

  static const char* kernel_id(Flavor flavor) {
    switch (flavor) {
      case Flavor::BaseN:
        return "basen_encode";
      case Flavor::Binary:
        return "binary_encode";
      case Flavor::Gray:
        break;
    }
    return "gray_encode";
  }

  static const char* bind_verb(Flavor flavor) {
    switch (flavor) {
      case Flavor::BaseN:
        return "basen-encode";
      case Flavor::Binary:
        return "binary-encode";
      case Flavor::Gray:
        break;
    }
    return "gray-encode";
  }

  // Ranks and width per column, plus the output names. Derived from the
  // fitted categories and the params, so the artifact carries neither.
  void lay_out() {
    layouts_.clear();
    names_.clear();
    output_index_.clear();
    generated_ = 0;
    for (const std::string& name : columns_) {
      const Categories& table = tables_.at(name);
      const auto categories = static_cast<int64_t>(table.sorted.size());
      Layout layout;
      int64_t levels = categories;
      if (table.has_null) {
        ++levels;
        layout.null_rank = levels;  // the null sorts last, so it ranks last
      }
      // The pre-encoder's slot for the missing value is still a level under
      // every option but 'value', so every other mode pays for one level it
      // never spells — which costs a digit whenever the count crosses a power
      // of the base. See missing_placeholder in kernels/categories.hpp; the
      // contrast coders pay for the same slot the same way.
      int64_t width = levels + static_cast<int64_t>(missing_placeholder(missing_));
      if (unknown_ == Unknown::Indicator) {
        layout.indicator_rank = width + 1;  // a real extra level, after that slot
        ++width;
      }
      layout.digits = static_cast<std::size_t>(base_ == 1 ? width + 1 : ceil_log_int(width + 1, base_));
      check_output_width(layout.digits, static_cast<std::size_t>(levels), stage, name);
      std::vector<std::string> generated;
      for (std::size_t digit = 0; digit < layout.digits; ++digit) {
        std::string output = name + "_" + std::to_string(digit);
        output_index_.emplace(output, generated_++);
        generated.push_back(std::move(output));
      }
      layouts_.emplace(name, layout);
      names_.emplace(name, std::move(generated));
    }
  }

  std::optional<Word> word_for(const std::string& name, const Categories& table, const Layout& layout, bool is_null,
                               std::string_view value) const {
    const bool gray = flavor_ == Flavor::Gray;
    if (is_null) {
      if (missing_ == Missing::ReturnNan) return std::nullopt;
      refuse_null(missing_, stage, name);
      if (layout.null_rank > 0) return Word{layout.null_rank, gray};
      return Word{};  // a null that was never a fitted level: the all-zero word
    }
    const int64_t rank = table.rank(value);
    if (rank >= 0) return Word{rank + 1, gray};
    switch (unknown_) {
      case Unknown::Value:
        return Word{};
      case Unknown::ReturnNan:
        return std::nullopt;
      case Unknown::Indicator:
        return Word{layout.indicator_rank, false};
      case Unknown::Error:
        break;
    }
    throw Error(stage + ": column '" + name + "' has a category that was not seen at fit");
  }

  // Write a rank out as `digits` digits, most significant first.
  void spell(const Word& code, std::size_t digits, std::vector<int64_t>& word) const {
    word.assign(digits, 0);
    if (code.rank <= 0) return;  // the all-zero word
    // Gray: consecutive ranks differ in exactly one digit, which is what the
    // code is for. Only defined in base 2, the only base the Gray kernel has.
    int64_t left = code.gray ? code.rank ^ (code.rank >> 1) : code.rank;
    if (base_ == 1) {
      // Not really base 1: upstream's number_to_base puts a single 1 at
      // position n of levels + 1 columns, so this is one-hot with a leading
      // column that nothing ever lights up.
      if (static_cast<std::size_t>(left) < digits) word[static_cast<std::size_t>(left)] = 1;
      return;
    }
    for (std::size_t i = digits; i > 0 && left > 0; --i) {
      word[i - 1] = left % base_;
      left /= base_;
    }
  }

  Flavor flavor_ = Flavor::BaseN;
  std::string what_;
  std::vector<std::string> columns_;
  bool explicit_columns_ = false;
  int64_t base_ = 2;
  Unknown unknown_ = Unknown::Value;
  Missing missing_ = Missing::Value;
  std::unordered_map<std::string, Categories> tables_;
  std::unordered_map<std::string, Layout> layouts_;
  std::unordered_map<std::string, std::vector<std::string>> names_;
  // Both frozen by the fit, so the record path never reads a resolve-owned
  // member: the generated names, and how many of them there are.
  StringIndex output_index_;
  std::size_t generated_ = 0;
};

}  // namespace

std::unique_ptr<Kernel> make_basen_encode(const nlohmann::json& params) {
  return std::make_unique<BaseN>(params, Flavor::BaseN);
}

std::unique_ptr<Kernel> make_binary_encode(const nlohmann::json& params) {
  return std::make_unique<BaseN>(params, Flavor::Binary);
}

std::unique_ptr<Kernel> make_gray_encode(const nlohmann::json& params) {
  return std::make_unique<BaseN>(params, Flavor::Gray);
}

}  // namespace fastfeaturetransform
