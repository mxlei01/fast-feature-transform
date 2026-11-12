// The shrinkage target encoders — james_stein_encode, leave_one_out_encode and
// cat_boost_encode. All three replace a categorical column with a number read
// off the target's per-category statistics, pulled toward the target's global
// mean; they differ only in how much weight a category's own mean keeps. The
// target is named in the params, not handed to fit, so a fitted artifact
// rebuilds from the manifest alone; the fitted table is per-category and
// order-free, so it never depends on how the rows were split across workers.

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

namespace fastfeaturetransform {
namespace {

// Target values stay below 2^450, so a squared value stays below 2^900 and no
// running sum of them can leave f64's range (that would take 2^124 of them).
// Bounding each value keeps a fit from failing on one thread count and passing
// on another.
constexpr double kMaxTarget = 0x1p450;

// One category's target statistics. The sums are exact (Shewchuk partials), so
// merging worker states is a monoid: any split gives the same table.
struct Acc {
  int64_t count = 0;
  ExactSum sum;
  ExactSum square;  // sum of squares, kept only by the encoders that need a variance

  [[nodiscard]] bool add(double value, bool squares) {
    ++count;
    if (!sum.add(value)) return false;
    if (!squares) return true;
    // Split the square in two so the sum of squares is exact as well: hi is
    // the rounded product and lo the part it dropped (exact under FMA).
    const double hi = value * value;
    return square.add(hi) && square.add(std::fma(value, value, -hi));
  }
  [[nodiscard]] bool merge(const Acc& other) {
    count += other.count;
    return sum.merge(other.sum) && square.merge(other.square);
  }
};

// ---------------------------------------------------------------------------
// The half the three shrinkage encoders share: the params, the target column,
// the per-category {count, sum, sum of squares} accumulators, the artifact and
// both transform paths. A subclass supplies only its per-category value.
// params: {"columns": [...] | null, "target": column,
//          "handle_unknown": "value" | "return_nan" | "error",
//          "handle_missing": "value" | "return_nan" | "error"}
// Semantics:
//   * the target is named in the params (never passed to fit) and must be an
//     integer or float column that is not itself encoded; it passes through to
//     the output untouched, so a trailing Select drops it
//   * an unfitted encoder reads the target as well as its columns; a fitted
//     one reads only its columns, so a serving file without the label still
//     transforms
//   * a row whose target is null contributes nothing to any statistic, as if
//     it were not there, though a null in an encoded column on that row still
//     counts toward handle_missing='error'; an integer target is widened to
//     f64, as upstream does
//   * a null is a category of its own — it counts toward every statistic, as
//     upstream's ordinal encoder makes it — and yields that category's value
//     under handle_missing='value', a null under 'return_nan', and an error
//     under 'error', at transform as well as at fit; a null never seen at fit
//     reads as the prior
//   * an unseen category yields the prior under handle_unknown='value', a null
//     under 'return_nan', and an error under 'error'
//   * every encoded column becomes f64; columns=null encodes every utf8 column
//     that reaches the step
//   * the sums are exact and the table is emitted in sorted key order, so the
//     artifact is the same however the rows were split across workers
// ---------------------------------------------------------------------------
class Shrinkage : public Kernel {
 public:
  Shrinkage(const nlohmann::json& params, std::string what, bool squares) : what_(std::move(what)), squares_(squares) {
    if (params.contains("columns") && !params.at("columns").is_null()) {
      columns_ = column_list(params, "columns", what_);
      explicit_columns_ = true;
    }
    if (!params.contains("target") || !params.at("target").is_string() ||
        params.at("target").get<std::string>().empty()) {
      throw Error(what_ + ": 'target' must be a column name");
    }
    target_ = params.at("target").get<std::string>();
    unknown_ = unknown_from(params, what_, false);
    missing_ = missing_from(params, what_);
    bind();
  }

  nlohmann::json params() const override {
    nlohmann::json out = extra_params();
    out["columns"] = explicit_columns_ ? nlohmann::json(columns_) : nlohmann::json();
    out["target"] = target_;
    out["handle_unknown"] = unknown_name(unknown_);
    out["handle_missing"] = missing_name(missing_);
    return out;
  }

  void resolve(Schema& schema) override {
    // A fitted encoder's columns come from its state and are never re-derived,
    // and it no longer reads the target, which may be absent when serving.
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
    bind_columns(schema, stage, columns_, explicit_columns_, fitted, "encode", &target_);
    for (auto& [name, type] : schema.cols) {
      if (std::find(columns_.begin(), columns_.end(), name) != columns_.end()) type = "f64";
    }
    bind();
  }

  bool needs_fit() const override { return true; }

  struct Table {
    StringIndex slots;  // category -> position in accs
    std::vector<Acc> accs;
    Acc nulls;              // the null category, which counts like any other
    int64_t null_rows = 0;  // every null in the column, for handle_missing

    std::size_t slot(std::string_view key) {
      auto found = slots.find(key);
      if (found != slots.end()) return found->second;
      slots.emplace(std::string(key), accs.size());
      accs.emplace_back();
      return accs.size() - 1;
    }

    Acc& at(std::string_view key) { return accs[slot(key)]; }
  };

  struct State final : AccState {
    std::unordered_map<std::string, Table> tables;  // column -> its categories
    Acc global;                                     // the target over every row it saw
  };

  std::unique_ptr<AccState> make_state() const override { return std::make_unique<State>(); }

  void consume(AccState& raw, const Batch& batch) const override {
    auto& state = static_cast<State&>(raw);
    const int index = batch.find(target_);
    if (index < 0) throw Error(stage + ": target column '" + target_ + "' missing from batch");
    const arrow::Array& targets = *batch.cols[static_cast<std::size_t>(index)];

    // Read the target once, then every column reuses it row by row.
    std::vector<double> values(static_cast<std::size_t>(targets.length()), 0.0);
    std::vector<bool> known(static_cast<std::size_t>(targets.length()), false);
    auto keep = [&](int64_t row, bool is_null, double value) {
      if (is_null) return;  // a row without a target contributes nothing anywhere
      check_target(value);
      values[static_cast<std::size_t>(row)] = value;
      known[static_cast<std::size_t>(row)] = true;
      if (!state.global.add(value, squares_)) throw Error(overflow());
    };
    if (target_family_ == Family::Integer) {
      for_each_integer(targets, stage, target_, [&](int64_t row, bool is_null, int64_t value) {
        keep(row, is_null, static_cast<double>(value));
      });
    } else {
      for_each_floating(targets, stage, target_, keep);
    }

    for (const std::string& name : columns_) {
      const int column = batch.find(name);
      if (column < 0) throw Error(stage + ": column '" + name + "' missing from batch");
      Table& table = state.tables[name];
      for_each_encoded_utf8_if(
          *batch.cols[static_cast<std::size_t>(column)], stage, name,
          [&](int64_t row, bool is_null) {
            // Every null counts toward handle_missing='error', even one with
            // a null target.
            if (is_null) ++table.null_rows;
            return known[static_cast<std::size_t>(row)];
          },
          [&](bool is_null, std::string_view value) -> std::optional<std::size_t> {
            return is_null ? std::nullopt : std::optional<std::size_t>{table.slot(value)};
          },
          [&](int64_t row, const std::optional<std::size_t>& category) {
            Acc& acc = category.has_value() ? table.accs[*category] : table.nulls;
            if (!acc.add(values[static_cast<std::size_t>(row)], squares_)) throw Error(overflow());
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

    // Without a target there is nothing to learn: every category would fall
    // back to a prior of zero, and a mis-joined label column would fit, save,
    // load and serve a constant with nothing anywhere saying so.
    if (merged->global.count == 0) {
      throw Error(stage + ": target column '" + target_ + "' has no non-null values to learn from");
    }
    rows_ = merged->global.count;
    prior_ = mean_of(merged->global);
    target_var_ = variance_of(merged->global);
    tables_.clear();
    null_values_.clear();
    for (const std::string& name : columns_) {
      Table& table = merged->tables[name];
      if (missing_ == Missing::Error && table.null_rows > 0) {
        throw Error(stage + ": column '" + name + "' has nulls and handle_missing is 'error'");
      }
      // Sorted, so the fitted table is the same however the rows were split.
      std::vector<Category> categories;
      categories.reserve(table.slots.size() + 1);
      for (const auto& [key, slot] : table.slots) {
        if (!arrow::util::ValidateUTF8(key)) {
          throw Error(stage + ": column '" + name + "' has a category that is not valid UTF-8; " + what_ +
                      " keys must be UTF-8 to serialize");
        }
        categories.push_back(category_of(key, false, table.accs[slot]));
      }
      std::sort(categories.begin(), categories.end(),
                [](const Category& a, const Category& b) { return a.key < b.key; });
      // The null is a category like any other for the statistics, but it has
      // no key to sort by, so it goes last.
      if (table.nulls.count > 0) categories.push_back(category_of({}, true, table.nulls));

      const std::vector<double> values = encode(categories);
      StringValues& into = tables_[name];
      for (std::size_t i = 0; i < categories.size(); ++i) {
        // A big enough 'a' sends cat_boost_encode's quotient to infinity,
        // which state() refuses to write and load_state to read, so the fit
        // refuses it here and names the param.
        const double value = finite_encoding(values[i], stage, name, encode_param());
        if (!categories[i].is_null) {
          into.emplace(categories[i].key, value);
        } else if (missing_ == Missing::Value) {
          null_values_.emplace(name, value);
        }
      }
    }
    merged.reset();
    index_tables();
    fitted = true;
    bind();
  }

  nlohmann::json state() const override {
    nlohmann::json values = nlohmann::json::object();
    nlohmann::json null_values = nlohmann::json::object();
    for (const std::string& name : columns_) {
      nlohmann::json entries = nlohmann::json::object();
      for (const auto& [key, value] : tables_.at(name)) entries[key] = serializable(value, stage, name);
      values[name] = std::move(entries);
      auto found = null_values_.find(name);
      null_values[name] =
          found == null_values_.end() ? nlohmann::json() : nlohmann::json(serializable(found->second, stage, name));
    }
    return {{"columns", columns_}, {"prior", prior_}, {"values", values}, {"null_values", null_values}};
  }

  void load_state(const nlohmann::json& state) override {
    if (!state.is_object() || !state.contains("prior") || !state.at("prior").is_number() || !state.contains("values") ||
        !state.at("values").is_object()) {
      throw Error(stage + ": state must hold a numeric 'prior' and a 'values' object");
    }
    prior_ = state.at("prior").get<double>();
    if (!std::isfinite(prior_)) throw Error(stage + ": 'prior' must be a finite number");
    const nlohmann::json& values = state.at("values");
    load_fitted_columns(state, stage, columns_, explicit_columns_, &values);
    // The null category is the entry a hand-edited manifest is likeliest to
    // mangle, so a wrong shape there is an error rather than a default.
    const nlohmann::json* null_values = optional_per_column_object(state, "null_values", columns_, stage);
    tables_.clear();
    null_values_.clear();
    for (const std::string& name : columns_) {
      if (!values.contains(name) || !values.at(name).is_object()) {
        throw Error(stage + ": values['" + name + "'] must be an object");
      }
      StringValues& into = tables_[name];
      for (const auto& [key, value] : values.at(name).items()) {
        if (!value.is_number() || !std::isfinite(value.get<double>())) {
          throw Error(stage + ": values['" + name + "']['" + key + "'] must be a finite number");
        }
        into.emplace(key, value.get<double>());
      }
      if (null_values == nullptr || null_values->at(name).is_null()) continue;
      const nlohmann::json& value = null_values->at(name);
      if (!value.is_number() || !std::isfinite(value.get<double>())) {
        throw Error(stage + ": null_values['" + name + "'] must be a finite number or null");
      }
      null_values_.emplace(name, value.get<double>());
    }
    index_tables();
    fitted = true;
    bind();
  }

  Batch transform(Batch batch) const override {
    for (const std::string& name : columns_) {
      const int index = batch.find(name);
      if (index < 0) throw Error(stage + ": column '" + name + "' missing from batch");
      const arrow::Array& array = *batch.cols[static_cast<std::size_t>(index)];
      PrimitiveOutput<arrow::DoubleType> values(array.length(), stage + ": column '" + name + "'");
      const StringValues& table = tables_.at(name);
      const ShortValues<double>& fast = short_tables_.at(name);
      for_each_encoded_utf8(
          array, stage, name,
          [&](bool is_null, std::string_view value) {
            return is_null ? look_up(name, true, value) : look_up(table, fast, name, value);
          },
          [&](int64_t row, const std::optional<double>& encoded) {
            if (encoded.has_value()) {
              values.mutable_values()[row] = *encoded;
            } else {
              values.set_null(row);
            }
          });
      batch.cols[static_cast<std::size_t>(index)] = values.finish();
    }
    return batch;
  }

  Row transform_row(Row row) const override {
    const RecordLookup lookup(row, columns_.size());
    for (const std::string& name : columns_) {
      RowValue* value = lookup.find(name);
      if (value == nullptr) throw Error(stage + ": key '" + name + "' missing from record");
      const bool is_null = std::holds_alternative<std::monostate>(*value);
      const std::string* text = std::get_if<std::string>(value);
      if (!is_null && text == nullptr) throw Error(stage + ": key '" + name + "' must be a string or null");
      const std::optional<double> encoded = look_up(name, is_null, is_null ? std::string_view{} : *text);
      if (encoded.has_value()) {
        *value = *encoded;
      } else {
        *value = std::monostate{};
      }
    }
    return row;
  }

 protected:
  // One category of one column at finalize, in sorted key order with the null
  // category (when there is one) last.
  struct Category {
    std::string key;
    bool is_null = false;
    int64_t count = 0;
    double sum = 0.0;
    double variance = 0.0;  // the sample variance (variance_of), only for the encoders that asked

    double mean() const { return count == 0 ? 0.0 : sum / static_cast<double>(count); }
  };

  // Every category's encoded value, in the order they arrive.
  virtual std::vector<double> encode(const std::vector<Category>& categories) const = 0;
  // The params this encoder adds to the shared four.
  virtual nlohmann::json extra_params() const = 0;
  // The param an encoding that is not finite points at.
  virtual const char* encode_param() const { return "target"; }

  double prior_ = 0.0;       // the target's mean over the rows that had one
  double target_var_ = 0.0;  // its sample variance (ddof=1), 0 for fewer than two rows
  int64_t rows_ = 0;         // rows whose target was not null

 private:
  // Each column's table again, through a perfect hash when its keys allow.
  void index_tables() {
    short_tables_.clear();
    for (const auto& [name, table] : tables_) short_tables_[name].index(table);
  }

  void bind() {
    inputs = columns_;
    // Fit reads the target too; once fitted, transform reads only the columns.
    if (!fitted) inputs.push_back(target_);
    outputs = columns_;
  }

  void check_target(double value) const {
    if (!std::isfinite(value)) {
      throw Error(stage + ": target column '" + target_ + "': value " + std::format("{}", value) +
                  " is not finite; a target cannot be NaN or infinity");
    }
    if (std::fabs(value) >= kMaxTarget) {
      throw Error(stage + ": target column '" + target_ + "': value " + std::format("{}", value) +
                  " is 2^450 or more in magnitude, more than the squared sums can hold (scale the target first)");
    }
  }

  std::string overflow() const { return stage + ": target column '" + target_ + "': the sums overflow f64"; }

  static double mean_of(const Acc& acc) {
    return acc.count == 0 ? 0.0 : acc.sum.value() / static_cast<double>(acc.count);
  }

  // The sample variance (ddof=1) pandas' var() computes; a category of one has
  // none, which upstream reads as zero. n * sum(x^2) - sum(x)^2 is n times the
  // sum of squared deviations, and it is formed exactly from the exact sums
  // before its one rounding. Rounding the sums first leaves two terms that
  // cancel: for a target near 1e9 with a spread of 1, n * sum(x^2) and sum(x)^2
  // agree in every digit a double keeps, and their difference is noise.
  //
  // REGRESSION: the products are formed from the sums' terms(), which depend
  // on their values alone. They used to be formed from the partials, which
  // two workers can leave split differently than one. Near 1e-155 the
  // products underflow and round, so a category's value depended on the
  // thread count in its last bits.
  double variance_of(const Acc& acc) const {
    if (acc.count < 2) return 0.0;
    const auto n = static_cast<double>(acc.count);
    ExactSum scaled;
    bool held = true;
    const std::vector<double> squares = acc.square.terms();
    const std::vector<double> sums = acc.sum.terms();
    for (double square : squares) held = held && add_product(scaled, n, square);
    for (double left : sums) {
      for (double right : sums) held = held && add_product(scaled, -left, right);
    }
    if (!held) throw Error(overflow());
    return std::max(0.0, scaled.value() / (n * (n - 1)));
  }

  Category category_of(std::string key, bool is_null, const Acc& acc) const {
    return Category{std::move(key), is_null, acc.count, acc.sum.value(), squares_ ? variance_of(acc) : 0.0};
  }

  void merge_into(State& into, State& from) const {
    if (!into.global.merge(from.global)) throw Error(overflow());
    for (auto& [name, table] : from.tables) {
      Table& target = into.tables[name];
      if (!target.nulls.merge(table.nulls)) throw Error(overflow());
      target.null_rows += table.null_rows;
      for (const auto& [key, slot] : table.slots) {
        if (!target.at(key).merge(table.accs[slot])) throw Error(overflow());
      }
    }
  }

  // The encoded value for one input value, or nullopt when the output is null.
  std::optional<double> look_up(const std::string& name, bool is_null, std::string_view value) const {
    if (is_null) {
      if (missing_ == Missing::ReturnNan) return std::nullopt;
      refuse_null(missing_, stage, name);
      auto found = null_values_.find(name);
      // A null the fit never saw reads as the prior, as an unseen category does.
      return found == null_values_.end() ? prior_ : found->second;
    }
    return look_up(tables_.at(name), short_tables_.at(name), name, value);
  }

  std::optional<double> look_up(const StringValues& table, const ShortValues<double>& fast, const std::string& name,
                                std::string_view value) const {
    if (const double* found = fast.find(table, value)) return *found;
    switch (unknown_) {
      case Unknown::Value:
        return prior_;
      case Unknown::ReturnNan:
        return std::nullopt;
      case Unknown::Indicator:
      case Unknown::Error:
        break;
    }
    throw Error(stage + ": column '" + name + "' has a category that was not seen at fit");
  }

  std::string what_;
  bool squares_;
  std::vector<std::string> columns_;
  bool explicit_columns_ = false;
  std::string target_;
  Family target_family_ = Family::Floating;
  Unknown unknown_ = Unknown::Value;
  Missing missing_ = Missing::Value;
  std::unordered_map<std::string, StringValues> tables_;               // column -> category -> encoded value
  std::unordered_map<std::string, ShortValues<double>> short_tables_;  // the same, indexed
  std::unordered_map<std::string, double> null_values_;                // column -> the null category's value
};

// ---------------------------------------------------------------------------
// james_stein_encode — the target mean of a category, shrunk toward the global
// mean by a closed-form weight.
// params: the shared four, plus {"model": "independent" | "pooled" | "beta"}.
// Semantics (m_c = the category's target mean, K = categories, N = rows):
//   * independent (the default) weighs each category on its own variance:
//     B_c = i_var_c / (var(y) + i_var_c) * (K - 3) / (K - 1) with i_var_c the
//     within-category sample variance (0 for a category of one), the weight is
//     clip(1 - B_c, 0, 1), and the value is w_c*m_c + (1 - w_c)*prior
//   * pooled assumes one variance for every category: variance = var(y)/(N/K),
//     SSE = sum (m_c - prior)^2, B = clip((K - 3)*variance/SSE, 0, 1), and the
//     value is prior + (1 - B)*(m_c - prior); an SSE of zero means every mean
//     is already the prior
//   * beta weighs a category by its size alone: w_c = n_c/(n_c + N)
//   * as many categories as rows means the column identifies the row, so every
//     category collapses to the prior (upstream's id-like guard)
//   * model='binary' is refused: upstream reads its estimate out of a module
//     global that its optimizer happens to leave behind, so the result is not
//     reproducible from the published formula
//   * a constant target leaves the independent weight 0/0, which upstream
//     answers with NaN; we keep the category mean, which is the prior
//   * upstream's randomized/sigma add Gaussian noise at training time only, so
//     there is nothing to reproduce at inference and they are not params here
// ---------------------------------------------------------------------------
enum class Model { Independent, Pooled, Beta };

class JamesSteinEncode final : public Shrinkage {
 public:
  explicit JamesSteinEncode(const nlohmann::json& params) : Shrinkage(params, "james_stein_encode", true) {
    if (!params.contains("model") || params.at("model").is_null()) return;
    if (!params.at("model").is_string()) throw Error(kModels);
    const std::string text = params.at("model").get<std::string>();
    if (text == "independent") return;
    if (text == "pooled") {
      model_ = Model::Pooled;
    } else if (text == "beta") {
      model_ = Model::Beta;
    } else if (text == "binary") {
      throw Error(
          "james_stein_encode: model='binary' reads its estimate from a module global that category_encoders' "
          "optimizer leaves behind, so it is not reproducible from the formula and not portable; choose "
          "'independent', 'pooled' or 'beta'");
    } else {
      throw Error(kModels);
    }
  }

  std::string id() const override { return "james_stein_encode"; }

 protected:
  nlohmann::json extra_params() const override {
    return {{"model", model_ == Model::Independent ? "independent" : model_ == Model::Pooled ? "pooled" : "beta"}};
  }

  std::vector<double> encode(const std::vector<Category>& categories) const override {
    const auto count = static_cast<double>(categories.size());
    std::vector<double> values;
    values.reserve(categories.size());
    // One category per row: the column identifies the row and teaches nothing.
    if (static_cast<int64_t>(categories.size()) == rows_) return std::vector<double>(categories.size(), prior_);
    const double pooled_weight = model_ == Model::Pooled ? pooled_shrinkage(categories, count) : 0.0;
    for (const Category& category : categories) {
      const double mean = category.mean();
      switch (model_) {
        case Model::Independent: {
          double weight = 1.0;  // a single category is its own prior, whatever the weight
          if (count >= 2) {
            const double variance = category.variance;
            const double total = target_var_ + variance;
            const double shrinkage = (total == 0.0 ? 0.0 : variance / total) * (count - 3) / (count - 1);
            weight = std::clamp(1.0 - shrinkage, 0.0, 1.0);
          }
          values.push_back(weight * mean + (1 - weight) * prior_);
          break;
        }
        case Model::Pooled:
          values.push_back(prior_ + (1 - pooled_weight) * (mean - prior_));
          break;
        case Model::Beta: {
          const auto n = static_cast<double>(category.count);
          const double weight = n / (n + static_cast<double>(rows_));
          values.push_back(weight * mean + (1 - weight) * prior_);
          break;
        }
      }
    }
    return values;
  }

 private:
  static constexpr const char* kModels = "james_stein_encode: 'model' must be 'independent', 'pooled' or 'beta'";

  // The one weight every category shares under the pooled model.
  double pooled_shrinkage(const std::vector<Category>& categories, double count) const {
    ExactSum errors;
    for (const Category& category : categories) {
      const double error = category.mean() - prior_;
      if (!errors.add(error * error)) throw Error(stage + ": the squared category means overflow f64");
    }
    const double sse = errors.value();
    if (sse <= 0) return 0.0;  // every mean is the prior already, so the weight cannot matter
    const double variance = target_var_ / (static_cast<double>(rows_) / count);
    return std::clamp((count - 3) * variance / sse, 0.0, 1.0);
  }

  Model model_ = Model::Independent;
};

// ---------------------------------------------------------------------------
// leave_one_out_encode — the target mean of a category, with a singleton
// collapsing to the prior.
// params: the shared four.
// Semantics:
//   * the value is sum_c/n_c when n_c > 1 and the prior otherwise, so a
//     category of one row cannot leak that row's own target
//   * this is upstream's inference form (its transform without a target).
//     Upstream's fit_transform instead leaves each training row's own target
//     out of its own value, which is a function of the row, not of the fitted
//     state; our transform is a pure function of frozen state, so those
//     training-time values are not reproduced
//   * upstream's sigma/random_state add Gaussian noise at training time only,
//     so there is nothing to reproduce at inference and they are not params
// ---------------------------------------------------------------------------
class LeaveOneOutEncode final : public Shrinkage {
 public:
  explicit LeaveOneOutEncode(const nlohmann::json& params) : Shrinkage(params, "leave_one_out_encode", false) {}

  std::string id() const override { return "leave_one_out_encode"; }

 protected:
  nlohmann::json extra_params() const override { return nlohmann::json::object(); }

  std::vector<double> encode(const std::vector<Category>& categories) const override {
    std::vector<double> values;
    values.reserve(categories.size());
    for (const Category& category : categories) {
      values.push_back(category.count > 1 ? category.mean() : prior_);
    }
    return values;
  }
};

// ---------------------------------------------------------------------------
// cat_boost_encode — the target mean of a category, smoothed toward the prior
// by `a` extra prior-valued observations.
// params: the shared four, plus {"a": number >= 0}.
// Semantics:
//   * the value is (sum_c + prior*a)/(n_c + a) when n_c > 1, and the raw prior
//     otherwise: a singleton is not smoothed, it is replaced
//   * this is upstream's inference form (its transform without a target).
//     Upstream's fit_transform instead walks the training rows in order and
//     gives each row the running statistic of the rows before it, which
//     depends on row order; our fits do not, so those training-time values are
//     not reproduced
//   * upstream's sigma/random_state add Gaussian noise at training time only;
//     there is nothing to reproduce at inference, so they are not params here
// ---------------------------------------------------------------------------
class CatBoostEncode final : public Shrinkage {
 public:
  explicit CatBoostEncode(const nlohmann::json& params) : Shrinkage(params, "cat_boost_encode", false) {
    if (!params.contains("a") || params.at("a").is_null()) return;
    const nlohmann::json& value = params.at("a");
    if (!value.is_number() || value.get<double>() < 0 || !std::isfinite(value.get<double>())) {
      throw Error("cat_boost_encode: 'a' must be a finite number >= 0");
    }
    a_ = value.get<double>();
  }

  std::string id() const override { return "cat_boost_encode"; }

 protected:
  nlohmann::json extra_params() const override { return {{"a", a_}}; }
  const char* encode_param() const override { return "a"; }

  std::vector<double> encode(const std::vector<Category>& categories) const override {
    std::vector<double> values;
    values.reserve(categories.size());
    for (const Category& category : categories) {
      const auto n = static_cast<double>(category.count);
      values.push_back(category.count > 1 ? (category.sum + prior_ * a_) / (n + a_) : prior_);
    }
    return values;
  }

 private:
  double a_ = 1.0;
};

}  // namespace

std::unique_ptr<Kernel> make_james_stein_encode(const nlohmann::json& params) {
  return std::make_unique<JamesSteinEncode>(params);
}

std::unique_ptr<Kernel> make_leave_one_out_encode(const nlohmann::json& params) {
  return std::make_unique<LeaveOneOutEncode>(params);
}

std::unique_ptr<Kernel> make_cat_boost_encode(const nlohmann::json& params) {
  return std::make_unique<CatBoostEncode>(params);
}

}  // namespace fastfeaturetransform
