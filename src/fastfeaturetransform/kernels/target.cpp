// The target encoders of category_encoders: target_encode, m_estimate_encode
// and woe_encode, and the shrinkage encoders james_stein_encode,
// leave_one_out_encode and cat_boost_encode. All six replace a categorical
// column with a number learned from a target column, and all six learn it
// from the same per-category {count, sum of the target} accumulator plus the
// fit-wide count and sum (james_stein_encode sums the squares as well); only
// the formula finalize applies to those numbers differs, so one kernel class
// serves the six ids.

#include <arrow/util/utf8.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "fastfeaturetransform/kernels/cast_table.hpp"
#include "fastfeaturetransform/kernels/categories.hpp"
#include "fastfeaturetransform/kernels/kernels.hpp"
#include "fastfeaturetransform/kernels/output.hpp"
#include "fastfeaturetransform/kernels/util.hpp"

namespace fastfeaturetransform {
namespace {

// Which statistic of a category's target a kernel computes.
enum class Formula { Target, MEstimate, Woe, JamesStein, LeaveOneOut, CatBoost };

// How james_stein_encode weighs a category's mean against the prior.
enum class Model { Independent, Pooled, Beta };

const char* formula_id(Formula formula) {
  switch (formula) {
    case Formula::Target:
      return "target_encode";
    case Formula::MEstimate:
      return "m_estimate_encode";
    case Formula::Woe:
      return "woe_encode";
    case Formula::JamesStein:
      return "james_stein_encode";
    case Formula::LeaveOneOut:
      return "leave_one_out_encode";
    case Formula::CatBoost:
      return "cat_boost_encode";
  }
  return "target_encode";
}

double positive_number(const nlohmann::json& params, const char* key, double fallback, const std::string& what) {
  if (!params.contains(key) || params.at(key).is_null()) return fallback;
  const nlohmann::json& value = params.at(key);
  if (!value.is_number() || value.get<double>() <= 0 || !std::isfinite(value.get<double>())) {
    throw Error(what + ": '" + std::string(key) + "' must be a positive number");
  }
  return value.get<double>();
}

double non_negative_number(const nlohmann::json& params, const char* key, double fallback, const std::string& what) {
  if (!params.contains(key) || params.at(key).is_null()) return fallback;
  const nlohmann::json& value = params.at(key);
  if (!value.is_number() || value.get<double>() < 0 || !std::isfinite(value.get<double>())) {
    throw Error(what + ": '" + std::string(key) + "' must be a finite number >= 0");
  }
  return value.get<double>();
}

Model model_from(const nlohmann::json& params) {
  const char* models = "james_stein_encode: 'model' must be 'independent', 'pooled' or 'beta'";
  if (!params.contains("model") || params.at("model").is_null()) return Model::Independent;
  if (!params.at("model").is_string()) throw Error(models);
  const std::string text = params.at("model").get<std::string>();
  if (text == "independent") return Model::Independent;
  if (text == "pooled") return Model::Pooled;
  if (text == "beta") return Model::Beta;
  if (text == "binary") {
    throw Error(
        "james_stein_encode: model='binary' reads its estimate from a module global that category_encoders' "
        "optimizer leaves behind, so it is not reproducible from the formula and not portable; choose "
        "'independent', 'pooled' or 'beta'");
  }
  throw Error(models);
}

// The logistic curve as scipy.special.expit computes it, which is what
// category_encoders calls: 1 / (1 + exp(-x)), in that grouping. The
// two-branch form, e / (1 + e) with e = exp(x) for a negative x, rounds
// differently, and for a category whose count is below min_samples_leaf it
// can miss upstream's last bit. This form needs no second branch: exp() of a
// large positive argument gives infinity, and 1 / (1 + infinity) is 0, not a
// NaN. Between about x = -745 and x = -710 it gives 0 where the two-branch
// form gives a subnormal; upstream gives 0 too, so the weight is 0 and the
// encoding is the prior.
double expit(double x) { return 1.0 / (1.0 + std::exp(-x)); }

// A float target of james_stein_encode, which sums the squares, stays below
// 2^450, so that a square stays below kMaxSummand, the bound of the others.
constexpr double kMaxSquared = 0x1p450;

// One category's share of the target: how many rows carried it and what they
// summed to. The 128-bit sum serves an integer target (it cannot overflow in
// practice) and the exact one a float target, so neither depends on how the
// rows were split across workers. james_stein_encode also sums the squares,
// exactly.
struct Acc {
  int64_t count = 0;
  Int128 isum = 0;
  ExactSum fsum;
  ExactSum squares;
};

double sum_of(const Acc& acc, Family family) {
  return family == Family::Integer ? static_cast<double>(acc.isum) : acc.fsum.value();
}

// A 128-bit integer into an exact sum, as three doubles that total it: the
// integer cut at bits 22 and 75, so that each piece has at most 53 bits and
// an f64 holds it exactly (the top piece keeps the sign). The cut takes
// shifts and masks, where converting the integer itself to f64 and back
// calls a library routine each way: a james_stein_encode fit of 1,000,000
// targets above 2^53 in 1,000 categories takes 30 ms this way and 50 ms
// through the conversions (one thread, Apple M4). Every integer added here
// is below 2^127, and a fit adds at most 2^63 of them to one sum, so no add
// can fail.
void add_whole(ExactSum& sum, Int128 value) {
  const Int128 low53 = (Int128{1} << 53) - 1;
  (void)sum.add(static_cast<double>(static_cast<int64_t>(value >> 75)) * 0x1p75);
  (void)sum.add(static_cast<double>(static_cast<int64_t>((value >> 22) & low53)) * 0x1p22);
  (void)sum.add(static_cast<double>(static_cast<int64_t>(value & ((Int128{1} << 22) - 1))));
}

// The doubles that total a category's sum exactly and depend on its value
// alone, as ExactSum::terms() gives them.
std::vector<double> sum_terms(const Acc& acc, Family family) {
  if (family != Family::Integer) return acc.fsum.terms();
  ExactSum sum;
  add_whole(sum, acc.isum);
  return sum.terms();
}

// An integer's square into an exact sum. Below 2^53 in magnitude the integer
// is its own f64, whose square add_product holds exactly; above, the square
// is formed in 128 bits. Out of line, so that the loops of the kernels that
// keep no squares stay short.
[[gnu::noinline]] void add_square(ExactSum& squares, int64_t value) {
  const auto x = static_cast<double>(value);
  if (std::fabs(x) < 0x1p53) {
    (void)add_product(squares, x, x);
  } else {
    add_whole(squares, static_cast<Int128>(value) * value);
  }
}

using AccMap = CategoryAccumulator<Acc>;

// How many of the smallest distinct target values the binary-target check
// keeps: two say the target is binary, a third says it is not.
constexpr std::size_t kDistinctProbe = 3;

// Taking the smallest distinct values (plus the largest separately) keeps
// the probe the same however the rows were split: the smallest three of a
// union are the smallest three of the two smallest-three sets.
void note_distinct(std::vector<double>& smallest, double& largest, double value) {
  largest = std::max(largest, value);
  auto at = std::lower_bound(smallest.begin(), smallest.end(), value);
  if (at != smallest.end() && *at == value) return;
  smallest.insert(at, value);
  if (smallest.size() > kDistinctProbe) smallest.pop_back();
}

// ---------------------------------------------------------------------------
// target_encode / m_estimate_encode / woe_encode / james_stein_encode /
// leave_one_out_encode / cat_boost_encode — categorical utf8 -> a number
// learned from a target column, in place.
// params: {"columns": [...] | null, "target": column,
//          "handle_unknown": "value" | "return_nan" | "error",
//          "handle_missing": "value" | "return_nan" | "error"} plus, per id,
//          "min_samples_leaf": int and "smoothing": float > 0 (target_encode),
//          "m": float >= 0 (m_estimate_encode),
//          "regularization": float > 0 (woe_encode),
//          "model": "independent" | "pooled" | "beta" (james_stein_encode), or
//          "a": float >= 0 (cat_boost_encode).
// Semantics:
//   * the target is named in the params, not passed to fit: it is an integer
//     or float column of the input that passes through to the output (add a
//     Select to drop it), and a row whose target is null is skipped whole,
//     though a null in an encoded column on that row still counts toward
//     handle_missing='error'
//   * with n rows and sum s in a category, prior = S/N over the fit rows:
//     target_encode blends prior*(1-w) + (s/n)*w with
//     w = 1/(1+exp(-(n - min_samples_leaf)/smoothing));
//     m_estimate_encode gives (s + prior*m)/(n + m);
//     woe_encode gives ln(((s+r)/(S+2r)) / (((n-s)+r)/((N-S)+2r)));
//     leave_one_out_encode gives s/n;
//     cat_boost_encode gives (s + prior*a)/(n + a);
//     james_stein_encode shrinks s/n toward the prior, as described below
//   * m_estimate_encode and james_stein_encode fall back to the prior for a
//     whole column whose categories are as many as the fit rows (an id-like
//     column); leave_one_out_encode and cat_boost_encode give the prior to a
//     category seen exactly once, so that it cannot leak its row's own
//     target, and woe_encode gives it 0
//   * woe_encode needs a binary target: exactly two distinct values, one of
//     them at most 0 and the other at least 1
//   * a null is a category of its own under handle_missing='value' (a null
//     unseen at fit takes the prior, or 0 for woe_encode), a null under
//     'return_nan', and an error under 'error', at fit and at transform alike;
//     under every option its rows count toward the category count
//   * an unseen category takes the prior under handle_unknown='value' (0 for
//     woe_encode), a null under 'return_nan', and an error under 'error'
//   * the output is f64, whatever the target's own type
//   * the target's sums are exact (128-bit for an integer target, Shewchuk
//     partials for a float one), so the fitted table is the same however the
//     row groups were split; a float target of 2^900 or more in magnitude
//     (2^450 for james_stein_encode, which sums the squares), and a NaN or
//     infinity, are errors
//   * once fitted the target is no longer read, so a serving file holding
//     only the encoded columns still transforms
//   * columns=null encodes every utf8 column that reaches the step
// james_stein_encode, with m the category's target mean, K the column's
// categories (the null one included), N the fit rows and var(y) the target's
// sample variance:
//   * independent (the default) weighs each category on its own variance:
//     B = i_var / (var(y) + i_var) * (K - 3) / (K - 1) with i_var the
//     category's sample variance (0 for a category of one), the weight is
//     w = clip(1 - B, 0, 1), and the value is w*m + (1 - w)*prior
//   * pooled assumes one variance for every category: variance = var(y)/(N/K),
//     SSE = sum (m - prior)^2, B = clip((K - 3)*variance/SSE, 0, 1), and the
//     value is prior + (1 - B)*(m - prior); an SSE of zero means every mean
//     is already the prior
//   * beta weighs a category by its size alone: w = n/(n + N)
//   * model='binary' is refused: upstream reads its estimate out of a module
//     global that its optimizer happens to leave behind, so the result is not
//     reproducible from the published formula
//   * a constant target leaves the independent weight 0/0, which upstream
//     answers with NaN; here the category keeps its mean, which is the prior
// leave_one_out_encode and cat_boost_encode take upstream's inference form,
// its transform without a target. Upstream's fit_transform instead leaves
// each training row's own target out of that row's value, or gives each row
// the running statistic of the rows before it; both are functions of the
// row, not of the fitted state, so they are not reproduced. Upstream's
// randomized, sigma and random_state add noise at training time only, so
// they are not params here.
// ---------------------------------------------------------------------------
class TargetBlend final : public Kernel {
 public:
  TargetBlend(Formula formula, const nlohmann::json& params) : formula_(formula) {
    const std::string what = formula_id(formula);
    if (params.contains("columns") && !params.at("columns").is_null()) {
      columns_ = column_list(params, "columns", what);
      explicit_columns_ = true;
    }
    if (!params.contains("target") || !params.at("target").is_string() ||
        params.at("target").get<std::string>().empty()) {
      throw Error(what + ": 'target' must be a column name");
    }
    target_ = params.at("target").get<std::string>();
    switch (formula_) {
      case Formula::Target:
        if (params.contains("min_samples_leaf") && !params.at("min_samples_leaf").is_null()) {
          if (!is_int64(params.at("min_samples_leaf"))) {
            throw Error(what + ": 'min_samples_leaf' must be an integer from -2^63 to 2^63 - 1");
          }
          min_samples_leaf_ = params.at("min_samples_leaf").get<int64_t>();
        }
        smoothing_ = positive_number(params, "smoothing", smoothing_, what);
        break;
      case Formula::MEstimate:
        m_ = non_negative_number(params, "m", m_, what);
        break;
      case Formula::Woe:
        regularization_ = positive_number(params, "regularization", regularization_, what);
        break;
      case Formula::JamesStein:
        model_ = model_from(params);
        break;
      case Formula::LeaveOneOut:
        break;
      case Formula::CatBoost:
        a_ = non_negative_number(params, "a", a_, what);
        break;
    }
    unknown_ = unknown_from(params, what, false);
    missing_ = missing_from(params, what);
    // Fit reads the target as well as the encoded columns; finalize and
    // load_state narrow the inputs to the encoded columns alone.
    inputs = columns_;
    inputs.push_back(target_);
    outputs = columns_;
  }

  std::string id() const override { return formula_id(formula_); }

  nlohmann::json params() const override {
    nlohmann::json out;
    out["columns"] = explicit_columns_ ? nlohmann::json(columns_) : nlohmann::json();
    out["target"] = target_;
    switch (formula_) {
      case Formula::Target:
        out["min_samples_leaf"] = min_samples_leaf_;
        out["smoothing"] = smoothing_;
        break;
      case Formula::MEstimate:
        out["m"] = m_;
        break;
      case Formula::Woe:
        out["regularization"] = regularization_;
        break;
      case Formula::JamesStein:
        out["model"] = model_ == Model::Independent ? "independent" : model_ == Model::Pooled ? "pooled" : "beta";
        break;
      case Formula::LeaveOneOut:
        break;
      case Formula::CatBoost:
        out["a"] = a_;
        break;
    }
    out["handle_unknown"] = unknown_name(unknown_);
    out["handle_missing"] = missing_name(missing_);
    return out;
  }

  void resolve(Schema& schema) override {
    // A fitted encoder never reads the target again, so a serving file may
    // omit it; its columns come from its state and are never re-derived.
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
    inputs = columns_;
    if (!fitted) inputs.push_back(target_);
    outputs = columns_;
  }

  bool needs_fit() const override { return true; }

  // One encoded column at fit.
  struct Column {
    AccMap categories;      // category -> acc
    Acc nulls;              // the null category's acc
    int64_t null_rows = 0;  // nulls seen, for handle_missing
  };

  struct State final : AccState {
    std::unordered_map<std::string, Column> columns;
    Acc total;                     // the fit-wide count and sum
    std::vector<double> smallest;  // the smallest distinct target values
    double largest = -std::numeric_limits<double>::infinity();
  };

  std::unique_ptr<AccState> make_state() const override { return std::make_unique<State>(); }

  void consume(AccState& raw, const Batch& batch) const override {
    auto& state = static_cast<State&>(raw);
    const int index = batch.find(target_);
    if (index < 0) throw Error(stage + ": target column '" + target_ + "' missing from batch");
    const arrow::Array& target = *batch.cols[static_cast<std::size_t>(index)];
    const auto rows = static_cast<std::size_t>(target.length());
    // The target is read once per batch, so each encoded column only walks
    // its own values afterwards.
    std::vector<int64_t> integers(target_family_ == Family::Integer ? rows : 0, 0);
    std::vector<double> doubles(target_family_ == Family::Floating ? rows : 0, 0.0);
    std::vector<bool> present(rows, false);
    read_target(state, target, integers, doubles, present);

    for (const std::string& name : columns_) {
      const int idx = batch.find(name);
      if (idx < 0) throw Error(stage + ": column '" + name + "' missing from batch");
      Column& column = state.columns[name];
      for_each_encoded_utf8_if(
          *batch.cols[static_cast<std::size_t>(idx)], stage, name,
          [&](int64_t row, bool is_null) {
            // Every null counts toward handle_missing='error', even one with
            // a null target.
            if (is_null) ++column.null_rows;
            return present[static_cast<std::size_t>(row)];
          },
          [&](bool is_null, std::string_view value) -> std::optional<std::size_t> {
            return is_null ? std::nullopt : std::optional<std::size_t>{column.categories.slot(value)};
          },
          [&](int64_t row, const std::optional<std::size_t>& category) {
            const auto slot = static_cast<std::size_t>(row);
            Acc& acc = category.has_value() ? column.categories.values[*category] : column.nulls;
            if (target_family_ == Family::Integer) {
              add(acc, integers[slot]);
            } else {
              add(acc, doubles[slot]);
            }
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
    Totals totals;
    totals.rows = merged->total.count;
    if (totals.rows == 0) {
      throw Error(stage + ": target column '" + target_ + "' has no non-null values to learn from");
    }
    totals.sum = sum_of(merged->total, target_family_);
    prior_ = totals.sum / static_cast<double>(totals.rows);
    if (formula_ == Formula::Woe) {
      check_binary(*merged);
      zero_one_target_ = merged->smallest.front() == 0.0 && merged->largest == 1.0;
    }
    // Only james_stein_encode reads the variance, and only its bound keeps
    // the square of the sum inside f64.
    if (formula_ == Formula::JamesStein) totals.variance = variance_of(merged->total);

    tables_.clear();
    for (const std::string& name : columns_) {
      const Column& column = merged->columns[name];
      if (missing_ == Missing::Error && column.null_rows > 0) {
        throw Error(stage + ": column '" + name + "' has nulls and handle_missing is 'error'");
      }
      // Sorted, so the fitted table is the same however the rows were split.
      // Each key is a view of the accumulator's own, beside its slot there.
      std::vector<std::pair<std::string_view, std::size_t>> keys;
      keys.reserve(column.categories.slots.size());
      for (const auto& [key, slot] : column.categories.slots) {
        if (!arrow::util::ValidateUTF8(key)) {
          throw Error(stage + ": column '" + name + "' has a category that is not valid UTF-8; " + id() +
                      " keys must be UTF-8 to serialize");
        }
        keys.emplace_back(key, slot);
      }
      std::sort(keys.begin(), keys.end());
      totals.levels = keys.size() + (column.nulls.count > 0 ? 1U : 0U);
      totals.pooled = formula_ == Formula::JamesStein && model_ == Model::Pooled ? pooled_weight(column, totals) : 0.0;
      // Each category is encoded in slot order, which reads the accumulators
      // in memory order rather than at random, and then placed by its key:
      // a leave_one_out_encode fit of 1,000,000 distinct keys takes 300 ms
      // this way and 319 ms in key order (one thread, Apple M4).
      std::vector<double> values;
      values.reserve(column.categories.values.size());
      for (const Acc& acc : column.categories.values) values.push_back(encoded(name, acc, totals));
      Table& table = tables_[name];
      for (const auto& [key, slot] : keys) table.values.emplace(key, values[slot]);
      table.short_values.index(table.values);
      table.has_null_category = missing_ == Missing::Value && column.nulls.count > 0;
      if (table.has_null_category) table.null_value = encoded(name, column.nulls, totals);
    }
    inputs = columns_;  // the target is read at fit only
    outputs = columns_;
    fitted = true;
  }

  nlohmann::json state() const override {
    nlohmann::json values = nlohmann::json::object();
    nlohmann::json null_values = nlohmann::json::object();
    for (const std::string& name : columns_) {
      const Table& table = tables_.at(name);
      nlohmann::json entries = nlohmann::json::object();
      for (const auto& [key, value] : table.values) entries[key] = serializable(value, stage, name);
      values[name] = std::move(entries);
      null_values[name] =
          table.has_null_category ? nlohmann::json(serializable(table.null_value, stage, name)) : nlohmann::json();
    }
    return {{"columns", columns_}, {"prior", prior_}, {"values", values}, {"null_values", null_values}};
  }

  void load_state(const nlohmann::json& state) override {
    if (!state.is_object() || !state.contains("values") || !state.at("values").is_object() ||
        !state.contains("prior") || !state.at("prior").is_number()) {
      throw Error(stage + (shrinks() ? ": state must hold a numeric 'prior' and a 'values' object"
                                     : ": state must hold a 'values' object and a numeric 'prior'"));
    }
    prior_ = state.at("prior").get<double>();
    if (!std::isfinite(prior_)) throw Error(stage + ": 'prior' must be a finite number");
    const nlohmann::json& values = state.at("values");
    load_fitted_columns(state, stage, columns_, explicit_columns_, &values);
    // The null category is the entry a hand-edited manifest is likeliest to
    // mangle, so a wrong shape there is an error rather than a default.
    const nlohmann::json* null_values = optional_per_column_object(state, "null_values", columns_, stage);
    tables_.clear();
    for (const std::string& name : columns_) {
      if (!values.contains(name) || !values.at(name).is_object()) {
        throw Error(stage + ": values['" + name + "'] must be an object");
      }
      Table& table = tables_[name];
      for (const auto& [key, value] : values.at(name).items()) {
        if (!value.is_number() || !std::isfinite(value.get<double>())) {
          throw Error(stage + ": values['" + name + "']['" + key + "'] must be a finite number");
        }
        table.values.emplace(key, value.get<double>());
      }
      table.short_values.index(table.values);
      if (null_values != nullptr && !null_values->at(name).is_null()) {
        const nlohmann::json& value = null_values->at(name);
        if (!value.is_number() || !std::isfinite(value.get<double>())) {
          throw Error(stage + ": null_values['" + name + "'] must be a finite number or null");
        }
        table.has_null_category = true;
        table.null_value = value.get<double>();
      }
    }
    inputs = columns_;
    outputs = columns_;
    fitted = true;
  }

  Batch transform(Batch batch) const override {
    for (const std::string& name : columns_) {
      const int idx = batch.find(name);
      if (idx < 0) throw Error(stage + ": column '" + name + "' missing from batch");
      const arrow::Array& array = *batch.cols[static_cast<std::size_t>(idx)];
      PrimitiveOutput<arrow::DoubleType> values(array.length(), stage + ": column '" + name + "'");
      const Table& table = tables_.at(name);
      for_each_encoded_utf8(
          array, stage, name,
          [&](bool is_null, std::string_view value) { return look_up(table, name, is_null, value); },
          [&](int64_t row, const std::optional<double>& encoded) {
            if (encoded.has_value()) {
              values.mutable_values()[row] = *encoded;
            } else {
              values.set_null(row);
            }
          });
      batch.cols[static_cast<std::size_t>(idx)] = values.finish();
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

 private:
  struct Table {
    StringValues values;
    ShortValues<double> short_values;  // `values` again, when its keys allow a perfect hash
    double null_value = 0.0;
    bool has_null_category = false;
  };

  // What every category of a column is encoded against: the target's
  // fit-wide sum, row count and variance (james_stein_encode only), and the
  // column's category count, the null category included, with
  // james_stein_encode's pooled weight.
  struct Totals {
    double sum = 0.0;
    int64_t rows = 0;
    double variance = 0.0;
    std::size_t levels = 0;
    double pooled = 0.0;
  };

  // Every target value of one batch, plus which rows carry one at all. The
  // fit-wide accumulator is updated here, once per row rather than once per
  // encoded column.
  void read_target(State& state, const arrow::Array& target, std::vector<int64_t>& integers,
                   std::vector<double>& doubles, std::vector<bool>& present) const {
    if (target_family_ == Family::Integer) {
      for_each_integer(target, stage, target_, [&](int64_t row, bool is_null, int64_t value) {
        if (is_null) return;
        const auto slot = static_cast<std::size_t>(row);
        integers[slot] = value;
        present[slot] = true;
        add(state.total, value);
        if (formula_ == Formula::Woe) note_distinct(state.smallest, state.largest, static_cast<double>(value));
      });
      return;
    }
    const std::string where = stage + ": target column '" + target_ + "'";
    const bool squares = formula_ == Formula::JamesStein;
    for_each_floating(target, stage, target_, [&](int64_t row, bool is_null, double value) {
      if (is_null) return;
      if (!std::isfinite(value)) {
        throw Error(where + ": value " + std::format("{}", value) +
                    " is not finite; a target cannot be NaN or infinity");
      }
      // Bounding each value keeps every running sum inside f64 whatever the
      // split, so a fit can never fail on one thread count and pass on another.
      if (std::fabs(value) >= (squares ? kMaxSquared : kMaxSummand)) {
        throw Error(where + ": value " + std::format("{}", value) +
                    (squares ? " is 2^450 or more in magnitude, more than the squared sums can hold"
                             : " is 2^900 or more in magnitude, more than a sum can hold") +
                    " (scale the target first)");
      }
      if (value == 0.0) value = 0.0;  // one zero: -0.0 would make the distinct-value probe depend on the split
      const auto slot = static_cast<std::size_t>(row);
      doubles[slot] = value;
      present[slot] = true;
      add(state.total, value);
      if (formula_ == Formula::Woe) note_distinct(state.smallest, state.largest, value);
    });
  }

  // One target value into an accumulator, with its square for
  // james_stein_encode.
  void add(Acc& acc, int64_t value) const {
    ++acc.count;
    acc.isum += value;
    if (formula_ == Formula::JamesStein) add_square(acc.squares, value);
  }

  void add(Acc& acc, double value) const {
    ++acc.count;
    if (!acc.fsum.add(value) || (formula_ == Formula::JamesStein && !add_product(acc.squares, value, value))) {
      throw Error(overflow());
    }
  }

  void merge_into(State& into, State& from) const {
    merge_acc(into.total, from.total);
    for (double value : from.smallest) note_distinct(into.smallest, into.largest, value);
    into.largest = std::max(into.largest, from.largest);
    for (auto& [name, column] : from.columns) {
      Column& dest = into.columns[name];
      for (auto& [key, slot] : column.categories.slots) {
        Acc& acc = column.categories.values[slot];
        auto found = dest.categories.slots.find(key);
        if (found == dest.categories.slots.end()) {
          dest.categories.slots.emplace(std::move(key), dest.categories.values.size());
          dest.categories.values.push_back(std::move(acc));
          continue;
        }
        merge_acc(dest.categories.values[found->second], acc);
      }
      merge_acc(dest.nulls, column.nulls);
      dest.null_rows += column.null_rows;
    }
  }

  void merge_acc(Acc& into, const Acc& from) const {
    into.count += from.count;
    into.isum += from.isum;
    if (!into.fsum.merge(from.fsum) || !into.squares.merge(from.squares)) throw Error(overflow());
  }

  // woe_encode's target must be binary, which upstream reads as: exactly two
  // distinct values, at least one of them 1 or more and at least one 0 or less.
  void check_binary(const State& state) const {
    const std::string where = stage + ": target column '" + target_ + "'";
    if (state.smallest.size() != 2) {
      throw Error(where + " must be binary, but it holds " +
                  (state.smallest.size() == 1 ? "only one distinct value" : "more than two distinct values"));
    }
    if (state.largest < 1.0) throw Error(where + " must be binary with values 0 and 1; no value 1 was found");
    if (state.smallest.front() > 0.0) throw Error(where + " must be binary with values 0 and 1; no value 0 was found");
  }

  // The one place the six kernels differ: n rows summing to s in this
  // category, against what every category of its column shares. A value that
  // is not finite is refused here, naming the param to check, because state()
  // refuses to write it and load_state to read it.
  double encoded(const std::string& name, const Acc& acc, const Totals& totals) const {
    const auto n = static_cast<double>(acc.count);
    const double s = sum_of(acc, target_family_);
    // An id-like column, one category per fit row, teaches nothing.
    const bool id_like = totals.levels == static_cast<std::size_t>(totals.rows);
    switch (formula_) {
      case Formula::Target: {
        const double weight = expit((n - static_cast<double>(min_samples_leaf_)) / smoothing_);
        return finite_encoding(prior_ * (1 - weight) + (s / n) * weight, stage, name, "smoothing");
      }
      case Formula::MEstimate:
        // A large enough m makes prior*m infinite and the whole quotient with
        // it; so does a large enough 'a' below.
        return id_like ? prior_ : finite_encoding((s + prior_ * m_) / (n + m_), stage, name, "m");
      case Formula::JamesStein:
        return finite_encoding(id_like ? prior_ : james_stein(acc, s / n, totals), stage, name, "target");
      case Formula::LeaveOneOut:
        return finite_encoding(acc.count > 1 ? s / n : prior_, stage, name, "target");
      case Formula::CatBoost:
        return finite_encoding(acc.count > 1 ? (s + prior_ * a_) / (n + a_) : prior_, stage, name, "a");
      case Formula::Woe:
        break;
    }
    if (acc.count == 1) return 0.0;  // a category seen once teaches nothing
    const double good = (s + regularization_) / (totals.sum + 2 * regularization_);
    const double bad =
        ((n - s) + regularization_) / ((static_cast<double>(totals.rows) - totals.sum) + 2 * regularization_);
    const double woe = std::log(good / bad);
    if (!std::isfinite(woe)) {
      // A target of 0s and 1s keeps both shares in (0, 1], unless a
      // regularization too small or too large for the row count pushes one
      // to 0. Upstream's binary check also passes other pairs, such as -1
      // and 2, which can make a share negative.
      throw Error(stage + ": column '" + name + "' has a category whose weight of evidence is not finite; " +
                  (zero_one_target_ ? "check 'regularization'" : "woe_encode needs a target of 0s and 1s"));
    }
    return woe;
  }

  // james_stein_encode's value for a category whose target mean is `mean`.
  double james_stein(const Acc& acc, double mean, const Totals& totals) const {
    const auto levels = static_cast<double>(totals.levels);
    switch (model_) {
      case Model::Independent: {
        double weight = 1.0;  // a single category is its own prior, whatever the weight
        if (levels >= 2) {
          const double variance = variance_of(acc);
          const double total = totals.variance + variance;
          const double shrinkage = (total == 0.0 ? 0.0 : variance / total) * (levels - 3) / (levels - 1);
          weight = std::clamp(1.0 - shrinkage, 0.0, 1.0);
        }
        return weight * mean + (1 - weight) * prior_;
      }
      case Model::Pooled:
        return prior_ + (1 - totals.pooled) * (mean - prior_);
      case Model::Beta:
        break;
    }
    const auto n = static_cast<double>(acc.count);
    const double weight = n / (n + static_cast<double>(totals.rows));
    return weight * mean + (1 - weight) * prior_;
  }

  // The one weight every category of a column shares under
  // james_stein_encode's pooled model.
  double pooled_weight(const Column& column, const Totals& totals) const {
    ExactSum errors;
    const auto add_error = [&](const Acc& acc) {
      const double error = sum_of(acc, target_family_) / static_cast<double>(acc.count) - prior_;
      if (!errors.add(error * error)) throw Error(stage + ": the squared category means overflow f64");
    };
    for (const Acc& acc : column.categories.values) add_error(acc);
    if (column.nulls.count > 0) add_error(column.nulls);
    const double sse = errors.value();
    if (sse <= 0) return 0.0;  // every mean is the prior already, so the weight cannot matter
    const auto levels = static_cast<double>(totals.levels);
    const double variance = totals.variance / (static_cast<double>(totals.rows) / levels);
    return std::clamp((levels - 3) * variance / sse, 0.0, 1.0);
  }

  // The sample variance (ddof=1) pandas' var() computes; a category of one has
  // none, which upstream reads as zero. n * sum(x^2) - sum(x)^2 is n times the
  // sum of squared deviations, and it is formed exactly from the exact sums
  // before its one rounding. Rounding the sums first leaves two terms that
  // cancel: for a target near 1e9 with a spread of 1, n * sum(x^2) and sum(x)^2
  // agree in every digit a double keeps, and their difference is noise.
  //
  // REGRESSION: the products are formed from the sums' terms, which depend on
  // their values alone. They used to be formed from the partials, which two
  // workers can leave split differently than one. Near 1e-155 the products
  // underflow and round, so a category's value depended on the thread count
  // in its last bits.
  double variance_of(const Acc& acc) const {
    if (acc.count < 2) return 0.0;
    const auto n = static_cast<double>(acc.count);
    ExactSum scaled;
    bool held = true;
    const std::vector<double> squares = acc.squares.terms();
    const std::vector<double> sums = sum_terms(acc, target_family_);
    for (double square : squares) held = held && add_product(scaled, n, square);
    for (double left : sums) {
      for (double right : sums) held = held && add_product(scaled, -left, right);
    }
    if (!held) throw Error(overflow());
    return std::max(0.0, scaled.value() / (n * (n - 1)));
  }

  // The encoded value for one input value, or nullopt when the output is null.
  std::optional<double> look_up(const std::string& name, bool is_null, std::string_view value) const {
    return look_up(tables_.at(name), name, is_null, value);
  }

  std::optional<double> look_up(const Table& table, const std::string& name, bool is_null,
                                std::string_view value) const {
    if (is_null) {
      if (missing_ == Missing::ReturnNan) return std::nullopt;
      refuse_null(missing_, stage, name);
      // A null unseen at fit is worth the same as an unseen category, but it
      // is not one: handle_unknown='error' does not fire on it.
      return table.has_null_category ? table.null_value : default_value();
    }
    if (const double* found = table.short_values.find(table.values, value)) return *found;
    switch (unknown_) {
      case Unknown::Value:
        return default_value();
      case Unknown::ReturnNan:
        return std::nullopt;
      case Unknown::Indicator:
      case Unknown::Error:
        break;
    }
    throw Error(stage + ": column '" + name + "' has a category that was not seen at fit");
  }

  // What a category the fit never saw is worth: the prior, except for
  // woe_encode, whose scale is centered on 0.
  double default_value() const { return formula_ == Formula::Woe ? 0.0 : prior_; }

  // The three shrinkage encoders word two messages their own way.
  bool shrinks() const {
    return formula_ == Formula::JamesStein || formula_ == Formula::LeaveOneOut || formula_ == Formula::CatBoost;
  }

  std::string overflow() const {
    return stage + ": target column '" + target_ +
           (shrinks() ? "': the sums overflow f64" : "': the sum overflows f64");
  }

  Formula formula_;
  std::vector<std::string> columns_;
  bool explicit_columns_ = false;
  std::string target_;
  int64_t min_samples_leaf_ = 20;     // target_encode
  double smoothing_ = 10;             // target_encode
  double m_ = 1.0;                    // m_estimate_encode
  double regularization_ = 1.0;       // woe_encode
  bool zero_one_target_ = true;       // woe_encode, at fit: the target's two values are 0 and 1
  Model model_ = Model::Independent;  // james_stein_encode
  double a_ = 1.0;                    // cat_boost_encode
  Unknown unknown_ = Unknown::Value;
  Missing missing_ = Missing::Value;
  Family target_family_ = Family::Floating;  // fixed by resolve, read at fit only
  double prior_ = 0.0;
  std::unordered_map<std::string, Table> tables_;
};

}  // namespace

std::unique_ptr<Kernel> make_target_encode(const nlohmann::json& params) {
  return std::make_unique<TargetBlend>(Formula::Target, params);
}

std::unique_ptr<Kernel> make_m_estimate_encode(const nlohmann::json& params) {
  return std::make_unique<TargetBlend>(Formula::MEstimate, params);
}

std::unique_ptr<Kernel> make_woe_encode(const nlohmann::json& params) {
  return std::make_unique<TargetBlend>(Formula::Woe, params);
}

std::unique_ptr<Kernel> make_james_stein_encode(const nlohmann::json& params) {
  return std::make_unique<TargetBlend>(Formula::JamesStein, params);
}

std::unique_ptr<Kernel> make_leave_one_out_encode(const nlohmann::json& params) {
  return std::make_unique<TargetBlend>(Formula::LeaveOneOut, params);
}

std::unique_ptr<Kernel> make_cat_boost_encode(const nlohmann::json& params) {
  return std::make_unique<TargetBlend>(Formula::CatBoost, params);
}

}  // namespace fastfeaturetransform
