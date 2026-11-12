// The sklearn.preprocessing scalers: standard_scale, min_max_scale,
// max_abs_scale, binarize and normalize. Where the category_encoders family
// keys a table by category, these read numbers and write numbers, so they
// share a different substrate: numeric.hpp's visitor over an
// integer-or-float column and its binder for a numeric column list, plus
// sklearn's near-constant rule below, which replaces a divisor that is
// indistinguishable from zero with 1 so a constant column or an all-zero row
// comes out finite rather than NaN.
//
// The three fitted scalers are monoids: min, max and the exact sums below
// merge in any order, so the fitted table is the same however the rows were
// split across workers. binarize and normalize learn nothing at all, which
// is why both name their columns in the params: an artifact that did not
// would transform a different set of columns for every file it was given.

#include <algorithm>
#include <array>
#include <cmath>
#include <format>
#include <limits>
#include <optional>
#include <unordered_map>

#include "fastfeaturetransform/kernels/cast_table.hpp"
#include "fastfeaturetransform/kernels/categories.hpp"
#include "fastfeaturetransform/kernels/kernels.hpp"
#include "fastfeaturetransform/kernels/numeric.hpp"
#include "fastfeaturetransform/kernels/output.hpp"
#include "fastfeaturetransform/kernels/scale_transform.hpp"
#include "fastfeaturetransform/kernels/util.hpp"

namespace fastfeaturetransform {
namespace {

// --- the substrate the five scalers share ----------------------------------

constexpr double kEpsilon = std::numeric_limits<double>::epsilon();

// sklearn's _handle_zeros_in_scale: a divisor below ten machine epsilons is
// replaced by 1. It is not only the zero case — a range or a norm that small
// is indistinguishable from zero, and dividing by it would turn a rounding
// error into an enormous number. Reproduced exactly, threshold included.
constexpr double kNearConstant = 10.0 * kEpsilon;

double handle_zeros(double divisor) { return divisor < kNearConstant ? 1.0 : divisor; }

// The quotient of an exact sum by a positive count, correctly rounded unless
// the exact quotient lies within about 2^-54 ulp of halfway between two
// doubles, where it can be one ulp off. Rounding the sum first and dividing
// that would round twice, which costs the last bit of a mean or a variance
// often enough to show up against numpy; one step of refinement over the
// exact remainder puts it back.
double divided(const ExactSum& sum, double count, const std::string& what) {
  const double quotient = sum.value() / count;
  ExactSum remainder = sum;
  if (!add_product(remainder, -quotient, count)) throw Error(what + " leaves the range of f64");
  return quotient + remainder.value() / count;
}

// A number a scaler may read. sklearn spells a missing number NaN, which its
// fitted scalers ignore in the statistics and pass through at transform; we
// spell it as a null, which does the same, and refuse the other spelling so a
// column never carries two of them. An infinity sklearn refuses outright.
//
// The throw is a function of its own, as in numeric.hpp, so the check stays
// small enough to inline: normalize runs it once per value, and an
// out-of-line check cost more than half of its transform. Only a step that
// passes a null through adds 'write a missing value as a null' to the
// message; binarize and normalize refuse a null too.
[[noreturn]] void refuse_non_finite(double value, const std::string& stage, const std::string& name, bool null_passes) {
  throw Error(stage + ": column '" + name + "': value " + std::format("{}", value) +
              (null_passes ? " is not finite; write a missing value as a null" : " is not finite"));
}

double finite_number(double value, const std::string& stage, const std::string& name, bool null_passes = true) {
  if (!std::isfinite(value)) refuse_non_finite(value, stage, name, null_passes);
  return value;
}

// Rewrite the schema type of every column a scaler owns. Our output type is
// declared here rather than taken from the input, which is the one place we
// part company with sklearn's dtype-preserving transforms.
void retype(Schema& schema, const std::vector<std::string>& columns, const char* tag) {
  for (auto& [name, type] : schema.cols) {
    if (std::find(columns.begin(), columns.end(), name) != columns.end()) type = tag;
  }
}

// One record value a scaler may read, or nullopt for a null.
std::optional<double> row_number(const RowValue& value, const std::string& stage, const std::string& name,
                                 bool null_passes = true) {
  if (std::holds_alternative<std::monostate>(value)) return std::nullopt;
  if (const int64_t* integer = std::get_if<int64_t>(&value)) return static_cast<double>(*integer);
  if (const double* number = std::get_if<double>(&value)) return finite_number(*number, stage, name, null_passes);
  throw Error(stage + ": key '" + name + "' must be a number or null");
}

// One per-column object of a fitted table, keyed by column name (nlohmann
// writes the keys sorted, whatever order they were added in).
nlohmann::json numbers_state(const std::vector<std::string>& columns,
                             const std::unordered_map<std::string, double>& values, const std::string& stage) {
  nlohmann::json out = nlohmann::json::object();
  for (const std::string& name : columns) out[name] = serializable(values.at(name), stage, name);
  return out;
}

// Read one per-column object of finite numbers out of a manifest. Absent, the
// wrong shape, or not finite is an error naming the key and the column: a
// scaler that quietly defaulted a scale would serve every row of that column
// wrong and say nothing.
std::unordered_map<std::string, double> load_numbers(const nlohmann::json& state, const std::string& key,
                                                     const std::vector<std::string>& columns,
                                                     const std::string& stage) {
  if (!state.is_object() || !state.contains(key) || !state.at(key).is_object()) {
    throw Error(stage + ": state must hold a '" + key + "' object");
  }
  const nlohmann::json& object = state.at(key);
  std::unordered_map<std::string, double> values;
  for (const std::string& name : columns) {
    if (!object.contains(name) || !object.at(name).is_number() || !std::isfinite(object.at(name).get<double>())) {
      throw Error(stage + ": " + key + "['" + name + "'] must be a finite number");
    }
    values.emplace(name, object.at(name).get<double>());
  }
  return values;
}

// A count out of a manifest: the rows a scaler saw, or a column's non-null
// values. Never negative, and a fitted column always saw at least one value.
int64_t load_count(const nlohmann::json& object, const std::string& key, const std::string& what, int64_t least) {
  if (!object.is_object() || !object.contains(key) || !object.at(key).is_number_integer() ||
      object.at(key).get<int64_t>() < least) {
    throw Error(what + " must be an integer >= " + std::to_string(least));
  }
  return object.at(key).get<int64_t>();
}

// The extrema the two range scalers fit. min and max are exact whatever the
// split, so both artifacts are bit-identical under any thread count; -0.0
// reads as 0.0, because min() and max() cannot tell the two apart and which
// zero they keep would depend on row order.
struct Extrema {
  double low = std::numeric_limits<double>::infinity();
  double high = -std::numeric_limits<double>::infinity();
  int64_t count = 0;
};

struct ExtremaState final : AccState {
  std::unordered_map<std::string, Extrema> columns;
  int64_t rows = 0;
};

template <typename Value>
void dense_extrema(const Value* values, int64_t length, Extrema& extrema, const std::string& stage,
                   const std::string& name) {
  bool finite = true;
  for (int64_t row = 0; row < length; ++row) finite &= std::isfinite(values[row]);
  if (!finite) {
    for (int64_t row = 0; row < length; ++row) finite_number(static_cast<double>(values[row]), stage, name);
  }
  // Independent lanes permit SIMD without a floating sum or reassociation.
  // Finite minima/maxima are exact; canonicalizing zero removes tie order.
  std::array<double, 4> lows;
  std::array<double, 4> highs;
  lows.fill(extrema.low);
  highs.fill(extrema.high);
  int64_t row = 0;
  for (; row + 4 <= length; row += 4) {
    for (std::size_t lane = 0; lane < 4; ++lane) {
      double value = static_cast<double>(values[row + static_cast<int64_t>(lane)]);
      if (value == 0.0) value = 0.0;
      lows[lane] = std::min(lows[lane], value);
      highs[lane] = std::max(highs[lane], value);
    }
  }
  for (std::size_t lane = 0; lane < 4; ++lane) {
    extrema.low = std::min(extrema.low, lows[lane]);
    extrema.high = std::max(extrema.high, highs[lane]);
  }
  for (; row < length; ++row) {
    double value = static_cast<double>(values[row]);
    if (value == 0.0) value = 0.0;
    extrema.low = std::min(extrema.low, value);
    extrema.high = std::max(extrema.high, value);
  }
  extrema.count += length;
}

void consume_extrema(const std::vector<std::string>& columns, const std::string& stage, ExtremaState& state,
                     const Batch& batch) {
  state.rows += batch.rows;
  for (const std::string& name : columns) {
    Extrema& extrema = state.columns[name];
    const arrow::Array& array = column_array(batch, stage, name);
    if (array.null_count() == 0 && array.type_id() == arrow::Type::DOUBLE) {
      dense_extrema(static_cast<const arrow::DoubleArray&>(array).raw_values(), array.length(), extrema, stage, name);
      continue;
    }
    if (array.null_count() == 0 && array.type_id() == arrow::Type::FLOAT) {
      dense_extrema(static_cast<const arrow::FloatArray&>(array).raw_values(), array.length(), extrema, stage, name);
      continue;
    }
    for_each_numeric(array, stage, name, [&](int64_t, bool is_null, double value) {
      if (is_null) return;
      double number = finite_number(value, stage, name);
      if (number == 0.0) number = 0.0;  // one zero: min() and max() cannot separate -0.0 from 0.0
      extrema.low = std::min(extrema.low, number);
      extrema.high = std::max(extrema.high, number);
      ++extrema.count;
    });
  }
}

std::unordered_map<std::string, Extrema> merge_extrema(const std::vector<std::string>& columns,
                                                       const std::string& stage,
                                                       std::vector<std::unique_ptr<AccState>>& states, int64_t& rows) {
  std::unordered_map<std::string, Extrema> merged;
  rows = 0;
  for (auto& raw : states) {
    auto& state = static_cast<ExtremaState&>(*raw);
    rows += state.rows;
    for (const auto& [name, extrema] : state.columns) {
      Extrema& into = merged[name];
      into.low = std::min(into.low, extrema.low);
      into.high = std::max(into.high, extrema.high);
      into.count += extrema.count;
    }
    raw.reset();
  }
  for (const std::string& name : columns) {
    if (merged[name].count == 0) {
      throw Error(stage + ": column '" + name + "' has no non-null values to learn from");
    }
  }
  return merged;
}

// ---------------------------------------------------------------------------
// standard_scale — integer or float columns -> (x - mean) / sqrt(var), f64 in
// place.
// params: {"columns": [...] | null, "with_mean": bool, "with_std": bool}
// Semantics:
//   * mean and variance are the population statistics (ddof=0) of a column's
//     non-null values, and the transform is (x - mean) / scale, subtracting
//     and then dividing in that order, as sklearn does
//   * a column whose variance is indistinguishable from zero scales by 1
//     instead of by 0, so a centered constant column comes out 0, not NaN;
//     "indistinguishable" is sklearn's own bound, var <= n*eps*var +
//     (n*mean*eps)^2, taken on the variance before the square root
//   * a null teaches nothing — the count is the column's non-null values, as
//     sklearn's per-feature n_samples_seen_ is — and comes out a null
//   * with_mean=false leaves the centering out and with_std=false the
//     division; both statistics are fitted and recorded either way, because
//     they cost one pass whichever flags are set
//   * the sums are exact partials and the variance is the exact
//     Sum (x - mean)^2 / n, rounded as divided() describes, so the fitted
//     table is the same however the rows were split; sklearn accumulates its
//     variance incrementally instead, so the two are the same number to
//     within a rounding rather than by construction: every parity case lands
//     on the same bits, but a pathological column need not
//   * columns=null scales every integer or float column that reaches the
//     step, and the output is f64 whatever the input width
// ---------------------------------------------------------------------------
class StandardScale final : public Kernel {
 public:
  explicit StandardScale(const nlohmann::json& params) {
    if (params.contains("columns") && !params.at("columns").is_null()) {
      columns_ = column_list(params, "columns", "standard_scale");
      explicit_columns_ = true;
      inputs = columns_;
      outputs = columns_;
    }
    with_mean_ = bool_param(params, "with_mean", true, "standard_scale");
    with_std_ = bool_param(params, "with_std", true, "standard_scale");
  }

  std::string id() const override { return "standard_scale"; }

  nlohmann::json params() const override {
    nlohmann::json out;
    out["columns"] = explicit_columns_ ? nlohmann::json(columns_) : nlohmann::json();
    out["with_mean"] = with_mean_;
    out["with_std"] = with_std_;
    return out;
  }

  void resolve(Schema& schema) override {
    bind_numeric(schema, stage, columns_, explicit_columns_, fitted, "scale");
    retype(schema, columns_, "f64");
    inputs = columns_;
    outputs = columns_;
  }

  bool needs_fit() const override { return true; }

  // One column's running statistics: the exact sum of its values, the exact
  // sum of their exact squares, how many there were, and the smallest and
  // largest, which say whether the column is constant.
  struct Accumulator {
    ExactSum sum;
    ExactSum squares;
    int64_t count = 0;
    double low = std::numeric_limits<double>::infinity();
    double high = -std::numeric_limits<double>::infinity();
  };

  struct State final : AccState {
    std::unordered_map<std::string, Accumulator> columns;
  };

  std::unique_ptr<AccState> make_state() const override { return std::make_unique<State>(); }

  void consume(AccState& raw, const Batch& batch) const override {
    auto& state = static_cast<State&>(raw);
    for (const std::string& name : columns_) {
      Accumulator& accumulator = state.columns[name];
      const arrow::Array& array = column_array(batch, stage, name);
      // Precompute independent rounded squares and their deliberate FMA
      // residuals; the exact accumulation below still adds them in row order.
      const auto dense = [&]<typename Value>(const Value* input) {
        constexpr int64_t width = 64;
        std::array<double, width> numbers;
        std::array<double, width> products;
        std::array<double, width> residuals;
        ExactSum::Adder sum(accumulator.sum);
        ExactSum::Adder squares(accumulator.squares);
        const auto refuse = [&] { throw Error(stage + ": column '" + name + "' sums past the range of f64"); };
        for (int64_t first = 0; first < array.length(); first += width) {
          const int64_t count = std::min(width, array.length() - first);
          bool inexact = false;
          for (int64_t i = 0; i < count; ++i) {
            const auto slot = static_cast<std::size_t>(i);
            numbers[slot] = static_cast<double>(input[first + i]);
            products[slot] = numbers[slot] * numbers[slot];
            residuals[slot] = std::fma(numbers[slot], numbers[slot], -products[slot]);
            inexact |= residuals[slot] != 0.0;
          }
          // Validation and the accumulator updates run in row order and stop
          // at the first failure, as in the general path below. A block whose
          // squares are all exact, as those of float32 values and of integers
          // up to 2^26 are, has no residual to add and takes the short add:
          // its sums mostly stay within their heads. The choice is made once
          // per block because a test per value between the adds costs more
          // than it saves: the short add for every value takes a fit of
          // 1,000,000 normal float64 values from 6.5 to 6.9 ms (one thread,
          // Apple M4).
          if (inexact) {
            for (int64_t i = 0; i < count; ++i) {
              const auto slot = static_cast<std::size_t>(i);
              const double number = finite_number(numbers[slot], stage, name);
              if (!sum.add(number) || !squares.add(products[slot]) || !squares.add(residuals[slot])) refuse();
              ++accumulator.count;
              accumulator.low = std::min(accumulator.low, number);
              accumulator.high = std::max(accumulator.high, number);
            }
          } else {
            for (int64_t i = 0; i < count; ++i) {
              const auto slot = static_cast<std::size_t>(i);
              const double number = finite_number(numbers[slot], stage, name);
              if (!sum.add_short(number) || !squares.add_short(products[slot])) refuse();
              ++accumulator.count;
              accumulator.low = std::min(accumulator.low, number);
              accumulator.high = std::max(accumulator.high, number);
            }
          }
        }
      };
      if (array.null_count() == 0 && array.type_id() == arrow::Type::DOUBLE) {
        dense(static_cast<const arrow::DoubleArray&>(array).raw_values());
        continue;
      }
      if (array.null_count() == 0 && array.type_id() == arrow::Type::FLOAT) {
        dense(static_cast<const arrow::FloatArray&>(array).raw_values());
        continue;
      }
      for_each_numeric(array, stage, name, [&](int64_t, bool is_null, double value) {
        if (is_null) return;
        const double number = finite_number(value, stage, name);
        if (!accumulator.sum.add(number) || !add_product(accumulator.squares, number, number)) {
          throw Error(stage + ": column '" + name + "' sums past the range of f64");
        }
        ++accumulator.count;
        accumulator.low = std::min(accumulator.low, number);
        accumulator.high = std::max(accumulator.high, number);
      });
    }
  }

  void finalize(std::vector<std::unique_ptr<AccState>> states) override {
    std::unordered_map<std::string, Accumulator> merged;
    for (auto& raw : states) {
      auto& state = static_cast<State&>(*raw);
      for (const auto& [name, accumulator] : state.columns) {
        Accumulator& into = merged[name];
        if (!into.sum.merge(accumulator.sum) || !into.squares.merge(accumulator.squares)) {
          throw Error(stage + ": column '" + name + "' sums past the range of f64");
        }
        into.count += accumulator.count;
        into.low = std::min(into.low, accumulator.low);
        into.high = std::max(into.high, accumulator.high);
      }
      raw.reset();
    }

    tables_.clear();
    for (const std::string& name : columns_) {
      const Accumulator& accumulator = merged[name];
      if (accumulator.count == 0) {
        throw Error(stage + ": column '" + name + "' has no non-null values to learn from");
      }
      const auto rows = static_cast<double>(accumulator.count);
      Statistics statistics;
      statistics.count = accumulator.count;
      const std::string where = stage + ": column '" + name + "'";
      statistics.mean = finite_encoding(divided(accumulator.sum, rows, where + ": the mean"), stage, name, "columns");
      // A constant column's variance is 0, as sklearn's is. Below about
      // 1e-146 add_product's residuals fall under the smallest subnormal and
      // round, so the exact sums can miss that by a subnormal step either way:
      // a constant 1e-158 fit -5e-324, which load_state refuses, and a
      // constant 9e-159 fit +5e-324, which scaled it by 2.2e-162. Other
      // columns keep the variance the sums give, never below 0.
      statistics.variance = accumulator.low == accumulator.high
                                ? 0.0
                                : std::max(0.0, population_variance(accumulator, statistics.mean, rows, where));
      // sklearn's _is_constant_feature, on the raw variance and in its own
      // order of operations: a feature this flat is a constant one, and its
      // scale becomes 1 rather than a square root of noise.
      const double bias = (rows * statistics.mean) * kEpsilon;
      const bool constant = statistics.variance <= (rows * kEpsilon) * statistics.variance + bias * bias;
      statistics.scale = constant ? 1.0 : std::sqrt(statistics.variance);
      tables_.emplace(name, statistics);
    }
    fitted = true;
  }

  nlohmann::json state() const override {
    std::unordered_map<std::string, double> means;
    std::unordered_map<std::string, double> variances;
    std::unordered_map<std::string, double> scales;
    nlohmann::json counts = nlohmann::json::object();
    for (const std::string& name : columns_) {
      const Statistics& statistics = tables_.at(name);
      means.emplace(name, statistics.mean);
      variances.emplace(name, statistics.variance);
      scales.emplace(name, statistics.scale);
      counts[name] = statistics.count;
    }
    return {{"columns", columns_},
            {"means", numbers_state(columns_, means, stage)},
            {"variances", numbers_state(columns_, variances, stage)},
            {"scales", numbers_state(columns_, scales, stage)},
            {"counts", std::move(counts)}};
  }

  void load_state(const nlohmann::json& state) override {
    columns_from_state(state, "means", columns_, explicit_columns_, stage);
    const std::unordered_map<std::string, double> means = load_numbers(state, "means", columns_, stage);
    const std::unordered_map<std::string, double> variances = load_numbers(state, "variances", columns_, stage);
    const std::unordered_map<std::string, double> scales = load_numbers(state, "scales", columns_, stage);
    if (!state.contains("counts") || !state.at("counts").is_object()) {
      throw Error(stage + ": state must hold a 'counts' object");
    }
    tables_.clear();
    for (const std::string& name : columns_) {
      Statistics statistics;
      statistics.mean = means.at(name);
      statistics.variance = variances.at(name);
      statistics.scale = scales.at(name);
      if (statistics.variance < 0.0) throw Error(stage + ": variances['" + name + "'] must not be negative");
      // A scale of zero divides every row of the column by zero; the fit can
      // never produce one, so this is a hand-edited manifest.
      if (statistics.scale <= 0.0) throw Error(stage + ": scales['" + name + "'] must be a positive number");
      statistics.count = load_count(state.at("counts"), name, stage + ": counts['" + name + "']", 1);
      tables_.emplace(name, statistics);
    }
    inputs = columns_;
    outputs = columns_;
    fitted = true;
  }

  Batch transform(Batch batch) const override { return transform_owned(std::move(batch), {}); }

  std::span<const std::string> owned_output_columns() const override { return columns_; }

  Batch transform_owned(Batch batch, std::span<const std::string> reusable) const override {
    for (const std::string& name : columns_) {
      const int idx = batch.find(name);
      if (idx < 0) throw Error(stage + ": column '" + name + "' missing from batch");
      // The column's statistics are the same for every row, so the table is
      // read once here rather than once per value inside scaled().
      const Statistics& statistics = tables_.at(name);
      batch.cols[static_cast<std::size_t>(idx)] = scale_numeric_array(
          batch.cols[static_cast<std::size_t>(idx)], stage, name,
          [&](double value) { return scaled(statistics, value); },
          [&](double value) { finite_number(value, stage, name); },
          std::find(reusable.begin(), reusable.end(), name) != reusable.end());
    }
    return batch;
  }

  Row transform_row(Row row) const override {
    RecordLookup lookup(row, columns_.size());
    for (const std::string& name : columns_) {
      RowValue* value = lookup.find(name);
      if (value == nullptr) throw Error(stage + ": key '" + name + "' missing from record");
      const std::optional<double> number = row_number(*value, stage, name);
      *value = number.has_value() ? RowValue(scaled(tables_.at(name), *number)) : RowValue(std::monostate{});
    }
    return row;
  }

 private:
  struct Statistics {
    double mean = 0.0;
    double variance = 0.0;
    double scale = 1.0;
    int64_t count = 0;
  };

  // The population variance around a mean the caller has already rounded:
  // Sum x^2 - 2*mean*Sum x + n*mean^2, every term expanded so no bit is lost
  // until the division. It is the exact Sum (x - mean)^2, which is
  // both order-independent and free of the catastrophic cancellation a
  // Sum x^2 - n*mean^2 would suffer on a column of large, close values.
  //
  // REGRESSION: mean * Sum x is formed from the sum's terms(), which depend
  // on its value alone. It used to be formed from the partials, which two
  // workers can leave split differently than one. Near 1e-155 the products
  // underflow and round, so the fitted variance depended on the thread
  // count by a subnormal step.
  static double population_variance(const Accumulator& accumulator, double mean, double rows,
                                    const std::string& where) {
    ExactSum residual = accumulator.squares;
    const std::vector<double> terms = accumulator.sum.terms();
    bool held = true;
    for (double term : terms) held = held && add_product(residual, -2.0 * mean, term);
    const double square = mean * mean;
    held = held && add_product(residual, rows, square) && add_product(residual, rows, std::fma(mean, mean, -square));
    if (!held) throw Error(where + ": the variance leaves the range of f64");
    return divided(residual, rows, where + ": the variance");
  }

  // sklearn subtracts and then divides, so the two roundings happen in that
  // order here too — which is what makes the parity cases exact.
  double scaled(const Statistics& statistics, double value) const {
    double out = value;
    if (with_mean_) out -= statistics.mean;
    if (with_std_) out /= statistics.scale;
    return out;
  }

  std::vector<std::string> columns_;
  bool explicit_columns_ = false;
  bool with_mean_ = true;
  bool with_std_ = true;
  std::unordered_map<std::string, Statistics> tables_;
};

// ---------------------------------------------------------------------------
// min_max_scale — integer or float columns -> x * scale + min, f64 in place.
// params: {"columns": [...] | null, "feature_range": [low, high],
//          "clip": bool}
// Semantics:
//   * the fit records each column's smallest and largest non-null value, and
//     scale = (high - low) / (data_max - data_min), min = low - data_min *
//     scale, so the fitted range maps onto the feature range, up to
//     rounding that sklearn shares bit for bit
//   * a column whose range is indistinguishable from zero divides by 1
//     instead, so a constant column comes out at the low end of the range,
//     up to that rounding, rather than NaN
//   * a column that spans past the range of f64, or a feature range so
//     narrow beside it that the scale rounds to 0, is a fit-time error: a
//     scale of 0 maps every value to the low end, which sklearn does
//   * transform multiplies and then adds, in that order, as sklearn does;
//     clip=true then clamps the result into the feature range, which a
//     value outside the fitted one can leave, and a fitted extreme by that
//     rounding
//   * a null teaches nothing and comes out a null; the recorded row count is
//     every fitted row, nulls included, as sklearn's n_samples_seen_ is
//   * min and max are exact under any split, so the artifact is identical
//     however the rows were divided across workers, and every parity case is
//     exact
//   * columns=null scales every integer or float column that reaches the
//     step, and the output is f64 whatever the input width
// ---------------------------------------------------------------------------
class MinMaxScale final : public Kernel {
 public:
  explicit MinMaxScale(const nlohmann::json& params) {
    if (params.contains("columns") && !params.at("columns").is_null()) {
      columns_ = column_list(params, "columns", "min_max_scale");
      explicit_columns_ = true;
      inputs = columns_;
      outputs = columns_;
    }
    if (params.contains("feature_range") && !params.at("feature_range").is_null()) {
      const std::string message =
          "min_max_scale: 'feature_range' must be two finite numbers, the first smaller than the second";
      const nlohmann::json& range = params.at("feature_range");
      if (!range.is_array() || range.size() != 2 || !range.at(0).is_number() || !range.at(1).is_number()) {
        throw Error(message);
      }
      low_ = range.at(0).get<double>();
      high_ = range.at(1).get<double>();
      if (!std::isfinite(low_) || !std::isfinite(high_) || low_ >= high_) throw Error(message);
    }
    if (params.contains("clip") && !params.at("clip").is_null()) {
      if (!params.at("clip").is_boolean()) throw Error("min_max_scale: 'clip' must be true or false");
      clip_ = params.at("clip").get<bool>();
    }
  }

  std::string id() const override { return "min_max_scale"; }

  nlohmann::json params() const override {
    nlohmann::json out;
    out["columns"] = explicit_columns_ ? nlohmann::json(columns_) : nlohmann::json();
    out["feature_range"] = nlohmann::json::array({low_, high_});
    out["clip"] = clip_;
    return out;
  }

  void resolve(Schema& schema) override {
    bind_numeric(schema, stage, columns_, explicit_columns_, fitted, "scale");
    retype(schema, columns_, "f64");
    inputs = columns_;
    outputs = columns_;
  }

  bool needs_fit() const override { return true; }

  std::unique_ptr<AccState> make_state() const override { return std::make_unique<ExtremaState>(); }

  void consume(AccState& raw, const Batch& batch) const override {
    consume_extrema(columns_, stage, static_cast<ExtremaState&>(raw), batch);
  }

  void finalize(std::vector<std::unique_ptr<AccState>> states) override {
    const std::unordered_map<std::string, Extrema> merged = merge_extrema(columns_, stage, states, rows_);
    tables_.clear();
    for (const std::string& name : columns_) {
      const Extrema& extrema = merged.at(name);
      Range range;
      range.data_min = extrema.low;
      range.data_max = extrema.high;
      const double span = extrema.high - extrema.low;
      if (std::isinf(span)) {
        throw Error(stage + ": column '" + name + "' spans past the range of f64");
      }
      range.scale = finite_encoding((high_ - low_) / handle_zeros(span), stage, name, "feature_range");
      if (range.scale == 0.0) {
        throw Error(stage + ": column '" + name + "' fits a scale that rounds to 0; widen 'feature_range'");
      }
      range.offset = finite_encoding(low_ - extrema.low * range.scale, stage, name, "feature_range");
      tables_.emplace(name, range);
    }
    fitted = true;
  }

  nlohmann::json state() const override {
    std::unordered_map<std::string, double> data_min;
    std::unordered_map<std::string, double> data_max;
    std::unordered_map<std::string, double> scales;
    std::unordered_map<std::string, double> mins;
    for (const std::string& name : columns_) {
      const Range& range = tables_.at(name);
      data_min.emplace(name, range.data_min);
      data_max.emplace(name, range.data_max);
      scales.emplace(name, range.scale);
      mins.emplace(name, range.offset);
    }
    return {{"columns", columns_},
            {"data_min", numbers_state(columns_, data_min, stage)},
            {"data_max", numbers_state(columns_, data_max, stage)},
            {"scales", numbers_state(columns_, scales, stage)},
            {"mins", numbers_state(columns_, mins, stage)},
            {"rows", rows_}};
  }

  void load_state(const nlohmann::json& state) override {
    columns_from_state(state, "data_min", columns_, explicit_columns_, stage);
    const std::unordered_map<std::string, double> data_min = load_numbers(state, "data_min", columns_, stage);
    const std::unordered_map<std::string, double> data_max = load_numbers(state, "data_max", columns_, stage);
    const std::unordered_map<std::string, double> scales = load_numbers(state, "scales", columns_, stage);
    const std::unordered_map<std::string, double> mins = load_numbers(state, "mins", columns_, stage);
    rows_ = load_count(state, "rows", stage + ": 'rows'", 0);
    tables_.clear();
    for (const std::string& name : columns_) {
      Range range;
      range.data_min = data_min.at(name);
      range.data_max = data_max.at(name);
      range.scale = scales.at(name);
      range.offset = mins.at(name);
      if (range.data_min > range.data_max) {
        throw Error(stage + ": data_min['" + name + "'] must not be above data_max['" + name + "']");
      }
      // A zero scale would collapse every row of the column onto the offset
      // and a negative one would reverse it, both without a word, so both are
      // refused here, as in the other scalers.
      if (range.scale <= 0.0) throw Error(stage + ": scales['" + name + "'] must be a positive number");
      tables_.emplace(name, range);
    }
    inputs = columns_;
    outputs = columns_;
    fitted = true;
  }

  Batch transform(Batch batch) const override { return transform_owned(std::move(batch), {}); }

  std::span<const std::string> owned_output_columns() const override { return columns_; }

  Batch transform_owned(Batch batch, std::span<const std::string> reusable) const override {
    for (const std::string& name : columns_) {
      const int idx = batch.find(name);
      if (idx < 0) throw Error(stage + ": column '" + name + "' missing from batch");
      // The column's range is the same for every row, so the table is read
      // once here rather than once per value inside scaled().
      const Range& range = tables_.at(name);
      batch.cols[static_cast<std::size_t>(idx)] = scale_numeric_array(
          batch.cols[static_cast<std::size_t>(idx)], stage, name, [&](double value) { return scaled(range, value); },
          [&](double value) { finite_number(value, stage, name); },
          std::find(reusable.begin(), reusable.end(), name) != reusable.end());
    }
    return batch;
  }

  Row transform_row(Row row) const override {
    RecordLookup lookup(row, columns_.size());
    for (const std::string& name : columns_) {
      RowValue* value = lookup.find(name);
      if (value == nullptr) throw Error(stage + ": key '" + name + "' missing from record");
      const std::optional<double> number = row_number(*value, stage, name);
      *value = number.has_value() ? RowValue(scaled(tables_.at(name), *number)) : RowValue(std::monostate{});
    }
    return row;
  }

 private:
  struct Range {
    double data_min = 0.0;
    double data_max = 0.0;
    double scale = 1.0;
    double offset = 0.0;
  };

  // sklearn multiplies and then adds, so the two roundings happen in that
  // order here too.
  double scaled(const Range& range, double value) const {
    const double out = value * range.scale + range.offset;
    if (!clip_) return out;
    // Ordered comparisons preserve NaN and signed-zero behavior while
    // exposing a pair of independent selects to the vectorizer.
    return out < low_ ? low_ : (out > high_ ? high_ : out);
  }

  std::vector<std::string> columns_;
  bool explicit_columns_ = false;
  double low_ = 0.0;
  double high_ = 1.0;
  bool clip_ = false;
  int64_t rows_ = 0;
  std::unordered_map<std::string, Range> tables_;
};

// ---------------------------------------------------------------------------
// max_abs_scale — integer or float columns -> x / max|x|, f64 in place.
// params: {"columns": [...] | null, "clip": bool}
// Semantics:
//   * the fit records each column's largest absolute non-null value, and the
//     transform divides by it, so the fitted data lands in [-1, 1] and a zero
//     stays a zero — the scaler that leaves a sparse column sparse
//   * a column whose largest absolute value is indistinguishable from zero
//     divides by 1 instead, so an all-zero column comes out zeros
//   * clip=true clamps the result to [-1, 1], which only a value larger in
//     magnitude than any seen at fit can leave
//   * a null teaches nothing and comes out a null; the recorded row count is
//     every fitted row, nulls included, as sklearn's n_samples_seen_ is
//   * max is exact under any split, so the artifact is identical however the
//     rows were divided across workers, and every parity case is exact
//   * columns=null scales every integer or float column that reaches the
//     step, and the output is f64 whatever the input width
// ---------------------------------------------------------------------------
class MaxAbsScale final : public Kernel {
 public:
  explicit MaxAbsScale(const nlohmann::json& params) {
    if (params.contains("columns") && !params.at("columns").is_null()) {
      columns_ = column_list(params, "columns", "max_abs_scale");
      explicit_columns_ = true;
      inputs = columns_;
      outputs = columns_;
    }
    if (params.contains("clip") && !params.at("clip").is_null()) {
      if (!params.at("clip").is_boolean()) throw Error("max_abs_scale: 'clip' must be true or false");
      clip_ = params.at("clip").get<bool>();
    }
  }

  std::string id() const override { return "max_abs_scale"; }

  nlohmann::json params() const override {
    nlohmann::json out;
    out["columns"] = explicit_columns_ ? nlohmann::json(columns_) : nlohmann::json();
    out["clip"] = clip_;
    return out;
  }

  void resolve(Schema& schema) override {
    bind_numeric(schema, stage, columns_, explicit_columns_, fitted, "scale");
    retype(schema, columns_, "f64");
    inputs = columns_;
    outputs = columns_;
  }

  bool needs_fit() const override { return true; }

  std::unique_ptr<AccState> make_state() const override { return std::make_unique<ExtremaState>(); }

  void consume(AccState& raw, const Batch& batch) const override {
    consume_extrema(columns_, stage, static_cast<ExtremaState&>(raw), batch);
  }

  void finalize(std::vector<std::unique_ptr<AccState>> states) override {
    const std::unordered_map<std::string, Extrema> merged = merge_extrema(columns_, stage, states, rows_);
    tables_.clear();
    for (const std::string& name : columns_) {
      const Extrema& extrema = merged.at(name);
      Magnitude magnitude;
      magnitude.max_abs = std::max(std::fabs(extrema.low), std::fabs(extrema.high));
      magnitude.scale = handle_zeros(magnitude.max_abs);
      tables_.emplace(name, magnitude);
    }
    fitted = true;
  }

  nlohmann::json state() const override {
    std::unordered_map<std::string, double> max_abs;
    std::unordered_map<std::string, double> scales;
    for (const std::string& name : columns_) {
      max_abs.emplace(name, tables_.at(name).max_abs);
      scales.emplace(name, tables_.at(name).scale);
    }
    return {{"columns", columns_},
            {"max_abs", numbers_state(columns_, max_abs, stage)},
            {"scales", numbers_state(columns_, scales, stage)},
            {"rows", rows_}};
  }

  void load_state(const nlohmann::json& state) override {
    columns_from_state(state, "max_abs", columns_, explicit_columns_, stage);
    const std::unordered_map<std::string, double> max_abs = load_numbers(state, "max_abs", columns_, stage);
    const std::unordered_map<std::string, double> scales = load_numbers(state, "scales", columns_, stage);
    rows_ = load_count(state, "rows", stage + ": 'rows'", 0);
    tables_.clear();
    for (const std::string& name : columns_) {
      Magnitude magnitude;
      magnitude.max_abs = max_abs.at(name);
      magnitude.scale = scales.at(name);
      if (magnitude.max_abs < 0.0) throw Error(stage + ": max_abs['" + name + "'] must not be negative");
      // A scale of zero divides every row of the column by zero; the fit can
      // never produce one, so this is a hand-edited manifest.
      if (magnitude.scale <= 0.0) throw Error(stage + ": scales['" + name + "'] must be a positive number");
      tables_.emplace(name, magnitude);
    }
    inputs = columns_;
    outputs = columns_;
    fitted = true;
  }

  Batch transform(Batch batch) const override { return transform_owned(std::move(batch), {}); }

  std::span<const std::string> owned_output_columns() const override { return columns_; }

  Batch transform_owned(Batch batch, std::span<const std::string> reusable) const override {
    for (const std::string& name : columns_) {
      const int idx = batch.find(name);
      if (idx < 0) throw Error(stage + ": column '" + name + "' missing from batch");
      // The column's magnitude is the same for every row, so the table is
      // read once here rather than once per value inside scaled().
      const Magnitude& magnitude = tables_.at(name);
      batch.cols[static_cast<std::size_t>(idx)] = scale_numeric_array(
          batch.cols[static_cast<std::size_t>(idx)], stage, name,
          [&](double value) { return scaled(magnitude, value); },
          [&](double value) { finite_number(value, stage, name); },
          std::find(reusable.begin(), reusable.end(), name) != reusable.end());
    }
    return batch;
  }

  Row transform_row(Row row) const override {
    RecordLookup lookup(row, columns_.size());
    for (const std::string& name : columns_) {
      RowValue* value = lookup.find(name);
      if (value == nullptr) throw Error(stage + ": key '" + name + "' missing from record");
      const std::optional<double> number = row_number(*value, stage, name);
      *value = number.has_value() ? RowValue(scaled(tables_.at(name), *number)) : RowValue(std::monostate{});
    }
    return row;
  }

 private:
  struct Magnitude {
    double max_abs = 0.0;
    double scale = 1.0;
  };

  double scaled(const Magnitude& magnitude, double value) const {
    const double out = value / magnitude.scale;
    return clip_ ? std::clamp(out, -1.0, 1.0) : out;
  }

  std::vector<std::string> columns_;
  bool explicit_columns_ = false;
  bool clip_ = false;
  int64_t rows_ = 0;
  std::unordered_map<std::string, Magnitude> tables_;
};

// ---------------------------------------------------------------------------
// binarize — integer or float columns -> 0 or 1, i64 in place. Stateless.
// params: {"columns": [...], "threshold": number}
// Semantics:
//   * a value strictly above the threshold becomes 1 and everything else 0,
//     so a value sitting exactly on the threshold is a 0
//   * a null is an error, not a 0 and not a 1: sklearn refuses a NaN here
//     rather than guessing, and so do we, on the batch and record paths alike
//   * sklearn keeps the input dtype; our output type is declared, so the
//     column becomes i64 whatever came in
//   * nothing is learned: needs_fit() is false and the state is empty, which
//     is why `columns` is required — an artifact that did not name them would
//     binarize a different set of columns for every file it was handed
// ---------------------------------------------------------------------------
class Binarize final : public Kernel {
 public:
  explicit Binarize(const nlohmann::json& params) : columns_(column_list(params, "columns", "binarize")) {
    if (params.contains("threshold") && !params.at("threshold").is_null()) {
      const nlohmann::json& threshold = params.at("threshold");
      if (!threshold.is_number() || !std::isfinite(threshold.get<double>())) {
        throw Error("binarize: 'threshold' must be a finite number");
      }
      threshold_ = threshold.get<double>();
    }
    inputs = columns_;
    outputs = columns_;
  }

  std::string id() const override { return "binarize"; }

  nlohmann::json params() const override { return {{"columns", columns_}, {"threshold", threshold_}}; }

  void resolve(Schema& schema) override {
    // `columns` is required here, so there is nothing to derive: this only
    // checks that every listed column is present and numeric.
    bind_numeric(schema, stage, columns_, true, false, "binarize");
    retype(schema, columns_, "i64");
    inputs = columns_;
    outputs = columns_;
  }

  // Nothing is learned, so state() stays empty; a manifest that carries a
  // table for this stage was hand-edited and is refused rather than ignored.
  void load_state(const nlohmann::json& state) override {
    if (!state.is_object() || !state.empty()) {
      throw Error(stage + ": binarize learns nothing, so its 'state' must be an empty object");
    }
  }

  Batch transform(Batch batch) const override {
    for (const std::string& name : columns_) {
      const int idx = batch.find(name);
      if (idx < 0) throw Error(stage + ": column '" + name + "' missing from batch");
      const arrow::Array& array = *batch.cols[static_cast<std::size_t>(idx)];
      const std::string where = stage + ": column '" + name + "'";
      PrimitiveOutput<arrow::Int64Type> output(array.length(), where);
      int64_t* values = output.mutable_values();
      const auto compare = [&](double value) { return above(value); };
      const auto check = [&](double value) { finite_number(value, stage, name, false); };
      if (array.null_count() == 0 && array.type_id() == arrow::Type::DOUBLE) {
        scale_dense_values(static_cast<const arrow::DoubleArray&>(array).raw_values(), values, array.length(), compare,
                           check);
      } else if (array.null_count() == 0 && array.type_id() == arrow::Type::FLOAT) {
        scale_dense_values(static_cast<const arrow::FloatArray&>(array).raw_values(), values, array.length(), compare,
                           check);
      } else {
        for_each_numeric(array, stage, name, [&](int64_t row, bool is_null, double value) {
          if (is_null) refuse_missing(name);
          values[row] = above(finite_number(value, stage, name, false));
        });
      }
      batch.cols[static_cast<std::size_t>(idx)] = output.finish();
    }
    return batch;
  }

  Row transform_row(Row row) const override {
    RecordLookup lookup(row, columns_.size());
    for (const std::string& name : columns_) {
      RowValue* value = lookup.find(name);
      if (value == nullptr) throw Error(stage + ": key '" + name + "' missing from record");
      const std::optional<double> number = row_number(*value, stage, name, false);
      if (!number.has_value()) refuse_missing(name);
      *value = above(*number);
    }
    return row;
  }

 private:
  [[noreturn]] void refuse_missing(const std::string& name) const {
    throw Error(stage + ": column '" + name +
                "' has a null, which binarize has no answer for; fill or drop it before this step");
  }

  int64_t above(double value) const { return value > threshold_ ? 1 : 0; }

  std::vector<std::string> columns_;
  double threshold_ = 0.0;
};

// ---------------------------------------------------------------------------
// normalize — every listed column of one row divided by that row's norm, f64
// in place. Stateless.
// params: {"columns": [...], "norm": "l1" | "l2" | "max"}
// Semantics:
//   * each row is divided by its norm over the listed columns: the sum of
//     their absolute values ("l1"), the square root of the sum of their
//     squares ("l2"), or the largest of their absolute values ("max")
//   * a row whose norm is indistinguishable from zero is left alone rather
//     than divided, so an all-zero row stays all zeros
//   * this is the one scaler whose output depends on other columns of the
//     same row, so `columns` is a required, ordered parameter: the artifact
//     names the whole vector, and a file with one column missing is an error
//     rather than a quietly different answer
//   * a null is an error, as sklearn's NaN is: a norm over a missing number
//     has no value
//   * the l1 and l2 sums are exact partials, so the norm does not depend on
//     the order the columns were listed in
//   * every listed column becomes f64 whatever the input width
// ---------------------------------------------------------------------------
class Normalize final : public Kernel {
 public:
  explicit Normalize(const nlohmann::json& params) : columns_(column_list(params, "columns", "normalize")) {
    if (params.contains("norm") && !params.at("norm").is_null()) {
      const nlohmann::json& norm = params.at("norm");
      const std::string text = norm.is_string() ? norm.get<std::string>() : std::string();
      if (text == "l1") {
        norm_ = Norm::L1;
      } else if (text == "l2") {
        norm_ = Norm::L2;
      } else if (text == "max") {
        norm_ = Norm::Max;
      } else {
        throw Error("normalize: 'norm' must be 'l1', 'l2' or 'max'");
      }
    }
    inputs = columns_;
    outputs = columns_;
  }

  std::string id() const override { return "normalize"; }

  nlohmann::json params() const override { return {{"columns", columns_}, {"norm", norm_name()}}; }

  void resolve(Schema& schema) override {
    // `columns` is required here, so there is nothing to derive: this only
    // checks that every listed column is present and numeric.
    bind_numeric(schema, stage, columns_, true, false, "normalize");
    retype(schema, columns_, "f64");
    inputs = columns_;
    outputs = columns_;
  }

  void load_state(const nlohmann::json& state) override {
    if (!state.is_object() || !state.empty()) {
      throw Error(stage + ": normalize learns nothing, so its 'state' must be an empty object");
    }
  }

  Batch transform(Batch batch) const override {
    const auto rows = static_cast<std::size_t>(column_array(batch, stage, columns_.front()).length());
    // Validate in column order into fresh output storage before computing any
    // row's norm: the first bad value in column order is the one reported,
    // and the input arrays, which other holders may share, are never written.
    std::vector<PrimitiveOutput<arrow::DoubleType>> output;
    output.reserve(columns_.size());
    std::vector<double*> values;
    values.reserve(columns_.size());
    for (std::size_t index = 0; index < columns_.size(); ++index) {
      const std::string& name = columns_[index];
      output.emplace_back(static_cast<int64_t>(rows), stage + ": column '" + name + "'");
      double* column = output.back().mutable_values();
      values.push_back(column);
      for_each_numeric(column_array(batch, stage, name), stage, name, [&](int64_t row, bool is_null, double value) {
        if (is_null) refuse_missing(name);
        column[row] = finite_number(value, stage, name, false);
      });
    }
    ExactSum total;
    std::vector<double> divisors(rows);
    for (std::size_t row = 0; row < rows; ++row) {
      divisors[row] = row_divisor(columns_.size(), [&](std::size_t index) { return values[index][row]; }, total);
    }
    for (std::size_t index = 0; index < columns_.size(); ++index) {
      // Independent divisions vectorize across rows, keeping division rather
      // than multiplication by a rounded reciprocal.
      double* column = values[index];
      for (std::size_t row = 0; row < rows; ++row) column[row] /= divisors[row];
      const int idx = batch.find(columns_[index]);
      if (idx < 0) throw Error(stage + ": column '" + columns_[index] + "' missing from batch");
      batch.cols[static_cast<std::size_t>(idx)] = output[index].finish();
    }
    return batch;
  }

  Row transform_row(Row row) const override {
    RecordLookup lookup(row, columns_.size());
    // The whole vector first, then the divisor, then the write-back: a row's
    // answer for one column needs every other listed column of it.
    std::vector<double> one(columns_.size(), 0.0);
    std::vector<RowValue*> slots(columns_.size(), nullptr);
    for (std::size_t index = 0; index < columns_.size(); ++index) {
      slots[index] = lookup.find(columns_[index]);
      if (slots[index] == nullptr) throw Error(stage + ": key '" + columns_[index] + "' missing from record");
      const std::optional<double> number = row_number(*slots[index], stage, columns_[index], false);
      if (!number.has_value()) refuse_missing(columns_[index]);
      one[index] = *number;
    }
    ExactSum total;
    const double divisor = row_divisor(one.size(), [&](std::size_t index) { return one[index]; }, total);
    for (std::size_t index = 0; index < columns_.size(); ++index) *slots[index] = one[index] / divisor;
    return row;
  }

 private:
  enum class Norm { L1, L2, Max };

  const char* norm_name() const {
    switch (norm_) {
      case Norm::L1:
        return "l1";
      case Norm::Max:
        return "max";
      case Norm::L2:
        break;
    }
    return "l2";
  }

  [[noreturn]] void refuse_missing(const std::string& name) const {
    throw Error(stage + ": column '" + name +
                "' has a null, and a row's norm has no value over a missing number; fill or drop it before this step");
  }

  // The divisor one row is scaled by, with sklearn's zero rule: a norm below
  // ten epsilons leaves the row alone rather than blowing it up.
  template <typename At>
  double row_divisor(std::size_t count, At&& at, ExactSum& total) const {
    if (norm_ == Norm::Max) {
      double largest = 0.0;
      for (std::size_t index = 0; index < count; ++index) largest = std::max(largest, std::fabs(at(index)));
      return handle_zeros(largest);
    }
    // For l2 these are the very squares numpy's einsum sums, correctly
    // rounded once instead of accumulated left to right. IEEE addition
    // already rounds the sum of two doubles correctly, so a row of one or two
    // values gets the same answer without partials.
    if (count <= 2) {
      double sum = 0.0;
      for (std::size_t index = 0; index < count; ++index) {
        const double value = at(index);
        sum += norm_ == Norm::L1 ? std::fabs(value) : value * value;
      }
      if (!std::isfinite(sum)) throw Error(stage + ": a row's " + norm_name() + " norm leaves the range of f64");
      return handle_zeros(norm_ == Norm::L1 ? sum : std::sqrt(sum));
    }
    total.clear();
    bool held = true;
    {
      // The Adder is gone, and its terms written back, before the sum is read.
      ExactSum::Adder adder(total);
      for (std::size_t index = 0; index < count; ++index) {
        const double value = at(index);
        held = held && adder.add(norm_ == Norm::L1 ? std::fabs(value) : value * value);
      }
    }
    if (!held) throw Error(stage + ": a row's " + norm_name() + " norm leaves the range of f64");
    return handle_zeros(norm_ == Norm::L1 ? total.value() : std::sqrt(total.value()));
  }

  std::vector<std::string> columns_;
  Norm norm_ = Norm::L2;
};

}  // namespace

std::unique_ptr<Kernel> make_standard_scale(const nlohmann::json& params) {
  return std::make_unique<StandardScale>(params);
}

std::unique_ptr<Kernel> make_min_max_scale(const nlohmann::json& params) {
  return std::make_unique<MinMaxScale>(params);
}

std::unique_ptr<Kernel> make_max_abs_scale(const nlohmann::json& params) {
  return std::make_unique<MaxAbsScale>(params);
}

std::unique_ptr<Kernel> make_binarize(const nlohmann::json& params) { return std::make_unique<Binarize>(params); }

std::unique_ptr<Kernel> make_normalize(const nlohmann::json& params) { return std::make_unique<Normalize>(params); }

}  // namespace fastfeaturetransform
