// The three sklearn.preprocessing transformers that turn numeric columns into
// a wider or reshaped block of numeric columns: polynomial_features (every
// monomial across the listed columns), spline_transform (a B-spline basis per
// column, from fitted knots) and power_transform (one Box-Cox or Yeo-Johnson
// lambda per column, fitted by maximum likelihood).
//
// They share a substrate the categorical kernels do not need: reading an
// integer or float column as f64, and refusing anything else rather than
// converting it. Nothing here coerces a value; a column of the wrong type is
// an error naming the column and saying to Cast it.

#include <algorithm>
#include <cmath>
#include <format>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "fastfeaturetransform/kernels/cast_table.hpp"
#include "fastfeaturetransform/kernels/categories.hpp"
#include "fastfeaturetransform/kernels/kernels.hpp"
#include "fastfeaturetransform/kernels/numeric.hpp"
#include "fastfeaturetransform/kernels/output.hpp"
#include "fastfeaturetransform/kernels/util.hpp"
#include "fastfeaturetransform/spill.hpp"

namespace fastfeaturetransform {
namespace {

// --- the numeric substrate -------------------------------------------------

// An integer column read as f64. Widening is exact up to 2^53 and an error
// above it, which is the rule Cast follows: sklearn's own validate_data
// widens an integer frame to float64 silently, and a silently rounded id is
// worse than a refusal.
double exact_double(int64_t value, const std::string& where) {
  constexpr int64_t kExact = int64_t{1} << 53;
  if (value > kExact || value < -kExact) {
    throw Error(where + ": value " + std::to_string(value) +
                " is beyond 2^53 in magnitude, where f64 no longer holds every integer exactly");
  }
  return static_cast<double>(value);
}

// A float column's value. A NaN or an infinity is an error. sklearn refuses
// NaN too, except where it reads NaN as missing (PowerTransformer, and
// SplineTransformer with handle_missing='zeros'); a null takes that role
// here. Refusing both also keeps the fitted tables JSON-serializable.
//
// The throw is a function of its own: std::format's buffer would otherwise
// give every call a stack frame and a stack-protector check, and this runs
// once per value read.
[[noreturn]] void refuse_non_finite(double value, const std::string& where) {
  throw Error(where + ": value " + std::format("{}", value) + " is not finite");
}

double finite_value(double value, const std::string& where) {
  if (!std::isfinite(value)) refuse_non_finite(value, where);
  return value;
}

// numeric.hpp's visitor with this family's value policy on top: an integer
// past 2^53 is refused rather than rounded, and a NaN or an infinity in a
// float column is refused wherever it turns up. A null carries no value, so
// neither check sees one.
template <typename F>
void for_each_number(const arrow::Array& array, const std::string& stage, const std::string& name, F&& fn) {
  const std::string where = stage + ": column '" + name + "'";
  for_each_numeric(
      array, stage, name, [&](int64_t value) { return exact_double(value, where); },
      [&](int64_t row, bool is_null, double value) { fn(row, is_null, is_null ? 0.0 : finite_value(value, where)); });
}

// The two expanders both consume their input columns and emit generated ones
// in their place, so both ask the same two questions of a batch.
bool is_expanded(const std::vector<std::string>& columns, const std::string& name) {
  return std::find(columns.begin(), columns.end(), name) != columns.end();
}

void drop_expanded_columns(const std::vector<std::string>& columns, Batch& batch) {
  Batch kept;
  kept.rows = batch.rows;
  for (std::size_t i = 0; i < batch.names.size(); ++i) {
    if (is_expanded(columns, batch.names[i])) continue;
    kept.names.push_back(std::move(batch.names[i]));
    kept.cols.push_back(std::move(batch.cols[i]));
  }
  batch = std::move(kept);
}

// A value one of the two expanders generated. Every kernel in the engine that
// reads a number refuses one that is not finite, so an expander that emitted
// an infinity would hand the next step a column that step cannot read: the
// pipeline would fit, write its manifest, and only fail when a later numeric
// step or a downstream reader met the value. a^2 for a = 1e200 and a spline
// basis continued far past its knots both reach it, so the refusal is here,
// at the step that produced it and naming the column it produced.
//
// Both transform paths call this, so a record and a batch refuse the same row.
// It runs once per generated value, so the throw is out of line for the same
// reason as refuse_non_finite's.
[[noreturn]] void refuse_generated(double value, const std::string& stage, const std::string& name) {
  throw Error(stage + ": column '" + name + "' generates the value " + std::format("{}", value) +
              ", which is not finite");
}

double generated_number(double value, const std::string& stage, const std::string& name) {
  if (!std::isfinite(value)) refuse_generated(value, stage, name);
  return value;
}

// One record value as a number, or nullopt for a null. The record path's twin
// of for_each_number, so both paths refuse the same values. A finite double
// returns before the key is spelled into a message: that string is a heap
// allocation per value, a real share of a call that takes well under a
// microsecond.
std::optional<double> number_value(const RowValue& value, const std::string& stage, const std::string& name) {
  if (std::holds_alternative<std::monostate>(value)) return std::nullopt;
  const double* number = std::get_if<double>(&value);
  if (number != nullptr && std::isfinite(*number)) return *number;
  const std::string where = stage + ": key '" + name + "'";
  if (const int64_t* integer = std::get_if<int64_t>(&value)) return exact_double(*integer, where);
  if (number != nullptr) return finite_value(*number, where);
  throw Error(where + " must be a number or null");
}

// The exact sum, by Shewchuk's partials. Spilled fit buffers are read
// sequentially in bounded chunks; chunk boundaries do not change the
// accumulator or any arithmetic operation.
template <typename F>
double exact_sum(const SpillVector<double>& values, const std::string& where, F&& fn) {
  ExactSum sum;
  values.for_each_chunk([&](std::span<const double> chunk) {
    for (double value : chunk) {
      if (!sum.add(fn(value))) throw Error(where + ": a sum over the fitted column left the range of f64");
    }
  });
  return sum.value();
}

// numpy's default percentile method ("linear") over an ascending vector, with
// the rank in 0..100 as np.percentile takes it: the divide by 100 is done
// here, not by the caller, because 100*t/100 is not always t.
double percentile_of(const SpillVector<double>& sorted, double rank) {
  const double quantile = rank / 100.0;
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

// A fitted number read back out of a manifest.
double manifest_number(const nlohmann::json& value, const std::string& where) {
  if (!value.is_number() || !std::isfinite(value.get<double>())) throw Error(where + " must hold finite numbers");
  return value.get<double>();
}

// ---------------------------------------------------------------------------
// polynomial_features — numeric columns -> one column per monomial of the
// listed columns, replacing them.
// params: {"columns": [...], "degree": int | [min, max],
//          "interaction_only": bool, "include_bias": bool}
// Semantics:
//   * every monomial of total degree between min_degree and degree over the
//     listed columns, in sklearn's order: the bias first when include_bias,
//     then degree by degree, and within a degree by the column order given
//   * interaction_only drops any monomial with an exponent above 1, so only
//     products of distinct columns survive
//   * the width is a function of the params alone, so this kernel never fits:
//     resolve() declares its outputs before any data is read, and a later
//     step in the same pipeline can name them
//   * outputs are named 'a*b' and 'a^2' where sklearn writes 'a b' and 'a^2',
//     and the bias column is named '1' as sklearn names it; a source column
//     whose name holds '*', '^' or a space is refused so a generated name is
//     never ambiguous
//   * every factor is multiplied in sklearn's own order — the monomial folded
//     from its last column to its first — so the products agree bit for bit
//   * a null factor makes the monomial null; the bias has no factors and
//     stays 1 (sklearn refuses a NaN outright)
//   * outputs are f64 whatever the inputs are (sklearn keeps a float32 input
//     float32); an integer column is read exactly or refused
// ---------------------------------------------------------------------------
class PolynomialFeatures final : public Kernel {
 public:
  explicit PolynomialFeatures(const nlohmann::json& params)
      : columns_(column_list(params, "columns", "polynomial_features")) {
    for (const std::string& name : columns_) {
      if (name.find_first_of("*^ ") != std::string::npos) {
        throw Error("polynomial_features: column '" + name +
                    "' holds one of '*', '^' or a space, which would make a generated name ambiguous");
      }
    }
    read_degree(params);
    if (params.contains("interaction_only") && !params.at("interaction_only").is_null()) {
      if (!params.at("interaction_only").is_boolean()) {
        throw Error("polynomial_features: 'interaction_only' must be true or false");
      }
      interaction_only_ = params.at("interaction_only").get<bool>();
    }
    if (params.contains("include_bias") && !params.at("include_bias").is_null()) {
      if (!params.at("include_bias").is_boolean()) {
        throw Error("polynomial_features: 'include_bias' must be true or false");
      }
      include_bias_ = params.at("include_bias").get<bool>();
    }
    if (max_degree_ == 0 && !include_bias_) {
      throw Error("polynomial_features: degree 0 with include_bias false would generate no columns");
    }
    // The width is known here: the column count and the degree decide it, so
    // an artifact-loaded kernel (which carries no state) declares what it
    // reads and writes without ever seeing a schema.
    build_monomials();
    inputs = columns_;
    outputs = names_;
  }

  std::string id() const override { return "polynomial_features"; }

  nlohmann::json params() const override {
    return {{"columns", columns_},
            {"degree", tuple_degree_ ? nlohmann::json({min_degree_, max_degree_}) : nlohmann::json(max_degree_)},
            {"interaction_only", interaction_only_},
            {"include_bias", include_bias_}};
  }

  void resolve(Schema& schema) override {
    bind_numeric(schema, stage, columns_, true, false, "expand");
    // The expanded columns leave first, so a generated name only collides
    // with a column that actually survives the step (expanding 'a' beside
    // 'a^2' has to mean the same thing in both paths).
    std::erase_if(schema.cols, [&](const auto& col) { return expanded(col.first); });
    for (const std::string& name : names_) {
      if (schema.find(name) != nullptr) throw Error(stage + ": output column '" + name + "' already exists in input");
    }
    for (const std::string& name : names_) schema.cols.emplace_back(name, "f64");
    inputs = columns_;
    outputs = names_;
  }

  // Nothing is learned, so state() stays empty; a manifest that carries a
  // table for this stage was hand-edited and is refused rather than ignored.
  void load_state(const nlohmann::json& state) override {
    if (!state.is_object() || !state.empty()) {
      throw Error(stage + ": polynomial_features learns nothing, so its 'state' must be an empty object");
    }
  }

  Batch transform(Batch batch) const override {
    const auto rows = static_cast<std::size_t>(column_array(batch, stage, columns_.front()).length());
    // The visitor writes every row, so the values need no zeroing first.
    std::vector<std::unique_ptr<double[]>> values(columns_.size());
    std::vector<std::vector<bool>> nulls(columns_.size(), std::vector<bool>(rows, false));
    std::vector<bool> has_nulls(columns_.size(), false);
    for (std::size_t c = 0; c < columns_.size(); ++c) {
      values[c] = std::make_unique_for_overwrite<double[]>(rows);
      for_each_number(column_array(batch, stage, columns_[c]), stage, columns_[c],
                      [&](int64_t row, bool is_null, double value) {
                        const auto slot = static_cast<std::size_t>(row);
                        values[c][slot] = value;
                        // Both flags start false, so only a null row writes
                        // them: a packed bit written every row is a
                        // read-modify-write that waits on the row before.
                        if (is_null) {
                          nulls[c][slot] = true;
                          has_nulls[c] = true;
                        }
                      });
    }
    drop_expanded(batch);
    const std::size_t generated_begin = batch.cols.size();
    for (std::size_t m = 0; m < monomials_.size(); ++m) {
      const std::string where = stage + ": column '" + names_[m] + "'";
      PrimitiveOutput<arrow::DoubleType> output(static_cast<int64_t>(rows), where);
      double* column = output.mutable_values();
      const std::size_t suffix = suffixes_[m];
      if (suffix != monomials_.size()) {
        const auto& previous = static_cast<const arrow::DoubleArray&>(*batch.cols[generated_begin + suffix]);
        const std::size_t leading = monomials_[m].front();
        if (previous.null_count() == 0 && !has_nulls[leading]) {
          bool finite = true;
          const double* preceding = previous.raw_values();
          const double* factor = values[leading].get();
          for (std::size_t row = 0; row < rows; ++row) {
            column[row] = preceding[row] * factor[row];
            finite &= std::isfinite(column[row]);
          }
          if (!finite) {
            for (std::size_t row = 0; row < rows; ++row) generated_number(column[row], stage, names_[m]);
          }
        } else {
          for (std::size_t row = 0; row < rows; ++row) {
            if (previous.IsNull(static_cast<int64_t>(row)) || nulls[leading][row]) {
              output.set_null(static_cast<int64_t>(row));
            } else {
              column[row] =
                  generated_number(previous.Value(static_cast<int64_t>(row)) * values[leading][row], stage, names_[m]);
            }
          }
        }
      } else if (monomials_[m].empty()) {
        // The fold below, without its per-row work: the bias has no factor
        // that could be null, and a lone factor is its column, every value
        // of which was checked finite on the way in.
        std::fill_n(column, rows, 1.0);
      } else if (monomials_[m].size() == 1 && !has_nulls[monomials_[m].front()]) {
        std::copy_n(values[monomials_[m].front()].get(), rows, column);
      } else {
        for (std::size_t row = 0; row < rows; ++row) {
          const std::optional<double> product = monomial_value(monomials_[m], [&](std::size_t c) {
            return nulls[c][row] ? std::optional<double>{} : std::optional<double>{values[c][row]};
          });
          if (product.has_value()) {
            column[row] = generated_number(*product, stage, names_[m]);
          } else {
            output.set_null(static_cast<int64_t>(row));
          }
        }
      }
      batch.names.push_back(names_[m]);
      batch.cols.push_back(output.finish());
    }
    return batch;
  }

  Row transform_row(Row row) const override {
    RecordLookup lookup(row, columns_.size());
    std::vector<std::optional<double>> values(columns_.size());
    for (std::size_t c = 0; c < columns_.size(); ++c) {
      const RowValue* value = lookup.find(columns_[c]);
      if (value == nullptr) throw Error(stage + ": key '" + columns_[c] + "' missing from record");
      values[c] = number_value(*value, stage, columns_[c]);
    }
    // The expanded keys leave first, so a key named like a generated column
    // is only a collision when it is not itself expanded.
    std::erase_if(row, [&](const auto& entry) { return expanded(entry.first); });
    for (const auto& [key, value] : row) {
      if (output_index_.contains(key)) throw Error(stage + ": key '" + key + "' already present in record");
    }
    row.reserve(row.size() + monomials_.size());
    for (std::size_t m = 0; m < monomials_.size(); ++m) {
      const std::optional<double> product = monomial_value(monomials_[m], [&](std::size_t c) { return values[c]; });
      row.emplace_back(names_[m], product.has_value() ? RowValue(generated_number(*product, stage, names_[m]))
                                                      : RowValue(std::monostate{}));
    }
    return row;
  }

 private:
  // One monomial: the column indices it multiplies, ascending, with a repeat
  // per exponent — sklearn's combinations_with_replacement tuple.
  using Monomial = std::vector<std::size_t>;

  void read_degree(const nlohmann::json& params) {
    if (!params.contains("degree") || params.at("degree").is_null()) return;
    const nlohmann::json& degree = params.at("degree");
    const std::string message =
        "polynomial_features: 'degree' must be an integer from 0 to 2^63 - 1, or a [min, max] pair of them with "
        "min <= max";
    if (is_int64(degree)) {
      if (degree.get<int64_t>() < 0) throw Error(message);
      max_degree_ = degree.get<int64_t>();
      return;
    }
    if (!degree.is_array() || degree.size() != 2) throw Error(message);
    for (const nlohmann::json& bound : degree) {
      if (!is_int64(bound) || bound.get<int64_t>() < 0) throw Error(message);
    }
    tuple_degree_ = true;
    min_degree_ = degree.at(0).get<int64_t>();
    max_degree_ = degree.at(1).get<int64_t>();
    if (min_degree_ > max_degree_) throw Error(message);
  }

  // sklearn's _combinations: the bias first when include_bias, then every
  // combination (with replacement unless interaction_only) of the column
  // indices, degree by degree.
  void build_monomials() {
    if (include_bias_) monomials_.emplace_back();
    const int64_t start = std::max<int64_t>(1, min_degree_);
    for (int64_t degree = start; degree <= max_degree_; ++degree) {
      // With interaction_only nothing above the column count can be picked,
      // so a huge degree stops here rather than looping over empty degrees.
      if (interaction_only_ && static_cast<std::size_t>(degree) > columns_.size()) break;
      combinations(static_cast<std::size_t>(degree));
    }
    names_.reserve(monomials_.size());
    for (const Monomial& monomial : monomials_) names_.push_back(monomial_name(monomial));
    reject_duplicate_outputs(names_, "polynomial_features");
    for (std::size_t m = 0; m < names_.size(); ++m) output_index_.emplace(names_[m], m);
    // Reuse only suffix products emitted earlier in output order, so each was
    // checked finite before this monomial reads it. A monomial whose suffix
    // is not emitted (a degree range that skips lower degrees) folds its own
    // factors, so no product the output does not hold can raise an error.
    suffixes_.assign(monomials_.size(), monomials_.size());
    for (std::size_t m = 0; m < monomials_.size(); ++m) {
      if (monomials_[m].size() < 2) continue;
      const Monomial suffix(monomials_[m].begin() + 1, monomials_[m].end());
      const auto found = output_index_.find(monomial_name(suffix));
      if (found != output_index_.end() && found->second < m && monomials_[found->second] == suffix) {
        suffixes_[m] = found->second;
      }
    }
  }

  // Every combination of `degree` column indices, in itertools order. Built
  // by hand rather than counted with a binomial: the width cap has to stop a
  // configuration that would generate millions of columns before it allocates
  // the list, and a binomial of a large degree overflows on its own.
  void combinations(std::size_t degree) {
    Monomial pick(degree, 0);
    const std::size_t columns = columns_.size();
    if (interaction_only_ && degree > columns) return;
    for (std::size_t i = 0; i < degree; ++i) pick[i] = interaction_only_ ? i : 0;
    while (true) {
      monomials_.push_back(pick);
      if (monomials_.size() > kMaxOutputs) {
        throw Error("polynomial_features: " + std::to_string(columns) + " columns of degree " +
                    std::to_string(max_degree_) + " would generate more than the " + std::to_string(kMaxOutputs) +
                    " columns allowed");
      }
      // Advance the rightmost index that still has room, then refill the ones
      // after it — itertools' own odometer.
      std::size_t position = degree;
      while (position > 0) {
        --position;
        const std::size_t limit = interaction_only_ ? columns - degree + position : columns - 1;
        if (pick[position] < limit) break;
        if (position == 0) return;
      }
      ++pick[position];
      for (std::size_t i = position + 1; i < degree; ++i)
        pick[i] = interaction_only_ ? pick[i - 1] + 1 : pick[position];
    }
  }

  // 'a*b' and 'a^2' where sklearn writes 'a b' and 'a^2'; '1' for the bias.
  std::string monomial_name(const Monomial& monomial) const {
    if (monomial.empty()) return "1";
    std::string name;
    for (std::size_t i = 0; i < monomial.size();) {
      std::size_t exponent = 1;
      while (i + exponent < monomial.size() && monomial[i + exponent] == monomial[i]) ++exponent;
      if (!name.empty()) name += "*";
      name += columns_[monomial[i]];
      if (exponent > 1) name += "^" + std::to_string(exponent);
      i += exponent;
    }
    return name;
  }

  // The product of one row's factors, or nullopt when any of them is null.
  // sklearn builds a degree-d column by multiplying the degree-(d-1) column
  // that drops the leading factor by that factor, which folds the monomial
  // from its last column to its first; floating multiplication is not
  // associative, so the order is the difference between agreeing bit for bit
  // and agreeing to a few ulp.
  template <typename At>
  static std::optional<double> monomial_value(const Monomial& monomial, At&& at) {
    if (monomial.empty()) return 1.0;
    std::size_t i = monomial.size() - 1;
    const std::optional<double> last = at(monomial[i]);
    if (!last.has_value()) return std::nullopt;
    double product = *last;
    while (i > 0) {
      --i;
      const std::optional<double> factor = at(monomial[i]);
      if (!factor.has_value()) return std::nullopt;
      product *= *factor;
    }
    return product;
  }

  bool expanded(const std::string& name) const { return is_expanded(columns_, name); }

  void drop_expanded(Batch& batch) const { drop_expanded_columns(columns_, batch); }

  std::vector<std::string> columns_;
  int64_t min_degree_ = 0;
  int64_t max_degree_ = 2;
  bool tuple_degree_ = false;
  bool interaction_only_ = false;
  bool include_bias_ = true;
  std::vector<Monomial> monomials_;
  std::vector<std::size_t> suffixes_;
  std::vector<std::string> names_;
  StringIndex output_index_;  // generated name -> its position, for record-path collisions
};

// --- the B-spline basis ----------------------------------------------------

// scipy's find_interval: the index with t[ell] <= x < t[ell+1], clamped to
// [k, n-1] when extrapolating, and -1 when x is outside the base interval and
// it is not.
int64_t find_interval(const std::vector<double>& knots, std::size_t degree, double x, bool extrapolate) {
  const std::size_t last = knots.size() - degree - 1;
  if ((x < knots[degree] || x > knots[last]) && !extrapolate) return -1;
  if (last - degree > 32 && !std::isnan(x)) {
    return static_cast<int64_t>(std::upper_bound(knots.begin() + static_cast<std::ptrdiff_t>(degree + 1),
                                                 knots.begin() + static_cast<std::ptrdiff_t>(last), x) -
                                knots.begin()) -
           1;
  }
  // Counting the interior knots at most x lands where a scan for the first
  // knot above x stops, as the knots are sorted, without the scan's exit
  // branch: a value anywhere in the base interval mispredicts it. NaN counts
  // none, as the scan stops at once.
  std::size_t ell = degree + 1;
  for (std::size_t i = degree + 1; i < last; ++i) ell += x >= knots[i] ? 1 : 0;
  return static_cast<int64_t>(ell) - 1;
}

// scipy's deBoor_D: the degree + 1 basis values (nu = 0) or derivatives
// (nu = 1) that are non-zero on interval `ell`, written into `values` as
// B_{ell-degree}, ..., B_{ell}. The Cox-de Boor recursion, with scipy's own
// grouping of the arithmetic. The scipy build this is checked against fuses
// the multiply-add that accumulates each value, where this build rounds
// twice, so from degree 2 the last bits can differ.
void de_boor(const std::vector<double>& knots, double x, std::size_t degree, std::size_t ell, std::size_t nu,
             std::vector<double>& values, std::vector<double>& scratch) {
  // Sized on first use rather than assigned on every call: assign() is an
  // out-of-line call, and this runs once per value. The value recursion
  // (nu = 0) writes every entry before it reads it, and so does every pass
  // over `scratch`. A derivative pass can leave an entry as it found it where
  // two knots coincide (scipy writes values[nu] there), so it still starts
  // from zeros.
  if (values.size() != degree + 1) values.assign(degree + 1, 0.0);
  if (scratch.size() != degree + 1) scratch.assign(degree + 1, 0.0);
  if (nu > 0) std::fill(values.begin(), values.end(), 0.0);
  values[0] = 1.0;
  for (std::size_t j = 1; j + nu <= degree; ++j) {  // degree - nu standard iterations
    for (std::size_t i = 0; i < j; ++i) scratch[i] = values[i];
    values[0] = 0.0;
    for (std::size_t n = 1; n <= j; ++n) {
      const double high = knots[ell + n];
      const double low = knots[ell + n - j];
      if (high == low) {
        values[n] = 0.0;
        continue;
      }
      const double weight = scratch[n - 1] / (high - low);
      values[n - 1] += weight * (high - x);
      values[n] = weight * (x - low);
    }
  }
  for (std::size_t j = degree - nu + 1; j <= degree; ++j) {  // nu derivative transformations
    for (std::size_t i = 0; i < j; ++i) scratch[i] = values[i];
    values[0] = 0.0;
    for (std::size_t n = 1; n <= j; ++n) {
      const double high = knots[ell + n];
      const double low = knots[ell + n - j];
      if (high == low) {
        values[nu] = 0.0;
        continue;
      }
      const double weight = static_cast<double>(j) * scratch[n - 1] / (high - low);
      values[n - 1] -= weight;
      values[n] = weight;
    }
  }
}

// Scratch a spline evaluation needs, so the row loop allocates nothing.
struct BasisScratch {
  std::vector<double> values;
  std::vector<double> work;
  std::vector<double> full;  // the unfolded basis, periodic extrapolation only
};

// Every basis value (or derivative) at x, or false when x is outside the base
// interval and extrapolate is off.
bool basis_at(const std::vector<double>& knots, std::size_t degree, double x, bool extrapolate, std::size_t nu,
              std::vector<double>& out, BasisScratch& scratch) {
  std::fill(out.begin(), out.end(), 0.0);
  const int64_t ell = find_interval(knots, degree, x, extrapolate);
  if (ell < 0) return false;
  de_boor(knots, x, degree, static_cast<std::size_t>(ell), nu, scratch.values, scratch.work);
  const auto base = static_cast<std::size_t>(ell) - degree;
  for (std::size_t i = 0; i <= degree; ++i) out[base + i] = scratch.values[i];
  return true;
}

// ---------------------------------------------------------------------------
// spline_transform — numeric columns -> a B-spline basis per column,
// replacing them.
// params: {"columns": [...], "n_knots": int, "degree": int,
//          "knots": "uniform" | "quantile",
//          "extrapolation": "error" | "constant" | "linear" | "continue"
//                           | "periodic",
//          "include_bias": bool, "handle_missing": "error" | "zeros"}
// Semantics:
//   * knots='uniform' puts n_knots knots at numpy's linspace(min, max) of the
//     column, a min/max monoid; knots='quantile' puts them at the percentiles
//     of the whole column, which is BUFFERED: each worker holds every value
//     of its row groups, as group_agg's median and quantile_encode do, until
//     finalize merges them and selects the order statistics the knots read
//   * degree knots are added beyond each end, spaced like the two knots at
//     that end ('periodic' continues the other end's spacing instead), and
//     the basis is the Cox-de Boor recursion over that vector, as scipy's
//     BSpline computes it
//   * each column becomes n_knots + degree - 1 columns named
//     '<column>_sp_<j>' (n_knots - 1 under extrapolation='periodic'), one
//     fewer when include_bias is false, which drops the last of them; the
//     width comes from the params alone, so a later step in the same pipeline
//     can name the outputs even though the knots are fitted
//   * extrapolation says what happens outside the fitted range: 'error'
//     refuses the value, 'constant' holds the boundary basis values, 'linear'
//     continues them along the boundary derivative, 'continue' evaluates the
//     boundary polynomial pieces on, and 'periodic' wraps the column onto the
//     interval between the first and last base knot
//   * handle_missing='error' refuses a null, at transform as well as at fit;
//     'zeros' encodes it as zeros in every generated column
//   * outputs are f64 whatever the inputs are (sklearn keeps a float32 input
//     float32)
// ---------------------------------------------------------------------------
class SplineTransform final : public Kernel {
 public:
  explicit SplineTransform(const nlohmann::json& params)
      : columns_(column_list(params, "columns", "spline_transform")) {
    n_knots_ = bounded_integer(params, "n_knots", 5, 2);
    degree_ = bounded_integer(params, "degree", 3, 0);
    if (params.contains("knots") && !params.at("knots").is_null()) {
      const nlohmann::json& knots = params.at("knots");
      if (!knots.is_string() || (knots.get<std::string>() != "uniform" && knots.get<std::string>() != "quantile")) {
        throw Error("spline_transform: 'knots' must be 'uniform' or 'quantile'");
      }
      by_quantile_ = knots.get<std::string>() == "quantile";
    }
    if (params.contains("extrapolation") && !params.at("extrapolation").is_null()) {
      const nlohmann::json& mode = params.at("extrapolation");
      if (!mode.is_string()) throw Error(kExtrapolationMessage);
      const std::string text = mode.get<std::string>();
      if (text == "error") {
        extrapolation_ = Extrapolation::Error;
      } else if (text == "constant") {
        extrapolation_ = Extrapolation::Constant;
      } else if (text == "linear") {
        extrapolation_ = Extrapolation::Linear;
      } else if (text == "continue") {
        extrapolation_ = Extrapolation::Continue;
      } else if (text == "periodic") {
        extrapolation_ = Extrapolation::Periodic;
      } else {
        throw Error(kExtrapolationMessage);
      }
    }
    if (params.contains("include_bias") && !params.at("include_bias").is_null()) {
      if (!params.at("include_bias").is_boolean())
        throw Error("spline_transform: 'include_bias' must be true or false");
      include_bias_ = params.at("include_bias").get<bool>();
    }
    if (params.contains("handle_missing") && !params.at("handle_missing").is_null()) {
      const nlohmann::json& missing = params.at("handle_missing");
      if (!missing.is_string() || (missing.get<std::string>() != "error" && missing.get<std::string>() != "zeros")) {
        throw Error("spline_transform: 'handle_missing' must be 'error' or 'zeros'");
      }
      zero_missing_ = missing.get<std::string>() == "zeros";
    }
    if (periodic() && n_knots_ <= degree_) {
      throw Error("spline_transform: extrapolation 'periodic' needs degree below n_knots, got n_knots " +
                  std::to_string(n_knots_) + " and degree " + std::to_string(degree_));
    }
    // The width comes from the params alone, so the outputs are named before
    // any data is read; only the knot positions are fitted.
    name_outputs();
    inputs = columns_;
    outputs = names_;
  }

  std::string id() const override { return "spline_transform"; }

  nlohmann::json params() const override {
    return {{"columns", columns_},
            {"n_knots", n_knots_},
            {"degree", degree_},
            {"knots", by_quantile_ ? "quantile" : "uniform"},
            {"extrapolation", extrapolation_name()},
            {"include_bias", include_bias_},
            {"handle_missing", zero_missing_ ? "zeros" : "error"}};
  }

  void resolve(Schema& schema) override {
    bind_numeric(schema, stage, columns_, true, false, "expand");
    std::erase_if(schema.cols, [&](const auto& col) { return expanded(col.first); });
    for (const std::string& name : names_) {
      if (schema.find(name) != nullptr) throw Error(stage + ": output column '" + name + "' already exists in input");
    }
    for (const std::string& name : names_) schema.cols.emplace_back(name, "f64");
    inputs = columns_;
    outputs = names_;
  }

  bool needs_fit() const override { return true; }

  struct State final : AccState {
    struct Column {
      double low = std::numeric_limits<double>::infinity();
      double high = -std::numeric_limits<double>::infinity();
      int64_t count = 0;
      int64_t nulls = 0;
      SpillVector<double> buffer;  // quantile knots only
    };
    std::unordered_map<std::string, Column> columns;
  };

  std::unique_ptr<AccState> make_state() const override { return std::make_unique<State>(); }

  void consume(AccState& raw, const Batch& batch) const override {
    auto& state = static_cast<State&>(raw);
    for (const std::string& name : columns_) {
      State::Column& column = state.columns[name];
      const arrow::Array& array = column_array(batch, stage, name);
      for_each_number(array, stage, name, [&](int64_t, bool is_null, double value) {
        if (is_null) {
          ++column.nulls;
          return;
        }
        ++column.count;
        // One zero: min() and max() return their first argument on a tie and
        // std::sort is not stable, so -0.0 would make both the extremes and
        // the ends of the sorted buffer depend on the row-group split. Every
        // reader treats -0.0 and 0.0 alike, so only the artifact bytes move.
        if (value == 0.0) value = 0.0;
        column.low = std::min(column.low, value);
        column.high = std::max(column.high, value);
        if (by_quantile_) column.buffer.push_back(value);
      });
    }
  }

  void finalize(std::vector<std::unique_ptr<AccState>> states) override {
    // Adopt the first worker's state rather than copying into a fresh one:
    // the largest buffer is then never duplicated, which is one whole extra
    // O(rows) buffer at the peak of the fit.
    std::unique_ptr<State> owned;
    for (auto& raw : states) {
      if (raw == nullptr) continue;
      if (owned == nullptr) {
        owned.reset(static_cast<State*>(raw.release()));
        continue;
      }
      auto& state = static_cast<State&>(*raw);
      for (auto& [name, column] : state.columns) {
        State::Column& into = owned->columns[name];
        into.low = std::min(into.low, column.low);
        into.high = std::max(into.high, column.high);
        into.count += column.count;
        into.nulls += column.nulls;
        into.buffer.append(std::move(column.buffer));
      }
      raw.reset();
    }
    if (owned == nullptr) owned = std::make_unique<State>();
    State& merged = *owned;

    splines_.clear();
    for (const std::string& name : columns_) {
      State::Column& column = merged.columns[name];
      if (!zero_missing_ && column.nulls > 0) {
        throw Error(stage + ": column '" + name + "' has nulls and handle_missing is 'error'");
      }
      // sklearn's fit takes at least two samples, and one value would put
      // every knot in the same place and encode the whole column as zeros.
      if (column.count < 2) {
        throw Error(stage + ": column '" + name + "' has fewer than two non-null values to place knots between");
      }
      Spline spline;
      if (by_quantile_) {
        const auto references = linspace(0.0, 1.0, static_cast<std::size_t>(n_knots_));
        std::vector<std::size_t> ranks;
        ranks.reserve(references.size() * 2);
        for (double rank : references) linear_quantile_ranks(column.buffer.size(), 100.0 * rank / 100.0, ranks);
        select_numeric_ranks(column.buffer, std::move(ranks));
        spline.base.reserve(static_cast<std::size_t>(n_knots_));
        for (double rank : references) {
          spline.base.push_back(percentile_of(column.buffer, 100.0 * rank));
        }
        column.buffer.release();
      } else {
        spline.base = linspace(column.low, column.high, static_cast<std::size_t>(n_knots_));
      }
      for (double knot : spline.base) serializable(knot, stage, name);
      // Over a span of a few subnormal units, linspace's step can round up
      // enough to put the second-to-last knot past the maximum, which
      // linspace then writes as the last knot, so the knots decrease. scipy's
      // BSpline refuses such knots, so upstream's fit fails; so does this one,
      // which would otherwise save a table load_state refuses.
      if (!std::is_sorted(spline.base.begin(), spline.base.end())) {
        throw Error(stage + ": column '" + name + "' spans too little to place " + std::to_string(n_knots_) +
                    " knots in order");
      }
      // From degree 1 up, de Boor's recursion divides by the gaps between
      // knots, and under about 5.6e-309, 1 / f64's largest, that overflows.
      // Uniform knots all share one gap, so then every row, the fitted ones
      // included, would transform to an infinity or NaN. Quantile knots can
      // put one tiny gap among wide ones, which fails only the rows beside it,
      // at transform, as in sklearn; a degree-0 basis divides by nothing.
      for (std::size_t i = 1; !by_quantile_ && degree_ > 0 && i < spline.base.size(); ++i) {
        const double gap = spline.base[i] - spline.base[i - 1];
        if (gap > 0.0 && std::isinf(1.0 / gap)) {
          throw Error(stage + ": column '" + name + "' spans too little to place " + std::to_string(n_knots_) +
                      " knots at least about 5.6e-309 apart");
        }
      }
      splines_.emplace(name, std::move(spline));
    }
    build_splines();
    fitted = true;
  }

  nlohmann::json state() const override {
    nlohmann::json knots = nlohmann::json::object();
    for (const std::string& name : columns_) knots[name] = splines_.at(name).base;
    return {{"knots", knots}};
  }

  void load_state(const nlohmann::json& state) override {
    if (!state.is_object() || !state.contains("knots") || !state.at("knots").is_object()) {
      throw Error(stage + ": state must hold a 'knots' object");
    }
    const nlohmann::json& knots = state.at("knots");
    splines_.clear();
    for (const std::string& name : columns_) {
      const std::string where = stage + ": knots['" + name + "']";
      if (!knots.contains(name) || !knots.at(name).is_array() ||
          knots.at(name).size() != static_cast<std::size_t>(n_knots_)) {
        throw Error(where + " must be a list of " + std::to_string(n_knots_) + " knots");
      }
      Spline spline;
      for (const nlohmann::json& knot : knots.at(name)) spline.base.push_back(manifest_number(knot, where));
      if (!std::is_sorted(spline.base.begin(), spline.base.end())) throw Error(where + " must be sorted");
      splines_.emplace(name, std::move(spline));
    }
    build_splines();
    inputs = columns_;
    outputs = names_;
    fitted = true;
  }

  Batch transform(Batch batch) const override {
    const std::size_t width = generated_per_column();
    const auto rows = static_cast<std::size_t>(column_array(batch, stage, columns_.front()).length());
    std::vector<PrimitiveOutput<arrow::DoubleType>> generated;
    generated.reserve(names_.size());
    for (std::size_t block_index = 0; block_index < columns_.size(); ++block_index) {
      const std::string& name = columns_[block_index];
      const Spline& spline = splines_.at(name);
      for (std::size_t j = 0; j < width; ++j) {
        generated.emplace_back(static_cast<int64_t>(rows), stage + ": column '" + names_[block_index * width + j] + "'",
                               true);
      }
      std::vector<double> row_values(spline.basis_width, 0.0);
      BasisScratch scratch;
      for_each_number(column_array(batch, stage, name), stage, name, [&](int64_t row, bool is_null, double value) {
        const auto slot = static_cast<std::size_t>(row);
        if (is_null) {
          refuse_missing(name);
          return;  // handle_missing='zeros': every generated column stays 0
        }
        // A non-periodic basis inside the base interval, or anywhere under
        // 'continue', has only degree + 1 nonzero entries. The output columns
        // start at zero, so scatter those entries without clearing or copying
        // a whole dense basis for every row.
        if (!periodic() && ((value >= spline.base.front() && value <= spline.base.back()) ||
                            extrapolation_ == Extrapolation::Continue)) {
          const auto degree = static_cast<std::size_t>(degree_);
          const int64_t ell = find_interval(spline.knots, degree, value, extrapolation_ == Extrapolation::Continue);
          if (ell >= 0) {
            de_boor(spline.knots, value, degree, static_cast<std::size_t>(ell), 0, scratch.values, scratch.work);
            const std::size_t first = static_cast<std::size_t>(ell) - degree;
            for (std::size_t i = 0; i <= degree && first + i < width; ++i) {
              const std::size_t output = block_index * width + first + i;
              generated[output].mutable_values()[slot] = generated_number(scratch.values[i], stage, names_[output]);
            }
            return;
          }
        }
        evaluate(name, spline, value, row_values, scratch);
        for (std::size_t j = 0; j < width; ++j) {
          const std::size_t output = block_index * width + j;
          generated[output].mutable_values()[slot] = generated_number(row_values[j], stage, names_[output]);
        }
      });
    }
    drop_expanded(batch);
    // A generated column is never null: a missing value is zeros or an error.
    for (std::size_t c = 0; c < columns_.size(); ++c) {
      for (std::size_t j = 0; j < width; ++j) {
        const std::string& name = names_[c * width + j];
        batch.names.push_back(name);
        batch.cols.push_back(generated[c * width + j].finish());
      }
    }
    return batch;
  }

  Row transform_row(Row row) const override {
    RecordLookup lookup(row, columns_.size());
    const std::size_t width = generated_per_column();
    std::vector<std::optional<double>> values(columns_.size());
    for (std::size_t c = 0; c < columns_.size(); ++c) {
      const RowValue* value = lookup.find(columns_[c]);
      if (value == nullptr) throw Error(stage + ": key '" + columns_[c] + "' missing from record");
      values[c] = number_value(*value, stage, columns_[c]);
    }
    std::erase_if(row, [&](const auto& entry) { return expanded(entry.first); });
    for (const auto& [key, value] : row) {
      if (output_index_.contains(key)) throw Error(stage + ": key '" + key + "' already present in record");
    }
    row.reserve(row.size() + columns_.size() * width);
    BasisScratch scratch;
    for (std::size_t c = 0; c < columns_.size(); ++c) {
      const Spline& spline = splines_.at(columns_[c]);
      std::vector<double> row_values(spline.basis_width, 0.0);
      if (values[c].has_value()) {
        evaluate(columns_[c], spline, *values[c], row_values, scratch);
      } else {
        refuse_missing(columns_[c]);  // handle_missing='zeros': every generated column is 0
      }
      for (std::size_t j = 0; j < width; ++j) {
        row.emplace_back(names_[c * width + j], generated_number(row_values[j], stage, names_[c * width + j]));
      }
    }
    return row;
  }

 private:
  enum class Extrapolation { Error, Constant, Linear, Continue, Periodic };

  // One fitted column: the base knots, the padded knot vector the recursion
  // runs over, and the boundary values the two extrapolations need.
  struct Spline {
    std::vector<double> base;
    std::vector<double> knots;
    std::vector<double> low_values;   // the basis at the first base knot
    std::vector<double> high_values;  // the basis at the last base knot
    std::vector<double> low_slopes;   // its derivative there
    std::vector<double> high_slopes;
    std::size_t basis_width = 0;  // knots.size() - degree - 1
  };

  static constexpr const char* kExtrapolationMessage =
      "spline_transform: 'extrapolation' must be 'error', 'constant', 'linear', 'continue' or 'periodic'";

  // Both counts are bounded above by the output cap: past it, only an n_knots
  // one or two over could still generate a legal width, and the bound is also
  // what keeps n_knots + degree from leaving int64 on its way to the width
  // check.
  static int64_t bounded_integer(const nlohmann::json& params, const char* key, int64_t fallback, int64_t least) {
    if (!params.contains(key) || params.at(key).is_null()) return fallback;
    const nlohmann::json& value = params.at(key);
    const auto most = static_cast<int64_t>(kMaxOutputs);
    if (!value.is_number_integer() ||
        (value.is_number_unsigned() && value.get<uint64_t>() > static_cast<uint64_t>(most)) ||
        value.get<int64_t>() < least || value.get<int64_t>() > most) {
      throw Error("spline_transform: '" + std::string(key) + "' must be an integer between " + std::to_string(least) +
                  " and " + std::to_string(most));
    }
    return value.get<int64_t>();
  }

  bool periodic() const { return extrapolation_ == Extrapolation::Periodic; }

  const char* extrapolation_name() const {
    switch (extrapolation_) {
      case Extrapolation::Error:
        return "error";
      case Extrapolation::Constant:
        return "constant";
      case Extrapolation::Linear:
        return "linear";
      case Extrapolation::Continue:
        return "continue";
      case Extrapolation::Periodic:
        break;
    }
    return "periodic";
  }

  // n_knots + degree - 1 basis functions, or n_knots - 1 when periodic: the
  // periodic basis folds the last `degree` of them onto the first.
  std::size_t splines_per_column() const {
    return static_cast<std::size_t>(periodic() ? n_knots_ - 1 : n_knots_ + degree_ - 1);
  }

  // sklearn drops the last basis function per column without a bias.
  std::size_t generated_per_column() const { return splines_per_column() - (include_bias_ ? 0 : 1); }

  void name_outputs() {
    const std::size_t width = generated_per_column();
    // Without a bias, n_knots 2 with degree 0, or with degree 1 and periodic
    // extrapolation, leaves nothing to write: the fit would succeed, the
    // manifest would be written, and the batch writer would refuse a frame of
    // no columns at serving time. Refuse the plan instead.
    if (width == 0) {
      throw Error("spline_transform: n_knots " + std::to_string(n_knots_) + ", degree " + std::to_string(degree_) +
                  ", extrapolation '" + extrapolation_name() + "' and include_bias " +
                  (include_bias_ ? "true" : "false") + " generate no columns at all");
    }
    if (width * columns_.size() > kMaxOutputs) {
      throw Error("spline_transform: " + std::to_string(columns_.size()) + " columns of " + std::to_string(width) +
                  " splines would generate more than the " + std::to_string(kMaxOutputs) + " columns allowed");
    }
    names_.clear();
    output_index_.clear();
    for (const std::string& name : columns_) {
      for (std::size_t j = 0; j < width; ++j) {
        output_index_.emplace(name + "_sp_" + std::to_string(j), names_.size());
        names_.push_back(name + "_sp_" + std::to_string(j));
      }
    }
    reject_duplicate_outputs(names_, "spline_transform");
  }

  // The padded knot vector and the boundary values, for every fitted column.
  void build_splines() {
    const auto degree = static_cast<std::size_t>(degree_);
    for (const std::string& name : columns_) {
      Spline& spline = splines_.at(name);
      spline.knots = pad_knots(spline.base);
      for (double knot : spline.knots) {
        if (!std::isfinite(knot)) {
          throw Error(stage + ": column '" + name + "' has knots so far apart that padding them leaves f64's range");
        }
      }
      spline.basis_width = spline.knots.size() - degree - 1;
      BasisScratch scratch;
      spline.low_values.assign(spline.basis_width, 0.0);
      spline.high_values.assign(spline.basis_width, 0.0);
      spline.low_slopes.assign(spline.basis_width, 0.0);
      spline.high_slopes.assign(spline.basis_width, 0.0);
      const double low = spline.base.front();
      const double high = spline.base.back();
      (void)basis_at(spline.knots, degree, low, false, 0, spline.low_values, scratch);
      (void)basis_at(spline.knots, degree, high, false, 0, spline.high_values, scratch);
      (void)basis_at(spline.knots, degree, low, false, 1, spline.low_slopes, scratch);
      (void)basis_at(spline.knots, degree, high, false, 1, spline.high_slopes, scratch);
    }
  }

  // sklearn adds `degree` knots beyond each end. A periodic basis continues
  // the spacing of the far end so the two ends meet; every other one reuses
  // the spacing of the two knots at that end (Eilers & Marx, rather than
  // repeating the boundary knot).
  std::vector<double> pad_knots(const std::vector<double>& base) const {
    const auto degree = static_cast<std::size_t>(degree_);
    std::vector<double> knots;
    if (periodic()) {
      const double period = base.back() - base.front();
      for (std::size_t i = base.size() - degree - 1; i + 1 < base.size(); ++i) knots.push_back(base[i] - period);
      knots.insert(knots.end(), base.begin(), base.end());
      for (std::size_t i = 1; i <= degree; ++i) knots.push_back(base[i] + period);
      return knots;
    }
    if (degree == 0) return base;
    const double low_step = base[1] - base[0];
    const double high_step = base[base.size() - 1] - base[base.size() - 2];
    const auto span = static_cast<double>(degree);
    knots = linspace(base.front() - span * low_step, base.front() - low_step, degree);
    knots.insert(knots.end(), base.begin(), base.end());
    const std::vector<double> above = linspace(base.back() + high_step, base.back() + span * high_step, degree);
    knots.insert(knots.end(), above.begin(), above.end());
    return knots;
  }

  void refuse_missing(const std::string& name) const {
    if (!zero_missing_) throw Error(stage + ": column '" + name + "' has a null and handle_missing is 'error'");
  }

  // One value's generated columns, sklearn's transform branch by branch.
  void evaluate(const std::string& name, const Spline& spline, double value, std::vector<double>& out,
                BasisScratch& scratch) const {
    const auto degree = static_cast<std::size_t>(degree_);
    const std::size_t width = splines_per_column();
    const double low = spline.base.front();
    const double high = spline.base.back();
    if (periodic()) {
      // The value maps onto [first base knot, last base knot) before it is
      // evaluated, and the last `degree` basis functions fold onto the first.
      // A column whose values were all equal has no period and reads as a
      // constant.
      const double period = high - low;
      double folded = 0.0;
      if (period > 0.0) {
        double rest = std::fmod(value - low, period);
        if (rest != 0.0 && (rest < 0.0) != (period < 0.0)) rest += period;  // numpy's floor-mod, not fmod's
        folded = low + rest;
      }
      scratch.full.assign(spline.basis_width, 0.0);
      (void)basis_at(spline.knots, degree, folded, true, 0, scratch.full, scratch);
      std::fill(out.begin(), out.end(), 0.0);
      for (std::size_t j = 0; j < width; ++j) out[j] = scratch.full[j];
      for (std::size_t j = 0; j < degree; ++j) out[j] += scratch.full[width + j];
      return;
    }
    if (extrapolation_ == Extrapolation::Continue) {
      (void)basis_at(spline.knots, degree, value, true, 0, out, scratch);
      return;
    }
    if (extrapolation_ == Extrapolation::Error) {
      if (!basis_at(spline.knots, degree, value, false, 0, out, scratch)) {
        throw Error(stage + ": column '" + name + "': value " + std::format("{}", value) +
                    " is beyond the limits of the knots and extrapolation is 'error'");
      }
      return;
    }
    if (value >= low && value <= high) {
      (void)basis_at(spline.knots, degree, value, false, 0, out, scratch);
      return;
    }
    std::fill(out.begin(), out.end(), 0.0);
    if (extrapolation_ == Extrapolation::Constant) {
      // Only the first and last `degree` basis functions are non-zero at the
      // boundary, so only those carry over.
      if (value < low) {
        for (std::size_t j = 0; j < degree; ++j) out[j] = spline.low_values[j];
      } else {
        for (std::size_t j = 0; j < degree; ++j) out[width - degree + j] = spline.high_values[width - degree + j];
      }
      return;
    }
    // 'linear': continue each boundary basis function along its own slope.
    // At degree 1 the second basis function still has a slope at the
    // boundary, so sklearn widens the reach by one there, and at degree 0 too.
    const std::size_t reach = degree <= 1 ? degree + 1 : degree;
    for (std::size_t j = 0; j < reach; ++j) {
      if (value < low) {
        out[j] = spline.low_values[j] + (value - low) * spline.low_slopes[j];
      } else {
        const std::size_t index = width - 1 - j;
        out[index] = spline.high_values[index] + (value - high) * spline.high_slopes[index];
      }
    }
  }

  bool expanded(const std::string& name) const { return is_expanded(columns_, name); }

  void drop_expanded(Batch& batch) const { drop_expanded_columns(columns_, batch); }

  std::vector<std::string> columns_;
  int64_t n_knots_ = 5;
  int64_t degree_ = 3;
  bool by_quantile_ = false;
  Extrapolation extrapolation_ = Extrapolation::Constant;
  bool include_bias_ = true;
  bool zero_missing_ = false;
  std::vector<std::string> names_;
  StringIndex output_index_;  // generated name -> its position, for record-path collisions
  std::unordered_map<std::string, Spline> splines_;
};

// --- the power-transform likelihood and its solver -------------------------

constexpr double kEps = 2.220446049250313e-16;  // f64 epsilon, scipy's np.finfo(float64).eps
constexpr double kSmallestNormal = 2.2250738585072014e-308;
constexpr double kLargest = 1.7976931348623157e308;

// scipy.special.logsumexp: log(sum(exp(a))), shifted by the largest term.
double log_sum_exp(const SpillVector<double>& values, const std::string& where) {
  double high = values.front();
  values.for_each_chunk([&](std::span<const double> chunk) {
    for (double value : chunk) high = std::max(high, value);
  });
  if (!std::isfinite(high)) high = 0.0;
  const double total = exact_sum(values, where, [&](double value) { return std::exp(value - high); });
  return std::log(total) + high;
}

// scipy.stats._morestats._log_var: log of the variance of exp(logx), taken in
// log space so a large lambda cannot overflow the powers themselves.
//
// `logx` is rewritten in place with the squared log deviations rather than
// allocating a second vector: this runs once per objective evaluation inside
// the solver, and every caller passes a scratch buffer it does not read again.
double log_variance(SpillVector<double>& logx, const std::string& where) {
  const auto count = static_cast<double>(logx.size());
  const double log_mean = log_sum_exp(logx, where) - std::log(count);
  logx.rewrite([&](std::size_t, double value) {
    double high = std::max(value, log_mean);
    double difference;
    if (std::isfinite(high)) {
      // The larger operand subtracts itself: exp(+/-0) is exactly one, so
      // each branch is exactly exp(value - high) - exp(log_mean - high),
      // which the fallback below computes when high is not finite.
      difference = value < log_mean ? std::exp(value - high) - 1.0 : 1.0 - std::exp(log_mean - high);
    } else {
      high = 0.0;
      difference = std::exp(value - high) - std::exp(log_mean - high);
    }
    return difference == 0.0 ? -std::numeric_limits<double>::infinity()
                             : 2.0 * (std::log(std::fabs(difference)) + high);
  });
  return log_sum_exp(logx, where) - std::log(count);
}

// numpy's var: the population variance, mean first and then the squares.
double numpy_variance(const SpillVector<double>& values, const std::string& where) {
  const auto count = static_cast<double>(values.size());
  const double mean = exact_sum(values, where, [](double value) { return value; }) / count;
  return exact_sum(values, where, [&](double value) { return (value - mean) * (value - mean); }) / count;
}

// sklearn's _is_constant_feature: the error bound of the two-pass variance.
bool constant_feature(double variance, double mean, double count) {
  const double bound = count * kEps * variance + (count * mean * kEps) * (count * mean * kEps);
  return variance <= bound;
}

double yeo_johnson_logged(double x, double logarithm, double lambda) {
  if (x >= 0.0) {
    if (std::fabs(lambda) < kEps) return logarithm;
    return std::expm1(lambda * logarithm) / lambda;
  }
  if (std::fabs(lambda - 2.0) > kEps) return -std::expm1((2.0 - lambda) * logarithm) / (2.0 - lambda);
  return -logarithm;
}

double yeo_johnson_value(double x, double lambda) {
  return yeo_johnson_logged(x, std::log1p(x >= 0.0 ? x : -x), lambda);
}

// scipy.special.boxcox, branch for branch. A lambda too small to matter gives
// log(x). From lambda * log(x) = 709.78 up, where expm1 nears overflow, the
// value is x^lambda / lambda - 1 / lambda with the quotient taken through
// logarithms, so it stays finite whenever the quotient fits in f64 although
// the power doesn't. scipy's ARM64 build fuses that exponent's multiply and
// subtract, where this build rounds both, so there the two can differ by
// about 1e-13 relative.
double box_cox_value(double x, double lambda) {
  if (std::fabs(lambda) < 1e-19) return std::log(x);
  const double power = lambda * std::log(x);
  if (power < 709.78) return std::expm1(power) / lambda;
  return std::copysign(1.0, lambda) * std::exp(power - std::log(std::fabs(lambda))) - 1.0 / lambda;
}

// One buffered column plus the parts of the profile log-likelihood that do
// not depend on lambda. scipy maximizes this; the solver minimizes its
// negation.
class Likelihood {
 public:
  Likelihood(const SpillVector<double>& values, bool box_cox, const std::string& where)
      : count_(values.size()), box_cox_(box_cox), where_(where) {
    values.for_each([&](std::size_t, double value) {
      all_positive_ = all_positive_ && value >= 0.0;
      all_negative_ = all_negative_ && value < 0.0;
    });
    if (box_cox_) {
      logs_.rewrite_from(values, [](std::size_t, double value) { return std::log(value); });
      total_ = exact_sum(logs_, where_, [](double value) { return value; });
      return;
    }
    // Cache the exact Jacobian terms, retaining the input sign. Their sign
    // also identifies the Yeo-Johnson branch, so each objective needs one
    // sequential source pass instead of zipping two spilled files. Negative
    // zero still takes the nonnegative branch and retains log1p(-0.0).
    logs_.rewrite_from(values, [](std::size_t, double value) {
      return std::copysign(std::log1p(value >= 0.0 ? value : -value), value);
    });
    total_ = exact_sum(logs_, where_, [](double value) { return value; });
  }

  double negative(double lambda) {
    const double value = box_cox_ ? box_cox_llf(lambda) : yeo_johnson_llf(lambda);
    // scipy's yeojohnson_normmax rejects an infinite likelihood outright: a
    // transformed column of near-zero variance would otherwise win.
    if (!box_cox_ && std::isinf(value)) return std::numeric_limits<double>::infinity();
    return -value;
  }

 private:
  double box_cox_llf(double lambda) {
    const auto count = static_cast<double>(count_);
    double log_var = 0.0;
    if (lambda == 0.0) {
      log_var = std::log(numpy_variance(logs_, where_));
    } else {
      work_.rewrite_from(logs_, [&](std::size_t, double value) { return lambda * value; });
      log_var = log_variance(work_, where_) - 2.0 * std::log(std::fabs(lambda));
    }
    return (lambda - 1.0) * total_ - count / 2.0 * log_var;
  }

  double yeo_johnson_llf(double lambda) {
    const auto count = static_cast<double>(count_);
    SpillVector<double>& work = work_;
    double log_var = 0.0;
    // A column with no negative value, or with only negative ones, takes the
    // log-space branch, which is where scipy avoids the overflow of a large
    // lambda; a mixed column has to transform the values themselves.
    if (all_positive_) {
      if (std::fabs(lambda) < kEps) {
        log_var = std::log(numpy_variance(logs_, where_));
      } else {
        work.rewrite_from(logs_, [&](std::size_t, double value) { return value * lambda; });
        log_var = log_variance(work, where_) - 2.0 * std::log(std::fabs(lambda));
      }
    } else if (all_negative_) {
      if (std::fabs(lambda - 2.0) < kEps) {
        work.rewrite_from(logs_, [](std::size_t, double value) { return -value; });
        log_var = std::log(numpy_variance(work, where_));
      } else {
        work.rewrite_from(logs_, [&](std::size_t, double value) { return -value * (2.0 - lambda); });
        log_var = log_variance(work, where_) - 2.0 * std::log(std::fabs(2.0 - lambda));
      }
    } else {
      work.rewrite_from(logs_, [&](std::size_t, double value) {
        return yeo_johnson_logged(value, value < 0.0 ? -value : value, lambda);
      });
      const double sigma = numpy_variance(work, where_);
      log_var = sigma >= kSmallestNormal ? std::log(sigma) : -std::numeric_limits<double>::infinity();
    }
    return -count / 2.0 * log_var + (lambda - 1.0) * total_;
  }

  std::size_t count_ = 0;
  SpillVector<double> logs_;  // log(x), or copysign(log1p(|x|), x)
  SpillVector<double> work_;  // bounded scratch storage, reused by every evaluation
  double total_ = 0.0;        // sum of log(x), or of sign(x) * log1p(|x|)
  bool box_cox_ = false;
  bool all_positive_ = true;
  bool all_negative_ = true;
  const std::string& where_;
};

// scipy.optimize.bracket, iteration for iteration: three points around the
// minimum, downhill from the two the caller gave.
struct Bracket {
  double low = 0.0;
  double middle = 0.0;
  double high = 0.0;
  double f_low = 0.0;
  double f_middle = 0.0;
  double f_high = 0.0;
};

template <typename F>
Bracket bracket_minimum(F&& func, double xa, double xb, const std::string& where) {
  constexpr double kGold = 1.618034;
  constexpr double kVerySmall = 1e-21;
  constexpr double kGrowLimit = 110.0;
  double fa = func(xa);
  double fb = func(xb);
  if (fa < fb) {
    std::swap(xa, xb);
    std::swap(fa, fb);
  }
  double xc = xb + kGold * (xb - xa);
  double fc = func(xc);
  for (int iteration = 0; fc < fb; ++iteration) {
    const double first = (xb - xa) * (fb - fc);
    const double second = (xb - xc) * (fb - fa);
    const double value = second - first;
    const double denominator = std::fabs(value) < kVerySmall ? 2.0 * kVerySmall : 2.0 * value;
    double w = xb - ((xb - xc) * second - (xb - xa) * first) / denominator;
    const double limit = xb + kGrowLimit * (xc - xb);
    if (iteration > 1000) throw Error(where + ": the likelihood has no bracketed minimum");
    double fw = 0.0;
    if ((w - xc) * (xb - w) > 0.0) {
      fw = func(w);
      if (fw < fc) {
        xa = xb;
        fa = fb;
        xb = w;
        fb = fw;
        break;
      }
      if (fw > fb) {
        xc = w;
        fc = fw;
        break;
      }
      w = xc + kGold * (xc - xb);
      fw = func(w);
    } else if ((w - limit) * (limit - xc) >= 0.0) {
      w = limit;
      fw = func(w);
    } else if ((w - limit) * (xc - w) > 0.0) {
      fw = func(w);
      if (fw < fc) {
        xb = xc;
        fb = fc;
        xc = w;
        fc = fw;
        w = xc + kGold * (xc - xb);
        fw = func(w);
      }
    } else {
      w = xc + kGold * (xc - xb);
      fw = func(w);
    }
    xa = xb;
    fa = fb;
    xb = xc;
    fb = fc;
    xc = w;
    fc = fw;
  }
  const bool ordered = (xa < xb && xb < xc) || (xc < xb && xb < xa);
  const bool lowest = (fb < fc && fb <= fa) || (fb < fa && fb <= fc);
  if (!ordered || !lowest || !std::isfinite(xa) || !std::isfinite(xb) || !std::isfinite(xc)) {
    throw Error(where + ": the likelihood has no bracketed minimum");
  }
  return {xa, xb, xc, fa, fb, fc};
}

// scipy.optimize.brent: golden section with parabolic interpolation over the
// bracket, at scipy's own tolerance. Box-Cox's lambda comes out of this.
template <typename F>
double brent_minimize(F&& func, double xa, double xb, const std::string& where) {
  constexpr double kTol = 1.48e-8;
  constexpr double kMinTol = 1.0e-11;
  constexpr double kGolden = 0.3819660;
  const Bracket bracket = bracket_minimum(func, xa, xb, where);
  double x = bracket.middle;
  double w = x;
  double v = x;
  double fx = bracket.f_middle;
  double fw = fx;
  double fv = fx;
  double a = std::min(bracket.low, bracket.high);
  double b = std::max(bracket.low, bracket.high);
  double delta = 0.0;
  double rat = 0.0;
  for (int iteration = 0; iteration < 500; ++iteration) {
    const double tol1 = kTol * std::fabs(x) + kMinTol;
    const double tol2 = 2.0 * tol1;
    const double middle = 0.5 * (a + b);
    if (std::fabs(x - middle) < (tol2 - 0.5 * (b - a))) break;
    if (std::fabs(delta) <= tol1) {
      delta = x >= middle ? a - x : b - x;
      rat = kGolden * delta;
    } else {
      const double first = (x - w) * (fx - fv);
      double second = (x - v) * (fx - fw);
      double p = (x - v) * second - (x - w) * first;
      second = 2.0 * (second - first);
      if (second > 0.0) p = -p;
      second = std::fabs(second);
      const double previous = delta;
      delta = rat;
      if (p > second * (a - x) && p < second * (b - x) && std::fabs(p) < std::fabs(0.5 * second * previous)) {
        rat = p / second;
        const double candidate = x + rat;
        if ((candidate - a) < tol2 || (b - candidate) < tol2) rat = middle - x >= 0 ? tol1 : -tol1;
      } else {
        delta = x >= middle ? a - x : b - x;
        rat = kGolden * delta;
      }
    }
    double u = 0.0;
    if (std::fabs(rat) < tol1) {
      u = rat >= 0 ? x + tol1 : x - tol1;
    } else {
      u = x + rat;
    }
    const double fu = func(u);
    if (fu > fx) {
      if (u < x) {
        a = u;
      } else {
        b = u;
      }
      if (fu <= fw || w == x) {
        v = w;
        fv = fw;
        w = u;
        fw = fu;
      } else if (fu <= fv || v == x || v == w) {
        v = u;
        fv = fu;
      }
    } else {
      if (u >= x) {
        a = x;
      } else {
        b = x;
      }
      v = w;
      fv = fw;
      w = x;
      fw = fx;
      x = u;
      fx = fu;
    }
  }
  return x;
}

// scipy.special.lambertw on its lower branch (k=-1), for z in [-1/e, 0):
// Halley's iteration from log(-z), stopping once a step moves w by at most
// scipy's tolerance of 1e-8 relative. scipy iterates in complex arithmetic,
// whose imaginary parts stay zero for such z. Its macOS build divides those
// complex numbers with a scaling that rounds differently from a real
// division, so the two can part in the last bit, and by more where z is
// subnormal (the column's most extreme value within about 0.04% of 1).
double lambert_w_lower(double z) {
  constexpr double kTol = 1e-8;
  double w = std::log(-z);
  for (int iteration = 0; iteration < 100; ++iteration) {
    const double ew = std::exp(w);
    const double wew = w * ew;
    const double wewz = wew - z;
    const double next = w - wewz / (wew + ew - (w + 2.0) * wewz / (2.0 * w + 2.0));
    if (std::fabs(next - w) <= kTol * std::fabs(next)) return next;
    w = next;
  }
  return std::numeric_limits<double>::quiet_NaN();
}

// scipy's boxcox_normmax caps the lambda its solver found, so that the
// column's most extreme value transforms to at most f64's largest over 10,000
// in magnitude (scipy's safety margin). The most extreme value is the largest
// when every value is at least 1, the smallest when every value is at most 1,
// and otherwise whichever transforms further from 0. Past the limit, lambda
// becomes the one that maps that value exactly onto it: (x^l - 1) / l = y
// solves in closed form with Lambert W's lower branch.
double cap_box_cox_lambda(double lambda, double smallest, double largest, const std::string& where) {
  constexpr double kLimit = kLargest / 10000.0;
  double extreme = 0.0;
  if (smallest >= 1.0) {
    extreme = largest;
  } else if (largest <= 1.0) {
    extreme = smallest;
  } else {
    extreme = box_cox_value(largest, lambda) > std::fabs(box_cox_value(smallest, lambda)) ? largest : smallest;
  }
  if (std::fabs(box_cox_value(extreme, lambda)) <= kLimit) return lambda;
  const double bound = extreme > 1.0 ? kLimit : -kLimit;
  const double logarithm = std::log(extreme);
  const double w = lambert_w_lower(-std::pow(extreme, -1.0 / bound) * logarithm / bound);
  // numpy divides the complex W by the real logarithm as a product with the
  // reciprocal, which can round differently from a division.
  const double capped = -w * (1.0 / logarithm) - 1.0 / bound;
  // The iteration, scipy's too, fails to converge when the most extreme value
  // lies within about 1e-11 of 1, where scipy returns a NaN lambda.
  if (!std::isfinite(capped)) throw Error(where + ": box-cox's lambda needs capping, and the cap did not converge");
  return capped;
}

// scipy.optimize.fminbound: Brent's method confined to [low, high].
// Yeo-Johnson's lambda comes out of this, over the bounds scipy derives from
// the data's magnitude.
template <typename F>
double bounded_minimize(F&& func, double low, double high) {
  constexpr double kTol = 1.48e-8;
  const double sqrt_eps = std::sqrt(2.2e-16);
  const double golden = 0.5 * (3.0 - std::sqrt(5.0));
  double a = low;
  double b = high;
  double fulcrum = a + golden * (b - a);
  double next = fulcrum;
  double best = fulcrum;
  double rat = 0.0;
  double step = 0.0;
  double f_best = func(best);
  double f_fulcrum = f_best;
  double f_next = f_best;
  double middle = 0.5 * (a + b);
  double tol1 = sqrt_eps * std::fabs(best) + kTol / 3.0;
  double tol2 = 2.0 * tol1;
  for (int calls = 1; std::fabs(best - middle) > (tol2 - 0.5 * (b - a)) && calls < 500; ++calls) {
    bool use_golden = true;
    if (std::fabs(step) > tol1) {
      use_golden = false;
      double r = (best - next) * (f_best - f_fulcrum);
      double q = (best - fulcrum) * (f_best - f_next);
      double p = (best - fulcrum) * q - (best - next) * r;
      q = 2.0 * (q - r);
      if (q > 0.0) p = -p;
      q = std::fabs(q);
      r = step;
      step = rat;
      if (std::fabs(p) < std::fabs(0.5 * q * r) && p > q * (a - best) && p < q * (b - best)) {
        rat = p / q;
        const double candidate = best + rat;
        if ((candidate - a) < tol2 || (b - candidate) < tol2) {
          rat = tol1 * (middle - best >= 0.0 ? 1.0 : -1.0);
        }
      } else {
        use_golden = true;
      }
    }
    if (use_golden) {
      step = best >= middle ? a - best : b - best;
      rat = golden * step;
    }
    const double direction = rat >= 0.0 ? 1.0 : -1.0;
    const double candidate = best + direction * std::max(std::fabs(rat), tol1);
    const double f_candidate = func(candidate);
    if (f_candidate <= f_best) {
      if (candidate >= best) {
        a = best;
      } else {
        b = best;
      }
      fulcrum = next;
      f_fulcrum = f_next;
      next = best;
      f_next = f_best;
      best = candidate;
      f_best = f_candidate;
    } else {
      if (candidate < best) {
        a = candidate;
      } else {
        b = candidate;
      }
      if (f_candidate <= f_next || next == best) {
        fulcrum = next;
        f_fulcrum = f_next;
        next = candidate;
        f_next = f_candidate;
      } else if (f_candidate <= f_fulcrum || fulcrum == best || fulcrum == next) {
        fulcrum = candidate;
        f_fulcrum = f_candidate;
      }
    }
    middle = 0.5 * (a + b);
    tol1 = sqrt_eps * std::fabs(best) + kTol / 3.0;
    tol2 = 2.0 * tol1;
  }
  return best;
}

// ---------------------------------------------------------------------------
// power_transform — numeric columns -> the Box-Cox or Yeo-Johnson power that
// makes them most Gaussian, in place.
// params: {"columns": [...], "method": "yeo-johnson" | "box-cox",
//          "standardize": bool}
// Semantics:
//   * one lambda per column, the maximizer of the profile log-likelihood:
//     Box-Cox brackets (-2, 2) and runs Brent's method, Yeo-Johnson runs
//     Brent's method bounded by the interval scipy derives from the column's
//     magnitude, both on scipy's own objective and at its own tolerance
//   * Box-Cox then caps lambda as scipy does, so that the column's most
//     extreme value transforms to at most f64's largest over 10,000
//   * Box-Cox needs strictly positive data and refuses anything else, at
//     transform as well as at fit; it also refuses a constant column, which
//     has no maximizer. Yeo-Johnson takes any value, and a constant column
//     keeps lambda 1, the identity, as sklearn does
//   * standardize=true then centers and scales each transformed column by its
//     own mean and standard deviation, which sklearn computes with the
//     corrected two-pass formula; a column whose variance is within the error
//     bound of constant keeps scale 1. A Yeo-Johnson column the constant
//     check caught takes them from its raw values, as sklearn, which leaves
//     that column untransformed when it fits the standardizer, does
//   * a null teaches nothing and passes through as a null, which is how
//     sklearn treats a NaN
//   * BUFFERED and iterative: each worker holds every value of its row groups
//     until the merge; the solver then sweeps a log of each value and a work
//     array several times per objective evaluation, and standardizing
//     rewrites the values once and takes three exact sums. Every sum is an
//     exact one, so the fitted lambda does not depend on how the rows were
//     split
//   * the transformed columns become f64 in place, keeping their names
// ---------------------------------------------------------------------------
class PowerTransform final : public Kernel {
 public:
  explicit PowerTransform(const nlohmann::json& params) : columns_(column_list(params, "columns", "power_transform")) {
    if (params.contains("method") && !params.at("method").is_null()) {
      const nlohmann::json& method = params.at("method");
      if (!method.is_string() ||
          (method.get<std::string>() != "yeo-johnson" && method.get<std::string>() != "box-cox")) {
        throw Error("power_transform: 'method' must be 'yeo-johnson' or 'box-cox'");
      }
      box_cox_ = method.get<std::string>() == "box-cox";
    }
    if (params.contains("standardize") && !params.at("standardize").is_null()) {
      if (!params.at("standardize").is_boolean()) throw Error("power_transform: 'standardize' must be true or false");
      standardize_ = params.at("standardize").get<bool>();
    }
    inputs = columns_;
    outputs = columns_;
  }

  std::string id() const override { return "power_transform"; }

  nlohmann::json params() const override {
    return {{"columns", columns_}, {"method", box_cox_ ? "box-cox" : "yeo-johnson"}, {"standardize", standardize_}};
  }

  void resolve(Schema& schema) override {
    bind_numeric(schema, stage, columns_, true, false, "expand");
    for (auto& [name, type] : schema.cols) {
      if (std::find(columns_.begin(), columns_.end(), name) != columns_.end()) type = "f64";
    }
    inputs = columns_;
    outputs = columns_;
  }

  bool needs_fit() const override { return true; }

  struct State final : AccState {
    std::unordered_map<std::string, SpillVector<double>> buffers;
  };

  std::unique_ptr<AccState> make_state() const override { return std::make_unique<State>(); }

  void consume(AccState& raw, const Batch& batch) const override {
    auto& state = static_cast<State&>(raw);
    for (const std::string& name : columns_) {
      SpillVector<double>& buffer = state.buffers[name];
      const std::string where = stage + ": column '" + name + "'";
      const arrow::Array& array = column_array(batch, stage, name);
      for_each_number(array, stage, name, [&](int64_t, bool is_null, double value) {
        if (is_null) return;
        if (box_cox_ && value <= 0.0) {
          throw Error(where + ": value " + std::format("{}", value) +
                      " is not strictly positive, which box-cox requires");
        }
        buffer.push_back(value);
      });
    }
  }

  void finalize(std::vector<std::unique_ptr<AccState>> states) override {
    // Adopt the first worker's buffers instead of copying into a fresh state:
    // the largest buffer is then never duplicated at the peak of the fit.
    std::unique_ptr<State> owned;
    for (auto& raw : states) {
      if (raw == nullptr) continue;
      if (owned == nullptr) {
        owned.reset(static_cast<State*>(raw.release()));
        continue;
      }
      auto& state = static_cast<State&>(*raw);
      for (auto& [name, buffer] : state.buffers) {
        SpillVector<double>& into = owned->buffers[name];
        into.append(std::move(buffer));
      }
      raw.reset();
    }
    if (owned == nullptr) owned = std::make_unique<State>();
    State& merged = *owned;

    fits_.clear();
    for (const std::string& name : columns_) {
      SpillVector<double>& buffer = merged.buffers[name];
      const std::string where = stage + ": column '" + name + "'";
      if (buffer.size() < 2) {
        throw Error(where + " has fewer than two non-null values to fit a power transform from");
      }
      Fit fit;
      // sklearn leaves a Yeo-Johnson column it calls constant at lambda 1,
      // the identity, rather than running the optimizer over a likelihood
      // with no maximum. It also leaves that column untransformed when it
      // fits the standardizer, so the mean and scale come from the raw
      // values, while transform still applies lambda 1.
      const bool constant = !box_cox_ && constant_column(buffer, where);
      fit.lambda = constant ? 1.0 : serializable(solve(name, buffer), stage, name);
      if (standardize_) standardizer(where, buffer, fit, !constant);
      fits_.emplace(name, fit);
      buffer.release();
    }
    fitted = true;
  }

  nlohmann::json state() const override {
    nlohmann::json lambdas = nlohmann::json::object();
    nlohmann::json means = nlohmann::json::object();
    nlohmann::json scales = nlohmann::json::object();
    for (const std::string& name : columns_) {
      const Fit& fit = fits_.at(name);
      lambdas[name] = serializable(fit.lambda, stage, name);
      if (!standardize_) continue;
      means[name] = serializable(fit.mean, stage, name);
      scales[name] = serializable(fit.scale, stage, name);
    }
    // The mean and the scale are written exactly when they were fitted, so
    // a manifest can never half-describe a standardized column.
    nlohmann::json state{{"lambdas", std::move(lambdas)}, {"means", nullptr}, {"scales", nullptr}};
    if (standardize_) {
      state["means"] = std::move(means);
      state["scales"] = std::move(scales);
    }
    return state;
  }

  void load_state(const nlohmann::json& state) override {
    if (!state.is_object() || !state.contains("lambdas") || !state.at("lambdas").is_object()) {
      throw Error(stage + ": state must hold a 'lambdas' object");
    }
    // Both are present exactly when the params say the columns were
    // standardized: a manifest missing them would silently serve values
    // that are not standardized.
    const nlohmann::json* means = optional_per_column_object(state, "means", columns_, stage);
    const nlohmann::json* scales = optional_per_column_object(state, "scales", columns_, stage);
    if (standardize_ != (means != nullptr) || standardize_ != (scales != nullptr)) {
      throw Error(stage + ": 'means' and 'scales' must be present exactly when standardize is true");
    }
    fits_.clear();
    for (const std::string& name : columns_) {
      const nlohmann::json& lambdas = state.at("lambdas");
      if (!lambdas.contains(name)) throw Error(stage + ": lambdas has no entry for column '" + name + "'");
      Fit fit;
      fit.lambda = manifest_number(lambdas.at(name), stage + ": lambdas['" + name + "']");
      if (means != nullptr) {
        fit.mean = manifest_number(means->at(name), stage + ": means['" + name + "']");
        fit.scale = manifest_number(scales->at(name), stage + ": scales['" + name + "']");
        if (fit.scale <= 0.0) throw Error(stage + ": scales['" + name + "'] must be positive");
      }
      fits_.emplace(name, fit);
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
      const std::string where = stage + ": column '" + name + "'";
      const Fit& fit = fits_.at(name);
      PrimitiveOutput<arrow::DoubleType> output(array.length(), where);
      double* values = output.mutable_values();
      for_each_number(array, stage, name, [&](int64_t row, bool is_null, double value) {
        if (is_null) {
          output.set_null(row);
          return;
        }
        if (box_cox_ && value <= 0.0) refuse_non_positive(value, where);
        values[row] = apply(fit, value);
      });
      batch.cols[static_cast<std::size_t>(idx)] = output.finish();
    }
    return batch;
  }

  Row transform_row(Row row) const override {
    RecordLookup lookup(row, columns_.size());
    for (const std::string& name : columns_) {
      RowValue* slot = lookup.find(name);
      if (slot == nullptr) throw Error(stage + ": key '" + name + "' missing from record");
      const std::optional<double> value = number_value(*slot, stage, name);
      if (!value.has_value()) {
        *slot = std::monostate{};
        continue;
      }
      if (box_cox_ && *value <= 0.0) refuse_non_positive(*value, stage + ": key '" + name + "'");
      *slot = apply(fits_.at(name), *value);
    }
    return row;
  }

 private:
  // One fitted column: its lambda, and the mean and scale of the transformed
  // values when standardize is on (0 and 1 when it is off).
  struct Fit {
    double lambda = 1.0;
    double mean = 0.0;
    double scale = 1.0;
  };

  // Box-Cox is defined for positive values only. Callers build `where` only
  // on this path, so a record pays one comparison per value rather than an
  // allocated message.
  [[noreturn]] static void refuse_non_positive(double value, const std::string& where) {
    throw Error(where + ": value " + std::format("{}", value) + " is not strictly positive, which box-cox requires");
  }

  double apply(const Fit& fit, double value) const {
    const double transformed = box_cox_ ? box_cox_value(value, fit.lambda) : yeo_johnson_value(value, fit.lambda);
    return standardize_ ? (transformed - fit.mean) / fit.scale : transformed;
  }

  // sklearn's _is_constant_feature over the column's mean and numpy variance.
  static bool constant_column(const SpillVector<double>& buffer, const std::string& where) {
    const auto count = static_cast<double>(buffer.size());
    const double mean = exact_sum(buffer, where, [](double value) { return value; }) / count;
    return constant_feature(numpy_variance(buffer, where), mean, count);
  }

  // The lambda that maximizes the profile log-likelihood, by the solver scipy
  // uses for that method and over the interval scipy hands it. Box-Cox's is
  // then capped as scipy caps it.
  double solve(const std::string& name, const SpillVector<double>& buffer) const {
    const std::string where = stage + ": column '" + name + "'";
    if (box_cox_) {
      // scipy's boxcox refuses a constant column: its likelihood has no
      // interior maximum, and every lambda transforms it to a constant.
      const double first = buffer.front();
      bool constant = true;
      double smallest = first;
      double largest = first;
      buffer.for_each([&](std::size_t, double value) {
        constant = constant && value == first;
        smallest = std::min(smallest, value);
        largest = std::max(largest, value);
      });
      if (constant) {
        throw Error(where + " is constant, which box-cox cannot fit a lambda for");
      }
      Likelihood likelihood(buffer, true, where);
      const double found = brent_minimize([&](double lambda) { return likelihood.negative(lambda); }, -2.0, 2.0, where);
      return cap_box_cox_lambda(found, smallest, largest, where);
    }
    bool all_zero = true;
    double largest = 0.0;
    bool any_negative = false;
    bool all_negative = true;
    buffer.for_each([&](std::size_t, double value) {
      all_zero = all_zero && value == 0.0;
      largest = std::max(largest, std::fabs(value));
      any_negative = any_negative || value < 0.0;
      all_negative = all_negative && value < 0.0;
    });
    if (all_zero) return 1.0;
    // scipy's bounds: the lambdas whose transform stays inside half of f64's
    // exponent range for data up to twenty times the largest magnitude seen.
    const double reach = std::log1p(20.0 * largest);
    const double log_eps = std::log(kEps);
    double low = (std::log(kSmallestNormal) - log_eps) / 2.0 / reach;
    double high = (std::log(kLargest) + log_eps) / 2.0 / reach;
    if (all_negative) {
      const double swapped = 2.0 - high;
      high = 2.0 - low;
      low = swapped;
    } else if (any_negative) {
      low = std::max(2.0 - high, low);
      high = std::min(2.0 - low, high);
    }
    Likelihood likelihood(buffer, false, where);
    return bounded_minimize([&](double lambda) { return likelihood.negative(lambda); }, low, high);
  }

  // sklearn standardizes with StandardScaler, whose variance is the corrected
  // two-pass formula and whose scale is 1 for a column indistinguishable from
  // constant. The mean and variance are of the transformed values, or of the
  // raw ones for a column sklearn left untransformed, so this is three
  // statistic sweeps after the solver. The solver is done with the fit
  // buffer, so it is transformed in place, without a second O(rows) buffer,
  // and the three exact sums read the transformed values from it.
  void standardizer(const std::string& where, SpillVector<double>& values, Fit& fit, bool transformed) const {
    const auto count = static_cast<double>(values.size());
    if (transformed) {
      values.rewrite([&](std::size_t, double value) {
        return box_cox_ ? box_cox_value(value, fit.lambda) : yeo_johnson_value(value, fit.lambda);
      });
    }
    const double mean = exact_sum(values, where, [](double value) { return value; }) / count;
    const double correction = exact_sum(values, where, [&](double value) { return value - mean; });
    const double squares = exact_sum(values, where, [&](double value) { return (value - mean) * (value - mean); });
    const double variance = (squares - correction * correction / count) / count;
    fit.mean = mean;
    fit.scale = constant_feature(variance, mean, count) ? 1.0 : std::sqrt(variance);
  }

  std::vector<std::string> columns_;
  bool box_cox_ = false;
  bool standardize_ = true;
  std::unordered_map<std::string, Fit> fits_;
};

}  // namespace

std::unique_ptr<Kernel> make_polynomial_features(const nlohmann::json& params) {
  return std::make_unique<PolynomialFeatures>(params);
}

std::unique_ptr<Kernel> make_spline_transform(const nlohmann::json& params) {
  return std::make_unique<SplineTransform>(params);
}

std::unique_ptr<Kernel> make_power_transform(const nlohmann::json& params) {
  return std::make_unique<PowerTransform>(params);
}

}  // namespace fastfeaturetransform
