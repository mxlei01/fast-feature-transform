// The three sklearn.preprocessing transformers whose fit is a quantile:
// robust_scale, quantile_transform and k_bins_discretize. An exact quantile
// needs every value of a column, so all three sit in the buffered tier with
// group_agg's median and quantile_encode: each worker keeps every value it
// reads, spilling past the shared fit memory budget, until finalize merges
// them and reads the order statistics it needs. That accumulator, the numpy
// quantile methods and the inverse normal CDF they draw on live here too.
//
// k_bins_discretize(strategy="uniform") is the one exception: its edges are
// linspace(min, max, n_bins + 1), a plain min/max monoid, so it turns the
// buffering off and keeps O(1) memory per worker.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <format>
#include <limits>
#include <numbers>
#include <optional>
#include <unordered_map>

#include "fastfeaturetransform/kernels/cast_table.hpp"
#include "fastfeaturetransform/kernels/categories.hpp"
#include "fastfeaturetransform/kernels/kernels.hpp"
#include "fastfeaturetransform/kernels/numeric.hpp"
#include "fastfeaturetransform/kernels/output.hpp"
#include "fastfeaturetransform/kernels/scale_transform.hpp"
#include "fastfeaturetransform/kernels/util.hpp"
#include "fastfeaturetransform/spill.hpp"

namespace fastfeaturetransform {
namespace {

// --- the inverse normal CDF -------------------------------------------------
// scipy's stats.norm.ppf, which RobustScaler(unit_variance=True) and
// QuantileTransformer(output_distribution="normal") both call. Peter Acklam's
// rational approximation (about 1.2e-9 relative) refined by two Halley steps
// on the residual Phi(x) - p. Over 114k points (50k spanning (0, 1), 44k
// within 1e-3 of 0.5 and 20k uniform), the worst absolute difference from
// scipy is 1.8e-15 and the worst relative one 8.3e-16. p > 0.5 is answered
// by mirroring 1 - p, because the refinement's erfc term would otherwise
// cancel against a p that is nearly 1. Cephes' ndtri, which scipy calls,
// mirrors its upper tail too. Below the smallest normal double
// normal_ppf_deep refines instead.

constexpr std::array<double, 6> kCentralNumerator = {-3.969683028665376e+01, 2.209460984245205e+02,
                                                     -2.759285104469687e+02, 1.383577518672690e+02,
                                                     -3.066479806614716e+01, 2.506628277459239e+00};
constexpr std::array<double, 5> kCentralDenominator = {-5.447609879822406e+01, 1.615858368580409e+02,
                                                       -1.556989798598866e+02, 6.680131188771972e+01,
                                                       -1.328068155288572e+01};
constexpr std::array<double, 6> kTailNumerator = {-7.784894002430293e-03, -3.223964580411365e-01,
                                                  -2.400758277161838e+00, -2.549732539343734e+00,
                                                  4.374664141464968e+00,  2.938163982698783e+00};
constexpr std::array<double, 4> kTailDenominator = {7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
                                                    3.754408661907416e+00};
constexpr double kTailBreak = 0.02425;

// Newton's method on log Phi(x) = log p, for a p below the smallest normal
// double. There the erfc residual is subnormal and loses bits, and below
// about 6e-311, where x passes -37.677, exp(x * x / 2) overflows and a
// Halley step gives NaN. x is below -37.5 here, where the asymptotic series
// Phi(x) = phi(x) / -x * (1 - 1/x^2 + 3/x^4 - ...) is exact to double
// precision after seven terms. Over 1,163 p from 5e-324 to 3e-308, the
// answer is within one double of the correctly rounded one, and within 2
// ulp of scipy's.
double normal_ppf_deep(double p, double x) {
  const double log_p = std::log(p);
  for (int step = 0; step < 2; ++step) {
    const double inverse_square = 1.0 / (x * x);
    double term = 1.0;
    double series = 1.0;
    for (int k = 1; k <= 7; ++k) {
      term *= -(2.0 * k - 1.0) * inverse_square;
      series += term;
    }
    const double log_cdf = -0.5 * x * x - std::log(-x) - 0.5 * std::log(2.0 * std::numbers::pi) + std::log(series);
    // The slope of log Phi is phi(x) / Phi(x) = -x / series.
    x -= (log_cdf - log_p) * series / -x;
  }
  return x;
}

// The lower half only: 0 < p <= 0.5.
double normal_ppf_lower(double p) {
  double x = 0.0;
  if (p < kTailBreak) {
    const double q = std::sqrt(-2.0 * std::log(p));
    x = (((((kTailNumerator[0] * q + kTailNumerator[1]) * q + kTailNumerator[2]) * q + kTailNumerator[3]) * q +
          kTailNumerator[4]) *
             q +
         kTailNumerator[5]) /
        ((((kTailDenominator[0] * q + kTailDenominator[1]) * q + kTailDenominator[2]) * q + kTailDenominator[3]) * q +
         1.0);
  } else {
    const double q = p - 0.5;
    const double r = q * q;
    x = (((((kCentralNumerator[0] * r + kCentralNumerator[1]) * r + kCentralNumerator[2]) * r + kCentralNumerator[3]) *
              r +
          kCentralNumerator[4]) *
             r +
         kCentralNumerator[5]) *
        q /
        (((((kCentralDenominator[0] * r + kCentralDenominator[1]) * r + kCentralDenominator[2]) * r +
           kCentralDenominator[3]) *
              r +
          kCentralDenominator[4]) *
             r +
         1.0);
  }
  if (p < std::numeric_limits<double>::min()) return normal_ppf_deep(p, x);
  for (int step = 0; step < 2; ++step) {
    // Above p = 0.25, 0.5 * erfc(...) and p are both near 0.5 and cancel,
    // leaving a residual only accurate to about 3e-17, which near p = 0.5 is
    // large beside x. There the residual is erf's half minus p - 0.5 instead:
    // p - 0.5 is exact above 0.25, and erf keeps x's relative accuracy. At
    // 0.25 itself the two are equally accurate, and erfc's gives sklearn's
    // ppf(0.25), and with it RobustScaler's default divisor, to the last bit
    // with macOS's libm.
    const double error =
        p > 0.25 ? 0.5 * std::erf(x / std::numbers::sqrt2) - (p - 0.5) : 0.5 * std::erfc(-x / std::numbers::sqrt2) - p;
    const double slope = error * std::sqrt(2.0 * std::numbers::pi) * std::exp(x * x / 2.0);
    x -= slope / (1.0 + x * slope / 2.0);
  }
  return x;
}

double normal_ppf(double p) {
  if (p <= 0.0) return -std::numeric_limits<double>::infinity();
  if (p >= 1.0) return std::numeric_limits<double>::infinity();
  return p > 0.5 ? -normal_ppf_lower(1.0 - p) : normal_ppf_lower(p);
}

// --- numpy's quantile machinery ---------------------------------------------

// numpy's _lerp: interpolate from whichever end is nearer, so the answer is
// the bound itself at t = 0 and t = 1.
double lerp(double low, double high, double weight) {
  const double span = high - low;
  return weight >= 0.5 ? high - span * (1.0 - weight) : low + span * weight;
}

// Nine of the methods np.percentile accepts (all but lower, higher, midpoint
// and nearest), the ones KBinsDiscretizer exposes as quantile_method. Each is
// a rule for the "virtual index" the quantile sits at in the sorted sample;
// the two discrete ones round it to a real index instead of interpolating.
// Hyndman & Fan (1996) numbers them; numpy's names are the ones spelled here.
enum class QuantileMethod {
  InvertedCdf,
  AveragedInvertedCdf,
  ClosestObservation,
  InterpolatedInvertedCdf,
  Hazen,
  Weibull,
  Linear,
  MedianUnbiased,
  NormalUnbiased,
};

inline const char* kQuantileMethods =
    "'inverted_cdf', 'averaged_inverted_cdf', 'closest_observation', 'interpolated_inverted_cdf', 'hazen', "
    "'weibull', 'linear', 'median_unbiased' or 'normal_unbiased'";

QuantileMethod quantile_method_from(const std::string& text, const std::string& what) {
  static const std::unordered_map<std::string, QuantileMethod> kMethods{
      {"inverted_cdf", QuantileMethod::InvertedCdf},
      {"averaged_inverted_cdf", QuantileMethod::AveragedInvertedCdf},
      {"closest_observation", QuantileMethod::ClosestObservation},
      {"interpolated_inverted_cdf", QuantileMethod::InterpolatedInvertedCdf},
      {"hazen", QuantileMethod::Hazen},
      {"weibull", QuantileMethod::Weibull},
      {"linear", QuantileMethod::Linear},
      {"median_unbiased", QuantileMethod::MedianUnbiased},
      {"normal_unbiased", QuantileMethod::NormalUnbiased},
  };
  auto found = kMethods.find(text);
  if (found == kMethods.end()) {
    throw Error(what + ": 'quantile_method' must be one of " + std::string(kQuantileMethods));
  }
  return found->second;
}

const char* quantile_method_name(QuantileMethod method) {
  switch (method) {
    case QuantileMethod::InvertedCdf:
      return "inverted_cdf";
    case QuantileMethod::AveragedInvertedCdf:
      return "averaged_inverted_cdf";
    case QuantileMethod::ClosestObservation:
      return "closest_observation";
    case QuantileMethod::InterpolatedInvertedCdf:
      return "interpolated_inverted_cdf";
    case QuantileMethod::Hazen:
      return "hazen";
    case QuantileMethod::Weibull:
      return "weibull";
    case QuantileMethod::MedianUnbiased:
      return "median_unbiased";
    case QuantileMethod::NormalUnbiased:
      return "normal_unbiased";
    case QuantileMethod::Linear:
      break;
  }
  return "linear";
}

// numpy's plotting-position constants: virtual index
// n*q + (alpha + q*(1 - alpha - beta)) - 1.
std::pair<double, double> plotting_position(QuantileMethod method) {
  switch (method) {
    case QuantileMethod::InterpolatedInvertedCdf:
      return {0.0, 1.0};
    case QuantileMethod::Hazen:
      return {0.5, 0.5};
    case QuantileMethod::Weibull:
      return {0.0, 0.0};
    case QuantileMethod::MedianUnbiased:
      return {1.0 / 3.0, 1.0 / 3.0};
    case QuantileMethod::NormalUnbiased:
      return {3.0 / 8.0, 3.0 / 8.0};
    // quantile_at answers the other four without a plotting position, so they
    // are listed rather than defaulted: a tenth method draws a -Wswitch
    // warning here instead of silently taking Weibull's constants.
    case QuantileMethod::Linear:
    case QuantileMethod::InvertedCdf:
    case QuantileMethod::AveragedInvertedCdf:
    case QuantileMethod::ClosestObservation:
      break;
  }
  return {0.0, 0.0};
}

// The quantile of a sorted, non-empty sample, value for value as
// np.quantile(values, q, method=...) computes it.
double quantile_at(const SpillVector<double>& sorted, double q, QuantileMethod method) {
  if (sorted.empty()) return 0.0;
  const auto count = static_cast<double>(sorted.size());
  const auto last = static_cast<std::ptrdiff_t>(sorted.size()) - 1;
  auto clamp = [&](double index) {
    return sorted[static_cast<std::size_t>(std::clamp(static_cast<std::ptrdiff_t>(index), std::ptrdiff_t{0}, last))];
  };
  // The two discrete methods take an order statistic rather than interpolate:
  // numpy rounds the virtual index down when its fraction meets the method's
  // condition and up otherwise.
  if (method == QuantileMethod::InvertedCdf || method == QuantileMethod::ClosestObservation) {
    // closest_observation shifts the index half a step down, so it rounds to
    // the nearest order statistic, and a tie goes to the even one (counting
    // from 1) rather than always to the upper one.
    const double index = method == QuantileMethod::InvertedCdf ? (count * q) - 1.0 : (count * q) - 1.5;
    const double previous = std::floor(index);
    // numpy's %, which follows the divisor's sign rather than C's fmod.
    const double parity = previous - 2.0 * std::floor(previous / 2.0);
    const bool down = method == QuantileMethod::InvertedCdf ? index == previous
                                                            : index == previous && parity == 1.0;  // the nearest even
                                                                                                   // order statistic
    return clamp(std::max(down ? previous : previous + 1.0, 0.0));
  }
  double index = 0.0;
  if (method == QuantileMethod::AveragedInvertedCdf) {
    index = (count * q) - 1.0;
  } else if (method == QuantileMethod::Linear) {
    index = (count - 1.0) * q;
  } else {
    const auto [alpha, beta] = plotting_position(method);
    index = count * q + (alpha + q * (1.0 - alpha - beta)) - 1.0;
  }
  double weight = index - std::floor(index);
  // averaged_inverted_cdf averages the two neighbors when the index lands
  // exactly on one, and takes the upper neighbor otherwise.
  if (method == QuantileMethod::AveragedInvertedCdf) weight = weight == 0.0 ? 0.5 : 1.0;
  // numpy pins both neighbors to the same end outside the sample, so the
  // interpolation there is the end value whatever the weight became.
  if (index >= count - 1.0) return sorted.back();
  if (index < 0.0) return sorted.front();
  const auto previous = static_cast<std::size_t>(std::floor(index));
  return lerp(sorted[previous], sorted[previous + 1], weight);
}

// np.median of a sorted, non-empty sample: the midpoint of the two middle
// values for an even count. That is not the "linear" quantile at 0.5, which
// lerp computes down from the upper value and can round differently.
double median_at(const SpillVector<double>& sorted) {
  if (sorted.empty()) return 0.0;
  const std::size_t half = sorted.size() / 2;
  if (sorted.size() % 2 == 1) return sorted[half];
  return (sorted[half - 1] + sorted[half]) / 2.0;
}

// np.interp(x, xp, fp) for one value, with xp non-decreasing and fp the same
// length. The neighbor is the last index whose xp is at most x, which is
// what makes a run of equal xp answer with its upper end. The product is
// fused, as the compiled numpy this is checked against fuses it.
double interpolate_interval(double x, const std::vector<double>& xp, const std::vector<double>& fp, std::size_t index) {
  if (index == xp.size() - 1 || xp[index] == x) return fp[index];
  const double slope = (fp[index + 1] - fp[index]) / (xp[index + 1] - xp[index]);
  // Between two xp less than (fp[index + 1] - fp[index]) / f64's largest
  // apart the slope overflows, and numpy then answers inf, which turns the
  // averaged rank into NaN. Dividing the gaps first stays in range; elsewhere
  // numpy's own formula keeps its last bit.
  if (std::isinf(slope)) {
    return fp[index] + (fp[index + 1] - fp[index]) * ((x - xp[index]) / (xp[index + 1] - xp[index]));
  }
  double result = std::fma(slope, x - xp[index], fp[index]);
  if (std::isnan(result)) {  // numpy tries the other end before giving up
    result = std::fma(slope, x - xp[index + 1], fp[index + 1]);
    if (std::isnan(result) && fp[index] == fp[index + 1]) result = fp[index];
  }
  return result;
}

double interpolate(double x, const std::vector<double>& xp, const std::vector<double>& fp) {
  const std::size_t count = xp.size();
  if (count == 1) return fp.front();
  if (x > xp.back()) return fp.back();
  if (x < xp.front()) return fp.front();
  std::size_t low = 0;
  std::size_t high = count;
  while (low < high) {
    const std::size_t mid = low + (high - low) / 2;
    if (x >= xp[mid]) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }
  return interpolate_interval(x, xp, fp, low - 1);
}

// The index of the last grid value at most x, for a sorted grid whose first
// value is at most x: std::upper_bound less one, without its branch. For a
// value inside the grid that branch goes either way at random, and its
// mispredictions cost more than the loads. On a sorted grid every binary
// search lands on the same partition point, so the index is upper_bound's.
std::size_t last_at_most(const std::vector<double>& grid, double x) {
  const double* base = grid.data();
  std::size_t count = grid.size();
  while (count > 1) {
    const std::size_t half = count / 2;
    base = base[half] <= x ? base + half : base;
    count -= half;
  }
  return static_cast<std::size_t>(base - grid.data());
}

std::pair<double, double> interpolate_pair(double x, const std::vector<double>& grid,
                                           const std::vector<double>& references,
                                           const std::vector<double>& mirrored_grid,
                                           const std::vector<double>& mirrored_references) {
  if (grid.size() < 2 || x < grid.front() || x > grid.back()) {
    return {interpolate(x, grid, references), interpolate(-x, mirrored_grid, mirrored_references)};
  }
  const std::size_t index = last_at_most(grid, x);
  const double forward = interpolate_interval(x, grid, references, index);
  if (grid[index] == x) {
    // An exact hit can sit in a run of equal grid values, whose other end the
    // mirrored side needs, so that side keeps its own search. Strictly inside
    // an interval, the mirrored index follows directly.
    return {forward, interpolate(-x, mirrored_grid, mirrored_references)};
  }
  return {forward, interpolate_interval(-x, mirrored_grid, mirrored_references, grid.size() - index - 2)};
}

// --- the numeric substrate --------------------------------------------------

// One record value as a double, or nullopt when it is null. Types are
// explicit, so a string or a bool is an error rather than a parse.
std::optional<double> row_number(const RowValue& value, const std::string& stage, const std::string& name) {
  if (std::holds_alternative<std::monostate>(value)) return std::nullopt;
  if (const int64_t* integer = std::get_if<int64_t>(&value)) return static_cast<double>(*integer);
  if (const double* number = std::get_if<double>(&value)) return *number;
  throw Error(stage + ": key '" + name + "' must be a number or null");
}

// NaN and infinity are refused wherever these kernels read a value, at fit
// and at transform alike: a null is how a missing value is spelled here (it
// is what upstream's NaN maps onto), a fitted table has to be finite to be
// JSON, and the quantile interpolation has no answer for a value that
// compares false against every bound it is given. Callers build `where` only
// on this path, so the hot loop pays a single isfinite. Only a step that
// passes a null through adds 'write a missing value as a null' to the
// message; k_bins_discretize refuses a null too.
[[noreturn]] void refuse_non_finite(double value, const std::string& where, bool null_passes = true) {
  throw Error(where + ": value " + std::format("{}", value) +
              (null_passes ? " is not finite; write a missing value as a null" : " is not finite"));
}

// A number the fit computed. Every input value is finite, but the midpoint or
// the interpolation of two enormous ones can still overflow and come out
// infinite or NaN, and an artifact must never carry either: state() would
// refuse to write it, so the fit fails here instead.
double fitted_number(double value, const std::string& stage, const std::string& name, const char* what) {
  if (!std::isfinite(value)) {
    throw Error(stage + ": column '" + name + "' fits a " + what + " that is not finite; its values are too large");
  }
  return value;
}

// One worker's view of the fit columns: every non-null value when the kernel
// buffers, and the extremes and counts always. The buffers spill past the
// fit memory budget; k_bins_discretize(strategy="uniform") leaves them empty.
struct Values final : AccState {
  std::unordered_map<std::string, SpillVector<double>> buffers;
  std::unordered_map<std::string, double> low;
  std::unordered_map<std::string, double> high;
  std::unordered_map<std::string, int64_t> counts;  // non-null values
  std::unordered_map<std::string, int64_t> nulls;
  int64_t rows = 0;
};

void consume_values(const std::vector<std::string>& columns, const std::string& stage, bool buffering, bool null_passes,
                    Values& state, const Batch& batch) {
  state.rows += batch.rows;
  for (const std::string& name : columns) {
    SpillVector<double>& buffer = state.buffers[name];
    auto low = state.low.find(name);
    if (low == state.low.end()) low = state.low.emplace(name, std::numeric_limits<double>::infinity()).first;
    auto high = state.high.find(name);
    if (high == state.high.end()) high = state.high.emplace(name, -std::numeric_limits<double>::infinity()).first;
    int64_t& counted = state.counts[name];
    int64_t& nulls = state.nulls[name];
    const arrow::Array& array = column_array(batch, stage, name);
    for_each_numeric(array, stage, name, [&](int64_t, bool is_null, double value) {
      if (is_null) {
        ++nulls;
        return;
      }
      if (!std::isfinite(value)) refuse_non_finite(value, stage + ": column '" + name + "'", null_passes);
      if (value == 0.0) value = 0.0;  // one zero: -0.0 would make the sorted buffer depend on the split
      ++counted;
      low->second = std::min(low->second, value);
      high->second = std::max(high->second, value);
      if (buffering) buffer.push_back(value);
    });
  }
}

// Fold every worker into the first, freeing each as it is consumed, then sort
// each buffer unless the caller selects its own ranks: a sorted concatenation
// is the same sequence however the rows were split, so every quantile below
// is split-independent.
std::unique_ptr<Values> merge_values(std::vector<std::unique_ptr<AccState>>& states, bool sort = true) {
  std::unique_ptr<Values> merged;
  for (auto& raw : states) {
    if (raw == nullptr) continue;
    if (merged == nullptr) {
      merged.reset(static_cast<Values*>(raw.release()));
      continue;
    }
    auto& from = static_cast<Values&>(*raw);
    merged->rows += from.rows;
    for (auto& [name, buffer] : from.buffers) {
      merged->buffers[name].append(std::move(buffer));
    }
    for (const auto& [name, value] : from.low) {
      auto found = merged->low.find(name);
      if (found == merged->low.end()) {
        merged->low.emplace(name, value);
      } else {
        found->second = std::min(found->second, value);
      }
    }
    for (const auto& [name, value] : from.high) {
      auto found = merged->high.find(name);
      if (found == merged->high.end()) {
        merged->high.emplace(name, value);
      } else {
        found->second = std::max(found->second, value);
      }
    }
    for (const auto& [name, count] : from.counts) merged->counts[name] += count;
    for (const auto& [name, count] : from.nulls) merged->nulls[name] += count;
    raw.reset();
  }
  if (merged == nullptr) merged = std::make_unique<Values>();
  if (sort) {
    for (auto& [name, buffer] : merged->buffers) buffer.sort();
  }
  return merged;
}

// The per-column list a state() object writes and load_state reads back.
nlohmann::json number_lists(const std::vector<std::string>& columns, const std::string& stage,
                            const std::unordered_map<std::string, std::vector<double>>& tables) {
  nlohmann::json out = nlohmann::json::object();
  for (const std::string& name : columns) {
    std::vector<double> checked;
    for (double value : tables.at(name)) checked.push_back(serializable(value, stage, name));
    out[name] = std::move(checked);
  }
  return out;
}

// One per-column list out of a manifest, checked before it is trusted: an
// object holding, for every column, a non-empty list of finite numbers.
std::unordered_map<std::string, std::vector<double>> load_number_lists(const nlohmann::json& state,
                                                                       const std::string& key, const std::string& stage,
                                                                       std::vector<std::string>& columns,
                                                                       bool explicit_columns) {
  columns_from_state(state, key, columns, explicit_columns, stage);
  const nlohmann::json& lists = state.at(key);
  std::unordered_map<std::string, std::vector<double>> tables;
  for (const std::string& name : columns) {
    if (!lists.contains(name) || !lists.at(name).is_array() || lists.at(name).empty()) {
      throw Error(stage + ": " + key + "['" + name + "'] must be a non-empty list of numbers");
    }
    std::vector<double> values;
    for (const nlohmann::json& value : lists.at(name)) {
      if (!value.is_number() || !std::isfinite(value.get<double>())) {
        throw Error(stage + ": " + key + "['" + name + "'] must hold finite numbers");
      }
      values.push_back(value.get<double>());
    }
    tables.emplace(name, std::move(values));
  }
  return tables;
}

// ---------------------------------------------------------------------------
// robust_scale — numeric columns -> (x - median) / the width of their
// quantile_range, in place.
// params: {"columns": [...] | null, "with_centering": bool,
//          "with_scaling": bool, "quantile_range": [q_min, q_max],
//          "unit_variance": bool}
// Semantics:
//   * center is the column's median and scale the width of its
//     quantile_range, both over the non-null fit values; transform is
//     (x - center) / scale, and the column becomes f64 in place
//   * a scale below 10 * f64 epsilon becomes 1, so a constant column is
//     centered and not divided, which is upstream's _handle_zeros_in_scale
//   * unit_variance=True divides the scale by
//     ppf(q_max/100) - ppf(q_min/100), the width of the same range on a
//     standard normal, so a normal column comes out with unit variance; it
//     needs 0 < q_min < q_max < 100, since the other ranges make that
//     divisor zero or infinite, and the constructor refuses the few ranges
//     inside those bounds that still do: a q_min whose q_min/100 rounds to
//     0, or two percentiles so close that their normal quantiles are equal
//   * with_centering=False leaves center out and with_scaling=False leaves
//     scale out; the state records the missing one as null rather than as a
//     neutral number, so the artifact says which was fitted
//   * a null is ignored at fit and passes through at transform, which is
//     what sklearn's NaN does; a value that is NaN or infinity is an error
//   * columns=null scales every integer or float column that reaches the
//     step; an integer column is widened to f64, so a magnitude past 2^53
//     rounds
//   * BUFFERED: an exact median and quantile need every value, so each
//     worker buffers doubles per column within the shared spill budget until
//     finalize concatenates them and selects the order statistics it reads
// ---------------------------------------------------------------------------
class RobustScale final : public Kernel {
 public:
  explicit RobustScale(const nlohmann::json& params) {
    if (params.contains("columns") && !params.at("columns").is_null()) {
      columns_ = column_list(params, "columns", "robust_scale");
      explicit_columns_ = true;
      inputs = columns_;
      outputs = columns_;
    }
    centering_ = bool_param(params, "with_centering", true, "robust_scale");
    scaling_ = bool_param(params, "with_scaling", true, "robust_scale");
    unit_variance_ = bool_param(params, "unit_variance", false, "robust_scale");
    if (params.contains("quantile_range") && !params.at("quantile_range").is_null()) {
      const nlohmann::json& range = params.at("quantile_range");
      const std::string message = "robust_scale: 'quantile_range' must be two numbers with 0 <= q_min <= q_max <= 100";
      if (!range.is_array() || range.size() != 2 || !range.at(0).is_number() || !range.at(1).is_number()) {
        throw Error(message);
      }
      low_ = range.at(0).get<double>();
      high_ = range.at(1).get<double>();
      if (!(low_ >= 0.0) || !(low_ <= high_) || !(high_ <= 100.0)) throw Error(message);
    }
    if (unit_variance_) {
      if (!(low_ > 0.0 && low_ < high_ && high_ < 100.0)) {
        throw Error(
            "robust_scale: unit_variance needs 'quantile_range' strictly inside 0..100 with q_min < q_max, or the "
            "normal-quantile divisor is zero or infinite");
      }
      adjust_ = normal_ppf(high_ / 100.0) - normal_ppf(low_ / 100.0);
      if (!std::isfinite(adjust_) || adjust_ <= 0.0) {
        throw Error("robust_scale: unit_variance divides by ppf(q_max / 100) - ppf(q_min / 100), which is " +
                    std::format("{}", adjust_) + " for this 'quantile_range'; it must be finite and above 0");
      }
    }
  }

  std::string id() const override { return "robust_scale"; }

  nlohmann::json params() const override {
    return {{"columns", explicit_columns_ ? nlohmann::json(columns_) : nlohmann::json()},
            {"with_centering", centering_},
            {"with_scaling", scaling_},
            {"quantile_range", nlohmann::json::array({low_, high_})},
            {"unit_variance", unit_variance_}};
  }

  void resolve(Schema& schema) override {
    bind_numeric(schema, stage, columns_, explicit_columns_, fitted, "scale");
    for (auto& [name, type] : schema.cols) {
      if (std::find(columns_.begin(), columns_.end(), name) != columns_.end()) type = "f64";
    }
    inputs = columns_;
    outputs = columns_;
  }

  bool needs_fit() const override { return true; }
  std::unique_ptr<AccState> make_state() const override { return std::make_unique<Values>(); }

  void consume(AccState& raw, const Batch& batch) const override {
    consume_values(columns_, stage, centering_ || scaling_, true, static_cast<Values&>(raw), batch);
  }

  void finalize(std::vector<std::unique_ptr<AccState>> states) override {
    std::unique_ptr<Values> merged = merge_values(states, false);
    centers_.clear();
    scales_.clear();
    for (const std::string& name : columns_) {
      SpillVector<double>& buffer = merged->buffers[name];
      if (merged->counts[name] == 0) {
        throw Error(stage + ": column '" + name + "' has no non-null values to learn from");
      }
      std::vector<std::size_t> ranks;
      if (centering_) {
        ranks.push_back(buffer.size() / 2);
        if (buffer.size() % 2 == 0) ranks.push_back(buffer.size() / 2 - 1);
      }
      if (scaling_) {
        linear_quantile_ranks(buffer.size(), low_ / 100.0, ranks);
        linear_quantile_ranks(buffer.size(), high_ / 100.0, ranks);
      }
      select_numeric_ranks(buffer, std::move(ranks));
      centers_[name] = centering_ ? fitted_number(median_at(buffer), stage, name, "center") : 0.0;
      scales_[name] = scaling_ ? scale_of(name, buffer) : 1.0;
      buffer.release();
    }
    fitted = true;
  }

  nlohmann::json state() const override {
    nlohmann::json centers = nlohmann::json::object();
    nlohmann::json scales = nlohmann::json::object();
    for (const std::string& name : columns_) {
      centers[name] = centering_ ? nlohmann::json(serializable(centers_.at(name), stage, name)) : nlohmann::json();
      scales[name] = scaling_ ? nlohmann::json(serializable(scales_.at(name), stage, name)) : nlohmann::json();
    }
    return {{"columns", columns_}, {"center", centers}, {"scale", scales}};
  }

  void load_state(const nlohmann::json& state) override {
    if (!state.is_object() || !state.contains("scale") || !state.at("scale").is_object()) {
      throw Error(stage + ": state must hold a 'center' object and a 'scale' object");
    }
    columns_from_state(state, "center", columns_, explicit_columns_, stage);
    centers_.clear();
    scales_.clear();
    for (const std::string& name : columns_) {
      centers_[name] = statistic(state.at("center"), "center", name, centering_, 0.0);
      const double scale = statistic(state.at("scale"), "scale", name, scaling_, 1.0);
      // The fit divides by a width, which is strictly positive; a negative one
      // would reverse the column and a zero would turn it into infinities and
      // NaN, both silently.
      if (scale <= 0.0) throw Error(stage + ": scale['" + name + "'] must be a positive number");
      scales_[name] = scale;
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
      // The column's center and scale are the same for every row, so the two
      // tables are read once here rather than twice per value inside
      // scale_value().
      const double center = centers_.at(name);
      const double scale = scales_.at(name);
      batch.cols[static_cast<std::size_t>(idx)] = scale_numeric_array(
          batch.cols[static_cast<std::size_t>(idx)], stage, name,
          [&](double value) { return scale_value(center, scale, value); },
          [&](double value) {
            if (!std::isfinite(value)) refuse_non_finite(value, stage + ": column '" + name + "'");
          },
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
      if (number.has_value() && !std::isfinite(*number)) refuse_non_finite(*number, stage + ": key '" + name + "'");
      *value = number.has_value() ? RowValue(scale_value(centers_.at(name), scales_.at(name), *number))
                                  : RowValue(std::monostate{});
    }
    return row;
  }

 private:
  // The scale of one column: its quantile width, divided by adjust_. A width
  // below 10 * epsilon becomes 1 first, upstream's near-zero rule, so a
  // constant column survives.
  double scale_of(const std::string& name, const SpillVector<double>& buffer) const {
    const double width = quantile_at(buffer, high_ / 100.0, QuantileMethod::Linear) -
                         quantile_at(buffer, low_ / 100.0, QuantileMethod::Linear);
    const double scale = width < 10.0 * std::numeric_limits<double>::epsilon() ? 1.0 : width;
    return fitted_number(scale / adjust_, stage, name, "scale");
  }

  double statistic(const nlohmann::json& object, const char* key, const std::string& name, bool wanted,
                   double fallback) const {
    if (!object.contains(name)) throw Error(stage + ": " + key + " has no entry for column '" + name + "'");
    const nlohmann::json& value = object.at(name);
    if (!wanted) {
      // The flag says this statistic was never fitted, so a number here would
      // be a statistic no transform can use: refuse rather than ignore it.
      if (!value.is_null()) {
        throw Error(stage + ": " + key + "['" + name + "'] must be null when this scaler does not fit it");
      }
      return fallback;
    }
    if (!value.is_number() || !std::isfinite(value.get<double>())) {
      throw Error(stage + ": " + key + "['" + name + "'] must be a finite number");
    }
    return value.get<double>();
  }

  double scale_value(double center, double scale, double value) const { return (value - center) / scale; }

  std::vector<std::string> columns_;
  bool explicit_columns_ = false;
  bool centering_ = true;
  bool scaling_ = true;
  bool unit_variance_ = false;
  double low_ = 25.0;
  double high_ = 75.0;
  // The divisor of unit_variance, ppf(q_max/100) - ppf(q_min/100): finite and
  // above 0, as the constructor checks, and 1 without unit_variance.
  double adjust_ = 1.0;
  std::unordered_map<std::string, double> centers_;  // 0 when with_centering is false
  std::unordered_map<std::string, double> scales_;   // 1 when with_scaling is false
};

// ---------------------------------------------------------------------------
// quantile_transform — numeric columns -> their rank in the fitted
// distribution, in place.
// params: {"columns": [...] | null, "n_quantiles": int,
//          "output_distribution": "uniform" | "normal"}
// Semantics:
//   * the fit records n_quantiles values per column, the percentiles at
//     references * 100 over its non-null values, against the shared grid
//     references = linspace(0, 1, n_quantiles); n_quantiles is capped at the
//     fit row count, as upstream caps it
//   * transform interpolates a value into that grid forwards and backwards
//     and averages the two, so a run of equal quantiles answers with the
//     middle of its range rather than one end
//   * output_distribution="uniform" gives a number in [0, 1], up to
//     rounding: the fitted minimum maps to exactly 0 and the fitted maximum
//     to exactly 1, and a value outside the fitted range is clipped to them
//   * output_distribution="normal" then maps that through the inverse normal
//     CDF, clipped to ppf(1e-7 - eps) below and ppf(1 - (1e-7 - eps)) above
//     so a bound cannot map to infinity; there the clip to 0 and 1 applies
//     within 1e-7 of the fitted bounds, which is upstream's BOUNDS_THRESHOLD
//   * a null is ignored at fit and passes through at transform, which is what
//     sklearn's NaN does; a value that is NaN or infinity is an error
//   * the column becomes f64 in place; columns=null transforms every integer
//     or float column that reaches the step
//   * BUFFERED: the grid needs every value, so each worker stores doubles per
//     column within the shared spill budget until finalize selects the order
//     statistics the grid reads
// ---------------------------------------------------------------------------
class QuantileTransform final : public Kernel {
 public:
  explicit QuantileTransform(const nlohmann::json& params) {
    if (params.contains("columns") && !params.at("columns").is_null()) {
      columns_ = column_list(params, "columns", "quantile_transform");
      explicit_columns_ = true;
      inputs = columns_;
      outputs = columns_;
    }
    if (params.contains("n_quantiles") && !params.at("n_quantiles").is_null()) {
      const nlohmann::json& count = params.at("n_quantiles");
      if (!count.is_number_integer() ||
          (count.is_number_unsigned() && count.get<uint64_t>() > static_cast<uint64_t>(kMaxQuantiles)) ||
          count.get<int64_t>() < 1 || count.get<int64_t>() > kMaxQuantiles) {
        throw Error("quantile_transform: 'n_quantiles' must be an integer between 1 and " +
                    std::to_string(kMaxQuantiles));
      }
      quantiles_ = static_cast<std::size_t>(count.get<int64_t>());
    }
    if (params.contains("output_distribution") && !params.at("output_distribution").is_null()) {
      const nlohmann::json& distribution = params.at("output_distribution");
      if (!distribution.is_string() ||
          (distribution.get<std::string>() != "uniform" && distribution.get<std::string>() != "normal")) {
        throw Error("quantile_transform: 'output_distribution' must be 'uniform' or 'normal'");
      }
      normal_ = distribution.get<std::string>() == "normal";
    }
  }

  std::string id() const override { return "quantile_transform"; }

  nlohmann::json params() const override {
    return {{"columns", explicit_columns_ ? nlohmann::json(columns_) : nlohmann::json()},
            {"n_quantiles", quantiles_},
            {"output_distribution", normal_ ? "normal" : "uniform"}};
  }

  void resolve(Schema& schema) override {
    bind_numeric(schema, stage, columns_, explicit_columns_, fitted, "transform");
    for (auto& [name, type] : schema.cols) {
      if (std::find(columns_.begin(), columns_.end(), name) != columns_.end()) type = "f64";
    }
    inputs = columns_;
    outputs = columns_;
  }

  bool needs_fit() const override { return true; }
  std::unique_ptr<AccState> make_state() const override { return std::make_unique<Values>(); }

  void consume(AccState& raw, const Batch& batch) const override {
    consume_values(columns_, stage, true, true, static_cast<Values&>(raw), batch);
  }

  void finalize(std::vector<std::unique_ptr<AccState>> states) override {
    std::unique_ptr<Values> merged = merge_values(states, false);
    // Upstream caps the grid at the fit row count — nulls included, because
    // it counts rows of the frame and not values of the column.
    const auto rows = static_cast<std::size_t>(std::max<int64_t>(merged->rows, 0));
    const std::size_t width = std::max<std::size_t>(1, std::min(quantiles_, rows));
    references_ = linspace(0.0, 1.0, width);
    grids_.clear();
    for (const std::string& name : columns_) {
      SpillVector<double>& buffer = merged->buffers[name];
      if (buffer.empty()) {
        throw Error(stage + ": column '" + name + "' has no non-null values to learn from");
      }
      std::vector<double> grid;
      grid.reserve(width);
      std::vector<std::size_t> ranks;
      ranks.reserve(width * 2);
      for (double reference : references_) {
        linear_quantile_ranks(buffer.size(), reference * 100.0 / 100.0, ranks);
      }
      select_numeric_ranks(buffer, std::move(ranks));
      // Upstream passes each reference times 100 as a percentile, which numpy
      // divides by 100 again, so the round trip is reproduced rather than
      // shortened away.
      for (double reference : references_) {
        grid.push_back(fitted_number(quantile_at(buffer, reference * 100.0 / 100.0, kMethod), stage, name, "quantile"));
      }
      grids_.emplace(name, std::move(grid));
      buffer.release();
    }
    index_grids();
    fitted = true;
  }

  nlohmann::json state() const override {
    return {{"columns", columns_}, {"references", references_}, {"quantiles", number_lists(columns_, stage, grids_)}};
  }

  void load_state(const nlohmann::json& state) override {
    grids_ = load_number_lists(state, "quantiles", stage, columns_, explicit_columns_);
    if (!state.contains("references") || !state.at("references").is_array() || state.at("references").empty()) {
      throw Error(stage + ": state must hold a non-empty 'references' list");
    }
    // references is linspace(0, 1, n) and nothing else: it is written out so
    // the artifact says what grid it holds, and checked so a hand-edited one
    // cannot silently shift every transformed value.
    const std::vector<double> expected = linspace(0.0, 1.0, state.at("references").size());
    references_.clear();
    for (const nlohmann::json& value : state.at("references")) {
      if (!value.is_number()) throw Error(stage + ": 'references' must hold numbers");
      references_.push_back(value.get<double>());
    }
    if (references_ != expected) {
      throw Error(stage + ": 'references' must be " + std::to_string(references_.size()) +
                  " points evenly spaced from 0 to 1");
    }
    for (const std::string& name : columns_) {
      const std::vector<double>& grid = grids_.at(name);
      if (grid.size() != references_.size()) {
        throw Error(stage + ": quantiles['" + name + "'] must hold one value per reference");
      }
      if (!std::is_sorted(grid.begin(), grid.end())) {
        throw Error(stage + ": quantiles['" + name + "'] must not decrease");
      }
    }
    index_grids();
    inputs = columns_;
    outputs = columns_;
    fitted = true;
  }

  Batch transform(Batch batch) const override {
    for (const std::string& name : columns_) {
      const int idx = batch.find(name);
      if (idx < 0) throw Error(stage + ": column '" + name + "' missing from batch");
      const arrow::Array& array = *batch.cols[static_cast<std::size_t>(idx)];
      // Both grids are the same for every row, so the two tables are read
      // once here rather than once per value inside rank_of().
      const std::vector<double>& grid = grids_.at(name);
      const std::vector<double>& mirrored_grid = mirrored_grids_.at(name);
      const std::string where = stage + ": column '" + name + "'";
      PrimitiveOutput<arrow::DoubleType> output(array.length(), where);
      double* ranks = output.mutable_values();
      for_each_numeric(array, stage, name, [&](int64_t row, bool is_null, double value) {
        if (is_null) {
          output.set_null(row);
          return;
        }
        if (!std::isfinite(value)) refuse_non_finite(value, stage + ": column '" + name + "'");
        ranks[row] = rank_of(grid, mirrored_grid, value);
      });
      batch.cols[static_cast<std::size_t>(idx)] = output.finish();
    }
    return batch;
  }

  Row transform_row(Row row) const override {
    RecordLookup lookup(row, columns_.size());
    for (const std::string& name : columns_) {
      RowValue* value = lookup.find(name);
      if (value == nullptr) throw Error(stage + ": key '" + name + "' missing from record");
      const std::optional<double> number = row_number(*value, stage, name);
      if (number.has_value() && !std::isfinite(*number)) refuse_non_finite(*number, stage + ": key '" + name + "'");
      *value = number.has_value() ? RowValue(rank_of(grids_.at(name), mirrored_grids_.at(name), *number))
                                  : RowValue(std::monostate{});
    }
    return row;
  }

 private:
  // A grid of more points than this would be a fitted table nobody wants, and
  // every point costs a number in the manifest for every column.
  static constexpr int64_t kMaxQuantiles = 100000;
  // np.nanpercentile's default, which is what upstream's _dense_fit calls.
  static constexpr QuantileMethod kMethod = QuantileMethod::Linear;
  // Upstream's BOUNDS_THRESHOLD: how near a fitted bound counts as being on
  // it when the output is normal, and how far the clipped output may go.
  static constexpr double kBoundsThreshold = 1e-7;

  // The reversed, negated grid the backward interpolation reads, plus the
  // clip bounds. Frozen by the fit, so the record path never reads a member
  // resolve() rewrites.
  void index_grids() {
    mirrored_references_ = mirrored(references_);
    mirrored_grids_.clear();
    for (const std::string& name : columns_) mirrored_grids_.emplace(name, mirrored(grids_.at(name)));
    const double edge = kBoundsThreshold - std::numeric_limits<double>::epsilon();
    clip_low_ = normal_ppf(edge);
    clip_high_ = normal_ppf(1.0 - edge);
  }

  static std::vector<double> mirrored(const std::vector<double>& values) {
    std::vector<double> out;
    out.reserve(values.size());
    for (auto value = values.rbegin(); value != values.rend(); ++value) out.push_back(-*value);
    return out;
  }

  // Upstream's _transform_col for one value: interpolate both ways and take
  // the mean, pin the bounds, then map through the output distribution.
  double rank_of(const std::vector<double>& grid, const std::vector<double>& mirrored_grid, double value) const {
    const bool at_low = normal_ ? value - kBoundsThreshold < grid.front() : value == grid.front();
    const bool at_high = normal_ ? value + kBoundsThreshold > grid.back() : value == grid.back();
    const auto [forward, backward] = interpolate_pair(value, grid, references_, mirrored_grid, mirrored_references_);
    double rank = 0.5 * (forward - backward);
    if (at_high) rank = 1.0;
    if (at_low) rank = 0.0;  // a constant column is at both bounds, and upstream pins the low one last
    if (!normal_) return rank;
    return std::clamp(normal_ppf(rank), clip_low_, clip_high_);
  }

  std::vector<std::string> columns_;
  bool explicit_columns_ = false;
  std::size_t quantiles_ = 1000;
  bool normal_ = false;
  std::vector<double> references_;  // linspace(0, 1, n_quantiles_), shared by every column
  std::unordered_map<std::string, std::vector<double>> grids_;
  std::vector<double> mirrored_references_;
  std::unordered_map<std::string, std::vector<double>> mirrored_grids_;
  double clip_low_ = 0.0;
  double clip_high_ = 0.0;
};

// ---------------------------------------------------------------------------
// k_bins_discretize — numeric columns -> the bin each value falls in.
// params: {"columns": [...] | null, "n_bins": int,
//          "encode": "onehot" | "onehot-dense" | "ordinal",
//          "strategy": "uniform" | "quantile" | "kmeans",
//          "quantile_method": one of the nine sklearn accepts}
// Semantics:
//   * the fit records the bin edges of each column: "uniform" spaces
//     n_bins + 1 of them evenly from the column's minimum to its maximum,
//     "quantile" puts them at the percentiles of linspace(0, 100, n_bins + 1)
//     under quantile_method, and "kmeans" runs 1-D Lloyd's algorithm from the
//     midpoints of the "uniform" bins and puts an edge midway between each
//     pair of sorted centers
//   * transform answers how many interior edges a value is at or past, so a
//     value below the first edge is bin 0 and one past the last is the top
//     bin; the outer edges bound the fit data and never split it
//   * "quantile" and "kmeans" then drop every edge that is not more than
//     1e-8 above the one before it, exactly as upstream drops them, so a
//     column can end up with fewer bins than n_bins asked for; a column left
//     with no bin at all is a fit-time error
//   * a constant column is one bin holding everything, as upstream makes it
//   * encode="ordinal" replaces the column with an i64 bin index in place;
//     "onehot" and "onehot-dense" replace it with one i64 column per bin,
//     named <column>_<bin>, so how many columns there are is known only
//     after the fit and a later step cannot name them
//   * a null is an error, at fit and at transform, because upstream refuses
//     one outright; so is a value that is NaN or infinity
//   * columns=null bins every integer or float column that reaches the step;
//     an integer column is widened to f64, so a magnitude past 2^53 rounds
//   * BUFFERED for "quantile" and "kmeans": each worker stores doubles per
//     column within the shared spill budget until finalize sorts them.
//     "uniform" needs only the extremes, so it buffers nothing
// ---------------------------------------------------------------------------
class KBinsDiscretize final : public Kernel {
 public:
  explicit KBinsDiscretize(const nlohmann::json& params) {
    if (params.contains("columns") && !params.at("columns").is_null()) {
      columns_ = column_list(params, "columns", "k_bins_discretize");
      explicit_columns_ = true;
      inputs = columns_;
    }
    if (params.contains("n_bins") && !params.at("n_bins").is_null()) {
      const nlohmann::json& count = params.at("n_bins");
      if (!count.is_number_integer() || (count.is_number_unsigned() && count.get<uint64_t>() > kMaxOutputs) ||
          count.get<int64_t>() < 2 || count.get<int64_t>() > static_cast<int64_t>(kMaxOutputs)) {
        throw Error("k_bins_discretize: 'n_bins' must be an integer between 2 and " + std::to_string(kMaxOutputs));
      }
      bins_ = static_cast<std::size_t>(count.get<int64_t>());
    }
    if (params.contains("encode") && !params.at("encode").is_null()) {
      const nlohmann::json& encode = params.at("encode");
      if (!encode.is_string()) throw Error(kEncodeMessage);
      const std::string text = encode.get<std::string>();
      if (text == "ordinal") {
        expanding_ = false;
      } else if (text == "onehot" || text == "onehot-dense") {
        expanding_ = true;
        dense_ = text == "onehot-dense";
      } else {
        throw Error(kEncodeMessage);
      }
    }
    if (params.contains("strategy") && !params.at("strategy").is_null()) {
      const nlohmann::json& strategy = params.at("strategy");
      const std::string message = "k_bins_discretize: 'strategy' must be 'uniform', 'quantile' or 'kmeans'";
      if (!strategy.is_string()) throw Error(message);
      const std::string text = strategy.get<std::string>();
      if (text == "uniform") {
        strategy_ = Strategy::Uniform;
      } else if (text == "quantile") {
        strategy_ = Strategy::Quantile;
      } else if (text == "kmeans") {
        strategy_ = Strategy::KMeans;
      } else {
        throw Error(message);
      }
    }
    if (params.contains("quantile_method") && !params.at("quantile_method").is_null()) {
      if (!params.at("quantile_method").is_string()) {
        throw Error("k_bins_discretize: 'quantile_method' must be one of " + std::string(kQuantileMethods));
      }
      method_ = quantile_method_from(params.at("quantile_method").get<std::string>(), "k_bins_discretize");
    }
  }

  std::string id() const override { return "k_bins_discretize"; }

  nlohmann::json params() const override {
    return {{"columns", explicit_columns_ ? nlohmann::json(columns_) : nlohmann::json()},
            {"n_bins", bins_},
            {"encode", expanding_ ? (dense_ ? "onehot-dense" : "onehot") : "ordinal"},
            {"strategy", strategy_ == Strategy::Uniform    ? "uniform"
                         : strategy_ == Strategy::Quantile ? "quantile"
                                                           : "kmeans"},
            {"quantile_method", quantile_method_name(method_)}};
  }

  void resolve(Schema& schema) override {
    bind_numeric(schema, stage, columns_, explicit_columns_, fitted, "discretize");
    if (!expanding_) {
      for (auto& [name, type] : schema.cols) {
        if (binned(name)) type = "i64";
      }
      inputs = columns_;
      outputs = columns_;
      return;
    }
    // Until the fit has run there are no bins, so the step consumes its
    // columns and emits nothing; the engine resolves again afterwards.
    outputs.clear();
    if (fitted) {
      for (const std::string& name : columns_) {
        for (const std::string& output : names_.at(name)) outputs.push_back(output);
      }
    }
    // The binned columns leave first, so a generated name only collides with
    // a column that actually survives the step.
    std::erase_if(schema.cols, [&](const auto& col) { return binned(col.first); });
    reject_duplicate_outputs(outputs, stage);
    for (const std::string& name : outputs) {
      if (schema.find(name) != nullptr) throw Error(stage + ": output column '" + name + "' already exists in input");
    }
    for (const std::string& name : outputs) schema.cols.emplace_back(name, "i64");
    inputs = columns_;
  }

  bool needs_fit() const override { return true; }
  std::unique_ptr<AccState> make_state() const override { return std::make_unique<Values>(); }

  void consume(AccState& raw, const Batch& batch) const override {
    consume_values(columns_, stage, strategy_ != Strategy::Uniform, false, static_cast<Values&>(raw), batch);
  }

  void finalize(std::vector<std::unique_ptr<AccState>> states) override {
    std::unique_ptr<Values> merged = merge_values(states);
    edges_.clear();
    for (const std::string& name : columns_) {
      if (merged->nulls[name] > 0) {
        throw Error(stage + ": column '" + name + "' has nulls, which k_bins_discretize refuses");
      }
      if (merged->counts[name] == 0) {
        throw Error(stage + ": column '" + name + "' has no values to learn from");
      }
      edges_.emplace(name, edges_of(name, *merged));
      merged->buffers[name].release();
    }
    merged.reset();
    name_outputs();
    fitted = true;
  }

  nlohmann::json state() const override {
    return {{"columns", columns_}, {"bin_edges", number_lists(columns_, stage, edges_)}};
  }

  void load_state(const nlohmann::json& state) override {
    edges_ = load_number_lists(state, "bin_edges", stage, columns_, explicit_columns_);
    for (const std::string& name : columns_) {
      const std::vector<double>& edges = edges_.at(name);
      if (edges.size() < 2) throw Error(stage + ": bin_edges['" + name + "'] must hold at least two edges");
      // Only the interior has to be in order, as bin_of reads nothing else.
      // The outer edges bound the fit data and never split it, and a fit can
      // write either one out of order: a k-means cluster left empty takes a
      // center outside the data, as upstream's does, and over a subnormal span
      // linspace's step can round up enough to put the last interior edge
      // past the maximum, which upstream writes the same way.
      if (!std::is_sorted(edges.begin() + 1, edges.end() - 1)) {
        throw Error(stage + ": bin_edges['" + name + "'] must not decrease between its outer edges");
      }
      // Non-decreasing, not strictly increasing: linspace over a span of a few
      // ulps repeats an edge, and strategy='uniform' keeps those (upstream
      // keeps them too), so a strict check here would refuse a table finalize
      // legitimately produced. bin_of reads a repeated interior edge the way
      // upstream's searchsorted(side='right') does — that bin is empty.
      //
      // What the count means is fixed, though: n_bins + 1 edges at most, since
      // the fit only ever drops edges from that grid.
      if (edges.size() > bins_ + 1) {
        throw Error(stage + ": bin_edges['" + name + "'] holds " + std::to_string(edges.size()) +
                    " edges, more than the " + std::to_string(bins_ + 1) + " that 'n_bins' allows");
      }
    }
    name_outputs();
    inputs = columns_;
    fitted = true;
  }

  Batch transform(Batch batch) const override {
    if (!expanding_) {
      for (const std::string& name : columns_) {
        const int idx = batch.find(name);
        if (idx < 0) throw Error(stage + ": column '" + name + "' missing from batch");
        const arrow::Array& array = *batch.cols[static_cast<std::size_t>(idx)];
        const std::vector<double>& edges = edges_.at(name);
        const std::string where = stage + ": column '" + name + "'";
        PrimitiveOutput<arrow::Int64Type> output(array.length(), where);
        int64_t* indices = output.mutable_values();
        if (const double* values = dense_values(array, edges)) {
          const auto rows = static_cast<std::size_t>(array.length());
          for (std::size_t first = 0; first < rows; first += kBlock) {
            block_bins(values + first, std::min(kBlock, rows - first), edges, indices + first);
          }
        } else {
          read_column(array, name, [&](std::size_t slot, double value) { indices[slot] = bin_of(edges, value); });
        }
        batch.cols[static_cast<std::size_t>(idx)] = output.finish();
      }
      return batch;
    }
    Batch out;
    out.rows = batch.rows;
    for (std::size_t i = 0; i < batch.names.size(); ++i) {
      if (binned(batch.names[i])) continue;
      out.names.push_back(std::move(batch.names[i]));
      out.cols.push_back(std::move(batch.cols[i]));
    }
    for (const std::string& name : columns_) {
      const int idx = batch.find(name);
      if (idx < 0) throw Error(stage + ": column '" + name + "' missing from batch");
      const arrow::Array& array = *batch.cols[static_cast<std::size_t>(idx)];
      const std::vector<std::string>& names = names_.at(name);
      const std::vector<double>& edges = edges_.at(name);
      const std::string where = stage + ": column '" + name + "'";
      const double* values = dense_values(array, edges);
      std::vector<PrimitiveOutput<arrow::Int64Type>> columns;
      columns.reserve(names.size());
      for (std::size_t c = 0; c < names.size(); ++c) columns.emplace_back(array.length(), where, values == nullptr);
      if (values != nullptr) {
        // A block's bins first, then each column's part of the block in one
        // pass of stores, as rank-hot fills its columns: writing a 1 into one
        // of the zero-filled columns per row would scatter the stores across
        // all of them.
        const auto rows = static_cast<std::size_t>(array.length());
        std::array<int64_t, kBlock> bins;
        for (std::size_t first = 0; first < rows; first += kBlock) {
          const std::size_t count = std::min(kBlock, rows - first);
          block_bins(values + first, count, edges, bins.data());
          for (std::size_t c = 0; c < names.size(); ++c) {
            int64_t* column = columns[c].mutable_values() + first;
            const auto bin = static_cast<int64_t>(c);
            for (std::size_t row = 0; row < count; ++row) column[row] = bins[row] == bin ? 1 : 0;
          }
        }
      } else {
        read_column(array, name, [&](std::size_t slot, double value) {
          columns[static_cast<std::size_t>(bin_of(edges, value))].mutable_values()[slot] = 1;
        });
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
    if (!expanding_) {
      for (const std::string& name : columns_) {
        RowValue* value = lookup.find(name);
        if (value == nullptr) throw Error(stage + ": key '" + name + "' missing from record");
        *value = bin_of(name, row_value(*value, name));
      }
      return row;
    }
    Row out;
    // Sized from what the fit froze, never from `outputs`: resolve() rewrites
    // that member under a mutex while this may be running on another thread.
    out.reserve(row.size() + generated_);
    for (auto& [key, value] : row) {
      if (binned(key)) continue;
      // A passthrough field named like a generated column would be dropped
      // silently; the batch path refuses the same shape at resolve.
      if (output_index_.contains(key)) throw Error(stage + ": key '" + key + "' already present in record");
      out.emplace_back(key, std::move(value));
    }
    for (const std::string& name : columns_) {
      const RowValue* value = lookup.find(name);
      if (value == nullptr) throw Error(stage + ": key '" + name + "' missing from record");
      const int64_t bin = bin_of(name, row_value(*value, name));
      const std::vector<std::string>& names = names_.at(name);
      for (std::size_t c = 0; c < names.size(); ++c) {
        out.emplace_back(names[c], static_cast<int64_t>(c) == bin ? int64_t{1} : int64_t{0});
      }
    }
    return out;
  }

 private:
  enum class Strategy { Uniform, Quantile, KMeans };

  static constexpr const char* kEncodeMessage =
      "k_bins_discretize: 'encode' must be 'onehot', 'onehot-dense' or 'ordinal'";
  // Upstream drops an edge that is not more than this above the one before it.
  static constexpr double kMinWidth = 1e-8;
  // sklearn's KMeans defaults: 300 Lloyd iterations and tol = variance * 1e-4.
  static constexpr int kMaxIterations = 300;
  static constexpr double kTolerance = 1e-4;

  bool binned(const std::string& name) const {
    return std::find(columns_.begin(), columns_.end(), name) != columns_.end();
  }

  // Every value of one column, refusing a null the way upstream refuses NaN.
  template <typename F>
  void read_column(const arrow::Array& array, const std::string& name, F&& fn) const {
    for_each_numeric(array, stage, name, [&](int64_t row, bool is_null, double value) {
      if (is_null) throw Error(stage + ": column '" + name + "' has a null, which k_bins_discretize refuses");
      if (!std::isfinite(value)) refuse_non_finite(value, stage + ": column '" + name + "'", false);
      fn(static_cast<std::size_t>(row), value);
    });
  }

  // The values of a float64 column without nulls, all of them finite, binned
  // against at most 16 interior edges, which block_bins can count; nullptr
  // for any other column. That one is read row by row, which refuses its
  // first null or value that is not finite.
  static const double* dense_values(const arrow::Array& array, const std::vector<double>& edges) {
    if (array.type_id() != arrow::Type::DOUBLE || array.null_count() != 0 || edges.size() - 2 > 16) return nullptr;
    const double* values = static_cast<const arrow::DoubleArray&>(array).raw_values();
    const auto rows = static_cast<std::size_t>(array.length());
    bool finite = true;
    for (std::size_t row = 0; row < rows; ++row) finite &= std::isfinite(values[row]);
    return finite ? values : nullptr;
  }

  // The bins of a block of values, small enough to stay in the L1 cache
  // through one pass per interior edge that adds whether each value is at or
  // past it: bin_of's count, in loops that vectorize. The first interior edge
  // sets each bin, and a constant column, with no interior edge, is all bin 0.
  static constexpr std::size_t kBlock = 2048;
  static void block_bins(const double* values, std::size_t count, const std::vector<double>& edges, int64_t* bins) {
    if (edges.size() == 2) {
      std::fill_n(bins, count, 0);
      return;
    }
    for (std::size_t row = 0; row < count; ++row) bins[row] = values[row] >= edges[1] ? 1 : 0;
    for (std::size_t edge = 2; edge + 1 < edges.size(); ++edge) {
      const double at = edges[edge];
      for (std::size_t row = 0; row < count; ++row) bins[row] += values[row] >= at ? 1 : 0;
    }
  }

  double row_value(const RowValue& value, const std::string& name) const {
    const std::optional<double> number = row_number(value, stage, name);
    if (!number.has_value()) throw Error(stage + ": key '" + name + "' is null, which k_bins_discretize refuses");
    if (!std::isfinite(*number)) refuse_non_finite(*number, stage + ": key '" + name + "'", false);
    return *number;
  }

  // How many interior edges the value is at or past: upstream's
  // searchsorted(edges[1:-1], x, side="right"). The interior edges never
  // decrease and the value is not NaN, so that is the number of them at or
  // below it. Up to 16 edges are counted without a branch: a binary search's
  // turns are unpredictable, and over the 4 interior edges of 5 bins they
  // cost most of a transform.
  int64_t bin_of(const std::string& name, double value) const { return bin_of(edges_.at(name), value); }

  static int64_t bin_of(const std::vector<double>& edges, double value) {
    const double* first = edges.data() + 1;
    const std::size_t count = edges.size() - 2;
    if (count > 16) return static_cast<int64_t>(std::upper_bound(first, first + count, value) - first);
    int64_t bin = 0;
    for (std::size_t i = 0; i < count; ++i) bin += first[i] <= value ? 1 : 0;
    return bin;
  }

  // One column's edges, then upstream's pruning of the ones too close
  // together (which "uniform" is spared, as upstream spares it).
  std::vector<double> edges_of(const std::string& name, Values& values) const {
    const double low = values.low.at(name);
    const double high = values.high.at(name);
    // A constant column is one bin holding everything. Upstream records its
    // edges as [-inf, inf], which no JSON manifest can hold; the interior is
    // empty either way, so every value still falls in bin 0.
    std::vector<double> kept;
    if (low == high) {
      kept = {low, high};
    } else if (strategy_ == Strategy::Uniform) {
      kept = linspace(low, high, bins_ + 1);
    } else {
      SpillVector<double>& buffer = values.buffers.at(name);
      std::vector<double> edges;
      if (strategy_ == Strategy::Quantile) {
        for (double level : linspace(0.0, 100.0, bins_ + 1)) {
          edges.push_back(quantile_at(buffer, level / 100.0, method_));
        }
      } else {
        edges = kmeans_edges(name, buffer, low, high);
      }
      kept.push_back(edges.front());
      for (std::size_t i = 1; i < edges.size(); ++i) {
        if (edges[i] - edges[i - 1] > kMinWidth) kept.push_back(edges[i]);
      }
      if (kept.size() < 2) {
        throw Error(stage + ": column '" + name + "' has no bin left: every edge after the first is within " +
                    std::format("{}", kMinWidth) +
                    " of the one before it (use strategy='uniform', which keeps every edge)");
      }
    }
    for (double edge : kept) (void)fitted_number(edge, stage, name, "bin edge");
    return kept;
  }

  // 1-D Lloyd's algorithm, as sklearn's KMeans(init=<uniform midpoints>,
  // n_init=1) runs it: the data is centered on its mean first (upstream
  // centers it for accuracy), a point joins the nearest center by the
  // ||c||^2 - 2xc surrogate with ties going to the lower-numbered center, and
  // the run stops when the labels repeat or the total squared center shift
  // falls to variance * 1e-4. It returns, as upstream's _kmeans_single_lloyd
  // does, the means the stopping iteration just computed and not the centers
  // its assignment used (upstream swaps centers_new into centers before it
  // tests for convergence, then breaks).
  struct KMeansPoint {
    double value;
    uint64_t label;
  };

  std::vector<double> kmeans_edges(const std::string& name, SpillVector<double>& sorted, double low,
                                   double high) const {
    const std::size_t count = sorted.size();
    if (count < bins_) {
      throw Error(stage + ": column '" + name + "' has " + std::to_string(count) + " values, fewer than the " +
                  std::to_string(bins_) + " clusters strategy='kmeans' needs");
    }
    ExactSum total;
    sorted.for_each([&](std::size_t, double value) {
      if (!total.add(value)) throw Error(stage + ": column '" + name + "' sums past the range of f64");
    });
    const double mean = total.value() / static_cast<double>(count);
    // Past 2^500 from the mean the squares, ||c||^2 - 2xc and the center
    // shifts would overflow, as sklearn's do. 1-D k-means is scale-invariant
    // and scaling by a power of two is exact, so such a column runs divided
    // by the power of two that brings its farthest value just under 2^500,
    // which is the same run, bit for bit, as that smaller column's. Nearer
    // the mean nothing is scaled.
    if (std::isinf(high - low)) throw Error(stage + ": column '" + name + "' spans past the range of f64");
    const double largest = std::max(high - mean, mean - low);
    const int exponent = largest >= 0x1p500 ? std::ilogb(largest) - 499 : 0;
    const auto centered = [&](double value) { return std::ldexp(value - mean, -exponent); };
    ExactSum squares;
    sorted.for_each([&](std::size_t, double value) { (void)squares.add(centered(value) * centered(value)); });
    const double tolerance = squares.value() / static_cast<double>(count) * kTolerance;

    // Point and previous label travel together through sequential spill-file
    // passes. The sentinel is outside the validated [0, bins_) label range,
    // so the first assignment cannot report repeated labels.
    SpillVector<KMeansPoint> points;
    sorted.for_each(
        [&](std::size_t, double value) { points.push_back({centered(value), static_cast<uint64_t>(bins_)}); });
    sorted.release();
    std::vector<double> centers;
    const std::vector<double> grid = linspace(low, high, bins_ + 1);
    for (std::size_t k = 0; k < bins_; ++k) {
      // Two neighbors near f64's largest overflow their sum; halving each
      // first gives the same midpoint wherever the sum is finite.
      const double sum = grid[k + 1] + grid[k];
      centers.push_back(centered(std::isinf(sum) ? grid[k + 1] * 0.5 + grid[k] * 0.5 : sum * 0.5));
    }

    std::vector<ExactSum> sums(bins_);
    std::vector<int64_t> sizes(bins_, 0);
    std::vector<double> norms(bins_, 0.0);
    std::vector<double> means(bins_, 0.0);
    ExactSum shift;
    for (int iteration = 0; iteration < kMaxIterations; ++iteration) {
      for (ExactSum& sum : sums) sum.clear();
      std::fill(sizes.begin(), sizes.end(), 0);
      const bool repeated = assign(points, centers, norms, sums, sizes);
      relocate_empty(points, centers, sums, sizes);
      average_centers(sums, sizes, means);
      shift.clear();
      for (std::size_t k = 0; k < bins_; ++k) {
        (void)shift.add((means[k] - centers[k]) * (means[k] - centers[k]));
      }
      const bool converged = repeated || shift.value() <= tolerance;
      centers.swap(means);
      if (converged) break;
    }

    for (double& center : centers) center = std::ldexp(center, exponent) + mean;
    std::sort(centers.begin(), centers.end());
    std::vector<double> edges{low};
    for (std::size_t k = 0; k + 1 < bins_; ++k) edges.push_back((centers[k] + centers[k + 1]) * 0.5);
    edges.push_back(high);
    return edges;
  }

  static bool assign(SpillVector<KMeansPoint>& points, const std::vector<double>& centers, std::vector<double>& norms,
                     std::vector<ExactSum>& sums, std::vector<int64_t>& sizes) {
    for (std::size_t k = 0; k < centers.size(); ++k) norms[k] = centers[k] * centers[k];
    bool repeated = true;
    // Sorted points change cluster only a few times in a pass, so one Adder
    // stays on a cluster's sum from one change to the next. Adding to the
    // sums themselves takes a fit of 1,000,000 lognormal values 242 ms in
    // place of 113 (one thread, Apple M4).
    std::size_t cluster = 0;  // the one whose sum `adder` is on
    ExactSum::Adder adder(sums[cluster]);
    points.rewrite([&](std::size_t, KMeansPoint point) {
      double best = norms[0] - 2.0 * point.value * centers[0];
      std::size_t label = 0;
      for (std::size_t k = 1; k < centers.size(); ++k) {
        const double distance = norms[k] - 2.0 * point.value * centers[k];
        if (distance < best) {
          best = distance;
          label = k;
        }
      }
      repeated = repeated && point.label == static_cast<uint64_t>(label);
      point.label = static_cast<uint64_t>(label);
      // Assignment and accumulation both walk the points in sorted order.
      if (label != cluster) {
        adder.move_to(sums[label]);
        cluster = label;
      }
      (void)adder.add(point.value);
      ++sizes[label];
      return point;
    });
    return repeated;
  }

  // Upstream's _relocate_empty_clusters_dense: an empty cluster is given the
  // point furthest from the center it was assigned to, moving that point's
  // contribution out of the cluster that claimed it. Three details of it
  // decide whole bin layouts; the first two are reproduced here, and the
  // third as closely as a sorted buffer allows.
  //   * `labels` is `const` upstream, so a relocated point keeps the label
  //     the assignment gave it. That is what the next iteration's strict
  //     convergence test compares against, and it also means the cluster a
  //     point is taken from is never itself one of the empty ones.
  //   * a relocation over points that all sit exactly on their center is
  //     pointless, and upstream returns before doing it ("happens when there
  //     are more clusters than non-duplicate samples").
  //   * where several points tie on that distance, upstream's argpartition
  //     picks one of them by the internal state of numpy's introselect over
  //     the rows in file order; we pick the lowest index of the sorted
  //     buffer. No rule on a sorted buffer can reproduce a choice made on an
  //     unsorted one (taking the highest index instead only changes which
  //     cases differ), so this is the one place a kmeans fit can still land
  //     on a different local optimum than upstream's.
  static void relocate_empty(const SpillVector<KMeansPoint>& points, const std::vector<double>& centers,
                             std::vector<ExactSum>& sums, std::vector<int64_t>& sizes) {
    std::vector<std::size_t> empty;
    for (std::size_t k = 0; k < sizes.size(); ++k) {
      if (sizes[k] == 0) empty.push_back(k);
    }
    if (empty.empty()) return;
    struct Candidate {
      double distance;
      std::size_t index;
      KMeansPoint point;
    };
    const auto better = [](const Candidate& left, const Candidate& right) {
      return left.distance != right.distance ? left.distance > right.distance : left.index < right.index;
    };
    const std::size_t wanted = std::min(empty.size(), points.size());
    std::vector<Candidate> candidates;
    candidates.reserve(wanted);
    double furthest = 0.0;
    points.for_each([&](std::size_t index, KMeansPoint point) {
      // Labels were produced from the bounded centers vector by assign().
      const double gap = point.value - centers[static_cast<std::size_t>(point.label)];
      const double distance = gap * gap;
      furthest = std::max(furthest, distance);
      const Candidate candidate{distance, index, point};
      if (candidates.size() < wanted) {
        candidates.push_back(candidate);
        if (candidates.size() == wanted) std::make_heap(candidates.begin(), candidates.end(), better);
      } else if (better(candidate, candidates.front())) {
        std::pop_heap(candidates.begin(), candidates.end(), better);
        candidates.back() = candidate;
        std::push_heap(candidates.begin(), candidates.end(), better);
      }
    });
    if (furthest == 0.0) return;
    std::sort_heap(candidates.begin(), candidates.end(), better);
    for (std::size_t slot = 0; slot < wanted; ++slot) {
      const KMeansPoint point = candidates[slot].point;
      const std::size_t from = static_cast<std::size_t>(point.label);
      (void)sums[from].add(-point.value);
      --sizes[from];
      sums[empty[slot]].clear();
      (void)sums[empty[slot]].add(point.value);
      sizes[empty[slot]] = 1;
    }
  }

  // Upstream's _average_centers. A cluster with points becomes its mean —
  // scaled by the reciprocal rather than divided, because the two round
  // differently often enough to change a later assignment. One still empty
  // after relocation is placed at the biggest cluster's slot, read as that
  // slot stands at the time: already the mean when the biggest cluster comes
  // first, still the unaveraged sum when it comes later. That is where
  // upstream's out-of-range center comes from, and it is load-bearing.
  static void average_centers(const std::vector<ExactSum>& sums, const std::vector<int64_t>& sizes,
                              std::vector<double>& out) {
    for (std::size_t k = 0; k < sums.size(); ++k) out[k] = sums[k].value();
    std::size_t biggest = 0;
    for (std::size_t k = 1; k < sizes.size(); ++k) {
      if (sizes[k] > sizes[biggest]) biggest = k;
    }
    for (std::size_t k = 0; k < out.size(); ++k) {
      out[k] = sizes[k] > 0 ? out[k] * (1.0 / static_cast<double>(sizes[k])) : out[biggest];
    }
  }

  // '<column>_<bin>' per bin, only once the edges are known.
  void name_outputs() {
    names_.clear();
    output_index_.clear();
    generated_ = 0;
    if (!expanding_) return;
    for (const std::string& name : columns_) {
      const std::size_t width = edges_.at(name).size() - 1;
      check_output_width(width, width, stage, name);
      std::vector<std::string> generated;
      for (std::size_t bin = 0; bin < width; ++bin) {
        generated.push_back(name + "_" + std::to_string(bin));
        output_index_.emplace(generated.back(), generated_++);
      }
      names_.emplace(name, std::move(generated));
      // The cap is on the whole step, not on one column: the two sibling
      // expanders bound their total the same way, and it is the total that
      // costs a row-length vector per batch per worker.
      check_output_width(generated_, width, stage, name);
    }
  }

  std::vector<std::string> columns_;
  bool explicit_columns_ = false;
  std::size_t bins_ = 5;
  bool expanding_ = true;
  bool dense_ = false;  // 'onehot-dense' vs 'onehot': the same thing here, kept so params() round-trips
  Strategy strategy_ = Strategy::Quantile;
  QuantileMethod method_ = QuantileMethod::AveragedInvertedCdf;
  std::unordered_map<std::string, std::vector<double>> edges_;
  std::unordered_map<std::string, std::vector<std::string>> names_;
  // Both frozen by the fit, so the record path never reads a resolve-owned
  // member: the generated names, and how many of them there are.
  StringIndex output_index_;
  std::size_t generated_ = 0;
};

}  // namespace

std::unique_ptr<Kernel> make_robust_scale(const nlohmann::json& params) {
  return std::make_unique<RobustScale>(params);
}

std::unique_ptr<Kernel> make_quantile_transform(const nlohmann::json& params) {
  return std::make_unique<QuantileTransform>(params);
}

std::unique_ptr<Kernel> make_k_bins_discretize(const nlohmann::json& params) {
  return std::make_unique<KBinsDiscretize>(params);
}

}  // namespace fastfeaturetransform
