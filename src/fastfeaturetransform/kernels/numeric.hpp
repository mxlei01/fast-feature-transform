// The numeric substrate the sklearn.preprocessing ports share, as
// categories.hpp is the categorical one: binding the columns a numeric kernel
// touches, visiting a column's values whatever width they arrived as, the
// state of a fit that buffers a column's values, placing the order statistics
// a quantile reads, and reading a boolean parameter and a loaded step's
// column list.
//
// scale.cpp, binning.cpp and expand.cpp share these, so a column list is bound
// and a column is read one way for the whole family. quantile_encode takes
// its target quantiles from linear_quantile too.
//
// What a value means stays with the kernel that reads it. A null, a NaN and
// an integer past 2^53 mean different things to a scaler, to a discretizer
// and to an expander, and each says so in its own words, so the policy is a
// parameter (`widen`, `null_passes`, `exact_integers`) or a check at the call
// site rather than a rule imposed here.
#pragma once

#include <arrow/array.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "fastfeaturetransform/kernel.hpp"
#include "fastfeaturetransform/kernels/cast_table.hpp"
#include "fastfeaturetransform/kernels/util.hpp"
#include "fastfeaturetransform/spill.hpp"

namespace fastfeaturetransform {

// Append the two positions the linear quantile formula reads for q. Pass q
// exactly as that formula will get it, percentile round trip included: even
// q * 100 / 100 can move a rank.
inline void linear_quantile_ranks(std::size_t count, double q, std::vector<std::size_t>& ranks) {
  if (count == 0) return;
  const double index = (static_cast<double>(count) - 1.0) * q;
  const auto previous = static_cast<std::size_t>(std::floor(index));
  ranks.push_back(std::min(previous, count - 1));
  ranks.push_back(std::min(previous + 1, count - 1));
}

// Place requested order statistics exactly where a full sort would put
// them. Inputs must be finite with canonical zeros; equal values then have
// identical bits. Balanced partitioning avoids rescanning every value for
// each quantile. Measurements favor sorting once more than eight distinct
// ranks are needed; keep selection for the few statistics a RobustScaler fits.
// Small inputs and dense grids also retain the ordinary sort.
inline void select_numeric_ranks(std::span<double> values, std::vector<std::size_t> ranks) {
  if (values.empty() || ranks.empty()) return;
  std::sort(ranks.begin(), ranks.end());
  ranks.erase(std::unique(ranks.begin(), ranks.end()), ranks.end());
  constexpr std::size_t kMaxSelectedRanks = 8;
  if (values.size() <= 2048 || ranks.size() > kMaxSelectedRanks || ranks.size() >= values.size() / 16) {
    std::sort(values.begin(), values.end());
    return;
  }
  const auto partition = [&](auto&& self, std::size_t begin, std::size_t end, std::size_t first,
                             std::size_t last) -> void {
    if (first == last) return;
    const std::size_t middle = first + (last - first) / 2;
    const std::size_t rank = ranks[middle];
    std::nth_element(values.begin() + static_cast<std::ptrdiff_t>(begin),
                     values.begin() + static_cast<std::ptrdiff_t>(rank),
                     values.begin() + static_cast<std::ptrdiff_t>(end));
    self(self, begin, rank, first, middle);
    self(self, rank + 1, end, middle + 1, last);
  };
  partition(partition, 0, values.size(), 0, ranks.size());
}

inline void select_numeric_ranks(SpillVector<double>& values, std::vector<std::size_t> ranks) {
  if (values.empty() || ranks.empty()) return;
  if (values.spilled()) {
    values.sort();
  } else {
    select_numeric_ranks(values.mutable_memory(), std::move(ranks));
  }
}

// numpy's default quantile method ("linear") over an ascending sample: with
// h = q * (n - 1), the answer interpolates between the values at floor(h) and
// floor(h) + 1. numpy interpolates from whichever end is nearer (its _lerp),
// reproduced here so that the last bit agrees too. A caller that starts from
// a percentile divides it by 100 before the call: 100 * t / 100 is not always
// t. An empty sample gives 0.
inline double linear_quantile(const SpillVector<double>& sorted, double q) {
  if (sorted.empty()) return 0.0;
  const auto last = static_cast<double>(sorted.size() - 1);
  const double index = q * last;
  if (index <= 0.0) return sorted.front();
  if (index >= last) return sorted.back();
  const auto lower = static_cast<std::size_t>(index);  // index is positive and finite, so this floors
  const double weight = index - static_cast<double>(lower);
  const double left = sorted[lower];
  const double right = sorted[lower + 1];
  const double span = right - left;
  return weight >= 0.5 ? right - span * (1.0 - weight) : left + span * weight;
}

// Visit every value of an integer or float column as fn(row, is_null, double):
// the numeric twin of util.hpp's for_each_utf8_value, with integers and floats
// behind one callback and one type refusal.
//
// `widen` turns an integer into the double the callback sees, and is never
// called for a null. The four-argument overload below widens the way
// sklearn's own float64 cast does, rounding a magnitude past 2^53; a kernel
// that would rather refuse one (the rule Cast follows) passes its own.
template <typename W, typename F>
void for_each_numeric(const arrow::Array& array, const std::string& stage, const std::string& name, W&& widen, F&& fn) {
  switch (array.type_id()) {
    case arrow::Type::FLOAT:
    case arrow::Type::DOUBLE:
      for_each_floating(array, stage, name, std::forward<F>(fn));
      return;
    case arrow::Type::INT8:
    case arrow::Type::INT16:
    case arrow::Type::INT32:
    case arrow::Type::INT64:
    case arrow::Type::UINT8:
    case arrow::Type::UINT16:
    case arrow::Type::UINT32:
    case arrow::Type::UINT64:
      for_each_integer(array, stage, name, [&](int64_t row, bool is_null, int64_t value) {
        fn(row, is_null, is_null ? 0.0 : widen(value));
      });
      return;
    default:
      break;
  }
  throw Error(stage + ": column '" + name + "' is " + array.type()->ToString() +
              ", expected an integer or float column (cast it first)");
}

template <typename F>
void for_each_numeric(const arrow::Array& array, const std::string& stage, const std::string& name, F&& fn) {
  for_each_numeric(array, stage, name, [](int64_t value) { return static_cast<double>(value); }, std::forward<F>(fn));
}

// An integer read as f64. Widening is exact up to 2^53 and an error above it,
// which is the rule Cast follows: sklearn's own validate_data widens an
// integer frame to float64 silently, and a silently rounded id is worse than
// a refusal.
inline double exact_double(int64_t value, const std::string& where) {
  constexpr int64_t kExact = int64_t{1} << 53;
  if (value > kExact || value < -kExact) {
    throw Error(where + ": value " + std::to_string(value) +
                " is beyond 2^53 in magnitude, where f64 no longer holds every integer exactly");
  }
  return static_cast<double>(value);
}

// NaN and infinity are refused wherever these kernels read a value, at fit
// and at transform alike: a null is how a missing value is spelled here (it
// is what upstream's NaN maps onto), a fitted table has to be finite to be
// JSON, and the quantile interpolation has no answer for a value that
// compares false against every bound it is given. Callers build `where`
// outside the loop over the values or only on this path, so the hot loop
// pays a single isfinite. Only a step that passes a null through adds 'write
// a missing value as a null' to the message; k_bins_discretize and the
// expanders leave it out.
//
// The throw is a function of its own: std::format's buffer would otherwise
// give every caller a stack frame and a stack-protector check, and the check
// runs once per value read.
[[noreturn]] inline void refuse_non_finite(double value, const std::string& where, bool null_passes = true) {
  throw Error(where + ": value " + std::format("{}", value) +
              (null_passes ? " is not finite; write a missing value as a null" : " is not finite"));
}

// One worker's view of the fit columns, as consume_values fills it: every
// non-null value when the kernel buffers, and the extremes and counts always.
// The buffers spill past the fit memory budget; a kernel that fits from the
// extremes alone, as k_bins_discretize(strategy="uniform") does, leaves them
// empty.
struct Values final : AccState {
  std::unordered_map<std::string, SpillVector<double>> buffers;
  std::unordered_map<std::string, double> low;
  std::unordered_map<std::string, double> high;
  std::unordered_map<std::string, int64_t> counts;  // non-null values
  std::unordered_map<std::string, int64_t> nulls;
  int64_t rows = 0;
};

// Add one batch to a worker's state. An integer is widened as sklearn's
// float64 cast widens it, rounding a magnitude past 2^53, or refused there
// when `exact_integers` is set, which is the expanders' rule.
inline void consume_values(const std::vector<std::string>& columns, const std::string& stage, bool buffering,
                           bool null_passes, bool exact_integers, Values& state, const Batch& batch) {
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
    // The message's prefix is built once per column and batch, and the
    // extremes and counts stay in locals until the batch's values are read:
    // with both in the lambda, a spline_transform fit of 1,000,000 integers
    // took 10% longer (uniform knots, one thread, Apple M4).
    const std::string where = stage + ": column '" + name + "'";
    double least = low->second;
    double most = high->second;
    int64_t seen = 0;
    int64_t missing = 0;
    const auto take = [&](int64_t, bool is_null, double value) {
      if (is_null) {
        ++missing;
        return;
      }
      if (!std::isfinite(value)) refuse_non_finite(value, where, null_passes);
      // One zero: min() and max() return their first argument on a tie and
      // std::sort is not stable, so -0.0 would make both the extremes and the
      // ends of the sorted buffer depend on the row-group split.
      if (value == 0.0) value = 0.0;
      ++seen;
      least = std::min(least, value);
      most = std::max(most, value);
      if (buffering) buffer.push_back(value);
    };
    if (exact_integers) {
      for_each_numeric(array, stage, name, [&](int64_t value) { return exact_double(value, where); }, take);
    } else {
      for_each_numeric(array, stage, name, take);
    }
    low->second = least;
    high->second = most;
    counted += seen;
    nulls += missing;
  }
}

// Fold every worker into the first, freeing each as it is consumed, then sort
// each buffer unless the caller selects its own ranks: a sorted concatenation
// is the same sequence however the rows were split, so every quantile read
// from it is split-independent. Adopting the first worker's state instead of
// copying into a fresh one never duplicates the largest buffer, which would
// be one whole extra O(rows) buffer at the peak of the fit.
inline std::unique_ptr<Values> merge_values(std::vector<std::unique_ptr<AccState>>& states, bool sort = true) {
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

// The numeric columns a `columns: null` step takes, or the listed ones,
// checked against the incoming schema: categories.hpp's bind_columns for
// numbers, with the same rule for when the list is derived. A fitted step's
// columns come from its state and are never re-derived, which is also what
// keeps resolve() from writing a member the record path reads. A step whose
// `columns` is required passes explicit_columns true and never reaches the
// derivation.
inline void bind_numeric(const Schema& schema, const std::string& stage, std::vector<std::string>& columns,
                         bool explicit_columns, bool fitted, const std::string& what) {
  const auto numeric = [](const std::string& tag) {
    const std::optional<Family> family = family_of_tag(tag);
    return family == Family::Integer || family == Family::Floating;
  };
  if (!fitted && !explicit_columns && columns.empty()) {
    for (const auto& [name, type] : schema.cols) {
      if (numeric(type)) columns.push_back(name);
    }
    if (columns.empty()) throw Error(stage + ": no integer or float columns to " + what);
  }
  for (const std::string& name : columns) {
    const std::string* type = schema.find(name);
    if (type == nullptr) throw Error(stage + ": column '" + name + "' not found in input");
    if (!numeric(*type)) {
      throw Error(stage + ": column '" + name + "' is " + *type +
                  ", expected an integer or float column (cast it first)");
    }
  }
}

// A boolean parameter, with a default for an absent or null key, refusing
// anything else that is not a bool rather than coercing it. `what` names the
// step in the error, so the same helper serves every kernel that takes one.
inline bool bool_param(const nlohmann::json& params, const char* key, bool fallback, const char* what) {
  if (!params.contains(key) || params.at(key).is_null()) return fallback;
  if (!params.at(key).is_boolean()) {
    throw Error(std::string(what) + ": '" + std::string(key) + "' must be true or false");
  }
  return params.at(key).get<bool>();
}

// The columns a loaded step owns: the ordered list state() wrote, checked
// against the fitted table under `key`, which must hold an entry for each.
//
// The order is read rather than rebuilt from the table's keys, which is
// categories.hpp's rule too and is explained at load_fitted_columns: a
// `columns: null` step resolves in schema order and the keys come back
// alphabetical. load_fitted_columns also refuses an empty list: a fitted
// table naming no columns would load as `fitted = true` owning nothing, so
// the step would silently do nothing, and an expanding step would silently
// drop its whole generated feature block from the served frame.
inline void columns_from_state(const nlohmann::json& state, const std::string& key, std::vector<std::string>& columns,
                               bool explicit_columns, const std::string& stage) {
  if (!state.is_object() || !state.contains(key) || !state.at(key).is_object()) {
    throw Error(stage + ": state must hold a '" + key + "' object");
  }
  load_fitted_columns(state, stage, columns, explicit_columns, &state.at(key));
}

}  // namespace fastfeaturetransform
