// The numeric substrate the sklearn.preprocessing ports share, as
// categories.hpp is the categorical one: binding the columns a numeric kernel
// touches, visiting a column's values whatever width they arrived as, placing
// the order statistics a quantile reads, and reading a boolean parameter and
// a loaded step's column list.
//
// scale.cpp, binning.cpp and expand.cpp share these, so a column list is bound
// and a column is read one way for the whole family.
//
// What a value means stays with the kernel that reads it. A null, a NaN and
// an integer past 2^53 mean different things to a scaler, to a discretizer
// and to an expander, and each says so in its own words, so the policy is a
// parameter (`widen`) or a check at the call site rather than a rule imposed
// here.
#pragma once

#include <arrow/array.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
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

// The numeric columns a `columns: null` step takes, or the listed ones,
// checked against the incoming schema: categories.hpp's bind_columns for
// numbers. A fitted step's columns come from its state and are never
// re-derived, which is also what keeps resolve() from writing a member the
// record path reads. A step whose `columns` is required passes
// explicit_columns true and never reaches the derivation.
inline void bind_numeric(const Schema& schema, const std::string& stage, std::vector<std::string>& columns,
                         bool explicit_columns, bool fitted, const std::string& what) {
  const auto numeric = [](const std::string& tag) {
    const std::optional<Family> family = family_of_tag(tag);
    return family == Family::Integer || family == Family::Floating;
  };
  if (!fitted && !explicit_columns) {
    columns.clear();
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
