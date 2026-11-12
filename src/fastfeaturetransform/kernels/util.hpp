// Shared kernel helpers: the string maps every categorical kernel keys its
// tables with, the param validators, and the Arrow visitors that hide the
// width and layout of a column behind one callback. Header-only because they
// are mostly templates and every kernel file needs them.
#pragma once

#include <ankerl/unordered_dense.h>
#include <arrow/array.h>
#include <arrow/type.h>
#include <arrow/util/bit_util.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include "fastfeaturetransform/core.hpp"

namespace fastfeaturetransform {

// Flat map with heterogeneous lookup: find(string_view) never constructs a
// std::string, so per-value allocations vanish from the hot loops.
struct TransparentStringHash {
  using is_transparent = void;
  using is_avalanching = void;
  uint64_t operator()(std::string_view value) const noexcept {
    return ankerl::unordered_dense::hash<std::string_view>{}(value);
  }
};

// std::equal_to compares through a call into the platform memcmp, which costs
// more than the comparison itself for a key the size of a category. From 4 to
// 16 bytes, two overlapping loads from each side cover every byte without
// reading past either string; a shorter key compares byte by byte.
struct TransparentStringEqual {
  using is_transparent = void;
  bool operator()(std::string_view left, std::string_view right) const noexcept {
    const std::size_t size = left.size();
    if (size != right.size()) return false;
    if (size > 16) return std::memcmp(left.data(), right.data(), size) == 0;
    if (size >= 8) return same_ends<uint64_t>(left, right);
    if (size >= 4) return same_ends<uint32_t>(left, right);
    for (std::size_t i = 0; i < size; ++i) {
      if (left[i] != right[i]) return false;
    }
    return true;
  }

 private:
  template <typename Word>
  static bool same_ends(std::string_view left, std::string_view right) noexcept {
    const std::size_t tail = left.size() - sizeof(Word);
    Word left_head, left_tail, right_head, right_tail;
    std::memcpy(&left_head, left.data(), sizeof(Word));
    std::memcpy(&left_tail, left.data() + tail, sizeof(Word));
    std::memcpy(&right_head, right.data(), sizeof(Word));
    std::memcpy(&right_tail, right.data() + tail, sizeof(Word));
    return ((left_head ^ right_head) | (left_tail ^ right_tail)) == 0;
  }
};

using StringCounts = ankerl::unordered_dense::map<std::string, int64_t, TransparentStringHash, TransparentStringEqual>;
using StringIndex =
    ankerl::unordered_dense::map<std::string, std::size_t, TransparentStringHash, TransparentStringEqual>;
using StringValues = ankerl::unordered_dense::map<std::string, double, TransparentStringHash, TransparentStringEqual>;

// A kernel can bind many fields of a wide record without rescanning all its
// keys for each column. Values may change while this view lives, but keys and
// positions must remain fixed. Under 128 fields or 16 lookups, the record
// keeps its allocation-free linear lookup; the index is call-local and never
// retained by fitted state.
//
// The linear lookup keeps a cursor on the field after its previous hit and
// tries that field before it scans from the first one. A step usually lists
// its columns in the record's own order, and a column whose field follows
// the previous column's then costs one comparison instead of one per field
// up to its own. Any other lookup costs at most one comparison more than the
// scan. find() moves the cursor, so a lookup belongs to one transform_row
// call on one thread, like the record it reads.
//
// A record that holds one name twice answers with one of the two fields: the
// index and the scan with the first, the cursor with the one it stands on.
template <typename Record>
class RecordLookup {
 public:
  using Pointer = std::conditional_t<std::is_const_v<Record>, const RowValue*, RowValue*>;

  RecordLookup(Record& row, std::size_t lookups) : row_(row), next_(row.size()) {
    if (row.size() < 128 || lookups < 16) return;
    positions_.emplace(row.size());
    for (std::size_t i = 0; i < row.size(); ++i) positions_->try_emplace(row[i].first, i);
  }

  Pointer find(const std::string& name) const {
    if (positions_) {
      const auto found = positions_->find(name);
      return found == positions_->end() ? nullptr : &row_[found->second].second;
    }
    // Names compare through TransparentStringEqual, as the string maps above
    // do: == calls memcmp for every field whose name is as long as the one
    // asked for, and those calls were about half of a scan's time for a
    // record of four-byte names (measured on an Apple M4).
    const TransparentStringEqual same{};
    const std::string_view wanted = name;
    if (next_ < row_.size() && same(row_[next_].first, wanted)) return &row_[next_++].second;
    for (std::size_t i = 0; i < row_.size(); ++i) {
      if (same(row_[i].first, wanted)) {
        next_ = i + 1;
        return &row_[i].second;
      }
    }
    return nullptr;
  }

 private:
  Record& row_;
  // The cursor: the field find() tries first, the one after its previous hit.
  // It starts past the last field, where there is none to try. Before any hit
  // the scan compares the first field anyway, so trying it here as well would
  // cost every lookup that begins elsewhere a second comparison of that field.
  mutable std::size_t next_;
  std::optional<ankerl::unordered_dense::map<std::string_view, std::size_t>> positions_;
};

// The widths a few kernels need above 64 bits: an exact integer sum, a code
// that must be proved to fit in an i64, a digest read as one number, and the
// product ShortKeys folds into a slot.
__extension__ typedef __int128 Int128;            // NOLINT(modernize-use-using): __extension__ needs the typedef form
__extension__ typedef unsigned __int128 UInt128;  // NOLINT(modernize-use-using): likewise

// A perfect hash over up to 64 distinct keys of at most 16 bytes, the shape
// of most categorical vocabularies: seeds under which every key has a slot of
// its own in a table at least sixteen times the key count, each slot holding
// a copy of its key's bytes (at most 1,024 slots of 24 bytes). find() is then
// one hash, one slot and one comparison, with no probing branch to
// mispredict; a hash map probing for the same keys spent about half of a
// transform of plain strings on its first bucket (measured with
// benchmark/profile_kernels.py before this class replaced it).
// A key set that is too large, too long or that no seed separates leaves the
// index empty, and its owner keeps looking keys up its own way.
class ShortKeys {
 public:
  static constexpr std::size_t kMaxKeys = 64;

  // Indexes `keys`, any range of string-like values; find() answers with a
  // key's position in it. A key that repeats leaves the index empty, as no
  // seed can separate it from itself.
  template <typename Keys>
  void index(const Keys& keys) {
    slots_.clear();
    std::vector<std::string_view> views;
    for (const auto& key : keys) {
      views.emplace_back(key);
      if (views.size() > kMaxKeys || views.back().size() > 16) return;
    }
    if (views.empty()) return;
    int bits = 4;
    while ((std::size_t{1} << bits) < views.size() * 16) ++bits;
    std::vector<Slot> slots(std::size_t{1} << bits);
    uint64_t state = 0;
    // A seed works about as often as random slots allow: three times in five
    // for 16 keys and once in eight for 64. After 256 failures in a row,
    // which at 64 keys has odds of about one in 10^16, the index stays empty.
    for (int attempt = 0; attempt < 256; ++attempt) {
      const uint64_t head = next_seed(state);
      const uint64_t tail = next_seed(state);
      std::fill(slots.begin(), slots.end(), Slot{});
      bool distinct = true;
      for (std::size_t i = 0; i < views.size() && distinct; ++i) {
        const Key key = key_of(views[i]);
        Slot& slot = slots[place(key, views[i].size(), head, tail, 64 - bits)];
        distinct = slot.size == kNoKey;
        slot = {key.head, key.tail, static_cast<uint32_t>(views[i].size()), static_cast<int32_t>(i)};
      }
      if (distinct) {
        slots_ = std::move(slots);
        head_seed_ = head;
        tail_seed_ = tail;
        shift_ = 64 - bits;
        return;
      }
    }
  }

  bool empty() const { return slots_.empty(); }

  // The position of `value` among the indexed keys, or -1. Only for an index
  // that is not empty.
  int64_t find(std::string_view value) const {
    if (value.size() > 16) return -1;  // longer than every key
    const Key key = key_of(value);
    const Slot& slot = slots_[place(key, value.size(), head_seed_, tail_seed_, shift_)];
    return slot.size == value.size() && slot.head == key.head && slot.tail == key.tail ? slot.position : -1;
  }

 private:
  static constexpr uint32_t kNoKey = 0xffffffffu;
  struct Key {
    uint64_t head, tail;
  };
  struct Slot {
    uint64_t head = 0;  // the key's first and last eight bytes, or four below
    uint64_t tail = 0;  // eight, overlapping as needed; under four, all in head
    uint32_t size = kNoKey;
    int32_t position = -1;
  };

  // Equal size, head and tail mean equal bytes: the two words cover the whole
  // key, overlapping when it is shorter than the two together, and never read
  // past it.
  static Key key_of(std::string_view value) {
    const std::size_t size = value.size();
    const char* data = value.data();
    Key key{0, 0};
    if (size >= 8) {
      std::memcpy(&key.head, data, 8);
      std::memcpy(&key.tail, data + size - 8, 8);
    } else if (size >= 4) {
      uint32_t head = 0, tail = 0;
      std::memcpy(&head, data, 4);
      std::memcpy(&tail, data + size - 4, 4);
      key.head = head;
      key.tail = tail;
    } else {
      for (std::size_t i = 0; i < size; ++i) {
        key.head |= static_cast<uint64_t>(static_cast<unsigned char>(data[i])) << (8 * i);
      }
    }
    return key;
  }

  // Each word is XORed with a seed word, the size is mixed into the tail's,
  // and the two meet in one 128-bit product folded to 64 bits, whose top bits
  // pick the slot. Whichever bytes or sizes two keys differ in, whether they
  // share a slot depends on the seed. A sum of the tail and the size could
  // not tell "1000" from "10000" under any seed.
  static std::size_t place(Key key, std::size_t size, uint64_t head_seed, uint64_t tail_seed, int shift) {
    const UInt128 product =
        static_cast<UInt128>(key.head ^ head_seed) * (key.tail ^ (tail_seed + size * 0x9E3779B97F4A7C15ull));
    return static_cast<std::size_t>((static_cast<uint64_t>(product) ^ static_cast<uint64_t>(product >> 64)) >> shift);
  }

  // splitmix64: the seeds come from a fixed sequence, so the table depends
  // only on the keys and their order.
  static uint64_t next_seed(uint64_t& state) {
    uint64_t z = state += 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
  }

  std::vector<Slot> slots_;
  uint64_t head_seed_ = 0;
  uint64_t tail_seed_ = 0;
  int shift_ = 64;
};

// A fixed string map's values in the order of a ShortKeys index over its
// keys, built once its owner's table is complete: after a fit, and after a
// load. find() reads the index when there is one and the map otherwise.
template <typename Value>
struct ShortValues {
  ShortKeys keys;
  std::vector<Value> values;

  // `entries` is the map, or any range of (key, value) pairs. More entries
  // than the index can take are not copied at all, and an index left empty
  // keeps no values, so a large map costs nothing beside itself.
  template <typename Entries>
  void index(const Entries& entries) {
    // Fresh objects, not = {}: that assigns an empty list and keeps the old
    // allocation.
    keys = ShortKeys();
    values = std::vector<Value>();
    if (std::size(entries) > ShortKeys::kMaxKeys) return;
    std::vector<std::string_view> names;
    std::vector<Value> found;
    for (const auto& [key, value] : entries) {
      names.emplace_back(key);
      found.push_back(value);
    }
    keys.index(names);
    if (!keys.empty()) values = std::move(found);
  }

  // The value of `key`, or nullptr when it is not one; `map` is the map the
  // index was built from.
  template <typename Map>
  const Value* find(const Map& map, std::string_view key) const {
    if (keys.empty()) return find_in(map, key);
    const int64_t at = keys.find(key);
    return at < 0 ? nullptr : &values[static_cast<std::size_t>(at)];
  }

 private:
  // Out of line, so that find() stays small enough to inline into a row loop.
  template <typename Map>
  [[gnu::noinline]] static const Value* find_in(const Map& map, std::string_view key) {
    const auto found = map.find(key);
    return found == map.end() ? nullptr : &found->second;
  }
};

// One column of a batch by name, or an error naming the stage and the column.
inline const arrow::Array& column_array(const Batch& batch, const std::string& stage, const std::string& name) {
  const int idx = batch.find(name);
  if (idx < 0) throw Error(stage + ": column '" + name + "' missing from batch");
  return *batch.cols[static_cast<std::size_t>(idx)];
}

// Exact float summation: Shewchuk's non-overlapping partials, the algorithm
// behind Python's math.fsum. The running sum is held exactly, so partials
// merged in any order give the same correctly rounded result. add() and
// merge() answer false when the sum leaves f64's range.
struct ExactSum {
  std::vector<double> partials;  // non-overlapping, by increasing magnitude

  [[nodiscard]] bool add(double x) {
    std::size_t kept = 0;
    for (std::size_t i = 0; i < partials.size(); ++i) {
      double y = partials[i];
      if (std::fabs(x) < std::fabs(y)) std::swap(x, y);
      const double hi = x + y;
      const double lo = y - (hi - x);
      if (lo != 0.0) partials[kept++] = lo;
      x = hi;
    }
    partials.resize(kept);
    if (x != 0.0) {
      if (!std::isfinite(x)) return false;
      partials.push_back(x);
    }
    return true;
  }
  [[nodiscard]] bool merge(const ExactSum& other) {
    for (double partial : other.partials) {
      if (!add(partial)) return false;
    }
    return true;
  }
  // The correctly rounded sum of the partials (CPython's final rounding,
  // which makes half-even rounding work across partials).
  double value() const {
    if (partials.empty()) return 0.0;
    std::size_t n = partials.size();
    double hi = partials[--n];
    double lo = 0.0;
    while (n > 0) {
      const double x = hi;
      const double y = partials[--n];
      hi = x + y;
      lo = y - (hi - x);
      if (lo != 0.0) break;
    }
    if (n > 0 && ((lo < 0.0 && partials[n - 1] < 0.0) || (lo > 0.0 && partials[n - 1] > 0.0))) {
      const double y = lo * 2.0;
      const double x = hi + y;
      if (y == x - hi) hi = x;
    }
    return hi;
  }
};

// The exact product of two doubles, folded into an exact sum: the rounded
// product plus the residual fma leaves behind, which together hold the
// product with no bit lost.
[[nodiscard]] inline bool add_product(ExactSum& sum, double left, double right) {
  const double product = left * right;
  if (!sum.add(product)) return false;
  return sum.add(std::fma(left, right, -product));
}

// numpy's linspace, reproduced step for step: every value is start + i*step
// except the last, which is stop exactly. Bin edges and spline knots both come
// out of it, so the last bits have to agree with what the fixture recorded.
//
// numpy falls back to (i/div)*delta + start when the step underflows to zero,
// which is what keeps a span of a few ulps graded rather than flat. Every
// kernel that spaces a grid uses this one copy, so the same subnormal span
// comes out graded from spline_transform and from k_bins_discretize.
//
// One deliberate difference: sklearn's SplineTransformer passes numpy every
// feature's min and max in one call, and numpy switches on whether any
// column's step is zero, so one constant column changes the last bits of
// every other column's grid. Each column is computed on its own here.
inline std::vector<double> linspace(double start, double stop, std::size_t count) {
  std::vector<double> out;
  if (count == 0) return out;
  if (count == 1) return {start};
  const auto div = static_cast<double>(count - 1);
  const double delta = stop - start;
  const double step = delta / div;
  out.reserve(count);
  for (std::size_t i = 0; i < count; ++i) {
    const auto index = static_cast<double>(i);
    out.push_back(step == 0.0 ? (index / div) * delta + start : index * step + start);
  }
  out[count - 1] = stop;
  return out;
}

// Values a summed column may hold: below 2^900 in magnitude, so that no
// running sum of any realistic number of them can leave f64's range (that
// would take 2^124 of them), whatever order the workers add them in.
constexpr double kMaxSummand = 0x1p900;

inline void reject_duplicates(const std::vector<std::string>& columns, const std::string& what) {
  std::vector<std::string> sorted = columns;
  std::sort(sorted.begin(), sorted.end());
  auto dup = std::adjacent_find(sorted.begin(), sorted.end());
  if (dup != sorted.end()) throw Error(what + ": column '" + *dup + "' listed more than once");
}

// An integer param that get<int64_t>() reads as itself. nlohmann parses every
// non-negative integer as unsigned, and get<int64_t>() wraps one from 2^63 up
// to a negative number, so is_number_integer() alone is not enough.
inline bool is_int64(const nlohmann::json& value) {
  return value.is_number_integer() &&
         !(value.is_number_unsigned() &&
           value.get<uint64_t>() > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()));
}

// Params arrive as JSON from a plan or an artifact; checking their shape here
// means a mistake names the kernel and the key instead of surfacing as a
// JSON library error.
inline std::vector<std::string> column_list(const nlohmann::json& params, const char* key, const std::string& what) {
  const std::string message = what + ": '" + std::string(key) + "' must be a non-empty list of column names";
  if (!params.contains(key) || !params.at(key).is_array() || params.at(key).empty()) throw Error(message);
  std::vector<std::string> columns;
  for (const auto& item : params.at(key)) {
    if (!item.is_string()) throw Error(message);
    columns.push_back(item.get<std::string>());
  }
  reject_duplicates(columns, what);
  return columns;
}

// The column sequence a fit resolved, read back out of a fitted state.
//
// A step whose `columns` is null resolves its list from the incoming schema,
// in schema order, and that order is part of the fitted result: an expanding
// encoder names its generated columns walking it, and the '#' that
// disambiguates two colliding generated names is appended walking it too.
// Rebuilding the list from the keys of a fitted JSON object gives
// alphabetical order instead, which would silently move the generated block
// in the output schema and, where two generated names collide, a value from
// one feature name to another, on every save/load and every pickle.
// So `state()` writes the sequence out as a list and load_state reads it back
// here rather than reconstructing it, and a state without the list is
// refused rather than loaded to mean something else.
//
// `table` is the kernel's own per-column fitted object, when it has one: the
// two must name exactly the same columns, or the state either encodes a
// column it holds nothing for or holds a table it never reads. This compares
// their sizes; each caller then looks every column up in the table, which
// checks the names.
inline void load_fitted_columns(const nlohmann::json& state, const std::string& stage,
                                std::vector<std::string>& columns, bool explicit_columns,
                                const nlohmann::json* table = nullptr) {
  const std::string message = stage + ": state must hold a 'columns' list naming the fitted columns in order";
  if (!state.is_object() || !state.contains("columns") || !state.at("columns").is_array() ||
      state.at("columns").empty()) {
    throw Error(message);
  }
  std::vector<std::string> order;
  for (const nlohmann::json& name : state.at("columns")) {
    if (!name.is_string()) throw Error(message);
    order.push_back(name.get<std::string>());
  }
  reject_duplicates(order, stage + ": state 'columns'");
  // The params named the columns, so the state may only agree with them,
  // order included: a manifest where the two disagree describes two steps.
  if (explicit_columns && order != columns) {
    throw Error(stage + ": state 'columns' does not match the 'columns' in params");
  }
  if (table != nullptr && table->size() != order.size()) {
    throw Error(stage + ": state 'columns' does not name the same columns as the fitted table");
  }
  columns = std::move(order);
}

// Whether a row is null, as Array::IsNull answers for every array a kernel
// is given (valid, so that its null count agrees with its bitmap, and never
// a union or run-end encoded), but without going back through the array for
// every row. IsNull loads the bitmap and the offset through the array's
// data, which a loop that stores anything must reload; the dictionary loops
// spent a third of their time there (measured with
// benchmark/profile_kernels.py before this class). A column without nulls
// answers without looking.
class RowNulls {
 public:
  explicit RowNulls(const arrow::Array& array)
      : bits_(array.null_count() == 0 ? nullptr : array.null_bitmap_data()),
        offset_(array.offset()),
        all_(bits_ == nullptr && array.null_count() != 0 && array.null_count() == array.length()) {}
  bool operator()(int64_t row) const {
    return bits_ != nullptr ? !arrow::bit_util::GetBit(bits_, static_cast<uint64_t>(offset_ + row)) : all_;
  }

 private:
  const uint8_t* bits_;
  int64_t offset_;
  bool all_;  // no bitmap, and every row null
};

// Visit every value of a utf8 column as fn(row, is_null, string_view),
// handling both 32-bit (STRING) and 64-bit (LARGE_STRING — polars' default)
// offset layouts. Dictionary columns take the callers' own fast paths.
//
// Without a validity bitmap, IsNull re-reads the array's type and its atomic
// null count on every call, a cost that shows in every categorical transform;
// a column without nulls asks once, as visit_numeric does.
template <typename F>
void for_each_utf8(const arrow::Array& array, const std::string& stage, const std::string& name, F&& fn) {
  if (array.type_id() == arrow::Type::STRING) {
    const auto& strings = static_cast<const arrow::StringArray&>(array);
    const bool nulls = strings.null_count() != 0;
    for (int64_t row = 0; row < strings.length(); ++row) {
      const bool is_null = nulls && strings.IsNull(row);
      fn(row, is_null, is_null ? std::string_view{} : strings.GetView(row));
    }
  } else if (array.type_id() == arrow::Type::LARGE_STRING) {
    const auto& strings = static_cast<const arrow::LargeStringArray&>(array);
    const bool nulls = strings.null_count() != 0;
    for (int64_t row = 0; row < strings.length(); ++row) {
      const bool is_null = nulls && strings.IsNull(row);
      fn(row, is_null, is_null ? std::string_view{} : strings.GetView(row));
    }
  } else {
    throw Error(stage + ": column '" + name + "' is " + array.type()->ToString() + ", expected utf8");
  }
}

// A dictionary array's indices as int32, read straight from their buffer.
// Arrow's GetValueIndex is an out-of-line call that switches on the index
// width for every row, and was about two thirds of resolving a dictionary
// column (measured with benchmark/profile_kernels.py before this function
// replaced it). Parquet's reader gives int32 indices; indices of another
// width, which only the tests' in-memory arrays have, are copied once. Only
// for dictionaries small enough that every valid index fits, which the
// per-entry caches that call this guarantee; a null slot's value is whatever
// its producer left there, and callers skip it.
inline const int32_t* int32_indices(const arrow::DictionaryArray& dictionary, std::vector<int32_t>& copied) {
  const arrow::ArrayData& data = *dictionary.indices()->data();
  if (data.type->id() == arrow::Type::INT32) return data.GetValues<int32_t>(1);
  copied.resize(static_cast<std::size_t>(data.length));
  const auto copy = [&](const auto* in) {
    for (int64_t row = 0; row < data.length; ++row)
      copied[static_cast<std::size_t>(row)] = static_cast<int32_t>(in[row]);
  };
  switch (data.type->id()) {
    case arrow::Type::INT8:
      copy(data.GetValues<int8_t>(1));
      break;
    case arrow::Type::UINT8:
      copy(data.GetValues<uint8_t>(1));
      break;
    case arrow::Type::INT16:
      copy(data.GetValues<int16_t>(1));
      break;
    case arrow::Type::UINT16:
      copy(data.GetValues<uint16_t>(1));
      break;
    case arrow::Type::UINT32:
      copy(data.GetValues<uint32_t>(1));
      break;
    case arrow::Type::INT64:
      copy(data.GetValues<int64_t>(1));
      break;
    case arrow::Type::UINT64:
      copy(data.GetValues<uint64_t>(1));
      break;
    default:
      throw Error("dictionary indices are " + data.type->ToString() + ", expected an integer type");
  }
  return copied.data();
}

// Runs fn(dictionary_entry_index, is_null, string_view) once per dictionary
// entry. Parquet never stores null entries; the tests' in-memory arrays may.
template <typename F>
void for_each_dictionary_entry(const arrow::Array& values, const std::string& stage, const std::string& name, F&& fn) {
  if (values.type_id() == arrow::Type::STRING) {
    const auto& strings = static_cast<const arrow::StringArray&>(values);
    for (int64_t i = 0; i < strings.length(); ++i) {
      fn(i, strings.IsNull(i), strings.IsNull(i) ? std::string_view{} : strings.GetView(i));
    }
  } else if (values.type_id() == arrow::Type::LARGE_STRING) {
    const auto& strings = static_cast<const arrow::LargeStringArray&>(values);
    for (int64_t i = 0; i < strings.length(); ++i) {
      fn(i, strings.IsNull(i), strings.IsNull(i) ? std::string_view{} : strings.GetView(i));
    }
  } else {
    throw Error(stage + ": column '" + name + "' dictionary is " + values.type()->ToString() + ", expected utf8");
  }
}

// Like for_each_utf8, but also walks a dictionary-encoded utf8 column row by
// row (most Parquet strings arrive that way). For kernels without a per-entry
// fast path of their own.
template <typename F>
void for_each_utf8_value(const arrow::Array& array, const std::string& stage, const std::string& name, F&& fn) {
  if (array.type_id() != arrow::Type::DICTIONARY) {
    for_each_utf8(array, stage, name, fn);
    return;
  }
  const auto& dictionary_array = static_cast<const arrow::DictionaryArray&>(array);
  const arrow::Array& values = *dictionary_array.dictionary();
  if (values.type_id() != arrow::Type::STRING && values.type_id() != arrow::Type::LARGE_STRING) {
    throw Error(stage + ": column '" + name + "' dictionary is " + values.type()->ToString() + ", expected utf8");
  }
  // Nulls are asked for once per array, as in for_each_utf8.
  const bool row_nulls = dictionary_array.null_count() != 0;
  const bool entry_nulls = values.null_count() != 0;
  for (int64_t row = 0; row < dictionary_array.length(); ++row) {
    if (row_nulls && dictionary_array.IsNull(row)) {
      fn(row, true, std::string_view{});
      continue;
    }
    const int64_t entry = dictionary_array.GetValueIndex(row);
    const bool entry_null = entry_nulls && values.IsNull(entry);
    std::string_view view;
    if (!entry_null) {
      view = values.type_id() == arrow::Type::STRING
                 ? static_cast<const arrow::StringArray&>(values).GetView(entry)
                 : static_cast<const arrow::LargeStringArray&>(values).GetView(entry);
    }
    fn(row, entry_null, view);
  }
}

template <typename ArrowType, typename F>
void visit_numeric(const arrow::Array& array, F&& fn) {
  const auto& typed = static_cast<const arrow::NumericArray<ArrowType>&>(array);
  if (typed.null_count() == 0) {
    const auto* values = typed.raw_values();
    const int64_t length = typed.length();
    for (int64_t row = 0; row < length; ++row) fn(row, false, values[row]);
    return;
  }
  const RowNulls null_at(typed);
  const auto* values = typed.raw_values();
  const int64_t length = typed.length();
  for (int64_t row = 0; row < length; ++row) {
    if (null_at(row)) {
      fn(row, true, typename ArrowType::c_type{});
    } else {
      fn(row, false, values[row]);
    }
  }
}

// Visit every value of an integer column of any width as
// fn(row, is_null, int64_t) — the numeric twin of for_each_utf8. Widening is
// exact; an unsigned 64-bit value above INT64_MAX does not fit and is an error.
template <typename F>
void for_each_integer(const arrow::Array& array, const std::string& stage, const std::string& name, F&& fn) {
  auto widen = [&](int64_t row, bool is_null, auto value) { fn(row, is_null, static_cast<int64_t>(value)); };
  switch (array.type_id()) {
    case arrow::Type::INT8:
      visit_numeric<arrow::Int8Type>(array, widen);
      break;
    case arrow::Type::INT16:
      visit_numeric<arrow::Int16Type>(array, widen);
      break;
    case arrow::Type::INT32:
      visit_numeric<arrow::Int32Type>(array, widen);
      break;
    case arrow::Type::INT64:
      visit_numeric<arrow::Int64Type>(array, widen);
      break;
    case arrow::Type::UINT8:
      visit_numeric<arrow::UInt8Type>(array, widen);
      break;
    case arrow::Type::UINT16:
      visit_numeric<arrow::UInt16Type>(array, widen);
      break;
    case arrow::Type::UINT32:
      visit_numeric<arrow::UInt32Type>(array, widen);
      break;
    case arrow::Type::UINT64:
      visit_numeric<arrow::UInt64Type>(array, [&](int64_t row, bool is_null, uint64_t value) {
        if (!is_null && value > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
          throw Error(stage + ": column '" + name + "': value " + std::to_string(value) +
                      " does not fit in a 64-bit integer");
        }
        fn(row, is_null, static_cast<int64_t>(value));
      });
      break;
    default:
      throw Error(stage + ": column '" + name + "' is " + array.type()->ToString() + ", expected an integer column");
  }
}

// Visit every value of a float32 or float64 column as fn(row, is_null, double).
template <typename F>
void for_each_floating(const arrow::Array& array, const std::string& stage, const std::string& name, F&& fn) {
  auto widen = [&](int64_t row, bool is_null, auto value) { fn(row, is_null, static_cast<double>(value)); };
  switch (array.type_id()) {
    case arrow::Type::FLOAT:
      visit_numeric<arrow::FloatType>(array, widen);
      break;
    case arrow::Type::DOUBLE:
      visit_numeric<arrow::DoubleType>(array, widen);
      break;
    default:
      throw Error(stage + ": column '" + name + "' is " + array.type()->ToString() +
                  ", expected a floating-point column");
  }
}
}  // namespace fastfeaturetransform
