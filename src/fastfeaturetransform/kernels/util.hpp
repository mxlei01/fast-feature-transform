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
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
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

// An exact sum of doubles. add() and merge() lose no bit, so the same values
// give the same sum in any order and however they are split across sums that
// are merged later, and value() rounds that sum once, correctly. add() and
// merge() answer false when a value is not finite or the sum leaves f64's
// range; the sum then holds nothing a caller can use until clear().
//
// The sum is the exact total of two parts.
//
// The three terms head_, carry_ and spare_. A value is added to head_, what
// that addition rounds away is added to carry_, and what that one rounds
// away is added to spare_. Each addition is a TwoSum, which gives the rounded
// sum and, as a second double, exactly what the rounding dropped. So the
// three terms together grow by exactly the value whenever the third addition
// drops nothing, which is almost always: between them they hold 159 bits,
// and the values of one column seldom span more.
//
// The partials, Shewchuk's non-overlapping doubles by increasing magnitude
// (the algorithm of Python's math.fsum). They take what the third addition
// drops, in a walk over every partial.
//
// One test after the three additions finds every value they cannot finish:
//   * e3 != 0: the third addition dropped e3, which joins the partials. A
//     NaN compares unequal too, and a NaN is what any of the three leaves in
//     e3 after meeting a NaN, an infinity or an overflow.
//   * |s| < 2^1023 fails: the new head is in f64's top binade or is not
//     finite. A head up there can overflow at its next addition while the
//     exact sum is still in range, and three terms that never overflow can
//     total a sum that is out of range, which add() has to refuse. So the
//     head never enters the top binade. From the value that would take it
//     there the partials alone hold the sum, and Shewchuk's step answers
//     false when its largest partial overflows. head_ is NaN from then on:
//     every later value fails the test the same way, and value() and merge()
//     see that the three terms no longer count.
// A TwoSum is exact unless one of its operations overflows. The first cannot
// while the head stays below 2^1023, before and after; an overflow in the
// other two leaves a NaN in e3, which the partials refuse. So a sum whose
// every add() answered true is exact.
class ExactSum {
 public:
  class Adder;

  ExactSum() = default;
  // A copy has partials of its own.
  ExactSum(const ExactSum& other)
      : partials_(other.partials_ ? std::make_unique<std::vector<double>>(*other.partials_) : nullptr),
        head_(other.head_),
        carry_(other.carry_),
        spare_(other.spare_) {}
  ExactSum(ExactSum&&) noexcept = default;
  ExactSum& operator=(const ExactSum& other) { return *this = ExactSum(other); }
  ExactSum& operator=(ExactSum&&) noexcept = default;

  [[nodiscard]] bool add(double x) { return add_to(*this, head_, carry_, spare_, x); }

  [[nodiscard]] bool merge(const ExactSum& other) {
    if (other.partials_) {
      for (double partial : *other.partials_) {
        if (!add(partial)) return false;
      }
    }
    return std::isnan(other.head_) || (add(other.spare_) && add(other.carry_) && add(other.head_));
  }

  void clear() {
    if (partials_) partials_->clear();
    head_ = carry_ = spare_ = 0.0;
  }

  // The correctly rounded sum. The three terms join a copy of the partials
  // by Shewchuk's step, on the stack when the partials are few; the rounding
  // is CPython's, which rounds half to even across partials.
  double value() const {
    std::array<double, 16> local;
    std::vector<double> heap;
    double* parts = local.data();
    std::size_t n = partials_ ? partials_->size() : 0;
    if (n + 3 > local.size()) {
      heap.resize(n + 3);
      parts = heap.data();
    }
    if (partials_) std::copy(partials_->begin(), partials_->end(), parts);
    if (!std::isnan(head_)) {
      const double front[] = {spare_, carry_, head_};
      for (double term : front) n = grow(parts, n, term);
    }
    if (n == 0) return 0.0;
    double hi = parts[--n];
    double lo = 0.0;
    while (n > 0) {
      const double x = hi;
      const double y = parts[--n];
      hi = x + y;
      lo = y - (hi - x);
      if (lo != 0.0) break;
    }
    if (n > 0 && ((lo < 0.0 && parts[n - 1] < 0.0) || (lo > 0.0 && parts[n - 1] > 0.0))) {
      const double y = lo * 2.0;
      const double x = hi + y;
      if (y == x - hi) hi = x;
    }
    return hi;
  }

  // The sum as doubles that total it exactly and depend on its value alone:
  // the first is value(), and each next one is value() of what the ones
  // before leave, until nothing is left. Each is at most 2^-53 of the one
  // before, so there are at most 40. The three terms and the partials do not
  // depend on the value alone: the same values in another order, or merged
  // from other splits, can leave other doubles with the same total. A reader
  // that rounds something it computes from each double, such as a product
  // that underflows, reads these, and its result is then the same for every
  // order and split.
  std::vector<double> terms() const {
    ExactSum rest = *this;
    std::vector<double> all;
    for (double top = rest.value(); top != 0.0; top = rest.value()) {
      all.push_back(top);
      // Taking the rounded sum away cannot fail, short of a sum that add()
      // has refused, whose value is not finite.
      if (!rest.add(-top)) break;
    }
    return all;
  }

 private:
  // The head stays below this in magnitude: f64's top binade starts here.
  static constexpr double kTopBinade = 0x1p1023;

  struct Rounded {
    double sum;   // a + b, rounded
    double lost;  // a + b - sum, exactly
  };

  // Knuth's TwoSum (The Art of Computer Programming, volume 2, section
  // 4.2.2), six operations with no branch: `sum - a` is the part of the
  // rounded sum that came from b, `sum` minus that the part that came from
  // a, and what a and b each have beyond their part is what was lost.
  static Rounded two_sum(double a, double b) {
    const double sum = a + b;
    const double from_b = sum - a;
    return {sum, (a - (sum - from_b)) + (b - from_b)};
  }

  // One value into a sum. `head`, `carry` and `spare` are the sum's own
  // terms, or an Adder's copies of them; s, c and t are their sums with what
  // comes down to each, and e1, e2 and e3 what each of those rounds away.
  static bool add_to(ExactSum& sum, double& head, double& carry, double& spare, double x) {
    const auto [s, e1] = two_sum(head, x);
    const auto [c, e2] = two_sum(carry, e1);
    const auto [t, e3] = two_sum(spare, e2);
    if (e3 != 0.0 || !(std::fabs(s) < kTopBinade)) [[unlikely]] {
      if (!sum.add_slowly(x, s, c, t, e3, head, carry, spare)) return false;
      head = sum.head_;
      carry = sum.carry_;
      spare = sum.spare_;
      return true;
    }
    head = s;
    carry = c;
    spare = t;
    return true;
  }

  // The value that failed add_to's test: x, the three sums s, c and t it
  // gave, what the third dropped, and the terms it was added to, which are
  // passed by value because an Adder holds them outside the sum. Out of line
  // and cold, so that a loop of adds carries only the additions.
  [[gnu::noinline, gnu::cold]] bool add_slowly(double x, double s, double c, double t, double e3, double head,
                                               double carry, double spare) {
    // Only the third addition dropped something.
    if (std::fabs(s) < kTopBinade) {
      head_ = s;
      carry_ = c;
      spare_ = t;
      return join(e3);
    }
    // The first value to take the head to the top binade or past it: the
    // terms before it move to the partials, and so does every value from it.
    if (!std::isnan(head)) {
      head_ = std::numeric_limits<double>::quiet_NaN();
      carry_ = spare_ = 0.0;
      const double before[] = {spare, carry, head};
      for (double term : before) {
        if (!join(term)) return false;
      }
    }
    return join(x);
  }

  // Dekker's Fast2Sum, the step of math.fsum: the same pair in three
  // operations, exact once the larger of the two comes first.
  static Rounded fast_two_sum(double a, double b) {
    if (std::fabs(a) < std::fabs(b)) std::swap(a, b);
    const double sum = a + b;
    return {sum, b - (sum - a)};
  }

  // Shewchuk's step: x joins `size` non-overlapping partials by increasing
  // magnitude, which stay so, and the new count comes back. `parts` has room
  // for one more. Each partial and x make a rounded sum, which moves on as
  // x, and what the rounding dropped, which stays as a partial unless it is
  // zero.
  //
  // Up to seven partials are tested for that zero as they go. In a longer
  // walk over the partials of a column the zeros fall where the processor
  // cannot predict, and a wrong guess costs more than a partial's
  // arithmetic: eight partials or more are walked with no test, each dropped
  // part stored in its partial's place, and the zeros are closed up in a
  // second pass. A StandardScaler fit of 1,000,000 float64 values spread
  // over 160 binades takes 57 ms that way and 74 ms with the test in every
  // walk. The second pass is the dearer of the two where the test is
  // predictable: for walks of one or two partials (values spread over 40
  // binades take 20.0 ms with two passes for every walk, in place of 17.5),
  // and for a Normalizer row of 16 values spread over 600 binades, whose
  // partials never drop a zero (200,000 rows take 90 ms, in place of 82).
  // One thread, Apple M4.
  static constexpr std::size_t kLongWalk = 8;

  static std::size_t grow(double* parts, std::size_t size, double x) {
    std::size_t kept = 0;
    if (size < kLongWalk) {
      for (std::size_t i = 0; i < size; ++i) {
        const auto [sum, lost] = fast_two_sum(x, parts[i]);
        if (lost != 0.0) parts[kept++] = lost;
        x = sum;
      }
      if (x != 0.0) parts[kept++] = x;
      return kept;
    }
    for (std::size_t i = 0; i < size; ++i) {
      const auto [sum, lost] = fast_two_sum(x, parts[i]);
      parts[i] = lost;
      x = sum;
    }
    parts[size] = x;
    for (std::size_t i = 0; i <= size; ++i) {
      const double part = parts[i];
      parts[kept] = part;
      kept += part != 0.0 ? 1 : 0;
    }
    return kept;
  }

  // The same step on the partials themselves. False when the largest partial
  // is not finite: the sum left f64's range, or x was a NaN or an infinity.
  // That partial is then dropped: a caller that goes on adding to a refused
  // sum, as k-means does, would leave one more with every value, for each
  // later value to walk.
  bool join(double x) {
    if (!partials_) partials_ = std::make_unique<std::vector<double>>();
    std::vector<double>& parts = *partials_;
    parts.push_back(0.0);
    const std::size_t kept = grow(parts.data(), parts.size() - 1, x);
    parts.erase(parts.begin() + static_cast<std::ptrdiff_t>(kept), parts.end());
    if (parts.empty() || std::isfinite(parts.back())) return true;
    parts.pop_back();
    return false;
  }

  // Behind a pointer and absent until a value needs them, which almost none
  // does: a sum is then 32 bytes. Held inline they make it 48, and an
  // accumulator such as target_encode's 80 bytes instead of 64, which its
  // loop indexes with a multiply instead of a shift: a fit of 1,000,000 rows
  // on an integer target then takes 6.8 ms instead of 6.6 (Apple M4).
  std::unique_ptr<std::vector<double>> partials_;
  double head_ = 0.0;  // NaN once the partials alone hold the sum
  double carry_ = 0.0;
  double spare_ = 0.0;
};

// One sum's three terms in locals for the length of a loop. add() on the sum
// itself loads and stores them every time, because its rare path is a call
// that can change them; an Adder's copies stay in registers. A StandardScaler
// fit of 1,000,000 normal float64 values takes 6.5 ms through Adders and
// 9.4 ms through add() (one thread, Apple M4). The destructor and move_to()
// write the terms back, so a sum is stale while an Adder is on it: read it
// and add to it again only afterwards.
class ExactSum::Adder {
 public:
  explicit Adder(ExactSum& sum) : sum_(&sum), head_(sum.head_), carry_(sum.carry_), spare_(sum.spare_) {}
  Adder(const Adder&) = delete;
  Adder& operator=(const Adder&) = delete;
  ~Adder() { write_back(); }

  // Leaves the sum it is on, its terms written back, for another one.
  void move_to(ExactSum& sum) {
    write_back();
    sum_ = &sum;
    head_ = sum.head_;
    carry_ = sum.carry_;
    spare_ = sum.spare_;
  }

  [[nodiscard]] bool add(double x) { return add_to(*sum_, head_, carry_, spare_, x); }

  // The same sum as add(), for a value that is likely short, such as an
  // integer: an addition to the head that drops nothing leaves the carry and
  // the spare as they are, and one TwoSum is the whole add. Any other value
  // takes add(), having paid for that TwoSum twice.
  [[nodiscard]] bool add_short(double x) {
    const auto [s, lost] = two_sum(head_, x);
    if (lost == 0.0 && std::fabs(s) < kTopBinade) {
      head_ = s;
      return true;
    }
    return add(x);
  }

 private:
  void write_back() {
    sum_->head_ = head_;
    sum_->carry_ = carry_;
    sum_->spare_ = spare_;
  }

  ExactSum* sum_;
  double head_;
  double carry_;
  double spare_;
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
