// Explicit C++ tests for ExactSum: its value, its terms and its refusals,
// through add(), merge() and an Adder, against Shewchuk's partials written
// out with no fast path, against 128-bit integer sums, and against values
// worked out by hand.

#include "fastfeaturetransform/kernels/util.hpp"

#include <algorithm>
#include <bit>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace fft = fastfeaturetransform;

namespace {

constexpr double kMax = std::numeric_limits<double>::max();
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kInfinity = std::numeric_limits<double>::infinity();

// Shewchuk's partials as Python's math.fsum keeps them, the oracle of these
// tests: every value walks every partial, and nothing is kept beside them.
struct Partials {
  std::vector<double> parts;  // non-overlapping, by increasing magnitude

  // False when the largest partial is not finite, and that one is not kept.
  bool add(double x) {
    std::vector<double> next;
    for (double y : parts) {
      if (std::fabs(x) < std::fabs(y)) std::swap(x, y);
      const double hi = x + y;
      const double lo = y - (hi - x);
      if (lo != 0.0) next.push_back(lo);
      x = hi;
    }
    parts = std::move(next);
    if (x == 0.0) return true;
    if (!std::isfinite(x)) return false;
    parts.push_back(x);
    return true;
  }

  // The correctly rounded sum, by CPython's final rounding.
  double value() const {
    if (parts.empty()) return 0.0;
    std::size_t n = parts.size();
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
};

Partials partials_of(const std::vector<double>& values) {
  Partials sum;
  for (double value : values) REQUIRE(sum.add(value));
  return sum;
}

// The terms of the oracle's sum, as ExactSum::terms() defines them: the
// rounded sum, then the rounded sum of what it leaves, until nothing is left.
std::vector<double> terms_of(Partials rest) {
  std::vector<double> all;
  for (double top = rest.value(); top != 0.0; top = rest.value()) {
    all.push_back(top);
    REQUIRE(rest.add(-top));
  }
  return all;
}

fft::ExactSum sum_of(const std::vector<double>& values) {
  fft::ExactSum sum;
  for (double value : values) REQUIRE(sum.add(value));
  return sum;
}

uint64_t bits(double value) { return std::bit_cast<uint64_t>(value); }

std::vector<uint64_t> bits(const std::vector<double>& values) {
  std::vector<uint64_t> out;
  for (double value : values) out.push_back(bits(value));
  return out;
}

// The same double, or a NaN each: what a refused sum reads as.
bool alike(double left, double right) { return (std::isnan(left) && std::isnan(right)) || bits(left) == bits(right); }

// splitmix64: the lists below are the same on every run and every compiler.
uint64_t next(uint64_t& state) {
  uint64_t z = state += 0x9E3779B97F4A7C15ull;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

// A double with a full random mantissa and either sign, in [2^exponent,
// 2^(exponent + 1)).
double full(uint64_t& state, int exponent) {
  const double mantissa = 1.0 + static_cast<double>(next(state) >> 12) * 0x1p-52;
  return std::ldexp((next(state) & 1) != 0 ? mantissa : -mantissa, exponent);
}

int between(uint64_t& state, int low, int high) {
  return low + static_cast<int>(next(state) % static_cast<uint64_t>(high - low + 1));
}

// One list of each shape that strains an exact sum, `count` values long.
std::vector<std::pair<std::string, std::vector<double>>> adversarial_lists(uint64_t seed, std::size_t count) {
  uint64_t state = seed;
  double held = 0.0;  // a value one draw keeps for the next
  std::vector<std::pair<std::string, std::vector<double>>> lists;
  const auto list = [&](const std::string& label, const auto& draw) {
    std::vector<double> values;
    for (std::size_t i = 0; i < count; ++i) values.push_back(draw(i));
    lists.emplace_back(label, std::move(values));
  };
  list("one binade", [&](std::size_t) { return full(state, 0); });
  list("sixty binades", [&](std::size_t) { return full(state, between(state, -30, 30)); });
  // Each tier is more than 53 bits under the one above, so three tiers fill
  // the three terms and the fourth and fifth can only go to the partials.
  list("three tiers", [&](std::size_t) { return full(state, -60 * between(state, 0, 2)); });
  list("five tiers", [&](std::size_t) { return full(state, -60 * between(state, 0, 4)); });
  list("every binade", [&](std::size_t) { return full(state, between(state, -1074, 1000)); });
  list("cancellation", [&](std::size_t i) {
    // A large value, small ones, and the large value again with the other
    // sign: what is left is far below what the head once held.
    if (i % 4 == 0) held = full(state, between(state, 100, 900));
    return i % 4 == 0 ? held : i % 4 == 3 ? -held : full(state, between(state, -40, 40));
  });
  // One value in f64's top binade, which moves the sum to the partials, and
  // values too small to take any order or split of the list out of range.
  list("one value in the top binade",
       [&](std::size_t i) { return i == count / 2 ? 0x1.4p1023 : full(state, between(state, 900, 1000)); });
  list("subnormals and zeros", [&](std::size_t i) {
    const double small = std::ldexp(static_cast<double>(next(state) >> 12), -1074);
    const double choices[] = {0.0, -0.0, small, -small, 0x1p-1022, 0x1p-1074, -0x1p-1074, full(state, -1040)};
    return choices[i % 8];
  });
  list("halfway cases", [&](std::size_t) {
    const double choices[] = {1.0,       -1.0,   0x1p-53,  -0x1p-53,  0x1p-54,    0x1.8p-54, 0x1p-106,
                              -0x1p-107, 0x1p53, 0x1p-159, -0x1p-160, 0x1.8p-160, 3.0,       1.0 + 0x1p-52};
    return choices[next(state) % 14];
  });
  list("integers", [&](std::size_t) { return static_cast<double>(between(state, -100000, 100000)); });
  list("squares and their residuals", [&](std::size_t i) {
    if (i % 2 == 0) held = full(state, between(state, -520, 500));
    const double product = held * held;
    return i % 2 == 0 ? product : std::fma(held, held, -product);
  });
  return lists;
}

// The sum through an Adder: add() for values at even positions and
// add_short() for the others when `mixed`, else add_short() for all.
std::optional<fft::ExactSum> through_adder(const std::vector<double>& values, bool mixed) {
  fft::ExactSum sum;
  bool held = true;
  {
    fft::ExactSum::Adder adder(sum);
    for (std::size_t i = 0; i < values.size() && held; ++i) {
      held = mixed && i % 2 == 0 ? adder.add(values[i]) : adder.add_short(values[i]);
    }
  }
  if (!held) return std::nullopt;
  return sum;
}

// The index of the first value a sum refuses, or nullopt when it takes all.
template <typename Sum>
std::optional<std::size_t> first_refusal(Sum& sum, const std::vector<double>& values) {
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (!sum.add(values[i])) return i;
  }
  return std::nullopt;
}

}  // namespace

TEST_CASE("ExactSum rounds the exact sum once, to nearest and half to even") {
  struct Case {
    std::vector<double> values;
    double sum;
  };
  const std::vector<Case> cases{
      {{}, 0.0},
      {{-0.0}, 0.0},
      {{-0.0, -0.0}, 0.0},
      {{1.5}, 1.5},
      {{1.0, -1.0}, 0.0},
      {{1e100, 1.0, -1e100}, 1.0},
      {{1.0, 1e100, -1e100}, 1.0},
      // A tie goes to the even neighbor, 2^53 and not 2^53 + 2.
      {{0x1p53, 1.0}, 0x1p53},
      {{0x1p53 + 2.0, 1.0}, 0x1p53 + 4.0},
      // A third value below the tie decides it: the rounding looks past the
      // two leading terms.
      {{0x1p53, 1.0, 0x1p-60}, 0x1p53 + 2.0},
      {{0x1p53, 1.0, -0x1p-60}, 0x1p53},
      {{0x1p53 + 2.0, 1.0, -0x1p-60}, 0x1p53 + 2.0},
      {{1.0, 0x1p-53, 0x1p-106}, 1.0 + 0x1p-52},
      {{1.0, 0x1p-53, 0x1p-200}, 1.0 + 0x1p-52},
      {{1.0, 0x1p-53, -0x1p-200}, 1.0},
      // Four tiers, one more than the three terms hold: 2^-170 goes to the
      // partials, and it alone breaks the tie that is left when the two
      // values of 2^-110 cancel.
      {{1.0, 0x1p-53, -0x1p-110, 0x1p-170, 0x1p-110}, 1.0 + 0x1p-52},
      {{1.0, 0x1p-53, 0x1p-110, -0x1p-170, -0x1p-110}, 1.0},
      {{1.0, 0x1p-53, -0x1p-110, 0x1p-170, 0x1p-230}, 1.0},
      {{0x1p-1074, 0x1p-1074, -0x1p-1074}, 0x1p-1074},
      {{0x1p-1022, -0x1p-1074}, 0x1p-1022 - 0x1p-1074},
      {{kMax, -kMax, 0x1p-1074}, 0x1p-1074},
      {{0x1p1023, -0x1p970, 0x1p1023 - 0x1p970}, kMax},
  };
  for (const Case& entry : cases) {
    CAPTURE(entry.values);
    // The sign of a zero counts: an empty or cancelled sum is +0.0.
    REQUIRE(bits(sum_of(entry.values).value()) == bits(entry.sum));
    std::vector<double> reversed(entry.values.rbegin(), entry.values.rend());
    REQUIRE(bits(sum_of(reversed).value()) == bits(entry.sum));
    REQUIRE(bits(through_adder(entry.values, true)->value()) == bits(entry.sum));
    REQUIRE(bits(through_adder(entry.values, false)->value()) == bits(entry.sum));
  }
}

TEST_CASE("ExactSum agrees with 128-bit integer sums over long runs of integers") {
  // 53-bit integers shifted by up to 60 bits: every value is a double, and
  // the exact sum of 20,000 of them fits in 128 bits, whose conversion to
  // double rounds correctly.
  uint64_t state = 7;
  for (const int shift : {0, 20, 60}) {
    fft::ExactSum sum;
    fft::ExactSum shorts;
    fft::Int128 exact = 0;
    {
      fft::ExactSum::Adder adder(shorts);
      for (int i = 0; i < 20000; ++i) {
        const auto integer = static_cast<int64_t>(next(state) >> 11) - (int64_t{1} << 52);
        const int scale = between(state, 0, shift);
        const double value = std::ldexp(static_cast<double>(integer), scale);
        exact += static_cast<fft::Int128>(integer) * (fft::Int128{1} << scale);
        REQUIRE(sum.add(value));
        REQUIRE(adder.add_short(value));
        if (i % 997 == 0) {
          CAPTURE(shift, i);
          REQUIRE(bits(sum.value()) == bits(static_cast<double>(exact)));
        }
      }
    }
    REQUIRE(bits(sum.value()) == bits(static_cast<double>(exact)));
    REQUIRE(bits(shorts.value()) == bits(static_cast<double>(exact)));
    REQUIRE(bits(shorts.terms()) == bits(sum.terms()));
  }
  // Small integers never leave the head: the sum is the integer itself.
  fft::ExactSum counts;
  int64_t total = 0;
  {
    fft::ExactSum::Adder adder(counts);
    for (int i = 0; i < 100000; ++i) {
      const int value = between(state, -1000, 1000);
      total += value;
      REQUIRE(adder.add_short(static_cast<double>(value)));
    }
  }
  REQUIRE(counts.value() == static_cast<double>(total));
  REQUIRE(counts.terms() == (total == 0 ? std::vector<double>{} : std::vector<double>{static_cast<double>(total)}));
}

TEST_CASE("ExactSum agrees with plain partials on adversarial values, in any order and split") {
  for (const std::size_t count : {std::size_t{1}, std::size_t{2}, std::size_t{7}, std::size_t{64}, std::size_t{400}}) {
    const auto lists = adversarial_lists(count * 31 + 5, count);
    for (const auto& [label, values] : lists) {
      CAPTURE(label, count);
      const Partials oracle = partials_of(values);
      const fft::ExactSum sum = sum_of(values);
      const std::vector<double> terms = sum.terms();
      REQUIRE(bits(sum.value()) == bits(oracle.value()));

      // The terms total the sum exactly, and each is the correctly rounded
      // value of what the ones before it leave.
      Partials rest = oracle;
      for (double term : terms) {
        REQUIRE(bits(term) == bits(rest.value()));
        REQUIRE(rest.add(-term));
      }
      REQUIRE(rest.parts.empty());

      // An Adder gives the same sum, with either add.
      for (bool mixed : {true, false}) {
        const std::optional<fft::ExactSum> through = through_adder(values, mixed);
        REQUIRE(through.has_value());
        REQUIRE(bits(through->value()) == bits(oracle.value()));
        REQUIRE(bits(through->terms()) == bits(terms));
      }

      // Other orders of the same values.
      std::vector<double> reversed(values.rbegin(), values.rend());
      std::vector<double> rotated = values;
      std::rotate(rotated.begin(), rotated.begin() + static_cast<std::ptrdiff_t>(count / 3), rotated.end());
      for (const std::vector<double>& order : {reversed, rotated}) {
        const fft::ExactSum other = sum_of(order);
        REQUIRE(bits(other.value()) == bits(oracle.value()));
        REQUIRE(bits(other.terms()) == bits(terms));
      }

      // Three workers, merged in every order and both groupings.
      const std::size_t first = count / 3;
      const std::size_t second = count - count / 4;
      const auto cut = [&](std::size_t from, std::size_t to) {
        return std::vector<double>(values.begin() + static_cast<std::ptrdiff_t>(from),
                                   values.begin() + static_cast<std::ptrdiff_t>(to));
      };
      const std::vector<fft::ExactSum> workers{sum_of(cut(0, first)), sum_of(cut(first, second)),
                                               sum_of(cut(second, count))};
      const std::size_t orders[6][3] = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
      for (const auto& order : orders) {
        fft::ExactSum flat;
        for (std::size_t worker : order) REQUIRE(flat.merge(workers[worker]));
        REQUIRE(bits(flat.value()) == bits(oracle.value()));
        REQUIRE(bits(flat.terms()) == bits(terms));
        fft::ExactSum pair = workers[order[1]];
        REQUIRE(pair.merge(workers[order[2]]));
        fft::ExactSum nested = workers[order[0]];
        REQUIRE(nested.merge(pair));
        REQUIRE(bits(nested.value()) == bits(oracle.value()));
        REQUIRE(bits(nested.terms()) == bits(terms));
      }

      // A copy is a sum of its own: it has the same value, and adding to it
      // leaves the original as it was.
      fft::ExactSum copied = sum;
      fft::ExactSum assigned;
      assigned = sum;
      for (fft::ExactSum* copy : {&copied, &assigned}) {
        REQUIRE(bits(copy->value()) == bits(oracle.value()));
        REQUIRE(bits(copy->terms()) == bits(terms));
        REQUIRE(copy->add(0x1p-300));
        REQUIRE(copy->add(0x1p300));
      }
      REQUIRE(bits(sum.value()) == bits(oracle.value()));
      REQUIRE(bits(sum.terms()) == bits(terms));

      // clear() leaves nothing behind, in the terms or in the partials.
      fft::ExactSum cleared = sum;
      cleared.clear();
      REQUIRE(bits(cleared.value()) == bits(0.0));
      REQUIRE(cleared.terms().empty());
      REQUIRE(cleared.add(values.front()));
      REQUIRE(bits(cleared.value()) == bits(values.front() + 0.0));
    }
  }
}

TEST_CASE("ExactSum refuses a sum past f64's range, a value that is not finite, and the steps math.fsum overflows") {
  struct Case {
    std::string label;
    std::vector<double> values;
    std::optional<std::size_t> refused;  // the first value refused
    double sum;                          // when none is
  };
  const std::vector<Case> cases{
      {"NaN", {kNaN}, 0, 0.0},
      {"infinity", {kInfinity}, 0, 0.0},
      {"negative infinity after values", {1.0, 2.0, -kInfinity}, 2, 0.0},
      {"NaN after a spill", {1.0, 0x1p-60, 0x1p-120, 0x1p-180, kNaN}, 4, 0.0},
      {"twice the largest double", {kMax, kMax}, 1, 0.0},
      {"twice the largest double, negative", {-kMax, -kMax}, 1, 0.0},
      {"the largest double alone", {kMax}, std::nullopt, kMax},
      {"under half a step past the largest double", {kMax, 0x1p969}, std::nullopt, kMax},
      // The head never overflows here, and the sum still leaves the range.
      {"half a step past the largest double", {kMax, 0x1p969, 0x1p969}, 2, 0.0},
      {"half a step past, negative", {-kMax, -0x1p969, -0x1p969}, 2, 0.0},
      // The carry holds 2^970 when exact additions bring the head to the
      // largest double.
      {"past the range by the carry", {0x1p1022, 0x1p969, 0x1p969, 0x1.8p1023 - 0x1p971}, 3, 0.0},
      {"in range by a hair", {0x1p1023, 0x1p971 - 0x1p969 - 0x1p960, 0x1p1023 - 0x1p972 + 0x1p970}, std::nullopt, kMax},
      {"through the top binade and back", {kMax, -kMax, 1.0}, std::nullopt, 1.0},
      {"cancelling in the top binade",
       {0x1.8p1023, -0x1p1023, 0x1.4p1023, -0x1.cp1023, 0x1p970},
       std::nullopt,
       0x1p970},
      {"small values under a large head", {0x1.fp1022, 1.0, 0x1p-1074, -1.0, 0x1p900}, std::nullopt, 0x1.fp1022},
      // The head, the carry, the spare and a partial all hold something when
      // the head reaches the top binade.
      {"entering the top binade with three terms",
       {0x1.cp1022, 0x1p960, 0x1p900, 0x1p840, 0x1.cp1022, 0x1p1020},
       std::nullopt,
       0x1.ep1023},
      {"out of range with three terms", {0x1.cp1022, 0x1p960, 0x1p900, 0x1p840, 0x1.cp1022, 0x1p1021}, 5, 0.0},
      // A step of the walk over the partials can overflow while the exact sum
      // rounds to a finite double, as in math.fsum, which refuses these too.
      // Here every running sum is within the largest double, the last one
      // 0x1.8p1023 - 2^970. The partials before the last value are 2^970 and
      // -2^1022, and the largest double meets 2^970 first: their sum, half a
      // step past the largest double, rounds to infinity.
      {"the largest double over partials of both signs", {0x1p971, 0x1.fffffffffffffp1022, -0x1.8p1023, kMax}, 3, 0.0},
      // The sum is 2^1024 - 2^970 - 2^968, under half a step past the largest
      // double, and rounds to it. In this order the partials before the last
      // value are -2^968 and 2^1023: 0x1.fffffffffffffp1022 absorbs the first
      // as a rounding error and meets the second, and that sum, half a step
      // past, rounds to infinity. Last, the small value finds the sum at the
      // largest double and rounds into it.
      {"under half a step past, small value first",
       {0x1.8p969, 0x1.fffffffffffffp1022, 0x1.fffffffffffffp1022},
       2,
       0.0},
      {"under half a step past, small value last",
       {0x1.fffffffffffffp1022, 0x1.fffffffffffffp1022, 0x1.8p969},
       std::nullopt,
       kMax},
  };
  for (const Case& entry : cases) {
    CAPTURE(entry.label);
    // The plain partials refuse the same value.
    Partials oracle;
    REQUIRE(first_refusal(oracle, entry.values) == entry.refused);
    fft::ExactSum sum;
    REQUIRE(first_refusal(sum, entry.values) == entry.refused);
    for (bool mixed : {true, false}) {
      const std::optional<fft::ExactSum> through = through_adder(entry.values, mixed);
      REQUIRE(through.has_value() == !entry.refused.has_value());
      if (through.has_value()) REQUIRE(bits(through->value()) == bits(entry.sum));
    }
    if (entry.refused.has_value()) continue;
    REQUIRE(bits(sum.value()) == bits(entry.sum));
    REQUIRE(bits(sum.value()) == bits(oracle.value()));
    // Every split of the values, merged into a fresh sum, is the same sum.
    for (std::size_t cut = 0; cut <= entry.values.size(); ++cut) {
      fft::ExactSum left;
      fft::ExactSum right;
      bool held = true;
      for (std::size_t i = 0; i < entry.values.size(); ++i) {
        held = held && (i < cut ? left : right).add(entry.values[i]);
      }
      fft::ExactSum merged;
      // A half of the values can leave the range on its own, and so can the
      // halves in this order.
      if (!held || !merged.merge(left) || !merged.merge(right)) continue;
      CAPTURE(cut);
      REQUIRE(bits(merged.value()) == bits(entry.sum));
      REQUIRE(bits(merged.terms()) == bits(sum.terms()));
    }
  }

  // Merging refuses as adding does.
  fft::ExactSum largest = sum_of({kMax});
  REQUIRE_FALSE(largest.merge(sum_of({kMax})));
  fft::ExactSum half = sum_of({0x1p1023});
  REQUIRE_FALSE(half.merge(sum_of({0x1p1022, 0x1p1022})));
  // So does a sum that still holds its three terms, when what it merges is
  // already in the top binade: the merged partials are added like values.
  fft::ExactSum below = sum_of({0x1.8p1022});
  REQUIRE_FALSE(below.merge(sum_of({0x1.8p1023})));
  fft::ExactSum fits = sum_of({0x1p1022, 0x1p-80});
  REQUIRE(fits.merge(sum_of({0x1p1023, -0x1p-80})));
  REQUIRE(fits.value() == 0x1.8p1023);
  // A refused sum is of no use until clear(), which makes it new.
  fft::ExactSum refused;
  REQUIRE_FALSE(refused.add(kInfinity));
  (void)refused.terms();  // returns
  refused.clear();
  REQUIRE(refused.add(2.5));
  REQUIRE(refused.value() == 2.5);
  REQUIRE(refused.terms() == std::vector<double>{2.5});
}

TEST_CASE("ExactSum drops the partial a refusal leaves and goes on as the plain partials do") {
  // A caller can go on adding to a sum that has answered false: k-means does,
  // to the sum of squared distances its tolerance comes from. The partial
  // that is not finite is dropped at the refusal, as the plain partials drop
  // it, and the sums below then answer and read as those do, whatever is
  // added. Were it kept, the sum would gain a partial with every later
  // value, and every value walks them all: 240,000 values after the refusal
  // then take a k-means fit of 17,017,224 values 21.8 s in place of 1.5 (one
  // thread, Apple M4).
  std::vector<std::vector<double>> refused{{kInfinity},
                                           {-kInfinity},
                                           {kNaN},
                                           {1.0, 2.0, -kInfinity},
                                           {1.0, 0x1p-60, 0x1p-120, 0x1p-180, kNaN},
                                           {kMax, kMax},
                                           {-kMax, -kMax},
                                           {kMax, 0x1p969, 0x1p969},
                                           {0x1.8p1022, 0x1p-80, kMax}};
  // Twelve powers of two 54 binades apart are twelve partials once the NaN
  // moves the three terms to them: the later values take the long walk.
  refused.emplace_back();
  for (int exponent = 600; exponent > 600 - 12 * 54; exponent -= 54)
    refused.back().push_back(std::ldexp(1.0, exponent));
  refused.back().push_back(kNaN);
  uint64_t state = 11;
  for (const std::vector<double>& values : refused) {
    CAPTURE(values);
    fft::ExactSum sum;
    Partials oracle;
    REQUIRE(first_refusal(sum, values) == std::optional<std::size_t>(values.size() - 1));
    REQUIRE(first_refusal(oracle, values) == std::optional<std::size_t>(values.size() - 1));
    REQUIRE(alike(sum.value(), oracle.value()));
    for (int i = 0; i < 2000; ++i) {
      const double value = full(state, between(state, -60, 1000));
      CAPTURE(i, value);
      REQUIRE(sum.add(value) == oracle.add(value));
      REQUIRE(alike(sum.value(), oracle.value()));
    }
  }
  // An infinity offered to an empty sum leaves nothing behind.
  fft::ExactSum empty;
  REQUIRE_FALSE(empty.add(kInfinity));
  REQUIRE(bits(empty.value()) == bits(0.0));
  REQUIRE(empty.add(1.5));
  REQUIRE(empty.value() == 1.5);
}

TEST_CASE("ExactSum terms depend on the sum's value alone") {
  // The partials do not: 2^53 rounds 3 + 2^-50 apart into 4 and -1 + 2^-50,
  // and cancelling 2^53 afterwards leaves the two pieces.
  const double small = 3.0 + 0x1p-50;
  const Partials split = partials_of({0x1p53, small, -0x1p53});
  const Partials whole = partials_of({0x1p53, -0x1p53, small});
  REQUIRE(split.parts == std::vector<double>{-1.0 + 0x1p-50, 4.0});
  REQUIRE(whole.parts == std::vector<double>{small});
  REQUIRE(sum_of({0x1p53, small, -0x1p53}).terms() == std::vector<double>{small});
  REQUIRE(sum_of({0x1p53, -0x1p53, small}).terms() == std::vector<double>{small});
  fft::ExactSum merged = sum_of({0x1p53, small});
  REQUIRE(merged.merge(sum_of({-0x1p53})));
  REQUIRE(merged.terms() == std::vector<double>{small});

  // A tie that a lower term breaks: the leading term is the rounded sum, not
  // the leading partial.
  REQUIRE(partials_of({0x1p53, 1.0, 0x1p-60}).parts == std::vector<double>{0x1p-60, 1.0, 0x1p53});
  REQUIRE(sum_of({0x1p53, 1.0, 0x1p-60}).terms() == std::vector<double>{0x1p53 + 2.0, -1.0, 0x1p-60});

  REQUIRE(fft::ExactSum().terms().empty());
  REQUIRE(sum_of({-0.0}).terms().empty());
  REQUIRE(sum_of({-2.5}).terms() == std::vector<double>{-2.5});

  // Powers of two 54 binades apart: none rounds into the one above, so each
  // is a term of its own, 39 of them from 2^1022 down. The same from 2^1023
  // down keeps the whole sum in the partials.
  for (const int top : {1022, 1023}) {
    std::vector<double> powers;
    for (int exponent = top; exponent >= -1074; exponent -= 54) powers.push_back(std::ldexp(1.0, exponent));
    REQUIRE(powers.size() == 39);
    std::vector<double> ascending(powers.rbegin(), powers.rend());
    CAPTURE(top);
    REQUIRE(sum_of(powers).terms() == powers);
    REQUIRE(sum_of(ascending).terms() == powers);
    REQUIRE(sum_of(powers).value() == powers.front());
    REQUIRE(through_adder(powers, true)->terms() == powers);
    REQUIRE(through_adder(ascending, false)->terms() == powers);
    // Every length from one power to all of them. From 2^1022 down the three
    // largest are the terms and the rest the partials, and thirteen partials
    // are the most that value() joins the terms to on its stack: lengths 16
    // and 17 are the two sides of that.
    for (std::size_t count = 1; count <= powers.size(); ++count) {
      CAPTURE(count);
      const std::vector<double> leading(powers.begin(), powers.begin() + static_cast<std::ptrdiff_t>(count));
      REQUIRE(sum_of(leading).value() == powers.front());
      REQUIRE(sum_of(leading).terms() == leading);
    }
  }
}

TEST_CASE("ExactSum::Adder writes its terms back, to be opened again or moved to another sum") {
  fft::ExactSum sum;
  Partials oracle;
  uint64_t state = 99;
  // Several Adders in turn on one sum, with spills to the partials inside
  // each, and plain adds between them. The terms are compared as well as the
  // value: a term an Adder drops as it opens a sum seldom moves the rounded
  // sum.
  for (int round = 0; round < 20; ++round) {
    {
      fft::ExactSum::Adder adder(sum);
      for (int i = 0; i < 50; ++i) {
        const double value = full(state, -60 * between(state, 0, 4));
        REQUIRE(oracle.add(value));
        REQUIRE((i % 3 == 0 ? adder.add_short(value) : adder.add(value)));
      }
    }
    REQUIRE(bits(sum.value()) == bits(oracle.value()));
    REQUIRE(bits(sum.terms()) == bits(terms_of(oracle)));
    const double value = full(state, between(state, -200, 200));
    REQUIRE(oracle.add(value));
    REQUIRE(sum.add(value));
  }
  REQUIRE(bits(sum.value()) == bits(oracle.value()));

  // The short add takes the full add for a value the head cannot hold whole.
  fft::ExactSum inexact;
  {
    fft::ExactSum::Adder adder(inexact);
    REQUIRE(adder.add_short(0x1p53));
    REQUIRE(adder.add_short(1.0));
    REQUIRE(adder.add_short(1.0));
  }
  REQUIRE(inexact.value() == 0x1p53 + 2.0);

  // One Adder that moves between three sums, as k-means moves it between
  // clusters: the sum it leaves is whole at once, and each of them ends as
  // if its values had been added to it alone.
  std::vector<fft::ExactSum> sums(3);
  std::vector<Partials> oracles(3);
  {
    std::size_t on = 0;
    fft::ExactSum::Adder adder(sums[on]);
    for (int i = 0; i < 600; ++i) {
      const std::size_t to = i % 7 == 0 ? static_cast<std::size_t>(next(state) % 3) : on;
      if (to != on) {
        adder.move_to(sums[to]);
        REQUIRE(bits(sums[on].value()) == bits(oracles[on].value()));
        REQUIRE(bits(sums[on].terms()) == bits(terms_of(oracles[on])));
        on = to;
      }
      const double value = full(state, -60 * between(state, 0, 4));
      REQUIRE(oracles[on].add(value));
      REQUIRE(adder.add(value));
    }
  }
  for (std::size_t k = 0; k < sums.size(); ++k) {
    CAPTURE(k);
    REQUIRE(bits(sums[k].value()) == bits(oracles[k].value()));
    REQUIRE(bits(sums[k].terms()) == bits(terms_of(oracles[k])));
  }
}
