// Explicit C++ tests for fit memory and spill files: FitMemory's scopes and
// shared budget, SpillVector's reads, rewrites, appends and external sort, the
// cap on open files, and cleanup after a failure.

#include "fastfeaturetransform/spill.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <cmath>
#include <exception>
#include <filesystem>
#include <limits>
#include <thread>
#include <vector>

#include "fastfeaturetransform/test_support.hpp"

namespace fft = fastfeaturetransform;
using Catch::Matchers::ContainsSubstring;
using fft::testing::TempDir;

namespace {

std::size_t file_count(const std::filesystem::path& directory) {
  std::size_t count = 0;
  for (const auto& entry : std::filesystem::directory_iterator(directory)) count += entry.is_regular_file();
  return count;
}

template <typename T>
std::vector<T> contents(const fft::SpillVector<T>& values) {
  std::vector<T> result;
  values.for_each_chunk([&](std::span<const T> chunk) { result.insert(result.end(), chunk.begin(), chunk.end()); });
  return result;
}

}  // namespace

TEST_CASE("fit memory is lazy, shared across nested scopes, and restored on exceptions") {
  TempDir dir;
  const auto original = fft::current_fit_memory();
  auto first = std::make_shared<fft::FitMemory>(fft::FitMemoryOptions{4096, dir.path.string()});
  auto second = std::make_shared<fft::FitMemory>(fft::FitMemoryOptions{0, dir.path.string()});
  {
    fft::FitMemoryScope outer(first);
    REQUIRE(fft::current_fit_memory() == first);
    fft::SpillVector<double> values;
    values.push_back(3.0);
    REQUIRE_FALSE(values.spilled());
    REQUIRE(first->bytes_in_memory() > 0);
    REQUIRE(first->temp_directory().empty());
    REQUIRE(std::filesystem::is_empty(dir.path));
    try {
      fft::FitMemoryScope inner(second);
      REQUIRE(fft::current_fit_memory() == second);
      fft::SpillVector<double> spilled;
      spilled.push_back(4.0);
      REQUIRE(spilled.spilled());
      REQUIRE(second->bytes_in_memory() == 0);
      throw fft::Error("test unwind");
    } catch (const fft::Error&) {
    }
    REQUIRE(fft::current_fit_memory() == first);
    REQUIRE(first->bytes_in_memory() > 0);
    REQUIRE(second->open_files() == 0);
    REQUIRE(std::filesystem::is_empty(second->temp_directory()));
  }
  REQUIRE(fft::current_fit_memory() == original);
  REQUIRE(first->bytes_in_memory() == 0);
  const auto private_directory = second->temp_directory();
  second.reset();
  REQUIRE_FALSE(std::filesystem::exists(private_directory));
}

TEST_CASE("spill buffers preserve floating bits through cached reads, rewrites, and ownership moves") {
  TempDir dir;
  auto memory = std::make_shared<fft::FitMemory>(fft::FitMemoryOptions{256, dir.path.string()});
  fft::SpillVector<double> values(memory);
  std::vector<double> expected;
  const std::array special{0.0,
                           -0.0,
                           std::numeric_limits<double>::denorm_min(),
                           -std::numeric_limits<double>::denorm_min(),
                           std::numeric_limits<double>::max(),
                           -std::numeric_limits<double>::max(),
                           std::nextafter(1.0, 2.0)};
  for (std::size_t index = 0; index < 103; ++index) {
    expected.push_back(special[index % special.size()]);
    values.push_back(expected.back());
    REQUIRE(std::bit_cast<std::uint64_t>(values.back()) == std::bit_cast<std::uint64_t>(expected.back()));
    if (index % 11 == 0) REQUIRE(std::bit_cast<std::uint64_t>(values.front()) == 0);
  }
  REQUIRE(values.spilled());
  REQUIRE(values.mutable_memory().empty());
  REQUIRE(memory->peak_bytes_in_memory() <= 256);
  REQUIRE_THROWS_AS(values[values.size()], fft::Error);
  values.rewrite([](std::size_t index, double value) { return index % 2 == 0 ? -value : value; });
  for (std::size_t index = 0; index < expected.size(); index += 2) expected[index] = -expected[index];
  fft::SpillVector<double> moved(std::move(values));
  REQUIRE(values.empty());
  values.push_back(11.0);  // the moved-from object remains reusable
  REQUIRE(values.front() == 11.0);
  bool exact = true;
  moved.for_each([&](std::size_t index, double value) {
    exact = exact && std::bit_cast<std::uint64_t>(value) == std::bit_cast<std::uint64_t>(expected[index]);
  });
  REQUIRE(exact);
  moved.release();
  REQUIRE(moved.empty());
  REQUIRE_THROWS_AS(moved.front(), fft::Error);
  REQUIRE_THROWS_AS(moved.back(), fft::Error);
  values.clear();
  REQUIRE(memory->bytes_in_memory() == 0);
  REQUIRE(memory->open_files() == 0);
  REQUIRE(std::filesystem::is_empty(memory->temp_directory()));
}

TEST_CASE("spill append combines memory and disk sources in order across memory contexts") {
  TempDir dir;
  for (const std::size_t first_limit : {0U, 4096U}) {
    for (const std::size_t second_limit : {0U, 4096U}) {
      auto first_memory = std::make_shared<fft::FitMemory>(fft::FitMemoryOptions{first_limit, dir.path.string()});
      auto second_memory = std::make_shared<fft::FitMemory>(fft::FitMemoryOptions{second_limit, dir.path.string()});
      fft::SpillVector<std::uint64_t> first(first_memory), second(second_memory);
      std::vector<std::uint64_t> expected;
      for (std::uint64_t index = 0; index < 75; ++index) {
        (index < 31 ? first : second).push_back(index * index);
        expected.push_back(index * index);
      }
      first.append(std::move(second));
      REQUIRE(contents(first) == expected);
      REQUIRE(second.empty());
      REQUIRE(second_memory->bytes_in_memory() == 0);
      REQUIRE(second_memory->open_files() == 0);
      REQUIRE_THROWS_AS(first.append(std::move(first)), fft::Error);
      fft::SpillVector<std::uint64_t> destination(first_memory);
      const auto files = first_memory->spill_files_created();
      destination.append(std::move(first));
      REQUIRE(contents(destination) == expected);
      REQUIRE(first.empty());
      REQUIRE(first_memory->spill_files_created() == files);
    }
  }
}

TEST_CASE("spill rewrite_from supports records and reuses files with no cache budget") {
  struct Point {
    double value;
    std::uint64_t label;
  };
  TempDir dir;
  auto source_memory = std::make_shared<fft::FitMemory>(fft::FitMemoryOptions{4096, dir.path.string()});
  auto work_memory = std::make_shared<fft::FitMemory>(fft::FitMemoryOptions{0, dir.path.string()});
  fft::SpillVector<double> source(source_memory);
  constexpr std::size_t count = 10003;
  for (std::size_t index = 0; index < count; ++index) source.push_back(static_cast<double>(index));
  fft::SpillVector<Point> work(work_memory);
  for (std::uint64_t pass = 1; pass <= 3; ++pass) {
    work.rewrite_from(
        source, [pass](std::size_t index, double value) { return Point{value * static_cast<double>(pass), index}; });
    REQUIRE(work.spilled());
    REQUIRE(work.size() == count);
    REQUIRE(work_memory->bytes_in_memory() == 0);
    REQUIRE(work_memory->spill_files_created() == 1);
    bool exact = true;
    work.for_each([&](std::size_t index, Point value) {
      exact = exact && value.value == static_cast<double>(index * pass) && value.label == index;
    });
    REQUIRE(exact);
  }
  work.rewrite([](std::size_t index, Point value) { return Point{-value.value, index % 7}; });
  REQUIRE(work.back().value == -3.0 * static_cast<double>(count - 1));
  REQUIRE(work.back().label == (count - 1) % 7);
  fft::SpillVector<double> converted(source_memory);
  converted.rewrite_from(work, [](std::size_t, Point point) { return point.value; });
  converted.rewrite_from(converted, [](std::size_t, double value) { return -value; });
  REQUIRE(converted.back() == 3.0 * static_cast<double>(count - 1));
  source.push_back(123.0);
  work.rewrite_from(source, [](std::size_t index, double value) { return Point{value, index}; });
  REQUIRE(work.size() == count + 1);
  REQUIRE(work.back().value == 123.0);
  REQUIRE(work_memory->spill_files_created() == 2);
  REQUIRE(file_count(work_memory->temp_directory()) == 1);
}

TEST_CASE("external spill sort merges multiple passes with bounded files and exact ranks") {
  TempDir dir;
  auto memory = std::make_shared<fft::FitMemory>(fft::FitMemoryOptions{65536, dir.path.string()});
  fft::SpillVector<std::uint64_t> values(memory);
  // More than eight full runs force two merge passes, including partial runs
  // and partial output chunks. The values repeat a multiplicative permutation
  // of 0..100002, so they include ties.
  const auto count = fft::spill_detail::sort_bytes / sizeof(std::uint64_t) * (fft::spill_detail::merge_fanin + 1) + 19;
  std::vector<std::uint64_t> expected;
  expected.reserve(count);
  for (std::size_t index = 0; index < count; ++index) {
    const auto value = static_cast<std::uint64_t>((index * 7919) % 100003);
    values.push_back(value);
    expected.push_back(value);
  }
  std::sort(expected.begin(), expected.end());
  REQUIRE(values.spilled());
  values.sort();
  REQUIRE(memory->spill_files_created() == 2);
  REQUIRE(memory->open_files() == 1);
  REQUIRE(file_count(memory->temp_directory()) == 1);
  REQUIRE(memory->peak_bytes_in_memory() <= 65536);
  bool exact = true;
  values.for_each([&](std::size_t index, std::uint64_t value) { exact = exact && value == expected[index]; });
  REQUIRE(exact);
  for (const std::size_t index : {std::size_t{0}, count / 4, count / 2, count * 3 / 4, count - 1}) {
    REQUIRE(values[index] == expected[index]);
  }
  values.sort(std::greater<>{});
  REQUIRE(values.front() == expected.back());
  REQUIRE(values.back() == expected.front());
  exact = true;
  values.for_each(
      [&](std::size_t index, std::uint64_t value) { exact = exact && value == expected[count - index - 1]; });
  REQUIRE(exact);
}

TEST_CASE("spill sort preserves finite double ordering for subnormals and extremes") {
  TempDir dir;
  auto memory = std::make_shared<fft::FitMemory>(fft::FitMemoryOptions{0, dir.path.string()});
  fft::SpillVector<double> values(memory);
  std::vector<double> expected{std::numeric_limits<double>::max(),
                               std::numeric_limits<double>::denorm_min(),
                               -std::numeric_limits<double>::max(),
                               0.0,
                               -std::numeric_limits<double>::denorm_min(),
                               std::nextafter(1.0, 2.0),
                               1.0,
                               std::nextafter(1.0, 0.0),
                               0.0};
  for (double value : expected) values.push_back(value);
  values.sort();
  std::sort(expected.begin(), expected.end());
  REQUIRE(contents(values) == expected);
}

TEST_CASE("many grouped spill buffers share at most eight open descriptors and clean up") {
  TempDir dir;
  auto memory = std::make_shared<fft::FitMemory>(fft::FitMemoryOptions{0, dir.path.string()});
  std::vector<fft::SpillVector<std::uint64_t>> groups;
  for (std::size_t group = 0; group < 96; ++group) {
    groups.emplace_back(memory);
    groups.back().push_back(group);
    REQUIRE(memory->open_files() <= 8);
  }
  REQUIRE(memory->spill_files_created() == groups.size());
  REQUIRE(file_count(memory->temp_directory()) == groups.size());
  const auto permissions = std::filesystem::status(memory->temp_directory()).permissions();
  REQUIRE((permissions & (std::filesystem::perms::group_all | std::filesystem::perms::others_all)) ==
          std::filesystem::perms::none);
  for (std::size_t group = 0; group < groups.size(); ++group) {
    groups[group].push_back(group + 1000);
    REQUIRE(contents(groups[group]) == std::vector<std::uint64_t>{group, group + 1000});
    REQUIRE(groups[group].front() == group);
    REQUIRE(groups[group].back() == group + 1000);
    REQUIRE(memory->open_files() <= 8);
  }
  const auto private_directory = memory->temp_directory();
  memory.reset();  // buffers retain the context until the final owner releases it
  REQUIRE(std::filesystem::exists(private_directory));
  groups.clear();
  REQUIRE_FALSE(std::filesystem::exists(private_directory));
  REQUIRE(std::filesystem::is_empty(dir.path));
}

TEST_CASE("spill append caches stay small and exhausted targets retain fixed per-group buffers") {
  TempDir dir;
  auto normal = std::make_shared<fft::FitMemory>(fft::FitMemoryOptions{16384, dir.path.string()});
  {
    fft::SpillVector<double> values(normal);
    for (std::size_t index = 0; index < 10003; ++index) values.push_back(static_cast<double>(index));
    REQUIRE(values.spilled());
    REQUIRE(normal->bytes_in_memory() == fft::spill_detail::cache_bytes);
    REQUIRE(normal->fallback_bytes_in_memory() == 0);
    REQUIRE(values.back() == 10002.0);
  }
  REQUIRE(normal->bytes_in_memory() == 0);
  auto exhausted = std::make_shared<fft::FitMemory>(fft::FitMemoryOptions{0, dir.path.string()});
  std::vector<fft::SpillVector<std::uint64_t>> groups;
  for (std::size_t group = 0; group < 12; ++group) {
    groups.emplace_back(exhausted);
    for (std::uint64_t index = 0; index < 7; ++index) groups.back().push_back(index);
  }
  const auto fallback_bytes = fft::spill_detail::fallback_bytes * groups.size();
  REQUIRE(exhausted->bytes_in_memory() == 0);
  REQUIRE(exhausted->fallback_bytes_in_memory() == fallback_bytes);
  // Tiny appends are buffered in memory, even when the memory limit is zero.
  for (const auto& entry : std::filesystem::directory_iterator(exhausted->temp_directory())) {
    REQUIRE(entry.file_size() == 0);
  }
  // A group's 256-byte buffer holds 32 values, so 96 of the 100 reach its file.
  for (std::uint64_t index = 7; index < 100; ++index) {
    for (auto& group : groups) group.push_back(index);
  }
  REQUIRE(exhausted->fallback_bytes_in_memory() == fallback_bytes);
  REQUIRE(exhausted->peak_fallback_bytes_in_memory() == fallback_bytes);
  for (const auto& entry : std::filesystem::directory_iterator(exhausted->temp_directory())) {
    REQUIRE(entry.file_size() == 96 * sizeof(std::uint64_t));
  }
  for (auto& group : groups) {
    std::uint64_t total = 0;
    group.for_each([&](std::size_t, std::uint64_t value) { total += value; });
    REQUIRE(total == 4950);
  }
  fft::SpillVector<std::uint64_t> moved(std::move(groups.front()));
  REQUIRE(exhausted->fallback_bytes_in_memory() == fallback_bytes);
  moved.release();
  REQUIRE(exhausted->fallback_bytes_in_memory() == fallback_bytes - fft::spill_detail::fallback_bytes);
  groups.clear();
  REQUIRE(exhausted->fallback_bytes_in_memory() == 0);
  REQUIRE(exhausted->open_files() == 0);
  REQUIRE(std::filesystem::is_empty(exhausted->temp_directory()));
}

TEST_CASE("worker scopes enforce one RAM budget across independently owned buffers") {
  TempDir dir;
  auto memory = std::make_shared<fft::FitMemory>(fft::FitMemoryOptions{4096, dir.path.string()});
  std::array<std::exception_ptr, 4> errors{};
  std::array<std::uint64_t, 4> totals{};
  std::vector<std::thread> workers;
  for (std::size_t worker = 0; worker < errors.size(); ++worker) {
    workers.emplace_back([&, worker] {
      try {
        fft::FitMemoryScope scope(memory);
        fft::SpillVector<std::uint64_t> values;
        for (std::uint64_t index = 0; index < 2003; ++index) values.push_back(index + worker);
        values.for_each([&](std::size_t, std::uint64_t value) { totals[worker] += value; });
      } catch (...) {
        errors[worker] = std::current_exception();
      }
    });
  }
  for (auto& worker : workers) worker.join();
  for (std::size_t worker = 0; worker < errors.size(); ++worker) {
    REQUIRE(errors[worker] == nullptr);
    REQUIRE(totals[worker] == 2003 * 2002 / 2 + 2003 * worker);
  }
  REQUIRE(memory->peak_bytes_in_memory() <= 4096);
  REQUIRE(memory->bytes_in_memory() == 0);
  REQUIRE(memory->fallback_bytes_in_memory() == 0);
  REQUIRE(memory->spill_files_created() >= 4);
  REQUIRE(memory->open_files() == 0);
  REQUIRE(std::filesystem::is_empty(memory->temp_directory()));
}

TEST_CASE("spill failures identify the temporary path and leave no retained files") {
  TempDir dir;
  auto invalid = std::make_shared<fft::FitMemory>(fft::FitMemoryOptions{0, (dir.path / "missing").string()});
  fft::SpillVector<double> cannot_create(invalid);
  REQUIRE_THROWS_WITH(cannot_create.push_back(1.0), ContainsSubstring("create temporary directory"));
  REQUIRE(invalid->bytes_in_memory() == 0);
  REQUIRE(invalid->temp_directory().empty());
  REQUIRE(cannot_create.empty());
  auto memory = std::make_shared<fft::FitMemory>(fft::FitMemoryOptions{0, dir.path.string()});
  {
    fft::SpillVector<double> values(memory);
    values.push_back(1.0);
    values.rewrite([](std::size_t, double value) { return value; });  // flush and invalidate the append cache
    const auto path = std::filesystem::directory_iterator(memory->temp_directory())->path();
    std::filesystem::resize_file(path, 0);
    REQUIRE_THROWS_WITH(values.front(), ContainsSubstring("unexpected end of file"));
    REQUIRE_THROWS_WITH(values.front(), ContainsSubstring(path.string()));
  }
  REQUIRE(memory->open_files() == 0);
  REQUIRE(std::filesystem::is_empty(memory->temp_directory()));
  {
    fft::SpillFile file(memory);
    std::array<std::byte, 8> bytes{};
    REQUIRE_THROWS_WITH(file.read(std::numeric_limits<std::size_t>::max(), bytes),
                        ContainsSubstring("offset overflow"));
    REQUIRE_THROWS_WITH(file.write(std::numeric_limits<std::size_t>::max(), bytes),
                        ContainsSubstring("offset overflow"));
  }
  REQUIRE(std::filesystem::is_empty(memory->temp_directory()));
}

TEST_CASE("failed cached spill reads remain invalid on retries") {
  TempDir dir;
  auto memory = std::make_shared<fft::FitMemory>(fft::FitMemoryOptions{256, dir.path.string()});
  {
    fft::SpillVector<std::uint64_t> values(memory);
    for (std::uint64_t index = 0; index < 103; ++index) values.push_back(index);
    REQUIRE(values.spilled());
    REQUIRE(memory->bytes_in_memory() == 256);
    REQUIRE(memory->fallback_bytes_in_memory() == 0);
    values.rewrite([](std::size_t, std::uint64_t value) { return value; });
    REQUIRE(values.front() == 0);  // a successful cached page precedes the failing page
    const auto path = std::filesystem::directory_iterator(memory->temp_directory())->path();
    std::filesystem::resize_file(path, 40 * sizeof(std::uint64_t));
    // Page 32..63 is partly readable. Neither a first failure nor a later retry
    // may mark its incomplete range valid or return stale bytes from page zero.
    REQUIRE_THROWS_WITH(values[50], ContainsSubstring("unexpected end of file"));
    REQUIRE_THROWS_WITH(values[50], ContainsSubstring("unexpected end of file"));
    REQUIRE(values[0] == 0);
    REQUIRE_THROWS_WITH(values[50], ContainsSubstring("unexpected end of file"));
  }
  REQUIRE(memory->bytes_in_memory() == 0);
  REQUIRE(memory->fallback_bytes_in_memory() == 0);
  REQUIRE(memory->open_files() == 0);
}
