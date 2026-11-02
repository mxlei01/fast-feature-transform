// Explicit C++ tests for FitMemory's cleanup, with and without an allocation
// failure, as a program of its own: it replaces the global allocation
// functions, so never link it into fft_tests or a sanitizer build, where the
// replacements would reach unrelated tests and the sanitizers' allocator hooks.
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <initializer_list>
#include <memory>
#include <new>
#include <stdexcept>

#include "fastfeaturetransform/spill.hpp"
#include "fastfeaturetransform/test_support.hpp"

namespace {

thread_local bool reject_allocations = false;
thread_local std::size_t rejected_allocations = 0;

// Declared before the context so rejection remains active through the context's
// destructor, including while the primary exception unwinds. Reporting and
// fixture cleanup run after this guard has restored ordinary allocation.
class AllocationFailure {
 public:
  AllocationFailure() = default;
  ~AllocationFailure() { reject_allocations = false; }
  AllocationFailure(const AllocationFailure&) = delete;
  AllocationFailure& operator=(const AllocationFailure&) = delete;
  void arm() { reject_allocations = true; }
};

struct PrimaryFailure {
  int value;
};

void check_cleanup(bool inject_failure, bool unwinding) {
  fastfeaturetransform::testing::TempDir temporary;
  std::filesystem::path spill_directory;
  std::weak_ptr<fastfeaturetransform::FitMemory> lifetime;
  rejected_allocations = 0;
  bool caught_primary = false;
  try {
    AllocationFailure failure;
    auto memory = std::make_shared<fastfeaturetransform::FitMemory>(
        fastfeaturetransform::FitMemoryOptions{0, temporary.path.string()});
    {
      fastfeaturetransform::SpillFile file(memory);
    }
    spill_directory = memory->temp_directory();
    lifetime = memory;
    if (spill_directory.empty() || !std::filesystem::is_directory(spill_directory))
      throw std::runtime_error("spill directory was not created");

    if (inject_failure) failure.arm();
    if (unwinding) throw PrimaryFailure{17};
    memory.reset();
  } catch (const PrimaryFailure& error) {
    caught_primary = error.value == 17;
  }

  if (caught_primary != unwinding) throw std::runtime_error("cleanup did not preserve the primary exception");
  if (!lifetime.expired()) throw std::runtime_error("cleanup retained the fit context");
  if (inject_failure != (rejected_allocations != 0))
    throw std::runtime_error("cleanup allocation fault was not exercised as requested");
  if (!inject_failure && std::filesystem::exists(spill_directory))
    throw std::runtime_error("ordinary cleanup left its private directory behind");
  // An allocation failure can leave the private directory behind. The outer
  // fixture removes it now that allocation is available again.
}

}  // namespace

void* operator new(std::size_t size) {
  if (reject_allocations) {
    ++rejected_allocations;
    throw std::bad_alloc();
  }
  if (void* result = std::malloc(size == 0 ? 1 : size)) return result;
  throw std::bad_alloc();
}

void* operator new[](std::size_t size) { return ::operator new(size); }
void operator delete(void* pointer) noexcept { std::free(pointer); }
void operator delete[](void* pointer) noexcept { std::free(pointer); }
void operator delete(void* pointer, std::size_t) noexcept { std::free(pointer); }
void operator delete[](void* pointer, std::size_t) noexcept { std::free(pointer); }

int main() {
  try {
    for (const bool inject_failure : {false, true}) {
      for (const bool unwinding : {false, true}) check_cleanup(inject_failure, unwinding);
    }
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "spill cleanup regression: %s\n", error.what());
    return 1;
  }
}
