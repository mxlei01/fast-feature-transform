#include "fastfeaturetransform/spill.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cstdlib>
#include <filesystem>
#include <list>
#include <mutex>
#include <system_error>

namespace fastfeaturetransform {
namespace {

thread_local std::shared_ptr<FitMemory> fit_memory;

[[noreturn]] void io_error(const std::string& operation, const std::string& path, int error) {
  throw Error("fit spill: " + operation + " '" + path +
              "': " + std::error_code(error, std::generic_category()).message());
}

void check_range(std::size_t offset, std::size_t length) {
  const auto maximum = static_cast<std::uintmax_t>(std::numeric_limits<off_t>::max());
  if (offset > maximum || length > maximum - offset) throw Error("fit spill: file offset overflow");
}

}  // namespace

struct FitMemory::Impl {
  explicit Impl(FitMemoryOptions value) : options(std::move(value)) {}
  struct Descriptor {
    std::string path;
    int fd;
  };
  static constexpr std::size_t descriptor_limit = 8;
  FitMemoryOptions options;
  std::atomic<std::size_t> bytes{0}, peak{0}, files_created{0};
  std::atomic<std::size_t> fallback_bytes{0}, fallback_peak{0};
  mutable std::mutex mutex;
  std::string directory;
  std::list<Descriptor> descriptors;

  // Called only under mutex. IO also holds that lock, so eviction cannot close
  // a descriptor in use and the context never opens more than eight at once.
  int descriptor(const std::string& path) {
    for (auto entry = descriptors.begin(); entry != descriptors.end(); ++entry) {
      if (entry->path == path) {
        const int fd = entry->fd;
        descriptors.splice(descriptors.begin(), descriptors, entry);
        return fd;
      }
    }
    evict_if_full();
    const int fd = ::open(path.c_str(), O_RDWR | O_CLOEXEC);
    if (fd < 0) io_error("open", path, errno);
    try {
      descriptors.push_front({path, fd});
    } catch (...) {
      ::close(fd);
      throw;
    }
    return fd;
  }

  void evict_if_full() {
    if (descriptors.size() == descriptor_limit) {
      ::close(descriptors.back().fd);
      descriptors.pop_back();
    }
  }
};

FitMemory::FitMemory(FitMemoryOptions options) : impl_(std::make_unique<Impl>(std::move(options))) {}

FitMemory::~FitMemory() noexcept {
  for (const auto& descriptor : impl_->descriptors) ::close(descriptor.fd);
  if (!impl_->directory.empty()) {
    try {
      std::error_code error;
      std::filesystem::remove_all(impl_->directory, error);
    } catch (...) {
      // Path conversion and recursive removal can allocate even with an
      // error_code. Best-effort scratch cleanup must preserve an active error.
    }
  }
}

const FitMemoryOptions& FitMemory::options() const { return impl_->options; }
std::size_t FitMemory::bytes_in_memory() const { return impl_->bytes.load(std::memory_order_relaxed); }
std::size_t FitMemory::peak_bytes_in_memory() const { return impl_->peak.load(std::memory_order_relaxed); }
std::size_t FitMemory::fallback_bytes_in_memory() const {
  return impl_->fallback_bytes.load(std::memory_order_relaxed);
}
std::size_t FitMemory::peak_fallback_bytes_in_memory() const {
  return impl_->fallback_peak.load(std::memory_order_relaxed);
}
std::size_t FitMemory::spill_files_created() const { return impl_->files_created.load(std::memory_order_relaxed); }

std::size_t FitMemory::open_files() const {
  std::lock_guard lock(impl_->mutex);
  return impl_->descriptors.size();
}

std::string FitMemory::temp_directory() const {
  std::lock_guard lock(impl_->mutex);
  return impl_->directory;
}

bool FitMemory::try_reserve(std::size_t bytes) {
  auto current = impl_->bytes.load(std::memory_order_relaxed);
  for (;;) {
    if (bytes > impl_->options.memory_limit - current) return false;
    if (impl_->bytes.compare_exchange_weak(current, current + bytes, std::memory_order_relaxed)) break;
  }
  const auto reserved = current + bytes;
  auto peak = impl_->peak.load(std::memory_order_relaxed);
  while (peak < reserved && !impl_->peak.compare_exchange_weak(peak, reserved, std::memory_order_relaxed)) {
  }
  return true;
}

void FitMemory::release(std::size_t bytes) noexcept { impl_->bytes.fetch_sub(bytes, std::memory_order_relaxed); }

void FitMemory::reserve_fallback(std::size_t bytes) {
  auto current = impl_->fallback_bytes.load(std::memory_order_relaxed);
  for (;;) {
    if (bytes > std::numeric_limits<std::size_t>::max() - current) throw Error("fit spill: IO buffer size overflow");
    if (impl_->fallback_bytes.compare_exchange_weak(current, current + bytes, std::memory_order_relaxed)) break;
  }
  const auto reserved = current + bytes;
  auto peak = impl_->fallback_peak.load(std::memory_order_relaxed);
  while (peak < reserved && !impl_->fallback_peak.compare_exchange_weak(peak, reserved, std::memory_order_relaxed)) {
  }
}

void FitMemory::release_fallback(std::size_t bytes) noexcept {
  impl_->fallback_bytes.fetch_sub(bytes, std::memory_order_relaxed);
}

std::string FitMemory::create_file() {
  std::lock_guard lock(impl_->mutex);
  if (impl_->directory.empty()) {
    std::error_code error;
    const auto parent = impl_->options.temp_dir.empty() ? std::filesystem::temp_directory_path(error)
                                                        : std::filesystem::path(impl_->options.temp_dir);
    if (error) throw Error("fit spill: temporary directory: " + error.message());
    auto pattern = (parent / "fast-feature-transform-fit-XXXXXX").string();
    if (::mkdtemp(pattern.data()) == nullptr) io_error("create temporary directory", parent.string(), errno);
    impl_->directory = std::move(pattern);
  }
  auto path = (std::filesystem::path(impl_->directory) / "buffer-XXXXXX").string();
  // Make room before mkstemp as well as before ordinary reopen operations.
  impl_->evict_if_full();
  const int fd = ::mkstemp(path.data());
  if (fd < 0) io_error("create temporary file", impl_->directory, errno);
  if (::fcntl(fd, F_SETFD, FD_CLOEXEC) < 0) {
    const int error = errno;
    ::close(fd);
    ::unlink(path.c_str());
    io_error("set close-on-exec", path, error);
  }
  try {
    impl_->descriptors.push_front({path, fd});
  } catch (...) {
    ::close(fd);
    ::unlink(path.c_str());
    throw;
  }
  impl_->files_created.fetch_add(1, std::memory_order_relaxed);
  return path;
}

void FitMemory::read_file(const std::string& path, std::size_t offset, std::span<std::byte> bytes) {
  check_range(offset, bytes.size());
  if (bytes.empty()) return;
  std::lock_guard lock(impl_->mutex);
  const int fd = impl_->descriptor(path);
  while (!bytes.empty()) {
    const auto length = std::min(bytes.size(), std::size_t{1} << 30);
    const auto result = ::pread(fd, bytes.data(), length, static_cast<off_t>(offset));
    if (result < 0) {
      if (errno == EINTR) continue;
      io_error("read", path, errno);
    }
    if (result == 0) throw Error("fit spill: unexpected end of file reading '" + path + "'");
    bytes = bytes.subspan(static_cast<std::size_t>(result));
    offset += static_cast<std::size_t>(result);
  }
}

void FitMemory::write_file(const std::string& path, std::size_t offset, std::span<const std::byte> bytes) {
  check_range(offset, bytes.size());
  if (bytes.empty()) return;
  std::lock_guard lock(impl_->mutex);
  const int fd = impl_->descriptor(path);
  while (!bytes.empty()) {
    const auto length = std::min(bytes.size(), std::size_t{1} << 30);
    const auto result = ::pwrite(fd, bytes.data(), length, static_cast<off_t>(offset));
    if (result < 0) {
      if (errno == EINTR) continue;
      io_error("write", path, errno);
    }
    if (result == 0) throw Error("fit spill: zero-byte write to '" + path + "'");
    bytes = bytes.subspan(static_cast<std::size_t>(result));
    offset += static_cast<std::size_t>(result);
  }
}

void FitMemory::remove_file(const std::string& path) noexcept {
  std::lock_guard lock(impl_->mutex);
  for (auto entry = impl_->descriptors.begin(); entry != impl_->descriptors.end(); ++entry) {
    if (entry->path == path) {
      ::close(entry->fd);
      impl_->descriptors.erase(entry);
      break;
    }
  }
  ::unlink(path.c_str());
}

std::shared_ptr<FitMemory> current_fit_memory() {
  if (fit_memory) return fit_memory;
  static const auto fallback = std::make_shared<FitMemory>();
  return fallback;
}

FitMemoryScope::FitMemoryScope(std::shared_ptr<FitMemory> memory) : previous_(std::move(fit_memory)) {
  fit_memory = std::move(memory);
}
FitMemoryScope::~FitMemoryScope() { fit_memory = std::move(previous_); }

SpillFile::SpillFile(std::shared_ptr<FitMemory> memory) : memory_(std::move(memory)), path_(memory_->create_file()) {}
SpillFile::~SpillFile() { memory_->remove_file(path_); }
void SpillFile::read(std::size_t offset, std::span<std::byte> bytes) const { memory_->read_file(path_, offset, bytes); }
void SpillFile::write(std::size_t offset, std::span<const std::byte> bytes) {
  memory_->write_file(path_, offset, bytes);
}

}  // namespace fastfeaturetransform
