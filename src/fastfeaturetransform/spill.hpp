// Buffers for exact fits: the values they keep share one automatic memory
// target and spill to private temporary files beyond it.
#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <new>
#include <span>
#include <string>
#include <type_traits>
#include <utility>

#include "fastfeaturetransform/core.hpp"

namespace fastfeaturetransform {

struct FitMemoryOptions {
  std::size_t memory_limit = std::size_t{64} << 20;
  std::string temp_dir;
};

class SpillFile;

class FitMemory {
 public:
  explicit FitMemory(FitMemoryOptions options = {});
  ~FitMemory() noexcept;
  FitMemory(const FitMemory&) = delete;
  FitMemory& operator=(const FitMemory&) = delete;
  const FitMemoryOptions& options() const;
  // In-memory values and ordinary caches stay inside the automatic target.
  // Once it is exhausted, each spilled vector still gets one small fixed IO
  // buffer, reported separately as fallback bytes.
  std::size_t bytes_in_memory() const;
  std::size_t peak_bytes_in_memory() const;
  std::size_t fallback_bytes_in_memory() const;
  std::size_t peak_fallback_bytes_in_memory() const;
  std::size_t spill_files_created() const;
  std::size_t open_files() const;
  std::string temp_directory() const;
  bool try_reserve(std::size_t bytes);
  void release(std::size_t bytes) noexcept;
  void reserve_fallback(std::size_t bytes);
  void release_fallback(std::size_t bytes) noexcept;

 private:
  friend class SpillFile;
  struct Impl;
  std::unique_ptr<Impl> impl_;
  std::string create_file();
  void read_file(const std::string& path, std::size_t offset, std::span<std::byte> bytes);
  void write_file(const std::string& path, std::size_t offset, std::span<const std::byte> bytes);
  void remove_file(const std::string& path) noexcept;
};

std::shared_ptr<FitMemory> current_fit_memory();

class FitMemoryScope {
 public:
  explicit FitMemoryScope(std::shared_ptr<FitMemory> memory);
  ~FitMemoryScope();
  FitMemoryScope(const FitMemoryScope&) = delete;
  FitMemoryScope& operator=(const FitMemoryScope&) = delete;

 private:
  std::shared_ptr<FitMemory> previous_;
};

// A file holds its context alive. Readers never use mmap; offsets and short IO
// are checked centrally, and file/context destruction removes temporary data.
class SpillFile {
 public:
  explicit SpillFile(std::shared_ptr<FitMemory> memory);
  ~SpillFile();
  SpillFile(const SpillFile&) = delete;
  SpillFile& operator=(const SpillFile&) = delete;
  void read(std::size_t offset, std::span<std::byte> bytes) const;
  void write(std::size_t offset, std::span<const std::byte> bytes);

 private:
  std::shared_ptr<FitMemory> memory_;
  std::string path_;
};

namespace spill_detail {

inline constexpr std::size_t chunk_bytes = std::size_t{64} << 10;
inline constexpr std::size_t cache_bytes = std::size_t{4} << 10;
inline constexpr std::size_t fallback_bytes = 256;
inline constexpr std::size_t sort_bytes = std::size_t{1} << 20;
inline constexpr std::size_t merge_fanin = 8;

template <typename T>
T* allocate(std::size_t count) {
  if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) throw Error("fit buffer: size overflow");
  constexpr auto alignment = std::align_val_t(std::max(alignof(T), alignof(std::max_align_t)));
  return static_cast<T*>(::operator new(count * sizeof(T), alignment));
}

template <typename T>
void deallocate(T* pointer) noexcept {
  constexpr auto alignment = std::align_val_t(std::max(alignof(T), alignof(std::max_align_t)));
  ::operator delete(pointer, alignment);
}

template <typename T>
struct WorkBuffer {
  explicit WorkBuffer(std::size_t count) : data(allocate<T>(count)) {}
  ~WorkBuffer() { deallocate(data); }
  WorkBuffer(const WorkBuffer&) = delete;
  WorkBuffer& operator=(const WorkBuffer&) = delete;
  T* data;
};

}  // namespace spill_detail

// RAM backing and ordinary IO caches share the FitMemory target. On exhaustion,
// each spilled vector keeps one separately accounted IO buffer of at most 256
// bytes (or one T for larger records). This overhead scales with columns/groups,
// not values per group. Each active chunk operation additionally uses <=64 KiB
// scratch (rewrite_from <=128 KiB), or one T if a record exceeds a chunk.
// External sort uses <=1 MiB run scratch, then <=128 KiB merge scratch, eight
// run cursors and two files. Scratch never grows with rows or the number of runs.
// One vector has one owner at a time; independent vectors can share a context.
template <typename T>
class SpillVector {
  static_assert(std::is_trivially_copyable_v<T>, "spill storage requires trivially copyable values");
  template <typename U>
  friend class SpillVector;

 public:
  SpillVector() : memory_(current_fit_memory()) {}
  explicit SpillVector(std::shared_ptr<FitMemory> memory) : memory_(std::move(memory)) {
    if (!memory_) memory_ = current_fit_memory();
  }
  ~SpillVector() { clear(); }
  SpillVector(const SpillVector&) = delete;
  SpillVector& operator=(const SpillVector&) = delete;
  SpillVector(SpillVector&& other) noexcept { swap(other); }
  SpillVector& operator=(SpillVector&& other) noexcept {
    if (this != &other) {
      clear();
      swap(other);
    }
    return *this;
  }

  std::size_t size() const noexcept { return size_; }
  bool empty() const noexcept { return size_ == 0; }
  bool spilled() const noexcept { return file_ != nullptr; }
  std::span<T> mutable_memory() noexcept { return file_ ? std::span<T>{} : std::span<T>{values_, size_}; }
  T front() const { return (*this)[0]; }
  T back() const {
    if (empty()) throw Error("fit buffer: back of empty buffer");
    return (*this)[size_ - 1];
  }

  void push_back(T value) {
    if (!file_ && size_ < capacity_) {
      std::memcpy(values_ + size_, &value, sizeof(T));
      ++size_;
      return;
    }
    append_span(std::span<const T>(&value, 1));
  }

  void append(SpillVector&& other) {
    if (this == &other) throw Error("fit buffer: cannot append itself");
    if (other.empty()) return;
    if (empty() && memory_ == other.memory_) {
      *this = std::move(other);
      return;
    }
    other.for_each_chunk([&](std::span<const T> values) { append_span(values); });
    other.clear();
  }

  void clear() noexcept {
    file_.reset();
    if (values_ != nullptr) {
      spill_detail::deallocate(values_);
      memory_->release(capacity_ * sizeof(T));
    }
    values_ = nullptr;
    capacity_ = 0;
    size_ = 0;
    free_cache();
  }
  void release() noexcept { clear(); }

  T operator[](std::size_t index) const {
    if (index >= size_) throw Error("fit buffer: index out of range");
    if (!file_) return values_[index];
    if (cache_ && index >= cache_start_ && index - cache_start_ < cache_count_) {
      return cache_[index - cache_start_];
    }
    flush();
    ensure_cache();
    if (cache_) {
      const auto start = index / cache_capacity_ * cache_capacity_;
      const auto count = std::min(cache_capacity_, size_ - start);
      invalidate_cache();
      file_->read(bytes(start), std::as_writable_bytes(std::span<T>(cache_, count)));
      cache_start_ = start;
      cache_count_ = count;
      return cache_[index - cache_start_];
    }
    std::array<std::byte, sizeof(T)> value;
    file_->read(bytes(index), value);
    return std::bit_cast<T>(value);
  }

  template <typename F>
  void for_each_chunk(F&& function) const {
    if (empty()) return;
    constexpr auto count = std::max(std::size_t{1}, spill_detail::chunk_bytes / sizeof(T));
    if (!file_) {
      for (std::size_t first = 0; first < size_;) {
        const auto length = std::min(count, size_ - first);
        std::invoke(function, std::span<const T>(values_ + first, length));
        first += length;
      }
      return;
    }
    flush();
    spill_detail::WorkBuffer<T> scratch(count);
    for (std::size_t first = 0; first < size_;) {
      const auto length = std::min(count, size_ - first);
      auto chunk = std::span<T>(scratch.data, length);
      file_->read(bytes(first), std::as_writable_bytes(chunk));
      std::invoke(function, std::span<const T>(chunk));
      first += length;
    }
  }

  template <typename F>
  void for_each(F&& function) const {
    std::size_t index = 0;
    for_each_chunk([&](std::span<const T> chunk) {
      for (const T& value : chunk) std::invoke(function, index++, value);
    });
  }

  template <typename F>
  void rewrite(F&& function) {
    if (!file_) {
      for (std::size_t index = 0; index < size_; ++index) values_[index] = std::invoke(function, index, values_[index]);
      return;
    }
    flush();
    invalidate_cache();
    constexpr auto count = std::max(std::size_t{1}, spill_detail::chunk_bytes / sizeof(T));
    spill_detail::WorkBuffer<T> scratch(count);
    for (std::size_t first = 0; first < size_;) {
      const auto length = std::min(count, size_ - first);
      auto chunk = std::span<T>(scratch.data, length);
      file_->read(bytes(first), std::as_writable_bytes(chunk));
      for (std::size_t index = 0; index < length; ++index)
        chunk[index] = std::invoke(function, first + index, chunk[index]);
      file_->write(bytes(first), std::as_bytes(chunk));
      invalidate_cache();
      first += length;
    }
  }

  template <typename U, typename F>
  void rewrite_from(const SpillVector<U>& source, F&& function) {
    if constexpr (std::is_same_v<T, U>) {
      if (this == &source) {
        rewrite(std::forward<F>(function));
        return;
      }
    }
    if (size_ != source.size()) {
      clear();
      constexpr auto count = std::max(std::size_t{1}, spill_detail::chunk_bytes / sizeof(T));
      spill_detail::WorkBuffer<T> scratch(count);
      std::size_t first = 0;
      source.for_each_chunk([&](std::span<const U> source_chunk) {
        for (std::size_t consumed = 0; consumed < source_chunk.size();) {
          const auto length = std::min(count, source_chunk.size() - consumed);
          for (std::size_t index = 0; index < length; ++index) {
            scratch.data[index] = std::invoke(function, first + index, source_chunk[consumed + index]);
          }
          append_span(std::span<const T>(scratch.data, length));
          first += length;
          consumed += length;
        }
      });
      return;
    }
    if (file_) flush();
    invalidate_cache();
    constexpr auto count = std::max(std::size_t{1}, spill_detail::chunk_bytes / sizeof(T));
    spill_detail::WorkBuffer<T> scratch(file_ ? count : 0);
    std::size_t first = 0;
    source.for_each_chunk([&](std::span<const U> source_chunk) {
      for (std::size_t consumed = 0; consumed < source_chunk.size();) {
        const auto length = std::min(count, source_chunk.size() - consumed);
        T* output = file_ ? scratch.data : values_ + first;
        for (std::size_t index = 0; index < length; ++index) {
          output[index] = std::invoke(function, first + index, source_chunk[consumed + index]);
        }
        if (file_) file_->write(bytes(first), std::as_bytes(std::span<T>(output, length)));
        first += length;
        consumed += length;
      }
    });
  }

  template <typename Compare = std::less<T>>
  void sort(Compare compare = {}) {
    if (size_ < 2) return;
    if (!file_) {
      std::sort(values_, values_ + size_, compare);
      return;
    }
    flush();
    invalidate_cache();
    auto alternate = std::make_unique<SpillFile>(memory_);
    const auto run_capacity = std::max(std::size_t{1}, spill_detail::sort_bytes / sizeof(T));
    {
      spill_detail::WorkBuffer<T> run(std::min(size_, run_capacity));
      for (std::size_t first = 0; first < size_;) {
        const auto length = std::min(run_capacity, size_ - first);
        auto chunk = std::span<T>(run.data, length);
        file_->read(bytes(first), std::as_writable_bytes(chunk));
        std::sort(chunk.begin(), chunk.end(), compare);
        alternate->write(bytes(first), std::as_bytes(chunk));
        first += length;
      }
    }
    std::swap(file_, alternate);
    for (std::size_t run_length = run_capacity; run_length < size_;) {
      merge_pass(*alternate, run_length, compare);
      std::swap(file_, alternate);
      run_length = run_length > size_ / spill_detail::merge_fanin ? size_ : run_length * spill_detail::merge_fanin;
    }
  }

 private:
  static std::size_t bytes(std::size_t count) {
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) throw Error("fit buffer: size overflow");
    return count * sizeof(T);
  }

  void swap(SpillVector& other) noexcept {
    using std::swap;
    swap(memory_, other.memory_);
    swap(values_, other.values_);
    swap(size_, other.size_);
    swap(capacity_, other.capacity_);
    swap(file_, other.file_);
    swap(cache_, other.cache_);
    swap(cache_capacity_, other.cache_capacity_);
    swap(cache_start_, other.cache_start_);
    swap(cache_count_, other.cache_count_);
    swap(cache_dirty_, other.cache_dirty_);
    swap(cache_fallback_, other.cache_fallback_);
  }

  void ensure_capacity(std::size_t required) {
    if (file_ || required <= capacity_) return;
    if (!memory_) memory_ = current_fit_memory();
    const auto doubled = capacity_ > std::numeric_limits<std::size_t>::max() / 2 ? required : capacity_ * 2;
    const auto capacity = std::max({std::size_t{16}, doubled, required});
    const auto allocation = bytes(capacity);
    // Charge the new allocation before releasing the old one so growth's
    // temporary double allocation also remains within the shared RAM budget.
    if (memory_->try_reserve(allocation)) {
      T* replacement;
      try {
        replacement = spill_detail::allocate<T>(capacity);
      } catch (...) {
        memory_->release(allocation);
        throw;
      }
      if (size_ != 0) std::memcpy(replacement, values_, bytes(size_));
      if (values_) {
        spill_detail::deallocate(values_);
        memory_->release(bytes(capacity_));
      }
      values_ = replacement;
      capacity_ = capacity;
      return;
    }
    auto file = std::make_unique<SpillFile>(memory_);
    if (size_ != 0) file->write(0, std::as_bytes(std::span<T>(values_, size_)));
    file_ = std::move(file);
    if (values_) {
      spill_detail::deallocate(values_);
      memory_->release(bytes(capacity_));
      values_ = nullptr;
      capacity_ = 0;
    }
    ensure_cache();
  }

  void append_span(std::span<const T> values) {
    if (values.size() > std::numeric_limits<std::size_t>::max() - size_) throw Error("fit buffer: size overflow");
    ensure_capacity(size_ + values.size());
    if (!file_) {
      if (!values.empty()) std::memcpy(values_ + size_, values.data(), values.size_bytes());
      size_ += values.size();
      return;
    }
    if (!cache_dirty_) invalidate_cache();
    ensure_cache();
    if (!cache_) {
      file_->write(bytes(size_), std::as_bytes(values));
      size_ += values.size();
      return;
    }
    while (!values.empty()) {
      if (!cache_dirty_ && values.size() >= cache_capacity_) {
        // Sequential chunk writers need not turn one large write into many
        // small writes just because this vector has a tiny append cache.
        const auto count = values.size() / cache_capacity_ * cache_capacity_;
        file_->write(bytes(size_), std::as_bytes(values.first(count)));
        size_ += count;
        values = values.subspan(count);
        continue;
      }
      if (!cache_dirty_) {
        cache_start_ = size_;
        cache_count_ = 0;
        cache_dirty_ = true;
      }
      const auto count = std::min(values.size(), cache_capacity_ - cache_count_);
      std::memcpy(cache_ + cache_count_, values.data(), bytes(count));
      cache_count_ += count;
      size_ += count;
      values = values.subspan(count);
      if (cache_count_ == cache_capacity_) flush();
    }
  }

  void ensure_cache() const {
    if (cache_ || !memory_) return;
    const auto desired = std::min(spill_detail::cache_bytes, std::max(spill_detail::fallback_bytes, bytes(size_)));
    const auto minimum = std::max(std::size_t{1}, spill_detail::fallback_bytes / sizeof(T));
    auto capacity = std::max(minimum, desired / sizeof(T));
    while (capacity >= minimum) {
      const auto allocation = bytes(capacity);
      if (memory_->try_reserve(allocation)) {
        try {
          cache_ = spill_detail::allocate<T>(capacity);
        } catch (...) {
          memory_->release(allocation);
          throw;
        }
        cache_capacity_ = capacity;
        return;
      }
      capacity /= 2;
    }
    // Keep a small append/read buffer for every spilled group even when
    // in-memory values fill the whole target. It is fixed metadata-like
    // overhead, separately visible in diagnostics, and never grows with rows.
    const auto allocation = bytes(minimum);
    memory_->reserve_fallback(allocation);
    try {
      cache_ = spill_detail::allocate<T>(minimum);
    } catch (...) {
      memory_->release_fallback(allocation);
      throw;
    }
    cache_capacity_ = minimum;
    cache_fallback_ = true;
  }

  void flush() const {
    if (!cache_dirty_) return;
    file_->write(bytes(cache_start_), std::as_bytes(std::span<T>(cache_, cache_count_)));
    cache_dirty_ = false;
    cache_count_ = 0;
  }
  void invalidate_cache() const noexcept {
    cache_start_ = 0;
    cache_count_ = 0;
    cache_dirty_ = false;
  }
  void free_cache() noexcept {
    if (cache_) {
      spill_detail::deallocate(cache_);
      if (cache_fallback_) {
        memory_->release_fallback(cache_capacity_ * sizeof(T));
      } else {
        memory_->release(cache_capacity_ * sizeof(T));
      }
    }
    cache_ = nullptr;
    cache_capacity_ = 0;
    cache_fallback_ = false;
    invalidate_cache();
  }

  template <typename Compare>
  void merge_pass(SpillFile& destination, std::size_t run_length, Compare& compare) {
    constexpr auto fanin = spill_detail::merge_fanin;
    constexpr auto page_count = std::max(std::size_t{1}, spill_detail::chunk_bytes / fanin / sizeof(T));
    constexpr auto output_count = std::max(std::size_t{1}, spill_detail::chunk_bytes / sizeof(T));
    spill_detail::WorkBuffer<T> pages(fanin * page_count);
    spill_detail::WorkBuffer<T> output(output_count);
    struct Cursor {
      std::size_t next = 0, end = 0, position = 0, count = 0;
    };
    const auto group_length = run_length > size_ / fanin ? size_ : run_length * fanin;
    for (std::size_t first = 0; first < size_;) {
      const auto end = first + std::min(group_length, size_ - first);
      std::array<Cursor, fanin> cursors{};
      std::array<std::size_t, fanin> heap{};
      std::size_t heap_size = 0;
      const auto refill = [&](std::size_t run) {
        auto& cursor = cursors[run];
        cursor.position = 0;
        cursor.count = std::min(page_count, cursor.end - cursor.next);
        if (cursor.count != 0) {
          file_->read(bytes(cursor.next),
                      std::as_writable_bytes(std::span<T>(pages.data + run * page_count, cursor.count)));
          cursor.next += cursor.count;
        }
      };
      const auto later = [&](std::size_t left, std::size_t right) {
        const auto& a = pages.data[left * page_count + cursors[left].position];
        const auto& b = pages.data[right * page_count + cursors[right].position];
        if (std::invoke(compare, b, a)) return true;
        if (std::invoke(compare, a, b)) return false;
        return left > right;
      };
      for (std::size_t run = 0, start = first; run < fanin && start < end; ++run) {
        const auto stop = start + std::min(run_length, end - start);
        cursors[run].next = start;
        cursors[run].end = stop;
        refill(run);
        heap[heap_size++] = run;
        std::push_heap(heap.begin(), heap.begin() + heap_size, later);
        start = stop;
      }
      std::size_t output_size = 0, written = first;
      while (heap_size != 0) {
        std::pop_heap(heap.begin(), heap.begin() + heap_size, later);
        const auto run = heap[--heap_size];
        auto& cursor = cursors[run];
        output.data[output_size++] = pages.data[run * page_count + cursor.position++];
        if (cursor.position == cursor.count) refill(run);
        if (cursor.count != 0) {
          heap[heap_size++] = run;
          std::push_heap(heap.begin(), heap.begin() + heap_size, later);
        }
        if (output_size == output_count) {
          destination.write(bytes(written), std::as_bytes(std::span<T>(output.data, output_size)));
          written += output_size;
          output_size = 0;
        }
      }
      if (output_size != 0) destination.write(bytes(written), std::as_bytes(std::span<T>(output.data, output_size)));
      first = end;
    }
  }

  std::shared_ptr<FitMemory> memory_;
  T* values_ = nullptr;
  std::size_t size_ = 0, capacity_ = 0;
  std::unique_ptr<SpillFile> file_;
  mutable T* cache_ = nullptr;
  mutable std::size_t cache_capacity_ = 0, cache_start_ = 0, cache_count_ = 0;
  mutable bool cache_dirty_ = false;
  mutable bool cache_fallback_ = false;
};

}  // namespace fastfeaturetransform
