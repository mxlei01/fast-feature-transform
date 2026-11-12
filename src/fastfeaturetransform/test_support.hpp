// Shared helpers for the C++ unit tests: a self-cleaning temporary directory,
// builders for hand-made Arrow columns/batches, and a reader of one task of a
// source.
#pragma once

#include <arrow/array/builder_binary.h>
#include <arrow/array/builder_primitive.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "fastfeaturetransform/core.hpp"
#include "fastfeaturetransform/io.hpp"

namespace fastfeaturetransform::testing {

// RAII temporary directory: created with a unique name under the system's
// temporary directory and removed on scope exit, even when a test fails.
struct TempDir {
  std::filesystem::path path;

  TempDir() {
    static std::atomic<int> counter{0};
    path = std::filesystem::temp_directory_path() /
           ("fft_test_" + std::to_string(::getpid()) + "_" + std::to_string(counter.fetch_add(1)));
    std::filesystem::create_directories(path);
  }
  ~TempDir() {
    std::error_code ec;
    std::filesystem::remove_all(path, ec);
  }
  TempDir(const TempDir&) = delete;
  TempDir& operator=(const TempDir&) = delete;
};

// nullopt entries become nulls.
inline std::shared_ptr<arrow::Array> utf8_column(const std::vector<std::optional<std::string>>& values) {
  arrow::StringBuilder builder;
  for (const auto& value : values) {
    if (value.has_value()) {
      (void)builder.Append(*value);
    } else {
      (void)builder.AppendNull();
    }
  }
  std::shared_ptr<arrow::Array> array;
  (void)builder.Finish(&array);
  return array;
}

// nullopt entries become nulls (64-bit offset layout, polars' default).
inline std::shared_ptr<arrow::Array> large_utf8_column(const std::vector<std::optional<std::string>>& values) {
  arrow::LargeStringBuilder builder;
  for (const auto& value : values) {
    if (value.has_value()) {
      (void)builder.Append(*value);
    } else {
      (void)builder.AppendNull();
    }
  }
  std::shared_ptr<arrow::Array> array;
  (void)builder.Finish(&array);
  return array;
}

inline std::shared_ptr<arrow::Array> f64_column(const std::vector<double>& values) {
  arrow::DoubleBuilder builder;
  for (double value : values) (void)builder.Append(value);
  std::shared_ptr<arrow::Array> array;
  (void)builder.Finish(&array);
  return array;
}

// Any Arrow numeric type (arrow::Int32Type, arrow::FloatType, ...); nullopt entries become nulls.
template <typename ArrowType>
std::shared_ptr<arrow::Array> numeric_column(const std::vector<std::optional<typename ArrowType::c_type>>& values) {
  arrow::NumericBuilder<ArrowType> builder;
  for (const auto& value : values) {
    if (value.has_value()) {
      (void)builder.Append(*value);
    } else {
      (void)builder.AppendNull();
    }
  }
  std::shared_ptr<arrow::Array> array;
  (void)builder.Finish(&array);
  return array;
}

// A timestamp column of the given unit and time zone ("" for none) from its
// stored counts; nullopt entries become nulls.
inline std::shared_ptr<arrow::Array> timestamp_column(arrow::TimeUnit::type unit,
                                                      const std::vector<std::optional<int64_t>>& values,
                                                      const std::string& zone = "") {
  arrow::TimestampBuilder builder(arrow::timestamp(unit, zone), arrow::default_memory_pool());
  for (const auto& value : values) {
    if (value.has_value()) {
      (void)builder.Append(*value);
    } else {
      (void)builder.AppendNull();
    }
  }
  std::shared_ptr<arrow::Array> array;
  (void)builder.Finish(&array);
  return array;
}

inline std::shared_ptr<arrow::Array> bool_column(const std::vector<std::optional<bool>>& values) {
  arrow::BooleanBuilder builder;
  for (const auto& value : values) {
    if (value.has_value()) {
      (void)builder.Append(*value);
    } else {
      (void)builder.AppendNull();
    }
  }
  std::shared_ptr<arrow::Array> array;
  (void)builder.Finish(&array);
  return array;
}

inline Batch make_batch(std::vector<std::string> names, std::vector<std::shared_ptr<arrow::Array>> cols) {
  Batch batch;
  batch.rows = cols.empty() ? 0 : cols.front()->length();
  batch.names = std::move(names);
  batch.cols = std::move(cols);
  return batch;
}

// Task number `task` of a source as one batch: read by the reader the engine
// uses, with the task's row count as the batch size. `columns` names the
// columns to decode (nullptr = every column). A task that arrives as more
// than one batch is an error, so a test reads the rows it means to.
inline Batch read_task(const RecordBatchSource& source, std::size_t task,
                       const std::vector<std::string>* columns = nullptr) {
  std::vector<Batch> batches;
  const int64_t rows = std::max<int64_t>(source.task_rows()[task], 1);
  const std::vector<std::unique_ptr<TaskReader>> readers = source.readers(1, columns);
  readers.front()->read_batches(source.plan_tasks()[task], rows,
                                [&](Batch batch) { batches.push_back(std::move(batch)); });
  if (batches.size() != 1) {
    throw Error("task " + std::to_string(task) + " was read as " + std::to_string(batches.size()) + " batches");
  }
  return std::move(batches.front());
}

}  // namespace fastfeaturetransform::testing
