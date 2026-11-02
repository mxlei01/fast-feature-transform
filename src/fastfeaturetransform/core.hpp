// Core data shapes shared by the whole engine.
//
// A Batch is one in-flight slice of columnar data, typically a bounded piece
// of a Parquet row group. A Row is one record on the record path. A Schema
// tracks column name -> type tag while the planner resolves a plan against a
// source.
#pragma once

#include <arrow/array.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace fastfeaturetransform {

struct Batch {
  std::vector<std::string> names;
  std::vector<std::shared_ptr<arrow::Array>> cols;
  int64_t rows = 0;

  // Index of a column by name, or -1 when absent.
  int find(const std::string& name) const {
    for (std::size_t i = 0; i < names.size(); ++i) {
      if (names[i] == name) return static_cast<int>(i);
    }
    return -1;
  }
};

// Null is monostate. The set is closed: the binding converts datetimes and
// numpy scalars to these types and refuses any other value.
using RowValue = std::variant<std::monostate, int64_t, double, std::string, bool>;

// Ordered so the record path's output column order is deterministic.
using Row = std::vector<std::pair<std::string, RowValue>>;

inline RowValue* row_find(Row& row, const std::string& name) {
  for (auto& [key, value] : row) {
    if (key == name) return &value;
  }
  return nullptr;
}

// Column name -> type tag: "utf8", "bool", "i8" to "i64", "u8" to "u64",
// "f16", "f32", "f64", "f32[8]" for a fixed-size list of 8 floats, or Arrow's
// type string for anything else, such as a plain list or a date.
struct Schema {
  std::vector<std::pair<std::string, std::string>> cols;

  const std::string* find(const std::string& name) const {
    for (const auto& [col, type] : cols) {
      if (col == name) return &type;
    }
    return nullptr;
  }
};

// what() is a C string, so it ends at a message's first NUL byte; message
// keeps every byte, for the binding to raise the whole text.
struct Error : std::runtime_error {
  explicit Error(const std::string& text) : std::runtime_error(text), message(text) {}
  std::string message;
};

}  // namespace fastfeaturetransform
