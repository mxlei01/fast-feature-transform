// Dictionary fixtures shared by the categorical kernels' tests.
#pragma once

#include <arrow/array/array_dict.h>
#include <arrow/array/builder_primitive.h>
#include <arrow/buffer.h>
#include <arrow/util/bit_util.h>

#include <limits>

#include "fastfeaturetransform/test_support.hpp"

namespace fastfeaturetransform::testing {

struct DictionaryFixture {
  std::shared_ptr<arrow::Array> encoded;
  std::shared_ptr<arrow::Array> plain;
  std::vector<std::optional<std::string>> decoded;
};

template <typename IndexType = arrow::Int32Type>
DictionaryFixture dictionary_fixture(const std::vector<std::optional<std::string>>& words,
                                     const std::vector<std::optional<int64_t>>& rows, bool large = false,
                                     int64_t row_offset = 0, int64_t value_offset = 0) {
  std::vector<std::optional<std::string>> stored(static_cast<std::size_t>(value_offset), "prefix");
  stored.insert(stored.end(), words.begin(), words.end());
  auto values = (large ? large_utf8_column(stored) : utf8_column(stored))->Slice(value_offset);
  std::vector<std::optional<typename IndexType::c_type>> indices(static_cast<std::size_t>(row_offset), std::nullopt);
  std::vector<std::optional<std::string>> decoded;
  for (const auto& index : rows) {
    indices.push_back(index.has_value()
                          ? std::optional<typename IndexType::c_type>{static_cast<typename IndexType::c_type>(*index)}
                          : std::nullopt);
    decoded.push_back(index.has_value() ? words.at(static_cast<std::size_t>(*index)) : std::nullopt);
  }
  auto result = arrow::DictionaryArray::FromArrays(numeric_column<IndexType>(indices), values);
  if (!result.ok()) throw Error(result.status().ToString());
  auto encoded = (*result)->Slice(row_offset);
  if (auto status = encoded->ValidateFull(); !status.ok()) throw Error(status.ToString());
  auto plain = large ? large_utf8_column(decoded) : utf8_column(decoded);
  return {std::move(encoded), std::move(plain), std::move(decoded)};
}

// Indices of any width, each null slot holding the type's largest value: far
// outside the dictionary, as a producer may leave it, since Arrow leaves a
// null slot's value undefined.
template <typename IndexType>
std::shared_ptr<arrow::Array> indices_with_garbage(const std::vector<std::optional<int64_t>>& rows) {
  using Index = typename IndexType::c_type;
  arrow::NumericBuilder<IndexType> values;
  auto bitmap = arrow::AllocateEmptyBitmap(static_cast<int64_t>(rows.size()));
  if (!bitmap.ok()) throw Error(bitmap.status().ToString());
  int64_t nulls = 0;
  for (std::size_t i = 0; i < rows.size(); ++i) {
    const Index value = rows[i].has_value() ? static_cast<Index>(*rows[i]) : std::numeric_limits<Index>::max();
    if (auto status = values.Append(value); !status.ok()) throw Error(status.ToString());
    if (rows[i].has_value()) {
      arrow::bit_util::SetBit((*bitmap)->mutable_data(), static_cast<int64_t>(i));
    } else {
      ++nulls;
    }
  }
  std::shared_ptr<arrow::Array> dense;
  if (auto status = values.Finish(&dense); !status.ok()) throw Error(status.ToString());
  return arrow::MakeArray(
      arrow::ArrayData::Make(dense->type(), dense->length(),
                             {std::shared_ptr<arrow::Buffer>(std::move(*bitmap)), dense->data()->buffers[1]}, nulls));
}

}  // namespace fastfeaturetransform::testing
