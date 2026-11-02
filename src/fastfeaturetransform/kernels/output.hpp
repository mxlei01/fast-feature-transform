// Fresh Arrow-owned output storage. Validity is derived from the result,
// rather than borrowed from inputs: an encoder can null an unknown value or
// encode an input null as a valid fitted category.
#pragma once

#include <arrow/array.h>
#include <arrow/buffer.h>
#include <arrow/type_traits.h>
#include <arrow/util/bit_util.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>

#include "fastfeaturetransform/core.hpp"

namespace fastfeaturetransform {

class OutputValidity {
 public:
  OutputValidity(int64_t length, std::string where) : length_(length), where_(std::move(where)) {}

  void set_null(int64_t row) {
    if (!bitmap_) {
      auto allocation = arrow::AllocateBitmap(length_);
      if (!allocation.ok()) throw Error(where_ + ": allocation failed");
      bitmap_ = *std::move(allocation);
      std::memset(bitmap_->mutable_data(), 0xff, static_cast<std::size_t>(bitmap_->size()));
    }
    if (arrow::bit_util::GetBit(bitmap_->data(), static_cast<uint64_t>(row))) {
      arrow::bit_util::ClearBit(bitmap_->mutable_data(), row);
      ++null_count_;
    }
  }

  const std::shared_ptr<arrow::Buffer>& bitmap() const { return bitmap_; }
  int64_t null_count() const { return null_count_; }

 private:
  int64_t length_;
  std::string where_;
  std::shared_ptr<arrow::Buffer> bitmap_;
  int64_t null_count_ = 0;
};

template <typename ArrowType>
class PrimitiveOutput {
 public:
  using Value = typename ArrowType::c_type;

  PrimitiveOutput(int64_t length, const std::string& where, bool zero_initialize = false,
                  std::shared_ptr<OutputValidity> validity = {})
      : length_(length), where_(where), validity_(std::move(validity)) {
    static_assert(!std::is_same_v<ArrowType, arrow::BooleanType>, "Boolean Arrow values are bit packed");
    if (length < 0 || length > std::numeric_limits<int64_t>::max() / static_cast<int64_t>(sizeof(Value))) {
      throw Error(where + ": allocation failed");
    }
    auto allocation = arrow::AllocateBuffer(length * static_cast<int64_t>(sizeof(Value)));
    if (!allocation.ok()) throw Error(where + ": allocation failed");
    values_ = *std::move(allocation);
    if (zero_initialize && length != 0) std::fill_n(mutable_values(), length, Value{});
  }

  Value* mutable_values() { return reinterpret_cast<Value*>(values_->mutable_data()); }

  void set_null(int64_t row) {
    if (!validity_) validity_ = std::make_shared<OutputValidity>(length_, where_);
    validity_->set_null(row);
    mutable_values()[row] = Value{};
  }

  // Finish only after every value and all shared validity bits are written.
  // No input buffers are modified, and the returned array owns its storage.
  std::shared_ptr<arrow::Array> finish() const {
    return arrow::MakeArray(arrow::ArrayData::Make(arrow::TypeTraits<ArrowType>::type_singleton(), length_,
                                                   {validity_ ? validity_->bitmap() : nullptr, values_},
                                                   validity_ ? validity_->null_count() : 0));
  }

 private:
  int64_t length_;
  std::string where_;
  std::shared_ptr<arrow::Buffer> values_;
  std::shared_ptr<OutputValidity> validity_;
};

}  // namespace fastfeaturetransform
