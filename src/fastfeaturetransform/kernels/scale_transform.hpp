// Batch output for the elementwise scalers. Keep their arithmetic in the
// caller, shared with transform_row; this helper only handles Arrow storage
// and makes the dense floating-point loop available to the vectorizer.
#pragma once

#include <arrow/util/bit_block_counter.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <string>

#include "fastfeaturetransform/kernels/numeric.hpp"
#include "fastfeaturetransform/kernels/output.hpp"

namespace fastfeaturetransform {

template <typename Value, typename Output, typename F, typename Check>
void scale_dense_values(const Value* input, Output* output, int64_t length, F&& scale, Check&& check) {
  // A boolean reduction can vectorize without reassociating any floating-
  // point arithmetic. Keep throwing out of the hot loop; on an invalid
  // column, the caller's own validator then reports the first bad value.
  bool finite = true;
  for (int64_t row = 0; row < length; ++row) {
    finite &= std::isfinite(input[row]);
  }
  if (!finite) {
    for (int64_t row = 0; row < length; ++row) check(static_cast<double>(input[row]));
  }
  // Float32 widens before the arithmetic, as in for_each_numeric.
  // -ffp-contract=off still applies: in particular, multiply then add must
  // remain two roundings, and division must not become reciprocal multiply.
  for (int64_t row = 0; row < length; ++row) {
    output[row] = scale(static_cast<double>(input[row]));
  }
}

template <typename Value, typename F, typename Check, typename Null>
void scale_floating_values(const arrow::Array& array, const Value* input, double* values, F&& scale, Check&& check,
                           Null&& set_null) {
  if (array.null_count() == 0) {
    scale_dense_values(input, values, array.length(), scale, check);
    return;
  }
  arrow::internal::BitBlockCounter blocks(array.null_bitmap_data(), array.offset(), array.length());
  int64_t row = 0;
  while (row < array.length()) {
    const auto block = blocks.NextWord();
    const int64_t end = row + block.length;
    if (block.popcount == block.length) {
      scale_dense_values(input + row, values + row, block.length, scale, check);
      row = end;
      continue;
    }
    // A mixed block still exposes each valid run to the same arithmetic
    // loop; null payloads are never read, including NaNs hidden by validity.
    while (row < end) {
      if (array.IsNull(row)) {
        set_null(row++);
      } else {
        const int64_t begin = row++;
        while (row < end && array.IsValid(row)) ++row;
        scale_dense_values(input + begin, values + begin, row - begin, scale, check);
      }
    }
  }
}

template <typename F, typename Check>
std::shared_ptr<arrow::Array> scale_numeric_array(const arrow::Array& array, const std::string& stage,
                                                  const std::string& name, F&& scale, Check&& check) {
  const std::string where = stage + ": column '" + name + "'";
  const int64_t length = array.length();
  PrimitiveOutput<arrow::DoubleType> output(length, where);
  const auto set_null = [&](int64_t row) { output.set_null(row); };
  if (array.type_id() == arrow::Type::DOUBLE) {
    scale_floating_values(array, static_cast<const arrow::DoubleArray&>(array).raw_values(), output.mutable_values(),
                          scale, check, set_null);
  } else if (array.type_id() == arrow::Type::FLOAT) {
    scale_floating_values(array, static_cast<const arrow::FloatArray&>(array).raw_values(), output.mutable_values(),
                          scale, check, set_null);
  } else {
    // Integers widen in for_each_numeric, rounding a magnitude past 2^53 as
    // sklearn's float64 cast does; a uint64 past int64's range is refused.
    double* values = output.mutable_values();
    for_each_numeric(array, stage, name, [&](int64_t row, bool is_null, double value) {
      if (is_null) {
        output.set_null(row);
      } else {
        check(value);
        values[row] = scale(value);
      }
    });
  }
  return output.finish();
}

// Only the engine can grant reuse, after an immediately preceding scaler
// produced fresh output for this column. Refuse aliases as well: an owned
// array may still have another reference in the Batch or a retained view.
// This saves allocations, while retaining every stage's arithmetic and
// validation pass. It never composes affine coefficients.
template <typename F, typename Check>
std::shared_ptr<arrow::Array> scale_numeric_array(const std::shared_ptr<arrow::Array>& source, const std::string& stage,
                                                  const std::string& name, F&& scale, Check&& check, bool reusable) {
  const auto& data = source->data();
  if (reusable && source->type_id() == arrow::Type::DOUBLE && source->offset() == 0 && source.use_count() == 1 &&
      data.use_count() == 1 && data->buffers[1] && data->buffers[1].use_count() == 1 &&
      data->buffers[1]->is_mutable()) {
    double* values = reinterpret_cast<double*>(data->buffers[1]->mutable_data());
    scale_floating_values(*source, values, values, scale, check, [](int64_t) {});
    return source;
  }
  return scale_numeric_array(*source, stage, name, std::forward<F>(scale), std::forward<Check>(check));
}

}  // namespace fastfeaturetransform
