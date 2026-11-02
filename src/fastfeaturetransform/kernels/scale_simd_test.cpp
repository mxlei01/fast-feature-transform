// The dense floating-point scaler paths must preserve the scalar arithmetic,
// Arrow slice boundaries, missing values, and ownership of shared input buffers.

#include <arrow/array.h>
#include <arrow/buffer.h>
#include <arrow/util/bit_util.h>

#include <bit>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "fastfeaturetransform/kernel.hpp"
#include "fastfeaturetransform/test_support.hpp"

namespace fft = fastfeaturetransform;
using Catch::Matchers::ContainsSubstring;
using fft::testing::make_batch;

namespace {

struct ScalerCase {
  std::string id;
  nlohmann::json params;
  nlohmann::json state;
};

// Loading fixed statistics isolates transform from fit. The min/max multiplier
// makes contraction observable for a value one ulp below 1: multiply then add
// yields zero, while a fused multiply-add leaves a nonzero residual.
std::vector<ScalerCase> scaler_cases() {
  std::vector<ScalerCase> cases;
  for (bool mean : {false, true}) {
    for (bool deviation : {false, true}) {
      cases.push_back({"standard_scale",
                       {{"columns", {"x"}}, {"with_mean", mean}, {"with_std", deviation}},
                       {{"columns", {"x"}},
                        {"means", {{"x", 0.1}}},
                        {"variances", {{"x", 9.0}}},
                        {"scales", {{"x", 3.0}}},
                        {"counts", {{"x", 32}}}}});
    }
  }
  for (bool clip : {false, true}) {
    cases.push_back({"min_max_scale",
                     {{"columns", {"x"}}, {"feature_range", {-2.0, 3.0}}, {"clip", clip}},
                     {{"columns", {"x"}},
                      {"data_min", {{"x", 0.0}}},
                      {"data_max", {{"x", 5.0}}},
                      {"scales", {{"x", 0x1.0000000000001p0}}},
                      {"mins", {{"x", -1.0}}},
                      {"rows", 32}}});
    cases.push_back({"max_abs_scale",
                     {{"columns", {"x"}}, {"clip", clip}},
                     {{"columns", {"x"}}, {"max_abs", {{"x", 3.0}}}, {"scales", {{"x", 3.0}}}, {"rows", 32}}});
  }
  for (bool center : {false, true}) {
    for (bool scale : {false, true}) {
      for (bool unit : {false, true}) {
        cases.push_back(
            {"robust_scale",
             {{"columns", {"x"}}, {"with_centering", center}, {"with_scaling", scale}, {"unit_variance", unit}},
             {{"columns", {"x"}},
              {"center", {{"x", center ? nlohmann::json(0.1) : nlohmann::json()}}},
              {"scale", {{"x", scale ? nlohmann::json(unit ? 1.5 : 3.0) : nlohmann::json()}}}}});
      }
    }
  }
  return cases;
}

std::unique_ptr<fft::Kernel> load_scaler(const ScalerCase& spec) {
  auto kernel = fft::make_kernel(spec.id, spec.params);
  kernel->stage = "scale";
  kernel->load_state(spec.state);
  return kernel;
}

template <typename Scalar>
std::vector<Scalar> probe_values() {
  const Scalar zero = 0;
  const Scalar one = 1;
  const Scalar maximum = std::numeric_limits<Scalar>::max();
  const Scalar minimum = std::numeric_limits<Scalar>::min();
  const Scalar subnormal = std::numeric_limits<Scalar>::denorm_min();
  return {-zero,
          zero,
          subnormal,
          -subnormal,
          minimum,
          -minimum,
          maximum,
          -maximum,
          std::nextafter(maximum, zero),
          std::nextafter(-maximum, zero),
          std::nextafter(one, zero),
          std::nextafter(one, maximum),
          std::nextafter(-one, zero),
          std::nextafter(-one, -maximum),
          static_cast<Scalar>(0.1),
          static_cast<Scalar>(-0.1),
          static_cast<Scalar>(3),
          static_cast<Scalar>(-3),
          static_cast<Scalar>(8.5),
          static_cast<Scalar>(-8.5)};
}

// Prefix/suffix values are deliberately nonfinite. Only the logical slice is
// input; reading the whole backing buffer would incorrectly reject this array.
template <typename ArrowType>
std::shared_ptr<arrow::Array> dense_slice(int64_t offset, int64_t length) {
  using Scalar = typename ArrowType::c_type;
  const std::vector<Scalar> probes = probe_values<Scalar>();
  std::vector<Scalar> values(static_cast<std::size_t>(offset + length + 3), std::numeric_limits<Scalar>::quiet_NaN());
  for (int64_t row = 0; row < length; ++row) {
    values[static_cast<std::size_t>(offset + row)] = probes[static_cast<std::size_t>(row) % probes.size()];
  }
  const auto backing_length = static_cast<int64_t>(values.size());
  auto array = std::make_shared<arrow::NumericArray<ArrowType>>(
      backing_length, arrow::Buffer::FromVector(std::move(values)), nullptr, 0);
  return array->Slice(offset, length);
}

std::vector<uint8_t> buffer_bytes(const std::shared_ptr<arrow::Buffer>& buffer) {
  if (buffer == nullptr || buffer->size() == 0) return {};
  return {buffer->data(), buffer->data() + buffer->size()};
}

// Compare f64 representations rather than ==, which misses the sign of zero.
// The batch copy shares Arrow storage with the retained source and its alias.
template <typename ArrowType>
void require_record_bits_and_unchanged_source(const fft::Kernel& kernel, const std::shared_ptr<arrow::Array>& source) {
  const auto& typed = static_cast<const arrow::NumericArray<ArrowType>&>(*source);
  const auto values_before = buffer_bytes(source->data()->buffers[1]);
  const auto validity_before = buffer_bytes(source->data()->buffers[0]);
  const fft::Batch input = make_batch({"x"}, {source});
  const fft::Batch alias = input;
  const fft::Batch output = kernel.transform(input);
  REQUIRE(output.rows == input.rows);
  REQUIRE(output.names == input.names);
  REQUIRE(output.cols[0]->type_id() == arrow::Type::DOUBLE);
  REQUIRE(output.cols[0]->length() == source->length());
  REQUIRE(input.cols[0] == source);
  REQUIRE(alias.cols[0] == source);
  REQUIRE(buffer_bytes(source->data()->buffers[1]) == values_before);
  REQUIRE(buffer_bytes(source->data()->buffers[0]) == validity_before);
  if (source->length() != 0) {
    REQUIRE(output.cols[0]->data()->buffers[1]->data() != source->data()->buffers[1]->data());
  }
  const auto& scaled = static_cast<const arrow::DoubleArray&>(*output.cols[0]);
  for (int64_t row = 0; row < source->length(); ++row) {
    CAPTURE(row);
    const fft::RowValue input_value =
        typed.IsNull(row) ? fft::RowValue(std::monostate{}) : fft::RowValue(static_cast<double>(typed.Value(row)));
    const fft::Row record = kernel.transform_row({{"x", input_value}});
    REQUIRE(scaled.IsNull(row) == typed.IsNull(row));
    if (typed.IsNull(row)) {
      REQUIRE(std::holds_alternative<std::monostate>(record[0].second));
    } else {
      REQUIRE(std::bit_cast<uint64_t>(scaled.Value(row)) ==
              std::bit_cast<uint64_t>(std::get<double>(record[0].second)));
    }
  }
}

template <typename ArrowType>
void check_dense_slices() {
  for (const ScalerCase& spec : scaler_cases()) {
    CAPTURE(spec.id, spec.params);
    const auto kernel = load_scaler(spec);
    for (int64_t offset : {0, 3}) {
      for (int64_t length : {0, 1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 65}) {
        CAPTURE(offset, length);
        require_record_bits_and_unchanged_source<ArrowType>(*kernel, dense_slice<ArrowType>(offset, length));
      }
    }
  }
}

template <typename ArrowType>
void check_null_payloads() {
  using Scalar = typename ArrowType::c_type;
  constexpr int64_t kOffset = 3;
  constexpr int64_t kLength = 257;
  constexpr int64_t kBackingLength = kOffset + kLength + 3;
  for (const ScalerCase& spec : scaler_cases()) {
    CAPTURE(spec.id, spec.params);
    const auto kernel = load_scaler(spec);
    // A partially null slice, an entirely null slice, an all-valid slice of a
    // nullable parent, and a slice of alternating 64-row valid and null runs.
    // The all-valid slice may take the dense path.
    for (int mode : {0, 1, 2, 3}) {
      CAPTURE(mode);
      std::vector<Scalar> values(static_cast<std::size_t>(kBackingLength), std::numeric_limits<Scalar>::quiet_NaN());
      std::vector<uint8_t> bitmap(static_cast<std::size_t>((kBackingLength + 7) / 8), 0);
      for (int64_t row = 0; row < kLength; ++row) {
        const bool valid = mode == 2 || (mode == 0 && row % 8 != 0) || (mode == 3 && (row / 64) % 2 == 0);
        if (valid) {
          values[static_cast<std::size_t>(kOffset + row)] = static_cast<Scalar>(row) / static_cast<Scalar>(8);
          arrow::bit_util::SetBit(bitmap.data(), kOffset + row);
        } else if (row % 3 != 0) {
          values[static_cast<std::size_t>(kOffset + row)] =
              row % 3 == 1 ? std::numeric_limits<Scalar>::infinity() : -std::numeric_limits<Scalar>::infinity();
        }
      }
      const auto parent = std::make_shared<arrow::NumericArray<ArrowType>>(
          kBackingLength, arrow::Buffer::FromVector(std::move(values)), arrow::Buffer::FromVector(std::move(bitmap)));
      require_record_bits_and_unchanged_source<ArrowType>(*kernel, parent->Slice(kOffset, kLength));
    }
  }
}

template <typename ArrowType>
void check_visible_nonfinite() {
  using Scalar = typename ArrowType::c_type;
  for (const ScalerCase& spec : scaler_cases()) {
    CAPTURE(spec.id, spec.params);
    const auto kernel = load_scaler(spec);
    for (Scalar invalid : {std::numeric_limits<Scalar>::quiet_NaN(), std::numeric_limits<Scalar>::infinity(),
                           -std::numeric_limits<Scalar>::infinity()}) {
      for (int64_t position : {0, 7, 16, 32}) {
        CAPTURE(position, invalid);
        std::vector<Scalar> values(39, static_cast<Scalar>(1));
        values[static_cast<std::size_t>(3 + position)] = invalid;
        const auto parent = std::make_shared<arrow::NumericArray<ArrowType>>(
            39, arrow::Buffer::FromVector(std::move(values)), nullptr, 0);
        const auto source = parent->Slice(3, 33);
        const auto before = buffer_bytes(source->data()->buffers[1]);
        REQUIRE_THROWS_WITH(kernel->transform(make_batch({"x"}, {source})),
                            ContainsSubstring("scale: column 'x': value") && ContainsSubstring("is not finite"));
        REQUIRE(buffer_bytes(source->data()->buffers[1]) == before);
      }
    }
  }
}

}  // namespace

TEST_CASE("dense scaler f64 arithmetic agrees bitwise at slice and vector tail boundaries", "[scale_simd]") {
  check_dense_slices<arrow::DoubleType>();
}

TEST_CASE("dense scaler f32 input widens before arithmetic at slice and vector tail boundaries", "[scale_simd]") {
  check_dense_slices<arrow::FloatType>();
}

TEST_CASE("scalers ignore nonfinite payloads beneath nulls and outside valid slices", "[scale_simd]") {
  check_null_payloads<arrow::DoubleType>();
  check_null_payloads<arrow::FloatType>();
}

TEST_CASE("dense scaler finite checks include the first value and vector tails", "[scale_simd]") {
  check_visible_nonfinite<arrow::DoubleType>();
  check_visible_nonfinite<arrow::FloatType>();
}

TEST_CASE("adjacent scaler ownership reuses fresh values and refuses retained aliases", "[scale_simd]") {
  std::vector<std::unique_ptr<fft::Kernel>> chain;
  for (const std::string id : {"standard_scale", "min_max_scale", "robust_scale", "max_abs_scale"}) {
    for (const ScalerCase& spec : scaler_cases()) {
      if (spec.id == id) {
        chain.push_back(load_scaler(spec));
        break;
      }
    }
  }
  for (bool nullable : {false, true}) {
    std::vector<std::optional<double>> values;
    for (int row = 0; row < 257; ++row) {
      values.push_back(nullable && row % 7 == 0 ? std::optional<double>{}
                                                : std::optional<double>{row % 9 == 0 ? -0.0 : (row - 128) / 16.0});
    }
    const auto source = fft::testing::numeric_column<arrow::DoubleType>(values);
    const auto original = buffer_bytes(source->data()->buffers[1]);
    const fft::Batch input = make_batch({"x"}, {source});
    fft::Batch expected = input;
    fft::Batch reused = input;
    std::span<const std::string> owned;
    for (const auto& kernel : chain) {
      expected = kernel->transform(std::move(expected));
      const uint8_t* previous = reused.cols[0]->data()->buffers[1]->data();
      reused = kernel->transform_owned(std::move(reused), owned);
      if (!owned.empty()) REQUIRE(reused.cols[0]->data()->buffers[1]->data() == previous);
      owned = kernel->owned_output_columns();
    }
    const auto& actual = static_cast<const arrow::DoubleArray&>(*reused.cols[0]);
    const auto& reference = static_cast<const arrow::DoubleArray&>(*expected.cols[0]);
    for (int64_t row = 0; row < actual.length(); ++row) {
      REQUIRE(actual.IsNull(row) == reference.IsNull(row));
      if (actual.IsValid(row)) {
        REQUIRE(std::bit_cast<uint64_t>(actual.Value(row)) == std::bit_cast<uint64_t>(reference.Value(row)));
      }
    }
    REQUIRE(buffer_bytes(source->data()->buffers[1]) == original);

    fft::Batch intermediate = chain[0]->transform(input);
    const fft::Batch alias = intermediate;
    const auto alias_before = buffer_bytes(alias.cols[0]->data()->buffers[1]);
    const fft::Batch guarded = chain[1]->transform_owned(std::move(intermediate), chain[0]->owned_output_columns());
    REQUIRE(guarded.cols[0]->data()->buffers[1]->data() != alias.cols[0]->data()->buffers[1]->data());
    REQUIRE(buffer_bytes(alias.cols[0]->data()->buffers[1]) == alias_before);

    for (const bool retain_data : {false, true}) {
      fft::Batch owned_batch = chain[0]->transform(input);
      std::shared_ptr<arrow::ArrayData> held_data;
      std::shared_ptr<arrow::Buffer> held_buffer;
      if (retain_data)
        held_data = owned_batch.cols[0]->data();
      else
        held_buffer = owned_batch.cols[0]->data()->buffers[1];
      const auto& retained = retain_data ? held_data->buffers[1] : held_buffer;
      const auto before = buffer_bytes(retained);
      const fft::Batch result = chain[1]->transform_owned(std::move(owned_batch), chain[0]->owned_output_columns());
      REQUIRE(result.cols[0]->data()->buffers[1]->data() != retained->data());
      REQUIRE(buffer_bytes(retained) == before);
    }
  }
}
