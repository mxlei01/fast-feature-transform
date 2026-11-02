// In-memory scaler transform benchmark. Build this same source against each
// revision's Release fftcore; compare_scalers.py alternates the binaries and
// reports medians and IQRs. Parquet and fitting are outside this measurement.
#include <arrow/array/builder_primitive.h>
#include <arrow/buffer.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "fastfeaturetransform/kernel.hpp"

namespace fft = fastfeaturetransform;
using Clock = std::chrono::steady_clock;
using Json = nlohmann::json;

uint64_t hash_array(const arrow::Array& array) {
  uint64_t hash = 1469598103934665603ull;
  for (const auto& buffer : array.data()->buffers) {
    if (!buffer) continue;
    for (int64_t i = 0; i < buffer->size(); ++i) {
      hash ^= buffer->data()[i];
      hash *= 1099511628211ull;
    }
  }
  return hash;
}

template <typename T>
fft::Batch make_batch(int64_t rows, bool nulls) {
  arrow::NumericBuilder<T> builder;
  auto status = builder.Reserve(rows);
  if (!status.ok()) throw std::runtime_error(status.ToString());
  std::mt19937_64 rng(1729);
  std::uniform_real_distribution<double> distribution(-137.125, 991.75);
  for (int64_t i = 0; i < rows; ++i) {
    const auto value = static_cast<typename T::c_type>(distribution(rng));
    if (nulls && i % 7 == 0)
      builder.UnsafeAppendNull();
    else
      builder.UnsafeAppend(value);
  }
  std::shared_ptr<arrow::Array> array;
  status = builder.Finish(&array);
  if (!status.ok()) throw std::runtime_error(status.ToString());
  return {{"x"}, {std::move(array)}, rows};
}

struct Case {
  std::string name;
  fft::Batch batch;
  std::unique_ptr<fft::Kernel> kernel;
  uint64_t input_hash;
};

int main(int argc, char** argv) {
  const int64_t rows = argc > 1 ? std::stoll(argv[1]) : 500000;
  const int inner = argc > 2 ? std::stoi(argv[2]) : 100;
  const unsigned seed = argc > 3 ? static_cast<unsigned>(std::stoul(argv[3])) : 7u;
  if (rows < 2 || inner < 1) {
    std::cerr << "usage: fft_scaler_benchmark [rows >= 2] [iterations >= 1] [seed]\n";
    return 1;
  }
  auto training = make_batch<arrow::DoubleType>(257, false);
  std::vector<Case> cases;
  for (const std::string dtype : {"f64", "f32"}) {
    for (bool nulls : {false, true}) {
      auto batch =
          dtype == "f64" ? make_batch<arrow::DoubleType>(rows, nulls) : make_batch<arrow::FloatType>(rows, nulls);
      for (const std::string id : {"standard_scale", "min_max_scale", "max_abs_scale", "robust_scale",
                                   "min_max_scale_clip", "max_abs_scale_clip"}) {
        auto params = Json{{"columns", {"x"}}};
        const auto base = id.substr(0, id.find("_clip"));
        if (base != id) params["clip"] = true;
        if (base == "min_max_scale") params["feature_range"] = {-2.0, 3.0};
        auto kernel = fft::make_kernel(base, params);
        kernel->stage = "benchmark";
        fft::Schema schema{{{"x", "f64"}}};
        kernel->resolve(schema);
        std::vector<std::unique_ptr<fft::AccState>> states;
        states.push_back(kernel->make_state());
        kernel->consume(*states.back(), training);
        kernel->finalize(std::move(states));
        cases.push_back({id + "/" + dtype + "/" + (nulls ? "null_1_in_7" : "no_null"), batch, std::move(kernel),
                         hash_array(*batch.cols[0])});
      }
    }
  }
  std::mt19937 order(seed);
  std::shuffle(cases.begin(), cases.end(), order);
  Json results = Json::array();
  uint64_t checksum = 0;
  for (const auto& item : cases) {
    for (int i = 0; i < 5; ++i) {
      auto out = item.kernel->transform(item.batch);
      checksum ^= std::bit_cast<uint64_t>(static_cast<const arrow::DoubleArray&>(*out.cols[0]).Value(1));
    }
    const auto start = Clock::now();
    for (int i = 0; i < inner; ++i) {
      auto out = item.kernel->transform(item.batch);
      // No fitting, Python, file I/O or input creation inside this interval.
      // Returned arrays are freed each iteration, matching actual transform use.
      checksum ^= std::bit_cast<uint64_t>(static_cast<const arrow::DoubleArray&>(*out.cols[0]).Value(1));
    }
    const auto elapsed = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    if (hash_array(*item.batch.cols[0]) != item.input_hash) throw std::runtime_error("input array mutated");
    auto checked = item.kernel->transform(item.batch);
    const auto& output = static_cast<const arrow::DoubleArray&>(*checked.cols[0]);
    uint64_t output_hash = 1469598103934665603ull;
    for (int64_t i = 0; i < rows; ++i) {
      output_hash ^= output.IsNull(i) ? 0xfeedfaceull : std::bit_cast<uint64_t>(output.Value(i));
      output_hash *= 1099511628211ull;
    }
    results.push_back({{"case", item.name}, {"milliseconds", elapsed / inner}, {"output_hash", output_hash}});
  }
  std::cout
      << Json{{"rows", rows}, {"inner", inner}, {"seed", seed}, {"checksum", checksum}, {"results", results}}.dump()
      << '\n';
}
