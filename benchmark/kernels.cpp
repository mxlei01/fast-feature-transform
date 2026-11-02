// In-memory benchmark of every kernel, with rank-hot stage probes and an
// optional line-level sampler. Build it from a Release (-g) configuration; it
// measures the production kernels without Python or Parquet I/O.
//
//   fft_kernel_benchmark [--rows N] [--iterations N] [--min-seconds S] [--list]
//                       [--case SUBSTRING]... [--profile FILE] [--interval-us N]
//
// Each case runs one kernel with the parameters it has in the README's latency
// charts; label_encode, select, cast and group_agg, which the charts leave
// out, have cases of their own. A kernel is fitted on 100,000 rows and then
// transforms --rows rows of fresh data. Categorical kernels take one string
// column "c" drawn from the published 32 categories ("c000".."c031", weights
// falling as rank^-0.8), plain or dictionary-encoded, and the supervised ones
// fit with a binary target whose rate rises with the category. Numeric kernels
// take two correlated normal columns x and z (positive ones for Box-Cox), and
// group_agg takes c and x. Every case times the fit and the transform, prints
// the median and quartiles per call, and hashes the transform's output so that
// two builds can be checked for identical results. One-hot and rank-hot also
// transform at 50, 90 and 99% nulls, and rank-hot's stages run apart:
// resolving every row, recording each row's rank, filling the columns from the
// ranks, and allocating them zero-filled.
//
// With --profile FILE, a sampler thread suspends the measuring thread,
// records its program counter and sleeps --interval-us microseconds (50 by
// default) before the next sample. At exit FILE gets a "base <load address>"
// line followed by "<pc> <count>" lines; benchmark/line_profile.py turns that
// into per-line and per-instruction counts with atos and llvm-objdump, and
// benchmark/profile_kernels.py profiles every case that way. Sampling uses
// Mach thread calls and reads arm64 registers, so the benchmark builds on
// arm64 macOS only.
#include <arrow/array.h>
#include <arrow/array/builder_binary.h>
#include <arrow/array/builder_primitive.h>
#include <mach-o/dyld.h>
#include <mach/mach.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "fastfeaturetransform/kernel.hpp"
#include "fastfeaturetransform/kernels/categories.hpp"
#include "fastfeaturetransform/kernels/output.hpp"

namespace fft = fastfeaturetransform;
using Clock = std::chrono::steady_clock;
using Json = nlohmann::json;

namespace {

constexpr int kCategories = 32;
constexpr int64_t kFitRows = 100000;
constexpr uint64_t kFitSeed = 1729;
constexpr uint64_t kTransformSeed = 2718;
// Every stage a case can run; all but fit and transform are rank-hot's alone.
constexpr std::array<const char*, 6> kStages{"fit", "transform", "resolve", "resolve_rank", "fill", "alloc_zero"};

// One kernel and its parameters; `name` adds a variant where one kernel has
// several published configurations. `dataset` is "categorical", "numeric",
// "positive" or "mixed".
struct Case {
  const char* name;
  const char* kernel;
  const char* params;
  const char* dataset;
};

const Case kCases[] = {
    {"count_encode", "count_encode",
     R"({"columns":["c"],"normalize":false,"min_group_size":null,"min_group_name":null,"combine_min_nan_groups":true,"handle_unknown":"value","handle_missing":"value"})",
     "categorical"},
    {"ordinal_encode", "ordinal_encode",
     R"({"columns":["c"],"index_start":1,"handle_unknown":"value","handle_missing":"value"})", "categorical"},
    {"one_hot_encode", "one_hot_encode",
     R"({"columns":["c"],"use_cat_names":true,"handle_unknown":"value","handle_missing":"value"})", "categorical"},
    {"rank_hot_encode", "rank_hot_encode",
     R"({"columns":["c"],"use_cat_names":true,"handle_unknown":"value","handle_missing":"value"})", "categorical"},
    {"basen_encode", "basen_encode", R"({"columns":["c"],"handle_unknown":"value","handle_missing":"value","base":3})",
     "categorical"},
    {"binary_encode", "binary_encode", R"({"columns":["c"],"handle_unknown":"value","handle_missing":"value"})",
     "categorical"},
    {"gray_encode", "gray_encode", R"({"columns":["c"],"handle_unknown":"value","handle_missing":"value"})",
     "categorical"},
    {"backward_difference_encode", "backward_difference_encode",
     R"({"columns":["c"],"handle_unknown":"value","handle_missing":"value"})", "categorical"},
    {"helmert_encode", "helmert_encode", R"({"columns":["c"],"handle_unknown":"value","handle_missing":"value"})",
     "categorical"},
    {"sum_encode", "sum_encode", R"({"columns":["c"],"handle_unknown":"value","handle_missing":"value"})",
     "categorical"},
    {"polynomial_encode", "polynomial_encode", R"({"columns":["c"],"handle_unknown":"value","handle_missing":"value"})",
     "categorical"},
    {"hash_encode", "hash_encode", R"({"columns":["c"],"n_components":8,"hash_method":"md5"})", "categorical"},
    {"target_encode", "target_encode",
     R"({"columns":["c"],"target":"target","handle_unknown":"value","handle_missing":"value","min_samples_leaf":20,"smoothing":10.0})",
     "categorical"},
    {"m_estimate_encode", "m_estimate_encode",
     R"({"columns":["c"],"target":"target","handle_unknown":"value","handle_missing":"value","m":1.0})", "categorical"},
    {"woe_encode", "woe_encode",
     R"({"columns":["c"],"target":"target","handle_unknown":"value","handle_missing":"value","regularization":1.0})",
     "categorical"},
    {"james_stein_encode", "james_stein_encode",
     R"({"columns":["c"],"target":"target","handle_unknown":"value","handle_missing":"value","model":"independent"})",
     "categorical"},
    {"leave_one_out_encode", "leave_one_out_encode",
     R"({"columns":["c"],"target":"target","handle_unknown":"value","handle_missing":"value"})", "categorical"},
    {"cat_boost_encode", "cat_boost_encode",
     R"({"columns":["c"],"target":"target","handle_unknown":"value","handle_missing":"value","a":1.0})", "categorical"},
    {"quantile_encode", "quantile_encode",
     R"({"columns":["c"],"target":"target","quantile":0.5,"m":1.0,"handle_unknown":"value","handle_missing":"value"})",
     "categorical"},
    {"summary_encode", "summary_encode",
     R"({"columns":["c"],"target":"target","quantiles":[0.25,0.75],"m":1.0,"handle_unknown":"value","handle_missing":"value"})",
     "categorical"},
    {"label_encode", "label_encode", R"({"columns":["c"],"min_count":1})", "categorical"},
    {"standard_scale", "standard_scale", R"({"columns":["x","z"],"with_mean":true,"with_std":true})", "numeric"},
    {"min_max_scale", "min_max_scale", R"({"columns":["x","z"],"feature_range":[0.0,1.0],"clip":false})", "numeric"},
    {"max_abs_scale", "max_abs_scale", R"({"columns":["x","z"],"clip":false})", "numeric"},
    {"binarize", "binarize", R"({"columns":["x","z"],"threshold":0.0})", "numeric"},
    {"normalize.l2", "normalize", R"({"columns":["x","z"],"norm":"l2"})", "numeric"},
    {"normalize.l1", "normalize", R"({"columns":["x","z"],"norm":"l1"})", "numeric"},
    {"normalize.max", "normalize", R"({"columns":["x","z"],"norm":"max"})", "numeric"},
    {"robust_scale", "robust_scale",
     R"({"columns":["x","z"],"with_centering":true,"with_scaling":true,"quantile_range":[25.0,75.0],"unit_variance":false})",
     "numeric"},
    {"robust_scale.unit_variance", "robust_scale",
     R"({"columns":["x","z"],"with_centering":true,"with_scaling":true,"quantile_range":[25.0,75.0],"unit_variance":true})",
     "numeric"},
    {"quantile_transform.uniform", "quantile_transform",
     R"({"columns":["x","z"],"n_quantiles":1000,"output_distribution":"uniform"})", "numeric"},
    {"quantile_transform.normal", "quantile_transform",
     R"({"columns":["x","z"],"n_quantiles":1000,"output_distribution":"normal"})", "numeric"},
    {"k_bins_discretize.uniform", "k_bins_discretize",
     R"({"columns":["x","z"],"n_bins":5,"encode":"onehot-dense","strategy":"uniform","quantile_method":"averaged_inverted_cdf"})",
     "numeric"},
    {"k_bins_discretize.quantile", "k_bins_discretize",
     R"({"columns":["x","z"],"n_bins":5,"encode":"onehot-dense","strategy":"quantile","quantile_method":"averaged_inverted_cdf"})",
     "numeric"},
    {"k_bins_discretize.kmeans", "k_bins_discretize",
     R"({"columns":["x","z"],"n_bins":5,"encode":"onehot-dense","strategy":"kmeans","quantile_method":"averaged_inverted_cdf"})",
     "numeric"},
    {"polynomial_features", "polynomial_features",
     R"({"columns":["x","z"],"degree":2,"interaction_only":false,"include_bias":true})", "numeric"},
    {"spline_transform", "spline_transform",
     R"({"columns":["x","z"],"n_knots":5,"degree":3,"knots":"uniform","extrapolation":"constant","include_bias":true,"handle_missing":"error"})",
     "numeric"},
    {"power_transform.yeo-johnson", "power_transform",
     R"({"columns":["x","z"],"method":"yeo-johnson","standardize":true})", "numeric"},
    {"power_transform.box-cox", "power_transform", R"({"columns":["x","z"],"method":"box-cox","standardize":true})",
     "positive"},
    {"select", "select", R"({"columns":["x"]})", "numeric"},
    {"cast", "cast", R"({"types":{"x":"f32","z":"f32"}})", "numeric"},
    {"group_agg", "group_agg", R"({"key":"c","aggs":{"x":["count","sum","mean","min","max","median"]}})", "mixed"},
};

std::string label(int i) {
  std::string out = "c000";
  out[1] = static_cast<char>('0' + i / 100);
  out[2] = static_cast<char>('0' + i / 10 % 10);
  out[3] = static_cast<char>('0' + i % 10);
  return out;
}

std::vector<int> draw(int64_t rows, uint64_t seed) {
  std::vector<double> weights(kCategories);
  for (int i = 0; i < kCategories; ++i) weights[static_cast<std::size_t>(i)] = std::pow(i + 1.0, -0.8);
  std::discrete_distribution<int> pick(weights.begin(), weights.end());
  std::mt19937_64 rng(seed);
  std::vector<int> codes(static_cast<std::size_t>(rows));
  for (auto& code : codes) code = pick(rng);
  return codes;
}

std::shared_ptr<arrow::Array> strings(const std::vector<int>& codes, double null_fraction, bool dictionary,
                                      uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  std::shared_ptr<arrow::Array> out;
  if (!dictionary) {
    arrow::StringBuilder builder;
    for (int code : codes) {
      const auto status = unit(rng) < null_fraction ? builder.AppendNull() : builder.Append(label(code));
      if (!status.ok()) throw std::runtime_error(status.ToString());
    }
    if (!builder.Finish(&out).ok()) throw std::runtime_error("string builder");
    return out;
  }
  arrow::Int32Builder indices;
  for (int code : codes) {
    const auto status = unit(rng) < null_fraction ? indices.AppendNull() : indices.Append(code);
    if (!status.ok()) throw std::runtime_error(status.ToString());
  }
  std::shared_ptr<arrow::Array> index_array;
  if (!indices.Finish(&index_array).ok()) throw std::runtime_error("index builder");
  arrow::StringBuilder values;
  for (int i = 0; i < kCategories; ++i) {
    if (!values.Append(label(i)).ok()) throw std::runtime_error("dictionary builder");
  }
  std::shared_ptr<arrow::Array> dictionary_values;
  if (!values.Finish(&dictionary_values).ok()) throw std::runtime_error("dictionary builder");
  auto made = arrow::DictionaryArray::FromArrays(arrow::dictionary(arrow::int32(), arrow::utf8()), index_array,
                                                 dictionary_values);
  if (!made.ok()) throw std::runtime_error(made.status().ToString());
  return *made;
}

template <typename Builder, typename Value>
std::shared_ptr<arrow::Array> column(const std::vector<Value>& values) {
  Builder builder;
  if (!builder.AppendValues(values).ok()) throw std::runtime_error("numeric builder");
  std::shared_ptr<arrow::Array> out;
  if (!builder.Finish(&out).ok()) throw std::runtime_error("numeric builder");
  return out;
}

// The comparison's numeric data: x and z are correlated normals, or both
// 0.25 + exp(N(0, 1)) for "positive".
std::pair<std::vector<double>, std::vector<double>> numbers(int64_t rows, uint64_t seed, bool positive) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<double> normal;
  std::vector<double> x(static_cast<std::size_t>(rows)), z(static_cast<std::size_t>(rows));
  for (std::size_t i = 0; i < x.size(); ++i) {
    const double a = normal(rng), b = normal(rng);
    x[i] = positive ? 0.25 + std::exp(a) : 0.7 * a + 0.25;
    z[i] = positive ? 0.25 + std::exp(b) : 0.35 * a + 1.3 * b + 0.5;
  }
  return {std::move(x), std::move(z)};
}

// A batch for one dataset. A training batch with a "c" column starts with
// every category in sorted order and carries the target.
fft::Batch batch_for(const std::string& dataset, int64_t rows, bool training, bool dictionary, double nulls) {
  const uint64_t seed = training ? kFitSeed : kTransformSeed;
  fft::Batch batch;
  batch.rows = rows;
  if (dataset == "categorical" || dataset == "mixed") {
    std::vector<int> codes = draw(rows, seed);
    if (training) {
      for (int i = 0; i < kCategories && i < rows; ++i) codes[static_cast<std::size_t>(i)] = i;
    }
    batch.names.push_back("c");
    batch.cols.push_back(strings(codes, nulls, dictionary, seed + 1));
    if (training) {
      std::mt19937_64 rng(seed + 2);
      std::uniform_real_distribution<double> unit(0.0, 1.0);
      std::vector<int64_t> target(codes.size());
      for (std::size_t i = 0; i < codes.size(); ++i) target[i] = unit(rng) < 0.1 + 0.8 * codes[i] / 31.0 ? 1 : 0;
      batch.names.push_back("target");
      batch.cols.push_back(column<arrow::Int64Builder>(target));
    }
  }
  if (dataset != "categorical") {
    auto [x, z] = numbers(rows, seed + 3, dataset == "positive");
    batch.names.push_back("x");
    batch.cols.push_back(column<arrow::DoubleBuilder>(x));
    if (dataset != "mixed") {
      batch.names.push_back("z");
      batch.cols.push_back(column<arrow::DoubleBuilder>(z));
    }
  }
  return batch;
}

fft::Schema schema_of(const fft::Batch& batch) {
  fft::Schema schema;
  for (std::size_t i = 0; i < batch.cols.size(); ++i) {
    const auto id = batch.cols[i]->type_id();
    schema.cols.emplace_back(batch.names[i], id == arrow::Type::DOUBLE  ? "f64"
                                             : id == arrow::Type::INT64 ? "i64"
                                                                        : "utf8");
  }
  return schema;
}

// Fitted as the engine fits a one-step pipeline: resolve, one pass, finalize,
// and resolve again against what the transform will see.
std::unique_ptr<fft::Kernel> fit(const Case& c, const fft::Batch& training, const fft::Schema& input) {
  auto kernel = fft::make_kernel(c.kernel, Json::parse(c.params));
  kernel->stage = "benchmark";
  fft::Schema schema = schema_of(training);
  kernel->resolve(schema);
  if (kernel->needs_fit()) {
    std::vector<std::unique_ptr<fft::AccState>> states;
    states.push_back(kernel->make_state());
    kernel->consume(*states.back(), training);
    kernel->finalize(std::move(states));
  }
  fft::Schema transformed = input;
  kernel->resolve(transformed);
  return kernel;
}

uint64_t hash_batch(const fft::Batch& batch) {
  uint64_t hash = 1469598103934665603ull;
  for (const auto& column : batch.cols) {
    for (const auto& buffer : column->data()->buffers) {
      if (!buffer) continue;
      for (int64_t i = 0; i < buffer->size(); ++i) {
        hash ^= buffer->data()[i];
        hash *= 1099511628211ull;
      }
    }
  }
  return hash;
}

// The sampler suspends the measured thread, reads its program counter and
// resumes it. Suspension costs a few microseconds, so the interval should be
// tens of microseconds or more. The sample is recorded only after the thread
// resumes: recording allocates, and a suspended thread may hold the
// allocator's lock.
class Sampler {
 public:
  Sampler(mach_port_t target, std::chrono::microseconds interval) : target_(target) {
    thread_ = std::thread([this, interval] {
      while (!stop_.load(std::memory_order_relaxed)) {
        if (thread_suspend(target_) == KERN_SUCCESS) {
          arm_thread_state64_t state;
          mach_msg_type_number_t count = ARM_THREAD_STATE64_COUNT;
          const kern_return_t read =
              thread_get_state(target_, ARM_THREAD_STATE64, reinterpret_cast<thread_state_t>(&state), &count);
          thread_resume(target_);
          if (read == KERN_SUCCESS) ++counts_[arm_thread_state64_get_pc(state)];
        }
        std::this_thread::sleep_for(interval);
      }
    });
  }
  std::map<uint64_t, uint64_t> stop() {
    stop_.store(true);
    thread_.join();
    mach_port_deallocate(mach_task_self(), target_);
    return counts_;
  }

 private:
  mach_port_t target_;
  std::atomic<bool> stop_{false};
  std::map<uint64_t, uint64_t> counts_;
  std::thread thread_;
};

struct Timing {
  double median = 0, q1 = 0, q3 = 0;
  std::size_t calls = 0;
};

// At least `iterations` timed calls, and more until `min_seconds` have passed.
template <typename Fn>
Timing time_calls(int iterations, double min_seconds, Fn&& call) {
  for (int i = 0; i < 3; ++i) call();
  std::vector<double> ms;
  const auto begin = Clock::now();
  while (std::cmp_less(ms.size(), iterations) ||
         std::chrono::duration<double>(Clock::now() - begin).count() < min_seconds) {
    const auto start = Clock::now();
    call();
    ms.push_back(std::chrono::duration<double, std::milli>(Clock::now() - start).count());
  }
  std::sort(ms.begin(), ms.end());
  const auto at = [&](double q) { return ms[static_cast<std::size_t>(q * static_cast<double>(ms.size() - 1))]; };
  return {at(0.5), at(0.25), at(0.75), ms.size()};
}

// Stage probes for rank-hot: the same loops as Expand::transform, cut apart.
struct Probes {
  fft::Categories table;
  explicit Probes() {
    for (int i = 0; i < kCategories; ++i) table.sorted.push_back(label(i));
    table.index();
  }
  int64_t position(bool is_null, std::string_view value) const { return is_null ? -1 : table.rank(value); }
};

}  // namespace

int main(int argc, char** argv) {
  int64_t rows = 1000000;
  int iterations = 50;
  double min_seconds = 0;
  bool list = false;
  std::vector<std::string> wanted;
  std::string profile;
  int interval_us = 50;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    const auto next = [&]() -> std::string {
      if (i + 1 >= argc) throw std::runtime_error(arg + " needs a value");
      return argv[++i];
    };
    if (arg == "--rows") {
      rows = std::stoll(next());
    } else if (arg == "--iterations") {
      iterations = std::stoi(next());
    } else if (arg == "--min-seconds") {
      min_seconds = std::stod(next());
    } else if (arg == "--list") {
      list = true;
    } else if (arg == "--case") {
      wanted.push_back(next());
    } else if (arg == "--profile") {
      profile = next();
    } else if (arg == "--interval-us") {
      interval_us = std::stoi(next());
    } else {
      throw std::runtime_error("unknown argument " + arg);
    }
  }
  const Probes probes;
  Json results = Json::array();
  std::map<uint64_t, uint64_t> samples;
  const auto selected = [&](const std::string& name) {
    if (wanted.empty()) return true;
    return std::any_of(wanted.begin(), wanted.end(),
                       [&](const std::string& w) { return name.find(w) != std::string::npos; });
  };
  // Runs time_calls, under the sampler when profiling.
  const auto measure = [&](const auto& call) {
    std::unique_ptr<Sampler> sampler;
    if (!profile.empty())
      sampler = std::make_unique<Sampler>(mach_thread_self(), std::chrono::microseconds(interval_us));
    const Timing timing = time_calls(iterations, min_seconds, call);
    if (sampler) {
      for (const auto& [pc, n] : sampler->stop()) samples[pc] += n;
    }
    return timing;
  };
  const auto record = [&](const std::string& name, const Timing& t, Json extra = Json::object()) {
    extra["case"] = name;
    extra["median_ms"] = t.median;
    extra["q1_ms"] = t.q1;
    extra["q3_ms"] = t.q3;
    extra["calls"] = t.calls;
    results.push_back(std::move(extra));
  };
  for (const Case& c : kCases) {
    const std::string dataset = c.dataset;
    const bool categorical = dataset == "categorical";
    const bool null_sweep = std::string(c.kernel) == "one_hot_encode" || std::string(c.kernel) == "rank_hot_encode";
    for (const bool dictionary : categorical ? std::vector<bool>{false, true} : std::vector<bool>{false}) {
      for (const double nulls : null_sweep ? std::vector<double>{0.0, 0.5, 0.9, 0.99} : std::vector<double>{0.0}) {
        const std::string input = categorical ? (dictionary ? "dict" : "plain") : dataset;
        const std::string where = std::string(c.name) + "/" + input + "/null" + std::to_string(nulls).substr(0, 4);
        const auto stage_selected = [&](const char* stage) { return selected(where + "/" + stage); };
        const bool rank_hot = std::string(c.kernel) == "rank_hot_encode";
        const bool any_stage = std::any_of(kStages.begin(), kStages.end(), [&](const char* stage) {
          return (rank_hot || std::string(stage) == "fit" || std::string(stage) == "transform") &&
                 stage_selected(stage);
        });
        if (list) {
          std::cout << where << '\n';
          continue;
        }
        if (!any_stage) continue;
        try {
          const fft::Batch training = batch_for(dataset, kFitRows, true, dictionary, 0.0);
          const fft::Batch batch = batch_for(dataset, rows, false, dictionary, nulls);
          const fft::Schema input_schema = schema_of(batch);
          if (stage_selected("fit") && nulls == 0.0) {
            uint64_t sink = 0;
            const Timing t = measure([&] { sink ^= fit(c, training, input_schema)->fitted ? 1u : 2u; });
            record(where + "/fit", t, {{"sink", sink}});
          }
          const auto kernel = fit(c, training, input_schema);
          if (stage_selected("transform")) {
            uint64_t sink = 0;
            const Timing t = measure([&] { sink ^= kernel->transform(batch).cols.size(); });
            record(where + "/transform", t, {{"output_hash", hash_batch(kernel->transform(batch))}, {"sink", sink}});
          }
          if (!rank_hot) {
            continue;
          }
          const arrow::Array& array = *batch.cols[0];
          const auto position = [&](bool is_null, std::string_view value) { return probes.position(is_null, value); };
          // Resolution alone: every row looked up, nothing stored.
          if (stage_selected("resolve")) {
            int64_t sink = 0;
            const Timing t = measure([&] {
              fft::for_each_encoded_utf8(array, "benchmark", "c", position,
                                         [&](int64_t, int64_t rank) { sink += rank; });
            });
            record(where + "/resolve", t, {{"sink", sink}});
          }
          // Resolution plus the per-row rank the column fill reads.
          std::vector<int32_t> rank_of(static_cast<std::size_t>(rows), -1);
          const auto record_ranks = [&] {
            fft::for_each_encoded_utf8(array, "benchmark", "c", position, [&](int64_t row, int64_t rank) {
              rank_of[static_cast<std::size_t>(row)] = static_cast<int32_t>(rank);
            });
          };
          if (stage_selected("resolve_rank")) {
            const Timing t = measure([&] {
              std::fill(rank_of.begin(), rank_of.end(), -1);
              record_ranks();
            });
            record(where + "/resolve_rank", t);
          } else {
            record_ranks();
          }
          // Output allocation and the column fill from a recorded rank.
          if (stage_selected("fill")) {
            uint64_t sink = 0;
            const Timing t = measure([&] {
              std::vector<fft::PrimitiveOutput<arrow::Int64Type>> columns;
              columns.reserve(kCategories);
              for (int k = 0; k < kCategories; ++k) columns.emplace_back(rows, "benchmark");
              for (int k = 0; k < kCategories; ++k) {
                int64_t* values = columns[static_cast<std::size_t>(k)].mutable_values();
                for (std::size_t row = 0; row < rank_of.size(); ++row) values[row] = rank_of[row] >= k ? 1 : 0;
              }
              for (auto& output : columns) sink ^= static_cast<uint64_t>(output.finish()->length());
            });
            record(where + "/fill", t, {{"sink", sink}});
          }
          // Output allocation alone, every page touched once by a zero fill.
          if (stage_selected("alloc_zero")) {
            uint64_t sink = 0;
            const Timing t = measure([&] {
              std::vector<fft::PrimitiveOutput<arrow::Int64Type>> columns;
              columns.reserve(kCategories);
              for (int k = 0; k < kCategories; ++k) columns.emplace_back(rows, "benchmark", true);
              for (auto& output : columns) sink ^= static_cast<uint64_t>(output.finish()->length());
            });
            record(where + "/alloc_zero", t, {{"sink", sink}});
          }
        } catch (const std::exception& e) {
          results.push_back({{"case", where}, {"error", e.what()}});
        }
      }
    }
  }
  if (list) return 0;
  if (!profile.empty()) {
    std::ofstream out(profile);
    out << "base 0x" << std::hex << reinterpret_cast<uint64_t>(_dyld_get_image_header(0)) << '\n';
    for (const auto& [pc, n] : samples) out << "0x" << pc << ' ' << std::dec << n << std::hex << '\n';
  }
  std::cout << Json{{"rows", rows}, {"iterations", iterations}, {"results", results}}.dump(1) << '\n';
}
