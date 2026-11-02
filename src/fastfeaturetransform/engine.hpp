// The engine: a flat op list with a cursor.
//
// fit() slices the (topologically ordered) op list into segments — each pass
// streams every row group once, applying ready ops in memory and feeding the
// accumulators of ops fitting this pass — then finalizes and advances. No
// intermediate transformed datasets are written; re-streaming recomputes
// cheap row maps, while the values exact fits keep can spill to temporary
// files.
#pragma once

#include <functional>
#include <iosfwd>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "fastfeaturetransform/execution.hpp"
#include "fastfeaturetransform/kernel.hpp"
#include "fastfeaturetransform/spill.hpp"

namespace fastfeaturetransform {

class RecordBatchSource;

// Where an artifact came from: the manifest layout it uses, the release that
// fitted it, and when. Restored verbatim on load so a load -> save or pickle
// round trip never relabels which release fitted it.
struct Provenance {
  std::string schema_version;
  std::string library_version;
  std::string created_at;
};

// The package version this core was built from (0.0.0 outside the package).
std::string library_version();

// Thread-safety contract: a FittedPipeline's fitted state is immutable after
// construction, so every const method may be called concurrently from any
// number of threads without external locking. Schema resolution is
// serialized, then each call executes against its own projection/schema
// snapshot. Fitted transforms must never read inputs/outputs, which a
// concurrent resolve may update.
class FittedPipeline {
 public:
  FittedPipeline(std::vector<std::unique_ptr<Kernel>> ops, unsigned threads, Provenance provenance)
      : ops_(std::move(ops)), threads_(threads), provenance_(std::move(provenance)) {}

  // Move-only: vector<unique_ptr> lies about copyability, so say it outright
  // (nanobind would otherwise instantiate the ill-formed copy constructor).
  FittedPipeline(FittedPipeline&&) = default;
  FittedPipeline& operator=(FittedPipeline&&) = default;
  FittedPipeline(const FittedPipeline&) = delete;
  FittedPipeline& operator=(const FittedPipeline&) = delete;

  // The batch path: one streaming pass, Parquet in -> ordered part files out.
  // Once the run starts, existing part-NNNNN.parquet files in output_dir are
  // replaced and a run that fails midway leaves none; a failure before the
  // run starts (bad input, schema mismatch) leaves the directory's files
  // untouched.
  std::vector<std::string> transform_batch(const std::vector<std::string>& sources,
                                           const std::string& output_dir) const;
  // Training output goes directly from transformed Arrow batches to packed
  // records, without writing and decoding intermediate Parquet files.
  std::string transform_memmap(const std::vector<std::string>& sources, const std::string& output_dir) const;

  // The record path: one record through every op's transform_row.
  Row transform_row(Row row) const;

  // Fitted state of one stage, as JSON (the artifact's view of it).
  std::string stage_state(const std::string& stage) const;
  int64_t vocab_size(const std::string& stage, const std::string& column) const;

  std::string manifest_json() const;
  // The ops without their state: [{stage, kernel, params}] — cheap to
  // produce however large the vocabularies are.
  std::string stages_json() const;
  // Every op's fitted state, {stage: state} in op order: the artifact's
  // "state" entries, as large as the vocabularies.
  std::string stages_state() const;
  // Writes <dir>/manifest.json atomically (temporary file + rename).
  void save(const std::string& dir) const;
  // Both reject a manifest whose schema_version this build does not read.
  static FittedPipeline load(const std::string& dir, unsigned threads = 0);
  static FittedPipeline load_json(const std::string& manifest, unsigned threads = 0);

 private:
  struct Execution {
    Schema schema;
    std::vector<Kernel*> ops;
    std::vector<std::string> columns;
    int64_t batch_size = 65536;
  };
  // `source`'s footers give list columns their lengths for batch sizing.
  Execution resolve_execution(const Schema& input, const RecordBatchSource& source) const;
  void write_manifest(std::ostream& stream) const;
  static FittedPipeline from_manifest(nlohmann::json parsed, unsigned threads);
  std::vector<std::unique_ptr<Kernel>> ops_;
  unsigned threads_ = 0;
  Provenance provenance_;
  // Guards the kernel re-resolve in resolve_execution (unique_ptr keeps the
  // class movable).
  std::unique_ptr<std::mutex> batch_mutex_ = std::make_unique<std::mutex>();
};

// Parquet in, one memmap out: <output_dir>/data.mmap (row-major packed
// records, the layout of a numpy structured dtype) plus index.json describing
// it; returns the index path. Meant for transform_batch's parts: a string
// column, a null, or a type a memmap cannot hold is an error. A run that fails
// before publishing leaves any previous output in the directory untouched;
// publishing renames the data, then the index, and is not atomic as a pair.
std::string write_memmap(const std::vector<std::string>& sources, const std::string& output_dir, unsigned threads = 0);

class Engine {
 public:
  // threads == 0 means hardware_concurrency.
  explicit Engine(unsigned threads = 0, std::string temp_dir = {})
      : threads_(threads), temp_dir_(std::move(temp_dir)) {}

  // Compile the plan tree, then run the segment loop over the sources.
  FittedPipeline fit(const std::string& plan_json, const std::vector<std::string>& sources) const;

 private:
  unsigned threads_ = 0;
  std::string temp_dir_;
};

// Flatten a nested plan tree ({"type": "pipeline", "steps": [[name, node]...]})
// into kernels with dotted stage names. Lives in C++ so the C++ and Python
// composition APIs share one compiler.
std::vector<std::unique_ptr<Kernel>> compile_plan(const std::string& plan_json);

}  // namespace fastfeaturetransform
