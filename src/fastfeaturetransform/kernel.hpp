// The one kernel interface. A kernel = params + fitted state -> behavior.
//
// Fit-needing kernels expose a mergeable accumulator (make_state / consume /
// finalize) so the engine can fold row groups across workers and merge once.
// Transform is row-wise by contract: transform() on a batch and
// transform_row() on a record must agree value-for-value.
#pragma once

#include <iosfwd>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "fastfeaturetransform/core.hpp"

namespace fastfeaturetransform {

// Per-worker accumulator state; concrete kernels downcast their own.
struct AccState {
  virtual ~AccState() = default;
};

class Kernel {
 public:
  virtual ~Kernel() = default;

  virtual std::string id() const = 0;
  virtual nlohmann::json params() const = 0;

  // Resolve column bindings against the incoming schema (raising Error on a
  // missing/mistyped column) and update the schema to what flows out. On a
  // fitted kernel this may only validate and assign inputs/outputs: it runs
  // inside transform_batch and transform_memmap while transform_row may be
  // running elsewhere.
  virtual void resolve(Schema& schema) = 0;

  virtual bool needs_fit() const { return false; }
  virtual std::unique_ptr<AccState> make_state() const { return nullptr; }
  virtual void consume(AccState& state, const Batch& batch) const { (void)state, (void)batch; }
  // Merge all worker states and freeze the fitted result onto the kernel.
  virtual void finalize(std::vector<std::unique_ptr<AccState>> states) { (void)states; }

  virtual nlohmann::json state() const { return nlohmann::json::object(); }
  // Write the same JSON as state().dump(). The default builds the whole DOM;
  // an override can stream it instead.
  virtual void write_state(std::ostream& output) const;
  // Small fitted metadata must not require materializing a potentially huge
  // state JSON object. Only kernels whose state lists vocab_sizes support it.
  virtual std::optional<int64_t> vocab_size(const std::string& column) const {
    (void)column;
    return std::nullopt;
  }
  // Restore fitted state from an artifact. resolve() never runs on load, so
  // this (or the constructor) must also fill inputs/outputs; validate the
  // JSON shape and throw Error rather than trusting a hand-edited file.
  virtual void load_state(const nlohmann::json& state) { (void)state; }

  // transform(), transform_owned() and transform_row() must be pure functions
  // of the params and fitted state: they run concurrently on many threads.
  virtual Batch transform(Batch batch) const = 0;
  // Only the executor may grant reuse of a preceding stage's fresh buffers.
  // Empty by default: source/refcount observations never imply ownership.
  virtual Batch transform_owned(Batch batch, std::span<const std::string> reusable) const {
    (void)reusable;
    return transform(std::move(batch));
  }
  virtual std::span<const std::string> owned_output_columns() const { return {}; }
  virtual Row transform_row(Row row) const = 0;

  // Columns this kernel reads / writes, filled by resolve(). `inputs` must
  // name every column transform()/consume() touches: the source decodes only
  // the columns the pass's kernels list, plus those reaching the output
  // untouched.
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
  // Dotted stage name from the plan ("features.categorical.encode").
  std::string stage;
  // True once finalize() ran or state was loaded from an artifact.
  bool fitted = false;
};

// The kernel catalog: one flat table in one place (kernel.cpp), with each
// kernel implemented under kernels/. Throws Error on an unknown id or
// malformed params so artifacts fail loudly at load time.
std::unique_ptr<Kernel> make_kernel(const std::string& id, const nlohmann::json& params);

std::vector<std::string> list_kernels();

}  // namespace fastfeaturetransform
