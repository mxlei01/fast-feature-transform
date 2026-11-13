#include "fastfeaturetransform/engine.hpp"

#include <arrow/util/utf8.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
#include <mutex>
#include <span>
#include <sstream>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#include "fastfeaturetransform/io.hpp"

// Baked at build time from the package version (see CMakeLists); 0.0.0 means
// the core was built outside the Python package (the C++ test binary).
#ifndef FFT_LIBRARY_VERSION
#define FFT_LIBRARY_VERSION "0.0.0"
#endif

namespace fastfeaturetransform {
namespace {

// Versions the manifest file format, not the code: bump only when a reader
// of this version can no longer read the JSON layout (a new required field,
// state that only the newer layout holds), never on releases. A layout that
// the reader accepts beside the old one keeps the version: count_encode's
// groups have two under this one, and a reader that knows only the older
// refuses the newer by its state check. Readers accept exactly this version:
// a fitted pipeline should be served by the release that fitted it, so any
// other format is refused with a message to refit.
//
// Every fitted state that resolves its own column list stores it as
// 'columns', in fit order, and loading requires it: rebuilding the order from
// the keys of the fitted table (alphabetical) would move the generated feature
// block and could hand a feature name another feature's value. The polynomial
// contrast basis is stored as fitted too, and used as stored.
constexpr const char* kSchemaVersion = "0.3.0";

unsigned effective_threads(unsigned requested, std::size_t n_tasks) {
  unsigned threads = requested != 0 ? requested : std::max(1u, std::thread::hardware_concurrency());
  return static_cast<unsigned>(std::min<std::size_t>(threads, std::max<std::size_t>(1, n_tasks)));
}

// Leave room for decoded input, output and common kernel scratch. This is an
// automatic sizing heuristic: variable-length values, dictionaries, Parquet
// pages, the batches of fewer than 4,096 rows that a Parquet part holds until
// their row group is written (up to five more) and fitted tables are
// additional, so it is not a process memory cap.
constexpr std::size_t kBatchRowBytes = (std::size_t{16} << 20) / 3;
constexpr int64_t kBatchRows = 65536;

std::size_t estimated_field_bytes(std::string_view tag);

// The estimated bytes of `count` values of the type `tag` names: a vector's
// elements ("f32[768]"), each counted as much as a scalar of its type, or a
// fixed-size binary value's bytes ("fixed_size_binary[16]"), counted one each.
// A narrow element counts as much as a wide one because the Parquet reader
// decodes every element of a list with a repetition and a definition level.
std::size_t repeated_bytes(std::size_t count, std::string_view tag) {
  const std::size_t each = tag == "fixed_size_binary" ? 1 : estimated_field_bytes(tag);
  return count > kBatchRowBytes / each ? kBatchRowBytes : std::max(std::size_t{8}, count * each);
}

std::size_t estimated_field_bytes(std::string_view tag) {
  const auto bracket = tag.rfind('[');
  if (bracket != std::string_view::npos && tag.back() == ']') {
    std::size_t count = 0;
    const auto first = tag.data() + bracket + 1;
    const auto last = tag.data() + tag.size() - 1;
    const auto parsed = std::from_chars(first, last, count);
    if (parsed.ec == std::errc{} && parsed.ptr == last) return repeated_bytes(count, tag.substr(0, bracket));
  }
  if (tag == "bool" || tag == "i8" || tag == "i16" || tag == "i32" || tag == "i64" || tag == "u8" || tag == "u16" ||
      tag == "u32" || tag == "u64" || tag == "f16" || tag == "f32" || tag == "f64")
    return 8;
  return 32;
}

// Keep only each stage's changed fields, not a complete schema per stage.
// Replay after projection is known so unread source columns do not shrink
// batches, while a wide intermediate followed by Select still counts.
class BatchSizing {
 public:
  // A list column's tag says nothing of its length, so the source's
  // footers give its average; the column keeps that width while it passes
  // through under the same tag.
  explicit BatchSizing(const Schema& input, const RecordBatchSource& source) {
    for (const auto& [name, type] : input.cols) {
      const ListShape* shape = source.list_shape(name);
      if (shape == nullptr) {
        input_[name] = estimated_field_bytes(type);
        continue;
      }
      input_[name] = repeated_bytes(static_cast<std::size_t>(std::ceil(shape->average)), shape->element);
      lists_.emplace(name, std::pair{type, input_[name]});
    }
    previous_ = input_;
  }

  void record(const Schema& schema, const std::vector<std::string>& outputs) {
    const std::unordered_set<std::string> produced(outputs.begin(), outputs.end());
    Widths next;
    next.reserve(schema.cols.size());
    std::vector<std::pair<std::string, std::size_t>> changes;
    for (const auto& [name, type] : schema.cols) {
      const auto list = lists_.find(name);
      const auto width =
          list != lists_.end() && list->second.first == type ? list->second.second : estimated_field_bytes(type);
      next.emplace(name, width);
      const auto before = previous_.find(name);
      if (before == previous_.end() || before->second != width || produced.contains(name))
        changes.emplace_back(name, width);
    }
    for (const auto& [name, width] : previous_) {
      if (!next.contains(name)) changes.emplace_back(name, 0);
    }
    changes_.push_back(std::move(changes));
    previous_ = std::move(next);
  }

  int64_t estimate(const std::vector<std::string>& columns, std::span<const uint8_t> transforming = {}) const {
    Widths active;
    if (columns.empty()) {
      active = input_;
    } else {
      for (const auto& name : columns) {
        const auto found = input_.find(name);
        if (found != input_.end()) active.emplace(*found);
      }
    }
    auto widest = width(active);
    for (std::size_t stage = 0; stage < changes_.size(); ++stage) {
      // A fitting op consumes its input without applying its output schema.
      // Stages beyond this fit pass also do not execute.
      if (!transforming.empty() && (stage >= transforming.size() || !transforming[stage])) continue;
      const auto& changes = changes_[stage];
      for (const auto& [name, bytes] : changes) {
        if (bytes == 0)
          active.erase(name);
        else
          active[name] = bytes;
      }
      widest = std::max(widest, width(active));
    }
    return std::min(kBatchRows, static_cast<int64_t>(kBatchRowBytes / widest));
  }

 private:
  using Widths = std::unordered_map<std::string, std::size_t>;
  static std::size_t width(const Widths& columns) {
    std::size_t bytes = 0;
    for (const auto& [name, value] : columns) {
      bytes += value;
      if (bytes >= kBatchRowBytes) return kBatchRowBytes;
    }
    return std::max(std::size_t{1}, bytes);
  }
  Widths input_, previous_;
  std::unordered_map<std::string, std::pair<std::string, std::size_t>> lists_;  // input tag and footer width
  std::vector<std::vector<std::pair<std::string, std::size_t>>> changes_;
};

void flatten_node(const nlohmann::json& node, const std::string& prefix, std::vector<std::unique_ptr<Kernel>>& ops) {
  if (node.value("type", "") == "pipeline") {
    const nlohmann::json& steps = node.at("steps");
    if (!prefix.empty() && steps.empty()) throw Error("pipeline '" + prefix + "' has no steps");
    for (const auto& step : steps) {
      const std::string name = step.at(0).get<std::string>();
      if (name.empty() || name.find('.') != std::string::npos) {
        throw Error("invalid step name '" + name + "': must be non-empty and must not contain '.'");
      }
      flatten_node(step.at(1), prefix.empty() ? name : prefix + "." + name, ops);
    }
    return;
  }
  if (prefix.empty()) throw Error("plan root must be a pipeline");
  auto kernel = make_kernel(node.at("kernel").get<std::string>(), node.at("params"));
  kernel->stage = prefix;
  if (node.contains("state") && !node.at("state").empty()) kernel->load_state(node.at("state"));
  ops.push_back(std::move(kernel));
}

// The invariants every op list must satisfy, whether it came from a plan or
// from an artifact: at least one op, and unique, well-formed stage names.
void validate_ops(const std::vector<std::unique_ptr<Kernel>>& ops) {
  if (ops.empty()) throw Error("no ops to run");
  std::unordered_set<std::string> stages;
  for (const auto& op : ops) {
    const std::string& stage = op->stage;
    const bool well_formed =
        !stage.empty() && stage.front() != '.' && stage.back() != '.' && stage.find("..") == std::string::npos;
    if (!well_formed) throw Error("invalid stage name '" + stage + "'");
    if (!stages.insert(stage).second) throw Error("duplicate stage name '" + stage + "'");
  }
}

// A string field of a manifest object, `fallback` when absent; any other
// type is a malformed manifest rather than a JSON library error.
std::string string_field(const nlohmann::json& object, const char* key, const char* fallback) {
  if (!object.contains(key)) return fallback;
  if (!object.at(key).is_string()) throw Error(std::string("invalid manifest: '") + key + "' must be a string");
  return object.at(key).get<std::string>();
}

// nlohmann's parse errors quote the offending input, which need not be valid
// UTF-8. Keep the message valid UTF-8: drop the quote and what follows it,
// then, if the rest is still invalid, replace every non-ASCII byte with '?'.
std::string json_error_text(const nlohmann::json::exception& e) {
  std::string text = e.what();
  if (const std::size_t cut = text.find("; last read"); cut != std::string::npos) text.resize(cut);
  arrow::util::InitializeUTF8();
  if (!arrow::util::ValidateUTF8(text)) {
    for (char& c : text) {
      if (static_cast<unsigned char>(c) >= 0x80) c = '?';
    }
  }
  return text;
}

// One streaming pass: every task flows through the ops in order — ops fitting
// this pass consume the batch as it stands, everything else transforms it.
// run_pass has two kinds of callers: fit passes (feed, no sink) and the
// sinking passes of transform_batch, transform_memmap and write_memmap (sink,
// no feed). The sink receives consecutive slices of a task from its worker,
// then finishes that task only after its full row count has been validated.
// `columns` (nullptr = all) is the projection handed to the source, so a
// column that no op reads and that does not reach the output is never
// decoded.
struct Sink {
  std::function<void(unsigned worker, std::size_t task, int64_t row_offset, const Batch& batch)> consume;
  std::function<void(unsigned worker, std::size_t task)> finish;
};

void run_pass(const RecordBatchSource& source, unsigned n_threads, const std::vector<Kernel*>& ops,
              const std::unordered_set<const Kernel*>& fitting,
              std::vector<std::vector<std::unique_ptr<AccState>>>* states, const Sink* sink,
              const std::vector<std::string>* columns, int64_t batch_size = 65536) {
  const auto& tasks = source.plan_tasks();
  auto readers = source.readers(n_threads, columns);
  const auto fit_memory = states != nullptr ? current_fit_memory() : nullptr;
  parallel_for(n_threads, tasks.size(), [&](unsigned w, std::size_t i) {
    FitMemoryScope memory_scope(fit_memory);
    int64_t row_offset = 0;
    const int64_t expected_rows = source.task_rows()[i];
    auto consume = [&](Batch batch) {
      const int64_t rows = batch.rows;
      if (rows < 0 || rows > expected_rows - row_offset) throw Error("input task row count changed while reading");
      std::span<const std::string> reusable;
      for (std::size_t op = 0; op < ops.size(); ++op) {
        if (fitting.contains(ops[op])) {
          ops[op]->consume(*(*states)[op][w], batch);
          reusable = {};
        } else {
          batch = ops[op]->transform_owned(std::move(batch), reusable);
          reusable = ops[op]->owned_output_columns();
        }
      }
      if (batch.rows != rows) throw Error("transform changed the number of rows in a batch");
      if (sink != nullptr) sink->consume(w, i, row_offset, batch);
      row_offset += rows;
    };
    readers[w]->read_batches(tasks[i], batch_size, consume);
    if (row_offset != expected_rows) throw Error("input task row count changed while reading");
    if (sink != nullptr && sink->finish) sink->finish(w, i);
  });
}

std::string now_iso8601() {
  return std::format("{:%FT%TZ}", std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now()));
}

void push_unique(std::vector<std::string>& into, const std::string& name) {
  if (std::find(into.begin(), into.end(), name) == into.end()) into.push_back(name);
}

// A memmap field needs a vector's width, which a Parquet file records only in
// the Arrow schema pyarrow stores for a fixed-size list. A list column passing
// through under its source tag takes the width from the source's footers when
// every row group holds the same number of values per row; MemmapSink checks
// each row.
Schema memmap_schema(const RecordBatchSource& source, Schema schema) {
  for (auto& [name, tag] : schema.cols) {
    const ListShape* shape = source.list_shape(name);
    const std::string* stored = source.schema().find(name);
    if (shape != nullptr && shape->width > 0 && stored != nullptr && *stored == tag) {
      tag = shape->element + "[" + std::to_string(shape->width) + "]";
    }
  }
  return schema;
}

// Every source column some op in the pass reads, in first-use order. A column
// an earlier op creates (an output that is not also its input) is read from
// that op, not from the source.
std::vector<std::string> read_columns(const std::vector<Kernel*>& ops) {
  std::vector<std::string> needed;
  std::unordered_set<std::string> created;
  for (const Kernel* op : ops) {
    for (const std::string& name : op->inputs) {
      if (!created.contains(name)) push_unique(needed, name);
    }
    for (const std::string& name : op->outputs) {
      if (std::find(op->inputs.begin(), op->inputs.end(), name) == op->inputs.end()) created.insert(name);
    }
  }
  return needed;
}

// Columns whose presence or type an op's resolve() changed. Together with the
// op's declared outputs (a value-only rewrite keeps the type), this is what a
// downstream reader must wait for while that op is still fitting.
std::vector<std::string> schema_delta(const Schema& before, const Schema& after) {
  std::vector<std::string> delta;
  for (const auto& [name, type] : before.cols) {
    const std::string* now = after.find(name);
    if (now == nullptr || *now != type) push_unique(delta, name);
  }
  for (const auto& [name, type] : after.cols) {
    if (before.find(name) == nullptr) push_unique(delta, name);
  }
  return delta;
}

// Exactly the shape write_part emits — part-NNNNN.parquet, optionally .tmp —
// so a directory holding other writers' parts (Spark's
// part-00000-<uuid>-c000.snappy.parquet) never loses them.
bool is_part_file(const std::filesystem::path& path) {
  std::string name = path.filename().string();
  if (name.ends_with(".tmp")) name.resize(name.size() - 4);
  constexpr std::string_view prefix = "part-";
  constexpr std::string_view suffix = ".parquet";
  if (name.size() < prefix.size() + 5 + suffix.size() || !name.starts_with(prefix) || !name.ends_with(suffix)) {
    return false;
  }
  return std::all_of(name.begin() + static_cast<std::ptrdiff_t>(prefix.size()),
                     name.end() - static_cast<std::ptrdiff_t>(suffix.size()),
                     [](unsigned char c) { return std::isdigit(c) != 0; });
}

// An absolute, `.`/`..`-free path without a trailing separator (symlinks
// left alone, unlike canonical()).
std::filesystem::path lexical_path(const std::string& path) {
  std::filesystem::path normalized = std::filesystem::absolute(path).lexically_normal();
  if (!normalized.has_filename()) normalized = normalized.parent_path();
  return normalized;
}

std::shared_ptr<std::mutex> output_mutex(const std::string& directory) {
  // Different handles may target the same directory. Coordinate output and
  // artifact APIs without retaining locks for every directory ever seen.
  static std::mutex registry_mutex;
  static std::unordered_map<std::string, std::weak_ptr<std::mutex>> registry;
  std::string key;
  struct stat identity{};
  if (::stat(directory.c_str(), &identity) == 0) {
    // stat follows symlinks and identifies case aliases even when canonical()
    // preserves their different spellings on a case-insensitive filesystem.
    key = "inode:" + std::to_string(static_cast<std::uintmax_t>(identity.st_dev)) + ":" +
          std::to_string(static_cast<std::uintmax_t>(identity.st_ino));
  } else {
    std::error_code ec;
    const auto canonical = std::filesystem::canonical(directory, ec);
    key = "path:" + (ec ? lexical_path(directory) : canonical).string();
  }
  std::lock_guard<std::mutex> lock(registry_mutex);
  for (auto entry = registry.begin(); entry != registry.end();) {
    if (entry->second.expired())
      entry = registry.erase(entry);
    else
      ++entry;
  }
  if (auto found = registry.find(key); found != registry.end()) {
    if (auto mutex = found->second.lock()) return mutex;
  }
  auto mutex = std::make_shared<std::mutex>();
  registry[key] = mutex;
  return mutex;
}

// A source living in the output directory would be deleted or overwritten
// mid-read. Both sides exist here, so compare the resolved paths (catches
// any spelling of the same directory) and the literal ones (catches a
// symlinked source whose target lives elsewhere).
void reject_sources_inside(const std::vector<std::string>& sources, const std::string& output_dir) {
  std::error_code ec;
  const std::filesystem::path resolved_dir = std::filesystem::canonical(output_dir, ec);
  const std::filesystem::path literal_dir = lexical_path(output_dir);
  std::unordered_map<std::string, bool> checked_parents;
  const auto same_directory = [&](const std::filesystem::path& parent) {
    if (parent == resolved_dir || parent == literal_dir) return true;
    const std::string key = parent.string();
    if (const auto found = checked_parents.find(key); found != checked_parents.end()) return found->second;
    std::error_code error;
    const bool same = std::filesystem::equivalent(parent, literal_dir, error) && !error;
    checked_parents.emplace(key, same);
    return same;
  };
  for (const std::string& path : sources) {
    const std::filesystem::path resolved = std::filesystem::canonical(path, ec);
    if ((!ec && same_directory(resolved.parent_path())) || same_directory(lexical_path(path).parent_path())) {
      throw Error(path + ": is inside the output directory " + output_dir);
    }
  }
}

// The output directory is ours to overwrite: drop every part file (and any
// half-written .tmp part) so a directory read as a dataset never mixes runs.
void clear_parts(const std::string& dir) {
  std::error_code ec;
  for (const auto& entry : std::filesystem::directory_iterator(dir, ec)) {
    if (is_part_file(entry.path())) std::filesystem::remove(entry.path(), ec);
  }
}

}  // namespace

std::string library_version() { return FFT_LIBRARY_VERSION; }

// The whole executor: N workers racing a shared atomic counter over a static
// task list. fn(worker, task_index) runs to completion. Tasks are claimed in
// index order, so once one fails only the claims above it are skipped: every
// lower task still runs and may fail in its place. The lowest failing task's
// exception is rethrown here, as running the tasks in order would raise it,
// whichever worker fails first. Zero threads means one.
void parallel_for(unsigned n_threads, std::size_t n_tasks, const std::function<void(unsigned, std::size_t)>& fn) {
  if (n_threads == 0) n_threads = 1;
  if (n_threads == 1) {
    for (std::size_t task = 0; task < n_tasks; ++task) fn(0, task);
    return;
  }
  std::atomic<std::size_t> next{0};
  std::atomic<std::size_t> failed{n_tasks};  // the lowest failing task so far
  std::exception_ptr error;
  std::mutex error_mutex;
  const auto work = [&](unsigned w) {
    for (std::size_t i = next.fetch_add(1); i < n_tasks && i < failed.load(); i = next.fetch_add(1)) {
      try {
        fn(w, i);
      } catch (...) {
        std::lock_guard<std::mutex> lock(error_mutex);
        if (i < failed.load()) {
          error = std::current_exception();
          failed.store(i);
        }
      }
    }
  };
  {
    // The calling thread is worker 0, as with one thread. A worker that cannot
    // be started, with no thread or memory left, only means fewer workers:
    // they still claim every task, and the lowest failing task's error does
    // not depend on how many there are.
    std::vector<std::jthread> workers;
    try {
      workers.reserve(n_threads - 1);
      for (unsigned w = 1; w < n_threads; ++w) workers.emplace_back(work, w);
    } catch (const std::exception&) {
      // Run with the workers already started.
    }
    work(0);
  }  // jthread destructors join every worker here
  if (error) std::rethrow_exception(error);
}

std::vector<std::unique_ptr<Kernel>> compile_plan(const std::string& plan_json) {
  std::vector<std::unique_ptr<Kernel>> ops;
  try {
    flatten_node(nlohmann::json::parse(plan_json), "", ops);
  } catch (const nlohmann::json::exception& e) {
    throw Error("invalid plan: " + json_error_text(e));
  }
  validate_ops(ops);
  return ops;
}

FittedPipeline Engine::fit(const std::string& plan_json, const std::vector<std::string>& sources) const {
  // An automatic memory target for the values exact fits keep, not a cap on
  // total process memory.
  FitMemoryOptions buffering;
  buffering.temp_dir = temp_dir_;
  FitMemoryScope memory_scope(std::make_shared<FitMemory>(std::move(buffering)));
  std::vector<std::unique_ptr<Kernel>> ops = compile_plan(plan_json);
  std::unique_ptr<RecordBatchSource> source = open_source(sources, effective_threads(threads_, sources.size()));

  // Resolving the whole list gives every op its column bindings and tells the
  // segment scheduler which columns each one changes. It runs again after
  // every fit pass, because some kernels learn what they emit only from the
  // data, as a one-hot encoder learns its column per category, and the ops
  // after them resolve against those columns. An op given no columns takes
  // its list at the first of these resolves and keeps it (bind_columns in
  // kernels/categories.hpp), so the generated columns never join it.
  std::vector<std::vector<std::string>> deltas;
  BatchSizing sizing(source->schema(), *source);
  auto resolve_all = [&] {
    Schema schema = source->schema();
    sizing = BatchSizing(schema, *source);
    deltas.clear();
    deltas.reserve(ops.size());
    for (auto& op : ops) {
      Schema before = schema;
      op->resolve(schema);
      sizing.record(schema, op->outputs);
      deltas.push_back(schema_delta(before, schema));
    }
  };
  resolve_all();

  unsigned n_threads = effective_threads(threads_, source->plan_tasks().size());

  std::size_t frontier = 0;
  while (frontier < ops.size()) {
    // Find the next segment: walk forward collecting ops until one reads a
    // column that an op fitting in this very pass changes.
    std::unordered_set<const Kernel*> fitting;
    std::unordered_set<std::string> unavailable;
    std::size_t segment_end = frontier;
    for (std::size_t i = frontier; i < ops.size(); ++i) {
      bool blocked = std::any_of(ops[i]->inputs.begin(), ops[i]->inputs.end(),
                                 [&](const std::string& col) { return unavailable.contains(col); });
      if (blocked) break;
      if (ops[i]->needs_fit() && !ops[i]->fitted) {
        fitting.insert(ops[i].get());
        unavailable.insert(deltas[i].begin(), deltas[i].end());
        unavailable.insert(ops[i]->outputs.begin(), ops[i]->outputs.end());
      }
      segment_end = i + 1;
    }

    if (fitting.empty()) {
      frontier = segment_end;  // nothing to learn in this stretch (all stateless/fitted)
      continue;
    }

    // The pass only needs to run ops up to the last one fitting; the ops
    // after it in the segment have nothing to feed.
    std::size_t pass_end = frontier;
    for (std::size_t i = frontier; i < segment_end; ++i) {
      if (fitting.contains(ops[i].get())) pass_end = i + 1;
    }
    std::vector<Kernel*> pass_ops;
    std::vector<uint8_t> transforming(pass_end, 1);
    std::vector<std::vector<std::unique_ptr<AccState>>> states(pass_end);
    for (std::size_t i = 0; i < pass_end; ++i) {
      pass_ops.push_back(ops[i].get());
      if (fitting.contains(ops[i].get())) {
        transforming[i] = 0;
        for (unsigned w = 0; w < n_threads; ++w) states[i].push_back(ops[i]->make_state());
      }
    }

    std::vector<std::string> needed = read_columns(pass_ops);
    run_pass(*source, n_threads, pass_ops, fitting, &states, nullptr, needed.empty() ? nullptr : &needed,
             sizing.estimate(needed, transforming));

    for (std::size_t i = 0; i < pass_end; ++i) {
      if (fitting.contains(ops[i].get())) ops[i]->finalize(std::move(states[i]));
    }
    resolve_all();  // what the ops just fitted emit is only known now
    frontier = segment_end;
  }

  return FittedPipeline(std::move(ops), threads_, Provenance{kSchemaVersion, library_version(), now_iso8601()});
}

std::vector<std::string> FittedPipeline::transform_batch(const std::vector<std::string>& sources,
                                                         const std::string& output_dir) const {
  std::unique_ptr<RecordBatchSource> source = open_source(sources, effective_threads(threads_, sources.size()));
  std::error_code ec;
  std::filesystem::create_directories(output_dir, ec);
  if (ec) throw Error(output_dir + ": cannot create output directory: " + ec.message());
  reject_sources_inside(sources, output_dir);

  const auto directory_mutex = output_mutex(output_dir);
  std::lock_guard<std::mutex> directory_lock(*directory_mutex);
  const Execution execution = resolve_execution(source->schema(), *source);
  // Arrow's Parquet writer counts a row group's rows from its columns, so rows
  // without columns would be written as none. Refuse before reading any rows,
  // not at the first part.
  const auto& task_rows = source->task_rows();
  if (execution.schema.cols.empty() &&
      std::any_of(task_rows.begin(), task_rows.end(), [](int64_t r) { return r > 0; })) {
    throw Error("the pipeline emits no columns, so its rows cannot be written to Parquet");
  }

  clear_parts(output_dir);
  // Each task owns its slot; completion order requires neither a shared push
  // lock nor a sort. A failed run discards this entire local vector.
  std::vector<std::string> paths(source->plan_tasks().size());
  unsigned n_threads = effective_threads(threads_, source->plan_tasks().size());
  std::vector<std::unique_ptr<PartWriter>> writers(n_threads);
  const Sink sink{[&](unsigned worker, std::size_t task, int64_t, const Batch& batch) {
                    if (!writers[worker])
                      writers[worker] = std::make_unique<PartWriter>(output_dir, task, source->task_rows()[task]);
                    writers[worker]->append(batch);
                  },
                  [&](unsigned worker, std::size_t task) {
                    paths[task] = writers[worker]->finish();
                    writers[worker].reset();
                  }};
  try {
    run_pass(*source, n_threads, execution.ops, {}, nullptr, &sink,
             execution.columns.empty() ? nullptr : &execution.columns, execution.batch_size);
  } catch (...) {
    writers.clear();
    clear_parts(output_dir);  // a failed run leaves no part that looks valid
    throw;
  }
  return paths;
}

FittedPipeline::Execution FittedPipeline::resolve_execution(const Schema& input,
                                                            const RecordBatchSource& source) const {
  // Only resolving and reading its mutable planning metadata need the lock.
  // The workers that run the pass later read fitted state, never
  // inputs/outputs. Keep the resulting schema/projection local so another
  // caller may resolve safely.
  std::lock_guard<std::mutex> lock(*batch_mutex_);
  Execution execution{input, {}, {}};
  BatchSizing sizing(input, source);
  for (const auto& op : ops_) {
    op->resolve(execution.schema);
    sizing.record(execution.schema, op->outputs);
    execution.ops.push_back(op.get());
  }
  execution.columns = read_columns(execution.ops);
  std::unordered_set<std::string> produced;
  for (const Kernel* op : execution.ops) produced.insert(op->outputs.begin(), op->outputs.end());
  for (const auto& [name, type] : input.cols) {
    if (execution.schema.find(name) != nullptr && !produced.contains(name)) push_unique(execution.columns, name);
  }
  execution.batch_size = sizing.estimate(execution.columns);
  return execution;
}

std::string FittedPipeline::transform_memmap(const std::vector<std::string>& sources,
                                             const std::string& output_dir) const {
  auto source = open_source(sources, effective_threads(threads_, sources.size()));
  std::error_code ec;
  std::filesystem::create_directories(output_dir, ec);
  if (ec) throw Error(output_dir + ": cannot create output directory: " + ec.message());
  reject_sources_inside(sources, output_dir);
  const auto directory_mutex = output_mutex(output_dir);
  std::lock_guard<std::mutex> directory_lock(*directory_mutex);
  const Execution execution = resolve_execution(source->schema(), *source);
  if (execution.schema.cols.empty()) {
    throw Error("the pipeline emits no columns, and a memmap record needs at least one field");
  }
  MemmapSink memmap(output_dir, memmap_schema(*source, execution.schema), source->task_rows());
  const Sink sink{
      [&](unsigned, std::size_t task, int64_t offset, const Batch& batch) { memmap.write_slice(task, offset, batch); },
      {}};
  run_pass(*source, effective_threads(threads_, source->plan_tasks().size()), execution.ops, {}, nullptr, &sink,
           execution.columns.empty() ? nullptr : &execution.columns, execution.batch_size);
  return memmap.finish();
}

std::string write_memmap(const std::vector<std::string>& sources, const std::string& output_dir, unsigned threads) {
  std::unique_ptr<RecordBatchSource> source = open_source(sources, effective_threads(threads, sources.size()));
  std::error_code ec;
  std::filesystem::create_directories(output_dir, ec);
  if (ec) throw Error(output_dir + ": cannot create output directory: " + ec.message());
  reject_sources_inside(sources, output_dir);
  const auto directory_mutex = output_mutex(output_dir);
  std::lock_guard<std::mutex> directory_lock(*directory_mutex);
  MemmapSink memmap(output_dir, memmap_schema(*source, source->schema()), source->task_rows());
  const Sink sink{
      [&](unsigned, std::size_t task, int64_t offset, const Batch& batch) { memmap.write_slice(task, offset, batch); },
      {}};
  run_pass(*source, effective_threads(threads, source->plan_tasks().size()), {}, {}, nullptr, &sink, nullptr,
           BatchSizing(source->schema(), *source).estimate({}));
  return memmap.finish();
}

Row FittedPipeline::transform_row(Row row) const {
  for (const auto& op : ops_) row = op->transform_row(std::move(row));
  return row;
}

std::string FittedPipeline::stage_state(const std::string& stage) const {
  for (const auto& op : ops_) {
    if (op->stage == stage) return op->state().dump();
  }
  throw Error("no stage named '" + stage + "' in this pipeline");
}

int64_t FittedPipeline::vocab_size(const std::string& stage, const std::string& column) const {
  for (const auto& op : ops_) {
    if (op->stage != stage) continue;
    const auto size = op->vocab_size(column);
    if (!size) throw Error("stage '" + stage + "' does not expose a vocabulary size");
    return *size;
  }
  throw Error("no stage named '" + stage + "' in this pipeline");
}

std::string FittedPipeline::stages_json() const {
  nlohmann::json ops = nlohmann::json::array();
  for (const auto& op : ops_) ops.push_back({{"stage", op->stage}, {"kernel", op->id()}, {"params", op->params()}});
  return ops.dump();
}

std::string FittedPipeline::stages_state() const {
  // Streamed like the manifest, so only one stage's state is serialized at a
  // time, with the stages kept in op order rather than sorted.
  std::ostringstream stream;
  stream << '{';
  bool first = true;
  for (const auto& op : ops_) {
    if (!first) stream << ',';
    first = false;
    stream << nlohmann::json(op->stage).dump() << ':';
    op->write_state(stream);
  }
  stream << '}';
  return std::move(stream).str();
}

std::string FittedPipeline::manifest_json() const {
  std::ostringstream stream;
  write_manifest(stream);
  return std::move(stream).str();
}

void FittedPipeline::write_manifest(std::ostream& stream) const {
  // Write the compact, sorted-key JSON a dump of the whole object would give,
  // but only hold one stage's serialized state at a time. save() writes this
  // directly to its temporary file rather than making another full string.
  stream << "{\"created_at\":" << nlohmann::json(provenance_.created_at).dump()
         << ",\"library_version\":" << nlohmann::json(provenance_.library_version).dump() << ",\"ops\":[";
  bool first = true;
  for (const auto& op : ops_) {
    if (!first) stream << ',';
    first = false;
    stream << "{\"kernel\":" << nlohmann::json(op->id()).dump() << ",\"params\":" << op->params().dump()
           << ",\"stage\":" << nlohmann::json(op->stage).dump() << ",\"state\":";
    op->write_state(stream);
    stream << '}';
  }
  stream << "],\"schema_version\":" << nlohmann::json(provenance_.schema_version).dump() << '}';
}

void FittedPipeline::save(const std::string& dir) const {
  // Stream to a temporary file and publish only a complete artifact. A failed
  // write or rename, serialization included, removes the temporary and keeps
  // the previously published manifest. The temporary is flushed before the
  // rename replaces the old manifest, and the directory after it, so a crash
  // never leaves a partial manifest, and the new name survives one once save
  // returns. A power loss is covered too, as far as sync_to_disk reaches
  // stable storage and flush_to_disk could flush the directory (io.hpp).
  std::error_code ec;
  std::filesystem::create_directories(dir, ec);
  if (ec) throw Error(dir + ": cannot create directory: " + ec.message());
  const auto directory_mutex = output_mutex(dir);
  std::lock_guard<std::mutex> directory_lock(*directory_mutex);
  const std::string path = dir + "/manifest.json";
  const std::string temp = path + ".tmp";
  try {
    std::ofstream file(temp);
    if (!file) throw Error(temp + ": cannot write");
    write_manifest(file);
    file.close();
    if (!file) throw Error(temp + ": failed writing");
    flush_to_disk(temp, false);
  } catch (...) {
    std::filesystem::remove(temp, ec);
    throw;
  }
  std::filesystem::rename(temp, path, ec);
  if (ec) {
    std::error_code ignored;
    std::filesystem::remove(temp, ignored);  // a failed save leaves nothing new behind
    throw Error(path + ": " + ec.message());
  }
  flush_to_disk(dir, true);
}

FittedPipeline FittedPipeline::load(const std::string& dir, unsigned threads) {
  const std::string path = dir + "/manifest.json";
  std::error_code ec;
  if (!std::filesystem::is_regular_file(path, ec)) throw Error(path + ": not found");
  std::ifstream file(path);
  if (!file) throw Error(path + ": cannot read");
  try {
    // Parsing the stream avoids retaining a second full manifest byte string
    // beside the DOM and the reconstructed fitted tables.
    return from_manifest(nlohmann::json::parse(file), threads);
  } catch (const nlohmann::json::exception& e) {
    throw Error("invalid manifest: " + json_error_text(e));
  }
}

FittedPipeline FittedPipeline::load_json(const std::string& manifest, unsigned threads) {
  nlohmann::json parsed;
  try {
    parsed = nlohmann::json::parse(manifest);
  } catch (const nlohmann::json::exception& e) {
    throw Error("invalid manifest: " + json_error_text(e));
  }
  return from_manifest(std::move(parsed), threads);
}

FittedPipeline FittedPipeline::from_manifest(nlohmann::json parsed, unsigned threads) {
  if (!parsed.is_object()) throw Error("invalid manifest: expected a JSON object");
  const std::string schema_version = string_field(parsed, "schema_version", "");
  if (schema_version != kSchemaVersion) {
    throw Error("manifest schema_version '" + schema_version + "' is not supported by this build (expects " +
                kSchemaVersion + "); refit the pipeline and save it again");
  }
  Provenance provenance{schema_version, string_field(parsed, "library_version", "unknown"),
                        string_field(parsed, "created_at", "unknown")};
  if (!parsed.contains("ops") || !parsed.at("ops").is_array()) throw Error("invalid manifest: 'ops' must be a list");

  std::vector<std::unique_ptr<Kernel>> ops;
  for (auto& entry : parsed.at("ops")) {
    if (!entry.is_object()) throw Error("invalid manifest: every op must be an object");
    const std::string stage = string_field(entry, "stage", "");
    try {
      auto kernel = make_kernel(entry.at("kernel").get<std::string>(), entry.at("params"));
      kernel->stage = stage;
      kernel->load_state(entry.at("state"));
      ops.push_back(std::move(kernel));
      entry = nullptr;  // release each stage's DOM as its fitted table takes over
    } catch (const nlohmann::json::exception& e) {
      throw Error("invalid manifest: op '" + stage + "': " + json_error_text(e));
    }
  }
  validate_ops(ops);
  return FittedPipeline(std::move(ops), threads, std::move(provenance));
}

}  // namespace fastfeaturetransform
