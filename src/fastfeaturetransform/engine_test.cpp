// Explicit C++ tests for the engine: the executor, the plan compiler, and a
// pure-C++ end-to-end fit -> artifact -> transform with no Python anywhere.

#include "fastfeaturetransform/engine.hpp"

#include <arrow/array.h>
#include <sys/resource.h>

#include <algorithm>
#include <atomic>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <chrono>
#include <csignal>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

#include "fastfeaturetransform/io.hpp"
#include "fastfeaturetransform/test_support.hpp"

namespace fft = fastfeaturetransform;
using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::Equals;
using fft::testing::f64_column;
using fft::testing::make_batch;
using fft::testing::read_task;
using fft::testing::TempDir;
using fft::testing::utf8_column;

TEST_CASE("parallel_for runs every task exactly once with valid worker ids") {
  constexpr std::size_t kTasks = 1000;
  constexpr unsigned kThreads = 8;
  std::vector<std::atomic<int>> runs(kTasks);
  std::atomic<bool> bad_worker_id{false};
  // Catch2 assertions are not thread-safe: workers only record, the main
  // thread asserts.
  fft::parallel_for(kThreads, kTasks, [&](unsigned worker, std::size_t task) {
    if (worker >= kThreads) bad_worker_id.store(true);
    runs[task].fetch_add(1);
  });
  REQUIRE_FALSE(bad_worker_id.load());
  for (const auto& count : runs) REQUIRE(count.load() == 1);
}

TEST_CASE("parallel_for propagates a worker exception") {
  REQUIRE_THROWS_AS(fft::parallel_for(4, 100,
                                      [&](unsigned, std::size_t task) {
                                        if (task == 5) throw fft::Error("boom");
                                      }),
                    fft::Error);
}

// REGRESSION: parallel_for rethrows the lowest failing task's exception,
// whichever worker fails first. The first exception to arrive used to win, so
// a run could report a later task's error that running in order never reaches.
// Task 1 fails only after task 6 has; task 1's error must still be the one.
TEST_CASE("parallel_for reports the lowest failing task's exception, whichever fails first") {
  for (int round = 0; round < 20; ++round) {
    CAPTURE(round);
    std::atomic<bool> later_failed{false};
    REQUIRE_THROWS_WITH(fft::parallel_for(4, 16,
                                          [&](unsigned, std::size_t task) {
                                            if (task == 1) {
                                              for (int wait = 0; wait < 2000 && !later_failed.load(); ++wait) {
                                                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                                              }
                                              throw fft::Error("task 1");
                                            }
                                            if (task == 6) {
                                              later_failed.store(true);
                                              throw fft::Error("task 6");
                                            }
                                          }),
                        "task 1");
  }
}

// REGRESSION: a worker that cannot be started only means fewer workers, and
// the rest still run every task. It used to throw out of parallel_for after
// the others had run every task, dropping their errors and failing a run that
// had succeeded. On Linux a thread counts against RLIMIT_NPROC, so a soft
// limit of 1 stops every new one; elsewhere, or for a user the limit does not
// bind, threads still start and there is nothing to check.
TEST_CASE("parallel_for runs every task when its workers cannot be started") {
  rlimit original{};
  REQUIRE(getrlimit(RLIMIT_NPROC, &original) == 0);
  struct Restore {
    rlimit limit;
    ~Restore() { setrlimit(RLIMIT_NPROC, &limit); }
  } restore{original};
  rlimit lowered = original;
  lowered.rlim_cur = 1;
  REQUIRE(setrlimit(RLIMIT_NPROC, &lowered) == 0);
  bool blocked = false;
  try {
    std::thread([] {}).join();
  } catch (const std::system_error&) {
    blocked = true;
  }
  if (!blocked) SKIP("threads start despite RLIMIT_NPROC here");

  std::vector<std::atomic<int>> runs(64);
  std::atomic<unsigned> highest_worker{0};
  fft::parallel_for(4, runs.size(), [&](unsigned worker, std::size_t task) {
    highest_worker.store(std::max(highest_worker.load(), worker));
    runs[task].fetch_add(1);
  });
  for (const auto& count : runs) REQUIRE(count.load() == 1);
  REQUIRE(highest_worker.load() == 0);  // the calling thread ran them all

  for (auto& count : runs) count.store(0);
  REQUIRE_THROWS_WITH(fft::parallel_for(4, runs.size(),
                                        [&](unsigned, std::size_t task) {
                                          runs[task].fetch_add(1);
                                          if (task == 9 || task == 40) throw fft::Error("task " + std::to_string(task));
                                        }),
                      "task 9");
  for (std::size_t task = 0; task < runs.size(); ++task) REQUIRE(runs[task].load() == (task <= 9 ? 1 : 0));
}

TEST_CASE("compile_plan flattens nested pipelines into dotted stages") {
  const std::string plan = R"({
    "type": "pipeline",
    "steps": [["features", {"type": "pipeline", "steps": [
      ["select", {"kernel": "select", "params": {"columns": ["device"]}}],
      ["encode", {"kernel": "label_encode", "params": {"columns": ["device"], "min_count": 1}}]
    ]}]]
  })";
  auto ops = fft::compile_plan(plan);
  REQUIRE(ops.size() == 2);
  REQUIRE(ops[0]->stage == "features.select");
  REQUIRE(ops[1]->stage == "features.encode");
}

TEST_CASE("compile_plan rejects dotted step names and duplicate stages") {
  REQUIRE_THROWS_AS(
      fft::compile_plan(R"({"type":"pipeline","steps":[["a.b",{"kernel":"select","params":{"columns":["x"]}}]]})"),
      fft::Error);
  REQUIRE_THROWS_AS(fft::compile_plan(R"({"type":"pipeline","steps":[
        ["a",{"kernel":"select","params":{"columns":["x"]}}],
        ["a",{"kernel":"select","params":{"columns":["x"]}}]]})"),
                    fft::Error);
}

TEST_CASE("compile_plan rejects a bare kernel root, empty nested pipelines, and malformed JSON") {
  REQUIRE_THROWS_WITH(fft::compile_plan(R"({"kernel":"select","params":{"columns":["x"]}})"),
                      ContainsSubstring("root must be a pipeline"));
  REQUIRE_THROWS_WITH(fft::compile_plan(R"({"type":"pipeline","steps":[["empty",{"type":"pipeline","steps":[]}]]})"),
                      ContainsSubstring("has no steps"));
  REQUIRE_THROWS_WITH(fft::compile_plan("not json"), ContainsSubstring("invalid plan"));
  REQUIRE_THROWS_WITH(fft::compile_plan(R"({"type":"pipeline","steps":[["s",{"kernel":"select"}]]})"),
                      ContainsSubstring("invalid plan"));
}

TEST_CASE("parallel_for treats zero threads as one") {
  std::atomic<int> runs{0};
  fft::parallel_for(0, 5, [&](unsigned, std::size_t) { runs.fetch_add(1); });
  REQUIRE(runs.load() == 5);
}

TEST_CASE("single-worker execution stays on the caller and stops immediately on failure") {
  const auto caller = std::this_thread::get_id();
  for (unsigned threads : {0u, 1u}) {
    std::size_t visited = 0;
    REQUIRE_THROWS_WITH(fft::parallel_for(threads, 8,
                                          [&](unsigned worker, std::size_t task) {
                                            REQUIRE(std::this_thread::get_id() == caller);
                                            REQUIRE(worker == 0);
                                            REQUIRE(task == visited++);
                                            if (task == 3) throw fft::Error("stop");
                                          }),
                        "stop");
    REQUIRE(visited == 4);
  }
}

namespace {

const std::string kManifest = R"({
  "schema_version": "0.3.0", "library_version": "9.9.9", "created_at": "1999-01-01T00:00:00Z",
  "ops": [{"stage": "enc", "kernel": "label_encode", "params": {"columns": ["device"], "min_count": 1},
           "state": {"columns": ["device"], "label_maps": {"device": {"ios": 1, "web": 2}},
                     "vocab_sizes": {"device": 3}}}]
})";

}  // namespace

TEST_CASE("artifacts keep their provenance across load and reject other schema versions") {
  fft::FittedPipeline loaded = fft::FittedPipeline::load_json(kManifest);
  nlohmann::json again = nlohmann::json::parse(loaded.manifest_json());
  REQUIRE(again.at("schema_version") == "0.3.0");
  REQUIRE(again.at("library_version") == "9.9.9");
  REQUIRE(again.at("created_at") == "1999-01-01T00:00:00Z");
  REQUIRE(loaded.manifest_json() == loaded.manifest_json());  // deterministic

  nlohmann::json other = nlohmann::json::parse(kManifest);
  for (const char* version : {"0.2.0", "42.0.0"}) {
    other["schema_version"] = version;
    REQUIRE_THROWS_WITH(fft::FittedPipeline::load_json(other.dump()),
                        ContainsSubstring("schema_version '" + std::string(version) + "'") &&
                            ContainsSubstring("refit the pipeline and save it again"));
  }
  other.erase("schema_version");
  REQUIRE_THROWS_WITH(fft::FittedPipeline::load_json(other.dump()), ContainsSubstring("schema_version"));
}

TEST_CASE("a polynomial artifact serves its stored basis and refuses one without it") {
  nlohmann::json manifest = nlohmann::json::parse(R"({
    "schema_version": "0.3.0", "library_version": "9.9.9", "created_at": "1999-01-01T00:00:00Z",
    "ops": [{"stage": "poly", "kernel": "polynomial_encode",
             "params": {"columns": ["c"], "handle_unknown": "value", "handle_missing": "value"},
             "state": {"columns": ["c"], "categories": {"c": ["a", "b"]}, "has_null": {"c": false},
                       "polynomial_basis": {"version": 1, "matrices": {"c": [[-0.25], [0.25]]}}}}]
  })");
  // Persisted values are authoritative; reading and saving must not recreate
  // a different matrix from the categories. Two categories would give about
  // -0.7071 and 0.7071, so an output of -0.25 shows the stored one is served.
  auto loaded = fft::FittedPipeline::load_json(manifest.dump());
  REQUIRE(nlohmann::json::parse(loaded.manifest_json()) == manifest);
  REQUIRE(loaded.transform_row({{"c", std::string("a")}}) == fft::Row{{"c_0", -0.25}});

  // A stripped basis is refused rather than rebuilt.
  manifest["ops"][0]["state"].erase("polynomial_basis");
  REQUIRE_THROWS_WITH(fft::FittedPipeline::load_json(manifest.dump()),
                      ContainsSubstring("poly: state must hold a 'polynomial_basis' object"));
}

TEST_CASE("load_json enforces the same invariants as compile_plan, with context") {
  nlohmann::json manifest = nlohmann::json::parse(kManifest);
  nlohmann::json op = manifest.at("ops").at(0);

  nlohmann::json empty = manifest;
  empty["ops"] = nlohmann::json::array();
  REQUIRE_THROWS_WITH(fft::FittedPipeline::load_json(empty.dump()), ContainsSubstring("no ops"));

  nlohmann::json duplicate = manifest;
  duplicate["ops"].push_back(op);
  REQUIRE_THROWS_WITH(fft::FittedPipeline::load_json(duplicate.dump()), ContainsSubstring("duplicate stage"));

  for (const char* stage : {"", ".a", "a.", "a..b"}) {
    nlohmann::json bad = manifest;
    bad["ops"][0]["stage"] = stage;
    REQUIRE_THROWS_WITH(fft::FittedPipeline::load_json(bad.dump()), ContainsSubstring("invalid stage name"));
  }

  nlohmann::json missing = manifest;
  missing["ops"][0].erase("params");
  REQUIRE_THROWS_WITH(fft::FittedPipeline::load_json(missing.dump()), ContainsSubstring("op 'enc'"));

  REQUIRE_THROWS_WITH(fft::FittedPipeline::load_json("[]"), ContainsSubstring("expected a JSON object"));
  REQUIRE_THROWS_WITH(fft::FittedPipeline::load_json("{"), ContainsSubstring("invalid manifest"));
  REQUIRE_THROWS_WITH(fft::FittedPipeline::load_json(R"({"schema_version":"0.3.0","ops":{}})"),
                      ContainsSubstring("'ops' must be a list"));

  // Wrong JSON types for the provenance and op fields are still our errors.
  for (const char* field : {"schema_version", "library_version", "created_at"}) {
    nlohmann::json typed = manifest;
    typed[field] = 5;
    REQUIRE_THROWS_WITH(fft::FittedPipeline::load_json(typed.dump()),
                        ContainsSubstring(std::string("'") + field + "' must be a string"));
  }
  nlohmann::json scalar_op = manifest;
  scalar_op["ops"] = nlohmann::json::array({5});
  REQUIRE_THROWS_WITH(fft::FittedPipeline::load_json(scalar_op.dump()),
                      ContainsSubstring("every op must be an object"));
  nlohmann::json typed_stage = manifest;
  typed_stage["ops"][0]["stage"] = 5;
  REQUIRE_THROWS_WITH(fft::FittedPipeline::load_json(typed_stage.dump()),
                      ContainsSubstring("'stage' must be a string"));

  // A parse error quoting non-UTF-8 input still yields an ASCII message.
  const std::string bad_bytes = std::string(R"({"schema_version":"0.3.0","ops":[{"stage":"i)") + "\xff" + R"(s"}]})";
  try {
    fft::FittedPipeline::load_json(bad_bytes);
    FAIL("expected an error");
  } catch (const fft::Error& e) {
    const std::string message = e.what();
    REQUIRE(message.starts_with("invalid manifest"));
    for (char c : message) REQUIRE(static_cast<unsigned char>(c) < 0x80);
  }
}

// make_kernel's refusal reaches the caller as it is. A plan fails when it is
// compiled, which a fit does before it opens a file, and a manifest when it
// is loaded.
TEST_CASE("a plan and a manifest are refused for a parameter name the kernel does not report") {
  const std::string message = "label_encode: unknown parameter 'min_counts' (accepted: columns, min_count)";
  const std::string plan = R"({"type": "pipeline", "steps": [
    ["enc", {"kernel": "label_encode", "params": {"columns": ["device"], "min_counts": 2}}]]})";
  REQUIRE_THROWS_WITH(fft::compile_plan(plan), Equals(message));
  REQUIRE_THROWS_WITH(fft::Engine(1).fit(plan, {"no-such-file.parquet"}), Equals(message));

  nlohmann::json manifest = nlohmann::json::parse(kManifest);
  nlohmann::json& params = manifest["ops"][0]["params"];
  params["min_counts"] = params.at("min_count");
  params.erase("min_count");
  REQUIRE_THROWS_WITH(fft::FittedPipeline::load_json(manifest.dump()), Equals(message));
}

TEST_CASE("a failed save leaves nothing new behind") {
  TempDir dir;
  fft::FittedPipeline loaded = fft::FittedPipeline::load_json(kManifest);
  std::filesystem::create_directories(dir.path / "artifact" / "manifest.json");  // a directory in the way
  REQUIRE_THROWS_AS(loaded.save((dir.path / "artifact").string()), fft::Error);
  REQUIRE_FALSE(std::filesystem::exists(dir.path / "artifact" / "manifest.json.tmp"));
}

// REGRESSION: a save whose temporary cannot be opened removes nothing. Its
// cleanup used to remove what the temporary's name reached when that was an
// empty directory: one in the temporary's place, or the directory a NUL in
// the path cuts the name to, which the caller did not name.
TEST_CASE("a save that cannot open its temporary removes no directory") {
  TempDir dir;
  fft::FittedPipeline loaded = fft::FittedPipeline::load_json(kManifest);
  SECTION("a directory in the temporary's place") {
    const std::filesystem::path in_place = dir.path / "artifact" / "manifest.json.tmp";
    std::filesystem::create_directories(in_place);
    REQUIRE_THROWS_WITH(loaded.save((dir.path / "artifact").string()), ContainsSubstring("cannot write"));
    REQUIRE(std::filesystem::is_directory(in_place));
  }
  SECTION("a NUL in the path") {
    const std::filesystem::path kept = dir.path / "kept";
    std::filesystem::create_directory(kept);
    const std::string cut = kept.string() + std::string(1, '\0') + "junk";
    REQUIRE_THROWS_AS(loaded.save(cut), fft::Error);  // what() ends at the NUL
    REQUIRE(std::filesystem::is_directory(kept));
  }
}

// A save whose write fails after its temporary opened removes the temporary
// and publishes nothing. A file size limit of 16 bytes stages a disk that
// refuses the write; the signal sent with the refusal is ignored meanwhile.
// Nothing is asserted while the limit holds, because it also binds a file
// that Catch2 reports to.
TEST_CASE("a save whose write fails removes its temporary") {
  TempDir dir;
  fft::FittedPipeline loaded = fft::FittedPipeline::load_json(kManifest);
  const std::filesystem::path artifact = dir.path / "artifact";
  std::filesystem::create_directory(artifact);
  rlimit original{};
  REQUIRE(getrlimit(RLIMIT_FSIZE, &original) == 0);
  std::string message = "the file size limit could not be set";
  {
    struct Restore {
      rlimit limit;
      void (*handler)(int);
      ~Restore() {
        setrlimit(RLIMIT_FSIZE, &limit);
        std::signal(SIGXFSZ, handler);
      }
    } restore{original, std::signal(SIGXFSZ, SIG_IGN)};
    rlimit lowered = original;
    lowered.rlim_cur = 16;
    if (setrlimit(RLIMIT_FSIZE, &lowered) == 0) {
      message = "save did not throw";
      try {
        loaded.save(artifact.string());
      } catch (const fft::Error& error) {
        message = error.message;
      }
    }
  }
  REQUIRE(message == (artifact / "manifest.json.tmp").string() + ": failed writing");
  REQUIRE(std::filesystem::is_empty(artifact));
}

TEST_CASE("save writes atomically and load names a missing manifest") {
  TempDir dir;
  fft::FittedPipeline loaded = fft::FittedPipeline::load_json(kManifest);
  loaded.save((dir.path / "artifact").string());
  REQUIRE(std::filesystem::is_regular_file(dir.path / "artifact" / "manifest.json"));
  REQUIRE_FALSE(std::filesystem::exists(dir.path / "artifact" / "manifest.json.tmp"));
  REQUIRE_THROWS_WITH(fft::FittedPipeline::load((dir.path / "nowhere").string()), ContainsSubstring("not found"));
}

TEST_CASE("pure C++ end to end: fit over parquet, save, load, transform") {
  TempDir dir;
  // Two source "row groups" as two part files written by our own sink.
  fft::write_part(dir.path.string(), 0,
                  make_batch({"device", "price"}, {utf8_column({"mob", "desk", "mob"}), f64_column({1.0, 2.0, 3.0})}));
  fft::write_part(dir.path.string(), 1,
                  make_batch({"device", "price"}, {utf8_column({"tab", std::nullopt}), f64_column({4.0, 5.0})}));
  std::vector<std::string> sources{dir.path.string() + "/part-00000.parquet",
                                   dir.path.string() + "/part-00001.parquet"};

  const std::string plan = R"({
    "type": "pipeline",
    "steps": [
      ["select", {"kernel": "select", "params": {"columns": ["device"]}}],
      ["encode", {"kernel": "label_encode", "params": {"columns": null, "min_count": 1}}]
    ]
  })";
  fft::FittedPipeline fitted = fft::Engine(2).fit(plan, sources);
  REQUIRE(nlohmann::json::parse(fitted.manifest_json()).at("schema_version") == "0.3.0");

  fft::Row row = fitted.transform_row({{"device", std::string("mob")}, {"price", 9.9}});
  REQUIRE(row.size() == 1);                        // select dropped price
  REQUIRE(std::get<int64_t>(row[0].second) == 2);  // byte-sorted: desk=1, mob=2, tab=3

  // Artifact round-trip through disk, then batch transform of the sources.
  fitted.save((dir.path / "artifact").string());
  fft::FittedPipeline loaded = fft::FittedPipeline::load((dir.path / "artifact").string());
  const std::string out = (dir.path / "out").string();
  std::filesystem::create_directories(out);
  std::ofstream((dir.path / "out" / "part-00007.parquet").string()) << "stale";  // from an earlier, larger run
  std::ofstream((dir.path / "out" / "part-00000-1234-c000.snappy.parquet").string()) << "spark";  // not ours
  std::vector<std::string> parts = loaded.transform_batch(sources, out);
  REQUIRE(parts.size() == 2);
  REQUIRE_FALSE(std::filesystem::exists(dir.path / "out" / "part-00007.parquet"));
  REQUIRE(std::filesystem::exists(dir.path / "out" / "part-00000-1234-c000.snappy.parquet"));

  // The parts hold the batch-path output, which must equal the record path.
  auto written = fft::open_source(parts);
  std::vector<int64_t> batch_values;
  for (std::size_t task = 0; task < written->plan_tasks().size(); ++task) {
    fft::Batch batch = read_task(*written, task);
    REQUIRE(batch.names == std::vector<std::string>{"device"});
    const auto& values = static_cast<const arrow::Int64Array&>(*batch.cols[0]);
    for (int64_t i = 0; i < values.length(); ++i) batch_values.push_back(values.Value(i));
  }
  REQUIRE(batch_values == std::vector<int64_t>{2, 1, 2, 3, 0});

  fft::Row unknown = loaded.transform_row({{"device", std::string("spaceship")}});
  REQUIRE(std::get<int64_t>(unknown[0].second) == 0);
}

TEST_CASE("stages_state maps every stage to its fitted state in op order") {
  TempDir dir;
  fft::write_part(dir.path.string(), 0, make_batch({"z", "a"}, {utf8_column({"x", "y"}), utf8_column({"p", "q"})}));
  // Op order is not sorted order, and one stage name needs escaping.
  const std::string plan = R"({"type": "pipeline", "steps": [
    ["zz", {"kernel": "label_encode", "params": {"columns": ["z"]}}],
    ["a\"b", {"kernel": "select", "params": {"columns": ["z", "a"]}}]]})";
  fft::FittedPipeline fitted = fft::Engine(1).fit(plan, {dir.path.string() + "/part-00000.parquet"});

  const auto states = nlohmann::ordered_json::parse(fitted.stages_state());
  std::vector<std::string> stages;
  for (const auto& [stage, state] : states.items()) stages.push_back(stage);
  REQUIRE(stages == std::vector<std::string>{"zz", "a\"b"});
  REQUIRE(states.at("zz") == nlohmann::ordered_json::parse(fitted.stage_state("zz")));
  REQUIRE(states.at("zz").at("label_maps") == nlohmann::ordered_json({{"z", {{"x", 1}, {"y", 2}}}}));
  REQUIRE(states.at("a\"b") == nlohmann::ordered_json::object());

  // The same states the artifact stores, so a loaded pipeline reports them too.
  fitted.save((dir.path / "artifact").string());
  REQUIRE(fft::FittedPipeline::load((dir.path / "artifact").string()).stages_state() == fitted.stages_state());
}

TEST_CASE("a fit whose second encoder depends on the first runs in two passes") {
  TempDir dir;
  // Two row groups so the second pass has to re-stream and re-apply e1.
  fft::write_part(
      dir.path.string(), 0,
      make_batch({"a", "b", "c"}, {utf8_column({"x", "y"}), utf8_column({"p", "q"}), utf8_column({"z", "z"})}));
  fft::write_part(dir.path.string(), 1,
                  make_batch({"a", "b", "c"},
                             {utf8_column({"x", std::nullopt}), utf8_column({"r", "p"}), utf8_column({"z", "z"})}));
  std::vector<std::string> sources{(dir.path / "part-00000.parquet").string(),
                                   (dir.path / "part-00001.parquet").string()};
  // e1 encodes a; select keeps a (now i64) and b; e2 with columns=null sees
  // only b as utf8, and it can only fit once e1 has been applied.
  const std::string plan = R"({"type": "pipeline", "steps": [
    ["e1", {"kernel": "label_encode", "params": {"columns": ["a"]}}],
    ["s", {"kernel": "select", "params": {"columns": ["a", "b"]}}],
    ["e2", {"kernel": "label_encode", "params": {"columns": null}}]]})";
  fft::FittedPipeline fitted = fft::Engine(2).fit(plan, sources);

  nlohmann::json e1 = nlohmann::json::parse(fitted.stage_state("e1"));
  nlohmann::json e2 = nlohmann::json::parse(fitted.stage_state("e2"));
  REQUIRE(e1.at("label_maps") == nlohmann::json({{"a", {{"x", 1}, {"y", 2}}}}));
  REQUIRE(e2.at("label_maps") == nlohmann::json({{"b", {{"p", 1}, {"q", 2}, {"r", 3}}}}));

  // Batch output equals the record path for every source row.
  std::vector<std::string> parts = fitted.transform_batch(sources, (dir.path / "out").string());
  auto written = fft::open_source(parts);
  std::vector<std::pair<int64_t, int64_t>> batch_rows;
  for (std::size_t task = 0; task < written->plan_tasks().size(); ++task) {
    fft::Batch batch = read_task(*written, task);
    REQUIRE(batch.names == std::vector<std::string>{"a", "b"});
    const auto& a = static_cast<const arrow::Int64Array&>(*batch.cols[0]);
    const auto& b = static_cast<const arrow::Int64Array&>(*batch.cols[1]);
    for (int64_t i = 0; i < batch.rows; ++i) batch_rows.emplace_back(a.Value(i), b.Value(i));
  }
  std::vector<std::pair<std::optional<std::string>, std::string>> inputs{
      {"x", "p"}, {"y", "q"}, {"x", "r"}, {std::nullopt, "p"}};
  REQUIRE(batch_rows.size() == inputs.size());
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    fft::RowValue a = inputs[i].first.has_value() ? fft::RowValue(*inputs[i].first) : fft::RowValue(std::monostate{});
    fft::Row row = fitted.transform_row({{"a", a}, {"b", inputs[i].second}, {"c", std::string("z")}});
    REQUIRE(row.size() == 2);
    REQUIRE(std::get<int64_t>(row[0].second) == batch_rows[i].first);
    REQUIRE(std::get<int64_t>(row[1].second) == batch_rows[i].second);
  }
}

// REGRESSION: a step given no columns takes the columns known when the fit
// starts, whichever pass it fits in. The scaler used to take the one-hot
// columns too when a step in front made it fit a pass after the encoder.
TEST_CASE("a step given no columns takes the same columns whichever pass it fits in") {
  TempDir dir;
  fft::write_part(dir.path.string(), 0,
                  make_batch({"c", "x"}, {utf8_column({"a", "b", "a", "c"}), f64_column({1.0, 2.0, 3.0, 6.0})}));
  const std::vector<std::string> sources{(dir.path / "part-00000.parquet").string()};
  const std::string alone_plan = R"({"type": "pipeline", "steps": [
    ["hot", {"kernel": "one_hot_encode", "params": {"columns": ["c"]}}],
    ["scale", {"kernel": "standard_scale", "params": {"columns": null}}]]})";
  // The scaler in front writes x as it is. The last step reads x, so it fits
  // one pass later, after the encoder is fitted.
  const std::string waiting_plan = R"({"type": "pipeline", "steps": [
    ["wait", {"kernel": "standard_scale", "params": {"columns": ["x"], "with_mean": false, "with_std": false}}],
    ["hot", {"kernel": "one_hot_encode", "params": {"columns": ["c"]}}],
    ["scale", {"kernel": "standard_scale", "params": {"columns": null}}]]})";
  fft::FittedPipeline alone = fft::Engine(2).fit(alone_plan, sources);
  fft::FittedPipeline waiting = fft::Engine(2).fit(waiting_plan, sources);

  const nlohmann::json state = nlohmann::json::parse(alone.stage_state("scale"));
  REQUIRE(state.at("columns") == nlohmann::json::array({"x"}));
  REQUIRE(nlohmann::json::parse(waiting.stage_state("scale")) == state);
  const fft::Row record{{"c", std::string("b")}, {"x", 2.0}};
  REQUIRE(waiting.transform_row(record) == alone.transform_row(record));
}

TEST_CASE("write_memmap turns transform_batch's parts into one memmap and refuses strings") {
  TempDir dir;
  fft::write_part(dir.path.string(), 0,
                  make_batch({"device", "price"}, {utf8_column({"mob", "desk"}), f64_column({1.5, 2.5})}));
  fft::write_part(dir.path.string(), 1, make_batch({"device", "price"}, {utf8_column({"mob"}), f64_column({3.5})}));
  std::vector<std::string> sources{(dir.path / "part-00000.parquet").string(),
                                   (dir.path / "part-00001.parquet").string()};
  const std::string plan = R"({"type": "pipeline", "steps": [
    ["encode", {"kernel": "label_encode", "params": {"columns": ["device"]}}],
    ["types", {"kernel": "cast", "params": {"types": {"price": "f32"}}}]]})";
  fft::FittedPipeline fitted = fft::Engine(2).fit(plan, sources);

  const std::vector<std::string> parts = fitted.transform_batch(sources, (dir.path / "parts").string());
  const std::string index_path = fft::write_memmap(parts, (dir.path / "mmap").string());
  nlohmann::json index = nlohmann::json::parse(std::ifstream(index_path));
  REQUIRE(index.at("rows") == 3);
  REQUIRE(index.at("itemsize") == 12);  // int64 device + float32 price
  REQUIRE(index.at("columns").at(0).at("dtype") == "int64");
  REQUIRE(index.at("columns").at(1).at("dtype") == "float32");
  std::ifstream file(fft::MemmapSink::data_path((dir.path / "mmap").string()), std::ios::binary);
  std::vector<char> bytes(3 * 12);
  file.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  int64_t device = 0;
  float price = 0;
  std::memcpy(&device, bytes.data() + 2 * 12, 8);  // third row: "mob" -> 2, 3.5f
  std::memcpy(&price, bytes.data() + 2 * 12 + 8, 4);
  REQUIRE(device == 2);
  REQUIRE(price == 3.5f);

  // The string column is refused without a pipeline to encode it.
  REQUIRE_THROWS_WITH(fft::write_memmap(sources, (dir.path / "raw").string()), ContainsSubstring("cast or encode"));
  REQUIRE_FALSE(std::filesystem::exists(dir.path / "raw" / "data.mmap"));
}

TEST_CASE("transform_batch refuses a source inside the output directory") {
  TempDir dir;
  fft::write_part(dir.path.string(), 0, make_batch({"device"}, {utf8_column({"mob"})}));
  const std::string plan = R"({"type": "pipeline", "steps": [
    ["select", {"kernel": "select", "params": {"columns": ["device"]}}]]})";
  std::vector<std::string> sources{(dir.path / "part-00000.parquet").string()};
  fft::FittedPipeline fitted = fft::Engine(1).fit(plan, sources);
  REQUIRE_THROWS_AS(fitted.transform_batch(sources, dir.path.string()), fft::Error);
  // Other spellings of the same directory, including a nonexistent
  // component folded away by "..", and a symlinked source.
  REQUIRE_THROWS_AS(fitted.transform_batch(sources, (dir.path / "nope" / "..").string()), fft::Error);
  REQUIRE_THROWS_AS(fitted.transform_batch(sources, dir.path.string() + "/"), fft::Error);
  std::filesystem::create_directories(dir.path / "elsewhere");
  std::filesystem::copy_file(sources[0], dir.path / "elsewhere" / "real.parquet");
  std::filesystem::create_symlink(dir.path / "elsewhere" / "real.parquet", dir.path / "part-00001.parquet");
  REQUIRE_THROWS_AS(fitted.transform_batch({(dir.path / "part-00001.parquet").string()}, dir.path.string()),
                    fft::Error);
  REQUIRE(std::filesystem::is_symlink(dir.path / "part-00001.parquet"));
  REQUIRE(std::filesystem::file_size(sources[0]) > 0);  // the input survived untouched
}
