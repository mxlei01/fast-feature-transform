// Data in and out. ParquetSource plans one task per row group (the
// parallelism unit) and decodes tasks independently; PartWriter writes one
// task's output as an ordered part file, its batches gathered into row
// groups; MemmapSink writes every task's rows at fixed offsets into one
// preallocated data.mmap.
// open_source sniffs content (PAR1 magic bytes), never just the extension.
#pragma once

#include <arrow/record_batch.h>
#include <parquet/metadata.h>
#include <parquet/properties.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "fastfeaturetransform/core.hpp"

namespace fastfeaturetransform {

struct Task {
  std::size_t file = 0;  // index into the source's file list
  int row_group = 0;
};

// What the Parquet footers show of a list column of numbers or bools: Parquet
// has no fixed-size list type, so most writers store a vector this way, and only
// the Arrow schema pyarrow adds keeps its width. The footers count values per
// row group, a null or empty list counting as one, which gives the average
// number a row holds and, when every row group holds the same number per row,
// that width (0 otherwise, and when no row group has rows).
struct ListShape {
  std::string element;  // the element's tag, "f32" for list<float>
  double average = 0;
  int64_t width = 0;
};

// A reader belongs to one worker, while its source and projection are shared.
// Keeping mutable Parquet readers here avoids cross-thread reader access.
class TaskReader {
 public:
  virtual ~TaskReader() = default;
  // Hand the task's rows to `consume` in order, at most `batch_size` at a time;
  // an empty task gives one empty batch. ParquetSource's reader decodes one
  // batch at a time instead of the whole row group first.
  virtual void read_batches(const Task& task, int64_t batch_size, const std::function<void(Batch)>& consume) = 0;
};

class RecordBatchSource {
 public:
  virtual ~RecordBatchSource() = default;
  virtual const Schema& schema() const = 0;
  virtual const std::vector<Task>& plan_tasks() const = 0;
  // Rows in each task, known before any task runs (from the Parquet footers).
  virtual const std::vector<int64_t>& task_rows() const = 0;
  // One reader for each worker. `columns` names the columns to decode
  // (nullptr = every column); a batch carries them in source order.
  virtual std::vector<std::unique_ptr<TaskReader>> readers(unsigned workers,
                                                           const std::vector<std::string>* columns) const = 0;
  // The footer shape of a list column of numbers or bools, or nullptr when
  // there is none (any other column, unusable counts, or no footers).
  virtual const ListShape* list_shape(const std::string& /*column*/) const { return nullptr; }
};

class ParquetSource final : public RecordBatchSource {
 public:
  // Parses every file's footer once up front: schema from the first file
  // (every later file must match it column for column, by name, type tag and
  // source_kind), one Task per (file, row group), and the shape of each list
  // column of numbers or bools over every file.
  explicit ParquetSource(const std::vector<std::string>& files, unsigned workers = 1);

  const Schema& schema() const override { return schema_; }
  const std::vector<Task>& plan_tasks() const override { return tasks_; }
  const std::vector<int64_t>& task_rows() const override { return task_rows_; }
  std::vector<std::unique_ptr<TaskReader>> readers(unsigned workers,
                                                   const std::vector<std::string>* columns) const override;
  const ListShape* list_shape(const std::string& column) const override;

 private:
  class Reader;
  struct File {
    std::string path;
    std::shared_ptr<parquet::FileMetaData> metadata;  // parsed once, reused by every task
    parquet::ArrowReaderProperties properties;        // dictionary columns, and INT96 read at microseconds
    std::vector<std::vector<int>> leaves;             // top-level field -> Parquet leaf columns
  };
  std::vector<File> files_;
  Schema schema_;
  // What each column held in the first file before timestamps became i64 (see source_kind in io.cpp).
  Schema sources_;
  std::vector<Task> tasks_;
  std::vector<int64_t> task_rows_;
  std::unordered_map<std::string, ListShape> list_shapes_;
};

// One ordered part per source task, written a batch at a time. The first
// batch establishes the physical Arrow schema; even an empty task must append
// one typed empty batch. Nonempty batches are gathered into row groups: a row
// group closes once it holds 4,096 rows or six batches, or once one more
// batch as large as its last would bring its buffers past 36 MiB. A batch of
// 4,096 rows or more thus closes the row group it joins, and is a row group
// of its own when no batch waits before it. A batch that would bring the
// waiting rows past 65,536 starts a row group, so no row group passes 65,536
// rows unless one batch does. Between appends the writer holds at most five
// batches, of at most 36 MiB together.
// Publication occurs only after schema/total-row validation and successful
// writer close.
// A batch reaches the Parquet writer with its row group, so a batch that
// writer refuses fails the append() that completes the row group, or
// finish(). The file is written through a buffer, so a write the disk refuses
// fails a later append() or finish(), not always the one that produced the
// bytes.
class PartWriter {
 public:
  PartWriter(const std::string& dir, std::size_t index, int64_t expected_rows);
  ~PartWriter();
  PartWriter(const PartWriter&) = delete;
  PartWriter& operator=(const PartWriter&) = delete;
  void append(const Batch& batch);
  std::string finish();
  void abandon() noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// One raw row-major file of packed records — the layout of a numpy structured
// dtype — plus index.json describing it. The file is sized up front from the
// tasks' row counts, so tasks write their disjoint slices concurrently in any
// order. Rows go to data.mmap.tmp; finish() publishes the data, then the
// completed index. These two renames are not atomic as a pair. Destruction
// and abandon() remove any owned temporaries after a failure. Field types come
// from the schema's tags: integers, f32, f64 and bool, or vectors of those
// (tags such as "f32[8]"), which arrive as fixed-size lists or as plain lists
// holding that many values in every row; strings and anything else are
// refused up front.
class MemmapSink {
 public:
  // chunk_bytes bounds the records packed and handed to one write call: a
  // single write of 2 GiB or more fails on macOS, and a whole row group's
  // records would otherwise sit in memory next to the Arrow batch.
  MemmapSink(const std::string& dir, const Schema& schema, const std::vector<int64_t>& task_rows,
             std::size_t chunk_bytes = std::size_t{64} << 20);
  ~MemmapSink() noexcept;
  MemmapSink(const MemmapSink&) = delete;
  MemmapSink& operator=(const MemmapSink&) = delete;

  // Write one task's rows at their fixed offset. Every field's column must be
  // present with exactly the declared type and no nulls.
  void write(std::size_t task, const Batch& batch) const;
  // Write a bounded slice at a task-relative row offset. The caller verifies
  // complete, nonoverlapping coverage before publication; null diagnostics
  // still report the row's absolute offset in the complete output.
  void write_slice(std::size_t task, int64_t row_offset, const Batch& batch) const;
  // Publish the data and index.json, then flush the directory, so both names
  // are on disk once it returns; returns the index path. On failure the owned
  // temporaries are removed. Failure between renames can leave the new data
  // with the old index; use a fresh directory when replacing a live pair.
  std::string finish();
  void abandon() noexcept;

  static std::string data_path(const std::string& dir) { return dir + "/data.mmap"; }
  static std::string temp_path(const std::string& dir) { return dir + "/data.mmap.tmp"; }
  static std::string index_path(const std::string& dir) { return dir + "/index.json"; }

 private:
  struct Field {
    std::string name;
    std::string dtype;   // numpy name: int8 ... uint64, float32, float64, bool
    std::string tag;     // the element's schema tag
    std::size_t size;    // elements per record (1 for scalars)
    std::size_t width;   // bytes per element
    std::size_t offset;  // byte offset within a record
    bool list;           // a vector field, even one of a single element
    arrow::Type::type arrow_type = arrow::Type::NA;
  };
  std::string dir_;
  std::vector<Field> fields_;
  std::size_t itemsize_ = 0;
  std::size_t chunk_rows_ = 1;  // records per write call
  std::vector<int64_t> task_rows_;
  std::vector<int64_t> task_offsets_;  // first row of each task
  int64_t rows_ = 0;
  struct Files;
  std::unique_ptr<Files> files_;
};

// Content-sniffing front door: every path must be a Parquet file (PAR1
// magic). Errors name the offending path and what was found instead.
std::unique_ptr<RecordBatchSource> open_source(const std::vector<std::string>& files, unsigned workers = 1);

// Write one task's batch as <dir>/part-<index>.parquet: Snappy-compressed,
// written under a temporary name and renamed into place on success. Returns
// the final path.
std::string write_part(const std::string& dir, std::size_t index, const Batch& batch);

// Flush a file descriptor to stable storage, as far as the drive honors a
// flush. On Linux that is fsync. On macOS fsync stops at the drive's own
// cache, so this uses F_FULLFSYNC, which asks the drive to write its cache out
// too. When F_FULLFSYNC fails, for any reason, this falls back to fsync, as
// SQLite does: a filesystem that refuses it (smbfs and devfs answer ENOTSUP)
// still gets the flush it can do, which on macOS stops at the drive's cache.
// Returns 0, or -1 with errno set, as fsync does.
int sync_to_disk(int fd);

// Flush a file's bytes, or a directory's entries, with sync_to_disk. A rename
// publishes whatever the disk holds, so without this a crash can leave a new
// name pointing at data that never arrived. A file its owner may write but not
// read is opened for writing instead, as coreutils' sync does. A directory that
// cannot be opened (EACCES) or flushed (EINVAL or ENOTSUP, on filesystems
// without directory sync) is skipped: its rename already happened.
void flush_to_disk(const std::string& path, bool directory);

// Arrow schemas and record batches in the engine's terms: type tags with a
// duplicate-name check, timestamps as int64 epoch microseconds, string_view
// columns as large_string, and each dictionary of strings or bytes cut to the
// entries its rows use when it holds more entries than the batch has rows
// with a value. The Parquet reader uses both; PartWriter checks its schema
// with the first.
Schema schema_from_arrow(const arrow::Schema& schema, const std::string& where);
Batch batch_from_arrow(const arrow::RecordBatch& batch);

}  // namespace fastfeaturetransform
