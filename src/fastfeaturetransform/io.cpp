#include "fastfeaturetransform/io.hpp"

#include <arrow/api.h>
#include <arrow/array/data.h>
#include <arrow/array/util.h>
#include <arrow/io/buffered.h>
#include <arrow/io/file.h>
#include <arrow/io/interfaces.h>
#include <arrow/table.h>
#include <arrow/type_traits.h>
#include <arrow/util/int_util.h>
#include <fcntl.h>
#include <parquet/arrow/reader.h>
#include <parquet/arrow/schema.h>
#include <parquet/arrow/writer.h>
#include <parquet/file_reader.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <bit>
#include <cctype>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>
#include <optional>
#include <string_view>
#include <unordered_map>
#include <utility>

#include "fastfeaturetransform/execution.hpp"
#include "fastfeaturetransform/kernels/output.hpp"

namespace fastfeaturetransform {
namespace {

std::string type_tag(const arrow::DataType& type) {
  switch (type.id()) {
    case arrow::Type::DICTIONARY:
      return type_tag(*static_cast<const arrow::DictionaryType&>(type).value_type());
    case arrow::Type::STRING:
    case arrow::Type::LARGE_STRING:
      return "utf8";
    case arrow::Type::BOOL:
      return "bool";
    case arrow::Type::INT8:
      return "i8";
    case arrow::Type::INT16:
      return "i16";
    case arrow::Type::INT32:
      return "i32";
    case arrow::Type::INT64:
      return "i64";
    case arrow::Type::UINT8:
      return "u8";
    case arrow::Type::UINT16:
      return "u16";
    case arrow::Type::UINT32:
      return "u32";
    case arrow::Type::UINT64:
      return "u64";
    case arrow::Type::HALF_FLOAT:
      return "f16";
    case arrow::Type::FLOAT:
      return "f32";
    case arrow::Type::DOUBLE:
      return "f64";
    case arrow::Type::TIMESTAMP:
      return "i64";  // read as epoch microseconds: see epoch_microseconds
    case arrow::Type::FIXED_SIZE_LIST: {
      // An embedding-style column: the element tag plus its length, "f32[8]".
      // Only whole timestamp columns are read as integers, so a list of them
      // keeps its Arrow name.
      const auto& list = static_cast<const arrow::FixedSizeListType&>(type);
      const arrow::DataType& element = *list.value_type();
      return (element.id() == arrow::Type::TIMESTAMP ? element.ToString() : type_tag(element)) + "[" +
             std::to_string(list.list_size()) + "]";
    }
    default:
      return type.ToString();
  }
}

std::shared_ptr<arrow::io::RandomAccessFile> open_file(const std::string& path) {
  auto input = arrow::io::ReadableFile::Open(path);
  if (!input.ok()) throw Error(path + ": " + input.status().ToString());
  return *input;
}

// Parquet's dictionary reader folds plain and dictionary pages into an Arrow
// dictionary but cannot decode the delta and byte-stream-split encodings, so
// a column using any of those is decoded as plain strings instead.
bool dictionary_readable(const parquet::FileMetaData& metadata, int column) {
  if (metadata.schema()->Column(column)->physical_type() != parquet::Type::BYTE_ARRAY) return false;
  try {
    for (int group = 0; group < metadata.num_row_groups(); ++group) {
      // Named so the metadata objects outlive the loop: encodings() returns a
      // reference into the chunk object, and gcc 14 does not extend
      // temporaries in a range-for initializer.
      const std::unique_ptr<parquet::RowGroupMetaData> row_group = metadata.RowGroup(group);
      const std::unique_ptr<parquet::ColumnChunkMetaData> chunk = row_group->ColumnChunk(column);
      for (parquet::Encoding::type encoding : chunk->encodings()) {
        switch (encoding) {
          case parquet::Encoding::DELTA_BYTE_ARRAY:
          case parquet::Encoding::DELTA_LENGTH_BYTE_ARRAY:
          case parquet::Encoding::DELTA_BINARY_PACKED:
          case parquet::Encoding::BYTE_STREAM_SPLIT:
            return false;
          default:
            break;
        }
      }
    }
  } catch (const parquet::ParquetException&) {
    // An encrypted column's metadata is unreadable without its key, which
    // the engine never has; the column fails only when it is decoded.
    return false;
  }
  return true;
}

void collect_leaves(const parquet::arrow::SchemaField& field, std::vector<int>& out) {
  if (field.is_leaf()) {
    out.push_back(field.column_index);
    return;
  }
  for (const auto& child : field.children) collect_leaves(child, out);
}

std::unique_ptr<parquet::arrow::FileReader> make_reader(const std::string& path,
                                                        std::unique_ptr<parquet::ParquetFileReader> raw,
                                                        const parquet::ArrowReaderProperties& properties) {
  auto reader = parquet::arrow::FileReader::Make(arrow::default_memory_pool(), std::move(raw), properties);
  if (!reader.ok()) throw Error(path + ": " + reader.status().ToString());
  return std::move(*reader);
}

std::unique_ptr<parquet::ParquetFileReader> open_raw(const std::string& path,
                                                     std::shared_ptr<parquet::FileMetaData> metadata,
                                                     bool bounded = false) {
  try {
    auto properties = parquet::default_reader_properties();
    if (bounded) {
      properties.enable_buffered_stream();
      properties.set_buffer_size(64 * 1024);
    }
    return parquet::ParquetFileReader::Open(open_file(path), properties, std::move(metadata));
  } catch (const parquet::ParquetException& e) {
    throw Error(path + ": " + e.what());
  }
}

void reject_duplicate_columns(const std::string& path, const Schema& schema) {
  std::vector<std::string> names;
  for (const auto& [name, type] : schema.cols) names.push_back(name);
  std::sort(names.begin(), names.end());
  auto dup = std::adjacent_find(names.begin(), names.end());
  if (dup != names.end()) throw Error(path + ": duplicate column name '" + *dup + "'");
}

// A mismatched file would silently poison every fitted vocabulary, so the
// error names exactly what differs.
void require_same_schema(const std::string& path, const std::string& first, const Schema& got, const Schema& expected) {
  if (got.cols == expected.cols) return;
  for (const auto& [name, type] : expected.cols) {
    const std::string* actual = got.find(name);
    if (actual == nullptr) throw Error(path + ": column '" + name + "' is missing (present in " + first + ")");
    if (*actual != type) throw Error(path + ": column '" + name + "' is " + *actual + " but " + first + " has " + type);
  }
  for (const auto& [name, type] : got.cols) {
    if (expected.find(name) == nullptr) throw Error(path + ": column '" + name + "' is not present in " + first);
  }
  for (std::size_t i = 0; i < got.cols.size(); ++i) {
    if (got.cols[i].first != expected.cols[i].first) {
      throw Error(path + ": column " + std::to_string(i) + " is '" + got.cols[i].first + "' but " + first + " has '" +
                  expected.cols[i].first + "' (columns must be in the same order)");
    }
  }
  throw Error(path + ": schema differs from " + first);
}

void require_parquet(const std::string& path) {
  std::error_code ec;
  std::filesystem::file_status status = std::filesystem::status(path, ec);
  if (ec || !std::filesystem::exists(status)) throw Error(path + ": no such file");
  if (std::filesystem::is_directory(status)) {
    throw Error(path + ": is a directory (pass Parquet files, not a dataset directory)");
  }
  std::ifstream file(path, std::ios::binary);
  if (!file) throw Error(path + ": cannot open file");
  char magic[4] = {};
  file.read(magic, 4);
  if (file.gcount() == 0) throw Error(path + ": file is empty");
  // A file whose footer is encrypted ends, as it starts, with PARE: a file
  // that only starts with it, such as a CSV headed PARENT_ID, is not Parquet.
  char tail[4] = {};
  const auto size = std::filesystem::file_size(path, ec);
  if (file.gcount() == 4 && std::string_view(magic, 4) == "PARE" && !ec && size >= 12 &&
      file.seekg(-4, std::ios::end).read(tail, 4) && std::string_view(tail, 4) == "PARE") {
    throw Error(path + ": is a Parquet file with an encrypted footer; encrypted Parquet is not supported");
  }
  if (file.gcount() != 4 || std::string_view(magic, 4) != "PAR1") {
    throw Error(path + (path.ends_with(".parquet") ? ": has a .parquet extension but is not a Parquet file"
                                                   : ": unrecognized format (expected Parquet)"));
  }
}

// Arrow's Parquet reader decodes dictionary indices without checking them
// against the dictionary, and the kernels index their caches and the
// dictionary with them unchecked, so a corrupt file would be read outside
// both. One pass per batch here covers every kernel. The reader's indices
// are int32, and their largest value read as unsigned (a negative one is
// then too large) settles almost every batch in a tenth of the time Arrow's
// check takes. Null slots count in that maximum, and Parquet leaves zeros
// there; when the maximum is out of range, Arrow's check, which skips null
// slots, decides.
void check_dictionary_indices(const arrow::Array& column, const std::string& where, const std::string& name) {
  if (column.type_id() != arrow::Type::DICTIONARY) return;
  const auto& dictionary = static_cast<const arrow::DictionaryArray&>(column);
  const arrow::ArrayData& indices = *dictionary.indices()->data();
  const auto entries = static_cast<uint64_t>(dictionary.dictionary()->length());
  if (indices.type->id() == arrow::Type::INT32) {
    const int32_t* values = indices.GetValues<int32_t>(1);
    uint32_t largest = 0;
    for (int64_t row = 0; row < indices.length; ++row) largest = std::max(largest, static_cast<uint32_t>(values[row]));
    if (largest < entries) return;
  }
  const arrow::Status status = arrow::internal::CheckIndexBounds(arrow::ArraySpan(indices), entries);
  if (!status.ok()) {
    throw Error(where + ": column '" + name + "' has a dictionary index outside its dictionary: " + status.message());
  }
}

// Every timestamp column enters the engine as int64 microseconds since
// 1970-01-01: the count Arrow stores (the UTC instant for a column with a
// time zone, the wall clock for one without) rescaled from the column's unit,
// which is also what a record's datetime becomes. Nanoseconds floor, as numpy
// and Python do; seconds and milliseconds outside the int64 microsecond range
// are refused rather than wrapped.
int64_t to_microseconds(int64_t value, const arrow::TimestampType& type, const std::string& name) {
  switch (type.unit()) {
    case arrow::TimeUnit::MICRO:
      return value;
    case arrow::TimeUnit::NANO:
      return value / 1'000 - (value % 1'000 < 0 ? 1 : 0);
    default: {
      const int64_t scale = type.unit() == arrow::TimeUnit::SECOND ? 1'000'000 : 1'000;
      if (value > std::numeric_limits<int64_t>::max() / scale || value < std::numeric_limits<int64_t>::min() / scale) {
        throw Error("column '" + name + "' holds " + type.ToString() + " value " + std::to_string(value) +
                    ", outside the range of int64 microseconds");
      }
      return value * scale;
    }
  }
}

// A timestamp column as int64 microseconds. Microseconds are the same buffers
// viewed as int64. A dictionary of timestamps decodes to a plain int64
// column, rescaling only the entries its rows use. Parquet's reader gives
// dictionaries only of strings and bytes (see dictionary_readable), so only
// the tests' in-memory arrays reach that branch.
std::shared_ptr<arrow::Array> epoch_microseconds(const std::shared_ptr<arrow::Array>& column, const std::string& name) {
  const bool dictionary = column->type_id() == arrow::Type::DICTIONARY;
  const arrow::Array& values = dictionary ? *static_cast<const arrow::DictionaryArray&>(*column).dictionary() : *column;
  if (values.type_id() != arrow::Type::TIMESTAMP) return column;
  const auto& stamps = static_cast<const arrow::TimestampArray&>(values);
  const auto& type = static_cast<const arrow::TimestampType&>(*values.type());
  if (!dictionary && type.unit() == arrow::TimeUnit::MICRO) {
    std::shared_ptr<arrow::ArrayData> data = column->data()->Copy();
    data->type = arrow::int64();
    return arrow::MakeArray(data);
  }
  PrimitiveOutput<arrow::Int64Type> out(column->length(), "column '" + name + "'");
  for (int64_t row = 0; row < column->length(); ++row) {
    if (column->IsNull(row)) {
      out.set_null(row);
      continue;
    }
    const int64_t entry = dictionary ? static_cast<const arrow::DictionaryArray&>(*column).GetValueIndex(row) : row;
    if (dictionary && stamps.IsNull(entry)) {
      out.set_null(row);
    } else {
      out.mutable_values()[row] = to_microseconds(stamps.Value(entry), type, name);
    }
  }
  return out.finish();
}

// What a column holds in its file, for the cross-file check. The engine reads
// every timestamp as int64 microseconds, but epoch integers, wall-clock
// timestamps and UTC instants do not mean the same numbers, so files must
// agree on which a column holds; the unit may differ, since every unit is
// rescaled to microseconds.
std::string source_kind(const arrow::DataType& type) {
  if (type.id() == arrow::Type::DICTIONARY) {
    return source_kind(*static_cast<const arrow::DictionaryType&>(type).value_type());
  }
  if (type.id() == arrow::Type::TIMESTAMP) {
    return static_cast<const arrow::TimestampType&>(type).timezone().empty() ? "a timestamp without a time zone"
                                                                             : "a timestamp with a time zone";
  }
  return type_tag(type);
}

// A list column's element tag, named as a fixed-size list's would be ("f32"
// for list<float>), when it holds numbers or bools, as a vector does; empty
// for any other column. Any other list, such as one of strings read as a
// dictionary, keeps the estimate of one variable-length value: sized by its
// length, it would be cut into small batches that each write its whole
// dictionary again.
std::string plain_list_element(const arrow::DataType& type) {
  if (type.id() != arrow::Type::LIST && type.id() != arrow::Type::LARGE_LIST) return "";
  const arrow::DataType& element = *static_cast<const arrow::BaseListType&>(type).value_type();
  const bool plain =
      arrow::is_integer(element.id()) || arrow::is_floating(element.id()) || element.id() == arrow::Type::BOOL;
  return plain ? type_tag(element) : "";
}

}  // namespace

Schema schema_from_arrow(const arrow::Schema& schema, const std::string& where) {
  Schema tagged;
  for (const auto& field : schema.fields()) tagged.cols.emplace_back(field->name(), type_tag(*field->type()));
  reject_duplicate_columns(where, tagged);
  return tagged;
}

Batch batch_from_arrow(const arrow::RecordBatch& input) {
  Batch batch;
  batch.rows = input.num_rows();
  for (int i = 0; i < input.num_columns(); ++i) {
    batch.names.push_back(input.column_name(i));
    batch.cols.push_back(epoch_microseconds(input.column(i), input.column_name(i)));
  }
  return batch;
}

void TaskReader::read_batches(const Task& task, int64_t batch_size, const std::function<void(Batch)>& consume) {
  if (batch_size <= 0) throw Error("batch_size must be a positive integer");
  Batch input = read(task);
  if (input.rows == 0) {
    consume(std::move(input));
    return;
  }
  for (int64_t offset = 0; offset < input.rows; offset += batch_size) {
    Batch part;
    part.rows = std::min(batch_size, input.rows - offset);
    part.names = input.names;
    for (const auto& column : input.cols) part.cols.push_back(column->Slice(offset, part.rows));
    consume(std::move(part));
  }
}

std::vector<std::unique_ptr<TaskReader>> RecordBatchSource::readers(unsigned workers,
                                                                    const std::vector<std::string>* columns) const {
  class Reader final : public TaskReader {
   public:
    Reader(const RecordBatchSource& source, const std::vector<std::string>* columns) : source_(source) {
      if (columns != nullptr) columns_ = *columns;
    }
    Batch read(const Task& task) override { return source_.read_task(task, columns_ ? &*columns_ : nullptr); }

   private:
    const RecordBatchSource& source_;
    std::optional<std::vector<std::string>> columns_;
  };
  std::vector<std::unique_ptr<TaskReader>> out;
  for (unsigned worker = 0; worker < workers; ++worker) out.push_back(std::make_unique<Reader>(*this, columns));
  return out;
}

ParquetSource::ParquetSource(const std::vector<std::string>& files, unsigned workers) {
  if (files.empty()) throw Error("no input files given");
  struct Prepared {
    File file;
    Schema schema;
    Schema sources;                          // source_kind per column, for the cross-file check
    std::vector<std::string> list_elements;  // plain_list_element per column
    std::exception_ptr error;
  };
  auto prepare = [&](std::size_t index) -> Prepared {
    const std::string& path = files[index];
    File file;
    file.path = path;
    std::unique_ptr<parquet::ParquetFileReader> raw = open_raw(path, nullptr);
    file.metadata = raw->metadata();
    for (int column = 0; column < file.metadata->num_columns(); ++column) {
      // Read string columns as the dictionaries Parquet already stores:
      // kernels then hash each dictionary entry once instead of every row.
      if (dictionary_readable(*file.metadata, column)) file.properties.set_read_dictionary(column, true);
    }
    // Legacy INT96 timestamps decode as nanoseconds by default, which cannot
    // hold dates outside 1677-2262; the engine reads microseconds anyway.
    file.properties.set_coerce_int96_timestamp_unit(arrow::TimeUnit::MICRO);
    std::unique_ptr<parquet::arrow::FileReader> reader = make_reader(path, std::move(raw), file.properties);

    std::shared_ptr<arrow::Schema> schema;
    arrow::Status status = reader->GetSchema(&schema);
    if (!status.ok()) throw Error(path + ": failed to read schema: " + status.ToString());
    Schema tagged = schema_from_arrow(*schema, path);
    Schema sources;
    std::vector<std::string> list_elements;
    for (const auto& field : schema->fields()) {
      sources.cols.emplace_back(field->name(), source_kind(*field->type()));
      list_elements.push_back(plain_list_element(*field->type()));
    }
    for (const auto& field : reader->manifest().schema_fields) {
      std::vector<int> leaves;
      collect_leaves(field, leaves);
      file.leaves.push_back(std::move(leaves));
    }

    return {std::move(file), std::move(tagged), std::move(sources), std::move(list_elements), {}};
  };
  std::vector<std::string> list_elements;  // the first file's, which every later file matches
  auto append = [&](Prepared prepared) {
    if (prepared.error) std::rethrow_exception(prepared.error);
    File& file = prepared.file;
    if (files_.empty()) {
      schema_ = std::move(prepared.schema);
      sources_ = std::move(prepared.sources);
      list_elements = std::move(prepared.list_elements);
    } else {
      require_same_schema(file.path, files_.front().path, prepared.schema, schema_);
      require_same_schema(file.path, files_.front().path, prepared.sources, sources_);
    }
    for (int group = 0; group < file.metadata->num_row_groups(); ++group) {
      tasks_.push_back(Task{files_.size(), group});
      task_rows_.push_back(file.metadata->RowGroup(group)->num_rows());
    }
    files_.push_back(std::move(file));
  };
  // A handful of files costs less to inspect inline than to start workers.
  // Larger inputs use at most eight open readers, within the caller's budget.
  const unsigned preparation_workers = files.size() < 16 ? 1 : std::max(1u, std::min(8u, workers));
  files_.reserve(files.size());
  if (preparation_workers == 1) {
    for (std::size_t index = 0; index < files.size(); ++index) append(prepare(index));
  } else {
    std::vector<Prepared> prepared(files.size());
    parallel_for(preparation_workers, files.size(), [&](unsigned, std::size_t index) {
      try {
        prepared[index] = prepare(index);
      } catch (...) {
        // Ordered assembly below chooses the same failing file as serial
        // preparation, even if a later file fails first on another worker.
        prepared[index].error = std::current_exception();
      }
    });
    for (Prepared& file : prepared) append(std::move(file));
  }
  if (tasks_.empty()) throw Error("input files contain no row groups");

  for (std::size_t field = 0; field < list_elements.size(); ++field) {
    if (list_elements[field].empty()) continue;
    int64_t values = 0, rows = 0, width = 0;
    bool uniform = true, sane = true;
    try {
      for (const File& file : files_) {
        for (int group = 0; sane && group < file.metadata->num_row_groups(); ++group) {
          const std::unique_ptr<parquet::RowGroupMetaData> row_group = file.metadata->RowGroup(group);
          const int64_t group_rows = row_group->num_rows();
          if (group_rows == 0) continue;
          // A list of plain values has a single leaf, holding at least one
          // value per row; counts no writer produces leave the column unsized.
          const int64_t group_values = row_group->ColumnChunk(file.leaves[field].front())->num_values();
          sane = group_rows > 0 && group_values >= group_rows &&
                 values <= std::numeric_limits<int64_t>::max() - group_values;
          if (!sane) break;
          if (rows == 0) width = group_values / group_rows;
          uniform = uniform && group_values % group_rows == 0 && group_values / group_rows == width;
          values += group_values;
          rows += group_rows;
        }
      }
    } catch (const parquet::ParquetException&) {
      // A column encrypted with a key the engine lacks shows no counts; it
      // fails only when it is decoded.
      continue;
    }
    if (!sane) continue;
    list_shapes_.emplace(
        schema_.cols[field].first,
        ListShape{list_elements[field], rows > 0 ? static_cast<double>(values) / static_cast<double>(rows) : 0.0,
                  uniform && rows > 0 ? width : 0});
  }
}

const ListShape* ParquetSource::list_shape(const std::string& column) const {
  const auto found = list_shapes_.find(column);
  return found == list_shapes_.end() ? nullptr : &found->second;
}

class ParquetSource::Reader final : public TaskReader {
 public:
  using Projection = std::vector<std::vector<int>>;
  Reader(const ParquetSource& source, std::shared_ptr<const Projection> projection)
      : source_(source), projection_(std::move(projection)) {}

  Batch read(const Task& task) override {
    const File& file = source_.files_[task.file];
    // One reader/file descriptor per worker, not one per input file. Dynamic
    // task assignment retains reuse for adjacent groups without unbounded FDs.
    if (!reader_ || file_ != task.file || bounded_) {
      reader_ = make_reader(file.path, open_raw(file.path, file.metadata), file.properties);
      file_ = task.file;
      bounded_ = false;
    }
    auto read = projection_ ? reader_->ReadRowGroup(task.row_group, (*projection_)[task.file])
                            : reader_->ReadRowGroup(task.row_group);
    if (!read.ok()) {
      throw Error(file.path + ": failed to read row group " + std::to_string(task.row_group) + ": " +
                  read.status().ToString());
    }
    std::shared_ptr<arrow::Table> table = *read;
    if (std::any_of(table->columns().begin(), table->columns().end(),
                    [](const auto& column) { return column->num_chunks() > 1; })) {
      auto combined = table->CombineChunks(arrow::default_memory_pool());
      if (!combined.ok()) throw Error(file.path + ": " + combined.status().ToString());
      table = *combined;
    }

    Batch batch;
    batch.rows = table->num_rows();
    for (int i = 0; i < table->num_columns(); ++i) {
      const auto& chunks = table->column(i);
      std::shared_ptr<arrow::Array> column;
      if (chunks->num_chunks() == 1) {
        column = chunks->chunk(0);
      } else if (chunks->num_chunks() == 0) {
        // A 0-row row group (empty partition) decodes to no chunks at all.
        auto empty = arrow::MakeEmptyArray(chunks->type());
        if (!empty.ok()) throw Error(file.path + ": " + empty.status().ToString());
        column = *empty;
      } else {
        throw Error(file.path + ": column '" + table->field(i)->name() + "' has " +
                    std::to_string(chunks->num_chunks()) + " chunks after CombineChunks");
      }
      check_dictionary_indices(*column, file.path + ": failed to read row group " + std::to_string(task.row_group),
                               table->field(i)->name());
      batch.names.push_back(table->field(i)->name());
      batch.cols.push_back(epoch_microseconds(column, table->field(i)->name()));
    }
    return batch;
  }

  void read_batches(const Task& task, int64_t batch_size, const std::function<void(Batch)>& consume) override {
    if (batch_size <= 0) throw Error("batch_size must be a positive integer");
    const File& file = source_.files_[task.file];
    if (!reader_ || file_ != task.file || !bounded_) {
      auto properties = file.properties;
      // Otherwise Arrow may prefetch the entire compressed row group, even
      // though decoded record batches are small.
      properties.set_pre_buffer(false);
      reader_ = make_reader(file.path, open_raw(file.path, file.metadata, true), properties);
      file_ = task.file;
      bounded_ = true;
    }
    reader_->set_batch_size(batch_size);
    auto stream = projection_ ? reader_->GetRecordBatchReader({task.row_group}, (*projection_)[task.file])
                              : reader_->GetRecordBatchReader({task.row_group});
    const std::string where = file.path + ": failed to read row group " + std::to_string(task.row_group);
    if (!stream.ok()) throw Error(where + ": " + stream.status().ToString());
    bool emitted = false;
    for (;;) {
      std::shared_ptr<arrow::RecordBatch> batch;
      const auto status = (*stream)->ReadNext(&batch);
      if (!status.ok()) throw Error(where + ": " + status.ToString());
      if (!batch) break;
      for (int i = 0; i < batch->num_columns(); ++i) {
        check_dictionary_indices(*batch->column(i), where, batch->schema()->field(i)->name());
      }
      emitted = true;
      consume(batch_from_arrow(*batch));
    }
    if (!emitted) {
      Batch empty;
      for (const auto& field : (*stream)->schema()->fields()) {
        auto column = arrow::MakeEmptyArray(field->type());
        if (!column.ok()) throw Error(where + ": " + column.status().ToString());
        empty.names.push_back(field->name());
        empty.cols.push_back(epoch_microseconds(*column, field->name()));
      }
      consume(std::move(empty));
    }
  }

 private:
  const ParquetSource& source_;
  std::shared_ptr<const Projection> projection_;
  std::size_t file_ = 0;
  std::unique_ptr<parquet::arrow::FileReader> reader_;
  bool bounded_ = false;
};

std::vector<std::unique_ptr<TaskReader>> ParquetSource::readers(unsigned workers,
                                                                const std::vector<std::string>* columns) const {
  std::shared_ptr<Reader::Projection> projection;
  if (columns != nullptr) {
    projection = std::make_shared<Reader::Projection>();
    std::unordered_map<std::string, std::size_t> positions;
    for (std::size_t field = 0; field < schema_.cols.size(); ++field)
      positions.emplace(schema_.cols[field].first, field);
    for (const File& file : files_) {
      std::vector<int> leaves;
      for (const std::string& name : *columns) {
        const auto found = positions.find(name);
        if (found == positions.end()) throw Error(file.path + ": column '" + name + "' not found");
        const auto& selected = file.leaves[found->second];
        leaves.insert(leaves.end(), selected.begin(), selected.end());
      }
      std::sort(leaves.begin(), leaves.end());
      leaves.erase(std::unique(leaves.begin(), leaves.end()), leaves.end());
      projection->push_back(std::move(leaves));
    }
  }
  std::vector<std::unique_ptr<TaskReader>> out;
  for (unsigned worker = 0; worker < workers; ++worker) out.push_back(std::make_unique<Reader>(*this, projection));
  return out;
}

Batch ParquetSource::read_task(const Task& task, const std::vector<std::string>* columns) const {
  return readers(1, columns).front()->read(task);
}

std::unique_ptr<RecordBatchSource> open_source(const std::vector<std::string>& files, unsigned workers) {
  for (const std::string& path : files) require_parquet(path);
  return std::make_unique<ParquetSource>(files, workers);
}

namespace {

// The Parquet writer hands its sink every page header and every page body in
// a write of its own, and asks for the position before each page. The buffer
// gathers them into write calls of up to 64 KiB; a single write of 64 KiB or
// more still goes straight to the file. A write the disk refuses is
// therefore reported up to 64 KiB of output later, and for a part smaller
// than the buffer only when finish() closes the file.
constexpr int64_t kPartBufferBytes = 64 * 1024;

// The dictionary page limit of a part in which every column passes
// suits_low_dictionary_limit. Arrow's writer hashes each column of plain
// values into a dictionary and gives the dictionary up for plain pages only
// when its page reaches a limit, 1 MiB by default. A row group here holds at
// most 65,536 rows, so a column of distinct 8-byte values (an id, a
// timestamp) never reaches 1 MiB: it is written as a dictionary of every
// value plus an index per row, which is larger than the values. The writer
// checks the limit after each batch of at most 1,024 values, so under
// 248 KiB a dictionary of 8-byte values ends with at most 32,767 entries,
// one fewer than the 32,768 at which the encoder's hash table grows fourfold.
//
// Measured against the 1 MiB limit on an Apple M4, with a transform that
// passes one i64 column through on one thread, in row groups of 65,536 rows:
// distinct values take 0.57 to 0.74 of the time and 0.83 to 0.90 of the
// bytes (at 256 KiB, 0.77 to 0.88 of the time), and a row group of at most
// 31,743 distinct values is written byte for byte as under 1 MiB. The cost
// is a row group of 31,744 to about 55,000 distinct values: its rows after
// the limit are written whole, though the dictionary would have held them.
// It is up to 13% larger with the values in random order (38,000 distinct),
// and up to 54% larger when the distinct values come first and the rest
// repeat them (32,000 distinct).
constexpr int64_t kLowDictionaryLimit = 248 * 1024;

// Whether the lower limit suits a column of this type. An integer, a date, a
// time, a timestamp and a duration are stored as 4 or 8 bytes; a float and a
// bool are written without a dictionary; a dictionary of strings or bytes is
// written as the dictionary it already is, whatever the limit. A plain
// string, a decimal and a list keep Arrow's limit: with 248 KiB forced on
// them, 36-character strings of 8,000 distinct values take 3.8 times the
// bytes, decimals of 20,000 distinct values 1.6 times and lists of integers
// drawn from 100,000 values 1.4 times (row groups of 65,536 rows).
bool suits_low_dictionary_limit(const arrow::DataType& type) {
  const arrow::Type::type id = type.id();
  if (id == arrow::Type::DICTIONARY) {
    return arrow::is_base_binary_like(static_cast<const arrow::DictionaryType&>(type).value_type()->id());
  }
  return arrow::is_integer(id) || arrow::is_floating(id) || id == arrow::Type::BOOL || id == arrow::Type::TIMESTAMP ||
         id == arrow::Type::DATE32 || id == arrow::Type::DATE64 || id == arrow::Type::TIME32 ||
         id == arrow::Type::TIME64 || id == arrow::Type::DURATION;
}

}  // namespace

struct PartWriter::Impl {
  std::string path, temp;
  int64_t expected_rows = 0, rows = 0;
  std::shared_ptr<arrow::Schema> schema;
  std::shared_ptr<arrow::io::OutputStream> output;  // the part file behind its write buffer
  std::unique_ptr<parquet::arrow::FileWriter> writer;
  bool finished = false, abandoned = false, owns_temp = false;
};

PartWriter::PartWriter(const std::string& dir, std::size_t index, int64_t expected_rows)
    : impl_(std::make_unique<Impl>()) {
  if (expected_rows < 0) throw Error(dir + ": expected row count must not be negative");
  char name[32];
  std::snprintf(name, sizeof(name), "part-%05zu.parquet", index);
  impl_->path = dir + "/" + name;
  impl_->temp = impl_->path + ".tmp";
  impl_->expected_rows = expected_rows;
}

PartWriter::~PartWriter() { abandon(); }

void PartWriter::append(const Batch& batch) {
  auto& state = *impl_;
  if (state.finished || state.abandoned) throw Error(state.path + ": part writer is no longer active");
  try {
    if (batch.rows < 0 || batch.rows > state.expected_rows - state.rows) {
      throw Error(state.path + ": batch row count exceeds expected total " + std::to_string(state.expected_rows));
    }
    if (batch.names.empty() && batch.rows > 0) {
      throw Error(state.path + ": cannot write " + std::to_string(batch.rows) + " rows with no columns");
    }
    if (batch.names.size() != batch.cols.size()) throw Error(state.path + ": batch column names and arrays differ");
    arrow::FieldVector fields;
    for (std::size_t i = 0; i < batch.names.size(); ++i) {
      const auto& column = batch.cols[i];
      if (!column || column->length() != batch.rows) {
        throw Error(state.path + ": column '" + batch.names[i] + "' length differs from batch row count");
      }
      fields.push_back(arrow::field(batch.names[i], column->type()));
    }
    auto schema = arrow::schema(std::move(fields));
    auto table = arrow::Table::Make(schema, batch.cols, batch.rows);
    auto status = table->Validate();
    if (!status.ok()) throw Error(state.path + ": " + status.ToString());
    if (state.schema) {
      if (!state.schema->Equals(*schema, false)) throw Error(state.path + ": batch schema differs from first batch");
    } else {
      (void)schema_from_arrow(*schema, state.path);
      state.schema = std::move(schema);
      parquet::WriterProperties::Builder properties;
      properties.compression(parquet::Compression::SNAPPY);
      // The limit is one setting for the whole file, so it is lowered only
      // when that suits every column.
      const arrow::FieldVector& columns = state.schema->fields();
      if (std::all_of(columns.begin(), columns.end(), [](const std::shared_ptr<arrow::Field>& column) {
            return suits_low_dictionary_limit(*column->type());
          })) {
        properties.dictionary_pagesize_limit(kLowDictionaryLimit);
      }
      // Floats rarely repeat, so a dictionary of them costs time and space.
      // Encoding settings name Parquet leaves, and the floats of a list or
      // struct column sit below its name ("emb.list.element"), so the leaves
      // come from the Parquet schema the writer will produce.
      std::shared_ptr<parquet::SchemaDescriptor> leaves;
      status = parquet::arrow::ToParquetSchema(state.schema.get(), *parquet::default_writer_properties(), &leaves);
      if (!status.ok()) throw Error(state.path + ": " + status.ToString());
      for (int i = 0; i < leaves->num_columns(); ++i) {
        const parquet::ColumnDescriptor& leaf = *leaves->Column(i);
        const auto& logical = leaf.logical_type();
        if (leaf.physical_type() == parquet::Type::FLOAT || leaf.physical_type() == parquet::Type::DOUBLE ||
            (logical && logical->is_float16())) {
          properties.disable_dictionary(leaf.path());
        }
      }
      auto file = arrow::io::FileOutputStream::Open(state.temp);
      if (!file.ok()) throw Error(state.temp + ": " + file.status().ToString());
      state.output = *file;
      state.owns_temp = true;
      auto buffered = arrow::io::BufferedOutputStream::Create(kPartBufferBytes, arrow::default_memory_pool(), *file);
      if (!buffered.ok()) throw Error(state.temp + ": " + buffered.status().ToString());
      state.output = *buffered;
      auto writer = parquet::arrow::FileWriter::Open(*state.schema, arrow::default_memory_pool(), state.output,
                                                     properties.build());
      if (!writer.ok()) throw Error(state.path + ": " + writer.status().ToString());
      state.writer = std::move(*writer);
    }
    if (batch.rows > 0) {
      // WriteTable flushes a complete row group, unlike WriteRecordBatch's
      // buffered-row-group mode. Only this callback's batch stays resident.
      status = state.writer->WriteTable(*table, batch.rows);
      if (!status.ok()) throw Error(state.path + ": " + status.ToString());
    }
    state.rows += batch.rows;
  } catch (...) {
    abandon();
    throw;
  }
}

std::string PartWriter::finish() {
  auto& state = *impl_;
  if (state.finished) return state.path;
  if (state.abandoned) throw Error(state.path + ": part writer is no longer active");
  try {
    std::string path = state.path;  // allocate the return value before publication
    if (!state.writer) throw Error(state.path + ": no batches supplied to establish the output schema");
    if (state.rows != state.expected_rows) {
      throw Error(state.path + ": produced " + std::to_string(state.rows) + " rows, expected " +
                  std::to_string(state.expected_rows));
    }
    if (state.rows == 0) {
      arrow::ArrayVector arrays;
      for (const auto& field : state.schema->fields()) {
        auto empty = arrow::MakeEmptyArray(field->type());
        if (!empty.ok()) throw Error(state.path + ": " + empty.status().ToString());
        arrays.push_back(*empty);
      }
      const auto table = arrow::Table::Make(state.schema, arrays, 0);
      const auto status = state.writer->WriteTable(*table, 1);
      if (!status.ok()) throw Error(state.path + ": " + status.ToString());
    }
    auto status = state.writer->Close();
    if (!status.ok()) throw Error(state.path + ": " + status.ToString());
    state.writer.reset();
    // Closing writes the bytes still in the buffer, then closes the file. A
    // part smaller than the buffer reaches the disk only here, so a write
    // the disk refuses fails this call and the part is not published.
    status = state.output->Close();
    if (!status.ok()) throw Error(state.path + ": " + status.ToString());
    state.output.reset();
    std::error_code error;
    std::filesystem::rename(state.temp, state.path, error);
    if (error) throw Error(state.path + ": " + error.message());
    state.owns_temp = false;
    state.finished = true;
    return path;
  } catch (...) {
    abandon();
    throw;
  }
}

void PartWriter::abandon() noexcept {
  auto& state = *impl_;
  if (state.finished || state.abandoned) return;
  state.abandoned = true;
  if (state.writer) {
    try {
      (void)state.writer->Close();
    } catch (...) {
    }
    state.writer.reset();
  }
  if (state.output) {
    // The file is removed below, so a buffer that cannot be written is not
    // an error here. Closing explicitly matters: Arrow's stream closes
    // itself when destroyed, and a debug build of Arrow aborts the process
    // if that close fails.
    try {
      (void)state.output->Close();
    } catch (...) {
    }
    state.output.reset();
  }
  if (state.owns_temp) {
    // The path already exists as a string: cleanup must not allocate while
    // another exception (including allocation failure) is unwinding.
    ::unlink(state.temp.c_str());
    state.owns_temp = false;
  }
}

std::string write_part(const std::string& dir, std::size_t index, const Batch& batch) {
  PartWriter writer(dir, index, batch.rows);
  writer.append(batch);
  return writer.finish();
}

int sync_to_disk(int fd) {
#ifdef F_FULLFSYNC
  if (::fcntl(fd, F_FULLFSYNC) == 0) return 0;
#endif
  return ::fsync(fd);
}

void flush_to_disk(const std::string& path, bool directory) {
  int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | (directory ? O_DIRECTORY : 0));
  if (fd < 0 && errno == EACCES && !directory) fd = ::open(path.c_str(), O_WRONLY | O_CLOEXEC);
  if (fd < 0) {
    if (directory && errno == EACCES) return;
    throw Error(path + ": cannot open to flush: " + std::strerror(errno));
  }
  const int result = sync_to_disk(fd);
  const int error = errno;
  ::close(fd);
  if (result != 0 && !(directory && (error == EINVAL || error == ENOTSUP))) {
    throw Error(path + ": flush failed: " + std::strerror(error));
  }
}

// ---------------------------------------------------------------------------
// MemmapSink
// ---------------------------------------------------------------------------
namespace {

static_assert(std::endian::native == std::endian::little, "the memmap layout is little-endian");

struct ElementType {
  const char* dtype;
  std::size_t width;
  arrow::Type::type arrow_type;
};

// The schema tags a memmap can hold, with numpy's names for them.
std::optional<ElementType> element_type(const std::string& tag) {
  static const std::unordered_map<std::string, ElementType> kElements{
      {"i8", {"int8", 1, arrow::Type::INT8}},      {"i16", {"int16", 2, arrow::Type::INT16}},
      {"i32", {"int32", 4, arrow::Type::INT32}},   {"i64", {"int64", 8, arrow::Type::INT64}},
      {"u8", {"uint8", 1, arrow::Type::UINT8}},    {"u16", {"uint16", 2, arrow::Type::UINT16}},
      {"u32", {"uint32", 4, arrow::Type::UINT32}}, {"u64", {"uint64", 8, arrow::Type::UINT64}},
      {"f32", {"float32", 4, arrow::Type::FLOAT}}, {"f64", {"float64", 8, arrow::Type::DOUBLE}},
      {"bool", {"bool", 1, arrow::Type::BOOL}},
  };
  auto found = kElements.find(tag);
  if (found == kElements.end()) return std::nullopt;
  return found->second;
}

std::string memmap_error(const std::string& name) { return "memmap: column '" + name + "'"; }

// Copy rows [first, first + count) of a field's element values into the
// record buffer: element k of a row lands at row * stride + offset + k *
// width. `values` holds `size` elements per row (a vector's child window, or
// the column itself for a scalar field) and must be null-free.
void scatter(const arrow::Array& values, const ElementType& element, std::size_t size, std::size_t first,
             std::size_t count, uint8_t* buffer, std::size_t stride, std::size_t offset) {
  if (element.arrow_type == arrow::Type::BOOL) {
    const auto& bools = static_cast<const arrow::BooleanArray&>(values);
    for (std::size_t row = 0; row < count; ++row) {
      for (std::size_t k = 0; k < size; ++k) {
        buffer[row * stride + offset + k] = bools.Value(static_cast<int64_t>((first + row) * size + k)) ? 1 : 0;
      }
    }
    return;
  }
  const std::size_t bytes = size * element.width;  // one row's slice of the field
  const uint8_t* data =
      values.data()->buffers[1]->data() + (static_cast<std::size_t>(values.offset()) + first * size) * element.width;
  for (std::size_t row = 0; row < count; ++row) {
    std::memcpy(buffer + row * stride + offset, data + row * bytes, bytes);
  }
}

// The first row holding a null, given `size` elements per row.
int64_t first_null_row(const arrow::Array& values, std::size_t size) {
  for (int64_t i = 0; i < values.length(); ++i) {
    if (values.IsNull(i)) return i / static_cast<int64_t>(size);
  }
  return -1;
}

// A plain list's values as a vector field's: the window of its child that
// this batch's rows use, once every row holds exactly `size` values. Rows are
// reported at their offset in the complete output.
template <typename List>
std::shared_ptr<arrow::Array> uniform_window(const List& list, std::size_t size, int64_t global_offset,
                                             const std::string& where) {
  for (int64_t row = 0; row < list.length(); ++row) {
    if (list.value_length(row) == 0) {
      throw Error(where + ": row " + std::to_string(global_offset + row) +
                  " holds an empty list, and a vector field cannot hold an empty row");
    }
    if (static_cast<std::size_t>(list.value_length(row)) != size) {
      throw Error(where + ": row " + std::to_string(global_offset + row) + " holds a list of " +
                  std::to_string(list.value_length(row)) + ", and a vector field needs " + std::to_string(size) +
                  " values in every row");
    }
  }
  if (list.length() == 0) return list.values()->Slice(0, 0);
  return list.values()->Slice(list.value_offset(0), list.length() * static_cast<int64_t>(size));
}

}  // namespace

// Construct every path before acquiring a descriptor. This member owns cleanup
// even when the enclosing MemmapSink constructor throws; its destructor does not
// allocate, and it never removes a temporary it has already published.
struct MemmapSink::Files {
  explicit Files(const std::string& dir)
      : data(data_path(dir)), temp(temp_path(dir)), index(index_path(dir)), index_temp(index + ".tmp") {}
  ~Files() noexcept { abandon(); }
  Files(const Files&) = delete;
  Files& operator=(const Files&) = delete;

  static int close(int& descriptor) noexcept { return descriptor >= 0 ? ::close(std::exchange(descriptor, -1)) : 0; }

  void abandon() noexcept {
    close(fd);
    close(index_fd);
    if (owns_temp) {
      ::unlink(temp.c_str());
      owns_temp = false;
    }
    if (owns_index_temp) {
      ::unlink(index_temp.c_str());
      owns_index_temp = false;
    }
  }

  std::string data, temp, index, index_temp;
  int fd = -1, index_fd = -1;
  bool owns_temp = false, owns_index_temp = false, finished = false;
};

MemmapSink::MemmapSink(const std::string& dir, const Schema& schema, const std::vector<int64_t>& task_rows,
                       std::size_t chunk_bytes)
    : dir_(dir), task_rows_(task_rows) {
  const bool no_rows = std::all_of(task_rows.begin(), task_rows.end(), [](int64_t rows) { return rows == 0; });
  for (const auto& [name, tag] : schema.cols) {
    Field field{name, "", tag, 1, 0, itemsize_, false};
    // "f32[8]" is a vector; anything else with brackets (a nested list,
    // "date32[day]") is a tag the table below will not know.
    const std::size_t bracket = tag.find('[');
    if (bracket != std::string::npos && tag.back() == ']') {
      const std::string digits = tag.substr(bracket + 1, tag.size() - bracket - 2);
      const bool numeric = !digits.empty() && digits.size() <= 9 &&
                           std::all_of(digits.begin(), digits.end(), [](unsigned char c) { return std::isdigit(c); });
      const auto size = numeric ? static_cast<std::size_t>(std::stoull(digits)) : 0;
      if (size > 0) {
        field.tag = tag.substr(0, bracket);
        field.size = size;
        field.list = true;
      }
    }
    const std::optional<ElementType> element = element_type(field.tag);
    if (!element.has_value()) {
      if (field.tag == "utf8" && !field.list) {
        throw Error(memmap_error(name) + " is utf8; cast or encode strings before writing a memmap");
      }
      // A list column arrives under its Arrow name when its footers show no
      // single width (see memmap_schema in engine.cpp).
      std::string reason;
      if (tag.starts_with("list<") || tag.starts_with("large_list<")) {
        reason =
            ": a list column is written as a vector only when it holds integers, float32, float64 or bools, "
            "the same number in every row, with no null or empty row";
        if (no_rows) reason += "; the input has no rows to show how many values its vectors hold";
      }
      throw Error(memmap_error(name) + " is " + tag + ", which cannot be written to a memmap" + reason);
    }
    field.dtype = element->dtype;
    field.width = element->width;
    field.arrow_type = element->arrow_type;
    if (field.size > (std::numeric_limits<std::size_t>::max() - itemsize_) / field.width) {
      throw Error("memmap: record size exceeds supported limit");
    }
    itemsize_ += field.size * field.width;
    fields_.push_back(std::move(field));
  }
  if (fields_.empty()) throw Error("memmap: the input has no columns");
  chunk_rows_ = std::max<std::size_t>(1, chunk_bytes / itemsize_);

  task_offsets_.reserve(task_rows_.size());
  for (int64_t rows : task_rows_) {
    if (rows < 0) throw Error("memmap: task row count must not be negative");
    if (rows > std::numeric_limits<int64_t>::max() - rows_) throw Error("memmap: total row count overflows");
    task_offsets_.push_back(rows_);
    rows_ += rows;
  }
  const auto max_bytes = static_cast<std::uintmax_t>(std::numeric_limits<off_t>::max());
  if (itemsize_ > max_bytes || static_cast<std::uintmax_t>(rows_) > max_bytes / itemsize_) {
    throw Error("memmap: file size exceeds supported limit");
  }

  files_ = std::make_unique<Files>(dir_);
  std::error_code ec;
  std::filesystem::create_directories(dir_, ec);
  if (ec) throw Error(dir_ + ": cannot create output directory: " + ec.message());
  files_->fd = ::open(files_->temp.c_str(), O_RDWR | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
  if (files_->fd < 0) throw Error(files_->temp + ": cannot create: " + std::strerror(errno));
  files_->owns_temp = true;
  struct stat info{};
  if (::fstat(files_->fd, &info) != 0) throw Error(files_->temp + ": cannot inspect: " + std::strerror(errno));
  if (!S_ISREG(info.st_mode)) throw Error(files_->temp + ": cannot preallocate: output is not a regular file");
  if (::ftruncate(files_->fd, static_cast<off_t>(rows_) * static_cast<off_t>(itemsize_)) != 0) {
    throw Error(files_->temp + ": cannot preallocate: " + std::strerror(errno));
  }
}

MemmapSink::~MemmapSink() noexcept = default;

void MemmapSink::write(std::size_t task, const Batch& batch) const {
  if (task >= task_rows_.size()) throw Error("memmap: task index out of range");
  if (batch.rows != task_rows_[task]) {
    throw Error("memmap: task " + std::to_string(task) + " produced " + std::to_string(batch.rows) +
                " rows, expected " + std::to_string(task_rows_[task]));
  }
  write_slice(task, 0, batch);
}

void MemmapSink::write_slice(std::size_t task, int64_t row_offset, const Batch& batch) const {
  if (task >= task_rows_.size()) throw Error("memmap: task index out of range");
  if (row_offset < 0 || batch.rows < 0 || row_offset > task_rows_[task] || batch.rows > task_rows_[task] - row_offset) {
    throw Error("memmap: slice lies outside task " + std::to_string(task));
  }
  if (files_->fd < 0) throw Error("memmap: sink is no longer active");
  if (batch.names.size() != batch.cols.size()) throw Error("memmap: batch column names and arrays differ");
  const auto global_offset = task_offsets_[task] + row_offset;
  const auto rows = static_cast<std::size_t>(batch.rows);
  const std::string nulls = "; a memmap cannot hold nulls (fill or drop them first)";

  // Validate every field first, keeping the array of element values each
  // scatters from: the column itself, or a vector's child window.
  std::vector<std::shared_ptr<arrow::Array>> values;
  values.reserve(fields_.size());
  for (const Field& field : fields_) {
    const int idx = batch.find(field.name);
    if (idx < 0) throw Error(memmap_error(field.name) + " missing from batch");
    const std::shared_ptr<arrow::Array>& array = batch.cols[static_cast<std::size_t>(idx)];
    const ElementType element{field.dtype.c_str(), field.width, field.arrow_type};
    const std::string where = memmap_error(field.name);
    const std::string expected = field.list ? field.tag + "[" + std::to_string(field.size) + "]" : field.tag;

    if (!array || array->length() != batch.rows) throw Error(where + " length differs from batch row count");

    if (array->null_count() > 0) {
      throw Error(where + " has a null at row " + std::to_string(global_offset + first_null_row(*array, 1)) + nulls);
    }
    if (!field.list) {
      if (array->type_id() != element.arrow_type) {
        throw Error(where + " is " + array->type()->ToString() + ", expected " + expected);
      }
      values.push_back(array);
      continue;
    }
    // A vector arrives as a fixed-size list, or as the plain list Parquet
    // stores, whose rows must each hold the field's size.
    const arrow::Type::type kind = array->type_id();
    if ((kind != arrow::Type::FIXED_SIZE_LIST && kind != arrow::Type::LIST && kind != arrow::Type::LARGE_LIST) ||
        static_cast<const arrow::BaseListType&>(*array->type()).value_type()->id() != element.arrow_type ||
        (kind == arrow::Type::FIXED_SIZE_LIST &&
         static_cast<const arrow::FixedSizeListArray&>(*array).value_length() != static_cast<int32_t>(field.size))) {
      throw Error(where + " is " + array->type()->ToString() + ", expected " + expected);
    }
    std::shared_ptr<arrow::Array> window;
    if (kind == arrow::Type::FIXED_SIZE_LIST) {
      // The child covers every list; take the window this batch's rows use.
      const auto& list = static_cast<const arrow::FixedSizeListArray&>(*array);
      window = list.values()->Slice(list.value_offset(0), static_cast<int64_t>(rows * field.size));
    } else if (kind == arrow::Type::LIST) {
      window = uniform_window(static_cast<const arrow::ListArray&>(*array), field.size, global_offset, where);
    } else {
      window = uniform_window(static_cast<const arrow::LargeListArray&>(*array), field.size, global_offset, where);
    }
    if (window->null_count() > 0) {
      throw Error(where + " has a null at row " + std::to_string(global_offset + first_null_row(*window, field.size)) +
                  nulls);
    }
    values.push_back(std::move(window));
  }

  // Pack and write a bounded number of records at a time.
  const bool contiguous = fields_.size() == 1 && fields_[0].arrow_type != arrow::Type::BOOL;
  // A single numeric field is already in the required row-major layout.
  // Otherwise every byte of the buffer is overwritten by scatter, so it needs
  // no preliminary zero-fill.
  std::unique_ptr<uint8_t[]> buffer;
  if (!contiguous) buffer = std::make_unique_for_overwrite<uint8_t[]>(std::min(rows, chunk_rows_) * itemsize_);
  for (std::size_t first = 0; first < rows; first += chunk_rows_) {
    const std::size_t count = std::min(chunk_rows_, rows - first);
    const uint8_t* cursor;
    if (contiguous) {
      cursor = values[0]->data()->buffers[1]->data() +
               (static_cast<std::size_t>(values[0]->offset()) + first * fields_[0].size) * fields_[0].width;
    } else {
      for (std::size_t i = 0; i < fields_.size(); ++i) {
        const Field& field = fields_[i];
        const ElementType element{field.dtype.c_str(), field.width, field.arrow_type};
        scatter(*values[i], element, field.size, first, count, buffer.get(), itemsize_, field.offset);
      }
      cursor = buffer.get();
    }
    std::size_t remaining = count * itemsize_;
    off_t position = (static_cast<off_t>(global_offset) + static_cast<off_t>(first)) * static_cast<off_t>(itemsize_);
    while (remaining > 0) {
      const ssize_t written = ::pwrite(files_->fd, cursor, std::min(remaining, std::size_t{1} << 30), position);
      if (written < 0) {
        if (errno == EINTR) continue;
        throw Error(temp_path(dir_) + ": write failed: " + std::strerror(errno));
      }
      if (written == 0) throw Error(temp_path(dir_) + ": write failed: zero-byte write");
      cursor += written;
      remaining -= static_cast<std::size_t>(written);
      position += written;
    }
  }
}

std::string MemmapSink::finish() {
  auto& files = *files_;
  if (files.finished) return files.index;
  if (files.fd < 0) throw Error("memmap: sink is no longer active");
  try {
    // Allocate the return value and metadata before publication: no allocation
    // failure can then turn a published result into a reported failure.
    std::string path = files.index;
    nlohmann::json columns = nlohmann::json::array();
    for (const Field& field : fields_) {
      columns.push_back({{"name", field.name}, {"dtype", field.dtype}, {"size", field.size}, {"offset", field.offset}});
    }
    const nlohmann::json index{{"format", "fast-feature-transform-memmap"},
                               {"version", 1},
                               {"file", "data.mmap"},
                               {"rows", rows_},
                               {"itemsize", itemsize_},
                               {"byte_order", "little"},
                               {"columns", std::move(columns)}};
    const std::string metadata = index.dump(2);
    if (sync_to_disk(files.fd) != 0) throw Error(files.temp + ": flush failed: " + std::strerror(errno));
    if (Files::close(files.fd) != 0) throw Error(files.temp + ": close failed: " + std::strerror(errno));

    files.index_fd = ::open(files.index_temp.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    if (files.index_fd < 0) throw Error(files.index_temp + ": cannot write: " + std::strerror(errno));
    files.owns_index_temp = true;
    std::size_t offset = 0;
    while (offset < metadata.size()) {
      const ssize_t written =
          ::write(files.index_fd, metadata.data() + offset, std::min(metadata.size() - offset, std::size_t{1} << 30));
      if (written < 0) {
        if (errno == EINTR) continue;
        throw Error(files.index_temp + ": write failed: " + std::strerror(errno));
      }
      if (written == 0) throw Error(files.index_temp + ": write failed: zero-byte write");
      offset += static_cast<std::size_t>(written);
    }
    if (sync_to_disk(files.index_fd) != 0) throw Error(files.index_temp + ": flush failed: " + std::strerror(errno));
    if (Files::close(files.index_fd) != 0) throw Error(files.index_temp + ": close failed: " + std::strerror(errno));

    // Publish data first, then its index. The pair is not atomic: an index
    // rename failure leaves the new data in place. Release each temporary
    // immediately so later cleanup cannot unlink a subsequent writer's file.
    if (::rename(files.temp.c_str(), files.data.c_str()) != 0) {
      throw Error(files.data + ": " + std::strerror(errno));
    }
    files.owns_temp = false;
    if (::rename(files.index_temp.c_str(), files.index.c_str()) != 0) {
      throw Error(files.index + ": " + std::strerror(errno));
    }
    files.owns_index_temp = false;
    flush_to_disk(dir_, true);  // both new names, so they survive once finish returns
    files.finished = true;
    return path;
  } catch (...) {
    abandon();
    throw;
  }
}

void MemmapSink::abandon() noexcept { files_->abandon(); }

}  // namespace fastfeaturetransform
