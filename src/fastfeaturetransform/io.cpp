#include "fastfeaturetransform/io.hpp"

#include <arrow/api.h>
#include <arrow/array/array_dict.h>
#include <arrow/array/concatenate.h>
#include <arrow/array/data.h>
#include <arrow/array/util.h>
#include <arrow/chunked_array.h>
#include <arrow/extension_type.h>
#include <arrow/io/buffered.h>
#include <arrow/io/file.h>
#include <arrow/io/interfaces.h>
#include <arrow/table.h>
#include <arrow/type_traits.h>
#include <arrow/util/binary_view_util.h>
#include <arrow/util/bit_run_reader.h>
#include <arrow/util/bitmap_ops.h>
#include <arrow/util/bitmap_reader.h>
#include <arrow/util/byte_size.h>
#include <arrow/util/int_util.h>
#include <fcntl.h>
#include <parquet/arrow/reader.h>
#include <parquet/arrow/schema.h>
#include <parquet/arrow/writer.h>
#include <parquet/column_page.h>
#include <parquet/column_reader.h>
#include <parquet/encoding.h>
#include <parquet/exception.h>
#include <parquet/file_reader.h>
#include <parquet/schema.h>
#include <parquet/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <cerrno>
#include <cstddef>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>
#include <numeric>
#include <optional>
#include <string_view>
#include <system_error>
#include <unordered_map>
#include <utility>

#include "fastfeaturetransform/execution.hpp"
#include "fastfeaturetransform/kernels/output.hpp"

namespace fastfeaturetransform {
namespace {

std::shared_ptr<arrow::DataType> without_dictionaries(const std::shared_ptr<arrow::DataType>& type);

arrow::FieldVector fields_without_dictionaries(const arrow::DataType& type) {
  arrow::FieldVector fields;
  for (const std::shared_ptr<arrow::Field>& field : type.fields()) {
    fields.push_back(field->WithType(without_dictionaries(field->type())));
  }
  return fields;
}

// A list, struct or map with dictionaries replaced by their value types and
// strings and bytes named by their logical types. Arrow's dictionary reader
// produces string or binary for all storage layouts of those types. Delta
// pages retain their stored layout. Use the same tags for both.
// REGRESSION: two files that hold the same list of strings, one in
// dictionary pages and one in delta pages, have one tag and are read
// together, whether the files store string, large_string or string_view. The
// tag used to be the Arrow type's own name, dictionary and all, and the
// second file was refused as holding another type; once the dictionary was
// left out, a list of large_string or string_view still was.
std::shared_ptr<arrow::DataType> nested_without_dictionaries(const arrow::DataType& type) {
  switch (type.id()) {
    case arrow::Type::LIST:
      return arrow::list(fields_without_dictionaries(type).front());
    case arrow::Type::LARGE_LIST:
      return arrow::large_list(fields_without_dictionaries(type).front());
    case arrow::Type::FIXED_SIZE_LIST:
      return arrow::fixed_size_list(fields_without_dictionaries(type).front(),
                                    static_cast<const arrow::FixedSizeListType&>(type).list_size());
    case arrow::Type::STRUCT:
      return arrow::struct_(fields_without_dictionaries(type));
    case arrow::Type::MAP:
      return std::make_shared<arrow::MapType>(fields_without_dictionaries(type).front(),
                                              static_cast<const arrow::MapType&>(type).keys_sorted());
    default:
      return nullptr;
  }
}

std::shared_ptr<arrow::DataType> without_dictionaries(const std::shared_ptr<arrow::DataType>& type) {
  if (type->id() == arrow::Type::DICTIONARY) {
    return without_dictionaries(static_cast<const arrow::DictionaryType&>(*type).value_type());
  }
  if (type->id() == arrow::Type::LARGE_STRING || type->id() == arrow::Type::STRING_VIEW) return arrow::utf8();
  if (type->id() == arrow::Type::LARGE_BINARY || type->id() == arrow::Type::BINARY_VIEW) return arrow::binary();
  std::shared_ptr<arrow::DataType> nested = nested_without_dictionaries(*type);
  return nested ? nested : type;
}

std::string type_tag(const arrow::DataType& type) {
  switch (type.id()) {
    case arrow::Type::DICTIONARY:
      return type_tag(*static_cast<const arrow::DictionaryType&>(type).value_type());
    case arrow::Type::STRING:
    case arrow::Type::LARGE_STRING:
    case arrow::Type::STRING_VIEW:  // read as large_string: see strings_from_views
      return "utf8";
    case arrow::Type::BINARY:
    case arrow::Type::LARGE_BINARY:
    case arrow::Type::BINARY_VIEW:
      return "binary";
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
    default: {
      const std::shared_ptr<arrow::DataType> nested = nested_without_dictionaries(type);
      return nested ? nested->ToString() : type.ToString();
    }
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

const arrow::DataType& storage_type(const arrow::DataType& type) {
  if (type.id() != arrow::Type::EXTENSION) return type;
  return storage_type(*static_cast<const arrow::ExtensionType&>(type).storage_type());
}

// Arrow's reader fills the null rows of a fixed-size list by concatenating
// pieces of its child array. For a child read as a dictionary, that merges
// the pieces' dictionaries and renumbers every index through a table it reads
// without a bounds check: a corrupt index is looked up outside the table
// before check_dictionary_indices sees the batch, and the number found there
// names a wrong entry, which passes the check when it lies inside the merged
// dictionary. Strings at any depth inside a fixed-size list are therefore
// read plain: Arrow's decoder refuses such an index itself, and the reader
// names the column it fails on (undecodable_column).
void read_plain_inside_fixed_size_lists(const parquet::arrow::SchemaField& field, bool inside,
                                        parquet::ArrowReaderProperties& properties) {
  inside = inside || storage_type(*field.field->type()).id() == arrow::Type::FIXED_SIZE_LIST;
  if (field.is_leaf()) {
    if (inside) properties.set_read_dictionary(field.column_index, false);
    return;
  }
  for (const auto& child : field.children) read_plain_inside_fixed_size_lists(child, inside, properties);
}

std::unique_ptr<parquet::arrow::FileReader> make_reader(const std::string& path,
                                                        std::unique_ptr<parquet::ParquetFileReader> raw,
                                                        const parquet::ArrowReaderProperties& properties) {
  auto reader = parquet::arrow::FileReader::Make(arrow::default_memory_pool(), std::move(raw), properties);
  if (!reader.ok()) throw Error(path + ": " + reader.status().ToString());
  return std::move(*reader);
}

// A column chunk is read through a 64 KiB buffer, so a reader holds the pages
// of the batch it decodes and not the whole chunk.
parquet::ReaderProperties buffered_properties() {
  auto properties = parquet::default_reader_properties();
  properties.enable_buffered_stream();
  properties.set_buffer_size(64 * 1024);
  return properties;
}

std::unique_ptr<parquet::ParquetFileReader> open_raw(const std::string& path,
                                                     std::shared_ptr<parquet::FileMetaData> metadata) {
  try {
    return parquet::ParquetFileReader::Open(open_file(path), buffered_properties(), std::move(metadata));
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
// against the dictionary. The kernels index their caches and the dictionary
// with them unchecked, and Arrow's Parquet writer indexes the dictionary of a
// column passed through, so a corrupt file would be read outside both. Strings
// are read as a dictionary at the top of a column and inside a list, large
// list, struct or map, where a child array holds them, so the check visits
// every child: one pass per batch here covers every kernel and the writer.
// It bounds indices by the batch's dictionary, which also holds the values of
// plain pages that follow the dictionary page; in a chunk that holds plain
// pages, each page of indices is also bounded by the dictionary page's own
// count (see WatchedPages).
// Strings inside a fixed-size list are read plain instead (see
// read_plain_inside_fixed_size_lists). The reader's indices are int32, and
// their largest value read as unsigned (a negative one is then too large)
// settles almost every batch in a tenth of the time Arrow's check takes. Null
// slots count in that maximum, and Parquet leaves zeros there; when the
// maximum is out of range, Arrow's check, which skips null slots, decides.
void check_dictionary_indices(const arrow::ArrayData& column, const std::string& where, const std::string& name) {
  const auto& dictionary = static_cast<const arrow::DictionaryType&>(storage_type(*column.type));
  const arrow::DataType& index_type = *dictionary.index_type();
  const auto entries = static_cast<uint64_t>(column.dictionary->length);
  if (index_type.id() == arrow::Type::INT32) {
    const int32_t* values = column.GetValues<int32_t>(1);
    uint32_t largest = 0;
    for (int64_t row = 0; row < column.length; ++row) largest = std::max(largest, static_cast<uint32_t>(values[row]));
    if (largest < entries) return;
  }
  arrow::ArraySpan indices(column);
  indices.type = &index_type;
  const arrow::Status status = arrow::internal::CheckIndexBounds(indices, entries);
  if (!status.ok()) {
    throw Error(where + ": column '" + name + "' has a dictionary index outside its dictionary: " + status.message());
  }
}

// Arrow inserts each dictionary value into a table of distinct values, then
// decodes dictionary indices against that table. Before any plain data page
// adds entries, the table length equals the page count exactly when all
// entries are distinct. After plain pages add values, a repeated entry
// places the page's last value before its declared position. Plain pages
// append only previously unseen values, so they cannot restore it there.
// A batch may contain only earlier plain rows. Check the original page
// once if the count and last value do not prove that its entries are unique.
struct DictionaryChunk {
  bool watched = false;
  int32_t entries = -1;
  std::optional<std::string> last;
  bool indexed = false;
  bool plain = false;
  bool settled = false;
  std::optional<int32_t> outside;  // a data page's index at or past `entries`
  int unchecked = 0;               // pages of indices passed before any plain page
  int64_t after_dictionary = 0;    // bytes from the chunk's start to the page after its dictionary page
};

// Parquet stores the length of a string or byte value in a PLAIN page as 4
// little-endian bytes, which this machine reads as its own integer.
static_assert(std::endian::native == std::endian::little, "Parquet stores lengths little-endian");

// The PLAIN string or byte value at `at`, its 4-byte length and then its
// bytes, with `at` moved past it; nullopt when the bytes before `end` do not
// hold it.
std::optional<std::string_view> next_plain_value(const uint8_t*& at, const uint8_t* end) {
  if (end - at < 4) return std::nullopt;
  uint32_t length = 0;
  std::memcpy(&length, at, 4);
  if (length > static_cast<uint64_t>(end - at - 4)) return std::nullopt;
  const std::string_view value(reinterpret_cast<const char*>(at + 4), length);
  at += 4 + static_cast<std::ptrdiff_t>(length);
  return value;
}

// Whether data page `page` holds dictionary indices: format 1.0 marks them
// PLAIN_DICTIONARY.
bool holds_indices(const parquet::DataPage& page) {
  return page.encoding() == parquet::Encoding::RLE_DICTIONARY || page.encoding() == parquet::Encoding::PLAIN_DICTIONARY;
}

// The dictionary indices of the data pages of a string or byte leaf, decoded
// as Arrow's reader decodes them.
class PageIndices {
 public:
  explicit PageIndices(const parquet::ColumnDescriptor& column) : column_(column) {}

  // The first index of data page `page` at or past `entries`, or nullopt.
  // Arrow decodes an index for each definition level that reaches the
  // leaf's maximum, so exactly those are checked. A page whose levels or
  // indices cannot be decoded is left to Arrow, which fails on the same
  // bytes.
  std::optional<int32_t> first_outside(const parquet::DataPage& page, int32_t entries) {
    try {
      const uint8_t* data = page.data();
      const int64_t size = page.size();
      const int16_t max_definition = column_.max_definition_level();
      int64_t levels = 0;  // the bytes of levels in front of the indices
      if (page.type() == parquet::PageType::DATA_PAGE_V2) {
        // The header states the lengths of the levels, which Arrow's page
        // reader has found to be at least 0, but not at most the page's size.
        // A version 1 page holds each length in its bytes, and LevelDecoder
        // refuses one past the page.
        const auto& v2 = static_cast<const parquet::DataPageV2&>(page);
        levels = int64_t{v2.repetition_levels_byte_length()} + v2.definition_levels_byte_length();
        if (levels > size) return std::nullopt;
        if (max_definition > 0) {
          definitions_.SetDataV2(v2.definition_levels_byte_length(), max_definition, page.num_values(),
                                 data + v2.repetition_levels_byte_length());
        }
      } else {
        const auto& v1 = static_cast<const parquet::DataPageV1&>(page);
        if (column_.max_repetition_level() > 0) {
          levels = repetitions_.SetData(v1.repetition_level_encoding(), column_.max_repetition_level(),
                                        page.num_values(), data, static_cast<int32_t>(size));
        }
        if (max_definition > 0) {
          levels += definitions_.SetData(v1.definition_level_encoding(), max_definition, page.num_values(),
                                         data + levels, static_cast<int32_t>(size - levels));
        }
      }
      // The levels are decoded and counted: Decode stops where the page's
      // levels end, and Arrow's reader then refuses a page short of levels.
      // Arrow's CountUpTo asserts, in a debug build of Arrow, that the
      // levels fill the page, which ends the process.
      int count = page.num_values();
      if (max_definition > 0) {
        count = 0;
        for (int left = page.num_values(); left > 0;) {
          const int step = definitions_.Decode(std::min(left, static_cast<int>(levels_.size())), levels_.data());
          if (step <= 0) break;
          count += static_cast<int>(std::count(levels_.data(), levels_.data() + step, max_definition));
          left -= step;
        }
      }
      if (count <= 0) return std::nullopt;
      if (!indices_) indices_ = parquet::MakeDictDecoder<parquet::ByteArrayType>(&column_);
      indices_->SetData(count, data + levels, static_cast<int>(size - levels));
      const auto bound = static_cast<uint32_t>(entries);
      for (int done = 0; done < count;) {
        const int step = std::min(count - done, static_cast<int>(decoded_.size()));
        indices_->DecodeIndices(step, decoded_.data());
        const int32_t* first = decoded_.data();
        const int32_t* last = first + step;
        // A negative index read as unsigned is past any count.
        uint32_t largest = 0;
        for (const int32_t* index = first; index != last; ++index) {
          largest = std::max(largest, static_cast<uint32_t>(*index));
        }
        if (largest >= bound) {
          return *std::find_if(first, last, [&](int32_t index) { return static_cast<uint32_t>(index) >= bound; });
        }
        done += step;
      }
    } catch (const parquet::ParquetException&) {
      // Arrow refuses the page when it decodes it.
    }
    return std::nullopt;
  }

 private:
  const parquet::ColumnDescriptor& column_;
  parquet::LevelDecoder repetitions_;
  parquet::LevelDecoder definitions_;
  std::unique_ptr<parquet::DictDecoder<parquet::ByteArrayType>> indices_;
  std::array<int16_t, 1024> levels_{};
  std::array<int32_t, 1024> decoded_{};
};

// The pages of a column chunk as Arrow reads them. A dictionary page that
// states more values than its bytes hold is refused before Arrow decodes it.
// For a leaf that `chunks` watches, this also tracks the dictionary count
// and data page encodings, and bounds each page of indices that follows a
// plain page by the dictionary page's count.
//
// Arrow's reader of strings as a dictionary decodes each page of indices
// without a bounds check, and appends the values of the plain pages that a
// writer falls back to after them to the same dictionary, which it keeps for
// the whole chunk. In a chunk that holds both, an index at or past the
// dictionary page's count can name the value of a plain page inside a
// batch's dictionary, where check_dictionary_indices cannot see it. Until a
// plain page has gone to Arrow, a batch's dictionary holds only the values
// of the dictionary page, and that check bounds every index of the batch by
// them. So a page of indices that follows a plain page is checked as it
// passes, with the decoder Arrow uses. One that comes before any plain page
// is only counted, and its bytes do not outlive the next page: the check of
// the first batch after a plain page reads those pages again
// (index_outside_before_plain), from where `stream`, the chunk's bytes, stood
// after the dictionary page. `stream` is null for a chunk that Arrow's own
// page reader reads, which is then read again from its start.
class WatchedPages final : public parquet::PageReader {
 public:
  WatchedPages(std::unique_ptr<parquet::PageReader> pages, std::shared_ptr<arrow::io::InputStream> stream,
               std::vector<DictionaryChunk>& chunks, std::size_t leaf, const parquet::ColumnDescriptor& column)
      : pages_(std::move(pages)),
        stream_(std::move(stream)),
        chunks_(chunks),
        leaf_(leaf),
        column_(column),
        watched_(chunks[leaf].watched) {
    if (watched_) {
      chunks_[leaf_] = DictionaryChunk{};
      chunks_[leaf_].watched = true;
    }
  }

  std::shared_ptr<parquet::Page> NextPage() override {
    std::shared_ptr<parquet::Page> page = pages_->NextPage();
    if (page == nullptr) return page;
    if (page->type() == parquet::PageType::DICTIONARY_PAGE) {
      check_values(static_cast<const parquet::DictionaryPage&>(*page));
    }
    if (!watched_) return page;
    DictionaryChunk& chunk = chunks_[leaf_];
    if (page->type() == parquet::PageType::DICTIONARY_PAGE) {
      const auto& dictionary = static_cast<const parquet::DictionaryPage&>(*page);
      chunk.entries = dictionary.num_values();
      if (stream_) {
        const arrow::Result<int64_t> after = stream_->Tell();
        if (after.ok()) chunk.after_dictionary = *after;
      }
      const uint8_t* at = dictionary.data();
      if (dictionary.size() > 0 && chunk.entries > 0) {
        const uint8_t* end = at + dictionary.size();
        for (int32_t i = 0; i < chunk.entries; ++i) {
          const auto value = next_plain_value(at, end);
          if (!value) break;
          // REGRESSION: the last value is kept whatever its length, so that
          // a batch that shows the page is checked without reading the page
          // again. Arrow's reader holds every value of the page for the
          // whole chunk, so the copy adds one value to that. A last value
          // over 256 bytes used to be left out, and a chunk of such values
          // whose first checked batch came after a plain page had its
          // dictionary page read and sorted a second time
          // (dictionary_page_repeats).
          if (i == chunk.entries - 1) chunk.last.emplace(*value);
        }
      }
    } else if (page->type() == parquet::PageType::DATA_PAGE || page->type() == parquet::PageType::DATA_PAGE_V2) {
      const auto& data = static_cast<const parquet::DataPage&>(*page);
      if (!holds_indices(data)) {
        chunk.plain = true;
      } else {
        chunk.indexed = true;
        if (!chunk.plain) {
          ++chunk.unchecked;
        } else if (chunk.entries >= 0 && !chunk.outside) {
          if (!indices_) indices_ = std::make_unique<PageIndices>(column_);
          chunk.outside = indices_->first_outside(data, chunk.entries);
        }
      }
    }
    return page;
  }

  void set_max_page_header_size(uint32_t size) override { pages_->set_max_page_header_size(size); }

 private:
  // Refuses `page` when its header states more values than its bytes hold.
  // Arrow decodes only PLAIN dictionary pages, which format 1.0 marks
  // PLAIN_DICTIONARY, into an array that it sizes by the count the header
  // states before it decodes a value. A PLAIN value takes its type's width,
  // a bool one bit, and a string or bytes value at least the 4 bytes of its
  // length.
  void check_values(const parquet::DictionaryPage& page) const {
    if (page.encoding() != parquet::Encoding::PLAIN && page.encoding() != parquet::Encoding::PLAIN_DICTIONARY) return;
    int64_t bits = 0;
    switch (column_.physical_type()) {
      case parquet::Type::BOOLEAN:
        bits = 1;
        break;
      case parquet::Type::INT32:
      case parquet::Type::FLOAT:
      case parquet::Type::BYTE_ARRAY:
        bits = 32;
        break;
      case parquet::Type::INT64:
      case parquet::Type::DOUBLE:
        bits = 64;
        break;
      case parquet::Type::INT96:
        bits = 96;
        break;
      case parquet::Type::FIXED_LEN_BYTE_ARRAY:
        bits = 8 * int64_t{column_.type_length()};
        break;
      default:
        break;
    }
    if (bits <= 0) return;
    const int64_t most = int64_t{page.size()} * 8 / bits;
    if (page.num_values() <= most) return;
    throw parquet::ParquetStatusException(arrow::Status::Invalid("a dictionary page states ", page.num_values(),
                                                                 " values where its ", page.size(),
                                                                 " bytes hold at most ", most));
  }

  std::unique_ptr<parquet::PageReader> pages_;
  std::shared_ptr<arrow::io::InputStream> stream_;
  std::vector<DictionaryChunk>& chunks_;
  std::size_t leaf_;
  const parquet::ColumnDescriptor& column_;
  bool watched_;
  std::unique_ptr<PageIndices> indices_;  // made for the first page it checks
};

// The bytes of a column chunk as Arrow's page reader reads them, refusing a
// read of more bytes than the chunk has left before the stream under it is
// asked for them. The page reader reads a page by the size its header
// states, up to 2 GiB, and the buffered stream under it requests that many
// bytes and fills what the chunk does not hold with zeros before the page
// is refused.
class ChunkStream final : public arrow::io::InputStream {
 public:
  ChunkStream(std::shared_ptr<arrow::io::InputStream> stream, int64_t length)
      : stream_(std::move(stream)), left_(length) {}

  arrow::Status Close() override { return stream_->Close(); }
  bool closed() const override { return stream_->closed(); }
  arrow::Result<int64_t> Tell() const override { return stream_->Tell(); }
  arrow::Result<std::string_view> Peek(int64_t nbytes) override { return stream_->Peek(nbytes); }

  arrow::Result<int64_t> Read(int64_t nbytes, void* out) override {
    ARROW_RETURN_NOT_OK(check(nbytes));
    ARROW_ASSIGN_OR_RAISE(const int64_t read, stream_->Read(nbytes, out));
    left_ -= read;
    return read;
  }

  arrow::Result<std::shared_ptr<arrow::Buffer>> Read(int64_t nbytes) override {
    ARROW_RETURN_NOT_OK(check(nbytes));
    ARROW_ASSIGN_OR_RAISE(std::shared_ptr<arrow::Buffer> read, stream_->Read(nbytes));
    left_ -= read->size();
    return read;
  }

 private:
  arrow::Status check(int64_t nbytes) const {
    if (nbytes <= left_) return arrow::Status::OK();
    return arrow::Status::Invalid("a page states ", nbytes, " bytes where its column chunk has ", left_, " left");
  }

  std::shared_ptr<arrow::io::InputStream> stream_;
  int64_t left_;
};

// A row group of a WatchedFile. Each column chunk's pages are opened as
// Arrow's row group reader opens them, except that they are read through a
// ChunkStream, and pass through WatchedPages. An encrypted chunk, which
// needs a key the engine never has, and a chunk that the footer places
// outside the file are left to Arrow's reader, which refuses both before it
// reads a page.
class WatchedRowGroup final : public parquet::RowGroupReader::Contents {
 public:
  WatchedRowGroup(std::shared_ptr<parquet::RowGroupReader> group, parquet::ReaderProperties properties,
                  std::vector<DictionaryChunk>& chunks, std::shared_ptr<arrow::io::RandomAccessFile> input,
                  int64_t size, bool short_chunks, bool always_compressed)
      : group_(std::move(group)),
        properties_(std::move(properties)),
        chunks_(chunks),
        input_(std::move(input)),
        size_(size),
        short_chunks_(short_chunks),
        always_compressed_(always_compressed) {}

  std::unique_ptr<parquet::PageReader> GetColumnPageReader(int i) override {
    std::shared_ptr<arrow::io::InputStream> stream;
    std::unique_ptr<parquet::PageReader> pages = open_pages(i, stream);
    return std::make_unique<WatchedPages>(std::move(pages), std::move(stream), chunks_, static_cast<std::size_t>(i),
                                          *group_->metadata()->schema()->Column(i));
  }
  const parquet::RowGroupMetaData* metadata() const override { return group_->metadata(); }
  const parquet::ReaderProperties* properties() const override { return &properties_; }

 private:
  // The pages of leaf `i`, read through `stream` unless Arrow's own page
  // reader reads them. A second read of a chunk's pages of indices
  // (index_outside_before_plain) gives the leaf an unwatched DictionaryChunk
  // that notes where the first read found the page after the dictionary
  // page, and starts there. A watched one belongs to a first read, and still
  // holds what the leaf's previous chunk noted.
  std::unique_ptr<parquet::PageReader> open_pages(int i, std::shared_ptr<arrow::io::InputStream>& stream) {
    const std::unique_ptr<parquet::ColumnChunkMetaData> chunk = group_->metadata()->ColumnChunk(i);
    int64_t start = chunk->data_page_offset();
    if (chunk->has_dictionary_page() && chunk->dictionary_page_offset() > 0 &&
        start > chunk->dictionary_page_offset()) {
      start = chunk->dictionary_page_offset();
    }
    int64_t length = chunk->total_compressed_size();
    if (chunk->crypto_metadata() != nullptr || start < 0 || length < 0 || start > size_ || length > size_ - start) {
      return group_->GetColumnPageReader(i);
    }
    if (short_chunks_) length += std::min<int64_t>(100, size_ - start - length);
    const DictionaryChunk& noted = chunks_[static_cast<std::size_t>(i)];
    if (!noted.watched && noted.after_dictionary > 0) {
      start += noted.after_dictionary;
      length -= noted.after_dictionary;
    }
    stream = std::make_shared<ChunkStream>(properties_.GetStream(input_, start, length), length);
    return parquet::PageReader::Open(stream, chunk->num_values(), chunk->compression(), properties_,
                                     *group_->metadata()->schema()->Column(i), always_compressed_);
  }

  std::shared_ptr<parquet::RowGroupReader> group_;
  parquet::ReaderProperties properties_;
  std::vector<DictionaryChunk>& chunks_;
  std::shared_ptr<arrow::io::RandomAccessFile> input_;
  int64_t size_;  // of the file
  bool short_chunks_;
  bool always_compressed_;
};

// A Parquet file whose row groups are WatchedRowGroups. Arrow pre-buffers a
// file by casting it to its own class, so a reader of a WatchedFile must not
// pre-buffer.
class WatchedFile final : public parquet::ParquetFileReader::Contents {
 public:
  WatchedFile(std::unique_ptr<parquet::ParquetFileReader::Contents> file, parquet::ReaderProperties properties,
              std::vector<DictionaryChunk>& chunks, std::shared_ptr<arrow::io::RandomAccessFile> input, int64_t size)
      : file_(std::move(file)),
        properties_(std::move(properties)),
        chunks_(chunks),
        input_(std::move(input)),
        size_(size) {
    const std::shared_ptr<parquet::FileMetaData> metadata = file_->metadata();
    // parquet-mr 1.2.8 and earlier left a dictionary page's header out of a
    // column chunk's size, and Arrow reads up to 100 bytes past such a chunk.
    short_chunks_ = metadata->writer_version().VersionLt(parquet::ApplicationVersion::PARQUET_816_FIXED_VERSION());
    // parquet-cpp before Arrow 3.0.0 marked the data pages v2 it compressed
    // as uncompressed, and Arrow decompresses them all the same.
    always_compressed_ =
        metadata->writer_version().VersionLt(parquet::ApplicationVersion::PARQUET_CPP_10353_FIXED_VERSION());
  }

  void Close() override { file_->Close(); }
  std::shared_ptr<parquet::RowGroupReader> GetRowGroup(int i) override {
    return std::make_shared<parquet::RowGroupReader>(std::make_unique<WatchedRowGroup>(
        file_->GetRowGroup(i), properties_, chunks_, input_, size_, short_chunks_, always_compressed_));
  }
  std::shared_ptr<parquet::FileMetaData> metadata() const override { return file_->metadata(); }
  std::shared_ptr<parquet::PageIndexReader> GetPageIndexReader() override { return file_->GetPageIndexReader(); }
  parquet::BloomFilterReader& GetBloomFilterReader() override { return file_->GetBloomFilterReader(); }

 private:
  std::unique_ptr<parquet::ParquetFileReader::Contents> file_;
  parquet::ReaderProperties properties_;
  std::vector<DictionaryChunk>& chunks_;
  std::shared_ptr<arrow::io::RandomAccessFile> input_;
  int64_t size_;
  bool short_chunks_ = false;
  bool always_compressed_ = false;
};

// The file at `path`, opened as open_raw opens it, reading the pages of each
// leaf through WatchedPages.
std::unique_ptr<parquet::ParquetFileReader> open_watched(const std::string& path,
                                                         const std::shared_ptr<arrow::io::RandomAccessFile>& input,
                                                         std::shared_ptr<parquet::FileMetaData> metadata,
                                                         std::vector<DictionaryChunk>& chunks) {
  try {
    const parquet::ReaderProperties properties = buffered_properties();
    auto size = input->GetSize();
    if (!size.ok()) throw Error(path + ": " + size.status().ToString());
    auto file = std::make_unique<parquet::ParquetFileReader>();
    file->Open(std::make_unique<WatchedFile>(
        parquet::ParquetFileReader::Contents::Open(input, properties, std::move(metadata)), properties, chunks, input,
        *size));
    return file;
  } catch (const parquet::ParquetException& e) {
    throw Error(path + ": " + e.what());
  }
}

// Marks in `chunks` the leaves under `field` that Arrow reads as a dictionary
// of strings or bytes: a BYTE_ARRAY leaf whose Arrow type is a dictionary.
// Arrow decodes any other leaf plain, and builds a dictionary of other values
// from what it decoded, so no page of theirs needs watching.
void watch_dictionary_leaves(const parquet::arrow::SchemaField& field, const parquet::SchemaDescriptor& schema,
                             std::vector<DictionaryChunk>& chunks, const std::vector<int>* selected,
                             const std::string& path, const std::string& name) {
  if (!field.is_leaf()) {
    for (const auto& child : field.children) watch_dictionary_leaves(child, schema, chunks, selected, path, name);
    return;
  }
  const arrow::DataType& type = *field.field->type();
  // Arrow cannot wrap an empty collection of dictionary chunks as an
  // extension. Only a projected leaf will enter its stream reader.
  if (type.id() == arrow::Type::EXTENSION && storage_type(type).id() == arrow::Type::DICTIONARY &&
      (selected == nullptr || std::binary_search(selected->begin(), selected->end(), field.column_index))) {
    throw Error(path + ": column '" + name + "' has an extension with dictionary storage, which cannot be streamed");
  }
  chunks[static_cast<std::size_t>(field.column_index)].watched =
      type.id() == arrow::Type::DICTIONARY &&
      schema.Column(field.column_index)->physical_type() == parquet::Type::BYTE_ARRAY;
}

// Whether `dictionary`, a batch's dictionary of strings or bytes, starts with
// the values of the chunk's dictionary page: it holds as many values as the
// page declares, or more, and the page's last value at the page's last
// position (see DictionaryChunk).
bool shows_dictionary_page(const arrow::ArrayData& dictionary, const DictionaryChunk& chunk) {
  if (!chunk.last || dictionary.length < chunk.entries) return false;
  const int64_t at = chunk.entries - 1;
  std::string_view value;
  switch (dictionary.type->id()) {
    case arrow::Type::STRING:
    case arrow::Type::BINARY: {
      if (dictionary.buffers.size() < 3) return false;
      const auto& data = dictionary.buffers[2];
      const char* bytes = data && data->data() ? reinterpret_cast<const char*>(data->data()) : "";
      const int32_t* offsets = dictionary.GetValues<int32_t>(1);
      value = std::string_view(bytes + offsets[at], static_cast<std::size_t>(offsets[at + 1] - offsets[at]));
      break;
    }
    case arrow::Type::LARGE_STRING:
    case arrow::Type::LARGE_BINARY: {
      if (dictionary.buffers.size() < 3) return false;
      const auto& data = dictionary.buffers[2];
      const char* bytes = data && data->data() ? reinterpret_cast<const char*>(data->data()) : "";
      const int64_t* offsets = dictionary.GetValues<int64_t>(1);
      value = std::string_view(bytes + offsets[at], static_cast<std::size_t>(offsets[at + 1] - offsets[at]));
      break;
    }
    default:
      return false;
  }
  return value == *chunk.last;
}

// Whether the dictionary page of column chunk (`row_group`, `leaf`) holds a
// value twice, read from the file again.
bool dictionary_page_repeats(const std::shared_ptr<arrow::io::RandomAccessFile>& input,
                             const std::shared_ptr<parquet::FileMetaData>& metadata, int row_group, int leaf,
                             const std::string& where) {
  try {
    const auto file = parquet::ParquetFileReader::Open(input, buffered_properties(), metadata);
    const std::shared_ptr<parquet::RowGroupReader> group = file->RowGroup(row_group);
    const std::unique_ptr<parquet::PageReader> pages = group->GetColumnPageReader(leaf);
    while (const std::shared_ptr<parquet::Page> page = pages->NextPage()) {
      if (page->type() != parquet::PageType::DICTIONARY_PAGE) continue;
      const auto& dictionary = static_cast<const parquet::DictionaryPage&>(*page);
      std::vector<std::string_view> values;
      const uint8_t* at = dictionary.data();
      if (dictionary.num_values() <= 0) return false;
      if (!at || dictionary.size() == 0) throw Error(where + ": incomplete dictionary page");
      const uint8_t* const end = at + dictionary.size();
      for (int32_t entry = 0; entry < dictionary.num_values(); ++entry) {
        const std::optional<std::string_view> value = next_plain_value(at, end);
        if (!value) throw Error(where + ": incomplete dictionary page");
        values.push_back(*value);
      }
      std::sort(values.begin(), values.end());
      return std::adjacent_find(values.begin(), values.end()) != values.end();
    }
  } catch (const parquet::ParquetException& e) {
    throw Error(where + ": " + e.what());
  }
  return false;
}

// The first index at or past `entries` in the first `count` pages of
// indices of column chunk (`row_group`, `leaf`) of the file at `path`, which
// come before its first plain page, or nullopt. The chunk's pages up to the
// last of them are read from `input` again, through the bounds of
// WatchedRowGroup; the plain page after them is not, nor is the dictionary
// page before them when the first read noted `after_dictionary`, the bytes
// from the chunk's start to the page that follows it.
std::optional<int32_t> index_outside_before_plain(const std::string& path,
                                                  const std::shared_ptr<arrow::io::RandomAccessFile>& input,
                                                  const std::shared_ptr<parquet::FileMetaData>& metadata, int row_group,
                                                  int leaf, int count, int32_t entries, int64_t after_dictionary,
                                                  const std::string& where) {
  try {
    std::vector<DictionaryChunk> unwatched(static_cast<std::size_t>(metadata->num_columns()));
    unwatched[static_cast<std::size_t>(leaf)].after_dictionary = after_dictionary;
    const std::unique_ptr<parquet::ParquetFileReader> file = open_watched(path, input, metadata, unwatched);
    const std::shared_ptr<parquet::RowGroupReader> group = file->RowGroup(row_group);
    const std::unique_ptr<parquet::PageReader> pages = group->GetColumnPageReader(leaf);
    PageIndices indices(*metadata->schema()->Column(leaf));
    for (int left = count; left > 0;) {
      const std::shared_ptr<parquet::Page> page = pages->NextPage();
      if (page == nullptr) break;
      if (page->type() != parquet::PageType::DATA_PAGE && page->type() != parquet::PageType::DATA_PAGE_V2) continue;
      const auto& data = static_cast<const parquet::DataPage&>(*page);
      if (!holds_indices(data)) break;
      if (const std::optional<int32_t> index = indices.first_outside(data, entries)) return index;
      --left;
    }
  } catch (const parquet::ParquetException& e) {
    throw Error(where + ": " + e.what());
  }
  return std::nullopt;
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

// The entries at `order` of a dictionary of strings or bytes, copied in that
// order into an array of their type. ArrowType is arrow::BinaryType for
// 32-bit offsets and arrow::LargeBinaryType for 64-bit ones; strings have the
// layout of bytes. A null entry stays null: Parquet stores none, and only an
// array made in memory holds one.
template <typename ArrowType>
std::shared_ptr<arrow::Array> take_entries(const arrow::Array& entries, const std::vector<int32_t>& order,
                                           const std::string& where) {
  const auto& values = static_cast<const typename arrow::TypeTraits<ArrowType>::ArrayType&>(entries);
  const bool nulls = values.null_count() != 0;
  int64_t bytes = 0;
  for (const int32_t entry : order) bytes += static_cast<int64_t>(values.value_length(entry));
  typename arrow::TypeTraits<ArrowType>::BuilderType builder;
  arrow::Status status = builder.Reserve(static_cast<int64_t>(order.size()));
  if (status.ok()) status = builder.ReserveData(bytes);
  std::shared_ptr<arrow::Array> taken;
  if (status.ok()) {
    for (const int32_t entry : order) {
      if (nulls && values.IsNull(entry)) {
        builder.UnsafeAppendNull();
      } else {
        builder.UnsafeAppend(values.GetView(entry));
      }
    }
    status = builder.Finish(&taken);
  }
  if (!status.ok()) throw Error(where + ": " + status.ToString());
  std::shared_ptr<arrow::ArrayData> data = taken->data()->Copy();
  data->type = entries.type();  // the builder names its values binary
  return arrow::MakeArray(data);
}

// Whether the rows of a dictionary column that hold a value use every one of
// its `entries` entries. A row whose index lies outside the dictionary can
// give either answer: the trim below leaves such a column as it is.
//
// When a writer builds a column chunk's dictionary while it encodes the rows,
// it numbers the entries in the order the rows first use them, as the trim
// below does. In that order no index is more than one past the largest
// before it, and the rows use every entry once the largest is the last.
// Checking that needs no table, and it settles every batch the reader
// trimmed and the first batch of a row group whose dictionary page such a
// writer wrote. Rows in another order, such as a later batch of a row group
// that the reader did not trim, are counted in a table of the entries seen;
// a column of few entries usually meets them all early.
//
// The check visits only the rows that hold a value, a 64-bit word of the
// validity bitmap at a time: at 90% nulls nearly every run of values is one
// row long, and finding a run costs more than a row. Passing a column of
// 30,000 distinct values, 30% null, through beside a scaled f64 column
// (1,000,000 rows in 16 row groups, Apple M4, one thread) took 1.066 times
// as long as with no check when the table counted every row, 1.022 times
// when the order was checked a run at a time, and 1.007 times a word at a
// time, where the same build timed against itself gave 1.001 (CPU time,
// medians of 11 interleaved pairs).
bool uses_every_entry(const arrow::Array& column, const int32_t* indices, int64_t entries) {
  const int64_t rows = column.length();
  int64_t high = -1;  // the largest index so far
  bool in_order = true;
  const auto settles = [&](int64_t row) {
    in_order = indices[row] <= high + 1;
    high = std::max<int64_t>(high, indices[row]);
    return !in_order || high == entries - 1;
  };
  bool settled = false;
  const uint8_t* validity = column.null_bitmap_data();
  if (validity == nullptr) {
    for (int64_t row = 0; row < rows && !settled; ++row) settled = settles(row);
  } else {
    arrow::internal::BitmapUInt64Reader words(validity, column.offset(), rows);
    for (int64_t first = 0; first < rows && !settled; first += 64) {
      for (uint64_t word = words.NextWord(); word != 0 && !settled; word &= word - 1) {
        settled = settles(first + std::countr_zero(word));
      }
    }
  }
  if (in_order) return high == entries - 1;
  std::vector<bool> seen(static_cast<std::size_t>(entries));
  int64_t missing = entries;
  for (int64_t row = 0; row < rows; ++row) {
    if (column.IsNull(row)) continue;
    const int32_t index = indices[row];
    if (index < 0 || index >= entries) return true;
    const auto slot = static_cast<std::size_t>(index);
    if (!seen[slot]) {
      seen[slot] = true;
      if (--missing == 0) return true;
    }
  }
  return false;
}

// A dictionary column with more entries than its batch has rows that hold a
// value keeps only the entries those rows use, numbered in the order the rows
// first use them.
//
// The reader hands each batch of a row group the dictionary of the column
// chunk as far as it has read it. For a column whose pages are plain (the
// file's writer gave up on its dictionary, or wrote none) it builds that
// dictionary as it reads: of 1,000,000 distinct values in one row group, the
// last batch arrives with all 1,000,000 entries. A sink writes every entry
// of every batch, so without this the time and the size of the output grow
// with the square of the row group's distinct values. A file's own
// dictionary is cut the same way. Afterwards a batch holds at most one entry
// for each row that holds a value, and so no more entries than rows, the
// bound under which the kernels resolve each entry once instead of hashing
// every row.
//
// Measured on an Apple M4 with one thread: one row group of 1,000,000
// distinct 12-byte strings passes through, beside a scaled f64 column, in
// 0.25 of the time and 0.24 of the bytes it takes with every batch's
// dictionary left whole.
//
// Only a top-level column of strings or bytes with int32 indices, which are
// the reader's, is cut. A dictionary inside a list or a struct stays whole.
// So does one whose indices have another width, and one whose type says the
// order of its entries has a meaning, which the new numbering would change:
// only an array made in memory is either. A null row's index is not a value:
// it is not followed, and the row is written with index 0. When a row's
// index lies outside the dictionary the column stays as it is; the reader
// refuses such a batch before this (check_dictionary_indices).
std::shared_ptr<arrow::Array> trim_dictionary(const std::shared_ptr<arrow::Array>& column, const std::string& name,
                                              bool for_statistics = false) {
  if (column->type_id() != arrow::Type::DICTIONARY) return column;
  const auto& array = static_cast<const arrow::DictionaryArray&>(*column);
  const arrow::Array& entries = *array.dictionary();
  const arrow::ArrayData& indices = *array.indices()->data();
  const int64_t rows = column->length();
  if ((!for_statistics && entries.length() <= rows - column->null_count()) || entries.length() == 0 ||
      (for_statistics && entries.null_count() != 0) || indices.type->id() != arrow::Type::INT32 ||
      !arrow::is_base_binary_like(entries.type_id()) || array.dict_type()->ordered()) {
    return column;
  }
  const int32_t* values = indices.GetValues<int32_t>(1);
  // A column whose rows use every entry keeps its buffers and numbering.
  if (for_statistics && entries.length() <= rows - column->null_count() &&
      uses_every_entry(*column, values, entries.length())) {
    return column;
  }
  // Visits the rows that hold a value, a run of them at a time, so a batch
  // of mostly nulls costs its values, not its rows.
  const uint8_t* validity = column->null_bitmap_data();
  const auto each_value = [&](const auto& visit) {
    arrow::internal::VisitSetBitRunsVoid(validity, column->offset(), rows, [&](int64_t first, int64_t count) {
      for (int64_t row = first; row < first + count; ++row) visit(row);
    });
  };
  // The table below spans only the entries between the smallest and the
  // largest index in use, so a batch of new values costs its rows, not the
  // entries of the row group before it.
  int32_t low = std::numeric_limits<int32_t>::max();
  int32_t high = std::numeric_limits<int32_t>::min();
  each_value([&](int64_t row) {
    low = std::min(low, values[row]);
    high = std::max(high, values[row]);
  });
  const bool any = low <= high;  // false when every row is null
  if (any && (low < 0 || high >= entries.length())) return column;
  std::vector<int32_t> slot(any ? static_cast<std::size_t>(int64_t{high} - low + 1) : 0, -1);
  std::vector<int32_t> order;  // the entries in use, in the order rows first use them
  bool run = any;              // whether those are one run of the dictionary, in its order
  const std::string where = "column '" + name + "'";
  auto buffer = arrow::AllocateBuffer(rows * static_cast<int64_t>(sizeof(int32_t)));
  if (!buffer.ok()) throw Error(where + ": " + buffer.status().ToString());
  auto* trimmed = reinterpret_cast<int32_t*>((*buffer)->mutable_data());
  std::fill_n(trimmed, rows, int32_t{0});
  each_value([&](int64_t row) {
    int32_t& at = slot[static_cast<std::size_t>(values[row] - low)];
    if (at < 0) {
      at = static_cast<int32_t>(order.size());
      run = run && values[row] - low == at;
      order.push_back(values[row]);
    }
    trimmed[row] = at;
  });
  // The rows' validity, copied so that it starts at the first row whatever
  // the column's offset is.
  std::shared_ptr<arrow::Buffer> bitmap;
  if (validity != nullptr) {
    auto copied = arrow::internal::CopyBitmap(arrow::default_memory_pool(), validity, column->offset(), rows);
    if (!copied.ok()) throw Error(where + ": " + copied.status().ToString());
    bitmap = std::move(*copied);
  }
  // Rows of new values use the entries the reader added for them, one after
  // the other: such a run is taken as a slice of the dictionary, which copies
  // nothing. Any other set of entries is copied.
  std::shared_ptr<arrow::Array> kept;
  if (run) {
    kept = entries.Slice(low, static_cast<int64_t>(order.size()));
  } else if (arrow::is_large_binary_like(entries.type_id())) {
    kept = take_entries<arrow::LargeBinaryType>(entries, order, where);
  } else {
    kept = take_entries<arrow::BinaryType>(entries, order, where);
  }
  std::shared_ptr<arrow::ArrayData> data = arrow::ArrayData::Make(
      column->type(), rows, {std::move(bitmap), std::shared_ptr<arrow::Buffer>(std::move(*buffer))},
      column->null_count());
  data->dictionary = kept->data();
  return arrow::MakeArray(data);
}

bool has_nullable_dictionary(const arrow::ArrayData& data) {
  if (data.type->id() == arrow::Type::DICTIONARY) return data.GetNullCount() != 0;
  return std::any_of(data.child_data.begin(), data.child_data.end(),
                     [](const auto& child) { return has_nullable_dictionary(*child); });
}

std::shared_ptr<arrow::ArrayData> trim_children(const std::shared_ptr<arrow::ArrayData>& data,
                                                const std::string& name) {
  if (data->type->id() == arrow::Type::DICTIONARY) {
    return trim_dictionary(arrow::MakeArray(data), name, true)->data();
  }
  if (data->child_data.empty()) return data;
  auto result = data->Copy();
  for (auto& child : result->child_data) child = trim_children(child, name);
  return result;
}

// Arrow counts null among the distinct indices used for Parquet statistics.
// If one row is null, trim every chunk before unifying their dictionaries:
// an unused entry from any chunk could become the reported minimum or maximum.
std::shared_ptr<arrow::ChunkedArray> trim_for_statistics(const std::shared_ptr<arrow::ChunkedArray>& column,
                                                         const std::string& name) {
  const auto& chunks = column->chunks();
  if (!std::any_of(chunks.begin(), chunks.end(),
                   [](const auto& chunk) { return has_nullable_dictionary(*chunk->data()); })) {
    return column;
  }
  arrow::ArrayVector trimmed;
  for (const auto& chunk : chunks) {
    if (chunk->data()->child_data.empty()) {
      trimmed.push_back(trim_dictionary(chunk, name, true));
    } else {
      // A sliced list or struct can share its children with rows outside
      // this chunk. Gather only the ranges that belong to these rows.
      auto gathered = arrow::Concatenate({chunk});
      if (!gathered.ok()) throw Error("column '" + name + "': " + gathered.status().ToString());
      trimmed.push_back(arrow::MakeArray(trim_children((*gathered)->data(), name)));
    }
  }
  return std::make_shared<arrow::ChunkedArray>(std::move(trimmed), column->type());
}

// A string_view column as large_string, which every kernel and cast read.
// Parquet's reader gives string_view only for a column whose file stores that
// type in pages it cannot read as a dictionary, such as delta pages; from
// dictionary or plain pages the same column comes as a dictionary of string
// (dictionary_readable). 64-bit offsets hold the bytes of any batch of views.
// A column inside a list, struct or map keeps its views: no kernel reads one,
// and the Parquet writer stores views and strings alike.
//
// The conversion works on the buffers. One pass over the views gives every
// row its end offset, and a null row takes no bytes whatever its view holds.
// Then each row that holds a value copies its bytes. The validity bitmap
// gives those rows a run at a time, so a batch of mostly nulls skips its
// nulls without reading their views again.
//
// Measured on an Apple M4 with one thread, CPU time, medians of 11
// interleaved pairs in each of two runs: passing 1,048,576 rows of 8-byte
// strings, 90% of them null, in 16 row groups through takes 0.85 to 0.87 of
// the time it took when each value was appended to a builder, and 262,144
// rows of 20 such columns in 4 row groups 0.82 of it.
std::shared_ptr<arrow::Array> strings_from_views(const std::shared_ptr<arrow::Array>& column, const std::string& name) {
  if (column->type_id() != arrow::Type::STRING_VIEW) return column;
  const auto& views = static_cast<const arrow::StringViewArray&>(*column);
  const arrow::BinaryViewType::c_type* view = views.raw_values();
  // A view that is not inline names one of the buffers that follow the
  // validity bitmap and the views.
  const std::shared_ptr<arrow::Buffer>* buffers = views.data()->buffers.data() + 2;
  const int64_t rows = views.length();
  const uint8_t* validity = views.null_count() == 0 ? nullptr : views.null_bitmap_data();
  const std::string where = "column '" + name + "'";
  auto offsets = arrow::AllocateBuffer((rows + 1) * static_cast<int64_t>(sizeof(int64_t)));
  if (!offsets.ok()) throw Error(where + ": " + offsets.status().ToString());
  auto* ends = reinterpret_cast<int64_t*>((*offsets)->mutable_data());
  int64_t bytes = 0;
  ends[0] = 0;
  if (validity == nullptr) {
    for (int64_t row = 0; row < rows; ++row) {
      bytes += view[row].size();
      ends[row + 1] = bytes;
    }
  } else {
    for (int64_t row = 0; row < rows; ++row) {
      const bool valid = arrow::bit_util::GetBit(validity, static_cast<uint64_t>(views.offset() + row));
      bytes += valid ? view[row].size() : 0;
      ends[row + 1] = bytes;
    }
  }
  auto data = arrow::AllocateBuffer(bytes);
  if (!data.ok()) throw Error(where + ": " + data.status().ToString());
  uint8_t* out = (*data)->mutable_data();
  arrow::internal::VisitSetBitRunsVoid(validity, views.offset(), rows, [&](int64_t first, int64_t count) {
    for (int64_t row = first; row < first + count; ++row) {
      const std::string_view value = arrow::util::FromBinaryView(view[row], buffers);
      std::memcpy(out + ends[row], value.data(), value.size());
    }
  });
  // The rows' validity, copied so that it starts at the first row whatever
  // the column's offset is.
  std::shared_ptr<arrow::Buffer> bitmap;
  if (validity != nullptr) {
    auto copied = arrow::internal::CopyBitmap(arrow::default_memory_pool(), validity, views.offset(), rows);
    if (!copied.ok()) throw Error(where + ": " + copied.status().ToString());
    bitmap = std::move(*copied);
  }
  return arrow::MakeArray(
      arrow::ArrayData::Make(arrow::large_utf8(), rows,
                             {std::move(bitmap), std::shared_ptr<arrow::Buffer>(std::move(*offsets)),
                              std::shared_ptr<arrow::Buffer>(std::move(*data))},
                             views.null_count()));
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
    const std::string& name = input.column_name(i);
    batch.names.push_back(name);
    batch.cols.push_back(epoch_microseconds(trim_dictionary(strings_from_views(input.column(i), name), name), name));
  }
  return batch;
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
      // The schema above is the same either way: a column's tag names the
      // type of its strings, not whether they come as a dictionary.
      read_plain_inside_fixed_size_lists(field, false, file.properties);
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

  void read_batches(const Task& task, int64_t batch_size, const std::function<void(Batch)>& consume) override {
    if (batch_size <= 0) throw Error("batch_size must be a positive integer");
    const File& file = source_.files_[task.file];
    // One reader/file descriptor per worker, not one per input file. Dynamic
    // task assignment retains reuse for adjacent groups without unbounded FDs.
    if (!reader_ || file_ != task.file) {
      auto properties = file.properties;
      // Otherwise Arrow may prefetch the entire compressed row group, even
      // though decoded record batches are small. A WatchedFile must not be
      // pre-buffered either.
      properties.set_pre_buffer(false);
      reader_.reset();
      chunks_.assign(static_cast<std::size_t>(file.metadata->num_columns()), DictionaryChunk{});
      input_ = open_file(file.path);
      auto reader = make_reader(file.path, open_watched(file.path, input_, file.metadata, chunks_), properties);
      const auto* selected = projection_ ? &(*projection_)[task.file] : nullptr;
      for (const auto& field : reader->manifest().schema_fields) {
        watch_dictionary_leaves(field, *file.metadata->schema(), chunks_, selected, file.path, field.field->name());
      }
      reader_ = std::move(reader);
      if (projection_) {
        leaves_ = (*projection_)[task.file];
      } else {
        leaves_.resize(chunks_.size());
        std::iota(leaves_.begin(), leaves_.end(), 0);
      }
      file_ = task.file;
    }
    reader_->set_batch_size(batch_size);
    auto stream = projection_ ? reader_->GetRecordBatchReader({task.row_group}, (*projection_)[task.file])
                              : reader_->GetRecordBatchReader({task.row_group});
    const std::string where = file.path + ": failed to read row group " + std::to_string(task.row_group);
    if (!stream.ok()) throw Error(where + ": " + stream.status().ToString());
    bool emitted = false;
    int64_t rows = 0;
    for (;;) {
      std::shared_ptr<arrow::RecordBatch> batch;
      const auto status = (*stream)->ReadNext(&batch);
      if (!status.ok()) {
        throw Error(where + ": " + undecodable_column(file, task.row_group, *(*stream)->schema()) + status.ToString());
      }
      if (!batch) break;
      rows += batch->num_rows();
      std::size_t leaf = 0;
      for (int i = 0; i < batch->num_columns(); ++i) {
        const std::string& name = batch->schema()->field(i)->name();
        check_dictionaries(*batch->column_data(i), leaf, file, task.row_group, where, name);
      }
      emitted = true;
      consume(batch_from_arrow(*batch));
    }
    // Arrow's reader stops at the rows the footer states, and ends the row
    // group early when the pages of a column read hold fewer.
    const int64_t stated = file.metadata->RowGroup(task.row_group)->num_rows();
    if (rows != stated) {
      throw Error(where + ": the columns read hold " + std::to_string(rows) + " rows where the footer states " +
                  std::to_string(stated));
    }
    if (!emitted) {
      auto empty = arrow::RecordBatch::MakeEmpty((*stream)->schema());
      if (!empty.ok()) throw Error(where + ": " + empty.status().ToString());
      consume(batch_from_arrow(**empty));
    }
  }

 private:
  // Checks each leaf of `column` read as a dictionary against the dictionary
  // page of its chunk (see DictionaryChunk). A batch's arrays without
  // children are the leaves read, in the order of `leaves_`; `leaf` counts
  // those visited so far.
  void check_dictionaries(const arrow::ArrayData& column, std::size_t& leaf, const File& file, int row_group,
                          const std::string& where, const std::string& name) {
    if (!column.child_data.empty()) {
      for (const std::shared_ptr<arrow::ArrayData>& child : column.child_data) {
        check_dictionaries(*child, leaf, file, row_group, where, name);
      }
      return;
    }
    if (leaf == leaves_.size()) throw Error(where + ": column '" + name + "' holds more leaves than the file");
    const int column_index = leaves_[leaf++];
    DictionaryChunk& chunk = chunks_[static_cast<std::size_t>(column_index)];
    if (!column.dictionary) return;
    // The pages of indices that came before the chunk's first plain page
    // passed unchecked (see WatchedPages), and are bounded once, by the
    // first batch after it. An index they hold comes first in the chunk.
    if (chunk.unchecked > 0 && chunk.plain) {
      const int count = chunk.unchecked;
      chunk.unchecked = 0;
      const std::optional<int32_t> earlier =
          index_outside_before_plain(file.path, input_, file.metadata, row_group, column_index, count, chunk.entries,
                                     chunk.after_dictionary, where);
      if (earlier) chunk.outside = earlier;
    }
    // Worded as Arrow's check words it, so both checks refuse an index alike.
    if (chunk.outside) {
      throw Error(where + ": column '" + name + "' has a dictionary index outside its dictionary: Index " +
                  std::to_string(*chunk.outside) + " out of bounds");
    }
    if (chunk.entries > 0 && chunk.indexed && !chunk.settled) {
      if (!chunk.plain && column.dictionary->length == chunk.entries) {
        chunk.settled = true;
      } else if (!shows_dictionary_page(*column.dictionary, chunk)) {
        if (dictionary_page_repeats(input_, file.metadata, row_group, column_index, where)) {
          throw Error(where + ": column '" + name + "' has a dictionary page that repeats a value");
        }
        chunk.settled = true;
      }
    }
    check_dictionary_indices(column, where, name);
  }

  // Arrow's reader does not name the column of a page it cannot decode, so
  // after a failure the row group's columns are read again one at a time,
  // and the first that fails on its own is named. A failure that no column
  // repeats on its own keeps Arrow's text alone.
  std::string undecodable_column(const File& file, int row_group, const arrow::Schema& read) {
    for (const std::shared_ptr<arrow::Field>& field : read.fields()) {
      std::size_t position = 0;
      while (position < source_.schema_.cols.size() && source_.schema_.cols[position].first != field->name()) {
        ++position;
      }
      if (position == source_.schema_.cols.size()) continue;
      auto stream = reader_->GetRecordBatchReader({row_group}, file.leaves[position]);
      bool failed = !stream.ok();
      while (!failed) {
        std::shared_ptr<arrow::RecordBatch> batch;
        failed = !(*stream)->ReadNext(&batch).ok();
        if (!batch) break;
      }
      if (failed) return "column '" + field->name() + "' cannot be decoded: ";
    }
    return "";
  }

  const ParquetSource& source_;
  std::shared_ptr<const Projection> projection_;
  std::size_t file_ = 0;
  std::shared_ptr<arrow::io::RandomAccessFile> input_;
  std::vector<DictionaryChunk> chunks_;  // per leaf of the open file; reader_ notes into it
  std::vector<int> leaves_;              // the leaves read from the open file, in order
  std::unique_ptr<parquet::arrow::FileReader> reader_;
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

// Whether every dictionary index in `data` lies inside its dictionary, the
// index of a null row too, for a dictionary column and for one nested in a
// list or a struct. Arrow's dictionary unifier maps every index of a chunk
// through a table as long as the chunk's dictionary without looking at
// validity, and a null row's index is whatever its producer left there. A
// chunk whose dictionary is empty has no index to map, and the unifier
// leaves its indices as they are.
bool indices_inside_dictionaries(const arrow::ArrayData& data) {
  if (data.type->id() == arrow::Type::DICTIONARY && data.dictionary->length > 0) {
    // The same index buffer as an array without validity, so that Arrow's
    // check reads every row.
    const auto& type = static_cast<const arrow::DictionaryType&>(*data.type);
    const arrow::ArrayData every_row(type.index_type(), data.length, arrow::BufferVector{nullptr, data.buffers[1]},
                                     int64_t{0}, data.offset);
    const auto entries = static_cast<uint64_t>(data.dictionary->length);
    if (!arrow::internal::CheckIndexBounds(arrow::ArraySpan(every_row), entries).ok()) return false;
  }
  return std::all_of(
      data.child_data.begin(), data.child_data.end(),
      [](const std::shared_ptr<arrow::ArrayData>& child) { return indices_inside_dictionaries(*child); });
}

}  // namespace

// A batch of fewer than 4,096 rows waits for the ones after it. A batch holds
// fewer rows the wider the output is, and every row group repeats each
// column's page header, statistics and footer entry, so a row group per batch
// gives a very wide output hundreds of row groups that are mostly those. The
// row group closes once it holds 4,096 rows or six batches, or once one more
// batch as large as its last would bring its buffers past 36 MiB. A batch of
// 4,096 rows or more, which an output of up to 170 numeric columns has,
// therefore closes the row group it joins, and is a row group of its own when
// no batch waits. A batch that would bring the waiting rows past 65,536 goes
// after them, so no row group passes 65,536 rows unless one batch does.
//
// Six batches and 36 MiB bound what a writer holds while batches wait. The
// engine sizes a batch at 16/3 MiB by an estimate that counts a string as 32
// bytes, so a batch of longer strings is larger: with 1,000 f64 columns and
// a string of 20,000 characters in each row, a batch of 696 rows holds
// 18.7 MiB. Passing 8,192 such rows through peaks at 578 to 585 MiB when
// six batches wait, and at 436 to 453 MiB when each is written as it comes
// (Apple M4, one thread). A batch of more than 18 MiB, half the bound,
// therefore never waits. 36 MiB is six batches of 6 MiB, an eighth more than
// the estimate, which leaves validity bitmaps out: a batch of 1,000 f64
// columns read from Parquet holds 5,680,000 bytes where the estimate counts
// 5,592,000. An output whose batches hold up to an eighth more than the
// estimate keeps its row groups of six.
//
// A batch counts the whole of every buffer it refers to, because a slice
// keeps its whole buffer alive. A column read as a dictionary from plain
// pages refers to a slice of the dictionary the reader has built for the
// row group so far. With 1,000 f64 columns beside distinct strings of 1,000
// characters, in a row group of 32,768 rows, a late batch of 696 rows refers
// to 6.4 MB and keeps 38.7 MB alive. Counted whole, such a batch is a row
// group of its own: the transform's Arrow allocations peak at 706.2 MB where
// they peak at 829.5 MB counting only the bytes referred to, and the output
// takes 37 row groups instead of 10 and 1.1 to 1.2 times the time (Apple M4,
// one thread). A buffer that two waiting batches share counts twice, which
// only writes them sooner.
//
// Gathering the larger batches too, up to 65,536 rows a row group, was
// measured and dropped: a worker then held five more batches for every
// output wider than about ten columns, and at eight threads one-hot outputs
// of 100 and 300 columns took 158 -> 379 and 200 -> 522 MiB of peak memory
// (524,288 rows, Apple M4). Under this rule the same outputs take 150 and
// 225 MiB, and a one-hot output of 1,000 columns has the layout either rule
// gives it: 0.59 of the time and 0.20 of the bytes of a row group per batch.
constexpr int64_t kRowGroupMinRows = 4096;
constexpr int64_t kRowGroupRows = 65536;
constexpr std::size_t kRowGroupBatches = 6;
constexpr int64_t kRowGroupBytes = int64_t{36} << 20;

struct PartWriter::Impl {
  std::string path, temp;
  int64_t expected_rows = 0, rows = 0;
  std::shared_ptr<arrow::Schema> schema;
  std::shared_ptr<arrow::io::OutputStream> output;  // the part file behind its write buffer
  std::unique_ptr<parquet::arrow::FileWriter> writer;
  std::vector<std::shared_ptr<arrow::Table>> pending;  // the batches of the row group being collected
  int64_t pending_rows = 0, pending_bytes = 0;
  bool finished = false, abandoned = false, owns_temp = false;

  // Write the collected batches as one row group.
  void write_pending() {
    if (pending.empty()) return;
    std::shared_ptr<arrow::Table> group = pending.front();
    if (pending.size() > 1) {
      auto combined = arrow::ConcatenateTables(pending);
      if (!combined.ok()) throw Error(path + ": " + combined.status().ToString());
      group = *combined;
    }
    // The Parquet writer keeps a column's dictionary through a row group
    // only while every batch brings an equal one. When a batch brings
    // another, it writes the rest of the row group's values plain, and
    // batches do differ: a column read from plain pages comes with a
    // dictionary that grows from batch to batch. So the batches of a row
    // group are given one dictionary first. A column goes to the writer as
    // it came in three cases: a null row of it holds an index outside its
    // dictionary, which the unifier would follow; its index type is too
    // narrow to count the entries of the one dictionary; or its dictionary
    // holds a null, which the Parquet writer then refuses, as it does in a
    // row group of one batch.
    arrow::ChunkedArrayVector columns = group->columns();
    for (int i = 0; i < group->num_columns(); ++i) {
      auto& column = columns[static_cast<std::size_t>(i)];
      column = trim_for_statistics(column, group->field(i)->name());
      if (pending.size() == 1) continue;
      const arrow::ArrayVector& chunks = column->chunks();
      if (!std::all_of(chunks.begin(), chunks.end(), [](const std::shared_ptr<arrow::Array>& chunk) {
            return indices_inside_dictionaries(*chunk->data());
          })) {
        continue;
      }
      auto unified = arrow::DictionaryUnifier::UnifyChunkedArray(column);
      if (unified.ok()) column = *unified;
    }
    group = arrow::Table::Make(group->schema(), std::move(columns), pending_rows);
    const auto status = writer->WriteTable(*group, pending_rows);
    if (!status.ok()) throw Error(path + ": " + status.ToString());
    pending.clear();
    pending_rows = 0;
    pending_bytes = 0;
  }
};

PartWriter::PartWriter(const std::string& dir, std::size_t index, int64_t expected_rows, std::size_t last_index)
    : impl_(std::make_unique<Impl>()) {
  if (expected_rows < 0) throw Error(dir + ": expected row count must not be negative");
  std::string digits = std::to_string(index);
  const std::size_t width = std::max<std::size_t>(5, std::to_string(last_index).size());
  if (digits.size() < width) digits.insert(0, width - digits.size(), '0');
  impl_->path = dir + "/part-" + digits + ".parquet";
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
      if (state.pending_rows + batch.rows > kRowGroupRows) state.write_pending();
      const int64_t bytes = arrow::util::TotalBufferSize(*table);
      state.pending.push_back(std::move(table));
      state.pending_rows += batch.rows;
      state.pending_bytes += bytes;
      if (state.pending.size() == kRowGroupBatches || state.pending_rows >= kRowGroupMinRows ||
          state.pending_bytes + bytes > kRowGroupBytes) {
        state.write_pending();
      }
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
    state.write_pending();
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
  state.pending.clear();
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

namespace {

// Thread-safe errno formatting (std::strerror is not required to be
// thread-safe, and write_slice / flush_to_disk run on worker threads).
std::string errno_message(int error) { return std::error_code(error, std::generic_category()).message(); }

}  // namespace

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
    const int error = errno;
    if (directory && error == EACCES) return;
    throw Error(path + ": cannot open to flush: " + errno_message(error));
  }
  const int result = sync_to_disk(fd);
  const int error = errno;
  ::close(fd);
  if (result != 0 && !(directory && (error == EINVAL || error == ENOTSUP))) {
    throw Error(path + ": flush failed: " + errno_message(error));
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
  if (files_->fd < 0) throw Error(files_->temp + ": cannot create: " + errno_message(errno));
  files_->owns_temp = true;
  struct stat info{};
  if (::fstat(files_->fd, &info) != 0) throw Error(files_->temp + ": cannot inspect: " + errno_message(errno));
  if (!S_ISREG(info.st_mode)) throw Error(files_->temp + ": cannot preallocate: output is not a regular file");
  if (::ftruncate(files_->fd, static_cast<off_t>(rows_) * static_cast<off_t>(itemsize_)) != 0) {
    throw Error(files_->temp + ": cannot preallocate: " + errno_message(errno));
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
        throw Error(temp_path(dir_) + ": write failed: " + errno_message(errno));
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
    if (sync_to_disk(files.fd) != 0) throw Error(files.temp + ": flush failed: " + errno_message(errno));
    if (Files::close(files.fd) != 0) throw Error(files.temp + ": close failed: " + errno_message(errno));

    files.index_fd = ::open(files.index_temp.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    if (files.index_fd < 0) throw Error(files.index_temp + ": cannot write: " + errno_message(errno));
    files.owns_index_temp = true;
    std::size_t offset = 0;
    while (offset < metadata.size()) {
      const ssize_t written =
          ::write(files.index_fd, metadata.data() + offset, std::min(metadata.size() - offset, std::size_t{1} << 30));
      if (written < 0) {
        if (errno == EINTR) continue;
        throw Error(files.index_temp + ": write failed: " + errno_message(errno));
      }
      if (written == 0) throw Error(files.index_temp + ": write failed: zero-byte write");
      offset += static_cast<std::size_t>(written);
    }
    if (sync_to_disk(files.index_fd) != 0) throw Error(files.index_temp + ": flush failed: " + errno_message(errno));
    if (Files::close(files.index_fd) != 0) throw Error(files.index_temp + ": close failed: " + errno_message(errno));

    // Publish data first, then its index. The pair is not atomic: an index
    // rename failure leaves the new data in place. Release each temporary
    // immediately so later cleanup cannot unlink a subsequent writer's file.
    if (::rename(files.temp.c_str(), files.data.c_str()) != 0) {
      throw Error(files.data + ": " + errno_message(errno));
    }
    files.owns_temp = false;
    if (::rename(files.index_temp.c_str(), files.index.c_str()) != 0) {
      throw Error(files.index + ": " + errno_message(errno));
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
