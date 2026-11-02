// Explicit C++ tests for Parquet I/O: content sniffing, the write_part ->
// ParquetSource round trip, projection, empty row groups, list shapes,
// timestamps, dictionary indices, cross-file schema validation, PartWriter,
// and MemmapSink.

#include "fastfeaturetransform/io.hpp"

#include <arrow/array.h>
#include <arrow/array/builder_nested.h>
#include <arrow/io/file.h>
#include <arrow/table.h>
#include <parquet/arrow/writer.h>
#include <parquet/file_reader.h>
#include <parquet/properties.h>
#include <sys/stat.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <nlohmann/json.hpp>
#include <tuple>

#include "fastfeaturetransform/test_support.hpp"

namespace fft = fastfeaturetransform;
using Catch::Matchers::ContainsSubstring;
using fft::testing::bool_column;
using fft::testing::f64_column;
using fft::testing::make_batch;
using fft::testing::numeric_column;
using fft::testing::TempDir;
using fft::testing::utf8_column;

namespace {

std::string part(const TempDir& dir, int index) {
  return (dir.path / ("part-0000" + std::to_string(index) + ".parquet")).string();
}

std::string file_contents(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  REQUIRE(file.good());
  return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

// An uncompressed file whose string column holds "a", "b" and "c" in runs of
// 16, twice: one data page, whose dictionary indices are six run-length
// encoded runs. `runs` replaces them, to point rows outside the three-entry
// dictionary as a corrupt file would; Arrow's writer will not write such
// indices itself.
std::string dictionary_file(const TempDir& dir, const std::string& name, const std::string& runs) {
  std::vector<std::optional<std::string>> values;
  for (int copy = 0; copy < 2; ++copy) {
    for (const char* value : {"a", "b", "c"}) values.insert(values.end(), 16, value);
  }
  const auto table = arrow::Table::Make(arrow::schema({arrow::field("c", arrow::utf8())}), {utf8_column(values)});
  const std::string path = (dir.path / (name + ".parquet")).string();
  auto output = arrow::io::FileOutputStream::Open(path);
  REQUIRE(output.ok());
  parquet::WriterProperties::Builder properties;
  properties.compression(parquet::Compression::UNCOMPRESSED)
      ->disable_statistics()
      ->data_page_version(parquet::ParquetDataPageVersion::V1);
  REQUIRE(parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), *output, 1024, properties.build()).ok());
  REQUIRE((*output)->Close().ok());
  std::string bytes = file_contents(path);
  // Bit width 2, then runs of 16 (header 0x20) of the indices 0, 1, 2, 0, 1, 2.
  const std::string written("\x02\x20\x00\x20\x01\x20\x02\x20\x00\x20\x01\x20\x02", 13);
  const std::size_t at = bytes.find(written);
  REQUIRE(at != std::string::npos);
  REQUIRE(bytes.find(written, at + 1) == std::string::npos);
  REQUIRE(runs.size() <= written.size());
  bytes.replace(at, runs.size(), runs);
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  file.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  REQUIRE(file.good());
  return path;
}

}  // namespace

TEST_CASE("open_source rejects non-parquet content by magic bytes") {
  TempDir dir;
  std::string fake = (dir.path / "fake.parquet").string();
  std::ofstream(fake) << "not parquet at all";
  REQUIRE_THROWS_WITH(fft::open_source({fake}), ContainsSubstring("not a Parquet file"));
}

TEST_CASE("open_source names missing paths, directories, and empty files") {
  TempDir dir;
  REQUIRE_THROWS_WITH(fft::open_source({(dir.path / "nope.parquet").string()}), ContainsSubstring("no such file"));
  REQUIRE_THROWS_WITH(fft::open_source({dir.path.string()}), ContainsSubstring("is a directory"));
  std::string empty = (dir.path / "empty.parquet").string();
  std::ofstream(empty).close();
  REQUIRE_THROWS_WITH(fft::open_source({empty}), ContainsSubstring("file is empty"));
}

TEST_CASE("write_part then ParquetSource round-trips schema, tasks, and rows") {
  TempDir dir;
  fft::write_part(dir.path.string(), 0,
                  make_batch({"device", "price"}, {utf8_column({"mob", "desk"}), f64_column({1.5, 2.5})}));
  auto source = fft::open_source({part(dir, 0)});

  REQUIRE(source->schema().cols == fft::Schema{{{"device", "utf8"}, {"price", "f64"}}}.cols);
  REQUIRE(source->plan_tasks().size() == 1);

  fft::Batch batch = source->read_task(source->plan_tasks()[0], nullptr);
  REQUIRE(batch.rows == 2);
  // utf8 columns arrive dictionary-encoded: the reader requests Parquet's own
  // dictionaries so kernels can encode per entry instead of per row.
  std::string encodings;
  const std::unique_ptr<parquet::ParquetFileReader> raw = parquet::ParquetFileReader::OpenFile(part(dir, 0));
  const std::unique_ptr<parquet::RowGroupMetaData> row_group = raw->metadata()->RowGroup(0);
  const std::unique_ptr<parquet::ColumnChunkMetaData> chunk = row_group->ColumnChunk(0);
  for (parquet::Encoding::type encoding : chunk->encodings()) {
    encodings += std::to_string(static_cast<int>(encoding)) + " ";
  }
  INFO("device column page encodings: " << encodings);
  REQUIRE(batch.cols[0]->type_id() == arrow::Type::DICTIONARY);
  const auto& device = static_cast<const arrow::DictionaryArray&>(*batch.cols[0]);
  const auto& dictionary = static_cast<const arrow::StringArray&>(*device.dictionary());
  REQUIRE(dictionary.GetString(device.GetValueIndex(0)) == "mob");
  REQUIRE(dictionary.GetString(device.GetValueIndex(1)) == "desk");
  REQUIRE_FALSE(std::filesystem::exists(part(dir, 0) + ".tmp"));  // renamed into place
}

TEST_CASE("read_task decodes only the projected columns") {
  TempDir dir;
  fft::write_part(
      dir.path.string(), 0,
      make_batch({"device", "price", "extra"}, {utf8_column({"mob"}), f64_column({1.5}), utf8_column({"unused"})}));
  auto source = fft::open_source({part(dir, 0)});
  std::vector<std::string> columns{"price"};
  fft::Batch batch = source->read_task(source->plan_tasks()[0], &columns);
  REQUIRE(batch.names == std::vector<std::string>{"price"});
  REQUIRE(batch.rows == 1);
  std::vector<std::string> missing{"nope"};
  REQUIRE_THROWS_WITH(source->read_task(source->plan_tasks()[0], &missing),
                      ContainsSubstring("column 'nope' not found"));
}

TEST_CASE("fit readers stream projected row groups in bounded ordered batches") {
  TempDir dir;
  std::vector<std::optional<double>> numbers;
  std::vector<std::optional<std::string>> labels;
  for (int i = 0; i < 103; ++i) {
    numbers.emplace_back(i % 7 == 0 ? std::nullopt : std::optional<double>{static_cast<double>(i)});
    labels.emplace_back(i % 2 == 0 ? "a" : "b");
  }
  fft::write_part(
      dir.path.string(), 0,
      make_batch({"label", "x"}, {utf8_column(labels), fft::testing::numeric_column<arrow::DoubleType>(numbers)}));
  auto source = fft::open_source({part(dir, 0)});
  std::vector<std::string> projected{"x"};
  auto readers = source->readers(1, &projected);
  std::size_t offset = 0;
  std::size_t batches = 0;
  readers[0]->read_batches(source->plan_tasks()[0], 11, [&](fft::Batch batch) {
    REQUIRE(batch.names == projected);
    REQUIRE(batch.rows > 0);
    REQUIRE(batch.rows <= 11);
    const auto& values = static_cast<const arrow::DoubleArray&>(*batch.cols[0]);
    for (int64_t row = 0; row < batch.rows; ++row) {
      REQUIRE(values.IsNull(row) == !numbers[offset].has_value());
      if (numbers[offset]) REQUIRE(values.Value(row) == *numbers[offset]);
      ++offset;
    }
    ++batches;
  });
  REQUIRE(offset == numbers.size());
  REQUIRE(batches == 10);
  // A reader can switch back to its ordinary whole-task mode safely.
  REQUIRE(readers[0]->read(source->plan_tasks()[0]).rows == 103);
  REQUIRE_THROWS_WITH(readers[0]->read_batches(source->plan_tasks()[0], 0, [](fft::Batch) {}),
                      ContainsSubstring("batch_size"));
}

TEST_CASE("fit readers preserve empty row group schema and callback errors") {
  TempDir dir;
  fft::write_part(dir.path.string(), 0, make_batch({"x"}, {f64_column({})}));
  auto source = fft::open_source({part(dir, 0)});
  auto readers = source->readers(1, nullptr);
  unsigned calls = 0;
  readers[0]->read_batches(source->plan_tasks()[0], 3, [&](fft::Batch batch) {
    REQUIRE(batch.rows == 0);
    REQUIRE(batch.names == std::vector<std::string>{"x"});
    REQUIRE(batch.cols[0]->type_id() == arrow::Type::DOUBLE);
    ++calls;
  });
  REQUIRE(calls == 1);
  REQUIRE_THROWS_WITH(
      readers[0]->read_batches(source->plan_tasks()[0], 3, [](fft::Batch) { throw fft::Error("consumer failed"); }),
      ContainsSubstring("consumer failed"));
}

TEST_CASE("worker readers retain their projection while switching files independently") {
  TempDir dir;
  fft::write_part(
      dir.path.string(), 0,
      make_batch({"unused", "x", "y"}, {utf8_column({"a", "b"}), f64_column({1.0, 2.0}), f64_column({10.0, 20.0})}));
  fft::write_part(dir.path.string(), 1,
                  make_batch({"unused", "x", "y"}, {utf8_column({"c"}), f64_column({3.0}), f64_column({30.0})}));
  auto source = fft::open_source({part(dir, 0), part(dir, 1)});
  std::vector<std::string> projection{"y", "x", "y"};
  auto readers = source->readers(2, &projection);
  projection = {"unused"};  // a reader owns the compiled snapshot, not this caller's vector
  fft::Batch retained = readers[0]->read(source->plan_tasks()[0]);
  REQUIRE(retained.names == std::vector<std::string>{"x", "y"});
  REQUIRE(retained.rows == 2);
  for (const std::size_t worker : {0U, 1U}) {
    for (const std::size_t task : {1U, 1U, 0U, 1U, 0U}) {
      fft::Batch batch = readers[worker]->read(source->plan_tasks()[task]);
      REQUIRE(batch.names == std::vector<std::string>{"x", "y"});
      REQUIRE(batch.rows == (task == 0 ? 2 : 1));
      REQUIRE(static_cast<const arrow::DoubleArray&>(*batch.cols[0]).Value(0) == (task == 0 ? 1.0 : 3.0));
      REQUIRE(static_cast<const arrow::DoubleArray&>(*batch.cols[1]).Value(0) == (task == 0 ? 10.0 : 30.0));
    }
  }
  REQUIRE(static_cast<const arrow::DoubleArray&>(*retained.cols[0]).Value(1) == 2.0);
}

TEST_CASE("a zero-row row group reads back as an empty batch") {
  TempDir dir;
  fft::write_part(dir.path.string(), 0, make_batch({"device", "price"}, {utf8_column({}), f64_column({})}));
  auto source = fft::open_source({part(dir, 0)});
  REQUIRE(source->plan_tasks().size() == 1);
  fft::Batch batch = source->read_task(source->plan_tasks()[0], nullptr);
  REQUIRE(batch.rows == 0);
  REQUIRE(batch.names == std::vector<std::string>{"device", "price"});
  REQUIRE(batch.cols[0]->length() == 0);
}

namespace {

// A list<float> column with one list per entry; nullopt is a null row.
template <typename Builder = arrow::ListBuilder>
std::shared_ptr<arrow::Array> float_lists(const std::vector<std::optional<std::vector<float>>>& rows) {
  auto values = std::make_shared<arrow::FloatBuilder>();
  Builder builder(arrow::default_memory_pool(), values);
  for (const auto& row : rows) {
    if (!row) {
      REQUIRE(builder.AppendNull().ok());
      continue;
    }
    REQUIRE(builder.Append().ok());
    REQUIRE(values->AppendValues(*row).ok());
  }
  std::shared_ptr<arrow::Array> out;
  REQUIRE(builder.Finish(&out).ok());
  return out;
}

}  // namespace

TEST_CASE("ParquetSource reports what its footers show of each list column of plain values") {
  TempDir dir;
  // One list of one list per row: two floats a row, but no list of plain values.
  const auto offsets = numeric_column<arrow::Int32Type>({0, 1, 2});
  const auto nested = [&](float first) {
    return *arrow::ListArray::FromArrays(*offsets, *float_lists({{{first, first}}, {{first, first}}}));
  };
  fft::PartWriter writer(dir.path.string(), 0, 4);
  for (const float first : {0.0f, 4.0f}) {
    writer.append(make_batch({"emb", "ragged", "nested", "x"},
                             {float_lists({{{first, first + 1}}, {{first + 2, first + 3}}}),
                              float_lists({{{first}}, {{first, first}}}), nested(first), f64_column({first, first})}));
  }
  const std::string even = writer.finish();
  const auto source = fft::open_source({even});
  // Tags stay the Arrow types; the shapes carry the counts.
  REQUIRE(source->schema().cols == fft::Schema{{{"emb", "list<element: float>"},
                                                {"ragged", "list<element: float>"},
                                                {"nested", "list<element: list<element: float>>"},
                                                {"x", "f64"}}}
                                       .cols);
  const fft::ListShape* emb = source->list_shape("emb");
  REQUIRE(emb != nullptr);
  REQUIRE(std::tuple{emb->element, emb->average, emb->width} == std::tuple{std::string("f32"), 2.0, int64_t{2}});
  const fft::ListShape* ragged = source->list_shape("ragged");
  REQUIRE(ragged != nullptr);
  REQUIRE(std::tuple{ragged->average, ragged->width} == std::tuple{1.5, int64_t{0}});
  REQUIRE(source->list_shape("nested") == nullptr);
  REQUIRE(source->list_shape("x") == nullptr);

  // Across files whose vectors differ in length, the average pools every row
  // (11 values over 5 rows) and the width is 0.
  const std::string wider = fft::write_part(
      dir.path.string(), 1,
      make_batch({"emb", "ragged", "nested", "x"},
                 {float_lists({{{1, 2, 3}}}), float_lists({{{1}}}),
                  *arrow::ListArray::FromArrays(*numeric_column<arrow::Int32Type>({0, 1}), *float_lists({{{1, 1}}})),
                  f64_column({1})}));
  const auto both = fft::open_source({even, wider});
  REQUIRE(std::tuple{both->list_shape("emb")->average, both->list_shape("emb")->width} ==
          std::tuple{11.0 / 5, int64_t{0}});
}

TEST_CASE("MemmapSink writes plain lists whose rows hold the field's size, and refuses any other list") {
  TempDir dir;
  const fft::Schema schema{{{"v", "f32[2]"}}};
  {
    fft::MemmapSink sink(dir.path.string(), schema, {3});
    // A slice, so its rows start past the child's first values.
    sink.write_slice(0, 0, make_batch({"v"}, {float_lists({{{9, 9}}, {{1, 2}}, {{3, 4}}})->Slice(1, 2)}));
    sink.write_slice(0, 2, make_batch({"v"}, {float_lists<arrow::LargeListBuilder>({{{5, 6}}})}));
    sink.finish();
  }
  std::vector<float> written(6);
  std::ifstream(fft::MemmapSink::data_path(dir.path.string()), std::ios::binary)
      .read(reinterpret_cast<char*>(written.data()), 6 * sizeof(float));
  REQUIRE(written == std::vector<float>{1, 2, 3, 4, 5, 6});

  fft::MemmapSink sink(dir.path.string(), schema, {2});
  REQUIRE_THROWS_WITH(sink.write(0, make_batch({"v"}, {float_lists({{{1, 2}}, {{3}}})})),
                      ContainsSubstring("memmap: column 'v': row 1 holds a list of 1, and a vector field needs 2"));
  REQUIRE_THROWS_WITH(sink.write(0, make_batch({"v"}, {float_lists({{{1, 2}}, std::nullopt})})),
                      ContainsSubstring("memmap: column 'v' has a null at row 1"));
  const auto doubles =
      *arrow::ListArray::FromArrays(*numeric_column<arrow::Int32Type>({0, 2, 4}), *f64_column({1, 2, 3, 4}));
  REQUIRE_THROWS_WITH(sink.write(0, make_batch({"v"}, {doubles})),
                      ContainsSubstring("is list<item: double>, expected f32[2]"));
  sink.abandon();
  REQUIRE_THROWS_WITH((fft::MemmapSink(dir.path.string(), fft::Schema{{{"v", "list<element: float>"}}}, {1})),
                      ContainsSubstring("the same number in every row, with no null or empty row"));
  REQUIRE_THROWS_WITH((fft::MemmapSink(dir.path.string(), fft::Schema{{{"v", "list<element: float>"}}}, {0, 0})),
                      ContainsSubstring("the input has no rows to show how many values its vectors hold"));
  // A vector of strings is not a column an encoder takes, so the message does
  // not suggest encoding it.
  REQUIRE_THROWS_WITH((fft::MemmapSink(dir.path.string(), fft::Schema{{{"v", "utf8[2]"}}}, {1})),
                      ContainsSubstring("memmap: column 'v' is utf8[2], which cannot be written to a memmap"));
}

namespace {

std::vector<std::optional<int64_t>> int64_values(const arrow::Array& column) {
  REQUIRE(column.type_id() == arrow::Type::INT64);
  const auto& values = static_cast<const arrow::Int64Array&>(column);
  std::vector<std::optional<int64_t>> out;
  for (int64_t row = 0; row < values.length(); ++row) {
    out.push_back(values.IsNull(row) ? std::nullopt : std::optional<int64_t>(values.Value(row)));
  }
  return out;
}

}  // namespace

TEST_CASE("timestamp columns enter the engine as int64 epoch microseconds") {
  using arrow::TimeUnit;
  using fft::testing::timestamp_column;
  constexpr int64_t kMax = std::numeric_limits<int64_t>::max();
  const auto list = arrow::fixed_size_list(arrow::timestamp(TimeUnit::MILLI), 2);
  const auto schema = arrow::schema({arrow::field("s", arrow::timestamp(TimeUnit::SECOND)),
                                     arrow::field("ns", arrow::timestamp(TimeUnit::NANO, "America/New_York")),
                                     arrow::field("d", arrow::date32()), arrow::field("pair", list)});
  REQUIRE(fft::schema_from_arrow(*schema, "test").cols ==
          fft::Schema{{{"s", "i64"}, {"ns", "i64"}, {"d", "date32[day]"}, {"pair", "timestamp[ms][2]"}}}.cols);

  const auto us = timestamp_column(TimeUnit::MICRO, {5, std::nullopt, -7});
  const auto input = arrow::RecordBatch::Make(
      arrow::schema({arrow::field("s", arrow::timestamp(TimeUnit::SECOND)),
                     arrow::field("ms", arrow::timestamp(TimeUnit::MILLI, "UTC")),
                     arrow::field("us", arrow::timestamp(TimeUnit::MICRO)),
                     arrow::field("ns", arrow::timestamp(TimeUnit::NANO))}),
      3,
      {timestamp_column(TimeUnit::SECOND, {1, -1, std::nullopt}), timestamp_column(TimeUnit::MILLI, {1, -1, 0}, "UTC"),
       us, timestamp_column(TimeUnit::NANO, {1'500, -1'500, -1})});
  const fft::Batch batch = fft::batch_from_arrow(*input);
  REQUIRE(int64_values(*batch.cols[0]) == std::vector<std::optional<int64_t>>{1'000'000, -1'000'000, std::nullopt});
  REQUIRE(int64_values(*batch.cols[1]) == std::vector<std::optional<int64_t>>{1'000, -1'000, 0});
  REQUIRE(int64_values(*batch.cols[2]) == std::vector<std::optional<int64_t>>{5, std::nullopt, -7});
  REQUIRE(batch.cols[2]->data()->buffers[1] == us->data()->buffers[1]);  // the same buffer, viewed as int64
  REQUIRE(int64_values(*batch.cols[3]) == std::vector<std::optional<int64_t>>{1, -2, -1});  // floored

  // A slice keeps its offset, in the shared buffer and in a rescaled one.
  const fft::Batch sliced = fft::batch_from_arrow(*input->Slice(1, 2));
  REQUIRE(int64_values(*sliced.cols[0]) == std::vector<std::optional<int64_t>>{-1'000'000, std::nullopt});
  REQUIRE(int64_values(*sliced.cols[2]) == std::vector<std::optional<int64_t>>{std::nullopt, -7});
  REQUIRE(int64_values(*sliced.cols[3]) == std::vector<std::optional<int64_t>>{-2, -1});

  // A dictionary of timestamps decodes to plain int64; a null index and a
  // null dictionary entry both read as null.
  const auto dictionary =
      *arrow::DictionaryArray::FromArrays(arrow::dictionary(arrow::int32(), arrow::timestamp(TimeUnit::MILLI)),
                                          numeric_column<arrow::Int32Type>({1, std::nullopt, 0, 2, 1}),
                                          timestamp_column(TimeUnit::MILLI, {3, -4, std::nullopt}));
  const auto decoded = fft::batch_from_arrow(
      *arrow::RecordBatch::Make(arrow::schema({arrow::field("t", dictionary->type())}), 5, {dictionary}));
  REQUIRE(int64_values(*decoded.cols[0]) ==
          std::vector<std::optional<int64_t>>{-4'000, std::nullopt, 3'000, std::nullopt, -4'000});
  const auto dictionary_batch =
      arrow::RecordBatch::Make(arrow::schema({arrow::field("t", dictionary->type())}), 5, {dictionary});
  REQUIRE(int64_values(*fft::batch_from_arrow(*dictionary_batch->Slice(1, 3)).cols[0]) ==
          std::vector<std::optional<int64_t>>{std::nullopt, 3'000, std::nullopt});
  // Only the entries that rows use are rescaled: an unused one beyond the range is no error.
  const auto far_entries = timestamp_column(TimeUnit::SECOND, {0, kMax});
  const auto unused =
      *arrow::DictionaryArray::FromArrays(arrow::dictionary(arrow::int32(), arrow::timestamp(TimeUnit::SECOND)),
                                          numeric_column<arrow::Int32Type>({0, 0}), far_entries);
  REQUIRE(int64_values(*fft::batch_from_arrow(
                            *arrow::RecordBatch::Make(arrow::schema({arrow::field("t", unused->type())}), 2, {unused}))
                            .cols[0]) == std::vector<std::optional<int64_t>>{0, 0});
  const auto used =
      *arrow::DictionaryArray::FromArrays(arrow::dictionary(arrow::int32(), arrow::timestamp(TimeUnit::SECOND)),
                                          numeric_column<arrow::Int32Type>({0, 1}), far_entries);
  REQUIRE_THROWS_WITH(
      fft::batch_from_arrow(*arrow::RecordBatch::Make(arrow::schema({arrow::field("t", used->type())}), 2, {used})),
      ContainsSubstring("outside the range"));

  // A list of timestamps is left as it is.
  arrow::FixedSizeListBuilder pairs(
      arrow::default_memory_pool(),
      std::make_shared<arrow::TimestampBuilder>(arrow::timestamp(TimeUnit::MILLI), arrow::default_memory_pool()), 2);
  REQUIRE(pairs.Append().ok());
  REQUIRE(static_cast<arrow::TimestampBuilder*>(pairs.value_builder())->AppendValues(std::vector<int64_t>{1, 2}).ok());
  const auto listed = fft::batch_from_arrow(
      *arrow::RecordBatch::Make(arrow::schema({arrow::field("pair", list)}), 1, {*pairs.Finish()}));
  REQUIRE(listed.cols[0]->type()->Equals(*list));

  // Seconds and milliseconds outside the microsecond range are refused, not wrapped.
  for (const auto& [unit, scale] : {std::pair{TimeUnit::SECOND, 1'000'000}, std::pair{TimeUnit::MILLI, 1'000}}) {
    for (const int64_t value : {kMax / scale + 1, -(kMax / scale) - 1}) {
      const auto far = arrow::RecordBatch::Make(arrow::schema({arrow::field("far", arrow::timestamp(unit))}), 1,
                                                {timestamp_column(unit, {value})});
      REQUIRE_THROWS_WITH(fft::batch_from_arrow(*far),
                          ContainsSubstring("column 'far' holds") && ContainsSubstring("outside the range"));
    }
    const auto edge = arrow::RecordBatch::Make(arrow::schema({arrow::field("edge", arrow::timestamp(unit))}), 2,
                                               {timestamp_column(unit, {kMax / scale, -(kMax / scale)})});
    REQUIRE(int64_values(*fft::batch_from_arrow(*edge).cols[0]) ==
            std::vector<std::optional<int64_t>>{kMax / scale * scale, -(kMax / scale) * scale});
  }
}

TEST_CASE("files must agree on whether a column holds integers, naive timestamps or zoned ones") {
  using arrow::TimeUnit;
  using fft::testing::timestamp_column;
  TempDir dir;
  const auto write = [&](const std::string& name, const std::shared_ptr<arrow::Array>& column) {
    const auto table = arrow::Table::Make(arrow::schema({arrow::field("t", column->type())}), {column});
    const std::string path = (dir.path / name).string();
    auto output = arrow::io::FileOutputStream::Open(path);
    REQUIRE(output.ok());
    REQUIRE(parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), *output, 1024).ok());
    REQUIRE((*output)->Close().ok());
    return path;
  };
  const std::string epochs = write("epochs.parquet", numeric_column<arrow::Int64Type>({1'700'000'000}));
  const std::string seconds = write("seconds.parquet", timestamp_column(TimeUnit::SECOND, {1'700'000'000}));
  const std::string micros = write("micros.parquet", timestamp_column(TimeUnit::MICRO, {5}));
  const std::string utc = write("utc.parquet", timestamp_column(TimeUnit::MICRO, {5}, "UTC"));
  const std::string york = write("york.parquet", timestamp_column(TimeUnit::MILLI, {5}, "America/New_York"));
  REQUIRE_THROWS_WITH(fft::open_source({epochs, seconds}),
                      ContainsSubstring("column 't' is a timestamp without a time zone but") &&
                          ContainsSubstring("epochs.parquet has i64"));
  REQUIRE_THROWS_WITH(fft::open_source({micros, utc}), ContainsSubstring("is a timestamp with a time zone but") &&
                                                           ContainsSubstring("has a timestamp without a time zone"));
  // The unit and the zone may differ: each unit rescales to microseconds, and
  // a zoned column holds UTC instants whatever its zone.
  REQUIRE(fft::open_source({seconds, micros})->schema().cols == fft::Schema{{{"t", "i64"}}}.cols);
  REQUIRE(fft::open_source({utc, york})->schema().cols == fft::Schema{{{"t", "i64"}}}.cols);
}

TEST_CASE("Parquet timestamp columns read as int64 microseconds in both reading paths") {
  using arrow::TimeUnit;
  using fft::testing::timestamp_column;
  TempDir dir;
  const auto table = arrow::Table::Make(arrow::schema({arrow::field("ms", arrow::timestamp(TimeUnit::MILLI)),
                                                       arrow::field("ns", arrow::timestamp(TimeUnit::NANO, "UTC"))}),
                                        {timestamp_column(TimeUnit::MILLI, {2, std::nullopt, -3}),
                                         timestamp_column(TimeUnit::NANO, {999, -999, 1'000}, "UTC")});
  const std::string path = (dir.path / "stamps.parquet").string();
  {
    auto output = arrow::io::FileOutputStream::Open(path);
    REQUIRE(output.ok());
    const arrow::Status written = parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), *output, 1024);
    INFO(written.ToString());
    REQUIRE(written.ok());
    REQUIRE((*output)->Close().ok());
  }
  auto source = fft::open_source({path});
  REQUIRE(source->schema().cols == fft::Schema{{{"ms", "i64"}, {"ns", "i64"}}}.cols);
  const std::vector<std::optional<int64_t>> ms{2'000, std::nullopt, -3'000};
  const std::vector<std::optional<int64_t>> ns{0, -1, 1};
  const fft::Batch whole = source->read_task(source->plan_tasks()[0], nullptr);
  REQUIRE(int64_values(*whole.cols[0]) == ms);
  REQUIRE(int64_values(*whole.cols[1]) == ns);
  auto readers = source->readers(1, nullptr);
  std::vector<std::optional<int64_t>> streamed_ms;
  std::vector<std::optional<int64_t>> streamed_ns;
  readers[0]->read_batches(source->plan_tasks()[0], 2, [&](fft::Batch batch) {
    for (const auto& value : int64_values(*batch.cols[0])) streamed_ms.push_back(value);
    for (const auto& value : int64_values(*batch.cols[1])) streamed_ns.push_back(value);
  });
  REQUIRE(streamed_ms == ms);
  REQUIRE(streamed_ns == ns);

  // An empty row group reads back as an empty int64 column too, in both paths.
  fft::write_part(dir.path.string(), 0, make_batch({"t"}, {timestamp_column(TimeUnit::MICRO, {})}));
  auto empty = fft::open_source({part(dir, 0)});
  REQUIRE(empty->read_task(empty->plan_tasks()[0], nullptr).cols[0]->type_id() == arrow::Type::INT64);
  unsigned calls = 0;
  empty->readers(1, nullptr)[0]->read_batches(empty->plan_tasks()[0], 2, [&](fft::Batch batch) {
    REQUIRE(batch.cols[0]->type_id() == arrow::Type::INT64);
    ++calls;
  });
  REQUIRE(calls == 1);
}

TEST_CASE("dictionary indices outside their dictionary are refused in both reading paths") {
  TempDir dir;
  const std::string clean = dictionary_file(dir, "clean", "");
  auto source = fft::open_source({clean});
  const fft::Batch read = source->read_task(source->plan_tasks()[0], nullptr);
  REQUIRE(read.cols[0]->type_id() == arrow::Type::DICTIONARY);
  const auto& column = static_cast<const arrow::DictionaryArray&>(*read.cols[0]);
  REQUIRE(column.dictionary()->length() == 3);
  REQUIRE(column.GetValueIndex(95) == 2);
  // From row 16 on, index 3 (one past the end) at bit width 2, or 0xffffffff
  // (-1 as int32) at bit width 32: a run of 16 zeros, then a run of 80.
  const std::vector<std::pair<std::string, std::string>> corrupt{
      {"past", std::string("\x02\x20\x00\xa0\x01\x03", 6)},
      {"negative", std::string("\x20\x20\x00\x00\x00\x00\xa0\x01\xff\xff\xff\xff", 12)}};
  for (const auto& [name, runs] : corrupt) {
    CAPTURE(name);
    const std::string path = dictionary_file(dir, name, runs);
    auto corrupt_source = fft::open_source({path});
    const fft::Task task = corrupt_source->plan_tasks()[0];
    const std::string message = path + ": failed to read row group 0: column 'c' has a dictionary index outside";
    REQUIRE_THROWS_WITH(corrupt_source->read_task(task, nullptr), ContainsSubstring(message));
    auto readers = corrupt_source->readers(1, nullptr);
    std::size_t batches = 0;
    REQUIRE_THROWS_WITH(readers[0]->read_batches(task, 8, [&](fft::Batch) { ++batches; }), ContainsSubstring(message));
    REQUIRE(batches == 2);  // rows 0 to 15 are whole, row 16 is not
  }
}

TEST_CASE("write_part refuses rows without columns") {
  TempDir dir;
  fft::Batch batch;
  batch.rows = 3;
  REQUIRE_THROWS_WITH(fft::write_part(dir.path.string(), 0, batch), ContainsSubstring("no columns"));
}

TEST_CASE("PartWriter streams multiple batches into one ordered part without retaining input arrays") {
  TempDir dir;
  fft::PartWriter writer(dir.path.string(), 7, 5);
  const auto output = (dir.path / "part-00007.parquet").string();
  std::weak_ptr<arrow::Array> released;
  {
    auto values = f64_column({90.0, 1.0, 2.0, 91.0})->Slice(1, 2);
    released = values;
    writer.append(make_batch({"s", "x"}, {utf8_column({"a", "b"}), values}));
  }
  REQUIRE(released.expired());
  REQUIRE_FALSE(std::filesystem::exists(output));
  REQUIRE(std::filesystem::exists(output + ".tmp"));
  writer.append(make_batch({"s", "x"}, {utf8_column({}), f64_column({})}));
  writer.append(make_batch({"s", "x"}, {utf8_column({"c", std::nullopt, "e"}), f64_column({3.0, 4.0, 5.0})}));
  REQUIRE(writer.finish() == output);
  REQUIRE(writer.finish() == output);
  REQUIRE_FALSE(std::filesystem::exists(output + ".tmp"));
  auto source = fft::open_source({output});
  REQUIRE(source->schema().cols == fft::Schema{{{"s", "utf8"}, {"x", "f64"}}}.cols);
  REQUIRE(source->task_rows() == std::vector<int64_t>{2, 3});
  std::vector<double> actual;
  for (const auto& task : source->plan_tasks()) {
    auto batch = source->read_task(task, nullptr);
    const auto& values = static_cast<const arrow::DoubleArray&>(*batch.cols[1]);
    for (int64_t row = 0; row < batch.rows; ++row) actual.push_back(values.Value(row));
  }
  REQUIRE(actual == std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0});
  auto raw = parquet::ParquetFileReader::OpenFile(output);
  for (int group = 0; group < raw->metadata()->num_row_groups(); ++group) {
    for (int column = 0; column < raw->metadata()->num_columns(); ++column) {
      REQUIRE(raw->metadata()->RowGroup(group)->ColumnChunk(column)->compression() == parquet::Compression::SNAPPY);
    }
  }
  REQUIRE_THROWS_WITH(writer.append(make_batch({"s", "x"}, {utf8_column({}), f64_column({})})),
                      ContainsSubstring("no longer active"));
}

TEST_CASE("PartWriter creates one typed empty row group from empty batches") {
  TempDir dir;
  fft::PartWriter writer(dir.path.string(), 0, 0);
  const auto batch = make_batch({"s", "x"}, {utf8_column({}), f64_column({})});
  writer.append(batch);
  writer.append(batch);
  const auto output = writer.finish();
  auto source = fft::open_source({output});
  REQUIRE(source->task_rows() == std::vector<int64_t>{0});
  auto empty = source->read_task(source->plan_tasks().front(), nullptr);
  REQUIRE(empty.names == batch.names);
  REQUIRE(empty.rows == 0);
  REQUIRE(source->schema().cols == fft::Schema{{{"s", "utf8"}, {"x", "f64"}}}.cols);
}

TEST_CASE("PartWriter accepts changing dictionary contents while preserving the column schema") {
  TempDir dir;
  fft::PartWriter writer(dir.path.string(), 0, 4);
  for (const std::vector<std::optional<std::string>>& entries :
       {std::vector<std::optional<std::string>>{"b", "a"}, {"c", "a"}}) {
    auto indices = numeric_column<arrow::Int32Type>({0, 1});
    auto dictionary = *arrow::DictionaryArray::FromArrays(arrow::dictionary(arrow::int32(), arrow::utf8()), indices,
                                                          utf8_column(entries));
    writer.append(make_batch({"s"}, {dictionary}));
  }
  auto source = fft::open_source({writer.finish()});
  REQUIRE(source->schema().cols == fft::Schema{{{"s", "utf8"}}}.cols);
  std::vector<std::string> actual;
  for (const auto& task : source->plan_tasks()) {
    auto batch = source->read_task(task, nullptr);
    const auto& column = static_cast<const arrow::DictionaryArray&>(*batch.cols[0]);
    const auto& entries = static_cast<const arrow::StringArray&>(*column.dictionary());
    for (int64_t row = 0; row < column.length(); ++row) actual.push_back(entries.GetString(column.GetValueIndex(row)));
  }
  REQUIRE(actual == std::vector<std::string>{"b", "a", "c", "a"});
}

TEST_CASE("PartWriter rejects schema or row mismatches and removes incomplete output") {
  TempDir dir;
  const auto previous = fft::write_part(dir.path.string(), 0, make_batch({"x"}, {f64_column({7.0})}));
  const auto previous_size = std::filesystem::file_size(previous);
  for (int failure = 0; failure < 6; ++failure) {
    fft::PartWriter writer(dir.path.string(), 0, 3);
    writer.append(make_batch({"x"}, {f64_column({1.0})}));
    switch (failure) {
      case 0:
        REQUIRE_THROWS_WITH(writer.finish(), ContainsSubstring("produced 1 rows, expected 3"));
        break;
      case 1:
        REQUIRE_THROWS_WITH(writer.append(make_batch({"x"}, {f64_column({2.0, 3.0, 4.0})})),
                            ContainsSubstring("exceeds expected total"));
        break;
      case 2:
        REQUIRE_THROWS_WITH(writer.append(make_batch({"x"}, {numeric_column<arrow::Int64Type>({2})})),
                            ContainsSubstring("schema differs"));
        break;
      case 3:
        REQUIRE_THROWS_WITH(writer.append(make_batch({"renamed"}, {f64_column({2.0})})),
                            ContainsSubstring("schema differs"));
        break;
      case 4: {
        auto malformed = make_batch({"x"}, {f64_column({2.0})});
        malformed.rows = 2;
        REQUIRE_THROWS_WITH(writer.append(malformed), ContainsSubstring("length differs"));
        break;
      }
      case 5: {
        auto malformed = make_batch({"x"}, {f64_column({2.0})});
        malformed.cols.clear();
        REQUIRE_THROWS_WITH(writer.append(malformed), ContainsSubstring("names and arrays differ"));
        break;
      }
    }
    REQUIRE_FALSE(std::filesystem::exists(previous + ".tmp"));
    REQUIRE(std::filesystem::file_size(previous) == previous_size);
    auto source = fft::open_source({previous});
    auto original = source->read_task(source->plan_tasks().front(), nullptr);
    REQUIRE(static_cast<const arrow::DoubleArray&>(*original.cols[0]).Value(0) == 7.0);
    REQUIRE_THROWS_WITH(writer.finish(), ContainsSubstring("no longer active"));
  }
  {
    fft::PartWriter writer(dir.path.string(), 1, 2);
    writer.append(make_batch({"x"}, {f64_column({1.0})}));
  }
  REQUIRE_FALSE(std::filesystem::exists(part(dir, 1)));
  REQUIRE_FALSE(std::filesystem::exists(part(dir, 1) + ".tmp"));
}

TEST_CASE("PartWriter removes temporary output when publication or initialization fails") {
  TempDir dir;
  std::filesystem::create_directory(part(dir, 0));
  fft::PartWriter writer(dir.path.string(), 0, 1);
  writer.append(make_batch({"x"}, {f64_column({1.0})}));
  REQUIRE_THROWS_AS(writer.finish(), fft::Error);
  REQUIRE(std::filesystem::is_directory(part(dir, 0)));
  REQUIRE_FALSE(std::filesystem::exists(part(dir, 0) + ".tmp"));
  fft::PartWriter missing((dir.path / "missing").string(), 0, 1);
  REQUIRE_THROWS_WITH(missing.append(make_batch({"x"}, {f64_column({1.0})})), ContainsSubstring("missing"));
  fft::PartWriter no_schema(dir.path.string(), 1, 0);
  REQUIRE_THROWS_WITH(no_schema.finish(), ContainsSubstring("no batches supplied"));
  REQUIRE_FALSE(std::filesystem::exists(part(dir, 1) + ".tmp"));
}

TEST_CASE("MemmapSink lays out packed records and writes tasks at their offsets in any order") {
  TempDir dir;
  const fft::Schema schema{{{"a", "i32"}, {"b", "f64"}, {"c", "bool"}}};
  // A 16-byte chunk holds one 13-byte record, so the two-row task is written
  // in two pieces.
  fft::MemmapSink sink(dir.path.string(), schema, {2, 1}, 16);
  REQUIRE(std::filesystem::exists(fft::MemmapSink::temp_path(dir.path.string())));
  REQUIRE_FALSE(std::filesystem::exists(fft::MemmapSink::data_path(dir.path.string())));
  // Task 1 first: the offset comes from the plan, not from arrival order.
  sink.write(
      1, make_batch({"a", "b", "c"}, {numeric_column<arrow::Int32Type>({3}), f64_column({0.5}), bool_column({true})}));
  sink.write(
      0, make_batch({"c", "b", "a"},  // column order in the batch does not matter
                    {bool_column({false, true}), f64_column({1.5, 2.5}), numeric_column<arrow::Int32Type>({1, 2})}));
  const std::string index_path = sink.finish();

  const std::string data = fft::MemmapSink::data_path(dir.path.string());
  REQUIRE_FALSE(std::filesystem::exists(fft::MemmapSink::temp_path(dir.path.string())));
  REQUIRE(std::filesystem::file_size(data) == 3 * 13);
  std::ifstream file(data, std::ios::binary);
  std::vector<char> bytes(3 * 13);
  file.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  auto record = [&](std::size_t row) {
    int32_t a = 0;
    double b = 0;
    std::memcpy(&a, bytes.data() + row * 13, 4);
    std::memcpy(&b, bytes.data() + row * 13 + 4, 8);
    return std::tuple{a, b, bytes[row * 13 + 12]};
  };
  REQUIRE(record(0) == std::tuple{int32_t{1}, 1.5, char{0}});
  REQUIRE(record(1) == std::tuple{int32_t{2}, 2.5, char{1}});
  REQUIRE(record(2) == std::tuple{int32_t{3}, 0.5, char{1}});

  nlohmann::json index = nlohmann::json::parse(std::ifstream(index_path));
  REQUIRE(index.at("rows") == 3);
  REQUIRE(index.at("itemsize") == 13);
  REQUIRE(index.at("columns").at(1) ==
          nlohmann::json({{"name", "b"}, {"dtype", "float64"}, {"size", 1}, {"offset", 4}}));
}

TEST_CASE("MemmapSink refuses strings, nulls, and mismatched types, and a failed run keeps the previous output") {
  TempDir dir;
  const std::string data = fft::MemmapSink::data_path(dir.path.string());
  const std::string temp = fft::MemmapSink::temp_path(dir.path.string());
  REQUIRE_THROWS_WITH((fft::MemmapSink(dir.path.string(), fft::Schema{{{"s", "utf8"}}}, {1})),
                      ContainsSubstring("cast or encode strings"));
  REQUIRE_THROWS_WITH((fft::MemmapSink(dir.path.string(), fft::Schema{{{"t", "date32[day]"}}}, {1})),
                      ContainsSubstring("cannot be written to a memmap"));
  REQUIRE_THROWS_WITH((fft::MemmapSink(dir.path.string(), fft::Schema{{{"e", "f32[0]"}}}, {1})),
                      ContainsSubstring("cannot be written to a memmap"));
  REQUIRE(std::filesystem::is_empty(dir.path));

  {
    fft::MemmapSink first(dir.path.string(), fft::Schema{{{"a", "i32"}}}, {1});
    first.write(0, make_batch({"a"}, {numeric_column<arrow::Int32Type>({7})}));
    first.finish();
  }
  fft::MemmapSink sink(dir.path.string(), fft::Schema{{{"a", "i32"}}}, {2, 2});
  REQUIRE(std::filesystem::exists(temp));
  REQUIRE(std::filesystem::file_size(data) == 4);  // the previous output is untouched while a run is open
  REQUIRE_THROWS_WITH(sink.write(1, make_batch({"a"}, {numeric_column<arrow::Int32Type>({1, std::nullopt})})),
                      ContainsSubstring("has a null at row 3"));
  REQUIRE_THROWS_WITH(sink.write(0, make_batch({"a"}, {f64_column({1.0, 2.0})})), ContainsSubstring("expected i32"));
  REQUIRE_THROWS_WITH(sink.write(0, make_batch({"a"}, {numeric_column<arrow::Int32Type>({1})})),
                      ContainsSubstring("produced 1 rows, expected 2"));
  sink.abandon();
  REQUIRE_FALSE(std::filesystem::exists(temp));
  REQUIRE(std::filesystem::file_size(data) == 4);
  REQUIRE(std::filesystem::exists(fft::MemmapSink::index_path(dir.path.string())));
}

TEST_CASE("MemmapSink removes unpublished output on ordinary scope exit and exception unwinding") {
  for (bool unwind : {false, true}) {
    TempDir dir;
    const auto run = [&] {
      fft::MemmapSink sink(dir.path.string(), fft::Schema{{{"a", "i32"}}}, {1});
      sink.write(0, make_batch({"a"}, {numeric_column<arrow::Int32Type>({7})}));
      REQUIRE(std::filesystem::exists(fft::MemmapSink::temp_path(dir.path.string())));
      if (unwind) throw fft::Error("caller failed after writing");
    };
    if (unwind) {
      REQUIRE_THROWS_WITH(run(), "caller failed after writing");
    } else {
      REQUIRE_NOTHROW(run());
    }
    REQUIRE(std::filesystem::is_empty(dir.path));
  }
}

TEST_CASE("MemmapSink rolls back a constructor failure after opening its temporary") {
  TempDir dir;
  const auto temp = fft::MemmapSink::temp_path(dir.path.string());
  const auto index_temp = fft::MemmapSink::index_path(dir.path.string()) + ".tmp";
  std::ofstream(index_temp) << "unowned index temporary";
  // Opening a FIFO read/write does not block, but it cannot hold a memmap.
  // Some platforms let ftruncate succeed on it, so the sink checks its kind.
  REQUIRE(::mkfifo(temp.c_str(), 0600) == 0);
  REQUIRE(std::filesystem::is_fifo(temp));
  REQUIRE_THROWS_WITH((fft::MemmapSink(dir.path.string(), fft::Schema{{{"a", "i32"}}}, {1})),
                      ContainsSubstring("cannot preallocate"));
  REQUIRE_FALSE(std::filesystem::exists(temp));
  REQUIRE(file_contents(index_temp) == "unowned index temporary");
  REQUIRE_FALSE(std::filesystem::exists(fft::MemmapSink::data_path(dir.path.string())));
  REQUIRE_FALSE(std::filesystem::exists(fft::MemmapSink::index_path(dir.path.string())));
}

TEST_CASE("MemmapSink cleans a metadata serialization failure before leaving scope") {
  TempDir dir;
  const auto data = fft::MemmapSink::data_path(dir.path.string());
  const auto index = fft::MemmapSink::index_path(dir.path.string());
  {
    fft::MemmapSink previous(dir.path.string(), fft::Schema{{{"a", "i32"}}}, {1});
    previous.write(0, make_batch({"a"}, {numeric_column<arrow::Int32Type>({7})}));
    previous.finish();
  }
  const auto previous_data = file_contents(data);
  const auto previous_index = file_contents(index);
  // JSON accepts the string as a value but rejects its encoding when dumped.
  // This exercises a non-I/O exception without changing the global allocator.
  const std::string invalid_name(1, '\xff');
  fft::MemmapSink sink(dir.path.string(), fft::Schema{{{invalid_name, "i32"}}}, {1});
  sink.write(0, make_batch({invalid_name}, {numeric_column<arrow::Int32Type>({99})}));
  REQUIRE_THROWS_AS(sink.finish(), nlohmann::json::type_error);
  REQUIRE_FALSE(std::filesystem::exists(fft::MemmapSink::temp_path(dir.path.string())));
  REQUIRE_FALSE(std::filesystem::exists(index + ".tmp"));
  REQUIRE(file_contents(data) == previous_data);
  REQUIRE(file_contents(index) == previous_index);
  REQUIRE_THROWS_WITH(sink.finish(), ContainsSubstring("no longer active"));
}

TEST_CASE("MemmapSink finished owners do not remove a later writer's temporary") {
  TempDir dir;
  const fft::Schema schema{{{"a", "i32"}}};
  const auto temp = fft::MemmapSink::temp_path(dir.path.string());
  const auto data = fft::MemmapSink::data_path(dir.path.string());
  auto finished = std::make_unique<fft::MemmapSink>(dir.path.string(), schema, std::vector<int64_t>{1});
  finished->write(0, make_batch({"a"}, {numeric_column<arrow::Int32Type>({7})}));
  const auto index = finished->finish();
  const auto first_data = file_contents(data);
  REQUIRE(finished->finish() == index);

  fft::MemmapSink later(dir.path.string(), schema, {1});
  const auto next_batch = make_batch({"a"}, {numeric_column<arrow::Int32Type>({99})});
  later.write(0, next_batch);
  const auto later_data = file_contents(temp);
  REQUIRE(finished->finish() == index);
  finished->abandon();
  REQUIRE(file_contents(temp) == later_data);
  REQUIRE(file_contents(data) == first_data);
  finished.reset();
  REQUIRE(file_contents(temp) == later_data);
  REQUIRE(file_contents(data) == first_data);
  REQUIRE(later.finish() == index);
  REQUIRE(file_contents(data) == later_data);
}

TEST_CASE("MemmapSink abandonment is final and removes only acquired temporaries") {
  TempDir dir;
  const auto temp = fft::MemmapSink::temp_path(dir.path.string());
  const auto index_temp = fft::MemmapSink::index_path(dir.path.string()) + ".tmp";
  std::ofstream(index_temp) << "unowned index temporary";
  const auto batch = make_batch({"a"}, {numeric_column<arrow::Int32Type>({7})});
  {
    fft::MemmapSink sink(dir.path.string(), fft::Schema{{{"a", "i32"}}}, {1});
    sink.abandon();
    sink.abandon();
    REQUIRE_FALSE(std::filesystem::exists(temp));
    REQUIRE(file_contents(index_temp) == "unowned index temporary");
    REQUIRE_THROWS_WITH(sink.write(0, batch), ContainsSubstring("no longer active"));
    REQUIRE_THROWS_WITH(sink.finish(), ContainsSubstring("no longer active"));
  }
  REQUIRE(file_contents(index_temp) == "unowned index temporary");
  REQUIRE_FALSE(std::filesystem::exists(fft::MemmapSink::data_path(dir.path.string())));
  REQUIRE_FALSE(std::filesystem::exists(fft::MemmapSink::index_path(dir.path.string())));
}

TEST_CASE("MemmapSink rejects invalid dimensions before acquiring output files") {
  TempDir dir;
  const auto output = (dir.path / "not-created").string();
  const int64_t maximum = std::numeric_limits<int64_t>::max();
  const fft::Schema scalar{{{"a", "i8"}}};
  REQUIRE_THROWS_WITH((fft::MemmapSink(output, scalar, {-1})), ContainsSubstring("negative"));
  REQUIRE_FALSE(std::filesystem::exists(output));
  REQUIRE_THROWS_WITH((fft::MemmapSink(output, scalar, {maximum, 1})), ContainsSubstring("row count"));
  REQUIRE_FALSE(std::filesystem::exists(output));
  REQUIRE_THROWS_WITH((fft::MemmapSink(output, fft::Schema{{{"a", "i64"}}}, {maximum / 8 + 1})),
                      ContainsSubstring("file size"));
  REQUIRE_FALSE(std::filesystem::exists(output));
  REQUIRE_THROWS_WITH((fft::MemmapSink(output, fft::Schema{{{"a", "f32[00]"}}}, {1})),
                      ContainsSubstring("cannot be written to a memmap"));
  REQUIRE_FALSE(std::filesystem::exists(output));
}

TEST_CASE("MemmapSink publication failures clean owned temporaries and preserve unowned paths") {
  for (const std::string blocked : {"data.mmap", "index.json", "index.json.tmp"}) {
    CAPTURE(blocked);
    TempDir dir;
    const auto data = fft::MemmapSink::data_path(dir.path.string());
    const auto index = fft::MemmapSink::index_path(dir.path.string());
    const auto obstruction = dir.path / blocked;
    std::filesystem::create_directory(obstruction);
    std::ofstream(obstruction / "marker") << "unowned";
    fft::MemmapSink sink(dir.path.string(), fft::Schema{{{"a", "i32"}}}, {1});
    sink.write(0, make_batch({"a"}, {numeric_column<arrow::Int32Type>({7})}));
    REQUIRE_THROWS_AS(sink.finish(), fft::Error);
    REQUIRE_FALSE(std::filesystem::exists(fft::MemmapSink::temp_path(dir.path.string())));
    REQUIRE(file_contents(obstruction / "marker") == "unowned");
    if (blocked != "index.json.tmp") REQUIRE_FALSE(std::filesystem::exists(index + ".tmp"));
    if (blocked == "index.json") {
      // Data was already published when the second rename failed.
      REQUIRE(std::filesystem::file_size(data) == sizeof(int32_t));
    } else if (blocked == "data.mmap") {
      REQUIRE_FALSE(std::filesystem::exists(index));
    } else {
      REQUIRE_FALSE(std::filesystem::exists(data));
      REQUIRE_FALSE(std::filesystem::exists(index));
    }
    REQUIRE_THROWS_WITH(sink.finish(), ContainsSubstring("no longer active"));
  }
}

TEST_CASE("MemmapSink writes task slices at fixed offsets with unchanged packed bytes") {
  TempDir slices, whole;
  const fft::Schema schema{{{"n", "i32"}, {"flag", "bool"}}};
  const auto numbers = numeric_column<arrow::Int32Type>({10, 11, 12, 13, 14});
  const auto flags = bool_column({false, true, false, true, true});
  const auto batch = [&](int64_t first, int64_t count) {
    return make_batch({"flag", "n"}, {flags->Slice(first, count), numbers->Slice(first, count)});
  };
  fft::MemmapSink sliced(slices.path.string(), schema, {3, 2, 0}, 6);
  sliced.write_slice(1, 1, batch(4, 1));
  sliced.write_slice(0, 2, batch(2, 1));
  sliced.write_slice(1, 0, batch(3, 1));
  sliced.write_slice(0, 0, batch(0, 2));
  sliced.write_slice(2, 0, batch(5, 0));
  const auto sliced_index = sliced.finish();
  fft::MemmapSink reference(whole.path.string(), schema, {3, 2, 0}, 6);
  reference.write(0, batch(0, 3));
  reference.write(1, batch(3, 2));
  reference.write(2, batch(5, 0));
  const auto whole_index = reference.finish();
  REQUIRE(nlohmann::json::parse(std::ifstream(sliced_index)) == nlohmann::json::parse(std::ifstream(whole_index)));
  std::ifstream actual_file(fft::MemmapSink::data_path(slices.path.string()), std::ios::binary);
  std::ifstream expected_file(fft::MemmapSink::data_path(whole.path.string()), std::ios::binary);
  std::vector<char> actual(25), expected(25);
  actual_file.read(actual.data(), static_cast<std::streamsize>(actual.size()));
  expected_file.read(expected.data(), static_cast<std::streamsize>(expected.size()));
  REQUIRE(actual_file.gcount() == 25);
  REQUIRE(expected_file.gcount() == 25);
  REQUIRE(actual == expected);
  REQUIRE_THROWS_WITH(sliced.write_slice(0, 0, batch(0, 1)), ContainsSubstring("no longer active"));
}

TEST_CASE("MemmapSink slice validation preserves absolute null rows and task boundaries") {
  TempDir dir;
  fft::MemmapSink sink(dir.path.string(), fft::Schema{{{"a", "i32"}}}, {2, 3});
  auto one = make_batch({"a"}, {numeric_column<arrow::Int32Type>({1})});
  REQUIRE_THROWS_WITH(sink.write_slice(2, 0, one), ContainsSubstring("task index out of range"));
  REQUIRE_THROWS_WITH(sink.write_slice(0, -1, one), ContainsSubstring("outside task"));
  REQUIRE_THROWS_WITH(sink.write_slice(0, 2, one), ContainsSubstring("outside task"));
  REQUIRE_THROWS_WITH(sink.write_slice(1, 3, one), ContainsSubstring("outside task"));
  auto short_column = one;
  short_column.rows = 2;
  REQUIRE_THROWS_WITH(sink.write_slice(1, 0, short_column), ContainsSubstring("length differs"));
  REQUIRE_THROWS_WITH(sink.write_slice(1, 1, make_batch({"a"}, {numeric_column<arrow::Int32Type>({1, std::nullopt})})),
                      ContainsSubstring("has a null at row 4"));
  REQUIRE_THROWS_WITH(sink.write_slice(1, 0, make_batch({"a"}, {f64_column({1.0})})),
                      ContainsSubstring("expected i32"));
  sink.abandon();
  REQUIRE_FALSE(std::filesystem::exists(fft::MemmapSink::temp_path(dir.path.string())));
  REQUIRE_FALSE(std::filesystem::exists(fft::MemmapSink::data_path(dir.path.string())));
  TempDir lists;
  fft::MemmapSink list_sink(lists.path.string(), fft::Schema{{{"v", "i16[2]"}}}, {2, 3});
  auto child = numeric_column<arrow::Int16Type>({99, 98, 1, 2, 3, std::nullopt})->Slice(2, 4);
  auto array = arrow::FixedSizeListArray::FromArrays(child, 2);
  REQUIRE(array.ok());
  REQUIRE_THROWS_WITH(list_sink.write_slice(1, 1, make_batch({"v"}, {*array})),
                      ContainsSubstring("has a null at row 4"));
  list_sink.abandon();
}

TEST_CASE("MemmapSink contiguous list writes respect child and parent offsets across tiny chunks") {
  TempDir dir;
  auto child = numeric_column<arrow::Int16Type>({99, 98, 1, 2, 3, 4, 5, 6, 7, 8, 97, 96})->Slice(2, 8);
  auto lists = arrow::FixedSizeListArray::FromArrays(child, 2);
  REQUIRE(lists.ok());
  auto parent = (*lists)->Slice(1, 3);
  REQUIRE(parent->offset() == 1);
  REQUIRE(static_cast<const arrow::FixedSizeListArray&>(*parent).values()->offset() == 2);
  fft::MemmapSink sink(dir.path.string(), fft::Schema{{{"vector", "i16[2]"}}}, {3}, 5);
  sink.write_slice(0, 2, make_batch({"vector"}, {parent->Slice(2, 1)}));
  sink.write_slice(0, 0, make_batch({"vector"}, {parent->Slice(0, 2)}));
  sink.finish();
  std::ifstream file(fft::MemmapSink::data_path(dir.path.string()), std::ios::binary);
  const std::vector<int16_t> expected{3, 4, 5, 6, 7, 8};
  std::vector<char> bytes(expected.size() * sizeof(int16_t));
  file.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  REQUIRE(file.gcount() == static_cast<std::streamsize>(bytes.size()));
  REQUIRE(std::memcmp(bytes.data(), expected.data(), bytes.size()) == 0);
  REQUIRE(std::filesystem::file_size(fft::MemmapSink::data_path(dir.path.string())) == bytes.size());
}

TEST_CASE("mismatched schemas across files name the difference") {
  TempDir dir;
  fft::write_part(dir.path.string(), 0, make_batch({"device", "price"}, {utf8_column({"mob"}), f64_column({1.0})}));
  fft::write_part(dir.path.string(), 1, make_batch({"device", "price"}, {f64_column({1.0}), f64_column({1.0})}));
  fft::write_part(dir.path.string(), 2, make_batch({"device"}, {utf8_column({"mob"})}));
  fft::write_part(dir.path.string(), 3, make_batch({"price", "device"}, {f64_column({1.0}), utf8_column({"mob"})}));
  fft::write_part(
      dir.path.string(), 4,
      make_batch({"device", "price", "more"}, {utf8_column({"mob"}), f64_column({1.0}), f64_column({2.0})}));

  REQUIRE_THROWS_WITH(fft::open_source({part(dir, 0), part(dir, 1)}), ContainsSubstring("'device' is f64 but"));
  REQUIRE_THROWS_WITH(fft::open_source({part(dir, 0), part(dir, 2)}), ContainsSubstring("'price' is missing"));
  REQUIRE_THROWS_WITH(fft::open_source({part(dir, 0), part(dir, 3)}), ContainsSubstring("same order"));
  REQUIRE_THROWS_WITH(fft::open_source({part(dir, 0), part(dir, 4)}), ContainsSubstring("'more' is not present"));
}

TEST_CASE("parallel metadata preparation retains ordered file tasks and row counts") {
  TempDir dir;
  std::vector<std::string> paths;
  std::vector<int64_t> rows;
  for (std::size_t file = 0; file < 20; ++file) {
    rows.push_back(static_cast<int64_t>(file % 4));
    paths.push_back(
        fft::write_part(dir.path.string(), file,
                        make_batch({"x"}, {f64_column(std::vector<double>(file % 4, static_cast<double>(file)))})));
  }
  for (const unsigned workers : {1U, 4U, 64U}) {
    auto source = fft::open_source(paths, workers);
    REQUIRE(source->task_rows() == rows);
    REQUIRE(source->plan_tasks().size() == paths.size());
    for (std::size_t index = 0; index < paths.size(); ++index) {
      const auto& task = source->plan_tasks()[index];
      REQUIRE(task.file == index);
      REQUIRE(task.row_group == 0);
      auto batch = source->read_task(task, nullptr);
      REQUIRE(batch.rows == rows[index]);
      if (batch.rows > 0)
        REQUIRE(static_cast<const arrow::DoubleArray&>(*batch.cols[0]).Value(0) == static_cast<double>(index));
    }
  }
}

TEST_CASE("parallel metadata errors preserve schema, parse, and prior magic-check precedence") {
  TempDir dir;
  std::vector<std::string> paths;
  for (std::size_t file = 0; file < 20; ++file) {
    paths.push_back(fft::write_part(dir.path.string(), file, make_batch({"x"}, {f64_column({1.0})})));
  }
  fft::write_part(dir.path.string(), 1, make_batch({"x"}, {utf8_column({"wrong-type"})}));
  std::ofstream(paths[2], std::ios::binary) << "PAR1broken-footer";
  const auto error = [&](unsigned workers) {
    try {
      (void)fft::open_source(paths, workers);
    } catch (const fft::Error& failure) {
      return std::string(failure.what());
    }
    return std::string();
  };
  const auto schema_error = error(1);
  REQUIRE_THAT(schema_error, ContainsSubstring(paths[1] + ": column 'x' is utf8 but"));
  REQUIRE(error(4) == schema_error);
  std::swap(paths[1], paths[2]);
  const auto parse_error = error(1);
  REQUIRE_THAT(parse_error, ContainsSubstring(paths[1]));
  REQUIRE(parse_error != schema_error);
  REQUIRE(error(4) == parse_error);
  std::ofstream(paths.back(), std::ios::binary) << "not parquet";
  const auto magic_error = error(1);
  REQUIRE_THAT(magic_error, ContainsSubstring(paths.back() + ": has a .parquet extension but is not a Parquet file"));
  REQUIRE(error(4) == magic_error);
}
