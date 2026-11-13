// The generic nanobind seam: Engine and FittedPipeline handles, plan JSON in.
// Adding a kernel never touches this file — the boundary speaks plans and
// state JSON, not per-transformer classes.

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
// After nanobind, which brings in Python.h.
#include <ankerl/unordered_dense.h>
#include <datetime.h>

#include <chrono>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "fastfeaturetransform/engine.hpp"

namespace nb = nanobind;
using namespace nb::literals;
using fastfeaturetransform::Engine;
using fastfeaturetransform::FittedPipeline;
using fastfeaturetransform::Row;
using fastfeaturetransform::RowValue;

namespace {

std::string python_type_name(nb::handle value) {
  return nb::cast<std::string>(nb::borrow<nb::str>(nb::getattr(value.type(), "__name__")));
}

// numpy's floating scalar types, the only non-builtin values converted
// through __float__ (anything else with a __float__, such as a numpy complex
// or a Decimal, is rejected rather than silently narrowed).
const std::unordered_set<std::string> kNumpyFloats{"float16", "float32", "float64", "longdouble", "float128"};

__extension__ typedef __int128 Int128;  // NOLINT(modernize-use-using): __extension__ needs the typedef form

// A datetime enters the engine as int64 microseconds since 1970-01-01, the
// count Arrow's timestamp[us] stores: the wall clock as if it were UTC for a
// naive value, the UTC instant for an aware one. A Python datetime holds
// whole microseconds in its fields.
int64_t epoch_microseconds(PyObject* value, const std::string& name) {
  using namespace std::chrono;
  const sys_days date{year{PyDateTime_GET_YEAR(value)} / static_cast<unsigned>(PyDateTime_GET_MONTH(value)) /
                      static_cast<unsigned>(PyDateTime_GET_DAY(value))};
  const int64_t seconds = (int64_t{PyDateTime_DATE_GET_HOUR(value)} * 60 + PyDateTime_DATE_GET_MINUTE(value)) * 60 +
                          PyDateTime_DATE_GET_SECOND(value);
  const int64_t micros = duration_cast<microseconds>(date.time_since_epoch()).count() + seconds * 1'000'000 +
                         PyDateTime_DATE_GET_MICROSECOND(value);
  if (PyDateTime_DATE_GET_TZINFO(value) == Py_None) return micros;
  nb::object offset = nb::steal(PyObject_CallMethod(value, "utcoffset", nullptr));
  if (!offset.is_valid()) throw nb::python_error();
  if (offset.is_none()) return micros;  // a tzinfo that gives no offset leaves the value naive
  if (!PyDelta_Check(offset.ptr())) {
    throw fastfeaturetransform::Error("record value for '" + name + "' has a utcoffset() that is not a timedelta");
  }
  const int64_t shift =
      (int64_t{PyDateTime_DELTA_GET_DAYS(offset.ptr())} * 86'400 + PyDateTime_DELTA_GET_SECONDS(offset.ptr())) *
          1'000'000 +
      PyDateTime_DELTA_GET_MICROSECONDS(offset.ptr());
  return micros - shift;
}

// Microseconds per unit of a numpy datetime64, as a fraction. Days and the
// coarser units are dates: a record refuses them as it refuses datetime.date,
// and pyarrow writes numpy's day arrays to Parquet as date32, which the batch
// path keeps as date32.
const std::unordered_map<std::string, std::pair<int64_t, int64_t>> kMicrosecondsPerUnit{{"h", {3'600'000'000, 1}},
                                                                                        {"m", {60'000'000, 1}},
                                                                                        {"s", {1'000'000, 1}},
                                                                                        {"ms", {1'000, 1}},
                                                                                        {"us", {1, 1}},
                                                                                        {"ns", {1, 1'000}},
                                                                                        {"ps", {1, 1'000'000}},
                                                                                        {"fs", {1, 1'000'000'000}},
                                                                                        {"as", {1, 1'000'000'000'000}}};

// A value's attribute, or an empty object when the value has none, as
// Python's hasattr tells it: an AttributeError means none, and any other
// error the lookup raises propagates. nb::hasattr instead hands such an
// error to sys.unraisablehook and answers false.
//
// The record path fetches every attribute it calls with into an object
// first, with this or nb::getattr. nanobind converts a call's arguments in a
// function that cannot throw, so a lookup passed to a call as
// value.attr(name) that fails there ends the process.
nb::object optional_attr(nb::handle value, const char* attribute) {
  PyObject* found = nullptr;
  if (PyObject_GetOptionalAttrString(value.ptr(), attribute, &found) < 0) throw nb::python_error();
  return nb::steal(found);
}

// A numpy.datetime64 (or a pandas Timestamp's asm8) from its stored count,
// which its astype reads, and its unit, which its dtype holds, rescaled
// exactly: numpy's own cast to microseconds can overflow on a value that
// fits. Finer units floor, as numpy and the batch path do; NaT, which numpy
// stores as INT64_MIN, becomes null.
std::optional<int64_t> numpy_epoch_microseconds(const nb::object& astype, const nb::object& dtype,
                                                const std::string& name) {
  const auto count = nb::cast<int64_t>(astype("int64").attr("item")());
  if (count == std::numeric_limits<int64_t>::min()) return std::nullopt;
  const nb::object unit_and_step = nb::module_::import_("numpy").attr("datetime_data")(dtype);
  const auto unit = nb::cast<std::string>(unit_and_step[0]);
  const auto step = nb::cast<int64_t>(unit_and_step[1]);
  const auto per_unit = kMicrosecondsPerUnit.find(unit);
  if (per_unit == kMicrosecondsPerUnit.end()) {
    throw fastfeaturetransform::Error("record value for '" + name + "' is a numpy datetime64 in '" + unit +
                                      "' units, which is a date rather than a time; only times are converted");
  }
  const auto [numerator, denominator] = per_unit->second;
  const Int128 scaled = Int128{count} * step * numerator;
  Int128 micros = scaled / denominator;
  if (scaled % denominator < 0) --micros;  // floor, not truncation, below the epoch
  if (micros > std::numeric_limits<int64_t>::max() || micros < std::numeric_limits<int64_t>::min()) {
    throw fastfeaturetransform::Error("record value for '" + name + "' is outside the range of int64 microseconds");
  }
  return static_cast<int64_t>(micros);
}

// A record that holds one key twice is an error: a step reads one of the two
// fields, and the output dict keeps the later one. A dict holds each key
// once, yet two of its keys can have one text: a str subclass with its own
// hash or equality beside a str. And code that runs while the record is read
// can take a key out and store it again past the field being read, where the
// walk reaches it a second time. So the names are checked only when a key is
// not an exact str or a value's conversion can run Python code: a record of
// exact str keys whose values are None, bools, ints, floats or strs,
// subclasses included, is not checked.
// Another thread that changes the record while this one waits for a str's
// lock, on a free-threaded build, can repeat a key too; that race in the
// caller's code is not checked, since checking it would cost every record
// that holds a str that is not ASCII.
void refuse_repeated_keys(const Row& row) {
  ankerl::unordered_dense::set<std::string_view> names;
  names.reserve(row.size());
  for (const auto& field : row) {
    if (!names.insert(field.first).second) {
      throw fastfeaturetransform::Error("record holds the key '" + field.first + "' twice");
    }
  }
}

// Record values: str, int, float, bool, None, datetimes (as epoch
// microseconds), plus numpy scalars (their integer and floating types convert
// through __index__/__float__). Each refusal of a value names its key.
Row row_from_dict(const nb::dict& record) {
  Row row;
  row.reserve(nb::len(record));
  bool can_repeat = false;  // whether refuse_repeated_keys must look
  for (auto [key, value] : record) {
    // A key is read with the calls nanobind's caster for std::string makes,
    // without the string the caster builds and then moves, and the test for
    // an exact str sits beside the test for a str. A record of 30 float
    // fields through one Select step took a sixth less time than through
    // nb::try_cast (an Apple M4, one thread).
    if (!PyUnicode_Check(key.ptr())) throw fastfeaturetransform::Error("record keys must be str");
    if (!PyUnicode_CheckExact(key.ptr())) can_repeat = true;
    // The record lends its keys and values, and its reference can be the
    // only one. Code that runs while a field is read can take the key or the
    // value out of the record and free it: a hook of the value or of its
    // type, or another thread, which on the standard build can run whenever
    // Python code runs and on a free-threaded build whenever this thread
    // waits, since waiting suspends the record's lock. So from the first step
    // that can run Python code or wait, the key and the value that are still
    // to be read are held. Reading None, a bool, an int, a float or an ASCII
    // str does neither. On a free-threaded build the first UTF-8 conversion
    // of a str that is not ASCII locks the str, and can wait for that lock.
    const bool key_can_wait = !PyUnicode_IS_ASCII(key.ptr());
    const nb::object held_key = key_can_wait ? nb::borrow(key) : nb::object();
    nb::object held_value = key_can_wait ? nb::borrow(value) : nb::object();
    Py_ssize_t name_size = 0;
    const char* const name_data = PyUnicode_AsUTF8AndSize(key.ptr(), &name_size);
    if (name_data == nullptr) {  // a str with a lone surrogate has no UTF-8 form
      PyErr_Clear();
      throw fastfeaturetransform::Error("record keys must be str");
    }
    std::string name(name_data, static_cast<std::size_t>(name_size));
    if (value.is_none()) {
      row.emplace_back(std::move(name), std::monostate{});
    } else if (nb::isinstance<nb::bool_>(value)) {
      row.emplace_back(std::move(name), nb::cast<bool>(value));
    } else if (nb::isinstance<nb::int_>(value)) {
      int64_t integer = 0;
      if (!nb::try_cast(value, integer)) {
        throw fastfeaturetransform::Error("record value for '" + name + "' does not fit in a 64-bit integer");
      }
      row.emplace_back(std::move(name), integer);
    } else if (nb::isinstance<nb::float_>(value)) {
      row.emplace_back(std::move(name), nb::cast<double>(value));
    } else if (nb::isinstance<nb::str>(value)) {
      if (!PyUnicode_IS_ASCII(value.ptr())) held_value = nb::borrow(value);
      std::string text;
      if (!nb::try_cast(value, text)) {
        throw fastfeaturetransform::Error("record value for '" + name + "' is not encodable as UTF-8");
      }
      row.emplace_back(std::move(name), std::move(text));
    } else {
      held_value = nb::borrow(value);
      can_repeat = true;
      const std::string type = python_type_name(value);
      const auto unsupported = [&] {
        return fastfeaturetransform::Error("record value for '" + name +
                                           "' must be str, int, float, bool, datetime, or None (got " + type + ")");
      };
      double real = 0.0;
      if (PyDateTime_Check(value.ptr()) || type == "datetime64") {
        std::optional<int64_t> micros;
        if (!PyDateTime_Check(value.ptr())) {
          // A value of another type named datetime64 that has no astype or no
          // dtype is not numpy's, and so not a supported type.
          const nb::object astype = optional_attr(value, "astype");
          const nb::object dtype = optional_attr(value, "dtype");
          if (!astype.is_valid() || !dtype.is_valid()) throw unsupported();
          micros = numpy_epoch_microseconds(astype, dtype, name);
        } else if (const nb::object asm8 = optional_attr(value, "asm8"); asm8.is_valid()) {
          // A pandas Timestamp, NaT included, is a datetime whose fields
          // cannot be trusted: they read 0001-01-01 for NaT and hold a
          // stand-in year outside 1 to 9999. Its asm8 is its true count (the
          // UTC instant when aware), and an asm8 without astype or dtype
          // raises the AttributeError of the one it lacks.
          const nb::object astype = nb::getattr(asm8, "astype");
          const nb::object dtype = nb::getattr(asm8, "dtype");
          micros = numpy_epoch_microseconds(astype, dtype, name);
        } else {
          micros = epoch_microseconds(value.ptr(), name);
        }
        if (micros) {
          row.emplace_back(std::move(name), *micros);
        } else {
          row.emplace_back(std::move(name), std::monostate{});
        }
      } else if (type == "bool" || type == "bool_") {  // numpy's bool (2.x / 1.x names)
        const int truth = PyObject_IsTrue(value.ptr());
        if (truth < 0) throw nb::python_error();  // the error its __bool__ raised
        row.emplace_back(std::move(name), truth == 1);
      } else if (optional_attr(value, "__index__").is_valid()) {  // integer-like: fits int64 or fails, never widens
        nb::object index = nb::steal(PyNumber_Index(value.ptr()));
        if (!index.is_valid()) {
          // A TypeError is __index__ refusing it (a float or non-scalar
          // ndarray, say); any other error is the value's own and propagates.
          if (!PyErr_ExceptionMatches(PyExc_TypeError)) throw nb::python_error();
          PyErr_Clear();
          throw unsupported();
        }
        int overflow = 0;
        const long long integer = PyLong_AsLongLongAndOverflow(index.ptr(), &overflow);
        if (overflow != 0) {
          throw fastfeaturetransform::Error("record value for '" + name + "' does not fit in a 64-bit integer");
        }
        row.emplace_back(std::move(name), static_cast<int64_t>(integer));
      } else if (kNumpyFloats.contains(type) && nb::try_cast(value, real)) {
        row.emplace_back(std::move(name), real);
      } else {
        throw unsupported();
      }
    }
  }
  if (can_repeat) refuse_repeated_keys(row);
  return row;
}

nb::dict row_to_dict(const Row& row) {
  nb::dict record;
  for (const auto& [name, value] : row) {
    nb::object out = std::visit(
        [](const auto& held) -> nb::object {
          using T = std::decay_t<decltype(held)>;
          if constexpr (std::is_same_v<T, std::monostate>) {
            return nb::none();
          } else {
            return nb::cast(held);
          }
        },
        value);
    // Straight into the dict: record[key] = out adds a reference round trip and
    // an out-of-line call per field, which an expanding encoder pays for every
    // generated column of every record it serves.
    nb::object key = nb::cast(name);
    if (PyDict_SetItem(record.ptr(), key.ptr(), out.ptr()) != 0) throw nb::python_error();
  }
  return record;
}

}  // namespace

NB_MODULE(_core, m) {
  m.doc() = "fast-feature-transform C++ core";
  // The datetime C API, which row_from_dict reads a record's datetimes with.
  PyDateTime_IMPORT;
  if (PyDateTimeAPI == nullptr) throw nb::python_error();

  nb::exception<fastfeaturetransform::Error> engine_error(m, "EngineError", PyExc_RuntimeError);
  // nb::exception raises with e.what(), a C string: it ends at a NUL byte, and
  // a byte that is not UTF-8 turns the error into a UnicodeDecodeError. A
  // message names columns, keys and files, which may hold either. This
  // translator, registered later and so tried first, decodes the whole
  // message and writes a byte that is not UTF-8 as \xNN.
  nb::register_exception_translator(
      [](const std::exception_ptr& error, void* type) {
        try {
          std::rethrow_exception(error);
        } catch (const fastfeaturetransform::Error& e) {
          PyObject* message =
              PyUnicode_DecodeUTF8(e.message.data(), static_cast<Py_ssize_t>(e.message.size()), "backslashreplace");
          if (message == nullptr) return;  // out of memory: that error stays set
          PyErr_SetObject(static_cast<PyObject*>(type), message);
          Py_DECREF(message);
        }
      },
      engine_error.ptr());

  // dynamic_attr: the Python layer caches the serialized manifest on the
  // handle so every transformer bound to it pickles one shared string.
  //
  // Every call that can take long releases the GIL: fit, the batch paths,
  // parquet_to_memmap, and the ones that serialize or parse a whole artifact
  // (save, load, pickling via manifest_json and load_json, stage_state,
  // stages_state). On a free-threaded build these already run beside every
  // other thread, and FittedPipeline allows it; on the standard build, holding
  // it would stop every other Python thread for as long as the call runs.
  // transform_row keeps it: a record takes about a microsecond or less, and
  // its dict conversion needs it.
  nb::class_<FittedPipeline>(m, "FittedPipeline", nb::dynamic_attr())
      .def("transform_batch", &FittedPipeline::transform_batch, "sources"_a, "output_dir"_a,
           nb::call_guard<nb::gil_scoped_release>())
      .def("transform_memmap", &FittedPipeline::transform_memmap, "sources"_a, "output_dir"_a,
           nb::call_guard<nb::gil_scoped_release>())
      .def(
          "transform_row",
          [](const FittedPipeline& self, const nb::dict& record) {
            return row_to_dict(self.transform_row(row_from_dict(record)));
          },
          "record"_a)
      .def("stage_state", &FittedPipeline::stage_state, "stage"_a, nb::call_guard<nb::gil_scoped_release>())
      .def("vocab_size", &FittedPipeline::vocab_size, "stage"_a, "column"_a)
      .def("manifest_json", &FittedPipeline::manifest_json, nb::call_guard<nb::gil_scoped_release>())
      .def("stages_json", &FittedPipeline::stages_json)
      .def("stages_state", &FittedPipeline::stages_state, nb::call_guard<nb::gil_scoped_release>())
      .def("save", &FittedPipeline::save, "dir"_a, nb::call_guard<nb::gil_scoped_release>())
      .def_static("load", &FittedPipeline::load, "dir"_a, "threads"_a = 0, nb::call_guard<nb::gil_scoped_release>())
      .def_static("load_json", &FittedPipeline::load_json, "manifest"_a, "threads"_a = 0,
                  nb::call_guard<nb::gil_scoped_release>());

  nb::class_<Engine>(m, "Engine")
      .def(nb::init<unsigned, std::string>(), "threads"_a = 0, "temp_dir"_a = "")
      .def("fit", &Engine::fit, "plan_json"_a, "sources"_a, nb::call_guard<nb::gil_scoped_release>());

  m.def("library_version", &fastfeaturetransform::library_version);
  m.def("parquet_to_memmap", &fastfeaturetransform::write_memmap, "sources"_a, "output_dir"_a, "threads"_a = 0,
        nb::call_guard<nb::gil_scoped_release>());
}
