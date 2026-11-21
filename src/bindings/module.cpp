// The generic nanobind seam: Engine and FittedPipeline handles, plan JSON in.
// Adding a kernel never touches this file — the boundary speaks plans and
// state JSON, not per-transformer classes.

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
// After nanobind, which brings in Python.h.
#include <ankerl/unordered_dense.h>
#include <datetime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "fastfeaturetransform/engine.hpp"

namespace nb = nanobind;
using namespace nb::literals;
using fastfeaturetransform::Engine;
using fastfeaturetransform::FittedPipeline;
using fastfeaturetransform::Row;
using fastfeaturetransform::RowValue;

namespace {

// The __name__ of a value's type, as the type holds it. Looking the
// attribute up instead runs the code of a metaclass, which can give a value
// that is not a str.
std::string python_type_name(nb::handle value) {
  const nb::object name = nb::steal(PyType_GetName(Py_TYPE(value.ptr())));
  if (!name.is_valid()) throw nb::python_error();
  Py_ssize_t size = 0;
  const char* const data = PyUnicode_AsUTF8AndSize(name.ptr(), &size);
  if (data == nullptr) throw nb::python_error();
  return {data, static_cast<std::size_t>(size)};
}

// The name of numpy's own type of a value, such as "float32", or an empty
// view for a value of any other type. numpy's types are static types, whose
// C name holds the module and the name together: "numpy.float32". A class
// made in Python is a heap type, whose C name is only its __name__, so it is
// never taken for numpy's, whatever its __name__ and __module__; neither is
// a subclass of a numpy type. numpy is not imported to tell: a value of one
// of its types means that it is loaded.
std::string_view numpy_type_name(PyTypeObject* type) {
  constexpr std::string_view numpy = "numpy.";
  const std::string_view name = type->tp_name;
  if (PyType_HasFeature(type, Py_TPFLAGS_HEAPTYPE) || !name.starts_with(numpy)) return {};
  return name.substr(numpy.size());
}

// numpy's floating scalar types, the only non-builtin values converted
// through __float__ (anything else with a __float__, such as a numpy complex
// or a Decimal, is rejected rather than silently narrowed).
constexpr std::array<std::string_view, 5> kNumpyFloats{"float16", "float32", "float64", "longdouble", "float128"};

__extension__ typedef __int128 Int128;  // NOLINT(modernize-use-using): __extension__ needs the typedef form

// A datetime enters the engine as int64 microseconds since 1970-01-01, the
// count Arrow's timestamp[us] stores: the wall clock as if it were UTC for a
// naive value, the UTC instant for an aware one. A Python datetime holds
// whole microseconds in its fields.
int64_t epoch_microseconds(PyObject* value, const std::string& name) {
  using namespace std::chrono;
  const sys_days date{year{PyDateTime_GET_YEAR(value)} / month{static_cast<unsigned>(PyDateTime_GET_MONTH(value))} /
                      day{static_cast<unsigned>(PyDateTime_GET_DAY(value))}};
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
  const Int128 shift =
      (Int128{PyDateTime_DELTA_GET_DAYS(offset.ptr())} * 86'400 + PyDateTime_DELTA_GET_SECONDS(offset.ptr())) *
          1'000'000 +
      PyDateTime_DELTA_GET_MICROSECONDS(offset.ptr());
  const Int128 adjusted = Int128{micros} - shift;
  if (adjusted > std::numeric_limits<int64_t>::max() || adjusted < std::numeric_limits<int64_t>::min()) {
    throw fastfeaturetransform::Error("record value for '" + name + "' is outside the range of int64 microseconds");
  }
  return static_cast<int64_t>(adjusted);
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

// A numpy.datetime64 (or a pandas Timestamp's asm8, which is one) from its
// stored count, which its astype reads, and its unit, which its dtype holds,
// rescaled exactly: numpy's own cast to microseconds can overflow on a value
// that fits. Finer units floor, as numpy and the batch path do; NaT, which
// numpy stores as INT64_MIN, becomes null.
std::optional<int64_t> numpy_epoch_microseconds(nb::handle datetime64, const std::string& name) {
  // numpy's own astype and item, which Python code cannot replace on numpy's
  // type, give the count as an int.
  const auto count = nb::cast<int64_t>(nb::getattr(datetime64, "astype")("int64").attr("item")());
  if (count == std::numeric_limits<int64_t>::min()) return std::nullopt;
  const nb::object dtype = nb::getattr(datetime64, "dtype");
  // numpy.datetime_data is looked up in numpy's module, where Python code can
  // replace it, so its answer is checked. numpy gives the unit as a str and
  // the step, which it stores as a C int, as an int of at least 1.
  const nb::object unit_and_step = nb::module_::import_("numpy").attr("datetime_data")(dtype);
  const nb::object unit_item = unit_and_step[0];
  const nb::object step_item = unit_and_step[1];
  std::string unit;
  int64_t step = 0;
  if (!nb::try_cast(unit_item, unit) || !nb::try_cast(step_item, step, false) || step < 1 ||
      step > std::numeric_limits<int32_t>::max()) {
    throw fastfeaturetransform::Error("record value for '" + name +
                                      "' is a numpy datetime64 whose unit numpy.datetime_data does not give as a "
                                      "str and an int from 1 to 2**31 - 1");
  }
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
// once, but it treats two keys of one text as one key only when their hashes
// are equal and they compare equal. Keys whose types use str's own hash and
// comparison always do: a str, a StrEnum member, a member of an Enum with a
// str mixin, a numpy.str_. A key whose type has its own hash or comparison
// can sit beside a str of its text, so the names are checked only when a
// key's type has one. A key whose hash or equality changes while it is in
// the record, as when its class or its class's __hash__ or __eq__ changes,
// breaks the rule for dict keys: the walk sees only the type the key has
// now, so such a key can repeat a text unchecked. Python code that a value
// runs cannot repeat a key: row_from_dict reads such a value only after it
// has read every other field.
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

// Whether row_from_dict reads a value that is not None, a bool, an int, a
// float or a str without running Python code, so that the value cannot
// change the record while it is read: an exact datetime whose tzinfo is None
// or a datetime.timezone, or a value of a numpy type other than datetime64,
// which numpy's C functions alone convert or refuse (see numpy_type_name).
// Any other value counts as one that can run Python code: the methods of a
// class made in Python, another tzinfo's utcoffset, a static type of another
// library that calls Python code from its C functions (a weakref proxy calls
// its referent's __index__), and a datetime64, which converts through
// numpy.datetime_data and builtins.__import__, both of which Python code can
// replace.
// It is kept out of line: inlined into row_from_dict, it made a record of 128
// str values that are not ASCII, which never reach it, take 5% more time
// through one Select step (an Apple M4, one thread).
[[gnu::noinline]] bool converts_without_python(PyObject* value) {
  if (PyDateTime_CheckExact(value)) {
    PyObject* const tzinfo = PyDateTime_DATE_GET_TZINFO(value);
    return tzinfo == Py_None || Py_IS_TYPE(tzinfo, Py_TYPE(PyDateTime_TimeZone_UTC));
  }
  const std::string_view numpy = numpy_type_name(Py_TYPE(value));
  return !numpy.empty() && numpy != "datetime64";
}

// Reads a record value that is not None, a bool, an int, a float or a str.
// name is its key, for the errors.
RowValue read_other_value(nb::handle value, const std::string& name) {
  const std::string_view numpy = numpy_type_name(Py_TYPE(value.ptr()));
  const auto unsupported = [&] {
    return fastfeaturetransform::Error("record value for '" + name +
                                       "' must be str, int, float, bool, datetime, or None (got " +
                                       python_type_name(value) + ")");
  };
  if (PyDateTime_Check(value.ptr()) || numpy == "datetime64") {
    std::optional<int64_t> micros;
    if (!PyDateTime_Check(value.ptr())) {
      micros = numpy_epoch_microseconds(value, name);
    } else if (const nb::object asm8 = optional_attr(value, "asm8"); asm8.is_valid()) {
      // A pandas Timestamp, NaT included, is a datetime whose fields cannot
      // be trusted: they read 0001-01-01 for NaT and hold a stand-in year
      // outside 1 to 9999. Its asm8, a numpy datetime64, is its true count
      // (the UTC instant when aware).
      if (numpy_type_name(Py_TYPE(asm8.ptr())) != "datetime64") {
        throw fastfeaturetransform::Error("record value for '" + name +
                                          "' is a datetime whose asm8 is not a numpy datetime64 (got " +
                                          python_type_name(asm8) + ")");
      }
      micros = numpy_epoch_microseconds(asm8, name);
    } else {
      micros = epoch_microseconds(value.ptr(), name);
    }
    if (micros) return *micros;
    return std::monostate{};
  }
  if (numpy == "bool" || numpy == "bool_") {  // numpy's bool (2.x / 1.x names)
    // numpy's own bool answers its truth test from its value, which cannot
    // fail; a subclass is a heap type, which numpy_type_name never names.
    return PyObject_IsTrue(value.ptr()) == 1;
  }
  if (optional_attr(value, "__index__").is_valid()) {  // integer-like: fits int64 or fails, never widens
    nb::object index = nb::steal(PyNumber_Index(value.ptr()));
    if (!index.is_valid()) {
      // A TypeError is __index__ refusing it (a float or non-scalar ndarray,
      // say); any other error is the value's own and propagates.
      if (!PyErr_ExceptionMatches(PyExc_TypeError)) throw nb::python_error();
      PyErr_Clear();
      throw unsupported();
    }
    int overflow = 0;
    const long long integer = PyLong_AsLongLongAndOverflow(index.ptr(), &overflow);
    if (overflow != 0) {
      throw fastfeaturetransform::Error("record value for '" + name + "' does not fit in a 64-bit integer");
    }
    return static_cast<int64_t>(integer);
  }
  if (double real = 0.0; std::ranges::find(kNumpyFloats, numpy) != kNumpyFloats.end() && nb::try_cast(value, real)) {
    return real;
  }
  throw unsupported();
}

// Record values: str, int, float, bool, None, datetimes (as epoch
// microseconds), plus numpy scalars (their integer and floating types convert
// through __index__/__float__). Each refusal of a value names its key.
//
// A record is read as it was when the call began. Reading a value can run
// Python code (see converts_without_python), and that code can change the
// record: take a key out, add keys, rebuild the dict's table or store
// another value. A walk over the dict would then read a key twice, skip a
// key that the record held throughout, or read a key or a value that the
// record did not hold when the call began. So the walk holds each such
// value, reads it only after the walk has read every other field, and runs
// no Python code itself. When a record holds more than one value that is
// refused, the refusal of such a value comes after those of the others.
Row row_from_dict(const nb::dict& record) {
  Row row;
  row.reserve(nb::len(record));
  bool can_repeat = false;  // whether refuse_repeated_keys must look
  // The values read after the walk, each with the index of its field in row.
  std::vector<std::pair<std::size_t, nb::object>> deferred;
  for (auto [key, value] : record) {
    // A key is read with the calls nanobind's caster for std::string makes,
    // without the string the caster builds and then moves, and the test for
    // an exact str sits beside the test for a str. A record of 30 float
    // fields through one Select step took a sixth less time than through
    // nb::try_cast (an Apple M4, one thread).
    if (!PyUnicode_Check(key.ptr())) throw fastfeaturetransform::Error("record keys must be str");
    if (const PyTypeObject* const key_type = Py_TYPE(key.ptr());
        key_type != &PyUnicode_Type &&
        (key_type->tp_hash != PyUnicode_Type.tp_hash || key_type->tp_richcompare != PyUnicode_Type.tp_richcompare)) {
      can_repeat = true;
    }
    // The record lends its keys and values, and its reference can be the
    // only one. On a free-threaded build another thread can take the key or
    // the value out of the record and free it whenever this thread waits,
    // since waiting suspends the record's lock. So from the first step that
    // can wait, the key and the value that are still to be read are held.
    // Reading None, a bool, an int, a float or an ASCII str does not wait. The
    // first UTF-8 conversion of a str that is not ASCII locks the str, and
    // can wait for that lock.
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
    } else if (converts_without_python(value.ptr())) {
      held_value = nb::borrow(value);
      RowValue read = read_other_value(value, name);
      row.emplace_back(std::move(name), std::move(read));
    } else {
      // The value is alive here: held_value holds it if the key waited, and
      // the record holds it if the key did not.
      deferred.emplace_back(row.size(), nb::borrow(value));
      row.emplace_back(std::move(name), std::monostate{});
    }
  }
  for (const auto& [index, value] : deferred) row[index].second = read_other_value(value, row[index].first);
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
          [](const FittedPipeline& self, nb::handle record) {
            // Only transform_dict calls this, so the TypeError for a record
            // that is not a dict, None included, names it.
            if (!PyDict_Check(record.ptr())) {
              throw nb::type_error(("transform_dict: X must be a dict, got " + python_type_name(record)).c_str());
            }
            return row_to_dict(self.transform_row(row_from_dict(nb::borrow<nb::dict>(record))));
          },
          "record"_a.none())
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
