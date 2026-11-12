// The closed cast table, shared by every kernel that has to read a number out
// of a column or put one back. Types are explicit: apart from reading
// datetimes as epoch microseconds (io.cpp for columns, row_from_dict in the
// binding for records), the engine never converts a value on its own, so each
// conversion here is listed on purpose and anything absent is an error.
#pragma once

#include <arrow/array/builder_binary.h>
#include <arrow/array/builder_primitive.h>

#include <charconv>
#include <cmath>
#include <format>
#include <optional>
#include <system_error>
#include <unordered_map>

#include "fastfeaturetransform/kernels/output.hpp"
#include "fastfeaturetransform/kernels/util.hpp"

namespace fastfeaturetransform {

// ---------------------------------------------------------------------------
// The cast kernel applies this table. Anything not listed is an error when
// the step resolves its input schema, and on the record path, where the
// value's own type is the source.
// ---------------------------------------------------------------------------
enum class Family { Integer, Floating, Utf8, Bool };

struct CastTarget {
  std::string tag;
  Family family = Family::Utf8;
  bool is_unsigned = false;
  int bits = 64;
};

inline const char* kCastTargets = "utf8, i8, i16, i32, i64, u8, u16, u32, u64, f32, f64";

inline CastTarget cast_target(const std::string& tag, const std::string& what) {
  static const std::unordered_map<std::string, CastTarget> kTargets{
      {"utf8", {"utf8", Family::Utf8}},
      {"i8", {"i8", Family::Integer, false, 8}},
      {"i16", {"i16", Family::Integer, false, 16}},
      {"i32", {"i32", Family::Integer, false, 32}},
      {"i64", {"i64", Family::Integer, false, 64}},
      {"u8", {"u8", Family::Integer, true, 8}},
      {"u16", {"u16", Family::Integer, true, 16}},
      {"u32", {"u32", Family::Integer, true, 32}},
      {"u64", {"u64", Family::Integer, true, 64}},
      {"f32", {"f32", Family::Floating, false, 32}},
      {"f64", {"f64", Family::Floating, false, 64}},
  };
  auto found = kTargets.find(tag);
  if (found == kTargets.end()) {
    throw Error(what + ": unsupported cast target '" + tag + "' (choose one of " + kCastTargets + ")");
  }
  return found->second;
}

// The family of a schema type tag that cast can read; nullopt otherwise.
inline std::optional<Family> family_of_tag(const std::string& tag) {
  static const std::unordered_map<std::string, Family> kFamilies{
      {"utf8", Family::Utf8},   {"bool", Family::Bool},   {"i8", Family::Integer},   {"i16", Family::Integer},
      {"i32", Family::Integer}, {"i64", Family::Integer}, {"u8", Family::Integer},   {"u16", Family::Integer},
      {"u32", Family::Integer}, {"u64", Family::Integer}, {"f32", Family::Floating}, {"f64", Family::Floating},
  };
  auto found = kFamilies.find(tag);
  if (found == kFamilies.end()) return std::nullopt;
  return found->second;
}

inline const char* family_name(Family family) {
  switch (family) {
    case Family::Integer:
      return "an integer";
    case Family::Floating:
      return "a float";
    case Family::Utf8:
      return "a string";
    case Family::Bool:
      return "a bool";
  }
  return "a value";
}

inline bool cast_allowed(Family from, Family to) {
  switch (from) {
    case Family::Integer:
      return to != Family::Bool;  // any width, to float when exact, to decimal digits
    case Family::Floating:
      return to == Family::Floating || to == Family::Integer;  // never to a string: no single spelling
    case Family::Utf8:
      return to != Family::Bool;  // parsed strictly
    case Family::Bool:
      return to == Family::Integer;  // 0 / 1
  }
  return false;
}

inline std::string cast_refusal(const std::string& where, Family from, const CastTarget& target) {
  std::string message = where + ": cannot cast " + family_name(from) + " to " + target.tag;
  if (from == Family::Floating && target.family == Family::Utf8) {
    message += " (a float has no single spelling; cast to an integer first)";
  }
  return message;
}

inline int64_t integer_in_range(int64_t value, const CastTarget& target, const std::string& where) {
  int64_t low = std::numeric_limits<int64_t>::min();
  int64_t high = std::numeric_limits<int64_t>::max();
  if (target.is_unsigned) {
    low = 0;
    if (target.bits < 64) high = (int64_t{1} << target.bits) - 1;
  } else if (target.bits < 64) {
    high = (int64_t{1} << (target.bits - 1)) - 1;
    low = -high - 1;
  }
  if (value < low || value > high) {
    throw Error(where + ": value " + std::to_string(value) + " does not fit in " + target.tag);
  }
  return value;
}

inline int64_t whole_number(double value, const std::string& where) {
  if (!std::isfinite(value) || std::trunc(value) != value) {
    throw Error(where + ": value " + std::format("{}", value) + " is not a whole number");
  }
  if (value < -9223372036854775808.0 || value >= 9223372036854775808.0) {
    throw Error(where + ": value " + std::format("{}", value) + " does not fit in a 64-bit integer");
  }
  return static_cast<int64_t>(value);
}

// Strict grammar for both parsers: an optional single sign, then digits (and
// for floats a point or exponent); nothing else, no whitespace, no "inf" or
// "nan" spellings.
inline std::string_view unsigned_part(std::string_view text) {
  if (!text.empty() && (text.front() == '+' || text.front() == '-')) text.remove_prefix(1);
  return text;
}

inline int64_t parse_integer(std::string_view text, const std::string& where) {
  const std::string_view digits = unsigned_part(text);
  std::string_view parsed = text.starts_with('+') ? digits : text;  // from_chars reads '-' itself, never '+'
  int64_t value = 0;
  auto [end, error] = std::from_chars(parsed.data(), parsed.data() + parsed.size(), value);
  if (digits.empty() || (digits.front() < '0' || digits.front() > '9') || error == std::errc::invalid_argument ||
      end != parsed.data() + parsed.size()) {
    throw Error(where + ": '" + std::string(text) + "' is not an integer");
  }
  if (error == std::errc::result_out_of_range) {
    throw Error(where + ": '" + std::string(text) + "' does not fit in a 64-bit integer");
  }
  return value;
}

inline double parse_floating(std::string_view text, const std::string& where) {
  const std::string_view digits = unsigned_part(text);
  std::string_view parsed = text.starts_with('+') ? digits : text;
  double value = 0.0;
  auto [end, error] = std::from_chars(parsed.data(), parsed.data() + parsed.size(), value);
  const bool starts_numeric =
      !digits.empty() && ((digits.front() >= '0' && digits.front() <= '9') || digits.front() == '.');
  if (!starts_numeric || error == std::errc::invalid_argument || end != parsed.data() + parsed.size()) {
    throw Error(where + ": '" + std::string(text) + "' is not a number");
  }
  if (error == std::errc::result_out_of_range || !std::isfinite(value)) {
    throw Error(where + ": '" + std::string(text) + "' is out of range for f64");
  }
  return value;
}

// Integers convert to floats only where the float holds every integer
// exactly: up to 2^53 in magnitude for f64 and 2^24 for f32.
inline double floating_from_integer(int64_t value, const CastTarget& target, const std::string& where) {
  const int64_t exact = target.bits == 32 ? (int64_t{1} << 24) : (int64_t{1} << 53);
  if (value > exact || value < -exact) {
    throw Error(where + ": value " + std::to_string(value) + " is beyond " + (target.bits == 32 ? "2^24" : "2^53") +
                " in magnitude, where " + target.tag + " no longer holds every integer exactly");
  }
  return target.bits == 32 ? static_cast<double>(static_cast<float>(value)) : static_cast<double>(value);
}

// Narrowing to f32 rounds to the nearest single (what the Parquet column will
// hold), but a value beyond f32's range is an error rather than infinity.
// The check runs before the conversion: narrowing an out-of-range double is
// undefined behavior, so the bound is the smallest double that would round
// up to infinity (FLT_MAX plus half a unit in the last place).
inline double floating_in_width(double value, const CastTarget& target, const std::string& where) {
  if (target.bits == 64) return value;
  constexpr double kRoundsToInfinity = 3.4028235677973366e38;
  if (std::isfinite(value) && std::fabs(value) >= kRoundsToInfinity) {
    throw Error(where + ": value " + std::format("{}", value) + " does not fit in f32");
  }
  return static_cast<double>(static_cast<float>(value));
}

template <typename ArrowType, typename Value>
std::shared_ptr<arrow::Array> build_numeric(const std::vector<Value>& values, const std::vector<bool>& nulls,
                                            const std::string& where) {
  arrow::NumericBuilder<ArrowType> builder;
  if (!builder.Reserve(static_cast<int64_t>(values.size())).ok()) throw Error(where + ": allocation failed");
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (nulls[i]) {
      builder.UnsafeAppendNull();
    } else {
      builder.UnsafeAppend(static_cast<typename ArrowType::c_type>(values[i]));
    }
  }
  std::shared_ptr<arrow::Array> array;
  if (!builder.Finish(&array).ok()) throw Error(where + ": failed to build the cast column");
  return array;
}

inline std::shared_ptr<arrow::Array> build_integer(const CastTarget& target, const std::vector<int64_t>& values,
                                                   const std::vector<bool>& nulls, const std::string& where) {
  if (target.is_unsigned) {
    switch (target.bits) {
      case 8:
        return build_numeric<arrow::UInt8Type>(values, nulls, where);
      case 16:
        return build_numeric<arrow::UInt16Type>(values, nulls, where);
      case 32:
        return build_numeric<arrow::UInt32Type>(values, nulls, where);
      default:
        return build_numeric<arrow::UInt64Type>(values, nulls, where);
    }
  }
  switch (target.bits) {
    case 8:
      return build_numeric<arrow::Int8Type>(values, nulls, where);
    case 16:
      return build_numeric<arrow::Int16Type>(values, nulls, where);
    case 32:
      return build_numeric<arrow::Int32Type>(values, nulls, where);
    default:
      return build_numeric<arrow::Int64Type>(values, nulls, where);
  }
}

inline Family family_of_array(const arrow::Array& array, const std::string& where) {
  switch (array.type_id()) {
    case arrow::Type::STRING:
    case arrow::Type::LARGE_STRING:
      return Family::Utf8;
    case arrow::Type::DICTIONARY: {
      const arrow::Type::type values = static_cast<const arrow::DictionaryArray&>(array).dictionary()->type_id();
      if (values == arrow::Type::STRING || values == arrow::Type::LARGE_STRING) return Family::Utf8;
      throw Error(where + " is " + array.type()->ToString() + ", which cast cannot read");
    }
    case arrow::Type::INT8:
    case arrow::Type::INT16:
    case arrow::Type::INT32:
    case arrow::Type::INT64:
    case arrow::Type::UINT8:
    case arrow::Type::UINT16:
    case arrow::Type::UINT32:
    case arrow::Type::UINT64:
      return Family::Integer;
    case arrow::Type::FLOAT:
    case arrow::Type::DOUBLE:
      return Family::Floating;
    case arrow::Type::BOOL:
      return Family::Bool;
    default:
      throw Error(where + " is " + array.type()->ToString() + ", which cast cannot read");
  }
}

// Whether an array already has the target's exact type, so the cast is a no-op
// (this keeps u64 values above INT64_MAX intact, which the int64 pivot of
// for_each_integer could not).
inline bool already_target(const arrow::Array& array, const CastTarget& target) {
  static const std::unordered_map<std::string, arrow::Type::type> kArrowTypes{
      {"i8", arrow::Type::INT8},    {"i16", arrow::Type::INT16},  {"i32", arrow::Type::INT32},
      {"i64", arrow::Type::INT64},  {"u8", arrow::Type::UINT8},   {"u16", arrow::Type::UINT16},
      {"u32", arrow::Type::UINT32}, {"u64", arrow::Type::UINT64}, {"f32", arrow::Type::FLOAT},
      {"f64", arrow::Type::DOUBLE},
  };
  auto found = kArrowTypes.find(target.tag);
  return found != kArrowTypes.end() && array.type_id() == found->second;
}

// Batch path: one scalar rule per (source family, target), applied per value.
inline std::shared_ptr<arrow::Array> cast_array(const std::shared_ptr<arrow::Array>& array, const CastTarget& target,
                                                const std::string& stage, const std::string& name) {
  const std::string where = stage + ": column '" + name + "'";
  const Family from = family_of_array(*array, where);
  if (!cast_allowed(from, target.family)) throw Error(cast_refusal(where, from, target));
  if (already_target(*array, target)) return array;

  if (target.family == Family::Utf8) {
    if (from == Family::Utf8) return array;
    arrow::StringBuilder builder;
    auto append = [&](bool is_null, const std::string& digits) {
      if (!(is_null ? builder.AppendNull() : builder.Append(digits)).ok()) throw Error(where + ": allocation failed");
    };
    if (array->type_id() == arrow::Type::UINT64) {  // digits of the full unsigned range, no int64 pivot
      visit_numeric<arrow::UInt64Type>(*array, [&](int64_t, bool is_null, uint64_t value) {
        append(is_null, is_null ? std::string() : std::to_string(value));
      });
    } else {
      for_each_integer(*array, stage, name, [&](int64_t, bool is_null, int64_t value) {
        append(is_null, is_null ? std::string() : std::to_string(value));
      });
    }
    std::shared_ptr<arrow::Array> out;
    if (!builder.Finish(&out).ok()) throw Error(where + ": failed to build the cast column");
    return out;
  }

  if (target.family == Family::Integer) {
    auto convert = [&]<typename ArrowType>() {
      PrimitiveOutput<ArrowType> output(array->length(), where);
      auto keep = [&](int64_t row, bool is_null, int64_t value) {
        if (is_null) {
          output.set_null(row);
        } else {
          output.mutable_values()[row] =
              static_cast<typename ArrowType::c_type>(integer_in_range(value, target, where));
        }
      };
      switch (from) {
        case Family::Integer:
          for_each_integer(*array, stage, name, keep);
          break;
        case Family::Floating:
          for_each_floating(*array, stage, name, [&](int64_t row, bool is_null, double value) {
            keep(row, is_null, is_null ? 0 : whole_number(value, where));
          });
          break;
        case Family::Utf8:
          for_each_utf8_value(*array, stage, name, [&](int64_t row, bool is_null, std::string_view value) {
            keep(row, is_null, is_null ? 0 : parse_integer(value, where));
          });
          break;
        case Family::Bool: {
          const auto& bools = static_cast<const arrow::BooleanArray&>(*array);
          for (int64_t row = 0; row < bools.length(); ++row) {
            keep(row, bools.IsNull(row), !bools.IsNull(row) && bools.Value(row) ? 1 : 0);
          }
          break;
        }
      }
      return output.finish();
    };
    if (target.is_unsigned) {
      switch (target.bits) {
        case 8:
          return convert.template operator()<arrow::UInt8Type>();
        case 16:
          return convert.template operator()<arrow::UInt16Type>();
        case 32:
          return convert.template operator()<arrow::UInt32Type>();
        default:
          return convert.template operator()<arrow::UInt64Type>();
      }
    }
    switch (target.bits) {
      case 8:
        return convert.template operator()<arrow::Int8Type>();
      case 16:
        return convert.template operator()<arrow::Int16Type>();
      case 32:
        return convert.template operator()<arrow::Int32Type>();
      default:
        return convert.template operator()<arrow::Int64Type>();
    }
  }

  auto convert = [&]<typename ArrowType>() {
    PrimitiveOutput<ArrowType> output(array->length(), where);
    auto keep = [&](int64_t row, bool is_null, double value) {
      if (is_null) {
        output.set_null(row);
      } else {
        output.mutable_values()[row] = static_cast<typename ArrowType::c_type>(value);
      }
    };
    switch (from) {
      case Family::Integer:
        for_each_integer(*array, stage, name, [&](int64_t row, bool is_null, int64_t value) {
          keep(row, is_null, is_null ? 0.0 : floating_from_integer(value, target, where));
        });
        break;
      case Family::Floating:
        for_each_floating(*array, stage, name, [&](int64_t row, bool is_null, double value) {
          keep(row, is_null, is_null ? 0.0 : floating_in_width(value, target, where));
        });
        break;
      case Family::Utf8:
        for_each_utf8_value(*array, stage, name, [&](int64_t row, bool is_null, std::string_view value) {
          keep(row, is_null, is_null ? 0.0 : floating_in_width(parse_floating(value, where), target, where));
        });
        break;
      case Family::Bool:
        throw Error(cast_refusal(where, from, target));
    }
    return output.finish();
  };
  return target.bits == 32 ? convert.template operator()<arrow::FloatType>()
                           : convert.template operator()<arrow::DoubleType>();
}

// Record path: the same scalar rules, keyed on the value's own type. Every
// message begins with `where`: the cast kernel hands in an empty one and
// puts the key's place in front of a refusal itself.
inline RowValue cast_value(RowValue value, const CastTarget& target, const std::string& where) {
  if (std::holds_alternative<std::monostate>(value)) return value;
  Family from = Family::Utf8;
  if (std::holds_alternative<bool>(value)) {
    from = Family::Bool;
  } else if (std::holds_alternative<int64_t>(value)) {
    from = Family::Integer;
  } else if (std::holds_alternative<double>(value)) {
    from = Family::Floating;
  }
  if (!cast_allowed(from, target.family)) throw Error(cast_refusal(where, from, target));

  if (target.family == Family::Utf8) {
    if (from == Family::Integer) return std::to_string(std::get<int64_t>(value));
    return value;  // a string stays as it is
  }
  if (target.family == Family::Integer) {
    int64_t integer = 0;
    switch (from) {
      case Family::Integer:
        integer = std::get<int64_t>(value);
        break;
      case Family::Floating:
        integer = whole_number(std::get<double>(value), where);
        break;
      case Family::Utf8:
        integer = parse_integer(std::get<std::string>(value), where);
        break;
      case Family::Bool:
        integer = std::get<bool>(value) ? 1 : 0;
        break;
    }
    return integer_in_range(integer, target, where);
  }
  switch (from) {
    case Family::Integer:
      return floating_from_integer(std::get<int64_t>(value), target, where);
    case Family::Floating:
      return floating_in_width(std::get<double>(value), target, where);
    case Family::Utf8:
      return floating_in_width(parse_floating(std::get<std::string>(value), where), target, where);
    case Family::Bool:
      break;
  }
  throw Error(cast_refusal(where, from, target));
}
}  // namespace fastfeaturetransform
