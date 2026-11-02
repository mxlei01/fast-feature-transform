// The four contrast coders, which differ only in the matrix they build: a
// column of n fitted levels becomes n - 1 f64 columns, and the level at
// 0-based index i emits row i of an n x (n - 1) matrix. There is no intercept
// column — upstream calls patsy's code_without_intercept — so a column with a
// single level contrasts against nothing and emits no columns at all.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <unordered_map>

#include "fastfeaturetransform/kernels/cast_table.hpp"
#include "fastfeaturetransform/kernels/categories.hpp"
#include "fastfeaturetransform/kernels/kernels.hpp"
#include "fastfeaturetransform/kernels/output.hpp"
#include "fastfeaturetransform/kernels/util.hpp"

namespace fastfeaturetransform {
namespace {

// Which matrix an encoder builds; everything else about the four is identical.
enum class Coding { BackwardDifference, Helmert, Sum, Polynomial };

const char* coding_id(Coding coding) {
  switch (coding) {
    case Coding::BackwardDifference:
      return "backward_difference_encode";
    case Coding::Helmert:
      return "helmert_encode";
    case Coding::Sum:
      return "sum_encode";
    case Coding::Polynomial:
      break;
  }
  return "polynomial_encode";
}

// M[i][j] = (j + 1 - n) / n on and above the diagonal, (j + 1) / n below it:
// column j contrasts level j + 1 against level j, the one just before it.
// Every entry is one correctly rounded division of exact small integers, so
// this is patsy's Diff() bit for bit.
std::vector<std::vector<double>> backward_difference_rows(std::size_t n) {
  const auto levels = static_cast<double>(n);
  std::vector<std::vector<double>> rows(n, std::vector<double>(n - 1, 0.0));
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j + 1 < n; ++j) {
      const auto above = static_cast<double>(j + 1);
      rows[i][j] = i <= j ? (above - levels) / levels : above / levels;
    }
  }
  return rows;
}

// M[i][j] = -1 on and above the diagonal, i just below it, 0 elsewhere:
// column j contrasts level j + 1 against the mean of the levels before it.
// Every entry is a small integer held exactly in a double.
std::vector<std::vector<double>> helmert_rows(std::size_t n) {
  std::vector<std::vector<double>> rows(n, std::vector<double>(n - 1, 0.0));
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j + 1 < n; ++j) {
      if (i <= j) {
        rows[i][j] = -1.0;
      } else if (i == j + 1) {
        rows[i][j] = static_cast<double>(i);
      }
    }
  }
  return rows;
}

// The identity on every level but the last, which is all -1: each column
// contrasts one level against the grand mean, and the last level is the
// reference the others are read against.
std::vector<std::vector<double>> sum_rows(std::size_t n) {
  std::vector<std::vector<double>> rows(n, std::vector<double>(n - 1, 0.0));
  for (std::size_t i = 0; i + 1 < n; ++i) rows[i][i] = 1.0;
  for (std::size_t j = 0; j + 1 < n; ++j) rows[n - 1][j] = -1.0;
  return rows;
}

// Arnoldi on the diagonal matrix of centered, scaled level positions. Each
// candidate multiplies the previous normalized polynomial by x, avoiding
// the ill-conditioned raw Vandermonde. Two orthogonalization passes retain
// orthogonality to lower-degree columns even at high degree. Positive
// normalization fixes the sign to a positive leading coefficient. This is the
// discrete orthonormal polynomial basis; patsy's raw-power QR can differ
// materially at higher cardinalities. A saved state stores these values, so
// loading never rebuilds them.
std::vector<std::vector<double>> polynomial_rows(std::size_t n, const std::string& stage, const std::string& name) {
  std::vector<double> scores(n);
  const double midpoint = static_cast<double>(n - 1) / 2.0;
  for (std::size_t i = 0; i < n; ++i) scores[i] = (static_cast<double>(i) - midpoint) / midpoint;

  std::vector<std::vector<double>> basis(n, std::vector<double>(n));
  std::fill(basis[0].begin(), basis[0].end(), 1.0 / std::sqrt(static_cast<double>(n)));
  for (std::size_t k = 1; k < n; ++k) {
    auto& column = basis[k];
    for (std::size_t i = 0; i < n; ++i) column[i] = scores[i] * basis[k - 1][i];
    for (int pass = 0; pass < 2; ++pass) {
      for (std::size_t j = 0; j < k; ++j) {
        const auto& done = basis[j];
        double dot = 0.0;
        for (std::size_t i = 0; i < n; ++i) dot += done[i] * column[i];
        for (std::size_t i = 0; i < n; ++i) column[i] -= dot * done[i];
      }
    }
    double norm = 0.0;
    for (const double value : column) norm += value * value;
    norm = std::sqrt(norm);
    if (!std::isfinite(norm) || norm == 0.0) {
      throw Error(stage + ": column '" + name + "' has no finite polynomial contrast");
    }
    for (double& value : column) value /= norm;
  }

  // Reuse the square allocation for row-major lookup storage instead of
  // temporarily holding two full matrices. Drop the constant column.
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = i + 1; j < n; ++j) std::swap(basis[i][j], basis[j][i]);
    basis[i].erase(basis[i].begin());
  }
  return basis;
}

std::vector<std::vector<double>> contrast_rows(Coding coding, std::size_t n, const std::string& stage,
                                               const std::string& name) {
  switch (coding) {
    case Coding::BackwardDifference:
      return backward_difference_rows(n);
    case Coding::Helmert:
      return helmert_rows(n);
    case Coding::Sum:
      return sum_rows(n);
    case Coding::Polynomial:
      break;
  }
  return polynomial_rows(n, stage, name);
}

// ---------------------------------------------------------------------------
// backward_difference_encode / helmert_encode / sum_encode / polynomial_encode
// — categorical utf8 -> n - 1 f64 columns, replacing the source column.
// params: {"columns": [...] | null,
//          "handle_unknown": "value" | "return_nan" | "indicator" | "error",
//          "handle_missing": "value" | "return_nan" | "error"}
// Semantics:
//   * categories sort; the level at 0-based index i emits row i of the n x
//     (n - 1) matrix into '<column>_0' .. '<column>_{n-2}', all f64
//   * a null seen at fit under handle_missing='value' is a level of its own,
//     sorted last, so it widens the matrix; under 'return_nan' a null is a
//     row of nulls, and 'error' refuses a null at transform as well as at fit
//   * under 'return_nan' and 'error' alike the pre-encoder's slot for the
//     missing value is still a level of its own, sorted last, that nothing
//     can reach: the matrix is one level wider and emits one more column than
//     the categories alone would. See missing_placeholder in
//     kernels/categories.hpp
//   * an unseen category is a row of zeros under handle_unknown='value', a
//     row of nulls under 'return_nan', an error under 'error', and a real
//     extra level (the last one, so the matrix grows and one more column
//     appears) under 'indicator'
//   * under handle_missing='value', a null the fit never saw is a row of
//     zeros, whatever handle_unknown says
//   * a column with a single level has nothing to contrast and emits no
//     columns at all, indicator or not, as upstream does; under 'return_nan'
//     and 'error' the placeholder level is the second one, so a single
//     category does emit its one column
//   * columns=null encodes every utf8 column that reaches the step
//   * how many columns there are is known only after the fit, so a later
//     step cannot name them in the same pipeline
// ---------------------------------------------------------------------------
class ContrastEncode final : public Kernel {
 public:
  ContrastEncode(const nlohmann::json& params, Coding coding) : coding_(coding), what_(coding_id(coding)) {
    if (params.contains("columns") && !params.at("columns").is_null()) {
      columns_ = column_list(params, "columns", what_);
      explicit_columns_ = true;
      inputs = columns_;
    }
    unknown_ = unknown_from(params, what_.c_str(), true);
    missing_ = missing_from(params, what_.c_str());
  }

  std::string id() const override { return what_; }

  nlohmann::json params() const override {
    return {{"columns", explicit_columns_ ? nlohmann::json(columns_) : nlohmann::json()},
            {"handle_unknown", unknown_name(unknown_)},
            {"handle_missing", missing_name(missing_)}};
  }

  void resolve(Schema& schema) override {
    std::string verb = what_;
    std::replace(verb.begin(), verb.end(), '_', '-');
    bind_columns(schema, stage, columns_, explicit_columns_, fitted, verb.c_str());
    // Until the fit has run there are no levels, so the encoder consumes its
    // columns and emits nothing; the engine resolves again afterwards.
    outputs.clear();
    if (fitted) {
      for (const std::string& name : columns_) {
        for (const std::string& output : names_.at(name)) outputs.push_back(output);
      }
    }
    // The encoded columns leave first, so a generated name only collides with
    // a column that actually survives the step; hash_encode erases in the
    // same order, and encoding 'a' beside 'a_0' has to mean the same thing.
    std::erase_if(schema.cols, [&](const auto& col) {
      return std::find(columns_.begin(), columns_.end(), col.first) != columns_.end();
    });
    reject_duplicate_outputs(outputs, stage);
    for (const std::string& name : outputs) {
      if (schema.find(name) != nullptr) throw Error(stage + ": output column '" + name + "' already exists in input");
    }
    for (const std::string& name : outputs) schema.cols.emplace_back(name, "f64");
    inputs = columns_;
  }

  bool needs_fit() const override { return true; }
  std::unique_ptr<AccState> make_state() const override { return std::make_unique<SeenState>(); }

  void consume(AccState& raw, const Batch& batch) const override {
    consume_categories(columns_, stage, static_cast<SeenState&>(raw), batch);
  }

  void finalize(std::vector<std::unique_ptr<AccState>> states) override {
    tables_ = merge_categories(columns_, stage, states, missing_);
    build_matrices();
    fitted = true;
  }

  nlohmann::json state() const override {
    auto result = categories_state(columns_, tables_);
    if (coding_ == Coding::Polynomial) {
      result["polynomial_basis"] = {{"version", 1}, {"matrices", matrices_}};
    }
    return result;
  }

  // Save matrix rows incrementally: the fitted lookup table can be large,
  // and serialization need not build a second full matrix in a JSON DOM.
  void write_state(std::ostream& stream) const override {
    auto categories = categories_state(columns_, tables_).dump();
    if (coding_ != Coding::Polynomial) {
      stream << categories;
      return;
    }
    categories.pop_back();
    stream << categories << ",\"polynomial_basis\":{\"matrices\":{";
    auto columns = columns_;
    std::sort(columns.begin(), columns.end());
    bool first = true;
    for (const auto& name : columns) {
      if (!first) stream << ',';
      first = false;
      stream << nlohmann::json(name).dump() << ":[";
      bool first_row = true;
      for (const auto& row : matrices_.at(name)) {
        if (!first_row) stream << ',';
        first_row = false;
        stream << nlohmann::json(row).dump();
      }
      stream << ']';
    }
    stream << "},\"version\":1}}";
  }

  void load_state(const nlohmann::json& state) override {
    tables_ = load_categories(state, stage, columns_, explicit_columns_);
    const nlohmann::json* saved_matrices = nullptr;
    if (coding_ == Coding::Polynomial) {
      if (!state.contains("polynomial_basis")) throw Error(stage + ": state must hold a 'polynomial_basis' object");
      const auto& basis = state.at("polynomial_basis");
      if (!basis.is_object() || !basis.contains("version") || !basis.at("version").is_number_integer() ||
          basis.at("version") != 1 || !basis.contains("matrices") || !basis.at("matrices").is_object()) {
        throw Error(stage + ": invalid polynomial_basis: expected version 1 and a matrices object");
      }
      saved_matrices = &basis.at("matrices");
      if (saved_matrices->size() != columns_.size()) {
        throw Error(stage + ": polynomial_basis matrices must match the fitted columns");
      }
      for (const auto& name : columns_) {
        if (!saved_matrices->contains(name)) {
          throw Error(stage + ": polynomial_basis is missing column '" + name + "'");
        }
      }
    }
    build_matrices(saved_matrices);
    inputs = columns_;
    fitted = true;
  }

  Batch transform(Batch batch) const override {
    Batch out;
    out.rows = batch.rows;
    for (std::size_t i = 0; i < batch.names.size(); ++i) {
      if (std::find(columns_.begin(), columns_.end(), batch.names[i]) != columns_.end()) continue;
      out.names.push_back(std::move(batch.names[i]));
      out.cols.push_back(std::move(batch.cols[i]));
    }
    for (const std::string& name : columns_) {
      const int idx = batch.find(name);
      if (idx < 0) throw Error(stage + ": column '" + name + "' missing from batch");
      const arrow::Array& array = *batch.cols[static_cast<std::size_t>(idx)];
      const std::vector<std::string>& names = names_.at(name);
      const std::vector<std::vector<double>>& matrix = matrices_.at(name);
      const Categories& table = tables_.at(name);
      const std::string where = stage + ": column '" + name + "'";
      auto validity = std::make_shared<OutputValidity>(array.length(), where);
      // Two ways to write a batch. Row at a time writes each resolved row across
      // all of its columns, into buffers zero-filled up front, so a row that
      // writes nothing costs nothing. Column at a time resolves every row
      // first and then fills each column front to back; it stores every value,
      // but a row's stores into many columns evict each other from the cache,
      // which made 31 columns about five times slower row at a time. Measured
      // at 1 to 31 columns, row at a time wins once nulls that write nothing
      // are more than half the batch, up to 10 columns; wider, where its
      // stores collide, only while the coded rows times the width stay under
      // 1.5 times the batch (from 90% nulls at 15 columns, 95% at 31). Unknown
      // values are only found while resolving, so a batch of them is handled
      // below instead.
      const int64_t rows = array.length();
      const int64_t empty_nulls = null_writes_nothing(table) ? array.null_count() : 0;
      const auto width = static_cast<int64_t>(names.size());
      const bool row_at_a_time = width <= 10 ? empty_nulls * 2 > rows : (rows - empty_nulls) * width * 2 < rows * 3;
      std::vector<PrimitiveOutput<arrow::DoubleType>> columns;
      columns.reserve(names.size());
      for (std::size_t c = 0; c < names.size(); ++c) columns.emplace_back(rows, where, row_at_a_time, validity);
      // Either way every row is resolved, even when nothing is emitted:
      // handle_unknown='error' still refuses an unseen category in a
      // single-level column, as upstream does.
      if (row_at_a_time) {
        write_rows(array, name, table, matrix, columns, *validity);
      } else {
        // The buffers start uninitialized. The gather reads the zero row and a
        // null row from the 0 after each column's values; the mostly-zero
        // path zero-fills instead. Either way a null keeps +0.0 under its
        // cleared validity bit.
        const std::vector<double>& by_column = by_columns_.at(name);
        const auto zero_row = static_cast<uint32_t>(matrix.size());
        std::vector<uint32_t> level_of(names.empty() ? 0 : static_cast<std::size_t>(rows));
        for_each_encoded_utf8(
            array, stage, name,
            [&](bool is_null, std::string_view value) { return level(name, table, is_null, value); },
            [&](int64_t row, const std::optional<int64_t>& which) {
              if (names.empty()) return;
              if (!which.has_value()) validity->set_null(row);
              // -1 is the zero row: an unknown under 'value', or an unfitted null
              level_of[static_cast<std::size_t>(row)] =
                  which.has_value() && *which >= 0 ? static_cast<uint32_t>(*which) : zero_row;
            });
        // A batch whose rows resolved mostly to the zero row (unknowns, or
        // nulls too few to choose row at a time) is zero-filled and only its
        // other rows written: at 31 columns and 1M rows the gather took 1.3
        // times as long on a column of unknowns. It was no slower from 7 rows
        // in 8 beyond 10 columns; at 4 to 10 it lost below 15 rows in 16, and
        // under 4 columns it saved little. 64 evenly spaced rows decide, since
        // counting every row cost dense batches up to 6%; the answer only
        // changes the speed.
        bool mostly_zero = false;
        if (names.size() >= 4 && !level_of.empty()) {
          std::size_t sampled = 0;
          for (std::size_t i = 0; i < 64; ++i) sampled += level_of[i * level_of.size() / 64] == zero_row ? 1 : 0;
          mostly_zero = sampled > (names.size() > 10 ? 56U : 60U);
        }
        std::vector<std::size_t> coded_rows;
        if (mostly_zero) {
          for (std::size_t row = 0; row < level_of.size(); ++row) {
            if (level_of[row] != zero_row) coded_rows.push_back(row);
          }
        }
        for (std::size_t c = 0; c < names.size(); ++c) {
          const double* by_level = by_column.data() + c * (matrix.size() + 1);
          double* values = columns[c].mutable_values();
          if (mostly_zero) {
            std::fill_n(values, level_of.size(), 0.0);
            for (const std::size_t row : coded_rows) values[row] = by_level[level_of[row]];
          } else {
            for (std::size_t row = 0; row < level_of.size(); ++row) values[row] = by_level[level_of[row]];
          }
        }
      }
      for (std::size_t c = 0; c < names.size(); ++c) {
        out.names.push_back(names[c]);
        out.cols.push_back(columns[c].finish());
      }
    }
    return out;
  }

  Row transform_row(Row row) const override {
    RecordLookup lookup(row, columns_.size());
    Row out;
    // Sized from what the fit froze, never from `outputs`: resolve() rewrites
    // that member under a mutex while this may be running on another thread.
    out.reserve(row.size() + generated_);
    for (auto& [key, value] : row) {
      if (std::find(columns_.begin(), columns_.end(), key) != columns_.end()) continue;
      // A passthrough field named like a generated column would be dropped
      // silently; the batch path refuses the same shape at resolve.
      if (output_index_.contains(key)) throw Error(stage + ": key '" + key + "' already present in record");
      out.emplace_back(key, std::move(value));
    }
    for (const std::string& name : columns_) {
      const RowValue* value = lookup.find(name);
      if (value == nullptr) throw Error(stage + ": key '" + name + "' missing from record");
      const bool is_null = std::holds_alternative<std::monostate>(*value);
      const std::string* text = std::get_if<std::string>(value);
      if (!is_null && text == nullptr) throw Error(stage + ": key '" + name + "' must be a string or null");
      const std::optional<int64_t> which = level(name, tables_.at(name), is_null, is_null ? std::string_view{} : *text);
      const std::vector<std::string>& names = names_.at(name);
      if (names.empty()) continue;
      const std::vector<std::vector<double>>& matrix = matrices_.at(name);
      for (std::size_t c = 0; c < names.size(); ++c) {
        if (!which.has_value()) {
          out.emplace_back(names[c], std::monostate{});
        } else if (*which < 0) {
          out.emplace_back(names[c], 0.0);
        } else {
          out.emplace_back(names[c], matrix[static_cast<std::size_t>(*which)][c]);
        }
      }
    }
    return out;
  }

 private:
  // Row at a time: each resolved row across all of its columns, into buffers
  // zero-filled up front, so a row that writes nothing costs nothing. Kept
  // out of line: in builds that inlined it beside the column-at-a-time loop,
  // that loop ran 2 to 7% slower on dictionary input with nulls.
  [[gnu::noinline]] void write_rows(const arrow::Array& array, const std::string& name, const Categories& table,
                                    const std::vector<std::vector<double>>& matrix,
                                    std::vector<PrimitiveOutput<arrow::DoubleType>>& columns,
                                    OutputValidity& validity) const {
    for_each_encoded_utf8(
        array, stage, name, [&](bool is_null, std::string_view value) { return level(name, table, is_null, value); },
        [&](int64_t row, const std::optional<int64_t>& which) {
          if (columns.empty()) return;
          if (!which.has_value()) {
            validity.set_null(row);
            return;
          }
          if (*which < 0) return;  // the zero row: an unknown under 'value', or an unfitted null
          const std::vector<double>& coded_row = matrix[static_cast<std::size_t>(*which)];
          for (std::size_t c = 0; c < columns.size(); ++c) columns[c].mutable_values()[row] = coded_row[c];
        });
  }

  // Whether a null input writes no values, coding to a null output or the
  // zero row: always under 'return_nan' (and 'error' refuses it anyway), and
  // under 'value' unless a null was fitted as a level of its own.
  bool null_writes_nothing(const Categories& table) const { return missing_ != Missing::Value || !table.has_null; }

  // Which row of the matrix a value takes: a level index, -1 for a row of
  // zeros, or nullopt for a row of nulls.
  std::optional<int64_t> level(const std::string& name, const Categories& table, bool is_null,
                               std::string_view value) const {
    if (is_null) {
      if (missing_ == Missing::ReturnNan) return std::nullopt;
      refuse_null(missing_, stage, name);
      if (table.has_null) return static_cast<int64_t>(table.sorted.size());
      return -1;  // a null was never a fitted level, not even the indicator one
    }
    const int64_t rank = table.rank(value);
    if (rank >= 0) return rank;
    switch (unknown_) {
      case Unknown::Value:
        return -1;
      case Unknown::ReturnNan:
        return std::nullopt;
      case Unknown::Indicator:
        // The extra level, appended after the placeholder one.
        return static_cast<int64_t>(table.levels() + missing_placeholder(missing_));
      case Unknown::Error:
        break;
    }
    throw Error(stage + ": column '" + name + "' has a category that was not seen at fit");
  }

  // One matrix and one output-name list per column, both fixed by the level
  // count: the fitted categories, the null level when there was one, the
  // pre-encoder's placeholder level when handle_missing keeps it, and the
  // indicator level when handle_unknown asks for it.
  void build_matrices(const nlohmann::json* saved_matrices = nullptr) {
    names_.clear();
    matrices_.clear();
    by_columns_.clear();
    output_index_.clear();
    generated_ = 0;
    for (const std::string& name : columns_) {
      std::vector<std::string> generated;
      std::vector<std::vector<double>> matrix;
      const std::size_t fitted_levels = tables_.at(name).levels();
      const std::size_t coded_levels = fitted_levels + missing_placeholder(missing_);
      const std::size_t levels = coded_levels >= 2 ? coded_levels + (unknown_ == Unknown::Indicator ? 1 : 0) : 0;
      // One level has nothing to contrast against; upstream drops the column
      // and never reaches the indicator, so neither do we. It counts the
      // placeholder level here, which is why a single category still emits a
      // column under 'return_nan' and 'error'.
      if (coded_levels >= 2) {
        check_output_width(levels - 1, fitted_levels, stage, name);
        if (saved_matrices == nullptr) matrix = contrast_rows(coding_, levels, stage, name);
        for (std::size_t i = 0; i + 1 < levels; ++i) {
          std::string output = name + "_" + std::to_string(i);
          output_index_.emplace(output, generated_++);
          generated.push_back(std::move(output));
        }
      }
      if (saved_matrices != nullptr) {
        const auto& rows = saved_matrices->at(name);
        const std::string where = stage + ": polynomial_basis column '" + name + "'";
        if (!rows.is_array() || rows.size() != levels) throw Error(where + " has an invalid row count");
        matrix.reserve(levels);
        for (const auto& row : rows) {
          if (!row.is_array() || row.size() != levels - 1) throw Error(where + " has an invalid row width");
          std::vector<double> values;
          values.reserve(levels - 1);
          for (const auto& cell : row) {
            if (!cell.is_number() || !std::isfinite(cell.get<double>())) {
              throw Error(where + " must contain finite numbers");
            }
            values.push_back(cell.get<double>());
          }
          matrix.push_back(std::move(values));
        }
      }
      // The same values by column, each column's followed by a 0 for the
      // zero row, for the batch gather: a level's row and then its column
      // would be two dependent loads for every value.
      const std::size_t stride = matrix.size() + 1;
      std::vector<double> by_column(generated.size() * stride, 0.0);
      for (std::size_t level = 0; level < matrix.size(); ++level) {
        for (std::size_t c = 0; c < generated.size(); ++c) by_column[c * stride + level] = matrix[level][c];
      }
      names_.emplace(name, std::move(generated));
      matrices_.emplace(name, std::move(matrix));
      by_columns_.emplace(name, std::move(by_column));
    }
  }

  Coding coding_ = Coding::BackwardDifference;
  std::string what_;
  std::vector<std::string> columns_;
  bool explicit_columns_ = false;
  Unknown unknown_ = Unknown::Value;
  Missing missing_ = Missing::Value;
  std::unordered_map<std::string, Categories> tables_;
  std::unordered_map<std::string, std::vector<std::string>> names_;
  std::unordered_map<std::string, std::vector<std::vector<double>>> matrices_;
  std::unordered_map<std::string, std::vector<double>> by_columns_;  // the same, by column
  // Both frozen by the fit, so the record path never reads a resolve-owned
  // member: the generated names, and how many of them there are.
  StringIndex output_index_;
  std::size_t generated_ = 0;
};

}  // namespace

std::unique_ptr<Kernel> make_backward_difference_encode(const nlohmann::json& params) {
  return std::make_unique<ContrastEncode>(params, Coding::BackwardDifference);
}

std::unique_ptr<Kernel> make_helmert_encode(const nlohmann::json& params) {
  return std::make_unique<ContrastEncode>(params, Coding::Helmert);
}

std::unique_ptr<Kernel> make_sum_encode(const nlohmann::json& params) {
  return std::make_unique<ContrastEncode>(params, Coding::Sum);
}

std::unique_ptr<Kernel> make_polynomial_encode(const nlohmann::json& params) {
  return std::make_unique<ContrastEncode>(params, Coding::Polynomial);
}

}  // namespace fastfeaturetransform
