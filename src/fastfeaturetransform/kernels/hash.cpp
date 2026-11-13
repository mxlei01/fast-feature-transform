// hash_encode — the hashing trick: every encoded column collapses into one
// shared block of n_components counters. The only encoder with no fitted
// state at all: the bucket of a value is a pure function of its bytes, so
// the encoder never grows, never learns a vocabulary, and accepts values it
// has never seen. The output width comes from the params, so the artifact
// carries nothing beyond them.

#include <algorithm>
#include <array>
#include <cstring>

#include "fastfeaturetransform/kernels/cast_table.hpp"
#include "fastfeaturetransform/kernels/categories.hpp"
#include "fastfeaturetransform/kernels/kernels.hpp"
#include "fastfeaturetransform/kernels/output.hpp"
#include "fastfeaturetransform/kernels/util.hpp"

namespace fastfeaturetransform {
namespace {

// --- md5 (RFC 1321) --------------------------------------------------------
// Vendored, in about sixty lines, rather than linked from a crypto library:
// upstream's bucket is int.from_bytes(hashlib.md5(value).digest(), 'big')
// % n_components, so the digest has to be byte-for-byte hashlib's, and this
// is the whole of what the kernel needs. It spreads values over buckets; it
// is not used for anything that needs a secure hash.

// The per-round left rotations and the sine-derived additive constants,
// floor(abs(sin(i + 1)) * 2^32).
constexpr std::array<uint32_t, 64> kRotations = {7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
                                                 5, 9,  14, 20, 5, 9,  14, 20, 5, 9,  14, 20, 5, 9,  14, 20,
                                                 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
                                                 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21};
constexpr std::array<uint32_t, 64> kSines = {
    0xd76aa478u, 0xe8c7b756u, 0x242070dbu, 0xc1bdceeeu, 0xf57c0fafu, 0x4787c62au, 0xa8304613u, 0xfd469501u,
    0x698098d8u, 0x8b44f7afu, 0xffff5bb1u, 0x895cd7beu, 0x6b901122u, 0xfd987193u, 0xa679438eu, 0x49b40821u,
    0xf61e2562u, 0xc040b340u, 0x265e5a51u, 0xe9b6c7aau, 0xd62f105du, 0x02441453u, 0xd8a1e681u, 0xe7d3fbc8u,
    0x21e1cde6u, 0xc33707d6u, 0xf4d50d87u, 0x455a14edu, 0xa9e3e905u, 0xfcefa3f8u, 0x676f02d9u, 0x8d2a4c8au,
    0xfffa3942u, 0x8771f681u, 0x6d9d6122u, 0xfde5380cu, 0xa4beea44u, 0x4bdecfa9u, 0xf6bb4b60u, 0xbebfbc70u,
    0x289b7ec6u, 0xeaa127fau, 0xd4ef3085u, 0x04881d05u, 0xd9d4d039u, 0xe6db99e5u, 0x1fa27cf8u, 0xc4ac5665u,
    0xf4292244u, 0x432aff97u, 0xab9423a7u, 0xfc93a039u, 0x655b59c3u, 0x8f0ccc92u, 0xffeff47du, 0x85845dd1u,
    0x6fa87e4fu, 0xfe2ce6e0u, 0xa3014314u, 0x4e0811a1u, 0xf7537e82u, 0xbd3af235u, 0x2ad7d2bbu, 0xeb86d391u};

constexpr uint32_t rotate_left(uint32_t value, uint32_t bits) {
  return static_cast<uint32_t>((value << bits) | (value >> (32 - bits)));
}

// One 64-byte block folded into the running state: four rounds of sixteen
// steps. Each round is its own fully unrolled loop, so every word index and
// rotation is a constant. One loop choosing the round per step compiled to
// all four round functions, selects and variable rotations on every step,
// and took about twice as long; clang left three of four plain round loops
// rolled, hence the pragmas.
void md5_block(std::array<uint32_t, 4>& state, const uint8_t* chunk) {
  std::array<uint32_t, 16> words{};
  for (std::size_t i = 0; i < 16; ++i) {
    words[i] = static_cast<uint32_t>(chunk[i * 4]) | static_cast<uint32_t>(chunk[i * 4 + 1] << 8) |
               static_cast<uint32_t>(chunk[i * 4 + 2] << 16) | static_cast<uint32_t>(chunk[i * 4 + 3] << 24);
  }
  uint32_t a = state[0];
  uint32_t b = state[1];
  uint32_t c = state[2];
  uint32_t d = state[3];
  const auto step = [&](std::size_t i, uint32_t mixed, std::size_t word) {
    const uint32_t round = a + mixed + kSines[i] + words[word];
    a = d;
    d = c;
    c = b;
    b = b + rotate_left(round, kRotations[i]);
  };
#pragma GCC unroll 16
  for (std::size_t i = 0; i < 16; ++i) step(i, (b & c) | (~b & d), i);
#pragma GCC unroll 16
  for (std::size_t i = 16; i < 32; ++i) step(i, (d & b) | (~d & c), (5 * i + 1) % 16);
#pragma GCC unroll 16
  for (std::size_t i = 32; i < 48; ++i) step(i, b ^ c ^ d, (3 * i + 5) % 16);
#pragma GCC unroll 16
  for (std::size_t i = 48; i < 64; ++i) step(i, c ^ (b | ~d), (7 * i) % 16);
  state[0] += a;
  state[1] += b;
  state[2] += c;
  state[3] += d;
}

std::array<uint8_t, 16> md5(std::string_view value) {
  std::array<uint32_t, 4> state{0x67452301u, 0xefcdab89u, 0x98badcfeu, 0x10325476u};
  const auto* bytes = reinterpret_cast<const uint8_t*>(value.data());  // NOLINT(*-reinterpret-cast): bytes of a string
  const std::size_t blocks = value.size() / 64;
  for (std::size_t i = 0; i < blocks; ++i) md5_block(state, bytes + i * 64);
  // The tail: whatever is left, a 0x80 byte, zeros, and the message length
  // in bits as a little-endian 64-bit integer — two blocks when the
  // remainder leaves no room for that length.
  std::array<uint8_t, 128> tail{};
  const std::size_t rest = value.size() - blocks * 64;
  if (rest > 0) std::memcpy(tail.data(), bytes + blocks * 64, rest);
  tail[rest] = 0x80;
  const std::size_t padded = rest < 56 ? 64U : 128U;
  const uint64_t bits = static_cast<uint64_t>(value.size()) * 8;
  for (std::size_t i = 0; i < 8; ++i) tail[padded - 8 + i] = static_cast<uint8_t>(bits >> (8 * i));
  md5_block(state, tail.data());
  if (padded == 128) md5_block(state, tail.data() + 64);
  // The digest is the four state words, each little-endian.
  std::array<uint8_t, 16> digest{};
  for (std::size_t i = 0; i < 4; ++i) {
    for (std::size_t byte = 0; byte < 4; ++byte) {
      digest[i * 4 + byte] = static_cast<uint8_t>(state[i] >> (8 * byte));
    }
  }
  return digest;
}

// More buckets than this would be more output columns than any table wants,
// and every bucket costs a column of counters per batch.
constexpr int64_t kMaxComponents = 4096;

// Each encoded column of a batch remembers at most one distinct value per
// this many rows. A miss and insert costs a fraction of a digest, so a column
// of distinct values pays about a percent for the attempt, while a column
// that fits skips the digest of every repeated value.
constexpr int64_t kRowsPerRemembered = 8;
// And at most this many whatever the batch size: a few MiB of table.
constexpr std::size_t kMaxRemembered = std::size_t{1} << 17;

// ---------------------------------------------------------------------------
// hash_encode — the hashing trick: utf8 columns -> one shared block of
// n_components integer counters. Stateless.
// params: {"columns": [...], "n_components": int, "hash_method": "md5"}
// Semantics:
//   * for every row, each encoded value is hashed with md5 and the counter
//     int.from_bytes(digest, 'big') % n_components is incremented, so a row
//     with k non-null encoded values sums to k across the block
//   * the encoded columns all feed one shared block: they are dropped and
//     col_0 .. col_{n_components - 1} appended after the columns that stay
//   * only utf8 columns can be hashed, and the hashed bytes are the value's
//     own — cast a number to utf8 first, which spells an integer as decimal
//     digits, exactly what upstream's str() gives
//   * a null contributes nothing to any counter
//   * the outputs are i64 and their names are fixed, so two hash encoders
//     cannot share one pipeline; the collision is a resolve-time error
//   * nothing is learned: needs_fit() is false, the state is empty, and an
//     unseen value is no different from a seen one
// ---------------------------------------------------------------------------
class HashEncode final : public Kernel {
 public:
  explicit HashEncode(const nlohmann::json& params) : columns_(column_list(params, "columns", "hash_encode")) {
    if (params.contains("n_components") && !params.at("n_components").is_null()) {
      const nlohmann::json& components = params.at("n_components");
      if (!components.is_number_integer() ||
          (components.is_number_unsigned() && components.get<uint64_t>() > static_cast<uint64_t>(kMaxComponents)) ||
          components.get<int64_t>() < 1 || components.get<int64_t>() > kMaxComponents) {
        throw Error("hash_encode: 'n_components' must be an integer between 1 and " + std::to_string(kMaxComponents));
      }
      components_ = static_cast<std::size_t>(components.get<int64_t>());
    }
    if (params.contains("hash_method") && !params.at("hash_method").is_null()) {
      if (!params.at("hash_method").is_string() || params.at("hash_method").get<std::string>() != "md5") {
        throw Error("hash_encode: 'hash_method' must be 'md5'; this runtime hashes with md5 only");
      }
    }
    power_of_two_ = (components_ & (components_ - 1)) == 0;
    // Known without a schema: the columns are named and the width is a
    // param, so an artifact-loaded encoder (which carries no state) still
    // declares what it reads and writes.
    for (std::size_t i = 0; i < components_; ++i) {
      buckets_.push_back("col_" + std::to_string(i));
      bucket_index_.emplace(buckets_.back(), i);
    }
    inputs = columns_;
    outputs = buckets_;
  }

  std::string id() const override { return "hash_encode"; }

  nlohmann::json params() const override {
    return {{"columns", columns_}, {"n_components", components_}, {"hash_method", "md5"}};
  }

  void resolve(Schema& schema) override {
    // `columns` is required here, so there is nothing to derive: this only
    // checks that every listed column is present and utf8.
    bind_columns(schema, stage, columns_, true, false, "hash");
    // Every encoded column is replaced by the one shared block.
    std::erase_if(schema.cols, [&](const auto& col) { return encoded(col.first); });
    for (const std::string& name : buckets_) {
      if (schema.find(name) != nullptr) {
        throw Error(stage + ": output column '" + name +
                    "' already exists in input; hash_encode always names its outputs col_0..col_" +
                    std::to_string(components_ - 1) + ", so two hash encoders cannot share one pipeline");
      }
      schema.cols.emplace_back(name, "i64");
    }
    inputs = columns_;
    outputs = buckets_;
  }

  // Nothing is learned, so state() stays empty; a manifest that carries a
  // table for this stage was hand-edited and is refused rather than ignored.
  void load_state(const nlohmann::json& state) override {
    if (!state.is_object() || !state.empty()) {
      throw Error(stage + ": hash_encode learns nothing, so its 'state' must be an empty object");
    }
  }

  Batch transform(Batch batch) const override {
    const int64_t rows = column_array(batch, stage, columns_.front()).length();
    std::vector<PrimitiveOutput<arrow::Int64Type>> counters;
    counters.reserve(components_);
    for (const std::string& bucket : buckets_) counters.emplace_back(rows, stage + ": column '" + bucket + "'", true);
    for (const std::string& name : columns_) {
      const arrow::Array& array = column_array(batch, stage, name);
      // A digest costs far more than a table lookup and categorical values
      // repeat, so a plain utf8 column remembers the bucket of each value it
      // hashes (a dictionary no larger than the batch is already hashed once
      // per entry). The keys view this batch's own strings, so the table lives
      // for one column of one call. A column with more distinct values than
      // the limit repeats too little for lookups to pay, and the rest of it is
      // hashed directly.
      ankerl::unordered_dense::map<std::string_view, std::size_t> remembered;
      const std::size_t limit = array.type_id() == arrow::Type::DICTIONARY
                                    ? 0
                                    : std::min(static_cast<std::size_t>(rows / kRowsPerRemembered), kMaxRemembered);
      // Sized once: growing it rehashed every key, three quarters of what a
      // column of distinct values paid for the table, and the sparse table
      // answers a repeat about three times faster than one grown to fit.
      remembered.reserve(limit);
      bool remembering = limit > 0;
      for_each_encoded_utf8(
          array, stage, name,
          [&](bool is_null, std::string_view value) {
            if (is_null) return components_;
            if (!remembering) return bucket_of(value);
            const auto [entry, inserted] = remembered.try_emplace(value, components_);
            if (inserted) {
              entry->second = bucket_of(value);
              remembering = remembered.size() < limit;
            }
            return entry->second;
          },
          [&](int64_t row, std::size_t bucket) {
            if (bucket != components_) ++counters[bucket].mutable_values()[row];
          });
    }
    drop_encoded(batch);
    for (std::size_t i = 0; i < components_; ++i) {
      batch.names.push_back(buckets_[i]);
      batch.cols.push_back(counters[i].finish());
    }
    return batch;
  }

  Row transform_row(Row row) const override {
    std::vector<int64_t> counters(components_, 0);
    {
      RecordLookup lookup(row, columns_.size());
      for (const std::string& name : columns_) {
        const RowValue* value = lookup.find(name);
        if (value == nullptr) throw Error(stage + ": key '" + name + "' missing from record");
        if (std::holds_alternative<std::monostate>(*value)) continue;
        const std::string* text = std::get_if<std::string>(value);
        if (text == nullptr) throw Error(stage + ": key '" + name + "' must be a string or null");
        ++counters[bucket_of(*text)];
      }
    }
    // The encoded keys leave first, so a column named like a bucket is only
    // a collision when it is not itself encoded.
    std::erase_if(row, [&](const auto& entry) { return encoded(entry.first); });
    for (const auto& [name, value] : row) {
      if (bucket_index_.contains(name)) throw Error(stage + ": key '" + name + "' already present in record");
    }
    row.reserve(row.size() + components_);
    for (std::size_t i = 0; i < components_; ++i) row.emplace_back(buckets_[i], counters[i]);
    return row;
  }

 private:
  // The bucket one value falls in: its md5 digest read as one big-endian
  // 128-bit integer, modulo the block width (upstream's int.from_bytes).
  std::size_t bucket_of(std::string_view value) const {
    const std::array<uint8_t, 16> digest = md5(value);
    if (power_of_two_) {
      // At most 4096 components: a power-of-two remainder uses only the low
      // twelve bits of the big-endian digest, including the one-bucket case.
      const auto low = static_cast<std::size_t>(digest[14]) * 256 + digest[15];
      return low & (components_ - 1);
    }
    UInt128 number = 0;
    for (uint8_t byte : digest) number = (number << 8) | static_cast<UInt128>(byte);
    return static_cast<std::size_t>(number % static_cast<UInt128>(components_));
  }

  bool encoded(const std::string& name) const {
    return std::find(columns_.begin(), columns_.end(), name) != columns_.end();
  }

  void drop_encoded(Batch& batch) const {
    Batch kept;
    kept.rows = batch.rows;
    for (std::size_t i = 0; i < batch.names.size(); ++i) {
      if (encoded(batch.names[i])) continue;
      kept.names.push_back(std::move(batch.names[i]));
      kept.cols.push_back(std::move(batch.cols[i]));
    }
    batch = std::move(kept);
  }

  std::vector<std::string> columns_;
  std::size_t components_ = 8;
  bool power_of_two_ = true;
  std::vector<std::string> buckets_;  // col_0 .. col_{components_ - 1}
  StringIndex bucket_index_;
};

}  // namespace

std::unique_ptr<Kernel> make_hash_encode(const nlohmann::json& params) { return std::make_unique<HashEncode>(params); }

}  // namespace fastfeaturetransform
