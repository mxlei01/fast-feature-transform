# Changelog

## 0.0.3

Pipelines and pickles saved by 0.0.1 or 0.0.2 load in this version. Version 0.0.2 can also load pipelines and pickles
saved by 0.0.3. The version running a transform determines its behavior in the corrected cases below, including
normalization near the limit of f64. Measurements compare 0.0.3 development builds with 0.0.2 on one thread of an
Apple M4.

Fixed:

- Reading a nested string or binary column with an out-of-range dictionary index raises an `EngineError` naming the
  column. In 0.0.2, transforming such a file could crash the process or write wrong values, and fitting could succeed.
- A repeated entry in a string or binary dictionary page can shift subsequent values when Arrow decodes the indices.
  This version checks the page and raises an `EngineError` naming the column when the decoded dictionary loses a repeat.
- `transform_dict` retains record fields before invoking a value's Python conversion, so a conversion that changes
  the input dictionary no longer causes skipped fields or a read after the value was freed. Unrelated classes named
  like NumPy scalars get the unsupported-type error, and passing something other than a dict names the public method.
- Record keys with custom hashing or equality are checked for repeated text. A key whose hash changes after insertion,
  or another thread changing the dict on a free-threaded build, can still leave a repeated key undetected.
- Concurrent fitted-attribute reads, transforms, copies and pickles see a consistent fit. In particular,
  `MinMaxScaler.data_range_` takes both endpoints from one fit, and an unpickled pipeline retains its fitted state
  when another thread first uses it.
- `StandardScaler` now fits columns such as `[9e153, 8e153]` and rejects a sum of squares that rounds to infinity
  in every row order and worker split. `Normalizer` applies the same overflow rule to its l1 or l2 sum in every
  column order.
- `KBinsDiscretizer(strategy="kmeans")` checks the sum of squared deviations at the original scale. When that sum
  overflows, it stops after the first center update, matching scikit-learn even when distances are scaled internally.
- `WOEEncoder` distinguishes integer target values above 2^53. `JamesSteinEncoder`, `LeaveOneOutEncoder` and
  `CatBoostEncoder` sum integer targets exactly, and the latter two accept larger finite float targets.
- Files can be read together when the same string or binary column uses large offsets or view types in different
  Parquet encodings, including delta pages and nested columns. Top-level `string_view` in delta pages is read as
  `utf8` by string transformers and `Cast`.
- Row-group minimum and maximum statistics for nullable string and binary dictionary columns use values present in
  the rows. Numbered output parts sort in row order even when a transform writes more than 100,000 files.
- Directory arguments accept text paths and path-like objects and reject embedded NULs. Batch calls validate input
  paths before creating temporary output and remove their own temporary directory if an exception or interrupt
  prevents them from returning it. An empty `temp_dir` still selects the system temporary directory.

Changed:

- Reading an Arrow extension column backed by a dictionary raises an `EngineError` before decoding. Selecting only
  other columns still works.
- Loading a saved manifest rejects missing required fitted state, invalid nested stage names and malformed handling
  options. On a new `CountEncoder` fit, `min_group_name` and nondefault `combine_min_nan_groups` require
  `min_group_size`. Older saved encoders with those options still load.
- String subclasses used as names or string options are stored as plain strings. Numeric options accept compatible
  NumPy scalars, and pipeline indexing accepts integer-like values.
- `JamesSteinEncoder`, `LeaveOneOutEncoder` and `CatBoostEncoder` expose fitted `null_values_`. Python annotations
  cover tuples of column names and distinguish file lists from a single path.
- The writer flushes a row group after six buffered batches or when another batch of the previous size would take
  its buffers past 36 MiB. On 8,192 rows with 1,000 f64 columns and distinct 20,000-character strings, peak memory
  fell from 585 MiB to 436–453 MiB and the writer produced 12 row groups instead of two.
- Wheels use pinned CMake and ninja versions. The Linux and macOS extensions export only the Python module initializer.

Faster:

- Looking up 30 long field names in reverse or shuffled order took 89% or 64% of the time of the 0.0.2 wheel,
  respectively. Loading a saved `Cast` with 20,000 columns took 9.44 ms instead of 16.03 ms.
- On a million rows in 16 row groups, fitting a `TargetEncoder` over a million distinct keys took 288 ms instead of
  357 ms. Leave-one-out and CatBoost fits over 1,000 categories with integer targets took 7.2 ms instead of 12.3 ms.

Costs more:

- Strings inside fixed-size lists are decoded without a dictionary. Passing through 500,000 rows of four 64-byte
  strings took 79–95 ms instead of 42 ms with no null rows, and about 60% of the previous time with 30% null rows.
- A top-level `string_view` column in delta pages is copied while reading. Passing through 1,048,576 eight-byte
  strings took 30.5 ms instead of 24.5 ms with 90% nulls.
- The wide output with 20,000-character strings above took 333 ms instead of 286 ms. With 32,768 rows of distinct
  1,000-character dictionary strings beside 1,000 f64 columns, writing took about 18% longer and used 2% more bytes.
- The million-key `TargetEncoder` fit above peaked at 346 MiB instead of 300 MiB. Fitting `JamesSteinEncoder` on a
  million integer targets above 2^53 over 1,000 categories took 29.4 ms instead of 16.6 ms.

## 0.0.2

Faster fits, faster and smaller Parquet output, and fixes. A pipeline saved by 0.0.1 loads and answers as before.
Timings are from an Apple M4 on one thread.

Faster:

- Fits that sum floats take a half to a sixth of the time: `StandardScaler` 42.3 -> 6.5 ms per million rows, and
  `TargetEncoder`, `JamesSteinEncoder`, `LeaveOneOutEncoder`, `CatBoostEncoder`, `GroupAgg` and `PowerTransformer`
  with it. k-means bins take about 0.6 of the time.
- `transform_parquet` writes each part through a buffer and ends a dictionary of distinct integers at 248 KiB: an id
  and two floats take 60.3 -> 48.0 ms per million rows.
- A string column stored in plain pages passes through in bytes that grow with its rows, not with their square: a
  row group of a million distinct strings takes 664 -> 165 ms and 69.8 -> 16.7 MB.
- A very wide output gathers its small batches into row groups: a one-hot output of 1,000 columns and 131,072
  rows takes 4.0 -> 1.2 s and 45.4 -> 9.3 MB.
- `transform_dict` finds a step's columns in one comparison each when the record's keys are in the step's order: 30
  fields take 2.68 -> 1.84 µs. `Cast` takes 640 -> 443 ns for eight columns.
- Importing the package takes 10 ms where it took 20.
- A `CountEncoder` that folds rare categories saves each group once: 8,000 rare categories save 232 KB where they
  saved 576 MB.

Changed:

- A step given no columns takes the columns known when the fit starts. Columns whose number is known only after a
  fit, such as those `OneHotEncoder` generates, never join them, whichever pass the step fits in.
- A parameter name a kernel does not know is an error at fit and at load. 0.0.1 ignored it and used the default.
  Params that are not a JSON object are an error at load too.
- Parquet output has another layout where it is very wide, from about 170 float columns, or holds distinct
  integers: fewer and larger row groups, and plain pages after 248 KiB of dictionary. The values are the same. An
  integer column with 31,744 to about 55,000 distinct values in a row group is written up to 13% larger when its
  values are in random order, and up to 54% larger when its distinct values come first.
- While a very wide output is written, up to five more batches wait in memory for each thread: about 130 MiB for
  each thread with 1,000 float columns and one string of 20,000 characters in each row.
- A string option given as a subclass of `str`, such as an `Enum` with a `str` mixin, fits in every transformer and
  is stored as the plain string. Four options keep the value as given, so a pickle of the transformer needs its
  class: the type tags of `Cast`, the aggregates of `GroupAgg`, `hash_method` of `HashingEncoder` and
  `combine_min_nan_groups` of `CountEncoder`.
- A `CountEncoder` that folds rare categories reports and saves its groups as `{group name: [members]}`, in
  `stages_state()`, the manifest and pickles. 0.0.1 cannot load a pipeline saved by this version whose
  `CountEncoder` folded a group.
- Three calls take longer. A record that `Cast` refuses takes about 1.2 times as long to raise. A record of 30
  names of 23 bytes takes about 10% longer when its keys are not in the step's order. `Pipeline.load` of a `Cast`
  over 20,000 columns takes 1.3 times as long.

Fixed:

- Files that store the same `list<string>` column in different encodings are read together.
- `StandardScaler` and `JamesSteinEncoder` fit values near 1e-155 alike for every thread count and row-group split.
- A `transform_parquet` or `parquet_to_memmap` that fails no longer leaves an empty temporary directory behind.

## 0.0.1

The first release.

- A `Pipeline` fits on Parquet files, saves as one JSON manifest, and transforms one record (`transform_dict`) or
  Parquet files, writing Parquet parts (`transform_parquet`) or a packed file for NumPy (`transform_memmap`).
- Wheels for CPython 3.14, standard and free-threaded, on Linux x86_64, Linux arm64, and macOS 26 or later on arm64.
