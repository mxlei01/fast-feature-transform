# Changelog

## 0.0.3

Fixes for damaged Parquet files, for records and transformers that other code changes while they are read, and for
results that 0.0.2 got wrong or refused. A pipeline saved by 0.0.1 or 0.0.2 loads and answers as before. 0.0.2 loads a
pipeline or pickle saved by this version and answers alike. Timings are from an Apple M4 on one thread.

Fixed:

- A string column nested in a list, large list, fixed-size list, struct or map whose dictionary index lies outside
  its dictionary is an `EngineError` that names the column. On such a file 0.0.2 ended the process with a bus error in
  `transform_parquet`, or wrote wrong values for a fixed-size list with null rows, and a fit that read the column
  returned without an error.
- A record value whose own code takes it out of the record, such as its `__index__` lookup or its type's `__name__`,
  or that another thread replaces while such code runs, is read as it was; 0.0.2 read it after it was freed, which
  ended the process with a segmentation fault or read freed memory. A value of a class named `datetime64` that has
  `astype` but no `dtype` gets the `EngineError` of an unsupported type, and one whose `dtype` raises, or a datetime
  whose `asm8` has `astype` but no `dtype`, raises the error of that lookup; 0.0.2 ended the process with an abort on
  each.
- A copy, deep copy or pickle no longer raises `RuntimeError: dictionary changed size during iteration` when it is
  taken of a transformer while another thread fits it for the first time, or of an unpickled pipeline while another
  thread first uses it. On a free-threaded build 0.0.2 sometimes ended the process with a segmentation fault in the
  first case, and restored the pipeline unfitted in the second.
- A transformer fitted again while other threads use it answers for the old fit or the new one. In 0.0.2 some
  `transform_dict` calls of a standalone transformer raised "was fitted as part of a Pipeline", some pickles of it
  restored unfitted, and a fitted attribute read that the refit overtook left the old fit's state for the reads after
  it.
- A record that holds one key twice, such as a `str` subclass with its own hash beside a `str` of its text, or a key
  that a value's conversion takes out and stores again, is an `EngineError` "record holds the key '<name>' twice";
  0.0.2 answered with one of the two fields. A key whose hash changes after it is stored, and on a free-threaded build
  another thread that changes the record while it is read, can still repeat a key unchecked.
- A name or string option given as a subclass of `str`, such as a member of an `Enum` with a `str` mixin, is stored
  as the plain string it equals, so a pickle loads where the caller's class cannot be imported. In 0.0.2 a fitted
  attribute of a step nested under such step names raised "no stage named ...", and `PolynomialFeatures` and
  `SplineTransformer` listed outputs such as `Col.X^2` that they do not write; a pickle made by 0.0.2 keeps the names
  as given.
- `StandardScaler` fits columns whose squares sum within f64 that 0.0.2 refused with "the variance leaves the range
  of f64": [9e153, 8e153] gets the variance 2.500000000000001e+305, as in scikit-learn. A column whose squares sum
  within half a step of the largest f64 is still fitted in one order or split of its rows and refused in another.
- `KBinsDiscretizer(strategy="kmeans")` stops after its first pass, as scikit-learn does, when the squared deviations
  of a column sum past f64's range, which takes more than 2^24 values. 0.0.2 ran on after its first pass: on
  16,943,876 values near 2^500 in magnitude it gave other edges than scikit-learn at 2 and at 5 bins, where this
  version gives scikit-learn's.
- `WOEEncoder` counts the distinct values of an integer target as integers, as category_encoders does: a target of 0,
  2^62 and 2^62 + 1 is refused for holding more than two values, where 0.0.2 fitted every category to 0.0 or, on
  other rows, said a category's weight of evidence is not finite, and one of 2^62 and 2^62 + 1 for holding no 0, where
  0.0.2 said it holds one value.
- `JamesSteinEncoder`, `LeaveOneOutEncoder` and `CatBoostEncoder` sum an integer target exactly, as `TargetEncoder`
  does: `LeaveOneOutEncoder` encodes a category of the targets 2^53 + 1 and 1 to 4503599627370497.0, where 0.0.2 gave
  4503599627370496.0. `LeaveOneOutEncoder` and `CatBoostEncoder` take a float target below 2^900 in magnitude, which
  0.0.2 refused from 2^450, and `CatBoostEncoder` with `handle_missing="return_nan"` fits when only the null
  category's value, which it never keeps, would not be finite.
- A list, large list, fixed-size list, struct or map of `large_string` or `string_view` in delta pages is read
  together with the same column in dictionary or plain pages, and a top-level `string_view` column in delta pages is
  read as `utf8`, which every string step and `Cast` take. 0.0.2 refused each of these but a fixed-size list of
  `large_string`.
- `output_dir`, the path of `save` and `load`, and `temp_dir` are read through `os.fspath`: an `os.PathLike` names its
  own directory, and bytes, `None` where a directory is needed, other types, an empty `output_dir` or path, and a NUL
  are a `TypeError` or `ValueError` before anything is written. 0.0.2 wrote into `./b'out'`, `./None`, `./5` or a
  directory named by the value's `repr`, created the directory named by the text before a NUL or in `save` removed it,
  and read `/manifest.json` for `load("")`.

Changed:

- A manifest that lacks an entry of a step's state that every version writes, or holds null for one written as an
  object, is an `EngineError` at load: `has_null`, `null_values`, `null_counts`, `groups`, `vocab_sizes`, an output's
  entry in `nulls`, `means` or `scales`. So is a `select` or `cast` state other than `{}`; 0.0.2 loaded most of these
  manifests, and some answered differently: an `OrdinalEncoder` null became -2, a folded `CountEncoder` category 0.
- The batches that wait for a Parquet row group are bounded by 36 MiB as well as by six batches, so a wide output
  with long values holds less memory and is written in more row groups. A `Select` of 8,192 rows of 1,000 f64 columns
  and a distinct string of 20,000 characters peaks at 436 to 453 MiB where 0.0.2 peaked at 585 MiB, in 12 row groups
  where 0.0.2 wrote 2.
- An option that takes a fraction takes a numpy integer or float, an integer array of no dimensions and a `Fraction`,
  and `CountEncoder`'s `handle_unknown` a numpy integer and an integer array of no dimensions, as the options that
  take only an integer already did; each is stored as a plain int or float. 0.0.2 refused these but `float64`, and
  kept `np.float64(0.5)` as given for `CountEncoder`'s `min_group_size`.
- A number whose nearest f64 is infinite or 0 though it is neither, such as `10**400`, is a `ValueError` that names
  the option, where 0.0.2 raised `OverflowError` from most of the options that take a fraction.
- A string option, a `Cast` type tag and a `GroupAgg` aggregate take only a `str`, and any other value gets the
  option's `ValueError`: 0.0.2 raised the bare `TypeError` of hashing a list or numpy's error about the truth value of
  an array, and took `np.array(["md5"])` for `hash_method` and failed at fit. `pipeline[np.int64(0)]` returns the
  step, where 0.0.2 raised `KeyError`, and a key that is not a `str` and that `operator.index` refuses is a `KeyError`,
  even one that compares equal to a step name, which 0.0.2 found.
- `transform_dict` raises the error of a record value whose `__index__` raises something other than `TypeError`, whose
  lookup of an attribute such as `__index__` or a datetime's `asm8` raises something other than `AttributeError`, or
  whose type is named `bool` or `bool_` and whose `__bool__` raises or returns something other than a bool. 0.0.2
  raised the `EngineError` of an unsupported type or a `SystemError` instead, and answered such a datetime as if it
  had no `asm8`. A value of a class named `datetime64` without `astype` gets the `EngineError` of an unsupported type,
  where 0.0.2 raised `AttributeError`.
- Messages: a row group that fails to decode names the first column that fails alone, as in "column 'b' cannot be
  decoded: IOError: Corrupt snappy compressed data."; `StandardScaler` says a column's squares "sum to the largest f64
  or more (scale the column down first)" where it said the column "sums past the range of f64"; and `MinMaxScaler`
  says when its two `feature_range` bounds round to one f64.
- `fast_feature_transform._core.list_kernels` is gone, and `_core.pyi` no longer declares it; the error for an unknown
  kernel still lists them. A transformer keeps its fit in `_binding` where 0.0.2 kept `_handle`, `_stage` and
  `_standalone`.

Faster:

- The record path takes 0.84 to 0.94 of the time on a record of 30 to 2,000 fields of floats, strs, numpy numbers or
  naive datetimes under `str` keys: 30 float fields through one `Select` take 513 ns per `transform_dict` call where
  they took 610. Records of 1 to 127 fields through one to twenty steps take 0.95 to 1.02 of the time.
- Fits over a million rows in 16 row groups: `CountEncoder` and `OrdinalEncoder` on a dictionary column of 32
  categories take 0.82 and 0.92 of the time, `SplineTransformer` with uniform knots on integers 0.80, `TargetEncoder`
  over a million distinct keys 0.81 (357 -> 288 ms), and `LeaveOneOutEncoder` and `CatBoostEncoder` over 1,000
  categories with an integer target 0.59 (12.3 -> 7.2 ms).
- `pickle.dumps` of a fitted `LabelEncoder`, or of a pipeline of one and a `StandardScaler`, takes 0.90 to 0.93 of the
  time.

Costs more:

- Strings inside a fixed-size list are read plain: passing 500,000 rows of four 64-byte strings from 100 distinct
  values, in 16 row groups, through takes 1.9 to 2.3 times as long without null rows (42 ms -> 79 to 95 ms, from one
  run to the next), and 0.57 to 0.63 of the time with 30% of the rows null.
- A top-level `string_view` column in delta pages is copied as it is read: passing 1,048,576 rows of 8-byte strings
  in 16 row groups through takes 1.25 times as long when 90% are null (24.5 -> 30.5 ms), and 1.02 times when none is.
- The wide output with long strings above takes 1.17 times as long (286 -> 333 ms). Distinct strings of 1,000
  characters read as a dictionary beside 1,000 f64 columns, 32,768 rows in one row group, are written in 37 row groups
  where 0.0.2 wrote 8, 2.0% larger, in 1.18 times the time and with a peak of 1,620 MiB where it was 1,601.
- A `TargetEncoder` fit over a million distinct keys peaks at 346 MiB where it peaked at 300, and `LeaveOneOutEncoder`
  and `JamesSteinEncoder` there at 346 MiB where they peaked at 337, in 1.08 times the time. `JamesSteinEncoder` on a
  million integer targets above 2^53 over 1,000 categories takes 1.77 times as long (16.6 -> 29.4 ms), and
  `TargetEncoder` and `LeaveOneOutEncoder` over four columns of 1,000 categories 1.05 and 1.02 times.
- A record that is checked for a repeated key takes longer: three keys of a `str` subclass with its own hash 1.20 times
  as long (171 -> 206 ns per `transform_dict` call), and 30 fields with one `numpy.datetime64` or one datetime in a
  `zoneinfo` time zone 1.07 and 1.09 times.
- Reading a cached fitted attribute takes about 11 ns more, and `transform_dict` of a standalone transformer 3 ns
  more. Constructing a `GroupAgg` of three aggregates takes 524 ns where it took 346, a `CountEncoder` given a share
  1,006 ns where it took 642, and a `Cast` over 20,000 names given as a `str` subclass 2.38 ms where it took 0.59.

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
