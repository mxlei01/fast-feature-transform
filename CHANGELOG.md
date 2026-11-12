# Changelog

## 0.0.2

Faster fits, faster and smaller Parquet output, and fixes. A pipeline saved by 0.0.1 loads and answers as before.
Timings are from an Apple M4 on one thread.

Faster:

- Fits that sum floats take a half to a sixth of the time: `StandardScaler` 42.3 -> 6.5 ms per million rows, and
  `TargetEncoder`, `JamesSteinEncoder`, `LeaveOneOutEncoder`, `CatBoostEncoder`, `GroupAgg`, `PowerTransformer` and
  k-means bins with it.
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

- A step given no columns takes the columns known when the fit starts. The columns an expanding encoder such as
  `OneHotEncoder` generates never join them, whichever pass the step fits in.
- A parameter name a kernel does not know is an error at fit and at load. 0.0.1 ignored it and used the default.
- Parquet output has another layout where it is very wide or holds distinct integers: fewer and larger row groups,
  and plain pages after 248 KiB of dictionary. The values are the same. An integer column with 32,000 to 55,000
  distinct values in a row group is written up to 13% larger.
- A string option given as an `Enum` with a `str` mixin works in every transformer.
- 0.0.1 cannot load a pipeline saved by this version whose `CountEncoder` folded a group.

Fixed:

- Files that store the same `list<string>` column in different encodings are read together.
- `StandardScaler` and `JamesSteinEncoder` fit values near 1e-155 alike for every thread count and row-group split.
- A `transform_parquet` or `parquet_to_memmap` that fails no longer leaves an empty temporary directory behind.

## 0.0.1

The first release.

- A `Pipeline` fits on Parquet files, saves as one JSON manifest, and transforms one record (`transform_dict`) or
  Parquet files, writing Parquet parts (`transform_parquet`) or a packed file for NumPy (`transform_memmap`).
- Wheels for CPython 3.14, standard and free-threaded, on Linux x86_64, Linux arm64, and macOS 26 or later on arm64.
