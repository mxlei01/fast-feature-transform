# Changelog

## 0.0.4

Pipelines and pickles saved by 0.0.1, 0.0.2 or 0.0.3 load in this version and answer as they do in 0.0.3, and versions
0.0.3 and 0.0.2 load pipelines and pickles saved by this version and answer as it does, except that the version that
runs a transform decides its answer in the cases below that changed, and 0.0.2 also in those of the 0.0.3 entry, such as
a `Normalizer` row that 0.0.2 refuses and this version normalizes. A saved `KBinsDiscretizer` keeps its edges; only a
new fit gets the k-means edges below. A `CountEncoder` saved by 0.0.1 or 0.0.2 that sets `min_group_name` or a
nondefault `combine_min_nan_groups` without `min_group_size` still cannot be fitted again, with a new message (see
Changed). Loading refuses some `KBinsDiscretizer` edge lists and `QuantileTransformer` grids that no fit writes (see
Changed). Measurements compare the published macOS wheels of 0.0.2 and 0.0.3 with the macOS wheel that the release
workflow builds from this version's source, for free-threaded CPython 3.14 on one thread of an Apple M4: CPU time, the
median of 11 pairs of fresh processes unless a line says otherwise. The Linux wheels were not measured.

Fixed:

- A string or binary column chunk that holds a dictionary page, pages of indices and then plain pages, with an index at
  or past the number of values its dictionary page states, raises an `EngineError` naming the column when a fit or a
  transform reads it, at the top level and in nested columns, such as "<file>: failed to read row group 0: column 'c'
  has a dictionary index outside its dictionary: Index 10 out of bounds". 0.0.3 raised this error only for an index that
  also lay past the values that the plain pages read by the end of its batch had added, naming the first index past
  them, and read the rows that hold any other such index as the values of later rows, without an error. A read takes a
  row group in batches of at most 65,536 rows, and fewer when its rows are wide, such as 41,120 for a transform that
  passes four string columns and an f64 column through. Such an index in a page of indices that follows a plain page and
  ends its indices before its count of values gets the same error when a batch ends inside that page no later than its
  indices end, and "cannot be decoded: IOError: Unexpected end of stream" otherwise, which 0.0.3 raised in both cases
  unless its check of the batch refused the index first. The pages of indices before the chunk's first plain page are
  checked by a second read, so a file written over in place between the two reads, damaged when first read and valid
  when read again, still passes such an index on without the error. A file cut short in place inside those pages between
  the two reads raises an `EngineError` naming the column, such as "has fewer pages of indices than when it was first
  read: the file changed while it was read", where 0.0.3, which read them once, read on to the cut and raised "IOError:
  Unexpected end of stream".
- A page whose header states more bytes than its column chunk has left is refused before those bytes are requested.
  0.0.3 requested the stated bytes and filled with zeros what the chunk did not hold before it refused the page: a
  transform of a 40 KB file whose dictionary page states 300,000,000 bytes peaked at 374 MB of resident memory, against
  74 MB for the undamaged file, where both peak at 74 to 75 MB here (three runs each).
- A record datetime whose `utcoffset()` moves it outside the range of int64 microseconds, such as a `datetime` subclass
  whose `utcoffset()` returns `timedelta(days=999_999_999)`, raises `EngineError` "record value for 't' is outside the
  range of int64 microseconds"; 0.0.3 returned a wrapped, wrong integer. Offsets that keep it inside the range answer as
  before.
- `output_dir` and the path of `save` and `load` given as a `str` subclass whose truth test or `in` operator hides an
  empty text or a NUL raise `ValueError` before anything is written, as a plain `str` does. 0.0.3 created the directory
  named by the text before the NUL, read from the path cut there in `load`, and raised `EngineError` for an empty text.
  A `temp_dir` given such a NUL raises "temp_dir must not hold a NUL character", where 0.0.3 said it "must be an
  existing directory".
- Assigning `Pipeline.steps` after a fit of that pipeline, or of a pipeline that holds it, has read that pipeline's
  steps, and before that fit returns or raises, raises `RuntimeError` "Pipeline is being fitted, so its steps cannot be
  reassigned; build a new Pipeline with the steps you want and fit that". 0.0.3 took the new steps and bound them to the
  plan compiled from the old ones, so the pipeline listed the new steps and ran the old ones. A fit reads the steps of
  the pipeline it fits as it starts, and those of a nested pipeline when it reaches that pipeline among the steps: an
  assignment that lands before that read is taken, and the fit runs the new steps, as in 0.0.3. After a fit that raised,
  an assignment is taken, as before.
- A batch call without `output_dir`, such as `transform_parquet`, `fit_transform` or `parquet_to_memmap`, whose removal
  of its temporary directory is hit by a `KeyboardInterrupt` starts the removal over, so the directory and its parts are
  gone; 0.0.3 stopped the removal and left them. When the call's own exception was a `KeyboardInterrupt`, the call then
  raises that one, with no other interrupt as its context, where 0.0.3 raised the second interrupt with the first as its
  context; after another exception, such as an `EngineError`, it raises the first interrupt that hit the removal with
  that exception as its context, as 0.0.3 did. An interrupt that lands between two attempts still stops the removal, as
  in 0.0.3, and is raised with the call's own exception as its context.
- `KBinsDiscretizer(strategy="kmeans")` no longer stops after its first center update on a column whose farthest value
  lies 2^512 (about 1.34e154) or more from its mean, while the column's squared deviations, scaled down by a power of
  two, sum within f64: it runs such a column to convergence, as 0.0.2 did. On the 140 columns of 10 to 100 values at one
  scale between 1e155 and 1e300 that the 0.0.3 entry names, at 2 to 8 bins, the edges equal 0.0.2's in all 980 fits, and
  the bins differ from 0.0.3's in 428, of which 420 equal scikit-learn 1.9.0's bins for the column scaled down by a
  power of two and none of 0.0.3's did. `[-a, -a, a, a]` with `a = 2^600` at 4 bins gets the edges `[-a, -a, 0, a]` and
  puts its values in bins 1 and 2, where 0.0.3 gave `[-a, 0, 1.5a]` and bins 0 and 1, and from
  `a = 0x1.5555555555555p+1022` (about 5.99e307) it still raises, as in 0.0.3. Columns closer than 2^512 to their mean
  keep 0.0.3's edges. Such fits take longer (see Costs more).

Changed:

- `Pipeline.load`, and the first use of an unpickled pipeline, raise `EngineError` for a `KBinsDiscretizer` whose edges
  decrease where no fit lets them, or whose `strategy="uniform"` table holds neither `n_bins + 1` edges nor the two
  equal edges of a constant column, naming the step and the column, such as "bin_edges['x'] must not decrease from its
  first edge to its second", and for a `QuantileTransformer` whose `references` hold more points than its
  `n_quantiles`, naming the step: "'references' holds 5 points, more than the 3 that 'n_quantiles' allows". 0.0.3
  loaded them and answered: a uniform table of `n_bins=5` with an edge deleted gave 4 one-hot columns instead of 5. A
  fit of `"kmeans"` can write its first edge above its second, and one of `"uniform"` its last edge below the one
  before, and those load; so does a `"quantile"` or `"kmeans"` table with an edge deleted.
- `KBinsDiscretizer(strategy="kmeans")` raises "fits a bin edge that is not finite; its values are too large" when the
  edge midway between two centers overflows, as on `[-1.7e308, -1.6e308]` followed by twenty values of `9e306` at 3, 4
  or 8 bins, which 0.0.3 binned by dropping that edge and scikit-learn 1.9.0 refuses. A `"quantile"` fit left with fewer
  than two edges because a computed edge overflowed raises the same error, as `[-1e308, -1e308, 1e308, 1e308]` at 3 or 8
  bins does, where 0.0.3 said the column "has no bin left" and to use `strategy='uniform'`; a column whose computed
  edges are all finite keeps 0.0.3's text.
- `LabelEncoder.vocab_size` of a column name that holds a lone surrogate, such as `"\ud800"`, raises `KeyError`, as for
  any name the encoder does not hold; 0.0.3 raised the binding's `TypeError` "incompatible function arguments".
- Messages: a page that states more bytes than its column chunk has left reads "column 'n' cannot be decoded: Invalid: a
  page states 1048575 bytes where its column chunk has 16007 left", where 0.0.3 gave Arrow's "IOError: Unexpected end of
  stream: Page was smaller (16007) than expected (1048575)"; a dictionary page that states more values than its bytes
  hold, "Invalid: a dictionary page states 60 values where its 60 bytes hold at most 15", where 0.0.3 said "IOError:
  Unexpected end of stream"; a row group whose pages hold fewer rows than its footer states, "<file>: failed to read row
  group 0: the columns read hold 300 rows where the footer states 400", where 0.0.3 said "input task row count changed
  while reading"; a `CountEncoder` fit without `min_group_size` names the option that needs it and the fix, as in
  "min_group_name needs min_group_size; set min_group_size, or leave min_group_name unset" and
  "combine_min_nan_groups=False needs min_group_size; set min_group_size, or leave combine_min_nan_groups at True",
  where 0.0.3 said "min_group_name and nondefault combine_min_nan_groups require min_group_size"; and the refusal of an
  extension column stored as a dictionary ends "; leave it out with a Select step".
- For those who build or release from a checkout: `make vcpkg` builds the C++ dependencies with the CMake 4.4.3 and
  ninja 1.13.2 that the pinned vcpkg checkout names, which vcpkg downloads, and not with a CMake or ninja found on the
  machine, which 0.0.3's `make vcpkg` let vcpkg use.

Faster:

- A string column whose dictionary page fills up, so that the rest of a row group is in plain pages, and whose page ends
  in a value over 256 bytes is read in about 0.0.2's time again: passing 12 row groups of 8,000 distinct random 300-byte
  strings through takes 29.1 -> 24.6 ms (0.84 of 0.0.3's time, 1.01 of 0.0.2's), 8 row groups of 6,000 distinct 384-byte
  strings 17.2 -> 15.1 ms (0.88 and 1.01), and 16 such row groups whose dictionary is in sorted order 0.95 and 1.00 of
  the time (21 pairs each).
- Passing a dictionary string column with nulls and many distinct values through takes 0.95 to 0.96 of 0.0.3's time and
  still 1.01 to 1.02 of 0.0.2's: 4,000,000 rows of 20,000 values, 90% of them null, in 16 row groups, take
  120.0 -> 115.4 ms, and 1,000,000 rows of 30,000 values with 30% null 76.4 -> 72.9 ms, where 0.0.2 takes 114.1 and
  71.6 ms (21 pairs each).
- A top-level `string_view` column in delta pages is converted on its buffers: passing 1,048,576 rows of 8-byte strings,
  90% of them null, in 16 row groups, through takes 31.4 -> 26.6 ms, 0.85 of 0.0.3's time and still 1.05 of 0.0.2's,
  which passed the column through without converting it (21 pairs).
- `Normalizer(norm="l2")` on rows of three or more values whose sum of squares lies above 2^1023 takes less time than in
  0.0.3, and on rows of three or four such values less than in 0.0.2: 250,000 rows of four f64 near 6.2e153 take
  23.8 -> 14.5 ms, 0.61 of 0.0.3's time and 0.86 to 0.87 of 0.0.2's (three sessions of 21 pairs). Rows of 8, 16 and 32
  such values take 0.62 to 0.82 of 0.0.3's time, and 0.94 to 0.96 of 0.0.2's when their squares sum near 1.5e308, as
  those four values do, but 1.02 when they sum near 1e308 (21 pairs each). A `StandardScaler` fit of 1,000,000 values
  near 1.2e151, whose squares sum past 2^1023 after 62% of the rows, takes 12.5 -> 9.5 ms (0.76); 0.0.2 refuses that
  column.

Costs more:

- `KBinsDiscretizer(strategy="kmeans")` fits take longer. A column 2^512 or more from its mean runs its updates to
  convergence again, in about 0.0.2's time: 1,000,000 values in 16 row groups at 8 bins take 40.1 -> 427 ms when they
  are lognormal times 1e200, and 40.0 -> 120.9 ms when they are normal times 1e200, where 0.0.2, with the same edges,
  takes 427 and 121.0 ms. The same lognormal values at their own scale, whose edges do not change, take 418 -> 426 ms,
  1.02 of 0.0.3's time and 1.00 of 0.0.2's (21 pairs each).
- Reading a string or binary column keeps a copy of the last value of each dictionary page, whatever its length, until
  the thread that read the page reads the column's next row group or the read ends; 0.0.3 kept it only up to 256 bytes.
  Passing through one row group of 10 strings whose dictionary page ends in a 64 MB value takes 26.9 -> 28.4 ms (21
  pairs) and peaks at 579 -> 643 MB of resident memory, and with an 8 MB value peaks at 115 -> 123 MB; a page that ends
  in a value of up to 256 bytes peaks as before. Each thread and each such column keeps its own copy: two such row
  groups read on two threads peak at 1,092 -> 1,220 MB, and one row group of two such columns at 771 -> 899 MB (three
  runs each).
- On a million rows in 16 row groups, fitting `JamesSteinEncoder` over 1,000 categories with an integer target from
  2^53 takes 1.02 times as long: 28.7 -> 29.1 ms (21 pairs), where 0.0.2 takes 16.2 ms.
- Assigning two steps to `Pipeline.steps` of an unfitted pipeline takes 299 -> 563 ns, and `copy.copy` of a fitted
  pipeline 252 -> 276 ns.

## 0.0.3

Pipelines and pickles saved by 0.0.1 or 0.0.2 load in this version and answer as before; a `CountEncoder` among them
that sets `min_group_name` or a nondefault `combine_min_nan_groups` without `min_group_size` cannot be fitted again
(see Changed). Version 0.0.2 loads pipelines and pickles saved by 0.0.3. The version that runs a transform decides its
answer in the cases below that changed: a `Normalizer` saved by 0.0.2 and loaded here normalizes a row that 0.0.2
refuses, and one saved here and loaded by 0.0.2 refuses it. Measurements compare the published 0.0.2 and 0.0.3 wheels
for free-threaded CPython 3.14 on one thread of an Apple M4: CPU time, the median of 11 pairs of fresh processes
unless a line says otherwise.

Fixed:

- Reading a nested string or binary column, such as a list, a struct or a fixed-size list of strings, whose
  dictionary index lies outside its dictionary raises an `EngineError` naming the column; 0.0.2 ended the process
  with a bus error. A column chunk whose dictionary is followed by plain pages is still read without the error, and
  the rows with an out-of-range index get the values of later rows.
- A string or binary column whose dictionary page holds one value twice raises an `EngineError` naming the column,
  also when no row uses the repeated entry. Arrow drops the repeat when it decodes the page, which shifts the values
  after it: 0.0.2 wrote the shifted values, refused the file with "has a dictionary index outside its dictionary", or
  ended the process with a bus error on a list of such strings. When no row uses the repeat or an entry after it,
  0.0.2 read the right values, and this version refuses the file too. Strings in a fixed-size list are read without
  the dictionary and get the values the page holds, where 0.0.2 ended the process with a bus error.
- Reading a column of Arrow's `arrow.opaque` extension type stored as a dictionary, at the top level or in a list,
  raises an `EngineError` naming the column; 0.0.2 ended the process with an abort. Selecting only other columns
  works. An extension type that this package's Arrow does not define, such as one defined in pyarrow, is read as its
  storage, as before.
- A record value of a class named `datetime64` that has `astype` but no `dtype`, or whose `dtype` raises, and a
  datetime whose `asm8` is such a value, raise an `EngineError`; 0.0.2 ended the process with an abort. A datetime
  whose `asm8` is not a numpy datetime64 raises an `EngineError` that says so; 0.0.2 raised `AttributeError` for an
  `asm8` of `None`.
- A record value of a class named like a numpy scalar that is not one, such as a class named `float64` with
  `__float__` or one named `bool_`, gets the unsupported-type `EngineError`. 0.0.2 converted such a value, raised
  `AttributeError` for a class named `datetime64` without `astype`, and `SystemError` for a `bool_` whose `__bool__`
  raises.
- `transform_dict` reads a record as it was when the call began: a value whose conversion code changes the record no
  longer makes it skip a field, which 0.0.2 then reported as missing ("select key 'b' missing from record").
- A record that holds one key twice, such as a `str` subclass with its own hash beside a `str` of the same text,
  raises `EngineError` "record holds the key 'x' twice"; 0.0.2 answered with the later field. A key whose class's
  `__hash__` changes while the record holds it is not checked.
- Fitted-attribute reads, `transform_dict`, copies and pickles of a transformer or pipeline that another thread fits
  again each answer for one fit, the old or the new. In 0.0.2 such calls sometimes mixed two fits:
  `MinMaxScaler.data_range_` took its endpoints from both or raised `TypeError`, a standalone transformer's
  `transform_dict` raised "was fitted as part of a Pipeline", and a copy, deep copy, pickle or save of a pipeline that
  another thread fitted as a step of a larger one answered for the larger one. A copy, deep copy or pickle of an
  unpickled pipeline that other threads use for the first time no longer raises
  `RuntimeError: dictionary changed size during iteration`.
- `StandardScaler` fits a column whose squares sum within f64, such as `[9e153, 8e153]`, in every row order,
  row-group split and thread count, and refuses one whose squares sum to infinity in all of them. Of 118 columns of
  2 to 4 values whose squares sum to within a step of the largest f64 and round to a finite value, 0.0.2 refused 31
  in every order, split and thread count and 11 in some of them, and it refused `[9e153, 8e153]` with "the variance
  leaves the range of f64". `Normalizer` decides from the exact l1 or l2 sum of a row, so a row gets one answer in
  every column order: 0.0.2 refused the l1 row `[0x1.8p969, 0x1.fffffffffffffp1022, 0x1.fffffffffffffp1022]` in that
  order and normalized it in another.
- `KBinsDiscretizer(strategy="kmeans")` stops after its first center update when the squared deviations of a column,
  at the column's own scale, sum past the range of f64. On 30 columns of 50 to 200 values whose squared deviations sum
  past f64 while no value lies 2^512 or more from the mean, at 2 to 8 bins, its bins equal scikit-learn 1.9.0's in
  209 of 210 fits, where 0.0.2's equal them in 44. Further than 2^512 (about 1.34e154) from the mean, scikit-learn's
  own squares overflow, and this version also stops after one update: on 140 columns of 10 to 100 values at one
  scale between 1e155 and 1e300, its bins differ from 0.0.2's in 428 of 980 fits, and none of those 428 equal
  scikit-learn's bins for the column scaled down by a power of two, which 0.0.2's equal in 420. The column
  `[-6e307, -6e307, 6e307, 6e307]` at 4 bins raises "fits a bin edge that is not finite", which 0.0.2 binned.
- `KBinsDiscretizer(strategy="kmeans")` fits a column whose sum lies within f64 though a running sum passes it, such
  as three values of -8e307 and three of 8e307, and one whose sum rounds to the largest f64; 0.0.2 refused both with
  "sums past the range of f64".
- `WOEEncoder` counts the distinct values of an integer target exactly above 2^53: a target of 0, 2^62 and 2^62 + 1
  is refused for holding more than two values, which 0.0.2 fitted, and one of 2^62 and 2^62 + 1 for holding no 0,
  where 0.0.2 said it holds one value. `JamesSteinEncoder`, `LeaveOneOutEncoder` and `CatBoostEncoder` sum an integer
  target exactly: `LeaveOneOutEncoder` encodes a category of the targets 2^53 + 1 and 1 to 4503599627370497.0, where
  0.0.2 gave 4503599627370496.0. `LeaveOneOutEncoder` and `CatBoostEncoder` take a float target below 2^900 in
  magnitude, which 0.0.2 refused from 2^450.
- Files are read together when one holds a string column in `string_view` delta pages, a binary column in
  `large_binary` delta pages, or a list or struct of `large_string` or `string_view` in delta pages, and another holds
  the same column in dictionary pages; 0.0.2 refused them as different types. A top-level `string_view` column in
  delta pages is read as `utf8`, so `Cast` and the string transformers take it, where 0.0.2 refused it.
- For a string or binary dictionary column, an output row group that holds a null keeps in its dictionary only the
  values its rows hold, and its minimum and maximum statistics come from those values. 0.0.2 wrote the whole
  dictionary into every row group, and gave a row group that held a null and all the dictionary's entries but one the
  minimum and maximum of the whole dictionary. Above 100,000 parts, every part name has one more digit, the first
  included, so the names sort in row order; 0.0.2 named the 100,001st part `part-100000.parquet`, which sorts before
  `part-10001.parquet`.
- `output_dir`, `temp_dir` and the path of `save` and `load` take a `str` or an `os.PathLike` that gives one. Bytes,
  `None` where a path is needed, and other types raise `TypeError`, and an empty path or one that holds a NUL raises
  `ValueError`, before anything is written. 0.0.2 wrote into or loaded from `./b'out'`, `./None` or `./5`, used the
  `str()` of an `os.PathLike` as its path, created the directory named by the text before a NUL, and read
  `/manifest.json` for `load("")`. An `output_dir` given as a `str` subclass whose `in` operator hides the NUL still
  creates that directory before the error. An empty `temp_dir` still selects the system temporary directory.
- A batch call without `output_dir` that a `KeyboardInterrupt` stops removes its temporary directory and the parts in
  it; 0.0.2 left them.

Changed:

- Loading a manifest that lacks required fitted state, such as an `OrdinalEncoder`'s `has_null`, or that holds state
  for a `Select`, raises an `EngineError`; 0.0.2 loaded such manifests, and some answered differently, such as -2 for
  a null `OrdinalEncoder` value. `Pipeline.load` raises `EngineError` instead of `ValueError` for stage names that
  make one name both a step and a nested pipeline or that split a nested pipeline, and an unpickled pipeline with such
  names raises it at first use, where 0.0.2 answered. A handling option that is not a string is reported by name
  instead of with the JSON library's text.
- A `CountEncoder` fit requires `min_group_size` when `min_group_name` or a nondefault `combine_min_nan_groups` is
  set, also a new fit of one loaded or unpickled from 0.0.1 or 0.0.2, which still loads and answers as before. With
  `handle_missing="error"` this check comes before the check for nulls.
- String subclasses used as names or string options are stored as plain strings, also the type tags of `Cast`, the
  aggregates of `GroupAgg`, `hash_method` of `HashingEncoder` and `combine_min_nan_groups` of `CountEncoder`, which
  0.0.2 kept as given, so a pickle no longer refers to their class. Numeric options take compatible numpy scalars, a
  `Fraction` and an integer array of no dimensions, and store each as a plain `int` or `float`. A number whose nearest
  f64 is infinite or 0 though it is neither, such as `10**400`, raises `ValueError` naming the option; 0.0.2 raised
  `OverflowError`, or `TypeError` for a `Fraction`. A string option given a list, and `hash_method` given an array,
  raise `ValueError` where 0.0.2 raised `TypeError`. `pipeline[np.int64(0)]` returns the step, and a key that is
  neither a `str` nor integer-like raises `KeyError` even when it compares equal to a step name, which 0.0.2 returned.
- `transform_dict` raises the error of a record value whose `__index__` raises something other than `TypeError`, or
  whose lookup of `__index__` or of a datetime's `asm8` raises something other than `AttributeError`, such as
  `KeyError` or `KeyboardInterrupt`; 0.0.2 raised the unsupported-type `EngineError` instead, and read such a
  datetime from its fields. A subclass of a numpy floating type gets the unsupported-type error whatever its class is
  named; 0.0.2 converted one named like the numpy type. When a record holds more than one refused value, a value that
  needs no Python code to read is refused first, and the conversion code of the others does not run.
- A path that cannot be encoded as UTF-8, such as one with a lone surrogate from `os.fsdecode`, raises `ValueError`
  naming the argument in `X`, `output_dir`, `temp_dir` and the path of `save` and `load`; 0.0.2 raised the binding's
  `TypeError` "incompatible function arguments", and for `temp_dir` a `ValueError` that it must be an existing
  directory. `fit_transform` checks `output_dir` before it fits, so a refused `output_dir` leaves the pipeline as it
  was; 0.0.2 fitted first.
- Messages: a row group that fails to decode names the first column that fails, as in "failed to read row group 0:
  column 'n' cannot be decoded: IOError: Corrupt snappy compressed data."; `StandardScaler` says a column's "sum of
  its squares rounds to infinity (scale the column down first)" where it said the column "sums past the range of
  f64"; `LeaveOneOutEncoder` and `CatBoostEncoder` say a target value of 2^900 or more is "more than a sum can hold"
  where they said a value of 2^450 or more is "more than the squared sums can hold"; `MinMaxScaler` says when its two
  `feature_range` bounds round to one f64; a `large_binary` or `binary_view` column in delta pages is named `binary` in
  type errors, as it is in other pages; `transform_dict` given something other than a dict, a `temp_dir` of another
  type than `str` or `os.PathLike`, and an `os.PathLike` in `X` that gives bytes raise a `TypeError` that names the
  argument and the type.
- `fast_feature_transform._core.list_kernels` is gone, and `_core.pyi` no longer declares it. A transformer keeps its
  fit in one private `_binding` where 0.0.2 kept `_handle`, `_stage` and `_standalone`, and a pipeline where 0.0.2
  kept `_handle`, `_prefix` and `_manifest`, so code that called `pipeline._handle.transform_row` raises
  `AttributeError`.
- `JamesSteinEncoder`, `LeaveOneOutEncoder` and `CatBoostEncoder` expose the fitted `null_values_`. Annotations take a
  tuple of column names and type `X` as a list or tuple of paths, which a single `str` does not match, where 0.0.2's
  sequence of paths matched one; a path is typed `str` or `os.PathLike[str]`, where 0.0.2 named `pathlib.Path`.
- The writer also closes a row group when one more batch as large as its last would take the waiting batches past
  36 MiB, so a wide output with long values holds less memory and is written in more row groups. Passing 4,176 rows of
  1,000 f64 columns and a distinct 20,000-character string, from one row group, through peaks at 1,253 -> 882 MiB of
  resident memory (median of 11 runs), in 6 row groups where 0.0.2 wrote 1 and 2.6% more bytes, and takes
  134 -> 142 ms (21 pairs).
- A build from a source checkout uses CMake 4.4.2 and ninja 1.13.0, where 0.0.2 took CMake 4.1.2 or later and ninja
  1.11 or later. The Linux extensions export only the Python module initializer, as the macOS ones did; 0.0.2's
  Linux extensions also exported 46 weak symbols of C++ templates from Arrow and the standard library.

Faster:

- `transform_dict` of a record of 30 fields whose 23-byte names are in reverse or shuffled order, through one
  `StandardScaler`, takes 0.75 of the time: 3.13 -> 2.34 µs and 3.11 -> 2.36 µs. `Pipeline.load` of a `Cast`
  over 20,000 columns, saved by 0.0.2, takes 15.9 -> 12.0 ms.
- On a million rows in 16 row groups, with a target of 0 or 1: fitting a `TargetEncoder` over a million distinct keys
  takes 363 -> 288 ms, and fitting `LeaveOneOutEncoder` and `CatBoostEncoder` over 1,000 categories takes
  12.8 -> 7.6 ms each.

Costs more:

- Strings inside fixed-size lists are read without a dictionary. Passing 500,000 rows of four 64-byte strings from
  100 values, in 16 row groups, through takes 45 -> 85 ms and peaks at 73 -> 98 MiB of resident memory with no null
  rows, and takes 247 -> 141 ms and peaks at 92 -> 141 MiB with 30% null rows (peaks: median of 11 runs).
- A top-level `string_view` column in delta pages is copied as it is read: passing 1,048,576 rows of 8-byte strings,
  90% of them null, in 16 row groups, through takes 26.5 -> 33.0 ms.
- A string column whose dictionary page fills up, so that the rest of a row group is in plain pages, takes longer to
  read when the page ends in a value over 256 bytes: passing 12 row groups of 8,000 distinct random 300-byte strings
  through takes 27.5 -> 31.7 ms, and 8 row groups of 6,000 distinct 384-byte strings 16.9 -> 19.9 ms; 16 such row
  groups whose dictionary is in sorted order take 1.08 times as long. Values of 200 bytes take the same time.
- Passing a dictionary string column with nulls and many distinct values through takes 1.05 to 1.07 times as long:
  4,000,000 rows of 20,000 values, 90% of them null, in 16 row groups, take 115 -> 120 ms (21 pairs), and 1,000,000
  rows of 30,000 values with 30% null 74 -> 79 ms (21 pairs). Without nulls, or with 100 values, it takes the same
  time.
- `Normalizer` takes 1.42 times as long on rows whose sum of squares lies above 2^1023: 250,000 rows of four f64 near
  6.2e153 through `Normalizer(norm="l2")` take 17.7 -> 24.9 ms. Rows of ordinary values take the same time.
- With 16,384 rows of distinct 1,000-character strings, which the input holds in dictionary pages and then plain
  pages, beside 1,000 f64 columns in one row group, passing them through takes 281 -> 321 ms, in 13 row groups where
  0.0.2 wrote 4, and 1.4% more bytes.
- On a million rows in 16 row groups, fitting `LeaveOneOutEncoder` and `JamesSteinEncoder` over a million distinct
  keys takes 1.07 and 1.09 times as long with a target of 0 or 1 (270 -> 288 ms and 271 -> 296 ms), and 1.12 and 1.10
  times with a float target. With a target of 0 or 1, one fit of either peaks at 333 -> 345 MiB of resident memory, and
  one `TargetEncoder` fit above at 297 -> 346 MiB (median of 11 runs). Fitting `JamesSteinEncoder` over 1,000
  categories with an integer target from 2^53 takes 17.0 -> 30.0 ms.
- Reading a cached fitted attribute such as `StandardScaler.mean_` takes 48.6 -> 61.7 ns. Constructing a `GroupAgg`
  of three aggregates takes 333 -> 507 ns, and a `CountEncoder` with `min_group_size=0.05` 599 -> 958 ns. A `Cast`
  over 20,000 names given as a `str` subclass or as `numpy.str_` takes 0.59 -> 2.44 ms and 0.60 -> 2.42 ms to
  construct, where plain `str` names take 0.60 -> 0.56 ms.
- `transform_dict` of a record whose keys are a `str` subclass with its own `__hash__` takes 1.15 times as long with
  3 fields (212 -> 243 ns) and 1.07 times with 30 fields (2.07 -> 2.24 µs).

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
