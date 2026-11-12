"""Out-of-core scale tests: many row groups across two files, high cardinality, memory that does not grow with the rows."""

import subprocess  # nosec B404
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fast_feature_transform import LabelEncoder, Pipeline, Select

ROWS_PER_FILE: int = 1_000_000
ROW_GROUP_SIZE: int = 10_000
FILES: int = 2
CATEGORIES: int = 50_000
SAMPLE: int = 2_000

# The memory check runs in a child process, whose peak memory no earlier test
# has raised. The child fits the two files once, then fits them REPEATS times
# over in one call, and prints how far the second fit raised the peak.
REPEATS: int = 8
# The second fit reads 16,000,000 rows. Held as 4-byte dictionary indices, the
# least a held row costs, they are 64 MB, 56 MB more than the first fit's
# rows. A fit that streams raises the peak by 3.0 MB (measured on macOS
# arm64), for the 14 more Parquet footers it keeps open. The bound sits
# between the two.
MAX_PEAK_GROWTH_BYTES: int = 16_000_000

CHILD: str = """
import resource
import sys

import pyarrow.parquet as pq

from fast_feature_transform import LabelEncoder, Pipeline, Select


def peak():
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return maxrss if sys.platform == "darwin" else maxrss * 1024  # macOS reports bytes, Linux kilobytes


def fit(files):
    Pipeline([("s", Select(columns=["user"])), ("e", LabelEncoder(min_count=2))]).fit(files, threads=1)


how, repeats, files = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
fit(files)
before = peak()
if how == "fit":
    fit(files * repeats)
else:
    held = [pq.read_table(path, columns=["user"], read_dictionary=["user"]) for path in files * repeats]
print(peak() - before)
"""


@pytest.fixture(scope="module")
def files(tmp_path_factory: pytest.TempPathFactory) -> list[str]:
    """Write two Parquet files of 100 row groups each, with a long-tailed string column.

    Args:
        tmp_path_factory: pytest-provided factory of temporary directories.

    Returns:
        Paths of the written Parquet files.

    """
    directory: Path = tmp_path_factory.mktemp("scale")
    rng = np.random.default_rng(11)
    categories: np.ndarray = np.array([f"user_{i:06d}" for i in range(CATEGORIES)])
    paths: list[str] = []
    for index in range(FILES):
        ids: np.ndarray = categories[rng.zipf(1.3, ROWS_PER_FILE) % CATEGORIES]
        table: pa.Table = pa.table({"user": ids, "price": rng.random(ROWS_PER_FILE)})
        path: Path = directory / f"part-{index}.parquet"
        pq.write_table(table, path, row_group_size=ROW_GROUP_SIZE)
        paths.append(str(path))
    return paths


def _peak_growth(how: str, files: list[str], directory: Path) -> int:
    """Run the child process and return how far its second step raised its peak memory.

    Args:
        how: "fit" to fit the files ``REPEATS`` times over, "hold" to read them that often and keep the rows.
        files: Paths of the input Parquet files.
        directory: The child's working directory, which holds no package to import by accident.

    Returns:
        The growth of the child's peak resident memory in bytes.

    """
    # A warning is an error in the child, as it is in this process.
    command: list[str] = [sys.executable, "-W", "error", "-c", CHILD, how, str(REPEATS), *files]
    child = subprocess.run(command, check=True, capture_output=True, text=True, cwd=directory)  # nosec B603
    return int(child.stdout)


def test_fit_streams_many_row_groups(files: list[str], tmp_path: Path) -> None:
    serial: Pipeline = Pipeline([("s", Select(columns=["user"])), ("e", LabelEncoder(min_count=2))]).fit(
        files, threads=1
    )
    parallel: Pipeline = Pipeline([("s", Select(columns=["user"])), ("e", LabelEncoder(min_count=2))]).fit(files)

    serial_encoder = serial["e"]
    parallel_encoder = parallel["e"]
    assert isinstance(serial_encoder, LabelEncoder)
    assert isinstance(parallel_encoder, LabelEncoder)
    assert serial_encoder.label_maps_ == parallel_encoder.label_maps_

    parts: list[Path] = parallel.transform_parquet(files, output_dir=tmp_path / "out")
    assert len(parts) == FILES * (ROWS_PER_FILE // ROW_GROUP_SIZE)
    output: pa.Table = pq.read_table(str(tmp_path / "out"))
    assert output.num_rows == FILES * ROWS_PER_FILE

    sample: pa.Table = pq.read_table(files[0]).slice(0, SAMPLE)
    encoded: list[int] = pq.read_table(parts[0]).column("user").to_pylist()[:SAMPLE]
    for value, expected in zip(sample.column("user").to_pylist(), encoded, strict=True):
        assert parallel.transform_dict({"user": value})["user"] == expected


def test_fit_memory_does_not_grow_with_the_rows(files: list[str], tmp_path: Path) -> None:
    streamed: int = _peak_growth("fit", files, tmp_path)
    assert streamed < MAX_PEAK_GROWTH_BYTES, f"the fit raised its peak memory by {streamed / 1e6:.1f} MB"
    # The same rows held in memory break the bound, so the check above can fail.
    held: int = _peak_growth("hold", files, tmp_path)
    assert held > MAX_PEAK_GROWTH_BYTES, f"holding the rows raised the peak memory by only {held / 1e6:.1f} MB"
