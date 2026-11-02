"""Out-of-core scale test: many row groups across two files, high cardinality, bounded memory (``--run-scale``)."""

import resource
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
MAX_RSS_GROWTH_BYTES: int = 1_500_000_000


def _peak_rss_bytes() -> int:
    """Return the process's peak resident set size in bytes.

    Returns:
        Peak RSS, normalized from the platform's ``ru_maxrss`` unit.

    """
    maxrss: int = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return maxrss if sys.platform == "darwin" else maxrss * 1024  # macOS reports bytes, Linux kilobytes


@pytest.mark.scale
def test_fit_streams_many_row_groups_with_bounded_memory(tmp_path: Path) -> None:
    rng = np.random.default_rng(11)
    categories: np.ndarray = np.array([f"user_{i:06d}" for i in range(CATEGORIES)])
    files: list[str] = []
    for index in range(FILES):
        ids: np.ndarray = categories[rng.zipf(1.3, ROWS_PER_FILE) % CATEGORIES]
        table: pa.Table = pa.table({"user": ids, "price": rng.random(ROWS_PER_FILE)})
        path: Path = tmp_path / f"part-{index}.parquet"
        pq.write_table(table, path, row_group_size=ROW_GROUP_SIZE)
        files.append(str(path))

    before: int = _peak_rss_bytes()
    serial: Pipeline = Pipeline([("s", Select(columns=["user"])), ("e", LabelEncoder(min_count=2))]).fit(
        files, threads=1
    )
    parallel: Pipeline = Pipeline([("s", Select(columns=["user"])), ("e", LabelEncoder(min_count=2))]).fit(files)
    growth: int = _peak_rss_bytes() - before
    assert growth < MAX_RSS_GROWTH_BYTES, f"peak RSS grew by {growth / 1e6:.0f} MB"

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
