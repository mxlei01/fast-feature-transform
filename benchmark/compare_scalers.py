"""Compare two Release builds of fft_scaler_benchmark on identical in-memory inputs.

Run with python benchmark/compare_scalers.py --baseline /path/to/old-binary
--optimized /path/to/new-binary --output results.json. In each round both
binaries get the same row count, iteration count and case-order seed, and run
in a shuffled order. The script reports the median and interquartile range
across rounds, keeps every observation in the output file, and raises
ValueError instead of reporting a speedup if any output hash differs.
"""

import argparse
import json
import platform
import random
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any


def summarise(values: list[float]) -> dict[str, float]:
    """Summarize per-transform milliseconds across independent rounds.

    Args:
        values: One timing per round.

    Returns:
        Median, quartiles, and interquartile range in milliseconds.

    """
    q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive")
    return {"median_ms": statistics.median(values), "q1_ms": q1, "q3_ms": q3, "iqr_ms": q3 - q1}


def main() -> None:
    """Run paired measurements, check that the outputs agree, and write the summary and raw results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--optimized", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=500_000)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=15)
    args = parser.parse_args()
    if args.rows < 2 or args.iterations < 1 or args.repeats < 2:
        parser.error("rows must be >= 2, iterations >= 1, and repeats >= 2")

    binaries = {"baseline": args.baseline.resolve(), "optimized": args.optimized.resolve()}
    rng = random.Random(1729)
    raw: list[dict[str, Any]] = []
    expected_cases: set[str] | None = None
    for round_number in range(args.repeats):
        order = list(binaries)
        rng.shuffle(order)
        for label in order:
            result = subprocess.run(
                [str(binaries[label]), str(args.rows), str(args.iterations), str(round_number + 73)],
                capture_output=True,
                text=True,
                check=True,
            )
            rows: list[dict[str, Any]] = json.loads(result.stdout)["results"]
            cases = {row["case"] for row in rows}
            if not rows or len(cases) != len(rows) or (expected_cases is not None and cases != expected_cases):
                raise ValueError(f"benchmark case mismatch: {label}, round {round_number + 1}")
            expected_cases = cases
            raw.extend({"round": round_number, "label": label, **row} for row in rows)
        print(f"round {round_number + 1}/{args.repeats}", file=sys.stderr, flush=True)

    summary: dict[str, Any] = {}
    for name in sorted(expected_cases or ()):
        observations = [row for row in raw if row["case"] == name]
        if len({row["output_hash"] for row in observations}) != 1:
            raise ValueError(f"output bits or nulls differ: {name}")
        summary[name] = {
            label: summarise([row["milliseconds"] for row in observations if row["label"] == label])
            for label in binaries
        }
        summary[name]["speedup"] = summary[name]["baseline"]["median_ms"] / summary[name]["optimized"]["median_ms"]

    args.output.write_text(
        json.dumps(
            {
                "platform": platform.platform(),
                "machine": platform.machine(),
                "binaries": {label: str(path) for label, path in binaries.items()},
                "rows": args.rows,
                "iterations": args.iterations,
                "repeats": args.repeats,
                "seed": 1729,
                "summary": summary,
                "raw": raw,
            },
            indent=2,
        )
        + "\n"
    )
    for name, row in summary.items():
        before, after = row["baseline"], row["optimized"]
        print(
            f"{name:45} {before['median_ms']:.3f} ({before['iqr_ms']:.3f}) -> "
            f"{after['median_ms']:.3f} ({after['iqr_ms']:.3f}) ms; {row['speedup']:.2f}x"
        )


if __name__ == "__main__":
    main()
