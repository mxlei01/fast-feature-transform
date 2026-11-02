"""Profile every case of fft_kernel_benchmark and print a Markdown table, one row per case.

Run with the benchmark binary, built with debug information (-g):
python benchmark/profile_kernels.py <binary> [--stage transform|fit] [--seconds S] [--case SUBSTRING]... [--top N]
                                             [--keep DIR]

Each case runs alone under the benchmark's sampler for at least the given number of seconds, and its
samples are symbolized with line_profile.py. A row gives the median time per call and the functions and
source lines that took the most samples. A function is the innermost frame's, so code inlined into a
kernel still shows under its own name. arm64 macOS only, as the benchmark is.
"""

import argparse
import collections
import json
import re
import subprocess
import tempfile
from pathlib import Path

from line_profile import function, location, read_profile, symbolize


def case_names(binary: Path, stage: str, wanted: list[str]) -> list[str]:
    """List the benchmark's cases for one stage.

    Args:
        binary: The benchmark executable.
        stage: The stage to keep, such as "transform".
        wanted: Substrings a case must contain one of; empty keeps every case.

    Returns:
        The full case names, stage included.

    """
    listed = subprocess.run([str(binary), "--list"], capture_output=True, text=True, check=True).stdout.split()
    names = [f"{name}/{stage}" for name in listed]
    return [name for name in names if not wanted or any(w in name for w in wanted)]


def run_case(binary: Path, case: str, seconds: float, samples: Path) -> dict | None:
    """Run one case under the sampler.

    Args:
        binary: The benchmark executable.
        case: The full case name.
        seconds: The least time to keep calling it for.
        samples: Where the sampler writes its counts.

    Returns:
        The benchmark's result for the case, or None when it has no such stage: a fit runs without nulls only.

    """
    command = [
        str(binary),
        "--case",
        case,
        "--iterations",
        "3",
        "--min-seconds",
        str(seconds),
        "--profile",
        str(samples),
    ]
    output = subprocess.run(command, capture_output=True, text=True, check=True).stdout
    # A case that fails is recorded under its name without the stage.
    results = [r for r in json.loads(output)["results"] if r["case"] in (case, case.rsplit("/", 1)[0])]
    return results[0] if results else None


def short_name(name: str) -> str:
    """Return a function name's last two parts, such as Class::method, without template arguments or parameters."""
    text = name.replace("(anonymous namespace)::", "")
    depth, kept = 0, []
    for char in text:
        if char == "<":
            depth += 1
        elif char == ">":
            depth -= 1
        elif depth == 0:
            kept.append(char)
    text = "".join(kept).split("(")[0]
    parts = [part for part in text.split("::") if part and part not in ("fastfeaturetransform", "std", "__1")]
    return "::".join(parts[-2:]) or name


def shares(binary: Path, samples: Path, top: int) -> tuple[int, list[tuple[str, float]], list[tuple[str, float]]]:
    """Find the functions and source lines that took the most samples.

    Args:
        binary: The benchmark executable.
        samples: The sampler's counts.
        top: How many of each to keep.

    Returns:
        The number of samples, then (name, share) pairs for functions and for source lines.

    """
    base, counts = read_profile(samples)
    frames = symbolize(binary, base, sorted(counts))
    by_function: collections.Counter[str] = collections.Counter()
    by_line: collections.Counter[str] = collections.Counter()
    for pc, n in counts.items():
        inner = frames[pc][0]
        by_function[short_name(function(inner))] += n
        by_line[re.sub(r"^.*/", "", location(inner))] += n
    total = sum(counts.values())
    return (
        total,
        [(name, n / total) for name, n in by_function.most_common(top)],
        [(name, n / total) for name, n in by_line.most_common(top)],
    )


def main() -> None:
    """Profile the selected cases and print one table row per case."""
    parser = argparse.ArgumentParser()
    parser.add_argument("binary", type=Path)
    parser.add_argument("--stage", default="transform")
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--top", type=int, default=3)
    parser.add_argument("--keep", type=Path, help="a directory to keep each case's samples in, for --annotate")
    args = parser.parse_args()
    if args.keep:
        args.keep.mkdir(parents=True, exist_ok=True)
    print("| case | ms per call | samples | functions | source lines |")
    print("| --- | ---: | ---: | --- | --- |")
    with tempfile.TemporaryDirectory() as directory:
        for case in case_names(args.binary, args.stage, args.case):
            samples = (args.keep or Path(directory)) / (case.replace("/", "_") + ".txt")
            result = run_case(args.binary, case, args.seconds, samples)
            if result is None:
                continue
            if "error" in result:
                print(f"| {case} | error: {result['error']} | | | |")
                continue
            total, functions, lines = shares(args.binary, samples, args.top)
            named = ", ".join(f"`{name}` {share:.0%}" for name, share in functions)
            located = ", ".join(f"`{line}` {share:.0%}" for line, share in lines)
            print(
                f"| `{case.removesuffix('/' + args.stage)}` | {result['median_ms']:.2f} | {total} | {named} | {located} |"
            )


if __name__ == "__main__":
    main()
