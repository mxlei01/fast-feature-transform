"""Turn fft_kernel_benchmark --profile samples into per-line and per-instruction counts.

Run with the benchmark binary and its profile:
python benchmark/line_profile.py <binary> <samples.txt> [--top N] [--annotate FUNCTION]

Every sampled program counter is symbolized with atos, inlined frames included. The script prints the
share of samples per innermost source line, with the function that line ran in, and the share per
outermost function, which is what a call-graph profiler would show. With --annotate, it also prints
the disassembly from the lowest to the highest sampled address in the functions whose names contain
FUNCTION, each instruction with its sample count. macOS only: atos and llvm-objdump come with the
Xcode command line tools, and the binary needs debug information (-g) with its object files still
in place.
"""

import argparse
import collections
import re
import subprocess
import sys
from pathlib import Path

TEXT_VMADDR = 0x100000000  # where ld64 places an executable's __TEXT segment
LOCATION = re.compile(r"\(([^()]+:\d+)\)\s*$")
FUNCTION = re.compile(r"^(.*?) \(in [^)]+\)")
INSTRUCTION = re.compile(r"\s*([0-9a-f]+):\s+(.*)")


def read_profile(path: Path) -> tuple[int, dict[int, int]]:
    """Read the load address and the sample count of every program counter.

    Args:
        path: The file the benchmark wrote with --profile.

    Returns:
        The executable's load address and a count per program counter.

    """
    lines = path.read_text().splitlines()
    base = int(lines[0].split()[1], 16)
    counts = {int(pc, 16): int(n) for pc, n in (line.split() for line in lines[1:] if line.strip())}
    return base, counts


def atos(binary: Path, base: int, pcs: list[int]) -> str:
    """Run atos on program counters, printing every inlined frame.

    Args:
        binary: The benchmark executable.
        base: Its load address in the profiled process.
        pcs: Program counters from that process.

    Returns:
        atos's output: one line per frame, innermost first, and a blank line between addresses.

    """
    command = ["atos", "-o", str(binary), "-arch", "arm64", "-l", hex(base), "-i", *map(hex, pcs)]
    return subprocess.run(command, capture_output=True, text=True, check=True).stdout


def symbolize(binary: Path, base: int, pcs: list[int]) -> dict[int, list[str]]:
    """Map each program counter to its frames, innermost first.

    Args:
        binary: The benchmark executable.
        base: Its load address in the profiled process.
        pcs: Program counters from that process.

    Returns:
        The atos frames of every program counter.

    """
    frames: dict[int, list[str]] = {}
    for start in range(0, len(pcs), 400):
        chunk = pcs[start : start + 400]
        groups = [group.strip().splitlines() for group in atos(binary, base, chunk).strip().split("\n\n")]
        if len(groups) != len(chunk):  # ambiguous grouping: ask about one address at a time
            groups = [atos(binary, base, [pc]).strip().splitlines() for pc in chunk]
        for pc, group in zip(chunk, groups, strict=True):
            frames[pc] = [frame for frame in group if frame] or ["?"]
    return frames


def location(frame: str) -> str:
    """Return a frame's file:line, or '?' when atos found none."""
    match = LOCATION.search(frame)
    return match.group(1) if match else "?"


def function(frame: str) -> str:
    """Return a frame's function name, shortened for printing."""
    match = FUNCTION.match(frame)
    return re.sub(r"\s+", " ", match.group(1) if match else frame)[:140]


def annotate(binary: Path, base: int, counts: dict[int, int], pcs: list[int], name: str) -> None:
    """Print the disassembly between the lowest and highest sampled address, with sample counts.

    Args:
        binary: The benchmark executable.
        base: Its load address in the profiled process.
        counts: The sample count per program counter.
        pcs: The sampled program counters inside the functions to annotate.
        name: What the functions were selected by, for the heading.

    """
    low, high = min(pcs) - base + TEXT_VMADDR, max(pcs) - base + TEXT_VMADDR + 4
    command = ["xcrun", "llvm-objdump", "-d", "--no-show-raw-insn", "-l"]
    command += [f"--start-address={low:#x}", f"--stop-address={high:#x}", str(binary)]
    dump = subprocess.run(command, capture_output=True, text=True, check=True).stdout
    print(f"\n{name}: {sum(counts[pc] for pc in pcs)} samples; count, address, instruction")
    for line in dump.splitlines():
        match = INSTRUCTION.match(line)
        if match:
            address = int(match.group(1), 16) - TEXT_VMADDR + base
            print(f"{counts.get(address, 0):7d}  {match.group(1)}  {match.group(2)}")
        elif line.startswith(";"):
            print(f"         {line}")


def main() -> None:
    """Print the profile's per-line and per-function shares, and optionally an annotated function."""
    parser = argparse.ArgumentParser()
    parser.add_argument("binary", type=Path)
    parser.add_argument("profile", type=Path)
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--annotate", default="")
    args = parser.parse_args()
    base, counts = read_profile(args.profile)
    total = sum(counts.values())
    frames = symbolize(args.binary, base, sorted(counts))
    by_line: collections.Counter[tuple[str, str]] = collections.Counter()
    by_function: collections.Counter[str] = collections.Counter()
    for pc, n in counts.items():
        by_line[(location(frames[pc][0]), function(frames[pc][-1]))] += n
        by_function[function(frames[pc][-1])] += n
    print(f"{total} samples\n\nper innermost source line (function it ran in):")
    for (where, name), n in by_line.most_common(args.top):
        print(f"{100 * n / total:6.2f}%  {n:7d}  {where:45s} {name}")
    print("\nper outermost function:")
    for name, n in by_function.most_common(15):
        print(f"{100 * n / total:6.2f}%  {n:7d}  {name}")
    if args.annotate:
        pcs = [pc for pc in counts if args.annotate in function(frames[pc][-1])]
        if not pcs:
            sys.exit(f"no samples in a function matching {args.annotate!r}")
        annotate(args.binary, base, counts, pcs, args.annotate)


if __name__ == "__main__":
    main()
