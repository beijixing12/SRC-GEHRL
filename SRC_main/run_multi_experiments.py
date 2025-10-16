from __future__ import annotations

import argparse
import math
import re
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Sequence


_FLOAT_PATTERN = r"([-+]?\d*\.\d+|[-+]?\d+)"


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a shell command multiple times and report the mean/variance "
            "of a metric extracted from its output."
        )
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of times to execute the target command (default: 5).",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default=_FLOAT_PATTERN,
        help=(
            "Regular expression used to locate the metric in the command's "
            "stdout. By default the last floating point number is used."
        ),
    )
    parser.add_argument(
        "--group",
        type=int,
        default=1,
        help="Which capturing group from --pattern contains the numeric value (default: 1).",
    )
    parser.add_argument(
        "--sample-variance",
        action="store_true",
        help="Report the unbiased sample variance instead of the population variance.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the command output; only log the extracted metric per run.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Abort immediately if any run exits with a non-zero status.",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        help="Optional path to append per-run metrics (CSV format).",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help=(
            "Command to execute. Separate this from the script options using '--'. "
            "Example: python Scripts/run_multi_experiments.py --runs 10 -- python trainSRC.py -d assist09"
        ),
    )
    args = parser.parse_args(argv)
    if not args.command:
        parser.error(
            "No command supplied. Use '--' to separate the runner options from the target command."
        )
    if args.runs <= 0:
        parser.error("--runs must be a positive integer")
    return args


def _extract_metric(pattern: re.Pattern[str], text: str, group: int) -> float:
    matches = list(pattern.finditer(text))
    if not matches:
        raise ValueError("Unable to locate metric in command output; adjust --pattern or --group.")
    try:
        value = matches[-1].group(group)
    except IndexError as exc:
        raise ValueError(
            "Requested capture group is not present in the regex match."
        ) from exc
    return float(value)


def _write_log(path: Path, metrics: Iterable[float]) -> None:
    header_needed = not path.exists()
    with path.open("a", encoding="utf-8") as handle:
        if header_needed:
            handle.write("run,metric\n")
        for index, metric in enumerate(metrics, start=1):
            handle.write(f"{index},{metric}\n")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    pattern = re.compile(args.pattern)
    metrics: List[float] = []
    logs: List[float] = []
    for run_idx in range(1, args.runs + 1):
        print(f"[run {run_idx}/{args.runs}] Executing: {' '.join(args.command)}")
        completed = subprocess.run(
            args.command,
            capture_output=args.quiet,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            message = (
                f"Command exited with status {completed.returncode} on run {run_idx}."
            )
            if args.fail_fast:
                raise subprocess.CalledProcessError(
                    completed.returncode,
                    args.command,
                    output=completed.stdout,
                    stderr=completed.stderr,
                )
            print(f"Warning: {message}")
        output_text = completed.stdout if completed.stdout is not None else ""
        if not args.quiet and completed.stdout:
            print(completed.stdout, end="")
        if completed.stderr:
            print(completed.stderr, file=sys.stderr, end="")
        try:
            metric = _extract_metric(pattern, output_text, args.group)
        except ValueError as err:
            print(f"Error parsing metric on run {run_idx}: {err}", file=sys.stderr)
            if args.fail_fast:
                return 2
            continue
        metrics.append(metric)
        logs.append(metric)
        print(f"[run {run_idx}] metric={metric}")

    if not metrics:
        print("No metrics were extracted; nothing to aggregate.", file=sys.stderr)
        return 1

    mean_value = statistics.fmean(metrics)
    if len(metrics) == 1:
        variance = 0.0
    else:
        if args.sample_variance and len(metrics) > 1:
            variance = statistics.variance(metrics)
        else:
            variance = statistics.pvariance(metrics)
    print("\nSummary statistics")
    print("===================")
    print(f"Runs: {len(metrics)}")
    print(f"Mean: {mean_value:.6f}")
    print(f"Variance: {variance:.6f}")
    std_dev = math.sqrt(variance)
    print(f"Std Dev: {std_dev:.6f}")

    if args.log is not None:
        _write_log(args.log, logs)
        print(f"Per-run metrics appended to {args.log}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())