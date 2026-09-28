#!/usr/bin/env python3
"""Summarize sharded benchmark summaries after a launcher run."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


AVERAGE_FIELDS = (
    "average_update_length",
    "average_original_length",
    "average_optimized_length",
    "average_fitten_length",
)


def _as_int(value: Any) -> int:
    if value is None:
        return 0
    return int(value)


def _as_float(value: Any) -> float:
    if value is None:
        return 0.0
    return float(value)


def _pct(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value * 100:.2f}%"


def _delta_pp(original: float | None, optimized: float | None) -> str:
    if original is None or optimized is None:
        return "NA"
    return f"{(optimized - original) * 100:+.2f}pp"


def _shard_sort_key(row: dict[str, Any]) -> tuple[int, str]:
    match = re.match(r"g(\d+)$", row["shard"])
    if match:
        return int(match.group(1)), row["shard"]
    return 10**9, row["shard"]


def _load_rows(base_output_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(base_output_dir.glob("g*/**/summary.json")):
        data = json.loads(summary_path.read_text())
        shard = summary_path.relative_to(base_output_dir).parts[0]
        processed = _as_int(data.get("processed_total"))
        evaluated = _as_int(data.get("evaluated_total"))
        original_correct = _as_int(data.get("original_correct"))
        optimized_correct = _as_int(data.get("optimized_correct"))
        original_accuracy = (
            original_correct / evaluated if evaluated else data.get("original_accuracy")
        )
        optimized_accuracy = (
            optimized_correct / evaluated if evaluated else data.get("optimized_accuracy")
        )
        rows.append(
            {
                "shard": shard,
                "summary_path": str(summary_path),
                "processed_total": processed,
                "evaluated_total": evaluated,
                "original_correct": original_correct,
                "optimized_correct": optimized_correct,
                "original_accuracy": original_accuracy,
                "optimized_accuracy": optimized_accuracy,
                **{
                    field: _as_float(data.get(field))
                    for field in AVERAGE_FIELDS
                },
            }
        )
    rows.sort(key=_shard_sort_key)
    return rows


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    processed_total = sum(row["processed_total"] for row in rows)
    evaluated_total = sum(row["evaluated_total"] for row in rows)
    original_correct = sum(row["original_correct"] for row in rows)
    optimized_correct = sum(row["optimized_correct"] for row in rows)
    weighted = {}
    for field in AVERAGE_FIELDS:
        weighted[field] = (
            sum(row[field] * row["processed_total"] for row in rows) / processed_total
            if processed_total
            else 0.0
        )
    original_accuracy = original_correct / evaluated_total if evaluated_total else None
    optimized_accuracy = optimized_correct / evaluated_total if evaluated_total else None
    return {
        "processed_total": processed_total,
        "evaluated_total": evaluated_total,
        "original_correct": original_correct,
        "optimized_correct": optimized_correct,
        "original_accuracy": original_accuracy,
        "optimized_accuracy": optimized_accuracy,
        "delta_correct": optimized_correct - original_correct,
        "delta_pp": (
            (optimized_accuracy - original_accuracy) * 100
            if original_accuracy is not None and optimized_accuracy is not None
            else None
        ),
        **weighted,
    }


def _print_summary(
    *,
    benchmark: str,
    method_tag: str,
    base_output_dir: Path,
    rows: list[dict[str, Any]],
    total: dict[str, Any],
    expected_shards: int | None,
    expected_total: int | None,
) -> None:
    print("")
    print("[sharded-summary]")
    print(f"Benchmark: {benchmark}")
    print(f"Method: {method_tag}")
    print(f"Output: {base_output_dir}")
    print(f"Shards: {len(rows)}" + (f" / {expected_shards}" if expected_shards else ""))
    if expected_total is not None and total["processed_total"] != expected_total:
        print(
            f"WARN processed_total={total['processed_total']} "
            f"does not match expected_total={expected_total}"
        )
    print(
        "Total: "
        f"processed={total['processed_total']} "
        f"evaluated={total['evaluated_total']} "
        f"original={total['original_correct']}/{total['evaluated_total']} "
        f"({_pct(total['original_accuracy'])}) "
        f"optimized={total['optimized_correct']}/{total['evaluated_total']} "
        f"({_pct(total['optimized_accuracy'])}) "
        f"delta={total['delta_correct']:+d} "
        f"({_delta_pp(total['original_accuracy'], total['optimized_accuracy'])})"
    )
    print(
        "Lengths: "
        f"avg_update={total['average_update_length']:.4f} "
        f"avg_original={total['average_original_length']:.4f} "
        f"avg_optimized={total['average_optimized_length']:.4f} "
        f"avg_fitten={total['average_fitten_length']:.4f}"
    )
    print("Per-shard:")
    print("  shard processed evaluated original optimized delta")
    for row in rows:
        print(
            f"  {row['shard']:<5} "
            f"{row['processed_total']:>9} "
            f"{row['evaluated_total']:>9} "
            f"{_pct(row['original_accuracy']):>8} "
            f"{_pct(row['optimized_accuracy']):>9} "
            f"{_delta_pp(row['original_accuracy'], row['optimized_accuracy']):>8}"
        )
    print("")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-output-dir", required=True, type=Path)
    parser.add_argument("--benchmark", default="")
    parser.add_argument("--method-tag", default="")
    parser.add_argument("--expected-shards", type=int, default=None)
    parser.add_argument("--expected-total", type=int, default=None)
    parser.add_argument(
        "--write-json",
        type=Path,
        default=None,
        help="Optional path for the aggregate summary JSON.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = _load_rows(args.base_output_dir)
    if not rows:
        print(f"[sharded-summary] No summary.json files under {args.base_output_dir}")
        return 1

    total = _aggregate(rows)
    output = {
        "benchmark": args.benchmark,
        "method_tag": args.method_tag,
        "base_output_dir": str(args.base_output_dir),
        "num_shards": len(rows),
        "expected_shards": args.expected_shards,
        "expected_total": args.expected_total,
        "total": total,
        "shards": rows,
    }

    write_json = args.write_json or args.base_output_dir / "aggregate_summary.json"
    write_json.parent.mkdir(parents=True, exist_ok=True)
    write_json.write_text(json.dumps(output, indent=2) + "\n")

    _print_summary(
        benchmark=args.benchmark,
        method_tag=args.method_tag,
        base_output_dir=args.base_output_dir,
        rows=rows,
        total=total,
        expected_shards=args.expected_shards,
        expected_total=args.expected_total,
    )
    print(f"[sharded-summary] Wrote {write_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
