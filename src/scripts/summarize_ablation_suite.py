#!/usr/bin/env python3
"""Print one ablation's benchmark table from sharded aggregate summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DISPLAY_NAMES = {
    "mmstar": "MMStar",
    "realworldqa": "RealWorldQA",
    "scienceqa": "ScienceQA-IMG",
    "hallusionbench": "HallusionBench",
    "mathvista": "MathVista",
    "logicvista": "LogicVista",
    "mmmu": "MMMU val",
}


def _pct(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value) * 100:.2f}"


def _signed_pp(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value):+.2f}"


def _find_summary(suite_dir: Path, benchmark: str) -> Path | None:
    candidates = sorted(
        (suite_dir / benchmark).glob("*/aggregate_summary.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-dir", type=Path, default=None)
    parser.add_argument("--suite-name", default="")
    parser.add_argument("--result-root", type=Path, default=Path("result"))
    parser.add_argument("--benchmarks", default="mmstar,realworldqa,scienceqa,hallusionbench,mathvista,logicvista,mmmu")
    parser.add_argument("--ablation", default="")
    parser.add_argument("--model-family", default="")
    parser.add_argument("--write-json", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    suite_dir = args.suite_dir
    if suite_dir is None:
        if not args.suite_name:
            raise SystemExit("--suite-dir or --suite-name is required")
        suite_dir = args.result_root / "suites" / args.suite_name
    suite_dir = suite_dir.resolve()

    benchmarks = [item.strip() for item in args.benchmarks.split(",") if item.strip()]
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for benchmark in benchmarks:
        summary_path = _find_summary(suite_dir, benchmark)
        if summary_path is None:
            missing.append(benchmark)
            rows.append(
                {
                    "benchmark": benchmark,
                    "display": DISPLAY_NAMES.get(benchmark, benchmark),
                    "status": "MISSING",
                }
            )
            continue
        data = json.loads(summary_path.read_text())
        total = data.get("total", {})
        rows.append(
            {
                "benchmark": benchmark,
                "display": DISPLAY_NAMES.get(benchmark, benchmark),
                "status": "OK",
                "summary_path": str(summary_path),
                "evaluated_total": total.get("evaluated_total", 0),
                "original_correct": total.get("original_correct", 0),
                "optimized_correct": total.get("optimized_correct", 0),
                "original_accuracy": total.get("original_accuracy"),
                "optimized_accuracy": total.get("optimized_accuracy"),
                "delta_correct": total.get("delta_correct", 0),
                "delta_pp": total.get("delta_pp"),
            }
        )

    valid_rows = [row for row in rows if row.get("status") == "OK" and row.get("optimized_accuracy") is not None]
    macro_original = sum(float(row["original_accuracy"]) for row in valid_rows) / len(valid_rows) if valid_rows else None
    macro_optimized = sum(float(row["optimized_accuracy"]) for row in valid_rows) / len(valid_rows) if valid_rows else None
    macro_delta = (
        (macro_optimized - macro_original) * 100
        if macro_original is not None and macro_optimized is not None
        else None
    )

    print("")
    print("[ablation-suite-summary]")
    print(f"Ablation: {args.ablation or 'NA'}")
    print(f"Model: {args.model_family or 'NA'}")
    print(f"Suite: {suite_dir}")
    print("Benchmark        Original  Optimized   Delta   Correct(opt/orig)  Evaluated")
    print("---------------  --------  ---------  ------  -----------------  ---------")
    for row in rows:
        if row.get("status") != "OK":
            print(f"{row['display']:<15}  MISSING")
            continue
        correct = f"{row['optimized_correct']}/{row['original_correct']}"
        print(
            f"{row['display']:<15}  "
            f"{_pct(row['original_accuracy']):>8}  "
            f"{_pct(row['optimized_accuracy']):>9}  "
            f"{_signed_pp(row['delta_pp']):>6}  "
            f"{correct:>17}  "
            f"{int(row['evaluated_total']):>9}"
        )
    print("---------------  --------  ---------  ------  -----------------  ---------")
    print(
        f"{'Macro avg':<15}  "
        f"{_pct(macro_original):>8}  "
        f"{_pct(macro_optimized):>9}  "
        f"{_signed_pp(macro_delta):>6}"
    )
    if missing:
        print(f"WARN missing summaries for: {','.join(missing)}")
    print("")

    output = {
        "ablation": args.ablation,
        "model_family": args.model_family,
        "suite_dir": str(suite_dir),
        "macro_original": macro_original,
        "macro_optimized": macro_optimized,
        "macro_delta_pp": macro_delta,
        "rows": rows,
        "missing": missing,
    }
    write_json = args.write_json or suite_dir / "ablation_suite_summary.json"
    write_json.parent.mkdir(parents=True, exist_ok=True)
    write_json.write_text(json.dumps(output, indent=2) + "\n")
    print(f"[ablation-suite-summary] Wrote {write_json}")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
