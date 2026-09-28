#!/usr/bin/env python
import argparse
import json
from collections import defaultdict
from pathlib import Path


def _load_records(test_dir: Path):
    records = []
    for path in sorted(test_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as f:
            records.append(json.load(f))
    return records


def _calc_aacc(records, key):
    vals = [1 if record.get(key) else 0 for record in records if record.get(key) is not None]
    if not vals:
        return None
    return 100.0 * sum(vals) / len(vals)


def _calc_group_consistency(records, key, group_key):
    groups = defaultdict(list)
    for record in records:
        value = record.get(key)
        if value is None:
            continue
        groups[group_key(record)].append(bool(value))
    if not groups:
        return None
    hits = [all(values) for values in groups.values()]
    return 100.0 * sum(hits) / len(hits)


def _build_split_metrics(records, key, split_name):
    subset = [record for record in records if split_name == "Overall" or record.get("source_split") == split_name]
    if not subset:
        return None
    return {
        "split": split_name,
        "aAcc": _calc_aacc(subset, key),
        "fAcc": _calc_group_consistency(
            subset,
            key,
            lambda r: f"{r.get('l2_category')}_{r.get('set_id')}_{r.get('figure_id')}",
        ),
        "qAcc": _calc_group_consistency(
            subset,
            key,
            lambda r: f"{r.get('l2_category')}_{r.get('set_id')}_{r.get('question_id')}",
        ),
        "count": len(subset),
    }


def _build_category_metrics(records, key, field_name):
    values = sorted({record.get(field_name) for record in records if record.get(field_name)})
    metrics = []
    for value in values:
        subset = [record for record in records if record.get(field_name) == value]
        metrics.append(
            {
                "split": value,
                "aAcc": _calc_aacc(subset, key),
                "fAcc": _calc_group_consistency(
                    subset,
                    key,
                    lambda r: f"{r.get('l2_category')}_{r.get('set_id')}_{r.get('figure_id')}",
                ),
                "qAcc": _calc_group_consistency(
                    subset,
                    key,
                    lambda r: f"{r.get('l2_category')}_{r.get('set_id')}_{r.get('question_id')}",
                ),
                "count": len(subset),
            }
        )
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=str)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    test_dir = run_dir / "test"
    records = _load_records(test_dir)
    if not records:
        raise SystemExit(f"No test json files found under {test_dir}")

    metrics = {
        "processed_total": len(records),
        "original": {
            "overall": _build_split_metrics(records, "original_true_false", "Overall"),
            "by_source_split": [
                _build_split_metrics(records, "original_true_false", name)
                for name in ["image", "non_image"]
                if _build_split_metrics(records, "original_true_false", name) is not None
            ],
            "by_category": _build_category_metrics(records, "original_true_false", "category"),
            "by_l2_category": _build_category_metrics(records, "original_true_false", "l2_category"),
        },
        "optimized": {
            "overall": _build_split_metrics(records, "optimized_true_false", "Overall"),
            "by_source_split": [
                _build_split_metrics(records, "optimized_true_false", name)
                for name in ["image", "non_image"]
                if _build_split_metrics(records, "optimized_true_false", name) is not None
            ],
            "by_category": _build_category_metrics(records, "optimized_true_false", "category"),
            "by_l2_category": _build_category_metrics(records, "optimized_true_false", "l2_category"),
        },
    }

    summary_path = run_dir / "hallusionbench_metrics.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(summary_path)
    print(json.dumps(metrics["optimized"]["overall"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
