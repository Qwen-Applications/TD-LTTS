#!/usr/bin/env python
"""Create a local MMMU validation split with multi-image cases removed."""

import argparse
from collections import Counter
from pathlib import Path

from datasets import DatasetDict, load_from_disk


def count_images(example):
    return sum(1 for idx in range(1, 8) if example.get(f"image_{idx}") is not None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="dataset/MMMU")
    parser.add_argument("--output", default="dataset/MMMU-val-single-image")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists() and not args.overwrite:
        print(f"Output exists: {output}")
        return

    dataset = load_from_disk(args.source)
    validation = dataset["validation"] if isinstance(dataset, DatasetDict) else dataset

    before_dist = Counter(count_images(example) for example in validation)
    filtered = validation.filter(lambda example: count_images(example) == 1)
    after_dist = Counter(count_images(example) for example in filtered)
    DatasetDict({"validation": filtered}).save_to_disk(str(output))

    print(f"source={args.source}")
    print(f"output={args.output}")
    print(f"validation_total={len(validation)}")
    print(f"single_image_total={len(filtered)}")
    print(f"removed_multi_image={len(validation) - len(filtered)}")
    print(f"before_image_count_dist={dict(sorted(before_dist.items()))}")
    print(f"after_image_count_dist={dict(sorted(after_dist.items()))}")


if __name__ == "__main__":
    main()
