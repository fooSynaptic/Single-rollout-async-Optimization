#!/usr/bin/env python3
"""Convert MATH-lighteval to the GSM8K-style schema consumed by AReaL."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from datasets import DatasetDict, load_dataset

DATASET_ID = "DigitalLearningGmbH/MATH-lighteval"


def last_boxed(text: str) -> str:
    """Return the content of the last balanced ``\boxed{...}`` expression."""
    starts = [m.end() for m in re.finditer(r"\\boxed\s*\{", text)]
    if not starts:
        raise ValueError("solution has no \\boxed{...} answer")
    start = starts[-1]
    depth = 1
    for index in range(start, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:index]
    raise ValueError("unbalanced \\boxed{...} answer")


def convert_example(example: dict) -> dict:
    return {
        "question": example["problem"],
        "answer": last_boxed(example["solution"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/gsm8k_hard"),
        help="Output directory for save_to_disk().",
    )
    parser.add_argument(
        "--config",
        default="data",
        help="MATH-lighteval builder config used by the experiments.",
    )
    args = parser.parse_args()

    raw = load_dataset(DATASET_ID, args.config)
    converted = DatasetDict()
    for split, dataset in raw.items():
        converted[split] = dataset.map(
            convert_example,
            remove_columns=dataset.column_names,
            desc=f"convert {split}",
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    converted.save_to_disk(str(args.output))
    print(
        f"saved {args.output}: "
        + ", ".join(f"{split}={len(dataset)}" for split, dataset in converted.items())
    )


if __name__ == "__main__":
    main()
