
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R10A2_3_external_feature_lock.py

Purpose:
    Remove outcome-bearing / evaluation-bearing columns from the
    selected external feature asset before domain invariance analysis.

Protocol:
    - Keep only representation inputs.
    - Do not use external labels.
    - Preserve sample identity fields.

Input:
    selected external feature csv

Output:
    external_feature_lock.csv
    removed_columns_audit.csv
"""

import argparse
from pathlib import Path

import pandas as pd


KEEP_ID_KEYWORDS = [
    "sample_id",
    "case_id",
    "image_id",
    "model_family",
]

LEAKAGE_KEYWORDS = [
    "outcome",
    "label",
    "gt",
    "dice",
    "mask",
    "ground_truth",
    "target",
]


def should_remove(col):

    name = str(col).lower()

    for k in LEAKAGE_KEYWORDS:
        if k in name:
            return True

    return False


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True
    )

    parser.add_argument(
        "--output_dir",
        required=True
    )

    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(
        parents=True,
        exist_ok=True
    )

    df = pd.read_csv(args.input)

    remove = []
    keep = []

    for c in df.columns:

        if should_remove(c):
            remove.append(c)
        else:
            keep.append(c)

    locked = df[keep].copy()

    locked.to_csv(
        out / "external_feature_lock.csv",
        index=False
    )

    pd.DataFrame(
        {
            "removed_column": remove
        }
    ).to_csv(
        out / "removed_columns_audit.csv",
        index=False
    )

    print("===== Q1-R10A2-3 EXTERNAL FEATURE LOCK =====")
    print("Input rows:", len(df))
    print("Original columns:", len(df.columns))
    print("Removed columns:", len(remove))
    print("Final columns:", len(locked.columns))
    print("Output:", out)

    assert len(locked) == len(df)
    assert not any(
        should_remove(c)
        for c in locked.columns
    )

    print("PASS")


if __name__ == "__main__":
    main()
