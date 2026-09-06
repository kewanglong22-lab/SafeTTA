
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R10A2_3_external_feature_lock_fix1.py

Read the selected external asset path automatically from:
    selected_external_feature_asset.csv

Then remove leakage/evaluation columns and lock the external
representation feature table.

Input:
    --selection_csv:
        Q1_R10A2_external_candidate_selection_v1/
        selected_external_feature_asset.csv

Output:
    external_feature_lock.csv
    removed_columns_audit.csv
    selected_asset_path.txt
"""

import argparse
from pathlib import Path

import pandas as pd


LEAKAGE_KEYWORDS = [
    "outcome",
    "label",
    "gt",
    "dice",
    "mask",
    "ground_truth",
    "target",
]


def is_leakage(col):
    name = str(col).lower()
    return any(k in name for k in LEAKAGE_KEYWORDS)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--selection_csv", required=True)
    parser.add_argument("--output_dir", required=True)

    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    selection = pd.read_csv(args.selection_csv)

    if "file" not in selection.columns:
        raise ValueError(
            "selected_external_feature_asset.csv missing file column"
        )

    external_path = str(selection["file"].iloc[0])

    df = pd.read_csv(external_path)

    remove_cols = [
        c for c in df.columns
        if is_leakage(c)
    ]

    keep_cols = [
        c for c in df.columns
        if c not in remove_cols
    ]

    locked = df[keep_cols].copy()

    locked.to_csv(
        out / "external_feature_lock.csv",
        index=False
    )

    pd.DataFrame(
        {"removed_column": remove_cols}
    ).to_csv(
        out / "removed_columns_audit.csv",
        index=False
    )

    (out / "selected_asset_path.txt").write_text(
        external_path,
        encoding="utf-8"
    )

    print("===== Q1-R10A2-3 EXTERNAL FEATURE LOCK FIX1 =====")
    print("Selected asset:", external_path)
    print("Rows:", len(df))
    print("Original columns:", len(df.columns))
    print("Removed columns:", len(remove_cols))
    print("Final columns:", len(locked.columns))
    print("Output:", out)

    assert len(locked) == len(df)
    assert not any(is_leakage(c) for c in locked.columns)

    print("PASS")


if __name__ == "__main__":
    main()
