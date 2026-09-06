
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R10A2_4_external_asset_reselection_fix1.py

Purpose:
    Re-select external feature asset correctly.

Fixes:
    1. Exclude source feature asset.
    2. Prefer external/target/neopolyp candidates.
    3. Rank by feature compatibility.
    4. Avoid selecting the source table itself.

Input:
    candidate_external_feature_assets.csv
    schema_comparison.csv

Output:
    candidate_external_feature_ranking_fix1.csv
    selected_external_feature_asset_fix1.csv
"""

import argparse
from pathlib import Path
import pandas as pd


def keyword_score(path):

    name = str(path).lower()

    score = 0

    for k in ["external", "target", "neopolyp", "neo"]:
        if k in name:
            score += 5

    for k in ["source_feature_assembly", "source_feature_table"]:
        if k in name:
            score -= 100

    return score


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--asset_csv", required=True)
    parser.add_argument("--schema_csv", required=True)
    parser.add_argument("--output_dir", required=True)

    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    assets = pd.read_csv(args.asset_csv)
    schema = pd.read_csv(args.schema_csv)

    df = assets.merge(
        schema[["file", "source_column_overlap"]],
        on="file",
        how="left"
    )

    df["path_score"] = df["file"].apply(keyword_score)

    df["feature_score"] = (
        df["matched_score"].fillna(0) * 10
        + df["source_column_overlap"].fillna(0)
        + df["path_score"]
    )

    df = df.sort_values(
        "feature_score",
        ascending=False
    )

    # hard reject source table candidates
    df = df[
        ~df["file"].str.contains(
            "pre_neopolyp_feature_assembly",
            case=False,
            na=False
        )
    ]

    df.to_csv(
        out / "candidate_external_feature_ranking_fix1.csv",
        index=False
    )

    selected = df.head(1)

    selected.to_csv(
        out / "selected_external_feature_asset_fix1.csv",
        index=False
    )

    print("===== Q1-R10A2-4 EXTERNAL RESELECTION FIX1 =====")
    print("Remaining candidates:", len(df))

    if len(selected):
        print(selected[["file", "feature_score"]])

    print("Output:", out)

    assert len(selected) == 1

    print("PASS")


if __name__ == "__main__":
    main()
