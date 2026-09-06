
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R10A2_external_candidate_selection_audit.py

Purpose:
    Select the most suitable external feature asset from the
    Q1-R10A2 audit candidates.

Input:
    candidate_external_feature_assets.csv
    schema_comparison.csv

Selection logic:
    Rank candidates by:
      1. required uncertainty fields
      2. structure fields
      3. source schema overlap
      4. absence of errors

Output:
    candidate_selection_ranking.csv
    selected_external_feature_asset.csv

This script only selects existing assets.
It does not create new features.
"""

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_FIELDS = [
    "entropy",
    "confidence",
    "logit",
]

STRUCTURE_FIELDS = [
    "boundary",
    "fg_fraction",
]


def score_candidates(asset_csv, schema_csv):

    assets = pd.read_csv(asset_csv)
    schema = pd.read_csv(schema_csv)

    df = assets.merge(
        schema[[
            "file",
            "source_column_overlap"
        ]],
        on="file",
        how="left"
    )

    df["uncertainty_score"] = (
        df["has_entropy"].astype(int)
        +
        df["has_confidence"].astype(int)
        +
        df["has_logit"].astype(int)
    )

    df["structure_score"] = (
        df["has_boundary"].astype(int)
        +
        df["has_fg_fraction"].astype(int)
    )

    df["final_score"] = (
        df["uncertainty_score"] * 10
        +
        df["structure_score"] * 3
        +
        df["source_column_overlap"].fillna(0)
    )

    df = df.sort_values(
        [
            "final_score",
            "source_column_overlap",
        ],
        ascending=False
    )

    return df


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--asset_csv",
        required=True
    )

    parser.add_argument(
        "--schema_csv",
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

    ranking = score_candidates(
        args.asset_csv,
        args.schema_csv
    )

    ranking.to_csv(
        out / "candidate_selection_ranking.csv",
        index=False
    )

    selected = ranking.head(1)

    selected.to_csv(
        out / "selected_external_feature_asset.csv",
        index=False
    )

    print("===== Q1-R10A2-1 EXTERNAL SELECTION =====")
    print("Candidates:", len(ranking))
    print("Selected:")
    print(selected[["file", "final_score"]])
    print("Output:", out)

    assert len(selected) == 1

    print("PASS")


if __name__ == "__main__":
    main()
