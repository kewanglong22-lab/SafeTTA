
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R10A2_7_external_representation_lock.py

Purpose:
    Build the final NeoPolyp external representation table for R10A.

Input:
    target_paot_probabilities.csv

Protocol:
    Keep only safety representation features.
    Remove:
        - image metadata
        - checkpoint/model metadata
        - optimization/TENT states
        - final harm probabilities
        - MRZ derived outputs

Output:
    neopolyp_external_R10A_feature_lock.csv
    removed_columns_audit.csv
"""

import argparse
from pathlib import Path

import pandas as pd


KEEP_EXACT_OR_PREFIX = [
    "sample_id",
    "source_entropy_",
    "source_prob_",
    "source_confidence_",
    "source_fg_fraction",
    "source_boundary_density",
    "source_uncertain_fraction_",
    "source_high_entropy_fraction_",
    "source_logit_abs_",
]


def keep_column(col):
    return any(
        str(col).startswith(k)
        for k in KEEP_EXACT_OR_PREFIX
    )


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

    keep = [
        c for c in df.columns
        if keep_column(c)
    ]

    remove = [
        c for c in df.columns
        if c not in keep
    ]

    locked = df[keep].copy()

    locked.to_csv(
        out / "neopolyp_external_R10A_feature_lock.csv",
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

    print("===== Q1-R10A2-7 EXTERNAL REPRESENTATION LOCK =====")
    print("Input rows:", len(df))
    print("Original columns:", len(df.columns))
    print("Kept columns:", len(keep))
    print("Removed columns:", len(remove))
    print("Output:", out)

    assert len(locked) == len(df)
    assert "sample_id" in locked.columns

    print("PASS")


if __name__ == "__main__":
    main()
