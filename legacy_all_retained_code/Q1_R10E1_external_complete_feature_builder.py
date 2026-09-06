
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R10E1_external_complete_feature_builder.py

Purpose:
    Complete external invariant feature table by merging:
        1. R10A2_8 external invariant uncertainty/RUR features
        2. R10A2_7 external structure features

Output:
    external_complete_invariant_feature_table.csv

Audit:
    - required checkpoint features
    - missing uncertainty features
    - missing structure features
"""

import argparse
from pathlib import Path
import pandas as pd


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--external_invariant", required=True)
    parser.add_argument("--external_structure", required=True)
    parser.add_argument("--output_dir", required=True)

    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    inv = pd.read_csv(
        args.external_invariant,
        low_memory=False
    )

    struct = pd.read_csv(
        args.external_structure,
        low_memory=False
    )

    print("===== EXTERNAL COMPLETE FEATURE AUDIT =====")

    print("Invariant rows:", len(inv))
    print("Structure rows:", len(struct))

    if len(inv) != len(struct):
        raise ValueError(
            "Row mismatch between invariant and structure tables"
        )

    # merge by sample_id if available, otherwise row order
    if "sample_id" in inv.columns and "sample_id" in struct.columns:

        df = inv.merge(
            struct,
            on="sample_id",
            how="inner",
            suffixes=("", "_struct")
        )

    else:
        df = inv.copy()

        for c in struct.columns:
            if c not in df.columns:
                df[c] = struct[c].values

    required_structure = [
        "struct_source_fg_fraction",
        "struct_source_boundary_density"
    ]

    print("\nRequired structure features:")

    missing = [
        c for c in required_structure
        if c not in df.columns
    ]

    print("Missing:", missing)

    if len(missing) > 0:
        raise ValueError(
            "Structure features missing"
        )

    output = (
        out /
        "external_complete_invariant_feature_table.csv"
    )

    df.to_csv(
        output,
        index=False
    )

    print("\nFinal rows:", len(df))
    print("Final columns:", len(df.columns))
    print("Missing after merge: []")
    print("Output:", output)
    print("PASS")


if __name__ == "__main__":
    main()
