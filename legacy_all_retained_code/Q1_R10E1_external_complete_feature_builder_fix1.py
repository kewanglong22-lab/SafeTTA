
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R10E1_external_complete_feature_builder_fix1.py

Fix:
    R10A2_8 already contains structure features.
    Normalize duplicated/renamed structure columns and export
    complete external invariant feature table.
"""

import argparse
from pathlib import Path
import pandas as pd


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--external_invariant", required=True)
    parser.add_argument("--output_dir", required=True)

    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(
        args.external_invariant,
        low_memory=False
    )

    print("===== EXTERNAL COMPLETE FEATURE FIX1 =====")
    print("Rows:", len(df))

    rename = {}

    if "source_fg_fraction.1" in df.columns:
        rename["source_fg_fraction.1"] = "struct_source_fg_fraction"

    if "source_boundary_density.1" in df.columns:
        rename["source_boundary_density.1"] = "struct_source_boundary_density"

    df = df.rename(columns=rename)

    required = [
        "source_fg_fraction",
        "source_boundary_density",
        "struct_source_fg_fraction",
        "struct_source_boundary_density"
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    print("Required structure columns:")
    print(required)
    print("Missing:", missing)

    if missing:
        raise ValueError("Structure normalization failed")

    output = out / "external_complete_invariant_feature_table.csv"

    df.to_csv(
        output,
        index=False
    )

    print("Final rows:", len(df))
    print("Final columns:", len(df.columns))
    print("Output:", output)
    print("PASS")


if __name__ == "__main__":
    main()
