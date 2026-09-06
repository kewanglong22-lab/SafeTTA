
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R10A2_8_external_rur_transform.py

Purpose:
    Apply source-fitted Relative Uncertainty Representation (RUR)
    transformation to external NeoPolyp features.

Important protocol:
    - Source statistics are fitted ONLY from source data.
    - External data is transformed using frozen source statistics.
    - No external labels are used.

Input:
    source invariant feature table (contains RUR reference source raw features)
    external locked raw feature table

Output:
    external invariant feature table with:
        raw features
        rur features
        structure features
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ID_COLS = [
    "sample_id",
    "model_family",
    "outcome",
]


def get_raw_features(df):

    exclude = [
        "rur_",
        "struct_",
        "domain",
        "outcome",
    ]

    cols = []

    for c in df.columns:
        if any(c.startswith(x) for x in exclude):
            continue
        if c not in ID_COLS:
            cols.append(c)

    return cols


def fit_source_stats(source, raw_cols):

    stats = {}

    for c in raw_cols:

        x = pd.to_numeric(
            source[c],
            errors="coerce"
        )

        stats[c] = {
            "mean": float(x.mean()),
            "std": float(x.std() + 1e-8),
            "median": float(x.median()),
            "mad": float(
                np.median(
                    np.abs(x - x.median())
                ) + 1e-8
            ),
        }

    return stats


def apply_rur(df, stats):

    out = pd.DataFrame(index=df.index)

    for c, s in stats.items():

        x = pd.to_numeric(
            df[c],
            errors="coerce"
        )

        out[f"rur_{c}_z"] = (
            (x - s["mean"])
            /
            s["std"]
        )

        out[f"rur_{c}_mad"] = (
            (x - s["median"])
            /
            s["mad"]
        )

    return out


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        required=True
    )

    parser.add_argument(
        "--external",
        required=True
    )

    parser.add_argument(
        "--output_dir",
        required=True
    )

    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(
        parents=True,
        exist_ok=True
    )

    source = pd.read_csv(args.source)
    external = pd.read_csv(args.external)

    raw_cols = get_raw_features(source)

    common_raw = [
        c for c in raw_cols
        if c in external.columns
    ]

    stats = fit_source_stats(
        source,
        common_raw
    )

    external_rur = apply_rur(
        external,
        stats
    )

    structure_cols = [
        c for c in external.columns
        if (
            "boundary" in c
            or "fg_fraction" in c
        )
    ]

    locked = pd.concat(
        [
            external[
                ["sample_id"] + common_raw
            ],
            external_rur,
            external[structure_cols],
        ],
        axis=1
    )

    locked.to_csv(
        outdir /
        "neopolyp_external_invariant_feature_table.csv",
        index=False
    )

    pd.DataFrame(
        {
            "raw_features": common_raw,
            "count": [len(common_raw)] * len(common_raw)
        }
    ).to_csv(
        outdir /
        "rur_source_fitted_feature_list.csv",
        index=False
    )

    print("===== Q1-R10A2-8 EXTERNAL RUR TRANSFORM =====")
    print("Source rows:", len(source))
    print("External rows:", len(external))
    print("Raw features:", len(common_raw))
    print("RUR features:", external_rur.shape[1])
    print("Structure features:", len(structure_cols))
    print("Output:", outdir)

    assert len(locked) == len(external)

    print("PASS")


if __name__ == "__main__":
    main()
