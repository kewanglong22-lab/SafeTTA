
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R10A_0_invariant_feature_builder_fix1.py

Build domain-invariant safety representation features
from existing Q1_R08E1 feature assembly table.

Input schema:
    sample_id
    model_family
    source_entropy_*
    source_prob_*
    source_confidence_*
    source_fg_fraction
    source_boundary_density
    source_logit_abs_*

Protocol:
    - Fit reference statistics on source rows only.
    - Do not use target/external labels.
    - Keep original features.
    - Add Relative Uncertainty Representation (RUR).
    - Add structure invariant features.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


RUR_BASE_COLS = [
    "source_entropy_mean",
    "source_prob_mean",
    "source_confidence_mean",
    "source_logit_abs_mean",
]

STRUCTURE_COLS = [
    "source_fg_fraction",
    "source_boundary_density",
]


def robust_scale(x, median, mad):
    return (x - median) / max(1.4826 * mad, 1e-12)


def fit_reference(df):
    ref = {}

    for col in RUR_BASE_COLS:
        if col not in df.columns:
            continue

        x = df[col].astype(float).values

        ref[col] = {
            "mean": float(np.mean(x)),
            "std": float(np.std(x)),
            "median": float(np.median(x)),
            "mad": float(np.median(np.abs(x - np.median(x)))),
            "q01": float(np.quantile(x, 0.01)),
            "q99": float(np.quantile(x, 0.99)),
        }

    return ref


def build_rur(df, ref):
    out = pd.DataFrame(index=df.index)

    for col, stat in ref.items():

        x = df[col].astype(float)

        out[f"rur_{col}_z"] = (
            x - stat["mean"]
        ) / max(stat["std"], 1e-12)

        out[f"rur_{col}_mad"] = x.apply(
            lambda v: robust_scale(
                v,
                stat["median"],
                stat["mad"]
            )
        )

        out[f"rur_{col}_tail"] = np.maximum(
            stat["q01"] - x,
            x - stat["q99"]
        )

    return out


def build_structure(df):
    out = pd.DataFrame(index=df.index)

    for col in STRUCTURE_COLS:
        if col in df.columns:
            out[f"struct_{col}"] = df[col].astype(float)

    return out


def build(input_csv, output_csv):

    df = pd.read_csv(input_csv)

    required = [
        "sample_id",
        "model_family",
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    print("Input samples:", len(df))

    ref = fit_reference(df)

    rur = build_rur(df, ref)
    structure = build_structure(df)

    result = pd.concat(
        [
            df,
            rur,
            structure
        ],
        axis=1
    )

    Path(output_csv).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result.to_csv(
        output_csv,
        index=False
    )

    print("\n===== Q1-R10A-0 FEATURE BUILD =====")
    print("Input:", input_csv)
    print("Output:", output_csv)
    print("Original features:", len(df.columns))
    print("RUR features:", len(rur.columns))
    print("Structure features:", len(structure.columns))
    print("Final features:", len(result.columns))

    assert len(result) == len(df)
    assert len(result.columns) > len(df.columns)

    print("PASS")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True
    )

    parser.add_argument(
        "--output",
        required=True
    )

    args = parser.parse_args()

    build(
        args.input,
        args.output
    )


if __name__ == "__main__":
    main()
