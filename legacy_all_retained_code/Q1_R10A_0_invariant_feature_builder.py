
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1-R10A-0 Domain-Invariant Safety Representation Feature Builder

Purpose:
    Build invariant safety features from frozen segmentation-derived features.

Input:
    CSV containing sample-level safety features.

Required columns (example):
    sample_id, domain, label,
    confidence_mean,
    entropy_mean,
    logit_abs_mean,
    boundary_density,
    boundary_length_ratio,
    foreground_fraction

Output:
    Q1_R10A_feature_table.csv

Main transformations:
    1. Raw uncertainty features
    2. Relative uncertainty representation (RUR)
       - source reference normalization
       - robust MAD normalization
       - distribution deviation features
    3. Structure invariant features
"""

import argparse
import numpy as np
import pandas as pd


UNCERTAINTY_COLS = [
    "confidence_mean",
    "entropy_mean",
    "logit_abs_mean",
    "margin_mean",
]

STRUCTURE_COLS = [
    "boundary_density",
    "boundary_length_ratio",
    "foreground_fraction",
    "connected_component_count",
    "largest_component_ratio",
    "hole_count",
    "compactness",
    "eccentricity",
]


def robust_z(x, median, mad):
    mad = mad if mad > 1e-12 else 1e-12
    return (x - median) / (1.4826 * mad)


def fit_source_reference(df):
    """
    Fit population statistics only on source domain.
    """
    ref = {}

    source = df[df["domain"].astype(str).str.lower() == "source"]

    if len(source) == 0:
        raise RuntimeError("No source samples found.")

    for c in UNCERTAINTY_COLS:
        if c not in df.columns:
            continue

        values = source[c].astype(float).values

        ref[c] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "median": float(np.median(values)),
            "mad": float(np.median(np.abs(values - np.median(values)))),
            "q01": float(np.quantile(values, 0.01)),
            "q99": float(np.quantile(values, 0.99)),
        }

    return ref


def build_rur(df, ref):
    out = pd.DataFrame(index=df.index)

    for c, stat in ref.items():

        x = df[c].astype(float)

        out[f"rur_{c}_z"] = (
            x - stat["mean"]
        ) / max(stat["std"], 1e-12)

        out[f"rur_{c}_mad"] = x.apply(
            lambda v: robust_z(
                v,
                stat["median"],
                stat["mad"]
            )
        )

        out[f"rur_{c}_tail"] = np.maximum(
            stat["q01"] - x,
            x - stat["q99"]
        )

    return out


def build_structure(df):

    out = pd.DataFrame(index=df.index)

    for c in STRUCTURE_COLS:
        if c in df.columns:
            out[f"struct_{c}"] = df[c].astype(float)

    return out


def build_features(input_csv, output_csv):

    df = pd.read_csv(input_csv)

    required = ["domain"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    print("Samples:", len(df))

    ref = fit_source_reference(df)

    raw = df.copy()

    rur = build_rur(df, ref)
    structure = build_structure(df)

    result = pd.concat(
        [
            raw,
            rur,
            structure
        ],
        axis=1
    )

    result.to_csv(
        output_csv,
        index=False
    )

    print("\n===== Q1-R10A FEATURE BUILD =====")
    print("Input:", input_csv)
    print("Output:", output_csv)
    print("Raw columns:", len(raw.columns))
    print("RUR columns:", len(rur.columns))
    print("Structure columns:", len(structure.columns))
    print("Final columns:", len(result.columns))
    print("PASS")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="raw18 feature csv"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="output invariant feature csv"
    )

    args = parser.parse_args()

    build_features(
        args.input,
        args.output
    )


if __name__ == "__main__":
    main()
