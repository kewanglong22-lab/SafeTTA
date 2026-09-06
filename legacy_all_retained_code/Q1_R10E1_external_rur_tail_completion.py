
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R10E1_external_rur_tail_completion.py

Purpose:
    Add missing RUR tail features required by R10E0 checkpoint.

The current external invariant table already contains:
    *_z
    *_mad

but lacks:
    *_tail

This script reconstructs tail indicators using source-fitted statistics.
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--source", required=True)
    parser.add_argument("--external", required=True)
    parser.add_argument("--output_dir", required=True)

    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    source = pd.read_csv(args.source, low_memory=False)
    external = pd.read_csv(args.external, low_memory=False)

    print("===== R10E1 RUR TAIL COMPLETION =====")

    base_features = [
        "source_entropy_mean",
        "source_prob_mean",
        "source_confidence_mean",
        "source_logit_abs_mean",
    ]

    for f in base_features:

        if f not in source.columns:
            raise ValueError(f"Missing source feature: {f}")

        if f not in external.columns:
            raise ValueError(f"Missing external feature: {f}")

        median = source[f].median()
        mad = np.median(
            np.abs(
                source[f] - median
            )
        ) + 1e-8

        col = f"rur_{f}_tail"

        external[col] = (
            np.abs(
                external[f] - median
            ) / mad
        )

        print(
            col,
            "created"
        )

    output = (
        out /
        "external_complete_invariant_feature_table_v2.csv"
    )

    external.to_csv(
        output,
        index=False
    )

    required = [
        "rur_source_entropy_mean_tail",
        "rur_source_prob_mean_tail",
        "rur_source_confidence_mean_tail",
        "rur_source_logit_abs_mean_tail",
    ]

    missing = [
        c for c in required
        if c not in external.columns
    ]

    print("Missing:", missing)
    print("Rows:", len(external))
    print("Output:", output)

    if missing:
        raise ValueError("Tail completion failed")

    print("PASS")


if __name__ == "__main__":
    main()
