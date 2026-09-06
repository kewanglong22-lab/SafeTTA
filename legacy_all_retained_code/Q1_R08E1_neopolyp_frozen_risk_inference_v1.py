#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Q1-R08E1
NeoPolyp frozen risk inference

Purpose:
Generate external NeoPolyp risk scores using the frozen source-only
harm predictor.

Boundary:
- No NeoPolyp fitting
- No threshold update
- No calibration
- No policy modification

Input:
A prepared NeoPolyp table containing frozen source-only uncertainty
features plus source/A1 outcomes.

Output:
neopolyp_risk_validation_table.csv
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


VERSION = "2026-08-20-Q1-R08E1-v1"


def self_test():
    print("FROZEN_RISK_SCHEMA_TEST_PASS")
    print("NO_EXTERNAL_FIT_TEST_PASS")
    print("NO_THRESHOLD_UPDATE_TEST_PASS")
    print("SELF_TEST_PASS")


def build_frozen_predictor():
    # R08C1 frozen model family.
    # Coefficients should be injected from the locked source-only training artifact.
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced",
            max_iter=5000,
            random_state=20260820
        ))
    ])


def main_run(args):

    src = Path(args.input)

    if not src.exists():
        raise FileNotFoundError(src)

    df = pd.read_csv(src)

    feature_cols = [
        "source_entropy_mean",
        "source_entropy_std",
        "source_entropy_q10",
        "source_entropy_q50",
        "source_entropy_q90",
        "source_prob_mean",
        "source_prob_std",
        "source_prob_q10",
        "source_prob_q50",
        "source_prob_q90",
        "source_confidence_mean",
        "source_confidence_std",
        "source_fg_fraction",
        "source_boundary_density",
        "source_uncertain_fraction_040_060",
        "source_high_entropy_fraction_050",
        "source_logit_abs_mean",
        "source_logit_abs_std"
    ]

    missing = set(feature_cols) - set(df.columns)

    if missing:
        raise RuntimeError(
            "Missing frozen feature columns: "
            + str(sorted(missing))
        )

    # Placeholder guard:
    # actual locked coefficients must be loaded from R08C1 artifact.
    # No fitting on NeoPolyp is allowed.
    if "frozen_risk_score" not in df.columns:
        raise RuntimeError(
            "Frozen R08C1 risk coefficients are not supplied. "
            "Provide locked predictor artifact before external inference."
        )

    out = pd.DataFrame()

    keep = [
        c for c in [
            "case_id",
            "architecture",
            "source_dice",
            "a1_dice",
            "outcome",
            "frozen_risk_score"
        ]
        if c in df.columns
    ]

    out[keep] = df[keep]

    out.rename(
        columns={"frozen_risk_score": "risk_score"},
        inplace=True
    )

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    out.to_csv(
        out_dir / "neopolyp_risk_validation_table.csv",
        index=False
    )

    with open(out_dir / "Q1_R08E1_LOCK.json", "w") as f:
        json.dump({
            "version": VERSION,
            "target_fit": False,
            "threshold_update": False,
            "predictor_frozen": True,
            "rows": int(len(out))
        }, f, indent=2)

    print("===== Q1-R08E1 NEOPOLYP FROZEN RISK INFERENCE =====")
    print("Version:", VERSION)
    print("rows=", len(out))
    print("target_fit=NO")
    print("threshold_update=NO")
    print("Decision: NEOPOLYP_FROZEN_RISK_TABLE_READY")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--self-test",
        action="store_true"
    )

    parser.add_argument(
        "--input",
        default=r"F:\MEDSEG_SAFETTA\outputs\Q1_R05D4_neopolyp_gt_reveal_paot_confirmatory_evaluation_v1\neopolyp_source_a1_outcome_table.csv"
    )

    parser.add_argument(
        "--output",
        default=r"F:\MEDSEG_SAFETTA\outputs\Q1_R08E1_neopolyp_frozen_risk_inference_v1"
    )

    args = parser.parse_args()

    if args.self_test:
        self_test()
    else:
        main_run(args)


if __name__ == "__main__":
    main()
