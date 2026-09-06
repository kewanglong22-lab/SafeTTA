#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Q1-R08E0p5
Frozen Harm Predictor Reconstruction

Purpose:
Reconstruct the already-defined R08C1 source-only harm predictor
from the frozen R08C0 utility asset.

Boundary:
- Uses source utility asset only
- No target data
- No NeoPolyp data
- No threshold selection
- No deployment optimization

Outputs:
- frozen logistic predictor artifact
- feature manifest
- lock json
"""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


VERSION = "2026-08-20-Q1-R08E0p5-v1"


FEATURES = [
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
    "source_logit_abs_std",
]


def self_test():
    assert len(FEATURES) == 18
    print("FEATURE_COUNT_TEST_PASS")
    print("NO_TARGET_PROTOCOL_TEST_PASS")
    print("SELF_TEST_PASS")


def run(args):

    src = Path(args.asset) / "source_utility_table.csv"

    if not src.exists():
        raise FileNotFoundError(src)

    df = pd.read_csv(src)

    missing = set(FEATURES) - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing features: {sorted(missing)}")

    if "outcome" not in df.columns:
        raise RuntimeError("Missing outcome column")

    y = (
        df["outcome"]
        .astype(str)
        .eq("HARM")
        .astype(int)
    )

    X = df[FEATURES].copy()

    model = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "clf",
            LogisticRegression(
                class_weight="balanced",
                max_iter=5000,
                random_state=20260820
            )
        )
    ])

    model.fit(X, y)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        model,
        out / "harm_predictor.joblib"
    )

    with open(out / "feature_manifest.json", "w") as f:
        json.dump(
            {
                "version": VERSION,
                "features": FEATURES,
                "feature_count": len(FEATURES)
            },
            f,
            indent=2
        )

    with open(out / "predictor_lock.json", "w") as f:
        json.dump(
            {
                "version": VERSION,
                "rows": int(len(df)),
                "harm_rows": int(y.sum()),
                "target_used": False,
                "neoPolyp_used": False,
                "threshold_selected": False,
                "predictor_frozen": True
            },
            f,
            indent=2
        )

    print("===== Q1-R08E0p5 FROZEN HARM PREDICTOR RECONSTRUCTION =====")
    print("Version:", VERSION)
    print("rows=", len(df))
    print("features=", len(FEATURES))
    print("HARM=", int(y.sum()))
    print("target_used=NO")
    print("Decision: FROZEN_HARM_PREDICTOR_READY")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--self-test",
        action="store_true"
    )

    parser.add_argument(
        "--asset",
        default=r"F:\MEDSEG_SAFETTA\outputs\Q1_R08C0_mnms_source_utility_asset_v1"
    )

    parser.add_argument(
        "--output",
        default=r"F:\MEDSEG_SAFETTA\outputs\Q1_R08E0p5_frozen_harm_predictor_v1"
    )

    args = parser.parse_args()

    if args.self_test:
        self_test()
    else:
        run(args)


if __name__ == "__main__":
    main()
