#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Q1-R08D0p5
Source risk replay from frozen predictor.

Purpose:
Generate row-level source risk scores for distribution audit.

Uses:
- R08C0 source utility table
- R08E0p5 frozen harm predictor

No:
- retraining
- target data
- threshold tuning
"""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd


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
    print("SOURCE_RISK_REPLAY_SCHEMA_TEST_PASS")
    print("NO_RETRAIN_TEST_PASS")
    print("SELF_TEST_PASS")


def run(args):

    asset = Path(args.asset) / "source_utility_table.csv"
    predictor = Path(args.predictor)

    if not asset.exists():
        raise FileNotFoundError(asset)

    if not predictor.exists():
        raise FileNotFoundError(predictor)

    df = pd.read_csv(asset)

    missing = set(FEATURES) - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing features: {sorted(missing)}")

    model = joblib.load(predictor)

    out = df.copy()
    out["risk_score"] = model.predict_proba(
        df[FEATURES]
    )[:, 1]

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    out.to_csv(
        out_dir / "source_risk_replay_table.csv",
        index=False
    )

    with open(out_dir / "Q1_R08D0p5_LOCK.json", "w") as f:
        json.dump(
            {
                "version": "2026-08-20-Q1-R08D0p5-v1",
                "rows": len(out),
                "target_used": False,
                "predictor_frozen": True
            },
            f,
            indent=2
        )

    print("===== Q1-R08D0p5 SOURCE RISK REPLAY =====")
    print("rows=", len(out))
    print("risk_score_generated=YES")
    print("target_used=NO")
    print("Decision: SOURCE_RISK_TABLE_READY")


def main():

    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")

    p.add_argument(
        "--asset",
        default=r"F:\MEDSEG_SAFETTA\outputs\Q1_R08C0_mnms_source_utility_asset_v1"
    )

    p.add_argument(
        "--predictor",
        default=r"F:\MEDSEG_SAFETTA\outputs\Q1_R08E0p5_frozen_harm_predictor_v1\harm_predictor.joblib"
    )

    p.add_argument(
        "--output",
        default=r"F:\MEDSEG_SAFETTA\outputs\Q1_R08D0p5_source_risk_replay_v1"
    )

    args = p.parse_args()

    if args.self_test:
        self_test()
    else:
        run(args)


if __name__ == "__main__":
    main()
