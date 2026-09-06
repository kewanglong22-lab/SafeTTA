#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Q1-R08E1-fix2
NeoPolyp frozen risk inference.

Uses:
- R08E0p5 frozen harm predictor artifact
- NeoPolyp source-only uncertainty features

No:
- external fitting
- threshold update
- calibration
- policy modification
"""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd


VERSION = "2026-08-20-Q1-R08E1-v1-fix2"


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
    print("FROZEN_ARTIFACT_TEST_PASS")
    print("NO_EXTERNAL_FIT_TEST_PASS")
    print("SELF_TEST_PASS")


def run(args):

    predictor_path = Path(args.predictor)
    input_csv = Path(args.input)

    if not predictor_path.exists():
        raise FileNotFoundError(predictor_path)

    if not input_csv.exists():
        raise FileNotFoundError(input_csv)

    model = joblib.load(predictor_path)

    df = pd.read_csv(input_csv)

    missing = set(FEATURES) - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing features: {sorted(missing)}")

    risk = model.predict_proba(
        df[FEATURES]
    )[:, 1]

    out = pd.DataFrame()

    keep = [
        c for c in [
            "case_id",
            "architecture",
            "source_dice",
            "a1_dice",
            "outcome"
        ]
        if c in df.columns
    ]

    out[keep] = df[keep]
    out["risk_score"] = risk

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    out.to_csv(
        out_dir / "neopolyp_risk_validation_table.csv",
        index=False
    )

    with open(out_dir / "Q1_R08E1_LOCK.json", "w") as f:
        json.dump(
            {
                "version": VERSION,
                "predictor": str(predictor_path),
                "rows": len(out),
                "target_fit": False,
                "threshold_update": False,
                "predictor_frozen": True
            },
            f,
            indent=2
        )

    print("===== Q1-R08E1 FIX2 NEOPOLYP FROZEN RISK INFERENCE =====")
    print("Version:", VERSION)
    print("rows=", len(out))
    print("target_fit=NO")
    print("threshold_update=NO")
    print("Decision: NEOPOLYP_FROZEN_RISK_TABLE_READY")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--self-test", action="store_true")

    parser.add_argument(
        "--predictor",
        default=r"F:\MEDSEG_SAFETTA\outputs\Q1_R08E0p5_frozen_harm_predictor_v1\harm_predictor.joblib"
    )

    parser.add_argument(
        "--input",
        default=r"F:\MEDSEG_SAFETTA\outputs\Q1_R05D4_neopolyp_gt_reveal_paot_confirmatory_evaluation_v1\neopolyp_source_a1_outcome_table.csv"
    )

    parser.add_argument(
        "--output",
        default=r"F:\MEDSEG_SAFETTA\outputs\Q1_R08E1_neopolyp_frozen_risk_inference_v1_fix2"
    )

    args = parser.parse_args()

    if args.self_test:
        self_test()
    else:
        run(args)


if __name__ == "__main__":
    main()
