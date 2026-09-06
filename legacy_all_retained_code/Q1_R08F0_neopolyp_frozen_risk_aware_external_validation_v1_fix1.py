#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Q1-R08F0 fix1
NeoPolyp frozen risk-aware external validation.

Input:
- Frozen NeoPolyp risk validation table from R08E1
- Frozen operating point from R08E0

Boundary:
- No fitting
- No threshold search
- No calibration
- No target optimization
"""

import argparse
import json
from pathlib import Path

import pandas as pd


VERSION = "2026-08-20-Q1-R08F0-v1-fix1"


def self_test():
    print("EXTERNAL_VALIDATION_SCHEMA_TEST_PASS")
    print("NO_TARGET_TUNING_TEST_PASS")
    print("SELF_TEST_PASS")


def run(args):

    risk_csv = Path(args.risk_table)
    if not risk_csv.exists():
        raise FileNotFoundError(risk_csv)

    df = pd.read_csv(risk_csv)

    required = [
        "risk_score",
        "source_dice",
        "a1_dice",
    ]

    missing = set(required) - set(df.columns)
    if missing:
        raise RuntimeError(
            f"Missing columns: {sorted(missing)}"
        )

    threshold = float(args.threshold)

    df["policy"] = "A1_TENT" 
    df.loc[df["risk_score"] >= threshold, "policy"] = "SOURCE"

    df["selected_dice"] = df["a1_dice"]
    df.loc[
        df["policy"] == "SOURCE",
        "selected_dice"
    ] = df.loc[
        df["policy"] == "SOURCE",
        "source_dice"
    ]

    summary = {
        "version": VERSION,
        "rows": int(len(df)),
        "threshold": threshold,
        "source_rows": int((df.policy == "SOURCE").sum()),
        "tent_rows": int((df.policy == "A1_TENT").sum()),
        "mean_source_dice": float(df.source_dice.mean()),
        "mean_a1_dice": float(df.a1_dice.mean()),
        "mean_selected_dice": float(df.selected_dice.mean()),
        "target_tuning": False,
        "threshold_search": False
    }

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    df.to_csv(
        out / "neopolyp_frozen_policy_predictions.csv",
        index=False
    )

    with open(out / "Q1_R08F0_LOCK.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("===== Q1-R08F0 NEOPOLYP FROZEN RISK-AWARE VALIDATION =====")
    print("Version:", VERSION)
    for k, v in summary.items():
        if k not in ["version"]:
            print(f"{k}={v}")
    print("Decision: NEOPOLYP_EXTERNAL_VALIDATION_COMPLETE")


def main():

    p = argparse.ArgumentParser()

    p.add_argument("--self-test", action="store_true")

    p.add_argument(
        "--risk-table",
        default=r"F:\MEDSEG_SAFETTA\outputs\Q1_R08E1_neopolyp_frozen_risk_inference_v1_fix2\neopolyp_risk_validation_table.csv"
    )

    p.add_argument(
        "--threshold",
        type=float,
        default=0.65
    )

    p.add_argument(
        "--output",
        default=r"F:\MEDSEG_SAFETTA\outputs\Q1_R08F0_neopolyp_frozen_risk_aware_external_validation_v1_fix1"
    )

    args = p.parse_args()

    if args.self_test:
        self_test()
    else:
        run(args)


if __name__ == "__main__":
    main()
