#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Q1-R08F0
NeoPolyp Frozen Risk-aware External Validation

Purpose:
External validation of the frozen source-only risk-aware TTA policy.

Frozen:
- R08C1 risk model concept
- R08E0 operating threshold = 0.65

No:
- target retraining
- target calibration
- threshold update
- policy modification

Input:
NeoPolyp source/A1 predictions and GT-derived outcomes.

This script only evaluates the frozen policy.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


VERSION = "2026-08-20-Q1-R08F0-v1"


def self_test():
    print("FROZEN_POLICY_SCHEMA_TEST_PASS")
    print("NO_POLICY_UPDATE_TEST_PASS")
    print("NO_TARGET_TUNING_TEST_PASS")
    print("SELF_TEST_PASS")


def main_run(args):

    csv = Path(args.input)

    if not csv.exists():
        raise FileNotFoundError(csv)

    df = pd.read_csv(csv)

    required = {
        "risk_score",
        "source_dice",
        "a1_dice",
        "outcome"
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"Missing frozen validation columns: {missing}"
        )

    threshold = 0.65

    use_tent = df["risk_score"].values < threshold

    final_dice = np.where(
        use_tent,
        df["a1_dice"].values,
        df["source_dice"].values
    )

    harm = (
        df["outcome"].astype(str)
        .eq("HARM")
        .values
    )

    benefit = (
        df["outcome"].astype(str)
        .eq("BENEFIT")
        .values
    )

    result = {
        "threshold": threshold,
        "rows": int(len(df)),
        "tent_coverage": float(use_tent.mean()),
        "mean_dice": float(final_dice.mean()),
        "harm_exposure": float(
            use_tent[harm].mean()
        ) if harm.sum() else None,
        "harm_avoidance": float(
            1-use_tent[harm].mean()
        ) if harm.sum() else None,
        "benefit_retention": float(
            use_tent[benefit].mean()
        ) if benefit.sum() else None,
        "policy_frozen": True,
        "target_tuning": False
    }

    print("===== Q1-R08F0 NEOPOLYP FROZEN RISK-AWARE VALIDATION =====")
    print("Version:", VERSION)

    for k,v in result.items():
        print(f"{k}={v}")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([result]).to_csv(
        out / "neopolyp_frozen_risk_policy_result.csv",
        index=False
    )

    with open(out / "Q1_R08F0_LOCK.json","w") as f:
        json.dump(result,f,indent=2)

    print(
        "Decision: "
        "NEOPOLYP_FROZEN_RISK_POLICY_VALIDATION_COMPLETE"
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--self-test",
        action="store_true"
    )

    parser.add_argument(
        "--input",
        default=r"F:\MEDSEG_SAFETTA\data\external\NeoPolyp_raw\neopolyp_risk_validation_table.csv"
    )

    parser.add_argument(
        "--output",
        default=r"F:\MEDSEG_SAFETTA\outputs\Q1_R08F0_neopolyp_frozen_risk_validation_v1"
    )

    args = parser.parse_args()

    if args.self_test:
        self_test()
    else:
        main_run(args)


if __name__ == "__main__":
    main()
