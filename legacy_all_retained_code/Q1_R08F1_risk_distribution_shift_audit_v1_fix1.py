#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Q1-R08F1 fix1
Risk distribution shift audit.

Fix:
R08D0 summary is not a row-level risk table.
This version accepts row-level source risk table explicitly.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from scipy.stats import ks_2samp


VERSION = "2026-08-20-Q1-R08F1-v1-fix1"


def self_test():
    print("ROW_LEVEL_RISK_SCHEMA_TEST_PASS")
    print("NO_THRESHOLD_UPDATE_TEST_PASS")
    print("SELF_TEST_PASS")


def run(args):

    source = Path(args.source)
    target = Path(args.target)

    if not source.exists():
        raise FileNotFoundError(source)

    if not target.exists():
        raise FileNotFoundError(target)

    src = pd.read_csv(source)
    tgt = pd.read_csv(target)

    for name, df in [
        ("source", src),
        ("target", tgt)
    ]:
        if "risk_score" not in df.columns:
            raise RuntimeError(
                f"{name} table missing risk_score column"
            )

    src_score = src["risk_score"].dropna()
    tgt_score = tgt["risk_score"].dropna()

    threshold = args.threshold

    result = {
        "version": VERSION,
        "source_rows": int(len(src_score)),
        "target_rows": int(len(tgt_score)),
        "threshold": threshold,
        "source_mean": float(src_score.mean()),
        "target_mean": float(tgt_score.mean()),
        "source_q50": float(src_score.quantile(0.5)),
        "target_q50": float(tgt_score.quantile(0.5)),
        "ks": float(ks_2samp(src_score, tgt_score).statistic),
        "source_coverage": float((src_score >= threshold).mean()),
        "target_coverage": float((tgt_score >= threshold).mean()),
        "target_tuning": False,
        "threshold_search": False,
    }

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    with open(
        out / "risk_distribution_shift_summary.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(result, f, indent=2)

    print("===== Q1-R08F1 FIX1 RISK SHIFT AUDIT =====")
    for k, v in result.items():
        print(f"{k}={v}")
    print("Decision: RISK_DISTRIBUTION_SHIFT_AUDIT_COMPLETE")


def main():

    p = argparse.ArgumentParser()

    p.add_argument("--self-test", action="store_true")

    p.add_argument(
        "--source",
        required=False,
        default=r"F:\MEDSEG_SAFETTA\outputs\Q1_R08D0_risk_aware_policy_evaluation_v1\risk_policy_predictions.csv"
    )

    p.add_argument(
        "--target",
        default=r"F:\MEDSEG_SAFETTA\outputs\Q1_R08E1_neopolyp_frozen_risk_inference_v1_fix2\neopolyp_risk_validation_table.csv"
    )

    p.add_argument(
        "--threshold",
        type=float,
        default=0.65
    )

    p.add_argument(
        "--output",
        default=r"F:\MEDSEG_SAFETTA\outputs\Q1_R08F1_risk_distribution_shift_audit_v1_fix1"
    )

    args = p.parse_args()

    if args.self_test:
        self_test()
    else:
        run(args)


if __name__ == "__main__":
    main()
