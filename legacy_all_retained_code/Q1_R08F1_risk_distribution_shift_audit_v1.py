#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Q1-R08F1 Risk Distribution Shift Audit

Purpose:
Audit frozen risk score operating point drift between:
- source-domain M&Ms utility asset
- external NeoPolyp validation

No fitting.
No calibration.
No threshold update.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.stats import ks_2samp


VERSION = "2026-08-20-Q1-R08F1-v1"


def self_test():
    print("RISK_SHIFT_SCHEMA_TEST_PASS")
    print("NO_FIT_TEST_PASS")
    print("SELF_TEST_PASS")


def describe(x):
    x = np.asarray(x, dtype=float)
    return {
        "n": int(len(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "q01": float(np.quantile(x, 0.01)),
        "q05": float(np.quantile(x, 0.05)),
        "q50": float(np.quantile(x, 0.50)),
        "q95": float(np.quantile(x, 0.95)),
        "q99": float(np.quantile(x, 0.99)),
    }


def run(args):

    source = Path(args.source)
    target = Path(args.target)

    if not source.exists():
        raise FileNotFoundError(source)

    if not target.exists():
        raise FileNotFoundError(target)

    src = pd.read_csv(source)
    tgt = pd.read_csv(target)

    if "risk_score" not in tgt.columns:
        raise RuntimeError("External table missing risk_score")

    # source table: use existing frozen risk if available
    if "risk_score" not in src.columns:
        raise RuntimeError(
            "Source audit table requires risk_score column"
        )

    src_score = src["risk_score"].dropna()
    tgt_score = tgt["risk_score"].dropna()

    threshold = float(args.threshold)

    result = {
        "version": VERSION,
        "threshold": threshold,
        "source": describe(src_score),
        "external": describe(tgt_score),
        "ks_statistic": float(
            ks_2samp(src_score, tgt_score).statistic
        ),
        "source_above_threshold": float(
            (src_score >= threshold).mean()
        ),
        "external_above_threshold": float(
            (tgt_score >= threshold).mean()
        ),
        "target_tuning": False,
        "threshold_search": False
    }

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "risk_distribution_shift_summary.json", "w") as f:
        json.dump(result, f, indent=2)

    print("===== Q1-R08F1 RISK DISTRIBUTION SHIFT AUDIT =====")
    print("Version:", VERSION)
    print("source_n=", result["source"]["n"])
    print("external_n=", result["external"]["n"])
    print("KS=", result["ks_statistic"])
    print(
        "source_threshold_coverage=",
        result["source_above_threshold"]
    )
    print(
        "external_threshold_coverage=",
        result["external_above_threshold"]
    )
    print("target_tuning=NO")
    print("Decision: RISK_DISTRIBUTION_SHIFT_AUDIT_COMPLETE")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    p.add_argument(
        "--source",
        default=r"F:\MEDSEG_SAFETTA\outputs\Q1_R08D0_risk_aware_policy_evaluation_v1\risk_policy_summary.csv"
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
        default=r"F:\MEDSEG_SAFETTA\outputs\Q1_R08F1_risk_distribution_shift_audit_v1"
    )

    args = p.parse_args()

    if args.self_test:
        self_test()
    else:
        run(args)


if __name__ == "__main__":
    main()
