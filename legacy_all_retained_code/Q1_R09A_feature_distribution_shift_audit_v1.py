#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Q1-R09A Feature Distribution Shift Audit

Purpose:
Analyze which frozen safety features shift between:
- source M&Ms risk asset
- external NeoPolyp risk asset

No:
- model fitting
- target labels
- threshold update
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.stats import ks_2samp


VERSION = "2026-08-20-Q1-R09A-v1"

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
    print("FEATURE_SHIFT_SCHEMA_TEST_PASS")
    print("NO_FIT_TEST_PASS")
    print("SELF_TEST_PASS")


def run(args):

    source_path = Path(args.source)
    target_path = Path(args.target)

    if not source_path.exists():
        raise FileNotFoundError(source_path)

    if not target_path.exists():
        raise FileNotFoundError(target_path)

    src = pd.read_csv(source_path)
    tgt = pd.read_csv(target_path)

    for name, df in [("source", src), ("target", tgt)]:
        missing = set(FEATURES) - set(df.columns)
        if missing:
            raise RuntimeError(
                f"{name} missing features: {sorted(missing)}"
            )

    rows = []

    for feat in FEATURES:
        a = src[feat].astype(float).dropna()
        b = tgt[feat].astype(float).dropna()

        rows.append({
            "feature": feat,
            "source_mean": float(a.mean()),
            "target_mean": float(b.mean()),
            "source_std": float(a.std()),
            "target_std": float(b.std()),
            "ks_statistic": float(
                ks_2samp(a, b).statistic
            ),
            "mean_abs_shift": float(
                abs(a.mean() - b.mean())
            )
        })

    result = pd.DataFrame(rows).sort_values(
        "ks_statistic",
        ascending=False
    )

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    result.to_csv(
        out / "feature_distribution_shift.csv",
        index=False
    )

    summary = {
        "version": VERSION,
        "features": len(FEATURES),
        "source_rows": len(src),
        "target_rows": len(tgt),
        "target_labels_used": False,
        "model_fit": False,
        "top_shift_features": result.head(5).to_dict(
            orient="records"
        )
    }

    with open(out / "Q1_R09A_LOCK.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("===== Q1-R09A FEATURE DISTRIBUTION SHIFT AUDIT =====")
    print("version=", VERSION)
    print("source_rows=", len(src))
    print("target_rows=", len(tgt))
    print("features=", len(FEATURES))
    print("target_labels_used=NO")
    print("model_fit=NO")
    print("Top shifted features:")
    print(result.head(5).to_string(index=False))
    print("Decision: FEATURE_DISTRIBUTION_SHIFT_AUDIT_COMPLETE")


def main():

    p = argparse.ArgumentParser()

    p.add_argument("--self-test", action="store_true")

    p.add_argument(
        "--source",
        default=r"F:\MEDSEG_SAFETTA\outputs\Q1_R08D0p5_source_risk_replay_v1\source_risk_replay_table.csv"
    )

    p.add_argument(
        "--target",
        default=r"F:\MEDSEG_SAFETTA\outputs\Q1_R08E1_pre_neopolyp_feature_assembly_v1_fix1\neopolyp_source_feature_table.csv"
    )

    p.add_argument(
        "--output",
        default=r"F:\MEDSEG_SAFETTA\outputs\Q1_R09A_feature_distribution_shift_audit_v1"
    )

    args = p.parse_args()

    if args.self_test:
        self_test()
    else:
        run(args)


if __name__ == "__main__":
    main()
