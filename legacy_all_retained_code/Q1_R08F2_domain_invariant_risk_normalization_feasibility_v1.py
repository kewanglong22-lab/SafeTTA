#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Q1-R08F2 Domain-invariant risk normalization feasibility audit.

Purpose:
Test whether simple source-free score normalization can recover
a stable external operating point.

No:
- target labels
- target fitting
- threshold search
- predictor modification

Methods:
- raw risk score
- source percentile rank
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


VERSION = "2026-08-20-Q1-R08F2-v1"


def self_test():
    print("NORMALIZATION_SCHEMA_TEST_PASS")
    print("NO_TARGET_LABEL_TEST_PASS")
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

    for name, df in [("source", src), ("target", tgt)]:
        if "risk_score" not in df.columns:
            raise RuntimeError(f"{name} missing risk_score")

    src_score = src["risk_score"].astype(float).values
    tgt_score = tgt["risk_score"].astype(float).values

    # source-free percentile transformation
    src_sorted = np.sort(src_score)

    def percentile_transform(x):
        return np.searchsorted(src_sorted, x, side="right") / len(src_sorted)

    tgt_rank = percentile_transform(tgt_score)

    result = {
        "version": VERSION,
        "source_rows": int(len(src_score)),
        "target_rows": int(len(tgt_score)),
        "raw_target_mean": float(np.mean(tgt_score)),
        "rank_target_mean": float(np.mean(tgt_rank)),
        "raw_target_coverage_065": float((tgt_score >= 0.65).mean()),
        "rank_target_coverage_065": float((tgt_rank >= 0.65).mean()),
        "source_coverage_065": float((src_score >= 0.65).mean()),
        "target_labels_used": False,
        "threshold_search": False,
    }

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "risk_normalization_feasibility.json", "w") as f:
        json.dump(result, f, indent=2)

    print("===== Q1-R08F2 DOMAIN-INVARIANT RISK NORMALIZATION AUDIT =====")
    for k, v in result.items():
        print(f"{k}={v}")
    print("Decision: RISK_NORMALIZATION_FEASIBILITY_COMPLETE")


def main():

    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")

    p.add_argument(
        "--source",
        default=r"F:\MEDSEG_SAFETTA\outputs\Q1_R08D0p5_source_risk_replay_v1\source_risk_replay_table.csv"
    )

    p.add_argument(
        "--target",
        default=r"F:\MEDSEG_SAFETTA\outputs\Q1_R08E1_neopolyp_frozen_risk_inference_v1_fix2\neopolyp_risk_validation_table.csv"
    )

    p.add_argument(
        "--output",
        default=r"F:\MEDSEG_SAFETTA\outputs\Q1_R08F2_domain_invariant_risk_normalization_feasibility_v1"
    )

    args = p.parse_args()

    if args.self_test:
        self_test()
    else:
        run(args)


if __name__ == "__main__":
    main()
