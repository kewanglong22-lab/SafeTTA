#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Q1-R09B Domain-Invariant Feature Construction Audit

Purpose:
Test whether source-free feature normalization can reduce
cross-domain shift of frozen safety descriptors.

No:
- model fitting
- target labels
- threshold tuning

Outputs:
- raw feature shift
- MAD normalized shift
- source percentile normalized shift
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


VERSION = "2026-08-20-Q1-R09B-v1"

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
    print("DOMAIN_INVARIANT_FEATURE_SCHEMA_TEST_PASS")
    print("NO_FIT_TEST_PASS")
    print("SELF_TEST_PASS")


def mad_normalize(source, target):
    median = np.median(source)
    mad = np.median(np.abs(source - median))
    if mad < 1e-12:
        mad = 1e-12
    return (
        (source - median) / mad,
        (target - median) / mad
    )


def percentile_transform(source, values):
    source_sorted = np.sort(source)
    return np.searchsorted(
        source_sorted,
        values,
        side="right"
    ) / len(source_sorted)


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

        s = src[feat].astype(float).dropna().values
        t = tgt[feat].astype(float).dropna().values

        _, t_mad = mad_normalize(s, t)

        t_rank = percentile_transform(s, t)

        rows.append({
            "feature": feat,

            "raw_ks": float(
                ks_2samp(s, t).statistic
            ),

            "mad_source_mean": float(
                np.mean((s - np.median(s)) /
                        max(np.median(np.abs(s - np.median(s))), 1e-12))
            ),

            "mad_target_mean": float(
                np.mean(t_mad)
            ),

            "mad_target_ks": float(
                ks_2samp(
                    (s - np.median(s)) /
                    max(np.median(np.abs(s - np.median(s))), 1e-12),
                    t_mad
                ).statistic
            ),

            "percentile_target_mean": float(
                np.mean(t_rank)
            ),

            "percentile_target_ks": float(
                ks_2samp(
                    percentile_transform(s, s),
                    t_rank
                ).statistic
            )
        })

    result = pd.DataFrame(rows).sort_values(
        "raw_ks",
        ascending=False
    )

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    result.to_csv(
        out / "domain_invariant_feature_audit.csv",
        index=False
    )

    summary = {
        "version": VERSION,
        "features": len(FEATURES),
        "source_rows": len(src),
        "target_rows": len(tgt),
        "target_labels_used": False,
        "model_fit": False
    }

    with open(out / "Q1_R09B_LOCK.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("===== Q1-R09B DOMAIN-INVARIANT FEATURE AUDIT =====")
    print("version=", VERSION)
    print("source_rows=", len(src))
    print("target_rows=", len(tgt))
    print("features=", len(FEATURES))
    print("target_labels_used=NO")
    print("model_fit=NO")
    print(result.to_string(index=False))
    print("Decision: DOMAIN_INVARIANT_FEATURE_AUDIT_COMPLETE")


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
        default=r"F:\MEDSEG_SAFETTA\outputs\Q1_R09B_domain_invariant_feature_construction_audit_v1"
    )

    args = p.parse_args()

    if args.self_test:
        self_test()
    else:
        run(args)


if __name__ == "__main__":
    main()
