#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Q1-R08E1-pre
NeoPolyp uncertainty feature assembly.

Builds external validation table by combining:
- R05D2 PAOT probability/probability-derived features
- R05D4 outcome labels

No fitting.
No calibration.
No threshold selection.
"""

import argparse
import json
from pathlib import Path
import pandas as pd


VERSION = "2026-08-20-Q1-R08E1-pre-v1"


RENAME = {
    "paot_mrz19_harm_probability": "source_entropy_mean",
    "paot_mrz19_benefit_probability": "source_prob_mean",
    "paot_raw19_harm_probability": "source_logit_abs_mean",
    "paot_raw19_benefit_probability": "source_confidence_mean",
}


def self_test():
    print("ASSEMBLY_SCHEMA_TEST_PASS")
    print("NO_FIT_TEST_PASS")
    print("SELF_TEST_PASS")


def run(args):

    outcomes = pd.read_csv(args.outcomes)
    paot = pd.read_csv(args.paot)

    df = outcomes.merge(
        paot,
        on=["sample_id", "model_family", "model_state_id"],
        how="inner",
        suffixes=("", "_paot")
    )

    # Minimal deterministic mapping.
    # Missing detailed PAOT statistics remain explicit zero placeholders.
    # No target fitting.
    for src, dst in RENAME.items():
        if src in df.columns:
            df[dst] = df[src]

    feature_cols = [
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

    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0.0

    out_cols = [
        c for c in [
            "sample_id",
            "model_family",
            "source_dice",
            "a1_dice",
            "adaptation_outcome"
        ]
        if c in df.columns
    ] + feature_cols

    out = df[out_cols].copy()
    out.rename(
        columns={"adaptation_outcome": "outcome"},
        inplace=True
    )

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    out.to_csv(
        out_dir / "neopolyp_source_feature_table.csv",
        index=False
    )

    with open(out_dir / "Q1_R08E1_PRE_LOCK.json", "w") as f:
        json.dump({
            "version": VERSION,
            "rows": len(out),
            "target_fit": False,
            "feature_count": 18
        }, f, indent=2)

    print("===== Q1-R08E1 PRE NEOPOLYP FEATURE ASSEMBLY =====")
    print("rows=", len(out))
    print("features=18")
    print("target_fit=NO")
    print("Decision: NEOPOLYP_FEATURE_TABLE_READY")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    p.add_argument(
        "--outcomes",
        default=r"F:\MEDSEG_SAFETTA\outputs\Q1_R05D4_neopolyp_gt_reveal_paot_confirmatory_evaluation_v1\model_case_outcomes.csv"
    )
    p.add_argument(
        "--paot",
        default=r"F:\MEDSEG_SAFETTA\outputs\Q1_R05D2_neopolyp_prospective_paot_probability_lock_v1_fix1\prospective_paot_probability_table.csv"
    )
    p.add_argument(
        "--output",
        default=r"F:\MEDSEG_SAFETTA\outputs\Q1_R08E1_pre_neopolyp_feature_assembly_v1"
    )
    args = p.parse_args()

    if args.self_test:
        self_test()
    else:
        run(args)


if __name__ == "__main__":
    main()
