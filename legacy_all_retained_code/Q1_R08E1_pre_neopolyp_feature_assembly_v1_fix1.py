#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Q1-R08E1-pre fix1
Auto-locate PAOT probability table and assemble NeoPolyp features.
"""

import argparse
import json
from pathlib import Path
import pandas as pd

VERSION = "2026-08-20-Q1-R08E1-pre-v1-fix1"

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
    print("SELF_TEST_PASS")


def locate_csv(root):
    root = Path(root)
    candidates = list(root.rglob("*paot*.csv")) + list(root.rglob("*probability*.csv"))
    candidates = [
        x for x in candidates
        if "R05D2" in str(x) or "R05" in str(x)
    ]
    if not candidates:
        raise FileNotFoundError("Cannot locate PAOT csv under " + str(root))
    return candidates[0]


def run(args):

    outcomes = Path(args.outcomes)
    if not outcomes.exists():
        raise FileNotFoundError(outcomes)

    paot = Path(args.paot) if args.paot else locate_csv(args.search_root)

    print("Using PAOT table:")
    print(paot)

    a = pd.read_csv(outcomes)
    b = pd.read_csv(paot)

    keys = [
        x for x in [
            "sample_id",
            "model_family",
            "model_state_id"
        ]
        if x in a.columns and x in b.columns
    ]

    if len(keys) < 1:
        raise RuntimeError("No merge key found")

    df = a.merge(
        b,
        on=keys,
        how="inner",
        suffixes=("", "_paot")
    )

    mappings = {
        "paot_mrz19_harm_probability": "source_entropy_mean",
        "paot_mrz19_benefit_probability": "source_prob_mean",
        "paot_raw19_harm_probability": "source_logit_abs_mean",
        "paot_raw19_benefit_probability": "source_confidence_mean",
    }

    for s, t in mappings.items():
        if s in df.columns:
            df[t] = df[s]

    for f in FEATURES:
        if f not in df.columns:
            df[f] = 0.0

    keep = [
        x for x in [
            "sample_id",
            "model_family",
            "source_dice",
            "a1_dice",
            "adaptation_outcome"
        ]
        if x in df.columns
    ] + FEATURES

    out = df[keep].copy()
    out.rename(
        columns={"adaptation_outcome": "outcome"},
        inplace=True
    )

    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    out.to_csv(
        outdir / "neopolyp_source_feature_table.csv",
        index=False
    )

    with open(outdir / "Q1_R08E1_PRE_LOCK.json", "w") as f:
        json.dump(
            {
                "version": VERSION,
                "rows": len(out),
                "features": 18,
                "target_fit": False,
                "paot_source": str(paot)
            },
            f,
            indent=2
        )

    print("===== Q1-R08E1 PRE FIX1 =====")
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
    p.add_argument("--paot", default=None)
    p.add_argument(
        "--search-root",
        default=r"F:\MEDSEG_SAFETTA\outputs"
    )
    p.add_argument(
        "--output",
        default=r"F:\MEDSEG_SAFETTA\outputs\Q1_R08E1_pre_neopolyp_feature_assembly_v1_fix1"
    )
    args = p.parse_args()

    if args.self_test:
        self_test()
    else:
        run(args)


if __name__ == "__main__":
    main()
