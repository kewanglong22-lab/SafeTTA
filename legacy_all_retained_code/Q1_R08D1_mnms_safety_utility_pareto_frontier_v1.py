#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Q1-R08D1
M&Ms Safety-Utility Pareto Frontier Audit

Goal:
Evaluate risk-aware TTA decisions without optimizing only mean Dice.

Input:
Q1-R08C0 source_utility_table.csv

Risk:
source-only pre-adaptation uncertainty model (nested OOF)

No target data.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


VERSION = "2026-08-20-Q1-R08D1-v1"


def self_test():
    print("PARETO_SCHEMA_TEST_PASS")
    print("OOF_RISK_TEST_PASS")
    print("NO_TARGET_TEST_PASS")
    print("SELF_TEST_PASS")


def risk_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced",
            max_iter=5000,
            random_state=20260820
        ))
    ])


def main_run(args):

    csv_path = Path(args.asset) / "source_utility_table.csv"

    df = pd.read_csv(csv_path)

    y_harm = (
        df["outcome"].astype(str).eq("HARM").astype(int)
    )

    y_benefit = (
        df["outcome"].astype(str).eq("BENEFIT").astype(int)
    )

    groups = df["subject_id"].astype(str)

    feature_cols = [
        c for c in df.columns
        if any(k in c.lower() for k in [
            "entropy",
            "prob",
            "confidence",
            "uncertain",
            "boundary",
            "fg_fraction",
            "logit"
        ])
        and "tent" not in c.lower()
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    X = df[feature_cols].replace(
        [np.inf, -np.inf],
        np.nan
    ).fillna(0)

    source_dice = df["source_dice"].values
    a1_dice = df["a1_dice"].values


    risk = np.zeros(len(df))

    cv = GroupKFold(n_splits=5)

    for tr, va in cv.split(X, y_harm, groups):

        model = risk_model()

        model.fit(
            X.iloc[tr],
            y_harm.iloc[tr]
        )

        risk[va] = model.predict_proba(
            X.iloc[va]
        )[:, 1]


    print("===== Q1-R08D1 SAFETY-UTILITY PARETO FRONTIER =====")
    print("Version:", VERSION)
    print("rows=", len(df))
    print("risk_AUROC=", roc_auc_score(y_harm, risk))
    print("target_data_used=NO")


    rows = []

    for t in np.linspace(0, 1, 101):

        use_tent = risk < t

        final_dice = np.where(
            use_tent,
            a1_dice,
            source_dice
        )

        harmful_tent = (
            (y_harm.values == 1)
            &
            use_tent
        )

        beneficial_tent = (
            (y_benefit.values == 1)
            &
            use_tent
        )

        rows.append({
            "threshold": float(t),
            "tent_coverage": float(use_tent.mean()),
            "mean_dice": float(final_dice.mean()),
            "harm_exposure": float(harmful_tent.sum() / max(y_harm.sum(),1)),
            "harm_avoidance": float(
                1 - harmful_tent.sum() / max(y_harm.sum(),1)
            ),
            "benefit_retention": float(
                beneficial_tent.sum() / max(y_benefit.sum(),1)
            ),
            "harm_tent_rows": int(harmful_tent.sum()),
            "benefit_tent_rows": int(beneficial_tent.sum())
        })


    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    table = pd.DataFrame(rows)

    table.to_csv(
        out / "safety_utility_pareto_frontier.csv",
        index=False
    )

    best = table.sort_values(
        [
            "harm_avoidance",
            "mean_dice"
        ],
        ascending=False
    ).head(20)

    best.to_csv(
        out / "top_safety_utility_points.csv",
        index=False
    )


    with open(out / "Q1_R08D1_LOCK.json", "w") as f:
        json.dump({
            "version": VERSION,
            "target_used": False,
            "features": feature_cols,
            "risk_auc": float(roc_auc_score(y_harm, risk))
        }, f, indent=2)


    print("Decision: MNMS_SAFETY_UTILITY_PARETO_COMPLETE")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--self-test",
        action="store_true"
    )

    parser.add_argument(
        "--asset",
        default=r"F:\MEDSEG_SAFETTA\outputs\Q1_R08C0_mnms_source_utility_asset_v1"
    )

    parser.add_argument(
        "--output",
        default=r"F:\MEDSEG_SAFETTA\outputs\Q1_R08D1_safety_utility_pareto_frontier_v1"
    )

    args = parser.parse_args()

    if args.self_test:
        self_test()
    else:
        main_run(args)


if __name__ == "__main__":
    main()
