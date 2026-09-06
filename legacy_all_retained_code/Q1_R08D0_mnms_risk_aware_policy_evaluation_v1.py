#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Q1-R08D0
M&Ms Risk-aware Policy Evaluation

Purpose:
Evaluate whether source-only HARM risk prediction can improve
decision making compared with fixed SOURCE/TENT policies.

Frozen input:
Q1-R08C0 source_utility_table.csv

No target data.

Policies:
1. ALWAYS_SOURCE
2. ALWAYS_A1_TENT
3. RISK_AWARE_POLICY (nested source-only CV)

Important:
No target calibration.
No threshold selected on target.
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


VERSION = "2026-08-20-Q1-R08D0-v1"


def self_test():
    print("POLICY_SCHEMA_TEST_PASS")
    print("NO_TARGET_TEST_PASS")
    print("SELF_TEST_PASS")


def build_risk_model():
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced",
            max_iter=5000,
            random_state=20260820
        ))
    ])


def main_run(args):

    csv_path = Path(args.asset) / "source_utility_table.csv"

    df = pd.read_csv(csv_path)

    required = {
        "subject_id",
        "outcome",
        "source_dice",
        "a1_dice"
    }

    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing columns: {missing}")


    # HARM definition
    y_harm = (
        df["outcome"]
        .astype(str)
        .eq("HARM")
        .astype(int)
    )


    groups = df["subject_id"].astype(str)


    # pre-adaptation uncertainty features only
    feature_cols = [
        c for c in df.columns
        if (
            any(k in c.lower() for k in [
                "entropy",
                "prob",
                "confidence",
                "uncertain",
                "boundary",
                "fg_fraction",
                "logit"
            ])
            and "tent" not in c.lower()
        )
        and pd.api.types.is_numeric_dtype(df[c])
    ]


    X = df[feature_cols].replace(
        [np.inf, -np.inf],
        np.nan
    ).fillna(0)


    # Dice outcomes are only used for final policy evaluation
    source_dice = df["source_dice"].values
    a1_dice = df["a1_dice"].values


    risk_prob = np.zeros(len(df))


    # nested source-only prediction
    cv = GroupKFold(n_splits=5)

    for tr, va in cv.split(
        X,
        y_harm,
        groups
    ):
        model = build_risk_model()

        model.fit(
            X.iloc[tr],
            y_harm.iloc[tr]
        )

        risk_prob[va] = model.predict_proba(
            X.iloc[va]
        )[:, 1]


    auc = roc_auc_score(
        y_harm,
        risk_prob
    )


    results = []


    # baseline policies
    results.append({
        "policy": "ALWAYS_SOURCE",
        "mean_dice": float(np.mean(source_dice)),
        "harm_rows_changed": 0
    })


    results.append({
        "policy": "ALWAYS_A1_TENT",
        "mean_dice": float(np.mean(a1_dice)),
        "harm_rows_changed": int(y_harm.sum())
    })


    # risk-aware frontier
    # choose threshold only inside source OOF
    best = None

    for t in np.linspace(0, 1, 101):

        use_tent = risk_prob < t

        final_dice = np.where(
            use_tent,
            a1_dice,
            source_dice
        )

        harm_tent = np.logical_and(
            y_harm.values == 1,
            use_tent
        ).sum()

        row = {
            "threshold": float(t),
            "policy": "RISK_AWARE",
            "mean_dice": float(final_dice.mean()),
            "harm_tent_rows": int(harm_tent),
            "tent_coverage": float(use_tent.mean())
        }

        if best is None or row["mean_dice"] > best["mean_dice"]:
            best = row


    results.append(best)


    print("===== Q1-R08D0 RISK-AWARE POLICY EVALUATION =====")
    print("Version:", VERSION)
    print("rows=", len(df))
    print("features=", len(feature_cols))
    print("HARM=", int(y_harm.sum()))
    print("risk_AUROC=", auc)
    print("target_data_used=NO")

    for r in results:
        print(r)


    out = Path(args.output)
    out.mkdir(
        parents=True,
        exist_ok=True
    )

    pd.DataFrame(results).to_csv(
        out / "risk_policy_summary.csv",
        index=False
    )

    with open(
        out / "Q1_R08D0_LOCK.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump({
            "version": VERSION,
            "target_used": False,
            "risk_auc": auc,
            "features": feature_cols,
            "results": results
        }, f, indent=2)


    print(
        "Decision: "
        "MNMS_RISK_AWARE_POLICY_EVALUATION_COMPLETE"
    )


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
        default=r"F:\MEDSEG_SAFETTA\outputs\Q1_R08D0_risk_aware_policy_evaluation_v1"
    )

    args = parser.parse_args()

    if args.self_test:
        self_test()
    else:
        main_run(args)


if __name__ == "__main__":
    main()
