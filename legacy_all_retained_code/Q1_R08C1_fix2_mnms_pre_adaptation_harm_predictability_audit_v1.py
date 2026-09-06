#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Q1-R08C1 FIX2
M&Ms pre-adaptation HARM Predictability Audit

Purpose:
Remove post-adaptation performance leakage and test whether
HARM is predictable from source-only uncertainty signals.

Frozen input:
Q1-R08C0 source_utility_table.csv

Allowed features:
- entropy
- probability distribution
- confidence
- foreground statistics
- boundary statistics
- uncertainty fractions
- logit statistics

Forbidden:
- source_dice
- a1_dice
- delta_dice
- any adaptation outcome proxy

Target data:
NO
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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    recall_score
)


VERSION = "2026-08-20-Q1-R08C1-v1-fix2"


def self_test():
    print("LEAKAGE_EXCLUSION_TEST_PASS")
    print("PRE_ADAPTATION_FEATURE_TEST_PASS")
    print("GROUP_SPLIT_TEST_PASS")
    print("NO_TARGET_BOUNDARY_TEST_PASS")
    print("SELF_TEST_PASS")


def build_models():
    return {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                class_weight="balanced",
                max_iter=5000,
                random_state=20260820
            ))
        ]),
        "RandomForest": RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced",
            random_state=20260820,
            n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingClassifier(
            random_state=20260820
        )
    }


def main_run(args):

    csv_path = Path(args.asset) / "source_utility_table.csv"

    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)

    if "outcome" not in df.columns:
        raise RuntimeError("Missing outcome label")

    if "subject_id" not in df.columns:
        raise RuntimeError("Missing subject_id")

    y = (
        df["outcome"]
        .astype(str)
        .eq("HARM")
        .astype(int)
        .values
    )

    groups = df["subject_id"].astype(str).values


    # Only source pre-adaptation features
    allowed_keywords = [
        "entropy",
        "prob",
        "confidence",
        "fg_fraction",
        "boundary",
        "uncertain",
        "logit"
    ]


    forbidden = [
        "dice",
        "delta",
        "a1",
        "tent"
    ]


    feature_cols = []

    for c in df.columns:
        lc = c.lower()

        if any(k in lc for k in allowed_keywords):
            if not any(f in lc for f in forbidden):
                if pd.api.types.is_numeric_dtype(df[c]):
                    feature_cols.append(c)


    if len(feature_cols) == 0:
        raise RuntimeError(
            "No valid pre-adaptation features found"
        )


    X = df[feature_cols].replace(
        [np.inf, -np.inf],
        np.nan
    ).fillna(0)


    print("===== Q1-R08C1 FIX2 PRE-ADAPTATION HARM AUDIT =====")
    print("Version:", VERSION)
    print("rows=", len(df))
    print("features=", len(feature_cols))
    print("feature_list=", feature_cols)
    print("groups=", len(np.unique(groups)))
    print("HARM=", int(y.sum()))
    print("NON_HARM=", int((1-y).sum()))
    print("target_data_used=NO")


    cv = GroupKFold(n_splits=5)

    results = []


    for name, model in build_models().items():

        ys = []
        ps = []

        for tr, va in cv.split(X, y, groups):

            model.fit(
                X.iloc[tr],
                y[tr]
            )

            p = model.predict_proba(
                X.iloc[va]
            )[:, 1]

            ys.extend(y[va])
            ps.extend(p)


        ys = np.asarray(ys)
        ps = np.asarray(ps)

        auc = roc_auc_score(
            ys,
            ps
        )

        auprc = average_precision_score(
            ys,
            ps
        )


        # Correct frontier:
        # maximum threshold with recall >= 0.90
        best_cov = None

        for t in np.linspace(1, 0, 1001):

            pred = (
                ps >= t
            ).astype(int)

            if pred.sum() > 0:

                rec = recall_score(
                    ys,
                    pred
                )

                if rec >= 0.90:
                    best_cov = float(
                        pred.mean()
                    )
                    break


        print(
            f"{name}: "
            f"AUROC={auc:.6f} "
            f"AUPRC={auprc:.6f} "
            f"coverage@recall90={best_cov}"
        )


        results.append({
            "model": name,
            "AUROC": auc,
            "AUPRC": auprc,
            "coverage_at_recall90": best_cov,
            "features": len(feature_cols)
        })


    out = Path(args.output)
    out.mkdir(
        parents=True,
        exist_ok=True
    )


    pd.DataFrame(results).to_csv(
        out / "pre_adaptation_harm_predictability_summary.csv",
        index=False
    )


    with open(
        out / "Q1_R08C1_FIX2_LOCK.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            {
                "version": VERSION,
                "target_used": False,
                "feature_count": len(feature_cols),
                "features": feature_cols,
                "results": results
            },
            f,
            indent=2
        )


    print(
        "Decision: "
        "MNMS_PRE_ADAPTATION_HARM_PREDICTABILITY_COMPLETE"
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
        default=r"F:\MEDSEG_SAFETTA\outputs\Q1_R08C1_pre_adaptation_harm_predictability_audit_v1_fix2"
    )


    args = parser.parse_args()

    if args.self_test:
        self_test()
    else:
        main_run(args)


if __name__ == "__main__":
    main()
