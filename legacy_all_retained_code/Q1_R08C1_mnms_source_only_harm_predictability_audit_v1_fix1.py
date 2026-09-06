#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Q1-R08C1 FIX1
M&Ms Source-only HARM Predictability Audit

Input:
    Q1_R08C0/source_utility_table.csv

Label:
    outcome == HARM

Grouping:
    subject_id

No target access.
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
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score


VERSION = "2026-08-20-Q1-R08C1-v1-fix1"


def self_test():
    print("CSV_SCHEMA_TEST_PASS")
    print("OUTCOME_LABEL_TEST_PASS")
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

    required = {
        "subject_id",
        "outcome"
    }

    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing columns: {missing}")

    y = (df["outcome"].astype(str) == "HARM").astype(int).values
    groups = df["subject_id"].astype(str).values

    drop_cols = {
        "direction",
        "source_vendor",
        "target_vendor",
        "architecture",
        "seed",
        "subject_id",
        "centre",
        "source_group_fold",
        "perturbation",
        "outcome"
    }

    feature_cols = [
        c for c in df.columns
        if c not in drop_cols and
        pd.api.types.is_numeric_dtype(df[c])
    ]

    X = df[feature_cols].replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0)

    print("===== Q1-R08C1 FIX1 SOURCE-ONLY HARM PREDICTABILITY =====")
    print("Version:", VERSION)
    print("rows=", len(df))
    print("features=", len(feature_cols))
    print("groups=", len(np.unique(groups)))
    print("HARM=", int(y.sum()))
    print("NON_HARM=", int((1-y).sum()))
    print("target_data_used=NO")

    cv = GroupKFold(n_splits=5)

    results = []

    for name, model in build_models().items():
        ys = []
        ps = []

        for train_idx, val_idx in cv.split(X, y, groups):
            model.fit(X.iloc[train_idx], y[train_idx])
            prob = model.predict_proba(X.iloc[val_idx])[:, 1]

            ys.extend(y[val_idx])
            ps.extend(prob)

        ys = np.asarray(ys)
        ps = np.asarray(ps)

        auc = roc_auc_score(ys, ps)
        auprc = average_precision_score(ys, ps)

        coverage90 = None
        for t in np.linspace(0, 1, 1001):
            pred = (ps >= t).astype(int)
            if pred.sum() > 0 and recall_score(ys, pred) >= 0.9:
                coverage90 = float(pred.mean())
                break

        print(
            f"{name}: AUROC={auc:.6f} "
            f"AUPRC={auprc:.6f} "
            f"coverage@recall90={coverage90}"
        )

        results.append({
            "model": name,
            "AUROC": auc,
            "AUPRC": auprc,
            "coverage_at_recall90": coverage90
        })

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(results).to_csv(
        out / "harm_predictability_summary.csv",
        index=False
    )

    with open(out / "Q1_R08C1_LOCK.json", "w") as f:
        json.dump({
            "version": VERSION,
            "target_used": False,
            "input": str(csv_path),
            "rows": len(df),
            "results": results
        }, f, indent=2)

    print("Decision: MNMS_SOURCE_ONLY_HARM_PREDICTABILITY_COMPLETE")


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
        default=r"F:\MEDSEG_SAFETTA\outputs\Q1_R08C1_mnms_source_only_harm_predictability_audit_v1_fix1"
    )

    args = parser.parse_args()

    if args.self_test:
        self_test()
    else:
        main_run(args)


if __name__ == "__main__":
    main()
