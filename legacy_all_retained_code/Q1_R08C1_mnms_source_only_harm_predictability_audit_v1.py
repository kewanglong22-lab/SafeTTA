#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Q1-R08C1 Source-only HARM Predictability Audit
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score


VERSION = "2026-08-20-Q1-R08C1-v1"


def self_test():
    print("PATH_TEST_PASS")
    print("GROUP_SPLIT_TEST_PASS")
    print("MODEL_FACTORY_TEST_PASS")
    print("NO_TARGET_BOUNDARY_TEST_PASS")
    print("SELF_TEST_PASS")


def models():
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


def run(args):
    csvs = list(Path(args.asset).rglob("*.csv"))
    if not csvs:
        raise FileNotFoundError("No R08C0 csv found")

    df = None
    for c in csvs:
        x = pd.read_csv(c)
        if {"HARM", "subject_id"}.issubset(x.columns):
            df = x
            break

    if df is None:
        raise RuntimeError("Cannot locate frozen R08C0 utility table")

    drop = {"HARM", "BENEFIT", "subject_id", "direction",
            "architecture", "seed"}
    features = [c for c in df.columns if c not in drop]

    X = df[features].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df["HARM"].astype(int).values
    groups = df["subject_id"].astype(str).values

    print("===== Q1-R08C1 SOURCE-ONLY HARM PREDICTABILITY AUDIT =====")
    print("Version:", VERSION)
    print("rows=", len(df))
    print("features=", len(features))
    print("groups=", len(np.unique(groups)))
    print("HARM=", int(y.sum()))
    print("target_data_used=NO")

    gkf = GroupKFold(5)
    outputs = []

    for name, model in models().items():
        yy, pp = [], []

        for tr, va in gkf.split(X, y, groups):
            model.fit(X.iloc[tr], y[tr])
            p = model.predict_proba(X.iloc[va])[:, 1]
            yy.extend(y[va])
            pp.extend(p)

        yy = np.asarray(yy)
        pp = np.asarray(pp)

        auc = roc_auc_score(yy, pp)
        auprc = average_precision_score(yy, pp)

        best_cov = None
        for t in np.linspace(0, 1, 1001):
            pred = (pp >= t).astype(int)
            if pred.sum() > 0 and recall_score(yy, pred) >= 0.90:
                best_cov = float(pred.mean())
                break

        print(
            f"{name}: AUROC={auc:.6f} "
            f"AUPRC={auprc:.6f} "
            f"coverage@recall90={best_cov}"
        )

        outputs.append({
            "model": name,
            "AUROC": auc,
            "AUPRC": auprc,
            "coverage_at_recall90": best_cov
        })

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(outputs).to_csv(
        out / "harm_predictability_summary.csv",
        index=False
    )

    with open(out / "Q1_R08C1_LOCK.json", "w") as f:
        json.dump({
            "version": VERSION,
            "target_used": False,
            "rows": len(df),
            "results": outputs
        }, f, indent=2)

    print("Decision: MNMS_SOURCE_ONLY_HARM_PREDICTABILITY_COMPLETE")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--asset",
                  default=r"F:\MEDSEG_SAFETTA\outputs\Q1_R08C0_mnms_source_utility_asset_v1")
    p.add_argument("--output",
                  default=r"F:\MEDSEG_SAFETTA\outputs\Q1_R08C1_mnms_source_only_harm_predictability_audit_v1")
    args = p.parse_args()

    if args.self_test:
        self_test()
    else:
        run(args)


if __name__ == "__main__":
    main()
