#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R10D0_DISR_baseline.py

Baseline evaluation for Domain-Invariant Safety Representation.

Compare:
- uncertainty_only
- structure_only
- uncertainty_plus_structure

Source labels only.
5-fold OOF.
"""

import argparse
from pathlib import Path
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, average_precision_score


def model():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=20260820
        ))
    ])


def evaluate(X, y):
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=20260820
    )

    p = cross_val_predict(
        model(),
        X,
        y,
        cv=cv,
        method="predict_proba",
        n_jobs=-1
    )[:, 1]

    return {
        "AUROC": roc_auc_score(y, p),
        "AUPRC": average_precision_score(y, p)
    }


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.source)

    y = (
        df["outcome"]
        .astype(str)
        .str.lower()
        .isin(["harm", "positive", "1"])
        .astype(int)
    )

    cols = [
        c for c in df.columns
        if c not in ["sample_id", "outcome", "domain"]
    ]

    uncertainty = [
        c for c in cols
        if any(k in c.lower() for k in
               ["entropy", "prob", "confidence", "logit", "rur"])
    ]

    structure = [
        c for c in cols
        if any(k in c.lower() for k in
               ["boundary", "fg_fraction", "component",
                "topology", "shape", "struct"])
    ]

    groups = {
        "uncertainty_only": uncertainty,
        "structure_only": structure,
        "uncertainty_plus_structure": uncertainty + structure
    }

    print("===== DISR BASELINE FEATURE AUDIT =====")
    print("Samples:", len(df))
    print("Harm:", int(y.sum()))
    print("Non-harm:", int((1-y).sum()))

    results = []

    for name, f in groups.items():
        print("\nGROUP:", name)
        print("Feature count:", len(f))
        print(f)

        r = evaluate(df[f], y)
        r["feature_group"] = name
        r["features"] = len(f)
        results.append(r)

    result = pd.DataFrame(results)

    print("\n===== RESULT =====")
    print(result.to_string(index=False))

    result.to_csv(
        out / "DISR_baseline_results.csv",
        index=False
    )

    print("Output:", out)
    print("PASS")


if __name__ == "__main__":
    main()
