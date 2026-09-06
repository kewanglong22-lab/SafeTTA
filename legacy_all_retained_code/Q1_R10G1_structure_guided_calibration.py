
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R10G1_structure_guided_calibration.py

Structure-guided uncertainty calibration baseline.

Compare:
1. uncertainty only
2. uncertainty + structure concat
3. structure gate calibration

Idea:
    uncertainty provides sensitivity
    structure provides reliability calibration
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score


def clf():
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(
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
        clf(),
        X,
        y,
        cv=cv,
        method="predict_proba",
        n_jobs=-1
    )[:, 1]

    idx = np.argsort(-p)
    cut = int(len(y) * 0.1)
    threshold = p[idx][cut]

    pred = (p >= threshold).astype(int)

    tp = ((pred == 1) & (y == 1)).sum()
    fp = ((pred == 1) & (y == 0)).sum()
    fn = ((pred == 0) & (y == 1)).sum()

    return {
        "AUROC": roc_auc_score(y, p),
        "AUPRC": average_precision_score(y, p),
        "Recall": tp/(tp+fn+1e-8),
        "FPR@90": fp/(np.sum(y==0)+1e-8),
        "PPV_1pct": tp/(tp+99*fp+1e-8)
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--panel", required=True)
    parser.add_argument("--output_dir", required=True)

    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.panel, low_memory=False)

    source = df[df["domain"] == 0].copy()

    y = (
        source["outcome"]
        .astype(str)
        .str.lower()
        .isin(["harm", "positive", "1"])
        .astype(int)
        .values
    )

    u_cols = [
        c for c in source.columns
        if any(k in c.lower()
               for k in [
                   "entropy",
                   "prob",
                   "confidence",
                   "logit",
                   "rur"
               ])
    ]

    s_cols = [
        c for c in source.columns
        if any(k in c.lower()
               for k in [
                   "struct",
                   "boundary",
                   "fg_fraction"
               ])
    ]

    u = source[u_cols].values
    s = source[s_cols].values

    experiments = {
        "uncertainty_only": u,
        "concat": np.concatenate([u, s], axis=1),
        "structure_gate_input": np.concatenate(
            [
                u,
                s,
                u * np.tanh(s.mean(axis=1, keepdims=True))
            ],
            axis=1
        )
    }

    print("===== R10G1 STRUCTURE GUIDED CALIBRATION =====")
    print("Samples:", len(source))
    print("Uncertainty:", len(u_cols))
    print("Structure:", len(s_cols))

    rows = []

    for name, x in experiments.items():

        r = evaluate(x, y)
        r["method"] = name
        r["features"] = x.shape[1]

        print("\n", name)
        print(r)

        rows.append(r)

    result = pd.DataFrame(rows)

    result.to_csv(
        out / "R10G1_structure_guided_results.csv",
        index=False
    )

    print("\n===== SUMMARY =====")
    print(result.to_string(index=False))
    print("PASS")


if __name__ == "__main__":
    main()
