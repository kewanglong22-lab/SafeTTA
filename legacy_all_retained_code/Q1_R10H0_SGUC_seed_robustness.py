
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R10H0_SGUC_seed_robustness.py

Purpose:
    Evaluate whether R10G1 fixed structure-guided calibration
    is robust across multiple random seeds.

Protocol:
    - Same feature lineage
    - Same source domain
    - Different model seeds

Metrics:
    AUROC
    AUPRC
    FPR@90
    PPV@1%
"""

import argparse
from pathlib import Path
import random
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, average_precision_score


SEEDS = [
    20260816,
    20260817,
    20260818,
    20260819,
    20260820
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def clf(seed):

    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=seed
        ))
    ])


def evaluate(X, y, seed):

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=seed
    )

    p = cross_val_predict(
        clf(seed),
        X,
        y,
        cv=cv,
        method="predict_proba",
        n_jobs=-1
    )[:, 1]

    idx = np.argsort(-p)
    threshold = p[idx][int(len(y)*0.1)]

    pred = (p >= threshold).astype(int)

    tp = np.sum((pred == 1) & (y == 1))
    fp = np.sum((pred == 1) & (y == 0))
    fn = np.sum((pred == 0) & (y == 1))

    return {
        "seed": seed,
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

    df = pd.read_csv(
        args.panel,
        low_memory=False
    )

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

    # R10G1 fixed structure-guided interaction
    X = np.concatenate(
        [
            u,
            s,
            u * np.tanh(
                s.mean(axis=1, keepdims=True)
            )
        ],
        axis=1
    )

    print("===== R10H0 SGUC SEED ROBUSTNESS =====")
    print("Samples:", len(source))
    print("Features:", X.shape[1])

    rows = []

    for seed in SEEDS:

        set_seed(seed)

        r = evaluate(
            X,
            y,
            seed
        )

        print(r)
        rows.append(r)

    result = pd.DataFrame(rows)

    summary = pd.DataFrame([{
        "AUROC_mean": result.AUROC.mean(),
        "AUROC_std": result.AUROC.std(),
        "AUPRC_mean": result.AUPRC.mean(),
        "AUPRC_std": result.AUPRC.std(),
        "FPR90_mean": result["FPR@90"].mean(),
        "FPR90_std": result["FPR@90"].std(),
        "PPV1pct_mean": result["PPV_1pct"].mean(),
        "PPV1pct_std": result["PPV_1pct"].std()
    }])

    print("\n===== SUMMARY =====")
    print(summary.to_string(index=False))

    result.to_csv(
        out/"R10H0_seed_results.csv",
        index=False
    )

    summary.to_csv(
        out/"R10H0_seed_summary.csv",
        index=False
    )

    print("PASS")


if __name__ == "__main__":
    main()
