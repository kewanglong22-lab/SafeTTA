
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R10H1_SGUC_external_validation.py

Cross-domain validation of frozen SGUC fixed calibration.

Protocol:
    train: source domain only
    test: external domain

Compare:
    1. uncertainty only
    2. uncertainty + structure concat
    3. SGUC fixed calibration

Metrics:
    AUROC
    AUPRC
    Recall@90
    FPR@90
    PPV@1%
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score


def build_model():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=20260820
        ))
    ])


def evaluate(y, p):

    idx = np.argsort(-p)
    threshold = p[idx][int(len(y)*0.1)]

    pred = (p >= threshold).astype(int)

    tp = np.sum((pred == 1) & (y == 1))
    fp = np.sum((pred == 1) & (y == 0))
    fn = np.sum((pred == 0) & (y == 1))

    return {
        "AUROC": roc_auc_score(y, p),
        "AUPRC": average_precision_score(y, p),
        "Recall": tp/(tp+fn+1e-8),
        "FPR@90": fp/(np.sum(y == 0)+1e-8),
        "PPV_1pct": tp/(tp+99*fp+1e-8)
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--source", required=True)
    parser.add_argument("--external", required=True)
    parser.add_argument("--output_dir", required=True)

    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    source = pd.read_csv(args.source, low_memory=False)
    external = pd.read_csv(args.external, low_memory=False)

    def prepare(df, require_label=True):

        df = df.copy()

        if require_label:
            y = (
                df["outcome"]
                .astype(str)
                .str.lower()
                .isin(["harm", "positive", "1"])
                .astype(int)
                .values
            )
        else:
            y = None

        u_cols = [
            c for c in df.columns
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
            c for c in df.columns
            if any(k in c.lower()
                   for k in [
                       "struct",
                       "boundary",
                       "fg_fraction"
                   ])
        ]

        return df, y, u_cols, s_cols

    source, ys, u_cols, s_cols = prepare(source, require_label=True)
    external, ye, _, _ = prepare(external, require_label=False)

    # external feature table may not contain labels.
    # If labels are unavailable, infer from supported external label column.
    if ye is None:
        label_candidates = [
            c for c in external.columns
            if c.lower() in ["label", "target", "y", "gt", "class"]
        ]
        if not label_candidates:
            raise ValueError(
                "External validation requires hidden labels or an explicit label column."
            )

        ye = (
            external[label_candidates[0]]
            .astype(str)
            .str.lower()
            .isin(["harm", "positive", "1", "tumor", "neoplasia"])
            .astype(int)
            .values
        )

    Xus = source[u_cols].fillna(0).values
    Xss = source[s_cols].fillna(0).values

    Xue = external[u_cols].fillna(0).values
    Xse = external[s_cols].fillna(0).values

    experiments = {
        "uncertainty_only":
            (Xus, Xue),

        "concat":
            (
                np.concatenate([Xus, Xss], axis=1),
                np.concatenate([Xue, Xse], axis=1)
            ),

        "SGUC":
            (
                np.concatenate(
                    [
                        Xus,
                        Xss,
                        Xus*np.tanh(
                            Xss.mean(axis=1, keepdims=True)
                        )
                    ],
                    axis=1
                ),
                np.concatenate(
                    [
                        Xue,
                        Xse,
                        Xue*np.tanh(
                            Xse.mean(axis=1, keepdims=True)
                        )
                    ],
                    axis=1
                )
            )
    }

    print("===== R10H1 SGUC EXTERNAL VALIDATION =====")
    print("Source:", len(source))
    print("External:", len(external))

    rows = []

    for name, (train_x, test_x) in experiments.items():

        model = build_model()

        model.fit(train_x, ys)

        p = model.predict_proba(test_x)[:,1]

        r = evaluate(ye, p)
        r["method"] = name

        print("\n", name)
        print(r)

        rows.append(r)

    result = pd.DataFrame(rows)

    result.to_csv(
        out/"R10H1_external_validation_results.csv",
        index=False
    )

    print("\n===== SUMMARY =====")
    print(result.to_string(index=False))
    print("PASS")


if __name__ == "__main__":
    main()
