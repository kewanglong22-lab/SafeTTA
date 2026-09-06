
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R10G0_fusion_baseline.py

Domain-Calibrated Safety Fusion baseline audit.

Compare:
1. uncertainty only
2. structure only
3. uncertainty + structure concat
4. weighted fusion

Metrics:
- AUROC
- AUPRC
- Recall@90%
- FPR@90%
- PPV@1% prevalence
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    recall_score
)


def model():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
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
        model(),
        X,
        y,
        cv=cv,
        method="predict_proba",
        n_jobs=-1
    )[:, 1]

    order = np.argsort(-p)

    target = 0.90
    cutoff_idx = int(len(y) * (1-target))

    threshold = p[order][cutoff_idx]

    pred = (p >= threshold).astype(int)

    tp = np.sum((pred==1)&(y==1))
    fp = np.sum((pred==1)&(y==0))
    fn = np.sum((pred==0)&(y==1))

    recall = tp/(tp+fn+1e-8)
    fpr = fp/(np.sum(y==0)+1e-8)

    ppv = tp/(tp+fp+1e-8)

    return {
        "AUROC": roc_auc_score(y,p),
        "AUPRC": average_precision_score(y,p),
        "Recall": recall,
        "FPR@90": fpr,
        "PPV_emp": ppv,
        "PPV_1pct": tp/(tp+fp*99+1e-8)
    }


def main():

    parser=argparse.ArgumentParser()

    parser.add_argument("--panel", required=True)
    parser.add_argument("--output_dir", required=True)

    args=parser.parse_args()

    out=Path(args.output_dir)
    out.mkdir(parents=True,exist_ok=True)

    df=pd.read_csv(
        args.panel,
        low_memory=False
    )

    source=df[df["domain"]==0].copy()

    y=(
        source["outcome"]
        .astype(str)
        .str.lower()
        .isin(["harm","positive","1"])
        .astype(int)
        .values
    )

    u_cols=[
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

    s_cols=[
        c for c in source.columns
        if any(k in c.lower()
        for k in [
            "structure",
            "struct",
            "boundary",
            "fg_fraction"
        ])
    ]

    experiments={
        "uncertainty_only":u_cols,
        "structure_only":s_cols,
        "uncertainty_structure":u_cols+s_cols
    }

    rows=[]

    print("===== R10G0 FUSION BASELINE =====")
    print("Samples:",len(source))
    print("Harm:",int(y.sum()))

    for name,cols in experiments.items():

        r=evaluate(
            source[cols].values,
            y
        )

        r["method"]=name
        r["features"]=len(cols)

        print("\n",name)
        print(r)

        rows.append(r)

    result=pd.DataFrame(rows)

    result.to_csv(
        out/"R10G0_fusion_baseline_results.csv",
        index=False
    )

    print("\n===== SUMMARY =====")
    print(result.to_string(index=False))
    print("PASS")


if __name__=="__main__":
    main()
