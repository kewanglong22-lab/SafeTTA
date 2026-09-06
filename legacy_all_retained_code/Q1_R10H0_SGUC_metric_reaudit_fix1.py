
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R10H0_SGUC_metric_reaudit_fix1.py

Critical metric re-audit for the frozen SGUC source-domain experiment.

Fixes:
1) FPR@90 is measured only at an operating point with Recall >= 0.90.
2) PPV@1% is prevalence-adjusted from sensitivity and FPR.

Protocol:
- source domain only
- 5-fold stratified OOF
- seeds = 20260816..20260820
- compare uncertainty_only, concat, SGUC_fixed
- frozen SGUC interaction: [U, S, U * tanh(mean(S))]
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

SEEDS = [20260816, 20260817, 20260818, 20260819, 20260820]
TARGET_RECALL = 0.90
TARGET_PREVALENCE = 0.01


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def build_model(seed):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=seed
        ))
    ])


def threshold_at_min_recall(y, p, target_recall=0.90):
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)

    pos_scores = np.sort(p[y == 1])[::-1]
    if len(pos_scores) == 0:
        raise ValueError("No positive samples.")

    k = int(np.ceil(target_recall * len(pos_scores)))
    k = min(max(k, 1), len(pos_scores))
    threshold = float(pos_scores[k - 1])

    pred = (p >= threshold).astype(int)

    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    tn = int(np.sum((pred == 0) & (y == 0)))

    recall = tp / (tp + fn) if (tp + fn) else np.nan
    fpr = fp / (fp + tn) if (fp + tn) else np.nan
    precision_emp = tp / (tp + fp) if (tp + fp) else np.nan

    pi = TARGET_PREVALENCE
    denom = recall * pi + fpr * (1.0 - pi)
    ppv_1pct = (recall * pi / denom) if denom > 0 else np.nan

    return {
        "threshold": threshold,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "Recall_at_op": float(recall),
        "FPR_at_recall90": float(fpr),
        "Empirical_Precision": float(precision_emp),
        "PPV_at_1pct_prevalence": float(ppv_1pct),
    }


def evaluate_oof(X, y, seed):
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=seed
    )
    p = cross_val_predict(
        build_model(seed),
        X,
        y,
        cv=cv,
        method="predict_proba",
        n_jobs=-1
    )[:, 1]

    op = threshold_at_min_recall(y, p, TARGET_RECALL)
    result = {
        "AUROC": float(roc_auc_score(y, p)),
        "AUPRC": float(average_precision_score(y, p)),
        **op,
    }
    return result, p


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
        if any(k in c.lower() for k in [
            "entropy", "prob", "confidence", "logit", "rur"
        ])
    ]
    s_cols = [
        c for c in source.columns
        if any(k in c.lower() for k in [
            "struct", "boundary", "fg_fraction"
        ])
    ]

    U = source[u_cols].fillna(0).values
    S = source[s_cols].fillna(0).values

    X_methods = {
        "uncertainty_only": U,
        "concat": np.concatenate([U, S], axis=1),
        "SGUC_fixed": np.concatenate([
            U,
            S,
            U * np.tanh(S.mean(axis=1, keepdims=True))
        ], axis=1),
    }

    print("===== R10H0 SGUC CRITICAL METRIC RE-AUDIT FIX1 =====")
    print("Samples:", len(source))
    print("Harm:", int(y.sum()))
    print("Non-harm:", int((y == 0).sum()))
    print("Target recall:", TARGET_RECALL)
    print("Target prevalence for PPV:", TARGET_PREVALENCE)
    print("Uncertainty features:", len(u_cols))
    print("Structure features:", len(s_cols))
    print("IMPORTANT: previous prototype FPR@90 values were invalid because recall was ~0.24.")
    print("This script recomputes the operating point at Recall >= 0.90.\n")

    rows = []
    pred_rows = []

    for method, X in X_methods.items():
        print(f"===== METHOD: {method} =====")
        for seed in SEEDS:
            set_seed(seed)
            r, p = evaluate_oof(X, y, seed)
            r["method"] = method
            r["seed"] = seed
            r["features"] = X.shape[1]
            rows.append(r)

            for sid, yy, pp in zip(source["sample_id"].astype(str), y, p):
                pred_rows.append({
                    "sample_id": sid,
                    "method": method,
                    "seed": seed,
                    "true_harm": int(yy),
                    "oof_probability": float(pp),
                })

            print(
                f"seed={seed} "
                f"AUROC={r['AUROC']:.6f} "
                f"AUPRC={r['AUPRC']:.6f} "
                f"Recall={r['Recall_at_op']:.6f} "
                f"FPR@R>=0.90={r['FPR_at_recall90']:.6f} "
                f"PPV@1%={r['PPV_at_1pct_prevalence']:.6f} "
                f"thr={r['threshold']:.6f}"
            )
        print()

    result = pd.DataFrame(rows)
    summary_rows = []

    for method, g in result.groupby("method", sort=False):
        summary_rows.append({
            "method": method,
            "seeds": len(g),
            "AUROC_mean": g["AUROC"].mean(),
            "AUROC_std": g["AUROC"].std(ddof=1),
            "AUPRC_mean": g["AUPRC"].mean(),
            "AUPRC_std": g["AUPRC"].std(ddof=1),
            "Recall_mean": g["Recall_at_op"].mean(),
            "Recall_min": g["Recall_at_op"].min(),
            "FPR90_mean": g["FPR_at_recall90"].mean(),
            "FPR90_std": g["FPR_at_recall90"].std(ddof=1),
            "PPV1pct_mean": g["PPV_at_1pct_prevalence"].mean(),
            "PPV1pct_std": g["PPV_at_1pct_prevalence"].std(ddof=1),
        })

    summary = pd.DataFrame(summary_rows)

    print("===== CORRECTED SUMMARY =====")
    print(summary.to_string(index=False))

    if (result["Recall_at_op"] + 1e-12 < TARGET_RECALL).any():
        raise AssertionError("At least one operating point has recall < 0.90.")

    result.to_csv(out / "R10H0_metric_reaudit_per_seed.csv", index=False)
    summary.to_csv(out / "R10H0_metric_reaudit_summary.csv", index=False)
    pd.DataFrame(pred_rows).to_csv(
        out / "R10H0_metric_reaudit_oof_predictions.csv", index=False
    )

    print("\nOutput:", out)
    print("PASS")


if __name__ == "__main__":
    main()
