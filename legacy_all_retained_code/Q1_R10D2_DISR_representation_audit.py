
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R10D2_DISR_representation_audit.py

Purpose:
    Audit whether DISR latent representations reduce
    domain leakage while preserving safety information.

Inputs:
    disr_latent_features.csv

Evaluates:
    z_u
    z_s
    z_u + z_s

Tasks:
    1. Domain classification
    2. Source safety classification

Metrics:
    AUROC
    AUPRC

No retraining of DISR.
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


def build_model():

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

    prob = cross_val_predict(
        build_model(),
        X,
        y,
        cv=cv,
        method="predict_proba",
        n_jobs=-1
    )[:, 1]

    return {
        "AUROC": float(roc_auc_score(y, prob)),
        "AUPRC": float(average_precision_score(y, prob))
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--latent",
        required=True
    )

    parser.add_argument(
        "--output_dir",
        required=True
    )

    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(
        parents=True,
        exist_ok=True
    )

    df = pd.read_csv(args.latent)

    print("===== DISR REPRESENTATION AUDIT =====")
    print("Samples:", len(df))
    print("Columns:", len(df.columns))

    domain = df["domain"].values

    source = (
        df["domain"] == 0
    )

    safety = (
        df.loc[source, "outcome"]
        .astype(str)
        .str.lower()
        .isin(
            [
                "harm",
                "positive",
                "1"
            ]
        )
        .astype(int)
        .values
    )

    source_df = df[source].copy()

    z_u = [
        c for c in df.columns
        if c.startswith("z_u_")
    ]

    z_s = [
        c for c in df.columns
        if c.startswith("z_s_")
    ]

    groups = {
        "z_u": z_u,
        "z_s": z_s,
        "z_u_plus_z_s": z_u + z_s
    }

    print("\n===== LATENT AUDIT =====")

    results = []

    for name, cols in groups.items():

        print("\nRepresentation:", name)
        print("Dim:", len(cols))

        # domain
        domain_result = evaluate(
            df[cols],
            domain
        )

        # safety only source
        safety_result = evaluate(
            source_df[cols],
            safety
        )

        print(
            "Domain AUROC:",
            round(domain_result["AUROC"], 6),
            "Domain AUPRC:",
            round(domain_result["AUPRC"], 6)
        )

        print(
            "Safety AUROC:",
            round(safety_result["AUROC"], 6),
            "Safety AUPRC:",
            round(safety_result["AUPRC"], 6)
        )

        results.append(
            {
                "representation": name,
                "dim": len(cols),
                "domain_AUROC": domain_result["AUROC"],
                "domain_AUPRC": domain_result["AUPRC"],
                "safety_AUROC": safety_result["AUROC"],
                "safety_AUPRC": safety_result["AUPRC"]
            }
        )

    result_df = pd.DataFrame(results)

    print("\n===== SUMMARY =====")
    print(result_df.to_string(index=False))

    result_df.to_csv(
        out / "DISR_representation_audit_results.csv",
        index=False
    )

    print("\nOutput:", out)
    print("PASS")


if __name__ == "__main__":
    main()
