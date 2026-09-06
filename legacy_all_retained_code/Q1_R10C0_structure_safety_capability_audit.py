
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R10C0_structure_safety_capability_audit.py

Purpose:
    Test whether domain-invariant structural features contain
    useful safety information.

Question:
    Although structure features are domain-invariant, can they
    discriminate harmful segmentation cases?

Protocol:
    Train only on source labels.
    Compare:
        1. structure_only
        2. uncertainty_only
        3. structure + uncertainty

Metrics:
    AUROC
    AUPRC

No external labels are used.
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
            max_iter=2000,
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
        "--source",
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

    df = pd.read_csv(args.source)

    if "outcome" not in df.columns:
        raise ValueError(
            "source table requires outcome labels"
        )

    y = (
        df["outcome"]
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
    )

    features = [
        c for c in df.columns
        if c not in [
            "sample_id",
            "outcome",
            "domain"
        ]
    ]

    structure = [
        c for c in features
        if any(
            k in c.lower()
            for k in [
                "boundary",
                "fg_fraction",
                "component",
                "topology",
                "shape"
            ]
        )
    ]

    uncertainty = [
        c for c in features
        if any(
            k in c.lower()
            for k in [
                "entropy",
                "prob",
                "confidence",
                "logit"
            ]
        )
    ]

    groups = {
        "structure_only": structure,
        "uncertainty_only": uncertainty,
        "structure_plus_uncertainty":
            structure + uncertainty
    }

    print("===== FEATURE AUDIT =====")
    print("Samples:", len(df))
    print("Harm:", int(y.sum()))
    print("Non-harm:", int((1-y).sum()))

    rows = []

    for name, cols in groups.items():

        print("\nGROUP:", name)
        print("Features:", len(cols))
        print(cols)

        if len(cols) == 0:
            continue

        result = evaluate(
            df[cols],
            y
        )

        result["feature_group"] = name
        result["features"] = len(cols)

        rows.append(result)

    result_df = pd.DataFrame(rows)

    print("\n===== RESULT =====")
    print(result_df.to_string(index=False))

    result_df.to_csv(
        out / "structure_safety_capability_results.csv",
        index=False
    )

    print("\nOutput:", out)
    print("PASS")


if __name__ == "__main__":
    main()
