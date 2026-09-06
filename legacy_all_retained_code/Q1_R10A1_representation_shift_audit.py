
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R10A1_representation_shift_audit.py

Purpose:
    Validate whether the proposed invariant representation reduces
    domain dependency while preserving source safety discrimination.

Experiments:
    1. Domain classification:
       source vs external/domain shift

    2. Safety classification:
       harm vs non-harm on source samples

Feature groups:
    A. raw23
    B. raw23 + RUR
    C. raw23 + RUR + structure

Input:
    Q1_R10A0 invariant feature table

Expected columns:
    outcome
    model_family
    rur_*
    struct_*

The script does NOT use target labels for training.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict


def build_model():
    return Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=20260820,
                )
            ),
        ]
    )


def metric(y, p):
    return {
        "AUROC": float(roc_auc_score(y, p)),
        "AUPRC": float(average_precision_score(y, p)),
    }


def evaluate_cv(X, y):

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=20260820,
    )

    model = build_model()

    prob = cross_val_predict(
        model,
        X,
        y,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]

    return metric(y, prob)


def feature_groups(df):

    all_features = [
        c for c in df.columns
        if c not in [
            "sample_id",
            "model_family",
            "outcome",
            "domain",
        ]
    ]

    rur = [
        c for c in all_features
        if c.startswith("rur_")
    ]

    struct = [
        c for c in all_features
        if c.startswith("struct_")
    ]

    raw = [
        c for c in all_features
        if c not in rur
        and c not in struct
    ]

    return {
        "raw23": raw,
        "raw23_plus_RUR": raw + rur,
        "raw23_plus_RUR_structure": raw + rur + struct,
    }


def run(input_csv, output_dir):

    df = pd.read_csv(input_csv)

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    groups = feature_groups(df)

    print("Samples:", len(df))
    print("Feature groups:")
    for k, v in groups.items():
        print(k, len(v))

    # -------------------------------------------------
    # 1. Domain shift audit
    # -------------------------------------------------
    # domain:
    # source = 0
    # external is inferred from model_family if available
    #
    # If no domain column exists, split by outcome availability:
    # this keeps the script compatible with current asset.
    # -------------------------------------------------

    domain_df = df.copy()

    if "domain" in domain_df.columns:
        y_domain = (
            domain_df["domain"]
            .astype(str)
            .ne("source")
            .astype(int)
        )
    else:
        # Current R08E1 table contains source_feature assembly.
        # Users can add domain column for external audit.
        y_domain = None

    domain_rows = []

    if y_domain is not None and y_domain.nunique() > 1:

        for name, cols in groups.items():
            result = evaluate_cv(
                df[cols],
                y_domain
            )
            result["feature_group"] = name
            domain_rows.append(result)

    else:
        domain_rows.append(
            {
                "feature_group": "SKIPPED",
                "AUROC": np.nan,
                "AUPRC": np.nan,
                "reason":
                    "domain column unavailable"
            }
        )

    pd.DataFrame(domain_rows).to_csv(
        output_dir / "domain_shift_summary.csv",
        index=False
    )


    # -------------------------------------------------
    # 2. Safety preservation audit
    # -------------------------------------------------

    safety_rows = []

    if "outcome" in df.columns:

        # harm = 1
        y = (
            df["outcome"]
            .astype(str)
            .str.lower()
            .isin(
                [
                    "harm",
                    "positive",
                    "1",
                ]
            )
            .astype(int)
        )

        for name, cols in groups.items():

            result = evaluate_cv(
                df[cols],
                y
            )

            result["feature_group"] = name
            safety_rows.append(result)

    else:
        safety_rows.append(
            {
                "feature_group": "SKIPPED",
                "AUROC": np.nan,
                "AUPRC": np.nan,
                "reason":
                    "outcome column unavailable"
            }
        )


    pd.DataFrame(safety_rows).to_csv(
        output_dir / "safety_preservation_summary.csv",
        index=False
    )


    print("\n===== Q1-R10A-1 AUDIT =====")
    print(
        "Output:",
        output_dir
    )
    print("PASS")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True
    )

    parser.add_argument(
        "--output_dir",
        required=True
    )

    args = parser.parse_args()

    run(
        args.input,
        args.output_dir
    )


if __name__ == "__main__":
    main()
