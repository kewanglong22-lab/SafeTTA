
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R10B0_structural_invariance_audit.py

Purpose:
    Test whether segmentation structural features are more
    domain-invariant than uncertainty features.

Protocol:
    Domain:
        source = 0
        external = 1

Compare:
    1. uncertainty_only
    2. structure_only
    3. uncertainty_plus_structure

No target labels are used.

This is an audit before designing the final
domain-invariant safety representation.
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
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=20260820
                )
            )
        ]
    )


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
        "AUROC": float(
            roc_auc_score(y, prob)
        ),
        "AUPRC": float(
            average_precision_score(y, prob)
        )
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        required=True
    )

    parser.add_argument(
        "--external",
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

    source = pd.read_csv(args.source)
    external = pd.read_csv(args.external)

    source["domain"] = 0
    external["domain"] = 1

    common = sorted(
        list(
            set(source.columns)
            &
            set(external.columns)
        )
    )

    panel = pd.concat(
        [
            source[common],
            external[common]
        ],
        ignore_index=True
    )

    y = panel["domain"].values

    all_features = [
        c for c in panel.columns
        if c not in [
            "sample_id",
            "domain"
        ]
    ]

    uncertainty = [
        c for c in all_features
        if any(
            k in c
            for k in [
                "entropy",
                "prob",
                "confidence",
                "logit"
            ]
        )
    ]

    structure = [
        c for c in all_features
        if any(
            k in c
            for k in [
                "boundary",
                "fg_fraction",
                "component",
                "topology",
                "shape"
            ]
        )
    ]

    groups = {
        "uncertainty_only": uncertainty,
        "structure_only": structure,
        "uncertainty_plus_structure":
            uncertainty + structure
    }

    rows = []

    for name, cols in groups.items():

        if len(cols) == 0:
            continue

        result = evaluate(
            panel[cols],
            y
        )

        result["feature_group"] = name
        result["features"] = len(cols)

        rows.append(result)

    result = pd.DataFrame(rows)

    result.to_csv(
        out / "structural_invariance_results.csv",
        index=False
    )

    panel.to_csv(
        out / "structural_invariance_domain_panel.csv",
        index=False
    )

    print("===== Q1-R10B0 STRUCTURAL INVARIANCE AUDIT =====")
    print(result)
    print("Output:", out)
    print("PASS")


if __name__ == "__main__":
    main()
