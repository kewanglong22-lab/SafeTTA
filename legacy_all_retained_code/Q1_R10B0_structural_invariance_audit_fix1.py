
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R10B0_structural_invariance_audit_fix1.py

Fix:
    Explicit feature grouping audit.

Purpose:
    Compare domain separability of:
        1. uncertainty-only
        2. structure-only
        3. uncertainty + structure

Additional prints:
    - feature counts
    - feature names
    - source/external rows
    - shared columns

No target labels are used.
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
        "AUROC": roc_auc_score(y, prob),
        "AUPRC": average_precision_score(y, prob)
    }


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--external", required=True)
    parser.add_argument("--output_dir", required=True)

    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    source = pd.read_csv(args.source)
    external = pd.read_csv(args.external)

    source["domain"] = 0
    external["domain"] = 1

    shared = sorted(
        list(
            set(source.columns)
            &
            set(external.columns)
        )
    )

    panel = pd.concat(
        [
            source[shared],
            external[shared]
        ],
        ignore_index=True
    )

    y = panel["domain"].values

    feature_cols = [
        c for c in panel.columns
        if c not in ["sample_id", "domain"]
    ]

    uncertainty = [
        c for c in feature_cols
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

    structure = [
        c for c in feature_cols
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

    groups = {
        "uncertainty_only": uncertainty,
        "structure_only": structure,
        "uncertainty_plus_structure":
            uncertainty + structure
    }

    print("\n===== FEATURE AUDIT =====")
    print("Source rows:", len(source))
    print("External rows:", len(external))
    print("Shared columns:", len(shared))

    rows = []

    for name, cols in groups.items():

        print("\nGROUP:", name)
        print("Feature count:", len(cols))
        print(cols)

        if len(cols) == 0:
            continue

        result = evaluate(
            panel[cols],
            y
        )

        result["feature_group"] = name
        result["features"] = len(cols)

        rows.append(result)

    result_df = pd.DataFrame(rows)

    print("\n===== RESULT =====")
    print(result_df.to_string(index=False))

    result_df.to_csv(
        out / "structural_invariance_results_fix1.csv",
        index=False
    )

    print("\nOutput:", out)
    print("PASS")


if __name__ == "__main__":
    main()
