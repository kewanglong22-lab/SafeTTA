
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R10A3_domain_invariance_experiment.py

Purpose:
    Test whether Relative Uncertainty Representation (RUR)
    reduces domain-dependent uncertainty encoding.

Input:
    1. Source invariant feature table from Q1_R10A0
    2. Locked NeoPolyp external feature table from Q1_R10A2-7

Protocol:
    Domain label:
        source = 0
        external = 1

Compare:
    A. Raw uncertainty features
    B. Raw + RUR
    C. Raw + RUR + Structure

Metrics:
    Domain AUROC
    Domain AUPRC

No external labels are used.
"""

import argparse
from pathlib import Path

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, average_precision_score


ID_COLS = [
    "sample_id",
    "domain",
]


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

    model = build_model()

    prob = cross_val_predict(
        model,
        X,
        y,
        cv=cv,
        method="predict_proba",
        n_jobs=-1
    )[:, 1]

    return {
        "AUROC": float(roc_auc_score(y, prob)),
        "AUPRC": float(average_precision_score(y, prob)),
    }


def prepare_features(df):

    all_cols = [
        c for c in df.columns
        if c not in ID_COLS
    ]

    rur = [
        c for c in all_cols
        if c.startswith("rur_")
    ]

    struct = [
        c for c in all_cols
        if c.startswith("struct_")
    ]

    raw = [
        c for c in all_cols
        if c not in rur
        and c not in struct
    ]

    return {
        "raw": raw,
        "raw_plus_RUR": raw + rur,
        "raw_plus_RUR_structure": raw + rur + struct,
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

    # align shared representation columns
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
            external[common],
        ],
        ignore_index=True
    )

    groups = prepare_features(panel)

    y = panel["domain"].values

    results = []

    for name, cols in groups.items():

        result = evaluate(
            panel[cols],
            y
        )

        result["feature_group"] = name
        result["features"] = len(cols)

        results.append(result)

    result_df = pd.DataFrame(results)

    result_df.to_csv(
        out / "domain_invariance_results.csv",
        index=False
    )

    panel.to_csv(
        out / "domain_panel.csv",
        index=False
    )

    print("===== Q1-R10A3 DOMAIN INVARIANCE =====")
    print(result_df)
    print("Output:", out)
    print("PASS")


if __name__ == "__main__":
    main()
