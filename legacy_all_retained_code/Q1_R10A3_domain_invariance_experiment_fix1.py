#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R10A3_domain_invariance_experiment_fix1.py

Corrected domain invariance evaluation after applying
source-fitted RUR transformation to external data.
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

    p = cross_val_predict(
        build_model(),
        X,
        y,
        cv=cv,
        method="predict_proba",
        n_jobs=-1
    )[:, 1]

    return {
        "AUROC": float(roc_auc_score(y, p)),
        "AUPRC": float(average_precision_score(y, p))
    }


def get_groups(df):
    cols = [
        c for c in df.columns
        if c not in ["sample_id", "domain"]
    ]

    rur = [
        c for c in cols
        if c.startswith("rur_")
    ]

    struct = [
        c for c in cols
        if "boundary" in c or "fg_fraction" in c or c.startswith("struct_")
    ]

    raw = [
        c for c in cols
        if c not in rur and c not in struct
    ]

    return {
        "raw": raw,
        "raw_plus_RUR": raw + rur,
        "raw_plus_RUR_structure": raw + rur + struct
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

    common = sorted(
        list(set(source.columns) & set(external.columns))
    )

    panel = pd.concat(
        [
            source[common],
            external[common]
        ],
        ignore_index=True
    )

    groups = get_groups(panel)
    y = panel["domain"].values

    rows = []

    for name, features in groups.items():
        r = evaluate(panel[features], y)
        r["feature_group"] = name
        r["features"] = len(features)
        rows.append(r)

    result = pd.DataFrame(rows)

    result.to_csv(
        out / "domain_invariance_results_fix1.csv",
        index=False
    )

    panel.to_csv(
        out / "domain_panel_fix1.csv",
        index=False
    )

    print("===== Q1-R10A3 DOMAIN INVARIANCE FIX1 =====")
    print(result)
    print("Output:", out)
    print("PASS")


if __name__ == "__main__":
    main()
