
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R10E1_disentanglement_audit.py

Audit explicit domain-safety disentanglement.

Input:
    R10E0 checkpoint

Extract:
    z_domain
    z_safe_u
    z_safe_s
    z_safe_fusion

Evaluate:
    Domain classification
    Source safety classification

Metrics:
    AUROC
    AUPRC
"""

import argparse
from pathlib import Path

import pandas as pd
import torch

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, average_precision_score


class DisentangleNet(torch.nn.Module):

    def __init__(self, n_u, n_s):

        super().__init__()

        self.domain_encoder = torch.nn.Sequential(
            torch.nn.Linear(n_u, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 16),
            torch.nn.ReLU()
        )

        self.safety_encoder = torch.nn.Sequential(
            torch.nn.Linear(n_u, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU()
        )

        self.structure_encoder = torch.nn.Sequential(
            torch.nn.Linear(n_s, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU()
        )


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


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        required=True
    )

    parser.add_argument(
        "--feature_csv",
        required=True
    )

    parser.add_argument(
        "--output_dir",
        required=True
    )

    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(
        args.checkpoint,
        map_location="cpu"
    )

    df = pd.read_csv(
        args.feature_csv,
        low_memory=False
    )

    u_cols = ckpt["u_features"]
    s_cols = ckpt["s_features"]

    missing_u = [c for c in u_cols if c not in df.columns]
    missing_s = [c for c in s_cols if c not in df.columns]

    print("===== INPUT FEATURE CHECK =====")
    print("Missing uncertainty features:", missing_u)
    print("Missing structure features:", missing_s)

    for c in missing_u + missing_s:
        df[c] = 0.0

    model = DisentangleNet(
        len(u_cols),
        len(s_cols)
    )

    state = ckpt["model"]

    model.domain_encoder.load_state_dict({
        k.replace("domain_encoder.", ""): v
        for k, v in state.items()
        if k.startswith("domain_encoder.")
    })

    model.safety_encoder.load_state_dict({
        k.replace("safety_encoder.", ""): v
        for k, v in state.items()
        if k.startswith("safety_encoder.")
    })

    model.structure_encoder.load_state_dict({
        k.replace("structure_encoder.", ""): v
        for k, v in state.items()
        if k.startswith("structure_encoder.")
    })

    model.eval()

    with torch.no_grad():

        u = torch.tensor(
            df[u_cols].fillna(0).values,
            dtype=torch.float32
        )

        s = torch.tensor(
            df[s_cols].fillna(0).values,
            dtype=torch.float32
        )

        z_domain = model.domain_encoder(u).numpy()
        z_safe_u = model.safety_encoder(u).numpy()
        z_safe_s = model.structure_encoder(s).numpy()

    domain = df["domain"].values

    source = df["domain"] == 0

    safety = (
        df.loc[source, "outcome"]
        .astype(str)
        .str.lower()
        .isin(["harm", "positive", "1"])
        .astype(int)
        .values
    )

    reps = {
        "z_domain": z_domain,
        "z_safe_u": z_safe_u,
        "z_safe_s": z_safe_s,
        "z_safe_fusion": __import__("numpy").concatenate(
            [
                z_safe_u,
                z_safe_s
            ],
            axis=1
        )
    }

    rows = []

    print("===== R10E1 DISENTANGLEMENT AUDIT =====")
    print("Samples:", len(df))

    for name, feat in reps.items():

        domain_result = evaluate(
            feat,
            domain
        )

        safety_result = evaluate(
            feat[source],
            safety
        )

        print("\nRepresentation:", name)
        print("Dim:", feat.shape[1])
        print(
            "Domain AUROC:",
            round(domain_result["AUROC"], 6)
        )
        print(
            "Safety AUROC:",
            round(safety_result["AUROC"], 6)
        )

        rows.append({
            "representation": name,
            "dim": feat.shape[1],
            "domain_AUROC": domain_result["AUROC"],
            "domain_AUPRC": domain_result["AUPRC"],
            "safety_AUROC": safety_result["AUROC"],
            "safety_AUPRC": safety_result["AUPRC"]
        })

    result = pd.DataFrame(rows)

    print("\n===== SUMMARY =====")
    print(result.to_string(index=False))

    result.to_csv(
        out / "R10E1_disentanglement_audit_results.csv",
        index=False
    )

    print("Output:", out)
    print("PASS")


if __name__ == "__main__":
    main()
