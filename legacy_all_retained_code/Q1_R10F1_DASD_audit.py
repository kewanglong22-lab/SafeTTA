
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R10F1_DASD_audit.py

Audit R10F0 DASD latent representations.

Evaluates:
    z_domain
    z_safe_u
    z_safe_s
    z_safe_fusion

Metrics:
    Domain AUROC/AUPRC
    Safety AUROC/AUPRC
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, average_precision_score


class DASDNet(torch.nn.Module):

    def __init__(self, n_u, n_s):
        super().__init__()

        self.domain_encoder = torch.nn.Sequential(
            torch.nn.Linear(n_u, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 16),
            torch.nn.ReLU()
        )

        self.safe_unc_encoder = torch.nn.Sequential(
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


def clf():

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
        clf(),
        X,
        y,
        cv=cv,
        method="predict_proba",
        n_jobs=-1
    )[:, 1]

    return {
        "AUROC": roc_auc_score(y, p),
        "AUPRC": average_precision_score(y, p)
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--panel", required=True)
    parser.add_argument("--output_dir", required=True)

    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(
        args.checkpoint,
        map_location="cpu"
    )

    df = pd.read_csv(
        args.panel,
        low_memory=False
    )

    u_cols = ckpt["u_features"]
    s_cols = ckpt["s_features"]

    missing = [
        c for c in u_cols + s_cols
        if c not in df.columns
    ]

    print("===== INPUT CHECK =====")
    print("Missing:", missing)

    if missing:
        raise ValueError("Feature lineage mismatch")

    model = DASDNet(
        len(u_cols),
        len(s_cols)
    )

    state = ckpt["model"]

    model.domain_encoder.load_state_dict({
        k.replace("domain_encoder.", ""): v
        for k, v in state.items()
        if k.startswith("domain_encoder.")
    })

    model.safe_unc_encoder.load_state_dict({
        k.replace("safe_unc_encoder.", ""): v
        for k, v in state.items()
        if k.startswith("safe_unc_encoder.")
    })

    model.structure_encoder.load_state_dict({
        k.replace("structure_encoder.", ""): v
        for k, v in state.items()
        if k.startswith("structure_encoder.")
    })

    model.eval()

    with torch.no_grad():

        u = torch.tensor(
            df[u_cols].values,
            dtype=torch.float32
        )

        s = torch.tensor(
            df[s_cols].values,
            dtype=torch.float32
        )

        z_domain = model.domain_encoder(u).numpy()
        z_safe_u = model.safe_unc_encoder(u).numpy()
        z_safe_s = model.structure_encoder(s).numpy()

    domain = df["domain"].values

    source_mask = domain == 0

    safety = (
        df.loc[source_mask, "outcome"]
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
        "z_safe_fusion": np.concatenate(
            [z_safe_u, z_safe_s],
            axis=1
        )
    }

    rows = []

    print("===== R10F1 DASD AUDIT =====")
    print("Samples:", len(df))

    for name, feat in reps.items():

        d = evaluate(
            feat,
            domain
        )

        s = evaluate(
            feat[source_mask],
            safety
        )

        print("\nRepresentation:", name)
        print("Dim:", feat.shape[1])
        print(
            "Domain AUROC:",
            round(d["AUROC"], 6)
        )
        print(
            "Safety AUROC:",
            round(s["AUROC"], 6)
        )

        rows.append({
            "representation": name,
            "dim": feat.shape[1],
            "domain_AUROC": d["AUROC"],
            "domain_AUPRC": d["AUPRC"],
            "safety_AUROC": s["AUROC"],
            "safety_AUPRC": s["AUPRC"]
        })

    result = pd.DataFrame(rows)

    print("\n===== SUMMARY =====")
    print(result.to_string(index=False))

    result.to_csv(
        out / "R10F1_DASD_audit_results.csv",
        index=False
    )

    print("Output:", out)
    print("PASS")


if __name__ == "__main__":
    main()
