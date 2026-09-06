
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R10D1_DISR_prototype.py

Domain-Invariant Safety Representation prototype.

Architecture:
    uncertainty features
            |
      uncertainty encoder
            |
            z_u
            |
            +---- domain adversarial head

    structure features
            |
      structure encoder
            |
            z_s

    concat(z_u, z_s)
            |
        safety head

Loss:
    safety classification loss
    + adversarial domain loss

Protocol:
    Source labels for safety.
    Source/external domain labels only for adversarial training.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score


class GradReverse(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lambd * grad, None


def grad_reverse(x, lambd):
    return GradReverse.apply(x, lambd)


class DISRNet(nn.Module):

    def __init__(self, n_u, n_s):

        super().__init__()

        self.u_encoder = nn.Sequential(
            nn.Linear(n_u, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        self.s_encoder = nn.Sequential(
            nn.Linear(n_s, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )

        self.safety_head = nn.Sequential(
            nn.Linear(48, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

        self.domain_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )


    def forward(self, u, s, grl=0.0):

        zu = self.u_encoder(u)
        zs = self.s_encoder(s)

        safety = self.safety_head(
            torch.cat([zu, zs], dim=1)
        )

        domain = self.domain_head(
            grad_reverse(zu, grl)
        )

        return safety, domain


class FeatureDataset(Dataset):

    def __init__(self, u, s, y, d):

        self.u = torch.tensor(
            u,
            dtype=torch.float32
        )

        self.s = torch.tensor(
            s,
            dtype=torch.float32
        )

        self.y = torch.tensor(
            y,
            dtype=torch.float32
        )

        self.d = torch.tensor(
            d,
            dtype=torch.float32
        )


    def __len__(self):
        return len(self.y)


    def __getitem__(self, idx):

        return (
            self.u[idx],
            self.s[idx],
            self.y[idx],
            self.d[idx]
        )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--source", required=True)
    parser.add_argument("--external", required=True)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lambda_domain", type=float, default=0.1)

    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)


    source = pd.read_csv(args.source)
    external = pd.read_csv(args.external)

    source["domain"] = 0
    external["domain"] = 1

    # external has no safety label
    # source safety training is used only

    features = [
        c for c in source.columns
        if c not in [
            "sample_id",
            "outcome",
            "domain"
        ]
    ]

    u_cols = [
        c for c in features
        if any(
            k in c.lower()
            for k in [
                "entropy",
                "prob",
                "confidence",
                "logit",
                "rur"
            ]
        )
    ]

    s_cols = [
        c for c in features
        if any(
            k in c.lower()
            for k in [
                "boundary",
                "fg_fraction",
                "component",
                "topology",
                "shape",
                "struct"
            ]
        )
    ]

    print("===== DISR FEATURE AUDIT =====")
    print("Uncertainty features:", len(u_cols))
    print(u_cols)
    print("Structure features:", len(s_cols))
    print(s_cols)


    panel = pd.concat(
        [
            source,
            external
        ],
        ignore_index=True
    )

    # align columns
    panel[u_cols + s_cols] = panel[u_cols + s_cols].fillna(
        panel[u_cols + s_cols].median()
    )

    X_u = panel[u_cols].values
    X_s = panel[s_cols].values

    y = (
        panel["outcome"]
        .astype(str)
        .str.lower()
        .isin(["harm", "positive", "1"])
        .fillna(False)
        .astype(float)
        .values
    )

    d = panel["domain"].values.astype(float)


    source_mask = panel["domain"].values == 0

    dataset = FeatureDataset(
        X_u,
        X_s,
        y,
        d
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True
    )

    model = DISRNet(
        len(u_cols),
        len(s_cols)
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3
    )

    bce = nn.BCEWithLogitsLoss()

    for epoch in range(args.epochs):

        model.train()

        losses = []

        for u, s, yy, dd in loader:

            pred_y, pred_d = model(
                u,
                s,
                args.lambda_domain
            )

            # safety loss only valid on source
            mask = (
                dd == 0
            )

            if mask.sum() > 0:
                loss_y = bce(
                    pred_y[mask].squeeze(),
                    yy[mask]
                )
            else:
                loss_y = 0.0

            loss_d = bce(
                pred_d.squeeze(),
                dd
            )

            loss = loss_y + loss_d

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(
                float(loss)
            )

        print(
            f"Epoch {epoch+1}/{args.epochs} "
            f"loss={np.mean(losses):.5f}"
        )

    print("PASS")


if __name__ == "__main__":
    main()
