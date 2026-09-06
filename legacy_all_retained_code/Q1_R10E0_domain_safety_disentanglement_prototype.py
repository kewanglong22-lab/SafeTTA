
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R10E0_domain_safety_disentanglement_prototype.py

Prototype:
    Explicitly disentangle uncertainty representation into:

        uncertainty feature
              |
        ----------------
        |              |
   domain branch   safety branch
        |              |
    domain loss    safety loss

Together with:
    structure branch

Goal:
    Learn safety-related representation while reducing
    domain information leakage.

This is a prototype for feasibility validation.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class DisentangleNet(nn.Module):

    def __init__(self, n_u, n_s):

        super().__init__()

        self.domain_encoder = nn.Sequential(
            nn.Linear(n_u, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU()
        )

        self.safety_encoder = nn.Sequential(
            nn.Linear(n_u, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        self.structure_encoder = nn.Sequential(
            nn.Linear(n_s, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )

        self.domain_head = nn.Sequential(
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

        self.safety_head = nn.Sequential(
            nn.Linear(48, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, u, s):

        z_d = self.domain_encoder(u)

        z_safe_u = self.safety_encoder(u)

        z_safe_s = self.structure_encoder(s)

        safety = self.safety_head(
            torch.cat(
                [
                    z_safe_u,
                    z_safe_s
                ],
                dim=1
            )
        )

        domain = self.domain_head(z_d)

        return safety, domain, z_d, z_safe_u, z_safe_s


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
    parser.add_argument("--domain_weight", type=float, default=0.1)

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

    panel = pd.concat(
        [
            source,
            external
        ],
        ignore_index=True
    )

    cols = [
        c for c in source.columns
        if c not in [
            "sample_id",
            "outcome",
            "domain"
        ]
    ]

    u_cols = [
        c for c in cols
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
        c for c in cols
        if any(
            k in c.lower()
            for k in [
                "boundary",
                "fg_fraction",
                "struct",
                "shape",
                "topology"
            ]
        )
    ]

    panel[u_cols+s_cols] = (
        panel[u_cols+s_cols]
        .fillna(
            panel[u_cols+s_cols].median()
        )
    )

    y = (
        panel["outcome"]
        .astype(str)
        .str.lower()
        .isin(
            [
                "harm",
                "positive",
                "1"
            ]
        )
        .astype(float)
        .values
    )

    d = panel["domain"].astype(float).values

    dataset = FeatureDataset(
        panel[u_cols].values,
        panel[s_cols].values,
        y,
        d
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True
    )

    model = DisentangleNet(
        len(u_cols),
        len(s_cols)
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3
    )

    bce = nn.BCEWithLogitsLoss()

    print("===== R10E0 FEATURE AUDIT =====")
    print("Uncertainty:", len(u_cols))
    print("Structure:", len(s_cols))

    for epoch in range(args.epochs):

        model.train()
        losses = []

        for u, s, yy, dd in loader:

            py, pd, _, _, _ = model(u, s)

            source_mask = dd == 0

            loss_safety = bce(
                py[source_mask].squeeze(),
                yy[source_mask]
            )

            loss_domain = bce(
                pd.squeeze(),
                dd
            )

            loss = (
                loss_safety
                +
                args.domain_weight * loss_domain
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(float(loss))

        print(
            f"Epoch {epoch+1}/{args.epochs} "
            f"loss={np.mean(losses):.5f}"
        )

    torch.save(
        {
            "model": model.state_dict(),
            "u_features": u_cols,
            "s_features": s_cols,
            "config": vars(args)
        },
        out / "R10E0_disentangle_checkpoint.pt"
    )

    print("Checkpoint saved")
    print("PASS")


if __name__ == "__main__":
    main()
