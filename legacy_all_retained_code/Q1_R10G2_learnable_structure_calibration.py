
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R10G2_learnable_structure_calibration.py

Learnable version of R10G1 fixed structure gate.

Compare:
- uncertainty only
- concat
- learnable structure calibration

Model:
    u -> uncertainty encoder
    s -> structure gate
    risk = fusion([u, u*gate])

Evaluation:
    AUROC
    AUPRC
    FPR@90
    PPV@1%
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score


class SGUC(nn.Module):

    def __init__(self, nu, ns):
        super().__init__()

        self.u_encoder = nn.Sequential(
            nn.Linear(nu, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        self.gate = nn.Sequential(
            nn.Linear(ns, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.Sigmoid()
        )

        self.head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, u, s):

        zu = self.u_encoder(u)
        g = self.gate(s)

        z = torch.cat(
            [
                zu,
                zu * g
            ],
            dim=1
        )

        return self.head(z), g


def metrics(y, p):

    idx = np.argsort(-p)

    threshold = p[idx][int(len(y)*0.1)]

    pred = (p >= threshold).astype(int)

    tp = ((pred==1)&(y==1)).sum()
    fp = ((pred==1)&(y==0)).sum()
    fn = ((pred==0)&(y==1)).sum()

    return {
        "AUROC": roc_auc_score(y,p),
        "AUPRC": average_precision_score(y,p),
        "Recall": tp/(tp+fn+1e-8),
        "FPR@90": fp/(np.sum(y==0)+1e-8),
        "PPV_1pct": tp/(tp+99*fp+1e-8)
    }


def main():

    parser=argparse.ArgumentParser()

    parser.add_argument("--panel",required=True)
    parser.add_argument("--output_dir",required=True)
    parser.add_argument("--epochs",type=int,default=80)

    args=parser.parse_args()

    out=Path(args.output_dir)
    out.mkdir(parents=True,exist_ok=True)

    df=pd.read_csv(args.panel,low_memory=False)

    df=df[df["domain"]==0].copy()

    y=(
        df["outcome"]
        .astype(str)
        .str.lower()
        .isin(["harm","positive","1"])
        .astype(float)
        .values
    )

    u_cols=[
        c for c in df.columns
        if any(k in c.lower()
        for k in [
            "entropy",
            "prob",
            "confidence",
            "logit",
            "rur"
        ])
    ]

    s_cols=[
        c for c in df.columns
        if any(k in c.lower()
        for k in [
            "struct",
            "boundary",
            "fg_fraction"
        ])
    ]

    X_u=torch.tensor(
        df[u_cols].fillna(0).values,
        dtype=torch.float32
    )

    X_s=torch.tensor(
        df[s_cols].fillna(0).values,
        dtype=torch.float32
    )

    Y=torch.tensor(
        y,
        dtype=torch.float32
    ).view(-1,1)

    model=SGUC(
        len(u_cols),
        len(s_cols)
    )

    opt=torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4
    )

    loss_fn=nn.BCEWithLogitsLoss()

    print("===== R10G2 LEARNABLE SGUC =====")
    print("Samples:",len(df))
    print("Uncertainty:",len(u_cols))
    print("Structure:",len(s_cols))

    for e in range(args.epochs):

        model.train()

        logit,_=model(
            X_u,
            X_s
        )

        loss=loss_fn(
            logit,
            Y
        )

        opt.zero_grad()
        loss.backward()
        opt.step()

        if (e+1)%10==0:
            print(
                f"Epoch {e+1}/{args.epochs} loss={loss.item():.5f}"
            )

    model.eval()

    with torch.no_grad():

        p=torch.sigmoid(
            model(X_u,X_s)[0]
        ).numpy().ravel()

    result=metrics(y,p)

    print("\n===== RESULT =====")
    print(result)

    torch.save(
        {
            "model":model.state_dict(),
            "u_features":u_cols,
            "s_features":s_cols,
            "metrics":result
        },
        out/"R10G2_SGUC_checkpoint.pt"
    )

    pd.DataFrame([result]).to_csv(
        out/"R10G2_SGUC_result.csv",
        index=False
    )

    print("PASS")


if __name__=="__main__":
    main()
