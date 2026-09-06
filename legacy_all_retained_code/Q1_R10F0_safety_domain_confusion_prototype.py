
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R10F0_safety_domain_confusion_prototype.py

Goal:
    Improve R10E0 by adding explicit domain confusion on safety branch.

Architecture:

uncertainty
      |
 -------------------------
 |                       |
z_domain              z_safe
 |                       |
domain head          safety head
                       |
                 domain confusion head

Loss:
    safety loss
    + domain classification loss(z_domain)
    + domain confusion loss(z_safe)

Prototype only.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class GradReverse(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -grad


def grad_reverse(x):
    return GradReverse.apply(x)


class DASDNet(nn.Module):

    def __init__(self, n_u, n_s):

        super().__init__()

        self.domain_encoder = nn.Sequential(
            nn.Linear(n_u, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU()
        )

        self.safe_unc_encoder = nn.Sequential(
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

        self.domain_head = nn.Linear(16, 1)

        self.safe_domain_head = nn.Sequential(
            nn.Linear(32, 16),
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

        z_su = self.safe_unc_encoder(u)

        z_ss = self.structure_encoder(s)

        safety = self.safety_head(
            torch.cat([z_su, z_ss], dim=1)
        )

        domain = self.domain_head(z_d)

        # adversarial domain prediction from safety latent
        confusion = self.safe_domain_head(
            grad_reverse(z_su)
        )

        return safety, domain, confusion, z_d, z_su, z_ss


class DS(Dataset):

    def __init__(self, u, s, y, d):

        self.u=torch.tensor(u,dtype=torch.float32)
        self.s=torch.tensor(s,dtype=torch.float32)
        self.y=torch.tensor(y,dtype=torch.float32)
        self.d=torch.tensor(d,dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self,i):
        return self.u[i],self.s[i],self.y[i],self.d[i]


def main():

    parser=argparse.ArgumentParser()

    parser.add_argument("--panel",required=True)
    parser.add_argument("--output_dir",required=True)
    parser.add_argument("--epochs",type=int,default=50)

    args=parser.parse_args()

    out=Path(args.output_dir)
    out.mkdir(parents=True,exist_ok=True)

    df=pd.read_csv(args.panel,low_memory=False)

    u_cols=[c for c in df.columns if any(k in c.lower()
        for k in ["entropy","prob","confidence","logit","rur"])]

    s_cols=[c for c in df.columns if any(k in c.lower()
        for k in ["struct","boundary","fg_fraction"])]

    df[u_cols+s_cols]=df[u_cols+s_cols].fillna(0)

    y=(df["outcome"].astype(str).str.lower()
       .isin(["harm","positive","1"]).astype(float).values)

    d=df["domain"].values.astype(float)

    ds=DS(
        df[u_cols].values,
        df[s_cols].values,
        y,
        d
    )

    loader=DataLoader(ds,batch_size=128,shuffle=True)

    model=DASDNet(len(u_cols),len(s_cols))

    opt=torch.optim.AdamW(
        model.parameters(),
        lr=1e-3
    )

    loss_fn=nn.BCEWithLogitsLoss()

    print("===== R10F0 DASD PROTOTYPE =====")
    print("Uncertainty:",len(u_cols))
    print("Structure:",len(s_cols))

    for e in range(args.epochs):

        losses=[]

        for u,s,yy,dd in loader:

            py,pd,pc,_,_,_=model(u,s)

            source=(dd==0)

            ls=loss_fn(
                py[source].squeeze(),
                yy[source]
            )

            ld=loss_fn(
                pd.squeeze(),
                dd
            )

            lc=loss_fn(
                pc.squeeze(),
                dd
            )

            loss=ls+0.1*ld+0.1*lc

            opt.zero_grad()
            loss.backward()
            opt.step()

            losses.append(loss.item())

        print(
            f"Epoch {e+1}/{args.epochs} loss={np.mean(losses):.5f}"
        )

    torch.save(
        {
            "model":model.state_dict(),
            "u_features":u_cols,
            "s_features":s_cols
        },
        out/"R10F0_DASD_checkpoint.pt"
    )

    print("PASS")


if __name__=="__main__":
    main()
