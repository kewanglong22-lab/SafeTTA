
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R10E1_feature_lineage_repair.py

Purpose:
    Build the exact feature panel required by R10E0/R10E1.

Checks:
    - checkpoint feature requirements
    - source/external feature availability
    - missing columns
    - no silent zero padding

Output:
    Q1_R10E1_verified_panel.csv
"""

import argparse
from pathlib import Path
import pandas as pd
import torch


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--external", required=True)
    parser.add_argument("--output_dir", required=True)

    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(
        args.checkpoint,
        map_location="cpu"
    )

    required_u = ckpt["u_features"]
    required_s = ckpt["s_features"]

    source = pd.read_csv(
        args.source,
        low_memory=False
    )

    external = pd.read_csv(
        args.external,
        low_memory=False
    )

    source["domain"] = 0
    external["domain"] = 1

    if "sample_id" not in external.columns:
        external["sample_id"] = [
            f"external_{i}"
            for i in range(len(external))
        ]

    if "outcome" not in external.columns:
        external["outcome"] = "unknown"

    required = (
        ["sample_id", "outcome", "domain"]
        + required_u
        + required_s
    )

    print("===== R10E1 LINEAGE REPAIR AUDIT =====")

    for name, df in [
        ("source", source),
        ("external", external)
    ]:
        missing = [
            c for c in required
            if c not in df.columns
        ]

        print("\n", name)
        print("Rows:", len(df))
        print("Missing:", missing)

        if len(missing) > 0:
            raise ValueError(
                f"{name} missing required checkpoint features"
            )

    source_panel = source[required].copy()
    external_panel = external[required].copy()

    panel = pd.concat(
        [
            source_panel,
            external_panel
        ],
        ignore_index=True
    )

    output = (
        out /
        "Q1_R10E1_verified_panel.csv"
    )

    panel.to_csv(
        output,
        index=False
    )

    print("\nFinal rows:", len(panel))
    print("Final columns:", len(panel.columns))
    print("Missing after repair: []")
    print("Output:", output)
    print("PASS")


if __name__ == "__main__":
    main()
