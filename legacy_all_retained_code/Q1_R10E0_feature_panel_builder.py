
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R10E0_feature_panel_builder.py

Build the correct R10E0/R10E1 feature panel.

Purpose:
    Merge source and external invariant feature tables.
    Preserve feature lineage.
    Add domain labels.

Output:
    Q1_R10E0_training_panel.csv

Checks:
    - required columns
    - shared features
    - missing features
    - row counts
"""

import argparse
from pathlib import Path
import pandas as pd


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--source", required=True)
    parser.add_argument("--external", required=True)
    parser.add_argument("--output_dir", required=True)

    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    source = pd.read_csv(
        args.source,
        low_memory=False
    )

    external = pd.read_csv(
        args.external,
        low_memory=False
    )

    print("===== R10E0 PANEL AUDIT =====")

    print("Source rows:", len(source))
    print("External rows:", len(external))

    source["domain"] = 0
    external["domain"] = 1

    if "outcome" not in external.columns:
        external["outcome"] = "unknown"

    if "sample_id" not in external.columns:
        external["sample_id"] = [
            f"external_{i}"
            for i in range(len(external))
        ]

    ignore = [
        "domain"
    ]

    source_features = [
        c for c in source.columns
        if c not in ignore
    ]

    external_features = [
        c for c in external.columns
        if c not in ignore
    ]

    shared = sorted(
        list(
            set(source_features)
            &
            set(external_features)
        )
    )

    missing_source = sorted(
        list(
            set(external_features)
            -
            set(source_features)
        )
    )

    missing_external = sorted(
        list(
            set(source_features)
            -
            set(external_features)
        )
    )

    print("Shared columns:", len(shared))

    print(
        "Missing in source:",
        missing_source
    )

    print(
        "Missing in external:",
        missing_external
    )

    # keep common feature lineage only
    keep = [
        c for c in shared
        if c not in [
            "outcome"
        ]
    ]

    required = [
        "sample_id",
        "outcome",
        "domain"
    ]

    for c in required:
        if c not in source.columns:
            raise ValueError(
                f"Source missing {c}"
            )

        if c not in external.columns:
            raise ValueError(
                f"External missing {c}"
            )

    source_panel = source[
        required + keep
    ].copy()

    external_panel = external[
        required + keep
    ].copy()

    panel = pd.concat(
        [
            source_panel,
            external_panel
        ],
        ignore_index=True
    )

    output = out / "Q1_R10E0_training_panel.csv"

    panel.to_csv(
        output,
        index=False
    )

    print("\nFinal rows:", len(panel))
    print("Final features:", len(keep))
    print("Output:", output)
    print("PASS")


if __name__ == "__main__":
    main()
