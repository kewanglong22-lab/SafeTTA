
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R10A2_build_domain_panel.py

Purpose:
    Build source/external domain panel for Q1-R10A representation audit.

Input:
    source invariant feature table
    external invariant feature table

Output:
    merged domain panel with:
        domain = source/external

This step is separated from classifier audit to avoid
domain label leakage during feature construction.

"""

import argparse
from pathlib import Path

import pandas as pd


def load_and_tag(path, domain):

    df = pd.read_csv(path)

    df["domain"] = domain

    return df


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
        "--output",
        required=True
    )

    args = parser.parse_args()

    source = load_and_tag(
        args.source,
        "source"
    )

    external = load_and_tag(
        args.external,
        "external"
    )

    common_cols = sorted(
        list(
            set(source.columns)
            &
            set(external.columns)
        )
    )

    source = source[common_cols]
    external = external[common_cols]

    panel = pd.concat(
        [
            source,
            external
        ],
        axis=0,
        ignore_index=True
    )

    Path(args.output).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    panel.to_csv(
        args.output,
        index=False
    )

    print("===== Q1-R10A-2 DOMAIN PANEL =====")
    print("Source:", len(source))
    print("External:", len(external))
    print("Features:", len(panel.columns))
    print("Output:", args.output)

    assert "domain" in panel.columns
    assert len(panel) == len(source)+len(external)

    print("PASS")


if __name__ == "__main__":
    main()
