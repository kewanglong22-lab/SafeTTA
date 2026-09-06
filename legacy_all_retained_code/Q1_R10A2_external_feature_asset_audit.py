
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R10A2_external_feature_asset_audit.py

Purpose:
    Audit existing external feature assets before building
    source-external domain panel.

Protocol:
    - Search only existing output CSV files.
    - Identify candidate external feature tables.
    - Check schema compatibility with R10A source feature table.

Outputs:
    candidate_external_feature_assets.csv
    schema_comparison.csv
"""

import argparse
from pathlib import Path

import pandas as pd


KEYWORDS = [
    "entropy",
    "confidence",
    "logit",
    "boundary",
    "fg_fraction",
    "feature",
    "risk",
]


REQUIRED_PATTERN = [
    "entropy",
    "confidence",
    "logit",
]


def scan_csv(root):

    rows = []

    root = Path(root)

    files = list(root.rglob("*.csv"))

    print("CSV files scanned:", len(files))

    for f in files:

        name = f.name.lower()

        if any(k in name for k in KEYWORDS):

            try:
                df = pd.read_csv(
                    f,
                    nrows=5
                )

                cols = [
                    str(c)
                    for c in df.columns
                ]

                text = " ".join(cols).lower()

                score = sum(
                    1
                    for k in REQUIRED_PATTERN
                    if k in text
                )

                rows.append(
                    {
                        "file": str(f),
                        "columns": len(cols),
                        "matched_score": score,
                        "has_entropy":
                            "entropy" in text,
                        "has_confidence":
                            "confidence" in text,
                        "has_logit":
                            "logit" in text,
                        "has_boundary":
                            "boundary" in text,
                        "has_fg_fraction":
                            "fg_fraction" in text,
                    }
                )

            except Exception as e:

                rows.append(
                    {
                        "file": str(f),
                        "error": str(e),
                        "matched_score": -1,
                    }
                )

    return pd.DataFrame(rows)


def compare_schema(source_csv, candidates):

    source = pd.read_csv(
        source_csv,
        nrows=5
    )

    source_cols = set(
        source.columns
    )

    rows = []

    for _, r in candidates.iterrows():

        path = r["file"]

        try:

            df = pd.read_csv(
                path,
                nrows=5
            )

            cols = set(
                df.columns
            )

            overlap = (
                len(
                    source_cols & cols
                )
                /
                max(len(source_cols), 1)
            )

            rows.append(
                {
                    "file": path,
                    "source_column_overlap":
                        overlap,
                    "matched_columns":
                        ",".join(
                            sorted(
                                source_cols & cols
                            )
                        ),
                    "candidate_columns":
                        len(cols),
                }
            )

        except Exception as e:

            rows.append(
                {
                    "file": path,
                    "error": str(e)
                }
            )

    return pd.DataFrame(rows)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        required=True
    )

    parser.add_argument(
        "--source",
        required=True
    )

    parser.add_argument(
        "--output_dir",
        required=True
    )

    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(
        parents=True,
        exist_ok=True
    )

    candidates = scan_csv(
        args.root
    )

    candidates.to_csv(
        out / "candidate_external_feature_assets.csv",
        index=False
    )

    comparison = compare_schema(
        args.source,
        candidates
    )

    comparison.to_csv(
        out / "schema_comparison.csv",
        index=False
    )

    print("\n===== Q1-R10A2 EXTERNAL ASSET AUDIT =====")
    print("Candidates:", len(candidates))
    print("Output:", out)
    print("PASS")


if __name__ == "__main__":
    main()
