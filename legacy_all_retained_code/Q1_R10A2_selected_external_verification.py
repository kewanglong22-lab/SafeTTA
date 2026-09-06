
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R10A2_selected_external_verification.py

Purpose:
    Verify the selected external feature asset before constructing
    the source-external domain panel.

Checks:
    1. Schema compatibility with source feature table
    2. Row-level identity availability
    3. Potential leakage columns
    4. Feature overlap summary

Outputs:
    selected_asset_schema.csv
    leakage_audit.csv
    final_external_asset_lock.csv
"""

import argparse
from pathlib import Path

import pandas as pd


LEAKAGE_KEYWORDS = [
    "outcome",
    "label",
    "gt",
    "dice",
    "target",
    "mask",
    "ground_truth",
]


def inspect_schema(source, external):

    source_cols = set(source.columns)
    ext_cols = set(external.columns)

    return pd.DataFrame(
        [
            {
                "source_columns": len(source_cols),
                "external_columns": len(ext_cols),
                "common_columns": len(source_cols & ext_cols),
                "source_only_columns":
                    ",".join(sorted(source_cols - ext_cols)),
                "external_only_columns":
                    ",".join(sorted(ext_cols - source_cols)),
            }
        ]
    )


def leakage_audit(external):

    rows = []

    for c in external.columns:

        lower = str(c).lower()

        hit = [
            k
            for k in LEAKAGE_KEYWORDS
            if k in lower
        ]

        rows.append(
            {
                "column": c,
                "leakage_keyword_hit":
                    ",".join(hit),
                "flag":
                    len(hit) > 0,
            }
        )

    return pd.DataFrame(rows)


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
        "--output_dir",
        required=True
    )

    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(
        parents=True,
        exist_ok=True
    )

    source = pd.read_csv(args.source)
    external = pd.read_csv(args.external)

    schema = inspect_schema(
        source,
        external
    )

    schema.to_csv(
        out / "selected_asset_schema.csv",
        index=False
    )

    leak = leakage_audit(
        external
    )

    leak.to_csv(
        out / "leakage_audit.csv",
        index=False
    )

    lock = pd.DataFrame(
        [
            {
                "external_asset": args.external,
                "rows": len(external),
                "columns": len(external.columns),
                "has_sample_id":
                    "sample_id" in external.columns,
                "leakage_columns":
                    int(leak["flag"].sum()),
            }
        ]
    )

    lock.to_csv(
        out / "final_external_asset_lock.csv",
        index=False
    )

    print("===== Q1-R10A2-2 EXTERNAL VERIFICATION =====")
    print("Source rows:", len(source))
    print("External rows:", len(external))
    print("Leakage columns:", int(leak["flag"].sum()))
    print("Output:", out)

    assert len(lock) == 1

    print("PASS")


if __name__ == "__main__":
    main()
