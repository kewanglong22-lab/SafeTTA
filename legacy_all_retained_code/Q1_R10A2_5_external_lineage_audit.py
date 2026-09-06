
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R10A2_5_external_lineage_audit.py

Purpose:
    Identify valid external feature assets by lineage constraints.

Reason:
    Feature similarity alone can incorrectly select source-side
    derived tables. This audit uses file lineage + schema checks.

Rules:
    Accept candidates only if:
      1. Path suggests external lineage
         (neopolyp/external/target/validation/prospective)
      2. Not generated from source-side analysis
      3. Contains required representation columns

Output:
    validated_external_candidates.csv
    rejected_candidates.csv
"""

import argparse
from pathlib import Path

import pandas as pd


POSITIVE_PATH_WORDS = [
    "neopolyp",
    "external",
    "target",
    "validation",
    "prospective",
]

NEGATIVE_PATH_WORDS = [
    "source",
    "r10a0",
    "r10a1",
    "audit",
    "summary",
    "invariant_feature",
]


REQUIRED_COL_GROUPS = [
    "entropy",
    "confidence",
    "logit",
]


def path_score(path):

    p = str(path).lower()

    positive = sum(
        w in p for w in POSITIVE_PATH_WORDS
    )

    negative = sum(
        w in p for w in NEGATIVE_PATH_WORDS
    )

    return positive - negative


def schema_score(path):

    try:
        df = pd.read_csv(
            path,
            nrows=3
        )

        cols = " ".join(
            [str(c).lower() for c in df.columns]
        )

        return {
            "has_entropy": "entropy" in cols,
            "has_confidence": "confidence" in cols,
            "has_logit": "logit" in cols,
            "rows_preview": len(df),
        }

    except Exception as e:
        return {
            "error": str(e)
        }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
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

    candidates = []

    for csv in Path(args.root).rglob("*.csv"):

        score = path_score(csv)

        if score <= 0:
            continue

        info = schema_score(csv)

        info["file"] = str(csv)
        info["lineage_score"] = score

        valid_schema = (
            info.get("has_entropy", False)
            and info.get("has_confidence", False)
            and info.get("has_logit", False)
        )

        info["schema_valid"] = valid_schema

        candidates.append(info)

    result = pd.DataFrame(candidates)

    if len(result):
        valid = result[result["schema_valid"]]
        rejected = result[~result["schema_valid"]]
    else:
        valid = result
        rejected = result

    valid.to_csv(
        out / "validated_external_candidates.csv",
        index=False
    )

    rejected.to_csv(
        out / "rejected_candidates.csv",
        index=False
    )

    print("===== Q1-R10A2-5 EXTERNAL LINEAGE AUDIT =====")
    print("Valid candidates:", len(valid))
    print("Rejected candidates:", len(rejected))
    print("Output:", out)

    print("PASS")


if __name__ == "__main__":
    main()
