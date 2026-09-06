
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R10A2_6_external_candidate_inspection.py

Purpose:
    Produce a human-auditable table for the remaining external
    feature candidates after lineage audit.

This step does NOT automatically select a candidate.
It only summarizes:
    - path
    - row count
    - column count
    - schema indicators
    - identity availability

Input:
    validated_external_candidates.csv

Output:
    external_candidate_inspection.csv
"""

import argparse
from pathlib import Path

import pandas as pd


def inspect_file(path):

    result = {
        "file": str(path),
    }

    try:
        df = pd.read_csv(path)

        cols = [
            str(c)
            for c in df.columns
        ]

        text = " ".join(
            [c.lower() for c in cols]
        )

        result.update(
            {
                "rows": len(df),
                "columns": len(cols),
                "has_sample_id":
                    any(
                        k in text
                        for k in [
                            "sample_id",
                            "image_id",
                            "case_id",
                        ]
                    ),
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
                "column_names":
                    "|".join(cols),
            }
        )

    except Exception as e:
        result["error"] = str(e)

    return result


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--candidate_csv",
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

    candidates = pd.read_csv(
        args.candidate_csv
    )

    rows = []

    for f in candidates["file"]:
        rows.append(
            inspect_file(f)
        )

    result = pd.DataFrame(rows)

    result.to_csv(
        out / "external_candidate_inspection.csv",
        index=False
    )

    print("===== Q1-R10A2-6 EXTERNAL CANDIDATE INSPECTION =====")
    print("Candidates:", len(result))
    print("Output:", out)

    assert len(result) == len(candidates)

    print("PASS")


if __name__ == "__main__":
    main()
