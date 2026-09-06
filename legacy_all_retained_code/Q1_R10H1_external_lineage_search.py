
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R10H1_external_lineage_search.py

Purpose:
    Search MEDSEG_SAFETTA outputs for possible external label lineage files.

Search keywords:
    label, class, outcome, diagnosis, gt, mask, neo, cancer,
    harm, sample_id, image_id, filename

Output:
    candidate_label_sources.csv
"""

import argparse
from pathlib import Path
import pandas as pd


KEYWORDS = [
    "label",
    "class",
    "outcome",
    "diagnosis",
    "gt",
    "mask",
    "neo",
    "cancer",
    "harm",
    "sample_id",
    "image_id",
    "filename"
]


def inspect_file(path):

    records = []

    try:
        if path.suffix.lower() in [".csv"]:
            df = pd.read_csv(
                path,
                nrows=5,
                low_memory=False
            )

            cols = list(df.columns)

            matched = [
                c for c in cols
                if any(
                    k in c.lower()
                    for k in KEYWORDS
                )
            ]

            if matched:
                records.append({
                    "file": str(path),
                    "type": "csv",
                    "columns": len(cols),
                    "matched_columns": ";".join(matched)
                })

        elif path.suffix.lower() in [".json"]:
            df = pd.read_json(path)

            cols = list(df.columns)

            matched = [
                c for c in cols
                if any(
                    k in c.lower()
                    for k in KEYWORDS
                )
            ]

            if matched:
                records.append({
                    "file": str(path),
                    "type": "json",
                    "columns": len(cols),
                    "matched_columns": ";".join(matched)
                })

        elif path.suffix.lower() in [".txt"]:
            text = path.read_text(
                encoding="utf-8",
                errors="ignore"
            ).lower()

            matched = [
                k for k in KEYWORDS
                if k in text
            ]

            if matched:
                records.append({
                    "file": str(path),
                    "type": "txt",
                    "columns": -1,
                    "matched_columns": ";".join(matched)
                })

    except Exception:
        pass

    return records


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

    root = Path(args.root)
    out = Path(args.output_dir)
    out.mkdir(
        parents=True,
        exist_ok=True
    )

    rows = []

    print("===== R10H1 EXTERNAL LABEL LINEAGE SEARCH =====")
    print("Root:", root)

    files = list(root.rglob("*"))

    print("Files scanned:", len(files))

    for i, f in enumerate(files):

        if f.is_file():

            rows.extend(
                inspect_file(f)
            )

            if (i + 1) % 1000 == 0:
                print(
                    "Scanned:",
                    i + 1
                )

    result = pd.DataFrame(rows)

    if len(result):
        result = result.sort_values(
            by=["file"]
        )

    result.to_csv(
        out / "candidate_label_sources.csv",
        index=False
    )

    print("\n===== RESULT =====")
    print("Candidates:", len(result))

    if len(result):
        print(
            result.to_string(index=False)
        )

    print("Output:", out)
    print("PASS")


if __name__ == "__main__":
    main()
