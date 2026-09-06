
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R10H1_external_explicit_key_lineage_search_fix1.py

Purpose
-------
Search the existing MEDSEG_SAFETTA outputs for an upstream NeoPolyp/external
CSV that preserves explicit row identity for the 9000-row / 1000-case×9
external feature lineage.

Why this is needed
------------------
The exact feature-fingerprint recovery attempt failed:
- external fingerprints: 9000 unique
- R08E1 fingerprints: 9000 unique
- intersection: 0
- within-sample max-abs distance is large

Therefore R08E1 and the current external invariant table must NOT be joined
through shared feature names or row order.

This audit searches for explicit keys such as:
    sample_id
    model_family
    model_state_id
    training_seed
    checkpoint_sha256

and checks whether candidate files:
1) have exactly the same 1000 sample IDs as the current external table,
2) have the same multiplicity pattern (9 rows per sample),
3) preserve a unique row key,
4) overlap structurally with the post-GT R05D4 outcome table.

No labels are attached and no row-order merge is performed.
"""

import argparse
from pathlib import Path
from itertools import combinations
import os
import pandas as pd
import numpy as np

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x


DEFAULT_ROOT = "F:/MEDSEG_SAFETTA/outputs"
DEFAULT_EXTERNAL = (
    "F:/MEDSEG_SAFETTA/outputs/"
    "Q1_R10E1_external_complete_feature_v3/"
    "external_complete_invariant_feature_table_v2.csv"
)
DEFAULT_R05D4 = (
    "F:/MEDSEG_SAFETTA/outputs/"
    "Q1_R05D4_neopolyp_gt_reveal_paot_confirmatory_evaluation_v1/"
    "model_case_outcomes.csv"
)
DEFAULT_OUTPUT = (
    "F:/MEDSEG_SAFETTA/outputs/"
    "Q1_R10H1_external_explicit_key_lineage_search_fix1_v1"
)

ID_COL = "sample_id"

IDENTITY_FIELDS = [
    "model_family",
    "model_state_id",
    "training_seed",
    "checkpoint_sha256",
    "model_id",
    "state_id",
    "seed",
    "fold",
    "method",
    "adaptation_method",
    "adaptation",
]

LABEL_FIELDS = [
    "outcome",
    "adaptation_outcome",
    "harm_label",
    "benefit_label",
    "source_dice",
    "a1_dice",
    "delta_dice",
]

NAME_HINTS = [
    "neopolyp",
    "external",
    "r10a2",
    "r10e1",
    "r08e1",
    "r08f0",
    "r05d",
]


def norm_sid(s):
    return (
        s.astype(str)
         .str.strip()
         .str.replace("\\", "/", regex=False)
         .str.lower()
    )


def safe_read_header(path):
    try:
        return list(pd.read_csv(path, nrows=0).columns)
    except Exception:
        return None


def key_uniqueness(df, cols):
    if not all(c in df.columns for c in cols):
        return None
    return int(df[cols].drop_duplicates().shape[0])


def compact_unique_key_audit(df):
    available = [c for c in IDENTITY_FIELDS if c in df.columns]
    results = []

    # sample_id + 1 or 2 explicit identity fields is enough to identify
    # the likely row key in this 9-state-per-case setting.
    for r in (1, 2):
        for comb in combinations(available, r):
            cols = [ID_COL] + list(comb)
            u = key_uniqueness(df, cols)
            results.append((cols, u))

    results.sort(key=lambda x: (x[1] != len(df), -x[1], len(x[0])))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--external", default=DEFAULT_EXTERNAL)
    ap.add_argument("--r05d4", default=DEFAULT_R05D4)
    ap.add_argument("--output_dir", default=DEFAULT_OUTPUT)
    ap.add_argument("--top", type=int, default=40)
    args = ap.parse_args()

    root = Path(args.root)
    ext_path = Path(args.external)
    r05_path = Path(args.r05d4)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("===== R10H1 EXPLICIT-KEY LINEAGE SEARCH FIX1 =====")
    print("Root:", root)
    print("ROW-ORDER MATCHING: FORBIDDEN")
    print("FEATURE-NAME ASSUMPTION: FORBIDDEN")
    print("SILENT LABEL ATTACHMENT: FORBIDDEN")
    print("Goal: find an upstream file with explicit row identity.\n")

    if not ext_path.exists():
        raise FileNotFoundError(ext_path)
    if not r05_path.exists():
        raise FileNotFoundError(r05_path)

    ext = pd.read_csv(ext_path, usecols=[ID_COL], low_memory=False)
    r05_header = safe_read_header(r05_path)
    r05_use = [c for c in [ID_COL] + IDENTITY_FIELDS + LABEL_FIELDS if c in r05_header]
    r05 = pd.read_csv(r05_path, usecols=r05_use, low_memory=False)

    ext_ids = norm_sid(ext[ID_COL])
    ext_set = set(ext_ids)
    ext_counts = ext_ids.value_counts()

    print("Current external rows:", len(ext))
    print("Current external unique IDs:", len(ext_set))
    print("Current external max multiplicity:", int(ext_counts.max()))
    print("Current external multiplicity distribution:")
    print(ext_counts.value_counts().sort_index().to_string())

    r05_ids = norm_sid(r05[ID_COL])
    r05_set = set(r05_ids)
    print("\nR05D4 rows:", len(r05))
    print("R05D4 unique IDs:", len(r05_set))
    print("External vs R05D4 ID-set equality:", ext_set == r05_set)

    csvs = sorted(root.rglob("*.csv"))
    print("\nCSV files discovered:", len(csvs))

    rows = []

    for p in tqdm(csvs, desc="Scanning CSV lineage"):
        # Skip our own current output to avoid self-recursion.
        try:
            if out in p.parents:
                continue
        except Exception:
            pass

        header = safe_read_header(p)
        if not header or ID_COL not in header:
            continue

        id_fields = [c for c in IDENTITY_FIELDS if c in header]
        label_fields = [c for c in LABEL_FIELDS if c in header]

        # We are specifically interested in files that either preserve explicit
        # identity or are strongly likely to belong to the NeoPolyp/external chain.
        path_l = str(p).lower()
        hinted = any(h in path_l for h in NAME_HINTS)
        if not id_fields and not hinted:
            continue

        usecols = [ID_COL] + id_fields + label_fields

        try:
            df = pd.read_csv(p, usecols=list(dict.fromkeys(usecols)), low_memory=False)
        except Exception as e:
            rows.append({
                "path": str(p),
                "read_error": repr(e),
            })
            continue

        ids = norm_sid(df[ID_COL])
        s = set(ids)
        counts = ids.value_counts()

        inter = len(s & ext_set)
        ext_coverage = inter / len(ext_set) if ext_set else np.nan
        cand_coverage = inter / len(s) if s else np.nan
        id_set_equal = s == ext_set

        same_rows = len(df) == len(ext)
        multiplicity_9 = (
            len(counts) > 0
            and int(counts.min()) == 9
            and int(counts.max()) == 9
        )

        key_audit = compact_unique_key_audit(df)
        unique_keys = [
            "+".join(cols)
            for cols, u in key_audit
            if u == len(df)
        ]
        best_unique_key = unique_keys[0] if unique_keys else ""

        # Compare structural suitability to R05D4.
        shared_identity = [
            c for c in IDENTITY_FIELDS
            if c in df.columns and c in r05.columns
        ]

        score = 0
        score += 100 if id_set_equal else int(50 * ext_coverage)
        score += 40 if same_rows else 0
        score += 40 if multiplicity_9 else 0
        score += 50 if best_unique_key else 0
        score += 10 * min(len(shared_identity), 4)
        score += 10 if hinted else 0
        score += 5 if label_fields else 0

        rows.append({
            "score": score,
            "path": str(p),
            "rows": len(df),
            "columns_in_header": len(header),
            "unique_sample_ids": len(s),
            "external_id_intersection": inter,
            "external_id_coverage": ext_coverage,
            "candidate_id_coverage": cand_coverage,
            "id_set_equal": id_set_equal,
            "same_9000_rows": same_rows,
            "multiplicity_exactly_9": multiplicity_9,
            "explicit_identity_fields": ";".join(id_fields),
            "shared_identity_with_R05D4": ";".join(shared_identity),
            "best_unique_key": best_unique_key,
            "all_unique_keys": ";".join(unique_keys[:10]),
            "label_fields": ";".join(label_fields),
            "path_hint": hinted,
            "read_error": "",
        })

    res = pd.DataFrame(rows)

    if len(res) == 0:
        print("\nNo candidate CSVs found.")
        print("DECISION: EXPLICIT_KEY_NOT_FOUND")
        return

    # Missing columns from error rows are filled for sorting.
    for c in [
        "score", "external_id_coverage", "id_set_equal",
        "same_9000_rows", "multiplicity_exactly_9"
    ]:
        if c not in res.columns:
            res[c] = 0

    res = res.sort_values(
        by=["score", "external_id_coverage", "path"],
        ascending=[False, False, True],
        na_position="last",
    )

    res.to_csv(out / "explicit_key_lineage_candidates.csv", index=False)

    strong = res[
        (res["id_set_equal"] == True)
        & (res["same_9000_rows"] == True)
        & (res["multiplicity_exactly_9"] == True)
    ].copy()

    strong.to_csv(out / "strong_9000row_candidates.csv", index=False)

    print("\n===== TOP CANDIDATES =====")
    show_cols = [
        "score",
        "path",
        "rows",
        "unique_sample_ids",
        "id_set_equal",
        "multiplicity_exactly_9",
        "explicit_identity_fields",
        "shared_identity_with_R05D4",
        "best_unique_key",
        "label_fields",
    ]
    print(
        res[show_cols]
        .head(args.top)
        .to_string(index=False)
    )

    print("\n===== STRONG 9000-ROW / 1000×9 CANDIDATES =====")
    if len(strong) == 0:
        print("NONE")
    else:
        print(
            strong[show_cols]
            .head(args.top)
            .to_string(index=False)
        )

    recoverable = strong[
        strong["best_unique_key"].astype(str).str.len() > 0
    ]

    print("\n===== FINAL DECISION =====")
    if len(recoverable) > 0:
        print("DECISION: EXPLICIT_KEY_CANDIDATE_FOUND")
        print("Recoverable candidates:", len(recoverable))
        print("Best candidate:")
        print(recoverable.iloc[0]["path"])
        print("Best unique key:", recoverable.iloc[0]["best_unique_key"])
        print(
            "NEXT: audit that candidate against R05D4 field-by-field before "
            "building any labeled external panel."
        )
    else:
        print("DECISION: EXPLICIT_KEY_NOT_FOUND")
        print(
            "No existing CSV simultaneously preserves the 9000-row external "
            "case set, 9× multiplicity, and a unique explicit row key."
        )
        print(
            "NEXT: trace upstream manifests/scripts that created the R10A2/R10E1 "
            "external table; do not use row order as a substitute."
        )

    print("\nOutputs:")
    print(out / "explicit_key_lineage_candidates.csv")
    print(out / "strong_9000row_candidates.csv")
    print("PASS: SEARCH COMPLETED")


if __name__ == "__main__":
    main()
