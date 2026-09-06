#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R14A_source_training_dataset_identity_audit_fix1.py

Purpose
-------
Locate the exact authoritative S01 source manifest by its frozen SHA256 and
audit the dataset identity of the 1450 SOURCE train+val images before choosing
a new untouched external cohort.

This is a read-only provenance audit:
- no model inference
- no GT decoding
- no outcome/safety score read
- no method selection by target performance

The script first searches likely manifest CSVs under F:\\MEDSEG_SAFETTA.
If not found, it falls back to all CSV files. It then:
1) verifies exact manifest SHA;
2) prints schema and low-cardinality categorical columns;
3) identifies likely path columns;
4) detects common public-polyp dataset tokens in paths/names;
5) summarizes TRAIN/VAL-like roles without guessing a scientific split;
6) writes a compact JSON/CSV audit.

No file is modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd
from tqdm import tqdm


ROOT = Path(r"F:\MEDSEG_SAFETTA")

EXPECTED_S01_MANIFEST_SHA256 = (
    "afb4bc1175af004819538cdbb22ee7a10f1be48035945c7e0d50fd2bbe2e3dc7"
)

OUTPUT_DIR = (
    ROOT / "outputs"
    / "Q1_R14A_source_training_dataset_identity_audit_fix1_v1"
)

DATASET_PATTERNS = {
    "Kvasir-SEG": [
        r"kvasir",
    ],
    "CVC-ClinicDB": [
        r"clinicdb",
        r"cvc[_\- ]?clinic",
        r"cvcclinic",
    ],
    "CVC-ColonDB": [
        r"colondb",
        r"cvc[_\- ]?colon",
        r"cvccolon",
    ],
    "ETIS-Larib": [
        r"etis",
        r"larib",
    ],
    "CVC-300/CVC-T": [
        r"cvc[_\- ]?300",
        r"cvc[_\- ]?t",
        r"cvc300",
    ],
    "EndoScene": [
        r"endoscene",
    ],
    "NeoPolyp": [
        r"neopolyp",
    ],
    "PolypGen": [
        r"polypgen",
    ],
}

LIKELY_MANIFEST_HINTS = (
    "manifest",
    "split",
    "fold",
    "source",
    "dataset",
    "index",
)


def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def candidate_csvs(root: Path):
    all_csv = []
    likely = []
    for p in root.rglob("*.csv"):
        try:
            if not p.is_file():
                continue
        except OSError:
            continue
        all_csv.append(p)
        name = p.name.lower()
        if any(h in name for h in LIKELY_MANIFEST_HINTS):
            likely.append(p)
    return likely, all_csv


def find_exact_manifest(root: Path):
    likely, all_csv = candidate_csvs(root)

    print(f"Likely CSV candidates: {len(likely)}")
    for p in tqdm(
        likely,
        desc="Hash likely manifests",
        unit="file",
        dynamic_ncols=True,
    ):
        try:
            if sha256_file(p) == EXPECTED_S01_MANIFEST_SHA256:
                return p, "likely_manifest_search"
        except (OSError, PermissionError):
            pass

    likely_set = set(likely)
    fallback = [p for p in all_csv if p not in likely_set]
    print(f"Fallback CSV candidates: {len(fallback)}")

    for p in tqdm(
        fallback,
        desc="Hash fallback CSVs",
        unit="file",
        dynamic_ncols=True,
    ):
        try:
            if sha256_file(p) == EXPECTED_S01_MANIFEST_SHA256:
                return p, "all_csv_fallback"
        except (OSError, PermissionError):
            pass

    return None, None


def low_cardinality_summary(df: pd.DataFrame):
    rows = []
    for c in df.columns:
        s = df[c].dropna()
        if len(s) == 0:
            continue
        nunique = s.astype(str).nunique()
        if nunique <= 30:
            counts = s.astype(str).value_counts(dropna=False)
            rows.append({
                "column": c,
                "nunique": int(nunique),
                "top_values": " | ".join(
                    f"{k}:{int(v)}"
                    for k, v in counts.head(20).items()
                ),
            })
    return pd.DataFrame(rows)


def likely_path_columns(df: pd.DataFrame):
    found = []
    for c in df.columns:
        lc = c.lower()
        if any(tok in lc for tok in ("path", "image", "file", "filename")):
            sample = df[c].dropna().astype(str).head(50)
            score = sum(
                (
                    "\\" in x
                    or "/" in x
                    or x.lower().endswith(
                        (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
                    )
                )
                for x in sample
            )
            if score > 0:
                found.append(c)
    return found


def dataset_token_audit(df: pd.DataFrame, path_cols):
    text = pd.Series([""] * len(df), index=df.index, dtype="object")
    for c in path_cols:
        text = text + " " + df[c].fillna("").astype(str)

    # Also include all object/string columns with modest average length.
    for c in df.columns:
        if c in path_cols:
            continue
        if df[c].dtype == "object":
            s = df[c].fillna("").astype(str)
            if len(s) and s.str.len().mean() < 200:
                text = text + " " + s

    lower = text.str.lower()

    rows = []
    matched_any = pd.Series(False, index=df.index)

    for name, pats in DATASET_PATTERNS.items():
        mask = pd.Series(False, index=df.index)
        for pat in pats:
            mask = mask | lower.str.contains(
                pat,
                regex=True,
                na=False,
            )
        matched_any = matched_any | mask
        rows.append({
            "dataset_token": name,
            "matched_rows": int(mask.sum()),
            "unique_rows_fraction": float(mask.mean()),
        })

    rows.append({
        "dataset_token": "NO_KNOWN_TOKEN",
        "matched_rows": int((~matched_any).sum()),
        "unique_rows_fraction": float((~matched_any).mean()),
    })
    return pd.DataFrame(rows)


def path_parent_tokens(df: pd.DataFrame, path_cols):
    ctr = Counter()
    for c in path_cols:
        for raw in df[c].dropna().astype(str):
            s = raw.replace("\\", "/")
            parts = [x.strip().lower() for x in s.split("/") if x.strip()]
            for part in parts[:-1]:
                if (
                    len(part) >= 3
                    and not re.fullmatch(r"[0-9a-f]{32,64}", part)
                ):
                    ctr[part] += 1

    return pd.DataFrame(
        [
            {"path_token": k, "count": int(v)}
            for k, v in ctr.most_common(100)
        ]
    )


def role_like_columns(df: pd.DataFrame):
    rows = []
    for c in df.columns:
        s = df[c].dropna().astype(str)
        if len(s) == 0:
            continue
        vals = s.str.lower()
        terms = (
            "train",
            "val",
            "validation",
            "test",
            "seen",
            "unseen",
            "source",
            "locked",
            "sanity",
        )
        hit = vals.str.contains(
            "|".join(terms),
            regex=True,
            na=False,
        )
        if hit.mean() >= 0.1 and s.nunique() <= 30:
            vc = s.value_counts()
            rows.append({
                "column": c,
                "nunique": int(s.nunique()),
                "values": " | ".join(
                    f"{k}:{int(v)}"
                    for k, v in vc.items()
                ),
            })
    return pd.DataFrame(rows)


def self_test():
    x = pd.DataFrame({
        "image_path": [
            r"X:\Kvasir-SEG\a.jpg",
            r"X:\CVC-ClinicDB\b.jpg",
            r"X:\ETIS-Larib\c.jpg",
        ],
        "role": ["train", "val", "unseen_locked"],
    })
    a = dataset_token_audit(x, ["image_path"])
    assert int(
        a.loc[a["dataset_token"] == "Kvasir-SEG", "matched_rows"].iloc[0]
    ) == 1
    assert int(
        a.loc[a["dataset_token"] == "CVC-ClinicDB", "matched_rows"].iloc[0]
    ) == 1
    assert int(
        a.loc[a["dataset_token"] == "ETIS-Larib", "matched_rows"].iloc[0]
    ) == 1
    print("DATASET_TOKEN_TEST_PASS")
    print("SELF_TEST_PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=Path,
        default=ROOT,
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
    )
    ap.add_argument(
        "--self-test",
        action="store_true",
    )
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    print("===== R14A SOURCE TRAINING DATASET IDENTITY AUDIT =====")
    print("Expected authoritative S01 manifest SHA256:")
    print(EXPECTED_S01_MANIFEST_SHA256)
    print("Read-only audit=YES")
    print("Model inference=NO")
    print("GT decode=NO")
    print("Safety/outcome score read=NO")
    print()

    manifest, search_mode = find_exact_manifest(args.root)
    if manifest is None:
        raise FileNotFoundError(
            "Could not locate a CSV with the exact authoritative S01 "
            f"manifest SHA256 under {args.root}"
        )

    actual_sha = sha256_file(manifest)
    print("\nExact S01 manifest found:")
    print(manifest)
    print("SHA256:", actual_sha)
    print("search mode:", search_mode)

    df = pd.read_csv(manifest, low_memory=False)

    print("\n===== MANIFEST SCHEMA =====")
    print("rows:", len(df))
    print("columns:", len(df.columns))
    print("column names:")
    for c in df.columns:
        print(" ", c)

    low = low_cardinality_summary(df)
    paths = likely_path_columns(df)
    role = role_like_columns(df)
    datasets = dataset_token_audit(df, paths)
    parents = path_parent_tokens(df, paths)

    print("\n===== LIKELY PATH COLUMNS =====")
    print(paths)

    print("\n===== LOW-CARDINALITY COLUMNS =====")
    if len(low):
        print(low.to_string(index=False))
    else:
        print("NONE")

    print("\n===== ROLE/SPLIT-LIKE COLUMNS =====")
    if len(role):
        print(role.to_string(index=False))
    else:
        print("NONE")

    print("\n===== DATASET TOKEN AUDIT =====")
    print(datasets.to_string(index=False))

    print("\n===== TOP PATH PARENT TOKENS =====")
    if len(parents):
        print(parents.head(50).to_string(index=False))
    else:
        print("NONE")

    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)

    low_path = args.output_dir / "R14A_low_cardinality_columns.csv"
    role_path = args.output_dir / "R14A_role_like_columns.csv"
    dataset_path = args.output_dir / "R14A_dataset_token_audit.csv"
    parent_path = args.output_dir / "R14A_path_parent_tokens.csv"

    low.to_csv(low_path, index=False)
    role.to_csv(role_path, index=False)
    datasets.to_csv(dataset_path, index=False)
    parents.to_csv(parent_path, index=False)

    summary = {
        "decision": "AUTHORITATIVE_S01_SOURCE_DATASET_IDENTITY_AUDIT_COMPLETE",
        "manifest_path": str(manifest),
        "manifest_sha256": actual_sha,
        "manifest_rows": int(len(df)),
        "manifest_columns": list(df.columns),
        "likely_path_columns": paths,
        "information_boundary": {
            "read_only": True,
            "model_inference": False,
            "gt_decode": False,
            "safety_or_outcome_score_read": False,
            "target_performance_selection": False,
        },
        "next_stage": (
            "SELECT_UNTOUCHED_EXTERNAL_COHORT_ONLY_AFTER_SOURCE_IDENTITY_REVIEW"
        ),
    }

    lock = args.output_dir / "R14A_SOURCE_DATASET_IDENTITY_AUDIT_LOCK.json"
    lock.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nDecision= AUTHORITATIVE_S01_SOURCE_DATASET_IDENTITY_AUDIT_COMPLETE")
    print("R14A LOCK:", lock)
    print("R14A LOCK SHA256:", sha256_file(lock))
    print("PASS")


if __name__ == "__main__":
    main()
