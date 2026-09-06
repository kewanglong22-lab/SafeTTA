#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R14B4A_reference_asset_schema_path_preflight_fix1.py

Preflight for SUN-SEG contamination audit.

Purpose:
- verify the exact R14B3 candidate manifest;
- audit the comparison manifests for S01, NeoPolyp, and PolypGen;
- report exact schemas, hash columns, path-like columns, and whether path values
  already resolve as absolute existing files;
- do NOT guess filesystem roots;
- do NOT decode any image pixels;
- do NOT perform contamination comparisons yet.

This stage exists to prevent silent path reconstruction or wrong-column use in
the actual R14B4 contamination audit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(r"F:\MEDSEG_SAFETTA")

R14B3_DIR = (
    ROOT / "outputs"
    / "Q1_R14B3_sunseg_unseen_case_balanced_candidate_lock_fix1_v1"
)

DEFAULT_SUN_MANIFEST = (
    R14B3_DIR / "R14B3_SUNSEG_PRECONTAMINATION_CANDIDATE_MANIFEST.csv"
)
DEFAULT_SUN_LOCK = (
    R14B3_DIR / "R14B3_SUNSEG_PRECONTAMINATION_CANDIDATE_LOCK.json"
)

DEFAULT_S01_MANIFEST = (
    ROOT / "data" / "splits" / "S01_locked_protocol_manifest_v1.csv"
)

DEFAULT_NEOPOLYP_MANIFEST = (
    ROOT / "outputs"
    / "Q1_R05D1_neopolyp_acquisition_pairing_duplicate_qc_v1_fix2"
    / "frozen_confirmatory_manifest.csv"
)

DEFAULT_POLYPGEN_MANIFEST = (
    ROOT / "outputs"
    / "Q1_R10L1B_polypgen_unique_static_manifest_lock_fix3_v1"
    / "R10L1B_POLYPGEN_UNIQUE_STATIC_MANIFEST.csv"
)

EXPECTED_SUN_MANIFEST_SHA = (
    "f38cc8e5af4a007df3394ec95fa621c0b819f488bdd797ddad9338ac12491b4d"
)
EXPECTED_SUN_ROWS = 980
EXPECTED_SUN_CASES = 49
EXPECTED_S01_ROWS = 2248
EXPECTED_NEOPOLYP_ROWS = 1000
EXPECTED_POLYPGEN_ROWS = 1532

DECISION = "R14B4_REFERENCE_ASSET_SCHEMA_PATH_PREFLIGHT_COMPLETE"


def sha256_file(path: Path, chunk=8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def read_csv(path: Path):
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return reader.fieldnames or [], rows


def write_csv(path: Path, rows, fields):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def classify_columns(columns):
    path_like = []
    hash_like = []
    id_like = []

    for c in columns:
        low = c.lower()
        if any(tok in low for tok in ("path", "filepath", "file_path", "relpath")):
            path_like.append(c)
        if any(tok in low for tok in ("sha256", "hash", "md5")):
            hash_like.append(c)
        if any(tok in low for tok in ("sample_id", "image_id", "case_id", "pair_key", "center")):
            id_like.append(c)

    return path_like, hash_like, id_like


def value_preview(rows, column, n=5):
    out = []
    for r in rows:
        v = str(r.get(column, "")).strip()
        if v:
            out.append(v)
        if len(out) >= n:
            break
    return out


def audit_path_column(rows, column):
    nonempty = 0
    absolute = 0
    existing = 0
    files = 0
    dirs = 0

    for r in rows:
        raw = str(r.get(column, "")).strip()
        if not raw:
            continue
        nonempty += 1
        p = Path(raw)
        if p.is_absolute():
            absolute += 1
            if p.exists():
                existing += 1
                files += int(p.is_file())
                dirs += int(p.is_dir())

    return {
        "column": column,
        "nonempty": nonempty,
        "absolute": absolute,
        "existing_as_is": existing,
        "existing_files_as_is": files,
        "existing_dirs_as_is": dirs,
    }


def print_dataset_audit(name, path, expected_rows):
    if not path.exists():
        raise FileNotFoundError(f"{name} manifest not found: {path}")

    columns, rows = read_csv(path)
    if len(rows) != expected_rows:
        raise RuntimeError(
            f"{name} row count {len(rows)} != expected {expected_rows}"
        )

    path_like, hash_like, id_like = classify_columns(columns)

    print(f"\n===== {name} =====")
    print("manifest:", path)
    print("rows:", len(rows))
    print("cols:", len(columns))
    print("SHA256:", sha256_file(path))
    print("columns:")
    for c in columns:
        print("  ", c)

    print("path-like columns:", path_like)
    print("hash-like columns:", hash_like)
    print("id-like columns:", id_like)

    path_audits = []
    if path_like:
        print("\nPATH COLUMN AUDIT (NO ROOT GUESSING):")
        for c in path_like:
            a = audit_path_column(rows, c)
            path_audits.append(a)
            print(
                f"  {c}: nonempty={a['nonempty']} "
                f"absolute={a['absolute']} "
                f"existing_as_is={a['existing_as_is']} "
                f"files={a['existing_files_as_is']} "
                f"dirs={a['existing_dirs_as_is']}"
            )
            for v in value_preview(rows, c):
                print("    example:", v)

    if hash_like:
        print("\nHASH COLUMN PREVIEW:")
        for c in hash_like:
            vals = value_preview(rows, c)
            print(f"  {c}:")
            for v in vals:
                print("    ", v)

    return {
        "name": name,
        "path": str(path),
        "manifest_sha256": sha256_file(path),
        "row_count": len(rows),
        "columns": columns,
        "path_like_columns": path_like,
        "hash_like_columns": hash_like,
        "id_like_columns": id_like,
        "path_audits": path_audits,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sun-manifest", type=Path, default=DEFAULT_SUN_MANIFEST)
    ap.add_argument("--sun-lock", type=Path, default=DEFAULT_SUN_LOCK)
    ap.add_argument("--s01-manifest", type=Path, default=DEFAULT_S01_MANIFEST)
    ap.add_argument("--neopolyp-manifest", type=Path, default=DEFAULT_NEOPOLYP_MANIFEST)
    ap.add_argument("--polypgen-manifest", type=Path, default=DEFAULT_POLYPGEN_MANIFEST)
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=(
            ROOT / "outputs"
            / "Q1_R14B4A_reference_asset_schema_path_preflight_fix1_v1"
        ),
    )
    args = ap.parse_args()

    print("===== Q1 R14B4A REFERENCE ASSET SCHEMA/PATH PREFLIGHT =====")
    print("PATH_ROOT_GUESSING=NO")
    print("IMAGE_PIXEL_DECODE=NO")
    print("GT_PIXEL_DECODE=NO")
    print("CONTAMINATION_COMPARISON=NO")
    print("MODEL_INFERENCE=NO")
    print("TTA_EXECUTION=NO")
    print()

    if not args.sun_manifest.exists():
        raise FileNotFoundError(args.sun_manifest)
    if not args.sun_lock.exists():
        raise FileNotFoundError(args.sun_lock)

    sun_sha = sha256_file(args.sun_manifest)
    if sun_sha != EXPECTED_SUN_MANIFEST_SHA:
        raise RuntimeError(
            f"R14B3 candidate manifest SHA mismatch: {sun_sha}"
        )

    sun_cols, sun_rows = read_csv(args.sun_manifest)
    if len(sun_rows) != EXPECTED_SUN_ROWS:
        raise RuntimeError(
            f"SUN candidate rows {len(sun_rows)} != {EXPECTED_SUN_ROWS}"
        )

    case_col = "cluster_id"
    if case_col not in sun_cols:
        raise RuntimeError("SUN candidate manifest missing cluster_id.")
    cases = {r[case_col] for r in sun_rows}
    if len(cases) != EXPECTED_SUN_CASES:
        raise RuntimeError(
            f"SUN candidate physical cases {len(cases)} != {EXPECTED_SUN_CASES}"
        )

    sun_lock = json.loads(args.sun_lock.read_text(encoding="utf-8"))
    if sun_lock.get("status") != "PASS":
        raise RuntimeError("R14B3 lock status is not PASS.")

    print("R14B3 candidate SHA gate=PASS")
    print("SUN rows:", len(sun_rows))
    print("SUN physical cases:", len(cases))
    print("SUN columns:")
    for c in sun_cols:
        print("  ", c)

    results = []
    results.append(
        print_dataset_audit("S01", args.s01_manifest, EXPECTED_S01_ROWS)
    )
    results.append(
        print_dataset_audit(
            "NeoPolyp", args.neopolyp_manifest, EXPECTED_NEOPOLYP_ROWS
        )
    )
    results.append(
        print_dataset_audit(
            "PolypGen", args.polypgen_manifest, EXPECTED_POLYPGEN_ROWS
        )
    )

    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)

    schema_rows = []
    for item in results:
        schema_rows.append({
            "dataset": item["name"],
            "manifest_path": item["path"],
            "manifest_sha256": item["manifest_sha256"],
            "row_count": item["row_count"],
            "path_like_columns": "|".join(item["path_like_columns"]),
            "hash_like_columns": "|".join(item["hash_like_columns"]),
            "id_like_columns": "|".join(item["id_like_columns"]),
        })

    schema_csv = args.output_dir / "R14B4A_reference_manifest_schema_summary.csv"
    write_csv(
        schema_csv,
        schema_rows,
        [
            "dataset",
            "manifest_path",
            "manifest_sha256",
            "row_count",
            "path_like_columns",
            "hash_like_columns",
            "id_like_columns",
        ],
    )

    path_rows = []
    for item in results:
        for a in item["path_audits"]:
            row = {"dataset": item["name"], **a}
            path_rows.append(row)

    path_csv = args.output_dir / "R14B4A_reference_path_resolution_summary.csv"
    write_csv(
        path_csv,
        path_rows,
        [
            "dataset",
            "column",
            "nonempty",
            "absolute",
            "existing_as_is",
            "existing_files_as_is",
            "existing_dirs_as_is",
        ],
    )

    lock = {
        "status": "PASS",
        "decision": DECISION,
        "sun_candidate": {
            "manifest_path": str(args.sun_manifest),
            "manifest_sha256": sun_sha,
            "row_count": len(sun_rows),
            "physical_case_count": len(cases),
        },
        "reference_manifests": results,
        "information_boundary": {
            "path_root_guessing": False,
            "image_pixels_decoded": False,
            "gt_pixels_decoded": False,
            "contamination_comparison": False,
            "model_inference": False,
            "tta_execution": False,
            "target_outcomes_accessed": False,
        },
        "next_stage": (
            "R14B4B_SELECTED_SUN_RGB_EXACT_PIXEL_AND_REENCODING_"
            "CONTAMINATION_AUDIT"
        ),
    }

    lock_path = args.output_dir / "R14B4A_REFERENCE_ASSET_PREFLIGHT_LOCK.json"
    lock_path.write_text(
        json.dumps(lock, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nDecision=", DECISION)
    print("LOCK=", lock_path)
    print("PASS")


if __name__ == "__main__":
    main()
