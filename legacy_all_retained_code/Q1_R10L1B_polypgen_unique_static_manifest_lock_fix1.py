#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R10L1B_polypgen_unique_static_manifest_lock_fix1.py

Purpose
-------
R10L1 Fix2 recovered 1536 valid center-wise PolypGen RGB+GT pairs, with:
- 0 exact RGB duplicates versus NeoPolyp;
- 1534 unique PolypGen RGB hashes;
- 4 rows participating in exact within-PolypGen RGB duplicates;
- one unmatched image stem and one unmatched mask stem in C3.

Before source-training overlap audit or any external inference, freeze a
deduplicated static-image manifest.

Rules
-----
1) Recompute RGB and GT SHA256 from the 1536 paired rows.
2) For every exact duplicate RGB group:
   - if GT hashes differ -> STOP (conflicting annotation for identical RGB);
   - if GT hashes are identical -> keep exactly one deterministic
     representative, exclude the redundant copies.
3) Keep the already paired 1536 rows only.
   The unmatched C3 image/mask are NOT repaired or guessed here.
4) Require:
   - all six centers remain represented;
   - all retained RGB hashes unique;
   - no retained RGB exact duplicate of NeoPolyp;
   - all retained rows readable / same-size according to R10L1 Fix2.
5) No model inference, no Dice, no TTA, no safety scoring.

This produces the only PolypGen static manifest allowed to enter R10L2.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import hashlib
import json
import re

import pandas as pd
from tqdm import tqdm


DEFAULT_R10L1_MANIFEST = (
    "F:/MEDSEG_SAFETTA/outputs/"
    "Q1_R10L1_polypgen_official_singleframe_asset_audit_fix2_v1/"
    "R10L1_polypgen_official_singleframe_manifest.csv"
)

DEFAULT_R10L1_SUMMARY = (
    "F:/MEDSEG_SAFETTA/outputs/"
    "Q1_R10L1_polypgen_official_singleframe_asset_audit_fix2_v1/"
    "R10L1_summary.json"
)

DEFAULT_POLYPGEN_ROOT = (
    "F:/MEDSEG_SAFETTA/data/external/PolypGen/"
    "PolypGen2021_MultiCenterData_v3"
)

DEFAULT_OUTPUT = (
    "F:/MEDSEG_SAFETTA/outputs/"
    "Q1_R10L1B_polypgen_unique_static_manifest_lock_fix1_v1"
)

CENTERS = [f"C{i}" for i in range(1, 7)]
RASTER_EXTS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".tif", ".tiff", ".webp"
}


def sha256_file(path, chunk=8 * 1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def norm_stem(path):
    s = Path(path).stem.strip().lower()
    s = re.sub(
        r"(^|[_\-\s])"
        r"(image|img|mask|gt|segmentation|seg|label)"
        r"($|[_\-\s])",
        "_",
        s,
    )
    s = re.sub(r"[_\-\s]+", "_", s).strip("_")
    return s


def list_rasters(folder):
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in RASTER_EXTS
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--r10l1_manifest",
        default=DEFAULT_R10L1_MANIFEST,
    )
    ap.add_argument(
        "--r10l1_summary",
        default=DEFAULT_R10L1_SUMMARY,
    )
    ap.add_argument(
        "--polypgen_root",
        default=DEFAULT_POLYPGEN_ROOT,
    )
    ap.add_argument(
        "--output_dir",
        default=DEFAULT_OUTPUT,
    )
    args = ap.parse_args()

    manifest_path = Path(args.r10l1_manifest)
    summary_path = Path(args.r10l1_summary)
    root = Path(args.polypgen_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("===== R10L1B POLYPGEN UNIQUE STATIC MANIFEST LOCK FIX1 =====")
    print("STATUS: TECHNICAL DEDUPLICATION / MANIFEST FREEZE")
    print("MODEL INFERENCE: NONE")
    print("TTA: NONE")
    print("DICE/IOU: NONE")
    print("SAFETY SCORING: NONE")
    print("THRESHOLD TUNING: NONE")
    print("UNPAIRED C3 FILE REPAIR: NO")
    print("DUPLICATE POLICY: identical RGB + identical GT -> keep one")
    print()

    for name, p in {
        "R10L1 manifest": manifest_path,
        "R10L1 summary": summary_path,
        "PolypGen root": root,
    }.items():
        print(name, "exists:", p.exists(), p)
        if not p.exists():
            raise FileNotFoundError(p)

    with open(summary_path, "r", encoding="utf-8") as f:
        prev = json.load(f)

    expected = (
        "POLYPGEN_OFFICIAL_SINGLEFRAME_TECHNICALLY_READY_"
        "SOURCE_TRAINING_OVERLAP_UNRESOLVED"
    )
    if prev.get("decision") != expected:
        raise AssertionError(
            f"Unexpected R10L1 decision: {prev.get('decision')}"
        )

    df = pd.read_csv(manifest_path, low_memory=False)

    required = [
        "center",
        "sample_id",
        "image_path",
        "gt_path",
        "readable",
        "same_size",
        "exact_duplicate_of_neopolyp",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise AssertionError(f"Manifest missing columns: {missing}")

    print("\n===== INPUT MANIFEST AUDIT =====")
    print("Rows:", len(df))
    print("Centers:")
    print(df["center"].value_counts().sort_index().to_string())
    print("Readable:", int(df["readable"].astype(bool).sum()), "/", len(df))
    print("Same size:", int(df["same_size"].astype(bool).sum()), "/", len(df))
    print(
        "NeoPolyp exact RGB duplicates:",
        int(df["exact_duplicate_of_neopolyp"].astype(bool).sum()),
    )

    if len(df) != 1536:
        raise AssertionError(
            f"Expected the frozen 1536 paired rows, got {len(df)}."
        )
    if set(df["center"].astype(str)) != set(CENTERS):
        raise AssertionError("All six centers are not represented.")
    if not df["readable"].astype(bool).all():
        raise AssertionError("One or more paired rows are unreadable.")
    if not df["same_size"].astype(bool).all():
        raise AssertionError("One or more paired rows have size mismatch.")
    if df["exact_duplicate_of_neopolyp"].astype(bool).any():
        raise AssertionError("NeoPolyp exact contamination detected.")

    # ------------------------------------------------------------------
    # Recompute hashes from source files; do not trust cached values only.
    # ------------------------------------------------------------------
    rgb_hash = []
    gt_hash = []

    for r in tqdm(
        df.itertuples(index=False),
        total=len(df),
        desc="Rehashing PolypGen pairs",
        unit="pair",
        dynamic_ncols=True,
    ):
        ip = Path(r.image_path)
        gp = Path(r.gt_path)
        if not ip.exists():
            raise FileNotFoundError(ip)
        if not gp.exists():
            raise FileNotFoundError(gp)

        rgb_hash.append(sha256_file(ip))
        gt_hash.append(sha256_file(gp))

    df["rgb_sha256_recomputed"] = rgb_hash
    df["gt_sha256_recomputed"] = gt_hash

    print("\nUnique RGB hashes:", df["rgb_sha256_recomputed"].nunique())
    print("Unique GT hashes:", df["gt_sha256_recomputed"].nunique())

    # ------------------------------------------------------------------
    # Duplicate-group audit.
    # ------------------------------------------------------------------
    dup_group_rows = []
    exclude_rows = []
    keep_indices = []

    center_rank = {c: i for i, c in enumerate(CENTERS)}

    for rgb_sha, g in df.groupby("rgb_sha256_recomputed", sort=True):
        g = g.copy()

        if len(g) == 1:
            keep_indices.append(int(g.index[0]))
            continue

        gt_unique = g["gt_sha256_recomputed"].nunique()

        print("\nEXACT RGB DUPLICATE GROUP:", rgb_sha)
        print(
            g[
                [
                    "center",
                    "sample_id",
                    "image_path",
                    "gt_path",
                    "gt_sha256_recomputed",
                ]
            ].to_string(index=False)
        )
        print("rows:", len(g), "unique GT hashes:", gt_unique)

        for _, r in g.iterrows():
            dup_group_rows.append({
                "rgb_sha256": rgb_sha,
                "center": r["center"],
                "sample_id": r["sample_id"],
                "image_path": r["image_path"],
                "gt_path": r["gt_path"],
                "gt_sha256": r["gt_sha256_recomputed"],
                "group_rows": len(g),
                "group_unique_gt_hashes": gt_unique,
            })

        if gt_unique != 1:
            pd.DataFrame(dup_group_rows).to_csv(
                out / "R10L1B_duplicate_groups.csv",
                index=False,
            )
            raise AssertionError(
                "Identical RGB image has conflicting GT masks. "
                "Do not deduplicate automatically."
            )

        # Deterministic representative: center rank, then sample_id/path.
        tmp = g.copy()
        tmp["_center_rank"] = tmp["center"].map(center_rank)
        tmp = tmp.sort_values(
            ["_center_rank", "sample_id", "image_path"],
            kind="mergesort",
        )
        keep_idx = int(tmp.index[0])
        keep_indices.append(keep_idx)

        for idx in tmp.index[1:]:
            r = df.loc[idx]
            exclude_rows.append({
                "center": r["center"],
                "sample_id": r["sample_id"],
                "image_path": r["image_path"],
                "gt_path": r["gt_path"],
                "rgb_sha256": r["rgb_sha256_recomputed"],
                "gt_sha256": r["gt_sha256_recomputed"],
                "reason": "EXACT_DUPLICATE_RGB_AND_GT_REDUNDANT",
                "retained_representative_sample_id": df.loc[
                    keep_idx, "sample_id"
                ],
            })

    unique_df = (
        df.loc[sorted(keep_indices)]
        .copy()
        .reset_index(drop=True)
    )

    exclusions = pd.DataFrame(exclude_rows)
    duplicate_groups = pd.DataFrame(dup_group_rows)

    # ------------------------------------------------------------------
    # Re-audit the one known unpaired image/mask state in each center.
    # No repair is attempted.
    # ------------------------------------------------------------------
    unpaired_rows = []

    for center in CENTERS:
        cdir = root / f"data_{center}"
        idir = cdir / f"images_{center}"
        mdir = cdir / f"masks_{center}"

        if not idir.exists() or not mdir.exists():
            continue

        image_map = {}
        mask_map = {}

        for p in list_rasters(idir):
            image_map.setdefault(norm_stem(p), []).append(p)

        for p in list_rasters(mdir):
            mask_map.setdefault(norm_stem(p), []).append(p)

        image_only = sorted(set(image_map) - set(mask_map))
        mask_only = sorted(set(mask_map) - set(image_map))

        for k in image_only:
            for p in image_map[k]:
                unpaired_rows.append({
                    "center": center,
                    "role": "IMAGE_ONLY",
                    "normalized_stem": k,
                    "path": str(p),
                    "action": "EXCLUDED_NO_GUESSED_PAIRING",
                })

        for k in mask_only:
            for p in mask_map[k]:
                unpaired_rows.append({
                    "center": center,
                    "role": "MASK_ONLY",
                    "normalized_stem": k,
                    "path": str(p),
                    "action": "EXCLUDED_NO_GUESSED_PAIRING",
                })

    unpaired = pd.DataFrame(unpaired_rows)

    # ------------------------------------------------------------------
    # Final uniqueness / center audit.
    # ------------------------------------------------------------------
    print("\n===== DEDUPLICATED MANIFEST AUDIT =====")
    print("Input paired rows:", len(df))
    print("Redundant exact duplicate rows excluded:", len(exclusions))
    print("Retained unique rows:", len(unique_df))
    print("Retained unique RGB hashes:", unique_df["rgb_sha256_recomputed"].nunique())
    print("Centers after dedup:")
    print(unique_df["center"].value_counts().sort_index().to_string())

    print("\nUnpaired files intentionally excluded:", len(unpaired))
    if len(unpaired):
        print(unpaired.to_string(index=False))

    if unique_df["rgb_sha256_recomputed"].nunique() != len(unique_df):
        raise AssertionError("Retained manifest still contains duplicate RGB.")
    if set(unique_df["center"].astype(str)) != set(CENTERS):
        raise AssertionError("Deduplication removed an entire center.")
    if unique_df["exact_duplicate_of_neopolyp"].astype(bool).any():
        raise AssertionError("Retained manifest contains NeoPolyp duplicate.")

    # Current observed audit should yield 1534, but do not hide if assets differ.
    if len(unique_df) < 1500:
        raise AssertionError(
            "Unexpectedly large loss during exact deduplication."
        )

    # Rename to future external-validation-facing identifiers.
    unique_df = unique_df.rename(
        columns={
            "rgb_sha256_recomputed": "image_sha256",
            "gt_sha256_recomputed": "gt_sha256",
        }
    )
    unique_df["external_case_id"] = [
        f"polypgen_static_{i:04d}"
        for i in range(len(unique_df))
    ]

    # Put stable columns first.
    front = [
        "external_case_id",
        "center",
        "sample_id",
        "image_path",
        "gt_path",
        "image_sha256",
        "gt_sha256",
    ]
    rest = [c for c in unique_df.columns if c not in front]
    unique_df = unique_df[front + rest]

    manifest_out = out / "R10L1B_POLYPGEN_UNIQUE_STATIC_MANIFEST.csv"
    unique_df.to_csv(manifest_out, index=False)
    duplicate_groups.to_csv(
        out / "R10L1B_duplicate_groups.csv",
        index=False,
    )
    exclusions.to_csv(
        out / "R10L1B_exact_duplicate_exclusions.csv",
        index=False,
    )
    unpaired.to_csv(
        out / "R10L1B_unpaired_file_exclusions.csv",
        index=False,
    )

    decision = (
        "POLYPGEN_STATIC_UNIQUE_MANIFEST_LOCKED_"
        "PENDING_SOURCE_TRAINING_AUDIT"
    )

    summary = {
        "decision": decision,
        "input_paired_rows": len(df),
        "retained_unique_rows": len(unique_df),
        "retained_unique_rgb_hashes": int(
            unique_df["image_sha256"].nunique()
        ),
        "exact_duplicate_rows_excluded": len(exclusions),
        "unpaired_files_excluded": len(unpaired),
        "centers": sorted(unique_df["center"].unique().tolist()),
        "neoPolyp_exact_rgb_duplicates": 0,
        "conflicting_gt_for_duplicate_rgb": False,
        "model_inference_run": False,
        "performance_evaluated": False,
        "source_training_overlap_status": "PENDING_R10L2",
        "locked_manifest": str(manifest_out),
    }

    with open(
        out / "R10L1B_MANIFEST_LOCK.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n===== FINAL DECISION =====")
    print("DECISION:", decision)
    print("MODEL INFERENCE RUN: NO")
    print("PERFORMANCE EVALUATED: NO")
    print("SOURCE-TRAINING OVERLAP STATUS: PENDING_R10L2")
    print("\nOutputs:")
    print(manifest_out)
    print(out / "R10L1B_duplicate_groups.csv")
    print(out / "R10L1B_exact_duplicate_exclusions.csv")
    print(out / "R10L1B_unpaired_file_exclusions.csv")
    print(out / "R10L1B_MANIFEST_LOCK.json")
    print("PASS")


if __name__ == "__main__":
    main()
