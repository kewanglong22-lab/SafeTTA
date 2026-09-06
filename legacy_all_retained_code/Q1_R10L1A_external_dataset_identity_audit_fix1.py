#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R10L1A_external_dataset_identity_audit_fix1.py

Purpose
-------
The candidate "PolypGen" root currently contains exactly:
- 1200 likely RGB/image files
- 1000 likely mask files

Those counts are suspiciously compatible with the NeoPolyp challenge asset
shape, so DO NOT repair pairing or run any external model evaluation yet.

This audit determines whether the candidate root is:
1) an accidental copy/repackaging of the existing NeoPolyp assets;
2) a different dataset with overlapping filenames but different bytes;
3) technically unresolved.

NO inference.
NO Dice/IoU.
NO safety scoring.
NO threshold tuning.
NO external outcome inspection.

Primary evidence:
- exact SHA256 overlap against frozen NeoPolyp RGB images;
- exact SHA256 overlap against frozen NeoPolyp GT masks;
- filename/stem overlap;
- directory-structure inventory.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import hashlib
import json
import re

import pandas as pd
from tqdm import tqdm


DEFAULT_CANDIDATE_ROOT = (
    "F:/MEDSEG_SAFETTA/data/external/PolypGen"
)

DEFAULT_NEOPOLYP_MANIFEST = (
    "F:/MEDSEG_SAFETTA/outputs/"
    "Q1_R05D1_neopolyp_acquisition_pairing_duplicate_qc_v1_fix2/"
    "frozen_confirmatory_manifest.csv"
)

DEFAULT_OUTPUT = (
    "F:/MEDSEG_SAFETTA/outputs/"
    "Q1_R10L1A_external_dataset_identity_audit_fix1_v1"
)

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


def norm_name(path):
    return Path(path).stem.strip().lower()


def path_tokens(path):
    return [
        t for t in re.split(
            r"[^a-z0-9]+",
            str(path).lower().replace("\\", "/")
        )
        if t
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--candidate_root",
        default=DEFAULT_CANDIDATE_ROOT,
    )
    ap.add_argument(
        "--neopolyp_manifest",
        default=DEFAULT_NEOPOLYP_MANIFEST,
    )
    ap.add_argument(
        "--output_dir",
        default=DEFAULT_OUTPUT,
    )
    args = ap.parse_args()

    root = Path(args.candidate_root)
    neo_manifest_path = Path(args.neopolyp_manifest)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("===== R10L1A EXTERNAL DATASET IDENTITY AUDIT FIX1 =====")
    print("STATUS: DATASET-IDENTITY / CONTAMINATION AUDIT")
    print("MODEL INFERENCE: NONE")
    print("GT PERFORMANCE METRICS: NONE")
    print("SAFETY SCORING: NONE")
    print("METHOD TUNING: NONE")
    print("CANDIDATE ROOT:", root)
    print()

    print("candidate root exists:", root.exists(), root)
    print(
        "NeoPolyp manifest exists:",
        neo_manifest_path.exists(),
        neo_manifest_path,
    )

    if not root.exists():
        raise FileNotFoundError(root)
    if not neo_manifest_path.exists():
        raise FileNotFoundError(neo_manifest_path)

    candidate_files = sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in RASTER_EXTS
    )

    print("\nCandidate raster files:", len(candidate_files))

    # ------------------------------------------------------------------
    # Directory-structure inventory.
    # ------------------------------------------------------------------
    top_counts = Counter()
    second_counts = Counter()
    ext_counts = Counter()
    token_counts = Counter()

    for p in candidate_files:
        rel = p.relative_to(root)
        if len(rel.parts) >= 1:
            top_counts[rel.parts[0]] += 1
        if len(rel.parts) >= 2:
            second_counts[
                f"{rel.parts[0]}/{rel.parts[1]}"
            ] += 1
        ext_counts[p.suffix.lower()] += 1
        token_counts.update(path_tokens(rel))

    print("\nTop-level raster counts:")
    for k, v in top_counts.most_common(30):
        print(f"  {k}: {v}")

    print("\nSecond-level raster counts:")
    for k, v in second_counts.most_common(50):
        print(f"  {k}: {v}")

    print("\nExtensions:")
    for k, v in ext_counts.most_common():
        print(f"  {k}: {v}")

    print("\nMost common path tokens:")
    for k, v in token_counts.most_common(30):
        print(f"  {k}: {v}")

    print("\nFirst 40 candidate relative paths:")
    for p in candidate_files[:40]:
        print(" ", p.relative_to(root))

    # ------------------------------------------------------------------
    # Frozen NeoPolyp reference assets.
    # ------------------------------------------------------------------
    neo = pd.read_csv(
        neo_manifest_path,
        low_memory=False,
    )

    required = ["image_path", "gt_path"]
    missing = [c for c in required if c not in neo.columns]
    if missing:
        raise AssertionError(
            f"NeoPolyp manifest missing columns: {missing}"
        )

    neo_rgb_paths = [
        Path(str(p))
        for p in neo["image_path"].tolist()
        if Path(str(p)).exists()
    ]
    neo_gt_paths = [
        Path(str(p))
        for p in neo["gt_path"].tolist()
        if Path(str(p)).exists()
    ]

    print("\nNeoPolyp frozen RGB paths existing:", len(neo_rgb_paths))
    print("NeoPolyp frozen GT paths existing:", len(neo_gt_paths))

    if len(neo_rgb_paths) != 1000:
        raise AssertionError(
            f"Expected 1000 frozen NeoPolyp RGB, got {len(neo_rgb_paths)}"
        )
    if len(neo_gt_paths) != 1000:
        raise AssertionError(
            f"Expected 1000 frozen NeoPolyp GT, got {len(neo_gt_paths)}"
        )

    # ------------------------------------------------------------------
    # Hash NeoPolyp references.
    # ------------------------------------------------------------------
    neo_rgb_hash_to_path = {}
    neo_gt_hash_to_path = {}

    for p in tqdm(
        neo_rgb_paths,
        desc="Hashing frozen NeoPolyp RGB",
        unit="file",
        dynamic_ncols=True,
    ):
        neo_rgb_hash_to_path[sha256_file(p)] = str(p)

    for p in tqdm(
        neo_gt_paths,
        desc="Hashing frozen NeoPolyp GT",
        unit="file",
        dynamic_ncols=True,
    ):
        neo_gt_hash_to_path[sha256_file(p)] = str(p)

    neo_rgb_hashes = set(neo_rgb_hash_to_path)
    neo_gt_hashes = set(neo_gt_hash_to_path)

    neo_rgb_stems = set(norm_name(p) for p in neo_rgb_paths)
    neo_gt_stems = set(norm_name(p) for p in neo_gt_paths)

    # ------------------------------------------------------------------
    # Candidate hash audit.
    # ------------------------------------------------------------------
    rows = []
    rgb_exact = 0
    gt_exact = 0
    any_neo_exact = 0
    rgb_stem_hits = 0
    gt_stem_hits = 0

    for p in tqdm(
        candidate_files,
        desc="Hashing candidate raster files",
        unit="file",
        dynamic_ncols=True,
    ):
        h = sha256_file(p)
        stem = norm_name(p)

        is_rgb = h in neo_rgb_hashes
        is_gt = h in neo_gt_hashes
        rgb_stem = stem in neo_rgb_stems
        gt_stem = stem in neo_gt_stems

        rgb_exact += int(is_rgb)
        gt_exact += int(is_gt)
        any_neo_exact += int(is_rgb or is_gt)
        rgb_stem_hits += int(rgb_stem)
        gt_stem_hits += int(gt_stem)

        rows.append({
            "candidate_path": str(p),
            "relative_path": str(p.relative_to(root)),
            "sha256": h,
            "exact_neopolyp_rgb": is_rgb,
            "exact_neopolyp_gt": is_gt,
            "same_stem_as_neopolyp_rgb": rgb_stem,
            "same_stem_as_neopolyp_gt": gt_stem,
            "matched_neopolyp_rgb_path": (
                neo_rgb_hash_to_path[h] if is_rgb else ""
            ),
            "matched_neopolyp_gt_path": (
                neo_gt_hash_to_path[h] if is_gt else ""
            ),
        })

    audit = pd.DataFrame(rows)

    print("\n===== EXACT OVERLAP SUMMARY =====")
    print("Candidate raster files:", len(candidate_files))
    print("Exact NeoPolyp RGB matches:", rgb_exact)
    print("Exact NeoPolyp GT matches:", gt_exact)
    print("Any exact NeoPolyp asset matches:", any_neo_exact)
    print("Same-stem hits vs NeoPolyp RGB:", rgb_stem_hits)
    print("Same-stem hits vs NeoPolyp GT:", gt_stem_hits)

    # ------------------------------------------------------------------
    # Conservative identity decision.
    # ------------------------------------------------------------------
    # 1000 exact RGB + 1000 exact GT is essentially definitive for this
    # development corpus. Lower thresholds are intentionally not used to
    # label the whole candidate dataset as a wrong copy.
    if rgb_exact >= 950 and gt_exact >= 950:
        decision = "CANDIDATE_IS_NEOPOLYP_COPY_OR_REPACKAGING"
    elif any_neo_exact > 0:
        decision = "CANDIDATE_HAS_NEOPOLYP_EXACT_CONTAMINATION"
    else:
        decision = "NO_EXACT_NEOPOLYP_OVERLAP_DATASET_IDENTITY_STILL_UNRESOLVED"

    print("\n===== FINAL DECISION =====")
    print("DECISION:", decision)

    if decision == "CANDIDATE_IS_NEOPOLYP_COPY_OR_REPACKAGING":
        print(
            "ACTION: DO NOT use this root as independent external validation."
        )
    elif decision == "CANDIDATE_HAS_NEOPOLYP_EXACT_CONTAMINATION":
        print(
            "ACTION: STOP external validation until contamination is resolved."
        )
    else:
        print(
            "ACTION: identity is not proven by hashes; inspect structure/source "
            "before repairing image-mask pairing."
        )

    audit.to_csv(
        out / "R10L1A_candidate_vs_neopolyp_hash_audit.csv",
        index=False,
    )

    summary = {
        "decision": decision,
        "candidate_root": str(root),
        "candidate_raster_files": len(candidate_files),
        "exact_neopolyp_rgb_matches": rgb_exact,
        "exact_neopolyp_gt_matches": gt_exact,
        "any_exact_neopolyp_matches": any_neo_exact,
        "same_stem_hits_vs_neopolyp_rgb": rgb_stem_hits,
        "same_stem_hits_vs_neopolyp_gt": gt_stem_hits,
        "top_level_counts": dict(top_counts),
        "second_level_counts": dict(second_counts),
        "model_inference_run": False,
        "performance_metrics_computed": False,
    }

    with open(
        out / "R10L1A_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nOutputs:")
    print(out / "R10L1A_candidate_vs_neopolyp_hash_audit.csv")
    print(out / "R10L1A_summary.json")
    print("PASS")


if __name__ == "__main__":
    main()
