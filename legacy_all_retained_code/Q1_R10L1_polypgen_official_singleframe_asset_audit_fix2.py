#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R10L1_polypgen_official_singleframe_asset_audit_fix2.py

Frozen technical audit for the OFFICIAL PolypGen 2021 MultiCenterData_v3
single-frame cohort.

Expected extracted root:
F:/MEDSEG_SAFETTA/data/external/PolypGen/PolypGen2021_MultiCenterData_v3

Use ONLY:
    data_C1 ... data_C6

Explicitly EXCLUDE:
    imagesAll_positive
    sequenceData
    codes
    bbox / bbox_image visualization assets

Why:
- data_C1...data_C6 are the center-stratified static-image component needed for
  a clean multi-center independent-cohort validation protocol.
- imagesAll_positive may aggregate/duplicate images.
- sequenceData is a different temporal data regime and is not mixed into this
  first confirmatory external validation.

Audit only:
- no segmentation inference
- no TTA
- no safety scoring
- no Dice/IoU/AUROC/AUPRC
- no threshold tuning
- no method tuning

Technical gates:
1) recover image/mask folders under each center;
2) pair by exact normalized filename stem within center;
3) verify readability and image-mask size equality;
4) SHA256 candidate RGB images;
5) SHA256 frozen NeoPolyp development RGB images;
6) require zero exact RGB duplicates with NeoPolyp;
7) write center-aware locked manifest.

Source-training overlap remains a separate R10L2 gate.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import hashlib
import json
import re

import pandas as pd
from PIL import Image
from tqdm import tqdm


DEFAULT_ROOT = (
    "F:/MEDSEG_SAFETTA/data/external/PolypGen/"
    "PolypGen2021_MultiCenterData_v3"
)

DEFAULT_NEOPOLYP_MANIFEST = (
    "F:/MEDSEG_SAFETTA/outputs/"
    "Q1_R05D1_neopolyp_acquisition_pairing_duplicate_qc_v1_fix2/"
    "frozen_confirmatory_manifest.csv"
)

DEFAULT_R10L0_LOCK = (
    "F:/MEDSEG_SAFETTA/outputs/"
    "Q1_R10L0_final_source_safety_estimator_lock_fix3_v1/"
    "R10L0_FINAL_SOURCE_ESTIMATOR_LOCK.json"
)

DEFAULT_OUTPUT = (
    "F:/MEDSEG_SAFETTA/outputs/"
    "Q1_R10L1_polypgen_official_singleframe_asset_audit_fix2_v1"
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

    # Remove only common semantic suffixes/prefixes, conservatively.
    s = re.sub(
        r"(^|[_\-\s])"
        r"(image|img|mask|gt|segmentation|seg|label)"
        r"($|[_\-\s])",
        "_",
        s,
    )
    s = re.sub(r"[_\-\s]+", "_", s).strip("_")
    return s


def image_meta(path):
    with Image.open(path) as im:
        w, h = im.size
        mode = im.mode
    return w, h, mode


def find_role_dirs(center_dir, center, role):
    """
    Official layout is typically images_Cx / masks_Cx.
    We still search recursively inside data_Cx to tolerate one extra wrapper.
    """
    center_l = center.lower()
    hits = []

    for p in center_dir.rglob("*"):
        if not p.is_dir():
            continue

        name = p.name.lower()
        if role == "image":
            if (
                name == f"images_{center_l}"
                or name == f"images{center_l}"
                or (
                    "image" in name
                    and "bbox" not in name
                    and center_l in name
                )
            ):
                hits.append(p)

        elif role == "mask":
            if (
                name == f"masks_{center_l}"
                or name == f"masks{center_l}"
                or (
                    "mask" in name
                    and center_l in name
                )
            ):
                hits.append(p)

    # Prefer shallowest matching directory.
    hits = sorted(
        set(hits),
        key=lambda x: (
            len(x.relative_to(center_dir).parts),
            str(x).lower(),
        ),
    )
    return hits


def list_rasters(folder):
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in RASTER_EXTS
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--polypgen_root", default=DEFAULT_ROOT)
    ap.add_argument(
        "--neopolyp_manifest",
        default=DEFAULT_NEOPOLYP_MANIFEST,
    )
    ap.add_argument(
        "--r10l0_lock",
        default=DEFAULT_R10L0_LOCK,
    )
    ap.add_argument(
        "--output_dir",
        default=DEFAULT_OUTPUT,
    )
    args = ap.parse_args()

    root = Path(args.polypgen_root)
    neo_manifest_path = Path(args.neopolyp_manifest)
    r10l0_lock_path = Path(args.r10l0_lock)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("===== R10L1 POLYPGEN OFFICIAL SINGLE-FRAME ASSET AUDIT FIX2 =====")
    print("STATUS: PROSPECTIVE TECHNICAL ASSET AUDIT")
    print("OFFICIAL COMPONENT: data_C1 ... data_C6 ONLY")
    print("imagesAll_positive USED: NO")
    print("sequenceData USED: NO")
    print("MODEL INFERENCE: NONE")
    print("TTA: NONE")
    print("SAFETY SCORING: NONE")
    print("GT PERFORMANCE METRICS: NONE")
    print("METHOD TUNING: NONE")
    print("THRESHOLD TUNING: NONE")
    print("SOURCE ESTIMATOR REFIT: NONE")
    print("ROOT:", root)
    print()

    print("root exists:", root.exists(), root)
    print("NeoPolyp manifest exists:", neo_manifest_path.exists())
    print("R10L0 lock exists:", r10l0_lock_path.exists())

    if not root.exists():
        raise FileNotFoundError(
            f"Extract the archive first. Expected root: {root}"
        )
    if not neo_manifest_path.exists():
        raise FileNotFoundError(neo_manifest_path)
    if not r10l0_lock_path.exists():
        raise FileNotFoundError(r10l0_lock_path)

    with open(r10l0_lock_path, "r", encoding="utf-8") as f:
        lock = json.load(f)

    expected_lock = (
        "FINAL_SOURCE_ESTIMATOR_LOCKED_FOR_INDEPENDENT_EXTERNAL_VALIDATION"
    )
    if lock.get("decision") != expected_lock:
        raise AssertionError("R10L0 source estimator lock is not frozen.")

    # ------------------------------------------------------------------
    # Center-wise folder and filename pairing audit.
    # ------------------------------------------------------------------
    pair_rows = []
    center_rows = []
    problems = []

    for center in CENTERS:
        cdir = root / f"data_{center}"
        print("\n------------------------------------------------------------")
        print("CENTER:", center)
        print("center dir exists:", cdir.exists(), cdir)

        if not cdir.exists():
            problems.append(f"{center}: data directory missing")
            center_rows.append({
                "center": center,
                "image_dir": "",
                "mask_dir": "",
                "image_files": 0,
                "mask_files": 0,
                "paired": 0,
                "image_only": 0,
                "mask_only": 0,
            })
            continue

        image_dirs = find_role_dirs(cdir, center, "image")
        mask_dirs = find_role_dirs(cdir, center, "mask")

        print("image-dir candidates:")
        for x in image_dirs:
            print(" ", x)
        print("mask-dir candidates:")
        for x in mask_dirs:
            print(" ", x)

        if len(image_dirs) != 1:
            problems.append(
                f"{center}: expected one image dir, got {len(image_dirs)}"
            )
            continue
        if len(mask_dirs) != 1:
            problems.append(
                f"{center}: expected one mask dir, got {len(mask_dirs)}"
            )
            continue

        image_dir = image_dirs[0]
        mask_dir = mask_dirs[0]

        images = list_rasters(image_dir)
        masks = list_rasters(mask_dir)

        image_map = {}
        mask_map = {}

        for p in images:
            k = norm_stem(p)
            image_map.setdefault(k, []).append(p)

        for p in masks:
            k = norm_stem(p)
            mask_map.setdefault(k, []).append(p)

        duplicate_image_keys = {
            k: v for k, v in image_map.items() if len(v) != 1
        }
        duplicate_mask_keys = {
            k: v for k, v in mask_map.items() if len(v) != 1
        }

        if duplicate_image_keys:
            problems.append(
                f"{center}: duplicate normalized image stems="
                f"{len(duplicate_image_keys)}"
            )
        if duplicate_mask_keys:
            problems.append(
                f"{center}: duplicate normalized mask stems="
                f"{len(duplicate_mask_keys)}"
            )

        image_keys = set(image_map)
        mask_keys = set(mask_map)
        common = sorted(image_keys & mask_keys)
        image_only = sorted(image_keys - mask_keys)
        mask_only = sorted(mask_keys - image_keys)

        print("image files:", len(images))
        print("mask files:", len(masks))
        print("paired stems:", len(common))
        print("image-only stems:", len(image_only))
        print("mask-only stems:", len(mask_only))

        center_rows.append({
            "center": center,
            "image_dir": str(image_dir),
            "mask_dir": str(mask_dir),
            "image_files": len(images),
            "mask_files": len(masks),
            "paired": len(common),
            "image_only": len(image_only),
            "mask_only": len(mask_only),
        })

        for k in common:
            if len(image_map[k]) != 1 or len(mask_map[k]) != 1:
                continue
            pair_rows.append({
                "center": center,
                "sample_id": f"polypgen_{center.lower()}_{k}",
                "pair_key": k,
                "image_path": str(image_map[k][0]),
                "gt_path": str(mask_map[k][0]),
            })

    center_df = pd.DataFrame(center_rows)
    pairs = pd.DataFrame(pair_rows)

    print("\n===== CENTER SUMMARY =====")
    print(center_df.to_string(index=False))

    if problems:
        print("\nFolder/pairing problems:")
        for x in problems:
            print(" -", x)

    print("\nTotal recovered pairs:", len(pairs))

    if len(pairs) == 0:
        center_df.to_csv(
            out / "R10L1_center_structure_audit.csv",
            index=False,
        )
        raise AssertionError("No official center-wise image-mask pairs recovered.")

    # ------------------------------------------------------------------
    # Readability and geometry.
    # ------------------------------------------------------------------
    audit_rows = []

    for r in tqdm(
        pairs.itertuples(index=False),
        total=len(pairs),
        desc="Auditing PolypGen pairs",
        unit="pair",
        dynamic_ncols=True,
    ):
        ip = Path(r.image_path)
        gp = Path(r.gt_path)

        try:
            iw, ih, imode = image_meta(ip)
            gw, gh, gmode = image_meta(gp)
            readable = True
            err = ""
        except Exception as e:
            iw = ih = gw = gh = -1
            imode = gmode = ""
            readable = False
            err = repr(e)

        audit_rows.append({
            "center": r.center,
            "sample_id": r.sample_id,
            "pair_key": r.pair_key,
            "image_path": r.image_path,
            "gt_path": r.gt_path,
            "readable": readable,
            "image_width": iw,
            "image_height": ih,
            "image_mode": imode,
            "gt_width": gw,
            "gt_height": gh,
            "gt_mode": gmode,
            "same_size": readable and iw == gw and ih == gh,
            "error": err,
        })

    audit = pd.DataFrame(audit_rows)

    readable_n = int(audit["readable"].sum())
    same_size_n = int(audit["same_size"].sum())

    print("\nReadable pairs:", readable_n, "/", len(audit))
    print("Same-size pairs:", same_size_n, "/", len(audit))

    # ------------------------------------------------------------------
    # Exact RGB overlap against frozen NeoPolyp development cohort.
    # ------------------------------------------------------------------
    neo = pd.read_csv(neo_manifest_path, low_memory=False)
    if "image_path" not in neo.columns:
        raise AssertionError("NeoPolyp manifest missing image_path.")

    neo_paths = [
        Path(str(p))
        for p in neo["image_path"].tolist()
        if Path(str(p)).exists()
    ]

    if len(neo_paths) != 1000:
        raise AssertionError(
            f"Expected 1000 existing NeoPolyp RGB images, got {len(neo_paths)}"
        )

    neo_hashes = set()
    for p in tqdm(
        neo_paths,
        desc="Hashing NeoPolyp RGB",
        unit="image",
        dynamic_ncols=True,
    ):
        neo_hashes.add(sha256_file(p))

    candidate_hashes = []
    exact_dups = 0

    for p in tqdm(
        audit["image_path"].map(Path).tolist(),
        desc="Hashing PolypGen RGB",
        unit="image",
        dynamic_ncols=True,
    ):
        h = sha256_file(p)
        dup = h in neo_hashes
        candidate_hashes.append(h)
        exact_dups += int(dup)

    audit["image_sha256"] = candidate_hashes
    audit["exact_duplicate_of_neopolyp"] = [
        h in neo_hashes for h in candidate_hashes
    ]

    print("Exact RGB duplicates with NeoPolyp:", exact_dups)

    # ------------------------------------------------------------------
    # Duplicate check WITHIN PolypGen static cohort.
    # ------------------------------------------------------------------
    dup_counts = pd.Series(candidate_hashes).value_counts()
    duplicated_hashes = set(
        dup_counts[dup_counts > 1].index.tolist()
    )
    audit["duplicate_within_polypgen_static"] = audit[
        "image_sha256"
    ].isin(duplicated_hashes)

    within_duplicate_rows = int(
        audit["duplicate_within_polypgen_static"].sum()
    )
    unique_rgb_hashes = int(audit["image_sha256"].nunique())

    print("Unique PolypGen RGB hashes:", unique_rgb_hashes)
    print(
        "Rows participating in exact within-PolypGen duplicates:",
        within_duplicate_rows,
    )

    # ------------------------------------------------------------------
    # Conservative technical decision.
    # ------------------------------------------------------------------
    centers_present = set(audit["center"].unique())
    all_centers = centers_present == set(CENTERS)

    technical_ready = (
        len(audit) >= 1000
        and all_centers
        and readable_n == len(audit)
        and same_size_n == len(audit)
        and exact_dups == 0
    )

    if technical_ready:
        decision = (
            "POLYPGEN_OFFICIAL_SINGLEFRAME_TECHNICALLY_READY_"
            "SOURCE_TRAINING_OVERLAP_UNRESOLVED"
        )
    else:
        decision = "POLYPGEN_OFFICIAL_SINGLEFRAME_TECHNICAL_AUDIT_NOT_READY"

    center_df.to_csv(
        out / "R10L1_center_structure_audit.csv",
        index=False,
    )
    audit.to_csv(
        out / "R10L1_polypgen_official_singleframe_manifest.csv",
        index=False,
    )

    summary = {
        "decision": decision,
        "official_component": "data_C1...data_C6",
        "imagesAll_positive_used": False,
        "sequenceData_used": False,
        "centers": sorted(centers_present),
        "pair_rows": len(audit),
        "readable_pairs": readable_n,
        "same_size_pairs": same_size_n,
        "unique_rgb_hashes": unique_rgb_hashes,
        "within_polypgen_duplicate_rows": within_duplicate_rows,
        "exact_neopolyp_rgb_duplicates": exact_dups,
        "source_training_overlap_status": "UNRESOLVED",
        "model_inference_run": False,
        "external_performance_evaluated": False,
    }

    with open(
        out / "R10L1_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n===== FINAL DECISION =====")
    print("DECISION:", decision)
    print("EXTERNAL PERFORMANCE EVALUATED: NO")
    print("SOURCE-TRAINING OVERLAP STATUS: UNRESOLVED")
    print("\nOutputs:")
    print(out / "R10L1_center_structure_audit.csv")
    print(out / "R10L1_polypgen_official_singleframe_manifest.csv")
    print(out / "R10L1_summary.json")
    print("PASS")


if __name__ == "__main__":
    main()
