#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R10L1_polypgen_external_cohort_asset_audit_fix1.py

Prospective asset audit for an independent PolypGen validation cohort.

NO MODEL INFERENCE.
NO SAFETY SCORING.
NO TARGET/EXTERNAL OUTCOME INSPECTION.
NO METHOD TUNING.

The script:
1) recursively inventories the candidate PolypGen root;
2) identifies likely RGB images and GT masks using path/name hints;
3) pairs image/mask files by normalized relative stem;
4) validates paired files are readable and size-compatible;
5) computes SHA256 of candidate RGB images and frozen NeoPolyp development
   images to detect exact duplicate images;
6) writes a locked candidate-pair manifest.

Important:
This audit can establish technical pairing + no exact NeoPolyp duplication.
It CANNOT by itself prove that PolypGen was absent from the original source
segmentation-model training data. That remains a separate lineage gate.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import hashlib
import json
import re

import pandas as pd
from PIL import Image
from tqdm import tqdm


DEFAULT_POLYPGEN_ROOT = (
    "F:/MEDSEG_SAFETTA/data/external/PolypGen"
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
    "Q1_R10L1_polypgen_external_cohort_asset_audit_fix1_v1"
)

IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".tif", ".tiff", ".webp"
}

MASK_HINTS = {
    "mask", "masks", "gt", "groundtruth", "ground_truth",
    "annotation", "annotations", "label", "labels",
    "seg", "segmentation"
}

IMAGE_HINTS = {
    "image", "images", "img", "imgs", "frame", "frames",
    "original", "originals", "rgb"
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


def tokenize_path(path):
    parts = []
    for p in path.parts:
        parts.extend(
            t for t in re.split(r"[^a-z0-9]+", p.lower())
            if t
        )
    return set(parts)


def classify_file(path):
    """
    Conservative heuristic:
    - mask if path/name has mask/gt/annotation/label hints;
    - image otherwise.
    """
    tokens = tokenize_path(path)
    stem = path.stem.lower()

    mask_score = sum(h in tokens or h in stem for h in MASK_HINTS)
    image_score = sum(h in tokens or h in stem for h in IMAGE_HINTS)

    if mask_score > image_score and mask_score > 0:
        return "mask"
    return "image"


def canonical_key(path, root):
    """
    Pairing key independent of common image/mask directory and suffix words.
    """
    rel = path.relative_to(root)
    stem = path.stem.lower()

    # Remove common mask/image suffixes/prefixes.
    stem = re.sub(
        r"(^|[_\-\s])"
        r"(mask|masks|gt|groundtruth|ground_truth|annotation|annotations|"
        r"label|labels|seg|segmentation|image|images|img|frame|frames)"
        r"($|[_\-\s])",
        "_",
        stem,
    )
    stem = re.sub(r"[_\-\s]+", "_", stem).strip("_")

    # Keep contextual directories, but drop obvious role directories.
    dirs = []
    for d in rel.parts[:-1]:
        dl = d.lower()
        toks = set(
            t for t in re.split(r"[^a-z0-9]+", dl)
            if t
        )
        if toks & MASK_HINTS:
            continue
        if toks & IMAGE_HINTS:
            continue
        dirs.append(dl)

    return "/".join(dirs + [stem])


def image_meta(path):
    with Image.open(path) as im:
        w, h = im.size
        mode = im.mode
    return w, h, mode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--polypgen_root",
        default=DEFAULT_POLYPGEN_ROOT,
    )
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

    print("===== R10L1 POLYPGEN EXTERNAL COHORT ASSET AUDIT FIX1 =====")
    print("STATUS: PROSPECTIVE EXTERNAL-ASSET AUDIT")
    print("MODEL INFERENCE: NONE")
    print("SAFETY SCORING: NONE")
    print("EXTERNAL GT USED FOR PERFORMANCE: NO")
    print("METHOD TUNING: NONE")
    print("THRESHOLD TUNING: NONE")
    print("SOURCE ESTIMATOR REFIT: NONE")
    print("CANDIDATE ROOT:", root)
    print()

    print("PolypGen root exists:", root.exists(), root)
    print("NeoPolyp manifest exists:", neo_manifest_path.exists())
    print("R10L0 source lock exists:", r10l0_lock_path.exists())

    if not r10l0_lock_path.exists():
        raise FileNotFoundError(r10l0_lock_path)

    with open(r10l0_lock_path, "r", encoding="utf-8") as f:
        lock = json.load(f)

    expected = (
        "FINAL_SOURCE_ESTIMATOR_LOCKED_FOR_INDEPENDENT_EXTERNAL_VALIDATION"
    )
    if lock.get("decision") != expected:
        raise AssertionError(
            "R10L0 source estimator is not in the expected frozen state."
        )

    if not root.exists():
        print("\n===== FINAL DECISION =====")
        print("DECISION: POLYPGEN_ASSET_MISSING")
        print(
            "Place/extract PolypGen under:",
            root,
        )
        print("PASS (AUDIT STOP: NO EXTERNAL DATA INSPECTED)")
        return

    if not neo_manifest_path.exists():
        raise FileNotFoundError(neo_manifest_path)

    # --------------------------------------------------------------
    # Inventory candidate image files.
    # --------------------------------------------------------------
    files = [
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]

    print("\nCandidate raster files:", len(files))
    if not files:
        raise AssertionError("No image-like files found under PolypGen root.")

    classified = []
    for p in files:
        role = classify_file(p)
        classified.append({
            "path": str(p),
            "role": role,
            "key": canonical_key(p, root),
        })

    inv = pd.DataFrame(classified)

    print("Likely RGB/image files:", int((inv["role"] == "image").sum()))
    print("Likely mask files:", int((inv["role"] == "mask").sum()))

    image_groups = defaultdict(list)
    mask_groups = defaultdict(list)

    for r in inv.itertuples(index=False):
        if r.role == "image":
            image_groups[r.key].append(Path(r.path))
        else:
            mask_groups[r.key].append(Path(r.path))

    all_keys = sorted(set(image_groups) | set(mask_groups))

    pair_rows = []

    for key in all_keys:
        imgs = image_groups.get(key, [])
        masks = mask_groups.get(key, [])

        pair_rows.append({
            "pair_key": key,
            "image_count": len(imgs),
            "mask_count": len(masks),
            "image_path": str(imgs[0]) if len(imgs) == 1 else "",
            "mask_path": str(masks[0]) if len(masks) == 1 else "",
            "unique_pair": len(imgs) == 1 and len(masks) == 1,
        })

    pairs = pd.DataFrame(pair_rows)

    unique_pairs = pairs[pairs["unique_pair"]].copy()
    ambiguous = pairs[~pairs["unique_pair"]].copy()

    print("\nUnique image-mask pairs:", len(unique_pairs))
    print("Ambiguous/unpaired keys:", len(ambiguous))

    if len(unique_pairs) == 0:
        inv.to_csv(out / "R10L1_file_inventory.csv", index=False)
        ambiguous.to_csv(
            out / "R10L1_ambiguous_or_unpaired_keys.csv",
            index=False,
        )
        print("\n===== FINAL DECISION =====")
        print("DECISION: POLYPGEN_PAIRING_NOT_RECOVERED")
        print("PASS")
        return

    # --------------------------------------------------------------
    # Readability / geometry audit.
    # --------------------------------------------------------------
    audit_rows = []

    for r in tqdm(
        unique_pairs.itertuples(index=False),
        total=len(unique_pairs),
        desc="Auditing image-mask pairs",
        unit="pair",
        dynamic_ncols=True,
    ):
        ip = Path(r.image_path)
        mp = Path(r.mask_path)

        try:
            iw, ih, imode = image_meta(ip)
            mw, mh, mmode = image_meta(mp)
            readable = True
            err = ""
        except Exception as e:
            iw = ih = mw = mh = -1
            imode = mmode = ""
            readable = False
            err = repr(e)

        audit_rows.append({
            "pair_key": r.pair_key,
            "image_path": str(ip),
            "mask_path": str(mp),
            "readable": readable,
            "image_width": iw,
            "image_height": ih,
            "image_mode": imode,
            "mask_width": mw,
            "mask_height": mh,
            "mask_mode": mmode,
            "same_size": readable and iw == mw and ih == mh,
            "error": err,
        })

    audit = pd.DataFrame(audit_rows)

    readable_pairs = int(audit["readable"].sum())
    same_size_pairs = int(audit["same_size"].sum())

    print("Readable pairs:", readable_pairs, "/", len(audit))
    print("Same-size image/mask pairs:", same_size_pairs, "/", len(audit))

    # --------------------------------------------------------------
    # Exact duplicate audit against NeoPolyp development RGB.
    # --------------------------------------------------------------
    neo = pd.read_csv(neo_manifest_path, low_memory=False)
    if "image_path" not in neo.columns:
        raise AssertionError("NeoPolyp manifest missing image_path.")

    neo_paths = [
        Path(str(p))
        for p in neo["image_path"].tolist()
        if Path(str(p)).exists()
    ]

    print("\nNeoPolyp existing development images:", len(neo_paths))

    neo_hashes = set()
    for p in tqdm(
        neo_paths,
        desc="Hashing NeoPolyp",
        unit="image",
        dynamic_ncols=True,
    ):
        neo_hashes.add(sha256_file(p))

    candidate_hashes = []
    duplicate_count = 0

    for p in tqdm(
        audit["image_path"].map(Path).tolist(),
        desc="Hashing PolypGen candidates",
        unit="image",
        dynamic_ncols=True,
    ):
        h = sha256_file(p)
        dup = h in neo_hashes
        duplicate_count += int(dup)
        candidate_hashes.append((h, dup))

    audit["image_sha256"] = [x[0] for x in candidate_hashes]
    audit["exact_duplicate_of_neopolyp"] = [
        x[1] for x in candidate_hashes
    ]

    print("Exact RGB duplicates with NeoPolyp:", duplicate_count)

    # --------------------------------------------------------------
    # Path-bucket descriptive audit; no performance use.
    # --------------------------------------------------------------
    def top_bucket(p):
        rel = Path(p).relative_to(root)
        return rel.parts[0] if len(rel.parts) else "ROOT"

    audit["top_level_bucket"] = audit["image_path"].map(top_bucket)

    print("\nTop-level pair counts:")
    print(
        audit["top_level_bucket"]
        .value_counts()
        .head(30)
        .to_string()
    )

    # --------------------------------------------------------------
    # Outputs and conservative decision.
    # --------------------------------------------------------------
    inv.to_csv(out / "R10L1_file_inventory.csv", index=False)
    ambiguous.to_csv(
        out / "R10L1_ambiguous_or_unpaired_keys.csv",
        index=False,
    )
    audit.to_csv(
        out / "R10L1_polypgen_candidate_pair_manifest.csv",
        index=False,
    )

    technical_ready = (
        len(audit) >= 100
        and readable_pairs == len(audit)
        and same_size_pairs == len(audit)
        and duplicate_count == 0
    )

    if technical_ready:
        decision = (
            "POLYPGEN_TECHNICAL_PAIRING_READY_"
            "SOURCE_TRAINING_OVERLAP_UNRESOLVED"
        )
    else:
        decision = "POLYPGEN_TECHNICAL_ASSET_AUDIT_NOT_READY"

    summary = {
        "decision": decision,
        "candidate_root": str(root),
        "raster_files": len(files),
        "unique_pairs": len(audit),
        "readable_pairs": readable_pairs,
        "same_size_pairs": same_size_pairs,
        "exact_neopolyp_rgb_duplicates": duplicate_count,
        "source_training_overlap_status": "UNRESOLVED",
        "model_inference_run": False,
        "external_performance_evaluated": False,
        "method_tuning_run": False,
    }

    with open(
        out / "R10L1_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n===== FINAL DECISION =====")
    print("DECISION:", decision)
    print("MODEL INFERENCE RUN: NO")
    print("EXTERNAL PERFORMANCE EVALUATED: NO")
    print("SOURCE-TRAINING OVERLAP STATUS: UNRESOLVED")
    print("\nOutputs:")
    print(out / "R10L1_polypgen_candidate_pair_manifest.csv")
    print(out / "R10L1_ambiguous_or_unpaired_keys.csv")
    print(out / "R10L1_summary.json")
    print("PASS")


if __name__ == "__main__":
    main()
