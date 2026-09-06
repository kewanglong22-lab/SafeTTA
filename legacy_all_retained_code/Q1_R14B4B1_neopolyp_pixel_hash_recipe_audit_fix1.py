#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R14B4B1_neopolyp_pixel_hash_recipe_audit_fix1.py

Purpose:
Diagnose why R14B4B Fix1 reproduced all NeoPolyp raw-file SHA256 values but
0/1000 stored `image_rgb_pixel_sha256` values.

This is a provenance/hash-recipe audit only.

It DOES NOT:
- access SUN-SEG RGB pixels;
- decode any GT pixels;
- perform contamination comparison;
- run model/TTA inference;
- read safety scores or outcomes;
- modify the frozen R14B3 cohort.

The script tests a fixed panel of plausible historical pixel-hash recipes
against the authoritative NeoPolyp manifest and reports exact match counts.
No recipe is accepted unless it reproduces 1000/1000 stored hashes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm

try:
    import cv2
except Exception:
    cv2 = None


ROOT = Path(r"F:\MEDSEG_SAFETTA")

DEFAULT_MANIFEST = (
    ROOT / "outputs"
    / "Q1_R05D1_neopolyp_acquisition_pairing_duplicate_qc_v1_fix2"
    / "frozen_confirmatory_manifest.csv"
)

DEFAULT_OUT = (
    ROOT / "outputs"
    / "Q1_R14B4B1_neopolyp_pixel_hash_recipe_audit_fix1_v1"
)

EXPECTED_MANIFEST_SHA = (
    "c944b95ec17f9c3fe26ea1c8555ca9309e490fc22a62da8fab43fa8ace1d39b9"
)
EXPECTED_ROWS = 1000
DECISION_PASS = "NEOPOLYP_HISTORICAL_PIXEL_HASH_RECIPE_EXACTLY_RECOVERED"
DECISION_STOP = "NEOPOLYP_HISTORICAL_PIXEL_HASH_RECIPE_NOT_RECOVERED_STOP"


def sha256_file(path: Path, chunk=4 * 1024 * 1024):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def h(data: bytes):
    return hashlib.sha256(data).hexdigest()


def read_csv(path: Path):
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        return r.fieldnames or [], list(r)


def write_csv(path: Path, rows, fields):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def encode_shape_ascii(arr):
    return f"{arr.shape[0]}x{arr.shape[1]}x{arr.shape[2]}|".encode("ascii")


def encode_shape_i32_le(arr):
    return struct.pack("<iii", arr.shape[0], arr.shape[1], arr.shape[2])


def encode_shape_i64_le(arr):
    return struct.pack("<qqq", arr.shape[0], arr.shape[1], arr.shape[2])


def recipe_hashes(path: Path):
    """
    Return a fixed, descriptive candidate panel.

    We intentionally test:
    - Pillow RGB C-order bytes;
    - channel-reversed BGR bytes;
    - EXIF-transposed RGB/BGR;
    - raw decoded mode bytes;
    - shape-prefixed variants;
    - Fortran-order variants;
    - OpenCV BGR/RGB when cv2 is available.

    This is not hyperparameter tuning: it is exact provenance reconstruction.
    """
    out = {}

    with Image.open(path) as im:
        rgb_im = im.convert("RGB")
        rgb = np.asarray(rgb_im, dtype=np.uint8)
        rgb_c = np.ascontiguousarray(rgb)
        bgr_c = np.ascontiguousarray(rgb[..., ::-1])

        out["pil_rgb_tobytes"] = h(rgb_im.tobytes())
        out["numpy_rgb_c"] = h(rgb_c.tobytes(order="C"))
        out["numpy_bgr_c"] = h(bgr_c.tobytes(order="C"))
        out["numpy_rgb_f"] = h(np.asfortranarray(rgb).tobytes(order="F"))
        out["numpy_bgr_f"] = h(np.asfortranarray(rgb[..., ::-1]).tobytes(order="F"))

        out["rgb_shape_ascii_plus_bytes"] = h(
            encode_shape_ascii(rgb_c) + rgb_c.tobytes(order="C")
        )
        out["bgr_shape_ascii_plus_bytes"] = h(
            encode_shape_ascii(bgr_c) + bgr_c.tobytes(order="C")
        )
        out["rgb_shape_i32le_plus_bytes"] = h(
            encode_shape_i32_le(rgb_c) + rgb_c.tobytes(order="C")
        )
        out["bgr_shape_i32le_plus_bytes"] = h(
            encode_shape_i32_le(bgr_c) + bgr_c.tobytes(order="C")
        )
        out["rgb_shape_i64le_plus_bytes"] = h(
            encode_shape_i64_le(rgb_c) + rgb_c.tobytes(order="C")
        )
        out["bgr_shape_i64le_plus_bytes"] = h(
            encode_shape_i64_le(bgr_c) + bgr_c.tobytes(order="C")
        )

        # Mode-preserving decoded bytes, if different from RGB.
        native = np.asarray(im)
        try:
            out["pil_native_tobytes"] = h(im.tobytes())
            out["numpy_native_c"] = h(
                np.ascontiguousarray(native).tobytes(order="C")
            )
        except Exception:
            pass

        exif_im = ImageOps.exif_transpose(im).convert("RGB")
        exif_rgb = np.asarray(exif_im, dtype=np.uint8)
        exif_rgb_c = np.ascontiguousarray(exif_rgb)
        exif_bgr_c = np.ascontiguousarray(exif_rgb[..., ::-1])
        out["pil_exif_rgb_tobytes"] = h(exif_im.tobytes())
        out["numpy_exif_rgb_c"] = h(exif_rgb_c.tobytes(order="C"))
        out["numpy_exif_bgr_c"] = h(exif_bgr_c.tobytes(order="C"))

    if cv2 is not None:
        cv_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if cv_bgr is not None:
            cv_bgr = np.ascontiguousarray(cv_bgr, dtype=np.uint8)
            cv_rgb = np.ascontiguousarray(
                cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2RGB),
                dtype=np.uint8,
            )
            out["opencv_bgr_c"] = h(cv_bgr.tobytes(order="C"))
            out["opencv_rgb_c"] = h(cv_rgb.tobytes(order="C"))
            out["opencv_bgr_shape_ascii_plus_bytes"] = h(
                encode_shape_ascii(cv_bgr) + cv_bgr.tobytes(order="C")
            )
            out["opencv_rgb_shape_ascii_plus_bytes"] = h(
                encode_shape_ascii(cv_rgb) + cv_rgb.tobytes(order="C")
            )

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    print("===== Q1 R14B4B1 NEOPOLYP PIXEL-HASH RECIPE AUDIT =====")
    print("SUN_RGB_ACCESS=NO")
    print("GT_PIXEL_DECODE=NO")
    print("CONTAMINATION_COMPARISON=NO")
    print("MODEL_INFERENCE=NO")
    print("TTA_EXECUTION=NO")
    print("TARGET_OUTCOME_ACCESS=NO")
    print("FROZEN_COHORT_CHANGE=NO")

    if not args.manifest.exists():
        raise FileNotFoundError(args.manifest)

    manifest_sha = sha256_file(args.manifest)
    print("manifest SHA256:", manifest_sha)
    if manifest_sha != EXPECTED_MANIFEST_SHA:
        raise RuntimeError("NeoPolyp manifest SHA mismatch.")

    cols, rows = read_csv(args.manifest)
    if len(rows) != EXPECTED_ROWS:
        raise RuntimeError(
            f"NeoPolyp rows {len(rows)} != expected {EXPECTED_ROWS}"
        )

    required = {
        "sample_id",
        "image_path",
        "image_raw_sha256",
        "image_rgb_pixel_sha256",
    }
    missing = required - set(cols)
    if missing:
        raise RuntimeError(f"Missing required columns: {sorted(missing)}")

    raw_mismatch = 0
    recipe_counts = {}
    per_image = []

    for idx, r in enumerate(
        tqdm(rows, desc="Audit NeoPolyp hash recipes", unit="image",
             dynamic_ncols=True)
    ):
        p = Path(r["image_path"])
        if not p.is_file():
            raise FileNotFoundError(p)

        raw_sha = sha256_file(p)
        if raw_sha != r["image_raw_sha256"]:
            raw_mismatch += 1

        candidates = recipe_hashes(p)
        if not recipe_counts:
            recipe_counts = {name: 0 for name in candidates}
        else:
            # Candidate panel must remain identical across rows.
            if set(candidates) != set(recipe_counts):
                raise RuntimeError(
                    "Hash recipe candidate set changed across images."
                )

        target = r["image_rgb_pixel_sha256"].strip().lower()
        matched = []
        for name, digest in candidates.items():
            if digest.lower() == target:
                recipe_counts[name] += 1
                matched.append(name)

        per_image.append({
            "sample_id": r["sample_id"],
            "image_path": r["image_path"],
            "stored_pixel_sha256": target,
            "matched_recipe_count": len(matched),
            "matched_recipes": "|".join(matched),
        })

    print("\n===== RAW-FILE LINEAGE =====")
    print("raw SHA mismatches:", raw_mismatch)
    if raw_mismatch:
        raise RuntimeError(
            "NeoPolyp raw-file SHA lineage changed; STOP."
        )

    print("\n===== RECIPE EXACT-MATCH COUNTS =====")
    ranking = sorted(
        recipe_counts.items(),
        key=lambda x: (-x[1], x[0]),
    )
    for name, count in ranking:
        print(f"{name}: {count}/{EXPECTED_ROWS}")

    full = [name for name, count in ranking if count == EXPECTED_ROWS]
    any_match_rows = sum(
        int(int(r["matched_recipe_count"]) > 0) for r in per_image
    )
    zero_match_rows = EXPECTED_ROWS - any_match_rows

    print("\nrows with >=1 candidate recipe match:", any_match_rows)
    print("rows with zero candidate recipe match:", zero_match_rows)
    print("1000/1000 reproducing recipes:", full)

    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)

    summary_rows = [
        {
            "recipe": name,
            "exact_match_count": count,
            "total": EXPECTED_ROWS,
            "match_fraction": count / EXPECTED_ROWS,
        }
        for name, count in ranking
    ]
    summary_path = (
        args.output_dir
        / "R14B4B1_NEOPOLYP_PIXEL_HASH_RECIPE_COUNTS.csv"
    )
    write_csv(
        summary_path,
        summary_rows,
        ["recipe", "exact_match_count", "total", "match_fraction"],
    )

    per_path = (
        args.output_dir
        / "R14B4B1_NEOPOLYP_PIXEL_HASH_PER_IMAGE_AUDIT.csv"
    )
    write_csv(
        per_path,
        per_image,
        [
            "sample_id",
            "image_path",
            "stored_pixel_sha256",
            "matched_recipe_count",
            "matched_recipes",
        ],
    )

    if len(full) == 1:
        status = "PASS"
        decision = DECISION_PASS
        recovered_recipe = full[0]
    elif len(full) > 1:
        # Multiple exact-equivalent recipes can happen if two implementations
        # produce identical bytes. This is still scientifically recoverable,
        # but requires choosing the simplest equivalent implementation later.
        status = "PASS_EQUIVALENT_RECIPES"
        decision = DECISION_PASS
        recovered_recipe = "|".join(full)
    else:
        status = "STOP"
        decision = DECISION_STOP
        recovered_recipe = None

    lock = {
        "status": status,
        "decision": decision,
        "manifest_path": str(args.manifest),
        "manifest_sha256": manifest_sha,
        "rows": EXPECTED_ROWS,
        "raw_sha_mismatch_count": raw_mismatch,
        "candidate_recipe_match_counts": dict(ranking),
        "rows_with_any_candidate_match": any_match_rows,
        "rows_with_zero_candidate_match": zero_match_rows,
        "full_reproducing_recipes": full,
        "recovered_recipe": recovered_recipe,
        "information_boundary": {
            "sun_rgb_accessed": False,
            "gt_pixels_decoded": False,
            "contamination_comparison": False,
            "model_inference": False,
            "tta_execution": False,
            "safety_scores_accessed": False,
            "target_outcomes_accessed": False,
            "frozen_cohort_changed": False,
        },
        "next_stage": (
            "R14B4B_FIX2_USE_RECOVERED_REFERENCE_PIXEL_HASH_RECIPE"
            if status.startswith("PASS")
            else "AUDIT_R05D1_HISTORICAL_HASH_CODE_BEFORE_CONTINUING"
        ),
    }

    lock_path = (
        args.output_dir
        / "R14B4B1_NEOPOLYP_PIXEL_HASH_RECIPE_AUDIT_LOCK.json"
    )
    lock_path.write_text(
        json.dumps(lock, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nstatus=", status)
    print("Decision=", decision)
    print("recovered recipe=", recovered_recipe)
    print("LOCK=", lock_path)

    if status.startswith("PASS"):
        print("PASS")
    else:
        print("STOP")


if __name__ == "__main__":
    main()
