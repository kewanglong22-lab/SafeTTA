#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R14B4B3_neopolyp_exact_historical_hash_recipe_verification_fix1.py

Purpose
-------
Verify the exact historical NeoPolyp pixel-hash recipe recovered from the
authoritative R05D1 source code.

Historical recipe recovered from R05D1:
    with Image.open(path) as img:
        rgb = img.convert("RGB")
        w, h = rgb.size
        raw = rgb.tobytes()

    hasher = hashlib.sha256()
    hasher.update(f"{w}x{h}|RGB|".encode("ascii"))
    hasher.update(raw)

This stage MUST reproduce 1000/1000 authoritative
`image_rgb_pixel_sha256` values before R14B4B contamination auditing resumes.

Information boundary:
- NeoPolyp IMAGE pixels: YES, for hash verification only.
- NeoPolyp GT pixels: NO.
- SUN-SEG RGB pixels: NO.
- contamination comparison: NO.
- model/TTA inference: NO.
- safety scores/outcomes: NO.
- frozen SUN cohort modification: NO.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import inspect
import json
from pathlib import Path

from PIL import Image
from tqdm import tqdm


ROOT = Path(r"F:\MEDSEG_SAFETTA")

DEFAULT_MANIFEST = (
    ROOT / "outputs"
    / "Q1_R05D1_neopolyp_acquisition_pairing_duplicate_qc_v1_fix2"
    / "frozen_confirmatory_manifest.csv"
)

DEFAULT_HISTORICAL_SCRIPT = (
    ROOT / "code"
    / "Q1_R05D1_neopolyp_acquisition_pairing_duplicate_qc_v1_fix2.py"
)

DEFAULT_OUT = (
    ROOT / "outputs"
    / "Q1_R14B4B3_neopolyp_exact_historical_hash_recipe_verification_fix1_v1"
)

EXPECTED_MANIFEST_SHA = (
    "c944b95ec17f9c3fe26ea1c8555ca9309e490fc22a62da8fab43fa8ace1d39b9"
)
EXPECTED_HISTORICAL_SCRIPT_SHA = (
    "55ebe1b051bc250c3d5d471ee224fb5cd121501621120aaeb4729b1b3e2eee31"
)
EXPECTED_ROWS = 1000

DECISION = "NEOPOLYP_HISTORICAL_PIXEL_HASH_RECIPE_VERIFIED_1000_OF_1000"

EXPECTED_FUNCTION_SNIPPETS = (
    'with Image.open(path) as img:',
    'rgb = img.convert("RGB")',
    'w, h = rgb.size',
    'raw = rgb.tobytes()',
    'hasher = hashlib.sha256()',
    'hasher.update(f"{w}x{h}|RGB|".encode("ascii"))',
    'hasher.update(raw)',
    'return hasher.hexdigest(), w, h',
)


def sha256_file(path: Path, chunk=4 * 1024 * 1024) -> str:
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
        r = csv.DictReader(f)
        return r.fieldnames or [], list(r)


def write_csv(path: Path, rows, fields):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def extract_function_source(script_path: Path, function_name: str) -> str:
    text = script_path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    lines = text.splitlines()

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            if not hasattr(node, "end_lineno"):
                raise RuntimeError("Python AST lacks end_lineno.")
            return "\n".join(lines[node.lineno - 1: node.end_lineno])

    raise RuntimeError(
        f"Function {function_name} not found in historical script."
    )


def historical_canonical_rgb_sha256(path: Path):
    """
    Exact historical R05D1 recipe.

    IMPORTANT: width precedes height in the ASCII prefix:
        "{w}x{h}|RGB|"
    """
    with Image.open(path) as img:
        rgb = img.convert("RGB")
        w, h = rgb.size
        raw = rgb.tobytes()

    hasher = hashlib.sha256()
    hasher.update(f"{w}x{h}|RGB|".encode("ascii"))
    hasher.update(raw)
    return hasher.hexdigest(), w, h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument(
        "--historical-script",
        type=Path,
        default=DEFAULT_HISTORICAL_SCRIPT,
    )
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    print("===== Q1 R14B4B3 NEOPOLYP EXACT HISTORICAL HASH RECIPE VERIFICATION =====")
    print("NEOPOLYP_IMAGE_PIXEL_ACCESS=YES_HASH_VERIFICATION_ONLY")
    print("NEOPOLYP_GT_PIXEL_ACCESS=NO")
    print("SUN_RGB_ACCESS=NO")
    print("CONTAMINATION_COMPARISON=NO")
    print("MODEL_INFERENCE=NO")
    print("TTA_EXECUTION=NO")
    print("TARGET_OUTCOME_ACCESS=NO")
    print("FROZEN_SUN_COHORT_CHANGE=NO")

    for p in (args.manifest, args.historical_script):
        if not p.exists():
            raise FileNotFoundError(p)

    manifest_sha = sha256_file(args.manifest)
    historical_script_sha = sha256_file(args.historical_script)

    print("\n===== LINEAGE GATES =====")
    print("NeoPolyp manifest SHA256:", manifest_sha)
    print("historical R05D1 fix2 SHA256:", historical_script_sha)

    if manifest_sha != EXPECTED_MANIFEST_SHA:
        raise RuntimeError("NeoPolyp manifest SHA mismatch.")
    if historical_script_sha != EXPECTED_HISTORICAL_SCRIPT_SHA:
        raise RuntimeError("Historical R05D1 fix2 script SHA mismatch.")

    source = extract_function_source(
        args.historical_script,
        "canonical_rgb_sha256",
    )

    print("\n===== HISTORICAL FUNCTION SOURCE =====")
    print(source)

    missing_snippets = [
        s for s in EXPECTED_FUNCTION_SNIPPETS if s not in source
    ]
    if missing_snippets:
        raise RuntimeError(
            "Historical canonical_rgb_sha256 source does not match "
            f"the recovered recipe. Missing: {missing_snippets}"
        )

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
        raise RuntimeError(f"Missing columns: {sorted(missing)}")

    raw_mismatch = 0
    pixel_mismatch = 0
    per_rows = []

    for r in tqdm(
        rows,
        desc="Verify exact historical NeoPolyp hashes",
        unit="image",
        dynamic_ncols=True,
    ):
        p = Path(r["image_path"])
        if not p.is_file():
            raise FileNotFoundError(p)

        raw_sha = sha256_file(p)
        pixel_sha, width, height = historical_canonical_rgb_sha256(p)

        raw_ok = raw_sha.lower() == r["image_raw_sha256"].strip().lower()
        pixel_ok = (
            pixel_sha.lower()
            == r["image_rgb_pixel_sha256"].strip().lower()
        )

        raw_mismatch += int(not raw_ok)
        pixel_mismatch += int(not pixel_ok)

        per_rows.append({
            "sample_id": r["sample_id"],
            "image_path": str(p),
            "width": width,
            "height": height,
            "raw_sha_match": int(raw_ok),
            "historical_pixel_sha_match": int(pixel_ok),
            "computed_historical_pixel_sha256": pixel_sha,
            "stored_image_rgb_pixel_sha256": r["image_rgb_pixel_sha256"],
        })

    print("\n===== EXACT RECIPE VERIFICATION =====")
    print("rows:", len(rows))
    print("raw SHA mismatches:", raw_mismatch)
    print("historical pixel SHA matches:", EXPECTED_ROWS - pixel_mismatch)
    print("historical pixel SHA mismatches:", pixel_mismatch)

    if raw_mismatch != 0:
        raise RuntimeError("NeoPolyp raw-file lineage mismatch.")
    if pixel_mismatch != 0:
        raise RuntimeError(
            f"Historical recipe failed: {pixel_mismatch}/"
            f"{EXPECTED_ROWS} pixel hashes mismatch."
        )

    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)

    audit_csv = (
        args.output_dir
        / "R14B4B3_NEOPOLYP_EXACT_HISTORICAL_HASH_VERIFICATION.csv"
    )
    write_csv(
        audit_csv,
        per_rows,
        [
            "sample_id",
            "image_path",
            "width",
            "height",
            "raw_sha_match",
            "historical_pixel_sha_match",
            "computed_historical_pixel_sha256",
            "stored_image_rgb_pixel_sha256",
        ],
    )

    lock = {
        "status": "PASS",
        "decision": DECISION,
        "manifest_path": str(args.manifest),
        "manifest_sha256": manifest_sha,
        "historical_script_path": str(args.historical_script),
        "historical_script_sha256": historical_script_sha,
        "historical_function": "canonical_rgb_sha256",
        "recovered_recipe": {
            "decoder": "Pillow Image.open",
            "color_conversion": 'convert("RGB")',
            "raw_pixels": "rgb.tobytes()",
            "sha256_prefix": '{w}x{h}|RGB|',
            "prefix_encoding": "ascii",
            "hash_payload_order": "prefix_then_raw_rgb_bytes",
        },
        "verification": {
            "rows": EXPECTED_ROWS,
            "raw_sha_mismatches": raw_mismatch,
            "historical_pixel_sha_matches": EXPECTED_ROWS - pixel_mismatch,
            "historical_pixel_sha_mismatches": pixel_mismatch,
        },
        "information_boundary": {
            "neopolyp_image_pixels_accessed_for_hash_verification": True,
            "neopolyp_gt_pixels_accessed": False,
            "sun_rgb_accessed": False,
            "contamination_comparison": False,
            "model_inference": False,
            "tta_execution": False,
            "safety_scores_accessed": False,
            "target_outcomes_accessed": False,
            "frozen_sun_cohort_changed": False,
        },
        "next_stage": (
            "R14B4B_FIX2_RESUME_SUNSEG_CONTAMINATION_AUDIT_"
            "USING_VERIFIED_HISTORICAL_PIXEL_HASH_RECIPE"
        ),
    }

    lock_path = (
        args.output_dir
        / "R14B4B3_NEOPOLYP_HISTORICAL_HASH_RECIPE_LOCK.json"
    )
    lock_path.write_text(
        json.dumps(lock, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nstatus= PASS")
    print("Decision=", DECISION)
    print("LOCK=", lock_path)
    print("PASS")


if __name__ == "__main__":
    main()
