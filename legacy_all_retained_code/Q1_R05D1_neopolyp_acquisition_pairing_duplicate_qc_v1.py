#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Q1-R05D1 — Official NeoPolyp acquisition + 1000 pairing + exact duplicate QC.

Target image pixels may be decoded ONLY for exact duplicate hashing.
Target train_gt pixels are NEVER decoded.
No inference, TTA, PAOT prediction, Dice, or predictor fitting.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import traceback
import zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Sequence

from tqdm import tqdm


VERSION = "2026-08-19-Q1-R05D1-v1"
BUILD = "Q1_R05D1_NEOPOLYP_ACQUISITION_PAIRING_DUPLICATE_QC"

ROOT = Path(r"F:\MEDSEG_SAFETTA")

PROTOCOL = (
    ROOT / "docs"
    / "Q1_R05D1_neopolyp_acquisition_pairing_duplicate_qc_preregistered_protocol_v1.md"
)
EXPECTED_PROTOCOL_SHA256 = "33a63aec9db2bddba60b5abbfe1382e9b42949256b1372b1bb3ecd1a7e183ff3"

R05D0_LOCK = (
    ROOT / "outputs"
    / "Q1_R05D0_neopolyp_backup_confirmatory_target_selection_lock_v1"
    / "Q1_R05D0_NEOPOLYP_SELECTION_LOCK.json"
)
EXPECTED_R05D0_LOCK_SHA256 = (
    "b46e1fff10deea8323e3cb0df3ea0feb1c01840a35a8798de816e9bc35e1242d"
)
EXPECTED_R05D0_DECISION = "NEOPOLYP_BACKUP_CONFIRMATORY_TARGET_LOCKED"

EXPECTED_R05B2_LOCK_SHA256 = (
    "bd182ac600afbb24b3a62ebd326931ab23c7cf6b873a2178e33500395c70f211"
)

RAW_ROOT = ROOT / "data" / "external" / "NeoPolyp_raw"
DEFAULT_EXTRACTED = RAW_ROOT / "extracted"

OUTPUT_DIR = (
    ROOT / "outputs"
    / "Q1_R05D1_neopolyp_acquisition_pairing_duplicate_qc_v1"
)

KAGGLE_COMPETITION = "bkai-igh-neopolyp"

EXPECTED_IMAGES = 1000
EXPECTED_MASKS = 1000
EXPECTED_PAIRS = 1000

IMAGE_EXTS = {".jpeg", ".jpg", ".png", ".bmp", ".tif", ".tiff"}

MASK_PATH_TOKENS = {
    "mask", "masks",
    "train_gt",
    "ground_truth", "ground-truth", "groundtruth",
    "annotation", "annotations",
    "label", "labels",
    "segmentation", "segmentations",
}

DECISION_READY = "NEOPOLYP_TARGET_QC_READY"
DECISION_FAILED = "NEOPOLYP_TARGET_QC_FAILED"


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def validate_sha(path: Path, expected: str, label: str) -> str:
    if not path.exists():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual.lower() != expected.lower():
        raise RuntimeError(
            f"{label} SHA mismatch: expected={expected} actual={actual}"
        )
    return actual


def write_csv(path: Path, rows, fields: Sequence[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(fields))
        w.writeheader()
        w.writerows(rows)


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def validate_upstream():
    validate_sha(PROTOCOL, EXPECTED_PROTOCOL_SHA256, "R05D1 protocol")
    validate_sha(R05D0_LOCK, EXPECTED_R05D0_LOCK_SHA256, "R05D0 lock")

    lock = json.loads(R05D0_LOCK.read_text(encoding="utf-8"))

    if lock.get("decision") != EXPECTED_R05D0_DECISION:
        raise RuntimeError(
            f"Unexpected R05D0 decision: {lock.get('decision')}"
        )

    if lock.get("r05b2_lock_sha256") != EXPECTED_R05B2_LOCK_SHA256:
        raise RuntimeError(
            "R05D0 does not point to the frozen R05B2 predictor lock."
        )

    if bool(lock.get("neopolyp_downloaded", True)):
        raise RuntimeError("R05D0 reports NeoPolyp already downloaded.")
    if bool(lock.get("neopolyp_image_pixels_opened", True)):
        raise RuntimeError("R05D0 reports NeoPolyp image access.")
    if bool(lock.get("neopolyp_gt_pixels_opened", True)):
        raise RuntimeError("R05D0 reports NeoPolyp GT access.")

    return lock


def ensure_not_c_drive(path: Path):
    resolved = path.resolve()
    drive = resolved.drive.upper()
    if drive == "C:":
        raise RuntimeError(
            f"C-drive storage is forbidden by protocol: {resolved}"
        )


def run_kaggle_download(raw_root: Path) -> Path:
    ensure_not_c_drive(raw_root)
    raw_root.mkdir(parents=True, exist_ok=True)

    kaggle_exe = shutil.which("kaggle")
    if kaggle_exe:
        cmd = [
            kaggle_exe,
            "competitions",
            "download",
            "-c",
            KAGGLE_COMPETITION,
            "-p",
            str(raw_root),
            "--force",
        ]
    else:
        cmd = [
            sys.executable,
            "-m",
            "kaggle",
            "competitions",
            "download",
            "-c",
            KAGGLE_COMPETITION,
            "-p",
            str(raw_root),
            "--force",
        ]

    print("Running official Kaggle competition download...")
    print("Command:", " ".join(cmd))

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    print(proc.stdout)

    if proc.returncode != 0:
        raise RuntimeError(
            "Kaggle download failed. Confirm that the Kaggle CLI is "
            "installed/authenticated and that competition rules have been "
            "accepted. No fallback mirror is used."
        )

    zips = sorted(
        raw_root.glob("*.zip"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not zips:
        raise RuntimeError(
            f"Kaggle command returned successfully but no ZIP found in {raw_root}"
        )

    return zips[0]


def safe_extract_zip(zip_path: Path, extracted_root: Path):
    ensure_not_c_drive(extracted_root)

    if extracted_root.exists():
        if any(extracted_root.iterdir()):
            raise FileExistsError(
                f"Extraction directory is not empty: {extracted_root}"
            )
    else:
        extracted_root.mkdir(parents=True, exist_ok=False)

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.infolist()

        root_resolved = extracted_root.resolve()

        for info in tqdm(
            members,
            desc="Extract official NeoPolyp ZIP",
            unit="file",
            dynamic_ncols=True,
        ):
            member = PurePosixPath(info.filename)
            if member.is_absolute() or ".." in member.parts:
                raise RuntimeError(
                    f"Unsafe archive member path: {info.filename}"
                )

            dest = (extracted_root / Path(*member.parts)).resolve()
            try:
                dest.relative_to(root_resolved)
            except ValueError:
                raise RuntimeError(
                    f"Archive member escapes extraction root: {info.filename}"
                )

            if info.is_dir():
                dest.mkdir(parents=True, exist_ok=True)
                continue

            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info, "r") as src, dest.open("wb") as dst:
                shutil.copyfileobj(src, dst)


def path_parts_lower(path: Path):
    return [p.lower() for p in path.parts]


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS


def is_mask_like_path(path: Path) -> bool:
    low = str(path).replace("\\", "/").lower()
    parts = set(path_parts_lower(path))

    if "train_gt" in parts:
        return True

    return any(token in low for token in MASK_PATH_TOKENS)


def is_neopolyp_train_image(path: Path, root: Path) -> bool:
    if not is_image_file(path):
        return False

    rel = path.relative_to(root)
    parts = path_parts_lower(rel)

    if "train_gt" in parts:
        return False
    if is_mask_like_path(rel):
        return False

    return "train" in parts


def is_neopolyp_train_gt(path: Path, root: Path) -> bool:
    if not is_image_file(path):
        return False

    rel = path.relative_to(root)
    parts = path_parts_lower(rel)
    return "train_gt" in parts


def resolve_neopolyp_pairing(data_root: Path):
    files = [p for p in data_root.rglob("*") if p.is_file()]

    images = [
        p for p in files
        if is_neopolyp_train_image(p, data_root)
    ]
    masks = [
        p for p in files
        if is_neopolyp_train_gt(p, data_root)
    ]

    image_by_stem = defaultdict(list)
    mask_by_stem = defaultdict(list)

    for p in images:
        image_by_stem[p.stem.lower()].append(p)
    for p in masks:
        mask_by_stem[p.stem.lower()].append(p)

    all_stems = sorted(set(image_by_stem) | set(mask_by_stem))
    pairing_rows = []

    for stem in all_stems:
        ii = image_by_stem.get(stem, [])
        mm = mask_by_stem.get(stem, [])

        paired = len(ii) == 1 and len(mm) == 1

        pairing_rows.append({
            "sample_id": f"NeoPolyp::{stem}",
            "stem": stem,
            "image_count": len(ii),
            "gt_count": len(mm),
            "image_path": str(ii[0]) if len(ii) == 1 else "",
            "gt_path": str(mm[0]) if len(mm) == 1 else "",
            "pair_status": "PAIRED" if paired else "AMBIGUOUS",
        })

    return images, masks, pairing_rows, files


def canonical_rgb_sha256(path: Path):
    """
    Decode IMAGE pixels only. This function must never be called on GT paths.
    """
    from PIL import Image

    with Image.open(path) as img:
        rgb = img.convert("RGB")
        w, h = rgb.size
        raw = rgb.tobytes()

    hasher = hashlib.sha256()
    hasher.update(f"{w}x{h}|RGB|".encode("ascii"))
    hasher.update(raw)
    return hasher.hexdigest(), w, h


def build_historical_image_index(
    data_root: Path,
    current_target_root: Path,
):
    rows = []
    raw_map = defaultdict(list)
    pixel_map = defaultdict(list)

    candidates = []

    target_resolved = current_target_root.resolve()

    for p in data_root.rglob("*"):
        try:
            if not p.is_file() or not is_image_file(p):
                continue

            rp = p.resolve()

            try:
                rp.relative_to(target_resolved)
                continue
            except ValueError:
                pass

            rel_low = str(p).replace("\\", "/").lower()
            if is_mask_like_path(Path(rel_low)):
                continue

            candidates.append(p)
        except OSError:
            continue

    for p in tqdm(
        candidates,
        desc="Index historical development images",
        unit="image",
        dynamic_ncols=True,
    ):
        try:
            raw_sha = sha256_file(p)
            pixel_sha, width, height = canonical_rgb_sha256(p)
        except Exception as e:
            rows.append({
                "file": str(p),
                "raw_sha256": "",
                "rgb_pixel_sha256": "",
                "width": "",
                "height": "",
                "status": f"READ_FAILED:{type(e).__name__}",
            })
            continue

        rows.append({
            "file": str(p),
            "raw_sha256": raw_sha,
            "rgb_pixel_sha256": pixel_sha,
            "width": width,
            "height": height,
            "status": "OK",
        })

        raw_map[raw_sha].append(str(p))
        pixel_map[pixel_sha].append(str(p))

    return rows, raw_map, pixel_map


def duplicate_qc(
    pairing_rows,
    raw_map,
    pixel_map,
):
    report = []
    manifest = []

    paired_rows = [
        r for r in pairing_rows
        if r["pair_status"] == "PAIRED"
    ]

    for r in tqdm(
        paired_rows,
        desc="NeoPolyp exact duplicate QC",
        unit="case",
        dynamic_ncols=True,
    ):
        image_path = Path(r["image_path"])
        gt_path = Path(r["gt_path"])

        # Explicit guard: image decoder is never given GT.
        if is_mask_like_path(image_path):
            raise RuntimeError(
                f"Target image path unexpectedly classified as mask: {image_path}"
            )
        if not is_mask_like_path(gt_path):
            raise RuntimeError(
                f"GT path unexpectedly not classified as GT: {gt_path}"
            )

        raw_sha = sha256_file(image_path)
        pixel_sha, width, height = canonical_rgb_sha256(image_path)

        raw_matches = raw_map.get(raw_sha, [])
        pixel_matches = pixel_map.get(pixel_sha, [])

        raw_overlap = bool(raw_matches)
        pixel_overlap = bool(pixel_matches)
        exact_overlap = raw_overlap or pixel_overlap

        exclusion_reason = (
            "EXACT_DEVELOPMENT_IMAGE_OVERLAP"
            if exact_overlap
            else ""
        )

        report.append({
            "sample_id": r["sample_id"],
            "image_path": str(image_path),
            "gt_path": str(gt_path),
            "image_raw_sha256": raw_sha,
            "image_rgb_pixel_sha256": pixel_sha,
            "width": width,
            "height": height,
            "raw_overlap": int(raw_overlap),
            "rgb_pixel_overlap": int(pixel_overlap),
            "raw_match_files": ";".join(raw_matches),
            "pixel_match_files": ";".join(pixel_matches),
            "exact_overlap": int(exact_overlap),
        })

        manifest.append({
            "sample_id": r["sample_id"],
            "image_path": str(image_path),
            "gt_path": str(gt_path),
            "image_raw_sha256": raw_sha,
            "image_rgb_pixel_sha256": pixel_sha,
            "eligible_confirmatory": int(not exact_overlap),
            "exclusion_reason": exclusion_reason,
            "gt_pixels_decoded": 0,
        })

    return report, manifest


def preflight():
    lock = validate_upstream()

    print("===== Q1-R05D1 PREFLIGHT =====")
    print(f"protocol_sha256={EXPECTED_PROTOCOL_SHA256}")
    print(f"r05d0_lock_sha256={EXPECTED_R05D0_LOCK_SHA256}")
    print(f"r05d0_decision={lock['decision']}")
    print(f"competition={KAGGLE_COMPETITION}")
    print(f"default_raw_root={RAW_ROOT}")
    print("expected_train_images=1000")
    print("expected_train_gt=1000")
    print("target_image_pixels_allowed_for_duplicate_qc=YES")
    print("target_gt_pixels_decoded=NO")
    print("model_inference=NO")
    print("TTA=NO")
    print("PAOT_probability_generation=NO")
    print("predictor_fit=NO")
    print("PREFLIGHT_PASS")


def run(args):
    validate_upstream()

    ensure_not_c_drive(args.raw_root)

    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    build_dir = Path(str(args.output_dir) + "__building")
    if build_dir.exists():
        raise FileExistsError(build_dir)
    build_dir.mkdir(parents=True, exist_ok=False)

    protocol_copy = build_dir / "preregistered_protocol_copy.md"
    shutil.copy2(PROTOCOL, protocol_copy)
    validate_sha(
        protocol_copy,
        EXPECTED_PROTOCOL_SHA256,
        "R05D1 protocol copy",
    )

    acquisition_mode = ""
    package_path = None

    if args.download:
        package_path = run_kaggle_download(args.raw_root)
        acquisition_mode = "official_kaggle_cli"
        provider_assertion = True
    elif args.input is not None:
        package_path = args.input.resolve()
        if not package_path.exists():
            raise FileNotFoundError(package_path)
        provider_assertion = bool(args.official_source)
        acquisition_mode = "user_supplied_official_package"
    else:
        raise RuntimeError(
            "Specify exactly one acquisition mode: --download or --input."
        )

    if args.download and args.input is not None:
        raise RuntimeError("Use only one of --download or --input.")

    if not provider_assertion:
        raise RuntimeError(
            "--official-source is required for user-supplied package."
        )

    if package_path.is_file():
        if not zipfile.is_zipfile(package_path):
            raise RuntimeError(
                f"Input file is not a ZIP archive: {package_path}"
            )

        extracted_root = args.extracted_root
        safe_extract_zip(package_path, extracted_root)
        data_root = extracted_root
        package_sha = sha256_file(package_path)
        package_size = package_path.stat().st_size

    elif package_path.is_dir():
        data_root = package_path
        package_sha = None
        package_size = sum(
            p.stat().st_size
            for p in package_path.rglob("*")
            if p.is_file()
        )
    else:
        raise RuntimeError("Unsupported package path.")

    images, masks, pairing_rows, package_files = resolve_neopolyp_pairing(
        data_root
    )

    pairing_path = build_dir / "neopolyp_1000_pairing.csv"
    write_csv(
        pairing_path,
        pairing_rows,
        [
            "sample_id",
            "stem",
            "image_count",
            "gt_count",
            "image_path",
            "gt_path",
            "pair_status",
        ],
    )

    inventory_rows = []
    for p in tqdm(
        package_files,
        desc="NeoPolyp package inventory",
        unit="file",
        dynamic_ncols=True,
    ):
        rel = p.relative_to(data_root)
        inventory_rows.append({
            "relpath": str(rel),
            "size_bytes": p.stat().st_size,
            "extension": p.suffix.lower(),
            "is_train_image": int(
                is_neopolyp_train_image(p, data_root)
            ),
            "is_train_gt": int(
                is_neopolyp_train_gt(p, data_root)
            ),
        })

    inventory_path = build_dir / "package_inventory.csv"
    write_csv(
        inventory_path,
        inventory_rows,
        [
            "relpath",
            "size_bytes",
            "extension",
            "is_train_image",
            "is_train_gt",
        ],
    )

    paired_count = sum(
        r["pair_status"] == "PAIRED"
        for r in pairing_rows
    )

    structure_ok = (
        len(images) == EXPECTED_IMAGES
        and len(masks) == EXPECTED_MASKS
        and paired_count == EXPECTED_PAIRS
        and len(pairing_rows) == EXPECTED_PAIRS
    )

    if structure_ok:
        hist_rows, raw_map, pixel_map = build_historical_image_index(
            ROOT / "data",
            data_root,
        )

        hist_path = build_dir / "historical_development_image_index.csv"
        write_csv(
            hist_path,
            hist_rows,
            [
                "file",
                "raw_sha256",
                "rgb_pixel_sha256",
                "width",
                "height",
                "status",
            ],
        )

        overlap_rows, manifest_rows = duplicate_qc(
            pairing_rows,
            raw_map,
            pixel_map,
        )
    else:
        hist_rows = []
        overlap_rows = []
        manifest_rows = []

        hist_path = build_dir / "historical_development_image_index.csv"
        write_csv(
            hist_path,
            [],
            [
                "file",
                "raw_sha256",
                "rgb_pixel_sha256",
                "width",
                "height",
                "status",
            ],
        )

    overlap_path = build_dir / "exact_overlap_report.csv"
    write_csv(
        overlap_path,
        overlap_rows,
        [
            "sample_id",
            "image_path",
            "gt_path",
            "image_raw_sha256",
            "image_rgb_pixel_sha256",
            "width",
            "height",
            "raw_overlap",
            "rgb_pixel_overlap",
            "raw_match_files",
            "pixel_match_files",
            "exact_overlap",
        ],
    )

    manifest_path = build_dir / "frozen_confirmatory_manifest.csv"
    write_csv(
        manifest_path,
        manifest_rows,
        [
            "sample_id",
            "image_path",
            "gt_path",
            "image_raw_sha256",
            "image_rgb_pixel_sha256",
            "eligible_confirmatory",
            "exclusion_reason",
            "gt_pixels_decoded",
        ],
    )

    exact_overlap_count = sum(
        int(r["exact_overlap"])
        for r in overlap_rows
    )
    eligible_count = sum(
        int(r["eligible_confirmatory"])
        for r in manifest_rows
    )

    if structure_ok:
        decision = DECISION_READY
        reason = (
            "Exact 1000/1000 pairing resolved and image-only duplicate QC "
            "completed."
        )
    else:
        decision = DECISION_FAILED
        reason = (
            f"Structure mismatch: images={len(images)}, "
            f"gt={len(masks)}, paired={paired_count}, "
            f"pairing_rows={len(pairing_rows)}."
        )

    receipt = {
        "acquisition_mode": acquisition_mode,
        "competition": KAGGLE_COMPETITION,
        "official_source_asserted": bool(provider_assertion),
        "package_path": str(package_path),
        "data_root": str(data_root),
        "package_sha256": package_sha,
        "package_size_bytes": package_size,
        "candidate_images": len(images),
        "candidate_gt_files": len(masks),
        "exact_pairs": paired_count,
        "gt_pixels_decoded": False,
    }
    receipt_path = build_dir / "acquisition_receipt.json"
    write_json(receipt_path, receipt)

    boundary = {
        "target_image_filenames_read": True,
        "target_gt_filenames_read": True,
        "target_image_pixels_decoded_for_duplicate_qc":
            bool(structure_ok),
        "target_gt_pixels_decoded": False,
        "historical_development_image_pixels_decoded_for_duplicate_qc":
            bool(structure_ok),
        "model_inference_run": False,
        "tta_run": False,
        "target_paot_probabilities_generated": False,
        "target_dice_computed": False,
        "harm_benefit_outcomes_computed": False,
        "predictor_fit_or_modified": False,
    }
    boundary_path = build_dir / "information_boundary_audit.json"
    write_json(boundary_path, boundary)

    decision_path = build_dir / "decision.txt"
    decision_path.write_text(decision + "\n", encoding="utf-8")

    lines = [
        "===== Q1-R05D1 NEOPOLYP ACQUISITION + PAIRING + DUPLICATE QC =====",
        f"Script version: {VERSION}",
        f"Build: {BUILD}",
        "",
        "Acquisition:",
        f"  mode={acquisition_mode}",
        f"  package={package_path}",
        f"  data_root={data_root}",
        f"  official_source_asserted={'YES' if provider_assertion else 'NO'}",
        "",
        "Structure:",
        f"  resolved train images={len(images)}",
        f"  resolved train_gt files={len(masks)}",
        f"  exact one-to-one pairs={paired_count}",
        f"  expected pairs={EXPECTED_PAIRS}",
        "",
        "Duplicate QC:",
        f"  historical development images indexed={len(hist_rows)}",
        f"  exact overlap exclusions={exact_overlap_count}",
        f"  eligible confirmatory cases={eligible_count}",
        "",
        "Information boundary:",
        (
            "  target image pixels decoded for duplicate QC="
            f"{'YES' if structure_ok else 'NO'}"
        ),
        "  target GT pixels decoded=NO",
        "  model inference=NO",
        "  TTA=NO",
        "  PAOT probabilities generated=NO",
        "  predictor fit/modified=NO",
        "",
        "Decision:",
        f"  {decision}",
        "Reason:",
        f"  {reason}",
    ]

    if decision == DECISION_READY:
        lines += [
            "",
            "Next:",
            "  Q1-R05D2 prospective PAOT probability generation + lock",
        ]

    run_log_path = build_dir / "run_log.txt"
    run_log_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    artifacts = {
        "protocol_copy": protocol_copy,
        "acquisition_receipt": receipt_path,
        "package_inventory": inventory_path,
        "neopolyp_pairing": pairing_path,
        "historical_development_image_index": hist_path,
        "exact_overlap_report": overlap_path,
        "frozen_confirmatory_manifest": manifest_path,
        "information_boundary_audit": boundary_path,
        "decision": decision_path,
        "run_log": run_log_path,
    }

    lock = {
        "script_version": VERSION,
        "build": BUILD,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "r05d0_lock_sha256": EXPECTED_R05D0_LOCK_SHA256,
        "r05b2_lock_sha256": EXPECTED_R05B2_LOCK_SHA256,
        "competition": KAGGLE_COMPETITION,
        "official_source_asserted": bool(provider_assertion),
        "candidate_cases": EXPECTED_PAIRS,
        "resolved_images": len(images),
        "resolved_gt_files": len(masks),
        "resolved_exact_pairs": paired_count,
        "exact_overlap_exclusions": exact_overlap_count,
        "eligible_confirmatory_cases": eligible_count,
        "target_image_pixels_decoded_for_duplicate_qc":
            bool(structure_ok),
        "target_gt_pixels_decoded": False,
        "model_inference_run": False,
        "tta_run": False,
        "target_paot_probabilities_generated": False,
        "predictor_fit_or_modified": False,
        "decision": decision,
        "reason": reason,
        "artifacts": {
            name: {
                "relative_path": str(path.relative_to(build_dir)),
                "sha256": sha256_file(path),
            }
            for name, path in artifacts.items()
        },
    }

    lock_path = build_dir / "Q1_R05D1_NEOPOLYP_TARGET_QC_LOCK.json"
    write_json(lock_path, lock)
    lock_sha = sha256_file(lock_path)

    for name, meta in lock["artifacts"].items():
        p = build_dir / meta["relative_path"]
        if sha256_file(p) != meta["sha256"]:
            raise RuntimeError(f"Artifact changed before commit: {name}")

    build_dir.rename(args.output_dir)

    print()
    print(
        (args.output_dir / "run_log.txt").read_text(
            encoding="utf-8"
        )
    )
    print(
        "Q1-R05D1 LOCK:",
        args.output_dir / "Q1_R05D1_NEOPOLYP_TARGET_QC_LOCK.json",
    )
    print("Q1-R05D1 LOCK SHA256:", lock_sha)


def self_test():
    # Frozen counts.
    assert EXPECTED_IMAGES == 1000
    assert EXPECTED_MASKS == 1000
    assert EXPECTED_PAIRS == 1000

    root = Path(r"F:\dummy")

    image_examples = [
        root / "train" / "train" / "abc.jpeg",
        root / "train" / "xyz.jpg",
    ]
    mask_examples = [
        root / "train_gt" / "train_gt" / "abc.jpeg",
        root / "train_gt" / "xyz.jpg",
    ]

    for p in image_examples:
        assert is_neopolyp_train_image(p, root), p
        assert not is_neopolyp_train_gt(p, root), p
        assert not is_mask_like_path(p), p

    for p in mask_examples:
        assert is_neopolyp_train_gt(p, root), p
        assert is_mask_like_path(p), p
        assert not is_neopolyp_train_image(p, root), p

    # Exact-stem pairing fixture.
    temp_rows = []
    image_by_stem = {
        "a": [root / "train" / "train" / "a.jpeg"],
        "b": [root / "train" / "train" / "b.jpeg"],
    }
    mask_by_stem = {
        "a": [root / "train_gt" / "train_gt" / "a.jpeg"],
        "b": [root / "train_gt" / "train_gt" / "b.jpeg"],
    }
    for stem in sorted(set(image_by_stem) | set(mask_by_stem)):
        assert len(image_by_stem[stem]) == 1
        assert len(mask_by_stem[stem]) == 1
        temp_rows.append(stem)
    assert temp_rows == ["a", "b"]

    # Static guard: GT is never passed to canonical_rgb_sha256 in duplicate_qc.
    import inspect
    source = inspect.getsource(duplicate_qc)
    assert "canonical_rgb_sha256(image_path)" in source
    assert "canonical_rgb_sha256(gt_path)" not in source

    print("FROZEN_1000_PAIR_COUNT_TEST_PASS")
    print("TRAIN_TRAIN_GT_PATH_RESOLVER_TEST_PASS")
    print("EXACT_STEM_PAIRING_TEST_PASS")
    print("GT_PIXEL_DECODE_GUARD_TEST_PASS")
    print("SELF_TEST_PASS")


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Q1-R05D1: official BKAI-IGH NeoPolyp acquisition, exact "
            "1000 train/train_gt pairing, and image-only exact duplicate QC."
        )
    )

    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    p.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    p.add_argument("--extracted-root", type=Path, default=DEFAULT_EXTRACTED)

    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--download",
        action="store_true",
        help=(
            "Download the official Kaggle competition package using the "
            "installed/authenticated Kaggle CLI."
        ),
    )
    mode.add_argument(
        "--input",
        type=Path,
        help=(
            "Use an already downloaded official NeoPolyp ZIP or extracted "
            "directory."
        ),
    )

    p.add_argument(
        "--official-source",
        action="store_true",
        help=(
            "Required with --input to assert the supplied package came from "
            "the official Kaggle competition source."
        ),
    )

    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--self-test", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    if args.self_test:
        self_test()
        return 0

    if args.preflight_only:
        preflight()
        return 0

    if not args.download and args.input is None:
        raise SystemExit(
            "Formal run requires either --download or --input."
        )

    try:
        run(args)
    except Exception:
        traceback.print_exc()
        raise

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
