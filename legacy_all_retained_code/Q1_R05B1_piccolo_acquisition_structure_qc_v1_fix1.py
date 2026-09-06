#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Q1-R05B1 FIX1 — PICCOLO acquisition and structure/QC audit.

No image decoding.
No mask decoding.
No model inference.
No TTA.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import traceback
import zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Sequence

from tqdm import tqdm


VERSION = "2026-08-19-Q1-R05B1-v1-fix1"
BUILD = "Q1_R05B1_PICCOLO_ACQUISITION_STRUCTURE_QC_MASKNAME_FIX1"

ROOT = Path(r"F:\MEDSEG_SAFETTA")

PROTOCOL = (
    ROOT / "docs"
    / "Q1_R05B1_piccolo_acquisition_structure_qc_preregistered_protocol_v1.md"
)
EXPECTED_PROTOCOL_SHA256 = "c56c88e98e0568efdbbf5707c35073f18829c79b7df0257f4086928e51024a1e"

R05B0_LOCK = (
    ROOT / "outputs"
    / "Q1_R05B0_piccolo_confirmatory_target_selection_lock_v1_fix1"
    / "Q1_R05B0_PICCOLO_SELECTION_LOCK.json"
)
EXPECTED_R05B0_LOCK_SHA256 = (
    "9618268026108df39330d35298ae34c526253c4ba2374b60edae0287c8e3b06f"
)
EXPECTED_R05B0_DECISION = "PICCOLO_CONFIRMATORY_TARGET_SELECTION_LOCKED"

DEFAULT_RAW_ROOT = ROOT / "data" / "external" / "PICCOLO_raw"
OUTPUT_DIR = (
    ROOT / "outputs"
    / "Q1_R05B1_piccolo_acquisition_structure_qc_v1_fix1"
)

EXPECTED_TEST_IMAGES = 333

IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp",
}
TEXT_EXTS = {
    ".txt", ".csv", ".tsv", ".json", ".jsonl", ".md", ".yaml", ".yml",
}
MAX_TEXT_BYTES = 16 * 1024 * 1024

TRAIN_TOKENS = {"train", "training"}
VAL_TOKENS = {"val", "valid", "validation"}
TEST_TOKENS = {"test", "testing"}

MASK_PATTERN = re.compile(
    r"(^|[/_.-])"
    r"(masks?|annotations?|ground[_-]?truth|groundtruth|gt|"
    r"segmentations?|labels?)"
    r"(?=$|[/_.-])",
    flags=re.IGNORECASE,
)

NORMALIZE_SUFFIX_PATTERNS = (
    r"[_-]masks?$",
    r"[_-]gt$",
    r"[_-]labels?$",
    r"[_-]annotations?$",
    r"[_-]segmentations?$",
    r"[_-]ground[_-]?truth$",
    r"[_-]groundtruth$",
)

DECISION_LOCKED = "PICCOLO_OFFICIAL_TEST_STRUCTURE_LOCKED"
DECISION_RESOLUTION = "PICCOLO_STRUCTURE_RESOLUTION_REQUIRED"
DECISION_INVALID = "PICCOLO_ACQUISITION_INVALID"


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
    validate_sha(PROTOCOL, EXPECTED_PROTOCOL_SHA256, "R05B1 protocol")
    validate_sha(R05B0_LOCK, EXPECTED_R05B0_LOCK_SHA256, "R05B0 lock")

    lock = json.loads(R05B0_LOCK.read_text(encoding="utf-8"))
    if lock.get("decision") != EXPECTED_R05B0_DECISION:
        raise RuntimeError(
            f"Unexpected R05B0 decision: {lock.get('decision')}"
        )
    if bool(lock.get("piccolo_downloaded", True)):
        raise RuntimeError("R05B0 reports PICCOLO already downloaded.")
    if bool(lock.get("piccolo_image_pixels_opened", True)):
        raise RuntimeError("R05B0 reports PICCOLO image access.")
    if bool(lock.get("piccolo_mask_pixels_opened", True)):
        raise RuntimeError("R05B0 reports PICCOLO mask access.")

    return {
        "r05b0_lock_sha256": EXPECTED_R05B0_LOCK_SHA256,
        "r05b0_decision": lock.get("decision"),
    }


def normalize_relpath(name: str) -> str:
    return str(PurePosixPath(name.replace("\\", "/")))


def path_tokens(name: str):
    low = normalize_relpath(name).lower()
    return {
        t for t in re.split(r"[^a-z0-9]+", low)
        if t
    }


def split_from_name(name: str) -> str:
    tokens = path_tokens(name)
    if tokens & TEST_TOKENS:
        return "test"
    if tokens & VAL_TOKENS:
        return "validation"
    if tokens & TRAIN_TOKENS:
        return "train"
    return "unknown"


def is_mask_like(name: str) -> bool:
    """
    FIX1: operate on the normalized path string rather than tokenized pieces,
    so compound semantics such as ground_truth / ground-truth remain visible.
    """
    low = normalize_relpath(name).lower()
    return MASK_PATTERN.search(low) is not None


def normalized_stem(name: str) -> str:
    stem = Path(PurePosixPath(name).name).stem.lower()
    changed = True
    while changed:
        changed = False
        for pattern in NORMALIZE_SUFFIX_PATTERNS:
            new_stem = re.sub(pattern, "", stem, flags=re.IGNORECASE)
            if new_stem != stem:
                stem = new_stem
                changed = True
    return stem


def inventory_zip(path: Path):
    rows = []
    text_rows = []

    with zipfile.ZipFile(path, "r") as zf:
        infos = zf.infolist()
        for info in tqdm(
            infos,
            desc="Q1-R05B1 ZIP metadata inventory",
            unit="member",
            dynamic_ncols=True,
        ):
            if info.is_dir():
                continue

            rel = normalize_relpath(info.filename)
            ext = Path(PurePosixPath(rel).name).suffix.lower()
            image_like = ext in IMAGE_EXTS
            mask_like = image_like and is_mask_like(rel)

            rows.append({
                "relpath": rel,
                "source_type": "zip_member",
                "size_bytes": int(info.file_size),
                "compressed_size_bytes": int(info.compress_size),
                "crc32": f"{info.CRC:08x}",
                "extension": ext,
                "image_like": int(image_like),
                "mask_like": int(mask_like),
                "split_from_path": split_from_name(rel),
                "normalized_stem":
                    normalized_stem(rel) if image_like else "",
            })

            if ext in TEXT_EXTS and info.file_size <= MAX_TEXT_BYTES:
                try:
                    raw = zf.read(info)
                    text = raw.decode("utf-8", errors="ignore").lower()
                    text_rows.append({
                        "relpath": rel,
                        "extension": ext,
                        "size_bytes": int(info.file_size),
                        "content_read": 1,
                        "train_token": int("train" in text),
                        "validation_token": int("validation" in text),
                        "test_token": int("test" in text),
                        "patient_token": int("patient" in text),
                        "lesion_token": int("lesion" in text),
                    })
                except Exception:
                    text_rows.append({
                        "relpath": rel,
                        "extension": ext,
                        "size_bytes": int(info.file_size),
                        "content_read": 0,
                        "train_token": 0,
                        "validation_token": 0,
                        "test_token": 0,
                        "patient_token": 0,
                        "lesion_token": 0,
                    })

    receipt = {
        "input_type": "zip_archive",
        "input_path": str(path),
        "input_size_bytes": path.stat().st_size,
        "input_sha256": sha256_file(path),
        "members": len(rows),
    }
    return rows, text_rows, receipt


def inventory_directory(path: Path):
    rows = []
    text_rows = []
    files = [p for p in path.rglob("*") if p.is_file()]

    for p in tqdm(
        files,
        desc="Q1-R05B1 directory metadata inventory",
        unit="file",
        dynamic_ncols=True,
    ):
        rel = normalize_relpath(str(p.relative_to(path)))
        ext = p.suffix.lower()
        image_like = ext in IMAGE_EXTS
        mask_like = image_like and is_mask_like(rel)

        rows.append({
            "relpath": rel,
            "source_type": "directory_file",
            "size_bytes": int(p.stat().st_size),
            "compressed_size_bytes": "",
            "crc32": "",
            "extension": ext,
            "image_like": int(image_like),
            "mask_like": int(mask_like),
            "split_from_path": split_from_name(rel),
            "normalized_stem":
                normalized_stem(rel) if image_like else "",
        })

        if ext in TEXT_EXTS and p.stat().st_size <= MAX_TEXT_BYTES:
            try:
                text = p.read_text(
                    encoding="utf-8",
                    errors="ignore",
                ).lower()
                text_rows.append({
                    "relpath": rel,
                    "extension": ext,
                    "size_bytes": int(p.stat().st_size),
                    "content_read": 1,
                    "train_token": int("train" in text),
                    "validation_token": int("validation" in text),
                    "test_token": int("test" in text),
                    "patient_token": int("patient" in text),
                    "lesion_token": int("lesion" in text),
                })
            except Exception:
                pass

    h = hashlib.sha256()
    for r in sorted(rows, key=lambda x: x["relpath"]):
        h.update(
            f"{r['relpath']}\t{r['size_bytes']}\n".encode("utf-8")
        )

    receipt = {
        "input_type": "directory",
        "input_path": str(path),
        "input_size_bytes": sum(int(r["size_bytes"]) for r in rows),
        "input_sha256": None,
        "directory_metadata_fingerprint_sha256": h.hexdigest(),
        "members": len(rows),
    }
    return rows, text_rows, receipt


def classify_structure(rows):
    summary = []

    for split in ("train", "validation", "test", "unknown"):
        rr = [r for r in rows if r["split_from_path"] == split]
        image_ext = [r for r in rr if int(r["image_like"]) == 1]
        masks = [r for r in image_ext if int(r["mask_like"]) == 1]
        images = [r for r in image_ext if int(r["mask_like"]) == 0]

        summary.append({
            "split": split,
            "all_files": len(rr),
            "image_extension_files": len(image_ext),
            "mask_like_files": len(masks),
            "nonmask_image_like_files": len(images),
        })

    test = [r for r in rows if r["split_from_path"] == "test"]
    test_images = [
        r for r in test
        if int(r["image_like"]) == 1 and int(r["mask_like"]) == 0
    ]
    test_masks = [
        r for r in test
        if int(r["image_like"]) == 1 and int(r["mask_like"]) == 1
    ]
    return summary, test_images, test_masks


def pair_test(test_images, test_masks):
    image_by_stem = defaultdict(list)
    mask_by_stem = defaultdict(list)

    for r in test_images:
        image_by_stem[r["normalized_stem"]].append(r)
    for r in test_masks:
        mask_by_stem[r["normalized_stem"]].append(r)

    rows = []
    ambiguous = False

    for stem in sorted(set(image_by_stem) | set(mask_by_stem)):
        imgs = image_by_stem.get(stem, [])
        masks = mask_by_stem.get(stem, [])
        status = (
            "PAIRED"
            if len(imgs) == 1 and len(masks) == 1
            else "AMBIGUOUS"
        )
        ambiguous = ambiguous or status != "PAIRED"

        rows.append({
            "normalized_stem": stem,
            "image_count": len(imgs),
            "mask_count": len(masks),
            "image_relpath":
                imgs[0]["relpath"] if len(imgs) == 1 else "",
            "mask_relpath":
                masks[0]["relpath"] if len(masks) == 1 else "",
            "pair_status": status,
        })

    return rows, ambiguous


def grouping_audit(text_rows):
    return {
        "metadata_text_files": len(text_rows),
        "files_with_patient_token":
            sum(int(r["patient_token"]) for r in text_rows),
        "files_with_lesion_token":
            sum(int(r["lesion_token"]) for r in text_rows),
        "patient_grouping_resolved": False,
        "lesion_grouping_resolved": False,
        "note": (
            "Presence-only audit. Exact grouping may require a later "
            "metadata-specific parser without decoding images/masks."
        ),
    }


def preflight():
    upstream = validate_upstream()

    print("===== Q1-R05B1 PREFLIGHT =====")
    print(f"protocol_sha256={EXPECTED_PROTOCOL_SHA256}")
    print(f"r05b0_lock_sha256={EXPECTED_R05B0_LOCK_SHA256}")
    print(f"r05b0_decision={upstream['r05b0_decision']}")
    print("dataset=PICCOLO")
    print("frozen_subset=official_test_only")
    print(f"expected_test_images={EXPECTED_TEST_IMAGES}")
    print(f"suggested_raw_root={DEFAULT_RAW_ROOT}")
    print("formal_input_received=NO")
    print("image_pixels_decoded=NO")
    print("mask_pixels_decoded=NO")
    print("model_inference=NO")
    print("TTA=NO")
    print("predictor_fit=NO")
    print("PREFLIGHT_PASS")


def run(args):
    validate_upstream()

    input_path = args.input.resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)

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
        "R05B1 protocol copy",
    )

    if input_path.is_file() and zipfile.is_zipfile(input_path):
        rows, text_rows, receipt = inventory_zip(input_path)
    elif input_path.is_dir():
        rows, text_rows, receipt = inventory_directory(input_path)
    else:
        raise RuntimeError(
            "PICCOLO_ACQUISITION_INVALID: input is not a ZIP/directory."
        )

    receipt["provider_authorized_acquisition_user_asserted"] = bool(
        args.provider_authorized
    )
    receipt["image_pixels_decoded"] = False
    receipt["mask_pixels_decoded"] = False

    inventory_path = build_dir / "package_inventory.csv"
    write_csv(
        inventory_path,
        rows,
        [
            "relpath",
            "source_type",
            "size_bytes",
            "compressed_size_bytes",
            "crc32",
            "extension",
            "image_like",
            "mask_like",
            "split_from_path",
            "normalized_stem",
        ],
    )

    metadata_path = build_dir / "metadata_text_inventory.csv"
    write_csv(
        metadata_path,
        text_rows,
        [
            "relpath",
            "extension",
            "size_bytes",
            "content_read",
            "train_token",
            "validation_token",
            "test_token",
            "patient_token",
            "lesion_token",
        ],
    )

    split_summary, test_images, test_masks = classify_structure(rows)
    split_path = build_dir / "split_structure_summary.csv"
    write_csv(
        split_path,
        split_summary,
        [
            "split",
            "all_files",
            "image_extension_files",
            "mask_like_files",
            "nonmask_image_like_files",
        ],
    )

    pairing_rows, ambiguous = pair_test(test_images, test_masks)
    pairing_path = build_dir / "official_test_pairing.csv"
    write_csv(
        pairing_path,
        pairing_rows,
        [
            "normalized_stem",
            "image_count",
            "mask_count",
            "image_relpath",
            "mask_relpath",
            "pair_status",
        ],
    )

    group_path = build_dir / "grouping_metadata_audit.json"
    write_json(group_path, grouping_audit(text_rows))

    receipt_path = build_dir / "acquisition_receipt.json"
    write_json(receipt_path, receipt)

    info = {
        "image_pixels_decoded": False,
        "mask_pixels_decoded": False,
        "model_inference_run": False,
        "tta_run": False,
        "predictor_fit": False,
        "target_probabilities_generated": False,
        "target_dice_computed": False,
        "directory_image_bytes_read": False,
    }
    info_path = build_dir / "information_boundary_audit.json"
    write_json(info_path, info)

    test_count = len(test_images)
    mask_count = len(test_masks)
    paired_count = sum(
        r["pair_status"] == "PAIRED"
        for r in pairing_rows
    )

    if not args.provider_authorized:
        decision = DECISION_INVALID
        reason = "provider-authorized acquisition not asserted"
    elif not rows:
        decision = DECISION_INVALID
        reason = "empty package"
    elif test_count == 0:
        decision = DECISION_RESOLUTION
        reason = "test split not structurally resolved"
    elif test_count != EXPECTED_TEST_IMAGES:
        decision = DECISION_RESOLUTION
        reason = (
            f"test images={test_count}, expected={EXPECTED_TEST_IMAGES}"
        )
    elif mask_count != EXPECTED_TEST_IMAGES:
        decision = DECISION_RESOLUTION
        reason = (
            f"test masks={mask_count}, expected={EXPECTED_TEST_IMAGES}"
        )
    elif ambiguous or paired_count != EXPECTED_TEST_IMAGES:
        decision = DECISION_RESOLUTION
        reason = "test pairing not exact one-to-one"
    else:
        decision = DECISION_LOCKED
        reason = "official test structure and pairing resolved"

    decision_path = build_dir / "decision.txt"
    decision_path.write_text(decision + "\n", encoding="utf-8")

    log = f"""===== Q1-R05B1 PICCOLO ACQUISITION + STRUCTURE/QC =====
Script version: {VERSION}
Build: {BUILD}

Acquisition:
  input={input_path}
  input_type={receipt['input_type']}
  provider_authorized={'YES' if args.provider_authorized else 'NO'}

Structure:
  package_files={len(rows)}
  resolved_test_images={test_count}
  resolved_test_masks={mask_count}
  exact_paired_test_cases={paired_count}
  expected_test_images={EXPECTED_TEST_IMAGES}

Information boundary:
  image_pixels_decoded=NO
  mask_pixels_decoded=NO
  model_inference=NO
  TTA=NO
  predictor_fit=NO
  target_Dice=NO

Decision:
  {decision}
Reason:
  {reason}
"""
    run_log_path = build_dir / "run_log.txt"
    run_log_path.write_text(log, encoding="utf-8")

    artifacts = {
        "protocol_copy": protocol_copy,
        "acquisition_receipt": receipt_path,
        "package_inventory": inventory_path,
        "metadata_text_inventory": metadata_path,
        "split_structure_summary": split_path,
        "official_test_pairing": pairing_path,
        "grouping_metadata_audit": group_path,
        "information_boundary_audit": info_path,
        "decision": decision_path,
        "run_log": run_log_path,
    }

    lock = {
        "script_version": VERSION,
        "build": BUILD,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "r05b0_lock_sha256": EXPECTED_R05B0_LOCK_SHA256,
        "input_type": receipt["input_type"],
        "input_path": str(input_path),
        "input_sha256": receipt.get("input_sha256"),
        "directory_metadata_fingerprint_sha256":
            receipt.get("directory_metadata_fingerprint_sha256"),
        "provider_authorized_acquisition_user_asserted":
            bool(args.provider_authorized),
        "expected_test_images": EXPECTED_TEST_IMAGES,
        "resolved_test_images": test_count,
        "resolved_test_masks": mask_count,
        "resolved_exact_pairs": paired_count,
        "image_pixels_decoded": False,
        "mask_pixels_decoded": False,
        "model_inference_run": False,
        "tta_run": False,
        "predictor_fit": False,
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

    lock_path = build_dir / "Q1_R05B1_PICCOLO_STRUCTURE_LOCK.json"
    write_json(lock_path, lock)
    lock_sha = sha256_file(lock_path)

    for name, meta in lock["artifacts"].items():
        p = build_dir / meta["relative_path"]
        if sha256_file(p) != meta["sha256"]:
            raise RuntimeError(f"Artifact changed before commit: {name}")

    build_dir.rename(args.output_dir)

    print()
    print((args.output_dir / "run_log.txt").read_text(encoding="utf-8"))
    print(
        "Q1-R05B1 LOCK:",
        args.output_dir / "Q1_R05B1_PICCOLO_STRUCTURE_LOCK.json",
    )
    print("Q1-R05B1 LOCK SHA256:", lock_sha)


def self_test():
    assert EXPECTED_TEST_IMAGES == 333

    assert split_from_name("root/test/images/a.jpg") == "test"
    assert split_from_name("root/validation/images/a.jpg") == "validation"
    assert split_from_name("root/train/images/a.jpg") == "train"
    assert split_from_name("root/images/a.jpg") == "unknown"

    positives = [
        "test/masks/a.png",
        "test/mask/a.png",
        "test/ground_truth/a.png",
        "test/ground-truth/a.png",
        "test/groundtruth/a.png",
        "test/gt/a.png",
        "test/annotations/a.png",
        "test/segmentation/a.png",
        "test/labels/a.png",
        "test/images/a_mask.png",
    ]
    for value in positives:
        assert is_mask_like(value), value

    negatives = [
        "test/images/a.png",
        "test/images/target.png",
        "test/images/segment.png",
    ]
    for value in negatives:
        assert not is_mask_like(value), value

    assert normalized_stem("x/a_mask.png") == "a"
    assert normalized_stem("x/a_segmentation.png") == "a"
    assert normalized_stem("x/a_ground_truth.png") == "a"
    assert normalized_stem("x/a.png") == "a"

    # Static design check: no PIL/cv2 image decoder imported.
    import sys as _sys
    assert "cv2" not in globals()
    assert "PIL" not in globals()

    print("SPLIT_RESOLVER_TEST_PASS")
    print("MASK_NAME_CLASSIFIER_TEST_PASS")
    print("PAIRING_STEM_NORMALIZER_TEST_PASS")
    print("NO_PIXEL_DECODER_TEST_PASS")
    print("SELF_TEST_PASS")


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Q1-R05B1 FIX1: metadata-only PICCOLO acquisition/structure QC."
        )
    )
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    p.add_argument(
        "--input",
        type=Path,
        help="Provider-supplied PICCOLO ZIP or extracted directory.",
    )
    p.add_argument(
        "--provider-authorized",
        action="store_true",
        help=(
            "Assert that the package was obtained through the provider-"
            "authorized access route for the submitted request."
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

    if args.input is None:
        raise SystemExit("--input is required for formal Q1-R05B1.")

    try:
        run(args)
    except Exception:
        traceback.print_exc()
        raise

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
