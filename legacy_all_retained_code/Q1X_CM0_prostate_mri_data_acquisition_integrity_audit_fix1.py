#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1X_CM0_prostate_mri_data_acquisition_integrity_audit_fix1.py

Cross-modality Q1 enhancement experiment: CM0 data acquisition + archive integrity audit.

Scientific role:
- freeze the second-task dataset pair before any model training;
- verify the exact official archive identities;
- inventory archive structure without extracting, decoding GT, or training;
- create a reproducible CM0 lock only after both official archives pass.

Frozen task:
  Source development: Prostate158 training data, T2W only
  External confirmation: PROMISE12 training data, T2W whole-gland GT
  Task: binary whole-gland prostate segmentation
  Primary TTA action: TENT1 only
  Safety formulation: reuse SafeTTA design; no representation search

This script does NOT:
- train a segmentation model;
- inspect voxel values;
- construct whole-gland labels;
- fit a safety model;
- reveal/use PROMISE12 GT for any decision;
- tune target-side parameters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path


VERSION = "2026-09-04-Q1X-CM0-v1-fix1"
BUILD = "Q1X_CM0_PROSTATE_MRI_DATA_ACQUISITION_INTEGRITY_AUDIT_FIX1"

DEFAULT_ROOT = Path(r"F:\MEDSEG_SAFETTA\cross_modality_prostate_mri")
DEFAULT_OUTPUT = (
    Path(r"F:\MEDSEG_SAFETTA\outputs")
    / "Q1X_CM0_prostate_mri_data_acquisition_integrity_audit_fix1_v1"
)

PROSTATE158 = {
    "dataset": "Prostate158",
    "role": "SOURCE_DEVELOPMENT",
    "record": "https://zenodo.org/records/6481141",
    "doi": "10.5281/zenodo.6481141",
    "filename": "prostate158_train.zip",
    "md5": "a0d5361b7c4cbf3f17f7ccb9a2e1739c",
    "expected_size_note": "approximately 2.8 GB",
}

PROMISE12 = {
    "dataset": "PROMISE12",
    "role": "INDEPENDENT_EXTERNAL_CONFIRMATION",
    "record": "https://zenodo.org/records/8026660",
    "doi": "10.5281/zenodo.8026660",
    "filename": "training_data.zip",
    "md5": "f6d23994117c989daf07e5291edd0aea",
    "expected_size_note": "approximately 323.7 MB",
}

FROZEN_PROTOCOL = {
    "source_dataset": "Prostate158 training",
    "source_input": "T2-weighted MRI only",
    "source_target": (
        "binary whole-gland prostate mask; exact label mapping NOT assumed "
        "until CM1 schema audit"
    ),
    "external_dataset": "PROMISE12 training set with public reference masks",
    "external_input": "transversal T2-weighted MRI",
    "external_target": "binary whole-gland prostate",
    "tta_primary": "TENT1 one-step episodic entropy minimization",
    "safety_representation": (
        "same SafeTTA formulation: frozen DINOv2-base + SOURCE-mask FG/BG "
        "conditioning + PCA64 + M2 + class-balanced logistic head"
    ),
    "harm_definition": "DeltaDice <= -0.02",
    "benefit_definition": "DeltaDice >= +0.02",
    "external_primary_endpoint": "patient-clustered AUROC",
    "success_rule": "external AUROC 95% clustered bootstrap CI lower bound > 0.50",
    "target_tuning": False,
    "new_representation_search": False,
    "promises12_gt_use_before_score_lock": False,
}


def md5_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def classify_member(name: str) -> str:
    low = name.lower()
    base = Path(low).name

    if low.endswith("/"):
        return "directory"
    if base.endswith(".nii.gz") or base.endswith(".nii"):
        if "anatom" in base or "seg" in base or "mask" in base or "label" in base:
            return "nifti_annotation_candidate"
        if "t2" in base:
            return "nifti_t2_candidate"
        if "adc" in base:
            return "nifti_adc_candidate"
        if "dwi" in base:
            return "nifti_dwi_candidate"
        return "nifti_other"
    if base.endswith(".mhd"):
        if "segmentation" in base or "seg" in base or "label" in base:
            return "mhd_annotation_candidate"
        return "mhd_image_candidate"
    if base.endswith(".raw"):
        return "raw_companion"
    if base.endswith(".csv"):
        return "csv"
    if base.endswith(".txt") or base.endswith(".md"):
        return "text"
    return "other"


def inventory_zip(path: Path) -> dict:
    with zipfile.ZipFile(path, "r") as zf:
        infos = zf.infolist()
        names = [i.filename for i in infos]
        counts = Counter(classify_member(n) for n in names)

        top_levels = Counter()
        basenames = Counter()
        for n in names:
            clean = n.replace("\\", "/").strip("/")
            if not clean:
                continue
            top_levels[clean.split("/", 1)[0]] += 1
            if not n.endswith("/"):
                basenames[Path(clean).name] += 1

        # Read-only filename-pattern audit. No image/GT payload is decoded.
        prostate_case_tokens = sorted(set(
            m.group(0)
            for n in names
            for m in [re.search(r"(?:case|patient|prostate)[_\-]?\d+", n, re.I)]
            if m
        ))

        promise_case_tokens = sorted(set(
            m.group(0)
            for n in names
            for m in [re.search(r"(?:case|trainingdata)[_\-]?\d+", n, re.I)]
            if m
        ))

        sample_members = [n for n in names if not n.endswith("/")][:40]

        return {
            "zip_member_count": len(infos),
            "uncompressed_bytes": int(sum(i.file_size for i in infos)),
            "compressed_bytes_inside_zip": int(sum(i.compress_size for i in infos)),
            "member_type_counts": dict(sorted(counts.items())),
            "top_level_entries": dict(top_levels.most_common(30)),
            "distinct_basenames": len(basenames),
            "prostate_like_case_tokens_count": len(prostate_case_tokens),
            "promise_like_case_tokens_count": len(promise_case_tokens),
            "sample_members_first_40": sample_members,
            "zip_test": zf.testzip(),  # None means CRC structure passes
        }


def audit_one(spec: dict, path: Path) -> dict:
    print(f"\n===== {spec['dataset']} =====")
    print("role:", spec["role"])
    print("expected file:", path)
    print("official record:", spec["record"])
    print("official DOI:", spec["doi"])
    print("official MD5:", spec["md5"])

    out = {
        "spec": spec,
        "path": str(path),
        "exists": path.is_file(),
        "integrity_pass": False,
        "md5": None,
        "sha256": None,
        "size_bytes": None,
        "inventory": None,
    }

    if not path.is_file():
        print("STATUS=MISSING")
        return out

    out["size_bytes"] = int(path.stat().st_size)
    print("size bytes:", out["size_bytes"])

    print("hashing MD5...")
    out["md5"] = md5_file(path)
    print("MD5:", out["md5"])

    if out["md5"].lower() != spec["md5"].lower():
        print("MD5_GATE=FAIL")
        return out

    print("MD5_GATE=PASS")

    print("hashing SHA256...")
    out["sha256"] = sha256_file(path)
    print("SHA256:", out["sha256"])

    try:
        inv = inventory_zip(path)
    except zipfile.BadZipFile as e:
        print("ZIP_GATE=FAIL", repr(e))
        return out

    out["inventory"] = inv
    print("ZIP members:", inv["zip_member_count"])
    print("member types:", inv["member_type_counts"])
    print("ZIP CRC test:", "PASS" if inv["zip_test"] is None else f"FAIL:{inv['zip_test']}")

    if inv["zip_test"] is not None:
        return out

    out["integrity_pass"] = True
    print("ARCHIVE_INTEGRITY=PASS")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=(
            "Frozen new branch root. Raw archives are expected under "
            "<root>/data/raw/Prostate158 and <root>/data/raw/PROMISE12."
        ),
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    args = ap.parse_args()

    root = args.root
    p158_dir = root / "data" / "raw" / "Prostate158"
    p12_dir = root / "data" / "raw" / "PROMISE12"
    p158_path = p158_dir / PROSTATE158["filename"]
    p12_path = p12_dir / PROMISE12["filename"]

    print("===== Q1X CM0 PROSTATE MRI CROSS-MODALITY DATA AUDIT =====")
    print("STATUS=PRETRAINING_DATA_ACQUISITION_AND_ARCHIVE_AUDIT")
    print("MODEL_TRAINING=NO")
    print("GT_VOXEL_DECODING=NO")
    print("WHOLE_GLAND_LABEL_MAPPING=NOT_YET_ASSUMED")
    print("TARGET_TUNING=NO")
    print("NEW_REPRESENTATION_SEARCH=NO")
    print("ROOT=", root)

    # Creating empty destination directories is operational only; no data are modified.
    p158_dir.mkdir(parents=True, exist_ok=True)
    p12_dir.mkdir(parents=True, exist_ok=True)

    print("\n===== FROZEN CROSS-MODALITY PROTOCOL =====")
    for k, v in FROZEN_PROTOCOL.items():
        print(f"{k} = {v}")

    r1 = audit_one(PROSTATE158, p158_path)
    r2 = audit_one(PROMISE12, p12_path)

    both_present = r1["exists"] and r2["exists"]
    both_pass = r1["integrity_pass"] and r2["integrity_pass"]

    print("\n===== CM0 DECISION =====")
    if not both_present:
        print("Decision=DATA_DOWNLOAD_REQUIRED")
        print("Place the official files exactly at:")
        print(" ", p158_path)
        print(" ", p12_path)
        print("Do not rename or extract before the first integrity audit.")
        print("Prostate158 official page:", PROSTATE158["record"])
        print("PROMISE12 official page:", PROMISE12["record"])
        print("PASS=NO")
        return

    if not both_pass:
        print("Decision=STOP_ARCHIVE_IDENTITY_OR_INTEGRITY_FAILURE")
        print("Do not extract, repair, or substitute archives.")
        print("PASS=NO")
        raise RuntimeError("CM0 archive integrity gate failed.")

    if args.output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists: {args.output_dir}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=False)

    lock = {
        "status": "PASS",
        "decision": "CROSS_MODALITY_DATASETS_LOCKED_READY_FOR_CM1_SCHEMA_AUDIT",
        "version": VERSION,
        "build": BUILD,
        "branch_root": str(root),
        "frozen_protocol": FROZEN_PROTOCOL,
        "datasets": {
            "Prostate158": r1,
            "PROMISE12": r2,
        },
        "information_boundary": {
            "model_training": False,
            "gt_voxel_decoding": False,
            "whole_gland_mapping_assumed": False,
            "target_tuning": False,
            "representation_search": False,
        },
        "next_stage": (
            "CM1_READ_ONLY_SCHEMA_AND_LABEL_ENCODING_AUDIT_BEFORE_EXTRACTION_PIPELINE"
        ),
    }

    lock_path = args.output_dir / "CM0_PROSTATE_MRI_DATASET_LOCK.json"
    lock_path.write_text(
        json.dumps(lock, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Decision= CROSS_MODALITY_DATASETS_LOCKED_READY_FOR_CM1_SCHEMA_AUDIT")
    print("LOCK=", lock_path)
    print("LOCK SHA256=", sha256_file(lock_path))
    print("PASS")


if __name__ == "__main__":
    main()
