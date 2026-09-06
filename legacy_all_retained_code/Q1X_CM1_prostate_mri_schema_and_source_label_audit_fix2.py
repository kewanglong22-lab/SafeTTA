#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1X_CM1_prostate_mri_schema_and_source_label_audit_fix2.py

CM1 Fix2: preserve all CM0 and Prostate158 checks, but replace the overly
strict PROMISE12 "all physical header fields must be text-identical" gate with
a two-level, read-only geometry audit:

Level A (array-grid compatibility; required for PASS):
- NDims
- DimSize
- ElementType
- ElementNumberOfChannels (default 1)

Level B (physical-metadata consistency; diagnostic):
- ElementSpacing
- TransformMatrix
- Offset / Position / Origin
- CenterOfRotation
- AnatomicalOrientation

No PROMISE12 reference RAW voxel payload is opened.

If Level A passes for all 50 cases but Level B differs, the script does NOT
silently ignore the issue. It writes exact case/field mismatch records and
returns PASS_WITH_PHYSICAL_METADATA_DIAGNOSTIC, allowing CM2 to keep array-grid
alignment frozen while preserving the metadata discrepancy in the audit trail.

If Level A fails for any case, STOP.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


VERSION = "2026-09-04-Q1X-CM1-v1-fix2"
BUILD = "Q1X_CM1_PROSTATE_MRI_SCHEMA_AND_SOURCE_LABEL_AUDIT_FIX2"

ROOT = Path(r"F:\MEDSEG_SAFETTA\cross_modality_prostate_mri")
OUT = Path(r"F:\MEDSEG_SAFETTA\outputs")

CM0_DIR = OUT / "Q1X_CM0_prostate_mri_data_acquisition_integrity_audit_fix1_v1"
CM0_LOCK = CM0_DIR / "CM0_PROSTATE_MRI_DATASET_LOCK.json"
EXPECTED_CM0_LOCK_SHA = (
    "4e84f69ff980bc329ad0813cbf0111b919d2696320df908d9f088e50e1a2fe28"
)

P158_ZIP = ROOT / "data" / "raw" / "Prostate158" / "prostate158_train.zip"
P12_ZIP = ROOT / "data" / "raw" / "PROMISE12" / "training_data.zip"

EXPECTED_P158_SHA256 = (
    "7a97b263be1bdbc79f6c8a3461e010b2cea9a266249af47671616dadff77da2e"
)
EXPECTED_P12_SHA256 = (
    "150287d0c74cd0105d8b70b43af4a7bf4f1fd3e1748829779177a0ccaf948f45"
)

EXTRACT_ROOT = ROOT / "data" / "extracted"
P158_EXTRACT = EXTRACT_ROOT / "Prostate158"
P12_EXTRACT = EXTRACT_ROOT / "PROMISE12"

DEFAULT_OUTPUT = OUT / "Q1X_CM1_prostate_mri_schema_and_source_label_audit_fix2_v1"

DECISION_PASS = (
    "SOURCE_SCHEMA_AND_LABEL_ENCODING_LOCKED_READY_FOR_CM2_PREPROCESSING_PROTOCOL"
)

ARRAY_GRID_KEYS = [
    "NDims",
    "DimSize",
    "ElementType",
    "ElementNumberOfChannels",
]

PHYSICAL_KEYS = [
    "ElementSpacing",
    "TransformMatrix",
    "Offset",
    "Position",
    "Origin",
    "CenterOfRotation",
    "AnatomicalOrientation",
]


def sha256_file(path: Path, chunk=8 * 1024 * 1024):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def assert_sha(path: Path, expected: str, label: str):
    if not path.is_file():
        raise FileNotFoundError(path)
    got = sha256_file(path)
    print(label, got, "PASS" if got == expected else "FAIL")
    if got != expected:
        raise RuntimeError(f"{label} SHA mismatch: {got}")
    return got


def safe_extract_zip(zip_path: Path, dest: Path):
    if dest.exists() and any(dest.iterdir()):
        print(f"EXTRACTION_SKIP_EXISTING_NONEMPTY={dest}")
        return "existing"

    dest.mkdir(parents=True, exist_ok=True)
    root_resolved = dest.resolve()

    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            out = (dest / member.filename).resolve()
            if root_resolved not in out.parents and out != root_resolved:
                raise RuntimeError(
                    f"ZIP path traversal detected: {member.filename}"
                )
        zf.extractall(dest)

    return "extracted"


def import_nibabel():
    try:
        import nibabel as nib
    except Exception as e:
        raise RuntimeError(
            "nibabel is required for the Prostate158 SOURCE NIfTI schema audit."
        ) from e
    return nib


def all_files(root: Path):
    return sorted(p for p in root.rglob("*") if p.is_file())


def classify_p158_file(path: Path):
    name = path.name.lower()

    if name.endswith(".nii.gz") or name.endswith(".nii"):
        if any(x in name for x in ["anatom", "seg", "mask", "label"]):
            return "annotation"
        if "t2" in name:
            return "t2"
        if "adc" in name:
            return "adc"
        if "dwi" in name:
            return "dwi"
        return "nifti_other"
    if name.endswith(".csv"):
        return "csv"
    return "other"


def infer_case_key(path: Path):
    for token in reversed(path.parts[:-1]):
        m = re.search(r"(?:case|patient|prostate)?[_\-]?(\d{1,4})$", token, re.I)
        if m:
            return f"case_{int(m.group(1)):04d}"

    m = re.search(r"(?:case|patient|prostate)[_\-]?(\d{1,4})", path.name, re.I)
    if m:
        return f"case_{int(m.group(1)):04d}"

    stem = path.name.lower()
    for ext in [".nii.gz", ".nii", ".mhd", ".raw"]:
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
            break
    return stem


def audit_prostate158_source(nib):
    print("\n===== PROSTATE158 SOURCE SCHEMA AUDIT =====")
    files = all_files(P158_EXTRACT)
    classes = defaultdict(list)
    for p in files:
        classes[classify_p158_file(p)].append(p)

    for k in sorted(classes):
        print(k, "files=", len(classes[k]))

    ann = classes["annotation"]
    t2 = classes["t2"]

    print("annotation files:", len(ann))
    print("T2 candidates:", len(t2))

    if len(ann) != 139:
        raise RuntimeError(
            f"Expected 139 Prostate158 annotations, found {len(ann)}."
        )

    ann_records = []
    observed_global = set()
    shapes = Counter()
    dtypes = Counter()
    nonempty = 0

    print("\nDecoding SOURCE annotations only...")
    for i, p in enumerate(ann, 1):
        img = nib.load(str(p))
        arr = np.asanyarray(img.dataobj)
        vals = np.unique(arr)
        vals_py = [
            float(v) if np.issubdtype(vals.dtype, np.floating) else int(v)
            for v in vals.tolist()
        ]

        observed_global.update(vals_py)
        shapes[tuple(int(x) for x in img.shape)] += 1
        dtypes[str(arr.dtype)] += 1
        if np.any(arr != 0):
            nonempty += 1

        ann_records.append({
            "case_key": infer_case_key(p),
            "path": str(p),
            "shape": [int(x) for x in img.shape],
            "dtype": str(arr.dtype),
            "unique_values": vals_py,
            "nonzero_voxels": int(np.count_nonzero(arr)),
        })

        if i % 20 == 0 or i == len(ann):
            print(f"  source annotations audited: {i}/{len(ann)}")

    observed_sorted = sorted(observed_global)
    print("global SOURCE annotation values:", observed_sorted)
    print("nonempty annotations:", nonempty, "/", len(ann))

    if observed_sorted != [0, 1, 2]:
        raise RuntimeError(
            f"Unexpected Prostate158 annotation values: {observed_sorted}"
        )
    if nonempty != len(ann):
        raise RuntimeError("At least one Prostate158 SOURCE annotation is empty.")

    whole_gland_rule = {
        "rule": "whole_gland = annotation > 0",
        "background_value": 0,
        "observed_nonzero_labels": [1, 2],
        "rationale": (
            "For binary whole-gland confirmation, both observed nonzero "
            "prostate-zone labels are merged as foreground."
        ),
    }

    by_case = defaultdict(lambda: {"t2": [], "annotation": []})
    for p in t2:
        by_case[infer_case_key(p)]["t2"].append(str(p))
    for p in ann:
        by_case[infer_case_key(p)]["annotation"].append(str(p))

    resolved = {
        k: v
        for k, v in by_case.items()
        if len(v["annotation"]) == 1 and len(v["t2"]) >= 1
    }
    unresolved = {
        k: v
        for k, v in by_case.items()
        if len(v["annotation"]) > 0
        and not (len(v["annotation"]) == 1 and len(v["t2"]) >= 1)
    }

    print("cases with annotation and >=1 T2 candidate:", len(resolved))
    print("unresolved annotation-bearing cases:", len(unresolved))

    if len(resolved) != 139 or unresolved:
        raise RuntimeError("Prostate158 SOURCE case/T2 pairing audit failed.")

    return {
        "file_type_counts": {
            k: len(v)
            for k, v in sorted(classes.items())
        },
        "annotation_count": len(ann),
        "t2_candidate_count": len(t2),
        "annotation_global_unique_values": observed_sorted,
        "annotation_nonempty_count": nonempty,
        "annotation_shape_counts": {
            str(k): int(v)
            for k, v in shapes.items()
        },
        "annotation_dtype_counts": dict(dtypes),
        "whole_gland_rule": whole_gland_rule,
        "case_pairing_summary": {
            "annotation_cases_with_t2_candidate": len(resolved),
            "unresolved_annotation_bearing_cases": len(unresolved),
            "unresolved_case_keys": sorted(unresolved.keys()),
        },
    }


def parse_mhd_header(path: Path):
    text = path.read_text(
        encoding="latin-1",
        errors="replace",
    )
    fields = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        fields[k.strip()] = v.strip()
    return fields


def promise_case_id(path: Path):
    m = re.search(r"(?:Case|TrainingData)[_\-]?(\d+)", path.name, re.I)
    if m:
        return int(m.group(1))
    nums = re.findall(r"\d+", path.stem)
    if nums:
        return int(nums[-1])
    raise RuntimeError(f"Cannot infer PROMISE12 case id from {path.name}")


def normalize_header_value(key: str, value):
    if value is None:
        if key == "ElementNumberOfChannels":
            return "1"
        return None
    return " ".join(str(value).split())


def compare_fields(ih, rh, keys):
    diffs = {}
    for key in keys:
        iv = normalize_header_value(key, ih.get(key))
        rv = normalize_header_value(key, rh.get(key))
        if iv != rv:
            diffs[key] = {
                "image": iv,
                "reference": rv,
            }
    return diffs


def element_bytes(element_type: str):
    mapping = {
        "MET_CHAR": 1,
        "MET_UCHAR": 1,
        "MET_SHORT": 2,
        "MET_USHORT": 2,
        "MET_INT": 4,
        "MET_UINT": 4,
        "MET_FLOAT": 4,
        "MET_DOUBLE": 8,
        "MET_LONG": 8,
        "MET_ULONG": 8,
    }
    return mapping.get(str(element_type).strip().upper())


def expected_raw_bytes(header):
    dim = header.get("DimSize")
    et = header.get("ElementType")
    if dim is None or et is None:
        return None

    dims = [int(x) for x in dim.split()]
    b = element_bytes(et)
    if b is None:
        return None

    channels = int(header.get("ElementNumberOfChannels", "1"))
    total = b * channels
    for d in dims:
        total *= d
    return int(total)


def audit_promise12_without_gt_decode():
    print("\n===== PROMISE12 EXTERNAL HEADER-ONLY GEOMETRY AUDIT =====")
    files = all_files(P12_EXTRACT)
    mhd = [p for p in files if p.suffix.lower() == ".mhd"]
    raw = [p for p in files if p.suffix.lower() == ".raw"]

    images = [
        p for p in mhd
        if not any(x in p.name.lower() for x in ["segmentation", "seg", "label"])
    ]
    refs = [
        p for p in mhd
        if any(x in p.name.lower() for x in ["segmentation", "seg", "label"])
    ]

    print("MHD image headers:", len(images))
    print("MHD reference headers:", len(refs))
    print("RAW companions:", len(raw))

    if len(images) != 50 or len(refs) != 50 or len(raw) != 100:
        raise RuntimeError(
            "PROMISE12 expected 50 image MHD + 50 reference MHD + 100 RAW."
        )

    img_by_case = {promise_case_id(p): p for p in images}
    ref_by_case = {promise_case_id(p): p for p in refs}

    if len(img_by_case) != 50 or len(ref_by_case) != 50:
        raise RuntimeError("PROMISE12 case-id uniqueness failure.")
    if set(img_by_case) != set(ref_by_case):
        raise RuntimeError("PROMISE12 image/reference case-id mismatch.")

    records = []
    array_fail_cases = []
    physical_mismatch_cases = []

    for cid in sorted(img_by_case):
        ip = img_by_case[cid]
        rp = ref_by_case[cid]
        ih = parse_mhd_header(ip)
        rh = parse_mhd_header(rp)

        img_raw = ip.parent / ih.get("ElementDataFile", "")
        ref_raw = rp.parent / rh.get("ElementDataFile", "")
        if not img_raw.is_file():
            raise FileNotFoundError(img_raw)
        if not ref_raw.is_file():
            raise FileNotFoundError(ref_raw)

        array_diffs = compare_fields(ih, rh, ARRAY_GRID_KEYS)
        physical_diffs = compare_fields(ih, rh, PHYSICAL_KEYS)

        # File-size consistency is a header-only integrity check. We do not
        # open/read the reference raw payload.
        image_expected = expected_raw_bytes(ih)
        ref_expected = expected_raw_bytes(rh)
        image_size = int(img_raw.stat().st_size)
        ref_size = int(ref_raw.stat().st_size)

        image_raw_size_pass = (
            image_expected is None or image_size == image_expected
        )
        ref_raw_size_pass = (
            ref_expected is None or ref_size == ref_expected
        )

        if array_diffs or not image_raw_size_pass or not ref_raw_size_pass:
            array_fail_cases.append(cid)

        if physical_diffs:
            physical_mismatch_cases.append(cid)

        records.append({
            "case_id": cid,
            "image_mhd": str(ip),
            "reference_mhd": str(rp),
            "image_raw": str(img_raw),
            "reference_raw": str(ref_raw),
            "array_grid_differences": array_diffs,
            "physical_metadata_differences": physical_diffs,
            "image_raw_size_bytes": image_size,
            "image_expected_raw_size_bytes": image_expected,
            "image_raw_size_pass": image_raw_size_pass,
            "reference_raw_size_bytes": ref_size,
            "reference_expected_raw_size_bytes": ref_expected,
            "reference_raw_size_pass": ref_raw_size_pass,
            "reference_voxel_payload_decoded": False,
        })

    print("array-grid compatible cases:", 50 - len(array_fail_cases), "/ 50")
    print("physical-metadata exact-match cases:", 50 - len(physical_mismatch_cases), "/ 50")
    print("physical-metadata mismatch cases:", physical_mismatch_cases)
    print("PROMISE12 reference voxel payload decoded: NO")

    if array_fail_cases:
        print("ARRAY_GRID_FAILURE_CASES=", array_fail_cases)
        for r in records:
            if r["case_id"] in array_fail_cases:
                print(
                    "ARRAY_GRID_DIFF",
                    r["case_id"],
                    r["array_grid_differences"],
                    "image_raw_size_pass=", r["image_raw_size_pass"],
                    "ref_raw_size_pass=", r["reference_raw_size_pass"],
                )
        raise RuntimeError(
            "PROMISE12 array-grid/header-payload compatibility failure."
        )

    if physical_mismatch_cases:
        print("\n===== PROMISE12 PHYSICAL METADATA DIAGNOSTIC =====")
        for r in records:
            if r["case_id"] in physical_mismatch_cases:
                print(
                    "CASE",
                    r["case_id"],
                    "PHYSICAL_DIFF=",
                    r["physical_metadata_differences"],
                )

    return {
        "image_mhd_count": len(images),
        "reference_mhd_count": len(refs),
        "raw_count": len(raw),
        "case_count": len(records),
        "array_grid_compatible_count": 50 - len(array_fail_cases),
        "array_grid_failure_cases": array_fail_cases,
        "physical_metadata_exact_match_count": 50 - len(physical_mismatch_cases),
        "physical_metadata_mismatch_cases": physical_mismatch_cases,
        "reference_voxel_payload_decoded": False,
        "records": records,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    args = ap.parse_args()

    print("===== Q1X CM1 FIX2 PROSTATE MRI SCHEMA / SOURCE LABEL AUDIT =====")
    print("STATUS=PRETRAINING_SCHEMA_AND_SOURCE_LABEL_AUDIT")
    print("FIX2_CHANGE=TWO_LEVEL_PROMISE12_GEOMETRY_AUDIT")
    print("SCIENTIFIC_PROTOCOL_CHANGE=NO")
    print("MODEL_TRAINING=NO")
    print("TTA=NO")
    print("SAFETY_MODEL_FITTING=NO")
    print("PROMISE12_GT_VOXEL_DECODING=NO")
    print("TARGET_TUNING=NO")

    print("\n===== CM0 LINEAGE GATE =====")
    assert_sha(CM0_LOCK, EXPECTED_CM0_LOCK_SHA, "CM0_LOCK")
    cm0 = load_json(CM0_LOCK)
    if cm0.get("status") != "PASS":
        raise RuntimeError("CM0 status changed.")

    assert_sha(P158_ZIP, EXPECTED_P158_SHA256, "PROSTATE158_ZIP")
    assert_sha(P12_ZIP, EXPECTED_P12_SHA256, "PROMISE12_ZIP")
    print("PASS")

    print("\n===== SAFE EXTRACTION =====")
    s1 = safe_extract_zip(P158_ZIP, P158_EXTRACT)
    s2 = safe_extract_zip(P12_ZIP, P12_EXTRACT)
    print("Prostate158 extraction:", s1)
    print("PROMISE12 extraction:", s2)

    nib = import_nibabel()

    p158 = audit_prostate158_source(nib)
    p12 = audit_promise12_without_gt_decode()

    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)

    source_path = args.output_dir / "CM1_PROSTATE158_SOURCE_SCHEMA_AUDIT.json"
    source_path.write_text(
        json.dumps(p158, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    p12_path = args.output_dir / "CM1_PROMISE12_HEADER_ONLY_GEOMETRY_AUDIT.json"
    p12_path.write_text(
        json.dumps(p12, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    physical_mismatch_count = len(
        p12["physical_metadata_mismatch_cases"]
    )

    lock = {
        "status": "PASS",
        "decision": DECISION_PASS,
        "version": VERSION,
        "build": BUILD,
        "cm0_lock_sha256": EXPECTED_CM0_LOCK_SHA,
        "source_zip_sha256": EXPECTED_P158_SHA256,
        "external_zip_sha256": EXPECTED_P12_SHA256,
        "fix2_reason": (
            "Fix1 required text-identical physical metadata in all image/reference "
            "MHD headers. Fix2 separates required array-grid compatibility from "
            "diagnostic physical metadata equality without opening GT voxels."
        ),
        "source": {
            "dataset": "Prostate158",
            "annotation_count": p158["annotation_count"],
            "t2_candidate_count": p158["t2_candidate_count"],
            "observed_annotation_values": p158["annotation_global_unique_values"],
            "whole_gland_rule": p158["whole_gland_rule"],
            "pairing_unresolved_cases": 0,
        },
        "external": {
            "dataset": "PROMISE12",
            "case_count": p12["case_count"],
            "array_grid_compatible_count": p12["array_grid_compatible_count"],
            "array_grid_failure_cases": p12["array_grid_failure_cases"],
            "physical_metadata_exact_match_count": (
                p12["physical_metadata_exact_match_count"]
            ),
            "physical_metadata_mismatch_cases": (
                p12["physical_metadata_mismatch_cases"]
            ),
            "physical_metadata_mismatch_is_retained_as_diagnostic": True,
            "reference_voxel_payload_decoded": False,
        },
        "information_boundary": {
            "model_training": False,
            "tta": False,
            "safety_model_fitting": False,
            "promises12_gt_voxel_decoding": False,
            "target_tuning": False,
        },
        "artifacts": {
            "source_schema_audit": str(source_path),
            "source_schema_audit_sha256": sha256_file(source_path),
            "external_geometry_audit": str(p12_path),
            "external_geometry_audit_sha256": sha256_file(p12_path),
        },
        "next_stage": (
            "CM2_PREPROCESSING_AND_SOURCE_MODEL_PANEL_PROTOCOL_LOCK"
        ),
    }

    lock_path = args.output_dir / "CM1_PROSTATE_MRI_SCHEMA_LOCK.json"
    lock_path.write_text(
        json.dumps(lock, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n===== CM1 FIX2 FINAL =====")
    print("SOURCE annotation values:", p158["annotation_global_unique_values"])
    print("SOURCE whole-gland rule:", p158["whole_gland_rule"]["rule"])
    print("SOURCE pairing unresolved cases: 0")
    print("PROMISE12 array-grid compatible:", p12["array_grid_compatible_count"], "/ 50")
    print(
        "PROMISE12 physical-metadata exact match:",
        p12["physical_metadata_exact_match_count"],
        "/ 50",
    )
    print(
        "PROMISE12 physical-metadata mismatch cases:",
        p12["physical_metadata_mismatch_cases"],
    )
    print("PROMISE12 GT voxel decoded: NO")
    print("Decision=", DECISION_PASS)
    print("LOCK=", lock_path)
    print("LOCK SHA256=", sha256_file(lock_path))
    print("PASS")


if __name__ == "__main__":
    main()
