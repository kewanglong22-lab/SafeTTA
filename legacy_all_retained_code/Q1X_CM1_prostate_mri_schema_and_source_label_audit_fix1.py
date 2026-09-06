#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1X_CM1_prostate_mri_schema_and_source_label_audit_fix1.py

Cross-modality Q1 enhancement experiment: CM1 archive extraction, schema audit,
and SOURCE-label encoding audit.

Scientific role:
- preserve the exact CM0 dataset lock;
- safely extract official archives to a new branch-local staging area;
- decode Prostate158 SOURCE annotations only;
- inventory/validate PROMISE12 image/reference pairing WITHOUT decoding
  PROMISE12 reference-mask voxel values;
- freeze the exact source whole-gland mapping only if supported by the
  Prostate158 annotation values actually observed.

This stage performs NO model training, NO TTA, NO safety-model fitting,
NO PROMISE12 GT voxel decoding, and NO target-side tuning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


VERSION = "2026-09-04-Q1X-CM1-v1-fix1"
BUILD = "Q1X_CM1_PROSTATE_MRI_SCHEMA_AND_SOURCE_LABEL_AUDIT_FIX1"

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

DEFAULT_OUTPUT = OUT / "Q1X_CM1_prostate_mri_schema_and_source_label_audit_fix1_v1"

DECISION_PASS = (
    "SOURCE_SCHEMA_AND_LABEL_ENCODING_LOCKED_READY_FOR_CM2_PREPROCESSING_PROTOCOL"
)


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

    if nonempty != len(ann):
        raise RuntimeError("At least one Prostate158 SOURCE annotation is empty.")

    nonzero_labels = [v for v in observed_sorted if float(v) != 0.0]
    if not nonzero_labels:
        raise RuntimeError("No nonzero SOURCE labels observed.")

    whole_gland_rule = {
        "rule": "whole_gland = annotation > 0",
        "background_value": 0,
        "observed_nonzero_labels": nonzero_labels,
        "rationale": (
            "Whole-gland binary target is the union of all nonzero prostate-zone labels."
        ),
    }

    by_case = defaultdict(lambda: {"t2": [], "annotation": []})
    for p in t2:
        by_case[infer_case_key(p)]["t2"].append(str(p))
    for p in ann:
        by_case[infer_case_key(p)]["annotation"].append(str(p))

    exact_pairs = {
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

    print("cases with annotation and >=1 T2 candidate:", len(exact_pairs))
    print("unresolved annotation-bearing cases:", len(unresolved))

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
        "annotation_records": ann_records,
        "case_pairing_summary": {
            "annotation_cases_with_t2_candidate": len(exact_pairs),
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


def audit_promise12_without_gt_decode():
    print("\n===== PROMISE12 EXTERNAL SCHEMA AUDIT =====")
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

    img_cases = set(img_by_case)
    ref_cases = set(ref_by_case)
    if img_cases != ref_cases:
        raise RuntimeError(
            f"PROMISE12 image/reference case mismatch: "
            f"image_only={sorted(img_cases-ref_cases)}, "
            f"ref_only={sorted(ref_cases-img_cases)}"
        )

    records = []

    for cid in sorted(img_cases):
        ip = img_by_case[cid]
        rp = ref_by_case[cid]

        ih = parse_mhd_header(ip)
        rh = parse_mhd_header(rp)

        keys = ["NDims", "DimSize", "ElementSpacing", "TransformMatrix", "Offset"]
        geom_equal = all(
            ih.get(k) == rh.get(k)
            for k in keys
            if k in ih or k in rh
        )

        img_raw = ip.parent / ih.get("ElementDataFile", "")
        ref_raw = rp.parent / rh.get("ElementDataFile", "")

        if not img_raw.is_file():
            raise FileNotFoundError(img_raw)
        if not ref_raw.is_file():
            raise FileNotFoundError(ref_raw)

        records.append({
            "case_id": cid,
            "image_mhd": str(ip),
            "reference_mhd": str(rp),
            "image_raw": str(img_raw),
            "reference_raw": str(ref_raw),
            "image_header": ih,
            "reference_header": rh,
            "header_geometry_equal": bool(geom_equal),
            "reference_voxel_payload_decoded": False,
        })

    geometry_pass = sum(r["header_geometry_equal"] for r in records)
    print("image/reference header geometry equal:", geometry_pass, "/ 50")
    print("PROMISE12 reference voxel payload decoded: NO")

    if geometry_pass != 50:
        raise RuntimeError("PROMISE12 image/reference geometry-header mismatch.")

    return {
        "image_mhd_count": len(images),
        "reference_mhd_count": len(refs),
        "raw_count": len(raw),
        "case_count": len(records),
        "geometry_header_match_count": geometry_pass,
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

    print("===== Q1X CM1 PROSTATE MRI SCHEMA / SOURCE LABEL AUDIT =====")
    print("STATUS=PRETRAINING_SCHEMA_AND_SOURCE_LABEL_AUDIT")
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

    source_pairing_unresolved = int(
        p158["case_pairing_summary"]["unresolved_annotation_bearing_cases"]
    )

    if source_pairing_unresolved > 0:
        decision = "STOP_BEFORE_CM2_SOURCE_T2_PAIRING_AMBIGUITY"
        status = "STOP"
    else:
        decision = DECISION_PASS
        status = "PASS"

    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)

    source_records_path = (
        args.output_dir / "CM1_PROSTATE158_SOURCE_ANNOTATION_AUDIT.json"
    )
    source_records_path.write_text(
        json.dumps(p158, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    promise_records_path = (
        args.output_dir / "CM1_PROMISE12_HEADER_ONLY_PAIRING_AUDIT.json"
    )
    promise_records_path.write_text(
        json.dumps(p12, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lock = {
        "status": status,
        "decision": decision,
        "version": VERSION,
        "build": BUILD,
        "cm0_lock_sha256": EXPECTED_CM0_LOCK_SHA,
        "source_zip_sha256": EXPECTED_P158_SHA256,
        "external_zip_sha256": EXPECTED_P12_SHA256,
        "source": {
            "dataset": "Prostate158",
            "annotation_count": p158["annotation_count"],
            "t2_candidate_count": p158["t2_candidate_count"],
            "observed_annotation_values": p158["annotation_global_unique_values"],
            "whole_gland_rule": p158["whole_gland_rule"],
            "pairing_unresolved_cases": source_pairing_unresolved,
        },
        "external": {
            "dataset": "PROMISE12",
            "case_count": p12["case_count"],
            "image_reference_header_geometry_match_count": (
                p12["geometry_header_match_count"]
            ),
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
            "source_audit": str(source_records_path),
            "source_audit_sha256": sha256_file(source_records_path),
            "external_header_audit": str(promise_records_path),
            "external_header_audit_sha256": sha256_file(promise_records_path),
        },
        "next_stage_if_pass": (
            "CM2_PREPROCESSING_AND_SOURCE_MODEL_PANEL_PROTOCOL_LOCK"
        ),
    }

    lock_path = args.output_dir / "CM1_PROSTATE_MRI_SCHEMA_LOCK.json"
    lock_path.write_text(
        json.dumps(lock, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n===== CM1 FINAL =====")
    print("SOURCE annotation values:", p158["annotation_global_unique_values"])
    print("SOURCE whole-gland rule:", p158["whole_gland_rule"]["rule"])
    print("SOURCE pairing unresolved cases:", source_pairing_unresolved)
    print("PROMISE12 cases:", p12["case_count"])
    print("PROMISE12 GT voxel decoded: NO")
    print("Decision=", decision)
    print("LOCK=", lock_path)
    print("LOCK SHA256=", sha256_file(lock_path))
    print("PASS" if status == "PASS" else "STOP")


if __name__ == "__main__":
    main()
