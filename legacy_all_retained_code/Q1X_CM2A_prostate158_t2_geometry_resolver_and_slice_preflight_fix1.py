#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1X_CM2A_prostate158_t2_geometry_resolver_and_slice_preflight_fix1.py

SafeTTA Q1 enhancement, CM2A.

Purpose
-------
1) Gate the exact CM1 Fix3 lock.
2) Resolve the exact Prostate158 SOURCE T2 volume for each of 139 cases using
   SOURCE annotation geometry only.
3) Do NOT use filename heuristics to break ties.
4) Audit SOURCE slice counts / positive-slice distribution.
5) Build a PROMISE12 image-only axial-slice manifest from image MHD headers
   without opening reference-mask RAW payloads.
6) Freeze the sample unit for the cross-modality experiment as an axial 2D
   slice, while preserving patient-level grouping.

This stage performs NO model training, NO TTA, NO safety-model fitting, and
NO PROMISE12 GT voxel decoding.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


VERSION = "2026-09-04-Q1X-CM2A-v1-fix1"
BUILD = "Q1X_CM2A_PROSTATE158_T2_GEOMETRY_RESOLVER_AND_SLICE_PREFLIGHT_FIX1"

ROOT = Path(r"F:\MEDSEG_SAFETTA\cross_modality_prostate_mri")
OUT = Path(r"F:\MEDSEG_SAFETTA\outputs")

CM1_DIR = OUT / "Q1X_CM1_prostate_mri_schema_and_source_label_audit_fix3_v1"
CM1_LOCK = CM1_DIR / "CM1_PROSTATE_MRI_SCHEMA_LOCK.json"
EXPECTED_CM1_LOCK_SHA = (
    "3605d93c416e98b702506c3f9716225bd55ae8d074dbbf47ffc76d46cd414090"
)

P158_ROOT = ROOT / "data" / "extracted" / "Prostate158"
P12_ROOT = ROOT / "data" / "extracted" / "PROMISE12"

DEFAULT_OUTPUT = (
    OUT
    / "Q1X_CM2A_prostate158_t2_geometry_resolver_and_slice_preflight_fix1_v1"
)

DECISION_PASS = (
    "SOURCE_T2_SERIES_AND_SLICE_SAMPLE_UNIT_LOCKED_READY_FOR_CM2B_PREPROCESSING_LOCK"
)

AFFINE_ATOL = 1e-4
AFFINE_RTOL = 1e-5


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


def import_nibabel():
    try:
        import nibabel as nib
    except Exception as e:
        raise RuntimeError("nibabel is required for CM2A.") from e
    return nib


def infer_case_key(path: Path):
    for token in reversed(path.parts[:-1]):
        m = re.search(r"(?:case|patient|prostate)?[_\-]?(\d{1,4})$", token, re.I)
        if m:
            return f"case_{int(m.group(1)):04d}"

    m = re.search(r"(?:case|patient|prostate)[_\-]?(\d{1,4})", path.name, re.I)
    if m:
        return f"case_{int(m.group(1)):04d}"

    raise RuntimeError(f"Cannot infer Prostate158 case key: {path}")


def classify_p158_nifti(path: Path):
    name = path.name.lower()
    if not (name.endswith(".nii") or name.endswith(".nii.gz")):
        return None
    if any(x in name for x in ["anatom", "seg", "mask", "label"]):
        return "annotation"
    if "t2" in name:
        return "t2"
    return "other"


def nifti_header_record(nib, path: Path):
    img = nib.load(str(path))
    shape = tuple(int(x) for x in img.shape)
    affine = np.asarray(img.affine, dtype=np.float64)
    zooms = tuple(float(x) for x in img.header.get_zooms()[: len(shape)])
    return {
        "path": str(path),
        "shape": shape,
        "affine": affine,
        "zooms": zooms,
        "axcodes": tuple(str(x) for x in nib.aff2axcodes(affine)),
    }


def resolve_source_t2(nib):
    print("\n===== PROSTATE158 EXACT T2 GEOMETRY RESOLUTION =====")

    files = sorted(p for p in P158_ROOT.rglob("*") if p.is_file())

    anns = defaultdict(list)
    t2s = defaultdict(list)

    for p in files:
        c = classify_p158_nifti(p)
        if c == "annotation":
            anns[infer_case_key(p)].append(p)
        elif c == "t2":
            t2s[infer_case_key(p)].append(p)

    if len(anns) != 139:
        raise RuntimeError(f"Expected 139 annotation cases, found {len(anns)}.")

    rows = []
    selected = {}
    ambiguous = []
    no_match = []

    for case_key in sorted(anns):
        if len(anns[case_key]) != 1:
            raise RuntimeError(
                f"{case_key}: expected exactly one annotation, found {len(anns[case_key])}."
            )

        ann_path = anns[case_key][0]
        ann = nifti_header_record(nib, ann_path)
        candidates = t2s.get(case_key, [])

        match_records = []

        for t2_path in candidates:
            rec = nifti_header_record(nib, t2_path)

            shape_equal = rec["shape"] == ann["shape"]
            affine_equal = bool(
                np.allclose(
                    rec["affine"],
                    ann["affine"],
                    atol=AFFINE_ATOL,
                    rtol=AFFINE_RTOL,
                )
            )
            zoom_equal = bool(
                len(rec["zooms"]) == len(ann["zooms"])
                and np.allclose(
                    np.asarray(rec["zooms"], dtype=float),
                    np.asarray(ann["zooms"], dtype=float),
                    atol=1e-5,
                    rtol=1e-5,
                )
            )

            match_records.append({
                "path": str(t2_path),
                "shape": list(rec["shape"]),
                "zooms": list(rec["zooms"]),
                "axcodes": list(rec["axcodes"]),
                "shape_equal": shape_equal,
                "affine_equal": affine_equal,
                "zoom_equal": zoom_equal,
                "exact_geometry_match": bool(shape_equal and affine_equal),
            })

        exact = [r for r in match_records if r["exact_geometry_match"]]

        if len(exact) == 1:
            selected[case_key] = exact[0]["path"]
            status = "UNIQUE_EXACT_GEOMETRY_MATCH"
        elif len(exact) == 0:
            no_match.append(case_key)
            status = "NO_EXACT_GEOMETRY_MATCH"
        else:
            ambiguous.append(case_key)
            status = "MULTIPLE_EXACT_GEOMETRY_MATCHES"

        rows.append({
            "case_key": case_key,
            "annotation_path": str(ann_path),
            "annotation_shape": list(ann["shape"]),
            "annotation_zooms": list(ann["zooms"]),
            "annotation_axcodes": list(ann["axcodes"]),
            "t2_candidate_count": len(candidates),
            "exact_geometry_match_count": len(exact),
            "status": status,
            "candidate_records": match_records,
        })

    print("annotation cases:", len(anns))
    print("unique exact T2 matches:", len(selected))
    print("no exact match cases:", no_match)
    print("multiple exact match cases:", ambiguous)

    return {
        "cases": rows,
        "selected": selected,
        "unique_exact_match_count": len(selected),
        "no_match_cases": no_match,
        "ambiguous_cases": ambiguous,
    }


def audit_source_slice_distribution(nib, selected_map):
    print("\n===== SOURCE SLICE DISTRIBUTION =====")

    case_rows = []
    total_slices = 0
    total_positive_slices = 0
    positive_counts = []

    annotation_by_case = {}
    for p in sorted(P158_ROOT.rglob("*")):
        if p.is_file() and classify_p158_nifti(p) == "annotation":
            annotation_by_case[infer_case_key(p)] = p

    for i, case_key in enumerate(sorted(selected_map), 1):
        ann_path = annotation_by_case[case_key]
        ann_img = nib.load(str(ann_path))
        arr = np.asanyarray(ann_img.dataobj)

        if arr.ndim != 3:
            raise RuntimeError(
                f"{case_key}: expected 3D annotation, got shape {arr.shape}."
            )

        whole = arr > 0
        # Freeze axial 2D sample unit as index along the final NIfTI axis.
        z_positive = np.any(whole, axis=(0, 1))
        n_slices = int(arr.shape[2])
        n_positive = int(z_positive.sum())

        total_slices += n_slices
        total_positive_slices += n_positive
        positive_counts.append(n_positive)

        case_rows.append({
            "case_key": case_key,
            "t2_path": selected_map[case_key],
            "annotation_path": str(ann_path),
            "volume_shape": "x".join(str(int(v)) for v in arr.shape),
            "axial_slice_axis": 2,
            "n_slices": n_slices,
            "n_source_gt_positive_slices": n_positive,
            "n_source_gt_empty_slices": n_slices - n_positive,
        })

        if i % 20 == 0 or i == len(selected_map):
            print(f"  audited source cases: {i}/{len(selected_map)}")

    print("source cases:", len(case_rows))
    print("total axial slices:", total_slices)
    print("SOURCE GT-positive slices:", total_positive_slices)
    print("SOURCE GT-empty slices:", total_slices - total_positive_slices)
    print(
        "positive slices per patient min/median/max:",
        int(np.min(positive_counts)),
        float(np.median(positive_counts)),
        int(np.max(positive_counts)),
    )

    return pd.DataFrame(case_rows)


def parse_mhd_header(path: Path):
    text = path.read_text(encoding="latin-1", errors="replace")
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
    raise RuntimeError(f"Cannot infer PROMISE12 case ID: {path}")


def build_external_image_only_slice_manifest():
    print("\n===== PROMISE12 IMAGE-ONLY SLICE MANIFEST =====")

    mhd = sorted(p for p in P12_ROOT.rglob("*.mhd") if p.is_file())
    images = [
        p for p in mhd
        if not any(x in p.name.lower() for x in ["segmentation", "seg", "label"])
    ]

    if len(images) != 50:
        raise RuntimeError(f"Expected 50 PROMISE12 image headers, found {len(images)}.")

    rows = []
    patient_rows = []

    for p in images:
        cid = promise_case_id(p)
        h = parse_mhd_header(p)

        dims = [int(x) for x in h["DimSize"].split()]
        if len(dims) != 3:
            raise RuntimeError(f"PROMISE12 case {cid}: expected 3D DimSize, got {dims}.")

        raw_path = p.parent / h["ElementDataFile"]
        if not raw_path.is_file():
            raise FileNotFoundError(raw_path)

        nx, ny, nz = dims

        patient_rows.append({
            "patient_id": f"PROMISE12_{cid:02d}",
            "case_id": cid,
            "image_mhd": str(p),
            "image_raw": str(raw_path),
            "dim_x": nx,
            "dim_y": ny,
            "dim_z": nz,
            "element_type": h.get("ElementType"),
            "element_spacing": h.get("ElementSpacing"),
        })

        for z in range(nz):
            rows.append({
                "sample_id": f"PROMISE12_{cid:02d}_z{z:04d}",
                "patient_id": f"PROMISE12_{cid:02d}",
                "case_id": cid,
                "slice_axis": 2,
                "slice_index": z,
                "image_mhd": str(p),
                "image_raw": str(raw_path),
                "dim_x": nx,
                "dim_y": ny,
                "dim_z": nz,
                "gt_used_for_manifest": False,
            })

    print("PROMISE12 patients:", len(patient_rows))
    print("PROMISE12 image-only axial slices:", len(rows))
    print("PROMISE12 GT used for slice manifest: NO")

    return pd.DataFrame(patient_rows), pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    args = ap.parse_args()

    print("===== Q1X CM2A PROSTATE MRI T2 / SLICE PREFLIGHT =====")
    print("STATUS=PRETRAINING_SOURCE_SERIES_AND_SAMPLE_UNIT_PREFLIGHT")
    print("MODEL_TRAINING=NO")
    print("TTA=NO")
    print("SAFETY_MODEL_FITTING=NO")
    print("PROMISE12_GT_VOXEL_DECODING=NO")
    print("TARGET_TUNING=NO")
    print("SOURCE_T2_TIE_BREAK_BY_FILENAME=NO")
    print("SAMPLE_UNIT=axial_2D_slice")
    print("PATIENT_GROUPING=YES")

    print("\n===== CM1 LINEAGE GATE =====")
    assert_sha(CM1_LOCK, EXPECTED_CM1_LOCK_SHA, "CM1_LOCK")
    cm1 = load_json(CM1_LOCK)
    if cm1.get("status") != "PASS":
        raise RuntimeError("CM1 status changed.")
    if cm1.get("decision") != (
        "SOURCE_SCHEMA_AND_LABEL_ENCODING_LOCKED_READY_FOR_CM2_PREPROCESSING_PROTOCOL"
    ):
        raise RuntimeError("CM1 decision changed.")
    print("PASS")

    nib = import_nibabel()

    resolution = resolve_source_t2(nib)

    if resolution["no_match_cases"] or resolution["ambiguous_cases"]:
        print("\n===== CM2A STOP =====")
        print("Decision=STOP_SOURCE_T2_EXACT_GEOMETRY_RESOLUTION_INCOMPLETE")
        print("No-match cases:", resolution["no_match_cases"])
        print("Ambiguous cases:", resolution["ambiguous_cases"])
        print("Do not select a T2 series heuristically.")
        raise RuntimeError(
            "Source T2 exact geometry resolution incomplete."
        )

    source_case_df = audit_source_slice_distribution(
        nib,
        resolution["selected"],
    )

    promise_patient_df, promise_slice_df = (
        build_external_image_only_slice_manifest()
    )

    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)

    source_case_path = (
        args.output_dir / "CM2A_PROSTATE158_EXACT_T2_CASE_MANIFEST.csv"
    )
    source_case_df.to_csv(source_case_path, index=False)

    p12_patient_path = (
        args.output_dir / "CM2A_PROMISE12_IMAGE_ONLY_PATIENT_MANIFEST.csv"
    )
    promise_patient_df.to_csv(p12_patient_path, index=False)

    p12_slice_path = (
        args.output_dir / "CM2A_PROMISE12_IMAGE_ONLY_SLICE_MANIFEST.csv"
    )
    promise_slice_df.to_csv(p12_slice_path, index=False)

    resolution_path = (
        args.output_dir / "CM2A_PROSTATE158_T2_GEOMETRY_AUDIT.json"
    )
    resolution_path.write_text(
        json.dumps(
            {
                "unique_exact_match_count": resolution["unique_exact_match_count"],
                "no_match_cases": resolution["no_match_cases"],
                "ambiguous_cases": resolution["ambiguous_cases"],
                "cases": resolution["cases"],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    lock = {
        "status": "PASS",
        "decision": DECISION_PASS,
        "version": VERSION,
        "build": BUILD,
        "cm1_lock_sha256": EXPECTED_CM1_LOCK_SHA,
        "source": {
            "dataset": "Prostate158",
            "case_count": int(len(source_case_df)),
            "unique_exact_t2_geometry_matches": int(
                resolution["unique_exact_match_count"]
            ),
            "whole_gland_rule": "annotation > 0",
            "sample_unit": "axial_2D_slice",
            "slice_axis": 2,
            "patient_grouping_required": True,
            "total_axial_slices": int(source_case_df["n_slices"].sum()),
            "source_gt_positive_slices": int(
                source_case_df["n_source_gt_positive_slices"].sum()
            ),
            "source_gt_empty_slices": int(
                source_case_df["n_source_gt_empty_slices"].sum()
            ),
        },
        "external": {
            "dataset": "PROMISE12",
            "patient_count": int(len(promise_patient_df)),
            "image_only_slice_count": int(len(promise_slice_df)),
            "sample_unit": "axial_2D_slice",
            "slice_axis": 2,
            "patient_grouping_required": True,
            "all_slices_included_before_gt_reveal": True,
            "gt_used_for_slice_manifest": False,
            "gt_voxel_decoded": False,
        },
        "information_boundary": {
            "model_training": False,
            "tta": False,
            "safety_model_fitting": False,
            "promises12_gt_voxel_decoding": False,
            "target_tuning": False,
            "filename_tie_break_for_source_t2": False,
        },
        "artifacts": {
            "source_case_manifest": str(source_case_path),
            "source_case_manifest_sha256": sha256_file(source_case_path),
            "source_geometry_audit": str(resolution_path),
            "source_geometry_audit_sha256": sha256_file(resolution_path),
            "external_patient_manifest": str(p12_patient_path),
            "external_patient_manifest_sha256": sha256_file(p12_patient_path),
            "external_slice_manifest": str(p12_slice_path),
            "external_slice_manifest_sha256": sha256_file(p12_slice_path),
        },
        "next_stage": (
            "CM2B_IMAGE_ONLY_INTENSITY_PREPROCESSING_AND_SEGMENTATION_PANEL_LOCK"
        ),
    }

    lock_path = args.output_dir / "CM2A_PROSTATE_MRI_SAMPLE_UNIT_LOCK.json"
    lock_path.write_text(
        json.dumps(lock, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n===== CM2A FINAL =====")
    print("SOURCE cases:", len(source_case_df))
    print(
        "SOURCE unique exact T2 matches:",
        resolution["unique_exact_match_count"],
    )
    print("SOURCE total axial slices:", int(source_case_df["n_slices"].sum()))
    print(
        "SOURCE GT-positive axial slices:",
        int(source_case_df["n_source_gt_positive_slices"].sum()),
    )
    print("PROMISE12 patients:", len(promise_patient_df))
    print("PROMISE12 image-only axial slices:", len(promise_slice_df))
    print("PROMISE12 GT used for manifest: NO")
    print("Decision=", DECISION_PASS)
    print("LOCK=", lock_path)
    print("LOCK SHA256=", sha256_file(lock_path))
    print("PASS")


if __name__ == "__main__":
    main()
