#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1X_CM2A_prostate158_official_manifest_t2_resolver_and_slice_preflight_fix2.py

CM2A Fix2.

Fix1 incorrectly treated every NIfTI filename containing "t2" as a possible
T2 image. In Prostate158, files such as t2_tumor_reader1.nii.gz are labels,
not image volumes. The official dataset archive and baseline code provide
authoritative train.csv / valid.csv columns named:
  - t2
  - t2_anatomy_reader1

Fix2 therefore resolves the SOURCE image/label pair from those official CSV
manifests, then verifies file existence and exact image/label geometry.

No filename heuristic is used for case selection.
No PROMISE12 reference voxel payload is decoded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


VERSION = "2026-09-04-Q1X-CM2A-v1-fix2"
BUILD = "Q1X_CM2A_PROSTATE158_OFFICIAL_MANIFEST_T2_RESOLVER_AND_SLICE_PREFLIGHT_FIX2"

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
    / "Q1X_CM2A_prostate158_official_manifest_t2_resolver_and_slice_preflight_fix2_v1"
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
        raise RuntimeError("nibabel is required for CM2A Fix2.") from e
    return nib


def find_unique_file(root: Path, name: str) -> Path:
    matches = sorted(root.rglob(name))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one {name}, found {len(matches)}: {matches}"
        )
    return matches[0]


def normalize_relpath(s: str) -> str:
    return str(s).replace("\\", "/").lstrip("./")


def resolve_manifest_path(csv_parent: Path, value: str) -> Path:
    """
    Official CSVs contain relative paths. Resolve conservatively:
    1) relative to CSV parent;
    2) relative to P158_ROOT;
    3) suffix match under P158_ROOT, but only if unique.
    """
    rel = normalize_relpath(value)

    candidates = [
        (csv_parent / rel).resolve(),
        (P158_ROOT / rel).resolve(),
    ]

    for p in candidates:
        if p.is_file():
            return p

    suffix = rel.lower()
    matches = [
        p for p in P158_ROOT.rglob("*")
        if p.is_file()
        and str(p).replace("\\", "/").lower().endswith(suffix)
    ]

    if len(matches) == 1:
        return matches[0].resolve()

    raise FileNotFoundError(
        f"Could not uniquely resolve official CSV path: {value}; matches={matches}"
    )


def infer_case_key_from_manifest_path(value: str) -> str:
    rel = normalize_relpath(value)
    parts = rel.split("/")
    for token in reversed(parts[:-1]):
        if token.isdigit():
            return f"case_{int(token):04d}"
        m = re.search(r"(\d{1,4})", token)
        if m:
            return f"case_{int(m.group(1)):04d}"
    raise RuntimeError(f"Cannot infer case ID from official manifest path: {value}")


def load_official_source_manifest():
    print("\n===== PROSTATE158 OFFICIAL CSV MANIFEST RESOLUTION =====")

    train_csv = find_unique_file(P158_ROOT, "train.csv")
    valid_csv = find_unique_file(P158_ROOT, "valid.csv")

    print("train.csv:", train_csv)
    print("valid.csv:", valid_csv)

    frames = []
    for split_name, path in [("train", train_csv), ("valid", valid_csv)]:
        df = pd.read_csv(path)
        print(split_name, "rows=", len(df), "columns=", list(df.columns))

        required = {"t2", "t2_anatomy_reader1"}
        missing = required - set(df.columns)
        if missing:
            raise RuntimeError(
                f"{path.name} missing required official columns: {sorted(missing)}"
            )

        x = df.copy()
        x["_official_split"] = split_name
        x["_csv_path"] = str(path)
        frames.append(x)

    panel = pd.concat(frames, ignore_index=True)

    if len(panel) != 139:
        raise RuntimeError(
            f"Expected 139 total official train+valid rows, found {len(panel)}."
        )

    rows = []
    for r in panel.itertuples(index=False):
        rec = r._asdict()
        split = rec["_official_split"]
        csv_path = Path(rec["_csv_path"])

        t2_value = str(rec["t2"])
        ann_value = str(rec["t2_anatomy_reader1"])

        t2_path = resolve_manifest_path(csv_path.parent, t2_value)
        ann_path = resolve_manifest_path(csv_path.parent, ann_value)

        if t2_path.name.lower() != "t2.nii.gz":
            raise RuntimeError(
                f"Official t2 column did not resolve to t2.nii.gz: {t2_path}"
            )
        if ann_path.name.lower() != "t2_anatomy_reader1.nii.gz":
            raise RuntimeError(
                "Official anatomy column did not resolve to "
                f"t2_anatomy_reader1.nii.gz: {ann_path}"
            )

        case_key_t2 = infer_case_key_from_manifest_path(t2_value)
        case_key_ann = infer_case_key_from_manifest_path(ann_value)
        if case_key_t2 != case_key_ann:
            raise RuntimeError(
                f"Official t2/anatomy case mismatch: {t2_value} vs {ann_value}"
            )

        rows.append({
            "case_key": case_key_t2,
            "official_split": split,
            "official_t2_relpath": t2_value,
            "official_anatomy_relpath": ann_value,
            "t2_path": str(t2_path),
            "annotation_path": str(ann_path),
        })

    out = pd.DataFrame(rows)

    if out["case_key"].nunique() != 139:
        dup = out[out["case_key"].duplicated(keep=False)]
        raise RuntimeError(
            "Official source manifest case IDs are not unique:\n"
            + dup.to_string(index=False)
        )

    if out["t2_path"].nunique() != 139:
        raise RuntimeError("Official t2 paths are not unique.")
    if out["annotation_path"].nunique() != 139:
        raise RuntimeError("Official annotation paths are not unique.")

    print("official train+valid rows:", len(out))
    print("unique cases:", out["case_key"].nunique())
    print("all image basenames = t2.nii.gz: YES")
    print("all label basenames = t2_anatomy_reader1.nii.gz: YES")

    return out, train_csv, valid_csv


def verify_source_geometry_and_slices(nib, manifest: pd.DataFrame):
    print("\n===== PROSTATE158 OFFICIAL IMAGE/LABEL GEOMETRY AUDIT =====")

    rows = []
    total_slices = 0
    total_positive = 0
    pos_counts = []

    for i, r in enumerate(manifest.itertuples(index=False), 1):
        t2_img = nib.load(r.t2_path)
        ann_img = nib.load(r.annotation_path)

        t2_shape = tuple(int(x) for x in t2_img.shape)
        ann_shape = tuple(int(x) for x in ann_img.shape)

        if t2_shape != ann_shape:
            raise RuntimeError(
                f"{r.case_key}: official t2/anatomy shape mismatch: "
                f"{t2_shape} vs {ann_shape}"
            )

        if not np.allclose(
            np.asarray(t2_img.affine, dtype=np.float64),
            np.asarray(ann_img.affine, dtype=np.float64),
            atol=AFFINE_ATOL,
            rtol=AFFINE_RTOL,
        ):
            raise RuntimeError(
                f"{r.case_key}: official t2/anatomy affine mismatch."
            )

        arr = np.asanyarray(ann_img.dataobj)
        vals = sorted(int(v) for v in np.unique(arr).tolist())
        if any(v not in (0, 1, 2) for v in vals):
            raise RuntimeError(
                f"{r.case_key}: unexpected source label values {vals}"
            )

        whole = arr > 0

        if whole.ndim != 3:
            raise RuntimeError(
                f"{r.case_key}: expected 3D anatomy label, got {whole.shape}"
            )

        # Dataset documentation states all examinations are axial.
        # For the NIfTI array used here, freeze slice sampling along axis 2.
        positive = np.any(whole, axis=(0, 1))
        n_slices = int(whole.shape[2])
        n_positive = int(positive.sum())

        total_slices += n_slices
        total_positive += n_positive
        pos_counts.append(n_positive)

        rows.append({
            "case_key": r.case_key,
            "official_split": r.official_split,
            "t2_path": r.t2_path,
            "annotation_path": r.annotation_path,
            "shape_x": t2_shape[0],
            "shape_y": t2_shape[1],
            "shape_z": t2_shape[2],
            "slice_axis": 2,
            "n_slices": n_slices,
            "n_source_gt_positive_slices": n_positive,
            "n_source_gt_empty_slices": n_slices - n_positive,
            "whole_gland_rule": "annotation > 0",
        })

        if i % 20 == 0 or i == len(manifest):
            print(f"  geometry/source-label audited: {i}/{len(manifest)}")

    print("geometry exact cases:", len(rows), "/", len(manifest))
    print("total axial slices:", total_slices)
    print("SOURCE GT-positive slices:", total_positive)
    print("SOURCE GT-empty slices:", total_slices - total_positive)
    print(
        "positive slices/patient min/median/max:",
        int(np.min(pos_counts)),
        float(np.median(pos_counts)),
        int(np.max(pos_counts)),
    )

    return pd.DataFrame(rows)


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
    print("\n===== PROMISE12 IMAGE-ONLY AXIAL SLICE MANIFEST =====")

    mhd = sorted(p for p in P12_ROOT.rglob("*.mhd") if p.is_file())
    images = [
        p for p in mhd
        if not any(x in p.name.lower() for x in ["segmentation", "seg", "label"])
    ]

    if len(images) != 50:
        raise RuntimeError(
            f"Expected 50 PROMISE12 image headers, found {len(images)}."
        )

    patient_rows = []
    slice_rows = []

    for p in images:
        cid = promise_case_id(p)
        h = parse_mhd_header(p)

        dims = [int(x) for x in h["DimSize"].split()]
        if len(dims) != 3:
            raise RuntimeError(
                f"PROMISE12 case {cid}: expected 3D DimSize, got {dims}."
            )

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
            slice_rows.append({
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
    print("PROMISE12 image-only axial slices:", len(slice_rows))
    print("PROMISE12 GT used for manifest: NO")

    return pd.DataFrame(patient_rows), pd.DataFrame(slice_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    args = ap.parse_args()

    print("===== Q1X CM2A FIX2 PROSTATE158 OFFICIAL-MANIFEST PREFLIGHT =====")
    print("STATUS=PRETRAINING_SOURCE_SERIES_AND_SAMPLE_UNIT_PREFLIGHT")
    print("FIX2_CHANGE=USE_OFFICIAL_PROSTATE158_CSV_T2_AND_ANATOMY_COLUMNS")
    print("SCIENTIFIC_PROTOCOL_CHANGE=NO")
    print("MODEL_TRAINING=NO")
    print("TTA=NO")
    print("SAFETY_MODEL_FITTING=NO")
    print("PROMISE12_GT_VOXEL_DECODING=NO")
    print("TARGET_TUNING=NO")
    print("SOURCE_T2_FILENAME_HEURISTIC=NO")
    print("SOURCE_T2_AUTHORITY=official_train_valid_csv")
    print("SAMPLE_UNIT=axial_2D_slice")
    print("PATIENT_GROUPING=YES")

    print("\n===== CM1 LINEAGE GATE =====")
    assert_sha(CM1_LOCK, EXPECTED_CM1_LOCK_SHA, "CM1_LOCK")
    cm1 = load_json(CM1_LOCK)
    if cm1.get("status") != "PASS":
        raise RuntimeError("CM1 status changed.")
    print("PASS")

    nib = import_nibabel()

    manifest, train_csv, valid_csv = load_official_source_manifest()
    source_case_df = verify_source_geometry_and_slices(nib, manifest)
    promise_patient_df, promise_slice_df = (
        build_external_image_only_slice_manifest()
    )

    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)

    source_manifest_path = (
        args.output_dir / "CM2A_PROSTATE158_OFFICIAL_T2_CASE_MANIFEST.csv"
    )
    source_case_df.to_csv(source_manifest_path, index=False)

    p12_patient_path = (
        args.output_dir / "CM2A_PROMISE12_IMAGE_ONLY_PATIENT_MANIFEST.csv"
    )
    promise_patient_df.to_csv(p12_patient_path, index=False)

    p12_slice_path = (
        args.output_dir / "CM2A_PROMISE12_IMAGE_ONLY_SLICE_MANIFEST.csv"
    )
    promise_slice_df.to_csv(p12_slice_path, index=False)

    lock = {
        "status": "PASS",
        "decision": DECISION_PASS,
        "version": VERSION,
        "build": BUILD,
        "cm1_lock_sha256": EXPECTED_CM1_LOCK_SHA,
        "fix2_reason": (
            "Fix1 over-counted T2 candidates because t2_tumor_reader*.nii.gz "
            "are label files. Fix2 uses the official train.csv/valid.csv "
            "columns t2 and t2_anatomy_reader1 as the authoritative source pair."
        ),
        "source": {
            "dataset": "Prostate158",
            "authoritative_manifest_files": [
                str(train_csv),
                str(valid_csv),
            ],
            "case_count": int(len(source_case_df)),
            "t2_basename": "t2.nii.gz",
            "anatomy_basename": "t2_anatomy_reader1.nii.gz",
            "whole_gland_rule": "annotation > 0",
            "geometry_exact_case_count": int(len(source_case_df)),
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
            "source_t2_filename_heuristic": False,
        },
        "artifacts": {
            "source_case_manifest": str(source_manifest_path),
            "source_case_manifest_sha256": sha256_file(source_manifest_path),
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

    print("\n===== CM2A FIX2 FINAL =====")
    print("SOURCE official cases:", len(source_case_df))
    print("SOURCE authoritative T2 basename: t2.nii.gz")
    print("SOURCE authoritative label basename: t2_anatomy_reader1.nii.gz")
    print("SOURCE exact image/label geometry:", len(source_case_df), "/", len(source_case_df))
    print("SOURCE total axial slices:", int(source_case_df["n_slices"].sum()))
    print(
        "SOURCE GT-positive axial slices:",
        int(source_case_df["n_source_gt_positive_slices"].sum()),
    )
    print(
        "SOURCE GT-empty axial slices:",
        int(source_case_df["n_source_gt_empty_slices"].sum()),
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
