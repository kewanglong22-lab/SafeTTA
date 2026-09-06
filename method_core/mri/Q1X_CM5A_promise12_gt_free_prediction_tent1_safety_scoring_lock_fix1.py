#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1X_CM5A_promise12_gt_free_prediction_tent1_safety_scoring_lock_fix1.py

SafeTTA Q1 enhancement — CM5A.

FIRST target-image stage.

Allowed:
- PROMISE12 image MHD headers;
- PROMISE12 image RAW payloads;
- frozen SOURCE segmentation models;
- frozen SOURCE safety estimator;
- frozen SOURCE operating point;
- TENT1;
- frozen DINOv2 representation.

Forbidden:
- PROMISE12 segmentation MHD header payloads;
- PROMISE12 segmentation RAW payloads;
- PROMISE12 GT-derived slice selection;
- target threshold tuning/calibration;
- target HARM labels / DeltaDice.

The PROMISE12 zip is accessed directly. Only members whose basename is exactly
CaseXX.mhd and the image RAW file referenced by those headers are opened.
Segmentation members are never opened.

Full-mode outputs:
1) image-only target preprocessing cache (all 50 patients / all 1377 slices);
2) frozen SOURCE predictions for three final all-SOURCE segmenters;
3) episodic one-slice A1_TENT_1STEP predictions;
4) prediction-conditioned DINOv2+M2 target representations;
5) frozen final SOURCE safety scores;
6) decisions under the frozen SOURCE deployment threshold.

No target operating-point transport is performed here. CM5B uses the locked
unlabeled representations from CM5A to test a preregistered support-aware
transport rule before any target GT reveal.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm


VERSION = "2026-09-05-Q1X-CM5A-v1-fix1"
BUILD = "Q1X_CM5A_PROMISE12_GT_FREE_PREDICTION_TENT1_SAFETY_SCORING_LOCK_FIX1"

ROOT = Path(r"F:\MEDSEG_SAFETTA")
OUT = ROOT / "outputs"
CODE = ROOT / "code"

PROMISE_ZIP = (
    ROOT
    / "cross_modality_prostate_mri"
    / "data"
    / "raw"
    / "PROMISE12"
    / "training_data.zip"
)
EXPECTED_PROMISE_ZIP_SIZE = 323664221
EXPECTED_PROMISE_ZIP_MD5 = "f6d23994117c989daf07e5291edd0aea"
EXPECTED_PROMISE_ZIP_SHA256 = (
    "150287d0c74cd0105d8b70b43af4a7bf4f1fd3e1748829779177a0ccaf948f45"
)

CM2B_LOCK = (
    OUT
    / "Q1X_CM2B_prostate_mri_preprocessing_and_model_protocol_lock_fix1_v1"
    / "CM2B_PROTOCOL_LOCK.json"
)
EXPECTED_CM2B_LOCK_SHA = (
    "9122400454853212f7cc9451be45789a052c5279141becdbfad2546c37b4019f"
)

CM4C_LOCK = (
    OUT
    / "Q1X_CM4C_final_source_safety_estimator_and_threshold_lock_fix2_v1"
    / "CM4C_FINAL_SOURCE_SAFETY_ESTIMATOR_LOCK.json"
)
EXPECTED_CM4C_LOCK_SHA = (
    "5ddf4a6a4b44d19a1e3c8cccc8a2a9cdba71da18850a7df58cbbee83a40527ff"
)

CM4D_LOCK = (
    OUT
    / "Q1X_CM4D_final_all_source_segmentation_models_lock_fix2_v1"
    / "CM4D_FINAL_ALL_SOURCE_SEGMENTATION_MODELS_LOCK.json"
)
EXPECTED_CM4D_LOCK_SHA = (
    "fbc55b3d09276e214491e115ca90dff72599222fa7ee92a584c8083de0afbfa7"
)

CM3_HELPER = (
    CODE
    / "Q1X_CM3_source_oof_segmentation_training_and_prediction_lock_fix3.py"
)
EXPECTED_CM3_HELPER_SHA = (
    "443e371c1a71173cc9fd3eaa4876cb8721a00497836e06d902b50da12cb9f57a"
)

CM4A_HELPER = (
    CODE
    / "Q1X_CM4A_source_oof_tent1_outcome_asset_lock_fix4.py"
)
EXPECTED_CM4A_HELPER_SHA = (
    "cd4af7a78fcfc72ff4a12df43578757f095d6a0cb3a6445c8b6c25e50e6ca6c0"
)

CM4B_HELPER = (
    CODE
    / "Q1X_CM4B_mri_source_prediction_conditioned_safety_grouped_oof_fix1.py"
)
EXPECTED_CM4B_HELPER_SHA = (
    "0b894a2db0d58dda6fa9fe26bf0b1482a3da89d25918405e45e39346d132cc40"
)

CM4C_HELPER = (
    CODE
    / "Q1X_CM4C_final_source_safety_estimator_and_threshold_lock_fix2.py"
)
EXPECTED_CM4C_HELPER_SHA = (
    "dd5b410f612b81bcbdc6ac2fbfe8bc4b22a21a049beeac80a2fb589d474a58bc"
)

DEFAULT_OUTPUT = (
    OUT
    / "Q1X_CM5A_promise12_gt_free_prediction_tent1_safety_scoring_lock_fix1_v1"
)

FAMILIES = ["UNet", "DeepLabV3_R50", "SegFormer_B0"]

N_TARGET_PATIENTS = 50
N_TARGET_SLICES = 1377

TARGET_SPACING_XY = 0.4017857015132904
SEG_SIZE = 352
SEG_BATCH_SIZE = 8

DINO_BATCH_SIZE = 16
DINO_DIM = 768
COND_DIM = 1536
FINAL_RAW_DIM = 1538

PACKED_BYTES = (SEG_SIZE * SEG_SIZE + 7) // 8

TENT_LR = 1e-3
TENT_WEIGHT_DECAY = 0.0
TENT_STEPS = 1

IMAGE_MHD_RE = re.compile(r"^Case(\d{2})\.mhd$", re.IGNORECASE)

PASS_DECISION = (
    "PROMISE12_GT_FREE_PREDICTIONS_SAFETY_SCORES_LOCKED_"
    "READY_FOR_CM5B_UNLABELED_OPERATING_POINT_TRANSPORT"
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


def md5_file(path: Path, chunk=8 * 1024 * 1024):
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, obj):
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def import_helper(label, path, expected_sha, module_name):
    if not path.is_file():
        raise FileNotFoundError(path)

    got = sha256_file(path)
    print(label, got, "PASS" if got == expected_sha else "FAIL")
    if got != expected_sha:
        raise RuntimeError(f"{label} SHA mismatch.")

    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import {label}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_upstream_lineage():
    for label, path, expected in [
        ("CM2B_LOCK", CM2B_LOCK, EXPECTED_CM2B_LOCK_SHA),
        ("CM4C_LOCK", CM4C_LOCK, EXPECTED_CM4C_LOCK_SHA),
        ("CM4D_LOCK", CM4D_LOCK, EXPECTED_CM4D_LOCK_SHA),
    ]:
        if not path.is_file():
            raise FileNotFoundError(path)
        got = sha256_file(path)
        print(label, got, "PASS" if got == expected else "FAIL")
        if got != expected:
            raise RuntimeError(f"{label} SHA mismatch.")

    cm2b = load_json(CM2B_LOCK)
    cm4c = load_json(CM4C_LOCK)
    cm4d = load_json(CM4D_LOCK)

    if cm4c.get("status") != "PASS":
        raise RuntimeError("CM4C status changed.")
    if cm4d.get("status") != "PASS":
        raise RuntimeError("CM4D status changed.")

    if cm4c.get("information_boundary", {}).get("promises12_access") is not False:
        raise RuntimeError("CM4C PROMISE12 boundary changed.")
    if cm4d.get("information_boundary", {}).get("promises12_access") is not False:
        raise RuntimeError("CM4D PROMISE12 boundary changed.")

    if cm4d.get("decision") != (
        "FINAL_ALL_SOURCE_SEGMENTATION_MODELS_LOCKED_READY_FOR_"
        "CM5_PROMISE12_GT_FREE_INFERENCE"
    ):
        raise RuntimeError("Unexpected CM4D decision.")

    return cm2b, cm4c, cm4d


def verify_promise_zip():
    if not PROMISE_ZIP.is_file():
        raise FileNotFoundError(PROMISE_ZIP)

    size = PROMISE_ZIP.stat().st_size
    md5 = md5_file(PROMISE_ZIP)
    sha = sha256_file(PROMISE_ZIP)

    print("\n===== PROMISE12 ARCHIVE GATE =====")
    print("path:", PROMISE_ZIP)
    print("size:", size, "PASS" if size == EXPECTED_PROMISE_ZIP_SIZE else "FAIL")
    print("MD5:", md5, "PASS" if md5 == EXPECTED_PROMISE_ZIP_MD5 else "FAIL")
    print("SHA256:", sha, "PASS" if sha == EXPECTED_PROMISE_ZIP_SHA256 else "FAIL")

    if size != EXPECTED_PROMISE_ZIP_SIZE:
        raise RuntimeError("PROMISE12 archive size mismatch.")
    if md5 != EXPECTED_PROMISE_ZIP_MD5:
        raise RuntimeError("PROMISE12 archive MD5 mismatch.")
    if sha != EXPECTED_PROMISE_ZIP_SHA256:
        raise RuntimeError("PROMISE12 archive SHA256 mismatch.")


def parse_mhd_header_bytes(data: bytes):
    text = data.decode("ascii")
    fields = {}

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        fields[key.strip()] = value.strip()

    required = [
        "NDims",
        "DimSize",
        "ElementSpacing",
        "ElementType",
        "ElementDataFile",
    ]
    missing = [k for k in required if k not in fields]
    if missing:
        raise RuntimeError(f"MHD header missing fields: {missing}")

    if int(fields["NDims"]) != 3:
        raise RuntimeError(f"Unexpected NDims={fields['NDims']}")

    dims = [int(v) for v in fields["DimSize"].split()]
    spacing = [float(v) for v in fields["ElementSpacing"].split()]

    if len(dims) != 3 or len(spacing) != 3:
        raise RuntimeError("Expected 3D DimSize/ElementSpacing.")

    if fields["ElementType"].upper() != "MET_SHORT":
        raise RuntimeError(
            f"Unexpected PROMISE image ElementType={fields['ElementType']}"
        )

    compressed = fields.get("CompressedData", "False").lower() == "true"
    if compressed:
        raise RuntimeError("Compressed PROMISE image RAW not expected.")

    msb_text = fields.get(
        "ElementByteOrderMSB",
        fields.get("BinaryDataByteOrderMSB", "False"),
    )
    is_msb = msb_text.lower() == "true"

    return {
        "fields": fields,
        "dims_xyz": dims,
        "spacing_xyz": spacing,
        "raw_ref": fields["ElementDataFile"],
        "is_msb": is_msb,
    }


def build_image_only_header_manifest():
    """
    Reads ZIP central-directory names and ONLY the 50 image MHD header payloads.
    No segmentation MHD header and no RAW payload is opened here.
    """
    accessed_headers = []
    rows = []

    with zipfile.ZipFile(PROMISE_ZIP, "r") as zf:
        names = zf.namelist()

        image_members = []
        for name in names:
            base = Path(name).name
            m = IMAGE_MHD_RE.fullmatch(base)
            if m is not None:
                image_members.append((int(m.group(1)), name))

        image_members.sort(key=lambda x: x[0])

        if len(image_members) != N_TARGET_PATIENTS:
            raise RuntimeError(
                f"Expected 50 exact CaseXX.mhd image members, "
                f"got {len(image_members)}"
            )

        expected_ids = list(range(N_TARGET_PATIENTS))
        got_ids = [case_id for case_id, _ in image_members]
        if got_ids != expected_ids:
            raise RuntimeError(
                f"PROMISE case IDs changed: {got_ids}"
            )

        for case_id, member in image_members:
            data = zf.read(member)
            accessed_headers.append(member)

            h = parse_mhd_header_bytes(data)

            expected_raw_base = f"Case{case_id:02d}.raw"
            raw_base = Path(h["raw_ref"]).name

            if raw_base.lower() != expected_raw_base.lower():
                raise RuntimeError(
                    f"Case{case_id:02d}: unexpected image RAW reference "
                    f"{h['raw_ref']}"
                )

            x, y, z = h["dims_xyz"]
            sx, sy, sz = h["spacing_xyz"]

            rows.append({
                "case_id": case_id,
                "case_key": f"Case{case_id:02d}",
                "mhd_member": member,
                "raw_ref": h["raw_ref"],
                "dim_x": x,
                "dim_y": y,
                "dim_z": z,
                "spacing_x": sx,
                "spacing_y": sy,
                "spacing_z": sz,
                "is_msb": bool(h["is_msb"]),
            })

    df = pd.DataFrame(rows)

    if len(df) != N_TARGET_PATIENTS:
        raise RuntimeError("PROMISE image-only patient count mismatch.")

    total_slices = int(df["dim_z"].sum())
    if total_slices != N_TARGET_SLICES:
        raise RuntimeError(
            f"PROMISE image-only slice count changed: "
            f"{total_slices} != {N_TARGET_SLICES}"
        )

    # Payload-level access audit.
    for member in accessed_headers:
        base = Path(member).name
        if IMAGE_MHD_RE.fullmatch(base) is None:
            raise RuntimeError(
                f"Non-image MHD payload was accessed: {member}"
            )

    print("\n===== PROMISE12 IMAGE-ONLY HEADER AUDIT =====")
    print("patients:", len(df))
    print("image-only axial slices:", total_slices)
    print(
        "spacing x min/median/max:",
        float(df["spacing_x"].min()),
        float(df["spacing_x"].median()),
        float(df["spacing_x"].max()),
    )
    print(
        "spacing y min/median/max:",
        float(df["spacing_y"].min()),
        float(df["spacing_y"].median()),
        float(df["spacing_y"].max()),
    )
    print(
        "spacing z min/median/max:",
        float(df["spacing_z"].min()),
        float(df["spacing_z"].median()),
        float(df["spacing_z"].max()),
    )
    print("PROMISE12 GT MHD payload read: NO")
    print("PROMISE12 GT RAW payload read: NO")

    return df


def resolve_raw_member(zf, mhd_member, raw_ref):
    parent = Path(mhd_member).parent
    candidate = (parent / raw_ref).as_posix()

    names = set(zf.namelist())
    if candidate in names:
        return candidate

    raw_base = Path(raw_ref).name.lower()
    matches = [
        name
        for name in names
        if Path(name).name.lower() == raw_base
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Unable to uniquely resolve image RAW {raw_ref}"
        )
    return matches[0]


def read_image_volume_from_zip(row):
    """
    Opens exactly one image MHD payload and its referenced image RAW payload.
    Segmentation members are never opened.
    """
    with zipfile.ZipFile(PROMISE_ZIP, "r") as zf:
        mhd_member = str(row.mhd_member)
        base = Path(mhd_member).name

        if IMAGE_MHD_RE.fullmatch(base) is None:
            raise RuntimeError(
                f"Forbidden/non-image MHD access attempted: {mhd_member}"
            )

        header = parse_mhd_header_bytes(zf.read(mhd_member))
        raw_member = resolve_raw_member(
            zf,
            mhd_member,
            header["raw_ref"],
        )

        expected_raw_base = f"{row.case_key}.raw"
        if Path(raw_member).name.lower() != expected_raw_base.lower():
            raise RuntimeError(
                f"Forbidden/non-image RAW access attempted: {raw_member}"
            )

        raw = zf.read(raw_member)

    x, y, z = header["dims_xyz"]
    expected_bytes = x * y * z * np.dtype(np.int16).itemsize
    if len(raw) != expected_bytes:
        raise RuntimeError(
            f"{row.case_key}: RAW bytes {len(raw)} != {expected_bytes}"
        )

    dtype = np.dtype(">i2" if header["is_msb"] else "<i2")
    arr = np.frombuffer(raw, dtype=dtype).reshape(z, y, x)

    return arr.astype(np.float32, copy=False), header


def normalize_volume_source_frozen(volume):
    v = np.asarray(volume, dtype=np.float32)

    finite = np.isfinite(v)
    nonzero = finite & (v != 0)

    if not np.any(nonzero):
        raise RuntimeError("PROMISE image volume has no finite nonzero voxels.")

    values = v[nonzero].astype(np.float64)
    q005 = float(np.quantile(values, 0.005))
    q995 = float(np.quantile(values, 0.995))

    if not np.isfinite(q005) or not np.isfinite(q995) or q995 <= q005:
        raise RuntimeError(
            f"Invalid per-volume normalization quantiles: "
            f"{q005}, {q995}"
        )

    out = np.zeros(v.shape, dtype=np.float32)
    clipped = np.clip(v[finite], q005, q995)
    mapped = (clipped - q005) / (q995 - q005)

    # Preserve background/nonfinite exactly at 0 according to CM2B.
    finite_nonzero_flat = nonzero[finite]
    tmp = np.zeros_like(clipped, dtype=np.float32)
    tmp[finite_nonzero_flat] = mapped[finite_nonzero_flat]
    out[finite] = tmp
    out[~finite] = 0.0

    return out, q005, q995


def center_crop_pad_2d_batch(x, torch):
    """
    x: torch [Z,1,H,W], CPU float32.
    """
    _, _, h, w = x.shape

    if h > SEG_SIZE:
        top = (h - SEG_SIZE) // 2
        x = x[:, :, top:top + SEG_SIZE, :]
    elif h < SEG_SIZE:
        total = SEG_SIZE - h
        top = total // 2
        bottom = total - top
        x = torch.nn.functional.pad(
            x,
            (0, 0, top, bottom),
            mode="constant",
            value=0.0,
        )

    h2 = x.shape[-2]
    if h2 != SEG_SIZE:
        raise RuntimeError(f"Unexpected H after crop/pad: {h2}")

    w = x.shape[-1]
    if w > SEG_SIZE:
        left = (w - SEG_SIZE) // 2
        x = x[:, :, :, left:left + SEG_SIZE]
    elif w < SEG_SIZE:
        total = SEG_SIZE - w
        left = total // 2
        right = total - left
        x = torch.nn.functional.pad(
            x,
            (left, right, 0, 0),
            mode="constant",
            value=0.0,
        )

    if tuple(x.shape[-2:]) != (SEG_SIZE, SEG_SIZE):
        raise RuntimeError(
            f"Unexpected crop/pad result: {tuple(x.shape)}"
        )

    return x


def preprocess_target_volume(volume, spacing_xyz, torch):
    norm, q005, q995 = normalize_volume_source_frozen(volume)

    sx, sy, _ = [float(v) for v in spacing_xyz]
    z, h, w = norm.shape

    out_h = int(round(h * sy / TARGET_SPACING_XY))
    out_w = int(round(w * sx / TARGET_SPACING_XY))

    if out_h <= 0 or out_w <= 0:
        raise RuntimeError("Invalid target resample shape.")

    x = torch.from_numpy(norm).unsqueeze(1)

    x = torch.nn.functional.interpolate(
        x,
        size=(out_h, out_w),
        mode="bilinear",
        align_corners=False,
    )
    x = center_crop_pad_2d_batch(x, torch)

    return (
        x[:, 0].numpy().astype(np.float16),
        {
            "q005": q005,
            "q995": q995,
            "original_shape_zyx": [int(z), int(h), int(w)],
            "resampled_shape_zyx": [int(z), int(out_h), int(out_w)],
        },
    )


def build_target_cache(header_df, stage_dir, torch):
    cache_dir = stage_dir / "target_image_only_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    images_path = cache_dir / "promise12_images_f16.npy"
    slices_path = cache_dir / "promise12_image_only_slices.csv"
    volumes_path = cache_dir / "promise12_image_only_volumes.csv"
    meta_path = cache_dir / "PROMISE12_IMAGE_ONLY_CACHE_META.json"

    if (
        images_path.is_file()
        and slices_path.is_file()
        and volumes_path.is_file()
        and meta_path.is_file()
    ):
        meta = load_json(meta_path)
        valid = (
            meta.get("status") == "PASS"
            and meta.get("patients") == N_TARGET_PATIENTS
            and meta.get("slices") == N_TARGET_SLICES
            and meta.get("promises12_gt_mhd_payload_read") is False
            and meta.get("promises12_gt_raw_payload_read") is False
            and meta.get("images_sha256") == sha256_file(images_path)
            and meta.get("slices_sha256") == sha256_file(slices_path)
            and meta.get("volumes_sha256") == sha256_file(volumes_path)
        )
        if valid:
            print("\nPROMISE12 IMAGE-ONLY CACHE=REUSE PASS")
            return (
                np.load(images_path, mmap_mode="r"),
                pd.read_csv(slices_path),
                pd.read_csv(volumes_path),
                meta,
            )

    images = np.lib.format.open_memmap(
        images_path,
        mode="w+",
        dtype=np.float16,
        shape=(N_TARGET_SLICES, SEG_SIZE, SEG_SIZE),
    )

    slice_rows = []
    volume_rows = []

    gi = 0

    for row in tqdm(
        list(header_df.itertuples(index=False)),
        total=len(header_df),
        desc="PROMISE12 image-only preprocessing",
        unit="patient",
        dynamic_ncols=True,
    ):
        vol, header = read_image_volume_from_zip(row)
        proc, pmeta = preprocess_target_volume(
            vol,
            header["spacing_xyz"],
            torch,
        )

        z = int(proc.shape[0])
        if z != int(row.dim_z):
            raise RuntimeError(
                f"{row.case_key}: processed slice count changed."
            )

        images[gi:gi + z] = proc

        for s in range(z):
            slice_rows.append({
                "global_index": gi + s,
                "case_id": int(row.case_id),
                "case_key": str(row.case_key),
                "slice_index": s,
            })

        volume_rows.append({
            "case_id": int(row.case_id),
            "case_key": str(row.case_key),
            "dim_x": int(row.dim_x),
            "dim_y": int(row.dim_y),
            "dim_z": int(row.dim_z),
            "spacing_x": float(row.spacing_x),
            "spacing_y": float(row.spacing_y),
            "spacing_z": float(row.spacing_z),
            "q005": pmeta["q005"],
            "q995": pmeta["q995"],
            "resampled_y": pmeta["resampled_shape_zyx"][1],
            "resampled_x": pmeta["resampled_shape_zyx"][2],
        })

        gi += z

    images.flush()

    if gi != N_TARGET_SLICES:
        raise RuntimeError(
            f"Target cache rows {gi} != {N_TARGET_SLICES}"
        )

    slice_df = pd.DataFrame(slice_rows)
    volume_df = pd.DataFrame(volume_rows)

    if slice_df["case_key"].nunique() != N_TARGET_PATIENTS:
        raise RuntimeError("Target cache patient count changed.")

    slice_df.to_csv(slices_path, index=False)
    volume_df.to_csv(volumes_path, index=False)

    meta = {
        "status": "PASS",
        "version": VERSION,
        "patients": N_TARGET_PATIENTS,
        "slices": N_TARGET_SLICES,
        "target_spacing_xy": TARGET_SPACING_XY,
        "seg_size": [SEG_SIZE, SEG_SIZE],
        "normalization": (
            "per-volume finite nonzero q0.5/q99.5 clip-map [0,1], "
            "background/nonfinite=0"
        ),
        "resampling": "in-plane bilinear only; no z resampling",
        "crop_pad": "center 352x352",
        "images": str(images_path),
        "images_sha256": sha256_file(images_path),
        "slices": str(slices_path),
        "slices_sha256": sha256_file(slices_path),
        "volumes": str(volumes_path),
        "volumes_sha256": sha256_file(volumes_path),
        "promises12_gt_mhd_payload_read": False,
        "promises12_gt_raw_payload_read": False,
        "gt_based_slice_selection": False,
    }
    save_json(meta_path, meta)

    return (
        np.load(images_path, mmap_mode="r"),
        slice_df,
        volume_df,
        meta,
    )


def model_input_from_numpy(image_batch, torch, device):
    arr = np.asarray(image_batch, dtype=np.float32)
    x = torch.from_numpy(arr).unsqueeze(1).repeat(1, 3, 1, 1)

    mean = torch.tensor(
        [0.485, 0.456, 0.406],
        dtype=x.dtype,
    ).view(1, 3, 1, 1)
    std = torch.tensor(
        [0.229, 0.224, 0.225],
        dtype=x.dtype,
    ).view(1, 3, 1, 1)

    x = (x - mean) / std
    return x.to(device, non_blocking=True)


def pack_masks(mask_batch):
    m = np.asarray(mask_batch, dtype=np.uint8)
    flat = m.reshape(m.shape[0], -1)
    return np.packbits(flat, axis=1, bitorder="little")


def unpack_masks(packed_batch):
    p = np.asarray(packed_batch, dtype=np.uint8)
    flat = np.unpackbits(
        p,
        axis=1,
        count=SEG_SIZE * SEG_SIZE,
        bitorder="little",
    )
    return flat.reshape(-1, SEG_SIZE, SEG_SIZE).astype(np.uint8)


def run_source_and_tent_family(
    family,
    cm3,
    cm4a,
    cm4d,
    images,
    slice_df,
    torch,
    nn,
    device,
    stage_dir,
):
    family_dir = stage_dir / "predictions" / family
    family_dir.mkdir(parents=True, exist_ok=True)

    source_path = family_dir / "source_masks_packbits.npy"
    tent_path = family_dir / "tent1_masks_packbits.npy"
    table_path = family_dir / "gt_free_prediction_diagnostics.csv"
    complete_path = family_dir / "COMPLETE.json"

    ckpt_path = Path(cm4d["final_models"][family]["checkpoint"])
    ckpt_sha = cm4d["final_models"][family]["checkpoint_sha256"]

    if complete_path.is_file():
        done = load_json(complete_path)
        valid = (
            done.get("status") == "PASS"
            and done.get("family") == family
            and done.get("checkpoint_sha256") == ckpt_sha
            and done.get("rows") == N_TARGET_SLICES
            and done.get("source_masks_sha256") == sha256_file(source_path)
            and done.get("tent1_masks_sha256") == sha256_file(tent_path)
            and done.get("diagnostics_sha256") == sha256_file(table_path)
            and done.get("promises12_gt_access") is False
        )
        if valid:
            print(f"{family}: GT-free predictions=REUSE PASS")
            return (
                np.load(source_path, mmap_mode="r"),
                np.load(tent_path, mmap_mode="r"),
                pd.read_csv(table_path),
                done,
            )

    model, _ = cm4a.build_loaded_model(
        cm3,
        family,
        ckpt_path,
        ckpt_sha,
        torch,
        device,
    )

    # First: frozen SOURCE inference at fixed batch=8, eval mode.
    model.eval()
    model.requires_grad_(False)

    source_packed = np.lib.format.open_memmap(
        source_path,
        mode="w+",
        dtype=np.uint8,
        shape=(N_TARGET_SLICES, PACKED_BYTES),
    )

    for start in tqdm(
        range(0, N_TARGET_SLICES, SEG_BATCH_SIZE),
        total=(N_TARGET_SLICES + SEG_BATCH_SIZE - 1) // SEG_BATCH_SIZE,
        desc=f"{family} PROMISE SOURCE",
        unit="batch",
        dynamic_ncols=True,
    ):
        end = min(start + SEG_BATCH_SIZE, N_TARGET_SLICES)
        x = model_input_from_numpy(images[start:end], torch, device)

        with torch.no_grad():
            with torch.autocast(
                device_type=device.type,
                enabled=(device.type == "cuda"),
                dtype=(
                    torch.float16
                    if device.type == "cuda"
                    else torch.bfloat16
                ),
            ):
                logits = model(x)

        mask = logits.detach().float().cpu().numpy()[:, 0] >= 0.0
        source_packed[start:end] = pack_masks(mask)

    source_packed.flush()

    # Rebuild pristine checkpoint before configuring TENT.
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    model, _ = cm4a.build_loaded_model(
        cm3,
        family,
        ckpt_path,
        ckpt_sha,
        torch,
        device,
    )

    cfg = cm4a.configure_tent(model, family, nn)
    source_values = cm4a.snapshot_params(cfg["params"])

    print(
        f"{family}: TENT affine tensors={len(cfg['params'])}, "
        f"numel={sum(p.numel() for p in cfg['params'])}, "
        f"ordinary_bn={len(cfg['ordinary_bn_modules'])}, "
        f"layernorm={len(cfg['layernorm_modules'])}, "
        f"unsafe_bn={len(cfg['unsafe_bn_modules'])}"
    )

    tent_packed = np.lib.format.open_memmap(
        tent_path,
        mode="w+",
        dtype=np.uint8,
        shape=(N_TARGET_SLICES, PACKED_BYTES),
    )

    rows = []

    for gi in tqdm(
        range(N_TARGET_SLICES),
        desc=f"{family} PROMISE TENT1",
        unit="slice",
        dynamic_ncols=True,
    ):
        x = model_input_from_numpy(images[gi:gi + 1], torch, device)

        cm4a.restore_params(cfg["params"], source_values)
        cm4a.set_tent_mode(model, cfg)

        optimizer = torch.optim.Adam(
            cfg["params"],
            lr=TENT_LR,
            weight_decay=TENT_WEIGHT_DECAY,
        )
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            enabled=(device.type == "cuda"),
            dtype=(
                torch.float16
                if device.type == "cuda"
                else torch.bfloat16
            ),
        ):
            pre_logits = model(x)
            pre_loss = cm4a.binary_entropy_from_logits(pre_logits)

        pre_entropy = float(pre_loss.detach().cpu())
        pre_loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            with torch.autocast(
                device_type=device.type,
                enabled=(device.type == "cuda"),
                dtype=(
                    torch.float16
                    if device.type == "cuda"
                    else torch.bfloat16
                ),
            ):
                post_logits = model(x)

        post_entropy = cm4a.binary_entropy_scalar(post_logits)
        tent_mask = (
            post_logits.detach().float().cpu().numpy()[0, 0] >= 0.0
        )

        source_mask = unpack_masks(
            np.asarray(source_packed[gi:gi + 1])
        )[0].astype(bool)

        tent_packed[gi] = pack_masks(
            tent_mask[None, ...]
        )[0]

        row = slice_df.iloc[gi]
        rows.append({
            "family": family,
            "global_index": gi,
            "case_key": str(row["case_key"]),
            "slice_index": int(row["slice_index"]),
            "source_fg_fraction": float(source_mask.mean()),
            "tent_fg_fraction": float(tent_mask.mean()),
            "source_tent_mask_disagreement": float(
                np.mean(source_mask != tent_mask)
            ),
            "tent_pre_entropy": pre_entropy,
            "tent_post_entropy": post_entropy,
            "tent_entropy_change": post_entropy - pre_entropy,
        })

    tent_packed.flush()

    diag = pd.DataFrame(rows)
    diag.to_csv(table_path, index=False)

    done = {
        "status": "PASS",
        "version": VERSION,
        "family": family,
        "rows": N_TARGET_SLICES,
        "patients": N_TARGET_PATIENTS,
        "checkpoint": str(ckpt_path),
        "checkpoint_sha256": ckpt_sha,
        "source_inference_batch_size": SEG_BATCH_SIZE,
        "tent_action": {
            "name": "A1_TENT_1STEP",
            "optimizer": "Adam",
            "lr": TENT_LR,
            "weight_decay": TENT_WEIGHT_DECAY,
            "steps": TENT_STEPS,
            "episodic_reset_per_slice": True,
        },
        "source_masks": str(source_path),
        "source_masks_sha256": sha256_file(source_path),
        "tent1_masks": str(tent_path),
        "tent1_masks_sha256": sha256_file(tent_path),
        "diagnostics": str(table_path),
        "diagnostics_sha256": sha256_file(table_path),
        "promises12_gt_access": False,
    }
    save_json(complete_path, done)

    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return (
        np.load(source_path, mmap_mode="r"),
        np.load(tent_path, mmap_mode="r"),
        diag,
        done,
    )


def preflight_one_target_slice(
    header_df,
    cm3,
    cm4a,
    cm4b,
    cm4c_helper,
    cm4c_lock,
    cm4d,
    dino_model,
    safety_bundle,
    torch,
    nn,
    F,
    device,
):
    print("\n===== CM5A ONE-TARGET-SLICE PREFLIGHT =====")

    row = next(header_df.itertuples(index=False))
    volume, header = read_image_volume_from_zip(row)
    proc, _ = preprocess_target_volume(
        volume,
        header["spacing_xyz"],
        torch,
    )

    mid = int(proc.shape[0] // 2)
    img = proc[mid:mid + 1]

    for family in FAMILIES:
        ckpt_path = Path(cm4d["final_models"][family]["checkpoint"])
        ckpt_sha = cm4d["final_models"][family]["checkpoint_sha256"]

        model, _ = cm4a.build_loaded_model(
            cm3,
            family,
            ckpt_path,
            ckpt_sha,
            torch,
            device,
        )
        model.eval()

        x = model_input_from_numpy(img, torch, device)

        with torch.no_grad():
            with torch.autocast(
                device_type=device.type,
                enabled=(device.type == "cuda"),
                dtype=(
                    torch.float16
                    if device.type == "cuda"
                    else torch.bfloat16
                ),
            ):
                src_logits = model(x)

        source_mask = (
            src_logits.detach().float().cpu().numpy()[0, 0] >= 0.0
        )

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        model, _ = cm4a.build_loaded_model(
            cm3,
            family,
            ckpt_path,
            ckpt_sha,
            torch,
            device,
        )
        cfg = cm4a.configure_tent(model, family, nn)
        source_values = cm4a.snapshot_params(cfg["params"])
        cm4a.restore_params(cfg["params"], source_values)
        cm4a.set_tent_mode(model, cfg)

        optimizer = torch.optim.Adam(
            cfg["params"],
            lr=TENT_LR,
            weight_decay=TENT_WEIGHT_DECAY,
        )
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            enabled=(device.type == "cuda"),
            dtype=(
                torch.float16
                if device.type == "cuda"
                else torch.bfloat16
            ),
        ):
            pre_logits = model(x)
            pre_loss = cm4a.binary_entropy_from_logits(pre_logits)

        pre_ent = float(pre_loss.detach().cpu())
        pre_loss.backward()
        optimizer.step()

        with torch.no_grad():
            with torch.autocast(
                device_type=device.type,
                enabled=(device.type == "cuda"),
                dtype=(
                    torch.float16
                    if device.type == "cuda"
                    else torch.bfloat16
                ),
            ):
                post_logits = model(x)

        post_ent = cm4a.binary_entropy_scalar(post_logits)
        tent_mask = (
            post_logits.detach().float().cpu().numpy()[0, 0] >= 0.0
        )

        # Frozen prediction-conditioned DINO representation from SOURCE mask.
        dino_x = cm4b.dino_preprocess(img, torch, F, device)

        with torch.inference_mode():
            with torch.autocast(
                device_type=device.type,
                enabled=(device.type == "cuda"),
                dtype=(
                    torch.float16
                    if device.type == "cuda"
                    else torch.bfloat16
                ),
            ):
                dino_out = dino_model(pixel_values=dino_x)

        tokens = dino_out.last_hidden_state[:, 1:, :].float()
        mb = source_mask[None, ...].astype(np.uint8)
        occ = cm4b.source_mask_occupancy(mb, torch, device)
        fg, bg = cm4b.pool_conditioned(tokens, occ, torch)
        m2 = cm4b.m2_from_masks(mb)

        raw_feat = np.concatenate(
            [
                fg.cpu().numpy().astype(np.float32),
                bg.cpu().numpy().astype(np.float32),
                m2.astype(np.float32),
            ],
            axis=1,
        )

        if raw_feat.shape != (1, FINAL_RAW_DIM):
            raise RuntimeError(
                f"{family}: preflight safety feature shape changed."
            )

        score = float(
            cm4c_helper.score_bundle(
                safety_bundle,
                raw_feat,
            )[0]
        )

        print(
            f"{family}: PASS "
            f"case={row.case_key} slice={mid} "
            f"SOURCE_fg={source_mask.mean():.6f} "
            f"TENT_fg={tent_mask.mean():.6f} "
            f"Hpre={pre_ent:.6f} Hpost={post_ent:.6f} "
            f"risk={score:.6f}"
        )

        del model, optimizer
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("PROMISE12 GT MHD payload read: NO")
    print("PROMISE12 GT RAW payload read: NO")
    print("Target threshold tuning: NO")
    print("CM5A target preflight: PASS")


def extract_target_features_and_scores(
    cm4b,
    cm4c_helper,
    cm4c_lock,
    dino_model,
    safety_bundle,
    images,
    source_packed_by_family,
    diagnostics_by_family,
    slice_df,
    torch,
    F,
    device,
    stage_dir,
):
    feature_path = stage_dir / "CM5A_TARGET_RAW_SAFETY_FEATURES_F16.npy"
    score_path = stage_dir / "CM5A_PROMISE12_GT_FREE_SAFETY_SCORES.csv"
    meta_path = stage_dir / "CM5A_TARGET_SAFETY_FEATURE_META.json"

    expected_shape = (
        len(FAMILIES),
        N_TARGET_SLICES,
        FINAL_RAW_DIM,
    )

    features = np.lib.format.open_memmap(
        feature_path,
        mode="w+",
        dtype=np.float16,
        shape=expected_shape,
    )

    score_rows = []

    family_to_idx = {f: i for i, f in enumerate(FAMILIES)}

    deployment_threshold = float(
        cm4c_lock["frozen_source_operating_point"]
        ["final_refit_deployment"]["threshold"]
    )

    print("\n===== TARGET PREDICTION-CONDITIONED DINO SAFETY SCORING =====")
    print("Frozen SOURCE deployment threshold:", deployment_threshold)

    for start in tqdm(
        range(0, N_TARGET_SLICES, DINO_BATCH_SIZE),
        total=(N_TARGET_SLICES + DINO_BATCH_SIZE - 1) // DINO_BATCH_SIZE,
        desc="PROMISE DINO safety",
        unit="batch",
        dynamic_ncols=True,
    ):
        end = min(start + DINO_BATCH_SIZE, N_TARGET_SLICES)
        gi = np.arange(start, end, dtype=np.int64)

        dino_x = cm4b.dino_preprocess(
            images[gi],
            torch,
            F,
            device,
        )

        with torch.inference_mode():
            with torch.autocast(
                device_type=device.type,
                enabled=(device.type == "cuda"),
                dtype=(
                    torch.float16
                    if device.type == "cuda"
                    else torch.bfloat16
                ),
            ):
                out = dino_model(pixel_values=dino_x)

        tokens = out.last_hidden_state[:, 1:, :].float()

        for family in FAMILIES:
            fi = family_to_idx[family]
            mb = unpack_masks(
                np.asarray(source_packed_by_family[family][gi])
            )

            occ = cm4b.source_mask_occupancy(
                mb,
                torch,
                device,
            )
            fg, bg = cm4b.pool_conditioned(
                tokens,
                occ,
                torch,
            )
            m2 = cm4b.m2_from_masks(mb)

            raw_feat = np.concatenate(
                [
                    fg.cpu().numpy().astype(np.float32),
                    bg.cpu().numpy().astype(np.float32),
                    m2.astype(np.float32),
                ],
                axis=1,
            )

            if raw_feat.shape != (len(gi), FINAL_RAW_DIM):
                raise RuntimeError(
                    f"{family}: target safety feature shape changed."
                )
            if not np.isfinite(raw_feat).all():
                raise RuntimeError(
                    f"{family}: target safety feature nonfinite."
                )

            features[fi, start:end] = raw_feat.astype(np.float16)

            scores = cm4c_helper.score_bundle(
                safety_bundle,
                raw_feat,
            )
            if not np.isfinite(scores).all():
                raise RuntimeError(
                    f"{family}: target safety score nonfinite."
                )

            diag = diagnostics_by_family[family].iloc[start:end]

            for local, global_i in enumerate(gi.tolist()):
                row = slice_df.iloc[global_i]
                drow = diag.iloc[local]
                score = float(scores[local])

                score_rows.append({
                    "family": family,
                    "global_index": global_i,
                    "case_key": str(row["case_key"]),
                    "slice_index": int(row["slice_index"]),
                    "risk_score": score,
                    "frozen_source_threshold": deployment_threshold,
                    "fixed_source_gate_retain_source": int(
                        score >= deployment_threshold
                    ),
                    "fixed_source_gate_adapt": int(
                        score < deployment_threshold
                    ),
                    "source_fg_fraction": float(drow["source_fg_fraction"]),
                    "tent_fg_fraction": float(drow["tent_fg_fraction"]),
                    "source_tent_mask_disagreement": float(
                        drow["source_tent_mask_disagreement"]
                    ),
                    "tent_pre_entropy": float(drow["tent_pre_entropy"]),
                    "tent_post_entropy": float(drow["tent_post_entropy"]),
                    "tent_entropy_change": float(drow["tent_entropy_change"]),
                })

    features.flush()

    score_df = pd.DataFrame(score_rows).sort_values(
        ["family", "global_index"]
    ).reset_index(drop=True)

    if len(score_df) != N_TARGET_SLICES * len(FAMILIES):
        raise RuntimeError("Target score table row count changed.")

    score_df.to_csv(score_path, index=False)

    summary = (
        score_df.groupby("family", as_index=False)
        .agg(
            rows=("risk_score", "size"),
            patients=("case_key", "nunique"),
            mean_risk=("risk_score", "mean"),
            median_risk=("risk_score", "median"),
            fixed_source_adaptation_coverage=(
                "fixed_source_gate_adapt",
                "mean",
            ),
            mean_source_tent_disagreement=(
                "source_tent_mask_disagreement",
                "mean",
            ),
            mean_entropy_change=("tent_entropy_change", "mean"),
        )
    )

    print("\n===== CM5A GT-FREE TARGET SCORE SUMMARY =====")
    print(summary.to_string(index=False))

    meta = {
        "status": "PASS",
        "version": VERSION,
        "feature_shape": list(expected_shape),
        "feature_dtype": "float16",
        "family_order": FAMILIES,
        "representation": "CondDINO FG768 + BG768 + M2",
        "feature_file": str(feature_path),
        "feature_sha256": sha256_file(feature_path),
        "score_file": str(score_path),
        "score_sha256": sha256_file(score_path),
        "frozen_source_deployment_threshold": deployment_threshold,
        "fixed_source_gate_only": True,
        "unlabeled_threshold_transport_performed": False,
        "promises12_gt_access": False,
    }
    save_json(meta_path, meta)

    return (
        np.load(feature_path, mmap_mode="r"),
        score_df,
        summary,
        meta,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    ap.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "Access image-only PROMISE12 headers plus one image RAW volume, "
            "then run one real SOURCE/TENT1/DINO safety slice per family. "
            "No target GT payload is opened."
        ),
    )
    args = ap.parse_args()

    print("===== Q1X CM5A PROMISE12 GT-FREE TARGET INFERENCE =====")
    print("TARGET_IMAGE_ACCESS=YES")
    print("PROMISE12_GT_MHD_PAYLOAD_READ=NO")
    print("PROMISE12_GT_RAW_PAYLOAD_READ=NO")
    print("GT_BASED_SLICE_SELECTION=NO")
    print("TARGET_GT=NO")
    print("TARGET_HARM_LABELS=NO")
    print("TARGET_THRESHOLD_TUNING=NO")
    print("UNLABELED_THRESHOLD_TRANSPORT=NO")
    print("ALL_PROMISE_IMAGE_SLICES_INCLUDED=YES")
    print("EXPECTED_PATIENTS=", N_TARGET_PATIENTS)
    print("EXPECTED_SLICES=", N_TARGET_SLICES)
    print("SOURCE_INFERENCE_BATCH_SIZE=", SEG_BATCH_SIZE)
    print("TENT=episodic single-slice A1_TENT_1STEP")
    print("TARGET_SPACING_XY=", TARGET_SPACING_XY)

    cm2b, cm4c_lock, cm4d = verify_upstream_lineage()
    verify_promise_zip()

    cm3 = import_helper(
        "CM3_HELPER",
        CM3_HELPER,
        EXPECTED_CM3_HELPER_SHA,
        "q1x_cm3_fix3",
    )
    cm4a = import_helper(
        "CM4A_HELPER",
        CM4A_HELPER,
        EXPECTED_CM4A_HELPER_SHA,
        "q1x_cm4a_fix4",
    )
    cm4b = import_helper(
        "CM4B_HELPER",
        CM4B_HELPER,
        EXPECTED_CM4B_HELPER_SHA,
        "q1x_cm4b_fix1",
    )
    cm4c_helper = import_helper(
        "CM4C_HELPER",
        CM4C_HELPER,
        EXPECTED_CM4C_HELPER_SHA,
        "q1x_cm4c_fix2",
    )

    header_df = build_image_only_header_manifest()

    if args.output_dir.exists():
        stage_dir = args.output_dir
        print("\nSTAGE_DIR=RESUME", stage_dir)
    else:
        stage_dir = args.output_dir
        stage_dir.mkdir(parents=True, exist_ok=False)
        print("\nSTAGE_DIR=NEW", stage_dir)

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\n===== RUNTIME =====")
    print("torch:", torch.__version__)
    print("device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(device))

    # Final frozen safety estimator.
    estimator_path = Path(
        cm4c_lock["final_estimator"]["path"]
    )
    estimator_sha = cm4c_lock["final_estimator"]["sha256"]

    if sha256_file(estimator_path) != estimator_sha:
        raise RuntimeError(
            "CM4C final safety estimator SHA mismatch."
        )
    safety_bundle = joblib.load(estimator_path)

    # Frozen DINO identity is carried through CM4C.
    dino_lock = cm4c_lock["frozen_representation"]["dino_lock"]
    dino_model = cm4b.load_dino_model(
        dino_lock,
        device,
    )

    if args.preflight_only:
        preflight_one_target_slice(
            header_df,
            cm3,
            cm4a,
            cm4b,
            cm4c_helper,
            cm4c_lock,
            cm4d,
            dino_model,
            safety_bundle,
            torch,
            nn,
            F,
            device,
        )

        preflight_path = stage_dir / "CM5A_PREFLIGHT.json"
        save_json(
            preflight_path,
            {
                "status": "PASS",
                "version": VERSION,
                "cm2b_lock_sha256": EXPECTED_CM2B_LOCK_SHA,
                "cm4c_lock_sha256": EXPECTED_CM4C_LOCK_SHA,
                "cm4d_lock_sha256": EXPECTED_CM4D_LOCK_SHA,
                "promise_zip_sha256": EXPECTED_PROMISE_ZIP_SHA256,
                "patients": N_TARGET_PATIENTS,
                "image_only_slices": N_TARGET_SLICES,
                "promises12_gt_mhd_payload_read": False,
                "promises12_gt_raw_payload_read": False,
                "target_threshold_tuning": False,
                "decision": (
                    "PROMISE12_GT_FREE_PREFLIGHT_PASS_READY_FOR_FULL_CM5A"
                ),
            },
        )

        print("\n===== CM5A PREFLIGHT-ONLY FINAL =====")
        print(
            "Decision= "
            "PROMISE12_GT_FREE_PREFLIGHT_PASS_READY_FOR_FULL_CM5A"
        )
        print("PROMISE12 GT MHD payload read: NO")
        print("PROMISE12 GT RAW payload read: NO")
        print("PASS")
        return

    images, slice_df, volume_df, cache_meta = build_target_cache(
        header_df,
        stage_dir,
        torch,
    )

    source_packed_by_family = {}
    tent_packed_by_family = {}
    diagnostics_by_family = {}
    family_sidecars = {}

    for family in FAMILIES:
        print(f"\n########## PROMISE12 {family} ##########")

        (
            source_packed,
            tent_packed,
            diag,
            done,
        ) = run_source_and_tent_family(
            family,
            cm3,
            cm4a,
            cm4d,
            images,
            slice_df,
            torch,
            nn,
            device,
            stage_dir,
        )

        source_packed_by_family[family] = source_packed
        tent_packed_by_family[family] = tent_packed
        diagnostics_by_family[family] = diag
        family_sidecars[family] = done

    (
        target_features,
        score_df,
        score_summary,
        feature_meta,
    ) = extract_target_features_and_scores(
        cm4b,
        cm4c_helper,
        cm4c_lock,
        dino_model,
        safety_bundle,
        images,
        source_packed_by_family,
        diagnostics_by_family,
        slice_df,
        torch,
        F,
        device,
        stage_dir,
    )

    summary_path = stage_dir / "CM5A_GT_FREE_TARGET_SUMMARY.csv"
    score_summary.to_csv(summary_path, index=False)

    lock = {
        "status": "PASS",
        "decision": PASS_DECISION,
        "version": VERSION,
        "build": BUILD,
        "cm2b_lock_sha256": EXPECTED_CM2B_LOCK_SHA,
        "cm4c_lock_sha256": EXPECTED_CM4C_LOCK_SHA,
        "cm4d_lock_sha256": EXPECTED_CM4D_LOCK_SHA,
        "cm3_helper_sha256": EXPECTED_CM3_HELPER_SHA,
        "cm4a_helper_sha256": EXPECTED_CM4A_HELPER_SHA,
        "cm4b_helper_sha256": EXPECTED_CM4B_HELPER_SHA,
        "cm4c_helper_sha256": EXPECTED_CM4C_HELPER_SHA,
        "promise_archive": {
            "path": str(PROMISE_ZIP),
            "size": EXPECTED_PROMISE_ZIP_SIZE,
            "md5": EXPECTED_PROMISE_ZIP_MD5,
            "sha256": EXPECTED_PROMISE_ZIP_SHA256,
        },
        "target": {
            "patients": N_TARGET_PATIENTS,
            "image_only_slices": N_TARGET_SLICES,
            "all_image_slices_included": True,
            "gt_based_slice_selection": False,
        },
        "preprocessing": {
            "target_spacing_xy": TARGET_SPACING_XY,
            "z_resampling": False,
            "seg_size": [SEG_SIZE, SEG_SIZE],
            "normalization": (
                "per-volume finite nonzero q0.5/q99.5 -> [0,1]"
            ),
        },
        "segmentation": {
            "families": FAMILIES,
            "final_source_models": {
                family: {
                    "checkpoint": cm4d["final_models"][family]["checkpoint"],
                    "checkpoint_sha256": cm4d["final_models"][family][
                        "checkpoint_sha256"
                    ],
                }
                for family in FAMILIES
            },
            "source_inference_batch_size": SEG_BATCH_SIZE,
            "tent_action": "A1_TENT_1STEP episodic single-slice",
        },
        "safety": {
            "final_estimator": str(estimator_path),
            "final_estimator_sha256": estimator_sha,
            "frozen_source_deployment_threshold": float(
                cm4c_lock["frozen_source_operating_point"]
                ["final_refit_deployment"]["threshold"]
            ),
            "representation": "CondDINO FG768 + BG768 + M2",
            "target_threshold_tuning": False,
            "unlabeled_threshold_transport_performed": False,
        },
        "information_boundary": {
            "promises12_image_access": True,
            "promises12_gt_mhd_payload_read": False,
            "promises12_gt_raw_payload_read": False,
            "target_gt": False,
            "target_harm_labels": False,
            "target_delta_dice": False,
            "target_threshold_tuning": False,
        },
        "artifacts": {
            "target_cache_meta": str(
                stage_dir
                / "target_image_only_cache"
                / "PROMISE12_IMAGE_ONLY_CACHE_META.json"
            ),
            "target_cache_meta_sha256": sha256_file(
                stage_dir
                / "target_image_only_cache"
                / "PROMISE12_IMAGE_ONLY_CACHE_META.json"
            ),
            "target_features": feature_meta["feature_file"],
            "target_features_sha256": feature_meta["feature_sha256"],
            "target_scores": feature_meta["score_file"],
            "target_scores_sha256": feature_meta["score_sha256"],
            "summary": str(summary_path),
            "summary_sha256": sha256_file(summary_path),
            "family_predictions": {
                family: family_sidecars[family]
                for family in FAMILIES
            },
        },
        "next_stage": (
            "CM5B_UNLABELED_SUPPORT_AWARE_OPERATING_POINT_TRANSPORT_LOCK"
        ),
    }

    lock_path = (
        stage_dir
        / "CM5A_PROMISE12_GT_FREE_PREDICTION_SAFETY_LOCK.json"
    )
    save_json(lock_path, lock)

    print("\n===== CM5A FINAL =====")
    print(score_summary.to_string(index=False))
    print("PROMISE12 GT MHD payload read: NO")
    print("PROMISE12 GT RAW payload read: NO")
    print("Target HARM labels: NO")
    print("Target threshold tuning: NO")
    print("Unlabeled threshold transport performed: NO")
    print("Decision=", PASS_DECISION)
    print("LOCK=", lock_path)
    print("LOCK SHA256=", sha256_file(lock_path))
    print("PASS")


if __name__ == "__main__":
    main()
