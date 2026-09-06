#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1X_CM2B_prostate_mri_preprocessing_and_model_protocol_lock_fix1.py

SafeTTA Q1 enhancement — CM2B.

This stage freezes the second-task preprocessing, SOURCE patient folds, and
segmentation-model training protocol before any model is trained.

Inputs:
- exact CM2A Fix3 PASS lock;
- exact SOURCE official case manifest;
- exact PROMISE12 image-only patient manifest.

No PROMISE12 reference-mask RAW payload is opened.
No model is trained.
No TTA is run.
No safety model is fit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm


VERSION = "2026-09-04-Q1X-CM2B-v1-fix1"
BUILD = "Q1X_CM2B_PROSTATE_MRI_PREPROCESSING_AND_MODEL_PROTOCOL_LOCK_FIX1"

ROOT = Path(r"F:\MEDSEG_SAFETTA\cross_modality_prostate_mri")
OUT = Path(r"F:\MEDSEG_SAFETTA\outputs")

CM2A_DIR = (
    OUT
    / "Q1X_CM2A_prostate158_official_manifest_t2_resolver_and_slice_preflight_fix3_v1"
)
CM2A_LOCK = CM2A_DIR / "CM2A_PROSTATE_MRI_SAMPLE_UNIT_LOCK.json"
EXPECTED_CM2A_LOCK_SHA = (
    "ef9a574911184ad5f049729f82488c8a1fc41e43f2c9e15bce76480fd68b5311"
)

SOURCE_MANIFEST = CM2A_DIR / "CM2A_PROSTATE158_OFFICIAL_T2_CASE_MANIFEST.csv"
P12_PATIENT_MANIFEST = CM2A_DIR / "CM2A_PROMISE12_IMAGE_ONLY_PATIENT_MANIFEST.csv"
P12_SLICE_MANIFEST = CM2A_DIR / "CM2A_PROMISE12_IMAGE_ONLY_SLICE_MANIFEST.csv"

DEFAULT_OUTPUT = (
    OUT
    / "Q1X_CM2B_prostate_mri_preprocessing_and_model_protocol_lock_fix1_v1"
)

FOLD_SEED = 20260904
N_FOLDS = 5
INPUT_SIZE = 352
DINO_SIZE = 224

ROBUST_LOW_Q = 0.5
ROBUST_HIGH_Q = 99.5

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DECISION_PASS = (
    "MRI_PREPROCESSING_AND_SOURCE_MODEL_PROTOCOL_LOCKED_READY_FOR_CM3_OOF_TRAINING"
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


def import_nibabel():
    try:
        import nibabel as nib
    except Exception as e:
        raise RuntimeError("nibabel is required for CM2B.") from e
    return nib


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


def freeze_source_spacing(nib, source_df: pd.DataFrame):
    print("\n===== SOURCE-ONLY SPACING FREEZE =====")

    rows = []
    for r in tqdm(
        source_df.itertuples(index=False),
        total=len(source_df),
        desc="source header audit",
    ):
        img = nib.load(r.t2_path)
        zooms = tuple(float(x) for x in img.header.get_zooms()[:3])
        axcodes = tuple(str(x) for x in nib.aff2axcodes(img.affine))

        rows.append({
            "case_key": r.case_key,
            "spacing_x": zooms[0],
            "spacing_y": zooms[1],
            "spacing_z": zooms[2],
            "axcode_0": axcodes[0],
            "axcode_1": axcodes[1],
            "axcode_2": axcodes[2],
        })

    spacing_df = pd.DataFrame(rows)

    target_x = float(np.median(spacing_df["spacing_x"].to_numpy()))
    target_y = float(np.median(spacing_df["spacing_y"].to_numpy()))

    print(
        "source spacing X min/median/max:",
        float(spacing_df["spacing_x"].min()),
        target_x,
        float(spacing_df["spacing_x"].max()),
    )
    print(
        "source spacing Y min/median/max:",
        float(spacing_df["spacing_y"].min()),
        target_y,
        float(spacing_df["spacing_y"].max()),
    )
    print(
        "source spacing Z min/median/max:",
        float(spacing_df["spacing_z"].min()),
        float(spacing_df["spacing_z"].median()),
        float(spacing_df["spacing_z"].max()),
    )

    orientation_counts = (
        spacing_df[["axcode_0", "axcode_1", "axcode_2"]]
        .astype(str)
        .agg("".join, axis=1)
        .value_counts()
        .to_dict()
    )
    print("source orientation-code counts:", orientation_counts)

    return spacing_df, target_x, target_y


def audit_external_image_headers_only(p12_patient_df: pd.DataFrame):
    print("\n===== PROMISE12 IMAGE-ONLY HEADER AUDIT =====")

    rows = []
    for r in tqdm(
        p12_patient_df.itertuples(index=False),
        total=len(p12_patient_df),
        desc="external image headers",
    ):
        h = parse_mhd_header(Path(r.image_mhd))
        spacing = [float(x) for x in h["ElementSpacing"].split()]
        dims = [int(x) for x in h["DimSize"].split()]

        rows.append({
            "patient_id": r.patient_id,
            "spacing_x": spacing[0],
            "spacing_y": spacing[1],
            "spacing_z": spacing[2],
            "dim_x": dims[0],
            "dim_y": dims[1],
            "dim_z": dims[2],
            "element_type": h.get("ElementType"),
            "gt_header_opened": False,
            "gt_raw_opened": False,
        })

    ext_df = pd.DataFrame(rows)

    print(
        "external image spacing X min/median/max:",
        float(ext_df["spacing_x"].min()),
        float(ext_df["spacing_x"].median()),
        float(ext_df["spacing_x"].max()),
    )
    print(
        "external image spacing Y min/median/max:",
        float(ext_df["spacing_y"].min()),
        float(ext_df["spacing_y"].median()),
        float(ext_df["spacing_y"].max()),
    )
    print(
        "external image spacing Z min/median/max:",
        float(ext_df["spacing_z"].min()),
        float(ext_df["spacing_z"].median()),
        float(ext_df["spacing_z"].max()),
    )
    print("PROMISE12 GT header opened in CM2B: NO")
    print("PROMISE12 GT RAW opened in CM2B: NO")

    return ext_df


def build_source_patient_folds(source_df: pd.DataFrame):
    print("\n===== SOURCE PATIENT-LEVEL 5-FOLD LOCK =====")

    work = source_df.copy().sort_values("case_key").reset_index(drop=True)

    order = np.argsort(
        work["n_source_gt_positive_slices"].to_numpy(),
        kind="stable",
    )
    rank = np.empty(len(work), dtype=int)
    rank[order] = np.arange(len(work))

    burden_stratum = np.floor(rank * 5 / len(work)).astype(int)
    burden_stratum = np.clip(burden_stratum, 0, 4)
    work["burden_stratum"] = burden_stratum
    work["fold"] = -1

    splitter = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=FOLD_SEED,
    )

    dummy_x = np.zeros(len(work), dtype=np.uint8)
    for fold, (_, val_idx) in enumerate(
        splitter.split(dummy_x, work["burden_stratum"].to_numpy())
    ):
        work.loc[val_idx, "fold"] = fold

    if (work["fold"] < 0).any():
        raise RuntimeError("At least one source patient was not assigned a fold.")
    if work["case_key"].nunique() != len(work):
        raise RuntimeError("SOURCE case IDs are not unique.")

    print("fold seed:", FOLD_SEED)
    print("fold counts:", work["fold"].value_counts().sort_index().to_dict())

    for fold in range(N_FOLDS):
        part = work[work["fold"] == fold]
        print(
            f"fold {fold}: patients={len(part)}, "
            f"positive_slices={int(part['n_source_gt_positive_slices'].sum())}, "
            f"total_slices={int(part['n_slices'].sum())}"
        )

    return work


def build_frozen_protocol(target_spacing_x: float, target_spacing_y: float):
    return {
        "task": "binary whole-gland prostate segmentation",
        "source_dataset": "Prostate158 train+valid official 139 cases",
        "external_dataset": "PROMISE12 public 50 cases",
        "sample_unit": "axial 2D slice",
        "patient_grouping": True,
        "source_oof_folds": N_FOLDS,
        "source_oof_seed": FOLD_SEED,

        "spatial_preprocessing": {
            "slice_axis": 2,
            "resample_in_plane": True,
            "target_spacing_source_only_mm": [
                target_spacing_x,
                target_spacing_y,
            ],
            "image_interpolation": "bilinear",
            "label_interpolation": "nearest",
            "through_plane_resampling": False,
            "post_resample_geometry": (
                f"center crop if larger and symmetric zero-pad if smaller "
                f"to {INPUT_SIZE}x{INPUT_SIZE}"
            ),
            "segmentation_input_hw": [INPUT_SIZE, INPUT_SIZE],
            "dino_input_hw": [DINO_SIZE, DINO_SIZE],
        },

        "intensity_preprocessing": {
            "scope": "per-volume independently; formula frozen before model training",
            "valid_voxels_for_quantiles": "finite voxels with intensity != 0",
            "low_quantile_percent": ROBUST_LOW_Q,
            "high_quantile_percent": ROBUST_HIGH_Q,
            "transform": (
                "clip to per-volume q0.5/q99.5, linearly map to [0,1], "
                "background/non-finite set to 0"
            ),
            "histogram_matching": False,
            "target_distribution_fitting": False,
            "target_parameter_tuning": False,
        },

        "channel_protocol": {
            "segmentation_input": "normalized grayscale repeated to 3 channels",
            "segmentation_backbone_normalization": {
                "mean": IMAGENET_MEAN,
                "std": IMAGENET_STD,
            },
            "safety_dino_input": (
                "same normalized grayscale repeated to RGB before frozen "
                "DINOv2 image processor; no adapted prediction or GT"
            ),
            "adjacent_slice_2p5d": False,
        },

        "source_label": {
            "whole_gland": "annotation > 0",
            "observed_source_values": [0, 1, 2],
            "segmentation_probability_threshold": 0.5,
            "zero_zero_dice": 1.0,
        },

        "segmentation_model_panel": {
            "UNet": {
                "family": "convolutional encoder-decoder",
                "implementation": "project-local vanilla 2D U-Net",
                "input_channels": 3,
                "base_channels": 32,
                "levels": 5,
                "pretrained": False,
                "output_channels": 1,
            },
            "DeepLabV3_R50": {
                "family": "dilated CNN",
                "implementation": "torchvision deeplabv3_resnet50",
                "backbone_initialization": "ImageNet pretrained ResNet-50",
                "input_channels": 3,
                "output_channels": 1,
            },
            "SegFormer_B0": {
                "family": "transformer segmentation",
                "implementation": "HuggingFace SegFormer / nvidia mit-b0",
                "backbone_initialization": "ImageNet pretrained MiT-B0",
                "input_channels": 3,
                "output_channels": 1,
            },
        },

        "training_protocol": {
            "source_only": True,
            "oof_training": "5 patient-level folds independently per model family",
            "final_external_model": (
                "after OOF protocol is frozen/completed, one final checkpoint "
                "per family trained on all 139 source patients using source-only "
                "epoch selection derived from OOF runs"
            ),
            "loss": "0.5 * BCEWithLogits + 0.5 * soft Dice loss",
            "optimizer": "AdamW",
            "learning_rate": 1e-4,
            "weight_decay": 1e-4,
            "max_epochs": 50,
            "batch_size": 8,
            "amp": True,
            "early_stopping_patience": 10,
            "checkpoint_metric": (
                "mean patient-level 3D whole-gland Dice on SOURCE validation fold"
            ),
            "checkpoint_direction": "maximize",
            "augmentations": {
                "horizontal_flip_p": 0.5,
                "rotation_degrees": 10,
                "scale_range": [0.9, 1.1],
                "external_augmentation": False,
            },
            "empty_source_slices_included": True,
            "external_slices_all_included_pre_gt": True,
        },

        "tta_protocol": {
            "primary_action": "TENT1",
            "episodic_reset": True,
            "description": (
                "same conceptual one-step entropy-minimization action as main "
                "SafeTTA study"
            ),
            "action_search": False,
        },

        "safety_protocol": {
            "representation_search": False,
            "method": (
                "frozen DINOv2-base + SOURCE-mask FG/BG pooling + PCA64 + "
                "M2(fg_fraction,boundary_density) + balanced logistic regression"
            ),
            "harm": "DeltaDice <= -0.02",
            "benefit": "DeltaDice >= +0.02",
            "source_only_fit": True,
            "external_primary_endpoint": "patient-clustered AUROC",
            "external_success_rule": (
                "95% clustered-bootstrap AUROC lower bound > 0.50"
            ),
            "external_threshold_retuning": False,
        },

        "external_information_boundary": {
            "PROMISE12_gt_header_read": False,
            "PROMISE12_gt_raw_decode": False,
            "PROMISE12_gt_based_slice_filtering": False,
            "PROMISE12_parameter_tuning": False,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    args = ap.parse_args()

    print("===== Q1X CM2B MRI PREPROCESSING / MODEL PROTOCOL LOCK =====")
    print("STATUS=PRETRAINING_PROTOCOL_LOCK")
    print("MODEL_TRAINING=NO")
    print("TTA=NO")
    print("SAFETY_MODEL_FITTING=NO")
    print("PROMISE12_GT_HEADER_READ=NO")
    print("PROMISE12_GT_RAW_DECODE=NO")
    print("TARGET_TUNING=NO")
    print("REPRESENTATION_SEARCH=NO")

    print("\n===== CM2A LINEAGE GATE =====")
    assert_sha(CM2A_LOCK, EXPECTED_CM2A_LOCK_SHA, "CM2A_LOCK")
    cm2a = load_json(CM2A_LOCK)
    if cm2a.get("status") != "PASS":
        raise RuntimeError("CM2A status changed.")
    if cm2a.get("decision") != (
        "SOURCE_T2_SERIES_AND_SLICE_SAMPLE_UNIT_LOCKED_READY_FOR_CM2B_PREPROCESSING_LOCK"
    ):
        raise RuntimeError("CM2A decision changed.")
    print("PASS")

    for path in [SOURCE_MANIFEST, P12_PATIENT_MANIFEST, P12_SLICE_MANIFEST]:
        if not path.is_file():
            raise FileNotFoundError(path)

    source_df = pd.read_csv(SOURCE_MANIFEST)
    p12_patient_df = pd.read_csv(P12_PATIENT_MANIFEST)
    p12_slice_df = pd.read_csv(P12_SLICE_MANIFEST)

    if len(source_df) != 139:
        raise RuntimeError(f"Expected 139 source cases, found {len(source_df)}.")
    if len(p12_patient_df) != 50:
        raise RuntimeError(
            f"Expected 50 PROMISE12 patients, found {len(p12_patient_df)}."
        )
    if len(p12_slice_df) != 1377:
        raise RuntimeError(
            f"Expected 1377 PROMISE12 image-only slices, found {len(p12_slice_df)}."
        )
    if bool(p12_slice_df["gt_used_for_manifest"].astype(bool).any()):
        raise RuntimeError("PROMISE12 slice manifest unexpectedly used GT.")

    nib = import_nibabel()

    source_spacing_df, target_spacing_x, target_spacing_y = (
        freeze_source_spacing(nib, source_df)
    )

    external_header_df = audit_external_image_headers_only(p12_patient_df)

    fold_df = build_source_patient_folds(source_df)

    protocol = build_frozen_protocol(
        target_spacing_x=target_spacing_x,
        target_spacing_y=target_spacing_y,
    )

    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)

    fold_path = args.output_dir / "CM2B_SOURCE_PATIENT_5FOLD_LOCK.csv"
    fold_df.to_csv(fold_path, index=False)

    source_spacing_path = args.output_dir / "CM2B_SOURCE_SPACING_AUDIT.csv"
    source_spacing_df.to_csv(source_spacing_path, index=False)

    external_header_path = (
        args.output_dir / "CM2B_PROMISE12_IMAGE_ONLY_HEADER_AUDIT.csv"
    )
    external_header_df.to_csv(external_header_path, index=False)

    protocol_path = args.output_dir / "CM2B_FROZEN_PROTOCOL.json"
    protocol_path.write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lock = {
        "status": "PASS",
        "decision": DECISION_PASS,
        "version": VERSION,
        "build": BUILD,
        "cm2a_lock_sha256": EXPECTED_CM2A_LOCK_SHA,
        "source_case_count": len(source_df),
        "external_patient_count": len(p12_patient_df),
        "external_image_only_slice_count": len(p12_slice_df),
        "source_target_spacing_mm": [target_spacing_x, target_spacing_y],
        "fold_seed": FOLD_SEED,
        "fold_count": N_FOLDS,
        "input_size": INPUT_SIZE,
        "dino_size": DINO_SIZE,
        "model_families": [
            "UNet",
            "DeepLabV3_R50",
            "SegFormer_B0",
        ],
        "information_boundary": {
            "model_training": False,
            "tta": False,
            "safety_model_fitting": False,
            "promises12_gt_header_read": False,
            "promises12_gt_raw_decode": False,
            "target_tuning": False,
            "representation_search": False,
        },
        "artifacts": {
            "fold_lock": str(fold_path),
            "fold_lock_sha256": sha256_file(fold_path),
            "source_spacing_audit": str(source_spacing_path),
            "source_spacing_audit_sha256": sha256_file(source_spacing_path),
            "external_image_header_audit": str(external_header_path),
            "external_image_header_audit_sha256": sha256_file(external_header_path),
            "frozen_protocol": str(protocol_path),
            "frozen_protocol_sha256": sha256_file(protocol_path),
        },
        "next_stage": (
            "CM3_SOURCE_OOF_SEGMENTATION_MODEL_TRAINING_AND_PREDICTION_LOCK"
        ),
    }

    lock_path = args.output_dir / "CM2B_PROTOCOL_LOCK.json"
    lock_path.write_text(
        json.dumps(lock, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n===== CM2B FINAL =====")
    print(
        "SOURCE-only target in-plane spacing mm:",
        [target_spacing_x, target_spacing_y],
    )
    print(
        "SOURCE patient folds:",
        fold_df["fold"].value_counts().sort_index().to_dict(),
    )
    print("Segmentation input:", f"{INPUT_SIZE}x{INPUT_SIZE}x3")
    print("DINO input:", f"{DINO_SIZE}x{DINO_SIZE}x3")
    print("Model panel: UNet / DeepLabV3-R50 / SegFormer-B0")
    print("PROMISE12 GT header read: NO")
    print("PROMISE12 GT RAW decoded: NO")
    print("Decision=", DECISION_PASS)
    print("LOCK=", lock_path)
    print("LOCK SHA256=", sha256_file(lock_path))
    print("PASS")


if __name__ == "__main__":
    main()
