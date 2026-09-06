#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R10L3B_polypgen_frozen_safety_score_lock_fix1.py

Prospective external safety-score lock AFTER R10L3A prediction lock and BEFORE
any PolypGen GT reveal.

Frozen method:
    M2_plus_CondDINO_PCA64

Per model-case:
1) RGB -> frozen facebook/dinov2-base, direct bicubic 224x224;
2) read SOURCE mask only (A1 value access forbidden);
3) SOURCE 352x352 -> exact 22x22 block occupancy -> 16x16;
4) foreground-weighted patch mean 768-D;
5) background-weighted patch mean 768-D;
6) preserve the R10K2A float16 feature-lock quantization, then cast to float32;
7) concatenate -> 1536-D;
8) frozen R10L0 PCA64.transform() ONLY;
9) append exact frozen M2:
       morph_fg_fraction
       morph_boundary_density
10) frozen R10L0 safety_head.predict_proba() ONLY;
11) frozen threshold = 0.300584763193734.

NO:
- PolypGen GT path read
- PolypGen GT pixel decode
- A1 mask value read
- PCA fit/refit
- safety-head fit/refit
- calibration
- threshold selection
- feature selection
- target-label access
- outcome metric computation

Outputs are cryptographically locked before GT reveal.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


VERSION = "2026-08-26-Q1-R10L3B-v1-fix1"
BUILD = "Q1_R10L3B_POLYPGEN_FROZEN_SAFETY_SCORE_LOCK_FIX1"

ROOT = Path(r"F:\MEDSEG_SAFETTA")

# ---------------------------------------------------------------------
# Exact upstream external prediction lock.
# ---------------------------------------------------------------------
R10L3A_DIR = (
    ROOT / "outputs"
    / "Q1_R10L3A_polypgen_source_a1_prediction_lock_fix2_v1"
)
R10L3A_LOCK = (
    R10L3A_DIR
    / "R10L3A_POLYPGEN_SOURCE_A1_PREDICTION_LOCK.json"
)
EXPECTED_R10L3A_LOCK_SHA256 = (
    "551a2bb372e1cd36cdbe982661b20f7050617a1398e1a773b88f027f75e3bb53"
)
EXPECTED_R10L3A_DECISION = (
    "POLYPGEN_SOURCE_A1_PREDICTIONS_LOCKED_BEFORE_GT_REVEAL"
)
R10L3A_RGB_MANIFEST = (
    R10L3A_DIR / "frozen_polypgen_rgb_manifest_no_gt.csv"
)
R10L3A_MODEL_CASE_MANIFEST = (
    R10L3A_DIR / "model_case_prediction_manifest.csv"
)

# ---------------------------------------------------------------------
# Frozen development method / estimator.
# ---------------------------------------------------------------------
R10K2A_SCRIPT = (
    ROOT / "code"
    / "Q1_R10K2A_dinov2_image_mask_representation_lock_fix1.py"
)
R10J1_SCRIPT = (
    ROOT / "code"
    / "Q1_R10J1_source_mask_morphology_feature_builder_fix1.py"
)
R10K2D_LOCK = (
    ROOT / "outputs"
    / "Q1_R10K2D_confirmed_method_freeze_fix1_v1"
    / "R10K2D_CONFIRMED_METHOD_LOCK.json"
)

R10L0_DIR = (
    ROOT / "outputs"
    / "Q1_R10L0_final_source_safety_estimator_lock_fix3_v1"
)
R10L0_LOCK = R10L0_DIR / "R10L0_FINAL_SOURCE_ESTIMATOR_LOCK.json"
R10L0_PCA = R10L0_DIR / "R10L0_final_condDINO_PCA64.joblib"
R10L0_HEAD = R10L0_DIR / "R10L0_final_safety_head.joblib"
R10L0_THRESHOLD = R10L0_DIR / "R10L0_frozen_operating_threshold.json"

EXPECTED_R10L0_DECISION = (
    "FINAL_SOURCE_ESTIMATOR_LOCKED_FOR_INDEPENDENT_EXTERNAL_VALIDATION"
)
EXPECTED_K2D_DECISION = (
    "R10K2_PRIMARY_METHOD_FROZEN_FOR_EXTERNAL_VALIDATION"
)
FROZEN_THRESHOLD = 0.300584763193734

# ---------------------------------------------------------------------
# Development representation sentinel.
# This ensures the currently loaded cached DINOv2/preprocessing reproduces
# the already-frozen R10K2A representation before scoring PolypGen.
# ---------------------------------------------------------------------
DEV_IMAGE_MANIFEST = (
    ROOT / "outputs"
    / "Q1_R05D1_neopolyp_acquisition_pairing_duplicate_qc_v1_fix2"
    / "frozen_confirmatory_manifest.csv"
)
DEV_IMAGE_CLS = (
    ROOT / "outputs"
    / "Q1_R10K2A_dinov2_image_mask_representation_lock_fix1_v1"
    / "R10K2A_image_cls_features.npz"
)
SENTINEL_CASES = 8
SENTINEL_FLOAT16_MAX_ABS_TOL = 0.0025

OUTPUT_DIR = (
    ROOT / "outputs"
    / "Q1_R10L3B_polypgen_frozen_safety_score_lock_fix1_v1"
)

DEFAULT_HF_CACHE = ROOT / "cache" / "huggingface"
DEFAULT_TORCH_CACHE = ROOT / "cache" / "torch"

EXPECTED_CASES = 1532
EXPECTED_STATES = 9
EXPECTED_MODEL_CASES = 13788
EXPECTED_COND_DIM = 1536
EXPECTED_PCA_DIM = 64
EXPECTED_FINAL_DIM = 66

MASK_H = 352
MASK_W = 352
PACKED_BYTES = 15488
BITORDER = "little"

DECISION_READY = (
    "POLYPGEN_FROZEN_SAFETY_SCORES_LOCKED_BEFORE_GT_REVEAL"
)


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_csv(path: Path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(fields))
        w.writeheader()
        w.writerows(rows)


def configure_project_local_caches(k2a):
    hf, tc = k2a.configure_project_local_caches(
        str(DEFAULT_HF_CACHE),
        str(DEFAULT_TORCH_CACHE),
    )
    return Path(hf), Path(tc)


def verify_required_paths():
    paths = {
        "R10L3A dir": R10L3A_DIR,
        "R10L3A lock": R10L3A_LOCK,
        "R10L3A RGB manifest": R10L3A_RGB_MANIFEST,
        "R10L3A model-case manifest": R10L3A_MODEL_CASE_MANIFEST,
        "R10K2A script": R10K2A_SCRIPT,
        "R10J1 script": R10J1_SCRIPT,
        "R10K2D lock": R10K2D_LOCK,
        "R10L0 lock": R10L0_LOCK,
        "R10L0 PCA": R10L0_PCA,
        "R10L0 head": R10L0_HEAD,
        "R10L0 threshold": R10L0_THRESHOLD,
        "development image manifest": DEV_IMAGE_MANIFEST,
        "development image CLS": DEV_IMAGE_CLS,
    }
    for label, p in paths.items():
        print(label, "exists:", p.exists(), p)
        if not p.exists():
            raise FileNotFoundError(p)
    return paths


def verify_upstream_locks():
    actual_l3a_sha = sha256_file(R10L3A_LOCK)
    if actual_l3a_sha.lower() != EXPECTED_R10L3A_LOCK_SHA256.lower():
        raise RuntimeError(
            "R10L3A lock SHA changed: "
            f"expected={EXPECTED_R10L3A_LOCK_SHA256} "
            f"actual={actual_l3a_sha}"
        )

    l3a = json.loads(R10L3A_LOCK.read_text(encoding="utf-8"))
    if l3a.get("decision") != EXPECTED_R10L3A_DECISION:
        raise RuntimeError(
            f"Unexpected R10L3A decision: {l3a.get('decision')}"
        )
    if int(l3a.get("target_cases", -1)) != EXPECTED_CASES:
        raise RuntimeError("R10L3A target case count changed.")
    if int(l3a.get("model_states", -1)) != EXPECTED_STATES:
        raise RuntimeError("R10L3A model-state count changed.")
    if int(l3a.get("model_case_rows", -1)) != EXPECTED_MODEL_CASES:
        raise RuntimeError("R10L3A model-case count changed.")
    if bool(l3a.get("target_gt_path_column_read", True)):
        raise RuntimeError("R10L3A reports target GT path read.")
    if bool(l3a.get("target_gt_pixels_decoded", True)):
        raise RuntimeError("R10L3A reports target GT pixels decoded.")
    if bool(l3a.get("target_outcomes_revealed", True)):
        raise RuntimeError("R10L3A reports target outcomes revealed.")
    if bool(l3a.get("safety_estimator_read", True)):
        raise RuntimeError("R10L3A reports premature safety estimator read.")

    k2d = json.loads(R10K2D_LOCK.read_text(encoding="utf-8"))
    if k2d.get("decision") != EXPECTED_K2D_DECISION:
        raise RuntimeError(
            f"Unexpected R10K2D decision: {k2d.get('decision')}"
        )

    l0 = json.loads(R10L0_LOCK.read_text(encoding="utf-8"))
    if l0.get("decision") != EXPECTED_R10L0_DECISION:
        raise RuntimeError(
            f"Unexpected R10L0 decision: {l0.get('decision')}"
        )

    lock_threshold = float(
        l0["operating_point"]["final_threshold"]
    )
    if abs(lock_threshold - FROZEN_THRESHOLD) > 1e-15:
        raise RuntimeError(
            f"R10L0 lock threshold changed: {lock_threshold}"
        )

    thr = json.loads(R10L0_THRESHOLD.read_text(encoding="utf-8"))
    artifact_threshold = float(thr["final_threshold"])
    if abs(artifact_threshold - FROZEN_THRESHOLD) > 1e-15:
        raise RuntimeError(
            f"Threshold artifact changed: {artifact_threshold}"
        )

    return {
        "r10l3a_lock_sha256": actual_l3a_sha,
        "r10k2a_script_sha256": sha256_file(R10K2A_SCRIPT),
        "r10j1_script_sha256": sha256_file(R10J1_SCRIPT),
        "r10k2d_lock_sha256": sha256_file(R10K2D_LOCK),
        "r10l0_lock_sha256": sha256_file(R10L0_LOCK),
        "r10l0_pca_sha256": sha256_file(R10L0_PCA),
        "r10l0_head_sha256": sha256_file(R10L0_HEAD),
        "r10l0_threshold_sha256": sha256_file(R10L0_THRESHOLD),
    }


def load_no_gt_manifests():
    rgb_required = [
        "sample_id",
        "original_polypgen_sample_id",
        "center",
        "image_path",
        "image_raw_sha256",
    ]
    rgb = pd.read_csv(
        R10L3A_RGB_MANIFEST,
        usecols=rgb_required,
        low_memory=False,
    )
    if len(rgb) != EXPECTED_CASES:
        raise RuntimeError(
            f"RGB manifest rows={len(rgb)}, expected={EXPECTED_CASES}"
        )
    if rgb["sample_id"].astype(str).nunique() != EXPECTED_CASES:
        raise RuntimeError("RGB sample_id not unique.")
    if set(rgb["center"].astype(str)) != {
        "C1", "C2", "C3", "C4", "C5", "C6"
    }:
        raise RuntimeError("All six centers not present.")

    state_required = [
        "sample_id",
        "center",
        "original_polypgen_sample_id",
        "image_path",
        "image_raw_sha256",
        "model_family",
        "model_state_id",
        "training_seed",
        "checkpoint_sha256",
        "state_prediction_npz",
        "state_row_index",
        "source_foreground_pixels",
        "a1_foreground_pixels",
        "changed_pixels",
    ]
    states = pd.read_csv(
        R10L3A_MODEL_CASE_MANIFEST,
        usecols=state_required,
        low_memory=False,
    )
    if len(states) != EXPECTED_MODEL_CASES:
        raise RuntimeError(
            f"Model-case rows={len(states)}, expected={EXPECTED_MODEL_CASES}"
        )
    if states["sample_id"].astype(str).nunique() != EXPECTED_CASES:
        raise RuntimeError("Model-case manifest case count != 1532.")
    if states["model_state_id"].astype(str).nunique() != EXPECTED_STATES:
        raise RuntimeError("Model-case manifest state count != 9.")
    if (
        states[["sample_id", "model_state_id"]]
        .drop_duplicates()
        .shape[0]
        != EXPECTED_MODEL_CASES
    ):
        raise RuntimeError("sample_id + model_state_id is not unique.")

    # Frozen deterministic representation row order, analogous to R10K2A.
    states = states.sort_values(
        ["model_family", "training_seed", "sample_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    states["_feature_row"] = np.arange(len(states), dtype=int)

    rgb = rgb.sort_values(
        "sample_id",
        kind="mergesort",
    ).reset_index(drop=True)

    if set(rgb["sample_id"].astype(str)) != set(
        states["sample_id"].astype(str)
    ):
        raise RuntimeError("RGB and state case sets differ.")

    counts = states.groupby("sample_id").size()
    if not (counts == 9).all():
        raise RuntimeError("Every external case must have exactly 9 states.")

    return rgb, states


def verify_rgb_hashes(rgb):
    for r in tqdm(
        rgb.itertuples(index=False),
        total=len(rgb),
        desc="Verifying locked PolypGen RGB",
        unit="image",
        dynamic_ncols=True,
    ):
        p = Path(str(r.image_path))
        if not p.exists():
            raise FileNotFoundError(p)
        actual = sha256_file(p).lower()
        expected = str(r.image_raw_sha256).lower()
        if actual != expected:
            raise RuntimeError(
                f"RGB SHA mismatch for {r.sample_id}"
            )


def source_npz_path(raw):
    p = Path(str(raw))
    if not p.is_absolute():
        p = R10L3A_DIR / p
    return p


def build_source_mask_features(states, k2a, j1):
    """
    Build:
      occupancy [13788,256]
      M2 [13788,2]

    Only source_masks_packed is accessed.
    a1_masks_packed is not indexed.
    """
    occupancy = np.zeros(
        (len(states), k2a.PATCH_GRID * k2a.PATCH_GRID),
        dtype=np.float32,
    )
    m2 = np.zeros((len(states), 2), dtype=np.float64)

    source_mask_reads = 0

    for state_id, g in states.groupby("model_state_id", sort=True):
        npz_values = g["state_prediction_npz"].astype(str).unique()
        if len(npz_values) != 1:
            raise RuntimeError(
                f"{state_id}: expected one prediction NPZ, got {npz_values}"
            )
        npz_path = source_npz_path(npz_values[0])
        if not npz_path.exists():
            raise FileNotFoundError(npz_path)

        with np.load(npz_path, allow_pickle=False) as z:
            if "source_masks_packed" not in z.files:
                raise RuntimeError(
                    f"{state_id}: source_masks_packed missing."
                )
            # Schema presence is allowed; value access is forbidden.
            if "a1_masks_packed" not in z.files:
                raise RuntimeError(
                    f"{state_id}: locked A1 schema key missing."
                )

            source = z["source_masks_packed"]
            if source.shape != (EXPECTED_CASES, PACKED_BYTES):
                raise RuntimeError(
                    f"{state_id}: SOURCE shape={source.shape}"
                )

            print(
                state_id,
                "SOURCE=",
                source.shape,
                "A1 value access=NO",
            )

            for r in tqdm(
                g.itertuples(index=True),
                total=len(g),
                desc=f"SOURCE features {state_id}",
                unit="mask",
                dynamic_ncols=True,
            ):
                out_i = int(r.Index)
                row_i = int(r.state_row_index)

                packed = source[row_i]
                source_mask_reads += 1

                occ = k2a.unpack_mask_occupancy16(packed)
                occupancy[out_i] = occ

                mask = j1.unpack_source_mask(packed)
                fg = float(mask.mean())
                bd = float(j1.boundary_density(mask))

                m2[out_i, 0] = fg
                m2[out_i, 1] = bd

    if source_mask_reads != EXPECTED_MODEL_CASES:
        raise RuntimeError(
            f"SOURCE mask reads={source_mask_reads}, "
            f"expected={EXPECTED_MODEL_CASES}"
        )
    if occupancy.shape != (EXPECTED_MODEL_CASES, 256):
        raise RuntimeError("Occupancy shape invalid.")
    if m2.shape != (EXPECTED_MODEL_CASES, 2):
        raise RuntimeError("M2 shape invalid.")
    if not np.isfinite(occupancy).all():
        raise RuntimeError("Non-finite occupancy.")
    if not np.isfinite(m2).all():
        raise RuntimeError("Non-finite M2.")

    return occupancy, m2


def load_frozen_dinov2(k2a, local_files_only=True, device_name="cuda"):
    hf_cache, torch_cache = configure_project_local_caches(k2a)

    try:
        import torch
        from transformers import AutoImageProcessor, AutoModel
    except Exception as e:
        raise RuntimeError(
            "torch + transformers are required; no fallback encoder allowed."
        ) from e

    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")

    device = torch.device(device_name)

    processor = AutoImageProcessor.from_pretrained(
        k2a.MODEL_NAME,
        cache_dir=str(hf_cache),
        local_files_only=local_files_only,
    )
    model = AutoModel.from_pretrained(
        k2a.MODEL_NAME,
        cache_dir=str(hf_cache),
        local_files_only=local_files_only,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.to(device)

    if int(getattr(model.config, "hidden_size", -1)) != 768:
        raise RuntimeError("Unexpected DINOv2 hidden size.")

    print("DINOv2:", k2a.MODEL_NAME)
    print("DINOv2 trainable:", False)
    print("Device:", device)
    if device.type == "cuda":
        print("CUDA:", torch.cuda.get_device_name(device))
    print("HF cache:", hf_cache)
    print("Torch cache:", torch_cache)
    print("Local files only:", local_files_only)

    return torch, processor, model, device


def run_dino_batch(paths, processor, model, device, k2a, torch):
    pil_images = []
    for p in paths:
        with Image.open(Path(str(p))) as im:
            rgb = im.convert("RGB")
            rgb = rgb.resize(
                (k2a.IMAGE_SIZE, k2a.IMAGE_SIZE),
                resample=Image.Resampling.BICUBIC,
            )
            pil_images.append(rgb.copy())

    inputs = processor(
        images=pil_images,
        return_tensors="pt",
        do_resize=False,
        do_center_crop=False,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    h = (
        outputs.last_hidden_state
        .detach()
        .float()
        .cpu()
        .numpy()
    )

    expected = (
        len(paths),
        1 + k2a.PATCH_GRID * k2a.PATCH_GRID,
        768,
    )
    if h.shape != expected:
        raise RuntimeError(
            f"DINO last_hidden_state={h.shape}, expected={expected}"
        )

    return h[:, 0, :], h[:, 1:, :]


def representation_sentinel(
    processor,
    model,
    device,
    k2a,
    torch,
):
    """
    Reproduce frozen development CLS values on 8 deterministic cases.
    Stored R10K2A CLS values are float16, so compare in the same quantized
    representation. This catches encoder/preprocessing/cache drift before any
    PolypGen safety score is generated.
    """
    dev = pd.read_csv(
        DEV_IMAGE_MANIFEST,
        usecols=["sample_id", "image_path"],
        low_memory=False,
    )
    if len(dev) != 1000:
        raise RuntimeError("Development image manifest rows != 1000.")

    dev["_sid"] = k2a.norm_sid_series(dev["sample_id"])
    dev = dev.sort_values("_sid").reset_index(drop=True)

    with np.load(DEV_IMAGE_CLS, allow_pickle=False) as z:
        cls_frozen = np.asarray(z["cls"])

    if cls_frozen.shape != (1000, 768):
        raise RuntimeError("Frozen development CLS shape invalid.")

    idx = np.linspace(
        0,
        len(dev) - 1,
        SENTINEL_CASES,
        dtype=int,
    )
    sub = dev.iloc[idx]

    cls_new, _ = run_dino_batch(
        sub["image_path"].tolist(),
        processor,
        model,
        device,
        k2a,
        torch,
    )

    # R10K2A persisted CLS as float16.
    cls_new_q = cls_new.astype(np.float16)
    ref_q = cls_frozen[idx].astype(np.float16)

    max_abs = float(
        np.max(
            np.abs(
                cls_new_q.astype(np.float32)
                - ref_q.astype(np.float32)
            )
        )
    )

    print("\n===== FROZEN DINOV2 REPRESENTATION SENTINEL =====")
    print("Development sentinel cases:", SENTINEL_CASES)
    print("Stored dtype:", cls_frozen.dtype)
    print("Comparison dtype: float16-locked -> float32")
    print("Max abs CLS difference:", max_abs)
    print("Tolerance:", SENTINEL_FLOAT16_MAX_ABS_TOL)

    if max_abs > SENTINEL_FLOAT16_MAX_ABS_TOL:
        raise RuntimeError(
            "Frozen DINOv2/preprocessing representation sentinel failed. "
            "Do not score PolypGen."
        )

    print("REPRESENTATION_SENTINEL_PASS")
    return max_abs


def build_conditioned_dino(
    rgb,
    states,
    occupancy,
    processor,
    model,
    device,
    k2a,
    torch,
    batch_size,
):
    """
    Extract one DINO patch-token tensor per RGB case, then pool with all 9
    source-mask occupancy vectors for that case.

    Critical frozen precision rule:
    R10K2A stored fg/bg feature matrices as float16, and R10K2B/R10L0 later
    loaded them and cast to float32. External scoring reproduces that exact
    quantization path before PCA.transform().
    """
    fg = np.zeros((EXPECTED_MODEL_CASES, 768), dtype=np.float32)
    bg = np.zeros((EXPECTED_MODEL_CASES, 768), dtype=np.float32)

    pos_by_sid = {
        str(sid): g.index.to_numpy(dtype=int)
        for sid, g in states.groupby("sample_id", sort=False)
    }
    if not all(len(v) == 9 for v in pos_by_sid.values()):
        raise RuntimeError("Every RGB case must map to 9 state rows.")

    for start in tqdm(
        range(0, len(rgb), batch_size),
        desc="PolypGen frozen DINOv2 batches",
        unit="batch",
        dynamic_ncols=True,
    ):
        stop = min(start + batch_size, len(rgb))
        b = rgb.iloc[start:stop]

        _, patch = run_dino_batch(
            b["image_path"].tolist(),
            processor,
            model,
            device,
            k2a,
            torch,
        )

        for j, r in enumerate(b.itertuples(index=False)):
            sid = str(r.sample_id)
            pos = pos_by_sid[sid]
            occ = occupancy[pos]
            f, bkg = k2a.pooled_region_features(
                patch[j],
                occ,
            )
            fg[pos] = f
            bg[pos] = bkg

    if not np.isfinite(fg).all() or not np.isfinite(bg).all():
        raise RuntimeError("Non-finite external conditioned DINO.")

    # Exact frozen feature-lock precision path.
    fg_locked = fg.astype(np.float16)
    bg_locked = bg.astype(np.float16)
    cond = np.concatenate(
        [
            fg_locked.astype(np.float32),
            bg_locked.astype(np.float32),
        ],
        axis=1,
    )

    if cond.shape != (EXPECTED_MODEL_CASES, EXPECTED_COND_DIM):
        raise RuntimeError(f"CondDINO shape={cond.shape}")
    if not np.isfinite(cond).all():
        raise RuntimeError("Non-finite quantized CondDINO.")

    return fg_locked, bg_locked, cond


def validate_frozen_estimators():
    pca = joblib.load(R10L0_PCA)
    head = joblib.load(R10L0_HEAD)

    if not hasattr(pca, "components_"):
        raise RuntimeError("Frozen PCA missing components_.")
    if np.asarray(pca.components_).shape != (
        EXPECTED_PCA_DIM,
        EXPECTED_COND_DIM,
    ):
        raise RuntimeError(
            f"Frozen PCA components={np.asarray(pca.components_).shape}"
        )
    if int(getattr(pca, "n_features_in_", -1)) != EXPECTED_COND_DIM:
        raise RuntimeError("Frozen PCA n_features_in_ != 1536.")

    if int(getattr(head, "n_features_in_", -1)) != EXPECTED_FINAL_DIM:
        raise RuntimeError(
            f"Frozen safety head n_features_in_="
            f"{getattr(head, 'n_features_in_', None)}"
        )

    required_steps = ["imputer", "scaler", "lr"]
    if list(head.named_steps.keys()) != required_steps:
        raise RuntimeError(
            f"Frozen head pipeline steps changed: {list(head.named_steps.keys())}"
        )

    return pca, head


def score_external(cond, m2, pca, head):
    # Strict deployment path: transform only. Never fit/fit_transform.
    z = pca.transform(cond)
    if z.shape != (EXPECTED_MODEL_CASES, EXPECTED_PCA_DIM):
        raise RuntimeError(f"External PCA output={z.shape}")
    if not np.isfinite(z).all():
        raise RuntimeError("Non-finite external PCA values.")

    X = np.column_stack(
        [
            np.asarray(m2, dtype=np.float64),
            np.asarray(z, dtype=np.float64),
        ]
    )
    if X.shape != (EXPECTED_MODEL_CASES, EXPECTED_FINAL_DIM):
        raise RuntimeError(f"External safety input={X.shape}")

    p = head.predict_proba(X)[:, 1]
    if p.shape != (EXPECTED_MODEL_CASES,):
        raise RuntimeError("External probability shape invalid.")
    if not np.isfinite(p).all():
        raise RuntimeError("Non-finite external safety probabilities.")
    if np.any((p < 0.0) | (p > 1.0)):
        raise RuntimeError("External probabilities outside [0,1].")

    decision = p >= FROZEN_THRESHOLD
    return z, X, p, decision


def run(args):
    print("===== R10L3B POLYPGEN FROZEN SAFETY SCORE LOCK FIX1 =====")
    print("STATUS: PROSPECTIVE PRE-GT SAFETY SCORE GENERATION")
    print("PRIMARY METHOD: M2_plus_CondDINO_PCA64")
    print("TARGET CASES:", EXPECTED_CASES)
    print("MODEL STATES:", EXPECTED_STATES)
    print("MODEL-CASE ROWS:", EXPECTED_MODEL_CASES)
    print("SOURCE MASK VALUES ACCESSED: YES")
    print("A1 MASK VALUES ACCESSED: NO")
    print("POLYPGEN GT PATH COLUMN READ: NO")
    print("POLYPGEN GT PIXELS DECODED: NO")
    print("PCA FIT/REFIT: NO")
    print("SAFETY HEAD FIT/REFIT: NO")
    print("TARGET CALIBRATION: NO")
    print("TARGET THRESHOLD TUNING: NO")
    print("FROZEN THRESHOLD:", FROZEN_THRESHOLD)
    print()

    verify_required_paths()
    upstream_hashes = verify_upstream_locks()

    # Import exact pre-outcome representation / M2 implementations.
    k2a = import_module(
        R10K2A_SCRIPT,
        "q1_r10l3b_authoritative_k2a",
    )
    j1 = import_module(
        R10J1_SCRIPT,
        "q1_r10l3b_authoritative_j1",
    )

    if int(k2a.MASK_SIZE) != MASK_H:
        raise RuntimeError("K2A mask size changed.")
    if int(k2a.PACKED_BYTES) != PACKED_BYTES:
        raise RuntimeError("K2A packed bytes changed.")
    if str(k2a.BITORDER) != BITORDER:
        raise RuntimeError("K2A bitorder changed.")
    if int(k2a.PATCH_GRID) != 16:
        raise RuntimeError("K2A patch grid changed.")
    if int(k2a.MASK_BLOCK) != 22:
        raise RuntimeError("K2A mask block changed.")

    if int(j1.MASK_H) != MASK_H or int(j1.MASK_W) != MASK_W:
        raise RuntimeError("J1 mask shape changed.")
    if str(j1.BITORDER) != BITORDER:
        raise RuntimeError("J1 bitorder changed.")

    rgb, states = load_no_gt_manifests()
    verify_rgb_hashes(rgb)

    print("\n===== SOURCE MASK REPRESENTATION =====")
    occupancy, m2 = build_source_mask_features(states, k2a, j1)

    print("Occupancy:", occupancy.shape)
    print("M2:", m2.shape)
    print(
        "Empty SOURCE masks:",
        int((occupancy.sum(axis=1) == 0).sum()),
    )
    print(
        "M2 fg fraction range:",
        float(m2[:, 0].min()),
        float(m2[:, 0].max()),
    )
    print(
        "M2 boundary density range:",
        float(m2[:, 1].min()),
        float(m2[:, 1].max()),
    )

    torch, processor, model, device = load_frozen_dinov2(
        k2a,
        local_files_only=True,
        device_name=args.device,
    )

    sentinel_diff = representation_sentinel(
        processor,
        model,
        device,
        k2a,
        torch,
    )

    print("\n===== POLYPGEN CONDITIONED DINOV2 =====")
    fg16, bg16, cond = build_conditioned_dino(
        rgb=rgb,
        states=states,
        occupancy=occupancy,
        processor=processor,
        model=model,
        device=device,
        k2a=k2a,
        torch=torch,
        batch_size=args.batch_size,
    )

    print("FG locked feature:", fg16.shape, fg16.dtype)
    print("BG locked feature:", bg16.shape, bg16.dtype)
    print("CondDINO deployment matrix:", cond.shape, cond.dtype)

    print("\n===== FROZEN R10L0 ESTIMATOR =====")
    pca, head = validate_frozen_estimators()
    z64, X66, prob, flagged = score_external(
        cond,
        m2,
        pca,
        head,
    )

    print("PCA transform output:", z64.shape)
    print("Safety input:", X66.shape)
    print("Probability finite:", bool(np.isfinite(prob).all()))
    print("Frozen-threshold flagged:", int(flagged.sum()), "/", len(flagged))

    # Scores are allowed to be summarized without labels, but no decisions may
    # be changed from these summaries.
    print("\n===== UNLABELED SCORE DISTRIBUTION AUDIT =====")
    print("min:", float(np.min(prob)))
    print("q25:", float(np.quantile(prob, 0.25)))
    print("median:", float(np.median(prob)))
    print("mean:", float(np.mean(prob)))
    print("q75:", float(np.quantile(prob, 0.75)))
    print("max:", float(np.max(prob)))

    if args.preflight_only:
        print(
            "\nPREFLIGHT NOTE: --preflight-only stops BEFORE writing "
            "external safety-score artifacts."
        )
        print("PREFLIGHT_PASS")
        return

    out = args.output_dir
    if out.exists():
        raise FileExistsError(
            f"Final R10L3B output already exists: {out}"
        )

    build = Path(str(out) + "__building")
    if build.exists():
        raise FileExistsError(
            f"Partial R10L3B build exists: {build}"
        )
    build.mkdir(parents=True, exist_ok=False)

    # Frozen representation artifacts.
    np.savez_compressed(
        build / "R10L3B_external_conditioned_dino_features.npz",
        foreground_patch_mean=fg16,
        background_patch_mean=bg16,
        source_mask_patch_occupancy=occupancy.astype(np.float16),
        m2=m2.astype(np.float64),
    )

    np.savez_compressed(
        build / "R10L3B_external_pca64_features.npz",
        pca64=z64.astype(np.float64),
    )

    score_rows = []
    for i, r in states.iterrows():
        score_rows.append({
            "feature_row": int(i),
            "sample_id": str(r["sample_id"]),
            "center": str(r["center"]),
            "original_polypgen_sample_id":
                str(r["original_polypgen_sample_id"]),
            "image_path": str(r["image_path"]),
            "image_raw_sha256": str(r["image_raw_sha256"]),
            "model_family": str(r["model_family"]),
            "model_state_id": str(r["model_state_id"]),
            "training_seed": int(r["training_seed"]),
            "checkpoint_sha256": str(r["checkpoint_sha256"]),
            "state_prediction_npz": str(r["state_prediction_npz"]),
            "state_row_index": int(r["state_row_index"]),
            "morph_fg_fraction": float(m2[i, 0]),
            "morph_boundary_density": float(m2[i, 1]),
            "frozen_safety_probability": float(prob[i]),
            "frozen_operating_threshold": float(FROZEN_THRESHOLD),
            "frozen_risk_flag": int(flagged[i]),
        })

    score_csv = build / "R10L3B_FROZEN_POLYPGEN_SAFETY_SCORES.csv"
    write_csv(
        score_csv,
        score_rows,
        list(score_rows[0].keys()),
    )

    # Descriptive summaries are fixed and unlabeled.
    score_df = pd.DataFrame(score_rows)
    family_summary = (
        score_df.groupby("model_family", sort=True)
        .agg(
            rows=("frozen_safety_probability", "size"),
            probability_mean=("frozen_safety_probability", "mean"),
            probability_median=("frozen_safety_probability", "median"),
            flagged=("frozen_risk_flag", "sum"),
        )
        .reset_index()
    )
    family_summary["flagged_fraction"] = (
        family_summary["flagged"] / family_summary["rows"]
    )
    family_summary.to_csv(
        build / "R10L3B_unlabeled_family_score_summary.csv",
        index=False,
    )

    center_summary = (
        score_df.groupby("center", sort=True)
        .agg(
            rows=("frozen_safety_probability", "size"),
            probability_mean=("frozen_safety_probability", "mean"),
            probability_median=("frozen_safety_probability", "median"),
            flagged=("frozen_risk_flag", "sum"),
        )
        .reset_index()
    )
    center_summary["flagged_fraction"] = (
        center_summary["flagged"] / center_summary["rows"]
    )
    center_summary.to_csv(
        build / "R10L3B_unlabeled_center_score_summary.csv",
        index=False,
    )

    info_boundary = {
        "polypgen_rgb_pixels_decoded": True,
        "source_mask_values_accessed": True,
        "a1_mask_values_accessed": False,
        "polypgen_gt_path_column_read": False,
        "polypgen_gt_pixels_decoded": False,
        "polypgen_outcomes_revealed": False,
        "external_dice_computed": False,
        "external_delta_dice_computed": False,
        "external_harm_label_computed": False,
        "pca_fit_called": False,
        "pca_refit_called": False,
        "safety_head_fit_called": False,
        "safety_head_refit_called": False,
        "target_calibration": False,
        "target_threshold_selection": False,
        "target_feature_selection": False,
        "target_score_orientation_change": False,
        "target_case_selection_from_scores": False,
        "frozen_threshold": FROZEN_THRESHOLD,
    }
    write_json(
        build / "information_boundary_audit.json",
        info_boundary,
    )

    artifacts = {}
    for p in [
        build / "R10L3B_external_conditioned_dino_features.npz",
        build / "R10L3B_external_pca64_features.npz",
        score_csv,
        build / "R10L3B_unlabeled_family_score_summary.csv",
        build / "R10L3B_unlabeled_center_score_summary.csv",
        build / "information_boundary_audit.json",
    ]:
        artifacts[p.name] = {
            "sha256": sha256_file(p),
            "bytes": p.stat().st_size,
        }

    lock = {
        "status": "FROZEN",
        "decision": DECISION_READY,
        "script_version": VERSION,
        "build": BUILD,
        "primary_method": "M2_plus_CondDINO_PCA64",
        "target_dataset": "PolypGen2021_MultiCenterData_v3",
        "target_component": "R10L1B_unique_static_C1_to_C6",
        "target_cases": EXPECTED_CASES,
        "model_states": EXPECTED_STATES,
        "model_case_rows": EXPECTED_MODEL_CASES,
        "representation": {
            "encoder": "facebook/dinov2-base",
            "encoder_trainable": False,
            "image_resize": "224x224 direct bicubic",
            "processor_resize": False,
            "processor_center_crop": False,
            "patch_grid": "16x16",
            "source_mask": "352x352 binary pre-adaptation SOURCE only",
            "source_mask_alignment":
                "exact 22x22 block occupancy -> 16x16",
            "foreground_patch_mean_dim": 768,
            "background_patch_mean_dim": 768,
            "feature_lock_precision":
                "float32 pooling -> float16 storage-equivalent -> float32 PCA input",
            "conditioned_dim": EXPECTED_COND_DIM,
            "m2": [
                "morph_fg_fraction",
                "morph_boundary_density",
            ],
            "pca_dim": EXPECTED_PCA_DIM,
            "final_input_dim": EXPECTED_FINAL_DIM,
        },
        "estimator": {
            "pca_fit_on_external": False,
            "safety_head_fit_on_external": False,
            "frozen_threshold": FROZEN_THRESHOLD,
            "probability_rows": EXPECTED_MODEL_CASES,
            "flagged_rows": int(flagged.sum()),
        },
        "representation_sentinel": {
            "development_cases_checked": SENTINEL_CASES,
            "float16_locked_cls_max_abs_difference": sentinel_diff,
            "tolerance": SENTINEL_FLOAT16_MAX_ABS_TOL,
            "pass": True,
        },
        "information_boundary": info_boundary,
        "upstream_hashes": upstream_hashes,
        "artifacts": artifacts,
        "post_score_rules": [
            "Do not refit PCA on PolypGen.",
            "Do not refit or recalibrate the safety head on PolypGen.",
            "Do not change score orientation.",
            "Do not change the frozen threshold.",
            "Do not select centers/cases/states based on score performance.",
            "Do not alter the representation after GT reveal.",
            "Next stage may reveal GT only after this lock is finalized.",
        ],
    }

    lock_path = build / "R10L3B_POLYPGEN_FROZEN_SAFETY_SCORE_LOCK.json"
    write_json(lock_path, lock)

    decision_path = build / "decision.txt"
    decision_path.write_text(DECISION_READY + "\n", encoding="utf-8")

    run_log = "\n".join([
        "===== R10L3B POLYPGEN FROZEN SAFETY SCORE LOCK FIX1 =====",
        f"Script version: {VERSION}",
        f"Build: {BUILD}",
        "",
        "Frozen target:",
        f"  cases={EXPECTED_CASES}",
        f"  states={EXPECTED_STATES}",
        f"  model-case rows={EXPECTED_MODEL_CASES}",
        "",
        "Frozen representation:",
        "  encoder=facebook/dinov2-base",
        "  trainable=NO",
        "  RGB=direct bicubic 224x224",
        "  patch grid=16x16",
        "  SOURCE mask=352x352 -> 22x22 occupancy -> 16x16",
        "  A1 values accessed=NO",
        "  FG mean=768",
        "  BG mean=768",
        "  CondDINO=1536",
        "  R10K2A float16 feature-lock precision reproduced=YES",
        "  M2=fg_fraction + boundary_density",
        "",
        "Frozen estimator:",
        "  PCA64 transform only=YES",
        "  PCA fit/refit=NO",
        "  safety-head predict_proba only=YES",
        "  safety-head fit/refit=NO",
        f"  frozen threshold={FROZEN_THRESHOLD:.15f}",
        "",
        "Pre-GT boundary:",
        "  PolypGen GT path column read=NO",
        "  PolypGen GT pixels decoded=NO",
        "  Dice/DeltaDice/HARM=NO",
        "  target calibration=NO",
        "  target threshold selection=NO",
        "  target feature selection=NO",
        "",
        "Safety-score lock:",
        f"  probability rows={EXPECTED_MODEL_CASES}",
        f"  frozen-threshold flagged rows={int(flagged.sum())}",
        f"  DINO development sentinel max abs={sentinel_diff:.9g}",
        "",
        "Decision:",
        f"  {DECISION_READY}",
        "",
        "Next:",
        "  R10L3C GT reveal + segmentation adaptation outcome construction",
        "  then evaluate the already-locked safety scores without recalibration.",
    ]) + "\n"
    (build / "run_log.txt").write_text(run_log, encoding="utf-8")

    # Add lock SHA only after lock is finalized.
    lock_sha = sha256_file(lock_path)

    build.rename(out)

    print()
    print((out / "run_log.txt").read_text(encoding="utf-8"))
    print(
        "R10L3B LOCK:",
        out / "R10L3B_POLYPGEN_FROZEN_SAFETY_SCORE_LOCK.json",
    )
    print("R10L3B LOCK SHA256:", lock_sha)
    print("PASS")


def self_test():
    # Mask occupancy / M2 mechanics independent of external data.
    m = np.zeros((MASK_H, MASK_W), dtype=np.uint8)
    m[50:100, 70:150] = 1
    packed = np.packbits(m.reshape(-1), bitorder=BITORDER)
    assert packed.shape == (PACKED_BYTES,)

    bits = np.unpackbits(
        packed,
        count=MASK_H * MASK_W,
        bitorder=BITORDER,
    ).reshape(MASK_H, MASK_W)
    assert np.array_equal(bits.astype(np.uint8), m)

    # Expected dimensions.
    assert EXPECTED_CASES * EXPECTED_STATES == EXPECTED_MODEL_CASES
    assert EXPECTED_COND_DIM == 768 * 2
    assert EXPECTED_FINAL_DIM == 2 + EXPECTED_PCA_DIM
    assert abs(FROZEN_THRESHOLD - 0.300584763193734) <= 1e-15

    print("PACKBITS_ROUNDTRIP_TEST_PASS")
    print("CARDINALITY_TEST_PASS")
    print("FROZEN_THRESHOLD_TEST_PASS")
    print("NO_GT_REQUIRED_FOR_SELF_TEST_PASS")
    print("SELF_TEST_PASS")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )
    ap.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda",
    )
    ap.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "Run all upstream checks, representation sentinel and external "
            "feature/score computation in memory, but write no score lock."
        ),
    )
    ap.add_argument("--self-test", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()

    if args.self_test:
        self_test()
        return 0

    if args.batch_size < 1:
        raise ValueError("--batch-size must be >=1")

    try:
        run(args)
    except Exception:
        traceback.print_exc()
        raise

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
