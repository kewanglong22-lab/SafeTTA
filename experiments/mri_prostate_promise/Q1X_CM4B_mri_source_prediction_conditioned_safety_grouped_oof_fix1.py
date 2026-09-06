#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1X_CM4B_mri_source_prediction_conditioned_safety_grouped_oof_fix1.py

SafeTTA Q1 enhancement — CM4B.

Purpose
-------
Confirm on the new prostate-MRI SOURCE task whether the frozen
prediction-conditioned semantic safety representation can rank TENT1 HARM
before adaptation.

This stage is SOURCE only:
- Prostate158 images + frozen CM3 SOURCE OOF masks;
- CM4A SOURCE TENT1 outcome labels;
- frozen DINOv2-base image encoder;
- patient-grouped 5-fold OOF safety evaluation.

No PROMISE12 access.
No target calibration.
No operating-point threshold selection.
No final all-source safety head fitting.

Primary representation
----------------------
Frozen cross-task framework:
  DINOv2-base patch tokens (224x224, 16x16 patches)
  + SOURCE-prediction conditioning:
      352x352 SOURCE mask -> 16x16 occupancy using exact 22x22 blocks
      foreground weighted token mean (768)
      background weighted token mean (768)
  + M2:
      SOURCE foreground fraction
      SOURCE boundary density

Fold-local safety head:
  train-only median imputer
  train-only randomized whiten PCA64 on CondDINO
  append M2
  train-only StandardScaler
  balanced LogisticRegression(max_iter=3000)

All folds are grouped by patient/case_key.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


VERSION = "2026-09-05-Q1X-CM4B-v1-fix1"
BUILD = "Q1X_CM4B_MRI_SOURCE_PREDICTION_CONDITIONED_SAFETY_GROUPED_OOF_FIX1"

ROOT = Path(r"F:\MEDSEG_SAFETTA")
OUT = ROOT / "outputs"

CM3_DIR = (
    OUT
    / "Q1X_CM3_source_oof_segmentation_training_and_prediction_lock_fix3_v1"
)
CM3_LOCK = CM3_DIR / "CM3_SOURCE_OOF_SEGMENTATION_PANEL_LOCK.json"
EXPECTED_CM3_LOCK_SHA = (
    "4cfbe8f108f53e014973111d96244c666bfabd13eb022675fa31a34edaae7cec"
)

CM4A_DIR = (
    OUT
    / "Q1X_CM4A_source_oof_tent1_outcome_asset_lock_fix4_v1"
)
CM4A_LOCK = CM4A_DIR / "CM4A_SOURCE_TENT1_OUTCOME_LOCK.json"
EXPECTED_CM4A_LOCK_SHA = (
    "7230c3eacf5e25b70e91ebdeb274fe4bc29b86c6e922f5350f69e5818569f4cb"
)

DEFAULT_OUTPUT = (
    OUT
    / "Q1X_CM4B_mri_source_prediction_conditioned_safety_grouped_oof_fix1_v1"
)

HF_CACHE = ROOT / "cache" / "huggingface"

FAMILIES = ["UNet", "DeepLabV3_R50", "SegFormer_B0"]
N_SLICES = 3553
N_PATIENTS = 139
N_FOLDS = 5

INPUT_SIZE = 352
DINO_SIZE = 224
DINO_PATCH = 14
DINO_GRID = 16
SOURCE_BLOCK = 22
DINO_DIM = 768
COND_DIM = 1536
M2_DIM = 2
FINAL_DIM = 1538

DINO_REPO = "facebook/dinov2-base"
DINO_WEIGHT_FILE = "model.safetensors"
DINO_BATCH_SIZE = 16

PCA_COMPONENTS = 64
LR_MAX_ITER = 3000
SEED = 20260904
BOOTSTRAP_SEED = 20260905
BOOTSTRAP_REPS = 2000

SOURCE_SIGNAL_AUROC_GATE = 0.65

IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)

PASS_DECISION = (
    "MRI_SOURCE_HARM_RANKING_SUPPORTED_READY_FOR_CM4C_FINAL_SOURCE_ESTIMATOR_LOCK"
)
STOP_DECISION = "STOP_MRI_SOURCE_HARM_RANKING_WEAK"


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


def save_json(path: Path, obj):
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def resolve_snapshot_commit(path: Path):
    parts = list(path.parts)
    if "snapshots" not in parts:
        raise RuntimeError(
            f"Unable to resolve Hugging Face snapshot commit from {path}"
        )
    i = parts.index("snapshots")
    if i + 1 >= len(parts):
        raise RuntimeError("Malformed Hugging Face snapshot path.")
    return parts[i + 1]


def lock_dino_snapshot(stage_dir: Path):
    from huggingface_hub import hf_hub_download

    lock_path = stage_dir / "CM4B_DINOV2_BASE_LOCK.json"

    if lock_path.is_file():
        lock = load_json(lock_path)
        commit = lock["commit"]

        weight_path = Path(
            hf_hub_download(
                repo_id=DINO_REPO,
                filename=DINO_WEIGHT_FILE,
                revision=commit,
                cache_dir=str(HF_CACHE),
            )
        )
        got = sha256_file(weight_path)

        if got != lock["weight_sha256"]:
            raise RuntimeError("Frozen DINOv2 weight SHA mismatch.")

        print(
            "DINOv2 lock=REUSE PASS "
            f"commit={commit[:12]} sha256={got}"
        )
        return lock, weight_path

    weight_path = Path(
        hf_hub_download(
            repo_id=DINO_REPO,
            filename=DINO_WEIGHT_FILE,
            revision="main",
            cache_dir=str(HF_CACHE),
        )
    )
    commit = resolve_snapshot_commit(weight_path)
    weight_sha = sha256_file(weight_path)

    config_path = Path(
        hf_hub_download(
            repo_id=DINO_REPO,
            filename="config.json",
            revision=commit,
            cache_dir=str(HF_CACHE),
        )
    )
    config_sha = sha256_file(config_path)

    lock = {
        "status": "PASS",
        "repo": DINO_REPO,
        "commit": commit,
        "weight_file": DINO_WEIGHT_FILE,
        "weight_path": str(weight_path),
        "weight_sha256": weight_sha,
        "config_path": str(config_path),
        "config_sha256": config_sha,
        "input": {
            "grayscale_repeat_rgb": True,
            "resize": "bicubic",
            "size": [DINO_SIZE, DINO_SIZE],
            "imagenet_mean": IMAGENET_MEAN.tolist(),
            "imagenet_std": IMAGENET_STD.tolist(),
        },
        "promises12_access": False,
    }
    save_json(lock_path, lock)

    print(
        "DINOv2 lock=NEW PASS "
        f"commit={commit[:12]} sha256={weight_sha}"
    )

    return lock, weight_path


def load_dino_model(lock, device):
    import torch
    from transformers import Dinov2Model

    model = Dinov2Model.from_pretrained(
        DINO_REPO,
        revision=lock["commit"],
        use_safetensors=True,
        cache_dir=str(HF_CACHE),
    )
    model.to(device)
    model.eval()
    model.requires_grad_(False)

    hidden = int(model.config.hidden_size)
    patch = int(model.config.patch_size)

    if hidden != DINO_DIM:
        raise RuntimeError(f"DINO hidden size changed: {hidden}")
    if patch != DINO_PATCH:
        raise RuntimeError(f"DINO patch size changed: {patch}")

    return model


def dino_preprocess(image_batch, torch, F, device):
    x = torch.from_numpy(
        np.asarray(image_batch, dtype=np.float32)
    ).unsqueeze(1)

    x = F.interpolate(
        x,
        size=(DINO_SIZE, DINO_SIZE),
        mode="bicubic",
        align_corners=False,
    )
    x = x.repeat(1, 3, 1, 1)

    mean = torch.tensor(
        IMAGENET_MEAN,
        dtype=x.dtype,
    ).view(1, 3, 1, 1)
    std = torch.tensor(
        IMAGENET_STD,
        dtype=x.dtype,
    ).view(1, 3, 1, 1)

    x = (x - mean) / std
    return x.to(device, non_blocking=True)


def source_mask_occupancy(mask_batch, torch, device):
    m = torch.from_numpy(
        np.asarray(mask_batch, dtype=np.float32)
    ).to(device, non_blocking=True)

    if tuple(m.shape[1:]) != (INPUT_SIZE, INPUT_SIZE):
        raise RuntimeError(f"Unexpected mask shape {tuple(m.shape)}")

    # 352 = 16 * 22 exactly.
    occ = (
        m.reshape(
            m.shape[0],
            DINO_GRID,
            SOURCE_BLOCK,
            DINO_GRID,
            SOURCE_BLOCK,
        )
        .mean(dim=(2, 4))
        .reshape(m.shape[0], DINO_GRID * DINO_GRID)
    )
    return occ


def pool_conditioned(tokens, occupancy, torch):
    if tokens.ndim != 3:
        raise RuntimeError(f"Unexpected DINO tokens shape {tokens.shape}")
    if tokens.shape[1:] != (DINO_GRID * DINO_GRID, DINO_DIM):
        raise RuntimeError(
            f"Unexpected DINO patch tensor shape {tokens.shape}"
        )

    fg_w = occupancy
    bg_w = 1.0 - occupancy

    fg_num = torch.sum(tokens * fg_w.unsqueeze(-1), dim=1)
    bg_num = torch.sum(tokens * bg_w.unsqueeze(-1), dim=1)

    fg_den = torch.sum(fg_w, dim=1, keepdim=True)
    bg_den = torch.sum(bg_w, dim=1, keepdim=True)

    fg = fg_num / torch.clamp(fg_den, min=1e-6)
    bg = bg_num / torch.clamp(bg_den, min=1e-6)

    # Explicit zero vector when SOURCE prediction is empty/full.
    fg = torch.where(fg_den > 0, fg, torch.zeros_like(fg))
    bg = torch.where(bg_den > 0, bg, torch.zeros_like(bg))

    return fg, bg


def m2_from_masks(mask_batch):
    m = np.asarray(mask_batch, dtype=bool)

    fg_fraction = m.mean(axis=(1, 2), dtype=np.float64)

    h = np.mean(m[:, :, 1:] != m[:, :, :-1], axis=(1, 2))
    v = np.mean(m[:, 1:, :] != m[:, :-1, :], axis=(1, 2))
    boundary = 0.5 * (h + v)

    return np.stack(
        [fg_fraction, boundary],
        axis=1,
    ).astype(np.float32)


def verify_lineage():
    if not CM3_LOCK.is_file():
        raise FileNotFoundError(CM3_LOCK)
    if not CM4A_LOCK.is_file():
        raise FileNotFoundError(CM4A_LOCK)

    cm3_sha = sha256_file(CM3_LOCK)
    cm4a_sha = sha256_file(CM4A_LOCK)

    print(
        "CM3_LOCK",
        cm3_sha,
        "PASS" if cm3_sha == EXPECTED_CM3_LOCK_SHA else "FAIL",
    )
    print(
        "CM4A_LOCK",
        cm4a_sha,
        "PASS" if cm4a_sha == EXPECTED_CM4A_LOCK_SHA else "FAIL",
    )

    if cm3_sha != EXPECTED_CM3_LOCK_SHA:
        raise RuntimeError("CM3 lock SHA mismatch.")
    if cm4a_sha != EXPECTED_CM4A_LOCK_SHA:
        raise RuntimeError("CM4A lock SHA mismatch.")

    cm3 = load_json(CM3_LOCK)
    cm4a = load_json(CM4A_LOCK)

    if cm3.get("status") != "PASS":
        raise RuntimeError("CM3 status changed.")
    if cm4a.get("status") != "PASS":
        raise RuntimeError("CM4A status changed.")

    if cm4a.get("decision") != (
        "SOURCE_TENT1_OUTCOME_ASSET_LOCKED_READY_FOR_CM4B_SAFETY_REPRESENTATION"
    ):
        raise RuntimeError("Unexpected CM4A decision.")

    if cm4a.get("information_boundary", {}).get("promises12_access") is not False:
        raise RuntimeError("CM4A information boundary changed.")

    return cm3, cm4a


def load_source_assets(cm3, cm4a):
    cache_meta_path = Path(cm3["source_cache_meta"])
    if sha256_file(cache_meta_path) != cm3["source_cache_meta_sha256"]:
        raise RuntimeError("SOURCE cache-meta SHA mismatch.")

    cache_meta = load_json(cache_meta_path)
    cache_dir = cache_meta_path.parent

    image_path = cache_dir / "source_images_f16.npy"
    slices_path = cache_dir / "source_slices.csv"

    if sha256_file(image_path) != cache_meta["images_sha256"]:
        raise RuntimeError("SOURCE image-cache SHA mismatch.")
    if sha256_file(slices_path) != cache_meta["slices_sha256"]:
        raise RuntimeError("SOURCE slice-manifest SHA mismatch.")

    images = np.load(image_path, mmap_mode="r")
    slices = pd.read_csv(slices_path)

    if images.shape != (N_SLICES, INPUT_SIZE, INPUT_SIZE):
        raise RuntimeError(f"Unexpected image cache shape {images.shape}")
    if len(slices) != N_SLICES:
        raise RuntimeError(f"Unexpected slice rows {len(slices)}")
    if slices["case_key"].nunique() != N_PATIENTS:
        raise RuntimeError("Expected 139 SOURCE patients.")

    outcome_path = Path(cm4a["artifacts"]["outcome_table"])
    if sha256_file(outcome_path) != cm4a["artifacts"]["outcome_table_sha256"]:
        raise RuntimeError("CM4A outcome-table SHA mismatch.")

    outcomes = pd.read_csv(outcome_path)

    if len(outcomes) != N_SLICES * len(FAMILIES):
        raise RuntimeError(
            f"Unexpected CM4A outcome rows {len(outcomes)}"
        )

    outcomes["family"] = outcomes["family"].astype(str)

    got_families = sorted(outcomes["family"].unique().tolist())
    if got_families != sorted(FAMILIES):
        raise RuntimeError(f"Unexpected family set {got_families}")

    # Exact one-to-one family/global_index coverage.
    for family in FAMILIES:
        sub = outcomes[outcomes["family"] == family]
        if len(sub) != N_SLICES:
            raise RuntimeError(f"{family}: expected {N_SLICES} rows.")
        if sub["global_index"].nunique() != N_SLICES:
            raise RuntimeError(
                f"{family}: global_index coverage is not one-to-one."
            )

    # Keep exact frozen outcome ordering for the safety table.
    outcomes = (
        outcomes
        .sort_values(["family", "global_index"])
        .reset_index(drop=True)
    )

    return images, slices, outcomes


def load_family_masks(cm3):
    masks = {}

    for family in FAMILIES:
        info = cm3["family_oof_artifacts"][family]
        path = Path(info["oof_mask"])

        if sha256_file(path) != info["oof_mask_sha256"]:
            raise RuntimeError(f"{family}: OOF-mask SHA mismatch.")

        arr = np.load(path, mmap_mode="r")
        if arr.shape != (N_SLICES, INPUT_SIZE, INPUT_SIZE):
            raise RuntimeError(
                f"{family}: unexpected OOF-mask shape {arr.shape}"
            )
        masks[family] = arr

    return masks


def print_class_audit(outcomes):
    print("\n===== SOURCE OUTCOME CLASS AUDIT =====")

    rows = []
    for family in FAMILIES:
        sub = outcomes[outcomes["family"] == family]
        harm_patients = int(
            sub.loc[sub["harmful"] == 1, "case_key"].nunique()
        )
        rows.append({
            "family": family,
            "rows": len(sub),
            "harm": int(sub["harmful"].sum()),
            "harm_prevalence": float(sub["harmful"].mean()),
            "harm_patients": harm_patients,
            "benefit": int(sub["beneficial"].sum()),
        })

    audit = pd.DataFrame(rows)
    print(audit.to_string(index=False))

    fold_table = (
        outcomes
        .groupby(["family", "fold"], as_index=False)
        .agg(
            rows=("harmful", "size"),
            harm=("harmful", "sum"),
            patients=("case_key", "nunique"),
        )
    )
    print("\nFamily/fold HARM counts:")
    print(fold_table.to_string(index=False))

    # Hard feasibility only: every held-out fold must contain both classes.
    for row in fold_table.itertuples(index=False):
        if int(row.harm) <= 0 or int(row.harm) >= int(row.rows):
            raise RuntimeError(
                f"{row.family} fold{row.fold}: "
                "held-out safety fold lacks both classes."
            )

    return audit, fold_table


def preflight_dino(
    model,
    images,
    masks,
    torch,
    F,
    device,
):
    print("\n===== DINO CONDITIONING PREFLIGHT =====")

    gi = np.arange(min(4, N_SLICES), dtype=np.int64)
    x = dino_preprocess(images[gi], torch, F, device)

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
            out = model(pixel_values=x)

    hidden = out.last_hidden_state
    if hidden.shape[1] != 1 + DINO_GRID * DINO_GRID:
        raise RuntimeError(
            f"Unexpected DINO token count {hidden.shape}"
        )

    tokens = hidden[:, 1:, :].float()

    for family in FAMILIES:
        mb = np.asarray(masks[family][gi], dtype=np.uint8)
        occ = source_mask_occupancy(mb, torch, device)
        fg, bg = pool_conditioned(tokens, occ, torch)
        m2 = m2_from_masks(mb)

        if fg.shape != (len(gi), DINO_DIM):
            raise RuntimeError(f"{family}: bad FG shape {fg.shape}")
        if bg.shape != (len(gi), DINO_DIM):
            raise RuntimeError(f"{family}: bad BG shape {bg.shape}")
        if m2.shape != (len(gi), 2):
            raise RuntimeError(f"{family}: bad M2 shape {m2.shape}")

        finite = (
            torch.isfinite(fg).all().item()
            and torch.isfinite(bg).all().item()
            and np.isfinite(m2).all()
        )
        if not finite:
            raise RuntimeError(f"{family}: nonfinite preflight features.")

        print(
            f"{family}: PASS "
            f"tokens={tuple(tokens.shape)} "
            f"FG={tuple(fg.shape)} "
            f"BG={tuple(bg.shape)} "
            f"M2={tuple(m2.shape)} "
            f"fg_fraction_range="
            f"[{m2[:,0].min():.6f},{m2[:,0].max():.6f}]"
        )

    print("DINO conditioning preflight: PASS")


def extract_features(
    model,
    images,
    masks,
    stage_dir,
    torch,
    F,
    device,
):
    feature_path = stage_dir / "CM4B_CONDITIONED_DINO_M2_FEATURES_F16.npy"
    meta_path = stage_dir / "CM4B_FEATURE_EXTRACTION_META.json"

    expected_shape = (len(FAMILIES), N_SLICES, FINAL_DIM)

    if feature_path.is_file() and meta_path.is_file():
        meta = load_json(meta_path)
        arr = np.load(feature_path, mmap_mode="r")

        valid = (
            meta.get("status") == "PASS"
            and tuple(meta.get("shape", [])) == expected_shape
            and meta.get("feature_sha256") == sha256_file(feature_path)
            and arr.shape == expected_shape
        )
        if valid:
            print("\nFEATURE EXTRACTION=REUSE PASS")
            return arr, meta

    print("\n===== EXTRACT FROZEN CONDITIONED DINO FEATURES =====")
    print("DINO batch size:", DINO_BATCH_SIZE)
    print("Expected shape:", expected_shape)

    features = np.lib.format.open_memmap(
        feature_path,
        mode="w+",
        dtype=np.float16,
        shape=expected_shape,
    )

    starts = range(0, N_SLICES, DINO_BATCH_SIZE)

    for start in tqdm(
        starts,
        total=(N_SLICES + DINO_BATCH_SIZE - 1) // DINO_BATCH_SIZE,
        desc="DINOv2 SOURCE features",
        unit="batch",
        dynamic_ncols=True,
    ):
        end = min(start + DINO_BATCH_SIZE, N_SLICES)
        gi = np.arange(start, end, dtype=np.int64)

        x = dino_preprocess(images[gi], torch, F, device)

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
                out = model(pixel_values=x)

        hidden = out.last_hidden_state
        tokens = hidden[:, 1:, :].float()

        if tokens.shape[1:] != (
            DINO_GRID * DINO_GRID,
            DINO_DIM,
        ):
            raise RuntimeError(
                f"Unexpected DINO patch tensor shape {tokens.shape}"
            )

        for fi, family in enumerate(FAMILIES):
            mb = np.asarray(
                masks[family][gi],
                dtype=np.uint8,
            )

            occ = source_mask_occupancy(mb, torch, device)
            fg, bg = pool_conditioned(tokens, occ, torch)
            m2 = m2_from_masks(mb)

            cond = torch.cat([fg, bg], dim=1).cpu().numpy()
            feat = np.concatenate(
                [cond.astype(np.float32), m2],
                axis=1,
            )

            if feat.shape != (len(gi), FINAL_DIM):
                raise RuntimeError(
                    f"{family}: bad feature shape {feat.shape}"
                )
            if not np.isfinite(feat).all():
                raise RuntimeError(
                    f"{family}: nonfinite feature values."
                )

            features[fi, start:end, :] = feat.astype(np.float16)

    features.flush()

    meta = {
        "status": "PASS",
        "version": VERSION,
        "shape": list(expected_shape),
        "dtype": "float16",
        "family_order": FAMILIES,
        "feature_layout": {
            "0:768": "SOURCE-mask foreground weighted DINO patch mean",
            "768:1536": "SOURCE-mask background weighted DINO patch mean",
            "1536": "SOURCE foreground fraction",
            "1537": "SOURCE boundary density",
        },
        "source_mask_to_dino": {
            "source_size": 352,
            "grid": [16, 16],
            "block_size": [22, 22],
            "occupancy_weighted": True,
        },
        "feature_file": str(feature_path),
        "feature_sha256": sha256_file(feature_path),
        "promises12_access": False,
    }
    save_json(meta_path, meta)

    arr = np.load(feature_path, mmap_mode="r")
    return arr, meta


def make_feature_row_table(features, outcomes):
    # Feature memmap is [family_index, global_index, dim].
    family_to_idx = {
        family: i for i, family in enumerate(FAMILIES)
    }

    X = np.empty((len(outcomes), FINAL_DIM), dtype=np.float32)

    for family in FAMILIES:
        mask = outcomes["family"].to_numpy() == family
        rows = np.flatnonzero(mask)
        gi = outcomes.loc[mask, "global_index"].to_numpy(
            dtype=np.int64
        )
        X[rows] = np.asarray(
            features[family_to_idx[family], gi, :],
            dtype=np.float32,
        )

    if not np.isfinite(X).all():
        raise RuntimeError("Nonfinite aligned safety features.")

    return X


def fit_predict_fold(
    X_train,
    y_train,
    X_val,
    representation,
):
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    if representation == "M2":
        imp = SimpleImputer(strategy="median")
        tr = imp.fit_transform(X_train[:, COND_DIM:FINAL_DIM])
        va = imp.transform(X_val[:, COND_DIM:FINAL_DIM])

        scaler = StandardScaler()
        tr = scaler.fit_transform(tr)
        va = scaler.transform(va)

        clf = LogisticRegression(
            class_weight="balanced",
            max_iter=LR_MAX_ITER,
            random_state=SEED,
        )
        clf.fit(tr, y_train)
        return clf.predict_proba(va)[:, 1]

    cond_imp = SimpleImputer(strategy="median")
    tr_cond = cond_imp.fit_transform(X_train[:, :COND_DIM])
    va_cond = cond_imp.transform(X_val[:, :COND_DIM])

    pca = PCA(
        n_components=PCA_COMPONENTS,
        whiten=True,
        svd_solver="randomized",
        random_state=SEED,
    )
    tr_pca = pca.fit_transform(tr_cond)
    va_pca = pca.transform(va_cond)

    if representation == "CondDINO":
        tr = tr_pca
        va = va_pca

    elif representation == "Final":
        m2_imp = SimpleImputer(strategy="median")
        tr_m2 = m2_imp.fit_transform(
            X_train[:, COND_DIM:FINAL_DIM]
        )
        va_m2 = m2_imp.transform(
            X_val[:, COND_DIM:FINAL_DIM]
        )

        tr = np.concatenate([tr_pca, tr_m2], axis=1)
        va = np.concatenate([va_pca, va_m2], axis=1)

    else:
        raise ValueError(representation)

    scaler = StandardScaler()
    tr = scaler.fit_transform(tr)
    va = scaler.transform(va)

    clf = LogisticRegression(
        class_weight="balanced",
        max_iter=LR_MAX_ITER,
        random_state=SEED,
    )
    clf.fit(tr, y_train)

    return clf.predict_proba(va)[:, 1]


def grouped_oof_scores(X, outcomes):
    print("\n===== PATIENT-GROUPED SOURCE SAFETY OOF =====")

    representations = ["M2", "CondDINO", "Final"]
    scores = {
        name: np.full(len(outcomes), np.nan, dtype=np.float64)
        for name in representations
    }

    y = outcomes["harmful"].to_numpy(dtype=np.int64)
    folds = outcomes["fold"].to_numpy(dtype=np.int64)

    for fold in range(N_FOLDS):
        train = folds != fold
        val = folds == fold

        train_patients = set(outcomes.loc[train, "case_key"])
        val_patients = set(outcomes.loc[val, "case_key"])

        overlap = train_patients.intersection(val_patients)
        if overlap:
            raise RuntimeError(
                f"Patient leakage in safety fold {fold}: "
                f"{sorted(overlap)[:5]}"
            )

        y_tr = y[train]
        y_va = y[val]

        if len(np.unique(y_tr)) != 2 or len(np.unique(y_va)) != 2:
            raise RuntimeError(
                f"Safety fold{fold}: both classes required."
            )

        print(
            f"fold{fold}: "
            f"train_rows={train.sum()} train_patients={len(train_patients)} "
            f"train_HARM={int(y_tr.sum())} | "
            f"val_rows={val.sum()} val_patients={len(val_patients)} "
            f"val_HARM={int(y_va.sum())}"
        )

        for representation in representations:
            pred = fit_predict_fold(
                X[train],
                y_tr,
                X[val],
                representation,
            )
            scores[representation][val] = pred

    for name, arr in scores.items():
        if not np.isfinite(arr).all():
            raise RuntimeError(f"{name}: incomplete OOF scores.")

    return scores


def metric_table(outcomes, scores):
    from scipy.stats import spearmanr
    from sklearn.metrics import (
        average_precision_score,
        roc_auc_score,
    )

    y = outcomes["harmful"].to_numpy(dtype=np.int64)
    neg_delta = -outcomes["delta_dice"].to_numpy(dtype=np.float64)

    rows = []

    for representation, score in scores.items():
        pooled_auc = float(roc_auc_score(y, score))
        pooled_ap = float(average_precision_score(y, score))
        rho = float(spearmanr(score, neg_delta).statistic)

        family_aucs = []
        family_aps = []

        for family in FAMILIES:
            mask = outcomes["family"].to_numpy() == family
            yf = y[mask]
            sf = score[mask]

            auc = float(roc_auc_score(yf, sf))
            ap = float(average_precision_score(yf, sf))
            family_aucs.append(auc)
            family_aps.append(ap)

            rows.append({
                "representation": representation,
                "scope": family,
                "auroc": auc,
                "auprc": ap,
                "spearman_score_vs_minus_delta": float(
                    spearmanr(
                        sf,
                        neg_delta[mask],
                    ).statistic
                ),
            })

        rows.append({
            "representation": representation,
            "scope": "MACRO_FAMILY",
            "auroc": float(np.mean(family_aucs)),
            "auprc": float(np.mean(family_aps)),
            "spearman_score_vs_minus_delta": np.nan,
        })
        rows.append({
            "representation": representation,
            "scope": "POOLED",
            "auroc": pooled_auc,
            "auprc": pooled_ap,
            "spearman_score_vs_minus_delta": rho,
        })

    return pd.DataFrame(rows)


def patient_cluster_bootstrap(outcomes, score):
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    patients = np.asarray(
        sorted(outcomes["case_key"].astype(str).unique())
    )
    if len(patients) != N_PATIENTS:
        raise RuntimeError("Unexpected bootstrap patient count.")

    case = outcomes["case_key"].astype(str).to_numpy()
    family = outcomes["family"].astype(str).to_numpy()
    y = outcomes["harmful"].to_numpy(dtype=np.int64)

    patient_to_rows = {
        p: np.flatnonzero(case == p)
        for p in patients
    }

    vals = []

    for _ in tqdm(
        range(BOOTSTRAP_REPS),
        desc="patient bootstrap",
        unit="rep",
        dynamic_ncols=True,
    ):
        sampled = rng.choice(
            patients,
            size=len(patients),
            replace=True,
        )
        idx = np.concatenate(
            [patient_to_rows[p] for p in sampled]
        )

        family_aucs = []
        valid = True

        for fam in FAMILIES:
            m = family[idx] == fam
            yy = y[idx][m]
            ss = score[idx][m]

            if len(np.unique(yy)) != 2:
                valid = False
                break

            family_aucs.append(
                float(roc_auc_score(yy, ss))
            )

        if valid:
            vals.append(float(np.mean(family_aucs)))

    vals = np.asarray(vals, dtype=np.float64)

    if len(vals) < int(0.95 * BOOTSTRAP_REPS):
        raise RuntimeError(
            f"Too many invalid patient bootstrap replicates: "
            f"{len(vals)}/{BOOTSTRAP_REPS}"
        )

    return {
        "reps_requested": BOOTSTRAP_REPS,
        "reps_valid": int(len(vals)),
        "mean": float(vals.mean()),
        "ci95_low": float(np.quantile(vals, 0.025)),
        "ci95_high": float(np.quantile(vals, 0.975)),
    }


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
            "Run exact lineage/class audits and a small real DINOv2 "
            "conditioning forward pass, then stop."
        ),
    )
    args = ap.parse_args()

    print(
        "===== Q1X CM4B MRI SOURCE PREDICTION-CONDITIONED SAFETY OOF ====="
    )
    print("SOURCE_ONLY=YES")
    print("PROMISE12_ACCESS=NO")
    print("TARGET_CALIBRATION=NO")
    print("THRESHOLD_SELECTION=NO")
    print("FINAL_SOURCE_HEAD_FITTING=NO")
    print("SAMPLE_UNIT=axial_2D_slice")
    print("GROUPING_UNIT=patient/case_key")
    print("DINO=", DINO_REPO)
    print("DINO_INPUT=224x224 grayscale repeated RGB")
    print("DINO_PATCH_GRID=16x16")
    print("SOURCE_MASK_GRID=16x16 occupancy, 22x22 source blocks")
    print("PRIMARY_REP=CondDINO_FG768_BG768_PCA64_plus_M2")
    print("SAFETY_HEAD=balanced LogisticRegression max_iter=3000")
    print("SOURCE_SIGNAL_AUROC_GATE=", SOURCE_SIGNAL_AUROC_GATE)

    cm3, cm4a = verify_lineage()
    images, slices, outcomes = load_source_assets(cm3, cm4a)
    masks = load_family_masks(cm3)

    class_audit, fold_audit = print_class_audit(outcomes)

    if args.output_dir.exists():
        stage_dir = args.output_dir
        print("\nSTAGE_DIR=RESUME", stage_dir)
    else:
        stage_dir = args.output_dir
        stage_dir.mkdir(parents=True, exist_ok=False)
        print("\nSTAGE_DIR=NEW", stage_dir)

    dino_lock, weight_path = lock_dino_snapshot(stage_dir)

    import torch
    import torch.nn.functional as F

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n===== RUNTIME =====")
    print("torch:", torch.__version__)
    print("device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(device))

    set_seed(SEED)

    model = load_dino_model(dino_lock, device)
    preflight_dino(
        model,
        images,
        masks,
        torch,
        F,
        device,
    )

    preflight_path = stage_dir / "CM4B_PREFLIGHT.json"
    save_json(
        preflight_path,
        {
            "status": "PASS",
            "version": VERSION,
            "cm3_lock_sha256": EXPECTED_CM3_LOCK_SHA,
            "cm4a_lock_sha256": EXPECTED_CM4A_LOCK_SHA,
            "dino_lock": dino_lock,
            "dino_weight_sha256": sha256_file(weight_path),
            "source_rows": int(len(outcomes)),
            "source_patients": int(outcomes["case_key"].nunique()),
            "harm_counts": {
                family: int(
                    outcomes.loc[
                        outcomes["family"] == family,
                        "harmful",
                    ].sum()
                )
                for family in FAMILIES
            },
            "patient_grouped_folds": True,
            "promises12_access": False,
            "threshold_selection": False,
        },
    )
    print("CM4B_PREFLIGHT_SHA256=", sha256_file(preflight_path))

    if args.preflight_only:
        print("\n===== CM4B PREFLIGHT-ONLY FINAL =====")
        print(
            "Decision= "
            "MRI_SOURCE_SAFETY_PREFLIGHT_PASS_READY_FOR_FULL_CM4B"
        )
        print("PROMISE12 access: NO")
        print("PASS")
        return

    features, feature_meta = extract_features(
        model,
        images,
        masks,
        stage_dir,
        torch,
        F,
        device,
    )

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    X = make_feature_row_table(features, outcomes)
    scores = grouped_oof_scores(X, outcomes)

    score_df = outcomes[
        [
            "family",
            "fold",
            "global_index",
            "case_key",
            "slice_index",
            "harmful",
            "neutral",
            "beneficial",
            "source_dice",
            "tent1_dice",
            "delta_dice",
        ]
    ].copy()

    for name, arr in scores.items():
        score_df[f"score_{name}"] = arr

    score_path = stage_dir / "CM4B_SOURCE_SAFETY_OOF_SCORES.csv"
    score_df.to_csv(score_path, index=False)

    metrics = metric_table(outcomes, scores)
    metric_path = stage_dir / "CM4B_SOURCE_SAFETY_OOF_METRICS.csv"
    metrics.to_csv(metric_path, index=False)

    final_macro = float(
        metrics.loc[
            (metrics["representation"] == "Final")
            & (metrics["scope"] == "MACRO_FAMILY"),
            "auroc",
        ].iloc[0]
    )

    final_auc_by_family = {
        family: float(
            metrics.loc[
                (metrics["representation"] == "Final")
                & (metrics["scope"] == family),
                "auroc",
            ].iloc[0]
        )
        for family in FAMILIES
    }

    print("\n===== PATIENT-CLUSTERED BOOTSTRAP: FINAL =====")
    bootstrap = patient_cluster_bootstrap(
        outcomes,
        scores["Final"],
    )

    signal_supported = (
        final_macro >= SOURCE_SIGNAL_AUROC_GATE
        and bootstrap["ci95_low"] > 0.5
    )
    decision = PASS_DECISION if signal_supported else STOP_DECISION

    bootstrap_path = (
        stage_dir
        / "CM4B_FINAL_MACRO_AUROC_PATIENT_BOOTSTRAP.json"
    )
    save_json(bootstrap_path, bootstrap)

    lock = {
        "status": "PASS" if signal_supported else "STOP",
        "decision": decision,
        "version": VERSION,
        "build": BUILD,
        "cm3_lock_sha256": EXPECTED_CM3_LOCK_SHA,
        "cm4a_lock_sha256": EXPECTED_CM4A_LOCK_SHA,
        "source_patients": N_PATIENTS,
        "source_slices": N_SLICES,
        "family_slice_rows": int(len(outcomes)),
        "sample_unit": "axial_2D_slice",
        "grouping_unit": "patient/case_key",
        "harm_rule": "DeltaDice <= -0.02",
        "action": "A1_TENT_1STEP",
        "dino_lock": dino_lock,
        "representation": {
            "source_prediction_conditioned": True,
            "fg_dim": DINO_DIM,
            "bg_dim": DINO_DIM,
            "source_mask_grid": [DINO_GRID, DINO_GRID],
            "source_mask_block": [SOURCE_BLOCK, SOURCE_BLOCK],
            "pca_components": PCA_COMPONENTS,
            "pca_whiten": True,
            "pca_solver": "randomized",
            "m2": [
                "source_fg_fraction",
                "source_boundary_density",
            ],
        },
        "safety_head": {
            "median_imputer": "train-fold only",
            "standard_scaler": "train-fold only",
            "classifier": "LogisticRegression",
            "class_weight": "balanced",
            "max_iter": LR_MAX_ITER,
        },
        "primary_endpoint": {
            "metric": "macro-family AUROC",
            "point_estimate": final_macro,
            "patient_cluster_bootstrap": bootstrap,
            "source_signal_gate": SOURCE_SIGNAL_AUROC_GATE,
            "required_ci95_low_gt": 0.5,
        },
        "final_family_auroc": final_auc_by_family,
        "threshold_selection": False,
        "final_all_source_head_fitting": False,
        "promises12_access": False,
        "target_calibration": False,
        "artifacts": {
            "feature_meta": str(
                stage_dir / "CM4B_FEATURE_EXTRACTION_META.json"
            ),
            "feature_meta_sha256": sha256_file(
                stage_dir / "CM4B_FEATURE_EXTRACTION_META.json"
            ),
            "scores": str(score_path),
            "scores_sha256": sha256_file(score_path),
            "metrics": str(metric_path),
            "metrics_sha256": sha256_file(metric_path),
            "bootstrap": str(bootstrap_path),
            "bootstrap_sha256": sha256_file(bootstrap_path),
            "preflight": str(preflight_path),
            "preflight_sha256": sha256_file(preflight_path),
        },
        "next_stage": (
            "CM4C_FINAL_SOURCE_SAFETY_ESTIMATOR_AND_SOURCE_THRESHOLD_LOCK"
            if signal_supported
            else "STOP_MRI_SAFETY_BRANCH"
        ),
    }

    lock_path = stage_dir / "CM4B_MRI_SOURCE_SAFETY_OOF_LOCK.json"
    save_json(lock_path, lock)

    print("\n===== CM4B SOURCE SAFETY RESULTS =====")
    print(metrics.to_string(index=False))
    print("\nPrimary Final macro-family AUROC:", final_macro)
    print(
        "Patient-cluster bootstrap 95% CI:",
        f"[{bootstrap['ci95_low']:.6f}, "
        f"{bootstrap['ci95_high']:.6f}]",
    )
    print("Final family AUROC:", final_auc_by_family)

    print("\n===== CM4B FINAL =====")
    print("PROMISE12 access: NO")
    print("Threshold selection: NO")
    print("Final all-source safety head fitting: NO")
    print("Decision=", decision)
    print("LOCK=", lock_path)
    print("LOCK SHA256=", sha256_file(lock_path))
    print("PASS" if signal_supported else "STOP")


if __name__ == "__main__":
    main()
