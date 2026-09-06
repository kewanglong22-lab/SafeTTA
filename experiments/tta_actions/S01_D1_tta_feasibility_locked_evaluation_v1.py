#!/usr/bin/env python
# -*- coding: utf-8 -*-

r"""
S01-D1: Locked TTA feasibility experiment.

Order is scientifically fixed:

PHASE A — NO LABELS
  TENT × 3 frozen seeds
  TestFit-compatible × 3 frozen seeds
  -> save six 352x352 float16 logit files
  -> save no-label diagnostics
  -> SHA256 lock everything

Only after the six-file lock exists:

PHASE B — LABEL REVEAL
  -> open GT masks
  -> reconstruct Source-Only from S01-C locked logits
  -> evaluate TTA with identical resize/threshold rules
  -> compute DeltaDice
  -> harmful / beneficial / neutral
  -> apply preregistered GO-A / GO-B without tuning

No SafeTTA method is implemented here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


VERSION = "2026-08-17-S01-D1-v1"
BUILD = "S01_D1_SIX_TTA_NO_LABEL_LOCK_THEN_DELTA_DICE_GATE"

ROOT = Path(r"F:\MEDSEG_SAFETTA")
DATA_ROOT = ROOT / "data" / "processed" / "S00_polyp_locked_v1"

MANIFEST = ROOT / "data" / "splits" / "S01_locked_protocol_manifest_v1.csv"
EXPECTED_MANIFEST_SHA256 = "afb4bc1175af004819538cdbb22ee7a10f1be48035945c7e0d50fd2bbe2e3dc7"

PROTOCOL = ROOT / "docs" / "S01_D0_tta_feasibility_protocol_freeze_v1_fix2.md"
EXPECTED_PROTOCOL_SHA256 = "edccc6c3ec3d8701957fc3b8077a15f4aeaeb0df79a60fe392b7b1a171c51ec4"

TRAINING_HELPER = ROOT / "code" / "S01_B_train_pranet_source_only_frozen_seeds_v1.py"
EXPECTED_TRAINING_HELPER_SHA256 = "2b2c9c1b75ff60bd087b403e43dc77468b31cbd54fb607481cc80ab47182b67a"

SOURCE_ONLY_DIR = ROOT / "outputs" / "S01_C_source_only_frozen_seeds_evaluation_v1"
SOURCE_ONLY_LOCK = SOURCE_ONLY_DIR / "NO_LABEL_PREDICTION_LOCK.json"
EXPECTED_SOURCE_ONLY_LOCK_SHA256 = "64cdd6c69f7287b6caa3f348eb1c5516be9bd04bc4e1834c842a1782dac1ec60"
SOURCE_ONLY_PER_IMAGE_METRICS = SOURCE_ONLY_DIR / "per_image_source_only_metrics.csv"

OUTPUT_DIR = ROOT / "outputs" / "S01_D1_tta_feasibility_v1"

IMAGE_SIZE = 352
SEEDS = (20260817, 20260818, 20260819)
METHODS = ("TENT", "TESTFIT")

CHECKPOINTS = {
    20260817: {
        "path": ROOT / "outputs" / "S01_B_pranet_source_only_seed20260817_v1"
        / "checkpoints" / "best_source_val_dice.pt",
        "sha256": "ef623c0f1207bab02377ec17932db43fe2fd1843dca858cc79ef87ead29b975a",
    },
    20260818: {
        "path": ROOT / "outputs" / "S01_B_pranet_source_only_seed20260818_v1"
        / "checkpoints" / "best_source_val_dice.pt",
        "sha256": "e56dd8ba4b5fc7785fedf3918c275427178fadb5f035b47087d48cde41ccec22",
    },
    20260819: {
        "path": ROOT / "outputs" / "S01_B_pranet_source_only_seed20260819_v1"
        / "checkpoints" / "best_source_val_dice.pt",
        "sha256": "f8cad00bbc9bbbff04c9a29547bd2a5f3beb97d53980c269cf4b7d2c30130b72",
    },
}

EXPECTED_ROLE_COUNTS = {
    "source_train": 1305,
    "source_val": 145,
    "seen_sanity": 162,
    "unseen_locked": 636,
}

EXPECTED_EVAL_COUNTS = {
    ("seen_sanity", "Kvasir-SEG"): 100,
    ("seen_sanity", "CVC-ClinicDB"): 62,
    ("unseen_locked", "CVC-ColonDB"): 380,
    ("unseen_locked", "CVC-300"): 60,
    ("unseen_locked", "ETIS-LaribPolypDB"): 196,
}

EXPECTED_EVAL_N = 798

# Frozen TENT.
TENT_LR = 1e-3
TENT_WEIGHT_DECAY = 0.0
TENT_STEPS = 1

# Frozen episodic TestFit-compatible.
TESTFIT_LR = 1e-3
TESTFIT_WEIGHT_DECAY = 0.01
TESTFIT_PSEUDO_THRESHOLD = 0.9
TESTFIT_ALPHA_COUNT = 101

# Frozen scientific gates.
HARM_THRESHOLD = -0.02
BENEFIT_THRESHOLD = +0.02
DOMAIN_MEAN_DROP_THRESHOLD = -0.01
DOMAIN_HARM_FRACTION_THRESHOLD = 0.20
POOLED_HARM_FRACTION_THRESHOLD = 0.15
POOLED_BENEFIT_FRACTION_THRESHOLD = 0.15

try:
    RESAMPLE_BILINEAR = Image.Resampling.BILINEAR
except AttributeError:
    RESAMPLE_BILINEAR = Image.BILINEAR


# =====================================================================
# Generic utilities
# =====================================================================

def file_sha256(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def read_csv(path: Path):
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = reader.fieldnames or []
    return rows, fields


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def robust_torch_load(path: Path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def final_logit(model, x):
    out = model(x)
    if isinstance(out, (tuple, list)):
        return out[-1]
    return out


def binary_entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    Elementwise Bernoulli entropy, stable for large logits.
    H(sigmoid(z)) = softplus(z) - sigmoid(z)*z
    """
    p = torch.sigmoid(logits)
    return F.softplus(logits) - p * logits


def mean_binary_entropy(logits: torch.Tensor) -> torch.Tensor:
    return binary_entropy_from_logits(logits).mean()


def classification_from_delta(delta: float) -> str:
    if delta <= HARM_THRESHOLD:
        return "harmful"
    if delta >= BENEFIT_THRESHOLD:
        return "beneficial"
    return "neutral"


# =====================================================================
# Frozen manifest / model provenance
# =====================================================================

def validate_manifest():
    if not MANIFEST.exists():
        raise FileNotFoundError(MANIFEST)
    if file_sha256(MANIFEST).lower() != EXPECTED_MANIFEST_SHA256.lower():
        raise RuntimeError("Frozen S01 manifest SHA256 mismatch.")

    rows, fields = read_csv(MANIFEST)
    required = {
        "sample_id", "dataset", "image_relpath", "mask_relpath", "s01_role"
    }
    missing = sorted(required - set(fields))
    if missing:
        raise RuntimeError(f"Manifest missing columns: {missing}")

    role_counts = Counter(r["s01_role"] for r in rows)
    if role_counts != Counter(EXPECTED_ROLE_COUNTS):
        raise RuntimeError(
            f"Role count mismatch: expected={EXPECTED_ROLE_COUNTS}, "
            f"actual={dict(role_counts)}"
        )

    eval_rows = [
        r for r in rows
        if r["s01_role"] in {"seen_sanity", "unseen_locked"}
    ]
    eval_rows = sorted(
        eval_rows,
        key=lambda r: (r["s01_role"], r["dataset"], r["sample_id"])
    )

    if len(eval_rows) != EXPECTED_EVAL_N:
        raise RuntimeError(
            f"Eval N mismatch: expected={EXPECTED_EVAL_N}, actual={len(eval_rows)}"
        )

    counts = Counter((r["s01_role"], r["dataset"]) for r in eval_rows)
    if counts != Counter(EXPECTED_EVAL_COUNTS):
        raise RuntimeError(
            f"Eval group mismatch: expected={EXPECTED_EVAL_COUNTS}, "
            f"actual={dict(counts)}"
        )

    ids = [r["sample_id"] for r in eval_rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate evaluation sample IDs.")

    return eval_rows


def import_training_helper():
    if not TRAINING_HELPER.exists():
        raise FileNotFoundError(TRAINING_HELPER)

    actual = file_sha256(TRAINING_HELPER)
    if actual.lower() != EXPECTED_TRAINING_HELPER_SHA256.lower():
        raise RuntimeError(
            "Frozen training helper SHA mismatch.\n"
            f"Expected: {EXPECTED_TRAINING_HELPER_SHA256}\n"
            f"Actual  : {actual}"
        )

    spec = importlib.util.spec_from_file_location(
        "s01b_frozen_helper", str(TRAINING_HELPER)
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to import frozen training helper.")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "PraNet") or not hasattr(module, "image_to_tensor"):
        raise RuntimeError("Frozen helper lacks PraNet/image_to_tensor.")
    return module


def validate_checkpoints():
    checked = {}
    for seed in SEEDS:
        path = CHECKPOINTS[seed]["path"]
        if not path.exists():
            raise FileNotFoundError(path)
        actual = file_sha256(path)
        if actual.lower() != CHECKPOINTS[seed]["sha256"].lower():
            raise RuntimeError(
                f"Checkpoint SHA mismatch seed={seed}\n"
                f"Expected: {CHECKPOINTS[seed]['sha256']}\n"
                f"Actual  : {actual}"
            )
        checked[seed] = actual
    return checked


def load_pranet(helper, seed: int, device: torch.device):
    ckpt = robust_torch_load(CHECKPOINTS[seed]["path"], map_location="cpu")
    if int(ckpt.get("seed", -1)) != seed:
        raise RuntimeError(f"Internal seed mismatch for checkpoint {seed}.")
    if ckpt.get("manifest_sha256") != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError(f"Checkpoint manifest SHA mismatch seed={seed}.")

    model = helper.PraNet(channel=32, pretrained_backbone_state=None)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device)
    return model


def validate_protocol():
    if not PROTOCOL.exists():
        raise FileNotFoundError(
            f"Frozen D0 FIX2 protocol missing: {PROTOCOL}"
        )
    actual = file_sha256(PROTOCOL)
    if actual.lower() != EXPECTED_PROTOCOL_SHA256.lower():
        raise RuntimeError(
            "D0 FIX2 protocol SHA mismatch.\n"
            f"Expected: {EXPECTED_PROTOCOL_SHA256}\n"
            f"Actual  : {actual}"
        )
    return actual


# =====================================================================
# S01-C Source-Only lock provenance
# =====================================================================

def validate_source_only_lock(eval_rows):
    if not SOURCE_ONLY_LOCK.exists():
        raise FileNotFoundError(SOURCE_ONLY_LOCK)

    lock_sha = file_sha256(SOURCE_ONLY_LOCK)
    if lock_sha.lower() != EXPECTED_SOURCE_ONLY_LOCK_SHA256.lower():
        raise RuntimeError(
            "S01-C no-label lock SHA mismatch.\n"
            f"Expected: {EXPECTED_SOURCE_ONLY_LOCK_SHA256}\n"
            f"Actual  : {lock_sha}"
        )

    lock = json.loads(SOURCE_ONLY_LOCK.read_text(encoding="utf-8"))
    locked_files = lock.get("locked_files", {})

    expected_ids = [r["sample_id"] for r in eval_rows]
    resolved = {}

    for seed in SEEDS:
        meta = locked_files.get(str(seed))
        if meta is None:
            raise RuntimeError(f"S01-C lock missing seed {seed}.")

        # The S01-C JSON was written before its __building directory was renamed.
        # Therefore the embedded absolute path can be stale. We reuse ONLY the
        # basename and the frozen SHA, and resolve it under the committed folder.
        basename = Path(meta["path"]).name
        actual_path = SOURCE_ONLY_DIR / "locked_no_label_logits" / basename

        if not actual_path.exists():
            raise FileNotFoundError(actual_path)

        actual_sha = file_sha256(actual_path)
        expected_sha = meta["sha256"]
        if actual_sha.lower() != expected_sha.lower():
            raise RuntimeError(
                f"Source-only locked logits changed seed={seed}."
            )

        data = np.load(actual_path, allow_pickle=False)
        ids = data["sample_ids"].tolist()
        if ids != expected_ids:
            raise RuntimeError(
                f"Source-only locked sample order mismatch seed={seed}."
            )
        if data["logits"].shape != (
            EXPECTED_EVAL_N, IMAGE_SIZE, IMAGE_SIZE
        ):
            raise RuntimeError(
                f"Source-only locked shape mismatch seed={seed}: "
                f"{data['logits'].shape}"
            )

        resolved[seed] = {
            "path": actual_path,
            "sha256": actual_sha,
        }

    return resolved, lock_sha


def load_source_only_metric_reference():
    if not SOURCE_ONLY_PER_IMAGE_METRICS.exists():
        raise FileNotFoundError(SOURCE_ONLY_PER_IMAGE_METRICS)

    rows, fields = read_csv(SOURCE_ONLY_PER_IMAGE_METRICS)
    required = {"seed", "sample_id", "dice", "iou"}
    missing = sorted(required - set(fields))
    if missing:
        raise RuntimeError(
            f"S01-C per-image metric file missing columns: {missing}"
        )

    ref = {}
    for row in rows:
        key = (int(row["seed"]), row["sample_id"])
        ref[key] = (float(row["dice"]), float(row["iou"]))

    if len(ref) != len(SEEDS) * EXPECTED_EVAL_N:
        raise RuntimeError(
            f"S01-C metric-reference count mismatch: {len(ref)}"
        )
    return ref


# =====================================================================
# Image-only / GT-only loading
# =====================================================================

def load_target_image_no_label(row, helper):
    """
    PHASE A only. This function never dereferences mask_relpath.
    """
    if row["s01_role"] not in {"seen_sanity", "unseen_locked"}:
        raise RuntimeError(f"Unexpected eval role: {row['s01_role']}")

    path = DATA_ROOT / Path(row["image_relpath"])
    if not path.exists():
        raise FileNotFoundError(path)

    with Image.open(path) as im:
        image = im.convert("RGB")
        image = image.resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            RESAMPLE_BILINEAR
        )

    return helper.image_to_tensor(image).unsqueeze(0)


def load_gt_binary(row):
    """
    PHASE B only. First label access occurs only after D1 no-label lock.
    """
    path = DATA_ROOT / Path(row["mask_relpath"])
    if not path.exists():
        raise FileNotFoundError(path)

    with Image.open(path) as ma:
        gray = np.asarray(ma.convert("L"), dtype=np.float32)

    if gray.ndim != 2:
        raise RuntimeError(f"GT is not 2D: {row['sample_id']}")

    maxv = float(gray.max())
    if maxv <= 0:
        raise RuntimeError(f"Empty GT: {row['sample_id']}")

    return (gray > (0.5 * maxv)).astype(np.uint8)


def resize_logits_to_shape(logits_352: np.ndarray, shape_hw):
    t = torch.from_numpy(
        logits_352.astype(np.float32)
    )[None, None, ...]
    return F.interpolate(
        t,
        size=tuple(shape_hw),
        mode="bilinear",
        align_corners=False,
    )[0, 0].numpy()


def dice_iou(pred: np.ndarray, gt: np.ndarray):
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    inter = int(np.logical_and(pred, gt).sum())
    pred_n = int(pred.sum())
    gt_n = int(gt.sum())
    union = int(np.logical_or(pred, gt).sum())

    dice = (2.0 * inter + 1e-7) / (pred_n + gt_n + 1e-7)
    iou = (inter + 1e-7) / (union + 1e-7)
    return float(dice), float(iou)


def metrics_from_locked_logit(logit_352, gt):
    resized = resize_logits_to_shape(logit_352, gt.shape)

    # sigmoid(z) >= 0.5 is exactly z >= 0. We avoid exp overflow.
    pred = (resized >= 0.0).astype(np.uint8)
    return dice_iou(pred, gt)


# =====================================================================
# TENT
# =====================================================================

def configure_tent(model):
    model.train()
    model.requires_grad_(False)

    params = []
    names = []

    for module_name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            # Functionally equivalent to official forced batch-stat behavior.
            # We retain source buffers in memory for efficient episodic reset,
            # but they are neither used nor updated while track_running_stats=False.
            module.track_running_stats = False

            if module.weight is not None:
                module.weight.requires_grad_(True)
                params.append(module.weight)
                names.append(f"{module_name}.weight")

            if module.bias is not None:
                module.bias.requires_grad_(True)
                params.append(module.bias)
                names.append(f"{module_name}.bias")

    if not params:
        raise RuntimeError("TENT found no BatchNorm2d affine parameters.")

    return params, names


def snapshot_parameters(params):
    return [p.detach().clone() for p in params]


@torch.no_grad()
def restore_parameters(params, source_values):
    if len(params) != len(source_values):
        raise RuntimeError("TENT reset length mismatch.")
    for p, src in zip(params, source_values):
        p.copy_(src)
        p.grad = None


def tent_adapt_one(model, params, source_values, x):
    restore_parameters(params, source_values)

    optimizer = torch.optim.Adam(
        params,
        lr=TENT_LR,
        weight_decay=TENT_WEIGHT_DECAY,
    )
    optimizer.zero_grad(set_to_none=True)

    z_pre = final_logit(model, x)
    pre_entropy = mean_binary_entropy(z_pre)

    pre_entropy.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    with torch.no_grad():
        z_post = final_logit(model, x)
        post_entropy = mean_binary_entropy(z_post)

    result = {
        "logit": z_post[0, 0].detach().float().cpu().numpy(),
        "pre_entropy": float(pre_entropy.detach().cpu()),
        "post_entropy": float(post_entropy.detach().cpu()),
    }
    result["entropy_change"] = (
        result["post_entropy"] - result["pre_entropy"]
    )

    del optimizer, z_pre, z_post
    return result


# =====================================================================
# TestFit-compatible episodic FIX2
# =====================================================================

def make_testfit_smoothed_input(x):
    pooled = F.avg_pool2d(x, kernel_size=2, stride=1)
    return F.interpolate(
        pooled,
        size=(IMAGE_SIZE, IMAGE_SIZE),
        mode="bilinear",
        align_corners=False,
    )


def select_testfit_alphas(z_p_detached, z_r_detached):
    """
    Vectorized alpha search with exact FIX2 tie behavior:
    reference uses <= for low and >= for high, hence LAST tied index.
    """
    alphas = torch.linspace(
        0.0, 1.0, TESTFIT_ALPHA_COUNT,
        device=z_p_detached.device,
        dtype=z_p_detached.dtype,
    )

    blends = (
        alphas[:, None, None, None]
        * z_p_detached
        + (1.0 - alphas[:, None, None, None])
        * z_r_detached
    )

    scores = binary_entropy_from_logits(blends).mean(
        dim=(1, 2, 3)
    )

    # Reverse argmin/argmax -> last occurrence in ascending alpha order.
    low_idx = (TESTFIT_ALPHA_COUNT - 1) - torch.argmin(
        torch.flip(scores, dims=(0,))
    )
    high_idx = (TESTFIT_ALPHA_COUNT - 1) - torch.argmax(
        torch.flip(scores, dims=(0,))
    )

    low_idx_i = int(low_idx.item())
    high_idx_i = int(high_idx.item())

    return {
        "alpha_low": low_idx_i / 100.0,
        "alpha_high": high_idx_i / 100.0,
        "entropy_low": float(scores[low_idx_i].detach().cpu()),
        "entropy_high": float(scores[high_idx_i].detach().cpu()),
    }


def reset_predictor_from_reference(predictor, reference):
    predictor.load_state_dict(reference.state_dict(), strict=True)
    predictor.train()
    predictor.requires_grad_(True)
    for p in predictor.parameters():
        p.grad = None


def testfit_adapt_one(predictor, reference, x):
    reset_predictor_from_reference(predictor, reference)

    optimizer = torch.optim.AdamW(
        predictor.parameters(),
        lr=TESTFIT_LR,
        weight_decay=TESTFIT_WEIGHT_DECAY,
    )
    optimizer.zero_grad(set_to_none=True)

    x_s = make_testfit_smoothed_input(x)

    z_p = final_logit(predictor, x_s)
    with torch.no_grad():
        z_r = final_logit(reference, x)

    alpha_info = select_testfit_alphas(
        z_p.detach(), z_r.detach()
    )

    a_low = alpha_info["alpha_low"]
    a_high = alpha_info["alpha_high"]

    # Reported one-pass output is PRE-update low-entropy blend.
    z_out = (
        a_low * z_p.detach()
        + (1.0 - a_low) * z_r.detach()
    )

    # Pseudo-label branch follows MedSeg-TTA weighted TestFit logic.
    with torch.no_grad():
        z_high = (
            a_high * z_p.detach()
            + (1.0 - a_high) * z_r.detach()
        )

        labels_prob = torch.sigmoid(z_high)

        weight1 = 2.0 * torch.abs(0.5 - labels_prob)

        p_pred = torch.sigmoid(z_p.detach())
        weight2 = 1.0 - 2.0 * torch.abs(0.5 - p_pred)

        pseudo = (
            labels_prob > TESTFIT_PSEUDO_THRESHOLD
        ).to(dtype=z_p.dtype)

    loss_map = F.binary_cross_entropy_with_logits(
        z_p,
        pseudo,
        reduction="none",
    )
    loss = torch.mean(weight1 * weight2 * loss_map)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    result = {
        "logit": z_out[0, 0].float().cpu().numpy(),
        "alpha_low": float(a_low),
        "alpha_high": float(a_high),
        "entropy_low": alpha_info["entropy_low"],
        "entropy_high": alpha_info["entropy_high"],
        "adapt_loss": float(loss.detach().cpu()),
    }

    del optimizer, z_p, z_r, z_out, loss, loss_map, x_s
    return result


# =====================================================================
# No-label file generation and resume
# =====================================================================

def no_label_npz_path(build_dir, method, seed):
    return (
        build_dir
        / "locked_no_label_tta_logits"
        / f"{method.lower()}_seed{seed}_logits_NO_LABELS.npz"
    )


def no_label_diag_path(build_dir, method, seed):
    return (
        build_dir
        / "no_label_diagnostics"
        / f"{method.lower()}_seed{seed}_diagnostics_NO_LABELS.csv"
    )


def validate_existing_no_label_npz(
    path: Path,
    method: str,
    seed: int,
    eval_rows,
):
    if not path.exists():
        return None

    data = np.load(path, allow_pickle=False)

    if data["logits"].shape != (
        EXPECTED_EVAL_N, IMAGE_SIZE, IMAGE_SIZE
    ):
        return None

    if data["sample_ids"].tolist() != [
        r["sample_id"] for r in eval_rows
    ]:
        return None

    method_stored = str(data["method"].tolist()[0])
    seed_stored = int(data["seed"].tolist()[0])
    protocol_stored = str(data["protocol_sha256"].tolist()[0])

    if method_stored != method:
        return None
    if seed_stored != seed:
        return None
    if protocol_stored != EXPECTED_PROTOCOL_SHA256:
        return None

    return file_sha256(path)


def save_no_label_npz(
    path,
    method,
    seed,
    logits,
    eval_rows,
):
    path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        path,
        logits=logits,
        sample_ids=np.asarray(
            [r["sample_id"] for r in eval_rows],
            dtype=str,
        ),
        roles=np.asarray(
            [r["s01_role"] for r in eval_rows],
            dtype=str,
        ),
        datasets=np.asarray(
            [r["dataset"] for r in eval_rows],
            dtype=str,
        ),
        method=np.asarray([method], dtype=str),
        seed=np.asarray([seed], dtype=np.int64),
        image_size=np.asarray([IMAGE_SIZE], dtype=np.int64),
        protocol_sha256=np.asarray(
            [EXPECTED_PROTOCOL_SHA256],
            dtype=str,
        ),
        checkpoint_sha256=np.asarray(
            [CHECKPOINTS[seed]["sha256"]],
            dtype=str,
        ),
    )


def generate_tent_seed(
    helper,
    seed,
    eval_rows,
    device,
    build_dir,
):
    model = load_pranet(helper, seed, device)
    params, names = configure_tent(model)
    source_values = snapshot_parameters(params)

    logits = np.empty(
        (EXPECTED_EVAL_N, IMAGE_SIZE, IMAGE_SIZE),
        dtype=np.float16,
    )
    diag_rows = []

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    pbar = tqdm(
        enumerate(eval_rows),
        total=EXPECTED_EVAL_N,
        desc=f"TENT NO-LABEL seed {seed}",
        unit="img",
        dynamic_ncols=True,
    )

    for idx, row in pbar:
        x = load_target_image_no_label(row, helper).to(
            device, non_blocking=True
        )

        result = tent_adapt_one(
            model, params, source_values, x
        )

        if not np.isfinite(result["logit"]).all():
            raise RuntimeError(
                f"TENT non-finite logit seed={seed} "
                f"sample={row['sample_id']}"
            )

        logits[idx] = result["logit"].astype(np.float16)

        diag_rows.append({
            "sample_id": row["sample_id"],
            "role": row["s01_role"],
            "dataset": row["dataset"],
            "pre_entropy": f"{result['pre_entropy']:.10f}",
            "post_entropy": f"{result['post_entropy']:.10f}",
            "entropy_change": f"{result['entropy_change']:.10f}",
        })

        pbar.set_postfix(
            Hpre=f"{result['pre_entropy']:.4f}",
            Hpost=f"{result['post_entropy']:.4f}",
        )

    npz_path = no_label_npz_path(build_dir, "TENT", seed)
    save_no_label_npz(
        npz_path, "TENT", seed, logits, eval_rows
    )

    diag_path = no_label_diag_path(build_dir, "TENT", seed)
    write_csv(
        diag_path,
        diag_rows,
        [
            "sample_id", "role", "dataset",
            "pre_entropy", "post_entropy", "entropy_change",
        ],
    )

    peak_gb = (
        float(torch.cuda.max_memory_allocated() / 1024**3)
        if device.type == "cuda"
        else 0.0
    )

    metadata = {
        "bn_affine_tensor_count": len(params),
        "bn_affine_numel": int(sum(p.numel() for p in params)),
        "peak_gpu_memory_gb": peak_gb,
    }

    del model, params, source_values, logits
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return npz_path, diag_path, metadata


def generate_testfit_seed(
    helper,
    seed,
    eval_rows,
    device,
    build_dir,
):
    predictor = load_pranet(helper, seed, device)
    reference = load_pranet(helper, seed, device)
    reference.eval()
    reference.requires_grad_(False)

    logits = np.empty(
        (EXPECTED_EVAL_N, IMAGE_SIZE, IMAGE_SIZE),
        dtype=np.float16,
    )
    diag_rows = []

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    pbar = tqdm(
        enumerate(eval_rows),
        total=EXPECTED_EVAL_N,
        desc=f"TESTFIT NO-LABEL seed {seed}",
        unit="img",
        dynamic_ncols=True,
    )

    for idx, row in pbar:
        x = load_target_image_no_label(row, helper).to(
            device, non_blocking=True
        )

        result = testfit_adapt_one(
            predictor, reference, x
        )

        if not np.isfinite(result["logit"]).all():
            raise RuntimeError(
                f"TESTFIT non-finite logit seed={seed} "
                f"sample={row['sample_id']}"
            )

        logits[idx] = result["logit"].astype(np.float16)

        diag_rows.append({
            "sample_id": row["sample_id"],
            "role": row["s01_role"],
            "dataset": row["dataset"],
            "alpha_low": f"{result['alpha_low']:.2f}",
            "alpha_high": f"{result['alpha_high']:.2f}",
            "entropy_low": f"{result['entropy_low']:.10f}",
            "entropy_high": f"{result['entropy_high']:.10f}",
            "adapt_loss": f"{result['adapt_loss']:.10f}",
        })

        pbar.set_postfix(
            aLow=f"{result['alpha_low']:.2f}",
            aHigh=f"{result['alpha_high']:.2f}",
            loss=f"{result['adapt_loss']:.4f}",
        )

    npz_path = no_label_npz_path(
        build_dir, "TESTFIT", seed
    )
    save_no_label_npz(
        npz_path, "TESTFIT", seed, logits, eval_rows
    )

    diag_path = no_label_diag_path(
        build_dir, "TESTFIT", seed
    )
    write_csv(
        diag_path,
        diag_rows,
        [
            "sample_id", "role", "dataset",
            "alpha_low", "alpha_high",
            "entropy_low", "entropy_high", "adapt_loss",
        ],
    )

    peak_gb = (
        float(torch.cuda.max_memory_allocated() / 1024**3)
        if device.type == "cuda"
        else 0.0
    )
    metadata = {
        "peak_gpu_memory_gb": peak_gb,
    }

    del predictor, reference, logits
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return npz_path, diag_path, metadata


def update_partial_completion(
    build_dir,
    completed,
):
    path = build_dir / "NO_LABEL_PARTIAL_COMPLETION.json"
    payload = {
        "script_version": VERSION,
        "build": BUILD,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "target_masks_opened": False,
        "target_metrics_computed": False,
        "completed": completed,
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def produce_or_resume_no_label(
    helper,
    eval_rows,
    device,
    build_dir,
    resume: bool,
):
    completed = {}

    for method in METHODS:
        for seed in SEEDS:
            key = f"{method}::{seed}"
            npz_path = no_label_npz_path(
                build_dir, method, seed
            )
            diag_path = no_label_diag_path(
                build_dir, method, seed
            )

            reused = False
            existing_sha = None

            if resume and npz_path.exists() and diag_path.exists():
                existing_sha = validate_existing_no_label_npz(
                    npz_path, method, seed, eval_rows
                )
                if existing_sha is not None:
                    reused = True

            if reused:
                print(
                    f"[RESUME] Reusing verified no-label output "
                    f"{method} seed={seed}"
                )
                metadata = {"resumed": True}
            else:
                # Never silently overwrite a suspicious partial file.
                if npz_path.exists():
                    npz_path.unlink()
                if diag_path.exists():
                    diag_path.unlink()

                started = time.time()

                if method == "TENT":
                    npz_path, diag_path, metadata = (
                        generate_tent_seed(
                            helper, seed, eval_rows,
                            device, build_dir
                        )
                    )
                elif method == "TESTFIT":
                    npz_path, diag_path, metadata = (
                        generate_testfit_seed(
                            helper, seed, eval_rows,
                            device, build_dir
                        )
                    )
                else:
                    raise AssertionError(method)

                metadata["seconds"] = time.time() - started

            npz_sha = file_sha256(npz_path)
            diag_sha = file_sha256(diag_path)

            completed[key] = {
                "method": method,
                "seed": seed,
                "logits_path": str(npz_path),
                "logits_sha256": npz_sha,
                "diagnostics_path": str(diag_path),
                "diagnostics_sha256": diag_sha,
                "metadata": metadata,
            }
            update_partial_completion(build_dir, completed)

    return completed


def make_no_label_lock(build_dir, completed):
    if len(completed) != len(METHODS) * len(SEEDS):
        raise RuntimeError(
            f"Six-file no-label set incomplete: {len(completed)}"
        )

    # Re-verify every file immediately before lock creation.
    for key, meta in completed.items():
        lp = Path(meta["logits_path"])
        dp = Path(meta["diagnostics_path"])
        if file_sha256(lp) != meta["logits_sha256"]:
            raise RuntimeError(f"Logit file changed before lock: {key}")
        if file_sha256(dp) != meta["diagnostics_sha256"]:
            raise RuntimeError(f"Diagnostic file changed before lock: {key}")

    lock = {
        "script_version": VERSION,
        "build": BUILD,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "source_only_lock_sha256": EXPECTED_SOURCE_ONLY_LOCK_SHA256,
        "target_masks_opened_before_lock": False,
        "target_metrics_computed_before_lock": False,
        "tta_method_count": len(METHODS),
        "seed_count": len(SEEDS),
        "locked_tta_logit_file_count": len(completed),
        "entries": completed,
    }

    path = build_dir / "NO_LABEL_TTA_LOCK.json"
    path.write_text(
        json.dumps(lock, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    sha = file_sha256(path)

    print()
    print("===== S01-D1 NO-LABEL TTA LOCK COMPLETE =====")
    for key in sorted(completed):
        print(
            f"  {key}: "
            f"{completed[key]['logits_sha256']}"
        )
    print(f"  LOCK SHA256: {sha}")
    print("  Target masks opened before lock: NO")
    print("  Target metrics computed before lock: NO")
    print()

    return path, sha


# =====================================================================
# Label-reveal evaluation
# =====================================================================

def load_locked_npz_checked(path, expected_sha, method, seed, eval_rows):
    if file_sha256(path) != expected_sha:
        raise RuntimeError(
            f"Locked TTA SHA mismatch {method} seed={seed}"
        )

    data = np.load(path, allow_pickle=False)
    if data["sample_ids"].tolist() != [
        r["sample_id"] for r in eval_rows
    ]:
        raise RuntimeError(
            f"TTA sample order mismatch {method} seed={seed}"
        )
    if data["logits"].shape != (
        EXPECTED_EVAL_N, IMAGE_SIZE, IMAGE_SIZE
    ):
        raise RuntimeError(
            f"TTA shape mismatch {method} seed={seed}"
        )
    return data["logits"]


def evaluate_after_lock(
    eval_rows,
    completed,
    source_locked,
):
    """
    First D1 target-mask access occurs only inside this function.
    """
    source_reference = load_source_only_metric_reference()

    per_image_rows = []

    for seed in SEEDS:
        source_npz = np.load(
            source_locked[seed]["path"],
            allow_pickle=False
        )
        source_logits = source_npz["logits"]

        for method in METHODS:
            meta = completed[f"{method}::{seed}"]
            tta_logits = load_locked_npz_checked(
                Path(meta["logits_path"]),
                meta["logits_sha256"],
                method,
                seed,
                eval_rows,
            )

            pbar = tqdm(
                enumerate(eval_rows),
                total=EXPECTED_EVAL_N,
                desc=f"LABEL-REVEAL {method} seed {seed}",
                unit="img",
                dynamic_ncols=True,
            )

            for idx, row in pbar:
                gt = load_gt_binary(row)

                source_dice, source_iou = metrics_from_locked_logit(
                    source_logits[idx], gt
                )
                tta_dice, tta_iou = metrics_from_locked_logit(
                    tta_logits[idx], gt
                )

                ref_dice, ref_iou = source_reference[
                    (seed, row["sample_id"])
                ]

                # S01-C used the same locked float16 logits and the same
                # resize/threshold rule. Any mismatch indicates protocol drift.
                if abs(source_dice - ref_dice) > 2e-8:
                    raise RuntimeError(
                        f"Source Dice reconstruction mismatch "
                        f"seed={seed} sample={row['sample_id']} "
                        f"{source_dice} vs {ref_dice}"
                    )
                if abs(source_iou - ref_iou) > 2e-8:
                    raise RuntimeError(
                        f"Source IoU reconstruction mismatch "
                        f"seed={seed} sample={row['sample_id']}"
                    )

                delta = tta_dice - source_dice
                label = classification_from_delta(delta)

                per_image_rows.append({
                    "method": method,
                    "seed": seed,
                    "sample_id": row["sample_id"],
                    "role": row["s01_role"],
                    "dataset": row["dataset"],
                    "source_dice": f"{source_dice:.10f}",
                    "tta_dice": f"{tta_dice:.10f}",
                    "delta_dice": f"{delta:.10f}",
                    "source_iou": f"{source_iou:.10f}",
                    "tta_iou": f"{tta_iou:.10f}",
                    "delta_class": label,
                })

                pbar.set_postfix(
                    dDice=f"{delta:+.4f}",
                    cls=label[0].upper(),
                )

    return per_image_rows


# =====================================================================
# Diagnostics and scientific gate
# =====================================================================

def fraction(count, n):
    return float(count) / float(n) if n else float("nan")


def summarize_delta(per_image_rows):
    group_rows = []

    for method in METHODS:
        for seed in SEEDS:
            for (role, dataset), expected_n in EXPECTED_EVAL_COUNTS.items():
                group = [
                    r for r in per_image_rows
                    if r["method"] == method
                    and int(r["seed"]) == seed
                    and r["role"] == role
                    and r["dataset"] == dataset
                ]
                if len(group) != expected_n:
                    raise RuntimeError(
                        f"Group count mismatch "
                        f"{method}/{seed}/{role}/{dataset}: "
                        f"{len(group)} vs {expected_n}"
                    )

                source = np.asarray(
                    [float(r["source_dice"]) for r in group],
                    dtype=np.float64,
                )
                tta = np.asarray(
                    [float(r["tta_dice"]) for r in group],
                    dtype=np.float64,
                )
                delta = np.asarray(
                    [float(r["delta_dice"]) for r in group],
                    dtype=np.float64,
                )

                cls = Counter(r["delta_class"] for r in group)
                harmful_n = cls["harmful"]
                beneficial_n = cls["beneficial"]
                neutral_n = cls["neutral"]

                mean_delta = float(delta.mean())
                harmful_fraction = fraction(harmful_n, len(group))

                domain_harmful = (
                    role == "unseen_locked"
                    and mean_delta <= DOMAIN_MEAN_DROP_THRESHOLD
                    and harmful_fraction >= DOMAIN_HARM_FRACTION_THRESHOLD
                )

                group_rows.append({
                    "method": method,
                    "seed": seed,
                    "role": role,
                    "dataset": dataset,
                    "n": len(group),
                    "source_mean_dice": f"{source.mean():.10f}",
                    "tta_mean_dice": f"{tta.mean():.10f}",
                    "mean_delta_dice": f"{mean_delta:.10f}",
                    "median_delta_dice": f"{np.median(delta):.10f}",
                    "harmful_n": harmful_n,
                    "harmful_fraction": f"{harmful_fraction:.10f}",
                    "beneficial_n": beneficial_n,
                    "beneficial_fraction": f"{fraction(beneficial_n, len(group)):.10f}",
                    "neutral_n": neutral_n,
                    "neutral_fraction": f"{fraction(neutral_n, len(group)):.10f}",
                    "domain_harmful": domain_harmful,
                })

    pooled_rows = []
    go_b_by_method = {}

    for method in METHODS:
        group = [
            r for r in per_image_rows
            if r["method"] == method
            and r["role"] == "unseen_locked"
        ]

        expected = len(SEEDS) * EXPECTED_ROLE_COUNTS["unseen_locked"]
        if len(group) != expected:
            raise RuntimeError(
                f"Pooled unseen count mismatch method={method}: "
                f"{len(group)} vs {expected}"
            )

        cls = Counter(r["delta_class"] for r in group)
        harmful_fraction = fraction(cls["harmful"], len(group))
        beneficial_fraction = fraction(cls["beneficial"], len(group))
        deltas = np.asarray(
            [float(r["delta_dice"]) for r in group],
            dtype=np.float64,
        )

        go_b = (
            harmful_fraction >= POOLED_HARM_FRACTION_THRESHOLD
            and beneficial_fraction >= POOLED_BENEFIT_FRACTION_THRESHOLD
        )
        go_b_by_method[method] = go_b

        pooled_rows.append({
            "method": method,
            "n_image_seed_pairs": len(group),
            "mean_delta_dice": f"{deltas.mean():.10f}",
            "median_delta_dice": f"{np.median(deltas):.10f}",
            "harmful_n": cls["harmful"],
            "harmful_fraction": f"{harmful_fraction:.10f}",
            "beneficial_n": cls["beneficial"],
            "beneficial_fraction": f"{beneficial_fraction:.10f}",
            "neutral_n": cls["neutral"],
            "neutral_fraction": f"{fraction(cls['neutral'], len(group)):.10f}",
            "GO_B": go_b,
        })

    go_a_groups = [
        r for r in group_rows
        if r["role"] == "unseen_locked"
        and bool(r["domain_harmful"])
    ]
    go_a = len(go_a_groups) > 0
    go_b = any(go_b_by_method.values())

    decision = (
        "GO_TO_S02_UNLABELED_HARM_PREDICTION"
        if (go_a or go_b)
        else "STOP_SAFETTA_NEGATIVE_TRANSFER_NOT_SUFFICIENT"
    )

    return group_rows, pooled_rows, go_a_groups, go_a, go_b, decision


def summarize_no_label_diagnostics(build_dir):
    tent_summary = []
    testfit_summary = []

    for seed in SEEDS:
        tent_rows, _ = read_csv(
            no_label_diag_path(build_dir, "TENT", seed)
        )
        for role, dataset in EXPECTED_EVAL_COUNTS:
            group = [
                r for r in tent_rows
                if r["role"] == role and r["dataset"] == dataset
            ]
            pre = np.asarray(
                [float(r["pre_entropy"]) for r in group]
            )
            post = np.asarray(
                [float(r["post_entropy"]) for r in group]
            )
            tent_summary.append({
                "seed": seed,
                "role": role,
                "dataset": dataset,
                "n": len(group),
                "mean_pre_entropy": f"{pre.mean():.10f}",
                "mean_post_entropy": f"{post.mean():.10f}",
                "mean_entropy_change": f"{(post-pre).mean():.10f}",
                "median_entropy_change": f"{np.median(post-pre):.10f}",
            })

        tf_rows, _ = read_csv(
            no_label_diag_path(build_dir, "TESTFIT", seed)
        )
        for role, dataset in EXPECTED_EVAL_COUNTS:
            group = [
                r for r in tf_rows
                if r["role"] == role and r["dataset"] == dataset
            ]
            low = np.asarray(
                [float(r["alpha_low"]) for r in group]
            )
            high = np.asarray(
                [float(r["alpha_high"]) for r in group]
            )
            testfit_summary.append({
                "seed": seed,
                "role": role,
                "dataset": dataset,
                "n": len(group),
                "mean_alpha_low": f"{low.mean():.10f}",
                "median_alpha_low": f"{np.median(low):.10f}",
                "mean_alpha_high": f"{high.mean():.10f}",
                "median_alpha_high": f"{np.median(high):.10f}",
                "frac_alpha_low_0": f"{np.mean(low == 0.0):.10f}",
                "frac_alpha_low_1": f"{np.mean(low == 1.0):.10f}",
            })

    return tent_summary, testfit_summary


# =====================================================================
# Preflight / run
# =====================================================================

def preflight():
    protocol_sha = validate_protocol()
    eval_rows = validate_manifest()
    checkpoint_hashes = validate_checkpoints()
    helper = import_training_helper()
    source_locked, source_lock_sha = validate_source_only_lock(eval_rows)

    print("===== S01-D1 PREFLIGHT PASS =====")
    print(f"Protocol SHA256: {protocol_sha}")
    print(f"Manifest SHA256: {EXPECTED_MANIFEST_SHA256}")
    print(f"Training helper SHA256: {EXPECTED_TRAINING_HELPER_SHA256}")
    print(f"Source-only lock SHA256: {source_lock_sha}")
    print(f"Evaluation images: {len(eval_rows)}")
    print(f"Seeds: {SEEDS}")
    print(f"Methods: {METHODS}")
    print("Checkpoints:")
    for seed in SEEDS:
        print(f"  {seed}: {checkpoint_hashes[seed]}")
    print("Source-only locked logits:")
    for seed in SEEDS:
        print(f"  {seed}: {source_locked[seed]['sha256']}")
    print("Target images opened=NO")
    print("Target masks opened=NO")
    print("Target metrics computed=NO")
    print("Decision: READY_FOR_S01_D1_NO_LABEL_TTA")
    return eval_rows, helper, source_locked


def run(args):
    eval_rows, helper, source_locked = preflight()

    if args.output_dir.exists():
        raise FileExistsError(
            f"Final D1 output already exists: {args.output_dir}\n"
            "The first locked D1 result must not be overwritten."
        )

    build_dir = Path(str(args.output_dir) + "__building")

    if build_dir.exists() and not args.resume:
        raise FileExistsError(
            f"Partial D1 build exists: {build_dir}\n"
            "Use --resume to continue the same frozen no-label protocol."
        )

    if not build_dir.exists():
        build_dir.mkdir(parents=True, exist_ok=False)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )

    print()
    print(f"[BUILD] {VERSION} | {BUILD}")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    print("AMP during adaptation: NO")
    print("Phase A target masks: LOCKED / NOT OPENED")
    print()

    # --------------------------------------------------------------
    # PHASE A — six no-label TTA files.
    # --------------------------------------------------------------
    completed = produce_or_resume_no_label(
        helper=helper,
        eval_rows=eval_rows,
        device=device,
        build_dir=build_dir,
        resume=args.resume,
    )

    lock_path = build_dir / "NO_LABEL_TTA_LOCK.json"

    if lock_path.exists() and args.resume:
        # Existing lock is only accepted if all current files match its hashes.
        old_lock = json.loads(lock_path.read_text(encoding="utf-8"))
        if old_lock.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA256:
            raise RuntimeError("Existing D1 lock protocol SHA mismatch.")

        for key, meta in completed.items():
            old = old_lock["entries"].get(key)
            if old is None:
                raise RuntimeError(
                    f"Existing D1 lock missing entry {key}"
                )
            if old["logits_sha256"] != meta["logits_sha256"]:
                raise RuntimeError(
                    f"Existing D1 lock logit SHA mismatch {key}"
                )
        lock_sha = file_sha256(lock_path)
        print(
            f"[RESUME] Existing six-file no-label lock verified: "
            f"{lock_sha}"
        )
    else:
        lock_path, lock_sha = make_no_label_lock(
            build_dir, completed
        )

    # Re-verify lock immediately before the first target-mask read.
    if not lock_path.exists():
        raise RuntimeError("D1 no-label lock missing before label reveal.")
    lock_sha_before_labels = file_sha256(lock_path)

    # --------------------------------------------------------------
    # PHASE B — first target-mask access.
    # --------------------------------------------------------------
    print("===== BEGIN LABEL REVEAL =====")
    print(
        f"NO_LABEL_TTA_LOCK verified SHA256="
        f"{lock_sha_before_labels}"
    )

    per_image_rows = evaluate_after_lock(
        eval_rows=eval_rows,
        completed=completed,
        source_locked=source_locked,
    )

    group_rows, pooled_rows, go_a_groups, go_a, go_b, decision = (
        summarize_delta(per_image_rows)
    )

    tent_diag_summary, testfit_diag_summary = (
        summarize_no_label_diagnostics(build_dir)
    )

    # --------------------------------------------------------------
    # Persist label-derived outputs.
    # --------------------------------------------------------------
    write_csv(
        build_dir / "per_image_delta_dice.csv",
        per_image_rows,
        [
            "method", "seed", "sample_id", "role", "dataset",
            "source_dice", "tta_dice", "delta_dice",
            "source_iou", "tta_iou", "delta_class",
        ],
    )

    write_csv(
        build_dir / "method_seed_dataset_summary.csv",
        group_rows,
        [
            "method", "seed", "role", "dataset", "n",
            "source_mean_dice", "tta_mean_dice",
            "mean_delta_dice", "median_delta_dice",
            "harmful_n", "harmful_fraction",
            "beneficial_n", "beneficial_fraction",
            "neutral_n", "neutral_fraction",
            "domain_harmful",
        ],
    )

    write_csv(
        build_dir / "pooled_unseen_method_summary.csv",
        pooled_rows,
        [
            "method", "n_image_seed_pairs",
            "mean_delta_dice", "median_delta_dice",
            "harmful_n", "harmful_fraction",
            "beneficial_n", "beneficial_fraction",
            "neutral_n", "neutral_fraction", "GO_B",
        ],
    )

    write_csv(
        build_dir / "tent_no_label_diagnostic_summary.csv",
        tent_diag_summary,
        [
            "seed", "role", "dataset", "n",
            "mean_pre_entropy", "mean_post_entropy",
            "mean_entropy_change", "median_entropy_change",
        ],
    )

    write_csv(
        build_dir / "testfit_no_label_diagnostic_summary.csv",
        testfit_diag_summary,
        [
            "seed", "role", "dataset", "n",
            "mean_alpha_low", "median_alpha_low",
            "mean_alpha_high", "median_alpha_high",
            "frac_alpha_low_0", "frac_alpha_low_1",
        ],
    )

    result = {
        "script_version": VERSION,
        "build": BUILD,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "source_only_lock_sha256": EXPECTED_SOURCE_ONLY_LOCK_SHA256,
        "d1_no_label_tta_lock_sha256": lock_sha_before_labels,
        "all_six_tta_logits_locked_before_label_reveal": True,
        "target_masks_opened_before_d1_lock": False,
        "target_metrics_computed_before_d1_lock": False,
        "methods": list(METHODS),
        "seeds": list(SEEDS),
        "harm_threshold": HARM_THRESHOLD,
        "benefit_threshold": BENEFIT_THRESHOLD,
        "domain_mean_drop_threshold": DOMAIN_MEAN_DROP_THRESHOLD,
        "domain_harm_fraction_threshold": DOMAIN_HARM_FRACTION_THRESHOLD,
        "pooled_harm_fraction_threshold": POOLED_HARM_FRACTION_THRESHOLD,
        "pooled_benefit_fraction_threshold": POOLED_BENEFIT_FRACTION_THRESHOLD,
        "GO_A": go_a,
        "GO_A_groups": go_a_groups,
        "GO_B": go_b,
        "GO_B_by_method": {
            row["method"]: bool(row["GO_B"])
            for row in pooled_rows
        },
        "decision": decision,
    }

    (build_dir / "decision.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = [
        "===== S01-D1 TTA NEGATIVE-TRANSFER FEASIBILITY =====",
        f"Script version: {VERSION}",
        f"Build: {BUILD}",
        "",
        "Locking:",
        "  Six TTA no-label logit files locked before GT reveal=YES",
        f"  NO_LABEL_TTA_LOCK SHA256={lock_sha_before_labels}",
        "  Target masks opened before lock=NO",
        "  Target metrics computed before lock=NO",
        "",
        "Frozen thresholds:",
        f"  harmful: DeltaDice <= {HARM_THRESHOLD:+.2f}",
        f"  beneficial: DeltaDice >= {BENEFIT_THRESHOLD:+.2f}",
        f"  domain harmful: meanDelta <= {DOMAIN_MEAN_DROP_THRESHOLD:+.2f} "
        f"AND harmful_fraction >= {DOMAIN_HARM_FRACTION_THRESHOLD:.2f}",
        "",
        "Per method / seed / dataset:",
    ]

    for row in group_rows:
        summary.append(
            f"  {row['method']:7s} seed={row['seed']} "
            f"{row['role']:13s} {row['dataset']:22s} "
            f"N={row['n']:3d} "
            f"Src={float(row['source_mean_dice']):.6f} "
            f"TTA={float(row['tta_mean_dice']):.6f} "
            f"dDice={float(row['mean_delta_dice']):+.6f} "
            f"H={float(row['harmful_fraction']):.3f} "
            f"B={float(row['beneficial_fraction']):.3f} "
            f"DomainHarm={row['domain_harmful']}"
        )

    summary += [
        "",
        "Pooled unseen across 3 frozen seeds:",
    ]

    for row in pooled_rows:
        summary.append(
            f"  {row['method']:7s} "
            f"N={row['n_image_seed_pairs']} "
            f"meanDelta={float(row['mean_delta_dice']):+.6f} "
            f"harmful={float(row['harmful_fraction']):.3f} "
            f"beneficial={float(row['beneficial_fraction']):.3f} "
            f"GO-B={row['GO_B']}"
        )

    summary += [
        "",
        f"GO-A={go_a}",
        f"GO-B={go_b}",
    ]

    if go_a_groups:
        summary.append("GO-A triggering groups:")
        for row in go_a_groups:
            summary.append(
                f"  {row['method']} seed={row['seed']} "
                f"{row['dataset']} "
                f"meanDelta={float(row['mean_delta_dice']):+.6f} "
                f"harmful={float(row['harmful_fraction']):.3f}"
            )

    summary += [
        "",
        f"Decision: {decision}",
    ]

    (build_dir / "summary.txt").write_text(
        "\n".join(summary) + "\n",
        encoding="utf-8",
    )

    # Final atomic commit. No result exists under final name until everything passes.
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    build_dir.rename(args.output_dir)

    print()
    print((args.output_dir / "summary.txt").read_text(encoding="utf-8"))
    print(f"[OK] Outputs: {args.output_dir}")


# =====================================================================
# Self-test
# =====================================================================

class ToySeg(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(4)
        self.conv2 = nn.Conv2d(4, 1, 1)

    def forward(self, x):
        x = F.relu(self.bn(self.conv1(x)))
        return self.conv2(x)


def self_test():
    # Entropy exact point.
    z = torch.zeros(1, 1, 4, 4)
    h = float(mean_binary_entropy(z))
    assert abs(h - math.log(2.0)) < 1e-6

    # FIX2 tie semantics: equal predictor/reference => all alphas tied;
    # <= and >= reference semantics both end at alpha=1.00.
    zp = torch.zeros(1, 1, 8, 8)
    zr = torch.zeros(1, 1, 8, 8)
    info = select_testfit_alphas(zp, zr)
    assert info["alpha_low"] == 1.0
    assert info["alpha_high"] == 1.0

    # Smoothing shape.
    x = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    xs = make_testfit_smoothed_input(x)
    assert xs.shape == x.shape
    assert torch.isfinite(xs).all()

    # Delta thresholds.
    assert classification_from_delta(-0.0200001) == "harmful"
    assert classification_from_delta(-0.0199999) == "neutral"
    assert classification_from_delta(+0.0199999) == "neutral"
    assert classification_from_delta(+0.0200001) == "beneficial"

    # TENT toy step: only BN affine is trainable.
    torch.manual_seed(7)
    tent_model = ToySeg()
    params, names = configure_tent(tent_model)
    assert len(params) == 2
    assert all("bn." in n for n in names)
    assert sum(p.requires_grad for p in tent_model.parameters()) == 2
    src = snapshot_parameters(params)
    result = tent_adapt_one(
        tent_model, params, src, torch.randn(1, 3, 16, 16)
    )
    assert result["logit"].shape == (16, 16)
    assert np.isfinite(result["logit"]).all()

    # TestFit toy execution.
    torch.manual_seed(11)
    predictor = ToySeg()
    reference = ToySeg()
    reference.load_state_dict(predictor.state_dict())
    reference.eval()
    reference.requires_grad_(False)
    tf = testfit_adapt_one(
        predictor, reference, torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    )
    assert tf["logit"].shape == (IMAGE_SIZE, IMAGE_SIZE)
    assert 0.0 <= tf["alpha_low"] <= 1.0
    assert 0.0 <= tf["alpha_high"] <= 1.0
    assert np.isfinite(tf["logit"]).all()

    # Metric.
    gt = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    d, i = dice_iou(gt.copy(), gt)
    assert abs(d - 1.0) < 1e-6
    assert abs(i - 1.0) < 1e-6

    # Count / gate constants.
    assert sum(EXPECTED_EVAL_COUNTS.values()) == EXPECTED_EVAL_N
    assert SEEDS == (20260817, 20260818, 20260819)
    assert METHODS == ("TENT", "TESTFIT")
    assert HARM_THRESHOLD == -0.02
    assert BENEFIT_THRESHOLD == 0.02

    print("BINARY_ENTROPY_TEST_PASS")
    print("TESTFIT_LAST_TIE_ALPHA_TEST_PASS")
    print("TESTFIT_SMOOTHING_TEST_PASS")
    print("DELTA_THRESHOLD_TEST_PASS")
    print("TENT_TOY_ADAPT_TEST_PASS")
    print("TESTFIT_TOY_EXECUTION_TEST_PASS")
    print("METRIC_TEST_PASS")
    print("FROZEN_GATE_TEST_PASS")
    print("SELF_TEST_PASS")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "S01-D1: generate six no-label TTA locks, then evaluate "
            "preregistered negative-transfer GO/STOP."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume the same frozen D1 build after interruption. "
            "Verified completed no-label method/seed files are reused."
        ),
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help=(
            "Verify protocol, manifest, checkpoints, helper, and "
            "S01-C locks without opening target images or masks."
        ),
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="CPU execution; intended only for debugging.",
    )
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.self_test:
        self_test()
        return 0

    if args.preflight:
        preflight()
        return 0

    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
