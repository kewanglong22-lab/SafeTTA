#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
S05-C: Build and lock the DeepLabV3-specific SafeTTA-v1 source-side gate.

No target images, masks, predictions, or target DeltaDice are accessed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
from pathlib import Path

ROOT = Path(r"F:\MEDSEG_SAFETTA")
os.environ["TORCH_HOME"] = str(ROOT / "assets" / "torchvision_cache")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

try:
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, average_precision_score
except ImportError as exc:
    raise RuntimeError("scikit-learn is required for S05-C.") from exc


VERSION = "2026-08-18-S05-C-v1"
BUILD = "S05_C_DEEPLAB_SOURCE_GATE_SINGLETON_SAFE_TENT"

DATA_ROOT = ROOT / "data" / "processed" / "S00_polyp_locked_v1"

MANIFEST = ROOT / "data" / "splits" / "S01_locked_protocol_manifest_v1.csv"
EXPECTED_MANIFEST_SHA256 = "afb4bc1175af004819538cdbb22ee7a10f1be48035945c7e0d50fd2bbe2e3dc7"

PROTOCOL = ROOT / "docs" / "S05_C_deeplab_source_side_gate_protocol_v1.md"
EXPECTED_PROTOCOL_SHA256 = "b04a1cb1d82c97b18fdc01cd936f5fc3899960ccf46469eca0324c08415c0552"

TRAINING_HELPER = (
    ROOT / "code"
    / "S05_B_train_deeplabv3_resnet50_source_only_frozen_seeds_v1_fix2.py"
)
EXPECTED_TRAINING_HELPER_SHA256 = "7d0fd5192a9af40dbae0aeac692a65641745fbef2e986f1cd4e163a59ad93a31"

OUTPUT_DIR = ROOT / "outputs" / "S05_C_deeplab_source_side_safettta_gate_v1"

SEEDS = (20260817, 20260818, 20260819)
SOURCE_VAL_N = 145
IMAGE_SIZE = 352

CHECKPOINTS = {
    20260817: {
        "path": ROOT / "outputs"
        / "S05_B_deeplabv3_r50_source_only_seed20260817_v1"
        / "checkpoints" / "best_source_val_dice.pt",
        "sha256": "d63f914e337295652627c773332d8e7382fbcd6a39d69805e1316dcc759605a2",
    },
    20260818: {
        "path": ROOT / "outputs"
        / "S05_B_deeplabv3_r50_source_only_seed20260818_v1"
        / "checkpoints" / "best_source_val_dice.pt",
        "sha256": "d4ede45d5b30f62fdbc92cd9aa8fc84e0d5a52073d5ef2c3ac36baa09a56eb25",
    },
    20260819: {
        "path": ROOT / "outputs"
        / "S05_B_deeplabv3_r50_source_only_seed20260819_v1"
        / "checkpoints" / "best_source_val_dice.pt",
        "sha256": "66aeae338d350db5aa878bcaab2e68bbe98eec8e626e28a1bb02b6708fa358eb",
    },
}

PERTURBATIONS = (
    "identity",
    "brightness_070",
    "brightness_130",
    "contrast_065",
    "saturation_050",
    "gamma_070",
    "gamma_150",
    "gaussian_blur_r15",
    "gaussian_noise_s004",
    "downsample_050",
)

HARM_THRESHOLD = -0.02
BENEFIT_THRESHOLD = +0.02
REQUIRED_HARM_RECALL = 0.80
TENT_LR = 1e-3
TENT_WEIGHT_DECAY = 0.0

FEATURE_NAMES = [
    "source_entropy_mean",
    "source_entropy_std",
    "source_prob_mean",
    "source_prob_std",
    "source_prob_q10",
    "source_prob_q50",
    "source_prob_q90",
    "source_fg_fraction",
    "source_boundary_density",
    "source_confidence_mean",
    "tent_entropy_mean",
    "tent_entropy_std",
    "tent_prob_mean",
    "tent_prob_std",
    "tent_prob_q10",
    "tent_prob_q50",
    "tent_prob_q90",
    "tent_fg_fraction",
    "tent_boundary_density",
    "tent_confidence_mean",
    "entropy_mean_shift",
    "prob_abs_change_mean",
    "mask_disagreement_fraction",
    "abs_fg_fraction_shift",
]

try:
    RESAMPLE_BILINEAR = Image.Resampling.BILINEAR
except AttributeError:
    RESAMPLE_BILINEAR = Image.BILINEAR


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
        return list(reader), reader.fieldnames or []


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def robust_torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def validate_protocol():
    if not PROTOCOL.exists():
        raise FileNotFoundError(PROTOCOL)
    actual = file_sha256(PROTOCOL)
    if actual.lower() != EXPECTED_PROTOCOL_SHA256.lower():
        raise RuntimeError(
            f"S05-C protocol SHA mismatch.\n"
            f"Expected: {EXPECTED_PROTOCOL_SHA256}\nActual: {actual}"
        )
    return actual


def import_training_helper():
    if not TRAINING_HELPER.exists():
        raise FileNotFoundError(TRAINING_HELPER)
    actual = file_sha256(TRAINING_HELPER)
    if actual.lower() != EXPECTED_TRAINING_HELPER_SHA256.lower():
        raise RuntimeError(
            f"S05-B FIX2 helper SHA mismatch.\n"
            f"Expected: {EXPECTED_TRAINING_HELPER_SHA256}\nActual: {actual}"
        )

    spec = importlib.util.spec_from_file_location(
        "s05b_fix2_helper", str(TRAINING_HELPER)
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to import S05-B FIX2 helper.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    for attr in ("build_deeplab", "image_to_tensor", "deeplab_logits"):
        if not hasattr(module, attr):
            raise RuntimeError(f"S05-B helper missing required function: {attr}")

    return module


def load_source_val_rows():
    if not MANIFEST.exists():
        raise FileNotFoundError(MANIFEST)
    actual = file_sha256(MANIFEST)
    if actual.lower() != EXPECTED_MANIFEST_SHA256.lower():
        raise RuntimeError("Frozen S01 manifest SHA mismatch.")

    rows, fields = read_csv(MANIFEST)
    required = {"sample_id", "image_relpath", "mask_relpath", "s01_role"}
    missing = sorted(required - set(fields))
    if missing:
        raise RuntimeError(f"Manifest missing fields: {missing}")

    selected = [r for r in rows if r["s01_role"] == "source_val"]
    if len(selected) != SOURCE_VAL_N:
        raise RuntimeError(
            f"source_val count mismatch: expected={SOURCE_VAL_N}, actual={len(selected)}"
        )

    assert all(r["s01_role"] == "source_val" for r in selected)
    return sorted(selected, key=lambda r: r["sample_id"])


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
                f"Expected: {CHECKPOINTS[seed]['sha256']}\nActual: {actual}"
            )
        checked[seed] = actual
    return checked


def load_model(helper, seed: int, device: torch.device):
    ck = robust_torch_load(CHECKPOINTS[seed]["path"])
    if int(ck.get("seed", -1)) != seed:
        raise RuntimeError(f"Checkpoint internal seed mismatch: {seed}")
    if ck.get("manifest_sha256") != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError(f"Checkpoint manifest SHA mismatch: {seed}")

    model, _ = helper.build_deeplab(pretrained_backbone=False)
    model.load_state_dict(ck["model_state_dict"], strict=True)
    model.to(device)
    return model


def is_singleton_unsafe_aspp_bn(name: str, module: nn.Module) -> bool:
    return (
        isinstance(module, nn.BatchNorm2d)
        and name.startswith("classifier.0.convs.4")
    )


def configure_singleton_safe_tent(model: nn.Module):
    """
    Ordinary BN -> current-sample statistics + trainable affine.
    ASPP pooled BN -> source running stats + frozen affine.
    Dropout -> eval mode during TENT.
    """
    model.requires_grad_(False)

    params = []
    trainable_bn_names = []
    unsafe_bn_names = []
    unsafe_bn_modules = []
    dropout_names = []
    dropout_modules = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Dropout):
            module.eval()
            dropout_names.append(name)
            dropout_modules.append(module)

        if not isinstance(module, nn.BatchNorm2d):
            continue

        if is_singleton_unsafe_aspp_bn(name, module):
            module.track_running_stats = True
            module.eval()
            if module.weight is not None:
                module.weight.requires_grad_(False)
            if module.bias is not None:
                module.bias.requires_grad_(False)
            unsafe_bn_names.append(name)
            unsafe_bn_modules.append(module)
            continue

        module.track_running_stats = False
        module.train()

        if module.weight is not None:
            module.weight.requires_grad_(True)
            params.append(module.weight)
            trainable_bn_names.append(f"{name}.weight")

        if module.bias is not None:
            module.bias.requires_grad_(True)
            params.append(module.bias)
            trainable_bn_names.append(f"{name}.bias")

    if not params:
        raise RuntimeError("No ordinary BN affine parameters found for TENT.")
    if not unsafe_bn_modules:
        raise RuntimeError("DeepLab ASPP singleton-unsafe BN was not found.")

    return (
        params,
        trainable_bn_names,
        unsafe_bn_names,
        unsafe_bn_modules,
        dropout_names,
        dropout_modules,
    )


def set_singleton_safe_tent_mode(
    model: nn.Module,
    unsafe_bn_modules,
    dropout_modules,
):
    model.train()

    # model.train() recursively re-enables both, so override them afterwards.
    for module in unsafe_bn_modules:
        module.eval()
    for module in dropout_modules:
        module.eval()


def snapshot_params(params):
    return [p.detach().clone() for p in params]


@torch.no_grad()
def restore_params(params, source_values):
    for p, src in zip(params, source_values):
        p.copy_(src)
        p.grad = None


def mean_binary_entropy(logits: torch.Tensor):
    p = torch.sigmoid(logits)
    return (F.softplus(logits) - p * logits).mean()


def source_and_tent_logits(
    helper,
    model,
    params,
    source_values,
    unsafe_bn_modules,
    dropout_modules,
    x,
):
    # Source-only uses the exact frozen checkpoint in eval mode.
    restore_params(params, source_values)
    model.eval()
    with torch.no_grad():
        z_source = helper.deeplab_logits(model, x).detach()

    # Reset and run one episodic TENT step.
    restore_params(params, source_values)
    set_singleton_safe_tent_mode(
        model,
        unsafe_bn_modules,
        dropout_modules,
    )

    optimizer = torch.optim.Adam(
        params,
        lr=TENT_LR,
        weight_decay=TENT_WEIGHT_DECAY,
    )
    optimizer.zero_grad(set_to_none=True)

    z_pre = helper.deeplab_logits(model, x)
    loss = mean_binary_entropy(z_pre)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    with torch.no_grad():
        z_tent = helper.deeplab_logits(model, x).detach()

    result = (
        z_source[0, 0].float().cpu().numpy(),
        z_tent[0, 0].float().cpu().numpy(),
        float(loss.detach().cpu()),
    )
    del optimizer, z_pre, loss, z_source, z_tent
    return result


def deterministic_noise_seed(sample_id: str, perturbation: str) -> int:
    payload = f"S03::{sample_id}::{perturbation}".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16)


def fold_for_sample(sample_id: str) -> int:
    payload = f"S03::{sample_id}".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16) % 5


def apply_perturbation(image: Image.Image, name: str, sample_id: str):
    im = image.convert("RGB")

    if name == "identity":
        return im
    if name == "brightness_070":
        return ImageEnhance.Brightness(im).enhance(0.70)
    if name == "brightness_130":
        return ImageEnhance.Brightness(im).enhance(1.30)
    if name == "contrast_065":
        return ImageEnhance.Contrast(im).enhance(0.65)
    if name == "saturation_050":
        return ImageEnhance.Color(im).enhance(0.50)

    if name in {"gamma_070", "gamma_150"}:
        gamma = 0.70 if name == "gamma_070" else 1.50
        arr = np.asarray(im, dtype=np.float32) / 255.0
        arr = np.power(arr, gamma)
        arr = np.clip(np.round(arr * 255.0), 0, 255).astype(np.uint8)
        return Image.fromarray(arr, mode="RGB")

    if name == "gaussian_blur_r15":
        return im.filter(ImageFilter.GaussianBlur(radius=1.5))

    if name == "gaussian_noise_s004":
        arr = np.asarray(im, dtype=np.float32) / 255.0
        rng = np.random.RandomState(
            deterministic_noise_seed(sample_id, name)
        )
        arr = np.clip(
            arr + rng.normal(0.0, 0.04, size=arr.shape),
            0.0,
            1.0,
        )
        arr = np.clip(np.round(arr * 255.0), 0, 255).astype(np.uint8)
        return Image.fromarray(arr, mode="RGB")

    if name == "downsample_050":
        w, h = im.size
        small = im.resize(
            (
                max(1, int(round(w * 0.5))),
                max(1, int(round(h * 0.5))),
            ),
            RESAMPLE_BILINEAR,
        )
        return small.resize((w, h), RESAMPLE_BILINEAR)

    raise RuntimeError(f"Unknown perturbation: {name}")


def image_to_model_tensor(helper, image: Image.Image):
    resized = image.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        RESAMPLE_BILINEAR,
    )
    return helper.image_to_tensor(resized).unsqueeze(0)


def load_gt(row):
    path = DATA_ROOT / Path(row["mask_relpath"])
    if not path.exists():
        raise FileNotFoundError(path)

    with Image.open(path) as ma:
        arr = np.asarray(ma.convert("L"), dtype=np.float32)

    maxv = float(arr.max())
    if maxv <= 0:
        raise RuntimeError(f"Empty source-val GT: {row['sample_id']}")

    return (arr > 0.5 * maxv).astype(np.uint8)


def resize_logit(z, shape_hw):
    t = torch.from_numpy(z.astype(np.float32))[None, None]
    return F.interpolate(
        t,
        size=tuple(shape_hw),
        mode="bilinear",
        align_corners=False,
    )[0, 0].numpy()


def dice_from_logit(z, gt):
    pred = resize_logit(z, gt.shape) >= 0.0
    gt_bool = gt.astype(bool)
    inter = np.logical_and(pred, gt_bool).sum()
    return float(
        (2.0 * inter + 1e-7)
        / (pred.sum() + gt_bool.sum() + 1e-7)
    )


def sigmoid_np(z):
    z = np.asarray(z, dtype=np.float32)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def entropy_np(p):
    p = np.clip(p, 1e-7, 1.0 - 1e-7)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


def boundary_density(mask):
    m = np.asarray(mask, dtype=bool)
    h = np.mean(m[:, 1:] != m[:, :-1]) if m.shape[1] > 1 else 0.0
    v = np.mean(m[1:, :] != m[:-1, :]) if m.shape[0] > 1 else 0.0
    return float(0.5 * (h + v))


def summarize_logit(z):
    p = sigmoid_np(z)
    h = entropy_np(p)
    m = p >= 0.5
    conf = 2.0 * np.abs(p - 0.5)

    return {
        "p": p,
        "m": m,
        "entropy_mean": float(h.mean()),
        "entropy_std": float(h.std()),
        "prob_mean": float(p.mean()),
        "prob_std": float(p.std()),
        "prob_q10": float(np.quantile(p, 0.10)),
        "prob_q50": float(np.quantile(p, 0.50)),
        "prob_q90": float(np.quantile(p, 0.90)),
        "fg_fraction": float(m.mean()),
        "boundary_density": boundary_density(m),
        "confidence_mean": float(conf.mean()),
    }


def extract_features(z_source, z_tent):
    s = summarize_logit(z_source)
    t = summarize_logit(z_tent)

    return {
        "source_entropy_mean": s["entropy_mean"],
        "source_entropy_std": s["entropy_std"],
        "source_prob_mean": s["prob_mean"],
        "source_prob_std": s["prob_std"],
        "source_prob_q10": s["prob_q10"],
        "source_prob_q50": s["prob_q50"],
        "source_prob_q90": s["prob_q90"],
        "source_fg_fraction": s["fg_fraction"],
        "source_boundary_density": s["boundary_density"],
        "source_confidence_mean": s["confidence_mean"],

        "tent_entropy_mean": t["entropy_mean"],
        "tent_entropy_std": t["entropy_std"],
        "tent_prob_mean": t["prob_mean"],
        "tent_prob_std": t["prob_std"],
        "tent_prob_q10": t["prob_q10"],
        "tent_prob_q50": t["prob_q50"],
        "tent_prob_q90": t["prob_q90"],
        "tent_fg_fraction": t["fg_fraction"],
        "tent_boundary_density": t["boundary_density"],
        "tent_confidence_mean": t["confidence_mean"],

        "entropy_mean_shift": t["entropy_mean"] - s["entropy_mean"],
        "prob_abs_change_mean": float(
            np.mean(np.abs(t["p"] - s["p"]))
        ),
        "mask_disagreement_fraction": float(
            np.mean(t["m"] != s["m"])
        ),
        "abs_fg_fraction_shift": abs(
            t["fg_fraction"] - s["fg_fraction"]
        ),
    }


def make_gate_model():
    return Pipeline([
        ("scale", StandardScaler()),
        (
            "clf",
            LogisticRegression(
                C=1.0,
                penalty="l2",
                solver="liblinear",
                class_weight="balanced",
                max_iter=5000,
                random_state=20260817,
            ),
        ),
    ])


def feature_matrix(rows):
    return np.asarray(
        [[float(r[f]) for f in FEATURE_NAMES] for r in rows],
        dtype=np.float64,
    )


def select_threshold(oof_rows):
    harmful_scores = np.asarray(
        [
            float(r["oof_harm_score"])
            for r in oof_rows
            if int(r["harmful"]) == 1
        ],
        dtype=np.float64,
    )

    if harmful_scores.size == 0:
        raise RuntimeError("No harmful source-side DeepLab cases.")

    all_scores = np.asarray(
        [float(r["oof_harm_score"]) for r in oof_rows],
        dtype=np.float64,
    )

    candidates = sorted(
        set([0.0, 1.0] + all_scores.tolist())
    )

    feasible = []
    for tau in candidates:
        recall = float(np.mean(harmful_scores >= tau))
        if recall + 1e-12 >= REQUIRED_HARM_RECALL:
            acceptance = float(np.mean(all_scores < tau))
            feasible.append((float(tau), recall, acceptance))

    if not feasible:
        return 0.0, 1.0, 0.0, True

    tau, recall, acceptance = max(feasible, key=lambda x: x[0])
    return tau, recall, acceptance, False


def export_gate_artifact(model, tau, metadata):
    scaler = model.named_steps["scale"]
    clf = model.named_steps["clf"]

    return {
        "artifact_version": "SafeTTA-v1-DeepLabV3-source-side-logistic",
        "feature_names": FEATURE_NAMES,
        "scaler_mean": scaler.mean_.astype(float).tolist(),
        "scaler_scale": scaler.scale_.astype(float).tolist(),
        "logistic_coef": clf.coef_[0].astype(float).tolist(),
        "logistic_intercept": float(clf.intercept_[0]),
        "harm_threshold_tau": float(tau),
        "decision_rule": (
            "reject_TENT_if_p_harm_greater_equal_tau_else_accept_TENT"
        ),
        **metadata,
    }


def generate_seed_cases(
    helper,
    seed,
    source_rows,
    device,
    seed_csv_path,
):
    model = load_model(helper, seed, device)

    (
        params,
        trainable_bn_names,
        unsafe_bn_names,
        unsafe_bn_modules,
        dropout_names,
        dropout_modules,
    ) = configure_singleton_safe_tent(model)

    source_values = snapshot_params(params)

    rows = []
    total = len(source_rows) * len(PERTURBATIONS)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    pbar = tqdm(
        total=total,
        desc=f"S05-C source gate seed {seed}",
        unit="case",
        dynamic_ncols=True,
    )

    for row in source_rows:
        image_path = DATA_ROOT / Path(row["image_relpath"])
        if not image_path.exists():
            raise FileNotFoundError(image_path)

        with Image.open(image_path) as im:
            native = im.convert("RGB")

        gt = load_gt(row)

        for perturbation in PERTURBATIONS:
            perturbed = apply_perturbation(
                native,
                perturbation,
                row["sample_id"],
            )
            x = image_to_model_tensor(helper, perturbed).to(
                device,
                non_blocking=True,
            )

            z_source, z_tent, adapt_loss = source_and_tent_logits(
                helper=helper,
                model=model,
                params=params,
                source_values=source_values,
                unsafe_bn_modules=unsafe_bn_modules,
                dropout_modules=dropout_modules,
                x=x,
            )

            source_dice = dice_from_logit(z_source, gt)
            tent_dice = dice_from_logit(z_tent, gt)
            delta = tent_dice - source_dice

            item = {
                "seed": seed,
                "sample_id": row["sample_id"],
                "fold": fold_for_sample(row["sample_id"]),
                "perturbation": perturbation,
                "source_dice": source_dice,
                "tent_dice": tent_dice,
                "delta_dice": delta,
                "harmful": int(delta <= HARM_THRESHOLD),
                "beneficial": int(delta >= BENEFIT_THRESHOLD),
                "tent_adapt_loss": adapt_loss,
            }
            item.update(extract_features(z_source, z_tent))
            rows.append(item)

            pbar.update(1)
            pbar.set_postfix(
                dDice=f"{delta:+.3f}",
                H=int(delta <= HARM_THRESHOLD),
            )

    pbar.close()

    fields = [
        "seed",
        "sample_id",
        "fold",
        "perturbation",
        "source_dice",
        "tent_dice",
        "delta_dice",
        "harmful",
        "beneficial",
        "tent_adapt_loss",
    ] + FEATURE_NAMES

    write_csv(seed_csv_path, rows, fields)

    meta = {
        "seed": seed,
        "rows": len(rows),
        "csv_sha256": file_sha256(seed_csv_path),
        "trainable_bn_affine_tensor_count": len(params),
        "trainable_bn_affine_names": trainable_bn_names,
        "singleton_unsafe_bn_names": unsafe_bn_names,
        "dropout_eval_names": dropout_names,
        "peak_gpu_memory_gb": (
            float(torch.cuda.max_memory_allocated() / 1024**3)
            if device.type == "cuda"
            else 0.0
        ),
    }

    del model, params, source_values, unsafe_bn_modules, dropout_modules
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return rows, meta


def validate_resumable_seed_csv(path, seed):
    if not path.exists():
        return None

    rows, fields = read_csv(path)
    required = {
        "seed", "sample_id", "fold", "perturbation",
        "source_dice", "tent_dice", "delta_dice",
        "harmful", "beneficial", "tent_adapt_loss",
    } | set(FEATURE_NAMES)

    if not required.issubset(fields):
        return None
    if len(rows) != SOURCE_VAL_N * len(PERTURBATIONS):
        return None
    if any(int(r["seed"]) != seed for r in rows):
        return None

    return rows


def run(args):
    seed_everything(20260817)

    protocol_sha = validate_protocol()
    helper = import_training_helper()
    source_rows = load_source_val_rows()
    checkpoint_hashes = validate_checkpoints()

    if args.output_dir.exists():
        raise FileExistsError(
            f"Final S05-C output already exists: {args.output_dir}"
        )

    build_dir = Path(str(args.output_dir) + "__building")
    build_dir.mkdir(parents=True, exist_ok=True)

    partial_dir = build_dir / "source_case_tables"
    partial_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available() and not args.cpu
        else "cpu"
    )

    print(f"[BUILD] {VERSION} | {BUILD}")
    print(f"Device: {device}")
    print(f"source_val={len(source_rows)}")
    print(f"perturbations={len(PERTURBATIONS)}")
    print(f"seeds={SEEDS}")
    print("seen_sanity opened=NO")
    print("unseen_locked opened=NO")
    print("target predictions opened=NO")
    print("target DeltaDice read=NO")
    print()

    all_rows = []
    seed_meta = {}

    for seed in SEEDS:
        seed_csv = (
            partial_dir / f"seed{seed}_source_gate_cases.csv"
        )

        resumed = False
        rows = None

        if args.resume:
            rows = validate_resumable_seed_csv(seed_csv, seed)
            resumed = rows is not None

        if resumed:
            print(
                f"[RESUME] seed={seed} verified source-only table "
                f"SHA256={file_sha256(seed_csv)}"
            )
            # BN metadata must still be recovered deterministically from model topology.
            model = load_model(helper, seed, device)
            (
                params,
                trainable_bn_names,
                unsafe_bn_names,
                unsafe_bn_modules,
                dropout_names,
                dropout_modules,
            ) = configure_singleton_safe_tent(model)

            seed_meta[str(seed)] = {
                "seed": seed,
                "rows": len(rows),
                "csv_sha256": file_sha256(seed_csv),
                "trainable_bn_affine_tensor_count": len(params),
                "trainable_bn_affine_names": trainable_bn_names,
                "singleton_unsafe_bn_names": unsafe_bn_names,
                "dropout_eval_names": dropout_names,
                "resumed": True,
            }

            del model, params, unsafe_bn_modules, dropout_modules
            if device.type == "cuda":
                torch.cuda.empty_cache()
        else:
            if seed_csv.exists():
                raise RuntimeError(
                    f"Existing invalid partial seed table: {seed_csv}\n"
                    "Do not overwrite it silently."
                )
            rows, meta = generate_seed_cases(
                helper=helper,
                seed=seed,
                source_rows=source_rows,
                device=device,
                seed_csv_path=seed_csv,
            )
            meta["resumed"] = False
            seed_meta[str(seed)] = meta

        all_rows.extend(rows)

    expected = SOURCE_VAL_N * len(PERTURBATIONS) * len(SEEDS)
    if len(all_rows) != expected:
        raise RuntimeError(
            f"Source case count mismatch: expected={expected}, actual={len(all_rows)}"
        )

    # Base-image grouping invariant.
    folds_by_sample = {}
    for row in all_rows:
        sid = row["sample_id"]
        fold = int(row["fold"])
        if sid in folds_by_sample and folds_by_sample[sid] != fold:
            raise RuntimeError(f"Base-image fold leakage: {sid}")
        folds_by_sample[sid] = fold

    fields = [
        "seed",
        "sample_id",
        "fold",
        "perturbation",
        "source_dice",
        "tent_dice",
        "delta_dice",
        "harmful",
        "beneficial",
        "tent_adapt_loss",
    ] + FEATURE_NAMES

    training_table = (
        build_dir / "deeplab_source_side_gate_training_table.csv"
    )
    write_csv(training_table, all_rows, fields)
    training_table_sha = file_sha256(training_table)

    # Five-fold base-image grouped OOF.
    oof_rows = []

    for fold in range(5):
        train_rows = [
            r for r in all_rows
            if int(r["fold"]) != fold
        ]
        test_rows = [
            r for r in all_rows
            if int(r["fold"]) == fold
        ]

        y_train = np.asarray(
            [int(r["harmful"]) for r in train_rows],
            dtype=np.int64,
        )
        y_test = np.asarray(
            [int(r["harmful"]) for r in test_rows],
            dtype=np.int64,
        )

        if len(np.unique(y_train)) != 2:
            raise RuntimeError(
                f"Fold {fold} train set lacks both harm classes."
            )
        if len(np.unique(y_test)) != 2:
            raise RuntimeError(
                f"Fold {fold} held-out set lacks both harm classes."
            )

        gate = make_gate_model()
        gate.fit(feature_matrix(train_rows), y_train)
        score = gate.predict_proba(
            feature_matrix(test_rows)
        )[:, 1]

        for row, s in zip(test_rows, score.tolist()):
            item = dict(row)
            item["oof_harm_score"] = float(s)
            oof_rows.append(item)

    if len(oof_rows) != len(all_rows):
        raise RuntimeError("OOF prediction count mismatch.")

    y_oof = np.asarray(
        [int(r["harmful"]) for r in oof_rows],
        dtype=np.int64,
    )
    s_oof = np.asarray(
        [float(r["oof_harm_score"]) for r in oof_rows],
        dtype=np.float64,
    )

    oof_auroc = float(roc_auc_score(y_oof, s_oof))
    oof_auprc = float(average_precision_score(y_oof, s_oof))

    tau, harm_recall, acceptance, degenerate = select_threshold(
        oof_rows
    )

    for row in oof_rows:
        accepted = int(float(row["oof_harm_score"]) < tau)
        row["tent_accepted"] = accepted
        row["selective_dice"] = (
            float(row["tent_dice"])
            if accepted
            else float(row["source_dice"])
        )

    harmful_rows = [
        r for r in oof_rows if int(r["harmful"]) == 1
    ]
    beneficial_rows = [
        r for r in oof_rows if int(r["beneficial"]) == 1
    ]
    neutral_rows = [
        r for r in oof_rows
        if int(r["harmful"]) == 0
        and int(r["beneficial"]) == 0
    ]

    benefit_retention = (
        float(np.mean([
            int(r["tent_accepted"])
            for r in beneficial_rows
        ]))
        if beneficial_rows
        else float("nan")
    )

    neutral_acceptance = (
        float(np.mean([
            int(r["tent_accepted"])
            for r in neutral_rows
        ]))
        if neutral_rows
        else float("nan")
    )

    source_mean = float(np.mean([
        float(r["source_dice"]) for r in oof_rows
    ]))
    tent_mean = float(np.mean([
        float(r["tent_dice"]) for r in oof_rows
    ]))
    selective_mean = float(np.mean([
        float(r["selective_dice"]) for r in oof_rows
    ]))

    write_csv(
        build_dir / "deeplab_source_side_oof_gate_predictions.csv",
        oof_rows,
        fields + [
            "oof_harm_score",
            "tent_accepted",
            "selective_dice",
        ],
    )

    # Final source-only fit.
    final_gate = make_gate_model()
    final_gate.fit(
        feature_matrix(all_rows),
        np.asarray(
            [int(r["harmful"]) for r in all_rows],
            dtype=np.int64,
        ),
    )

    unsafe_sets = [
        tuple(meta["singleton_unsafe_bn_names"])
        for meta in seed_meta.values()
    ]
    dropout_sets = [
        tuple(meta["dropout_eval_names"])
        for meta in seed_meta.values()
    ]

    if not all(x == unsafe_sets[0] for x in unsafe_sets):
        raise RuntimeError(
            "Singleton-unsafe BN topology differs across seeds."
        )
    if not all(x == dropout_sets[0] for x in dropout_sets):
        raise RuntimeError(
            "Dropout topology differs across seeds."
        )

    metadata = {
        "script_version": VERSION,
        "build": BUILD,
        "protocol_sha256": protocol_sha,
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "training_helper_sha256": EXPECTED_TRAINING_HELPER_SHA256,
        "checkpoint_sha256": checkpoint_hashes,
        "source_training_table_sha256": training_table_sha,
        "source_val_image_count": SOURCE_VAL_N,
        "perturbations": list(PERTURBATIONS),
        "source_case_count": len(all_rows),
        "source_harm_prevalence": float(y_oof.mean()),
        "source_oof_auroc": oof_auroc,
        "source_oof_auprc": oof_auprc,
        "required_source_harm_recall": REQUIRED_HARM_RECALL,
        "source_oof_harm_recall_at_tau": harm_recall,
        "source_oof_tent_acceptance_rate_at_tau": acceptance,
        "source_oof_benefit_retention_at_tau": benefit_retention,
        "source_oof_neutral_acceptance_at_tau": neutral_acceptance,
        "source_oof_source_mean_dice": source_mean,
        "source_oof_always_tent_mean_dice": tent_mean,
        "source_oof_selective_mean_dice": selective_mean,
        "singleton_unsafe_bn_names": list(unsafe_sets[0]),
        "singleton_unsafe_bn_rule": (
            "eval_source_running_stats_frozen_affine"
        ),
        "ordinary_bn_rule": (
            "current_sample_stats_trainable_affine"
        ),
        "dropout_eval_names": list(dropout_sets[0]),
        "dropout_rule": "eval_mode_during_tent",
        "seed_case_metadata": seed_meta,
        "degenerate_reject_all": degenerate,
        "seen_sanity_opened": False,
        "unseen_locked_opened": False,
        "target_predictions_opened": False,
        "target_delta_dice_read": False,
    }

    artifact = export_gate_artifact(
        final_gate,
        tau,
        metadata,
    )

    artifact_path = (
        build_dir
        / "SafeTTA_v1_DEEPLAB_SOURCE_LOCKED_GATE.json"
    )
    artifact_path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    artifact_sha = file_sha256(artifact_path)

    decision = (
        "S05_C_DEEPLAB_SOURCE_GATE_LOCKED_READY_FOR_TARGET_REPLICATION"
        if not degenerate
        else "S05_C_DEEPLAB_SOURCE_GATE_DEGENERATE_REVIEW_BEFORE_TARGET"
    )

    audit = {
        "gate_artifact_sha256": artifact_sha,
        "decision": decision,
        **metadata,
    }
    (build_dir / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = f"""===== S05-C DEEPLAB SOURCE-SIDE SAFETTA GATE BUILD =====
Script version: {VERSION}
Build: {BUILD}

Leakage control:
  source_val images used={SOURCE_VAL_N}
  seen_sanity opened=NO
  unseen_locked opened=NO
  target predictions opened=NO
  target DeltaDice read=NO

Frozen DeepLab checkpoints:
  seed20260817={checkpoint_hashes[20260817]}
  seed20260818={checkpoint_hashes[20260818]}
  seed20260819={checkpoint_hashes[20260819]}

Singleton-safe episodic TENT:
  ordinary BN=current-sample stats + affine adaptation
  singleton-unsafe ASPP BN=source running stats + frozen affine
  singleton-unsafe BN modules={list(unsafe_sets[0])}
  Dropout mode during TENT=eval
  Dropout modules={list(dropout_sets[0])}

Source-side supervision:
  perturbations={len(PERTURBATIONS)}
  seeds={len(SEEDS)}
  total cases={len(all_rows)}
  harm threshold={HARM_THRESHOLD:+.2f}
  harmful prevalence={float(y_oof.mean()):.6f}

5-fold base-image-grouped OOF gate:
  AUROC={oof_auroc:.6f}
  AUPRC={oof_auprc:.6f}

Source-only threshold:
  required harmful recall={REQUIRED_HARM_RECALL:.2f}
  tau={tau:.10f}
  OOF harmful recall={harm_recall:.6f}
  OOF TENT acceptance rate={acceptance:.6f}
  OOF beneficial retention={benefit_retention:.6f}
  OOF neutral acceptance={neutral_acceptance:.6f}

Source-side Dice diagnostics:
  Source-Only mean Dice={source_mean:.6f}
  Always-TENT mean Dice={tent_mean:.6f}
  Selective SafeTTA mean Dice={selective_mean:.6f}

Locked gate artifact:
  {artifact_path}
  SHA256={artifact_sha}

Decision: {decision}
"""

    (build_dir / "summary.txt").write_text(
        summary,
        encoding="utf-8",
    )

    # Atomic finalization.
    build_dir.rename(args.output_dir)

    print()
    print((args.output_dir / "summary.txt").read_text(encoding="utf-8"))
    print(f"[OK] Outputs: {args.output_dir}")


# --------------------------- self-test ---------------------------

class ToyASPP(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Identity(),
            nn.Identity(),
            nn.Identity(),
            nn.Identity(),
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(4, 4, 1, bias=False),
                nn.BatchNorm2d(4),
                nn.ReLU(),
            ),
        ])
        self.project = nn.Sequential(
            nn.Conv2d(4, 1, 1),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        y = self.convs[4](x)
        y = F.interpolate(
            y,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return self.project(y)


class ToyDeepLab(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone_bn = nn.BatchNorm2d(4)
        self.classifier = nn.Sequential(ToyASPP())

    def forward(self, x):
        return self.classifier(self.backbone_bn(x))


def self_test():
    # Feature parity.
    z = np.zeros((16, 16), dtype=np.float32)
    feat = extract_features(z, z)
    assert len(feat) == 24
    assert abs(feat["prob_abs_change_mean"]) < 1e-12
    assert abs(feat["mask_disagreement_fraction"]) < 1e-12

    # Frozen perturbation / grouping.
    assert len(PERTURBATIONS) == 10
    assert fold_for_sample("abc") == fold_for_sample("abc")
    assert (
        deterministic_noise_seed("abc", "gaussian_noise_s004")
        == deterministic_noise_seed("abc", "gaussian_noise_s004")
    )

    # Threshold.
    toy = []
    scores = [
        0.95, 0.85, 0.75, 0.65, 0.55,
        0.45, 0.35, 0.25, 0.15, 0.05,
    ]
    labels = [1,1,1,1,1,0,0,0,0,0]
    for score, label in zip(scores, labels):
        toy.append({
            "oof_harm_score": score,
            "harmful": label,
        })
    tau, recall, acceptance, degenerate = select_threshold(toy)
    assert not degenerate
    assert recall >= 0.80
    assert 0 <= tau <= 1
    assert 0 <= acceptance <= 1

    # Singleton-safe DeepLab-like topology.
    model = ToyDeepLab()
    (
        params,
        trainable_names,
        unsafe_names,
        unsafe_modules,
        dropout_names,
        dropout_modules,
    ) = configure_singleton_safe_tent(model)

    assert len(params) == 2  # backbone_bn weight + bias
    assert len(unsafe_names) == 1
    assert unsafe_names[0].startswith("classifier.0.convs.4")
    assert len(dropout_names) == 1

    set_singleton_safe_tent_mode(
        model,
        unsafe_modules,
        dropout_modules,
    )
    assert all(not m.training for m in unsafe_modules)
    assert all(not m.training for m in dropout_modules)

    x = torch.randn(1, 4, 8, 8)
    y = model(x)
    assert tuple(y.shape) == (1, 1, 8, 8)
    loss = mean_binary_entropy(y)
    assert torch.isfinite(loss)

    # The trainable BN remains train-mode and has enough spatial support.
    assert model.backbone_bn.training
    assert model.backbone_bn.track_running_stats is False

    assert HARM_THRESHOLD == -0.02
    assert BENEFIT_THRESHOLD == 0.02
    assert REQUIRED_HARM_RECALL == 0.80

    print("FEATURE_PARITY_TEST_PASS")
    print("PERTURBATION_PARITY_TEST_PASS")
    print("GROUP_FOLD_TEST_PASS")
    print("SOURCE_THRESHOLD_TEST_PASS")
    print("DEEPLAB_SINGLETON_UNSAFE_BN_DETECTION_PASS")
    print("DROPOUT_EVAL_GUARD_TEST_PASS")
    print("DEEPLAB_SINGLETON_TENT_FORWARD_PASS")
    print("FROZEN_S05C_RULES_TEST_PASS")
    print("SELF_TEST_PASS")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build DeepLabV3-specific SafeTTA gate using source_val only."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
    )
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume verified completed per-seed SOURCE-ONLY S05-C case tables "
            "after a technical interruption. No target data are involved."
        ),
    )
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
