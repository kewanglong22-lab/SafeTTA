#!/usr/bin/env python
# -*- coding: utf-8 -*-

r"""
S03-B: Build and lock SafeTTA-v1 using SOURCE-VAL supervision only.

This script:
- opens only source_val images/masks;
- creates a frozen deterministic source-side perturbation bank;
- computes Source-Only and episodic one-step TENT;
- creates source-side DeltaDice harm labels;
- extracts the same 24 unlabeled features as S02;
- produces 5-fold base-image-grouped OOF harm scores;
- selects a source-only harm threshold;
- fits the final StandardScaler + LogisticRegression;
- exports a deployment JSON artifact.

It does NOT open seen_sanity or unseen_locked images/masks.
It does NOT read S01-D1 target DeltaDice.
It does NOT evaluate on target data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import shutil
from collections import Counter
from pathlib import Path

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
    raise RuntimeError("scikit-learn is required for S03-B.") from exc


VERSION = "2026-08-17-S03-B-v1"
BUILD = "S03_B_SOURCE_VAL_PERTURBATION_HARM_GATE_LOCK"

ROOT = Path(r"F:\MEDSEG_SAFETTA")
DATA_ROOT = ROOT / "data" / "processed" / "S00_polyp_locked_v1"

PROTOCOL = ROOT / "docs" / "S03_A_source_side_safettta_development_protocol_v1.md"
EXPECTED_PROTOCOL_SHA256 = "4f94008a4335d22802b064eef00d23f2e60baaaaf739eed6b231dc252d365ed5"

MANIFEST = ROOT / "data" / "splits" / "S01_locked_protocol_manifest_v1.csv"
EXPECTED_MANIFEST_SHA256 = "afb4bc1175af004819538cdbb22ee7a10f1be48035945c7e0d50fd2bbe2e3dc7"

TRAINING_HELPER = ROOT / "code" / "S01_B_train_pranet_source_only_frozen_seeds_v1.py"
EXPECTED_TRAINING_HELPER_SHA256 = "2b2c9c1b75ff60bd087b403e43dc77468b31cbd54fb607481cc80ab47182b67a"

OUTPUT_DIR = ROOT / "outputs" / "S03_B_source_side_safettta_gate_v1"

SEEDS = (20260817, 20260818, 20260819)
IMAGE_SIZE = 352
SOURCE_VAL_N = 145
N_FOLDS = 5

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
    RESAMPLE_NEAREST = Image.Resampling.NEAREST
except AttributeError:
    RESAMPLE_BILINEAR = Image.BILINEAR
    RESAMPLE_NEAREST = Image.NEAREST


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


def import_training_helper():
    if not TRAINING_HELPER.exists():
        raise FileNotFoundError(TRAINING_HELPER)
    actual = file_sha256(TRAINING_HELPER)
    if actual.lower() != EXPECTED_TRAINING_HELPER_SHA256.lower():
        raise RuntimeError("Frozen training helper SHA mismatch.")

    spec = importlib.util.spec_from_file_location("s01b_helper", str(TRAINING_HELPER))
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import S01-B helper.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_protocol():
    if not PROTOCOL.exists():
        raise FileNotFoundError(PROTOCOL)
    actual = file_sha256(PROTOCOL)
    if actual.lower() != EXPECTED_PROTOCOL_SHA256.lower():
        raise RuntimeError(
            f"S03-A protocol SHA mismatch. expected={EXPECTED_PROTOCOL_SHA256} actual={actual}"
        )
    return actual


def load_source_val_rows():
    if file_sha256(MANIFEST).lower() != EXPECTED_MANIFEST_SHA256.lower():
        raise RuntimeError("Frozen manifest SHA mismatch.")
    rows, fields = read_csv(MANIFEST)
    required = {"sample_id", "image_relpath", "mask_relpath", "s01_role"}
    missing = sorted(required - set(fields))
    if missing:
        raise RuntimeError(f"Manifest missing columns: {missing}")

    selected = [r for r in rows if r["s01_role"] == "source_val"]
    if len(selected) != SOURCE_VAL_N:
        raise RuntimeError(
            f"source_val count mismatch: expected={SOURCE_VAL_N}, actual={len(selected)}"
        )

    # Hard safety assertion.
    assert all(r["s01_role"] == "source_val" for r in selected)
    return sorted(selected, key=lambda r: r["sample_id"])


def validate_checkpoints():
    for seed in SEEDS:
        path = CHECKPOINTS[seed]["path"]
        if not path.exists():
            raise FileNotFoundError(path)
        actual = file_sha256(path)
        if actual.lower() != CHECKPOINTS[seed]["sha256"].lower():
            raise RuntimeError(f"Checkpoint SHA mismatch seed={seed}")


def load_model(helper, seed, device):
    ck = robust_torch_load(CHECKPOINTS[seed]["path"])
    if int(ck.get("seed", -1)) != seed:
        raise RuntimeError(f"Checkpoint seed mismatch: {seed}")
    model = helper.PraNet(channel=32, pretrained_backbone_state=None)
    model.load_state_dict(ck["model_state_dict"], strict=True)
    model.to(device)
    return model


def final_logit(model, x):
    y = model(x)
    return y[-1] if isinstance(y, (tuple, list)) else y


def configure_tent(model):
    model.train()
    model.requires_grad_(False)
    params = []
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.track_running_stats = False
            if module.weight is not None:
                module.weight.requires_grad_(True)
                params.append(module.weight)
            if module.bias is not None:
                module.bias.requires_grad_(True)
                params.append(module.bias)
    if not params:
        raise RuntimeError("No BN affine parameters found for TENT.")
    return params


def snapshot_params(params):
    return [p.detach().clone() for p in params]


@torch.no_grad()
def restore_params(params, source_values):
    for p, src in zip(params, source_values):
        p.copy_(src)
        p.grad = None


def mean_binary_entropy(logits):
    p = torch.sigmoid(logits)
    return (F.softplus(logits) - p * logits).mean()


def source_and_tent_logits(model, params, source_values, x):
    restore_params(params, source_values)

    model.eval()
    with torch.no_grad():
        z_source = final_logit(model, x).detach()

    restore_params(params, source_values)
    model.train()

    optimizer = torch.optim.Adam(
        params,
        lr=TENT_LR,
        weight_decay=TENT_WEIGHT_DECAY,
    )
    optimizer.zero_grad(set_to_none=True)

    z_pre = final_logit(model, x)
    loss = mean_binary_entropy(z_pre)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    with torch.no_grad():
        z_tent = final_logit(model, x).detach()

    del optimizer, z_pre, loss
    return (
        z_source[0, 0].float().cpu().numpy(),
        z_tent[0, 0].float().cpu().numpy(),
    )


def deterministic_noise_seed(sample_id, perturbation):
    payload = f"S03::{sample_id}::{perturbation}".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16)


def apply_perturbation(image: Image.Image, name: str, sample_id: str) -> Image.Image:
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
    if name == "gamma_070" or name == "gamma_150":
        gamma = 0.70 if name == "gamma_070" else 1.50
        arr = np.asarray(im, dtype=np.float32) / 255.0
        arr = np.power(arr, gamma)
        arr = np.clip(np.round(arr * 255.0), 0, 255).astype(np.uint8)
        return Image.fromarray(arr, mode="RGB")
    if name == "gaussian_blur_r15":
        return im.filter(ImageFilter.GaussianBlur(radius=1.5))
    if name == "gaussian_noise_s004":
        arr = np.asarray(im, dtype=np.float32) / 255.0
        rng = np.random.RandomState(deterministic_noise_seed(sample_id, name))
        arr = np.clip(arr + rng.normal(0.0, 0.04, size=arr.shape), 0.0, 1.0)
        arr = np.clip(np.round(arr * 255.0), 0, 255).astype(np.uint8)
        return Image.fromarray(arr, mode="RGB")
    if name == "downsample_050":
        w, h = im.size
        small = im.resize(
            (max(1, int(round(w * 0.5))), max(1, int(round(h * 0.5)))),
            RESAMPLE_BILINEAR,
        )
        return small.resize((w, h), RESAMPLE_BILINEAR)

    raise RuntimeError(f"Unknown perturbation: {name}")


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
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


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
        "h": h,
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
        "prob_abs_change_mean": float(np.mean(np.abs(t["p"] - s["p"]))),
        "mask_disagreement_fraction": float(np.mean(t["m"] != s["m"])),
        "abs_fg_fraction_shift": abs(t["fg_fraction"] - s["fg_fraction"]),
    }


def load_gt(row):
    path = DATA_ROOT / Path(row["mask_relpath"])
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
    g = gt.astype(bool)
    inter = np.logical_and(pred, g).sum()
    return float((2.0 * inter + 1e-7) / (pred.sum() + g.sum() + 1e-7))


def image_to_model_tensor(helper, image):
    resized = image.resize((IMAGE_SIZE, IMAGE_SIZE), RESAMPLE_BILINEAR)
    return helper.image_to_tensor(resized).unsqueeze(0)


def fold_for_sample(sample_id):
    payload = f"S03::{sample_id}".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16) % N_FOLDS


def make_model():
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


def matrix(rows):
    return np.asarray(
        [[float(r[f]) for f in FEATURE_NAMES] for r in rows],
        dtype=np.float64,
    )


def select_threshold(oof_rows):
    harmful_scores = np.asarray(
        [float(r["oof_harm_score"]) for r in oof_rows if int(r["harmful"]) == 1],
        dtype=np.float64,
    )
    if harmful_scores.size == 0:
        raise RuntimeError("No source-side harmful cases; gate cannot be trained.")

    candidates = sorted(
        set([0.0, 1.0] + [float(r["oof_harm_score"]) for r in oof_rows])
    )

    feasible = []
    for tau in candidates:
        recall = float(np.mean(harmful_scores >= tau))
        if recall + 1e-12 >= REQUIRED_HARM_RECALL:
            acceptance = float(
                np.mean(
                    np.asarray(
                        [float(r["oof_harm_score"]) for r in oof_rows]
                    ) < tau
                )
            )
            feasible.append((tau, recall, acceptance))

    if not feasible:
        return 0.0, 1.0, 0.0, True

    tau, recall, acceptance = max(feasible, key=lambda x: x[0])
    return float(tau), float(recall), float(acceptance), False


def logistic_artifact(model, tau, metadata):
    scaler = model.named_steps["scale"]
    clf = model.named_steps["clf"]

    return {
        "artifact_version": "SafeTTA-v1-source-side-logistic",
        "feature_names": FEATURE_NAMES,
        "scaler_mean": scaler.mean_.astype(float).tolist(),
        "scaler_scale": scaler.scale_.astype(float).tolist(),
        "logistic_coef": clf.coef_[0].astype(float).tolist(),
        "logistic_intercept": float(clf.intercept_[0]),
        "harm_threshold_tau": float(tau),
        "decision_rule": "reject_TENT_if_p_harm_greater_equal_tau_else_accept_TENT",
        **metadata,
    }


def run(args):
    protocol_sha = validate_protocol()
    rows = load_source_val_rows()
    validate_checkpoints()
    helper = import_training_helper()

    if args.output_dir.exists():
        raise FileExistsError(f"Final S03-B output already exists: {args.output_dir}")

    build_dir = Path(str(args.output_dir) + "__building")
    if build_dir.exists():
        if not args.technical_rerun:
            raise FileExistsError(
                f"Partial S03-B output exists: {build_dir}. "
                "Use --technical-rerun only after a technical failure."
            )
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True, exist_ok=False)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )

    print(f"[BUILD] {VERSION} | {BUILD}")
    print(f"Device: {device}")
    print(f"Source-val images: {len(rows)}")
    print(f"Perturbations: {len(PERTURBATIONS)}")
    print(f"Frozen seeds: {SEEDS}")
    print("seen_sanity opened=NO")
    print("unseen_locked opened=NO")
    print("target DeltaDice read=NO")
    print()

    all_rows = []

    for seed in SEEDS:
        model = load_model(helper, seed, device)
        params = configure_tent(model)
        source_values = snapshot_params(params)

        total = len(rows) * len(PERTURBATIONS)
        pbar = tqdm(
            total=total,
            desc=f"S03 source supervision seed {seed}",
            unit="case",
            dynamic_ncols=True,
        )

        for row in rows:
            image_path = DATA_ROOT / Path(row["image_relpath"])
            with Image.open(image_path) as im:
                native = im.convert("RGB")

            gt = load_gt(row)

            for perturbation in PERTURBATIONS:
                perturbed = apply_perturbation(
                    native, perturbation, row["sample_id"]
                )
                x = image_to_model_tensor(helper, perturbed).to(
                    device, non_blocking=True
                )

                z_source, z_tent = source_and_tent_logits(
                    model, params, source_values, x
                )

                src_dice = dice_from_logit(z_source, gt)
                tent_dice = dice_from_logit(z_tent, gt)
                delta = tent_dice - src_dice
                harmful = int(delta <= HARM_THRESHOLD)
                beneficial = int(delta >= BENEFIT_THRESHOLD)

                feat = extract_features(z_source, z_tent)

                out_row = {
                    "seed": seed,
                    "sample_id": row["sample_id"],
                    "fold": fold_for_sample(row["sample_id"]),
                    "perturbation": perturbation,
                    "source_dice": src_dice,
                    "tent_dice": tent_dice,
                    "delta_dice": delta,
                    "harmful": harmful,
                    "beneficial": beneficial,
                }
                out_row.update(feat)
                all_rows.append(out_row)

                pbar.set_postfix(
                    dDice=f"{delta:+.3f}",
                    H=harmful,
                )
                pbar.update(1)

        pbar.close()
        del model, params, source_values
        if device.type == "cuda":
            torch.cuda.empty_cache()

    expected_rows = SOURCE_VAL_N * len(PERTURBATIONS) * len(SEEDS)
    if len(all_rows) != expected_rows:
        raise RuntimeError(
            f"Source supervision row mismatch: expected={expected_rows}, "
            f"actual={len(all_rows)}"
        )

    # Verify grouping invariant.
    folds_by_sample = {}
    for r in all_rows:
        sid = r["sample_id"]
        f = int(r["fold"])
        if sid in folds_by_sample and folds_by_sample[sid] != f:
            raise RuntimeError(f"Base-image fold leakage: {sid}")
        folds_by_sample[sid] = f

    fields = [
        "seed", "sample_id", "fold", "perturbation",
        "source_dice", "tent_dice", "delta_dice",
        "harmful", "beneficial",
    ] + FEATURE_NAMES

    table_path = build_dir / "source_side_gate_training_table.csv"
    write_csv(table_path, all_rows, fields)
    table_sha = file_sha256(table_path)

    # 5-fold OOF scores grouped by base image.
    oof_rows = []
    for fold in range(N_FOLDS):
        train = [r for r in all_rows if int(r["fold"]) != fold]
        test = [r for r in all_rows if int(r["fold"]) == fold]

        y_train = np.asarray([int(r["harmful"]) for r in train], dtype=np.int64)
        y_test = np.asarray([int(r["harmful"]) for r in test], dtype=np.int64)

        if len(np.unique(y_train)) != 2 or len(np.unique(y_test)) != 2:
            raise RuntimeError(
                f"Fold {fold} lacks both harm classes; source gate development invalid."
            )

        model = make_model()
        model.fit(matrix(train), y_train)
        score = model.predict_proba(matrix(test))[:, 1]

        for r, s in zip(test, score.tolist()):
            item = dict(r)
            item["oof_harm_score"] = float(s)
            oof_rows.append(item)

    if len(oof_rows) != len(all_rows):
        raise RuntimeError("OOF prediction count mismatch.")

    oof_rows = sorted(
        oof_rows,
        key=lambda r: (
            int(r["fold"]), r["sample_id"], int(r["seed"]), r["perturbation"]
        ),
    )

    y_oof = np.asarray([int(r["harmful"]) for r in oof_rows], dtype=np.int64)
    s_oof = np.asarray([float(r["oof_harm_score"]) for r in oof_rows])

    oof_auroc = float(roc_auc_score(y_oof, s_oof))
    oof_auprc = float(average_precision_score(y_oof, s_oof))

    tau, harm_recall, acceptance_rate, degenerate = select_threshold(oof_rows)

    # OOF deployment diagnostics using the frozen threshold.
    for r in oof_rows:
        r["tent_accepted"] = int(float(r["oof_harm_score"]) < tau)
        r["selective_dice"] = (
            float(r["tent_dice"])
            if int(r["tent_accepted"]) == 1
            else float(r["source_dice"])
        )

    harmful_rows = [r for r in oof_rows if int(r["harmful"]) == 1]
    beneficial_rows = [r for r in oof_rows if int(r["beneficial"]) == 1]
    neutral_rows = [
        r for r in oof_rows
        if int(r["harmful"]) == 0 and int(r["beneficial"]) == 0
    ]

    benefit_retention = (
        float(np.mean([int(r["tent_accepted"]) for r in beneficial_rows]))
        if beneficial_rows else float("nan")
    )
    neutral_acceptance = (
        float(np.mean([int(r["tent_accepted"]) for r in neutral_rows]))
        if neutral_rows else float("nan")
    )

    source_mean = float(np.mean([float(r["source_dice"]) for r in oof_rows]))
    tent_mean = float(np.mean([float(r["tent_dice"]) for r in oof_rows]))
    selective_mean = float(
        np.mean([float(r["selective_dice"]) for r in oof_rows])
    )

    write_csv(
        build_dir / "source_side_oof_gate_predictions.csv",
        oof_rows,
        fields + ["oof_harm_score", "tent_accepted", "selective_dice"],
    )

    # Final fit on all source-side examples.
    final_model = make_model()
    final_model.fit(
        matrix(all_rows),
        np.asarray([int(r["harmful"]) for r in all_rows], dtype=np.int64),
    )

    metadata = {
        "script_version": VERSION,
        "build": BUILD,
        "protocol_sha256": protocol_sha,
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "training_helper_sha256": EXPECTED_TRAINING_HELPER_SHA256,
        "checkpoint_sha256": {
            str(seed): CHECKPOINTS[seed]["sha256"] for seed in SEEDS
        },
        "source_training_table_sha256": table_sha,
        "source_val_image_count": SOURCE_VAL_N,
        "perturbations": list(PERTURBATIONS),
        "source_case_count": len(all_rows),
        "source_harm_prevalence": float(y_oof.mean()),
        "source_oof_auroc": oof_auroc,
        "source_oof_auprc": oof_auprc,
        "required_source_harm_recall": REQUIRED_HARM_RECALL,
        "source_oof_harm_recall_at_tau": harm_recall,
        "source_oof_tent_acceptance_rate_at_tau": acceptance_rate,
        "source_oof_benefit_retention_at_tau": benefit_retention,
        "source_oof_neutral_acceptance_at_tau": neutral_acceptance,
        "source_oof_source_mean_dice": source_mean,
        "source_oof_always_tent_mean_dice": tent_mean,
        "source_oof_selective_mean_dice": selective_mean,
        "degenerate_reject_all": degenerate,
        "target_images_opened": False,
        "target_masks_opened": False,
        "target_delta_dice_read": False,
    }

    artifact = logistic_artifact(final_model, tau, metadata)
    artifact_path = build_dir / "SafeTTA_v1_SOURCE_LOCKED_GATE.json"
    artifact_path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    artifact_sha = file_sha256(artifact_path)

    decision = (
        "S03_B_SOURCE_GATE_LOCKED_READY_FOR_S03_C_TARGET_APPLICATION"
        if not degenerate
        else "S03_B_GATE_DEGENERATE_REJECT_ALL_REVIEW_BEFORE_TARGET"
    )

    summary = f"""===== S03-B SOURCE-SIDE SAFETTA GATE BUILD =====
Script version: {VERSION}
Build: {BUILD}

Leakage control:
  source_val images used={SOURCE_VAL_N}
  seen_sanity opened=NO
  unseen_locked opened=NO
  target DeltaDice read=NO

Source-side supervision:
  perturbations={len(PERTURBATIONS)}
  seeds={len(SEEDS)}
  total image-seed-perturbation cases={len(all_rows)}
  harm threshold={HARM_THRESHOLD:+.2f}
  harmful prevalence={float(y_oof.mean()):.6f}

5-fold base-image-grouped OOF gate:
  AUROC={oof_auroc:.6f}
  AUPRC={oof_auprc:.6f}

Source-only threshold selection:
  required harmful recall={REQUIRED_HARM_RECALL:.2f}
  tau={tau:.10f}
  OOF harmful recall={harm_recall:.6f}
  OOF TENT acceptance rate={acceptance_rate:.6f}
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
    (build_dir / "summary.txt").write_text(summary, encoding="utf-8")

    audit = {
        "script_version": VERSION,
        "build": BUILD,
        "gate_artifact_sha256": artifact_sha,
        "decision": decision,
        **metadata,
    }
    (build_dir / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    build_dir.rename(args.output_dir)

    print()
    print((args.output_dir / "summary.txt").read_text(encoding="utf-8"))
    print(f"[OK] Outputs: {args.output_dir}")


def self_test():
    # Perturbation names frozen.
    assert len(PERTURBATIONS) == 10
    assert PERTURBATIONS[0] == "identity"
    assert "gaussian_noise_s004" in PERTURBATIONS

    # Deterministic noise.
    assert (
        deterministic_noise_seed("abc", "gaussian_noise_s004")
        == deterministic_noise_seed("abc", "gaussian_noise_s004")
    )

    # Feature math.
    z = np.zeros((16, 16), dtype=np.float32)
    feat = extract_features(z, z)
    assert len(feat) == 24
    assert abs(feat["mask_disagreement_fraction"]) < 1e-12
    assert abs(feat["prob_abs_change_mean"]) < 1e-12

    # Fold grouping.
    f = fold_for_sample("sample_x")
    assert 0 <= f < 5
    assert f == fold_for_sample("sample_x")

    # Threshold selector synthetic.
    toy = []
    scores = [0.95, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25, 0.15, 0.05]
    labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    for s, y in zip(scores, labels):
        toy.append({"oof_harm_score": s, "harmful": y})
    tau, recall, acceptance, deg = select_threshold(toy)
    assert not deg
    assert recall >= REQUIRED_HARM_RECALL - 1e-12
    assert 0.0 <= tau <= 1.0
    assert 0.0 <= acceptance <= 1.0

    # Manual logistic artifact dimensions.
    rng = np.random.RandomState(7)
    rows = []
    for i in range(100):
        r = {}
        y = i % 2
        for j, name in enumerate(FEATURE_NAMES):
            r[name] = float(y + rng.normal(0, 0.1))
        r["harmful"] = y
        rows.append(r)
    model = make_model()
    model.fit(matrix(rows), np.asarray([r["harmful"] for r in rows]))
    art = logistic_artifact(model, 0.5, {"self_test": True})
    assert len(art["scaler_mean"]) == 24
    assert len(art["scaler_scale"]) == 24
    assert len(art["logistic_coef"]) == 24

    assert HARM_THRESHOLD == -0.02
    assert BENEFIT_THRESHOLD == 0.02
    assert REQUIRED_HARM_RECALL == 0.80

    print("PERTURBATION_BANK_TEST_PASS")
    print("DETERMINISTIC_NOISE_TEST_PASS")
    print("FEATURE_PARITY_TEST_PASS")
    print("GROUP_FOLD_TEST_PASS")
    print("SOURCE_THRESHOLD_TEST_PASS")
    print("GATE_ARTIFACT_TEST_PASS")
    print("FROZEN_S03_RULES_TEST_PASS")
    print("SELF_TEST_PASS")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build SafeTTA-v1 gate using source_val supervision only."
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--technical-rerun",
        action="store_true",
        help="Only after a technical failure before final S03-B output exists.",
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
