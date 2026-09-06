#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1X_CM4A_source_oof_tent1_outcome_asset_lock_fix1.py

SafeTTA Q1 enhancement — CM4A.

Purpose
-------
Use the frozen CM3 patient-separated OOF prostate segmentation models to
construct the SOURCE-only counterfactual outcome asset:

    SOURCE vs A1_TENT_1STEP

for every Prostate158 axial slice and every frozen model family.

This stage:
- uses SOURCE GT (allowed);
- does NOT access PROMISE12;
- does NOT fit the safety predictor;
- does NOT perform threshold selection;
- does NOT tune the TTA action.

Frozen A1_TENT_1STEP lineage
----------------------------
Inherited from the main SafeTTA experiment:
- optimizer = Adam
- lr = 1e-3
- weight_decay = 0
- steps = 1
- episodic reset = YES
- normalization-affine parameters only
- source prediction from untouched checkpoint in eval mode

Architecture-compatible normalization:
- UNet: BatchNorm2d affine
- DeepLabV3-R50: ordinary BatchNorm2d affine; ASPP pooled singleton-unsafe
  BN stays eval with source running statistics; Dropout stays eval
- SegFormer-B0: LayerNorm + compatible BatchNorm2d affine; Dropout stays eval

Outcome:
- HARM    : DeltaDice <= -0.02
- NEUTRAL : -0.02 < DeltaDice < +0.02
- BENEFIT : DeltaDice >= +0.02

Sample unit remains the frozen axial 2D slice. Patient ID is retained for all
future grouped cross-validation/bootstrap.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


VERSION = "2026-09-05-Q1X-CM4A-v1-fix1"
BUILD = "Q1X_CM4A_SOURCE_OOF_TENT1_OUTCOME_ASSET_LOCK_FIX1"

ROOT = Path(r"F:\MEDSEG_SAFETTA")
OUT = ROOT / "outputs"
CODE = ROOT / "code"

CM3_DIR = (
    OUT
    / "Q1X_CM3_source_oof_segmentation_training_and_prediction_lock_fix3_v1"
)
CM3_LOCK = CM3_DIR / "CM3_SOURCE_OOF_SEGMENTATION_PANEL_LOCK.json"
EXPECTED_CM3_LOCK_SHA = (
    "4cfbe8f108f53e014973111d96244c666bfabd13eb022675fa31a34edaae7cec"
)

CM3_HELPER = (
    CODE
    / "Q1X_CM3_source_oof_segmentation_training_and_prediction_lock_fix3.py"
)
EXPECTED_CM3_HELPER_SHA = (
    "443e371c1a71173cc9fd3eaa4876cb8721a00497836e06d902b50da12cb9f57a"
)

DEFAULT_OUTPUT = (
    OUT
    / "Q1X_CM4A_source_oof_tent1_outcome_asset_lock_fix1_v1"
)

FAMILIES = ["UNet", "DeepLabV3_R50", "SegFormer_B0"]
N_FOLDS = 5
INPUT_SIZE = 352
PACKED_BYTES = (INPUT_SIZE * INPUT_SIZE + 7) // 8

TENT_LR = 1e-3
TENT_WEIGHT_DECAY = 0.0
TENT_STEPS = 1

HARM_THRESHOLD = -0.02
BENEFIT_THRESHOLD = +0.02

IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)

DECISION_PASS = (
    "SOURCE_TENT1_OUTCOME_ASSET_LOCKED_READY_FOR_CM4B_SAFETY_REPRESENTATION"
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


def save_json(path: Path, obj):
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def import_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except Exception as e:
        raise RuntimeError("PyTorch is required for CM4A.") from e
    return torch, nn, F


def import_cm3_helper():
    if not CM3_HELPER.is_file():
        raise FileNotFoundError(CM3_HELPER)

    got = sha256_file(CM3_HELPER)
    print(
        "CM3_HELPER",
        got,
        "PASS" if got == EXPECTED_CM3_HELPER_SHA else "FAIL",
    )
    if got != EXPECTED_CM3_HELPER_SHA:
        raise RuntimeError("CM3 helper SHA mismatch.")

    spec = importlib.util.spec_from_file_location(
        "q1x_cm3_fix3_helper",
        str(CM3_HELPER),
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to import CM3 helper.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    for name in ["build_model"]:
        if not hasattr(module, name):
            raise RuntimeError(f"CM3 helper missing: {name}")

    return module


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


def dice_binary(pred: np.ndarray, gt: np.ndarray):
    p = np.asarray(pred, dtype=bool)
    y = np.asarray(gt, dtype=bool)

    ps = int(p.sum())
    ys = int(y.sum())
    if ps == 0 and ys == 0:
        return 1.0

    inter = int(np.logical_and(p, y).sum())
    return float((2.0 * inter) / (ps + ys))


def binary_entropy_from_logits(logits):
    import torch
    import torch.nn.functional as F

    p = torch.sigmoid(logits)
    return (F.softplus(logits) - p * logits).mean()


def binary_entropy_scalar(logits):
    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        p = torch.sigmoid(logits)
        h = F.softplus(logits) - p * logits
        return float(h.mean().detach().cpu())


def boundary_density(mask):
    m = np.asarray(mask, dtype=bool)
    if m.shape != (INPUT_SIZE, INPUT_SIZE):
        raise RuntimeError(f"Unexpected mask shape {m.shape}")

    h = (
        float(np.mean(m[:, 1:] != m[:, :-1]))
        if m.shape[1] > 1 else 0.0
    )
    v = (
        float(np.mean(m[1:, :] != m[:-1, :]))
        if m.shape[0] > 1 else 0.0
    )
    return 0.5 * (h + v)


def pack_mask(mask):
    flat = np.asarray(mask, dtype=np.uint8).reshape(-1)
    packed = np.packbits(flat, bitorder="little")
    if packed.size != PACKED_BYTES:
        raise RuntimeError(
            f"Packed mask bytes mismatch {packed.size} != {PACKED_BYTES}"
        )
    return packed


def load_source_tensor(image_memmap, global_index, torch, device):
    image = np.asarray(
        image_memmap[int(global_index)],
        dtype=np.float32,
    )
    x = torch.from_numpy(image).unsqueeze(0).repeat(3, 1, 1)

    mean = torch.tensor(
        IMAGENET_MEAN,
        dtype=x.dtype,
    ).view(3, 1, 1)
    std = torch.tensor(
        IMAGENET_STD,
        dtype=x.dtype,
    ).view(3, 1, 1)

    x = (x - mean) / std
    return x.unsqueeze(0).to(device, non_blocking=True)


def is_dropout(module, nn):
    return isinstance(
        module,
        (nn.Dropout, nn.Dropout2d, nn.Dropout3d),
    )


def configure_tent(model, family, nn):
    """
    Freeze all non-normalization parameters, then expose only the
    architecture-compatible normalization affine parameters.
    """
    model.requires_grad_(False)

    params = []
    trainable_names = []
    unsafe_bn_modules = []
    unsafe_bn_names = []
    dropout_modules = []
    dropout_names = []
    ordinary_bn_modules = []
    layernorm_modules = []

    for name, module in model.named_modules():
        if is_dropout(module, nn):
            module.eval()
            dropout_modules.append(module)
            dropout_names.append(name)

        if isinstance(module, nn.BatchNorm2d):
            unsafe = (
                family == "DeepLabV3_R50"
                and "classifier.0.convs.4" in name
            )

            if unsafe:
                module.track_running_stats = True
                module.eval()
                if module.weight is not None:
                    module.weight.requires_grad_(False)
                if module.bias is not None:
                    module.bias.requires_grad_(False)
                unsafe_bn_modules.append(module)
                unsafe_bn_names.append(name)
                continue

            module.track_running_stats = False
            module.train()
            ordinary_bn_modules.append(module)

            if module.weight is not None:
                module.weight.requires_grad_(True)
                params.append(module.weight)
                trainable_names.append(f"{name}.weight")

            if module.bias is not None:
                module.bias.requires_grad_(True)
                params.append(module.bias)
                trainable_names.append(f"{name}.bias")

        elif family == "SegFormer_B0" and isinstance(module, nn.LayerNorm):
            layernorm_modules.append(module)

            if module.weight is not None:
                module.weight.requires_grad_(True)
                params.append(module.weight)
                trainable_names.append(f"{name}.weight")

            if module.bias is not None:
                module.bias.requires_grad_(True)
                params.append(module.bias)
                trainable_names.append(f"{name}.bias")

    if not params:
        raise RuntimeError(f"{family}: no TENT affine parameters found.")

    if family == "DeepLabV3_R50" and not unsafe_bn_modules:
        raise RuntimeError(
            "DeepLabV3-R50 singleton-unsafe ASPP pooled BN was not found."
        )

    if family == "SegFormer_B0" and not layernorm_modules:
        raise RuntimeError("SegFormer-B0 LayerNorm modules were not found.")

    return {
        "params": params,
        "trainable_names": trainable_names,
        "ordinary_bn_modules": ordinary_bn_modules,
        "unsafe_bn_modules": unsafe_bn_modules,
        "unsafe_bn_names": unsafe_bn_names,
        "dropout_modules": dropout_modules,
        "dropout_names": dropout_names,
        "layernorm_modules": layernorm_modules,
    }


def set_tent_mode(model, cfg):
    model.train()

    # model.train() recursively re-enables these, so force them back to eval.
    for module in cfg["unsafe_bn_modules"]:
        module.eval()
    for module in cfg["dropout_modules"]:
        module.eval()


def snapshot_params(params):
    return [p.detach().clone() for p in params]


def restore_params(params, source_values):
    import torch

    with torch.no_grad():
        for p, src in zip(params, source_values):
            p.copy_(src)
            p.grad = None


def build_loaded_model(
    helper,
    family,
    checkpoint_path,
    checkpoint_sha,
    torch,
    device,
):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    got = sha256_file(checkpoint_path)
    if got != checkpoint_sha:
        raise RuntimeError(
            f"{family}: checkpoint SHA mismatch\n"
            f"expected={checkpoint_sha}\nactual={got}"
        )

    model = helper.build_model(family)

    ck = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(ck["model"], strict=True)
    model.to(device)

    return model, ck


def verify_source_oof_reproduction(
    helper,
    family,
    fold,
    checkpoint_path,
    checkpoint_sha,
    fold_indices,
    image_memmap,
    stored_oof_masks,
    torch,
    device,
):
    print(
        f"\n===== SOURCE OOF REPRODUCTION {family} FOLD {fold} ====="
    )

    model, _ = build_loaded_model(
        helper,
        family,
        checkpoint_path,
        checkpoint_sha,
        torch,
        device,
    )
    model.eval()
    model.requires_grad_(False)

    mismatch_pixels = 0
    mismatch_slices = 0

    for gi in tqdm(
        fold_indices,
        desc=f"{family} F{fold} source replay",
        unit="slice",
        dynamic_ncols=True,
    ):
        x = load_source_tensor(
            image_memmap,
            gi,
            torch,
            device,
        )

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

        prob_f16 = (
            torch.sigmoid(logits)
            .float()
            .cpu()
            .numpy()[0, 0]
            .astype(np.float16)
        )
        pred = prob_f16 >= np.float16(0.5)
        ref = np.asarray(stored_oof_masks[int(gi)], dtype=bool)

        diff = int(np.count_nonzero(pred != ref))
        mismatch_pixels += diff
        mismatch_slices += int(diff > 0)

    print("source replay mismatch slices:", mismatch_slices)
    print("source replay mismatch pixels:", mismatch_pixels)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if mismatch_slices != 0 or mismatch_pixels != 0:
        raise RuntimeError(
            f"{family} fold{fold}: CM3 SOURCE OOF reproduction failed."
        )


def run_tent_fold(
    helper,
    family,
    fold,
    checkpoint_path,
    checkpoint_sha,
    fold_rows,
    image_memmap,
    gt_memmap,
    stored_oof_masks,
    torch,
    nn,
    device,
    fold_dir,
):
    fold_dir.mkdir(parents=True, exist_ok=True)

    csv_path = fold_dir / "source_tent1_outcomes.csv"
    packed_path = fold_dir / "tent1_masks_packbits.npy"
    sidecar_path = fold_dir / "COMPLETE.json"

    expected_rows = len(fold_rows)

    if sidecar_path.is_file() and csv_path.is_file() and packed_path.is_file():
        side = load_json(sidecar_path)
        valid = (
            side.get("status") == "PASS"
            and side.get("family") == family
            and int(side.get("fold", -1)) == fold
            and int(side.get("rows", -1)) == expected_rows
            and side.get("checkpoint_sha256") == checkpoint_sha
            and side.get("outcome_csv_sha256") == sha256_file(csv_path)
            and side.get("tent_masks_sha256") == sha256_file(packed_path)
        )
        if valid:
            print(
                f"{family} fold{fold}: REUSE COMPLETE rows={expected_rows}"
            )
            return side

    model, _ = build_loaded_model(
        helper,
        family,
        checkpoint_path,
        checkpoint_sha,
        torch,
        device,
    )

    cfg = configure_tent(model, family, nn)
    source_values = snapshot_params(cfg["params"])

    print(
        f"{family} fold{fold}: "
        f"TENT affine tensors={len(cfg['params'])}, "
        f"numel={sum(p.numel() for p in cfg['params'])}, "
        f"ordinary_bn={len(cfg['ordinary_bn_modules'])}, "
        f"layernorm={len(cfg['layernorm_modules'])}, "
        f"unsafe_bn={len(cfg['unsafe_bn_modules'])}, "
        f"dropout={len(cfg['dropout_modules'])}"
    )

    packed = np.lib.format.open_memmap(
        packed_path,
        mode="w+",
        dtype=np.uint8,
        shape=(expected_rows, PACKED_BYTES),
    )

    result_rows = []

    pbar = tqdm(
        enumerate(fold_rows.itertuples(index=False)),
        total=expected_rows,
        desc=f"{family} F{fold} TENT1",
        unit="slice",
        dynamic_ncols=True,
    )

    for local_i, row in pbar:
        gi = int(row.global_index)

        x = load_source_tensor(
            image_memmap,
            gi,
            torch,
            device,
        )
        gt = np.asarray(gt_memmap[gi], dtype=bool)
        source_mask = np.asarray(stored_oof_masks[gi], dtype=bool)

        restore_params(cfg["params"], source_values)
        set_tent_mode(model, cfg)

        optimizer = torch.optim.Adam(
            cfg["params"],
            lr=TENT_LR,
            weight_decay=TENT_WEIGHT_DECAY,
        )
        optimizer.zero_grad(set_to_none=True)

        # Exactly one entropy-minimization update.
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
            pre_loss = binary_entropy_from_logits(pre_logits)

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

        post_entropy = binary_entropy_scalar(post_logits)
        tent_mask = (
            post_logits.detach().float().cpu().numpy()[0, 0] >= 0.0
        )

        source_dice = dice_binary(source_mask, gt)
        tent_dice = dice_binary(tent_mask, gt)
        delta = tent_dice - source_dice

        harmful = int(delta <= HARM_THRESHOLD)
        beneficial = int(delta >= BENEFIT_THRESHOLD)
        neutral = int(not harmful and not beneficial)

        packed[local_i] = pack_mask(tent_mask)

        changed = float(np.mean(source_mask != tent_mask))

        result_rows.append({
            "family": family,
            "fold": fold,
            "global_index": gi,
            "case_key": row.case_key,
            "slice_index": int(row.slice_index),
            "source_gt_positive": int(row.source_gt_positive),
            "source_dice": source_dice,
            "tent1_dice": tent_dice,
            "delta_dice": delta,
            "harmful": harmful,
            "neutral": neutral,
            "beneficial": beneficial,
            "tent_pre_entropy": pre_entropy,
            "tent_post_entropy": post_entropy,
            "tent_entropy_change": post_entropy - pre_entropy,
            "source_fg_fraction": float(source_mask.mean()),
            "tent_fg_fraction": float(tent_mask.mean()),
            "source_boundary_density": boundary_density(source_mask),
            "tent_boundary_density": boundary_density(tent_mask),
            "source_tent_mask_disagreement": changed,
        })

        pbar.set_postfix(
            d=f"{delta:+.3f}",
            H=harmful,
            Hpre=f"{pre_entropy:.3f}",
            Hpost=f"{post_entropy:.3f}",
        )

        del optimizer, pre_logits, pre_loss, post_logits

    packed.flush()

    result_df = pd.DataFrame(result_rows)
    result_df.to_csv(csv_path, index=False)

    if len(result_df) != expected_rows:
        raise RuntimeError(
            f"{family} fold{fold}: row count mismatch."
        )

    if (
        result_df[["harmful", "neutral", "beneficial"]]
        .sum(axis=1)
        .ne(1)
        .any()
    ):
        raise RuntimeError(
            f"{family} fold{fold}: invalid tri-state outcome encoding."
        )

    side = {
        "status": "PASS",
        "version": VERSION,
        "family": family,
        "fold": fold,
        "rows": expected_rows,
        "patients": int(fold_rows["case_key"].nunique()),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "tent": {
            "action": "A1_TENT_1STEP",
            "optimizer": "Adam",
            "lr": TENT_LR,
            "weight_decay": TENT_WEIGHT_DECAY,
            "steps": TENT_STEPS,
            "episodic_reset": True,
            "trainable_affine_tensors": len(cfg["params"]),
            "trainable_affine_numel": int(
                sum(p.numel() for p in cfg["params"])
            ),
            "ordinary_bn_modules": len(cfg["ordinary_bn_modules"]),
            "layernorm_modules": len(cfg["layernorm_modules"]),
            "singleton_unsafe_bn_modules": len(cfg["unsafe_bn_modules"]),
            "dropout_modules_forced_eval": len(cfg["dropout_modules"]),
        },
        "outcome_csv": str(csv_path),
        "outcome_csv_sha256": sha256_file(csv_path),
        "tent_masks": str(packed_path),
        "tent_masks_sha256": sha256_file(packed_path),
        "harm_count": int(result_df["harmful"].sum()),
        "neutral_count": int(result_df["neutral"].sum()),
        "benefit_count": int(result_df["beneficial"].sum()),
        "mean_delta_dice": float(result_df["delta_dice"].mean()),
        "median_delta_dice": float(result_df["delta_dice"].median()),
        "promises12_access": False,
    }
    save_json(sidecar_path, side)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return side


def preflight_one_slice_per_family(
    helper,
    cm3,
    slice_df,
    image_memmap,
    gt_memmap,
    torch,
    nn,
    device,
    output_dir,
):
    print("\n===== CM4A TENT1 ONE-SLICE PREFLIGHT =====")
    rows = []

    for family in FAMILIES:
        fold = 0
        fold_complete = (
            CM3_DIR
            / "models"
            / family
            / f"fold{fold}"
            / "COMPLETE.json"
        )
        complete = load_json(fold_complete)
        checkpoint = Path(complete["best_checkpoint"])
        checkpoint_sha = complete["best_checkpoint_sha256"]

        oof_info = cm3["family_oof_artifacts"][family]
        oof_mask_path = Path(oof_info["oof_mask"])
        if sha256_file(oof_mask_path) != oof_info["oof_mask_sha256"]:
            raise RuntimeError(f"{family}: OOF mask SHA mismatch.")
        oof_masks = np.load(oof_mask_path, mmap_mode="r")

        candidate = slice_df[slice_df["fold"] == fold].iloc[0]
        gi = int(candidate["global_index"])

        # Exact source replay for one slice.
        source_model, _ = build_loaded_model(
            helper,
            family,
            checkpoint,
            checkpoint_sha,
            torch,
            device,
        )
        source_model.eval()
        source_model.requires_grad_(False)

        x = load_source_tensor(
            image_memmap,
            gi,
            torch,
            device,
        )
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
                z = source_model(x)

        p16 = (
            torch.sigmoid(z)
            .float()
            .cpu()
            .numpy()[0, 0]
            .astype(np.float16)
        )
        replay = p16 >= np.float16(0.5)
        ref = np.asarray(oof_masks[gi], dtype=bool)
        mismatch = int(np.count_nonzero(replay != ref))

        del source_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        if mismatch != 0:
            raise RuntimeError(
                f"{family}: one-slice SOURCE replay mismatch={mismatch}"
            )

        # One true episodic TENT1 update.
        model, _ = build_loaded_model(
            helper,
            family,
            checkpoint,
            checkpoint_sha,
            torch,
            device,
        )
        cfg = configure_tent(model, family, nn)
        source_values = snapshot_params(cfg["params"])
        restore_params(cfg["params"], source_values)
        set_tent_mode(model, cfg)

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
            loss = binary_entropy_from_logits(pre_logits)

        pre_h = float(loss.detach().cpu())
        loss.backward()
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

        post_h = binary_entropy_scalar(post_logits)

        tent_mask = (
            post_logits.detach().float().cpu().numpy()[0, 0] >= 0.0
        )
        gt = np.asarray(gt_memmap[gi], dtype=bool)
        source_dice = dice_binary(ref, gt)
        tent_dice = dice_binary(tent_mask, gt)

        row = {
            "family": family,
            "fold": fold,
            "global_index": gi,
            "source_replay_mismatch_pixels": mismatch,
            "tent_affine_tensors": len(cfg["params"]),
            "tent_affine_numel": int(
                sum(p.numel() for p in cfg["params"])
            ),
            "ordinary_bn": len(cfg["ordinary_bn_modules"]),
            "layernorm": len(cfg["layernorm_modules"]),
            "unsafe_bn": len(cfg["unsafe_bn_modules"]),
            "dropout_eval": len(cfg["dropout_modules"]),
            "pre_entropy": pre_h,
            "post_entropy": post_h,
            "source_dice": source_dice,
            "tent_dice": tent_dice,
            "delta_dice": tent_dice - source_dice,
        }
        rows.append(row)

        print(
            f"{family}: PASS "
            f"source_replay_mismatch=0 "
            f"affine_tensors={row['tent_affine_tensors']} "
            f"affine_numel={row['tent_affine_numel']} "
            f"Hpre={pre_h:.6f} "
            f"Hpost={post_h:.6f} "
            f"delta={row['delta_dice']:+.6f}"
        )

        del model, optimizer, pre_logits, post_logits, loss
        if device.type == "cuda":
            torch.cuda.empty_cache()

    preflight_path = output_dir / "CM4A_TENT1_PREFLIGHT.json"
    save_json(
        preflight_path,
        {
            "status": "PASS",
            "version": VERSION,
            "rows": rows,
            "tent_action": {
                "optimizer": "Adam",
                "lr": TENT_LR,
                "weight_decay": TENT_WEIGHT_DECAY,
                "steps": 1,
                "episodic_reset": True,
            },
            "promises12_access": False,
        },
    )

    return preflight_path


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
            "Run exact lineage checks and one true episodic TENT1 step "
            "per family, then stop before the full SOURCE outcome asset."
        ),
    )
    args = ap.parse_args()

    print("===== Q1X CM4A SOURCE OOF TENT1 OUTCOME ASSET =====")
    print("STATUS=SOURCE_ONLY_COUNTERFACTUAL_OUTCOME_CONSTRUCTION")
    print("CM3_LOCK_REQUIRED=", EXPECTED_CM3_LOCK_SHA)
    print("A1_TENT_1STEP optimizer=Adam lr=1e-3 wd=0 steps=1")
    print("EPISODIC_RESET=YES")
    print("HARM_THRESHOLD=-0.02")
    print("BENEFIT_THRESHOLD=+0.02")
    print("SAMPLE_UNIT=axial_2D_slice")
    print("PATIENT_GROUPING_RETAINED=YES")
    print("PROMISE12_ACCESS=NO")
    print("SAFETY_MODEL_FITTING=NO")
    print("THRESHOLD_SELECTION=NO")
    print("TARGET_TUNING=NO")

    if not CM3_LOCK.is_file():
        raise FileNotFoundError(CM3_LOCK)

    got_cm3 = sha256_file(CM3_LOCK)
    print(
        "\nCM3_LOCK",
        got_cm3,
        "PASS" if got_cm3 == EXPECTED_CM3_LOCK_SHA else "FAIL",
    )
    if got_cm3 != EXPECTED_CM3_LOCK_SHA:
        raise RuntimeError("CM3 lock SHA mismatch.")

    cm3 = load_json(CM3_LOCK)
    if cm3.get("status") != "PASS":
        raise RuntimeError("CM3 status changed.")
    if cm3.get("decision") != (
        "SOURCE_OOF_SEGMENTATION_PANEL_LOCKED_READY_FOR_CM4_SOURCE_TENT1_SAFETY_DEVELOPMENT"
    ):
        raise RuntimeError("Unexpected CM3 decision.")

    helper = import_cm3_helper()

    torch, nn, F = import_torch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n===== RUNTIME =====")
    print("torch:", torch.__version__)
    print("device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(device))

    if args.output_dir.exists():
        stage_dir = args.output_dir
        print("STAGE_DIR=RESUME", stage_dir)
    else:
        stage_dir = args.output_dir
        stage_dir.mkdir(parents=True, exist_ok=False)
        print("STAGE_DIR=NEW", stage_dir)

    # Exact SOURCE cache inherited from CM3.
    cache_meta_path = Path(cm3["source_cache_meta"])
    if sha256_file(cache_meta_path) != cm3["source_cache_meta_sha256"]:
        raise RuntimeError("CM3 source-cache-meta SHA mismatch.")

    cache_meta = load_json(cache_meta_path)
    cache_dir = cache_meta_path.parent
    image_path = cache_dir / "source_images_f16.npy"
    gt_path = cache_dir / "source_masks_u8.npy"
    slices_path = cache_dir / "source_slices.csv"

    for path, key in [
        (image_path, "images_sha256"),
        (gt_path, "masks_sha256"),
        (slices_path, "slices_sha256"),
    ]:
        if not path.is_file():
            raise FileNotFoundError(path)
        got = sha256_file(path)
        if got != cache_meta[key]:
            raise RuntimeError(
                f"SOURCE cache SHA mismatch: {path.name}"
            )

    image_memmap = np.load(image_path, mmap_mode="r")
    gt_memmap = np.load(gt_path, mmap_mode="r")
    slice_df = pd.read_csv(slices_path)

    if image_memmap.shape != (3553, INPUT_SIZE, INPUT_SIZE):
        raise RuntimeError(
            f"Unexpected SOURCE image cache shape {image_memmap.shape}"
        )
    if gt_memmap.shape != (3553, INPUT_SIZE, INPUT_SIZE):
        raise RuntimeError(
            f"Unexpected SOURCE GT cache shape {gt_memmap.shape}"
        )
    if len(slice_df) != 3553:
        raise RuntimeError(
            f"Unexpected SOURCE slice rows {len(slice_df)}"
        )
    if slice_df["case_key"].nunique() != 139:
        raise RuntimeError("Expected 139 SOURCE patients.")

    preflight_path = stage_dir / "CM4A_TENT1_PREFLIGHT.json"

    if not preflight_path.is_file():
        preflight_one_slice_per_family(
            helper,
            cm3,
            slice_df,
            image_memmap,
            gt_memmap,
            torch,
            nn,
            device,
            stage_dir,
        )
    else:
        preflight = load_json(preflight_path)
        if preflight.get("status") != "PASS":
            raise RuntimeError("Existing CM4A preflight is not PASS.")
        print("\nCM4A PREFLIGHT=REUSE PASS")

    print("CM4A_PREFLIGHT_SHA256=", sha256_file(preflight_path))

    if args.preflight_only:
        print("\n===== CM4A PREFLIGHT-ONLY FINAL =====")
        print("Decision= TENT1_IMPLEMENTATION_PREFLIGHT_PASS_READY_FOR_FULL_CM4A")
        print("PROMISE12 access: NO")
        print("PASS")
        return

    all_sidecars = []
    all_frames = []

    for family in FAMILIES:
        print(f"\n\n########## FAMILY {family} ##########")

        oof_info = cm3["family_oof_artifacts"][family]
        oof_mask_path = Path(oof_info["oof_mask"])
        if sha256_file(oof_mask_path) != oof_info["oof_mask_sha256"]:
            raise RuntimeError(f"{family}: CM3 OOF mask SHA mismatch.")

        stored_oof_masks = np.load(oof_mask_path, mmap_mode="r")
        if stored_oof_masks.shape != (
            3553,
            INPUT_SIZE,
            INPUT_SIZE,
        ):
            raise RuntimeError(
                f"{family}: unexpected OOF mask shape "
                f"{stored_oof_masks.shape}"
            )

        family_frames = []

        for fold in range(N_FOLDS):
            complete_path = (
                CM3_DIR
                / "models"
                / family
                / f"fold{fold}"
                / "COMPLETE.json"
            )
            if not complete_path.is_file():
                raise FileNotFoundError(complete_path)

            complete = load_json(complete_path)
            if complete.get("status") != "PASS":
                raise RuntimeError(
                    f"{family} fold{fold}: CM3 COMPLETE not PASS."
                )

            checkpoint_path = Path(complete["best_checkpoint"])
            checkpoint_sha = complete["best_checkpoint_sha256"]

            fold_rows = (
                slice_df[slice_df["fold"].astype(int) == fold]
                .copy()
                .sort_values("global_index")
                .reset_index(drop=True)
            )
            fold_indices = fold_rows["global_index"].to_numpy(
                dtype=np.int64
            )

            # A complete CM3 replay gate is deliberately performed before
            # producing counterfactual outcomes.
            replay_marker = (
                stage_dir
                / "source_replay"
                / family
                / f"fold{fold}.json"
            )
            replay_marker.parent.mkdir(parents=True, exist_ok=True)

            if replay_marker.is_file():
                replay = load_json(replay_marker)
                if (
                    replay.get("status") != "PASS"
                    or replay.get("checkpoint_sha256") != checkpoint_sha
                    or int(replay.get("rows", -1)) != len(fold_rows)
                ):
                    raise RuntimeError(
                        f"{family} fold{fold}: invalid replay marker."
                    )
                print(
                    f"{family} fold{fold}: SOURCE REPLAY=REUSE PASS"
                )
            else:
                verify_source_oof_reproduction(
                    helper,
                    family,
                    fold,
                    checkpoint_path,
                    checkpoint_sha,
                    fold_indices,
                    image_memmap,
                    stored_oof_masks,
                    torch,
                    device,
                )
                save_json(
                    replay_marker,
                    {
                        "status": "PASS",
                        "family": family,
                        "fold": fold,
                        "rows": len(fold_rows),
                        "checkpoint_sha256": checkpoint_sha,
                        "source_oof_mask_sha256": (
                            oof_info["oof_mask_sha256"]
                        ),
                        "mismatch_slices": 0,
                        "mismatch_pixels": 0,
                    },
                )

            fold_dir = (
                stage_dir
                / "families"
                / family
                / f"fold{fold}"
            )

            side = run_tent_fold(
                helper,
                family,
                fold,
                checkpoint_path,
                checkpoint_sha,
                fold_rows,
                image_memmap,
                gt_memmap,
                stored_oof_masks,
                torch,
                nn,
                device,
                fold_dir,
            )
            all_sidecars.append(side)

            df = pd.read_csv(side["outcome_csv"])
            family_frames.append(df)
            all_frames.append(df)

        family_df = pd.concat(
            family_frames,
            ignore_index=True,
        ).sort_values("global_index")

        if len(family_df) != 3553:
            raise RuntimeError(
                f"{family}: expected 3553 rows, got {len(family_df)}."
            )
        if family_df["global_index"].nunique() != 3553:
            raise RuntimeError(
                f"{family}: OOF global indices not one-to-one."
            )

        print(f"\n===== {family} SOURCE TENT1 SUMMARY =====")
        print("rows:", len(family_df))
        print("patients:", family_df["case_key"].nunique())
        print("HARM:", int(family_df["harmful"].sum()))
        print("NEUTRAL:", int(family_df["neutral"].sum()))
        print("BENEFIT:", int(family_df["beneficial"].sum()))
        print(
            "HARM prevalence:",
            float(family_df["harmful"].mean()),
        )
        print(
            "mean DeltaDice:",
            float(family_df["delta_dice"].mean()),
        )
        print(
            "median DeltaDice:",
            float(family_df["delta_dice"].median()),
        )

    final_df = pd.concat(
        all_frames,
        ignore_index=True,
    ).sort_values(
        ["family", "global_index"]
    ).reset_index(drop=True)

    if len(final_df) != 3553 * 3:
        raise RuntimeError(
            f"Final CM4A row count mismatch: {len(final_df)}"
        )

    if final_df.groupby("family")["global_index"].nunique().to_dict() != {
        "DeepLabV3_R50": 3553,
        "SegFormer_B0": 3553,
        "UNet": 3553,
    }:
        raise RuntimeError("Final family/global-index coverage failure.")

    final_path = stage_dir / "CM4A_SOURCE_TENT1_OUTCOME_TABLE.csv"
    final_df.to_csv(final_path, index=False)

    summary_rows = []
    for family, df in final_df.groupby("family", sort=True):
        summary_rows.append({
            "family": family,
            "rows": len(df),
            "patients": df["case_key"].nunique(),
            "harm_count": int(df["harmful"].sum()),
            "harm_prevalence": float(df["harmful"].mean()),
            "neutral_count": int(df["neutral"].sum()),
            "benefit_count": int(df["beneficial"].sum()),
            "benefit_prevalence": float(df["beneficial"].mean()),
            "mean_source_dice": float(df["source_dice"].mean()),
            "mean_tent1_dice": float(df["tent1_dice"].mean()),
            "mean_delta_dice": float(df["delta_dice"].mean()),
            "median_delta_dice": float(df["delta_dice"].median()),
            "mean_mask_disagreement": float(
                df["source_tent_mask_disagreement"].mean()
            ),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = stage_dir / "CM4A_SOURCE_TENT1_SUMMARY.csv"
    summary_df.to_csv(summary_path, index=False)

    lock = {
        "status": "PASS",
        "decision": DECISION_PASS,
        "version": VERSION,
        "build": BUILD,
        "cm3_lock_sha256": EXPECTED_CM3_LOCK_SHA,
        "cm3_helper_sha256": EXPECTED_CM3_HELPER_SHA,
        "source_patients": 139,
        "source_slices": 3553,
        "model_families": FAMILIES,
        "rows": int(len(final_df)),
        "sample_unit": "axial_2D_slice",
        "future_grouping_unit": "patient/case_key",
        "tent_action": {
            "name": "A1_TENT_1STEP",
            "optimizer": "Adam",
            "learning_rate": TENT_LR,
            "weight_decay": TENT_WEIGHT_DECAY,
            "steps": TENT_STEPS,
            "episodic_reset": True,
            "entropy": "binary foreground entropy over all pixels",
        },
        "outcome_rule": {
            "source_dice": "binary slice Dice; both-empty=1",
            "tent1_dice": "binary slice Dice; both-empty=1",
            "delta_dice": "tent1_dice - source_dice",
            "harm": "delta_dice <= -0.02",
            "neutral": "-0.02 < delta_dice < +0.02",
            "benefit": "delta_dice >= +0.02",
        },
        "information_boundary": {
            "source_gt": True,
            "promises12_access": False,
            "safety_model_fitting": False,
            "threshold_selection": False,
            "target_tuning": False,
        },
        "artifacts": {
            "outcome_table": str(final_path),
            "outcome_table_sha256": sha256_file(final_path),
            "summary": str(summary_path),
            "summary_sha256": sha256_file(summary_path),
            "tent1_preflight": str(preflight_path),
            "tent1_preflight_sha256": sha256_file(preflight_path),
        },
        "next_stage": (
            "CM4B_PRE_ADAPTATION_DINO_CONDITIONED_SAFETY_REPRESENTATION_AND_GROUPED_OOF"
        ),
    }

    lock_path = stage_dir / "CM4A_SOURCE_TENT1_OUTCOME_LOCK.json"
    save_json(lock_path, lock)

    print("\n===== CM4A FINAL =====")
    print(summary_df.to_string(index=False))
    print("TOTAL rows:", len(final_df))
    print("PROMISE12 access: NO")
    print("Safety model fitting: NO")
    print("Threshold selection: NO")
    print("Decision=", DECISION_PASS)
    print("LOCK=", lock_path)
    print("LOCK SHA256=", sha256_file(lock_path))
    print("PASS")


if __name__ == "__main__":
    main()
