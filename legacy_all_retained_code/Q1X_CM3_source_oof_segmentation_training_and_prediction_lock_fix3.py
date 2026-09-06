#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1X_CM3_source_oof_segmentation_training_and_prediction_lock_fix1.py

SafeTTA Q1 enhancement — CM3.

Scientific role
---------------
Train the frozen three-family SOURCE segmentation panel under the exact CM2B
protocol and generate patient-separated 5-fold out-of-fold (OOF) SOURCE
predictions for Prostate158.

Families:
- UNet
- DeepLabV3_R50
- SegFormer_B0

This script:
1) verifies the exact CM2B lock and all child artifacts;
2) builds a SOURCE-only preprocessed memmap cache;
3) runs a batch-size-8 forward/backward preflight for all three families;
4) trains 5 patient-level folds per family;
5) selects checkpoints by mean patient-level 3D whole-gland Dice;
6) writes OOF probabilities and binary SOURCE masks for every SOURCE slice;
7) writes fold/model metrics and a lineage lock.

This script does NOT:
- read PROMISE12 GT;
- run TTA;
- fit the SafeTTA safety model;
- tune on PROMISE12;
- train final all-source external models.

Resume behavior
---------------
The stage is resumable. Each fold writes a last-state checkpoint containing
optimizer/scaler/RNG state. Existing completed folds are reused only after
their completion metadata and best checkpoint are found.

Windows / RTX 4060 defaults are intentionally conservative:
- DataLoader num_workers=0
- AMP on CUDA
- frozen batch size=8
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


VERSION = "2026-09-04-Q1X-CM3-v1-fix3"
BUILD = "Q1X_CM3_SOURCE_OOF_SEGMENTATION_TRAINING_AND_PREDICTION_LOCK_FIX3"

ROOT = Path(r"F:\MEDSEG_SAFETTA\cross_modality_prostate_mri")
OUT = Path(r"F:\MEDSEG_SAFETTA\outputs")

CM2B_DIR = OUT / "Q1X_CM2B_prostate_mri_preprocessing_and_model_protocol_lock_fix1_v1"
CM2B_LOCK = CM2B_DIR / "CM2B_PROTOCOL_LOCK.json"
EXPECTED_CM2B_LOCK_SHA = (
    "9122400454853212f7cc9451be45789a052c5279141becdbfad2546c37b4019f"
)

DEFAULT_OUTPUT = (
    OUT
    / "Q1X_CM3_source_oof_segmentation_training_and_prediction_lock_fix3_v1"
)

# SOURCE preprocessing is scientifically identical across CM3 fix1/fix2/fix3.
# Reuse a prior PASS source-only cache read-only after exact metadata/hash
# verification to avoid another ~1.3 GB duplicate.
REUSABLE_SOURCE_CACHE_DIRS = [
    OUT
    / "Q1X_CM3_source_oof_segmentation_training_and_prediction_lock_fix2_v1"
    / "source_cache",
    OUT
    / "Q1X_CM3_source_oof_segmentation_training_and_prediction_lock_fix1_v1"
    / "source_cache",
]

TORCH_HOME = Path(r"F:\MEDSEG_SAFETTA\cache\torch")
HF_CACHE = Path(r"F:\MEDSEG_SAFETTA\cache\huggingface")

SEGFORMER_ENCODER_REPO = "nvidia/mit-b0"
SEGFORMER_SAFE_REVISION = "25ce79d97e6d9d509ed12e17cb2eb89b0a83a2dc"
SEGFORMER_SAFE_FILENAME = "model.safetensors"
SEGFORMER_SAFE_SHA256 = (
    "3e5ad9cd1dd8ecf8305c23fcdf01ef241f08c7b2dddacb6ec7de5a887188798a"
)

FAMILIES = ["UNet", "DeepLabV3_R50", "SegFormer_B0"]
N_FOLDS = 5

INPUT_SIZE = 352
BATCH_SIZE = 8
MAX_EPOCHS = 50
PATIENCE = 10
LR = 1e-4
WEIGHT_DECAY = 1e-4
MASK_THRESHOLD = 0.5

ROBUST_LOW_Q = 0.5
ROBUST_HIGH_Q = 99.5

IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)

BASE_SEED = 20260904

DECISION_PASS = (
    "SOURCE_OOF_SEGMENTATION_PANEL_LOCKED_READY_FOR_CM4_SOURCE_TENT1_SAFETY_DEVELOPMENT"
)


def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, obj: dict):
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def verify_child_artifact(lock: dict, key: str) -> Path:
    art = lock["artifacts"]
    path = Path(art[key])
    expected = art[f"{key}_sha256"]
    if not path.is_file():
        raise FileNotFoundError(path)
    got = sha256_file(path)
    print(f"{key} SHA256:", got, "PASS" if got == expected else "FAIL")
    if got != expected:
        raise RuntimeError(f"{key} SHA mismatch")
    return path


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def capture_rng_state() -> dict:
    import torch

    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict):
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def import_runtime():
    try:
        import nibabel as nib
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, Dataset
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms import functional as TF
        from tqdm import tqdm
    except Exception as e:
        raise RuntimeError(
            "CM3 requires nibabel, torch, torchvision, pandas, sklearn, tqdm."
        ) from e

    return nib, torch, nn, F, DataLoader, Dataset, InterpolationMode, TF, tqdm


def robust_volume_normalize(vol: np.ndarray) -> np.ndarray:
    x = np.asarray(vol, dtype=np.float32)
    valid = np.isfinite(x) & (x != 0)

    if not np.any(valid):
        return np.zeros_like(x, dtype=np.float32)

    vals = x[valid]
    lo = float(np.percentile(vals, ROBUST_LOW_Q))
    hi = float(np.percentile(vals, ROBUST_HIGH_Q))

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        raise RuntimeError(
            f"Invalid robust intensity range: q0.5={lo}, q99.5={hi}"
        )

    out = np.zeros_like(x, dtype=np.float32)
    clipped = np.clip(x[valid], lo, hi)
    out[valid] = (clipped - lo) / (hi - lo)
    out[~np.isfinite(out)] = 0
    return out


def center_crop_or_pad_stack(tensor, target_hw: int, mode: str):
    """
    tensor: [Z, 1, H, W]
    mode: image or mask
    """
    import torch.nn.functional as F

    z, c, h, w = tensor.shape

    if h > target_hw:
        top = (h - target_hw) // 2
        tensor = tensor[:, :, top:top + target_hw, :]
        h = target_hw
    if w > target_hw:
        left = (w - target_hw) // 2
        tensor = tensor[:, :, :, left:left + target_hw]
        w = target_hw

    pad_h = target_hw - h
    pad_w = target_hw - w

    if pad_h < 0 or pad_w < 0:
        raise RuntimeError("Unexpected negative pad after crop.")

    if pad_h or pad_w:
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        tensor = F.pad(
            tensor,
            (left, right, top, bottom),
            mode="constant",
            value=0.0,
        )

    return tensor


def try_reuse_source_cache(
    expected_total: int,
    target_spacing: Tuple[float, float],
):
    for cache_dir in REUSABLE_SOURCE_CACHE_DIRS:
        images_path = cache_dir / "source_images_f16.npy"
        masks_path = cache_dir / "source_masks_u8.npy"
        slices_path = cache_dir / "source_slices.csv"
        meta_path = cache_dir / "source_cache_meta.json"

        if not all(
            p.is_file()
            for p in [images_path, masks_path, slices_path, meta_path]
        ):
            continue

        try:
            meta = load_json(meta_path)
        except Exception:
            continue

        structural_ok = (
            meta.get("status") == "PASS"
            and meta.get("n_slices") == expected_total
            and meta.get("input_size") == INPUT_SIZE
            and meta.get("target_spacing_mm") == list(target_spacing)
            and meta.get("source_only") is True
            and meta.get("promises12_accessed") is False
        )
        if not structural_ok:
            continue

        sha_ok = (
            sha256_file(images_path) == meta.get("images_sha256")
            and sha256_file(masks_path) == meta.get("masks_sha256")
            and sha256_file(slices_path) == meta.get("slices_sha256")
        )
        if not sha_ok:
            continue

        print("\\nSOURCE CACHE=REUSE VERIFIED")
        print("cache dir:", cache_dir)
        print("cache slices:", expected_total)
        print("cache source-only: YES")
        print("cache PROMISE12 accessed: NO")

        return images_path, masks_path, slices_path, meta_path

    return None


def build_source_cache(
    nib,
    torch,
    F,
    source_df: pd.DataFrame,
    target_spacing: Tuple[float, float],
    cache_dir: Path,
):
    """
    Build:
      images.npy float16 [N, 352, 352], normalized to [0,1]
      masks.npy uint8 [N, 352, 352]
      slices.csv
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    images_path = cache_dir / "source_images_f16.npy"
    masks_path = cache_dir / "source_masks_u8.npy"
    slices_path = cache_dir / "source_slices.csv"
    meta_path = cache_dir / "source_cache_meta.json"

    expected_total = int(source_df["n_slices"].sum())

    reusable = try_reuse_source_cache(
        expected_total=expected_total,
        target_spacing=target_spacing,
    )
    if reusable is not None:
        return reusable

    if (
        images_path.is_file()
        and masks_path.is_file()
        and slices_path.is_file()
        and meta_path.is_file()
    ):
        meta = load_json(meta_path)
        if (
            meta.get("status") == "PASS"
            and meta.get("n_slices") == expected_total
            and meta.get("input_size") == INPUT_SIZE
            and meta.get("target_spacing_mm") == list(target_spacing)
        ):
            print("\nSOURCE CACHE=REUSE")
            print("cache slices:", expected_total)
            return images_path, masks_path, slices_path, meta_path

    print("\n===== BUILD SOURCE PREPROCESSED CACHE =====")
    print("total slices:", expected_total)

    for p in [images_path, masks_path, slices_path, meta_path]:
        if p.exists():
            p.unlink()

    image_mm = np.lib.format.open_memmap(
        images_path,
        mode="w+",
        dtype=np.float16,
        shape=(expected_total, INPUT_SIZE, INPUT_SIZE),
    )
    mask_mm = np.lib.format.open_memmap(
        masks_path,
        mode="w+",
        dtype=np.uint8,
        shape=(expected_total, INPUT_SIZE, INPUT_SIZE),
    )

    slice_rows = []
    cursor = 0

    from tqdm import tqdm

    for r in tqdm(
        source_df.itertuples(index=False),
        total=len(source_df),
        desc="preprocess source volumes",
    ):
        img = nib.load(r.t2_path)
        ann = nib.load(r.annotation_path)

        x = robust_volume_normalize(np.asanyarray(img.dataobj))
        y = (np.asanyarray(ann.dataobj) > 0).astype(np.float32)

        if x.shape != y.shape:
            raise RuntimeError(
                f"{r.case_key}: image/label shape mismatch in cache build."
            )

        zooms = tuple(float(v) for v in img.header.get_zooms()[:3])
        sx, sy = zooms[0], zooms[1]
        tx, ty = target_spacing

        new_w = max(1, int(round(x.shape[0] * sx / tx)))
        new_h = max(1, int(round(x.shape[1] * sy / ty)))

        # [X,Y,Z] -> [Z,1,Y,X]
        x_t = torch.from_numpy(np.transpose(x, (2, 1, 0))).unsqueeze(1)
        y_t = torch.from_numpy(np.transpose(y, (2, 1, 0))).unsqueeze(1)

        x_t = F.interpolate(
            x_t,
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
        )
        y_t = F.interpolate(
            y_t,
            size=(new_h, new_w),
            mode="nearest",
        )

        x_t = center_crop_or_pad_stack(x_t, INPUT_SIZE, "image")
        y_t = center_crop_or_pad_stack(y_t, INPUT_SIZE, "mask")

        x_np = x_t[:, 0].numpy().astype(np.float16, copy=False)
        y_np = (y_t[:, 0].numpy() >= 0.5).astype(np.uint8, copy=False)

        z_count = x_np.shape[0]
        if z_count != int(r.n_slices):
            raise RuntimeError(
                f"{r.case_key}: slice count changed unexpectedly "
                f"{z_count} vs {r.n_slices}"
            )

        end = cursor + z_count
        image_mm[cursor:end] = x_np
        mask_mm[cursor:end] = y_np

        for z in range(z_count):
            slice_rows.append({
                "global_index": cursor + z,
                "case_key": r.case_key,
                "fold": int(r.fold),
                "slice_index": z,
                "source_gt_positive": int(y_np[z].any()),
            })

        cursor = end

    if cursor != expected_total:
        raise RuntimeError(
            f"Source cache cursor mismatch: {cursor} vs {expected_total}"
        )

    image_mm.flush()
    mask_mm.flush()

    slice_df = pd.DataFrame(slice_rows)
    slice_df.to_csv(slices_path, index=False)

    meta = {
        "status": "PASS",
        "n_slices": expected_total,
        "input_size": INPUT_SIZE,
        "target_spacing_mm": list(target_spacing),
        "image_dtype": "float16",
        "mask_dtype": "uint8",
        "image_range": [0.0, 1.0],
        "source_only": True,
        "promises12_accessed": False,
        "images_sha256": sha256_file(images_path),
        "masks_sha256": sha256_file(masks_path),
        "slices_sha256": sha256_file(slices_path),
    }
    save_json(meta_path, meta)

    print("SOURCE CACHE PASS")
    print("images:", images_path)
    print("masks:", masks_path)
    print("slices:", slices_path)

    return images_path, masks_path, slices_path, meta_path


class JointAugment:
    def __init__(self, TF, InterpolationMode):
        self.TF = TF
        self.InterpolationMode = InterpolationMode

    def __call__(self, image, mask):
        import torch

        if torch.rand(()) < 0.5:
            image = self.TF.hflip(image)
            mask = self.TF.hflip(mask)

        angle = float(torch.empty(1).uniform_(-10.0, 10.0).item())
        scale = float(torch.empty(1).uniform_(0.9, 1.1).item())

        image = self.TF.affine(
            image,
            angle=angle,
            translate=[0, 0],
            scale=scale,
            shear=[0.0, 0.0],
            interpolation=self.InterpolationMode.BILINEAR,
            fill=0.0,
        )
        mask = self.TF.affine(
            mask,
            angle=angle,
            translate=[0, 0],
            scale=scale,
            shear=[0.0, 0.0],
            interpolation=self.InterpolationMode.NEAREST,
            fill=0.0,
        )

        return image, mask


def build_dataset_class(Dataset, TF, InterpolationMode):
    class SourceSliceDataset(Dataset):
        def __init__(
            self,
            images_path: Path,
            masks_path: Path,
            slice_df: pd.DataFrame,
            indices: np.ndarray,
            training: bool,
        ):
            self.images = np.load(images_path, mmap_mode="r")
            self.masks = np.load(masks_path, mmap_mode="r")
            self.slice_df = slice_df.reset_index(drop=True)
            self.indices = np.asarray(indices, dtype=np.int64)
            self.training = bool(training)
            self.augment = JointAugment(TF, InterpolationMode)

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, i):
            import torch

            idx = int(self.indices[i])
            image = np.asarray(self.images[idx], dtype=np.float32)
            mask = np.asarray(self.masks[idx], dtype=np.float32)

            image_t = torch.from_numpy(image).unsqueeze(0)
            mask_t = torch.from_numpy(mask).unsqueeze(0)

            if self.training:
                image_t, mask_t = self.augment(image_t, mask_t)

            image_t = image_t.repeat(3, 1, 1)

            mean = torch.tensor(
                IMAGENET_MEAN,
                dtype=image_t.dtype,
            ).view(3, 1, 1)
            std = torch.tensor(
                IMAGENET_STD,
                dtype=image_t.dtype,
            ).view(3, 1, 1)
            image_t = (image_t - mean) / std

            return {
                "image": image_t,
                "mask": mask_t,
                "global_index": idx,
                "case_key": str(self.slice_df.iloc[idx]["case_key"]),
                "slice_index": int(self.slice_df.iloc[idx]["slice_index"]),
            }

    return SourceSliceDataset


def build_unet(nn):
    class ConvBlock(nn.Module):
        def __init__(self, cin, cout):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.net(x)

    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            ch = [32, 64, 128, 256, 512]
            self.e1 = ConvBlock(3, ch[0])
            self.e2 = ConvBlock(ch[0], ch[1])
            self.e3 = ConvBlock(ch[1], ch[2])
            self.e4 = ConvBlock(ch[2], ch[3])
            self.b = ConvBlock(ch[3], ch[4])

            self.pool = nn.MaxPool2d(2)
            self.u4 = nn.ConvTranspose2d(ch[4], ch[3], 2, 2)
            self.d4 = ConvBlock(ch[3] + ch[3], ch[3])
            self.u3 = nn.ConvTranspose2d(ch[3], ch[2], 2, 2)
            self.d3 = ConvBlock(ch[2] + ch[2], ch[2])
            self.u2 = nn.ConvTranspose2d(ch[2], ch[1], 2, 2)
            self.d2 = ConvBlock(ch[1] + ch[1], ch[1])
            self.u1 = nn.ConvTranspose2d(ch[1], ch[0], 2, 2)
            self.d1 = ConvBlock(ch[0] + ch[0], ch[0])
            self.out = nn.Conv2d(ch[0], 1, 1)

        def forward(self, x):
            e1 = self.e1(x)
            e2 = self.e2(self.pool(e1))
            e3 = self.e3(self.pool(e2))
            e4 = self.e4(self.pool(e3))
            b = self.b(self.pool(e4))

            d4 = self.d4(torch.cat([self.u4(b), e4], dim=1))
            d3 = self.d3(torch.cat([self.u3(d4), e3], dim=1))
            d2 = self.d2(torch.cat([self.u2(d3), e2], dim=1))
            d1 = self.d1(torch.cat([self.u1(d2), e1], dim=1))
            return self.out(d1)

    import torch

    return UNet()


def remap_legacy_segformer_state_dict(state: dict) -> dict:
    """
    Convert legacy Hugging Face SegFormer checkpoint keys to the current
    Transformers SegFormer module names.

    Examples:
      segformer.encoder.block.0.0... -> segformer.stages.0.blocks.0...
      attention.self.query          -> attention.q_proj
      attention.output.dense        -> attention.o_proj

    The mapping follows the current official Transformers conversion table.
    Classification-head keys (classifier.*) are dropped because CM2B freezes
    an ImageNet-pretrained MiT-B0 encoder plus a newly initialized binary
    segmentation decode head.
    """
    import re

    mapped = {}

    for old_key, value in state.items():
        if old_key.startswith("classifier."):
            continue

        key = old_key

        # Encoder hierarchy.
        key = re.sub(
            r"segformer\.encoder\.patch_embeddings\.(\d+)\.",
            r"segformer.stages.\1.patch_embeddings.",
            key,
        )
        key = re.sub(
            r"segformer\.encoder\.block\.(\d+)\.",
            r"segformer.stages.\1.blocks.",
            key,
        )
        key = re.sub(
            r"segformer\.encoder\.layer_norm\.(\d+)",
            r"segformer.stages.\1.layer_norm",
            key,
        )

        # Attention.
        key = key.replace("attention.self.query", "attention.q_proj")
        key = key.replace("attention.self.key", "attention.k_proj")
        key = key.replace("attention.self.value", "attention.v_proj")
        key = key.replace(
            "attention.self.sr",
            "attention.sequence_reduction.sequence_reduction",
        )
        key = key.replace(
            "attention.self.layer_norm",
            "attention.sequence_reduction.layer_norm",
        )
        key = key.replace("attention.output.dense", "attention.o_proj")

        # MLP and LayerNorm.
        key = key.replace("mlp.dense1", "mlp.fc1")
        key = key.replace("mlp.dense2", "mlp.fc2")
        key = key.replace("layer_norm_1", "layernorm_before")
        key = key.replace("layer_norm_2", "layernorm_after")

        if key in mapped:
            raise RuntimeError(
                "SegFormer key-remap collision: "
                f"{old_key} -> {key}"
            )

        mapped[key] = value

    return mapped


def build_model(family: str):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_HOME.mkdir(parents=True, exist_ok=True)
    HF_CACHE.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(TORCH_HOME)
    os.environ.setdefault("HF_HOME", str(HF_CACHE))

    if family == "UNet":
        return build_unet(nn)

    if family == "DeepLabV3_R50":
        try:
            from torchvision.models import ResNet50_Weights
            from torchvision.models.segmentation import deeplabv3_resnet50
        except Exception as e:
            raise RuntimeError("torchvision DeepLabV3 is unavailable.") from e

        core = deeplabv3_resnet50(
            weights=None,
            weights_backbone=ResNet50_Weights.IMAGENET1K_V1,
            num_classes=1,
            aux_loss=False,
        )

        class Wrapper(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, x):
                return self.m(x)["out"]

        return Wrapper(core)

    if family == "SegFormer_B0":
        try:
            from huggingface_hub import hf_hub_download
            from safetensors.torch import load_file as safe_load_file
            from transformers import (
                SegformerConfig,
                SegformerForSemanticSegmentation,
            )
        except Exception as e:
            raise RuntimeError(
                "SegFormer-B0 requires transformers, huggingface_hub, "
                "and safetensors."
            ) from e

        safe_weight_path = Path(
            hf_hub_download(
                repo_id=SEGFORMER_ENCODER_REPO,
                filename=SEGFORMER_SAFE_FILENAME,
                revision=SEGFORMER_SAFE_REVISION,
                cache_dir=str(HF_CACHE),
            )
        )
        got_sha = sha256_file(safe_weight_path)
        if got_sha != SEGFORMER_SAFE_SHA256:
            raise RuntimeError(
                "SegFormer safe encoder SHA mismatch: "
                f"{got_sha} != {SEGFORMER_SAFE_SHA256}"
            )

        config = SegformerConfig.from_pretrained(
            SEGFORMER_ENCODER_REPO,
            revision=SEGFORMER_SAFE_REVISION,
            cache_dir=str(HF_CACHE),
        )
        config.num_labels = 1
        config.id2label = {0: "prostate"}
        config.label2id = {"prostate": 0}

        core = SegformerForSemanticSegmentation(config)

        legacy_state = safe_load_file(str(safe_weight_path), device="cpu")
        mapped_state = remap_legacy_segformer_state_dict(legacy_state)

        incompatible = core.load_state_dict(mapped_state, strict=False)

        bad_missing = [
            k for k in incompatible.missing_keys
            if not k.startswith("decode_head.")
        ]
        bad_unexpected = list(incompatible.unexpected_keys)

        if bad_missing or bad_unexpected:
            raise RuntimeError(
                "Unexpected SegFormer encoder load incompatibility after "
                "official legacy-key remap: "
                f"bad_missing={bad_missing[:20]}, "
                f"bad_unexpected={bad_unexpected[:20]}"
            )

        encoder_model_keys = [
            k for k in core.state_dict().keys()
            if k.startswith("segformer.")
        ]
        loaded_encoder_keys = [
            k for k in mapped_state.keys()
            if k.startswith("segformer.")
        ]

        missing_encoder = sorted(
            set(encoder_model_keys) - set(loaded_encoder_keys)
        )
        extra_encoder = sorted(
            set(loaded_encoder_keys) - set(encoder_model_keys)
        )
        if missing_encoder or extra_encoder:
            raise RuntimeError(
                "SegFormer encoder coverage gate failed: "
                f"missing={missing_encoder[:20]}, "
                f"extra={extra_encoder[:20]}"
            )

        print(
            "SegFormer_B0 legacy-key remap: PASS "
            f"legacy_keys={len(legacy_state)} "
            f"mapped_keys={len(mapped_state)} "
            f"encoder_keys={len(encoder_model_keys)}"
        )
        print(
            "SegFormer_B0 encoder coverage: PASS "
            f"{len(loaded_encoder_keys)}/{len(encoder_model_keys)}"
        )
        print(
            "SegFormer_B0 encoder load: PASS "
            f"repo={SEGFORMER_ENCODER_REPO} "
            f"revision={SEGFORMER_SAFE_REVISION[:8]} "
            f"safetensors_sha256={got_sha}"
        )
        print(
            "SegFormer_B0 initialization: "
            "ImageNet MiT-B0 encoder + random binary decode head"
        )

        class Wrapper(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, x):
                logits = self.m(pixel_values=x).logits
                return F.interpolate(
                    logits,
                    size=x.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

        return Wrapper(core)

    raise ValueError(f"Unknown family: {family}")


def soft_dice_loss(logits, target, eps=1e-6):
    import torch

    prob = torch.sigmoid(logits)
    dims = (1, 2, 3)
    inter = (prob * target).sum(dim=dims)
    den = prob.sum(dim=dims) + target.sum(dim=dims)
    dice = (2.0 * inter + eps) / (den + eps)
    return 1.0 - dice.mean()


def combined_loss(logits, target):
    import torch.nn.functional as F

    bce = F.binary_cross_entropy_with_logits(logits, target)
    dice = soft_dice_loss(logits, target)
    return 0.5 * bce + 0.5 * dice


def model_parameter_count(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return int(total), int(trainable)


def batch8_preflight(
    torch,
    DataLoader,
    DatasetClass,
    images_path,
    masks_path,
    slice_df,
    device,
):
    print("\n===== BATCH-8 FORWARD/BACKWARD PREFLIGHT =====")

    all_idx = np.arange(len(slice_df), dtype=np.int64)
    ds = DatasetClass(
        images_path,
        masks_path,
        slice_df,
        all_idx[:max(BATCH_SIZE, 16)],
        training=False,
    )
    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    batch = next(iter(loader))
    x = batch["image"].to(device, non_blocking=True)
    y = batch["mask"].to(device, non_blocking=True)

    results = {}

    for fi, family in enumerate(FAMILIES):
        seed = BASE_SEED + 9000 + fi
        set_global_seed(seed)

        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

        model = build_model(family).to(device)
        total, trainable = model_parameter_count(model)
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=LR,
            weight_decay=WEIGHT_DECAY,
        )

        try:
            opt.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                enabled=(device.type == "cuda"),
                dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
            ):
                logits = model(x)
                if logits.shape != y.shape:
                    raise RuntimeError(
                        f"{family}: output shape {tuple(logits.shape)} "
                        f"!= target {tuple(y.shape)}"
                    )
                loss = combined_loss(logits, y)

            loss.backward()
            opt.step()

            peak_mb = None
            if device.type == "cuda":
                torch.cuda.synchronize()
                peak_mb = (
                    torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                )

            print(
                f"{family}: PASS loss={float(loss.detach().cpu()):.6f} "
                f"params={total:,} trainable={trainable:,} "
                f"peak_cuda_MiB={peak_mb}"
            )

            results[family] = {
                "status": "PASS",
                "loss": float(loss.detach().cpu()),
                "params": total,
                "trainable": trainable,
                "peak_cuda_mib": peak_mb,
            }

        except RuntimeError as e:
            msg = str(e)
            if "out of memory" in msg.lower():
                print(f"{family}: CUDA OOM at frozen batch size {BATCH_SIZE}")
                raise RuntimeError(
                    f"STOP_BATCH8_PREFLIGHT_OOM::{family}"
                ) from e
            raise
        finally:
            del model, opt
            if device.type == "cuda":
                torch.cuda.empty_cache()

    return results


def patient_level_dice_from_arrays(
    probs: np.ndarray,
    masks: np.ndarray,
    rows: pd.DataFrame,
):
    pred = probs >= MASK_THRESHOLD
    truth = masks > 0

    per_patient = []
    for case_key, grp in rows.groupby("case_key", sort=True):
        idx = grp["local_position"].to_numpy(dtype=np.int64)

        p = pred[idx]
        y = truth[idx]

        tp = int(np.logical_and(p, y).sum())
        fp = int(np.logical_and(p, ~y).sum())
        fn = int(np.logical_and(~p, y).sum())

        den = 2 * tp + fp + fn
        dice = 1.0 if den == 0 else (2.0 * tp) / den

        per_patient.append({
            "case_key": case_key,
            "dice_3d": float(dice),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        })

    pdf = pd.DataFrame(per_patient)
    return float(pdf["dice_3d"].mean()), pdf


def run_validation(
    torch,
    DataLoader,
    model,
    ds,
    device,
    mask_memmap,
    slice_df,
):
    model.eval()

    probs = np.zeros(len(ds), dtype=np.float32)
    global_indices = np.zeros(len(ds), dtype=np.int64)
    losses = []

    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    pos = 0
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["mask"].to(device, non_blocking=True)

            with torch.autocast(
                device_type=device.type,
                enabled=(device.type == "cuda"),
                dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
            ):
                logits = model(x)
                loss = combined_loss(logits, y)

            p = torch.sigmoid(logits).float().cpu().numpy()[:, 0]
            gi = batch["global_index"].cpu().numpy().astype(np.int64)

            n = len(gi)
            probs[pos:pos+n] = p.reshape(n, -1).mean(axis=1) * 0.0
            # Replace scalar placeholder immediately with slice masks stored
            # separately below. We keep this branch only to ensure no accidental
            # scalar Dice is used.
            global_indices[pos:pos+n] = gi
            pos += n
            losses.append(float(loss.detach().cpu()))

    # Full-resolution probability inference is needed for 3D Dice.
    # Re-run validation in a memory-bounded way and store [N,H,W] float16.
    prob_masks = np.empty(
        (len(ds), INPUT_SIZE, INPUT_SIZE),
        dtype=np.float16,
    )
    pos = 0
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                enabled=(device.type == "cuda"),
                dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
            ):
                logits = model(x)
            p = torch.sigmoid(logits).float().cpu().numpy()[:, 0]
            n = p.shape[0]
            prob_masks[pos:pos+n] = p.astype(np.float16)
            pos += n

    global_indices = ds.indices.copy()
    truth_masks = np.asarray(mask_memmap[global_indices], dtype=np.uint8)

    rows = slice_df.iloc[global_indices].copy().reset_index(drop=True)
    rows["local_position"] = np.arange(len(rows), dtype=np.int64)

    mean_dice, patient_df = patient_level_dice_from_arrays(
        prob_masks.astype(np.float32),
        truth_masks,
        rows,
    )

    return {
        "mean_patient_dice": mean_dice,
        "mean_loss": float(np.mean(losses)) if losses else float("nan"),
        "patient_df": patient_df,
        "prob_masks": prob_masks,
        "global_indices": global_indices,
    }


def train_one_fold(
    torch,
    DataLoader,
    DatasetClass,
    family: str,
    fold: int,
    images_path: Path,
    masks_path: Path,
    slice_df: pd.DataFrame,
    device,
    stage_dir: Path,
):
    fold_dir = stage_dir / "models" / family / f"fold{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    best_path = fold_dir / "best.pt"
    last_path = fold_dir / "last.pt"
    history_path = fold_dir / "history.csv"
    done_path = fold_dir / "COMPLETE.json"

    if done_path.is_file() and best_path.is_file():
        done = load_json(done_path)
        if done.get("status") == "PASS":
            print(
                f"{family} fold{fold}: REUSE COMPLETE "
                f"best_dice={done['best_mean_patient_dice']:.6f}"
            )
            return done

    train_idx = slice_df.index[slice_df["fold"] != fold].to_numpy(dtype=np.int64)
    val_idx = slice_df.index[slice_df["fold"] == fold].to_numpy(dtype=np.int64)

    train_ds = DatasetClass(
        images_path,
        masks_path,
        slice_df,
        train_idx,
        training=True,
    )
    val_ds = DatasetClass(
        images_path,
        masks_path,
        slice_df,
        val_idx,
        training=False,
    )

    generator = torch.Generator()
    seed = BASE_SEED + FAMILIES.index(family) * 100 + fold
    generator.manual_seed(seed)
    set_global_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model = build_model(family).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(device.type == "cuda"),
    )

    start_epoch = 1
    best_dice = -1.0
    best_epoch = -1
    bad_epochs = 0
    history = []

    if last_path.is_file():
        ckpt = torch.load(
            last_path,
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_dice = float(ckpt["best_dice"])
        best_epoch = int(ckpt["best_epoch"])
        bad_epochs = int(ckpt["bad_epochs"])
        history = list(ckpt.get("history", []))
        restore_rng_state(ckpt["rng_state"])
        print(
            f"{family} fold{fold}: RESUME epoch={start_epoch} "
            f"best={best_dice:.6f}"
        )

    mask_memmap = np.load(masks_path, mmap_mode="r")

    from tqdm import tqdm

    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        model.train()
        running = 0.0
        n_seen = 0

        pbar = tqdm(
            train_loader,
            desc=f"{family} F{fold} E{epoch:02d}",
            leave=False,
        )

        for batch in pbar:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["mask"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                enabled=(device.type == "cuda"),
                dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
            ):
                logits = model(x)
                loss = combined_loss(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = x.shape[0]
            running += float(loss.detach().cpu()) * bs
            n_seen += bs
            pbar.set_postfix(loss=f"{float(loss.detach().cpu()):.4f}")

        train_loss = running / max(1, n_seen)

        val = run_validation(
            torch,
            DataLoader,
            model,
            val_ds,
            device,
            mask_memmap,
            slice_df,
        )
        val_dice = float(val["mean_patient_dice"])
        val_loss = float(val["mean_loss"])

        improved = val_dice > best_dice + 1e-8
        if improved:
            best_dice = val_dice
            best_epoch = epoch
            bad_epochs = 0
            torch.save(
                {
                    "version": VERSION,
                    "family": family,
                    "fold": fold,
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "best_mean_patient_dice": best_dice,
                    "seed": seed,
                },
                best_path,
            )
        else:
            bad_epochs += 1

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_mean_patient_dice": val_dice,
            "best_mean_patient_dice": best_dice,
            "bad_epochs": bad_epochs,
        })

        pd.DataFrame(history).to_csv(history_path, index=False)

        torch.save(
            {
                "version": VERSION,
                "family": family,
                "fold": fold,
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "best_dice": best_dice,
                "best_epoch": best_epoch,
                "bad_epochs": bad_epochs,
                "history": history,
                "rng_state": capture_rng_state(),
                "seed": seed,
            },
            last_path,
        )

        print(
            f"{family} fold{fold} epoch {epoch:02d}: "
            f"train_loss={train_loss:.5f} "
            f"val_loss={val_loss:.5f} "
            f"val_patient_dice={val_dice:.6f} "
            f"best={best_dice:.6f}@{best_epoch}"
        )

        if bad_epochs >= PATIENCE:
            print(
                f"{family} fold{fold}: EARLY STOP "
                f"after {bad_epochs} non-improving epochs."
            )
            break

    if not best_path.is_file():
        raise RuntimeError(f"{family} fold{fold}: best checkpoint missing.")

    best_ckpt = torch.load(
        best_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(best_ckpt["model"])

    final_val = run_validation(
        torch,
        DataLoader,
        model,
        val_ds,
        device,
        mask_memmap,
        slice_df,
    )

    patient_path = fold_dir / "best_patient_metrics.csv"
    final_val["patient_df"].to_csv(patient_path, index=False)

    oof_prob_path = fold_dir / "best_oof_prob_f16.npy"
    oof_bin_path = fold_dir / "best_oof_mask_u8.npy"
    oof_index_path = fold_dir / "best_oof_index.csv"

    np.save(oof_prob_path, final_val["prob_masks"].astype(np.float16))
    np.save(
        oof_bin_path,
        (final_val["prob_masks"] >= MASK_THRESHOLD).astype(np.uint8),
    )

    index_df = slice_df.iloc[
        final_val["global_indices"]
    ][["global_index", "case_key", "fold", "slice_index"]].copy()
    index_df.to_csv(oof_index_path, index=False)

    total_params, trainable_params = model_parameter_count(model)

    done = {
        "status": "PASS",
        "version": VERSION,
        "family": family,
        "fold": fold,
        "seed": seed,
        "best_epoch": int(best_ckpt["epoch"]),
        "best_mean_patient_dice": float(final_val["mean_patient_dice"]),
        "val_patient_count": int(
            slice_df.iloc[val_idx]["case_key"].nunique()
        ),
        "val_slice_count": int(len(val_idx)),
        "params": total_params,
        "trainable_params": trainable_params,
        "best_checkpoint": str(best_path),
        "best_checkpoint_sha256": sha256_file(best_path),
        "oof_prob": str(oof_prob_path),
        "oof_prob_sha256": sha256_file(oof_prob_path),
        "oof_mask": str(oof_bin_path),
        "oof_mask_sha256": sha256_file(oof_bin_path),
        "oof_index": str(oof_index_path),
        "oof_index_sha256": sha256_file(oof_index_path),
        "patient_metrics": str(patient_path),
        "patient_metrics_sha256": sha256_file(patient_path),
    }
    save_json(done_path, done)

    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return done


def assemble_family_oof(
    family: str,
    stage_dir: Path,
    slice_df: pd.DataFrame,
):
    print(f"\n===== ASSEMBLE OOF {family} =====")

    n = len(slice_df)
    prob_mm_path = stage_dir / f"CM3_{family}_OOF_PROB_F16.npy"
    mask_mm_path = stage_dir / f"CM3_{family}_OOF_MASK_U8.npy"

    prob_mm = np.lib.format.open_memmap(
        prob_mm_path,
        mode="w+",
        dtype=np.float16,
        shape=(n, INPUT_SIZE, INPUT_SIZE),
    )
    mask_mm = np.lib.format.open_memmap(
        mask_mm_path,
        mode="w+",
        dtype=np.uint8,
        shape=(n, INPUT_SIZE, INPUT_SIZE),
    )

    filled = np.zeros(n, dtype=np.uint8)

    fold_summaries = []

    for fold in range(N_FOLDS):
        fold_dir = stage_dir / "models" / family / f"fold{fold}"
        done = load_json(fold_dir / "COMPLETE.json")
        idx_df = pd.read_csv(done["oof_index"])
        probs = np.load(done["oof_prob"], mmap_mode="r")
        masks = np.load(done["oof_mask"], mmap_mode="r")

        gi = idx_df["global_index"].to_numpy(dtype=np.int64)

        if len(gi) != len(probs):
            raise RuntimeError(
                f"{family} fold{fold}: OOF index/prob length mismatch."
            )
        if np.any(filled[gi] != 0):
            raise RuntimeError(
                f"{family} fold{fold}: duplicate OOF global indices."
            )

        prob_mm[gi] = probs
        mask_mm[gi] = masks
        filled[gi] = 1

        fold_summaries.append(done)

    if not np.all(filled == 1):
        missing = np.flatnonzero(filled == 0)
        raise RuntimeError(
            f"{family}: missing OOF rows: {missing[:20].tolist()}"
        )

    prob_mm.flush()
    mask_mm.flush()

    summary_df = pd.DataFrame([
        {
            "family": d["family"],
            "fold": d["fold"],
            "best_epoch": d["best_epoch"],
            "best_mean_patient_dice": d["best_mean_patient_dice"],
            "val_patient_count": d["val_patient_count"],
            "val_slice_count": d["val_slice_count"],
            "params": d["params"],
            "trainable_params": d["trainable_params"],
        }
        for d in fold_summaries
    ])

    summary_path = stage_dir / f"CM3_{family}_FOLD_SUMMARY.csv"
    summary_df.to_csv(summary_path, index=False)

    print(
        f"{family}: mean fold patient Dice="
        f"{summary_df['best_mean_patient_dice'].mean():.6f} "
        f"std={summary_df['best_mean_patient_dice'].std(ddof=1):.6f}"
    )
    print(
        f"{family}: best epochs="
        f"{summary_df['best_epoch'].astype(int).tolist()}"
    )

    return {
        "family": family,
        "oof_prob": str(prob_mm_path),
        "oof_prob_sha256": sha256_file(prob_mm_path),
        "oof_mask": str(mask_mm_path),
        "oof_mask_sha256": sha256_file(mask_mm_path),
        "fold_summary": str(summary_path),
        "fold_summary_sha256": sha256_file(summary_path),
        "mean_fold_patient_dice": float(
            summary_df["best_mean_patient_dice"].mean()
        ),
        "std_fold_patient_dice": float(
            summary_df["best_mean_patient_dice"].std(ddof=1)
        ),
        "best_epochs": summary_df["best_epoch"].astype(int).tolist(),
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
            "Build/verify the SOURCE preprocessing cache and run the frozen "
            "batch-size-8 forward/backward preflight, then stop before training."
        ),
    )
    args = ap.parse_args()

    print("===== Q1X CM3 FIX3 SOURCE OOF SEGMENTATION TRAINING =====")
    print("FIX3_CHANGE=OFFICIAL_SEGFORMER_LEGACY_KEY_REMAP_AND_CACHE_REUSE")
    print("SCIENTIFIC_PROTOCOL_CHANGE=NO")
    print("TORCH_UPGRADE_REQUIRED=NO")
    print("MODEL_TRAINING=", "NO (preflight-only)" if args.preflight_only else "YES")
    print("SOURCE_ONLY=YES")
    print("PROMISE12_GT_ACCESS=NO")
    print("TTA=NO")
    print("SAFETY_MODEL_FITTING=NO")
    print("TARGET_TUNING=NO")
    print("BATCH_SIZE=", BATCH_SIZE)
    print("MAX_EPOCHS=", MAX_EPOCHS)
    print("MODEL_PANEL=", FAMILIES)

    print("\n===== CM2B LINEAGE GATE =====")
    if not CM2B_LOCK.is_file():
        raise FileNotFoundError(CM2B_LOCK)
    cm2b_sha = sha256_file(CM2B_LOCK)
    print(
        "CM2B_LOCK",
        cm2b_sha,
        "PASS" if cm2b_sha == EXPECTED_CM2B_LOCK_SHA else "FAIL",
    )
    if cm2b_sha != EXPECTED_CM2B_LOCK_SHA:
        raise RuntimeError("CM2B lock SHA mismatch.")

    cm2b = load_json(CM2B_LOCK)
    if cm2b.get("status") != "PASS":
        raise RuntimeError("CM2B status changed.")
    if cm2b.get("decision") != (
        "MRI_PREPROCESSING_AND_SOURCE_MODEL_PROTOCOL_LOCKED_READY_FOR_CM3_OOF_TRAINING"
    ):
        raise RuntimeError("CM2B decision changed.")

    fold_path = verify_child_artifact(cm2b, "fold_lock")
    protocol_path = verify_child_artifact(cm2b, "frozen_protocol")
    verify_child_artifact(cm2b, "source_spacing_audit")
    verify_child_artifact(cm2b, "external_image_header_audit")
    print("CM2B lineage PASS")

    protocol = load_json(protocol_path)

    target_spacing = tuple(
        float(v)
        for v in protocol["spatial_preprocessing"]["target_spacing_source_only_mm"]
    )
    if protocol["training_protocol"]["batch_size"] != BATCH_SIZE:
        raise RuntimeError("Frozen batch size changed.")
    if protocol["training_protocol"]["max_epochs"] != MAX_EPOCHS:
        raise RuntimeError("Frozen max epochs changed.")

    (
        nib,
        torch,
        nn,
        F,
        DataLoader,
        Dataset,
        InterpolationMode,
        TF,
        tqdm,
    ) = import_runtime()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n===== RUNTIME =====")
    print("torch:", torch.__version__)
    print("device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(device))
        props = torch.cuda.get_device_properties(device)
        print("VRAM GiB:", props.total_memory / (1024 ** 3))

    if args.output_dir.exists():
        stage_dir = args.output_dir
        print("STAGE_DIR=RESUME", stage_dir)
    else:
        stage_dir = args.output_dir
        stage_dir.mkdir(parents=True, exist_ok=False)
        print("STAGE_DIR=NEW", stage_dir)

    source_df = pd.read_csv(fold_path)
    if len(source_df) != 139:
        raise RuntimeError(
            f"Expected 139 source patient rows, got {len(source_df)}."
        )
    if set(source_df["fold"].astype(int).unique()) != set(range(5)):
        raise RuntimeError("SOURCE fold values are not exactly 0..4.")

    cache_dir = stage_dir / "source_cache"
    images_path, masks_path, slices_path, cache_meta_path = (
        build_source_cache(
            nib,
            torch,
            F,
            source_df,
            target_spacing,
            cache_dir,
        )
    )

    slice_df = pd.read_csv(slices_path)
    if len(slice_df) != 3553:
        raise RuntimeError(
            f"Expected 3553 cached source slices, got {len(slice_df)}."
        )

    DatasetClass = build_dataset_class(
        Dataset,
        TF,
        InterpolationMode,
    )

    preflight_path = stage_dir / "CM3_BATCH8_MODEL_PREFLIGHT.json"

    if preflight_path.is_file():
        preflight = load_json(preflight_path)
        if preflight.get("status") != "PASS":
            raise RuntimeError("Existing CM3 model preflight is not PASS.")
        print("\nBATCH8 PREFLIGHT=REUSE PASS")
    else:
        preflight_results = batch8_preflight(
            torch,
            DataLoader,
            DatasetClass,
            images_path,
            masks_path,
            slice_df,
            device,
        )
        preflight = {
            "status": "PASS",
            "version": VERSION,
            "batch_size": BATCH_SIZE,
            "input_size": INPUT_SIZE,
            "device": str(device),
            "gpu": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else None
            ),
            "families": preflight_results,
            "segformer_encoder": {
                "repo": SEGFORMER_ENCODER_REPO,
                "revision": SEGFORMER_SAFE_REVISION,
                "filename": SEGFORMER_SAFE_FILENAME,
                "sha256": SEGFORMER_SAFE_SHA256,
                "format": "safetensors",
                "initialization": (
                    "ImageNet-pretrained MiT-B0 encoder + random binary decode head"
                ),
                "legacy_key_remap": "official_transformers_conversion_mapping",
                "encoder_coverage_required": "100%",
            },
            "promises12_gt_access": False,
        }
        save_json(preflight_path, preflight)

    print("BATCH8_PREFLIGHT_SHA256=", sha256_file(preflight_path))

    if args.preflight_only:
        print("\n===== CM3 PREFLIGHT-ONLY FINAL =====")
        print("Decision= BATCH8_MODEL_PREFLIGHT_PASS_READY_FOR_FULL_CM3")
        print("SOURCE cache slices:", len(slice_df))
        print("PROMISE12 GT access: NO")
        print("PASS")
        return

    all_done = []

    for family in FAMILIES:
        print(f"\n\n########## FAMILY {family} ##########")
        for fold in range(N_FOLDS):
            print(f"\n===== TRAIN {family} FOLD {fold} =====")
            done = train_one_fold(
                torch,
                DataLoader,
                DatasetClass,
                family,
                fold,
                images_path,
                masks_path,
                slice_df,
                device,
                stage_dir,
            )
            all_done.append(done)

    family_locks = {}
    for family in FAMILIES:
        family_locks[family] = assemble_family_oof(
            family,
            stage_dir,
            slice_df,
        )

    overall_rows = []
    for family, info in family_locks.items():
        overall_rows.append({
            "family": family,
            "mean_fold_patient_dice": info["mean_fold_patient_dice"],
            "std_fold_patient_dice": info["std_fold_patient_dice"],
            "best_epochs": json.dumps(info["best_epochs"]),
        })

    overall_df = pd.DataFrame(overall_rows)
    overall_path = stage_dir / "CM3_SOURCE_OOF_MODEL_PANEL_SUMMARY.csv"
    overall_df.to_csv(overall_path, index=False)

    lock = {
        "status": "PASS",
        "decision": DECISION_PASS,
        "version": VERSION,
        "build": BUILD,
        "cm2b_lock_sha256": EXPECTED_CM2B_LOCK_SHA,
        "source_patient_count": 139,
        "source_slice_count": 3553,
        "fold_count": N_FOLDS,
        "families": FAMILIES,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "target_spacing_mm": list(target_spacing),
        "promises12_gt_access": False,
        "segformer_encoder_lock": {
            "repo": SEGFORMER_ENCODER_REPO,
            "revision": SEGFORMER_SAFE_REVISION,
            "filename": SEGFORMER_SAFE_FILENAME,
            "sha256": SEGFORMER_SAFE_SHA256,
            "format": "safetensors",
            "torch_upgrade_required": False,
            "legacy_key_remap": "official_transformers_conversion_mapping",
            "encoder_coverage_required": "100%",
        },
        "tta": False,
        "safety_model_fitting": False,
        "family_oof_artifacts": family_locks,
        "panel_summary": str(overall_path),
        "panel_summary_sha256": sha256_file(overall_path),
        "source_cache_meta": str(cache_meta_path),
        "source_cache_meta_sha256": sha256_file(cache_meta_path),
        "batch8_preflight": str(preflight_path),
        "batch8_preflight_sha256": sha256_file(preflight_path),
        "next_stage": (
            "CM4_SOURCE_TENT1_OUTCOMES_AND_FROZEN_SAFETY_REPRESENTATION_DEVELOPMENT"
        ),
    }

    lock_path = stage_dir / "CM3_SOURCE_OOF_SEGMENTATION_PANEL_LOCK.json"
    save_json(lock_path, lock)

    print("\n===== CM3 FINAL =====")
    print(overall_df.to_string(index=False))
    print("PROMISE12 GT access: NO")
    print("Decision=", DECISION_PASS)
    print("LOCK=", lock_path)
    print("LOCK SHA256=", sha256_file(lock_path))
    print("PASS")


if __name__ == "__main__":
    main()
