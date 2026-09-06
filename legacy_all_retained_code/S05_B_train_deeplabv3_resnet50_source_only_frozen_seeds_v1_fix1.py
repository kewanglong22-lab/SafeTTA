#!/usr/bin/env python
# -*- coding: utf-8 -*-

r"""
S05-B: Train second-backbone DeepLabV3-ResNet50 source-only models.

Safety:
- opens source_train/source_val only;
- never opens seen_sanity/unseen_locked;
- checkpoint selection uses source_val mean Dice only;
- three seeds are frozen in advance.

TorchVision cache is forced under F:\MEDSEG_SAFETTA\assets\torchvision_cache.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import inspect
import json
import os
import random
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import List, Tuple

# Force Torch/TorchVision pretrained downloads away from C: BEFORE torch import.
ROOT = Path(r"F:\MEDSEG_SAFETTA")
TORCH_CACHE_ROOT = ROOT / "assets" / "torchvision_cache"
os.environ["TORCH_HOME"] = str(TORCH_CACHE_ROOT)
# Must be set before CUDA/cuBLAS initialization for deterministic matmul/conv paths.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from tqdm import tqdm

import torchvision
from torchvision.models.segmentation import deeplabv3_resnet50

try:
    from torchvision.models import ResNet50_Weights
except ImportError:
    ResNet50_Weights = None


VERSION = "2026-08-17-S05-B-v1-fix1"
BUILD = "S05_B_FIX1_SINGLETON_SAFE_BATCHING_AND_EPOCH_WORKERS"

DATA_ROOT = ROOT / "data" / "processed" / "S00_polyp_locked_v1"
MANIFEST = ROOT / "data" / "splits" / "S01_locked_protocol_manifest_v1.csv"
EXPECTED_MANIFEST_SHA256 = "afb4bc1175af004819538cdbb22ee7a10f1be48035945c7e0d50fd2bbe2e3dc7"

PROTOCOL = ROOT / "docs" / "S05_A_second_backbone_replication_protocol_v1_fix1.md"
EXPECTED_PROTOCOL_SHA256 = "6a872f892ad718c301e64fa8336ffb1fc0a4cef7422d037f7b7e64a46a1ef2f5"

FROZEN_SEEDS = (20260817, 20260818, 20260819)
SEED = 20260817

IMAGE_SIZE = 352
BATCH_SIZE = 8
EPOCHS = 80
LR = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4
GRAD_CLIP_NORM = 1.0
SCALE_CHOICES = (0.75, 1.0, 1.25)

EXPECTED_ROLE_COUNTS = {
    "source_train": 1305,
    "source_val": 145,
    "seen_sanity": 162,
    "unseen_locked": 636,
}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

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


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
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


def read_manifest(path: Path) -> Tuple[List[dict], List[str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return list(reader), reader.fieldnames or []


def validate_protocol() -> str:
    if not PROTOCOL.exists():
        raise FileNotFoundError(PROTOCOL)
    sha = file_sha256(PROTOCOL)
    if sha.lower() != EXPECTED_PROTOCOL_SHA256.lower():
        raise RuntimeError(
            "Frozen S05-A protocol SHA mismatch.\n"
            f"Expected: {EXPECTED_PROTOCOL_SHA256}\n"
            f"Actual  : {sha}"
        )
    return sha


def validate_manifest(rows: List[dict], fields: List[str]) -> dict:
    required = {
        "sample_id", "split", "dataset",
        "image_relpath", "mask_relpath", "s01_role",
    }
    missing = sorted(required - set(fields))
    if missing:
        raise RuntimeError(f"Manifest missing columns: {missing}")

    sha = file_sha256(MANIFEST)
    if sha.lower() != EXPECTED_MANIFEST_SHA256.lower():
        raise RuntimeError(
            "Frozen manifest SHA mismatch.\n"
            f"Expected: {EXPECTED_MANIFEST_SHA256}\n"
            f"Actual  : {sha}"
        )

    counts = Counter(r["s01_role"] for r in rows)
    if counts != Counter(EXPECTED_ROLE_COUNTS):
        raise RuntimeError(
            f"Role count mismatch: expected={EXPECTED_ROLE_COUNTS}, "
            f"actual={dict(counts)}"
        )

    ids = [r["sample_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate sample_id in frozen manifest.")

    return {
        "manifest_sha256": sha,
        "role_counts": dict(counts),
    }


def resolve_source_rows(rows: List[dict], role: str) -> List[dict]:
    if role not in {"source_train", "source_val"}:
        raise RuntimeError(
            f"Safety stop: S05-B may only open source_train/source_val; got {role}"
        )
    selected = [r for r in rows if r["s01_role"] == role]
    expected = EXPECTED_ROLE_COUNTS[role]
    if len(selected) != expected:
        raise RuntimeError(
            f"{role} count mismatch: expected={expected}, actual={len(selected)}"
        )
    return selected


def resize_pair(image, mask, size):
    return (
        image.resize((size, size), RESAMPLE_BILINEAR),
        mask.resize((size, size), RESAMPLE_NEAREST),
    )


def random_scale_crop_pad(image, mask, base_size, rng):
    scale = rng.choice(SCALE_CHOICES)
    scaled = int(round(base_size * scale))

    image = image.resize((scaled, scaled), RESAMPLE_BILINEAR)
    mask = mask.resize((scaled, scaled), RESAMPLE_NEAREST)

    if scaled > base_size:
        max_off = scaled - base_size
        left = rng.randint(0, max_off)
        top = rng.randint(0, max_off)
        box = (left, top, left + base_size, top + base_size)
        image = image.crop(box)
        mask = mask.crop(box)

    elif scaled < base_size:
        pad = base_size - scaled
        left = rng.randint(0, pad)
        top = rng.randint(0, pad)
        right = pad - left
        bottom = pad - top
        image = ImageOps.expand(image, border=(left, top, right, bottom), fill=0)
        mask = ImageOps.expand(mask, border=(left, top, right, bottom), fill=0)

    if image.size != (base_size, base_size):
        raise RuntimeError(f"Aug image size mismatch: {image.size}")
    if mask.size != (base_size, base_size):
        raise RuntimeError(f"Aug mask size mismatch: {mask.size}")

    return image, mask


def image_to_tensor(image):
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1)).copy()
    return torch.from_numpy(arr)


def mask_to_tensor(mask):
    arr = np.asarray(mask.convert("L"), dtype=np.float32)
    maxv = float(arr.max())
    if maxv <= 0:
        raise RuntimeError("Empty source mask encountered.")
    arr = (arr > 0.5 * maxv).astype(np.float32)
    return torch.from_numpy(arr[None].copy())


class SingletonSafeBatchSampler(Sampler):
    """
    Deterministic shuffled batches with max size BATCH_SIZE and min size >=2.

    For N=1305, B=8:
      default chunks end as [..., 8, 1]
      FIX1 rebalances only the tail to [..., 7, 2].

    No source sample is dropped or duplicated.
    """
    def __init__(self, dataset_len: int, batch_size: int, base_seed: int):
        self.dataset_len = int(dataset_len)
        self.batch_size = int(batch_size)
        self.base_seed = int(base_seed)
        self.epoch = 0

        if self.dataset_len < 2:
            raise ValueError("SingletonSafeBatchSampler requires dataset_len >= 2.")
        if self.batch_size < 2:
            raise ValueError("batch_size must be >=2.")

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def _ordered_indices(self):
        payload = (
            f"S05_FIX1_BATCH_ORDER::{self.base_seed}::{self.epoch}"
        ).encode("utf-8")
        stable_seed = int(hashlib.sha256(payload).hexdigest()[:16], 16)
        g = torch.Generator()
        g.manual_seed(stable_seed % (2**63 - 1))
        return torch.randperm(self.dataset_len, generator=g).tolist()

    def _build_batches(self):
        indices = self._ordered_indices()
        batches = [
            indices[i:i + self.batch_size]
            for i in range(0, len(indices), self.batch_size)
        ]

        if len(batches) >= 2 and len(batches[-1]) == 1:
            moved = batches[-2].pop()
            batches[-1].insert(0, moved)

        if not batches:
            raise RuntimeError("No training batches constructed.")

        flat = [idx for batch in batches for idx in batch]
        if len(flat) != self.dataset_len:
            raise RuntimeError(
                f"Batch sampler lost samples: {len(flat)} vs {self.dataset_len}"
            )
        if len(set(flat)) != self.dataset_len:
            raise RuntimeError("Batch sampler duplicated source samples.")
        if min(len(batch) for batch in batches) < 2:
            raise RuntimeError(
                f"Unsafe singleton training batch remains: "
                f"{[len(b) for b in batches[-3:]]}"
            )
        if max(len(batch) for batch in batches) > self.batch_size:
            raise RuntimeError("Batch sampler exceeded frozen nominal batch size.")

        return batches

    def __iter__(self):
        yield from self._build_batches()

    def __len__(self):
        # Rebalancing does not change the normal ceil(N/B) batch count.
        return (self.dataset_len + self.batch_size - 1) // self.batch_size


class SourcePolypDataset(Dataset):
    def __init__(self, rows, data_root, training, base_seed):
        self.rows = list(rows)
        self.data_root = Path(data_root)
        self.training = bool(training)
        self.base_seed = int(base_seed)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.rows)

    def _rng(self, index):
        sid = self.rows[index]["sample_id"]
        payload = f"{self.base_seed}::{self.epoch}::{sid}".encode("utf-8")
        value = int(hashlib.sha256(payload).hexdigest()[:16], 16)
        return random.Random(value)

    def __getitem__(self, index):
        row = self.rows[index]

        if row["s01_role"] not in {"source_train", "source_val"}:
            raise RuntimeError(
                f"Dataset safety violation: role={row['s01_role']}"
            )

        image_path = self.data_root / Path(row["image_relpath"])
        mask_path = self.data_root / Path(row["mask_relpath"])

        if not image_path.exists():
            raise FileNotFoundError(image_path)
        if not mask_path.exists():
            raise FileNotFoundError(mask_path)

        with Image.open(image_path) as im:
            image = im.convert("RGB")
        with Image.open(mask_path) as ma:
            mask = ma.convert("L")

        if image.size != mask.size:
            raise RuntimeError(
                f"Image/mask size mismatch {row['sample_id']}: "
                f"{image.size} vs {mask.size}"
            )

        if self.training:
            rng = self._rng(index)
            if rng.random() < 0.5:
                image = ImageOps.mirror(image)
                mask = ImageOps.mirror(mask)
            if rng.random() < 0.5:
                image = ImageOps.flip(image)
                mask = ImageOps.flip(mask)

            image, mask = resize_pair(image, mask, IMAGE_SIZE)
            image, mask = random_scale_crop_pad(
                image, mask, IMAGE_SIZE, rng
            )
        else:
            image, mask = resize_pair(image, mask, IMAGE_SIZE)

        return {
            "image": image_to_tensor(image),
            "mask": mask_to_tensor(mask),
            "sample_id": row["sample_id"],
        }


def structure_loss(pred, mask):
    weight = 1 + 5 * torch.abs(
        F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask
    )

    wbce = F.binary_cross_entropy_with_logits(
        pred, mask, reduction="none"
    )
    wbce = (weight * wbce).sum(dim=(2, 3)) / weight.sum(dim=(2, 3))

    prob = torch.sigmoid(pred)
    inter = ((prob * mask) * weight).sum(dim=(2, 3))
    union = ((prob + mask) * weight).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()


@torch.no_grad()
def binary_metrics_from_logits(logits, mask):
    pred = (torch.sigmoid(logits) >= 0.5).float()
    target = (mask >= 0.5).float()

    dims = (1, 2, 3)
    inter = (pred * target).sum(dim=dims)
    pred_sum = pred.sum(dim=dims)
    target_sum = target.sum(dim=dims)
    union = pred_sum + target_sum - inter

    dice = (2 * inter + 1e-7) / (pred_sum + target_sum + 1e-7)
    iou = (inter + 1e-7) / (union + 1e-7)
    return dice, iou


def build_deeplab(pretrained_backbone: bool):
    """
    New TorchVision API first; legacy fallback only for older environments.
    """
    sig = inspect.signature(deeplabv3_resnet50)

    if "weights_backbone" in sig.parameters:
        backbone_weights = (
            ResNet50_Weights.IMAGENET1K_V1
            if pretrained_backbone
            else None
        )
        model = deeplabv3_resnet50(
            weights=None,
            weights_backbone=backbone_weights,
            num_classes=1,
            aux_loss=False,
        )
        descriptor = (
            "ResNet50_Weights.IMAGENET1K_V1"
            if pretrained_backbone
            else "NONE"
        )
        return model, descriptor

    # Legacy TorchVision API.
    model = deeplabv3_resnet50(
        pretrained=False,
        pretrained_backbone=bool(pretrained_backbone),
        num_classes=1,
        aux_loss=False,
    )
    descriptor = (
        "legacy_pretrained_backbone=True"
        if pretrained_backbone
        else "NONE"
    )
    return model, descriptor


def find_resnet50_cached_weight():
    """
    Record the actual cached ImageNet backbone file if TorchVision downloaded it.
    No hard-coded dependency on a particular filename.
    """
    hub_dir = Path(torch.hub.get_dir())
    checkpoint_dir = hub_dir / "checkpoints"

    if not checkpoint_dir.exists():
        return None

    candidates = sorted(
        checkpoint_dir.glob("resnet50-*.pth"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def deeplab_logits(model, images):
    out = model(images)
    if not isinstance(out, dict) or "out" not in out:
        raise RuntimeError(
            "Unexpected TorchVision DeepLabV3 output; expected dict with 'out'."
        )
    logits = out["out"]
    if logits.ndim != 4 or logits.shape[1] != 1:
        raise RuntimeError(f"Unexpected DeepLabV3 logit shape: {tuple(logits.shape)}")
    return logits


def make_grad_scaler(enabled):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            pass
    return torch.cuda.amp.GradScaler(enabled=enabled)


@contextlib.contextmanager
def autocast_context(enabled):
    if not enabled:
        yield
        return

    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        try:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                yield
            return
        except TypeError:
            pass

    with torch.cuda.amp.autocast(dtype=torch.float16):
        yield


def train_one_epoch(
    model, loader, dataset, batch_sampler, optimizer, scaler,
    device, epoch, amp_enabled
):
    model.train()
    dataset.set_epoch(epoch)
    batch_sampler.set_epoch(epoch)

    loss_sum = 0.0
    n_samples = 0

    pbar = tqdm(
        loader,
        desc=f"Train {epoch:03d}/{EPOCHS}",
        unit="batch",
        dynamic_ncols=True,
    )

    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast_context(amp_enabled):
            logits = deeplab_logits(model, images)
            loss = structure_loss(logits, masks)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=GRAD_CLIP_NORM
        )
        scaler.step(optimizer)
        scaler.update()

        bs = images.shape[0]
        loss_sum += float(loss.detach().cpu()) * bs
        n_samples += bs

        pbar.set_postfix(
            loss=f"{loss_sum / max(n_samples, 1):.4f}",
            gpu_mem=(
                f"{torch.cuda.max_memory_allocated() / 1024**3:.2f}G"
                if device.type == "cuda"
                else "CPU"
            ),
        )

    return loss_sum / max(n_samples, 1)


@torch.no_grad()
def validate(model, loader, device, amp_enabled):
    model.eval()
    dices = []
    ious = []

    pbar = tqdm(
        loader,
        desc="Source-val",
        unit="batch",
        dynamic_ncols=True,
    )

    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        with autocast_context(amp_enabled):
            logits = deeplab_logits(model, images)

        dice, iou = binary_metrics_from_logits(
            logits.float(), masks.float()
        )
        dices.extend(dice.detach().cpu().numpy().tolist())
        ious.extend(iou.detach().cpu().numpy().tolist())

        pbar.set_postfix(
            dice=f"{np.mean(dices):.4f}",
            iou=f"{np.mean(ious):.4f}",
        )

    return {
        "mean_dice": float(np.mean(dices)),
        "median_dice": float(np.median(dices)),
        "mean_iou": float(np.mean(ious)),
        "n": len(dices),
    }


def save_csv(path, rows, fieldnames):
    with Path(path).open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def train(args):
    seed_everything(SEED)

    protocol_sha = validate_protocol()

    if not MANIFEST.exists():
        raise FileNotFoundError(MANIFEST)
    if not DATA_ROOT.exists():
        raise FileNotFoundError(DATA_ROOT)

    output_dir = args.output_dir

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output already exists: {output_dir}\n"
                "Refusing to overwrite a frozen-seed training run."
            )
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=False)

    rows, fields = read_manifest(MANIFEST)
    manifest_info = validate_manifest(rows, fields)

    train_rows = resolve_source_rows(rows, "source_train")
    val_rows = resolve_source_rows(rows, "source_val")

    assert all(r["s01_role"] == "source_train" for r in train_rows)
    assert all(r["s01_role"] == "source_val" for r in val_rows)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    amp_enabled = device.type == "cuda"

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    print(f"[BUILD] {VERSION} | {BUILD}")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Torch: {torch.__version__}")
    print(f"TorchVision: {torchvision.__version__}")
    print(f"TORCH_HOME: {os.environ.get('TORCH_HOME')}")
    print()
    print("Frozen data:")
    print(f"  source_train={len(train_rows)}")
    print(f"  source_val={len(val_rows)}")
    print("  seen_sanity opened=NO")
    print("  unseen_locked opened=NO")
    print("  target metrics computed=NO")
    print()
    print("Training:")
    print(f"  seed={SEED}")
    print(f"  image={IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"  epochs={EPOCHS}")
    print(f"  batch={BATCH_SIZE}")
    print(f"  optimizer=AdamW lr={LR} wd={WEIGHT_DECAY}")
    print(f"  scales={SCALE_CHOICES}")
    print("  singleton-safe batching=True (tail 8+1 -> 7+2 when needed)")
    print("  source samples dropped per epoch=0")
    print("  persistent_workers=False")
    print(f"  CUBLAS_WORKSPACE_CONFIG={os.environ.get('CUBLAS_WORKSPACE_CONFIG')}")
    print()

    print("Building DeepLabV3-ResNet50 with ImageNet-pretrained backbone...")
    model, backbone_descriptor = build_deeplab(
        pretrained_backbone=True
    )
    model = model.to(device)

    cached_backbone = find_resnet50_cached_weight()
    cached_backbone_sha = (
        file_sha256(cached_backbone)
        if cached_backbone is not None
        else "NOT_LOCATED"
    )

    param_total = sum(p.numel() for p in model.parameters())
    param_trainable = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    print(f"Architecture: DeepLabV3-ResNet50")
    print(f"Backbone pretrained: {backbone_descriptor}")
    print(f"Parameters total/trainable: {param_total:,}/{param_trainable:,}")
    print(f"Backbone cache file: {cached_backbone}")
    print(f"Backbone cache SHA256: {cached_backbone_sha}")

    train_ds = SourcePolypDataset(
        train_rows, DATA_ROOT, training=True, base_seed=SEED
    )
    val_ds = SourcePolypDataset(
        val_rows, DATA_ROOT, training=False, base_seed=SEED
    )

    train_batch_sampler = SingletonSafeBatchSampler(
        dataset_len=len(train_ds),
        batch_size=BATCH_SIZE,
        base_seed=SEED,
    )

    # FIX1: persistent workers are disabled so dataset.set_epoch(epoch)
    # is reflected in worker-side deterministic augmentation every epoch.
    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_batch_sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        persistent_workers=False,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    scaler = make_grad_scaler(amp_enabled)

    best_dice = -1.0
    best_epoch = -1
    history = []
    started = time.time()

    try:
        for epoch in range(1, EPOCHS + 1):
            epoch_start = time.time()

            train_loss = train_one_epoch(
                model=model,
                loader=train_loader,
                dataset=train_ds,
                batch_sampler=train_batch_sampler,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                epoch=epoch,
                amp_enabled=amp_enabled,
            )

            val = validate(
                model=model,
                loader=val_loader,
                device=device,
                amp_enabled=amp_enabled,
            )

            elapsed = time.time() - epoch_start

            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "source_val_mean_dice": val["mean_dice"],
                "source_val_median_dice": val["median_dice"],
                "source_val_mean_iou": val["mean_iou"],
                "epoch_seconds": elapsed,
            }
            history.append(record)

            if val["mean_dice"] > best_dice:
                best_dice = val["mean_dice"]
                best_epoch = epoch

                checkpoint = {
                    "script_version": VERSION,
                    "build": BUILD,
                    "architecture": "DeepLabV3-ResNet50",
                    "torch_version": torch.__version__,
                    "torchvision_version": torchvision.__version__,
                    "seed": SEED,
                    "epoch": epoch,
                    "source_val_mean_dice": best_dice,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "manifest_sha256": manifest_info["manifest_sha256"],
                    "protocol_sha256": protocol_sha,
                    "backbone_pretrain_descriptor": backbone_descriptor,
                    "backbone_cache_path": (
                        str(cached_backbone)
                        if cached_backbone is not None
                        else None
                    ),
                    "backbone_cache_sha256": cached_backbone_sha,
                    "config": {
                        "image_size": IMAGE_SIZE,
                        "batch_size": BATCH_SIZE,
                        "epochs": EPOCHS,
                        "lr": LR,
                        "weight_decay": WEIGHT_DECAY,
                        "grad_clip_norm": GRAD_CLIP_NORM,
                        "scales": list(SCALE_CHOICES),
                        "amp": amp_enabled,
                        "loss": "single-output structure_loss",
                        "aux_loss": False,
                    },
                }

                torch.save(
                    checkpoint,
                    checkpoints_dir / "best_source_val_dice.pt",
                )

            print(
                f"[Epoch {epoch:03d}/{EPOCHS}] "
                f"loss={train_loss:.5f} "
                f"valDice={val['mean_dice']:.5f} "
                f"valIoU={val['mean_iou']:.5f} "
                f"best={best_dice:.5f}@{best_epoch}"
            )

            save_csv(
                output_dir / "history.csv",
                history,
                [
                    "epoch",
                    "train_loss",
                    "source_val_mean_dice",
                    "source_val_median_dice",
                    "source_val_mean_iou",
                    "epoch_seconds",
                ],
            )

        total_seconds = time.time() - started
        peak_gpu_gb = (
            float(torch.cuda.max_memory_allocated() / 1024**3)
            if device.type == "cuda"
            else 0.0
        )

        best_ckpt = checkpoints_dir / "best_source_val_dice.pt"
        if not best_ckpt.exists():
            raise RuntimeError("Best checkpoint missing after training.")

        best_ckpt_sha = file_sha256(best_ckpt)

        audit = {
            "script_version": VERSION,
            "build": BUILD,
            "architecture": "DeepLabV3-ResNet50",
            "seed": SEED,
            "protocol_sha256": protocol_sha,
            "manifest_sha256": manifest_info["manifest_sha256"],
            "source_train_n": len(train_rows),
            "source_val_n": len(val_rows),
            "seen_sanity_images_opened": False,
            "seen_sanity_masks_opened": False,
            "unseen_locked_images_opened": False,
            "unseen_locked_masks_opened": False,
            "target_metrics_computed": False,
            "checkpoint_selection_metric": "source_val_mean_dice",
            "singleton_safe_batching": True,
            "train_samples_dropped_per_epoch": 0,
            "train_min_batch_size": 2,
            "train_max_batch_size": BATCH_SIZE,
            "persistent_workers": False,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "best_epoch": best_epoch,
            "best_source_val_mean_dice": best_dice,
            "backbone_pretrain_descriptor": backbone_descriptor,
            "backbone_cache_path": (
                str(cached_backbone)
                if cached_backbone is not None
                else None
            ),
            "backbone_cache_sha256": cached_backbone_sha,
            "best_checkpoint": str(best_ckpt),
            "best_checkpoint_sha256": best_ckpt_sha,
            "parameters_total": param_total,
            "parameters_trainable": param_trainable,
            "peak_gpu_memory_gb": peak_gpu_gb,
            "training_seconds": total_seconds,
            "decision": (
                f"S05_B_SEED{SEED}_DEEPLAB_SOURCE_MODEL_READY"
            ),
        }

        (output_dir / "audit.json").write_text(
            json.dumps(audit, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        summary = f"""===== S05-B DEEPLABV3-R50 SOURCE-ONLY TRAINING =====
Script version: {VERSION}
Build: {BUILD}

Device: {device}
Seed: {SEED}

Frozen data:
  source_train={len(train_rows)}
  source_val={len(val_rows)}
  seen_sanity opened=NO
  unseen_locked opened=NO
  target metrics computed=NO

Architecture:
  DeepLabV3
  backbone=ResNet-50 ImageNet pretrained
  TorchVision={torchvision.__version__}
  parameters={param_total:,}
  trainable={param_trainable:,}

Training:
  input={IMAGE_SIZE}x{IMAGE_SIZE}
  epochs={EPOCHS}
  batch={BATCH_SIZE}
  optimizer=AdamW
  lr={LR}
  weight_decay={WEIGHT_DECAY}
  AMP={amp_enabled}
  random scales={SCALE_CHOICES}
  horizontal flip=True
  vertical flip=True
  loss=single-output structure loss
  singleton-safe batching=True
  tail singleton rule=8+1 -> 7+2
  source samples dropped per epoch=0
  persistent_workers=False
  CUBLAS_WORKSPACE_CONFIG={os.environ.get("CUBLAS_WORKSPACE_CONFIG")}

Checkpoint selection:
  metric=source_val mean Dice
  best epoch={best_epoch}
  best source-val mean Dice={best_dice:.6f}

Best checkpoint:
  {best_ckpt}
  SHA256={best_ckpt_sha}

Backbone pretrained:
  descriptor={backbone_descriptor}
  cache={cached_backbone}
  SHA256={cached_backbone_sha}

Peak GPU memory GB={peak_gpu_gb:.3f}
Training seconds={total_seconds:.1f}

Decision: S05_B_SEED{SEED}_DEEPLAB_SOURCE_MODEL_READY
"""
        (output_dir / "summary.txt").write_text(
            summary, encoding="utf-8"
        )

        print()
        print(summary)
        print(f"[OK] Outputs: {output_dir}")

    except Exception:
        failed = {
            "script_version": VERSION,
            "build": BUILD,
            "status": "FAILED_TECHNICAL",
            "seed": SEED,
            "seen_sanity_images_opened": False,
            "unseen_locked_images_opened": False,
            "target_metrics_computed": False,
        }
        (output_dir / "FAILED_TECHNICAL.json").write_text(
            json.dumps(failed, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        raise


def self_test():
    # Loss.
    logits = torch.zeros(2, 1, 64, 64)
    masks = torch.zeros(2, 1, 64, 64)
    masks[:, :, 16:48, 16:48] = 1.0
    loss = structure_loss(logits, masks)
    assert torch.isfinite(loss)
    assert float(loss) > 0

    # Metric.
    perfect = torch.where(
        masks > 0.5,
        torch.tensor(20.0),
        torch.tensor(-20.0),
    )
    dice, iou = binary_metrics_from_logits(perfect, masks)
    assert torch.allclose(dice, torch.ones_like(dice), atol=1e-5)
    assert torch.allclose(iou, torch.ones_like(iou), atol=1e-5)

    # Architecture without network download.
    model, desc = build_deeplab(pretrained_backbone=False)
    model.eval()
    with torch.no_grad():
        x = torch.randn(1, 3, 64, 64)
        logits = deeplab_logits(model, x)

    assert tuple(logits.shape) == (1, 1, 64, 64)
    assert torch.isfinite(logits).all()
    assert desc == "NONE"

    # Leakage helper.
    fake_rows = (
        [
            {"sample_id": f"train_{i}", "s01_role": "source_train"}
            for i in range(EXPECTED_ROLE_COUNTS["source_train"])
        ]
        + [
            {"sample_id": f"val_{i}", "s01_role": "source_val"}
            for i in range(EXPECTED_ROLE_COUNTS["source_val"])
        ]
        + [{"sample_id": "u0", "s01_role": "unseen_locked"}]
    )
    assert len(resolve_source_rows(fake_rows, "source_train")) == 1305
    assert len(resolve_source_rows(fake_rows, "source_val")) == 145

    try:
        resolve_source_rows(fake_rows, "unseen_locked")
        raise AssertionError("Leakage guard failed.")
    except RuntimeError:
        pass

    # FIX1 singleton-safe batching: N=1305, B=8 must end 7+2.
    sampler = SingletonSafeBatchSampler(
        dataset_len=1305,
        batch_size=8,
        base_seed=20260817,
    )
    sampler.set_epoch(1)
    batches = list(iter(sampler))
    flat = [i for b in batches for i in b]
    assert len(batches) == 164
    assert [len(b) for b in batches[-2:]] == [7, 2]
    assert min(len(b) for b in batches) >= 2
    assert max(len(b) for b in batches) <= 8
    assert len(flat) == 1305
    assert len(set(flat)) == 1305

    # Different epoch -> different deterministic shuffle.
    sampler.set_epoch(2)
    batches2 = list(iter(sampler))
    assert batches2 != batches

    assert FROZEN_SEEDS == (20260817, 20260818, 20260819)
    assert str(TORCH_CACHE_ROOT).startswith(r"F:\MEDSEG_SAFETTA")
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"

    print(f"TORCHVISION_VERSION={torchvision.__version__}")
    print("STRUCTURE_LOSS_TEST_PASS")
    print("METRIC_TEST_PASS")
    print("DEEPLAB_FORWARD_TEST_PASS")
    print("UNSEEN_ROLE_GUARD_TEST_PASS")
    print("SINGLETON_SAFE_BATCH_TEST_PASS")
    print("EPOCH_SHUFFLE_PROPAGATION_TEST_PASS")
    print("CUBLAS_DETERMINISM_CONFIG_TEST_PASS")
    print("FROZEN_SEED_SET_TEST_PASS")
    print("F_DRIVE_CACHE_TEST_PASS")
    print("SELF_TEST_PASS")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train DeepLabV3-ResNet50 source-only for one of the "
            "three frozen S05 replication seeds."
        )
    )
    parser.add_argument(
        "--seed",
        type=int,
        choices=FROZEN_SEEDS,
        default=20260817,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=NUM_WORKERS,
    )
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Delete this exact seed output and rerun. "
            "Do not use after a scientific result is accepted/frozen."
        ),
    )
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main():
    global SEED

    args = parse_args()

    if args.self_test:
        self_test()
        return 0

    SEED = int(args.seed)

    if args.output_dir is None:
        args.output_dir = (
            ROOT / "outputs"
            / f"S05_B_deeplabv3_r50_source_only_seed{SEED}_v1"
        )

    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
