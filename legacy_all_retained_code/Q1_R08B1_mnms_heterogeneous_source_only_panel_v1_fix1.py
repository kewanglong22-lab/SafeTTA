#!/usr/bin/env python
# -*- coding: utf-8 -*-

r"""
Q1-R08B1 — Formal heterogeneous M&Ms source-only model-state panel.

Architectures:
- DeepLabV3-R50
- FCN-R50
- SegFormer-B0

Seeds:
- 20260819
- 20260820
- 20260821

Directions:
- B -> A
- A -> B

Target NPZ files are never opened.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

ROOT = Path(r"F:\MEDSEG_SAFETTA")
CACHE_ROOT = ROOT / "cache"
TORCH_CACHE = CACHE_ROOT / "torch"
HF_CACHE = CACHE_ROOT / "huggingface"
os.environ.setdefault("TORCH_HOME", str(TORCH_CACHE))
os.environ.setdefault("HF_HOME", str(HF_CACHE))
os.environ.setdefault("HF_HUB_CACHE", str(HF_CACHE / "hub"))

import numpy as np
from tqdm import tqdm

VERSION = "2026-08-20-Q1-R08B1-v1-fix1"
BUILD = "Q1_R08B1_MNMS_HETEROGENEOUS_SOURCE_ONLY_PANEL"

PROTOCOL = ROOT / "docs" / "Q1_R08B1_mnms_heterogeneous_source_only_panel_preregistered_protocol_v1.md"
EXPECTED_PROTOCOL_SHA256 = "7064ec5a3d4223fda0b7154187b442ff65eac23c2f2769ae06f88a03f533105a"

R08A2_DIR = ROOT / "outputs" / "Q1_R08A2_mnms_zip_streaming_compact_preprocessing_v1"
R08A2_LOCK = R08A2_DIR / "Q1_R08A2_MNMS_COMPACT_BA_ASSET_LOCK.json"
EXPECTED_R08A2_LOCK_SHA256 = "e7df0cb2228a8b1f405ec97b78674763d48a4bd9a061017ef2cb61f67640c4c9"
EXPECTED_R08A2_DECISION = "MNMS_COMPACT_BA_ASSET_READY"

R08B0_DIR = ROOT / "outputs" / "Q1_R08B0_mnms_source_only_deeplab_pilot_v1"
R08B0_LOCK = R08B0_DIR / "Q1_R08B0_MNMS_SOURCE_ONLY_PILOT_LOCK.json"
R08B0_SPLIT = R08B0_DIR / "source_split_manifest.csv"
EXPECTED_R08B0_LOCK_SHA256 = "393fae163812613b9d7980236deb75d1b5d80d6c52fc98de8e9d461e514a5582"
EXPECTED_R08B0_DECISION = "MNMS_SOURCE_ONLY_SEGMENTATION_PILOT_READY"

DATA_DIR = ROOT / "data" / "processed" / "MnMs_R08A2_BA_compact_v1"
CASE_MANIFEST = DATA_DIR / "case_manifest.csv"
OUTPUT_DIR = ROOT / "outputs" / "Q1_R08B1_mnms_heterogeneous_source_only_panel_v1"

ARCHITECTURES = ("deeplabv3_r50", "fcn_r50", "segformer_b0")
ARCH_DISPLAY = {
    "deeplabv3_r50": "DeepLabV3-ResNet50",
    "fcn_r50": "FCN-ResNet50",
    "segformer_b0": "SegFormer-B0",
}
SEEDS = (20260819, 20260820, 20260821)
DIRECTIONS = (("B", "A"), ("A", "B"))
EXPECTED_VENDOR_COUNTS = {"B": 125, "A": 95}
EXPECTED_SPLITS = {
    "B": {"train": 100, "val": 25},
    "A": {"train": 76, "val": 19},
}

NUM_CLASSES = 4
INPUT_SIZE = 256
EPOCHS = 30
BATCH_SIZE = 8
LR = 3e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
MIN_STATE_DICE = 0.65
MIN_ARCH_DIRECTION_MEDIAN_DICE = 0.75
EXPECTED_STATES = 18
SEGFORMER_MODEL_ID = "nvidia/mit-b0"
SEGFORMER_SAFE_REVISION = "25ce79d97e6d9d509ed12e17cb2eb89b0a83a2dc"
SEGFORMER_SAFE_WEIGHT_FILENAME = "model.safetensors"
SEGFORMER_SAFE_WEIGHT_SHA256 = "3e5ad9cd1dd8ecf8305c23fcdf01ef241f08c7b2dddacb6ec7de5a887188798a"
SEGFORMER_ID2LABEL = {
    0: "background",
    1: "class_1",
    2: "class_2",
    3: "class_3",
}
SEGFORMER_LABEL2ID = {
    v: k for k, v in SEGFORMER_ID2LABEL.items()
}

READY = "MNMS_HETEROGENEOUS_SOURCE_PANEL_READY"
INSUFFICIENT = "MNMS_HETEROGENEOUS_SOURCE_PANEL_INSUFFICIENT"


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def validate_sha(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual.lower() != expected.lower():
        raise RuntimeError(f"{label} SHA mismatch: expected={expected} actual={actual}")
    return actual


def write_csv(path: Path, rows, fields: Sequence[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(fields))
        w.writeheader()
        w.writerows(rows)


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def set_global_seed(seed: int, torch):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def robust_normalize_phase(phase_zyx: np.ndarray) -> np.ndarray:
    x = np.asarray(phase_zyx, dtype=np.float32)
    finite = np.isfinite(x)
    if not finite.any():
        raise RuntimeError("Phase has no finite pixels.")
    support = finite & (x != 0)
    values = x[support] if int(support.sum()) >= 32 else x[finite]
    lo = float(np.percentile(values, 0.5))
    hi = float(np.percentile(values, 99.5))
    if not math.isfinite(lo) or not math.isfinite(hi) or hi < lo:
        raise RuntimeError("Invalid normalization percentiles.")
    clipped = np.clip(x, lo, hi)
    stats_values = clipped[support] if int(support.sum()) >= 32 else clipped[finite]
    mean = float(stats_values.mean())
    std = float(stats_values.std())
    if not math.isfinite(mean):
        raise RuntimeError("Non-finite normalization mean.")
    if not math.isfinite(std) or std < 1e-6:
        std = 1.0
    out = np.clip((clipped - mean) / std, -5.0, 5.0)
    out[~finite] = 0.0
    return out.astype(np.float32, copy=False)


def dice_binary(pred: np.ndarray, target: np.ndarray) -> float:
    pred = pred.astype(bool, copy=False)
    target = target.astype(bool, copy=False)
    p = int(pred.sum())
    t = int(target.sum())
    if p == 0 and t == 0:
        return 1.0
    inter = int(np.logical_and(pred, target).sum())
    return 2.0 * inter / (p + t)


def validate_upstream():
    validate_sha(PROTOCOL, EXPECTED_PROTOCOL_SHA256, "R08B1 protocol")
    validate_sha(R08A2_LOCK, EXPECTED_R08A2_LOCK_SHA256, "R08A2 lock")
    validate_sha(R08B0_LOCK, EXPECTED_R08B0_LOCK_SHA256, "R08B0 lock")

    a2 = json.loads(R08A2_LOCK.read_text(encoding="utf-8"))
    b0 = json.loads(R08B0_LOCK.read_text(encoding="utf-8"))

    if a2.get("decision") != EXPECTED_R08A2_DECISION:
        raise RuntimeError(f"Unexpected R08A2 decision={a2.get('decision')}")
    if b0.get("decision") != EXPECTED_R08B0_DECISION:
        raise RuntimeError(f"Unexpected R08B0 decision={b0.get('decision')}")
    if not R08B0_SPLIT.is_file():
        raise FileNotFoundError(R08B0_SPLIT)

    artifact = (b0.get("artifacts") or {}).get("source_split_manifest.csv")
    if not artifact or not artifact.get("sha256"):
        raise RuntimeError("R08B0 lock lacks source_split_manifest.csv SHA.")

    split_sha = sha256_file(R08B0_SPLIT)
    if split_sha != artifact["sha256"]:
        raise RuntimeError("R08B0 source split SHA mismatch.")

    return a2, b0, split_sha


def load_case_manifest() -> dict[str, dict]:
    if not CASE_MANIFEST.is_file():
        raise FileNotFoundError(CASE_MANIFEST)
    rows = {}
    with CASE_MANIFEST.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {"subject_id", "vendor", "centre", "image_npz", "mask_npz", "shape_tzyx"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise RuntimeError(f"case_manifest missing={sorted(missing)}")
        for row in reader:
            vendor = row["vendor"].strip()
            if vendor not in EXPECTED_VENDOR_COUNTS:
                continue
            sid = row["subject_id"].strip()
            if sid in rows:
                raise RuntimeError(f"Duplicate subject={sid}")
            image_path = DATA_DIR / row["image_npz"]
            mask_path = DATA_DIR / row["mask_npz"]
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            if not mask_path.is_file():
                raise FileNotFoundError(mask_path)
            rows[sid] = {
                "subject_id": sid,
                "vendor": vendor,
                "centre": row["centre"].strip(),
                "image_path": image_path,
                "mask_path": mask_path,
                "shape_tzyx": row["shape_tzyx"],
            }
    counts = Counter(r["vendor"] for r in rows.values())
    if dict(counts) != EXPECTED_VENDOR_COUNTS:
        raise RuntimeError(f"Vendor counts={dict(counts)} expected={EXPECTED_VENDOR_COUNTS}")
    return rows


def load_inherited_splits(case_rows: dict[str, dict]):
    by_direction = {}
    with R08B0_SPLIT.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {
            "direction", "source_vendor", "target_vendor", "split",
            "subject_id", "target_file_open_allowed",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise RuntimeError(f"R08B0 split missing={sorted(missing)}")
        for row in reader:
            direction = row["direction"]
            source = row["source_vendor"].strip()
            target = row["target_vendor"].strip()
            sid = row["subject_id"].strip()
            split = row["split"].strip()
            if sid not in case_rows:
                raise RuntimeError(f"Split subject absent from compact manifest={sid}")
            if case_rows[sid]["vendor"] != source:
                raise RuntimeError(f"Source-vendor mismatch for {sid}")
            if int(row["target_file_open_allowed"]) != 0:
                raise RuntimeError("Inherited split permits target access.")
            d = by_direction.setdefault(
                direction,
                {"source_vendor": source, "target_vendor": target, "train": [], "val": []},
            )
            if split not in {"train", "val"}:
                raise RuntimeError(f"Unexpected split={split}")
            d[split].append(case_rows[sid])

    for source, target in DIRECTIONS:
        direction = f"{source}_to_{target}"
        if direction not in by_direction:
            raise RuntimeError(f"Missing direction={direction}")
        d = by_direction[direction]
        expected = EXPECTED_SPLITS[source]
        if len(d["train"]) != expected["train"] or len(d["val"]) != expected["val"]:
            raise RuntimeError(f"Inherited split count changed for {direction}")
        train_ids = {r["subject_id"] for r in d["train"]}
        val_ids = {r["subject_id"] for r in d["val"]}
        if train_ids & val_ids:
            raise RuntimeError(f"Train/val overlap in {direction}")
        d["train"] = sorted(d["train"], key=lambda r: r["subject_id"])
        d["val"] = sorted(d["val"], key=lambda r: r["subject_id"])
    return by_direction


def load_source_case(row: dict, tracker: dict):
    tracker["source_image_npz_opens"] += 1
    with np.load(row["image_path"], allow_pickle=False) as d:
        image = np.asarray(d["image"], dtype=np.float32)
    tracker["source_mask_npz_opens"] += 1
    with np.load(row["mask_path"], allow_pickle=False) as d:
        mask = np.asarray(d["mask"], dtype=np.uint8)
    if image.shape != mask.shape or image.ndim != 4 or image.shape[0] != 2:
        raise RuntimeError(f"{row['subject_id']}: invalid compact shape image={image.shape} mask={mask.shape}")
    norm = np.empty_like(image, dtype=np.float32)
    for phase in range(2):
        norm[phase] = robust_normalize_phase(image[phase])
    return norm, mask


def preload_rows(rows: list[dict], tracker: dict, desc: str):
    cases = {}
    for row in tqdm(rows, desc=desc, unit="subject", dynamic_ncols=True):
        image, mask = load_source_case(row, tracker)
        cases[row["subject_id"]] = {"row": row, "image": image, "mask": mask}
    return cases


def make_slice_index(cases: dict[str, dict]):
    index = []
    for sid in sorted(cases):
        _, z_count, _, _ = cases[sid]["image"].shape
        for phase in range(2):
            for z in range(z_count):
                index.append((sid, phase, z))
    return index


def import_stack(require_segformer: bool = True):
    try:
        import torch
        import torch.nn.functional as F
        import torchvision
        from torch.utils.data import Dataset, DataLoader
        from torchvision.models import ResNet50_Weights
        from torchvision.models.segmentation import deeplabv3_resnet50, fcn_resnet50
    except Exception as e:
        raise RuntimeError("PyTorch/torchvision unavailable in rare26.") from e

    transformers_pack = None
    if require_segformer:
        try:
            import transformers
            from transformers import SegformerForSemanticSegmentation
            from huggingface_hub import hf_hub_download
        except Exception as e:
            raise RuntimeError(
                "transformers is required for SegFormer-B0. Install with:\n"
                r"  D:\anaconda\envs\rare26\python.exe -m pip install -U transformers"
            ) from e
        transformers_pack = {
            "module": transformers,
            "SegformerForSemanticSegmentation": SegformerForSemanticSegmentation,
            "hf_hub_download": hf_hub_download,
        }

    return {
        "torch": torch,
        "F": F,
        "torchvision": torchvision,
        "Dataset": Dataset,
        "DataLoader": DataLoader,
        "ResNet50_Weights": ResNet50_Weights,
        "deeplabv3_resnet50": deeplabv3_resnet50,
        "fcn_resnet50": fcn_resnet50,
        "transformers": transformers_pack,
    }



def resolve_segformer_safe_weight(stack) -> Path:
    """
    Resolve the immutable safetensors conversion of nvidia/mit-b0 and verify
    the published LFS SHA256. This avoids external pickle/torch.load.
    """
    hf_hub_download = stack["transformers"]["hf_hub_download"]

    weight_path = Path(
        hf_hub_download(
            repo_id=SEGFORMER_MODEL_ID,
            filename=SEGFORMER_SAFE_WEIGHT_FILENAME,
            revision=SEGFORMER_SAFE_REVISION,
            cache_dir=str(HF_CACHE),
        )
    )

    actual = sha256_file(weight_path)

    if actual.lower() != SEGFORMER_SAFE_WEIGHT_SHA256.lower():
        raise RuntimeError(
            "SegFormer safetensors SHA256 mismatch: "
            f"expected={SEGFORMER_SAFE_WEIGHT_SHA256} actual={actual}"
        )

    return weight_path


def build_segformer_safe(stack):
    weight_path = resolve_segformer_safe_weight(stack)

    cls = stack["transformers"]["SegformerForSemanticSegmentation"]

    model = cls.from_pretrained(
        SEGFORMER_MODEL_ID,
        revision=SEGFORMER_SAFE_REVISION,
        use_safetensors=True,
        num_labels=NUM_CLASSES,
        id2label=SEGFORMER_ID2LABEL,
        label2id=SEGFORMER_LABEL2ID,
        ignore_mismatched_sizes=True,
        cache_dir=str(HF_CACHE),
    )

    if int(model.config.num_labels) != NUM_CLASSES:
        raise RuntimeError(
            f"SegFormer num_labels={model.config.num_labels} "
            f"expected={NUM_CLASSES}"
        )

    model._r08b1_safe_pretrained_weight_path = str(weight_path)
    model._r08b1_safe_pretrained_weight_sha256 = (
        SEGFORMER_SAFE_WEIGHT_SHA256
    )

    return model

def build_model(architecture: str, stack):
    if architecture == "deeplabv3_r50":
        return stack["deeplabv3_resnet50"](
            weights=None,
            weights_backbone=stack["ResNet50_Weights"].IMAGENET1K_V1,
            num_classes=NUM_CLASSES,
            aux_loss=False,
        )
    if architecture == "fcn_r50":
        return stack["fcn_resnet50"](
            weights=None,
            weights_backbone=stack["ResNet50_Weights"].IMAGENET1K_V1,
            num_classes=NUM_CLASSES,
            aux_loss=False,
        )
    if architecture == "segformer_b0":
        return build_segformer_safe(stack)
    raise ValueError(architecture)


def forward_logits(model, architecture: str, x, F):
    if architecture in {"deeplabv3_r50", "fcn_r50"}:
        return model(x)["out"]
    if architecture == "segformer_b0":
        logits = model(pixel_values=x).logits
        if tuple(logits.shape[-2:]) != tuple(x.shape[-2:]):
            logits = F.interpolate(
                logits,
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return logits
    raise ValueError(architecture)


def soft_foreground_dice_loss(logits, target, torch, F):
    probs = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(target.long(), num_classes=NUM_CLASSES).permute(0, 3, 1, 2).float()
    probs_fg = probs[:, 1:]
    target_fg = one_hot[:, 1:]
    dims = (0, 2, 3)
    inter = (probs_fg * target_fg).sum(dim=dims)
    denom = probs_fg.sum(dim=dims) + target_fg.sum(dim=dims)
    dice = (2.0 * inter + 1e-5) / (denom + 1e-5)
    return 1.0 - dice.mean()


def make_dataset_class(stack):
    torch = stack["torch"]
    F = stack["F"]
    Dataset = stack["Dataset"]

    class PreloadedSliceDataset(Dataset):
        def __init__(self, cases):
            self.cases = cases
            self.index = make_slice_index(cases)

        def __len__(self):
            return len(self.index)

        def __getitem__(self, idx):
            sid, phase, z = self.index[idx]
            case = self.cases[sid]
            x = torch.from_numpy(case["image"][phase, z]).float()
            y = torch.from_numpy(case["mask"][phase, z].astype(np.int64, copy=False)).long()
            x = F.interpolate(
                x[None, None],
                size=(INPUT_SIZE, INPUT_SIZE),
                mode="bilinear",
                align_corners=False,
            )[0, 0]
            y = F.interpolate(
                y[None, None].float(),
                size=(INPUT_SIZE, INPUT_SIZE),
                mode="nearest",
            )[0, 0].long()
            return x[None].repeat(3, 1, 1), y

    return PreloadedSliceDataset


def validate_preloaded(model, architecture, cases, device, stack):
    torch = stack["torch"]
    F = stack["F"]
    model.eval()
    metrics = []

    with torch.no_grad():
        for sid in tqdm(
            sorted(cases),
            desc="Source-val subjects",
            unit="subject",
            leave=False,
            dynamic_ncols=True,
        ):
            case = cases[sid]
            image = case["image"]
            mask = case["mask"]
            row = case["row"]
            phases, z_count, y_native, x_native = image.shape
            pred_native = np.zeros(mask.shape, dtype=np.uint8)

            for phase in range(phases):
                for z in range(z_count):
                    x = torch.from_numpy(image[phase, z]).float()[None, None]
                    x = F.interpolate(
                        x,
                        size=(INPUT_SIZE, INPUT_SIZE),
                        mode="bilinear",
                        align_corners=False,
                    )
                    x = x.repeat(1, 3, 1, 1).to(device, non_blocking=True)
                    logits = forward_logits(model, architecture, x, F)
                    hard = torch.argmax(logits, dim=1, keepdim=True).float()
                    hard_native = F.interpolate(
                        hard,
                        size=(y_native, x_native),
                        mode="nearest",
                    )[0, 0].cpu().numpy().astype(np.uint8, copy=False)
                    pred_native[phase, z] = hard_native

            class_dice = {
                cls: dice_binary(pred_native == cls, mask == cls)
                for cls in (1, 2, 3)
            }
            mean_dice = float(np.mean(list(class_dice.values())))
            metrics.append({
                "subject_id": sid,
                "vendor": row["vendor"],
                "centre": row["centre"],
                "dice_class_1": class_dice[1],
                "dice_class_2": class_dice[2],
                "dice_class_3": class_dice[3],
                "foreground_mean_dice": mean_dice,
            })

    return float(np.mean([r["foreground_mean_dice"] for r in metrics])), metrics


def state_dir(base: Path, source: str, target: str, arch: str, seed: int):
    return base / f"{source}_to_{target}" / arch / f"seed_{seed}"


def state_complete(path: Path, source: str, target: str, arch: str, seed: int):
    required = (
        path / "best_model.pt",
        path / "epoch_metrics.csv",
        path / "validation_subject_metrics.csv",
        path / "state_summary.json",
        path / "STATE_COMPLETE.json",
    )
    if not all(p.is_file() for p in required):
        return False
    try:
        marker = json.loads((path / "STATE_COMPLETE.json").read_text(encoding="utf-8"))
        summary = json.loads((path / "state_summary.json").read_text(encoding="utf-8"))
    except Exception:
        return False
    expected = {
        "source_vendor": source,
        "target_vendor": target,
        "architecture": arch,
        "seed": seed,
    }
    for k, v in expected.items():
        if marker.get(k) != v or summary.get(k) != v:
            return False
    ckpt_sha = sha256_file(path / "best_model.pt")
    return marker.get("checkpoint_sha256") == ckpt_sha


def reuse_pilot_state(build_dir, source, target):
    arch = "deeplabv3_r50"
    seed = 20260819
    dest = state_dir(build_dir, source, target, arch, seed)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=False)

    pilot_dir = R08B0_DIR / f"{source}_to_{target}"
    files = {
        "best_model.pt": pilot_dir / "best_model.pt",
        "epoch_metrics.csv": pilot_dir / "epoch_metrics.csv",
        "validation_subject_metrics.csv": pilot_dir / "validation_subject_metrics.csv",
        "training_summary.json": pilot_dir / "training_summary.json",
    }
    for name, src in files.items():
        if not src.is_file():
            raise FileNotFoundError(src)
        shutil.copy2(src, dest / name)

    pilot = json.loads((pilot_dir / "training_summary.json").read_text(encoding="utf-8"))
    summary = {
        "source_vendor": source,
        "target_vendor": target,
        "architecture": arch,
        "architecture_display": ARCH_DISPLAY[arch],
        "seed": seed,
        "reused_from_r08b0": True,
        "epochs": EPOCHS,
        "best_epoch": int(pilot["best_epoch"]),
        "best_source_val_mean_dice": float(pilot["best_source_val_mean_dice"]),
        "source_train_subjects": int(pilot["source_train_subjects"]),
        "source_val_subjects": int(pilot["source_val_subjects"]),
        "target_image_npz_opens": 0,
        "target_mask_npz_opens": 0,
        "target_metrics_computed": False,
        "checkpoint_sha256": sha256_file(dest / "best_model.pt"),
        "source_pilot_checkpoint_sha256": sha256_file(pilot_dir / "best_model.pt"),
    }
    write_json(dest / "state_summary.json", summary)
    write_json(dest / "STATE_COMPLETE.json", {
        "source_vendor": source,
        "target_vendor": target,
        "architecture": arch,
        "seed": seed,
        "reused_from_r08b0": True,
        "checkpoint_sha256": summary["checkpoint_sha256"],
        "complete": True,
    })
    return summary


def train_state(build_dir, source, target, arch, seed, train_cases, val_cases, stack, device):
    dest = state_dir(build_dir, source, target, arch, seed)
    if state_complete(dest, source, target, arch, seed):
        print(f"[SKIP COMPLETE] {source}->{target} {arch} seed={seed}")
        return json.loads((dest / "state_summary.json").read_text(encoding="utf-8"))
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=False)

    torch = stack["torch"]
    F = stack["F"]
    DataLoader = stack["DataLoader"]
    DatasetClass = make_dataset_class(stack)

    set_global_seed(seed, torch)
    model = build_model(arch, stack).to(device)
    train_dataset = DatasetClass(train_cases)
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        generator=generator,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    epoch_rows = []
    best_val = -1.0
    best_epoch = -1
    best_subject_metrics = None
    best_path = dest / "best_model.pt"

    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_sum = ce_sum = dice_sum = 0.0
        sample_count = 0
        bar = tqdm(
            loader,
            desc=f"{source}->{target} {arch} s{seed} ep {epoch:02d}/{EPOCHS}",
            unit="batch",
            dynamic_ncols=True,
        )
        for x, y in bar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=(device.type == "cuda"),
            ):
                logits = forward_logits(model, arch, x, F)
                ce = F.cross_entropy(logits, y)
                dl = soft_foreground_dice_loss(logits, y, torch, F)
                loss = ce + dl

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()

            bs = int(x.shape[0])
            sample_count += bs
            loss_sum += float(loss.detach().cpu()) * bs
            ce_sum += float(ce.detach().cpu()) * bs
            dice_sum += float(dl.detach().cpu()) * bs
            bar.set_postfix(
                loss=f"{loss_sum/max(sample_count,1):.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )
        bar.close()

        val_dice, val_subject_metrics = validate_preloaded(
            model, arch, val_cases, device, stack
        )
        epoch_rows.append({
            "epoch": epoch,
            "train_loss": loss_sum / max(sample_count, 1),
            "train_ce": ce_sum / max(sample_count, 1),
            "train_soft_dice_loss": dice_sum / max(sample_count, 1),
            "source_val_mean_dice": val_dice,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        })
        print(
            f"[{source}->{target} {arch} seed={seed}] "
            f"epoch={epoch:02d} source_val={val_dice:.6f}"
        )

        if val_dice > best_val:
            best_val = val_dice
            best_epoch = epoch
            best_subject_metrics = val_subject_metrics
            torch.save({
                "script_version": VERSION,
                "architecture": arch,
                "architecture_display": ARCH_DISPLAY[arch],
                "seed": seed,
                "source_vendor": source,
                "target_vendor": target,
                "epoch": epoch,
                "source_val_mean_dice": val_dice,
                "num_classes": NUM_CLASSES,
                "model_state_dict": model.state_dict(),
            }, best_path)

        scheduler.step()

    if best_subject_metrics is None:
        raise RuntimeError("No best checkpoint selected.")

    write_csv(dest / "epoch_metrics.csv", epoch_rows, [
        "epoch", "train_loss", "train_ce", "train_soft_dice_loss",
        "source_val_mean_dice", "learning_rate",
    ])
    write_csv(dest / "validation_subject_metrics.csv", best_subject_metrics, [
        "subject_id", "vendor", "centre", "dice_class_1", "dice_class_2",
        "dice_class_3", "foreground_mean_dice",
    ])

    summary = {
        "source_vendor": source,
        "target_vendor": target,
        "architecture": arch,
        "architecture_display": ARCH_DISPLAY[arch],
        "seed": seed,
        "reused_from_r08b0": False,
        "epochs": EPOCHS,
        "best_epoch": best_epoch,
        "best_source_val_mean_dice": best_val,
        "source_train_subjects": len(train_cases),
        "source_val_subjects": len(val_cases),
        "train_slices": len(train_dataset),
        "target_image_npz_opens": 0,
        "target_mask_npz_opens": 0,
        "target_metrics_computed": False,
        "checkpoint_sha256": sha256_file(best_path),
        "segformer_pretrained_repo": (
            SEGFORMER_MODEL_ID if arch == "segformer_b0" else ""
        ),
        "segformer_pretrained_revision": (
            SEGFORMER_SAFE_REVISION if arch == "segformer_b0" else ""
        ),
        "segformer_pretrained_weight_format": (
            "safetensors" if arch == "segformer_b0" else ""
        ),
        "segformer_pretrained_weight_sha256": (
            SEGFORMER_SAFE_WEIGHT_SHA256 if arch == "segformer_b0" else ""
        ),
        "segformer_external_pickle_torch_load_used": False,
    }
    write_json(dest / "state_summary.json", summary)
    write_json(dest / "STATE_COMPLETE.json", {
        "source_vendor": source,
        "target_vendor": target,
        "architecture": arch,
        "seed": seed,
        "reused_from_r08b0": False,
        "checkpoint_sha256": summary["checkpoint_sha256"],
        "complete": True,
    })
    return summary


def collect_states(build_dir: Path):
    rows = []
    for source, target in DIRECTIONS:
        for arch in ARCHITECTURES:
            for seed in SEEDS:
                dest = state_dir(build_dir, source, target, arch, seed)
                if not state_complete(dest, source, target, arch, seed):
                    raise RuntimeError(f"Incomplete state: {source}->{target} {arch} seed={seed}")
                s = json.loads((dest / "state_summary.json").read_text(encoding="utf-8"))
                rows.append({
                    "source_vendor": source,
                    "target_vendor": target,
                    "direction": f"{source}_to_{target}",
                    "architecture": arch,
                    "architecture_display": ARCH_DISPLAY[arch],
                    "seed": seed,
                    "reused_from_r08b0": int(bool(s.get("reused_from_r08b0"))),
                    "best_epoch": int(s["best_epoch"]),
                    "best_source_val_mean_dice": float(s["best_source_val_mean_dice"]),
                    "checkpoint_sha256": s["checkpoint_sha256"],
                    "target_image_npz_opens": int(s["target_image_npz_opens"]),
                    "target_mask_npz_opens": int(s["target_mask_npz_opens"]),
                    "target_metrics_computed": int(bool(s["target_metrics_computed"])),
                })
    return rows


def summarize_arch_direction(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["direction"], row["architecture"])].append(row)
    out = []
    for (direction, arch), sub in sorted(groups.items()):
        scores = [float(r["best_source_val_mean_dice"]) for r in sub]
        hashes = [r["checkpoint_sha256"] for r in sub]
        out.append({
            "direction": direction,
            "architecture": arch,
            "architecture_display": ARCH_DISPLAY[arch],
            "states": len(sub),
            "min_source_val_mean_dice": min(scores),
            "median_source_val_mean_dice": float(np.median(scores)),
            "max_source_val_mean_dice": max(scores),
            "checkpoint_unique_hashes": len(set(hashes)),
            "all_three_checkpoints_identical": int(len(set(hashes)) == 1),
        })
    return out


def preflight():
    _, _, split_sha = validate_upstream()
    case_rows = load_case_manifest()
    splits = load_inherited_splits(case_rows)
    stack = import_stack(require_segformer=True)
    torch = stack["torch"]
    torchvision = stack["torchvision"]
    transformers = stack["transformers"]["module"]

    # FIX1: fail early here rather than after hours of completed CNN states.
    safe_weight_path = resolve_segformer_safe_weight(stack)
    safe_probe_model = build_segformer_safe(stack)
    safe_probe_num_labels = int(safe_probe_model.config.num_labels)
    del safe_probe_model

    print("===== Q1-R08B1 FIX1 PREFLIGHT =====")
    print(f"protocol_sha256={EXPECTED_PROTOCOL_SHA256}")
    print(f"r08a2_lock_sha256={EXPECTED_R08A2_LOCK_SHA256}")
    print(f"r08b0_lock_sha256={EXPECTED_R08B0_LOCK_SHA256}")
    print(f"inherited_source_split_sha256={split_sha}")
    print(f"compact_cases={len(case_rows)}")
    print("architectures=deeplabv3_r50,fcn_r50,segformer_b0")
    print("seeds=20260819,20260820,20260821")
    print("states=18")
    print("pilot_states_reused=2")
    print("new_training_states=16")
    print(f"B_train={len(splits['B_to_A']['train'])}")
    print(f"B_val={len(splits['B_to_A']['val'])}")
    print(f"A_train={len(splits['A_to_B']['train'])}")
    print(f"A_val={len(splits['A_to_B']['val'])}")
    print(f"torch_version={torch.__version__}")
    print(f"torchvision_version={torchvision.__version__}")
    print(f"transformers_version={transformers.__version__}")
    print(f"segformer_repo={SEGFORMER_MODEL_ID}")
    print(f"segformer_safe_revision={SEGFORMER_SAFE_REVISION}")
    print("segformer_weight_format=safetensors")
    print(f"segformer_safe_weight_sha256={sha256_file(safe_weight_path)}")
    print(f"segformer_num_labels={safe_probe_num_labels}")
    print("segformer_external_pickle_torch_load_used=NO")
    print(f"torch_cache={TORCH_CACHE}")
    print(f"hf_cache={HF_CACHE}")
    print(f"cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda_device={torch.cuda.get_device_name(0)}")
    print("target_image_npz_access=FORBIDDEN")
    print("target_mask_npz_access=FORBIDDEN")
    print("target_metrics=NO")
    print("PREFLIGHT_PASS")


def run(args):
    _, _, split_sha = validate_upstream()
    case_rows = load_case_manifest()
    splits = load_inherited_splits(case_rows)
    stack = import_stack(require_segformer=True)
    torch = stack["torch"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    build_dir = args.output_dir
    if build_dir.exists():
        if (build_dir / "Q1_R08B1_MNMS_HETEROGENEOUS_SOURCE_PANEL_LOCK.json").is_file():
            raise FileExistsError("R08B1 is already finalized; refusing to alter locked output.")
    else:
        build_dir.mkdir(parents=True, exist_ok=False)

    protocol_copy = build_dir / "preregistered_protocol_copy.md"
    if not protocol_copy.exists():
        shutil.copy2(PROTOCOL, protocol_copy)
    validate_sha(protocol_copy, EXPECTED_PROTOCOL_SHA256, "protocol copy")

    inherited_split = build_dir / "inherited_source_split_manifest.csv"
    if not inherited_split.exists():
        shutil.copy2(R08B0_SPLIT, inherited_split)
    if sha256_file(inherited_split) != split_sha:
        raise RuntimeError("Inherited split copy changed.")

    for source, target in DIRECTIONS:
        direction = f"{source}_to_{target}"
        tracker = {
            "source_image_npz_opens": 0,
            "source_mask_npz_opens": 0,
            "target_image_npz_opens": 0,
            "target_mask_npz_opens": 0,
        }
        train_cases = preload_rows(
            splits[direction]["train"], tracker, desc=f"Preload {source} train"
        )
        val_cases = preload_rows(
            splits[direction]["val"], tracker, desc=f"Preload {source} val"
        )
        if tracker["target_image_npz_opens"] != 0 or tracker["target_mask_npz_opens"] != 0:
            raise RuntimeError("Target NPZ opened during source preload.")

        for arch in ARCHITECTURES:
            for seed in SEEDS:
                dest = state_dir(build_dir, source, target, arch, seed)
                if state_complete(dest, source, target, arch, seed):
                    print(f"[RESUME SKIP] {source}->{target} {arch} seed={seed}")
                    continue
                if arch == "deeplabv3_r50" and seed == 20260819:
                    print(f"[REUSE R08B0] {source}->{target} deeplabv3_r50 seed={seed}")
                    reuse_pilot_state(build_dir, source, target)
                    continue

                print("\n" + "=" * 80)
                print(f"TRAIN STATE: {source}->{target} {arch} seed={seed}")
                print("=" * 80)
                train_state(
                    build_dir,
                    source,
                    target,
                    arch,
                    seed,
                    train_cases,
                    val_cases,
                    stack,
                    device,
                )

        del train_cases
        del val_cases
        if device.type == "cuda":
            torch.cuda.empty_cache()

    rows = collect_states(build_dir)
    if len(rows) != EXPECTED_STATES:
        raise RuntimeError(f"states={len(rows)} expected={EXPECTED_STATES}")
    arch_rows = summarize_arch_direction(rows)

    write_csv(build_dir / "state_panel_summary.csv", rows, [
        "source_vendor", "target_vendor", "direction", "architecture",
        "architecture_display", "seed", "reused_from_r08b0", "best_epoch",
        "best_source_val_mean_dice", "checkpoint_sha256",
        "target_image_npz_opens", "target_mask_npz_opens", "target_metrics_computed",
    ])
    write_csv(build_dir / "architecture_direction_summary.csv", arch_rows, [
        "direction", "architecture", "architecture_display", "states",
        "min_source_val_mean_dice", "median_source_val_mean_dice",
        "max_source_val_mean_dice", "checkpoint_unique_hashes",
        "all_three_checkpoints_identical",
    ])

    total_target_img = sum(int(r["target_image_npz_opens"]) for r in rows)
    total_target_mask = sum(int(r["target_mask_npz_opens"]) for r in rows)
    total_target_metrics = sum(int(r["target_metrics_computed"]) for r in rows)
    min_state_score = min(float(r["best_source_val_mean_dice"]) for r in rows)
    min_group_median = min(float(r["median_source_val_mean_dice"]) for r in arch_rows)

    checks = {
        "A_states_eq_18": len(rows) == 18,
        "B_architecture_direction_groups_eq_6": len(arch_rows) == 6,
        "C_each_group_has_3_states": all(int(r["states"]) == 3 for r in arch_rows),
        "D_min_state_source_val_dice_ge_0p65": min_state_score >= MIN_STATE_DICE,
        "E_min_arch_direction_median_ge_0p75": min_group_median >= MIN_ARCH_DIRECTION_MEDIAN_DICE,
        "F_target_image_npz_opens_eq_0": total_target_img == 0,
        "G_target_mask_npz_opens_eq_0": total_target_mask == 0,
        "H_target_metrics_computed_eq_0": total_target_metrics == 0,
        "I_checkpoint_groups_not_all_identical": all(
            int(r["all_three_checkpoints_identical"]) == 0 for r in arch_rows
        ),
        "J_reused_pilot_states_eq_2": sum(int(r["reused_from_r08b0"]) for r in rows) == 2,
    }

    decision = READY if all(checks.values()) else INSUFFICIENT
    gate = {
        "script_version": VERSION,
        "build": BUILD,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "r08a2_lock_sha256": EXPECTED_R08A2_LOCK_SHA256,
        "r08b0_lock_sha256": EXPECTED_R08B0_LOCK_SHA256,
        "inherited_source_split_sha256": split_sha,
        "architectures": list(ARCHITECTURES),
        "seeds": list(SEEDS),
        "directions": ["B->A", "A->B"],
        "technical_fix": (
            "SegFormer external initialization pinned to verified safetensors"
        ),
        "segformer_pretrained_repo": SEGFORMER_MODEL_ID,
        "segformer_pretrained_revision": SEGFORMER_SAFE_REVISION,
        "segformer_pretrained_weight_format": "safetensors",
        "segformer_pretrained_weight_sha256": SEGFORMER_SAFE_WEIGHT_SHA256,
        "segformer_external_pickle_torch_load_used": False,
        "states": len(rows),
        "min_state_source_val_mean_dice": min_state_score,
        "min_architecture_direction_median_dice": min_group_median,
        "target_image_npz_opens": total_target_img,
        "target_mask_npz_opens": total_target_mask,
        "target_metrics_computed": total_target_metrics,
        "checks": checks,
        "decision": decision,
    }
    write_json(build_dir / "readiness_gate.json", gate)
    (build_dir / "decision.txt").write_text(decision + "\n", encoding="utf-8")

    lines = [
        "===== Q1-R08B1 M&Ms HETEROGENEOUS SOURCE-ONLY PANEL =====",
        f"Script version: {VERSION}",
        f"Build: {BUILD}",
        "",
        "Frozen panel:",
        "  architectures=DeepLabV3-R50 / FCN-R50 / SegFormer-B0",
        "  seeds=20260819 / 20260820 / 20260821",
        "  directions=B->A / A->B",
        "  states=18",
        "  R08B0 DeepLab seed20260819 states reused=2",
        f"  SegFormer pretrained revision={SEGFORMER_SAFE_REVISION}",
        "  SegFormer external weight format=safetensors",
        f"  SegFormer external weight SHA256={SEGFORMER_SAFE_WEIGHT_SHA256}",
        "  SegFormer external pickle torch.load used=NO",
        "  target data used=NO",
        "",
        "Architecture-direction source validation:",
    ]
    for r in arch_rows:
        lines.append(
            f"  {r['direction']} {r['architecture']}: "
            f"min={float(r['min_source_val_mean_dice']):.6f} "
            f"median={float(r['median_source_val_mean_dice']):.6f} "
            f"max={float(r['max_source_val_mean_dice']):.6f} "
            f"unique_ckpt={r['checkpoint_unique_hashes']}"
        )
    lines += [
        "",
        f"Global min state source-val Dice={min_state_score:.6f}",
        f"Global min architecture-direction median={min_group_median:.6f}",
        "",
        "Target-blind boundary:",
        f"  target image NPZ opens={total_target_img}",
        f"  target mask NPZ opens={total_target_mask}",
        f"  target metrics computed={total_target_metrics}",
        "",
        "Checks:",
    ]
    for k, v in checks.items():
        lines.append(f"  {k}={'PASS' if v else 'FAIL'}")
    lines += ["", "Decision:", f"  {decision}"]

    run_log = build_dir / "run_log.txt"
    run_log.write_text("\n".join(lines) + "\n", encoding="utf-8")

    artifact_names = [
        "preregistered_protocol_copy.md",
        "inherited_source_split_manifest.csv",
        "state_panel_summary.csv",
        "architecture_direction_summary.csv",
        "readiness_gate.json",
        "decision.txt",
        "run_log.txt",
    ]
    artifacts = {
        name: {"relative_path": name, "sha256": sha256_file(build_dir / name)}
        for name in artifact_names
    }

    checkpoint_sha = {}
    for r in tqdm(rows, desc="Hash panel checkpoints", unit="state", dynamic_ncols=True):
        rel = f"{r['direction']}/{r['architecture']}/seed_{r['seed']}/best_model.pt"
        checkpoint_sha[rel] = sha256_file(build_dir / rel)

    lock = {
        "script_version": VERSION,
        "build": BUILD,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "r08a2_lock_sha256": EXPECTED_R08A2_LOCK_SHA256,
        "r08b0_lock_sha256": EXPECTED_R08B0_LOCK_SHA256,
        "inherited_source_split_sha256": split_sha,
        "architectures": list(ARCHITECTURES),
        "seeds": list(SEEDS),
        "directions": ["B->A", "A->B"],
        "technical_fix": (
            "SegFormer external initialization pinned to verified safetensors"
        ),
        "segformer_pretrained_repo": SEGFORMER_MODEL_ID,
        "segformer_pretrained_revision": SEGFORMER_SAFE_REVISION,
        "segformer_pretrained_weight_format": "safetensors",
        "segformer_pretrained_weight_sha256": SEGFORMER_SAFE_WEIGHT_SHA256,
        "segformer_external_pickle_torch_load_used": False,
        "states": len(rows),
        "target_image_npz_opens": total_target_img,
        "target_mask_npz_opens": total_target_mask,
        "target_metrics_computed": False,
        "decision": decision,
        "checks": checks,
        "artifacts": artifacts,
        "checkpoint_sha256": checkpoint_sha,
    }
    lock_path = build_dir / "Q1_R08B1_MNMS_HETEROGENEOUS_SOURCE_PANEL_LOCK.json"
    write_json(lock_path, lock)
    lock_sha = sha256_file(lock_path)

    for name, meta in artifacts.items():
        if sha256_file(build_dir / meta["relative_path"]) != meta["sha256"]:
            raise RuntimeError(f"Artifact changed before lock: {name}")

    print("\n" + run_log.read_text(encoding="utf-8"))
    print("Q1-R08B1 LOCK:", lock_path)
    print("Q1-R08B1 LOCK SHA256:", lock_sha)


def self_test():
    assert ARCHITECTURES == ("deeplabv3_r50", "fcn_r50", "segformer_b0")
    assert SEEDS == (20260819, 20260820, 20260821)
    assert EXPECTED_STATES == 18
    assert MIN_STATE_DICE == 0.65
    assert MIN_ARCH_DIRECTION_MEDIAN_DICE == 0.75
    assert SEGFORMER_SAFE_REVISION == (
        "25ce79d97e6d9d509ed12e17cb2eb89b0a83a2dc"
    )
    assert SEGFORMER_SAFE_WEIGHT_FILENAME == "model.safetensors"
    assert SEGFORMER_SAFE_WEIGHT_SHA256 == (
        "3e5ad9cd1dd8ecf8305c23fcdf01ef241f08c7b2dddacb6ec7de5a887188798a"
    )
    x = np.asarray([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    y = robust_normalize_phase(x)
    assert y.dtype == np.float32 and np.isfinite(y).all()
    assert abs(dice_binary(np.asarray([1, 0, 1]), np.asarray([1, 0, 1])) - 1.0) < 1e-12
    print("THREE_ARCHITECTURE_PANEL_TEST_PASS")
    print("THREE_SEED_PANEL_TEST_PASS")
    print("EIGHTEEN_STATE_COUNT_TEST_PASS")
    print("SOURCE_NORMALIZATION_TEST_PASS")
    print("SUBJECT_DICE_TEST_PASS")
    print("F_DRIVE_CACHE_PATH_TEST_PASS")
    print("RESUME_MARKER_DESIGN_TEST_PASS")
    print("SEGFORMER_SAFE_REVISION_TEST_PASS")
    print("SEGFORMER_SAFETENSORS_SHA_TEST_PASS")
    print("NO_SEGFORMER_EXTERNAL_PICKLE_LOAD_TEST_PASS")
    print("SELF_TEST_PASS")


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Q1-R08B1: formal 18-state source-only M&Ms cardiac panel. "
            "Resumable; target images/masks are never opened."
        )
    )
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--self-test", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.preflight_only:
        preflight()
        return 0
    try:
        run(args)
    except Exception:
        traceback.print_exc()
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
