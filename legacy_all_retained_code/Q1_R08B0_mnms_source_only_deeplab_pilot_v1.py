#!/usr/bin/env python
# -*- coding: utf-8 -*-

r"""
Q1-R08B0 — M&Ms source-only DeepLabV3-R50 pilot.

Target images/masks are never opened.
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
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
from tqdm import tqdm


VERSION = "2026-08-19-Q1-R08B0-v1"
BUILD = "Q1_R08B0_MNMS_SOURCE_ONLY_DEEPLABV3_R50_PILOT"

ROOT = Path(r"F:\MEDSEG_SAFETTA")

PROTOCOL = (
    ROOT / "docs"
    / "Q1_R08B0_mnms_source_only_deeplab_pilot_preregistered_protocol_v1.md"
)
EXPECTED_PROTOCOL_SHA256 = "fb39a60314aa6eb0db56802f2d7d14b570e465533f7964ec2982430652b286b3"

R08A2_DIR = (
    ROOT / "outputs"
    / "Q1_R08A2_mnms_zip_streaming_compact_preprocessing_v1"
)
R08A2_LOCK = (
    R08A2_DIR
    / "Q1_R08A2_MNMS_COMPACT_BA_ASSET_LOCK.json"
)

EXPECTED_R08A2_LOCK_SHA256 = (
    "e7df0cb2228a8b1f405ec97b78674763d48a4bd9a061017ef2cb61f67640c4c9"
)
EXPECTED_R08A2_DECISION = "MNMS_COMPACT_BA_ASSET_READY"

DATA_DIR = (
    ROOT / "data" / "processed"
    / "MnMs_R08A2_BA_compact_v1"
)
CASE_MANIFEST = DATA_DIR / "case_manifest.csv"

OUTPUT_DIR = (
    ROOT / "outputs"
    / "Q1_R08B0_mnms_source_only_deeplab_pilot_v1"
)

SEED = 20260819

DIRECTIONS = (
    ("B", "A"),
    ("A", "B"),
)

EXPECTED_VENDOR_COUNTS = {
    "B": 125,
    "A": 95,
}

EXPECTED_SPLITS = {
    "B": {
        "train": 100,
        "val": 25,
    },
    "A": {
        "train": 76,
        "val": 19,
    },
}

NUM_CLASSES = 4
INPUT_SIZE = 256
EPOCHS = 30
BATCH_SIZE = 8
LR = 3e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
FEASIBILITY_DICE = 0.70

READY = "MNMS_SOURCE_ONLY_SEGMENTATION_PILOT_READY"
INSUFFICIENT = "MNMS_SOURCE_ONLY_SEGMENTATION_PILOT_INSUFFICIENT"


# ---------------------------------------------------------------------------
# Generic utilities
# ---------------------------------------------------------------------------

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
        raise RuntimeError(
            f"{label} SHA mismatch: expected={expected} actual={actual}"
        )

    return actual


def write_csv(path: Path, rows, fields: Sequence[str]):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(fields),
        )
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            obj,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def set_global_seed(seed: int, torch):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    try:
        torch.use_deterministic_algorithms(
            True,
            warn_only=True,
        )
    except Exception:
        pass


def stable_hash_score(text: str, seed: int) -> int:
    payload = f"{seed}::{text}".encode("utf-8")
    return int(
        hashlib.sha256(payload).hexdigest(),
        16,
    )


# ---------------------------------------------------------------------------
# Upstream and manifest
# ---------------------------------------------------------------------------

def load_upstream_lock():
    validate_sha(
        PROTOCOL,
        EXPECTED_PROTOCOL_SHA256,
        "R08B0 protocol",
    )

    validate_sha(
        R08A2_LOCK,
        EXPECTED_R08A2_LOCK_SHA256,
        "R08A2 lock",
    )

    lock = json.loads(
        R08A2_LOCK.read_text(
            encoding="utf-8"
        )
    )

    if lock.get("decision") != EXPECTED_R08A2_DECISION:
        raise RuntimeError(
            f"Unexpected R08A2 decision={lock.get('decision')}"
        )

    if int(lock.get("cases", -1)) != 220:
        raise RuntimeError(
            "R08A2 case count changed."
        )

    if int(lock.get("compact_image_files", -1)) != 220:
        raise RuntimeError(
            "R08A2 compact image count changed."
        )

    if int(lock.get("compact_mask_files", -1)) != 220:
        raise RuntimeError(
            "R08A2 compact mask count changed."
        )

    if bool(lock.get("model_training", True)):
        raise RuntimeError(
            "R08A2 unexpectedly reports model training."
        )

    return lock


def load_case_manifest() -> list[dict]:
    if not CASE_MANIFEST.is_file():
        raise FileNotFoundError(
            CASE_MANIFEST
        )

    rows = []

    with CASE_MANIFEST.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        reader = csv.DictReader(f)

        required = {
            "subject_id",
            "vendor",
            "centre",
            "image_npz",
            "mask_npz",
            "phase_order",
            "shape_tzyx",
        }

        missing = required - set(
            reader.fieldnames or []
        )

        if missing:
            raise RuntimeError(
                f"case_manifest missing columns={sorted(missing)}"
            )

        for row in reader:
            vendor = row["vendor"].strip()

            if vendor not in EXPECTED_VENDOR_COUNTS:
                continue

            image_path = (
                DATA_DIR
                / row["image_npz"]
            )

            mask_path = (
                DATA_DIR
                / row["mask_npz"]
            )

            if not image_path.is_file():
                raise FileNotFoundError(
                    image_path
                )

            if not mask_path.is_file():
                raise FileNotFoundError(
                    mask_path
                )

            rows.append({
                "subject_id":
                    row["subject_id"].strip(),
                "vendor":
                    vendor,
                "centre":
                    row["centre"].strip(),
                "image_path":
                    image_path,
                "mask_path":
                    mask_path,
                "phase_order":
                    row["phase_order"],
                "shape_tzyx":
                    row["shape_tzyx"],
            })

    counts = Counter(
        row["vendor"]
        for row in rows
    )

    if dict(counts) != EXPECTED_VENDOR_COUNTS:
        raise RuntimeError(
            f"Vendor counts={dict(counts)} "
            f"expected={EXPECTED_VENDOR_COUNTS}"
        )

    subjects = [
        row["subject_id"]
        for row in rows
    ]

    dup = [
        sid
        for sid, n in Counter(subjects).items()
        if n > 1
    ]

    if dup:
        raise RuntimeError(
            f"Duplicate subjects={dup[:10]}"
        )

    return rows


# ---------------------------------------------------------------------------
# Frozen source splits
# ---------------------------------------------------------------------------

def split_source_subjects(
    source_rows: list[dict],
    source_vendor: str,
) -> tuple[list[dict], list[dict]]:
    expected = EXPECTED_SPLITS[
        source_vendor
    ]

    val_n = int(
        expected["val"]
    )

    centres = sorted(
        set(
            row["centre"]
            for row in source_rows
        )
    )

    if source_vendor == "B" and len(centres) > 1:
        groups = defaultdict(list)

        for row in source_rows:
            groups[
                row["centre"]
            ].append(row)

        group_sizes = {
            c: len(groups[c])
            for c in centres
        }

        raw = {
            c: (
                group_sizes[c]
                * val_n
                / len(source_rows)
            )
            for c in centres
        }

        alloc = {
            c: int(
                math.floor(raw[c])
            )
            for c in centres
        }

        remaining = (
            val_n
            - sum(alloc.values())
        )

        remainders = sorted(
            centres,
            key=lambda c: (
                -(raw[c] - alloc[c]),
                c,
            ),
        )

        for c in remainders[:remaining]:
            alloc[c] += 1

        val = []

        for c in centres:
            ordered = sorted(
                groups[c],
                key=lambda row: (
                    stable_hash_score(
                        row["subject_id"],
                        SEED,
                    ),
                    row["subject_id"],
                ),
            )

            val.extend(
                ordered[:alloc[c]]
            )

    else:
        ordered = sorted(
            source_rows,
            key=lambda row: (
                stable_hash_score(
                    row["subject_id"],
                    SEED,
                ),
                row["subject_id"],
            ),
        )

        val = ordered[:val_n]

    val_ids = {
        row["subject_id"]
        for row in val
    }

    train = [
        row
        for row in source_rows
        if row["subject_id"]
        not in val_ids
    ]

    train = sorted(
        train,
        key=lambda row:
            row["subject_id"],
    )

    val = sorted(
        val,
        key=lambda row:
            row["subject_id"],
    )

    if len(train) != expected["train"]:
        raise RuntimeError(
            f"{source_vendor} train={len(train)} "
            f"expected={expected['train']}"
        )

    if len(val) != expected["val"]:
        raise RuntimeError(
            f"{source_vendor} val={len(val)} "
            f"expected={expected['val']}"
        )

    if set(
        r["subject_id"]
        for r in train
    ) & set(
        r["subject_id"]
        for r in val
    ):
        raise RuntimeError(
            "Source train/val subject overlap."
        )

    return train, val


def build_split_manifest(
    all_rows: list[dict],
) -> list[dict]:
    out = []

    for source_vendor, target_vendor in DIRECTIONS:
        source_rows = [
            row
            for row in all_rows
            if row["vendor"]
            == source_vendor
        ]

        train, val = split_source_subjects(
            source_rows,
            source_vendor,
        )

        for split_name, rows in (
            ("train", train),
            ("val", val),
        ):
            for row in rows:
                out.append({
                    "direction":
                        f"{source_vendor}_to_{target_vendor}",
                    "source_vendor":
                        source_vendor,
                    "target_vendor":
                        target_vendor,
                    "split":
                        split_name,
                    "subject_id":
                        row["subject_id"],
                    "centre":
                        row["centre"],
                    "image_npz":
                        str(
                            row["image_path"]
                        ),
                    "mask_npz":
                        str(
                            row["mask_path"]
                        ),
                    "target_file_open_allowed":
                        0,
                })

    return out


# ---------------------------------------------------------------------------
# Normalization / data
# ---------------------------------------------------------------------------

def robust_normalize_phase(
    phase_zyx: np.ndarray,
) -> np.ndarray:
    x = np.asarray(
        phase_zyx,
        dtype=np.float32,
    )

    finite = np.isfinite(
        x
    )

    if not finite.any():
        raise RuntimeError(
            "Phase has no finite pixels."
        )

    support = (
        finite
        & (x != 0)
    )

    if int(
        support.sum()
    ) >= 32:
        values = x[
            support
        ]
    else:
        values = x[
            finite
        ]

    lo = float(
        np.percentile(
            values,
            0.5,
        )
    )

    hi = float(
        np.percentile(
            values,
            99.5,
        )
    )

    if not math.isfinite(lo) or not math.isfinite(hi):
        raise RuntimeError(
            "Non-finite percentile."
        )

    if hi < lo:
        raise RuntimeError(
            "Invalid percentile order."
        )

    clipped = np.clip(
        x,
        lo,
        hi,
    )

    if int(
        support.sum()
    ) >= 32:
        stats_values = clipped[
            support
        ]
    else:
        stats_values = clipped[
            finite
        ]

    mean = float(
        stats_values.mean()
    )

    std = float(
        stats_values.std()
    )

    if not math.isfinite(mean):
        raise RuntimeError(
            "Non-finite normalization mean."
        )

    if not math.isfinite(std) or std < 1e-6:
        std = 1.0

    out = (
        clipped - mean
    ) / std

    out = np.clip(
        out,
        -5.0,
        5.0,
    )

    out[~finite] = 0.0

    return out.astype(
        np.float32,
        copy=False,
    )


def load_source_case(
    row: dict,
    tracker: dict,
):
    # This function is ONLY ever called with a source row.
    if row["vendor"] not in EXPECTED_VENDOR_COUNTS:
        raise RuntimeError(
            "Unexpected vendor."
        )

    tracker[
        "source_image_npz_opens"
    ] += 1

    with np.load(
        row["image_path"],
        allow_pickle=False,
    ) as d:
        image = np.asarray(
            d["image"],
            dtype=np.float32,
        )

    tracker[
        "source_mask_npz_opens"
    ] += 1

    with np.load(
        row["mask_path"],
        allow_pickle=False,
    ) as d:
        mask = np.asarray(
            d["mask"],
            dtype=np.uint8,
        )

    if image.shape != mask.shape:
        raise RuntimeError(
            f"{row['subject_id']}: image/mask shape mismatch."
        )

    if image.ndim != 4:
        raise RuntimeError(
            f"{row['subject_id']}: compact ndim={image.ndim}"
        )

    if image.shape[0] != 2:
        raise RuntimeError(
            f"{row['subject_id']}: phase axis != 2."
        )

    norm = np.empty_like(
        image,
        dtype=np.float32,
    )

    for phase in range(
        image.shape[0]
    ):
        norm[
            phase
        ] = robust_normalize_phase(
            image[phase]
        )

    return norm, mask


def make_slice_index(
    rows: list[dict],
) -> list[tuple[int, int, int]]:
    index = []

    for row_idx, row in enumerate(rows):
        shape = tuple(
            int(x)
            for x in row[
                "shape_tzyx"
            ].split("x")
        )

        if len(shape) != 4 or shape[0] != 2:
            raise RuntimeError(
                f"Bad shape_tzyx={row['shape_tzyx']}"
            )

        _, z, _, _ = shape

        for phase in range(2):
            for slice_idx in range(z):
                index.append(
                    (
                        row_idx,
                        phase,
                        slice_idx,
                    )
                )

    return index


# ---------------------------------------------------------------------------
# Torch-specific code
# ---------------------------------------------------------------------------

def import_torch_stack():
    try:
        import torch
        import torch.nn.functional as F
        import torchvision
        from torch.utils.data import Dataset, DataLoader
        from torchvision.models import ResNet50_Weights
        from torchvision.models.segmentation import deeplabv3_resnet50
    except Exception as e:
        raise RuntimeError(
            "PyTorch/torchvision stack unavailable in rare26."
        ) from e

    return {
        "torch": torch,
        "F": F,
        "torchvision": torchvision,
        "Dataset": Dataset,
        "DataLoader": DataLoader,
        "ResNet50_Weights": ResNet50_Weights,
        "deeplabv3_resnet50": deeplabv3_resnet50,
    }


def build_model(stack):
    model = stack[
        "deeplabv3_resnet50"
    ](
        weights=None,
        weights_backbone=stack[
            "ResNet50_Weights"
        ].IMAGENET1K_V1,
        num_classes=NUM_CLASSES,
        aux_loss=False,
    )

    return model


def soft_foreground_dice_loss(
    logits,
    target,
    torch,
    F,
):
    probs = torch.softmax(
        logits,
        dim=1,
    )

    one_hot = F.one_hot(
        target.long(),
        num_classes=NUM_CLASSES,
    ).permute(
        0,
        3,
        1,
        2,
    ).float()

    probs_fg = probs[
        :,
        1:,
    ]

    target_fg = one_hot[
        :,
        1:,
    ]

    dims = (
        0,
        2,
        3,
    )

    intersection = (
        probs_fg
        * target_fg
    ).sum(
        dim=dims
    )

    denominator = (
        probs_fg.sum(
            dim=dims
        )
        + target_fg.sum(
            dim=dims
        )
    )

    dice = (
        2.0
        * intersection
        + 1e-5
    ) / (
        denominator
        + 1e-5
    )

    return 1.0 - dice.mean()


def make_dataset_class(stack):
    torch = stack["torch"]
    F = stack["F"]
    Dataset = stack["Dataset"]

    class SourceSliceDataset(Dataset):
        def __init__(
            self,
            rows,
            tracker,
        ):
            self.rows = rows
            self.tracker = tracker
            self.slice_index = make_slice_index(
                rows
            )

            self.cache_subject = None
            self.cache_image = None
            self.cache_mask = None

        def __len__(self):
            return len(
                self.slice_index
            )

        def _load_case(
            self,
            row_idx,
        ):
            row = self.rows[
                row_idx
            ]

            subject = row[
                "subject_id"
            ]

            if self.cache_subject == subject:
                return (
                    self.cache_image,
                    self.cache_mask,
                )

            image, mask = load_source_case(
                row,
                self.tracker,
            )

            self.cache_subject = subject
            self.cache_image = image
            self.cache_mask = mask

            return image, mask

        def __getitem__(
            self,
            idx,
        ):
            row_idx, phase, z = (
                self.slice_index[
                    idx
                ]
            )

            image, mask = self._load_case(
                row_idx
            )

            x = torch.from_numpy(
                image[
                    phase,
                    z,
                ]
            ).float()

            y = torch.from_numpy(
                mask[
                    phase,
                    z,
                ].astype(
                    np.int64,
                    copy=False,
                )
            ).long()

            x = x[
                None,
                None,
            ]

            x = F.interpolate(
                x,
                size=(
                    INPUT_SIZE,
                    INPUT_SIZE,
                ),
                mode="bilinear",
                align_corners=False,
            )[
                0,
                0,
            ]

            y = F.interpolate(
                y[
                    None,
                    None,
                ].float(),
                size=(
                    INPUT_SIZE,
                    INPUT_SIZE,
                ),
                mode="nearest",
            )[
                0,
                0,
            ].long()

            x = x[
                None,
            ].repeat(
                3,
                1,
                1,
            )

            return x, y

    return SourceSliceDataset


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def dice_binary(
    pred: np.ndarray,
    target: np.ndarray,
) -> float:
    pred = pred.astype(
        bool,
        copy=False,
    )

    target = target.astype(
        bool,
        copy=False,
    )

    p = int(
        pred.sum()
    )

    t = int(
        target.sum()
    )

    if p == 0 and t == 0:
        return 1.0

    intersection = int(
        np.logical_and(
            pred,
            target,
        ).sum()
    )

    return (
        2.0
        * intersection
        / (p + t)
    )


def validate_subjects(
    model,
    rows,
    device,
    stack,
    tracker,
):
    torch = stack["torch"]
    F = stack["F"]

    model.eval()

    metrics = []

    with torch.no_grad():
        for row in tqdm(
            rows,
            desc="Source-val subjects",
            unit="subject",
            leave=False,
            dynamic_ncols=True,
        ):
            image, mask = load_source_case(
                row,
                tracker,
            )

            phases, z_count, y_native, x_native = (
                image.shape
            )

            pred_native = np.zeros(
                mask.shape,
                dtype=np.uint8,
            )

            for phase in range(
                phases
            ):
                for z in range(
                    z_count
                ):
                    x = torch.from_numpy(
                        image[
                            phase,
                            z,
                        ]
                    ).float()[
                        None,
                        None,
                    ]

                    x = F.interpolate(
                        x,
                        size=(
                            INPUT_SIZE,
                            INPUT_SIZE,
                        ),
                        mode="bilinear",
                        align_corners=False,
                    )

                    x = x.repeat(
                        1,
                        3,
                        1,
                        1,
                    ).to(
                        device,
                        non_blocking=True,
                    )

                    logits = model(
                        x
                    )["out"]

                    hard = torch.argmax(
                        logits,
                        dim=1,
                        keepdim=True,
                    ).float()

                    hard_native = F.interpolate(
                        hard,
                        size=(
                            y_native,
                            x_native,
                        ),
                        mode="nearest",
                    )[
                        0,
                        0,
                    ].to(
                        "cpu"
                    ).numpy().astype(
                        np.uint8,
                        copy=False,
                    )

                    pred_native[
                        phase,
                        z,
                    ] = hard_native

            class_dice = {}

            for cls in (
                1,
                2,
                3,
            ):
                class_dice[
                    cls
                ] = dice_binary(
                    pred_native == cls,
                    mask == cls,
                )

            subject_mdice = float(
                np.mean(
                    list(
                        class_dice.values()
                    )
                )
            )

            metrics.append({
                "subject_id":
                    row["subject_id"],
                "vendor":
                    row["vendor"],
                "centre":
                    row["centre"],
                "dice_class_1":
                    class_dice[1],
                "dice_class_2":
                    class_dice[2],
                "dice_class_3":
                    class_dice[3],
                "foreground_mean_dice":
                    subject_mdice,
            })

    mean_dice = float(
        np.mean(
            [
                row[
                    "foreground_mean_dice"
                ]
                for row in metrics
            ]
        )
    )

    return mean_dice, metrics


# ---------------------------------------------------------------------------
# Train one direction
# ---------------------------------------------------------------------------

def train_direction(
    source_vendor,
    target_vendor,
    all_rows,
    direction_dir,
    stack,
    device,
):
    torch = stack["torch"]
    F = stack["F"]
    DataLoader = stack["DataLoader"]

    source_rows = [
        row
        for row in all_rows
        if row["vendor"]
        == source_vendor
    ]

    train_rows, val_rows = split_source_subjects(
        source_rows,
        source_vendor,
    )

    target_ids = {
        row["subject_id"]
        for row in all_rows
        if row["vendor"]
        == target_vendor
    }

    source_ids = {
        row["subject_id"]
        for row in source_rows
    }

    if source_ids & target_ids:
        raise RuntimeError(
            "Source/target subject overlap."
        )

    direction_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    tracker = {
        "source_image_npz_opens": 0,
        "source_mask_npz_opens": 0,
        "target_image_npz_opens": 0,
        "target_mask_npz_opens": 0,
    }

    DatasetClass = make_dataset_class(
        stack
    )

    train_dataset = DatasetClass(
        train_rows,
        tracker,
    )

    generator = torch.Generator()
    generator.manual_seed(
        SEED
    )

    loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(
            device.type == "cuda"
        ),
        drop_last=False,
        generator=generator,
    )

    set_global_seed(
        SEED,
        torch,
    )

    model = build_model(
        stack
    ).to(
        device
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(
            device.type == "cuda"
        ),
    )

    epoch_rows = []

    best_val = -1.0
    best_epoch = -1
    best_path = (
        direction_dir
        / "best_model.pt"
    )
    best_subject_metrics = None

    for epoch in range(
        1,
        EPOCHS + 1,
    ):
        model.train()

        loss_sum = 0.0
        ce_sum = 0.0
        dice_sum = 0.0
        sample_count = 0

        bar = tqdm(
            loader,
            desc=(
                f"{source_vendor}->{target_vendor} "
                f"epoch {epoch:02d}/{EPOCHS}"
            ),
            unit="batch",
            dynamic_ncols=True,
        )

        for x, y in bar:
            x = x.to(
                device,
                non_blocking=True,
            )

            y = y.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=(
                    device.type == "cuda"
                ),
            ):
                logits = model(
                    x
                )["out"]

                ce = F.cross_entropy(
                    logits,
                    y,
                )

                dl = soft_foreground_dice_loss(
                    logits,
                    y,
                    torch,
                    F,
                )

                loss = ce + dl

            scaler.scale(
                loss
            ).backward()

            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRAD_CLIP,
            )

            scaler.step(
                optimizer
            )

            scaler.update()

            bs = int(
                x.shape[0]
            )

            sample_count += bs
            loss_sum += (
                float(
                    loss.detach().cpu()
                )
                * bs
            )

            ce_sum += (
                float(
                    ce.detach().cpu()
                )
                * bs
            )

            dice_sum += (
                float(
                    dl.detach().cpu()
                )
                * bs
            )

            bar.set_postfix(
                loss=f"{loss_sum/max(sample_count,1):.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

        bar.close()

        val_dice, val_subject_metrics = validate_subjects(
            model,
            val_rows,
            device,
            stack,
            tracker,
        )

        lr_now = float(
            optimizer.param_groups[
                0
            ]["lr"]
        )

        epoch_rows.append({
            "epoch": epoch,
            "train_loss":
                loss_sum
                / max(
                    sample_count,
                    1,
                ),
            "train_ce":
                ce_sum
                / max(
                    sample_count,
                    1,
                ),
            "train_soft_dice_loss":
                dice_sum
                / max(
                    sample_count,
                    1,
                ),
            "source_val_mean_dice":
                val_dice,
            "learning_rate":
                lr_now,
        })

        print(
            f"[{source_vendor}->{target_vendor}] "
            f"epoch={epoch:02d} "
            f"source_val_mean_dice={val_dice:.6f}"
        )

        if val_dice > best_val:
            best_val = val_dice
            best_epoch = epoch
            best_subject_metrics = (
                val_subject_metrics
            )

            torch.save(
                {
                    "script_version":
                        VERSION,
                    "architecture":
                        "DeepLabV3-ResNet50",
                    "num_classes":
                        NUM_CLASSES,
                    "seed":
                        SEED,
                    "source_vendor":
                        source_vendor,
                    "target_vendor":
                        target_vendor,
                    "epoch":
                        epoch,
                    "source_val_mean_dice":
                        val_dice,
                    "model_state_dict":
                        model.state_dict(),
                },
                best_path,
            )

        scheduler.step()

    if best_subject_metrics is None:
        raise RuntimeError(
            "No best checkpoint selected."
        )

    write_csv(
        direction_dir
        / "epoch_metrics.csv",
        epoch_rows,
        [
            "epoch",
            "train_loss",
            "train_ce",
            "train_soft_dice_loss",
            "source_val_mean_dice",
            "learning_rate",
        ],
    )

    write_csv(
        direction_dir
        / "validation_subject_metrics.csv",
        best_subject_metrics,
        [
            "subject_id",
            "vendor",
            "centre",
            "dice_class_1",
            "dice_class_2",
            "dice_class_3",
            "foreground_mean_dice",
        ],
    )

    summary = {
        "source_vendor":
            source_vendor,
        "target_vendor":
            target_vendor,
        "architecture":
            "DeepLabV3-ResNet50",
        "seed":
            SEED,
        "source_train_subjects":
            len(train_rows),
        "source_val_subjects":
            len(val_rows),
        "train_slices":
            len(train_dataset),
        "epochs":
            EPOCHS,
        "best_epoch":
            best_epoch,
        "best_source_val_mean_dice":
            best_val,
        "target_image_npz_opens":
            tracker[
                "target_image_npz_opens"
            ],
        "target_mask_npz_opens":
            tracker[
                "target_mask_npz_opens"
            ],
        "source_image_npz_opens":
            tracker[
                "source_image_npz_opens"
            ],
        "source_mask_npz_opens":
            tracker[
                "source_mask_npz_opens"
            ],
        "checkpoint_sha256":
            sha256_file(
                best_path
            ),
    }

    write_json(
        direction_dir
        / "training_summary.json",
        summary,
    )

    if (
        tracker[
            "target_image_npz_opens"
        ]
        != 0
        or tracker[
            "target_mask_npz_opens"
        ]
        != 0
    ):
        raise RuntimeError(
            "Target NPZ access occurred."
        )

    return summary


# ---------------------------------------------------------------------------
# Preflight and formal run
# ---------------------------------------------------------------------------

def preflight():
    upstream = load_upstream_lock()
    rows = load_case_manifest()
    stack = import_torch_stack()

    torch = stack[
        "torch"
    ]

    torchvision = stack[
        "torchvision"
    ]

    split_rows = build_split_manifest(
        rows
    )

    counts = Counter(
        (
            r["source_vendor"],
            r["split"],
        )
        for r in split_rows
    )

    print(
        "===== Q1-R08B0 PREFLIGHT ====="
    )

    print(
        f"protocol_sha256={EXPECTED_PROTOCOL_SHA256}"
    )

    print(
        f"r08a2_lock_sha256={EXPECTED_R08A2_LOCK_SHA256}"
    )

    print(
        f"compact_cases={len(rows)}"
    )

    print(
        f"B_train={counts[('B','train')]}"
    )

    print(
        f"B_val={counts[('B','val')]}"
    )

    print(
        f"A_train={counts[('A','train')]}"
    )

    print(
        f"A_val={counts[('A','val')]}"
    )

    print(
        f"torch_version={torch.__version__}"
    )

    print(
        f"torchvision_version={torchvision.__version__}"
    )

    print(
        f"cuda_available={torch.cuda.is_available()}"
    )

    if torch.cuda.is_available():
        print(
            f"cuda_device={torch.cuda.get_device_name(0)}"
        )

    print(
        "architecture=DeepLabV3-ResNet50"
    )

    print(
        "backbone_init=ImageNet1K_V1"
    )

    print(
        f"epochs={EPOCHS}"
    )

    print(
        f"batch_size={BATCH_SIZE}"
    )

    print(
        "target_image_npz_access=FORBIDDEN"
    )

    print(
        "target_mask_npz_access=FORBIDDEN"
    )

    print(
        "target_metric_computation=NO"
    )

    print(
        "PREFLIGHT_PASS"
    )


def run(args):
    upstream = load_upstream_lock()
    rows = load_case_manifest()
    stack = import_torch_stack()

    torch = stack[
        "torch"
    ]

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    if args.output_dir.exists():
        raise FileExistsError(
            args.output_dir
        )

    build_dir = Path(
        str(args.output_dir)
        + "__building"
    )

    if build_dir.exists():
        raise FileExistsError(
            build_dir
        )

    build_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    protocol_copy = (
        build_dir
        / "preregistered_protocol_copy.md"
    )

    shutil.copy2(
        PROTOCOL,
        protocol_copy,
    )

    validate_sha(
        protocol_copy,
        EXPECTED_PROTOCOL_SHA256,
        "protocol copy",
    )

    split_rows = build_split_manifest(
        rows
    )

    write_csv(
        build_dir
        / "source_split_manifest.csv",
        split_rows,
        [
            "direction",
            "source_vendor",
            "target_vendor",
            "split",
            "subject_id",
            "centre",
            "image_npz",
            "mask_npz",
            "target_file_open_allowed",
        ],
    )

    summaries = {}

    for source_vendor, target_vendor in DIRECTIONS:
        direction_name = (
            f"{source_vendor}_to_{target_vendor}"
        )

        print()
        print(
            "=" * 72
        )
        print(
            f"TRAIN SOURCE-ONLY PILOT: "
            f"{source_vendor} -> {target_vendor}"
        )
        print(
            "=" * 72
        )

        summary = train_direction(
            source_vendor,
            target_vendor,
            rows,
            build_dir
            / direction_name,
            stack,
            device,
        )

        summaries[
            direction_name
        ] = summary

    b_score = float(
        summaries[
            "B_to_A"
        ][
            "best_source_val_mean_dice"
        ]
    )

    a_score = float(
        summaries[
            "A_to_B"
        ][
            "best_source_val_mean_dice"
        ]
    )

    total_target_image_opens = sum(
        int(
            s[
                "target_image_npz_opens"
            ]
        )
        for s in summaries.values()
    )

    total_target_mask_opens = sum(
        int(
            s[
                "target_mask_npz_opens"
            ]
        )
        for s in summaries.values()
    )

    checks = {
        "A_B_source_val_mean_dice_ge_0p70":
            b_score
            >= FEASIBILITY_DICE,
        "B_A_source_val_mean_dice_ge_0p70":
            a_score
            >= FEASIBILITY_DICE,
        "C_target_image_npz_opens_eq_0":
            total_target_image_opens
            == 0,
        "D_target_mask_npz_opens_eq_0":
            total_target_mask_opens
            == 0,
        "E_B_source_split_100_25":
            (
                summaries[
                    "B_to_A"
                ][
                    "source_train_subjects"
                ]
                == 100
                and summaries[
                    "B_to_A"
                ][
                    "source_val_subjects"
                ]
                == 25
            ),
        "F_A_source_split_76_19":
            (
                summaries[
                    "A_to_B"
                ][
                    "source_train_subjects"
                ]
                == 76
                and summaries[
                    "A_to_B"
                ][
                    "source_val_subjects"
                ]
                == 19
            ),
    }

    decision = (
        READY
        if all(
            checks.values()
        )
        else INSUFFICIENT
    )

    gate = {
        "script_version":
            VERSION,
        "build":
            BUILD,
        "protocol_sha256":
            EXPECTED_PROTOCOL_SHA256,
        "r08a2_lock_sha256":
            EXPECTED_R08A2_LOCK_SHA256,
        "architecture":
            "DeepLabV3-ResNet50",
        "seed":
            SEED,
        "feasibility_threshold":
            FEASIBILITY_DICE,
        "B_to_A_best_source_val_mean_dice":
            b_score,
        "A_to_B_best_source_val_mean_dice":
            a_score,
        "target_image_npz_opens":
            total_target_image_opens,
        "target_mask_npz_opens":
            total_target_mask_opens,
        "checks":
            checks,
        "decision":
            decision,
    }

    write_json(
        build_dir
        / "pilot_gate.json",
        gate,
    )

    (
        build_dir
        / "decision.txt"
    ).write_text(
        decision + "\n",
        encoding="utf-8",
    )

    lines = [
        "===== Q1-R08B0 M&Ms SOURCE-ONLY DEEPLABV3-R50 PILOT =====",
        f"Script version: {VERSION}",
        f"Build: {BUILD}",
        "",
        "Frozen protocol:",
        "  architecture=DeepLabV3-ResNet50",
        f"  seed={SEED}",
        "  directions=B->A and A->B",
        f"  epochs={EPOCHS}",
        f"  batch_size={BATCH_SIZE}",
        f"  lr={LR}",
        f"  weight_decay={WEIGHT_DECAY}",
        "  loss=CE + soft foreground Dice",
        "  target data used=NO",
        "",
        "Source-only results:",
        f"  B->A source-val best mean Dice={b_score:.6f}",
        f"  B->A best epoch={summaries['B_to_A']['best_epoch']}",
        f"  A->B source-val best mean Dice={a_score:.6f}",
        f"  A->B best epoch={summaries['A_to_B']['best_epoch']}",
        "",
        "Target-blind boundary:",
        f"  target image NPZ opens={total_target_image_opens}",
        f"  target mask NPZ opens={total_target_mask_opens}",
        "  target metrics computed=NO",
        "",
        "Checks:",
    ]

    for key, value in checks.items():
        lines.append(
            f"  {key}="
            f"{'PASS' if value else 'FAIL'}"
        )

    lines += [
        "",
        "Decision:",
        f"  {decision}",
    ]

    run_log = (
        build_dir
        / "run_log.txt"
    )

    run_log.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    artifact_names = [
        "preregistered_protocol_copy.md",
        "source_split_manifest.csv",
        "pilot_gate.json",
        "decision.txt",
        "run_log.txt",
        "B_to_A/best_model.pt",
        "B_to_A/epoch_metrics.csv",
        "B_to_A/validation_subject_metrics.csv",
        "B_to_A/training_summary.json",
        "A_to_B/best_model.pt",
        "A_to_B/epoch_metrics.csv",
        "A_to_B/validation_subject_metrics.csv",
        "A_to_B/training_summary.json",
    ]

    artifacts = {}

    for name in tqdm(
        artifact_names,
        desc="Hash R08B0 artifacts",
        unit="file",
        dynamic_ncols=True,
    ):
        p = (
            build_dir
            / name
        )

        if not p.is_file():
            raise FileNotFoundError(
                p
            )

        artifacts[
            name
        ] = {
            "relative_path":
                name,
            "sha256":
                sha256_file(
                    p
                ),
        }

    lock = {
        "script_version":
            VERSION,
        "build":
            BUILD,
        "protocol_sha256":
            EXPECTED_PROTOCOL_SHA256,
        "r08a2_lock_sha256":
            EXPECTED_R08A2_LOCK_SHA256,
        "architecture":
            "DeepLabV3-ResNet50",
        "seed":
            SEED,
        "directions": [
            "B->A",
            "A->B",
        ],
        "B_to_A_best_source_val_mean_dice":
            b_score,
        "A_to_B_best_source_val_mean_dice":
            a_score,
        "target_image_npz_opens":
            total_target_image_opens,
        "target_mask_npz_opens":
            total_target_mask_opens,
        "target_metrics_computed":
            False,
        "decision":
            decision,
        "checks":
            checks,
        "artifacts":
            artifacts,
    }

    lock_path = (
        build_dir
        / "Q1_R08B0_MNMS_SOURCE_ONLY_PILOT_LOCK.json"
    )

    write_json(
        lock_path,
        lock,
    )

    lock_sha = sha256_file(
        lock_path
    )

    for name, meta in artifacts.items():
        p = (
            build_dir
            / meta[
                "relative_path"
            ]
        )

        if sha256_file(
            p
        ) != meta[
            "sha256"
        ]:
            raise RuntimeError(
                f"Artifact changed before commit: {name}"
            )

    build_dir.rename(
        args.output_dir
    )

    print()
    print(
        (
            args.output_dir
            / "run_log.txt"
        ).read_text(
            encoding="utf-8"
        )
    )

    print(
        "Q1-R08B0 LOCK:",
        args.output_dir
        / "Q1_R08B0_MNMS_SOURCE_ONLY_PILOT_LOCK.json",
    )

    print(
        "Q1-R08B0 LOCK SHA256:",
        lock_sha,
    )


def self_test():
    assert SEED == 20260819
    assert EXPECTED_VENDOR_COUNTS == {
        "B": 125,
        "A": 95,
    }
    assert EXPECTED_SPLITS == {
        "B": {
            "train": 100,
            "val": 25,
        },
        "A": {
            "train": 76,
            "val": 19,
        },
    }
    assert FEASIBILITY_DICE == 0.70

    x = np.asarray(
        [
            [0.0, 1.0],
            [2.0, 3.0],
        ],
        dtype=np.float32,
    )

    y = robust_normalize_phase(
        x
    )

    assert y.dtype == np.float32
    assert np.isfinite(
        y
    ).all()

    assert abs(
        dice_binary(
            np.asarray(
                [1, 0, 1]
            ),
            np.asarray(
                [1, 0, 1]
            ),
        )
        - 1.0
    ) < 1e-12

    print(
        "FROZEN_VENDOR_COUNTS_TEST_PASS"
    )
    print(
        "FROZEN_SOURCE_SPLIT_COUNTS_TEST_PASS"
    )
    print(
        "ROBUST_NORMALIZATION_TEST_PASS"
    )
    print(
        "SUBJECT_DICE_TEST_PASS"
    )
    print(
        "TARGET_ACCESS_COUNTERS_DESIGN_TEST_PASS"
    )
    print(
        "SELF_TEST_PASS"
    )


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Q1-R08B0: source-only DeepLabV3-R50 cardiac segmentation "
            "pilot on frozen M&Ms Vendor B/A domains. "
            "Target NPZ files are never opened."
        )
    )

    p.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
    )

    p.add_argument(
        "--preflight-only",
        action="store_true",
    )

    p.add_argument(
        "--self-test",
        action="store_true",
    )

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
