#!/usr/bin/env python
# -*- coding: utf-8 -*-

r"""
S01-B: PraNet source-only training for one of the three frozen seeds.

Scientific safety:
- Reads the full S01 manifest only as metadata.
- Opens image/mask files ONLY for roles source_train and source_val.
- Does NOT open seen_sanity or unseen_locked images/masks.
- Does NOT compute target metrics.
- Checkpoint selection uses source_val mean Dice only.

Architecture:
- PraNet (MICCAI 2020) with Res2Net-50-v1b-26w-4s backbone.
- Four supervised outputs.
- PraNet structure loss: weighted BCE + weighted IoU.

Frozen S00 training settings:
- 352 x 352
- AdamW
- LR 1e-4
- weight decay 1e-4
- 80 epochs
- batch size 8
- AMP
- random H/V flip
- random scale in {0.75, 1.0, 1.25}, then crop/pad to 352
- ImageNet normalization
- source_val mean Dice selects the best checkpoint

Project root:
    F:\MEDSEG_SAFETTA
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
import time
from collections import Counter
from pathlib import Path
from typing import List, Dict, Tuple
from urllib.parse import urlparse

import numpy as np
from PIL import Image, ImageOps

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


VERSION = "2026-08-17-S01-B-frozen-seeds-v1"
BUILD = "S01_B_FROZEN_SEED_PARAMETERIZATION_NO_PROTOCOL_CHANGE"

ROOT = Path(r"F:\MEDSEG_SAFETTA")
DATA_ROOT = ROOT / "data" / "processed" / "S00_polyp_locked_v1"
MANIFEST = ROOT / "data" / "splits" / "S01_locked_protocol_manifest_v1.csv"
EXPECTED_MANIFEST_SHA256 = "afb4bc1175af004819538cdbb22ee7a10f1be48035945c7e0d50fd2bbe2e3dc7"

PRETRAIN_DIR = ROOT / "assets" / "pretrained"
PRETRAIN_FILENAME = "res2net50_v1b_26w_4s-3cf99910.pth"
PRETRAIN_PATH = PRETRAIN_DIR / PRETRAIN_FILENAME
PRETRAIN_URL = (
    "https://shanghuagao.oss-cn-beijing.aliyuncs.com/"
    "res2net/res2net50_v1b_26w_4s-3cf99910.pth"
)

DEFAULT_OUTPUT = None

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


# ---------------------------------------------------------------------
# Reproducibility / filesystem
# ---------------------------------------------------------------------

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

    # Deterministic where reasonably available. This is a pilot baseline,
    # so reproducibility is prioritized over maximum speed.
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
        rows = list(reader)
        fields = reader.fieldnames or []
    return rows, fields


def validate_manifest(path: Path, rows: List[dict], fields: List[str]) -> dict:
    required = {
        "sample_id",
        "split",
        "dataset",
        "image_relpath",
        "mask_relpath",
        "s01_role",
    }
    missing = sorted(required - set(fields))
    if missing:
        raise RuntimeError(f"Manifest missing required columns: {missing}")

    sha = file_sha256(path)
    if sha.lower() != EXPECTED_MANIFEST_SHA256.lower():
        raise RuntimeError(
            "Frozen S01 manifest SHA256 mismatch.\n"
            f"Expected: {EXPECTED_MANIFEST_SHA256}\n"
            f"Actual  : {sha}\n"
            "Refusing to train on a changed split."
        )

    counts = Counter(r["s01_role"] for r in rows)
    if counts != Counter(EXPECTED_ROLE_COUNTS):
        raise RuntimeError(
            f"Frozen role counts mismatch: expected={EXPECTED_ROLE_COUNTS}, "
            f"actual={dict(counts)}"
        )

    ids = [r["sample_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate sample_id in manifest.")

    return {
        "manifest_sha256": sha,
        "role_counts": dict(counts),
    }


def resolve_source_rows(rows: List[dict], role: str) -> List[dict]:
    if role not in {"source_train", "source_val"}:
        raise RuntimeError(
            f"Safety stop: S01-B may only open source_train/source_val, got {role}"
        )
    selected = [r for r in rows if r["s01_role"] == role]
    expected = EXPECTED_ROLE_COUNTS[role]
    if len(selected) != expected:
        raise RuntimeError(
            f"{role} count mismatch: expected={expected}, actual={len(selected)}"
        )
    return selected


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------

def resize_pair(image: Image.Image, mask: Image.Image, size: int) -> Tuple[Image.Image, Image.Image]:
    return (
        image.resize((size, size), RESAMPLE_BILINEAR),
        mask.resize((size, size), RESAMPLE_NEAREST),
    )


def random_scale_crop_pad(
    image: Image.Image,
    mask: Image.Image,
    base_size: int,
    rng: random.Random,
) -> Tuple[Image.Image, Image.Image]:
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

    if image.size != (base_size, base_size) or mask.size != (base_size, base_size):
        raise RuntimeError(
            f"Augmentation output size mismatch: image={image.size}, mask={mask.size}"
        )
    return image, mask


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1)).copy()
    return torch.from_numpy(arr)


def mask_to_tensor(mask: Image.Image) -> torch.Tensor:
    arr = np.asarray(mask.convert("L"), dtype=np.float32)
    maxv = float(arr.max())
    if maxv <= 0:
        raise RuntimeError("Empty source mask encountered.")
    # Source masks were audited as hard 0/255, but thresholding here makes the
    # training target explicitly binary and robust to file storage details.
    arr = (arr > (0.5 * maxv)).astype(np.float32)
    return torch.from_numpy(arr[None, ...].copy())


class SourcePolypDataset(Dataset):
    def __init__(
        self,
        rows: List[dict],
        data_root: Path,
        training: bool,
        base_seed: int,
    ):
        self.rows = list(rows)
        self.data_root = data_root
        self.training = bool(training)
        self.base_seed = int(base_seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.rows)

    def _rng(self, index: int) -> random.Random:
        # Deterministic per epoch/sample, independent of DataLoader worker order.
        sid = self.rows[index]["sample_id"]
        payload = f"{self.base_seed}::{self.epoch}::{sid}".encode("utf-8")
        value = int(hashlib.sha256(payload).hexdigest()[:16], 16)
        return random.Random(value)

    def __getitem__(self, index: int):
        row = self.rows[index]
        role = row["s01_role"]
        if role not in {"source_train", "source_val"}:
            raise RuntimeError(f"Dataset safety violation: attempted to open role={role}")

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
                f"Image/mask mismatch for {row['sample_id']}: "
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

            # First normalize FOV to the frozen 352 base resolution, then apply
            # the frozen random scale/crop/pad augmentation.
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


# ---------------------------------------------------------------------
# Res2Net backbone — adapted from the official PraNet/Res2Net implementation
# ---------------------------------------------------------------------

class Bottle2neck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        base_width=26,
        scale=4,
        stype="normal",
    ):
        super().__init__()
        width = int(math.floor(planes * (base_width / 64.0)))
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)

        self.nums = 1 if scale == 1 else scale - 1
        if stype == "stage":
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)

        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    width, width, kernel_size=3,
                    stride=stride, padding=1, bias=False
                )
                for _ in range(self.nums)
            ]
        )
        self.bns = nn.ModuleList(
            [nn.BatchNorm2d(width) for _ in range(self.nums)]
        )

        self.conv3 = nn.Conv2d(
            width * scale, planes * self.expansion,
            kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        residual = x

        out = self.relu(self.bn1(self.conv1(x)))
        spx = torch.split(out, self.width, dim=1)

        sp = None
        for i in range(self.nums):
            if i == 0 or self.stype == "stage":
                sp = spx[i]
            else:
                sp = sp + spx[i]

            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))

            if i == 0:
                out_cat = sp
            else:
                out_cat = torch.cat((out_cat, sp), dim=1)

        if self.scale != 1 and self.stype == "normal":
            out_cat = torch.cat((out_cat, spx[self.nums]), dim=1)
        elif self.scale != 1 and self.stype == "stage":
            out_cat = torch.cat((out_cat, self.pool(spx[self.nums])), dim=1)

        out_cat = self.bn3(self.conv3(out_cat))

        if self.downsample is not None:
            residual = self.downsample(x)

        out_cat = self.relu(out_cat + residual)
        return out_cat


class Res2Net(nn.Module):
    def __init__(self, block, layers, base_width=26, scale=4, num_classes=1000):
        super().__init__()
        self.inplanes = 64
        self.base_width = base_width
        self.scale = scale

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False),
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(
                    kernel_size=stride,
                    stride=stride,
                    ceil_mode=True,
                    count_include_pad=False,
                ),
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=1,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [
            block(
                self.inplanes,
                planes,
                stride,
                downsample=downsample,
                stype="stage",
                base_width=self.base_width,
                scale=self.scale,
            )
        ]
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    base_width=self.base_width,
                    scale=self.scale,
                )
            )
        return nn.Sequential(*layers)


def build_res2net50_v1b_26w_4s() -> Res2Net:
    return Res2Net(
        Bottle2neck,
        [3, 4, 6, 3],
        base_width=26,
        scale=4,
    )


# ---------------------------------------------------------------------
# PraNet
# ---------------------------------------------------------------------

class BasicConv2d(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        return self.bn(self.conv(x))


class RFBModified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.relu = nn.ReLU(True)

        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(
                out_channel, out_channel,
                kernel_size=(1, 3), padding=(0, 1)
            ),
            BasicConv2d(
                out_channel, out_channel,
                kernel_size=(3, 1), padding=(1, 0)
            ),
            BasicConv2d(
                out_channel, out_channel, 3,
                padding=3, dilation=3
            ),
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(
                out_channel, out_channel,
                kernel_size=(1, 5), padding=(0, 2)
            ),
            BasicConv2d(
                out_channel, out_channel,
                kernel_size=(5, 1), padding=(2, 0)
            ),
            BasicConv2d(
                out_channel, out_channel, 3,
                padding=5, dilation=5
            ),
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(
                out_channel, out_channel,
                kernel_size=(1, 7), padding=(0, 3)
            ),
            BasicConv2d(
                out_channel, out_channel,
                kernel_size=(7, 1), padding=(3, 0)
            ),
            BasicConv2d(
                out_channel, out_channel, 3,
                padding=7, dilation=7
            ),
        )
        self.conv_cat = BasicConv2d(
            4 * out_channel, out_channel, 3, padding=1
        )
        self.conv_res = BasicConv2d(
            in_channel, out_channel, 1
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))
        return self.relu(x_cat + self.conv_res(x))


class Aggregation(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(
            2 * channel, 2 * channel, 3, padding=1
        )
        self.conv_concat2 = BasicConv2d(
            2 * channel, 2 * channel, 3, padding=1
        )
        self.conv_concat3 = BasicConv2d(
            3 * channel, 3 * channel, 3, padding=1
        )
        self.conv4 = BasicConv2d(
            3 * channel, 3 * channel, 3, padding=1
        )
        self.conv5 = nn.Conv2d(3 * channel, 1, 1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = (
            self.conv_upsample2(self.upsample(self.upsample(x1)))
            * self.conv_upsample3(self.upsample(x2))
            * x3
        )
        x2_2 = torch.cat(
            (x2_1, self.conv_upsample4(self.upsample(x1_1))), 1
        )
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat(
            (x3_1, self.conv_upsample5(self.upsample(x2_2))), 1
        )
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        return self.conv5(x)


class PraNet(nn.Module):
    def __init__(self, channel=32, pretrained_backbone_state=None):
        super().__init__()
        self.resnet = build_res2net50_v1b_26w_4s()
        if pretrained_backbone_state is not None:
            missing, unexpected = self.resnet.load_state_dict(
                pretrained_backbone_state, strict=False
            )
            # Classification fc is present in official ImageNet weight, so
            # normally both lists should be empty. Fail on backbone mismatch.
            if missing or unexpected:
                raise RuntimeError(
                    f"Res2Net pretrained load mismatch: "
                    f"missing={missing[:20]}, unexpected={unexpected[:20]}"
                )

        self.rfb2_1 = RFBModified(512, channel)
        self.rfb3_1 = RFBModified(1024, channel)
        self.rfb4_1 = RFBModified(2048, channel)

        self.agg1 = Aggregation(channel)

        self.ra4_conv1 = BasicConv2d(2048, 256, kernel_size=1)
        self.ra4_conv2 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv3 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv4 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv5 = BasicConv2d(256, 1, kernel_size=1)

        self.ra3_conv1 = BasicConv2d(1024, 64, kernel_size=1)
        self.ra3_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra3_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra3_conv4 = BasicConv2d(64, 1, kernel_size=3, padding=1)

        self.ra2_conv1 = BasicConv2d(512, 64, kernel_size=1)
        self.ra2_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra2_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra2_conv4 = BasicConv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, x):
        input_size = x.shape[-2:]

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x1 = self.resnet.layer1(x)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)

        x2_rfb = self.rfb2_1(x2)
        x3_rfb = self.rfb3_1(x3)
        x4_rfb = self.rfb4_1(x4)

        ra5_feat = self.agg1(x4_rfb, x3_rfb, x2_rfb)
        lateral_map_5 = F.interpolate(
            ra5_feat, size=input_size,
            mode="bilinear", align_corners=True
        )

        crop_4 = F.interpolate(
            ra5_feat, size=x4.shape[-2:],
            mode="bilinear", align_corners=True
        )
        rev4 = 1.0 - torch.sigmoid(crop_4)
        rev4 = rev4.expand(-1, 2048, -1, -1) * x4
        rev4 = self.ra4_conv1(rev4)
        rev4 = F.relu(self.ra4_conv2(rev4))
        rev4 = F.relu(self.ra4_conv3(rev4))
        rev4 = F.relu(self.ra4_conv4(rev4))
        ra4_feat = self.ra4_conv5(rev4)
        x_ra4 = ra4_feat + crop_4
        lateral_map_4 = F.interpolate(
            x_ra4, size=input_size,
            mode="bilinear", align_corners=True
        )

        crop_3 = F.interpolate(
            x_ra4, size=x3.shape[-2:],
            mode="bilinear", align_corners=True
        )
        rev3 = 1.0 - torch.sigmoid(crop_3)
        rev3 = rev3.expand(-1, 1024, -1, -1) * x3
        rev3 = self.ra3_conv1(rev3)
        rev3 = F.relu(self.ra3_conv2(rev3))
        rev3 = F.relu(self.ra3_conv3(rev3))
        ra3_feat = self.ra3_conv4(rev3)
        x_ra3 = ra3_feat + crop_3
        lateral_map_3 = F.interpolate(
            x_ra3, size=input_size,
            mode="bilinear", align_corners=True
        )

        crop_2 = F.interpolate(
            x_ra3, size=x2.shape[-2:],
            mode="bilinear", align_corners=True
        )
        rev2 = 1.0 - torch.sigmoid(crop_2)
        rev2 = rev2.expand(-1, 512, -1, -1) * x2
        rev2 = self.ra2_conv1(rev2)
        rev2 = F.relu(self.ra2_conv2(rev2))
        rev2 = F.relu(self.ra2_conv3(rev2))
        ra2_feat = self.ra2_conv4(rev2)
        x_ra2 = ra2_feat + crop_2
        lateral_map_2 = F.interpolate(
            x_ra2, size=input_size,
            mode="bilinear", align_corners=True
        )

        return (
            lateral_map_5,
            lateral_map_4,
            lateral_map_3,
            lateral_map_2,
        )


# ---------------------------------------------------------------------
# Loss / metrics
# ---------------------------------------------------------------------

def structure_loss(pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
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
def binary_metrics_from_logits(
    logits: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
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


# ---------------------------------------------------------------------
# Pretrained backbone loading — force cache under F:\MEDSEG_SAFETTA
# ---------------------------------------------------------------------

def load_pretrained_res2net() -> Tuple[dict, str]:
    PRETRAIN_DIR.mkdir(parents=True, exist_ok=True)

    # load_state_dict_from_url respects model_dir; no torch hub cache on C:.
    state = torch.hub.load_state_dict_from_url(
        PRETRAIN_URL,
        model_dir=str(PRETRAIN_DIR),
        file_name=PRETRAIN_FILENAME,
        progress=True,
        check_hash=True,
        map_location="cpu",
    )

    if not PRETRAIN_PATH.exists():
        raise RuntimeError(
            f"Expected pretrained file not found after download: {PRETRAIN_PATH}"
        )

    sha = file_sha256(PRETRAIN_PATH)
    return state, sha


# ---------------------------------------------------------------------
# AMP compatibility
# ---------------------------------------------------------------------

def make_grad_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            pass
    return torch.cuda.amp.GradScaler(enabled=enabled)


@contextlib.contextmanager
def autocast_context(enabled: bool):
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


# ---------------------------------------------------------------------
# Training / validation
# ---------------------------------------------------------------------

def train_one_epoch(
    model,
    loader,
    dataset,
    optimizer,
    scaler,
    device,
    epoch,
    amp_enabled,
):
    model.train()
    dataset.set_epoch(epoch)

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
            outputs = model(images)
            losses = [structure_loss(out, masks) for out in outputs]
            loss = sum(losses)

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
            outputs = model(images)
            logits = outputs[-1]

        dice, iou = binary_metrics_from_logits(logits.float(), masks.float())
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


def save_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def train(args):
    seed_everything(SEED)

    if not MANIFEST.exists():
        raise FileNotFoundError(MANIFEST)
    if not DATA_ROOT.exists():
        raise FileNotFoundError(DATA_ROOT)

    output_dir = args.output_dir

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output already exists: {output_dir}\n"
                "Refusing to overwrite a training run."
            )
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=False)

    rows, fields = read_manifest(MANIFEST)
    manifest_info = validate_manifest(MANIFEST, rows, fields)

    train_rows = resolve_source_rows(rows, "source_train")
    val_rows = resolve_source_rows(rows, "source_val")

    # Strong leakage assertion: file opening datasets are built only from these.
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
    print(f"AMP: {amp_enabled}")
    print()
    print("Frozen protocol:")
    print(f"  Manifest SHA256={manifest_info['manifest_sha256']}")
    print(f"  source_train={len(train_rows)}")
    print(f"  source_val={len(val_rows)}")
    print("  seen_sanity opened=NO")
    print("  unseen_locked opened=NO")
    print(f"  seed={SEED}")
    print(f"  image_size={IMAGE_SIZE}")
    print(f"  epochs={EPOCHS}")
    print(f"  batch_size={BATCH_SIZE}")
    print(f"  optimizer=AdamW lr={LR} wd={WEIGHT_DECAY}")
    print(f"  scales={SCALE_CHOICES}")
    print()

    print("Loading/downloading official Res2Net ImageNet backbone...")
    backbone_state, backbone_sha = load_pretrained_res2net()
    print(f"Backbone path  : {PRETRAIN_PATH}")
    print(f"Backbone SHA256: {backbone_sha}")

    model = PraNet(
        channel=32,
        pretrained_backbone_state=backbone_state,
    ).to(device)

    param_total = sum(p.numel() for p in model.parameters())
    param_trainable = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"Parameters total/trainable: {param_total:,}/{param_trainable:,}")

    train_ds = SourcePolypDataset(
        train_rows, DATA_ROOT, training=True, base_seed=SEED
    )
    val_ds = SourcePolypDataset(
        val_rows, DATA_ROOT, training=False, base_seed=SEED
    )

    generator = torch.Generator()
    generator.manual_seed(SEED)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        persistent_workers=(args.num_workers > 0),
        generator=generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        persistent_workers=(args.num_workers > 0),
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

            is_best = val["mean_dice"] > best_dice
            if is_best:
                best_dice = val["mean_dice"]
                best_epoch = epoch

                checkpoint = {
                    "script_version": VERSION,
                    "build": BUILD,
                    "seed": SEED,
                    "epoch": epoch,
                    "source_val_mean_dice": best_dice,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "manifest_sha256": manifest_info["manifest_sha256"],
                    "backbone_pretrain_sha256": backbone_sha,
                    "config": {
                        "image_size": IMAGE_SIZE,
                        "batch_size": BATCH_SIZE,
                        "epochs": EPOCHS,
                        "lr": LR,
                        "weight_decay": WEIGHT_DECAY,
                        "scales": list(SCALE_CHOICES),
                        "amp": amp_enabled,
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
            "seed": SEED,
            "manifest_sha256": manifest_info["manifest_sha256"],
            "source_train_n": len(train_rows),
            "source_val_n": len(val_rows),
            "seen_sanity_images_opened": False,
            "seen_sanity_masks_opened": False,
            "unseen_locked_images_opened": False,
            "unseen_locked_masks_opened": False,
            "target_metrics_computed": False,
            "checkpoint_selection_metric": "source_val_mean_dice",
            "best_epoch": best_epoch,
            "best_source_val_mean_dice": best_dice,
            "backbone_pretrain_path": str(PRETRAIN_PATH),
            "backbone_pretrain_sha256": backbone_sha,
            "best_checkpoint": str(best_ckpt),
            "best_checkpoint_sha256": best_ckpt_sha,
            "parameters_total": param_total,
            "parameters_trainable": param_trainable,
            "peak_gpu_memory_gb": peak_gpu_gb,
            "training_seconds": total_seconds,
            "decision": f"S01_B_SEED{SEED}_SOURCE_MODEL_READY_FOR_SANITY_REVIEW",
        }

        (output_dir / "audit.json").write_text(
            json.dumps(audit, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        summary = f"""===== S01-B PRANET SOURCE-ONLY TRAINING =====
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
  PraNet
  backbone=Res2Net-50-v1b-26w-4s ImageNet pretrained
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

Checkpoint selection:
  metric=source_val mean Dice
  best epoch={best_epoch}
  best source-val mean Dice={best_dice:.6f}

Best checkpoint:
  {best_ckpt}
  SHA256={best_ckpt_sha}

Backbone weight:
  {PRETRAIN_PATH}
  SHA256={backbone_sha}

Peak GPU memory GB={peak_gpu_gb:.3f}
Training seconds={total_seconds:.1f}

Decision: S01_B_SEED{SEED}_SOURCE_MODEL_READY_FOR_SANITY_REVIEW
"""
        (output_dir / "summary.txt").write_text(summary, encoding="utf-8")

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
        }
        (output_dir / "FAILED_TECHNICAL.json").write_text(
            json.dumps(failed, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        raise


# ---------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------

def self_test():
    # Loss test.
    logits = torch.zeros(2, 1, 64, 64)
    masks = torch.zeros(2, 1, 64, 64)
    masks[:, :, 16:48, 16:48] = 1
    loss = structure_loss(logits, masks)
    assert torch.isfinite(loss)
    assert float(loss) > 0

    # Metric test.
    perfect_logits = torch.where(
        masks > 0.5,
        torch.tensor(20.0),
        torch.tensor(-20.0),
    )
    dice, iou = binary_metrics_from_logits(perfect_logits, masks)
    assert torch.allclose(dice, torch.ones_like(dice), atol=1e-5)
    assert torch.allclose(iou, torch.ones_like(iou), atol=1e-5)

    # Architecture test WITHOUT network/pretrained weights.
    model = PraNet(channel=32, pretrained_backbone_state=None)
    model.eval()
    with torch.no_grad():
        x = torch.randn(1, 3, 64, 64)
        outputs = model(x)

    assert len(outputs) == 4
    for out in outputs:
        assert tuple(out.shape) == (1, 1, 64, 64)
        assert torch.isfinite(out).all()

    # Leakage helper test with frozen-count parity.
    fake_rows = (
        [
            {"sample_id": f"train_{i:04d}", "s01_role": "source_train"}
            for i in range(EXPECTED_ROLE_COUNTS["source_train"])
        ]
        + [
            {"sample_id": f"val_{i:04d}", "s01_role": "source_val"}
            for i in range(EXPECTED_ROLE_COUNTS["source_val"])
        ]
        + [{"sample_id": "locked_0000", "s01_role": "unseen_locked"}]
    )

    resolved_train = resolve_source_rows(fake_rows, "source_train")
    resolved_val = resolve_source_rows(fake_rows, "source_val")
    assert len(resolved_train) == EXPECTED_ROLE_COUNTS["source_train"]
    assert len(resolved_val) == EXPECTED_ROLE_COUNTS["source_val"]

    try:
        resolve_source_rows(fake_rows, "unseen_locked")
        raise AssertionError("Leakage guard failed.")
    except RuntimeError:
        pass

    assert FROZEN_SEEDS == (20260817, 20260818, 20260819)
    assert 20260820 not in FROZEN_SEEDS
    print("FROZEN_SEED_SET_TEST_PASS")
    print("FROZEN_COUNT_PARITY_TEST_PASS")
    print("STRUCTURE_LOSS_TEST_PASS")
    print("METRIC_TEST_PASS")
    print("PRANET_FORWARD_TEST_PASS")
    print("UNSEEN_ROLE_GUARD_TEST_PASS")
    print("SELF_TEST_PASS")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train S01-B PraNet source-only for one frozen seed."
    )
    parser.add_argument(
        "--seed",
        type=int,
        choices=FROZEN_SEEDS,
        default=20260817,
        help="Frozen S01 seed. Only 20260817/18/19 are permitted.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Optional explicit output directory. Default is derived from --seed.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=NUM_WORKERS,
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="CPU only (debug use).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete this exact S01-B seed output and rerun. "
             "Do not use after scientific results are accepted/frozen.",
    )
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main():
    global SEED

    args = parse_args()
    if args.self_test:
        self_test()
        return 0

    if args.seed not in FROZEN_SEEDS:
        raise RuntimeError(
            f"Seed {args.seed} is not in frozen S01 seeds: {FROZEN_SEEDS}"
        )

    SEED = int(args.seed)

    if args.output_dir is None:
        args.output_dir = (
            ROOT / "outputs" / f"S01_B_pranet_source_only_seed{SEED}_v1"
        )

    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
