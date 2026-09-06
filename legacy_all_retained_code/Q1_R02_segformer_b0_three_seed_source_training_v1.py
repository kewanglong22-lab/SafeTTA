#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Q1-R02 — SegFormer-B0 Three-Seed Source Training

Frozen architecture:
    SegFormer-B0 / MiT-B0

Frozen seeds:
    20260820, 20260821, 20260822

This script trains source-only segmentation models. It must not use target
data or utility labels.

Expected project root:
    F:\\MEDSEG_SAFETTA
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import random
import re
import shutil
import tempfile
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm


VERSION = "2026-08-19-Q1-R02-v1"
BUILD = "Q1_R02_SEGFORMER_B0_THREE_SEED_SOURCE_TRAINING"

ROOT = Path(r"F:\MEDSEG_SAFETTA")
OUTPUT_DIR = ROOT / "outputs" / "Q1_R02_segformer_b0_three_seed_source_training_v1"

PROTOCOL = (
    ROOT
    / "docs"
    / "Q1_R02_segformer_b0_three_seed_source_training_preregistered_protocol_v1.md"
)
EXPECTED_PROTOCOL_SHA256 = (
    "f69e620c0fa2b96f2998a97c6e70d6b00fd8421fb7ce2f37461fddef154921c1"
)

Q1_R01C_DIR = (
    ROOT
    / "outputs"
    / "Q1_R01C_third_architecture_selection_training_plan_v1"
)
Q1_R01C_LOCK = Q1_R01C_DIR / "Q1_R01C_THIRD_ARCHITECTURE_PLAN_LOCK.json"
EXPECTED_Q1_R01C_LOCK_SHA256 = (
    "6c65597f59c3dfaf9827200af4f7dec4e4f7ee04896b972ee238d00aa9c3bf28"
)
EXPECTED_Q1_R01C_DECISION = "READY_TO_IMPLEMENT_SEGFORMER_B0_TRAINING"

R01C_SOURCE_CANDIDATES = Q1_R01C_DIR / "source_data_interface_candidates.csv"
R01C_CODE_CANDIDATES = Q1_R01C_DIR / "training_code_reference_candidates.csv"
R01C_SOURCE_PROTOCOL = Q1_R01C_DIR / "source_protocol_evidence.json"

ARCHITECTURE = "SegFormer-B0"
PRETRAINED_MODEL = "nvidia/mit-b0"
SEEDS = (20260820, 20260821, 20260822)

IMAGE_SIZE = 352
TRAIN_BATCH_SIZE = 4
VAL_BATCH_SIZE = 4
NUM_WORKERS = 0
MAX_EPOCHS = 50
LEARNING_RATE = 6e-5
WEIGHT_DECAY = 0.01
MIN_LR = 1e-6
GRAD_CLIP_NORM = 1.0
HFLIP_P = 0.5
VFLIP_P = 0.5

HF_CACHE = ROOT / "cache" / "huggingface"

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
}

TEXT_SUFFIXES = {
    ".py", ".ps1", ".sh", ".json", ".yaml", ".yml",
    ".toml", ".md", ".txt", ".ini", ".cfg"
}

EXCLUDED_TEXT_SCAN_DIRS = {
    ".git", ".conda", "env", "venv", "__pycache__",
    "external", "external_data", "external_datasets",
}

TARGET_FORBIDDEN_TOKENS = (
    "polypgen",
    "target_domain",
    "target-data",
    "target_data",
    "targetdataset",
)

MASK_SUFFIXES = (
    "_mask", "-mask", "_masks",
    "_gt", "-gt",
    "_label", "-label",
    "_seg", "-seg",
)

DECISION_READY = "SEGFORMER_B0_THREE_STATE_PANEL_READY"
DECISION_SOURCE = "SOURCE_INTERFACE_RESOLUTION_REQUIRED"
DECISION_CUDA = "CUDA_REQUIRED_FOR_Q1_R02"
DECISION_INIT = "SEGFORMER_PRETRAINED_INITIALIZATION_FAILURE"
DECISION_DUP = "INVALID_SEGFORMER_STATE_DUPLICATION"


# ---------------------------------------------------------------------
# File / serialization helpers
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


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Sequence[dict], fields: Sequence[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(fields))
        w.writeheader()
        w.writerows(rows)


def read_csv(path: Path):
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return list(reader), reader.fieldnames or []


def safe_read_text(path: Path, max_bytes: int = 4 * 1024 * 1024) -> str:
    try:
        if path.stat().st_size > max_bytes:
            with path.open("rb") as f:
                raw = f.read(max_bytes)
            return raw.decode("utf-8", errors="ignore")
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return str(path)


def is_forbidden_target_path(path: Path) -> bool:
    low = str(path).replace("\\", "/").lower()
    return any(token in low for token in TARGET_FORBIDDEN_TOKENS)


# ---------------------------------------------------------------------
# Frozen provenance
# ---------------------------------------------------------------------

def validate_upstream():
    if not PROTOCOL.exists():
        raise FileNotFoundError(PROTOCOL)
    protocol_sha = file_sha256(PROTOCOL)
    if protocol_sha != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError(
            "Q1-R02 protocol SHA mismatch: "
            f"expected={EXPECTED_PROTOCOL_SHA256} actual={protocol_sha}"
        )

    if not Q1_R01C_LOCK.exists():
        raise FileNotFoundError(Q1_R01C_LOCK)

    upstream_sha = file_sha256(Q1_R01C_LOCK)
    if upstream_sha != EXPECTED_Q1_R01C_LOCK_SHA256:
        raise RuntimeError(
            "Q1-R01C lock SHA mismatch: "
            f"expected={EXPECTED_Q1_R01C_LOCK_SHA256} actual={upstream_sha}"
        )

    lock = json.loads(Q1_R01C_LOCK.read_text(encoding="utf-8"))
    if lock.get("decision") != EXPECTED_Q1_R01C_DECISION:
        raise RuntimeError(
            f"Unexpected Q1-R01C decision: {lock.get('decision')}"
        )

    if lock.get("third_variant") != "SegFormer-B0":
        raise RuntimeError(
            f"Upstream architecture changed: {lock.get('third_variant')}"
        )

    if tuple(lock.get("new_seeds", [])) != SEEDS:
        raise RuntimeError(
            f"Upstream seeds changed: {lock.get('new_seeds')}"
        )

    return {
        "protocol_sha256": protocol_sha,
        "q1_r01c_lock_sha256": upstream_sha,
        "q1_r01c_decision": lock["decision"],
        "architecture": ARCHITECTURE,
        "pretrained_model": PRETRAINED_MODEL,
        "seeds": list(SEEDS),
    }


# ---------------------------------------------------------------------
# Source-interface resolver
# ---------------------------------------------------------------------

def iter_project_text_files(root: Path):
    roots = [
        root / "code",
        root / "configs",
        root / "docs",
        root / "outputs",
    ]
    roots = [p for p in roots if p.exists()]
    if not roots:
        roots = [root]

    seen = set()
    for sr in roots:
        for current, dirs, files in os.walk(sr, topdown=True):
            cp = Path(current)
            dirs[:] = sorted(
                d for d in dirs
                if d.lower() not in EXCLUDED_TEXT_SCAN_DIRS
                and not (cp / d).is_symlink()
            )
            for name in sorted(files):
                p = cp / name
                if p.suffix.lower() not in TEXT_SUFFIXES:
                    continue
                try:
                    if p.stat().st_size > 8 * 1024 * 1024:
                        continue
                    key = str(p.resolve()).lower()
                except Exception:
                    continue
                if key in seen:
                    continue
                seen.add(key)
                yield p


def categorize_variable(var_name: str) -> Optional[str]:
    v = var_name.lower()

    train = ("train" in v) or ("training" in v)
    val = (
        ("val" in v)
        or ("valid" in v)
        or ("validation" in v)
    )
    image = ("image" in v) or ("images" in v) or ("img" in v)
    mask = (
        ("mask" in v)
        or ("masks" in v)
        or ("label" in v)
        or ("labels" in v)
        or v.endswith("_gt")
        or v.startswith("gt_")
    )

    if train and image and not mask:
        return "train_images"
    if train and mask:
        return "train_masks"
    if val and image and not mask:
        return "val_images"
    if val and mask:
        return "val_masks"
    if train and ("root" in v or "path" in v or "dir" in v):
        return "train_root"
    if val and ("root" in v or "path" in v or "dir" in v):
        return "val_root"
    return None


def resolve_string_path(raw: str, source_file: Path, root: Path) -> List[Path]:
    raw = raw.strip().strip("\"'")
    if not raw:
        return []

    raw = os.path.expandvars(raw)
    p = Path(raw)

    candidates = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append(source_file.parent / p)
        candidates.append(root / p)

    out = []
    seen = set()
    for c in candidates:
        try:
            cr = c.resolve()
        except Exception:
            cr = c
        key = str(cr).lower()
        if key not in seen:
            seen.add(key)
            out.append(cr)
    return out


def extract_path_assignments(text: str):
    pattern = re.compile(
        r"""(?m)^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*[rRuUbBfF]*["']([^"'\r\n]+)["']"""
    )
    return pattern.findall(text)


def existing_child_dir(root_dir: Path, names: Sequence[str]) -> Optional[Path]:
    if not root_dir.exists() or not root_dir.is_dir():
        return None
    children = {p.name.lower(): p for p in root_dir.iterdir() if p.is_dir()}
    for name in names:
        if name.lower() in children:
            return children[name.lower()].resolve()
    return None


def expand_root_candidate(category: str, path: Path):
    if not path.exists() or not path.is_dir():
        return {}

    if category == "train_root":
        img = existing_child_dir(
            path,
            ("images", "image", "imgs", "img", "Imgs"),
        )
        mask = existing_child_dir(
            path,
            ("masks", "mask", "gt", "gts", "GT", "labels", "label"),
        )
        out = {}
        if img:
            out["train_images"] = img
        if mask:
            out["train_masks"] = mask
        return out

    if category == "val_root":
        img = existing_child_dir(
            path,
            ("images", "image", "imgs", "img", "Imgs"),
        )
        mask = existing_child_dir(
            path,
            ("masks", "mask", "gt", "gts", "GT", "labels", "label"),
        )
        out = {}
        if img:
            out["val_images"] = img
        if mask:
            out["val_masks"] = mask
        return out

    return {}


def source_interface_candidates(root: Path):
    per_file = []

    for p in tqdm(
        list(iter_project_text_files(root)),
        desc="Q1-R02 resolving source interface",
        unit="file",
        dynamic_ncols=True,
    ):
        # Avoid feeding this Q1-R02 script back into its own resolver.
        if p.name == Path(__file__).name:
            continue

        text = safe_read_text(p)
        if not text:
            continue

        found: Dict[str, List[Path]] = defaultdict(list)

        for var, raw_path in extract_path_assignments(text):
            category = categorize_variable(var)
            if category is None:
                continue

            for resolved in resolve_string_path(raw_path, p, root):
                if not resolved.exists() or not resolved.is_dir():
                    continue
                if is_forbidden_target_path(resolved):
                    continue

                if category in {
                    "train_images", "train_masks",
                    "val_images", "val_masks"
                }:
                    found[category].append(resolved)
                elif category in {"train_root", "val_root"}:
                    expanded = expand_root_candidate(category, resolved)
                    for k, v in expanded.items():
                        if not is_forbidden_target_path(v):
                            found[k].append(v)

        normalized = {}
        for k, vals in found.items():
            unique = {}
            for v in vals:
                unique[str(v).lower()] = v
            if unique:
                normalized[k] = list(unique.values())

        score = sum(k in normalized for k in (
            "train_images",
            "train_masks",
            "val_images",
            "val_masks",
        ))

        if score:
            per_file.append({
                "source_file": p,
                "score": score,
                "paths": normalized,
            })

    per_file.sort(
        key=lambda x: (
            -x["score"],
            str(x["source_file"]).lower(),
        )
    )
    return per_file


def choose_auto_interface(root: Path):
    candidates = source_interface_candidates(root)

    # Strongest case: one file provides all four exact interface directories.
    complete = []
    for c in candidates:
        if c["score"] != 4:
            continue
        if all(len(c["paths"].get(k, [])) == 1 for k in (
            "train_images", "train_masks", "val_images", "val_masks"
        )):
            complete.append(c)

    signatures = {}
    for c in complete:
        sig = tuple(
            str(c["paths"][k][0].resolve()).lower()
            for k in ("train_images", "train_masks", "val_images", "val_masks")
        )
        signatures.setdefault(sig, c)

    if len(signatures) == 1:
        c = next(iter(signatures.values()))
        return {
            k: c["paths"][k][0].resolve()
            for k in ("train_images", "train_masks", "val_images", "val_masks")
        }, {
            "resolution_mode": "AUTO_SINGLE_COMPLETE_SIGNATURE",
            "evidence_file": rel(c["source_file"], root),
            "candidate_file_count": len(candidates),
        }

    # Conservative global fallback: each category must have one unique
    # existing directory across all evidence files.
    global_by_cat: Dict[str, Dict[str, Path]] = {
        "train_images": {},
        "train_masks": {},
        "val_images": {},
        "val_masks": {},
    }
    for c in candidates:
        for cat in global_by_cat:
            for p in c["paths"].get(cat, []):
                global_by_cat[cat][str(p.resolve()).lower()] = p.resolve()

    if all(len(global_by_cat[k]) == 1 for k in global_by_cat):
        interface = {
            k: next(iter(global_by_cat[k].values()))
            for k in global_by_cat
        }
        return interface, {
            "resolution_mode": "AUTO_GLOBAL_UNIQUE",
            "evidence_file": "",
            "candidate_file_count": len(candidates),
        }

    details = {
        k: sorted(str(v) for v in d.values())
        for k, d in global_by_cat.items()
    }
    raise RuntimeError(
        f"{DECISION_SOURCE}: automatic resolution is ambiguous/incomplete. "
        f"Candidates={json.dumps(details, ensure_ascii=False)}"
    )


def resolve_interface(args, root: Path):
    override_values = {
        "train_images": args.train_images,
        "train_masks": args.train_masks,
        "val_images": args.val_images,
        "val_masks": args.val_masks,
    }

    provided = {k: v for k, v in override_values.items() if v is not None}

    if provided and len(provided) != 4:
        raise RuntimeError(
            f"{DECISION_SOURCE}: if CLI overrides are used, provide all four "
            "--train-images --train-masks --val-images --val-masks."
        )

    if len(provided) == 4:
        interface = {k: Path(v).resolve() for k, v in provided.items()}
        meta = {
            "resolution_mode": "CLI_EXPLICIT_EXISTING_INTERFACE",
            "evidence_file": "",
            "candidate_file_count": None,
        }
    else:
        interface, meta = choose_auto_interface(root)

    for k, p in interface.items():
        if not p.exists() or not p.is_dir():
            raise RuntimeError(f"{DECISION_SOURCE}: {k} does not exist: {p}")
        if is_forbidden_target_path(p):
            raise RuntimeError(
                f"Forbidden target-domain token found in {k}: {p}"
            )

    # Ensure four directories are not literally the same.
    vals = [str(interface[k]).lower() for k in sorted(interface)]
    if len(set(vals)) < 4:
        raise RuntimeError(
            f"{DECISION_SOURCE}: source interface directories overlap unexpectedly."
        )

    return interface, meta


# ---------------------------------------------------------------------
# Image-mask pairing
# ---------------------------------------------------------------------

def normalize_stem(stem: str) -> str:
    s = stem.lower().strip()
    changed = True
    while changed:
        changed = False
        for suffix in MASK_SUFFIXES:
            if s.endswith(suffix):
                s = s[:-len(suffix)]
                changed = True
    return s.strip("_- ")


def enumerate_image_files(directory: Path) -> List[Path]:
    files = [
        p.resolve()
        for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    files.sort(key=lambda p: str(p).lower())
    return files


def build_pairing(image_dir: Path, mask_dir: Path, split_name: str):
    images = enumerate_image_files(image_dir)
    masks = enumerate_image_files(mask_dir)

    if not images:
        raise RuntimeError(f"No {split_name} images found in {image_dir}")
    if not masks:
        raise RuntimeError(f"No {split_name} masks found in {mask_dir}")

    exact_masks: Dict[str, List[Path]] = defaultdict(list)
    norm_masks: Dict[str, List[Path]] = defaultdict(list)

    for m in masks:
        exact_masks[m.stem.lower()].append(m)
        norm_masks[normalize_stem(m.stem)].append(m)

    duplicate_norm = {
        k: [str(p) for p in v]
        for k, v in norm_masks.items()
        if len(v) > 1
    }
    if duplicate_norm:
        raise RuntimeError(
            f"Duplicate normalized {split_name} mask stems: "
            f"{json.dumps(duplicate_norm, ensure_ascii=False)[:4000]}"
        )

    pairs = []
    unmatched = []
    used_masks = set()

    for img in images:
        exact = exact_masks.get(img.stem.lower(), [])
        chosen = None

        if len(exact) == 1:
            chosen = exact[0]
        elif len(exact) > 1:
            raise RuntimeError(
                f"Multiple exact masks for {split_name} image {img}"
            )
        else:
            norm = norm_masks.get(normalize_stem(img.stem), [])
            if len(norm) == 1:
                chosen = norm[0]
            elif len(norm) > 1:
                raise RuntimeError(
                    f"Multiple normalized masks for {split_name} image {img}"
                )

        if chosen is None:
            unmatched.append(str(img))
            continue

        if chosen in used_masks:
            raise RuntimeError(
                f"Mask reused for multiple {split_name} images: {chosen}"
            )
        used_masks.add(chosen)

        pairs.append((img, chosen))

    if unmatched:
        raise RuntimeError(
            f"Unmatched {split_name} images: {len(unmatched)}. "
            f"Examples={unmatched[:10]}"
        )

    if len(pairs) != len(images):
        raise RuntimeError(
            f"Incomplete {split_name} pairing: {len(pairs)}/{len(images)}"
        )

    return pairs, {
        "split": split_name,
        "image_dir": str(image_dir),
        "mask_dir": str(mask_dir),
        "image_count": len(images),
        "mask_count": len(masks),
        "paired_count": len(pairs),
        "all_images_paired": True,
        "mask_reuse_detected": False,
    }


def assert_train_val_disjoint(train_pairs, val_pairs):
    train_images = {str(p[0]).lower() for p in train_pairs}
    val_images = {str(p[0]).lower() for p in val_pairs}
    overlap = train_images & val_images
    if overlap:
        raise RuntimeError(
            f"Train/val image-path overlap detected: {list(overlap)[:10]}"
        )


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

def set_seed(seed: int, torch_module=None):
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))

    if torch_module is not None:
        torch_module.manual_seed(seed)
        if torch_module.cuda.is_available():
            torch_module.cuda.manual_seed_all(seed)
        try:
            torch_module.backends.cudnn.benchmark = False
            torch_module.backends.cudnn.deterministic = True
        except Exception:
            pass


# ---------------------------------------------------------------------
# Training implementation
# ---------------------------------------------------------------------

def train_all_seeds(
    build_dir: Path,
    train_pairs,
    val_pairs,
    provenance: dict,
):
    # Project-local Hugging Face caches. Set before Transformers model loading.
    HF_CACHE.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(HF_CACHE)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(HF_CACHE / "hub")

    try:
        import torch
        import torch.nn.functional as F
        from PIL import Image
        from torch.utils.data import DataLoader, Dataset
        from transformers import (
            AutoImageProcessor,
            SegformerConfig,
            SegformerForSemanticSegmentation,
        )
    except Exception as e:
        raise RuntimeError(
            f"{DECISION_INIT}: dependency import failed: {type(e).__name__}: {e}"
        ) from e

    if not torch.cuda.is_available():
        raise RuntimeError(DECISION_CUDA)

    device = torch.device("cuda")

    try:
        processor = AutoImageProcessor.from_pretrained(
            PRETRAINED_MODEL,
            cache_dir=str(HF_CACHE),
            do_reduce_labels=False,
        )
        mean = np.asarray(processor.image_mean, dtype=np.float32)
        std = np.asarray(processor.image_std, dtype=np.float32)
        if mean.shape != (3,) or std.shape != (3,):
            raise RuntimeError(
                f"Unexpected image processor normalization: mean={mean} std={std}"
            )
    except Exception as e:
        raise RuntimeError(
            f"{DECISION_INIT}: image processor initialization failed: "
            f"{type(e).__name__}: {e}"
        ) from e

    class PairedDataset(Dataset):
        def __init__(self, pairs, augment: bool, seed: int):
            self.pairs = list(pairs)
            self.augment = augment
            self.rng = random.Random(seed)

        def __len__(self):
            return len(self.pairs)

        def __getitem__(self, idx):
            img_path, mask_path = self.pairs[idx]

            with Image.open(img_path) as im:
                image = im.convert("RGB").resize(
                    (IMAGE_SIZE, IMAGE_SIZE),
                    resample=Image.Resampling.BILINEAR,
                )

            with Image.open(mask_path) as mm:
                mask = mm.convert("L").resize(
                    (IMAGE_SIZE, IMAGE_SIZE),
                    resample=Image.Resampling.NEAREST,
                )

            if self.augment:
                if self.rng.random() < HFLIP_P:
                    image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                    mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                if self.rng.random() < VFLIP_P:
                    image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                    mask = mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

            arr = np.asarray(image, dtype=np.float32) / 255.0
            arr = (arr - mean[None, None, :]) / std[None, None, :]
            arr = np.transpose(arr, (2, 0, 1)).copy()

            mask_arr = np.asarray(mask, dtype=np.uint8)
            mask_arr = (mask_arr > 0).astype(np.int64)

            return (
                torch.from_numpy(arr),
                torch.from_numpy(mask_arr),
                str(img_path),
            )

    def build_model():
        try:
            config = SegformerConfig.from_pretrained(
                PRETRAINED_MODEL,
                cache_dir=str(HF_CACHE),
                num_labels=2,
                id2label={0: "background", 1: "foreground"},
                label2id={"background": 0, "foreground": 1},
            )
            model = SegformerForSemanticSegmentation.from_pretrained(
                PRETRAINED_MODEL,
                cache_dir=str(HF_CACHE),
                config=config,
                ignore_mismatched_sizes=True,
            )
            return model
        except Exception as e:
            raise RuntimeError(
                f"{DECISION_INIT}: model initialization failed: "
                f"{type(e).__name__}: {e}"
            ) from e

    def segmentation_loss(logits, mask):
        logits_up = F.interpolate(
            logits,
            size=mask.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        ce = F.cross_entropy(logits_up, mask)

        prob = torch.softmax(logits_up, dim=1)[:, 1]
        target = mask.float()

        dims = (1, 2)
        inter = torch.sum(prob * target, dim=dims)
        denom = torch.sum(prob, dim=dims) + torch.sum(target, dim=dims)
        dice = (2.0 * inter + 1e-6) / (denom + 1e-6)
        dice_loss = 1.0 - dice.mean()

        total = 0.5 * ce + 0.5 * dice_loss
        return total, ce.detach(), dice_loss.detach(), logits_up

    def hard_dice_per_image(logits_up, mask):
        prob = torch.softmax(logits_up, dim=1)[:, 1]
        pred = prob >= 0.5
        target = mask.bool()

        vals = []
        for i in range(pred.shape[0]):
            p = pred[i]
            t = target[i]
            p_sum = int(p.sum().item())
            t_sum = int(t.sum().item())
            if p_sum == 0 and t_sum == 0:
                vals.append(1.0)
            else:
                inter = int((p & t).sum().item())
                vals.append(
                    (2.0 * inter) / max(p_sum + t_sum, 1)
                )
        return vals

    # Torch AMP API varies by version. Keep compatibility.
    def make_scaler():
        try:
            return torch.amp.GradScaler("cuda", enabled=True)
        except Exception:
            return torch.cuda.amp.GradScaler(enabled=True)

    def autocast_context():
        try:
            return torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=True,
            )
        except Exception:
            return torch.cuda.amp.autocast(enabled=True)

    summaries = []

    for seed in SEEDS:
        print()
        print("=" * 72)
        print(f"Q1-R02 training seed {seed}")
        print("=" * 72)

        set_seed(seed, torch)

        train_dataset = PairedDataset(
            train_pairs,
            augment=True,
            seed=seed,
        )
        val_dataset = PairedDataset(
            val_pairs,
            augment=False,
            seed=seed + 1000,
        )

        train_gen = torch.Generator()
        train_gen.manual_seed(seed)

        train_loader = DataLoader(
            train_dataset,
            batch_size=TRAIN_BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            generator=train_gen,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=VAL_BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

        model = build_model().to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=MAX_EPOCHS,
            eta_min=MIN_LR,
        )
        scaler = make_scaler()

        seed_dir = build_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=False)

        best_path = seed_dir / "best_model_state.pt"
        last_path = seed_dir / "last_model_state.pt"
        history_path = seed_dir / "training_history.csv"
        seed_summary_path = seed_dir / "seed_summary.json"

        best_dice = -float("inf")
        best_val_loss = float("inf")
        best_epoch = None

        history = []

        for epoch in range(1, MAX_EPOCHS + 1):
            model.train()
            train_loss_sum = 0.0
            train_ce_sum = 0.0
            train_dice_loss_sum = 0.0
            train_items = 0

            pbar = tqdm(
                train_loader,
                desc=f"seed {seed} epoch {epoch:02d}/{MAX_EPOCHS} train",
                unit="batch",
                dynamic_ncols=True,
            )

            for images, masks, _ in pbar:
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with autocast_context():
                    outputs = model(pixel_values=images)
                    loss, ce, dice_loss, _ = segmentation_loss(
                        outputs.logits,
                        masks,
                    )

                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"Non-finite training loss seed={seed} epoch={epoch}"
                    )

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    GRAD_CLIP_NORM,
                )
                scaler.step(optimizer)
                scaler.update()

                bs = images.shape[0]
                train_items += bs
                train_loss_sum += float(loss.detach().item()) * bs
                train_ce_sum += float(ce.item()) * bs
                train_dice_loss_sum += float(dice_loss.item()) * bs

                pbar.set_postfix(
                    loss=f"{train_loss_sum/max(train_items,1):.4f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                )

            model.eval()
            val_loss_sum = 0.0
            val_items = 0
            dice_values = []

            with torch.no_grad():
                vbar = tqdm(
                    val_loader,
                    desc=f"seed {seed} epoch {epoch:02d}/{MAX_EPOCHS} val",
                    unit="batch",
                    dynamic_ncols=True,
                )
                for images, masks, _ in vbar:
                    images = images.to(device, non_blocking=True)
                    masks = masks.to(device, non_blocking=True)

                    with autocast_context():
                        outputs = model(pixel_values=images)
                        loss, _, _, logits_up = segmentation_loss(
                            outputs.logits,
                            masks,
                        )

                    if not torch.isfinite(loss):
                        raise RuntimeError(
                            f"Non-finite validation loss seed={seed} epoch={epoch}"
                        )

                    bs = images.shape[0]
                    val_items += bs
                    val_loss_sum += float(loss.item()) * bs
                    dice_values.extend(
                        hard_dice_per_image(logits_up, masks)
                    )

                    vbar.set_postfix(
                        dice=f"{np.mean(dice_values):.4f}",
                        loss=f"{val_loss_sum/max(val_items,1):.4f}",
                    )

            train_loss = train_loss_sum / max(train_items, 1)
            train_ce = train_ce_sum / max(train_items, 1)
            train_dice_loss = train_dice_loss_sum / max(train_items, 1)
            val_loss = val_loss_sum / max(val_items, 1)
            val_dice = float(np.mean(dice_values))

            if not (
                math.isfinite(train_loss)
                and math.isfinite(val_loss)
                and math.isfinite(val_dice)
            ):
                raise RuntimeError(
                    f"Non-finite epoch metric seed={seed} epoch={epoch}"
                )

            row = {
                "epoch": epoch,
                "seed": seed,
                "train_loss": train_loss,
                "train_ce": train_ce,
                "train_dice_loss": train_dice_loss,
                "val_loss": val_loss,
                "val_mean_image_dice": val_dice,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            history.append(row)

            better = False
            if val_dice > best_dice + 1e-12:
                better = True
            elif abs(val_dice - best_dice) <= 1e-12:
                if val_loss < best_val_loss - 1e-12:
                    better = True
                elif (
                    abs(val_loss - best_val_loss) <= 1e-12
                    and (best_epoch is None or epoch < best_epoch)
                ):
                    better = True

            if better:
                best_dice = val_dice
                best_val_loss = val_loss
                best_epoch = epoch

                best_payload = {
                    "script_version": VERSION,
                    "architecture": ARCHITECTURE,
                    "pretrained_model": PRETRAINED_MODEL,
                    "seed": seed,
                    "epoch": epoch,
                    "val_mean_image_dice": val_dice,
                    "val_loss": val_loss,
                    "image_size": IMAGE_SIZE,
                    "num_labels": 2,
                    "state_dict": {
                        k: v.detach().cpu()
                        for k, v in model.state_dict().items()
                    },
                }
                torch.save(best_payload, best_path)

            scheduler.step()

            write_csv(
                history_path,
                history,
                [
                    "epoch",
                    "seed",
                    "train_loss",
                    "train_ce",
                    "train_dice_loss",
                    "val_loss",
                    "val_mean_image_dice",
                    "learning_rate",
                ],
            )

            print(
                f"[seed={seed}] epoch={epoch:02d} "
                f"train_loss={train_loss:.6f} "
                f"val_loss={val_loss:.6f} "
                f"val_dice={val_dice:.6f} "
                f"best={best_dice:.6f}@{best_epoch}"
            )

        last_payload = {
            "script_version": VERSION,
            "architecture": ARCHITECTURE,
            "pretrained_model": PRETRAINED_MODEL,
            "seed": seed,
            "epoch": MAX_EPOCHS,
            "val_mean_image_dice": history[-1]["val_mean_image_dice"],
            "val_loss": history[-1]["val_loss"],
            "image_size": IMAGE_SIZE,
            "num_labels": 2,
            "state_dict": {
                k: v.detach().cpu()
                for k, v in model.state_dict().items()
            },
        }
        torch.save(last_payload, last_path)

        if not best_path.exists():
            raise RuntimeError(
                f"Best checkpoint was never written for seed {seed}"
            )

        # Checkpoint structural readback.
        try:
            payload = torch.load(
                best_path,
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            payload = torch.load(best_path, map_location="cpu")

        if (
            payload.get("seed") != seed
            or payload.get("architecture") != ARCHITECTURE
            or "state_dict" not in payload
            or not payload["state_dict"]
        ):
            raise RuntimeError(
                f"Best checkpoint readback failed for seed {seed}"
            )

        best_sha = file_sha256(best_path)
        last_sha = file_sha256(last_path)

        seed_summary = {
            "seed": seed,
            "architecture": ARCHITECTURE,
            "pretrained_model": PRETRAINED_MODEL,
            "best_epoch": best_epoch,
            "best_val_mean_image_dice": best_dice,
            "best_val_loss": best_val_loss,
            "best_checkpoint": best_path.name,
            "best_checkpoint_sha256": best_sha,
            "last_checkpoint_sha256": last_sha,
            "train_images": len(train_pairs),
            "val_images": len(val_pairs),
            "completed_epochs": MAX_EPOCHS,
            "target_data_used": False,
            "utility_labels_used": False,
        }
        write_json(seed_summary_path, seed_summary)
        summaries.append(seed_summary)

        del model, optimizer, scheduler, scaler
        torch.cuda.empty_cache()

    return summaries


# ---------------------------------------------------------------------
# Self-test: no Transformers/model download
# ---------------------------------------------------------------------

def self_test():
    assert ARCHITECTURE == "SegFormer-B0"
    assert PRETRAINED_MODEL == "nvidia/mit-b0"
    assert SEEDS == (20260820, 20260821, 20260822)
    assert IMAGE_SIZE == 352
    assert TRAIN_BATCH_SIZE == 4
    assert MAX_EPOCHS == 50
    assert abs(LEARNING_RATE - 6e-5) < 1e-15

    assert normalize_stem("case001_mask") == "case001"
    assert normalize_stem("case001-GT") == "case001"
    assert categorize_variable("train_images") == "train_images"
    assert categorize_variable("train_masks") == "train_masks"
    assert categorize_variable("val_images") == "val_images"
    assert categorize_variable("validation_masks") == "val_masks"

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        train_i = root / "train_images"
        train_m = root / "train_masks"
        val_i = root / "val_images"
        val_m = root / "val_masks"

        for d in (train_i, train_m, val_i, val_m):
            d.mkdir()

        # Pairing works from filenames alone; content is not opened here.
        for idx in range(3):
            (train_i / f"case{idx}.png").write_bytes(b"x")
            (train_m / f"case{idx}_mask.png").write_bytes(b"y")

        for idx in range(2):
            (val_i / f"val{idx}.png").write_bytes(b"x")
            (val_m / f"val{idx}.png").write_bytes(b"y")

        train_pairs, train_audit = build_pairing(
            train_i,
            train_m,
            "train",
        )
        val_pairs, val_audit = build_pairing(
            val_i,
            val_m,
            "val",
        )

        assert len(train_pairs) == 3
        assert len(val_pairs) == 2
        assert train_audit["all_images_paired"] is True
        assert val_audit["all_images_paired"] is True
        assert_train_val_disjoint(train_pairs, val_pairs)

        code_dir = root / "code"
        code_dir.mkdir()
        cfg = code_dir / "train_cfg.py"
        cfg.write_text(
            (
                f"train_images = r'{train_i}'\n"
                f"train_masks = r'{train_m}'\n"
                f"val_images = r'{val_i}'\n"
                f"val_masks = r'{val_m}'\n"
            ),
            encoding="utf-8",
        )

        interface, meta = choose_auto_interface(root)
        assert interface["train_images"] == train_i.resolve()
        assert interface["train_masks"] == train_m.resolve()
        assert interface["val_images"] == val_i.resolve()
        assert interface["val_masks"] == val_m.resolve()

    print("FROZEN_CONSTANTS_TEST_PASS")
    print("PAIRING_TEST_PASS")
    print("TRAIN_VAL_DISJOINT_TEST_PASS")
    print("SOURCE_INTERFACE_RESOLVER_TEST_PASS")
    print("SELF_TEST_PASS")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def run(args):
    root = args.root.resolve()
    output_dir = args.output_dir.resolve()

    if not root.exists():
        raise FileNotFoundError(root)

    try:
        output_dir.relative_to(root)
    except Exception:
        raise RuntimeError(
            f"Output must remain under project root: {output_dir}"
        )

    if str(output_dir).lower().startswith("c:\\"):
        raise RuntimeError("C-drive output is forbidden.")

    if output_dir.exists():
        raise FileExistsError(
            f"Q1-R02 output already exists: {output_dir}"
        )

    build_dir = Path(str(output_dir) + "__building")
    if build_dir.exists():
        raise FileExistsError(
            f"Partial Q1-R02 build exists: {build_dir}. "
            "Remove only this __building directory if a technical run failed."
        )
    build_dir.mkdir(parents=True, exist_ok=False)

    print("===== Q1-R02 SEGFORMER-B0 THREE-SEED SOURCE TRAINING =====")
    print(f"Script version: {VERSION}")
    print(f"Build: {BUILD}")
    print(f"Architecture: {ARCHITECTURE}")
    print(f"Pretrained encoder: {PRETRAINED_MODEL}")
    print(f"Seeds: {SEEDS}")
    print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Epochs: {MAX_EPOCHS}")
    print(f"Batch size: {TRAIN_BATCH_SIZE}")
    print(f"HF cache: {HF_CACHE}")
    print("Target data allowed: NO")
    print("Utility labels allowed: NO")
    print("Hyperparameter sweep: NO")
    print()

    provenance = validate_upstream()

    interface, interface_meta = resolve_interface(args, root)

    train_pairs, train_pairing = build_pairing(
        interface["train_images"],
        interface["train_masks"],
        "train",
    )
    val_pairs, val_pairing = build_pairing(
        interface["val_images"],
        interface["val_masks"],
        "val",
    )
    assert_train_val_disjoint(train_pairs, val_pairs)

    resolved_interface_payload = {
        **{k: str(v) for k, v in interface.items()},
        **interface_meta,
        "target_path_token_check_pass": True,
    }
    resolved_interface_path = build_dir / "resolved_source_interface.json"
    write_json(
        resolved_interface_path,
        resolved_interface_payload,
    )

    pairing_path = build_dir / "pairing_audit.json"
    write_json(
        pairing_path,
        {
            "train": train_pairing,
            "validation": val_pairing,
            "train_val_path_overlap": 0,
            "target_forbidden_tokens": list(TARGET_FORBIDDEN_TOKENS),
        },
    )

    # Lazy CUDA/environment check before any model download.
    try:
        import torch
        import transformers
        import PIL
    except Exception as e:
        raise RuntimeError(
            f"{DECISION_INIT}: dependency import failed: "
            f"{type(e).__name__}: {e}"
        ) from e

    if not torch.cuda.is_available():
        decision_path = build_dir / "decision.txt"
        decision_path.write_text(DECISION_CUDA + "\n", encoding="utf-8")
        raise RuntimeError(DECISION_CUDA)

    environment = {
        "python": os.sys.version,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "pillow_version": getattr(PIL, "__version__", ""),
        "cuda_available": True,
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_name": torch.cuda.get_device_name(0),
        "cuda_capability": list(torch.cuda.get_device_capability(0)),
        "hf_home": str(HF_CACHE),
        "huggingface_hub_cache": str(HF_CACHE / "hub"),
    }
    environment_path = build_dir / "environment_audit.json"
    write_json(environment_path, environment)

    protocol_copy = build_dir / "preregistered_protocol_copy.md"
    shutil.copy2(PROTOCOL, protocol_copy)

    provenance.update({
        "root": str(root),
        "output_dir": str(output_dir),
        "resolved_source_interface": resolved_interface_payload,
        "train_image_count": len(train_pairs),
        "val_image_count": len(val_pairs),
        "image_size": IMAGE_SIZE,
        "batch_size": TRAIN_BATCH_SIZE,
        "epochs": MAX_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "target_data_used": False,
        "utility_labels_used": False,
        "hyperparameter_sweep": False,
    })

    print("Resolved source interface:")
    for k, v in interface.items():
        print(f"  {k}: {v}")
    print(
        f"Pairing: train={len(train_pairs)} "
        f"val={len(val_pairs)}"
    )
    print()

    summaries = train_all_seeds(
        build_dir,
        train_pairs,
        val_pairs,
        provenance,
    )

    summary_path = build_dir / "three_seed_summary.csv"
    write_csv(
        summary_path,
        summaries,
        [
            "seed",
            "architecture",
            "pretrained_model",
            "best_epoch",
            "best_val_mean_image_dice",
            "best_val_loss",
            "best_checkpoint",
            "best_checkpoint_sha256",
            "last_checkpoint_sha256",
            "train_images",
            "val_images",
            "completed_epochs",
            "target_data_used",
            "utility_labels_used",
        ],
    )

    best_shas = [r["best_checkpoint_sha256"] for r in summaries]
    unique_best_shas = len(set(best_shas))

    sha_audit = {
        "best_checkpoint_sha256": {
            str(r["seed"]): r["best_checkpoint_sha256"]
            for r in summaries
        },
        "unique_best_checkpoint_sha256_count": unique_best_shas,
        "required_unique_count": len(SEEDS),
        "all_unique": unique_best_shas == len(SEEDS),
    }
    sha_path = build_dir / "checkpoint_sha256_audit.json"
    write_json(sha_path, sha_audit)

    if unique_best_shas != len(SEEDS):
        decision = DECISION_DUP
    else:
        decision = DECISION_READY

    provenance["best_checkpoint_sha256"] = sha_audit[
        "best_checkpoint_sha256"
    ]
    provenance["decision"] = decision
    provenance_path = build_dir / "provenance_audit.json"
    write_json(provenance_path, provenance)

    decision_path = build_dir / "decision.txt"
    decision_path.write_text(decision + "\n", encoding="utf-8")

    run_log = f"""===== Q1-R02 SEGFORMER-B0 THREE-SEED SOURCE TRAINING =====
Script version: {VERSION}
Build: {BUILD}

Architecture:
  {ARCHITECTURE}
Pretrained encoder:
  {PRETRAINED_MODEL}

Frozen seeds:
  {SEEDS}

Source interface:
  train images={interface['train_images']}
  train masks={interface['train_masks']}
  val images={interface['val_images']}
  val masks={interface['val_masks']}

Pairing:
  train pairs={len(train_pairs)}
  validation pairs={len(val_pairs)}
  train/val image overlap=0

Frozen training:
  image size={IMAGE_SIZE}
  max epochs={MAX_EPOCHS}
  batch size={TRAIN_BATCH_SIZE}
  lr={LEARNING_RATE}
  weight decay={WEIGHT_DECAY}
  loss=0.5*CrossEntropy + 0.5*ForegroundSoftDice
  optimizer=AdamW
  scheduler=CosineAnnealingLR
  AMP=YES
  target data=NO
  utility labels=NO
  hyperparameter sweep=NO

Seed results:
"""
    for r in summaries:
        run_log += (
            f"  seed={r['seed']}: "
            f"best_epoch={r['best_epoch']} "
            f"best_val_dice={r['best_val_mean_image_dice']:.6f} "
            f"best_val_loss={r['best_val_loss']:.6f} "
            f"sha256={r['best_checkpoint_sha256']}\n"
        )

    run_log += f"""
Unique best checkpoint states:
  {unique_best_shas}/{len(SEEDS)}

Decision:
  {decision}

If ready:
  next = Q1-R03 Nine-State Heterogeneous Panel Prediction & Utility Asset Completion

[OK] Outputs: {output_dir}
"""
    run_log_path = build_dir / "run_log.txt"
    run_log_path.write_text(run_log, encoding="utf-8")

    if decision == DECISION_DUP:
        raise RuntimeError(DECISION_DUP)

    # Lock only after complete valid three-state training.
    artifact_paths = {
        "protocol_copy": protocol_copy,
        "resolved_source_interface": resolved_interface_path,
        "pairing_audit": pairing_path,
        "environment_audit": environment_path,
        "three_seed_summary": summary_path,
        "checkpoint_sha256_audit": sha_path,
        "provenance_audit": provenance_path,
        "decision": decision_path,
        "run_log": run_log_path,
    }

    for seed in SEEDS:
        seed_dir = build_dir / f"seed_{seed}"
        for name in (
            "best_model_state.pt",
            "last_model_state.pt",
            "training_history.csv",
            "seed_summary.json",
        ):
            p = seed_dir / name
            if not p.exists():
                raise RuntimeError(
                    f"Missing required seed artifact: {p}"
                )
            artifact_paths[f"seed_{seed}_{name}"] = p

    lock = {
        "script_version": VERSION,
        "build": BUILD,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "q1_r01c_lock_sha256": EXPECTED_Q1_R01C_LOCK_SHA256,
        "q1_r01c_decision": EXPECTED_Q1_R01C_DECISION,
        "architecture": ARCHITECTURE,
        "pretrained_model": PRETRAINED_MODEL,
        "seeds": list(SEEDS),
        "target_data_used": False,
        "utility_labels_used": False,
        "hyperparameter_sweep": False,
        "checkpoint_sha256_unique": True,
        "artifacts": {
            name: {
                "relative_path": rel(p, build_dir),
                "sha256": file_sha256(p),
            }
            for name, p in artifact_paths.items()
        },
        "decision": decision,
    }

    lock_path = build_dir / "Q1_R02_SEGFORMER_SOURCE_TRAINING_LOCK.json"
    write_json(lock_path, lock)
    lock_sha = file_sha256(lock_path)

    for name, p in artifact_paths.items():
        if file_sha256(p) != lock["artifacts"][name]["sha256"]:
            raise RuntimeError(
                f"Artifact changed before final commit: {name}"
            )

    build_dir.rename(output_dir)

    print()
    print((output_dir / "run_log.txt").read_text(encoding="utf-8"))
    print(
        "Q1-R02 LOCK:",
        output_dir / "Q1_R02_SEGFORMER_SOURCE_TRAINING_LOCK.json",
    )
    print("Q1-R02 LOCK SHA256:", lock_sha)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Q1-R02: train exactly three source-only SegFormer-B0 model "
            "states under the frozen SafeTTA source protocol."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
    )

    # Only for resolving an existing source interface if auto-resolution is
    # conservative/ambiguous. All four are required together.
    parser.add_argument("--train-images", type=Path, default=None)
    parser.add_argument("--train-masks", type=Path, default=None)
    parser.add_argument("--val-images", type=Path, default=None)
    parser.add_argument("--val-masks", type=Path, default=None)

    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run implementation tests without model download/training.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.self_test:
        self_test()
        return 0

    try:
        run(args)
    except Exception:
        traceback.print_exc()
        raise

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
