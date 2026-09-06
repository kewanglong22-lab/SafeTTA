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


VERSION = "2026-08-19-Q1-R02-v1-fix2"
BUILD = "Q1_R02_SEGFORMER_B0_THREE_SEED_SOURCE_TRAINING_SOURCE_LAYOUT_DIAGNOSTIC_FIX2"

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
# Source-interface resolver — FIX1
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
    v = var_name.lower().replace("-", "_")

    exact = {
        "train_images": "train_images",
        "training_images": "train_images",
        "train_imgs": "train_images",
        "train_image_dir": "train_images",
        "train_images_dir": "train_images",
        "train_img_dir": "train_images",
        "train_masks": "train_masks",
        "training_masks": "train_masks",
        "train_mask_dir": "train_masks",
        "train_masks_dir": "train_masks",
        "train_labels": "train_masks",
        "train_label_dir": "train_masks",
        "val_images": "val_images",
        "valid_images": "val_images",
        "validation_images": "val_images",
        "val_imgs": "val_images",
        "val_image_dir": "val_images",
        "val_images_dir": "val_images",
        "validation_image_dir": "val_images",
        "val_masks": "val_masks",
        "valid_masks": "val_masks",
        "validation_masks": "val_masks",
        "val_mask_dir": "val_masks",
        "val_masks_dir": "val_masks",
        "validation_mask_dir": "val_masks",
        "val_labels": "val_masks",
    }
    if v in exact:
        return exact[v]

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
    raw = str(raw).strip().strip("\"'")
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
    """Direct Python/config style: train_images = 'F:\\...'."""
    pattern = re.compile(
        r"""(?m)^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*[rRuUbBfF]*["']([^"'\r\n]+)["']"""
    )
    return pattern.findall(text)


def extract_mapping_paths(text: str):
    """
    Dict/JSON/YAML-ish forms:
        "train_images": "..."
        train_images: "..."
    """
    pattern = re.compile(
        r"""(?im)["']?([A-Za-z_][A-Za-z0-9_\-]*)["']?\s*:\s*[rRuUbBfF]*["']([^"'\r\n]+)["']"""
    )
    return pattern.findall(text)


def extract_argparse_default_paths(text: str):
    """
    argparse forms such as:
        add_argument("--train-images", default=r"F:\\...")
    """
    out = []
    block_pat = re.compile(
        r"""add_argument\s*\((.{0,800}?)\)""",
        flags=re.IGNORECASE | re.DOTALL,
    )
    for block in block_pat.findall(text):
        flag = re.search(
            r"""["']--([A-Za-z0-9_\-]+)["']""",
            block,
            flags=re.IGNORECASE,
        )
        default = re.search(
            r"""default\s*=\s*[rRuUbBfF]*["']([^"'\r\n]+)["']""",
            block,
            flags=re.IGNORECASE,
        )
        if flag and default:
            out.append((flag.group(1), default.group(1)))
    return out


class _StaticPathEvaluator:
    """
    Conservative AST evaluator for path expressions only.
    It never executes project code.

    Supported examples:
        ROOT = Path(r"F:\\MEDSEG_SAFETTA")
        TRAIN = ROOT / "data" / "train" / "images"
        TRAIN = os.path.join(ROOT, "data", "train", "images")
        HERE = Path(__file__).resolve().parent
    """

    def __init__(self, source_file: Path, root: Path):
        self.source_file = source_file.resolve()
        self.root = root.resolve()
        self.env = {
            "__file__": self.source_file,
        }

    def _call_name(self, func):
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            parts = []
            cur = func
            while isinstance(cur, ast.Attribute):
                parts.append(cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name):
                parts.append(cur.id)
                return ".".join(reversed(parts))
        return ""

    def eval(self, node):
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (str, int)):
                return node.value
            return None

        if isinstance(node, ast.Name):
            return self.env.get(node.id)

        if isinstance(node, ast.JoinedStr):
            pieces = []
            for value in node.values:
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    pieces.append(value.value)
                elif isinstance(value, ast.FormattedValue):
                    v = self.eval(value.value)
                    if v is None:
                        return None
                    pieces.append(str(v))
                else:
                    return None
            return "".join(pieces)

        if isinstance(node, ast.BinOp):
            left = self.eval(node.left)
            right = self.eval(node.right)
            if left is None or right is None:
                return None
            if isinstance(node.op, ast.Div):
                try:
                    return Path(left) / str(right)
                except Exception:
                    return None
            if isinstance(node.op, ast.Add):
                try:
                    return str(left) + str(right)
                except Exception:
                    return None
            return None

        if isinstance(node, ast.Attribute):
            base = self.eval(node.value)
            if base is None:
                return None
            try:
                if node.attr == "parent":
                    return Path(base).parent
                if node.attr == "name":
                    return Path(base).name
            except Exception:
                return None
            return None

        if isinstance(node, ast.Subscript):
            base_node = node.value
            if isinstance(base_node, ast.Attribute) and base_node.attr == "parents":
                base = self.eval(base_node.value)
                idx = self.eval(node.slice)
                if base is not None and isinstance(idx, int):
                    try:
                        return Path(base).parents[idx]
                    except Exception:
                        return None
            return None

        if isinstance(node, ast.Call):
            name = self._call_name(node.func)

            if name in {"Path", "pathlib.Path"} and len(node.args) == 1:
                v = self.eval(node.args[0])
                if v is not None:
                    return Path(v)
                return None

            if name in {"os.path.join", "posixpath.join", "ntpath.join"}:
                vals = [self.eval(a) for a in node.args]
                if any(v is None for v in vals):
                    return None
                return Path(os.path.join(*(str(v) for v in vals)))

            if isinstance(node.func, ast.Attribute):
                base = self.eval(node.func.value)
                if base is not None:
                    if node.func.attr == "resolve" and not node.args:
                        try:
                            return Path(base).resolve()
                        except Exception:
                            return Path(base)
                    if node.func.attr == "joinpath":
                        vals = [self.eval(a) for a in node.args]
                        if any(v is None for v in vals):
                            return None
                        p = Path(base)
                        for v in vals:
                            p = p / str(v)
                        return p
            return None

        return None

    def assignments(self, tree):
        results = []
        for node in tree.body:
            target_names = []
            value_node = None

            if isinstance(node, ast.Assign):
                value_node = node.value
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        target_names.append(target.id)

            elif isinstance(node, ast.AnnAssign):
                value_node = node.value
                if isinstance(node.target, ast.Name):
                    target_names.append(node.target.id)

            if not target_names or value_node is None:
                continue

            value = self.eval(value_node)
            if value is None:
                continue

            for name in target_names:
                self.env[name] = value
                if isinstance(value, (Path, str)):
                    results.append((name, str(value)))
        return results


def extract_ast_path_assignments(text: str, source_file: Path, root: Path):
    if source_file.suffix.lower() != ".py":
        return []
    try:
        tree = ast.parse(text)
    except Exception:
        return []
    evaluator = _StaticPathEvaluator(source_file, root)
    return evaluator.assignments(tree)


def extract_r01c_evidence_paths(root: Path):
    """Use paths already surfaced by the successful Q1-R01C audit."""
    evidence_files = []
    if R01C_SOURCE_CANDIDATES.exists():
        rows, _ = read_csv(R01C_SOURCE_CANDIDATES)
        for r in rows:
            rp = (r.get("relative_path") or "").strip()
            if rp:
                p = root / rp
                if p.exists() and p.is_file():
                    evidence_files.append(p.resolve())

    if R01C_CODE_CANDIDATES.exists():
        rows, _ = read_csv(R01C_CODE_CANDIDATES)
        for r in rows:
            rp = (r.get("relative_path") or "").strip()
            if rp:
                p = root / rp
                if p.exists() and p.is_file():
                    evidence_files.append(p.resolve())

    unique = {}
    for p in evidence_files:
        unique[str(p).lower()] = p
    return list(unique.values())


def existing_child_dir(root_dir: Path, names: Sequence[str]) -> Optional[Path]:
    if not root_dir.exists() or not root_dir.is_dir():
        return None
    try:
        children = {
            p.name.lower(): p
            for p in root_dir.iterdir()
            if p.is_dir()
        }
    except Exception:
        return None
    for name in names:
        if name.lower() in children:
            return children[name.lower()].resolve()
    return None


def expand_root_candidate(category: str, path: Path):
    if not path.exists() or not path.is_dir():
        return {}

    image_names = (
        "images", "image", "imgs", "img",
        "train_images", "val_images", "validation_images",
    )
    mask_names = (
        "masks", "mask", "gt", "gts", "labels", "label",
        "train_masks", "val_masks", "validation_masks",
    )

    if category == "train_root":
        img = existing_child_dir(path, image_names)
        mask = existing_child_dir(path, mask_names)
        out = {}
        if img:
            out["train_images"] = img
        if mask:
            out["train_masks"] = mask
        return out

    if category == "val_root":
        img = existing_child_dir(path, image_names)
        mask = existing_child_dir(path, mask_names)
        out = {}
        if img:
            out["val_images"] = img
        if mask:
            out["val_masks"] = mask
        return out

    return {}


def _add_found_path(
    found: Dict[str, List[Path]],
    category: str,
    raw_path: str,
    source_file: Path,
    root: Path,
):
    for resolved in resolve_string_path(raw_path, source_file, root):
        if not resolved.exists() or not resolved.is_dir():
            continue
        if is_forbidden_target_path(resolved):
            continue

        if category in {
            "train_images", "train_masks",
            "val_images", "val_masks",
        }:
            found[category].append(resolved.resolve())

        elif category in {"train_root", "val_root"}:
            expanded = expand_root_candidate(category, resolved)
            for k, v in expanded.items():
                if not is_forbidden_target_path(v):
                    found[k].append(v.resolve())


def parse_source_file_for_interface(
    source_file: Path,
    root: Path,
):
    text = safe_read_text(source_file)
    if not text:
        return None

    found: Dict[str, List[Path]] = defaultdict(list)

    assignments = []
    assignments.extend(extract_path_assignments(text))
    assignments.extend(extract_mapping_paths(text))
    assignments.extend(extract_argparse_default_paths(text))
    assignments.extend(extract_ast_path_assignments(text, source_file, root))

    # Stable dedup preserving order.
    seen_pairs = set()
    deduped = []
    for var, raw in assignments:
        key = (str(var).lower(), str(raw).lower())
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        deduped.append((var, raw))

    for var, raw_path in deduped:
        category = categorize_variable(var)
        if category is None:
            continue
        _add_found_path(
            found,
            category,
            raw_path,
            source_file,
            root,
        )

    normalized = {}
    for k, vals in found.items():
        unique = {}
        for v in vals:
            unique[str(v.resolve()).lower()] = v.resolve()
        if unique:
            normalized[k] = list(unique.values())

    score = sum(
        k in normalized
        for k in (
            "train_images",
            "train_masks",
            "val_images",
            "val_masks",
        )
    )

    if not score:
        return None

    return {
        "source_file": source_file,
        "score": score,
        "paths": normalized,
        "assignment_count": len(deduped),
    }


def source_interface_candidates(root: Path):
    """
    FIX1:
    - scan Q1-R01C evidence files first;
    - then all normal project text/config files;
    - support direct strings, mappings, argparse defaults and safe AST path
      expressions.
    """
    priority_files = extract_r01c_evidence_paths(root)

    all_files = list(iter_project_text_files(root))
    ordered = []
    seen = set()

    for p in priority_files + all_files:
        try:
            key = str(p.resolve()).lower()
        except Exception:
            key = str(p).lower()
        if key in seen:
            continue
        seen.add(key)

        # Do not parse current fix script as evidence.
        if p.name in {
            Path(__file__).name,
            "Q1_R02_segformer_b0_three_seed_source_training_v1.py",
            "Q1_R02_segformer_b0_three_seed_source_training_v1_fix1.py",
        }:
            continue
        ordered.append(p)

    per_file = []
    for p in tqdm(
        ordered,
        desc="Q1-R02 FIX1 resolving source interface",
        unit="file",
        dynamic_ncols=True,
    ):
        result = parse_source_file_for_interface(p, root)
        if result is not None:
            per_file.append(result)

    per_file.sort(
        key=lambda x: (
            -x["score"],
            0 if x["source_file"] in priority_files else 1,
            str(x["source_file"]).lower(),
        )
    )
    return per_file


def _name_kind(name: str) -> Optional[str]:
    n = name.lower().replace("-", "_").strip()

    image_tokens = {"image", "images", "img", "imgs"}
    mask_tokens = {"mask", "masks", "gt", "gts", "label", "labels"}

    if n in image_tokens:
        return "images"
    if n in mask_tokens:
        return "masks"

    if "image" in n or re.search(r"(^|_)imgs?($|_)", n):
        if "mask" not in n and "label" not in n:
            return "images"

    if (
        "mask" in n
        or "label" in n
        or n.endswith("_gt")
        or n.startswith("gt_")
    ):
        return "masks"

    return None


def _path_role(path: Path) -> Optional[str]:
    parts = [p.lower().replace("-", "_") for p in path.parts[-5:]]

    # Exact-ish role terms first.
    train_hit = any(
        (
            p == "train"
            or p == "training"
            or "traindataset" in p
            or "train_dataset" in p
            or "source_train" in p
        )
        for p in parts
    )
    val_hit = any(
        (
            p == "val"
            or p == "valid"
            or p == "validation"
            or "valdataset" in p
            or "val_dataset" in p
            or "source_val" in p
            or "source_validation" in p
        )
        for p in parts
    )

    if train_hit and not val_hit:
        return "train"
    if val_hit and not train_hit:
        return "val"
    return None


def structural_pair_candidates(root: Path):
    """
    Directory-name-only fallback. It does not open images.

    It searches project dataset/source-like areas for high-confidence sibling
    image/mask directory pairs under train/validation ancestors.
    """
    skip_names = {
        ".git", ".conda", "env", "venv", "__pycache__",
        "outputs", "cache", "checkpoints", "models", "weights",
        "code", "docs", "configs",
    }

    candidate_dirs = []

    for current, dirs, files in os.walk(root, topdown=True):
        cp = Path(current)

        # Never enter known target-domain paths.
        dirs[:] = sorted(
            d for d in dirs
            if d.lower() not in skip_names
            and not is_forbidden_target_path(cp / d)
            and not (cp / d).is_symlink()
        )

        if is_forbidden_target_path(cp):
            dirs[:] = []
            continue

        kind = _name_kind(cp.name)
        role = _path_role(cp)
        if kind and role:
            candidate_dirs.append(
                {
                    "path": cp.resolve(),
                    "kind": kind,
                    "role": role,
                }
            )

    pairs = []
    for role in ("train", "val"):
        images = [x for x in candidate_dirs if x["role"] == role and x["kind"] == "images"]
        masks = [x for x in candidate_dirs if x["role"] == role and x["kind"] == "masks"]

        for im in images:
            for ma in masks:
                # High-confidence same-parent or parents sharing same split dir.
                same_parent = im["path"].parent == ma["path"].parent
                same_grandparent = (
                    im["path"].parent.parent == ma["path"].parent.parent
                    and _path_role(im["path"].parent) == role
                    and _path_role(ma["path"].parent) == role
                )
                if not (same_parent or same_grandparent):
                    continue

                score = 3 if same_parent else 2
                pairs.append(
                    {
                        "role": role,
                        "images": im["path"],
                        "masks": ma["path"],
                        "score": score,
                    }
                )

    # Dedup.
    unique = {}
    for p in pairs:
        key = (
            p["role"],
            str(p["images"]).lower(),
            str(p["masks"]).lower(),
        )
        if key not in unique or p["score"] > unique[key]["score"]:
            unique[key] = p

    return sorted(
        unique.values(),
        key=lambda x: (
            x["role"],
            -x["score"],
            str(x["images"]).lower(),
            str(x["masks"]).lower(),
        )
    )


def build_resolution_diagnostic(root: Path, text_candidates, structural_pairs):
    global_by_cat: Dict[str, Dict[str, Path]] = {
        "train_images": {},
        "train_masks": {},
        "val_images": {},
        "val_masks": {},
    }
    for c in text_candidates:
        for cat in global_by_cat:
            for p in c["paths"].get(cat, []):
                global_by_cat[cat][str(p.resolve()).lower()] = p.resolve()

    return {
        "text_candidates": [
            {
                "source_file": rel(c["source_file"], root),
                "score": c["score"],
                "assignment_count": c.get("assignment_count", 0),
                "paths": {
                    k: [str(p) for p in v]
                    for k, v in c["paths"].items()
                },
            }
            for c in text_candidates[:30]
        ],
        "global_text_paths": {
            k: sorted(str(v) for v in d.values())
            for k, d in global_by_cat.items()
        },
        "structural_pairs": [
            {
                "role": p["role"],
                "images": str(p["images"]),
                "masks": str(p["masks"]),
                "score": p["score"],
            }
            for p in structural_pairs[:30]
        ],
    }


def choose_auto_interface(root: Path):
    text_candidates = source_interface_candidates(root)

    # 1) strongest text/config evidence: one unique complete signature.
    complete = []
    for c in text_candidates:
        if c["score"] != 4:
            continue
        if all(
            len(c["paths"].get(k, [])) == 1
            for k in (
                "train_images",
                "train_masks",
                "val_images",
                "val_masks",
            )
        ):
            complete.append(c)

    signatures = {}
    for c in complete:
        sig = tuple(
            str(c["paths"][k][0].resolve()).lower()
            for k in (
                "train_images",
                "train_masks",
                "val_images",
                "val_masks",
            )
        )
        signatures.setdefault(sig, c)

    if len(signatures) == 1:
        c = next(iter(signatures.values()))
        return {
            k: c["paths"][k][0].resolve()
            for k in (
                "train_images",
                "train_masks",
                "val_images",
                "val_masks",
            )
        }, {
            "resolution_mode": "FIX1_AUTO_SINGLE_COMPLETE_SIGNATURE",
            "evidence_file": rel(c["source_file"], root),
            "candidate_file_count": len(text_candidates),
            "resolver_fix": "v1-fix1",
        }

    # 2) global text evidence: each category has exactly one unique directory.
    global_by_cat: Dict[str, Dict[str, Path]] = {
        "train_images": {},
        "train_masks": {},
        "val_images": {},
        "val_masks": {},
    }
    for c in text_candidates:
        for cat in global_by_cat:
            for p in c["paths"].get(cat, []):
                global_by_cat[cat][str(p.resolve()).lower()] = p.resolve()

    if all(len(global_by_cat[k]) == 1 for k in global_by_cat):
        interface = {
            k: next(iter(global_by_cat[k].values()))
            for k in global_by_cat
        }
        return interface, {
            "resolution_mode": "FIX1_AUTO_GLOBAL_UNIQUE",
            "evidence_file": "",
            "candidate_file_count": len(text_candidates),
            "resolver_fix": "v1-fix1",
        }

    # 3) conservative structural fallback.
    structural = structural_pair_candidates(root)
    train_pairs = [p for p in structural if p["role"] == "train"]
    val_pairs = [p for p in structural if p["role"] == "val"]

    # Only accept if there is exactly one top-score signature for each split.
    def unique_top(pairs):
        if not pairs:
            return None
        top = max(p["score"] for p in pairs)
        top_pairs = [p for p in pairs if p["score"] == top]
        signatures = {}
        for p in top_pairs:
            key = (str(p["images"]).lower(), str(p["masks"]).lower())
            signatures[key] = p
        if len(signatures) == 1:
            return next(iter(signatures.values()))
        return None

    tp = unique_top(train_pairs)
    vp = unique_top(val_pairs)

    if tp is not None and vp is not None:
        return {
            "train_images": tp["images"],
            "train_masks": tp["masks"],
            "val_images": vp["images"],
            "val_masks": vp["masks"],
        }, {
            "resolution_mode": "FIX1_AUTO_STRUCTURAL_UNIQUE",
            "evidence_file": "",
            "candidate_file_count": len(text_candidates),
            "resolver_fix": "v1-fix1",
        }

    diagnostic = build_resolution_diagnostic(
        root,
        text_candidates,
        structural,
    )
    raise RuntimeError(
        f"{DECISION_SOURCE}: FIX1 could not resolve one unique source interface. "
        f"Diagnostic={json.dumps(diagnostic, ensure_ascii=False)[:16000]}"
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
        interface = {
            k: Path(v).resolve()
            for k, v in provided.items()
        }
        meta = {
            "resolution_mode": "CLI_EXPLICIT_EXISTING_INTERFACE",
            "evidence_file": "",
            "candidate_file_count": None,
            "resolver_fix": "v1-fix1",
        }
    else:
        interface, meta = choose_auto_interface(root)

    for k, p in interface.items():
        if not p.exists() or not p.is_dir():
            raise RuntimeError(
                f"{DECISION_SOURCE}: {k} does not exist: {p}"
            )
        if is_forbidden_target_path(p):
            raise RuntimeError(
                f"Forbidden target-domain token found in {k}: {p}"
            )

    vals = [str(interface[k]).lower() for k in sorted(interface)]
    if len(set(vals)) < 4:
        raise RuntimeError(
            f"{DECISION_SOURCE}: source interface directories overlap unexpectedly."
        )

    return interface, meta


def run_resolve_only(args):
    root = args.root.resolve()
    if not root.exists():
        raise FileNotFoundError(root)

    print("===== Q1-R02 SOURCE INTERFACE RESOLUTION ONLY =====")
    print(f"Script version: {VERSION}")
    print(f"Build: {BUILD}")
    print("Model download=NO")
    print("Training=NO")
    print("Image opening=NO")
    print()

    provenance = validate_upstream()
    interface, meta = resolve_interface(args, root)

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

    print("RESOLUTION_PASS")
    print(f"resolution_mode={meta['resolution_mode']}")
    if meta.get("evidence_file"):
        print(f"evidence_file={meta['evidence_file']}")
    print(f"train_images={interface['train_images']}")
    print(f"train_masks={interface['train_masks']}")
    print(f"val_images={interface['val_images']}")
    print(f"val_masks={interface['val_masks']}")
    print(f"train_pairs={len(train_pairs)}")
    print(f"val_pairs={len(val_pairs)}")
    print("train_val_overlap=0")
    print("Model download=NO")
    print("Training=NO")
    return 0



# ---------------------------------------------------------------------
# FIX2 source-layout / split-manifest diagnostic
# ---------------------------------------------------------------------

S00_LOCKED_ROOT = ROOT / "data" / "processed" / "S00_polyp_locked_v1"

MANIFEST_SUFFIXES = {
    ".csv", ".json", ".jsonl", ".txt", ".tsv",
    ".yaml", ".yml", ".ini", ".cfg", ".toml",
}

SPLIT_HINT_TOKENS = (
    "split", "fold", "train", "val", "valid", "validation",
    "manifest", "index", "list", "case", "sample",
)

PATH_FIELD_HINTS = (
    "image", "img", "mask", "label", "gt", "path", "file", "filename",
)

SPLIT_FIELD_HINTS = (
    "split", "subset", "partition", "fold", "phase", "set",
)


def _count_image_like_files(directory: Path):
    if not directory.exists() or not directory.is_dir():
        return 0
    count = 0
    for p in directory.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            count += 1
    return count


def _summarize_directory_tree(base: Path, max_depth: int = 4):
    rows = []
    if not base.exists():
        return rows

    base_depth = len(base.parts)

    for current, dirs, files in os.walk(base, topdown=True):
        cp = Path(current)
        depth = len(cp.parts) - base_depth

        if depth >= max_depth:
            dirs[:] = []
        else:
            dirs[:] = sorted(
                d for d in dirs
                if not (cp / d).is_symlink()
                and d.lower() not in {"__pycache__", ".git"}
            )

        image_like_count = sum(
            1
            for name in files
            if Path(name).suffix.lower() in IMAGE_EXTENSIONS
        )
        manifest_like_count = sum(
            1
            for name in files
            if Path(name).suffix.lower() in MANIFEST_SUFFIXES
        )

        rows.append({
            "relative_dir": rel(cp, base),
            "depth": depth,
            "subdir_count": len(dirs),
            "file_count": len(files),
            "image_like_files_direct": image_like_count,
            "manifest_like_files_direct": manifest_like_count,
        })

    rows.sort(
        key=lambda r: (
            int(r["depth"]),
            str(r["relative_dir"]).lower(),
        )
    )
    return rows


def _candidate_manifest_files(base: Path):
    if not base.exists():
        return []

    out = []
    for p in base.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in MANIFEST_SUFFIXES:
            continue
        try:
            if p.stat().st_size > 16 * 1024 * 1024:
                continue
        except Exception:
            continue

        low = str(p).lower()
        name_hit = any(tok in low for tok in SPLIT_HINT_TOKENS)

        # Keep all small table/config files under the locked processed root,
        # but score split-like filenames higher.
        out.append((0 if name_hit else 1, p.resolve()))

    out.sort(key=lambda x: (x[0], str(x[1]).lower()))
    return [p for _, p in out]


def _csv_tsv_preview(path: Path, delimiter: str):
    result = {
        "format": "csv" if delimiter == "," else "tsv",
        "columns": [],
        "row_count_scanned": 0,
        "split_like_columns": [],
        "path_like_columns": [],
        "unique_split_values_preview": {},
        "first_rows_preview": [],
    }

    try:
        with path.open("r", newline="", encoding="utf-8-sig", errors="ignore") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            fields = reader.fieldnames or []
            result["columns"] = fields
            result["split_like_columns"] = [
                c for c in fields
                if any(tok in c.lower() for tok in SPLIT_FIELD_HINTS)
            ]
            result["path_like_columns"] = [
                c for c in fields
                if any(tok in c.lower() for tok in PATH_FIELD_HINTS)
            ]

            split_values = {
                c: set()
                for c in result["split_like_columns"]
            }

            for i, row in enumerate(reader):
                if i < 5:
                    result["first_rows_preview"].append(
                        {
                            k: (v[:300] if isinstance(v, str) else v)
                            for k, v in row.items()
                        }
                    )

                for c in split_values:
                    v = (row.get(c) or "").strip()
                    if v and len(split_values[c]) < 30:
                        split_values[c].add(v)

                result["row_count_scanned"] += 1
                if result["row_count_scanned"] >= 200000:
                    break

            result["unique_split_values_preview"] = {
                c: sorted(vals)
                for c, vals in split_values.items()
            }
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"

    return result


def _json_preview(path: Path):
    result = {
        "format": "json",
        "top_level_type": "",
        "top_level_keys": [],
        "list_length": None,
        "split_key_hits": [],
        "path_key_hits": [],
        "preview": None,
    }

    try:
        raw = path.read_text(encoding="utf-8", errors="ignore")
        obj = json.loads(raw)

        result["top_level_type"] = type(obj).__name__

        if isinstance(obj, dict):
            keys = [str(k) for k in obj.keys()]
            result["top_level_keys"] = keys[:100]
            result["split_key_hits"] = [
                k for k in keys
                if any(tok in k.lower() for tok in SPLIT_FIELD_HINTS)
            ]
            result["path_key_hits"] = [
                k for k in keys
                if any(tok in k.lower() for tok in PATH_FIELD_HINTS)
            ]
            preview_items = list(obj.items())[:10]
            result["preview"] = {
                str(k): v
                for k, v in preview_items
            }

        elif isinstance(obj, list):
            result["list_length"] = len(obj)
            result["preview"] = obj[:5]

            if obj and isinstance(obj[0], dict):
                keys = sorted({
                    str(k)
                    for item in obj[:100]
                    if isinstance(item, dict)
                    for k in item.keys()
                })
                result["top_level_keys"] = keys
                result["split_key_hits"] = [
                    k for k in keys
                    if any(tok in k.lower() for tok in SPLIT_FIELD_HINTS)
                ]
                result["path_key_hits"] = [
                    k for k in keys
                    if any(tok in k.lower() for tok in PATH_FIELD_HINTS)
                ]

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"

    return result


def _text_preview(path: Path):
    result = {
        "format": path.suffix.lower().lstrip("."),
        "line_count_scanned": 0,
        "split_keyword_lines": [],
        "first_nonempty_lines": [],
    }

    try:
        text = safe_read_text(path, max_bytes=2 * 1024 * 1024)
        lines = text.splitlines()

        for line_no, line in enumerate(lines[:20000], 1):
            stripped = line.strip()

            if stripped and len(result["first_nonempty_lines"]) < 20:
                result["first_nonempty_lines"].append(
                    f"{line_no}: {stripped[:500]}"
                )

            low = stripped.lower()
            if (
                stripped
                and any(tok in low for tok in SPLIT_HINT_TOKENS)
                and len(result["split_keyword_lines"]) < 100
            ):
                result["split_keyword_lines"].append(
                    f"{line_no}: {stripped[:800]}"
                )

            result["line_count_scanned"] += 1

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"

    return result


def _manifest_preview(path: Path):
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return _csv_tsv_preview(path, ",")

    if suffix == ".tsv":
        return _csv_tsv_preview(path, "\t")

    if suffix == ".json":
        return _json_preview(path)

    return _text_preview(path)


def _project_references_to_locked_root(root: Path):
    needles = (
        "S00_polyp_locked_v1",
        "source_train",
    )

    rows = []

    for p in tqdm(
        list(iter_project_text_files(root)),
        desc="Q1-R02 FIX2 scanning S00/split references",
        unit="file",
        dynamic_ncols=True,
    ):
        # Do not use Q1-R02 scripts themselves as evidence.
        if p.name.startswith("Q1_R02_segformer_b0_three_seed_source_training"):
            continue

        text = safe_read_text(p, max_bytes=4 * 1024 * 1024)
        if not text:
            continue

        lines = text.splitlines()

        hit_lines = []
        for i, line in enumerate(lines, 1):
            low = line.lower()
            if any(n.lower() in low for n in needles):
                start = max(0, i - 3)
                end = min(len(lines), i + 2)
                context = " || ".join(
                    f"{j+1}:{lines[j].strip()[:500]}"
                    for j in range(start, end)
                )
                hit_lines.append(context)

            if len(hit_lines) >= 20:
                break

        if hit_lines:
            rows.append({
                "relative_path": rel(p, root),
                "reference_count_preview": len(hit_lines),
                "context_preview": " ### ".join(hit_lines),
            })

    rows.sort(key=lambda r: r["relative_path"].lower())
    return rows


def _discover_sibling_validation_candidates(locked_root: Path):
    rows = []

    if not locked_root.exists():
        return rows

    # Inspect directories under locked root and one parent level for likely
    # source-validation names, but do not open image contents.
    roots = [locked_root]
    if locked_root.parent.exists():
        roots.append(locked_root.parent)

    seen = set()
    for base in roots:
        try:
            dirs = [
                p
                for p in base.rglob("*")
                if p.is_dir()
            ]
        except Exception:
            dirs = []

        for p in dirs:
            key = str(p.resolve()).lower()
            if key in seen:
                continue
            seen.add(key)

            low_name = p.name.lower()
            low_path = str(p).lower()

            val_name_hit = (
                low_name in {"val", "valid", "validation", "source_val", "source_validation"}
                or "source_val" in low_path
                or "validation" in low_name
            )
            if not val_name_hit:
                continue

            image_child = existing_child_dir(
                p,
                ("images", "image", "imgs", "img", "Imgs"),
            )
            mask_child = existing_child_dir(
                p,
                ("masks", "mask", "gt", "gts", "GT", "labels", "label"),
            )

            rows.append({
                "candidate_dir": str(p.resolve()),
                "images_child": str(image_child) if image_child else "",
                "masks_child": str(mask_child) if mask_child else "",
                "direct_image_like_count": sum(
                    1
                    for q in p.iterdir()
                    if q.is_file() and q.suffix.lower() in IMAGE_EXTENSIONS
                ),
                "has_image_mask_children": bool(image_child and mask_child),
            })

    rows.sort(key=lambda r: r["candidate_dir"].lower())
    return rows


def run_inspect_source_layout(args):
    root = args.root.resolve()

    if not root.exists():
        raise FileNotFoundError(root)

    print("===== Q1-R02 FIX2 SOURCE-LAYOUT DIAGNOSTIC =====")
    print(f"Script version: {VERSION}")
    print(f"Build: {BUILD}")
    print(f"Project root: {root}")
    print(f"Expected locked source root: {S00_LOCKED_ROOT}")
    print("Model download=NO")
    print("Training=NO")
    print("Checkpoint loading=NO")
    print("Image opening=NO")
    print()

    provenance = validate_upstream()

    locked_root = S00_LOCKED_ROOT.resolve()

    if not locked_root.exists():
        raise RuntimeError(
            f"Locked source root does not exist: {locked_root}"
        )

    out_dir = (
        root
        / "outputs"
        / "Q1_R02_source_interface_diagnostic_fix2"
    )
    if out_dir.exists():
        raise FileExistsError(
            f"Diagnostic output already exists: {out_dir}"
        )

    build_dir = Path(str(out_dir) + "__building")
    if build_dir.exists():
        raise FileExistsError(
            f"Diagnostic build directory already exists: {build_dir}"
        )
    build_dir.mkdir(parents=True, exist_ok=False)

    tree_rows = _summarize_directory_tree(
        locked_root,
        max_depth=5,
    )
    tree_path = build_dir / "locked_source_tree_summary.csv"
    write_csv(
        tree_path,
        tree_rows,
        [
            "relative_dir",
            "depth",
            "subdir_count",
            "file_count",
            "image_like_files_direct",
            "manifest_like_files_direct",
        ],
    )

    manifest_files = _candidate_manifest_files(locked_root)

    manifest_rows = []
    manifest_detail = {}

    for p in tqdm(
        manifest_files,
        desc="Q1-R02 FIX2 inspecting split/manifest files",
        unit="file",
        dynamic_ncols=True,
    ):
        preview = _manifest_preview(p)
        rp = rel(p, locked_root)

        manifest_rows.append({
            "relative_path": rp,
            "extension": p.suffix.lower(),
            "bytes": p.stat().st_size,
            "format": preview.get("format", ""),
            "columns_or_keys": ";".join(
                preview.get("columns", [])
                or preview.get("top_level_keys", [])
            )[:4000],
            "split_hits": ";".join(
                preview.get("split_like_columns", [])
                or preview.get("split_key_hits", [])
            )[:2000],
            "path_hits": ";".join(
                preview.get("path_like_columns", [])
                or preview.get("path_key_hits", [])
            )[:2000],
            "error": preview.get("error", ""),
        })

        manifest_detail[rp] = preview

    manifest_index_path = build_dir / "manifest_candidate_index.csv"
    write_csv(
        manifest_index_path,
        manifest_rows,
        [
            "relative_path",
            "extension",
            "bytes",
            "format",
            "columns_or_keys",
            "split_hits",
            "path_hits",
            "error",
        ],
    )

    manifest_detail_path = build_dir / "manifest_candidate_details.json"
    write_json(
        manifest_detail_path,
        manifest_detail,
    )

    refs = _project_references_to_locked_root(root)
    refs_path = build_dir / "project_s00_split_references.csv"
    write_csv(
        refs_path,
        refs,
        [
            "relative_path",
            "reference_count_preview",
            "context_preview",
        ],
    )

    siblings = _discover_sibling_validation_candidates(locked_root)
    siblings_path = build_dir / "validation_directory_candidates.csv"
    write_csv(
        siblings_path,
        siblings,
        [
            "candidate_dir",
            "images_child",
            "masks_child",
            "direct_image_like_count",
            "has_image_mask_children",
        ],
    )

    train_images = locked_root / "source_train" / "images"
    train_masks = locked_root / "source_train" / "masks"

    train_pair_status = {
        "train_images_exists": train_images.exists(),
        "train_masks_exists": train_masks.exists(),
        "train_image_count": (
            _count_image_like_files(train_images)
            if train_images.exists()
            else 0
        ),
        "train_mask_count": (
            _count_image_like_files(train_masks)
            if train_masks.exists()
            else 0
        ),
    }

    likely_manifest_count = sum(
        1
        for r in manifest_rows
        if r["split_hits"]
    )

    likely_val_dir_count = sum(
        1
        for r in siblings
        if r["has_image_mask_children"]
    )

    diagnostic = {
        "script_version": VERSION,
        "build": BUILD,
        "locked_source_root": str(locked_root),
        "train_pair_status": train_pair_status,
        "tree_rows": len(tree_rows),
        "manifest_candidates": len(manifest_rows),
        "manifest_candidates_with_split_fields": likely_manifest_count,
        "project_reference_files": len(refs),
        "validation_directory_candidates": len(siblings),
        "validation_directory_candidates_with_image_mask_children": likely_val_dir_count,
        "model_download": False,
        "training": False,
        "checkpoint_loading": False,
        "image_opening": False,
        "q1_r01c_lock_sha256": provenance["q1_r01c_lock_sha256"],
        "q1_r01c_decision": provenance["q1_r01c_decision"],
    }

    diagnostic_path = build_dir / "diagnostic_summary.json"
    write_json(diagnostic_path, diagnostic)

    run_log = f"""===== Q1-R02 FIX2 SOURCE-LAYOUT DIAGNOSTIC =====
Script version: {VERSION}
Build: {BUILD}

Locked source root:
  {locked_root}

Known physical source-train pair:
  images exists={train_pair_status['train_images_exists']}
  masks exists={train_pair_status['train_masks_exists']}
  image files={train_pair_status['train_image_count']}
  mask files={train_pair_status['train_mask_count']}

Directory audit:
  tree summary rows={len(tree_rows)}
  validation-directory candidates={len(siblings)}
  validation candidates with images+masks={likely_val_dir_count}

Manifest/split audit:
  candidate files={len(manifest_rows)}
  candidates with explicit split/fold-like fields={likely_manifest_count}

Project reference audit:
  files referencing S00/source_train={len(refs)}

Model download=NO
Training=NO
Checkpoint loading=NO
Image opening=NO

Diagnostic output:
  {out_dir}

NEXT:
  Inspect manifest_candidate_index.csv,
  validation_directory_candidates.csv,
  and project_s00_split_references.csv.

[OK] SOURCE_LAYOUT_DIAGNOSTIC_COMPLETE
"""

    run_log_path = build_dir / "run_log.txt"
    run_log_path.write_text(run_log, encoding="utf-8")

    # Hash diagnostic artifacts.
    artifact_paths = {
        "tree_summary": tree_path,
        "manifest_index": manifest_index_path,
        "manifest_details": manifest_detail_path,
        "project_references": refs_path,
        "validation_candidates": siblings_path,
        "diagnostic_summary": diagnostic_path,
        "run_log": run_log_path,
    }

    lock = {
        "script_version": VERSION,
        "build": BUILD,
        "q1_r01c_lock_sha256": provenance["q1_r01c_lock_sha256"],
        "model_download": False,
        "training": False,
        "checkpoint_loading": False,
        "image_opening": False,
        "artifacts": {
            k: {
                "filename": p.name,
                "sha256": file_sha256(p),
            }
            for k, p in artifact_paths.items()
        },
        "decision": "SOURCE_LAYOUT_DIAGNOSTIC_COMPLETE",
    }

    lock_path = build_dir / "Q1_R02_SOURCE_LAYOUT_DIAGNOSTIC_FIX2_LOCK.json"
    write_json(lock_path, lock)
    lock_sha = file_sha256(lock_path)

    build_dir.rename(out_dir)

    print()
    print((out_dir / "run_log.txt").read_text(encoding="utf-8"))
    print(
        "Q1-R02 FIX2 DIAGNOSTIC LOCK:",
        out_dir / "Q1_R02_SOURCE_LAYOUT_DIAGNOSTIC_FIX2_LOCK.json",
    )
    print("LOCK SHA256:", lock_sha)

    return 0


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
        "--inspect-source-layout",
        action="store_true",
        help=(
            "Audit S00 locked source directory structure, manifest/split files, "
            "and project references only; do not download a model or train."
        ),
    )
    parser.add_argument(
        "--resolve-only",
        action="store_true",
        help=(
            "Resolve and validate source train/mask/val interface only; "
            "do not download the model or train."
        ),
    )
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

    if args.inspect_source_layout:
        try:
            return run_inspect_source_layout(args)
        except Exception:
            traceback.print_exc()
            raise

    if args.resolve_only:
        try:
            return run_resolve_only(args)
        except Exception:
            traceback.print_exc()
            raise

    try:
        run(args)
    except Exception:
        traceback.print_exc()
        raise

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
