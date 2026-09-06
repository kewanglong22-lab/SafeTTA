#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Q1-R03 — Nine-State Heterogeneous Source Utility Asset Completion.

Source-side only. No target data. No utility predictor is fitted here.

Typical usage:
    python Q1_R03_nine_state_heterogeneous_source_utility_asset_v1.py --self-test
    python Q1_R03_nine_state_heterogeneous_source_utility_asset_v1.py --preflight-only
    python Q1_R03_nine_state_heterogeneous_source_utility_asset_v1.py
    python Q1_R03_nine_state_heterogeneous_source_utility_asset_v1.py --resume
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
import shutil
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm


VERSION = "2026-08-19-Q1-R03-v1-fix2"
BUILD = "Q1_R03_NINE_STATE_HETEROGENEOUS_SOURCE_UTILITY_ASSET_DETERMINISTIC_PARITY_FIX2"

ROOT = Path(r"F:\MEDSEG_SAFETTA")
# FIX2: exact frozen S05-C CUDA determinism environment.
# This must be set before importing torch.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
OUTPUT_DIR = ROOT / "outputs" / "Q1_R03_nine_state_heterogeneous_source_utility_asset_v1_fix2"

PROTOCOL = (
    ROOT / "docs"
    / "Q1_R03_nine_state_heterogeneous_source_utility_asset_preregistered_protocol_v1.md"
)
EXPECTED_PROTOCOL_SHA256 = "acc978a748d4b0a5f50aea26fb79d92a7c36a81c53de35a889203e9e11095b27"

MANIFEST = ROOT / "data" / "splits" / "S01_locked_protocol_manifest_v1.csv"
EXPECTED_MANIFEST_SHA256 = (
    "afb4bc1175af004819538cdbb22ee7a10f1be48035945c7e0d50fd2bbe2e3dc7"
)
DATA_ROOT = ROOT / "data" / "processed" / "S00_polyp_locked_v1"

R02_DIR = ROOT / "outputs" / "Q1_R02_segformer_b0_three_seed_source_training_v1_fix7"
R02_LOCK = R02_DIR / "Q1_R02_SEGFORMER_SOURCE_TRAINING_LOCK.json"
EXPECTED_R02_LOCK_SHA256 = (
    "42d27708b95211491df5c920a83a5223f7d6f3ee22cf1ff673609e6ec8b956a0"
)
EXPECTED_R02_DECISION = "SEGFORMER_B0_THREE_STATE_PANEL_READY"

Q1_S00A_DIR = ROOT / "outputs" / "Q1_S00A_iae_pre_adaptation_gradient_probe_v1"
Q1_S00A_LOCK = Q1_S00A_DIR / "Q1_S00A_IAE_FEASIBILITY_LOCK.json"
EXPECTED_Q1_S00A_LOCK_SHA256 = (
    "ed0d82f26f717975a7a1cf9d8bcf9e866a394087b38efdb9903858d4bc020a85"
)
Q1_S00A_FEATURES = Q1_S00A_DIR / "gradient_probe_features.csv"

S07B_DIR = ROOT / "outputs" / "S07_B_pranet_source_side_counterfactual_utility_dataset_v1"
S07B_TABLE = S07B_DIR / "source_side_counterfactual_utility_table.csv"
EXPECTED_S07B_TABLE_SHA256 = (
    "f8d36a10a41db8255847526312b8b2794d3d18c81a8d7b152a210b3c07bd0678"
)

PRANET_HELPER = ROOT / "code" / "S03_B_build_source_side_safettta_gate_v1.py"
EXPECTED_PRANET_HELPER_SHA256 = (
    "07ac65247216dea376e8996cbfbc98c54c62157ecdd1e9b7d9b539b5a74a2b5c"
)
PRANET_TRAINING_HELPER = ROOT / "code" / "S01_B_train_pranet_source_only_frozen_seeds_v1.py"
EXPECTED_PRANET_TRAINING_HELPER_SHA256 = (
    "2b2c9c1b75ff60bd087b403e43dc77468b31cbd54fb607481cc80ab47182b67a"
)

DEEPLAB_HELPER = ROOT / "code" / "S05_C_build_deeplab_source_side_safettta_gate_v1.py"
EXPECTED_DEEPLAB_HELPER_SHA256 = (
    "ab1d98f9847b2e338f07ec26ff426f9cd428e68da4813b90e2227782319aa424"
)
DEEPLAB_TRAINING_HELPER = (
    ROOT / "code" / "S05_B_train_deeplabv3_resnet50_source_only_frozen_seeds_v1_fix2.py"
)
EXPECTED_DEEPLAB_TRAINING_HELPER_SHA256 = (
    "7d0fd5192a9af40dbae0aeac692a65641745fbef2e986f1cd4e163a59ad93a31"
)
DEEPLAB_DIR = ROOT / "outputs" / "S05_C_deeplab_source_side_safettta_gate_v1"
DEEPLAB_TABLE = DEEPLAB_DIR / "deeplab_source_side_gate_training_table.csv"
DEEPLAB_AUDIT = DEEPLAB_DIR / "audit.json"

HF_CACHE = ROOT / "cache" / "huggingface"

SOURCE_VAL_N = 145
N_FOLDS = 5
IMAGE_SIZE = 352
A1_ACTION = "A1_TENT_1STEP"
TENT_LR = 1e-3
TENT_WEIGHT_DECAY = 0.0
HARM_THRESHOLD = -0.02
BENEFIT_THRESHOLD = +0.02

PRANET_SEEDS = (20260817, 20260818, 20260819)
DEEPLAB_SEEDS = (20260817, 20260818, 20260819)
SEGFORMER_SEEDS = (20260820, 20260821, 20260822)

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

EXPECTED_FOLD_GROUP_COUNTS = {
    0: 23,
    1: 34,
    2: 32,
    3: 27,
    4: 29,
}

SOURCE_FEATURE_NAMES = [
    "source_entropy_mean",
    "source_entropy_std",
    "source_entropy_q10",
    "source_entropy_q50",
    "source_entropy_q90",
    "source_prob_mean",
    "source_prob_std",
    "source_prob_q10",
    "source_prob_q50",
    "source_prob_q90",
    "source_confidence_mean",
    "source_confidence_std",
    "source_fg_fraction",
    "source_boundary_density",
    "source_uncertain_fraction_040_060",
    "source_high_entropy_fraction_050",
    "source_logit_abs_mean",
    "source_logit_abs_std",
    "tent_preupdate_entropy_loss",
]

PRANET_CHECKPOINTS = {
    20260817: (
        ROOT / "outputs" / "S01_B_pranet_source_only_seed20260817_v1"
        / "checkpoints" / "best_source_val_dice.pt",
        "ef623c0f1207bab02377ec17932db43fe2fd1843dca858cc79ef87ead29b975a",
    ),
    20260818: (
        ROOT / "outputs" / "S01_B_pranet_source_only_seed20260818_v1"
        / "checkpoints" / "best_source_val_dice.pt",
        "e56dd8ba4b5fc7785fedf3918c275427178fadb5f035b47087d48cde41ccec22",
    ),
    20260819: (
        ROOT / "outputs" / "S01_B_pranet_source_only_seed20260819_v1"
        / "checkpoints" / "best_source_val_dice.pt",
        "f8cad00bbc9bbbff04c9a29547bd2a5f3beb97d53980c269cf4b7d2c30130b72",
    ),
}

DEEPLAB_CHECKPOINTS = {
    20260817: (
        ROOT / "outputs" / "S05_B_deeplabv3_r50_source_only_seed20260817_v1"
        / "checkpoints" / "best_source_val_dice.pt",
        "d63f914e337295652627c773332d8e7382fbcd6a39d69805e1316dcc759605a2",
    ),
    20260818: (
        ROOT / "outputs" / "S05_B_deeplabv3_r50_source_only_seed20260818_v1"
        / "checkpoints" / "best_source_val_dice.pt",
        "d4ede45d5b30f62fdbc92cd9aa8fc84e0d5a52073d5ef2c3ac36baa09a56eb25",
    ),
    20260819: (
        ROOT / "outputs" / "S05_B_deeplabv3_r50_source_only_seed20260819_v1"
        / "checkpoints" / "best_source_val_dice.pt",
        "66aeae338d350db5aa878bcaab2e68bbe98eec8e626e28a1bb02b6708fa358eb",
    ),
}

SEGFORMER_CHECKPOINTS = {
    20260820: (
        R02_DIR / "seed_20260820" / "best_model_state.pt",
        "9dae2ae907b193ea36c2bccc8e376ddbad1699ba27ae0769cf40bcba13e95605",
    ),
    20260821: (
        R02_DIR / "seed_20260821" / "best_model_state.pt",
        "ad75290168eab7d116aa3de61b3eafc1e986f11fc0bbfa4a6108abbf669a258d",
    ),
    20260822: (
        R02_DIR / "seed_20260822" / "best_model_state.pt",
        "ea0b3881373e9f966475a082490fabe4b0acae81179b8596c48a06ee2661a34a",
    ),
}

EXPECTED_ROWS_PER_STATE = SOURCE_VAL_N * len(PERTURBATIONS)
EXPECTED_ROWS_PER_FAMILY = EXPECTED_ROWS_PER_STATE * 3
EXPECTED_TOTAL_ROWS = EXPECTED_ROWS_PER_FAMILY * 3
DECISION_READY = "NINE_STATE_SOURCE_UTILITY_ASSET_READY"


# ---------------------------------------------------------------------
# Generic IO
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


def validate_sha(path: Path, expected: str, label: str) -> str:
    if not path.exists():
        raise FileNotFoundError(path)
    actual = file_sha256(path)
    if actual.lower() != expected.lower():
        raise RuntimeError(
            f"{label} SHA mismatch: expected={expected} actual={actual}"
        )
    return actual


def read_csv(path: Path):
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return list(reader), reader.fieldnames or []


def write_csv(path: Path, rows, fields: Sequence[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def robust_torch_load(path: Path, map_location="cpu"):
    import torch
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


# ---------------------------------------------------------------------
# Frozen feature math
# ---------------------------------------------------------------------

def binary_entropy_from_probability(p: np.ndarray) -> np.ndarray:
    q = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    return -(q * np.log(q) + (1.0 - q) * np.log(1.0 - q))


def boundary_density(mask: np.ndarray) -> float:
    m = np.asarray(mask, dtype=np.uint8)
    if m.ndim != 2:
        raise RuntimeError(f"Expected 2D mask for boundary density: {m.shape}")
    diff_h = np.not_equal(m[1:, :], m[:-1, :]).sum()
    diff_w = np.not_equal(m[:, 1:], m[:, :-1]).sum()
    denom = max(
        (m.shape[0] - 1) * m.shape[1]
        + m.shape[0] * (m.shape[1] - 1),
        1,
    )
    return float((diff_h + diff_w) / denom)


def stable_sigmoid(z):
    z = np.asarray(z, dtype=np.float64)
    p = np.empty_like(z)
    pos = z >= 0
    p[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    p[~pos] = ez / (1.0 + ez)
    return p


def extract_source_features(logit_2d: np.ndarray, tent_loss: float) -> Dict[str, float]:
    z = np.asarray(logit_2d, dtype=np.float64)
    if z.ndim != 2 or not np.isfinite(z).all():
        raise RuntimeError(f"Invalid source logit: shape={z.shape}")

    p = stable_sigmoid(z)
    ent = binary_entropy_from_probability(p)
    conf = np.maximum(p, 1.0 - p)
    mask = p >= 0.5

    feat = {
        "source_entropy_mean": float(np.mean(ent)),
        "source_entropy_std": float(np.std(ent)),
        "source_entropy_q10": float(np.quantile(ent, 0.10)),
        "source_entropy_q50": float(np.quantile(ent, 0.50)),
        "source_entropy_q90": float(np.quantile(ent, 0.90)),
        "source_prob_mean": float(np.mean(p)),
        "source_prob_std": float(np.std(p)),
        "source_prob_q10": float(np.quantile(p, 0.10)),
        "source_prob_q50": float(np.quantile(p, 0.50)),
        "source_prob_q90": float(np.quantile(p, 0.90)),
        "source_confidence_mean": float(np.mean(conf)),
        "source_confidence_std": float(np.std(conf)),
        "source_fg_fraction": float(np.mean(mask)),
        "source_boundary_density": boundary_density(mask),
        "source_uncertain_fraction_040_060": float(
            np.mean((p >= 0.40) & (p <= 0.60))
        ),
        "source_high_entropy_fraction_050": float(np.mean(ent >= 0.50)),
        "source_logit_abs_mean": float(np.mean(np.abs(z))),
        "source_logit_abs_std": float(np.std(np.abs(z))),
        "tent_preupdate_entropy_loss": float(tent_loss),
    }
    if list(feat.keys()) != SOURCE_FEATURE_NAMES:
        raise RuntimeError("Exact Q1-S00A feature order changed.")
    if not np.isfinite(np.asarray(list(feat.values()), dtype=np.float64)).all():
        raise RuntimeError("Non-finite B0 feature.")
    return feat


def class_from_delta(delta: float) -> Tuple[int, int, int, str]:
    if delta <= HARM_THRESHOLD:
        return 1, 0, 0, "HARM"
    if delta >= BENEFIT_THRESHOLD:
        return 0, 0, 1, "BENEFIT"
    return 0, 1, 0, "NEUTRAL"


# ---------------------------------------------------------------------
# Manifest / panel provenance
# ---------------------------------------------------------------------

def load_source_val_rows():
    validate_sha(MANIFEST, EXPECTED_MANIFEST_SHA256, "S01 manifest")
    rows, fields = read_csv(MANIFEST)
    required = {"sample_id", "image_relpath", "mask_relpath", "s01_role"}
    missing = sorted(required - set(fields))
    if missing:
        raise RuntimeError(f"Manifest missing fields: {missing}")
    selected = sorted(
        [r for r in rows if r["s01_role"] == "source_val"],
        key=lambda r: r["sample_id"],
    )
    if len(selected) != SOURCE_VAL_N:
        raise RuntimeError(
            f"source_val count mismatch: expected={SOURCE_VAL_N} actual={len(selected)}"
        )
    if len({r["sample_id"] for r in selected}) != SOURCE_VAL_N:
        raise RuntimeError("Duplicate source_val sample_id.")
    return selected


def validate_upstream():
    validate_sha(PROTOCOL, EXPECTED_PROTOCOL_SHA256, "Q1-R03 protocol")
    validate_sha(R02_LOCK, EXPECTED_R02_LOCK_SHA256, "Q1-R02 lock")
    r02 = json.loads(R02_LOCK.read_text(encoding="utf-8"))
    if r02.get("decision") != EXPECTED_R02_DECISION:
        raise RuntimeError(
            f"INVALID_Q1_R03_UPSTREAM_PROVENANCE: R02 decision={r02.get('decision')}"
        )

    validate_sha(
        Q1_S00A_LOCK, EXPECTED_Q1_S00A_LOCK_SHA256, "Q1-S00A lock"
    )
    validate_sha(
        S07B_TABLE, EXPECTED_S07B_TABLE_SHA256, "S07-B utility table"
    )
    validate_sha(
        PRANET_HELPER, EXPECTED_PRANET_HELPER_SHA256, "PraNet S03 helper"
    )
    validate_sha(
        PRANET_TRAINING_HELPER,
        EXPECTED_PRANET_TRAINING_HELPER_SHA256,
        "PraNet training helper",
    )
    validate_sha(
        DEEPLAB_HELPER, EXPECTED_DEEPLAB_HELPER_SHA256, "DeepLab S05-C helper"
    )
    validate_sha(
        DEEPLAB_TRAINING_HELPER,
        EXPECTED_DEEPLAB_TRAINING_HELPER_SHA256,
        "DeepLab training helper",
    )

    q1_lock = json.loads(Q1_S00A_LOCK.read_text(encoding="utf-8"))
    feature_art = q1_lock.get("artifacts", {}).get("gradient_probe_features", {})
    feature_expected = feature_art.get("sha256")
    if not feature_expected:
        raise RuntimeError(
            "INVALID_Q1_R03_UPSTREAM_PROVENANCE: Q1-S00A lock lacks feature SHA"
        )
    validate_sha(Q1_S00A_FEATURES, feature_expected, "Q1-S00A B0 table")

    if not DEEPLAB_AUDIT.exists():
        raise FileNotFoundError(DEEPLAB_AUDIT)
    deep_audit = json.loads(DEEPLAB_AUDIT.read_text(encoding="utf-8"))
    deep_table_expected = deep_audit.get("source_training_table_sha256")
    if not deep_table_expected:
        raise RuntimeError(
            "INVALID_Q1_R03_UPSTREAM_PROVENANCE: S05-C audit lacks table SHA"
        )
    validate_sha(DEEPLAB_TABLE, deep_table_expected, "S05-C frozen table")

    panel = []
    for family, mapping in (
        ("PraNet", PRANET_CHECKPOINTS),
        ("DeepLabV3-R50", DEEPLAB_CHECKPOINTS),
        ("SegFormer-B0", SEGFORMER_CHECKPOINTS),
    ):
        for seed, (path, sha) in mapping.items():
            validate_sha(path, sha, f"{family} checkpoint seed {seed}")
            panel.append({
                "model_family": family,
                "seed": seed,
                "model_state_id": f"{family}::{seed}",
                "checkpoint": str(path),
                "checkpoint_sha256": sha,
            })

    if len(panel) != 9 or len({r["checkpoint_sha256"] for r in panel}) != 9:
        raise RuntimeError("INVALID_Q1_R03_PANEL_CHECKPOINT")

    return panel, {
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "r02_lock_sha256": EXPECTED_R02_LOCK_SHA256,
        "q1_s00a_lock_sha256": EXPECTED_Q1_S00A_LOCK_SHA256,
        "q1_s00a_feature_sha256": feature_expected,
        "s07b_table_sha256": EXPECTED_S07B_TABLE_SHA256,
        "deeplab_table_sha256": deep_table_expected,
    }


def fold_for_sample(sample_id: str) -> int:
    payload = f"S03::{sample_id}".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16) % N_FOLDS


def build_fold_audit(source_rows):
    by_fold = Counter()
    rows = []
    for r in source_rows:
        f = fold_for_sample(r["sample_id"])
        by_fold[f] += 1
        rows.append({"source_group_id": r["sample_id"], "fold": f})
    if dict(sorted(by_fold.items())) != EXPECTED_FOLD_GROUP_COUNTS:
        raise RuntimeError(
            "INVALID_Q1_R03_GROUP_FOLD: "
            f"expected={EXPECTED_FOLD_GROUP_COUNTS} actual={dict(by_fold)}"
        )
    return rows


# ---------------------------------------------------------------------
# Common row helpers
# ---------------------------------------------------------------------

COMMON_FIELDS = [
    "model_family",
    "model_state_id",
    "training_seed",
    "checkpoint_sha256",
    "sample_id",
    "source_group_id",
    "fold",
    "perturbation",
    "action",
    "action_steps",
    "source_dice",
    "action_dice",
    "delta_dice",
    "harmful",
    "neutral",
    "beneficial",
    "outcome_class",
] + SOURCE_FEATURE_NAMES


def make_common_row(
    family,
    seed,
    checkpoint_sha,
    sample_id,
    fold,
    perturbation,
    source_dice,
    action_dice,
    features,
):
    delta = float(action_dice) - float(source_dice)
    harmful, neutral, beneficial, outcome = class_from_delta(delta)
    row = {
        "model_family": family,
        "model_state_id": f"{family}::{seed}",
        "training_seed": int(seed),
        "checkpoint_sha256": checkpoint_sha,
        "sample_id": sample_id,
        "source_group_id": sample_id,
        "fold": int(fold),
        "perturbation": perturbation,
        "action": A1_ACTION,
        "action_steps": 1,
        "source_dice": float(source_dice),
        "action_dice": float(action_dice),
        "delta_dice": delta,
        "harmful": harmful,
        "neutral": neutral,
        "beneficial": beneficial,
        "outcome_class": outcome,
    }
    row.update(features)
    if set(row) != set(COMMON_FIELDS):
        raise RuntimeError("Common row schema mismatch.")
    return row


def validate_state_rows(rows, family, seed, checkpoint_sha):
    if len(rows) != EXPECTED_ROWS_PER_STATE:
        raise RuntimeError(
            f"INVALID_Q1_R03_CARDINALITY: {family} seed={seed} rows={len(rows)}"
        )
    keys = {
        (r["sample_id"], r["perturbation"])
        for r in rows
    }
    if len(keys) != EXPECTED_ROWS_PER_STATE:
        raise RuntimeError(
            f"INVALID_Q1_R03_CARDINALITY: duplicate/missing state keys {family} {seed}"
        )
    if {r["perturbation"] for r in rows} != set(PERTURBATIONS):
        raise RuntimeError("Perturbation set mismatch.")
    if {int(r["training_seed"]) for r in rows} != {int(seed)}:
        raise RuntimeError("State seed mismatch.")
    if {r["checkpoint_sha256"] for r in rows} != {checkpoint_sha}:
        raise RuntimeError("State checkpoint SHA mismatch.")
    for r in rows:
        if int(r["fold"]) != fold_for_sample(r["sample_id"]):
            raise RuntimeError("INVALID_Q1_R03_GROUP_FOLD")
        vals = [float(r[x]) for x in SOURCE_FEATURE_NAMES]
        vals += [
            float(r["source_dice"]),
            float(r["action_dice"]),
            float(r["delta_dice"]),
        ]
        if not np.isfinite(np.asarray(vals, dtype=np.float64)).all():
            raise RuntimeError("Non-finite state row.")


# ---------------------------------------------------------------------
# State cache
# ---------------------------------------------------------------------

def cache_paths(cache_dir: Path, family: str, seed: int):
    token = family.lower().replace("-", "_").replace(" ", "_")
    csv_path = cache_dir / f"{token}_seed{seed}.csv"
    lock_path = cache_dir / f"{token}_seed{seed}.lock.json"
    return csv_path, lock_path


def save_state_cache(cache_dir, family, seed, checkpoint_sha, rows):
    csv_path, lock_path = cache_paths(cache_dir, family, seed)
    write_csv(csv_path, rows, COMMON_FIELDS)
    csv_sha = file_sha256(csv_path)
    lock = {
        "script_version": VERSION,
        "family": family,
        "seed": seed,
        "checkpoint_sha256": checkpoint_sha,
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "perturbations": list(PERTURBATIONS),
        "feature_names": list(SOURCE_FEATURE_NAMES),
        "rows": len(rows),
        "csv_sha256": csv_sha,
    }
    write_json(lock_path, lock)
    return csv_path, lock_path


def load_state_cache(cache_dir, family, seed, checkpoint_sha):
    csv_path, lock_path = cache_paths(cache_dir, family, seed)
    if not csv_path.exists() or not lock_path.exists():
        return None
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    checks = (
        lock.get("script_version") == VERSION,
        lock.get("family") == family,
        int(lock.get("seed", -1)) == int(seed),
        lock.get("checkpoint_sha256") == checkpoint_sha,
        lock.get("manifest_sha256") == EXPECTED_MANIFEST_SHA256,
        list(lock.get("perturbations", [])) == list(PERTURBATIONS),
        list(lock.get("feature_names", [])) == list(SOURCE_FEATURE_NAMES),
        int(lock.get("rows", -1)) == EXPECTED_ROWS_PER_STATE,
        lock.get("csv_sha256") == file_sha256(csv_path),
    )
    if not all(checks):
        raise RuntimeError(f"Invalid resumable state cache: {lock_path}")
    rows, fields = read_csv(csv_path)
    if fields != COMMON_FIELDS:
        raise RuntimeError(f"Cached schema mismatch: {csv_path}")
    validate_state_rows(rows, family, seed, checkpoint_sha)
    return rows


# ---------------------------------------------------------------------
# PraNet frozen reuse
# ---------------------------------------------------------------------

def build_pranet_rows():
    feat_rows, feat_fields = read_csv(Q1_S00A_FEATURES)
    missing = sorted(set(SOURCE_FEATURE_NAMES) - set(feat_fields))
    if missing:
        raise RuntimeError(f"INVALID_Q1_R03_PRANET_REUSE: missing B0={missing}")

    utility_rows, utility_fields = read_csv(S07B_TABLE)
    required_u = {
        "seed", "sample_id", "source_group_id", "fold", "perturbation",
        "action", "source_dice", "action_dice", "delta_dice",
        "harmful", "neutral", "beneficial",
    }
    if not required_u.issubset(utility_fields):
        raise RuntimeError("INVALID_Q1_R03_PRANET_REUSE: S07-B schema")

    utility_a1 = {
        (int(r["seed"]), r["sample_id"], r["perturbation"]): r
        for r in utility_rows
        if r["action"] == A1_ACTION
    }
    if len(utility_a1) != EXPECTED_ROWS_PER_FAMILY:
        raise RuntimeError(
            f"INVALID_Q1_R03_PRANET_REUSE: A1 rows={len(utility_a1)}"
        )

    out = []
    crosscheck_max = 0.0
    for f in feat_rows:
        key = (int(f["seed"]), f["sample_id"], f["perturbation"])
        if key not in utility_a1:
            raise RuntimeError(f"INVALID_Q1_R03_PRANET_REUSE: missing {key}")
        u = utility_a1[key]

        for name in ("source_group_id", "fold"):
            if str(f[name]) != str(u[name]):
                raise RuntimeError(
                    f"INVALID_Q1_R03_PRANET_REUSE: {name} mismatch {key}"
                )

        delta_diff = abs(float(f["delta_dice"]) - float(u["delta_dice"]))
        crosscheck_max = max(crosscheck_max, delta_diff)
        if delta_diff > 1e-12:
            raise RuntimeError(
                f"INVALID_Q1_R03_PRANET_REUSE: delta mismatch {key} {delta_diff}"
            )

        for name in ("harmful", "neutral", "beneficial"):
            if int(f[name]) != int(u[name]):
                raise RuntimeError(
                    f"INVALID_Q1_R03_PRANET_REUSE: class mismatch {key} {name}"
                )

        seed = int(f["seed"])
        if seed not in PRANET_CHECKPOINTS:
            raise RuntimeError(f"Unexpected PraNet seed {seed}")
        sha = PRANET_CHECKPOINTS[seed][1]
        features = {name: float(f[name]) for name in SOURCE_FEATURE_NAMES}
        row = make_common_row(
            "PraNet",
            seed,
            sha,
            f["sample_id"],
            int(f["fold"]),
            f["perturbation"],
            float(u["source_dice"]),
            float(u["action_dice"]),
            features,
        )
        if abs(row["delta_dice"] - float(u["delta_dice"])) > 1e-12:
            raise RuntimeError("PraNet rebuilt delta mismatch.")
        out.append(row)

    if len(out) != EXPECTED_ROWS_PER_FAMILY:
        raise RuntimeError("INVALID_Q1_R03_PRANET_REUSE")
    for seed in PRANET_SEEDS:
        validate_state_rows(
            [r for r in out if int(r["training_seed"]) == seed],
            "PraNet", seed, PRANET_CHECKPOINTS[seed][1],
        )
    return out, {
        "rows": len(out),
        "max_abs_delta_crosscheck": crosscheck_max,
        "new_inference": False,
        "source": "Q1-S00A B0 + S07-B frozen A1 outcomes",
    }


# ---------------------------------------------------------------------
# DeepLab exact-B0 reconstruction from source-only forward
# ---------------------------------------------------------------------

def build_deeplab_rows(source_rows, device, cache_dir, resume):
    import torch

    helper = import_module(
        DEEPLAB_HELPER,
        "q1_r03_s05c_helper",
    )
    training = import_module(
        DEEPLAB_TRAINING_HELPER,
        "q1_r03_s05b_training_helper",
    )

    frozen_rows, frozen_fields = read_csv(DEEPLAB_TABLE)
    required = {
        "seed", "sample_id", "fold", "perturbation",
        "source_dice", "tent_dice", "delta_dice",
        "harmful", "beneficial", "tent_adapt_loss",
    }
    if not required.issubset(frozen_fields):
        raise RuntimeError("INVALID_Q1_R03_DEEPLAB_REPRODUCTION: frozen schema")

    frozen = {
        (int(r["seed"]), r["sample_id"], r["perturbation"]): r
        for r in frozen_rows
    }
    if len(frozen) != EXPECTED_ROWS_PER_FAMILY:
        raise RuntimeError(
            f"INVALID_Q1_R03_DEEPLAB_REPRODUCTION: frozen rows={len(frozen)}"
        )

    # FIX2: reproduce the exact frozen S05-C runtime state.
    # S05-C calls seed_everything(20260817) once before generating all
    # three seed tables; this enables deterministic algorithms and freezes
    # cuDNN benchmark/deterministic flags.
    helper.seed_everything(20260817)

    all_rows = []
    max_source_dice_diff = 0.0

    for seed in DEEPLAB_SEEDS:
        sha = DEEPLAB_CHECKPOINTS[seed][1]
        if resume:
            cached = load_state_cache(cache_dir, "DeepLabV3-R50", seed, sha)
            if cached is not None:
                print(f"[RESUME] DeepLabV3-R50 seed={seed} rows={len(cached)}")
                all_rows.extend(cached)
                continue

        model = helper.load_model(training, seed, device)

        # FIX1 — exact S05-C source-forward parity.
        #
        # The frozen S05-C table was produced only after
        # configure_singleton_safe_tent() had changed ordinary BN modules to
        # track_running_stats=False while preserving the singleton-unsafe ASPP
        # BN on source running statistics.  Its SOURCE branch then called
        # model.eval() without undoing those normalization flags.
        #
        # Therefore a plain checkpoint model.eval() is NOT guaranteed to
        # reproduce the frozen S05-C source logit.  Apply the exact frozen
        # normalization configuration first, then evaluate source-only.
        (
            parity_params,
            parity_trainable_bn_names,
            parity_unsafe_bn_names,
            parity_unsafe_bn_modules,
            parity_dropout_names,
            parity_dropout_modules,
        ) = helper.configure_singleton_safe_tent(model)
        model.eval()
        state_rows = []

        pbar = tqdm(
            total=EXPECTED_ROWS_PER_STATE,
            desc=f"Q1-R03 DeepLab exact-B0 seed {seed}",
            unit="case",
            dynamic_ncols=True,
        )

        for src in source_rows:
            image_path = DATA_ROOT / Path(src["image_relpath"])
            with Image.open(image_path) as im:
                native = im.convert("RGB")
            gt = helper.load_gt(src)

            for perturbation in PERTURBATIONS:
                key = (seed, src["sample_id"], perturbation)
                fr = frozen.get(key)
                if fr is None:
                    raise RuntimeError(
                        f"INVALID_Q1_R03_DEEPLAB_REPRODUCTION: missing frozen {key}"
                    )

                perturbed = helper.apply_perturbation(
                    native, perturbation, src["sample_id"]
                )
                x = helper.image_to_model_tensor(training, perturbed).to(
                    device, non_blocking=True
                )
                with torch.no_grad():
                    z = training.deeplab_logits(model, x).detach()
                z_np = z[0, 0].float().cpu().numpy()

                source_dice = helper.dice_from_logit(z_np, gt)
                diff = abs(source_dice - float(fr["source_dice"]))
                max_source_dice_diff = max(max_source_dice_diff, diff)
                if diff > 1e-6:
                    raise RuntimeError(
                        "INVALID_Q1_R03_DEEPLAB_REPRODUCTION: "
                        f"seed={seed} sample={src['sample_id']} "
                        f"pert={perturbation} source_dice_diff={diff}"
                    )

                features = extract_source_features(
                    z_np, float(fr["tent_adapt_loss"])
                )
                action_dice = float(fr["tent_dice"])
                row = make_common_row(
                    "DeepLabV3-R50",
                    seed,
                    sha,
                    src["sample_id"],
                    int(fr["fold"]),
                    perturbation,
                    source_dice,
                    action_dice,
                    features,
                )
                if abs(row["delta_dice"] - float(fr["delta_dice"])) > 1e-6:
                    raise RuntimeError(
                        "INVALID_Q1_R03_DEEPLAB_REPRODUCTION: delta mismatch"
                    )
                state_rows.append(row)
                pbar.update(1)

        pbar.close()
        validate_state_rows(state_rows, "DeepLabV3-R50", seed, sha)
        save_state_cache(cache_dir, "DeepLabV3-R50", seed, sha, state_rows)
        all_rows.extend(state_rows)

        del (
            model,
            parity_params,
            parity_trainable_bn_names,
            parity_unsafe_bn_names,
            parity_unsafe_bn_modules,
            parity_dropout_names,
            parity_dropout_modules,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return all_rows, {
        "rows": len(all_rows),
        "max_abs_source_dice_reproduction_error": max_source_dice_diff,
        "new_adaptation_step": False,
        "new_source_forward_for_exact_b0": True,
        "source_forward_parity_rule": (
            "configure_singleton_safe_tent_then_model_eval_exact_S05C_source_branch"
        ),
    }


# ---------------------------------------------------------------------
# SegFormer one-step normalization-affine TENT
# ---------------------------------------------------------------------

def segformer_tensor(image, mean, std, torch):
    resized = image.convert("RGB").resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        resample=Image.Resampling.BILINEAR,
    )
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    arr = (arr - mean[None, None, :]) / std[None, None, :]
    arr = np.transpose(arr, (2, 0, 1)).copy()
    return torch.from_numpy(arr).unsqueeze(0)


def load_native_gt(row):
    path = DATA_ROOT / Path(row["mask_relpath"])
    with Image.open(path) as im:
        arr = np.asarray(im.convert("L"), dtype=np.float32)
    maxv = float(arr.max())
    if maxv <= 0:
        raise RuntimeError(f"Empty source-val GT: {row['sample_id']}")
    return (arr > 0.5 * maxv).astype(np.uint8)


def dice_from_z352(z, gt, torch, F):
    t = torch.from_numpy(np.asarray(z, dtype=np.float32))[None, None]
    up = F.interpolate(
        t,
        size=tuple(gt.shape),
        mode="bilinear",
        align_corners=False,
    )[0, 0].numpy()
    pred = up >= 0.0
    target = gt.astype(bool)
    inter = np.logical_and(pred, target).sum()
    return float(
        (2.0 * inter + 1e-7)
        / (pred.sum() + target.sum() + 1e-7)
    )


def configure_segformer_tent(model, nn):
    model.requires_grad_(False)
    params = []
    names = []
    bn_modules = []
    bn_original_track = {}
    dropout_modules = []
    dropout_names = []

    for name, module in model.named_modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.AlphaDropout)):
            dropout_modules.append(module)
            dropout_names.append(name)

        if isinstance(module, nn.BatchNorm2d):
            bn_modules.append(module)
            bn_original_track[id(module)] = bool(module.track_running_stats)

        if isinstance(module, (nn.LayerNorm, nn.BatchNorm2d)):
            for suffix in ("weight", "bias"):
                p = getattr(module, suffix, None)
                if p is not None:
                    p.requires_grad_(True)
                    params.append(p)
                    names.append(f"{name}.{suffix}")

    if not params:
        raise RuntimeError("INVALID_Q1_R03_SEGFORMER_TENT: no norm affine params")
    if len(names) != len(set(names)):
        raise RuntimeError("INVALID_Q1_R03_SEGFORMER_TENT: duplicate params")

    return params, names, bn_modules, bn_original_track, dropout_modules, dropout_names


def snapshot_params(params):
    return [p.detach().clone() for p in params]


def restore_params(params, values, torch):
    with torch.no_grad():
        for p, src in zip(params, values):
            p.copy_(src)
            p.grad = None


def segformer_source_mode(
    model, bn_modules, bn_original_track, dropout_modules
):
    model.eval()
    for module in bn_modules:
        module.track_running_stats = bn_original_track[id(module)]
        module.eval()
    for module in dropout_modules:
        module.eval()


def segformer_tent_mode(model, bn_modules, dropout_modules):
    model.train()
    for module in bn_modules:
        module.track_running_stats = False
        module.train()
    for module in dropout_modules:
        module.eval()


def segformer_logits_and_z(model, x, F):
    outputs = model(pixel_values=x)
    logits = F.interpolate(
        outputs.logits,
        size=(IMAGE_SIZE, IMAGE_SIZE),
        mode="bilinear",
        align_corners=False,
    )
    if logits.ndim != 4 or logits.shape[1] != 2:
        raise RuntimeError(
            f"INVALID_Q1_R03_SEGFORMER_TENT: logits={tuple(logits.shape)}"
        )
    z = logits[:, 1] - logits[:, 0]
    return logits, z


def categorical_entropy(logits, torch):
    p = torch.softmax(logits, dim=1)
    return -(p * torch.log(p.clamp_min(1e-7))).sum(dim=1).mean()


def build_segformer_context(device):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import AutoImageProcessor, SegformerConfig, SegformerForSemanticSegmentation

    HF_CACHE.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(HF_CACHE)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(HF_CACHE / "hub")

    processor = AutoImageProcessor.from_pretrained(
        "nvidia/mit-b0",
        cache_dir=str(HF_CACHE),
        do_reduce_labels=False,
    )
    mean = np.asarray(processor.image_mean, dtype=np.float32)
    std = np.asarray(processor.image_std, dtype=np.float32)
    if mean.shape != (3,) or std.shape != (3,):
        raise RuntimeError("INVALID_Q1_R03_SEGFORMER_TENT: processor stats")

    config = SegformerConfig.from_pretrained(
        "nvidia/mit-b0",
        cache_dir=str(HF_CACHE),
    )
    config.num_labels = 2
    config.id2label = {0: "background", 1: "foreground"}
    config.label2id = {"background": 0, "foreground": 1}

    return torch, nn, F, SegformerForSemanticSegmentation, config, mean, std


def load_segformer_state(
    seed,
    device,
    SegformerForSemanticSegmentation,
    config,
    torch,
):
    path, expected_sha = SEGFORMER_CHECKPOINTS[seed]
    validate_sha(path, expected_sha, f"SegFormer checkpoint seed {seed}")
    payload = robust_torch_load(path, map_location="cpu")
    if int(payload.get("seed", -1)) != seed:
        raise RuntimeError("INVALID_Q1_R03_SEGFORMER_TENT: checkpoint seed")
    if payload.get("architecture") != "SegFormer-B0":
        raise RuntimeError("INVALID_Q1_R03_SEGFORMER_TENT: architecture")
    state = payload.get("state_dict")
    if not isinstance(state, dict) or not state:
        raise RuntimeError("INVALID_Q1_R03_SEGFORMER_TENT: state_dict")

    model = SegformerForSemanticSegmentation(config)
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError("INVALID_Q1_R03_SEGFORMER_TENT: strict load")
    return model.to(device)


def build_segformer_rows(source_rows, device, cache_dir, resume):
    (
        torch, nn, F, SegformerForSemanticSegmentation,
        config, mean, std,
    ) = build_segformer_context(device)

    all_rows = []
    parameter_names_reference = None
    dropout_names_reference = None
    bn_count_reference = None
    per_seed_audit = {}

    for seed in SEGFORMER_SEEDS:
        sha = SEGFORMER_CHECKPOINTS[seed][1]
        if resume:
            cached = load_state_cache(cache_dir, "SegFormer-B0", seed, sha)
            if cached is not None:
                print(f"[RESUME] SegFormer-B0 seed={seed} rows={len(cached)}")
                all_rows.extend(cached)
                continue

        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        model = load_segformer_state(
            seed, device, SegformerForSemanticSegmentation, config, torch
        )
        (
            params, param_names, bn_modules, bn_original_track,
            dropout_modules, dropout_names,
        ) = configure_segformer_tent(model, nn)
        source_values = snapshot_params(params)

        if parameter_names_reference is None:
            parameter_names_reference = list(param_names)
            dropout_names_reference = list(dropout_names)
            bn_count_reference = len(bn_modules)
        else:
            if list(param_names) != parameter_names_reference:
                raise RuntimeError(
                    "INVALID_Q1_R03_SEGFORMER_TENT: parameter schema differs by seed"
                )
            if list(dropout_names) != dropout_names_reference:
                raise RuntimeError(
                    "INVALID_Q1_R03_SEGFORMER_TENT: dropout schema differs by seed"
                )
            if len(bn_modules) != bn_count_reference:
                raise RuntimeError(
                    "INVALID_Q1_R03_SEGFORMER_TENT: BN schema differs by seed"
                )

        state_rows = []
        pbar = tqdm(
            total=EXPECTED_ROWS_PER_STATE,
            desc=f"Q1-R03 SegFormer A1 seed {seed}",
            unit="case",
            dynamic_ncols=True,
        )

        for src in source_rows:
            image_path = DATA_ROOT / Path(src["image_relpath"])
            with Image.open(image_path) as im:
                native = im.convert("RGB")
            gt = load_native_gt(src)

            for perturbation in PERTURBATIONS:
                # Use frozen S03 perturbation implementation exactly.
                # Imported lazily here to avoid target/helper coupling.
                pranet_helper = import_module(
                    PRANET_HELPER,
                    "q1_r03_pranet_perturb_helper",
                ) if "q1_r03_pranet_perturb_helper" not in sys.modules else sys.modules[
                    "q1_r03_pranet_perturb_helper"
                ]
                perturbed = pranet_helper.apply_perturbation(
                    native, perturbation, src["sample_id"]
                )
                x = segformer_tensor(perturbed, mean, std, torch).to(
                    device, non_blocking=True
                )

                # Untouched source prediction.
                restore_params(params, source_values, torch)
                segformer_source_mode(
                    model, bn_modules, bn_original_track, dropout_modules
                )
                with torch.no_grad():
                    _, z_source_t = segformer_logits_and_z(model, x, F)
                z_source = z_source_t[0].detach().float().cpu().numpy()

                # Episodic A1 TENT.
                restore_params(params, source_values, torch)
                segformer_tent_mode(model, bn_modules, dropout_modules)
                optimizer = torch.optim.Adam(
                    params, lr=TENT_LR, weight_decay=TENT_WEIGHT_DECAY
                )
                optimizer.zero_grad(set_to_none=True)
                logits_pre, _ = segformer_logits_and_z(model, x, F)
                loss = categorical_entropy(logits_pre, torch)
                if not torch.isfinite(loss):
                    raise RuntimeError(
                        "INVALID_Q1_R03_SEGFORMER_TENT: non-finite entropy"
                    )
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                with torch.no_grad():
                    _, z_action_t = segformer_logits_and_z(model, x, F)
                z_action = z_action_t[0].detach().float().cpu().numpy()
                tent_loss = float(loss.detach().cpu())

                source_dice = dice_from_z352(z_source, gt, torch, F)
                action_dice = dice_from_z352(z_action, gt, torch, F)
                features = extract_source_features(z_source, tent_loss)

                row = make_common_row(
                    "SegFormer-B0",
                    seed,
                    sha,
                    src["sample_id"],
                    fold_for_sample(src["sample_id"]),
                    perturbation,
                    source_dice,
                    action_dice,
                    features,
                )
                state_rows.append(row)
                pbar.update(1)
                pbar.set_postfix(dDice=f"{row['delta_dice']:+.3f}")

                del optimizer, logits_pre, loss, z_source_t, z_action_t

        pbar.close()
        restore_params(params, source_values, torch)
        validate_state_rows(state_rows, "SegFormer-B0", seed, sha)
        save_state_cache(cache_dir, "SegFormer-B0", seed, sha, state_rows)
        all_rows.extend(state_rows)

        per_seed_audit[str(seed)] = {
            "trainable_norm_affine_tensor_count": len(params),
            "trainable_norm_affine_names": list(param_names),
            "batchnorm_module_count": len(bn_modules),
            "dropout_module_count": len(dropout_modules),
            "dropout_names": list(dropout_names),
            "rows": len(state_rows),
        }

        del model, params, source_values, bn_modules, dropout_modules
        torch.cuda.empty_cache()

    return all_rows, {
        "parameter_names_reference": parameter_names_reference,
        "dropout_names_reference": dropout_names_reference,
        "batchnorm_module_count": bn_count_reference,
        "per_seed": per_seed_audit,
        "tent_lr": TENT_LR,
        "tent_weight_decay": TENT_WEIGHT_DECAY,
        "tent_steps": 1,
        "adapted_module_types": ["LayerNorm", "BatchNorm2d"],
        "dropout_rule": "eval_during_tent",
        "batchnorm_rule": "current_sample_stats_during_tent",
    }


# ---------------------------------------------------------------------
# Final audit
# ---------------------------------------------------------------------

def distribution_summary(rows):
    out = []
    groups = [("ALL", rows)]
    for family in ("PraNet", "DeepLabV3-R50", "SegFormer-B0"):
        groups.append((family, [r for r in rows if r["model_family"] == family]))

    for name, rr in groups:
        d = np.asarray([float(r["delta_dice"]) for r in rr], dtype=np.float64)
        src = np.asarray([float(r["source_dice"]) for r in rr], dtype=np.float64)
        act = np.asarray([float(r["action_dice"]) for r in rr], dtype=np.float64)
        out.append({
            "group": name,
            "rows": len(rr),
            "source_dice_mean": float(src.mean()),
            "action_dice_mean": float(act.mean()),
            "delta_dice_mean": float(d.mean()),
            "delta_dice_std": float(d.std()),
            "delta_dice_q10": float(np.quantile(d, 0.10)),
            "delta_dice_q50": float(np.quantile(d, 0.50)),
            "delta_dice_q90": float(np.quantile(d, 0.90)),
            "harm_fraction": float(np.mean(d <= HARM_THRESHOLD)),
            "neutral_fraction": float(
                np.mean((d > HARM_THRESHOLD) & (d < BENEFIT_THRESHOLD))
            ),
            "benefit_fraction": float(np.mean(d >= BENEFIT_THRESHOLD)),
        })
    return out


def validate_global_rows(rows):
    if len(rows) != EXPECTED_TOTAL_ROWS:
        raise RuntimeError(
            f"INVALID_Q1_R03_CARDINALITY: total={len(rows)} "
            f"expected={EXPECTED_TOTAL_ROWS}"
        )

    states = Counter(r["model_state_id"] for r in rows)
    if len(states) != 9 or any(v != EXPECTED_ROWS_PER_STATE for v in states.values()):
        raise RuntimeError(f"INVALID_Q1_R03_CARDINALITY: states={dict(states)}")

    families = Counter(r["model_family"] for r in rows)
    expected_family = {
        "PraNet": EXPECTED_ROWS_PER_FAMILY,
        "DeepLabV3-R50": EXPECTED_ROWS_PER_FAMILY,
        "SegFormer-B0": EXPECTED_ROWS_PER_FAMILY,
    }
    if dict(families) != expected_family:
        raise RuntimeError(
            f"INVALID_Q1_R03_CARDINALITY: families={dict(families)}"
        )

    global_keys = {
        (
            r["model_state_id"],
            r["sample_id"],
            r["perturbation"],
        )
        for r in rows
    }
    if len(global_keys) != EXPECTED_TOTAL_ROWS:
        raise RuntimeError("INVALID_Q1_R03_CARDINALITY: duplicate global key")

    groups = defaultdict(set)
    for r in rows:
        groups[r["sample_id"]].add(int(r["fold"]))
    if len(groups) != SOURCE_VAL_N:
        raise RuntimeError("INVALID_Q1_R03_GROUP_FOLD: group count")
    if any(len(v) != 1 for v in groups.values()):
        raise RuntimeError("INVALID_Q1_R03_GROUP_FOLD: fold leakage")

    fold_counts = Counter(next(iter(v)) for v in groups.values())
    if dict(sorted(fold_counts.items())) != EXPECTED_FOLD_GROUP_COUNTS:
        raise RuntimeError("INVALID_Q1_R03_GROUP_FOLD: fold counts")


# ---------------------------------------------------------------------
# Preflight / run
# ---------------------------------------------------------------------


def run_deeplab_parity_only():
    """
    Technical diagnostic only.

    Reproduce the first frozen source-val sample under all 10 perturbations
    for DeepLab seed 20260817. No adaptation optimizer step is executed.
    No target rows are opened.
    """
    panel, provenance = validate_upstream()
    source_rows = load_source_val_rows()

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for DeepLab parity diagnostic.")

    device = torch.device("cuda")
    helper = import_module(
        DEEPLAB_HELPER,
        "q1_r03_s05c_helper_parity_only",
    )
    training = import_module(
        DEEPLAB_TRAINING_HELPER,
        "q1_r03_s05b_training_helper_parity_only",
    )

    # Exact frozen S05-C runtime.
    helper.seed_everything(20260817)

    frozen_rows, frozen_fields = read_csv(DEEPLAB_TABLE)
    frozen = {
        (int(r["seed"]), r["sample_id"], r["perturbation"]): r
        for r in frozen_rows
    }

    seed = 20260817
    src = source_rows[0]
    model = helper.load_model(training, seed, device)
    (
        params,
        trainable_bn_names,
        unsafe_bn_names,
        unsafe_bn_modules,
        dropout_names,
        dropout_modules,
    ) = helper.configure_singleton_safe_tent(model)

    # Exact frozen source branch state:
    # configure_singleton_safe_tent() first, then model.eval().
    model.eval()

    image_path = DATA_ROOT / Path(src["image_relpath"])
    with Image.open(image_path) as im:
        native = im.convert("RGB")
    gt = helper.load_gt(src)

    rows = []
    max_diff = 0.0

    print("===== Q1-R03 FIX2 DEEPLAB PARITY ONLY =====")
    print(f"seed={seed}")
    print(f"sample_id={src['sample_id']}")
    print(f"CUBLAS_WORKSPACE_CONFIG={os.environ.get('CUBLAS_WORKSPACE_CONFIG')}")
    print(f"deterministic_algorithms={torch.are_deterministic_algorithms_enabled()}")
    print(f"cudnn_benchmark={torch.backends.cudnn.benchmark}")
    print(f"cudnn_deterministic={torch.backends.cudnn.deterministic}")
    print("optimizer_step=NO")
    print("target_rows_opened=NO")
    print()

    for perturbation in tqdm(
        PERTURBATIONS,
        desc="DeepLab parity diagnostic",
        unit="pert",
        dynamic_ncols=True,
    ):
        key = (seed, src["sample_id"], perturbation)
        fr = frozen.get(key)
        if fr is None:
            raise RuntimeError(f"Missing frozen S05-C row: {key}")

        perturbed = helper.apply_perturbation(
            native,
            perturbation,
            src["sample_id"],
        )
        x = helper.image_to_model_tensor(training, perturbed).to(
            device,
            non_blocking=True,
        )

        # Restore source affine values is unnecessary before the first source
        # forward because no optimizer step occurs in this diagnostic.
        # model.eval() exactly matches source_and_tent_logits source branch.
        with torch.no_grad():
            z = training.deeplab_logits(model, x).detach()
        z_np = z[0, 0].float().cpu().numpy()

        reproduced = helper.dice_from_logit(z_np, gt)
        frozen_dice = float(fr["source_dice"])
        diff = abs(reproduced - frozen_dice)
        max_diff = max(max_diff, diff)

        rows.append({
            "perturbation": perturbation,
            "frozen_source_dice": frozen_dice,
            "reproduced_source_dice": reproduced,
            "abs_diff": diff,
        })

        print(
            f"{perturbation}: frozen={frozen_dice:.12f} "
            f"reproduced={reproduced:.12f} diff={diff:.3e}"
        )

    print()
    print(f"max_abs_source_dice_error={max_diff:.12g}")
    if max_diff > 1e-6:
        raise RuntimeError(
            "INVALID_Q1_R03_DEEPLAB_REPRODUCTION: parity-only max error "
            f"{max_diff} exceeds frozen tolerance 1e-6"
        )

    print("DEEPLAB_PARITY_PASS")
    print("optimizer_step=NO")
    print("target_rows_opened=NO")

    del (
        model,
        params,
        unsafe_bn_modules,
        dropout_modules,
    )
    torch.cuda.empty_cache()
    return 0



def run_preflight():
    panel, provenance = validate_upstream()
    source_rows = load_source_val_rows()
    fold_rows = build_fold_audit(source_rows)

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for Q1-R03 formal SegFormer utility pass.")

    (
        torch, nn, F, SegformerForSemanticSegmentation,
        config, mean, std,
    ) = build_segformer_context(torch.device("cuda"))
    model = load_segformer_state(
        SEGFORMER_SEEDS[0],
        torch.device("cuda"),
        SegformerForSemanticSegmentation,
        config,
        torch,
    )
    (
        params, names, bn_modules, bn_original_track,
        dropout_modules, dropout_names,
    ) = configure_segformer_tent(model, nn)

    dummy = torch.zeros((1, 3, IMAGE_SIZE, IMAGE_SIZE), device="cuda")
    segformer_source_mode(model, bn_modules, bn_original_track, dropout_modules)
    with torch.no_grad():
        logits, z = segformer_logits_and_z(model, dummy, F)
    if tuple(z.shape) != (1, IMAGE_SIZE, IMAGE_SIZE):
        raise RuntimeError("SegFormer preflight z shape failed.")

    print("===== Q1-R03 PREFLIGHT =====")
    print(f"protocol_sha256={EXPECTED_PROTOCOL_SHA256}")
    print(f"r02_lock_sha256={EXPECTED_R02_LOCK_SHA256}")
    print(f"panel_states={len(panel)}")
    print(f"source_val={len(source_rows)}")
    print(f"expected_total_rows={EXPECTED_TOTAL_ROWS}")
    print(f"segformer_norm_affine_tensors={len(params)}")
    print(f"segformer_batchnorm_modules={len(bn_modules)}")
    print(f"segformer_dropout_modules={len(dropout_modules)}")
    print(f"dummy_logits_shape={tuple(logits.shape)}")
    print(f"dummy_binary_z_shape={tuple(z.shape)}")
    print("target_rows_opened=NO")
    print("optimizer_step=NO")
    print("PREFLIGHT_PASS")

    del model, dummy, logits, z, params, bn_modules, dropout_modules
    torch.cuda.empty_cache()


def run(args):
    panel, provenance = validate_upstream()
    source_rows = load_source_val_rows()
    fold_rows = build_fold_audit(source_rows)

    if args.output_dir.exists():
        raise FileExistsError(f"Final Q1-R03 output exists: {args.output_dir}")

    build_dir = Path(str(args.output_dir) + "__building")
    if build_dir.exists() and not args.resume:
        raise FileExistsError(
            f"Partial Q1-R03 output exists: {build_dir}. Use --resume only after "
            "a technical interruption."
        )
    build_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = build_dir / "state_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for Q1-R03.")
    device = torch.device("cuda")

    protocol_copy = build_dir / "preregistered_protocol_copy.md"
    if not protocol_copy.exists():
        shutil.copy2(PROTOCOL, protocol_copy)
    validate_sha(protocol_copy, EXPECTED_PROTOCOL_SHA256, "protocol copy")

    panel_path = build_dir / "panel_checkpoint_manifest.csv"
    write_csv(
        panel_path,
        panel,
        [
            "model_family",
            "seed",
            "model_state_id",
            "checkpoint",
            "checkpoint_sha256",
        ],
    )

    fold_path = build_dir / "source_group_fold_audit.csv"
    write_csv(fold_path, fold_rows, ["source_group_id", "fold"])

    print("===== Q1-R03 NINE-STATE SOURCE UTILITY ASSET =====")
    print(f"Device={device}")
    print(f"Source groups={SOURCE_VAL_N}")
    print(f"Perturbations={len(PERTURBATIONS)}")
    print("Model states=9")
    print(f"Expected rows={EXPECTED_TOTAL_ROWS}")
    print("Target data=NO")
    print("Utility predictor fitted=NO")
    print()

    pranet_rows, pranet_audit = build_pranet_rows()
    write_json(build_dir / "pranet_reuse_audit.json", pranet_audit)
    print(f"[PASS] PraNet frozen reuse rows={len(pranet_rows)}")

    deep_rows, deep_audit = build_deeplab_rows(
        source_rows, device, cache_dir, args.resume
    )
    write_json(build_dir / "deeplab_reuse_and_b0_audit.json", deep_audit)
    print(
        "[PASS] DeepLab exact-B0 rows="
        f"{len(deep_rows)} max_source_dice_error="
        f"{deep_audit['max_abs_source_dice_reproduction_error']:.3e}"
    )

    seg_rows, seg_audit = build_segformer_rows(
        source_rows, device, cache_dir, args.resume
    )
    write_json(build_dir / "segformer_tent_parameter_audit.json", seg_audit)
    print(f"[PASS] SegFormer A1 rows={len(seg_rows)}")

    all_rows = pranet_rows + deep_rows + seg_rows
    validate_global_rows(all_rows)
    all_rows.sort(
        key=lambda r: (
            r["model_family"],
            int(r["training_seed"]),
            r["sample_id"],
            r["perturbation"],
        )
    )

    table_path = build_dir / "nine_state_source_utility_table.csv"
    write_csv(table_path, all_rows, COMMON_FIELDS)
    table_sha = file_sha256(table_path)

    dist = distribution_summary(all_rows)
    dist_path = build_dir / "architecture_distribution_summary.csv"
    write_csv(dist_path, dist, list(dist[0].keys()))

    provenance_payload = {
        **provenance,
        "script_version": VERSION,
        "build": BUILD,
        "target_data_used": False,
        "utility_predictor_fitted": False,
        "threshold_selected": False,
        "hyperparameter_sweep": False,
        "source_val_groups": SOURCE_VAL_N,
        "perturbations": list(PERTURBATIONS),
        "action": A1_ACTION,
        "model_states": 9,
        "rows": len(all_rows),
        "table_sha256": table_sha,
    }
    provenance_path = build_dir / "provenance_audit.json"
    write_json(provenance_path, provenance_payload)

    decision_path = build_dir / "decision.txt"
    decision_path.write_text(DECISION_READY + "\n", encoding="utf-8")

    run_log = f"""===== Q1-R03 NINE-STATE HETEROGENEOUS SOURCE UTILITY ASSET =====
Script version: {VERSION}
Build: {BUILD}

Source:
  manifest_sha256={EXPECTED_MANIFEST_SHA256}
  source_val_groups={SOURCE_VAL_N}
  perturbations={len(PERTURBATIONS)}

Panel:
  PraNet states=3 rows={len(pranet_rows)}
  DeepLabV3-R50 states=3 rows={len(deep_rows)}
  SegFormer-B0 states=3 rows={len(seg_rows)}
  total states=9
  total rows={len(all_rows)}

Information boundary:
  seen_sanity opened=NO
  unseen_locked opened=NO
  PolypGen target opened=NO
  target labels opened=NO
  utility predictor fitted=NO
  threshold selected=NO
  hyperparameter sweep=NO

DeepLab exact-B0 reproduction:
  max_abs_source_dice_error={deep_audit['max_abs_source_dice_reproduction_error']:.12g}

Combined table:
  sha256={table_sha}

Decision:
  {DECISION_READY}

If ready:
  next = Q1-R04 Model-Relative Prospective Utility Feasibility with LOAO
"""
    run_log_path = build_dir / "run_log.txt"
    run_log_path.write_text(run_log, encoding="utf-8")

    artifact_paths = {
        "protocol_copy": protocol_copy,
        "panel_checkpoint_manifest": panel_path,
        "source_group_fold_audit": fold_path,
        "pranet_reuse_audit": build_dir / "pranet_reuse_audit.json",
        "deeplab_reuse_and_b0_audit": build_dir / "deeplab_reuse_and_b0_audit.json",
        "segformer_tent_parameter_audit": build_dir / "segformer_tent_parameter_audit.json",
        "nine_state_source_utility_table": table_path,
        "architecture_distribution_summary": dist_path,
        "provenance_audit": provenance_path,
        "decision": decision_path,
        "run_log": run_log_path,
    }

    lock = {
        "script_version": VERSION,
        "build": BUILD,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "r02_lock_sha256": EXPECTED_R02_LOCK_SHA256,
        "model_states": 9,
        "rows": EXPECTED_TOTAL_ROWS,
        "feature_names": SOURCE_FEATURE_NAMES,
        "target_data_used": False,
        "utility_predictor_fitted": False,
        "artifacts": {
            name: {
                "relative_path": str(path.relative_to(build_dir)),
                "sha256": file_sha256(path),
            }
            for name, path in artifact_paths.items()
        },
        "decision": DECISION_READY,
    }
    lock_path = build_dir / "Q1_R03_NINE_STATE_SOURCE_UTILITY_LOCK.json"
    write_json(lock_path, lock)
    lock_sha = file_sha256(lock_path)

    # Final immutability check.
    for name, meta in lock["artifacts"].items():
        path = build_dir / meta["relative_path"]
        if file_sha256(path) != meta["sha256"]:
            raise RuntimeError(f"Artifact changed before commit: {name}")

    build_dir.rename(args.output_dir)

    print()
    print((args.output_dir / "run_log.txt").read_text(encoding="utf-8"))
    print(
        "Q1-R03 LOCK:",
        args.output_dir / "Q1_R03_NINE_STATE_SOURCE_UTILITY_LOCK.json",
    )
    print("Q1-R03 LOCK SHA256:", lock_sha)


# ---------------------------------------------------------------------
# Self test
# ---------------------------------------------------------------------

def self_test():
    assert len(SOURCE_FEATURE_NAMES) == 19
    assert len(PERTURBATIONS) == 10
    assert EXPECTED_ROWS_PER_STATE == 1450
    assert EXPECTED_ROWS_PER_FAMILY == 4350
    assert EXPECTED_TOTAL_ROWS == 13050
    assert sum(EXPECTED_FOLD_GROUP_COUNTS.values()) == 145
    assert A1_ACTION == "A1_TENT_1STEP"
    assert TENT_LR == 1e-3
    assert TENT_WEIGHT_DECAY == 0.0

    z = np.zeros((8, 8), dtype=np.float64)
    feat = extract_source_features(z, math.log(2.0))
    assert list(feat) == SOURCE_FEATURE_NAMES
    assert abs(feat["source_prob_mean"] - 0.5) < 1e-12
    assert abs(feat["source_confidence_mean"] - 0.5) < 1e-12
    assert abs(feat["source_boundary_density"]) < 1e-12
    assert abs(feat["source_uncertain_fraction_040_060"] - 1.0) < 1e-12
    assert abs(feat["source_high_entropy_fraction_050"] - 1.0) < 1e-12

    toy = np.zeros((4, 4), dtype=np.uint8)
    toy[:, 2:] = 1
    bd = boundary_density(toy)
    assert 0.0 < bd < 1.0

    h, n, b, c = class_from_delta(-0.02)
    assert (h, n, b, c) == (1, 0, 0, "HARM")
    h, n, b, c = class_from_delta(0.02)
    assert (h, n, b, c) == (0, 0, 1, "BENEFIT")
    h, n, b, c = class_from_delta(0.0)
    assert (h, n, b, c) == (0, 1, 0, "NEUTRAL")

    f = fold_for_sample("toy")
    assert 0 <= f < 5
    assert f == fold_for_sample("toy")

    print("FEATURE_SCHEMA_TEST_PASS")
    print("FEATURE_MATH_TEST_PASS")
    print("BOUNDARY_DENSITY_TEST_PASS")
    print("UTILITY_CLASS_TEST_PASS")
    print("GROUP_FOLD_TEST_PASS")
    print("CARDINALITY_TEST_PASS")
    print("SELF_TEST_PASS")


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Q1-R03: complete a source-only 9-state heterogeneous "
            "A1_TENT_1STEP utility asset. No utility predictor is fitted."
        )
    )
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    p.add_argument(
        "--deeplab-parity-only",
        action="store_true",
        help=(
            "Technical diagnostic: reproduce frozen S05-C source Dice for "
            "DeepLab seed 20260817 on the first source-val image across all "
            "10 perturbations. No adaptation optimizer step and no target access."
        ),
    )
    p.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "Verify all frozen provenance/checkpoints and one SegFormer dummy "
            "forward/TENT parameter discovery; no source image inference and no "
            "optimizer step."
        ),
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume only verified Q1-R03 per-state caches after technical interruption."
        ),
    )
    p.add_argument("--self-test", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.deeplab_parity_only:
        return run_deeplab_parity_only()
    if args.preflight_only:
        run_preflight()
        return 0
    try:
        run(args)
    except Exception:
        traceback.print_exc()
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
