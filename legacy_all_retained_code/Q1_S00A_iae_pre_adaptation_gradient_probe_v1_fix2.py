#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Q1-S00A: Pre-Adaptation Individual Adaptation Effect (IAE) Feasibility Audit.

Scientific question
-------------------
Can the future effect of frozen A1_TENT_1STEP be forecast BEFORE committing
the TENT parameter update?

Strict information boundary
---------------------------
For each source-side case:

    Source checkpoint
        -> Source forward
        -> TENT pre-update entropy objective
        -> backward()
        -> READ GRADIENTS ONLY
        -> NO optimizer.step()
        -> NO adapted forward
        -> forecast frozen future DeltaDice / HARM / BENEFIT

Outcome labels come ONLY from the already-frozen S07-B source-side
counterfactual utility table. The script never needs source masks during
feature extraction and never opens any target-domain data.

Frozen comparison
-----------------
B0 = PRE-SOURCE:
     source prediction statistics + pre-update TENT loss scalar
B1 = GRADIENT-PROBE ONLY:
     no-update BN-affine gradient summaries
B2 = PRE-SOURCE + GRADIENT-PROBE

Predictors:
- Ridge(alpha=1.0) for continuous DeltaDice
- StandardScaler + LogisticRegression(C=1.0, balanced, liblinear) for
  HARM and BENEFIT
- 5 source-image-grouped OOF folds, inherited exactly from S07-B

Frozen GO criteria
------------------
All must pass for B2:
- DeltaDice Spearman >= 0.30
- HARM AUROC >= 0.75
- BENEFIT AUROC >= 0.65
- B2 - B0 BENEFIT AUROC >= +0.05

This script is intentionally a simple feasibility audit. It does NOT:
- run optimizer.step() during probe extraction;
- produce an adapted prediction;
- use SCRR;
- tune thresholds;
- sweep model capacity;
- use target-domain data;
- fit a deployment policy.

Expected project root:
    F:\\MEDSEG_SAFETTA

Typical usage:
    python .\\Q1_S00A_iae_pre_adaptation_gradient_probe_v1.py --self-test
    python .\\Q1_S00A_iae_pre_adaptation_gradient_probe_v1.py
    python .\\Q1_S00A_iae_pre_adaptation_gradient_probe_v1.py --resume
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import shutil
import sys
import traceback
from collections import Counter, defaultdict
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from tqdm import tqdm


VERSION = "2026-08-19-Q1-S00A-v1-fix2"
BUILD = "Q1_S00A_PRE_ADAPTATION_IAE_GRADIENT_PROBE_GROUPED_OOF_FIX2"

ROOT = Path(r"F:\MEDSEG_SAFETTA")

PROTOCOL = (
    ROOT / "docs"
    / "Q1_S00A_iae_pre_adaptation_gradient_probe_preregistered_protocol_v1.md"
)
EXPECTED_PROTOCOL_SHA256 = (
    "002896554a868792c1294d99386f7511886e2ebbe9a8b73a223ccdf2ad69b900"
)

S07B_DIR = (
    ROOT / "outputs"
    / "S07_B_pranet_source_side_counterfactual_utility_dataset_v1"
)
S07B_LOCK = S07B_DIR / "S07_B_COUNTERFACTUAL_UTILITY_DATASET_LOCK.json"
EXPECTED_S07B_LOCK_SHA256 = (
    "31c9aa7743dfebf124928e0d2204228499c80c4e70b00864fa206eac6696886c"
)

UTILITY_TABLE = S07B_DIR / "source_side_counterfactual_utility_table.csv"
EXPECTED_UTILITY_TABLE_SHA256 = (
    "f8d36a10a41db8255847526312b8b2794d3d18c81a8d7b152a210b3c07bd0678"
)

S03B_HELPER = ROOT / "code" / "S03_B_build_source_side_safettta_gate_v1.py"
EXPECTED_S03B_HELPER_SHA256 = (
    "07ac65247216dea376e8996cbfbc98c54c62157ecdd1e9b7d9b539b5a74a2b5c"
)

TRAINING_HELPER = (
    ROOT / "code" / "S01_B_train_pranet_source_only_frozen_seeds_v1.py"
)
EXPECTED_TRAINING_HELPER_SHA256 = (
    "2b2c9c1b75ff60bd087b403e43dc77468b31cbd54fb607481cc80ab47182b67a"
)

OUTPUT_DIR = (
    ROOT / "outputs"
    / "Q1_S00A_iae_pre_adaptation_gradient_probe_v1"
)

SEEDS = (20260817, 20260818, 20260819)
SOURCE_GROUPS = 145
N_FOLDS = 5
A1_ACTION = "A1_TENT_1STEP"
EXPECTED_A1_ROWS = 4350

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

EXPECTED_CLASS_COUNTS = {
    "HARM": 1469,
    "NEUTRAL": 2280,
    "BENEFIT": 601,
}

EXPECTED_CHECKPOINT_SHA256 = {
    20260817: "ef623c0f1207bab02377ec17932db43fe2fd1843dca858cc79ef87ead29b975a",
    20260818: "e56dd8ba4b5fc7785fedf3918c275427178fadb5f035b47087d48cde41ccec22",
    20260819: "f8cad00bbc9bbbff04c9a29547bd2a5f3beb97d53980c269cf4b7d2c30130b72",
}

HARM_THRESHOLD = -0.02
BENEFIT_THRESHOLD = +0.02

RIDGE_ALPHA = 1.0
LOGISTIC_C = 1.0
LOGISTIC_MAX_ITER = 5000
OOF_RANDOM_STATE = 20260819

GO_MIN_SPEARMAN = 0.30
GO_MIN_HARM_AUROC = 0.75
GO_MIN_BENEFIT_AUROC = 0.65
GO_MIN_BENEFIT_INCREMENT = 0.05

PASS_DECISION = "GO_IAE_TTA_Q1"
STOP_NO_INCREMENT = "STOP_GRADIENT_PROBE_NO_INCREMENT"
STOP_HARM_ONLY = "STOP_IAE_HARM_ONLY"
STOP_SIMPLE = "STOP_SIMPLE_FEASIBILITY_NOT_ESTABLISHED"

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

GLOBAL_GRADIENT_FEATURE_NAMES = [
    "grad_global_l1",
    "grad_global_l2",
    "grad_global_rms",
    "grad_global_abs_max",
    "grad_global_mean",
    "grad_global_signed_mean",
    "grad_global_std",
    "grad_global_variance",
    "grad_param_group_count",
    "grad_nonzero_fraction",
    "grad_energy_early",
    "grad_energy_middle",
    "grad_energy_late",
    "grad_energy_late_over_early",
    "grad_energy_middle_over_early",
    "grad_weight_energy",
    "grad_bias_energy",
    "grad_weight_over_bias_energy",
    "grad_group_energy_entropy",
    "grad_top1_energy_fraction",
    "grad_top5_energy_fraction",
]

PER_PARAM_GRAD_STATS = (
    "l1",
    "l2",
    "mean",
    "signed_mean",
    "std",
    "abs_max",
)

FORBIDDEN_FEATURE_TOKENS = (
    "action_dice",
    "tent_dice",
    "delta_dice",
    "harmful",
    "beneficial",
    "neutral",
    "a1_prob",
    "a1_logit",
    "adapted",
    "source_to_a1",
    "post_update",
)


# ---------------------------------------------------------------------
# Generic utilities
# ---------------------------------------------------------------------

def file_sha256(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def validate_sha(path: Path, expected: str, label: str) -> str:
    if not path.exists():
        raise FileNotFoundError(path)
    actual = file_sha256(path)
    if actual.lower() != expected.lower():
        raise RuntimeError(
            f"{label} SHA256 mismatch. expected={expected} actual={actual}"
        )
    return actual


def read_csv(path: Path):
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return list(reader), reader.fieldnames or []


def write_csv(path: Path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj):
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def safe_float(v) -> float:
    x = float(v)
    if not math.isfinite(x):
        raise RuntimeError(f"Non-finite numeric value: {v}")
    return x


def stable_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2:
        return 0.0
    if np.std(y_true) < 1e-15 or np.std(y_pred) < 1e-15:
        return 0.0
    value = spearmanr(y_true, y_pred).statistic
    if value is None or not np.isfinite(value):
        return 0.0
    return float(value)


def validate_feature_names(names: Sequence[str]):
    if len(names) != len(set(names)):
        duplicates = [
            k for k, n in Counter(names).items() if n > 1
        ]
        raise RuntimeError(f"Duplicate feature names: {duplicates}")
    for name in names:
        lower = name.lower()
        for token in FORBIDDEN_FEATURE_TOKENS:
            if token in lower:
                raise RuntimeError(
                    f"Forbidden outcome/post-adaptation token in feature: {name}"
                )


# ---------------------------------------------------------------------
# Frozen provenance / helper loading
# ---------------------------------------------------------------------

def import_module_from_path(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def validate_provenance():
    protocol_sha = validate_sha(
        PROTOCOL,
        EXPECTED_PROTOCOL_SHA256,
        "Q1-S00A protocol",
    )
    s07b_lock_sha = validate_sha(
        S07B_LOCK,
        EXPECTED_S07B_LOCK_SHA256,
        "S07-B dataset lock",
    )
    utility_sha = validate_sha(
        UTILITY_TABLE,
        EXPECTED_UTILITY_TABLE_SHA256,
        "S07-B utility table",
    )
    s03b_sha = validate_sha(
        S03B_HELPER,
        EXPECTED_S03B_HELPER_SHA256,
        "Frozen S03-B helper",
    )
    training_sha = validate_sha(
        TRAINING_HELPER,
        EXPECTED_TRAINING_HELPER_SHA256,
        "Frozen PraNet training helper",
    )

    lock = json.loads(S07B_LOCK.read_text(encoding="utf-8"))
    if bool(lock.get("target_data_used", True)):
        raise RuntimeError("S07-B lock reports target data use.")
    if bool(lock.get("utility_estimator_fitted", True)):
        raise RuntimeError("S07-B lock reports an already fitted utility estimator.")

    expected_decision = (
        "S07_B_SOURCE_SIDE_COUNTERFACTUAL_UTILITY_DATASET_"
        "LOCKED_READY_FOR_S07C_GROUPED_OOF_QUANTILE_FEASIBILITY"
    )
    if lock.get("decision") != expected_decision:
        raise RuntimeError(
            f"Unexpected S07-B decision: {lock.get('decision')}"
        )
    if lock.get("table_sha256") != EXPECTED_UTILITY_TABLE_SHA256:
        raise RuntimeError("S07-B lock utility-table SHA provenance mismatch.")

    return {
        "protocol_sha256": protocol_sha,
        "s07b_lock_sha256": s07b_lock_sha,
        "utility_table_sha256": utility_sha,
        "s03b_helper_sha256": s03b_sha,
        "training_helper_sha256": training_sha,
    }


def load_and_validate_helper():
    helper = import_module_from_path(
        S03B_HELPER,
        "q1_s00a_frozen_s03b_helper",
    )
    required = (
        "load_source_val_rows",
        "validate_checkpoints",
        "import_training_helper",
        "load_model",
        "configure_tent",
        "snapshot_params",
        "restore_params",
        "final_logit",
        "mean_binary_entropy",
        "apply_perturbation",
        "image_to_model_tensor",
        "fold_for_sample",
        "PERTURBATIONS",
        "SEEDS",
        "TENT_LR",
        "TENT_WEIGHT_DECAY",
        "CHECKPOINTS",
        "DATA_ROOT",
    )
    for name in required:
        if not hasattr(helper, name):
            raise RuntimeError(f"Frozen S03-B helper missing symbol: {name}")

    if tuple(helper.PERTURBATIONS) != PERTURBATIONS:
        raise RuntimeError("Frozen perturbation bank changed.")
    if tuple(helper.SEEDS) != SEEDS:
        raise RuntimeError("Frozen seed list changed.")
    if abs(float(helper.TENT_LR) - 1e-3) > 1e-15:
        raise RuntimeError("Frozen TENT learning rate changed.")
    if abs(float(helper.TENT_WEIGHT_DECAY) - 0.0) > 1e-15:
        raise RuntimeError("Frozen TENT weight decay changed.")

    for seed in SEEDS:
        declared = str(helper.CHECKPOINTS[seed]["sha256"])
        expected = EXPECTED_CHECKPOINT_SHA256[seed]
        if declared != expected:
            raise RuntimeError(f"Checkpoint declaration mismatch seed={seed}")
        actual = file_sha256(Path(helper.CHECKPOINTS[seed]["path"]))
        if actual != expected:
            raise RuntimeError(f"Checkpoint file SHA mismatch seed={seed}")

    helper.validate_checkpoints()
    return helper


# ---------------------------------------------------------------------
# Frozen A1 outcome table
# ---------------------------------------------------------------------

def class_from_delta(delta: float) -> str:
    if delta <= HARM_THRESHOLD:
        return "HARM"
    if delta >= BENEFIT_THRESHOLD:
        return "BENEFIT"
    return "NEUTRAL"


def load_and_validate_a1_outcomes():
    rows, fields = read_csv(UTILITY_TABLE)
    required = {
        "seed",
        "sample_id",
        "source_group_id",
        "fold",
        "perturbation",
        "base_case_id",
        "action",
        "source_dice",
        "action_dice",
        "delta_dice",
        "harmful",
        "neutral",
        "beneficial",
    }
    missing = sorted(required - set(fields))
    if missing:
        raise RuntimeError(f"Utility table missing required fields: {missing}")

    a1 = [r for r in rows if r["action"] == A1_ACTION]
    if len(a1) != EXPECTED_A1_ROWS:
        raise RuntimeError(
            f"A1 row count mismatch: expected={EXPECTED_A1_ROWS} actual={len(a1)}"
        )

    groups = defaultdict(set)
    fold_group_counts = Counter()
    class_counts = Counter()
    keys = set()

    for r in a1:
        seed = int(r["seed"])
        if seed not in SEEDS:
            raise RuntimeError(f"Unexpected seed: {seed}")
        if r["perturbation"] not in PERTURBATIONS:
            raise RuntimeError(f"Unexpected perturbation: {r['perturbation']}")

        fold = int(r["fold"])
        if not 0 <= fold < N_FOLDS:
            raise RuntimeError(f"Invalid fold: {fold}")

        group = r["source_group_id"]
        groups[group].add(fold)

        delta = safe_float(r["delta_dice"])
        cls = class_from_delta(delta)
        class_counts[cls] += 1

        harmful = int(r["harmful"])
        neutral = int(r["neutral"])
        beneficial = int(r["beneficial"])
        expected_triplet = {
            "HARM": (1, 0, 0),
            "NEUTRAL": (0, 1, 0),
            "BENEFIT": (0, 0, 1),
        }[cls]
        if (harmful, neutral, beneficial) != expected_triplet:
            raise RuntimeError(
                f"Frozen label mismatch base_case_id={r['base_case_id']}"
            )

        key = (seed, r["sample_id"], r["perturbation"])
        if key in keys:
            raise RuntimeError(f"Duplicate A1 base-case key: {key}")
        keys.add(key)

    if len(groups) != SOURCE_GROUPS:
        raise RuntimeError(
            f"Source-group count mismatch: expected={SOURCE_GROUPS} actual={len(groups)}"
        )

    leakage = [
        group for group, folds in groups.items() if len(folds) != 1
    ]
    if leakage:
        raise RuntimeError(f"Source-group fold leakage: {len(leakage)} groups")

    for group, folds in groups.items():
        fold_group_counts[next(iter(folds))] += 1

    if dict(sorted(fold_group_counts.items())) != EXPECTED_FOLD_GROUP_COUNTS:
        raise RuntimeError(
            "Frozen fold-group counts changed: "
            f"{dict(sorted(fold_group_counts.items()))}"
        )

    if dict(class_counts) != EXPECTED_CLASS_COUNTS:
        raise RuntimeError(
            f"Frozen class counts changed: expected={EXPECTED_CLASS_COUNTS} "
            f"actual={dict(class_counts)}"
        )

    return sorted(
        a1,
        key=lambda r: (
            int(r["seed"]),
            r["sample_id"],
            r["perturbation"],
        ),
    ), {
        "a1_rows": len(a1),
        "source_groups": len(groups),
        "fold_group_counts": dict(sorted(fold_group_counts.items())),
        "class_counts": dict(class_counts),
    }


# ---------------------------------------------------------------------
# Source-only pre-adaptation features
# ---------------------------------------------------------------------

def binary_entropy_from_probability(p: np.ndarray) -> np.ndarray:
    eps = 1e-7
    q = np.clip(p.astype(np.float64), eps, 1.0 - eps)
    return -(q * np.log(q) + (1.0 - q) * np.log(1.0 - q))


def boundary_density(mask: np.ndarray) -> float:
    m = mask.astype(np.uint8)
    diff_h = np.not_equal(m[1:, :], m[:-1, :]).sum()
    diff_w = np.not_equal(m[:, 1:], m[:, :-1]).sum()
    denom = max(
        (m.shape[0] - 1) * m.shape[1]
        + m.shape[0] * (m.shape[1] - 1),
        1,
    )
    return float((diff_h + diff_w) / denom)


def extract_source_features(logit_2d: np.ndarray, tent_loss: float) -> Dict[str, float]:
    z = np.asarray(logit_2d, dtype=np.float64)
    if z.ndim != 2 or not np.isfinite(z).all():
        raise RuntimeError(f"Invalid source logit shape/values: {z.shape}")

    # Stable sigmoid.
    p = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    p[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    p[~pos] = ez / (1.0 + ez)

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
        "source_high_entropy_fraction_050": float(
            np.mean(ent >= 0.50)
        ),
        "source_logit_abs_mean": float(np.mean(np.abs(z))),
        "source_logit_abs_std": float(np.std(np.abs(z))),
        "tent_preupdate_entropy_loss": float(tent_loss),
    }

    if set(feat) != set(SOURCE_FEATURE_NAMES):
        raise RuntimeError("Source feature schema mismatch.")
    if not all(np.isfinite(list(feat.values()))):
        raise RuntimeError("Non-finite source feature.")
    return feat


# ---------------------------------------------------------------------
# No-update gradient probe
# ---------------------------------------------------------------------

def trainable_param_names(model: nn.Module, params: Sequence[torch.Tensor]) -> List[str]:
    by_id = {id(p): name for name, p in model.named_parameters()}
    names = []
    for p in params:
        if id(p) not in by_id:
            raise RuntimeError("TENT parameter not found in model.named_parameters().")
        names.append(by_id[id(p)])
    if len(names) != len(set(names)):
        raise RuntimeError("Duplicate TENT parameter names.")
    return names


def sanitize_feature_token(name: str) -> str:
    return (
        name.replace(".", "__")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )


def gradient_feature_names(param_names: Sequence[str]) -> List[str]:
    names = list(GLOBAL_GRADIENT_FEATURE_NAMES)
    for param_name in param_names:
        token = sanitize_feature_token(param_name)
        for stat in PER_PARAM_GRAD_STATS:
            names.append(f"grad_param__{token}__{stat}")
    validate_feature_names(names)
    return names


def _tensor_grad_stats(g: torch.Tensor) -> Dict[str, float]:
    x = g.detach().float().reshape(-1).cpu().numpy().astype(np.float64)
    if x.size == 0 or not np.isfinite(x).all():
        raise RuntimeError("Invalid gradient tensor.")
    return {
        "l1": float(np.sum(np.abs(x))),
        "l2": float(np.sqrt(np.sum(x * x))),
        "mean": float(np.mean(np.abs(x))),
        "signed_mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "abs_max": float(np.max(np.abs(x))),
    }


def _energy_entropy(energies: np.ndarray) -> float:
    e = np.asarray(energies, dtype=np.float64)
    total = float(e.sum())
    if total <= 0:
        return 0.0
    p = e / total
    p = p[p > 0]
    if p.size <= 1:
        return 0.0
    value = -float(np.sum(p * np.log(p))) / math.log(len(e))
    return float(value)


def extract_gradient_features(
    params: Sequence[torch.Tensor],
    param_names: Sequence[str],
) -> Dict[str, float]:
    if len(params) != len(param_names):
        raise RuntimeError("Gradient parameter/name count mismatch.")

    flat_parts = []
    per_stats = []
    energies = []

    for p, name in zip(params, param_names):
        if p.grad is None:
            raise RuntimeError(f"Missing TENT gradient: {name}")
        if not torch.isfinite(p.grad).all():
            raise RuntimeError(f"Non-finite TENT gradient: {name}")
        g = p.grad.detach().float()
        flat_parts.append(g.reshape(-1))
        stat = _tensor_grad_stats(g)
        per_stats.append((name, stat))
        energies.append(stat["l2"] ** 2)

    flat = torch.cat(flat_parts).cpu().numpy().astype(np.float64)
    abs_flat = np.abs(flat)
    sq = flat * flat
    group_energy = np.asarray(energies, dtype=np.float64)

    n_groups = len(group_energy)
    n1 = max(1, n_groups // 3)
    n2 = max(n1 + 1, (2 * n_groups) // 3)
    n2 = min(n2, n_groups)

    early = float(group_energy[:n1].sum())
    middle = float(group_energy[n1:n2].sum())
    late = float(group_energy[n2:].sum())

    weight_energy = 0.0
    bias_energy = 0.0
    for name, energy in zip(param_names, group_energy):
        if name.endswith(".weight"):
            weight_energy += float(energy)
        elif name.endswith(".bias"):
            bias_energy += float(energy)
        else:
            raise RuntimeError(
                f"Unexpected TENT affine parameter suffix: {name}"
            )

    sorted_energy = np.sort(group_energy)[::-1]
    total_energy = float(group_energy.sum())
    top1 = (
        float(sorted_energy[:1].sum() / total_energy)
        if total_energy > 0 else 0.0
    )
    top5 = (
        float(sorted_energy[: min(5, n_groups)].sum() / total_energy)
        if total_energy > 0 else 0.0
    )

    feat = {
        "grad_global_l1": float(abs_flat.sum()),
        "grad_global_l2": float(np.sqrt(sq.sum())),
        "grad_global_rms": float(np.sqrt(np.mean(sq))),
        "grad_global_abs_max": float(abs_flat.max()),
        "grad_global_mean": float(np.mean(abs_flat)),
        "grad_global_signed_mean": float(np.mean(flat)),
        "grad_global_std": float(np.std(flat)),
        "grad_global_variance": float(np.var(flat)),
        "grad_param_group_count": float(n_groups),
        "grad_nonzero_fraction": float(np.mean(abs_flat > 0)),
        "grad_energy_early": early,
        "grad_energy_middle": middle,
        "grad_energy_late": late,
        "grad_energy_late_over_early": float(late / max(early, 1e-30)),
        "grad_energy_middle_over_early": float(middle / max(early, 1e-30)),
        "grad_weight_energy": weight_energy,
        "grad_bias_energy": bias_energy,
        "grad_weight_over_bias_energy": float(
            weight_energy / max(bias_energy, 1e-30)
        ),
        "grad_group_energy_entropy": _energy_entropy(group_energy),
        "grad_top1_energy_fraction": top1,
        "grad_top5_energy_fraction": top5,
    }

    for name, stat in per_stats:
        token = sanitize_feature_token(name)
        for key in PER_PARAM_GRAD_STATS:
            feat[f"grad_param__{token}__{key}"] = float(stat[key])

    expected = gradient_feature_names(param_names)
    if set(feat) != set(expected):
        missing = sorted(set(expected) - set(feat))
        extra = sorted(set(feat) - set(expected))
        raise RuntimeError(
            f"Gradient feature schema mismatch missing={missing} extra={extra}"
        )
    if not all(np.isfinite(list(feat.values()))):
        raise RuntimeError("Non-finite gradient feature.")
    return feat


def assert_params_equal(
    params: Sequence[torch.Tensor],
    source_values: Sequence[torch.Tensor],
):
    if len(params) != len(source_values):
        raise RuntimeError("Parameter snapshot count mismatch.")
    for idx, (p, src) in enumerate(zip(params, source_values)):
        if not torch.equal(p.detach(), src):
            max_delta = float(
                torch.max(torch.abs(p.detach() - src)).cpu()
            )
            raise RuntimeError(
                f"NO-UPDATE integrity failure param_index={idx} "
                f"max_abs_delta={max_delta}"
            )


def clear_grads(params: Sequence[torch.Tensor]):
    for p in params:
        p.grad = None


def source_forward_and_gradient_probe(
    helper,
    model,
    params,
    source_values,
    x: torch.Tensor,
):
    # Exact episodic reset.
    helper.restore_params(params, source_values)
    clear_grads(params)

    # Source-only prediction: no adapted state and no labels.
    model.eval()
    with torch.no_grad():
        z_source = helper.final_logit(model, x).detach()

    # Pre-adaptation TENT objective. No optimizer is constructed.
    helper.restore_params(params, source_values)
    clear_grads(params)
    model.train()

    z_pre = helper.final_logit(model, x)
    loss = helper.mean_binary_entropy(z_pre)
    loss.backward()

    # Critical integrity check: backward must not mutate parameters.
    assert_params_equal(params, source_values)

    result = (
        z_source[0, 0].float().cpu().numpy().copy(),
        float(loss.detach().cpu()),
    )

    del z_pre, loss
    return result


# ---------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------

def expected_seed_key_set(outcome_rows, seed: int):
    return {
        (int(r["seed"]), r["sample_id"], r["perturbation"])
        for r in outcome_rows
        if int(r["seed"]) == seed
    }


def validate_seed_feature_rows(
    feature_rows,
    outcome_rows,
    seed: int,
    source_feature_names,
    grad_feature_names,
):
    expected = expected_seed_key_set(outcome_rows, seed)
    actual = {
        (int(r["seed"]), r["sample_id"], r["perturbation"])
        for r in feature_rows
    }
    if actual != expected:
        raise RuntimeError(
            f"Seed feature key mismatch seed={seed}: "
            f"expected={len(expected)} actual={len(actual)} "
            f"missing={len(expected-actual)} extra={len(actual-expected)}"
        )
    if len(feature_rows) != SOURCE_GROUPS * len(PERTURBATIONS):
        raise RuntimeError(
            f"Seed row count mismatch seed={seed}: {len(feature_rows)}"
        )
    required = (
        {
            "seed",
            "sample_id",
            "source_group_id",
            "fold",
            "perturbation",
            "base_case_id",
            "delta_dice",
            "harmful",
            "neutral",
            "beneficial",
            "outcome_class",
        }
        | set(source_feature_names)
        | set(grad_feature_names)
    )
    for row in feature_rows:
        missing = required - set(row)
        if missing:
            raise RuntimeError(f"Seed feature row missing: {sorted(missing)}")
        for name in source_feature_names + grad_feature_names:
            if not math.isfinite(float(row[name])):
                raise RuntimeError(f"Non-finite feature {name}")


def extract_seed_features(
    helper,
    training_helper,
    outcome_rows,
    seed: int,
    device: torch.device,
    seed_cache_path: Path,
):
    source_rows = helper.load_source_val_rows()
    source_by_id = {r["sample_id"]: r for r in source_rows}
    if len(source_by_id) != SOURCE_GROUPS:
        raise RuntimeError("Source-val sample_id uniqueness/count mismatch.")

    seed_outcomes = [
        r for r in outcome_rows if int(r["seed"]) == seed
    ]
    outcome_by_key = {
        (r["sample_id"], r["perturbation"]): r
        for r in seed_outcomes
    }
    if len(outcome_by_key) != SOURCE_GROUPS * len(PERTURBATIONS):
        raise RuntimeError(f"Seed outcome key count mismatch seed={seed}")

    model = helper.load_model(training_helper, seed, device)
    params = helper.configure_tent(model)
    source_values = helper.snapshot_params(params)
    assert_params_equal(params, source_values)

    param_names = trainable_param_names(model, params)
    grad_names = gradient_feature_names(param_names)
    validate_feature_names(SOURCE_FEATURE_NAMES)
    validate_feature_names(grad_names)
    validate_feature_names(SOURCE_FEATURE_NAMES + grad_names)

    all_rows = []
    total = len(source_rows) * len(PERTURBATIONS)
    pbar = tqdm(
        total=total,
        desc=f"Q1-S00A gradient probe seed {seed}",
        unit="case",
        dynamic_ncols=True,
    )

    for source_row in source_rows:
        sample_id = source_row["sample_id"]
        image_path = Path(helper.DATA_ROOT) / Path(source_row["image_relpath"])
        if not image_path.exists():
            raise FileNotFoundError(image_path)

        # IMPORTANT: mask_relpath is never dereferenced here.
        with Image.open(image_path) as im:
            native = im.convert("RGB")

        for perturbation in PERTURBATIONS:
            key = (sample_id, perturbation)
            if key not in outcome_by_key:
                raise RuntimeError(f"Missing frozen A1 outcome key: {key}")
            target = outcome_by_key[key]

            perturbed = helper.apply_perturbation(
                native,
                perturbation,
                sample_id,
            )
            x = helper.image_to_model_tensor(
                training_helper,
                perturbed,
            ).to(device, non_blocking=True)

            z_source, tent_loss = source_forward_and_gradient_probe(
                helper,
                model,
                params,
                source_values,
                x,
            )

            src_feat = extract_source_features(z_source, tent_loss)
            grad_feat = extract_gradient_features(params, param_names)

            # AFTER feature extraction, still verify no update occurred.
            assert_params_equal(params, source_values)

            delta = safe_float(target["delta_dice"])
            cls = class_from_delta(delta)

            row = {
                "seed": seed,
                "sample_id": sample_id,
                "source_group_id": target["source_group_id"],
                "fold": int(target["fold"]),
                "perturbation": perturbation,
                "base_case_id": target["base_case_id"],
                "delta_dice": delta,
                "harmful": int(target["harmful"]),
                "neutral": int(target["neutral"]),
                "beneficial": int(target["beneficial"]),
                "outcome_class": cls,
            }
            row.update(src_feat)
            row.update(grad_feat)
            all_rows.append(row)

            clear_grads(params)
            assert_params_equal(params, source_values)

            pbar.set_postfix(
                loss=f"{tent_loss:.4f}",
                gL2=f"{grad_feat['grad_global_l2']:.2e}",
                cls=cls[0],
            )
            pbar.update(1)

            del x, z_source

    pbar.close()

    validate_seed_feature_rows(
        all_rows,
        outcome_rows,
        seed,
        SOURCE_FEATURE_NAMES,
        grad_names,
    )

    fields = [
        "seed",
        "sample_id",
        "source_group_id",
        "fold",
        "perturbation",
        "base_case_id",
        "delta_dice",
        "harmful",
        "neutral",
        "beneficial",
        "outcome_class",
    ] + SOURCE_FEATURE_NAMES + grad_names

    write_csv(seed_cache_path, all_rows, fields)

    del model, params, source_values
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return all_rows, param_names, grad_names


def load_seed_cache(
    path: Path,
    outcome_rows,
    seed: int,
    expected_grad_names: Sequence[str] | None,
):
    rows, fields = read_csv(path)
    grad_names = [
        name for name in fields if name.startswith("grad_")
    ]
    if expected_grad_names is not None and list(grad_names) != list(expected_grad_names):
        raise RuntimeError(
            f"Resume gradient schema mismatch seed={seed}"
        )

    validate_seed_feature_rows(
        rows,
        outcome_rows,
        seed,
        SOURCE_FEATURE_NAMES,
        grad_names,
    )
    return rows, grad_names


# ---------------------------------------------------------------------
# Grouped OOF prediction
# ---------------------------------------------------------------------

def make_ridge():
    return Pipeline([
        ("scale", StandardScaler()),
        ("reg", Ridge(alpha=RIDGE_ALPHA)),
    ])


def make_logistic():
    return Pipeline([
        ("scale", StandardScaler()),
        (
            "clf",
            LogisticRegression(
                C=LOGISTIC_C,
                penalty="l2",
                solver="liblinear",
                class_weight="balanced",
                max_iter=LOGISTIC_MAX_ITER,
                random_state=OOF_RANDOM_STATE,
            ),
        ),
    ])


def matrix(rows, feature_names: Sequence[str]) -> np.ndarray:
    x = np.asarray(
        [
            [float(row[name]) for name in feature_names]
            for row in rows
        ],
        dtype=np.float64,
    )
    if x.ndim != 2 or x.shape[1] != len(feature_names):
        raise RuntimeError("Feature matrix shape mismatch.")
    if not np.isfinite(x).all():
        raise RuntimeError("Feature matrix contains NaN/Inf.")
    return x


def binary_target(rows, field: str) -> np.ndarray:
    y = np.asarray([int(row[field]) for row in rows], dtype=np.int64)
    if set(np.unique(y).tolist()) - {0, 1}:
        raise RuntimeError(f"Invalid binary target: {field}")
    return y


def run_grouped_oof(rows, family: str, feature_names: Sequence[str]):
    predictions = []

    for fold in range(N_FOLDS):
        train_rows = [r for r in rows if int(r["fold"]) != fold]
        val_rows = [r for r in rows if int(r["fold"]) == fold]

        train_groups = {r["source_group_id"] for r in train_rows}
        val_groups = {r["source_group_id"] for r in val_rows}
        overlap = train_groups & val_groups
        if overlap:
            raise RuntimeError(
                f"Fold leakage family={family} fold={fold}: {len(overlap)} groups"
            )

        x_train = matrix(train_rows, feature_names)
        x_val = matrix(val_rows, feature_names)

        y_delta_train = np.asarray(
            [float(r["delta_dice"]) for r in train_rows],
            dtype=np.float64,
        )
        y_harm_train = binary_target(train_rows, "harmful")
        y_benefit_train = binary_target(train_rows, "beneficial")

        if len(np.unique(y_harm_train)) != 2:
            raise RuntimeError(f"HARM train fold lacks both classes: {fold}")
        if len(np.unique(y_benefit_train)) != 2:
            raise RuntimeError(f"BENEFIT train fold lacks both classes: {fold}")

        ridge = make_ridge()
        harm_model = make_logistic()
        benefit_model = make_logistic()

        ridge.fit(x_train, y_delta_train)
        harm_model.fit(x_train, y_harm_train)
        benefit_model.fit(x_train, y_benefit_train)

        pred_delta = ridge.predict(x_val)
        pred_harm = harm_model.predict_proba(x_val)[:, 1]
        pred_benefit = benefit_model.predict_proba(x_val)[:, 1]

        for r, pd, ph, pb in zip(
            val_rows,
            pred_delta,
            pred_harm,
            pred_benefit,
        ):
            predictions.append({
                "family": family,
                "seed": int(r["seed"]),
                "sample_id": r["sample_id"],
                "source_group_id": r["source_group_id"],
                "fold": int(r["fold"]),
                "perturbation": r["perturbation"],
                "base_case_id": r["base_case_id"],
                "delta_dice": float(r["delta_dice"]),
                "harmful": int(r["harmful"]),
                "neutral": int(r["neutral"]),
                "beneficial": int(r["beneficial"]),
                "pred_delta_dice": float(pd),
                "pred_harm_probability": float(ph),
                "pred_benefit_probability": float(pb),
            })

    if len(predictions) != EXPECTED_A1_ROWS:
        raise RuntimeError(
            f"OOF prediction row count mismatch family={family}: "
            f"{len(predictions)}"
        )

    keys = [
        (r["seed"], r["sample_id"], r["perturbation"])
        for r in predictions
    ]
    if len(keys) != len(set(keys)):
        raise RuntimeError(f"Duplicate OOF keys family={family}")

    return sorted(
        predictions,
        key=lambda r: (r["seed"], r["sample_id"], r["perturbation"]),
    )


def metrics_for_predictions(pred_rows):
    y_delta = np.asarray(
        [float(r["delta_dice"]) for r in pred_rows],
        dtype=np.float64,
    )
    p_delta = np.asarray(
        [float(r["pred_delta_dice"]) for r in pred_rows],
        dtype=np.float64,
    )
    y_harm = np.asarray(
        [int(r["harmful"]) for r in pred_rows],
        dtype=np.int64,
    )
    p_harm = np.asarray(
        [float(r["pred_harm_probability"]) for r in pred_rows],
        dtype=np.float64,
    )
    y_benefit = np.asarray(
        [int(r["beneficial"]) for r in pred_rows],
        dtype=np.int64,
    )
    p_benefit = np.asarray(
        [float(r["pred_benefit_probability"]) for r in pred_rows],
        dtype=np.float64,
    )

    return {
        "delta_dice_spearman": stable_spearman(y_delta, p_delta),
        "delta_dice_mae": float(mean_absolute_error(y_delta, p_delta)),
        "harm_auroc": float(roc_auc_score(y_harm, p_harm)),
        "harm_auprc": float(average_precision_score(y_harm, p_harm)),
        "benefit_auroc": float(roc_auc_score(y_benefit, p_benefit)),
        "benefit_auprc": float(average_precision_score(y_benefit, p_benefit)),
    }


def decide(metrics_by_family):
    b0 = metrics_by_family["B0_PRE_SOURCE"]
    b2 = metrics_by_family["B2_SOURCE_PLUS_GRADIENT"]
    increment = b2["benefit_auroc"] - b0["benefit_auroc"]

    criteria = {
        "delta_dice_spearman": {
            "value": b2["delta_dice_spearman"],
            "threshold": GO_MIN_SPEARMAN,
            "pass": b2["delta_dice_spearman"] >= GO_MIN_SPEARMAN,
        },
        "harm_auroc": {
            "value": b2["harm_auroc"],
            "threshold": GO_MIN_HARM_AUROC,
            "pass": b2["harm_auroc"] >= GO_MIN_HARM_AUROC,
        },
        "benefit_auroc": {
            "value": b2["benefit_auroc"],
            "threshold": GO_MIN_BENEFIT_AUROC,
            "pass": b2["benefit_auroc"] >= GO_MIN_BENEFIT_AUROC,
        },
        "b2_minus_b0_benefit_auroc": {
            "value": increment,
            "threshold": GO_MIN_BENEFIT_INCREMENT,
            "pass": increment >= GO_MIN_BENEFIT_INCREMENT,
        },
    }

    if all(v["pass"] for v in criteria.values()):
        decision = PASS_DECISION
    elif increment < GO_MIN_BENEFIT_INCREMENT:
        decision = STOP_NO_INCREMENT
    elif (
        b2["harm_auroc"] >= GO_MIN_HARM_AUROC
        and (
            b2["benefit_auroc"] < GO_MIN_BENEFIT_AUROC
            or b2["delta_dice_spearman"] < GO_MIN_SPEARMAN
        )
    ):
        decision = STOP_HARM_ONLY
    else:
        decision = STOP_SIMPLE

    return decision, criteria


# ---------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------

def prepare_build_dir(output_dir: Path, resume: bool):
    if output_dir.exists():
        raise FileExistsError(
            f"Final Q1-S00A output already exists: {output_dir}"
        )

    build_dir = Path(str(output_dir) + "__building")
    if build_dir.exists() and not resume:
        raise FileExistsError(
            f"Partial Q1-S00A build exists: {build_dir}. "
            "Use --resume only after a technical interruption."
        )
    if not build_dir.exists():
        build_dir.mkdir(parents=True, exist_ok=False)
    return build_dir


def run(args):
    provenance = validate_provenance()
    outcome_rows, dataset_audit = load_and_validate_a1_outcomes()
    helper = load_and_validate_helper()
    training_helper = helper.import_training_helper()

    source_rows = helper.load_source_val_rows()
    if len(source_rows) != SOURCE_GROUPS:
        raise RuntimeError("Frozen source-val count mismatch.")

    # Fold relation must match frozen utility table.
    outcome_fold_by_sample = {}
    for r in outcome_rows:
        sid = r["sample_id"]
        fold = int(r["fold"])
        if sid in outcome_fold_by_sample and outcome_fold_by_sample[sid] != fold:
            raise RuntimeError(f"Outcome fold inconsistency sample={sid}")
        outcome_fold_by_sample[sid] = fold

    for r in source_rows:
        sid = r["sample_id"]
        helper_fold = int(helper.fold_for_sample(sid))
        if sid not in outcome_fold_by_sample:
            raise RuntimeError(f"Source sample absent from outcome table: {sid}")
        if helper_fold != outcome_fold_by_sample[sid]:
            raise RuntimeError(
                f"Frozen fold mismatch sample={sid}: "
                f"helper={helper_fold} outcome={outcome_fold_by_sample[sid]}"
            )

    build_dir = prepare_build_dir(args.output_dir, args.resume)

    # Keep an exact frozen protocol copy in the experiment output.
    protocol_copy = build_dir / "preregistered_protocol_copy.md"
    if not protocol_copy.exists():
        shutil.copy2(PROTOCOL, protocol_copy)
    if file_sha256(protocol_copy) != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("Copied Q1-S00A protocol SHA mismatch.")

    device = torch.device(
        "cuda"
        if torch.cuda.is_available() and not args.cpu
        else "cpu"
    )

    print(f"===== Q1-S00A PRE-ADAPTATION IAE FEASIBILITY =====")
    print("INTEGRITY_SCHEMA: POSITIVE_POLARITY_FIX2")
    print(f"Script version: {VERSION}")
    print(f"Build: {BUILD}")
    print(f"Device: {device}")
    print(f"Source groups: {SOURCE_GROUPS}")
    print(f"A1 rows: {EXPECTED_A1_ROWS}")
    print(f"Folds: {N_FOLDS}")
    print(f"Action: {A1_ACTION}")
    print("optimizer.step during probe: FORBIDDEN / NOT USED")
    print("adapted forward during probe: FORBIDDEN / NOT USED")
    print("source masks opened during feature extraction: NO")
    print("target data used: NO")
    print()

    all_feature_rows = []

    # Freeze the exact adaptable-parameter schema before reading/reusing any
    # per-seed cache. This also makes --resume independently auditable.
    schema_model = helper.load_model(
        training_helper,
        SEEDS[0],
        device,
    )
    schema_params = helper.configure_tent(schema_model)
    schema_source_values = helper.snapshot_params(schema_params)
    assert_params_equal(schema_params, schema_source_values)
    param_names_reference = trainable_param_names(
        schema_model,
        schema_params,
    )
    expected_grad_names = gradient_feature_names(
        param_names_reference
    )
    del schema_model, schema_params, schema_source_values
    if device.type == "cuda":
        torch.cuda.empty_cache()

    cache_dir = build_dir / "_seed_feature_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    for seed in SEEDS:
        seed_cache = cache_dir / f"gradient_probe_features_seed_{seed}.csv"

        if args.resume and seed_cache.exists():
            rows, grad_names = load_seed_cache(
                seed_cache,
                outcome_rows,
                seed,
                expected_grad_names,
            )
            print(
                f"[RESUME] validated seed cache {seed}: "
                f"{len(rows)} rows SHA={file_sha256(seed_cache)}"
            )
            all_feature_rows.extend(rows)
            continue

        rows, param_names, grad_names = extract_seed_features(
            helper,
            training_helper,
            outcome_rows,
            seed,
            device,
            seed_cache,
        )
        if list(grad_names) != list(expected_grad_names):
            raise RuntimeError(f"Gradient schema differs across seeds: {seed}")
        if list(param_names) != list(param_names_reference):
            raise RuntimeError(f"TENT parameter names differ across seeds: {seed}")

        all_feature_rows.extend(rows)
        print(
            f"[LOCK] seed={seed} rows={len(rows)} "
            f"SHA={file_sha256(seed_cache)}"
        )

    if expected_grad_names is None:
        raise RuntimeError("No gradient feature schema generated.")

    # Global feature-table integrity.
    if len(all_feature_rows) != EXPECTED_A1_ROWS:
        raise RuntimeError(
            f"Combined feature row mismatch: {len(all_feature_rows)}"
        )

    global_keys = [
        (
            int(r["seed"]),
            r["sample_id"],
            r["perturbation"],
        )
        for r in all_feature_rows
    ]
    if len(global_keys) != len(set(global_keys)):
        raise RuntimeError("Duplicate combined feature keys.")

    validate_feature_names(SOURCE_FEATURE_NAMES)
    validate_feature_names(expected_grad_names)
    validate_feature_names(SOURCE_FEATURE_NAMES + expected_grad_names)

    # Freeze actual feature table.
    feature_fields = [
        "seed",
        "sample_id",
        "source_group_id",
        "fold",
        "perturbation",
        "base_case_id",
        "delta_dice",
        "harmful",
        "neutral",
        "beneficial",
        "outcome_class",
    ] + SOURCE_FEATURE_NAMES + list(expected_grad_names)

    feature_table = build_dir / "gradient_probe_features.csv"
    write_csv(feature_table, all_feature_rows, feature_fields)
    feature_table_sha = file_sha256(feature_table)

    # Fold assignment output (one row per source group).
    fold_rows = []
    seen_groups = {}
    for r in all_feature_rows:
        group = r["source_group_id"]
        fold = int(r["fold"])
        if group in seen_groups and seen_groups[group] != fold:
            raise RuntimeError(f"Group fold leakage in extracted features: {group}")
        seen_groups[group] = fold
    for group, fold in sorted(seen_groups.items()):
        fold_rows.append({
            "source_group_id": group,
            "fold": fold,
        })
    fold_path = build_dir / "fold_assignment.csv"
    write_csv(
        fold_path,
        fold_rows,
        ["source_group_id", "fold"],
    )

    # Dataset audit.
    dataset_audit_full = {
        **dataset_audit,
        "action": A1_ACTION,
        "seeds": list(SEEDS),
        "perturbations": list(PERTURBATIONS),
        "expected_rows": EXPECTED_A1_ROWS,
        "source_masks_opened_for_feature_extraction": False,
        "target_data_used": False,
        "adapted_prediction_generated": False,
        "optimizer_step_called_during_probe": False,
        "feature_table_sha256": feature_table_sha,
        **provenance,
    }
    dataset_audit_path = build_dir / "dataset_audit.json"
    write_json(dataset_audit_path, dataset_audit_full)

    # Feature manifest.
    feature_manifest = {
        "script_version": VERSION,
        "build": BUILD,
        "B0_PRE_SOURCE": list(SOURCE_FEATURE_NAMES),
        "B1_GRADIENT_PROBE": list(expected_grad_names),
        "B2_SOURCE_PLUS_GRADIENT": (
            list(SOURCE_FEATURE_NAMES) + list(expected_grad_names)
        ),
        "source_feature_count": len(SOURCE_FEATURE_NAMES),
        "gradient_feature_count": len(expected_grad_names),
        "combined_feature_count": (
            len(SOURCE_FEATURE_NAMES) + len(expected_grad_names)
        ),
        "tent_param_names": param_names_reference,
        "forbidden_feature_tokens": list(FORBIDDEN_FEATURE_TOKENS),
        "adapted_features_used": False,
        "outcome_features_used": False,
    }
    feature_manifest_path = build_dir / "preadapt_feature_manifest.json"
    write_json(feature_manifest_path, feature_manifest)

    # Three preregistered OOF families.
    families = {
        "B0_PRE_SOURCE": list(SOURCE_FEATURE_NAMES),
        "B1_GRADIENT_PROBE": list(expected_grad_names),
        "B2_SOURCE_PLUS_GRADIENT": (
            list(SOURCE_FEATURE_NAMES) + list(expected_grad_names)
        ),
    }

    all_oof = []
    metrics_by_family = {}

    for family, names in families.items():
        print(
            f"[OOF] {family}: {len(names)} features | "
            f"Ridge alpha={RIDGE_ALPHA} | Logistic C={LOGISTIC_C}"
        )
        pred = run_grouped_oof(
            all_feature_rows,
            family,
            names,
        )
        met = metrics_for_predictions(pred)
        metrics_by_family[family] = met
        all_oof.extend(pred)

        print(
            f"  Spearman={met['delta_dice_spearman']:.6f} "
            f"HARM_AUROC={met['harm_auroc']:.6f} "
            f"BENEFIT_AUROC={met['benefit_auroc']:.6f}"
        )

    oof_fields = [
        "family",
        "seed",
        "sample_id",
        "source_group_id",
        "fold",
        "perturbation",
        "base_case_id",
        "delta_dice",
        "harmful",
        "neutral",
        "beneficial",
        "pred_delta_dice",
        "pred_harm_probability",
        "pred_benefit_probability",
    ]
    oof_path = build_dir / "oof_predictions.csv"
    write_csv(oof_path, all_oof, oof_fields)

    b0 = metrics_by_family["B0_PRE_SOURCE"]
    b2 = metrics_by_family["B2_SOURCE_PLUS_GRADIENT"]
    benefit_increment = b2["benefit_auroc"] - b0["benefit_auroc"]

    oof_metrics = {
        "script_version": VERSION,
        "build": BUILD,
        "models": {
            "delta_dice": {
                "family": "Ridge",
                "alpha": RIDGE_ALPHA,
                "standardization": "training_fold_only",
            },
            "harm": {
                "family": "LogisticRegression",
                "C": LOGISTIC_C,
                "class_weight": "balanced",
                "solver": "liblinear",
                "max_iter": LOGISTIC_MAX_ITER,
                "standardization": "training_fold_only",
            },
            "benefit": {
                "family": "LogisticRegression",
                "C": LOGISTIC_C,
                "class_weight": "balanced",
                "solver": "liblinear",
                "max_iter": LOGISTIC_MAX_ITER,
                "standardization": "training_fold_only",
            },
        },
        "families": metrics_by_family,
        "b2_minus_b0_benefit_auroc": benefit_increment,
    }
    oof_metrics_path = build_dir / "oof_metrics.json"
    write_json(oof_metrics_path, oof_metrics)

    comparison_rows = []
    for family, met in metrics_by_family.items():
        comparison_rows.append({
            "family": family,
            "feature_count": len(families[family]),
            **met,
            "benefit_auroc_minus_b0": (
                met["benefit_auroc"] - b0["benefit_auroc"]
            ),
        })
    comparison_path = build_dir / "b0_b1_b2_comparison.csv"
    write_csv(
        comparison_path,
        comparison_rows,
        [
            "family",
            "feature_count",
            "delta_dice_spearman",
            "delta_dice_mae",
            "harm_auroc",
            "harm_auprc",
            "benefit_auroc",
            "benefit_auprc",
            "benefit_auroc_minus_b0",
        ],
    )

    decision, criteria = decide(metrics_by_family)
    decision_path = build_dir / "decision.txt"
    decision_path.write_text(
        decision + "\n",
        encoding="utf-8",
    )

    # Leakage / integrity audit.
    family_oof_counts = Counter(r["family"] for r in all_oof)
    integrity = {
        "script_version": VERSION,
        "build": BUILD,
        "checks": {
            "source_groups_exact_145": len(seen_groups) == SOURCE_GROUPS,
            "rows_exact_4350": len(all_feature_rows) == EXPECTED_A1_ROWS,
            "folds_exact_5": set(seen_groups.values()) == set(range(N_FOLDS)),
            "grouped_fold_counts_match": (
                dict(Counter(seen_groups.values()))
                == EXPECTED_FOLD_GROUP_COUNTS
            ),
            "action_exact_A1_TENT_1STEP": True,
            # Positive-polarity integrity checks: True always means PASS.
            "target_data_not_loaded": True,
            "adapted_prediction_not_generated": True,
            "optimizer_step_not_called_during_probe": True,
            "source_masks_not_opened_during_feature_extraction": True,
            "forbidden_adapted_features_absent": True,
            "parameter_update_not_detected": True,
            "oof_count_each_family_exact_4350": all(
                family_oof_counts[k] == EXPECTED_A1_ROWS
                for k in families
            ),
            "feature_values_finite": all(
                math.isfinite(float(r[name]))
                for r in all_feature_rows
                for name in (
                    list(SOURCE_FEATURE_NAMES)
                    + list(expected_grad_names)
                )
            ),
            "standardization_fit_training_fold_only": True,
            "hyperparameter_sweep_not_used": True,
            "threshold_tuning_not_used": True,
        },
        "go_stop_criteria": criteria,
        "decision": decision,
    }

    if not all(integrity["checks"].values()):
        failed = [
            k for k, v in integrity["checks"].items() if not v
        ]
        raise RuntimeError(f"Integrity audit failed: {failed}")

    integrity_path = build_dir / "leakage_integrity_audit.json"
    write_json(integrity_path, integrity)

    # Human-readable run summary.
    summary = f"""===== Q1-S00A PRE-ADAPTATION IAE FEASIBILITY =====
Script version: {VERSION}
Build: {BUILD}

Frozen protocol:
  action={A1_ACTION}
  source groups={SOURCE_GROUPS}
  A1 rows={EXPECTED_A1_ROWS}
  folds={N_FOLDS}
  seeds={SEEDS}
  perturbations={len(PERTURBATIONS)}
  optimizer.step during probe=NO
  adapted prediction generated=NO
  source masks opened during feature extraction=NO
  target data used=NO
  SCRR used=NO
  hyperparameter sweep=NO
  threshold tuning=NO

Feature families:
  B0 PRE-SOURCE={len(SOURCE_FEATURE_NAMES)}
  B1 GRADIENT-PROBE={len(expected_grad_names)}
  B2 SOURCE+GRADIENT={len(SOURCE_FEATURE_NAMES) + len(expected_grad_names)}

[B0 PRE-SOURCE]
  DeltaDice Spearman={metrics_by_family["B0_PRE_SOURCE"]["delta_dice_spearman"]:.6f}
  DeltaDice MAE={metrics_by_family["B0_PRE_SOURCE"]["delta_dice_mae"]:.6f}
  HARM AUROC={metrics_by_family["B0_PRE_SOURCE"]["harm_auroc"]:.6f}
  HARM AUPRC={metrics_by_family["B0_PRE_SOURCE"]["harm_auprc"]:.6f}
  BENEFIT AUROC={metrics_by_family["B0_PRE_SOURCE"]["benefit_auroc"]:.6f}
  BENEFIT AUPRC={metrics_by_family["B0_PRE_SOURCE"]["benefit_auprc"]:.6f}

[B1 GRADIENT-PROBE]
  DeltaDice Spearman={metrics_by_family["B1_GRADIENT_PROBE"]["delta_dice_spearman"]:.6f}
  DeltaDice MAE={metrics_by_family["B1_GRADIENT_PROBE"]["delta_dice_mae"]:.6f}
  HARM AUROC={metrics_by_family["B1_GRADIENT_PROBE"]["harm_auroc"]:.6f}
  HARM AUPRC={metrics_by_family["B1_GRADIENT_PROBE"]["harm_auprc"]:.6f}
  BENEFIT AUROC={metrics_by_family["B1_GRADIENT_PROBE"]["benefit_auroc"]:.6f}
  BENEFIT AUPRC={metrics_by_family["B1_GRADIENT_PROBE"]["benefit_auprc"]:.6f}

[B2 SOURCE+GRADIENT]
  DeltaDice Spearman={metrics_by_family["B2_SOURCE_PLUS_GRADIENT"]["delta_dice_spearman"]:.6f}
  DeltaDice MAE={metrics_by_family["B2_SOURCE_PLUS_GRADIENT"]["delta_dice_mae"]:.6f}
  HARM AUROC={metrics_by_family["B2_SOURCE_PLUS_GRADIENT"]["harm_auroc"]:.6f}
  HARM AUPRC={metrics_by_family["B2_SOURCE_PLUS_GRADIENT"]["harm_auprc"]:.6f}
  BENEFIT AUROC={metrics_by_family["B2_SOURCE_PLUS_GRADIENT"]["benefit_auroc"]:.6f}
  BENEFIT AUPRC={metrics_by_family["B2_SOURCE_PLUS_GRADIENT"]["benefit_auprc"]:.6f}

[INCREMENTAL INFORMATION]
  B2-B0 BENEFIT AUROC={benefit_increment:+.6f}

[FROZEN GO CRITERIA]
  DeltaDice Spearman >= {GO_MIN_SPEARMAN:.2f}: {criteria["delta_dice_spearman"]["pass"]}
  HARM AUROC >= {GO_MIN_HARM_AUROC:.2f}: {criteria["harm_auroc"]["pass"]}
  BENEFIT AUROC >= {GO_MIN_BENEFIT_AUROC:.2f}: {criteria["benefit_auroc"]["pass"]}
  B2-B0 BENEFIT AUROC >= +{GO_MIN_BENEFIT_INCREMENT:.2f}: {criteria["b2_minus_b0_benefit_auroc"]["pass"]}

Decision: {decision}

PASS meaning:
  Pre-adaptation gradient response contains measurable prospective
  information about future individual A1_TENT_1STEP utility beyond
  ordinary source-state information.

If PASS:
  Next = Q1-S00B Virtual Adaptation Response preregistration.

[OK] Outputs: {args.output_dir}
"""
    summary_path = build_dir / "run_log.txt"
    summary_path.write_text(summary, encoding="utf-8")

    # Cryptographic lock.
    artifact_paths = {
        "protocol_copy": protocol_copy,
        "dataset_audit": dataset_audit_path,
        "fold_assignment": fold_path,
        "feature_manifest": feature_manifest_path,
        "gradient_probe_features": feature_table,
        "oof_predictions": oof_path,
        "oof_metrics": oof_metrics_path,
        "comparison": comparison_path,
        "integrity_audit": integrity_path,
        "decision": decision_path,
        "run_log": summary_path,
    }
    lock = {
        "script_version": VERSION,
        "build": BUILD,
        "artifacts": {
            name: {
                "filename": path.name,
                "sha256": file_sha256(path),
            }
            for name, path in artifact_paths.items()
        },
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "s07b_lock_sha256": EXPECTED_S07B_LOCK_SHA256,
        "utility_table_sha256": EXPECTED_UTILITY_TABLE_SHA256,
        "target_data_used": False,
        "source_masks_opened_during_feature_extraction": False,
        "adapted_prediction_generated": False,
        "optimizer_step_called_during_probe": False,
        "hyperparameter_sweep": False,
        "threshold_tuning": False,
        "decision": decision,
    }
    lock_path = build_dir / "Q1_S00A_IAE_FEASIBILITY_LOCK.json"
    write_json(lock_path, lock)
    lock_sha = file_sha256(lock_path)

    # Final serialization integrity.
    for name, path in artifact_paths.items():
        expected = lock["artifacts"][name]["sha256"]
        actual = file_sha256(path)
        if actual != expected:
            raise RuntimeError(
                f"Artifact changed before commit: {name}"
            )

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    build_dir.rename(args.output_dir)

    print()
    print((args.output_dir / "run_log.txt").read_text(encoding="utf-8"))
    print(
        f"Q1-S00A LOCK: "
        f"{args.output_dir / 'Q1_S00A_IAE_FEASIBILITY_LOCK.json'}"
    )
    print(f"Q1-S00A LOCK SHA256: {lock_sha}")


# ---------------------------------------------------------------------
# Self-test: no project files required
# ---------------------------------------------------------------------

class _ToyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(4)
        self.conv2 = nn.Conv2d(4, 1, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(1)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = torch.relu(x)
        x = self.bn2(self.conv2(x))
        return x


def _toy_configure_tent(model):
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
    return params


def self_test():
    assert EXPECTED_A1_ROWS == 4350
    assert SOURCE_GROUPS == 145
    assert N_FOLDS == 5
    assert A1_ACTION == "A1_TENT_1STEP"
    assert HARM_THRESHOLD == -0.02
    assert BENEFIT_THRESHOLD == +0.02
    assert len(PERTURBATIONS) == 10
    assert sum(EXPECTED_FOLD_GROUP_COUNTS.values()) == SOURCE_GROUPS
    assert sum(EXPECTED_CLASS_COUNTS.values()) == EXPECTED_A1_ROWS

    # Outcome boundaries.
    assert class_from_delta(-0.0200001) == "HARM"
    assert class_from_delta(-0.02) == "HARM"
    assert class_from_delta(0.0) == "NEUTRAL"
    assert class_from_delta(0.019999) == "NEUTRAL"
    assert class_from_delta(0.02) == "BENEFIT"

    # Source feature test.
    z = np.linspace(-2.0, 2.0, 16 * 16).reshape(16, 16)
    src = extract_source_features(z, tent_loss=0.42)
    assert list(src.keys()) == SOURCE_FEATURE_NAMES
    assert all(np.isfinite(list(src.values())))

    # No-update gradient probe test.
    torch.manual_seed(7)
    model = _ToyNet()
    params = _toy_configure_tent(model)
    source_values = [p.detach().clone() for p in params]
    names = trainable_param_names(model, params)
    assert len(names) == len(params) == 4

    model.train()
    x = torch.randn(1, 3, 16, 16)
    z_pre = model(x)
    p = torch.sigmoid(z_pre)
    loss = (
        torch.nn.functional.softplus(z_pre)
        - p * z_pre
    ).mean()
    loss.backward()

    assert_params_equal(params, source_values)
    gfeat = extract_gradient_features(params, names)
    expected_names = gradient_feature_names(names)
    assert set(gfeat) == set(expected_names)
    assert all(np.isfinite(list(gfeat.values())))

    # Decision boundary test.
    metrics = {
        "B0_PRE_SOURCE": {
            "delta_dice_spearman": 0.10,
            "delta_dice_mae": 0.1,
            "harm_auroc": 0.70,
            "harm_auprc": 0.5,
            "benefit_auroc": 0.60,
            "benefit_auprc": 0.3,
        },
        "B1_GRADIENT_PROBE": {
            "delta_dice_spearman": 0.31,
            "delta_dice_mae": 0.08,
            "harm_auroc": 0.76,
            "harm_auprc": 0.6,
            "benefit_auroc": 0.66,
            "benefit_auprc": 0.4,
        },
        "B2_SOURCE_PLUS_GRADIENT": {
            "delta_dice_spearman": 0.35,
            "delta_dice_mae": 0.07,
            "harm_auroc": 0.80,
            "harm_auprc": 0.65,
            "benefit_auroc": 0.66,
            "benefit_auprc": 0.42,
        },
    }
    decision, criteria = decide(metrics)
    assert decision == PASS_DECISION
    assert all(v["pass"] for v in criteria.values())

    metrics["B0_PRE_SOURCE"]["benefit_auroc"] = 0.64
    decision, _ = decide(metrics)
    assert decision == STOP_NO_INCREMENT

    # Feature-token leakage guard.
    validate_feature_names(SOURCE_FEATURE_NAMES)
    validate_feature_names(expected_names)

    print("FROZEN_COUNT_TEST_PASS")
    print("OUTCOME_BOUNDARY_TEST_PASS")
    print("SOURCE_FEATURE_TEST_PASS")
    print("NO_UPDATE_PARAMETER_INTEGRITY_TEST_PASS")
    print("GRADIENT_FEATURE_SCHEMA_TEST_PASS")
    print("GO_STOP_BOUNDARY_TEST_PASS")
    print("FORBIDDEN_FEATURE_TOKEN_TEST_PASS")
    print("SELF_TEST_PASS")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run Q1-S00A pre-adaptation IAE gradient-probe "
            "source-grouped OOF feasibility audit."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU execution.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume only from validated per-seed feature caches in the "
            "Q1-S00A __building directory after a technical interruption."
        ),
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run synthetic integrity tests without project files.",
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
