#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Q1-S03A — Prospective Segmentation Quality Proxy Feasibility

Question
--------
Can true pre-adaptation source segmentation quality be estimated from the
frozen deployment-visible B0 representation, including for an unseen
checkpoint regime, and does that predicted quality retain the utility-
relevant signal identified in Q1-S02C?

No new segmentation inference is performed.
No target-domain data or target GT are used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


VERSION = "2026-08-19-Q1-S03A-v1"
BUILD = "Q1_S03A_PROSPECTIVE_SEGMENTATION_QUALITY_PROXY"

ROOT = Path(r"F:\MEDSEG_SAFETTA")

PROTOCOL = (
    ROOT
    / "docs"
    / "Q1_S03A_prospective_segmentation_quality_proxy_preregistered_protocol_v1.md"
)
EXPECTED_PROTOCOL_SHA256 = (
    "8ca5fe02e601c839dd7d6cb8e54398d36b2b289f46ac1061e43beb6ff6955e99"
)

Q1_S02C_DIR = (
    ROOT
    / "outputs"
    / "Q1_S02C_checkpoint_adaptation_regime_audit_v1"
)
Q1_S02C_LOCK = Q1_S02C_DIR / "Q1_S02C_CHECKPOINT_REGIME_LOCK.json"
EXPECTED_Q1_S02C_LOCK_SHA256 = (
    "37f4bd24367b15c14a75d7962889f169b83c94e6c20ec67b83414381975aee62"
)
EXPECTED_Q1_S02C_DECISION = "STOP_REGIME_VISIBLE_BUT_UTILITY_UNEXPLAINED"

Q1_S00A_DIR = (
    ROOT
    / "outputs"
    / "Q1_S00A_iae_pre_adaptation_gradient_probe_v1"
)
Q1_S00A_LOCK = Q1_S00A_DIR / "Q1_S00A_IAE_FEASIBILITY_LOCK.json"
EXPECTED_Q1_S00A_LOCK_SHA256 = (
    "ed0d82f26f717975a7a1cf9d8bcf9e866a394087b38efdb9903858d4bc020a85"
)
Q1_S00A_FEATURES = Q1_S00A_DIR / "gradient_probe_features.csv"

S07B_DIR = (
    ROOT
    / "outputs"
    / "S07_B_pranet_source_side_counterfactual_utility_dataset_v1"
)
UTILITY_TABLE = S07B_DIR / "source_side_counterfactual_utility_table.csv"
EXPECTED_UTILITY_TABLE_SHA256 = (
    "f8d36a10a41db8255847526312b8b2794d3d18c81a8d7b152a210b3c07bd0678"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "Q1_S03A_prospective_segmentation_quality_proxy_v1"
)

SOURCE_GROUPS = 145
MATCHED_UNITS = 1450
EXPECTED_ROWS = 4350
N_FOLDS = 5

SEEDS = (20260817, 20260818, 20260819)
BASE_SEEDS = (20260817, 20260818)
HELDOUT_REGIME_SEED = 20260819

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

A1_ACTION = "A1_TENT_1STEP"

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

RIDGE_ALPHA = 1.0

TH_GLOBAL = 0.50
TH_WITHIN_MEDIAN = 0.50
TH_WITHIN_EACH = 0.45
TH_TWO_TO_ONE_MEDIAN = 0.45
TH_HELDOUT19 = 0.40
TH_QSHIFT_FIDELITY = 0.35
TH_UTILITY_BRIDGE_ABS = 0.20
TH_NESTED_RESIDUAL = 0.20

ORACLE_REFERENCE_SPEARMAN = 0.392020
ORACLE_TOL = 1e-6

INVALID_ORACLE = "INVALID_Q1_S03A_ORACLE_REPRODUCTION_FAILURE"
STOP_NOT_PREDICTABLE = "STOP_QUALITY_PROXY_NOT_PREDICTABLE"
STOP_CHECKPOINT = "STOP_QUALITY_PROXY_CHECKPOINT_GENERALIZATION_FAILURE"
STOP_BRIDGE = "STOP_QUALITY_PROXY_UTILITY_BRIDGE_FAILURE"
GO_DECISION = "GO_PROSPECTIVE_QUALITY_MEDIATOR_FEASIBLE"


# ---------------------------------------------------------------------
# Utilities
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


def write_csv(path: Path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
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
        raise RuntimeError(f"Non-finite value: {v}")
    return x


def safe_spearman(x, y) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2 or np.std(x) < 1e-15 or np.std(y) < 1e-15:
        return 0.0
    r = spearmanr(x, y).statistic
    if r is None or not np.isfinite(r):
        return 0.0
    return float(r)


def safe_pearson(x, y) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2 or np.std(x) < 1e-15 or np.std(y) < 1e-15:
        return 0.0
    r = pearsonr(x, y).statistic
    if r is None or not np.isfinite(r):
        return 0.0
    return float(r)


def make_ridge():
    return Pipeline([
        ("scale", StandardScaler()),
        ("reg", Ridge(alpha=RIDGE_ALPHA)),
    ])


def feature_matrix(rows, feature_names=SOURCE_FEATURE_NAMES):
    x = np.asarray(
        [[safe_float(r[f]) for f in feature_names] for r in rows],
        dtype=np.float64,
    )
    if x.ndim != 2 or x.shape[1] != len(feature_names):
        raise RuntimeError("Feature-matrix shape mismatch.")
    if not np.isfinite(x).all():
        raise RuntimeError("Feature matrix contains NaN/Inf.")
    return x


def regression_metrics(y, pred):
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    if not np.isfinite(y).all() or not np.isfinite(pred).all():
        raise RuntimeError("Non-finite regression values.")
    return {
        "spearman": safe_spearman(y, pred),
        "pearson": safe_pearson(y, pred),
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(math.sqrt(mean_squared_error(y, pred))),
    }


# ---------------------------------------------------------------------
# Frozen asset loading
# ---------------------------------------------------------------------

def load_rows():
    protocol_sha = validate_sha(
        PROTOCOL,
        EXPECTED_PROTOCOL_SHA256,
        "Q1-S03A protocol",
    )
    q1_s02c_sha = validate_sha(
        Q1_S02C_LOCK,
        EXPECTED_Q1_S02C_LOCK_SHA256,
        "Q1-S02C lock",
    )
    q1_s00a_sha = validate_sha(
        Q1_S00A_LOCK,
        EXPECTED_Q1_S00A_LOCK_SHA256,
        "Q1-S00A lock",
    )
    utility_sha = validate_sha(
        UTILITY_TABLE,
        EXPECTED_UTILITY_TABLE_SHA256,
        "S07-B utility table",
    )

    q1_s02c_lock = json.loads(Q1_S02C_LOCK.read_text(encoding="utf-8"))
    if q1_s02c_lock.get("decision") != EXPECTED_Q1_S02C_DECISION:
        raise RuntimeError(
            f"Unexpected Q1-S02C decision: {q1_s02c_lock.get('decision')}"
        )

    q1_s00a_lock = json.loads(Q1_S00A_LOCK.read_text(encoding="utf-8"))
    feature_art = q1_s00a_lock.get("artifacts", {}).get("gradient_probe_features")
    if not feature_art or "sha256" not in feature_art:
        raise RuntimeError("Q1-S00A lock missing feature-table SHA.")
    feature_sha = validate_sha(
        Q1_S00A_FEATURES,
        feature_art["sha256"],
        "Q1-S00A feature table",
    )

    feature_rows, feature_fields = read_csv(Q1_S00A_FEATURES)
    utility_rows, utility_fields = read_csv(UTILITY_TABLE)

    required_feature = {
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
    } | set(SOURCE_FEATURE_NAMES)
    missing = sorted(required_feature - set(feature_fields))
    if missing:
        raise RuntimeError(f"Feature table missing columns: {missing}")

    required_utility = {
        "seed",
        "sample_id",
        "perturbation",
        "action",
        "source_dice",
        "action_dice",
        "delta_dice",
    }
    missing_u = sorted(required_utility - set(utility_fields))
    if missing_u:
        raise RuntimeError(f"Utility table missing columns: {missing_u}")

    if len(feature_rows) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Feature row count changed: {len(feature_rows)} != {EXPECTED_ROWS}"
        )

    a1 = [r for r in utility_rows if r["action"] == A1_ACTION]
    if len(a1) != EXPECTED_ROWS:
        raise RuntimeError(
            f"A1 row count changed: {len(a1)} != {EXPECTED_ROWS}"
        )

    utility_map = {}
    for r in a1:
        key = (int(r["seed"]), r["sample_id"], r["perturbation"])
        if key in utility_map:
            raise RuntimeError(f"Duplicate utility key: {key}")
        utility_map[key] = r

    merged = []
    group_folds = defaultdict(set)
    seen = set()

    for r in feature_rows:
        key = (int(r["seed"]), r["sample_id"], r["perturbation"])
        if key in seen:
            raise RuntimeError(f"Duplicate feature key: {key}")
        seen.add(key)

        if key not in utility_map:
            raise RuntimeError(f"Missing utility match: {key}")
        u = utility_map[key]

        seed = int(r["seed"])
        fold = int(r["fold"])
        pert = r["perturbation"]

        if seed not in SEEDS:
            raise RuntimeError(f"Unexpected seed: {seed}")
        if fold not in range(N_FOLDS):
            raise RuntimeError(f"Unexpected fold: {fold}")
        if pert not in PERTURBATIONS:
            raise RuntimeError(f"Unexpected perturbation: {pert}")
        if abs(safe_float(r["delta_dice"]) - safe_float(u["delta_dice"])) > 1e-12:
            raise RuntimeError(f"DeltaDice mismatch: {key}")

        group_folds[r["source_group_id"]].add(fold)

        row = {
            "seed": seed,
            "sample_id": r["sample_id"],
            "source_group_id": r["source_group_id"],
            "fold": fold,
            "perturbation": pert,
            "base_case_id": r["base_case_id"],
            "source_dice": safe_float(u["source_dice"]),
            "action_dice": safe_float(u["action_dice"]),
            "delta_dice": safe_float(r["delta_dice"]),
            "harmful": int(r["harmful"]),
            "neutral": int(r["neutral"]),
            "beneficial": int(r["beneficial"]),
            "outcome_class": r["outcome_class"],
        }
        for f in SOURCE_FEATURE_NAMES:
            row[f] = safe_float(r[f])
        merged.append(row)

    if len(group_folds) != SOURCE_GROUPS:
        raise RuntimeError(
            f"Source-group count changed: {len(group_folds)} != {SOURCE_GROUPS}"
        )
    if any(len(v) != 1 for v in group_folds.values()):
        raise RuntimeError("Frozen group-fold assignment inconsistent.")

    matched = defaultdict(list)
    for r in merged:
        matched[(r["sample_id"], r["perturbation"])].append(r)

    if len(matched) != MATCHED_UNITS:
        raise RuntimeError(
            f"Matched-unit count changed: {len(matched)} != {MATCHED_UNITS}"
        )

    for unit, rr in matched.items():
        if len(rr) != 3 or {int(r["seed"]) for r in rr} != set(SEEDS):
            raise RuntimeError(f"Invalid matched unit: {unit}")
        if len({r["source_group_id"] for r in rr}) != 1:
            raise RuntimeError(f"Matched-unit group mismatch: {unit}")
        if len({int(r["fold"]) for r in rr}) != 1:
            raise RuntimeError(f"Matched-unit fold mismatch: {unit}")

    provenance = {
        "script_version": VERSION,
        "build": BUILD,
        "protocol_sha256": protocol_sha,
        "q1_s02c_lock_sha256": q1_s02c_sha,
        "q1_s02c_decision": q1_s02c_lock["decision"],
        "q1_s00a_lock_sha256": q1_s00a_sha,
        "q1_s00a_feature_sha256": feature_sha,
        "utility_table_sha256": utility_sha,
        "rows": len(merged),
        "source_groups": len(group_folds),
        "matched_units": len(matched),
        "seeds": list(SEEDS),
        "perturbations": list(PERTURBATIONS),
        "action": A1_ACTION,
        "new_segmentation_inference": False,
        "target_data_used": False,
        "target_gt_used": False,
        "gradient_features_used": False,
        "adapted_prediction_features_used": False,
        "source_dice_used_as_input": False,
        "hyperparameter_sweep": False,
        "feature_selection": False,
        "threshold_tuning": False,
    }

    return merged, matched, provenance


# ---------------------------------------------------------------------
# Generic quality prediction splits
# ---------------------------------------------------------------------

def fit_predict_quality(train_rows, test_rows):
    x_train = feature_matrix(train_rows)
    y_train = np.asarray([r["source_dice"] for r in train_rows], dtype=np.float64)
    x_test = feature_matrix(test_rows)

    model = make_ridge()
    model.fit(x_train, y_train)
    return model.predict(x_test)


def grouped_oof_quality(rows):
    pred = [None] * len(rows)
    split_audit = []

    for fold in range(N_FOLDS):
        train_idx = [i for i, r in enumerate(rows) if int(r["fold"]) != fold]
        test_idx = [i for i, r in enumerate(rows) if int(r["fold"]) == fold]

        train_groups = {rows[i]["source_group_id"] for i in train_idx}
        test_groups = {rows[i]["source_group_id"] for i in test_idx}
        if train_groups & test_groups:
            raise RuntimeError(f"Global quality OOF leakage fold={fold}")

        p = fit_predict_quality(
            [rows[i] for i in train_idx],
            [rows[i] for i in test_idx],
        )
        for i, v in zip(test_idx, p):
            pred[i] = float(v)

        split_audit.append({
            "fold": fold,
            "train_rows": len(train_idx),
            "test_rows": len(test_idx),
            "train_groups": len(train_groups),
            "test_groups": len(test_groups),
            "group_overlap": 0,
        })

    if any(v is None for v in pred):
        raise RuntimeError("Incomplete global quality OOF predictions.")

    y = [r["source_dice"] for r in rows]
    metrics = regression_metrics(y, pred)
    metrics["split_audit"] = split_audit
    return np.asarray(pred, dtype=np.float64), metrics


def within_seed_quality(rows, seed):
    rr = [r for r in rows if int(r["seed"]) == seed]
    pred, metrics = grouped_oof_quality(rr)
    return rr, pred, metrics


def cross_seed_quality(rows, train_seed, test_seed):
    pred_rows = []
    split_audit = []

    for fold in range(N_FOLDS):
        train = [
            r for r in rows
            if int(r["seed"]) == train_seed and int(r["fold"]) != fold
        ]
        test = [
            r for r in rows
            if int(r["seed"]) == test_seed and int(r["fold"]) == fold
        ]

        train_groups = {r["source_group_id"] for r in train}
        test_groups = {r["source_group_id"] for r in test}
        if train_groups & test_groups:
            raise RuntimeError(
                f"Cross-seed quality leakage {train_seed}->{test_seed} fold={fold}"
            )

        p = fit_predict_quality(train, test)
        for r, v in zip(test, p):
            pred_rows.append({**r, "pred_source_dice": float(v)})

        split_audit.append({
            "train_seed": train_seed,
            "test_seed": test_seed,
            "fold": fold,
            "group_overlap": 0,
        })

    expected_test_rows = sum(
        1 for r in rows if int(r["seed"]) == test_seed
    )
    if len(pred_rows) != expected_test_rows:
        raise RuntimeError(
            f"Cross-seed prediction count mismatch {train_seed}->{test_seed}: "
            f"{len(pred_rows)} != {expected_test_rows}"
        )

    metrics = regression_metrics(
        [r["source_dice"] for r in pred_rows],
        [r["pred_source_dice"] for r in pred_rows],
    )
    metrics["split_audit"] = split_audit
    return pred_rows, metrics


def two_to_one_quality(rows, heldout_seed):
    train_seeds = [s for s in SEEDS if s != heldout_seed]
    pred_rows = []
    split_audit = []

    for fold in range(N_FOLDS):
        train = [
            r for r in rows
            if int(r["seed"]) in train_seeds and int(r["fold"]) != fold
        ]
        test = [
            r for r in rows
            if int(r["seed"]) == heldout_seed and int(r["fold"]) == fold
        ]

        train_groups = {r["source_group_id"] for r in train}
        test_groups = {r["source_group_id"] for r in test}
        if train_groups & test_groups:
            raise RuntimeError(
                f"2->1 quality leakage heldout={heldout_seed} fold={fold}"
            )

        p = fit_predict_quality(train, test)
        for r, v in zip(test, p):
            pred_rows.append({**r, "pred_source_dice": float(v)})

        split_audit.append({
            "heldout_seed": heldout_seed,
            "train_seeds": train_seeds,
            "fold": fold,
            "group_overlap": 0,
        })

    expected_test_rows = sum(
        1 for r in rows if int(r["seed"]) == heldout_seed
    )
    if len(pred_rows) != expected_test_rows:
        raise RuntimeError(
            f"2->1 prediction count mismatch heldout={heldout_seed}: "
            f"{len(pred_rows)} != {expected_test_rows}"
        )

    metrics = regression_metrics(
        [r["source_dice"] for r in pred_rows],
        [r["pred_source_dice"] for r in pred_rows],
    )
    metrics["split_audit"] = split_audit
    return pred_rows, metrics


# ---------------------------------------------------------------------
# Base-regime OOF predictions for all three seeds
# ---------------------------------------------------------------------

def base_regime_quality_predictions(rows):
    pred_map = {}
    output = []
    split_audit = []

    for fold in range(N_FOLDS):
        train = [
            r for r in rows
            if int(r["seed"]) in BASE_SEEDS and int(r["fold"]) != fold
        ]
        test = [r for r in rows if int(r["fold"]) == fold]

        train_groups = {r["source_group_id"] for r in train}
        test_groups = {r["source_group_id"] for r in test}
        if train_groups & test_groups:
            raise RuntimeError(f"Base-regime quality leakage fold={fold}")

        x_train = feature_matrix(train)
        y_train = np.asarray([r["source_dice"] for r in train], dtype=np.float64)
        model = make_ridge()
        model.fit(x_train, y_train)

        p = model.predict(feature_matrix(test))
        for r, v in zip(test, p):
            key = (r["sample_id"], r["perturbation"], int(r["seed"]))
            if key in pred_map:
                raise RuntimeError(f"Duplicate base-regime prediction key: {key}")
            pred_map[key] = float(v)
            output.append({
                "sample_id": r["sample_id"],
                "perturbation": r["perturbation"],
                "source_group_id": r["source_group_id"],
                "fold": r["fold"],
                "seed": r["seed"],
                "true_source_dice": r["source_dice"],
                "pred_source_dice": float(v),
            })

        split_audit.append({
            "fold": fold,
            "train_seeds": list(BASE_SEEDS),
            "train_rows": len(train),
            "test_rows": len(test),
            "group_overlap": 0,
        })

    if len(pred_map) != len(rows):
        raise RuntimeError(
            f"Base-regime prediction count mismatch: "
            f"{len(pred_map)} != {len(rows)}"
        )

    return pred_map, output, split_audit


# ---------------------------------------------------------------------
# Matched units and quality-shift bridge
# ---------------------------------------------------------------------

def build_units(matched):
    units = []
    for (sample_id, perturbation), rr in sorted(matched.items()):
        by_seed = {int(r["seed"]): r for r in rr}
        u = {
            "sample_id": sample_id,
            "perturbation": perturbation,
            "source_group_id": by_seed[SEEDS[0]]["source_group_id"],
            "fold": int(by_seed[SEEDS[0]]["fold"]),
        }
        for s in SEEDS:
            u[f"source_dice_{s}"] = by_seed[s]["source_dice"]
            u[f"delta_{s}"] = by_seed[s]["delta_dice"]
        units.append(u)
    return units


def quality_shift_bridge(units, pred_map):
    rows = []

    for u in units:
        sample = u["sample_id"]
        pert = u["perturbation"]

        qhat = {
            s: pred_map[(sample, pert, s)]
            for s in SEEDS
        }
        q = {
            s: u[f"source_dice_{s}"]
            for s in SEEDS
        }
        d = {
            s: u[f"delta_{s}"]
            for s in SEEDS
        }

        pred_shift = qhat[HELDOUT_REGIME_SEED] - (
            qhat[BASE_SEEDS[0]] + qhat[BASE_SEEDS[1]]
        ) / 2.0

        true_shift = q[HELDOUT_REGIME_SEED] - (
            q[BASE_SEEDS[0]] + q[BASE_SEEDS[1]]
        ) / 2.0

        residual = d[HELDOUT_REGIME_SEED] - (
            d[BASE_SEEDS[0]] + d[BASE_SEEDS[1]]
        ) / 2.0

        rows.append({
            "sample_id": sample,
            "perturbation": pert,
            "source_group_id": u["source_group_id"],
            "fold": u["fold"],
            "qhat17": qhat[BASE_SEEDS[0]],
            "qhat18": qhat[BASE_SEEDS[1]],
            "qhat19": qhat[HELDOUT_REGIME_SEED],
            "true_q17": q[BASE_SEEDS[0]],
            "true_q18": q[BASE_SEEDS[1]],
            "true_q19": q[HELDOUT_REGIME_SEED],
            "predicted_quality_shift19": pred_shift,
            "true_quality_shift19": true_shift,
            "seed19_utility_residual": residual,
        })

    pred_shift = np.asarray(
        [r["predicted_quality_shift19"] for r in rows],
        dtype=np.float64,
    )
    true_shift = np.asarray(
        [r["true_quality_shift19"] for r in rows],
        dtype=np.float64,
    )
    residual = np.asarray(
        [r["seed19_utility_residual"] for r in rows],
        dtype=np.float64,
    )

    qshift_rho = safe_spearman(pred_shift, true_shift)
    qshift_mae = float(mean_absolute_error(true_shift, pred_shift))
    oracle_bridge = safe_spearman(true_shift, residual)
    proxy_bridge = safe_spearman(pred_shift, residual)

    sign_match = (
        oracle_bridge == 0.0
        or proxy_bridge == 0.0
        or math.copysign(1.0, oracle_bridge) == math.copysign(1.0, proxy_bridge)
    )

    metrics = {
        "quality_shift_fidelity_spearman": qshift_rho,
        "quality_shift_mae": qshift_mae,
        "quality_shift_threshold": TH_QSHIFT_FIDELITY,
        "quality_shift_pass": qshift_rho >= TH_QSHIFT_FIDELITY,
        "oracle_quality_shift_vs_utility_residual_spearman": oracle_bridge,
        "proxy_quality_shift_vs_utility_residual_spearman": proxy_bridge,
        "proxy_oracle_sign_match": sign_match,
        "utility_bridge_abs_threshold": TH_UTILITY_BRIDGE_ABS,
        "utility_bridge_pass": (
            sign_match and abs(proxy_bridge) >= TH_UTILITY_BRIDGE_ABS
        ),
    }

    return rows, metrics


# ---------------------------------------------------------------------
# Oracle reproduction
# ---------------------------------------------------------------------

def grouped_oof_scalar_to_target(rows, feature_name, target_name):
    pred = [None] * len(rows)
    for fold in range(N_FOLDS):
        train_idx = [i for i, r in enumerate(rows) if int(r["fold"]) != fold]
        test_idx = [i for i, r in enumerate(rows) if int(r["fold"]) == fold]

        train_groups = {rows[i]["source_group_id"] for i in train_idx}
        test_groups = {rows[i]["source_group_id"] for i in test_idx}
        if train_groups & test_groups:
            raise RuntimeError(f"Scalar OOF leakage fold={fold}")

        x_train = np.asarray(
            [[safe_float(rows[i][feature_name])] for i in train_idx],
            dtype=np.float64,
        )
        y_train = np.asarray(
            [safe_float(rows[i][target_name]) for i in train_idx],
            dtype=np.float64,
        )
        x_test = np.asarray(
            [[safe_float(rows[i][feature_name])] for i in test_idx],
            dtype=np.float64,
        )

        model = make_ridge()
        model.fit(x_train, y_train)
        p = model.predict(x_test)

        for i, v in zip(test_idx, p):
            pred[i] = float(v)

    if any(v is None for v in pred):
        raise RuntimeError("Incomplete scalar OOF predictions.")

    y = [r[target_name] for r in rows]
    return np.asarray(pred, dtype=np.float64), regression_metrics(y, pred)


def oracle_reproduction(bridge_rows):
    pred, metrics = grouped_oof_scalar_to_target(
        bridge_rows,
        "true_quality_shift19",
        "seed19_utility_residual",
    )

    diff = abs(metrics["spearman"] - ORACLE_REFERENCE_SPEARMAN)
    return {
        "reference_spearman": ORACLE_REFERENCE_SPEARMAN,
        "actual_spearman": metrics["spearman"],
        "absolute_difference": diff,
        "tolerance": ORACLE_TOL,
        "pass": diff <= ORACLE_TOL,
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
    }


# ---------------------------------------------------------------------
# Strict nested proxy-shift -> residual
# ---------------------------------------------------------------------

def fit_base_quality_model(train_rows):
    train = [r for r in train_rows if int(r["seed"]) in BASE_SEEDS]
    model = make_ridge()
    model.fit(
        feature_matrix(train),
        np.asarray([r["source_dice"] for r in train], dtype=np.float64),
    )
    return model


def predict_unit_qshift(model, unit_rows):
    by_seed = {int(r["seed"]): r for r in unit_rows}
    pred = {}
    for s in SEEDS:
        pred[s] = float(model.predict(feature_matrix([by_seed[s]]))[0])
    return pred[HELDOUT_REGIME_SEED] - (
        pred[BASE_SEEDS[0]] + pred[BASE_SEEDS[1]]
    ) / 2.0


def nested_proxy_residual(rows, matched):
    units_by_fold = defaultdict(list)
    for key, rr in matched.items():
        fold = int(rr[0]["fold"])
        units_by_fold[fold].append((key, rr))

    all_pred_rows = []
    audit = []

    for outer_fold in range(N_FOLDS):
        outer_test_units = units_by_fold[outer_fold]
        outer_train_folds = [f for f in range(N_FOLDS) if f != outer_fold]

        outer_test_groups = {
            rr[0]["source_group_id"]
            for _, rr in outer_test_units
        }

        # Generate inner cross-fitted qshift for every outer-training unit.
        train_qshift = {}
        train_target = {}
        train_meta = {}

        for inner_fold in outer_train_folds:
            inner_train_rows = [
                r for r in rows
                if int(r["seed"]) in BASE_SEEDS
                and int(r["fold"]) not in {outer_fold, inner_fold}
            ]

            inner_train_groups = {r["source_group_id"] for r in inner_train_rows}
            if inner_train_groups & outer_test_groups:
                raise RuntimeError(
                    f"Nested quality outer-test leakage outer={outer_fold} inner={inner_fold}"
                )

            model = fit_base_quality_model(inner_train_rows)

            for key, rr in units_by_fold[inner_fold]:
                qshift = predict_unit_qshift(model, rr)
                by_seed = {int(r["seed"]): r for r in rr}
                residual = by_seed[HELDOUT_REGIME_SEED]["delta_dice"] - (
                    by_seed[BASE_SEEDS[0]]["delta_dice"]
                    + by_seed[BASE_SEEDS[1]]["delta_dice"]
                ) / 2.0
                train_qshift[key] = qshift
                train_target[key] = residual
                train_meta[key] = rr[0]

        expected_train_units = sum(len(units_by_fold[f]) for f in outer_train_folds)
        if len(train_qshift) != expected_train_units:
            raise RuntimeError(
                f"Nested train-unit coverage mismatch outer={outer_fold}: "
                f"{len(train_qshift)} != {expected_train_units}"
            )

        # Fit residual mapping using only outer-training units.
        ordered_keys = sorted(train_qshift.keys())
        x_train = np.asarray(
            [[train_qshift[k]] for k in ordered_keys],
            dtype=np.float64,
        )
        y_train = np.asarray(
            [train_target[k] for k in ordered_keys],
            dtype=np.float64,
        )

        residual_model = make_ridge()
        residual_model.fit(x_train, y_train)

        # Fit outer quality model on all outer-training base-seed rows.
        outer_quality_train = [
            r for r in rows
            if int(r["seed"]) in BASE_SEEDS and int(r["fold"]) != outer_fold
        ]
        outer_quality_groups = {r["source_group_id"] for r in outer_quality_train}
        if outer_quality_groups & outer_test_groups:
            raise RuntimeError(
                f"Outer quality model leakage outer={outer_fold}"
            )
        quality_model = fit_base_quality_model(outer_quality_train)

        test_qshift = []
        test_residual = []
        test_payload = []

        for key, rr in outer_test_units:
            qshift = predict_unit_qshift(quality_model, rr)
            by_seed = {int(r["seed"]): r for r in rr}
            residual = by_seed[HELDOUT_REGIME_SEED]["delta_dice"] - (
                by_seed[BASE_SEEDS[0]]["delta_dice"]
                + by_seed[BASE_SEEDS[1]]["delta_dice"]
            ) / 2.0
            test_qshift.append(qshift)
            test_residual.append(residual)
            test_payload.append((key, rr, qshift, residual))

        p = residual_model.predict(
            np.asarray([[v] for v in test_qshift], dtype=np.float64)
        )

        for (key, rr, qshift, residual), pred in zip(test_payload, p):
            all_pred_rows.append({
                "sample_id": key[0],
                "perturbation": key[1],
                "source_group_id": rr[0]["source_group_id"],
                "fold": outer_fold,
                "nested_predicted_quality_shift19": qshift,
                "true_seed19_utility_residual": residual,
                "pred_seed19_utility_residual": float(pred),
            })

        audit.append({
            "outer_fold": outer_fold,
            "outer_test_groups": len(outer_test_groups),
            "outer_train_units": expected_train_units,
            "outer_test_units": len(outer_test_units),
            "outer_test_group_overlap_in_quality_training": 0,
            "outer_test_group_overlap_in_residual_training": 0,
        })

    if len(all_pred_rows) != len(matched):
        raise RuntimeError(
            f"Nested residual prediction count mismatch: "
            f"{len(all_pred_rows)} != {len(matched)}"
        )

    y = [r["true_seed19_utility_residual"] for r in all_pred_rows]
    p = [r["pred_seed19_utility_residual"] for r in all_pred_rows]
    metrics = regression_metrics(y, p)
    metrics["threshold_spearman"] = TH_NESTED_RESIDUAL
    metrics["pass"] = metrics["spearman"] >= TH_NESTED_RESIDUAL
    metrics["outer_split_audit"] = audit

    return all_pred_rows, metrics


# ---------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------

def choose_decision(
    oracle,
    global_metrics,
    within_metrics,
    two_to_one_metrics,
    bridge_metrics,
    nested_metrics,
):
    within_values = [within_metrics[s]["spearman"] for s in SEEDS]
    within_median = float(np.median(within_values))
    within_each_min = float(np.min(within_values))

    two_values = [two_to_one_metrics[s]["spearman"] for s in SEEDS]
    two_median = float(np.median(two_values))
    heldout19 = float(two_to_one_metrics[HELDOUT_REGIME_SEED]["spearman"])

    ordinary_pass = (
        global_metrics["spearman"] >= TH_GLOBAL
        and within_median >= TH_WITHIN_MEDIAN
        and within_each_min >= TH_WITHIN_EACH
    )

    checkpoint_pass = (
        two_median >= TH_TWO_TO_ONE_MEDIAN
        and heldout19 >= TH_HELDOUT19
        and bridge_metrics["quality_shift_fidelity_spearman"] >= TH_QSHIFT_FIDELITY
    )

    utility_bridge_pass = (
        bridge_metrics["utility_bridge_pass"]
        and nested_metrics["spearman"] >= TH_NESTED_RESIDUAL
    )

    if not oracle["pass"]:
        decision = INVALID_ORACLE
    elif not ordinary_pass:
        decision = STOP_NOT_PREDICTABLE
    elif not checkpoint_pass:
        decision = STOP_CHECKPOINT
    elif not utility_bridge_pass:
        decision = STOP_BRIDGE
    else:
        decision = GO_DECISION

    criteria = {
        "oracle_reproduction": {
            "value": oracle["actual_spearman"],
            "reference": ORACLE_REFERENCE_SPEARMAN,
            "tolerance": ORACLE_TOL,
            "pass": oracle["pass"],
        },
        "global_quality_spearman": {
            "value": global_metrics["spearman"],
            "threshold": TH_GLOBAL,
            "pass": global_metrics["spearman"] >= TH_GLOBAL,
        },
        "median_within_seed_quality_spearman": {
            "value": within_median,
            "threshold": TH_WITHIN_MEDIAN,
            "pass": within_median >= TH_WITHIN_MEDIAN,
        },
        "minimum_within_seed_quality_spearman": {
            "value": within_each_min,
            "threshold": TH_WITHIN_EACH,
            "pass": within_each_min >= TH_WITHIN_EACH,
        },
        "median_two_to_one_quality_spearman": {
            "value": two_median,
            "threshold": TH_TWO_TO_ONE_MEDIAN,
            "pass": two_median >= TH_TWO_TO_ONE_MEDIAN,
        },
        "heldout19_quality_spearman": {
            "value": heldout19,
            "threshold": TH_HELDOUT19,
            "pass": heldout19 >= TH_HELDOUT19,
        },
        "quality_shift_fidelity_spearman": {
            "value": bridge_metrics["quality_shift_fidelity_spearman"],
            "threshold": TH_QSHIFT_FIDELITY,
            "pass": bridge_metrics["quality_shift_pass"],
        },
        "proxy_utility_bridge_abs_spearman": {
            "value": abs(
                bridge_metrics[
                    "proxy_quality_shift_vs_utility_residual_spearman"
                ]
            ),
            "threshold": TH_UTILITY_BRIDGE_ABS,
            "sign_match": bridge_metrics["proxy_oracle_sign_match"],
            "pass": bridge_metrics["utility_bridge_pass"],
        },
        "nested_proxy_residual_spearman": {
            "value": nested_metrics["spearman"],
            "threshold": TH_NESTED_RESIDUAL,
            "pass": nested_metrics["spearman"] >= TH_NESTED_RESIDUAL,
        },
    }

    return decision, criteria


# ---------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------

def synthetic_rows():
    rng = np.random.default_rng(20260819)
    rows = []

    for g in range(30):
        fold = g % N_FOLDS
        group_latent = rng.normal()
        for p_i, pert in enumerate(PERTURBATIONS):
            case = group_latent + 0.15 * p_i + rng.normal(scale=0.25)
            for s_i, seed in enumerate(SEEDS):
                model_shift = 0.15 * s_i
                f1 = case + model_shift + rng.normal(scale=0.2)
                f2 = rng.normal() + 0.3 * case
                quality = np.clip(
                    0.72
                    + 0.10 * np.tanh(f1)
                    - 0.05 * abs(f2)
                    - 0.025 * s_i
                    + rng.normal(scale=0.025),
                    0.0,
                    1.0,
                )
                delta = (
                    -0.25 * (quality - 0.75)
                    + 0.02 * case
                    + rng.normal(scale=0.025)
                )

                row = {
                    "seed": seed,
                    "sample_id": f"s{g:03d}",
                    "source_group_id": f"g{g:03d}",
                    "fold": fold,
                    "perturbation": pert,
                    "source_dice": float(quality),
                    "action_dice": float(np.clip(quality + delta, 0, 1)),
                    "delta_dice": float(delta),
                }

                for j, f in enumerate(SOURCE_FEATURE_NAMES):
                    if j == 0:
                        val = f1
                    elif j == 1:
                        val = f2
                    else:
                        val = (
                            0.65 * f1
                            + 0.15 * f2
                            + rng.normal(scale=0.35 + 0.01 * j)
                        )
                    row[f] = float(val)
                rows.append(row)
    return rows


def synthetic_matched(rows):
    matched = defaultdict(list)
    for r in rows:
        matched[(r["sample_id"], r["perturbation"])].append(r)
    return matched


def self_test():
    assert len(SOURCE_FEATURE_NAMES) == 19
    assert len(SEEDS) == 3
    assert len(PERTURBATIONS) == 10
    assert N_FOLDS == 5
    assert TH_GLOBAL == 0.50
    assert TH_WITHIN_MEDIAN == 0.50
    assert TH_WITHIN_EACH == 0.45
    assert TH_TWO_TO_ONE_MEDIAN == 0.45
    assert TH_HELDOUT19 == 0.40
    assert TH_QSHIFT_FIDELITY == 0.35
    assert TH_UTILITY_BRIDGE_ABS == 0.20
    assert TH_NESTED_RESIDUAL == 0.20
    assert ORACLE_REFERENCE_SPEARMAN == 0.392020

    rows = synthetic_rows()
    matched = synthetic_matched(rows)

    p, gm = grouped_oof_quality(rows)
    assert len(p) == len(rows)
    assert math.isfinite(gm["spearman"])

    for seed in SEEDS:
        rr, pp, mm = within_seed_quality(rows, seed)
        assert len(rr) == 30 * len(PERTURBATIONS)
        assert len(pp) == len(rr)
        assert math.isfinite(mm["spearman"])

    _, cm = cross_seed_quality(rows, SEEDS[0], SEEDS[1])
    assert math.isfinite(cm["spearman"])

    _, tm = two_to_one_quality(rows, HELDOUT_REGIME_SEED)
    assert math.isfinite(tm["spearman"])

    pred_map, out_rows, _ = base_regime_quality_predictions(rows)
    assert len(pred_map) == len(rows)
    assert len(out_rows) == len(rows)

    units = build_units(matched)
    bridge_rows, bridge_metrics = quality_shift_bridge(units, pred_map)
    assert len(bridge_rows) == len(units)
    assert math.isfinite(bridge_metrics["quality_shift_fidelity_spearman"])

    nested_rows, nested_metrics = nested_proxy_residual(rows, matched)
    assert len(nested_rows) == len(units)
    assert math.isfinite(nested_metrics["spearman"])

    # Implementation-only oracle smoke check; not against production reference.
    _, oracle_smoke = grouped_oof_scalar_to_target(
        bridge_rows,
        "true_quality_shift19",
        "seed19_utility_residual",
    )
    assert math.isfinite(oracle_smoke["spearman"])

    print("FROZEN_CONSTANTS_TEST_PASS")
    print("QUALITY_OOF_TEST_PASS")
    print("CROSS_SEED_TEST_PASS")
    print("BASE_REGIME_BRIDGE_TEST_PASS")
    print("NESTED_BRIDGE_TEST_PASS")
    print("SELF_TEST_PASS")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def run(args):
    if args.output_dir.exists():
        raise FileExistsError(
            f"Q1-S03A output already exists: {args.output_dir}"
        )

    build_dir = Path(str(args.output_dir) + "__building")
    if build_dir.exists():
        raise FileExistsError(
            f"Partial Q1-S03A build exists: {build_dir}. "
            "Remove only this __building directory if a prior technical run failed."
        )
    build_dir.mkdir(parents=True, exist_ok=False)

    print("===== Q1-S03A PROSPECTIVE SEGMENTATION QUALITY PROXY =====")
    print(f"Script version: {VERSION}")
    print(f"Build: {BUILD}")
    print("New segmentation inference: NO")
    print("Target data used: NO")
    print("Target GT used: NO")
    print("source_dice used as deployment input: NO")
    print("Gradient features used: NO")
    print("Adapted prediction features used: NO")
    print("Hyperparameter sweep: NO")
    print("Feature selection: NO")
    print("Threshold tuning: NO")
    print()

    rows, matched, provenance = load_rows()
    units = build_units(matched)

    protocol_copy = build_dir / "preregistered_protocol_copy.md"
    shutil.copy2(PROTOCOL, protocol_copy)
    if file_sha256(protocol_copy) != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("Protocol copy SHA mismatch.")

    provenance_path = build_dir / "provenance_audit.json"
    write_json(provenance_path, provenance)

    # A — global OOF
    global_pred, global_metrics = grouped_oof_quality(rows)
    global_pred_rows = []
    for r, p in zip(rows, global_pred):
        global_pred_rows.append({
            "seed": r["seed"],
            "sample_id": r["sample_id"],
            "perturbation": r["perturbation"],
            "source_group_id": r["source_group_id"],
            "fold": r["fold"],
            "true_source_dice": r["source_dice"],
            "pred_source_dice": float(p),
        })

    global_pred_path = build_dir / "global_quality_oof_predictions.csv"
    write_csv(
        global_pred_path,
        global_pred_rows,
        [
            "seed",
            "sample_id",
            "perturbation",
            "source_group_id",
            "fold",
            "true_source_dice",
            "pred_source_dice",
        ],
    )
    global_metrics_path = build_dir / "global_quality_metrics.json"
    write_json(global_metrics_path, global_metrics)

    # B — within seed
    within_metrics = {}
    within_rows = []
    within_predictions = {}

    for seed in SEEDS:
        rr, pp, mm = within_seed_quality(rows, seed)
        within_predictions[seed] = (rr, pp)
        within_metrics[seed] = mm
        within_rows.append({
            "seed": seed,
            "spearman": mm["spearman"],
            "pearson": mm["pearson"],
            "mae": mm["mae"],
            "rmse": mm["rmse"],
        })

    within_path = build_dir / "within_seed_quality_metrics.csv"
    write_csv(
        within_path,
        within_rows,
        ["seed", "spearman", "pearson", "mae", "rmse"],
    )

    # C — 3x3 transfer matrices
    cross_metrics = {}
    for train_seed in SEEDS:
        for test_seed in SEEDS:
            if train_seed == test_seed:
                cross_metrics[(train_seed, test_seed)] = within_metrics[train_seed]
            else:
                _, mm = cross_seed_quality(rows, train_seed, test_seed)
                cross_metrics[(train_seed, test_seed)] = mm

    matrix_fields = ["train_seed"] + [f"test_{s}" for s in SEEDS]

    spearman_matrix_rows = []
    mae_matrix_rows = []
    for train_seed in SEEDS:
        rs = {"train_seed": train_seed}
        rm = {"train_seed": train_seed}
        for test_seed in SEEDS:
            rs[f"test_{test_seed}"] = cross_metrics[
                (train_seed, test_seed)
            ]["spearman"]
            rm[f"test_{test_seed}"] = cross_metrics[
                (train_seed, test_seed)
            ]["mae"]
        spearman_matrix_rows.append(rs)
        mae_matrix_rows.append(rm)

    cross_spear_path = build_dir / "cross_seed_quality_matrix_spearman.csv"
    cross_mae_path = build_dir / "cross_seed_quality_matrix_mae.csv"
    write_csv(cross_spear_path, spearman_matrix_rows, matrix_fields)
    write_csv(cross_mae_path, mae_matrix_rows, matrix_fields)

    # D — 2->1
    two_metrics = {}
    two_rows = []
    for seed in SEEDS:
        _, mm = two_to_one_quality(rows, seed)
        two_metrics[seed] = mm
        two_rows.append({
            "heldout_seed": seed,
            "train_seeds": "+".join(str(s) for s in SEEDS if s != seed),
            "spearman": mm["spearman"],
            "pearson": mm["pearson"],
            "mae": mm["mae"],
            "rmse": mm["rmse"],
        })

    two_path = build_dir / "two_seed_to_one_quality_metrics.csv"
    write_csv(
        two_path,
        two_rows,
        ["heldout_seed", "train_seeds", "spearman", "pearson", "mae", "rmse"],
    )

    # E/F/G
    base_pred_map, base_pred_rows, base_split_audit = (
        base_regime_quality_predictions(rows)
    )
    base_pred_path = build_dir / "base_regime_quality_predictions.csv"
    write_csv(
        base_pred_path,
        base_pred_rows,
        [
            "sample_id",
            "perturbation",
            "source_group_id",
            "fold",
            "seed",
            "true_source_dice",
            "pred_source_dice",
        ],
    )

    bridge_rows, bridge_metrics = quality_shift_bridge(
        units,
        base_pred_map,
    )
    bridge_path = build_dir / "quality_shift_bridge.csv"
    write_csv(
        bridge_path,
        bridge_rows,
        [
            "sample_id",
            "perturbation",
            "source_group_id",
            "fold",
            "qhat17",
            "qhat18",
            "qhat19",
            "true_q17",
            "true_q18",
            "true_q19",
            "predicted_quality_shift19",
            "true_quality_shift19",
            "seed19_utility_residual",
        ],
    )
    bridge_metrics["base_regime_split_audit"] = base_split_audit
    bridge_metrics_path = build_dir / "quality_shift_bridge_metrics.json"
    write_json(bridge_metrics_path, bridge_metrics)

    # H — nested bridge
    nested_rows, nested_metrics = nested_proxy_residual(
        rows,
        matched,
    )
    nested_path = build_dir / "nested_proxy_residual_predictions.csv"
    write_csv(
        nested_path,
        nested_rows,
        [
            "sample_id",
            "perturbation",
            "source_group_id",
            "fold",
            "nested_predicted_quality_shift19",
            "true_seed19_utility_residual",
            "pred_seed19_utility_residual",
        ],
    )
    nested_metrics_path = build_dir / "nested_proxy_residual_metrics.json"
    write_json(nested_metrics_path, nested_metrics)

    # I — oracle reproduction
    oracle = oracle_reproduction(bridge_rows)
    oracle_path = build_dir / "oracle_quality_residual_reproduction.json"
    write_json(oracle_path, oracle)

    decision, criteria = choose_decision(
        oracle,
        global_metrics,
        within_metrics,
        two_metrics,
        bridge_metrics,
        nested_metrics,
    )

    criteria_rows = []
    for name, item in criteria.items():
        criteria_rows.append({
            "criterion": name,
            "value": item.get("value"),
            "threshold_or_reference": (
                item.get("threshold")
                if "threshold" in item
                else item.get("reference")
            ),
            "pass": item["pass"],
        })

    criteria_path = build_dir / "core_criteria.csv"
    write_csv(
        criteria_path,
        criteria_rows,
        ["criterion", "value", "threshold_or_reference", "pass"],
    )

    decision_path = build_dir / "decision.txt"
    decision_path.write_text(decision + "\n", encoding="utf-8")

    # Positive-polarity integrity checks.
    global_group_ok = all(
        x["group_overlap"] == 0
        for x in global_metrics["split_audit"]
    )
    base_group_ok = all(
        x["group_overlap"] == 0
        for x in base_split_audit
    )
    nested_group_ok = all(
        x["outer_test_group_overlap_in_quality_training"] == 0
        and x["outer_test_group_overlap_in_residual_training"] == 0
        for x in nested_metrics["outer_split_audit"]
    )

    integrity = {
        "script_version": VERSION,
        "build": BUILD,
        "checks": {
            "q1_s02c_lock_verified": provenance["q1_s02c_lock_sha256"] == EXPECTED_Q1_S02C_LOCK_SHA256,
            "q1_s02c_decision_verified": provenance["q1_s02c_decision"] == EXPECTED_Q1_S02C_DECISION,
            "rows_exact_4350": len(rows) == EXPECTED_ROWS,
            "source_groups_exact_145": len({r["source_group_id"] for r in rows}) == SOURCE_GROUPS,
            "matched_units_exact_1450": len(units) == MATCHED_UNITS,
            "perturbations_exact_10": {r["perturbation"] for r in rows} == set(PERTURBATIONS),
            "seeds_exact_3": {int(r["seed"]) for r in rows} == set(SEEDS),
            "each_matched_unit_has_three_seeds": all(
                len(rr) == 3 and {int(r["seed"]) for r in rr} == set(SEEDS)
                for rr in matched.values()
            ),
            "b0_features_exact_19": len(SOURCE_FEATURE_NAMES) == 19,
            "source_dice_not_used_as_input_feature": "source_dice" not in SOURCE_FEATURE_NAMES,
            "delta_dice_not_used_as_quality_input_feature": "delta_dice" not in SOURCE_FEATURE_NAMES,
            "gradient_features_not_used": all(not f.startswith("grad_") for f in SOURCE_FEATURE_NAMES),
            "adapted_prediction_features_not_used": True,
            "target_data_not_loaded": True,
            "target_gt_not_loaded": True,
            "new_segmentation_inference_not_run": True,
            "global_oof_group_overlap_zero": global_group_ok,
            "base_regime_oof_group_overlap_zero": base_group_ok,
            "nested_outer_test_group_overlap_zero": nested_group_ok,
            "standardizers_fit_training_only": True,
            "hyperparameter_sweep_not_used": True,
            "feature_selection_not_used": True,
            "threshold_tuning_not_used": True,
            "all_primary_statistics_finite": all(
                math.isfinite(float(v))
                for v in [
                    global_metrics["spearman"],
                    np.median([within_metrics[s]["spearman"] for s in SEEDS]),
                    np.median([two_metrics[s]["spearman"] for s in SEEDS]),
                    two_metrics[HELDOUT_REGIME_SEED]["spearman"],
                    bridge_metrics["quality_shift_fidelity_spearman"],
                    bridge_metrics[
                        "proxy_quality_shift_vs_utility_residual_spearman"
                    ],
                    nested_metrics["spearman"],
                    oracle["actual_spearman"],
                ]
            ),
        },
        "criteria": criteria,
        "decision": decision,
    }

    failed_integrity = [
        k for k, v in integrity["checks"].items() if not v
    ]
    if failed_integrity:
        raise RuntimeError(
            f"Q1-S03A integrity failure: {failed_integrity}"
        )

    integrity_path = build_dir / "leakage_integrity_audit.json"
    write_json(integrity_path, integrity)

    within_vals = [within_metrics[s]["spearman"] for s in SEEDS]
    two_vals = [two_metrics[s]["spearman"] for s in SEEDS]

    summary = f"""===== Q1-S03A PROSPECTIVE SEGMENTATION QUALITY PROXY =====
Script version: {VERSION}
Build: {BUILD}

Frozen data:
  rows={EXPECTED_ROWS}
  source groups={SOURCE_GROUPS}
  matched units={MATCHED_UNITS}
  seeds={SEEDS}
  perturbations={len(PERTURBATIONS)}
  B0 features={len(SOURCE_FEATURE_NAMES)}
  new segmentation inference=NO
  target data used=NO
  target GT used=NO
  source_dice deployment input=NO
  gradient features used=NO
  adapted prediction features used=NO
  hyperparameter sweep=NO
  feature selection=NO
  threshold tuning=NO

[A — GLOBAL GROUPED-OOF QUALITY]
  Spearman={global_metrics['spearman']:.6f}
  Pearson={global_metrics['pearson']:.6f}
  MAE={global_metrics['mae']:.6f}
  RMSE={global_metrics['rmse']:.6f}
  Required Spearman >= {TH_GLOBAL:.2f}

[B — WITHIN-SEED QUALITY]
"""
    for seed in SEEDS:
        m = within_metrics[seed]
        summary += (
            f"  seed={seed}: Spearman={m['spearman']:.6f} "
            f"MAE={m['mae']:.6f} RMSE={m['rmse']:.6f}\n"
        )
    summary += (
        f"  Median within-seed Spearman={np.median(within_vals):.6f}\n"
        f"  Minimum within-seed Spearman={np.min(within_vals):.6f}\n"
        f"  Required median >= {TH_WITHIN_MEDIAN:.2f}; "
        f"each >= {TH_WITHIN_EACH:.2f}\n"
    )

    summary += "\n[C — FULL CROSS-SEED QUALITY SPEARMAN]\n"
    for train_seed in SEEDS:
        vals = []
        for test_seed in SEEDS:
            vals.append(
                f"test{test_seed}={cross_metrics[(train_seed,test_seed)]['spearman']:.6f}"
            )
        summary += f"  train{train_seed}: " + " | ".join(vals) + "\n"

    summary += "\n[D — STRICT TWO-SEEDS-TO-ONE QUALITY]\n"
    for seed in SEEDS:
        m = two_metrics[seed]
        summary += (
            f"  holdout={seed}: Spearman={m['spearman']:.6f} "
            f"MAE={m['mae']:.6f}\n"
        )
    summary += (
        f"  Median 2->1 Spearman={np.median(two_vals):.6f}\n"
        f"  Required median >= {TH_TWO_TO_ONE_MEDIAN:.2f}\n"
        f"  17+18->19 Spearman={two_metrics[HELDOUT_REGIME_SEED]['spearman']:.6f}\n"
        f"  Required seed19 >= {TH_HELDOUT19:.2f}\n"
    )

    summary += f"""
[F — PREDICTED QUALITY-SHIFT FIDELITY]
  qhat-shift vs true quality-shift Spearman={bridge_metrics['quality_shift_fidelity_spearman']:.6f}
  Required >= {TH_QSHIFT_FIDELITY:.2f}
  Pass={bridge_metrics['quality_shift_pass']}

[G — UTILITY-RELEVANCE BRIDGE]
  Oracle quality-shift vs residual Spearman={bridge_metrics['oracle_quality_shift_vs_utility_residual_spearman']:.6f}
  Proxy quality-shift vs residual Spearman={bridge_metrics['proxy_quality_shift_vs_utility_residual_spearman']:.6f}
  Same sign={bridge_metrics['proxy_oracle_sign_match']}
  Required |proxy Spearman| >= {TH_UTILITY_BRIDGE_ABS:.2f} and same sign
  Pass={bridge_metrics['utility_bridge_pass']}

[H — NESTED PROXY-SHIFT -> RESIDUAL]
  Spearman={nested_metrics['spearman']:.6f}
  MAE={nested_metrics['mae']:.6f}
  Required Spearman >= {TH_NESTED_RESIDUAL:.2f}
  Pass={nested_metrics['pass']}

[I — ORACLE REPRODUCTION]
  Reference Spearman={oracle['reference_spearman']:.6f}
  Actual Spearman={oracle['actual_spearman']:.6f}
  Absolute difference={oracle['absolute_difference']:.10f}
  Tolerance={ORACLE_TOL}
  Pass={oracle['pass']}

Decision: {decision}

If GO:
  Next = Q1-S03B Quality-Aware Model-Conditioned Individual Adaptation Effect.

If STOP:
  Do not rescue with nonlinear quality models or post-hoc feature selection.

[OK] Outputs: {args.output_dir}
"""

    run_log_path = build_dir / "run_log.txt"
    run_log_path.write_text(summary, encoding="utf-8")

    artifact_paths = {
        "protocol_copy": protocol_copy,
        "provenance_audit": provenance_path,
        "global_quality_oof_predictions": global_pred_path,
        "global_quality_metrics": global_metrics_path,
        "within_seed_quality_metrics": within_path,
        "cross_seed_quality_matrix_spearman": cross_spear_path,
        "cross_seed_quality_matrix_mae": cross_mae_path,
        "two_seed_to_one_quality_metrics": two_path,
        "base_regime_quality_predictions": base_pred_path,
        "quality_shift_bridge": bridge_path,
        "quality_shift_bridge_metrics": bridge_metrics_path,
        "nested_proxy_residual_predictions": nested_path,
        "nested_proxy_residual_metrics": nested_metrics_path,
        "oracle_quality_residual_reproduction": oracle_path,
        "core_criteria": criteria_path,
        "integrity_audit": integrity_path,
        "decision": decision_path,
        "run_log": run_log_path,
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
        "q1_s02c_lock_sha256": EXPECTED_Q1_S02C_LOCK_SHA256,
        "q1_s02c_decision": EXPECTED_Q1_S02C_DECISION,
        "q1_s00a_lock_sha256": EXPECTED_Q1_S00A_LOCK_SHA256,
        "utility_table_sha256": EXPECTED_UTILITY_TABLE_SHA256,
        "target_data_used": False,
        "target_gt_used": False,
        "new_segmentation_inference": False,
        "source_dice_deployment_input": False,
        "gradient_features_used": False,
        "adapted_prediction_features_used": False,
        "hyperparameter_sweep": False,
        "feature_selection": False,
        "threshold_tuning": False,
        "criteria": criteria,
        "decision": decision,
    }

    lock_path = build_dir / "Q1_S03A_QUALITY_PROXY_LOCK.json"
    write_json(lock_path, lock)
    lock_sha = file_sha256(lock_path)

    for name, path in artifact_paths.items():
        actual = file_sha256(path)
        expected = lock["artifacts"][name]["sha256"]
        if actual != expected:
            raise RuntimeError(f"Artifact changed before commit: {name}")

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    build_dir.rename(args.output_dir)

    print()
    print((args.output_dir / "run_log.txt").read_text(encoding="utf-8"))
    print(
        "Q1-S03A LOCK:",
        args.output_dir / "Q1_S03A_QUALITY_PROXY_LOCK.json",
    )
    print("Q1-S03A LOCK SHA256:", lock_sha)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run Q1-S03A prospective segmentation quality-proxy feasibility "
            "using frozen source-side assets."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run implementation tests without project files.",
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
