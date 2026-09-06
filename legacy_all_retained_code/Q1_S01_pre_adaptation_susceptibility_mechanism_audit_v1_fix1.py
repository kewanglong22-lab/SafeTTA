#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Q1-S01 — Pre-Adaptation Susceptibility Mechanism Audit

Purpose
-------
Determine whether the strong PRE-SOURCE B0 signal discovered in Q1-S00A
(BENEFIT AUROC = 0.807722) reflects a non-trivial and generalizable
pre-adaptation susceptibility signal, or whether it is adequately
explained by:

1) one trivial scalar proxy (entropy/confidence/etc.),
2) perturbation identity,
3) checkpoint seed identity,
4) source segmentation quality / room-for-improvement,
5) perturbation-specific effects,
6) checkpoint-specific effects.

This is a source-side mechanism audit. It does not load target-domain data,
does not run segmentation inference, and does not use gradient features in
the susceptibility models.

Expected project root
---------------------
F:\\MEDSEG_SAFETTA

Required inputs
---------------
docs\\Q1_S01_pre_adaptation_susceptibility_mechanism_audit_preregistered_protocol_v1.md

outputs\\Q1_S00A_iae_pre_adaptation_gradient_probe_v1\\
    Q1_S00A_IAE_FEASIBILITY_LOCK.json
    gradient_probe_features.csv
    oof_predictions.csv
    oof_metrics.json

outputs\\S07_B_pranet_source_side_counterfactual_utility_dataset_v1\\
    source_side_counterfactual_utility_table.csv

Typical usage
-------------
python .\\Q1_S01_pre_adaptation_susceptibility_mechanism_audit_v1.py --self-test
python .\\Q1_S01_pre_adaptation_susceptibility_mechanism_audit_v1.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


VERSION = "2026-08-19-Q1-S01-v1-fix1"
BUILD = "Q1_S01_PRE_ADAPTATION_SUSCEPTIBILITY_MECHANISM_AUDIT_SCHEMA_FIX1"

ROOT = Path(r"F:\MEDSEG_SAFETTA")

PROTOCOL = (
    ROOT
    / "docs"
    / "Q1_S01_pre_adaptation_susceptibility_mechanism_audit_preregistered_protocol_v1_fix1.md"
)
EXPECTED_PROTOCOL_SHA256 = (
    "fe5d646b0000d004490122aeb844bd9a491b064b86bb6ace90fb2c3bb1b45d2a"
)

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
Q1_S00A_OOF = Q1_S00A_DIR / "oof_predictions.csv"
Q1_S00A_METRICS = Q1_S00A_DIR / "oof_metrics.json"

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
    / "Q1_S01_pre_adaptation_susceptibility_mechanism_audit_v1_fix1"
)

SOURCE_GROUPS = 145
N_FOLDS = 5
A1_ACTION = "A1_TENT_1STEP"
EXPECTED_ROWS = 4350

SEEDS = (20260817, 20260818, 20260819)
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

REFERENCE_B0 = {
    "delta_dice_spearman": 0.263722,
    "harm_auroc": 0.771570,
    "benefit_auroc": 0.807722,
}
REPRO_TOL = 1e-6

RIDGE_ALPHA = 1.0
LOGISTIC_C = 1.0
LOGISTIC_MAX_ITER = 5000
RANDOM_STATE = 20260819

GO_MULTI_MINUS_SINGLE = 0.03
GO_B0_MINUS_C2 = 0.05
GO_D2_MINUS_D1 = 0.03
GO_WITHIN_PERT_MEDIAN = 0.70
GO_WITHIN_PERT_COUNT_THRESHOLD = 0.65
GO_WITHIN_PERT_MIN_COUNT = 7
GO_LOPO_GROUP = 0.68
GO_LOSEED_GROUP = 0.68

INVALID_REPRO = "INVALID_Q1_S01_B0_REPRODUCTION_FAILURE"
STOP_SINGLE = "STOP_SUSCEPTIBILITY_TRIVIAL_SINGLE_PROXY"
STOP_QUALITY = "STOP_SUSCEPTIBILITY_SOURCE_QUALITY_EXPLAINS_SIGNAL"
STOP_CONDITION = "STOP_SUSCEPTIBILITY_CONDITION_IDENTITY_EXPLAINS_SIGNAL"
STOP_PERT = "STOP_SUSCEPTIBILITY_PERTURBATION_GENERALIZATION_FAILURE"
STOP_SEED = "STOP_SUSCEPTIBILITY_CHECKPOINT_GENERALIZATION_FAILURE"
GO_DECISION = "GO_ADAPTATION_SUSCEPTIBILITY_Q1"


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
            f"{label} SHA256 mismatch: expected={expected} actual={actual}"
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
        raise RuntimeError(f"Non-finite value: {v}")
    return x


def stable_spearman(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if len(y_true) < 2 or np.std(y_true) < 1e-15 or np.std(y_pred) < 1e-15:
        return 0.0
    s = spearmanr(y_true, y_pred).statistic
    if s is None or not np.isfinite(s):
        return 0.0
    return float(s)


def safe_auroc(y, p):
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(p, dtype=np.float64)
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, p))


def safe_auprc(y, p):
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(p, dtype=np.float64)
    if len(np.unique(y)) < 2:
        return None
    return float(average_precision_score(y, p))


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
                random_state=RANDOM_STATE,
            ),
        ),
    ])


def make_ridge():
    return Pipeline([
        ("scale", StandardScaler()),
        ("reg", Ridge(alpha=RIDGE_ALPHA)),
    ])


def numeric_matrix(rows, feature_names: Sequence[str]) -> np.ndarray:
    x = np.asarray(
        [[safe_float(r[name]) for name in feature_names] for r in rows],
        dtype=np.float64,
    )
    if x.ndim != 2 or x.shape[1] != len(feature_names):
        raise RuntimeError("Numeric feature-matrix shape mismatch.")
    if not np.isfinite(x).all():
        raise RuntimeError("Numeric feature matrix contains NaN/Inf.")
    return x


def binary_target(rows, field: str) -> np.ndarray:
    y = np.asarray([int(r[field]) for r in rows], dtype=np.int64)
    if set(np.unique(y).tolist()) - {0, 1}:
        raise RuntimeError(f"Invalid binary field: {field}")
    return y


def delta_target(rows) -> np.ndarray:
    y = np.asarray([safe_float(r["delta_dice"]) for r in rows], dtype=np.float64)
    if not np.isfinite(y).all():
        raise RuntimeError("Non-finite DeltaDice target.")
    return y


def one_hot_categories(
    rows,
    include_perturbation: bool,
    include_seed: bool,
) -> np.ndarray:
    cols = []
    if include_perturbation:
        for p in PERTURBATIONS:
            cols.append(
                np.asarray([1.0 if r["perturbation"] == p else 0.0 for r in rows])
            )
    if include_seed:
        for s in SEEDS:
            cols.append(
                np.asarray([1.0 if int(r["seed"]) == s else 0.0 for r in rows])
            )
    if not cols:
        return np.empty((len(rows), 0), dtype=np.float64)
    x = np.stack(cols, axis=1).astype(np.float64)
    return x


def combine_arrays(*xs):
    xs = [np.asarray(x, dtype=np.float64) for x in xs if x is not None]
    if not xs:
        raise RuntimeError("No arrays to combine.")
    if len(xs) == 1:
        return xs[0]
    return np.concatenate(xs, axis=1)


# ---------------------------------------------------------------------
# Input validation / provenance
# ---------------------------------------------------------------------

def load_inputs():
    protocol_sha = validate_sha(PROTOCOL, EXPECTED_PROTOCOL_SHA256, "Q1-S01 protocol")
    q1_lock_sha = validate_sha(
        Q1_S00A_LOCK,
        EXPECTED_Q1_S00A_LOCK_SHA256,
        "Q1-S00A lock",
    )
    utility_sha = validate_sha(
        UTILITY_TABLE,
        EXPECTED_UTILITY_TABLE_SHA256,
        "S07-B utility table",
    )

    q1_lock = json.loads(Q1_S00A_LOCK.read_text(encoding="utf-8"))

    if q1_lock.get("decision") != "STOP_GRADIENT_PROBE_NO_INCREMENT":
        raise RuntimeError(
            f"Unexpected Q1-S00A decision: {q1_lock.get('decision')}"
        )
    if bool(q1_lock.get("target_data_used", True)):
        raise RuntimeError("Q1-S00A lock reports target data use.")
    if bool(q1_lock.get("adapted_prediction_generated", True)):
        raise RuntimeError("Q1-S00A lock reports adapted prediction generation.")
    if bool(q1_lock.get("optimizer_step_called_during_probe", True)):
        raise RuntimeError("Q1-S00A lock reports optimizer.step usage.")

    for required_name in ("gradient_probe_features", "oof_predictions", "oof_metrics"):
        if required_name not in q1_lock.get("artifacts", {}):
            raise RuntimeError(f"Q1-S00A lock missing artifact: {required_name}")

    feature_expected_sha = q1_lock["artifacts"]["gradient_probe_features"]["sha256"]
    oof_expected_sha = q1_lock["artifacts"]["oof_predictions"]["sha256"]
    metrics_expected_sha = q1_lock["artifacts"]["oof_metrics"]["sha256"]

    feature_sha = validate_sha(
        Q1_S00A_FEATURES,
        feature_expected_sha,
        "Q1-S00A feature table",
    )
    oof_sha = validate_sha(
        Q1_S00A_OOF,
        oof_expected_sha,
        "Q1-S00A OOF predictions",
    )
    metrics_sha = validate_sha(
        Q1_S00A_METRICS,
        metrics_expected_sha,
        "Q1-S00A OOF metrics",
    )

    feature_rows, feature_fields = read_csv(Q1_S00A_FEATURES)
    oof_rows, oof_fields = read_csv(Q1_S00A_OOF)
    utility_rows, utility_fields = read_csv(UTILITY_TABLE)

    return {
        "protocol_sha256": protocol_sha,
        "q1_s00a_lock_sha256": q1_lock_sha,
        "q1_s00a_feature_sha256": feature_sha,
        "q1_s00a_oof_sha256": oof_sha,
        "q1_s00a_metrics_sha256": metrics_sha,
        "utility_table_sha256": utility_sha,
        "q1_s00a_lock": q1_lock,
        "feature_rows": feature_rows,
        "feature_fields": feature_fields,
        "oof_rows": oof_rows,
        "oof_fields": oof_fields,
        "utility_rows": utility_rows,
        "utility_fields": utility_fields,
    }


def validate_and_merge(inputs):
    feature_rows = inputs["feature_rows"]
    feature_fields = set(inputs["feature_fields"])
    utility_rows = inputs["utility_rows"]
    utility_fields = set(inputs["utility_fields"])

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

    missing = sorted(required_feature - feature_fields)
    if missing:
        raise RuntimeError(f"Q1-S00A feature table missing: {missing}")

    if len(feature_rows) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Q1-S00A feature-row mismatch: expected={EXPECTED_ROWS} actual={len(feature_rows)}"
        )

    gradient_fields = [f for f in feature_fields if f.startswith("grad_")]
    if not gradient_fields:
        raise RuntimeError("Expected gradient columns absent from Q1-S00A asset.")

    # Q1-S01 is allowed to read the file containing gradients for provenance,
    # but susceptibility matrices must never include them.
    for f in SOURCE_FEATURE_NAMES:
        if f.startswith("grad_"):
            raise RuntimeError(f"Gradient feature illegally in B0 list: {f}")

    required_utility = {
        "seed",
        "sample_id",
        "perturbation",
        "action",
        "source_dice",
        "delta_dice",
        "harmful",
        "neutral",
        "beneficial",
    }
    missing_u = sorted(required_utility - utility_fields)
    if missing_u:
        raise RuntimeError(f"Utility table missing: {missing_u}")

    a1_utility = [r for r in utility_rows if r["action"] == A1_ACTION]
    if len(a1_utility) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Utility A1 row mismatch: expected={EXPECTED_ROWS} actual={len(a1_utility)}"
        )

    utility_by_key = {}
    for r in a1_utility:
        key = (int(r["seed"]), r["sample_id"], r["perturbation"])
        if key in utility_by_key:
            raise RuntimeError(f"Duplicate utility key: {key}")
        utility_by_key[key] = r

    merged = []
    keys = set()
    group_folds = defaultdict(set)
    class_counts = Counter()

    for r in feature_rows:
        key = (int(r["seed"]), r["sample_id"], r["perturbation"])
        if key in keys:
            raise RuntimeError(f"Duplicate Q1-S00A feature key: {key}")
        keys.add(key)

        if key not in utility_by_key:
            raise RuntimeError(f"Missing utility match: {key}")
        u = utility_by_key[key]

        if abs(safe_float(r["delta_dice"]) - safe_float(u["delta_dice"])) > 1e-12:
            raise RuntimeError(f"DeltaDice mismatch: {key}")
        if int(r["harmful"]) != int(u["harmful"]):
            raise RuntimeError(f"HARM label mismatch: {key}")
        if int(r["neutral"]) != int(u["neutral"]):
            raise RuntimeError(f"NEUTRAL label mismatch: {key}")
        if int(r["beneficial"]) != int(u["beneficial"]):
            raise RuntimeError(f"BENEFIT label mismatch: {key}")

        seed = int(r["seed"])
        fold = int(r["fold"])
        perturbation = r["perturbation"]

        if seed not in SEEDS:
            raise RuntimeError(f"Unexpected seed: {seed}")
        if perturbation not in PERTURBATIONS:
            raise RuntimeError(f"Unexpected perturbation: {perturbation}")
        if fold not in range(N_FOLDS):
            raise RuntimeError(f"Unexpected fold: {fold}")

        group = r["source_group_id"]
        group_folds[group].add(fold)

        outcome = r["outcome_class"]
        class_counts[outcome] += 1

        row = dict(r)
        row["seed"] = seed
        row["fold"] = fold
        row["source_dice"] = safe_float(u["source_dice"])
        row["delta_dice"] = safe_float(r["delta_dice"])
        row["harmful"] = int(r["harmful"])
        row["neutral"] = int(r["neutral"])
        row["beneficial"] = int(r["beneficial"])
        for name in SOURCE_FEATURE_NAMES:
            row[name] = safe_float(r[name])
        merged.append(row)

    if len(group_folds) != SOURCE_GROUPS:
        raise RuntimeError(
            f"Source-group mismatch: expected={SOURCE_GROUPS} actual={len(group_folds)}"
        )

    leaking = [g for g, folds in group_folds.items() if len(folds) != 1]
    if leaking:
        raise RuntimeError(f"Frozen fold leakage: {len(leaking)} groups")

    fold_group_counts = Counter()
    for g, folds in group_folds.items():
        fold_group_counts[next(iter(folds))] += 1
    if dict(sorted(fold_group_counts.items())) != EXPECTED_FOLD_GROUP_COUNTS:
        raise RuntimeError(
            f"Fold-group counts changed: {dict(sorted(fold_group_counts.items()))}"
        )

    if dict(class_counts) != EXPECTED_CLASS_COUNTS:
        raise RuntimeError(
            f"Outcome class counts changed: expected={EXPECTED_CLASS_COUNTS} actual={dict(class_counts)}"
        )

    pert_counts = Counter(r["perturbation"] for r in merged)
    seed_counts = Counter(r["seed"] for r in merged)
    if set(pert_counts) != set(PERTURBATIONS):
        raise RuntimeError("Perturbation set mismatch.")
    if set(seed_counts) != set(SEEDS):
        raise RuntimeError("Seed set mismatch.")
    if any(v != SOURCE_GROUPS * len(SEEDS) for v in pert_counts.values()):
        raise RuntimeError(f"Perturbation row counts changed: {dict(pert_counts)}")
    if any(v != SOURCE_GROUPS * len(PERTURBATIONS) for v in seed_counts.values()):
        raise RuntimeError(f"Seed row counts changed: {dict(seed_counts)}")

    return sorted(
        merged,
        key=lambda r: (r["seed"], r["sample_id"], r["perturbation"]),
    ), {
        "rows": len(merged),
        "source_groups": len(group_folds),
        "fold_group_counts": dict(sorted(fold_group_counts.items())),
        "class_counts": dict(class_counts),
        "perturbation_counts": dict(pert_counts),
        "seed_counts": {str(k): v for k, v in seed_counts.items()},
        "gradient_columns_present_in_source_asset_but_not_used": len(gradient_fields),
    }


# ---------------------------------------------------------------------
# Generic OOF prediction helpers
# ---------------------------------------------------------------------

def grouped_oof_from_matrix(
    rows,
    x: np.ndarray,
    mode: str,
):
    if x.shape[0] != len(rows):
        raise RuntimeError("OOF matrix row mismatch.")

    out = [None] * len(rows)

    for fold in range(N_FOLDS):
        train_idx = [i for i, r in enumerate(rows) if int(r["fold"]) != fold]
        test_idx = [i for i, r in enumerate(rows) if int(r["fold"]) == fold]

        train_groups = {rows[i]["source_group_id"] for i in train_idx}
        test_groups = {rows[i]["source_group_id"] for i in test_idx}
        if train_groups & test_groups:
            raise RuntimeError(f"Grouped OOF leakage fold={fold}")

        x_train = x[train_idx]
        x_test = x[test_idx]

        if mode == "benefit":
            y_train = binary_target([rows[i] for i in train_idx], "beneficial")
            if len(np.unique(y_train)) < 2:
                raise RuntimeError(f"BENEFIT training fold lacks both classes: {fold}")
            model = make_logistic()
            model.fit(x_train, y_train)
            pred = model.predict_proba(x_test)[:, 1]
        elif mode == "harm":
            y_train = binary_target([rows[i] for i in train_idx], "harmful")
            if len(np.unique(y_train)) < 2:
                raise RuntimeError(f"HARM training fold lacks both classes: {fold}")
            model = make_logistic()
            model.fit(x_train, y_train)
            pred = model.predict_proba(x_test)[:, 1]
        elif mode == "delta":
            y_train = delta_target([rows[i] for i in train_idx])
            model = make_ridge()
            model.fit(x_train, y_train)
            pred = model.predict(x_test)
        else:
            raise ValueError(mode)

        for i, p in zip(test_idx, pred):
            out[i] = float(p)

    if any(v is None for v in out):
        raise RuntimeError(f"Incomplete OOF predictions mode={mode}")
    arr = np.asarray(out, dtype=np.float64)
    if not np.isfinite(arr).all():
        raise RuntimeError(f"Non-finite OOF predictions mode={mode}")
    return arr


def metrics_from_predictions(rows, pred_delta, pred_harm, pred_benefit):
    y_delta = delta_target(rows)
    y_harm = binary_target(rows, "harmful")
    y_benefit = binary_target(rows, "beneficial")

    return {
        "delta_dice_spearman": stable_spearman(y_delta, pred_delta),
        "delta_dice_mae": float(mean_absolute_error(y_delta, pred_delta)),
        "harm_auroc": float(roc_auc_score(y_harm, pred_harm)),
        "harm_auprc": float(average_precision_score(y_harm, pred_harm)),
        "benefit_auroc": float(roc_auc_score(y_benefit, pred_benefit)),
        "benefit_auprc": float(average_precision_score(y_benefit, pred_benefit)),
    }


# ---------------------------------------------------------------------
# Analysis A: B0 reproduction
# ---------------------------------------------------------------------

def reproduce_b0(rows):
    x = numeric_matrix(rows, SOURCE_FEATURE_NAMES)
    pred_delta = grouped_oof_from_matrix(rows, x, "delta")
    pred_harm = grouped_oof_from_matrix(rows, x, "harm")
    pred_benefit = grouped_oof_from_matrix(rows, x, "benefit")
    metrics = metrics_from_predictions(
        rows,
        pred_delta,
        pred_harm,
        pred_benefit,
    )

    reproduction_checks = {
        k: {
            "reference": REFERENCE_B0[k],
            "actual": metrics[k],
            "absolute_difference": abs(metrics[k] - REFERENCE_B0[k]),
            "pass": abs(metrics[k] - REFERENCE_B0[k]) <= REPRO_TOL,
        }
        for k in REFERENCE_B0
    }

    return {
        "metrics": metrics,
        "checks": reproduction_checks,
        "all_pass": all(v["pass"] for v in reproduction_checks.values()),
    }, pred_delta, pred_harm, pred_benefit


# ---------------------------------------------------------------------
# Analysis B: single-feature audit
# ---------------------------------------------------------------------

def run_single_feature_audit(rows, b0_benefit_auroc):
    results = []

    pbar = tqdm(
        SOURCE_FEATURE_NAMES,
        desc="Q1-S01 single-feature audit",
        unit="feature",
        dynamic_ncols=True,
    )
    for name in pbar:
        x = numeric_matrix(rows, [name])
        benefit_pred = grouped_oof_from_matrix(rows, x, "benefit")
        harm_pred = grouped_oof_from_matrix(rows, x, "harm")

        benefit_auroc = float(
            roc_auc_score(binary_target(rows, "beneficial"), benefit_pred)
        )
        harm_auroc = float(
            roc_auc_score(binary_target(rows, "harmful"), harm_pred)
        )

        results.append({
            "feature": name,
            "benefit_auroc": benefit_auroc,
            "harm_auroc": harm_auroc,
            "b0_minus_single_benefit_auroc": b0_benefit_auroc - benefit_auroc,
        })
        pbar.set_postfix(benefit=f"{benefit_auroc:.4f}")

    results.sort(key=lambda r: r["benefit_auroc"], reverse=True)
    best = results[0]
    gap = float(b0_benefit_auroc - best["benefit_auroc"])
    return results, {
        "best_single_feature": best["feature"],
        "best_single_benefit_auroc": best["benefit_auroc"],
        "b0_benefit_auroc": b0_benefit_auroc,
        "b0_minus_best_single_benefit_auroc": gap,
        "threshold": GO_MULTI_MINUS_SINGLE,
        "pass": gap >= GO_MULTI_MINUS_SINGLE,
    }


# ---------------------------------------------------------------------
# Analysis C: condition-identity baselines
# ---------------------------------------------------------------------

def run_condition_identity_audit(rows, b0_benefit_auroc):
    configs = [
        ("C0_PERTURBATION_ONLY", True, False),
        ("C1_SEED_ONLY", False, True),
        ("C2_PERTURBATION_PLUS_SEED", True, True),
    ]

    results = []
    for name, use_pert, use_seed in configs:
        x = one_hot_categories(rows, use_pert, use_seed)
        benefit_pred = grouped_oof_from_matrix(rows, x, "benefit")
        harm_pred = grouped_oof_from_matrix(rows, x, "harm")

        benefit_auroc = float(
            roc_auc_score(binary_target(rows, "beneficial"), benefit_pred)
        )
        harm_auroc = float(
            roc_auc_score(binary_target(rows, "harmful"), harm_pred)
        )
        results.append({
            "model": name,
            "benefit_auroc": benefit_auroc,
            "harm_auroc": harm_auroc,
            "b0_minus_benefit_auroc": b0_benefit_auroc - benefit_auroc,
        })

    c2 = next(r for r in results if r["model"] == "C2_PERTURBATION_PLUS_SEED")
    gap = float(b0_benefit_auroc - c2["benefit_auroc"])

    return results, {
        "b0_benefit_auroc": b0_benefit_auroc,
        "c2_benefit_auroc": c2["benefit_auroc"],
        "b0_minus_c2_benefit_auroc": gap,
        "threshold": GO_B0_MINUS_C2,
        "pass": gap >= GO_B0_MINUS_C2,
    }


# ---------------------------------------------------------------------
# Analysis D: source-Dice audit
# ---------------------------------------------------------------------

def run_source_dice_audit(rows):
    source_dice = np.asarray(
        [[safe_float(r["source_dice"])] for r in rows],
        dtype=np.float64,
    )
    condition = one_hot_categories(rows, True, True)
    b0 = numeric_matrix(rows, SOURCE_FEATURE_NAMES)

    configs = {
        "D0_SOURCE_DICE_ONLY": source_dice,
        "D1_SOURCE_DICE_PLUS_PERTURBATION_PLUS_SEED": combine_arrays(
            source_dice,
            condition,
        ),
        "D2_SOURCE_DICE_PLUS_PERTURBATION_PLUS_SEED_PLUS_B0": combine_arrays(
            source_dice,
            condition,
            b0,
        ),
    }

    results = []
    for name, x in configs.items():
        benefit_pred = grouped_oof_from_matrix(rows, x, "benefit")
        harm_pred = grouped_oof_from_matrix(rows, x, "harm")
        results.append({
            "model": name,
            "benefit_auroc": float(
                roc_auc_score(binary_target(rows, "beneficial"), benefit_pred)
            ),
            "harm_auroc": float(
                roc_auc_score(binary_target(rows, "harmful"), harm_pred)
            ),
        })

    d1 = next(
        r for r in results
        if r["model"] == "D1_SOURCE_DICE_PLUS_PERTURBATION_PLUS_SEED"
    )
    d2 = next(
        r for r in results
        if r["model"] == "D2_SOURCE_DICE_PLUS_PERTURBATION_PLUS_SEED_PLUS_B0"
    )
    gap = float(d2["benefit_auroc"] - d1["benefit_auroc"])

    source_dice_vec = np.asarray(
        [safe_float(r["source_dice"]) for r in rows],
        dtype=np.float64,
    )
    delta = delta_target(rows)

    return results, {
        "d1_benefit_auroc": d1["benefit_auroc"],
        "d2_benefit_auroc": d2["benefit_auroc"],
        "d2_minus_d1_benefit_auroc": gap,
        "threshold": GO_D2_MINUS_D1,
        "pass": gap >= GO_D2_MINUS_D1,
        "spearman_source_dice_vs_delta_dice": stable_spearman(
            source_dice_vec,
            delta,
        ),
    }


# ---------------------------------------------------------------------
# Analysis E: within-perturbation using frozen Q1-S00A B0 OOF
# ---------------------------------------------------------------------

def load_frozen_b0_oof(inputs, rows):
    oof_rows = [
        r for r in inputs["oof_rows"]
        if r["family"] == "B0_PRE_SOURCE"
    ]
    if len(oof_rows) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Frozen B0 OOF row mismatch: expected={EXPECTED_ROWS} actual={len(oof_rows)}"
        )

    by_key = {}
    for r in oof_rows:
        key = (int(r["seed"]), r["sample_id"], r["perturbation"])
        if key in by_key:
            raise RuntimeError(f"Duplicate frozen B0 OOF key: {key}")
        by_key[key] = r

    out = []
    for r in rows:
        key = (int(r["seed"]), r["sample_id"], r["perturbation"])
        if key not in by_key:
            raise RuntimeError(f"Missing frozen B0 OOF row: {key}")
        p = by_key[key]
        if abs(safe_float(p["delta_dice"]) - safe_float(r["delta_dice"])) > 1e-12:
            raise RuntimeError(f"Frozen B0 OOF target mismatch: {key}")
        out.append({
            "pred_delta_dice": safe_float(p["pred_delta_dice"]),
            "pred_harm_probability": safe_float(p["pred_harm_probability"]),
            "pred_benefit_probability": safe_float(p["pred_benefit_probability"]),
        })
    return out


def run_within_perturbation_audit(rows, frozen_oof):
    results = []
    evaluable_benefit = []

    for perturbation in PERTURBATIONS:
        idx = [i for i, r in enumerate(rows) if r["perturbation"] == perturbation]
        y_b = np.asarray([rows[i]["beneficial"] for i in idx], dtype=np.int64)
        y_h = np.asarray([rows[i]["harmful"] for i in idx], dtype=np.int64)
        p_b = np.asarray(
            [frozen_oof[i]["pred_benefit_probability"] for i in idx],
            dtype=np.float64,
        )
        p_h = np.asarray(
            [frozen_oof[i]["pred_harm_probability"] for i in idx],
            dtype=np.float64,
        )

        b_auc = safe_auroc(y_b, p_b)
        h_auc = safe_auroc(y_h, p_h)

        if b_auc is not None:
            evaluable_benefit.append(b_auc)

        results.append({
            "perturbation": perturbation,
            "rows": len(idx),
            "benefit_positive": int(y_b.sum()),
            "benefit_negative": int((1 - y_b).sum()),
            "harm_positive": int(y_h.sum()),
            "harm_negative": int((1 - y_h).sum()),
            "benefit_auroc": "" if b_auc is None else b_auc,
            "harm_auroc": "" if h_auc is None else h_auc,
            "benefit_evaluable": b_auc is not None,
            "harm_evaluable": h_auc is not None,
            "benefit_pass_065": bool(
                b_auc is not None and b_auc >= GO_WITHIN_PERT_COUNT_THRESHOLD
            ),
        })

    if not evaluable_benefit:
        raise RuntimeError("No evaluable perturbation for BENEFIT AUROC.")

    median_auc = float(np.median(evaluable_benefit))
    pass_count = sum(
        1
        for r in results
        if r["benefit_evaluable"] and r["benefit_pass_065"]
    )

    return results, {
        "evaluable_perturbations": len(evaluable_benefit),
        "median_benefit_auroc": median_auc,
        "median_threshold": GO_WITHIN_PERT_MEDIAN,
        "median_pass": median_auc >= GO_WITHIN_PERT_MEDIAN,
        "count_benefit_auroc_ge_065": pass_count,
        "count_threshold": GO_WITHIN_PERT_MIN_COUNT,
        "count_pass": pass_count >= GO_WITHIN_PERT_MIN_COUNT,
        "pass": (
            median_auc >= GO_WITHIN_PERT_MEDIAN
            and pass_count >= GO_WITHIN_PERT_MIN_COUNT
        ),
    }


# ---------------------------------------------------------------------
# Analyses F/G: strict held-out condition + held-out source group
# ---------------------------------------------------------------------

def strict_condition_group_predictions(rows, condition: str):
    if condition not in {"perturbation", "seed"}:
        raise ValueError(condition)

    x_all = numeric_matrix(rows, SOURCE_FEATURE_NAMES)
    pred_delta = np.full(len(rows), np.nan, dtype=np.float64)
    pred_harm = np.full(len(rows), np.nan, dtype=np.float64)
    pred_benefit = np.full(len(rows), np.nan, dtype=np.float64)

    values = list(PERTURBATIONS) if condition == "perturbation" else list(SEEDS)

    total_jobs = len(values) * N_FOLDS
    pbar = tqdm(
        total=total_jobs,
        desc=f"Q1-S01 strict leave-one-{condition}-out+group",
        unit="split",
        dynamic_ncols=True,
    )

    split_audit = []

    for value in values:
        for fold in range(N_FOLDS):
            if condition == "perturbation":
                train_idx = [
                    i for i, r in enumerate(rows)
                    if r["perturbation"] != value and int(r["fold"]) != fold
                ]
                test_idx = [
                    i for i, r in enumerate(rows)
                    if r["perturbation"] == value and int(r["fold"]) == fold
                ]
            else:
                train_idx = [
                    i for i, r in enumerate(rows)
                    if int(r["seed"]) != int(value) and int(r["fold"]) != fold
                ]
                test_idx = [
                    i for i, r in enumerate(rows)
                    if int(r["seed"]) == int(value) and int(r["fold"]) == fold
                ]

            if not test_idx:
                raise RuntimeError(
                    f"Empty strict test split condition={condition} value={value} fold={fold}"
                )

            train_groups = {rows[i]["source_group_id"] for i in train_idx}
            test_groups = {rows[i]["source_group_id"] for i in test_idx}
            if train_groups & test_groups:
                raise RuntimeError(
                    f"Strict group leakage condition={condition} value={value} fold={fold}"
                )

            if condition == "perturbation":
                if any(rows[i]["perturbation"] == value for i in train_idx):
                    raise RuntimeError("Held-out perturbation appears in training.")
                if any(rows[i]["perturbation"] != value for i in test_idx):
                    raise RuntimeError("Strict perturbation test mismatch.")
            else:
                if any(int(rows[i]["seed"]) == int(value) for i in train_idx):
                    raise RuntimeError("Held-out seed appears in training.")
                if any(int(rows[i]["seed"]) != int(value) for i in test_idx):
                    raise RuntimeError("Strict seed test mismatch.")

            x_train = x_all[train_idx]
            x_test = x_all[test_idx]

            train_rows = [rows[i] for i in train_idx]

            y_b_train = binary_target(train_rows, "beneficial")
            y_h_train = binary_target(train_rows, "harmful")
            y_d_train = delta_target(train_rows)

            if len(np.unique(y_b_train)) < 2:
                raise RuntimeError("Strict BENEFIT train split lacks both classes.")
            if len(np.unique(y_h_train)) < 2:
                raise RuntimeError("Strict HARM train split lacks both classes.")

            benefit_model = make_logistic()
            harm_model = make_logistic()
            delta_model = make_ridge()

            benefit_model.fit(x_train, y_b_train)
            harm_model.fit(x_train, y_h_train)
            delta_model.fit(x_train, y_d_train)

            pb = benefit_model.predict_proba(x_test)[:, 1]
            ph = harm_model.predict_proba(x_test)[:, 1]
            pd = delta_model.predict(x_test)

            for i, a, b, c in zip(test_idx, pd, ph, pb):
                if np.isfinite(pred_delta[i]):
                    raise RuntimeError(
                        f"Strict prediction duplicated row={i} condition={condition}"
                    )
                pred_delta[i] = float(a)
                pred_harm[i] = float(b)
                pred_benefit[i] = float(c)

            split_audit.append({
                "condition": condition,
                "held_out_value": str(value),
                "fold": fold,
                "train_rows": len(train_idx),
                "test_rows": len(test_idx),
                "train_groups": len(train_groups),
                "test_groups": len(test_groups),
                "group_overlap": 0,
            })

            pbar.update(1)

    pbar.close()

    if not (
        np.isfinite(pred_delta).all()
        and np.isfinite(pred_harm).all()
        and np.isfinite(pred_benefit).all()
    ):
        raise RuntimeError(f"Incomplete strict predictions condition={condition}")

    y_b = binary_target(rows, "beneficial")
    y_h = binary_target(rows, "harmful")
    y_d = delta_target(rows)

    metrics = {
        "condition": condition,
        "rows": len(rows),
        "benefit_auroc": float(roc_auc_score(y_b, pred_benefit)),
        "benefit_auprc": float(average_precision_score(y_b, pred_benefit)),
        "harm_auroc": float(roc_auc_score(y_h, pred_harm)),
        "harm_auprc": float(average_precision_score(y_h, pred_harm)),
        "delta_dice_spearman": stable_spearman(y_d, pred_delta),
        "delta_dice_mae": float(mean_absolute_error(y_d, pred_delta)),
    }

    pred_rows = []
    for r, pd, ph, pb in zip(rows, pred_delta, pred_harm, pred_benefit):
        pred_rows.append({
            "seed": int(r["seed"]),
            "sample_id": r["sample_id"],
            "source_group_id": r["source_group_id"],
            "fold": int(r["fold"]),
            "perturbation": r["perturbation"],
            "base_case_id": r["base_case_id"],
            "delta_dice": float(r["delta_dice"]),
            "harmful": int(r["harmful"]),
            "beneficial": int(r["beneficial"]),
            "pred_delta_dice": float(pd),
            "pred_harm_probability": float(ph),
            "pred_benefit_probability": float(pb),
        })

    return pred_rows, metrics, split_audit


# ---------------------------------------------------------------------
# Decision logic
# ---------------------------------------------------------------------

def choose_decision(repro, single, condition, quality, within, lopo, loseed):
    criteria = {
        "exact_b0_reproduction": {
            "value": bool(repro["all_pass"]),
            "threshold": True,
            "pass": bool(repro["all_pass"]),
        },
        "b0_minus_best_single_benefit_auroc": {
            "value": single["b0_minus_best_single_benefit_auroc"],
            "threshold": GO_MULTI_MINUS_SINGLE,
            "pass": single["pass"],
        },
        "b0_minus_condition_identity_benefit_auroc": {
            "value": condition["b0_minus_c2_benefit_auroc"],
            "threshold": GO_B0_MINUS_C2,
            "pass": condition["pass"],
        },
        "d2_minus_d1_benefit_auroc": {
            "value": quality["d2_minus_d1_benefit_auroc"],
            "threshold": GO_D2_MINUS_D1,
            "pass": quality["pass"],
        },
        "within_perturbation_median_benefit_auroc": {
            "value": within["median_benefit_auroc"],
            "threshold": GO_WITHIN_PERT_MEDIAN,
            "pass": within["median_pass"],
        },
        "within_perturbation_count_ge_065": {
            "value": within["count_benefit_auroc_ge_065"],
            "threshold": GO_WITHIN_PERT_MIN_COUNT,
            "pass": within["count_pass"],
        },
        "strict_lopo_group_benefit_auroc": {
            "value": lopo["benefit_auroc"],
            "threshold": GO_LOPO_GROUP,
            "pass": lopo["benefit_auroc"] >= GO_LOPO_GROUP,
        },
        "strict_loseed_group_benefit_auroc": {
            "value": loseed["benefit_auroc"],
            "threshold": GO_LOSEED_GROUP,
            "pass": loseed["benefit_auroc"] >= GO_LOSEED_GROUP,
        },
    }

    failed = [k for k, v in criteria.items() if not v["pass"]]

    if not repro["all_pass"]:
        decision = INVALID_REPRO
    elif not single["pass"]:
        decision = STOP_SINGLE
    elif not quality["pass"]:
        decision = STOP_QUALITY
    elif not condition["pass"]:
        decision = STOP_CONDITION
    elif not (
        within["median_pass"]
        and within["count_pass"]
        and lopo["benefit_auroc"] >= GO_LOPO_GROUP
    ):
        decision = STOP_PERT
    elif loseed["benefit_auroc"] < GO_LOSEED_GROUP:
        decision = STOP_SEED
    else:
        decision = GO_DECISION

    return decision, criteria, failed


# ---------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------

def make_synthetic_rows():
    rng = np.random.default_rng(42)
    groups = 20
    folds = 5
    seeds = SEEDS
    perts = PERTURBATIONS[:4]

    rows = []
    for g in range(groups):
        fold = g % folds
        latent = rng.normal()
        for seed in seeds:
            seed_shift = (seed - seeds[0]) * 0.02
            for p_idx, perturbation in enumerate(perts):
                x1 = latent + rng.normal(scale=0.6)
                x2 = rng.normal()
                score = 0.9 * x1 + 0.2 * x2 + 0.1 * p_idx + seed_shift
                benefit = int(score > 0.6)
                harm = int(score < -0.6)
                if benefit:
                    delta = 0.03 + 0.03 * max(score, 0)
                elif harm:
                    delta = -0.03 - 0.03 * max(-score, 0)
                else:
                    delta = 0.005 * score

                row = {
                    "seed": seed,
                    "sample_id": f"s{g:03d}",
                    "source_group_id": f"g{g:03d}",
                    "fold": fold,
                    "perturbation": perturbation,
                    "base_case_id": f"{seed}_s{g:03d}_{perturbation}",
                    "delta_dice": float(delta),
                    "harmful": harm,
                    "neutral": int(not benefit and not harm),
                    "beneficial": benefit,
                    "outcome_class": (
                        "BENEFIT" if benefit else "HARM" if harm else "NEUTRAL"
                    ),
                    "source_dice": float(np.clip(0.75 - 0.08 * latent + rng.normal(scale=0.03), 0, 1)),
                }
                for i, name in enumerate(SOURCE_FEATURE_NAMES):
                    if i == 0:
                        row[name] = float(x1)
                    elif i == 1:
                        row[name] = float(x2)
                    else:
                        row[name] = float(rng.normal() + 0.05 * x1)
                rows.append(row)
    return rows


def self_test():
    assert SOURCE_GROUPS == 145
    assert EXPECTED_ROWS == 4350
    assert N_FOLDS == 5
    assert len(PERTURBATIONS) == 10
    assert len(SEEDS) == 3
    assert len(SOURCE_FEATURE_NAMES) == 19
    assert A1_ACTION == "A1_TENT_1STEP"
    assert abs(GO_MULTI_MINUS_SINGLE - 0.03) < 1e-15
    assert abs(GO_B0_MINUS_C2 - 0.05) < 1e-15
    assert abs(GO_D2_MINUS_D1 - 0.03) < 1e-15
    assert abs(GO_LOPO_GROUP - 0.68) < 1e-15
    assert abs(GO_LOSEED_GROUP - 0.68) < 1e-15

    rows = make_synthetic_rows()

    x = numeric_matrix(rows, SOURCE_FEATURE_NAMES[:2])
    pb = grouped_oof_from_matrix(rows, x, "benefit")
    ph = grouped_oof_from_matrix(rows, x, "harm")
    pd = grouped_oof_from_matrix(rows, x, "delta")
    assert len(pb) == len(rows)
    assert len(ph) == len(rows)
    assert len(pd) == len(rows)
    assert np.isfinite(pb).all()
    assert np.isfinite(ph).all()
    assert np.isfinite(pd).all()

    cond = one_hot_categories(rows, True, True)
    assert cond.shape[0] == len(rows)
    assert cond.shape[1] == len(PERTURBATIONS) + len(SEEDS)

    metrics = metrics_from_predictions(rows, pd, ph, pb)
    assert 0.0 <= metrics["benefit_auroc"] <= 1.0
    assert 0.0 <= metrics["harm_auroc"] <= 1.0

    print("FROZEN_CONSTANTS_TEST_PASS")
    print("GROUPED_OOF_TEST_PASS")
    print("ONE_HOT_CONDITION_TEST_PASS")
    print("METRICS_TEST_PASS")
    print("SELF_TEST_PASS")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def run(args):
    if args.output_dir.exists():
        raise FileExistsError(
            f"Q1-S01 output already exists: {args.output_dir}"
        )

    build_dir = Path(str(args.output_dir) + "__building")
    if build_dir.exists():
        raise FileExistsError(
            f"Partial Q1-S01 build exists: {build_dir}. "
            "Remove it only if the previous run was technically interrupted."
        )
    build_dir.mkdir(parents=True, exist_ok=False)

    print("===== Q1-S01 PRE-ADAPTATION SUSCEPTIBILITY MECHANISM AUDIT =====")
    print("SCHEMA_FIX: source_high_entropy_fraction_050")
    print(f"Script version: {VERSION}")
    print(f"Build: {BUILD}")
    print("Target data used: NO")
    print("Target GT used: NO")
    print("Gradient features used in susceptibility models: NO")
    print("New segmentation inference: NO")
    print("Hyperparameter sweep: NO")
    print("Feature selection: NO")
    print()

    inputs = load_inputs()
    rows, dataset_audit = validate_and_merge(inputs)

    protocol_copy = build_dir / "preregistered_protocol_copy.md"
    shutil.copy2(PROTOCOL, protocol_copy)
    if file_sha256(protocol_copy) != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("Protocol copy SHA mismatch.")

    provenance = {
        "script_version": VERSION,
        "build": BUILD,
        "protocol_sha256": inputs["protocol_sha256"],
        "q1_s00a_lock_sha256": inputs["q1_s00a_lock_sha256"],
        "q1_s00a_feature_sha256": inputs["q1_s00a_feature_sha256"],
        "q1_s00a_oof_sha256": inputs["q1_s00a_oof_sha256"],
        "q1_s00a_metrics_sha256": inputs["q1_s00a_metrics_sha256"],
        "utility_table_sha256": inputs["utility_table_sha256"],
        "q1_s00a_decision": inputs["q1_s00a_lock"]["decision"],
        "dataset_audit": dataset_audit,
        "target_data_used": False,
        "target_gt_used": False,
        "gradient_features_used_in_susceptibility_models": False,
        "new_segmentation_inference": False,
        "hyperparameter_sweep": False,
        "feature_selection": False,
    }
    provenance_path = build_dir / "provenance_audit.json"
    write_json(provenance_path, provenance)

    # A — exact B0 reproduction
    b0_repro, b0_pred_delta, b0_pred_harm, b0_pred_benefit = reproduce_b0(rows)
    b0_repro_path = build_dir / "b0_reproduction.json"
    write_json(b0_repro_path, b0_repro)

    if not b0_repro["all_pass"]:
        decision_path = build_dir / "decision.txt"
        decision_path.write_text(INVALID_REPRO + "\n", encoding="utf-8")
        raise RuntimeError(
            "Q1-S01 exact B0 reproduction failed. "
            f"See {b0_repro_path}"
        )

    # Compare recalculated B0 metrics against the actual frozen OOF artifact too.
    frozen_b0_oof = load_frozen_b0_oof(inputs, rows)
    frozen_b0_benefit = np.asarray(
        [r["pred_benefit_probability"] for r in frozen_b0_oof],
        dtype=np.float64,
    )
    frozen_b0_harm = np.asarray(
        [r["pred_harm_probability"] for r in frozen_b0_oof],
        dtype=np.float64,
    )
    frozen_b0_delta = np.asarray(
        [r["pred_delta_dice"] for r in frozen_b0_oof],
        dtype=np.float64,
    )

    frozen_artifact_metrics = metrics_from_predictions(
        rows,
        frozen_b0_delta,
        frozen_b0_harm,
        frozen_b0_benefit,
    )

    # The same fixed code/model/folds should reproduce the saved OOF metrics.
    for key in ("delta_dice_spearman", "harm_auroc", "benefit_auroc"):
        if abs(
            frozen_artifact_metrics[key] - b0_repro["metrics"][key]
        ) > REPRO_TOL:
            raise RuntimeError(
                f"Recomputed B0 differs from frozen Q1-S00A OOF artifact: {key}"
            )

    # B — single-feature
    single_rows, single_summary = run_single_feature_audit(
        rows,
        b0_repro["metrics"]["benefit_auroc"],
    )
    single_path = build_dir / "single_feature_audit.csv"
    write_csv(
        single_path,
        single_rows,
        [
            "feature",
            "benefit_auroc",
            "harm_auroc",
            "b0_minus_single_benefit_auroc",
        ],
    )

    # C — perturbation/seed identity
    condition_rows, condition_summary = run_condition_identity_audit(
        rows,
        b0_repro["metrics"]["benefit_auroc"],
    )
    condition_path = build_dir / "condition_identity_baselines.csv"
    write_csv(
        condition_path,
        condition_rows,
        [
            "model",
            "benefit_auroc",
            "harm_auroc",
            "b0_minus_benefit_auroc",
        ],
    )

    # D — source Dice audit
    source_dice_rows, source_dice_summary = run_source_dice_audit(rows)
    source_dice_path = build_dir / "source_dice_audit.csv"
    write_csv(
        source_dice_path,
        source_dice_rows,
        ["model", "benefit_auroc", "harm_auroc"],
    )

    # E — within perturbation using frozen B0 OOF
    within_rows, within_summary = run_within_perturbation_audit(
        rows,
        frozen_b0_oof,
    )
    within_path = build_dir / "within_perturbation_audit.csv"
    write_csv(
        within_path,
        within_rows,
        [
            "perturbation",
            "rows",
            "benefit_positive",
            "benefit_negative",
            "harm_positive",
            "harm_negative",
            "benefit_auroc",
            "harm_auroc",
            "benefit_evaluable",
            "harm_evaluable",
            "benefit_pass_065",
        ],
    )

    # F — strict LOPO + GROUP
    lopo_pred_rows, lopo_metrics, lopo_splits = strict_condition_group_predictions(
        rows,
        "perturbation",
    )
    lopo_pred_path = build_dir / "strict_lopo_group_predictions.csv"
    write_csv(
        lopo_pred_path,
        lopo_pred_rows,
        [
            "seed",
            "sample_id",
            "source_group_id",
            "fold",
            "perturbation",
            "base_case_id",
            "delta_dice",
            "harmful",
            "beneficial",
            "pred_delta_dice",
            "pred_harm_probability",
            "pred_benefit_probability",
        ],
    )
    lopo_metrics["split_audit"] = lopo_splits
    lopo_metrics["go_threshold_benefit_auroc"] = GO_LOPO_GROUP
    lopo_metrics["pass"] = lopo_metrics["benefit_auroc"] >= GO_LOPO_GROUP
    lopo_metrics_path = build_dir / "strict_lopo_group_metrics.json"
    write_json(lopo_metrics_path, lopo_metrics)

    # G — strict leave-one-seed + GROUP
    loseed_pred_rows, loseed_metrics, loseed_splits = strict_condition_group_predictions(
        rows,
        "seed",
    )
    loseed_pred_path = build_dir / "strict_loseed_group_predictions.csv"
    write_csv(
        loseed_pred_path,
        loseed_pred_rows,
        [
            "seed",
            "sample_id",
            "source_group_id",
            "fold",
            "perturbation",
            "base_case_id",
            "delta_dice",
            "harmful",
            "beneficial",
            "pred_delta_dice",
            "pred_harm_probability",
            "pred_benefit_probability",
        ],
    )
    loseed_metrics["split_audit"] = loseed_splits
    loseed_metrics["go_threshold_benefit_auroc"] = GO_LOSEED_GROUP
    loseed_metrics["pass"] = loseed_metrics["benefit_auroc"] >= GO_LOSEED_GROUP
    loseed_metrics_path = build_dir / "strict_loseed_group_metrics.json"
    write_json(loseed_metrics_path, loseed_metrics)

    decision, criteria, failed = choose_decision(
        b0_repro,
        single_summary,
        condition_summary,
        source_dice_summary,
        within_summary,
        lopo_metrics,
        loseed_metrics,
    )

    criteria_rows = []
    for name, item in criteria.items():
        criteria_rows.append({
            "criterion": name,
            "value": item["value"],
            "threshold": item["threshold"],
            "pass": item["pass"],
        })
    criteria_path = build_dir / "q1_s01_core_criteria.csv"
    write_csv(
        criteria_path,
        criteria_rows,
        ["criterion", "value", "threshold", "pass"],
    )

    decision_path = build_dir / "decision.txt"
    decision_path.write_text(decision + "\n", encoding="utf-8")

    # Positive-polarity integrity checks: True always means PASS.
    integrity = {
        "script_version": VERSION,
        "build": BUILD,
        "checks": {
            "source_groups_exact_145": dataset_audit["source_groups"] == SOURCE_GROUPS,
            "rows_exact_4350": dataset_audit["rows"] == EXPECTED_ROWS,
            "folds_exact_5": set(int(r["fold"]) for r in rows) == set(range(N_FOLDS)),
            "perturbations_exact_10": set(r["perturbation"] for r in rows) == set(PERTURBATIONS),
            "seeds_exact_3": set(int(r["seed"]) for r in rows) == set(SEEDS),
            "action_exact_A1_TENT_1STEP": True,
            "q1_s00a_lock_provenance_verified": True,
            "q1_s00a_feature_provenance_verified": True,
            "frozen_19_b0_features_present": all(
                name in inputs["feature_fields"]
                for name in SOURCE_FEATURE_NAMES
            ),
            "gradient_features_not_used_in_susceptibility_models": True,
            "target_data_not_loaded": True,
            "target_gt_not_loaded": True,
            "source_dice_used_audit_only": True,
            "standard_group_overlap_zero": True,
            "strict_lopo_group_overlap_zero": all(
                item["group_overlap"] == 0
                for item in lopo_splits
            ),
            "strict_loseed_group_overlap_zero": all(
                item["group_overlap"] == 0
                for item in loseed_splits
            ),
            "strict_lopo_heldout_perturbation_separation": True,
            "strict_loseed_heldout_seed_separation": True,
            "standardization_fit_training_only": True,
            "hyperparameter_sweep_not_used": True,
            "threshold_tuning_not_used": True,
            "feature_selection_not_used": True,
            "all_predictions_complete_finite": (
                len(lopo_pred_rows) == EXPECTED_ROWS
                and len(loseed_pred_rows) == EXPECTED_ROWS
                and all(
                    math.isfinite(float(r["pred_benefit_probability"]))
                    and math.isfinite(float(r["pred_harm_probability"]))
                    and math.isfinite(float(r["pred_delta_dice"]))
                    for r in lopo_pred_rows + loseed_pred_rows
                )
            ),
        },
        "failed_core_criteria": failed,
        "decision": decision,
    }

    failed_integrity = [
        k for k, v in integrity["checks"].items() if not v
    ]
    if failed_integrity:
        raise RuntimeError(
            f"Q1-S01 integrity failure: {failed_integrity}"
        )

    integrity_path = build_dir / "leakage_integrity_audit.json"
    write_json(integrity_path, integrity)

    summary = f"""===== Q1-S01 PRE-ADAPTATION SUSCEPTIBILITY MECHANISM AUDIT =====
Script version: {VERSION}
Build: {BUILD}

Frozen data:
  source groups={SOURCE_GROUPS}
  rows={EXPECTED_ROWS}
  folds={N_FOLDS}
  perturbations={len(PERTURBATIONS)}
  seeds={len(SEEDS)}
  action={A1_ACTION}
  target data used=NO
  target GT used=NO
  gradient features used in susceptibility models=NO
  new segmentation inference=NO
  hyperparameter sweep=NO
  feature selection=NO

[A — B0 REPRODUCTION]
  DeltaDice Spearman={b0_repro["metrics"]["delta_dice_spearman"]:.6f}
  HARM AUROC={b0_repro["metrics"]["harm_auroc"]:.6f}
  BENEFIT AUROC={b0_repro["metrics"]["benefit_auroc"]:.6f}
  Exact reproduction={b0_repro["all_pass"]}

[B — SINGLE-FEATURE TRIVIALITY]
  Best single feature={single_summary["best_single_feature"]}
  Best single BENEFIT AUROC={single_summary["best_single_benefit_auroc"]:.6f}
  B0-best single={single_summary["b0_minus_best_single_benefit_auroc"]:+.6f}
  Required >= +{GO_MULTI_MINUS_SINGLE:.2f}
  Pass={single_summary["pass"]}

[C — CONDITION IDENTITY]
  C2 perturbation+seed BENEFIT AUROC={condition_summary["c2_benefit_auroc"]:.6f}
  B0-C2={condition_summary["b0_minus_c2_benefit_auroc"]:+.6f}
  Required >= +{GO_B0_MINUS_C2:.2f}
  Pass={condition_summary["pass"]}

[D — SOURCE-DICE QUALITY AUDIT]
  D1 SourceDice+Perturbation+Seed BENEFIT AUROC={source_dice_summary["d1_benefit_auroc"]:.6f}
  D2 D1+B0 BENEFIT AUROC={source_dice_summary["d2_benefit_auroc"]:.6f}
  D2-D1={source_dice_summary["d2_minus_d1_benefit_auroc"]:+.6f}
  Required >= +{GO_D2_MINUS_D1:.2f}
  SourceDice vs DeltaDice Spearman={source_dice_summary["spearman_source_dice_vs_delta_dice"]:.6f}
  Pass={source_dice_summary["pass"]}

[E — WITHIN-PERTURBATION]
  Evaluable perturbations={within_summary["evaluable_perturbations"]}
  Median BENEFIT AUROC={within_summary["median_benefit_auroc"]:.6f}
  Required median >= {GO_WITHIN_PERT_MEDIAN:.2f}
  Perturbations BENEFIT AUROC >= {GO_WITHIN_PERT_COUNT_THRESHOLD:.2f}: {within_summary["count_benefit_auroc_ge_065"]}/10
  Required >= {GO_WITHIN_PERT_MIN_COUNT}/10
  Pass={within_summary["pass"]}

[F — STRICT LOPO+GROUP]
  BENEFIT AUROC={lopo_metrics["benefit_auroc"]:.6f}
  HARM AUROC={lopo_metrics["harm_auroc"]:.6f}
  DeltaDice Spearman={lopo_metrics["delta_dice_spearman"]:.6f}
  Required BENEFIT AUROC >= {GO_LOPO_GROUP:.2f}
  Pass={lopo_metrics["pass"]}

[G — STRICT LOSEED+GROUP]
  BENEFIT AUROC={loseed_metrics["benefit_auroc"]:.6f}
  HARM AUROC={loseed_metrics["harm_auroc"]:.6f}
  DeltaDice Spearman={loseed_metrics["delta_dice_spearman"]:.6f}
  Required BENEFIT AUROC >= {GO_LOSEED_GROUP:.2f}
  Pass={loseed_metrics["pass"]}

Failed core criteria:
  {failed}

Decision: {decision}

If GO:
  Next = Q1-S02 Susceptibility Representation / Action-Generalization Study.

If STOP:
  Do not rescue with a larger model. Reinterpret according to the failed mechanism audit.

[OK] Outputs: {args.output_dir}
"""
    run_log_path = build_dir / "run_log.txt"
    run_log_path.write_text(summary, encoding="utf-8")

    artifact_paths = {
        "protocol_copy": protocol_copy,
        "provenance_audit": provenance_path,
        "b0_reproduction": b0_repro_path,
        "single_feature_audit": single_path,
        "condition_identity_baselines": condition_path,
        "source_dice_audit": source_dice_path,
        "within_perturbation_audit": within_path,
        "strict_lopo_group_predictions": lopo_pred_path,
        "strict_lopo_group_metrics": lopo_metrics_path,
        "strict_loseed_group_predictions": loseed_pred_path,
        "strict_loseed_group_metrics": loseed_metrics_path,
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
        "q1_s00a_lock_sha256": EXPECTED_Q1_S00A_LOCK_SHA256,
        "utility_table_sha256": EXPECTED_UTILITY_TABLE_SHA256,
        "target_data_used": False,
        "target_gt_used": False,
        "gradient_features_used_in_susceptibility_models": False,
        "source_dice_deployment_feature": False,
        "hyperparameter_sweep": False,
        "feature_selection": False,
        "failed_core_criteria": failed,
        "decision": decision,
    }
    lock_path = build_dir / "Q1_S01_SUSCEPTIBILITY_MECHANISM_LOCK.json"
    write_json(lock_path, lock)
    lock_sha = file_sha256(lock_path)

    # Final artifact immutability check.
    for name, path in artifact_paths.items():
        expected = lock["artifacts"][name]["sha256"]
        actual = file_sha256(path)
        if actual != expected:
            raise RuntimeError(f"Artifact changed before final commit: {name}")

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    build_dir.rename(args.output_dir)

    print()
    print((args.output_dir / "run_log.txt").read_text(encoding="utf-8"))
    print(
        "Q1-S01 LOCK:",
        args.output_dir / "Q1_S01_SUSCEPTIBILITY_MECHANISM_LOCK.json",
    )
    print("Q1-S01 LOCK SHA256:", lock_sha)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run Q1-S01 pre-adaptation susceptibility mechanism audit "
            "using frozen Q1-S00A source-side assets."
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
        help="Run synthetic implementation tests without project files.",
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
