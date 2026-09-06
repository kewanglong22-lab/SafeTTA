#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Q1-S02A — Model–Case Interaction Mechanism Audit

Primary question
----------------
Why did Q1-S01's pre-adaptation B0 utility predictor fail to generalize
across checkpoint seeds?

Competing mechanisms:
  H1) outcome instability / model–case interaction:
      the SAME image under the SAME perturbation has a materially different
      true A1_TENT_1STEP effect for different deployed source checkpoints.

  H2) mapping instability / representation non-invariance:
      true adaptation effect is relatively checkpoint-stable, but the mapping
      from the 19-dimensional pre-adaptation source state to future utility
      changes across checkpoints.

This script performs no new segmentation inference and uses no target-domain
data. It reads only frozen source-side assets.

Expected project root:
    F:\\MEDSEG_SAFETTA

Typical usage:
    python .\\Q1_S02A_model_case_interaction_mechanism_audit_v1.py --self-test
    python .\\Q1_S02A_model_case_interaction_mechanism_audit_v1.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    cohen_kappa_score,
    mean_absolute_error,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


VERSION = "2026-08-19-Q1-S02A-v1"
BUILD = "Q1_S02A_MODEL_CASE_INTERACTION_MECHANISM_AUDIT"

ROOT = Path(r"F:\MEDSEG_SAFETTA")

PROTOCOL = (
    ROOT
    / "docs"
    / "Q1_S02A_model_case_interaction_mechanism_audit_preregistered_protocol_v1.md"
)
EXPECTED_PROTOCOL_SHA256 = (
    "bb0e9dea3166ade31a37c8df897859594693405311c885339ee06c144b37b90b"
)

Q1_S01_DIR = (
    ROOT
    / "outputs"
    / "Q1_S01_pre_adaptation_susceptibility_mechanism_audit_v1_fix2"
)
Q1_S01_LOCK = Q1_S01_DIR / "Q1_S01_SUSCEPTIBILITY_MECHANISM_LOCK.json"
EXPECTED_Q1_S01_LOCK_SHA256 = (
    "be15ed43cf55f4d79cc7a29f0f9de8b595ecead05f5236b747086a16805559bc"
)
EXPECTED_Q1_S01_DECISION = (
    "STOP_SUSCEPTIBILITY_CHECKPOINT_GENERALIZATION_FAILURE"
)

Q1_S00A_DIR = (
    ROOT
    / "outputs"
    / "Q1_S00A_iae_pre_adaptation_gradient_probe_v1"
)
Q1_S00A_LOCK = Q1_S00A_DIR / "Q1_S00A_IAE_FEASIBILITY_LOCK.json"
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
    / "Q1_S02A_model_case_interaction_mechanism_audit_v1"
)

SOURCE_GROUPS = 145
N_FOLDS = 5
MATCHED_UNITS = 1450
EXPECTED_ROWS = 4350
A1_ACTION = "A1_TENT_1STEP"

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

HARM_THRESHOLD = -0.02
BENEFIT_THRESHOLD = +0.02

LOGISTIC_C = 1.0
LOGISTIC_MAX_ITER = 5000
RIDGE_ALPHA = 1.0
RANDOM_STATE = 20260819

# Frozen mechanism thresholds.
TH_WITHIN_SEED_BENEFIT = 0.70
TH_CROSS_SEED_POOR = 0.68
TH_DELTA_STABLE = 0.50
TH_BH_FLIP_MATERIAL = 0.10
TH_COEF_COSINE_STABLE = 0.70

INVALID_MATCHED = "INVALID_Q1_S02A_MATCHED_UNIT_FAILURE"
STOP_WITHIN = "STOP_MODEL_CASE_INTERACTION_NO_STABLE_WITHIN_SEED_SIGNAL"
GO_MIXED = "GO_MIXED_MODEL_CASE_AND_MAPPING_INTERACTION"
GO_MCAC = "GO_MODEL_CASE_ADAPTATION_COMPATIBILITY"
GO_MAPPING = "GO_MODEL_CONDITIONED_REPRESENTATION"
STOP_UNRESOLVED = "STOP_Q1_S02A_MECHANISM_UNRESOLVED"


# ---------------------------------------------------------------------
# IO / math helpers
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


def safe_cosine(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-15:
        return 0.0
    return float(np.dot(a, b) / denom)


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


def outcome_from_delta(delta: float) -> str:
    if delta <= HARM_THRESHOLD:
        return "HARM"
    if delta >= BENEFIT_THRESHOLD:
        return "BENEFIT"
    return "NEUTRAL"


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


def matrix(rows, feature_names: Sequence[str]) -> np.ndarray:
    x = np.asarray(
        [[safe_float(r[f]) for f in feature_names] for r in rows],
        dtype=np.float64,
    )
    if x.ndim != 2 or x.shape[1] != len(feature_names):
        raise RuntimeError("Feature matrix shape mismatch.")
    if not np.isfinite(x).all():
        raise RuntimeError("Feature matrix contains NaN/Inf.")
    return x


def binary_target(rows, field: str) -> np.ndarray:
    y = np.asarray([int(r[field]) for r in rows], dtype=np.int64)
    if set(np.unique(y).tolist()) - {0, 1}:
        raise RuntimeError(f"Invalid binary target field: {field}")
    return y


def delta_target(rows) -> np.ndarray:
    y = np.asarray([safe_float(r["delta_dice"]) for r in rows], dtype=np.float64)
    if not np.isfinite(y).all():
        raise RuntimeError("Non-finite DeltaDice target.")
    return y


# ---------------------------------------------------------------------
# Frozen provenance + merged source-side rows
# ---------------------------------------------------------------------

def load_and_validate_inputs():
    protocol_sha = validate_sha(
        PROTOCOL,
        EXPECTED_PROTOCOL_SHA256,
        "Q1-S02A protocol",
    )
    q1_s01_sha = validate_sha(
        Q1_S01_LOCK,
        EXPECTED_Q1_S01_LOCK_SHA256,
        "Q1-S01 lock",
    )
    utility_sha = validate_sha(
        UTILITY_TABLE,
        EXPECTED_UTILITY_TABLE_SHA256,
        "S07-B utility table",
    )

    q1_s01_lock = json.loads(Q1_S01_LOCK.read_text(encoding="utf-8"))
    if q1_s01_lock.get("decision") != EXPECTED_Q1_S01_DECISION:
        raise RuntimeError(
            "Unexpected Q1-S01 decision: "
            f"{q1_s01_lock.get('decision')}"
        )
    if bool(q1_s01_lock.get("target_data_used", True)):
        raise RuntimeError("Q1-S01 lock reports target data use.")
    if bool(q1_s01_lock.get("target_gt_used", True)):
        raise RuntimeError("Q1-S01 lock reports target GT use.")

    expected_q1_s00a_lock_sha = q1_s01_lock.get("q1_s00a_lock_sha256")
    if not expected_q1_s00a_lock_sha:
        raise RuntimeError("Q1-S01 lock missing q1_s00a_lock_sha256.")

    q1_s00a_lock_sha = validate_sha(
        Q1_S00A_LOCK,
        expected_q1_s00a_lock_sha,
        "Q1-S00A lock",
    )
    q1_s00a_lock = json.loads(Q1_S00A_LOCK.read_text(encoding="utf-8"))

    if q1_s00a_lock.get("decision") != "STOP_GRADIENT_PROBE_NO_INCREMENT":
        raise RuntimeError(
            f"Unexpected Q1-S00A decision: {q1_s00a_lock.get('decision')}"
        )

    feature_art = q1_s00a_lock.get("artifacts", {}).get("gradient_probe_features")
    if not feature_art or "sha256" not in feature_art:
        raise RuntimeError("Q1-S00A lock missing gradient_probe_features artifact.")
    feature_sha = validate_sha(
        Q1_S00A_FEATURES,
        feature_art["sha256"],
        "Q1-S00A frozen feature table",
    )

    feature_rows, feature_fields = read_csv(Q1_S00A_FEATURES)
    utility_rows, utility_fields = read_csv(UTILITY_TABLE)

    required_features = {
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
    missing = sorted(required_features - set(feature_fields))
    if missing:
        raise RuntimeError(f"Frozen feature table missing: {missing}")

    required_utility = {
        "seed",
        "sample_id",
        "perturbation",
        "action",
        "source_dice",
        "action_dice",
        "delta_dice",
        "harmful",
        "neutral",
        "beneficial",
    }
    missing_u = sorted(required_utility - set(utility_fields))
    if missing_u:
        raise RuntimeError(f"Frozen utility table missing: {missing_u}")

    if len(feature_rows) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Feature row count mismatch: {len(feature_rows)} != {EXPECTED_ROWS}"
        )

    a1 = [r for r in utility_rows if r["action"] == A1_ACTION]
    if len(a1) != EXPECTED_ROWS:
        raise RuntimeError(
            f"A1 utility row count mismatch: {len(a1)} != {EXPECTED_ROWS}"
        )

    utility_by_key = {}
    for r in a1:
        key = (int(r["seed"]), r["sample_id"], r["perturbation"])
        if key in utility_by_key:
            raise RuntimeError(f"Duplicate utility key: {key}")
        utility_by_key[key] = r

    merged = []
    seen = set()
    group_folds = defaultdict(set)

    for r in feature_rows:
        key = (int(r["seed"]), r["sample_id"], r["perturbation"])
        if key in seen:
            raise RuntimeError(f"Duplicate feature key: {key}")
        seen.add(key)
        if key not in utility_by_key:
            raise RuntimeError(f"Missing utility match: {key}")
        u = utility_by_key[key]

        delta_f = safe_float(r["delta_dice"])
        delta_u = safe_float(u["delta_dice"])
        if abs(delta_f - delta_u) > 1e-12:
            raise RuntimeError(f"DeltaDice mismatch: {key}")

        outcome = outcome_from_delta(delta_f)
        if outcome != r["outcome_class"]:
            raise RuntimeError(f"Outcome class mismatch: {key}")

        seed = int(r["seed"])
        fold = int(r["fold"])
        if seed not in SEEDS:
            raise RuntimeError(f"Unexpected seed: {seed}")
        if fold not in range(N_FOLDS):
            raise RuntimeError(f"Unexpected fold: {fold}")
        if r["perturbation"] not in PERTURBATIONS:
            raise RuntimeError(f"Unexpected perturbation: {r['perturbation']}")

        group_folds[r["source_group_id"]].add(fold)

        row = {
            "seed": seed,
            "sample_id": r["sample_id"],
            "source_group_id": r["source_group_id"],
            "fold": fold,
            "perturbation": r["perturbation"],
            "base_case_id": r["base_case_id"],
            "source_dice": safe_float(u["source_dice"]),
            "action_dice": safe_float(u["action_dice"]),
            "delta_dice": delta_f,
            "harmful": int(r["harmful"]),
            "neutral": int(r["neutral"]),
            "beneficial": int(r["beneficial"]),
            "outcome_class": outcome,
        }
        for f in SOURCE_FEATURE_NAMES:
            row[f] = safe_float(r[f])
        merged.append(row)

    if len(group_folds) != SOURCE_GROUPS:
        raise RuntimeError(
            f"Source-group count mismatch: {len(group_folds)} != {SOURCE_GROUPS}"
        )
    if any(len(v) != 1 for v in group_folds.values()):
        raise RuntimeError("Frozen source-group fold leakage detected.")

    # Validate matched units: same sample x perturbation, exactly 3 seeds.
    matched = defaultdict(list)
    for r in merged:
        matched[(r["sample_id"], r["perturbation"])].append(r)

    if len(matched) != MATCHED_UNITS:
        raise RuntimeError(
            f"{INVALID_MATCHED}: matched units={len(matched)} != {MATCHED_UNITS}"
        )

    for unit, rows_u in matched.items():
        seed_set = {int(r["seed"]) for r in rows_u}
        if len(rows_u) != 3 or seed_set != set(SEEDS):
            raise RuntimeError(
                f"{INVALID_MATCHED}: unit={unit} rows={len(rows_u)} seeds={seed_set}"
            )
        folds = {int(r["fold"]) for r in rows_u}
        groups = {r["source_group_id"] for r in rows_u}
        if len(folds) != 1 or len(groups) != 1:
            raise RuntimeError(
                f"{INVALID_MATCHED}: unit={unit} fold/group inconsistency"
            )

    return merged, matched, {
        "protocol_sha256": protocol_sha,
        "q1_s01_lock_sha256": q1_s01_sha,
        "q1_s00a_lock_sha256": q1_s00a_lock_sha,
        "q1_s00a_feature_sha256": feature_sha,
        "utility_table_sha256": utility_sha,
        "rows": len(merged),
        "source_groups": len(group_folds),
        "matched_units": len(matched),
        "seeds": list(SEEDS),
        "perturbations": list(PERTURBATIONS),
        "action": A1_ACTION,
        "target_data_used": False,
        "target_gt_used": False,
        "new_segmentation_inference": False,
        "gradient_features_used": False,
        "adapted_prediction_features_used": False,
        "hyperparameter_sweep": False,
        "feature_selection": False,
        "threshold_tuning": False,
    }


# ---------------------------------------------------------------------
# Analysis A/B/C: matched-unit outcome behavior
# ---------------------------------------------------------------------

def organize_matched_units(matched):
    units = []
    for (sample_id, perturbation), rows_u in sorted(matched.items()):
        by_seed = {int(r["seed"]): r for r in rows_u}
        deltas = [by_seed[s]["delta_dice"] for s in SEEDS]
        classes = [by_seed[s]["outcome_class"] for s in SEEDS]
        severe_bh = ("BENEFIT" in classes and "HARM" in classes)
        units.append({
            "sample_id": sample_id,
            "perturbation": perturbation,
            "source_group_id": by_seed[SEEDS[0]]["source_group_id"],
            "fold": by_seed[SEEDS[0]]["fold"],
            f"delta_{SEEDS[0]}": deltas[0],
            f"delta_{SEEDS[1]}": deltas[1],
            f"delta_{SEEDS[2]}": deltas[2],
            f"class_{SEEDS[0]}": classes[0],
            f"class_{SEEDS[1]}": classes[1],
            f"class_{SEEDS[2]}": classes[2],
            "exact_three_seed_class_agreement": len(set(classes)) == 1,
            "any_class_disagreement": len(set(classes)) > 1,
            "any_benefit_harm_flip": severe_bh,
            "delta_range": float(max(deltas) - min(deltas)),
        })
    return units


def pairwise_delta_consistency(matched):
    out = []
    for i, s1 in enumerate(SEEDS):
        for s2 in SEEDS[i + 1:]:
            x, y = [], []
            for rows_u in matched.values():
                by_seed = {int(r["seed"]): r for r in rows_u}
                x.append(by_seed[s1]["delta_dice"])
                y.append(by_seed[s2]["delta_dice"])
            x = np.asarray(x, dtype=np.float64)
            y = np.asarray(y, dtype=np.float64)
            abs_diff = np.abs(x - y)
            out.append({
                "seed_a": s1,
                "seed_b": s2,
                "rows": len(x),
                "delta_spearman": safe_spearman(x, y),
                "delta_pearson": safe_pearson(x, y),
                "delta_mae": float(np.mean(abs_diff)),
                "delta_median_abs_difference": float(np.median(abs_diff)),
                "delta_p90_abs_difference": float(np.quantile(abs_diff, 0.90)),
            })
    return out


def pairwise_class_agreement(matched):
    rows = []
    pairwise_agreements = []

    for i, s1 in enumerate(SEEDS):
        for s2 in SEEDS[i + 1:]:
            a, b = [], []
            transitions = Counter()
            for rows_u in matched.values():
                by_seed = {int(r["seed"]): r for r in rows_u}
                c1 = by_seed[s1]["outcome_class"]
                c2 = by_seed[s2]["outcome_class"]
                a.append(c1)
                b.append(c2)
                transitions[(c1, c2)] += 1

            a_arr = np.asarray(a, dtype=object)
            b_arr = np.asarray(b, dtype=object)
            agreement = float(np.mean(a_arr == b_arr))
            pairwise_agreements.append(agreement)

            benefit_mask = (a_arr == "BENEFIT") | (b_arr == "BENEFIT")
            harm_mask = (a_arr == "HARM") | (b_arr == "HARM")
            benefit_agreement = (
                float(np.mean(a_arr[benefit_mask] == b_arr[benefit_mask]))
                if np.any(benefit_mask) else float("nan")
            )
            harm_agreement = (
                float(np.mean(a_arr[harm_mask] == b_arr[harm_mask]))
                if np.any(harm_mask) else float("nan")
            )

            row = {
                "scope": "PAIR",
                "seed_a": s1,
                "seed_b": s2,
                "pairwise_agreement": agreement,
                "cohen_kappa": float(cohen_kappa_score(a, b)),
                "benefit_involved_agreement": benefit_agreement,
                "harm_involved_agreement": harm_agreement,
            }
            for c1 in ("BENEFIT", "NEUTRAL", "HARM"):
                for c2 in ("BENEFIT", "NEUTRAL", "HARM"):
                    row[f"transition_{c1}_to_{c2}"] = transitions[(c1, c2)]
            rows.append(row)

    # Overall 3-seed agreement.
    exact = 0
    for rows_u in matched.values():
        classes = [r["outcome_class"] for r in rows_u]
        exact += int(len(set(classes)) == 1)

    overall = {
        "scope": "OVERALL",
        "seed_a": "",
        "seed_b": "",
        "pairwise_agreement": float(np.mean(pairwise_agreements)),
        "cohen_kappa": "",
        "benefit_involved_agreement": "",
        "harm_involved_agreement": "",
        "exact_three_seed_agreement": exact / MATCHED_UNITS,
    }
    rows.append(overall)
    return rows


def sign_flip_summary(unit_rows):
    bh_count = sum(int(r["any_benefit_harm_flip"]) for r in unit_rows)
    disagreement_count = sum(int(r["any_class_disagreement"]) for r in unit_rows)
    ranges = np.asarray([r["delta_range"] for r in unit_rows], dtype=np.float64)
    return {
        "matched_units": len(unit_rows),
        "benefit_harm_flip_count": bh_count,
        "benefit_harm_flip_rate": bh_count / len(unit_rows),
        "any_class_disagreement_count": disagreement_count,
        "any_class_disagreement_rate": disagreement_count / len(unit_rows),
        "median_three_seed_delta_range": float(np.median(ranges)),
        "p90_three_seed_delta_range": float(np.quantile(ranges, 0.90)),
        "max_three_seed_delta_range": float(np.max(ranges)),
    }


# ---------------------------------------------------------------------
# Generic training/evaluation
# ---------------------------------------------------------------------

def train_predict(
    train_rows,
    test_rows,
    mode: str,
    feature_names: Sequence[str] = SOURCE_FEATURE_NAMES,
):
    x_train = matrix(train_rows, feature_names)
    x_test = matrix(test_rows, feature_names)

    if mode == "benefit":
        y = binary_target(train_rows, "beneficial")
        if len(np.unique(y)) < 2:
            raise RuntimeError("BENEFIT training split lacks both classes.")
        model = make_logistic()
        model.fit(x_train, y)
        return model.predict_proba(x_test)[:, 1], model
    if mode == "harm":
        y = binary_target(train_rows, "harmful")
        if len(np.unique(y)) < 2:
            raise RuntimeError("HARM training split lacks both classes.")
        model = make_logistic()
        model.fit(x_train, y)
        return model.predict_proba(x_test)[:, 1], model
    if mode == "delta":
        y = delta_target(train_rows)
        model = make_ridge()
        model.fit(x_train, y)
        return model.predict(x_test), model
    raise ValueError(mode)


def evaluate_predictions(test_rows, pred_delta, pred_harm, pred_benefit):
    yd = delta_target(test_rows)
    yh = binary_target(test_rows, "harmful")
    yb = binary_target(test_rows, "beneficial")
    return {
        "benefit_auroc": float(roc_auc_score(yb, pred_benefit)),
        "benefit_auprc": float(average_precision_score(yb, pred_benefit)),
        "harm_auroc": float(roc_auc_score(yh, pred_harm)),
        "harm_auprc": float(average_precision_score(yh, pred_harm)),
        "delta_dice_spearman": safe_spearman(yd, pred_delta),
        "delta_dice_mae": float(mean_absolute_error(yd, pred_delta)),
    }


def grouped_oof_for_seed(rows, seed: int):
    seed_rows = [r for r in rows if int(r["seed"]) == seed]
    if len(seed_rows) != MATCHED_UNITS:
        raise RuntimeError(f"Seed {seed} row count mismatch: {len(seed_rows)}")

    out = []
    for fold in range(N_FOLDS):
        train_rows = [r for r in seed_rows if int(r["fold"]) != fold]
        test_rows = [r for r in seed_rows if int(r["fold"]) == fold]

        train_groups = {r["source_group_id"] for r in train_rows}
        test_groups = {r["source_group_id"] for r in test_rows}
        if train_groups & test_groups:
            raise RuntimeError(f"Within-seed group leakage seed={seed} fold={fold}")

        pd, _ = train_predict(train_rows, test_rows, "delta")
        ph, _ = train_predict(train_rows, test_rows, "harm")
        pb, _ = train_predict(train_rows, test_rows, "benefit")

        for r, d, h, b in zip(test_rows, pd, ph, pb):
            out.append({
                **r,
                "pred_delta_dice": float(d),
                "pred_harm_probability": float(h),
                "pred_benefit_probability": float(b),
            })

    if len(out) != MATCHED_UNITS:
        raise RuntimeError(f"Incomplete within-seed OOF seed={seed}")
    return out, evaluate_predictions(
        out,
        [r["pred_delta_dice"] for r in out],
        [r["pred_harm_probability"] for r in out],
        [r["pred_benefit_probability"] for r in out],
    )


def cross_seed_transfer(rows, train_seed: int, test_seed: int):
    if train_seed == test_seed:
        raise ValueError("Use grouped_oof_for_seed for diagonal.")

    preds = []
    split_audit = []
    for fold in range(N_FOLDS):
        train_rows = [
            r for r in rows
            if int(r["seed"]) == train_seed and int(r["fold"]) != fold
        ]
        test_rows = [
            r for r in rows
            if int(r["seed"]) == test_seed and int(r["fold"]) == fold
        ]

        train_groups = {r["source_group_id"] for r in train_rows}
        test_groups = {r["source_group_id"] for r in test_rows}
        overlap = train_groups & test_groups
        if overlap:
            raise RuntimeError(
                f"Cross-seed leakage {train_seed}->{test_seed} fold={fold}"
            )

        pd, _ = train_predict(train_rows, test_rows, "delta")
        ph, _ = train_predict(train_rows, test_rows, "harm")
        pb, _ = train_predict(train_rows, test_rows, "benefit")

        for r, d, h, b in zip(test_rows, pd, ph, pb):
            preds.append({
                **r,
                "train_seed": train_seed,
                "test_seed": test_seed,
                "pred_delta_dice": float(d),
                "pred_harm_probability": float(h),
                "pred_benefit_probability": float(b),
            })

        split_audit.append({
            "train_seed": train_seed,
            "test_seed": test_seed,
            "fold": fold,
            "train_rows": len(train_rows),
            "test_rows": len(test_rows),
            "train_groups": len(train_groups),
            "test_groups": len(test_groups),
            "group_overlap": 0,
        })

    if len(preds) != MATCHED_UNITS:
        raise RuntimeError(
            f"Cross-seed prediction count mismatch {train_seed}->{test_seed}"
        )

    metrics = evaluate_predictions(
        preds,
        [r["pred_delta_dice"] for r in preds],
        [r["pred_harm_probability"] for r in preds],
        [r["pred_benefit_probability"] for r in preds],
    )
    return preds, metrics, split_audit


def two_seed_to_one_transfer(rows, held_out_seed: int):
    train_seeds = [s for s in SEEDS if s != held_out_seed]
    preds = []
    split_audit = []

    for fold in range(N_FOLDS):
        train_rows = [
            r for r in rows
            if int(r["seed"]) in train_seeds and int(r["fold"]) != fold
        ]
        test_rows = [
            r for r in rows
            if int(r["seed"]) == held_out_seed and int(r["fold"]) == fold
        ]

        train_groups = {r["source_group_id"] for r in train_rows}
        test_groups = {r["source_group_id"] for r in test_rows}
        if train_groups & test_groups:
            raise RuntimeError(
                f"2->1 group leakage held_out_seed={held_out_seed} fold={fold}"
            )
        if any(int(r["seed"]) == held_out_seed for r in train_rows):
            raise RuntimeError("Held-out seed leaked into 2->1 training.")

        pd, _ = train_predict(train_rows, test_rows, "delta")
        ph, _ = train_predict(train_rows, test_rows, "harm")
        pb, _ = train_predict(train_rows, test_rows, "benefit")

        for r, d, h, b in zip(test_rows, pd, ph, pb):
            preds.append({
                **r,
                "held_out_seed": held_out_seed,
                "train_seeds": "+".join(str(s) for s in train_seeds),
                "pred_delta_dice": float(d),
                "pred_harm_probability": float(h),
                "pred_benefit_probability": float(b),
            })

        split_audit.append({
            "held_out_seed": held_out_seed,
            "fold": fold,
            "train_seeds": train_seeds,
            "train_rows": len(train_rows),
            "test_rows": len(test_rows),
            "group_overlap": 0,
        })

    if len(preds) != MATCHED_UNITS:
        raise RuntimeError(f"Incomplete 2->1 transfer held_out_seed={held_out_seed}")

    return preds, evaluate_predictions(
        preds,
        [r["pred_delta_dice"] for r in preds],
        [r["pred_harm_probability"] for r in preds],
        [r["pred_benefit_probability"] for r in preds],
    ), split_audit


# ---------------------------------------------------------------------
# Analysis D: per-seed outcome distribution
# ---------------------------------------------------------------------

def per_seed_outcome_distribution(rows):
    out = []
    for seed in SEEDS:
        rr = [r for r in rows if int(r["seed"]) == seed]
        delta = np.asarray([r["delta_dice"] for r in rr], dtype=np.float64)
        harm = sum(r["harmful"] for r in rr)
        neutral = sum(r["neutral"] for r in rr)
        benefit = sum(r["beneficial"] for r in rr)
        out.append({
            "seed": seed,
            "rows": len(rr),
            "delta_mean": float(np.mean(delta)),
            "delta_median": float(np.median(delta)),
            "delta_std": float(np.std(delta)),
            "harm_count": harm,
            "harm_rate": harm / len(rr),
            "neutral_count": neutral,
            "neutral_rate": neutral / len(rr),
            "benefit_count": benefit,
            "benefit_rate": benefit / len(rr),
            "source_dice_mean": float(np.mean([r["source_dice"] for r in rr])),
            "action_dice_mean": float(np.mean([r["action_dice"] for r in rr])),
        })
    return out


# ---------------------------------------------------------------------
# Analysis H: feature-level stability
# ---------------------------------------------------------------------

def feature_cross_seed_stability(matched):
    rows = []
    for feature in tqdm(
        SOURCE_FEATURE_NAMES,
        desc="Q1-S02A feature cross-seed stability",
        unit="feature",
        dynamic_ncols=True,
    ):
        for i, s1 in enumerate(SEEDS):
            for s2 in SEEDS[i + 1:]:
                x, y = [], []
                for rows_u in matched.values():
                    by_seed = {int(r["seed"]): r for r in rows_u}
                    x.append(by_seed[s1][feature])
                    y.append(by_seed[s2][feature])
                x = np.asarray(x, dtype=np.float64)
                y = np.asarray(y, dtype=np.float64)

                pooled_std = float(np.sqrt((np.var(x) + np.var(y)) / 2.0))
                mean_diff = float(np.mean(x) - np.mean(y))
                smd = (
                    mean_diff / pooled_std
                    if pooled_std > 1e-15
                    else 0.0
                )

                rows.append({
                    "feature": feature,
                    "seed_a": s1,
                    "seed_b": s2,
                    "mean_a": float(np.mean(x)),
                    "mean_b": float(np.mean(y)),
                    "mean_difference_a_minus_b": mean_diff,
                    "standardized_mean_difference": float(smd),
                    "matched_spearman": safe_spearman(x, y),
                })
    return rows


def single_feature_per_seed(rows):
    out = []
    for seed in SEEDS:
        seed_rows = [r for r in rows if int(r["seed"]) == seed]
        for feature in tqdm(
            SOURCE_FEATURE_NAMES,
            desc=f"Q1-S02A single-feature seed {seed}",
            unit="feature",
            leave=False,
            dynamic_ncols=True,
        ):
            preds = []
            for fold in range(N_FOLDS):
                train_rows = [r for r in seed_rows if int(r["fold"]) != fold]
                test_rows = [r for r in seed_rows if int(r["fold"]) == fold]

                train_groups = {r["source_group_id"] for r in train_rows}
                test_groups = {r["source_group_id"] for r in test_rows}
                if train_groups & test_groups:
                    raise RuntimeError(
                        f"Single-feature group leakage seed={seed} feature={feature}"
                    )

                pb, _ = train_predict(
                    train_rows,
                    test_rows,
                    "benefit",
                    [feature],
                )
                for r, p in zip(test_rows, pb):
                    preds.append((int(r["beneficial"]), float(p)))

            y = [a for a, _ in preds]
            p = [b for _, b in preds]
            out.append({
                "seed": seed,
                "feature": feature,
                "benefit_auroc": float(roc_auc_score(y, p)),
            })
    return out


# ---------------------------------------------------------------------
# Analysis I: coefficient stability
# ---------------------------------------------------------------------

def fit_standardized_benefit_coefficients(seed_rows):
    x = matrix(seed_rows, SOURCE_FEATURE_NAMES)
    y = binary_target(seed_rows, "beneficial")

    scaler = StandardScaler()
    xs = scaler.fit_transform(x)

    clf = LogisticRegression(
        C=LOGISTIC_C,
        penalty="l2",
        solver="liblinear",
        class_weight="balanced",
        max_iter=LOGISTIC_MAX_ITER,
        random_state=RANDOM_STATE,
    )
    clf.fit(xs, y)
    coef = clf.coef_.reshape(-1).astype(np.float64)

    if len(coef) != len(SOURCE_FEATURE_NAMES):
        raise RuntimeError("Coefficient length mismatch.")
    if not np.isfinite(coef).all():
        raise RuntimeError("Non-finite standardized coefficients.")
    return coef


def coefficient_stability(rows):
    coeff = {}
    for seed in SEEDS:
        rr = [r for r in rows if int(r["seed"]) == seed]
        coeff[seed] = fit_standardized_benefit_coefficients(rr)

    out = []
    for i, s1 in enumerate(SEEDS):
        for s2 in SEEDS[i + 1:]:
            a = coeff[s1]
            b = coeff[s2]

            top5_a = set(np.argsort(np.abs(a))[-5:].tolist())
            top5_b = set(np.argsort(np.abs(b))[-5:].tolist())
            top10_a = set(np.argsort(np.abs(a))[-10:].tolist())
            top10_b = set(np.argsort(np.abs(b))[-10:].tolist())

            sign_a = np.sign(a)
            sign_b = np.sign(b)

            out.append({
                "seed_a": s1,
                "seed_b": s2,
                "coefficient_cosine": safe_cosine(a, b),
                "coefficient_pearson": safe_pearson(a, b),
                "coefficient_spearman": safe_spearman(a, b),
                "coefficient_sign_agreement": float(np.mean(sign_a == sign_b)),
                "top5_overlap_count": len(top5_a & top5_b),
                "top5_overlap_fraction": len(top5_a & top5_b) / 5.0,
                "top10_overlap_count": len(top10_a & top10_b),
                "top10_overlap_fraction": len(top10_a & top10_b) / 10.0,
            })

    coeff_rows = []
    for seed in SEEDS:
        for feature, value in zip(SOURCE_FEATURE_NAMES, coeff[seed]):
            coeff_rows.append({
                "seed": seed,
                "feature": feature,
                "standardized_logistic_coefficient": float(value),
                "abs_coefficient": float(abs(value)),
            })

    return out, coeff_rows


# ---------------------------------------------------------------------
# Transfer matrix serialization
# ---------------------------------------------------------------------

def matrix_csv(metric_name: str, metric_lookup: Dict[Tuple[int, int], Dict]):
    rows = []
    for train_seed in SEEDS:
        row = {"train_seed": train_seed}
        for test_seed in SEEDS:
            row[f"test_{test_seed}"] = metric_lookup[(train_seed, test_seed)][metric_name]
        rows.append(row)
    return rows


# ---------------------------------------------------------------------
# Mechanism decision
# ---------------------------------------------------------------------

def decide_mechanism(
    pairwise_delta_rows,
    sign_summary,
    within_metrics,
    transfer_metrics,
    coefficient_rows,
):
    median_delta_spearman = float(
        np.median([r["delta_spearman"] for r in pairwise_delta_rows])
    )
    bh_flip_rate = float(sign_summary["benefit_harm_flip_rate"])
    median_within_benefit = float(
        np.median([within_metrics[s]["benefit_auroc"] for s in SEEDS])
    )

    off_diag = [
        m["benefit_auroc"]
        for (train_seed, test_seed), m in transfer_metrics.items()
        if train_seed != test_seed
    ]
    median_cross_benefit = float(np.median(off_diag))
    median_coef_cosine = float(
        np.median([r["coefficient_cosine"] for r in coefficient_rows])
    )

    within_stable = median_within_benefit >= TH_WITHIN_SEED_BENEFIT
    cross_poor = median_cross_benefit < TH_CROSS_SEED_POOR
    outcome_unstable = (
        median_delta_spearman < TH_DELTA_STABLE
        or bh_flip_rate >= TH_BH_FLIP_MATERIAL
    )
    outcome_stable = (
        median_delta_spearman >= TH_DELTA_STABLE
        and bh_flip_rate < TH_BH_FLIP_MATERIAL
    )
    mapping_unstable = median_coef_cosine < TH_COEF_COSINE_STABLE

    mixed = (
        within_stable
        and cross_poor
        and outcome_unstable
        and mapping_unstable
    )
    mcac = (
        within_stable
        and cross_poor
        and outcome_unstable
    )
    mapping = (
        within_stable
        and cross_poor
        and outcome_stable
        and mapping_unstable
    )

    if not within_stable:
        decision = STOP_WITHIN
    elif mixed:
        decision = GO_MIXED
    elif mcac:
        decision = GO_MCAC
    elif mapping:
        decision = GO_MAPPING
    else:
        decision = STOP_UNRESOLVED

    criteria = {
        "median_pairwise_delta_spearman": {
            "value": median_delta_spearman,
            "stable_threshold": TH_DELTA_STABLE,
            "outcome_unstable_by_spearman": median_delta_spearman < TH_DELTA_STABLE,
        },
        "benefit_harm_flip_rate": {
            "value": bh_flip_rate,
            "material_threshold": TH_BH_FLIP_MATERIAL,
            "outcome_unstable_by_flip": bh_flip_rate >= TH_BH_FLIP_MATERIAL,
        },
        "median_within_seed_benefit_auroc": {
            "value": median_within_benefit,
            "threshold": TH_WITHIN_SEED_BENEFIT,
            "pass": within_stable,
        },
        "median_off_diagonal_cross_seed_benefit_auroc": {
            "value": median_cross_benefit,
            "poor_threshold": TH_CROSS_SEED_POOR,
            "cross_seed_transfer_poor": cross_poor,
        },
        "median_coefficient_cosine": {
            "value": median_coef_cosine,
            "stable_threshold": TH_COEF_COSINE_STABLE,
            "mapping_unstable": mapping_unstable,
        },
        "outcome_unstable": outcome_unstable,
        "outcome_stable": outcome_stable,
        "mapping_unstable": mapping_unstable,
        "mixed_state_satisfied": mixed,
        "mcac_state_satisfied": mcac,
        "mapping_state_satisfied": mapping,
    }
    return decision, criteria


# ---------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------

def _synthetic_rows():
    rng = np.random.default_rng(123)
    rows = []
    # Match the frozen production cardinality so production-only row-count
    # assertions are exercised by --self-test as well.
    n_groups = SOURCE_GROUPS
    perts = PERTURBATIONS

    for g in range(n_groups):
        fold = g % N_FOLDS
        base = rng.normal()
        for p_i, pert in enumerate(perts):
            case_effect = base + 0.2 * p_i + rng.normal(scale=0.2)
            for s_i, seed in enumerate(SEEDS):
                # deliberately create model-conditioned outcome
                model_interaction = (s_i - 1) * 0.5 * case_effect
                delta = 0.05 * case_effect + 0.05 * model_interaction + rng.normal(scale=0.03)
                cls = outcome_from_delta(delta)

                row = {
                    "seed": seed,
                    "sample_id": f"s{g:03d}",
                    "source_group_id": f"g{g:03d}",
                    "fold": fold,
                    "perturbation": pert,
                    "base_case_id": f"{seed}_s{g:03d}_{pert}",
                    "source_dice": float(np.clip(0.75 + 0.05 * base, 0, 1)),
                    "action_dice": float(np.clip(0.75 + 0.05 * base + delta, 0, 1)),
                    "delta_dice": float(delta),
                    "harmful": int(cls == "HARM"),
                    "neutral": int(cls == "NEUTRAL"),
                    "beneficial": int(cls == "BENEFIT"),
                    "outcome_class": cls,
                }
                for j, f in enumerate(SOURCE_FEATURE_NAMES):
                    row[f] = float(case_effect + 0.1 * s_i + rng.normal(scale=0.3 + 0.01*j))
                rows.append(row)
    return rows


def self_test():
    assert EXPECTED_ROWS == 4350
    assert SOURCE_GROUPS == 145
    assert MATCHED_UNITS == 1450
    assert len(SEEDS) == 3
    assert len(PERTURBATIONS) == 10
    assert len(SOURCE_FEATURE_NAMES) == 19
    assert A1_ACTION == "A1_TENT_1STEP"

    rows = _synthetic_rows()
    matched = defaultdict(list)
    for r in rows:
        matched[(r["sample_id"], r["perturbation"])].append(r)

    unit_rows = organize_matched_units(matched)
    assert len(unit_rows) == MATCHED_UNITS

    pair = pairwise_delta_consistency(matched)
    assert len(pair) == 3
    assert all(np.isfinite(r["delta_spearman"]) for r in pair)

    agree = pairwise_class_agreement(matched)
    assert len(agree) == 4

    sign = sign_flip_summary(unit_rows)
    assert 0 <= sign["benefit_harm_flip_rate"] <= 1

    dist = per_seed_outcome_distribution(rows)
    assert len(dist) == 3

    # A minimal grouped OOF check on one seed.
    oof, metrics = grouped_oof_for_seed(rows, SEEDS[0])
    assert len(oof) == MATCHED_UNITS
    assert 0 <= metrics["benefit_auroc"] <= 1
    assert 0 <= metrics["harm_auroc"] <= 1

    coef_stab, coef_rows = coefficient_stability(rows)
    assert len(coef_stab) == 3
    assert len(coef_rows) == len(SEEDS) * len(SOURCE_FEATURE_NAMES)

    print("FROZEN_CONSTANTS_TEST_PASS")
    print("MATCHED_UNIT_TEST_PASS")
    print("PAIRWISE_OUTCOME_TEST_PASS")
    print("GROUPED_OOF_TEST_PASS")
    print("COEFFICIENT_STABILITY_TEST_PASS")
    print("SELF_TEST_PASS")


# ---------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------

def run(args):
    if args.output_dir.exists():
        raise FileExistsError(f"Q1-S02A output already exists: {args.output_dir}")

    build_dir = Path(str(args.output_dir) + "__building")
    if build_dir.exists():
        raise FileExistsError(
            f"Partial Q1-S02A build exists: {build_dir}. "
            "Remove only this __building directory if a prior technical run failed."
        )
    build_dir.mkdir(parents=True, exist_ok=False)

    print("===== Q1-S02A MODEL-CASE INTERACTION MECHANISM AUDIT =====")
    print(f"Script version: {VERSION}")
    print(f"Build: {BUILD}")
    print("New segmentation inference: NO")
    print("Target data used: NO")
    print("Target GT used: NO")
    print("Gradient features used: NO")
    print("Adapted prediction features used: NO")
    print("Hyperparameter sweep: NO")
    print("Feature selection: NO")
    print("Threshold tuning: NO")
    print()

    rows, matched, provenance = load_and_validate_inputs()

    protocol_copy = build_dir / "preregistered_protocol_copy.md"
    shutil.copy2(PROTOCOL, protocol_copy)
    if file_sha256(protocol_copy) != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("Protocol copy SHA mismatch.")

    provenance_path = build_dir / "provenance_audit.json"
    write_json(provenance_path, provenance)

    # A/B/C
    matched_unit_rows = organize_matched_units(matched)
    matched_unit_path = build_dir / "matched_unit_audit.csv"
    write_csv(
        matched_unit_path,
        matched_unit_rows,
        list(matched_unit_rows[0].keys()),
    )

    pairwise_delta_rows = pairwise_delta_consistency(matched)
    pairwise_delta_path = build_dir / "pairwise_delta_consistency.csv"
    write_csv(
        pairwise_delta_path,
        pairwise_delta_rows,
        list(pairwise_delta_rows[0].keys()),
    )

    class_agreement_rows = pairwise_class_agreement(matched)
    class_agreement_fields = sorted(
        set().union(*(r.keys() for r in class_agreement_rows))
    )
    class_agreement_path = build_dir / "outcome_class_agreement.csv"
    write_csv(
        class_agreement_path,
        class_agreement_rows,
        class_agreement_fields,
    )

    sign_summary = sign_flip_summary(matched_unit_rows)
    sign_path = build_dir / "sign_flip_audit.csv"
    write_csv(
        sign_path,
        matched_unit_rows,
        [
            "sample_id",
            "perturbation",
            "source_group_id",
            "fold",
            f"delta_{SEEDS[0]}",
            f"delta_{SEEDS[1]}",
            f"delta_{SEEDS[2]}",
            f"class_{SEEDS[0]}",
            f"class_{SEEDS[1]}",
            f"class_{SEEDS[2]}",
            "exact_three_seed_class_agreement",
            "any_class_disagreement",
            "any_benefit_harm_flip",
            "delta_range",
        ],
    )

    # D
    per_seed_dist = per_seed_outcome_distribution(rows)
    per_seed_dist_path = build_dir / "per_seed_outcome_distribution.csv"
    write_csv(
        per_seed_dist_path,
        per_seed_dist,
        list(per_seed_dist[0].keys()),
    )

    # E within-seed OOF
    within_metrics = {}
    within_rows = []
    within_predictions = {}
    for seed in SEEDS:
        pred, met = grouped_oof_for_seed(rows, seed)
        within_predictions[seed] = pred
        within_metrics[seed] = met
        within_rows.append({"seed": seed, **met})

    within_path = build_dir / "within_seed_oof_metrics.csv"
    write_csv(
        within_path,
        within_rows,
        list(within_rows[0].keys()),
    )

    # F full 3x3 transfer
    transfer_metrics = {}
    transfer_split_audit = []
    transfer_predictions = []

    for train_seed in SEEDS:
        for test_seed in SEEDS:
            if train_seed == test_seed:
                transfer_metrics[(train_seed, test_seed)] = within_metrics[train_seed]
                continue

            pred, met, split_audit = cross_seed_transfer(
                rows,
                train_seed,
                test_seed,
            )
            transfer_metrics[(train_seed, test_seed)] = met
            transfer_split_audit.extend(split_audit)
            transfer_predictions.extend(pred)

    benefit_matrix = matrix_csv("benefit_auroc", transfer_metrics)
    harm_matrix = matrix_csv("harm_auroc", transfer_metrics)
    delta_matrix = matrix_csv("delta_dice_spearman", transfer_metrics)

    benefit_matrix_path = build_dir / "cross_seed_transfer_matrix_benefit.csv"
    harm_matrix_path = build_dir / "cross_seed_transfer_matrix_harm.csv"
    delta_matrix_path = build_dir / "cross_seed_transfer_matrix_delta.csv"

    matrix_fields = ["train_seed"] + [f"test_{s}" for s in SEEDS]
    write_csv(benefit_matrix_path, benefit_matrix, matrix_fields)
    write_csv(harm_matrix_path, harm_matrix, matrix_fields)
    write_csv(delta_matrix_path, delta_matrix, matrix_fields)

    # G 2 seeds -> 1
    two_to_one_rows = []
    two_to_one_split_audit = []
    for held_out_seed in SEEDS:
        _, met, split_audit = two_seed_to_one_transfer(rows, held_out_seed)
        two_to_one_rows.append({
            "held_out_seed": held_out_seed,
            "train_seeds": "+".join(str(s) for s in SEEDS if s != held_out_seed),
            **met,
        })
        two_to_one_split_audit.extend(split_audit)

    two_to_one_path = build_dir / "two_seed_to_one_transfer.csv"
    write_csv(
        two_to_one_path,
        two_to_one_rows,
        list(two_to_one_rows[0].keys()),
    )

    # H feature stability + single feature per seed
    feature_stability_rows = feature_cross_seed_stability(matched)
    feature_stability_path = build_dir / "feature_cross_seed_stability.csv"
    write_csv(
        feature_stability_path,
        feature_stability_rows,
        list(feature_stability_rows[0].keys()),
    )

    single_feature_rows = single_feature_per_seed(rows)
    single_feature_path = build_dir / "single_feature_per_seed_auroc.csv"
    write_csv(
        single_feature_path,
        single_feature_rows,
        list(single_feature_rows[0].keys()),
    )

    # I coefficient stability
    coef_stability_rows, coef_value_rows = coefficient_stability(rows)
    coef_stability_path = build_dir / "coefficient_stability.csv"
    coef_fields = list(coef_stability_rows[0].keys())
    write_csv(coef_stability_path, coef_stability_rows, coef_fields)

    coef_values_path = build_dir / "coefficient_values_by_seed.csv"
    write_csv(
        coef_values_path,
        coef_value_rows,
        list(coef_value_rows[0].keys()),
    )

    # J mechanism decision
    decision, mechanism = decide_mechanism(
        pairwise_delta_rows,
        sign_summary,
        within_metrics,
        transfer_metrics,
        coef_stability_rows,
    )

    mechanism_rows = [
        {
            "criterion": "median_pairwise_delta_spearman",
            "value": mechanism["median_pairwise_delta_spearman"]["value"],
            "threshold": TH_DELTA_STABLE,
            "interpretation": "outcome unstable if < threshold",
        },
        {
            "criterion": "benefit_harm_flip_rate",
            "value": mechanism["benefit_harm_flip_rate"]["value"],
            "threshold": TH_BH_FLIP_MATERIAL,
            "interpretation": "outcome unstable if >= threshold",
        },
        {
            "criterion": "median_within_seed_benefit_auroc",
            "value": mechanism["median_within_seed_benefit_auroc"]["value"],
            "threshold": TH_WITHIN_SEED_BENEFIT,
            "interpretation": "stable within-seed signal if >= threshold",
        },
        {
            "criterion": "median_off_diagonal_cross_seed_benefit_auroc",
            "value": mechanism["median_off_diagonal_cross_seed_benefit_auroc"]["value"],
            "threshold": TH_CROSS_SEED_POOR,
            "interpretation": "cross-seed transfer poor if < threshold",
        },
        {
            "criterion": "median_coefficient_cosine",
            "value": mechanism["median_coefficient_cosine"]["value"],
            "threshold": TH_COEF_COSINE_STABLE,
            "interpretation": "mapping unstable if < threshold",
        },
    ]
    mechanism_path = build_dir / "mechanism_criteria.csv"
    write_csv(
        mechanism_path,
        mechanism_rows,
        ["criterion", "value", "threshold", "interpretation"],
    )

    decision_path = build_dir / "decision.txt"
    decision_path.write_text(decision + "\n", encoding="utf-8")

    # Positive-polarity integrity checks: True means PASS.
    integrity = {
        "script_version": VERSION,
        "build": BUILD,
        "checks": {
            "rows_exact_4350": len(rows) == EXPECTED_ROWS,
            "source_groups_exact_145": len({r["source_group_id"] for r in rows}) == SOURCE_GROUPS,
            "perturbations_exact_10": set(r["perturbation"] for r in rows) == set(PERTURBATIONS),
            "seeds_exact_3": set(int(r["seed"]) for r in rows) == set(SEEDS),
            "matched_units_exact_1450": len(matched) == MATCHED_UNITS,
            "every_matched_unit_has_exactly_3_seeds": all(
                len(v) == 3 and {int(r["seed"]) for r in v} == set(SEEDS)
                for v in matched.values()
            ),
            "action_exact_A1_TENT_1STEP": provenance["action"] == A1_ACTION,
            "q1_s01_lock_verified": provenance["q1_s01_lock_sha256"] == EXPECTED_Q1_S01_LOCK_SHA256,
            "q1_s01_decision_verified": True,
            "target_data_not_loaded": True,
            "target_gt_not_loaded": True,
            "new_segmentation_inference_not_run": True,
            "gradient_features_not_used": True,
            "adapted_prediction_features_not_used": True,
            "source_group_overlap_zero_cross_seed": all(
                item["group_overlap"] == 0
                for item in transfer_split_audit
            ),
            "source_group_overlap_zero_two_to_one": all(
                item["group_overlap"] == 0
                for item in two_to_one_split_audit
            ),
            "standardizers_fit_training_only": True,
            "hyperparameter_sweep_not_used": True,
            "feature_selection_not_used": True,
            "threshold_tuning_not_used": True,
            "all_primary_statistics_finite": all(
                math.isfinite(float(x))
                for x in [
                    mechanism["median_pairwise_delta_spearman"]["value"],
                    mechanism["benefit_harm_flip_rate"]["value"],
                    mechanism["median_within_seed_benefit_auroc"]["value"],
                    mechanism["median_off_diagonal_cross_seed_benefit_auroc"]["value"],
                    mechanism["median_coefficient_cosine"]["value"],
                ]
            ),
        },
        "mechanism": mechanism,
        "decision": decision,
    }

    failed_integrity = [k for k, v in integrity["checks"].items() if not v]
    if failed_integrity:
        raise RuntimeError(f"Q1-S02A integrity failure: {failed_integrity}")

    integrity_path = build_dir / "leakage_integrity_audit.json"
    write_json(integrity_path, integrity)

    # Human-readable summary.
    median_pair_delta = mechanism["median_pairwise_delta_spearman"]["value"]
    bh_flip = mechanism["benefit_harm_flip_rate"]["value"]
    median_within = mechanism["median_within_seed_benefit_auroc"]["value"]
    median_cross = mechanism["median_off_diagonal_cross_seed_benefit_auroc"]["value"]
    median_coef = mechanism["median_coefficient_cosine"]["value"]

    summary = f"""===== Q1-S02A MODEL-CASE INTERACTION MECHANISM AUDIT =====
Script version: {VERSION}
Build: {BUILD}

Frozen data:
  rows={EXPECTED_ROWS}
  source groups={SOURCE_GROUPS}
  matched sample×perturbation units={MATCHED_UNITS}
  perturbations={len(PERTURBATIONS)}
  seeds={SEEDS}
  action={A1_ACTION}
  new segmentation inference=NO
  target data used=NO
  target GT used=NO
  gradient features used=NO
  adapted prediction features used=NO
  hyperparameter sweep=NO
  feature selection=NO
  threshold tuning=NO

[A — SAME-CASE CROSS-SEED DELTADICE]
"""
    for r in pairwise_delta_rows:
        summary += (
            f"  {r['seed_a']} vs {r['seed_b']}: "
            f"Spearman={r['delta_spearman']:.6f} "
            f"Pearson={r['delta_pearson']:.6f} "
            f"MAE={r['delta_mae']:.6f}\n"
        )
    summary += f"""  Median pairwise DeltaDice Spearman={median_pair_delta:.6f}
  Stable threshold >= {TH_DELTA_STABLE:.2f}

[B/C — CLASS AGREEMENT / SIGN FLIP]
  Exact three-seed class agreement={next(r for r in class_agreement_rows if r['scope']=='OVERALL')['exact_three_seed_agreement']:.6f}
  Any class disagreement rate={sign_summary['any_class_disagreement_rate']:.6f}
  BENEFIT<->HARM flip rate={bh_flip:.6f}
  Material threshold >= {TH_BH_FLIP_MATERIAL:.2f}
  Median three-seed DeltaDice range={sign_summary['median_three_seed_delta_range']:.6f}
  P90 three-seed DeltaDice range={sign_summary['p90_three_seed_delta_range']:.6f}

[D — PER-SEED OUTCOME DISTRIBUTION]
"""
    for r in per_seed_dist:
        summary += (
            f"  seed={r['seed']}: "
            f"meanDelta={r['delta_mean']:+.6f} "
            f"HARM={r['harm_rate']:.4f} "
            f"NEUTRAL={r['neutral_rate']:.4f} "
            f"BENEFIT={r['benefit_rate']:.4f}\n"
        )

    summary += "\n[E — WITHIN-SEED GROUPED OOF]\n"
    for seed in SEEDS:
        m = within_metrics[seed]
        summary += (
            f"  seed={seed}: BENEFIT_AUROC={m['benefit_auroc']:.6f} "
            f"HARM_AUROC={m['harm_auroc']:.6f} "
            f"DeltaSpearman={m['delta_dice_spearman']:.6f}\n"
        )
    summary += (
        f"  Median within-seed BENEFIT AUROC={median_within:.6f}\n"
        f"  Required stable within-seed signal >= {TH_WITHIN_SEED_BENEFIT:.2f}\n"
    )

    summary += "\n[F — FULL CROSS-SEED BENEFIT TRANSFER]\n"
    for train_seed in SEEDS:
        vals = []
        for test_seed in SEEDS:
            vals.append(
                f"test{test_seed}={transfer_metrics[(train_seed,test_seed)]['benefit_auroc']:.6f}"
            )
        summary += f"  train{train_seed}: " + " | ".join(vals) + "\n"
    summary += (
        f"  Median off-diagonal BENEFIT AUROC={median_cross:.6f}\n"
        f"  Cross-seed considered poor if < {TH_CROSS_SEED_POOR:.2f}\n"
    )

    summary += "\n[G — TWO-SEEDS-TO-ONE]\n"
    for r in two_to_one_rows:
        summary += (
            f"  holdout={r['held_out_seed']}: "
            f"BENEFIT_AUROC={r['benefit_auroc']:.6f} "
            f"HARM_AUROC={r['harm_auroc']:.6f} "
            f"DeltaSpearman={r['delta_dice_spearman']:.6f}\n"
        )

    summary += "\n[I — COEFFICIENT STABILITY]\n"
    for r in coef_stability_rows:
        summary += (
            f"  {r['seed_a']} vs {r['seed_b']}: "
            f"cosine={r['coefficient_cosine']:.6f} "
            f"sign_agreement={r['coefficient_sign_agreement']:.6f} "
            f"top5_overlap={r['top5_overlap_count']}/5\n"
        )
    summary += (
        f"  Median coefficient cosine={median_coef:.6f}\n"
        f"  Mapping considered stable if >= {TH_COEF_COSINE_STABLE:.2f}\n"
    )

    summary += f"""
[J — MECHANISM CLASSIFICATION]
  outcome_unstable={mechanism['outcome_unstable']}
  outcome_stable={mechanism['outcome_stable']}
  mapping_unstable={mechanism['mapping_unstable']}
  mixed_state_satisfied={mechanism['mixed_state_satisfied']}
  mcac_state_satisfied={mechanism['mcac_state_satisfied']}
  mapping_state_satisfied={mechanism['mapping_state_satisfied']}

Decision: {decision}

Decision meanings:
  {GO_MCAC}
    -> adaptation utility behaves as a model-case interaction property.

  {GO_MAPPING}
    -> outcome is relatively stable, but the predictor mapping is checkpoint-dependent.

  {GO_MIXED}
    -> both true outcome and predictor mapping vary materially with checkpoint.

  {STOP_WITHIN}
    -> no stable within-checkpoint forecasting signal.

  {STOP_UNRESOLVED}
    -> preregistered mechanism states are not cleanly resolved.

[OK] Outputs: {args.output_dir}
"""
    run_log_path = build_dir / "run_log.txt"
    run_log_path.write_text(summary, encoding="utf-8")

    # Lock.
    artifact_paths = {
        "protocol_copy": protocol_copy,
        "provenance_audit": provenance_path,
        "matched_unit_audit": matched_unit_path,
        "pairwise_delta_consistency": pairwise_delta_path,
        "outcome_class_agreement": class_agreement_path,
        "sign_flip_audit": sign_path,
        "per_seed_outcome_distribution": per_seed_dist_path,
        "within_seed_oof_metrics": within_path,
        "cross_seed_transfer_matrix_benefit": benefit_matrix_path,
        "cross_seed_transfer_matrix_harm": harm_matrix_path,
        "cross_seed_transfer_matrix_delta": delta_matrix_path,
        "two_seed_to_one_transfer": two_to_one_path,
        "feature_cross_seed_stability": feature_stability_path,
        "single_feature_per_seed_auroc": single_feature_path,
        "coefficient_stability": coef_stability_path,
        "coefficient_values_by_seed": coef_values_path,
        "mechanism_criteria": mechanism_path,
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
        "q1_s01_lock_sha256": EXPECTED_Q1_S01_LOCK_SHA256,
        "utility_table_sha256": EXPECTED_UTILITY_TABLE_SHA256,
        "target_data_used": False,
        "target_gt_used": False,
        "new_segmentation_inference": False,
        "gradient_features_used": False,
        "adapted_prediction_features_used": False,
        "hyperparameter_sweep": False,
        "feature_selection": False,
        "threshold_tuning": False,
        "mechanism": mechanism,
        "decision": decision,
    }

    lock_path = build_dir / "Q1_S02A_MODEL_CASE_INTERACTION_LOCK.json"
    write_json(lock_path, lock)
    lock_sha = file_sha256(lock_path)

    for name, path in artifact_paths.items():
        expected = lock["artifacts"][name]["sha256"]
        actual = file_sha256(path)
        if expected != actual:
            raise RuntimeError(f"Artifact changed before final commit: {name}")

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    build_dir.rename(args.output_dir)

    print()
    print((args.output_dir / "run_log.txt").read_text(encoding="utf-8"))
    print(
        "Q1-S02A LOCK:",
        args.output_dir / "Q1_S02A_MODEL_CASE_INTERACTION_LOCK.json",
    )
    print("Q1-S02A LOCK SHA256:", lock_sha)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run Q1-S02A model-case interaction mechanism audit using "
            "frozen source-side Q1-S00A/Q1-S01 assets."
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
