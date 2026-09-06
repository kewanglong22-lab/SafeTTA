#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Q1-S02C — Checkpoint Adaptation-Regime Audit

Frozen scientific question
--------------------------
Is the seed-20260819 adaptation-response shift observable BEFORE adaptation
in the frozen 19-dimensional PRE-SOURCE representation, and does that
pre-adaptation model-state shift explain seed19-specific future utility
residual beyond a source-quality-only explanation?

This script:
- reads frozen source-side assets only;
- performs NO new segmentation inference;
- uses NO target-domain data / target GT;
- uses NO gradient feature;
- uses NO adapted-prediction feature;
- performs NO hyperparameter sweep / feature selection / threshold tuning.

Expected project root
---------------------
F:\\MEDSEG_SAFETTA

Typical usage
-------------
python .\\Q1_S02C_checkpoint_adaptation_regime_audit_v1.py --self-test
python .\\Q1_S02C_checkpoint_adaptation_regime_audit_v1.py
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
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


VERSION = "2026-08-19-Q1-S02C-v1"
BUILD = "Q1_S02C_CHECKPOINT_ADAPTATION_REGIME_AUDIT"

ROOT = Path(r"F:\MEDSEG_SAFETTA")

PROTOCOL = (
    ROOT
    / "docs"
    / "Q1_S02C_checkpoint_adaptation_regime_audit_preregistered_protocol_v1.md"
)
EXPECTED_PROTOCOL_SHA256 = (
    "44ac72a46a0bf81035cb30c5ad9dcbd08b449b71bb51bd7ec4135fdeed047f76"
)

Q1_S02A_DIR = (
    ROOT
    / "outputs"
    / "Q1_S02A_model_case_interaction_mechanism_audit_v1"
)
Q1_S02A_LOCK = Q1_S02A_DIR / "Q1_S02A_MODEL_CASE_INTERACTION_LOCK.json"
EXPECTED_Q1_S02A_LOCK_SHA256 = (
    "e360a433cd77a04f17974d7023763f1e1f22e54746e011d5b18fb99d18595559"
)
EXPECTED_Q1_S02A_DECISION = "STOP_Q1_S02A_MECHANISM_UNRESOLVED"

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
    / "Q1_S02C_checkpoint_adaptation_regime_audit_v1"
)

SOURCE_GROUPS = 145
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
SEEDS = (20260817, 20260818, 20260819)
BASE_SEEDS = (20260817, 20260818)
REGIME_SEED = 20260819
MATCHED_UNITS = 1450
EXPECTED_ROWS = 4350
N_FOLDS = 5
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

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_RNG_SEED = 20260819

RIDGE_ALPHA = 1.0
LOGISTIC_C = 1.0
LOGISTIC_MAX_ITER = 5000
MODEL_RANDOM_STATE = 20260819

TH_DISTANCE_RATIO = 1.15
TH_RESIDUAL_SPEARMAN = 0.20
TH_DIVERGENCE_AUROC = 0.65
TH_QUALITY_INCREMENT = 0.05
MIN_DIVERGENCE_POSITIVES = 30

INVALID_MATCHED = "INVALID_Q1_S02C_MATCHED_UNIT_FAILURE"
STOP_NO_REGIME = "STOP_NO_OBSERVABLE_CHECKPOINT_REGIME"
STOP_UNEXPLAINED = "STOP_REGIME_VISIBLE_BUT_UTILITY_UNEXPLAINED"
GO_REGIME = "GO_OBSERVABLE_CHECKPOINT_ADAPTATION_REGIME"


# ---------------------------------------------------------------------
# General utilities
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


def write_csv(path: Path, rows: Sequence[dict], fields: Sequence[str]):
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
        raise RuntimeError(f"Non-finite numeric value: {v}")
    return x


def safe_spearman(x, y) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2 or np.std(x) < 1e-15 or np.std(y) < 1e-15:
        return 0.0
    value = spearmanr(x, y).statistic
    if value is None or not np.isfinite(value):
        return 0.0
    return float(value)


def paired_smd(x, y) -> float:
    """Paired standardized mean difference: mean(x-y) / sd(x-y)."""
    d = np.asarray(x, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    sd = float(np.std(d, ddof=1)) if len(d) > 1 else 0.0
    if sd <= 1e-15:
        return 0.0
    return float(np.mean(d) / sd)


def outcome_from_delta(delta: float) -> str:
    if delta <= -0.02:
        return "HARM"
    if delta >= +0.02:
        return "BENEFIT"
    return "NEUTRAL"


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
                random_state=MODEL_RANDOM_STATE,
            ),
        ),
    ])


# ---------------------------------------------------------------------
# Frozen input validation
# ---------------------------------------------------------------------

def load_frozen_rows():
    protocol_sha = validate_sha(
        PROTOCOL,
        EXPECTED_PROTOCOL_SHA256,
        "Q1-S02C protocol",
    )
    q1_s02a_sha = validate_sha(
        Q1_S02A_LOCK,
        EXPECTED_Q1_S02A_LOCK_SHA256,
        "Q1-S02A lock",
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

    q1_s02a_lock = json.loads(Q1_S02A_LOCK.read_text(encoding="utf-8"))
    if q1_s02a_lock.get("decision") != EXPECTED_Q1_S02A_DECISION:
        raise RuntimeError(
            "Unexpected Q1-S02A decision: "
            f"{q1_s02a_lock.get('decision')}"
        )
    if bool(q1_s02a_lock.get("target_data_used", True)):
        raise RuntimeError("Q1-S02A lock reports target-data use.")
    if bool(q1_s02a_lock.get("target_gt_used", True)):
        raise RuntimeError("Q1-S02A lock reports target-GT use.")
    if bool(q1_s02a_lock.get("new_segmentation_inference", True)):
        raise RuntimeError("Q1-S02A lock reports new inference.")

    q1_s00a_lock = json.loads(Q1_S00A_LOCK.read_text(encoding="utf-8"))
    feature_art = q1_s00a_lock.get("artifacts", {}).get("gradient_probe_features")
    if not feature_art or "sha256" not in feature_art:
        raise RuntimeError("Q1-S00A lock lacks gradient_probe_features SHA.")
    feature_sha = validate_sha(
        Q1_S00A_FEATURES,
        feature_art["sha256"],
        "Q1-S00A frozen feature table",
    )

    feature_rows, feature_fields = read_csv(Q1_S00A_FEATURES)
    utility_rows, utility_fields = read_csv(UTILITY_TABLE)

    required_feature_fields = {
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
    missing = sorted(required_feature_fields - set(feature_fields))
    if missing:
        raise RuntimeError(f"Frozen feature table missing columns: {missing}")

    required_utility_fields = {
        "seed",
        "sample_id",
        "perturbation",
        "action",
        "source_dice",
        "action_dice",
        "delta_dice",
    }
    missing_u = sorted(required_utility_fields - set(utility_fields))
    if missing_u:
        raise RuntimeError(f"Utility table missing columns: {missing_u}")

    if len(feature_rows) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Feature rows changed: expected={EXPECTED_ROWS} actual={len(feature_rows)}"
        )

    a1_rows = [r for r in utility_rows if r["action"] == A1_ACTION]
    if len(a1_rows) != EXPECTED_ROWS:
        raise RuntimeError(
            f"A1 rows changed: expected={EXPECTED_ROWS} actual={len(a1_rows)}"
        )

    utility_map = {}
    for r in a1_rows:
        key = (int(r["seed"]), r["sample_id"], r["perturbation"])
        if key in utility_map:
            raise RuntimeError(f"Duplicate A1 utility key: {key}")
        utility_map[key] = r

    merged = []
    seen = set()
    group_folds = defaultdict(set)

    for r in feature_rows:
        key = (int(r["seed"]), r["sample_id"], r["perturbation"])
        if key in seen:
            raise RuntimeError(f"Duplicate frozen feature key: {key}")
        seen.add(key)

        if key not in utility_map:
            raise RuntimeError(f"Missing utility match: {key}")
        u = utility_map[key]

        seed = int(r["seed"])
        fold = int(r["fold"])
        perturbation = r["perturbation"]
        delta = safe_float(r["delta_dice"])

        if seed not in SEEDS:
            raise RuntimeError(f"Unexpected seed: {seed}")
        if fold not in range(N_FOLDS):
            raise RuntimeError(f"Unexpected fold: {fold}")
        if perturbation not in PERTURBATIONS:
            raise RuntimeError(f"Unexpected perturbation: {perturbation}")
        if abs(delta - safe_float(u["delta_dice"])) > 1e-12:
            raise RuntimeError(f"DeltaDice mismatch: {key}")
        if outcome_from_delta(delta) != r["outcome_class"]:
            raise RuntimeError(f"Outcome label mismatch: {key}")

        group_folds[r["source_group_id"]].add(fold)

        row = {
            "seed": seed,
            "sample_id": r["sample_id"],
            "source_group_id": r["source_group_id"],
            "fold": fold,
            "perturbation": perturbation,
            "base_case_id": r["base_case_id"],
            "source_dice": safe_float(u["source_dice"]),
            "action_dice": safe_float(u["action_dice"]),
            "delta_dice": delta,
            "harmful": int(r["harmful"]),
            "neutral": int(r["neutral"]),
            "beneficial": int(r["beneficial"]),
            "outcome_class": r["outcome_class"],
        }
        for name in SOURCE_FEATURE_NAMES:
            row[name] = safe_float(r[name])
        merged.append(row)

    if len(group_folds) != SOURCE_GROUPS:
        raise RuntimeError(
            f"Source-group count changed: {len(group_folds)} != {SOURCE_GROUPS}"
        )
    if any(len(folds) != 1 for folds in group_folds.values()):
        raise RuntimeError("Frozen source-group fold assignment is inconsistent.")

    matched = defaultdict(list)
    for r in merged:
        matched[(r["sample_id"], r["perturbation"])].append(r)

    if len(matched) != MATCHED_UNITS:
        raise RuntimeError(
            f"{INVALID_MATCHED}: expected={MATCHED_UNITS} actual={len(matched)}"
        )

    for unit, rr in matched.items():
        seeds = {int(r["seed"]) for r in rr}
        groups = {r["source_group_id"] for r in rr}
        folds = {int(r["fold"]) for r in rr}
        if len(rr) != 3 or seeds != set(SEEDS):
            raise RuntimeError(
                f"{INVALID_MATCHED}: unit={unit} rows={len(rr)} seeds={seeds}"
            )
        if len(groups) != 1 or len(folds) != 1:
            raise RuntimeError(
                f"{INVALID_MATCHED}: unit={unit} group/fold mismatch"
            )

    provenance = {
        "script_version": VERSION,
        "build": BUILD,
        "protocol_sha256": protocol_sha,
        "q1_s02a_lock_sha256": q1_s02a_sha,
        "q1_s02a_decision": q1_s02a_lock["decision"],
        "q1_s00a_lock_sha256": q1_s00a_sha,
        "q1_s00a_feature_sha256": feature_sha,
        "utility_table_sha256": utility_sha,
        "rows": len(merged),
        "matched_units": len(matched),
        "source_groups": len(group_folds),
        "perturbations": list(PERTURBATIONS),
        "seeds": list(SEEDS),
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

    # Preserve frozen feature-table row order; matched analyses below explicitly
    # sort by unit key only when constructing paired units.
    return merged, matched, provenance


# ---------------------------------------------------------------------
# Matched-unit table
# ---------------------------------------------------------------------

def build_units(matched):
    units = []
    for (sample_id, perturbation), rr in sorted(matched.items()):
        by_seed = {int(r["seed"]): r for r in rr}
        row = {
            "sample_id": sample_id,
            "perturbation": perturbation,
            "source_group_id": by_seed[SEEDS[0]]["source_group_id"],
            "fold": int(by_seed[SEEDS[0]]["fold"]),
        }
        for s in SEEDS:
            row[f"delta_{s}"] = float(by_seed[s]["delta_dice"])
            row[f"source_dice_{s}"] = float(by_seed[s]["source_dice"])
            row[f"action_dice_{s}"] = float(by_seed[s]["action_dice"])
            row[f"class_{s}"] = by_seed[s]["outcome_class"]
            for f in SOURCE_FEATURE_NAMES:
                row[f"{f}__{s}"] = float(by_seed[s][f])
        units.append(row)

    if len(units) != MATCHED_UNITS:
        raise RuntimeError(
            f"{INVALID_MATCHED}: built units={len(units)} != {MATCHED_UNITS}"
        )
    return units


# ---------------------------------------------------------------------
# Analysis A/B: representation distances + cluster bootstrap
# ---------------------------------------------------------------------

def pooled_standardization(units):
    all_vectors = []
    for u in units:
        for s in SEEDS:
            all_vectors.append([u[f"{f}__{s}"] for f in SOURCE_FEATURE_NAMES])
    x = np.asarray(all_vectors, dtype=np.float64)
    mu = np.mean(x, axis=0)
    sd = np.std(x, axis=0)
    sd = np.where(sd <= 1e-12, 1.0, sd)
    return mu, sd


def standardized_vector(unit, seed, mu, sd):
    x = np.asarray(
        [unit[f"{f}__{seed}"] for f in SOURCE_FEATURE_NAMES],
        dtype=np.float64,
    )
    return (x - mu) / sd


def compute_pairwise_distance_rows(units):
    mu, sd = pooled_standardization(units)
    pairs = [
        (SEEDS[0], SEEDS[1]),
        (SEEDS[0], SEEDS[2]),
        (SEEDS[1], SEEDS[2]),
    ]

    rows = []
    distance_by_pair = {pair: [] for pair in pairs}
    group_by_pair = {pair: [] for pair in pairs}

    for u in units:
        z = {s: standardized_vector(u, s, mu, sd) for s in SEEDS}
        for a, b in pairs:
            d = float(np.linalg.norm(z[a] - z[b]) / math.sqrt(len(SOURCE_FEATURE_NAMES)))
            rows.append({
                "sample_id": u["sample_id"],
                "perturbation": u["perturbation"],
                "source_group_id": u["source_group_id"],
                "fold": u["fold"],
                "seed_a": a,
                "seed_b": b,
                "rms_standardized_b0_distance": d,
            })
            distance_by_pair[(a, b)].append(d)
            group_by_pair[(a, b)].append(u["source_group_id"])

    return rows, distance_by_pair, group_by_pair, mu, sd


def cluster_bootstrap_indices(units, n_boot, rng_seed):
    groups = sorted({u["source_group_id"] for u in units})
    group_to_indices = defaultdict(list)
    for i, u in enumerate(units):
        group_to_indices[u["source_group_id"]].append(i)

    if len(groups) != SOURCE_GROUPS and len(units) == MATCHED_UNITS:
        raise RuntimeError(
            f"Production bootstrap group count mismatch: {len(groups)}"
        )

    rng = np.random.default_rng(rng_seed)
    boot = []
    for _ in tqdm(
        range(n_boot),
        desc="Q1-S02C source-group bootstrap",
        unit="resample",
        dynamic_ncols=True,
    ):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        idx = []
        for g in sampled:
            idx.extend(group_to_indices[g])
        boot.append(np.asarray(idx, dtype=np.int64))
    return boot


def bootstrap_distance_statistics(units, distance_by_pair, n_boot=BOOTSTRAP_RESAMPLES):
    pairs = [
        (SEEDS[0], SEEDS[1]),
        (SEEDS[0], SEEDS[2]),
        (SEEDS[1], SEEDS[2]),
    ]
    arrays = {
        pair: np.asarray(distance_by_pair[pair], dtype=np.float64)
        for pair in pairs
    }

    boot_idx = cluster_bootstrap_indices(units, n_boot, BOOTSTRAP_RNG_SEED)

    pair_boot = {
        pair: {"mean": [], "median": []}
        for pair in pairs
    }
    delta17_boot = []
    delta18_boot = []

    for idx in boot_idx:
        med = {}
        for pair in pairs:
            vals = arrays[pair][idx]
            pair_boot[pair]["mean"].append(float(np.mean(vals)))
            pair_boot[pair]["median"].append(float(np.median(vals)))
            med[pair] = float(np.median(vals))

        base = med[(SEEDS[0], SEEDS[1])]
        delta17_boot.append(med[(SEEDS[0], SEEDS[2])] - base)
        delta18_boot.append(med[(SEEDS[1], SEEDS[2])] - base)

    def ci(v):
        a = np.asarray(v, dtype=np.float64)
        return [float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975))]

    result = {"bootstrap_resamples": n_boot, "rng_seed": BOOTSTRAP_RNG_SEED, "pairs": {}}
    for pair in pairs:
        arr = arrays[pair]
        result["pairs"][f"{pair[0]}_{pair[1]}"] = {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "p90": float(np.quantile(arr, 0.90)),
            "mean_ci95": ci(pair_boot[pair]["mean"]),
            "median_ci95": ci(pair_boot[pair]["median"]),
        }

    result["delta17_median_increment_ci95"] = ci(delta17_boot)
    result["delta18_median_increment_ci95"] = ci(delta18_boot)
    return result


def seed19_regime_separation(distance_by_pair, bootstrap_stats):
    d_17_18 = float(np.median(distance_by_pair[(SEEDS[0], SEEDS[1])]))
    d_17_19 = float(np.median(distance_by_pair[(SEEDS[0], SEEDS[2])]))
    d_18_19 = float(np.median(distance_by_pair[(SEEDS[1], SEEDS[2])]))

    d19 = (d_17_19 + d_18_19) / 2.0
    ratio = d19 / d_17_18 if d_17_18 > 1e-15 else float("inf")
    delta17 = d_17_19 - d_17_18
    delta18 = d_18_19 - d_17_18

    ci17 = bootstrap_stats["delta17_median_increment_ci95"]
    ci18 = bootstrap_stats["delta18_median_increment_ci95"]

    distinct = (
        ratio >= TH_DISTANCE_RATIO
        and ci17[0] > 0.0
        and ci18[0] > 0.0
    )

    return {
        "median_distance_17_18": d_17_18,
        "median_distance_17_19": d_17_19,
        "median_distance_18_19": d_18_19,
        "d_base": d_17_18,
        "d_19_average": d19,
        "distance_ratio_R_D": ratio,
        "distance_ratio_threshold": TH_DISTANCE_RATIO,
        "deltaD_17": delta17,
        "deltaD_17_ci95": ci17,
        "deltaD_18": delta18,
        "deltaD_18_ci95": ci18,
        "seed19_representation_distinct": distinct,
    }


# ---------------------------------------------------------------------
# Analysis C: feature-wise regime localization
# ---------------------------------------------------------------------

def feature_regime_localization(units):
    out = []
    for feature in SOURCE_FEATURE_NAMES:
        vals = {
            s: np.asarray([u[f"{feature}__{s}"] for u in units], dtype=np.float64)
            for s in SEEDS
        }
        pair_metrics = {}
        for a, b in [
            (SEEDS[0], SEEDS[1]),
            (SEEDS[0], SEEDS[2]),
            (SEEDS[1], SEEDS[2]),
        ]:
            diff = vals[a] - vals[b]
            pair_metrics[(a, b)] = {
                "spearman": safe_spearman(vals[a], vals[b]),
                "mean_difference": float(np.mean(diff)),
                "paired_smd": paired_smd(vals[a], vals[b]),
                "median_abs_difference": float(np.median(np.abs(diff))),
            }

        s_j = (
            (
                abs(pair_metrics[(SEEDS[0], SEEDS[2])]["paired_smd"])
                + abs(pair_metrics[(SEEDS[1], SEEDS[2])]["paired_smd"])
            ) / 2.0
            - abs(pair_metrics[(SEEDS[0], SEEDS[1])]["paired_smd"])
        )

        row = {"feature": feature, "regime_localization_score_Sj": float(s_j)}
        for (a, b), m in pair_metrics.items():
            prefix = f"{a}_{b}"
            row[f"spearman_{prefix}"] = m["spearman"]
            row[f"mean_diff_{prefix}"] = m["mean_difference"]
            row[f"paired_smd_{prefix}"] = m["paired_smd"]
            row[f"median_abs_diff_{prefix}"] = m["median_abs_difference"]
        out.append(row)

    out.sort(key=lambda r: r["regime_localization_score_Sj"], reverse=True)
    for rank, r in enumerate(out, 1):
        r["rank"] = rank
    return out


# ---------------------------------------------------------------------
# Analysis D/E: seed19 utility residual + B0 shift
# ---------------------------------------------------------------------

def build_seed19_residual_rows(units):
    rows = []
    shift_names = [f"deltaB19__{f}" for f in SOURCE_FEATURE_NAMES]

    for u in units:
        d17 = float(u[f"delta_{SEEDS[0]}"])
        d18 = float(u[f"delta_{SEEDS[1]}"])
        d19 = float(u[f"delta_{SEEDS[2]}"])
        consensus = (d17 + d18) / 2.0
        residual = d19 - consensus

        sd17 = float(u[f"source_dice_{SEEDS[0]}"])
        sd18 = float(u[f"source_dice_{SEEDS[1]}"])
        sd19 = float(u[f"source_dice_{SEEDS[2]}"])
        source_dice_shift = sd19 - (sd17 + sd18) / 2.0

        row = {
            "sample_id": u["sample_id"],
            "perturbation": u["perturbation"],
            "source_group_id": u["source_group_id"],
            "fold": int(u["fold"]),
            "delta17": d17,
            "delta18": d18,
            "delta19": d19,
            "delta17_18_consensus": consensus,
            "seed19_utility_residual": residual,
            "source_dice17": sd17,
            "source_dice18": sd18,
            "source_dice19": sd19,
            "source_dice_shift19_vs_consensus": source_dice_shift,
            "class17": u[f"class_{SEEDS[0]}"],
            "class18": u[f"class_{SEEDS[1]}"],
            "class19": u[f"class_{SEEDS[2]}"],
        }
        for f, shift_name in zip(SOURCE_FEATURE_NAMES, shift_names):
            x17 = float(u[f"{f}__{SEEDS[0]}"])
            x18 = float(u[f"{f}__{SEEDS[1]}"])
            x19 = float(u[f"{f}__{SEEDS[2]}"])
            row[shift_name] = x19 - (x17 + x18) / 2.0
        rows.append(row)

    return rows, shift_names


def residual_summary(rows):
    residual = np.asarray(
        [r["seed19_utility_residual"] for r in rows],
        dtype=np.float64,
    )
    consensus = np.asarray(
        [r["delta17_18_consensus"] for r in rows],
        dtype=np.float64,
    )
    d19 = np.asarray([r["delta19"] for r in rows], dtype=np.float64)
    d17 = np.asarray([r["delta17"] for r in rows], dtype=np.float64)
    d18 = np.asarray([r["delta18"] for r in rows], dtype=np.float64)

    return {
        "rows": len(rows),
        "residual_mean": float(np.mean(residual)),
        "residual_median": float(np.median(residual)),
        "residual_std": float(np.std(residual)),
        "residual_q10": float(np.quantile(residual, 0.10)),
        "residual_q90": float(np.quantile(residual, 0.90)),
        "delta17_delta18_spearman": safe_spearman(d17, d18),
        "consensus17_18_vs_delta19_spearman": safe_spearman(consensus, d19),
    }


# ---------------------------------------------------------------------
# Grouped OOF helpers
# ---------------------------------------------------------------------

def grouped_oof_ridge(rows, feature_names, target_name):
    predictions = [None] * len(rows)
    split_audit = []

    for fold in range(N_FOLDS):
        train_idx = [i for i, r in enumerate(rows) if int(r["fold"]) != fold]
        test_idx = [i for i, r in enumerate(rows) if int(r["fold"]) == fold]

        train_groups = {rows[i]["source_group_id"] for i in train_idx}
        test_groups = {rows[i]["source_group_id"] for i in test_idx}
        if train_groups & test_groups:
            raise RuntimeError(f"Ridge source-group leakage fold={fold}")

        x_train = np.asarray(
            [[safe_float(rows[i][f]) for f in feature_names] for i in train_idx],
            dtype=np.float64,
        )
        x_test = np.asarray(
            [[safe_float(rows[i][f]) for f in feature_names] for i in test_idx],
            dtype=np.float64,
        )
        y_train = np.asarray(
            [safe_float(rows[i][target_name]) for i in train_idx],
            dtype=np.float64,
        )

        model = make_ridge()
        model.fit(x_train, y_train)
        pred = model.predict(x_test)

        for i, p in zip(test_idx, pred):
            predictions[i] = float(p)

        split_audit.append({
            "fold": fold,
            "train_rows": len(train_idx),
            "test_rows": len(test_idx),
            "train_groups": len(train_groups),
            "test_groups": len(test_groups),
            "group_overlap": 0,
        })

    if any(p is None for p in predictions):
        raise RuntimeError("Incomplete grouped OOF Ridge predictions.")
    pred = np.asarray(predictions, dtype=np.float64)
    y = np.asarray([safe_float(r[target_name]) for r in rows], dtype=np.float64)

    if not np.isfinite(pred).all():
        raise RuntimeError("Non-finite grouped OOF Ridge predictions.")

    metrics = {
        "rows": len(rows),
        "spearman": safe_spearman(y, pred),
        "mae": float(mean_absolute_error(y, pred)),
    }
    return pred, metrics, split_audit


def grouped_oof_logistic(rows, feature_names, target_name):
    predictions = [None] * len(rows)
    split_audit = []

    for fold in range(N_FOLDS):
        train_idx = [i for i, r in enumerate(rows) if int(r["fold"]) != fold]
        test_idx = [i for i, r in enumerate(rows) if int(r["fold"]) == fold]

        train_groups = {rows[i]["source_group_id"] for i in train_idx}
        test_groups = {rows[i]["source_group_id"] for i in test_idx}
        if train_groups & test_groups:
            raise RuntimeError(f"Logistic source-group leakage fold={fold}")

        x_train = np.asarray(
            [[safe_float(rows[i][f]) for f in feature_names] for i in train_idx],
            dtype=np.float64,
        )
        x_test = np.asarray(
            [[safe_float(rows[i][f]) for f in feature_names] for i in test_idx],
            dtype=np.float64,
        )
        y_train = np.asarray([int(rows[i][target_name]) for i in train_idx], dtype=np.int64)

        if len(np.unique(y_train)) < 2:
            raise RuntimeError(
                f"Logistic training fold lacks both classes fold={fold}"
            )

        model = make_logistic()
        model.fit(x_train, y_train)
        pred = model.predict_proba(x_test)[:, 1]

        for i, p in zip(test_idx, pred):
            predictions[i] = float(p)

        split_audit.append({
            "fold": fold,
            "train_rows": len(train_idx),
            "test_rows": len(test_idx),
            "train_groups": len(train_groups),
            "test_groups": len(test_groups),
            "group_overlap": 0,
        })

    if any(p is None for p in predictions):
        raise RuntimeError("Incomplete grouped OOF Logistic predictions.")

    pred = np.asarray(predictions, dtype=np.float64)
    y = np.asarray([int(r[target_name]) for r in rows], dtype=np.int64)

    if len(np.unique(y)) < 2:
        raise RuntimeError("Full logistic target lacks both classes.")
    if not np.isfinite(pred).all():
        raise RuntimeError("Non-finite grouped OOF Logistic predictions.")

    metrics = {
        "rows": len(rows),
        "positive_count": int(y.sum()),
        "negative_count": int((1 - y).sum()),
        "auroc": float(roc_auc_score(y, pred)),
        "auprc": float(average_precision_score(y, pred)),
    }
    return pred, metrics, split_audit


# ---------------------------------------------------------------------
# Analysis F
# ---------------------------------------------------------------------

def residual_oof_analysis(residual_rows, shift_names):
    pred, metrics, split_audit = grouped_oof_ridge(
        residual_rows,
        shift_names,
        "seed19_utility_residual",
    )

    prediction_rows = []
    for r, p in zip(residual_rows, pred):
        prediction_rows.append({
            "sample_id": r["sample_id"],
            "perturbation": r["perturbation"],
            "source_group_id": r["source_group_id"],
            "fold": r["fold"],
            "true_seed19_utility_residual": r["seed19_utility_residual"],
            "pred_seed19_utility_residual": float(p),
            "absolute_error": abs(float(p) - r["seed19_utility_residual"]),
        })

    metrics["go_threshold_spearman"] = TH_RESIDUAL_SPEARMAN
    metrics["pass"] = metrics["spearman"] >= TH_RESIDUAL_SPEARMAN
    metrics["split_audit"] = split_audit
    return prediction_rows, metrics, pred


# ---------------------------------------------------------------------
# Analysis G
# ---------------------------------------------------------------------

def consensus_divergence_analysis(residual_rows, shift_names):
    consensus_rows = []
    for r in residual_rows:
        if r["class17"] != r["class18"]:
            continue
        rr = dict(r)
        rr["consensus_class"] = r["class17"]
        rr["seed19_diverges"] = int(r["class19"] != r["class17"])
        consensus_rows.append(rr)

    positives = sum(r["seed19_diverges"] for r in consensus_rows)
    negatives = len(consensus_rows) - positives
    powered = positives >= MIN_DIVERGENCE_POSITIVES and negatives > 0

    if not powered:
        metrics = {
            "status": "UNDERPOWERED",
            "rows": len(consensus_rows),
            "positive_count": positives,
            "negative_count": negatives,
            "minimum_positive_count": MIN_DIVERGENCE_POSITIVES,
            "required_for_go": False,
        }
        pred_rows = [
            {
                "sample_id": r["sample_id"],
                "perturbation": r["perturbation"],
                "source_group_id": r["source_group_id"],
                "fold": r["fold"],
                "consensus_class": r["consensus_class"],
                "seed19_class": r["class19"],
                "seed19_diverges": r["seed19_diverges"],
                "pred_divergence_probability": "",
            }
            for r in consensus_rows
        ]
        return pred_rows, metrics, []

    pred, base_metrics, split_audit = grouped_oof_logistic(
        consensus_rows,
        shift_names,
        "seed19_diverges",
    )
    metrics = {
        "status": "POWERED",
        **base_metrics,
        "go_threshold_auroc": TH_DIVERGENCE_AUROC,
        "pass": base_metrics["auroc"] >= TH_DIVERGENCE_AUROC,
        "required_for_go": True,
        "split_audit": split_audit,
    }

    pred_rows = []
    for r, p in zip(consensus_rows, pred):
        pred_rows.append({
            "sample_id": r["sample_id"],
            "perturbation": r["perturbation"],
            "source_group_id": r["source_group_id"],
            "fold": r["fold"],
            "consensus_class": r["consensus_class"],
            "seed19_class": r["class19"],
            "seed19_diverges": r["seed19_diverges"],
            "pred_divergence_probability": float(p),
        })
    return pred_rows, metrics, split_audit


# ---------------------------------------------------------------------
# Analysis H
# ---------------------------------------------------------------------

def source_quality_alternative_audit(residual_rows, shift_names):
    source_shift_name = "source_dice_shift19_vs_consensus"
    residual = np.asarray(
        [r["seed19_utility_residual"] for r in residual_rows],
        dtype=np.float64,
    )
    source_shift = np.asarray(
        [r[source_shift_name] for r in residual_rows],
        dtype=np.float64,
    )

    direct_spearman = safe_spearman(source_shift, residual)

    pred_quality, m_quality, split_quality = grouped_oof_ridge(
        residual_rows,
        [source_shift_name],
        "seed19_utility_residual",
    )

    pred_combined, m_combined, split_combined = grouped_oof_ridge(
        residual_rows,
        [source_shift_name] + list(shift_names),
        "seed19_utility_residual",
    )

    increment = float(m_combined["spearman"] - m_quality["spearman"])

    return {
        "source_dice_shift_vs_residual_spearman": direct_spearman,
        "source_dice_shift_only_oof": {
            **m_quality,
            "split_audit": split_quality,
        },
        "source_dice_shift_plus_b0_shift_oof": {
            **m_combined,
            "split_audit": split_combined,
        },
        "incremental_spearman_B0_given_quality": increment,
        "go_threshold_incremental_spearman": TH_QUALITY_INCREMENT,
        "pass": increment >= TH_QUALITY_INCREMENT,
        "source_dice_is_audit_only": True,
    }


# ---------------------------------------------------------------------
# Analysis I
# ---------------------------------------------------------------------

def perturbation_residual_stability(residual_rows, residual_pred):
    out = []
    correlations = []

    for p in PERTURBATIONS:
        idx = [i for i, r in enumerate(residual_rows) if r["perturbation"] == p]
        y = np.asarray(
            [residual_rows[i]["seed19_utility_residual"] for i in idx],
            dtype=np.float64,
        )
        pred = np.asarray([residual_pred[i] for i in idx], dtype=np.float64)
        rho = safe_spearman(y, pred)
        mae = float(mean_absolute_error(y, pred))
        correlations.append(rho)
        out.append({
            "perturbation": p,
            "rows": len(idx),
            "residual_spearman": rho,
            "residual_mae": mae,
            "spearman_positive": rho > 0.0,
        })

    summary = {
        "median_perturbation_residual_spearman": float(np.median(correlations)),
        "positive_spearman_perturbations": int(sum(v > 0 for v in correlations)),
        "total_perturbations": len(correlations),
    }
    return out, summary


# ---------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------

def mechanism_decision(regime, residual_metrics, divergence_metrics, quality_audit):
    representation_distinct = bool(regime["seed19_representation_distinct"])
    residual_pass = residual_metrics["spearman"] >= TH_RESIDUAL_SPEARMAN
    quality_pass = (
        quality_audit["incremental_spearman_B0_given_quality"]
        >= TH_QUALITY_INCREMENT
    )

    divergence_powered = divergence_metrics["status"] == "POWERED"
    if divergence_powered:
        divergence_pass = divergence_metrics["auroc"] >= TH_DIVERGENCE_AUROC
    else:
        divergence_pass = True  # preregistered: not required when UNDERPOWERED

    if not representation_distinct:
        decision = STOP_NO_REGIME
    elif not (residual_pass and quality_pass and divergence_pass):
        decision = STOP_UNEXPLAINED
    else:
        decision = GO_REGIME

    criteria = {
        "representation_distinct": {
            "value": representation_distinct,
            "distance_ratio": regime["distance_ratio_R_D"],
            "distance_ratio_threshold": TH_DISTANCE_RATIO,
            "deltaD17_ci95": regime["deltaD_17_ci95"],
            "deltaD18_ci95": regime["deltaD_18_ci95"],
            "pass": representation_distinct,
        },
        "residual_oof_spearman": {
            "value": residual_metrics["spearman"],
            "threshold": TH_RESIDUAL_SPEARMAN,
            "pass": residual_pass,
        },
        "source_quality_incremental_spearman": {
            "value": quality_audit["incremental_spearman_B0_given_quality"],
            "threshold": TH_QUALITY_INCREMENT,
            "pass": quality_pass,
        },
        "divergence_auroc": {
            "status": divergence_metrics["status"],
            "positive_count": divergence_metrics["positive_count"],
            "threshold": TH_DIVERGENCE_AUROC,
            "required": divergence_powered,
            "value": (
                divergence_metrics.get("auroc")
                if divergence_powered
                else None
            ),
            "pass": divergence_pass,
        },
    }
    return decision, criteria


# ---------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------

def make_synthetic_units():
    rng = np.random.default_rng(20260819)
    units = []

    # 25 groups x 10 perturbations = 250 units; enough for 5 grouped folds.
    for g in range(25):
        fold = g % N_FOLDS
        group = f"g{g:03d}"
        latent = rng.normal()
        for p_i, perturbation in enumerate(PERTURBATIONS):
            row = {
                "sample_id": f"s{g:03d}",
                "perturbation": perturbation,
                "source_group_id": group,
                "fold": fold,
            }

            common = latent + 0.1 * p_i + rng.normal(scale=0.2)
            for s_i, seed in enumerate(SEEDS):
                regime_shift = 0.0 if seed in BASE_SEEDS else 0.9
                x = common + regime_shift + rng.normal(scale=0.2)
                delta = (
                    0.04 * common
                    + (0.0 if seed in BASE_SEEDS else 0.06 * x)
                    + rng.normal(scale=0.03)
                )

                row[f"delta_{seed}"] = float(delta)
                row[f"source_dice_{seed}"] = float(
                    np.clip(0.75 + 0.03 * common - 0.02 * s_i, 0, 1)
                )
                row[f"action_dice_{seed}"] = float(
                    np.clip(row[f"source_dice_{seed}"] + delta, 0, 1)
                )
                row[f"class_{seed}"] = outcome_from_delta(delta)

                for j, f in enumerate(SOURCE_FEATURE_NAMES):
                    row[f"{f}__{seed}"] = float(
                        x * (1.0 + 0.02 * j)
                        + rng.normal(scale=0.15 + 0.005 * j)
                    )
            units.append(row)
    return units


def self_test():
    assert len(SOURCE_FEATURE_NAMES) == 19
    assert len(SEEDS) == 3
    assert len(PERTURBATIONS) == 10
    assert SOURCE_GROUPS == 145
    assert MATCHED_UNITS == 1450
    assert EXPECTED_ROWS == 4350
    assert BOOTSTRAP_RESAMPLES == 10_000
    assert TH_DISTANCE_RATIO == 1.15
    assert TH_RESIDUAL_SPEARMAN == 0.20
    assert TH_DIVERGENCE_AUROC == 0.65
    assert TH_QUALITY_INCREMENT == 0.05
    assert MIN_DIVERGENCE_POSITIVES == 30

    units = make_synthetic_units()
    assert len(units) == 250

    distance_rows, distance_by_pair, _, mu, sd = compute_pairwise_distance_rows(units)
    assert len(distance_rows) == len(units) * 3
    assert len(mu) == 19 and len(sd) == 19

    # Small bootstrap for implementation test only.
    boot = bootstrap_distance_statistics(units, distance_by_pair, n_boot=50)
    assert boot["bootstrap_resamples"] == 50

    regime = seed19_regime_separation(distance_by_pair, boot)
    assert math.isfinite(regime["distance_ratio_R_D"])

    feature_rows = feature_regime_localization(units)
    assert len(feature_rows) == 19

    residual_rows, shift_names = build_seed19_residual_rows(units)
    assert len(shift_names) == 19
    assert len(residual_rows) == len(units)

    residual_pred_rows, residual_metrics, residual_pred = residual_oof_analysis(
        residual_rows,
        shift_names,
    )
    assert len(residual_pred_rows) == len(units)
    assert math.isfinite(residual_metrics["spearman"])

    div_rows, div_metrics, _ = consensus_divergence_analysis(
        residual_rows,
        shift_names,
    )
    assert div_metrics["status"] in {"POWERED", "UNDERPOWERED"}

    quality = source_quality_alternative_audit(residual_rows, shift_names)
    assert math.isfinite(quality["incremental_spearman_B0_given_quality"])

    pert_rows, pert_summary = perturbation_residual_stability(
        residual_rows,
        residual_pred,
    )
    assert len(pert_rows) == len(PERTURBATIONS)
    represented = {r["perturbation"] for r in residual_rows}
    assert represented == set(PERTURBATIONS)

    print("FROZEN_CONSTANTS_TEST_PASS")
    print("DISTANCE_AND_BOOTSTRAP_TEST_PASS")
    print("REGIME_LOCALIZATION_TEST_PASS")
    print("RESIDUAL_OOF_TEST_PASS")
    print("DIVERGENCE_TEST_PASS")
    print("SOURCE_QUALITY_AUDIT_TEST_PASS")
    print("SELF_TEST_PASS")


# ---------------------------------------------------------------------
# Production main
# ---------------------------------------------------------------------

def run(args):
    if args.output_dir.exists():
        raise FileExistsError(f"Q1-S02C output already exists: {args.output_dir}")

    build_dir = Path(str(args.output_dir) + "__building")
    if build_dir.exists():
        raise FileExistsError(
            f"Partial Q1-S02C build exists: {build_dir}. "
            "Remove only this __building directory if a previous technical run failed."
        )
    build_dir.mkdir(parents=True, exist_ok=False)

    print("===== Q1-S02C CHECKPOINT ADAPTATION-REGIME AUDIT =====")
    print(f"Script version: {VERSION}")
    print(f"Build: {BUILD}")
    print(f"Bootstrap resamples: {BOOTSTRAP_RESAMPLES}")
    print("New segmentation inference: NO")
    print("Target data used: NO")
    print("Target GT used: NO")
    print("Gradient features used: NO")
    print("Adapted prediction features used: NO")
    print("Hyperparameter sweep: NO")
    print("Feature selection: NO")
    print("Threshold tuning: NO")
    print()

    merged, matched, provenance = load_frozen_rows()
    units = build_units(matched)

    protocol_copy = build_dir / "preregistered_protocol_copy.md"
    shutil.copy2(PROTOCOL, protocol_copy)
    if file_sha256(protocol_copy) != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("Protocol copy SHA mismatch.")

    provenance_path = build_dir / "provenance_audit.json"
    write_json(provenance_path, provenance)

    # A/B
    (
        distance_rows,
        distance_by_pair,
        _,
        pooled_mu,
        pooled_sd,
    ) = compute_pairwise_distance_rows(units)

    pairwise_distance_path = build_dir / "pairwise_b0_distance.csv"
    write_csv(
        pairwise_distance_path,
        distance_rows,
        [
            "sample_id",
            "perturbation",
            "source_group_id",
            "fold",
            "seed_a",
            "seed_b",
            "rms_standardized_b0_distance",
        ],
    )

    bootstrap_stats = bootstrap_distance_statistics(
        units,
        distance_by_pair,
        n_boot=BOOTSTRAP_RESAMPLES,
    )
    bootstrap_stats["pooled_standardization"] = {
        "feature_names": SOURCE_FEATURE_NAMES,
        "mean": [float(v) for v in pooled_mu],
        "std": [float(v) for v in pooled_sd],
    }

    bootstrap_path = build_dir / "pairwise_b0_distance_bootstrap.json"
    write_json(bootstrap_path, bootstrap_stats)

    regime = seed19_regime_separation(distance_by_pair, bootstrap_stats)
    regime_path = build_dir / "seed19_regime_separation.json"
    write_json(regime_path, regime)

    # C
    feature_rows = feature_regime_localization(units)
    feature_path = build_dir / "feature_regime_localization.csv"
    feature_fields = list(feature_rows[0].keys())
    write_csv(feature_path, feature_rows, feature_fields)

    # D/E
    residual_rows, shift_names = build_seed19_residual_rows(units)
    residual_summary_obj = residual_summary(residual_rows)

    residual_table_path = build_dir / "seed19_utility_residual.csv"
    residual_fields = [
        "sample_id",
        "perturbation",
        "source_group_id",
        "fold",
        "delta17",
        "delta18",
        "delta19",
        "delta17_18_consensus",
        "seed19_utility_residual",
        "source_dice17",
        "source_dice18",
        "source_dice19",
        "source_dice_shift19_vs_consensus",
        "class17",
        "class18",
        "class19",
    ] + shift_names
    write_csv(residual_table_path, residual_rows, residual_fields)

    # F
    residual_pred_rows, residual_metrics, residual_pred = residual_oof_analysis(
        residual_rows,
        shift_names,
    )
    residual_pred_path = build_dir / "residual_oof_predictions.csv"
    write_csv(
        residual_pred_path,
        residual_pred_rows,
        [
            "sample_id",
            "perturbation",
            "source_group_id",
            "fold",
            "true_seed19_utility_residual",
            "pred_seed19_utility_residual",
            "absolute_error",
        ],
    )
    residual_metrics["residual_distribution"] = residual_summary_obj
    residual_metrics_path = build_dir / "residual_oof_metrics.json"
    write_json(residual_metrics_path, residual_metrics)

    # G
    divergence_rows, divergence_metrics, divergence_split_audit = (
        consensus_divergence_analysis(residual_rows, shift_names)
    )
    divergence_pred_path = build_dir / "consensus_divergence_oof_predictions.csv"
    write_csv(
        divergence_pred_path,
        divergence_rows,
        [
            "sample_id",
            "perturbation",
            "source_group_id",
            "fold",
            "consensus_class",
            "seed19_class",
            "seed19_diverges",
            "pred_divergence_probability",
        ],
    )
    divergence_metrics_path = build_dir / "consensus_divergence_metrics.json"
    write_json(divergence_metrics_path, divergence_metrics)

    # H
    quality_audit = source_quality_alternative_audit(
        residual_rows,
        shift_names,
    )
    quality_path = build_dir / "source_quality_alternative_audit.json"
    write_json(quality_path, quality_audit)

    # I
    perturbation_rows, perturbation_summary = perturbation_residual_stability(
        residual_rows,
        residual_pred,
    )
    perturbation_path = build_dir / "perturbation_residual_stability.csv"
    write_csv(
        perturbation_path,
        perturbation_rows,
        [
            "perturbation",
            "rows",
            "residual_spearman",
            "residual_mae",
            "spearman_positive",
        ],
    )

    # Decision
    decision, criteria = mechanism_decision(
        regime,
        residual_metrics,
        divergence_metrics,
        quality_audit,
    )

    criteria_rows = [
        {
            "criterion": "seed19_representation_distinct",
            "value": criteria["representation_distinct"]["value"],
            "threshold": "R_D>=1.15 and both delta-distance CI lower bounds >0",
            "pass": criteria["representation_distinct"]["pass"],
        },
        {
            "criterion": "residual_oof_spearman",
            "value": criteria["residual_oof_spearman"]["value"],
            "threshold": TH_RESIDUAL_SPEARMAN,
            "pass": criteria["residual_oof_spearman"]["pass"],
        },
        {
            "criterion": "source_quality_incremental_spearman",
            "value": criteria["source_quality_incremental_spearman"]["value"],
            "threshold": TH_QUALITY_INCREMENT,
            "pass": criteria["source_quality_incremental_spearman"]["pass"],
        },
        {
            "criterion": "divergence_auroc",
            "value": (
                ""
                if criteria["divergence_auroc"]["value"] is None
                else criteria["divergence_auroc"]["value"]
            ),
            "threshold": (
                f"{TH_DIVERGENCE_AUROC} if powered; "
                f"UNDERPOWERED if positives<{MIN_DIVERGENCE_POSITIVES}"
            ),
            "pass": criteria["divergence_auroc"]["pass"],
        },
    ]

    criteria_path = build_dir / "mechanism_criteria.csv"
    write_csv(
        criteria_path,
        criteria_rows,
        ["criterion", "value", "threshold", "pass"],
    )

    decision_path = build_dir / "decision.txt"
    decision_path.write_text(decision + "\n", encoding="utf-8")

    # Positive-polarity integrity audit.
    divergence_group_ok = all(
        item.get("group_overlap", 0) == 0
        for item in divergence_split_audit
    )
    residual_group_ok = all(
        item.get("group_overlap", 0) == 0
        for item in residual_metrics["split_audit"]
    )
    quality_group_ok = all(
        item.get("group_overlap", 0) == 0
        for item in (
            quality_audit["source_dice_shift_only_oof"]["split_audit"]
            + quality_audit["source_dice_shift_plus_b0_shift_oof"]["split_audit"]
        )
    )

    integrity = {
        "script_version": VERSION,
        "build": BUILD,
        "checks": {
            "q1_s02a_lock_verified": provenance["q1_s02a_lock_sha256"] == EXPECTED_Q1_S02A_LOCK_SHA256,
            "q1_s02a_decision_verified": provenance["q1_s02a_decision"] == EXPECTED_Q1_S02A_DECISION,
            "rows_exact_4350": len(merged) == EXPECTED_ROWS,
            "matched_units_exact_1450": len(units) == MATCHED_UNITS,
            "source_groups_exact_145": len({u["source_group_id"] for u in units}) == SOURCE_GROUPS,
            "perturbations_exact_10": {u["perturbation"] for u in units} == set(PERTURBATIONS),
            "seeds_exact_3": set(provenance["seeds"]) == set(SEEDS),
            "every_matched_unit_has_three_seeds": all(
                len(rr) == 3 and {int(r["seed"]) for r in rr} == set(SEEDS)
                for rr in matched.values()
            ),
            "action_exact_A1_TENT_1STEP": provenance["action"] == A1_ACTION,
            "b0_features_exact_19": len(SOURCE_FEATURE_NAMES) == 19,
            "gradient_features_not_used": True,
            "adapted_prediction_features_not_used": True,
            "target_data_not_loaded": True,
            "target_gt_not_loaded": True,
            "new_image_inference_not_run": True,
            "source_dice_audit_only": True,
            "residual_oof_group_overlap_zero": residual_group_ok,
            "divergence_oof_group_overlap_zero": divergence_group_ok,
            "quality_oof_group_overlap_zero": quality_group_ok,
            "standardizers_fit_training_only_in_predictive_oof": True,
            "bootstrap_resamples_source_groups": True,
            "bootstrap_resamples_exact_10000": bootstrap_stats["bootstrap_resamples"] == BOOTSTRAP_RESAMPLES,
            "hyperparameter_sweep_not_used": True,
            "feature_selection_not_used": True,
            "threshold_tuning_not_used": True,
            "primary_statistics_finite": all(
                math.isfinite(float(v))
                for v in [
                    regime["distance_ratio_R_D"],
                    residual_metrics["spearman"],
                    quality_audit["incremental_spearman_B0_given_quality"],
                    perturbation_summary["median_perturbation_residual_spearman"],
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
            f"Q1-S02C integrity failure: {failed_integrity}"
        )

    integrity_path = build_dir / "leakage_integrity_audit.json"
    write_json(integrity_path, integrity)

    # Human-readable final report.
    d_pair = bootstrap_stats["pairs"]
    div_status = divergence_metrics["status"]
    if div_status == "POWERED":
        div_line = (
            f"  Consensus divergence AUROC={divergence_metrics['auroc']:.6f}\n"
            f"  AUPRC={divergence_metrics['auprc']:.6f}\n"
            f"  positives={divergence_metrics['positive_count']}\n"
            f"  Required AUROC >= {TH_DIVERGENCE_AUROC:.2f}"
        )
    else:
        div_line = (
            f"  Status=UNDERPOWERED\n"
            f"  positives={divergence_metrics['positive_count']}\n"
            f"  Minimum positives={MIN_DIVERGENCE_POSITIVES}\n"
            f"  Divergence criterion excluded from GO decision"
        )

    summary = f"""===== Q1-S02C CHECKPOINT ADAPTATION-REGIME AUDIT =====
Script version: {VERSION}
Build: {BUILD}

Frozen data:
  rows={EXPECTED_ROWS}
  source groups={SOURCE_GROUPS}
  matched units={MATCHED_UNITS}
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

[A — PAIRWISE PRE-SOURCE REPRESENTATION DISTANCE]
  17-18 mean={d_pair[f'{SEEDS[0]}_{SEEDS[1]}']['mean']:.6f}
        median={d_pair[f'{SEEDS[0]}_{SEEDS[1]}']['median']:.6f}
        median_CI95={d_pair[f'{SEEDS[0]}_{SEEDS[1]}']['median_ci95']}
  17-19 mean={d_pair[f'{SEEDS[0]}_{SEEDS[2]}']['mean']:.6f}
        median={d_pair[f'{SEEDS[0]}_{SEEDS[2]}']['median']:.6f}
        median_CI95={d_pair[f'{SEEDS[0]}_{SEEDS[2]}']['median_ci95']}
  18-19 mean={d_pair[f'{SEEDS[1]}_{SEEDS[2]}']['mean']:.6f}
        median={d_pair[f'{SEEDS[1]}_{SEEDS[2]}']['median']:.6f}
        median_CI95={d_pair[f'{SEEDS[1]}_{SEEDS[2]}']['median_ci95']}

[B — SEED19 REGIME SEPARATION]
  D_base(17-18)={regime['d_base']:.6f}
  D_19 average={regime['d_19_average']:.6f}
  R_D={regime['distance_ratio_R_D']:.6f}
  Required R_D >= {TH_DISTANCE_RATIO:.2f}
  DeltaD17={regime['deltaD_17']:+.6f}
  DeltaD17 CI95={regime['deltaD_17_ci95']}
  DeltaD18={regime['deltaD_18']:+.6f}
  DeltaD18 CI95={regime['deltaD_18_ci95']}
  Representation-distinct={regime['seed19_representation_distinct']}

[C — TOP FEATURE REGIME LOCALIZATION]
"""
    for r in feature_rows[:5]:
        summary += (
            f"  rank={r['rank']} {r['feature']}: "
            f"Sj={r['regime_localization_score_Sj']:+.6f}\n"
        )

    summary += f"""
[D — SEED19 UTILITY RESIDUAL]
  mean={residual_summary_obj['residual_mean']:+.6f}
  median={residual_summary_obj['residual_median']:+.6f}
  std={residual_summary_obj['residual_std']:.6f}
  17-vs-18 DeltaDice Spearman={residual_summary_obj['delta17_delta18_spearman']:.6f}
  consensus17/18-vs-19 Spearman={residual_summary_obj['consensus17_18_vs_delta19_spearman']:.6f}

[F — PRE-SOURCE SHIFT -> SEED19 RESIDUAL]
  OOF Spearman={residual_metrics['spearman']:.6f}
  OOF MAE={residual_metrics['mae']:.6f}
  Required Spearman >= {TH_RESIDUAL_SPEARMAN:.2f}
  Pass={residual_metrics['pass']}

[G — 17/18 CONSENSUS -> SEED19 DIVERGENCE]
{div_line}

[H — SOURCE-QUALITY ALTERNATIVE]
  SourceDiceShift-vs-residual Spearman={quality_audit['source_dice_shift_vs_residual_spearman']:.6f}
  Quality-only OOF Spearman={quality_audit['source_dice_shift_only_oof']['spearman']:.6f}
  Quality+B0Shift OOF Spearman={quality_audit['source_dice_shift_plus_b0_shift_oof']['spearman']:.6f}
  Incremental Spearman={quality_audit['incremental_spearman_B0_given_quality']:+.6f}
  Required increment >= +{TH_QUALITY_INCREMENT:.2f}
  Pass={quality_audit['pass']}

[I — PERTURBATION STABILITY]
  Median perturbation residual Spearman={perturbation_summary['median_perturbation_residual_spearman']:.6f}
  Positive perturbations={perturbation_summary['positive_spearman_perturbations']}/{perturbation_summary['total_perturbations']}

Decision: {decision}

Decision meanings:
  {GO_REGIME}
    -> seed19 has a distinct pre-adaptation representation regime and the
       representation shift prospectively explains its utility residual
       beyond source-quality shift.

  {STOP_UNEXPLAINED}
    -> seed19 is representation-distinct, but the visible shift does not
       sufficiently explain adaptation-utility change.

  {STOP_NO_REGIME}
    -> seed19 is not sufficiently distinct in frozen deployment-visible B0 space.

If GO:
  Next = Q1-S03 Model-Conditioned Individual Adaptation Effect Estimation.

If STOP:
  Do not rescue with a larger fingerprint network or post-hoc feature selection.

[OK] Outputs: {args.output_dir}
"""
    run_log_path = build_dir / "run_log.txt"
    run_log_path.write_text(summary, encoding="utf-8")

    artifact_paths = {
        "protocol_copy": protocol_copy,
        "provenance_audit": provenance_path,
        "pairwise_b0_distance": pairwise_distance_path,
        "pairwise_b0_distance_bootstrap": bootstrap_path,
        "seed19_regime_separation": regime_path,
        "feature_regime_localization": feature_path,
        "seed19_utility_residual": residual_table_path,
        "residual_oof_predictions": residual_pred_path,
        "residual_oof_metrics": residual_metrics_path,
        "consensus_divergence_oof_predictions": divergence_pred_path,
        "consensus_divergence_metrics": divergence_metrics_path,
        "source_quality_alternative_audit": quality_path,
        "perturbation_residual_stability": perturbation_path,
        "mechanism_criteria": criteria_path,
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
        "q1_s02a_lock_sha256": EXPECTED_Q1_S02A_LOCK_SHA256,
        "q1_s02a_decision": EXPECTED_Q1_S02A_DECISION,
        "q1_s00a_lock_sha256": EXPECTED_Q1_S00A_LOCK_SHA256,
        "utility_table_sha256": EXPECTED_UTILITY_TABLE_SHA256,
        "target_data_used": False,
        "target_gt_used": False,
        "new_segmentation_inference": False,
        "gradient_features_used": False,
        "adapted_prediction_features_used": False,
        "source_dice_deployment_feature": False,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "hyperparameter_sweep": False,
        "feature_selection": False,
        "threshold_tuning": False,
        "criteria": criteria,
        "decision": decision,
    }

    lock_path = build_dir / "Q1_S02C_CHECKPOINT_REGIME_LOCK.json"
    write_json(lock_path, lock)
    lock_sha = file_sha256(lock_path)

    # Final immutability verification before commit.
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
        "Q1-S02C LOCK:",
        args.output_dir / "Q1_S02C_CHECKPOINT_REGIME_LOCK.json",
    )
    print("Q1-S02C LOCK SHA256:", lock_sha)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run Q1-S02C checkpoint adaptation-regime audit using frozen "
            "Q1-S00A/Q1-S02A source-side assets."
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
