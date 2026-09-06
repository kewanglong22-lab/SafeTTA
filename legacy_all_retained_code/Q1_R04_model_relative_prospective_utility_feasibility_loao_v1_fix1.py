#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Q1-R04 — Model-Relative Prospective Utility Feasibility.

Primary evaluation:
    Joint Leave-One-Architecture-and-Source-Group-Fold-Out

Representations:
    RAW19
    MRZ19

No target data.
No feature selection.
No hyperparameter sweep.
No nonlinear rescue model.
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
from tqdm import tqdm


VERSION = "2026-08-19-Q1-R04-v1-fix1"
BUILD = "Q1_R04_MODEL_RELATIVE_PROSPECTIVE_UTILITY_JOINT_LOAO_GROUP_FIX1"

ROOT = Path(r"F:\MEDSEG_SAFETTA")

PROTOCOL = (
    ROOT
    / "docs"
    / "Q1_R04_model_relative_prospective_utility_feasibility_loao_preregistered_protocol_v1.md"
)
EXPECTED_PROTOCOL_SHA256 = "dfc2d6bdf55a4b2ec0c6d37dfb270efdd255306d304516c11843e1a5f1bea08a"

R03_DIR = (
    ROOT
    / "outputs"
    / "Q1_R03_nine_state_heterogeneous_source_utility_asset_v1_fix2"
)
R03_LOCK = R03_DIR / "Q1_R03_NINE_STATE_SOURCE_UTILITY_LOCK.json"
EXPECTED_R03_LOCK_SHA256 = (
    "2c773f5eb70bedb50e2c231ef2512c06dff5ecca425eb43b2719637c1bc06dad"
)
R03_TABLE = R03_DIR / "nine_state_source_utility_table.csv"
EXPECTED_R03_TABLE_SHA256 = (
    "969ef666c4e54d0b6f3152709b99a601ff3b82ce6b9f26d46efeb23e53960642"
)
EXPECTED_R03_DECISION = "NINE_STATE_SOURCE_UTILITY_ASSET_READY"

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "Q1_R04_model_relative_prospective_utility_feasibility_loao_v1_fix1"
)

FAMILIES = (
    "PraNet",
    "DeepLabV3-R50",
    "SegFormer-B0",
)
FOLDS = (0, 1, 2, 3, 4)

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

REPRESENTATIONS = ("RAW19", "MRZ19")
REFERENCE_STD_FLOOR = 1e-6

RIDGE_ALPHA = 1.0
LOGISTIC_C = 1.0
LOGISTIC_MAX_ITER = 5000
RANDOM_STATE = 20260819

HARM_THRESHOLD = -0.02
BENEFIT_THRESHOLD = +0.02

EXPECTED_ROWS = 13050
EXPECTED_ROWS_PER_FAMILY = 4350
EXPECTED_MODEL_STATES = 9
EXPECTED_SOURCE_GROUPS = 145
EXPECTED_PERTURBATIONS = 10

DECISION_GO = "GO_MODEL_RELATIVE_UTILITY_FEASIBILITY"
DECISION_STOP = "STOP_MODEL_RELATIVE_UTILITY_NO_ROBUST_LOAO_SIGNAL"


# ---------------------------------------------------------------------
# IO / integrity
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


def load_and_validate_upstream():
    validate_sha(PROTOCOL, EXPECTED_PROTOCOL_SHA256, "Q1-R04 protocol")
    validate_sha(R03_LOCK, EXPECTED_R03_LOCK_SHA256, "Q1-R03 lock")
    validate_sha(R03_TABLE, EXPECTED_R03_TABLE_SHA256, "Q1-R03 table")

    lock = json.loads(R03_LOCK.read_text(encoding="utf-8"))
    if lock.get("decision") != EXPECTED_R03_DECISION:
        raise RuntimeError(
            f"Q1-R03 decision mismatch: {lock.get('decision')}"
        )
    if bool(lock.get("target_data_used", True)):
        raise RuntimeError("Q1-R03 lock indicates target data use.")
    if bool(lock.get("utility_predictor_fitted", True)):
        raise RuntimeError("Q1-R03 lock indicates prior utility predictor fit.")
    if int(lock.get("model_states", -1)) != EXPECTED_MODEL_STATES:
        raise RuntimeError("Q1-R03 model-state count mismatch.")
    if int(lock.get("rows", -1)) != EXPECTED_ROWS:
        raise RuntimeError("Q1-R03 row count mismatch.")

    rows, fields = read_csv(R03_TABLE)

    required = {
        "model_family",
        "model_state_id",
        "training_seed",
        "checkpoint_sha256",
        "sample_id",
        "source_group_id",
        "fold",
        "perturbation",
        "action",
        "source_dice",
        "action_dice",
        "delta_dice",
        "harmful",
        "neutral",
        "beneficial",
        "outcome_class",
        *SOURCE_FEATURE_NAMES,
    }
    missing = sorted(required - set(fields))
    if missing:
        raise RuntimeError(f"Q1-R03 table missing columns: {missing}")

    if len(rows) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Q1-R03 table rows mismatch: {len(rows)} != {EXPECTED_ROWS}"
        )

    families = Counter(r["model_family"] for r in rows)
    if set(families) != set(FAMILIES):
        raise RuntimeError(f"Family set mismatch: {dict(families)}")
    if any(families[f] != EXPECTED_ROWS_PER_FAMILY for f in FAMILIES):
        raise RuntimeError(f"Family cardinality mismatch: {dict(families)}")

    states = Counter(r["model_state_id"] for r in rows)
    if len(states) != EXPECTED_MODEL_STATES:
        raise RuntimeError(f"State count mismatch: {len(states)}")
    if any(v != 1450 for v in states.values()):
        raise RuntimeError(f"State row count mismatch: {dict(states)}")

    groups = sorted({r["source_group_id"] for r in rows})
    if len(groups) != EXPECTED_SOURCE_GROUPS:
        raise RuntimeError(f"Source-group count mismatch: {len(groups)}")

    per_group_folds = defaultdict(set)
    per_group_families = defaultdict(set)
    for r in rows:
        per_group_folds[r["source_group_id"]].add(int(r["fold"]))
        per_group_families[r["source_group_id"]].add(r["model_family"])
    if any(len(v) != 1 for v in per_group_folds.values()):
        raise RuntimeError("Source-group fold leakage detected.")
    if any(v != set(FAMILIES) for v in per_group_families.values()):
        raise RuntimeError("Not every source group appears in every family.")

    fold_counts = Counter(
        next(iter(v)) for v in per_group_folds.values()
    )
    if dict(sorted(fold_counts.items())) != EXPECTED_FOLD_GROUP_COUNTS:
        raise RuntimeError(
            f"Fold group counts mismatch: {dict(sorted(fold_counts.items()))}"
        )

    perturbations = {r["perturbation"] for r in rows}
    if len(perturbations) != EXPECTED_PERTURBATIONS:
        raise RuntimeError(
            f"Perturbation count mismatch: {len(perturbations)}"
        )
    if "identity" not in perturbations:
        raise RuntimeError("Identity perturbation missing.")

    numeric_fields = (
        SOURCE_FEATURE_NAMES
        + ["source_dice", "action_dice", "delta_dice"]
    )
    for i, r in enumerate(rows):
        vals = np.asarray(
            [float(r[name]) for name in numeric_fields],
            dtype=np.float64,
        )
        if not np.isfinite(vals).all():
            raise RuntimeError(f"Non-finite row at index {i}")

        delta = float(r["delta_dice"])
        harmful = int(r["harmful"])
        beneficial = int(r["beneficial"])
        expected_h = int(delta <= HARM_THRESHOLD)
        expected_b = int(delta >= BENEFIT_THRESHOLD)
        if harmful != expected_h or beneficial != expected_b:
            raise RuntimeError(
                f"Frozen utility class mismatch at index {i}"
            )

    return rows, fields, lock


# ---------------------------------------------------------------------
# Numerical metrics
# ---------------------------------------------------------------------

def average_rank(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)

    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        avg = 0.5 * ((i + 1) + j)
        ranks[order[i:j]] = avg
        i = j

    return ranks


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) != len(b) or len(a) < 2:
        raise RuntimeError("Invalid Pearson input.")
    ac = a - a.mean()
    bc = b - b.mean()
    denom = math.sqrt(float(np.dot(ac, ac) * np.dot(bc, bc)))
    if denom <= 0:
        return 0.0
    return float(np.dot(ac, bc) / denom)


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    return pearson_corr(average_rank(a), average_rank(b))


def metric_bundle(y_delta, pred_delta, y_harm, p_harm, y_benefit, p_benefit):
    from sklearn.metrics import roc_auc_score, average_precision_score

    y_delta = np.asarray(y_delta, dtype=np.float64)
    pred_delta = np.asarray(pred_delta, dtype=np.float64)
    y_harm = np.asarray(y_harm, dtype=np.int64)
    p_harm = np.asarray(p_harm, dtype=np.float64)
    y_benefit = np.asarray(y_benefit, dtype=np.int64)
    p_benefit = np.asarray(p_benefit, dtype=np.float64)

    if len(np.unique(y_harm)) != 2:
        raise RuntimeError("HARM metric split has only one class.")
    if len(np.unique(y_benefit)) != 2:
        raise RuntimeError("BENEFIT metric split has only one class.")

    harm_auc = float(roc_auc_score(y_harm, p_harm))
    benefit_auc = float(roc_auc_score(y_benefit, p_benefit))

    return {
        "spearman": spearman_corr(pred_delta, y_delta),
        "pearson": pearson_corr(pred_delta, y_delta),
        "mae": float(np.mean(np.abs(pred_delta - y_delta))),
        "harm_auroc": harm_auc,
        "harm_auprc": float(average_precision_score(y_harm, p_harm)),
        "benefit_auroc": benefit_auc,
        "benefit_auprc": float(
            average_precision_score(y_benefit, p_benefit)
        ),
        "safety_mean_auc": 0.5 * (harm_auc + benefit_auc),
        "harm_prevalence": float(np.mean(y_harm)),
        "benefit_prevalence": float(np.mean(y_benefit)),
    }


# ---------------------------------------------------------------------
# Array conversion / representation construction
# ---------------------------------------------------------------------

def rows_to_arrays(rows):
    n = len(rows)
    X = np.zeros((n, len(SOURCE_FEATURE_NAMES)), dtype=np.float64)

    family = np.empty(n, dtype=object)
    state = np.empty(n, dtype=object)
    sample = np.empty(n, dtype=object)
    perturb = np.empty(n, dtype=object)
    fold = np.zeros(n, dtype=np.int64)
    delta = np.zeros(n, dtype=np.float64)
    harm = np.zeros(n, dtype=np.int64)
    benefit = np.zeros(n, dtype=np.int64)

    metadata = []

    for i, r in enumerate(rows):
        X[i] = [float(r[name]) for name in SOURCE_FEATURE_NAMES]
        family[i] = r["model_family"]
        state[i] = r["model_state_id"]
        sample[i] = r["source_group_id"]
        perturb[i] = r["perturbation"]
        fold[i] = int(r["fold"])
        delta[i] = float(r["delta_dice"])
        harm[i] = int(r["harmful"])
        benefit[i] = int(r["beneficial"])

        metadata.append({
            "model_family": r["model_family"],
            "model_state_id": r["model_state_id"],
            "training_seed": r["training_seed"],
            "checkpoint_sha256": r["checkpoint_sha256"],
            "sample_id": r["sample_id"],
            "source_group_id": r["source_group_id"],
            "fold": int(r["fold"]),
            "perturbation": r["perturbation"],
        })

    if not np.isfinite(X).all():
        raise RuntimeError("Non-finite raw feature matrix.")

    return {
        "X_raw": X,
        "family": family,
        "state": state,
        "sample": sample,
        "perturbation": perturb,
        "fold": fold,
        "delta": delta,
        "harm": harm,
        "benefit": benefit,
        "metadata": metadata,
    }


def build_mrz_for_fold(arrays, held_fold: int):
    X = arrays["X_raw"]
    state = arrays["state"]
    perturb = arrays["perturbation"]
    fold = arrays["fold"]

    Z = np.empty_like(X)
    audit_rows = []

    states = sorted(set(state.tolist()))
    for sid in states:
        state_mask = state == sid
        reference_mask = (
            state_mask
            & (perturb == "identity")
            & (fold != held_fold)
        )

        ref = X[reference_mask]
        expected_ref = EXPECTED_SOURCE_GROUPS - EXPECTED_FOLD_GROUP_COUNTS[held_fold]
        if len(ref) != expected_ref:
            raise RuntimeError(
                f"Reference count mismatch state={sid} held_fold={held_fold}: "
                f"{len(ref)} != {expected_ref}"
            )

        mu = ref.mean(axis=0)
        sigma = ref.std(axis=0, ddof=0)
        sigma_eff = np.maximum(sigma, REFERENCE_STD_FLOOR)

        Z[state_mask] = (X[state_mask] - mu) / sigma_eff

        for j, feature in enumerate(SOURCE_FEATURE_NAMES):
            audit_rows.append({
                "held_group_fold": held_fold,
                "model_state_id": sid,
                "feature": feature,
                "reference_count": len(ref),
                "mean": float(mu[j]),
                "std": float(sigma[j]),
                "effective_std": float(sigma_eff[j]),
                "std_floored": int(sigma[j] < REFERENCE_STD_FLOOR),
            })

    if not np.isfinite(Z).all():
        raise RuntimeError(
            f"Non-finite MRZ feature matrix for held fold {held_fold}"
        )

    return Z, audit_rows


# ---------------------------------------------------------------------
# Predictor fitting
# ---------------------------------------------------------------------

def fit_predict_fixed(X_train, X_test, y_delta, y_harm, y_benefit):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge, LogisticRegression

    if len(np.unique(y_harm)) != 2:
        raise RuntimeError("Training HARM labels degenerate.")
    if len(np.unique(y_benefit)) != 2:
        raise RuntimeError("Training BENEFIT labels degenerate.")

    ridge = Pipeline([
        ("scale", StandardScaler()),
        ("model", Ridge(alpha=RIDGE_ALPHA)),
    ])
    harm_model = Pipeline([
        ("scale", StandardScaler()),
        (
            "model",
            LogisticRegression(
                C=LOGISTIC_C,
                class_weight="balanced",
                solver="lbfgs",
                max_iter=LOGISTIC_MAX_ITER,
                random_state=RANDOM_STATE,
            ),
        ),
    ])
    benefit_model = Pipeline([
        ("scale", StandardScaler()),
        (
            "model",
            LogisticRegression(
                C=LOGISTIC_C,
                class_weight="balanced",
                solver="lbfgs",
                max_iter=LOGISTIC_MAX_ITER,
                random_state=RANDOM_STATE,
            ),
        ),
    ])

    ridge.fit(X_train, y_delta)
    harm_model.fit(X_train, y_harm)
    benefit_model.fit(X_train, y_benefit)

    pred_delta = ridge.predict(X_test)
    p_harm = harm_model.predict_proba(X_test)[:, 1]
    p_benefit = benefit_model.predict_proba(X_test)[:, 1]

    coefficient_rows = []
    for estimator_name, pipeline in (
        ("ridge_delta", ridge),
        ("logistic_harm", harm_model),
        ("logistic_benefit", benefit_model),
    ):
        model = pipeline.named_steps["model"]
        coef = np.asarray(model.coef_, dtype=np.float64).reshape(-1)
        if len(coef) != len(SOURCE_FEATURE_NAMES):
            raise RuntimeError(
                f"Coefficient length mismatch for {estimator_name}"
            )
        intercept = float(np.asarray(model.intercept_).reshape(-1)[0])
        scale = pipeline.named_steps["scale"]
        for j, feature in enumerate(SOURCE_FEATURE_NAMES):
            coefficient_rows.append({
                "estimator": estimator_name,
                "feature": feature,
                "coefficient_standardized_space": float(coef[j]),
                "intercept": intercept,
                "training_scaler_mean": float(scale.mean_[j]),
                "training_scaler_scale": float(scale.scale_[j]),
            })

    return pred_delta, p_harm, p_benefit, coefficient_rows


# ---------------------------------------------------------------------
# Split evaluation
# ---------------------------------------------------------------------

def evaluate_joint_splits(arrays):
    all_prediction_rows = []
    split_metric_rows = []
    coefficient_rows = []
    reference_audit_rows = []
    split_definition_rows = []

    mrz_cache = {}
    mrz_audit_cache = {}

    total = len(FAMILIES) * len(FOLDS) * len(REPRESENTATIONS)
    pbar = tqdm(
        total=total,
        desc="Q1-R04 joint architecture+group evaluation",
        unit="fit",
        dynamic_ncols=True,
    )

    for held_fold in FOLDS:
        Z, ref_audit = build_mrz_for_fold(arrays, held_fold)
        mrz_cache[held_fold] = Z
        mrz_audit_cache[held_fold] = ref_audit

    # Write each reference statistic once per held group fold.
    for held_fold in FOLDS:
        reference_audit_rows.extend(mrz_audit_cache[held_fold])

    for held_family in FAMILIES:
        for held_fold in FOLDS:
            train_mask = (
                (arrays["family"] != held_family)
                & (arrays["fold"] != held_fold)
            )
            test_mask = (
                (arrays["family"] == held_family)
                & (arrays["fold"] == held_fold)
            )

            n_group = EXPECTED_FOLD_GROUP_COUNTS[held_fold]
            expected_test = 3 * n_group * EXPECTED_PERTURBATIONS
            expected_train = (
                2
                * 3
                * (EXPECTED_SOURCE_GROUPS - n_group)
                * EXPECTED_PERTURBATIONS
            )
            if int(test_mask.sum()) != expected_test:
                raise RuntimeError(
                    f"Test cardinality mismatch {held_family} fold={held_fold}: "
                    f"{test_mask.sum()} != {expected_test}"
                )
            if int(train_mask.sum()) != expected_train:
                raise RuntimeError(
                    f"Train cardinality mismatch {held_family} fold={held_fold}: "
                    f"{train_mask.sum()} != {expected_train}"
                )

            train_groups = set(arrays["sample"][train_mask].tolist())
            test_groups = set(arrays["sample"][test_mask].tolist())
            train_families = set(arrays["family"][train_mask].tolist())
            test_families = set(arrays["family"][test_mask].tolist())

            if train_groups & test_groups:
                raise RuntimeError("Case-group leakage.")
            if held_family in train_families:
                raise RuntimeError("Held architecture leakage.")
            if test_families != {held_family}:
                raise RuntimeError("Test family definition failure.")

            split_definition_rows.append({
                "held_out_architecture": held_family,
                "held_out_group_fold": held_fold,
                "train_rows": int(train_mask.sum()),
                "test_rows": int(test_mask.sum()),
                "train_unique_groups": len(train_groups),
                "test_unique_groups": len(test_groups),
                "train_families": ";".join(sorted(train_families)),
                "test_families": held_family,
                "architecture_leakage": 0,
                "group_leakage": 0,
            })

            for representation in REPRESENTATIONS:
                if representation == "RAW19":
                    X = arrays["X_raw"]
                elif representation == "MRZ19":
                    X = mrz_cache[held_fold]
                else:
                    raise RuntimeError(representation)

                X_train = X[train_mask]
                X_test = X[test_mask]

                (
                    pred_delta,
                    p_harm,
                    p_benefit,
                    fold_coef,
                ) = fit_predict_fixed(
                    X_train,
                    X_test,
                    arrays["delta"][train_mask],
                    arrays["harm"][train_mask],
                    arrays["benefit"][train_mask],
                )

                metrics = metric_bundle(
                    arrays["delta"][test_mask],
                    pred_delta,
                    arrays["harm"][test_mask],
                    p_harm,
                    arrays["benefit"][test_mask],
                    p_benefit,
                )

                split_metric_rows.append({
                    "held_out_architecture": held_family,
                    "held_out_group_fold": held_fold,
                    "representation": representation,
                    "train_rows": int(train_mask.sum()),
                    "test_rows": int(test_mask.sum()),
                    **metrics,
                })

                for r in fold_coef:
                    coefficient_rows.append({
                        "held_out_architecture": held_family,
                        "held_out_group_fold": held_fold,
                        "representation": representation,
                        **r,
                    })

                test_indices = np.flatnonzero(test_mask)
                for local_i, global_i in enumerate(test_indices):
                    meta = arrays["metadata"][global_i]
                    all_prediction_rows.append({
                        **meta,
                        "held_out_architecture": held_family,
                        "held_out_group_fold": held_fold,
                        "representation": representation,
                        "true_delta_dice": float(arrays["delta"][global_i]),
                        "true_harmful": int(arrays["harm"][global_i]),
                        "true_beneficial": int(arrays["benefit"][global_i]),
                        "pred_delta_dice": float(pred_delta[local_i]),
                        "pred_harm_probability": float(p_harm[local_i]),
                        "pred_benefit_probability": float(p_benefit[local_i]),
                    })

                pbar.update(1)
                pbar.set_postfix(
                    arch=held_family,
                    fold=held_fold,
                    rep=representation,
                    rho=f"{metrics['spearman']:.3f}",
                )

    pbar.close()

    # Exact cross-fit uniqueness: every upstream row exactly once per rep.
    for representation in REPRESENTATIONS:
        rr = [
            r for r in all_prediction_rows
            if r["representation"] == representation
        ]
        if len(rr) != EXPECTED_ROWS:
            raise RuntimeError(
                f"Crossfit rows mismatch {representation}: {len(rr)}"
            )
        keys = {
            (
                r["model_state_id"],
                r["source_group_id"],
                r["perturbation"],
            )
            for r in rr
        }
        if len(keys) != EXPECTED_ROWS:
            raise RuntimeError(
                f"Crossfit duplicate/missing keys {representation}"
            )

    return (
        all_prediction_rows,
        split_metric_rows,
        coefficient_rows,
        reference_audit_rows,
        split_definition_rows,
    )


# ---------------------------------------------------------------------
# Pooled metrics / decision
# ---------------------------------------------------------------------

def pooled_metrics(pred_rows, representation: str, family: str | None):
    rr = [r for r in pred_rows if r["representation"] == representation]
    if family is not None:
        rr = [r for r in rr if r["model_family"] == family]

    return metric_bundle(
        [float(r["true_delta_dice"]) for r in rr],
        [float(r["pred_delta_dice"]) for r in rr],
        [int(r["true_harmful"]) for r in rr],
        [float(r["pred_harm_probability"]) for r in rr],
        [int(r["true_beneficial"]) for r in rr],
        [float(r["pred_benefit_probability"]) for r in rr],
    )


def build_pooled_outputs(pred_rows):
    architecture_rows = []

    by_key = {}
    for representation in REPRESENTATIONS:
        for family in FAMILIES:
            m = pooled_metrics(pred_rows, representation, family)
            row = {
                "representation": representation,
                "held_out_architecture": family,
                "rows": EXPECTED_ROWS_PER_FAMILY,
                **m,
            }
            architecture_rows.append(row)
            by_key[(representation, family)] = row

    # Candidate gains vs raw.
    for family in FAMILIES:
        raw = by_key[("RAW19", family)]
        mr = by_key[("MRZ19", family)]
        mr["delta_spearman_vs_RAW19"] = mr["spearman"] - raw["spearman"]
        mr["delta_harm_auroc_vs_RAW19"] = mr["harm_auroc"] - raw["harm_auroc"]
        mr["delta_benefit_auroc_vs_RAW19"] = (
            mr["benefit_auroc"] - raw["benefit_auroc"]
        )
        mr["delta_safety_mean_auc_vs_RAW19"] = (
            mr["safety_mean_auc"] - raw["safety_mean_auc"]
        )

        raw["delta_spearman_vs_RAW19"] = 0.0
        raw["delta_harm_auroc_vs_RAW19"] = 0.0
        raw["delta_benefit_auroc_vs_RAW19"] = 0.0
        raw["delta_safety_mean_auc_vs_RAW19"] = 0.0

    global_rows = []
    for representation in REPRESENTATIONS:
        m = pooled_metrics(pred_rows, representation, None)
        global_rows.append({
            "representation": representation,
            "rows": EXPECTED_ROWS,
            **m,
        })

    return architecture_rows, global_rows


def evaluate_decision(architecture_rows):
    mr = {
        r["held_out_architecture"]: r
        for r in architecture_rows
        if r["representation"] == "MRZ19"
    }
    if set(mr) != set(FAMILIES):
        raise RuntimeError("Missing MRZ19 architecture rows.")

    spearman = np.asarray([mr[f]["spearman"] for f in FAMILIES])
    harm_auc = np.asarray([mr[f]["harm_auroc"] for f in FAMILIES])
    benefit_auc = np.asarray([mr[f]["benefit_auroc"] for f in FAMILIES])
    gain_rho = np.asarray(
        [mr[f]["delta_spearman_vs_RAW19"] for f in FAMILIES]
    )
    gain_safety = np.asarray(
        [mr[f]["delta_safety_mean_auc_vs_RAW19"] for f in FAMILIES]
    )

    actuals = {
        "median_family_spearman": float(np.median(spearman)),
        "minimum_family_spearman": float(np.min(spearman)),
        "median_family_harm_auroc": float(np.median(harm_auc)),
        "minimum_family_harm_auroc": float(np.min(harm_auc)),
        "median_family_benefit_auroc": float(np.median(benefit_auc)),
        "minimum_family_benefit_auroc": float(np.min(benefit_auc)),
        "median_delta_spearman_vs_RAW19": float(np.median(gain_rho)),
        "median_delta_safety_mean_auc_vs_RAW19": float(
            np.median(gain_safety)
        ),
        "minimum_delta_safety_mean_auc_vs_RAW19": float(
            np.min(gain_safety)
        ),
    }

    checks = {
        "A_median_spearman_ge_0_30":
            actuals["median_family_spearman"] >= 0.30,
        "A_min_spearman_ge_0_15":
            actuals["minimum_family_spearman"] >= 0.15,
        "B_median_harm_auroc_ge_0_75":
            actuals["median_family_harm_auroc"] >= 0.75,
        "B_min_harm_auroc_ge_0_65":
            actuals["minimum_family_harm_auroc"] >= 0.65,
        "C_median_benefit_auroc_ge_0_70":
            actuals["median_family_benefit_auroc"] >= 0.70,
        "C_min_benefit_auroc_ge_0_65":
            actuals["minimum_family_benefit_auroc"] >= 0.65,
        "D_median_delta_spearman_ge_0_05":
            actuals["median_delta_spearman_vs_RAW19"] >= 0.05,
        "D_median_delta_safety_auc_ge_0_03":
            actuals["median_delta_safety_mean_auc_vs_RAW19"] >= 0.03,
        "E_min_delta_safety_auc_ge_minus_0_02":
            actuals["minimum_delta_safety_mean_auc_vs_RAW19"] >= -0.02,
    }

    decision = DECISION_GO if all(checks.values()) else DECISION_STOP

    return {
        "thresholds": {
            "median_family_spearman": 0.30,
            "minimum_family_spearman": 0.15,
            "median_family_harm_auroc": 0.75,
            "minimum_family_harm_auroc": 0.65,
            "median_family_benefit_auroc": 0.70,
            "minimum_family_benefit_auroc": 0.65,
            "median_delta_spearman_vs_RAW19": 0.05,
            "median_delta_safety_mean_auc_vs_RAW19": 0.03,
            "minimum_delta_safety_mean_auc_vs_RAW19": -0.02,
        },
        "actuals": actuals,
        "checks": checks,
        "all_required_checks_pass": bool(all(checks.values())),
        "decision": decision,
    }


# ---------------------------------------------------------------------
# Preflight / formal run
# ---------------------------------------------------------------------

def preflight():
    rows, fields, lock = load_and_validate_upstream()
    arrays = rows_to_arrays(rows)

    # Validate one strict joint split and MR reference construction.
    held_family = FAMILIES[0]
    held_fold = FOLDS[0]
    Z, ref_audit = build_mrz_for_fold(arrays, held_fold)

    train_mask = (
        (arrays["family"] != held_family)
        & (arrays["fold"] != held_fold)
    )
    test_mask = (
        (arrays["family"] == held_family)
        & (arrays["fold"] == held_fold)
    )

    if set(arrays["sample"][train_mask]) & set(arrays["sample"][test_mask]):
        raise RuntimeError("Preflight group leakage.")
    if held_family in set(arrays["family"][train_mask]):
        raise RuntimeError("Preflight architecture leakage.")

    try:
        import sklearn
    except Exception as e:
        raise RuntimeError(f"scikit-learn unavailable: {e}") from e

    print("===== Q1-R04 PREFLIGHT =====")
    print(f"protocol_sha256={EXPECTED_PROTOCOL_SHA256}")
    print(f"r03_lock_sha256={EXPECTED_R03_LOCK_SHA256}")
    print(f"r03_table_sha256={EXPECTED_R03_TABLE_SHA256}")
    print(f"rows={len(rows)}")
    print(f"families={len(set(arrays['family']))}")
    print(f"states={len(set(arrays['state']))}")
    print(f"source_groups={len(set(arrays['sample']))}")
    print(f"features={arrays['X_raw'].shape[1]}")
    print(f"representations={REPRESENTATIONS}")
    print(f"joint_splits={len(FAMILIES) * len(FOLDS)}")
    print(
        f"example_split={held_family}+fold{held_fold} "
        f"train={int(train_mask.sum())} test={int(test_mask.sum())}"
    )
    print(f"reference_stat_rows_for_one_fold={len(ref_audit)}")
    print(f"sklearn_version={sklearn.__version__}")
    print("source_dice_predictor_input=NO")
    print("model_identity_predictor_input=NO")
    print("perturbation_identity_predictor_input=NO")
    print("target_data=NO")
    print("predictor_fit=NO")
    print("PREFLIGHT_PASS")


def run(args):
    rows, fields, upstream_lock = load_and_validate_upstream()
    arrays = rows_to_arrays(rows)

    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    build_dir = Path(str(args.output_dir) + "__building")
    if build_dir.exists():
        raise FileExistsError(
            f"Partial output already exists: {build_dir}"
        )
    build_dir.mkdir(parents=True, exist_ok=False)

    protocol_copy = build_dir / "preregistered_protocol_copy.md"
    shutil.copy2(PROTOCOL, protocol_copy)
    validate_sha(
        protocol_copy,
        EXPECTED_PROTOCOL_SHA256,
        "Q1-R04 protocol copy",
    )

    upstream_audit = {
        "script_version": VERSION,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "r03_lock_sha256": EXPECTED_R03_LOCK_SHA256,
        "r03_table_sha256": EXPECTED_R03_TABLE_SHA256,
        "r03_decision": upstream_lock.get("decision"),
        "rows": len(rows),
        "model_states": len(set(arrays["state"])),
        "families": sorted(set(arrays["family"])),
        "source_groups": len(set(arrays["sample"])),
        "target_data_used": False,
        "source_dice_used_as_predictor": False,
        "model_identity_used_as_predictor": False,
        "perturbation_identity_used_as_predictor": False,
        "hyperparameter_sweep": False,
        "feature_selection": False,
    }
    upstream_audit_path = build_dir / "upstream_audit.json"
    write_json(upstream_audit_path, upstream_audit)

    (
        pred_rows,
        split_metrics,
        coefficient_rows,
        reference_rows,
        split_definition_rows,
    ) = evaluate_joint_splits(arrays)

    architecture_rows, global_rows = build_pooled_outputs(pred_rows)
    criteria = evaluate_decision(architecture_rows)
    decision = criteria["decision"]

    reference_path = build_dir / "reference_statistics_audit.csv"
    write_csv(
        reference_path,
        reference_rows,
        [
            "held_group_fold",
            "model_state_id",
            "feature",
            "reference_count",
            "mean",
            "std",
            "effective_std",
            "std_floored",
        ],
    )

    split_def_path = build_dir / "split_definition_audit.csv"
    write_csv(
        split_def_path,
        split_definition_rows,
        [
            "held_out_architecture",
            "held_out_group_fold",
            "train_rows",
            "test_rows",
            "train_unique_groups",
            "test_unique_groups",
            "train_families",
            "test_families",
            "architecture_leakage",
            "group_leakage",
        ],
    )

    split_metrics_path = build_dir / "split_metrics.csv"
    split_metric_fields = [
        "held_out_architecture",
        "held_out_group_fold",
        "representation",
        "train_rows",
        "test_rows",
        "spearman",
        "pearson",
        "mae",
        "harm_auroc",
        "harm_auprc",
        "benefit_auroc",
        "benefit_auprc",
        "safety_mean_auc",
        "harm_prevalence",
        "benefit_prevalence",
    ]
    write_csv(split_metrics_path, split_metrics, split_metric_fields)

    coef_path = build_dir / "fold_coefficients.csv"
    write_csv(
        coef_path,
        coefficient_rows,
        [
            "held_out_architecture",
            "held_out_group_fold",
            "representation",
            "estimator",
            "feature",
            "coefficient_standardized_space",
            "intercept",
            "training_scaler_mean",
            "training_scaler_scale",
        ],
    )

    pred_path = build_dir / "crossfitted_predictions.csv"
    pred_fields = [
        "model_family",
        "model_state_id",
        "training_seed",
        "checkpoint_sha256",
        "sample_id",
        "source_group_id",
        "fold",
        "perturbation",
        "held_out_architecture",
        "held_out_group_fold",
        "representation",
        "true_delta_dice",
        "true_harmful",
        "true_beneficial",
        "pred_delta_dice",
        "pred_harm_probability",
        "pred_benefit_probability",
    ]
    write_csv(pred_path, pred_rows, pred_fields)

    arch_path = build_dir / "architecture_pooled_metrics.csv"
    arch_fields = [
        "representation",
        "held_out_architecture",
        "rows",
        "spearman",
        "pearson",
        "mae",
        "harm_auroc",
        "harm_auprc",
        "benefit_auroc",
        "benefit_auprc",
        "safety_mean_auc",
        "harm_prevalence",
        "benefit_prevalence",
        "delta_spearman_vs_RAW19",
        "delta_harm_auroc_vs_RAW19",
        "delta_benefit_auroc_vs_RAW19",
        "delta_safety_mean_auc_vs_RAW19",
    ]
    write_csv(arch_path, architecture_rows, arch_fields)

    global_path = build_dir / "global_pooled_metrics.csv"
    global_fields = [
        "representation",
        "rows",
        "spearman",
        "pearson",
        "mae",
        "harm_auroc",
        "harm_auprc",
        "benefit_auroc",
        "benefit_auprc",
        "safety_mean_auc",
        "harm_prevalence",
        "benefit_prevalence",
    ]
    write_csv(global_path, global_rows, global_fields)

    criteria_path = build_dir / "decision_criteria.json"
    write_json(criteria_path, criteria)

    decision_path = build_dir / "decision.txt"
    decision_path.write_text(decision + "\n", encoding="utf-8")

    # Human-readable summary.
    raw_by_family = {
        r["held_out_architecture"]: r
        for r in architecture_rows
        if r["representation"] == "RAW19"
    }
    mr_by_family = {
        r["held_out_architecture"]: r
        for r in architecture_rows
        if r["representation"] == "MRZ19"
    }
    global_by_rep = {
        r["representation"]: r
        for r in global_rows
    }

    lines = [
        "===== Q1-R04 MODEL-RELATIVE PROSPECTIVE UTILITY FEASIBILITY =====",
        f"Script version: {VERSION}",
        f"Build: {BUILD}",
        "",
        "Frozen evaluation:",
        "  primary=joint Leave-One-Architecture + source-group-fold out",
        "  splits=15",
        "  representations=RAW19, MRZ19",
        "  feature count=19",
        "  nonlinear model=NO",
        "  hyperparameter sweep=NO",
        "  source_dice predictor input=NO",
        "  model identity predictor input=NO",
        "  perturbation identity predictor input=NO",
        "  target data=NO",
        "",
        "Architecture-pooled cross-fitted metrics:",
    ]

    for family in FAMILIES:
        raw = raw_by_family[family]
        mr = mr_by_family[family]
        lines += [
            f"  {family}:",
            (
                "    RAW19: "
                f"rho={raw['spearman']:.6f} "
                f"HarmAUROC={raw['harm_auroc']:.6f} "
                f"BenefitAUROC={raw['benefit_auroc']:.6f} "
                f"SafetyMean={raw['safety_mean_auc']:.6f}"
            ),
            (
                "    MRZ19: "
                f"rho={mr['spearman']:.6f} "
                f"HarmAUROC={mr['harm_auroc']:.6f} "
                f"BenefitAUROC={mr['benefit_auroc']:.6f} "
                f"SafetyMean={mr['safety_mean_auc']:.6f}"
            ),
            (
                "    gains: "
                f"dRho={mr['delta_spearman_vs_RAW19']:+.6f} "
                f"dHarm={mr['delta_harm_auroc_vs_RAW19']:+.6f} "
                f"dBenefit={mr['delta_benefit_auroc_vs_RAW19']:+.6f} "
                f"dSafety={mr['delta_safety_mean_auc_vs_RAW19']:+.6f}"
            ),
        ]

    lines += [
        "",
        "Global cross-fitted pooled metrics:",
        (
            "  RAW19: "
            f"rho={global_by_rep['RAW19']['spearman']:.6f} "
            f"HarmAUROC={global_by_rep['RAW19']['harm_auroc']:.6f} "
            f"BenefitAUROC={global_by_rep['RAW19']['benefit_auroc']:.6f}"
        ),
        (
            "  MRZ19: "
            f"rho={global_by_rep['MRZ19']['spearman']:.6f} "
            f"HarmAUROC={global_by_rep['MRZ19']['harm_auroc']:.6f} "
            f"BenefitAUROC={global_by_rep['MRZ19']['benefit_auroc']:.6f}"
        ),
        "",
        "Preregistered decision actuals:",
    ]

    for k, v in criteria["actuals"].items():
        lines.append(f"  {k}={v:.12g}")

    lines += [
        "",
        "Required checks:",
    ]
    for k, v in criteria["checks"].items():
        lines.append(f"  {k}={'PASS' if v else 'FAIL'}")

    lines += [
        "",
        "Decision:",
        f"  {decision}",
    ]

    if decision == DECISION_GO:
        lines += [
            "",
            "If GO:",
            "  next = Q1-R05 frozen target-domain prospective utility validation",
        ]
    else:
        lines += [
            "",
            "If STOP:",
            "  no post-hoc feature/model/threshold rescue inside Q1-R04",
        ]

    run_log = "\n".join(lines) + "\n"
    run_log_path = build_dir / "run_log.txt"
    run_log_path.write_text(run_log, encoding="utf-8")

    artifact_paths = {
        "protocol_copy": protocol_copy,
        "upstream_audit": upstream_audit_path,
        "reference_statistics_audit": reference_path,
        "split_definition_audit": split_def_path,
        "split_metrics": split_metrics_path,
        "fold_coefficients": coef_path,
        "crossfitted_predictions": pred_path,
        "architecture_pooled_metrics": arch_path,
        "global_pooled_metrics": global_path,
        "decision_criteria": criteria_path,
        "decision": decision_path,
        "run_log": run_log_path,
    }

    lock = {
        "script_version": VERSION,
        "build": BUILD,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "r03_lock_sha256": EXPECTED_R03_LOCK_SHA256,
        "r03_table_sha256": EXPECTED_R03_TABLE_SHA256,
        "representations": list(REPRESENTATIONS),
        "features": list(SOURCE_FEATURE_NAMES),
        "joint_splits": 15,
        "crossfitted_rows_per_representation": EXPECTED_ROWS,
        "target_data_used": False,
        "source_dice_used_as_predictor": False,
        "model_identity_used_as_predictor": False,
        "perturbation_identity_used_as_predictor": False,
        "hyperparameter_sweep": False,
        "feature_selection": False,
        "decision": decision,
        "artifacts": {
            name: {
                "relative_path": str(path.relative_to(build_dir)),
                "sha256": file_sha256(path),
            }
            for name, path in artifact_paths.items()
        },
    }

    lock_path = build_dir / "Q1_R04_MODEL_RELATIVE_UTILITY_LOCK.json"
    write_json(lock_path, lock)
    lock_sha = file_sha256(lock_path)

    for name, meta in lock["artifacts"].items():
        p = build_dir / meta["relative_path"]
        if file_sha256(p) != meta["sha256"]:
            raise RuntimeError(f"Artifact changed before commit: {name}")

    build_dir.rename(args.output_dir)

    print()
    print((args.output_dir / "run_log.txt").read_text(encoding="utf-8"))
    print(
        "Q1-R04 LOCK:",
        args.output_dir / "Q1_R04_MODEL_RELATIVE_UTILITY_LOCK.json",
    )
    print("Q1-R04 LOCK SHA256:", lock_sha)


# ---------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------

def self_test():
    assert len(SOURCE_FEATURE_NAMES) == 19
    assert len(FAMILIES) == 3
    assert len(FOLDS) == 5
    assert len(REPRESENTATIONS) == 2
    assert sum(EXPECTED_FOLD_GROUP_COUNTS.values()) == 145

    # Rank ties.
    x = np.asarray([10.0, 20.0, 20.0, 40.0])
    ranks = average_rank(x)
    assert np.allclose(ranks, [1.0, 2.5, 2.5, 4.0])

    # Correlation.
    a = np.asarray([1, 2, 3, 4, 5], dtype=np.float64)
    b = np.asarray([2, 4, 6, 8, 10], dtype=np.float64)
    assert abs(pearson_corr(a, b) - 1.0) < 1e-12
    assert abs(spearman_corr(a, b) - 1.0) < 1e-12

    # Synthetic model-relative normalization with the exact frozen
    # 145-group fold cardinality expected by the formal R04 protocol.
    fold_sequence = []
    for f in FOLDS:
        fold_sequence.extend([f] * EXPECTED_FOLD_GROUP_COUNTS[f])
    fold_sequence = np.asarray(fold_sequence, dtype=np.int64)
    assert len(fold_sequence) == EXPECTED_SOURCE_GROUPS

    n_states = 2
    n = EXPECTED_SOURCE_GROUPS * n_states
    X = np.zeros((n, len(SOURCE_FEATURE_NAMES)), dtype=np.float64)
    state = np.asarray(
        ["A"] * EXPECTED_SOURCE_GROUPS
        + ["B"] * EXPECTED_SOURCE_GROUPS,
        dtype=object,
    )
    perturb = np.asarray(["identity"] * n, dtype=object)
    fold = np.concatenate([fold_sequence, fold_sequence])

    # Same relative pattern with a large model-state absolute offset.
    base = np.linspace(
        -1.0,
        1.0,
        EXPECTED_SOURCE_GROUPS,
        dtype=np.float64,
    )
    for j in range(len(SOURCE_FEATURE_NAMES)):
        X[:EXPECTED_SOURCE_GROUPS, j] = base * (j + 1) * 0.01
        X[EXPECTED_SOURCE_GROUPS:, j] = (
            100.0 + base * (j + 1) * 0.01
        )

    arrays = {
        "X_raw": X,
        "state": state,
        "perturbation": perturb,
        "fold": fold,
    }
    Z, audit = build_mrz_for_fold(arrays, held_fold=4)
    assert Z.shape == X.shape
    assert np.isfinite(Z).all()
    assert len(audit) == n_states * len(SOURCE_FEATURE_NAMES)

    expected_reference_count = (
        EXPECTED_SOURCE_GROUPS - EXPECTED_FOLD_GROUP_COUNTS[4]
    )
    assert {
        int(r["reference_count"]) for r in audit
    } == {expected_reference_count}

    # Thresholds immutable.
    assert HARM_THRESHOLD == -0.02
    assert BENEFIT_THRESHOLD == 0.02
    assert RIDGE_ALPHA == 1.0
    assert LOGISTIC_C == 1.0
    assert REFERENCE_STD_FLOOR == 1e-6

    print("FEATURE_COUNT_TEST_PASS")
    print("RANK_CORRELATION_TEST_PASS")
    print("MODEL_RELATIVE_NORMALIZATION_TEST_PASS")
    print("FROZEN_THRESHOLD_TEST_PASS")
    print("SELF_TEST_PASS")


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Q1-R04: test RAW19 vs model-relative MRZ19 prospective utility "
            "under joint held-out architecture and held-out source-group folds."
        )
    )
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    p.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "Validate the frozen Q1-R03 asset and strict joint split/reference "
            "construction. No utility predictor is fitted."
        ),
    )
    p.add_argument("--self-test", action="store_true")
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
