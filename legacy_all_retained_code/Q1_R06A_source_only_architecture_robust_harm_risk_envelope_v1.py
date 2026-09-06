#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Q1-R06A — Source-only architecture-robust prospective HARM risk envelope.

Methods:
  B0 = RAW19 pooled logistic
  B1 = MRZ19 pooled logistic
  H1 = MRZ19 two single-family experts, risk=max(p1,p2)

No target data are read.
No probability threshold is selected.
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
from typing import Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


VERSION = "2026-08-19-Q1-R06A-v1"
BUILD = "Q1_R06A_SOURCE_ONLY_ARCHITECTURE_ROBUST_HARM_RISK_ENVELOPE"

ROOT = Path(r"F:\MEDSEG_SAFETTA")

PROTOCOL = (
    ROOT / "docs"
    / "Q1_R06A_source_only_architecture_robust_harm_risk_envelope_preregistered_protocol_v1.md"
)
EXPECTED_PROTOCOL_SHA256 = "35a087ccea82c811c7a8158d41498395eb4fc33ec71530e7f777c3e7cbc800e9"

R03_DIR = (
    ROOT / "outputs"
    / "Q1_R03_nine_state_heterogeneous_source_utility_asset_v1_fix2"
)
R03_LOCK = R03_DIR / "Q1_R03_NINE_STATE_SOURCE_UTILITY_LOCK.json"
R03_TABLE = R03_DIR / "nine_state_source_utility_table.csv"

EXPECTED_R03_LOCK_SHA256 = (
    "2c773f5eb70bedb50e2c231ef2512c06dff5ecca425eb43b2719637c1bc06dad"
)
EXPECTED_R03_TABLE_SHA256 = (
    "969ef666c4e54d0b6f3152709b99a601ff3b82ce6b9f26d46efeb23e53960642"
)
EXPECTED_R03_DECISION = "NINE_STATE_SOURCE_UTILITY_ASSET_READY"

R04_LOCK = (
    ROOT / "outputs"
    / "Q1_R04_model_relative_prospective_utility_feasibility_loao_v1_fix1"
    / "Q1_R04_MODEL_RELATIVE_UTILITY_LOCK.json"
)
EXPECTED_R04_LOCK_SHA256 = (
    "9147a3b3f9f3d7526c057c4c049ac3119ad0f82dfc7b529b32883164ee03cc3a"
)
EXPECTED_R04_DECISION = "STOP_MODEL_RELATIVE_UTILITY_NO_ROBUST_LOAO_SIGNAL"

OUTPUT_DIR = (
    ROOT / "outputs"
    / "Q1_R06A_source_only_architecture_robust_harm_risk_envelope_v1"
)

FAMILIES = ("PraNet", "DeepLabV3-R50", "SegFormer-B0")
FOLDS = (0, 1, 2, 3, 4)
METHODS = ("B0_RAW19_POOLED", "B1_MRZ19_POOLED", "H1_MRZ19_MAX_ENVELOPE")

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

REFERENCE_STD_FLOOR = 1e-6
LOGISTIC_C = 1.0
LOGISTIC_MAX_ITER = 5000
RANDOM_STATE = 20260819
HARM_THRESHOLD = -0.02

EXPECTED_ROWS = 13050
EXPECTED_ROWS_PER_FAMILY = 4350
EXPECTED_STATES = 9
EXPECTED_SOURCE_GROUPS = 145
EXPECTED_PERTURBATIONS = 10

GO_CRITERIA = {
    "median_architecture_h1_auroc_min": 0.76,
    "minimum_architecture_h1_auroc_min": 0.72,
    "median_architecture_h1_minus_b1_auroc_min": 0.01,
    "minimum_architecture_h1_minus_b1_auroc_min": -0.01,
}

DECISION_GO = "GO_HARM_RISK_ENVELOPE_FOR_R06B"
DECISION_STOP = "STOP_MAX_HARM_RISK_ENVELOPE_NO_ROBUST_GAIN"


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
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
    actual = sha256_file(path)
    if actual.lower() != str(expected).lower():
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


def make_logistic():
    return Pipeline([
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


def validate_and_load():
    validate_sha(PROTOCOL, EXPECTED_PROTOCOL_SHA256, "R06A protocol")
    validate_sha(R03_LOCK, EXPECTED_R03_LOCK_SHA256, "R03 lock")
    validate_sha(R03_TABLE, EXPECTED_R03_TABLE_SHA256, "R03 table")
    validate_sha(R04_LOCK, EXPECTED_R04_LOCK_SHA256, "R04 lock")

    r03 = json.loads(R03_LOCK.read_text(encoding="utf-8"))
    r04 = json.loads(R04_LOCK.read_text(encoding="utf-8"))

    if r03.get("decision") != EXPECTED_R03_DECISION:
        raise RuntimeError(f"Unexpected R03 decision: {r03.get('decision')}")
    if bool(r03.get("target_data_used", True)):
        raise RuntimeError("R03 lock reports target-data use.")
    if int(r03.get("rows", -1)) != EXPECTED_ROWS:
        raise RuntimeError("R03 row count changed.")
    if int(r03.get("model_states", -1)) != EXPECTED_STATES:
        raise RuntimeError("R03 model-state count changed.")

    if r04.get("decision") != EXPECTED_R04_DECISION:
        raise RuntimeError(f"Unexpected R04 decision: {r04.get('decision')}")
    if bool(r04.get("target_data_used", True)):
        raise RuntimeError("R04 lock reports target-data use.")

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
        raise RuntimeError(f"R03 table missing columns: {missing}")
    if len(rows) != EXPECTED_ROWS:
        raise RuntimeError(f"R03 rows={len(rows)} expected={EXPECTED_ROWS}")

    family_counts = Counter(r["model_family"] for r in rows)
    if set(family_counts) != set(FAMILIES):
        raise RuntimeError(f"Family set mismatch: {dict(family_counts)}")
    if any(family_counts[f] != EXPECTED_ROWS_PER_FAMILY for f in FAMILIES):
        raise RuntimeError(f"Family row mismatch: {dict(family_counts)}")

    state_counts = Counter(r["model_state_id"] for r in rows)
    if len(state_counts) != EXPECTED_STATES:
        raise RuntimeError(f"State count={len(state_counts)}")
    if any(v != 1450 for v in state_counts.values()):
        raise RuntimeError(f"State row mismatch: {dict(state_counts)}")

    group_folds = defaultdict(set)
    group_families = defaultdict(set)
    for r in rows:
        group_folds[r["source_group_id"]].add(int(r["fold"]))
        group_families[r["source_group_id"]].add(r["model_family"])

    if len(group_folds) != EXPECTED_SOURCE_GROUPS:
        raise RuntimeError(f"Source groups={len(group_folds)}")
    if any(len(v) != 1 for v in group_folds.values()):
        raise RuntimeError("Source-group fold leakage in R03.")
    if any(v != set(FAMILIES) for v in group_families.values()):
        raise RuntimeError("Not every source group occurs in every family.")

    fold_counts = Counter(next(iter(v)) for v in group_folds.values())
    if dict(sorted(fold_counts.items())) != EXPECTED_FOLD_GROUP_COUNTS:
        raise RuntimeError(f"Fold-group counts changed: {dict(fold_counts)}")

    perturbations = {r["perturbation"] for r in rows}
    if len(perturbations) != EXPECTED_PERTURBATIONS or "identity" not in perturbations:
        raise RuntimeError(f"Perturbation set invalid: {sorted(perturbations)}")

    for i, r in enumerate(rows):
        delta = float(r["delta_dice"])
        h = int(r["harmful"])
        if h != int(delta <= HARM_THRESHOLD):
            raise RuntimeError(f"HARM label mismatch row={i}")
        vals = np.asarray(
            [float(r[f]) for f in SOURCE_FEATURE_NAMES],
            dtype=np.float64,
        )
        if not np.isfinite(vals).all():
            raise RuntimeError(f"Non-finite feature row={i}")

    return rows, r03, r04


def rows_to_arrays(rows):
    return {
        "X_raw": np.asarray(
            [[float(r[f]) for f in SOURCE_FEATURE_NAMES] for r in rows],
            dtype=np.float64,
        ),
        "family": np.asarray([r["model_family"] for r in rows], dtype=object),
        "state": np.asarray([r["model_state_id"] for r in rows], dtype=object),
        "seed": np.asarray([int(r["training_seed"]) for r in rows], dtype=np.int64),
        "sample": np.asarray([r["source_group_id"] for r in rows], dtype=object),
        "fold": np.asarray([int(r["fold"]) for r in rows], dtype=np.int64),
        "perturb": np.asarray([r["perturbation"] for r in rows], dtype=object),
        "harm": np.asarray([int(r["harmful"]) for r in rows], dtype=np.int64),
        "metadata": [
            {
                "model_family": r["model_family"],
                "model_state_id": r["model_state_id"],
                "training_seed": int(r["training_seed"]),
                "checkpoint_sha256": r["checkpoint_sha256"],
                "sample_id": r["sample_id"],
                "source_group_id": r["source_group_id"],
                "fold": int(r["fold"]),
                "perturbation": r["perturbation"],
                "true_delta_dice": float(r["delta_dice"]),
                "true_harmful": int(r["harmful"]),
            }
            for r in rows
        ],
    }


def build_fold_safe_mrz(arrays, held_fold: int):
    X = arrays["X_raw"]
    Z = np.empty_like(X)
    audit = []

    n_held = EXPECTED_FOLD_GROUP_COUNTS[held_fold]
    expected_ref = EXPECTED_SOURCE_GROUPS - n_held

    for sid in sorted(set(arrays["state"].tolist())):
        ref_mask = (
            (arrays["state"] == sid)
            & (arrays["perturb"] == "identity")
            & (arrays["fold"] != held_fold)
        )
        ref = X[ref_mask]

        if len(ref) != expected_ref:
            raise RuntimeError(
                f"MRZ reference state={sid} fold={held_fold}: "
                f"{len(ref)} != {expected_ref}"
            )

        mu = ref.mean(axis=0)
        std = ref.std(axis=0, ddof=0)
        eff = np.maximum(std, REFERENCE_STD_FLOOR)

        state_mask = arrays["state"] == sid
        Z[state_mask] = (X[state_mask] - mu) / eff

        family = str(arrays["family"][np.flatnonzero(state_mask)[0]])
        seed = int(arrays["seed"][np.flatnonzero(state_mask)[0]])

        for j, feature in enumerate(SOURCE_FEATURE_NAMES):
            audit.append({
                "held_out_group_fold": held_fold,
                "model_family": family,
                "model_state_id": sid,
                "training_seed": seed,
                "reference_count": len(ref),
                "feature": feature,
                "mean": float(mu[j]),
                "std": float(std[j]),
                "effective_std": float(eff[j]),
                "std_floored": int(std[j] < REFERENCE_STD_FLOOR),
            })

    if not np.isfinite(Z).all():
        raise RuntimeError(f"Non-finite MRZ fold={held_fold}")
    return Z, audit


def safe_metrics(y, p):
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(p, dtype=np.float64)
    if len(np.unique(y)) < 2:
        raise RuntimeError("HARM evaluation split is class-degenerate.")
    return {
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "positive": int(y.sum()),
        "negative": int(len(y) - y.sum()),
    }


def corr(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if np.std(a) <= 0 or np.std(b) <= 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def evaluate(arrays):
    predictions = []
    split_metrics = []
    split_defs = []
    expert_diag = []
    reference_rows = []

    mrz_by_fold = {}
    for held_fold in FOLDS:
        Z, audit = build_fold_safe_mrz(arrays, held_fold)
        mrz_by_fold[held_fold] = Z
        reference_rows.extend(audit)

    pbar = tqdm(
        total=len(FAMILIES) * len(FOLDS),
        desc="Q1-R06A strict LOAO+group",
        unit="split",
        dynamic_ncols=True,
    )

    for held_family in FAMILIES:
        expert_families = tuple(f for f in FAMILIES if f != held_family)

        for held_fold in FOLDS:
            n_group = EXPECTED_FOLD_GROUP_COUNTS[held_fold]

            train_pool = (
                (arrays["family"] != held_family)
                & (arrays["fold"] != held_fold)
            )
            test_mask = (
                (arrays["family"] == held_family)
                & (arrays["fold"] == held_fold)
            )

            expected_train = 2 * 3 * (EXPECTED_SOURCE_GROUPS - n_group) * EXPECTED_PERTURBATIONS
            expected_test = 3 * n_group * EXPECTED_PERTURBATIONS

            if int(train_pool.sum()) != expected_train:
                raise RuntimeError(
                    f"Train count {held_family}/fold{held_fold}: "
                    f"{train_pool.sum()} != {expected_train}"
                )
            if int(test_mask.sum()) != expected_test:
                raise RuntimeError(
                    f"Test count {held_family}/fold{held_fold}: "
                    f"{test_mask.sum()} != {expected_test}"
                )

            train_groups = set(arrays["sample"][train_pool].tolist())
            test_groups = set(arrays["sample"][test_mask].tolist())
            if train_groups & test_groups:
                raise RuntimeError("Source-group leakage.")
            if held_family in set(arrays["family"][train_pool].tolist()):
                raise RuntimeError("Held architecture leakage.")

            split_defs.append({
                "held_out_architecture": held_family,
                "held_out_group_fold": held_fold,
                "train_rows": int(train_pool.sum()),
                "test_rows": int(test_mask.sum()),
                "train_unique_groups": len(train_groups),
                "test_unique_groups": len(test_groups),
                "training_families": ";".join(expert_families),
                "architecture_leakage": 0,
                "group_leakage": 0,
            })

            Xraw = arrays["X_raw"]
            Xmrz = mrz_by_fold[held_fold]
            y_train = arrays["harm"][train_pool]
            y_test = arrays["harm"][test_mask]

            if len(np.unique(y_train)) != 2:
                raise RuntimeError("Pooled HARM training labels degenerate.")

            b0 = make_logistic()
            b0.fit(Xraw[train_pool], y_train)
            p_b0 = b0.predict_proba(Xraw[test_mask])[:, 1]

            b1 = make_logistic()
            b1.fit(Xmrz[train_pool], y_train)
            p_b1 = b1.predict_proba(Xmrz[test_mask])[:, 1]

            expert_probs = []
            expert_meta = []

            for expert_family in expert_families:
                expert_train = (
                    (arrays["family"] == expert_family)
                    & (arrays["fold"] != held_fold)
                )
                expected_expert_train = 3 * (EXPECTED_SOURCE_GROUPS - n_group) * EXPECTED_PERTURBATIONS
                if int(expert_train.sum()) != expected_expert_train:
                    raise RuntimeError(
                        f"Expert count {expert_family} -> {held_family}/fold{held_fold}: "
                        f"{expert_train.sum()} != {expected_expert_train}"
                    )

                y_expert = arrays["harm"][expert_train]
                if len(np.unique(y_expert)) != 2:
                    raise RuntimeError(
                        f"Expert HARM labels degenerate: {expert_family}/fold{held_fold}"
                    )

                model = make_logistic()
                model.fit(Xmrz[expert_train], y_expert)
                p = model.predict_proba(Xmrz[test_mask])[:, 1]
                expert_probs.append(p)
                expert_meta.append({
                    "expert_family": expert_family,
                    "training_rows": int(expert_train.sum()),
                    "positive_rows": int(y_expert.sum()),
                })

            p_e1, p_e2 = expert_probs
            p_h1 = np.maximum(p_e1, p_e2)

            method_probs = {
                "B0_RAW19_POOLED": p_b0,
                "B1_MRZ19_POOLED": p_b1,
                "H1_MRZ19_MAX_ENVELOPE": p_h1,
            }

            for method, prob in method_probs.items():
                m = safe_metrics(y_test, prob)
                split_metrics.append({
                    "held_out_architecture": held_family,
                    "held_out_group_fold": held_fold,
                    "method": method,
                    "train_rows": (
                        int(train_pool.sum())
                        if method != "H1_MRZ19_MAX_ENVELOPE"
                        else int(sum(x["training_rows"] for x in expert_meta))
                    ),
                    "test_rows": int(test_mask.sum()),
                    **m,
                })

            expert_diag.append({
                "held_out_architecture": held_family,
                "held_out_group_fold": held_fold,
                "expert_1_family": expert_families[0],
                "expert_2_family": expert_families[1],
                "expert_1_training_rows": expert_meta[0]["training_rows"],
                "expert_2_training_rows": expert_meta[1]["training_rows"],
                "expert_1_positive_rows": expert_meta[0]["positive_rows"],
                "expert_2_positive_rows": expert_meta[1]["positive_rows"],
                "expert_probability_pearson": corr(p_e1, p_e2),
                "mean_abs_expert_probability_difference":
                    float(np.mean(np.abs(p_e1 - p_e2))),
                "fraction_expert1_higher": float(np.mean(p_e1 > p_e2)),
                "fraction_expert2_higher": float(np.mean(p_e2 > p_e1)),
                "fraction_exact_tie": float(np.mean(p_e1 == p_e2)),
            })

            test_indices = np.flatnonzero(test_mask)
            for local_i, global_i in enumerate(test_indices):
                meta = arrays["metadata"][global_i]
                predictions.append({
                    **meta,
                    "held_out_architecture": held_family,
                    "held_out_group_fold": held_fold,
                    "p_b0_raw19_pooled": float(p_b0[local_i]),
                    "p_b1_mrz19_pooled": float(p_b1[local_i]),
                    "expert_1_family": expert_families[0],
                    "expert_1_harm_probability": float(p_e1[local_i]),
                    "expert_2_family": expert_families[1],
                    "expert_2_harm_probability": float(p_e2[local_i]),
                    "p_h1_mrz19_max_envelope": float(p_h1[local_i]),
                })

            pbar.update(1)
            pbar.set_postfix(
                arch=held_family,
                fold=held_fold,
                h1=f"{safe_metrics(y_test, p_h1)['auroc']:.3f}",
            )

    pbar.close()

    if len(predictions) != EXPECTED_ROWS:
        raise RuntimeError(
            f"OOF prediction rows={len(predictions)} expected={EXPECTED_ROWS}"
        )

    keys = {
        (
            r["model_state_id"],
            r["source_group_id"],
            r["perturbation"],
        )
        for r in predictions
    }
    if len(keys) != EXPECTED_ROWS:
        raise RuntimeError("OOF row key duplication/missing.")

    return predictions, split_metrics, split_defs, expert_diag, reference_rows


def summarize_architecture(predictions):
    rows = []
    field_map = {
        "B0_RAW19_POOLED": "p_b0_raw19_pooled",
        "B1_MRZ19_POOLED": "p_b1_mrz19_pooled",
        "H1_MRZ19_MAX_ENVELOPE": "p_h1_mrz19_max_envelope",
    }

    for family in FAMILIES:
        subset = [r for r in predictions if r["model_family"] == family]
        if len(subset) != EXPECTED_ROWS_PER_FAMILY:
            raise RuntimeError(f"Family OOF rows {family}={len(subset)}")

        y = np.asarray([int(r["true_harmful"]) for r in subset], dtype=np.int64)

        for method, field in field_map.items():
            p = np.asarray([float(r[field]) for r in subset], dtype=np.float64)
            m = safe_metrics(y, p)
            rows.append({
                "model_family": family,
                "method": method,
                "rows": len(subset),
                **m,
            })

    # Add paired H1-B1 deltas.
    index = {(r["model_family"], r["method"]): r for r in rows}
    for family in FAMILIES:
        h1 = index[(family, "H1_MRZ19_MAX_ENVELOPE")]
        b1 = index[(family, "B1_MRZ19_POOLED")]
        h1["delta_auroc_vs_b1"] = float(h1["auroc"] - b1["auroc"])
        h1["delta_auprc_vs_b1"] = float(h1["auprc"] - b1["auprc"])
        b1["delta_auroc_vs_b1"] = 0.0
        b1["delta_auprc_vs_b1"] = 0.0
        b0 = index[(family, "B0_RAW19_POOLED")]
        b0["delta_auroc_vs_b1"] = float(b0["auroc"] - b1["auroc"])
        b0["delta_auprc_vs_b1"] = float(b0["auprc"] - b1["auprc"])

    return rows


def summarize_global(predictions):
    y = np.asarray([int(r["true_harmful"]) for r in predictions], dtype=np.int64)
    field_map = {
        "B0_RAW19_POOLED": "p_b0_raw19_pooled",
        "B1_MRZ19_POOLED": "p_b1_mrz19_pooled",
        "H1_MRZ19_MAX_ENVELOPE": "p_h1_mrz19_max_envelope",
    }
    out = []
    for method, field in field_map.items():
        p = np.asarray([float(r[field]) for r in predictions], dtype=np.float64)
        out.append({
            "method": method,
            "rows": len(predictions),
            **safe_metrics(y, p),
        })
    return out


def evaluate_gate(architecture_rows):
    idx = {(r["model_family"], r["method"]): r for r in architecture_rows}

    h1_auc = np.asarray([
        float(idx[(f, "H1_MRZ19_MAX_ENVELOPE")]["auroc"])
        for f in FAMILIES
    ])
    delta = np.asarray([
        float(idx[(f, "H1_MRZ19_MAX_ENVELOPE")]["auroc"])
        - float(idx[(f, "B1_MRZ19_POOLED")]["auroc"])
        for f in FAMILIES
    ])

    actuals = {
        "architecture_h1_aurocs": {
            f: float(idx[(f, "H1_MRZ19_MAX_ENVELOPE")]["auroc"])
            for f in FAMILIES
        },
        "architecture_b1_aurocs": {
            f: float(idx[(f, "B1_MRZ19_POOLED")]["auroc"])
            for f in FAMILIES
        },
        "architecture_h1_minus_b1_aurocs": {
            f: (
                float(idx[(f, "H1_MRZ19_MAX_ENVELOPE")]["auroc"])
                - float(idx[(f, "B1_MRZ19_POOLED")]["auroc"])
            )
            for f in FAMILIES
        },
        "median_architecture_h1_auroc": float(np.median(h1_auc)),
        "minimum_architecture_h1_auroc": float(np.min(h1_auc)),
        "median_architecture_h1_minus_b1_auroc": float(np.median(delta)),
        "minimum_architecture_h1_minus_b1_auroc": float(np.min(delta)),
    }

    checks = {
        "A_median_h1_auroc_ge_0p76":
            actuals["median_architecture_h1_auroc"] >= 0.76,
        "B_min_h1_auroc_ge_0p72":
            actuals["minimum_architecture_h1_auroc"] >= 0.72,
        "C_median_h1_minus_b1_ge_0p01":
            actuals["median_architecture_h1_minus_b1_auroc"] >= 0.01,
        "D_min_h1_minus_b1_ge_minus_0p01":
            actuals["minimum_architecture_h1_minus_b1_auroc"] >= -0.01,
    }

    decision = DECISION_GO if all(checks.values()) else DECISION_STOP

    return {
        "criteria": GO_CRITERIA,
        "actuals": actuals,
        "checks": checks,
        "decision": decision,
    }


def preflight():
    rows, _, _ = validate_and_load()
    arrays = rows_to_arrays(rows)

    Z, audit = build_fold_safe_mrz(arrays, 0)

    held_family = "PraNet"
    held_fold = 0
    train_pool = (
        (arrays["family"] != held_family)
        & (arrays["fold"] != held_fold)
    )
    test = (
        (arrays["family"] == held_family)
        & (arrays["fold"] == held_fold)
    )

    if set(arrays["sample"][train_pool]) & set(arrays["sample"][test]):
        raise RuntimeError("Preflight source-group leakage.")
    if held_family in set(arrays["family"][train_pool]):
        raise RuntimeError("Preflight architecture leakage.")

    for expert_family in ("DeepLabV3-R50", "SegFormer-B0"):
        mask = (
            (arrays["family"] == expert_family)
            & (arrays["fold"] != held_fold)
        )
        if len(np.unique(arrays["harm"][mask])) != 2:
            raise RuntimeError(f"Preflight expert labels degenerate: {expert_family}")

    print("===== Q1-R06A PREFLIGHT =====")
    print(f"protocol_sha256={EXPECTED_PROTOCOL_SHA256}")
    print(f"r03_lock_sha256={EXPECTED_R03_LOCK_SHA256}")
    print(f"r03_table_sha256={EXPECTED_R03_TABLE_SHA256}")
    print(f"rows={len(rows)}")
    print(f"families={len(set(arrays['family']))}")
    print(f"states={len(set(arrays['state']))}")
    print(f"source_groups={len(set(arrays['sample']))}")
    print(f"features={arrays['X_raw'].shape[1]}")
    print("methods=B0_RAW19_POOLED,B1_MRZ19_POOLED,H1_MRZ19_MAX_ENVELOPE")
    print("joint_splits=15")
    print(f"example_train_rows={int(train_pool.sum())}")
    print(f"example_test_rows={int(test.sum())}")
    print(f"fold0_reference_audit_rows={len(audit)}")
    print("BENEFIT_model=NO")
    print("continuous_DeltaDice_model=NO")
    print("probability_threshold_selection=NO")
    print("target_data_read_by_script=NO")
    print("PREFLIGHT_PASS")


def run(args):
    rows, r03, r04 = validate_and_load()
    arrays = rows_to_arrays(rows)

    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    build_dir = Path(str(args.output_dir) + "__building")
    if build_dir.exists():
        raise FileExistsError(build_dir)
    build_dir.mkdir(parents=True, exist_ok=False)

    protocol_copy = build_dir / "preregistered_protocol_copy.md"
    shutil.copy2(PROTOCOL, protocol_copy)
    validate_sha(protocol_copy, EXPECTED_PROTOCOL_SHA256, "protocol copy")

    upstream_path = build_dir / "upstream_audit.json"
    write_json(
        upstream_path,
        {
            "script_version": VERSION,
            "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
            "r03_lock_sha256": EXPECTED_R03_LOCK_SHA256,
            "r03_table_sha256": EXPECTED_R03_TABLE_SHA256,
            "r03_decision": r03.get("decision"),
            "r04_lock_sha256": EXPECTED_R04_LOCK_SHA256,
            "r04_decision": r04.get("decision"),
            "rows": EXPECTED_ROWS,
            "families": list(FAMILIES),
            "model_states": EXPECTED_STATES,
            "source_groups": EXPECTED_SOURCE_GROUPS,
            "perturbations": EXPECTED_PERTURBATIONS,
            "target_data_read_by_script": False,
            "target_pixels_read": False,
            "target_masks_read": False,
            "target_probabilities_read": False,
            "target_utility_labels_read": False,
            "neopolyp_motivated_harm_only_formulation": True,
            "neopolyp_used_for_fit_or_selection": False,
        },
    )

    (
        predictions,
        split_metrics,
        split_defs,
        expert_diag,
        reference_rows,
    ) = evaluate(arrays)

    architecture_rows = summarize_architecture(predictions)
    global_rows = summarize_global(predictions)
    gate = evaluate_gate(architecture_rows)
    decision = gate["decision"]

    write_csv(
        build_dir / "fold_safe_mrz_reference_statistics.csv",
        reference_rows,
        [
            "held_out_group_fold",
            "model_family",
            "model_state_id",
            "training_seed",
            "reference_count",
            "feature",
            "mean",
            "std",
            "effective_std",
            "std_floored",
        ],
    )
    write_csv(
        build_dir / "split_definitions.csv",
        split_defs,
        [
            "held_out_architecture",
            "held_out_group_fold",
            "train_rows",
            "test_rows",
            "train_unique_groups",
            "test_unique_groups",
            "training_families",
            "architecture_leakage",
            "group_leakage",
        ],
    )
    write_csv(
        build_dir / "oof_harm_predictions.csv",
        predictions,
        [
            "model_family",
            "model_state_id",
            "training_seed",
            "checkpoint_sha256",
            "sample_id",
            "source_group_id",
            "fold",
            "perturbation",
            "true_delta_dice",
            "true_harmful",
            "held_out_architecture",
            "held_out_group_fold",
            "p_b0_raw19_pooled",
            "p_b1_mrz19_pooled",
            "expert_1_family",
            "expert_1_harm_probability",
            "expert_2_family",
            "expert_2_harm_probability",
            "p_h1_mrz19_max_envelope",
        ],
    )
    write_csv(
        build_dir / "split_metrics.csv",
        split_metrics,
        [
            "held_out_architecture",
            "held_out_group_fold",
            "method",
            "train_rows",
            "test_rows",
            "auroc",
            "auprc",
            "positive",
            "negative",
        ],
    )
    write_csv(
        build_dir / "architecture_metrics.csv",
        architecture_rows,
        [
            "model_family",
            "method",
            "rows",
            "auroc",
            "auprc",
            "positive",
            "negative",
            "delta_auroc_vs_b1",
            "delta_auprc_vs_b1",
        ],
    )
    write_csv(
        build_dir / "global_metrics.csv",
        global_rows,
        [
            "method",
            "rows",
            "auroc",
            "auprc",
            "positive",
            "negative",
        ],
    )
    write_csv(
        build_dir / "expert_diagnostics.csv",
        expert_diag,
        [
            "held_out_architecture",
            "held_out_group_fold",
            "expert_1_family",
            "expert_2_family",
            "expert_1_training_rows",
            "expert_2_training_rows",
            "expert_1_positive_rows",
            "expert_2_positive_rows",
            "expert_probability_pearson",
            "mean_abs_expert_probability_difference",
            "fraction_expert1_higher",
            "fraction_expert2_higher",
            "fraction_exact_tie",
        ],
    )
    write_json(build_dir / "method_gate.json", gate)
    (build_dir / "decision.txt").write_text(decision + "\n", encoding="utf-8")

    arch_idx = {
        (r["model_family"], r["method"]): r
        for r in architecture_rows
    }
    glob_idx = {r["method"]: r for r in global_rows}

    lines = [
        "===== Q1-R06A SOURCE-ONLY ARCHITECTURE-ROBUST HARM RISK ENVELOPE =====",
        f"Script version: {VERSION}",
        f"Build: {BUILD}",
        "",
        "Source-only protocol:",
        f"  rows={EXPECTED_ROWS}",
        f"  source groups={EXPECTED_SOURCE_GROUPS}",
        f"  model states={EXPECTED_STATES}",
        "  target data read by script=NO",
        "  NeoPolyp used for fit/selection=NO",
        "",
        "Methods:",
        "  B0=RAW19 pooled logistic",
        "  B1=MRZ19 pooled logistic",
        "  H1=MRZ19 two-expert MAX harm-risk envelope",
        "",
        "Architecture-pooled HARM AUROC:",
    ]

    for family in FAMILIES:
        b0 = arch_idx[(family, "B0_RAW19_POOLED")]
        b1 = arch_idx[(family, "B1_MRZ19_POOLED")]
        h1 = arch_idx[(family, "H1_MRZ19_MAX_ENVELOPE")]
        lines.append(
            f"  {family}: "
            f"B0={b0['auroc']:.6f} "
            f"B1={b1['auroc']:.6f} "
            f"H1={h1['auroc']:.6f} "
            f"H1-B1={h1['delta_auroc_vs_b1']:+.6f}"
        )

    lines += [
        "",
        "Global OOF:",
        (
            f"  B0 AUROC={glob_idx['B0_RAW19_POOLED']['auroc']:.6f} "
            f"AUPRC={glob_idx['B0_RAW19_POOLED']['auprc']:.6f}"
        ),
        (
            f"  B1 AUROC={glob_idx['B1_MRZ19_POOLED']['auroc']:.6f} "
            f"AUPRC={glob_idx['B1_MRZ19_POOLED']['auprc']:.6f}"
        ),
        (
            f"  H1 AUROC={glob_idx['H1_MRZ19_MAX_ENVELOPE']['auroc']:.6f} "
            f"AUPRC={glob_idx['H1_MRZ19_MAX_ENVELOPE']['auprc']:.6f}"
        ),
        "",
        "Frozen H1 GO criteria:",
        "  median architecture H1 AUROC >= 0.76",
        "  minimum architecture H1 AUROC >= 0.72",
        "  median architecture H1-B1 AUROC >= +0.01",
        "  minimum architecture H1-B1 AUROC >= -0.01",
        "",
        "Actuals:",
    ]

    for k, v in gate["actuals"].items():
        lines.append(f"  {k}={v}")

    lines += ["", "Checks:"]
    for k, v in gate["checks"].items():
        lines.append(f"  {k}={'PASS' if v else 'FAIL'}")

    lines += [
        "",
        "Decision:",
        f"  {decision}",
        "",
        "Next:",
        (
            "  Q1-R06B source-only recall-anchored harm gate construction"
            if decision == DECISION_GO
            else "  Stop MAX-envelope; preregister a different harm learner if justified."
        ),
    ]

    run_log = build_dir / "run_log.txt"
    run_log.write_text("\n".join(lines) + "\n", encoding="utf-8")

    artifact_names = [
        "preregistered_protocol_copy.md",
        "upstream_audit.json",
        "fold_safe_mrz_reference_statistics.csv",
        "split_definitions.csv",
        "oof_harm_predictions.csv",
        "split_metrics.csv",
        "architecture_metrics.csv",
        "global_metrics.csv",
        "expert_diagnostics.csv",
        "method_gate.json",
        "decision.txt",
        "run_log.txt",
    ]
    artifacts = {
        name: {
            "relative_path": name,
            "sha256": sha256_file(build_dir / name),
        }
        for name in artifact_names
    }

    lock = {
        "script_version": VERSION,
        "build": BUILD,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "r03_lock_sha256": EXPECTED_R03_LOCK_SHA256,
        "r03_table_sha256": EXPECTED_R03_TABLE_SHA256,
        "r04_lock_sha256": EXPECTED_R04_LOCK_SHA256,
        "rows": EXPECTED_ROWS,
        "families": list(FAMILIES),
        "states": EXPECTED_STATES,
        "source_groups": EXPECTED_SOURCE_GROUPS,
        "joint_splits": 15,
        "methods": list(METHODS),
        "task": "HARM",
        "harm_threshold": HARM_THRESHOLD,
        "probability_threshold_selected": False,
        "target_data_read_by_script": False,
        "neopolyp_motivated_harm_only_formulation": True,
        "neopolyp_used_for_fit_or_selection": False,
        "gate": gate,
        "decision": decision,
        "artifacts": artifacts,
    }

    lock_path = build_dir / "Q1_R06A_HARM_RISK_ENVELOPE_LOCK.json"
    write_json(lock_path, lock)
    lock_sha = sha256_file(lock_path)

    for name, meta in artifacts.items():
        if sha256_file(build_dir / meta["relative_path"]) != meta["sha256"]:
            raise RuntimeError(f"Artifact changed before commit: {name}")

    build_dir.rename(args.output_dir)

    print()
    print((args.output_dir / "run_log.txt").read_text(encoding="utf-8"))
    print(
        "Q1-R06A LOCK:",
        args.output_dir / "Q1_R06A_HARM_RISK_ENVELOPE_LOCK.json",
    )
    print("Q1-R06A LOCK SHA256:", lock_sha)


def self_test():
    assert len(SOURCE_FEATURE_NAMES) == 19
    assert EXPECTED_ROWS == 13050
    assert EXPECTED_STATES == 9
    assert EXPECTED_SOURCE_GROUPS == 145
    assert sum(EXPECTED_FOLD_GROUP_COUNTS.values()) == 145
    assert HARM_THRESHOLD == -0.02
    assert METHODS == (
        "B0_RAW19_POOLED",
        "B1_MRZ19_POOLED",
        "H1_MRZ19_MAX_ENVELOPE",
    )
    assert GO_CRITERIA["median_architecture_h1_auroc_min"] == 0.76
    assert GO_CRITERIA["minimum_architecture_h1_auroc_min"] == 0.72
    assert GO_CRITERIA["median_architecture_h1_minus_b1_auroc_min"] == 0.01
    assert GO_CRITERIA["minimum_architecture_h1_minus_b1_auroc_min"] == -0.01

    p1 = np.asarray([0.1, 0.8, 0.3])
    p2 = np.asarray([0.7, 0.2, 0.4])
    assert np.allclose(np.maximum(p1, p2), [0.7, 0.8, 0.4])

    toy_x = np.asarray([
        [0.0, 0.0],
        [0.1, 0.2],
        [1.0, 1.0],
        [1.2, 0.9],
    ])
    toy_y = np.asarray([0, 0, 1, 1])
    model = Pipeline([
        ("scale", StandardScaler()),
        (
            "model",
            LogisticRegression(
                C=1.0,
                class_weight="balanced",
                solver="lbfgs",
                max_iter=5000,
                random_state=RANDOM_STATE,
            ),
        ),
    ])
    model.fit(toy_x, toy_y)
    p = model.predict_proba(toy_x)[:, 1]
    assert np.isfinite(p).all()
    assert safe_metrics(toy_y, p)["auroc"] > 0.9

    print("CARDINALITY_TEST_PASS")
    print("FROZEN_HARM_ONLY_OBJECT_TEST_PASS")
    print("MAX_ENVELOPE_TEST_PASS")
    print("FIXED_LOGISTIC_TEST_PASS")
    print("FROZEN_GO_CRITERIA_TEST_PASS")
    print("SELF_TEST_PASS")


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Q1-R06A: source-only strict LOAO+group evaluation of "
            "RAW19 pooled, MRZ19 pooled, and two-expert MAX HARM risk."
        )
    )
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    p.add_argument("--preflight-only", action="store_true")
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
