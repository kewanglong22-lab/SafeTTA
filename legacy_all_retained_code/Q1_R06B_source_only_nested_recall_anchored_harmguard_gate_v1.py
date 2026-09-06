#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Q1-R06B — Source-only nested recall-anchored HarmGuard gate.

Risk estimator:
  B1 = MRZ19 pooled logistic only.

Threshold:
  inner source-only OOF;
  >=90% HARM recall in EACH of the two training architectures;
  tau = min(per-family maximum admissible threshold).

No target data.
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
from typing import Iterable, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


VERSION = "2026-08-19-Q1-R06B-v1"
BUILD = "Q1_R06B_SOURCE_ONLY_NESTED_RECALL_ANCHORED_HARMGUARD_GATE"

ROOT = Path(r"F:\MEDSEG_SAFETTA")

PROTOCOL = (
    ROOT / "docs"
    / "Q1_R06B_source_only_nested_recall_anchored_harmguard_gate_preregistered_protocol_v1.md"
)
EXPECTED_PROTOCOL_SHA256 = "bae3cfc4eb20b2f90959dbb410aa194fb4badb595307f74fdf2fa49f73aed46a"

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

R06A_DIR = (
    ROOT / "outputs"
    / "Q1_R06A_source_only_architecture_robust_harm_risk_envelope_v1"
)
R06A_LOCK = R06A_DIR / "Q1_R06A_HARM_RISK_ENVELOPE_LOCK.json"
EXPECTED_R06A_LOCK_SHA256 = (
    "c0ba773fb65f379170e8e600c79e5b38cc32cd503f91a320bef17bb7c782e7d4"
)
EXPECTED_R06A_DECISION = "STOP_MAX_HARM_RISK_ENVELOPE_NO_ROBUST_GAIN"
EXPECTED_R06A_SCRIPT_SHA256 = (
    "09340432564f1e6a3fbdb2f112814e577744e38f6f44982af0c172593fdc7170"
)
R06A_SCRIPT = (
    ROOT / "code"
    / "Q1_R06A_source_only_architecture_robust_harm_risk_envelope_v1.py"
)

OUTPUT_DIR = (
    ROOT / "outputs"
    / "Q1_R06B_source_only_nested_recall_anchored_harmguard_gate_v1"
)

FAMILIES = ("PraNet", "DeepLabV3-R50", "SegFormer-B0")
FOLDS = (0, 1, 2, 3, 4)

EXPECTED_FOLD_GROUP_COUNTS = {
    0: 23,
    1: 34,
    2: 32,
    3: 27,
    4: 29,
}

EXPECTED_ROWS = 13050
EXPECTED_ROWS_PER_FAMILY = 4350
EXPECTED_SOURCE_GROUPS = 145
EXPECTED_STATES = 9
EXPECTED_PERTURBATIONS = 10

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
BENEFIT_THRESHOLD = 0.02
INNER_HARM_RECALL_TARGET = 0.90

GO_CRITERIA = {
    "median_architecture_harm_prevention_recall_min": 0.80,
    "minimum_architecture_harm_prevention_recall_min": 0.70,
    "median_architecture_adaptation_coverage_min": 0.30,
    "median_architecture_benefit_retention_min": 0.50,
    "minimum_architecture_gated_minus_source_mean_dice_min": -0.005,
}

DECISION_GO = "GO_HARMGUARD_GATE_FOR_R06C"
DECISION_STOP = "STOP_RECALL_ANCHORED_HARMGUARD_GATE"

EXPECTED_R06A_B1_AUROC = {
    "PraNet": 0.7643881216079906,
    "DeepLabV3-R50": 0.72321783167855,
    "SegFormer-B0": 0.7612361621523523,
}


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


def artifact_path_from_lock(base_dir: Path, lock: dict, key: str) -> Path:
    meta = lock.get("artifacts", {}).get(key)
    if not isinstance(meta, dict):
        raise RuntimeError(f"Lock missing artifact: {key}")
    rel = meta.get("relative_path") or meta.get("filename")
    if not rel:
        raise RuntimeError(f"Locked artifact path missing: {key}")
    path = base_dir / str(rel)
    validate_sha(path, str(meta.get("sha256", "")), f"locked artifact {key}")
    return path


def make_model():
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
    validate_sha(PROTOCOL, EXPECTED_PROTOCOL_SHA256, "R06B protocol")
    validate_sha(R03_LOCK, EXPECTED_R03_LOCK_SHA256, "R03 lock")
    validate_sha(R03_TABLE, EXPECTED_R03_TABLE_SHA256, "R03 table")
    validate_sha(R06A_LOCK, EXPECTED_R06A_LOCK_SHA256, "R06A lock")
    validate_sha(R06A_SCRIPT, EXPECTED_R06A_SCRIPT_SHA256, "R06A script")

    r03 = json.loads(R03_LOCK.read_text(encoding="utf-8"))
    r06a = json.loads(R06A_LOCK.read_text(encoding="utf-8"))

    if r03.get("decision") != EXPECTED_R03_DECISION:
        raise RuntimeError(f"Unexpected R03 decision: {r03.get('decision')}")
    if bool(r03.get("target_data_used", True)):
        raise RuntimeError("R03 reports target data use.")

    if r06a.get("decision") != EXPECTED_R06A_DECISION:
        raise RuntimeError(f"Unexpected R06A decision: {r06a.get('decision')}")
    if bool(r06a.get("target_data_read_by_script", True)):
        raise RuntimeError("R06A reports target data read.")

    arch_metrics_path = artifact_path_from_lock(
        R06A_DIR, r06a, "architecture_metrics.csv"
    )
    arch_rows, arch_fields = read_csv(arch_metrics_path)
    required_arch = {"model_family", "method", "auroc"}
    if not required_arch.issubset(set(arch_fields)):
        raise RuntimeError("R06A architecture metrics schema changed.")

    seen_b1 = {}
    for row in arch_rows:
        if row["method"] == "B1_MRZ19_POOLED":
            seen_b1[row["model_family"]] = float(row["auroc"])

    if set(seen_b1) != set(FAMILIES):
        raise RuntimeError("R06A B1 architecture grid incomplete.")
    for family, expected in EXPECTED_R06A_B1_AUROC.items():
        if abs(seen_b1[family] - expected) > 1e-12:
            raise RuntimeError(
                f"R06A B1 AUROC changed for {family}: "
                f"{seen_b1[family]} != {expected}"
            )

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
        raise RuntimeError(f"Family rows mismatch: {dict(family_counts)}")

    state_counts = Counter(r["model_state_id"] for r in rows)
    if len(state_counts) != EXPECTED_STATES:
        raise RuntimeError("State count changed.")
    if any(v != 1450 for v in state_counts.values()):
        raise RuntimeError("Per-state row count changed.")

    group_folds = defaultdict(set)
    for r in rows:
        group_folds[r["source_group_id"]].add(int(r["fold"]))
    if len(group_folds) != EXPECTED_SOURCE_GROUPS:
        raise RuntimeError("Source-group count changed.")
    if any(len(x) != 1 for x in group_folds.values()):
        raise RuntimeError("Source-group fold leakage.")

    fold_counts = Counter(next(iter(v)) for v in group_folds.values())
    if dict(sorted(fold_counts.items())) != EXPECTED_FOLD_GROUP_COUNTS:
        raise RuntimeError(f"Fold group counts changed: {dict(fold_counts)}")

    perturbations = {r["perturbation"] for r in rows}
    if len(perturbations) != EXPECTED_PERTURBATIONS:
        raise RuntimeError("Perturbation count changed.")
    if "identity" not in perturbations:
        raise RuntimeError("Identity perturbation missing.")

    for i, row in enumerate(rows):
        delta = float(row["delta_dice"])
        if int(row["harmful"]) != int(delta <= HARM_THRESHOLD):
            raise RuntimeError(f"HARM label mismatch row={i}")
        if int(row["beneficial"]) != int(delta >= BENEFIT_THRESHOLD):
            raise RuntimeError(f"BENEFIT label mismatch row={i}")
        if row["action"] != "A1_TENT_1STEP":
            raise RuntimeError(f"Unexpected action row={i}: {row['action']}")

    return rows, r03, r06a


def to_arrays(rows):
    return {
        "X": np.asarray(
            [[float(r[f]) for f in SOURCE_FEATURE_NAMES] for r in rows],
            dtype=np.float64,
        ),
        "family": np.asarray([r["model_family"] for r in rows], dtype=object),
        "state": np.asarray([r["model_state_id"] for r in rows], dtype=object),
        "fold": np.asarray([int(r["fold"]) for r in rows], dtype=np.int64),
        "group": np.asarray([r["source_group_id"] for r in rows], dtype=object),
        "perturb": np.asarray([r["perturbation"] for r in rows], dtype=object),
        "harm": np.asarray([int(r["harmful"]) for r in rows], dtype=np.int64),
        "benefit": np.asarray([int(r["beneficial"]) for r in rows], dtype=np.int64),
        "source_dice": np.asarray([float(r["source_dice"]) for r in rows], dtype=np.float64),
        "action_dice": np.asarray([float(r["action_dice"]) for r in rows], dtype=np.float64),
        "meta": [
            {
                "model_family": r["model_family"],
                "model_state_id": r["model_state_id"],
                "training_seed": int(r["training_seed"]),
                "checkpoint_sha256": r["checkpoint_sha256"],
                "sample_id": r["sample_id"],
                "source_group_id": r["source_group_id"],
                "fold": int(r["fold"]),
                "perturbation": r["perturbation"],
            }
            for r in rows
        ],
    }


def mrz_for_excluded_folds(arr, excluded_folds: Iterable[int]):
    excluded_folds = set(int(x) for x in excluded_folds)
    X = arr["X"]
    Z = np.empty_like(X)

    allowed_groups = EXPECTED_SOURCE_GROUPS - sum(
        EXPECTED_FOLD_GROUP_COUNTS[f] for f in excluded_folds
    )

    for sid in sorted(set(arr["state"].tolist())):
        ref = (
            (arr["state"] == sid)
            & (arr["perturb"] == "identity")
            & np.asarray(
                [int(f) not in excluded_folds for f in arr["fold"]],
                dtype=bool,
            )
        )
        ref_x = X[ref]
        if len(ref_x) != allowed_groups:
            raise RuntimeError(
                f"MRZ reference {sid} excluded={sorted(excluded_folds)} "
                f"rows={len(ref_x)} expected={allowed_groups}"
            )

        mu = ref_x.mean(axis=0)
        std = ref_x.std(axis=0, ddof=0)
        eff = np.maximum(std, REFERENCE_STD_FLOOR)
        state_mask = arr["state"] == sid
        Z[state_mask] = (X[state_mask] - mu) / eff

    if not np.isfinite(Z).all():
        raise RuntimeError(f"Non-finite MRZ excluded={sorted(excluded_folds)}")
    return Z


def family_max_threshold_for_recall(y, p, target_recall):
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(p, dtype=np.float64)

    harm_scores = np.sort(p[y == 1])
    n = len(harm_scores)
    if n == 0:
        raise RuntimeError("Cannot choose HARM recall threshold: no HARM positives.")

    max_missed = int(math.floor((1.0 - target_recall) * n + 1e-12))
    max_missed = max(0, min(max_missed, n - 1))
    tau = float(harm_scores[max_missed])

    recall = float(np.mean(p[y == 1] >= tau))
    if recall + 1e-12 < target_recall:
        raise RuntimeError(
            f"Discrete threshold failed target recall: {recall} < {target_recall}"
        )

    return tau, recall, n, max_missed


def nested_threshold(arr, held_family, held_fold):
    training_families = tuple(f for f in FAMILIES if f != held_family)
    inner_folds = tuple(f for f in FOLDS if f != held_fold)

    inner_records = []

    for inner_val_fold in inner_folds:
        Zinner = mrz_for_excluded_folds(
            arr,
            excluded_folds=(held_fold, inner_val_fold),
        )

        inner_train = (
            np.isin(arr["family"], training_families)
            & (arr["fold"] != held_fold)
            & (arr["fold"] != inner_val_fold)
        )
        inner_val = (
            np.isin(arr["family"], training_families)
            & (arr["fold"] == inner_val_fold)
        )

        train_groups = set(arr["group"][inner_train].tolist())
        val_groups = set(arr["group"][inner_val].tolist())
        if train_groups & val_groups:
            raise RuntimeError("Inner source-group leakage.")
        if held_family in set(arr["family"][inner_train].tolist()):
            raise RuntimeError("Inner held-architecture leakage.")

        y_train = arr["harm"][inner_train]
        if len(np.unique(y_train)) != 2:
            raise RuntimeError("Inner HARM training labels degenerate.")

        model = make_model()
        model.fit(Zinner[inner_train], y_train)
        p = model.predict_proba(Zinner[inner_val])[:, 1]

        for local_i, global_i in enumerate(np.flatnonzero(inner_val)):
            inner_records.append({
                "family": str(arr["family"][global_i]),
                "harm": int(arr["harm"][global_i]),
                "probability": float(p[local_i]),
                "inner_val_fold": int(inner_val_fold),
            })

    expected_inner_rows = (
        2
        * 3
        * (EXPECTED_SOURCE_GROUPS - EXPECTED_FOLD_GROUP_COUNTS[held_fold])
        * EXPECTED_PERTURBATIONS
    )
    if len(inner_records) != expected_inner_rows:
        raise RuntimeError(
            f"Inner OOF rows={len(inner_records)} expected={expected_inner_rows}"
        )

    family_thresholds = {}
    family_recalls = {}
    family_harm_counts = {}

    for family in training_families:
        subset = [r for r in inner_records if r["family"] == family]
        y = np.asarray([r["harm"] for r in subset], dtype=np.int64)
        p = np.asarray([r["probability"] for r in subset], dtype=np.float64)

        tau_f, recall_f, harm_n, max_missed = family_max_threshold_for_recall(
            y, p, INNER_HARM_RECALL_TARGET
        )
        family_thresholds[family] = tau_f
        family_recalls[family] = recall_f
        family_harm_counts[family] = harm_n

    tau = min(family_thresholds.values())

    achieved = {}
    for family in training_families:
        subset = [r for r in inner_records if r["family"] == family]
        y = np.asarray([r["harm"] for r in subset], dtype=np.int64)
        p = np.asarray([r["probability"] for r in subset], dtype=np.float64)
        achieved[family] = float(np.mean(p[y == 1] >= tau))
        if achieved[family] + 1e-12 < INNER_HARM_RECALL_TARGET:
            raise RuntimeError("Final nested tau violates source-family recall constraint.")

    return {
        "tau": float(tau),
        "training_families": training_families,
        "family_thresholds": family_thresholds,
        "family_recalls_at_own_tau": family_recalls,
        "family_recalls_at_final_tau": achieved,
        "family_harm_counts": family_harm_counts,
        "inner_oof_rows": len(inner_records),
    }


def gate_metrics(rows):
    n = len(rows)
    harm_n = sum(int(r["true_harmful"]) for r in rows)
    benefit_n = sum(int(r["true_beneficial"]) for r in rows)
    adapted_n = sum(int(r["decision"] == "A1_TENT_1STEP") for r in rows)

    blocked_harm = sum(
        int(r["true_harmful"]) and int(r["decision"] == "SOURCE")
        for r in rows
    )
    residual_harm = sum(
        int(r["true_harmful"]) and int(r["decision"] == "A1_TENT_1STEP")
        for r in rows
    )
    retained_benefit = sum(
        int(r["true_beneficial"]) and int(r["decision"] == "A1_TENT_1STEP")
        for r in rows
    )

    if harm_n == 0:
        raise RuntimeError("Gate metric set has zero HARM rows.")
    if benefit_n == 0:
        raise RuntimeError("Gate metric set has zero BENEFIT rows.")

    source = np.asarray([float(r["source_dice"]) for r in rows])
    action = np.asarray([float(r["action_dice"]) for r in rows])
    gated = np.asarray([float(r["gated_dice"]) for r in rows])
    delta = gated - source

    return {
        "rows": n,
        "harm_rows": harm_n,
        "benefit_rows": benefit_n,
        "blocked_harm_rows": blocked_harm,
        "residual_harm_rows": residual_harm,
        "retained_benefit_rows": retained_benefit,
        "harm_prevention_recall": float(blocked_harm / harm_n),
        "harm_event_relative_reduction": float(1.0 - residual_harm / harm_n),
        "residual_harm_rate_all_cases": float(residual_harm / n),
        "adaptation_coverage": float(adapted_n / n),
        "benefit_retention": float(retained_benefit / benefit_n),
        "source_mean_dice": float(source.mean()),
        "blind_a1_mean_dice": float(action.mean()),
        "harmguard_mean_dice": float(gated.mean()),
        "harmguard_minus_source_mean_dice": float(gated.mean() - source.mean()),
        "harmguard_minus_blind_a1_mean_dice": float(gated.mean() - action.mean()),
        "gated_delta_median": float(np.median(delta)),
        "gated_delta_q05": float(np.quantile(delta, 0.05)),
        "gated_delta_worst": float(np.min(delta)),
    }


def evaluate(arr):
    outer_decisions = []
    split_metrics = []
    thresholds = []
    split_defs = []

    pbar = tqdm(
        total=len(FAMILIES) * len(FOLDS),
        desc="Q1-R06B nested HarmGuard",
        unit="outer-split",
        dynamic_ncols=True,
    )

    for held_family in FAMILIES:
        train_families = tuple(f for f in FAMILIES if f != held_family)

        for held_fold in FOLDS:
            threshold = nested_threshold(arr, held_family, held_fold)
            tau = threshold["tau"]

            Zouter = mrz_for_excluded_folds(arr, excluded_folds=(held_fold,))

            outer_train = (
                np.isin(arr["family"], train_families)
                & (arr["fold"] != held_fold)
            )
            outer_test = (
                (arr["family"] == held_family)
                & (arr["fold"] == held_fold)
            )

            train_groups = set(arr["group"][outer_train].tolist())
            test_groups = set(arr["group"][outer_test].tolist())
            if train_groups & test_groups:
                raise RuntimeError("Outer source-group leakage.")
            if held_family in set(arr["family"][outer_train].tolist()):
                raise RuntimeError("Outer architecture leakage.")

            model = make_model()
            model.fit(Zouter[outer_train], arr["harm"][outer_train])
            prob = model.predict_proba(Zouter[outer_test])[:, 1]

            local_rows = []
            test_indices = np.flatnonzero(outer_test)

            for local_i, global_i in enumerate(test_indices):
                risk = float(prob[local_i])
                decision = "SOURCE" if risk >= tau else "A1_TENT_1STEP"
                source_dice = float(arr["source_dice"][global_i])
                action_dice = float(arr["action_dice"][global_i])
                gated_dice = source_dice if decision == "SOURCE" else action_dice

                meta = arr["meta"][global_i]
                row = {
                    **meta,
                    "held_out_architecture": held_family,
                    "held_out_group_fold": held_fold,
                    "harm_probability": risk,
                    "nested_threshold": tau,
                    "decision": decision,
                    "true_harmful": int(arr["harm"][global_i]),
                    "true_beneficial": int(arr["benefit"][global_i]),
                    "source_dice": source_dice,
                    "action_dice": action_dice,
                    "gated_dice": gated_dice,
                    "gated_delta_vs_source": float(gated_dice - source_dice),
                }
                local_rows.append(row)
                outer_decisions.append(row)

            m = gate_metrics(local_rows)
            split_metrics.append({
                "held_out_architecture": held_family,
                "held_out_group_fold": held_fold,
                "nested_threshold": tau,
                **m,
            })

            thresholds.append({
                "held_out_architecture": held_family,
                "held_out_group_fold": held_fold,
                "training_family_1": train_families[0],
                "training_family_2": train_families[1],
                "inner_oof_rows": threshold["inner_oof_rows"],
                "target_harm_recall": INNER_HARM_RECALL_TARGET,
                "family_1_max_threshold":
                    threshold["family_thresholds"][train_families[0]],
                "family_2_max_threshold":
                    threshold["family_thresholds"][train_families[1]],
                "final_threshold": tau,
                "family_1_harm_rows":
                    threshold["family_harm_counts"][train_families[0]],
                "family_2_harm_rows":
                    threshold["family_harm_counts"][train_families[1]],
                "family_1_recall_at_final_threshold":
                    threshold["family_recalls_at_final_tau"][train_families[0]],
                "family_2_recall_at_final_threshold":
                    threshold["family_recalls_at_final_tau"][train_families[1]],
            })

            split_defs.append({
                "held_out_architecture": held_family,
                "held_out_group_fold": held_fold,
                "training_families": ";".join(train_families),
                "outer_train_rows": int(outer_train.sum()),
                "outer_test_rows": int(outer_test.sum()),
                "outer_train_groups": len(train_groups),
                "outer_test_groups": len(test_groups),
                "inner_validation_folds": ";".join(
                    str(f) for f in FOLDS if f != held_fold
                ),
                "group_leakage": 0,
                "architecture_leakage": 0,
            })

            pbar.update(1)
            pbar.set_postfix(
                arch=held_family,
                fold=held_fold,
                prevent=f"{m['harm_prevention_recall']:.2f}",
                cov=f"{m['adaptation_coverage']:.2f}",
            )

    pbar.close()

    if len(outer_decisions) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Outer OOF rows={len(outer_decisions)} expected={EXPECTED_ROWS}"
        )

    keys = {
        (
            r["model_state_id"],
            r["source_group_id"],
            r["perturbation"],
        )
        for r in outer_decisions
    }
    if len(keys) != EXPECTED_ROWS:
        raise RuntimeError("Outer OOF model-case key mismatch.")

    return outer_decisions, split_metrics, thresholds, split_defs


def architecture_summary(rows):
    out = []
    for family in FAMILIES:
        subset = [r for r in rows if r["model_family"] == family]
        if len(subset) != EXPECTED_ROWS_PER_FAMILY:
            raise RuntimeError(f"Architecture rows {family}={len(subset)}")
        out.append({
            "model_family": family,
            **gate_metrics(subset),
        })
    return out


def evaluate_gate(arch_rows):
    idx = {r["model_family"]: r for r in arch_rows}

    prevention = np.asarray(
        [float(idx[f]["harm_prevention_recall"]) for f in FAMILIES]
    )
    coverage = np.asarray(
        [float(idx[f]["adaptation_coverage"]) for f in FAMILIES]
    )
    benefit = np.asarray(
        [float(idx[f]["benefit_retention"]) for f in FAMILIES]
    )
    dice_delta = np.asarray(
        [float(idx[f]["harmguard_minus_source_mean_dice"]) for f in FAMILIES]
    )

    actuals = {
        "architecture_harm_prevention_recall": {
            f: float(idx[f]["harm_prevention_recall"]) for f in FAMILIES
        },
        "architecture_adaptation_coverage": {
            f: float(idx[f]["adaptation_coverage"]) for f in FAMILIES
        },
        "architecture_benefit_retention": {
            f: float(idx[f]["benefit_retention"]) for f in FAMILIES
        },
        "architecture_harmguard_minus_source_mean_dice": {
            f: float(idx[f]["harmguard_minus_source_mean_dice"]) for f in FAMILIES
        },
        "median_architecture_harm_prevention_recall": float(np.median(prevention)),
        "minimum_architecture_harm_prevention_recall": float(np.min(prevention)),
        "median_architecture_adaptation_coverage": float(np.median(coverage)),
        "median_architecture_benefit_retention": float(np.median(benefit)),
        "minimum_architecture_harmguard_minus_source_mean_dice":
            float(np.min(dice_delta)),
    }

    checks = {
        "A_median_harm_prevention_ge_0p80":
            actuals["median_architecture_harm_prevention_recall"] >= 0.80,
        "B_min_harm_prevention_ge_0p70":
            actuals["minimum_architecture_harm_prevention_recall"] >= 0.70,
        "C_median_adaptation_coverage_ge_0p30":
            actuals["median_architecture_adaptation_coverage"] >= 0.30,
        "D_median_benefit_retention_ge_0p50":
            actuals["median_architecture_benefit_retention"] >= 0.50,
        "E_min_gate_minus_source_mean_dice_ge_minus_0p005":
            actuals["minimum_architecture_harmguard_minus_source_mean_dice"]
            >= -0.005,
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
    arr = to_arrays(rows)

    test_threshold = nested_threshold(arr, "PraNet", 0)

    print("===== Q1-R06B PREFLIGHT =====")
    print(f"protocol_sha256={EXPECTED_PROTOCOL_SHA256}")
    print(f"r03_lock_sha256={EXPECTED_R03_LOCK_SHA256}")
    print(f"r06a_lock_sha256={EXPECTED_R06A_LOCK_SHA256}")
    print(f"rows={len(rows)}")
    print(f"families={len(set(arr['family']))}")
    print(f"states={len(set(arr['state']))}")
    print(f"source_groups={len(set(arr['group']))}")
    print("risk_estimator=B1_MRZ19_POOLED_LOGISTIC")
    print(f"inner_harm_recall_target={INNER_HARM_RECALL_TARGET}")
    print("threshold_selection=INNER_SOURCE_OOF_ONLY")
    print("outer_joint_splits=15")
    print(
        "example_PraNet_fold0_threshold="
        f"{test_threshold['tau']:.12f}"
    )
    print("H1_envelope=RETIRED")
    print("target_data_read_by_script=NO")
    print("PREFLIGHT_PASS")


def run(args):
    rows, r03, r06a = validate_and_load()
    arr = to_arrays(rows)

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
            "r06a_lock_sha256": EXPECTED_R06A_LOCK_SHA256,
            "r06a_decision": r06a.get("decision"),
            "r06a_script_sha256": EXPECTED_R06A_SCRIPT_SHA256,
            "risk_estimator": "B1_MRZ19_POOLED_LOGISTIC",
            "h1_envelope_retired": True,
            "inner_harm_recall_target": INNER_HARM_RECALL_TARGET,
            "outer_splits": 15,
            "target_data_read_by_script": False,
            "neopolyp_used_in_computation": False,
            "benefit_used_for_predictor_fit": False,
            "benefit_used_for_threshold_selection": False,
            "dice_used_for_threshold_selection": False,
        },
    )

    (
        outer_rows,
        split_rows,
        threshold_rows,
        split_defs,
    ) = evaluate(arr)

    arch_rows = architecture_summary(outer_rows)
    global_metrics = gate_metrics(outer_rows)
    gate = evaluate_gate(arch_rows)
    decision = gate["decision"]

    write_csv(
        build_dir / "outer_split_definitions.csv",
        split_defs,
        [
            "held_out_architecture",
            "held_out_group_fold",
            "training_families",
            "outer_train_rows",
            "outer_test_rows",
            "outer_train_groups",
            "outer_test_groups",
            "inner_validation_folds",
            "group_leakage",
            "architecture_leakage",
        ],
    )

    write_csv(
        build_dir / "nested_thresholds.csv",
        threshold_rows,
        [
            "held_out_architecture",
            "held_out_group_fold",
            "training_family_1",
            "training_family_2",
            "inner_oof_rows",
            "target_harm_recall",
            "family_1_max_threshold",
            "family_2_max_threshold",
            "final_threshold",
            "family_1_harm_rows",
            "family_2_harm_rows",
            "family_1_recall_at_final_threshold",
            "family_2_recall_at_final_threshold",
        ],
    )

    write_csv(
        build_dir / "outer_model_case_decisions.csv",
        outer_rows,
        [
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
            "harm_probability",
            "nested_threshold",
            "decision",
            "true_harmful",
            "true_beneficial",
            "source_dice",
            "action_dice",
            "gated_dice",
            "gated_delta_vs_source",
        ],
    )

    metric_fields = [
        "rows",
        "harm_rows",
        "benefit_rows",
        "blocked_harm_rows",
        "residual_harm_rows",
        "retained_benefit_rows",
        "harm_prevention_recall",
        "harm_event_relative_reduction",
        "residual_harm_rate_all_cases",
        "adaptation_coverage",
        "benefit_retention",
        "source_mean_dice",
        "blind_a1_mean_dice",
        "harmguard_mean_dice",
        "harmguard_minus_source_mean_dice",
        "harmguard_minus_blind_a1_mean_dice",
        "gated_delta_median",
        "gated_delta_q05",
        "gated_delta_worst",
    ]

    write_csv(
        build_dir / "split_gate_metrics.csv",
        split_rows,
        [
            "held_out_architecture",
            "held_out_group_fold",
            "nested_threshold",
            *metric_fields,
        ],
    )

    write_csv(
        build_dir / "architecture_gate_metrics.csv",
        arch_rows,
        ["model_family", *metric_fields],
    )

    write_json(build_dir / "global_gate_metrics.json", global_metrics)
    write_json(build_dir / "method_gate.json", gate)

    (build_dir / "decision.txt").write_text(decision + "\n", encoding="utf-8")

    arch_idx = {r["model_family"]: r for r in arch_rows}

    lines = [
        "===== Q1-R06B SOURCE-ONLY NESTED RECALL-ANCHORED HARMGUARD GATE =====",
        f"Script version: {VERSION}",
        f"Build: {BUILD}",
        "",
        "Frozen method:",
        "  risk estimator=B1 MRZ19 pooled logistic",
        "  H1 MAX envelope=RETIRED",
        f"  inner source HARM recall anchor={INNER_HARM_RECALL_TARGET}",
        "  threshold source=inner OOF only",
        "  high risk -> SOURCE",
        "  low risk -> A1_TENT_1STEP",
        "  target data read by script=NO",
        "",
        "Architecture-pooled safety/utility:",
    ]

    for family in FAMILIES:
        m = arch_idx[family]
        lines += [
            (
                f"  {family}: "
                f"harm_prevention={m['harm_prevention_recall']:.6f} "
                f"coverage={m['adaptation_coverage']:.6f} "
                f"benefit_retention={m['benefit_retention']:.6f}"
            ),
            (
                f"    SourceDice={m['source_mean_dice']:.6f} "
                f"BlindA1Dice={m['blind_a1_mean_dice']:.6f} "
                f"HarmGuardDice={m['harmguard_mean_dice']:.6f} "
                f"Gate-Source={m['harmguard_minus_source_mean_dice']:+.6f} "
                f"Gate-A1={m['harmguard_minus_blind_a1_mean_dice']:+.6f}"
            ),
        ]

    lines += [
        "",
        "Global source OOF:",
        f"  harm_prevention={global_metrics['harm_prevention_recall']:.6f}",
        f"  adaptation_coverage={global_metrics['adaptation_coverage']:.6f}",
        f"  benefit_retention={global_metrics['benefit_retention']:.6f}",
        f"  SourceDice={global_metrics['source_mean_dice']:.6f}",
        f"  BlindA1Dice={global_metrics['blind_a1_mean_dice']:.6f}",
        f"  HarmGuardDice={global_metrics['harmguard_mean_dice']:.6f}",
        f"  Gate-Source={global_metrics['harmguard_minus_source_mean_dice']:+.6f}",
        f"  Gate-A1={global_metrics['harmguard_minus_blind_a1_mean_dice']:+.6f}",
        "",
        "Frozen GO criteria:",
        "  median architecture harm prevention >= 0.80",
        "  minimum architecture harm prevention >= 0.70",
        "  median architecture adaptation coverage >= 0.30",
        "  median architecture benefit retention >= 0.50",
        "  minimum architecture Gate-Source mean Dice >= -0.005",
        "",
        "Actuals:",
    ]

    for key, value in gate["actuals"].items():
        lines.append(f"  {key}={value}")

    lines += ["", "Checks:"]
    for key, passed in gate["checks"].items():
        lines.append(f"  {key}={'PASS' if passed else 'FAIL'}")

    lines += [
        "",
        "Decision:",
        f"  {decision}",
        "",
        "Next:",
        (
            "  Q1-R06C freeze final source-only HarmGuard deployment packages"
            if decision == DECISION_GO
            else "  STOP current recall-anchored HarmGuard gate; no same-experiment rescue."
        ),
    ]

    run_log_path = build_dir / "run_log.txt"
    run_log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    artifact_names = [
        "preregistered_protocol_copy.md",
        "upstream_audit.json",
        "outer_split_definitions.csv",
        "nested_thresholds.csv",
        "outer_model_case_decisions.csv",
        "split_gate_metrics.csv",
        "architecture_gate_metrics.csv",
        "global_gate_metrics.json",
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
        "r06a_lock_sha256": EXPECTED_R06A_LOCK_SHA256,
        "r06a_script_sha256": EXPECTED_R06A_SCRIPT_SHA256,
        "risk_estimator": "B1_MRZ19_POOLED_LOGISTIC",
        "h1_envelope_retired": True,
        "inner_harm_recall_target": INNER_HARM_RECALL_TARGET,
        "outer_joint_splits": 15,
        "target_data_read_by_script": False,
        "neopolyp_used_in_computation": False,
        "benefit_used_for_fit_or_threshold": False,
        "dice_used_for_threshold": False,
        "gate": gate,
        "decision": decision,
        "artifacts": artifacts,
    }

    lock_path = build_dir / "Q1_R06B_HARMGUARD_GATE_LOCK.json"
    write_json(lock_path, lock)
    lock_sha = sha256_file(lock_path)

    for name, meta in artifacts.items():
        if sha256_file(build_dir / meta["relative_path"]) != meta["sha256"]:
            raise RuntimeError(f"Artifact changed before commit: {name}")

    build_dir.rename(args.output_dir)

    print()
    print((args.output_dir / "run_log.txt").read_text(encoding="utf-8"))
    print(
        "Q1-R06B LOCK:",
        args.output_dir / "Q1_R06B_HARMGUARD_GATE_LOCK.json",
    )
    print("Q1-R06B LOCK SHA256:", lock_sha)


def self_test():
    assert len(SOURCE_FEATURE_NAMES) == 19
    assert EXPECTED_ROWS == 13050
    assert EXPECTED_STATES == 9
    assert EXPECTED_SOURCE_GROUPS == 145
    assert sum(EXPECTED_FOLD_GROUP_COUNTS.values()) == 145
    assert INNER_HARM_RECALL_TARGET == 0.90
    assert HARM_THRESHOLD == -0.02
    assert BENEFIT_THRESHOLD == 0.02

    y = np.asarray([1] * 10 + [0] * 5)
    p = np.asarray(
        [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
        + [0.05, 0.15, 0.25, 0.35, 0.45]
    )
    tau, recall, n, missed = family_max_threshold_for_recall(y, p, 0.90)
    assert n == 10
    assert missed == 1
    assert abs(tau - 0.20) < 1e-12
    assert abs(recall - 0.90) < 1e-12

    toy_rows = [
        {
            "true_harmful": 1,
            "true_beneficial": 0,
            "decision": "SOURCE",
            "source_dice": 0.8,
            "action_dice": 0.6,
            "gated_dice": 0.8,
        },
        {
            "true_harmful": 0,
            "true_beneficial": 1,
            "decision": "A1_TENT_1STEP",
            "source_dice": 0.7,
            "action_dice": 0.8,
            "gated_dice": 0.8,
        },
    ]
    gm = gate_metrics(toy_rows)
    assert gm["harm_prevention_recall"] == 1.0
    assert gm["adaptation_coverage"] == 0.5
    assert gm["benefit_retention"] == 1.0
    assert abs(gm["harmguard_mean_dice"] - 0.8) < 1e-12

    assert GO_CRITERIA[
        "median_architecture_harm_prevention_recall_min"
    ] == 0.80
    assert GO_CRITERIA[
        "minimum_architecture_harm_prevention_recall_min"
    ] == 0.70
    assert GO_CRITERIA[
        "median_architecture_adaptation_coverage_min"
    ] == 0.30
    assert GO_CRITERIA[
        "median_architecture_benefit_retention_min"
    ] == 0.50
    assert GO_CRITERIA[
        "minimum_architecture_gated_minus_source_mean_dice_min"
    ] == -0.005

    print("CARDINALITY_TEST_PASS")
    print("DISCRETE_90PCT_HARM_RECALL_THRESHOLD_TEST_PASS")
    print("GATE_COUNTERFACTUAL_METRIC_TEST_PASS")
    print("FROZEN_GO_CRITERIA_TEST_PASS")
    print("SELF_TEST_PASS")


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Q1-R06B: strict nested source-only evaluation of a 90%-HARM-recall "
            "anchored gate using B1 MRZ19 pooled logistic."
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
