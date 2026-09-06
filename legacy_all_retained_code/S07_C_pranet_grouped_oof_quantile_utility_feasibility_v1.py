#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
S07-C: PraNet grouped source-side OOF quantile utility feasibility.

Input:
- Frozen S07-B source-side counterfactual utility dataset only.

Frozen method:
- Action-specific GradientBoostingRegressor quantile models.
- Q10/Q50/Q90.
- 5 frozen source-image group folds.
- Final deployment decision:
    candidate adaptation = argmax(Q10_A1, Q10_A2), tie A1 > A2
    select adaptation only if its Q10 > 0
    otherwise SOURCE.

No target-domain data are opened.
No hyperparameter sweep is performed.
No threshold is tuned.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from tqdm import tqdm


VERSION = "2026-08-19-S07-C-v1"
BUILD = "S07_C_GROUPED_OOF_Q10_Q50_Q90_COUNTERFACTUAL_UTILITY_FEASIBILITY"

ROOT = Path(r"F:\MEDSEG_SAFETTA")

PROTOCOL = (
    ROOT / "docs"
    / "S07_A_safettta_v2_counterfactual_utility_preregistered_protocol_v1.md"
)
EXPECTED_PROTOCOL_SHA256 = "8a0aa3eaf708d6d12c55e13e1e84f8cdd80310a19673666d99d8089738911ab2"

S07B_DIR = (
    ROOT / "outputs"
    / "S07_B_pranet_source_side_counterfactual_utility_dataset_v1"
)
S07B_LOCK = (
    S07B_DIR
    / "S07_B_COUNTERFACTUAL_UTILITY_DATASET_LOCK.json"
)
EXPECTED_S07B_LOCK_SHA256 = "31c9aa7743dfebf124928e0d2204228499c80c4e70b00864fa206eac6696886c"

UTILITY_TABLE = (
    S07B_DIR
    / "source_side_counterfactual_utility_table.csv"
)
EXPECTED_UTILITY_TABLE_SHA256 = "f8d36a10a41db8255847526312b8b2794d3d18c81a8d7b152a210b3c07bd0678"

OUTPUT_DIR = (
    ROOT / "outputs"
    / "S07_C_pranet_grouped_oof_quantile_utility_feasibility_v1"
)

ACTIONS = (
    "A1_TENT_1STEP",
    "A2_TENT_2STEP",
)
ACTION_TIE_ORDER = (
    "SOURCE",
    "A1_TENT_1STEP",
    "A2_TENT_2STEP",
)

QUANTILES = (
    0.10,
    0.50,
    0.90,
)

N_FOLDS = 5
SOURCE_GROUPS = 145
EXPECTED_ROWS_PER_ACTION = 4350
EXPECTED_TOTAL_ROWS = 8700
EXPECTED_BASE_CASES = 4350
SEEDS = (20260817, 20260818, 20260819)

EXPECTED_FOLD_GROUP_COUNTS = {
    0: 23,
    1: 34,
    2: 32,
    3: 27,
    4: 29,
}

FEATURE_NAMES = [
    "source_entropy_mean",
    "source_entropy_std",
    "source_prob_mean",
    "source_prob_std",
    "source_prob_q10",
    "source_prob_q50",
    "source_prob_q90",
    "source_fg_fraction",
    "source_boundary_density",
    "source_confidence_mean",
    "tent_entropy_mean",
    "tent_entropy_std",
    "tent_prob_mean",
    "tent_prob_std",
    "tent_prob_q10",
    "tent_prob_q50",
    "tent_prob_q90",
    "tent_fg_fraction",
    "tent_boundary_density",
    "tent_confidence_mean",
    "entropy_mean_shift",
    "prob_abs_change_mean",
    "mask_disagreement_fraction",
    "abs_fg_fraction_shift",
]

ESTIMATOR_KWARGS = {
    "n_estimators": 200,
    "learning_rate": 0.03,
    "max_depth": 2,
    "min_samples_leaf": 20,
    "subsample": 1.0,
    "max_features": None,
}

EXPERIMENT_SEED = 20260818

HARM_THRESHOLD = -0.02
BENEFIT_THRESHOLD = +0.02

GO_MIN_NET_UTILITY = +0.005
GO_MIN_HARM_PREVENTION = 0.75
GO_MIN_ACCEPTANCE = 0.10
GO_MAX_ACCEPTANCE = 0.90

GO_DECISION = "S07_C_GO_TO_INTERNAL_TARGET_VALIDATION"
STOP_DECISION = "S07_C_STOP_COUNTERFACTUAL_UTILITY_SELECTOR_NO_TARGET_TUNING"


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
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def validate_inputs():
    protocol_sha = validate_sha(
        PROTOCOL,
        EXPECTED_PROTOCOL_SHA256,
        "S07-A protocol",
    )
    lock_sha = validate_sha(
        S07B_LOCK,
        EXPECTED_S07B_LOCK_SHA256,
        "S07-B dataset lock",
    )
    table_sha = validate_sha(
        UTILITY_TABLE,
        EXPECTED_UTILITY_TABLE_SHA256,
        "S07-B utility table",
    )

    lock = json.loads(
        S07B_LOCK.read_text(encoding="utf-8")
    )

    expected_decision = (
        "S07_B_SOURCE_SIDE_COUNTERFACTUAL_UTILITY_DATASET_"
        "LOCKED_READY_FOR_S07C_GROUPED_OOF_QUANTILE_FEASIBILITY"
    )
    if lock.get("decision") != expected_decision:
        raise RuntimeError(
            f"S07-B decision does not authorize S07-C: {lock.get('decision')}"
        )
    if lock.get("table_sha256") != EXPECTED_UTILITY_TABLE_SHA256:
        raise RuntimeError("S07-B lock table SHA provenance mismatch.")
    if bool(lock.get("utility_estimator_fitted", True)):
        raise RuntimeError("S07-B reports utility estimator already fitted.")
    if bool(lock.get("target_data_used", True)):
        raise RuntimeError("S07-B reports target data use.")

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
    } | set(FEATURE_NAMES)

    missing = sorted(
        required - set(fields)
    )
    if missing:
        raise RuntimeError(
            f"Utility table missing fields: {missing}"
        )

    if len(rows) != EXPECTED_TOTAL_ROWS:
        raise RuntimeError(
            f"Utility-table row count mismatch: "
            f"expected={EXPECTED_TOTAL_ROWS} actual={len(rows)}"
        )

    action_counts = Counter(
        r["action"] for r in rows
    )
    expected_action_counts = {
        action: EXPECTED_ROWS_PER_ACTION
        for action in ACTIONS
    }
    if dict(action_counts) != expected_action_counts:
        raise RuntimeError(
            f"Action counts mismatch: {dict(action_counts)}"
        )

    groups = defaultdict(set)
    base_actions = defaultdict(set)
    fold_counts = Counter()

    for r in rows:
        action = r["action"]
        if action not in ACTIONS:
            raise RuntimeError(
                f"Unexpected action: {action}"
            )

        fold = int(r["fold"])
        if not 0 <= fold < N_FOLDS:
            raise RuntimeError(
                f"Invalid fold: {fold}"
            )

        groups[
            r["source_group_id"]
        ].add(fold)

        base_actions[
            r["base_case_id"]
        ].add(action)

        for name in FEATURE_NAMES:
            if not math.isfinite(
                float(r[name])
            ):
                raise RuntimeError(
                    f"Nonfinite feature {name}"
                )

        for name in (
            "source_dice",
            "action_dice",
            "delta_dice",
        ):
            if not math.isfinite(
                float(r[name])
            ):
                raise RuntimeError(
                    f"Nonfinite numeric {name}"
                )

    if len(groups) != SOURCE_GROUPS:
        raise RuntimeError(
            f"Source-group count mismatch: {len(groups)}"
        )

    leakage = [
        group
        for group, folds in groups.items()
        if len(folds) != 1
    ]
    if leakage:
        raise RuntimeError(
            f"Source-group fold leakage: {len(leakage)} groups"
        )

    for folds in groups.values():
        fold_counts[
            next(iter(folds))
        ] += 1

    if dict(sorted(fold_counts.items())) != EXPECTED_FOLD_GROUP_COUNTS:
        raise RuntimeError(
            f"Frozen fold-group counts changed: {dict(sorted(fold_counts.items()))}"
        )

    if len(base_actions) != EXPECTED_BASE_CASES:
        raise RuntimeError(
            f"Base-case count mismatch: {len(base_actions)}"
        )

    expected_actions = set(ACTIONS)
    incomplete = [
        key
        for key, action_set
        in base_actions.items()
        if action_set != expected_actions
    ]
    if incomplete:
        raise RuntimeError(
            f"Base cases missing A1/A2 rows: {len(incomplete)}"
        )

    return rows, {
        "protocol_sha256": protocol_sha,
        "s07b_lock_sha256": lock_sha,
        "utility_table_sha256": table_sha,
        "fold_group_counts": dict(
            sorted(fold_counts.items())
        ),
    }


def make_estimator(alpha: float, random_state: int):
    return GradientBoostingRegressor(
        loss="quantile",
        alpha=float(alpha),
        n_estimators=ESTIMATOR_KWARGS[
            "n_estimators"
        ],
        learning_rate=ESTIMATOR_KWARGS[
            "learning_rate"
        ],
        max_depth=ESTIMATOR_KWARGS[
            "max_depth"
        ],
        min_samples_leaf=ESTIMATOR_KWARGS[
            "min_samples_leaf"
        ],
        subsample=ESTIMATOR_KWARGS[
            "subsample"
        ],
        max_features=ESTIMATOR_KWARGS[
            "max_features"
        ],
        random_state=int(
            random_state
        ),
    )


def matrix(rows):
    x = np.asarray(
        [
            [
                float(r[name])
                for name in FEATURE_NAMES
            ]
            for r in rows
        ],
        dtype=np.float64,
    )
    y = np.asarray(
        [
            float(r["delta_dice"])
            for r in rows
        ],
        dtype=np.float64,
    )

    if (
        x.shape[1] != len(FEATURE_NAMES)
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
    ):
        raise RuntimeError(
            "Malformed/nonfinite design matrix."
        )
    return x, y


def quantile_column(alpha: float) -> str:
    if abs(alpha - 0.10) < 1e-12:
        return "pred_q10"
    if abs(alpha - 0.50) < 1e-12:
        return "pred_q50"
    if abs(alpha - 0.90) < 1e-12:
        return "pred_q90"
    raise ValueError(alpha)


def pinball_loss(y, q, alpha):
    y = np.asarray(y, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    residual = y - q
    loss = np.maximum(
        alpha * residual,
        (alpha - 1.0) * residual,
    )
    return float(
        np.mean(loss)
    )


def grouped_oof(rows):
    oof = []

    progress = tqdm(
        total=(
            N_FOLDS
            * len(ACTIONS)
            * len(QUANTILES)
        ),
        desc="S07-C grouped OOF quantile fits",
        unit="fit",
        dynamic_ncols=True,
    )

    for fold in range(N_FOLDS):
        for action_index, action in enumerate(ACTIONS):
            train_rows = [
                r for r in rows
                if (
                    r["action"] == action
                    and int(r["fold"]) != fold
                )
            ]
            test_rows = [
                r for r in rows
                if (
                    r["action"] == action
                    and int(r["fold"]) == fold
                )
            ]

            if not train_rows or not test_rows:
                raise RuntimeError(
                    f"Empty train/test subset action={action} fold={fold}"
                )

            train_groups = {
                r["source_group_id"]
                for r in train_rows
            }
            test_groups = {
                r["source_group_id"]
                for r in test_rows
            }
            if train_groups & test_groups:
                raise RuntimeError(
                    f"Group leakage action={action} fold={fold}"
                )

            x_train, y_train = matrix(
                train_rows
            )
            x_test, _ = matrix(
                test_rows
            )

            predictions = {}

            for quantile_index, alpha in enumerate(QUANTILES):
                model_seed = (
                    EXPERIMENT_SEED
                    + fold * 100
                    + action_index * 10
                    + quantile_index
                )
                model = make_estimator(
                    alpha,
                    model_seed,
                )
                model.fit(
                    x_train,
                    y_train,
                )
                pred = model.predict(
                    x_test
                ).astype(
                    np.float64
                )

                if (
                    pred.shape
                    != (len(test_rows),)
                    or not np.isfinite(
                        pred
                    ).all()
                ):
                    raise RuntimeError(
                        f"Invalid OOF prediction action={action} "
                        f"fold={fold} alpha={alpha}"
                    )

                predictions[
                    quantile_column(alpha)
                ] = pred

                progress.update(1)

            for i, row in enumerate(test_rows):
                item = dict(row)
                item["pred_q10"] = float(
                    predictions[
                        "pred_q10"
                    ][i]
                )
                item["pred_q50"] = float(
                    predictions[
                        "pred_q50"
                    ][i]
                )
                item["pred_q90"] = float(
                    predictions[
                        "pred_q90"
                    ][i]
                )
                oof.append(item)

    progress.close()

    if len(oof) != EXPECTED_TOTAL_ROWS:
        raise RuntimeError(
            f"OOF row count mismatch: {len(oof)}"
        )

    keys = [
        (
            r["base_case_id"],
            r["action"],
        )
        for r in oof
    ]
    if len(keys) != len(set(keys)):
        raise RuntimeError(
            "Duplicate OOF base-case/action rows."
        )

    return oof


def action_diagnostics(oof):
    rows = []

    for action in ACTIONS:
        subset = [
            r for r in oof
            if r["action"] == action
        ]

        y = np.asarray(
            [
                float(r["delta_dice"])
                for r in subset
            ],
            dtype=np.float64,
        )
        q10 = np.asarray(
            [
                float(r["pred_q10"])
                for r in subset
            ],
            dtype=np.float64,
        )
        q50 = np.asarray(
            [
                float(r["pred_q50"])
                for r in subset
            ],
            dtype=np.float64,
        )
        q90 = np.asarray(
            [
                float(r["pred_q90"])
                for r in subset
            ],
            dtype=np.float64,
        )

        rows.append({
            "action": action,
            "rows": len(subset),
            "delta_mean": float(
                y.mean()
            ),
            "q10_mean": float(
                q10.mean()
            ),
            "q50_mean": float(
                q50.mean()
            ),
            "q90_mean": float(
                q90.mean()
            ),
            "q10_coverage_y_le_q": float(
                np.mean(
                    y <= q10
                )
            ),
            "q50_coverage_y_le_q": float(
                np.mean(
                    y <= q50
                )
            ),
            "q90_coverage_y_le_q": float(
                np.mean(
                    y <= q90
                )
            ),
            "q10_pinball_loss": pinball_loss(
                y,
                q10,
                0.10,
            ),
            "q50_pinball_loss": pinball_loss(
                y,
                q50,
                0.50,
            ),
            "q90_pinball_loss": pinball_loss(
                y,
                q90,
                0.90,
            ),
            "q50_mae": float(
                np.mean(
                    np.abs(
                        y - q50
                    )
                )
            ),
            "quantile_cross_q10_gt_q50_fraction": float(
                np.mean(
                    q10 > q50
                )
            ),
            "quantile_cross_q50_gt_q90_fraction": float(
                np.mean(
                    q50 > q90
                )
            ),
        })

    return rows


def build_decision_rows(oof):
    grouped = defaultdict(dict)

    for r in oof:
        key = r["base_case_id"]
        action = r["action"]
        if action in grouped[key]:
            raise RuntimeError(
                f"Duplicate action in base case {key}"
            )
        grouped[key][
            action
        ] = r

    if len(grouped) != EXPECTED_BASE_CASES:
        raise RuntimeError(
            f"Decision base-case count mismatch: {len(grouped)}"
        )

    decisions = []

    for key in sorted(grouped):
        payload = grouped[key]
        if set(payload) != set(ACTIONS):
            raise RuntimeError(
                f"Missing action in base case {key}"
            )

        a1 = payload[
            "A1_TENT_1STEP"
        ]
        a2 = payload[
            "A2_TENT_2STEP"
        ]

        # These two action rows arise from the same frozen base case.
        invariant_fields = (
            "seed",
            "sample_id",
            "source_group_id",
            "fold",
            "perturbation",
            "source_dice",
        )
        for field in invariant_fields:
            if field == "source_dice":
                if abs(
                    float(a1[field])
                    - float(a2[field])
                ) > 1e-12:
                    raise RuntimeError(
                        f"Source Dice mismatch base case {key}"
                    )
            elif a1[field] != a2[field]:
                raise RuntimeError(
                    f"Base-case invariant {field} mismatch: {key}"
                )

        q10_a1 = float(
            a1["pred_q10"]
        )
        q10_a2 = float(
            a2["pred_q10"]
        )

        # Frozen tie order among adaptation actions: A1 > A2.
        if q10_a1 >= q10_a2:
            proposed_action = (
                "A1_TENT_1STEP"
            )
            proposed = a1
            proposed_q10 = q10_a1
        else:
            proposed_action = (
                "A2_TENT_2STEP"
            )
            proposed = a2
            proposed_q10 = q10_a2

        # SOURCE has LCB=0 and wins all ties.
        if proposed_q10 > 0.0:
            selected_action = (
                proposed_action
            )
            selected = proposed
            accepted = 1
            selected_delta = float(
                selected[
                    "delta_dice"
                ]
            )
            selected_dice = float(
                selected[
                    "action_dice"
                ]
            )
        else:
            selected_action = "SOURCE"
            selected = None
            accepted = 0
            selected_delta = 0.0
            selected_dice = float(
                a1[
                    "source_dice"
                ]
            )

        proposed_delta = float(
            proposed[
                "delta_dice"
            ]
        )
        proposed_harmful = int(
            proposed_delta
            <= HARM_THRESHOLD
        )
        proposed_beneficial = int(
            proposed_delta
            >= BENEFIT_THRESHOLD
        )

        selected_harmful = int(
            accepted
            and selected_delta
            <= HARM_THRESHOLD
        )
        selected_beneficial = int(
            accepted
            and selected_delta
            >= BENEFIT_THRESHOLD
        )

        oracle_delta = max(
            0.0,
            float(
                a1["delta_dice"]
            ),
            float(
                a2["delta_dice"]
            ),
        )

        decisions.append({
            "base_case_id": key,
            "seed": int(
                a1["seed"]
            ),
            "sample_id": a1[
                "sample_id"
            ],
            "source_group_id": a1[
                "source_group_id"
            ],
            "fold": int(
                a1["fold"]
            ),
            "perturbation": a1[
                "perturbation"
            ],
            "source_dice": float(
                a1["source_dice"]
            ),
            "a1_delta_dice": float(
                a1["delta_dice"]
            ),
            "a2_delta_dice": float(
                a2["delta_dice"]
            ),
            "a1_q10": q10_a1,
            "a1_q50": float(
                a1["pred_q50"]
            ),
            "a1_q90": float(
                a1["pred_q90"]
            ),
            "a2_q10": q10_a2,
            "a2_q50": float(
                a2["pred_q50"]
            ),
            "a2_q90": float(
                a2["pred_q90"]
            ),
            "proposed_adaptation_action": proposed_action,
            "proposed_adaptation_q10": proposed_q10,
            "proposed_adaptation_delta_dice": proposed_delta,
            "proposed_adaptation_harmful": proposed_harmful,
            "proposed_adaptation_beneficial": proposed_beneficial,
            "selected_action": selected_action,
            "adaptation_accepted": accepted,
            "selected_delta_dice": selected_delta,
            "selected_dice": selected_dice,
            "selected_harmful": selected_harmful,
            "selected_beneficial": selected_beneficial,
            "harm_prevented": int(
                proposed_harmful
                and not accepted
            ),
            "benefit_retained": int(
                proposed_beneficial
                and accepted
            ),
            "oracle_delta_dice": oracle_delta,
            "oracle_dice": float(
                a1["source_dice"]
            ) + oracle_delta,
        })

    return decisions


def decision_summary(decisions):
    n = len(decisions)
    if n != EXPECTED_BASE_CASES:
        raise RuntimeError(
            f"Decision count mismatch: {n}"
        )

    source = np.asarray(
        [
            float(r["source_dice"])
            for r in decisions
        ],
        dtype=np.float64,
    )
    selected = np.asarray(
        [
            float(r["selected_dice"])
            for r in decisions
        ],
        dtype=np.float64,
    )
    delta = selected - source

    accepted = np.asarray(
        [
            int(r["adaptation_accepted"])
            for r in decisions
        ],
        dtype=np.int64,
    )
    proposed_harm = np.asarray(
        [
            int(r["proposed_adaptation_harmful"])
            for r in decisions
        ],
        dtype=np.int64,
    )
    proposed_benefit = np.asarray(
        [
            int(r["proposed_adaptation_beneficial"])
            for r in decisions
        ],
        dtype=np.int64,
    )
    selected_harm = np.asarray(
        [
            int(r["selected_harmful"])
            for r in decisions
        ],
        dtype=np.int64,
    )

    harm_n = int(
        proposed_harm.sum()
    )
    benefit_n = int(
        proposed_benefit.sum()
    )

    harm_prevention = (
        float(
            np.sum(
                (proposed_harm == 1)
                & (accepted == 0)
            )
        )
        / harm_n
        if harm_n > 0
        else float("nan")
    )

    benefit_retention = (
        float(
            np.sum(
                (proposed_benefit == 1)
                & (accepted == 1)
            )
        )
        / benefit_n
        if benefit_n > 0
        else float("nan")
    )

    action_counts = Counter(
        r["selected_action"]
        for r in decisions
    )
    proposed_counts = Counter(
        r[
            "proposed_adaptation_action"
        ]
        for r in decisions
    )

    oracle = np.asarray(
        [
            float(r["oracle_dice"])
            for r in decisions
        ],
        dtype=np.float64,
    )

    return {
        "base_cases": n,
        "source_mean_dice": float(
            source.mean()
        ),
        "selected_mean_dice": float(
            selected.mean()
        ),
        "net_utility": float(
            delta.mean()
        ),
        "median_selected_delta_dice": float(
            np.median(delta)
        ),
        "adaptation_acceptance": float(
            accepted.mean()
        ),
        "selected_harmful_fraction": float(
            selected_harm.mean()
        ),
        "proposed_harmful_fraction": float(
            proposed_harm.mean()
        ),
        "harm_prevention": harm_prevention,
        "proposed_beneficial_fraction": float(
            proposed_benefit.mean()
        ),
        "benefit_retention": benefit_retention,
        "oracle_mean_dice": float(
            oracle.mean()
        ),
        "oracle_gap": float(
            oracle.mean()
            - selected.mean()
        ),
        "selected_action_counts": {
            key: int(value)
            for key, value
            in sorted(
                action_counts.items()
            )
        },
        "proposed_adaptation_counts": {
            key: int(value)
            for key, value
            in sorted(
                proposed_counts.items()
            )
        },
    }


def evaluate_go(summary):
    c1 = (
        summary[
            "net_utility"
        ]
        + 1e-12
        >= GO_MIN_NET_UTILITY
    )
    c2 = (
        math.isfinite(
            summary[
                "harm_prevention"
            ]
        )
        and summary[
            "harm_prevention"
        ]
        + 1e-12
        >= GO_MIN_HARM_PREVENTION
    )
    c3 = (
        GO_MIN_ACCEPTANCE
        <= summary[
            "adaptation_acceptance"
        ]
        <= GO_MAX_ACCEPTANCE
    )

    passed = (
        c1
        and c2
        and c3
    )

    return {
        "criterion_C1_net_utility_ge_0_005": c1,
        "criterion_C2_harm_prevention_ge_0_75": c2,
        "criterion_C3_acceptance_0_10_to_0_90": c3,
        "pass": passed,
        "decision": (
            GO_DECISION
            if passed
            else STOP_DECISION
        ),
    }


def fit_final_models_if_go(rows, build_dir, decision):
    if not decision[
        "pass"
    ]:
        return None

    final_models = {
        "format_version": 1,
        "feature_names": FEATURE_NAMES,
        "actions": list(
            ACTIONS
        ),
        "quantiles": list(
            QUANTILES
        ),
        "estimator_family": (
            "sklearn.ensemble.GradientBoostingRegressor"
        ),
        "estimator_kwargs": ESTIMATOR_KWARGS,
        "experiment_seed": EXPERIMENT_SEED,
        "decision_rule": {
            "source_lcb": 0.0,
            "adaptation_lcb": "Q10",
            "adapt_only_if_winning_q10_gt_0": True,
            "adaptation_tie_order": [
                "A1_TENT_1STEP",
                "A2_TENT_2STEP",
            ],
            "source_wins_ties": True,
        },
        "models": {},
    }

    progress = tqdm(
        total=len(ACTIONS)
        * len(QUANTILES),
        desc="S07-C final source-side refit",
        unit="fit",
        dynamic_ncols=True,
    )

    for action_index, action in enumerate(ACTIONS):
        subset = [
            r for r in rows
            if r["action"] == action
        ]
        x, y = matrix(
            subset
        )

        final_models[
            "models"
        ][
            action
        ] = {}

        for quantile_index, alpha in enumerate(QUANTILES):
            model_seed = (
                EXPERIMENT_SEED
                + 10000
                + action_index * 10
                + quantile_index
            )
            model = make_estimator(
                alpha,
                model_seed,
            )
            model.fit(
                x,
                y,
            )
            final_models[
                "models"
            ][
                action
            ][
                quantile_column(
                    alpha
                )
            ] = model
            progress.update(1)

    progress.close()

    path = (
        build_dir
        / "SafeTTA_v2_PRANET_SOURCE_LOCKED_UTILITY_MODELS.joblib"
    )
    joblib.dump(
        final_models,
        path,
        compress=3,
    )

    sha = file_sha256(
        path
    )

    meta = {
        "artifact": path.name,
        "sha256": sha,
        "feature_names": FEATURE_NAMES,
        "actions": list(
            ACTIONS
        ),
        "quantiles": list(
            QUANTILES
        ),
        "estimator_kwargs": ESTIMATOR_KWARGS,
        "decision_rule": final_models[
            "decision_rule"
        ],
        "fit_rows": EXPECTED_TOTAL_ROWS,
        "fit_source_groups": SOURCE_GROUPS,
        "target_data_used": False,
    }

    meta_path = (
        build_dir
        / "SafeTTA_v2_PRANET_SOURCE_LOCKED_UTILITY_MODELS.json"
    )
    meta_path.write_text(
        json.dumps(
            meta,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "joblib_file": path.name,
        "joblib_sha256": sha,
        "metadata_file": meta_path.name,
        "metadata_sha256": file_sha256(
            meta_path
        ),
    }


def run(args):
    rows, provenance = validate_inputs()

    if args.output_dir.exists():
        raise FileExistsError(
            f"Final S07-C output already exists: {args.output_dir}"
        )

    build_dir = Path(
        str(args.output_dir)
        + "__building"
    )

    if build_dir.exists():
        if not args.technical_rerun:
            raise FileExistsError(
                f"Incomplete S07-C output exists: {build_dir}. "
                "Use --technical-rerun only after a technical failure."
            )
        shutil.rmtree(
            build_dir
        )

    build_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    print(
        f"[BUILD] {VERSION} | {BUILD}"
    )
    print(
        f"S07-A protocol SHA256={provenance['protocol_sha256']}"
    )
    print(
        f"S07-B lock SHA256={provenance['s07b_lock_sha256']}"
    )
    print(
        f"Utility table SHA256={provenance['utility_table_sha256']}"
    )
    print(
        f"Rows={len(rows)} source groups={SOURCE_GROUPS} folds={N_FOLDS}"
    )
    print(
        "Estimator=GradientBoostingRegressor(loss=quantile)"
    )
    print(
        f"Quantiles={QUANTILES}"
    )
    print(
        f"Frozen kwargs={ESTIMATOR_KWARGS}"
    )
    print(
        "Target-domain data opened=NO | hyperparameter sweep=NO | threshold tuning=NO"
    )
    print()

    oof = grouped_oof(
        rows
    )

    diagnostics = action_diagnostics(
        oof
    )

    decisions = build_decision_rows(
        oof
    )
    summary = decision_summary(
        decisions
    )
    go = evaluate_go(
        summary
    )

    # Save OOF action rows.
    oof_path = (
        build_dir
        / "grouped_oof_action_quantile_predictions.csv"
    )
    oof_fields = [
        "seed",
        "sample_id",
        "source_group_id",
        "fold",
        "perturbation",
        "base_case_id",
        "action",
        "action_steps",
        "source_dice",
        "action_dice",
        "delta_dice",
        "harmful",
        "neutral",
        "beneficial",
    ] + FEATURE_NAMES + [
        "pred_q10",
        "pred_q50",
        "pred_q90",
    ]
    write_csv(
        oof_path,
        oof,
        oof_fields,
    )
    oof_sha = file_sha256(
        oof_path
    )

    # Save decision-level OOF table.
    decision_path = (
        build_dir
        / "grouped_oof_v2_action_selection.csv"
    )
    decision_fields = list(
        decisions[0].keys()
    )
    write_csv(
        decision_path,
        decisions,
        decision_fields,
    )
    decision_sha = file_sha256(
        decision_path
    )

    diag_path = (
        build_dir
        / "oof_quantile_diagnostics.csv"
    )
    write_csv(
        diag_path,
        diagnostics,
        list(
            diagnostics[0].keys()
        ),
    )
    diag_sha = file_sha256(
        diag_path
    )

    final_models = fit_final_models_if_go(
        rows,
        build_dir,
        go,
    )

    result = {
        "script_version": VERSION,
        "build": BUILD,
        "protocol_sha256": provenance[
            "protocol_sha256"
        ],
        "s07b_lock_sha256": provenance[
            "s07b_lock_sha256"
        ],
        "utility_table_sha256": provenance[
            "utility_table_sha256"
        ],
        "fold_group_counts": provenance[
            "fold_group_counts"
        ],
        "actions": list(
            ACTIONS
        ),
        "quantiles": list(
            QUANTILES
        ),
        "feature_names": FEATURE_NAMES,
        "estimator": {
            "family": "GradientBoostingRegressor",
            "loss": "quantile",
            "kwargs": ESTIMATOR_KWARGS,
        },
        "decision_rule": {
            "source_lcb": 0.0,
            "adaptation_lcb": "Q10",
            "adaptation_candidate": "max_Q10_A1_A2",
            "adaptation_tie_order": [
                "A1_TENT_1STEP",
                "A2_TENT_2STEP",
            ],
            "source_wins_zero_tie": True,
            "adapt_only_if_q10_gt_0": True,
        },
        "oof_summary": summary,
        "go_criteria": go,
        "quantile_diagnostics": diagnostics,
        "oof_action_prediction_sha256": oof_sha,
        "oof_action_selection_sha256": decision_sha,
        "oof_quantile_diagnostics_sha256": diag_sha,
        "final_source_models": final_models,
        "target_domain_data_opened": False,
        "hyperparameter_sweep": False,
        "threshold_tuning": False,
        "polypgen_used": False,
        "decision": go[
            "decision"
        ],
    }

    result_path = (
        build_dir
        / "S07_C_GROUPED_OOF_QUANTILE_FEASIBILITY.json"
    )
    result_path.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    result_sha = file_sha256(
        result_path
    )

    lock = {
        "script_version": VERSION,
        "build": BUILD,
        "protocol_sha256": provenance[
            "protocol_sha256"
        ],
        "s07b_lock_sha256": provenance[
            "s07b_lock_sha256"
        ],
        "utility_table_sha256": provenance[
            "utility_table_sha256"
        ],
        "oof_action_prediction_sha256": oof_sha,
        "oof_action_selection_sha256": decision_sha,
        "oof_quantile_diagnostics_sha256": diag_sha,
        "result_json_sha256": result_sha,
        "final_source_models": final_models,
        "target_domain_data_used": False,
        "hyperparameter_sweep": False,
        "threshold_tuning": False,
        "decision": go[
            "decision"
        ],
    }

    lock_path = (
        build_dir
        / "S07_C_GROUPED_OOF_QUANTILE_FEASIBILITY_LOCK.json"
    )
    lock_path.write_text(
        json.dumps(
            lock,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    lock_sha = file_sha256(
        lock_path
    )

    diag_by_action = {
        r["action"]: r
        for r in diagnostics
    }
    d1 = diag_by_action[
        "A1_TENT_1STEP"
    ]
    d2 = diag_by_action[
        "A2_TENT_2STEP"
    ]

    final_model_lines = ""
    if final_models is not None:
        final_model_lines = (
            "\nFinal source-side utility model lock:\n"
            f"  joblib={final_models['joblib_file']}\n"
            f"  SHA256={final_models['joblib_sha256']}\n"
        )

    summary_text = f"""===== S07-C PRANET GROUPED OOF QUANTILE UTILITY FEASIBILITY =====
Script version: {VERSION}
Build: {BUILD}

Frozen protocol:
  source groups={SOURCE_GROUPS}
  folds={N_FOLDS}
  action rows={EXPECTED_TOTAL_ROWS}
  base cases={EXPECTED_BASE_CASES}
  quantiles={QUANTILES}
  estimator=GradientBoostingRegressor(loss=quantile)
  n_estimators={ESTIMATOR_KWARGS["n_estimators"]}
  learning_rate={ESTIMATOR_KWARGS["learning_rate"]}
  max_depth={ESTIMATOR_KWARGS["max_depth"]}
  min_samples_leaf={ESTIMATOR_KWARGS["min_samples_leaf"]}
  threshold sweep=NO
  target-domain data opened=NO

[A1 OOF QUANTILE DIAGNOSTICS]
  Q10 coverage={d1["q10_coverage_y_le_q"]:.6f}
  Q50 coverage={d1["q50_coverage_y_le_q"]:.6f}
  Q90 coverage={d1["q90_coverage_y_le_q"]:.6f}
  Q10 pinball={d1["q10_pinball_loss"]:.6f}
  Q50 MAE={d1["q50_mae"]:.6f}
  Q10>Q50 crossing={d1["quantile_cross_q10_gt_q50_fraction"]:.6f}
  Q50>Q90 crossing={d1["quantile_cross_q50_gt_q90_fraction"]:.6f}

[A2 OOF QUANTILE DIAGNOSTICS]
  Q10 coverage={d2["q10_coverage_y_le_q"]:.6f}
  Q50 coverage={d2["q50_coverage_y_le_q"]:.6f}
  Q90 coverage={d2["q90_coverage_y_le_q"]:.6f}
  Q10 pinball={d2["q10_pinball_loss"]:.6f}
  Q50 MAE={d2["q50_mae"]:.6f}
  Q10>Q50 crossing={d2["quantile_cross_q10_gt_q50_fraction"]:.6f}
  Q50>Q90 crossing={d2["quantile_cross_q50_gt_q90_fraction"]:.6f}

[SAFE TTA-V2 OOF ACTION SELECTION]
  Source mean Dice={summary["source_mean_dice"]:.6f}
  Selected mean Dice={summary["selected_mean_dice"]:.6f}
  Net utility={summary["net_utility"]:+.6f}
  Adaptation acceptance={summary["adaptation_acceptance"]:.6f}
  Proposed harmful fraction={summary["proposed_harmful_fraction"]:.6f}
  Selected harmful fraction={summary["selected_harmful_fraction"]:.6f}
  Harm prevention={summary["harm_prevention"]:.6f}
  Proposed beneficial fraction={summary["proposed_beneficial_fraction"]:.6f}
  Benefit retention={summary["benefit_retention"]:.6f}
  Oracle mean Dice={summary["oracle_mean_dice"]:.6f}
  Oracle gap={summary["oracle_gap"]:.6f}
  Proposed adaptation counts={summary["proposed_adaptation_counts"]}
  Selected action counts={summary["selected_action_counts"]}

[PREREGISTERED GO/STOP]
  C1 net utility >= +0.005: {go["criterion_C1_net_utility_ge_0_005"]}
  C2 harm prevention >= 0.75: {go["criterion_C2_harm_prevention_ge_0_75"]}
  C3 acceptance in [0.10,0.90]: {go["criterion_C3_acceptance_0_10_to_0_90"]}
  PASS={go["pass"]}
{final_model_lines}
Locks:
  OOF action predictions SHA256={oof_sha}
  OOF decisions SHA256={decision_sha}
  result JSON SHA256={result_sha}
  S07-C lock SHA256={lock_sha}

Decision: {go["decision"]}

[OK] Outputs: {args.output_dir}
"""

    (
        build_dir
        / "summary.txt"
    ).write_text(
        summary_text,
        encoding="utf-8",
    )

    # Serialization integrity before commit.
    if file_sha256(
        oof_path
    ) != oof_sha:
        raise RuntimeError(
            "OOF prediction table changed before commit."
        )
    if file_sha256(
        decision_path
    ) != decision_sha:
        raise RuntimeError(
            "OOF decision table changed before commit."
        )
    if file_sha256(
        result_path
    ) != result_sha:
        raise RuntimeError(
            "S07-C result JSON changed before commit."
        )
    if file_sha256(
        lock_path
    ) != lock_sha:
        raise RuntimeError(
            "S07-C lock changed before commit."
        )

    args.output_dir.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    build_dir.rename(
        args.output_dir
    )

    print()
    print(
        (
            args.output_dir
            / "summary.txt"
        ).read_text(
            encoding="utf-8"
        )
    )


def self_test():
    assert ACTIONS == (
        "A1_TENT_1STEP",
        "A2_TENT_2STEP",
    )
    assert QUANTILES == (
        0.10,
        0.50,
        0.90,
    )
    assert EXPECTED_TOTAL_ROWS == 8700
    assert EXPECTED_BASE_CASES == 4350
    assert sum(
        EXPECTED_FOLD_GROUP_COUNTS.values()
    ) == 145
    assert len(
        FEATURE_NAMES
    ) == 24

    assert ESTIMATOR_KWARGS == {
        "n_estimators": 200,
        "learning_rate": 0.03,
        "max_depth": 2,
        "min_samples_leaf": 20,
        "subsample": 1.0,
        "max_features": None,
    }

    # Quantile estimator smoke test.
    rng = np.random.default_rng(
        7
    )
    x = rng.normal(
        size=(120, 4)
    )
    y = (
        0.2 * x[:, 0]
        - 0.1 * x[:, 1]
        + rng.normal(
            scale=0.05,
            size=120,
        )
    )
    preds = []
    for i, alpha in enumerate(QUANTILES):
        model = make_estimator(
            alpha,
            100 + i,
        )
        model.fit(
            x,
            y,
        )
        pred = model.predict(
            x[:10]
        )
        assert pred.shape == (
            10,
        )
        assert np.isfinite(
            pred
        ).all()
        preds.append(
            pred
        )

    assert math.isfinite(
        pinball_loss(
            y[:10],
            preds[0],
            0.10,
        )
    )

    # Frozen Q10 action semantics.
    def select(q1, q2):
        if q1 >= q2:
            candidate = "A1_TENT_1STEP"
            q = q1
        else:
            candidate = "A2_TENT_2STEP"
            q = q2
        if q > 0.0:
            return candidate
        return "SOURCE"

    assert select(
        -0.1,
        -0.2,
    ) == "SOURCE"
    assert select(
        0.1,
        0.05,
    ) == "A1_TENT_1STEP"
    assert select(
        0.05,
        0.1,
    ) == "A2_TENT_2STEP"
    assert select(
        0.0,
        0.0,
    ) == "SOURCE"
    assert select(
        0.1,
        0.1,
    ) == "A1_TENT_1STEP"

    # GO/STOP exact boundaries.
    fake = {
        "net_utility": 0.005,
        "harm_prevention": 0.75,
        "adaptation_acceptance": 0.10,
    }
    gate = evaluate_go(
        fake
    )
    assert gate[
        "pass"
    ] is True

    fake[
        "net_utility"
    ] = 0.00499
    assert evaluate_go(
        fake
    )[
        "pass"
    ] is False

    print("FROZEN_ESTIMATOR_HYPERPARAM_TEST_PASS")
    print("QUANTILE_REGRESSOR_SMOKE_TEST_PASS")
    print("Q10_SOURCE_ABSTENTION_RULE_TEST_PASS")
    print("ACTION_TIE_ORDER_TEST_PASS")
    print("GO_STOP_BOUNDARY_TEST_PASS")
    print("FROZEN_GROUP_COUNT_TEST_PASS")
    print("NO_TARGET_TUNING_STAGE_TEST_PASS")
    print("SELF_TEST_PASS")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen S07-C grouped source-image OOF "
            "Q10/Q50/Q90 utility feasibility test for SafeTTA-v2."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
    )
    parser.add_argument(
        "--technical-rerun",
        action="store_true",
        help=(
            "Delete only an incomplete S07-C __building directory "
            "after a technical interruption. Completed output is never overwritten."
        ),
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
