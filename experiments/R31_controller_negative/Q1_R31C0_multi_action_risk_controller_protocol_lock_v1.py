#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SafeTTA R31C0
Multi-Action Risk-Controlled Controller Protocol Lock

R31B3 has already established STRONG three-action transferability:
  decision = GO_STRONG_ACTION_TRANSFERABLE_SAFETY
  final lock SHA256 =
    eb91d991752c50e71b1993e4ee0a50e0b9e9cdb35fffdd58b36553e0cebaaadc

R31C upgrades SafeTTA from action-harm ranking into an explicit multi-action
transactional controller:

  per action a in {TENT1, PL-CONF90, MEMO-SEG4-1STEP}
      shared harm predictor     -> p(HARM | x, a)
      shared utility predictor  -> predicted DeltaDice(x, a)
      action-wise CRC threshold -> safe / unsafe

  safe action iff:
      predicted utility > 0
      AND p(HARM) <= action CRC tau

  final decision:
      no safe actions -> SOURCE rollback
      otherwise       -> safe action with largest predicted utility
      exact ties       -> lower p(HARM), then fixed action order

This script performs NO controller fitting and NO controller evaluation.
It freezes the R31C method, nested grouped partitions, risk budget, finite-
sample CRC rule, baselines, and evaluation metrics before R31C1 outcomes are
computed.

Nested grouped protocol
-----------------------
800 physical NeoPolyp evaluation cases from R31B3.
For each of 5 split seeds and each of 5 outer folds:
  test        = outer fold                    = 160 physical cases
  calibration = (outer + 1) mod 5            = 160 physical cases
  train       = remaining 3 folds             = 480 physical cases

The folds are assigned using only SHA256(seed|sample_id), not outcome values.
Thus the split itself is outcome-independent.

Frozen models
-------------
Shared utility model:
  StandardScaler(train only) + Ridge(alpha=1)
  target = true action DeltaDice

Shared harm model:
  StandardScaler(train only) +
  LogisticRegression(C=1, class_weight='balanced')
  target = 1[DeltaDice <= -0.02]

Features for both:
  q_source66 + delta_semantic64 + ActionID + FamilyActionID

CRC calibration
---------------
Total deployed HARM budget alpha_total = 0.020.
Three candidate actions => per-action alpha = alpha_total / 3.

For each action separately on the calibration fold:
  * sort candidate harm-probability thresholds tau;
  * accepted calibration rows satisfy p_harm <= tau;
  * Rhat = empirical HARM rate among accepted rows;
  * corrected risk = n/(n+1) * Rhat + 1/(n+1);
  * choose the largest tau whose corrected risk <= alpha_total/3;
  * if none exists, tau = -inf and that action is unavailable.

This preserves the exact finite-sample correction used in R30 while extending
it prospectively from two actions to three actions.

Frozen baselines
----------------
SOURCE_ONLY
ALWAYS_TENT1
ALWAYS_PL_CONF90
ALWAYS_MEMO
UTILITY_ONLY
CRC_RISK_ONLY
R31C_RISK_UTILITY_CRC   <- proposed
ORACLE_BEST_ACTION      <- analysis upper bound only

Primary controller outputs
--------------------------
mean deployed DeltaDice
HARM rate
BENEFIT rate
adaptation coverage
SOURCE/TENT/PL/MEMO action counts
prevented HARM vs each always-adapt baseline
retained BENEFIT where applicable
per-seed and pooled summaries

Controller feasibility gate (development only)
----------------------------------------------
PASS_CONTROLLER_FEASIBILITY requires, averaged over the 5 complete OOF
split-seed panels:
  * R31C HARM rate <= 0.020
  * mean deployed DeltaDice > 0
  * adaptation coverage >= 0.05

This gate is not a clinical significance statement. It prevents the trivial
SOURCE-only solution from being called successful.

Information boundary
--------------------
R31C development may use the already-revealed R31B3 NeoPolyp outcomes.
EndoTect remains completely excluded from R31 development and may NOT be used
for controller tuning or revalidation.
No external cohort may be used to change this frozen controller protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


VERSION = "2026-09-12-R31C0-v1"

ROOT = Path(r"F:\MEDSEG_SAFETTA")
CODE = ROOT / "code"

R31B3_SCRIPT = CODE / "Q1_R31B3_first_800case_gt_reveal_and_three_action_loao_v1.py"
EXPECTED_R31B3_SCRIPT_SHA256 = (
    "5c97224ac7bc1fd7f1e9d3d497a9a30fcbb0fe6050a8024383922d444c1f1479"
)

R31B3_DIR = ROOT / "R31B3_first_800case_gt_reveal_and_three_action_loao_v1"
R31B3_FINAL = R31B3_DIR / "R31B3_FINAL_LOCK.json"
EXPECTED_R31B3_FINAL_SHA256 = (
    "eb91d991752c50e71b1993e4ee0a50e0b9e9cdb35fffdd58b36553e0cebaaadc"
)
R31B3_DECISION = R31B3_DIR / "R31B3_DECISION.json"
R31B3_TABLE = R31B3_DIR / "R31B3_THREE_ACTION_MODELCASE_TABLE.csv"

R31B0_PARTITION = (
    ROOT
    / "R31B0_memo_seg4_third_action_protocol_lock_v1"
    / "R31B0_SOURCE_CASE_PARTITION.csv"
)

DEFAULT_OUT = ROOT / "R31C0_multi_action_risk_controller_protocol_lock_v1"

EXPECTED_DECISION = "GO_STRONG_ACTION_TRANSFERABLE_SAFETY"
EXPECTED_CASES = 800
EXPECTED_STATES = 9
ACTIONS = ("TENT1", "PL-CONF90", "MEMO-SEG4-1STEP")
N_FOLDS = 5
SPLIT_SEEDS = [20260912, 20260913, 20260914, 20260915, 20260916]

HARM_THRESHOLD = -0.02
BENEFIT_THRESHOLD = +0.02

UTILITY_MODEL = {
    "scaler": "StandardScaler fitted on train partition only",
    "estimator": "Ridge(alpha=1)",
    "target": "action DeltaDice",
}
HARM_MODEL = {
    "scaler": "StandardScaler fitted on train partition only",
    "estimator": "LogisticRegression(C=1,class_weight='balanced',solver='lbfgs')",
    "target": "1[DeltaDice <= -0.02]",
}
FEATURE_CONTRACT = (
    "q_source66 + delta_semantic64 + ActionID + FamilyActionID"
)

ALPHA_TOTAL = 0.020
ALPHA_PER_ACTION = ALPHA_TOTAL / len(ACTIONS)
CRC_CORRECTION = "n/(n+1)*Rhat + 1/(n+1)"

BASELINES = (
    "SOURCE_ONLY",
    "ALWAYS_TENT1",
    "ALWAYS_PL_CONF90",
    "ALWAYS_MEMO",
    "UTILITY_ONLY",
    "CRC_RISK_ONLY",
    "R31C_RISK_UTILITY_CRC",
    "ORACLE_BEST_ACTION",
)

FEASIBILITY = {
    "harm_rate_at_most": 0.020,
    "mean_deployed_delta_dice_strictly_greater_than": 0.0,
    "adaptation_coverage_at_least": 0.05,
}

FORBIDDEN_EXTERNAL_TOKENS = (
    "endotect",
    "polypgen",
    "sun-seg",
    "sunseg",
    "promise12",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require_sha(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"{label}: {path}")
    got = sha256_file(path)
    if got.lower() != expected.lower():
        raise RuntimeError(
            f"{label} SHA mismatch\nexpected={expected}\n"
            f"observed={got}\npath={path}"
        )
    return got


def atomic_json(path: Path, payload: Any):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def atomic_csv(df: pd.DataFrame, path: Path):
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def reject_external(path: Path):
    low = str(path).lower()
    if any(tok in low for tok in FORBIDDEN_EXTERNAL_TOKENS):
        raise RuntimeError(f"Forbidden external path in R31C0: {path}")


def verify_r31b3() -> Dict[str, Any]:
    require_sha(R31B3_SCRIPT, EXPECTED_R31B3_SCRIPT_SHA256, "R31B3 script")
    require_sha(R31B3_FINAL, EXPECTED_R31B3_FINAL_SHA256, "R31B3 final lock")

    if not R31B3_DECISION.is_file():
        raise FileNotFoundError(R31B3_DECISION)
    if not R31B3_TABLE.is_file():
        raise FileNotFoundError(R31B3_TABLE)

    final = json.loads(R31B3_FINAL.read_text(encoding="utf-8"))
    decision = json.loads(R31B3_DECISION.read_text(encoding="utf-8"))

    if str(final.get("decision")) != EXPECTED_DECISION:
        raise RuntimeError(
            f"R31B3 final decision={final.get('decision')!r}; "
            f"expected={EXPECTED_DECISION!r}"
        )
    if str(decision.get("decision")) != EXPECTED_DECISION:
        raise RuntimeError("R31B3 decision JSON no longer matches strong GO.")
    if bool(final.get("post_gt_tuning", True)):
        raise RuntimeError("R31B3 reports post-GT tuning.")
    if bool(final.get("external_data_access", True)):
        raise RuntimeError("R31B3 reports external data access.")

    return {"final": final, "decision": decision}


def load_case_ids_without_outcomes() -> List[str]:
    if not R31B0_PARTITION.is_file():
        raise FileNotFoundError(R31B0_PARTITION)

    d = pd.read_csv(
        R31B0_PARTITION,
        usecols=["sample_id", "r31b_partition"],
        dtype={"sample_id": str, "r31b_partition": str},
    )
    d = d[d["r31b_partition"] == "R31B_LOAO_EVALUATION"].copy()
    ids = sorted(d["sample_id"].astype(str).tolist())

    if len(ids) != EXPECTED_CASES or len(set(ids)) != EXPECTED_CASES:
        raise RuntimeError(
            f"R31C0 evaluation case IDs={len(ids)} unique={len(set(ids))}"
        )
    return ids


def fold_map_for_seed(case_ids: List[str], seed: int) -> Dict[str, int]:
    ranked = []
    for sid in case_ids:
        key = hashlib.sha256(
            f"R31C0|{seed}|{sid}".encode("utf-8")
        ).hexdigest()
        ranked.append((key, sid))
    ranked.sort(key=lambda x: (x[0], x[1]))

    out: Dict[str, int] = {}
    for rank, (_, sid) in enumerate(ranked):
        out[sid] = int(rank % N_FOLDS)

    counts = pd.Series(list(out.values())).value_counts().sort_index().to_dict()
    if counts != {0: 160, 1: 160, 2: 160, 3: 160, 4: 160}:
        raise RuntimeError(f"Unexpected fold counts for seed={seed}: {counts}")
    return out


def build_nested_partitions(case_ids: List[str]) -> pd.DataFrame:
    rows = []
    for seed in SPLIT_SEEDS:
        fmap = fold_map_for_seed(case_ids, seed)
        for outer in range(N_FOLDS):
            cal_fold = (outer + 1) % N_FOLDS
            train_folds = sorted(set(range(N_FOLDS)) - {outer, cal_fold})

            for sid in case_ids:
                fold = fmap[sid]
                if fold == outer:
                    role = "TEST"
                elif fold == cal_fold:
                    role = "CALIBRATION"
                else:
                    role = "TRAIN"

                rows.append({
                    "split_seed": int(seed),
                    "outer_fold": int(outer),
                    "sample_id": sid,
                    "hash_fold": int(fold),
                    "role": role,
                    "calibration_fold": int(cal_fold),
                    "train_folds": "+".join(map(str, train_folds)),
                })

    d = pd.DataFrame(rows)

    if len(d) != len(SPLIT_SEEDS) * N_FOLDS * EXPECTED_CASES:
        raise RuntimeError("Nested partition row count drift.")

    audit = (
        d.groupby(["split_seed", "outer_fold", "role"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    for row in audit.itertuples(index=False):
        if int(row.TRAIN) != 480:
            raise RuntimeError("Nested TRAIN count != 480")
        if int(row.CALIBRATION) != 160:
            raise RuntimeError("Nested CALIBRATION count != 160")
        if int(row.TEST) != 160:
            raise RuntimeError("Nested TEST count != 160")

    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    out = args.output_dir.resolve()
    if out.exists():
        raise FileExistsError(
            f"R31C0 is a one-time protocol lock; output exists: {out}"
        )

    print("=" * 124)
    print("SafeTTA R31C0 Multi-Action Risk-Controlled Controller Protocol Lock")
    print("Version                       :", VERSION)
    print("Controller fitting            : NO")
    print("Controller metric evaluation  : NO")
    print("Outcome-dependent split design: NO")
    print("External cohort access        : NO")
    print("Required R31B3 decision       :", EXPECTED_DECISION)
    print("=" * 124)

    upstream = verify_r31b3()
    case_ids = load_case_ids_without_outcomes()
    nested = build_nested_partitions(case_ids)

    out.mkdir(parents=True, exist_ok=False)

    partition_path = out / "R31C0_NESTED_480_160_160_PARTITIONS.csv"
    atomic_csv(nested, partition_path)

    protocol = {
        "status": "LOCKED_BEFORE_R31C_CONTROLLER_OUTCOMES",
        "version": VERSION,
        "upstream": {
            "R31B3_script": str(R31B3_SCRIPT),
            "R31B3_script_sha256": EXPECTED_R31B3_SCRIPT_SHA256,
            "R31B3_final": str(R31B3_FINAL),
            "R31B3_final_sha256": EXPECTED_R31B3_FINAL_SHA256,
            "R31B3_decision": EXPECTED_DECISION,
            "R31B3_decision_json_sha256": sha256_file(R31B3_DECISION),
            "R31B3_three_action_table_sha256": sha256_file(R31B3_TABLE),
        },
        "development_population": {
            "dataset": "NeoPolyp R31B 800-case evaluation partition",
            "physical_cases": EXPECTED_CASES,
            "model_states": EXPECTED_STATES,
            "candidate_actions": list(ACTIONS),
            "external_data_used": False,
            "EndoTect_used": False,
        },
        "nested_grouped_evaluation": {
            "split_seeds": SPLIT_SEEDS,
            "outer_folds_per_seed": N_FOLDS,
            "physical_group": "sample_id",
            "fold_assignment": (
                "outcome-independent rank of SHA256('R31C0|seed|sample_id'), "
                "rank modulo 5"
            ),
            "test": "outer fold = 160 cases",
            "calibration": "(outer+1) mod 5 = 160 cases",
            "train": "remaining 3 folds = 480 cases",
            "partition_csv": str(partition_path),
            "partition_csv_sha256": sha256_file(partition_path),
        },
        "feature_contract": FEATURE_CONTRACT,
        "utility_model": UTILITY_MODEL,
        "harm_model": HARM_MODEL,
        "risk_control": {
            "alpha_total": ALPHA_TOTAL,
            "candidate_action_count": len(ACTIONS),
            "alpha_per_action": ALPHA_PER_ACTION,
            "crc_finite_sample_correction": CRC_CORRECTION,
            "threshold_selection": (
                "per action on CALIBRATION only: choose largest harm-score tau "
                "such that corrected empirical HARM risk among p_harm<=tau "
                "is <= alpha_total/3; if none, tau=-inf"
            ),
        },
        "safe_action_rule": {
            "safe": "utility_pred_delta_dice>0 AND p_harm<=action_crc_tau",
            "none_safe": "SOURCE",
            "one_or_more_safe": "largest predicted utility",
            "tie_break_1": "lower predicted HARM probability",
            "tie_break_2": list(ACTIONS),
        },
        "baselines": list(BASELINES),
        "outcomes": {
            "harm": f"deployed DeltaDice <= {HARM_THRESHOLD}",
            "benefit": f"deployed DeltaDice >= {BENEFIT_THRESHOLD}",
            "source_action_delta_dice": 0.0,
            "primary_metrics": [
                "mean_deployed_delta_dice",
                "harm_rate",
                "benefit_rate",
                "adaptation_coverage",
                "action_counts",
            ],
            "secondary_metrics": [
                "prevented_harm_vs_always_actions",
                "retained_benefit",
                "per_family_metrics",
                "per_seed_metrics",
            ],
        },
        "development_feasibility_gate": FEASIBILITY,
        "interpretation": (
            "PASS_CONTROLLER_FEASIBILITY is an internal method-development "
            "gate, not a clinical significance or external-confirmation claim."
        ),
        "post_protocol_prohibitions": [
            "no change to alpha_total=0.020 based on R31C1 results",
            "no change to alpha_per_action=0.020/3 based on R31C1 results",
            "no controller hyperparameter search",
            "no EndoTect use for R31C development",
            "no external cohort use to choose controller settings",
        ],
        "next": "R31C1_NESTED_MULTI_ACTION_RISK_CONTROLLER_OOF",
    }

    protocol_path = out / "R31C0_CONTROLLER_PROTOCOL_LOCK.json"
    atomic_json(protocol_path, protocol)

    final = {
        "status": "PASS_R31C0_MULTI_ACTION_CONTROLLER_PROTOCOL_LOCK_COMPLETE",
        "version": VERSION,
        "R31B3_final_sha256": EXPECTED_R31B3_FINAL_SHA256,
        "R31B3_decision": EXPECTED_DECISION,
        "physical_cases": EXPECTED_CASES,
        "actions": list(ACTIONS),
        "nested_partitions": len(SPLIT_SEEDS) * N_FOLDS,
        "train_cal_test": [480, 160, 160],
        "alpha_total": ALPHA_TOTAL,
        "alpha_per_action": ALPHA_PER_ACTION,
        "protocol": str(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "partition_sha256": sha256_file(partition_path),
        "controller_fit": False,
        "controller_metrics_evaluated": False,
        "external_data_access": False,
        "next": "R31C1_NESTED_MULTI_ACTION_RISK_CONTROLLER_OOF",
    }

    final_path = out / "R31C0_FINAL_LOCK.json"
    atomic_json(final_path, final)

    print("\nR31C0 NESTED CONTROLLER PROTOCOL")
    print("  split seeds          :", SPLIT_SEEDS)
    print("  outer folds/seed     :", N_FOLDS)
    print("  train/cal/test       : 480 / 160 / 160 physical cases")
    print("  candidate actions    :", list(ACTIONS))
    print("  feature contract     :", FEATURE_CONTRACT)
    print("  utility              : Ridge(alpha=1), shared")
    print("  harm                 : balanced LogisticRegression(C=1), shared")
    print("  total HARM budget    :", ALPHA_TOTAL)
    print("  per-action budget    :", ALPHA_PER_ACTION)
    print("  CRC correction       :", CRC_CORRECTION)
    print("  final selection      : CRC safe-set -> max predicted utility -> SOURCE rollback")
    print("  EndoTect/external    : NO / NO")

    print("\nFINAL STATUS : PASS_R31C0_MULTI_ACTION_CONTROLLER_PROTOCOL_LOCK_COMPLETE")
    print("Protocol SHA256 :", sha256_file(protocol_path))
    print("Final lock SHA256:", sha256_file(final_path))
    print("Output           :", out)
    print("NEXT             : R31C1_NESTED_MULTI_ACTION_RISK_CONTROLLER_OOF")
    print("=" * 124)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
