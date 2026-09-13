#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SafeTTA R31D0
Utility-First Policy-Level CRC Controller Protocol Lock

Motivation
----------
R31B3 established strong action-transferable safety prediction.
R31C1 then tested a first multi-action controller using a total HARM budget
of 0.02 mechanically divided into three per-action CRC budgets:

    alpha_action = 0.02 / 3 = 0.006666...

That controller was too conservative:
    HARM rate           = 0.001111...
    mean deployed dDice = -0.000290...
    adaptation coverage = 0.048306...

The failure is therefore attributed to THIS R31C controller design, not to
the R31B action-transferable representation.

R31D is a NEW POST-R31C DEVELOPMENTAL CONTROLLER. It is not a correction to
R31C1 and must not be presented as preregistered before R31C1 outcomes.

Core change
-----------
Control the risk of the FINAL DEPLOYED POLICY directly rather than applying a
Bonferroni-like equal budget to every candidate action.

For each model-case:

1. Shared utility and HARM models predict all three actions:
      TENT1, PL-CONF90, MEMO-SEG4-1STEP.

2. Utility-first candidate selection:
      among actions with predicted utility > 0,
      choose the action with largest predicted utility.
      tie -> lower predicted HARM probability -> fixed action order.
      if no action has predicted utility > 0 -> SOURCE.

3. The selected candidate carries its own predicted HARM probability.

4. On the CALIBRATION split only, calibrate ONE policy-level CRC threshold tau
   over these selected candidates:
      corrected risk = n/(n+1)*Rhat + 1/(n+1)
      alpha_policy = 0.020
   Choose the largest tau satisfying corrected empirical policy HARM risk
   <= 0.020.
   SOURCE/no-positive-utility cases are not adaptation candidates and do not
   enter the accepted-action CRC numerator/denominator.

5. TEST deployment:
      if no positive-utility candidate -> SOURCE
      else if candidate p(HARM) <= tau -> commit candidate
      else -> SOURCE rollback.

Why this is materially different from R31C
------------------------------------------
R31C calibrated three independent action thresholds at alpha=0.006667 and
then searched a safe set.

R31D calibrates the risk of the ACTUAL utility-selected deployment policy at
alpha=0.020. There is one final candidate per model-case and one CRC threshold.

This is closer to the scientific question:
    "Can the final adaptive deployment policy keep harmful adaptation below
     the global risk budget while retaining useful adaptation coverage?"

Development/evaluation status
-----------------------------
R31D is designed AFTER observing R31C1. Therefore:
  * its nested NeoPolyp OOF experiment is DEVELOPMENTAL / internal validation;
  * it cannot be called a prospective confirmation;
  * if R31D succeeds, a new untouched external cohort must be locked BEFORE
    any final external claim.

Frozen nested evaluation
------------------------
Reuse the exact R31C0 partitions:
  5 split seeds x 5 outer folds
  480 TRAIN / 160 CALIBRATION / 160 TEST physical cases

TRAIN:
  fit shared Ridge(alpha=1) utility model;
  fit shared balanced LogisticRegression(C=1) HARM model.

CALIBRATION:
  create utility-first candidates;
  calibrate one policy CRC threshold tau at alpha_policy=0.020.

TEST:
  apply frozen candidate rule and frozen tau.

No hyperparameter search.

Primary controller
------------------
R31D_UTILITY_FIRST_POLICY_CRC

Frozen baselines
----------------
SOURCE_ONLY
ALWAYS_TENT1
ALWAYS_PL_CONF90
ALWAYS_MEMO
UTILITY_ONLY
R31C_RISK_UTILITY_CRC     # exact previous controller result as historical baseline
ORACLE_BEST_ACTION

Primary feasibility gate
------------------------
Keep the SAME gate used by R31C0:
  HARM rate <= 0.020
  mean deployed DeltaDice > 0
  adaptation coverage >= 0.05

Additional descriptive comparisons:
  delta coverage vs R31C
  delta mean deployed DeltaDice vs R31C
  delta HARM rate vs R31C
  prevented HARM / retained BENEFIT vs always-action baselines

Information boundary
--------------------
EndoTect: NO
Any external cohort: NO
Controller hyperparameter search: NO

If R31D succeeds:
  next = R31D2_FREEZE_FINAL_POLICY_AND_PLAN_NEW_UNTOUCHED_EXTERNAL_CONFIRMATION
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


VERSION = "2026-09-12-R31D0-v1"

ROOT = Path(r"F:\MEDSEG_SAFETTA")
CODE = ROOT / "code"

R31B3_DIR = ROOT / "R31B3_first_800case_gt_reveal_and_three_action_loao_v1"
R31B3_FINAL = R31B3_DIR / "R31B3_FINAL_LOCK.json"
EXPECTED_R31B3_FINAL_SHA256 = (
    "eb91d991752c50e71b1993e4ee0a50e0b9e9cdb35fffdd58b36553e0cebaaadc"
)
EXPECTED_R31B3_DECISION = "GO_STRONG_ACTION_TRANSFERABLE_SAFETY"

R31C0_DIR = ROOT / "R31C0_multi_action_risk_controller_protocol_lock_v1"
R31C0_FINAL = R31C0_DIR / "R31C0_FINAL_LOCK.json"
EXPECTED_R31C0_FINAL_SHA256 = (
    "e0df9dd61199dbe491651a1fffa23b7d4c954cf290b2f14a9d4d519ad566ac7c"
)
R31C0_PARTITIONS = R31C0_DIR / "R31C0_NESTED_480_160_160_PARTITIONS.csv"

R31C1_DIR = ROOT / "R31C1_nested_multi_action_risk_controller_oof_v1"
R31C1_FINAL = R31C1_DIR / "R31C1_FINAL_LOCK.json"
EXPECTED_R31C1_FINAL_SHA256 = (
    "ef217393ac5d3f09cbd39273cc1442229feb503c9c0cad7d25ea35ca614bd23e"
)
R31C1_SUMMARY = R31C1_DIR / "R31C1_CONTROLLER_SUMMARY.csv"
R31C1_DECISIONS = R31C1_DIR / "R31C1_OOF_CONTROLLER_DECISIONS.csv"

DEFAULT_OUT = ROOT / "R31D0_policy_level_crc_protocol_lock_v1"

ACTIONS = ("TENT1", "PL-CONF90", "MEMO-SEG4-1STEP")
FIXED_ACTION_ORDER = list(ACTIONS)

SPLIT_SEEDS = [20260912, 20260913, 20260914, 20260915, 20260916]
OUTER_FOLDS = 5
TRAIN_CASES = 480
CALIBRATION_CASES = 160
TEST_CASES = 160

FEATURE_CONTRACT = (
    "q_source66 + delta_semantic64 + ActionID + FamilyActionID"
)

UTILITY_MODEL = {
    "scaler": "StandardScaler fit on TRAIN only",
    "regressor": "Ridge",
    "alpha": 1.0,
    "target": "true action DeltaDice",
}

HARM_MODEL = {
    "scaler": "StandardScaler fit on TRAIN only",
    "classifier": "LogisticRegression",
    "C": 1.0,
    "class_weight": "balanced",
    "solver": "lbfgs",
    "target": "1[DeltaDice <= -0.02]",
}

HARM_THRESHOLD = -0.02
BENEFIT_THRESHOLD = +0.02

ALPHA_POLICY = 0.020
CRC_CORRECTION = "n/(n+1)*Rhat + 1/(n+1)"

FEASIBILITY_HARM_MAX = 0.020
FEASIBILITY_MEAN_DELTA_EXCLUSIVE_MIN = 0.0
FEASIBILITY_COVERAGE_MIN = 0.05

PRIMARY_CONTROLLER = "R31D_UTILITY_FIRST_POLICY_CRC"

BASELINES = [
    "SOURCE_ONLY",
    "ALWAYS_TENT1",
    "ALWAYS_PL_CONF90",
    "ALWAYS_MEMO",
    "UTILITY_ONLY",
    "R31C_RISK_UTILITY_CRC",
    "ORACLE_BEST_ACTION",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require_sha(path: Path, expected: str, label: str):
    if not path.is_file():
        raise FileNotFoundError(f"{label}: {path}")
    got = sha256_file(path)
    if got.lower() != expected.lower():
        raise RuntimeError(
            f"{label} SHA mismatch\n"
            f"expected={expected}\nobserved={got}\npath={path}"
        )
    return got


def atomic_json(path: Path, payload: Any):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def verify_upstream():
    require_sha(R31B3_FINAL, EXPECTED_R31B3_FINAL_SHA256, "R31B3 final")
    require_sha(R31C0_FINAL, EXPECTED_R31C0_FINAL_SHA256, "R31C0 final")
    require_sha(R31C1_FINAL, EXPECTED_R31C1_FINAL_SHA256, "R31C1 final")

    b3 = json.loads(R31B3_FINAL.read_text(encoding="utf-8"))
    c0 = json.loads(R31C0_FINAL.read_text(encoding="utf-8"))
    c1 = json.loads(R31C1_FINAL.read_text(encoding="utf-8"))

    if str(b3.get("decision")) != EXPECTED_R31B3_DECISION:
        raise RuntimeError(
            f"R31B3 decision={b3.get('decision')!r}, "
            f"expected={EXPECTED_R31B3_DECISION!r}"
        )

    if c1.get("status") != (
        "PASS_R31C1_NESTED_MULTI_ACTION_RISK_CONTROLLER_OOF_COMPLETE"
    ):
        raise RuntimeError("R31C1 final status changed.")

    if str(c1.get("decision")) != "NO_PASS_CONTROLLER_FEASIBILITY":
        raise RuntimeError(
            "R31D0 is specifically a post-R31C1 redesign protocol; "
            "R31C1 decision is not the expected failed-feasibility result."
        )

    if bool(c1.get("external_data_access", True)):
        raise RuntimeError("R31C1 reports external data access.")

    if not R31C0_PARTITIONS.is_file():
        raise FileNotFoundError(R31C0_PARTITIONS)

    expected_partition_sha = str(c0.get("partition_sha256", ""))
    observed_partition_sha = sha256_file(R31C0_PARTITIONS)
    if observed_partition_sha != expected_partition_sha:
        raise RuntimeError("R31C0 partition SHA changed.")

    if not R31C1_SUMMARY.is_file():
        raise FileNotFoundError(R31C1_SUMMARY)
    if not R31C1_DECISIONS.is_file():
        raise FileNotFoundError(R31C1_DECISIONS)

    return {
        "R31B3": b3,
        "R31C0": c0,
        "R31C1": c1,
        "partition_sha256": observed_partition_sha,
        "R31C1_summary_sha256": sha256_file(R31C1_SUMMARY),
        "R31C1_decisions_sha256": sha256_file(R31C1_DECISIONS),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    print("=" * 124)
    print("SafeTTA R31D0 Utility-First Policy-Level CRC Protocol Lock")
    print("Version                       :", VERSION)
    print("Controller fitting            : NO")
    print("Controller evaluation         : NO")
    print("Post-R31C redesign            : YES")
    print("Prospective-confirmation claim: NO")
    print("External cohort access        : NO")
    print("=" * 124)

    upstream = verify_upstream()

    out = args.output_dir.resolve()
    if out.exists():
        raise RuntimeError(
            f"Output exists; protocol lock cannot overwrite: {out}"
        )
    out.mkdir(parents=True, exist_ok=False)

    protocol = {
        "status": "LOCKED_R31D_POST_R31C_DEVELOPMENT_PROTOCOL",
        "version": VERSION,
        "scientific_status": {
            "designed_after_observing_R31C1": True,
            "internal_nested_OOF_is_developmental": True,
            "prospective_confirmation_claim_allowed": False,
            "new_untouched_external_confirmation_required_if_successful": True,
        },
        "upstream": {
            "R31B3_final_sha256": EXPECTED_R31B3_FINAL_SHA256,
            "R31B3_decision": EXPECTED_R31B3_DECISION,
            "R31C0_final_sha256": EXPECTED_R31C0_FINAL_SHA256,
            "R31C1_final_sha256": EXPECTED_R31C1_FINAL_SHA256,
            "R31C0_partition_sha256": upstream["partition_sha256"],
            "R31C1_summary_sha256": upstream["R31C1_summary_sha256"],
            "R31C1_decisions_sha256": upstream["R31C1_decisions_sha256"],
        },
        "frozen_nested_partitions": {
            "split_seeds": SPLIT_SEEDS,
            "outer_folds_per_seed": OUTER_FOLDS,
            "train_cases": TRAIN_CASES,
            "calibration_cases": CALIBRATION_CASES,
            "test_cases": TEST_CASES,
            "reuse_exact_R31C0_partitions": True,
            "partition_file": str(R31C0_PARTITIONS),
            "partition_sha256": upstream["partition_sha256"],
        },
        "actions": list(ACTIONS),
        "fixed_action_order": FIXED_ACTION_ORDER,
        "feature_contract": FEATURE_CONTRACT,
        "utility_model": UTILITY_MODEL,
        "harm_model": HARM_MODEL,
        "candidate_selection": {
            "stage": "before CRC commit/rollback",
            "eligible_action": "predicted utility > 0",
            "primary": "maximum predicted utility",
            "tie_break_1": "lower predicted HARM probability",
            "tie_break_2": "fixed action order",
            "no_positive_utility_action": "SOURCE",
        },
        "policy_level_crc": {
            "risk_target": (
                "HARM among utility-selected adaptation candidates that "
                "would be committed at p_harm <= tau"
            ),
            "alpha_policy": ALPHA_POLICY,
            "correction": CRC_CORRECTION,
            "calibration_population": (
                "one utility-selected candidate per CALIBRATION model-case "
                "when at least one action has predicted utility > 0"
            ),
            "threshold_rule": (
                "largest tau satisfying corrected empirical policy HARM "
                "risk <= 0.020"
            ),
            "if_no_tau_qualifies": "tau=-inf; all candidates rollback",
            "SOURCE_cases_in_crc_denominator": False,
        },
        "test_policy": {
            "no_positive_utility_candidate": "SOURCE",
            "candidate_p_harm_le_tau": "commit candidate",
            "candidate_p_harm_gt_tau": "SOURCE rollback",
        },
        "primary_controller": PRIMARY_CONTROLLER,
        "baselines": BASELINES,
        "harm_definition": f"DeltaDice <= {HARM_THRESHOLD}",
        "benefit_definition": f"DeltaDice >= {BENEFIT_THRESHOLD}",
        "feasibility_gate_same_as_R31C0": {
            "harm_rate_at_most": FEASIBILITY_HARM_MAX,
            "mean_deployed_delta_dice_strictly_greater_than": (
                FEASIBILITY_MEAN_DELTA_EXCLUSIVE_MIN
            ),
            "adaptation_coverage_at_least": FEASIBILITY_COVERAGE_MIN,
        },
        "required_descriptive_comparisons": [
            "R31D vs R31C mean deployed DeltaDice",
            "R31D vs R31C HARM rate",
            "R31D vs R31C adaptation coverage",
            "R31D vs UTILITY_ONLY",
            "R31D prevented HARM vs always-action baselines",
            "R31D retained BENEFIT vs always-action baselines",
        ],
        "prohibitions": [
            "no hyperparameter search",
            "no alpha_policy tuning",
            "no change to utility positivity rule",
            "no new feature engineering before R31D1 result",
            "no EndoTect use",
            "no external cohort use",
        ],
        "next_if_internal_success": (
            "R31D2_FREEZE_FINAL_POLICY_AND_PLAN_NEW_UNTOUCHED_EXTERNAL_CONFIRMATION"
        ),
        "next_if_internal_failure": (
            "STOP_MULTI_ACTION_CONTROLLER_ENHANCEMENT_AND_RETAIN_R31B_ACTION_TRANSFER_CLAIM"
        ),
    }

    protocol_path = out / "R31D0_POLICY_LEVEL_CRC_PROTOCOL_LOCK.json"
    atomic_json(protocol_path, protocol)

    final = {
        "status": "PASS_R31D0_POLICY_LEVEL_CRC_PROTOCOL_LOCK_COMPLETE",
        "version": VERSION,
        "scientific_status": "POST_R31C_DEVELOPMENTAL",
        "primary_controller": PRIMARY_CONTROLLER,
        "alpha_policy": ALPHA_POLICY,
        "R31B3_final_sha256": EXPECTED_R31B3_FINAL_SHA256,
        "R31C0_final_sha256": EXPECTED_R31C0_FINAL_SHA256,
        "R31C1_final_sha256": EXPECTED_R31C1_FINAL_SHA256,
        "partition_sha256": upstream["partition_sha256"],
        "protocol_sha256": sha256_file(protocol_path),
        "external_data_access": False,
        "next": "R31D1_NESTED_UTILITY_FIRST_POLICY_LEVEL_CRC_OOF",
    }
    final_path = out / "R31D0_FINAL_LOCK.json"
    atomic_json(final_path, final)

    print("\nR31D0 POLICY")
    print("  nested split          : 480 / 160 / 160")
    print("  candidate rule        : max positive predicted utility")
    print("  policy CRC alpha      :", ALPHA_POLICY)
    print("  per-action split      : NO")
    print("  CRC object            : FINAL utility-selected policy candidate")
    print("  commit                : candidate p(HARM) <= policy tau")
    print("  otherwise             : SOURCE rollback")
    print("  feasibility gate      : HARM<=2%, mean dDice>0, coverage>=5%")
    print("  scientific status     : POST-R31C DEVELOPMENTAL")
    print("  new external required : YES, if internal success")
    print("  EndoTect/external     : NO / NO")

    print("\nFINAL STATUS : PASS_R31D0_POLICY_LEVEL_CRC_PROTOCOL_LOCK_COMPLETE")
    print("Protocol SHA256 :", sha256_file(protocol_path))
    print("Final lock SHA256:", sha256_file(final_path))
    print("Output           :", out)
    print("NEXT             : R31D1_NESTED_UTILITY_FIRST_POLICY_LEVEL_CRC_OOF")
    print("=" * 124)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
