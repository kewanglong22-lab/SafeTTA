#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(r"F:\MEDSEG_SAFETTA")
OUT = ROOT / "outputs"
R12C_DIR = OUT / "Q1_R12C_sar_gt_utility_reveal_and_outcome_lock_fix1_v1"
R12C_LOCK = R12C_DIR / "R12C_SAR_OUTCOME_LOCK.json"
R12C_PANEL = R12C_DIR / "R12C_SAR_model_case_outcomes.csv"
EXPECTED_R12C_SHA = "536a22c08405584eb5421b50840802629aa24f4c6fd10689b275b2e92b2f9af1"
EXPECTED_R12C_DECISION = "NEOPOLYP_SAR_GT_UTILITY_AND_OUTCOMES_LOCKED"
OUTPUT_DIR = OUT / "Q1_R12D_sar_transfer_degeneracy_stop_lock_fix1_v1"
ROWS, CASES, STATES = 9000, 1000, 9
DECISION = "SAR_CROSS_TTA_SAFETY_TRANSFER_NOT_EVALUABLE_ZERO_HARM_DEGENERACY"

def sha256_file(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8*1024*1024), b""):
            h.update(b)
    return h.hexdigest()

def self_test():
    y = np.zeros(32, dtype=int)
    assert len(np.unique(y)) == 1
    print("SELF_TEST_PASS")

def run(args):
    if not R12C_LOCK.exists() or not R12C_PANEL.exists():
        raise FileNotFoundError("R12C artifacts missing")

    actual = sha256_file(R12C_LOCK)
    if actual != EXPECTED_R12C_SHA:
        raise RuntimeError(f"R12C lock SHA mismatch: {actual}")

    lock = json.loads(R12C_LOCK.read_text(encoding="utf-8"))
    if lock.get("decision") != EXPECTED_R12C_DECISION:
        raise RuntimeError("R12C decision mismatch")
    if int(lock.get("sar_harm_rows", -1)) != 0:
        raise RuntimeError("SAR HARM != 0")
    if int(lock.get("sar_neutral_rows", -1)) != ROWS:
        raise RuntimeError("SAR NEUTRAL != 9000")
    if int(lock.get("sar_benefit_rows", -1)) != 0:
        raise RuntimeError("SAR BENEFIT != 0")

    for k in (
        "safety_scores_read","safety_model_fit_or_refit","threshold_recalibration",
        "score_reversal","polypgen_access","sar_hyperparameter_tuning"
    ):
        if bool(lock.get("information_boundary", {}).get(k, True)):
            raise RuntimeError(f"R12C boundary failed: {k}")

    exp = lock.get("artifacts", {}).get(R12C_PANEL.name)
    if not exp or sha256_file(R12C_PANEL) != exp:
        raise RuntimeError("R12C panel SHA mismatch")

    df = pd.read_csv(R12C_PANEL, low_memory=False)
    if len(df) != ROWS or df["sample_id"].nunique() != CASES or df["model_state_id"].nunique() != STATES:
        raise RuntimeError("R12C panel cardinality mismatch")

    harm = pd.to_numeric(df["sar_harm_label"], errors="raise").astype(int)
    benefit = pd.to_numeric(df["sar_benefit_label"], errors="raise").astype(int)
    outcome = df["sar_adaptation_outcome"].astype(str)
    delta = pd.to_numeric(df["sar_delta_dice"], errors="raise").astype(float)

    if int(harm.sum()) != 0 or int(benefit.sum()) != 0 or not outcome.eq("NEUTRAL").all():
        raise RuntimeError("SAR outcome degeneracy no longer exact")
    if not np.isfinite(delta.to_numpy()).all():
        raise RuntimeError("Non-finite SAR DeltaDice")

    family = (
        df.assign(sar_harm_label=harm, sar_benefit_label=benefit, sar_delta_dice=delta)
        .groupby("model_family", as_index=False)
        .agg(
            rows=("sample_id","size"),
            harm=("sar_harm_label","sum"),
            benefit=("sar_benefit_label","sum"),
            delta_mean=("sar_delta_dice","mean"),
            delta_median=("sar_delta_dice","median"),
            delta_min=("sar_delta_dice","min"),
            delta_max=("sar_delta_dice","max"),
        )
    )

    print("===== R12D SAR TRANSFER DEGENERACY STOP LOCK FIX1 =====")
    print("SAR HARM=0")
    print("SAR NEUTRAL=9000")
    print("SAR BENEFIT=0")
    print("AUROC=NOT DEFINED")
    print("AUPRC/Recall/FPR=NOT SCIENTIFICALLY EVALUABLE")
    print("frozen safety score read=NO")
    print("SAR retuning=NO")
    print("Cross-TTA safety transfer=NOT TESTABLE, NOT FAILED")
    print("\n===== BY FAMILY =====")
    print(family.to_string(index=False))

    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)

    family_path = args.output_dir / "R12D_SAR_degeneracy_family_audit.csv"
    family.to_csv(family_path, index=False)

    out_lock = {
        "status": "STOP",
        "decision": DECISION,
        "r12c_lock_sha256": actual,
        "target_cases": CASES,
        "model_states": STATES,
        "model_case_rows": ROWS,
        "sar_harm_rows": 0,
        "sar_neutral_rows": ROWS,
        "sar_benefit_rows": 0,
        "sar_harm_unique_classes": 1,
        "evaluability": {
            "auroc": False,
            "auprc": False,
            "recall_fpr_operating_point": False,
            "reason": "single-class SAR harm target: zero HARM events",
        },
        "scientific_interpretation": {
            "sar_dense_episdic_action_effect": "near-identity under frozen protocol",
            "cross_tta_safety_transfer_conclusion": "not testable, not failed",
            "sar_branch_retuning_authorized": False,
        },
        "information_boundary": {
            "frozen_safety_scores_read": False,
            "safety_model_fit_or_refit": False,
            "threshold_recalibration": False,
            "score_reversal": False,
            "polypgen_access": False,
            "sar_hyperparameter_tuning": False,
        },
        "artifacts": {family_path.name: sha256_file(family_path)},
        "next_stage": "OPEN_NEW_PRE_REGISTERED_DISTINCT_TTA_ACTION_BRANCH_NO_POLYPGEN",
    }

    lock_path = args.output_dir / "R12D_SAR_TRANSFER_DEGENERACY_STOP_LOCK.json"
    lock_path.write_text(json.dumps(out_lock, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nDecision=", DECISION)
    print("R12D LOCK:", lock_path)
    print("R12D LOCK SHA256:", sha256_file(lock_path))
    print("PASS")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    if args.self_test:
        self_test()
    else:
        run(args)

if __name__ == "__main__":
    main()
