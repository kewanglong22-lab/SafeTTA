#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R15B0_selective_adaptation_external_utility_preflight_fix1.py

Post-freeze schema/lineage preflight for selective-adaptation / risk-coverage
utility analysis.

This stage DOES NOT compute a utility curve yet.

It verifies that the already-frozen independent external evaluation panels
contain exactly the information required for a label-free ranking policy:

  PolypGen / TENT1
  SUN-SEG / TENT1
  SUN-SEG / PL_CONF90

Future R15B1 policy:
- rank rows by the already-frozen pre-adaptation safety probability;
- adapt only the lowest-risk fraction;
- retain SOURCE prediction for withheld high-risk rows;
- evaluate utility AFTER the ranking policy is fixed;
- no score refit, no calibration, no threshold tuning, no feature changes.

R15B0 only freezes exact file identities, schemas, cardinalities, cluster units,
and the future coverage grid.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


VERSION = "2026-09-04-Q1-R15B0-v1-fix1"
BUILD = "Q1_R15B0_SELECTIVE_ADAPTATION_EXTERNAL_UTILITY_PREFLIGHT_FIX1"

ROOT = Path(r"F:\MEDSEG_SAFETTA")
OUT = ROOT / "outputs"

# ---------------- PolypGen frozen independent external evaluation ----------------
POLYPGEN_DIR = (
    OUT
    / "Q1_R10L3C_polypgen_gt_reveal_prospective_external_evaluation_fix1_v1"
)
POLYPGEN_LOCK = (
    POLYPGEN_DIR
    / "R10L3C_POLYPGEN_PROSPECTIVE_EXTERNAL_EVALUATION_LOCK.json"
)

# ---------------- SUN frozen dual-action confirmatory evaluation -----------------
SUN_DIR = (
    OUT
    / "Q1_R14C3B_sunseg_gt_reveal_dual_action_frozen_score_evaluation_fix1_v1"
)
SUN_LOCK = (
    SUN_DIR
    / "R14C3B_SUNSEG_DUAL_ACTION_FROZEN_SCORE_EVALUATION_LOCK.json"
)
EXPECTED_SUN_LOCK_SHA = (
    "d510c3f1b11aa6210265eba1f30c1c3d8c1c5a7e1c16539abe6b31269488329c"
)

EXPECTED_POLYPGEN_CASES = 1532
EXPECTED_POLYPGEN_STATES = 9
EXPECTED_POLYPGEN_ROWS = EXPECTED_POLYPGEN_CASES * EXPECTED_POLYPGEN_STATES

EXPECTED_SUN_FRAMES = 980
EXPECTED_SUN_CASES = 49
EXPECTED_SUN_STATES = 9
EXPECTED_SUN_ROWS = EXPECTED_SUN_FRAMES * EXPECTED_SUN_STATES

FROZEN_THRESHOLD = 0.300584763193734

# Pre-specified ranking coverage grid. Coverage means fraction of model-frame
# rows that receive adaptation; the remaining rows keep SOURCE.
COVERAGE_GRID = [
    1.00,
    0.90,
    0.80,
    0.70,
    0.60,
    0.50,
    0.40,
    0.30,
    0.20,
    0.10,
    0.00,
]

DEFAULT_OUTPUT = (
    OUT
    / "Q1_R15B0_selective_adaptation_external_utility_preflight_fix1_v1"
)

DECISION = (
    "EXTERNAL_SELECTIVE_ADAPTATION_UTILITY_INPUTS_LOCKED_READY_FOR_R15B1"
)


def sha256_file(path: Path, chunk=8 * 1024 * 1024):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def find_artifact_by_exact_name(lock, root: Path, exact_name: str):
    artifacts = lock.get("artifacts", {})
    if exact_name not in artifacts:
        raise RuntimeError(
            f"Lock does not contain required artifact: {exact_name}"
        )

    meta = artifacts[exact_name]

    # Historical lock formats may store either:
    #   artifacts[name] = {sha256, bytes, ...}
    # or a direct SHA string.
    if isinstance(meta, dict):
        rel = meta.get("relative_path", exact_name)
        expected_sha = str(meta.get("sha256", ""))
    else:
        rel = exact_name
        expected_sha = str(meta)

    path = root / rel
    if not path.is_file():
        raise FileNotFoundError(path)

    actual_sha = sha256_file(path)
    if expected_sha and actual_sha.lower() != expected_sha.lower():
        raise RuntimeError(
            f"Artifact SHA mismatch: {exact_name}"
        )

    return path, actual_sha


def audit_polypgen():
    print("===== POLYPGEN FROZEN PANEL PREFLIGHT =====")

    if not POLYPGEN_LOCK.is_file():
        raise FileNotFoundError(POLYPGEN_LOCK)

    lock_sha = sha256_file(POLYPGEN_LOCK)
    lock = load_json(POLYPGEN_LOCK)

    print("lock:", POLYPGEN_LOCK)
    print("lock SHA256:", lock_sha)
    print("status:", lock.get("status"))
    print("decision:", lock.get("decision"))

    if lock.get("status") != "COMPLETE":
        raise RuntimeError("PolypGen lock status is not COMPLETE.")

    if int(lock.get("target_cases", -1)) != EXPECTED_POLYPGEN_CASES:
        raise RuntimeError("PolypGen case count changed.")
    if int(lock.get("model_states", -1)) != EXPECTED_POLYPGEN_STATES:
        raise RuntimeError("PolypGen state count changed.")
    if int(lock.get("model_case_rows", -1)) != EXPECTED_POLYPGEN_ROWS:
        raise RuntimeError("PolypGen row count changed.")

    panel_name = "R10L3C_LOCKED_SCORE_GT_EVALUATION_PANEL.csv"
    panel_path, panel_sha = find_artifact_by_exact_name(
        lock,
        POLYPGEN_DIR,
        panel_name,
    )

    panel = pd.read_csv(panel_path, low_memory=False)

    required = {
        "sample_id",
        "model_family",
        "model_state_id",
        "source_dice",
        "a1_dice",
        "delta_dice",
        "adaptation_outcome",
        "harm_label",
        "benefit_label",
        "frozen_safety_probability",
        "frozen_operating_threshold",
        "frozen_risk_flag",
    }
    missing = sorted(required - set(panel.columns))
    if missing:
        raise RuntimeError(
            f"PolypGen panel missing columns: {missing}"
        )

    if len(panel) != EXPECTED_POLYPGEN_ROWS:
        raise RuntimeError("PolypGen panel rows changed.")
    if panel["sample_id"].astype(str).nunique() != EXPECTED_POLYPGEN_CASES:
        raise RuntimeError("PolypGen unique case count changed.")
    if panel["model_state_id"].astype(str).nunique() != EXPECTED_POLYPGEN_STATES:
        raise RuntimeError("PolypGen unique state count changed.")
    if (
        panel[["sample_id", "model_state_id"]]
        .drop_duplicates()
        .shape[0]
        != EXPECTED_POLYPGEN_ROWS
    ):
        raise RuntimeError("PolypGen model-case key is not unique.")

    thr = panel["frozen_operating_threshold"].to_numpy(float)
    if not np.allclose(
        thr,
        FROZEN_THRESHOLD,
        rtol=0.0,
        atol=1e-15,
    ):
        raise RuntimeError("PolypGen frozen threshold changed.")

    p = panel["frozen_safety_probability"].to_numpy(float)
    if not np.isfinite(p).all():
        raise RuntimeError("PolypGen score contains non-finite values.")

    print("panel:", panel_path)
    print("panel SHA256:", panel_sha)
    print("rows:", len(panel))
    print("cases:", panel["sample_id"].nunique())
    print("states:", panel["model_state_id"].nunique())
    print("cluster unit for R15B1: sample_id")
    print("score min/median/max:",
          float(np.min(p)),
          float(np.median(p)),
          float(np.max(p)))
    print("PASS")

    return {
        "lock_path": str(POLYPGEN_LOCK),
        "lock_sha256": lock_sha,
        "lock_decision": lock.get("decision"),
        "panel_path": str(panel_path),
        "panel_sha256": panel_sha,
        "rows": len(panel),
        "cases": panel["sample_id"].nunique(),
        "states": panel["model_state_id"].nunique(),
        "cluster_unit": "sample_id",
        "action": "TENT1",
        "source_dice_column": "source_dice",
        "adapted_dice_column": "a1_dice",
        "delta_column": "delta_dice",
        "harm_column": "harm_label",
        "benefit_column": "benefit_label",
        "score_column": "frozen_safety_probability",
    }


def audit_sun():
    print("\n===== SUN FROZEN DUAL-ACTION PANEL PREFLIGHT =====")

    if not SUN_LOCK.is_file():
        raise FileNotFoundError(SUN_LOCK)

    lock_sha = sha256_file(SUN_LOCK)
    if lock_sha != EXPECTED_SUN_LOCK_SHA:
        raise RuntimeError(
            f"SUN R14C3B lock SHA changed: {lock_sha}"
        )

    lock = load_json(SUN_LOCK)
    print("lock:", SUN_LOCK)
    print("lock SHA256:", lock_sha)
    print("status:", lock.get("status"))
    print("decision:", lock.get("decision"))

    if lock.get("status") != "COMPLETE":
        raise RuntimeError("SUN lock status is not COMPLETE.")

    if int(lock.get("target_frames", -1)) != EXPECTED_SUN_FRAMES:
        raise RuntimeError("SUN frame count changed.")
    if int(lock.get("physical_cases", -1)) != EXPECTED_SUN_CASES:
        raise RuntimeError("SUN physical-case count changed.")
    if int(lock.get("model_states", -1)) != EXPECTED_SUN_STATES:
        raise RuntimeError("SUN state count changed.")
    if int(lock.get("model_frame_rows", -1)) != EXPECTED_SUN_ROWS:
        raise RuntimeError("SUN row count changed.")

    panel_name = "R14C3B_LOCKED_SCORE_DUAL_ACTION_EVALUATION_PANEL.csv"
    panel_path, panel_sha = find_artifact_by_exact_name(
        lock,
        SUN_DIR,
        panel_name,
    )

    panel = pd.read_csv(panel_path, low_memory=False)

    required = {
        "sample_id",
        "cluster_id",
        "model_family",
        "model_state_id",
        "source_dice",
        "tent1_dice",
        "tent1_delta_dice",
        "tent1_harm_label",
        "tent1_benefit_label",
        "pl_dice",
        "pl_delta_dice",
        "pl_harm_label",
        "pl_benefit_label",
        "frozen_safety_probability",
        "frozen_operating_threshold",
        "frozen_risk_flag",
    }
    missing = sorted(required - set(panel.columns))
    if missing:
        raise RuntimeError(
            f"SUN panel missing columns: {missing}"
        )

    if len(panel) != EXPECTED_SUN_ROWS:
        raise RuntimeError("SUN panel rows changed.")
    if panel["sample_id"].astype(str).nunique() != EXPECTED_SUN_FRAMES:
        raise RuntimeError("SUN unique frame count changed.")
    if panel["cluster_id"].astype(str).nunique() != EXPECTED_SUN_CASES:
        raise RuntimeError("SUN physical-case count changed.")
    if panel["model_state_id"].astype(str).nunique() != EXPECTED_SUN_STATES:
        raise RuntimeError("SUN state count changed.")
    if (
        panel[["sample_id", "model_state_id"]]
        .drop_duplicates()
        .shape[0]
        != EXPECTED_SUN_ROWS
    ):
        raise RuntimeError("SUN model-frame key is not unique.")

    thr = panel["frozen_operating_threshold"].to_numpy(float)
    if not np.allclose(
        thr,
        FROZEN_THRESHOLD,
        rtol=0.0,
        atol=1e-15,
    ):
        raise RuntimeError("SUN frozen threshold changed.")

    p = panel["frozen_safety_probability"].to_numpy(float)
    if not np.isfinite(p).all():
        raise RuntimeError("SUN score contains non-finite values.")

    print("panel:", panel_path)
    print("panel SHA256:", panel_sha)
    print("rows:", len(panel))
    print("frames:", panel["sample_id"].nunique())
    print("physical cases:", panel["cluster_id"].nunique())
    print("states:", panel["model_state_id"].nunique())
    print("cluster unit for R15B1: cluster_id")
    print("score min/median/max:",
          float(np.min(p)),
          float(np.median(p)),
          float(np.max(p)))
    print("PASS")

    common = {
        "lock_path": str(SUN_LOCK),
        "lock_sha256": lock_sha,
        "lock_decision": lock.get("decision"),
        "panel_path": str(panel_path),
        "panel_sha256": panel_sha,
        "rows": len(panel),
        "frames": panel["sample_id"].nunique(),
        "physical_cases": panel["cluster_id"].nunique(),
        "states": panel["model_state_id"].nunique(),
        "cluster_unit": "cluster_id",
        "score_column": "frozen_safety_probability",
        "source_dice_column": "source_dice",
    }

    tent1 = dict(common)
    tent1.update({
        "action": "TENT1",
        "adapted_dice_column": "tent1_dice",
        "delta_column": "tent1_delta_dice",
        "harm_column": "tent1_harm_label",
        "benefit_column": "tent1_benefit_label",
    })

    pl = dict(common)
    pl.update({
        "action": "PL_CONF90",
        "adapted_dice_column": "pl_dice",
        "delta_column": "pl_delta_dice",
        "harm_column": "pl_harm_label",
        "benefit_column": "pl_benefit_label",
    })

    return tent1, pl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    args = parser.parse_args()

    print("===== Q1 R15B0 SELECTIVE-ADAPTATION UTILITY PREFLIGHT =====")
    print("STATUS=POST_FREEZE_SCHEMA_LINEAGE_PREFLIGHT")
    print("UTILITY_CURVE_COMPUTATION=NO")
    print("MODEL_FITTING=NO")
    print("PCA_FITTING=NO")
    print("SCORE_RECOMPUTATION=NO")
    print("TARGET_CALIBRATION=NO")
    print("THRESHOLD_TUNING=NO")
    print("FEATURE_SELECTION=NO")
    print("CASE_SELECTION_BY_GT=NO")
    print("FROZEN_THRESHOLD=", FROZEN_THRESHOLD)
    print("COVERAGE_GRID=", COVERAGE_GRID)

    polypgen = audit_polypgen()
    sun_tent1, sun_pl = audit_sun()

    print("\n===== R15B1 FROZEN POLICY DEFINITION =====")
    print("score orientation: higher = higher HARM risk")
    print("at each coverage c:")
    print("  adapt the lowest-risk c fraction")
    print("  retain SOURCE for the highest-risk 1-c fraction")
    print("coverage grid:", COVERAGE_GRID)
    print("frozen-threshold policy:")
    print("  adapt iff frozen_safety_probability < 0.300584763193734")
    print("  report as a separate fixed operating point")
    print("coverage-grid ranking is descriptive/post-freeze, not a deployable target-calibrated threshold")
    print("random policy reference will use the same coverage without labels")
    print("NO target labels will select coverage or threshold")

    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)

    lock = {
        "status": "PASS",
        "decision": DECISION,
        "script_version": VERSION,
        "build": BUILD,
        "scientific_role": (
            "post-freeze external selective-adaptation utility preflight"
        ),
        "frozen_threshold": FROZEN_THRESHOLD,
        "coverage_grid": COVERAGE_GRID,
        "policy": {
            "score_orientation": "higher = higher HARM risk",
            "ranked_coverage_rule": (
                "adapt lowest-risk c fraction; retain SOURCE for highest-risk 1-c"
            ),
            "fixed_threshold_rule": (
                "adapt iff frozen_safety_probability < frozen_threshold"
            ),
            "coverage_selected_by_target_labels": False,
            "threshold_changed": False,
            "score_changed": False,
            "model_changed": False,
        },
        "inputs": {
            "PolypGen_TENT1": polypgen,
            "SUN_TENT1": sun_tent1,
            "SUN_PL_CONF90": sun_pl,
        },
        "r15b1_planned_metrics": [
            "adaptation_coverage",
            "harm_rate_among_adapted",
            "benefit_rate_among_adapted",
            "prevented_harm_fraction",
            "withheld_benefit_fraction",
            "mean_deployed_dice",
            "median_deployed_dice",
            "mean_delta_vs_source",
            "mean_delta_vs_adapt_all",
            "fixed_threshold_policy_point",
        ],
        "r15b1_bootstrap": {
            "PolypGen_cluster": "sample_id",
            "SUN_cluster": "cluster_id",
            "replicates": 2000,
            "same_cluster_resample_across_coverages_within_setting": True,
        },
        "prohibitions": {
            "score_refit": True,
            "target_calibration": True,
            "threshold_tuning": True,
            "coverage_selection_by_gt": True,
            "feature_change": True,
            "case_exclusion": True,
            "action_specific_safety_retraining": True,
        },
        "next_stage": (
            "R15B1_SELECTIVE_ADAPTATION_RISK_COVERAGE_UTILITY"
        ),
    }

    lock_path = args.output_dir / "R15B0_SELECTIVE_ADAPTATION_UTILITY_INPUT_LOCK.json"
    lock_path.write_text(
        json.dumps(lock, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n===== R15B0 FINAL =====")
    print("Decision=", DECISION)
    print("LOCK=", lock_path)
    print("LOCK SHA256=", sha256_file(lock_path))
    print("PASS")


if __name__ == "__main__":
    main()
