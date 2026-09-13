#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SafeTTA R33A1
External Joint Domain + Unseen-Action Shift Protocol Lock

Primary experiment
------------------
SOURCE training domain:
    NeoPolyp
SOURCE training actions:
    TENT1 + PL-CONF90

External target domain:
    PolypGen
External target action:
    MEMO-SEG4-1STEP  (unseen during predictor training)

Primary question:
    Can the frozen SafeTTA transition representation trained on NeoPolyp
    transfer simultaneously across DOMAIN and ACTION to rank future HARM
    for MEMO on PolypGen?

Important chronology / claim boundary
-------------------------------------
PolypGen GT has been historically available and used in earlier R17A work.
Therefore this is NOT claimed as a pristine prospective / pre-GT external
study.

What IS locked relative to the NEW MEMO joint-shift execution:
  - exact PolypGen 1532-case x 3-DeepLab-state target cohort;
  - exact pre-existing SOURCE-reliability baseline scores;
  - exact NeoPolyp TENT+PL source-training panel;
  - exact SafeTTA model class / features / hyperparameters;
  - exact MEMO action configuration;
  - exact metrics / success criterion;
  - no target calibration or model selection.

R33A1 itself performs NO:
  - PolypGen MEMO inference
  - new DINO inference
  - model fitting
  - GT decode / HARM evaluation
  - target calibration
  - threshold tuning
  - score reversal
  - external web/network access

Why R33A0 keyword counts are NOT used
-------------------------------------
R33A0's broad inventory was a discovery audit. Its top matches included the
audit script itself and unrelated files whose text mentioned PolypGen/action
tokens. R33A1 therefore does not use broad keyword match counts as scientific
evidence. It binds exact R17A PRE-GT artifacts already independently frozen
in R32B1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


VERSION = "2026-09-12-R33A1-v1"

ROOT = Path(r"F:\MEDSEG_SAFETTA")
CODE = ROOT / "code"
OUTPUTS = ROOT / "outputs"

# ---------------------------------------------------------------------
# R33A0 / R32B lineage
# ---------------------------------------------------------------------

R33A0_SCRIPT = CODE / "Q1_R33A0_polypgen_joint_domain_action_shift_asset_audit_v1.py"
EXPECTED_R33A0_SCRIPT_SHA256 = (
    "63a30e62dbbace17c6b36d62e6454d750ec440441de9c0fc00dcd38caf7efc4d"
)

R33A0_DIR = ROOT / "R33A0_polypgen_joint_domain_action_shift_asset_audit_v1"
R33A0_FINAL = R33A0_DIR / "R33A0_FINAL_LOCK.json"
EXPECTED_R33A0_FINAL_SHA256 = (
    "907e7b04f64dda7a3493a73cd38d8f998cce28dd5e7de554e686a29ad27dfe72"
)

R32B3_DIR = ROOT / "R32B3_matched_three_action_harm_evaluation_v1_fix1"
R32B3_FINAL = R32B3_DIR / "R32B3_FINAL_LOCK.json"
EXPECTED_R32B3_FINAL_SHA256 = (
    "0503a49dc7fe2a9ffe882a8257ed066751d5fdbcc3178bc8898797a489073520"
)

R32B1_DIR = ROOT / "R32B1_faithful_common_baseline_panel_lock_v1"
R32B1_FINAL = R32B1_DIR / "R32B1_FINAL_LOCK.json"
EXPECTED_R32B1_FINAL_SHA256 = (
    "ce305d32f77b2c72e5ca6589043aa6644542e35ee74fa7f41e68db45d4cd31e4"
)
R32B1_KEY_ASSETS = R32B1_DIR / "R32B1_KEY_R17A_ASSET_BINDINGS.csv"

# ---------------------------------------------------------------------
# Source-training panel
# ---------------------------------------------------------------------

R31B3_DIR = ROOT / "R31B3_first_800case_gt_reveal_and_three_action_loao_v1"
R31B3_FINAL = R31B3_DIR / "R31B3_FINAL_LOCK.json"
EXPECTED_R31B3_FINAL_SHA256 = (
    "eb91d991752c50e71b1993e4ee0a50e0b9e9cdb35fffdd58b36553e0cebaaadc"
)
R31B3_TABLE = R31B3_DIR / "R31B3_THREE_ACTION_MODELCASE_TABLE.csv"

R30A0_SCRIPT = CODE / "Q1_R30A0_paired_action_crc_protocol_and_asset_audit_v1_fix1.py"
EXPECTED_R30A0_SHA256 = (
    "9dc5d3a13f3d64608c4c599f811f929f731d7fac05e2aaa66eb344e24ace2298"
)

# ---------------------------------------------------------------------
# Exact retained PolypGen PRE-GT target assets
# ---------------------------------------------------------------------

CCD_PREGT = (
    OUTPUTS
    / "Q1_R17A_CCD_polypgen_full9_preGT_score_lock_v1_fix2"
    / "R17A_POLYPGEN_CCD_FULL9_PREGT_SCORE_LOCK.csv"
)

ADIC_MC_PREGT = (
    OUTPUTS
    / "Q1_R17A_ADIC_MC_deeplab3_preGT_score_lock_v1_fix2"
    / "R17A_POLYPGEN_DEEPLAB3_ADIC_MC_PREGT_SCORE_LOCK.csv"
)

# ---------------------------------------------------------------------
# Frozen experiment definition
# ---------------------------------------------------------------------

SOURCE_DATASET = "NeoPolyp"
SOURCE_ACTIONS = ("TENT1", "PL-CONF90")
SOURCE_CASES = 800
SOURCE_STATES = 9
EXPECTED_SOURCE_ROWS = SOURCE_CASES * SOURCE_STATES * len(SOURCE_ACTIONS)

TARGET_DATASET = "PolypGen"
TARGET_FAMILY = "DeepLabV3-R50"
TARGET_SEEDS = (20260817, 20260818, 20260819)
TARGET_STATE_IDS = tuple(f"{TARGET_FAMILY}::{s}" for s in TARGET_SEEDS)
TARGET_CASES = 1532
TARGET_STATES = 3
EXPECTED_TARGET_MODELCASES = TARGET_CASES * TARGET_STATES

TARGET_ACTION = "MEMO-SEG4-1STEP"

MEMO_LR = 1e-5
MEMO_OPTIMIZER = "Adam"
MEMO_WEIGHT_DECAY = 0.0
MEMO_UPDATE_STEPS = 1
MEMO_TRANSFORMS = ("ID", "HFLIP", "VFLIP", "HVFLIP")
MEMO_OBJECTIVE = (
    "binary entropy of inverse-aligned marginal mean foreground probability"
)
MEMO_ADAPT_PARAMS = "all model parameters"
MEMO_FINAL_PREDICTION = "original image"

SAFE_FEATURE_SET = "QSOURCE66 + DSEM64"
SAFE_MODEL = "StandardScaler + balanced LogisticRegression"
SAFE_LR_C = 1.0
SAFE_LR_SOLVER = "lbfgs"
SAFE_LR_MAX_ITER = 5000
SAFE_LR_RANDOM_STATE = 0

HARM_DELTA_DICE_THRESHOLD = -0.02

PRIMARY_METRICS = ("AUROC", "AUPRC", "AUPRC_LIFT")
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 20260912
BOOTSTRAP_CLUSTER = "physical sample_id"

CONFIRM_AUROC_CI_LOW_GT = 0.5
CONFIRM_AUPRC_MINUS_PREVALENCE_CI_LOW_GT = 0.0

DEFAULT_OUT = ROOT / "R33A1_external_joint_shift_protocol_lock_v1"


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


def atomic_csv(df: pd.DataFrame, path: Path):
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def verify_upstream() -> Dict[str, Any]:
    require_sha(R33A0_SCRIPT, EXPECTED_R33A0_SCRIPT_SHA256, "R33A0 script")
    require_sha(R33A0_FINAL, EXPECTED_R33A0_FINAL_SHA256, "R33A0 final")
    require_sha(R32B3_FINAL, EXPECTED_R32B3_FINAL_SHA256, "R32B3 final")
    require_sha(R32B1_FINAL, EXPECTED_R32B1_FINAL_SHA256, "R32B1 final")
    require_sha(R31B3_FINAL, EXPECTED_R31B3_FINAL_SHA256, "R31B3 final")
    require_sha(R30A0_SCRIPT, EXPECTED_R30A0_SHA256, "R30A0 feature schema")

    a0 = json.loads(R33A0_FINAL.read_text(encoding="utf-8"))
    if a0.get("status") != (
        "PASS_R33A0_POLYPGEN_JOINT_SHIFT_ASSET_AUDIT_COMPLETE"
    ):
        raise RuntimeError("R33A0 final status changed.")
    if a0.get("decision") != (
        "READY_TO_LOCK_R33A1_EXTERNAL_JOINT_SHIFT_PROTOCOL"
    ):
        raise RuntimeError(f"R33A0 decision changed: {a0.get('decision')}")

    b3 = json.loads(R32B3_FINAL.read_text(encoding="utf-8"))
    if b3.get("status") != (
        "PASS_R32B3_MATCHED_THREE_ACTION_HARM_EVALUATION_COMPLETE"
    ):
        raise RuntimeError("R32B3 final status changed.")

    b1 = json.loads(R32B1_FINAL.read_text(encoding="utf-8"))
    if b1.get("status") != (
        "PASS_R32B1_FAITHFUL_COMMON_BASELINE_PANEL_LOCK_COMPLETE"
    ):
        raise RuntimeError("R32B1 final status changed.")

    r31 = json.loads(R31B3_FINAL.read_text(encoding="utf-8"))
    if r31.get("status") is None:
        raise RuntimeError("R31B3 final lock missing status.")

    return {
        "R33A0": a0,
        "R32B3": b3,
        "R32B1": b1,
        "R31B3": r31,
    }


def bind_exact_polypgen_assets() -> Dict[str, Dict[str, Any]]:
    if not R32B1_KEY_ASSETS.is_file():
        raise FileNotFoundError(R32B1_KEY_ASSETS)

    b1 = json.loads(R32B1_FINAL.read_text(encoding="utf-8"))
    expected = str(b1.get("key_asset_bindings_sha256", ""))
    if not expected:
        raise RuntimeError(
            "R32B1 final lock lacks key_asset_bindings_sha256."
        )

    got = sha256_file(R32B1_KEY_ASSETS)
    if got != expected:
        raise RuntimeError(
            f"R32B1 key-asset binding SHA drift expected={expected} got={got}"
        )

    d = pd.read_csv(R32B1_KEY_ASSETS, low_memory=False)

    needed = {
        "CCD_full9_preGT_scores": CCD_PREGT,
        "ADIC_MC_deeplab3_preGT_scores": ADIC_MC_PREGT,
    }

    out = {}
    for label, path in needed.items():
        g = d[d["asset_label"].astype(str) == label]
        if len(g) != 1:
            raise RuntimeError(
                f"Expected exactly one R32B1 asset binding for {label}; "
                f"got {len(g)}"
            )
        row = g.iloc[0]

        if not bool(row["exists"]):
            raise RuntimeError(f"R32B1 says exact asset absent: {label}")
        if not path.is_file():
            raise FileNotFoundError(path)

        bound_sha = str(row["sha256"])
        actual_sha = sha256_file(path)
        if bound_sha != actual_sha:
            raise RuntimeError(
                f"{label} SHA drift bound={bound_sha} actual={actual_sha}"
            )

        out[label] = {
            "path": str(path),
            "sha256": actual_sha,
            "rows_R32B1_binding": (
                int(row["rows"])
                if pd.notna(row["rows"])
                else None
            ),
        }

    return out


def build_target_manifest() -> Tuple[pd.DataFrame, pd.DataFrame]:
    adic = pd.read_csv(
        ADIC_MC_PREGT,
        dtype={
            "sample_id": str,
            "model_family": str,
            "model_state_id": str,
        },
        low_memory=False,
    )

    ccd = pd.read_csv(
        CCD_PREGT,
        dtype={
            "sample_id": str,
            "model_family": str,
            "model_state_id": str,
        },
        low_memory=False,
    )

    adic_required = {
        "sample_id",
        "image_path",
        "image_raw_sha256",
        "model_family",
        "model_state_id",
        "training_seed",
        "checkpoint_sha256",
        "adic_harm_risk",
        "mc_predictive_entropy_risk",
    }
    ccd_required = {
        "sample_id",
        "model_family",
        "model_state_id",
        "training_seed",
        "ccd_risk",
    }

    ma = sorted(adic_required.difference(adic.columns))
    mc = sorted(ccd_required.difference(ccd.columns))
    if ma:
        raise RuntimeError(f"ADIC/MC PRE-GT asset missing={ma}")
    if mc:
        raise RuntimeError(f"CCD PRE-GT asset missing={mc}")

    adic = adic[
        adic["model_family"].astype(str) == TARGET_FAMILY
    ].copy()
    ccd = ccd[
        ccd["model_family"].astype(str) == TARGET_FAMILY
    ].copy()

    if len(adic) != EXPECTED_TARGET_MODELCASES:
        raise RuntimeError(
            f"PolypGen DeepLab ADIC/MC rows={len(adic)} "
            f"expected={EXPECTED_TARGET_MODELCASES}"
        )
    if len(ccd) != EXPECTED_TARGET_MODELCASES:
        raise RuntimeError(
            f"PolypGen DeepLab CCD rows={len(ccd)} "
            f"expected={EXPECTED_TARGET_MODELCASES}"
        )

    for label, d in [("ADIC/MC", adic), ("CCD", ccd)]:
        if d["sample_id"].nunique() != TARGET_CASES:
            raise RuntimeError(
                f"{label}: PolypGen cases={d['sample_id'].nunique()} "
                f"expected={TARGET_CASES}"
            )
        if d["model_state_id"].nunique() != TARGET_STATES:
            raise RuntimeError(f"{label}: target state count drift.")
        if set(d["model_state_id"].astype(str)) != set(TARGET_STATE_IDS):
            raise RuntimeError(
                f"{label}: target state IDs drift="
                f"{sorted(d['model_state_id'].astype(str).unique())}"
            )
        if set(pd.to_numeric(d["training_seed"]).astype(int)) != set(TARGET_SEEDS):
            raise RuntimeError(f"{label}: target training seeds drift.")
        if d.duplicated(["sample_id", "model_state_id"]).any():
            raise RuntimeError(f"{label}: duplicate target model-case rows.")

    keep = [
        c for c in [
            "sample_id",
            "original_polypgen_sample_id",
            "center",
            "image_path",
            "image_raw_sha256",
            "model_family",
            "model_state_id",
            "training_seed",
            "checkpoint_sha256",
            "n_dropout",
            "native_dropout_p",
            "adic_harm_risk",
            "mc_predictive_entropy_risk",
        ]
        if c in adic.columns
    ]

    target = adic[keep].copy()

    ccd_small = ccd[
        ["sample_id", "model_state_id", "ccd_risk"]
    ].copy()

    target = target.merge(
        ccd_small,
        on=["sample_id", "model_state_id"],
        how="inner",
        validate="one_to_one",
    )

    if len(target) != EXPECTED_TARGET_MODELCASES:
        raise RuntimeError("Exact PolypGen PRE-GT common join lost rows.")

    for c in [
        "ccd_risk",
        "adic_harm_risk",
        "mc_predictive_entropy_risk",
    ]:
        if not np.isfinite(target[c].to_numpy(float)).all():
            raise RuntimeError(f"Non-finite target frozen baseline score: {c}")

    forbidden = []
    for c in target.columns:
        low = str(c).lower()
        if (
            "harm_label" in low
            or "delta_dice" in low
            or "adapted_dice" in low
            or "source_dice" in low
            or low in {"gt", "ground_truth"}
        ):
            forbidden.append(c)
    if forbidden:
        raise RuntimeError(
            f"Target PRE-GT manifest unexpectedly contains outcome columns={forbidden}"
        )

    target = target.sort_values(
        ["model_state_id", "sample_id"],
        kind="mergesort",
    ).reset_index(drop=True)

    summary = (
        target.groupby(
            ["model_state_id", "training_seed"],
            as_index=False,
        )
        .agg(
            rows=("sample_id", "size"),
            physical_cases=("sample_id", "nunique"),
            ccd_mean=("ccd_risk", "mean"),
            adic_harm_risk_mean=("adic_harm_risk", "mean"),
            mc_entropy_mean=("mc_predictive_entropy_risk", "mean"),
        )
        .sort_values("training_seed", kind="mergesort")
    )

    return target, summary


def build_source_training_manifest() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if not R31B3_TABLE.is_file():
        raise FileNotFoundError(R31B3_TABLE)

    d = pd.read_csv(
        R31B3_TABLE,
        dtype={
            "sample_id": str,
            "model_state_id": str,
            "model_family": str,
            "action": str,
        },
        low_memory=False,
    )

    required = {
        "sample_id",
        "model_state_id",
        "model_family",
        "action",
        "r31b3_harm_label",
    }
    missing = sorted(required.difference(d.columns))
    if missing:
        raise RuntimeError(f"R31B3 source table missing={missing}")

    g = d[d["action"].isin(SOURCE_ACTIONS)].copy()

    if len(g) != EXPECTED_SOURCE_ROWS:
        raise RuntimeError(
            f"NeoPolyp source TENT+PL rows={len(g)} "
            f"expected={EXPECTED_SOURCE_ROWS}"
        )
    if g["sample_id"].nunique() != SOURCE_CASES:
        raise RuntimeError("NeoPolyp source case count drift.")
    if g["model_state_id"].nunique() != SOURCE_STATES:
        raise RuntimeError("NeoPolyp source state count drift.")
    if set(g["action"].astype(str)) != set(SOURCE_ACTIONS):
        raise RuntimeError("NeoPolyp source action set drift.")
    if g.duplicated(["sample_id", "model_state_id", "action"]).any():
        raise RuntimeError("Duplicate NeoPolyp source-training row.")

    y = g["r31b3_harm_label"].to_numpy(int)
    if not set(np.unique(y)).issubset({0, 1}):
        raise RuntimeError("NeoPolyp HARM label drift.")

    manifest = (
        g[
            [
                "sample_id",
                "model_state_id",
                "model_family",
                "action",
                "r31b3_harm_label",
            ]
        ]
        .sort_values(
            ["sample_id", "model_state_id", "action"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    counts = (
        g.groupby("action")["r31b3_harm_label"]
        .agg(["count", "sum", "mean"])
        .reset_index()
    )

    info = {
        "source_table_sha256": sha256_file(R31B3_TABLE),
        "rows": int(len(g)),
        "physical_cases": int(g["sample_id"].nunique()),
        "model_states": int(g["model_state_id"].nunique()),
        "actions": list(SOURCE_ACTIONS),
        "harm_by_action": counts.to_dict(orient="records"),
    }

    return manifest, info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    print("=" * 124)
    print("SafeTTA R33A1 External Joint Domain + Unseen-Action Shift Protocol Lock")
    print("Version                       :", VERSION)
    print("Source                         : NeoPolyp / TENT1 + PL-CONF90")
    print("External target                : PolypGen / MEMO-SEG4-1STEP")
    print("Target family                  :", TARGET_FAMILY)
    print("Target physical cases          :", TARGET_CASES)
    print("Target model states            :", TARGET_STATES)
    print("Historical PolypGen GT existed : YES")
    print("Claimed pristine prospective   : NO")
    print("New inference                  : NO")
    print("GT/HARM evaluation             : NO")
    print("Target calibration             : NO")
    print("External web/network access    : NO")
    print("=" * 124)

    verify_upstream()
    exact_assets = bind_exact_polypgen_assets()
    target, target_summary = build_target_manifest()
    source_manifest, source_info = build_source_training_manifest()

    out = args.output_dir.resolve()
    if out.exists():
        raise RuntimeError(
            f"Output exists; R33A1 will not overwrite: {out}"
        )
    out.mkdir(parents=True, exist_ok=False)

    target_path = out / "R33A1_POLYPGEN_1532x3_PREGT_TARGET_MANIFEST.csv"
    target_summary_path = out / "R33A1_POLYPGEN_TARGET_STATE_SUMMARY.csv"
    source_path = out / "R33A1_NEOPOLYP_TENT_PL_SOURCE_TRAINING_MANIFEST.csv"

    atomic_csv(target, target_path)
    atomic_csv(target_summary, target_summary_path)
    atomic_csv(source_manifest, source_path)

    protocol = {
        "status": "LOCKED_R33A1_EXTERNAL_JOINT_DOMAIN_ACTION_SHIFT_PROTOCOL",
        "version": VERSION,
        "scientific_status": (
            "POST_R32B3; HISTORICALLY_GT_AVAILABLE_TARGET; "
            "LOCKED_BEFORE_NEW_POLYPGEN_MEMO_EXECUTION"
        ),
        "claim_boundaries": {
            "pristine_prospective_external_validation": False,
            "pre_GT_target_lock": False,
            "historical_PolypGen_GT_available": True,
            "protocol_locked_before_new_MEMO_execution": True,
            "target_calibration_allowed": False,
            "target_model_selection_allowed": False,
        },
        "R33A0_note": (
            "Broad keyword match counts are discovery-only and are NOT used "
            "as evidence of exact asset readiness. Exact SHA-bound R17A "
            "PRE-GT assets are used instead."
        ),
        "source_training": {
            "dataset": SOURCE_DATASET,
            "cases": SOURCE_CASES,
            "model_states": SOURCE_STATES,
            "actions": list(SOURCE_ACTIONS),
            "rows": EXPECTED_SOURCE_ROWS,
            "source_table_sha256": source_info["source_table_sha256"],
            "source_manifest_sha256": sha256_file(source_path),
            "harm_definition": "delta_Dice <= -0.02",
            "harm_threshold": HARM_DELTA_DICE_THRESHOLD,
            "feature_set": SAFE_FEATURE_SET,
            "model": SAFE_MODEL,
            "StandardScaler_fit": "NeoPolyp source training rows only",
            "LogisticRegression": {
                "C": SAFE_LR_C,
                "class_weight": "balanced",
                "solver": SAFE_LR_SOLVER,
                "max_iter": SAFE_LR_MAX_ITER,
                "random_state": SAFE_LR_RANDOM_STATE,
            },
            "target_data_used_for_fit": False,
            "harm_counts": source_info["harm_by_action"],
        },
        "external_target": {
            "dataset": TARGET_DATASET,
            "family": TARGET_FAMILY,
            "training_seeds": list(TARGET_SEEDS),
            "state_ids": list(TARGET_STATE_IDS),
            "physical_cases": TARGET_CASES,
            "model_states": TARGET_STATES,
            "model_cases": EXPECTED_TARGET_MODELCASES,
            "target_manifest_sha256": sha256_file(target_path),
            "cohort_source": (
                "exact intersection of retained R17A DeepLab3 ADIC/MC "
                "and CCD PRE-GT score locks"
            ),
            "preexisting_baseline_scores": {
                "SicTTA_CCD": "ccd_risk",
                "TEGDA_ADIC": "adic_harm_risk",
                "MC_dropout": "mc_predictive_entropy_risk",
            },
            "exact_asset_bindings": exact_assets,
        },
        "target_action": {
            "name": TARGET_ACTION,
            "optimizer": MEMO_OPTIMIZER,
            "learning_rate": MEMO_LR,
            "weight_decay": MEMO_WEIGHT_DECAY,
            "update_steps": MEMO_UPDATE_STEPS,
            "transforms": list(MEMO_TRANSFORMS),
            "objective": MEMO_OBJECTIVE,
            "adapted_parameters": MEMO_ADAPT_PARAMS,
            "final_prediction": MEMO_FINAL_PREDICTION,
            "hyperparameter_search_on_PolypGen": False,
        },
        "target_representation": {
            "q_source": (
                "frozen SOURCE prediction-conditioned semantic/morphology "
                "representation using existing SafeTTA source-side artifacts"
            ),
            "delta_semantic": (
                "64-D semantic transition between SOURCE and candidate MEMO "
                "prediction, computed without target GT"
            ),
            "final_feature": SAFE_FEATURE_SET,
            "ActionID_in_primary_model": False,
            "target_fit_or_recalibration": False,
        },
        "evaluation_after_score_and_action_lock": {
            "HARM_definition": "MEMO_delta_Dice <= -0.02",
            "metrics": list(PRIMARY_METRICS),
            "AUPRC_lift_definition": "AUPRC / HARM prevalence",
            "bootstrap": {
                "reps": BOOTSTRAP_REPS,
                "seed": BOOTSTRAP_SEED,
                "cluster": BOOTSTRAP_CLUSTER,
                "retain_all_3_states_per_sampled_case": True,
            },
            "primary_confirmation_criterion": {
                "AUROC_bootstrap_CI_low_gt": CONFIRM_AUROC_CI_LOW_GT,
                "AUPRC_minus_prevalence_bootstrap_CI_low_gt": (
                    CONFIRM_AUPRC_MINUS_PREVALENCE_CI_LOW_GT
                ),
                "joint_rule": "BOTH must pass",
            },
            "published_baseline_comparison": (
                "paired on the exact same 1532 x 3 target model-cases; "
                "descriptive + paired physical-case clustered bootstrap; "
                "no post-hoc direction reversal"
            ),
        },
        "prohibitions": [
            "no PolypGen target label use in predictor fit",
            "no target calibration",
            "no target feature selection",
            "no target threshold tuning",
            "no post-hoc score direction reversal",
            "no MEMO learning-rate tuning on PolypGen",
            "no action selection using PolypGen outcomes",
            "no architecture subset selection after seeing MEMO outcomes",
        ],
        "lineage": {
            "R33A0_final_sha256": EXPECTED_R33A0_FINAL_SHA256,
            "R32B3_final_sha256": EXPECTED_R32B3_FINAL_SHA256,
            "R32B1_final_sha256": EXPECTED_R32B1_FINAL_SHA256,
            "R31B3_final_sha256": EXPECTED_R31B3_FINAL_SHA256,
            "R30A0_schema_sha256": EXPECTED_R30A0_SHA256,
        },
        "next": "R33A2_EXTERNAL_MEMO_AND_TRANSITION_SCORE_LOCK",
    }

    protocol_path = out / "R33A1_EXTERNAL_JOINT_SHIFT_PROTOCOL_LOCK.json"
    atomic_json(protocol_path, protocol)

    final = {
        "status": "PASS_R33A1_EXTERNAL_JOINT_SHIFT_PROTOCOL_LOCK_COMPLETE",
        "version": VERSION,
        "source_direction": "NeoPolyp TENT1 + PL-CONF90",
        "target_direction": "PolypGen MEMO-SEG4-1STEP",
        "historical_PolypGen_GT_available": True,
        "pristine_prospective_claim": False,
        "protocol_locked_before_new_MEMO_execution": True,
        "target_cases": TARGET_CASES,
        "target_states": TARGET_STATES,
        "target_modelcases": EXPECTED_TARGET_MODELCASES,
        "source_training_rows": EXPECTED_SOURCE_ROWS,
        "target_manifest_sha256": sha256_file(target_path),
        "source_manifest_sha256": sha256_file(source_path),
        "protocol_sha256": sha256_file(protocol_path),
        "new_inference": False,
        "GT_HARM_evaluation": False,
        "target_calibration": False,
        "external_web_network_access": False,
        "next": "R33A2_EXTERNAL_MEMO_AND_TRANSITION_SCORE_LOCK",
    }

    final_path = out / "R33A1_FINAL_LOCK.json"
    atomic_json(final_path, final)

    print("\nR33A1 EXACT POLYPGEN TARGET COHORT")
    print("  physical cases :", target["sample_id"].nunique())
    print("  model states   :", target["model_state_id"].nunique())
    print("  model-cases    :", len(target))
    print("  state IDs      :", sorted(target["model_state_id"].unique().tolist()))
    print("  GT/HARM columns: NONE")

    print("\nR33A1 TARGET PRE-GT BASELINE ASSETS")
    for label, meta in exact_assets.items():
        print(f"  {label}")
        print(f"    rows : {meta['rows_R32B1_binding']}")
        print(f"    SHA  : {meta['sha256']}")

    print("\nR33A1 NEOPOLYP SOURCE TRAINING PANEL")
    print("  cases      :", source_info["physical_cases"])
    print("  states     :", source_info["model_states"])
    print("  actions    :", source_info["actions"])
    print("  rows       :", source_info["rows"])
    print("  HARM counts:")
    for row in source_info["harm_by_action"]:
        print("   ", row)

    print("\nR33A1 PRIMARY EXTERNAL DIRECTION")
    print("  NeoPolyp TENT1 + PL-CONF90")
    print("      -> PolypGen MEMO-SEG4-1STEP")
    print("  feature : QSOURCE66 + DSEM64")
    print("  model   : StandardScaler + balanced LogisticRegression(C=1)")
    print("  target calibration : NO")

    print("\nR33A1 CONFIRMATION CRITERION")
    print("  AUROC clustered-bootstrap CI lower > 0.5")
    print("  AND")
    print("  AUPRC - HARM prevalence clustered-bootstrap CI lower > 0")

    print(
        "\nFINAL STATUS : "
        "PASS_R33A1_EXTERNAL_JOINT_SHIFT_PROTOCOL_LOCK_COMPLETE"
    )
    print("Historical PolypGen GT existed : YES")
    print("Pristine prospective claim     : NO")
    print("New MEMO inference             : NO")
    print("GT/HARM evaluation             : NO")
    print("Protocol SHA256                :", sha256_file(protocol_path))
    print("Final lock SHA256              :", sha256_file(final_path))
    print("Output                         :", out)
    print("NEXT                           : R33A2_EXTERNAL_MEMO_AND_TRANSITION_SCORE_LOCK")
    print("=" * 124)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
