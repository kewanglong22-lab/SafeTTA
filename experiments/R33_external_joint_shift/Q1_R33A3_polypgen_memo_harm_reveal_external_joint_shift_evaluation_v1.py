#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SafeTTA R33A3
PolypGen MEMO HARM Reveal + External Joint Domain/Action Shift Evaluation

Scientific direction
--------------------
Fit:
    NeoPolyp TENT1 + PL-CONF90

External target:
    PolypGen MEMO-SEG4-1STEP

This stage is the FIRST evaluation of the newly locked PolypGen MEMO
outcomes.  PolypGen GT existed historically and was used in R10L3C/R17A,
so this is NOT described as first-ever GT access or pristine prospective
validation.  The relevant chronology is:

    R33A1 protocol locked
      -> R33A2B MEMO predictions/features/scores locked without new HARM read
      -> THIS R33A3 evaluates the newly generated MEMO outcomes.

Primary confirmatory criterion was frozen in R33A1:
  1) physical-case clustered-bootstrap 95% CI lower(AUROC) > 0.5
  2) physical-case clustered-bootstrap 95% CI lower(AUPRC - HARM prevalence) > 0
  BOTH must pass.

Primary evaluation scope locked here BEFORE new MEMO outcome construction:
    pooled 4596 DeepLab model-cases, with physical sample_id bootstrap clusters
    retaining all three DeepLab states together.

Secondary:
  - per-state and macro-state metrics
  - matched published reliability baselines:
      SicTTA-CCD, TEGDA-ADIC, MC-dropout
  - paired physical-case clustered bootstrap:
      SafeTTA - comparator for AUROC and AUPRC
  - segmentation outcome summaries.

No model inference, no TTA, no predictor refit, no target calibration,
no score reversal, no target feature/case/state selection.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.metrics import average_precision_score, roc_auc_score


VERSION = "2026-09-12-R33A3-v1"

ROOT = Path(r"F:\MEDSEG_SAFETTA")
CODE = ROOT / "code"
OUTPUTS = ROOT / "outputs"

# =============================================================================
# R33A1 / R33A2B exact locks
# =============================================================================

R33A1_DIR = ROOT / "R33A1_external_joint_shift_protocol_lock_v1"
R33A1_FINAL = R33A1_DIR / "R33A1_FINAL_LOCK.json"
EXPECTED_R33A1_FINAL_SHA256 = (
    "cff6f482b7b170d3be868d6a1ce4ec26e828d062a3a51501efed4a40a3fa684a"
)
R33A1_PROTOCOL = R33A1_DIR / "R33A1_EXTERNAL_JOINT_SHIFT_PROTOCOL_LOCK.json"
EXPECTED_R33A1_PROTOCOL_SHA256 = (
    "1779be8accd4d316170754eab8aaa642a42d7e4a5999625eda27bc662179dd2c"
)
R33A1_TARGET = R33A1_DIR / "R33A1_POLYPGEN_1532x3_PREGT_TARGET_MANIFEST.csv"

R33A2B_SCRIPT = (
    CODE / "Q1_R33A2B_external_polypgen_memo_transition_score_lock_v1_fix1.py"
)
EXPECTED_R33A2B_SCRIPT_SHA256 = (
    "712e862e57f97aabf46a65054bddadde896e0fd44ad5655a9b6490562cf96641"
)

R33A2B_DIR = ROOT / "R33A2B_external_polypgen_memo_transition_score_lock_v1_fix1"
R33A2B_FINAL = R33A2B_DIR / "R33A2B_FINAL_LOCK.json"
EXPECTED_R33A2B_FINAL_SHA256 = (
    "399e250b4e14ae1423f049fb21bb3623333b6e5899c45437c1be4cbd84d1f965"
)
R33A2B_SCORE_LOCK = R33A2B_DIR / "R33A2B_SCORE_PRE_HARM_LOCK.json"
EXPECTED_R33A2B_SCORE_LOCK_SHA256 = (
    "d8c31f233c7f52777e5fa5cb56298988a1e622dd896053f929f4a6414db2ce27"
)
R33A2B_SCORE = (
    R33A2B_DIR / "R33A2B_POLYPGEN_MEMO_EXTERNAL_SAFETTA_SCORE_PRE_HARM_LOCK.csv"
)
EXPECTED_R33A2B_SCORE_SHA256 = (
    "0b8bfca0594214b106880607216771dd6375a2b14ce8e0280aafbfd197b03cdf"
)
EXPECTED_R33A2B_PREDICTION_INVENTORY_SHA256 = (
    "3492869d282646c1d3406b67afdad2d98c764ff056def103e6ddfb928494a8b9"
)

# =============================================================================
# Exact historical PolypGen GT / Dice implementation
# =============================================================================

R10L3C_SCRIPT = (
    CODE / "Q1_R10L3C_polypgen_gt_reveal_prospective_external_evaluation_fix1.py"
)
EXPECTED_R10L3C_SCRIPT_SHA256 = (
    "6761de1da2c6ece73dd1b18f566517b0c438c37b060a369084bdf26624375e6f"
)

R10L3C_DIR = (
    OUTPUTS / "Q1_R10L3C_polypgen_gt_reveal_prospective_external_evaluation_fix1_v1"
)
R10L3C_PANEL = R10L3C_DIR / "R10L3C_LOCKED_SCORE_GT_EVALUATION_PANEL.csv"
EXPECTED_R10L3C_PANEL_SHA256 = (
    "c484d10c4196359f62a34c0b0fa34ae817a1fdf9d451abb1a37c333d92576a67"
)
R10L3C_LOCK = R10L3C_DIR / "R10L3C_POLYPGEN_PROSPECTIVE_EXTERNAL_EVALUATION_LOCK.json"
EXPECTED_R10L3C_LOCK_SHA256 = (
    "cf9e42ea7f0f6c1d53844a2358b05007b46d2a99124b585014626704245d3861"
)

# =============================================================================
# Frozen R33 protocol
# =============================================================================

TARGET_FAMILY = "DeepLabV3-R50"
TARGET_SEEDS = (20260817, 20260818, 20260819)
TARGET_STATE_IDS = tuple(f"{TARGET_FAMILY}::{s}" for s in TARGET_SEEDS)
TARGET_CASES = 1532
TARGET_STATES = 3
EXPECTED_ROWS = TARGET_CASES * TARGET_STATES

MASK_H = 352
MASK_W = 352
MASK_PIXELS = MASK_H * MASK_W
PACKED_BYTES = (MASK_PIXELS + 7) // 8

HARM_THRESHOLD = -0.02
BENEFIT_THRESHOLD = 0.02

SAFE_METHOD = "SafeTTA-Q66+dSemantic64"
METHOD_TO_SCORE = {
    SAFE_METHOD: "safettta_external_harm_risk",
    "SicTTA-CCD": "ccd_risk",
    "TEGDA-ADIC": "adic_harm_risk",
    "MC-dropout": "mc_predictive_entropy_risk",
}

BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 20260912

CONFIRM_AUROC_LOW_GT = 0.5
CONFIRM_AUPRC_MINUS_PREV_LOW_GT = 0.0

PRIMARY_SCOPE = "POOLED_4596_MODEL_CASES"
BOOTSTRAP_CLUSTER = "physical sample_id; all 3 DeepLab states retained"

DEFAULT_OUT = (
    ROOT / "R33A3_first_polypgen_memo_harm_reveal_external_joint_shift_evaluation_v1"
)


# =============================================================================
# Generic utilities
# =============================================================================

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


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


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


def unpack_mask(packed: np.ndarray, bitorder: str) -> np.ndarray:
    p = np.asarray(packed, dtype=np.uint8)
    if p.shape != (PACKED_BYTES,):
        raise RuntimeError(
            f"Packed mask shape={p.shape}; expected={(PACKED_BYTES,)}"
        )
    bits = np.unpackbits(
        p,
        count=MASK_PIXELS,
        bitorder=bitorder,
    )
    return bits.reshape(MASK_H, MASK_W).astype(np.uint8, copy=False)


def outcome_label(delta: float) -> str:
    if float(delta) <= HARM_THRESHOLD:
        return "HARM"
    if float(delta) >= BENEFIT_THRESHOLD:
        return "BENEFIT"
    return "NEUTRAL"


# =============================================================================
# Immutable upstream verification
# =============================================================================

def verify_upstream() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    require_sha(R33A1_FINAL, EXPECTED_R33A1_FINAL_SHA256, "R33A1 final")
    require_sha(R33A1_PROTOCOL, EXPECTED_R33A1_PROTOCOL_SHA256, "R33A1 protocol")
    require_sha(R33A2B_SCRIPT, EXPECTED_R33A2B_SCRIPT_SHA256, "R33A2B fix1 script")
    require_sha(R33A2B_FINAL, EXPECTED_R33A2B_FINAL_SHA256, "R33A2B final")
    require_sha(
        R33A2B_SCORE_LOCK,
        EXPECTED_R33A2B_SCORE_LOCK_SHA256,
        "R33A2B score lock",
    )
    require_sha(R33A2B_SCORE, EXPECTED_R33A2B_SCORE_SHA256, "R33A2B target score")

    require_sha(R10L3C_SCRIPT, EXPECTED_R10L3C_SCRIPT_SHA256, "R10L3C GT script")
    require_sha(R10L3C_PANEL, EXPECTED_R10L3C_PANEL_SHA256, "R10L3C historical panel")
    require_sha(R10L3C_LOCK, EXPECTED_R10L3C_LOCK_SHA256, "R10L3C historical lock")

    a1 = json.loads(R33A1_FINAL.read_text(encoding="utf-8"))
    if a1.get("status") != "PASS_R33A1_EXTERNAL_JOINT_SHIFT_PROTOCOL_LOCK_COMPLETE":
        raise RuntimeError("R33A1 final status drift.")

    protocol = json.loads(R33A1_PROTOCOL.read_text(encoding="utf-8"))
    criterion = protocol["evaluation_after_score_and_action_lock"][
        "primary_confirmation_criterion"
    ]
    if float(criterion["AUROC_bootstrap_CI_low_gt"]) != CONFIRM_AUROC_LOW_GT:
        raise RuntimeError("R33A1 AUROC confirmation criterion drift.")
    if (
        float(criterion["AUPRC_minus_prevalence_bootstrap_CI_low_gt"])
        != CONFIRM_AUPRC_MINUS_PREV_LOW_GT
    ):
        raise RuntimeError("R33A1 AUPRC-minus-prevalence criterion drift.")
    if int(
        protocol["evaluation_after_score_and_action_lock"]["bootstrap"]["reps"]
    ) != BOOTSTRAP_REPS:
        raise RuntimeError("R33A1 bootstrap reps drift.")
    if int(
        protocol["evaluation_after_score_and_action_lock"]["bootstrap"]["seed"]
    ) != BOOTSTRAP_SEED:
        raise RuntimeError("R33A1 bootstrap seed drift.")

    b = json.loads(R33A2B_FINAL.read_text(encoding="utf-8"))
    if b.get("status") != "PASS_R33A2B_EXTERNAL_MEMO_TRANSITION_SCORE_LOCK_COMPLETE":
        raise RuntimeError("R33A2B final status drift.")
    if bool(b.get("target_GT_HARM_read_evaluated", True)):
        raise RuntimeError("R33A2B reports target GT/HARM use.")
    if bool(b.get("target_calibration", True)):
        raise RuntimeError("R33A2B target calibration guard drift.")
    if bool(b.get("target_score_reversal", True)):
        raise RuntimeError("R33A2B target score reversal guard drift.")
    if b.get("target_score_sha256") != EXPECTED_R33A2B_SCORE_SHA256:
        raise RuntimeError("R33A2B final target-score SHA field drift.")
    if (
        b.get("prediction_inventory_sha256")
        != EXPECTED_R33A2B_PREDICTION_INVENTORY_SHA256
    ):
        raise RuntimeError("R33A2B prediction inventory SHA drift.")

    score_lock = json.loads(R33A2B_SCORE_LOCK.read_text(encoding="utf-8"))
    if bool(score_lock.get("target_HARM_evaluated", True)):
        raise RuntimeError("R33A2B score lock reports prior HARM evaluation.")
    if bool(score_lock.get("target_calibration", True)):
        raise RuntimeError("R33A2B score lock calibration guard drift.")
    if bool(score_lock.get("target_score_reversal", True)):
        raise RuntimeError("R33A2B score lock reversal guard drift.")

    return protocol, b


# =============================================================================
# Pre-new-MEMO-outcome analysis lock
# =============================================================================

def write_analysis_lock(out: Path) -> Path:
    payload = {
        "status": "LOCKED_R33A3_BEFORE_NEW_POLYPGEN_MEMO_OUTCOME_CONSTRUCTION",
        "version": VERSION,
        "historical_PolypGen_GT_previously_available": True,
        "claim_first_ever_GT_reveal": False,
        "primary_scope": PRIMARY_SCOPE,
        "primary_method": SAFE_METHOD,
        "primary_endpoint": "MEMO HARM where MEMO_delta_Dice <= -0.02",
        "primary_metrics": ["AUROC", "AUPRC", "AUPRC_LIFT"],
        "confirmation_criterion": {
            "AUROC_physical_case_bootstrap_95CI_lower_gt": CONFIRM_AUROC_LOW_GT,
            "AUPRC_minus_HARM_prevalence_bootstrap_95CI_lower_gt": (
                CONFIRM_AUPRC_MINUS_PREV_LOW_GT
            ),
            "joint_rule": "BOTH must pass",
        },
        "bootstrap": {
            "reps": BOOTSTRAP_REPS,
            "seed": BOOTSTRAP_SEED,
            "cluster": BOOTSTRAP_CLUSTER,
            "percentile_CI": [0.025, 0.975],
        },
        "secondary": [
            "per-state point metrics",
            "macro-state point metrics",
            "published reliability comparator point metrics",
            "paired clustered-bootstrap SafeTTA-minus-comparator AUROC/AUPRC",
            "SOURCE/MEMO Dice and HARM/NEUTRAL/BENEFIT summaries",
        ],
        "comparators": METHOD_TO_SCORE,
        "prohibitions": [
            "no predictor refit",
            "no target calibration",
            "no score direction reversal",
            "no threshold tuning",
            "no target feature selection",
            "no target case/state selection",
            "no model inference",
            "no TTA rerun",
        ],
        "R33A1_protocol_sha256": EXPECTED_R33A1_PROTOCOL_SHA256,
        "R33A2B_final_sha256": EXPECTED_R33A2B_FINAL_SHA256,
        "R33A2B_score_lock_sha256": EXPECTED_R33A2B_SCORE_LOCK_SHA256,
        "R33A2B_target_score_sha256": EXPECTED_R33A2B_SCORE_SHA256,
        "R10L3C_GT_implementation_sha256": EXPECTED_R10L3C_SCRIPT_SHA256,
    }
    path = out / "R33A3_PRE_NEW_MEMO_OUTCOME_ANALYSIS_LOCK.json"
    atomic_json(path, payload)
    return path


# =============================================================================
# Locked score/prediction loading
# =============================================================================

def load_score_panel() -> pd.DataFrame:
    d = pd.read_csv(
        R33A2B_SCORE,
        dtype={
            "sample_id": str,
            "model_state_id": str,
            "model_family": str,
        },
        low_memory=False,
    )

    required = {
        "sample_id",
        "model_state_id",
        "model_family",
        "training_seed",
        "checkpoint_sha256",
        "action",
        *METHOD_TO_SCORE.values(),
    }
    missing = sorted(required.difference(d.columns))
    if missing:
        raise RuntimeError(f"R33A2B score table missing={missing}")

    if len(d) != EXPECTED_ROWS:
        raise RuntimeError(f"Score rows={len(d)} expected={EXPECTED_ROWS}")
    if d["sample_id"].nunique() != TARGET_CASES:
        raise RuntimeError("Score physical-case count drift.")
    if d["model_state_id"].nunique() != TARGET_STATES:
        raise RuntimeError("Score state count drift.")
    if set(d["model_state_id"].astype(str)) != set(TARGET_STATE_IDS):
        raise RuntimeError("Score state IDs drift.")
    if set(d["model_family"].astype(str)) != {TARGET_FAMILY}:
        raise RuntimeError("Score family drift.")
    if set(d["action"].astype(str)) != {"MEMO-SEG4-1STEP"}:
        raise RuntimeError("Score action drift.")
    if d.duplicated(["sample_id", "model_state_id"]).any():
        raise RuntimeError("Duplicate locked score model-case.")

    for col in METHOD_TO_SCORE.values():
        if not np.isfinite(d[col].to_numpy(float)).all():
            raise RuntimeError(f"Non-finite frozen score={col}")

    return d


def state_npz_path(state_index: int) -> Path:
    return (
        R33A2B_DIR
        / f"state_{state_index:02d}_polypgen_source_memo_pre_harm.npz"
    )


def state_lock_path(state_index: int) -> Path:
    return (
        R33A2B_DIR
        / f"state_{state_index:02d}_polypgen_source_memo_pre_harm.lock.json"
    )


def load_prediction_state(
    state_index: int,
    state_id: str,
    training_seed: int,
) -> Tuple[np.ndarray, np.ndarray, str]:
    npz_path = state_npz_path(state_index)
    lock_path = state_lock_path(state_index)
    if not npz_path.is_file() or not lock_path.is_file():
        raise FileNotFoundError(f"Missing R33A2B prediction state={state_id}")

    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("status") != "PASS_R33A2B_STATE_SOURCE_MEMO_PRE_HARM_LOCK":
        raise RuntimeError(f"Prediction state lock failed: {state_id}")
    if lock.get("model_state_id") != state_id:
        raise RuntimeError("Prediction state_id mismatch.")
    if int(lock.get("training_seed")) != int(training_seed):
        raise RuntimeError("Prediction training_seed mismatch.")
    if int(lock.get("physical_cases")) != TARGET_CASES:
        raise RuntimeError("Prediction physical-case count mismatch.")
    if lock.get("prediction_npz_sha256") != sha256_file(npz_path):
        raise RuntimeError("Prediction NPZ SHA mismatch.")

    with np.load(npz_path, allow_pickle=False) as z:
        src = np.asarray(z["source_masks_packed"], dtype=np.uint8)
        memo = np.asarray(z["memo_masks_packed"], dtype=np.uint8)

    expected = (TARGET_CASES, PACKED_BYTES)
    if src.shape != expected or memo.shape != expected:
        raise RuntimeError(
            f"Prediction packed shape drift source={src.shape} memo={memo.shape}"
        )

    return src, memo, sha256_file(npz_path)


# =============================================================================
# Exact historical GT reveal + outcome construction
# =============================================================================

def validate_r10_gt_module(r10):
    required_functions = [
        "load_locked_gt_manifest",
        "reveal_gt_masks",
        "binary_dice",
    ]
    missing = [x for x in required_functions if not hasattr(r10, x)]
    if missing:
        raise RuntimeError(f"R10L3C exact GT functions missing={missing}")

    if int(getattr(r10, "EXPECTED_CASES")) != TARGET_CASES:
        raise RuntimeError("R10L3C PolypGen case count drift.")
    if int(getattr(r10, "MASK_H")) != MASK_H:
        raise RuntimeError("R10L3C mask height drift.")
    if int(getattr(r10, "MASK_W")) != MASK_W:
        raise RuntimeError("R10L3C mask width drift.")
    if float(getattr(r10, "HARM_THRESHOLD")) != HARM_THRESHOLD:
        raise RuntimeError("R10L3C HARM threshold drift.")
    if float(getattr(r10, "BENEFIT_THRESHOLD")) != BENEFIT_THRESHOLD:
        raise RuntimeError("R10L3C BENEFIT threshold drift.")


def reveal_exact_gt(r10, out: Path) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
    gt_df = r10.load_locked_gt_manifest()
    gt_masks, gt_audit = r10.reveal_gt_masks(gt_df)

    if len(gt_masks) != TARGET_CASES:
        raise RuntimeError(f"GT masks={len(gt_masks)} expected={TARGET_CASES}")

    # Standardize audit for artifact persistence.
    if isinstance(gt_audit, pd.DataFrame):
        audit_df = gt_audit.copy()
    else:
        audit_df = pd.DataFrame(gt_audit)

    audit_path = out / "R33A3_POLYPGEN_GT_REVEAL_AUDIT.csv"
    atomic_csv(audit_df, audit_path)
    return gt_masks, audit_df


def build_memo_outcomes(
    *,
    score: pd.DataFrame,
    gt_masks: Dict[str, np.ndarray],
    r10,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    bitorder = str(getattr(r10, "BITORDER", "big"))

    state_meta = (
        score[
            [
                "model_state_id",
                "training_seed",
                "checkpoint_sha256",
            ]
        ]
        .drop_duplicates()
        .sort_values("training_seed", kind="mergesort")
        .reset_index(drop=True)
    )
    if len(state_meta) != TARGET_STATES:
        raise RuntimeError("Target state metadata count drift.")

    for sidx, state in state_meta.iterrows():
        state_id = str(state["model_state_id"])
        seed = int(state["training_seed"])

        g = (
            score[score["model_state_id"].astype(str) == state_id]
            .sort_values("sample_id", kind="mergesort")
            .reset_index(drop=True)
        )
        if len(g) != TARGET_CASES:
            raise RuntimeError(f"{state_id}: score rows={len(g)}")

        src_pack, memo_pack, npz_sha = load_prediction_state(
            int(sidx),
            state_id,
            seed,
        )

        for i in tqdm(
            range(TARGET_CASES),
            desc=f"R33A3 MEMO outcome seed {seed}",
            unit="case",
            dynamic_ncols=True,
        ):
            r = g.iloc[i]
            sid = str(r["sample_id"])
            if sid not in gt_masks:
                raise RuntimeError(f"Missing GT mask sample={sid}")

            source_mask = unpack_mask(src_pack[i], bitorder)
            memo_mask = unpack_mask(memo_pack[i], bitorder)
            gt = np.asarray(gt_masks[sid], dtype=np.uint8)

            if gt.shape != (MASK_H, MASK_W):
                raise RuntimeError(f"GT shape={gt.shape} sample={sid}")

            source_dice = float(r10.binary_dice(source_mask, gt))
            memo_dice = float(r10.binary_dice(memo_mask, gt))
            delta = float(memo_dice - source_dice)
            label = outcome_label(delta)

            rows.append({
                "sample_id": sid,
                "model_family": TARGET_FAMILY,
                "model_state_id": state_id,
                "training_seed": seed,
                "checkpoint_sha256": str(r["checkpoint_sha256"]),
                "prediction_npz_sha256": npz_sha,
                "source_dice": source_dice,
                "memo_dice": memo_dice,
                "memo_delta_dice": delta,
                "memo_adaptation_outcome": label,
                "memo_harm_label": int(label == "HARM"),
                "memo_benefit_label": int(label == "BENEFIT"),
            })

    out = pd.DataFrame(rows)
    if len(out) != EXPECTED_ROWS:
        raise RuntimeError(f"Outcome rows={len(out)} expected={EXPECTED_ROWS}")
    if out.duplicated(["sample_id", "model_state_id"]).any():
        raise RuntimeError("Duplicate MEMO outcome key.")
    if set(out["memo_harm_label"].unique()).difference({0, 1}):
        raise RuntimeError("MEMO HARM label non-binary.")

    return out


def historical_source_dice_parity(outcomes: pd.DataFrame) -> Dict[str, Any]:
    hist = pd.read_csv(
        R10L3C_PANEL,
        dtype={
            "sample_id": str,
            "model_state_id": str,
            "model_family": str,
        },
        low_memory=False,
    )

    required = {"sample_id", "model_state_id", "model_family", "source_dice"}
    missing = sorted(required.difference(hist.columns))
    if missing:
        raise RuntimeError(f"Historical R10L3C panel missing={missing}")

    hist = hist[
        (hist["model_family"].astype(str) == TARGET_FAMILY)
        & (hist["model_state_id"].astype(str).isin(TARGET_STATE_IDS))
    ][["sample_id", "model_state_id", "source_dice"]].copy()

    if len(hist) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Historical DeepLab source rows={len(hist)} expected={EXPECTED_ROWS}"
        )
    if hist.duplicated(["sample_id", "model_state_id"]).any():
        raise RuntimeError("Historical source-Dice duplicate key.")

    chk = outcomes.merge(
        hist.rename(columns={"source_dice": "historical_source_dice"}),
        on=["sample_id", "model_state_id"],
        how="left",
        validate="one_to_one",
    )
    if chk["historical_source_dice"].isna().any():
        raise RuntimeError("Historical source-Dice alignment incomplete.")

    diff = (
        chk["source_dice"].to_numpy(float)
        - chk["historical_source_dice"].to_numpy(float)
    )
    max_abs = float(np.max(np.abs(diff)))
    mismatch = int(np.count_nonzero(np.abs(diff) > 1e-12))

    if mismatch != 0:
        raise RuntimeError(
            f"Historical SOURCE Dice parity failed mismatch={mismatch} "
            f"max_abs={max_abs}"
        )

    return {
        "rows": EXPECTED_ROWS,
        "exact_with_tolerance_1e_12": True,
        "mismatch_rows": mismatch,
        "max_abs_difference": max_abs,
        "historical_panel_sha256": EXPECTED_R10L3C_PANEL_SHA256,
    }


# =============================================================================
# Metrics
# =============================================================================

def point_metrics(y: np.ndarray, score: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)

    if set(np.unique(y)) != {0, 1}:
        raise RuntimeError("Point metric target is class-degenerate.")

    prevalence = float(np.mean(y))
    auroc = float(roc_auc_score(y, score))
    auprc = float(average_precision_score(y, score))
    return {
        "AUROC": auroc,
        "AUPRC": auprc,
        "HARM_PREVALENCE": prevalence,
        "AUPRC_MINUS_PREVALENCE": float(auprc - prevalence),
        "AUPRC_LIFT": float(auprc / prevalence),
    }


def build_point_tables(panel: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y = panel["memo_harm_label"].to_numpy(int)

    pooled_rows = []
    for method, col in METHOD_TO_SCORE.items():
        m = point_metrics(y, panel[col].to_numpy(float))
        pooled_rows.append({
            "scope": PRIMARY_SCOPE,
            "method": method,
            "rows": len(panel),
            "physical_cases": panel["sample_id"].nunique(),
            **m,
        })
    pooled = pd.DataFrame(pooled_rows)

    state_rows = []
    for state_id, g in panel.groupby("model_state_id", sort=True):
        yy = g["memo_harm_label"].to_numpy(int)
        for method, col in METHOD_TO_SCORE.items():
            m = point_metrics(yy, g[col].to_numpy(float))
            state_rows.append({
                "scope": "STATE",
                "model_state_id": state_id,
                "method": method,
                "rows": len(g),
                "physical_cases": g["sample_id"].nunique(),
                **m,
            })
    state = pd.DataFrame(state_rows)

    macro_rows = []
    for method in METHOD_TO_SCORE:
        g = state[state["method"] == method]
        macro_rows.append({
            "scope": "MACRO_3_STATES",
            "method": method,
            "states": TARGET_STATES,
            "macro_AUROC": float(g["AUROC"].mean()),
            "macro_AUPRC": float(g["AUPRC"].mean()),
            "macro_AUPRC_LIFT": float(g["AUPRC_LIFT"].mean()),
        })
    macro = pd.DataFrame(macro_rows)

    return pooled, state, macro


def segmentation_summary(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def summarize(label: str, g: pd.DataFrame):
        labels = g["memo_adaptation_outcome"].value_counts().to_dict()
        rows.append({
            "scope": label,
            "rows": len(g),
            "physical_cases": g["sample_id"].nunique(),
            "source_dice_mean": float(g["source_dice"].mean()),
            "source_dice_median": float(g["source_dice"].median()),
            "memo_dice_mean": float(g["memo_dice"].mean()),
            "memo_dice_median": float(g["memo_dice"].median()),
            "memo_delta_dice_mean": float(g["memo_delta_dice"].mean()),
            "memo_delta_dice_median": float(g["memo_delta_dice"].median()),
            "harm_rows": int(labels.get("HARM", 0)),
            "neutral_rows": int(labels.get("NEUTRAL", 0)),
            "benefit_rows": int(labels.get("BENEFIT", 0)),
            "harm_prevalence": float(g["memo_harm_label"].mean()),
            "benefit_prevalence": float(g["memo_benefit_label"].mean()),
        })

    summarize("POOLED", panel)
    for state_id, g in panel.groupby("model_state_id", sort=True):
        summarize(state_id, g)

    return pd.DataFrame(rows)


# =============================================================================
# Physical-case clustered bootstrap
# =============================================================================

def bootstrap_evaluation(panel: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # A row belongs to exactly one of 1532 physical clusters; each cluster has 3 states.
    case_ids = sorted(panel["sample_id"].astype(str).unique().tolist())
    if len(case_ids) != TARGET_CASES:
        raise RuntimeError("Bootstrap physical-case count drift.")

    case_to_i = {sid: i for i, sid in enumerate(case_ids)}
    row_case_idx = np.asarray(
        [case_to_i[str(s)] for s in panel["sample_id"].astype(str)],
        dtype=np.int32,
    )

    counts_per_case = panel.groupby("sample_id").size()
    if not (counts_per_case == TARGET_STATES).all():
        raise RuntimeError("Bootstrap clusters do not each contain 3 states.")

    y = panel["memo_harm_label"].to_numpy(int)
    scores = {
        method: panel[col].to_numpy(float)
        for method, col in METHOD_TO_SCORE.items()
    }

    rng = np.random.default_rng(BOOTSTRAP_SEED)

    method_stats = {
        method: {
            "AUROC": [],
            "AUPRC": [],
            "AUPRC_MINUS_PREVALENCE": [],
            "PREVALENCE": [],
        }
        for method in METHOD_TO_SCORE
    }
    paired = {
        comparator: {"AUROC_DELTA": [], "AUPRC_DELTA": []}
        for comparator in METHOD_TO_SCORE
        if comparator != SAFE_METHOD
    }

    valid = 0

    for _ in tqdm(
        range(BOOTSTRAP_REPS),
        desc="R33A3 physical-case clustered bootstrap",
        unit="rep",
        dynamic_ncols=True,
    ):
        sampled = rng.integers(0, TARGET_CASES, size=TARGET_CASES)
        case_weights = np.bincount(
            sampled,
            minlength=TARGET_CASES,
        ).astype(np.float64)
        w = case_weights[row_case_idx]

        pos = float(np.sum(w[y == 1]))
        neg = float(np.sum(w[y == 0]))
        if pos <= 0 or neg <= 0:
            continue

        prevalence = float(np.average(y, weights=w))
        rep_metrics = {}

        for method, score in scores.items():
            auc = float(roc_auc_score(y, score, sample_weight=w))
            ap = float(average_precision_score(y, score, sample_weight=w))
            method_stats[method]["AUROC"].append(auc)
            method_stats[method]["AUPRC"].append(ap)
            method_stats[method]["AUPRC_MINUS_PREVALENCE"].append(
                float(ap - prevalence)
            )
            method_stats[method]["PREVALENCE"].append(prevalence)
            rep_metrics[method] = (auc, ap)

        safe_auc, safe_ap = rep_metrics[SAFE_METHOD]
        for comparator in paired:
            auc, ap = rep_metrics[comparator]
            paired[comparator]["AUROC_DELTA"].append(float(safe_auc - auc))
            paired[comparator]["AUPRC_DELTA"].append(float(safe_ap - ap))

        valid += 1

    if valid < int(0.95 * BOOTSTRAP_REPS):
        raise RuntimeError(
            f"Too few valid clustered bootstrap replicates={valid}/{BOOTSTRAP_REPS}"
        )

    point = {
        method: point_metrics(y, score)
        for method, score in scores.items()
    }

    boot_rows = []
    for method in METHOD_TO_SCORE:
        for metric in ["AUROC", "AUPRC", "AUPRC_MINUS_PREVALENCE"]:
            vals = np.asarray(method_stats[method][metric], dtype=float)
            lo, hi = np.quantile(vals, [0.025, 0.975])
            point_key = metric
            boot_rows.append({
                "method": method,
                "metric": metric,
                "point": float(point[method][point_key]),
                "bootstrap_reps_requested": BOOTSTRAP_REPS,
                "bootstrap_reps_valid": valid,
                "ci95_low": float(lo),
                "ci95_high": float(hi),
            })

    paired_rows = []
    for comparator, metrics in paired.items():
        for metric, vals0 in metrics.items():
            vals = np.asarray(vals0, dtype=float)
            lo, hi = np.quantile(vals, [0.025, 0.975])
            base_metric = "AUROC" if metric == "AUROC_DELTA" else "AUPRC"
            observed = float(
                point[SAFE_METHOD][base_metric]
                - point[comparator][base_metric]
            )
            paired_rows.append({
                "comparator": comparator,
                "metric": base_metric,
                "delta_SafeTTA_minus_comparator": observed,
                "bootstrap_reps_requested": BOOTSTRAP_REPS,
                "bootstrap_reps_valid": valid,
                "ci95_low": float(lo),
                "ci95_high": float(hi),
                "SafeTTA_significantly_better_95CI": bool(lo > 0),
                "comparator_significantly_better_95CI": bool(hi < 0),
            })

    return pd.DataFrame(boot_rows), pd.DataFrame(paired_rows)


def confirmation_gate(
    pooled: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> Dict[str, Any]:
    safe = pooled[pooled["method"] == SAFE_METHOD]
    if len(safe) != 1:
        raise RuntimeError("SafeTTA pooled row missing.")
    safe = safe.iloc[0]

    def b(metric: str) -> pd.Series:
        g = bootstrap[
            (bootstrap["method"] == SAFE_METHOD)
            & (bootstrap["metric"] == metric)
        ]
        if len(g) != 1:
            raise RuntimeError(f"SafeTTA bootstrap row missing metric={metric}")
        return g.iloc[0]

    auc = b("AUROC")
    ap_minus = b("AUPRC_MINUS_PREVALENCE")

    pass_auc = bool(float(auc["ci95_low"]) > CONFIRM_AUROC_LOW_GT)
    pass_ap = bool(
        float(ap_minus["ci95_low"]) > CONFIRM_AUPRC_MINUS_PREV_LOW_GT
    )
    joint = bool(pass_auc and pass_ap)

    return {
        "status": "R33A3_EXTERNAL_JOINT_SHIFT_CONFIRMATION_GATE",
        "primary_scope": PRIMARY_SCOPE,
        "SafeTTA_point": {
            "AUROC": float(safe["AUROC"]),
            "AUPRC": float(safe["AUPRC"]),
            "HARM_prevalence": float(safe["HARM_PREVALENCE"]),
            "AUPRC_minus_prevalence": float(
                safe["AUPRC_MINUS_PREVALENCE"]
            ),
            "AUPRC_lift": float(safe["AUPRC_LIFT"]),
        },
        "criterion_1_AUROC": {
            "threshold": CONFIRM_AUROC_LOW_GT,
            "bootstrap_ci95_low": float(auc["ci95_low"]),
            "bootstrap_ci95_high": float(auc["ci95_high"]),
            "pass": pass_auc,
        },
        "criterion_2_AUPRC_minus_prevalence": {
            "threshold": CONFIRM_AUPRC_MINUS_PREV_LOW_GT,
            "bootstrap_ci95_low": float(ap_minus["ci95_low"]),
            "bootstrap_ci95_high": float(ap_minus["ci95_high"]),
            "pass": pass_ap,
        },
        "joint_rule": "BOTH must pass",
        "joint_confirmation": joint,
        "decision": (
            "EXTERNAL_JOINT_DOMAIN_ACTION_SHIFT_CONFIRMED"
            if joint
            else "NO_EXTERNAL_JOINT_DOMAIN_ACTION_SHIFT_CONFIRMATION"
        ),
    }


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    print("=" * 124)
    print("SafeTTA R33A3 PolypGen MEMO HARM Reveal + External Joint-Shift Evaluation")
    print("Version                         :", VERSION)
    print("Historical PolypGen GT existed  : YES")
    print("First-ever PolypGen GT claim     : NO")
    print("New MEMO outcome evaluation      : YES")
    print("Model inference                  : NO")
    print("TTA                              : NO")
    print("Predictor refit                  : NO")
    print("Target calibration               : NO")
    print("Score reversal                   : NO")
    print("Primary scope                    :", PRIMARY_SCOPE)
    print("Bootstrap                        :", BOOTSTRAP_REPS, "physical-case clusters")
    print("=" * 124)

    protocol, r33a2b_final = verify_upstream()
    score = load_score_panel()

    out = args.output_dir.resolve()
    if out.exists():
        raise RuntimeError(f"Output exists; R33A3 will not overwrite: {out}")
    out.mkdir(parents=True, exist_ok=False)

    # Freeze exact analysis interpretation before constructing new MEMO outcomes.
    analysis_lock = write_analysis_lock(out)
    print("\nR33A3 PRE-NEW-MEMO-OUTCOME ANALYSIS LOCK")
    print("  SHA256 :", sha256_file(analysis_lock))
    print("  primary:", PRIMARY_SCOPE)
    print("  criterion 1: AUROC CI low > 0.5")
    print("  criterion 2: AUPRC - prevalence CI low > 0")

    # Exact historical GT implementation.
    r10 = import_module(R10L3C_SCRIPT, "r33a3_r10l3c_exact_gt")
    validate_r10_gt_module(r10)

    print("\nR33A3 POLYPGEN GT ACCESS")
    print("  exact historical R10L3C implementation SHA:", EXPECTED_R10L3C_SCRIPT_SHA256)
    print("  decode: historical frozen PolypGen rule")
    gt_masks, gt_audit = reveal_exact_gt(r10, out)
    print("  GT cases decoded:", len(gt_masks))
    print(
        "  GT empty after 352x352 decode:",
        sum(int(np.asarray(m).sum()) == 0 for m in gt_masks.values()),
    )

    outcomes = build_memo_outcomes(
        score=score,
        gt_masks=gt_masks,
        r10=r10,
    )

    # Mandatory exact SOURCE Dice replay against historical PolypGen evaluation.
    parity = historical_source_dice_parity(outcomes)

    print("\nR33A3 HISTORICAL SOURCE-DICE PARITY")
    print("  rows       :", parity["rows"])
    print("  mismatches :", parity["mismatch_rows"])
    print("  max abs    :", parity["max_abs_difference"])
    print("  PASS       :", parity["exact_with_tolerance_1e_12"])

    # Join already-frozen scores only after outcomes have been constructed.
    panel = outcomes.merge(
        score,
        on=[
            "sample_id",
            "model_family",
            "model_state_id",
            "training_seed",
            "checkpoint_sha256",
        ],
        how="inner",
        validate="one_to_one",
    )
    if len(panel) != EXPECTED_ROWS:
        raise RuntimeError("Outcome/score matched panel row count drift.")

    pooled, state, macro = build_point_tables(panel)
    seg = segmentation_summary(panel)
    boot, paired = bootstrap_evaluation(panel)
    gate = confirmation_gate(pooled, boot)

    # Save artifacts.
    outcome_path = out / "R33A3_POLYPGEN_MEMO_MODELCASE_OUTCOMES.csv"
    panel_path = out / "R33A3_POLYPGEN_MEMO_MATCHED_SCORE_OUTCOME_PANEL.csv"
    pooled_path = out / "R33A3_PRIMARY_POOLED_METHOD_METRICS.csv"
    state_path = out / "R33A3_STATE_SECONDARY_METRICS.csv"
    macro_path = out / "R33A3_MACRO_STATE_SECONDARY_METRICS.csv"
    seg_path = out / "R33A3_SEGMENTATION_OUTCOME_SUMMARY.csv"
    boot_path = out / "R33A3_PHYSICAL_CASE_CLUSTERED_BOOTSTRAP.csv"
    paired_path = out / "R33A3_PAIRED_PUBLISHED_BASELINE_DELTAS.csv"
    parity_path = out / "R33A3_HISTORICAL_SOURCE_DICE_PARITY.json"
    gate_path = out / "R33A3_EXTERNAL_JOINT_SHIFT_CONFIRMATION_GATE.json"

    atomic_csv(outcomes, outcome_path)
    atomic_csv(panel, panel_path)
    atomic_csv(pooled, pooled_path)
    atomic_csv(state, state_path)
    atomic_csv(macro, macro_path)
    atomic_csv(seg, seg_path)
    atomic_csv(boot, boot_path)
    atomic_csv(paired, paired_path)
    atomic_json(parity_path, parity)
    atomic_json(gate_path, gate)

    final = {
        "status": "PASS_R33A3_POLYPGEN_MEMO_EXTERNAL_JOINT_SHIFT_EVALUATION_COMPLETE",
        "version": VERSION,
        "scientific_status": (
            "POST-R33A2B NEW_MEMO_OUTCOME_EVALUATION_ON_HISTORICALLY_GT_ACCESSED_POLYPGEN"
        ),
        "decision": gate["decision"],
        "joint_confirmation": gate["joint_confirmation"],
        "historical_PolypGen_GT_previously_available": True,
        "first_ever_GT_claim": False,
        "R33A1_final_sha256": EXPECTED_R33A1_FINAL_SHA256,
        "R33A1_protocol_sha256": EXPECTED_R33A1_PROTOCOL_SHA256,
        "R33A2B_script_sha256": EXPECTED_R33A2B_SCRIPT_SHA256,
        "R33A2B_final_sha256": EXPECTED_R33A2B_FINAL_SHA256,
        "R33A2B_score_pre_harm_lock_sha256": EXPECTED_R33A2B_SCORE_LOCK_SHA256,
        "R33A2B_target_score_sha256": EXPECTED_R33A2B_SCORE_SHA256,
        "R10L3C_GT_implementation_sha256": EXPECTED_R10L3C_SCRIPT_SHA256,
        "R10L3C_historical_panel_sha256": EXPECTED_R10L3C_PANEL_SHA256,
        "pre_outcome_analysis_lock_sha256": sha256_file(analysis_lock),
        "historical_source_dice_parity_sha256": sha256_file(parity_path),
        "outcomes_sha256": sha256_file(outcome_path),
        "matched_panel_sha256": sha256_file(panel_path),
        "pooled_metrics_sha256": sha256_file(pooled_path),
        "state_metrics_sha256": sha256_file(state_path),
        "macro_state_metrics_sha256": sha256_file(macro_path),
        "segmentation_summary_sha256": sha256_file(seg_path),
        "bootstrap_sha256": sha256_file(boot_path),
        "paired_baseline_delta_sha256": sha256_file(paired_path),
        "gate_sha256": sha256_file(gate_path),
        "model_inference": False,
        "TTA_rerun": False,
        "predictor_refit": False,
        "target_calibration": False,
        "score_reversal": False,
        "target_subset_selection": False,
        "next": "R33A4_PAPER_INTEGRATION_AND_CLAIM_FREEZE",
    }
    final_path = out / "R33A3_FINAL_LOCK.json"
    atomic_json(final_path, final)

    print("\n" + "=" * 124)
    print("R33A3 SEGMENTATION OUTCOME SUMMARY")
    print("=" * 124)
    print(seg.to_string(index=False))

    print("\n" + "=" * 124)
    print("R33A3 PRIMARY POOLED METHOD METRICS")
    print("=" * 124)
    print(
        pooled[
            [
                "method",
                "AUROC",
                "AUPRC",
                "HARM_PREVALENCE",
                "AUPRC_MINUS_PREVALENCE",
                "AUPRC_LIFT",
            ]
        ].sort_values("AUROC", ascending=False, kind="mergesort").to_string(index=False)
    )

    print("\n" + "=" * 124)
    print("R33A3 SAFETTA CONFIRMATORY BOOTSTRAP")
    print("=" * 124)
    print(
        boot[
            boot["method"] == SAFE_METHOD
        ][
            ["metric", "point", "ci95_low", "ci95_high", "bootstrap_reps_valid"]
        ].to_string(index=False)
    )

    print("\n" + "=" * 124)
    print("R33A3 PAIRED PUBLISHED-BASELINE DELTAS (SafeTTA - comparator)")
    print("=" * 124)
    print(paired.to_string(index=False))

    print("\n" + "=" * 124)
    print("R33A3 EXTERNAL JOINT-SHIFT CONFIRMATION GATE")
    print("=" * 124)
    print("  SafeTTA AUROC                       :", gate["SafeTTA_point"]["AUROC"])
    print("  SafeTTA AUPRC                       :", gate["SafeTTA_point"]["AUPRC"])
    print("  MEMO HARM prevalence                :", gate["SafeTTA_point"]["HARM_prevalence"])
    print("  SafeTTA AUPRC lift                  :", gate["SafeTTA_point"]["AUPRC_lift"])
    print(
        "  AUROC CI95                          :",
        f"[{gate['criterion_1_AUROC']['bootstrap_ci95_low']:.9f}, "
        f"{gate['criterion_1_AUROC']['bootstrap_ci95_high']:.9f}]",
    )
    print(
        "  AUPRC-prevalence CI95               :",
        f"[{gate['criterion_2_AUPRC_minus_prevalence']['bootstrap_ci95_low']:.9f}, "
        f"{gate['criterion_2_AUPRC_minus_prevalence']['bootstrap_ci95_high']:.9f}]",
    )
    print("  Criterion 1 AUROC CI low > 0.5      :", gate["criterion_1_AUROC"]["pass"])
    print(
        "  Criterion 2 AP-prev CI low > 0      :",
        gate["criterion_2_AUPRC_minus_prevalence"]["pass"],
    )
    print("  JOINT CONFIRMATION                   :", gate["joint_confirmation"])
    print("  DECISION                             :", gate["decision"])

    print(
        "\nFINAL STATUS : "
        "PASS_R33A3_POLYPGEN_MEMO_EXTERNAL_JOINT_SHIFT_EVALUATION_COMPLETE"
    )
    print("Model inference             : NO")
    print("TTA rerun                   : NO")
    print("Predictor refit             : NO")
    print("Target calibration          : NO")
    print("Score reversal              : NO")
    print("Final lock SHA256           :", sha256_file(final_path))
    print("Output                      :", out)
    print("NEXT                        : R33A4_PAPER_INTEGRATION_AND_CLAIM_FREEZE")
    print("=" * 124)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
