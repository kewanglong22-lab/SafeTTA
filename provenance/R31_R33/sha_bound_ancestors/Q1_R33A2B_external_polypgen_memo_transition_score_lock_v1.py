#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SafeTTA R33A2B
External PolypGen MEMO + Transition + Frozen SafeTTA Score Lock

Primary direction
-----------------
TRAIN / FIT:
    NeoPolyp, TENT1 + PL-CONF90
    800 physical cases x 9 model states x 2 actions = 14,400 rows

EXTERNAL TARGET:
    PolypGen, MEMO-SEG4-1STEP
    1532 physical cases x 3 DeepLabV3-R50 states = 4,596 model-cases

This execution performs:
  1) exact DeepLab SOURCE + MEMO prediction using the retained R31B1/R31B2
     transaction semantics;
  2) exact SOURCE/MEMO conditioned DINOv2 -> frozen PCA64 transition features;
  3) q_source66 + dSemantic64 construction;
  4) source-only fitting of StandardScaler + class-balanced LogisticRegression
     on NeoPolyp TENT1+PL-CONF90;
  5) target PolypGen MEMO HARM-risk scoring;
  6) immutable PRE-HARM score / prediction / feature locks.

This execution DOES NOT:
  - read/hash/decode PolypGen GT;
  - compute PolypGen SOURCE Dice, MEMO Dice, Delta Dice or HARM;
  - fit/recalibrate on PolypGen;
  - choose score direction from target outcomes;
  - tune MEMO LR on PolypGen;
  - use ActionID in the primary predictor;
  - use external web/network data by default.

The existing R17A CCD / ADIC / MC-dropout target scores are carried forward
unchanged from the R33A1 PRE-GT target manifest for later matched comparison.

Chronology:
    R33A1 protocol lock
      -> R33A2A exact execution-context lock
      -> THIS R33A2B prediction / feature / score lock
      -> R33A3 first new MEMO outcome reveal/evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


VERSION = "2026-09-12-R33A2B-v1"

ROOT = Path(r"F:\MEDSEG_SAFETTA")
CODE = ROOT / "code"

# =============================================================================
# Upstream R33 locks
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

R33A2A_SCRIPT = (
    CODE / "Q1_R33A2A_exact_polypgen_memo_transition_context_bundle_v1_fix2.py"
)
EXPECTED_R33A2A_SCRIPT_SHA256 = (
    "0826addd7c4df2716dc752eadfd6af6e2e6dc01e4ff1e9819a0f64d83a7b5830"
)
R33A2A_DIR = ROOT / "R33A2A_exact_polypgen_memo_transition_context_bundle_v1_fix2"
R33A2A_FINAL = R33A2A_DIR / "R33A2A_FINAL_LOCK.json"
EXPECTED_R33A2A_FINAL_SHA256 = (
    "a5d9f3a3a1947ee0afce8b4c2b2147232f9e9b001dfc5627b8dbffd14ba3a80a"
)
R33A2A_BUNDLE = R33A2A_DIR / "R33A2A_EXACT_EXECUTION_CONTEXT_BUNDLE.txt"
EXPECTED_R33A2A_BUNDLE_SHA256 = (
    "993f5628fe021a99f3e873738c50f7710f2dea4813e337831d67793f22b1a0c5"
)

# =============================================================================
# Exact historical implementation bindings
# =============================================================================

R31B2_SCRIPT = CODE / "Q1_R31B2_800case_memo_prediction_and_semantic_lock_v1.py"
EXPECTED_R31B2_SCRIPT_SHA256 = (
    "fa4802f62bb510b953b216f2a3211df8c556dee943f612493ae75463870bbf5b"
)

R31B1_SCRIPT = CODE / "Q1_R31B1_memo_seg4_action_design_and_lr_freeze_v1.py"
EXPECTED_R31B1_SCRIPT_SHA256 = (
    "3391fe4d7161df488f372379a65d72c0dc0cc4e1e774e02e90d68c9ad17f911b"
)

R22A1_SCRIPT = (
    CODE
    / "Q1_R22A1_gtfree_candidate_conditioned_transition_feature_extraction_v1_fix1.py"
)
EXPECTED_R22A1_SCRIPT_SHA256 = (
    "1f0a925721485226d7fd58f3e0968bc7e10213a8f4561023f8886b98fde57ba6"
)

R05D3_SCRIPT = CODE / "Q1_R05D3_neopolyp_source_a1_prediction_lock_v1_fix1.py"
EXPECTED_R05D3_SCRIPT_SHA256 = (
    "df22d6f3054257743d1f12d9e757323ea9f4a052b7642eb4ab610da46ff21251"
)

R30A0_SCRIPT = CODE / "Q1_R30A0_paired_action_crc_protocol_and_asset_audit_v1_fix1.py"
EXPECTED_R30A0_SCRIPT_SHA256 = (
    "9dc5d3a13f3d64608c4c599f811f929f731d7fac05e2aaa66eb344e24ace2298"
)

R31B2_DIR = ROOT / "R31B2_800case_memo_prediction_and_semantic_lock_v1"
R31B2_FINAL = R31B2_DIR / "R31B2_FINAL_PRE_GT_LOCK.json"
EXPECTED_R31B2_FINAL_SHA256 = (
    "f2d5c7b6edbffe596d40cc839e9ec437349f8e1bc127a3c7ab5df0c345a120e8"
)

# =============================================================================
# Source fitting assets
# =============================================================================

R31B3_DIR = ROOT / "R31B3_first_800case_gt_reveal_and_three_action_loao_v1"
R31B3_TABLE = R31B3_DIR / "R31B3_THREE_ACTION_MODELCASE_TABLE.csv"

# =============================================================================
# Frozen protocol
# =============================================================================

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
EXPECTED_TARGET_ROWS = TARGET_CASES * TARGET_STATES

THIRD_ACTION = "MEMO-SEG4-1STEP"
SELECTED_LR = 1e-5
IMAGE_SIZE = 352
PACKED_BYTES = (IMAGE_SIZE * IMAGE_SIZE + 7) // 8

QSOURCE = [f"qsrc_{i:02d}" for i in range(66)]
DSEM = [f"dq_{i:02d}" for i in range(2, 66)]
FEATURE_COLUMNS = QSOURCE + DSEM
assert len(QSOURCE) == 66
assert len(DSEM) == 64
assert len(FEATURE_COLUMNS) == 130

HARM_LABEL_COLUMN = "r31b3_harm_label"
HARM_THRESHOLD = -0.02

LR_C = 1.0
LR_CLASS_WEIGHT = "balanced"
LR_SOLVER = "lbfgs"
LR_MAX_ITER = 5000
LR_RANDOM_STATE = 0

DEFAULT_OUT = ROOT / "R33A2B_external_polypgen_memo_transition_score_lock_v1"

FORBIDDEN_TARGET_COLUMNS = (
    "harm_label",
    "delta_dice",
    "dice_delta",
    "source_dice",
    "memo_dice",
    "adapted_dice",
    "ground_truth",
    "gt_path",
    "mask_path",
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


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def assert_no_target_outcomes(df: pd.DataFrame, label: str):
    bad = []
    for c in df.columns:
        low = str(c).lower()
        if any(tok in low for tok in FORBIDDEN_TARGET_COLUMNS):
            bad.append(str(c))
    if bad:
        raise RuntimeError(
            f"{label} contains forbidden target outcome/GT columns={bad}"
        )


def state_cache_paths(out: Path, state_index: int) -> Tuple[Path, Path]:
    npz = out / f"state_{state_index:02d}_polypgen_source_memo_pre_harm.npz"
    lock = out / f"state_{state_index:02d}_polypgen_source_memo_pre_harm.lock.json"
    return npz, lock


# =============================================================================
# Immutable upstream verification
# =============================================================================


def verify_upstream() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    require_sha(R33A1_FINAL, EXPECTED_R33A1_FINAL_SHA256, "R33A1 final")
    require_sha(R33A1_PROTOCOL, EXPECTED_R33A1_PROTOCOL_SHA256, "R33A1 protocol")
    require_sha(R33A2A_SCRIPT, EXPECTED_R33A2A_SCRIPT_SHA256, "R33A2A fix2 script")
    require_sha(R33A2A_FINAL, EXPECTED_R33A2A_FINAL_SHA256, "R33A2A final")
    require_sha(R33A2A_BUNDLE, EXPECTED_R33A2A_BUNDLE_SHA256, "R33A2A bundle")

    require_sha(R31B2_SCRIPT, EXPECTED_R31B2_SCRIPT_SHA256, "R31B2 script")
    require_sha(R31B1_SCRIPT, EXPECTED_R31B1_SCRIPT_SHA256, "R31B1 exact source")
    require_sha(R22A1_SCRIPT, EXPECTED_R22A1_SCRIPT_SHA256, "R22A1 fix1")
    require_sha(R05D3_SCRIPT, EXPECTED_R05D3_SCRIPT_SHA256, "R05D3 fix1")
    require_sha(R30A0_SCRIPT, EXPECTED_R30A0_SCRIPT_SHA256, "R30A0 schema")
    require_sha(R31B2_FINAL, EXPECTED_R31B2_FINAL_SHA256, "R31B2 authoritative final")

    a1 = json.loads(R33A1_FINAL.read_text(encoding="utf-8"))
    if a1.get("status") != "PASS_R33A1_EXTERNAL_JOINT_SHIFT_PROTOCOL_LOCK_COMPLETE":
        raise RuntimeError(f"R33A1 status changed: {a1.get('status')}")
    if bool(a1.get("GT_HARM_evaluation", True)):
        raise RuntimeError("R33A1 GT/HARM boundary changed.")
    if bool(a1.get("target_calibration", True)):
        raise RuntimeError("R33A1 calibration boundary changed.")

    a2a = json.loads(R33A2A_FINAL.read_text(encoding="utf-8"))
    if a2a.get("status") != "PASS_R33A2A_EXACT_EXECUTION_CONTEXT_BINDING_COMPLETE":
        raise RuntimeError(f"R33A2A status changed: {a2a.get('status')}")
    if a2a.get("decision") != "READY_FOR_R33A2B_EXTERNAL_MEMO_TRANSITION_EXECUTION":
        raise RuntimeError(f"R33A2A decision changed: {a2a.get('decision')}")

    protocol = json.loads(R33A1_PROTOCOL.read_text(encoding="utf-8"))
    if protocol["source_training"]["feature_set"] != "QSOURCE66 + DSEM64":
        raise RuntimeError("R33A1 feature set changed.")
    if protocol["target_action"]["name"] != THIRD_ACTION:
        raise RuntimeError("R33A1 target action changed.")
    if float(protocol["target_action"]["learning_rate"]) != SELECTED_LR:
        raise RuntimeError("R33A1 MEMO LR changed.")
    if bool(protocol["target_action"]["hyperparameter_search_on_PolypGen"]):
        raise RuntimeError("R33A1 allowed target MEMO tuning unexpectedly.")

    return a1, protocol


# =============================================================================
# Target / source data loading
# =============================================================================


def load_target_manifest(r33a1_final: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not R33A1_TARGET.is_file():
        raise FileNotFoundError(R33A1_TARGET)
    if sha256_file(R33A1_TARGET) != str(r33a1_final["target_manifest_sha256"]):
        raise RuntimeError("R33A1 target manifest SHA changed.")

    d = pd.read_csv(
        R33A1_TARGET,
        dtype={
            "sample_id": str,
            "model_family": str,
            "model_state_id": str,
        },
        low_memory=False,
    )
    assert_no_target_outcomes(d, "R33A1 target manifest")

    required = {
        "sample_id",
        "image_path",
        "image_raw_sha256",
        "model_family",
        "model_state_id",
        "training_seed",
        "checkpoint_sha256",
        "ccd_risk",
        "adic_harm_risk",
        "mc_predictive_entropy_risk",
    }
    missing = sorted(required.difference(d.columns))
    if missing:
        raise RuntimeError(f"Target manifest missing={missing}")

    if len(d) != EXPECTED_TARGET_ROWS:
        raise RuntimeError(f"Target rows={len(d)} expected={EXPECTED_TARGET_ROWS}")
    if d["sample_id"].nunique() != TARGET_CASES:
        raise RuntimeError("Target physical case count drift.")
    if d["model_state_id"].nunique() != TARGET_STATES:
        raise RuntimeError("Target state count drift.")
    if set(d["model_family"].astype(str)) != {TARGET_FAMILY}:
        raise RuntimeError("Target family drift.")
    if set(d["model_state_id"].astype(str)) != set(TARGET_STATE_IDS):
        raise RuntimeError("Target state IDs drift.")
    if d.duplicated(["sample_id", "model_state_id"]).any():
        raise RuntimeError("Duplicate target model-case.")

    # Authoritative state-major/case-major order for this new execution.
    states = (
        d[
            [
                "model_state_id",
                "model_family",
                "training_seed",
                "checkpoint_sha256",
            ]
        ]
        .drop_duplicates()
        .sort_values(["training_seed"], kind="mergesort")
        .reset_index(drop=True)
    )
    if len(states) != TARGET_STATES:
        raise RuntimeError("State panel rows drift.")

    # All states must share the exact same physical-image identity.
    rgb_cols = ["sample_id", "image_path", "image_raw_sha256"]
    base = None
    for _, s in states.iterrows():
        g = (
            d[d["model_state_id"].astype(str) == str(s["model_state_id"])][rgb_cols]
            .sort_values("sample_id", kind="mergesort")
            .reset_index(drop=True)
        )
        if len(g) != TARGET_CASES or g["sample_id"].nunique() != TARGET_CASES:
            raise RuntimeError(f"{s['model_state_id']}: target image mapping drift.")
        if base is None:
            base = g
        elif not base.astype(str).equals(g.astype(str)):
            raise RuntimeError("Target RGB mapping differs across model states.")

    assert base is not None
    rgb = base.copy()

    # Canonicalize target d to same state-major / sample-major order.
    pieces = []
    for _, s in states.iterrows():
        g = (
            d[d["model_state_id"].astype(str) == str(s["model_state_id"])]
            .sort_values("sample_id", kind="mergesort")
            .reset_index(drop=True)
        )
        pieces.append(g)
    d = pd.concat(pieces, ignore_index=True)
    if len(d) != EXPECTED_TARGET_ROWS:
        raise RuntimeError("Canonical target row count drift.")

    return d, states


def load_source_training(protocol: Dict[str, Any]) -> pd.DataFrame:
    if not R31B3_TABLE.is_file():
        raise FileNotFoundError(R31B3_TABLE)

    expected_sha = str(protocol["source_training"]["source_table_sha256"])
    actual_sha = sha256_file(R31B3_TABLE)
    if actual_sha != expected_sha:
        raise RuntimeError(
            f"R31B3 source table SHA drift expected={expected_sha} got={actual_sha}"
        )

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
        HARM_LABEL_COLUMN,
        *FEATURE_COLUMNS,
    }
    missing = sorted(required.difference(d.columns))
    if missing:
        raise RuntimeError(f"R31B3 source table missing={missing}")

    d = d[d["action"].astype(str).isin(SOURCE_ACTIONS)].copy()

    if len(d) != EXPECTED_SOURCE_ROWS:
        raise RuntimeError(
            f"Source TENT+PL rows={len(d)} expected={EXPECTED_SOURCE_ROWS}"
        )
    if d["sample_id"].nunique() != SOURCE_CASES:
        raise RuntimeError("Source physical-case count drift.")
    if d["model_state_id"].nunique() != SOURCE_STATES:
        raise RuntimeError("Source model-state count drift.")
    if set(d["action"].astype(str)) != set(SOURCE_ACTIONS):
        raise RuntimeError("Source action set drift.")
    if d.duplicated(["sample_id", "model_state_id", "action"]).any():
        raise RuntimeError("Duplicate source fitting row.")

    X = d[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    y = d[HARM_LABEL_COLUMN].to_numpy(dtype=int)
    if not np.isfinite(X).all():
        raise RuntimeError("Source fit features contain non-finite values.")
    if set(np.unique(y)) != {0, 1}:
        raise RuntimeError("Source HARM label is not binary with both classes.")

    return d


# =============================================================================
# Exact model / MEMO transaction
# =============================================================================


def validate_exact_runtime_modules(r31b1, r31b2, r05, r30a0):
    required_b1 = [
        "snapshot_model",
        "restore_model",
        "augment_batch",
        "inverse_align_probs",
        "binary_entropy_mean",
        "action_seed",
        "set_seed",
    ]
    missing = [x for x in required_b1 if not hasattr(r31b1, x)]
    if missing:
        raise RuntimeError(f"R31B1 exact functions missing={missing}")

    required_b2 = [
        "pack_mask",
        "extract_semantics",
        "semantic_paths",
        "semantic_lock_valid",
    ]
    missing = [x for x in required_b2 if not hasattr(r31b2, x)]
    if missing:
        raise RuntimeError(f"R31B2 exact functions missing={missing}")

    q = list(getattr(r30a0, "QSOURCE", []))
    ds = list(getattr(r30a0, "DSEM", []))
    if q != QSOURCE:
        raise RuntimeError("R30A0 QSOURCE schema drift.")
    if ds != DSEM:
        raise RuntimeError("R30A0 DSEM schema drift.")

    if int(getattr(r31b2, "IMAGE_SIZE")) != IMAGE_SIZE:
        raise RuntimeError("R31B2 image size drift.")
    if float(getattr(r31b2, "SELECTED_LR")) != SELECTED_LR:
        raise RuntimeError("R31B2 selected MEMO LR drift.")


def build_deeplab_runtime(
    *,
    r31b1,
    r05,
    state: pd.Series,
    device: torch.device,
):
    """
    Exact DeepLab branch reconstructed from the SHA-bound R31B1 source:
      helper = R05D3.DEEPLAB_HELPER
      helper.seed_everything(20260817)
      helper.load_model(training, seed, device)
      training.deeplab_logits
      sigmoid foreground probability
      R05D3.logit_to_mask
      all model parameters trainable for MEMO
      source params + buffers snapshotted.
    """
    seed = int(state["training_seed"])

    helper = import_module(
        Path(r05.DEEPLAB_HELPER),
        f"r33a2b_deeplab_{seed}",
    )
    training = helper.import_training_helper()

    helper.seed_everything(20260817)
    model = helper.load_model(training, seed, device)
    model.to(device)

    def to_tensor(native):
        return helper.image_to_model_tensor(training, native).to(
            device,
            non_blocking=True,
        )

    def foreground_prob(x):
        z = training.deeplab_logits(model, x)
        if z.ndim != 4 or z.shape[1] != 1:
            raise RuntimeError(f"DeepLab logit shape={tuple(z.shape)}")
        return torch.sigmoid(z)

    def final_mask(x):
        z = training.deeplab_logits(model, x)
        a = z[0, 0].detach().float().cpu().numpy()
        return np.asarray(r05.logit_to_mask(a), dtype=np.uint8)

    for p in model.parameters():
        p.requires_grad_(True)
    if not list(model.parameters()):
        raise RuntimeError("DeepLab has no model parameters.")

    source_params, source_buffers = r31b1.snapshot_model(model)

    return {
        "model": model,
        "to_tensor": to_tensor,
        "foreground_prob": foreground_prob,
        "final_mask": final_mask,
        "source_params": source_params,
        "source_buffers": source_buffers,
        "helper": helper,
        "training": training,
    }


def gradient_l2(params: Sequence[Any]) -> float:
    total = 0.0
    for p in params:
        if p.grad is None:
            continue
        g = p.grad.detach().float()
        total += float(torch.sum(g * g).item())
    return float(math.sqrt(max(total, 0.0)))


def validate_state_cache(
    *,
    out: Path,
    state_index: int,
    state: pd.Series,
    rgb: pd.DataFrame,
) -> bool:
    npz_path, lock_path = state_cache_paths(out, state_index)
    if not npz_path.is_file() or not lock_path.is_file():
        return False

    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        if lock.get("status") != "PASS_R33A2B_STATE_SOURCE_MEMO_PRE_HARM_LOCK":
            return False
        if lock.get("model_state_id") != str(state["model_state_id"]):
            return False
        if int(lock.get("training_seed")) != int(state["training_seed"]):
            return False
        if lock.get("checkpoint_sha256") != str(state["checkpoint_sha256"]):
            return False
        if int(lock.get("physical_cases")) != TARGET_CASES:
            return False
        if lock.get("rgb_manifest_sha256") != sha256_file(
            out / "R31B2_800CASE_RGB_MANIFEST_PRE_GT.csv"
        ):
            return False
        if lock.get("prediction_npz_sha256") != sha256_file(npz_path):
            return False

        with np.load(npz_path, allow_pickle=False) as z:
            required = {
                "source_masks_packed",
                "memo_masks_packed",
                "memo_loss_preupdate",
                "grad_l2",
                "changed_pixels",
                "action_seeds",
            }
            if not required.issubset(set(z.files)):
                return False
            source = np.asarray(z["source_masks_packed"], dtype=np.uint8)
            memo = np.asarray(z["memo_masks_packed"], dtype=np.uint8)
            if source.shape != (TARGET_CASES, PACKED_BYTES):
                return False
            if memo.shape != source.shape:
                return False
            for key in [
                "memo_loss_preupdate",
                "grad_l2",
                "changed_pixels",
                "action_seeds",
            ]:
                if np.asarray(z[key]).shape != (TARGET_CASES,):
                    return False
        return True
    except Exception:
        return False


def run_state_memo(
    *,
    out: Path,
    state_index: int,
    state: pd.Series,
    rgb: pd.DataFrame,
    r31b1,
    r31b2,
    r05,
    device: torch.device,
    bitorder: str,
    resume: bool,
) -> Tuple[Path, Dict[str, Any]]:
    npz_path, lock_path = state_cache_paths(out, state_index)

    if resume and validate_state_cache(
        out=out,
        state_index=state_index,
        state=state,
        rgb=rgb,
    ):
        print(f"[RESUME] state {state_index:02d} {state['model_state_id']} PASS")
        return npz_path, json.loads(lock_path.read_text(encoding="utf-8"))

    if npz_path.exists() or lock_path.exists():
        raise RuntimeError(
            f"State cache already exists but is not valid/resumable: "
            f"{state['model_state_id']}"
        )

    # Check checkpoint identity against exact R31B1 frozen manifest constants.
    expected_ckpt = (
        getattr(r31b1, "EXPECTED_CHECKPOINT_SHAS")
        [TARGET_FAMILY]
        [int(state["training_seed"])]
    )
    if str(expected_ckpt) != str(state["checkpoint_sha256"]):
        raise RuntimeError(
            f"Checkpoint identity drift {state['model_state_id']}: "
            f"R31B1={expected_ckpt} target={state['checkpoint_sha256']}"
        )

    runtime = build_deeplab_runtime(
        r31b1=r31b1,
        r05=r05,
        state=state,
        device=device,
    )

    model = runtime["model"]
    to_tensor = runtime["to_tensor"]
    foreground_prob = runtime["foreground_prob"]
    final_mask = runtime["final_mask"]
    source_params = runtime["source_params"]
    source_buffers = runtime["source_buffers"]

    source_packed = np.empty((TARGET_CASES, PACKED_BYTES), dtype=np.uint8)
    memo_packed = np.empty_like(source_packed)
    memo_loss = np.empty(TARGET_CASES, dtype=np.float32)
    grad_norm = np.empty(TARGET_CASES, dtype=np.float32)
    changed_pixels = np.empty(TARGET_CASES, dtype=np.int32)
    action_seeds = np.empty(TARGET_CASES, dtype=np.int64)

    pbar = tqdm(
        range(TARGET_CASES),
        total=TARGET_CASES,
        desc=f"R33A2B MEMO seed {int(state['training_seed'])}",
        unit="img",
        dynamic_ncols=True,
    )

    for i in pbar:
        row = rgb.iloc[i]
        image_path = Path(str(row["image_path"]))
        if not image_path.is_file():
            raise FileNotFoundError(image_path)

        with Image.open(image_path) as im:
            native = im.convert("RGB")
        x = to_tensor(native)

        # Exact frozen SOURCE candidate.
        r31b1.restore_model(model, source_params, source_buffers)
        model.eval()
        with torch.no_grad():
            source_mask = np.asarray(final_mask(x), dtype=np.uint8)

        # Exact frozen MEMO transaction.
        r31b1.restore_model(model, source_params, source_buffers)
        seed_txn = int(
            r31b1.action_seed(
                str(state["model_state_id"]),
                str(row["sample_id"]),
            )
        )
        r31b1.set_seed(seed_txn)

        model.train()
        params = list(model.parameters())
        for p in params:
            p.requires_grad_(True)

        # R33A1 froze Adam, lr=1e-5, weight_decay=0, one step.
        opt = torch.optim.Adam(
            params,
            lr=SELECTED_LR,
            weight_decay=0.0,
        )
        opt.zero_grad(set_to_none=True)

        x4 = r31b1.augment_batch(x, torch)
        prob4 = foreground_prob(x4)
        aligned = r31b1.inverse_align_probs(prob4, torch)
        marginal = aligned.mean(dim=0, keepdim=True)
        loss = r31b1.binary_entropy_mean(marginal, torch)

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite MEMO loss "
                f"state={state['model_state_id']} sample={row['sample_id']}"
            )

        loss.backward()
        gnorm = gradient_l2(params)
        if not math.isfinite(gnorm):
            raise RuntimeError("Non-finite MEMO gradient norm.")

        opt.step()
        opt.zero_grad(set_to_none=True)

        model.eval()
        with torch.no_grad():
            memo_mask = np.asarray(final_mask(x), dtype=np.uint8)

        source_packed[i] = r31b2.pack_mask(source_mask, bitorder)
        memo_packed[i] = r31b2.pack_mask(memo_mask, bitorder)
        memo_loss[i] = float(loss.detach().cpu())
        grad_norm[i] = float(gnorm)
        changed_pixels[i] = int(np.count_nonzero(source_mask != memo_mask))
        action_seeds[i] = seed_txn

        pbar.set_postfix(
            chg=int(changed_pixels[i]),
            loss=f"{memo_loss[i]:.4f}",
        )

        # Exact transactional restoration after each case.
        r31b1.restore_model(model, source_params, source_buffers)

        del (
            opt,
            x4,
            prob4,
            aligned,
            marginal,
            loss,
            source_mask,
            memo_mask,
            x,
        )

    pbar.close()

    tmp_npz = npz_path.with_name(npz_path.name + ".tmp.npz")
    np.savez_compressed(
        tmp_npz,
        source_masks_packed=source_packed,
        memo_masks_packed=memo_packed,
        memo_loss_preupdate=memo_loss,
        grad_l2=grad_norm,
        changed_pixels=changed_pixels,
        action_seeds=action_seeds,
    )
    os.replace(tmp_npz, npz_path)

    lock = {
        "status": "PASS_R33A2B_STATE_SOURCE_MEMO_PRE_HARM_LOCK",
        "version": VERSION,
        "model_state_id": str(state["model_state_id"]),
        "model_family": str(state["model_family"]),
        "training_seed": int(state["training_seed"]),
        "checkpoint_sha256": str(state["checkpoint_sha256"]),
        "physical_cases": TARGET_CASES,
        "action": THIRD_ACTION,
        "selected_learning_rate": SELECTED_LR,
        "optimizer": "Adam",
        "weight_decay": 0.0,
        "update_steps": 1,
        "transforms": ["IDENTITY", "HFLIP", "VFLIP", "HVFLIP"],
        "adapted_parameters": "all model parameters",
        "prediction_npz": str(npz_path),
        "prediction_npz_sha256": sha256_file(npz_path),
        "rgb_manifest_sha256": sha256_file(
            out / "R31B2_800CASE_RGB_MANIFEST_PRE_GT.csv"
        ),
        "changed_pixels_mean": float(changed_pixels.mean()),
        "changed_pixels_nonzero_fraction": float(np.mean(changed_pixels > 0)),
        "memo_loss_mean": float(memo_loss.mean()),
        "grad_l2_mean": float(grad_norm.mean()),
        "GT_read_hash_decode": [False, False, False],
        "target_calibration": False,
    }
    atomic_json(lock_path, lock)

    if not validate_state_cache(
        out=out,
        state_index=state_index,
        state=state,
        rgb=rgb,
    ):
        raise RuntimeError(f"Fresh state cache validation failed: {state['model_state_id']}")

    del runtime, model, source_params, source_buffers
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return npz_path, lock


# =============================================================================
# Prediction inventory + exact R31B2 semantic replay
# =============================================================================


def build_prediction_inventory(
    *,
    out: Path,
    states: pd.DataFrame,
    npz_paths: Sequence[Path],
    state_locks: Sequence[Dict[str, Any]],
) -> pd.DataFrame:
    rows = []
    for idx, state in states.iterrows():
        npz_path = Path(npz_paths[idx])
        lock_path = state_cache_paths(out, idx)[1]
        rows.append({
            "state_no": int(idx + 1),
            "state_index": int(idx),
            "model_state_id": str(state["model_state_id"]),
            "model_family": str(state["model_family"]),
            "training_seed": int(state["training_seed"]),
            "checkpoint_sha256": str(state["checkpoint_sha256"]),
            "prediction_npz": str(npz_path),
            "prediction_npz_sha256": sha256_file(npz_path),
            "state_lock": str(lock_path),
            "state_lock_sha256": sha256_file(lock_path),
        })
    inv = pd.DataFrame(rows)
    if len(inv) != TARGET_STATES:
        raise RuntimeError("Prediction inventory state count drift.")
    return inv


def patch_r31b2_for_external_target(r31b2):
    """
    Reuse the exact retained R31B2 semantic implementation while adapting only
    frozen dimensional constants from NeoPolyp 800x9 to PolypGen 1532x3.
    No formula, PCA, DINO, conditioning or schema change.
    """
    r31b2.EXPECTED_EVAL_CASES = TARGET_CASES
    r31b2.EXPECTED_STATES = TARGET_STATES
    r31b2.EXPECTED_MODEL_CASES = EXPECTED_TARGET_ROWS
    r31b2.SELECTED_LR = SELECTED_LR
    r31b2.THIRD_ACTION = THIRD_ACTION
    r31b2.IMAGE_SIZE = IMAGE_SIZE
    r31b2.PACKED_BYTES = PACKED_BYTES


def run_exact_semantic_lock(
    *,
    out: Path,
    rgb: pd.DataFrame,
    states: pd.DataFrame,
    npz_paths: Sequence[Path],
    r31b2,
    bitorder: str,
    allow_download: bool,
    device_name: str,
    resume: bool,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, Dict[str, Any]]:
    patch_r31b2_for_external_target(r31b2)

    # R31B2 exact function binds hashes to these historical filenames.
    rgb_lock_path = out / "R31B2_800CASE_RGB_MANIFEST_PRE_GT.csv"
    pred_inv_path = out / "R31B2_STATE_PREDICTION_INVENTORY_PRE_GT.csv"

    if not rgb_lock_path.is_file() or not pred_inv_path.is_file():
        raise RuntimeError("Exact R31B2 semantic hash-binding sidecars missing.")

    r31b2.extract_semantics(
        out=out,
        rgb=rgb,
        panel=states,
        npz_paths=npz_paths,
        bitorder=bitorder,
        allow_download=allow_download,
        device_name=device_name,
        resume=resume,
    )

    sp = r31b2.semantic_paths(out)
    if not r31b2.semantic_lock_valid(
        out,
        sha256_file(rgb_lock_path),
        sha256_file(pred_inv_path),
    ):
        raise RuntimeError("Exact R31B2 semantic lock validation failed.")

    q = np.load(sp["q"], allow_pickle=False)
    dm = np.load(sp["memo"], allow_pickle=False)
    table = pd.read_csv(sp["table"], low_memory=False)
    lock = json.loads(sp["lock"].read_text(encoding="utf-8"))

    if q.shape != (EXPECTED_TARGET_ROWS, 66):
        raise RuntimeError(f"Target q_source66 shape={q.shape}")
    if dm.shape != (EXPECTED_TARGET_ROWS, 64):
        raise RuntimeError(f"Target dSemantic64 shape={dm.shape}")
    if q.dtype != np.float64 or dm.dtype != np.float64:
        raise RuntimeError(f"Target feature dtype drift q={q.dtype} dm={dm.dtype}")
    if not np.isfinite(q).all() or not np.isfinite(dm).all():
        raise RuntimeError("Target semantic features contain non-finite values.")
    if len(table) != EXPECTED_TARGET_ROWS:
        raise RuntimeError("R31B2 semantic table row count drift.")

    return q, dm, table, lock


# =============================================================================
# Source-only SafeTTA fit + external target scoring
# =============================================================================


def fit_source_predictor(
    source: pd.DataFrame,
) -> Tuple[StandardScaler, LogisticRegression, Dict[str, Any]]:
    X = source[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    y = source[HARM_LABEL_COLUMN].to_numpy(dtype=int)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    clf = LogisticRegression(
        C=LR_C,
        class_weight=LR_CLASS_WEIGHT,
        solver=LR_SOLVER,
        max_iter=LR_MAX_ITER,
        random_state=LR_RANDOM_STATE,
    )
    clf.fit(Xs, y)

    if list(clf.classes_) != [0, 1]:
        raise RuntimeError(f"Unexpected LogisticRegression classes={clf.classes_}")
    if int(clf.n_features_in_) != len(FEATURE_COLUMNS):
        raise RuntimeError("Predictor feature count drift.")

    info = {
        "training_rows": int(len(source)),
        "physical_cases": int(source["sample_id"].nunique()),
        "model_states": int(source["model_state_id"].nunique()),
        "actions": sorted(source["action"].astype(str).unique().tolist()),
        "harm_count": int(y.sum()),
        "harm_prevalence": float(y.mean()),
        "feature_count": int(X.shape[1]),
        "scaler_fit_domain": "NeoPolyp source rows only",
        "classifier": "LogisticRegression",
        "C": LR_C,
        "class_weight": LR_CLASS_WEIGHT,
        "solver": LR_SOLVER,
        "max_iter": LR_MAX_ITER,
        "random_state": LR_RANDOM_STATE,
        "n_iter": [int(x) for x in np.asarray(clf.n_iter_).reshape(-1)],
        "target_rows_used_for_fit": 0,
        "target_labels_used": False,
    }
    return scaler, clf, info


def score_target(
    *,
    target_manifest: pd.DataFrame,
    semantic_table: pd.DataFrame,
    q: np.ndarray,
    dm: np.ndarray,
    scaler: StandardScaler,
    clf: LogisticRegression,
) -> pd.DataFrame:
    # Exact R31B2 semantic ordering is state-major, case-major.
    keys = ["sample_id", "model_state_id"]

    sem_meta = semantic_table[
        ["sample_id", "model_state_id", "model_family", "training_seed", "action"]
    ].copy()

    if sem_meta.duplicated(keys).any():
        raise RuntimeError("Duplicate semantic model-case key.")
    if set(sem_meta["action"].astype(str)) != {THIRD_ACTION}:
        raise RuntimeError("Target semantic action drift.")

    # Verify semantic metadata order matches target manifest exactly.
    target_keys = target_manifest[keys].astype(str).reset_index(drop=True)
    sem_keys = sem_meta[keys].astype(str).reset_index(drop=True)
    if not target_keys.equals(sem_keys):
        raise RuntimeError(
            "Target manifest and semantic feature row order/key identity differ."
        )

    X = np.concatenate([q, dm], axis=1)
    if X.shape != (EXPECTED_TARGET_ROWS, 130):
        raise RuntimeError(f"Target SafeTTA X shape={X.shape}")

    Xs = scaler.transform(X)
    score = clf.predict_proba(Xs)[:, 1]
    if not np.isfinite(score).all():
        raise RuntimeError("Non-finite target SafeTTA scores.")

    out = target_manifest[
        [
            "sample_id",
            "image_path",
            "image_raw_sha256",
            "model_family",
            "model_state_id",
            "training_seed",
            "checkpoint_sha256",
            "ccd_risk",
            "adic_harm_risk",
            "mc_predictive_entropy_risk",
        ]
    ].copy()
    out["action"] = THIRD_ACTION
    out["safettta_external_harm_risk"] = score.astype(np.float64)

    # Useful GT-free action diagnostics are merged later from state caches.
    assert_no_target_outcomes(out, "R33A2B target score table")
    return out


def save_predictor_numeric_lock(
    *,
    out: Path,
    scaler: StandardScaler,
    clf: LogisticRegression,
    source_info: Dict[str, Any],
) -> Path:
    path = out / "R33A2B_NEOPOLYP_TENT_PL_FROZEN_PREDICTOR_NUMERIC_LOCK.npz"
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(
        tmp,
        feature_names=np.asarray(FEATURE_COLUMNS, dtype="U32"),
        scaler_mean=np.asarray(scaler.mean_, dtype=np.float64),
        scaler_scale=np.asarray(scaler.scale_, dtype=np.float64),
        scaler_var=np.asarray(scaler.var_, dtype=np.float64),
        lr_coef=np.asarray(clf.coef_, dtype=np.float64),
        lr_intercept=np.asarray(clf.intercept_, dtype=np.float64),
        lr_classes=np.asarray(clf.classes_, dtype=np.int64),
        lr_n_iter=np.asarray(clf.n_iter_, dtype=np.int64),
    )
    os.replace(tmp, path)

    lock = {
        "status": "PASS_R33A2B_SOURCE_ONLY_SAFETTA_PREDICTOR_LOCK",
        "version": VERSION,
        "source_training": source_info,
        "feature_names": FEATURE_COLUMNS,
        "numeric_npz": str(path),
        "numeric_npz_sha256": sha256_file(path),
        "target_rows_used_for_fit": 0,
        "target_labels_used": False,
        "target_calibration": False,
        "ActionID_used": False,
    }
    atomic_json(
        out / "R33A2B_NEOPOLYP_TENT_PL_FROZEN_PREDICTOR_LOCK.json",
        lock,
    )
    return path


def collect_action_diagnostics(
    out: Path,
    states: pd.DataFrame,
    rgb: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for sidx, state in states.iterrows():
        npz_path, _ = state_cache_paths(out, int(sidx))
        with np.load(npz_path, allow_pickle=False) as z:
            loss = np.asarray(z["memo_loss_preupdate"], dtype=np.float64)
            grad = np.asarray(z["grad_l2"], dtype=np.float64)
            chg = np.asarray(z["changed_pixels"], dtype=np.int64)
            seeds = np.asarray(z["action_seeds"], dtype=np.int64)

        for i, r in rgb.iterrows():
            rows.append({
                "sample_id": str(r["sample_id"]),
                "model_state_id": str(state["model_state_id"]),
                "memo_loss_preupdate": float(loss[i]),
                "memo_grad_l2": float(grad[i]),
                "memo_changed_pixels": int(chg[i]),
                "memo_action_seed": int(seeds[i]),
            })

    d = pd.DataFrame(rows)
    if len(d) != EXPECTED_TARGET_ROWS:
        raise RuntimeError("Action diagnostic row count drift.")
    if d.duplicated(["sample_id", "model_state_id"]).any():
        raise RuntimeError("Duplicate action diagnostic key.")
    return d


# =============================================================================
# Main
# =============================================================================


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--device",
        default="cuda",
        help="torch device used for DeepLab and DINO; default=cuda",
    )
    ap.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow DINO dependency download. Default is local-only.",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Resume only from SHA/lock-valid completed state/semantic caches.",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUT,
    )
    args = ap.parse_args()

    print("=" * 124)
    print("SafeTTA R33A2B External PolypGen MEMO + Transition + Score Lock")
    print("Version                     :", VERSION)
    print("Source fit                  : NeoPolyp / TENT1 + PL-CONF90")
    print("External target             : PolypGen / MEMO-SEG4-1STEP")
    print("Target cases                :", TARGET_CASES)
    print("Target DeepLab states       :", TARGET_STATES)
    print("Target model-cases          :", EXPECTED_TARGET_ROWS)
    print("Feature                     : QSOURCE66 + DSEM64")
    print("ActionID                    : NO")
    print("Target calibration          : NO")
    print("PolypGen GT read/hash/decode: NO / NO / NO")
    print("External web access         :", "ALLOWED FOR DINO ONLY" if args.allow_download else "NO")
    print("=" * 124)

    r33a1_final, protocol = verify_upstream()

    # Import exact SHA-bound historical implementation.
    r31b2 = import_module(R31B2_SCRIPT, "r33a2b_r31b2_exact")
    r31b1 = import_module(R31B1_SCRIPT, "r33a2b_r31b1_exact")
    r05 = import_module(R05D3_SCRIPT, "r33a2b_r05d3_exact")
    r30a0 = import_module(R30A0_SCRIPT, "r33a2b_r30a0_exact")
    validate_exact_runtime_modules(r31b1, r31b2, r05, r30a0)

    target_manifest, states = load_target_manifest(r33a1_final)
    source = load_source_training(protocol)

    # State-major / sample-major physical RGB order.
    rgb = (
        target_manifest[
            ["sample_id", "image_path", "image_raw_sha256"]
        ]
        .drop_duplicates()
        .sort_values("sample_id", kind="mergesort")
        .reset_index(drop=True)
    )
    if len(rgb) != TARGET_CASES:
        raise RuntimeError("Target RGB manifest count drift.")

    # Exact checkpoint identity against SHA-bound R31B1 constants.
    for _, state in states.iterrows():
        expected = (
            getattr(r31b1, "EXPECTED_CHECKPOINT_SHAS")
            [TARGET_FAMILY][int(state["training_seed"])]
        )
        if str(expected) != str(state["checkpoint_sha256"]):
            raise RuntimeError(
                f"Checkpoint mismatch {state['model_state_id']} "
                f"R31B1={expected} target={state['checkpoint_sha256']}"
            )

    out = args.output_dir.resolve()
    if out.exists() and not args.resume:
        raise RuntimeError(
            f"Output exists; refuse overwrite without --resume: {out}"
        )
    out.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # PRE-EXECUTION lock, before any new MEMO/DINO inference.
    # -------------------------------------------------------------------------
    pre_lock_path = out / "R33A2B_PRE_EXECUTION_LOCK.json"
    pre_lock_payload = {
        "status": "LOCKED_R33A2B_BEFORE_NEW_POLYPGEN_MEMO_INFERENCE",
        "version": VERSION,
        "R33A1_final_sha256": EXPECTED_R33A1_FINAL_SHA256,
        "R33A1_protocol_sha256": EXPECTED_R33A1_PROTOCOL_SHA256,
        "R33A2A_final_sha256": EXPECTED_R33A2A_FINAL_SHA256,
        "R33A2A_bundle_sha256": EXPECTED_R33A2A_BUNDLE_SHA256,
        "R31B2_script_sha256": EXPECTED_R31B2_SCRIPT_SHA256,
        "R31B1_script_sha256": EXPECTED_R31B1_SCRIPT_SHA256,
        "R22A1_script_sha256": EXPECTED_R22A1_SCRIPT_SHA256,
        "R05D3_script_sha256": EXPECTED_R05D3_SCRIPT_SHA256,
        "R30A0_script_sha256": EXPECTED_R30A0_SCRIPT_SHA256,
        "source_fit": {
            "dataset": "NeoPolyp",
            "actions": list(SOURCE_ACTIONS),
            "rows": EXPECTED_SOURCE_ROWS,
            "feature_columns": FEATURE_COLUMNS,
            "target_rows_used_for_fit": 0,
        },
        "target": {
            "dataset": TARGET_DATASET,
            "family": TARGET_FAMILY,
            "cases": TARGET_CASES,
            "states": list(TARGET_STATE_IDS),
            "model_cases": EXPECTED_TARGET_ROWS,
            "action": THIRD_ACTION,
        },
        "memo": {
            "optimizer": "Adam",
            "lr": SELECTED_LR,
            "weight_decay": 0.0,
            "steps": 1,
            "transforms": ["IDENTITY", "HFLIP", "VFLIP", "HVFLIP"],
            "all_model_parameters": True,
        },
        "target_calibration": False,
        "target_score_reversal": False,
        "target_GT_HARM_read_hash_decode": [False, False, False],
    }

    if pre_lock_path.exists():
        existing = json.loads(pre_lock_path.read_text(encoding="utf-8"))
        # Avoid comparing output-specific mutable fields: the complete payload
        # is deterministic for this execution.
        if existing != pre_lock_payload:
            raise RuntimeError("Existing R33A2B pre-execution lock differs.")
    else:
        atomic_json(pre_lock_path, pre_lock_payload)

    # -------------------------------------------------------------------------
    # Sidecars named exactly as expected by retained R31B2 semantic functions.
    # -------------------------------------------------------------------------
    rgb_sidecar = out / "R31B2_800CASE_RGB_MANIFEST_PRE_GT.csv"
    if rgb_sidecar.exists():
        old = pd.read_csv(rgb_sidecar, dtype={"sample_id": str}, low_memory=False)
        if not old.astype(str).equals(rgb.astype(str)):
            raise RuntimeError("Existing RGB semantic sidecar differs.")
    else:
        atomic_csv(rgb, rgb_sidecar)

    # Exact bitorder from R05D3; R31B2 later cross-checks against K2A.
    bitorder = str(getattr(r05, "BITORDER", "big"))

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")

    # -------------------------------------------------------------------------
    # 1) Exact SOURCE + MEMO predictions, state by state.
    # -------------------------------------------------------------------------
    npz_paths: List[Path] = []
    state_locks: List[Dict[str, Any]] = []

    for sidx, state in states.iterrows():
        npz, lock = run_state_memo(
            out=out,
            state_index=int(sidx),
            state=state,
            rgb=rgb,
            r31b1=r31b1,
            r31b2=r31b2,
            r05=r05,
            device=device,
            bitorder=bitorder,
            resume=args.resume,
        )
        npz_paths.append(npz)
        state_locks.append(lock)

    pred_inv = build_prediction_inventory(
        out=out,
        states=states,
        npz_paths=npz_paths,
        state_locks=state_locks,
    )
    pred_inv_path = out / "R31B2_STATE_PREDICTION_INVENTORY_PRE_GT.csv"
    if pred_inv_path.exists():
        old = pd.read_csv(pred_inv_path, low_memory=False)
        if not old.astype(str).equals(pred_inv.astype(str)):
            raise RuntimeError("Existing prediction inventory differs.")
    else:
        atomic_csv(pred_inv, pred_inv_path)

    print("\nR33A2B PREDICTION LOCK")
    print("  states              :", len(npz_paths))
    print("  target model-cases  :", EXPECTED_TARGET_ROWS)
    print("  inventory SHA256    :", sha256_file(pred_inv_path))
    print("  GT read/hash/decode : NO / NO / NO")

    # -------------------------------------------------------------------------
    # 2) Exact retained R31B2 SOURCE/MEMO DINO/PCA semantic representation.
    # -------------------------------------------------------------------------
    q, dm, semantic_table, semantic_lock = run_exact_semantic_lock(
        out=out,
        rgb=rgb,
        states=states,
        npz_paths=npz_paths,
        r31b2=r31b2,
        bitorder=bitorder,
        allow_download=bool(args.allow_download),
        device_name=args.device,
        resume=args.resume,
    )

    print("\nR33A2B SEMANTIC LOCK")
    print("  q_source66 :", q.shape, q.dtype)
    print("  dSemantic64:", dm.shape, dm.dtype)
    print("  GT/HARM    : NOT READ")

    # Freeze R33-named copies to make later evaluation independent of
    # historical R31B2 filenames.
    q_path = out / "R33A2B_POLYPGEN_SOURCE_Q66_PRE_HARM.npy"
    dm_path = out / "R33A2B_POLYPGEN_MEMO_DSEM64_PRE_HARM.npy"
    if not q_path.exists():
        np.save(q_path, q, allow_pickle=False)
    if not dm_path.exists():
        np.save(dm_path, dm, allow_pickle=False)

    if sha256_file(q_path) != sha256_file(Path(r31b2.semantic_paths(out)["q"])):
        raise RuntimeError("R33 q_source66 copy differs from exact R31B2 output.")
    if sha256_file(dm_path) != sha256_file(Path(r31b2.semantic_paths(out)["memo"])):
        raise RuntimeError("R33 dSemantic64 copy differs from exact R31B2 output.")

    # -------------------------------------------------------------------------
    # 3) Source-only predictor fit.
    # -------------------------------------------------------------------------
    scaler, clf, source_fit_info = fit_source_predictor(source)
    predictor_npz_path = save_predictor_numeric_lock(
        out=out,
        scaler=scaler,
        clf=clf,
        source_info=source_fit_info,
    )

    # -------------------------------------------------------------------------
    # 4) External PolypGen MEMO risk scores, still PRE-HARM.
    # -------------------------------------------------------------------------
    score = score_target(
        target_manifest=target_manifest,
        semantic_table=semantic_table,
        q=q,
        dm=dm,
        scaler=scaler,
        clf=clf,
    )

    diagnostics = collect_action_diagnostics(out, states, rgb)
    score = score.merge(
        diagnostics,
        on=["sample_id", "model_state_id"],
        how="left",
        validate="one_to_one",
    )
    if len(score) != EXPECTED_TARGET_ROWS:
        raise RuntimeError("Final target score rows drift.")
    assert_no_target_outcomes(score, "Final R33A2B score lock")

    # Build explicit 130-D feature table.
    feature_matrix = np.concatenate([q, dm], axis=1)
    feature_df = pd.DataFrame(feature_matrix, columns=FEATURE_COLUMNS)
    feature_meta = score[
        [
            "sample_id",
            "model_state_id",
            "model_family",
            "training_seed",
            "action",
        ]
    ].reset_index(drop=True)
    feature_table = pd.concat([feature_meta, feature_df], axis=1)
    assert_no_target_outcomes(feature_table, "R33A2B feature table")

    score_path = out / "R33A2B_POLYPGEN_MEMO_EXTERNAL_SAFETTA_SCORE_PRE_HARM_LOCK.csv"
    feature_path = out / "R33A2B_POLYPGEN_MEMO_Q66_DSEM64_FEATURE_PRE_HARM_LOCK.csv"
    atomic_csv(score, score_path)
    atomic_csv(feature_table, feature_path)

    # Prediction-only descriptive summary; never used for selection/tuning.
    summary = (
        score.groupby(
            ["model_state_id", "training_seed"],
            as_index=False,
        )
        .agg(
            rows=("sample_id", "size"),
            physical_cases=("sample_id", "nunique"),
            safettta_risk_mean=("safettta_external_harm_risk", "mean"),
            safettta_risk_std=("safettta_external_harm_risk", "std"),
            ccd_risk_mean=("ccd_risk", "mean"),
            adic_harm_risk_mean=("adic_harm_risk", "mean"),
            mc_entropy_mean=("mc_predictive_entropy_risk", "mean"),
            memo_changed_pixels_mean=("memo_changed_pixels", "mean"),
            memo_changed_fraction=("memo_changed_pixels", lambda x: float(np.mean(np.asarray(x) > 0))),
            memo_loss_mean=("memo_loss_preupdate", "mean"),
        )
        .sort_values("training_seed", kind="mergesort")
        .reset_index(drop=True)
    )
    summary_path = out / "R33A2B_PRE_HARM_SCORE_DIAGNOSTIC_SUMMARY.csv"
    atomic_csv(summary, summary_path)

    # -------------------------------------------------------------------------
    # 5) Immutable PRE-HARM lock.
    # -------------------------------------------------------------------------
    semantic_exact_paths = r31b2.semantic_paths(out)
    score_lock = {
        "status": "PASS_R33A2B_POLYPGEN_MEMO_EXTERNAL_SCORE_PRE_HARM_LOCK",
        "version": VERSION,
        "scientific_status": (
            "LOCKED_AFTER_NEW_POLYPGEN_MEMO_AND_FEATURE_INFERENCE; "
            "BEFORE_NEW_MEMO_GT_HARM_EVALUATION"
        ),
        "source_training": source_fit_info,
        "external_target": {
            "dataset": TARGET_DATASET,
            "family": TARGET_FAMILY,
            "physical_cases": TARGET_CASES,
            "states": TARGET_STATES,
            "model_cases": EXPECTED_TARGET_ROWS,
            "action": THIRD_ACTION,
        },
        "representation": {
            "q_source66": (
                "SOURCE morphology2 + SOURCE-conditioned frozen PCA64"
            ),
            "dSemantic64": (
                "MEMO candidate-conditioned frozen PCA64 - same-run SOURCE PCA64"
            ),
            "candidate_morphology_used": False,
            "ActionID_used": False,
        },
        "artifacts": {
            "prediction_inventory": {
                "path": str(pred_inv_path),
                "sha256": sha256_file(pred_inv_path),
            },
            "exact_R31B2_q": {
                "path": str(semantic_exact_paths["q"]),
                "sha256": sha256_file(semantic_exact_paths["q"]),
            },
            "exact_R31B2_dSemantic": {
                "path": str(semantic_exact_paths["memo"]),
                "sha256": sha256_file(semantic_exact_paths["memo"]),
            },
            "q_source66": {
                "path": str(q_path),
                "sha256": sha256_file(q_path),
            },
            "dSemantic64": {
                "path": str(dm_path),
                "sha256": sha256_file(dm_path),
            },
            "feature_table": {
                "path": str(feature_path),
                "sha256": sha256_file(feature_path),
            },
            "predictor_numeric_lock": {
                "path": str(predictor_npz_path),
                "sha256": sha256_file(predictor_npz_path),
            },
            "target_score_table": {
                "path": str(score_path),
                "sha256": sha256_file(score_path),
            },
            "diagnostic_summary": {
                "path": str(summary_path),
                "sha256": sha256_file(summary_path),
            },
        },
        "existing_published_reliability_scores": {
            "SicTTA_CCD": "ccd_risk",
            "TEGDA_ADIC": "adic_harm_risk",
            "MC_dropout": "mc_predictive_entropy_risk",
            "orientation_changed": False,
        },
        "target_GT_read": False,
        "target_GT_hash": False,
        "target_GT_decode": False,
        "target_HARM_evaluated": False,
        "target_calibration": False,
        "target_threshold_tuning": False,
        "target_score_reversal": False,
        "MEMO_LR_tuning_on_PolypGen": False,
        "external_web_network_access": bool(args.allow_download),
        "next": "R33A3_FIRST_POLYPGEN_MEMO_HARM_REVEAL_AND_EXTERNAL_JOINT_SHIFT_EVALUATION",
    }
    score_lock_path = out / "R33A2B_SCORE_PRE_HARM_LOCK.json"
    atomic_json(score_lock_path, score_lock)

    # Final lock.
    final = {
        "status": "PASS_R33A2B_EXTERNAL_MEMO_TRANSITION_SCORE_LOCK_COMPLETE",
        "version": VERSION,
        "R33A1_final_sha256": EXPECTED_R33A1_FINAL_SHA256,
        "R33A1_protocol_sha256": EXPECTED_R33A1_PROTOCOL_SHA256,
        "R33A2A_final_sha256": EXPECTED_R33A2A_FINAL_SHA256,
        "R33A2A_bundle_sha256": EXPECTED_R33A2A_BUNDLE_SHA256,
        "pre_execution_lock_sha256": sha256_file(pre_lock_path),
        "score_pre_harm_lock_sha256": sha256_file(score_lock_path),
        "target_score_sha256": sha256_file(score_path),
        "feature_table_sha256": sha256_file(feature_path),
        "q_source66_sha256": sha256_file(q_path),
        "dSemantic64_sha256": sha256_file(dm_path),
        "predictor_numeric_sha256": sha256_file(predictor_npz_path),
        "prediction_inventory_sha256": sha256_file(pred_inv_path),
        "target_cases": TARGET_CASES,
        "target_states": TARGET_STATES,
        "target_model_cases": EXPECTED_TARGET_ROWS,
        "target_GT_HARM_read_evaluated": False,
        "target_calibration": False,
        "target_score_reversal": False,
        "next": "R33A3_FIRST_POLYPGEN_MEMO_HARM_REVEAL_AND_EXTERNAL_JOINT_SHIFT_EVALUATION",
    }
    final_path = out / "R33A2B_FINAL_LOCK.json"
    atomic_json(final_path, final)

    print("\n" + "=" * 124)
    print("R33A2B SOURCE-ONLY PREDICTOR")
    print("=" * 124)
    print("  source rows        :", source_fit_info["training_rows"])
    print("  source cases       :", source_fit_info["physical_cases"])
    print("  source states      :", source_fit_info["model_states"])
    print("  source actions     :", source_fit_info["actions"])
    print("  source HARM count  :", source_fit_info["harm_count"])
    print("  source HARM prev   :", f"{source_fit_info['harm_prevalence']:.9f}")
    print("  features           :", source_fit_info["feature_count"])
    print("  target rows fit    : 0")
    print("  target labels fit  : NO")

    print("\n" + "=" * 124)
    print("R33A2B POLYPGEN PRE-HARM SCORE SUMMARY")
    print("=" * 124)
    print(summary.to_string(index=False))

    print("\nR33A2B PRE-HARM LOCK")
    print("  prediction inventory SHA :", sha256_file(pred_inv_path))
    print("  q_source66 SHA            :", sha256_file(q_path))
    print("  dSemantic64 SHA           :", sha256_file(dm_path))
    print("  feature table SHA         :", sha256_file(feature_path))
    print("  target score SHA          :", sha256_file(score_path))
    print("  score lock SHA            :", sha256_file(score_lock_path))
    print("  PolypGen GT/HARM read     : NO")

    print(
        "\nFINAL STATUS : "
        "PASS_R33A2B_EXTERNAL_MEMO_TRANSITION_SCORE_LOCK_COMPLETE"
    )
    print("Target calibration          : NO")
    print("Target score reversal       : NO")
    print("MEMO LR target tuning       : NO")
    print("Final lock SHA256           :", sha256_file(final_path))
    print("Output                      :", out)
    print(
        "NEXT                        : "
        "R33A3_FIRST_POLYPGEN_MEMO_HARM_REVEAL_AND_EXTERNAL_JOINT_SHIFT_EVALUATION"
    )
    print("=" * 124)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
