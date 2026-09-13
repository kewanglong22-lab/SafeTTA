#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SafeTTA R31B2
800-case PRE-GT MEMO-SEG4-1STEP Prediction + Semantic Transition Lock

Purpose
-------
R31B1 selected and froze the third-action learning rate using ONLY the
200-case NeoPolyp action-design subset:

    MEMO-SEG4-1STEP
    selected LR = 1e-5

R31B2 now applies that frozen action to the untouched 800-case R31B
leave-one-action-out evaluation subset, but STILL DOES NOT OPEN ITS GT.

This stage freezes, before evaluation GT reveal:
  1) SOURCE and MEMO candidate masks for all 800 cases x 9 frozen states;
  2) q_source66;
  3) MEMO delta_semantic64;
  4) exact row/state/image ordering and SHA identities.

Only after R31B2 PASS may R31B3 open the 800-case GT and perform the true
three-action leave-one-action-out HARM evaluation.

Information boundary
--------------------
NeoPolyp 800-case RGB             : YES
NeoPolyp 800-case GT read/hash    : NO / NO
NeoPolyp 800-case GT decode       : NO
MEMO LR tuning                    : NO
Model/checkpoint changes          : NO
EndoTect                          : NO
PolypGen / SUN-SEG / PROMISE12    : NO

Execution semantics
-------------------
R31B2 imports the exact frozen R31B1 implementation and reuses its:
  * 9 frozen model-state loading;
  * MEMO execution contract;
  * all-parameter Adam one-step update;
  * four exact flip transforms;
  * parameter+buffer episodic reset;
  * deterministic action seed;
  * source and deployed-mask definitions.

Each model state executes in a fresh Python/CUDA subprocess.

Run
---
python Q1_R31B2_800case_memo_prediction_and_semantic_lock_v1.py

Resume
------
python Q1_R31B2_800case_memo_prediction_and_semantic_lock_v1.py --resume
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


VERSION = "2026-09-12-R31B2-v1"

ROOT = Path(r"F:\MEDSEG_SAFETTA")
CODE = ROOT / "code"
OUTPUTS = ROOT / "outputs"

# ---------------------------------------------------------------------
# Exact upstream locks
# ---------------------------------------------------------------------

R31B0_DIR = ROOT / "R31B0_memo_seg4_third_action_protocol_lock_v1"
R31B0_FINAL = R31B0_DIR / "R31B0_FINAL_LOCK.json"
EXPECTED_R31B0_FINAL_SHA256 = (
    "03e16b4064611a927946b577d8a0740353e9037bd4eca72426d441c709010951"
)
R31B0_PARTITION = R31B0_DIR / "R31B0_SOURCE_CASE_PARTITION.csv"

R31B1_SCRIPT = CODE / "Q1_R31B1_memo_seg4_action_design_and_lr_freeze_v1.py"
EXPECTED_R31B1_SCRIPT_SHA256 = (
    "3391fe4d7161df488f372379a65d72c0dc0cc4e1e774e02e90d68c9ad17f911b"
)

R31B1_DIR = ROOT / "R31B1_memo_seg4_action_design_and_lr_freeze_v1"
R31B1_SELECTED = R31B1_DIR / "R31B1_SELECTED_MEMO_LR_LOCK.json"
EXPECTED_R31B1_SELECTED_SHA256 = (
    "6915ead8a8ccffc3c1b00cc5003a118c20b390b4303345943609fa94741b4551"
)
R31B1_FINAL = R31B1_DIR / "R31B1_FINAL_LOCK.json"
EXPECTED_R31B1_FINAL_SHA256 = (
    "09668378b0083876938bf67e0ebbbee17ba774760147ccce9dba5d1683869044"
)

# R30 semantic implementation reused exactly.
R30A4B2_FIX5_SCRIPT = (
    CODE / "Q1_R30A4B2_pre_gt_endotect_inference_and_r30_action_lock_v1_fix5.py"
)
EXPECTED_R30A4B2_FIX5_SHA256 = (
    "125d9914119fff6da4688ee4d31810119f36ba5ec2403234cc200e9fee7f3798"
)

R22_SCRIPT = CODE / "Q1_R22C1B2_gtfree_semantic_transition_features_and_commit_scores_v1.py"
EXPECTED_R22_SHA256 = (
    "7a2f9bebfc578914042d56f4cc9e9c1c245583e0760729979f636303b5977774"
)

R30A0_SCRIPT = CODE / "Q1_R30A0_paired_action_crc_protocol_and_asset_audit_v1_fix1.py"
EXPECTED_R30A0_SHA256 = (
    "9dc5d3a13f3d64608c4c599f811f929f731d7fac05e2aaa66eb344e24ace2298"
)

DEFAULT_OUT = ROOT / "R31B2_800case_memo_prediction_and_semantic_lock_v1"

# ---------------------------------------------------------------------
# Frozen protocol constants
# ---------------------------------------------------------------------

THIRD_ACTION = "MEMO-SEG4-1STEP"
SELECTED_LR = 1e-5

EVAL_PARTITION = "R31B_LOAO_EVALUATION"
EXPECTED_EVAL_CASES = 800
EXPECTED_STATES = 9
EXPECTED_MODEL_CASES = EXPECTED_EVAL_CASES * EXPECTED_STATES

IMAGE_SIZE = 352
PACKED_BYTES = (IMAGE_SIZE * IMAGE_SIZE + 7) // 8

DINO_BATCH_SIZE = 8

FORBIDDEN_EXTERNAL_TOKENS = (
    "endotect",
    "polypgen",
    "sun-seg",
    "sunseg",
    "promise12",
)

FORBIDDEN_PRE_GT_COLUMNS = (
    "gt",
    "dice",
    "delta_dice",
    "harm",
    "benefit",
    "outcome",
    "label",
)

# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------

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


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import module: {path}")
    mod = importlib.util.module_from_spec(spec)
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


def reject_external_path(path: Path):
    low = str(path).lower()
    if any(tok in low for tok in FORBIDDEN_EXTERNAL_TOKENS):
        raise RuntimeError(f"External path forbidden in R31B2: {path}")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def pack_mask(mask: np.ndarray, bitorder: str) -> np.ndarray:
    m = np.asarray(mask, dtype=np.uint8)
    if m.shape != (IMAGE_SIZE, IMAGE_SIZE):
        raise RuntimeError(f"Mask shape={m.shape}")
    p = np.packbits(
        (m > 0).reshape(-1).astype(np.uint8),
        bitorder=bitorder,
    )
    if p.shape != (PACKED_BYTES,):
        raise RuntimeError(f"Packed shape={p.shape}")
    return p


def unpack_mask(packed: np.ndarray, bitorder: str) -> np.ndarray:
    b = np.unpackbits(
        np.asarray(packed, dtype=np.uint8),
        count=IMAGE_SIZE * IMAGE_SIZE,
        bitorder=bitorder,
    )
    return b.reshape(IMAGE_SIZE, IMAGE_SIZE).astype(bool, copy=False)


# ---------------------------------------------------------------------
# Upstream / information-boundary verification
# ---------------------------------------------------------------------

def verify_upstream() -> Dict[str, Any]:
    require_sha(R31B0_FINAL, EXPECTED_R31B0_FINAL_SHA256, "R31B0 final")
    require_sha(R31B1_SCRIPT, EXPECTED_R31B1_SCRIPT_SHA256, "R31B1 script")
    require_sha(R31B1_SELECTED, EXPECTED_R31B1_SELECTED_SHA256, "R31B1 selected LR")
    require_sha(R31B1_FINAL, EXPECTED_R31B1_FINAL_SHA256, "R31B1 final")
    require_sha(
        R30A4B2_FIX5_SCRIPT,
        EXPECTED_R30A4B2_FIX5_SHA256,
        "R30A4B2 Fix5 semantic implementation",
    )
    require_sha(R22_SCRIPT, EXPECTED_R22_SHA256, "R22 semantic implementation")
    require_sha(R30A0_SCRIPT, EXPECTED_R30A0_SHA256, "R30 feature schema")

    selected = json.loads(R31B1_SELECTED.read_text(encoding="utf-8"))
    final = json.loads(R31B1_FINAL.read_text(encoding="utf-8"))

    lr = float(selected.get("selected_learning_rate", float("nan")))
    if not math.isclose(lr, SELECTED_LR, rel_tol=0.0, abs_tol=1e-15):
        raise RuntimeError(
            f"Frozen MEMO LR changed: observed={lr} expected={SELECTED_LR}"
        )

    if str(selected.get("third_action")) != THIRD_ACTION:
        raise RuntimeError("R31B1 third action changed.")

    if int(selected.get("evaluation_800_gt_access", 1)) != 0:
        raise RuntimeError("R31B1 reports evaluation GT access.")

    if int(final.get("evaluation_800_gt_access", 1)) != 0:
        raise RuntimeError("R31B1 final reports evaluation GT access.")

    if int(final.get("evaluation_cases_used_for_selection", -1)) != 0:
        raise RuntimeError("800-case evaluation subset contaminated LR selection.")

    return {"selected": selected, "final": final}


def load_eval_rgb_lock(out: Path) -> pd.DataFrame:
    if not R31B0_PARTITION.is_file():
        raise FileNotFoundError(R31B0_PARTITION)

    raw = pd.read_csv(R31B0_PARTITION, low_memory=False)
    required = {
        "sample_id",
        "r31b_partition",
        "image_path",
        "gt_path",
        "split_sha256_key",
    }
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise RuntimeError(f"R31B0 partition missing columns={missing}")

    d = raw[raw["r31b_partition"] == EVAL_PARTITION].copy()
    if len(d) != EXPECTED_EVAL_CASES:
        raise RuntimeError(f"Evaluation cases={len(d)} expected={EXPECTED_EVAL_CASES}")
    if d["sample_id"].astype(str).nunique() != EXPECTED_EVAL_CASES:
        raise RuntimeError("Evaluation sample_id not unique.")

    # GT path exists only in upstream partition metadata. R31B2 intentionally
    # drops it here and NEVER opens/hashes/decodes it.
    d = d[
        ["sample_id", "r31b_partition", "image_path", "split_sha256_key"]
    ].copy()
    d["sample_id"] = d["sample_id"].astype(str)
    d = d.sort_values("sample_id", kind="mergesort").reset_index(drop=True)

    rows = []
    for row in tqdm(
        d.to_dict(orient="records"),
        desc="R31B2 lock 800 evaluation RGB",
        unit="image",
        dynamic_ncols=True,
    ):
        p = Path(str(row["image_path"]))
        reject_external_path(p)
        if not p.is_file():
            raise FileNotFoundError(p)
        rows.append({
            **row,
            "image_path": str(p),
            "image_bytes": int(p.stat().st_size),
            "image_sha256": sha256_file(p),
        })

    locked = pd.DataFrame(rows)
    path = out / "R31B2_800CASE_RGB_MANIFEST_PRE_GT.csv"
    atomic_csv(locked, path)
    return locked


# ---------------------------------------------------------------------
# K2A bit packing semantics
# ---------------------------------------------------------------------

def resolve_k2a_bitorder(r22: Any) -> Tuple[str, Path]:
    p = getattr(r22, "R10K2A_SCRIPT", None)
    if p is not None and Path(p).is_file():
        p = Path(p)
    else:
        cands = sorted(CODE.glob("Q1_R10K2A*.py"))
        if len(cands) != 1:
            raise RuntimeError(
                f"Cannot uniquely resolve R10K2A script: {list(map(str, cands))}"
            )
        p = cands[0]

    k2a = import_module(p, "r31b2_k2a_preflight")
    if int(getattr(k2a, "MASK_SIZE", -1)) != IMAGE_SIZE:
        raise RuntimeError("K2A mask-size drift.")
    if int(getattr(k2a, "PACKED_BYTES", -1)) != PACKED_BYTES:
        raise RuntimeError("K2A packed-byte drift.")
    bitorder = str(getattr(k2a, "BITORDER", "big"))
    if bitorder not in {"big", "little"}:
        raise RuntimeError(f"Invalid K2A bitorder={bitorder}")
    return bitorder, p


# ---------------------------------------------------------------------
# Frozen third-action prediction
# ---------------------------------------------------------------------

def worker_run(
    state_index: int,
    out: Path,
    rgb_csv: Path,
    bitorder: str,
):
    import torch

    r31b1 = import_module(R31B1_SCRIPT, f"r31b2_r31b1_{state_index}")

    panel = r31b1.load_panel()
    if not (0 <= state_index < len(panel)):
        raise RuntimeError(f"Invalid state index={state_index}")
    state = panel.iloc[state_index].to_dict()

    rgb = pd.read_csv(rgb_csv, low_memory=False)
    if len(rgb) != EXPECTED_EVAL_CASES:
        raise RuntimeError("Worker RGB cohort size drift.")

    # The PRE-GT worker manifest contains no GT path by construction.
    for c in rgb.columns:
        if "gt" in c.lower():
            raise RuntimeError(f"GT-bearing column leaked to worker: {c}")

    r05 = r31b1.import_module(
        r31b1.R05D3_SCRIPT,
        f"r31b2_r05_{state_index}",
    )
    r03 = r31b1.import_module(
        r31b1.R03_SCRIPT,
        f"r31b2_r03_{state_index}",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"[WORKER {state_index+1}/{EXPECTED_STATES}] "
        f"{state['model_state_id']} | {state['model_family']} | "
        f"device={device} | LR={SELECTED_LR}"
    )

    runtime = r31b1.build_runtime_adapter(state, device, r05, r03)
    model = runtime["model"]
    to_tensor = runtime["to_tensor"]
    final_mask = runtime["final_mask"]
    foreground_prob = runtime["foreground_prob"]
    source_params = runtime["source_params"]
    source_buffers = runtime["source_buffers"]

    source_packed = np.empty(
        (EXPECTED_EVAL_CASES, PACKED_BYTES),
        dtype=np.uint8,
    )
    memo_packed = np.empty_like(source_packed)
    memo_loss = np.empty(EXPECTED_EVAL_CASES, dtype=np.float32)
    grad_l2 = np.empty(EXPECTED_EVAL_CASES, dtype=np.float32)
    changed_pixels = np.empty(EXPECTED_EVAL_CASES, dtype=np.int32)
    action_seeds = np.empty(EXPECTED_EVAL_CASES, dtype=np.int64)

    iterator = tqdm(
        rgb.to_dict(orient="records"),
        total=EXPECTED_EVAL_CASES,
        desc=f"R31B2 {state['model_family']} {state['training_seed']}",
        unit="case",
        dynamic_ncols=True,
    )

    for i, row in enumerate(iterator):
        image_path = Path(str(row["image_path"]))
        reject_external_path(image_path)

        with Image.open(image_path) as im:
            native = im.convert("RGB")
        x = to_tensor(native)

        # Frozen SOURCE candidate.
        r31b1.restore_model(model, source_params, source_buffers)
        model.eval()
        with torch.no_grad():
            source_mask = np.asarray(final_mask(x), dtype=np.uint8)

        # Frozen MEMO transaction.
        r31b1.restore_model(model, source_params, source_buffers)
        seed_txn = r31b1.action_seed(
            str(state["model_state_id"]),
            str(row["sample_id"]),
        )
        r31b1.set_seed(seed_txn)

        model.train()
        params = list(model.parameters())
        for p in params:
            p.requires_grad_(True)

        opt = torch.optim.Adam(
            params,
            lr=SELECTED_LR,
            weight_decay=r31b1.WEIGHT_DECAY,
        )
        opt.zero_grad(set_to_none=True)

        x4 = r31b1.augment_batch(x, torch)
        prob4 = foreground_prob(x4)
        aligned = r31b1.inverse_align_probs(prob4, torch)
        marginal = aligned.mean(dim=0, keepdim=True)
        loss = r31b1.binary_entropy_mean(marginal, torch)

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite MEMO loss state={state['model_state_id']} "
                f"sample={row['sample_id']}"
            )

        loss.backward()

        gsq = 0.0
        for p in params:
            if p.grad is not None:
                gsq += float(
                    torch.sum(p.grad.detach().float() ** 2).cpu()
                )
        gnorm = math.sqrt(max(0.0, gsq))

        opt.step()
        opt.zero_grad(set_to_none=True)

        model.eval()
        with torch.no_grad():
            memo_mask = np.asarray(final_mask(x), dtype=np.uint8)

        source_packed[i] = pack_mask(source_mask, bitorder)
        memo_packed[i] = pack_mask(memo_mask, bitorder)
        memo_loss[i] = float(loss.detach().cpu())
        grad_l2[i] = float(gnorm)
        changed_pixels[i] = int(np.count_nonzero(source_mask != memo_mask))
        action_seeds[i] = int(seed_txn)

        iterator.set_postfix(
            chg=int(changed_pixels[i]),
            loss=f"{memo_loss[i]:.4f}",
        )

        r31b1.restore_model(model, source_params, source_buffers)
        del opt, x4, prob4, aligned, marginal, loss, source_mask, memo_mask, x

    state_npz = out / f"state_{state_index:02d}_source_memo_pre_gt.npz"
    tmp_npz = state_npz.with_name(state_npz.name + ".tmp.npz")

    np.savez_compressed(
        tmp_npz,
        source_masks_packed=source_packed,
        memo_masks_packed=memo_packed,
        memo_loss_preupdate=memo_loss,
        grad_l2=grad_l2,
        changed_pixels=changed_pixels,
        action_seed=action_seeds,
        sample_id=np.asarray(rgb["sample_id"].astype(str).tolist(), dtype="U"),
    )
    os.replace(tmp_npz, state_npz)

    lock = {
        "status": "PASS_R31B2_STATE_PRE_GT_PREDICTION_LOCK",
        "version": VERSION,
        "state_index": int(state_index),
        "model_state_id": str(state["model_state_id"]),
        "model_family": str(state["model_family"]),
        "training_seed": int(state["training_seed"]),
        "checkpoint_sha256": str(state["checkpoint_sha256"]),
        "third_action": THIRD_ACTION,
        "selected_learning_rate": SELECTED_LR,
        "physical_cases": EXPECTED_EVAL_CASES,
        "source_memo_npz": str(state_npz),
        "source_memo_npz_sha256": sha256_file(state_npz),
        "bitorder": bitorder,
        "gt_read_hash_decode": [False, False, False],
        "external_data_access": False,
    }
    lock_path = out / f"state_{state_index:02d}_source_memo_pre_gt.lock.json"
    atomic_json(lock_path, lock)

    del runtime, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    print("STATE PRE-GT LOCK PASS:", state["model_state_id"])
    print("NPZ SHA256:", sha256_file(state_npz))


def validate_state_artifact(
    out: Path,
    idx: int,
    state: Mapping[str, Any],
    rgb: pd.DataFrame,
) -> bool:
    npz_path = out / f"state_{idx:02d}_source_memo_pre_gt.npz"
    lock_path = out / f"state_{idx:02d}_source_memo_pre_gt.lock.json"
    if not npz_path.is_file() or not lock_path.is_file():
        return False

    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        if lock.get("status") != "PASS_R31B2_STATE_PRE_GT_PREDICTION_LOCK":
            return False
        if str(lock.get("model_state_id")) != str(state["model_state_id"]):
            return False
        if not math.isclose(
            float(lock.get("selected_learning_rate")),
            SELECTED_LR,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            return False
        if sha256_file(npz_path) != str(lock.get("source_memo_npz_sha256")):
            return False

        with np.load(npz_path, allow_pickle=False) as z:
            src = np.asarray(z["source_masks_packed"])
            memo = np.asarray(z["memo_masks_packed"])
            sid = np.asarray(z["sample_id"]).astype(str)
        if src.shape != (EXPECTED_EVAL_CASES, PACKED_BYTES):
            return False
        if memo.shape != src.shape:
            return False
        if not np.array_equal(
            sid,
            rgb["sample_id"].astype(str).to_numpy(),
        ):
            return False
        return True
    except Exception:
        return False


def build_prediction_inventory(
    out: Path,
    panel: pd.DataFrame,
    rgb: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[Path]]:
    rows = []
    paths = []

    for idx, state in panel.iterrows():
        if not validate_state_artifact(out, idx, state, rgb):
            raise RuntimeError(
                f"Invalid state artifact idx={idx} {state['model_state_id']}"
            )
        p = out / f"state_{idx:02d}_source_memo_pre_gt.npz"
        l = out / f"state_{idx:02d}_source_memo_pre_gt.lock.json"
        rows.append({
            "state_no": int(idx + 1),
            "state_index": int(idx),
            "model_state_id": str(state["model_state_id"]),
            "model_family": str(state["model_family"]),
            "training_seed": int(state["training_seed"]),
            "checkpoint_sha256": str(state["checkpoint_sha256"]),
            "prediction_npz": str(p),
            "prediction_npz_sha256": sha256_file(p),
            "state_lock": str(l),
            "state_lock_sha256": sha256_file(l),
        })
        paths.append(p)

    inv = pd.DataFrame(rows)
    path = out / "R31B2_STATE_PREDICTION_INVENTORY_PRE_GT.csv"
    atomic_csv(inv, path)
    return inv, paths


# ---------------------------------------------------------------------
# Frozen R22 semantic representation for SOURCE + MEMO
# ---------------------------------------------------------------------

def load_prediction_stack(
    npz_paths: Sequence[Path],
) -> Tuple[np.ndarray, np.ndarray]:
    srcs, memos = [], []
    for p in npz_paths:
        with np.load(p, allow_pickle=False) as z:
            src = np.asarray(z["source_masks_packed"], dtype=np.uint8)
            memo = np.asarray(z["memo_masks_packed"], dtype=np.uint8)
        if src.shape != (EXPECTED_EVAL_CASES, PACKED_BYTES):
            raise RuntimeError(f"{p.name} SOURCE shape={src.shape}")
        if memo.shape != src.shape:
            raise RuntimeError(f"{p.name} MEMO shape={memo.shape}")
        srcs.append(src)
        memos.append(memo)

    source = np.concatenate(srcs, axis=0)
    memo = np.concatenate(memos, axis=0)

    if source.shape != (EXPECTED_MODEL_CASES, PACKED_BYTES):
        raise RuntimeError(f"Stack SOURCE shape={source.shape}")
    if memo.shape != source.shape:
        raise RuntimeError(f"Stack MEMO shape={memo.shape}")
    return source, memo


def build_occupancy_and_source_m2(
    source: np.ndarray,
    memo: np.ndarray,
    k2a: Any,
    j1: Any,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    occ_s = np.zeros((EXPECTED_MODEL_CASES, 256), dtype=np.float32)
    occ_m = np.zeros_like(occ_s)
    m2 = np.zeros((EXPECTED_MODEL_CASES, 2), dtype=np.float64)

    bitorder = str(getattr(k2a, "BITORDER", "big"))
    boundary_fn = getattr(j1, "boundary_density", None)
    unpack_fn = getattr(j1, "unpack_source_mask", None)
    if boundary_fn is None:
        raise RuntimeError("Frozen J1 boundary_density missing.")

    for i in tqdm(
        range(EXPECTED_MODEL_CASES),
        desc="R31B2 SOURCE/MEMO occupancy",
        unit="row",
        dynamic_ncols=True,
    ):
        occ_s[i] = k2a.unpack_mask_occupancy16(source[i])
        occ_m[i] = k2a.unpack_mask_occupancy16(memo[i])

        if unpack_fn is not None:
            sm = np.asarray(unpack_fn(source[i]), dtype=bool)
        else:
            sm = unpack_mask(source[i], bitorder)

        m2[i, 0] = float(sm.mean())
        m2[i, 1] = float(boundary_fn(sm))

    if not (
        np.isfinite(occ_s).all()
        and np.isfinite(occ_m).all()
        and np.isfinite(m2).all()
    ):
        raise RuntimeError("Non-finite occupancy/morphology values.")

    return occ_s, occ_m, m2


def semantic_paths(out: Path) -> Dict[str, Path]:
    return {
        "q": out / "R31B2_SOURCE_Q66_PRE_GT.npy",
        "memo": out / "R31B2_MEMO_DELTA_SEMANTIC64_PRE_GT.npy",
        "table": out / "R31B2_MEMO_FEATURE_TABLE_PRE_GT.csv",
        "lock": out / "R31B2_SEMANTIC_FEATURE_LOCK_PRE_GT.json",
    }


def semantic_lock_valid(
    out: Path,
    rgb_sha: str,
    pred_inv_sha: str,
) -> bool:
    p = semantic_paths(out)
    if not all(x.is_file() for x in p.values()):
        return False

    try:
        lock = json.loads(p["lock"].read_text(encoding="utf-8"))
        if lock.get("status") != "PASS_R31B2_PRE_GT_SEMANTIC_LOCK":
            return False
        if lock.get("rgb_manifest_sha256") != rgb_sha:
            return False
        if lock.get("prediction_inventory_sha256") != pred_inv_sha:
            return False

        q = np.load(p["q"], mmap_mode="r", allow_pickle=False)
        dm = np.load(p["memo"], mmap_mode="r", allow_pickle=False)
        if q.shape != (EXPECTED_MODEL_CASES, 66):
            return False
        if dm.shape != (EXPECTED_MODEL_CASES, 64):
            return False
        if q.dtype != np.float64 or dm.dtype != np.float64:
            return False
        if not np.isfinite(q).all() or not np.isfinite(dm).all():
            return False

        if lock["artifacts"]["q"]["sha256"] != sha256_file(p["q"]):
            return False
        if lock["artifacts"]["memo"]["sha256"] != sha256_file(p["memo"]):
            return False
        if lock["artifacts"]["table"]["sha256"] != sha256_file(p["table"]):
            return False

        return True
    except Exception:
        return False


def extract_semantics(
    out: Path,
    rgb: pd.DataFrame,
    panel: pd.DataFrame,
    npz_paths: Sequence[Path],
    bitorder: str,
    allow_download: bool,
    device_name: str,
    resume: bool,
):
    rgb_lock_path = out / "R31B2_800CASE_RGB_MANIFEST_PRE_GT.csv"
    pred_inv_path = out / "R31B2_STATE_PREDICTION_INVENTORY_PRE_GT.csv"
    rgb_sha = sha256_file(rgb_lock_path)
    pred_sha = sha256_file(pred_inv_path)

    if resume and semantic_lock_valid(out, rgb_sha, pred_sha):
        print("[RESUME] R31B2 semantic lock PASS")
        return

    fix5 = import_module(R30A4B2_FIX5_SCRIPT, "r31b2_fix5_semantic")
    r22 = import_module(R22_SCRIPT, "r31b2_r22")
    r30a0 = import_module(R30A0_SCRIPT, "r31b2_r30a0")

    qnames = list(getattr(r30a0, "QSOURCE", []))
    dnames = list(getattr(r30a0, "DSEM", []))
    if len(qnames) != 66 or len(dnames) != 64:
        raise RuntimeError(
            f"Frozen schema drift QSOURCE={len(qnames)} DSEM={len(dnames)}"
        )

    dep = fix5.resolve_semantic_dependencies(
        r22,
        allow_download,
        device_name,
    )
    k2a = dep["k2a"]
    j1 = dep["j1"]
    pca = dep["pca"]

    if str(getattr(k2a, "BITORDER", "big")) != bitorder:
        raise RuntimeError("Prediction bitorder != frozen K2A bitorder.")

    source, memo = load_prediction_stack(npz_paths)
    occ_s, occ_m, m2 = build_occupancy_and_source_m2(
        source,
        memo,
        k2a,
        j1,
    )

    src_pca = np.zeros((EXPECTED_MODEL_CASES, 64), dtype=np.float64)
    memo_pca = np.zeros_like(src_pca)

    run_dino_batch = getattr(dep["dino_helper"], "run_dino_batch", None)
    if run_dino_batch is None:
        raise RuntimeError("Frozen run_dino_batch missing.")

    image_paths = rgb["image_path"].astype(str).tolist()

    for start in tqdm(
        range(0, EXPECTED_EVAL_CASES, DINO_BATCH_SIZE),
        desc="R31B2 frozen DINO/PCA",
        unit="batch",
        dynamic_ncols=True,
    ):
        stop = min(start + DINO_BATCH_SIZE, EXPECTED_EVAL_CASES)
        _, patch_batch = run_dino_batch(
            image_paths[start:stop],
            dep["processor"],
            dep["model"],
            dep["device"],
            k2a,
            dep["torch"],
        )
        patch_batch = np.asarray(patch_batch)

        for j, case_idx in enumerate(range(start, stop)):
            rows = np.asarray(
                [
                    state_idx * EXPECTED_EVAL_CASES + case_idx
                    for state_idx in range(EXPECTED_STATES)
                ],
                dtype=int,
            )
            src_pca[rows] = fix5.conditioned_pca64_for_one_image(
                patch_batch[j],
                occ_s[rows],
                k2a,
                pca,
            )
            memo_pca[rows] = fix5.conditioned_pca64_for_one_image(
                patch_batch[j],
                occ_m[rows],
                k2a,
                pca,
            )

    q = np.concatenate([m2, src_pca], axis=1).astype(np.float64, copy=False)
    dm = (memo_pca - src_pca).astype(np.float64, copy=False)

    if q.shape != (EXPECTED_MODEL_CASES, 66):
        raise RuntimeError(f"q shape={q.shape}")
    if dm.shape != (EXPECTED_MODEL_CASES, 64):
        raise RuntimeError(f"MEMO delta-semantic shape={dm.shape}")
    if not np.isfinite(q).all() or not np.isfinite(dm).all():
        raise RuntimeError("Non-finite semantic features.")

    sp = semantic_paths(out)
    np.save(sp["q"], q, allow_pickle=False)
    np.save(sp["memo"], dm, allow_pickle=False)

    meta_rows = []
    for sidx, state in panel.iterrows():
        for cidx, row in rgb.iterrows():
            meta_rows.append({
                "r31b2_row": int(
                    sidx * EXPECTED_EVAL_CASES + cidx
                ),
                "sample_id": str(row["sample_id"]),
                "model_state_id": str(state["model_state_id"]),
                "model_family": str(state["model_family"]),
                "training_seed": int(state["training_seed"]),
                "action": THIRD_ACTION,
                "selected_learning_rate": SELECTED_LR,
            })
    meta = pd.DataFrame(meta_rows)
    if len(meta) != EXPECTED_MODEL_CASES:
        raise RuntimeError("Semantic metadata row count drift.")

    feature = meta.copy()
    for i, name in enumerate(qnames):
        feature[name] = q[:, i]
    for i, name in enumerate(dnames):
        feature[name] = dm[:, i]

    # This is a PRE-GT table. Prevent accidental outcome leakage.
    bad = [
        c for c in feature.columns
        if any(tok in c.lower() for tok in FORBIDDEN_PRE_GT_COLUMNS)
    ]
    if bad:
        raise RuntimeError(f"Forbidden PRE-GT columns generated: {bad}")

    atomic_csv(feature, sp["table"])

    lock = {
        "status": "PASS_R31B2_PRE_GT_SEMANTIC_LOCK",
        "version": VERSION,
        "third_action": THIRD_ACTION,
        "selected_learning_rate": SELECTED_LR,
        "physical_cases": EXPECTED_EVAL_CASES,
        "model_states": EXPECTED_STATES,
        "model_cases": EXPECTED_MODEL_CASES,
        "representation": {
            "q_source66": (
                "SOURCE morphology2 + SOURCE-conditioned frozen PCA64"
            ),
            "memo_delta_semantic64": (
                "MEMO candidate-conditioned PCA64 - same-run SOURCE PCA64"
            ),
            "candidate_morphology_used": False,
            "float16_feature_lock_replayed": True,
        },
        "rgb_manifest_sha256": rgb_sha,
        "prediction_inventory_sha256": pred_sha,
        "semantic_dependencies": {
            "R30A4B2_fix5_sha256": EXPECTED_R30A4B2_FIX5_SHA256,
            "R22_sha256": EXPECTED_R22_SHA256,
            "R30A0_sha256": EXPECTED_R30A0_SHA256,
            "k2a_script": str(dep["k2a_path"]),
            "k2a_script_sha256": sha256_file(dep["k2a_path"]),
            "j1_script": str(dep["j1_path"]),
            "j1_script_sha256": sha256_file(dep["j1_path"]),
            "pca_path": str(dep["pca_path"]),
            "pca_sha256": sha256_file(dep["pca_path"]),
        },
        "artifacts": {
            "q": {
                "path": str(sp["q"]),
                "sha256": sha256_file(sp["q"]),
            },
            "memo": {
                "path": str(sp["memo"]),
                "sha256": sha256_file(sp["memo"]),
            },
            "table": {
                "path": str(sp["table"]),
                "sha256": sha256_file(sp["table"]),
            },
        },
        "gt_read_hash_decode": [False, False, False],
        "external_data_access": False,
        "next": "R31B3_FIRST_800CASE_GT_REVEAL_AND_THREE_ACTION_LOAO",
    }
    atomic_json(sp["lock"], lock)

    if not semantic_lock_valid(out, rgb_sha, pred_sha):
        raise RuntimeError("Fresh R31B2 semantic lock failed validation.")

    print("R31B2 PRE-GT SEMANTIC LOCK: PASS")
    print("  q_source66 shape          :", q.shape)
    print("  MEMO delta_semantic shape:", dm.shape)
    print("  GT read/hash/decode       : NO / NO / NO")


def make_inventory(out: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(out.rglob("*")):
        if p.is_file() and not p.name.endswith(".tmp"):
            rows.append({
                "relative_path": str(p.relative_to(out)),
                "bytes": int(p.stat().st_size),
                "sha256": sha256_file(p),
            })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--allow-download", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--_state-worker", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--_rgb-csv", type=Path, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--_bitorder", type=str, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    out = args.output_dir.resolve()

    if args._state_worker is not None:
        if args._rgb_csv is None or args._bitorder is None:
            raise RuntimeError("State worker missing private arguments.")
        worker_run(
            args._state_worker,
            out,
            args._rgb_csv.resolve(),
            args._bitorder,
        )
        return 0

    print("=" * 124)
    print("SafeTTA R31B2 800-case PRE-GT MEMO Prediction + Semantic Lock")
    print("Version:", VERSION)
    print("Frozen third action          :", THIRD_ACTION)
    print("Frozen MEMO LR               :", SELECTED_LR)
    print("800-case RGB access          : YES")
    print("800-case GT read/hash/decode : NO / NO / NO")
    print("MEMO tuning                  : NO")
    print("External cohort access       : NO")
    print("Fresh CUDA process/state     : YES")
    print("Resume                       :", bool(args.resume))
    print("=" * 124)

    verify_upstream()

    r31b1 = import_module(R31B1_SCRIPT, "r31b2_r31b1_main")
    panel = r31b1.load_panel()
    if len(panel) != EXPECTED_STATES:
        raise RuntimeError("Frozen nine-state panel drift.")

    r22 = import_module(R22_SCRIPT, "r31b2_r22_preflight")
    bitorder, k2a_path = resolve_k2a_bitorder(r22)

    final_path = out / "R31B2_FINAL_PRE_GT_LOCK.json"
    if args.resume and final_path.is_file():
        final = json.loads(final_path.read_text(encoding="utf-8"))
        if final.get("status") == "PASS_R31B2_800CASE_PRE_GT_LOCK_COMPLETE":
            print("R31B2 already complete.")
            print("Final lock SHA256:", sha256_file(final_path))
            print("NEXT:", final.get("next"))
            return 0

    if out.exists() and not args.resume:
        raise RuntimeError(
            f"Output exists. Use --resume only to continue the exact run: {out}"
        )
    out.mkdir(parents=True, exist_ok=True)

    rgb_lock_path = out / "R31B2_800CASE_RGB_MANIFEST_PRE_GT.csv"
    if args.resume and rgb_lock_path.is_file():
        rgb = pd.read_csv(rgb_lock_path, low_memory=False)
        if len(rgb) != EXPECTED_EVAL_CASES:
            raise RuntimeError("Existing RGB lock size drift.")
        for c in rgb.columns:
            if "gt" in c.lower():
                raise RuntimeError(f"GT column in PRE-GT RGB lock: {c}")
        for _, row in tqdm(
            rgb.iterrows(),
            total=len(rgb),
            desc="R31B2 revalidate 800 RGB",
            unit="image",
            dynamic_ncols=True,
        ):
            p = Path(str(row["image_path"]))
            if sha256_file(p) != str(row["image_sha256"]):
                raise RuntimeError(f"RGB SHA drift: {p}")
    else:
        rgb = load_eval_rgb_lock(out)

    print("\nPRE-GT COHORT LOCK: PASS")
    print("  Physical cases       :", len(rgb))
    print("  RGB manifest SHA256  :", sha256_file(rgb_lock_path))
    print("  K2A bitorder         :", bitorder)
    print("  K2A script           :", k2a_path)
    print("  GT read/hash/decode  : NO / NO / NO")

    for idx, state in panel.iterrows():
        if args.resume and validate_state_artifact(out, idx, state, rgb):
            print(
                f"[RESUME] state {idx+1}/{EXPECTED_STATES} "
                f"{state['model_state_id']}: PASS, skipping"
            )
            continue

        print(
            f"[CUDA ISOLATION] launching state {idx+1}/{EXPECTED_STATES}: "
            f"{state['model_state_id']}"
        )
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--output-dir",
            str(out),
            "--_state-worker",
            str(idx),
            "--_rgb-csv",
            str(rgb_lock_path),
            "--_bitorder",
            bitorder,
        ]
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            raise RuntimeError(
                f"R31B2 state worker failed idx={idx} "
                f"state={state['model_state_id']} returncode={proc.returncode}"
            )
        if not validate_state_artifact(out, idx, state, rgb):
            raise RuntimeError(
                f"Worker success but artifact invalid: {state['model_state_id']}"
            )

    pred_inv, npz_paths = build_prediction_inventory(out, panel, rgb)
    pred_inv_path = out / "R31B2_STATE_PREDICTION_INVENTORY_PRE_GT.csv"

    print("\nPREDICTION LOCK: PASS")
    print("  states               : 9/9")
    print("  model-cases          :", EXPECTED_MODEL_CASES)
    print("  prediction inventory :", sha256_file(pred_inv_path))
    print("  GT read/hash/decode  : NO / NO / NO")

    extract_semantics(
        out=out,
        rgb=rgb,
        panel=panel,
        npz_paths=npz_paths,
        bitorder=bitorder,
        allow_download=args.allow_download,
        device_name=args.device,
        resume=args.resume,
    )

    sp = semantic_paths(out)

    inventory = make_inventory(out)
    inv_path = out / "R31B2_SHA256_INVENTORY_PRE_GT.csv"
    atomic_csv(inventory, inv_path)

    final = {
        "status": "PASS_R31B2_800CASE_PRE_GT_LOCK_COMPLETE",
        "version": VERSION,
        "third_action": THIRD_ACTION,
        "selected_learning_rate": SELECTED_LR,
        "physical_cases": EXPECTED_EVAL_CASES,
        "model_states": EXPECTED_STATES,
        "model_cases": EXPECTED_MODEL_CASES,
        "R31B0_final_sha256": EXPECTED_R31B0_FINAL_SHA256,
        "R31B1_script_sha256": EXPECTED_R31B1_SCRIPT_SHA256,
        "R31B1_selected_lr_lock_sha256": EXPECTED_R31B1_SELECTED_SHA256,
        "R31B1_final_sha256": EXPECTED_R31B1_FINAL_SHA256,
        "rgb_manifest": str(rgb_lock_path),
        "rgb_manifest_sha256": sha256_file(rgb_lock_path),
        "prediction_inventory": str(pred_inv_path),
        "prediction_inventory_sha256": sha256_file(pred_inv_path),
        "semantic_lock": str(sp["lock"]),
        "semantic_lock_sha256": sha256_file(sp["lock"]),
        "feature_table": str(sp["table"]),
        "feature_table_sha256": sha256_file(sp["table"]),
        "evaluation_gt_read_hash_decode": [False, False, False],
        "external_data_access": False,
        "memo_tuning_after_r31b1": False,
        "next": "R31B3_FIRST_800CASE_GT_REVEAL_AND_THREE_ACTION_LOAO",
    }
    atomic_json(final_path, final)

    print("\n" + "=" * 124)
    print("FINAL STATUS : PASS_R31B2_800CASE_PRE_GT_LOCK_COMPLETE")
    print("800-case GT read/hash/decode : NO / NO / NO")
    print("MEMO LR                      :", SELECTED_LR, "(frozen)")
    print("Prediction states            : 9/9")
    print("Semantic model-cases         :", EXPECTED_MODEL_CASES)
    print("Final lock SHA256            :", sha256_file(final_path))
    print("Output                       :", out)
    print("NEXT                         : R31B3_FIRST_800CASE_GT_REVEAL_AND_THREE_ACTION_LOAO")
    print("=" * 124)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
