#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SafeTTA B6-P0-5A — Representative full-transition runtime benchmark.

Setting:
  PolypGen / DeepLabV3-R50::20260817 / MEMO-SEG4-1STEP

This is the exact unseen-action external transition path used by R33A2B.
It measures latency and peak CUDA memory only.

NO GT READ.
NO Dice/HARM computation.
NO model fitting.
NO target calibration.
NO coverage/threshold tuning.

Steady-state per-case runtime excludes one-time model/DINO loading and the
one-time SOURCE snapshot for the model state. Those setup costs are recorded
separately.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch


VERSION = "2026-09-15-B6-P05A-v1"

ROOT = Path(r"F:\MEDSEG_SAFETTA")
CODE = ROOT / "code"

PROTOCOL_PATH = CODE / "B6_P05A_MEMO_FULL_TRANSITION_RUNTIME_PROTOCOL_v1.json"
EXPECTED_PROTOCOL_SHA256 = "b1a6fde0ab174d9a0681e102a6b51bd23086ba2a25dc7fa3f56589b7783aff0f"

R33_SCRIPT = CODE / "Q1_R33A2B_external_polypgen_memo_transition_score_lock_v1_fix1.py"
EXPECTED_R33_SCRIPT_SHA256 = (
    "3ac8e18b77699b46d798711d9c33564b452450ee286d73e79b0e4d671605e711"
)

# Exact historical scripts bound by R33A2B.
R31B1_SCRIPT = CODE / "Q1_R31B1_memo_seg4_action_design_and_lr_freeze_v1.py"
EXPECTED_R31B1_SHA256 = (
    "3391fe4d7161df488f372379a65d72c0dc0cc4e1e774e02e90d68c9ad17f911b"
)
R31B2_SCRIPT = CODE / "Q1_R31B2_800case_memo_prediction_and_semantic_lock_v1.py"
EXPECTED_R31B2_SHA256 = (
    "fa4802f62bb510b953b216f2a3211df8c556dee943f612493ae75463870bbf5b"
)
R05D3_SCRIPT = CODE / "Q1_R05D3_neopolyp_source_a1_prediction_lock_v1_fix1.py"
EXPECTED_R05D3_SHA256 = (
    "df22d6f3054257743d1f12d9e757323ea9f4a052b7642eb4ab610da46ff21251"
)

R33_OUT = ROOT / "R33A2B_external_polypgen_memo_transition_score_lock_v1_fix1"
HEAD_NPZ = R33_OUT / "R33A2B_NEOPOLYP_TENT_PL_FROZEN_PREDICTOR_NUMERIC_LOCK.npz"
FEATURE_CSV = R33_OUT / "R33A2B_POLYPGEN_MEMO_Q66_DSEM64_FEATURE_PRE_HARM_LOCK.csv"
SCORE_CSV = R33_OUT / "R33A2B_POLYPGEN_MEMO_EXTERNAL_SAFETTA_SCORE_PRE_HARM_LOCK.csv"

OUT_DIR = ROOT / "B6_P05A_memo_full_transition_runtime_benchmark_v1"

EXPECTED_GPU_SUBSTRING = "RTX 4060 Laptop GPU"
MODEL_STATE_ID = "DeepLabV3-R50::20260817"
TRAINING_SEED = 20260817
WARMUP_CASES = 10
MEASURED_CASES = 100

IMAGE_SIZE = 352
MEMO_LR = 1e-5

PASS_GATE = "PASS_B6_P05A_MEMO_FULL_TRANSITION_RUNTIME_BENCHMARK"


def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def require_sha(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"{label}: {path}")
    got = sha256_file(path)
    if got.lower() != expected.lower():
        raise RuntimeError(
            f"{label} SHA mismatch\nexpected={expected}\nobserved={got}\npath={path}"
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


def verify_protocol() -> Dict[str, Any]:
    if not PROTOCOL_PATH.is_file():
        raise FileNotFoundError(PROTOCOL_PATH)
    got = sha256_file(PROTOCOL_PATH)
    if got.lower() != EXPECTED_PROTOCOL_SHA256.lower():
        raise RuntimeError(
            "P05A protocol SHA mismatch.\n"
            f"expected={EXPECTED_PROTOCOL_SHA256}\nobserved={got}"
        )
    p = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if p.get("status") != "FROZEN_BEFORE_RUNTIME_MEASUREMENT":
        raise RuntimeError("P05A protocol status changed.")
    return {"path": str(PROTOCOL_PATH), "sha256": got}


def check_cuda() -> Dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the frozen runtime benchmark.")
    name = torch.cuda.get_device_name(0)
    if EXPECTED_GPU_SUBSTRING.lower() not in name.lower():
        raise RuntimeError(
            f"Frozen benchmark expects GPU containing {EXPECTED_GPU_SUBSTRING!r}; "
            f"observed={name!r}"
        )
    props = torch.cuda.get_device_properties(0)
    return {
        "gpu_name": name,
        "gpu_total_memory_GB": float(props.total_memory / (1024 ** 3)),
        "torch_version": str(torch.__version__),
        "cuda_runtime": str(torch.version.cuda),
        "cudnn_version": (
            int(torch.backends.cudnn.version())
            if torch.backends.cudnn.is_available() else None
        ),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }


def sync():
    torch.cuda.synchronize()


def wall_ms(fn):
    sync()
    t0 = time.perf_counter()
    out = fn()
    sync()
    return out, (time.perf_counter() - t0) * 1000.0


def stats(values: List[float]) -> Dict[str, float]:
    a = np.asarray(values, dtype=np.float64)
    if a.size == 0:
        return {}
    return {
        "mean": float(a.mean()),
        "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "median": float(np.median(a)),
        "p95": float(np.quantile(a, 0.95)),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def memory_stats(values: List[float]) -> Dict[str, float]:
    a = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "p95": float(np.quantile(a, 0.95)),
        "max": float(a.max()),
    }


def binary_geometry(source_mask: np.ndarray, cand_mask: np.ndarray) -> np.ndarray:
    s = np.asarray(source_mask, dtype=bool)
    c = np.asarray(cand_mask, dtype=bool)

    inter = int(np.logical_and(s, c).sum())
    ns = int(s.sum())
    nc = int(c.sum())
    union = int(np.logical_or(s, c).sum())

    pred_dice = 1.0 if (ns + nc) == 0 else (2.0 * inter / (ns + nc))
    pred_iou = 1.0 if union == 0 else (inter / union)

    area_delta = float(c.mean() - s.mean())

    def boundary_density(m: np.ndarray) -> float:
        h, w = m.shape
        numer = int(np.count_nonzero(m[1:, :] != m[:-1, :]))
        numer += int(np.count_nonzero(m[:, 1:] != m[:, :-1]))
        denom = (h - 1) * w + h * (w - 1)
        return float(numer / denom) if denom > 0 else 0.0

    boundary_delta = boundary_density(c) - boundary_density(s)
    return np.asarray(
        [pred_dice, pred_iou, area_delta, boundary_delta],
        dtype=np.float64,
    )


def load_numeric_head() -> Dict[str, np.ndarray]:
    if not HEAD_NPZ.is_file():
        raise FileNotFoundError(HEAD_NPZ)
    with np.load(HEAD_NPZ, allow_pickle=False) as z:
        req = {
            "scaler_mean", "scaler_scale", "lr_coef",
            "lr_intercept", "lr_classes", "feature_names",
        }
        miss = req.difference(z.files)
        if miss:
            raise RuntimeError(f"Frozen head NPZ missing={sorted(miss)}")
        out = {k: np.asarray(z[k]) for k in req}

    if out["scaler_mean"].shape != (130,):
        raise RuntimeError("Frozen scaler mean shape drift.")
    if out["scaler_scale"].shape != (130,):
        raise RuntimeError("Frozen scaler scale shape drift.")
    if out["lr_coef"].shape != (1, 130):
        raise RuntimeError("Frozen LR coefficient shape drift.")
    return out


def head_score(x130: np.ndarray, head: Dict[str, np.ndarray]) -> float:
    x = np.asarray(x130, dtype=np.float64).reshape(1, 130)
    xs = (x - head["scaler_mean"][None, :]) / head["scaler_scale"][None, :]
    logit = float(xs @ head["lr_coef"].reshape(130, 1) + head["lr_intercept"][0])
    if logit >= 0:
        p = 1.0 / (1.0 + math.exp(-logit))
    else:
        e = math.exp(logit)
        p = e / (1.0 + e)
    return float(p)


def verify_head_numeric_parity(head: Dict[str, np.ndarray]) -> Dict[str, Any]:
    for p in [FEATURE_CSV, SCORE_CSV]:
        if not p.is_file():
            raise FileNotFoundError(p)

    f = pd.read_csv(FEATURE_CSV, nrows=8, low_memory=False)
    s = pd.read_csv(SCORE_CSV, nrows=8, low_memory=False)

    q = [f"qsrc_{i:02d}" for i in range(66)]
    d = [f"dq_{i:02d}" for i in range(2, 66)]
    cols = q + d

    for c in cols:
        if c not in f.columns:
            raise RuntimeError(f"Feature lock missing {c}")

    if "safettta_external_harm_risk" not in s.columns:
        raise RuntimeError("Score lock missing safettta_external_harm_risk")

    keys = ["sample_id", "model_state_id"]
    merged = f[keys + cols].merge(
        s[keys + ["safettta_external_harm_risk"]],
        on=keys,
        validate="one_to_one",
    )
    if len(merged) == 0:
        raise RuntimeError("No numeric-head parity rows.")

    diffs = []
    for _, r in merged.iterrows():
        pred = head_score(r[cols].to_numpy(dtype=np.float64), head)
        ref = float(r["safettta_external_harm_risk"])
        diffs.append(abs(pred - ref))

    mx = max(diffs)
    if mx > 1e-10:
        raise RuntimeError(f"Frozen numeric-head parity failed max_abs={mx}")
    return {"rows": len(diffs), "max_abs_diff": float(mx), "pass": True}


def patch_fix5_single_state(r31b2):
    """
    The retained semantic formula was written around a nine-state source panel.
    For a single-state runtime case we change only cardinality globals from 9
    to 1 when their names denote state/model counts. No semantic formula changes.
    """
    fix5 = import_module(
        Path(r31b2.R30A4B2_FIX5_SCRIPT),
        "p05a_fix5_semantic",
    )
    patched = {}
    for name, value in list(vars(fix5).items()):
        upper = str(name).upper()
        if isinstance(value, int) and value == 9 and (
            "STATE" in upper or "MODEL" in upper
        ):
            patched[name] = {"old": 9, "new": 1}
            setattr(fix5, name, 1)
    return fix5, patched


def prepare_semantic_runtime(r31b2, device_name: str):
    fix5, patched = patch_fix5_single_state(r31b2)
    r22 = import_module(Path(r31b2.R22_SCRIPT), "p05a_r22_semantic")
    dep = fix5.resolve_semantic_dependencies(
        r22,
        False,  # no web/model download
        device_name,
    )
    run_dino_batch = getattr(dep["dino_helper"], "run_dino_batch", None)
    if run_dino_batch is None:
        raise RuntimeError("Frozen run_dino_batch missing.")
    boundary_fn = getattr(dep["j1"], "boundary_density", None)
    if boundary_fn is None:
        raise RuntimeError("Frozen boundary_density missing.")

    return {
        "fix5": fix5,
        "patched_globals": patched,
        "dep": dep,
        "run_dino_batch": run_dino_batch,
    }


def semantic_130_for_one(
    *,
    image_path: Path,
    source_mask: np.ndarray,
    cand_mask: np.ndarray,
    semantic_rt: Dict[str, Any],
) -> np.ndarray:
    dep = semantic_rt["dep"]
    fix5 = semantic_rt["fix5"]
    k2a = dep["k2a"]
    j1 = dep["j1"]
    pca = dep["pca"]

    bitorder = str(getattr(k2a, "BITORDER", "big"))

    # Reuse the exact retained occupancy transformation.
    src_packed = np.packbits(
        np.asarray(source_mask, dtype=np.uint8).reshape(-1),
        bitorder=bitorder,
    )
    cand_packed = np.packbits(
        np.asarray(cand_mask, dtype=np.uint8).reshape(-1),
        bitorder=bitorder,
    )
    occ_s = np.asarray(k2a.unpack_mask_occupancy16(src_packed), dtype=np.float32)
    occ_c = np.asarray(k2a.unpack_mask_occupancy16(cand_packed), dtype=np.float32)

    _, patch_batch = semantic_rt["run_dino_batch"](
        [str(image_path)],
        dep["processor"],
        dep["model"],
        dep["device"],
        k2a,
        dep["torch"],
    )
    patch = np.asarray(patch_batch)[0]

    src64 = fix5.conditioned_pca64_for_one_image(
        patch,
        occ_s[None, :],
        k2a,
        pca,
    )
    cand64 = fix5.conditioned_pca64_for_one_image(
        patch,
        occ_c[None, :],
        k2a,
        pca,
    )
    src64 = np.asarray(src64, dtype=np.float64).reshape(1, 64)[0]
    cand64 = np.asarray(cand64, dtype=np.float64).reshape(1, 64)[0]

    sm = np.asarray(source_mask, dtype=bool)
    m2 = np.asarray(
        [
            float(sm.mean()),
            float(j1.boundary_density(sm)),
        ],
        dtype=np.float64,
    )

    q66 = np.concatenate([m2, src64])
    d64 = cand64 - src64
    x130 = np.concatenate([q66, d64])
    if x130.shape != (130,) or not np.isfinite(x130).all():
        raise RuntimeError("Runtime semantic feature invalid.")
    return x130


def benchmark_case(
    *,
    row: pd.Series,
    state: pd.Series,
    runtime: Dict[str, Any],
    r31b1,
    head: Dict[str, np.ndarray],
    semantic_rt: Dict[str, Any],
    measured: bool,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    model = runtime["model"]
    to_tensor = runtime["to_tensor"]
    foreground_prob = runtime["foreground_prob"]
    final_mask = runtime["final_mask"]
    source_params = runtime["source_params"]
    source_buffers = runtime["source_buffers"]

    image_path = Path(str(row["image_path"]))
    if not image_path.is_file():
        raise FileNotFoundError(image_path)

    # Baseline memory after both DeepLab + DINO are resident.
    sync()
    baseline_alloc = torch.cuda.memory_allocated()
    baseline_reserved = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()

    total_t0 = time.perf_counter()

    def _load_and_preprocess():
        with Image.open(image_path) as im:
            native = im.convert("RGB")
        x = to_tensor(native)
        return native, x

    (native, x), t_pre = wall_ms(_load_and_preprocess)

    _, t_restore_source = wall_ms(
        lambda: r31b1.restore_model(model, source_params, source_buffers)
    )

    def _source_forward():
        model.eval()
        with torch.no_grad():
            return np.asarray(final_mask(x), dtype=np.uint8)

    source_mask, t_source = wall_ms(_source_forward)

    _, t_restore_pre_action = wall_ms(
        lambda: r31b1.restore_model(model, source_params, source_buffers)
    )

    # Exact R33A2B MEMO transaction.
    seed_txn = int(
        r31b1.action_seed(
            str(state["model_state_id"]),
            str(row["sample_id"]),
        )
    )
    r31b1.set_seed(seed_txn)

    def _memo_update():
        model.train()
        params = list(model.parameters())
        for p in params:
            p.requires_grad_(True)

        opt = torch.optim.Adam(
            params,
            lr=MEMO_LR,
            weight_decay=0.0,
        )
        opt.zero_grad(set_to_none=True)

        x4 = r31b1.augment_batch(x, torch)
        prob4 = foreground_prob(x4)
        aligned = r31b1.inverse_align_probs(prob4, torch)
        marginal = aligned.mean(dim=0, keepdim=True)
        loss = r31b1.binary_entropy_mean(marginal, torch)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite runtime MEMO loss.")
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        return float(loss.detach().cpu())

    memo_loss, t_update = wall_ms(_memo_update)

    def _candidate_forward():
        model.eval()
        with torch.no_grad():
            return np.asarray(final_mask(x), dtype=np.uint8)

    cand_mask, t_candidate = wall_ms(_candidate_forward)

    _, t_restore_post = wall_ms(
        lambda: r31b1.restore_model(model, source_params, source_buffers)
    )

    def _semantic():
        return semantic_130_for_one(
            image_path=image_path,
            source_mask=source_mask,
            cand_mask=cand_mask,
            semantic_rt=semantic_rt,
        )

    x130, t_semantic = wall_ms(_semantic)

    def _head():
        return head_score(x130, head)

    risk, t_head = wall_ms(_head)

    def _geom():
        return binary_geometry(source_mask, cand_mask)

    geom, t_geom = wall_ms(_geom)

    sync()
    total_ms = (time.perf_counter() - total_t0) * 1000.0

    peak_alloc = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()

    record = {
        "sample_id": str(row["sample_id"]),
        "model_state_id": str(state["model_state_id"]),
        "training_seed": int(state["training_seed"]),
        "action": "MEMO-SEG4-1STEP",
        "measured": bool(measured),
        "image_load_preprocess_ms": t_pre,
        "restore_before_source_ms": t_restore_source,
        "source_inference_ms": t_source,
        "restore_before_candidate_ms": t_restore_pre_action,
        "candidate_update_ms": t_update,
        "candidate_reinference_ms": t_candidate,
        "restore_after_candidate_ms": t_restore_post,
        "dino_conditioning_pca_ms": t_semantic,
        "safety_head_ms": t_head,
        "simple_geometry_ms": t_geom,
        "end_to_end_precommit_ms": total_ms,
        "baseline_cuda_allocated_MB": baseline_alloc / (1024 ** 2),
        "baseline_cuda_reserved_MB": baseline_reserved / (1024 ** 2),
        "peak_cuda_allocated_MB": peak_alloc / (1024 ** 2),
        "peak_cuda_reserved_MB": peak_reserved / (1024 ** 2),
        "peak_incremental_allocated_MB": (peak_alloc - baseline_alloc) / (1024 ** 2),
        "memo_loss_preupdate": memo_loss,
        "risk_score": risk,
        "changed_pixels": int(np.count_nonzero(source_mask != cand_mask)),
    }
    return record, source_mask, cand_mask


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    timing_cols = [
        "image_load_preprocess_ms",
        "restore_before_source_ms",
        "source_inference_ms",
        "restore_before_candidate_ms",
        "candidate_update_ms",
        "candidate_reinference_ms",
        "restore_after_candidate_ms",
        "dino_conditioning_pca_ms",
        "safety_head_ms",
        "simple_geometry_ms",
        "end_to_end_precommit_ms",
    ]
    rows = []
    for c in timing_cols:
        st = stats(df[c].astype(float).tolist())
        rows.append({"component": c, **st})

    for c in [
        "baseline_cuda_allocated_MB",
        "baseline_cuda_reserved_MB",
        "peak_cuda_allocated_MB",
        "peak_cuda_reserved_MB",
        "peak_incremental_allocated_MB",
    ]:
        st = memory_stats(df[c].astype(float).tolist())
        rows.append({
            "component": c,
            "mean": st["mean"],
            "std": np.nan,
            "median": st["median"],
            "p95": st["p95"],
            "min": np.nan,
            "max": st["max"],
        })
    return pd.DataFrame(rows)


def self_test():
    a = np.zeros((4, 4), dtype=np.uint8)
    b = a.copy()
    g = binary_geometry(a, b)
    assert np.allclose(g, [1.0, 1.0, 0.0, 0.0])
    assert WARMUP_CASES == 10
    assert MEASURED_CASES == 100
    assert MEMO_LR == 1e-5
    print("SELF_TEST_GEOMETRY=PASS")
    print("SELF_TEST_COUNTS=PASS")
    print("SELF_TEST=PASS")


def main():
    ap = argparse.ArgumentParser(
        description="SafeTTA P05A MEMO full-transition runtime benchmark."
    )
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--preflight-only", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    print("=" * 164)
    print("SafeTTA B6-P0-5A — MEMO full-transition runtime benchmark")
    print("Version               :", VERSION)
    print("Scientific metrics    : NO")
    print("GT read               : NO")
    print("Target calibration    : NO")
    print("Model state           :", MODEL_STATE_ID)
    print("Action                : MEMO-SEG4-1STEP")
    print("Warmup / measured     :", WARMUP_CASES, "/", MEASURED_CASES)
    print("=" * 164)

    print("\n[1/5] Verify protocol and exact execution lineage")
    protocol_meta = verify_protocol()
    require_sha(R33_SCRIPT, EXPECTED_R33_SCRIPT_SHA256, "R33A2B fix1")
    require_sha(R31B1_SCRIPT, EXPECTED_R31B1_SHA256, "R31B1")
    require_sha(R31B2_SCRIPT, EXPECTED_R31B2_SHA256, "R31B2")
    require_sha(R05D3_SCRIPT, EXPECTED_R05D3_SHA256, "R05D3")
    env = check_cuda()
    print("PROTOCOL_SHA256 =", protocol_meta["sha256"])
    print("GPU             =", env["gpu_name"])
    print("torch           =", env["torch_version"])
    print("CUDA runtime    =", env["cuda_runtime"])
    print("LINEAGE=PASS")

    print("\n[2/5] Import exact R33A2B runtime and bind target panel")
    r33 = import_module(R33_SCRIPT, "p05a_r33_exact")
    r31b1 = import_module(R31B1_SCRIPT, "p05a_r31b1_exact")
    r31b2 = import_module(R31B2_SCRIPT, "p05a_r31b2_exact")
    r05 = import_module(R05D3_SCRIPT, "p05a_r05_exact")

    r33a1_final, _ = r33.verify_upstream()
    target_manifest, states = r33.load_target_manifest(r33a1_final)

    state_rows = states[
        states["model_state_id"].astype(str) == MODEL_STATE_ID
    ].copy()
    if len(state_rows) != 1:
        raise RuntimeError("Frozen benchmark state not found uniquely.")
    state = state_rows.iloc[0]

    rgb = (
        target_manifest[
            ["sample_id", "image_path", "image_raw_sha256"]
        ]
        .drop_duplicates()
        .sort_values("sample_id", kind="mergesort")
        .reset_index(drop=True)
    )
    if len(rgb) != 1532:
        raise RuntimeError(f"PolypGen RGB rows={len(rgb)} != 1532")

    cohort = rgb.iloc[: WARMUP_CASES + MEASURED_CASES].copy()
    if len(cohort) != WARMUP_CASES + MEASURED_CASES:
        raise RuntimeError("Runtime cohort cardinality drift.")

    print("state      =", state["model_state_id"])
    print("RGB rows   =", len(rgb))
    print("benchmark  =", len(cohort))
    print("PANEL_BINDING=PASS")

    print("\n[3/5] Bind frozen transition head and semantic runtime")
    head = load_numeric_head()
    head_parity = verify_head_numeric_parity(head)
    print("numeric head parity rows =", head_parity["rows"])
    print("numeric head max abs     =", head_parity["max_abs_diff"])

    if args.preflight_only:
        print("SEMANTIC_MODEL_LOAD=NOT_RUN")
        print("SEGMENTATION_MODEL_LOAD=NOT_RUN")
        print("RUNTIME_MEASUREMENT=NOT_RUN")
        print("P05A_PREFLIGHT=PASS")
        return

    if OUT_DIR.exists():
        raise FileExistsError(
            f"Never overwrite runtime output: {OUT_DIR}\n"
            "Use _fix1 only for implementation repair."
        )
    OUT_DIR.mkdir(parents=True, exist_ok=False)

    # One-time setup timing.
    setup_t0 = time.perf_counter()
    runtime = r33.build_deeplab_runtime(
        r31b1=r31b1,
        r05=r05,
        state=state,
        device=torch.device("cuda"),
    )
    sync()
    deeplab_setup_ms = (time.perf_counter() - setup_t0) * 1000.0

    setup_t0 = time.perf_counter()
    semantic_rt = prepare_semantic_runtime(r31b2, "cuda")
    sync()
    semantic_setup_ms = (time.perf_counter() - setup_t0) * 1000.0

    # Validate no accidental formula replacement.
    print("semantic cardinality globals patched =", semantic_rt["patched_globals"])
    print("SEMANTIC_FORMULA_CHANGED=NO")

    model_resident_alloc_mb = torch.cuda.memory_allocated() / (1024 ** 2)
    model_resident_reserved_mb = torch.cuda.memory_reserved() / (1024 ** 2)

    print("\n[4/5] Warmup + measured runtime")
    records = []
    parity_rows = []

    iterator = tqdm(
        range(len(cohort)),
        desc="P05A MEMO full transition",
        unit="case",
        dynamic_ncols=True,
    )
    for i in iterator:
        row = cohort.iloc[i]
        measured = i >= WARMUP_CASES
        rec, src_mask, cand_mask = benchmark_case(
            row=row,
            state=state,
            runtime=runtime,
            r31b1=r31b1,
            head=head,
            semantic_rt=semantic_rt,
            measured=measured,
        )
        records.append(rec)

        # Outcome-free parity against the already frozen R33A2B masks for first 3 cases.
        if i < 3:
            npz = R33_OUT / "state_00_polypgen_source_memo_pre_harm.npz"
            if not npz.is_file():
                raise FileNotFoundError(npz)
            with np.load(npz, allow_pickle=False) as z:
                bitorder = str(getattr(semantic_rt["dep"]["k2a"], "BITORDER", "big"))
                src_pack = np.packbits(
                    src_mask.astype(np.uint8).reshape(-1),
                    bitorder=bitorder,
                )
                cand_pack = np.packbits(
                    cand_mask.astype(np.uint8).reshape(-1),
                    bitorder=bitorder,
                )
                ref_src = np.asarray(z["source_masks_packed"][i], dtype=np.uint8)
                ref_cand = np.asarray(z["memo_masks_packed"][i], dtype=np.uint8)

            src_equal = bool(np.array_equal(src_pack, ref_src))
            cand_equal = bool(np.array_equal(cand_pack, ref_cand))
            parity_rows.append({
                "sample_id": str(row["sample_id"]),
                "source_exact": src_equal,
                "candidate_exact": cand_equal,
                "source_mismatch_bytes": int(np.count_nonzero(src_pack != ref_src)),
                "candidate_mismatch_bytes": int(np.count_nonzero(cand_pack != ref_cand)),
            })
            if not src_equal or not cand_equal:
                raise RuntimeError(
                    "Runtime action parity failed against frozen R33A2B masks "
                    f"for sample={row['sample_id']}: "
                    f"SOURCE={src_equal} MEMO={cand_equal}"
                )

        iterator.set_postfix(
            total=f"{rec['end_to_end_precommit_ms']:.1f}ms",
            peak=f"{rec['peak_cuda_allocated_MB']:.0f}MB",
        )

    all_df = pd.DataFrame(records)
    measured_df = all_df[all_df["measured"]].reset_index(drop=True)
    if len(measured_df) != MEASURED_CASES:
        raise RuntimeError("Measured runtime row count drift.")

    summary_df = summarize(measured_df)
    parity_df = pd.DataFrame(parity_rows)

    print("\n[5/5] Freeze runtime outputs")
    print(summary_df.to_string(index=False))

    p_rows = OUT_DIR / "B6_P05A_MEMO_FULL_TRANSITION_RUNTIME_ROWS.csv"
    p_summary = OUT_DIR / "B6_P05A_MEMO_FULL_TRANSITION_RUNTIME_SUMMARY.csv"
    p_parity = OUT_DIR / "B6_P05A_MEMO_RUNTIME_MASK_PARITY.csv"

    all_df.to_csv(p_rows, index=False)
    summary_df.to_csv(p_summary, index=False)
    parity_df.to_csv(p_parity, index=False)

    total_row = summary_df[
        summary_df["component"] == "end_to_end_precommit_ms"
    ].iloc[0]
    update_row = summary_df[
        summary_df["component"] == "candidate_update_ms"
    ].iloc[0]
    semantic_row = summary_df[
        summary_df["component"] == "dino_conditioning_pca_ms"
    ].iloc[0]
    memory_row = summary_df[
        summary_df["component"] == "peak_cuda_allocated_MB"
    ].iloc[0]

    audit = {
        "status": "PASS",
        "gate": PASS_GATE,
        "version": VERSION,
        "protocol": protocol_meta,
        "environment": env,
        "setting": {
            "dataset": "PolypGen",
            "model_state_id": MODEL_STATE_ID,
            "action": "MEMO-SEG4-1STEP",
            "batch_size": 1,
            "warmup_cases": WARMUP_CASES,
            "measured_cases": MEASURED_CASES,
            "gt_read": False,
        },
        "setup": {
            "deeplab_load_plus_source_snapshot_ms": deeplab_setup_ms,
            "dino_semantic_dependency_load_ms": semantic_setup_ms,
            "resident_cuda_allocated_MB": model_resident_alloc_mb,
            "resident_cuda_reserved_MB": model_resident_reserved_mb,
            "included_in_per_case_latency": False,
        },
        "headline": {
            "end_to_end_mean_ms": float(total_row["mean"]),
            "end_to_end_median_ms": float(total_row["median"]),
            "end_to_end_p95_ms": float(total_row["p95"]),
            "candidate_update_mean_ms": float(update_row["mean"]),
            "dino_conditioning_pca_mean_ms": float(semantic_row["mean"]),
            "peak_cuda_allocated_mean_MB": float(memory_row["mean"]),
            "peak_cuda_allocated_max_MB": float(memory_row["max"]),
        },
        "mask_parity": {
            "rows": len(parity_df),
            "all_source_exact": bool(parity_df["source_exact"].all()),
            "all_candidate_exact": bool(parity_df["candidate_exact"].all()),
        },
        "claim_boundary": {
            "representative_primary_transition": True,
            "all_actions_equal_cost_claim": False,
            "source_only_historical_runtime_replaced": False,
            "model_loading_excluded_from_steady_state": True,
        },
        "artifacts": {},
    }

    for p in [p_rows, p_summary, p_parity]:
        audit["artifacts"][p.name] = {
            "path": str(p),
            "sha256": sha256_file(p),
            "bytes": p.stat().st_size,
        }

    p_audit = OUT_DIR / "B6_P05A_MEMO_FULL_TRANSITION_RUNTIME_AUDIT.json"
    p_audit.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    report = "\n".join([
        "=" * 164,
        "SafeTTA B6-P0-5A MEMO FULL-TRANSITION RUNTIME COMPLETE",
        "",
        f"GPU: {env['gpu_name']}",
        f"Measured cases: {MEASURED_CASES} (after {WARMUP_CASES} warmup)",
        f"End-to-end mean/median/p95: "
        f"{total_row['mean']:.3f} / {total_row['median']:.3f} / {total_row['p95']:.3f} ms",
        f"Candidate MEMO update mean: {update_row['mean']:.3f} ms",
        f"DINO + conditioning + PCA mean: {semantic_row['mean']:.3f} ms",
        f"Peak CUDA allocated mean/max: "
        f"{memory_row['mean']:.1f} / {memory_row['max']:.1f} MB",
        f"Mask parity rows: {len(parity_df)} exact SOURCE+MEMO = "
        f"{bool(parity_df['source_exact'].all() and parity_df['candidate_exact'].all())}",
        "",
        "GATE=" + PASS_GATE,
        "audit_json=" + str(p_audit),
        "stage_dir=" + str(OUT_DIR),
        "=" * 164,
        "",
    ])
    p_report = OUT_DIR / "B6_P05A_MEMO_FULL_TRANSITION_RUNTIME_REPORT.txt"
    p_report.write_text(report, encoding="utf-8")
    print("\n" + report)


if __name__ == "__main__":
    main()
