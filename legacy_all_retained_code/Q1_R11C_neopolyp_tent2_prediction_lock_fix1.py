#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1-R11C — NeoPolyp A2_TENT_2STEP prediction lock.

Scope: adaptation-intensity generalization, NOT cross-TTA-algorithm generalization.
The two-step action is historically defined in S07-B. This script extends the
same fixed two-step TENT intervention to the frozen PraNet / DeepLabV3-R50 /
SegFormer-B0 panel, while preserving each family's authoritative R05D3 TENT
semantics.

Reads NeoPolyp RGB and frozen R05D3 SOURCE masks for parity only.
Does NOT read R05D3 A1 mask values, NeoPolyp GT, safety scores, or PolypGen data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import random
import sys
import traceback
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
from PIL import Image
from tqdm import tqdm

VERSION = "2026-08-26-Q1-R11C-v1-fix1"
BUILD = "Q1_R11C_NEOPOLYP_TENT2_PREDICTION_LOCK_FIX1"
ROOT = Path(r"F:\MEDSEG_SAFETTA")

R11B_LOCK = ROOT / "outputs" / "Q1_R11B_candidate_action_semantic_lineage_audit_fix1_v1" / "R11B_CANDIDATE_ACTION_LOCK.json"
EXPECTED_R11B_DECISION = "R11B_CANDIDATE_ACTION_SEMANTICS_FROZEN"

R05D3_SCRIPT = ROOT / "code" / "Q1_R05D3_neopolyp_source_a1_prediction_lock_v1_fix1.py"
EXPECTED_R05D3_SCRIPT_SHA256 = "df22d6f3054257743d1f12d9e757323ea9f4a052b7642eb4ab610da46ff21251"
R05D3_DIR = ROOT / "outputs" / "Q1_R05D3_neopolyp_source_a1_prediction_lock_v1_fix1"
R05D3_LOCK = R05D3_DIR / "Q1_R05D3_NEOPOLYP_SOURCE_A1_PREDICTION_LOCK.json"
EXPECTED_R05D3_LOCK_SHA256 = "3da08e5ed0780a00788a297b1127d9eb460224326b11e9e30eefce2b3fb48c9a"
EXPECTED_R05D3_DECISION = "NEOPOLYP_SOURCE_A1_PREDICTIONS_LOCKED"

S07B_SCRIPT = ROOT / "code" / "S07_B_build_pranet_source_side_counterfactual_utility_dataset_v1.py"
POLYPGEN_LOCK = ROOT / "outputs" / "Q1_R10L3C_polypgen_gt_reveal_prospective_external_evaluation_fix1_v1" / "R10L3C_POLYPGEN_PROSPECTIVE_EXTERNAL_EVALUATION_LOCK.json"
EXPECTED_POLYPGEN_DECISION = "POLYPGEN_PROSPECTIVE_EXTERNAL_EVALUATION_COMPLETE_NO_RETUNING"

OUTPUT_DIR = ROOT / "outputs" / "Q1_R11C_neopolyp_tent2_prediction_lock_fix1_v1"

EXPECTED_CASES = 1000
EXPECTED_STATES = 9
EXPECTED_MODEL_CASES = 9000
MASK_H = 352
MASK_W = 352
PACKED_BYTES = 15488
ACTION = "A2_TENT_2STEP"
TENT_LR = 1e-3
TENT_WEIGHT_DECAY = 0.0
TENT_STEPS = 2
DECISION = "NEOPOLYP_A2_TENT_2STEP_PREDICTIONS_LOCKED_FOR_ACTION_INTENSITY_EVALUATION"

INDEX_FIELDS = [
    "row_index", "sample_id", "image_path", "image_raw_sha256",
    "model_family", "model_state_id", "training_seed", "checkpoint_sha256",
    "source_foreground_pixels", "a2_foreground_pixels",
    "source_a2_changed_pixels", "step1_entropy_loss", "step2_entropy_loss",
]


def sha256_file(path: Path, chunk: int = 16 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def validate_sha(path: Path, expected: str, label: str):
    if not path.exists():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual.lower() != expected.lower():
        raise RuntimeError(f"{label} SHA mismatch: expected={expected} actual={actual}")
    return actual


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def seed_runtime(seed: int):
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def state_token(state):
    return state["model_family"].lower().replace("-", "_").replace(" ", "_") + f"_seed{state['training_seed']}"


def out_state_paths(state_dir: Path, state):
    t = state_token(state)
    return (
        state_dir / f"{t}_a2_predictions.npz",
        state_dir / f"{t}_a2_index.csv",
        state_dir / f"{t}_a2_lock.json",
    )


def load_frozen_source(r05d3, state):
    pred, idx, side = r05d3.state_paths(R05D3_DIR / "state_predictions", state)
    if not (pred.exists() and idx.exists() and side.exists()):
        raise FileNotFoundError(f"Missing R05D3 state assets: {state['model_state_id']}")
    meta = json.loads(side.read_text(encoding="utf-8"))
    validate_sha(pred, meta["prediction_npz_sha256"], "R05D3 state NPZ")
    with np.load(pred, allow_pickle=False) as z:
        if "source_masks_packed" not in z.files:
            raise RuntimeError("source_masks_packed missing")
        # Deliberately never index the frozen A1 array.
        source = np.asarray(z["source_masks_packed"], dtype=np.uint8).copy()
    if source.shape != (EXPECTED_CASES, PACKED_BYTES):
        raise RuntimeError(f"Frozen SOURCE shape changed: {source.shape}")
    return source, pred


def compare_source(r05d3, z_source, frozen_source_row, state_id, i, sample_id):
    sm = r05d3.logit_to_mask(z_source)
    ref = r05d3.unpack_mask(frozen_source_row)
    if not np.array_equal(sm, ref):
        raise RuntimeError(f"SOURCE parity mismatch state={state_id} row={i} sample={sample_id}")
    return sm


def run_pranet(r05d3, state, targets, device, frozen_source):
    import torch
    helper = r05d3.import_module(r05d3.PRANET_HELPER, f"r11c_pranet_{state['training_seed']}")
    training = helper.import_training_helper()
    seed = int(state["training_seed"])
    seed_runtime(seed)
    model = helper.load_model(training, seed, device)
    params = helper.configure_tent(model)
    source_values = helper.snapshot_params(params)
    a2 = np.empty((EXPECTED_CASES, PACKED_BYTES), dtype=np.uint8)
    rows = []
    for i, target in tqdm(enumerate(targets), total=EXPECTED_CASES, desc=f"R11C PraNet {seed}", unit="img", dynamic_ncols=True):
        with Image.open(target["image_path"]) as im:
            x = helper.image_to_model_tensor(training, im.convert("RGB")).to(device, non_blocking=True)
        helper.restore_params(params, source_values)
        model.eval()
        with torch.no_grad():
            zsrc_t = helper.final_logit(model, x).detach()
        zsrc = zsrc_t[0, 0].float().cpu().numpy()
        sm = compare_source(r05d3, zsrc, frozen_source[i], state["model_state_id"], i, target["sample_id"])
        helper.restore_params(params, source_values)
        model.train()
        opt = torch.optim.Adam(params, lr=TENT_LR, weight_decay=TENT_WEIGHT_DECAY)
        losses = []
        for _ in range(TENT_STEPS):
            opt.zero_grad(set_to_none=True)
            z = helper.final_logit(model, x)
            loss = helper.mean_binary_entropy(z)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite PraNet entropy")
            loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
            losses.append(float(loss.detach().cpu()))
        with torch.no_grad():
            za2_t = helper.final_logit(model, x).detach()
        za2 = za2_t[0, 0].float().cpu().numpy()
        am = r05d3.logit_to_mask(za2)
        a2[i] = r05d3.pack_mask(am)
        rows.append(_row(target, state, i, sm, am, losses))
        del x, zsrc_t, zsrc, sm, opt, z, loss, za2_t, za2, am
    helper.restore_params(params, source_values)
    del model, params, source_values
    if device.type == "cuda": torch.cuda.empty_cache()
    return a2, rows


def run_deeplab(r05d3, state, targets, device, frozen_source):
    import torch
    helper = r05d3.import_module(r05d3.DEEPLAB_HELPER, f"r11c_deeplab_{state['training_seed']}")
    training = helper.import_training_helper()
    helper.seed_everything(20260817)
    seed = int(state["training_seed"])
    model = helper.load_model(training, seed, device)
    params, _, _, unsafe_bn, _, dropout = helper.configure_singleton_safe_tent(model)
    source_values = helper.snapshot_params(params)
    a2 = np.empty((EXPECTED_CASES, PACKED_BYTES), dtype=np.uint8)
    rows = []
    for i, target in tqdm(enumerate(targets), total=EXPECTED_CASES, desc=f"R11C DeepLab {seed}", unit="img", dynamic_ncols=True):
        with Image.open(target["image_path"]) as im:
            x = helper.image_to_model_tensor(training, im.convert("RGB")).to(device, non_blocking=True)
        helper.restore_params(params, source_values)
        model.eval()
        with torch.no_grad():
            zsrc_t = training.deeplab_logits(model, x).detach()
        zsrc = zsrc_t[0, 0].float().cpu().numpy()
        sm = compare_source(r05d3, zsrc, frozen_source[i], state["model_state_id"], i, target["sample_id"])
        helper.restore_params(params, source_values)
        helper.set_singleton_safe_tent_mode(model, unsafe_bn, dropout)
        opt = torch.optim.Adam(params, lr=TENT_LR, weight_decay=TENT_WEIGHT_DECAY)
        losses = []
        for _ in range(TENT_STEPS):
            opt.zero_grad(set_to_none=True)
            z = training.deeplab_logits(model, x)
            loss = helper.mean_binary_entropy(z)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite DeepLab entropy")
            loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
            losses.append(float(loss.detach().cpu()))
        with torch.no_grad():
            za2_t = training.deeplab_logits(model, x).detach()
        za2 = za2_t[0, 0].float().cpu().numpy()
        am = r05d3.logit_to_mask(za2)
        a2[i] = r05d3.pack_mask(am)
        rows.append(_row(target, state, i, sm, am, losses))
        del x, zsrc_t, zsrc, sm, opt, z, loss, za2_t, za2, am
    helper.restore_params(params, source_values)
    del model, params, source_values, unsafe_bn, dropout
    if device.type == "cuda": torch.cuda.empty_cache()
    return a2, rows


def run_segformer(r05d3, r03, state, targets, device, frozen_source, context=None):
    if context is None:
        context = r03.build_segformer_context(device)
    torch, nn, F, SegformerForSemanticSegmentation, config, mean, std = context
    seed = int(state["training_seed"])
    seed_runtime(seed)
    model = r03.load_segformer_state(seed, device, SegformerForSemanticSegmentation, config, torch)
    params, _, bn_modules, bn_original_track, dropout_modules, _ = r03.configure_segformer_tent(model, nn)
    source_values = r03.snapshot_params(params)
    a2 = np.empty((EXPECTED_CASES, PACKED_BYTES), dtype=np.uint8)
    rows = []
    for i, target in tqdm(enumerate(targets), total=EXPECTED_CASES, desc=f"R11C SegFormer {seed}", unit="img", dynamic_ncols=True):
        with Image.open(target["image_path"]) as im:
            x = r03.segformer_tensor(im.convert("RGB"), mean, std, torch).to(device, non_blocking=True)
        r03.restore_params(params, source_values, torch)
        r03.segformer_source_mode(model, bn_modules, bn_original_track, dropout_modules)
        with torch.no_grad():
            _, zsrc_t = r03.segformer_logits_and_z(model, x, F)
        zsrc = zsrc_t[0].detach().float().cpu().numpy()
        sm = compare_source(r05d3, zsrc, frozen_source[i], state["model_state_id"], i, target["sample_id"])
        r03.restore_params(params, source_values, torch)
        r03.segformer_tent_mode(model, bn_modules, dropout_modules)
        opt = torch.optim.Adam(params, lr=TENT_LR, weight_decay=TENT_WEIGHT_DECAY)
        losses = []
        for _ in range(TENT_STEPS):
            opt.zero_grad(set_to_none=True)
            logits, _ = r03.segformer_logits_and_z(model, x, F)
            loss = r03.categorical_entropy(logits, torch)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite SegFormer entropy")
            loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
            losses.append(float(loss.detach().cpu()))
        with torch.no_grad():
            _, za2_t = r03.segformer_logits_and_z(model, x, F)
        za2 = za2_t[0].detach().float().cpu().numpy()
        am = r05d3.logit_to_mask(za2)
        a2[i] = r05d3.pack_mask(am)
        rows.append(_row(target, state, i, sm, am, losses))
        del x, zsrc_t, zsrc, sm, opt, logits, loss, za2_t, za2, am
    r03.restore_params(params, source_values, torch)
    del model, params, source_values, bn_modules, bn_original_track, dropout_modules
    if device.type == "cuda": torch.cuda.empty_cache()
    return a2, rows, context


def _row(target, state, i, sm, am, losses):
    return {
        "row_index": i,
        "sample_id": target["sample_id"],
        "image_path": target["image_path"],
        "image_raw_sha256": target["image_raw_sha256"],
        "model_family": state["model_family"],
        "model_state_id": state["model_state_id"],
        "training_seed": int(state["training_seed"]),
        "checkpoint_sha256": state["checkpoint_sha256"],
        "source_foreground_pixels": int(sm.sum()),
        "a2_foreground_pixels": int(am.sum()),
        "source_a2_changed_pixels": int(np.count_nonzero(sm != am)),
        "step1_entropy_loss": float(losses[0]),
        "step2_entropy_loss": float(losses[1]),
    }


def save_state(r05d3, state_dir, state, a2, rows, source_ref):
    pred, idx, side = out_state_paths(state_dir, state)
    np.savez_compressed(pred, a2_masks_packed=np.asarray(a2, dtype=np.uint8))
    r05d3.write_csv(idx, rows, INDEX_FIELDS)
    meta = {
        "decision": "STATE_COMPLETE",
        "model_family": state["model_family"],
        "model_state_id": state["model_state_id"],
        "training_seed": int(state["training_seed"]),
        "checkpoint_sha256": state["checkpoint_sha256"],
        "action": ACTION,
        "optimizer": "Adam",
        "lr": TENT_LR,
        "weight_decay": TENT_WEIGHT_DECAY,
        "steps": TENT_STEPS,
        "episodic_reset": True,
        "target_cases": EXPECTED_CASES,
        "source_parity_reference_npz": str(source_ref),
        "source_parity_mismatches": 0,
        "r05d3_a1_mask_values_read": False,
        "neopolyp_gt_pixels_decoded": False,
        "safety_scores_read": False,
        "polypgen_data_read": False,
        "prediction_npz_sha256": sha256_file(pred),
        "index_csv_sha256": sha256_file(idx),
    }
    r05d3.write_json(side, meta)
    return pred, idx, side


def load_cached(r05d3, state_dir, state):
    pred, idx, side = out_state_paths(state_dir, state)
    if not (pred.exists() or idx.exists() or side.exists()):
        return None
    if not (pred.exists() and idx.exists() and side.exists()):
        raise RuntimeError(f"Orphan cache: {state['model_state_id']}")
    meta = json.loads(side.read_text(encoding="utf-8"))
    if meta.get("decision") != "STATE_COMPLETE" or meta.get("action") != ACTION:
        raise RuntimeError("Invalid cached state lock")
    validate_sha(pred, meta["prediction_npz_sha256"], "cached A2 NPZ")
    validate_sha(idx, meta["index_csv_sha256"], "cached A2 index")
    rows, fields = r05d3.read_csv(idx)
    if fields != INDEX_FIELDS or len(rows) != EXPECTED_CASES:
        raise RuntimeError("Cached state index invalid")
    return pred, idx, side, rows


def validate_lineage():
    validate_sha(R05D3_SCRIPT, EXPECTED_R05D3_SCRIPT_SHA256, "R05D3 script")
    validate_sha(R05D3_LOCK, EXPECTED_R05D3_LOCK_SHA256, "R05D3 lock")
    if not R11B_LOCK.exists() or not S07B_SCRIPT.exists() or not POLYPGEN_LOCK.exists():
        raise FileNotFoundError("Required lineage artifact missing")
    r11b = json.loads(R11B_LOCK.read_text(encoding="utf-8"))
    if r11b.get("decision") != EXPECTED_R11B_DECISION:
        raise RuntimeError("R11B decision changed")
    r05 = json.loads(R05D3_LOCK.read_text(encoding="utf-8"))
    if r05.get("decision") != EXPECTED_R05D3_DECISION or int(r05.get("steps", -1)) != 1:
        raise RuntimeError("R05D3 reference lock changed")
    s07 = S07B_SCRIPT.read_text(encoding="utf-8", errors="replace")
    if '("A2_TENT_2STEP", 2)' not in s07:
        raise RuntimeError("Historical A2_TENT_2STEP definition not found")
    pg = json.loads(POLYPGEN_LOCK.read_text(encoding="utf-8"))
    if pg.get("decision") != EXPECTED_POLYPGEN_DECISION:
        raise RuntimeError("PolypGen frozen state changed")


def self_test():
    assert TENT_STEPS == 2 and ACTION == "A2_TENT_2STEP"
    assert EXPECTED_CASES * EXPECTED_STATES == EXPECTED_MODEL_CASES
    print("ACTION_FREEZE_TEST_PASS")
    print("CARDINALITY_TEST_PASS")
    print("SELF_TEST_PASS")


def run(args):
    print("===== R11C NEOPOLYP TENT2 PREDICTION LOCK FIX1 =====")
    print("EXPERIMENT TYPE: ADAPTATION-INTENSITY GENERALIZATION")
    print("CROSS-TTA-ALGORITHM CLAIM: NO")
    print("ACTION:", ACTION)
    print("LR:", TENT_LR, "STEPS:", TENT_STEPS, "EPISODIC RESET: YES")
    print("NEOPOLYP GT PIXELS DECODED: NO")
    print("R05D3 A1 MASK VALUES READ: NO")
    print("SAFETY SCORES READ: NO")
    print("POLYPGEN DATA/PERFORMANCE READ: NO")
    print("HYPERPARAMETER SWEEP: NO\n")

    validate_lineage()
    r05d3 = import_module(R05D3_SCRIPT, "q1_r11c_r05d3")
    _, _, _, manifest_path, _ = r05d3.validate_upstream()
    targets, d2helper = r05d3.load_frozen_targets(manifest_path)
    panel, r03 = r05d3.build_frozen_panel(d2helper)
    if len(targets) != EXPECTED_CASES or len(panel) != EXPECTED_STATES:
        raise RuntimeError("Frozen target/panel cardinality changed")

    import torch
    if not torch.cuda.is_available() and not args.cpu:
        raise RuntimeError("CUDA unavailable; use --cpu only for debugging")
    device = torch.device("cpu" if args.cpu else "cuda")
    print("Device:", device)
    if device.type == "cuda": print("CUDA:", torch.cuda.get_device_name(device))

    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    build = Path(str(args.output_dir) + "__building")
    if build.exists() and not args.resume:
        raise FileExistsError(f"Partial output exists: {build}; use --resume")
    build.mkdir(parents=True, exist_ok=True)
    state_dir = build / "state_predictions"
    state_dir.mkdir(exist_ok=True)

    all_rows = []
    summaries = []
    seg_context = None

    for state in panel:
        print("\n" + "=" * 72)
        print("STATE:", state["model_state_id"])
        cached = load_cached(r05d3, state_dir, state) if args.resume else None
        if cached is not None:
            pred, idx, side, rows = cached
            print("[RESUME VERIFIED]", state["model_state_id"])
        else:
            frozen_source, source_ref = load_frozen_source(r05d3, state)
            fam = state["model_family"]
            if fam == "PraNet":
                a2, rows = run_pranet(r05d3, state, targets, device, frozen_source)
            elif fam == "DeepLabV3-R50":
                a2, rows = run_deeplab(r05d3, state, targets, device, frozen_source)
            elif fam == "SegFormer-B0":
                a2, rows, seg_context = run_segformer(r05d3, r03, state, targets, device, frozen_source, seg_context)
            else:
                raise RuntimeError(f"Unexpected family {fam}")
            pred, idx, side = save_state(r05d3, state_dir, state, a2, rows, source_ref)
            del a2, frozen_source

        changed = np.asarray([int(r["source_a2_changed_pixels"]) for r in rows])
        summaries.append({
            "model_state_id": state["model_state_id"],
            "model_family": state["model_family"],
            "training_seed": int(state["training_seed"]),
            "rows": len(rows),
            "source_parity_mismatches": 0,
            "changed_cases": int(np.count_nonzero(changed > 0)),
            "changed_pixels_mean": float(changed.mean()),
            "changed_pixels_median": float(np.median(changed)),
            "step1_entropy_mean": float(np.mean([float(r["step1_entropy_loss"]) for r in rows])),
            "step2_entropy_mean": float(np.mean([float(r["step2_entropy_loss"]) for r in rows])),
        })
        for r in rows:
            x = dict(r)
            x["state_prediction_npz"] = str(pred)
            x["state_index_csv"] = str(idx)
            x["state_lock_json"] = str(side)
            all_rows.append(x)

    if len(all_rows) != EXPECTED_MODEL_CASES:
        raise RuntimeError(f"Rows={len(all_rows)}, expected={EXPECTED_MODEL_CASES}")

    manifest = build / "R11C_model_case_a2_prediction_manifest.csv"
    summary = build / "R11C_state_prediction_change_summary.csv"
    boundary = build / "R11C_information_boundary_audit.json"
    global_fields = INDEX_FIELDS + ["state_prediction_npz", "state_index_csv", "state_lock_json"]
    r05d3.write_csv(manifest, all_rows, global_fields)
    r05d3.write_csv(summary, summaries, list(summaries[0].keys()))
    r05d3.write_json(boundary, {
        "cross_tta_algorithm_claim": False,
        "adaptation_intensity_generalization": True,
        "action": ACTION,
        "hyperparameter_sweep": False,
        "neopolyp_gt_pixels_decoded": False,
        "r05d3_source_masks_read_for_parity": True,
        "r05d3_a1_mask_values_read": False,
        "safety_scores_read": False,
        "polypgen_data_read": False,
        "source_parity_mismatches": 0,
    })

    lock = {
        "decision": DECISION,
        "script_version": VERSION,
        "build": BUILD,
        "action": ACTION,
        "optimizer": "Adam",
        "lr": TENT_LR,
        "weight_decay": TENT_WEIGHT_DECAY,
        "steps": TENT_STEPS,
        "episodic_reset": True,
        "target_cases": EXPECTED_CASES,
        "model_states": EXPECTED_STATES,
        "model_case_rows": EXPECTED_MODEL_CASES,
        "source_parity_mismatches": 0,
        "artifacts": {
            manifest.name: sha256_file(manifest),
            summary.name: sha256_file(summary),
            boundary.name: sha256_file(boundary),
        },
        "next_stage": "R11D_GT_UTILITY_REVEAL_AND_FROZEN_SAFETY_SCORE_TRANSFER_EVALUATION",
    }
    lock_path = build / "R11C_NEOPOLYP_TENT2_PREDICTION_LOCK.json"
    r05d3.write_json(lock_path, lock)
    lock_sha = sha256_file(lock_path)

    log = "\n".join([
        "===== R11C NEOPOLYP TENT2 PREDICTION LOCK FIX1 =====",
        f"action={ACTION}",
        "adaptation-intensity generalization=YES",
        "cross-TTA-algorithm generalization=NO",
        f"cases={EXPECTED_CASES}",
        f"states={EXPECTED_STATES}",
        f"model-case rows={EXPECTED_MODEL_CASES}",
        "SOURCE parity mismatches=0",
        "R05D3 A1 mask values read=NO",
        "NeoPolyp GT pixels decoded=NO",
        "safety scores read=NO",
        "PolypGen data/performance read=NO",
        "hyperparameter sweep=NO",
        f"Decision={DECISION}",
        "PASS",
    ]) + "\n"
    (build / "run_log.txt").write_text(log, encoding="utf-8")
    build.rename(args.output_dir)
    print("\n" + log)
    print("R11C LOCK:", args.output_dir / "R11C_NEOPOLYP_TENT2_PREDICTION_LOCK.json")
    print("R11C LOCK SHA256:", lock_sha)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    if args.self_test:
        self_test(); return 0
    try:
        run(args)
    except Exception:
        traceback.print_exc(); raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
