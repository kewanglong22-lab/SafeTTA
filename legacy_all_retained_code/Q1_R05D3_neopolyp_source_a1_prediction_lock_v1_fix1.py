#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Q1-R05D3 FIX1 — Freeze NeoPolyp SOURCE + A1_TENT_1STEP predictions.

Scientific boundary:
- all 1000 x 9 model-case pairs receive A1;
- D2 PAOT probability values are not parsed and are not used for gating;
- target GT paths remain provenance strings only and are never decoded;
- no target Dice/outcome is computed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import shutil
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

# Frozen DeepLab/R03 deterministic CUDA setting; must precede torch import.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
from PIL import Image
from tqdm import tqdm


VERSION = "2026-08-19-Q1-R05D3-v1-fix1"
BUILD = "Q1_R05D3_NEOPOLYP_SOURCE_A1_PREDICTION_LOCK_FIX1"

ROOT = Path(r"F:\MEDSEG_SAFETTA")

PROTOCOL = (
    ROOT / "docs"
    / "Q1_R05D3_neopolyp_source_a1_prediction_lock_preregistered_protocol_v1_fix1.md"
)
EXPECTED_PROTOCOL_SHA256 = "e980b7491e50206b48ee98069429036391bc8f35dc29c886cb193d411925e03e"

R05D1_DIR = (
    ROOT / "outputs"
    / "Q1_R05D1_neopolyp_acquisition_pairing_duplicate_qc_v1_fix2"
)
R05D1_LOCK = R05D1_DIR / "Q1_R05D1_NEOPOLYP_TARGET_QC_LOCK.json"
EXPECTED_R05D1_LOCK_SHA256 = (
    "bfe78a4d2fe42049f1658ead22b3e74deeac5afb036426d73a9cff7506b6cf41"
)
EXPECTED_R05D1_DECISION = "NEOPOLYP_TARGET_QC_READY"

R05D2_DIR = (
    ROOT / "outputs"
    / "Q1_R05D2_neopolyp_prospective_paot_probability_lock_v1_fix1"
)
R05D2_LOCK = R05D2_DIR / "Q1_R05D2_NEOPOLYP_PAOT_PROBABILITY_LOCK.json"
EXPECTED_R05D2_LOCK_SHA256 = (
    "35705ef25915e173f5ba624fcab524cc74225047c9b340572ffe64566631d7ab"
)
EXPECTED_R05D2_DECISION = "NEOPOLYP_PROSPECTIVE_PAOT_PROBABILITIES_LOCKED"
EXPECTED_D2_TABLE_SHA256 = (
    "7e9aa5d46f9024ab163933fe4afcf580d60b4ee9120ee34b97a0ea63b9a1bc4d"
)

R05D2_SCRIPT = (
    ROOT / "code"
    / "Q1_R05D2_neopolyp_prospective_paot_probability_lock_v1_fix1.py"
)
EXPECTED_R05D2_SCRIPT_SHA256 = (
    "663e3661e0903712065776cb6a2c0af7f011e731a3ffaf38ade0dcfad39eb077"
)

R03_DIR = (
    ROOT / "outputs"
    / "Q1_R03_nine_state_heterogeneous_source_utility_asset_v1_fix2"
)
R03_LOCK = R03_DIR / "Q1_R03_NINE_STATE_SOURCE_UTILITY_LOCK.json"
EXPECTED_R03_LOCK_SHA256 = (
    "2c773f5eb70bedb50e2c231ef2512c06dff5ecca425eb43b2719637c1bc06dad"
)
EXPECTED_R03_DECISION = "NINE_STATE_SOURCE_UTILITY_ASSET_READY"

R03_SCRIPT = (
    ROOT / "code"
    / "Q1_R03_nine_state_heterogeneous_source_utility_asset_v1_fix2.py"
)
EXPECTED_R03_SCRIPT_SHA256 = (
    "d737272b756a4d2051f9c74051d7085640eaec9cd8be6dbd951c325af2a17c31"
)

PRANET_HELPER = ROOT / "code" / "S03_B_build_source_side_safettta_gate_v1.py"
EXPECTED_PRANET_HELPER_SHA256 = (
    "07ac65247216dea376e8996cbfbc98c54c62157ecdd1e9b7d9b539b5a74a2b5c"
)

DEEPLAB_HELPER = (
    ROOT / "code" / "S05_C_build_deeplab_source_side_safettta_gate_v1.py"
)
EXPECTED_DEEPLAB_HELPER_SHA256 = (
    "ab1d98f9847b2e338f07ec26ff426f9cd428e68da4813b90e2227782319aa424"
)

OUTPUT_DIR = (
    ROOT / "outputs"
    / "Q1_R05D3_neopolyp_source_a1_prediction_lock_v1_fix1"
)

EXPECTED_CASES = 1000
EXPECTED_STATES = 9
EXPECTED_MODEL_CASES = 9000

MASK_H = 352
MASK_W = 352
MASK_PIXELS = MASK_H * MASK_W
PACKED_BYTES = (MASK_PIXELS + 7) // 8
BITORDER = "little"

TENT_LR = 1e-3
TENT_WEIGHT_DECAY = 0.0
TENT_STEPS = 1

FAMILIES = ("PraNet", "DeepLabV3-R50", "SegFormer-B0")
DECISION_READY = "NEOPOLYP_SOURCE_A1_PREDICTIONS_LOCKED"

GT_RULE = {
    "dataset": "BKAI-IGH NeoPolyp-Small",
    "evaluation_task": "binary_polyp_segmentation",
    "original_gt_classes": {
        "0": {"name": "background", "rgb": [0, 0, 0]},
        "1": {"name": "non_neoplastic_polyp", "rgb": [0, 255, 0]},
        "2": {"name": "neoplastic_polyp", "rgb": [255, 0, 0]},
    },
    "future_decode_rule": "nearest_RGB_prototype_euclidean",
    "binary_mapping": {
        "background": 0,
        "non_neoplastic_polyp": 1,
        "neoplastic_polyp": 1,
    },
    "future_gt_resize": {
        "height": MASK_H,
        "width": MASK_W,
        "interpolation": "nearest",
    },
    "source_prediction_threshold": "binary_logit>=0",
    "action_prediction_threshold": "binary_logit>=0",
    "dice_empty_empty": 1.0,
    "harm_threshold_delta_dice": -0.02,
    "benefit_threshold_delta_dice": 0.02,
}

STATE_INDEX_FIELDS = [
    "row_index",
    "sample_id",
    "image_path",
    "image_raw_sha256",
    "model_family",
    "model_state_id",
    "training_seed",
    "checkpoint_sha256",
    "source_foreground_pixels",
    "a1_foreground_pixels",
    "changed_pixels",
]

GLOBAL_FIELDS = [
    "sample_id",
    "image_path",
    "image_raw_sha256",
    "model_family",
    "model_state_id",
    "training_seed",
    "checkpoint_sha256",
    "state_prediction_npz",
    "state_index_csv",
    "state_row_index",
    "source_foreground_pixels",
    "a1_foreground_pixels",
    "changed_pixels",
]


# ---------------------------------------------------------------------
# Generic I/O
# ---------------------------------------------------------------------

def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def validate_sha(path: Path, expected: str, label: str) -> str:
    if not path.exists():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual.lower() != str(expected).lower():
        raise RuntimeError(
            f"{label} SHA mismatch: expected={expected} actual={actual}"
        )
    return actual


def read_csv(path: Path):
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return list(reader), reader.fieldnames or []


def write_csv(path: Path, rows, fields: Sequence[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(fields))
        w.writeheader()
        w.writerows(rows)


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def artifact_path_from_lock(base_dir: Path, lock: dict, key: str) -> Path:
    meta = lock.get("artifacts", {}).get(key)
    if not isinstance(meta, dict):
        raise RuntimeError(f"Lock missing artifact metadata: {key}")
    rel = meta.get("relative_path") or meta.get("filename")
    if not rel:
        raise RuntimeError(f"Locked artifact path missing: {key}")
    path = base_dir / str(rel)
    validate_sha(path, str(meta.get("sha256", "")), f"locked artifact {key}")
    return path


def set_runtime_seed(seed: int):
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
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


# ---------------------------------------------------------------------
# Upstream validation
# ---------------------------------------------------------------------

def validate_upstream():
    validate_sha(PROTOCOL, EXPECTED_PROTOCOL_SHA256, "D3 protocol")
    validate_sha(R05D1_LOCK, EXPECTED_R05D1_LOCK_SHA256, "D1 lock")
    validate_sha(R05D2_LOCK, EXPECTED_R05D2_LOCK_SHA256, "D2 lock")
    validate_sha(R05D2_SCRIPT, EXPECTED_R05D2_SCRIPT_SHA256, "D2 script")
    validate_sha(R03_LOCK, EXPECTED_R03_LOCK_SHA256, "R03 lock")
    validate_sha(R03_SCRIPT, EXPECTED_R03_SCRIPT_SHA256, "R03 script")
    validate_sha(PRANET_HELPER, EXPECTED_PRANET_HELPER_SHA256, "PraNet helper")
    validate_sha(DEEPLAB_HELPER, EXPECTED_DEEPLAB_HELPER_SHA256, "DeepLab helper")

    d1 = json.loads(R05D1_LOCK.read_text(encoding="utf-8"))
    d2 = json.loads(R05D2_LOCK.read_text(encoding="utf-8"))
    r03_lock = json.loads(R03_LOCK.read_text(encoding="utf-8"))

    if d1.get("decision") != EXPECTED_R05D1_DECISION:
        raise RuntimeError(f"Unexpected D1 decision: {d1.get('decision')}")
    if int(d1.get("eligible_confirmatory_cases", -1)) != EXPECTED_CASES:
        raise RuntimeError("D1 eligible target count changed.")
    if int(d1.get("exact_overlap_exclusions", -1)) != 0:
        raise RuntimeError("D1 overlap exclusion count changed.")
    if bool(d1.get("target_gt_pixels_decoded", True)):
        raise RuntimeError("D1 reports target GT decoding.")

    if d2.get("decision") != EXPECTED_R05D2_DECISION:
        raise RuntimeError(f"Unexpected D2 decision: {d2.get('decision')}")
    if int(d2.get("target_cases", -1)) != EXPECTED_CASES:
        raise RuntimeError("D2 target count changed.")
    if int(d2.get("model_states", -1)) != EXPECTED_STATES:
        raise RuntimeError("D2 model-state count changed.")
    if int(d2.get("model_case_rows", -1)) != EXPECTED_MODEL_CASES:
        raise RuntimeError("D2 model-case row count changed.")
    if bool(d2.get("target_gt_pixels_decoded", True)):
        raise RuntimeError("D2 reports target GT decoding.")
    if bool(d2.get("target_outcomes_revealed", True)):
        raise RuntimeError("D2 reports target outcome reveal.")
    if bool(d2.get("predictor_fit_or_modified", True)):
        raise RuntimeError("D2 reports predictor modification.")

    # IMPORTANT: only hash the frozen D2 table; never parse probability rows.
    d2_table = artifact_path_from_lock(
        R05D2_DIR,
        d2,
        "target_paot_probabilities",
    )
    validate_sha(d2_table, EXPECTED_D2_TABLE_SHA256, "D2 PAOT table")

    d1_manifest = artifact_path_from_lock(
        R05D1_DIR,
        d1,
        "frozen_confirmatory_manifest",
    )

    if r03_lock.get("decision") != EXPECTED_R03_DECISION:
        raise RuntimeError(f"Unexpected R03 decision: {r03_lock.get('decision')}")

    return d1, d2, r03_lock, d1_manifest, d2_table


def load_frozen_targets(manifest_path: Path):
    # Reuse the already locked D2 target-manifest loader, including raw image SHA checks.
    d2helper = import_module(R05D2_SCRIPT, "q1_r05d3_d2_helper")
    targets = d2helper.load_target_manifest(
        manifest_path,
        verify_image_hashes=True,
    )
    if len(targets) != EXPECTED_CASES:
        raise RuntimeError(f"Target rows={len(targets)} expected={EXPECTED_CASES}")
    return targets, d2helper


def build_frozen_panel(d2helper):
    r03 = import_module(R03_SCRIPT, "q1_r05d3_r03_helper")
    panel = d2helper.build_panel(r03)
    if len(panel) != EXPECTED_STATES:
        raise RuntimeError("Frozen panel state count changed.")
    if {r["model_family"] for r in panel} != set(FAMILIES):
        raise RuntimeError("Frozen panel family set changed.")
    return panel, r03


# ---------------------------------------------------------------------
# Packed mask format
# ---------------------------------------------------------------------

def logit_to_mask(z) -> np.ndarray:
    arr = np.asarray(z)
    if arr.shape != (MASK_H, MASK_W):
        raise RuntimeError(
            f"Unexpected binary logit shape={arr.shape}; expected={(MASK_H, MASK_W)}"
        )
    if not np.isfinite(arr).all():
        raise RuntimeError("Non-finite binary logit.")
    return (arr >= 0.0).astype(np.uint8)


def pack_mask(mask: np.ndarray) -> np.ndarray:
    m = np.asarray(mask, dtype=np.uint8)
    if m.shape != (MASK_H, MASK_W):
        raise RuntimeError("Mask shape mismatch before pack.")
    if not np.isin(m, (0, 1)).all():
        raise RuntimeError("Mask is not binary.")
    packed = np.packbits(m.reshape(-1), bitorder=BITORDER)
    if packed.shape != (PACKED_BYTES,):
        raise RuntimeError("Packed mask byte count mismatch.")
    return packed


def unpack_mask(packed: np.ndarray) -> np.ndarray:
    p = np.asarray(packed, dtype=np.uint8)
    if p.shape != (PACKED_BYTES,):
        raise RuntimeError("Packed mask shape mismatch.")
    bits = np.unpackbits(p, count=MASK_PIXELS, bitorder=BITORDER)
    return bits.reshape(MASK_H, MASK_W).astype(np.uint8, copy=False)


def validate_prediction_arrays(source_pack, a1_pack):
    source_pack = np.asarray(source_pack, dtype=np.uint8)
    a1_pack = np.asarray(a1_pack, dtype=np.uint8)

    expected = (EXPECTED_CASES, PACKED_BYTES)
    if source_pack.shape != expected:
        raise RuntimeError(
            f"SOURCE packed shape={source_pack.shape}; expected={expected}"
        )
    if a1_pack.shape != expected:
        raise RuntimeError(
            f"A1 packed shape={a1_pack.shape}; expected={expected}"
        )

    for idx in (0, EXPECTED_CASES // 2, EXPECTED_CASES - 1):
        sm = unpack_mask(source_pack[idx])
        am = unpack_mask(a1_pack[idx])
        if sm.shape != (MASK_H, MASK_W) or am.shape != (MASK_H, MASK_W):
            raise RuntimeError("Packed mask round-trip shape failure.")
        if not np.isin(sm, (0, 1)).all() or not np.isin(am, (0, 1)).all():
            raise RuntimeError("Packed mask round-trip binary failure.")


# ---------------------------------------------------------------------
# Frozen architecture/action execution
# ---------------------------------------------------------------------

def run_pranet_state(state, targets, device):
    helper = import_module(
        PRANET_HELPER,
        f"q1_r05d3_pranet_{state['training_seed']}",
    )
    training = helper.import_training_helper()

    seed = int(state["training_seed"])
    set_runtime_seed(seed)

    model = helper.load_model(training, seed, device)
    params = helper.configure_tent(model)
    source_values = helper.snapshot_params(params)

    source_pack = np.empty((EXPECTED_CASES, PACKED_BYTES), dtype=np.uint8)
    a1_pack = np.empty_like(source_pack)
    index_rows = []

    pbar = tqdm(
        enumerate(targets),
        total=EXPECTED_CASES,
        desc=f"D3 PraNet seed {seed}",
        unit="img",
        dynamic_ncols=True,
    )

    for i, target in pbar:
        with Image.open(target["image_path"]) as im:
            native = im.convert("RGB")

        x = helper.image_to_model_tensor(training, native).to(
            device,
            non_blocking=True,
        )

        # Exact S03-B SOURCE + episodic A1.
        z_source, z_a1 = helper.source_and_tent_logits(
            model,
            params,
            source_values,
            x,
        )

        sm = logit_to_mask(z_source)
        am = logit_to_mask(z_a1)

        source_pack[i] = pack_mask(sm)
        a1_pack[i] = pack_mask(am)

        changed = int(np.count_nonzero(sm != am))
        index_rows.append({
            "row_index": i,
            "sample_id": target["sample_id"],
            "image_path": target["image_path"],
            "image_raw_sha256": target["image_raw_sha256"],
            "model_family": state["model_family"],
            "model_state_id": state["model_state_id"],
            "training_seed": seed,
            "checkpoint_sha256": state["checkpoint_sha256"],
            "source_foreground_pixels": int(sm.sum()),
            "a1_foreground_pixels": int(am.sum()),
            "changed_pixels": changed,
        })
        pbar.set_postfix(chg=changed)

        del x, z_source, z_a1, sm, am

    helper.restore_params(params, source_values)
    del model, params, source_values

    if device.type == "cuda":
        __import__("torch").cuda.empty_cache()

    return source_pack, a1_pack, index_rows


def run_deeplab_state(state, targets, device):
    helper = import_module(
        DEEPLAB_HELPER,
        f"q1_r05d3_deeplab_{state['training_seed']}",
    )
    training = helper.import_training_helper()

    # Exact deterministic convention inherited from R03/S05-C.
    helper.seed_everything(20260817)

    seed = int(state["training_seed"])
    model = helper.load_model(training, seed, device)

    (
        params,
        trainable_bn_names,
        unsafe_bn_names,
        unsafe_bn_modules,
        dropout_names,
        dropout_modules,
    ) = helper.configure_singleton_safe_tent(model)

    source_values = helper.snapshot_params(params)

    source_pack = np.empty((EXPECTED_CASES, PACKED_BYTES), dtype=np.uint8)
    a1_pack = np.empty_like(source_pack)
    index_rows = []

    pbar = tqdm(
        enumerate(targets),
        total=EXPECTED_CASES,
        desc=f"D3 DeepLab seed {seed}",
        unit="img",
        dynamic_ncols=True,
    )

    for i, target in pbar:
        with Image.open(target["image_path"]) as im:
            native = im.convert("RGB")

        x = helper.image_to_model_tensor(training, native).to(
            device,
            non_blocking=True,
        )

        z_source, z_a1, adapt_loss = helper.source_and_tent_logits(
            helper=training,
            model=model,
            params=params,
            source_values=source_values,
            unsafe_bn_modules=unsafe_bn_modules,
            dropout_modules=dropout_modules,
            x=x,
        )

        if not math.isfinite(float(adapt_loss)):
            raise RuntimeError("Non-finite DeepLab A1 entropy loss.")

        sm = logit_to_mask(z_source)
        am = logit_to_mask(z_a1)

        source_pack[i] = pack_mask(sm)
        a1_pack[i] = pack_mask(am)

        changed = int(np.count_nonzero(sm != am))
        index_rows.append({
            "row_index": i,
            "sample_id": target["sample_id"],
            "image_path": target["image_path"],
            "image_raw_sha256": target["image_raw_sha256"],
            "model_family": state["model_family"],
            "model_state_id": state["model_state_id"],
            "training_seed": seed,
            "checkpoint_sha256": state["checkpoint_sha256"],
            "source_foreground_pixels": int(sm.sum()),
            "a1_foreground_pixels": int(am.sum()),
            "changed_pixels": changed,
        })
        pbar.set_postfix(chg=changed)

        del x, z_source, z_a1, adapt_loss, sm, am

    helper.restore_params(params, source_values)

    del (
        model,
        params,
        source_values,
        trainable_bn_names,
        unsafe_bn_names,
        unsafe_bn_modules,
        dropout_names,
        dropout_modules,
    )

    if device.type == "cuda":
        __import__("torch").cuda.empty_cache()

    return source_pack, a1_pack, index_rows


def run_segformer_state(state, targets, device, r03, seg_context=None):
    if seg_context is None:
        seg_context = r03.build_segformer_context(device)

    (
        torch,
        nn,
        F,
        SegformerForSemanticSegmentation,
        config,
        mean,
        std,
    ) = seg_context

    seed = int(state["training_seed"])
    set_runtime_seed(seed)

    model = r03.load_segformer_state(
        seed,
        device,
        SegformerForSemanticSegmentation,
        config,
        torch,
    )

    (
        params,
        param_names,
        bn_modules,
        bn_original_track,
        dropout_modules,
        dropout_names,
    ) = r03.configure_segformer_tent(model, nn)

    source_values = r03.snapshot_params(params)

    source_pack = np.empty((EXPECTED_CASES, PACKED_BYTES), dtype=np.uint8)
    a1_pack = np.empty_like(source_pack)
    index_rows = []

    pbar = tqdm(
        enumerate(targets),
        total=EXPECTED_CASES,
        desc=f"D3 SegFormer seed {seed}",
        unit="img",
        dynamic_ncols=True,
    )

    for i, target in pbar:
        with Image.open(target["image_path"]) as im:
            native = im.convert("RGB")

        x = r03.segformer_tensor(native, mean, std, torch).to(
            device,
            non_blocking=True,
        )

        # Frozen SOURCE.
        r03.restore_params(params, source_values, torch)
        r03.segformer_source_mode(
            model,
            bn_modules,
            bn_original_track,
            dropout_modules,
        )
        with torch.no_grad():
            _, z_source_t = r03.segformer_logits_and_z(model, x, F)

        # Frozen one-step episodic A1.
        r03.restore_params(params, source_values, torch)
        r03.segformer_tent_mode(
            model,
            bn_modules,
            dropout_modules,
        )

        optimizer = torch.optim.Adam(
            params,
            lr=TENT_LR,
            weight_decay=TENT_WEIGHT_DECAY,
        )
        optimizer.zero_grad(set_to_none=True)

        logits_pre, _ = r03.segformer_logits_and_z(model, x, F)
        loss = r03.categorical_entropy(logits_pre, torch)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite SegFormer A1 entropy loss.")

        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            _, z_a1_t = r03.segformer_logits_and_z(model, x, F)

        z_source = z_source_t[0].detach().float().cpu().numpy()
        z_a1 = z_a1_t[0].detach().float().cpu().numpy()

        sm = logit_to_mask(z_source)
        am = logit_to_mask(z_a1)

        source_pack[i] = pack_mask(sm)
        a1_pack[i] = pack_mask(am)

        changed = int(np.count_nonzero(sm != am))
        index_rows.append({
            "row_index": i,
            "sample_id": target["sample_id"],
            "image_path": target["image_path"],
            "image_raw_sha256": target["image_raw_sha256"],
            "model_family": state["model_family"],
            "model_state_id": state["model_state_id"],
            "training_seed": seed,
            "checkpoint_sha256": state["checkpoint_sha256"],
            "source_foreground_pixels": int(sm.sum()),
            "a1_foreground_pixels": int(am.sum()),
            "changed_pixels": changed,
        })
        pbar.set_postfix(chg=changed)

        del (
            optimizer,
            logits_pre,
            loss,
            z_source_t,
            z_a1_t,
            z_source,
            z_a1,
            sm,
            am,
            x,
        )

    r03.restore_params(params, source_values, torch)

    del (
        model,
        params,
        source_values,
        param_names,
        bn_modules,
        bn_original_track,
        dropout_modules,
        dropout_names,
    )

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return source_pack, a1_pack, index_rows, seg_context


# ---------------------------------------------------------------------
# State lock / resume
# ---------------------------------------------------------------------

def state_token(state):
    return (
        state["model_family"]
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
        + f"_seed{state['training_seed']}"
    )


def state_paths(state_dir: Path, state):
    token = state_token(state)
    return (
        state_dir / f"{token}_predictions.npz",
        state_dir / f"{token}_index.csv",
        state_dir / f"{token}_lock.json",
    )


def validate_index_rows(rows, state, targets):
    if len(rows) != EXPECTED_CASES:
        raise RuntimeError(
            f"State index rows={len(rows)} expected={EXPECTED_CASES}"
        )

    expected_ids = [r["sample_id"] for r in targets]
    actual_ids = [r["sample_id"] for r in rows]
    if actual_ids != expected_ids:
        raise RuntimeError("State index target ordering changed.")

    for i, row in enumerate(rows):
        if int(row["row_index"]) != i:
            raise RuntimeError("State row index changed.")
        if row["model_family"] != state["model_family"]:
            raise RuntimeError("State family mismatch.")
        if row["model_state_id"] != state["model_state_id"]:
            raise RuntimeError("State ID mismatch.")
        if int(row["training_seed"]) != int(state["training_seed"]):
            raise RuntimeError("State seed mismatch.")
        if row["checkpoint_sha256"] != state["checkpoint_sha256"]:
            raise RuntimeError("State checkpoint SHA mismatch.")

        for field in (
            "source_foreground_pixels",
            "a1_foreground_pixels",
            "changed_pixels",
        ):
            value = int(row[field])
            if value < 0 or value > MASK_PIXELS:
                raise RuntimeError(f"Invalid {field}={value}")


def target_order_sha(targets):
    payload = "\n".join(r["sample_id"] for r in targets).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def save_state(
    state_dir,
    state,
    targets,
    source_pack,
    a1_pack,
    index_rows,
    target_manifest_sha,
    target_id_order_sha,
):
    pred_path, index_path, lock_path = state_paths(state_dir, state)

    if pred_path.exists() or index_path.exists() or lock_path.exists():
        raise FileExistsError(
            f"State cache already exists: {state['model_state_id']}"
        )

    validate_prediction_arrays(source_pack, a1_pack)
    validate_index_rows(index_rows, state, targets)

    np.savez_compressed(
        pred_path,
        source_masks_packed=np.asarray(source_pack, dtype=np.uint8),
        a1_masks_packed=np.asarray(a1_pack, dtype=np.uint8),
    )

    write_csv(index_path, index_rows, STATE_INDEX_FIELDS)

    pred_sha = sha256_file(pred_path)
    index_sha = sha256_file(index_path)

    sidecar = {
        "script_version": VERSION,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "r05d1_lock_sha256": EXPECTED_R05D1_LOCK_SHA256,
        "r05d2_lock_sha256": EXPECTED_R05D2_LOCK_SHA256,
        "r03_lock_sha256": EXPECTED_R03_LOCK_SHA256,
        "target_manifest_sha256": target_manifest_sha,
        "target_id_order_sha256": target_id_order_sha,
        "model_family": state["model_family"],
        "model_state_id": state["model_state_id"],
        "training_seed": state["training_seed"],
        "checkpoint_sha256": state["checkpoint_sha256"],
        "target_cases": EXPECTED_CASES,
        "source_masks": EXPECTED_CASES,
        "a1_masks": EXPECTED_CASES,
        "mask_height": MASK_H,
        "mask_width": MASK_W,
        "mask_pixels": MASK_PIXELS,
        "packed_bytes_per_mask": PACKED_BYTES,
        "bitorder": BITORDER,
        "action": "A1_TENT_1STEP",
        "optimizer": "Adam",
        "lr": TENT_LR,
        "weight_decay": TENT_WEIGHT_DECAY,
        "steps": TENT_STEPS,
        "episodic_reset": True,
        "paot_probability_values_read": False,
        "paot_gating_used": False,
        "target_gt_pixels_decoded": False,
        "prediction_npz_sha256": pred_sha,
        "state_index_csv_sha256": index_sha,
    }
    write_json(lock_path, sidecar)

    return pred_path, index_path, lock_path


def load_state(
    state_dir,
    state,
    targets,
    target_manifest_sha,
    target_id_order_sha,
    allow_orphan_cleanup,
):
    pred_path, index_path, lock_path = state_paths(state_dir, state)
    exists = [pred_path.exists(), index_path.exists(), lock_path.exists()]

    if any(exists) and not all(exists):
        if allow_orphan_cleanup:
            for p in (pred_path, index_path, lock_path):
                if p.exists():
                    p.unlink()
            return None
        raise RuntimeError(
            f"Orphan state cache for {state['model_state_id']}; use --resume."
        )

    if not all(exists):
        return None

    sidecar = json.loads(lock_path.read_text(encoding="utf-8"))

    expected_exact = {
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "r05d1_lock_sha256": EXPECTED_R05D1_LOCK_SHA256,
        "r05d2_lock_sha256": EXPECTED_R05D2_LOCK_SHA256,
        "r03_lock_sha256": EXPECTED_R03_LOCK_SHA256,
        "target_manifest_sha256": target_manifest_sha,
        "target_id_order_sha256": target_id_order_sha,
        "model_family": state["model_family"],
        "model_state_id": state["model_state_id"],
        "checkpoint_sha256": state["checkpoint_sha256"],
        "bitorder": BITORDER,
        "action": "A1_TENT_1STEP",
        "optimizer": "Adam",
    }
    for key, expected in expected_exact.items():
        if sidecar.get(key) != expected:
            raise RuntimeError(
                f"State sidecar mismatch {key}: {state['model_state_id']}"
            )

    if int(sidecar.get("training_seed", -1)) != int(state["training_seed"]):
        raise RuntimeError("State sidecar seed mismatch.")
    if int(sidecar.get("target_cases", -1)) != EXPECTED_CASES:
        raise RuntimeError("State sidecar target count mismatch.")
    if int(sidecar.get("source_masks", -1)) != EXPECTED_CASES:
        raise RuntimeError("State sidecar source-mask count mismatch.")
    if int(sidecar.get("a1_masks", -1)) != EXPECTED_CASES:
        raise RuntimeError("State sidecar A1-mask count mismatch.")
    if int(sidecar.get("mask_height", -1)) != MASK_H:
        raise RuntimeError("State sidecar mask-height mismatch.")
    if int(sidecar.get("mask_width", -1)) != MASK_W:
        raise RuntimeError("State sidecar mask-width mismatch.")
    if int(sidecar.get("packed_bytes_per_mask", -1)) != PACKED_BYTES:
        raise RuntimeError("State sidecar packed-byte count mismatch.")
    if bool(sidecar.get("paot_probability_values_read", True)):
        raise RuntimeError("State sidecar reports PAOT probability read.")
    if bool(sidecar.get("paot_gating_used", True)):
        raise RuntimeError("State sidecar reports PAOT gating.")
    if bool(sidecar.get("target_gt_pixels_decoded", True)):
        raise RuntimeError("State sidecar reports target GT decoding.")

    validate_sha(
        pred_path,
        sidecar["prediction_npz_sha256"],
        "state prediction NPZ",
    )
    validate_sha(
        index_path,
        sidecar["state_index_csv_sha256"],
        "state index CSV",
    )

    rows, fields = read_csv(index_path)
    if fields != STATE_INDEX_FIELDS:
        raise RuntimeError("State index column order changed.")
    validate_index_rows(rows, state, targets)

    with np.load(pred_path, allow_pickle=False) as z:
        if set(z.files) != {"source_masks_packed", "a1_masks_packed"}:
            raise RuntimeError("Unexpected arrays in state prediction NPZ.")
        validate_prediction_arrays(
            z["source_masks_packed"],
            z["a1_masks_packed"],
        )

    return rows


def global_rows_from_state(index_rows, pred_path, index_path):
    rows = []
    for row in index_rows:
        rows.append({
            "sample_id": row["sample_id"],
            "image_path": row["image_path"],
            "image_raw_sha256": row["image_raw_sha256"],
            "model_family": row["model_family"],
            "model_state_id": row["model_state_id"],
            "training_seed": row["training_seed"],
            "checkpoint_sha256": row["checkpoint_sha256"],
            "state_prediction_npz": str(pred_path),
            "state_index_csv": str(index_path),
            "state_row_index": row["row_index"],
            "source_foreground_pixels": row["source_foreground_pixels"],
            "a1_foreground_pixels": row["a1_foreground_pixels"],
            "changed_pixels": row["changed_pixels"],
        })
    return rows


# ---------------------------------------------------------------------
# Preflight: dummy-only A1 smoke tests; no target image pixels
# ---------------------------------------------------------------------

def preflight():
    _, _, _, manifest_path, _ = validate_upstream()
    targets, d2helper = load_frozen_targets(manifest_path)
    panel, r03 = build_frozen_panel(d2helper)

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for Q1-R05D3.")
    device = torch.device("cuda")

    dummy_rgb = Image.new("RGB", (352, 352), color=(127, 127, 127))

    # PraNet dummy SOURCE+A1.
    ps = next(s for s in panel if s["model_family"] == "PraNet")
    ph = import_module(PRANET_HELPER, "q1_r05d3_preflight_pranet")
    pt = ph.import_training_helper()
    pm = ph.load_model(pt, int(ps["training_seed"]), device)
    pp = ph.configure_tent(pm)
    pv = ph.snapshot_params(pp)
    px = ph.image_to_model_tensor(pt, dummy_rgb).to(device)
    psrc, pa1 = ph.source_and_tent_logits(pm, pp, pv, px)
    if psrc.shape != (MASK_H, MASK_W) or pa1.shape != (MASK_H, MASK_W):
        raise RuntimeError("PraNet dummy SOURCE/A1 shape failed.")
    del pm, pp, pv, px, psrc, pa1

    # DeepLab dummy SOURCE+A1.
    ds = next(s for s in panel if s["model_family"] == "DeepLabV3-R50")
    dh = import_module(DEEPLAB_HELPER, "q1_r05d3_preflight_deeplab")
    dt = dh.import_training_helper()
    dh.seed_everything(20260817)
    dm = dh.load_model(dt, int(ds["training_seed"]), device)
    (
        dp, _, _, unsafe, _, dropouts
    ) = dh.configure_singleton_safe_tent(dm)
    dv = dh.snapshot_params(dp)
    dx = dh.image_to_model_tensor(dt, dummy_rgb).to(device)
    dsrc, da1, dloss = dh.source_and_tent_logits(
        helper=dt,
        model=dm,
        params=dp,
        source_values=dv,
        unsafe_bn_modules=unsafe,
        dropout_modules=dropouts,
        x=dx,
    )
    if dsrc.shape != (MASK_H, MASK_W) or da1.shape != (MASK_H, MASK_W):
        raise RuntimeError("DeepLab dummy SOURCE/A1 shape failed.")
    if not math.isfinite(float(dloss)):
        raise RuntimeError("DeepLab dummy entropy loss is non-finite.")
    del dm, dp, dv, unsafe, dropouts, dx, dsrc, da1, dloss

    # SegFormer dummy SOURCE+A1.
    segctx = r03.build_segformer_context(device)
    (
        torch2,
        nn,
        F,
        SegformerForSemanticSegmentation,
        config,
        mean,
        std,
    ) = segctx

    ss = next(s for s in panel if s["model_family"] == "SegFormer-B0")
    sm = r03.load_segformer_state(
        int(ss["training_seed"]),
        device,
        SegformerForSemanticSegmentation,
        config,
        torch2,
    )
    (
        sp,
        _,
        sbn,
        sbn_orig,
        sdrop,
        _,
    ) = r03.configure_segformer_tent(sm, nn)
    sv = r03.snapshot_params(sp)
    sx = r03.segformer_tensor(dummy_rgb, mean, std, torch2).to(device)

    r03.restore_params(sp, sv, torch2)
    r03.segformer_source_mode(sm, sbn, sbn_orig, sdrop)
    with torch2.no_grad():
        _, szsrc = r03.segformer_logits_and_z(sm, sx, F)

    r03.restore_params(sp, sv, torch2)
    r03.segformer_tent_mode(sm, sbn, sdrop)

    opt = torch2.optim.Adam(
        sp,
        lr=TENT_LR,
        weight_decay=TENT_WEIGHT_DECAY,
    )
    opt.zero_grad(set_to_none=True)
    slogits, _ = r03.segformer_logits_and_z(sm, sx, F)
    sloss = r03.categorical_entropy(slogits, torch2)
    sloss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)

    with torch2.no_grad():
        _, sza1 = r03.segformer_logits_and_z(sm, sx, F)

    if tuple(szsrc[0].shape) != (MASK_H, MASK_W):
        raise RuntimeError("SegFormer dummy SOURCE shape failed.")
    if tuple(sza1[0].shape) != (MASK_H, MASK_W):
        raise RuntimeError("SegFormer dummy A1 shape failed.")

    del (
        sm,
        sp,
        sbn,
        sbn_orig,
        sdrop,
        sv,
        sx,
        opt,
        slogits,
        sloss,
        szsrc,
        sza1,
    )
    torch.cuda.empty_cache()

    print("===== Q1-R05D3 PREFLIGHT =====")
    print(f"protocol_sha256={EXPECTED_PROTOCOL_SHA256}")
    print(f"r05d1_lock_sha256={EXPECTED_R05D1_LOCK_SHA256}")
    print(f"r05d2_lock_sha256={EXPECTED_R05D2_LOCK_SHA256}")
    print(f"d2_probability_table_sha256={EXPECTED_D2_TABLE_SHA256}")
    print(f"target_cases={len(targets)}")
    print(f"model_states={len(panel)}")
    print(f"expected_model_cases={EXPECTED_MODEL_CASES}")
    print("action=A1_TENT_1STEP")
    print("all_model_cases_receive_A1=YES")
    print("D2_probability_values_read=NO")
    print("PAOT_gating=NO")
    print("target_GT_pixels_decoded=NO")
    print("dummy_PraNet_SOURCE_A1=PASS")
    print("dummy_DeepLab_SOURCE_A1=PASS")
    print("dummy_SegFormer_SOURCE_A1=PASS")
    print("PREFLIGHT_PASS")


# ---------------------------------------------------------------------
# Formal run
# ---------------------------------------------------------------------

def run(args):
    d1, d2, r03_lock, manifest_path, _ = validate_upstream()
    targets, d2helper = load_frozen_targets(manifest_path)
    panel, r03 = build_frozen_panel(d2helper)

    if args.output_dir.exists():
        raise FileExistsError(
            f"Final Q1-R05D3 output already exists: {args.output_dir}"
        )

    build_dir = Path(str(args.output_dir) + "__building")
    if build_dir.exists() and not args.resume:
        raise FileExistsError(
            f"Partial Q1-R05D3 output exists: {build_dir}. "
            "Use --resume only after a technical interruption."
        )
    build_dir.mkdir(parents=True, exist_ok=True)

    state_dir = build_dir / "state_predictions"
    state_dir.mkdir(parents=True, exist_ok=True)

    protocol_copy = build_dir / "preregistered_protocol_copy.md"
    if not protocol_copy.exists():
        shutil.copy2(PROTOCOL, protocol_copy)
    validate_sha(protocol_copy, EXPECTED_PROTOCOL_SHA256, "D3 protocol copy")

    target_manifest_sha = sha256_file(manifest_path)
    order_sha = target_order_sha(targets)

    gt_rule_path = build_dir / "frozen_d4_gt_and_metric_rule.json"
    if not gt_rule_path.exists():
        write_json(gt_rule_path, GT_RULE)

    checkpoint_manifest_path = build_dir / "checkpoint_manifest.csv"
    if not checkpoint_manifest_path.exists():
        write_csv(
            checkpoint_manifest_path,
            panel,
            [
                "model_family",
                "training_seed",
                "model_state_id",
                "checkpoint",
                "checkpoint_sha256",
            ],
        )

    upstream = {
        "script_version": VERSION,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "r05d1_lock_sha256": EXPECTED_R05D1_LOCK_SHA256,
        "r05d1_decision": d1.get("decision"),
        "r05d2_lock_sha256": EXPECTED_R05D2_LOCK_SHA256,
        "r05d2_decision": d2.get("decision"),
        "r05d2_probability_table_sha256": EXPECTED_D2_TABLE_SHA256,
        "r05d2_probability_table_verified_by_hash_only": True,
        "r03_lock_sha256": EXPECTED_R03_LOCK_SHA256,
        "r03_decision": r03_lock.get("decision"),
        "target_manifest_sha256": target_manifest_sha,
        "target_id_order_sha256": order_sha,
        "target_cases": len(targets),
        "model_states": len(panel),
        "expected_model_cases": EXPECTED_MODEL_CASES,
        "target_gt_pixels_decoded": False,
        "paot_probability_values_read": False,
        "paot_gating_used": False,
        "target_fit_or_tuning": False,
    }
    upstream_path = build_dir / "upstream_audit.json"
    write_json(upstream_path, upstream)

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for Q1-R05D3 formal run.")
    device = torch.device("cuda")

    print("===== Q1-R05D3 SOURCE + A1 PREDICTION GENERATION =====")
    print(f"Device={device}")
    print(f"Target cases={len(targets)}")
    print(f"Model states={len(panel)}")
    print(f"Expected model-cases={EXPECTED_MODEL_CASES}")
    print("Action=A1_TENT_1STEP")
    print("D2 probability values read=NO")
    print("PAOT gating=NO")
    print("Target GT pixels decoded=NO")
    print()

    global_rows = []
    seg_context = None

    for state_idx, state in enumerate(panel, start=1):
        print(
            f"[STATE {state_idx}/{len(panel)}] "
            f"{state['model_state_id']} "
            f"checkpoint={state['checkpoint_sha256'][:12]}..."
        )

        cached_rows = load_state(
            state_dir,
            state,
            targets,
            target_manifest_sha,
            order_sha,
            allow_orphan_cleanup=bool(args.resume),
        )

        pred_path, index_path, _ = state_paths(state_dir, state)

        if cached_rows is not None:
            print(f"[RESUME] verified state rows={len(cached_rows)}")
            global_rows.extend(
                global_rows_from_state(cached_rows, pred_path, index_path)
            )
            continue

        if state["model_family"] == "PraNet":
            source_pack, a1_pack, index_rows = run_pranet_state(
                state,
                targets,
                device,
            )
        elif state["model_family"] == "DeepLabV3-R50":
            source_pack, a1_pack, index_rows = run_deeplab_state(
                state,
                targets,
                device,
            )
        elif state["model_family"] == "SegFormer-B0":
            (
                source_pack,
                a1_pack,
                index_rows,
                seg_context,
            ) = run_segformer_state(
                state,
                targets,
                device,
                r03,
                seg_context=seg_context,
            )
        else:
            raise RuntimeError(f"Unknown family: {state['model_family']}")

        validate_prediction_arrays(source_pack, a1_pack)
        validate_index_rows(index_rows, state, targets)

        pred_path, index_path, _ = save_state(
            state_dir,
            state,
            targets,
            source_pack,
            a1_pack,
            index_rows,
            target_manifest_sha,
            order_sha,
        )

        verified_rows = load_state(
            state_dir,
            state,
            targets,
            target_manifest_sha,
            order_sha,
            allow_orphan_cleanup=False,
        )
        if verified_rows is None:
            raise RuntimeError("Newly saved state failed verification.")

        global_rows.extend(
            global_rows_from_state(verified_rows, pred_path, index_path)
        )

        print(f"[PASS] locked state rows={len(verified_rows)}")

        del source_pack, a1_pack, index_rows, verified_rows
        torch.cuda.empty_cache()

    if len(global_rows) != EXPECTED_MODEL_CASES:
        raise RuntimeError(
            f"Global rows={len(global_rows)} expected={EXPECTED_MODEL_CASES}"
        )

    unique_keys = {
        (r["model_state_id"], r["sample_id"])
        for r in global_rows
    }
    if len(unique_keys) != EXPECTED_MODEL_CASES:
        raise RuntimeError("Duplicate/missing model-case keys.")

    state_counts = Counter(r["model_state_id"] for r in global_rows)
    if len(state_counts) != EXPECTED_STATES:
        raise RuntimeError("Global state count mismatch.")
    if any(v != EXPECTED_CASES for v in state_counts.values()):
        raise RuntimeError(f"Per-state count mismatch: {dict(state_counts)}")

    global_rows.sort(
        key=lambda r: (
            r["model_family"],
            int(r["training_seed"]),
            r["sample_id"],
        )
    )

    global_manifest_path = build_dir / "model_case_prediction_manifest.csv"
    write_csv(global_manifest_path, global_rows, GLOBAL_FIELDS)

    # Prediction-only descriptive summary; never used for selection.
    grouped = defaultdict(list)
    for row in global_rows:
        grouped[(row["model_family"], row["model_state_id"])].append(row)
        grouped[(row["model_family"], "ALL_3_STATES")].append(row)
        grouped[("ALL_ARCHITECTURES", "ALL_9_STATES")].append(row)

    summary_rows = []
    for (family, state_id), rows in sorted(grouped.items()):
        src = np.asarray(
            [int(r["source_foreground_pixels"]) / MASK_PIXELS for r in rows],
            dtype=np.float64,
        )
        a1 = np.asarray(
            [int(r["a1_foreground_pixels"]) / MASK_PIXELS for r in rows],
            dtype=np.float64,
        )
        chg = np.asarray(
            [int(r["changed_pixels"]) / MASK_PIXELS for r in rows],
            dtype=np.float64,
        )
        summary_rows.append({
            "model_family": family,
            "model_state_id": state_id,
            "rows": len(rows),
            "source_fg_fraction_mean": float(src.mean()),
            "a1_fg_fraction_mean": float(a1.mean()),
            "changed_fraction_mean": float(chg.mean()),
            "changed_fraction_q50": float(np.quantile(chg, 0.50)),
            "changed_fraction_q90": float(np.quantile(chg, 0.90)),
            "zero_change_fraction": float(np.mean(chg == 0.0)),
        })

    summary_path = build_dir / "prediction_change_summary.csv"
    write_csv(
        summary_path,
        summary_rows,
        [
            "model_family",
            "model_state_id",
            "rows",
            "source_fg_fraction_mean",
            "a1_fg_fraction_mean",
            "changed_fraction_mean",
            "changed_fraction_q50",
            "changed_fraction_q90",
            "zero_change_fraction",
        ],
    )

    boundary = {
        "target_image_pixels_decoded": True,
        "target_gt_paths_present_as_provenance_strings": True,
        "target_gt_pixels_decoded": False,
        "source_inference_run": True,
        "a1_tent_1step_run": True,
        "a1_optimizer": "Adam",
        "a1_lr": TENT_LR,
        "a1_weight_decay": TENT_WEIGHT_DECAY,
        "a1_steps_per_model_case": TENT_STEPS,
        "episodic_reset": True,
        "model_case_units_receiving_a1": EXPECTED_MODEL_CASES,
        "d2_probability_table_sha_verified": True,
        "d2_probability_values_read": False,
        "paot_probability_threshold_selected": False,
        "paot_gating_used": False,
        "predictor_fit_or_modified": False,
        "target_normalization_fit": False,
        "target_calibration": False,
        "target_hyperparameter_sweep": False,
        "target_case_exclusion_from_prediction_behavior": False,
        "target_dice_computed": False,
        "target_delta_dice_computed": False,
        "target_outcome_labels_computed": False,
    }
    boundary_path = build_dir / "information_boundary_audit.json"
    write_json(boundary_path, boundary)

    decision_path = build_dir / "decision.txt"
    decision_path.write_text(DECISION_READY + "\n", encoding="utf-8")

    run_lines = [
        "===== Q1-R05D3 NEOPOLYP SOURCE + A1 PREDICTION LOCK =====",
        f"Script version: {VERSION}",
        f"Build: {BUILD}",
        "",
        "Frozen target:",
        f"  NeoPolyp cases={EXPECTED_CASES}",
        "  exact overlap exclusions=0",
        "",
        "Frozen panel/action:",
        "  PraNet states=3",
        "  DeepLabV3-R50 states=3",
        "  SegFormer-B0 states=3",
        f"  total states={EXPECTED_STATES}",
        "  action=A1_TENT_1STEP",
        "  optimizer=Adam",
        f"  lr={TENT_LR}",
        f"  weight_decay={TENT_WEIGHT_DECAY}",
        "  steps=1",
        "  episodic reset=YES",
        "",
        "Prediction lock:",
        f"  model-case rows={len(global_rows)}",
        f"  SOURCE masks={len(global_rows)}",
        f"  A1 masks={len(global_rows)}",
        f"  mask resolution={MASK_H}x{MASK_W}",
        f"  packed bytes/mask={PACKED_BYTES}",
        f"  bitorder={BITORDER}",
        "",
        "Information boundary:",
        "  target image pixels decoded=YES",
        "  target GT pixels decoded=NO",
        "  D2 probability table SHA verified=YES",
        "  D2 probability values read=NO",
        "  PAOT gating used=NO",
        f"  all {EXPECTED_MODEL_CASES} model-case units received A1=YES",
        "  predictor fit/modified=NO",
        "  target tuning/calibration=NO",
        "  target Dice/DeltaDice/outcomes=NO",
        "",
        "Future D4 GT rule frozen:",
        "  black background -> binary background",
        "  green non-neoplastic -> binary foreground",
        "  red neoplastic -> binary foreground",
        "  RGB assignment=nearest fixed prototype",
        "  GT resize=352x352 nearest",
        "  HARM <= -0.02",
        "  BENEFIT >= +0.02",
        "",
        "Decision:",
        f"  {DECISION_READY}",
        "",
        "Next:",
        "  Q1-R05D4 GT reveal + PAOT confirmatory evaluation",
    ]
    run_log_path = build_dir / "run_log.txt"
    run_log_path.write_text(
        "\n".join(run_lines) + "\n",
        encoding="utf-8",
    )

    artifacts = {
        "protocol_copy": protocol_copy,
        "upstream_audit": upstream_path,
        "checkpoint_manifest": checkpoint_manifest_path,
        "model_case_prediction_manifest": global_manifest_path,
        "prediction_change_summary": summary_path,
        "frozen_d4_gt_and_metric_rule": gt_rule_path,
        "information_boundary_audit": boundary_path,
        "decision": decision_path,
        "run_log": run_log_path,
    }

    for state in panel:
        pred_path, index_path, state_lock_path = state_paths(state_dir, state)
        if not pred_path.exists() or not index_path.exists() or not state_lock_path.exists():
            raise RuntimeError(f"Missing state artifacts: {state['model_state_id']}")
        token = state_token(state)
        artifacts[f"state_npz__{token}"] = pred_path
        artifacts[f"state_index__{token}"] = index_path
        artifacts[f"state_lock__{token}"] = state_lock_path

    lock = {
        "script_version": VERSION,
        "build": BUILD,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "r05d1_lock_sha256": EXPECTED_R05D1_LOCK_SHA256,
        "r05d2_lock_sha256": EXPECTED_R05D2_LOCK_SHA256,
        "r05d2_probability_table_sha256": EXPECTED_D2_TABLE_SHA256,
        "r03_lock_sha256": EXPECTED_R03_LOCK_SHA256,
        "target_manifest_sha256": target_manifest_sha,
        "target_id_order_sha256": order_sha,
        "target_cases": EXPECTED_CASES,
        "model_states": EXPECTED_STATES,
        "model_case_rows": EXPECTED_MODEL_CASES,
        "source_prediction_masks": EXPECTED_MODEL_CASES,
        "a1_prediction_masks": EXPECTED_MODEL_CASES,
        "mask_height": MASK_H,
        "mask_width": MASK_W,
        "packed_bytes_per_mask": PACKED_BYTES,
        "bitorder": BITORDER,
        "action": "A1_TENT_1STEP",
        "optimizer": "Adam",
        "lr": TENT_LR,
        "weight_decay": TENT_WEIGHT_DECAY,
        "steps": TENT_STEPS,
        "episodic_reset": True,
        "all_model_cases_received_a1": True,
        "d2_probability_values_read": False,
        "paot_gating_used": False,
        "target_gt_pixels_decoded": False,
        "target_outcomes_revealed": False,
        "frozen_d4_gt_rule": GT_RULE,
        "decision": DECISION_READY,
        "artifacts": {
            name: {
                "relative_path": str(path.relative_to(build_dir)),
                "sha256": sha256_file(path),
            }
            for name, path in artifacts.items()
        },
    }

    final_lock_path = (
        build_dir / "Q1_R05D3_NEOPOLYP_SOURCE_A1_PREDICTION_LOCK.json"
    )
    write_json(final_lock_path, lock)
    lock_sha = sha256_file(final_lock_path)

    for name, meta in lock["artifacts"].items():
        path = build_dir / meta["relative_path"]
        if sha256_file(path) != meta["sha256"]:
            raise RuntimeError(f"Artifact changed before commit: {name}")

    build_dir.rename(args.output_dir)

    print()
    print((args.output_dir / "run_log.txt").read_text(encoding="utf-8"))
    print(
        "Q1-R05D3 LOCK:",
        args.output_dir / "Q1_R05D3_NEOPOLYP_SOURCE_A1_PREDICTION_LOCK.json",
    )
    print("Q1-R05D3 LOCK SHA256:", lock_sha)


# ---------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------

def self_test():
    assert EXPECTED_CASES == 1000
    assert EXPECTED_STATES == 9
    assert EXPECTED_MODEL_CASES == 9000
    assert MASK_H == 352 and MASK_W == 352
    assert MASK_PIXELS == 123904
    assert PACKED_BYTES == 15488

    assert TENT_LR == 1e-3
    assert TENT_WEIGHT_DECAY == 0.0
    assert TENT_STEPS == 1

    rng = np.random.RandomState(20260819)
    mask = (rng.rand(MASK_H, MASK_W) >= 0.7).astype(np.uint8)
    packed = pack_mask(mask)
    restored = unpack_mask(packed)
    assert np.array_equal(mask, restored)

    assert GT_RULE["binary_mapping"]["background"] == 0
    assert GT_RULE["binary_mapping"]["non_neoplastic_polyp"] == 1
    assert GT_RULE["binary_mapping"]["neoplastic_polyp"] == 1
    assert GT_RULE["future_decode_rule"] == "nearest_RGB_prototype_euclidean"
    assert GT_RULE["harm_threshold_delta_dice"] == -0.02
    assert GT_RULE["benefit_threshold_delta_dice"] == 0.02

    print("CARDINALITY_TEST_PASS")
    print("PACKBITS_ROUNDTRIP_TEST_PASS")
    print("FROZEN_A1_HYPERPARAMETER_TEST_PASS")
    print("FROZEN_D4_BINARY_GT_MAPPING_TEST_PASS")
    print("SELF_TEST_PASS")


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Q1-R05D3 FIX1: execute and lock SOURCE + A1_TENT_1STEP "
            "predictions for all 1000x9 NeoPolyp model-case pairs before GT reveal."
        )
    )
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    p.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume only fully verified per-state prediction caches after "
            "a technical interruption."
        ),
    )
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--self-test", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    if args.self_test:
        self_test()
        return 0

    if args.preflight_only:
        preflight()
        return 0

    try:
        run(args)
    except Exception:
        traceback.print_exc()
        raise

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
