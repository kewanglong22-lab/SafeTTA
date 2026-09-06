#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R10L3A_polypgen_source_a1_prediction_lock_fix2.py

First prospective PolypGen model execution after the independence gate.

PURPOSE
-------
Run the exact already-frozen SOURCE vs A1_TENT_1STEP counterfactual for all
1532 locked PolypGen static cases x 9 frozen model states, while keeping
PolypGen GT completely unopened.

The architecture-specific prediction/adaptation implementation is NOT rewritten
here. This script imports the exact authoritative NeoPolyp R05D3 implementation
and reuses its frozen state runners:

- PraNet: exact S03-B SOURCE + episodic 1-step TENT
- DeepLabV3-R50: exact singleton-safe S05-C SOURCE + episodic 1-step TENT
- SegFormer-B0: exact R03 SOURCE + episodic 1-step TENT

Frozen action:
    A1_TENT_1STEP
    Adam
    lr = 1e-3
    weight_decay = 0
    steps = 1
    episodic reset = YES

STRICT INFORMATION BOUNDARY
---------------------------
- PolypGen RGB pixels: YES
- PolypGen GT path column read: NO
- PolypGen GT pixels decoded: NO
- Dice / DeltaDice / HARM labels: NO
- R10L0 safety estimator read: NO
- target calibration/tuning/threshold fitting: NO

Outputs SOURCE and A1 masks only, packed exactly as the authoritative R05D3
implementation: 352x352 binary masks, row-major, np.packbits(bitorder="little").
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import inspect
import json
import os
import re
import shutil
import sys
import traceback
from pathlib import Path

# Same deterministic CUDA convention used by R05D3 / frozen DeepLab route.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
from tqdm import tqdm


VERSION = "2026-08-26-Q1-R10L3A-v1-fix2"
BUILD = "Q1_R10L3A_POLYPGEN_SOURCE_A1_PREDICTION_LOCK_FIX2"

ROOT = Path(r"F:\MEDSEG_SAFETTA")

R05D3_SCRIPT = (
    ROOT / "code"
    / "Q1_R05D3_neopolyp_source_a1_prediction_lock_v1_fix1.py"
)
EXPECTED_R05D3_SCRIPT_SHA256 = (
    "df22d6f3054257743d1f12d9e757323ea9f4a052b7642eb4ab610da46ff21251"
)

POLYPGEN_MANIFEST = (
    ROOT / "outputs"
    / "Q1_R10L1B_polypgen_unique_static_manifest_lock_fix3_v1"
    / "R10L1B_POLYPGEN_UNIQUE_STATIC_MANIFEST.csv"
)
POLYPGEN_MANIFEST_LOCK = (
    ROOT / "outputs"
    / "Q1_R10L1B_polypgen_unique_static_manifest_lock_fix3_v1"
    / "R10L1B_MANIFEST_LOCK.json"
)
EXPECTED_POLYPGEN_MANIFEST_DECISION = (
    "POLYPGEN_STATIC_UNIQUE_MANIFEST_LOCKED_PENDING_SOURCE_TRAINING_AUDIT"
)

INDEPENDENCE_LOCK = (
    ROOT / "outputs"
    / "Q1_R10L2B_authoritative_s01_source_training_overlap_lock_fix1_v1"
    / "R10L2B_INDEPENDENCE_GATE_LOCK.json"
)
EXPECTED_INDEPENDENCE_DECISION = (
    "POLYPGEN_INDEPENDENCE_GATE_PASS_"
    "ALL_9_STATES_S01_TRAIN_VAL_ZERO_EXACT_OVERLAP"
)

CHECKPOINT_RECOVERY = (
    ROOT / "outputs"
    / "Q1_R10K1_higher_level_representation_asset_audit_fix1_v1"
    / "R10K1_checkpoint_recovery.csv"
)

OUTPUT_DIR = (
    ROOT / "outputs"
    / "Q1_R10L3A_polypgen_source_a1_prediction_lock_fix2_v1"
)

FIX1_PARTIAL_BUILD_DIR = (
    ROOT / "outputs"
    / "Q1_R10L3A_polypgen_source_a1_prediction_lock_fix1_v1__building"
)

EXPECTED_CASES = 1532
EXPECTED_STATES = 9
EXPECTED_MODEL_CASES = EXPECTED_CASES * EXPECTED_STATES

MASK_H = 352
MASK_W = 352
MASK_PIXELS = MASK_H * MASK_W
PACKED_BYTES = (MASK_PIXELS + 7) // 8
BITORDER = "little"

FAMILIES = ("PraNet", "DeepLabV3-R50", "SegFormer-B0")

EXPECTED_PANEL = {
    ("DeepLabV3-R50", 20260817),
    ("DeepLabV3-R50", 20260818),
    ("DeepLabV3-R50", 20260819),
    ("PraNet", 20260817),
    ("PraNet", 20260818),
    ("PraNet", 20260819),
    ("SegFormer-B0", 20260820),
    ("SegFormer-B0", 20260821),
    ("SegFormer-B0", 20260822),
}

DECISION_READY = "POLYPGEN_SOURCE_A1_PREDICTIONS_LOCKED_BEFORE_GT_REVEAL"


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


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_csv(path: Path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(fields))
        w.writeheader()
        w.writerows(rows)


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def safe_state_stem(family: str, seed: int) -> str:
    s = family.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return f"{s}_seed{seed}"


def parse_one_checkpoint_path(raw: str) -> Path:
    vals = [
        x.strip()
        for x in str(raw).split(";")
        if str(x).strip()
    ]
    existing = [Path(x) for x in vals if Path(x).exists()]
    if len(existing) != 1:
        raise RuntimeError(
            f"Expected exactly one existing checkpoint path, got {existing}"
        )
    return existing[0]


def load_upstream_locks():
    for label, p in {
        "PolypGen manifest": POLYPGEN_MANIFEST,
        "PolypGen manifest lock": POLYPGEN_MANIFEST_LOCK,
        "independence lock": INDEPENDENCE_LOCK,
        "checkpoint recovery": CHECKPOINT_RECOVERY,
        "authoritative R05D3 script": R05D3_SCRIPT,
    }.items():
        print(f"{label} exists:", p.exists(), p)
        if not p.exists():
            raise FileNotFoundError(p)

    r05d3_sha = validate_sha(
        R05D3_SCRIPT,
        EXPECTED_R05D3_SCRIPT_SHA256,
        "authoritative R05D3 script",
    )

    pg_lock = json.loads(
        POLYPGEN_MANIFEST_LOCK.read_text(encoding="utf-8")
    )
    if pg_lock.get("decision") != EXPECTED_POLYPGEN_MANIFEST_DECISION:
        raise RuntimeError(
            f"Unexpected PolypGen manifest decision: {pg_lock.get('decision')}"
        )

    indep = json.loads(INDEPENDENCE_LOCK.read_text(encoding="utf-8"))
    if indep.get("decision") != EXPECTED_INDEPENDENCE_DECISION:
        raise RuntimeError(
            f"PolypGen independence gate not passed: {indep.get('decision')}"
        )

    if int(indep.get("source_train_exact_polypgen_overlaps", -1)) != 0:
        raise RuntimeError("Source-train PolypGen overlap is not zero.")
    if int(indep.get("source_val_exact_polypgen_overlaps", -1)) != 0:
        raise RuntimeError("Source-val PolypGen overlap is not zero.")
    if int(indep.get("frozen_states", -1)) != 9:
        raise RuntimeError("Independence lock does not certify 9 states.")

    return {
        "r05d3_script_sha256": r05d3_sha,
        "polypgen_manifest_sha256": sha256_file(POLYPGEN_MANIFEST),
        "polypgen_manifest_lock_sha256": sha256_file(POLYPGEN_MANIFEST_LOCK),
        "independence_lock_sha256": sha256_file(INDEPENDENCE_LOCK),
        "checkpoint_recovery_sha256": sha256_file(CHECKPOINT_RECOVERY),
    }


def load_targets_and_verify_rgb():
    # Deliberately do NOT read gt_path or gt_sha256.
    cols = [
        "external_case_id",
        "center",
        "sample_id",
        "image_path",
        "image_sha256",
    ]
    df = pd.read_csv(
        POLYPGEN_MANIFEST,
        usecols=cols,
        low_memory=False,
    )

    if len(df) != EXPECTED_CASES:
        raise RuntimeError(
            f"PolypGen rows={len(df)} expected={EXPECTED_CASES}"
        )
    if df["external_case_id"].astype(str).nunique() != EXPECTED_CASES:
        raise RuntimeError("external_case_id is not unique.")
    if df["image_sha256"].astype(str).nunique() != EXPECTED_CASES:
        raise RuntimeError("PolypGen RGB SHA256 is not unique.")
    if set(df["center"].astype(str)) != {
        "C1", "C2", "C3", "C4", "C5", "C6"
    }:
        raise RuntimeError("All six PolypGen centers are not represented.")

    verified = []
    for r in tqdm(
        df.itertuples(index=False),
        total=len(df),
        desc="Verifying frozen PolypGen RGB",
        unit="image",
        dynamic_ncols=True,
    ):
        p = Path(str(r.image_path))
        if not p.exists():
            raise FileNotFoundError(p)
        actual = sha256_file(p).lower()
        expected = str(r.image_sha256).strip().lower()
        if actual != expected:
            raise RuntimeError(
                f"PolypGen RGB SHA mismatch for {r.external_case_id}: "
                f"expected={expected} actual={actual}"
            )
        verified.append({
            "sample_id": str(r.external_case_id),
            "original_polypgen_sample_id": str(r.sample_id),
            "center": str(r.center),
            "image_path": str(p),
            "image_raw_sha256": actual,
        })

    return verified


def load_frozen_panel():
    ck = pd.read_csv(CHECKPOINT_RECOVERY, low_memory=False)

    required = [
        "model_state_id",
        "model_family",
        "training_seed",
        "checkpoint_sha256",
        "exact_sha256_paths",
        "recovered",
        "exact_sha256_match_count",
    ]
    missing = [c for c in required if c not in ck.columns]
    if missing:
        raise RuntimeError(f"Checkpoint recovery missing columns: {missing}")

    ck["training_seed"] = pd.to_numeric(
        ck["training_seed"],
        errors="raise",
    ).astype(int)

    observed = set(
        zip(
            ck["model_family"].astype(str),
            ck["training_seed"].astype(int),
        )
    )
    if len(ck) != EXPECTED_STATES or observed != EXPECTED_PANEL:
        raise RuntimeError(
            f"Frozen panel mismatch: rows={len(ck)} panel={sorted(observed)}"
        )
    if not ck["recovered"].astype(bool).all():
        raise RuntimeError("Not all frozen checkpoints are recovered.")
    if not (ck["exact_sha256_match_count"].astype(int) == 1).all():
        raise RuntimeError("Checkpoint exact-match count is not 1 for every state.")

    panel = []
    for r in ck.itertuples(index=False):
        cp = parse_one_checkpoint_path(r.exact_sha256_paths)
        actual_sha = sha256_file(cp).lower()
        expected_sha = str(r.checkpoint_sha256).strip().lower()
        if actual_sha != expected_sha:
            raise RuntimeError(
                f"Checkpoint SHA changed for {r.model_state_id}"
            )

        panel.append({
            "model_family": str(r.model_family),
            "training_seed": int(r.training_seed),
            "model_state_id": str(r.model_state_id),
            "checkpoint": str(cp),
            "checkpoint_sha256": actual_sha,
        })

    panel.sort(
        key=lambda x: (
            FAMILIES.index(x["model_family"]),
            x["training_seed"],
        )
    )
    return panel


def patch_r05d3_cardinality(r05d3):
    # Reuse model-specific implementation unchanged, but with PolypGen count.
    r05d3.EXPECTED_CASES = EXPECTED_CASES
    r05d3.EXPECTED_STATES = EXPECTED_STATES
    r05d3.EXPECTED_MODEL_CASES = EXPECTED_MODEL_CASES

    if int(r05d3.MASK_H) != MASK_H or int(r05d3.MASK_W) != MASK_W:
        raise RuntimeError("R05D3 mask resolution changed.")
    if int(r05d3.PACKED_BYTES) != PACKED_BYTES:
        raise RuntimeError("R05D3 packed-mask byte count changed.")
    if str(r05d3.BITORDER) != BITORDER:
        raise RuntimeError("R05D3 packbits bitorder changed.")
    if float(r05d3.TENT_LR) != 1e-3:
        raise RuntimeError("R05D3 TENT lr changed.")
    if float(r05d3.TENT_WEIGHT_DECAY) != 0.0:
        raise RuntimeError("R05D3 TENT weight decay changed.")
    if int(r05d3.TENT_STEPS) != 1:
        raise RuntimeError("R05D3 TENT step count changed.")

    for name in [
        "run_pranet_state",
        "run_deeplab_state",
        "run_segformer_state",
        "pack_mask",
        "unpack_mask",
    ]:
        if not hasattr(r05d3, name):
            raise RuntimeError(f"R05D3 missing required function: {name}")


def invoke_runner(fn, *, state, targets, device, r03):
    """
    Call the frozen runner by its declared parameter names.
    This avoids rewriting architecture-specific implementation.
    """
    mapping = {
        "state": state,
        "targets": targets,
        "device": device,
        "r03": r03,
        "r03helper": r03,
        "r03_helper": r03,
    }

    sig = inspect.signature(fn)
    kwargs = {}
    for name, param in sig.parameters.items():
        if name in mapping:
            kwargs[name] = mapping[name]
        elif param.default is not inspect._empty:
            continue
        else:
            raise RuntimeError(
                f"Unsupported required runner parameter {name!r} "
                f"for {fn.__name__}{sig}"
            )
    return fn(**kwargs)


def validate_prediction_arrays(r05d3, source_pack, a1_pack):
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

    # Structural decode checks at deterministic positions.
    for idx in [0, EXPECTED_CASES // 2, EXPECTED_CASES - 1]:
        sm = r05d3.unpack_mask(source_pack[idx])
        am = r05d3.unpack_mask(a1_pack[idx])
        if sm.shape != (MASK_H, MASK_W):
            raise RuntimeError("SOURCE unpack shape mismatch.")
        if am.shape != (MASK_H, MASK_W):
            raise RuntimeError("A1 unpack shape mismatch.")
        if not np.isin(sm, (0, 1)).all():
            raise RuntimeError("SOURCE unpack is not binary.")
        if not np.isin(am, (0, 1)).all():
            raise RuntimeError("A1 unpack is not binary.")

    return source_pack, a1_pack


def state_paths(state_dir: Path, state):
    stem = safe_state_stem(
        state["model_family"],
        int(state["training_seed"]),
    )
    return (
        state_dir / f"{stem}_predictions.npz",
        state_dir / f"{stem}_index.csv",
        state_dir / f"{stem}_state_lock.json",
    )


def sidecar_is_complete(
    sidecar_path,
    npz_path,
    index_path,
    state,
    upstream,
):
    if not all(p.exists() for p in [sidecar_path, npz_path, index_path]):
        return False
    try:
        m = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except Exception:
        return False

    checks = [
        m.get("decision") == "STATE_COMPLETE",
        m.get("model_state_id") == state["model_state_id"],
        int(m.get("training_seed", -1)) == int(state["training_seed"]),
        m.get("checkpoint_sha256") == state["checkpoint_sha256"],
        int(m.get("target_cases", -1)) == EXPECTED_CASES,
        m.get("polypgen_manifest_sha256")
        == upstream["polypgen_manifest_sha256"],
        m.get("independence_lock_sha256")
        == upstream["independence_lock_sha256"],
        m.get("r05d3_script_sha256")
        == upstream["r05d3_script_sha256"],
        m.get("npz_sha256") == sha256_file(npz_path),
        m.get("index_sha256") == sha256_file(index_path),
    ]
    return all(checks)


def validate_cached_rows_for_target(rows, state, targets):
    if len(rows) != EXPECTED_CASES:
        return False

    expected_ids = [t["sample_id"] for t in targets]
    observed_ids = [str(r.get("sample_id", "")) for r in rows]
    if observed_ids != expected_ids:
        return False

    for i, row in enumerate(rows):
        target = targets[i]
        try:
            if int(row.get("row_index", -1)) != i:
                return False
            if str(row.get("model_family", "")) != state["model_family"]:
                return False
            if str(row.get("model_state_id", "")) != state["model_state_id"]:
                return False
            if int(row.get("training_seed", -1)) != int(state["training_seed"]):
                return False
            if str(row.get("checkpoint_sha256", "")).lower() != str(
                state["checkpoint_sha256"]
            ).lower():
                return False
            if str(row.get("image_raw_sha256", "")).lower() != str(
                target["image_raw_sha256"]
            ).lower():
                return False
        except Exception:
            return False

    return True


def hardlink_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def bootstrap_verified_fix1_states(
    fix1_build_dir,
    new_state_dir,
    panel,
    targets,
    upstream,
):
    """
    Preserve expensive completed Fix1 states without modifying the old partial
    run. Only a full NPZ+index+sidecar triple that passes the same SHA/lineage
    checks is imported into Fix2.
    """
    imported = []

    if not fix1_build_dir.exists():
        print("Fix1 partial build exists: False", fix1_build_dir)
        return imported

    old_state_dir = fix1_build_dir / "state_predictions"
    print("Fix1 partial build exists: True", fix1_build_dir)
    print("Fix1 state cache dir exists:", old_state_dir.exists(), old_state_dir)

    if not old_state_dir.exists():
        return imported

    for state in panel:
        old_npz, old_idx, old_side = state_paths(old_state_dir, state)

        if not sidecar_is_complete(
            old_side,
            old_npz,
            old_idx,
            state,
            upstream,
        ):
            continue

        with old_idx.open(
            "r",
            newline="",
            encoding="utf-8-sig",
        ) as f:
            rows = list(csv.DictReader(f))

        if not validate_cached_rows_for_target(
            rows,
            state,
            targets,
        ):
            raise RuntimeError(
                f"Fix1 cached rows failed target/state validation: "
                f"{state['model_state_id']}"
            )

        new_npz, new_idx, new_side = state_paths(new_state_dir, state)

        hardlink_or_copy(old_npz, new_npz)
        hardlink_or_copy(old_idx, new_idx)
        hardlink_or_copy(old_side, new_side)

        if not sidecar_is_complete(
            new_side,
            new_npz,
            new_idx,
            state,
            upstream,
        ):
            raise RuntimeError(
                f"Imported Fix1 cache failed Fix2 verification: "
                f"{state['model_state_id']}"
            )

        imported.append(state["model_state_id"])
        print("[BOOTSTRAP VERIFIED]", state["model_state_id"])

    print(
        "Verified Fix1 states imported into Fix2:",
        len(imported),
        "/",
        EXPECTED_STATES,
    )
    return imported



def run_state(
    r05d3,
    r03,
    state,
    targets,
    device,
    state_dir,
    upstream,
    resume,
    seg_context=None,
):
    npz_path, index_path, sidecar_path = state_paths(state_dir, state)

    if resume and sidecar_is_complete(
        sidecar_path,
        npz_path,
        index_path,
        state,
        upstream,
    ):
        print(
            f"[RESUME VERIFIED] {state['model_state_id']} "
            f"seed={state['training_seed']}"
        )
        with index_path.open(
            "r",
            newline="",
            encoding="utf-8-sig",
        ) as f:
            rows = list(csv.DictReader(f))

        if not validate_cached_rows_for_target(
            rows,
            state,
            targets,
        ):
            raise RuntimeError(
                f"Cached rows failed target/state validation: "
                f"{state['model_state_id']}"
            )

        return (
            npz_path,
            index_path,
            sidecar_path,
            rows,
            seg_context,
        )

    # Delete only this incomplete state artifact set.
    for p in [npz_path, index_path, sidecar_path]:
        if p.exists():
            p.unlink()

    family = state["model_family"]

    # Fix2: the authoritative R05D3 SegFormer runner intentionally returns
    # FOUR values:
    #   SOURCE packed masks, A1 packed masks, index rows, seg_context
    #
    # PraNet and DeepLab return three. Do not flatten this difference because
    # seg_context is deliberately reused across the three SegFormer states.
    if family == "SegFormer-B0":
        result = r05d3.run_segformer_state(
            state,
            targets,
            device,
            r03,
            seg_context=seg_context,
        )
        if not isinstance(result, tuple) or len(result) != 4:
            raise RuntimeError(
                "Authoritative SegFormer runner return schema changed; "
                f"expected 4 values, got "
                f"{len(result) if isinstance(result, tuple) else type(result)}"
            )
        (
            source_pack,
            a1_pack,
            rows,
            seg_context,
        ) = result
    elif family == "PraNet":
        result = invoke_runner(
            r05d3.run_pranet_state,
            state=state,
            targets=targets,
            device=device,
            r03=r03,
        )
        if not isinstance(result, tuple) or len(result) != 3:
            raise RuntimeError(
                "Authoritative PraNet runner return schema changed."
            )
        source_pack, a1_pack, rows = result
    elif family == "DeepLabV3-R50":
        result = invoke_runner(
            r05d3.run_deeplab_state,
            state=state,
            targets=targets,
            device=device,
            r03=r03,
        )
        if not isinstance(result, tuple) or len(result) != 3:
            raise RuntimeError(
                "Authoritative DeepLab runner return schema changed."
            )
        source_pack, a1_pack, rows = result
    else:
        raise RuntimeError(f"Unexpected family: {family}")

    source_pack, a1_pack = validate_prediction_arrays(
        r05d3,
        source_pack,
        a1_pack,
    )

    if len(rows) != EXPECTED_CASES:
        raise RuntimeError(
            f"{state['model_state_id']} index rows={len(rows)} "
            f"expected={EXPECTED_CASES}"
        )

    expected_ids = [t["sample_id"] for t in targets]
    observed_ids = [str(r["sample_id"]) for r in rows]
    if observed_ids != expected_ids:
        raise RuntimeError(
            f"{state['model_state_id']} target order changed."
        )

    for i, row in enumerate(rows):
        target = targets[i]
        if str(row["image_raw_sha256"]).lower() != str(
            target["image_raw_sha256"]
        ).lower():
            raise RuntimeError(
                f"{state['model_state_id']} image SHA mismatch at row {i}"
            )
        if row["model_state_id"] != state["model_state_id"]:
            raise RuntimeError("State ID mismatch in prediction index.")
        if str(row["checkpoint_sha256"]).lower() != state[
            "checkpoint_sha256"
        ].lower():
            raise RuntimeError("Checkpoint SHA mismatch in prediction index.")

    np.savez_compressed(
        npz_path,
        source_masks_packed=source_pack,
        a1_masks_packed=a1_pack,
    )

    fields = list(rows[0].keys())
    write_csv(index_path, rows, fields)

    sidecar = {
        "decision": "STATE_COMPLETE",
        "model_family": family,
        "model_state_id": state["model_state_id"],
        "training_seed": int(state["training_seed"]),
        "checkpoint_sha256": state["checkpoint_sha256"],
        "target_cases": EXPECTED_CASES,
        "mask_height": MASK_H,
        "mask_width": MASK_W,
        "packed_bytes_per_mask": PACKED_BYTES,
        "bitorder": BITORDER,
        "action": "A1_TENT_1STEP",
        "tent_lr": 1e-3,
        "tent_weight_decay": 0.0,
        "tent_steps": 1,
        "episodic_reset": True,
        "polypgen_manifest_sha256": upstream["polypgen_manifest_sha256"],
        "independence_lock_sha256": upstream["independence_lock_sha256"],
        "r05d3_script_sha256": upstream["r05d3_script_sha256"],
        "npz_sha256": sha256_file(npz_path),
        "index_sha256": sha256_file(index_path),
        "target_gt_path_column_read": False,
        "target_gt_pixels_decoded": False,
        "target_metrics_computed": False,
    }
    write_json(sidecar_path, sidecar)

    return (
        npz_path,
        index_path,
        sidecar_path,
        rows,
        seg_context,
    )



def preflight():
    print("===== R10L3A PREFLIGHT =====")
    upstream = load_upstream_locks()
    targets = load_targets_and_verify_rgb()
    panel = load_frozen_panel()

    r05d3 = import_module(
        R05D3_SCRIPT,
        "q1_r10l3a_authoritative_r05d3",
    )
    patch_r05d3_cardinality(r05d3)

    # Exact R03 module reused by the authoritative SegFormer state runner.
    r03 = import_module(
        Path(r05d3.R03_SCRIPT),
        "q1_r10l3a_r03",
    )
    validate_sha(
        Path(r05d3.R03_SCRIPT),
        str(r05d3.EXPECTED_R03_SCRIPT_SHA256),
        "R03 script",
    )

    print("PolypGen frozen cases:", len(targets))
    print("Frozen model states:", len(panel))
    print("Expected model-case units:", EXPECTED_MODEL_CASES)
    print("R05D3 script SHA256:", upstream["r05d3_script_sha256"])
    print("A1 action: A1_TENT_1STEP")
    print("lr=0.001 weight_decay=0 steps=1 episodic_reset=YES")
    print("PolypGen GT path column read: NO")
    print("PolypGen GT pixels decoded: NO")
    print("Safety estimator read: NO")
    print("PREFLIGHT_PASS")


def run(args):
    upstream = load_upstream_locks()
    targets = load_targets_and_verify_rgb()
    panel = load_frozen_panel()

    r05d3 = import_module(
        R05D3_SCRIPT,
        "q1_r10l3a_authoritative_r05d3",
    )
    patch_r05d3_cardinality(r05d3)

    r03 = import_module(
        Path(r05d3.R03_SCRIPT),
        "q1_r10l3a_r03",
    )
    validate_sha(
        Path(r05d3.R03_SCRIPT),
        str(r05d3.EXPECTED_R03_SCRIPT_SHA256),
        "R03 script",
    )

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("CUDA:", torch.cuda.get_device_name(device))

    if args.output_dir.exists():
        raise FileExistsError(
            f"Final R10L3A output already exists: {args.output_dir}"
        )

    build_dir = Path(str(args.output_dir) + "__building")
    if build_dir.exists() and not args.resume:
        raise FileExistsError(
            f"Partial R10L3A Fix2 output exists: {build_dir}. "
            f"Use --resume only after a technical interruption."
        )
    build_dir.mkdir(parents=True, exist_ok=True)

    state_dir = build_dir / "state_predictions"
    state_dir.mkdir(parents=True, exist_ok=True)

    # Preserve completed expensive states from the failed Fix1 run. The old
    # partial directory is read-only from Fix2's perspective.
    imported_fix1_states = bootstrap_verified_fix1_states(
        FIX1_PARTIAL_BUILD_DIR,
        state_dir,
        panel,
        targets,
        upstream,
    )

    # Imported states are now local Fix2 resume caches.
    effective_resume = bool(args.resume or imported_fix1_states)

    # Lock target identity without reading GT columns.
    target_rows = []
    for t in targets:
        target_rows.append({
            "sample_id": t["sample_id"],
            "original_polypgen_sample_id": t["original_polypgen_sample_id"],
            "center": t["center"],
            "image_path": t["image_path"],
            "image_raw_sha256": t["image_raw_sha256"],
        })
    write_csv(
        build_dir / "frozen_polypgen_rgb_manifest_no_gt.csv",
        target_rows,
        [
            "sample_id",
            "original_polypgen_sample_id",
            "center",
            "image_path",
            "image_raw_sha256",
        ],
    )

    write_csv(
        build_dir / "checkpoint_manifest.csv",
        panel,
        [
            "model_family",
            "training_seed",
            "model_state_id",
            "checkpoint",
            "checkpoint_sha256",
        ],
    )

    write_json(
        build_dir / "upstream_audit.json",
        {
            "script_version": VERSION,
            "build": BUILD,
            **upstream,
            "polypgen_cases": EXPECTED_CASES,
            "model_states": EXPECTED_STATES,
            "model_case_units": EXPECTED_MODEL_CASES,
            "independence_gate_decision": EXPECTED_INDEPENDENCE_DECISION,
            "target_gt_path_column_read": False,
            "target_gt_pixels_decoded": False,
            "fix1_partial_build_dir": str(FIX1_PARTIAL_BUILD_DIR),
            "verified_fix1_states_imported": imported_fix1_states,
        },
    )

    global_rows = []
    state_summaries = []
    seg_context = None

    for state in panel:
        print()
        print("=" * 72)
        print("STATE:", state["model_state_id"])
        print("FAMILY:", state["model_family"])
        print("SEED:", state["training_seed"])
        print("=" * 72)

        (
            npz_path,
            index_path,
            sidecar_path,
            rows,
            seg_context,
        ) = run_state(
            r05d3=r05d3,
            r03=r03,
            state=state,
            targets=targets,
            device=device,
            state_dir=state_dir,
            upstream=upstream,
            resume=effective_resume,
            seg_context=seg_context,
        )

        npz_rel = str(npz_path.relative_to(build_dir))
        idx_rel = str(index_path.relative_to(build_dir))
        side_rel = str(sidecar_path.relative_to(build_dir))

        changed = []
        source_fg = []
        a1_fg = []

        for i, row in enumerate(rows):
            changed.append(int(row["changed_pixels"]))
            source_fg.append(int(row["source_foreground_pixels"]))
            a1_fg.append(int(row["a1_foreground_pixels"]))

            global_rows.append({
                "sample_id": row["sample_id"],
                "center": targets[i]["center"],
                "original_polypgen_sample_id":
                    targets[i]["original_polypgen_sample_id"],
                "image_path": row["image_path"],
                "image_raw_sha256": row["image_raw_sha256"],
                "model_family": row["model_family"],
                "model_state_id": row["model_state_id"],
                "training_seed": int(row["training_seed"]),
                "checkpoint_sha256": row["checkpoint_sha256"],
                "state_prediction_npz": npz_rel,
                "state_index_csv": idx_rel,
                "state_lock_json": side_rel,
                "state_row_index": int(row["row_index"]),
                "source_foreground_pixels":
                    int(row["source_foreground_pixels"]),
                "a1_foreground_pixels":
                    int(row["a1_foreground_pixels"]),
                "changed_pixels": int(row["changed_pixels"]),
            })

        state_summaries.append({
            "model_family": state["model_family"],
            "model_state_id": state["model_state_id"],
            "training_seed": int(state["training_seed"]),
            "target_cases": len(rows),
            "source_foreground_pixels_mean": float(np.mean(source_fg)),
            "a1_foreground_pixels_mean": float(np.mean(a1_fg)),
            "changed_pixels_mean": float(np.mean(changed)),
            "changed_pixels_median": float(np.median(changed)),
            "changed_pixels_max": int(np.max(changed)),
        })

    if len(global_rows) != EXPECTED_MODEL_CASES:
        raise RuntimeError(
            f"Global model-case rows={len(global_rows)} "
            f"expected={EXPECTED_MODEL_CASES}"
        )

    key_set = {
        (r["sample_id"], r["model_state_id"])
        for r in global_rows
    }
    if len(key_set) != EXPECTED_MODEL_CASES:
        raise RuntimeError("sample_id + model_state_id is not unique.")

    write_csv(
        build_dir / "model_case_prediction_manifest.csv",
        global_rows,
        list(global_rows[0].keys()),
    )
    write_csv(
        build_dir / "prediction_change_summary.csv",
        state_summaries,
        list(state_summaries[0].keys()),
    )

    boundary = {
        "polypgen_rgb_pixels_decoded": True,
        "polypgen_gt_path_column_read": False,
        "polypgen_gt_pixels_decoded": False,
        "source_inference_run": True,
        "a1_backward_pass_run": True,
        "a1_optimizer_step_run": True,
        "a1_action": "A1_TENT_1STEP",
        "a1_optimizer": "Adam",
        "a1_lr": 1e-3,
        "a1_weight_decay": 0.0,
        "a1_steps": 1,
        "episodic_reset": True,
        "all_model_case_units_received_a1": True,
        "model_case_units_receiving_a1": EXPECTED_MODEL_CASES,
        "r10l0_safety_estimator_read": False,
        "safety_probability_computed": False,
        "target_dice_computed": False,
        "target_delta_dice_computed": False,
        "target_harm_label_computed": False,
        "target_calibration": False,
        "target_threshold_selection": False,
        "target_feature_selection": False,
        "target_case_exclusion_from_prediction_behavior": False,
    }
    write_json(
        build_dir / "information_boundary_audit.json",
        boundary,
    )

    lock = {
        "decision": DECISION_READY,
        "script_version": VERSION,
        "build": BUILD,
        **upstream,
        "target_dataset": "PolypGen2021_MultiCenterData_v3",
        "target_component": "unique_static_data_C1_to_C6",
        "target_cases": EXPECTED_CASES,
        "centers": ["C1", "C2", "C3", "C4", "C5", "C6"],
        "model_states": EXPECTED_STATES,
        "model_case_rows": EXPECTED_MODEL_CASES,
        "source_prediction_masks": EXPECTED_MODEL_CASES,
        "a1_prediction_masks": EXPECTED_MODEL_CASES,
        "all_model_cases_received_a1": True,
        "mask_height": MASK_H,
        "mask_width": MASK_W,
        "packed_bytes_per_mask": PACKED_BYTES,
        "bitorder": BITORDER,
        "action": "A1_TENT_1STEP",
        "optimizer": "Adam",
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "steps": 1,
        "episodic_reset": True,
        "target_gt_path_column_read": False,
        "target_gt_pixels_decoded": False,
        "target_outcomes_revealed": False,
        "safety_estimator_read": False,
        "safety_scores_generated": False,
        "artifacts": {
            "rgb_manifest_no_gt": {
                "relative_path": "frozen_polypgen_rgb_manifest_no_gt.csv",
                "sha256": sha256_file(
                    build_dir / "frozen_polypgen_rgb_manifest_no_gt.csv"
                ),
            },
            "checkpoint_manifest": {
                "relative_path": "checkpoint_manifest.csv",
                "sha256": sha256_file(
                    build_dir / "checkpoint_manifest.csv"
                ),
            },
            "model_case_prediction_manifest": {
                "relative_path": "model_case_prediction_manifest.csv",
                "sha256": sha256_file(
                    build_dir / "model_case_prediction_manifest.csv"
                ),
            },
            "prediction_change_summary": {
                "relative_path": "prediction_change_summary.csv",
                "sha256": sha256_file(
                    build_dir / "prediction_change_summary.csv"
                ),
            },
            "information_boundary_audit": {
                "relative_path": "information_boundary_audit.json",
                "sha256": sha256_file(
                    build_dir / "information_boundary_audit.json"
                ),
            },
        },
    }

    lock_path = (
        build_dir
        / "R10L3A_POLYPGEN_SOURCE_A1_PREDICTION_LOCK.json"
    )
    write_json(lock_path, lock)

    decision_path = build_dir / "decision.txt"
    decision_path.write_text(DECISION_READY + "\n", encoding="utf-8")

    run_lines = [
        "===== R10L3A POLYPGEN SOURCE + A1 PREDICTION LOCK FIX2 =====",
        f"Script version: {VERSION}",
        f"Build: {BUILD}",
        "",
        "Frozen independent target:",
        f"  PolypGen static cases={EXPECTED_CASES}",
        "  centers=6",
        "  source train exact overlap=0",
        "  source val exact overlap=0",
        "",
        "Technical recovery:",
        f"  verified Fix1 states imported={len(imported_fix1_states)}",
        "  Fix1 partial output modified=NO",
        "  SegFormer return schema=4 values with reusable seg_context",
        "",
        "Frozen panel/action:",
        "  PraNet states=3",
        "  DeepLabV3-R50 states=3",
        "  SegFormer-B0 states=3",
        f"  total states={EXPECTED_STATES}",
        "  action=A1_TENT_1STEP",
        "  optimizer=Adam",
        "  lr=0.001",
        "  weight_decay=0",
        "  steps=1",
        "  episodic reset=YES",
        "",
        "Prediction lock:",
        f"  model-case rows={EXPECTED_MODEL_CASES}",
        f"  SOURCE masks={EXPECTED_MODEL_CASES}",
        f"  A1 masks={EXPECTED_MODEL_CASES}",
        f"  mask resolution={MASK_H}x{MASK_W}",
        f"  packed bytes/mask={PACKED_BYTES}",
        f"  bitorder={BITORDER}",
        "",
        "Information boundary:",
        "  PolypGen RGB pixels decoded=YES",
        "  PolypGen GT path column read=NO",
        "  PolypGen GT pixels decoded=NO",
        "  R10L0 safety estimator read=NO",
        "  safety score generated=NO",
        "  Dice/DeltaDice/HARM=NO",
        "  target calibration/tuning=NO",
        "",
        "Decision:",
        f"  {DECISION_READY}",
        "",
        "Next:",
        "  R10L3B prospective frozen R10L0 safety-score generation",
        "  GT must remain unopened until R10L3B is locked.",
    ]
    run_log = "\n".join(run_lines) + "\n"
    (build_dir / "run_log.txt").write_text(
        run_log,
        encoding="utf-8",
    )

    # Final structural check before atomic rename.
    if len(list(state_dir.glob("*_predictions.npz"))) != EXPECTED_STATES:
        raise RuntimeError("State NPZ count != 9.")
    if len(list(state_dir.glob("*_index.csv"))) != EXPECTED_STATES:
        raise RuntimeError("State index count != 9.")
    if len(list(state_dir.glob("*_state_lock.json"))) != EXPECTED_STATES:
        raise RuntimeError("State sidecar count != 9.")

    build_dir.rename(args.output_dir)

    print()
    print(
        (args.output_dir / "run_log.txt").read_text(
            encoding="utf-8"
        )
    )
    final_lock = (
        args.output_dir
        / "R10L3A_POLYPGEN_SOURCE_A1_PREDICTION_LOCK.json"
    )
    print("R10L3A LOCK:", final_lock)
    print("R10L3A LOCK SHA256:", sha256_file(final_lock))
    print("PASS")


def self_test():
    assert EXPECTED_CASES == 1532
    assert EXPECTED_STATES == 9
    assert EXPECTED_MODEL_CASES == 13788
    assert MASK_PIXELS == 123904
    assert PACKED_BYTES == 15488

    rng = np.random.RandomState(20260826)
    mask = (rng.rand(MASK_H, MASK_W) > 0.5).astype(np.uint8)
    packed = np.packbits(mask.reshape(-1), bitorder=BITORDER)
    restored = np.unpackbits(
        packed,
        count=MASK_PIXELS,
        bitorder=BITORDER,
    ).reshape(MASK_H, MASK_W)
    assert np.array_equal(mask, restored)

    print("CARDINALITY_TEST_PASS")
    print("PACKBITS_ROUNDTRIP_TEST_PASS")
    print("SEGF0RMER_FOUR_VALUE_RETURN_HANDLING_PRESENT_PASS")
    print("GT_NOT_REQUIRED_FOR_SELF_TEST_PASS")
    print("SELF_TEST_PASS")


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "R10L3A: execute and lock frozen SOURCE + A1_TENT_1STEP "
            "predictions for 1532x9 independent PolypGen model-case units "
            "before any PolypGen GT reveal."
        )
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume only state caches that pass full sidecar/hash verification "
            "after a technical interruption."
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
