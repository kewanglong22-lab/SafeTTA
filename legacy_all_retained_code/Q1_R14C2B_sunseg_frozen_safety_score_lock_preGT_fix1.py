#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R14C2B_sunseg_frozen_safety_score_lock_preGT_fix1.py

Prospective SUN-SEG frozen safety-score lock AFTER canonical R14C1 predictions
and BEFORE any SUN GT reveal.

Frozen method:
    M2_plus_CondDINO_PCA64

Exact deployment lineage is inherited from the already completed historical
PolypGen R10L3B external scoring implementation.

Per SUN model-frame row:
1) RGB -> frozen facebook/dinov2-base, direct bicubic 224x224;
2) read PRE-ADAPTATION SOURCE mask only;
3) SOURCE 352x352 -> exact 22x22 block occupancy -> 16x16;
4) foreground-weighted patch mean 768-D;
5) background-weighted patch mean 768-D;
6) float32 pooling -> float16 feature lock -> float32 PCA input;
7) concatenate -> 1536-D;
8) frozen R10L0 PCA64.transform() ONLY;
9) prepend exact frozen M2:
       morph_fg_fraction
       morph_boundary_density
10) frozen R10L0 safety_head.predict_proba() ONLY;
11) frozen threshold = 0.300584763193734.

Strictly forbidden:
- SUN GT path/member access
- SUN GT pixel decode
- TENT1/PL mask VALUE access
- PCA fit/refit
- safety-head fit/refit
- calibration
- threshold selection
- feature selection
- target-label/outcome access
- Dice/DeltaDice/HARM/BENEFIT computation

All output score artifacts are locked before GT reveal.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
import traceback
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm


VERSION = "2026-09-03-Q1-R14C2B-v1-fix1"
BUILD = "Q1_R14C2B_SUNSEG_FROZEN_SAFETY_SCORE_LOCK_PRE_GT_FIX1"

ROOT = Path(r"F:\MEDSEG_SAFETTA")
CODE = ROOT / "code"
OUT = ROOT / "outputs"

# ---------------------------------------------------------------------
# Canonical SUN prediction lock (NO GT).
# ---------------------------------------------------------------------
R14C1_DIR = (
    OUT
    / "Q1_R14C1_sunseg_source_tent1_pl_prediction_lock_preGT_fix3_v1"
)
R14C1_LOCK = (
    R14C1_DIR
    / "R14C1_SUNSEG_SOURCE_TENT1_PL_PREDICTION_LOCK.json"
)
R14C1_GLOBAL = (
    R14C1_DIR
    / "R14C1_SUNSEG_MODEL_FRAME_PREDICTION_MANIFEST.csv"
)

EXPECTED_R14C1_LOCK_SHA = (
    "802e585dc075df23e0eeddf2b7d613d229dd1f7c6ae99aeaa89cf7dd7b358163"
)
EXPECTED_R14C1_GLOBAL_SHA = (
    "1ea4fae25a270babb9b1b0b2d2f1c4f694f458079f573b6f070c098ab3a96e40"
)
EXPECTED_R14C1_DECISION = (
    "SUNSEG_SOURCE_TENT1_PL_PREDICTIONS_LOCKED_BEFORE_GT_REVEAL"
)
EXPECTED_R14C1_CANONICAL_DECISION = (
    "SUNSEG_SOURCE_TENT1_PL_PREDICTIONS_CANONICALIZED_BEFORE_GT_REVEAL"
)
EXPECTED_FINAL_MANIFEST_SHA = (
    "f38cc8e5af4a007df3394ec95fa621c0b819f488bdd797ddad9338ac12491b4d"
)

# ---------------------------------------------------------------------
# R14C2A frozen estimator / representation preflight.
# ---------------------------------------------------------------------
R14C2A_DIR = (
    OUT
    / "Q1_R14C2A_frozen_safety_estimator_lineage_schema_preflight_fix2_v1"
)
R14C2A_AUDIT = (
    R14C2A_DIR
    / "R14C2A_FROZEN_SAFETY_ESTIMATOR_LINEAGE_SCHEMA_AUDIT.json"
)
EXPECTED_R14C2A_DECISION = (
    "R10L0_FROZEN_SAFETY_ESTIMATOR_LINEAGE_AND_SCHEMA_LOCATED_"
    "READY_FOR_SUN_PREGT_SCORING"
)

# ---------------------------------------------------------------------
# Exact historical external-scoring implementation lineage.
# ---------------------------------------------------------------------
R10L3B_SCRIPT = (
    CODE
    / "Q1_R10L3B_polypgen_frozen_safety_score_lock_fix1.py"
)
R10K2A_SCRIPT = (
    CODE
    / "Q1_R10K2A_dinov2_image_mask_representation_lock_fix1.py"
)
R10J1_SCRIPT = (
    CODE
    / "Q1_R10J1_source_mask_morphology_feature_builder_fix1.py"
)

EXPECTED_R10L3B_SCRIPT_SHA = (
    "e19e802355dc36873af4bf78ac336ba24313eea8beb19c40c3f646eb4effc646"
)
EXPECTED_R10K2A_SCRIPT_SHA = (
    "25853e849fecaf847f7ba04dcc2c8b520aa3e248f50af5ac21d19bb0ce4c4778"
)

# Completed prior external deployment lock is used to cryptographically
# recover the authoritative R10J1 source-code SHA used by R10L3B.
POLYPGEN_R10L3B_LOCK = (
    OUT
    / "Q1_R10L3B_polypgen_frozen_safety_score_lock_fix1_v1"
    / "R10L3B_POLYPGEN_FROZEN_SAFETY_SCORE_LOCK.json"
)
EXPECTED_POLYPGEN_R10L3B_DECISION = (
    "POLYPGEN_FROZEN_SAFETY_SCORES_LOCKED_BEFORE_GT_REVEAL"
)

# ---------------------------------------------------------------------
# Frozen method / estimator.
# ---------------------------------------------------------------------
R10K2D_LOCK = (
    OUT
    / "Q1_R10K2D_confirmed_method_freeze_fix1_v1"
    / "R10K2D_CONFIRMED_METHOD_LOCK.json"
)
EXPECTED_K2D_DECISION = (
    "R10K2_PRIMARY_METHOD_FROZEN_FOR_EXTERNAL_VALIDATION"
)

R10L0_DIR = (
    OUT
    / "Q1_R10L0_final_source_safety_estimator_lock_fix3_v1"
)
R10L0_LOCK = R10L0_DIR / "R10L0_FINAL_SOURCE_ESTIMATOR_LOCK.json"
R10L0_PCA = R10L0_DIR / "R10L0_final_condDINO_PCA64.joblib"
R10L0_HEAD = R10L0_DIR / "R10L0_final_safety_head.joblib"
R10L0_THRESHOLD = R10L0_DIR / "R10L0_frozen_operating_threshold.json"

EXPECTED_R10L0_LOCK_SHA = (
    "e67ad3ac9306d44b1e1f2bc13d1633d93c780241090c53c3c702f962a5ec3d6f"
)
EXPECTED_R10L0_DECISION = (
    "FINAL_SOURCE_ESTIMATOR_LOCKED_FOR_INDEPENDENT_EXTERNAL_VALIDATION"
)

EXPECTED_PCA_SHA = (
    "2c6c289ce395105a0c79f7b7751df2a195d76a248c22b43e817d71511c4a24f1"
)
EXPECTED_HEAD_SHA = (
    "602574418a309400bf7e7fc854267a702496c1bd47e4f07f6f60415ec9bd14ce"
)
EXPECTED_THRESHOLD_SHA = (
    "4a37fb4f9a9b729a9aa6551b1417afea7348e2dfecdfcb6e0e2b5986e748e4d7"
)

FROZEN_THRESHOLD = 0.300584763193734

EXPECTED_CASES = 980
EXPECTED_PHYSICAL_CASES = 49
EXPECTED_STATES = 9
EXPECTED_MODEL_CASES = EXPECTED_CASES * EXPECTED_STATES
EXPECTED_COND_DIM = 1536
EXPECTED_PCA_DIM = 64
EXPECTED_FINAL_DIM = 66

MASK_H = 352
MASK_W = 352
PACKED_BYTES = 15488
BITORDER = "little"

OUTPUT_DIR = (
    OUT
    / "Q1_R14C2B_sunseg_frozen_safety_score_lock_preGT_fix1_v1"
)

DECISION_READY = (
    "SUNSEG_FROZEN_SAFETY_SCORES_LOCKED_BEFORE_GT_REVEAL"
)


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import module: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def write_csv(path: Path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(fields))
        w.writeheader()
        w.writerows(rows)


def verify_required_paths():
    required = {
        "R14C1 lock": R14C1_LOCK,
        "R14C1 global manifest": R14C1_GLOBAL,
        "R14C2A audit": R14C2A_AUDIT,
        "historical R10L3B script": R10L3B_SCRIPT,
        "R10K2A script": R10K2A_SCRIPT,
        "R10J1 script": R10J1_SCRIPT,
        "PolypGen historical R10L3B lock": POLYPGEN_R10L3B_LOCK,
        "R10K2D lock": R10K2D_LOCK,
        "R10L0 lock": R10L0_LOCK,
        "R10L0 PCA": R10L0_PCA,
        "R10L0 head": R10L0_HEAD,
        "R10L0 threshold": R10L0_THRESHOLD,
    }
    for label, p in required.items():
        print(label, "exists:", p.exists(), p)
        if not p.is_file():
            raise FileNotFoundError(p)


def verify_lineage():
    print("\n===== R14C2B LINEAGE GATES =====")

    # Canonical R14C1.
    if sha256_file(R14C1_LOCK) != EXPECTED_R14C1_LOCK_SHA:
        raise RuntimeError("R14C1 canonical lock SHA changed.")
    if sha256_file(R14C1_GLOBAL) != EXPECTED_R14C1_GLOBAL_SHA:
        raise RuntimeError("R14C1 canonical global manifest SHA changed.")

    r14 = load_json(R14C1_LOCK)
    if r14.get("status") != "PASS":
        raise RuntimeError("R14C1 canonical lock is not PASS.")
    if r14.get("decision") != EXPECTED_R14C1_DECISION:
        raise RuntimeError("R14C1 decision changed.")
    if (
        r14.get("canonicalization_decision")
        != EXPECTED_R14C1_CANONICAL_DECISION
    ):
        raise RuntimeError("R14C1 canonicalization decision changed.")

    cohort = r14.get("cohort", {})
    if cohort.get("final_manifest_sha256") != EXPECTED_FINAL_MANIFEST_SHA:
        raise RuntimeError("SUN final-manifest SHA changed.")
    if int(cohort.get("frames", -1)) != EXPECTED_CASES:
        raise RuntimeError("SUN frame count changed.")
    if int(cohort.get("physical_cases", -1)) != EXPECTED_PHYSICAL_CASES:
        raise RuntimeError("SUN physical-case count changed.")

    panel = r14.get("model_panel", {})
    if int(panel.get("states", -1)) != EXPECTED_STATES:
        raise RuntimeError("SUN model-state count changed.")
    if int(panel.get("model_frame_rows", -1)) != EXPECTED_MODEL_CASES:
        raise RuntimeError("SUN model-frame row count changed.")

    info = r14.get("information_boundary", {})
    for key in (
        "sun_gt_pixels_decoded",
        "dice_computed",
        "delta_dice_computed",
        "harm_benefit_accessed",
        "frozen_safety_scores_accessed",
        "target_outcome_based_selection",
        "target_tuning",
    ):
        if bool(info.get(key, True)):
            raise RuntimeError(f"R14C1 pre-GT boundary violated: {key}")

    # R14C2A frozen artifact/schema preflight.
    c2a = load_json(R14C2A_AUDIT)
    if c2a.get("status") != "PASS":
        raise RuntimeError("R14C2A audit is not PASS.")
    if c2a.get("decision") != EXPECTED_R14C2A_DECISION:
        raise RuntimeError("R14C2A decision changed.")

    freeze = c2a["r10l0"]["downstream_sha_freeze"]
    if freeze.get("PCA") != EXPECTED_PCA_SHA:
        raise RuntimeError("R14C2A PCA downstream SHA changed.")
    if freeze.get("HEAD") != EXPECTED_HEAD_SHA:
        raise RuntimeError("R14C2A HEAD downstream SHA changed.")
    if freeze.get("THRESH") != EXPECTED_THRESHOLD_SHA:
        raise RuntimeError("R14C2A threshold downstream SHA changed.")
    if abs(float(c2a["r10l0"]["frozen_threshold"]) - FROZEN_THRESHOLD) > 1e-15:
        raise RuntimeError("R14C2A frozen threshold changed.")

    strong = {
        Path(str(x["path"])).name: str(x["sha256"])
        for x in c2a.get("strong_representation_candidates", [])
    }
    if strong.get(R10K2A_SCRIPT.name) != EXPECTED_R10K2A_SCRIPT_SHA:
        raise RuntimeError("R14C2A did not freeze authoritative R10K2A code.")
    if strong.get(R10L3B_SCRIPT.name) != EXPECTED_R10L3B_SCRIPT_SHA:
        raise RuntimeError(
            "R14C2A did not freeze historical external-scoring R10L3B code."
        )

    # Exact historical code bytes.
    if sha256_file(R10K2A_SCRIPT) != EXPECTED_R10K2A_SCRIPT_SHA:
        raise RuntimeError("R10K2A script bytes changed.")
    if sha256_file(R10L3B_SCRIPT) != EXPECTED_R10L3B_SCRIPT_SHA:
        raise RuntimeError("Historical R10L3B script bytes changed.")

    # Recover and enforce exact R10J1 code lineage from the completed historical
    # PolypGen deployment lock.
    pg = load_json(POLYPGEN_R10L3B_LOCK)
    if pg.get("decision") != EXPECTED_POLYPGEN_R10L3B_DECISION:
        raise RuntimeError("Historical PolypGen R10L3B decision changed.")
    pg_up = pg.get("upstream_hashes", {})
    expected_j1_sha = str(pg_up.get("r10j1_script_sha256", ""))
    expected_k2a_sha_from_pg = str(pg_up.get("r10k2a_script_sha256", ""))

    if not expected_j1_sha:
        raise RuntimeError(
            "Historical PolypGen R10L3B lock lacks r10j1_script_sha256."
        )
    if expected_k2a_sha_from_pg != EXPECTED_R10K2A_SCRIPT_SHA:
        raise RuntimeError(
            "Historical PolypGen deployment K2A SHA disagrees with R14C2A."
        )
    if sha256_file(R10J1_SCRIPT) != expected_j1_sha:
        raise RuntimeError("R10J1 script bytes changed from PolypGen deployment.")

    # Method lock.
    k2d = load_json(R10K2D_LOCK)
    if k2d.get("decision") != EXPECTED_K2D_DECISION:
        raise RuntimeError("R10K2D method-freeze decision changed.")

    # Frozen estimator.
    if sha256_file(R10L0_LOCK) != EXPECTED_R10L0_LOCK_SHA:
        raise RuntimeError("R10L0 lock SHA changed.")
    l0 = load_json(R10L0_LOCK)
    if l0.get("decision") != EXPECTED_R10L0_DECISION:
        raise RuntimeError("R10L0 decision changed.")

    if sha256_file(R10L0_PCA) != EXPECTED_PCA_SHA:
        raise RuntimeError("Frozen PCA bytes changed.")
    if sha256_file(R10L0_HEAD) != EXPECTED_HEAD_SHA:
        raise RuntimeError("Frozen safety-head bytes changed.")
    if sha256_file(R10L0_THRESHOLD) != EXPECTED_THRESHOLD_SHA:
        raise RuntimeError("Frozen threshold artifact bytes changed.")

    lock_thr = float(l0["operating_point"]["final_threshold"])
    thr = load_json(R10L0_THRESHOLD)
    artifact_thr = float(thr["final_threshold"])
    if abs(lock_thr - FROZEN_THRESHOLD) > 1e-15:
        raise RuntimeError("R10L0 lock threshold changed.")
    if abs(artifact_thr - FROZEN_THRESHOLD) > 1e-15:
        raise RuntimeError("Threshold artifact value changed.")

    print("R14C1 canonical lock:", EXPECTED_R14C1_LOCK_SHA)
    print("R14C1 global manifest:", EXPECTED_R14C1_GLOBAL_SHA)
    print("R10K2A code:", EXPECTED_R10K2A_SCRIPT_SHA)
    print("R10J1 code:", expected_j1_sha)
    print("historical R10L3B code:", EXPECTED_R10L3B_SCRIPT_SHA)
    print("R10L0 PCA:", EXPECTED_PCA_SHA)
    print("R10L0 HEAD:", EXPECTED_HEAD_SHA)
    print("R10L0 THRESH:", EXPECTED_THRESHOLD_SHA)
    print("frozen threshold:", FROZEN_THRESHOLD)
    print("PASS")

    return {
        "r14c1_lock_sha256": EXPECTED_R14C1_LOCK_SHA,
        "r14c1_global_manifest_sha256": EXPECTED_R14C1_GLOBAL_SHA,
        "r14c2a_audit_sha256": sha256_file(R14C2A_AUDIT),
        "r10k2a_script_sha256": EXPECTED_R10K2A_SCRIPT_SHA,
        "r10j1_script_sha256": expected_j1_sha,
        "historical_r10l3b_script_sha256": EXPECTED_R10L3B_SCRIPT_SHA,
        "polypgen_r10l3b_lock_sha256": sha256_file(POLYPGEN_R10L3B_LOCK),
        "r10k2d_lock_sha256": sha256_file(R10K2D_LOCK),
        "r10l0_lock_sha256": EXPECTED_R10L0_LOCK_SHA,
        "r10l0_pca_sha256": EXPECTED_PCA_SHA,
        "r10l0_head_sha256": EXPECTED_HEAD_SHA,
        "r10l0_threshold_sha256": EXPECTED_THRESHOLD_SHA,
    }


def validate_state_artifacts(states):
    """
    Validate the canonical per-state NPZ/CSV/lock chain before reading SOURCE
    mask values.
    """
    print("\n===== CANONICAL STATE ARTIFACT GATES =====")

    for state_id, g in states.groupby("model_state_id", sort=True):
        pred_rels = g["state_prediction_npz_relpath"].astype(str).unique()
        idx_rels = g["state_index_csv_relpath"].astype(str).unique()
        lock_rels = g["state_lock_json_relpath"].astype(str).unique()

        if len(pred_rels) != 1 or len(idx_rels) != 1 or len(lock_rels) != 1:
            raise RuntimeError(f"{state_id}: non-unique state artifact paths.")

        pred = R14C1_DIR / pred_rels[0]
        idx = R14C1_DIR / idx_rels[0]
        lk_path = R14C1_DIR / lock_rels[0]

        for p in (pred, idx, lk_path):
            if not p.is_file():
                raise FileNotFoundError(p)

        lk = load_json(lk_path)
        if lk.get("status") != "PASS":
            raise RuntimeError(f"{state_id}: state lock is not PASS.")
        if lk.get("model_state_id") != state_id:
            raise RuntimeError(f"{state_id}: state-lock identity mismatch.")
        if int(lk.get("target_cases", -1)) != EXPECTED_CASES:
            raise RuntimeError(f"{state_id}: state target-case count changed.")
        if sha256_file(pred) != lk.get("prediction_npz_sha256"):
            raise RuntimeError(f"{state_id}: prediction NPZ SHA mismatch.")
        if sha256_file(idx) != lk.get("index_csv_sha256"):
            raise RuntimeError(f"{state_id}: state index CSV SHA mismatch.")

        with np.load(pred, allow_pickle=False) as z:
            if "source_masks_packed" not in z.files:
                raise RuntimeError(f"{state_id}: SOURCE key missing.")
            if "tent1_masks_packed" not in z.files:
                raise RuntimeError(f"{state_id}: TENT1 schema key missing.")
            if "pl_masks_packed" not in z.files:
                raise RuntimeError(f"{state_id}: PL schema key missing.")
            if z["source_masks_packed"].shape != (
                EXPECTED_CASES, PACKED_BYTES
            ):
                raise RuntimeError(
                    f"{state_id}: SOURCE array shape changed."
                )

        print(
            state_id,
            "| prediction SHA",
            lk["prediction_npz_sha256"][:12] + "...",
            "| PASS",
        )

    print("All 9 canonical state artifacts: PASS")


def load_sun_no_gt_manifests():
    """
    Build an RGB manifest and deterministic model-frame state table from the
    canonical R14C1 no-GT manifest.

    RGB files are resolved by exact encoded SHA from the technical R14C1 RGB
    cache recorded in the canonical lock, never by GT or outcome metadata.
    """
    df = pd.read_csv(R14C1_GLOBAL, low_memory=False)

    required = [
        "row_index",
        "sample_id",
        "external_case_id",
        "cluster_id",
        "clip_id",
        "source_split_label",
        "frame_member",
        "image_raw_sha256",
        "model_family",
        "model_state_id",
        "training_seed",
        "checkpoint_sha256",
        "source_foreground_pixels",
        "state_prediction_npz_relpath",
        "state_index_csv_relpath",
        "state_lock_json_relpath",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"R14C1 global manifest missing: {missing}")

    if len(df) != EXPECTED_MODEL_CASES:
        raise RuntimeError(
            f"R14C1 global rows={len(df)} expected={EXPECTED_MODEL_CASES}"
        )
    if df["sample_id"].astype(str).nunique() != EXPECTED_CASES:
        raise RuntimeError("SUN unique frame count != 980.")
    if df["cluster_id"].astype(str).nunique() != EXPECTED_PHYSICAL_CASES:
        raise RuntimeError("SUN physical-case count != 49.")
    if df["model_state_id"].astype(str).nunique() != EXPECTED_STATES:
        raise RuntimeError("SUN model-state count != 9.")
    if (
        df[["sample_id", "model_state_id"]]
        .drop_duplicates()
        .shape[0]
        != EXPECTED_MODEL_CASES
    ):
        raise RuntimeError("sample_id + model_state_id is not unique.")

    # Every state row index must be a valid source-mask NPZ row.
    if not df["row_index"].astype(int).between(0, EXPECTED_CASES - 1).all():
        raise RuntimeError("R14C1 row_index outside [0,979].")

    # State artifact validation precedes SOURCE mask value access.
    validate_state_artifacts(df)

    # Recover technical RGB cache only from canonical R14C1 lock metadata.
    r14 = load_json(R14C1_LOCK)
    rgb_cache_raw = (
        r14.get("artifacts", {})
        .get("rgb_cache", {})
        .get("technical_source_cache_only")
    )
    if not rgb_cache_raw:
        raise RuntimeError("Canonical R14C1 lock lacks technical RGB cache path.")
    rgb_cache = Path(str(rgb_cache_raw))
    if not rgb_cache.is_dir():
        raise FileNotFoundError(rgb_cache)

    # Build SHA -> local RGB cache path without relying on filename convention.
    cache_files = sorted(
        p for p in rgb_cache.iterdir()
        if p.is_file()
    )
    if len(cache_files) != EXPECTED_CASES:
        raise RuntimeError(
            f"RGB cache files={len(cache_files)} expected={EXPECTED_CASES}"
        )

    print("\n===== VERIFY R14C1 RGB CACHE BY ENCODED SHA =====")
    sha_to_paths = {}
    for p in tqdm(
        cache_files,
        desc="Hash SUN RGB cache",
        unit="image",
        dynamic_ncols=True,
    ):
        digest = sha256_file(p)
        sha_to_paths.setdefault(digest, []).append(p)

    rgb_base_cols = [
        "sample_id",
        "external_case_id",
        "cluster_id",
        "clip_id",
        "source_split_label",
        "frame_member",
        "image_raw_sha256",
    ]
    rgb = df[rgb_base_cols].drop_duplicates().copy()

    if len(rgb) != EXPECTED_CASES:
        raise RuntimeError(
            f"RGB unique rows={len(rgb)} expected={EXPECTED_CASES}"
        )

    image_paths = []
    for r in rgb.itertuples(index=False):
        digest = str(r.image_raw_sha256)
        paths = sha_to_paths.get(digest, [])
        if len(paths) != 1:
            raise RuntimeError(
                f"RGB cache SHA mapping for {r.sample_id}: "
                f"matches={len(paths)}"
            )
        image_paths.append(str(paths[0]))

    rgb["image_path"] = image_paths

    # No filename or cache path is accepted without exact SHA re-check.
    for r in tqdm(
        rgb.itertuples(index=False),
        total=len(rgb),
        desc="Verify mapped SUN RGB",
        unit="image",
        dynamic_ncols=True,
    ):
        if sha256_file(Path(r.image_path)) != str(r.image_raw_sha256):
            raise RuntimeError(f"Mapped RGB SHA mismatch: {r.sample_id}")

    # Exact deterministic historical deployment row order.
    states = df.copy()
    states["state_row_index"] = states["row_index"].astype(int)
    states["state_prediction_npz"] = states[
        "state_prediction_npz_relpath"
    ].map(lambda x: str(R14C1_DIR / str(x)))

    states = states.sort_values(
        ["model_family", "training_seed", "sample_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    states["_feature_row"] = np.arange(len(states), dtype=int)

    rgb = rgb.sort_values(
        "sample_id",
        kind="mergesort",
    ).reset_index(drop=True)

    if set(rgb["sample_id"].astype(str)) != set(
        states["sample_id"].astype(str)
    ):
        raise RuntimeError("RGB and model-frame case sets differ.")
    counts = states.groupby("sample_id").size()
    if not (counts == EXPECTED_STATES).all():
        raise RuntimeError("Every SUN frame must have exactly 9 states.")

    print("SUN RGB rows:", len(rgb))
    print("SUN model-frame rows:", len(states))
    print("SUN physical cases:", rgb["cluster_id"].astype(str).nunique())
    print("GT columns read: NO")
    print("PASS")

    return rgb, states, rgb_cache


def build_source_mask_features(states, k2a, j1):
    """
    Exact R10L3B SOURCE-only feature construction adapted only to SUN
    cardinality/schema.

    VALUE ACCESS:
      source_masks_packed: YES
      tent1_masks_packed: NO
      pl_masks_packed: NO
    """
    occupancy = np.zeros(
        (len(states), k2a.PATCH_GRID * k2a.PATCH_GRID),
        dtype=np.float32,
    )
    m2 = np.zeros((len(states), 2), dtype=np.float64)

    source_mask_reads = 0
    foreground_pixel_agreement = 0

    for state_id, g in states.groupby("model_state_id", sort=True):
        npz_values = g["state_prediction_npz"].astype(str).unique()
        if len(npz_values) != 1:
            raise RuntimeError(
                f"{state_id}: expected one prediction NPZ."
            )
        npz_path = Path(npz_values[0])
        if not npz_path.is_file():
            raise FileNotFoundError(npz_path)

        with np.load(npz_path, allow_pickle=False) as z:
            if "source_masks_packed" not in z.files:
                raise RuntimeError(f"{state_id}: SOURCE key missing.")
            # Adapted-array keys are schema-audited only. Their values are not
            # indexed/read by this feature builder.
            if "tent1_masks_packed" not in z.files:
                raise RuntimeError(f"{state_id}: TENT1 schema key missing.")
            if "pl_masks_packed" not in z.files:
                raise RuntimeError(f"{state_id}: PL schema key missing.")

            source = z["source_masks_packed"]
            if source.shape != (EXPECTED_CASES, PACKED_BYTES):
                raise RuntimeError(
                    f"{state_id}: SOURCE shape={source.shape}"
                )

            print(
                state_id,
                "SOURCE=",
                source.shape,
                "TENT1/PL value access=NO",
            )

            for r in tqdm(
                g.itertuples(index=True),
                total=len(g),
                desc=f"SOURCE features {state_id}",
                unit="mask",
                dynamic_ncols=True,
            ):
                out_i = int(r.Index)
                row_i = int(r.state_row_index)

                packed = source[row_i]
                source_mask_reads += 1

                occ = k2a.unpack_mask_occupancy16(packed)
                occupancy[out_i] = occ

                mask = j1.unpack_source_mask(packed)
                observed_fg = int(mask.sum())
                expected_fg = int(r.source_foreground_pixels)
                if observed_fg != expected_fg:
                    raise RuntimeError(
                        f"SOURCE foreground-pixel mismatch: "
                        f"{r.sample_id} {state_id} row={row_i} "
                        f"decoded={observed_fg} manifest={expected_fg}"
                    )
                foreground_pixel_agreement += 1

                m2[out_i, 0] = float(mask.mean())
                m2[out_i, 1] = float(j1.boundary_density(mask))

    if source_mask_reads != EXPECTED_MODEL_CASES:
        raise RuntimeError(
            f"SOURCE mask reads={source_mask_reads} "
            f"expected={EXPECTED_MODEL_CASES}"
        )
    if foreground_pixel_agreement != EXPECTED_MODEL_CASES:
        raise RuntimeError("SOURCE foreground-pixel audit incomplete.")
    if occupancy.shape != (EXPECTED_MODEL_CASES, 256):
        raise RuntimeError("Occupancy shape invalid.")
    if m2.shape != (EXPECTED_MODEL_CASES, 2):
        raise RuntimeError("M2 shape invalid.")
    if not np.isfinite(occupancy).all():
        raise RuntimeError("Non-finite occupancy.")
    if not np.isfinite(m2).all():
        raise RuntimeError("Non-finite M2.")

    return occupancy, m2, foreground_pixel_agreement


def validate_historical_modules(k2a, j1, l3b):
    if int(k2a.MASK_SIZE) != MASK_H:
        raise RuntimeError("K2A mask size changed.")
    if int(k2a.PACKED_BYTES) != PACKED_BYTES:
        raise RuntimeError("K2A packed bytes changed.")
    if str(k2a.BITORDER) != BITORDER:
        raise RuntimeError("K2A bitorder changed.")
    if int(k2a.PATCH_GRID) != 16:
        raise RuntimeError("K2A patch grid changed.")
    if int(k2a.MASK_BLOCK) != 22:
        raise RuntimeError("K2A mask block changed.")
    if int(k2a.IMAGE_SIZE) != 224:
        raise RuntimeError("K2A image size changed.")
    if str(k2a.MODEL_NAME) != "facebook/dinov2-base":
        raise RuntimeError("K2A DINO model changed.")

    if int(j1.MASK_H) != MASK_H or int(j1.MASK_W) != MASK_W:
        raise RuntimeError("J1 mask shape changed.")
    if int(j1.PACKED_BYTES) != PACKED_BYTES:
        raise RuntimeError("J1 packed bytes changed.")
    if str(j1.BITORDER) != BITORDER:
        raise RuntimeError("J1 bitorder changed.")

    # Patch only target cardinality globals in the historical R10L3B generic
    # deployment helpers. Representation and estimator semantics are untouched.
    l3b.EXPECTED_CASES = EXPECTED_CASES
    l3b.EXPECTED_STATES = EXPECTED_STATES
    l3b.EXPECTED_MODEL_CASES = EXPECTED_MODEL_CASES
    l3b.EXPECTED_COND_DIM = EXPECTED_COND_DIM
    l3b.EXPECTED_PCA_DIM = EXPECTED_PCA_DIM
    l3b.EXPECTED_FINAL_DIM = EXPECTED_FINAL_DIM
    l3b.FROZEN_THRESHOLD = FROZEN_THRESHOLD

    print("\n===== HISTORICAL IMPLEMENTATION LOCK =====")
    print("encoder:", k2a.MODEL_NAME)
    print("image resize:", k2a.IMAGE_SIZE, "bicubic")
    print("patch grid:", k2a.PATCH_GRID, "x", k2a.PATCH_GRID)
    print("mask block:", k2a.MASK_BLOCK)
    print("bitorder:", k2a.BITORDER)
    print("CondDINO dim:", EXPECTED_COND_DIM)
    print("PCA dim:", EXPECTED_PCA_DIM)
    print("final head dim:", EXPECTED_FINAL_DIM)
    print("cardinality patch only:", EXPECTED_CASES, EXPECTED_MODEL_CASES)
    print("representation/action semantics changed: NO")
    print("PASS")


def validate_frozen_estimators(l3b):
    # Historical R10L3B deployment validator.
    pca, head = l3b.validate_frozen_estimators()

    if sha256_file(R10L0_PCA) != EXPECTED_PCA_SHA:
        raise RuntimeError("PCA changed immediately before scoring.")
    if sha256_file(R10L0_HEAD) != EXPECTED_HEAD_SHA:
        raise RuntimeError("Head changed immediately before scoring.")

    return pca, head


def score_sun(l3b, cond, m2, pca, head):
    # Exact historical deployment path: transform only and M2-first column order.
    z64, X66, prob, flagged = l3b.score_external(
        cond,
        m2,
        pca,
        head,
    )

    if z64.shape != (EXPECTED_MODEL_CASES, EXPECTED_PCA_DIM):
        raise RuntimeError("SUN PCA output shape invalid.")
    if X66.shape != (EXPECTED_MODEL_CASES, EXPECTED_FINAL_DIM):
        raise RuntimeError("SUN safety-head input shape invalid.")
    if prob.shape != (EXPECTED_MODEL_CASES,):
        raise RuntimeError("SUN probability shape invalid.")

    return z64, X66, prob, flagged


def run(args):
    print("===== Q1 R14C2B SUN-SEG FROZEN SAFETY SCORE LOCK =====")
    print("STATUS=PROSPECTIVE_PRE_GT_SAFETY_SCORE_GENERATION")
    print("PRIMARY_METHOD=M2_plus_CondDINO_PCA64")
    print("TARGET_FRAMES=", EXPECTED_CASES)
    print("PHYSICAL_CASES=", EXPECTED_PHYSICAL_CASES)
    print("MODEL_STATES=", EXPECTED_STATES)
    print("MODEL_FRAME_ROWS=", EXPECTED_MODEL_CASES)
    print("SOURCE_MASK_VALUES_ACCESSED=YES")
    print("TENT1_MASK_VALUES_ACCESSED=NO")
    print("PL_MASK_VALUES_ACCESSED=NO")
    print("SUN_GT_PATH_OR_MEMBER_ACCESS=NO")
    print("SUN_GT_PIXELS_DECODED=NO")
    print("PCA_FIT_REFIT=NO")
    print("SAFETY_HEAD_FIT_REFIT=NO")
    print("TARGET_CALIBRATION=NO")
    print("TARGET_THRESHOLD_TUNING=NO")
    print("TARGET_FEATURE_SELECTION=NO")
    print("FROZEN_THRESHOLD=", FROZEN_THRESHOLD)

    verify_required_paths()
    upstream_hashes = verify_lineage()

    # Import exact historical source-only representation/deployment code.
    k2a = import_module(
        R10K2A_SCRIPT,
        "q1_r14c2b_authoritative_k2a",
    )
    j1 = import_module(
        R10J1_SCRIPT,
        "q1_r14c2b_authoritative_j1",
    )
    l3b = import_module(
        R10L3B_SCRIPT,
        "q1_r14c2b_historical_external_deployment",
    )

    validate_historical_modules(k2a, j1, l3b)

    rgb, states, rgb_cache = load_sun_no_gt_manifests()

    print("\n===== SOURCE MASK REPRESENTATION =====")
    occupancy, m2, fg_agree = build_source_mask_features(
        states,
        k2a,
        j1,
    )

    print("Occupancy:", occupancy.shape)
    print("M2:", m2.shape)
    print("SOURCE foreground-pixel agreement:", fg_agree)
    print(
        "Empty SOURCE masks:",
        int((occupancy.sum(axis=1) == 0).sum()),
    )
    print(
        "M2 fg fraction range:",
        float(m2[:, 0].min()),
        float(m2[:, 0].max()),
    )
    print(
        "M2 boundary density range:",
        float(m2[:, 1].min()),
        float(m2[:, 1].max()),
    )

    # Frozen DINOv2 deployment and development sentinel.
    torch, processor, model, device = l3b.load_frozen_dinov2(
        k2a,
        local_files_only=True,
        device_name=args.device,
    )

    sentinel_diff = l3b.representation_sentinel(
        processor,
        model,
        device,
        k2a,
        torch,
    )

    print("\n===== SUN CONDITIONED DINOV2 =====")
    fg16, bg16, cond = l3b.build_conditioned_dino(
        rgb=rgb,
        states=states,
        occupancy=occupancy,
        processor=processor,
        model=model,
        device=device,
        k2a=k2a,
        torch=torch,
        batch_size=args.batch_size,
    )

    print("FG locked:", fg16.shape, fg16.dtype)
    print("BG locked:", bg16.shape, bg16.dtype)
    print("CondDINO:", cond.shape, cond.dtype)

    # Explicit precision-path sentinel.
    if fg16.dtype != np.float16 or bg16.dtype != np.float16:
        raise RuntimeError("Frozen float16 feature-lock precision changed.")
    if cond.dtype != np.float32:
        raise RuntimeError("PCA deployment CondDINO dtype changed.")

    print("\n===== FROZEN R10L0 ESTIMATOR =====")
    pca, head = validate_frozen_estimators(l3b)
    z64, X66, prob, flagged = score_sun(
        l3b,
        cond,
        m2,
        pca,
        head,
    )

    print("PCA transform output:", z64.shape)
    print("Safety input:", X66.shape)
    print("Probability finite:", bool(np.isfinite(prob).all()))
    print(
        "Frozen-threshold flagged:",
        int(flagged.sum()),
        "/",
        len(flagged),
    )

    print("\n===== UNLABELED SCORE DISTRIBUTION AUDIT =====")
    print("min:", float(np.min(prob)))
    print("q25:", float(np.quantile(prob, 0.25)))
    print("median:", float(np.median(prob)))
    print("mean:", float(np.mean(prob)))
    print("q75:", float(np.quantile(prob, 0.75)))
    print("max:", float(np.max(prob)))
    print(
        "NOTE: unlabeled score distribution cannot change "
        "method/threshold/selection."
    )

    if args.preflight_only:
        print(
            "\nPREFLIGHT ONLY: stops before writing SUN safety-score "
            "artifacts."
        )
        print("PREFLIGHT_PASS")
        return

    if args.output_dir.exists():
        raise FileExistsError(
            f"Final R14C2B output already exists: {args.output_dir}"
        )

    build = Path(str(args.output_dir) + "__building")
    if build.exists():
        raise FileExistsError(
            f"Partial R14C2B build exists: {build}"
        )
    build.mkdir(parents=True, exist_ok=False)

    try:
        # Frozen representation artifacts.
        np.savez_compressed(
            build / "R14C2B_SUN_conditioned_dino_features.npz",
            foreground_patch_mean=fg16,
            background_patch_mean=bg16,
            source_mask_patch_occupancy=occupancy.astype(np.float16),
            m2=m2.astype(np.float64),
        )

        np.savez_compressed(
            build / "R14C2B_SUN_pca64_features.npz",
            pca64=z64.astype(np.float64),
        )

        score_rows = []
        for i, r in states.iterrows():
            score_rows.append({
                "feature_row": int(i),
                "sample_id": str(r["sample_id"]),
                "external_case_id": str(r["external_case_id"]),
                "cluster_id": str(r["cluster_id"]),
                "clip_id": str(r["clip_id"]),
                "source_split_label": str(r["source_split_label"]),
                "frame_member": str(r["frame_member"]),
                "image_raw_sha256": str(r["image_raw_sha256"]),
                "model_family": str(r["model_family"]),
                "model_state_id": str(r["model_state_id"]),
                "training_seed": int(r["training_seed"]),
                "checkpoint_sha256": str(r["checkpoint_sha256"]),
                "state_prediction_npz_relpath": str(
                    r["state_prediction_npz_relpath"]
                ),
                "state_row_index": int(r["state_row_index"]),
                "morph_fg_fraction": float(m2[i, 0]),
                "morph_boundary_density": float(m2[i, 1]),
                "frozen_safety_probability": float(prob[i]),
                "frozen_operating_threshold": float(FROZEN_THRESHOLD),
                "frozen_risk_flag": int(flagged[i]),
            })

        score_csv = (
            build
            / "R14C2B_FROZEN_SUNSEG_SAFETY_SCORES.csv"
        )
        write_csv(
            score_csv,
            score_rows,
            list(score_rows[0].keys()),
        )

        score_df = pd.DataFrame(score_rows)

        family_summary = (
            score_df.groupby("model_family", sort=True)
            .agg(
                rows=("frozen_safety_probability", "size"),
                probability_mean=("frozen_safety_probability", "mean"),
                probability_median=("frozen_safety_probability", "median"),
                flagged=("frozen_risk_flag", "sum"),
            )
            .reset_index()
        )
        family_summary["flagged_fraction"] = (
            family_summary["flagged"]
            / family_summary["rows"]
        )
        family_summary.to_csv(
            build / "R14C2B_unlabeled_family_score_summary.csv",
            index=False,
        )

        split_summary = (
            score_df.groupby("source_split_label", sort=True)
            .agg(
                rows=("frozen_safety_probability", "size"),
                unique_frames=("sample_id", "nunique"),
                unique_physical_cases=("cluster_id", "nunique"),
                probability_mean=("frozen_safety_probability", "mean"),
                probability_median=("frozen_safety_probability", "median"),
                flagged=("frozen_risk_flag", "sum"),
            )
            .reset_index()
        )
        split_summary["flagged_fraction"] = (
            split_summary["flagged"]
            / split_summary["rows"]
        )
        split_summary.to_csv(
            build / "R14C2B_unlabeled_split_score_summary.csv",
            index=False,
        )

        info_boundary = {
            "sun_rgb_pixels_decoded": True,
            "source_mask_values_accessed": True,
            "tent1_mask_values_accessed": False,
            "pl_mask_values_accessed": False,
            "sun_gt_path_or_member_accessed": False,
            "sun_gt_pixels_decoded": False,
            "sun_outcomes_revealed": False,
            "dice_computed": False,
            "delta_dice_computed": False,
            "harm_benefit_labels_computed": False,
            "pca_fit_called": False,
            "pca_refit_called": False,
            "safety_head_fit_called": False,
            "safety_head_refit_called": False,
            "target_calibration": False,
            "target_threshold_selection": False,
            "target_feature_selection": False,
            "target_score_orientation_change": False,
            "target_case_selection_from_scores": False,
            "frozen_threshold": FROZEN_THRESHOLD,
        }
        write_json(
            build / "information_boundary_audit.json",
            info_boundary,
        )

        artifact_files = [
            build / "R14C2B_SUN_conditioned_dino_features.npz",
            build / "R14C2B_SUN_pca64_features.npz",
            score_csv,
            build / "R14C2B_unlabeled_family_score_summary.csv",
            build / "R14C2B_unlabeled_split_score_summary.csv",
            build / "information_boundary_audit.json",
        ]

        artifacts = {
            p.name: {
                "sha256": sha256_file(p),
                "bytes": p.stat().st_size,
            }
            for p in artifact_files
        }

        lock = {
            "status": "FROZEN",
            "decision": DECISION_READY,
            "script_version": VERSION,
            "build": BUILD,
            "primary_method": "M2_plus_CondDINO_PCA64",
            "target_dataset": "SUN-SEG",
            "target_component": (
                "R14B4B_Fix3_final_confirmatory_cohort_"
                "EasyUnseen_plus_HardUnseen"
            ),
            "target_frames": EXPECTED_CASES,
            "physical_cases": EXPECTED_PHYSICAL_CASES,
            "model_states": EXPECTED_STATES,
            "model_frame_rows": EXPECTED_MODEL_CASES,
            "representation": {
                "historical_deployment_reference":
                    str(R10L3B_SCRIPT),
                "historical_deployment_reference_sha256":
                    EXPECTED_R10L3B_SCRIPT_SHA,
                "encoder": "facebook/dinov2-base",
                "encoder_trainable": False,
                "image_resize": "224x224 direct bicubic",
                "processor_resize": False,
                "processor_center_crop": False,
                "patch_grid": "16x16",
                "source_mask":
                    "352x352 binary pre-adaptation SOURCE only",
                "source_mask_alignment":
                    "exact 22x22 block occupancy -> 16x16",
                "foreground_patch_mean_dim": 768,
                "background_patch_mean_dim": 768,
                "feature_lock_precision":
                    "float32 pooling -> float16 lock -> "
                    "float32 PCA input",
                "conditioned_dim": EXPECTED_COND_DIM,
                "m2": [
                    "morph_fg_fraction",
                    "morph_boundary_density",
                ],
                "pca_dim": EXPECTED_PCA_DIM,
                "final_input_dim": EXPECTED_FINAL_DIM,
                "final_input_order":
                    "M2 first, then PCA64",
            },
            "estimator": {
                "pca_sha256": EXPECTED_PCA_SHA,
                "head_sha256": EXPECTED_HEAD_SHA,
                "threshold_artifact_sha256": EXPECTED_THRESHOLD_SHA,
                "pca_fit_on_sun": False,
                "safety_head_fit_on_sun": False,
                "frozen_threshold": FROZEN_THRESHOLD,
                "probability_rows": EXPECTED_MODEL_CASES,
                "flagged_rows": int(flagged.sum()),
            },
            "representation_sentinel": {
                "development_cases_checked": int(l3b.SENTINEL_CASES),
                "float16_locked_cls_max_abs_difference": sentinel_diff,
                "tolerance": float(
                    l3b.SENTINEL_FLOAT16_MAX_ABS_TOL
                ),
                "pass": True,
            },
            "source_mask_decode_audit": {
                "reads": EXPECTED_MODEL_CASES,
                "foreground_pixel_agreement": fg_agree,
                "tent1_values_accessed": False,
                "pl_values_accessed": False,
            },
            "rgb_cache": {
                "technical_source_cache_path": str(rgb_cache),
                "files": EXPECTED_CASES,
                "mapping_rule":
                    "exact encoded SHA256 only",
            },
            "information_boundary": info_boundary,
            "upstream_hashes": upstream_hashes,
            "artifacts": artifacts,
            "post_score_rules": [
                "Do not refit PCA on SUN-SEG.",
                "Do not refit or recalibrate the safety head on SUN-SEG.",
                "Do not change score orientation.",
                "Do not change the frozen threshold.",
                "Do not select split/case/state based on score performance.",
                "Do not alter the representation after GT reveal.",
                "Do not access TENT1/PL mask values for safety scoring.",
                "Next stage may reveal GT only after this lock is finalized.",
            ],
            "next_stage": (
                "R14C3_SUNSEG_GT_REVEAL_OUTCOME_CONSTRUCTION_AND_"
                "FROZEN_SCORE_EVALUATION"
            ),
        }

        lock_path = (
            build
            / "R14C2B_SUNSEG_FROZEN_SAFETY_SCORE_LOCK.json"
        )
        write_json(lock_path, lock)

        run_log = "\n".join([
            "===== R14C2B SUN-SEG FROZEN SAFETY SCORE LOCK =====",
            f"Script version: {VERSION}",
            f"Build: {BUILD}",
            "",
            "Frozen target:",
            f"  frames={EXPECTED_CASES}",
            f"  physical_cases={EXPECTED_PHYSICAL_CASES}",
            f"  states={EXPECTED_STATES}",
            f"  model-frame rows={EXPECTED_MODEL_CASES}",
            "",
            "Frozen representation:",
            "  encoder=facebook/dinov2-base",
            "  trainable=NO",
            "  RGB=direct bicubic 224x224",
            "  patch grid=16x16",
            "  SOURCE mask=352x352 -> 22x22 occupancy -> 16x16",
            "  TENT1/PL mask values accessed=NO",
            "  FG mean=768",
            "  BG mean=768",
            "  CondDINO=1536",
            "  float32->float16->float32 feature lock=YES",
            "  M2=fg_fraction + boundary_density",
            "",
            "Frozen estimator:",
            "  PCA64 transform only=YES",
            "  PCA fit/refit=NO",
            "  safety-head predict_proba only=YES",
            "  safety-head fit/refit=NO",
            f"  frozen threshold={FROZEN_THRESHOLD:.15f}",
            "",
            "Pre-GT boundary:",
            "  SUN GT path/member access=NO",
            "  SUN GT pixels decoded=NO",
            "  Dice/DeltaDice/HARM/BENEFIT=NO",
            "  target calibration=NO",
            "  target threshold selection=NO",
            "  target feature selection=NO",
            "",
            "Safety-score lock:",
            f"  probability rows={EXPECTED_MODEL_CASES}",
            f"  frozen-threshold flagged rows={int(flagged.sum())}",
            f"  SOURCE foreground-pixel agreement={fg_agree}",
            f"  DINO sentinel max abs={sentinel_diff:.9g}",
            "",
            "Decision:",
            f"  {DECISION_READY}",
            "",
            "Next:",
            "  R14C3 SUN GT reveal + TENT1/PL outcome construction",
            "  evaluate already-locked safety scores with no recalibration.",
        ]) + "\n"

        (build / "run_log.txt").write_text(
            run_log,
            encoding="utf-8",
        )

        lock_sha = sha256_file(lock_path)

        build.rename(args.output_dir)

        print("\n===== R14C2B FINAL =====")
        print(
            (args.output_dir / "run_log.txt").read_text(
                encoding="utf-8"
            )
        )
        print(
            "LOCK=",
            args.output_dir
            / "R14C2B_SUNSEG_FROZEN_SAFETY_SCORE_LOCK.json",
        )
        print("LOCK SHA256=", lock_sha)
        print("PASS")

    except Exception:
        # Leave no directory that could be mistaken for a finalized score lock.
        if build.exists():
            import shutil
            shutil.rmtree(build)
        raise


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
    )
    p.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )
    p.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "Execute all representation/estimator computation and sentinel "
            "checks, but stop before writing final SUN safety-score artifacts."
        ),
    )
    return p.parse_args()


def main():
    args = parse_args()

    if args.batch_size < 1:
        raise RuntimeError("--batch-size must be >=1.")

    try:
        run(args)
    except Exception:
        traceback.print_exc()
        raise

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
