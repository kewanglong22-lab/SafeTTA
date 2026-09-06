#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R14C3B_sunseg_gt_reveal_dual_action_frozen_score_evaluation_fix1.py

FIRST SUN-SEG GT pixel reveal after:
- R14C1 SOURCE/TENT1/PL prediction lock;
- R14C2B frozen pre-GT safety-score lock;
- R14C3A outcome-semantics lineage preflight.

Primary scientific question
---------------------------
Does the already-frozen SOURCE-conditioned safety score rank actual adaptation
harm under BOTH:
  A1_TENT_1STEP
  A4_PL_CONF90_1STEP
on the untouched SUN-SEG confirmatory cohort?

Frozen outcome rule (historical R05D4/R13C/R10L3C):
  delta_dice = adapted_dice - source_dice
  HARM     if delta_dice <= -0.02
  BENEFIT  if delta_dice >= +0.02
  NEUTRAL  otherwise

Frozen score:
  R14C2B frozen_safety_probability
  higher = higher HARM risk
  fixed operating threshold = 0.300584763193734

Primary evaluation:
- macro over the 3 model families;
- point metrics computed by the historical R10L3C implementation;
- bootstrap cluster = SUN physical case/video identity (`cluster_id`);
- no Easy/Hard split stratification, because R14B2 established overlap of
  physical cases across Easy-Unseen and Hard-Unseen;
- paired bootstrap comparison PL vs TENT1 uses identical sampled case clusters.

No model inference, no TTA rerun, no representation change, no PCA/head fit,
no calibration, no threshold change, no post-GT case/state/split exclusion.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
import shutil
import sys
import tarfile
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


VERSION = "2026-09-03-Q1-R14C3B-v1-fix1"
BUILD = "Q1_R14C3B_SUNSEG_GT_REVEAL_DUAL_ACTION_FROZEN_SCORE_EVALUATION_FIX1"

ROOT = Path(r"F:\MEDSEG_SAFETTA")
CODE = ROOT / "code"
OUT = ROOT / "outputs"

SUN_TAR = (
    ROOT / "data" / "external" / "SUN_SEG"
    / "SUN-SEG-FinalData-v20251212.tar.gz"
)
SUN_FINAL_MANIFEST = (
    OUT
    / "Q1_R14B4B_sunseg_selected_rgb_contamination_audit_fix3_v1"
    / "R14B4B_FIX3_FINAL_CONFIRMATORY_MANIFEST.csv"
)

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

R14C2B_DIR = (
    OUT
    / "Q1_R14C2B_sunseg_frozen_safety_score_lock_preGT_fix1_v1"
)
R14C2B_LOCK = (
    R14C2B_DIR
    / "R14C2B_SUNSEG_FROZEN_SAFETY_SCORE_LOCK.json"
)
R14C2B_SCORE = (
    R14C2B_DIR
    / "R14C2B_FROZEN_SUNSEG_SAFETY_SCORES.csv"
)

R14C3A_DIR = (
    OUT
    / "Q1_R14C3A_sunseg_gt_reveal_outcome_lineage_preflight_fix1_v1"
)
R14C3A_AUDIT = (
    R14C3A_DIR
    / "R14C3A_SUNSEG_GT_REVEAL_OUTCOME_LINEAGE_AUDIT.json"
)

R05D4_SCRIPT = (
    CODE
    / "Q1_R05D4_neopolyp_gt_reveal_paot_confirmatory_evaluation_v1.py"
)
R13C_SCRIPT = (
    CODE
    / "Q1_R13C_pl_conf90_gt_utility_reveal_and_outcome_lock_fix1.py"
)
R10L3C_SCRIPT = (
    CODE
    / "Q1_R10L3C_polypgen_gt_reveal_prospective_external_evaluation_fix1.py"
)

EXPECTED_R14C2B_LOCK_SHA = (
    "29030a2301178575971c8f5fab2c131d7cc06ffa99dbbd98431c288a3343337d"
)
EXPECTED_R14C2B_DECISION = (
    "SUNSEG_FROZEN_SAFETY_SCORES_LOCKED_BEFORE_GT_REVEAL"
)

EXPECTED_R14C1_LOCK_SHA = (
    "802e585dc075df23e0eeddf2b7d613d229dd1f7c6ae99aeaa89cf7dd7b358163"
)
EXPECTED_R14C1_GLOBAL_SHA = (
    "1ea4fae25a270babb9b1b0b2d2f1c4f694f458079f573b6f070c098ab3a96e40"
)
EXPECTED_SUN_FINAL_MANIFEST_SHA = (
    "f38cc8e5af4a007df3394ec95fa621c0b819f488bdd797ddad9338ac12491b4d"
)

EXPECTED_R05D4_SHA = (
    "f26cf378ceaee69acc4782a551c07d613a1cc882acad938d6151c824e54633ed"
)
EXPECTED_R13C_SHA = (
    "f2dd94d59e61be0432a2117e60aca1dc7ec4352af15e0aea0a27afdbf293590b"
)
EXPECTED_R10L3C_SHA = (
    "6761de1da2c6ece73dd1b18f566517b0c438c37b060a369084bdf26624375e6f"
)

EXPECTED_R14C3A_DECISION = (
    "SUNSEG_GT_REVEAL_OUTCOME_SEMANTICS_LOCATED_READY_FOR_"
    "LOCKED_OUTCOME_CONSTRUCTION"
)

EXPECTED_FRAMES = 980
EXPECTED_PHYSICAL_CASES = 49
EXPECTED_STATES = 9
EXPECTED_ROWS = EXPECTED_FRAMES * EXPECTED_STATES

MASK_H = 352
MASK_W = 352
PACKED_BYTES = 15488
BITORDER = "little"

HARM_THRESHOLD = -0.02
BENEFIT_THRESHOLD = 0.02
FROZEN_THRESHOLD = 0.300584763193734

FAMILIES = (
    "DeepLabV3-R50",
    "PraNet",
    "SegFormer-B0",
)

# This seed is frozen BEFORE GT pixel reveal by this script version.
# The number of resamples is inherited at runtime from the historical R10L3C
# implementation and written into the lock.
SUN_BOOTSTRAP_BASE_SEED = 20260903

OUTPUT_DIR = (
    OUT
    / "Q1_R14C3B_sunseg_gt_reveal_dual_action_frozen_score_evaluation_fix1_v1"
)

DECISION_COMPLETE = (
    "SUNSEG_DUAL_ACTION_GT_OUTCOMES_AND_FROZEN_SCORE_EVALUATION_COMPLETE"
)
DECISION_DEGENERATE = (
    "SUNSEG_DUAL_ACTION_GT_OUTCOMES_LOCKED_ONE_OR_MORE_ACTIONS_NOT_EVALUABLE"
)

POINT_METRICS = (
    "harm_prevalence",
    "auroc",
    "auprc",
    "recall",
    "fpr",
    "empirical_ppv",
    "ppv_at_1pct",
    "oracle_target_fpr_at_r90_diagnostic",
)

BOOTSTRAP_METRICS = (
    "auroc",
    "auprc",
    "recall",
    "fpr",
    "empirical_ppv",
    "ppv_at_1pct",
    "oracle_target_fpr_at_r90_diagnostic",
)


def sha256_file(path: Path, chunk=8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


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
        writer = csv.DictWriter(f, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def validate_upstream():
    print("===== R14C3B FROZEN UPSTREAM GATES =====")

    required = (
        SUN_TAR,
        SUN_FINAL_MANIFEST,
        R14C1_LOCK,
        R14C1_GLOBAL,
        R14C2B_LOCK,
        R14C2B_SCORE,
        R14C3A_AUDIT,
        R05D4_SCRIPT,
        R13C_SCRIPT,
        R10L3C_SCRIPT,
    )
    for p in required:
        if not p.is_file():
            raise FileNotFoundError(p)

    if sha256_file(R14C2B_LOCK) != EXPECTED_R14C2B_LOCK_SHA:
        raise RuntimeError("R14C2B lock SHA changed.")
    c2b = load_json(R14C2B_LOCK)
    if c2b.get("status") != "FROZEN":
        raise RuntimeError("R14C2B status changed.")
    if c2b.get("decision") != EXPECTED_R14C2B_DECISION:
        raise RuntimeError("R14C2B decision changed.")

    if int(c2b.get("target_frames", -1)) != EXPECTED_FRAMES:
        raise RuntimeError("R14C2B frame count changed.")
    if int(c2b.get("physical_cases", -1)) != EXPECTED_PHYSICAL_CASES:
        raise RuntimeError("R14C2B physical-case count changed.")
    if int(c2b.get("model_states", -1)) != EXPECTED_STATES:
        raise RuntimeError("R14C2B state count changed.")
    if int(c2b.get("model_frame_rows", -1)) != EXPECTED_ROWS:
        raise RuntimeError("R14C2B model-frame count changed.")

    info = c2b.get("information_boundary", {})
    forbidden_true = (
        "sun_gt_path_or_member_accessed",
        "sun_gt_pixels_decoded",
        "sun_outcomes_revealed",
        "dice_computed",
        "delta_dice_computed",
        "harm_benefit_labels_computed",
        "pca_fit_called",
        "pca_refit_called",
        "safety_head_fit_called",
        "safety_head_refit_called",
        "target_calibration",
        "target_threshold_selection",
        "target_feature_selection",
        "target_score_orientation_change",
        "target_case_selection_from_scores",
    )
    for key in forbidden_true:
        if bool(info.get(key, True)):
            raise RuntimeError(
                f"R14C2B information boundary invalid: {key}"
            )

    est = c2b.get("estimator", {})
    if abs(float(est.get("frozen_threshold", math.nan)) - FROZEN_THRESHOLD) > 1e-15:
        raise RuntimeError("R14C2B frozen threshold changed.")
    if int(est.get("probability_rows", -1)) != EXPECTED_ROWS:
        raise RuntimeError("R14C2B score row count changed.")

    score_meta = c2b.get("artifacts", {}).get(R14C2B_SCORE.name)
    if not isinstance(score_meta, dict):
        raise RuntimeError("R14C2B lock missing score artifact metadata.")
    if sha256_file(R14C2B_SCORE) != str(score_meta.get("sha256", "")):
        raise RuntimeError("R14C2B score CSV SHA mismatch.")

    if sha256_file(R14C1_LOCK) != EXPECTED_R14C1_LOCK_SHA:
        raise RuntimeError("R14C1 lock SHA changed.")
    if sha256_file(R14C1_GLOBAL) != EXPECTED_R14C1_GLOBAL_SHA:
        raise RuntimeError("R14C1 global CSV SHA changed.")
    if sha256_file(SUN_FINAL_MANIFEST) != EXPECTED_SUN_FINAL_MANIFEST_SHA:
        raise RuntimeError("SUN final manifest SHA changed.")

    if sha256_file(R05D4_SCRIPT) != EXPECTED_R05D4_SHA:
        raise RuntimeError("R05D4 outcome code SHA changed.")
    if sha256_file(R13C_SCRIPT) != EXPECTED_R13C_SHA:
        raise RuntimeError("R13C outcome code SHA changed.")
    if sha256_file(R10L3C_SCRIPT) != EXPECTED_R10L3C_SHA:
        raise RuntimeError("R10L3C external evaluation code SHA changed.")

    c3a = load_json(R14C3A_AUDIT)
    if c3a.get("status") != "PASS":
        raise RuntimeError("R14C3A status changed.")
    if c3a.get("decision") != EXPECTED_R14C3A_DECISION:
        raise RuntimeError("R14C3A decision changed.")

    c3a_tent = c3a.get("historical_tent1_outcome_code", {})
    c3a_pl = c3a.get("historical_pl_outcome_code", {})
    c3a_l3c = c3a.get("historical_polypgen_r10l3c", {}) or {}

    if c3a_tent.get("sha256") != EXPECTED_R05D4_SHA:
        raise RuntimeError("R14C3A R05D4 SHA mismatch.")
    if c3a_pl.get("sha256") != EXPECTED_R13C_SHA:
        raise RuntimeError("R14C3A R13C SHA mismatch.")
    if c3a_l3c.get("sha256") != EXPECTED_R10L3C_SHA:
        raise RuntimeError("R14C3A R10L3C SHA mismatch.")

    print("R14C2B lock:", EXPECTED_R14C2B_LOCK_SHA)
    print("R14C1 lock:", EXPECTED_R14C1_LOCK_SHA)
    print("SUN manifest:", EXPECTED_SUN_FINAL_MANIFEST_SHA)
    print("R05D4 code:", EXPECTED_R05D4_SHA)
    print("R13C code:", EXPECTED_R13C_SHA)
    print("R10L3C code:", EXPECTED_R10L3C_SHA)
    print("R14C3A audit SHA:", sha256_file(R14C3A_AUDIT))
    print("PASS")

    return {
        "r14c2b_lock_sha256": EXPECTED_R14C2B_LOCK_SHA,
        "r14c2b_score_csv_sha256": sha256_file(R14C2B_SCORE),
        "r14c1_lock_sha256": EXPECTED_R14C1_LOCK_SHA,
        "r14c1_global_manifest_sha256": EXPECTED_R14C1_GLOBAL_SHA,
        "sun_final_manifest_sha256": EXPECTED_SUN_FINAL_MANIFEST_SHA,
        "r14c3a_audit_sha256": sha256_file(R14C3A_AUDIT),
        "r05d4_script_sha256": EXPECTED_R05D4_SHA,
        "r13c_script_sha256": EXPECTED_R13C_SHA,
        "r10l3c_script_sha256": EXPECTED_R10L3C_SHA,
    }


def validate_historical_semantics(r05d4, r13c, r10l3c):
    print("\n===== FROZEN OUTCOME / METRIC SEMANTICS =====")

    if abs(float(r05d4.HARM_THRESHOLD) - HARM_THRESHOLD) > 1e-15:
        raise RuntimeError("R05D4 HARM threshold changed.")
    if abs(float(r05d4.BENEFIT_THRESHOLD) - BENEFIT_THRESHOLD) > 1e-15:
        raise RuntimeError("R05D4 BENEFIT threshold changed.")

    if abs(float(r13c.HARM_THR) - HARM_THRESHOLD) > 1e-15:
        raise RuntimeError("R13C HARM threshold changed.")
    if abs(float(r13c.BENEFIT_THR) - BENEFIT_THRESHOLD) > 1e-15:
        raise RuntimeError("R13C BENEFIT threshold changed.")

    if abs(float(r10l3c.HARM_THRESHOLD) - HARM_THRESHOLD) > 1e-15:
        raise RuntimeError("R10L3C HARM threshold changed.")
    if abs(float(r10l3c.BENEFIT_THRESHOLD) - BENEFIT_THRESHOLD) > 1e-15:
        raise RuntimeError("R10L3C BENEFIT threshold changed.")
    if abs(float(r10l3c.FROZEN_THRESHOLD) - FROZEN_THRESHOLD) > 1e-15:
        raise RuntimeError("R10L3C frozen safety threshold changed.")

    for name in ("point_metric_row", "macro_family_point"):
        if not hasattr(r10l3c, name):
            raise RuntimeError(f"R10L3C missing metric helper: {name}")

    if not hasattr(r10l3c, "BOOTSTRAP_RESAMPLES"):
        raise RuntimeError("R10L3C lacks BOOTSTRAP_RESAMPLES.")

    # Exact boundary tests from historical outcome semantics.
    test_values = (-0.021, -0.020, -0.019, 0.0, 0.019, 0.020, 0.021)
    expected = (
        "HARM", "HARM", "NEUTRAL", "NEUTRAL",
        "NEUTRAL", "BENEFIT", "BENEFIT"
    )
    observed = tuple(r05d4.outcome_label(float(x)) for x in test_values)
    if observed != expected:
        raise RuntimeError("R05D4 outcome boundary self-test failed.")

    if tuple(r13c.outcome(float(x)) for x in test_values) != expected:
        raise RuntimeError("R13C outcome boundary self-test failed.")
    if tuple(r10l3c.outcome_label(float(x)) for x in test_values) != expected:
        raise RuntimeError("R10L3C outcome boundary self-test failed.")

    z = np.zeros((MASK_H, MASK_W), dtype=np.uint8)
    o = np.ones((MASK_H, MASK_W), dtype=np.uint8)
    if float(r05d4.binary_dice(z, z)) != 1.0:
        raise RuntimeError("R05D4 zero-zero Dice changed.")
    if float(r05d4.binary_dice(o, o)) != 1.0:
        raise RuntimeError("R05D4 one-one Dice changed.")
    if float(r05d4.binary_dice(z, o)) != 0.0:
        raise RuntimeError("R05D4 zero-one Dice changed.")

    print("delta = adapted_dice - source_dice")
    print("HARM: delta <= -0.02")
    print("BENEFIT: delta >= +0.02")
    print("NEUTRAL: otherwise")
    print("frozen safety threshold:", FROZEN_THRESHOLD)
    print("historical R10L3C bootstrap resamples:", int(r10l3c.BOOTSTRAP_RESAMPLES))
    print("SUN bootstrap cluster: physical cluster_id")
    print("SUN bootstrap seed:", SUN_BOOTSTRAP_BASE_SEED)
    print("PASS")


def read_manifest():
    df = pd.read_csv(SUN_FINAL_MANIFEST, low_memory=False)

    required = [
        "pair_key",
        "external_case_id",
        "cluster_id",
        "clip_id",
        "source_split_label",
        "frame_member",
        "gt_member",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"SUN manifest missing: {missing}")

    if len(df) != EXPECTED_FRAMES:
        raise RuntimeError("SUN manifest row count changed.")
    if df["pair_key"].astype(str).nunique() != EXPECTED_FRAMES:
        raise RuntimeError("SUN pair_key not unique.")
    if df["gt_member"].astype(str).nunique() != EXPECTED_FRAMES:
        raise RuntimeError("SUN GT member not unique.")
    if df["cluster_id"].astype(str).nunique() != EXPECTED_PHYSICAL_CASES:
        raise RuntimeError("SUN physical-case count changed.")

    df = df.copy()
    df["sample_id"] = df["pair_key"].astype(str)
    return df


def reveal_gt_masks(manifest):
    """
    Frozen SUN GT decode rule chosen BEFORE pixel access in this script:
      - selected official `gt_member` only;
      - PIL native decode;
      - convert to grayscale L;
      - foreground = gray > 0;
      - nearest-neighbor resize to 352x352;
      - final binary = >0.

    This rule is never changed based on observed pixel values.
    """
    print("\n===== FIRST SUN GT PIXEL REVEAL =====")
    print("GT decode rule: PIL convert L")
    print("native binary rule: gray > 0")
    print("resize: nearest 352x352")
    print("post-resize binary: >0")
    print("adaptive threshold: NO")
    print("morphological repair: NO")
    print("post-reveal exclusion: NO")

    member_to_sid = {
        str(r.gt_member).replace("\\", "/"): str(r.sample_id)
        for r in manifest.itertuples(index=False)
    }
    target_members = set(member_to_sid)

    masks = {}
    audit_rows = []
    found = set()

    with tarfile.open(SUN_TAR, "r|gz") as tf:
        with tqdm(
            total=len(target_members),
            desc="Decode frozen SUN GT",
            unit="mask",
            dynamic_ncols=True,
        ) as bar:
            for member in tf:
                if not member.isfile():
                    continue

                name = member.name.replace("\\", "/")
                if name not in target_members:
                    continue

                handle = tf.extractfile(member)
                if handle is None:
                    raise RuntimeError(f"Cannot extract GT member: {name}")
                blob = handle.read()
                raw_sha = sha256_bytes(blob)

                with Image.open(io.BytesIO(blob)) as im:
                    native_mode = str(im.mode)
                    native_w, native_h = im.size
                    gray = np.asarray(
                        im.convert("L"),
                        dtype=np.uint8,
                    )

                native_binary = (gray > 0).astype(np.uint8)
                native_unique = np.unique(gray)

                resized = Image.fromarray(
                    native_binary * 255,
                    mode="L",
                ).resize(
                    (MASK_W, MASK_H),
                    resample=Image.Resampling.NEAREST,
                )
                mask = (
                    np.asarray(resized, dtype=np.uint8) > 0
                ).astype(np.uint8)

                sid = member_to_sid[name]
                if sid in masks:
                    raise RuntimeError(f"Duplicate GT sample_id: {sid}")
                masks[sid] = mask
                found.add(name)

                audit_rows.append({
                    "sample_id": sid,
                    "gt_member": name,
                    "gt_raw_sha256": raw_sha,
                    "native_mode": native_mode,
                    "native_width": int(native_w),
                    "native_height": int(native_h),
                    "native_unique_value_count": int(len(native_unique)),
                    "native_unique_min": int(native_unique.min()),
                    "native_unique_max": int(native_unique.max()),
                    "native_foreground_pixels": int(native_binary.sum()),
                    "resized_foreground_pixels": int(mask.sum()),
                    "decode_rule": (
                        "PIL_L__foreground_gray_gt_0__"
                        "nearest_352x352__post_gt_0"
                    ),
                })

                bar.update(1)
                if len(found) == len(target_members):
                    break

    if found != target_members:
        missing = sorted(target_members - found)
        raise RuntimeError(
            f"Missing GT members={len(missing)} examples={missing[:10]}"
        )
    if len(masks) != EXPECTED_FRAMES:
        raise RuntimeError("Decoded GT frame count != 980.")

    empty = sum(int(m.sum()) == 0 for m in masks.values())
    print("GT masks decoded:", len(masks))
    print("empty resized GT masks:", empty)
    print("GT decode rule changed after reveal: NO")
    print("PASS")

    return masks, audit_rows


def validate_prediction_panel():
    df = pd.read_csv(R14C1_GLOBAL, low_memory=False)

    required = [
        "row_index",
        "sample_id",
        "external_case_id",
        "cluster_id",
        "clip_id",
        "source_split_label",
        "model_family",
        "model_state_id",
        "training_seed",
        "checkpoint_sha256",
        "state_prediction_npz_relpath",
        "state_lock_json_relpath",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"R14C1 global missing: {missing}")

    if len(df) != EXPECTED_ROWS:
        raise RuntimeError("R14C1 row count != 8820.")
    if (
        df[["sample_id", "model_state_id"]]
        .drop_duplicates()
        .shape[0]
        != EXPECTED_ROWS
    ):
        raise RuntimeError("R14C1 model-frame key not unique.")

    if df["sample_id"].astype(str).nunique() != EXPECTED_FRAMES:
        raise RuntimeError("R14C1 sample count != 980.")
    if df["cluster_id"].astype(str).nunique() != EXPECTED_PHYSICAL_CASES:
        raise RuntimeError("R14C1 cluster count != 49.")
    if df["model_state_id"].astype(str).nunique() != EXPECTED_STATES:
        raise RuntimeError("R14C1 state count != 9.")

    return df


def load_locked_scores(pred_panel):
    score = pd.read_csv(R14C2B_SCORE, low_memory=False)

    required = [
        "sample_id",
        "external_case_id",
        "cluster_id",
        "clip_id",
        "source_split_label",
        "model_family",
        "model_state_id",
        "training_seed",
        "checkpoint_sha256",
        "state_row_index",
        "frozen_safety_probability",
        "frozen_operating_threshold",
        "frozen_risk_flag",
    ]
    missing = [c for c in required if c not in score.columns]
    if missing:
        raise RuntimeError(f"R14C2B score missing: {missing}")

    if len(score) != EXPECTED_ROWS:
        raise RuntimeError("R14C2B score rows != 8820.")
    if (
        score[["sample_id", "model_state_id"]]
        .drop_duplicates()
        .shape[0]
        != EXPECTED_ROWS
    ):
        raise RuntimeError("R14C2B score model-frame key not unique.")

    thresholds = score["frozen_operating_threshold"].to_numpy(dtype=float)
    if not np.allclose(
        thresholds,
        FROZEN_THRESHOLD,
        rtol=0.0,
        atol=1e-15,
    ):
        raise RuntimeError("R14C2B frozen threshold column changed.")

    probs = score["frozen_safety_probability"].to_numpy(dtype=float)
    flags = score["frozen_risk_flag"].to_numpy(dtype=int)
    if not np.isfinite(probs).all():
        raise RuntimeError("R14C2B safety probability non-finite.")
    expected_flags = (probs >= FROZEN_THRESHOLD).astype(int)
    if not np.array_equal(flags, expected_flags):
        raise RuntimeError("R14C2B risk flags no longer reproduce threshold.")

    left = pred_panel[
        [
            "sample_id",
            "model_state_id",
            "model_family",
            "training_seed",
            "checkpoint_sha256",
            "cluster_id",
        ]
    ].copy()
    right = score[
        [
            "sample_id",
            "model_state_id",
            "model_family",
            "training_seed",
            "checkpoint_sha256",
            "cluster_id",
        ]
    ].copy()

    joined = left.merge(
        right,
        on=["sample_id", "model_state_id"],
        how="inner",
        validate="one_to_one",
        suffixes=("_pred", "_score"),
    )
    if len(joined) != EXPECTED_ROWS:
        raise RuntimeError("Prediction/score key sets differ.")

    for col in (
        "model_family",
        "training_seed",
        "checkpoint_sha256",
        "cluster_id",
    ):
        if not (
            joined[f"{col}_pred"].astype(str)
            == joined[f"{col}_score"].astype(str)
        ).all():
            raise RuntimeError(f"Prediction/score metadata mismatch: {col}")

    return score


def unpack_mask(packed: np.ndarray) -> np.ndarray:
    arr = np.asarray(packed, dtype=np.uint8)
    if arr.ndim != 1 or arr.size != PACKED_BYTES:
        raise RuntimeError(f"Packed mask shape invalid: {arr.shape}")
    return np.unpackbits(
        arr,
        bitorder=BITORDER,
        count=MASK_H * MASK_W,
    ).reshape(MASK_H, MASK_W).astype(np.uint8)


def build_dual_action_outcomes(pred_panel, gt_masks, r05d4):
    print("\n===== FROZEN SOURCE/TENT1/PL OUTCOME CONSTRUCTION =====")

    rows = []

    for state_id, group in pred_panel.groupby("model_state_id", sort=True):
        if len(group) != EXPECTED_FRAMES:
            raise RuntimeError(
                f"{state_id}: rows={len(group)} expected={EXPECTED_FRAMES}"
            )

        pred_rel = group["state_prediction_npz_relpath"].astype(str).unique()
        lock_rel = group["state_lock_json_relpath"].astype(str).unique()
        if len(pred_rel) != 1 or len(lock_rel) != 1:
            raise RuntimeError(f"{state_id}: ambiguous state artifacts.")

        npz_path = R14C1_DIR / pred_rel[0]
        lock_path = R14C1_DIR / lock_rel[0]
        if not npz_path.is_file():
            raise FileNotFoundError(npz_path)
        if not lock_path.is_file():
            raise FileNotFoundError(lock_path)

        state_lock = load_json(lock_path)
        if state_lock.get("status") != "PASS":
            raise RuntimeError(f"{state_id}: state lock not PASS.")
        if state_lock.get("model_state_id") != state_id:
            raise RuntimeError(f"{state_id}: state lock identity mismatch.")
        if sha256_file(npz_path) != state_lock.get("prediction_npz_sha256"):
            raise RuntimeError(f"{state_id}: state NPZ SHA mismatch.")

        with np.load(npz_path, allow_pickle=False) as z:
            expected_keys = {
                "source_masks_packed",
                "tent1_masks_packed",
                "pl_masks_packed",
            }
            if set(z.files) != expected_keys:
                raise RuntimeError(
                    f"{state_id}: prediction keys={z.files}"
                )
            source = np.asarray(z["source_masks_packed"], dtype=np.uint8)
            tent1 = np.asarray(z["tent1_masks_packed"], dtype=np.uint8)
            pl = np.asarray(z["pl_masks_packed"], dtype=np.uint8)

        expected_shape = (EXPECTED_FRAMES, PACKED_BYTES)
        if source.shape != expected_shape:
            raise RuntimeError(f"{state_id}: SOURCE shape={source.shape}")
        if tent1.shape != expected_shape:
            raise RuntimeError(f"{state_id}: TENT1 shape={tent1.shape}")
        if pl.shape != expected_shape:
            raise RuntimeError(f"{state_id}: PL shape={pl.shape}")

        group = group.sort_values(
            "row_index",
            kind="mergesort",
        )

        if group["row_index"].astype(int).tolist() != list(range(EXPECTED_FRAMES)):
            raise RuntimeError(f"{state_id}: row_index is not 0..979.")

        for row in tqdm(
            group.itertuples(index=False),
            total=EXPECTED_FRAMES,
            desc=f"GT utility {state_id}",
            unit="frame",
            dynamic_ncols=True,
        ):
            sid = str(row.sample_id)
            idx = int(row.row_index)

            if sid not in gt_masks:
                raise RuntimeError(f"{state_id}: missing GT {sid}")

            gt = gt_masks[sid]
            source_mask = unpack_mask(source[idx])
            tent1_mask = unpack_mask(tent1[idx])
            pl_mask = unpack_mask(pl[idx])

            source_dice = float(r05d4.binary_dice(source_mask, gt))
            tent1_dice = float(r05d4.binary_dice(tent1_mask, gt))
            pl_dice = float(r05d4.binary_dice(pl_mask, gt))

            tent1_delta = float(tent1_dice - source_dice)
            pl_delta = float(pl_dice - source_dice)

            tent1_outcome = str(r05d4.outcome_label(tent1_delta))
            pl_outcome = str(r05d4.outcome_label(pl_delta))

            rows.append({
                "sample_id": sid,
                "external_case_id": str(row.external_case_id),
                "cluster_id": str(row.cluster_id),
                "clip_id": str(row.clip_id),
                "source_split_label": str(row.source_split_label),
                "model_family": str(row.model_family),
                "model_state_id": str(row.model_state_id),
                "training_seed": int(row.training_seed),
                "checkpoint_sha256": str(row.checkpoint_sha256),
                "state_row_index": idx,
                "source_dice": source_dice,
                "tent1_dice": tent1_dice,
                "tent1_delta_dice": tent1_delta,
                "tent1_adaptation_outcome": tent1_outcome,
                "tent1_harm_label": int(tent1_outcome == "HARM"),
                "tent1_benefit_label": int(tent1_outcome == "BENEFIT"),
                "pl_dice": pl_dice,
                "pl_delta_dice": pl_delta,
                "pl_adaptation_outcome": pl_outcome,
                "pl_harm_label": int(pl_outcome == "HARM"),
                "pl_benefit_label": int(pl_outcome == "BENEFIT"),
            })

        del source, tent1, pl

    out = pd.DataFrame(rows)
    if len(out) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Outcome rows={len(out)} expected={EXPECTED_ROWS}"
        )
    if (
        out[["sample_id", "model_state_id"]]
        .drop_duplicates()
        .shape[0]
        != EXPECTED_ROWS
    ):
        raise RuntimeError("Outcome model-frame key not unique.")

    return out


def join_locked_scores(outcomes, score):
    print("\n===== JOIN ALREADY-LOCKED SAFETY SCORES =====")

    score_cols = [
        "sample_id",
        "model_state_id",
        "model_family",
        "training_seed",
        "checkpoint_sha256",
        "cluster_id",
        "frozen_safety_probability",
        "frozen_operating_threshold",
        "frozen_risk_flag",
    ]

    panel = outcomes.merge(
        score[score_cols],
        on=["sample_id", "model_state_id"],
        how="inner",
        validate="one_to_one",
        suffixes=("_outcome", "_score"),
    )

    if len(panel) != EXPECTED_ROWS:
        raise RuntimeError("Outcome/score panel rows != 8820.")

    for col in (
        "model_family",
        "training_seed",
        "checkpoint_sha256",
        "cluster_id",
    ):
        if not (
            panel[f"{col}_outcome"].astype(str)
            == panel[f"{col}_score"].astype(str)
        ).all():
            raise RuntimeError(f"Outcome/score metadata mismatch: {col}")
        panel[col] = panel[f"{col}_outcome"]

    drop_cols = [
        f"{col}_{suffix}"
        for col in (
            "model_family",
            "training_seed",
            "checkpoint_sha256",
            "cluster_id",
        )
        for suffix in ("outcome", "score")
    ]
    panel = panel.drop(columns=drop_cols)

    if panel["cluster_id"].astype(str).nunique() != EXPECTED_PHYSICAL_CASES:
        raise RuntimeError("Evaluation panel cluster count !=49.")

    print("evaluation rows:", len(panel))
    print("unique frames:", panel["sample_id"].nunique())
    print("physical cases:", panel["cluster_id"].nunique())
    print("model states:", panel["model_state_id"].nunique())
    print(
        "frozen score min/median/max:",
        float(panel["frozen_safety_probability"].min()),
        float(panel["frozen_safety_probability"].median()),
        float(panel["frozen_safety_probability"].max()),
    )
    print(
        "frozen risk flags:",
        int(panel["frozen_risk_flag"].sum()),
        "/",
        len(panel),
    )
    print("PASS")

    return panel


def action_view(panel: pd.DataFrame, action: str) -> pd.DataFrame:
    if action not in {"TENT1", "PL"}:
        raise ValueError(action)

    prefix = "tent1" if action == "TENT1" else "pl"

    out = panel.copy()
    out["harm_label"] = out[f"{prefix}_harm_label"].astype(int)
    out["benefit_label"] = out[f"{prefix}_benefit_label"].astype(int)
    out["adaptation_outcome"] = out[f"{prefix}_adaptation_outcome"].astype(str)
    out["delta_dice"] = out[f"{prefix}_delta_dice"].astype(float)
    out["a1_dice"] = out[f"{prefix}_dice"].astype(float)
    return out


def action_evaluable(view: pd.DataFrame) -> bool:
    y = view["harm_label"].to_numpy(dtype=int)
    return bool(np.unique(y).size == 2 and y.sum() > 0 and y.sum() < len(y))


def point_metrics_for_action(view: pd.DataFrame, action: str, r10l3c):
    if not action_evaluable(view):
        return None, []

    family_rows = []
    for family in FAMILIES:
        subset = view[view["model_family"] == family]
        row = r10l3c.point_metric_row(
            family,
            f"{action}_FAMILY",
            subset,
        )
        family_rows.append(row)

    macro = r10l3c.macro_family_point(family_rows)
    macro = dict(macro)
    macro["action"] = action
    macro["level"] = "PRIMARY_MACRO_OVER_3_FAMILIES"

    return macro, family_rows


def safe_float(x):
    try:
        value = float(x)
    except Exception:
        return math.nan
    return value


def cluster_bootstrap_dual(panel: pd.DataFrame, r10l3c):
    """
    Primary confirmatory bootstrap unit is frozen SUN physical case/video
    identity (`cluster_id`), not frame/sample_id.

    The same sampled case multiplicities are used for TENT1 and PL, yielding
    paired action-difference CIs.
    """
    n_resamples = int(r10l3c.BOOTSTRAP_RESAMPLES)
    if n_resamples < 100:
        raise RuntimeError(
            f"Historical R10L3C bootstrap resamples too small: {n_resamples}"
        )

    clusters = sorted(panel["cluster_id"].astype(str).unique())
    if len(clusters) != EXPECTED_PHYSICAL_CASES:
        raise RuntimeError("Bootstrap cluster count !=49.")

    by_cluster = {
        cluster: panel[
            panel["cluster_id"].astype(str) == cluster
        ].copy()
        for cluster in clusters
    }

    rng = np.random.RandomState(SUN_BOOTSTRAP_BASE_SEED)

    action_records = {"TENT1": [], "PL": []}
    paired_records = []

    for rep in tqdm(
        range(n_resamples),
        desc="SUN physical-case bootstrap",
        unit="rep",
        dynamic_ncols=True,
    ):
        sampled = rng.choice(
            clusters,
            size=len(clusters),
            replace=True,
        )

        pieces = []
        for draw_id, cluster in enumerate(sampled):
            piece = by_cluster[str(cluster)].copy()
            # Unique bootstrap identity avoids any downstream uniqueness
            # assumptions while preserving the sampled cluster multiplicity.
            piece["_bootstrap_draw"] = int(draw_id)
            pieces.append(piece)

        boot = pd.concat(pieces, ignore_index=True)

        rep_metrics = {}

        for action in ("TENT1", "PL"):
            view = action_view(boot, action)

            # All three family metrics must be evaluable for the primary macro.
            try:
                macro, _ = point_metrics_for_action(
                    view,
                    action,
                    r10l3c,
                )
            except Exception:
                macro = None

            if macro is None:
                continue

            numeric = {
                metric: safe_float(macro.get(metric))
                for metric in BOOTSTRAP_METRICS
            }
            if not all(np.isfinite(v) for v in numeric.values()):
                continue

            rec = {
                "bootstrap_rep": rep,
                "action": action,
                **numeric,
            }
            action_records[action].append(rec)
            rep_metrics[action] = numeric

        if "TENT1" in rep_metrics and "PL" in rep_metrics:
            paired = {
                "bootstrap_rep": rep,
            }
            for metric in BOOTSTRAP_METRICS:
                paired[f"PL_minus_TENT1__{metric}"] = (
                    rep_metrics["PL"][metric]
                    - rep_metrics["TENT1"][metric]
                )
            paired_records.append(paired)

    return (
        n_resamples,
        action_records,
        paired_records,
    )


def percentile_ci(values):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return math.nan, math.nan, math.nan
    return (
        float(np.mean(x)),
        float(np.quantile(x, 0.025)),
        float(np.quantile(x, 0.975)),
    )


def summarize_bootstrap(
    n_resamples,
    action_records,
    paired_records,
):
    ci_rows = []

    for action in ("TENT1", "PL"):
        recs = action_records[action]
        for metric in BOOTSTRAP_METRICS:
            mean, low, high = percentile_ci(
                [r[metric] for r in recs]
            )
            ci_rows.append({
                "comparison": action,
                "metric": metric,
                "bootstrap_requested": n_resamples,
                "bootstrap_valid": len(recs),
                "bootstrap_mean": mean,
                "ci95_low": low,
                "ci95_high": high,
                "cluster_unit": "physical_case_cluster_id",
            })

    for metric in BOOTSTRAP_METRICS:
        key = f"PL_minus_TENT1__{metric}"
        mean, low, high = percentile_ci(
            [r[key] for r in paired_records]
        )
        ci_rows.append({
            "comparison": "PL_minus_TENT1",
            "metric": metric,
            "bootstrap_requested": n_resamples,
            "bootstrap_valid": len(paired_records),
            "bootstrap_mean": mean,
            "ci95_low": low,
            "ci95_high": high,
            "cluster_unit": "physical_case_cluster_id",
        })

    return ci_rows


def action_mechanism_summary(panel: pd.DataFrame):
    tent_h = panel["tent1_harm_label"].to_numpy(dtype=int)
    pl_h = panel["pl_harm_label"].to_numpy(dtype=int)

    intersection = int(np.sum((tent_h == 1) & (pl_h == 1)))
    union = int(np.sum((tent_h == 1) | (pl_h == 1)))
    jaccard = float(intersection / union) if union else math.nan

    agreement = float(np.mean(tent_h == pl_h))
    delta_spearman = float(
        panel["tent1_delta_dice"].corr(
            panel["pl_delta_dice"],
            method="spearman",
        )
    )

    # Cohen kappa implemented directly for the binary harm indicators.
    p0 = agreement
    p_tent = float(tent_h.mean())
    p_pl = float(pl_h.mean())
    pe = p_tent * p_pl + (1.0 - p_tent) * (1.0 - p_pl)
    kappa = (
        float((p0 - pe) / (1.0 - pe))
        if abs(1.0 - pe) > 1e-15
        else math.nan
    )

    return {
        "harm_intersection_rows": intersection,
        "harm_union_rows": union,
        "harm_jaccard": jaccard,
        "harm_binary_agreement": agreement,
        "harm_cohen_kappa": kappa,
        "delta_dice_spearman": delta_spearman,
    }


def outcome_summary(panel: pd.DataFrame, action: str):
    prefix = "tent1" if action == "TENT1" else "pl"

    labels = Counter(
        panel[f"{prefix}_adaptation_outcome"].astype(str)
    )
    delta = panel[f"{prefix}_delta_dice"].to_numpy(dtype=float)
    adapted_dice = panel[f"{prefix}_dice"].to_numpy(dtype=float)
    source_dice = panel["source_dice"].to_numpy(dtype=float)

    return {
        "action": action,
        "rows": len(panel),
        "frames": panel["sample_id"].nunique(),
        "physical_cases": panel["cluster_id"].nunique(),
        "states": panel["model_state_id"].nunique(),
        "harm_rows": int(labels.get("HARM", 0)),
        "neutral_rows": int(labels.get("NEUTRAL", 0)),
        "benefit_rows": int(labels.get("BENEFIT", 0)),
        "harm_prevalence": float(labels.get("HARM", 0) / len(panel)),
        "benefit_prevalence": float(labels.get("BENEFIT", 0) / len(panel)),
        "source_dice_mean": float(np.mean(source_dice)),
        "source_dice_median": float(np.median(source_dice)),
        "adapted_dice_mean": float(np.mean(adapted_dice)),
        "adapted_dice_median": float(np.median(adapted_dice)),
        "delta_dice_mean": float(np.mean(delta)),
        "delta_dice_median": float(np.median(delta)),
        "delta_dice_min": float(np.min(delta)),
        "delta_dice_max": float(np.max(delta)),
        "harm_evaluable": action_evaluable(action_view(panel, action)),
    }


def family_outcome_summary(panel: pd.DataFrame):
    rows = []
    for family in FAMILIES:
        sub = panel[panel["model_family"] == family]
        for action in ("TENT1", "PL"):
            row = outcome_summary(sub, action)
            row["model_family"] = family
            rows.append(row)
    return rows


def print_action_result(summary, macro, family_rows):
    action = summary["action"]

    print(f"\n===== {action} GT OUTCOME =====")
    print(
        "HARM/NEUTRAL/BENEFIT:",
        summary["harm_rows"],
        summary["neutral_rows"],
        summary["benefit_rows"],
    )
    print("HARM prevalence:", summary["harm_prevalence"])
    print(
        "SOURCE Dice mean/median:",
        summary["source_dice_mean"],
        summary["source_dice_median"],
    )
    print(
        f"{action} Dice mean/median:",
        summary["adapted_dice_mean"],
        summary["adapted_dice_median"],
    )
    print(
        "DeltaDice mean/median:",
        summary["delta_dice_mean"],
        summary["delta_dice_median"],
    )
    print("HARM evaluable:", summary["harm_evaluable"])

    if macro is None:
        print("PRIMARY MACRO FROZEN-SCORE EVALUATION: NOT EVALUABLE")
        return

    print(f"\n===== {action} PRIMARY MACRO OVER 3 FAMILIES =====")
    for metric in POINT_METRICS:
        print(
            f"{metric}:",
            float(macro[metric]),
        )

    print(f"\n===== {action} FAMILY POINT RESULTS =====")
    for row in family_rows:
        print(
            row["group"],
            f"AUROC={float(row['auroc']):.6f}",
            f"AUPRC={float(row['auprc']):.6f}",
            f"Recall={float(row['recall']):.6f}",
            f"FPR={float(row['fpr']):.6f}",
            f"PPV1%={float(row['ppv_at_1pct']):.6f}",
            f"OracleFPR90="
            f"{float(row['oracle_target_fpr_at_r90_diagnostic']):.6f}",
        )


def main_run(args):
    print(
        "===== Q1 R14C3B SUN-SEG FIRST GT REVEAL + "
        "DUAL-ACTION FROZEN-SCORE EVALUATION ====="
    )
    print("STATUS=FIRST_GT_PIXEL_REVEAL_AFTER_R14C2B_SCORE_LOCK")
    print("MODEL_INFERENCE=NO")
    print("TTA_RERUN=NO")
    print("SAFETY_MODEL_FIT_REFIT=NO")
    print("SCORE_ORIENTATION_CHANGE=NO")
    print("THRESHOLD_CHANGE=NO")
    print("TARGET_CALIBRATION=NO")
    print("TARGET_FEATURE_SELECTION=NO")
    print("TARGET_CASE_STATE_SPLIT_SELECTION=NO")
    print("FROZEN_SAFETY_THRESHOLD=", FROZEN_THRESHOLD)
    print("FROZEN_HARM_THRESHOLD=", HARM_THRESHOLD)
    print("FROZEN_BENEFIT_THRESHOLD=", BENEFIT_THRESHOLD)
    print("PRIMARY_BOOTSTRAP_CLUSTER=physical_case_cluster_id")
    print("SUN_BOOTSTRAP_BASE_SEED=", SUN_BOOTSTRAP_BASE_SEED)

    upstream_hashes = validate_upstream()

    r05d4 = import_module(
        R05D4_SCRIPT,
        "q1_r14c3b_authoritative_r05d4",
    )
    r13c = import_module(
        R13C_SCRIPT,
        "q1_r14c3b_authoritative_r13c",
    )
    r10l3c = import_module(
        R10L3C_SCRIPT,
        "q1_r14c3b_authoritative_r10l3c",
    )

    validate_historical_semantics(
        r05d4,
        r13c,
        r10l3c,
    )

    if args.output_dir.exists():
        raise FileExistsError(
            f"Final R14C3B output already exists: {args.output_dir}"
        )

    build = Path(str(args.output_dir) + "__building")
    if build.exists():
        raise FileExistsError(
            f"Partial post-GT build already exists: {build}. "
            "Do not silently delete a post-reveal partial run."
        )
    build.mkdir(parents=True, exist_ok=False)

    # Save the upstream freeze BEFORE the first GT pixel reveal.
    pre_reveal_path = build / "pre_gt_reveal_upstream_freeze.json"
    write_json(
        pre_reveal_path,
        {
            "script_version": VERSION,
            "build": BUILD,
            "upstream_hashes": upstream_hashes,
            "frozen_outcome_rule": {
                "delta": "adapted_dice - source_dice",
                "harm": "delta_dice <= -0.02",
                "benefit": "delta_dice >= +0.02",
                "neutral": "-0.02 < delta_dice < +0.02",
            },
            "sun_gt_decode_rule_frozen_before_pixel_access": {
                "native_decode": "PIL convert L",
                "binary_rule": "gray > 0",
                "resize": "nearest neighbor to 352x352",
                "post_resize_binary": ">0",
                "adaptive_threshold": False,
                "morphological_repair": False,
            },
            "frozen_score": {
                "threshold": FROZEN_THRESHOLD,
                "orientation": "higher = higher HARM risk",
                "recalibration": False,
            },
            "primary_evaluation": {
                "level": "macro over 3 model families",
                "bootstrap_cluster": "physical_case_cluster_id",
                "bootstrap_base_seed": SUN_BOOTSTRAP_BASE_SEED,
                "easy_hard_split_stratified": False,
                "reason": (
                    "Easy-Unseen and Hard-Unseen are not independent because "
                    "R14B2 established overlapping physical cases."
                ),
            },
        },
    )

    manifest = read_manifest()

    # -------- FIRST SUN GT PIXEL REVEAL --------
    gt_masks, gt_audit_rows = reveal_gt_masks(manifest)

    gt_audit_path = build / "R14C3B_SUN_GT_DECODE_AUDIT.csv"
    write_csv(
        gt_audit_path,
        gt_audit_rows,
        list(gt_audit_rows[0].keys()),
    )

    pred_panel = validate_prediction_panel()
    score = load_locked_scores(pred_panel)

    # Confirm exact frame sets.
    manifest_ids = set(manifest["sample_id"].astype(str))
    pred_ids = set(pred_panel["sample_id"].astype(str))
    score_ids = set(score["sample_id"].astype(str))
    gt_ids = set(gt_masks)

    if not (
        manifest_ids == pred_ids == score_ids == gt_ids
    ):
        raise RuntimeError(
            "SUN manifest/prediction/score/GT frame sets differ."
        )

    outcomes = build_dual_action_outcomes(
        pred_panel,
        gt_masks,
        r05d4,
    )
    outcomes_path = build / "R14C3B_DUAL_ACTION_MODEL_FRAME_OUTCOMES.csv"
    outcomes.to_csv(outcomes_path, index=False)

    panel = join_locked_scores(outcomes, score)
    panel_path = build / "R14C3B_LOCKED_SCORE_DUAL_ACTION_EVALUATION_PANEL.csv"
    panel.to_csv(panel_path, index=False)

    tent_summary = outcome_summary(panel, "TENT1")
    pl_summary = outcome_summary(panel, "PL")
    summaries = [tent_summary, pl_summary]

    family_outcomes = family_outcome_summary(panel)
    family_outcome_path = build / "R14C3B_FAMILY_ACTION_OUTCOME_SUMMARY.csv"
    pd.DataFrame(family_outcomes).to_csv(
        family_outcome_path,
        index=False,
    )

    tent_macro, tent_family = point_metrics_for_action(
        action_view(panel, "TENT1"),
        "TENT1",
        r10l3c,
    )
    pl_macro, pl_family = point_metrics_for_action(
        action_view(panel, "PL"),
        "PL",
        r10l3c,
    )

    print_action_result(
        tent_summary,
        tent_macro,
        tent_family,
    )
    print_action_result(
        pl_summary,
        pl_macro,
        pl_family,
    )

    mechanism = action_mechanism_summary(panel)

    print("\n===== TENT1 ↔ PL ACTION-SHIFT MECHANISM =====")
    for key, value in mechanism.items():
        print(key, "=", value)

    harm_transition = pd.crosstab(
        panel["tent1_harm_label"],
        panel["pl_harm_label"],
        rownames=["TENT1_harm"],
        colnames=["PL_harm"],
    )
    harm_transition_path = (
        build / "R14C3B_TENT1_TO_PL_HARM_TRANSITION.csv"
    )
    harm_transition.to_csv(harm_transition_path)

    tristate_transition = pd.crosstab(
        panel["tent1_adaptation_outcome"],
        panel["pl_adaptation_outcome"],
        rownames=["TENT1_outcome"],
        colnames=["PL_outcome"],
    )
    tristate_transition_path = (
        build / "R14C3B_TENT1_TO_PL_TRISTATE_TRANSITION.csv"
    )
    tristate_transition.to_csv(tristate_transition_path)

    point_rows = []
    for action, macro, family_rows in (
        ("TENT1", tent_macro, tent_family),
        ("PL", pl_macro, pl_family),
    ):
        if macro is not None:
            row = dict(macro)
            row["action"] = action
            row["scope"] = "PRIMARY_MACRO_OVER_3_FAMILIES"
            point_rows.append(row)

        for family_row in family_rows:
            row = dict(family_row)
            row["action"] = action
            row["scope"] = "SECONDARY_FAMILY"
            point_rows.append(row)

    point_path = build / "R14C3B_DUAL_ACTION_FROZEN_SCORE_POINT_METRICS.csv"
    if point_rows:
        pd.DataFrame(point_rows).to_csv(
            point_path,
            index=False,
        )
    else:
        pd.DataFrame(
            columns=["action", "scope"]
        ).to_csv(point_path, index=False)

    both_evaluable = (
        tent_macro is not None
        and pl_macro is not None
    )

    bootstrap_ci_rows = []
    n_bootstrap = int(r10l3c.BOOTSTRAP_RESAMPLES)
    valid_tent = 0
    valid_pl = 0
    valid_paired = 0

    if both_evaluable:
        print("\n===== PHYSICAL-CASE CLUSTERED BOOTSTRAP =====")
        (
            n_bootstrap,
            action_records,
            paired_records,
        ) = cluster_bootstrap_dual(
            panel,
            r10l3c,
        )

        valid_tent = len(action_records["TENT1"])
        valid_pl = len(action_records["PL"])
        valid_paired = len(paired_records)

        bootstrap_ci_rows = summarize_bootstrap(
            n_bootstrap,
            action_records,
            paired_records,
        )

        print(
            "valid bootstrap TENT1:",
            valid_tent,
            "/",
            n_bootstrap,
        )
        print(
            "valid bootstrap PL:",
            valid_pl,
            "/",
            n_bootstrap,
        )
        print(
            "valid paired bootstrap:",
            valid_paired,
            "/",
            n_bootstrap,
        )

        for row in bootstrap_ci_rows:
            print(
                row["comparison"],
                row["metric"],
                f"mean={row['bootstrap_mean']:.6f}",
                f"CI95=[{row['ci95_low']:.6f},"
                f"{row['ci95_high']:.6f}]",
            )

    bootstrap_path = (
        build / "R14C3B_PHYSICAL_CASE_BOOTSTRAP_CI.csv"
    )
    pd.DataFrame(bootstrap_ci_rows).to_csv(
        bootstrap_path,
        index=False,
    )

    outcome_summary_path = (
        build / "R14C3B_DUAL_ACTION_OUTCOME_SUMMARY.csv"
    )
    pd.DataFrame(summaries).to_csv(
        outcome_summary_path,
        index=False,
    )

    mechanism_path = (
        build / "R14C3B_ACTION_SHIFT_MECHANISM_SUMMARY.json"
    )
    write_json(mechanism_path, mechanism)

    decision = (
        DECISION_COMPLETE
        if both_evaluable
        else DECISION_DEGENERATE
    )

    info_boundary = {
        "r14c2b_score_lock_completed_before_gt": True,
        "sun_gt_pixels_first_decoded_in_r14c3b": True,
        "gt_frames_decoded": EXPECTED_FRAMES,
        "model_inference_run_in_r14c3b": False,
        "tta_run_in_r14c3b": False,
        "source_predictions_modified": False,
        "tent1_predictions_modified": False,
        "pl_predictions_modified": False,
        "safety_scores_modified": False,
        "safety_model_fit_or_refit": False,
        "score_orientation_changed": False,
        "threshold_changed": False,
        "target_calibration": False,
        "target_feature_selection": False,
        "target_case_selection": False,
        "target_state_selection": False,
        "target_split_selection": False,
        "post_gt_case_exclusion": False,
        "source_dice_computed": True,
        "tent1_dice_computed": True,
        "pl_dice_computed": True,
        "tent1_delta_dice_computed": True,
        "pl_delta_dice_computed": True,
        "dual_action_outcomes_revealed": True,
        "oracle_r90_threshold_deployed": False,
        "oracle_r90_metric_role": "DIAGNOSTIC_ONLY",
    }
    info_path = build / "information_boundary_audit.json"
    write_json(info_path, info_boundary)

    artifacts_to_hash = [
        pre_reveal_path,
        gt_audit_path,
        outcomes_path,
        panel_path,
        family_outcome_path,
        harm_transition_path,
        tristate_transition_path,
        point_path,
        bootstrap_path,
        outcome_summary_path,
        mechanism_path,
        info_path,
    ]

    artifacts = {
        p.name: {
            "sha256": sha256_file(p),
            "bytes": p.stat().st_size,
        }
        for p in artifacts_to_hash
    }

    lock = {
        "status": "COMPLETE",
        "decision": decision,
        "script_version": VERSION,
        "build": BUILD,
        "target_dataset": "SUN-SEG",
        "target_frames": EXPECTED_FRAMES,
        "physical_cases": EXPECTED_PHYSICAL_CASES,
        "model_states": EXPECTED_STATES,
        "model_frame_rows": EXPECTED_ROWS,
        "evidence_order": [
            "R14C1 frozen SOURCE/TENT1/PL prediction lock",
            "R14C2B frozen pre-GT safety-score lock",
            "R14C3A outcome-semantics lineage preflight",
            "R14C3B first SUN GT pixel reveal and evaluation",
        ],
        "gt_decode": {
            "native_decode": "PIL convert L",
            "binary_threshold": "gray > 0",
            "resize": "nearest neighbor to 352x352",
            "post_resize_binary": ">0",
            "adaptive_threshold": False,
            "morphological_repair": False,
            "post_reveal_exclusion": False,
        },
        "outcome_definition": {
            "source_metric": "historical R05D4 binary_dice",
            "tent1_delta":
                "tent1_dice - source_dice",
            "pl_delta":
                "pl_dice - source_dice",
            "harm": "delta_dice <= -0.02",
            "benefit": "delta_dice >= +0.02",
            "neutral": "-0.02 < delta_dice < +0.02",
        },
        "frozen_safety_evaluation": {
            "score":
                "R14C2B frozen_safety_probability",
            "score_orientation":
                "higher = higher HARM risk",
            "operating_threshold":
                FROZEN_THRESHOLD,
            "threshold_changed":
                False,
            "primary_level":
                "macro over 3 model families",
            "bootstrap_resamples":
                n_bootstrap,
            "bootstrap_seed":
                SUN_BOOTSTRAP_BASE_SEED,
            "bootstrap_cluster":
                "physical_case_cluster_id",
            "easy_hard_split_stratified":
                False,
            "oracle_target_fpr_at_r90":
                "diagnostic only; not deployed",
        },
        "action_outcomes": {
            "TENT1": tent_summary,
            "PL": pl_summary,
        },
        "action_shift_mechanism": mechanism,
        "primary_macro_point": {
            "TENT1": tent_macro,
            "PL": pl_macro,
        },
        "bootstrap_valid": {
            "TENT1": valid_tent,
            "PL": valid_pl,
            "paired_PL_minus_TENT1": valid_paired,
        },
        "information_boundary": info_boundary,
        "upstream_hashes": upstream_hashes,
        "artifacts": artifacts,
    }

    lock_path = (
        build
        / "R14C3B_SUNSEG_DUAL_ACTION_FROZEN_SCORE_EVALUATION_LOCK.json"
    )
    write_json(lock_path, lock)

    run_lines = [
        "===== R14C3B SUN-SEG DUAL-ACTION FROZEN-SCORE EVALUATION =====",
        f"Script version: {VERSION}",
        f"Build: {BUILD}",
        "",
        "Evidence order:",
        "  R14C1 predictions locked before GT=YES",
        "  R14C2B safety scores locked before GT=YES",
        "  first SUN GT pixel reveal=R14C3B",
        "",
        "Panel:",
        f"  frames={EXPECTED_FRAMES}",
        f"  physical_cases={EXPECTED_PHYSICAL_CASES}",
        f"  states={EXPECTED_STATES}",
        f"  model-frame rows={EXPECTED_ROWS}",
        "",
        "Frozen outcome:",
        "  delta_dice=adapted Dice - SOURCE Dice",
        "  HARM=delta<=-0.02",
        "  BENEFIT=delta>=+0.02",
        "  NEUTRAL=otherwise",
        "",
        "Frozen safety operating point:",
        f"  threshold={FROZEN_THRESHOLD:.15f}",
        "  recalibration=NO",
        "  score reversal=NO",
        "  target subset selection=NO",
        "",
    ]

    for summary, macro in (
        (tent_summary, tent_macro),
        (pl_summary, pl_macro),
    ):
        action = summary["action"]
        run_lines += [
            f"{action} outcomes:",
            f"  HARM={summary['harm_rows']}",
            f"  NEUTRAL={summary['neutral_rows']}",
            f"  BENEFIT={summary['benefit_rows']}",
            f"  HARM prevalence={summary['harm_prevalence']:.9f}",
            f"  DeltaDice mean={summary['delta_dice_mean']:.9f}",
            f"  DeltaDice median={summary['delta_dice_median']:.9f}",
        ]
        if macro is not None:
            run_lines += [
                f"  Macro AUROC={float(macro['auroc']):.9f}",
                f"  Macro AUPRC={float(macro['auprc']):.9f}",
                f"  Macro Recall={float(macro['recall']):.9f}",
                f"  Macro FPR={float(macro['fpr']):.9f}",
                f"  Macro empirical PPV={float(macro['empirical_ppv']):.9f}",
                f"  Macro PPV@1%={float(macro['ppv_at_1pct']):.9f}",
                "  Macro oracle target FPR@R90 diagnostic="
                f"{float(macro['oracle_target_fpr_at_r90_diagnostic']):.9f}",
            ]
        else:
            run_lines += [
                "  frozen-score ranking evaluation=NOT EVALUABLE",
            ]
        run_lines.append("")

    run_lines += [
        "Action-shift mechanism:",
        f"  HARM Jaccard={mechanism['harm_jaccard']}",
        f"  HARM agreement={mechanism['harm_binary_agreement']}",
        f"  HARM Cohen kappa={mechanism['harm_cohen_kappa']}",
        f"  DeltaDice Spearman={mechanism['delta_dice_spearman']}",
        "",
        "Primary bootstrap:",
        f"  cluster=physical_case_cluster_id",
        f"  requested={n_bootstrap}",
        f"  valid TENT1={valid_tent}",
        f"  valid PL={valid_pl}",
        f"  valid paired={valid_paired}",
        "",
        f"Decision={decision}",
        "PASS",
    ]

    run_log_path = build / "run_log.txt"
    run_log_path.write_text(
        "\n".join(run_lines) + "\n",
        encoding="utf-8",
    )

    # Add final log to artifact list and refresh lock once.
    lock["artifacts"][run_log_path.name] = {
        "sha256": sha256_file(run_log_path),
        "bytes": run_log_path.stat().st_size,
    }
    write_json(lock_path, lock)

    lock_sha = sha256_file(lock_path)

    # No silent rollback after first GT reveal. Successful commit only.
    build.rename(args.output_dir)

    print("\n===== R14C3B FINAL =====")
    print(
        (args.output_dir / "run_log.txt").read_text(
            encoding="utf-8"
        )
    )
    print(
        "LOCK=",
        args.output_dir
        / "R14C3B_SUNSEG_DUAL_ACTION_FROZEN_SCORE_EVALUATION_LOCK.json",
    )
    print("LOCK SHA256=", lock_sha)
    print("PASS")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        main_run(args)
    except Exception:
        traceback.print_exc()
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
