#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R10L3C_polypgen_gt_reveal_prospective_external_evaluation_fix1.py

FIRST PolypGen GT reveal after:
- R10L3A SOURCE/A1 prediction lock;
- R10L3B frozen safety-score lock.

This stage:
1) verifies all frozen upstream locks/artifacts;
2) reveals only the already-locked 1532 PolypGen GT masks;
3) computes SOURCE Dice, A1 Dice, DeltaDice;
4) applies the exact frozen development outcome rule:
       HARM    if DeltaDice <= -0.02
       BENEFIT if DeltaDice >= +0.02
       NEUTRAL otherwise
5) joins the already-locked R10L3B safety probability;
6) evaluates ranking + the already-frozen operating threshold WITHOUT
   recalibration, reversal, feature selection, threshold selection, or
   case/center/state selection;
7) runs 2000 case-clustered, center-stratified bootstrap resamples.

Primary external reporting:
- macro over the three model families;
- AUROC, AUPRC;
- frozen-threshold Recall, FPR, empirical PPV;
- prevalence-adjusted PPV at 1% harm prevalence;
- case-clustered 95% CIs.

Secondary:
- pooled metrics;
- family metrics;
- center metrics;
- state metrics;
- oracle target FPR@Recall>=0.90 as a diagnostic only (NEVER deployed).

No model inference or TTA is run in this stage.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm


VERSION = "2026-08-26-Q1-R10L3C-v1-fix1"
BUILD = "Q1_R10L3C_POLYPGEN_GT_REVEAL_PROSPECTIVE_EXTERNAL_EVALUATION_FIX1"

ROOT = Path(r"F:\MEDSEG_SAFETTA")

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

R10L3A_DIR = (
    ROOT / "outputs"
    / "Q1_R10L3A_polypgen_source_a1_prediction_lock_fix2_v1"
)
R10L3A_LOCK = (
    R10L3A_DIR
    / "R10L3A_POLYPGEN_SOURCE_A1_PREDICTION_LOCK.json"
)
EXPECTED_R10L3A_LOCK_SHA256 = (
    "551a2bb372e1cd36cdbe982661b20f7050617a1398e1a773b88f027f75e3bb53"
)
EXPECTED_R10L3A_DECISION = (
    "POLYPGEN_SOURCE_A1_PREDICTIONS_LOCKED_BEFORE_GT_REVEAL"
)
R10L3A_MODEL_CASE_MANIFEST = (
    R10L3A_DIR / "model_case_prediction_manifest.csv"
)

R10L3B_DIR = (
    ROOT / "outputs"
    / "Q1_R10L3B_polypgen_frozen_safety_score_lock_fix1_v1"
)
R10L3B_LOCK = (
    R10L3B_DIR
    / "R10L3B_POLYPGEN_FROZEN_SAFETY_SCORE_LOCK.json"
)
EXPECTED_R10L3B_LOCK_SHA256 = (
    "3c0dd916e56f55a0398dc085c5f007d1fec65878efabd62b5a1c46d00e77f218"
)
EXPECTED_R10L3B_DECISION = (
    "POLYPGEN_FROZEN_SAFETY_SCORES_LOCKED_BEFORE_GT_REVEAL"
)
R10L3B_SCORE_CSV = (
    R10L3B_DIR
    / "R10L3B_FROZEN_POLYPGEN_SAFETY_SCORES.csv"
)

R05D4_OUTCOMES = (
    ROOT / "outputs"
    / "Q1_R05D4_neopolyp_gt_reveal_paot_confirmatory_evaluation_v1"
    / "model_case_outcomes.csv"
)

OUTPUT_DIR = (
    ROOT / "outputs"
    / "Q1_R10L3C_polypgen_gt_reveal_prospective_external_evaluation_fix1_v1"
)

EXPECTED_CASES = 1532
EXPECTED_STATES = 9
EXPECTED_MODEL_CASES = 13788
MASK_H = 352
MASK_W = 352
MASK_PIXELS = MASK_H * MASK_W
PACKED_BYTES = 15488
BITORDER = "little"

FROZEN_THRESHOLD = 0.300584763193734
HARM_THRESHOLD = -0.02
BENEFIT_THRESHOLD = +0.02
ASSUMED_PREVALENCE = 0.01

BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20260826

FAMILIES = ("DeepLabV3-R50", "PraNet", "SegFormer-B0")
CENTERS = ("C1", "C2", "C3", "C4", "C5", "C6")

DECISION_COMPLETE = (
    "POLYPGEN_PROSPECTIVE_EXTERNAL_EVALUATION_COMPLETE_NO_RETUNING"
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


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def verify_required_paths():
    paths = {
        "PolypGen manifest": POLYPGEN_MANIFEST,
        "PolypGen manifest lock": POLYPGEN_MANIFEST_LOCK,
        "independence lock": INDEPENDENCE_LOCK,
        "R10L3A lock": R10L3A_LOCK,
        "R10L3A model-case manifest": R10L3A_MODEL_CASE_MANIFEST,
        "R10L3B lock": R10L3B_LOCK,
        "R10L3B score CSV": R10L3B_SCORE_CSV,
        "R05D4 development outcomes": R05D4_OUTCOMES,
    }
    for label, p in paths.items():
        print(label, "exists:", p.exists(), p)
        if not p.exists():
            raise FileNotFoundError(p)


def verify_upstream_locks():
    pg_lock = json.loads(
        POLYPGEN_MANIFEST_LOCK.read_text(encoding="utf-8")
    )
    if pg_lock.get("decision") != EXPECTED_POLYPGEN_MANIFEST_DECISION:
        raise RuntimeError(
            f"Unexpected PolypGen manifest decision: {pg_lock.get('decision')}"
        )
    if int(pg_lock.get("retained_unique_rows", -1)) != EXPECTED_CASES:
        raise RuntimeError("Frozen PolypGen case count changed.")

    indep = json.loads(INDEPENDENCE_LOCK.read_text(encoding="utf-8"))
    if indep.get("decision") != EXPECTED_INDEPENDENCE_DECISION:
        raise RuntimeError(
            f"Independence gate changed: {indep.get('decision')}"
        )
    if int(
        indep.get("total_source_development_exact_polypgen_overlaps", -1)
    ) != 0:
        raise RuntimeError("Source-development exact overlap is not zero.")

    l3a_sha = sha256_file(R10L3A_LOCK)
    if l3a_sha.lower() != EXPECTED_R10L3A_LOCK_SHA256.lower():
        raise RuntimeError(
            f"R10L3A lock SHA changed: {l3a_sha}"
        )
    l3a = json.loads(R10L3A_LOCK.read_text(encoding="utf-8"))
    if l3a.get("decision") != EXPECTED_R10L3A_DECISION:
        raise RuntimeError(
            f"R10L3A decision changed: {l3a.get('decision')}"
        )
    if int(l3a.get("model_case_rows", -1)) != EXPECTED_MODEL_CASES:
        raise RuntimeError("R10L3A model-case count changed.")
    if bool(l3a.get("target_gt_path_column_read", True)):
        raise RuntimeError("R10L3A reports GT-path read.")
    if bool(l3a.get("target_gt_pixels_decoded", True)):
        raise RuntimeError("R10L3A reports prior GT decode.")
    if bool(l3a.get("target_outcomes_revealed", True)):
        raise RuntimeError("R10L3A reports prior outcome reveal.")

    l3b_sha = sha256_file(R10L3B_LOCK)
    if l3b_sha.lower() != EXPECTED_R10L3B_LOCK_SHA256.lower():
        raise RuntimeError(
            f"R10L3B lock SHA changed: {l3b_sha}"
        )
    l3b = json.loads(R10L3B_LOCK.read_text(encoding="utf-8"))
    if l3b.get("decision") != EXPECTED_R10L3B_DECISION:
        raise RuntimeError(
            f"R10L3B decision changed: {l3b.get('decision')}"
        )
    if int(l3b.get("target_cases", -1)) != EXPECTED_CASES:
        raise RuntimeError("R10L3B target count changed.")
    if int(l3b.get("model_case_rows", -1)) != EXPECTED_MODEL_CASES:
        raise RuntimeError("R10L3B model-case count changed.")

    ib = l3b.get("information_boundary", {})
    forbidden_true = [
        "polypgen_gt_path_column_read",
        "polypgen_gt_pixels_decoded",
        "polypgen_outcomes_revealed",
        "external_dice_computed",
        "external_delta_dice_computed",
        "external_harm_label_computed",
        "pca_fit_called",
        "pca_refit_called",
        "safety_head_fit_called",
        "safety_head_refit_called",
        "target_calibration",
        "target_threshold_selection",
        "target_feature_selection",
        "target_score_orientation_change",
        "target_case_selection_from_scores",
    ]
    for key in forbidden_true:
        if bool(ib.get(key, True)):
            raise RuntimeError(
                f"R10L3B information boundary invalid: {key}={ib.get(key)}"
            )

    frozen_thr = float(
        l3b.get("estimator", {}).get("frozen_threshold", math.nan)
    )
    if abs(frozen_thr - FROZEN_THRESHOLD) > 1e-15:
        raise RuntimeError("R10L3B frozen threshold changed.")

    score_meta = l3b.get("artifacts", {}).get(
        R10L3B_SCORE_CSV.name
    )
    if not isinstance(score_meta, dict):
        raise RuntimeError("R10L3B lock missing score artifact metadata.")
    score_sha = sha256_file(R10L3B_SCORE_CSV)
    if score_sha.lower() != str(score_meta.get("sha256", "")).lower():
        raise RuntimeError("R10L3B score CSV SHA mismatch.")

    return {
        "polypgen_manifest_sha256": sha256_file(POLYPGEN_MANIFEST),
        "polypgen_manifest_lock_sha256": sha256_file(
            POLYPGEN_MANIFEST_LOCK
        ),
        "independence_lock_sha256": sha256_file(INDEPENDENCE_LOCK),
        "r10l3a_lock_sha256": l3a_sha,
        "r10l3b_lock_sha256": l3b_sha,
        "r10l3b_score_csv_sha256": score_sha,
    }


def verify_development_outcome_rule():
    df = pd.read_csv(
        R05D4_OUTCOMES,
        usecols=[
            "delta_dice",
            "adaptation_outcome",
            "harm_label",
            "benefit_label",
        ],
        low_memory=False,
    )
    if len(df) != 9000:
        raise RuntimeError("R05D4 development outcome table rows != 9000.")

    delta = df["delta_dice"].to_numpy(dtype=float)
    expected = np.where(
        delta <= HARM_THRESHOLD,
        "HARM",
        np.where(
            delta >= BENEFIT_THRESHOLD,
            "BENEFIT",
            "NEUTRAL",
        ),
    )
    outcome = (
        df["adaptation_outcome"]
        .astype(str)
        .str.upper()
        .to_numpy()
    )

    if not np.array_equal(expected, outcome):
        raise RuntimeError(
            "Frozen development outcome boundary no longer reproduces R05D4."
        )
    if not np.array_equal(
        (expected == "HARM").astype(int),
        df["harm_label"].to_numpy(dtype=int),
    ):
        raise RuntimeError("R05D4 harm_label boundary mismatch.")
    if not np.array_equal(
        (expected == "BENEFIT").astype(int),
        df["benefit_label"].to_numpy(dtype=int),
    ):
        raise RuntimeError("R05D4 benefit_label boundary mismatch.")

    print("\n===== FROZEN DEVELOPMENT OUTCOME RULE =====")
    print("HARM: DeltaDice <= -0.02")
    print("BENEFIT: DeltaDice >= +0.02")
    print("NEUTRAL: otherwise")
    print("R05D4 reproduction:", len(df), "/", len(df), "PASS")


def load_locked_gt_manifest():
    cols = [
        "external_case_id",
        "center",
        "sample_id",
        "image_path",
        "image_sha256",
        "gt_path",
        "gt_sha256",
    ]
    df = pd.read_csv(
        POLYPGEN_MANIFEST,
        usecols=cols,
        low_memory=False,
    )
    if len(df) != EXPECTED_CASES:
        raise RuntimeError("PolypGen locked GT manifest rows !=1532.")
    if df["external_case_id"].astype(str).nunique() != EXPECTED_CASES:
        raise RuntimeError("external_case_id not unique.")
    if set(df["center"].astype(str)) != set(CENTERS):
        raise RuntimeError("All six centers not represented.")
    return df


def reveal_gt_masks(gt_df):
    masks = {}
    audit_rows = []
    resampling = getattr(Image, "Resampling", Image)

    for r in tqdm(
        gt_df.itertuples(index=False),
        total=len(gt_df),
        desc="R10L3C FIRST POLYPGEN GT REVEAL",
        unit="gt",
        dynamic_ncols=True,
    ):
        sid = str(r.external_case_id)
        p = Path(str(r.gt_path))
        if not p.exists():
            raise FileNotFoundError(p)

        actual_sha = sha256_file(p).lower()
        expected_sha = str(r.gt_sha256).strip().lower()
        if actual_sha != expected_sha:
            raise RuntimeError(
                f"Frozen GT SHA mismatch: {sid}"
            )

        with Image.open(p) as im:
            native_w, native_h = im.size
            gray = np.asarray(im.convert("L"), dtype=np.uint8)

        native_min = int(gray.min())
        native_max = int(gray.max())
        native_unique = int(np.unique(gray).size)

        binary_native = (gray > 127).astype(np.uint8)
        binary_img = Image.fromarray(
            binary_native * 255,
            mode="L",
        )
        resized = binary_img.resize(
            (MASK_W, MASK_H),
            resample=resampling.NEAREST,
        )
        binary = (
            np.asarray(resized, dtype=np.uint8) > 0
        ).astype(np.uint8)

        if binary.shape != (MASK_H, MASK_W):
            raise RuntimeError(f"GT resize shape failure: {sid}")
        if not np.isin(binary, (0, 1)).all():
            raise RuntimeError(f"GT binary failure: {sid}")

        masks[sid] = binary
        audit_rows.append({
            "sample_id": sid,
            "center": str(r.center),
            "original_polypgen_sample_id": str(r.sample_id),
            "gt_path": str(p),
            "gt_sha256": actual_sha,
            "native_width": native_w,
            "native_height": native_h,
            "native_gray_min": native_min,
            "native_gray_max": native_max,
            "native_unique_gray_levels": native_unique,
            "native_foreground_pixels_gt127":
                int(binary_native.sum()),
            "resized_foreground_pixels":
                int(binary.sum()),
            "decode_rule":
                "grayscale_L_gt127_then_nearest_352x352",
        })

    if len(masks) != EXPECTED_CASES:
        raise RuntimeError("Not all PolypGen GT masks decoded.")

    return masks, audit_rows


def unpack_mask(packed: np.ndarray) -> np.ndarray:
    p = np.asarray(packed, dtype=np.uint8)
    if p.shape != (PACKED_BYTES,):
        raise RuntimeError(
            f"Packed mask shape={p.shape}; expected={(PACKED_BYTES,)}"
        )
    bits = np.unpackbits(
        p,
        count=MASK_PIXELS,
        bitorder=BITORDER,
    )
    return bits.reshape(MASK_H, MASK_W).astype(
        np.uint8,
        copy=False,
    )


def binary_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    p = np.asarray(pred, dtype=np.uint8)
    g = np.asarray(gt, dtype=np.uint8)
    if p.shape != (MASK_H, MASK_W):
        raise RuntimeError("Prediction mask shape mismatch.")
    if g.shape != (MASK_H, MASK_W):
        raise RuntimeError("GT mask shape mismatch.")

    ps = int(p.sum())
    gs = int(g.sum())
    denom = ps + gs
    if denom == 0:
        return 1.0

    inter = int(np.count_nonzero((p == 1) & (g == 1)))
    return float(2.0 * inter / denom)


def outcome_label(delta: float) -> str:
    if delta <= HARM_THRESHOLD:
        return "HARM"
    if delta >= BENEFIT_THRESHOLD:
        return "BENEFIT"
    return "NEUTRAL"


def resolve_r10l3a_path(raw: str) -> Path:
    p = Path(str(raw))
    if not p.is_absolute():
        p = R10L3A_DIR / p
    return p


def validate_state_sidecar(
    sidecar_path: Path,
    npz_path: Path,
    state_id: str,
):
    if not sidecar_path.exists():
        raise FileNotFoundError(sidecar_path)
    side = json.loads(sidecar_path.read_text(encoding="utf-8"))

    if side.get("decision") != "STATE_COMPLETE":
        raise RuntimeError(f"{state_id}: state sidecar incomplete.")
    if side.get("model_state_id") != state_id:
        raise RuntimeError(f"{state_id}: sidecar state ID mismatch.")
    if int(side.get("target_cases", -1)) != EXPECTED_CASES:
        raise RuntimeError(f"{state_id}: target count mismatch.")
    if bool(side.get("target_gt_pixels_decoded", True)):
        raise RuntimeError(f"{state_id}: prior GT decode reported.")
    if bool(side.get("target_metrics_computed", True)):
        raise RuntimeError(f"{state_id}: prior target metrics reported.")

    if sha256_file(npz_path).lower() != str(
        side.get("npz_sha256", "")
    ).lower():
        raise RuntimeError(f"{state_id}: prediction NPZ SHA mismatch.")

    return side


def build_external_outcomes(gt_masks):
    mc = pd.read_csv(
        R10L3A_MODEL_CASE_MANIFEST,
        low_memory=False,
    )

    required = [
        "sample_id",
        "center",
        "model_family",
        "model_state_id",
        "training_seed",
        "checkpoint_sha256",
        "state_prediction_npz",
        "state_lock_json",
        "state_row_index",
    ]
    missing = [c for c in required if c not in mc.columns]
    if missing:
        raise RuntimeError(
            f"R10L3A model-case manifest missing: {missing}"
        )
    if len(mc) != EXPECTED_MODEL_CASES:
        raise RuntimeError("R10L3A model-case rows !=13788.")
    if (
        mc[["sample_id", "model_state_id"]]
        .drop_duplicates()
        .shape[0]
        != EXPECTED_MODEL_CASES
    ):
        raise RuntimeError("R10L3A model-case key not unique.")

    rows = []

    for state_id, g in mc.groupby("model_state_id", sort=True):
        if len(g) != EXPECTED_CASES:
            raise RuntimeError(
                f"{state_id}: rows={len(g)} expected={EXPECTED_CASES}"
            )

        npz_vals = g["state_prediction_npz"].astype(str).unique()
        lock_vals = g["state_lock_json"].astype(str).unique()
        if len(npz_vals) != 1 or len(lock_vals) != 1:
            raise RuntimeError(
                f"{state_id}: expected one NPZ and one state lock."
            )

        npz_path = resolve_r10l3a_path(npz_vals[0])
        sidecar_path = resolve_r10l3a_path(lock_vals[0])

        if not npz_path.exists():
            raise FileNotFoundError(npz_path)

        validate_state_sidecar(
            sidecar_path,
            npz_path,
            state_id,
        )

        with np.load(npz_path, allow_pickle=False) as z:
            if "source_masks_packed" not in z.files:
                raise RuntimeError(f"{state_id}: SOURCE missing.")
            if "a1_masks_packed" not in z.files:
                raise RuntimeError(f"{state_id}: A1 missing.")
            source = np.asarray(
                z["source_masks_packed"],
                dtype=np.uint8,
            )
            a1 = np.asarray(
                z["a1_masks_packed"],
                dtype=np.uint8,
            )

        expected_shape = (EXPECTED_CASES, PACKED_BYTES)
        if source.shape != expected_shape:
            raise RuntimeError(
                f"{state_id}: SOURCE shape={source.shape}"
            )
        if a1.shape != expected_shape:
            raise RuntimeError(
                f"{state_id}: A1 shape={a1.shape}"
            )

        g = g.sort_values(
            "state_row_index",
            kind="mergesort",
        )

        for r in tqdm(
            g.itertuples(index=False),
            total=len(g),
            desc=f"GT utility {state_id}",
            unit="case",
            dynamic_ncols=True,
        ):
            sid = str(r.sample_id)
            if sid not in gt_masks:
                raise RuntimeError(
                    f"{state_id}: missing frozen GT {sid}"
                )

            i = int(r.state_row_index)
            if not 0 <= i < EXPECTED_CASES:
                raise RuntimeError(
                    f"{state_id}: invalid state_row_index={i}"
                )

            sm = unpack_mask(source[i])
            am = unpack_mask(a1[i])
            gt = gt_masks[sid]

            sd = binary_dice(sm, gt)
            ad = binary_dice(am, gt)
            delta = float(ad - sd)
            outcome = outcome_label(delta)

            rows.append({
                "sample_id": sid,
                "center": str(r.center),
                "model_family": str(r.model_family),
                "model_state_id": str(r.model_state_id),
                "training_seed": int(r.training_seed),
                "checkpoint_sha256": str(r.checkpoint_sha256),
                "state_row_index": i,
                "source_dice": sd,
                "a1_dice": ad,
                "delta_dice": delta,
                "adaptation_outcome": outcome,
                "harm_label": int(outcome == "HARM"),
                "benefit_label": int(outcome == "BENEFIT"),
            })

        del source, a1

    out = pd.DataFrame(rows)
    if len(out) != EXPECTED_MODEL_CASES:
        raise RuntimeError(
            f"Outcome rows={len(out)} expected={EXPECTED_MODEL_CASES}"
        )
    if (
        out[["sample_id", "model_state_id"]]
        .drop_duplicates()
        .shape[0]
        != EXPECTED_MODEL_CASES
    ):
        raise RuntimeError("Outcome model-case key not unique.")

    return out


def load_and_join_locked_scores(outcomes):
    score = pd.read_csv(
        R10L3B_SCORE_CSV,
        low_memory=False,
    )
    required = [
        "sample_id",
        "center",
        "model_family",
        "model_state_id",
        "training_seed",
        "checkpoint_sha256",
        "frozen_safety_probability",
        "frozen_operating_threshold",
        "frozen_risk_flag",
    ]
    missing = [c for c in required if c not in score.columns]
    if missing:
        raise RuntimeError(
            f"R10L3B score CSV missing: {missing}"
        )
    if len(score) != EXPECTED_MODEL_CASES:
        raise RuntimeError("R10L3B score rows !=13788.")
    if (
        score[["sample_id", "model_state_id"]]
        .drop_duplicates()
        .shape[0]
        != EXPECTED_MODEL_CASES
    ):
        raise RuntimeError("R10L3B score key not unique.")

    if not np.allclose(
        score["frozen_operating_threshold"].to_numpy(dtype=float),
        FROZEN_THRESHOLD,
        rtol=0.0,
        atol=1e-15,
    ):
        raise RuntimeError("Frozen threshold column changed.")

    p = score["frozen_safety_probability"].to_numpy(dtype=float)
    flags = score["frozen_risk_flag"].to_numpy(dtype=int)
    if not np.isfinite(p).all():
        raise RuntimeError("Non-finite frozen safety probability.")
    if not np.array_equal(
        (p >= FROZEN_THRESHOLD).astype(int),
        flags,
    ):
        raise RuntimeError("Frozen risk flags do not match frozen threshold.")

    key = ["sample_id", "model_state_id"]
    joined = outcomes.merge(
        score[required],
        on=key,
        how="left",
        validate="one_to_one",
        suffixes=("_outcome", "_score"),
    )

    if len(joined) != EXPECTED_MODEL_CASES:
        raise RuntimeError("Score join changed row count.")
    if joined["frozen_safety_probability"].isna().any():
        raise RuntimeError("Missing frozen safety score after join.")

    checks = [
        (
            joined["center_outcome"].astype(str)
            == joined["center_score"].astype(str)
        ),
        (
            joined["model_family_outcome"].astype(str)
            == joined["model_family_score"].astype(str)
        ),
        (
            joined["training_seed_outcome"].astype(int)
            == joined["training_seed_score"].astype(int)
        ),
        (
            joined["checkpoint_sha256_outcome"].astype(str)
            == joined["checkpoint_sha256_score"].astype(str)
        ),
    ]
    if not all(x.all() for x in checks):
        raise RuntimeError("Outcome/score identity metadata mismatch.")

    joined["center"] = joined["center_outcome"].astype(str)
    joined["model_family"] = (
        joined["model_family_outcome"].astype(str)
    )
    joined["training_seed"] = (
        joined["training_seed_outcome"].astype(int)
    )
    joined["checkpoint_sha256"] = (
        joined["checkpoint_sha256_outcome"].astype(str)
    )

    drop = [
        "center_outcome",
        "center_score",
        "model_family_outcome",
        "model_family_score",
        "training_seed_outcome",
        "training_seed_score",
        "checkpoint_sha256_outcome",
        "checkpoint_sha256_score",
    ]
    return joined.drop(columns=drop)


def adjusted_ppv_at_prevalence(
    recall: float,
    fpr: float,
    prevalence: float = ASSUMED_PREVALENCE,
) -> float:
    denom = prevalence * recall + (1.0 - prevalence) * fpr
    if denom <= 0:
        return math.nan
    return float(prevalence * recall / denom)


def oracle_fpr_at_recall90(y, p):
    y = np.asarray(y, dtype=np.int8)
    p = np.asarray(p, dtype=np.float64)
    positives = int(y.sum())
    negatives = int(len(y) - positives)

    if positives == 0 or negatives == 0:
        return math.nan, math.nan, math.nan

    positive_scores = np.sort(p[y == 1])[::-1]
    need = int(math.ceil(0.90 * positives))
    need = min(max(need, 1), positives)
    threshold = float(positive_scores[need - 1])

    pred = p >= threshold
    tp = int(np.count_nonzero(pred & (y == 1)))
    fp = int(np.count_nonzero(pred & (y == 0)))

    recall = tp / positives
    fpr = fp / negatives

    return float(fpr), float(recall), threshold


def binary_metrics(df, sample_weight=None):
    y = df["harm_label"].to_numpy(dtype=np.int8)
    p = df["frozen_safety_probability"].to_numpy(dtype=np.float64)
    pred = p >= FROZEN_THRESHOLD

    if sample_weight is None:
        w = np.ones(len(df), dtype=np.float64)
    else:
        w = np.asarray(sample_weight, dtype=np.float64)
        if w.shape != (len(df),):
            raise RuntimeError("sample_weight shape mismatch.")

    pos = float(np.sum(w * (y == 1)))
    neg = float(np.sum(w * (y == 0)))

    if pos <= 0 or neg <= 0:
        return None

    tp = float(np.sum(w * ((y == 1) & pred)))
    fn = float(np.sum(w * ((y == 1) & (~pred))))
    fp = float(np.sum(w * ((y == 0) & pred)))
    tn = float(np.sum(w * ((y == 0) & (~pred))))

    recall = tp / (tp + fn)
    fpr = fp / (fp + tn)
    specificity = tn / (tn + fp)
    empirical_ppv = tp / (tp + fp) if (tp + fp) > 0 else math.nan
    npv = tn / (tn + fn) if (tn + fn) > 0 else math.nan

    auroc = float(
        roc_auc_score(
            y,
            p,
            sample_weight=w,
        )
    )
    auprc = float(
        average_precision_score(
            y,
            p,
            sample_weight=w,
        )
    )

    return {
        "rows_weighted": float(w.sum()),
        "harm_weighted": pos,
        "nonharm_weighted": neg,
        "harm_prevalence": pos / (pos + neg),
        "auroc": auroc,
        "auprc": auprc,
        "recall": float(recall),
        "fpr": float(fpr),
        "specificity": float(specificity),
        "empirical_ppv": float(empirical_ppv),
        "npv": float(npv),
        "ppv_at_1pct": adjusted_ppv_at_prevalence(
            recall,
            fpr,
            ASSUMED_PREVALENCE,
        ),
        "tp_weighted": tp,
        "fp_weighted": fp,
        "tn_weighted": tn,
        "fn_weighted": fn,
    }


def point_metric_row(name, level, df):
    m = binary_metrics(df)
    if m is None:
        return {
            "level": level,
            "group": name,
            "rows": len(df),
            "harm": int(df["harm_label"].sum()),
            "nonharm": int(
                len(df) - df["harm_label"].sum()
            ),
            "class_degenerate": 1,
        }

    oracle_fpr, oracle_recall, oracle_thr = oracle_fpr_at_recall90(
        df["harm_label"].to_numpy(dtype=int),
        df["frozen_safety_probability"].to_numpy(dtype=float),
    )

    return {
        "level": level,
        "group": name,
        "rows": len(df),
        "cases": df["sample_id"].nunique(),
        "harm": int(df["harm_label"].sum()),
        "nonharm": int(len(df) - df["harm_label"].sum()),
        "class_degenerate": 0,
        "frozen_threshold": FROZEN_THRESHOLD,
        "harm_prevalence": m["harm_prevalence"],
        "auroc": m["auroc"],
        "auprc": m["auprc"],
        "recall": m["recall"],
        "fpr": m["fpr"],
        "specificity": m["specificity"],
        "empirical_ppv": m["empirical_ppv"],
        "npv": m["npv"],
        "ppv_at_1pct": m["ppv_at_1pct"],
        "oracle_target_fpr_at_r90_diagnostic": oracle_fpr,
        "oracle_target_recall_diagnostic": oracle_recall,
        "oracle_target_threshold_diagnostic": oracle_thr,
    }


def macro_family_point(family_rows):
    numeric = [
        r for r in family_rows
        if int(r.get("class_degenerate", 0)) == 0
    ]
    if len(numeric) != 3:
        raise RuntimeError(
            "All three model families must be non-degenerate."
        )

    keys = [
        "harm_prevalence",
        "auroc",
        "auprc",
        "recall",
        "fpr",
        "specificity",
        "empirical_ppv",
        "npv",
        "ppv_at_1pct",
        "oracle_target_fpr_at_r90_diagnostic",
        "oracle_target_recall_diagnostic",
    ]
    row = {
        "level": "PRIMARY_MACRO_FAMILY",
        "group": "MACRO_3_FAMILIES",
        "families": 3,
        "frozen_threshold": FROZEN_THRESHOLD,
    }
    for k in keys:
        row[k] = float(np.mean([float(r[k]) for r in numeric]))
    return row


def build_case_index(df):
    by_case = {}
    centers = {}

    for sid, g in df.groupby("sample_id", sort=True):
        if len(g) != 9:
            raise RuntimeError(
                f"{sid}: rows={len(g)} expected=9"
            )
        fam_counts = g["model_family"].value_counts().to_dict()
        if set(fam_counts) != set(FAMILIES):
            raise RuntimeError(
                f"{sid}: family set mismatch {fam_counts}"
            )
        if any(v != 3 for v in fam_counts.values()):
            raise RuntimeError(
                f"{sid}: expected three states/family {fam_counts}"
            )

        by_case[sid] = g.index.to_numpy(dtype=int)
        cvals = g["center"].astype(str).unique()
        if len(cvals) != 1:
            raise RuntimeError(f"{sid}: center mismatch.")
        centers[sid] = cvals[0]

    if len(by_case) != EXPECTED_CASES:
        raise RuntimeError("External case index !=1532.")

    return by_case, centers


def center_stratified_case_weights(df, rng, by_case, centers):
    case_counts = defaultdict(int)

    for center in CENTERS:
        sids = sorted(
            sid for sid, c in centers.items()
            if c == center
        )
        n = len(sids)
        if n == 0:
            raise RuntimeError(f"Center {center} has no cases.")
        draw = rng.choice(
            np.asarray(sids, dtype=object),
            size=n,
            replace=True,
        )
        for sid in draw:
            case_counts[str(sid)] += 1

    w = np.zeros(len(df), dtype=np.float64)
    for sid, count in case_counts.items():
        w[by_case[sid]] = float(count)

    if int(w.sum()) != EXPECTED_MODEL_CASES:
        raise RuntimeError(
            f"Bootstrap weighted rows={w.sum()} "
            f"expected={EXPECTED_MODEL_CASES}"
        )
    return w


def bootstrap_primary(df):
    by_case, centers = build_case_index(df)
    rng = np.random.RandomState(BOOTSTRAP_SEED)

    family_masks = {
        f: (
            df["model_family"].astype(str).to_numpy() == f
        )
        for f in FAMILIES
    }

    metric_names = [
        "auroc",
        "auprc",
        "recall",
        "fpr",
        "empirical_ppv",
        "ppv_at_1pct",
    ]

    pooled_store = {k: [] for k in metric_names}
    macro_store = {k: [] for k in metric_names}
    family_store = {
        f: {k: [] for k in metric_names}
        for f in FAMILIES
    }

    valid = 0

    for _ in tqdm(
        range(BOOTSTRAP_RESAMPLES),
        desc="R10L3C case-clustered center-stratified bootstrap",
        unit="rep",
        dynamic_ncols=True,
    ):
        w = center_stratified_case_weights(
            df,
            rng,
            by_case,
            centers,
        )

        pooled = binary_metrics(df, sample_weight=w)
        if pooled is None:
            continue

        family_metrics = {}
        bad = False
        for fam in FAMILIES:
            mask = family_masks[fam]
            fm = binary_metrics(
                df.loc[mask],
                sample_weight=w[mask],
            )
            if fm is None:
                bad = True
                break
            family_metrics[fam] = fm

        if bad:
            continue

        valid += 1
        for k in metric_names:
            pooled_store[k].append(float(pooled[k]))
            vals = [
                float(family_metrics[f][k])
                for f in FAMILIES
            ]
            macro_store[k].append(float(np.mean(vals)))
            for f in FAMILIES:
                family_store[f][k].append(
                    float(family_metrics[f][k])
                )

    if valid == 0:
        raise RuntimeError("No valid bootstrap resamples.")

    def ci_rows(level, group, store):
        rows = []
        for k, vals in store.items():
            arr = np.asarray(vals, dtype=float)
            rows.append({
                "level": level,
                "group": group,
                "metric": k,
                "bootstrap_resamples_requested": BOOTSTRAP_RESAMPLES,
                "bootstrap_resamples_valid": len(arr),
                "bootstrap_seed": BOOTSTRAP_SEED,
                "cluster": "sample_id",
                "center_stratified": "YES",
                "ci_low_2p5": float(np.quantile(arr, 0.025)),
                "bootstrap_median": float(np.quantile(arr, 0.50)),
                "ci_high_97p5": float(np.quantile(arr, 0.975)),
            })
        return rows

    rows = []
    rows.extend(
        ci_rows(
            "PRIMARY_MACRO_FAMILY",
            "MACRO_3_FAMILIES",
            macro_store,
        )
    )
    rows.extend(
        ci_rows(
            "SECONDARY_POOLED",
            "ALL_ROWS",
            pooled_store,
        )
    )
    for fam in FAMILIES:
        rows.extend(
            ci_rows(
                "SECONDARY_FAMILY",
                fam,
                family_store[fam],
            )
        )

    return rows, valid


def main_evaluation(args):
    print("===== R10L3C POLYPGEN GT REVEAL + PROSPECTIVE EXTERNAL EVALUATION FIX1 =====")
    print("STATUS: FIRST GT REVEAL AFTER R10L3A + R10L3B LOCKS")
    print("MODEL INFERENCE: NONE")
    print("TTA: NONE")
    print("SAFETY MODEL FIT/REFIT: NONE")
    print("SCORE ORIENTATION CHANGE: NO")
    print("THRESHOLD CHANGE: NO")
    print("TARGET CALIBRATION: NO")
    print("TARGET FEATURE SELECTION: NO")
    print("TARGET CASE/CENTER/STATE SELECTION: NO")
    print("FROZEN SAFETY THRESHOLD:", FROZEN_THRESHOLD)
    print("FROZEN OUTCOME HARM THRESHOLD:", HARM_THRESHOLD)
    print("FROZEN OUTCOME BENEFIT THRESHOLD:", BENEFIT_THRESHOLD)
    print("BOOTSTRAP:", BOOTSTRAP_RESAMPLES)
    print("BOOTSTRAP CLUSTER: sample_id")
    print("BOOTSTRAP CENTER STRATIFIED: YES")
    print()

    verify_required_paths()
    upstream = verify_upstream_locks()
    verify_development_outcome_rule()

    print("\n===== FIRST POLYPGEN GT ACCESS AUTHORIZED =====")
    gt_df = load_locked_gt_manifest()
    gt_masks, gt_audit = reveal_gt_masks(gt_df)

    print("GT cases decoded:", len(gt_masks))
    print(
        "GT empty after 352x352 decode:",
        int(
            sum(
                int(m.sum()) == 0
                for m in gt_masks.values()
            )
        ),
    )

    print("\n===== FROZEN SOURCE/A1 OUTCOME CONSTRUCTION =====")
    outcomes = build_external_outcomes(gt_masks)

    outcome_counts = (
        outcomes["adaptation_outcome"]
        .value_counts()
        .to_dict()
    )
    harm_n = int(outcomes["harm_label"].sum())
    benefit_n = int(outcomes["benefit_label"].sum())

    print("Outcome counts:", outcome_counts)
    print("HARM/NON-HARM:", harm_n, "/", len(outcomes) - harm_n)
    print("BENEFIT:", benefit_n)
    print(
        "SOURCE Dice mean/median:",
        float(outcomes["source_dice"].mean()),
        float(outcomes["source_dice"].median()),
    )
    print(
        "A1 Dice mean/median:",
        float(outcomes["a1_dice"].mean()),
        float(outcomes["a1_dice"].median()),
    )
    print(
        "DeltaDice mean/median:",
        float(outcomes["delta_dice"].mean()),
        float(outcomes["delta_dice"].median()),
    )

    print("\n===== JOIN ALREADY-LOCKED SAFETY SCORES =====")
    panel = load_and_join_locked_scores(outcomes)

    print("Evaluation rows:", len(panel))
    print("Unique cases:", panel["sample_id"].nunique())
    print("Unique states:", panel["model_state_id"].nunique())
    print(
        "Frozen score min/median/max:",
        float(panel["frozen_safety_probability"].min()),
        float(panel["frozen_safety_probability"].median()),
        float(panel["frozen_safety_probability"].max()),
    )
    print(
        "Frozen risk flags:",
        int(panel["frozen_risk_flag"].sum()),
        "/",
        len(panel),
    )

    point_rows = []

    pooled = point_metric_row(
        "ALL_ROWS",
        "SECONDARY_POOLED",
        panel,
    )
    point_rows.append(pooled)

    family_rows = []
    for fam in FAMILIES:
        r = point_metric_row(
            fam,
            "SECONDARY_FAMILY",
            panel[panel["model_family"] == fam],
        )
        family_rows.append(r)
        point_rows.append(r)

    macro = macro_family_point(family_rows)
    point_rows.insert(0, macro)

    for center in CENTERS:
        point_rows.append(
            point_metric_row(
                center,
                "SECONDARY_CENTER",
                panel[panel["center"] == center],
            )
        )

    for state_id in sorted(panel["model_state_id"].unique()):
        point_rows.append(
            point_metric_row(
                state_id,
                "SECONDARY_STATE",
                panel[panel["model_state_id"] == state_id],
            )
        )

    print("\n===== PRIMARY EXTERNAL RESULT: MACRO OVER 3 FAMILIES =====")
    for k in [
        "harm_prevalence",
        "auroc",
        "auprc",
        "recall",
        "fpr",
        "empirical_ppv",
        "ppv_at_1pct",
        "oracle_target_fpr_at_r90_diagnostic",
    ]:
        print(f"{k}: {macro[k]:.9f}")

    print("\n===== FAMILY POINT RESULTS =====")
    for r in family_rows:
        print(
            r["group"],
            f"AUROC={r['auroc']:.6f}",
            f"AUPRC={r['auprc']:.6f}",
            f"Recall={r['recall']:.6f}",
            f"FPR={r['fpr']:.6f}",
            f"PPV1%={r['ppv_at_1pct']:.6f}",
            f"OracleFPR90={r['oracle_target_fpr_at_r90_diagnostic']:.6f}",
        )

    print("\n===== CONFIRMATORY CASE-CLUSTERED BOOTSTRAP =====")
    bootstrap_rows, valid_bootstrap = bootstrap_primary(panel)

    macro_ci = pd.DataFrame(bootstrap_rows)
    macro_ci = macro_ci[
        macro_ci["level"] == "PRIMARY_MACRO_FAMILY"
    ]

    print("Valid bootstrap resamples:", valid_bootstrap)
    for r in macro_ci.itertuples(index=False):
        print(
            r.metric,
            f"CI95=[{r.ci_low_2p5:.6f}, {r.ci_high_97p5:.6f}]",
        )

    out = args.output_dir
    if out.exists():
        raise FileExistsError(
            f"R10L3C final output already exists: {out}"
        )

    build = Path(str(out) + "__building")
    if build.exists():
        raise FileExistsError(
            f"R10L3C partial build already exists: {build}"
        )
    build.mkdir(parents=True, exist_ok=False)

    gt_audit_path = build / "R10L3C_polypgen_gt_decode_audit.csv"
    pd.DataFrame(gt_audit).to_csv(
        gt_audit_path,
        index=False,
    )

    outcomes_path = build / "R10L3C_model_case_outcomes.csv"
    outcomes.to_csv(outcomes_path, index=False)

    panel_path = build / "R10L3C_LOCKED_SCORE_GT_EVALUATION_PANEL.csv"
    panel.to_csv(panel_path, index=False)

    point_path = build / "R10L3C_external_point_metrics.csv"
    pd.DataFrame(point_rows).to_csv(point_path, index=False)

    boot_path = build / "R10L3C_clustered_bootstrap_ci.csv"
    pd.DataFrame(bootstrap_rows).to_csv(boot_path, index=False)

    outcome_summary = (
        panel.groupby(
            ["model_family", "adaptation_outcome"],
            sort=True,
        )
        .size()
        .reset_index(name="rows")
    )
    outcome_summary.to_csv(
        build / "R10L3C_family_outcome_counts.csv",
        index=False,
    )

    center_outcome_summary = (
        panel.groupby(
            ["center", "adaptation_outcome"],
            sort=True,
        )
        .size()
        .reset_index(name="rows")
    )
    center_outcome_summary.to_csv(
        build / "R10L3C_center_outcome_counts.csv",
        index=False,
    )

    artifacts = {}
    for p in [
        gt_audit_path,
        outcomes_path,
        panel_path,
        point_path,
        boot_path,
        build / "R10L3C_family_outcome_counts.csv",
        build / "R10L3C_center_outcome_counts.csv",
    ]:
        artifacts[p.name] = {
            "sha256": sha256_file(p),
            "bytes": p.stat().st_size,
        }

    info_boundary = {
        "polypgen_gt_path_read_in_r10l3c": True,
        "polypgen_gt_pixels_decoded_in_r10l3c": True,
        "model_inference_run_in_r10l3c": False,
        "tta_run_in_r10l3c": False,
        "safety_model_fit_or_refit": False,
        "score_orientation_changed": False,
        "threshold_changed": False,
        "target_calibration": False,
        "target_feature_selection": False,
        "target_case_selection": False,
        "target_center_selection": False,
        "target_state_selection": False,
        "oracle_r90_threshold_deployed": False,
        "oracle_r90_metric_role": "DIAGNOSTIC_ONLY",
    }

    lock = {
        "status": "COMPLETE",
        "decision": DECISION_COMPLETE,
        "script_version": VERSION,
        "build": BUILD,
        "target_dataset": "PolypGen2021_MultiCenterData_v3",
        "target_cases": EXPECTED_CASES,
        "model_states": EXPECTED_STATES,
        "model_case_rows": EXPECTED_MODEL_CASES,
        "outcome_definition": {
            "source_metric": "binary_dice",
            "a1_metric": "binary_dice",
            "delta": "a1_dice - source_dice",
            "harm": "delta_dice <= -0.02",
            "benefit": "delta_dice >= +0.02",
            "neutral": "-0.02 < delta_dice < +0.02",
        },
        "polypgen_gt_decode": {
            "native_decode": "PIL convert L",
            "binary_threshold": "gray > 127",
            "resize": "nearest neighbor to 352x352",
            "post_resize_binary": ">0",
            "adaptive_threshold": False,
            "morphological_repair": False,
        },
        "frozen_safety_evaluation": {
            "score": "R10L3B frozen_safety_probability",
            "score_orientation": "higher = higher HARM risk",
            "operating_threshold": FROZEN_THRESHOLD,
            "threshold_changed": False,
            "assumed_prevalence_for_ppv_1pct": ASSUMED_PREVALENCE,
            "primary_level": "macro over 3 model families",
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_valid": valid_bootstrap,
            "bootstrap_cluster": "sample_id",
            "bootstrap_center_stratified": True,
            "oracle_target_fpr_at_r90": "diagnostic only; not deployed",
        },
        "outcome_counts": {
            k: int(v)
            for k, v in outcome_counts.items()
        },
        "harm_rows": harm_n,
        "nonharm_rows": int(len(panel) - harm_n),
        "benefit_rows": benefit_n,
        "primary_macro_point": {
            k: float(v)
            for k, v in macro.items()
            if isinstance(v, (int, float, np.integer, np.floating))
        },
        "information_boundary": info_boundary,
        "upstream_hashes": upstream,
        "artifacts": artifacts,
    }

    lock_path = (
        build
        / "R10L3C_POLYPGEN_PROSPECTIVE_EXTERNAL_EVALUATION_LOCK.json"
    )
    write_json(lock_path, lock)

    run_log = "\n".join([
        "===== R10L3C POLYPGEN PROSPECTIVE EXTERNAL EVALUATION FIX1 =====",
        f"Script version: {VERSION}",
        f"Build: {BUILD}",
        "",
        "Frozen evidence order:",
        "  R10L3A SOURCE/A1 prediction lock=PASS",
        "  R10L3B safety-score lock=PASS",
        "  first PolypGen GT reveal=R10L3C",
        "",
        "External panel:",
        f"  cases={EXPECTED_CASES}",
        f"  states={EXPECTED_STATES}",
        f"  model-case rows={EXPECTED_MODEL_CASES}",
        f"  HARM={harm_n}",
        f"  NON-HARM={len(panel)-harm_n}",
        f"  BENEFIT={benefit_n}",
        "",
        "Frozen outcome:",
        "  delta_dice=A1 Dice - SOURCE Dice",
        "  HARM=delta<=-0.02",
        "  BENEFIT=delta>=+0.02",
        "  NEUTRAL=otherwise",
        "",
        "Frozen safety operating point:",
        f"  threshold={FROZEN_THRESHOLD:.15f}",
        "  recalibration=NO",
        "  score reversal=NO",
        "  feature selection=NO",
        "  target subset selection=NO",
        "",
        "Primary macro-over-family external metrics:",
        f"  AUROC={macro['auroc']:.9f}",
        f"  AUPRC={macro['auprc']:.9f}",
        f"  Recall={macro['recall']:.9f}",
        f"  FPR={macro['fpr']:.9f}",
        f"  empirical PPV={macro['empirical_ppv']:.9f}",
        f"  PPV@1%={macro['ppv_at_1pct']:.9f}",
        f"  oracle target FPR@R90 diagnostic="
        f"{macro['oracle_target_fpr_at_r90_diagnostic']:.9f}",
        "",
        f"Bootstrap resamples valid={valid_bootstrap}/{BOOTSTRAP_RESAMPLES}",
        "Bootstrap cluster=sample_id",
        "Bootstrap center stratified=YES",
        "",
        f"Decision={DECISION_COMPLETE}",
        "PASS",
    ]) + "\n"

    (build / "run_log.txt").write_text(
        run_log,
        encoding="utf-8",
    )

    lock_sha = sha256_file(lock_path)
    build.rename(out)

    print()
    print((out / "run_log.txt").read_text(encoding="utf-8"))
    print(
        "R10L3C LOCK:",
        out
        / "R10L3C_POLYPGEN_PROSPECTIVE_EXTERNAL_EVALUATION_LOCK.json",
    )
    print("R10L3C LOCK SHA256:", lock_sha)
    print("PASS")


def self_test():
    vals = np.asarray(
        [-0.021, -0.020, -0.019, 0.0, 0.019, 0.020, 0.021]
    )
    labels = [outcome_label(float(x)) for x in vals]
    assert labels == [
        "HARM",
        "HARM",
        "NEUTRAL",
        "NEUTRAL",
        "NEUTRAL",
        "BENEFIT",
        "BENEFIT",
    ]

    z = np.zeros((MASK_H, MASK_W), dtype=np.uint8)
    o = np.ones((MASK_H, MASK_W), dtype=np.uint8)
    assert binary_dice(z, z) == 1.0
    assert binary_dice(o, o) == 1.0
    assert binary_dice(z, o) == 0.0

    val = adjusted_ppv_at_prevalence(0.90, 0.10, 0.01)
    expected = 0.009 / (0.009 + 0.099)
    assert abs(val - expected) < 1e-15

    assert EXPECTED_CASES * EXPECTED_STATES == EXPECTED_MODEL_CASES
    assert abs(FROZEN_THRESHOLD - 0.300584763193734) <= 1e-15

    print("OUTCOME_BOUNDARY_TEST_PASS")
    print("BINARY_DICE_TEST_PASS")
    print("PPV1PCT_TEST_PASS")
    print("CARDINALITY_TEST_PASS")
    print("FROZEN_THRESHOLD_TEST_PASS")
    print("SELF_TEST_PASS")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
    )
    ap.add_argument("--self-test", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()

    if args.self_test:
        self_test()
        return 0

    try:
        main_evaluation(args)
    except Exception:
        traceback.print_exc()
        raise

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
