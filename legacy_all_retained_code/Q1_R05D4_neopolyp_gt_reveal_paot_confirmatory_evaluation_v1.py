#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Q1-R05D4 — First NeoPolyp GT reveal + confirmatory PAOT evaluation.

This script:
- verifies frozen D1/D2/D3 artifacts;
- decodes the 1000 target GT masks for the first time;
- computes frozen SOURCE/A1 Dice and DeltaDice;
- derives HARM/NEUTRAL/BENEFIT;
- evaluates already-locked D2 probabilities;
- applies already-frozen MRZ19 confirmatory GO criteria.

It does NOT run models/TTA, fit predictors, calibrate probabilities, select
thresholds, or exclude target cases based on outcomes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm


VERSION = "2026-08-19-Q1-R05D4-v1"
BUILD = "Q1_R05D4_NEOPOLYP_GT_REVEAL_PAOT_CONFIRMATORY_EVALUATION"

ROOT = Path(r"F:\MEDSEG_SAFETTA")

PROTOCOL = (
    ROOT / "docs"
    / "Q1_R05D4_neopolyp_gt_reveal_paot_confirmatory_evaluation_preregistered_protocol_v1.md"
)
EXPECTED_PROTOCOL_SHA256 = "f18728ca9b67b661a9a3a956fb84cea425781981a441ce356b49374253ab50de"

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

R05D3_DIR = (
    ROOT / "outputs"
    / "Q1_R05D3_neopolyp_source_a1_prediction_lock_v1_fix1"
)
R05D3_LOCK = R05D3_DIR / "Q1_R05D3_NEOPOLYP_SOURCE_A1_PREDICTION_LOCK.json"
EXPECTED_R05D3_LOCK_SHA256 = (
    "3da08e5ed0780a00788a297b1127d9eb460224326b11e9e30eefce2b3fb48c9a"
)
EXPECTED_R05D3_DECISION = "NEOPOLYP_SOURCE_A1_PREDICTIONS_LOCKED"

R05D3_SCRIPT = (
    ROOT / "code"
    / "Q1_R05D3_neopolyp_source_a1_prediction_lock_v1_fix1.py"
)
EXPECTED_R05D3_SCRIPT_SHA256 = (
    "df22d6f3054257743d1f12d9e757323ea9f4a052b7642eb4ab610da46ff21251"
)

OUTPUT_DIR = (
    ROOT / "outputs"
    / "Q1_R05D4_neopolyp_gt_reveal_paot_confirmatory_evaluation_v1"
)

EXPECTED_CASES = 1000
EXPECTED_STATES = 9
EXPECTED_MODEL_CASES = 9000

MASK_H = 352
MASK_W = 352
MASK_PIXELS = MASK_H * MASK_W
PACKED_BYTES = (MASK_PIXELS + 7) // 8
BITORDER = "little"

FAMILIES = ("PraNet", "DeepLabV3-R50", "SegFormer-B0")
REPRESENTATIONS = ("MRZ19", "RAW19")
TASKS = ("HARM", "BENEFIT")

BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_BASE_SEED = 20260819

HARM_THRESHOLD = -0.02
BENEFIT_THRESHOLD = 0.02

PROB_FIELDS = {
    ("MRZ19", "HARM"): "paot_mrz19_harm_probability",
    ("MRZ19", "BENEFIT"): "paot_mrz19_benefit_probability",
    ("RAW19", "HARM"): "paot_raw19_harm_probability",
    ("RAW19", "BENEFIT"): "paot_raw19_benefit_probability",
}

PRIMARY_CRITERIA = {
    "primary_representation": "MRZ19",
    "median_architecture_harm_auroc_min": 0.70,
    "minimum_architecture_harm_auroc_min": 0.60,
    "median_architecture_benefit_auroc_min": 0.75,
    "minimum_architecture_benefit_auroc_min": 0.65,
    "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
    "bootstrap_cluster": "target_sample_id",
    "bootstrap_base_seed": BOOTSTRAP_BASE_SEED,
    "ci_used_as_go_gate": False,
}

DECISION_PASS = "NEOPOLYP_PAOT_CONFIRMATION_PASS"
DECISION_FAIL = "NEOPOLYP_PAOT_CONFIRMATION_FAIL"
DECISION_INCONCLUSIVE = "NEOPOLYP_PAOT_CONFIRMATION_INCONCLUSIVE"


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
        writer = csv.DictWriter(f, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


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


def validate_upstream():
    validate_sha(PROTOCOL, EXPECTED_PROTOCOL_SHA256, "D4 protocol")
    validate_sha(R05D1_LOCK, EXPECTED_R05D1_LOCK_SHA256, "D1 lock")
    validate_sha(R05D2_LOCK, EXPECTED_R05D2_LOCK_SHA256, "D2 lock")
    validate_sha(R05D3_LOCK, EXPECTED_R05D3_LOCK_SHA256, "D3 lock")
    validate_sha(R05D3_SCRIPT, EXPECTED_R05D3_SCRIPT_SHA256, "D3 script")

    d1 = json.loads(R05D1_LOCK.read_text(encoding="utf-8"))
    d2 = json.loads(R05D2_LOCK.read_text(encoding="utf-8"))
    d3 = json.loads(R05D3_LOCK.read_text(encoding="utf-8"))

    if d1.get("decision") != EXPECTED_R05D1_DECISION:
        raise RuntimeError(f"Unexpected D1 decision: {d1.get('decision')}")
    if int(d1.get("eligible_confirmatory_cases", -1)) != EXPECTED_CASES:
        raise RuntimeError("D1 target count changed.")
    if int(d1.get("exact_overlap_exclusions", -1)) != 0:
        raise RuntimeError("D1 overlap exclusions changed.")
    if bool(d1.get("target_gt_pixels_decoded", True)):
        raise RuntimeError("D1 reports target GT decoding.")

    if d2.get("decision") != EXPECTED_R05D2_DECISION:
        raise RuntimeError(f"Unexpected D2 decision: {d2.get('decision')}")
    if int(d2.get("target_cases", -1)) != EXPECTED_CASES:
        raise RuntimeError("D2 target count changed.")
    if int(d2.get("model_states", -1)) != EXPECTED_STATES:
        raise RuntimeError("D2 model-state count changed.")
    if int(d2.get("model_case_rows", -1)) != EXPECTED_MODEL_CASES:
        raise RuntimeError("D2 model-case count changed.")
    if bool(d2.get("target_gt_pixels_decoded", True)):
        raise RuntimeError("D2 reports target GT decoding.")
    if bool(d2.get("target_outcomes_revealed", True)):
        raise RuntimeError("D2 reports target outcome reveal.")

    d2_table = artifact_path_from_lock(
        R05D2_DIR, d2, "target_paot_probabilities"
    )
    validate_sha(d2_table, EXPECTED_D2_TABLE_SHA256, "D2 probability table")

    criteria = d2.get("confirmatory_criteria")
    if not isinstance(criteria, dict):
        raise RuntimeError("D2 lock missing confirmatory criteria.")

    expected_criteria = {
        "primary_representation": "MRZ19",
        "median_architecture_harm_auroc_min": 0.70,
        "minimum_architecture_harm_auroc_min": 0.60,
        "median_architecture_benefit_auroc_min": 0.75,
        "minimum_architecture_benefit_auroc_min": 0.65,
        "bootstrap_resamples": 10000,
        "ci_used_as_go_gate": False,
    }
    for key, expected in expected_criteria.items():
        if criteria.get(key) != expected:
            raise RuntimeError(
                f"D2 confirmatory criterion changed: "
                f"{key}={criteria.get(key)} expected={expected}"
            )

    if d3.get("decision") != EXPECTED_R05D3_DECISION:
        raise RuntimeError(f"Unexpected D3 decision: {d3.get('decision')}")
    if int(d3.get("target_cases", -1)) != EXPECTED_CASES:
        raise RuntimeError("D3 target count changed.")
    if int(d3.get("model_states", -1)) != EXPECTED_STATES:
        raise RuntimeError("D3 model-state count changed.")
    if int(d3.get("model_case_rows", -1)) != EXPECTED_MODEL_CASES:
        raise RuntimeError("D3 model-case count changed.")
    if int(d3.get("source_prediction_masks", -1)) != EXPECTED_MODEL_CASES:
        raise RuntimeError("D3 SOURCE-mask count changed.")
    if int(d3.get("a1_prediction_masks", -1)) != EXPECTED_MODEL_CASES:
        raise RuntimeError("D3 A1-mask count changed.")
    if not bool(d3.get("all_model_cases_received_a1", False)):
        raise RuntimeError("D3 does not confirm all model-cases received A1.")
    if bool(d3.get("d2_probability_values_read", True)):
        raise RuntimeError("D3 reports D2 probability values were read.")
    if bool(d3.get("paot_gating_used", True)):
        raise RuntimeError("D3 reports PAOT gating.")
    if bool(d3.get("target_gt_pixels_decoded", True)):
        raise RuntimeError("D3 reports GT decoding.")
    if bool(d3.get("target_outcomes_revealed", True)):
        raise RuntimeError("D3 reports target outcome reveal.")

    d1_manifest = artifact_path_from_lock(
        R05D1_DIR, d1, "frozen_confirmatory_manifest"
    )
    d3_gt_rule = artifact_path_from_lock(
        R05D3_DIR, d3, "frozen_d4_gt_and_metric_rule"
    )

    return d1, d2, d3, d1_manifest, d2_table, d3_gt_rule


def validate_gt_rule(path: Path):
    rule = json.loads(path.read_text(encoding="utf-8"))

    if rule.get("evaluation_task") != "binary_polyp_segmentation":
        raise RuntimeError("Frozen evaluation task changed.")
    if rule.get("future_decode_rule") != "nearest_RGB_prototype_euclidean":
        raise RuntimeError("Frozen GT decode rule changed.")

    if rule.get("binary_mapping") != {
        "background": 0,
        "non_neoplastic_polyp": 1,
        "neoplastic_polyp": 1,
    }:
        raise RuntimeError("Frozen binary GT mapping changed.")

    classes = rule.get("original_gt_classes", {})
    expected_rgb = {
        "0": [0, 0, 0],
        "1": [0, 255, 0],
        "2": [255, 0, 0],
    }
    for key, rgb in expected_rgb.items():
        if classes.get(key, {}).get("rgb") != rgb:
            raise RuntimeError(f"Frozen RGB prototype changed: class={key}")

    resize = rule.get("future_gt_resize", {})
    if (
        int(resize.get("height", -1)) != MASK_H
        or int(resize.get("width", -1)) != MASK_W
        or resize.get("interpolation") != "nearest"
    ):
        raise RuntimeError("Frozen GT resize rule changed.")

    if float(rule.get("harm_threshold_delta_dice")) != HARM_THRESHOLD:
        raise RuntimeError("Frozen HARM threshold changed.")
    if float(rule.get("benefit_threshold_delta_dice")) != BENEFIT_THRESHOLD:
        raise RuntimeError("Frozen BENEFIT threshold changed.")

    return rule


def load_d1_manifest(path: Path):
    rows, fields = read_csv(path)
    required = {
        "sample_id",
        "image_path",
        "gt_path",
        "image_raw_sha256",
        "eligible_confirmatory",
        "exclusion_reason",
        "gt_pixels_decoded",
    }
    missing = sorted(required - set(fields))
    if missing:
        raise RuntimeError(f"D1 manifest missing columns: {missing}")
    if len(rows) != EXPECTED_CASES:
        raise RuntimeError(f"D1 manifest rows={len(rows)}")

    out = {}
    for row in rows:
        if int(row["eligible_confirmatory"]) != 1:
            raise RuntimeError("D4 cannot silently drop a D1-ineligible case.")
        if row["exclusion_reason"].strip():
            raise RuntimeError("Eligible D1 row has exclusion reason.")
        if int(row["gt_pixels_decoded"]) != 0:
            raise RuntimeError("D1 manifest reports prior GT decode.")
        sid = row["sample_id"]
        if sid in out:
            raise RuntimeError(f"Duplicate D1 sample ID: {sid}")
        if not Path(row["gt_path"]).is_file():
            raise FileNotFoundError(row["gt_path"])
        out[sid] = row

    if len(out) != EXPECTED_CASES:
        raise RuntimeError("D1 sample-id count mismatch.")
    return out


def nearest_rgb_classes(rgb: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgb, dtype=np.int32)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise RuntimeError(f"Unexpected GT RGB shape: {arr.shape}")

    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]

    d_bg = r * r + g * g + b * b
    d_green = r * r + (g - 255) ** 2 + b * b
    d_red = (r - 255) ** 2 + g * g + b * b

    distances = np.stack((d_bg, d_green, d_red), axis=2)
    return np.argmin(distances, axis=2).astype(np.uint8)


def reveal_gt(manifest):
    masks = {}
    rows = []
    resampling = getattr(Image, "Resampling", Image)

    for sid in tqdm(
        sorted(manifest),
        desc="Q1-R05D4 FIRST GT REVEAL",
        unit="gt",
        dynamic_ncols=True,
    ):
        gt_path = Path(manifest[sid]["gt_path"])
        gt_sha = sha256_file(gt_path)

        with Image.open(gt_path) as im:
            rgb_img = im.convert("RGB")
            native_w, native_h = rgb_img.size
            rgb = np.asarray(rgb_img, dtype=np.uint8)

        labels = nearest_rgb_classes(rgb)
        n_bg = int(np.count_nonzero(labels == 0))
        n_non = int(np.count_nonzero(labels == 1))
        n_neo = int(np.count_nonzero(labels == 2))

        if n_bg + n_non + n_neo != native_w * native_h:
            raise RuntimeError(f"GT class accounting failure: {sid}")

        binary_native = (labels != 0).astype(np.uint8)
        binary_img = Image.fromarray(binary_native * 255, mode="L")
        resized_img = binary_img.resize(
            (MASK_W, MASK_H),
            resample=resampling.NEAREST,
        )
        binary = (np.asarray(resized_img, dtype=np.uint8) > 0).astype(np.uint8)

        if binary.shape != (MASK_H, MASK_W):
            raise RuntimeError(f"GT resize shape failure: {sid}")
        if not np.isin(binary, (0, 1)).all():
            raise RuntimeError(f"GT binary failure: {sid}")

        masks[sid] = binary
        rows.append({
            "sample_id": sid,
            "gt_path": str(gt_path),
            "gt_raw_sha256": gt_sha,
            "native_width": native_w,
            "native_height": native_h,
            "native_background_pixels": n_bg,
            "native_non_neoplastic_pixels": n_non,
            "native_neoplastic_pixels": n_neo,
            "native_binary_foreground_pixels": n_non + n_neo,
            "resized_binary_foreground_pixels": int(binary.sum()),
            "gt_decode_rule": "nearest_RGB_prototype_euclidean",
        })

    if len(masks) != EXPECTED_CASES:
        raise RuntimeError("Not all 1000 GT files were decoded.")
    return masks, rows


def load_d2_probabilities(path: Path):
    rows, fields = read_csv(path)
    required = {
        "sample_id",
        "model_family",
        "model_state_id",
        "training_seed",
        "checkpoint_sha256",
        *PROB_FIELDS.values(),
    }
    missing = sorted(required - set(fields))
    if missing:
        raise RuntimeError(f"D2 table missing columns: {missing}")
    if len(rows) != EXPECTED_MODEL_CASES:
        raise RuntimeError(f"D2 probability rows={len(rows)}")

    out = {}
    for row in rows:
        key = (row["model_state_id"], row["sample_id"])
        if key in out:
            raise RuntimeError(f"Duplicate D2 model-case key: {key}")

        probs = {}
        for rep_task, field in PROB_FIELDS.items():
            p = float(row[field])
            if not math.isfinite(p) or not 0.0 <= p <= 1.0:
                raise RuntimeError(f"Invalid D2 probability {field}: {key}")
            probs[rep_task] = p

        out[key] = {
            "sample_id": row["sample_id"],
            "model_family": row["model_family"],
            "model_state_id": row["model_state_id"],
            "training_seed": int(row["training_seed"]),
            "checkpoint_sha256": row["checkpoint_sha256"],
            "probabilities": probs,
        }

    if len(out) != EXPECTED_MODEL_CASES:
        raise RuntimeError("D2 probability key count mismatch.")
    return out


def token_from_family_seed(family: str, seed: int) -> str:
    return (
        family.lower().replace("-", "_").replace(" ", "_")
        + f"_seed{seed}"
    )


def state_artifacts(d3: dict, family: str, seed: int):
    token = token_from_family_seed(family, seed)
    keys = {
        "npz": f"state_npz__{token}",
        "index": f"state_index__{token}",
        "lock": f"state_lock__{token}",
    }
    for key in keys.values():
        if key not in d3.get("artifacts", {}):
            raise RuntimeError(f"D3 lock missing artifact: {key}")
    return {
        name: artifact_path_from_lock(R05D3_DIR, d3, key)
        for name, key in keys.items()
    }


def unpack_mask(packed: np.ndarray) -> np.ndarray:
    p = np.asarray(packed, dtype=np.uint8)
    if p.shape != (PACKED_BYTES,):
        raise RuntimeError(f"Packed mask shape={p.shape}")
    bits = np.unpackbits(p, count=MASK_PIXELS, bitorder=BITORDER)
    return bits.reshape(MASK_H, MASK_W).astype(np.uint8, copy=False)


def binary_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    p = np.asarray(pred, dtype=np.uint8)
    g = np.asarray(gt, dtype=np.uint8)
    if p.shape != (MASK_H, MASK_W) or g.shape != (MASK_H, MASK_W):
        raise RuntimeError("Dice mask shape mismatch.")

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


def build_outcomes(gt_masks, d2_probs, d3):
    rows = []
    seen_keys = set()
    state_meta = []

    for family, seeds in (
        ("PraNet", (20260817, 20260818, 20260819)),
        ("DeepLabV3-R50", (20260817, 20260818, 20260819)),
        ("SegFormer-B0", (20260820, 20260821, 20260822)),
    ):
        for seed in seeds:
            paths = state_artifacts(d3, family, seed)
            sidecar = json.loads(paths["lock"].read_text(encoding="utf-8"))

            if sidecar.get("model_family") != family:
                raise RuntimeError(f"D3 state family mismatch: {family}/{seed}")
            if int(sidecar.get("training_seed", -1)) != seed:
                raise RuntimeError(f"D3 state seed mismatch: {family}/{seed}")
            if int(sidecar.get("target_cases", -1)) != EXPECTED_CASES:
                raise RuntimeError(f"D3 state target count mismatch: {family}/{seed}")
            if bool(sidecar.get("target_gt_pixels_decoded", True)):
                raise RuntimeError(f"D3 state reports prior GT decode: {family}/{seed}")
            if bool(sidecar.get("paot_probability_values_read", True)):
                raise RuntimeError(f"D3 state reports probability read: {family}/{seed}")
            if bool(sidecar.get("paot_gating_used", True)):
                raise RuntimeError(f"D3 state reports PAOT gating: {family}/{seed}")

            idx_rows, idx_fields = read_csv(paths["index"])
            required_idx = {
                "row_index",
                "sample_id",
                "model_family",
                "model_state_id",
                "training_seed",
                "checkpoint_sha256",
            }
            if not required_idx.issubset(set(idx_fields)):
                raise RuntimeError(f"D3 index schema mismatch: {family}/{seed}")
            if len(idx_rows) != EXPECTED_CASES:
                raise RuntimeError(f"D3 index count mismatch: {family}/{seed}")

            with np.load(paths["npz"], allow_pickle=False) as z:
                if set(z.files) != {"source_masks_packed", "a1_masks_packed"}:
                    raise RuntimeError(f"D3 NPZ arrays mismatch: {family}/{seed}")
                source_pack = np.asarray(z["source_masks_packed"], dtype=np.uint8)
                a1_pack = np.asarray(z["a1_masks_packed"], dtype=np.uint8)

            expected_shape = (EXPECTED_CASES, PACKED_BYTES)
            if source_pack.shape != expected_shape or a1_pack.shape != expected_shape:
                raise RuntimeError(f"D3 packed shape mismatch: {family}/{seed}")

            state_id = sidecar["model_state_id"]
            cksha = sidecar["checkpoint_sha256"]

            for idx_row in tqdm(
                idx_rows,
                desc=f"Utility {state_id}",
                unit="case",
                dynamic_ncols=True,
            ):
                i = int(idx_row["row_index"])
                sid = idx_row["sample_id"]
                key = (state_id, sid)

                if key in seen_keys:
                    raise RuntimeError(f"Duplicate D3 model-case key: {key}")
                seen_keys.add(key)

                if sid not in gt_masks:
                    raise RuntimeError(f"Missing GT mask: {sid}")
                if key not in d2_probs:
                    raise RuntimeError(f"Missing D2 probability: {key}")

                d2row = d2_probs[key]
                if d2row["model_family"] != family:
                    raise RuntimeError(f"D2/D3 family mismatch: {key}")
                if d2row["training_seed"] != seed:
                    raise RuntimeError(f"D2/D3 seed mismatch: {key}")
                if d2row["checkpoint_sha256"] != cksha:
                    raise RuntimeError(f"D2/D3 checkpoint mismatch: {key}")

                source_mask = unpack_mask(source_pack[i])
                a1_mask = unpack_mask(a1_pack[i])
                gt = gt_masks[sid]

                source_dice = binary_dice(source_mask, gt)
                a1_dice = binary_dice(a1_mask, gt)
                delta = float(a1_dice - source_dice)
                label = outcome_label(delta)
                probs = d2row["probabilities"]

                rows.append({
                    "sample_id": sid,
                    "model_family": family,
                    "model_state_id": state_id,
                    "training_seed": seed,
                    "checkpoint_sha256": cksha,
                    "source_dice": source_dice,
                    "a1_dice": a1_dice,
                    "delta_dice": delta,
                    "adaptation_outcome": label,
                    "harm_label": int(label == "HARM"),
                    "benefit_label": int(label == "BENEFIT"),
                    "paot_mrz19_harm_probability": probs[("MRZ19", "HARM")],
                    "paot_mrz19_benefit_probability": probs[("MRZ19", "BENEFIT")],
                    "paot_raw19_harm_probability": probs[("RAW19", "HARM")],
                    "paot_raw19_benefit_probability": probs[("RAW19", "BENEFIT")],
                })

            state_meta.append({
                "model_family": family,
                "model_state_id": state_id,
                "training_seed": seed,
                "checkpoint_sha256": cksha,
                "prediction_npz_sha256": sha256_file(paths["npz"]),
                "state_index_sha256": sha256_file(paths["index"]),
                "state_lock_sha256": sha256_file(paths["lock"]),
            })

            del source_pack, a1_pack

    if len(rows) != EXPECTED_MODEL_CASES:
        raise RuntimeError(f"Outcome rows={len(rows)} expected={EXPECTED_MODEL_CASES}")
    if seen_keys != set(d2_probs):
        raise RuntimeError("D2/D3 model-case key sets differ.")
    if len(state_meta) != EXPECTED_STATES:
        raise RuntimeError("D3 state metadata count mismatch.")

    return rows, state_meta


def task_arrays(rows, representation: str, task: str):
    label_field = "harm_label" if task == "HARM" else "benefit_label"
    prob_field = PROB_FIELDS[(representation, task)]
    y = np.asarray([int(r[label_field]) for r in rows], dtype=np.int8)
    p = np.asarray([float(r[prob_field]) for r in rows], dtype=np.float64)
    return y, p


def point_metrics(rows, representation: str, task: str):
    y, p = task_arrays(rows, representation, task)
    if np.unique(y).size < 2:
        return None
    return {
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "positive_rows": int(y.sum()),
        "negative_rows": int(len(y) - y.sum()),
    }


def prepare_weighted_metric(y: np.ndarray, p: np.ndarray):
    order = np.argsort(p, kind="mergesort")
    ys = y[order]
    ps = p[order]

    _, inverse = np.unique(ps, return_inverse=True)
    group_count = int(inverse.max()) + 1 if len(inverse) else 0

    return {
        "order": order,
        "ys": ys.astype(np.float64),
        "inverse": inverse,
        "group_count": group_count,
    }


def weighted_metrics_prepared(prepared, row_weights: np.ndarray):
    w = np.asarray(row_weights, dtype=np.float64)[prepared["order"]]
    y = prepared["ys"]
    inv = prepared["inverse"]
    ng = prepared["group_count"]

    pos_g = np.bincount(inv, weights=w * y, minlength=ng)
    neg_g = np.bincount(inv, weights=w * (1.0 - y), minlength=ng)

    total_pos = float(pos_g.sum())
    total_neg = float(neg_g.sum())
    if total_pos <= 0.0 or total_neg <= 0.0:
        return None

    cum_neg_before = np.cumsum(neg_g) - neg_g
    auc_num = float(np.sum(pos_g * (cum_neg_before + 0.5 * neg_g)))
    auroc = auc_num / (total_pos * total_neg)

    pos_desc = pos_g[::-1]
    neg_desc = neg_g[::-1]
    cum_pos = np.cumsum(pos_desc)
    cum_all = np.cumsum(pos_desc + neg_desc)
    precision = np.divide(
        cum_pos,
        cum_all,
        out=np.ones_like(cum_pos),
        where=cum_all > 0,
    )
    auprc = float(np.sum(precision * (pos_desc / total_pos)))

    return float(auroc), float(auprc)


def stable_bootstrap_seed(family: str, task: str) -> int:
    payload = f"{BOOTSTRAP_BASE_SEED}|{family}|MRZ19|{task}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32 - 1)


def cluster_bootstrap_primary(family_rows, family: str, task: str):
    by_sample = defaultdict(list)
    for row in family_rows:
        by_sample[row["sample_id"]].append(row)

    sample_ids = sorted(by_sample)
    if len(sample_ids) != EXPECTED_CASES:
        raise RuntimeError(
            f"{family} target clusters={len(sample_ids)} expected={EXPECTED_CASES}"
        )

    ordered = []
    cluster_index = []
    for ci, sid in enumerate(sample_ids):
        cluster = sorted(by_sample[sid], key=lambda r: int(r["training_seed"]))
        if len(cluster) != 3:
            raise RuntimeError(f"{family}/{sid} state rows={len(cluster)}")
        for row in cluster:
            ordered.append(row)
            cluster_index.append(ci)

    y, p = task_arrays(ordered, "MRZ19", task)
    if np.unique(y).size < 2:
        return None

    cluster_index = np.asarray(cluster_index, dtype=np.int64)
    prepared = prepare_weighted_metric(y, p)
    rng = np.random.RandomState(stable_bootstrap_seed(family, task))
    probabilities = np.full(EXPECTED_CASES, 1.0 / EXPECTED_CASES)

    aurocs = []
    auprcs = []

    for _ in tqdm(
        range(BOOTSTRAP_RESAMPLES),
        desc=f"Bootstrap {family} MRZ19 {task}",
        unit="rep",
        dynamic_ncols=True,
    ):
        counts = rng.multinomial(EXPECTED_CASES, probabilities)
        row_weights = counts[cluster_index]
        metric = weighted_metrics_prepared(prepared, row_weights)
        if metric is None:
            continue
        aurocs.append(metric[0])
        auprcs.append(metric[1])

    if not aurocs:
        return {
            "valid_bootstrap_resamples": 0,
            "auroc_ci_low": math.nan,
            "auroc_ci_high": math.nan,
            "auprc_ci_low": math.nan,
            "auprc_ci_high": math.nan,
        }

    return {
        "valid_bootstrap_resamples": len(aurocs),
        "auroc_ci_low": float(np.quantile(aurocs, 0.025)),
        "auroc_ci_high": float(np.quantile(aurocs, 0.975)),
        "auprc_ci_low": float(np.quantile(auprcs, 0.025)),
        "auprc_ci_high": float(np.quantile(auprcs, 0.975)),
    }


def positive_clusters(rows, task: str) -> int:
    label_field = "harm_label" if task == "HARM" else "benefit_label"
    by_sample = defaultdict(list)
    for row in rows:
        by_sample[row["sample_id"]].append(int(row[label_field]))
    return int(sum(any(vals) for vals in by_sample.values()))


def architecture_metrics(outcome_rows):
    results = []

    for family in FAMILIES:
        family_rows = [r for r in outcome_rows if r["model_family"] == family]
        if len(family_rows) != 3000:
            raise RuntimeError(f"{family} rows={len(family_rows)} expected=3000")

        for representation in REPRESENTATIONS:
            for task in TASKS:
                point = point_metrics(family_rows, representation, task)
                pos_clusters = positive_clusters(family_rows, task)

                base = {
                    "model_family": family,
                    "representation": representation,
                    "task": task,
                    "rows": len(family_rows),
                    "target_clusters": EXPECTED_CASES,
                    "positive_clusters": pos_clusters,
                    "primary": int(representation == "MRZ19"),
                }

                if point is None:
                    base.update({
                        "positive_rows": (
                            sum(int(r["harm_label"]) for r in family_rows)
                            if task == "HARM"
                            else sum(int(r["benefit_label"]) for r in family_rows)
                        ),
                        "auroc": "",
                        "auprc": "",
                        "bootstrap_resamples_requested": (
                            BOOTSTRAP_RESAMPLES if representation == "MRZ19" else 0
                        ),
                        "valid_bootstrap_resamples": 0,
                        "auroc_ci_low": "",
                        "auroc_ci_high": "",
                        "auprc_ci_low": "",
                        "auprc_ci_high": "",
                        "class_degenerate": 1,
                    })
                    results.append(base)
                    continue

                if representation == "MRZ19":
                    ci = cluster_bootstrap_primary(family_rows, family, task)
                else:
                    ci = {
                        "valid_bootstrap_resamples": 0,
                        "auroc_ci_low": "",
                        "auroc_ci_high": "",
                        "auprc_ci_low": "",
                        "auprc_ci_high": "",
                    }

                base.update({
                    "positive_rows": point["positive_rows"],
                    "auroc": point["auroc"],
                    "auprc": point["auprc"],
                    "bootstrap_resamples_requested": (
                        BOOTSTRAP_RESAMPLES if representation == "MRZ19" else 0
                    ),
                    "valid_bootstrap_resamples": ci["valid_bootstrap_resamples"],
                    "auroc_ci_low": ci["auroc_ci_low"],
                    "auroc_ci_high": ci["auroc_ci_high"],
                    "auprc_ci_low": ci["auprc_ci_low"],
                    "auprc_ci_high": ci["auprc_ci_high"],
                    "class_degenerate": 0,
                })
                results.append(base)

    return results


def secondary_metrics(rows, by_state: bool):
    if by_state:
        groups = defaultdict(list)
        for row in rows:
            groups[(row["model_family"], row["model_state_id"])].append(row)
    else:
        groups = {("ALL_ARCHITECTURES", "ALL_9_STATES"): list(rows)}

    out = []
    for (family, state_id), subset in sorted(groups.items()):
        for rep in REPRESENTATIONS:
            for task in TASKS:
                point = point_metrics(subset, rep, task)
                out.append({
                    "model_family": family,
                    "model_state_id": state_id,
                    "representation": rep,
                    "task": task,
                    "rows": len(subset),
                    "positive_rows": (
                        point["positive_rows"]
                        if point is not None
                        else (
                            sum(int(r["harm_label"]) for r in subset)
                            if task == "HARM"
                            else sum(int(r["benefit_label"]) for r in subset)
                        )
                    ),
                    "auroc": "" if point is None else point["auroc"],
                    "auprc": "" if point is None else point["auprc"],
                    "class_degenerate": int(point is None),
                })
    return out


def segmentation_summary(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["model_family"], row["model_state_id"])].append(row)
        groups[(row["model_family"], "ALL_3_STATES")].append(row)
        groups[("ALL_ARCHITECTURES", "ALL_9_STATES")].append(row)

    out = []
    for (family, state_id), subset in sorted(groups.items()):
        src = np.asarray([float(r["source_dice"]) for r in subset])
        a1 = np.asarray([float(r["a1_dice"]) for r in subset])
        delta = np.asarray([float(r["delta_dice"]) for r in subset])
        labels = Counter(r["adaptation_outcome"] for r in subset)

        out.append({
            "model_family": family,
            "model_state_id": state_id,
            "rows": len(subset),
            "source_dice_mean": float(src.mean()),
            "source_dice_median": float(np.median(src)),
            "a1_dice_mean": float(a1.mean()),
            "a1_dice_median": float(np.median(a1)),
            "delta_dice_mean": float(delta.mean()),
            "delta_dice_median": float(np.median(delta)),
            "harm_rows": int(labels.get("HARM", 0)),
            "neutral_rows": int(labels.get("NEUTRAL", 0)),
            "benefit_rows": int(labels.get("BENEFIT", 0)),
            "harm_fraction": float(labels.get("HARM", 0) / len(subset)),
            "benefit_fraction": float(labels.get("BENEFIT", 0) / len(subset)),
        })
    return out


def evaluate_gate(arch_rows):
    primary = {
        (r["model_family"], r["task"]): r
        for r in arch_rows
        if r["representation"] == "MRZ19"
    }
    required = {(f, t) for f in FAMILIES for t in TASKS}
    if set(primary) != required:
        raise RuntimeError("Primary metric grid incomplete.")

    if any(int(primary[k]["class_degenerate"]) == 1 for k in required):
        return {
            "decision": DECISION_INCONCLUSIVE,
            "reason": "At least one primary architecture/task is class-degenerate.",
            "criteria": PRIMARY_CRITERIA,
            "metrics": {},
            "checks": {},
        }

    harm = np.asarray(
        [float(primary[(f, "HARM")]["auroc"]) for f in FAMILIES]
    )
    benefit = np.asarray(
        [float(primary[(f, "BENEFIT")]["auroc"]) for f in FAMILIES]
    )

    metrics = {
        "architecture_harm_aurocs": {
            f: float(primary[(f, "HARM")]["auroc"]) for f in FAMILIES
        },
        "architecture_benefit_aurocs": {
            f: float(primary[(f, "BENEFIT")]["auroc"]) for f in FAMILIES
        },
        "median_architecture_harm_auroc": float(np.median(harm)),
        "minimum_architecture_harm_auroc": float(np.min(harm)),
        "median_architecture_benefit_auroc": float(np.median(benefit)),
        "minimum_architecture_benefit_auroc": float(np.min(benefit)),
    }

    checks = {
        "A_median_harm_ge_0p70":
            metrics["median_architecture_harm_auroc"] >= 0.70,
        "B_min_harm_ge_0p60":
            metrics["minimum_architecture_harm_auroc"] >= 0.60,
        "C_median_benefit_ge_0p75":
            metrics["median_architecture_benefit_auroc"] >= 0.75,
        "D_min_benefit_ge_0p65":
            metrics["minimum_architecture_benefit_auroc"] >= 0.65,
    }

    decision = DECISION_PASS if all(checks.values()) else DECISION_FAIL
    return {
        "decision": decision,
        "reason": (
            "All four preregistered MRZ19 point-estimate criteria passed."
            if decision == DECISION_PASS
            else "At least one preregistered MRZ19 point-estimate criterion failed."
        ),
        "criteria": PRIMARY_CRITERIA,
        "metrics": metrics,
        "checks": checks,
    }


def preflight():
    d1, d2, d3, d1_manifest_path, d2_table_path, d3_rule_path = validate_upstream()
    validate_gt_rule(d3_rule_path)
    d1_manifest = load_d1_manifest(d1_manifest_path)
    d2_probs = load_d2_probabilities(d2_table_path)

    if set(sid for _, sid in d2_probs) != set(d1_manifest):
        raise RuntimeError("D2 sample set differs from D1 target set.")

    # Verify all 27 D3 state artifacts without decoding GT.
    state_count = 0
    for family, seeds in (
        ("PraNet", (20260817, 20260818, 20260819)),
        ("DeepLabV3-R50", (20260817, 20260818, 20260819)),
        ("SegFormer-B0", (20260820, 20260821, 20260822)),
    ):
        for seed in seeds:
            paths = state_artifacts(d3, family, seed)
            sidecar = json.loads(paths["lock"].read_text(encoding="utf-8"))
            if sidecar.get("model_family") != family:
                raise RuntimeError("D3 preflight family mismatch.")
            if int(sidecar.get("training_seed", -1)) != seed:
                raise RuntimeError("D3 preflight seed mismatch.")
            state_count += 1

    if state_count != EXPECTED_STATES:
        raise RuntimeError("D3 preflight state count mismatch.")

    print("===== Q1-R05D4 PREFLIGHT =====")
    print(f"protocol_sha256={EXPECTED_PROTOCOL_SHA256}")
    print(f"r05d1_lock_sha256={EXPECTED_R05D1_LOCK_SHA256}")
    print(f"r05d2_lock_sha256={EXPECTED_R05D2_LOCK_SHA256}")
    print(f"r05d3_lock_sha256={EXPECTED_R05D3_LOCK_SHA256}")
    print(f"target_cases={len(d1_manifest)}")
    print(f"prospective_model_case_rows={len(d2_probs)}")
    print(f"prediction_states={state_count}")
    print("GT_rule_verified=YES")
    print("GT_pixels_opened_in_preflight=NO")
    print("model_inference=NO")
    print("TTA=NO")
    print("predictor_fit=NO")
    print("target_threshold_selection=NO")
    print("PREFLIGHT_PASS")


def run(args):
    (
        d1,
        d2,
        d3,
        d1_manifest_path,
        d2_table_path,
        d3_rule_path,
    ) = validate_upstream()

    if args.output_dir.exists():
        raise FileExistsError(f"Final D4 output exists: {args.output_dir}")

    build_dir = Path(str(args.output_dir) + "__building")
    if build_dir.exists():
        raise FileExistsError(
            f"Partial D4 output exists: {build_dir}. "
            "Do not silently discard a post-reveal partial run."
        )
    build_dir.mkdir(parents=True, exist_ok=False)

    protocol_copy = build_dir / "preregistered_protocol_copy.md"
    shutil.copy2(PROTOCOL, protocol_copy)
    validate_sha(protocol_copy, EXPECTED_PROTOCOL_SHA256, "D4 protocol copy")

    validate_gt_rule(d3_rule_path)
    d1_manifest = load_d1_manifest(d1_manifest_path)
    d2_probs = load_d2_probabilities(d2_table_path)

    if set(sid for _, sid in d2_probs) != set(d1_manifest):
        raise RuntimeError("D2 sample set differs from frozen D1 cohort.")

    upstream_path = build_dir / "upstream_audit.json"
    write_json(
        upstream_path,
        {
            "script_version": VERSION,
            "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
            "r05d1_lock_sha256": EXPECTED_R05D1_LOCK_SHA256,
            "r05d1_decision": d1.get("decision"),
            "r05d2_lock_sha256": EXPECTED_R05D2_LOCK_SHA256,
            "r05d2_decision": d2.get("decision"),
            "r05d2_probability_table_sha256": EXPECTED_D2_TABLE_SHA256,
            "r05d3_lock_sha256": EXPECTED_R05D3_LOCK_SHA256,
            "r05d3_decision": d3.get("decision"),
            "target_cases": EXPECTED_CASES,
            "model_states": EXPECTED_STATES,
            "model_case_rows": EXPECTED_MODEL_CASES,
            "gt_rule_verified_before_reveal": True,
            "confirmatory_criteria_verified_before_reveal": True,
            "no_model_inference_in_d4": True,
            "no_tta_in_d4": True,
            "no_predictor_fit_in_d4": True,
            "no_target_threshold_selection": True,
        },
    )

    # -------- FIRST TARGET GT PIXEL REVEAL --------
    gt_masks, gt_rows = reveal_gt(d1_manifest)

    gt_path = build_dir / "gt_reveal_manifest.csv"
    write_csv(
        gt_path,
        gt_rows,
        [
            "sample_id",
            "gt_path",
            "gt_raw_sha256",
            "native_width",
            "native_height",
            "native_background_pixels",
            "native_non_neoplastic_pixels",
            "native_neoplastic_pixels",
            "native_binary_foreground_pixels",
            "resized_binary_foreground_pixels",
            "gt_decode_rule",
        ],
    )

    outcome_rows, state_meta = build_outcomes(gt_masks, d2_probs, d3)

    # Update upstream audit with verified state artifact identities.
    upstream = json.loads(upstream_path.read_text(encoding="utf-8"))
    upstream["d3_state_artifacts"] = state_meta
    write_json(upstream_path, upstream)

    outcome_path = build_dir / "model_case_outcomes.csv"
    write_csv(
        outcome_path,
        outcome_rows,
        [
            "sample_id",
            "model_family",
            "model_state_id",
            "training_seed",
            "checkpoint_sha256",
            "source_dice",
            "a1_dice",
            "delta_dice",
            "adaptation_outcome",
            "harm_label",
            "benefit_label",
            "paot_mrz19_harm_probability",
            "paot_mrz19_benefit_probability",
            "paot_raw19_harm_probability",
            "paot_raw19_benefit_probability",
        ],
    )

    seg_rows = segmentation_summary(outcome_rows)
    seg_path = build_dir / "segmentation_outcome_summary.csv"
    write_csv(
        seg_path,
        seg_rows,
        [
            "model_family",
            "model_state_id",
            "rows",
            "source_dice_mean",
            "source_dice_median",
            "a1_dice_mean",
            "a1_dice_median",
            "delta_dice_mean",
            "delta_dice_median",
            "harm_rows",
            "neutral_rows",
            "benefit_rows",
            "harm_fraction",
            "benefit_fraction",
        ],
    )

    arch_rows = architecture_metrics(outcome_rows)
    arch_path = build_dir / "architecture_confirmatory_metrics.csv"
    write_csv(
        arch_path,
        arch_rows,
        [
            "model_family",
            "representation",
            "task",
            "rows",
            "target_clusters",
            "positive_rows",
            "positive_clusters",
            "auroc",
            "auprc",
            "bootstrap_resamples_requested",
            "valid_bootstrap_resamples",
            "auroc_ci_low",
            "auroc_ci_high",
            "auprc_ci_low",
            "auprc_ci_high",
            "class_degenerate",
            "primary",
        ],
    )

    state_rows = secondary_metrics(outcome_rows, by_state=True)
    state_path = build_dir / "state_secondary_metrics.csv"
    write_csv(
        state_path,
        state_rows,
        [
            "model_family",
            "model_state_id",
            "representation",
            "task",
            "rows",
            "positive_rows",
            "auroc",
            "auprc",
            "class_degenerate",
        ],
    )

    pooled_rows = secondary_metrics(outcome_rows, by_state=False)
    pooled_path = build_dir / "pooled_secondary_metrics.csv"
    write_csv(
        pooled_path,
        pooled_rows,
        [
            "model_family",
            "model_state_id",
            "representation",
            "task",
            "rows",
            "positive_rows",
            "auroc",
            "auprc",
            "class_degenerate",
        ],
    )

    gate = evaluate_gate(arch_rows)
    gate_path = build_dir / "confirmatory_gate.json"
    write_json(gate_path, gate)

    decision = gate["decision"]
    decision_path = build_dir / "decision.txt"
    decision_path.write_text(decision + "\n", encoding="utf-8")

    boundary_path = build_dir / "information_boundary_audit.json"
    write_json(
        boundary_path,
        {
            "target_gt_pixels_decoded_for_first_time_in_d4": True,
            "gt_files_decoded": EXPECTED_CASES,
            "source_predictions_modified": False,
            "a1_predictions_modified": False,
            "d2_paot_probabilities_modified": False,
            "model_inference_run_in_d4": False,
            "tta_run_in_d4": False,
            "predictor_fit_or_modified": False,
            "target_calibration": False,
            "target_probability_threshold_selected": False,
            "target_hyperparameter_tuning": False,
            "target_case_exclusion_after_gt_reveal": False,
            "source_dice_computed": True,
            "a1_dice_computed": True,
            "delta_dice_computed": True,
            "adaptation_outcomes_revealed": True,
            "primary_paot_evaluation_run": True,
            "raw19_comparator_run": True,
            "target_image_cluster_bootstrap_run": True,
        },
    )

    primary = {
        (r["model_family"], r["task"]): r
        for r in arch_rows
        if r["representation"] == "MRZ19"
    }

    log = [
        "===== Q1-R05D4 NEOPOLYP GT REVEAL + PAOT CONFIRMATORY EVALUATION =====",
        f"Script version: {VERSION}",
        f"Build: {BUILD}",
        "",
        "GT reveal:",
        f"  GT files decoded={len(gt_rows)}",
        "  nearest RGB prototype rule=YES",
        "  non-neoplastic + neoplastic -> binary foreground",
        "  resized=352x352 nearest",
        "",
        "Frozen outcomes:",
        f"  model-case rows={len(outcome_rows)}",
        f"  HARM threshold={HARM_THRESHOLD}",
        f"  BENEFIT threshold={BENEFIT_THRESHOLD}",
        "",
        "Primary MRZ19 architecture metrics:",
    ]

    for family in FAMILIES:
        for task in TASKS:
            r = primary[(family, task)]
            log.append(
                f"  {family} {task}: "
                f"AUROC={r['auroc']} "
                f"AUPRC={r['auprc']} "
                f"AUROC95CI=[{r['auroc_ci_low']},{r['auroc_ci_high']}] "
                f"AUPRC95CI=[{r['auprc_ci_low']},{r['auprc_ci_high']}] "
                f"positive_rows={r['positive_rows']} "
                f"positive_clusters={r['positive_clusters']} "
                f"valid_bootstrap={r['valid_bootstrap_resamples']}"
            )

    log += [
        "",
        "Frozen GO criteria:",
        "  median architecture HARM AUROC >= 0.70",
        "  minimum architecture HARM AUROC >= 0.60",
        "  median architecture BENEFIT AUROC >= 0.75",
        "  minimum architecture BENEFIT AUROC >= 0.65",
        "",
        "Gate metrics:",
    ]

    if decision != DECISION_INCONCLUSIVE:
        for key, value in gate["metrics"].items():
            log.append(f"  {key}={value}")
        log += ["", "Checks:"]
        for key, passed in gate["checks"].items():
            log.append(f"  {key}={'PASS' if passed else 'FAIL'}")

    log += [
        "",
        "Decision:",
        f"  {decision}",
        "",
        "Post-reveal boundary:",
        "  model inference in D4=NO",
        "  TTA in D4=NO",
        "  predictor modification=NO",
        "  target calibration=NO",
        "  target threshold selection=NO",
        "  post-hoc target case exclusion=NO",
    ]

    run_log_path = build_dir / "run_log.txt"
    run_log_path.write_text("\n".join(log) + "\n", encoding="utf-8")

    artifacts = {
        "protocol_copy": protocol_copy,
        "upstream_audit": upstream_path,
        "gt_reveal_manifest": gt_path,
        "model_case_outcomes": outcome_path,
        "segmentation_outcome_summary": seg_path,
        "architecture_confirmatory_metrics": arch_path,
        "state_secondary_metrics": state_path,
        "pooled_secondary_metrics": pooled_path,
        "confirmatory_gate": gate_path,
        "information_boundary_audit": boundary_path,
        "decision": decision_path,
        "run_log": run_log_path,
    }

    final_lock = {
        "script_version": VERSION,
        "build": BUILD,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "r05d1_lock_sha256": EXPECTED_R05D1_LOCK_SHA256,
        "r05d2_lock_sha256": EXPECTED_R05D2_LOCK_SHA256,
        "r05d3_lock_sha256": EXPECTED_R05D3_LOCK_SHA256,
        "d2_probability_table_sha256": EXPECTED_D2_TABLE_SHA256,
        "target_cases": EXPECTED_CASES,
        "model_states": EXPECTED_STATES,
        "model_case_rows": EXPECTED_MODEL_CASES,
        "gt_files_decoded": EXPECTED_CASES,
        "target_gt_pixels_revealed": True,
        "primary_representation": "MRZ19",
        "confirmatory_criteria": PRIMARY_CRITERIA,
        "decision": decision,
        "gate": gate,
        "model_inference_run_in_d4": False,
        "tta_run_in_d4": False,
        "predictor_fit_or_modified": False,
        "target_calibration": False,
        "target_threshold_selected": False,
        "posthoc_case_exclusion": False,
        "artifacts": {
            name: {
                "relative_path": str(path.relative_to(build_dir)),
                "sha256": sha256_file(path),
            }
            for name, path in artifacts.items()
        },
    }

    lock_path = (
        build_dir / "Q1_R05D4_NEOPOLYP_CONFIRMATORY_EVALUATION_LOCK.json"
    )
    write_json(lock_path, final_lock)
    lock_sha = sha256_file(lock_path)

    for name, meta in final_lock["artifacts"].items():
        p = build_dir / meta["relative_path"]
        if sha256_file(p) != meta["sha256"]:
            raise RuntimeError(f"Artifact changed before D4 commit: {name}")

    build_dir.rename(args.output_dir)

    print()
    print((args.output_dir / "run_log.txt").read_text(encoding="utf-8"))
    print(
        "Q1-R05D4 LOCK:",
        args.output_dir / "Q1_R05D4_NEOPOLYP_CONFIRMATORY_EVALUATION_LOCK.json",
    )
    print("Q1-R05D4 LOCK SHA256:", lock_sha)


def self_test():
    assert EXPECTED_CASES == 1000
    assert EXPECTED_STATES == 9
    assert EXPECTED_MODEL_CASES == 9000
    assert MASK_H == 352 and MASK_W == 352
    assert PACKED_BYTES == 15488
    assert BOOTSTRAP_RESAMPLES == 10000
    assert BOOTSTRAP_BASE_SEED == 20260819

    z = np.zeros((MASK_H, MASK_W), dtype=np.uint8)
    o = np.ones((MASK_H, MASK_W), dtype=np.uint8)
    assert binary_dice(z, z) == 1.0
    assert binary_dice(o, o) == 1.0
    assert binary_dice(z, o) == 0.0

    assert outcome_label(-0.02) == "HARM"
    assert outcome_label(-0.019999) == "NEUTRAL"
    assert outcome_label(0.0) == "NEUTRAL"
    assert outcome_label(0.02) == "BENEFIT"

    rgb = np.asarray(
        [[[0, 0, 0], [0, 250, 0], [250, 0, 0]]],
        dtype=np.uint8,
    )
    assert nearest_rgb_classes(rgb).tolist() == [[0, 1, 2]]

    mask = np.zeros((MASK_H, MASK_W), dtype=np.uint8)
    mask[10:30, 15:40] = 1
    packed = np.packbits(mask.reshape(-1), bitorder=BITORDER)
    assert np.array_equal(mask, unpack_mask(packed))

    toy = [
        {
            "harm_label": y,
            "benefit_label": 1 - y,
            "paot_mrz19_harm_probability": p,
            "paot_mrz19_benefit_probability": 1 - p,
            "paot_raw19_harm_probability": p,
            "paot_raw19_benefit_probability": 1 - p,
        }
        for y, p in [(0, 0.1), (0, 0.2), (1, 0.8), (1, 0.9)]
    ]
    pm = point_metrics(toy, "MRZ19", "HARM")
    assert pm is not None
    assert abs(pm["auroc"] - 1.0) < 1e-12
    assert abs(pm["auprc"] - 1.0) < 1e-12

    # Validate fast weighted bootstrap metric against sklearn.
    rng = np.random.RandomState(7)
    y = np.asarray([0, 1, 0, 1, 1, 0], dtype=np.int8)
    p = np.asarray([0.1, 0.8, 0.3, 0.8, 0.6, 0.2], dtype=np.float64)
    w = rng.randint(0, 4, size=len(y)).astype(np.float64)
    if w[y == 1].sum() == 0:
        w[1] = 1
    if w[y == 0].sum() == 0:
        w[0] = 1

    prepared = prepare_weighted_metric(y, p)
    fast_auc, fast_ap = weighted_metrics_prepared(prepared, w)
    ref_auc = roc_auc_score(y, p, sample_weight=w)
    ref_ap = average_precision_score(y, p, sample_weight=w)
    assert abs(fast_auc - ref_auc) < 1e-12
    assert abs(fast_ap - ref_ap) < 1e-12

    assert PRIMARY_CRITERIA[
        "median_architecture_harm_auroc_min"
    ] == 0.70
    assert PRIMARY_CRITERIA[
        "minimum_architecture_harm_auroc_min"
    ] == 0.60
    assert PRIMARY_CRITERIA[
        "median_architecture_benefit_auroc_min"
    ] == 0.75
    assert PRIMARY_CRITERIA[
        "minimum_architecture_benefit_auroc_min"
    ] == 0.65

    print("CARDINALITY_TEST_PASS")
    print("DICE_TEST_PASS")
    print("OUTCOME_BOUNDARY_TEST_PASS")
    print("FROZEN_RGB_GT_RULE_TEST_PASS")
    print("PACKBITS_TEST_PASS")
    print("METRIC_SANITY_TEST_PASS")
    print("WEIGHTED_BOOTSTRAP_METRIC_EQUIVALENCE_TEST_PASS")
    print("FROZEN_CONFIRMATORY_CRITERIA_TEST_PASS")
    print("SELF_TEST_PASS")


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Q1-R05D4: first NeoPolyp GT reveal and confirmatory PAOT "
            "evaluation using frozen D2 probabilities and D3 predictions."
        )
    )
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
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
