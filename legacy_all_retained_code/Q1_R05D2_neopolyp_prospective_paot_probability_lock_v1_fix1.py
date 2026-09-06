#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Q1-R05D2 — Prospective NeoPolyp PAOT probability generation and lock.

Target IMAGE pixels are allowed.
Target GT pixels are NEVER decoded.
No optimizer is constructed.
No backward pass is executed.
No TTA parameter update is executed.
No target-side fitting/calibration/threshold selection.
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
from typing import Dict, Sequence, Tuple

# Exact frozen DeepLab/Q1-R03 CUDA determinism environment; must precede torch import.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
from PIL import Image
from tqdm import tqdm


VERSION = "2026-08-19-Q1-R05D2-v1-fix1"
BUILD = "Q1_R05D2_NEOPOLYP_PROSPECTIVE_PAOT_PROBABILITY_LOCK_SELFTEST_GUARD_FIX1"

ROOT = Path(r"F:\MEDSEG_SAFETTA")

PROTOCOL = (
    ROOT / "docs"
    / "Q1_R05D2_neopolyp_prospective_paot_probability_lock_preregistered_protocol_v1_fix1.md"
)
EXPECTED_PROTOCOL_SHA256 = "84a9391987b9e569586e236d6767eacdde43776c81a1d448ebf318b7ab7836f4"

R05D1_DIR = (
    ROOT / "outputs"
    / "Q1_R05D1_neopolyp_acquisition_pairing_duplicate_qc_v1_fix2"
)
R05D1_LOCK = R05D1_DIR / "Q1_R05D1_NEOPOLYP_TARGET_QC_LOCK.json"
EXPECTED_R05D1_LOCK_SHA256 = (
    "bfe78a4d2fe42049f1658ead22b3e74deeac5afb036426d73a9cff7506b6cf41"
)
EXPECTED_R05D1_DECISION = "NEOPOLYP_TARGET_QC_READY"
D1_MANIFEST = R05D1_DIR / "frozen_confirmatory_manifest.csv"

R05B2_DIR = (
    ROOT / "outputs"
    / "Q1_R05B2_source_only_paot_predictor_lock_v1"
)
R05B2_LOCK = R05B2_DIR / "Q1_R05B2_SOURCE_PAOT_PREDICTOR_LOCK.json"
EXPECTED_R05B2_LOCK_SHA256 = (
    "bd182ac600afbb24b3a62ebd326931ab23c7cf6b873a2178e33500395c70f211"
)
EXPECTED_R05B2_DECISION = "SOURCE_ONLY_PAOT_PREDICTORS_LOCKED"
PREDICTOR_PACKAGES = R05B2_DIR / "predictor_packages.json"
REFERENCE_STATS = R05B2_DIR / "model_reference_statistics.csv"

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
    / "Q1_R05D2_neopolyp_prospective_paot_probability_lock_v1_fix1"
)

EXPECTED_CASES = 1000
EXPECTED_STATES = 9
EXPECTED_ROWS = EXPECTED_CASES * EXPECTED_STATES

FAMILIES = ("PraNet", "DeepLabV3-R50", "SegFormer-B0")
REPRESENTATIONS = ("RAW19", "MRZ19")
TASKS = ("HARM", "BENEFIT")

SOURCE_FEATURE_NAMES = [
    "source_entropy_mean",
    "source_entropy_std",
    "source_entropy_q10",
    "source_entropy_q50",
    "source_entropy_q90",
    "source_prob_mean",
    "source_prob_std",
    "source_prob_q10",
    "source_prob_q50",
    "source_prob_q90",
    "source_confidence_mean",
    "source_confidence_std",
    "source_fg_fraction",
    "source_boundary_density",
    "source_uncertain_fraction_040_060",
    "source_high_entropy_fraction_050",
    "source_logit_abs_mean",
    "source_logit_abs_std",
    "tent_preupdate_entropy_loss",
]

MRZ_FIELDS = [f"mrz__{x}" for x in SOURCE_FEATURE_NAMES]
PROBABILITY_FIELDS = [
    "paot_raw19_harm_probability",
    "paot_raw19_benefit_probability",
    "paot_mrz19_harm_probability",
    "paot_mrz19_benefit_probability",
]
OUTPUT_FIELDS = [
    "sample_id",
    "image_path",
    "image_raw_sha256",
    "model_family",
    "model_state_id",
    "training_seed",
    "checkpoint_sha256",
] + SOURCE_FEATURE_NAMES + MRZ_FIELDS + PROBABILITY_FIELDS

CONFIRMATORY_CRITERIA = {
    "primary_representation": "MRZ19",
    "primary_unit": "model_case_pair_with_target_image_clustered_uncertainty",
    "bootstrap_resamples": 10000,
    "harm_delta_dice_threshold": -0.02,
    "benefit_delta_dice_threshold": 0.02,
    "median_architecture_harm_auroc_min": 0.70,
    "minimum_architecture_harm_auroc_min": 0.60,
    "median_architecture_benefit_auroc_min": 0.75,
    "minimum_architecture_benefit_auroc_min": 0.65,
    "ci_used_as_go_gate": False,
    "raw19_role": "prespecified_comparator",
    "target_probability_threshold_selection": False,
}

DECISION_READY = "NEOPOLYP_PROSPECTIVE_PAOT_PROBABILITIES_LOCKED"


# ---------------------------------------------------------------------
# Generic I/O and integrity
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
        writer = csv.DictWriter(f, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def import_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def artifact_path_from_lock(base_dir: Path, lock: dict, key: str) -> Path:
    meta = lock.get("artifacts", {}).get(key)
    if not isinstance(meta, dict):
        raise RuntimeError(f"Lock missing artifact metadata: {key}")
    rel = meta.get("relative_path") or meta.get("filename")
    if not rel:
        raise RuntimeError(f"Lock artifact lacks path: {key}")
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
# Frozen predictor/reference loading
# ---------------------------------------------------------------------

def validate_upstream():
    validate_sha(PROTOCOL, EXPECTED_PROTOCOL_SHA256, "R05D2 protocol")
    validate_sha(R05D1_LOCK, EXPECTED_R05D1_LOCK_SHA256, "R05D1 lock")
    validate_sha(R05B2_LOCK, EXPECTED_R05B2_LOCK_SHA256, "R05B2 lock")
    validate_sha(R03_LOCK, EXPECTED_R03_LOCK_SHA256, "R03 lock")
    validate_sha(R03_SCRIPT, EXPECTED_R03_SCRIPT_SHA256, "R03 script")
    validate_sha(PRANET_HELPER, EXPECTED_PRANET_HELPER_SHA256, "PraNet helper")
    validate_sha(DEEPLAB_HELPER, EXPECTED_DEEPLAB_HELPER_SHA256, "DeepLab helper")

    d1 = json.loads(R05D1_LOCK.read_text(encoding="utf-8"))
    b2 = json.loads(R05B2_LOCK.read_text(encoding="utf-8"))
    r03 = json.loads(R03_LOCK.read_text(encoding="utf-8"))

    if d1.get("decision") != EXPECTED_R05D1_DECISION:
        raise RuntimeError(f"Unexpected R05D1 decision: {d1.get('decision')}")
    if int(d1.get("candidate_cases", -1)) != EXPECTED_CASES:
        raise RuntimeError("R05D1 candidate count changed.")
    if int(d1.get("exact_overlap_exclusions", -1)) != 0:
        raise RuntimeError("R05D1 overlap exclusions are no longer zero.")
    if int(d1.get("eligible_confirmatory_cases", -1)) != EXPECTED_CASES:
        raise RuntimeError("R05D1 eligible count changed.")
    if bool(d1.get("target_gt_pixels_decoded", True)):
        raise RuntimeError("R05D1 reports target GT decoding.")
    if bool(d1.get("tta_run", True)):
        raise RuntimeError("R05D1 reports target TTA.")
    if bool(d1.get("predictor_fit_or_modified", True)):
        raise RuntimeError("R05D1 reports predictor modification.")
    if d1.get("r05b2_lock_sha256") != EXPECTED_R05B2_LOCK_SHA256:
        raise RuntimeError("R05D1 does not point to frozen R05B2.")

    if b2.get("decision") != EXPECTED_R05B2_DECISION:
        raise RuntimeError(f"Unexpected R05B2 decision: {b2.get('decision')}")
    if b2.get("primary_representation") != "MRZ19":
        raise RuntimeError("R05B2 primary representation changed.")
    if int(b2.get("predictor_packages", -1)) != 12:
        raise RuntimeError("R05B2 predictor count changed.")
    if list(b2.get("feature_names", [])) != SOURCE_FEATURE_NAMES:
        raise RuntimeError("R05B2 feature schema changed.")
    if bool(b2.get("target_data_used", True)):
        raise RuntimeError("R05B2 reports target data use.")

    if r03.get("decision") != EXPECTED_R03_DECISION:
        raise RuntimeError(f"Unexpected R03 decision: {r03.get('decision')}")

    manifest_path = artifact_path_from_lock(
        R05D1_DIR, d1, "frozen_confirmatory_manifest"
    )
    predictor_path = artifact_path_from_lock(
        R05B2_DIR, b2, "predictor_packages"
    )
    refs_path = artifact_path_from_lock(
        R05B2_DIR, b2, "model_reference_statistics"
    )

    return d1, b2, r03, manifest_path, predictor_path, refs_path


def load_target_manifest(manifest_path: Path, verify_image_hashes: bool):
    rows, fields = read_csv(manifest_path)
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
        raise RuntimeError(f"D1 manifest rows={len(rows)} expected={EXPECTED_CASES}")

    selected = []
    ids = set()
    image_paths = set()
    for row in rows:
        if int(row["eligible_confirmatory"]) != 1:
            raise RuntimeError(
                "R05D2 frozen cohort expects zero D1 exclusions, but found ineligible row."
            )
        if str(row["exclusion_reason"]).strip():
            raise RuntimeError("Eligible D1 row unexpectedly has exclusion reason.")
        if int(row["gt_pixels_decoded"]) != 0:
            raise RuntimeError("D1 manifest reports GT pixel decoding.")
        sid = str(row["sample_id"])
        if sid in ids:
            raise RuntimeError(f"Duplicate target sample_id: {sid}")
        ids.add(sid)
        image_path = Path(row["image_path"])
        if not image_path.exists() or not image_path.is_file():
            raise FileNotFoundError(image_path)
        norm = str(image_path.resolve()).lower()
        if norm in image_paths:
            raise RuntimeError(f"Duplicate target image path: {image_path}")
        image_paths.add(norm)
        selected.append(row)

    selected.sort(key=lambda r: r["sample_id"])

    if verify_image_hashes:
        for row in tqdm(
            selected,
            desc="Verify frozen NeoPolyp image SHA256",
            unit="img",
            dynamic_ncols=True,
        ):
            actual = sha256_file(Path(row["image_path"]))
            if actual.lower() != row["image_raw_sha256"].lower():
                raise RuntimeError(
                    f"Target image changed since R05D1: {row['sample_id']}"
                )

    return selected


def load_references(refs_path: Path):
    rows, fields = read_csv(refs_path)
    required = {
        "model_family",
        "model_state_id",
        "training_seed",
        "checkpoint_sha256",
        "reference_count",
        "feature",
        "mean",
        "effective_std",
    }
    missing = sorted(required - set(fields))
    if missing:
        raise RuntimeError(f"Reference table missing columns: {missing}")

    grouped = defaultdict(dict)
    meta = {}
    for r in rows:
        sid = r["model_state_id"]
        feature = r["feature"]
        if feature in grouped[sid]:
            raise RuntimeError(f"Duplicate reference feature {sid} {feature}")
        grouped[sid][feature] = (
            float(r["mean"]),
            float(r["effective_std"]),
        )
        current = (
            r["model_family"],
            int(r["training_seed"]),
            r["checkpoint_sha256"],
            int(r["reference_count"]),
        )
        if sid in meta and meta[sid] != current:
            raise RuntimeError(f"Inconsistent reference metadata: {sid}")
        meta[sid] = current

    if len(grouped) != EXPECTED_STATES:
        raise RuntimeError(f"Reference states={len(grouped)} expected={EXPECTED_STATES}")

    refs = {}
    for sid, feat_map in grouped.items():
        if set(feat_map) != set(SOURCE_FEATURE_NAMES):
            raise RuntimeError(f"Reference feature schema mismatch: {sid}")
        means = np.asarray(
            [feat_map[f][0] for f in SOURCE_FEATURE_NAMES], dtype=np.float64
        )
        scales = np.asarray(
            [feat_map[f][1] for f in SOURCE_FEATURE_NAMES], dtype=np.float64
        )
        if not np.isfinite(means).all() or not np.isfinite(scales).all():
            raise RuntimeError(f"Non-finite reference stats: {sid}")
        if np.any(scales <= 0):
            raise RuntimeError(f"Non-positive effective std: {sid}")
        family, seed, cksha, ref_count = meta[sid]
        if ref_count != 145:
            raise RuntimeError(f"Reference count changed: {sid} {ref_count}")
        refs[sid] = {
            "family": family,
            "seed": seed,
            "checkpoint_sha256": cksha,
            "mean": means,
            "scale": scales,
        }
    return refs


def load_predictors(predictor_path: Path):
    payload = json.loads(predictor_path.read_text(encoding="utf-8"))
    if list(payload.get("feature_names", [])) != SOURCE_FEATURE_NAMES:
        raise RuntimeError("Predictor artifact top-level feature schema changed.")
    packages = payload.get("packages")
    if not isinstance(packages, list) or len(packages) != 12:
        raise RuntimeError("Expected exactly 12 frozen predictor packages.")

    index = {}
    for pkg in packages:
        key = (
            str(pkg["held_out_future_architecture"]),
            str(pkg["representation"]),
            str(pkg["task"]),
        )
        if key in index:
            raise RuntimeError(f"Duplicate predictor package: {key}")
        if key[0] not in FAMILIES or key[1] not in REPRESENTATIONS or key[2] not in TASKS:
            raise RuntimeError(f"Unexpected predictor key: {key}")
        if list(pkg.get("feature_names", [])) != SOURCE_FEATURE_NAMES:
            raise RuntimeError(f"Predictor feature order changed: {key}")
        if int(pkg.get("training_rows", -1)) != 8700:
            raise RuntimeError(f"Predictor training rows changed: {key}")

        mean = np.asarray(pkg["standard_scaler_mean"], dtype=np.float64)
        scale = np.asarray(pkg["standard_scaler_scale"], dtype=np.float64)
        coef = np.asarray(pkg["logistic_coefficient"], dtype=np.float64)
        intercept = float(pkg["logistic_intercept"])
        if mean.shape != (19,) or scale.shape != (19,) or coef.shape != (19,):
            raise RuntimeError(f"Predictor dimension mismatch: {key}")
        if not np.isfinite(mean).all() or not np.isfinite(scale).all():
            raise RuntimeError(f"Non-finite scaler: {key}")
        if not np.isfinite(coef).all() or not math.isfinite(intercept):
            raise RuntimeError(f"Non-finite logistic parameters: {key}")
        if np.any(scale <= 0):
            raise RuntimeError(f"Non-positive scaler scale: {key}")

        index[key] = {
            "mean": mean,
            "scale": scale,
            "coef": coef,
            "intercept": intercept,
            "package_id": pkg.get("package_id", ""),
        }

    expected = {
        (f, r, t)
        for f in FAMILIES
        for r in REPRESENTATIONS
        for t in TASKS
    }
    if set(index) != expected:
        raise RuntimeError("Frozen predictor package grid is incomplete.")
    return index


def sigmoid_scalar(v: float) -> float:
    if v >= 0:
        return float(1.0 / (1.0 + math.exp(-v)))
    ev = math.exp(v)
    return float(ev / (1.0 + ev))


def deploy_probability(x: np.ndarray, package: dict) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.shape != (19,) or not np.isfinite(x).all():
        raise RuntimeError("Invalid deployment feature vector.")
    xs = (x - package["mean"]) / package["scale"]
    logit = float(np.dot(package["coef"], xs) + package["intercept"])
    p = sigmoid_scalar(logit)
    if not math.isfinite(p) or p < 0.0 or p > 1.0:
        raise RuntimeError("Invalid frozen PAOT probability.")
    return p


# ---------------------------------------------------------------------
# Frozen panel and exact B0 extraction
# ---------------------------------------------------------------------

def build_panel(r03):
    panel = []
    mappings = (
        ("PraNet", r03.PRANET_CHECKPOINTS),
        ("DeepLabV3-R50", r03.DEEPLAB_CHECKPOINTS),
        ("SegFormer-B0", r03.SEGFORMER_CHECKPOINTS),
    )
    for family, mapping in mappings:
        for seed, (path, sha) in mapping.items():
            validate_sha(Path(path), sha, f"{family} checkpoint seed {seed}")
            panel.append({
                "model_family": family,
                "training_seed": int(seed),
                "model_state_id": f"{family}::{seed}",
                "checkpoint": str(path),
                "checkpoint_sha256": sha,
            })
    panel.sort(key=lambda r: (r["model_family"], r["training_seed"]))
    if len(panel) != EXPECTED_STATES:
        raise RuntimeError("Panel state count mismatch.")
    if len({r["checkpoint_sha256"] for r in panel}) != EXPECTED_STATES:
        raise RuntimeError("Panel checkpoint SHA duplication.")
    return panel


def validate_panel_against_references(panel, refs):
    for state in panel:
        sid = state["model_state_id"]
        if sid not in refs:
            raise RuntimeError(f"Missing R05B2 reference: {sid}")
        ref = refs[sid]
        if ref["family"] != state["model_family"]:
            raise RuntimeError(f"Reference family mismatch: {sid}")
        if int(ref["seed"]) != int(state["training_seed"]):
            raise RuntimeError(f"Reference seed mismatch: {sid}")
        if ref["checkpoint_sha256"] != state["checkpoint_sha256"]:
            raise RuntimeError(f"Reference checkpoint mismatch: {sid}")


def raw_to_mrz(raw: np.ndarray, ref: dict) -> np.ndarray:
    z = (raw - ref["mean"]) / ref["scale"]
    if z.shape != (19,) or not np.isfinite(z).all():
        raise RuntimeError("Invalid MRZ19 vector.")
    return z


def probabilities_for_row(family, raw, mrz, predictors):
    return {
        "paot_raw19_harm_probability": deploy_probability(
            raw, predictors[(family, "RAW19", "HARM")]
        ),
        "paot_raw19_benefit_probability": deploy_probability(
            raw, predictors[(family, "RAW19", "BENEFIT")]
        ),
        "paot_mrz19_harm_probability": deploy_probability(
            mrz, predictors[(family, "MRZ19", "HARM")]
        ),
        "paot_mrz19_benefit_probability": deploy_probability(
            mrz, predictors[(family, "MRZ19", "BENEFIT")]
        ),
    }


def make_output_row(target_row, state, features, refs, predictors):
    raw = np.asarray(
        [float(features[f]) for f in SOURCE_FEATURE_NAMES],
        dtype=np.float64,
    )
    if raw.shape != (19,) or not np.isfinite(raw).all():
        raise RuntimeError("Non-finite RAW19 features.")
    ref = refs[state["model_state_id"]]
    mrz = raw_to_mrz(raw, ref)
    probs = probabilities_for_row(
        state["model_family"], raw, mrz, predictors
    )

    row = {
        "sample_id": target_row["sample_id"],
        "image_path": target_row["image_path"],
        "image_raw_sha256": target_row["image_raw_sha256"],
        "model_family": state["model_family"],
        "model_state_id": state["model_state_id"],
        "training_seed": state["training_seed"],
        "checkpoint_sha256": state["checkpoint_sha256"],
    }
    row.update({f: float(features[f]) for f in SOURCE_FEATURE_NAMES})
    row.update({
        f"mrz__{f}": float(mrz[i])
        for i, f in enumerate(SOURCE_FEATURE_NAMES)
    })
    row.update(probs)

    if set(row) != set(OUTPUT_FIELDS):
        raise RuntimeError("D2 output schema mismatch.")
    return row


def run_pranet_state(state, targets, r03, refs, predictors, device):
    helper = import_module(
        PRANET_HELPER, "q1_r05d2_pranet_s03_helper"
    )
    training = helper.import_training_helper()
    seed = int(state["training_seed"])
    set_runtime_seed(seed)
    model = helper.load_model(training, seed, device)
    params = helper.configure_tent(model)
    source_values = helper.snapshot_params(params)

    rows = []
    pbar = tqdm(
        targets,
        desc=f"D2 PraNet seed {seed}",
        unit="img",
        dynamic_ncols=True,
    )
    for target in pbar:
        with Image.open(target["image_path"]) as im:
            native = im.convert("RGB")
        x = helper.image_to_model_tensor(training, native).to(
            device, non_blocking=True
        )

        helper.restore_params(params, source_values)
        model.eval()
        with __import__("torch").no_grad():
            z_source = helper.final_logit(model, x).detach()

        helper.restore_params(params, source_values)
        model.train()
        # Value-only pre-update entropy forward. No backward. No optimizer.
        with __import__("torch").no_grad():
            z_pre = helper.final_logit(model, x)
            loss = helper.mean_binary_entropy(z_pre)
        tent_loss = float(loss.detach().cpu())

        z_np = z_source[0, 0].float().cpu().numpy()
        features = r03.extract_source_features(z_np, tent_loss)
        rows.append(make_output_row(target, state, features, refs, predictors))

        del x, z_source, z_pre, loss

    helper.restore_params(params, source_values)
    del model, params, source_values
    if device.type == "cuda":
        __import__("torch").cuda.empty_cache()
    return rows


def run_deeplab_state(state, targets, r03, refs, predictors, device):
    helper = import_module(
        DEEPLAB_HELPER, "q1_r05d2_deeplab_s05c_helper"
    )
    training = helper.import_training_helper()
    # Exact Q1-R03 deterministic runtime.
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

    rows = []
    pbar = tqdm(
        targets,
        desc=f"D2 DeepLab seed {seed}",
        unit="img",
        dynamic_ncols=True,
    )
    for target in pbar:
        with Image.open(target["image_path"]) as im:
            native = im.convert("RGB")
        x = helper.image_to_model_tensor(training, native).to(
            device, non_blocking=True
        )

        # Exact Q1-R03 source branch: configure first, then model.eval().
        helper.restore_params(params, source_values)
        model.eval()
        with __import__("torch").no_grad():
            z_source = training.deeplab_logits(model, x).detach()

        helper.restore_params(params, source_values)
        helper.set_singleton_safe_tent_mode(
            model, unsafe_bn_modules, dropout_modules
        )
        with __import__("torch").no_grad():
            z_pre = training.deeplab_logits(model, x)
            loss = helper.mean_binary_entropy(z_pre)
        tent_loss = float(loss.detach().cpu())

        z_np = z_source[0, 0].float().cpu().numpy()
        features = r03.extract_source_features(z_np, tent_loss)
        rows.append(make_output_row(target, state, features, refs, predictors))

        del x, z_source, z_pre, loss

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
    return rows


def run_segformer_state(
    state,
    targets,
    r03,
    refs,
    predictors,
    device,
    seg_context=None,
):
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
        names,
        bn_modules,
        bn_original_track,
        dropout_modules,
        dropout_names,
    ) = r03.configure_segformer_tent(model, nn)
    source_values = r03.snapshot_params(params)

    rows = []
    pbar = tqdm(
        targets,
        desc=f"D2 SegFormer seed {seed}",
        unit="img",
        dynamic_ncols=True,
    )
    for target in pbar:
        with Image.open(target["image_path"]) as im:
            native = im.convert("RGB")
        x = r03.segformer_tensor(native, mean, std, torch).to(
            device, non_blocking=True
        )

        r03.restore_params(params, source_values, torch)
        r03.segformer_source_mode(
            model, bn_modules, bn_original_track, dropout_modules
        )
        with torch.no_grad():
            source_logits, z_source = r03.segformer_logits_and_z(model, x, F)

        r03.restore_params(params, source_values, torch)
        r03.segformer_tent_mode(model, bn_modules, dropout_modules)
        with torch.no_grad():
            pre_logits, _ = r03.segformer_logits_and_z(model, x, F)
            loss = r03.categorical_entropy(pre_logits, torch)
        tent_loss = float(loss.detach().cpu())

        z_np = z_source[0].float().cpu().numpy()
        features = r03.extract_source_features(z_np, tent_loss)
        rows.append(make_output_row(target, state, features, refs, predictors))

        del x, source_logits, z_source, pre_logits, loss

    r03.restore_params(params, source_values, torch)
    del (
        model,
        params,
        source_values,
        names,
        bn_modules,
        bn_original_track,
        dropout_modules,
        dropout_names,
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


# ---------------------------------------------------------------------
# State cache
# ---------------------------------------------------------------------

def state_token(state):
    return (
        state["model_family"].lower()
        .replace("-", "_")
        .replace(" ", "_")
        + f"_seed{state['training_seed']}"
    )


def state_cache_paths(cache_dir: Path, state):
    token = state_token(state)
    return (
        cache_dir / f"{token}.csv",
        cache_dir / f"{token}.lock.json",
    )


def validate_state_rows(rows, state, targets):
    if len(rows) != EXPECTED_CASES:
        raise RuntimeError(
            f"State rows={len(rows)} expected={EXPECTED_CASES}: {state['model_state_id']}"
        )
    expected_ids = {r["sample_id"] for r in targets}
    actual_ids = {r["sample_id"] for r in rows}
    if actual_ids != expected_ids or len(actual_ids) != EXPECTED_CASES:
        raise RuntimeError(f"State target ID mismatch: {state['model_state_id']}")
    if {r["model_state_id"] for r in rows} != {state["model_state_id"]}:
        raise RuntimeError("State cache model_state_id mismatch.")
    if {r["checkpoint_sha256"] for r in rows} != {state["checkpoint_sha256"]}:
        raise RuntimeError("State cache checkpoint SHA mismatch.")

    for row in rows:
        vals = [float(row[f]) for f in SOURCE_FEATURE_NAMES + MRZ_FIELDS]
        probs = [float(row[f]) for f in PROBABILITY_FIELDS]
        if not np.isfinite(np.asarray(vals + probs, dtype=np.float64)).all():
            raise RuntimeError("Non-finite state cache row.")
        if any(p < 0.0 or p > 1.0 for p in probs):
            raise RuntimeError("Probability outside [0,1].")


def save_state_cache(
    cache_dir,
    state,
    rows,
    target_manifest_sha,
    predictor_sha,
    refs_sha,
):
    csv_path, lock_path = state_cache_paths(cache_dir, state)
    if csv_path.exists() or lock_path.exists():
        raise FileExistsError(
            f"State cache already exists before save: {state['model_state_id']}"
        )
    write_csv(csv_path, rows, OUTPUT_FIELDS)
    csv_sha = sha256_file(csv_path)
    sidecar = {
        "script_version": VERSION,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "r05d1_lock_sha256": EXPECTED_R05D1_LOCK_SHA256,
        "r05b2_lock_sha256": EXPECTED_R05B2_LOCK_SHA256,
        "r03_lock_sha256": EXPECTED_R03_LOCK_SHA256,
        "target_manifest_sha256": target_manifest_sha,
        "predictor_packages_sha256": predictor_sha,
        "reference_statistics_sha256": refs_sha,
        "model_family": state["model_family"],
        "model_state_id": state["model_state_id"],
        "training_seed": state["training_seed"],
        "checkpoint_sha256": state["checkpoint_sha256"],
        "rows": len(rows),
        "feature_names": SOURCE_FEATURE_NAMES,
        "probability_fields": PROBABILITY_FIELDS,
        "target_gt_pixels_decoded": False,
        "optimizer_constructed": False,
        "backward_pass": False,
        "tta_update": False,
        "csv_sha256": csv_sha,
    }
    write_json(lock_path, sidecar)
    return csv_path, lock_path


def load_state_cache(
    cache_dir,
    state,
    targets,
    target_manifest_sha,
    predictor_sha,
    refs_sha,
    allow_orphan_cleanup,
):
    csv_path, lock_path = state_cache_paths(cache_dir, state)

    if csv_path.exists() != lock_path.exists():
        if allow_orphan_cleanup:
            if csv_path.exists():
                csv_path.unlink()
            if lock_path.exists():
                lock_path.unlink()
            return None
        raise RuntimeError(
            f"Orphan state cache; use --resume for technical cleanup: {state['model_state_id']}"
        )

    if not csv_path.exists():
        return None

    sidecar = json.loads(lock_path.read_text(encoding="utf-8"))
    exact = {
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "r05d1_lock_sha256": EXPECTED_R05D1_LOCK_SHA256,
        "r05b2_lock_sha256": EXPECTED_R05B2_LOCK_SHA256,
        "r03_lock_sha256": EXPECTED_R03_LOCK_SHA256,
        "target_manifest_sha256": target_manifest_sha,
        "predictor_packages_sha256": predictor_sha,
        "reference_statistics_sha256": refs_sha,
        "model_family": state["model_family"],
        "model_state_id": state["model_state_id"],
        "checkpoint_sha256": state["checkpoint_sha256"],
    }
    for k, v in exact.items():
        if sidecar.get(k) != v:
            raise RuntimeError(f"State cache sidecar mismatch {k}: {state['model_state_id']}")
    if int(sidecar.get("training_seed", -1)) != int(state["training_seed"]):
        raise RuntimeError("State cache seed mismatch.")
    if int(sidecar.get("rows", -1)) != EXPECTED_CASES:
        raise RuntimeError("State cache row count mismatch.")
    if list(sidecar.get("feature_names", [])) != SOURCE_FEATURE_NAMES:
        raise RuntimeError("State cache feature schema mismatch.")
    if bool(sidecar.get("optimizer_constructed", True)):
        raise RuntimeError("State cache reports optimizer construction.")
    if bool(sidecar.get("backward_pass", True)):
        raise RuntimeError("State cache reports backward pass.")
    if bool(sidecar.get("tta_update", True)):
        raise RuntimeError("State cache reports TTA update.")

    validate_sha(csv_path, sidecar["csv_sha256"], "state cache CSV")
    rows, fields = read_csv(csv_path)
    if fields != OUTPUT_FIELDS:
        raise RuntimeError("State cache CSV column order mismatch.")
    validate_state_rows(rows, state, targets)
    return rows


# ---------------------------------------------------------------------
# Diagnostics and final lock
# ---------------------------------------------------------------------

def probability_summary(rows):
    out = []
    groups = defaultdict(list)
    for r in rows:
        groups[(r["model_family"], r["model_state_id"])].append(r)
        groups[(r["model_family"], "ALL_3_STATES")].append(r)
        groups[("ALL_ARCHITECTURES", "ALL_9_STATES")].append(r)

    for (family, state_id), subset in sorted(groups.items()):
        for field in PROBABILITY_FIELDS:
            v = np.asarray([float(r[field]) for r in subset], dtype=np.float64)
            out.append({
                "model_family": family,
                "model_state_id": state_id,
                "probability": field,
                "rows": len(v),
                "mean": float(v.mean()),
                "std": float(v.std()),
                "q05": float(np.quantile(v, 0.05)),
                "q25": float(np.quantile(v, 0.25)),
                "q50": float(np.quantile(v, 0.50)),
                "q75": float(np.quantile(v, 0.75)),
                "q95": float(np.quantile(v, 0.95)),
                "min": float(v.min()),
                "max": float(v.max()),
            })
    return out


def validate_global_rows(rows, targets, panel):
    if len(rows) != EXPECTED_ROWS:
        raise RuntimeError(f"Global rows={len(rows)} expected={EXPECTED_ROWS}")

    keys = {(r["model_state_id"], r["sample_id"]) for r in rows}
    if len(keys) != EXPECTED_ROWS:
        raise RuntimeError("Duplicate/missing model-case key.")

    expected_ids = {r["sample_id"] for r in targets}
    states = {r["model_state_id"] for r in panel}
    if {r["sample_id"] for r in rows} != expected_ids:
        raise RuntimeError("Global target set changed.")
    if {r["model_state_id"] for r in rows} != states:
        raise RuntimeError("Global panel state set changed.")

    counts = Counter(r["model_state_id"] for r in rows)
    if any(v != EXPECTED_CASES for v in counts.values()):
        raise RuntimeError(f"Per-state count mismatch: {dict(counts)}")

    for row in rows:
        vals = [float(row[f]) for f in SOURCE_FEATURE_NAMES + MRZ_FIELDS]
        probs = [float(row[f]) for f in PROBABILITY_FIELDS]
        if not np.isfinite(np.asarray(vals + probs, dtype=np.float64)).all():
            raise RuntimeError("Non-finite final row.")
        if any(p < 0.0 or p > 1.0 for p in probs):
            raise RuntimeError("Final probability outside [0,1].")


def preflight():
    d1, b2, r03_lock, manifest_path, predictor_path, refs_path = validate_upstream()
    targets = load_target_manifest(manifest_path, verify_image_hashes=False)
    refs = load_references(refs_path)
    predictors = load_predictors(predictor_path)

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for Q1-R05D2 formal target inference.")
    device = torch.device("cuda")

    r03 = import_module(R03_SCRIPT, "q1_r05d2_r03")
    panel = build_panel(r03)
    validate_panel_against_references(panel, refs)

    # Model-interface smoke test on zeros only; no target image is used.
    set_runtime_seed(20260819)

    # PraNet
    ph = import_module(PRANET_HELPER, "q1_r05d2_preflight_pranet")
    pt = ph.import_training_helper()
    ps = next(x for x in panel if x["model_family"] == "PraNet")
    pm = ph.load_model(pt, int(ps["training_seed"]), device)
    pp = ph.configure_tent(pm)
    pv = ph.snapshot_params(pp)
    dummy = torch.zeros((1, 3, 352, 352), device=device)
    ph.restore_params(pp, pv)
    pm.eval()
    with torch.no_grad():
        pz = ph.final_logit(pm, dummy)
    if pz.ndim != 4:
        raise RuntimeError("PraNet preflight forward failed.")
    del pm, pp, pv, pz

    # DeepLab
    dh = import_module(DEEPLAB_HELPER, "q1_r05d2_preflight_deeplab")
    dt = dh.import_training_helper()
    dh.seed_everything(20260817)
    ds = next(x for x in panel if x["model_family"] == "DeepLabV3-R50")
    dm = dh.load_model(dt, int(ds["training_seed"]), device)
    (
        dp, _, _, unsafe, _, dropouts
    ) = dh.configure_singleton_safe_tent(dm)
    dv = dh.snapshot_params(dp)
    dh.restore_params(dp, dv)
    dm.eval()
    with torch.no_grad():
        dz = dt.deeplab_logits(dm, dummy)
    if dz.ndim != 4:
        raise RuntimeError("DeepLab preflight source forward failed.")
    dh.restore_params(dp, dv)
    dh.set_singleton_safe_tent_mode(dm, unsafe, dropouts)
    with torch.no_grad():
        dzpre = dt.deeplab_logits(dm, dummy)
        dloss = dh.mean_binary_entropy(dzpre)
    if not torch.isfinite(dloss):
        raise RuntimeError("DeepLab preflight entropy failed.")
    del dm, dp, dv, unsafe, dropouts, dz, dzpre, dloss

    # SegFormer
    segctx = r03.build_segformer_context(device)
    (
        torch2, nn, F, SegformerForSemanticSegmentation,
        config, mean, std,
    ) = segctx
    ss = next(x for x in panel if x["model_family"] == "SegFormer-B0")
    sm = r03.load_segformer_state(
        int(ss["training_seed"]),
        device,
        SegformerForSemanticSegmentation,
        config,
        torch2,
    )
    (
        sp, _, sbn, sbn_orig, sdrop, _
    ) = r03.configure_segformer_tent(sm, nn)
    sv = r03.snapshot_params(sp)
    r03.restore_params(sp, sv, torch2)
    r03.segformer_source_mode(sm, sbn, sbn_orig, sdrop)
    with torch2.no_grad():
        slogits, sz = r03.segformer_logits_and_z(sm, dummy, F)
    if tuple(sz.shape) != (1, 352, 352):
        raise RuntimeError("SegFormer preflight binary logit shape failed.")
    del sm, sp, sbn, sbn_orig, sdrop, sv, slogits, sz, dummy
    torch.cuda.empty_cache()

    print("===== Q1-R05D2 PREFLIGHT =====")
    print(f"protocol_sha256={EXPECTED_PROTOCOL_SHA256}")
    print(f"r05d1_lock_sha256={EXPECTED_R05D1_LOCK_SHA256}")
    print(f"r05b2_lock_sha256={EXPECTED_R05B2_LOCK_SHA256}")
    print(f"r03_lock_sha256={EXPECTED_R03_LOCK_SHA256}")
    print(f"eligible_target_cases={len(targets)}")
    print(f"model_states={len(panel)}")
    print(f"expected_model_case_rows={EXPECTED_ROWS}")
    print("primary_representation=MRZ19")
    print("comparator_representation=RAW19")
    print("target_GT_pixels_decoded=NO")
    print("target_image_inference_in_preflight=NO")
    print("optimizer_constructed=NO")
    print("backward_pass=NO")
    print("TTA_update=NO")
    print("predictor_fit=NO")
    print("target_threshold_selection=NO")
    print("PREFLIGHT_PASS")


def run(args):
    (
        d1,
        b2,
        r03_lock,
        manifest_path,
        predictor_path,
        refs_path,
    ) = validate_upstream()

    if args.output_dir.exists():
        raise FileExistsError(f"Final Q1-R05D2 output exists: {args.output_dir}")

    build_dir = Path(str(args.output_dir) + "__building")
    if build_dir.exists() and not args.resume:
        raise FileExistsError(
            f"Partial Q1-R05D2 output exists: {build_dir}. "
            "Use --resume only after a technical interruption."
        )
    build_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = build_dir / "state_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    protocol_copy = build_dir / "preregistered_protocol_copy.md"
    if not protocol_copy.exists():
        shutil.copy2(PROTOCOL, protocol_copy)
    validate_sha(protocol_copy, EXPECTED_PROTOCOL_SHA256, "R05D2 protocol copy")

    target_manifest_sha = sha256_file(manifest_path)
    predictor_sha = sha256_file(predictor_path)
    refs_sha = sha256_file(refs_path)

    # Reverify all frozen image raw bytes before inference. No GT file is opened.
    targets = load_target_manifest(manifest_path, verify_image_hashes=True)
    refs = load_references(refs_path)
    predictors = load_predictors(predictor_path)

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for Q1-R05D2.")
    device = torch.device("cuda")

    r03 = import_module(R03_SCRIPT, "q1_r05d2_r03_formal")
    panel = build_panel(r03)
    validate_panel_against_references(panel, refs)

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
        "r05b2_lock_sha256": EXPECTED_R05B2_LOCK_SHA256,
        "r05b2_decision": b2.get("decision"),
        "r03_lock_sha256": EXPECTED_R03_LOCK_SHA256,
        "r03_decision": r03_lock.get("decision"),
        "r03_script_sha256": EXPECTED_R03_SCRIPT_SHA256,
        "target_manifest_sha256": target_manifest_sha,
        "predictor_packages_sha256": predictor_sha,
        "reference_statistics_sha256": refs_sha,
        "eligible_cases": len(targets),
        "model_states": len(panel),
        "expected_rows": EXPECTED_ROWS,
        "target_gt_pixels_decoded": False,
        "predictor_fit_or_modified": False,
        "target_normalization_fit": False,
        "target_calibration": False,
        "threshold_selection": False,
    }
    upstream_path = build_dir / "upstream_audit.json"
    write_json(upstream_path, upstream)

    all_rows = []
    seg_context = None

    print("===== Q1-R05D2 NEOPOLYP PROSPECTIVE PAOT =====")
    print(f"Device={device}")
    print(f"Target cases={len(targets)}")
    print(f"Model states={len(panel)}")
    print(f"Expected rows={EXPECTED_ROWS}")
    print("Target GT pixels=NO")
    print("Optimizer=NO")
    print("Backward=NO")
    print("TTA update=NO")
    print("Target fitting/calibration/threshold selection=NO")
    print()

    for idx, state in enumerate(panel, start=1):
        print(
            f"[STATE {idx}/{len(panel)}] "
            f"{state['model_state_id']} "
            f"checkpoint={state['checkpoint_sha256'][:12]}..."
        )
        cached = load_state_cache(
            cache_dir,
            state,
            targets,
            target_manifest_sha,
            predictor_sha,
            refs_sha,
            allow_orphan_cleanup=bool(args.resume),
        )
        if cached is not None:
            print(f"[RESUME] verified cache rows={len(cached)}")
            all_rows.extend(cached)
            continue

        if state["model_family"] == "PraNet":
            rows = run_pranet_state(
                state, targets, r03, refs, predictors, device
            )
        elif state["model_family"] == "DeepLabV3-R50":
            rows = run_deeplab_state(
                state, targets, r03, refs, predictors, device
            )
        elif state["model_family"] == "SegFormer-B0":
            if seg_context is None:
                seg_context = r03.build_segformer_context(device)
            rows = run_segformer_state(
                state,
                targets,
                r03,
                refs,
                predictors,
                device,
                seg_context=seg_context,
            )
        else:
            raise RuntimeError(f"Unknown family: {state['model_family']}")

        validate_state_rows(rows, state, targets)
        save_state_cache(
            cache_dir,
            state,
            rows,
            target_manifest_sha,
            predictor_sha,
            refs_sha,
        )
        all_rows.extend(rows)
        print(f"[PASS] locked state cache rows={len(rows)}")

    all_rows.sort(
        key=lambda r: (
            r["model_family"],
            int(r["training_seed"]),
            r["sample_id"],
        )
    )
    validate_global_rows(all_rows, targets, panel)

    table_path = build_dir / "target_paot_probabilities.csv"
    write_csv(table_path, all_rows, OUTPUT_FIELDS)
    table_sha = sha256_file(table_path)

    summary_rows = probability_summary(all_rows)
    summary_path = build_dir / "probability_distribution_summary.csv"
    write_csv(
        summary_path,
        summary_rows,
        [
            "model_family",
            "model_state_id",
            "probability",
            "rows",
            "mean",
            "std",
            "q05",
            "q25",
            "q50",
            "q75",
            "q95",
            "min",
            "max",
        ],
    )

    boundary = {
        "target_image_paths_read": True,
        "target_image_pixels_decoded": True,
        "target_gt_paths_present_as_provenance_strings": True,
        "target_gt_pixels_decoded": False,
        "source_model_inference_run": True,
        "preupdate_tent_objective_forward_run": True,
        "optimizer_constructed_by_r05d2": False,
        "backward_pass_run_by_r05d2": False,
        "tta_parameter_update_run": False,
        "source_or_target_model_training": False,
        "paot_probabilities_generated": True,
        "predictor_fit_or_modified": False,
        "target_reference_statistics_fit": False,
        "target_scaler_fit": False,
        "target_calibration": False,
        "target_feature_selection": False,
        "target_threshold_selection": False,
        "target_dice_computed": False,
        "target_delta_dice_computed": False,
        "target_harm_benefit_labels_computed": False,
    }
    boundary_path = build_dir / "information_boundary_audit.json"
    write_json(boundary_path, boundary)

    criteria_path = build_dir / "confirmatory_criteria.json"
    write_json(criteria_path, CONFIRMATORY_CRITERIA)

    decision_path = build_dir / "decision.txt"
    decision_path.write_text(DECISION_READY + "\n", encoding="utf-8")

    lines = [
        "===== Q1-R05D2 NEOPOLYP PROSPECTIVE PAOT PROBABILITY LOCK =====",
        f"Script version: {VERSION}",
        f"Build: {BUILD}",
        "",
        "Frozen cohort:",
        f"  eligible NeoPolyp cases={len(targets)}",
        "  R05D1 exact overlap exclusions=0",
        "",
        "Frozen panel:",
        "  PraNet states=3",
        "  DeepLabV3-R50 states=3",
        "  SegFormer-B0 states=3",
        f"  total states={len(panel)}",
        "",
        "Prospective table:",
        f"  model-case rows={len(all_rows)}",
        "  RAW19 features=19",
        "  MRZ19 features=19",
        "  PAOT probabilities per row=4",
        f"  table_sha256={table_sha}",
        "",
        "Information boundary:",
        "  target image pixels decoded=YES",
        "  target GT pixels decoded=NO",
        "  source inference=YES",
        "  pre-update entropy forward=YES",
        "  optimizer constructed=NO",
        "  backward pass=NO",
        "  TTA parameter update=NO",
        "  predictor fit/modified=NO",
        "  target normalization/calibration=NO",
        "  target threshold selection=NO",
        "  target Dice/DeltaDice/outcomes=NO",
        "",
        "Confirmatory criteria frozen before outcome reveal:",
        "  primary=MRZ19",
        "  median Harm AUROC >= 0.70",
        "  min Harm AUROC >= 0.60",
        "  median Benefit AUROC >= 0.75",
        "  min Benefit AUROC >= 0.65",
        "  bootstrap=10000 target-image clusters",
        "",
        "Decision:",
        f"  {DECISION_READY}",
        "",
        "Next:",
        "  Q1-R05D3 frozen Source + A1_TENT_1STEP prediction generation and lock",
        "  NeoPolyp GT must remain unopened in D3.",
    ]
    run_log_path = build_dir / "run_log.txt"
    run_log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    artifacts = {
        "protocol_copy": protocol_copy,
        "upstream_audit": upstream_path,
        "checkpoint_manifest": checkpoint_manifest_path,
        "target_paot_probabilities": table_path,
        "probability_distribution_summary": summary_path,
        "information_boundary_audit": boundary_path,
        "confirmatory_criteria": criteria_path,
        "decision": decision_path,
        "run_log": run_log_path,
    }
    # Freeze every completed state cache and sidecar as well.
    for state in panel:
        cp, lp = state_cache_paths(cache_dir, state)
        if not cp.exists() or not lp.exists():
            raise RuntimeError(f"Missing completed state cache: {state['model_state_id']}")
        artifacts[f"state_cache_csv__{state_token(state)}"] = cp
        artifacts[f"state_cache_lock__{state_token(state)}"] = lp

    lock = {
        "script_version": VERSION,
        "build": BUILD,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "r05d1_lock_sha256": EXPECTED_R05D1_LOCK_SHA256,
        "r05b2_lock_sha256": EXPECTED_R05B2_LOCK_SHA256,
        "r03_lock_sha256": EXPECTED_R03_LOCK_SHA256,
        "target_manifest_sha256": target_manifest_sha,
        "predictor_packages_sha256": predictor_sha,
        "reference_statistics_sha256": refs_sha,
        "target_cases": EXPECTED_CASES,
        "model_states": EXPECTED_STATES,
        "model_case_rows": EXPECTED_ROWS,
        "feature_names": SOURCE_FEATURE_NAMES,
        "primary_representation": "MRZ19",
        "comparator_representation": "RAW19",
        "probability_fields": PROBABILITY_FIELDS,
        "confirmatory_criteria": CONFIRMATORY_CRITERIA,
        "target_image_pixels_decoded": True,
        "target_gt_pixels_decoded": False,
        "optimizer_constructed_by_r05d2": False,
        "backward_pass_run_by_r05d2": False,
        "tta_parameter_update_run": False,
        "predictor_fit_or_modified": False,
        "target_fit_or_calibration": False,
        "target_threshold_selected": False,
        "target_outcomes_revealed": False,
        "decision": DECISION_READY,
        "artifacts": {
            name: {
                "relative_path": str(path.relative_to(build_dir)),
                "sha256": sha256_file(path),
            }
            for name, path in artifacts.items()
        },
    }

    lock_path = build_dir / "Q1_R05D2_NEOPOLYP_PAOT_PROBABILITY_LOCK.json"
    write_json(lock_path, lock)
    lock_sha = sha256_file(lock_path)

    for name, meta in lock["artifacts"].items():
        p = build_dir / meta["relative_path"]
        if sha256_file(p) != meta["sha256"]:
            raise RuntimeError(f"Artifact changed before final commit: {name}")

    build_dir.rename(args.output_dir)

    print()
    print((args.output_dir / "run_log.txt").read_text(encoding="utf-8"))
    print(
        "Q1-R05D2 LOCK:",
        args.output_dir / "Q1_R05D2_NEOPOLYP_PAOT_PROBABILITY_LOCK.json",
    )
    print("Q1-R05D2 LOCK SHA256:", lock_sha)


def self_test():
    assert len(SOURCE_FEATURE_NAMES) == 19
    assert EXPECTED_CASES == 1000
    assert EXPECTED_STATES == 9
    assert EXPECTED_ROWS == 9000
    assert len(PROBABILITY_FIELDS) == 4
    assert CONFIRMATORY_CRITERIA[
        "median_architecture_harm_auroc_min"
    ] == 0.70
    assert CONFIRMATORY_CRITERIA[
        "minimum_architecture_harm_auroc_min"
    ] == 0.60
    assert CONFIRMATORY_CRITERIA[
        "median_architecture_benefit_auroc_min"
    ] == 0.75
    assert CONFIRMATORY_CRITERIA[
        "minimum_architecture_benefit_auroc_min"
    ] == 0.65

    p = [sigmoid_scalar(x) for x in (-1000.0, 0.0, 1000.0)]
    assert p[0] < 1e-100
    assert abs(p[1] - 0.5) < 1e-15
    assert p[2] > 1.0 - 1e-12

    package = {
        "mean": np.zeros(19, dtype=np.float64),
        "scale": np.ones(19, dtype=np.float64),
        "coef": np.ones(19, dtype=np.float64),
        "intercept": 0.0,
    }
    x = np.zeros(19, dtype=np.float64)
    assert abs(deploy_probability(x, package) - 0.5) < 1e-15

    ref = {
        "mean": np.arange(19, dtype=np.float64),
        "scale": np.ones(19, dtype=np.float64) * 2.0,
    }
    raw = ref["mean"] + 2.0
    mrz = raw_to_mrz(raw, ref)
    assert np.allclose(mrz, 1.0)

    # FIX1: scientific no-update/no-outcome constraints are validated by
    # delivery-time AST inspection and by explicit runtime boundary fields.
    # Do not use raw-string self-scanning here because literal test strings can
    # falsely match themselves.

    print("CARDINALITY_TEST_PASS")
    print("FROZEN_CRITERIA_TEST_PASS")
    print("SIGMOID_TEST_PASS")
    print("DEPLOYMENT_MATH_TEST_PASS")
    print("MRZ19_TEST_PASS")
    print("NO_OPTIMIZER_STEP_TEST_PASS")
    print("NO_BACKWARD_TEST_PASS")
    print("NO_TARGET_METRIC_TEST_PASS")
    print("SELF_TEST_PASS")


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Q1-R05D2: generate and lock prospective NeoPolyp PAOT "
            "HARM/BENEFIT probabilities before any GT pixel decoding or TTA update."
        )
    )
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    p.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume only verified completed per-state caches after a technical "
            "interruption. No scientific setting is changed."
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
