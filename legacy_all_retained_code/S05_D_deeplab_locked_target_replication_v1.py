#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
S05-D: DeepLabV3 second-backbone locked target replication.

PHASE A:
  target images only -> Source/TENT logits -> 24 features -> frozen gate
  -> SHA lock. No target masks or target metrics.

PHASE B:
  only after Phase-A lock -> open masks -> evaluate locked predictions/decisions.

No model fitting, no threshold calibration, no feature selection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from collections import Counter
from pathlib import Path

ROOT = Path(r"F:\MEDSEG_SAFETTA")
os.environ["TORCH_HOME"] = str(ROOT / "assets" / "torchvision_cache")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from tqdm import tqdm


VERSION = "2026-08-18-S05-D-v1"
BUILD = "S05_D_DEEPLAB_NO_LABEL_LOCK_THEN_TARGET_REPLICATION"

DATA_ROOT = ROOT / "data" / "processed" / "S00_polyp_locked_v1"

MANIFEST = ROOT / "data" / "splits" / "S01_locked_protocol_manifest_v1.csv"
EXPECTED_MANIFEST_SHA256 = "afb4bc1175af004819538cdbb22ee7a10f1be48035945c7e0d50fd2bbe2e3dc7"

PROTOCOL = ROOT / "docs" / "S05_D_deeplab_locked_target_replication_protocol_v1.md"
EXPECTED_PROTOCOL_SHA256 = "08bd2f6c7cb1965db5061e004b1e359e822f1386ba050952bfd8ad94383d819a"

S05C_HELPER = (
    ROOT / "code"
    / "S05_C_build_deeplab_source_side_safettta_gate_v1.py"
)
EXPECTED_S05C_HELPER_SHA256 = "ab1d98f9847b2e338f07ec26ff426f9cd428e68da4813b90e2227782319aa424"

GATE_ARTIFACT = (
    ROOT / "outputs" / "S05_C_deeplab_source_side_safettta_gate_v1"
    / "SafeTTA_v1_DEEPLAB_SOURCE_LOCKED_GATE.json"
)
EXPECTED_GATE_SHA256 = "cfa1d67d43d291bff32c243ebfc6b98aedc873c18fcafd901961a4b779b11b31"
EXPECTED_TAU = 0.3938344633

OUTPUT_DIR = (
    ROOT / "outputs"
    / "S05_D_deeplab_locked_safettta_target_replication_v1"
)

SEEDS = (20260817, 20260818, 20260819)
IMAGE_SIZE = 352

CHECKPOINTS = {
    20260817: {
        "path": ROOT / "outputs"
        / "S05_B_deeplabv3_r50_source_only_seed20260817_v1"
        / "checkpoints" / "best_source_val_dice.pt",
        "sha256": "d63f914e337295652627c773332d8e7382fbcd6a39d69805e1316dcc759605a2",
    },
    20260818: {
        "path": ROOT / "outputs"
        / "S05_B_deeplabv3_r50_source_only_seed20260818_v1"
        / "checkpoints" / "best_source_val_dice.pt",
        "sha256": "d4ede45d5b30f62fdbc92cd9aa8fc84e0d5a52073d5ef2c3ac36baa09a56eb25",
    },
    20260819: {
        "path": ROOT / "outputs"
        / "S05_B_deeplabv3_r50_source_only_seed20260819_v1"
        / "checkpoints" / "best_source_val_dice.pt",
        "sha256": "66aeae338d350db5aa878bcaab2e68bbe98eec8e626e28a1bb02b6708fa358eb",
    },
}

EXPECTED_DATASET_COUNTS = {
    ("seen_sanity", "Kvasir-SEG"): 100,
    ("seen_sanity", "CVC-ClinicDB"): 62,
    ("unseen_locked", "CVC-ColonDB"): 380,
    ("unseen_locked", "CVC-300"): 60,
    ("unseen_locked", "ETIS-LaribPolypDB"): 196,
}
EXPECTED_EVAL_IMAGES = 798
EXPECTED_UNSEEN_IMAGES = 636
EXPECTED_UNSEEN_ROWS = 1908

UNSEEN_DOMAINS = (
    "CVC-ColonDB",
    "CVC-300",
    "ETIS-LaribPolypDB",
)

HARM_THRESHOLD = -0.02
BENEFIT_THRESHOLD = +0.02

PASS_HARM_PREVENTION = 0.50
PASS_BENEFIT_RETENTION = 0.30
PASS_ACCEPTANCE_MIN = 0.10
PASS_ACCEPTANCE_MAX = 0.90

try:
    RESAMPLE_BILINEAR = Image.Resampling.BILINEAR
except AttributeError:
    RESAMPLE_BILINEAR = Image.BILINEAR


def file_sha256(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def read_csv(path: Path):
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return list(reader), reader.fieldnames or []


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def validate_exact_sha(path: Path, expected: str, label: str):
    if not path.exists():
        raise FileNotFoundError(path)
    actual = file_sha256(path)
    if actual.lower() != expected.lower():
        raise RuntimeError(
            f"{label} SHA mismatch.\nExpected: {expected}\nActual: {actual}"
        )
    return actual


def import_s05c_helper():
    validate_exact_sha(
        S05C_HELPER,
        EXPECTED_S05C_HELPER_SHA256,
        "S05-C helper",
    )
    spec = importlib.util.spec_from_file_location(
        "s05c_helper",
        str(S05C_HELPER),
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import frozen S05-C helper.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    needed = (
        "load_model",
        "configure_singleton_safe_tent",
        "snapshot_params",
        "source_and_tent_logits",
        "image_to_model_tensor",
        "extract_features",
        "FEATURE_NAMES",
    )
    for name in needed:
        if not hasattr(module, name):
            raise RuntimeError(f"S05-C helper missing {name}")
    return module


def load_eval_rows():
    validate_exact_sha(
        MANIFEST,
        EXPECTED_MANIFEST_SHA256,
        "S01 manifest",
    )
    rows, fields = read_csv(MANIFEST)

    required = {
        "sample_id",
        "dataset",
        "image_relpath",
        "mask_relpath",
        "s01_role",
    }
    missing = sorted(required - set(fields))
    if missing:
        raise RuntimeError(f"Manifest missing columns: {missing}")

    eval_rows = [
        r for r in rows
        if r["s01_role"] in {"seen_sanity", "unseen_locked"}
    ]

    if len(eval_rows) != EXPECTED_EVAL_IMAGES:
        raise RuntimeError(
            f"Eval image count mismatch: expected={EXPECTED_EVAL_IMAGES}, "
            f"actual={len(eval_rows)}"
        )

    counts = Counter(
        (r["s01_role"], r["dataset"])
        for r in eval_rows
    )
    if counts != Counter(EXPECTED_DATASET_COUNTS):
        raise RuntimeError(
            f"Eval role/dataset counts mismatch.\n"
            f"Expected: {EXPECTED_DATASET_COUNTS}\nActual: {dict(counts)}"
        )

    ids = [r["sample_id"] for r in eval_rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate sample_id among eval rows.")

    return sorted(eval_rows, key=lambda r: r["sample_id"])


def validate_checkpoints():
    out = {}
    for seed in SEEDS:
        out[seed] = validate_exact_sha(
            CHECKPOINTS[seed]["path"],
            CHECKPOINTS[seed]["sha256"],
            f"DeepLab checkpoint seed {seed}",
        )
    return out


def load_gate(helper):
    validate_exact_sha(
        GATE_ARTIFACT,
        EXPECTED_GATE_SHA256,
        "S05-C gate artifact",
    )

    gate = json.loads(GATE_ARTIFACT.read_text(encoding="utf-8"))

    required = {
        "feature_names",
        "scaler_mean",
        "scaler_scale",
        "logistic_coef",
        "logistic_intercept",
        "harm_threshold_tau",
        "singleton_unsafe_bn_names",
        "dropout_eval_names",
        "checkpoint_sha256",
    }
    missing = sorted(required - set(gate))
    if missing:
        raise RuntimeError(f"Gate artifact missing fields: {missing}")

    feature_names = list(gate["feature_names"])
    if feature_names != list(helper.FEATURE_NAMES):
        raise RuntimeError("Gate feature order differs from frozen S05-C helper.")
    if len(feature_names) != 24:
        raise RuntimeError("Expected exactly 24 gate features.")

    mean = np.asarray(gate["scaler_mean"], dtype=np.float64)
    scale = np.asarray(gate["scaler_scale"], dtype=np.float64)
    coef = np.asarray(gate["logistic_coef"], dtype=np.float64)
    intercept = float(gate["logistic_intercept"])
    tau = float(gate["harm_threshold_tau"])

    if not (
        len(mean) == len(scale) == len(coef) == len(feature_names)
    ):
        raise RuntimeError("Gate parameter dimension mismatch.")
    if np.any(~np.isfinite(mean)):
        raise RuntimeError("Non-finite gate scaler mean.")
    if np.any(~np.isfinite(scale)) or np.any(scale <= 0):
        raise RuntimeError("Invalid gate scaler scale.")
    if np.any(~np.isfinite(coef)) or not np.isfinite(intercept):
        raise RuntimeError("Non-finite logistic parameter.")
    if abs(tau - EXPECTED_TAU) > 5e-10:
        raise RuntimeError(
            f"Frozen tau mismatch: expected={EXPECTED_TAU}, actual={tau}"
        )

    expected_ck = {
        str(seed): CHECKPOINTS[seed]["sha256"]
        for seed in SEEDS
    }
    actual_ck = {
        str(k): str(v)
        for k, v in gate["checkpoint_sha256"].items()
    }
    if actual_ck != expected_ck:
        raise RuntimeError(
            f"Gate checkpoint provenance mismatch.\n"
            f"Expected: {expected_ck}\nActual: {actual_ck}"
        )

    if gate["singleton_unsafe_bn_names"] != ["classifier.0.convs.4.2"]:
        raise RuntimeError(
            "Frozen singleton-unsafe BN topology mismatch in gate artifact."
        )
    if gate["dropout_eval_names"] != ["classifier.0.project.3"]:
        raise RuntimeError(
            "Frozen Dropout topology mismatch in gate artifact."
        )

    return {
        "raw": gate,
        "feature_names": feature_names,
        "mean": mean,
        "scale": scale,
        "coef": coef,
        "intercept": intercept,
        "tau": tau,
    }


def stable_sigmoid_scalar(z: float) -> float:
    if z >= 0:
        return float(1.0 / (1.0 + math.exp(-z)))
    ez = math.exp(z)
    return float(ez / (1.0 + ez))


def apply_exported_gate(features: dict, gate: dict):
    x = np.asarray(
        [float(features[name]) for name in gate["feature_names"]],
        dtype=np.float64,
    )
    if np.any(~np.isfinite(x)):
        raise RuntimeError("Non-finite no-label target feature.")

    xs = (x - gate["mean"]) / gate["scale"]
    logit = float(np.dot(gate["coef"], xs) + gate["intercept"])
    p_harm = stable_sigmoid_scalar(logit)
    accepted = int(p_harm < gate["tau"])
    return p_harm, accepted


def seed_artifact_paths(no_label_dir: Path, seed: int):
    return {
        "npz": no_label_dir / f"seed{seed}_source_tent_logits.npz",
        "csv": no_label_dir / f"seed{seed}_features_and_gate.csv",
        "lock": no_label_dir / f"seed{seed}_NO_LABEL_LOCK.json",
    }


def validate_seed_no_label_lock(paths: dict, seed: int, eval_rows):
    if not paths["lock"].exists():
        return None

    try:
        lock = json.loads(paths["lock"].read_text(encoding="utf-8"))
    except Exception:
        return None

    required = {
        "seed",
        "npz_sha256",
        "csv_sha256",
        "rows",
        "target_masks_opened",
        "target_metrics_computed",
    }
    if not required.issubset(lock):
        return None

    if int(lock["seed"]) != seed:
        return None
    if int(lock["rows"]) != len(eval_rows):
        return None
    if bool(lock["target_masks_opened"]):
        return None
    if bool(lock["target_metrics_computed"]):
        return None

    if not paths["npz"].exists() or not paths["csv"].exists():
        return None
    if file_sha256(paths["npz"]) != lock["npz_sha256"]:
        return None
    if file_sha256(paths["csv"]) != lock["csv_sha256"]:
        return None

    csv_rows, fields = read_csv(paths["csv"])
    required_fields = {
        "seed", "sample_id", "role", "dataset",
        "p_harm", "tau", "tent_accepted", "decision",
    }
    if not required_fields.issubset(fields):
        return None
    if len(csv_rows) != len(eval_rows):
        return None

    expected_ids = [r["sample_id"] for r in eval_rows]
    actual_ids = [r["sample_id"] for r in csv_rows]
    if actual_ids != expected_ids:
        return None

    try:
        with np.load(paths["npz"], allow_pickle=False) as data:
            ids = data["sample_ids"].tolist()
            if ids != expected_ids:
                return None
            if data["source_logits"].shape != (
                len(eval_rows), IMAGE_SIZE, IMAGE_SIZE
            ):
                return None
            if data["tent_logits"].shape != (
                len(eval_rows), IMAGE_SIZE, IMAGE_SIZE
            ):
                return None
    except Exception:
        return None

    return lock


def produce_seed_no_label(
    helper,
    gate,
    eval_rows,
    seed,
    device,
    no_label_dir,
):
    paths = seed_artifact_paths(no_label_dir, seed)

    if any(p.exists() for p in paths.values()):
        raise RuntimeError(
            f"Partial no-label artifacts already exist for seed={seed}. "
            "Use --resume so they are verified rather than overwritten."
        )

    model = helper.load_model(helper, seed, device)

    (
        params,
        trainable_bn_names,
        unsafe_bn_names,
        unsafe_bn_modules,
        dropout_names,
        dropout_modules,
    ) = helper.configure_singleton_safe_tent(model)

    if unsafe_bn_names != ["classifier.0.convs.4.2"]:
        raise RuntimeError(
            f"Unexpected singleton-unsafe BN names: {unsafe_bn_names}"
        )
    if dropout_names != ["classifier.0.project.3"]:
        raise RuntimeError(
            f"Unexpected Dropout names: {dropout_names}"
        )

    source_values = helper.snapshot_params(params)

    source_logits = np.empty(
        (len(eval_rows), IMAGE_SIZE, IMAGE_SIZE),
        dtype=np.float16,
    )
    tent_logits = np.empty_like(source_logits)

    feature_rows = []

    pbar = tqdm(
        enumerate(eval_rows),
        total=len(eval_rows),
        desc=f"S05-D no-label seed {seed}",
        unit="img",
        dynamic_ncols=True,
    )

    for i, row in pbar:
        image_path = DATA_ROOT / Path(row["image_relpath"])
        if not image_path.exists():
            raise FileNotFoundError(image_path)

        # PHASE A opens image only. It never touches mask_relpath.
        with Image.open(image_path) as im:
            image = im.convert("RGB")

        x = helper.image_to_model_tensor(helper, image).to(
            device,
            non_blocking=True,
        )

        z_source, z_tent, adapt_loss = helper.source_and_tent_logits(
            helper=helper,
            model=model,
            params=params,
            source_values=source_values,
            unsafe_bn_modules=unsafe_bn_modules,
            dropout_modules=dropout_modules,
            x=x,
        )

        features = helper.extract_features(z_source, z_tent)
        p_harm, accepted = apply_exported_gate(features, gate)

        source_logits[i] = z_source.astype(np.float16)
        tent_logits[i] = z_tent.astype(np.float16)

        item = {
            "seed": seed,
            "sample_id": row["sample_id"],
            "role": row["s01_role"],
            "dataset": row["dataset"],
            "p_harm": f"{p_harm:.12f}",
            "tau": f"{gate['tau']:.12f}",
            "tent_accepted": accepted,
            "decision": (
                "ACCEPT_TENT"
                if accepted
                else "REJECT_TENT_USE_SOURCE"
            ),
            "tent_adapt_loss": f"{adapt_loss:.12f}",
        }
        for name in gate["feature_names"]:
            item[name] = f"{float(features[name]):.12f}"

        feature_rows.append(item)

        pbar.set_postfix(
            pH=f"{p_harm:.3f}",
            accept=accepted,
        )

    feature_fields = [
        "seed", "sample_id", "role", "dataset",
        "p_harm", "tau", "tent_accepted", "decision",
        "tent_adapt_loss",
    ] + gate["feature_names"]

    write_csv(paths["csv"], feature_rows, feature_fields)

    np.savez_compressed(
        paths["npz"],
        source_logits=source_logits,
        tent_logits=tent_logits,
        sample_ids=np.asarray(
            [r["sample_id"] for r in eval_rows],
            dtype=str,
        ),
        roles=np.asarray(
            [r["s01_role"] for r in eval_rows],
            dtype=str,
        ),
        datasets=np.asarray(
            [r["dataset"] for r in eval_rows],
            dtype=str,
        ),
        seed=np.asarray([seed], dtype=np.int64),
        gate_artifact_sha256=np.asarray(
            [EXPECTED_GATE_SHA256],
            dtype=str,
        ),
        protocol_sha256=np.asarray(
            [EXPECTED_PROTOCOL_SHA256],
            dtype=str,
        ),
    )

    lock = {
        "script_version": VERSION,
        "build": BUILD,
        "seed": seed,
        "rows": len(eval_rows),
        "npz_file": paths["npz"].name,
        "npz_sha256": file_sha256(paths["npz"]),
        "csv_file": paths["csv"].name,
        "csv_sha256": file_sha256(paths["csv"]),
        "checkpoint_sha256": CHECKPOINTS[seed]["sha256"],
        "gate_artifact_sha256": EXPECTED_GATE_SHA256,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "singleton_unsafe_bn_names": unsafe_bn_names,
        "dropout_eval_names": dropout_names,
        "target_masks_opened": False,
        "target_metrics_computed": False,
        "model_refit": False,
        "threshold_recalibrated": False,
    }
    paths["lock"].write_text(
        json.dumps(lock, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Verify before releasing seed.
    checked = validate_seed_no_label_lock(
        paths,
        seed,
        eval_rows,
    )
    if checked is None:
        raise RuntimeError(
            f"Failed to verify completed no-label seed lock: {seed}"
        )

    del (
        model,
        params,
        source_values,
        unsafe_bn_modules,
        dropout_modules,
        source_logits,
        tent_logits,
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return checked


def build_global_no_label_lock(
    no_label_dir,
    eval_rows,
    checkpoint_hashes,
):
    seed_locks = {}
    for seed in SEEDS:
        paths = seed_artifact_paths(no_label_dir, seed)
        lock = validate_seed_no_label_lock(
            paths,
            seed,
            eval_rows,
        )
        if lock is None:
            raise RuntimeError(
                f"Cannot form global lock; seed {seed} is not valid."
            )
        seed_locks[str(seed)] = {
            "seed_lock_file": paths["lock"].name,
            "seed_lock_sha256": file_sha256(paths["lock"]),
            "npz_sha256": lock["npz_sha256"],
            "csv_sha256": lock["csv_sha256"],
        }

    global_lock = {
        "script_version": VERSION,
        "build": BUILD,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "gate_artifact_sha256": EXPECTED_GATE_SHA256,
        "checkpoint_sha256": {
            str(k): v for k, v in checkpoint_hashes.items()
        },
        "eval_images": len(eval_rows),
        "image_seed_pairs": len(eval_rows) * len(SEEDS),
        "seed_locks": seed_locks,
        "target_masks_opened_before_global_lock": False,
        "target_metrics_computed_before_global_lock": False,
        "model_refit": False,
        "threshold_recalibrated": False,
    }

    path = no_label_dir / "DEEPLAB_TARGET_NO_LABEL_GLOBAL_LOCK.json"
    path.write_text(
        json.dumps(global_lock, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    sha = file_sha256(path)

    # Re-verify seed contents after global lock creation.
    for seed in SEEDS:
        if validate_seed_no_label_lock(
            seed_artifact_paths(no_label_dir, seed),
            seed,
            eval_rows,
        ) is None:
            raise RuntimeError(
                f"No-label artifact changed while forming global lock: {seed}"
            )

    return path, sha


def load_gt(row):
    path = DATA_ROOT / Path(row["mask_relpath"])
    if not path.exists():
        raise FileNotFoundError(path)

    with Image.open(path) as ma:
        arr = np.asarray(ma.convert("L"), dtype=np.float32)

    maxv = float(arr.max())
    if maxv <= 0:
        raise RuntimeError(f"Empty target mask: {row['sample_id']}")

    return (arr > 0.5 * maxv).astype(np.uint8)


def resize_logit(logit, shape_hw):
    t = torch.from_numpy(
        np.asarray(logit, dtype=np.float32)
    )[None, None]
    return F.interpolate(
        t,
        size=tuple(shape_hw),
        mode="bilinear",
        align_corners=False,
    )[0, 0].numpy()


def metrics_from_logit(logit, gt):
    pred = resize_logit(logit, gt.shape) >= 0.0
    target = gt.astype(bool)

    inter = np.logical_and(pred, target).sum()
    pred_n = pred.sum()
    target_n = target.sum()
    union = np.logical_or(pred, target).sum()

    dice = float(
        (2.0 * inter + 1e-7)
        / (pred_n + target_n + 1e-7)
    )
    iou = float(
        (inter + 1e-7)
        / (union + 1e-7)
    )
    return dice, iou


def evaluate_after_lock(
    no_label_dir,
    eval_rows,
    global_lock_sha,
):
    per_rows = []

    row_by_id = {
        r["sample_id"]: r
        for r in eval_rows
    }

    for seed in SEEDS:
        paths = seed_artifact_paths(no_label_dir, seed)

        lock = validate_seed_no_label_lock(
            paths,
            seed,
            eval_rows,
        )
        if lock is None:
            raise RuntimeError(
                f"Seed no-label lock invalid before Phase B: {seed}"
            )

        decision_rows, _ = read_csv(paths["csv"])
        decision_by_id = {
            r["sample_id"]: r
            for r in decision_rows
        }

        with np.load(paths["npz"], allow_pickle=False) as data:
            ids = data["sample_ids"].tolist()
            source_logits = data["source_logits"]
            tent_logits = data["tent_logits"]

            pbar = tqdm(
                range(len(ids)),
                desc=f"S05-D evaluate seed {seed}",
                unit="img",
                dynamic_ncols=True,
            )

            for i in pbar:
                sid = ids[i]
                row = row_by_id[sid]
                d = decision_by_id[sid]

                # FIRST target-mask access occurs here, after global no-label lock.
                gt = load_gt(row)

                source_dice, source_iou = metrics_from_logit(
                    source_logits[i],
                    gt,
                )
                tent_dice, tent_iou = metrics_from_logit(
                    tent_logits[i],
                    gt,
                )

                accepted = int(d["tent_accepted"]) == 1

                if accepted:
                    safe_dice = tent_dice
                    safe_iou = tent_iou
                else:
                    safe_dice = source_dice
                    safe_iou = source_iou

                oracle_dice = max(source_dice, tent_dice)

                tent_delta = tent_dice - source_dice
                safe_delta = safe_dice - source_dice

                tent_harmful = int(
                    tent_delta <= HARM_THRESHOLD
                )
                tent_beneficial = int(
                    tent_delta >= BENEFIT_THRESHOLD
                )
                safe_harmful = int(
                    safe_delta <= HARM_THRESHOLD
                )
                safe_beneficial = int(
                    safe_delta >= BENEFIT_THRESHOLD
                )

                per_rows.append({
                    "seed": seed,
                    "sample_id": sid,
                    "role": row["s01_role"],
                    "dataset": row["dataset"],
                    "p_harm": d["p_harm"],
                    "tau": d["tau"],
                    "tent_accepted": int(accepted),
                    "decision": d["decision"],
                    "source_dice": source_dice,
                    "source_iou": source_iou,
                    "tent_dice": tent_dice,
                    "tent_iou": tent_iou,
                    "safettta_dice": safe_dice,
                    "safettta_iou": safe_iou,
                    "oracle_dice": oracle_dice,
                    "tent_delta_dice": tent_delta,
                    "safettta_delta_dice": safe_delta,
                    "tent_harmful": tent_harmful,
                    "tent_beneficial": tent_beneficial,
                    "safettta_harmful": safe_harmful,
                    "safettta_beneficial": safe_beneficial,
                    "harm_prevented": int(
                        tent_harmful and not accepted
                    ),
                    "benefit_retained": int(
                        tent_beneficial and accepted
                    ),
                    "global_no_label_lock_sha256": global_lock_sha,
                })

                pbar.set_postfix(
                    src=f"{source_dice:.3f}",
                    tent=f"{tent_dice:.3f}",
                    safe=f"{safe_dice:.3f}",
                )

    if len(per_rows) != EXPECTED_EVAL_IMAGES * len(SEEDS):
        raise RuntimeError(
            f"Evaluation row count mismatch: {len(per_rows)}"
        )

    return per_rows


def summarize_group(rows, label):
    if not rows:
        raise RuntimeError(f"Empty summary group: {label}")

    n = len(rows)
    source = np.asarray(
        [float(r["source_dice"]) for r in rows],
        dtype=np.float64,
    )
    tent = np.asarray(
        [float(r["tent_dice"]) for r in rows],
        dtype=np.float64,
    )
    safe = np.asarray(
        [float(r["safettta_dice"]) for r in rows],
        dtype=np.float64,
    )
    oracle = np.asarray(
        [float(r["oracle_dice"]) for r in rows],
        dtype=np.float64,
    )

    accepted = np.asarray(
        [int(r["tent_accepted"]) for r in rows],
        dtype=np.int64,
    )
    tent_harm = np.asarray(
        [int(r["tent_harmful"]) for r in rows],
        dtype=np.int64,
    )
    safe_harm = np.asarray(
        [int(r["safettta_harmful"]) for r in rows],
        dtype=np.int64,
    )
    tent_benefit = np.asarray(
        [int(r["tent_beneficial"]) for r in rows],
        dtype=np.int64,
    )

    harm_n = int(tent_harm.sum())
    benefit_n = int(tent_benefit.sum())

    harm_prevention = (
        float(
            np.sum(
                (tent_harm == 1)
                & (accepted == 0)
            )
        ) / harm_n
        if harm_n > 0
        else float("nan")
    )

    benefit_retention = (
        float(
            np.sum(
                (tent_benefit == 1)
                & (accepted == 1)
            )
        ) / benefit_n
        if benefit_n > 0
        else float("nan")
    )

    return {
        "group": label,
        "n": n,
        "source_mean_dice": float(source.mean()),
        "tent_mean_dice": float(tent.mean()),
        "safettta_mean_dice": float(safe.mean()),
        "oracle_mean_dice": float(oracle.mean()),
        "tent_mean_delta": float((tent - source).mean()),
        "safettta_mean_delta": float((safe - source).mean()),
        "tent_harmful_n": harm_n,
        "tent_harmful_fraction": float(tent_harm.mean()),
        "safettta_harmful_n": int(safe_harm.sum()),
        "safettta_harmful_fraction": float(safe_harm.mean()),
        "harm_prevention_rate": harm_prevention,
        "tent_beneficial_n": benefit_n,
        "benefit_retention_rate": benefit_retention,
        "tent_acceptance_rate": float(accepted.mean()),
        "oracle_gap": float(oracle.mean() - safe.mean()),
    }


def build_summaries(per_rows):
    unseen = [
        r for r in per_rows
        if r["role"] == "unseen_locked"
    ]
    if len(unseen) != EXPECTED_UNSEEN_ROWS:
        raise RuntimeError(
            f"Unseen image-seed count mismatch: "
            f"expected={EXPECTED_UNSEEN_ROWS}, actual={len(unseen)}"
        )

    summaries = [
        summarize_group(unseen, "POOLED_UNSEEN")
    ]

    for domain in UNSEEN_DOMAINS:
        group = [
            r for r in unseen
            if r["dataset"] == domain
        ]
        expected = EXPECTED_DATASET_COUNTS[
            ("unseen_locked", domain)
        ] * len(SEEDS)

        if len(group) != expected:
            raise RuntimeError(
                f"Domain count mismatch {domain}: "
                f"expected={expected}, actual={len(group)}"
            )
        summaries.append(
            summarize_group(group, domain)
        )

    for seed in SEEDS:
        group = [
            r for r in unseen
            if int(r["seed"]) == seed
        ]
        if len(group) != EXPECTED_UNSEEN_IMAGES:
            raise RuntimeError(
                f"Seed unseen count mismatch {seed}: {len(group)}"
            )
        summaries.append(
            summarize_group(
                group,
                f"SEED_{seed}",
            )
        )

    return summaries


def frozen_replication_decision(pooled):
    pass_a = (
        pooled["safettta_mean_dice"] + 1e-12
        >= pooled["source_mean_dice"]
    )
    pass_b = (
        pooled["harm_prevention_rate"] + 1e-12
        >= PASS_HARM_PREVENTION
    )
    pass_c = (
        pooled["benefit_retention_rate"] + 1e-12
        >= PASS_BENEFIT_RETENTION
    )
    pass_d = (
        pooled["tent_acceptance_rate"] + 1e-12
        >= PASS_ACCEPTANCE_MIN
        and pooled["tent_acceptance_rate"] - 1e-12
        <= PASS_ACCEPTANCE_MAX
    )

    passed = pass_a and pass_b and pass_c and pass_d

    return {
        "A_safe_mean_ge_source": pass_a,
        "B_harm_prevention_ge_0_50": pass_b,
        "C_benefit_retention_ge_0_30": pass_c,
        "D_acceptance_in_0_10_0_90": pass_d,
        "all_four": passed,
        "decision": (
            "S05_SECOND_BACKBONE_STRONG_REPLICATION_PASS"
            if passed
            else "S05_SECOND_BACKBONE_STRONG_REPLICATION_NOT_MET"
        ),
    }


def run(args):
    validate_exact_sha(
        PROTOCOL,
        EXPECTED_PROTOCOL_SHA256,
        "S05-D protocol",
    )
    helper = import_s05c_helper()
    eval_rows = load_eval_rows()
    checkpoint_hashes = validate_checkpoints()
    gate = load_gate(helper)

    if args.output_dir.exists():
        raise FileExistsError(
            f"Final S05-D output already exists: {args.output_dir}"
        )

    build_dir = Path(str(args.output_dir) + "__building")

    if build_dir.exists() and not args.resume:
        raise FileExistsError(
            f"Partial S05-D build exists: {build_dir}\n"
            "Use --resume so completed no-label seed locks are verified."
        )

    build_dir.mkdir(parents=True, exist_ok=True)
    no_label_dir = build_dir / "no_label_locked"
    no_label_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available() and not args.cpu
        else "cpu"
    )

    print(f"[BUILD] {VERSION} | {BUILD}")
    print(f"Device: {device}")
    print(f"Eval images: {len(eval_rows)}")
    print(f"Frozen tau: {gate['tau']:.10f}")
    print("Phase A target masks opened=NO")
    print("Phase A target metrics computed=NO")
    print("Model refit=NO")
    print("Threshold recalibration=NO")
    print()

    # ---------------------- PHASE A ----------------------
    for seed in SEEDS:
        paths = seed_artifact_paths(no_label_dir, seed)

        existing = None
        if args.resume:
            existing = validate_seed_no_label_lock(
                paths,
                seed,
                eval_rows,
            )

        if existing is not None:
            print(
                f"[RESUME] verified seed {seed} no-label lock "
                f"NPZ={existing['npz_sha256'][:12]}..."
            )
            continue

        # If invalid partial artifacts exist, stop rather than silently replacing.
        if any(p.exists() for p in paths.values()):
            raise RuntimeError(
                f"Invalid partial no-label artifacts for seed={seed}. "
                "Preserve them for audit; do not overwrite silently."
            )

        produce_seed_no_label(
            helper=helper,
            gate=gate,
            eval_rows=eval_rows,
            seed=seed,
            device=device,
            no_label_dir=no_label_dir,
        )

    global_lock_path, global_lock_sha = build_global_no_label_lock(
        no_label_dir,
        eval_rows,
        checkpoint_hashes,
    )

    print()
    print("===== S05-D DEEPLAB TARGET NO-LABEL GLOBAL LOCK =====")
    print(f"Images={len(eval_rows)}")
    print(f"Image-seed pairs={len(eval_rows) * len(SEEDS)}")
    print(f"Global lock SHA256={global_lock_sha}")
    print("Target masks opened before global lock=NO")
    print("Target metrics computed before global lock=NO")
    print("Model refit=NO")
    print("Threshold recalibration=NO")
    print()

    # Re-verify global lock itself before target-mask access.
    if file_sha256(global_lock_path) != global_lock_sha:
        raise RuntimeError("Global no-label lock changed before Phase B.")

    # ---------------------- PHASE B ----------------------
    print("===== BEGIN S05-D FROZEN TARGET GT EVALUATION =====")
    per_rows = evaluate_after_lock(
        no_label_dir,
        eval_rows,
        global_lock_sha,
    )

    summaries = build_summaries(per_rows)
    pooled = next(
        s for s in summaries
        if s["group"] == "POOLED_UNSEEN"
    )
    criterion = frozen_replication_decision(pooled)

    per_fields = [
        "seed", "sample_id", "role", "dataset",
        "p_harm", "tau", "tent_accepted", "decision",
        "source_dice", "source_iou",
        "tent_dice", "tent_iou",
        "safettta_dice", "safettta_iou",
        "oracle_dice",
        "tent_delta_dice", "safettta_delta_dice",
        "tent_harmful", "tent_beneficial",
        "safettta_harmful", "safettta_beneficial",
        "harm_prevented", "benefit_retained",
        "global_no_label_lock_sha256",
    ]

    write_csv(
        build_dir / "per_image_seed_target_evaluation.csv",
        per_rows,
        per_fields,
    )

    summary_fields = [
        "group", "n",
        "source_mean_dice", "tent_mean_dice",
        "safettta_mean_dice", "oracle_mean_dice",
        "tent_mean_delta", "safettta_mean_delta",
        "tent_harmful_n", "tent_harmful_fraction",
        "safettta_harmful_n", "safettta_harmful_fraction",
        "harm_prevention_rate",
        "tent_beneficial_n", "benefit_retention_rate",
        "tent_acceptance_rate", "oracle_gap",
    ]

    write_csv(
        build_dir / "target_summary.csv",
        summaries,
        summary_fields,
    )

    decision_json = {
        "script_version": VERSION,
        "build": BUILD,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "gate_artifact_sha256": EXPECTED_GATE_SHA256,
        "global_no_label_lock_sha256": global_lock_sha,
        "checkpoint_sha256": {
            str(k): v
            for k, v in checkpoint_hashes.items()
        },
        "model_refit": False,
        "threshold_recalibrated": False,
        "target_masks_opened_before_no_label_lock": False,
        "primary_pooled_unseen": pooled,
        "criterion_results": criterion,
        "decision": criterion["decision"],
    }

    (build_dir / "decision.json").write_text(
        json.dumps(
            decision_json,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = [
        "===== S05-D DEEPLAB LOCKED TARGET REPLICATION =====",
        f"Script version: {VERSION}",
        f"Build: {BUILD}",
        "",
        "No-label deployment lock:",
        f"  gate artifact SHA256={EXPECTED_GATE_SHA256}",
        f"  tau={gate['tau']:.10f}",
        f"  global no-label lock SHA256={global_lock_sha}",
        "  model refit=NO",
        "  threshold recalibration=NO",
        "  target masks opened before global lock=NO",
        "  target metrics computed before global lock=NO",
        "",
        "Primary pooled unseen:",
        f"  N={pooled['n']}",
        f"  Source-Only mean Dice={pooled['source_mean_dice']:.6f}",
        f"  Always-TENT mean Dice={pooled['tent_mean_dice']:.6f}",
        f"  SafeTTA-v1 mean Dice={pooled['safettta_mean_dice']:.6f}",
        f"  Oracle mean Dice={pooled['oracle_mean_dice']:.6f}",
        f"  SafeTTA mean DeltaDice={pooled['safettta_mean_delta']:+.6f}",
        f"  Always-TENT harmful fraction={pooled['tent_harmful_fraction']:.6f}",
        f"  SafeTTA harmful fraction={pooled['safettta_harmful_fraction']:.6f}",
        f"  harm prevention rate={pooled['harm_prevention_rate']:.6f}",
        f"  TENT beneficial cases={pooled['tent_beneficial_n']}",
        f"  benefit retention rate={pooled['benefit_retention_rate']:.6f}",
        f"  TENT acceptance rate={pooled['tent_acceptance_rate']:.6f}",
        f"  Oracle gap={pooled['oracle_gap']:.6f}",
        "",
        "Frozen second-backbone criteria:",
        f"  A SafeTTA mean Dice >= Source-Only: "
        f"{criterion['A_safe_mean_ge_source']}",
        f"  B harm prevention >= {PASS_HARM_PREVENTION:.2f}: "
        f"{criterion['B_harm_prevention_ge_0_50']}",
        f"  C benefit retention >= {PASS_BENEFIT_RETENTION:.2f}: "
        f"{criterion['C_benefit_retention_ge_0_30']}",
        f"  D acceptance in [{PASS_ACCEPTANCE_MIN:.2f}, "
        f"{PASS_ACCEPTANCE_MAX:.2f}]: "
        f"{criterion['D_acceptance_in_0_10_0_90']}",
        "",
        "Unseen domain summaries:",
    ]

    for s in summaries:
        if s["group"] in UNSEEN_DOMAINS:
            lines.append(
                f"  {s['group']:22s} "
                f"Src={s['source_mean_dice']:.6f} "
                f"TENT={s['tent_mean_dice']:.6f} "
                f"Safe={s['safettta_mean_dice']:.6f} "
                f"HarmPrev={s['harm_prevention_rate']:.3f} "
                f"BenefitRet={s['benefit_retention_rate']:.3f} "
                f"Accept={s['tent_acceptance_rate']:.3f}"
            )

    lines += [
        "",
        "Seed summaries:",
    ]

    for s in summaries:
        if s["group"].startswith("SEED_"):
            lines.append(
                f"  {s['group']:14s} "
                f"Src={s['source_mean_dice']:.6f} "
                f"TENT={s['tent_mean_dice']:.6f} "
                f"Safe={s['safettta_mean_dice']:.6f} "
                f"HarmPrev={s['harm_prevention_rate']:.3f} "
                f"BenefitRet={s['benefit_retention_rate']:.3f} "
                f"Accept={s['tent_acceptance_rate']:.3f}"
            )

    lines += [
        "",
        f"Decision: {criterion['decision']}",
    ]

    (build_dir / "summary.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    build_dir.rename(args.output_dir)

    print()
    print(
        (args.output_dir / "summary.txt").read_text(
            encoding="utf-8"
        )
    )
    print(f"[OK] Outputs: {args.output_dir}")


def self_test():
    # Exported gate math.
    gate = {
        "feature_names": ["a", "b"],
        "mean": np.asarray([1.0, 2.0]),
        "scale": np.asarray([2.0, 4.0]),
        "coef": np.asarray([0.5, -0.25]),
        "intercept": 0.1,
        "tau": 0.5,
    }
    p, accepted = apply_exported_gate(
        {"a": 3.0, "b": 2.0},
        gate,
    )
    assert 0.0 < p < 1.0
    assert accepted in (0, 1)

    # Segmentation metric.
    gt = np.zeros((16, 16), dtype=np.uint8)
    gt[4:12, 4:12] = 1
    logit = np.full((16, 16), -20.0, dtype=np.float32)
    logit[4:12, 4:12] = 20.0
    dice, iou = metrics_from_logit(logit, gt)
    assert abs(dice - 1.0) < 1e-6
    assert abs(iou - 1.0) < 1e-6

    # Frozen decision.
    pooled = {
        "safettta_mean_dice": 0.81,
        "source_mean_dice": 0.80,
        "harm_prevention_rate": 0.60,
        "benefit_retention_rate": 0.40,
        "tent_acceptance_rate": 0.50,
    }
    decision = frozen_replication_decision(pooled)
    assert decision["all_four"]
    assert (
        decision["decision"]
        == "S05_SECOND_BACKBONE_STRONG_REPLICATION_PASS"
    )

    assert HARM_THRESHOLD == -0.02
    assert BENEFIT_THRESHOLD == 0.02
    assert PASS_HARM_PREVENTION == 0.50
    assert PASS_BENEFIT_RETENTION == 0.30
    assert PASS_ACCEPTANCE_MIN == 0.10
    assert PASS_ACCEPTANCE_MAX == 0.90
    assert EXPECTED_EVAL_IMAGES == 798
    assert EXPECTED_UNSEEN_ROWS == 1908

    print("EXPORTED_GATE_MATH_TEST_PASS")
    print("TARGET_METRIC_TEST_PASS")
    print("FROZEN_REPLICATION_GATE_TEST_PASS")
    print("FROZEN_COUNTS_TEST_PASS")
    print("SELF_TEST_PASS")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run DeepLabV3 locked no-label target prediction/gating and "
            "only then evaluate target GT."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
    )
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume only from cryptographically verified completed "
            "per-seed no-label locks."
        ),
    )
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
