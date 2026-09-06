#!/usr/bin/env python
# -*- coding: utf-8 -*-

r"""
S01-C: Lock and evaluate Source-Only predictions for all three frozen seeds.

Scientific sequence
-------------------
PHASE 1 (NO LABELS):
- Read only evaluation images (roles: seen_sanity, unseen_locked).
- Load the three already-frozen S01-B checkpoints.
- Generate final PraNet logits at 352x352.
- Save logits as float16 NPZ files.
- SHA256-lock all three no-label score files.

Only after ALL three no-label score files are locked:

PHASE 2 (LABEL REVEAL):
- Open GT masks.
- Resize locked logits to original GT resolution.
- Apply sigmoid and fixed threshold 0.5.
- GT is converted with PIL convert("L") and binarized at 0.5*max(GT).
- Compute per-image Dice and IoU.
- Report each seed separately. NO ensemble is created.

No checkpoint/model/hyperparameter selection uses seen_sanity or unseen_locked.
No TTA is performed.

Frozen checkpoints:
- seed 20260817
- seed 20260818
- seed 20260819
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from tqdm import tqdm


VERSION = "2026-08-17-S01-C-v1-fix1"
BUILD = "S01_C_FIX1_MANIFEST_CSV_PATH_AND_PREFLIGHT"

ROOT = Path(r"F:\MEDSEG_SAFETTA")
DATA_ROOT = ROOT / "data" / "processed" / "S00_polyp_locked_v1"
MANIFEST = ROOT / "data" / "splits" / "S01_locked_protocol_manifest_v1.csv"
EXPECTED_MANIFEST_SHA256 = "afb4bc1175af004819538cdbb22ee7a10f1be48035945c7e0d50fd2bbe2e3dc7"

TRAINING_HELPER = ROOT / "code" / "S01_B_train_pranet_source_only_frozen_seeds_v1.py"
EXPECTED_TRAINING_HELPER_SHA256 = "2b2c9c1b75ff60bd087b403e43dc77468b31cbd54fb607481cc80ab47182b67a"

OUTPUT_DIR = ROOT / "outputs" / "S01_C_source_only_frozen_seeds_evaluation_v1"

IMAGE_SIZE = 352
SEEDS = (20260817, 20260818, 20260819)

CHECKPOINTS = {
    20260817: {
        "path": ROOT / "outputs" / "S01_B_pranet_source_only_seed20260817_v1" / "checkpoints" / "best_source_val_dice.pt",
        "sha256": "ef623c0f1207bab02377ec17932db43fe2fd1843dca858cc79ef87ead29b975a",
        "source_val_dice": 0.895586,
    },
    20260818: {
        "path": ROOT / "outputs" / "S01_B_pranet_source_only_seed20260818_v1" / "checkpoints" / "best_source_val_dice.pt",
        "sha256": "e56dd8ba4b5fc7785fedf3918c275427178fadb5f035b47087d48cde41ccec22",
        "source_val_dice": 0.898076,
    },
    20260819: {
        "path": ROOT / "outputs" / "S01_B_pranet_source_only_seed20260819_v1" / "checkpoints" / "best_source_val_dice.pt",
        "sha256": "f8cad00bbc9bbbff04c9a29547bd2a5f3beb97d53980c269cf4b7d2c30130b72",
        "source_val_dice": 0.896459,
    },
}

EXPECTED_ROLE_COUNTS = {
    "source_train": 1305,
    "source_val": 145,
    "seen_sanity": 162,
    "unseen_locked": 636,
}

EXPECTED_EVAL_COUNTS = {
    ("seen_sanity", "Kvasir-SEG"): 100,
    ("seen_sanity", "CVC-ClinicDB"): 62,
    ("unseen_locked", "CVC-ColonDB"): 380,
    ("unseen_locked", "CVC-300"): 60,
    ("unseen_locked", "ETIS-LaribPolypDB"): 196,
}

EXPECTED_EVAL_N = 798

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

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


def read_manifest(path: Path):
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = reader.fieldnames or []
    return rows, fields


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def validate_manifest(rows, fields):
    required = {
        "sample_id", "dataset", "image_relpath", "mask_relpath", "s01_role"
    }
    missing = sorted(required - set(fields))
    if missing:
        raise RuntimeError(f"Manifest missing required columns: {missing}")

    if file_sha256(MANIFEST).lower() != EXPECTED_MANIFEST_SHA256.lower():
        raise RuntimeError("Frozen S01 manifest SHA256 mismatch.")

    counts = Counter(r["s01_role"] for r in rows)
    if counts != Counter(EXPECTED_ROLE_COUNTS):
        raise RuntimeError(
            f"Role counts mismatch: expected={EXPECTED_ROLE_COUNTS}, actual={dict(counts)}"
        )

    eval_rows = [
        r for r in rows
        if r["s01_role"] in {"seen_sanity", "unseen_locked"}
    ]

    if len(eval_rows) != EXPECTED_EVAL_N:
        raise RuntimeError(
            f"Evaluation count mismatch: expected={EXPECTED_EVAL_N}, actual={len(eval_rows)}"
        )

    group_counts = Counter((r["s01_role"], r["dataset"]) for r in eval_rows)
    for key, expected in EXPECTED_EVAL_COUNTS.items():
        actual = group_counts[key]
        if actual != expected:
            raise RuntimeError(
                f"Eval group mismatch {key}: expected={expected}, actual={actual}"
            )

    # Stable evaluation order independent of input CSV order.
    eval_rows = sorted(
        eval_rows,
        key=lambda r: (r["s01_role"], r["dataset"], r["sample_id"]),
    )

    ids = [r["sample_id"] for r in eval_rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate evaluation sample_id.")

    return eval_rows


def import_training_helper():
    if not TRAINING_HELPER.exists():
        raise FileNotFoundError(
            f"Training helper not found: {TRAINING_HELPER}"
        )

    actual_sha = file_sha256(TRAINING_HELPER)
    if actual_sha.lower() != EXPECTED_TRAINING_HELPER_SHA256.lower():
        raise RuntimeError(
            "Training helper SHA256 mismatch.\n"
            f"Expected: {EXPECTED_TRAINING_HELPER_SHA256}\n"
            f"Actual  : {actual_sha}"
        )

    spec = importlib.util.spec_from_file_location(
        "s01b_frozen_training_helper",
        str(TRAINING_HELPER),
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not create import spec for training helper.")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "PraNet"):
        raise RuntimeError("Training helper does not expose PraNet.")
    if not hasattr(module, "image_to_tensor"):
        raise RuntimeError("Training helper does not expose image_to_tensor.")

    return module


def validate_checkpoints():
    checked = {}
    for seed in SEEDS:
        meta = CHECKPOINTS[seed]
        path = meta["path"]
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint missing for seed {seed}: {path}")

        actual_sha = file_sha256(path)
        if actual_sha.lower() != meta["sha256"].lower():
            raise RuntimeError(
                f"Checkpoint SHA mismatch for seed {seed}.\n"
                f"Expected: {meta['sha256']}\n"
                f"Actual  : {actual_sha}"
            )
        checked[seed] = actual_sha
    return checked


def load_model(helper, seed: int, device: torch.device):
    meta = CHECKPOINTS[seed]
    checkpoint = torch.load(
        meta["path"],
        map_location="cpu",
        weights_only=False,
    )

    if int(checkpoint.get("seed", -1)) != seed:
        raise RuntimeError(
            f"Checkpoint internal seed mismatch for {seed}: "
            f"{checkpoint.get('seed')}"
        )

    ck_manifest = checkpoint.get("manifest_sha256")
    if ck_manifest != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError(
            f"Checkpoint manifest SHA mismatch for seed {seed}: {ck_manifest}"
        )

    model = helper.PraNet(
        channel=32,
        pretrained_backbone_state=None,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model


def load_image_tensor_only(row, helper):
    """
    PHASE-1 safety function:
    Only image_relpath is accessed. mask_relpath is never dereferenced here.
    """
    if row["s01_role"] not in {"seen_sanity", "unseen_locked"}:
        raise RuntimeError(f"Unexpected evaluation role: {row['s01_role']}")

    image_path = DATA_ROOT / Path(row["image_relpath"])
    if not image_path.exists():
        raise FileNotFoundError(image_path)

    with Image.open(image_path) as im:
        image = im.convert("RGB")
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE), RESAMPLE_BILINEAR)

    tensor = helper.image_to_tensor(image).unsqueeze(0)
    return tensor


@torch.no_grad()
def generate_locked_logits_for_seed(
    helper,
    seed: int,
    eval_rows,
    device: torch.device,
    build_dir: Path,
):
    model = load_model(helper, seed, device)

    logits_all = np.empty(
        (len(eval_rows), IMAGE_SIZE, IMAGE_SIZE),
        dtype=np.float16,
    )

    pbar = tqdm(
        enumerate(eval_rows),
        total=len(eval_rows),
        desc=f"NO-LABEL logits seed {seed}",
        unit="img",
        dynamic_ncols=True,
    )

    for idx, row in pbar:
        x = load_image_tensor_only(row, helper).to(
            device, non_blocking=True
        )

        outputs = model(x)
        logits = outputs[-1][0, 0].float().cpu().numpy()
        if logits.shape != (IMAGE_SIZE, IMAGE_SIZE):
            raise RuntimeError(
                f"Unexpected logit shape for {row['sample_id']}: {logits.shape}"
            )
        if not np.isfinite(logits).all():
            raise RuntimeError(
                f"Non-finite logits for {row['sample_id']} seed={seed}"
            )

        logits_all[idx] = logits.astype(np.float16)

    out_path = (
        build_dir
        / "locked_no_label_logits"
        / f"source_only_seed{seed}_logits_NO_LABELS.npz"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sample_ids = np.asarray(
        [r["sample_id"] for r in eval_rows],
        dtype=str,
    )
    roles = np.asarray(
        [r["s01_role"] for r in eval_rows],
        dtype=str,
    )
    datasets = np.asarray(
        [r["dataset"] for r in eval_rows],
        dtype=str,
    )

    np.savez_compressed(
        out_path,
        logits=logits_all,
        sample_ids=sample_ids,
        roles=roles,
        datasets=datasets,
        seed=np.asarray([seed], dtype=np.int64),
        image_size=np.asarray([IMAGE_SIZE], dtype=np.int64),
    )

    sha = file_sha256(out_path)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return out_path, sha


def load_gt_binary(row):
    """
    PHASE 2 ONLY.

    GT is decoded with PIL convert('L'), then binarized by 0.5*max(GT).
    This is fixed before unseen performance is revealed.
    """
    mask_path = DATA_ROOT / Path(row["mask_relpath"])
    if not mask_path.exists():
        raise FileNotFoundError(mask_path)

    with Image.open(mask_path) as ma:
        gray = np.asarray(ma.convert("L"), dtype=np.float32)

    if gray.ndim != 2:
        raise RuntimeError(f"GT not 2D after convert(L): {row['sample_id']}")

    maxv = float(gray.max())
    if maxv <= 0:
        raise RuntimeError(f"Empty GT mask: {row['sample_id']}")

    target = (gray > (0.5 * maxv)).astype(np.uint8)
    return target


def resize_logits_to_shape(logits_352: np.ndarray, shape_hw):
    tensor = torch.from_numpy(
        logits_352.astype(np.float32)
    )[None, None, ...]

    resized = F.interpolate(
        tensor,
        size=tuple(shape_hw),
        mode="bilinear",
        align_corners=False,
    )[0, 0]

    return resized.numpy()


def dice_iou(pred: np.ndarray, gt: np.ndarray):
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    inter = int(np.logical_and(pred, gt).sum())
    pred_n = int(pred.sum())
    gt_n = int(gt.sum())
    union = int(np.logical_or(pred, gt).sum())

    dice = (2.0 * inter + 1e-7) / (pred_n + gt_n + 1e-7)
    iou = (inter + 1e-7) / (union + 1e-7)
    return float(dice), float(iou)


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def evaluate_locked_logits(
    locked_files,
    eval_rows,
    build_dir: Path,
):
    """
    First target-mask access in S01-C occurs inside this function.
    """
    all_rows = []

    for seed in SEEDS:
        path = locked_files[seed]["path"]
        expected_sha = locked_files[seed]["sha256"]

        if file_sha256(path) != expected_sha:
            raise RuntimeError(
                f"Locked no-label score SHA changed before label reveal: seed={seed}"
            )

        data = np.load(path, allow_pickle=False)
        logits_all = data["logits"]
        sample_ids = data["sample_ids"].tolist()

        expected_ids = [r["sample_id"] for r in eval_rows]
        if sample_ids != expected_ids:
            raise RuntimeError(
                f"Locked sample order mismatch for seed {seed}"
            )
        if logits_all.shape != (
            EXPECTED_EVAL_N, IMAGE_SIZE, IMAGE_SIZE
        ):
            raise RuntimeError(
                f"Locked logit shape mismatch seed {seed}: {logits_all.shape}"
            )

        pbar = tqdm(
            enumerate(eval_rows),
            total=len(eval_rows),
            desc=f"LABEL-REVEAL eval seed {seed}",
            unit="img",
            dynamic_ncols=True,
        )

        for idx, row in pbar:
            gt = load_gt_binary(row)

            resized_logits = resize_logits_to_shape(
                logits_all[idx],
                gt.shape,
            )
            prob = 1.0 / (1.0 + np.exp(-resized_logits))
            pred = (prob >= 0.5).astype(np.uint8)

            dice, iou = dice_iou(pred, gt)

            all_rows.append({
                "seed": seed,
                "sample_id": row["sample_id"],
                "role": row["s01_role"],
                "dataset": row["dataset"],
                "dice": f"{dice:.10f}",
                "iou": f"{iou:.10f}",
                "gt_height": gt.shape[0],
                "gt_width": gt.shape[1],
                "pred_foreground_fraction": f"{float(pred.mean()):.10f}",
                "gt_foreground_fraction": f"{float(gt.mean()):.10f}",
            })

    return all_rows


def summarize(all_rows):
    per_seed_dataset = []
    per_seed_role = []

    for seed in SEEDS:
        seed_rows = [r for r in all_rows if int(r["seed"]) == seed]

        # Dataset-level.
        for (role, dataset), expected_n in EXPECTED_EVAL_COUNTS.items():
            group = [
                r for r in seed_rows
                if r["role"] == role and r["dataset"] == dataset
            ]
            if len(group) != expected_n:
                raise RuntimeError(
                    f"Summary group count mismatch seed={seed} "
                    f"{role}/{dataset}: {len(group)} vs {expected_n}"
                )

            dices = [float(r["dice"]) for r in group]
            ious = [float(r["iou"]) for r in group]

            per_seed_dataset.append({
                "seed": seed,
                "role": role,
                "dataset": dataset,
                "n": len(group),
                "mean_dice": f"{np.mean(dices):.10f}",
                "median_dice": f"{np.median(dices):.10f}",
                "dice_q25": f"{percentile(dices, 25):.10f}",
                "dice_q75": f"{percentile(dices, 75):.10f}",
                "mean_iou": f"{np.mean(ious):.10f}",
                "median_iou": f"{np.median(ious):.10f}",
            })

        # Role-level pooled summary.
        for role, expected_n in (
            ("seen_sanity", 162),
            ("unseen_locked", 636),
        ):
            group = [r for r in seed_rows if r["role"] == role]
            if len(group) != expected_n:
                raise RuntimeError(
                    f"Role count mismatch seed={seed} role={role}"
                )
            dices = [float(r["dice"]) for r in group]
            ious = [float(r["iou"]) for r in group]

            per_seed_role.append({
                "seed": seed,
                "role": role,
                "n": len(group),
                "mean_dice": f"{np.mean(dices):.10f}",
                "median_dice": f"{np.median(dices):.10f}",
                "dice_q25": f"{percentile(dices, 25):.10f}",
                "dice_q75": f"{percentile(dices, 75):.10f}",
                "mean_iou": f"{np.mean(ious):.10f}",
                "median_iou": f"{np.median(ious):.10f}",
            })

    # Across-seed aggregation of dataset means (3 source seeds).
    across_seed = []
    for role, dataset in EXPECTED_EVAL_COUNTS:
        group = [
            r for r in per_seed_dataset
            if r["role"] == role and r["dataset"] == dataset
        ]
        vals_d = [float(r["mean_dice"]) for r in group]
        vals_i = [float(r["mean_iou"]) for r in group]
        across_seed.append({
            "role": role,
            "dataset": dataset,
            "seeds": len(group),
            "mean_of_seed_mean_dice": f"{np.mean(vals_d):.10f}",
            "sd_of_seed_mean_dice": f"{np.std(vals_d, ddof=1):.10f}",
            "min_seed_mean_dice": f"{np.min(vals_d):.10f}",
            "max_seed_mean_dice": f"{np.max(vals_d):.10f}",
            "mean_of_seed_mean_iou": f"{np.mean(vals_i):.10f}",
            "sd_of_seed_mean_iou": f"{np.std(vals_i, ddof=1):.10f}",
        })

    return per_seed_dataset, per_seed_role, across_seed


def run(args):
    # Immutable input preflight comes BEFORE creating/deleting any S01-C output.
    if not MANIFEST.exists():
        raise FileNotFoundError(f"Frozen manifest not found: {MANIFEST}")
    if MANIFEST.suffix.lower() != ".csv":
        raise RuntimeError(f"Frozen manifest must be a CSV file: {MANIFEST}")
    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"Frozen data root not found: {DATA_ROOT}")
    if not TRAINING_HELPER.exists():
        raise FileNotFoundError(f"Frozen S01-B helper not found: {TRAINING_HELPER}")

    rows, fields = read_manifest(MANIFEST)
    eval_rows = validate_manifest(rows, fields)
    checkpoint_hashes = validate_checkpoints()
    helper = import_training_helper()

    if args.output_dir.exists():
        if not args.technical_rerun:
            raise FileExistsError(
                f"Output already exists: {args.output_dir}\n"
                "Refusing to overwrite the first target-reveal evaluation."
            )
        shutil.rmtree(args.output_dir)

    build_dir = Path(str(args.output_dir) + "__building")
    if build_dir.exists():
        if not args.technical_rerun:
            raise FileExistsError(
                f"Partial build exists: {build_dir}\n"
                "Use --technical-rerun only if this is the same frozen protocol."
            )
        shutil.rmtree(build_dir)

    build_dir.mkdir(parents=True, exist_ok=False)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )

    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(f"[BUILD] {VERSION} | {BUILD}")
    print(f"Device: {device}")
    print()
    print("Frozen evaluation:")
    print(f"  seen_sanity={sum(r['s01_role']=='seen_sanity' for r in eval_rows)}")
    print(f"  unseen_locked={sum(r['s01_role']=='unseen_locked' for r in eval_rows)}")
    print(f"  seeds={SEEDS}")
    print("  ensemble=NO")
    print("  TTA=NO")
    print("  phase1 target masks opened=NO")
    print()

    # --------------------------------------------------------------
    # PHASE 1: predictions first, no GT masks opened.
    # --------------------------------------------------------------
    locked_files = {}

    for seed in SEEDS:
        path, sha = generate_locked_logits_for_seed(
            helper=helper,
            seed=seed,
            eval_rows=eval_rows,
            device=device,
            build_dir=build_dir,
        )
        locked_files[seed] = {
            "path": path,
            "sha256": sha,
        }

    lock_manifest = {
        "script_version": VERSION,
        "build": BUILD,
        "phase": "NO_LABEL_PREDICTION_LOCK",
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "training_helper_sha256": EXPECTED_TRAINING_HELPER_SHA256,
        "checkpoint_sha256": checkpoint_hashes,
        "evaluation_n": EXPECTED_EVAL_N,
        "seeds": list(SEEDS),
        "target_masks_opened_before_lock": False,
        "target_metrics_computed_before_lock": False,
        "locked_files": {
            str(seed): {
                "path": str(meta["path"]),
                "sha256": meta["sha256"],
            }
            for seed, meta in locked_files.items()
        },
    }

    lock_path = build_dir / "NO_LABEL_PREDICTION_LOCK.json"
    lock_path.write_text(
        json.dumps(lock_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lock_sha = file_sha256(lock_path)

    print()
    print("===== NO-LABEL PREDICTION LOCK COMPLETE =====")
    for seed in SEEDS:
        print(
            f"  seed {seed}: {locked_files[seed]['sha256']}"
        )
    print(f"  lock manifest SHA256: {lock_sha}")
    print("  Target masks opened before lock: NO")
    print("  Target metrics computed before lock: NO")
    print()

    # --------------------------------------------------------------
    # PHASE 2: first target-mask access.
    # --------------------------------------------------------------
    all_rows = evaluate_locked_logits(
        locked_files=locked_files,
        eval_rows=eval_rows,
        build_dir=build_dir,
    )

    per_seed_dataset, per_seed_role, across_seed = summarize(all_rows)

    write_csv(
        build_dir / "per_image_source_only_metrics.csv",
        all_rows,
        [
            "seed", "sample_id", "role", "dataset",
            "dice", "iou", "gt_height", "gt_width",
            "pred_foreground_fraction", "gt_foreground_fraction",
        ],
    )
    write_csv(
        build_dir / "per_seed_dataset_summary.csv",
        per_seed_dataset,
        [
            "seed", "role", "dataset", "n",
            "mean_dice", "median_dice", "dice_q25", "dice_q75",
            "mean_iou", "median_iou",
        ],
    )
    write_csv(
        build_dir / "per_seed_role_summary.csv",
        per_seed_role,
        [
            "seed", "role", "n",
            "mean_dice", "median_dice", "dice_q25", "dice_q75",
            "mean_iou", "median_iou",
        ],
    )
    write_csv(
        build_dir / "across_seed_dataset_summary.csv",
        across_seed,
        [
            "role", "dataset", "seeds",
            "mean_of_seed_mean_dice", "sd_of_seed_mean_dice",
            "min_seed_mean_dice", "max_seed_mean_dice",
            "mean_of_seed_mean_iou", "sd_of_seed_mean_iou",
        ],
    )

    audit = {
        "script_version": VERSION,
        "build": BUILD,
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "training_helper_sha256": EXPECTED_TRAINING_HELPER_SHA256,
        "checkpoint_sha256": checkpoint_hashes,
        "no_label_lock_manifest_sha256": lock_sha,
        "prediction_lock_before_label_reveal": True,
        "target_masks_opened_before_prediction_lock": False,
        "target_metrics_computed_before_prediction_lock": False,
        "tta_performed": False,
        "ensemble_used": False,
        "evaluation_probability_threshold": 0.5,
        "gt_rule": "PIL convert(L), binary GT = gray > 0.5*max(gray)",
        "prediction_rule": (
            "locked 352x352 final logits -> bilinear resize to original GT size "
            "(align_corners=False) -> sigmoid -> >=0.5"
        ),
        "decision": "S01_C_SOURCE_ONLY_BASELINE_FROZEN_READY_FOR_S01_D_TTA_FEASIBILITY",
    }
    (build_dir / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary_lines = [
        "===== S01-C SOURCE-ONLY FROZEN-SEED EVALUATION =====",
        f"Script version: {VERSION}",
        f"Build: {BUILD}",
        "",
        "Prediction locking:",
        "  All 3 source-only logit files generated before GT masks were opened=YES",
        f"  NO_LABEL_PREDICTION_LOCK SHA256={lock_sha}",
        "  TTA=NO",
        "  Ensemble=NO",
        "",
        "Per-seed dataset performance:",
    ]

    for row in per_seed_dataset:
        summary_lines.append(
            f"  seed={row['seed']} "
            f"{row['role']:13s} {row['dataset']:22s} "
            f"N={row['n']:3d} "
            f"Dice={float(row['mean_dice']):.6f} "
            f"IoU={float(row['mean_iou']):.6f}"
        )

    summary_lines += [
        "",
        "Across-seed dataset means:",
    ]

    for row in across_seed:
        summary_lines.append(
            f"  {row['role']:13s} {row['dataset']:22s} "
            f"Dice={float(row['mean_of_seed_mean_dice']):.6f}"
            f"±{float(row['sd_of_seed_mean_dice']):.6f} "
            f"IoU={float(row['mean_of_seed_mean_iou']):.6f}"
            f"±{float(row['sd_of_seed_mean_iou']):.6f}"
        )

    summary_lines += [
        "",
        "Important:",
        "  Source-only checkpoint selection used source_val only.",
        "  seen_sanity/unseen_locked did not select checkpoints or hyperparameters.",
        "  No post-target ensemble or threshold selection was performed.",
        "",
        "Decision: S01_C_SOURCE_ONLY_BASELINE_FROZEN_READY_FOR_S01_D_TTA_FEASIBILITY",
    ]

    (build_dir / "summary.txt").write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )

    # Atomic final commit after full evaluation.
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    build_dir.rename(args.output_dir)

    print()
    print((args.output_dir / "summary.txt").read_text(encoding="utf-8"))
    print(f"[OK] Outputs: {args.output_dir}")


def self_test():
    # Metric sanity.
    gt = np.array(
        [[0, 1], [1, 0]],
        dtype=np.uint8,
    )
    pred = gt.copy()
    d, i = dice_iou(pred, gt)
    assert abs(d - 1.0) < 1e-6
    assert abs(i - 1.0) < 1e-6

    # Resize-logit rule should preserve constant sign.
    pos = np.full((IMAGE_SIZE, IMAGE_SIZE), 5.0, dtype=np.float32)
    pos_r = resize_logits_to_shape(pos, (17, 23))
    assert pos_r.shape == (17, 23)
    assert np.all(pos_r > 0)

    neg = np.full((IMAGE_SIZE, IMAGE_SIZE), -5.0, dtype=np.float32)
    neg_r = resize_logits_to_shape(neg, (19, 11))
    assert np.all(neg_r < 0)

    # Frozen manifest path regression test.
    assert MANIFEST.name == "S01_locked_protocol_manifest_v1.csv"
    assert MANIFEST.suffix.lower() == ".csv"
    print("MANIFEST_CSV_PATH_TEST_PASS")

    # Frozen counts.
    assert sum(EXPECTED_EVAL_COUNTS.values()) == EXPECTED_EVAL_N
    assert SEEDS == (20260817, 20260818, 20260819)

    # No ensemble checkpoint exists in protocol.
    assert set(CHECKPOINTS) == set(SEEDS)

    print("METRIC_TEST_PASS")
    print("LOGIT_RESIZE_TEST_PASS")
    print("FROZEN_EVAL_COUNT_TEST_PASS")
    print("NO_ENSEMBLE_PROTOCOL_TEST_PASS")
    print("SELF_TEST_PASS")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Lock no-label Source-Only predictions, then reveal masks and "
            "evaluate all three frozen S01 seeds."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
    )
    parser.add_argument(
        "--technical-rerun",
        action="store_true",
        help=(
            "Overwrite this exact S01-C output only for a technical rerun "
            "without changing protocol/checkpoints."
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
