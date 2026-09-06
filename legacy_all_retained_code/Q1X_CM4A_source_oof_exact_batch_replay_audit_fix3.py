#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1X_CM4A_source_oof_exact_batch_replay_audit_fix3.py

Read-only CM4A replay audit.

Hypothesis tested
-----------------
CM3 generated validation/OOF probabilities with:
- model.eval()
- autocast fp16 on CUDA
- batch_size = 8
- deterministic cuDNN enabled
- validation indices in original slice order

The previous replay audit used batch_size = 1. Floating-point kernels can change
with batch geometry, especially for transformer/attention paths.

This audit reproduces the exact CM3 validation batch geometry and deterministic
runtime, then compares:
  replay probability float16 vs frozen CM3 OOF probability float16
  replay binary mask      vs frozen CM3 OOF binary mask

No TENT. No PROMISE12. No tolerance relaxation.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


VERSION = "2026-09-05-Q1X-CM4A-EXACT-BATCH-REPLAY-v1-fix3"

ROOT = Path(r"F:\MEDSEG_SAFETTA")
OUT = ROOT / "outputs"
CODE = ROOT / "code"

CM3_DIR = (
    OUT
    / "Q1X_CM3_source_oof_segmentation_training_and_prediction_lock_fix3_v1"
)
CM3_LOCK = CM3_DIR / "CM3_SOURCE_OOF_SEGMENTATION_PANEL_LOCK.json"
EXPECTED_CM3_LOCK_SHA = (
    "4cfbe8f108f53e014973111d96244c666bfabd13eb022675fa31a34edaae7cec"
)

CM3_HELPER = (
    CODE
    / "Q1X_CM3_source_oof_segmentation_training_and_prediction_lock_fix3.py"
)
EXPECTED_CM3_HELPER_SHA = (
    "443e371c1a71173cc9fd3eaa4876cb8721a00497836e06d902b50da12cb9f57a"
)

DEFAULT_OUTPUT = (
    OUT
    / "Q1X_CM4A_source_oof_exact_batch_replay_audit_fix3_v1"
)

FAMILIES = ["UNet", "DeepLabV3_R50", "SegFormer_B0"]
INPUT_SIZE = 352
BATCH_SIZE = 8
MASK_THRESHOLD = np.float16(0.5)

IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def sha256_file(path: Path, chunk=8 * 1024 * 1024):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, obj):
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def import_cm3_helper():
    got = sha256_file(CM3_HELPER)
    print(
        "CM3_HELPER",
        got,
        "PASS" if got == EXPECTED_CM3_HELPER_SHA else "FAIL",
    )
    if got != EXPECTED_CM3_HELPER_SHA:
        raise RuntimeError("CM3 helper SHA mismatch.")

    spec = importlib.util.spec_from_file_location(
        "q1x_cm3_fix3_helper",
        str(CM3_HELPER),
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to import CM3 helper.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configure_exact_cm3_runtime(torch):
    # CM3 set_global_seed() enforced these before each fold.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Report other numerical toggles; do not invent settings CM3 did not freeze.
    print("cudnn.benchmark:", torch.backends.cudnn.benchmark)
    print("cudnn.deterministic:", torch.backends.cudnn.deterministic)
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        print(
            "cuda.matmul.allow_tf32:",
            torch.backends.cuda.matmul.allow_tf32,
        )
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        print(
            "cudnn.allow_tf32:",
            torch.backends.cudnn.allow_tf32,
        )


def build_batch(images, global_indices, torch, device):
    arr = np.asarray(
        images[np.asarray(global_indices, dtype=np.int64)],
        dtype=np.float32,
    )  # [B,H,W]

    x = torch.from_numpy(arr).unsqueeze(1).repeat(1, 3, 1, 1)

    mean = torch.tensor(
        IMAGENET_MEAN,
        dtype=x.dtype,
    ).view(1, 3, 1, 1)
    std = torch.tensor(
        IMAGENET_STD,
        dtype=x.dtype,
    ).view(1, 3, 1, 1)

    x = (x - mean) / std
    return x.to(device, non_blocking=True)


def main():
    print("===== Q1X CM4A FIX3 EXACT-BATCH SOURCE OOF REPLAY AUDIT =====")
    print("MODE=READ_ONLY_DIAGNOSTIC")
    print("REPLAY_BATCH_SIZE=8")
    print("REPLAY_ORDER=CM3_VALIDATION_INDEX_ORDER")
    print("CUDNN_DETERMINISTIC=YES")
    print("CUDNN_BENCHMARK=NO")
    print("TENT=NO")
    print("PROMISE12_ACCESS=NO")
    print("TOLERANCE_RELAXATION=NO")

    if not CM3_LOCK.is_file():
        raise FileNotFoundError(CM3_LOCK)

    got = sha256_file(CM3_LOCK)
    print(
        "\nCM3_LOCK",
        got,
        "PASS" if got == EXPECTED_CM3_LOCK_SHA else "FAIL",
    )
    if got != EXPECTED_CM3_LOCK_SHA:
        raise RuntimeError("CM3 lock SHA mismatch.")

    cm3 = load_json(CM3_LOCK)
    if cm3.get("status") != "PASS":
        raise RuntimeError("CM3 lock status changed.")

    helper = import_cm3_helper()

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n===== RUNTIME =====")
    print("torch:", torch.__version__)
    print("device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(device))
    configure_exact_cm3_runtime(torch)

    cache_meta_path = Path(cm3["source_cache_meta"])
    if sha256_file(cache_meta_path) != cm3["source_cache_meta_sha256"]:
        raise RuntimeError("CM3 source cache meta SHA mismatch.")

    cache_meta = load_json(cache_meta_path)
    cache_dir = cache_meta_path.parent
    image_path = cache_dir / "source_images_f16.npy"
    slices_path = cache_dir / "source_slices.csv"

    if sha256_file(image_path) != cache_meta["images_sha256"]:
        raise RuntimeError("SOURCE image cache SHA mismatch.")
    if sha256_file(slices_path) != cache_meta["slices_sha256"]:
        raise RuntimeError("SOURCE slice manifest SHA mismatch.")

    images = np.load(image_path, mmap_mode="r")
    slice_df = pd.read_csv(slices_path)

    if images.shape != (3553, INPUT_SIZE, INPUT_SIZE):
        raise RuntimeError(f"Unexpected SOURCE cache shape: {images.shape}")
    if len(slice_df) != 3553:
        raise RuntimeError("Unexpected SOURCE slice manifest length.")

    stage_dir = DEFAULT_OUTPUT
    if stage_dir.exists():
        raise FileExistsError(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=False)

    summary_rows = []
    mismatch_rows = []

    for family in FAMILIES:
        print(f"\n########## {family} ##########")

        info = cm3["family_oof_artifacts"][family]
        prob_path = Path(info["oof_prob"])
        mask_path = Path(info["oof_mask"])

        if sha256_file(prob_path) != info["oof_prob_sha256"]:
            raise RuntimeError(f"{family}: OOF probability SHA mismatch.")
        if sha256_file(mask_path) != info["oof_mask_sha256"]:
            raise RuntimeError(f"{family}: OOF mask SHA mismatch.")

        frozen_prob = np.load(prob_path, mmap_mode="r")
        frozen_mask = np.load(mask_path, mmap_mode="r")

        for fold in range(5):
            complete_path = (
                CM3_DIR
                / "models"
                / family
                / f"fold{fold}"
                / "COMPLETE.json"
            )
            complete = load_json(complete_path)
            if complete.get("status") != "PASS":
                raise RuntimeError(
                    f"{family} fold{fold}: COMPLETE not PASS."
                )

            ckpt_path = Path(complete["best_checkpoint"])
            ckpt_sha = complete["best_checkpoint_sha256"]
            if sha256_file(ckpt_path) != ckpt_sha:
                raise RuntimeError(
                    f"{family} fold{fold}: checkpoint SHA mismatch."
                )

            model = helper.build_model(family)
            ckpt = torch.load(
                ckpt_path,
                map_location="cpu",
                weights_only=False,
            )
            model.load_state_dict(ckpt["model"], strict=True)
            model.to(device)
            model.eval()
            model.requires_grad_(False)

            # Exact CM3 val_idx semantics: original row/index order.
            fold_indices = (
                slice_df.index[
                    slice_df["fold"].astype(int) == fold
                ]
                .to_numpy(dtype=np.int64)
            )

            mismatch_slices = 0
            mismatch_pixels = 0
            max_abs_prob_diff = 0.0
            max_mismatch_dist_to_half = 0.0
            exact_prob_pixels = 0
            total_pixels = 0

            starts = range(0, len(fold_indices), BATCH_SIZE)
            for start in tqdm(
                starts,
                total=(len(fold_indices) + BATCH_SIZE - 1) // BATCH_SIZE,
                desc=f"{family} F{fold} batch8 replay",
                unit="batch",
                dynamic_ncols=True,
            ):
                gi = fold_indices[start:start + BATCH_SIZE]

                x = build_batch(images, gi, torch, device)

                with torch.no_grad():
                    with torch.autocast(
                        device_type=device.type,
                        enabled=(device.type == "cuda"),
                        dtype=(
                            torch.float16
                            if device.type == "cuda"
                            else torch.bfloat16
                        ),
                    ):
                        logits = model(x)

                replay_prob = (
                    torch.sigmoid(logits)
                    .float()
                    .cpu()
                    .numpy()[:, 0]
                    .astype(np.float16)
                )

                stored_prob = np.asarray(
                    frozen_prob[gi],
                    dtype=np.float16,
                )
                stored_mask = np.asarray(
                    frozen_mask[gi],
                    dtype=bool,
                )
                replay_mask = replay_prob >= MASK_THRESHOLD

                abs_diff = np.abs(
                    replay_prob.astype(np.float32)
                    - stored_prob.astype(np.float32)
                )

                total_pixels += int(replay_prob.size)
                exact_prob_pixels += int(
                    np.count_nonzero(replay_prob == stored_prob)
                )
                max_abs_prob_diff = max(
                    max_abs_prob_diff,
                    float(abs_diff.max()),
                )

                for local_i, global_i in enumerate(gi.tolist()):
                    mm = replay_mask[local_i] != stored_mask[local_i]
                    n = int(np.count_nonzero(mm))
                    if not n:
                        continue

                    mismatch_slices += 1
                    mismatch_pixels += n

                    rp = replay_prob[local_i][mm].astype(np.float32)
                    sp = stored_prob[local_i][mm].astype(np.float32)
                    dist = np.maximum(
                        np.abs(rp - 0.5),
                        np.abs(sp - 0.5),
                    )
                    max_mismatch_dist_to_half = max(
                        max_mismatch_dist_to_half,
                        float(dist.max()),
                    )

                    ys, xs = np.nonzero(mm)
                    for y, x0 in zip(ys.tolist(), xs.tolist()):
                        mismatch_rows.append({
                            "family": family,
                            "fold": fold,
                            "global_index": int(global_i),
                            "case_key": str(
                                slice_df.iloc[int(global_i)]["case_key"]
                            ),
                            "slice_index": int(
                                slice_df.iloc[int(global_i)]["slice_index"]
                            ),
                            "y": y,
                            "x": x0,
                            "stored_prob_f16": float(
                                stored_prob[local_i, y, x0]
                            ),
                            "replay_prob_f16": float(
                                replay_prob[local_i, y, x0]
                            ),
                            "stored_mask": int(
                                stored_mask[local_i, y, x0]
                            ),
                            "replay_mask": int(
                                replay_mask[local_i, y, x0]
                            ),
                            "abs_prob_diff": float(
                                abs_diff[local_i, y, x0]
                            ),
                        })

            exact_fraction = (
                float(exact_prob_pixels) / float(total_pixels)
            )
            mismatch_fraction = (
                float(mismatch_pixels) / float(total_pixels)
            )

            summary_rows.append({
                "family": family,
                "fold": fold,
                "slices": len(fold_indices),
                "batch_size": BATCH_SIZE,
                "mismatch_slices": mismatch_slices,
                "mismatch_pixels": mismatch_pixels,
                "pixel_mismatch_fraction": mismatch_fraction,
                "exact_float16_probability_fraction": exact_fraction,
                "max_abs_probability_difference": max_abs_prob_diff,
                "max_mismatch_distance_to_0.5": (
                    max_mismatch_dist_to_half
                ),
            })

            print(
                f"{family} fold{fold}: "
                f"mismatch_slices={mismatch_slices} "
                f"mismatch_pixels={mismatch_pixels} "
                f"pixel_fraction={mismatch_fraction:.3e} "
                f"prob_exact={exact_fraction:.9f} "
                f"max_abs_prob_diff={max_abs_prob_diff:.8f} "
                f"max_mismatch_dist_to_0.5="
                f"{max_mismatch_dist_to_half:.8f}"
            )

            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    summary_df = pd.DataFrame(summary_rows)
    mismatch_df = pd.DataFrame(mismatch_rows)

    summary_path = stage_dir / "CM4A_EXACT_BATCH_REPLAY_SUMMARY.csv"
    mismatch_path = stage_dir / "CM4A_EXACT_BATCH_REPLAY_MISMATCH_PIXELS.csv"
    summary_df.to_csv(summary_path, index=False)
    mismatch_df.to_csv(mismatch_path, index=False)

    total_mismatch_slices = int(summary_df["mismatch_slices"].sum())
    total_mismatch_pixels = int(summary_df["mismatch_pixels"].sum())
    total_pixels = int(
        (summary_df["slices"] * INPUT_SIZE * INPUT_SIZE).sum()
    )

    exact_replay_pass = (
        total_mismatch_slices == 0
        and total_mismatch_pixels == 0
    )

    decision = (
        "EXACT_CM3_BATCH8_BINARY_REPLAY_CONFIRMED_READY_FOR_CM4A_FIX4"
        if exact_replay_pass
        else "EXACT_CM3_BATCH8_REPLAY_STILL_DIFFERS_STOP_AND_REVIEW"
    )

    audit = {
        "status": "PASS_DIAGNOSTIC",
        "version": VERSION,
        "cm3_lock_sha256": EXPECTED_CM3_LOCK_SHA,
        "cm3_helper_sha256": EXPECTED_CM3_HELPER_SHA,
        "replay_batch_size": BATCH_SIZE,
        "replay_order": "exact CM3 validation index order",
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "total_family_slices": int(summary_df["slices"].sum()),
        "total_pixels": total_pixels,
        "total_mismatch_slices": total_mismatch_slices,
        "total_mismatch_pixels": total_mismatch_pixels,
        "overall_pixel_mismatch_fraction": (
            float(total_mismatch_pixels) / float(total_pixels)
        ),
        "global_max_abs_probability_difference": float(
            summary_df["max_abs_probability_difference"].max()
        ),
        "global_max_mismatch_distance_to_0.5": float(
            summary_df["max_mismatch_distance_to_0.5"].max()
        ),
        "exact_binary_replay_pass": exact_replay_pass,
        "tent": False,
        "promises12_access": False,
        "tolerance_relaxation": False,
        "decision": decision,
        "artifacts": {
            "summary": str(summary_path),
            "summary_sha256": sha256_file(summary_path),
            "mismatch_pixels": str(mismatch_path),
            "mismatch_pixels_sha256": sha256_file(mismatch_path),
        },
    }

    audit_path = stage_dir / "CM4A_EXACT_BATCH_REPLAY_AUDIT.json"
    save_json(audit_path, audit)

    print("\n===== CM4A FIX3 EXACT-BATCH REPLAY FINAL =====")
    print(summary_df.to_string(index=False))
    print("TOTAL mismatch slices:", total_mismatch_slices)
    print("TOTAL mismatch pixels:", total_mismatch_pixels)
    print(
        "OVERALL pixel mismatch fraction:",
        audit["overall_pixel_mismatch_fraction"],
    )
    print(
        "GLOBAL max abs probability difference:",
        audit["global_max_abs_probability_difference"],
    )
    print(
        "GLOBAL max mismatch distance to 0.5:",
        audit["global_max_mismatch_distance_to_0.5"],
    )
    print("TENT: NO")
    print("PROMISE12 access: NO")
    print("Tolerance relaxation: NO")
    print("Decision=", decision)
    print("AUDIT=", audit_path)
    print("AUDIT SHA256=", sha256_file(audit_path))
    print("PASS_DIAGNOSTIC")


if __name__ == "__main__":
    main()
