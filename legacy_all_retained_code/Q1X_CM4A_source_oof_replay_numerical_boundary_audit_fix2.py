#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1X_CM4A_source_oof_replay_numerical_boundary_audit_fix2.py

Read-only diagnostic after CM4A Fix1 stopped because exact binary replay of
CM3 OOF masks differed by a tiny number of pixels.

This script does NOT run TENT and does NOT access PROMISE12.
It compares, for every SOURCE slice and family:
  1) frozen CM3 OOF float16 probabilities;
  2) frozen CM3 OOF uint8 masks;
  3) replayed checkpoint probabilities using the same preprocessing/model.

The goal is to determine whether binary mismatches are only threshold-boundary
numerical flips near p=0.5 or evidence of a real lineage/preprocessing mismatch.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


VERSION = "2026-09-05-Q1X-CM4A-REPLAY-AUDIT-v1-fix2"

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
    / "Q1X_CM4A_source_oof_replay_numerical_boundary_audit_fix2_v1"
)

FAMILIES = ["UNet", "DeepLabV3_R50", "SegFormer_B0"]
INPUT_SIZE = 352
THRESHOLD = np.float16(0.5)

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


def load_source_tensor(image_memmap, global_index, torch, device):
    image = np.asarray(
        image_memmap[int(global_index)],
        dtype=np.float32,
    )
    x = torch.from_numpy(image).unsqueeze(0).repeat(3, 1, 1)

    mean = torch.tensor(
        IMAGENET_MEAN,
        dtype=x.dtype,
    ).view(3, 1, 1)
    std = torch.tensor(
        IMAGENET_STD,
        dtype=x.dtype,
    ).view(3, 1, 1)

    x = (x - mean) / std
    return x.unsqueeze(0).to(device, non_blocking=True)


def main():
    print("===== Q1X CM4A FIX2 SOURCE OOF REPLAY NUMERICAL AUDIT =====")
    print("MODE=READ_ONLY_DIAGNOSTIC")
    print("TENT=NO")
    print("PROMISE12_ACCESS=NO")
    print("SAFETY_MODEL_FITTING=NO")
    print("TARGET_TUNING=NO")

    if not CM3_LOCK.is_file():
        raise FileNotFoundError(CM3_LOCK)

    got = sha256_file(CM3_LOCK)
    print(
        "CM3_LOCK",
        got,
        "PASS" if got == EXPECTED_CM3_LOCK_SHA else "FAIL",
    )
    if got != EXPECTED_CM3_LOCK_SHA:
        raise RuntimeError("CM3 lock SHA mismatch.")

    cm3 = load_json(CM3_LOCK)
    helper = import_cm3_helper()

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("torch:", torch.__version__)
    print("device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(device))

    cache_meta_path = Path(cm3["source_cache_meta"])
    if sha256_file(cache_meta_path) != cm3["source_cache_meta_sha256"]:
        raise RuntimeError("CM3 source-cache-meta SHA mismatch.")
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

    stage_dir = DEFAULT_OUTPUT
    if stage_dir.exists():
        raise FileExistsError(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=False)

    summary_rows = []
    mismatch_rows = []

    for family in FAMILIES:
        info = cm3["family_oof_artifacts"][family]

        prob_path = Path(info["oof_prob"])
        mask_path = Path(info["oof_mask"])

        if sha256_file(prob_path) != info["oof_prob_sha256"]:
            raise RuntimeError(f"{family}: frozen OOF probability SHA mismatch.")
        if sha256_file(mask_path) != info["oof_mask_sha256"]:
            raise RuntimeError(f"{family}: frozen OOF mask SHA mismatch.")

        frozen_prob = np.load(prob_path, mmap_mode="r")
        frozen_mask = np.load(mask_path, mmap_mode="r")

        if frozen_prob.shape != (3553, INPUT_SIZE, INPUT_SIZE):
            raise RuntimeError(f"{family}: unexpected frozen prob shape.")
        if frozen_mask.shape != (3553, INPUT_SIZE, INPUT_SIZE):
            raise RuntimeError(f"{family}: unexpected frozen mask shape.")

        for fold in range(5):
            complete_path = (
                CM3_DIR
                / "models"
                / family
                / f"fold{fold}"
                / "COMPLETE.json"
            )
            complete = load_json(complete_path)
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

            fold_rows = (
                slice_df[slice_df["fold"].astype(int) == fold]
                .sort_values("global_index")
                .reset_index(drop=True)
            )

            mismatch_slices = 0
            mismatch_pixels = 0
            exact_prob_equal_pixels = 0
            total_pixels = 0
            max_abs_prob_diff = 0.0
            max_mismatch_distance_to_half = 0.0
            max_mismatch_abs_prob_diff = 0.0
            mismatch_either_exact_half = 0

            for row in tqdm(
                fold_rows.itertuples(index=False),
                total=len(fold_rows),
                desc=f"{family} F{fold} replay audit",
                unit="slice",
                dynamic_ncols=True,
            ):
                gi = int(row.global_index)
                x = load_source_tensor(images, gi, torch, device)

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
                    .numpy()[0, 0]
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
                replay_mask = replay_prob >= THRESHOLD

                diff_prob = np.abs(
                    replay_prob.astype(np.float32)
                    - stored_prob.astype(np.float32)
                )

                total_pixels += replay_prob.size
                exact_prob_equal_pixels += int(
                    np.count_nonzero(replay_prob == stored_prob)
                )
                max_abs_prob_diff = max(
                    max_abs_prob_diff,
                    float(diff_prob.max()),
                )

                mismatch = replay_mask != stored_mask
                n_mis = int(np.count_nonzero(mismatch))

                if n_mis:
                    mismatch_slices += 1
                    mismatch_pixels += n_mis

                    rp = replay_prob[mismatch].astype(np.float32)
                    sp = stored_prob[mismatch].astype(np.float32)
                    dp = np.abs(rp - sp)

                    dist = np.maximum(
                        np.abs(rp - 0.5),
                        np.abs(sp - 0.5),
                    )

                    max_mismatch_distance_to_half = max(
                        max_mismatch_distance_to_half,
                        float(dist.max()),
                    )
                    max_mismatch_abs_prob_diff = max(
                        max_mismatch_abs_prob_diff,
                        float(dp.max()),
                    )

                    exact_half = np.logical_or(
                        rp == 0.5,
                        sp == 0.5,
                    )
                    mismatch_either_exact_half += int(
                        np.count_nonzero(exact_half)
                    )

                    ys, xs = np.nonzero(mismatch)
                    for y, x0 in zip(ys.tolist(), xs.tolist()):
                        mismatch_rows.append({
                            "family": family,
                            "fold": fold,
                            "global_index": gi,
                            "case_key": row.case_key,
                            "slice_index": int(row.slice_index),
                            "y": y,
                            "x": x0,
                            "stored_prob_f16": float(stored_prob[y, x0]),
                            "replay_prob_f16": float(replay_prob[y, x0]),
                            "stored_mask": int(stored_mask[y, x0]),
                            "replay_mask": int(replay_mask[y, x0]),
                            "abs_prob_diff": float(diff_prob[y, x0]),
                            "stored_distance_to_half": float(
                                abs(float(stored_prob[y, x0]) - 0.5)
                            ),
                            "replay_distance_to_half": float(
                                abs(float(replay_prob[y, x0]) - 0.5)
                            ),
                        })

            pixel_mismatch_fraction = (
                float(mismatch_pixels) / float(total_pixels)
            )
            prob_exact_fraction = (
                float(exact_prob_equal_pixels) / float(total_pixels)
            )

            row = {
                "family": family,
                "fold": fold,
                "slices": len(fold_rows),
                "mismatch_slices": mismatch_slices,
                "mismatch_pixels": mismatch_pixels,
                "pixel_mismatch_fraction": pixel_mismatch_fraction,
                "exact_float16_probability_fraction": prob_exact_fraction,
                "max_abs_probability_difference": max_abs_prob_diff,
                "max_mismatch_abs_probability_difference": (
                    max_mismatch_abs_prob_diff
                ),
                "max_mismatch_distance_to_0.5": (
                    max_mismatch_distance_to_half
                ),
                "mismatch_pixels_with_either_probability_exactly_0.5": (
                    mismatch_either_exact_half
                ),
            }
            summary_rows.append(row)

            print(
                f"{family} fold{fold}: "
                f"mismatch_slices={mismatch_slices}/{len(fold_rows)} "
                f"mismatch_pixels={mismatch_pixels} "
                f"pixel_fraction={pixel_mismatch_fraction:.3e} "
                f"prob_exact={prob_exact_fraction:.9f} "
                f"max_abs_prob_diff={max_abs_prob_diff:.8f} "
                f"max_mismatch_dist_to_0.5="
                f"{max_mismatch_distance_to_half:.8f}"
            )

            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    summary_df = pd.DataFrame(summary_rows)
    mismatch_df = pd.DataFrame(mismatch_rows)

    summary_path = stage_dir / "CM4A_REPLAY_NUMERICAL_AUDIT_SUMMARY.csv"
    mismatch_path = stage_dir / "CM4A_REPLAY_MISMATCH_PIXELS.csv"

    summary_df.to_csv(summary_path, index=False)
    mismatch_df.to_csv(mismatch_path, index=False)

    total_pixels = int(
        (summary_df["slices"] * INPUT_SIZE * INPUT_SIZE).sum()
    )
    total_mismatch_pixels = int(summary_df["mismatch_pixels"].sum())
    total_mismatch_slices = int(summary_df["mismatch_slices"].sum())

    final = {
        "status": "DIAGNOSTIC_COMPLETE",
        "version": VERSION,
        "cm3_lock_sha256": EXPECTED_CM3_LOCK_SHA,
        "cm3_helper_sha256": EXPECTED_CM3_HELPER_SHA,
        "families": FAMILIES,
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
        "promises12_access": False,
        "tent": False,
        "decision": (
            "AUDIT_ONLY_DO_NOT_RELAX_REPLAY_GATE_UNTIL_REVIEWED"
        ),
        "artifacts": {
            "summary": str(summary_path),
            "summary_sha256": sha256_file(summary_path),
            "mismatch_pixels": str(mismatch_path),
            "mismatch_pixels_sha256": sha256_file(mismatch_path),
        },
    }

    final_path = stage_dir / "CM4A_REPLAY_NUMERICAL_AUDIT.json"
    save_json(final_path, final)

    print("\n===== CM4A FIX2 NUMERICAL AUDIT FINAL =====")
    print(summary_df.to_string(index=False))
    print("TOTAL family-slices:", int(summary_df["slices"].sum()))
    print("TOTAL mismatch slices:", total_mismatch_slices)
    print("TOTAL mismatch pixels:", total_mismatch_pixels)
    print(
        "OVERALL pixel mismatch fraction:",
        final["overall_pixel_mismatch_fraction"],
    )
    print(
        "GLOBAL max abs probability difference:",
        final["global_max_abs_probability_difference"],
    )
    print(
        "GLOBAL max mismatch distance to 0.5:",
        final["global_max_mismatch_distance_to_0.5"],
    )
    print("PROMISE12 access: NO")
    print("TENT: NO")
    print(
        "Decision=",
        "AUDIT_ONLY_DO_NOT_RELAX_REPLAY_GATE_UNTIL_REVIEWED",
    )
    print("AUDIT=", final_path)
    print("AUDIT SHA256=", sha256_file(final_path))
    print("PASS_DIAGNOSTIC")


if __name__ == "__main__":
    main()
