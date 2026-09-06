#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R10K2A_dinov2_image_mask_representation_lock_fix1.py

Frozen feature-construction stage for an architecture-agnostic,
image-conditioned safety representation.

Inputs
------
1) NeoPolyp RGB image lineage:
   Q1_R05D1.../frozen_confirmatory_manifest.csv
2) Frozen state lineage:
   R10J0 relocated manifest
3) PRE-ADAPTATION binary SOURCE masks only:
   source_masks_packed

Frozen encoder
--------------
facebook/dinov2-base
- public pretrained encoder
- frozen, eval mode
- no fine-tuning
- input = RGB resized directly to 224x224
- DINOv2 patch grid = 16x16
- hidden size expected = 768

Mask alignment
--------------
The locked SOURCE mask is 352x352.
352 = 16 * 22, so mask occupancy is computed by exact 22x22 block averaging
to a 16x16 grid. This is aligned to the 16x16 DINOv2 patch grid after direct
224x224 image resize (224 = 16 * 14).

Representations
---------------
Per image:
- image CLS: 768-D

Per model-case:
- foreground-weighted patch mean: 768-D
- background-weighted patch mean: 768-D

No HARM label, GT, Dice, A1 mask value, or target outcome is read.

This stage DOES NOT:
- fit PCA
- fit safety models
- tune thresholds
- inspect outcome performance

The representation is frozen before downstream R10K2B evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


MODEL_NAME = "facebook/dinov2-base"
EXPECTED_HIDDEN = 768
IMAGE_SIZE = 224
PATCH_SIZE = 14
PATCH_GRID = 16
MASK_SIZE = 352
MASK_BLOCK = 22
PACKED_BYTES = (MASK_SIZE * MASK_SIZE) // 8
BITORDER = "little"

DEFAULT_IMAGE_MANIFEST = (
    "F:/MEDSEG_SAFETTA/outputs/"
    "Q1_R05D1_neopolyp_acquisition_pairing_duplicate_qc_v1_fix2/"
    "frozen_confirmatory_manifest.csv"
)

DEFAULT_STATE_MANIFEST = (
    "F:/MEDSEG_SAFETTA/outputs/"
    "Q1_R10J0_source_prediction_npz_relocation_schema_audit_fix2_v1/"
    "model_case_prediction_manifest_relocated_fix2.csv"
)

DEFAULT_OUTPUT = (
    "F:/MEDSEG_SAFETTA/outputs/"
    "Q1_R10K2A_dinov2_image_mask_representation_lock_fix1_v1"
)

DEFAULT_HF_CACHE = "F:/MEDSEG_SAFETTA/cache/huggingface"
DEFAULT_TORCH_CACHE = "F:/MEDSEG_SAFETTA/cache/torch"


def norm_sid_value(x):
    return str(x).strip().replace("\\", "/").lower()


def norm_sid_series(s):
    return (
        s.astype(str)
        .str.strip()
        .str.replace("\\", "/", regex=False)
        .str.lower()
    )


def unpack_mask_occupancy16(packed_row):
    p = np.asarray(packed_row, dtype=np.uint8)
    if p.shape != (PACKED_BYTES,):
        raise AssertionError(
            f"Packed source mask shape={p.shape}, expected={(PACKED_BYTES,)}"
        )

    bits = np.unpackbits(p, bitorder=BITORDER)
    if bits.size != MASK_SIZE * MASK_SIZE:
        raise AssertionError(
            f"Unpacked bits={bits.size}, expected={MASK_SIZE * MASK_SIZE}"
        )

    m = bits.reshape(MASK_SIZE, MASK_SIZE).astype(np.float32)

    # 352 = 16 * 22. Exact non-overlapping occupancy aligned to DINO patch grid.
    occ = (
        m.reshape(
            PATCH_GRID,
            MASK_BLOCK,
            PATCH_GRID,
            MASK_BLOCK,
        )
        .mean(axis=(1, 3))
        .astype(np.float32)
    )

    if occ.shape != (PATCH_GRID, PATCH_GRID):
        raise AssertionError(f"Unexpected occupancy shape: {occ.shape}")
    if not np.isfinite(occ).all():
        raise AssertionError("Non-finite mask occupancy.")
    if occ.min() < 0.0 or occ.max() > 1.0:
        raise AssertionError("Mask occupancy outside [0,1].")

    return occ.reshape(-1)


def pooled_region_features(tokens_256xd, occupancy_nx256, eps=1e-6):
    """
    tokens_256xd: float array [256, D]
    occupancy_nx256: float array [N, 256], values in [0,1]

    Returns:
      fg_mean [N,D]
      bg_mean [N,D]

    Empty foreground/background falls back deterministically to the zero
    vector through denominator clamping; no outcome information is used.
    """
    t = np.asarray(tokens_256xd, dtype=np.float32)
    w = np.asarray(occupancy_nx256, dtype=np.float32)

    if t.ndim != 2 or t.shape[0] != PATCH_GRID * PATCH_GRID:
        raise AssertionError(f"Unexpected token shape: {t.shape}")
    if w.ndim != 2 or w.shape[1] != PATCH_GRID * PATCH_GRID:
        raise AssertionError(f"Unexpected occupancy matrix: {w.shape}")

    fg_num = w @ t
    fg_den = np.maximum(w.sum(axis=1, keepdims=True), eps)
    fg = fg_num / fg_den

    bw = 1.0 - w
    bg_num = bw @ t
    bg_den = np.maximum(bw.sum(axis=1, keepdims=True), eps)
    bg = bg_num / bg_den

    # Explicit deterministic zero vector for genuinely empty regions.
    fg[w.sum(axis=1) <= eps] = 0.0
    bg[bw.sum(axis=1) <= eps] = 0.0

    return fg.astype(np.float32), bg.astype(np.float32)


def configure_project_local_caches(hf_cache, torch_cache):
    hf = Path(hf_cache)
    tc = Path(torch_cache)

    hf.mkdir(parents=True, exist_ok=True)
    tc.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(hf)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(hf / "transformers")
    os.environ["TORCH_HOME"] = str(tc)

    return hf, tc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--image_manifest",
        default=DEFAULT_IMAGE_MANIFEST,
    )
    ap.add_argument(
        "--state_manifest",
        default=DEFAULT_STATE_MANIFEST,
    )
    ap.add_argument(
        "--output_dir",
        default=DEFAULT_OUTPUT,
    )
    ap.add_argument(
        "--hf_cache",
        default=DEFAULT_HF_CACHE,
    )
    ap.add_argument(
        "--torch_cache",
        default=DEFAULT_TORCH_CACHE,
    )
    ap.add_argument(
        "--batch_size",
        type=int,
        default=16,
    )
    ap.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
    )
    ap.add_argument(
        "--local_files_only",
        action="store_true",
        help="Require DINOv2 weights to already exist in the project-local cache.",
    )
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    hf_cache, torch_cache = configure_project_local_caches(
        args.hf_cache,
        args.torch_cache,
    )

    print("===== R10K2A DINOV2 IMAGE-MASK REPRESENTATION LOCK FIX1 =====")
    print("STATUS: FROZEN REPRESENTATION CONSTRUCTION")
    print("ENCODER:", MODEL_NAME)
    print("ENCODER TRAINABLE: NO")
    print("IMAGE INPUT: RGB resized directly to 224x224")
    print("PATCH GRID: 16x16")
    print("SOURCE MASK: 352x352 -> exact 22x22 block occupancy -> 16x16")
    print("SOURCE MASK ARRAY: source_masks_packed")
    print("A1 MASK VALUES ACCESSED: NO")
    print("GT USED: NO")
    print("DICE USED: NO")
    print("HARM/BENEFIT LABEL USED: NO")
    print("PCA FITTING: NONE")
    print("SAFETY MODEL FITTING: NONE")
    print("THRESHOLD TUNING: NONE")
    print("HF CACHE:", hf_cache)
    print("TORCH CACHE:", torch_cache)
    print("LOCAL FILES ONLY:", args.local_files_only)
    print()

    image_manifest_path = Path(args.image_manifest)
    state_manifest_path = Path(args.state_manifest)

    print(
        "image manifest exists:",
        image_manifest_path.exists(),
        image_manifest_path,
    )
    print(
        "state manifest exists:",
        state_manifest_path.exists(),
        state_manifest_path,
    )

    if not image_manifest_path.exists():
        raise FileNotFoundError(image_manifest_path)
    if not state_manifest_path.exists():
        raise FileNotFoundError(state_manifest_path)

    images = pd.read_csv(image_manifest_path, low_memory=False)
    states = pd.read_csv(state_manifest_path, low_memory=False)

    required_image = ["sample_id", "image_path"]
    required_state = [
        "sample_id",
        "model_family",
        "model_state_id",
        "training_seed",
        "checkpoint_sha256",
        "state_prediction_npz",
        "state_row_index",
    ]

    for c in required_image:
        if c not in images.columns:
            raise AssertionError(f"Image manifest missing: {c}")
    for c in required_state:
        if c not in states.columns:
            raise AssertionError(f"State manifest missing: {c}")

    images["_sid"] = norm_sid_series(images["sample_id"])
    states["_sid"] = norm_sid_series(states["sample_id"])

    # Exactly one RGB image per case.
    if len(images) != 1000:
        raise AssertionError(
            f"Expected 1000 image-manifest rows, got {len(images)}."
        )
    if images["_sid"].nunique() != 1000:
        raise AssertionError("Image manifest sample_id is not one-to-one.")

    if len(states) != 9000:
        raise AssertionError(
            f"Expected 9000 state rows, got {len(states)}."
        )
    if states["_sid"].nunique() != 1000:
        raise AssertionError("Expected 1000 state-manifest cases.")
    if states["model_state_id"].nunique() != 9:
        raise AssertionError("Expected 9 frozen model states.")
    if (
        states[["_sid", "model_state_id"]]
        .drop_duplicates()
        .shape[0]
        != 9000
    ):
        raise AssertionError("sample_id + model_state_id is not unique.")

    if set(images["_sid"]) != set(states["_sid"]):
        raise AssertionError(
            "Image and state case sets are not exactly equal."
        )

    path_exists = images["image_path"].map(
        lambda p: Path(str(p)).exists()
    )
    print(
        "Existing image paths:",
        int(path_exists.sum()),
        "/",
        len(images),
    )
    if not path_exists.all():
        bad = images.loc[
            ~path_exists,
            ["sample_id", "image_path"],
        ]
        bad.to_csv(
            out / "R10K2A_missing_image_paths.csv",
            index=False,
        )
        raise AssertionError("One or more RGB image paths do not exist.")

    # ------------------------------------------------------------------
    # Freeze deterministic output row order before representation extraction.
    # No labels are present.
    # ------------------------------------------------------------------
    states = states.sort_values(
        ["model_family", "training_seed", "_sid"]
    ).reset_index(drop=True)
    states["_feature_row"] = np.arange(len(states), dtype=int)

    images = images.sort_values("_sid").reset_index(drop=True)

    sid_to_image_index = {
        sid: i
        for i, sid in enumerate(images["_sid"].tolist())
    }

    state_positions_by_sid = {
        sid: g.index.to_numpy(dtype=int)
        for sid, g in states.groupby("_sid", sort=False)
    }

    if not all(len(v) == 9 for v in state_positions_by_sid.values()):
        raise AssertionError("Every case must have exactly 9 model-state rows.")

    # ------------------------------------------------------------------
    # Build 16x16 source-mask occupancy matrix from SOURCE masks only.
    # ------------------------------------------------------------------
    print("\n===== BUILD SOURCE-MASK PATCH OCCUPANCY =====")

    occupancy = np.zeros(
        (len(states), PATCH_GRID * PATCH_GRID),
        dtype=np.float32,
    )

    for state_id, g in states.groupby("model_state_id", sort=True):
        npz_paths = g["state_prediction_npz"].astype(str).unique()
        if len(npz_paths) != 1:
            raise AssertionError(
                f"{state_id}: expected exactly one NPZ path."
            )

        npz_path = Path(npz_paths[0])
        if not npz_path.exists():
            raise FileNotFoundError(npz_path)

        with np.load(npz_path, allow_pickle=False) as z:
            if "source_masks_packed" not in z.files:
                raise AssertionError(
                    f"{state_id}: source_masks_packed missing."
                )
            if "a1_masks_packed" not in z.files:
                raise AssertionError(
                    f"{state_id}: locked A1 key missing."
                )

            source_packed = z["source_masks_packed"]

            if source_packed.shape != (1000, PACKED_BYTES):
                raise AssertionError(
                    f"{state_id}: unexpected SOURCE mask array "
                    f"{source_packed.shape}"
                )

            print(
                state_id,
                "SOURCE=",
                source_packed.shape,
                "A1 value access=NO",
            )

            for r in tqdm(
                g.itertuples(index=True),
                total=len(g),
                desc=f"Mask occupancy {state_id}",
                unit="mask",
                dynamic_ncols=True,
            ):
                state_df_index = int(r.Index)
                state_row_index = int(
                    getattr(r, "state_row_index")
                )
                occupancy[state_df_index] = (
                    unpack_mask_occupancy16(
                        source_packed[state_row_index]
                    )
                )

    if not np.isfinite(occupancy).all():
        raise AssertionError("Non-finite occupancy matrix.")

    print("Occupancy matrix:", occupancy.shape)
    print(
        "Empty SOURCE masks:",
        int((occupancy.sum(axis=1) == 0).sum()),
    )
    print(
        "Full SOURCE masks:",
        int((occupancy.sum(axis=1) == 256).sum()),
    )

    # ------------------------------------------------------------------
    # Load frozen public DINOv2 encoder.
    # Import only AFTER F-drive cache environment has been configured.
    # ------------------------------------------------------------------
    print("\n===== LOAD FROZEN DINOV2 ENCODER =====")

    try:
        import torch
        from transformers import AutoImageProcessor, AutoModel
    except Exception as e:
        raise RuntimeError(
            "R10K2A requires torch + transformers. "
            "No fallback/random encoder is permitted."
        ) from e

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but unavailable. "
            "Run with --device cpu only if intentionally desired."
        )

    device = torch.device(args.device)

    try:
        processor = AutoImageProcessor.from_pretrained(
            MODEL_NAME,
            cache_dir=str(hf_cache),
            local_files_only=args.local_files_only,
        )
        model = AutoModel.from_pretrained(
            MODEL_NAME,
            cache_dir=str(hf_cache),
            local_files_only=args.local_files_only,
        )
    except Exception as e:
        raise RuntimeError(
            "Could not load the frozen public encoder "
            f"{MODEL_NAME}. The script will not substitute another model. "
            f"Project cache: {hf_cache}"
        ) from e

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.to(device)

    hidden = int(getattr(model.config, "hidden_size", -1))
    if hidden != EXPECTED_HIDDEN:
        raise AssertionError(
            f"Expected DINOv2 hidden={EXPECTED_HIDDEN}, got {hidden}"
        )

    print("device:", device)
    print("hidden size:", hidden)
    print("trainable parameters:", sum(p.requires_grad for p in model.parameters()))

    # ------------------------------------------------------------------
    # Extract image CLS once per case and mask-conditioned FG/BG means
    # for all 9 state masks of that case.
    # ------------------------------------------------------------------
    image_cls = np.zeros(
        (1000, EXPECTED_HIDDEN),
        dtype=np.float32,
    )
    fg_features = np.zeros(
        (9000, EXPECTED_HIDDEN),
        dtype=np.float32,
    )
    bg_features = np.zeros(
        (9000, EXPECTED_HIDDEN),
        dtype=np.float32,
    )

    print("\n===== EXTRACT FROZEN DINOV2 REPRESENTATIONS =====")

    batch_size = int(args.batch_size)
    if batch_size < 1:
        raise AssertionError("batch_size must be >= 1.")

    for start in tqdm(
        range(0, len(images), batch_size),
        desc="DINOv2 batches",
        unit="batch",
        dynamic_ncols=True,
    ):
        stop = min(start + batch_size, len(images))
        b = images.iloc[start:stop]

        pil_images = []
        batch_sids = []

        for r in b.itertuples(index=False):
            img_path = Path(getattr(r, "image_path"))
            with Image.open(img_path) as im:
                rgb = im.convert("RGB")
                rgb = rgb.resize(
                    (IMAGE_SIZE, IMAGE_SIZE),
                    resample=Image.Resampling.BICUBIC,
                )
                pil_images.append(rgb.copy())

            batch_sids.append(
                norm_sid_value(getattr(r, "sample_id"))
            )

        # Images have already been resized to 224x224.  Disable processor
        # resize/crop so SOURCE-mask patch occupancy stays spatially aligned.
        inputs = processor(
            images=pil_images,
            return_tensors="pt",
            do_resize=False,
            do_center_crop=False,
        )
        inputs = {
            k: v.to(device)
            for k, v in inputs.items()
        }

        with torch.no_grad():
            outputs = model(**inputs)

        h = outputs.last_hidden_state.detach().float().cpu().numpy()

        if h.ndim != 3:
            raise AssertionError(f"Unexpected last_hidden_state: {h.shape}")
        if h.shape[2] != EXPECTED_HIDDEN:
            raise AssertionError(f"Unexpected hidden dimension: {h.shape}")
        if h.shape[1] != 1 + PATCH_GRID * PATCH_GRID:
            raise AssertionError(
                "Expected 1 CLS + 256 patch tokens for direct 224x224 input, "
                f"got shape {h.shape}"
            )

        cls_batch = h[:, 0, :]
        patch_batch = h[:, 1:, :]

        image_cls[start:stop] = cls_batch

        for j, sid in enumerate(batch_sids):
            state_pos = state_positions_by_sid[sid]
            occ = occupancy[state_pos]

            fg, bg = pooled_region_features(
                patch_batch[j],
                occ,
            )
            fg_features[state_pos] = fg
            bg_features[state_pos] = bg

    if not np.isfinite(image_cls).all():
        raise AssertionError("Non-finite image CLS features.")
    if not np.isfinite(fg_features).all():
        raise AssertionError("Non-finite foreground DINO features.")
    if not np.isfinite(bg_features).all():
        raise AssertionError("Non-finite background DINO features.")

    # ------------------------------------------------------------------
    # Save representation lock.  Still NO outcome labels.
    # ------------------------------------------------------------------
    image_out = out / "R10K2A_image_cls_features.npz"
    state_out = out / "R10K2A_mask_conditioned_dinov2_features.npz"
    meta_out = out / "R10K2A_state_metadata_no_labels.csv"

    np.savez_compressed(
        image_out,
        sample_id=images["sample_id"].astype(str).to_numpy(),
        image_path=images["image_path"].astype(str).to_numpy(),
        cls=image_cls.astype(np.float16),
    )

    np.savez_compressed(
        state_out,
        sample_id=states["sample_id"].astype(str).to_numpy(),
        model_family=states["model_family"].astype(str).to_numpy(),
        model_state_id=states["model_state_id"].astype(str).to_numpy(),
        foreground_patch_mean=fg_features.astype(np.float16),
        background_patch_mean=bg_features.astype(np.float16),
        source_mask_patch_occupancy=occupancy.astype(np.float16),
    )

    states[
        [
            "sample_id",
            "model_family",
            "model_state_id",
            "training_seed",
            "checkpoint_sha256",
            "state_prediction_npz",
            "state_row_index",
        ]
    ].to_csv(meta_out, index=False)

    # ------------------------------------------------------------------
    # Final audits.
    # ------------------------------------------------------------------
    print("\n===== REPRESENTATION AUDIT =====")
    print("Image CLS shape:", image_cls.shape)
    print("Foreground patch mean shape:", fg_features.shape)
    print("Background patch mean shape:", bg_features.shape)
    print("Occupancy shape:", occupancy.shape)

    # Source-mask conditioning must actually induce within-case/model-state
    # variation for at least some cases.
    varying_cases = 0
    for sid, pos in state_positions_by_sid.items():
        a = fg_features[pos]
        if np.max(np.abs(a - a[0:1])) > 1e-6:
            varying_cases += 1

    print(
        "Cases with state-dependent foreground representation:",
        varying_cases,
        "/ 1000",
    )

    if varying_cases == 0:
        raise AssertionError(
            "Mask conditioning produced no model-state-dependent representation."
        )

    lock = {
        "status": "PASS",
        "decision": "DINOV2_IMAGE_MASK_REPRESENTATION_LOCKED",
        "encoder": MODEL_NAME,
        "encoder_hidden_size": EXPECTED_HIDDEN,
        "encoder_trainable": False,
        "image_size": IMAGE_SIZE,
        "patch_grid": [PATCH_GRID, PATCH_GRID],
        "source_mask_size": [MASK_SIZE, MASK_SIZE],
        "mask_block": MASK_BLOCK,
        "source_mask_array_key": "source_masks_packed",
        "a1_mask_values_accessed": False,
        "gt_used": False,
        "dice_used": False,
        "outcome_labels_used": False,
        "pca_fitted": False,
        "safety_model_fitted": False,
        "image_cases": 1000,
        "model_case_rows": 9000,
        "model_states": 9,
        "image_cls_shape": list(image_cls.shape),
        "foreground_patch_mean_shape": list(fg_features.shape),
        "background_patch_mean_shape": list(bg_features.shape),
        "occupancy_shape": list(occupancy.shape),
        "state_dependent_cases": varying_cases,
        "image_feature_file": str(image_out),
        "state_feature_file": str(state_out),
        "metadata_file": str(meta_out),
    }

    with open(
        out / "R10K2A_REPRESENTATION_LOCK.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(lock, f, indent=2, ensure_ascii=False)

    print("\n===== FINAL DECISION =====")
    print("DECISION: DINOV2_IMAGE_MASK_REPRESENTATION_LOCKED")
    print("OUTCOME PERFORMANCE READ: NO")
    print("PCA FIT: NO")
    print("SAFETY HEAD FIT: NO")
    print("A1 mask values accessed: NO")
    print("GT used: NO")
    print()
    print("Outputs:")
    print(image_out)
    print(state_out)
    print(meta_out)
    print(out / "R10K2A_REPRESENTATION_LOCK.json")
    print("PASS")


if __name__ == "__main__":
    main()
