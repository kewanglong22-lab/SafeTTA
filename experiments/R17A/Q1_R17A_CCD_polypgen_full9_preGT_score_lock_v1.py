#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q1_R17A_CCD_polypgen_full9_preGT_score_lock_v1.py

SafeTTA R17A — COMPLETE PolypGen SicTTA-CCD pre-GT score lock (9/9 states).

This runner combines:
  A) the already locked 6-state DeepLabV3-R50/PraNet CCD table, and
  B) GT-free re-inference for the remaining 3 SegFormer-B0 SOURCE states.

CRITICAL VALIDATION
-------------------
Before any SegFormer CCD score is accepted, this runner requires the freshly
re-inferred SOURCE hard mask to match the existing frozen pre-GT PolypGen
SegFormer SOURCE mask BIT-FOR-BIT for every one of 1532 images and all 3 seeds.

If parity fails:
  - STOP immediately;
  - write no final 9-state CCD score lock.

NO LEAKAGE
----------
Training: NO
TTA: NO
Target GT loading: NO
HARM/outcome loading: NO
Threshold tuning: NO

RNG CONTINUITY
--------------
The previous 6-state lock used one torch.Generator(seed=20260908) across a
fixed 6-state traversal. To preserve that frozen RNG contract exactly, this
runner:
  1) replays/recomputes those same 6 states using the same helper;
  2) requires exact equality to the previously locked 6-state CCD values;
  3) continues the SAME generator stream for the 3 SegFormer states.

Therefore the final 9-state panel has one deterministic continuous sampling
stream without changing the already locked 6-state results.

Output:
  outputs/Q1_R17A_CCD_polypgen_full9_preGT_score_lock_v1/
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


VERSION = "2026-09-08-Q1-R17A-CCD-POLYPGEN-FULL9-PREGT-SCORE-LOCK-v1"
DEFAULT_ROOT = Path(r"F:\MEDSEG_SAFETTA")
OUT_REL = Path(r"outputs\Q1_R17A_CCD_polypgen_full9_preGT_score_lock_v1")

# Exact previously generated R17A partial-lock code and artifact.
PARTIAL_SCRIPT_REL = Path(r"code\Q1_R17A_CCD_polypgen_6state_preGT_score_lock_v1.py")
PARTIAL_SCRIPT_SHA256 = "cac9d2540514a6a76bc625b7b7d3fa6bdc258885c00d028bb3ca765a6f78a79c"

PARTIAL_DIR_REL = Path(r"outputs\Q1_R17A_CCD_polypgen_6state_preGT_score_lock_v1")
PARTIAL_SCORE_NAME = "R17A_POLYPGEN_CCD_6STATE_PREGT_SCORE_LOCK.csv"
PARTIAL_SCORE_SHA256 = "85a2e5d913ddbb88a88c8df09b7ef88822963ea9a1451eda6c6254f95b151786"
PARTIAL_INPUT_LOCK_NAME = "R17A_POLYPGEN_CCD_INPUT_ASSET_LOCK.json"
EXPECTED_PARTIAL_MAPPING_SHA256 = "f10f17a633289403949240378d9447e82998b543f18eb06234e9bf0a5e367ce3"

# Historical exact SegFormer implementation.
R03_REL = Path(r"code\Q1_R03_nine_state_heterogeneous_source_utility_asset_v1_fix2.py")
R03_SHA256 = "d737272b756a4d2051f9c74051d7085640eaec9cd8be6dbd951c325af2a17c31"

R05_REL = Path(r"code\Q1_R05D3_neopolyp_source_a1_prediction_lock_v1_fix1.py")
R05_SHA256 = "df22d6f3054257743d1f12d9e757323ea9f4a052b7642eb4ab610da46ff21251"

MANIFEST_REL = Path(
    r"outputs\Q1_R10L3A_polypgen_source_a1_prediction_lock_fix2_v1"
    r"\frozen_polypgen_rgb_manifest_no_gt.csv"
)

STATE_DIR_REL = Path(
    r"outputs\Q1_R10L3A_polypgen_source_a1_prediction_lock_fix2_v1"
    r"\state_predictions"
)

EXPECTED_MANIFEST_ROWS = 1532
EXPECTED_FULL_ROWS = 1532 * 9
EXPECTED_PARTIAL_ROWS = 1532 * 6
IMAGE_SIZE = 352
PACKED_BYTES = (IMAGE_SIZE * IMAGE_SIZE + 7) // 8

RNG_SEED = 20260908

SEGFORMER = {
    20260820: {
        "checkpoint_rel": Path(
            r"outputs\Q1_R02_segformer_b0_three_seed_source_training_v1_fix7"
            r"\seed_20260820\best_model_state.pt"
        ),
        "checkpoint_sha256": "9dae2ae907b193ea36c2bccc8e376ddbad1699ba27ae0769cf40bcba13e95605",
        "hard_npz": "segformer_b0_seed20260820_predictions.npz",
        "hard_npz_sha256": "d36ad27453e0532516f955c5c4d55e7282f38a04426aaf58bfb10bf5c9d089e0",
        "index_csv": "segformer_b0_seed20260820_index.csv",
    },
    20260821: {
        "checkpoint_rel": Path(
            r"outputs\Q1_R02_segformer_b0_three_seed_source_training_v1_fix7"
            r"\seed_20260821\best_model_state.pt"
        ),
        "checkpoint_sha256": "ad75290168eab7d116aa3de61b3eafc1e986f11fc0bbfa4a6108abbf669a258d",
        "hard_npz": "segformer_b0_seed20260821_predictions.npz",
        "hard_npz_sha256": "387fbadd066ab9c668bb59895a22fd2dd1580bad6c4225f9a42cd48a4e325dee",
        "index_csv": "segformer_b0_seed20260821_index.csv",
    },
    20260822: {
        "checkpoint_rel": Path(
            r"outputs\Q1_R02_segformer_b0_three_seed_source_training_v1_fix7"
            r"\seed_20260822\best_model_state.pt"
        ),
        "checkpoint_sha256": "ea0b3881373e9f966475a082490fabe4b0acae81179b8596c48a06ee2661a34a",
        "hard_npz": "segformer_b0_seed20260822_predictions.npz",
        "hard_npz_sha256": "a26d7c9eb9cb11055779410106d5f1e0476b3990609a0698ad7005e24b989611",
        "index_csv": "segformer_b0_seed20260822_index.csv",
    },
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def import_from_path(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def validate_sha(path: Path, expected: str, label: str) -> str:
    if not path.exists():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(
            f"{label} SHA256 mismatch\n"
            f"path={path}\nexpected={expected}\nactual={actual}"
        )
    return actual


def norm_path_text(x) -> str:
    s = str(x).strip().replace("\\", "/")
    while "//" in s:
        s = s.replace("//", "/")
    return s.lower()


def set_runtime_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_image_path(root: Path, raw: str) -> Path:
    p = Path(str(raw))
    candidates = [p]
    if not p.is_absolute():
        candidates.extend([
            root / p,
            root / str(raw).replace("/", "\\"),
        ])
    for c in candidates:
        if c.exists() and c.is_file():
            return c
    raise FileNotFoundError(f"PolypGen image not found: {raw}")


def check_manifest_no_gt(df: pd.DataFrame) -> None:
    banned = {
        "gt", "ground_truth", "groundtruth", "mask_gt", "gt_mask",
        "dice", "delta_dice", "harm", "harmful", "beneficial", "neutral",
    }
    cols = {str(c).strip().lower() for c in df.columns}
    hit = sorted(cols & banned)
    if hit:
        raise RuntimeError(f"GT/outcome columns present in pre-GT manifest: {hit}")


def validate_index_against_manifest(
    manifest: pd.DataFrame,
    index_df: pd.DataFrame,
    seed: int,
    expected_ckpt_sha: str,
) -> None:
    if len(index_df) != EXPECTED_MANIFEST_ROWS:
        raise RuntimeError(
            f"SegFormer seed={seed}: index rows={len(index_df)}, "
            f"expected={EXPECTED_MANIFEST_ROWS}"
        )

    required = {
        "row_index", "sample_id", "image_path", "image_raw_sha256",
        "model_family", "model_state_id", "training_seed",
        "checkpoint_sha256", "source_foreground_pixels",
    }
    missing = sorted(required - set(index_df.columns))
    if missing:
        raise RuntimeError(f"SegFormer seed={seed}: index missing columns {missing}")

    if list(index_df["row_index"].astype(int)) != list(range(EXPECTED_MANIFEST_ROWS)):
        raise RuntimeError(f"SegFormer seed={seed}: row_index is not canonical 0..1531")

    if not np.array_equal(
        index_df["sample_id"].astype(str).to_numpy(),
        manifest["sample_id"].astype(str).to_numpy(),
    ):
        raise RuntimeError(f"SegFormer seed={seed}: sample_id order differs from manifest")

    if not np.array_equal(
        index_df["image_raw_sha256"].astype(str).str.lower().to_numpy(),
        manifest["image_raw_sha256"].astype(str).str.lower().to_numpy(),
    ):
        raise RuntimeError(f"SegFormer seed={seed}: image SHA order differs from manifest")

    if "image_path" in manifest.columns:
        a = np.asarray([norm_path_text(x) for x in index_df["image_path"]])
        b = np.asarray([norm_path_text(x) for x in manifest["image_path"]])
        if not np.array_equal(a, b):
            raise RuntimeError(f"SegFormer seed={seed}: image_path order differs from manifest")

    if set(index_df["training_seed"].astype(int).unique()) != {seed}:
        raise RuntimeError(f"SegFormer seed={seed}: training_seed mismatch in frozen index")

    if set(index_df["checkpoint_sha256"].astype(str).str.lower().unique()) != {
        expected_ckpt_sha.lower()
    }:
        raise RuntimeError(f"SegFormer seed={seed}: checkpoint SHA mismatch in frozen index")

    fam = set(index_df["model_family"].astype(str).str.lower().unique())
    if fam != {"segformer-b0"}:
        raise RuntimeError(f"SegFormer seed={seed}: unexpected model family {fam}")


def verify_actual_images_once(root: Path, manifest: pd.DataFrame) -> List[Path]:
    paths: List[Path] = []
    iterator = range(len(manifest))
    if tqdm is not None:
        iterator = tqdm(
            iterator,
            total=len(manifest),
            desc="Verify PolypGen image SHA256",
            unit="img",
            dynamic_ncols=True,
        )

    for i in iterator:
        row = manifest.iloc[i]
        p = resolve_image_path(root, str(row["image_path"]))
        actual = sha256_file(p)
        expected = str(row["image_raw_sha256"]).lower()
        if actual.lower() != expected:
            raise RuntimeError(
                f"PolypGen image SHA mismatch at row={i}, sample={row['sample_id']}\n"
                f"path={p}\nexpected={expected}\nactual={actual}"
            )
        paths.append(p)
    return paths


def replay_partial6_and_advance_rng(
    root: Path,
    manifest: pd.DataFrame,
    partial_mod,
    generator: torch.Generator,
    locked6: pd.DataFrame,
) -> dict:
    """
    Recompute the 6 direct-logit states with the exact prior helper.
    This both validates the prior lock and advances RNG to the precise point
    where SegFormer state 7 must start.
    """
    rows = []
    map_sha = None
    extras = None

    iterator = partial_mod.UNITS
    if tqdm is not None:
        iterator = tqdm(
            iterator,
            desc="Replay locked 6-state CCD RNG stream",
            unit="state",
            dynamic_ncols=True,
        )

    for family, seed, relp in iterator:
        asset_path = root / partial_mod.S06_REL / relp
        if not asset_path.exists():
            raise FileNotFoundError(asset_path)

        asset_sha = sha256_file(asset_path)

        with np.load(asset_path, allow_pickle=False) as z:
            align = partial_mod.resolve_alignment(manifest, z)
            row_indices = list(align["row_indices"])
            current_map_sha = partial_mod.mapping_sha256(manifest, row_indices)

            if map_sha is None:
                map_sha = current_map_sha
                extras = list(align["extra_asset_rows"])
            else:
                if current_map_sha != map_sha:
                    raise RuntimeError("Partial six states no longer share identical alignment.")
                if list(align["extra_asset_rows"]) != extras:
                    raise RuntimeError("Partial six states no longer share identical extra rows.")

            logits = z["source_logits"]

            loop = range(EXPECTED_MANIFEST_ROWS)
            if tqdm is not None:
                loop = tqdm(
                    loop,
                    desc=f"Replay CCD {family} {seed}",
                    unit="img",
                    dynamic_ncols=True,
                    leave=False,
                )

            for m_i in loop:
                src_i = int(row_indices[m_i])
                ccd = partial_mod.ccd_from_binary_logits(logits[src_i], generator)
                mr = manifest.iloc[m_i]
                rows.append({
                    "sample_id": str(mr["sample_id"]),
                    "model_family": family,
                    "model_state_id": f"{family}::{seed}",
                    "training_seed": seed,
                    "ccd_risk": ccd,
                    "source_asset_sha256": asset_sha,
                    "source_asset_row_index": src_i,
                })

    if map_sha != EXPECTED_PARTIAL_MAPPING_SHA256:
        raise RuntimeError(
            f"Partial alignment SHA drift: expected={EXPECTED_PARTIAL_MAPPING_SHA256}, "
            f"actual={map_sha}"
        )

    replay = pd.DataFrame(rows)
    if len(replay) != EXPECTED_PARTIAL_ROWS:
        raise RuntimeError(f"Partial replay rows={len(replay)}, expected={EXPECTED_PARTIAL_ROWS}")

    keys = ["sample_id", "model_family", "model_state_id", "training_seed"]
    for k in keys:
        if not np.array_equal(
            replay[k].astype(str).to_numpy(),
            locked6[k].astype(str).to_numpy(),
        ):
            raise RuntimeError(f"Partial six-state replay key mismatch: {k}")

    a = replay["ccd_risk"].to_numpy(dtype=np.float64)
    b = locked6["ccd_risk"].to_numpy(dtype=np.float64)

    if not np.array_equal(a, b):
        max_abs = float(np.max(np.abs(a - b)))
        bad = int(np.count_nonzero(a != b))
        raise RuntimeError(
            f"Partial six-state CCD replay not bit-identical: bad={bad}, max_abs={max_abs}"
        )

    return {
        "rows": len(replay),
        "alignment_sha256": map_sha,
        "excluded_extra_rows": extras,
        "ccd_exact_match": True,
    }


def infer_one_segformer_state(
    root: Path,
    manifest: pd.DataFrame,
    image_paths: List[Path],
    seed: int,
    meta: dict,
    r03,
    r05,
    seg_context,
    generator: torch.Generator,
    device: torch.device,
) -> Tuple[pd.DataFrame, dict]:
    ckpt = root / meta["checkpoint_rel"]
    validate_sha(
        ckpt,
        meta["checkpoint_sha256"],
        f"SegFormer checkpoint seed={seed}",
    )

    state_dir = root / STATE_DIR_REL
    hard_path = state_dir / meta["hard_npz"]
    validate_sha(
        hard_path,
        meta["hard_npz_sha256"],
        f"PolypGen SegFormer hard lock seed={seed}",
    )

    index_path = state_dir / meta["index_csv"]
    if not index_path.exists():
        raise FileNotFoundError(index_path)

    index_df = pd.read_csv(index_path)
    validate_index_against_manifest(
        manifest, index_df, seed, meta["checkpoint_sha256"]
    )

    with np.load(hard_path, allow_pickle=False) as z:
        if "source_masks_packed" not in z.files:
            raise RuntimeError(f"{hard_path}: source_masks_packed missing")
        frozen_pack = np.asarray(z["source_masks_packed"], dtype=np.uint8)

    if frozen_pack.shape != (EXPECTED_MANIFEST_ROWS, PACKED_BYTES):
        raise RuntimeError(
            f"seed={seed}: frozen source mask shape={frozen_pack.shape}, "
            f"expected=({EXPECTED_MANIFEST_ROWS},{PACKED_BYTES})"
        )

    (
        torch_hist,
        nn,
        F_hist,
        SegformerForSemanticSegmentation,
        config,
        mean,
        std,
    ) = seg_context

    if torch_hist is not torch:
        # This should normally be the same imported torch module.
        raise RuntimeError("Historical SegFormer context returned unexpected torch module.")

    set_runtime_seed(seed)

    model = r03.load_segformer_state(
        seed,
        device,
        SegformerForSemanticSegmentation,
        config,
        torch_hist,
    )

    (
        params,
        param_names,
        bn_modules,
        bn_original_track,
        dropout_modules,
        dropout_names,
    ) = r03.configure_segformer_tent(model, nn)

    source_values = r03.snapshot_params(params)

    score_rows = []
    parity_rows = []
    mismatch_count = 0

    iterator = range(EXPECTED_MANIFEST_ROWS)
    if tqdm is not None:
        iterator = tqdm(
            iterator,
            total=EXPECTED_MANIFEST_ROWS,
            desc=f"SegFormer source+CCD seed {seed}",
            unit="img",
            dynamic_ncols=True,
        )

    for i in iterator:
        row = manifest.iloc[i]
        with Image.open(image_paths[i]) as im:
            native = im.convert("RGB")

        x = r03.segformer_tensor(native, mean, std, torch_hist).to(
            device, non_blocking=True
        )

        r03.restore_params(params, source_values, torch_hist)
        r03.segformer_source_mode(
            model,
            bn_modules,
            bn_original_track,
            dropout_modules,
        )

        with torch_hist.no_grad():
            _, z_source_t = r03.segformer_logits_and_z(model, x, F_hist)

        z_source = z_source_t[0].detach().float().cpu().numpy()

        # Exact historical threshold/packing functions.
        sm = r05.logit_to_mask(z_source)
        packed = np.asarray(r05.pack_mask(sm), dtype=np.uint8)

        if packed.shape != (PACKED_BYTES,):
            raise RuntimeError(
                f"seed={seed} row={i}: packed shape={packed.shape}, expected=({PACKED_BYTES},)"
            )

        frozen = frozen_pack[i]
        exact = bool(np.array_equal(packed, frozen))

        current_fg = int(np.asarray(sm, dtype=np.uint8).sum())
        frozen_fg = int(index_df.iloc[i]["source_foreground_pixels"])

        if current_fg != frozen_fg:
            raise RuntimeError(
                f"BITEXACT_PARITY_FAIL_FOREGROUND_COUNT seed={seed} row={i} "
                f"sample={row['sample_id']} current_fg={current_fg} frozen_fg={frozen_fg}"
            )

        if not exact:
            diff_bytes = int(np.count_nonzero(packed != frozen))
            mismatch_count += 1
            raise RuntimeError(
                f"BITEXACT_PARITY_FAIL_PACKED_MASK seed={seed} row={i} "
                f"sample={row['sample_id']} differing_bytes={diff_bytes}. "
                "STOP: no CCD score lock written."
            )

        # Same exact CCD helper as the already locked first six states.
        # z = logit_fg - logit_bg, and sigmoid(z) is exactly the foreground
        # probability of the two-class softmax up to floating arithmetic.
        ccd = partial_mod_global.ccd_from_binary_logits(z_source, generator)

        score_rows.append({
            "row_index": 0,  # assigned after concatenation
            "sample_id": str(row["sample_id"]),
            "original_polypgen_sample_id": str(row["original_polypgen_sample_id"]),
            "center": str(row["center"]),
            "image_path": str(row["image_path"]),
            "image_raw_sha256": str(row["image_raw_sha256"]),
            "model_family": "SegFormer-B0",
            "model_state_id": f"SegFormer-B0::{seed}",
            "training_seed": seed,
            "checkpoint_sha256": meta["checkpoint_sha256"],
            "ccd_risk": ccd,
        })

        parity_rows.append({
            "row_index": i,
            "sample_id": str(row["sample_id"]),
            "training_seed": seed,
            "packed_mask_exact": True,
            "source_foreground_pixels": current_fg,
            "frozen_source_foreground_pixels": frozen_fg,
        })

        del x, z_source_t, z_source, sm, packed

    r03.restore_params(params, source_values, torch_hist)

    del (
        model, params, source_values, param_names,
        bn_modules, bn_original_track, dropout_modules, dropout_names,
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if mismatch_count != 0:
        raise RuntimeError(f"Unexpected SegFormer parity mismatches: {mismatch_count}")

    score_df = pd.DataFrame(score_rows)
    parity_df = pd.DataFrame(parity_rows)

    return score_df, {
        "seed": seed,
        "checkpoint_sha256": meta["checkpoint_sha256"],
        "hard_lock_npz": str(hard_path.relative_to(root)).replace("\\", "/"),
        "hard_lock_npz_sha256": meta["hard_npz_sha256"],
        "index_csv": str(index_path.relative_to(root)).replace("\\", "/"),
        "index_csv_sha256": sha256_file(index_path),
        "rows": len(score_df),
        "bitexact_source_mask_matches": len(score_df),
        "bitexact_source_mask_mismatches": 0,
        "parity_df": parity_df,
    }


def main() -> int:
    global partial_mod_global

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Inference device. Default: cuda",
    )
    args = ap.parse_args()

    root = args.root
    if not root.exists():
        raise FileNotFoundError(root)

    out_dir = root / OUT_REL
    out_dir.mkdir(parents=True, exist_ok=True)

    print("===== R17A POLYPGEN CCD FULL 9-STATE PRE-GT SCORE LOCK =====")
    print("Version:", VERSION)
    print("Root:", root)
    print("Training: NO")
    print("TTA: NO")
    print("Target GT loading: NO")
    print("HARM/outcome loading: NO")
    print("Threshold tuning: NO")
    print("SegFormer parity requirement: BIT-FOR-BIT frozen SOURCE mask")
    print(f"CCD RNG seed: {RNG_SEED}")
    print()

    # ------------------------------------------------------------------
    # 1. Validate/import exact helper implementations.
    # ------------------------------------------------------------------
    partial_script = root / PARTIAL_SCRIPT_REL
    validate_sha(
        partial_script,
        PARTIAL_SCRIPT_SHA256,
        "R17A 6-state helper script",
    )
    partial_mod_global = import_from_path(
        partial_script,
        "q1_r17a_ccd_partial6_exact",
    )

    r03_path = root / R03_REL
    r05_path = root / R05_REL
    validate_sha(r03_path, R03_SHA256, "R03 SegFormer implementation")
    validate_sha(r05_path, R05_SHA256, "R05D3 prediction-lock implementation")

    r03 = import_from_path(r03_path, "q1_r03_exact_for_r17a")
    r05 = import_from_path(r05_path, "q1_r05d3_exact_for_r17a")

    required_r03 = [
        "build_segformer_context",
        "load_segformer_state",
        "configure_segformer_tent",
        "snapshot_params",
        "restore_params",
        "segformer_source_mode",
        "segformer_tensor",
        "segformer_logits_and_z",
    ]
    required_r05 = ["logit_to_mask", "pack_mask"]

    for name in required_r03:
        if not hasattr(r03, name):
            raise RuntimeError(f"Historical R03 helper missing: {name}")
    for name in required_r05:
        if not hasattr(r05, name):
            raise RuntimeError(f"Historical R05 helper missing: {name}")

    # ------------------------------------------------------------------
    # 2. Load/validate frozen no-GT manifest + prior 6-state lock.
    # ------------------------------------------------------------------
    manifest_path = root / MANIFEST_REL
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    manifest = pd.read_csv(manifest_path)
    check_manifest_no_gt(manifest)

    if len(manifest) != EXPECTED_MANIFEST_ROWS:
        raise RuntimeError(
            f"PolypGen manifest rows={len(manifest)}, expected={EXPECTED_MANIFEST_ROWS}"
        )

    required_manifest_cols = {
        "sample_id", "original_polypgen_sample_id", "center",
        "image_path", "image_raw_sha256",
    }
    missing = sorted(required_manifest_cols - set(manifest.columns))
    if missing:
        raise RuntimeError(f"PolypGen no-GT manifest missing columns: {missing}")

    if manifest["sample_id"].astype(str).duplicated().any():
        raise RuntimeError("PolypGen no-GT manifest sample_id not unique.")

    partial_score_path = root / PARTIAL_DIR_REL / PARTIAL_SCORE_NAME
    validate_sha(
        partial_score_path,
        PARTIAL_SCORE_SHA256,
        "Previously locked 6-state CCD score table",
    )
    locked6 = pd.read_csv(partial_score_path)

    if len(locked6) != EXPECTED_PARTIAL_ROWS:
        raise RuntimeError(
            f"Locked 6-state score rows={len(locked6)}, expected={EXPECTED_PARTIAL_ROWS}"
        )

    partial_input_lock = root / PARTIAL_DIR_REL / PARTIAL_INPUT_LOCK_NAME
    if not partial_input_lock.exists():
        raise FileNotFoundError(partial_input_lock)
    partial_input = json.loads(
        partial_input_lock.read_text(encoding="utf-8")
    )
    if partial_input.get("common_alignment_mapping_sha256") != EXPECTED_PARTIAL_MAPPING_SHA256:
        raise RuntimeError("Previously locked partial alignment SHA differs from frozen value.")

    # ------------------------------------------------------------------
    # 3. Validate actual images once.
    # ------------------------------------------------------------------
    print("[1/5] Verify frozen PolypGen image bytes...")
    image_paths = verify_actual_images_once(root, manifest)

    # ------------------------------------------------------------------
    # 4. Re-run first 6 CCD states ONLY to prove lock + advance RNG.
    # ------------------------------------------------------------------
    print("[2/5] Replay first 6 CCD states and continue frozen RNG stream...")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(RNG_SEED)

    partial_replay = replay_partial6_and_advance_rng(
        root, manifest, partial_mod_global, generator, locked6
    )

    # ------------------------------------------------------------------
    # 5. Build historical SegFormer context and infer three source states.
    # ------------------------------------------------------------------
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but torch.cuda.is_available() is False. "
            "Use --device cpu only if you intentionally want CPU inference."
        )

    device = torch.device(args.device)

    print("[3/5] Build exact historical SegFormer context...")
    seg_context = r03.build_segformer_context(device)

    seg_parts = []
    parity_meta = []
    parity_csv_paths = []

    print("[4/5] Run 3 SegFormer SOURCE states with bit-exact parity gate...")
    for seed in [20260820, 20260821, 20260822]:
        part, meta = infer_one_segformer_state(
            root=root,
            manifest=manifest,
            image_paths=image_paths,
            seed=seed,
            meta=SEGFORMER[seed],
            r03=r03,
            r05=r05,
            seg_context=seg_context,
            generator=generator,
            device=device,
        )

        parity_df = meta.pop("parity_df")
        parity_path = out_dir / f"R17A_POLYPGEN_SEGF_{seed}_BITEXACT_PARITY.csv"
        parity_df.to_csv(parity_path, index=False)
        meta["parity_csv"] = parity_path.name
        meta["parity_csv_sha256"] = sha256_file(parity_path)
        parity_csv_paths.append(parity_path)

        seg_parts.append(part)
        parity_meta.append(meta)

    seg3 = pd.concat(seg_parts, ignore_index=True)

    if len(seg3) != EXPECTED_MANIFEST_ROWS * 3:
        raise RuntimeError(
            f"SegFormer CCD rows={len(seg3)}, expected={EXPECTED_MANIFEST_ROWS * 3}"
        )

    if seg3.duplicated(["sample_id", "model_state_id"]).any():
        raise RuntimeError("Duplicate sample/model-state rows in SegFormer CCD.")

    if not np.isfinite(seg3["ccd_risk"].to_numpy(dtype=float)).all():
        raise RuntimeError("Non-finite SegFormer CCD.")

    # ------------------------------------------------------------------
    # 6. Final pre-GT 9-state lock.
    # ------------------------------------------------------------------
    print("[5/5] Assemble final 9-state PRE-GT score lock...")

    # Keep frozen original 6-state rows exactly; append newly validated SegFormer.
    common_cols = [
        "sample_id",
        "original_polypgen_sample_id",
        "center",
        "image_path",
        "image_raw_sha256",
        "model_family",
        "model_state_id",
        "training_seed",
        "ccd_risk",
    ]

    for col in common_cols:
        if col not in locked6.columns:
            raise RuntimeError(f"Locked six-state table missing column: {col}")
        if col not in seg3.columns:
            raise RuntimeError(f"SegFormer table missing column: {col}")

    full9 = pd.concat(
        [
            locked6[common_cols].copy(),
            seg3[common_cols].copy(),
        ],
        ignore_index=True,
    )
    full9.insert(0, "row_index", np.arange(len(full9), dtype=np.int64))

    if len(full9) != EXPECTED_FULL_ROWS:
        raise RuntimeError(f"Full 9-state rows={len(full9)}, expected={EXPECTED_FULL_ROWS}")

    if full9["sample_id"].nunique() != EXPECTED_MANIFEST_ROWS:
        raise RuntimeError("Full 9-state unique sample count mismatch.")

    state_counts = (
        full9.groupby(["model_family", "training_seed"])
        .size()
        .to_dict()
    )
    if len(state_counts) != 9 or any(v != EXPECTED_MANIFEST_ROWS for v in state_counts.values()):
        raise RuntimeError(f"Full panel state counts invalid: {state_counts}")

    if full9.duplicated(["sample_id", "model_state_id"]).any():
        raise RuntimeError("Duplicate sample/model-state rows in full 9-state panel.")

    if not np.isfinite(full9["ccd_risk"].to_numpy(dtype=float)).all():
        raise RuntimeError("Non-finite CCD in full 9-state panel.")

    seg3_path = out_dir / "R17A_POLYPGEN_CCD_SEGF_3STATE_PREGT_SCORE_LOCK.csv"
    seg3.to_csv(seg3_path, index=False)

    full_path = out_dir / "R17A_POLYPGEN_CCD_FULL9_PREGT_SCORE_LOCK.csv"
    full9.to_csv(full_path, index=False)

    summary = (
        full9.groupby(["model_family", "training_seed"], as_index=False)
        .agg(
            rows=("ccd_risk", "size"),
            ccd_mean=("ccd_risk", "mean"),
            ccd_std=("ccd_risk", "std"),
            ccd_min=("ccd_risk", "min"),
            ccd_max=("ccd_risk", "max"),
        )
    )
    summary_path = out_dir / "R17A_POLYPGEN_CCD_FULL9_STATE_SUMMARY.csv"
    summary.to_csv(summary_path, index=False)

    lock = {
        "status": "PASS",
        "decision": "PASS_R17A_POLYPGEN_CCD_FULL9_PREGT_SCORE_LOCK",
        "version": VERSION,
        "dataset": "PolypGen",
        "information_boundary": {
            "training": False,
            "tta": False,
            "target_gt_loading": False,
            "harm_outcome_loading": False,
            "threshold_tuning": False,
        },
        "ccd": {
            "definition": "SicTTA official-code-faithful binary instantiation",
            "rng_seed": RNG_SEED,
            "rng_stream": "continuous: replay locked six states, then 3 SegFormer states",
            "sample_pixels": 200,
            "risk_direction": "higher CCD = higher HARM risk",
            "helper_script": str(PARTIAL_SCRIPT_REL).replace("\\", "/"),
            "helper_script_sha256": PARTIAL_SCRIPT_SHA256,
        },
        "partial6_replay": partial_replay,
        "segformer_contract": {
            "r03_script": str(R03_REL).replace("\\", "/"),
            "r03_sha256": R03_SHA256,
            "r05_script": str(R05_REL).replace("\\", "/"),
            "r05_sha256": R05_SHA256,
            "states": parity_meta,
            "bitexact_source_mask_required": True,
            "total_bitexact_matches": EXPECTED_MANIFEST_ROWS * 3,
            "total_mismatches": 0,
        },
        "manifest": str(MANIFEST_REL).replace("\\", "/"),
        "manifest_sha256": sha256_file(manifest_path),
        "rows": len(full9),
        "unique_samples": int(full9["sample_id"].nunique()),
        "states": 9,
        "score_table": full_path.name,
        "score_table_sha256": sha256_file(full_path),
        "segformer_score_table": seg3_path.name,
        "segformer_score_table_sha256": sha256_file(seg3_path),
        "state_summary": summary_path.name,
        "state_summary_sha256": sha256_file(summary_path),
        "next": "JOIN_ONLY_FROZEN_POLYPGEN_HARM_OUTCOME_AND_EVALUATE_CCD",
    }

    lock_path = out_dir / "R17A_POLYPGEN_CCD_FULL9_PREGT_LOCK.json"
    lock_path.write_text(
        json.dumps(lock, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    report_lines = [
        "===== R17A POLYPGEN CCD FULL 9-STATE PRE-GT SCORE LOCK =====",
        f"Version: {VERSION}",
        "Dataset: PolypGen",
        f"Rows: {len(full9)}",
        f"Unique samples: {full9['sample_id'].nunique()}",
        "States: 9/9",
        f"CCD RNG seed: {RNG_SEED}",
        f"Partial 6-state replay exact: {partial_replay['ccd_exact_match']}",
        f"Partial alignment SHA256: {partial_replay['alignment_sha256']}",
        f"SegFormer bit-exact SOURCE-mask matches: {EXPECTED_MANIFEST_ROWS * 3}",
        "SegFormer bit-exact SOURCE-mask mismatches: 0",
        f"Full score SHA256: {sha256_file(full_path)}",
        "",
        "Information boundary:",
        "  Training: NO",
        "  TTA: NO",
        "  Target GT loading: NO",
        "  HARM/outcome loading: NO",
        "  Threshold tuning: NO",
        "",
        "GATE=PASS_R17A_POLYPGEN_CCD_FULL9_PREGT_SCORE_LOCK",
        "NEXT=JOIN_ONLY_FROZEN_POLYPGEN_HARM_OUTCOME_AND_EVALUATE_CCD",
    ]

    report_path = out_dir / "R17A_POLYPGEN_CCD_FULL9_PREGT_REPORT.txt"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print()
    for line in report_lines[-10:]:
        print(line)
    print("Report:", report_path)
    return 0


# Global handle is assigned only after SHA-validated import inside main.
partial_mod_global = None

if __name__ == "__main__":
    raise SystemExit(main())
