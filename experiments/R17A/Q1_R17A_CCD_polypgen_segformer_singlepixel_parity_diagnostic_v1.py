#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q1_R17A_CCD_polypgen_segformer_singlepixel_parity_diagnostic_v1.py

SafeTTA R17A — targeted diagnostic for the single-pixel SegFormer parity
failure observed at:

  seed   = 20260821
  row    = 1136
  sample = polypgen_static_1136

Observed failure:
  current SOURCE foreground pixels = 21962
  frozen  SOURCE foreground pixels = 21963

PURPOSE
-------
Determine whether the discrepancy is:
  A) true pipeline/state/preprocessing mismatch, or
  B) floating-point threshold instability at a SOURCE logit extremely close
     to zero / environment-level numerical variation.

This script does NOT compute CCD and does NOT relax any gate.

READ-ONLY / NO LEAKAGE
----------------------
Training: NO
TTA: NO
Target GT loading: NO
HARM/outcome loading: NO
Threshold tuning: NO
File modification: NO (except its own diagnostic report)

The script:
1) SHA-validates exact historical scripts/checkpoint/frozen hard lock.
2) SHA-validates the exact PolypGen image bytes.
3) re-runs SOURCE inference repeatedly under the current CUDA configuration;
4) re-runs under a deterministic CUDA configuration when available;
5) runs one CPU inference when feasible;
6) compares packed masks exactly and reports the differing pixel(s), SOURCE
   logit z at those pixels, and how close they are to the 0 threshold.

No tolerance is used to pass anything.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
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

VERSION = "2026-09-09-Q1-R17A-SEGF-SINGLEPIXEL-PARITY-DIAGNOSTIC-v1"
DEFAULT_ROOT = Path(r"F:\MEDSEG_SAFETTA")
OUT_REL = Path(r"outputs\Q1_R17A_CCD_polypgen_segformer_singlepixel_parity_diagnostic_v1")

SEED = 20260821
ROW = 1136
EXPECTED_SAMPLE_ID = "polypgen_static_1136"

R03_REL = Path(r"code\Q1_R03_nine_state_heterogeneous_source_utility_asset_v1_fix2.py")
R03_SHA = "d737272b756a4d2051f9c74051d7085640eaec9cd8be6dbd951c325af2a17c31"

R05_REL = Path(r"code\Q1_R05D3_neopolyp_source_a1_prediction_lock_v1_fix1.py")
R05_SHA = "df22d6f3054257743d1f12d9e757323ea9f4a052b7642eb4ab610da46ff21251"

CKPT_REL = Path(
    r"outputs\Q1_R02_segformer_b0_three_seed_source_training_v1_fix7"
    r"\seed_20260821\best_model_state.pt"
)
CKPT_SHA = "ad75290168eab7d116aa3de61b3eafc1e986f11fc0bbfa4a6108abbf669a258d"

MANIFEST_REL = Path(
    r"outputs\Q1_R10L3A_polypgen_source_a1_prediction_lock_fix2_v1"
    r"\frozen_polypgen_rgb_manifest_no_gt.csv"
)

INDEX_REL = Path(
    r"outputs\Q1_R10L3A_polypgen_source_a1_prediction_lock_fix2_v1"
    r"\state_predictions\segformer_b0_seed20260821_index.csv"
)

HARD_REL = Path(
    r"outputs\Q1_R10L3A_polypgen_source_a1_prediction_lock_fix2_v1"
    r"\state_predictions\segformer_b0_seed20260821_predictions.npz"
)
HARD_SHA = "387fbadd066ab9c668bb59895a22fd2dd1580bad6c4225f9a42cd48a4e325dee"

IMAGE_SIZE = 352
PACKED_BYTES = (IMAGE_SIZE * IMAGE_SIZE) // 8


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_sha(path: Path, expected: str, label: str) -> str:
    if not path.exists():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual.lower() != expected.lower():
        raise RuntimeError(
            f"{label} SHA mismatch\npath={path}\nexpected={expected}\nactual={actual}"
        )
    return actual


def import_from_path(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_image(root: Path, raw: str) -> Path:
    p = Path(str(raw))
    candidates = [p]
    if not p.is_absolute():
        candidates += [
            root / p,
            root / str(raw).replace("/", "\\"),
        ]
    for c in candidates:
        if c.exists() and c.is_file():
            return c
    raise FileNotFoundError(raw)


def infer_source_once(
    image_path: Path,
    device: torch.device,
    r03,
    r05,
    deterministic_mode: bool,
) -> Dict[str, object]:
    # Configure deterministic mode BEFORE model construction/inference.
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = bool(deterministic_mode)
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
        except Exception:
            pass
        try:
            torch.backends.cudnn.allow_tf32 = False
        except Exception:
            pass

    try:
        torch.use_deterministic_algorithms(bool(deterministic_mode))
        deterministic_api = "SET"
    except Exception as e:
        deterministic_api = f"ERROR:{type(e).__name__}:{e}"

    set_seed(SEED)

    seg_context = r03.build_segformer_context(device)
    (
        torch_hist,
        nn,
        F_hist,
        SegformerForSemanticSegmentation,
        config,
        mean,
        std,
    ) = seg_context

    model = r03.load_segformer_state(
        SEED,
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

    with Image.open(image_path) as im:
        native = im.convert("RGB")

    x = r03.segformer_tensor(native, mean, std, torch_hist).to(
        device, non_blocking=False
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

    z = z_source_t[0].detach().float().cpu().numpy()
    sm = np.asarray(r05.logit_to_mask(z), dtype=np.uint8)
    packed = np.asarray(r05.pack_mask(sm), dtype=np.uint8)

    if packed.shape != (PACKED_BYTES,):
        raise RuntimeError(f"Unexpected packed shape {packed.shape}")

    result = {
        "device": str(device),
        "deterministic_mode": bool(deterministic_mode),
        "deterministic_api": deterministic_api,
        "foreground_pixels": int(sm.sum()),
        "z_min": float(z.min()),
        "z_max": float(z.max()),
        "z_min_abs": float(np.min(np.abs(z))),
        "near_zero_le_1e-8": int(np.count_nonzero(np.abs(z) <= 1e-8)),
        "near_zero_le_1e-7": int(np.count_nonzero(np.abs(z) <= 1e-7)),
        "near_zero_le_1e-6": int(np.count_nonzero(np.abs(z) <= 1e-6)),
        "near_zero_le_1e-5": int(np.count_nonzero(np.abs(z) <= 1e-5)),
        "z": z,
        "mask": sm,
        "packed": packed,
    }

    del (
        model, params, source_values, param_names,
        bn_modules, bn_original_track, dropout_modules, dropout_names,
        x, z_source_t,
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return result


def determine_bitorder(current_mask: np.ndarray, current_pack: np.ndarray) -> str:
    flat = current_mask.reshape(-1).astype(np.uint8)
    for bitorder in ["big", "little"]:
        unpacked = np.unpackbits(current_pack, bitorder=bitorder)[: flat.size]
        if np.array_equal(unpacked, flat):
            return bitorder
    raise RuntimeError("Could not infer historical pack_mask bitorder.")


def compare_to_frozen(
    run: Dict[str, object],
    frozen_pack: np.ndarray,
    frozen_fg: int,
) -> Dict[str, object]:
    current_pack = run["packed"]
    current_mask = run["mask"]
    z = run["z"]

    bitorder = determine_bitorder(current_mask, current_pack)
    frozen_flat = np.unpackbits(
        frozen_pack, bitorder=bitorder
    )[: IMAGE_SIZE * IMAGE_SIZE].astype(np.uint8)
    frozen_mask = frozen_flat.reshape(IMAGE_SIZE, IMAGE_SIZE)

    diff = current_mask != frozen_mask
    coords = np.argwhere(diff)

    coord_rows = []
    for y, x in coords[:50]:
        coord_rows.append({
            "y": int(y),
            "x": int(x),
            "current_mask": int(current_mask[y, x]),
            "frozen_mask": int(frozen_mask[y, x]),
            "z_current": float(z[y, x]),
            "abs_z_current": float(abs(z[y, x])),
        })

    out = {
        "packed_exact": bool(np.array_equal(current_pack, frozen_pack)),
        "foreground_pixels": int(run["foreground_pixels"]),
        "frozen_foreground_pixels": int(frozen_fg),
        "foreground_delta": int(run["foreground_pixels"]) - int(frozen_fg),
        "mismatch_pixels": int(coords.shape[0]),
        "mismatch_examples": coord_rows,
        "pack_bitorder": bitorder,
        "z_min_abs": run["z_min_abs"],
        "near_zero_le_1e-8": run["near_zero_le_1e-8"],
        "near_zero_le_1e-7": run["near_zero_le_1e-7"],
        "near_zero_le_1e-6": run["near_zero_le_1e-6"],
        "near_zero_le_1e-5": run["near_zero_le_1e-5"],
    }
    return out


def strip_arrays(run: Dict[str, object]) -> Dict[str, object]:
    return {
        k: v for k, v in run.items()
        if k not in {"z", "mask", "packed"}
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--gpu_reps", type=int, default=3)
    ap.add_argument("--det_gpu_reps", type=int, default=3)
    ap.add_argument(
        "--skip_cpu",
        action="store_true",
        help="Skip the one-sample CPU diagnostic.",
    )
    args = ap.parse_args()

    root = args.root
    if not root.exists():
        raise FileNotFoundError(root)

    out_dir = root / OUT_REL
    out_dir.mkdir(parents=True, exist_ok=True)

    print("===== R17A SEGF SINGLE-PIXEL PARITY DIAGNOSTIC =====")
    print("Version:", VERSION)
    print("Seed:", SEED)
    print("Row:", ROW)
    print("Expected sample:", EXPECTED_SAMPLE_ID)
    print("Training: NO")
    print("TTA: NO")
    print("Target GT loading: NO")
    print("HARM/outcome loading: NO")
    print("Tolerance/pass relaxation: NO")
    print()

    r03_path = root / R03_REL
    r05_path = root / R05_REL
    ckpt_path = root / CKPT_REL
    hard_path = root / HARD_REL
    manifest_path = root / MANIFEST_REL
    index_path = root / INDEX_REL

    validate_sha(r03_path, R03_SHA, "R03")
    validate_sha(r05_path, R05_SHA, "R05")
    validate_sha(ckpt_path, CKPT_SHA, "SegFormer checkpoint")
    validate_sha(hard_path, HARD_SHA, "Frozen PolypGen hard lock")

    r03 = import_from_path(r03_path, "q1_r03_r17a_diag")
    r05 = import_from_path(r05_path, "q1_r05_r17a_diag")

    manifest = pd.read_csv(manifest_path)
    index_df = pd.read_csv(index_path)

    if len(manifest) != 1532 or len(index_df) != 1532:
        raise RuntimeError("Unexpected manifest/index row count.")

    mr = manifest.iloc[ROW]
    ir = index_df.iloc[ROW]

    if str(mr["sample_id"]) != EXPECTED_SAMPLE_ID:
        raise RuntimeError(
            f"Manifest row {ROW} sample={mr['sample_id']} != {EXPECTED_SAMPLE_ID}"
        )
    if str(ir["sample_id"]) != EXPECTED_SAMPLE_ID:
        raise RuntimeError(
            f"Index row {ROW} sample={ir['sample_id']} != {EXPECTED_SAMPLE_ID}"
        )
    if str(mr["image_raw_sha256"]).lower() != str(ir["image_raw_sha256"]).lower():
        raise RuntimeError("Manifest/index image SHA mismatch.")

    image_path = resolve_image(root, str(mr["image_path"]))
    image_sha = sha256_file(image_path)
    if image_sha.lower() != str(mr["image_raw_sha256"]).lower():
        raise RuntimeError("Actual image SHA mismatch.")

    with np.load(hard_path, allow_pickle=False) as z:
        frozen_pack = np.asarray(
            z["source_masks_packed"][ROW], dtype=np.uint8
        )

    frozen_fg = int(ir["source_foreground_pixels"])
    if frozen_pack.shape != (PACKED_BYTES,):
        raise RuntimeError(f"Unexpected frozen pack shape {frozen_pack.shape}")

    runs = []

    # Current/default CUDA repetitions.
    if torch.cuda.is_available() and args.gpu_reps > 0:
        for rep in range(args.gpu_reps):
            print(f"[GPU default] repetition {rep+1}/{args.gpu_reps}")
            run = infer_source_once(
                image_path,
                torch.device("cuda"),
                r03,
                r05,
                deterministic_mode=False,
            )
            cmp = compare_to_frozen(run, frozen_pack, frozen_fg)
            runs.append({
                "mode": "CUDA_DEFAULT",
                "rep": rep,
                **strip_arrays(run),
                **cmp,
            })

    # Deterministic CUDA repetitions.
    if torch.cuda.is_available() and args.det_gpu_reps > 0:
        for rep in range(args.det_gpu_reps):
            print(f"[GPU deterministic] repetition {rep+1}/{args.det_gpu_reps}")
            try:
                run = infer_source_once(
                    image_path,
                    torch.device("cuda"),
                    r03,
                    r05,
                    deterministic_mode=True,
                )
                cmp = compare_to_frozen(run, frozen_pack, frozen_fg)
                runs.append({
                    "mode": "CUDA_DETERMINISTIC",
                    "rep": rep,
                    **strip_arrays(run),
                    **cmp,
                })
            except Exception as e:
                runs.append({
                    "mode": "CUDA_DETERMINISTIC",
                    "rep": rep,
                    "error": f"{type(e).__name__}:{e}",
                })
                break

    # One CPU run to reveal device-level numerical boundary behavior.
    if not args.skip_cpu:
        print("[CPU] one repetition")
        try:
            run = infer_source_once(
                image_path,
                torch.device("cpu"),
                r03,
                r05,
                deterministic_mode=True,
            )
            cmp = compare_to_frozen(run, frozen_pack, frozen_fg)
            runs.append({
                "mode": "CPU_DETERMINISTIC",
                "rep": 0,
                **strip_arrays(run),
                **cmp,
            })
        except Exception as e:
            runs.append({
                "mode": "CPU_DETERMINISTIC",
                "rep": 0,
                "error": f"{type(e).__name__}:{e}",
            })

    # Summarize whether the mismatch is repeatable and threshold-local.
    valid_runs = [r for r in runs if "error" not in r]
    if not valid_runs:
        decision = "BLOCKED_NO_VALID_DIAGNOSTIC_RUN"
    else:
        exact_any = any(bool(r["packed_exact"]) for r in valid_runs)
        mismatch_counts = sorted(set(int(r["mismatch_pixels"]) for r in valid_runs))
        mismatch_z = []
        for r in valid_runs:
            for ex in r.get("mismatch_examples", []):
                mismatch_z.append(abs(float(ex["z_current"])))

        if exact_any:
            decision = "NUMERICAL_ENVIRONMENT_SENSITIVITY_CONFIRMED_EXACT_RUN_EXISTS"
        elif mismatch_z and max(mismatch_z) <= 1e-5 and max(mismatch_counts) <= 4:
            decision = "THRESHOLD_LOCAL_NUMERICAL_PARITY_DIFFERENCE_CONFIRMED"
        else:
            decision = "POTENTIAL_PIPELINE_MISMATCH_REQUIRES_DEEPER_AUDIT"

    report_obj = {
        "version": VERSION,
        "seed": SEED,
        "row": ROW,
        "sample_id": EXPECTED_SAMPLE_ID,
        "image_path": str(image_path),
        "image_sha256": image_sha,
        "checkpoint_sha256": CKPT_SHA,
        "frozen_hard_lock_sha256": HARD_SHA,
        "frozen_foreground_pixels": frozen_fg,
        "information_boundary": {
            "training": False,
            "tta": False,
            "target_gt_loading": False,
            "harm_outcome_loading": False,
            "tolerance_or_pass_relaxation": False,
        },
        "runs": runs,
        "decision": decision,
    }

    json_path = out_dir / "R17A_SEGF_SINGLEPIXEL_PARITY_DIAGNOSTIC.json"
    json_path.write_text(
        json.dumps(report_obj, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lines = [
        "===== R17A SEGF SINGLE-PIXEL PARITY DIAGNOSTIC REPORT =====",
        f"Version: {VERSION}",
        f"seed={SEED}",
        f"row={ROW}",
        f"sample_id={EXPECTED_SAMPLE_ID}",
        f"frozen_foreground_pixels={frozen_fg}",
        f"image_sha256={image_sha}",
        "",
        "RUNS:",
    ]

    for r in runs:
        if "error" in r:
            lines.append(
                f"{r['mode']} rep={r['rep']} ERROR={r['error']}"
            )
            continue

        lines.append(
            f"{r['mode']} rep={r['rep']} "
            f"fg={r['foreground_pixels']} frozen_fg={r['frozen_foreground_pixels']} "
            f"packed_exact={r['packed_exact']} mismatch_pixels={r['mismatch_pixels']} "
            f"z_min_abs={r['z_min_abs']:.12g} "
            f"near<=1e-8:{r['near_zero_le_1e-8']} "
            f"<=1e-7:{r['near_zero_le_1e-7']} "
            f"<=1e-6:{r['near_zero_le_1e-6']} "
            f"<=1e-5:{r['near_zero_le_1e-5']}"
        )
        for ex in r.get("mismatch_examples", [])[:10]:
            lines.append(
                f"  mismatch y={ex['y']} x={ex['x']} "
                f"current={ex['current_mask']} frozen={ex['frozen_mask']} "
                f"z={ex['z_current']:.12g} abs_z={ex['abs_z_current']:.12g}"
            )

    lines += [
        "",
        f"DECISION={decision}",
        "",
        "IMPORTANT:",
        "- This diagnostic does not compute CCD.",
        "- This diagnostic does not relax the bit-exact gate.",
        "- No target GT/HARM/outcome was read.",
        "- If the mismatch is threshold-local (~zero logit), the next step is to",
        "  freeze the current-environment SOURCE soft-logit inference contract",
        "  explicitly rather than pretending historical bit identity was achieved.",
    ]

    report_path = out_dir / "R17A_SEGF_SINGLEPIXEL_PARITY_DIAGNOSTIC_REPORT.txt"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print()
    print(f"DECISION={decision}")
    print("Report:", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
