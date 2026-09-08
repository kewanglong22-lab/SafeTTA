#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q1_R17A_CCD_polypgen_6state_preGT_score_lock_v1.py

SafeTTA R17A — PolypGen SicTTA-CCD pre-GT score lock, PARTIAL 6/9 panel.

WHY PARTIAL
-----------
The frozen PolypGen S06_C assets contain SOURCE logits for exactly:
  - DeepLabV3-R50 seeds 20260817/18/19
  - PraNet        seeds 20260817/18/19

The final PolypGen paper panel has 9 model states, so the three SegFormer-B0
states are deliberately NOT approximated from hard masks. They will require
GT-free SOURCE re-inference in the next step.

This script therefore:
  1) exact-aligns each 1537-row S06_C logit asset to the frozen 1532-image
     PolypGen no-GT manifest;
  2) computes official-code-faithful SicTTA CCD from SOURCE logits only;
  3) serializes + SHA256-locks the 6-state score table BEFORE any GT join;
  4) performs NO evaluation against HARM labels.

NO LEAKAGE
----------
Training: NO
TTA: NO
Target GT loading: NO
HARM/outcome loading: NO
Threshold tuning: NO

CCD
---
For each SOURCE logit map:
  p = sigmoid(logit)
  P = [1-p, p] at each pixel
  random min(200,N) pixels via torch.randperm
  L2 normalize pixel probability vectors
  G = P^T P
  Q = softmax(G, dim=1)
  CCD = mean row entropy[-sum Q log(Q+1e-5)]

Lower CCD = more source-friendly in SicTTA.
R17A uses CCD directly as HARM-risk direction.

Frozen RNG:
  global seed = 20260908
  deterministic state order
  frozen manifest row order
  one official-style random draw per state/sample

Output:
  outputs/Q1_R17A_CCD_polypgen_6state_preGT_score_lock_v1/
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

VERSION = "2026-09-08-Q1-R17A-CCD-POLYPGEN-6STATE-PREGT-SCORE-LOCK-v1"
DEFAULT_ROOT = Path(r"F:\MEDSEG_SAFETTA")
OUT_REL = Path(r"outputs\Q1_R17A_CCD_polypgen_6state_preGT_score_lock_v1")

MANIFEST_REL = Path(
    r"outputs\Q1_R10L3A_polypgen_source_a1_prediction_lock_fix2_v1"
    r"\frozen_polypgen_rgb_manifest_no_gt.csv"
)

S06_REL = Path(r"outputs\S06_C_polypgen_dual_backbone_no_label_lock_v1_fix1")

RNG_SEED = 20260908
SAMPLE_PIXELS = 200
EPS = 1e-5
EXPECTED_MANIFEST_ROWS = 1532
EXPECTED_ASSET_ROWS = 1537
EXPECTED_STATES = 6
FULL_PANEL_STATES = 9

UNITS = [
    ("DeepLabV3-R50", 20260817, Path(r"units\deeplab_seed20260817\locked_logits.npz")),
    ("DeepLabV3-R50", 20260818, Path(r"units\deeplab_seed20260818\locked_logits.npz")),
    ("DeepLabV3-R50", 20260819, Path(r"units\deeplab_seed20260819\locked_logits.npz")),
    ("PraNet", 20260817, Path(r"units\pranet_seed20260817\locked_logits.npz")),
    ("PraNet", 20260818, Path(r"units\pranet_seed20260818\locked_logits.npz")),
    ("PraNet", 20260819, Path(r"units\pranet_seed20260819\locked_logits.npz")),
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def norm_text(x) -> str:
    s = str(x).strip().replace("\\", "/").lower()
    while "//" in s:
        s = s.replace("//", "/")
    return s


def basename_text(x) -> str:
    return Path(norm_text(x)).name.lower()


def assert_no_gt_columns(df: pd.DataFrame) -> None:
    bad_exact = {
        "gt", "ground_truth", "groundtruth", "mask_gt", "gt_mask",
        "dice", "delta_dice", "harmful", "harm", "beneficial", "neutral",
    }
    cols = {str(c).strip().lower() for c in df.columns}
    found = sorted(cols & bad_exact)
    if found:
        raise RuntimeError(
            "BLOCKED_GT_OR_OUTCOME_COLUMN_IN_PREGT_MANIFEST: " + ",".join(found)
        )


def exact_unique_map(target_keys: List, asset_keys: List) -> Tuple[bool, List[int], str]:
    index = {}
    duplicate_asset_keys = set()
    for i, k in enumerate(asset_keys):
        if k in index:
            duplicate_asset_keys.add(k)
        else:
            index[k] = i

    missing = []
    mapped = []
    for k in target_keys:
        if k in duplicate_asset_keys or k not in index:
            missing.append(k)
        else:
            mapped.append(index[k])

    if missing:
        return False, [], f"missing_or_ambiguous={len(missing)}"

    if len(mapped) != len(set(mapped)):
        return False, [], "mapped_asset_rows_not_unique"

    return True, mapped, "PASS"


def resolve_alignment(manifest: pd.DataFrame, z) -> Dict[str, object]:
    required_npz = {
        "source_logits", "external_ids", "centers", "image_members",
        "architecture", "seed",
    }
    missing = sorted(required_npz - set(z.files))
    if missing:
        raise RuntimeError(f"NPZ_MISSING_REQUIRED_MEMBERS: {missing}")

    ext = [norm_text(x) for x in z["external_ids"]]
    centers = [norm_text(x) for x in z["centers"]]
    members = [norm_text(x) for x in z["image_members"]]

    if not (len(ext) == len(centers) == len(members) == EXPECTED_ASSET_ROWS):
        raise RuntimeError(
            f"Unexpected S06 asset metadata rows: "
            f"external_ids={len(ext)}, centers={len(centers)}, image_members={len(members)}"
        )

    strategies = []

    if "original_polypgen_sample_id" in manifest.columns:
        strategies += [
            (
                "PAIR_CENTER_EXTERNAL_ID_TO_CENTER_ORIGINAL_ID",
                [
                    (norm_text(c), norm_text(x))
                    for c, x in zip(manifest["center"], manifest["original_polypgen_sample_id"])
                ],
                list(zip(centers, ext)),
            ),
            (
                "EXTERNAL_ID_TO_ORIGINAL_ID",
                [norm_text(x) for x in manifest["original_polypgen_sample_id"]],
                ext,
            ),
        ]

    if "sample_id" in manifest.columns:
        strategies += [
            (
                "PAIR_CENTER_EXTERNAL_ID_TO_CENTER_SAMPLE_ID",
                [
                    (norm_text(c), norm_text(x))
                    for c, x in zip(manifest["center"], manifest["sample_id"])
                ],
                list(zip(centers, ext)),
            ),
            (
                "EXTERNAL_ID_TO_SAMPLE_ID",
                [norm_text(x) for x in manifest["sample_id"]],
                ext,
            ),
        ]

    if "image_path" in manifest.columns:
        strategies += [
            (
                "IMAGE_MEMBER_BASENAME_TO_IMAGE_PATH_BASENAME",
                [basename_text(x) for x in manifest["image_path"]],
                [basename_text(x) for x in members],
            ),
            (
                "PAIR_CENTER_IMAGE_BASENAME",
                [
                    (norm_text(c), basename_text(x))
                    for c, x in zip(manifest["center"], manifest["image_path"])
                ],
                list(zip(centers, [basename_text(x) for x in members])),
            ),
        ]

    diagnostics = []
    passing = []
    for name, target_keys, asset_keys in strategies:
        ok, mapped, reason = exact_unique_map(target_keys, asset_keys)
        diagnostics.append({
            "strategy": name,
            "ok": ok,
            "reason": reason,
            "mapped_rows": len(mapped) if ok else 0,
        })
        if ok:
            passing.append((name, mapped))

    if not passing:
        raise RuntimeError(
            "BLOCKED_NO_EXACT_1532_TO_1537_ALIGNMENT\n" +
            json.dumps(diagnostics, indent=2, ensure_ascii=False)
        )

    # Fixed strategy priority = order defined above.
    selected_name, selected = passing[0]

    if len(selected) != EXPECTED_MANIFEST_ROWS:
        raise RuntimeError(
            f"Alignment mapped {len(selected)} rows, expected {EXPECTED_MANIFEST_ROWS}"
        )

    extras = sorted(set(range(EXPECTED_ASSET_ROWS)) - set(selected))

    return {
        "strategy": selected_name,
        "row_indices": selected,
        "extra_asset_rows": extras,
        "diagnostics": diagnostics,
    }


def mapping_sha256(manifest: pd.DataFrame, row_indices: List[int]) -> str:
    h = hashlib.sha256()
    for i, src_i in enumerate(row_indices):
        sample_id = (
            str(manifest.iloc[i]["sample_id"])
            if "sample_id" in manifest.columns
            else str(i)
        )
        h.update(f"{i}\t{sample_id}\t{src_i}\n".encode("utf-8"))
    return h.hexdigest()


def ccd_from_binary_logits(
    logits_2d: np.ndarray,
    generator: torch.Generator,
) -> float:
    x = torch.from_numpy(np.asarray(logits_2d, dtype=np.float32))
    p = torch.sigmoid(x).reshape(-1)

    n = int(p.numel())
    k = min(SAMPLE_PIXELS, n)
    if k <= 0:
        raise RuntimeError("EMPTY_LOGIT_MAP")

    # Official-style random pixel sampling.
    idx = torch.randperm(n, generator=generator)[:k]
    p = p[idx]

    feats = torch.stack((1.0 - p, p), dim=1)  # [k,2]
    feats = F.normalize(feats, p=2.0, dim=1)
    gram = feats.t().matmul(feats)             # [2,2]
    q = torch.softmax(gram, dim=1)
    entropy = -(q * torch.log(q + EPS)).sum(dim=1).mean()
    return float(entropy.item())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = ap.parse_args()

    root = args.root
    if not root.exists():
        raise FileNotFoundError(root)

    out_dir = root / OUT_REL
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = root / MANIFEST_REL
    s06_dir = root / S06_REL

    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    if not s06_dir.exists():
        raise FileNotFoundError(s06_dir)

    print("===== R17A POLYPGEN CCD 6-STATE PRE-GT SCORE LOCK =====")
    print("Version:", VERSION)
    print("Root:", root)
    print("Training: NO")
    print("TTA: NO")
    print("Target GT loading: NO")
    print("HARM/outcome loading: NO")
    print("Threshold tuning: NO")
    print(f"CCD RNG seed: {RNG_SEED}")
    print(f"CCD sampled pixels: min({SAMPLE_PIXELS}, N)")
    print("Panel status: PARTIAL 6/9 (SegFormer deliberately not approximated)")
    print()

    manifest = pd.read_csv(manifest_path)
    assert_no_gt_columns(manifest)

    if len(manifest) != EXPECTED_MANIFEST_ROWS:
        raise RuntimeError(
            f"Frozen PolypGen manifest rows={len(manifest)}, "
            f"expected={EXPECTED_MANIFEST_ROWS}"
        )

    required_cols = {
        "sample_id", "original_polypgen_sample_id",
        "center", "image_path", "image_raw_sha256",
    }
    missing_cols = sorted(required_cols - set(manifest.columns))
    if missing_cols:
        raise RuntimeError(f"Manifest missing required no-GT columns: {missing_cols}")

    if manifest["sample_id"].astype(str).duplicated().any():
        raise RuntimeError("Frozen manifest sample_id is not unique.")

    manifest_sha = sha256_file(manifest_path)

    # One global RNG with deterministic fixed state/sample traversal.
    generator = torch.Generator(device="cpu")
    generator.manual_seed(RNG_SEED)

    score_rows = []
    asset_locks = []
    common_map_sha = None
    common_extra_rows = None

    iterator = UNITS
    if tqdm is not None:
        iterator = tqdm(UNITS, desc="PolypGen CCD states", unit="state", dynamic_ncols=True)

    for family, seed, rel in iterator:
        asset_path = s06_dir / rel
        if not asset_path.exists():
            raise FileNotFoundError(asset_path)

        asset_sha = sha256_file(asset_path)

        with np.load(asset_path, allow_pickle=False) as z:
            logits = z["source_logits"]
            if list(logits.shape) != [EXPECTED_ASSET_ROWS, 352, 352]:
                raise RuntimeError(
                    f"{asset_path}: source_logits shape={logits.shape}, "
                    f"expected=({EXPECTED_ASSET_ROWS},352,352)"
                )

            arch = str(np.asarray(z["architecture"]).reshape(-1)[0])
            npz_seed = int(np.asarray(z["seed"]).reshape(-1)[0])

            if npz_seed != seed:
                raise RuntimeError(
                    f"{asset_path}: embedded seed={npz_seed}, expected={seed}"
                )

            align = resolve_alignment(manifest, z)
            row_indices = list(align["row_indices"])
            map_sha = mapping_sha256(manifest, row_indices)

            if common_map_sha is None:
                common_map_sha = map_sha
                common_extra_rows = list(align["extra_asset_rows"])
            else:
                if map_sha != common_map_sha:
                    raise RuntimeError(
                        f"State-specific sample mapping differs for {family} seed={seed}. "
                        "Do not compute a pooled baseline until alignment is identical."
                    )
                if list(align["extra_asset_rows"]) != common_extra_rows:
                    raise RuntimeError(
                        f"Extra 1537->1532 rows differ for {family} seed={seed}."
                    )

            pbar = range(EXPECTED_MANIFEST_ROWS)
            if tqdm is not None:
                pbar = tqdm(
                    pbar,
                    desc=f"CCD {family} {seed}",
                    unit="image",
                    dynamic_ncols=True,
                    leave=False,
                )

            for m_i in pbar:
                src_i = int(row_indices[m_i])
                ccd = ccd_from_binary_logits(logits[src_i], generator)

                mr = manifest.iloc[m_i]
                score_rows.append({
                    "row_index": len(score_rows),
                    "sample_id": str(mr["sample_id"]),
                    "original_polypgen_sample_id": str(mr["original_polypgen_sample_id"]),
                    "center": str(mr["center"]),
                    "image_path": str(mr["image_path"]),
                    "image_raw_sha256": str(mr["image_raw_sha256"]),
                    "model_family": family,
                    "model_state_id": f"{family}::{seed}",
                    "training_seed": seed,
                    "source_asset_relpath": str(asset_path.relative_to(root)).replace("\\", "/"),
                    "source_asset_sha256": asset_sha,
                    "source_asset_row_index": src_i,
                    "ccd_risk": ccd,
                })

            asset_locks.append({
                "model_family": family,
                "training_seed": seed,
                "embedded_architecture": arch,
                "source_asset_relpath": str(asset_path.relative_to(root)).replace("\\", "/"),
                "source_asset_sha256": asset_sha,
                "source_logits_shape": [EXPECTED_ASSET_ROWS, 352, 352],
                "alignment_strategy": align["strategy"],
                "alignment_mapping_sha256": map_sha,
                "excluded_extra_asset_rows": list(align["extra_asset_rows"]),
                "excluded_extra_count": len(align["extra_asset_rows"]),
                "target_manifest_rows": EXPECTED_MANIFEST_ROWS,
                "target_gt_used": False,
            })

    expected_score_rows = EXPECTED_MANIFEST_ROWS * EXPECTED_STATES
    if len(score_rows) != expected_score_rows:
        raise RuntimeError(
            f"Score rows={len(score_rows)}, expected={expected_score_rows}"
        )

    score_df = pd.DataFrame(score_rows)

    # Structural checks only. No GT/outcome columns are allowed.
    assert_no_gt_columns(score_df)
    if score_df["ccd_risk"].isna().any():
        raise RuntimeError("NaN CCD scores.")
    if not np.isfinite(score_df["ccd_risk"].to_numpy(dtype=float)).all():
        raise RuntimeError("Non-finite CCD scores.")
    if score_df.duplicated(["sample_id", "model_state_id"]).any():
        raise RuntimeError("Duplicate sample/model-state rows.")

    per_state = (
        score_df.groupby(["model_family", "training_seed"], as_index=False)
        .agg(
            rows=("ccd_risk", "size"),
            ccd_mean=("ccd_risk", "mean"),
            ccd_std=("ccd_risk", "std"),
            ccd_min=("ccd_risk", "min"),
            ccd_max=("ccd_risk", "max"),
        )
    )

    score_path = out_dir / "R17A_POLYPGEN_CCD_6STATE_PREGT_SCORE_LOCK.csv"
    score_df.to_csv(score_path, index=False)
    score_sha = sha256_file(score_path)

    state_summary_path = out_dir / "R17A_POLYPGEN_CCD_6STATE_SUMMARY.csv"
    per_state.to_csv(state_summary_path, index=False)

    asset_lock_path = out_dir / "R17A_POLYPGEN_CCD_INPUT_ASSET_LOCK.json"
    asset_lock_path.write_text(
        json.dumps(
            {
                "version": VERSION,
                "manifest": str(manifest_path.relative_to(root)).replace("\\", "/"),
                "manifest_sha256": manifest_sha,
                "manifest_rows": EXPECTED_MANIFEST_ROWS,
                "asset_rows_each": EXPECTED_ASSET_ROWS,
                "common_alignment_mapping_sha256": common_map_sha,
                "excluded_extra_asset_rows": common_extra_rows,
                "states": asset_locks,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    lock = {
        "status": "PASS",
        "decision": "PASS_R17A_POLYPGEN_CCD_6STATE_PREGT_SCORE_LOCK_PARTIAL_PANEL",
        "version": VERSION,
        "dataset": "PolypGen",
        "target_gt_used": False,
        "harm_outcome_loaded": False,
        "tta_run": False,
        "segmentation_training": False,
        "ccd_definition": {
            "source": "SicTTA official-code-faithful",
            "binary_probability": "[1-sigmoid(logit), sigmoid(logit)]",
            "sample_pixels": SAMPLE_PIXELS,
            "sampling": "torch.randperm without replacement",
            "rng_seed": RNG_SEED,
            "feature_l2_normalization": True,
            "gram": "P^T P",
            "softmax_dim": 1,
            "entropy_eps": EPS,
            "risk_direction": "higher CCD = higher HARM risk",
        },
        "panel": {
            "locked_states": EXPECTED_STATES,
            "full_paper_states": FULL_PANEL_STATES,
            "missing_family": "SegFormer-B0",
            "status": "PARTIAL_6_OF_9",
        },
        "rows": len(score_df),
        "unique_samples": int(score_df["sample_id"].nunique()),
        "score_table": score_path.name,
        "score_table_sha256": score_sha,
        "state_summary": state_summary_path.name,
        "state_summary_sha256": sha256_file(state_summary_path),
        "input_asset_lock": asset_lock_path.name,
        "input_asset_lock_sha256": sha256_file(asset_lock_path),
        "next": "RUN_GT_FREE_SEGFOMER_POLYPGEN_CCD_FOR_REMAINING_3_STATES",
    }

    lock_path = out_dir / "R17A_POLYPGEN_CCD_6STATE_PREGT_LOCK.json"
    lock_path.write_text(
        json.dumps(lock, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    report_lines = [
        "===== R17A POLYPGEN CCD 6-STATE PRE-GT SCORE LOCK =====",
        f"Version: {VERSION}",
        f"Dataset: PolypGen",
        f"Frozen manifest rows: {EXPECTED_MANIFEST_ROWS}",
        f"S06 source-logit rows per state: {EXPECTED_ASSET_ROWS}",
        f"States locked: {EXPECTED_STATES}/{FULL_PANEL_STATES}",
        f"Score rows: {len(score_df)}",
        f"Unique samples: {score_df['sample_id'].nunique()}",
        f"CCD RNG seed: {RNG_SEED}",
        f"Common 1532<-1537 alignment SHA256: {common_map_sha}",
        f"Excluded extra asset rows: {common_extra_rows}",
        f"Score table SHA256: {score_sha}",
        "",
        "Information boundary:",
        "  Training: NO",
        "  TTA: NO",
        "  Target GT loading: NO",
        "  HARM/outcome loading: NO",
        "  Threshold tuning: NO",
        "",
        "IMPORTANT:",
        "- This is a PRE-GT score lock only; no AUROC/AUPRC is computed here.",
        "- DeepLabV3-R50 and PraNet are complete (3 seeds each).",
        "- SegFormer-B0 is intentionally absent because no faithful frozen soft-logit asset exists.",
        "- Do not approximate SegFormer CCD from hard masks.",
        "",
        "GATE=PASS_R17A_POLYPGEN_CCD_6STATE_PREGT_SCORE_LOCK_PARTIAL_PANEL",
        "NEXT=RUN_GT_FREE_SEGFOMER_POLYPGEN_CCD_FOR_REMAINING_3_STATES",
    ]

    report_path = out_dir / "R17A_POLYPGEN_CCD_6STATE_PREGT_REPORT.txt"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print()
    for line in report_lines[-12:]:
        print(line)
    print("Report:", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
