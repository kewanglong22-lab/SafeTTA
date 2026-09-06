#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
S06-D: first PolypGen GT reveal after the frozen S06-C global no-label lock.

No model loading, prediction recomputation, gate fitting, decision recomputation,
or threshold tuning occurs in this script.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import shutil
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from tqdm import tqdm

VERSION = "2026-08-18-S06-D-v1-fix1"
BUILD = "S06_D_FIX1_NULL_GT_RULE_FROZEN_AFTER_MASK_ONLY_AUDIT"

ROOT = Path(r"F:\MEDSEG_SAFETTA")
PROTOCOL = ROOT / "docs" / "S06_D_polypgen_external_gt_reveal_evaluation_protocol_v1_fix1.md"
EXPECTED_PROTOCOL_SHA256 = "58e78195e9639d1615837c587a908a16575208fc9f4f8c9692b7f7c1a24622cb"

ARCHIVE = ROOT / "data" / "downloads" / "polypgen" / "polypgen2021_multicenterdata_v3.zip"
EXPECTED_ARCHIVE_SHA256 = "a22f956a9f0fc3f941927108a3de8217fc61f1944c7eda598536dbabc27dbff3"

S06B_DIR = ROOT / "outputs" / "S06_B_polypgen_integrity_overlap_audit_v1_fix3"
ELIGIBILITY = S06B_DIR / "S06_polypgen_external_eligibility_manifest.csv"
EXPECTED_ELIGIBILITY_SHA256 = "b157d28634f8f53381ea6ee50db4142fbd06d94dc3f23fd858f2f62f7b71da06"

S06C_DIR = ROOT / "outputs" / "S06_C_polypgen_dual_backbone_no_label_lock_v1_fix1"
S06C_GLOBAL_LOCK = S06C_DIR / "POLYPGEN_DUAL_BACKBONE_NO_LABEL_GLOBAL_LOCK.json"
EXPECTED_S06C_GLOBAL_LOCK_SHA256 = "6459f0addcbddf2449fa5f7b03096cbff8432ffc19c21b0c4116a3d6f7354db5"
EXPECTED_S06C_SCRIPT_SHA256 = "4cd49d7f62dc6e9c055a06ab6eab2541f82fd7546803209513567c384773331a"
EXPECTED_S06C_PROTOCOL_SHA256 = "b1ae696b0a8eaf4b668ec0f4d26ebc3e40f05cdbce20a94ade92bf6a51556367"

S06D0_DIR = ROOT / "outputs" / "S06_D0_polypgen_mask_gt_integrity_audit_v1"
S06D0_LOCK = S06D0_DIR / "S06_D0_MASK_GT_INTEGRITY_LOCK.json"
EXPECTED_S06D0_LOCK_SHA256 = "fd2aa9db6cdc3f675cf88cbb2c29493d7e2a863cf20f266d136e751774563bea"
EXPECTED_S06D0_DECISION = "S06_D0_MASK_GT_INTEGRITY_AUDIT_LOCKED_READY_FOR_EMPTY_GT_RULE_FREEZE"
EXPECTED_NULL_MASKS = 125
EXPECTED_NONEMPTY_MASKS = 1412

OUTPUT_DIR = ROOT / "outputs" / "S06_D_polypgen_external_gt_reveal_evaluation_v1_fix1"

ARCHITECTURES = ("pranet", "deeplab")
SEEDS = (20260817, 20260818, 20260819)
IMAGE_SIZE = 352
EXPECTED_IMAGES = 1537
EXPECTED_ROWS_PER_ARCH = EXPECTED_IMAGES * len(SEEDS)
EXPECTED_TOTAL_ROWS = EXPECTED_ROWS_PER_ARCH * len(ARCHITECTURES)
EXPECTED_CENTERS = {"C1": 256, "C2": 301, "C3": 457, "C4": 227, "C5": 208, "C6": 88}
HARM_THRESHOLD = -0.02
BENEFIT_THRESHOLD = +0.02
PASS_HARM_PREVENTION = 0.50
PASS_ACCEPTANCE_MIN = 0.10
PASS_ACCEPTANCE_MAX = 0.90


def file_sha256(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def bytes_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_csv(path: Path):
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        return list(r), r.fieldnames or []


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def validate_sha(path: Path, expected: str, label: str) -> str:
    if not path.exists():
        raise FileNotFoundError(path)
    actual = file_sha256(path)
    if actual.lower() != expected.lower():
        raise RuntimeError(f"{label} SHA mismatch.\nExpected: {expected}\nActual: {actual}")
    return actual


def validate_inputs():
    validate_sha(PROTOCOL, EXPECTED_PROTOCOL_SHA256, "S06-D protocol")
    validate_sha(ARCHIVE, EXPECTED_ARCHIVE_SHA256, "PolypGen archive")
    eligibility_sha = validate_sha(ELIGIBILITY, EXPECTED_ELIGIBILITY_SHA256, "S06-B eligibility")
    global_sha = validate_sha(S06C_GLOBAL_LOCK, EXPECTED_S06C_GLOBAL_LOCK_SHA256, "S06-C global lock")
    d0_sha = validate_sha(S06D0_LOCK, EXPECTED_S06D0_LOCK_SHA256, "S06-D0 mask-only audit lock")

    d0_lock = json.loads(S06D0_LOCK.read_text(encoding="utf-8"))
    if d0_lock.get("decision") != EXPECTED_S06D0_DECISION:
        raise RuntimeError(
            f"S06-D0 lock does not authorize empty-GT rule freeze: {d0_lock.get('decision')}"
        )
    if bool(d0_lock.get("prediction_arrays_read", True)):
        raise RuntimeError("S06-D0 reports prediction-array access.")
    if bool(d0_lock.get("decision_csvs_read", True)):
        raise RuntimeError("S06-D0 reports decision-CSV access.")
    if bool(d0_lock.get("model_metrics_computed", True)):
        raise RuntimeError("S06-D0 reports model metrics before rule freeze.")

    lock = json.loads(S06C_GLOBAL_LOCK.read_text(encoding="utf-8"))
    required = {
        "script_sha256", "protocol_sha256", "archive_sha256", "eligibility_manifest_sha256",
        "external_images", "center_counts", "architectures", "seeds", "architecture_seed_units",
        "total_architecture_seed_image_observations", "units",
        "polypgen_mask_bytes_opened_before_global_lock", "target_metrics_computed_before_global_lock",
        "model_refit", "threshold_recalibrated", "feature_selection_performed", "decision",
    }
    missing = sorted(required - set(lock))
    if missing:
        raise RuntimeError(f"S06-C global lock missing fields: {missing}")
    if lock["script_sha256"] != EXPECTED_S06C_SCRIPT_SHA256:
        raise RuntimeError("S06-C script provenance mismatch.")
    if lock["protocol_sha256"] != EXPECTED_S06C_PROTOCOL_SHA256:
        raise RuntimeError("S06-C protocol provenance mismatch.")
    if lock["archive_sha256"] != EXPECTED_ARCHIVE_SHA256:
        raise RuntimeError("S06-C archive provenance mismatch.")
    if lock["eligibility_manifest_sha256"] != EXPECTED_ELIGIBILITY_SHA256:
        raise RuntimeError("S06-C eligibility provenance mismatch.")
    if int(lock["external_images"]) != EXPECTED_IMAGES:
        raise RuntimeError("S06-C image count mismatch.")
    if list(lock["architectures"]) != list(ARCHITECTURES):
        raise RuntimeError("S06-C architecture list mismatch.")
    if [int(x) for x in lock["seeds"]] != list(SEEDS):
        raise RuntimeError("S06-C seed list mismatch.")
    if int(lock["architecture_seed_units"]) != 6:
        raise RuntimeError("S06-C unit count mismatch.")
    if int(lock["total_architecture_seed_image_observations"]) != EXPECTED_TOTAL_ROWS:
        raise RuntimeError("S06-C observation count mismatch.")
    centers = {str(k): int(v) for k, v in lock["center_counts"].items()}
    if centers != EXPECTED_CENTERS:
        raise RuntimeError(f"S06-C center counts mismatch: {centers}")
    for key in (
        "polypgen_mask_bytes_opened_before_global_lock",
        "target_metrics_computed_before_global_lock",
        "model_refit", "threshold_recalibrated", "feature_selection_performed",
    ):
        if bool(lock[key]):
            raise RuntimeError(f"S06-C integrity flag unexpectedly true: {key}")
    if lock["decision"] != "S06_C_DUAL_BACKBONE_NO_LABEL_GLOBAL_LOCK_PASS_READY_FOR_S06D_GT_REVEAL":
        raise RuntimeError(f"S06-C does not authorize GT reveal: {lock['decision']}")

    rows, fields = read_csv(ELIGIBILITY)
    needed = {"external_id", "center", "image_member", "mask_member", "eligibility", "eligibility_uses_mask_content"}
    missing = sorted(needed - set(fields))
    if missing:
        raise RuntimeError(f"Eligibility fields missing: {missing}")
    if len(rows) != EXPECTED_IMAGES:
        raise RuntimeError(f"Eligibility row count mismatch: {len(rows)}")
    if any(r["eligibility"] != "ELIGIBLE_PENDING_S06C" for r in rows):
        raise RuntimeError("Eligibility contains non-eligible rows.")
    if any(str(r["eligibility_uses_mask_content"]).upper() != "NO" for r in rows):
        raise RuntimeError("Eligibility used mask content.")
    ids = [r["external_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate external_id in eligibility.")
    counts = Counter(r["center"] for r in rows)
    if dict(sorted(counts.items())) != EXPECTED_CENTERS:
        raise RuntimeError(f"Eligibility center mismatch: {dict(counts)}")
    rows = sorted(rows, key=lambda r: (r["center"], r["external_id"]))

    expected_unit_keys = {f"{arch}::{seed}" for arch in ARCHITECTURES for seed in SEEDS}
    if set(lock["units"]) != expected_unit_keys:
        raise RuntimeError(f"Global unit keys mismatch: {sorted(lock['units'])}")
    return rows, lock, global_sha, eligibility_sha, d0_sha


def unit_paths(architecture: str, seed: int):
    d = S06C_DIR / "units" / f"{architecture}_seed{seed}"
    return {
        "dir": d,
        "lock": d / "UNIT_NO_LABEL_LOCK.json",
        "npz": d / "locked_logits.npz",
        "csv": d / "no_label_features_and_gate_decisions.csv",
    }


def validate_unit(global_lock, architecture: str, seed: int, expected_ids):
    key = f"{architecture}::{seed}"
    ref = global_lock["units"][key]
    paths = unit_paths(architecture, seed)
    for p in paths.values():
        if not p.exists():
            raise FileNotFoundError(p)
    lock_sha = file_sha256(paths["lock"])
    if lock_sha != ref["unit_lock_sha256"]:
        raise RuntimeError(f"Unit lock SHA mismatch: {key}")
    lock = json.loads(paths["lock"].read_text(encoding="utf-8"))
    if lock["architecture"] != architecture or int(lock["seed"]) != seed:
        raise RuntimeError(f"Unit identity mismatch: {key}")
    if int(lock["rows"]) != EXPECTED_IMAGES:
        raise RuntimeError(f"Unit row count mismatch: {key}")
    for flag in ("polypgen_mask_bytes_opened", "target_metrics_computed", "model_refit", "threshold_recalibrated"):
        if bool(lock[flag]):
            raise RuntimeError(f"Unit integrity flag true {key}: {flag}")
    npz_sha = file_sha256(paths["npz"])
    csv_sha = file_sha256(paths["csv"])
    if npz_sha != ref["npz_sha256"] or npz_sha != lock["npz_sha256"]:
        raise RuntimeError(f"Unit NPZ SHA mismatch: {key}")
    if csv_sha != ref["csv_sha256"] or csv_sha != lock["csv_sha256"]:
        raise RuntimeError(f"Unit CSV SHA mismatch: {key}")

    decision_rows, fields = read_csv(paths["csv"])
    req = {"architecture", "seed", "external_id", "center", "image_member", "p_harm", "tau", "tent_accepted", "decision"}
    missing = sorted(req - set(fields))
    if missing:
        raise RuntimeError(f"Decision fields missing {key}: {missing}")
    if len(decision_rows) != EXPECTED_IMAGES:
        raise RuntimeError(f"Decision row count mismatch: {key}")
    if [r["external_id"] for r in decision_rows] != expected_ids:
        raise RuntimeError(f"Decision row order mismatch: {key}")
    for r in decision_rows:
        p_harm, tau = float(r["p_harm"]), float(r["tau"])
        accepted = int(r["tent_accepted"])
        if accepted != int(p_harm < tau):
            raise RuntimeError(f"Decision arithmetic mismatch: {key} {r['external_id']}")
        text = "ACCEPT_TENT" if accepted else "REJECT_TENT_USE_SOURCE"
        if r["decision"] != text:
            raise RuntimeError(f"Decision text mismatch: {key} {r['external_id']}")
    return paths, lock, decision_rows


def load_mask_gt(zf: zipfile.ZipFile, mask_member: str, external_id: str):
    # Authorized target-mask read after S06-C global lock and S06-D0 audit.
    with zf.open(mask_member, "r") as f:
        payload = f.read()

    raw_sha = bytes_sha256(payload)
    with Image.open(io.BytesIO(payload)) as im:
        gray = np.asarray(im.convert("L"), dtype=np.float32)

    if gray.ndim != 2 or gray.size == 0:
        raise RuntimeError(
            f"Invalid target mask decode: {external_id} shape={gray.shape}"
        )

    maxv = float(gray.max())
    if not math.isfinite(maxv) or maxv < 0:
        raise RuntimeError(f"Invalid target mask range: {external_id}")

    if maxv == 0:
        mask_state = "NULL_ALL_ZERO"
        gt = np.zeros(gray.shape, dtype=np.uint8)
    else:
        mask_state = "NONEMPTY"
        gt = (gray > 0.5 * maxv).astype(np.uint8)
        if int(gt.sum()) <= 0:
            raise RuntimeError(
                f"NONEMPTY target mask thresholded empty: {external_id}"
            )

    fg = int(gt.sum())
    header = f"BINARY|{gt.shape[1]}|{gt.shape[0]}|".encode("ascii")
    binary_sha = hashlib.sha256(header + gt.tobytes()).hexdigest()

    meta = {
        "external_id": external_id,
        "mask_member": mask_member,
        "mask_raw_sha256": raw_sha,
        "mask_binary_sha256": binary_sha,
        "mask_width": int(gt.shape[1]),
        "mask_height": int(gt.shape[0]),
        "mask_foreground_pixels": fg,
        "mask_state": mask_state,
        "mask_max_gray": maxv,
    }
    return gt, meta


def resize_logit(logit, shape_hw):
    t = torch.from_numpy(np.asarray(logit, dtype=np.float32))[None, None]
    return F.interpolate(t, size=tuple(int(x) for x in shape_hw), mode="bilinear", align_corners=False)[0, 0].numpy()


def metrics_from_logit(logit, gt):
    pred = resize_logit(logit, gt.shape) >= 0.0
    target = gt.astype(bool)

    pred_n = int(pred.sum())
    target_n = int(target.sum())

    # Frozen S06-D FIX1 empty-GT convention.
    if target_n == 0:
        if pred_n == 0:
            return 1.0, 1.0
        return 0.0, 0.0

    inter = int(np.logical_and(pred, target).sum())
    union = int(np.logical_or(pred, target).sum())
    dice = float(
        (2.0 * inter + 1e-7)
        / (pred_n + target_n + 1e-7)
    )
    iou = float(
        (inter + 1e-7)
        / (union + 1e-7)
    )
    return dice, iou


def evaluate_units(eligibility_rows, global_lock, global_lock_sha, zf):
    expected_ids = [r["external_id"] for r in eligibility_rows]
    row_by_id = {r["external_id"]: r for r in eligibility_rows}
    per_rows = []
    mask_meta_by_id = {}

    for architecture in ARCHITECTURES:
        for seed in SEEDS:
            paths, _, decision_rows = validate_unit(global_lock, architecture, seed, expected_ids)
            decision_by_id = {r["external_id"]: r for r in decision_rows}
            with np.load(paths["npz"], allow_pickle=False) as data:
                ids = data["external_ids"].tolist()
                centers = data["centers"].tolist()
                image_members = data["image_members"].tolist()
                source_logits = data["source_logits"]
                tent_logits = data["tent_logits"]
                if ids != expected_ids:
                    raise RuntimeError(f"NPZ order mismatch: {architecture} seed={seed}")
                if source_logits.shape != (EXPECTED_IMAGES, IMAGE_SIZE, IMAGE_SIZE):
                    raise RuntimeError(f"Source logit shape mismatch: {architecture} seed={seed}")
                if tent_logits.shape != source_logits.shape:
                    raise RuntimeError(f"TENT logit shape mismatch: {architecture} seed={seed}")

                pbar = tqdm(range(EXPECTED_IMAGES), desc=f"S06-D {architecture} seed {seed}", unit="img", dynamic_ncols=True)
                for i in pbar:
                    external_id = ids[i]
                    row = row_by_id[external_id]
                    d = decision_by_id[external_id]
                    if centers[i] != row["center"] or image_members[i] != row["image_member"]:
                        raise RuntimeError(f"Frozen identity mismatch: {external_id}")

                    gt, meta = load_mask_gt(zf, row["mask_member"], external_id)
                    if external_id not in mask_meta_by_id:
                        mask_meta_by_id[external_id] = meta
                    elif mask_meta_by_id[external_id] != meta:
                        raise RuntimeError(f"GT mask changed during evaluation: {external_id}")

                    source_dice, source_iou = metrics_from_logit(source_logits[i], gt)
                    tent_dice, tent_iou = metrics_from_logit(tent_logits[i], gt)
                    accepted = int(d["tent_accepted"]) == 1
                    safe_dice, safe_iou = ((tent_dice, tent_iou) if accepted else (source_dice, source_iou))
                    oracle_dice = max(source_dice, tent_dice)
                    oracle_iou = max(source_iou, tent_iou)
                    tent_delta = tent_dice - source_dice
                    safe_delta = safe_dice - source_dice
                    tent_harmful = int(tent_delta <= HARM_THRESHOLD)
                    tent_beneficial = int(tent_delta >= BENEFIT_THRESHOLD)
                    safe_harmful = int(safe_delta <= HARM_THRESHOLD)
                    safe_beneficial = int(safe_delta >= BENEFIT_THRESHOLD)

                    per_rows.append({
                        "architecture": architecture, "seed": seed, "external_id": external_id,
                        "center": row["center"], "image_member": row["image_member"], "mask_member": row["mask_member"],
                        "mask_state": meta["mask_state"],
                        "p_harm": d["p_harm"], "tau": d["tau"], "tent_accepted": int(accepted), "decision": d["decision"],
                        "source_dice": source_dice, "source_iou": source_iou,
                        "tent_dice": tent_dice, "tent_iou": tent_iou,
                        "safettta_dice": safe_dice, "safettta_iou": safe_iou,
                        "oracle_dice": oracle_dice, "oracle_iou": oracle_iou,
                        "tent_delta_dice": tent_delta, "safettta_delta_dice": safe_delta,
                        "tent_harmful": tent_harmful, "tent_beneficial": tent_beneficial,
                        "safettta_harmful": safe_harmful, "safettta_beneficial": safe_beneficial,
                        "harm_prevented": int(tent_harmful and not accepted),
                        "benefit_retained": int(tent_beneficial and accepted),
                        "s06c_global_no_label_lock_sha256": global_lock_sha,
                    })
                    pbar.set_postfix(src=f"{source_dice:.3f}", tent=f"{tent_dice:.3f}", safe=f"{safe_dice:.3f}")

    if len(per_rows) != EXPECTED_TOTAL_ROWS:
        raise RuntimeError(f"Evaluation rows mismatch: {len(per_rows)}")
    if len(mask_meta_by_id) != EXPECTED_IMAGES:
        raise RuntimeError(f"Unique GT mask count mismatch: {len(mask_meta_by_id)}")
    return per_rows, [mask_meta_by_id[x] for x in expected_ids]


def summarize_group(rows, architecture, group_type, group):
    if not rows:
        raise RuntimeError(f"Empty summary group: {architecture} {group_type} {group}")
    source = np.asarray([float(r["source_dice"]) for r in rows], dtype=np.float64)
    tent = np.asarray([float(r["tent_dice"]) for r in rows], dtype=np.float64)
    safe = np.asarray([float(r["safettta_dice"]) for r in rows], dtype=np.float64)
    oracle = np.asarray([float(r["oracle_dice"]) for r in rows], dtype=np.float64)
    source_iou = np.asarray([float(r["source_iou"]) for r in rows], dtype=np.float64)
    tent_iou = np.asarray([float(r["tent_iou"]) for r in rows], dtype=np.float64)
    safe_iou = np.asarray([float(r["safettta_iou"]) for r in rows], dtype=np.float64)
    accepted = np.asarray([int(r["tent_accepted"]) for r in rows], dtype=np.int64)
    tent_harm = np.asarray([int(r["tent_harmful"]) for r in rows], dtype=np.int64)
    safe_harm = np.asarray([int(r["safettta_harmful"]) for r in rows], dtype=np.int64)
    tent_benefit = np.asarray([int(r["tent_beneficial"]) for r in rows], dtype=np.int64)
    harm_n, benefit_n = int(tent_harm.sum()), int(tent_benefit.sum())
    harm_prevention = float(np.sum((tent_harm == 1) & (accepted == 0)) / harm_n) if harm_n else float("nan")
    benefit_retention = float(np.sum((tent_benefit == 1) & (accepted == 1)) / benefit_n) if benefit_n else float("nan")
    return {
        "architecture": architecture, "group_type": group_type, "group": group, "n_rows": len(rows),
        "source_mean_dice": float(source.mean()), "tent_mean_dice": float(tent.mean()),
        "safettta_mean_dice": float(safe.mean()), "oracle_mean_dice": float(oracle.mean()),
        "source_mean_iou": float(source_iou.mean()), "tent_mean_iou": float(tent_iou.mean()),
        "safettta_mean_iou": float(safe_iou.mean()),
        "safettta_minus_source": float(safe.mean() - source.mean()),
        "safettta_minus_tent": float(safe.mean() - tent.mean()),
        "tent_harmful_fraction": float(tent_harm.mean()), "safettta_harmful_fraction": float(safe_harm.mean()),
        "tent_harmful_n": harm_n, "harm_prevention_rate": harm_prevention,
        "tent_beneficial_fraction": float(tent_benefit.mean()), "tent_beneficial_n": benefit_n,
        "benefit_retention_rate": benefit_retention, "tent_acceptance_rate": float(accepted.mean()),
        "oracle_gap": float(oracle.mean() - safe.mean()),
    }


def make_summary(per_rows):
    out = []
    for architecture in ARCHITECTURES:
        arch_rows = [r for r in per_rows if r["architecture"] == architecture]
        if len(arch_rows) != EXPECTED_ROWS_PER_ARCH:
            raise RuntimeError(f"Architecture row mismatch: {architecture}")
        out.append(summarize_group(arch_rows, architecture, "pooled", "POLYPGEN"))
        for center, n_img in EXPECTED_CENTERS.items():
            rows = [r for r in arch_rows if r["center"] == center]
            if len(rows) != n_img * len(SEEDS):
                raise RuntimeError(f"Center row mismatch: {architecture} {center}")
            out.append(summarize_group(rows, architecture, "center", center))
        for seed in SEEDS:
            rows = [r for r in arch_rows if int(r["seed"]) == seed]
            if len(rows) != EXPECTED_IMAGES:
                raise RuntimeError(f"Seed row mismatch: {architecture} {seed}")
            out.append(summarize_group(rows, architecture, "seed", str(seed)))
    return out


def evaluate_point_gate(summary_rows):
    pooled = {r["architecture"]: r for r in summary_rows if r["group_type"] == "pooled" and r["group"] == "POLYPGEN"}
    if set(pooled) != set(ARCHITECTURES):
        raise RuntimeError("Missing pooled architecture summaries.")
    results, passed_architectures = {}, []
    for architecture in ARCHITECTURES:
        r = pooled[architecture]
        a = r["safettta_mean_dice"] + 1e-12 >= r["source_mean_dice"]
        b = math.isfinite(r["harm_prevention_rate"]) and r["harm_prevention_rate"] + 1e-12 >= PASS_HARM_PREVENTION
        c = PASS_ACCEPTANCE_MIN <= r["tent_acceptance_rate"] <= PASS_ACCEPTANCE_MAX
        passed = a and b and c
        if passed:
            passed_architectures.append(architecture)
        results[architecture] = {
            "criterion_A_utility_safe_ge_source": a,
            "criterion_B_harm_prevention_ge_0_50": b,
            "criterion_C_acceptance_in_0_10_0_90": c,
            "benefit_retention_secondary": r["benefit_retention_rate"],
            "point_estimate_pass": passed,
            "decision": f"S06_D_{architecture.upper()}_EXTERNAL_POINT_CONFIRMATION_{'PASS' if passed else 'NOT_MET'}",
        }
    if len(passed_architectures) == 2:
        decision = "S06_D_STRONG_CROSS_ARCH_EXTERNAL_POINT_CONFIRMATION_PASS_READY_FOR_S06E_STATS"
    elif len(passed_architectures) == 1:
        decision = "S06_D_ARCHITECTURE_SPECIFIC_EXTERNAL_POINT_CONFIRMATION_ONLY_READY_FOR_S06E_STATS"
    else:
        decision = "S06_D_EXTERNAL_POINT_CONFIRMATION_NOT_MET_REPORT_NO_TUNING_READY_FOR_S06E_STATS"
    return {"architecture_results": results, "passed_architectures": passed_architectures, "decision": decision}


def fmt(x):
    if isinstance(x, (float, np.floating)):
        return "nan" if not math.isfinite(float(x)) else f"{float(x):.6f}"
    return str(x)


def run(args):
    eligibility_rows, global_lock, global_sha, eligibility_sha, d0_sha = validate_inputs()
    if args.output_dir.exists():
        raise FileExistsError(f"Final S06-D output exists: {args.output_dir}")
    build_dir = Path(str(args.output_dir) + "__building")
    if build_dir.exists():
        if not args.technical_rerun:
            raise FileExistsError(f"Partial S06-D output exists: {build_dir}; use --technical-rerun only after technical failure.")
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True, exist_ok=False)

    print(f"[BUILD] {VERSION} | {BUILD}")
    print(f"S06-C global lock SHA256={global_sha}")
    print(f"S06-D0 mask-only audit lock SHA256={d0_sha}")
    print(f"External images={len(eligibility_rows)} | evaluation rows={EXPECTED_TOTAL_ROWS}")
    print("FIRST POLYPGEN GT REVEAL AUTHORIZED=YES")
    print("Prediction recomputation=NO | gate recomputation=NO | threshold tuning=NO")
    print()

    with zipfile.ZipFile(ARCHIVE, "r") as zf:
        names = set(zf.namelist())
        for row in eligibility_rows:
            if row["mask_member"] not in names:
                raise RuntimeError(f"Mask member missing: {row['external_id']}")
        per_rows, mask_manifest = evaluate_units(eligibility_rows, global_lock, global_sha, zf)

    actual_null_masks = sum(
        1 for item in mask_manifest
        if item["mask_state"] == "NULL_ALL_ZERO"
    )
    actual_nonempty_masks = sum(
        1 for item in mask_manifest
        if item["mask_state"] == "NONEMPTY"
    )
    if actual_null_masks != EXPECTED_NULL_MASKS:
        raise RuntimeError(
            f"NULL_ALL_ZERO count changed from S06-D0: "
            f"expected={EXPECTED_NULL_MASKS}, actual={actual_null_masks}"
        )
    if actual_nonempty_masks != EXPECTED_NONEMPTY_MASKS:
        raise RuntimeError(
            f"NONEMPTY count changed from S06-D0: "
            f"expected={EXPECTED_NONEMPTY_MASKS}, actual={actual_nonempty_masks}"
        )

    summary_rows = make_summary(per_rows)
    gate = evaluate_point_gate(summary_rows)

    per_path = build_dir / "per_architecture_seed_image_external_evaluation.csv"
    mask_path = build_dir / "polypgen_gt_reveal_manifest.csv"
    summary_path = build_dir / "external_point_estimate_summary.csv"

    per_fields = [
        "architecture", "seed", "external_id", "center", "image_member", "mask_member", "mask_state",
        "p_harm", "tau", "tent_accepted", "decision", "source_dice", "source_iou",
        "tent_dice", "tent_iou", "safettta_dice", "safettta_iou", "oracle_dice", "oracle_iou",
        "tent_delta_dice", "safettta_delta_dice", "tent_harmful", "tent_beneficial",
        "safettta_harmful", "safettta_beneficial", "harm_prevented", "benefit_retained",
        "s06c_global_no_label_lock_sha256",
    ]
    write_csv(per_path, per_rows, per_fields)
    write_csv(mask_path, mask_manifest, [
        "external_id", "mask_member", "mask_raw_sha256", "mask_binary_sha256",
        "mask_width", "mask_height", "mask_foreground_pixels", "mask_state", "mask_max_gray",
    ])
    summary_fields = [
        "architecture", "group_type", "group", "n_rows", "source_mean_dice", "tent_mean_dice",
        "safettta_mean_dice", "oracle_mean_dice", "source_mean_iou", "tent_mean_iou",
        "safettta_mean_iou", "safettta_minus_source", "safettta_minus_tent",
        "tent_harmful_fraction", "safettta_harmful_fraction", "tent_harmful_n",
        "harm_prevention_rate", "tent_beneficial_fraction", "tent_beneficial_n",
        "benefit_retention_rate", "tent_acceptance_rate", "oracle_gap",
    ]
    write_csv(summary_path, summary_rows, summary_fields)

    result_lock = {
        "script_version": VERSION, "build": BUILD, "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "archive_sha256": EXPECTED_ARCHIVE_SHA256, "eligibility_manifest_sha256": eligibility_sha,
        "s06c_global_no_label_lock_sha256": global_sha,
        "s06d0_mask_gt_integrity_lock_sha256": d0_sha,
        "expected_null_all_zero_masks": EXPECTED_NULL_MASKS,
        "expected_nonempty_masks": EXPECTED_NONEMPTY_MASKS,
        "empty_gt_rule": {
            "gt_empty_pred_empty": {"dice": 1.0, "iou": 1.0},
            "gt_empty_pred_nonempty": {"dice": 0.0, "iou": 0.0},
            "gt_nonempty": "standard_overlap_metrics",
        },
        "external_images": EXPECTED_IMAGES,
        "architecture_seed_image_rows": EXPECTED_TOTAL_ROWS, "mask_gt_unique_images_revealed": len(mask_manifest),
        "per_image_evaluation_sha256": file_sha256(per_path),
        "gt_reveal_manifest_sha256": file_sha256(mask_path),
        "point_estimate_summary_sha256": file_sha256(summary_path),
        "prediction_recomputed_after_gt": False, "gate_recomputed_after_gt": False,
        "threshold_tuned_after_gt": False, "cohort_changed_after_gt": False,
        "harm_threshold": HARM_THRESHOLD, "benefit_threshold": BENEFIT_THRESHOLD,
        "point_estimate_gate": gate, "decision": gate["decision"],
    }
    lock_path = build_dir / "S06_D_EXTERNAL_GT_EVALUATION_LOCK.json"
    lock_path.write_text(json.dumps(result_lock, ensure_ascii=False, indent=2), encoding="utf-8")
    lock_sha = file_sha256(lock_path)

    pooled = {r["architecture"]: r for r in summary_rows if r["group_type"] == "pooled"}
    lines = [
        "===== S06-D POLYPGEN EXTERNAL GT-REVEAL EVALUATION =====",
        f"Script version: {VERSION}", f"Build: {BUILD}", "",
        "Frozen provenance:", f"  S06-C global no-label lock SHA256={global_sha}",
        f"  S06-D0 mask-only audit lock SHA256={d0_sha}",
        f"  PolypGen archive SHA256={EXPECTED_ARCHIVE_SHA256}", f"  eligible images={EXPECTED_IMAGES}",
        f"  evaluation rows={EXPECTED_TOTAL_ROWS}", "", "GT reveal:",
        f"  unique masks opened={len(mask_manifest)}",
        f"  NULL_ALL_ZERO masks={actual_null_masks}",
        f"  NONEMPTY masks={actual_nonempty_masks}",
        "  empty GT + empty prediction => Dice=1 IoU=1",
        "  empty GT + nonempty prediction => Dice=0 IoU=0",
        "  prediction recomputation after GT=NO",
        "  gate recomputation after GT=NO", "  threshold tuning after GT=NO", "  cohort change after GT=NO", "",
    ]
    for architecture in ARCHITECTURES:
        r = pooled[architecture]
        ar = gate["architecture_results"][architecture]
        lines += [
            f"[{architecture.upper()} POOLED]",
            f"  Source Dice={fmt(r['source_mean_dice'])}", f"  TENT Dice={fmt(r['tent_mean_dice'])}",
            f"  SafeTTA Dice={fmt(r['safettta_mean_dice'])}", f"  Oracle Dice={fmt(r['oracle_mean_dice'])}",
            f"  Safe-Source={fmt(r['safettta_minus_source'])}", f"  Safe-TENT={fmt(r['safettta_minus_tent'])}",
            f"  TENT harmful fraction={fmt(r['tent_harmful_fraction'])}",
            f"  Safe harmful fraction={fmt(r['safettta_harmful_fraction'])}",
            f"  Harm prevention={fmt(r['harm_prevention_rate'])}",
            f"  TENT beneficial fraction={fmt(r['tent_beneficial_fraction'])}",
            f"  Benefit retention={fmt(r['benefit_retention_rate'])}",
            f"  TENT acceptance={fmt(r['tent_acceptance_rate'])}", f"  Oracle gap={fmt(r['oracle_gap'])}",
            f"  Criterion A utility={ar['criterion_A_utility_safe_ge_source']}",
            f"  Criterion B harm prevention={ar['criterion_B_harm_prevention_ge_0_50']}",
            f"  Criterion C nondegenerate acceptance={ar['criterion_C_acceptance_in_0_10_0_90']}",
            f"  Point pass={ar['point_estimate_pass']}", "",
        ]
    lines += [f"S06-D lock SHA256={lock_sha}", "", f"Decision: {gate['decision']}", "", f"[OK] Outputs: {args.output_dir}"]
    (build_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    if file_sha256(per_path) != result_lock["per_image_evaluation_sha256"]:
        raise RuntimeError("Per-image evaluation changed before commit.")
    if file_sha256(mask_path) != result_lock["gt_reveal_manifest_sha256"]:
        raise RuntimeError("GT manifest changed before commit.")
    if file_sha256(summary_path) != result_lock["point_estimate_summary_sha256"]:
        raise RuntimeError("Point summary changed before commit.")

    build_dir.rename(args.output_dir)
    print()
    print((args.output_dir / "summary.txt").read_text(encoding="utf-8"))


def self_test():
    gt = np.zeros((16, 20), dtype=np.uint8)
    gt[4:12, 5:15] = 1
    logit = np.full((16, 20), -20.0, dtype=np.float32)
    logit[4:12, 5:15] = 20.0
    dice, iou = metrics_from_logit(logit, gt)
    assert abs(dice - 1.0) < 1e-6 and abs(iou - 1.0) < 1e-6
    empty_gt = np.zeros((9, 13), dtype=np.uint8)
    empty_pred_logit = np.full((9, 13), -20.0, dtype=np.float32)
    d0, j0 = metrics_from_logit(empty_pred_logit, empty_gt)
    assert d0 == 1.0 and j0 == 1.0

    false_positive_logit = np.full((9, 13), -20.0, dtype=np.float32)
    false_positive_logit[4, 6] = 20.0
    d1, j1 = metrics_from_logit(false_positive_logit, empty_gt)
    assert d1 == 0.0 and j1 == 0.0

    resized = resize_logit(np.asarray([[-2.0, 2.0], [-2.0, 2.0]], dtype=np.float32), (7, 11))
    assert resized.shape == (7, 11) and np.isfinite(resized).all()
    fake = [
        {"architecture": "pranet", "group_type": "pooled", "group": "POLYPGEN",
         "source_mean_dice": 0.70, "safettta_mean_dice": 0.72, "harm_prevention_rate": 0.80,
         "benefit_retention_rate": 0.40, "tent_acceptance_rate": 0.43},
        {"architecture": "deeplab", "group_type": "pooled", "group": "POLYPGEN",
         "source_mean_dice": 0.70, "safettta_mean_dice": 0.71, "harm_prevention_rate": 0.70,
         "benefit_retention_rate": 0.25, "tent_acceptance_rate": 0.51},
    ]
    gate = evaluate_point_gate(fake)
    assert gate["decision"] == "S06_D_STRONG_CROSS_ARCH_EXTERNAL_POINT_CONFIRMATION_PASS_READY_FOR_S06E_STATS"
    assert gate["architecture_results"]["deeplab"]["point_estimate_pass"] is True
    assert EXPECTED_IMAGES == 1537 and sum(EXPECTED_CENTERS.values()) == 1537
    assert EXPECTED_ROWS_PER_ARCH == 4611 and EXPECTED_TOTAL_ROWS == 9222
    assert HARM_THRESHOLD == -0.02 and BENEFIT_THRESHOLD == 0.02
    assert PASS_HARM_PREVENTION == 0.50 and PASS_ACCEPTANCE_MIN == 0.10 and PASS_ACCEPTANCE_MAX == 0.90
    print("EMPTY_GT_EMPTY_PRED_PERFECT_TEST_PASS")
    print("EMPTY_GT_FALSE_POSITIVE_ZERO_TEST_PASS")
    print("FROZEN_BINARY_METRIC_TEST_PASS")
    print("BILINEAR_NATIVE_GT_RESIZE_TEST_PASS")
    print("EXTERNAL_POINT_GATE_TEST_PASS")
    print("BENEFIT_RETENTION_SECONDARY_ONLY_TEST_PASS")
    print("FROZEN_COUNTS_TEST_PASS")
    print("SELF_TEST_PASS")


def parse_args():
    p = argparse.ArgumentParser(description="Reveal PolypGen GT only after the frozen S06-C no-label global lock and evaluate locked predictions.")
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    p.add_argument("--technical-rerun", action="store_true", help="Delete only incomplete S06-D __building output after a technical failure.")
    p.add_argument("--self-test", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
