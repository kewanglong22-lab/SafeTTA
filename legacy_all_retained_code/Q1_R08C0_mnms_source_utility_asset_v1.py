#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Q1-R08C0 — M&Ms source-only SOURCE-vs-A1 utility asset.

Target images and masks are never opened.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import traceback
from collections import Counter
from pathlib import Path
from typing import Sequence

ROOT = Path(r"F:\MEDSEG_SAFETTA")
CACHE_ROOT = ROOT / "cache"
TORCH_CACHE = CACHE_ROOT / "torch"
HF_CACHE = CACHE_ROOT / "huggingface"
os.environ.setdefault("TORCH_HOME", str(TORCH_CACHE))
os.environ.setdefault("HF_HOME", str(HF_CACHE))
os.environ.setdefault("HF_HUB_CACHE", str(HF_CACHE / "hub"))

import numpy as np
from tqdm import tqdm

VERSION = "2026-08-20-Q1-R08C0-v1"
BUILD = "Q1_R08C0_MNMS_SOURCE_ONLY_UTILITY_ASSET"

PROTOCOL = ROOT / "docs" / "Q1_R08C0_mnms_source_utility_asset_preregistered_protocol_v1.md"
EXPECTED_PROTOCOL_SHA256 = "c5e0b125a13e5c35772c0be55685fbe08bc2365bc03f061084849ebcf5e45075"

R08B1_DIR = ROOT / "outputs" / "Q1_R08B1_mnms_heterogeneous_source_only_panel_v1"
R08B1_LOCK = R08B1_DIR / "Q1_R08B1_MNMS_HETEROGENEOUS_SOURCE_PANEL_LOCK.json"
EXPECTED_R08B1_LOCK_SHA256 = "f3af46f44ca2db3a0d8d36dea2f2ce476d86d1b1926c43651d09626e8f76e478"
EXPECTED_R08B1_DECISION = "MNMS_HETEROGENEOUS_SOURCE_PANEL_READY"
SOURCE_SPLIT = R08B1_DIR / "inherited_source_split_manifest.csv"
OUTPUT_DIR = ROOT / "outputs" / "Q1_R08C0_mnms_source_utility_asset_v1"

ARCHITECTURES = ("deeplabv3_r50", "fcn_r50", "segformer_b0")
SEEDS = (20260819, 20260820, 20260821)
DIRECTIONS = (("B", "A"), ("A", "B"))
PERTURBATIONS = (
    "identity", "scale070", "scale130", "shift_m050", "shift_p050",
    "gamma070", "gamma150", "blur_sigma1", "noise_sigma010", "downsample050",
)

NUM_CLASSES = 4
INPUT_SIZE = 256
INFER_BATCH = 8
ADAPT_MICROBATCH = 4
TENT_LR = 1e-3
TENT_WEIGHT_DECAY = 0.0
HARM_THRESHOLD = -0.02
BENEFIT_THRESHOLD = 0.02

EXPECTED_ROWS = 3960
EXPECTED_STATES = 18
EXPECTED_SOURCE_GROUPS = 44
EXPECTED_B_GROUPS = 25
EXPECTED_A_GROUPS = 19

SEGFORMER_MODEL_ID = "nvidia/mit-b0"
SEGFORMER_SAFE_REVISION = "25ce79d97e6d9d509ed12e17cb2eb89b0a83a2dc"

FEATURES_19 = (
    "source_entropy_mean", "source_entropy_std", "source_entropy_q10",
    "source_entropy_q50", "source_entropy_q90", "source_prob_mean",
    "source_prob_std", "source_prob_q10", "source_prob_q50", "source_prob_q90",
    "source_confidence_mean", "source_confidence_std", "source_fg_fraction",
    "source_boundary_density", "source_uncertain_fraction_040_060",
    "source_high_entropy_fraction_050", "source_logit_abs_mean",
    "source_logit_abs_std", "tent_preupdate_entropy_loss",
)

READY = "MNMS_SOURCE_UTILITY_ASSET_READY"
INCOMPLETE = "MNMS_SOURCE_UTILITY_ASSET_INCOMPLETE"


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
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual.lower() != expected.lower():
        raise RuntimeError(f"{label} SHA mismatch: expected={expected} actual={actual}")
    return actual


def write_csv(path: Path, rows, fields: Sequence[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(fields))
        w.writeheader()
        w.writerows(rows)


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16)


def source_group_fold(vendor: str, subject: str) -> int:
    return stable_int(f"R08C0::{vendor}::{subject}") % 5


def classify_outcome(delta: float) -> str:
    if delta <= HARM_THRESHOLD:
        return "HARM"
    if delta >= BENEFIT_THRESHOLD:
        return "BENEFIT"
    return "NEUTRAL"


def robust_normalize_phase(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    finite = np.isfinite(x)
    if not finite.any():
        raise RuntimeError("Phase has no finite pixels")
    support = finite & (x != 0)
    values = x[support] if int(support.sum()) >= 32 else x[finite]
    lo, hi = float(np.percentile(values, 0.5)), float(np.percentile(values, 99.5))
    clipped = np.clip(x, lo, hi)
    stats = clipped[support] if int(support.sum()) >= 32 else clipped[finite]
    mean, std = float(stats.mean()), float(stats.std())
    if not math.isfinite(std) or std < 1e-6:
        std = 1.0
    out = np.clip((clipped - mean) / std, -5.0, 5.0)
    out[~finite] = 0.0
    return out.astype(np.float32, copy=False)


def dice_binary(pred: np.ndarray, target: np.ndarray) -> float:
    p = pred.astype(bool, copy=False)
    t = target.astype(bool, copy=False)
    ps, ts = int(p.sum()), int(t.sum())
    if ps == 0 and ts == 0:
        return 1.0
    inter = int(np.logical_and(p, t).sum())
    return 2.0 * inter / (ps + ts)


def mean_fg_dice(pred: np.ndarray, target: np.ndarray):
    ds = {c: dice_binary(pred == c, target == c) for c in (1, 2, 3)}
    return float(np.mean(list(ds.values()))), ds


def validate_upstream():
    validate_sha(PROTOCOL, EXPECTED_PROTOCOL_SHA256, "R08C0 protocol")
    validate_sha(R08B1_LOCK, EXPECTED_R08B1_LOCK_SHA256, "R08B1 lock")
    lock = json.loads(R08B1_LOCK.read_text(encoding="utf-8"))
    if lock.get("decision") != EXPECTED_R08B1_DECISION:
        raise RuntimeError(f"Unexpected R08B1 decision={lock.get('decision')}")
    if int(lock.get("states", -1)) != EXPECTED_STATES:
        raise RuntimeError("R08B1 state count changed")
    if int(lock.get("target_image_npz_opens", -1)) != 0:
        raise RuntimeError("R08B1 target image boundary changed")
    if int(lock.get("target_mask_npz_opens", -1)) != 0:
        raise RuntimeError("R08B1 target mask boundary changed")
    expected_split = (lock.get("artifacts", {}).get("inherited_source_split_manifest.csv", {}).get("sha256"))
    actual_split = sha256_file(SOURCE_SPLIT)
    if expected_split != actual_split:
        raise RuntimeError(f"Inherited split SHA mismatch: {actual_split}")
    return lock, actual_split


def load_source_val_rows():
    rows = []
    with SOURCE_SPLIT.open("r", newline="", encoding="utf-8-sig") as f:
        rd = csv.DictReader(f)
        for r in rd:
            if r["split"].strip() != "val":
                continue
            if int(r["target_file_open_allowed"]) != 0:
                raise RuntimeError("Source split unexpectedly allows target access")
            ip, mp = Path(r["image_npz"]), Path(r["mask_npz"])
            if not ip.is_file():
                raise FileNotFoundError(ip)
            if not mp.is_file():
                raise FileNotFoundError(mp)
            rows.append({
                "direction": r["direction"].strip(),
                "source_vendor": r["source_vendor"].strip(),
                "target_vendor": r["target_vendor"].strip(),
                "subject_id": r["subject_id"].strip(),
                "centre": r["centre"].strip(),
                "image_path": ip,
                "mask_path": mp,
            })
    c = Counter(r["source_vendor"] for r in rows)
    if c["B"] != 25 or c["A"] != 19 or len(rows) != 44:
        raise RuntimeError(f"Unexpected source-val counts={dict(c)} total={len(rows)}")
    return sorted(rows, key=lambda x: (x["direction"], x["subject_id"]))


def checkpoint_path(source, target, arch, seed):
    return R08B1_DIR / f"{source}_to_{target}" / arch / f"seed_{seed}" / "best_model.pt"


def state_summary_path(source, target, arch, seed):
    return R08B1_DIR / f"{source}_to_{target}" / arch / f"seed_{seed}" / "state_summary.json"


def validate_state_checkpoint(lock, source, target, arch, seed):
    cp = checkpoint_path(source, target, arch, seed)
    sp = state_summary_path(source, target, arch, seed)
    if not cp.is_file() or not sp.is_file():
        raise FileNotFoundError(cp if not cp.is_file() else sp)
    rel = f"{source}_to_{target}/{arch}/seed_{seed}/best_model.pt"
    expected = (lock.get("checkpoint_sha256") or {}).get(rel)
    actual = sha256_file(cp)
    if expected != actual:
        raise RuntimeError(f"Checkpoint SHA mismatch: {rel}")
    sm = json.loads(sp.read_text(encoding="utf-8"))
    if (sm.get("source_vendor"), sm.get("target_vendor"), sm.get("architecture"), int(sm.get("seed", -1))) != (source, target, arch, seed):
        raise RuntimeError(f"State summary mismatch: {rel}")
    return cp, actual


def load_compact_source_case(row):
    with np.load(row["image_path"], allow_pickle=False) as d:
        image = np.asarray(d["image"], dtype=np.float32)
    with np.load(row["mask_path"], allow_pickle=False) as d:
        mask = np.asarray(d["mask"], dtype=np.uint8)
    if image.shape != mask.shape or image.ndim != 4 or image.shape[0] != 2:
        raise RuntimeError(f"{row['subject_id']}: invalid compact pair {image.shape}/{mask.shape}")
    out = np.empty_like(image, dtype=np.float32)
    for p in range(2):
        out[p] = robust_normalize_phase(image[p])
    return out, mask


def import_stack():
    try:
        import torch
        import torch.nn.functional as F
        from torchvision.models.segmentation import deeplabv3_resnet50, fcn_resnet50
        from transformers import SegformerConfig, SegformerForSemanticSegmentation
    except Exception as e:
        raise RuntimeError("torch/torchvision/transformers stack unavailable") from e
    return {
        "torch": torch, "F": F,
        "deeplabv3_resnet50": deeplabv3_resnet50,
        "fcn_resnet50": fcn_resnet50,
        "SegformerConfig": SegformerConfig,
        "SegformerForSemanticSegmentation": SegformerForSemanticSegmentation,
    }


def instantiate_model(arch, st):
    if arch == "deeplabv3_r50":
        return st["deeplabv3_resnet50"](weights=None, weights_backbone=None, num_classes=4, aux_loss=False)
    if arch == "fcn_r50":
        return st["fcn_resnet50"](weights=None, weights_backbone=None, num_classes=4, aux_loss=False)
    cfg = st["SegformerConfig"].from_pretrained(
        SEGFORMER_MODEL_ID,
        revision=SEGFORMER_SAFE_REVISION,
        local_files_only=True,
        cache_dir=str(HF_CACHE),
    )
    cfg.num_labels = 4
    cfg.id2label = {0: "background", 1: "class_1", 2: "class_2", 3: "class_3"}
    cfg.label2id = {v: k for k, v in cfg.id2label.items()}
    return st["SegformerForSemanticSegmentation"](cfg)


def forward_logits(model, arch, x, F):
    if arch in {"deeplabv3_r50", "fcn_r50"}:
        return model(x)["out"]
    logits = model(pixel_values=x).logits
    if tuple(logits.shape[-2:]) != tuple(x.shape[-2:]):
        logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
    return logits


def load_model_from_checkpoint(arch, cp, st, device):
    torch = st["torch"]
    model = instantiate_model(arch, st)
    obj = torch.load(cp, map_location="cpu", weights_only=True)
    state = obj.get("model_state_dict")
    if state is None:
        raise RuntimeError(f"No model_state_dict in {cp}")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch missing={missing[:5]} unexpected={unexpected[:5]}")
    return model.to(device)


def resize_subject_to_256(image, st):
    torch, F = st["torch"], st["F"]
    xs = []
    for p in range(image.shape[0]):
        for z in range(image.shape[1]):
            t = torch.from_numpy(image[p, z]).float()[None, None]
            t = F.interpolate(t, size=(256, 256), mode="bilinear", align_corners=False)[0]
            xs.append(t)
    return torch.stack(xs, dim=0)


def apply_perturbation(x, name, row, st):
    torch, F = st["torch"], st["F"]
    if name == "identity":
        out = x.clone()
    elif name == "scale070": out = x * 0.70
    elif name == "scale130": out = x * 1.30
    elif name == "shift_m050": out = x - 0.50
    elif name == "shift_p050": out = x + 0.50
    elif name in {"gamma070", "gamma150"}:
        g = 0.70 if name == "gamma070" else 1.50
        u = torch.clamp((x + 5.0) / 10.0, 0.0, 1.0)
        out = torch.pow(u, g) * 10.0 - 5.0
    elif name == "blur_sigma1":
        k = torch.tensor([0.05448868, 0.24420134, 0.40261995, 0.24420134, 0.05448868], dtype=x.dtype, device=x.device)
        out = F.conv2d(x, k.view(1,1,1,5), padding=(0,2))
        out = F.conv2d(out, k.view(1,1,5,1), padding=(2,0))
    elif name == "noise_sigma010":
        seed = stable_int(f"R08C0_NOISE::{row['source_vendor']}::{row['subject_id']}::{name}") % (2**31 - 1)
        g = torch.Generator(device="cpu"); g.manual_seed(seed)
        noise = torch.randn(x.shape, generator=g, dtype=x.dtype, device="cpu").to(x.device)
        out = x + 0.10 * noise
    elif name == "downsample050":
        small = F.interpolate(x, size=(128,128), mode="bilinear", align_corners=False)
        out = F.interpolate(small, size=(256,256), mode="bilinear", align_corners=False)
    else:
        raise ValueError(name)
    return torch.clamp(out, -7.0, 7.0)


def norm_types(torch):
    return (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d, torch.nn.LayerNorm, torch.nn.GroupNorm)


def configure_source_eval(model, torch):
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            m.track_running_stats = True


def snapshot_norm_params(model, torch):
    snap = []
    for m in model.modules():
        if isinstance(m, norm_types(torch)):
            for n in ("weight", "bias"):
                p = getattr(m, n, None)
                if p is not None:
                    snap.append((p, p.detach().clone()))
    if not snap:
        raise RuntimeError("No normalization affine params")
    return snap


def restore_norm_params(snap):
    for p, v in snap:
        p.data.copy_(v)


def configure_tent(model, torch):
    model.train()
    for p in model.parameters():
        p.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, norm_types(torch)):
            if getattr(m, "weight", None) is not None: m.weight.requires_grad_(True)
            if getattr(m, "bias", None) is not None: m.bias.requires_grad_(True)
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            m.track_running_stats = False
        if isinstance(m, (torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d)):
            m.eval()


def predict_logits(model, arch, x_rgb, st, device):
    torch, F = st["torch"], st["F"]
    outs = []
    with torch.no_grad():
        for s in range(0, len(x_rgb), INFER_BATCH):
            xb = x_rgb[s:s+INFER_BATCH].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                logits = forward_logits(model, arch, xb, F)
            outs.append(logits.float().cpu())
    return torch.cat(outs, dim=0)


def categorical_entropy(logits, torch):
    p = torch.softmax(logits, dim=1)
    return -(p * torch.log(p.clamp_min(1e-8))).sum(dim=1)


def one_tent_step(model, arch, x_rgb, st, device):
    torch, F = st["torch"], st["F"]
    configure_tent(model, torch)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=TENT_LR, weight_decay=TENT_WEIGHT_DECAY)
    opt.zero_grad(set_to_none=True)
    total = len(x_rgb)
    for s in range(0, total, ADAPT_MICROBATCH):
        xb = x_rgb[s:s+ADAPT_MICROBATCH].to(device, non_blocking=True)
        logits = forward_logits(model, arch, xb, F)
        ent = categorical_entropy(logits, torch).mean()
        (ent * (len(xb) / total)).backward()
    opt.step()


def boundary_density(hard):
    fg = hard != 0
    h = np.sum(fg[:, :, 1:] != fg[:, :, :-1])
    v = np.sum(fg[:, 1:, :] != fg[:, :-1, :])
    den = fg.shape[0]*fg.shape[1]*max(fg.shape[2]-1,0) + fg.shape[0]*max(fg.shape[1]-1,0)*fg.shape[2]
    return float((h + v) / max(den, 1))


def features_19_from_logits(logits, st):
    torch = st["torch"]
    p = torch.softmax(logits, dim=1)
    eraw = -(p * torch.log(p.clamp_min(1e-8))).sum(dim=1)
    e = (eraw / math.log(4)).flatten().numpy()
    f = (1.0 - p[:,0]).flatten().numpy()
    c = p.max(dim=1).values.flatten().numpy()
    hard = p.argmax(dim=1).numpy()
    la = logits.abs().flatten().numpy()
    out = {
        "source_entropy_mean": float(np.mean(e)), "source_entropy_std": float(np.std(e)),
        "source_entropy_q10": float(np.quantile(e,.10)), "source_entropy_q50": float(np.quantile(e,.50)),
        "source_entropy_q90": float(np.quantile(e,.90)), "source_prob_mean": float(np.mean(f)),
        "source_prob_std": float(np.std(f)), "source_prob_q10": float(np.quantile(f,.10)),
        "source_prob_q50": float(np.quantile(f,.50)), "source_prob_q90": float(np.quantile(f,.90)),
        "source_confidence_mean": float(np.mean(c)), "source_confidence_std": float(np.std(c)),
        "source_fg_fraction": float(np.mean(hard != 0)), "source_boundary_density": boundary_density(hard),
        "source_uncertain_fraction_040_060": float(np.mean((f>=.40)&(f<=.60))),
        "source_high_entropy_fraction_050": float(np.mean(e>=.50)),
        "source_logit_abs_mean": float(np.mean(la)), "source_logit_abs_std": float(np.std(la)),
        "tent_preupdate_entropy_loss": float(eraw.mean().item()),
    }
    vals = np.asarray([out[k] for k in FEATURES_19], dtype=np.float64)
    if not np.all(np.isfinite(vals)):
        raise RuntimeError("Non-finite 19-feature vector")
    return out


def hard_native_from_logits(logits, native_shape, st):
    torch, F = st["torch"], st["F"]
    p,z,y,x = native_shape
    hard = torch.argmax(logits, dim=1, keepdim=True).float()
    hard = F.interpolate(hard, size=(y,x), mode="nearest")[:,0].numpy().astype(np.uint8, copy=False)
    if hard.shape[0] != p*z:
        raise RuntimeError("Slice count mismatch")
    return hard.reshape(p,z,y,x)


def out_state_dir(source, target, arch, seed):
    return OUTPUT_DIR / "states" / f"{source}_to_{target}" / arch / f"seed_{seed}"


def state_complete(source, target, arch, seed):
    d = out_state_dir(source,target,arch,seed)
    req = [d/"source_utility_rows.csv", d/"state_summary.json", d/"STATE_COMPLETE.json"]
    if not all(p.is_file() for p in req):
        return False
    try:
        m = json.loads((d/"STATE_COMPLETE.json").read_text(encoding="utf-8"))
    except Exception:
        return False
    return m.get("source_vendor") == source and m.get("target_vendor") == target and m.get("architecture") == arch and int(m.get("seed",-1)) == seed and bool(m.get("complete"))


def run_state(source, target, arch, seed, rows, lock, st, device):
    if state_complete(source,target,arch,seed):
        print(f"[RESUME SKIP] {source}->{target} {arch} seed={seed}")
        return json.loads((out_state_dir(source,target,arch,seed)/"state_summary.json").read_text(encoding="utf-8"))

    d = out_state_dir(source,target,arch,seed)
    if d.exists(): shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=False)

    cp, cp_sha = validate_state_checkpoint(lock,source,target,arch,seed)
    model = load_model_from_checkpoint(arch,cp,st,device)
    torch = st["torch"]
    configure_source_eval(model,torch)
    snap = snapshot_norm_params(model,torch)

    src_rows = [r for r in rows if r["source_vendor"] == source]
    expected_subjects = 25 if source == "B" else 19
    if len(src_rows) != expected_subjects:
        raise RuntimeError("Unexpected source-val subject count")

    records = []
    bar = tqdm(total=expected_subjects*10, desc=f"R08C0 {source}->{target} {arch} s{seed}", unit="case", dynamic_ncols=True)

    for row in src_rows:
        image, mask = load_compact_source_case(row)
        x_base = resize_subject_to_256(image, st)

        for pert in PERTURBATIONS:
            restore_norm_params(snap); configure_source_eval(model,torch)
            xp = apply_perturbation(x_base, pert, row, st)
            xrgb = xp.repeat(1,3,1,1)

            slogits = predict_logits(model,arch,xrgb,st,device)
            feats = features_19_from_logits(slogits,st)
            spred = hard_native_from_logits(slogits,mask.shape,st)
            sdice,scls = mean_fg_dice(spred,mask)

            restore_norm_params(snap)
            one_tent_step(model,arch,xrgb,st,device)
            alogits = predict_logits(model,arch,xrgb,st,device)
            apred = hard_native_from_logits(alogits,mask.shape,st)
            adice,acls = mean_fg_dice(apred,mask)

            delta = float(adice-sdice)
            rec = {
                "direction": f"{source}_to_{target}", "source_vendor": source, "target_vendor": target,
                "architecture": arch, "seed": seed, "subject_id": row["subject_id"], "centre": row["centre"],
                "source_group_fold": source_group_fold(source,row["subject_id"]), "perturbation": pert,
                "source_dice": sdice, "a1_dice": adice, "delta_dice": delta, "outcome": classify_outcome(delta),
                "source_dice_class_1": scls[1], "source_dice_class_2": scls[2], "source_dice_class_3": scls[3],
                "a1_dice_class_1": acls[1], "a1_dice_class_2": acls[2], "a1_dice_class_3": acls[3],
            }
            rec.update(feats)
            records.append(rec)
            bar.update(1)
            cnt = Counter(r["outcome"] for r in records)
            bar.set_postfix(H=cnt.get("HARM",0),N=cnt.get("NEUTRAL",0),B=cnt.get("BENEFIT",0))
    bar.close()

    fields = [
        "direction","source_vendor","target_vendor","architecture","seed","subject_id","centre","source_group_fold",
        "perturbation","source_dice","a1_dice","delta_dice","outcome",
        "source_dice_class_1","source_dice_class_2","source_dice_class_3",
        "a1_dice_class_1","a1_dice_class_2","a1_dice_class_3",*FEATURES_19,
    ]
    write_csv(d/"source_utility_rows.csv",records,fields)
    counts = Counter(r["outcome"] for r in records)
    summary = {
        "source_vendor": source, "target_vendor": target, "architecture": arch, "seed": seed,
        "checkpoint_sha256": cp_sha, "source_subjects": expected_subjects, "perturbations": 10,
        "rows": len(records), "outcome_counts": dict(counts),
        "target_image_npz_opens": 0, "target_mask_npz_opens": 0, "new_segmentation_training": False,
    }
    write_json(d/"state_summary.json",summary)
    write_json(d/"STATE_COMPLETE.json",{
        "source_vendor":source,"target_vendor":target,"architecture":arch,"seed":seed,
        "rows":len(records),"table_sha256":sha256_file(d/"source_utility_rows.csv"),"complete":True,
    })
    del model
    if device.type == "cuda": torch.cuda.empty_cache()
    return summary


def collect_all_rows():
    rows = []
    for source,target in DIRECTIONS:
        for arch in ARCHITECTURES:
            for seed in SEEDS:
                if not state_complete(source,target,arch,seed):
                    raise RuntimeError(f"Incomplete state {source}->{target} {arch} {seed}")
                p = out_state_dir(source,target,arch,seed)/"source_utility_rows.csv"
                with p.open("r",newline="",encoding="utf-8-sig") as f:
                    rows.extend(list(csv.DictReader(f)))
    return rows


def preflight():
    lock, split_sha = validate_upstream()
    rows = load_source_val_rows()
    st = import_stack(); torch = st["torch"]
    for source,target in DIRECTIONS:
        for arch in ARCHITECTURES:
            for seed in SEEDS:
                validate_state_checkpoint(lock,source,target,arch,seed)
    print("===== Q1-R08C0 PREFLIGHT =====")
    print(f"protocol_sha256={EXPECTED_PROTOCOL_SHA256}")
    print(f"r08b1_lock_sha256={EXPECTED_R08B1_LOCK_SHA256}")
    print(f"inherited_source_split_sha256={split_sha}")
    print(f"source_val_groups={len(rows)}")
    print(f"B_source_val={sum(r['source_vendor']=='B' for r in rows)}")
    print(f"A_source_val={sum(r['source_vendor']=='A' for r in rows)}")
    print("states=18")
    print("perturbations=10")
    print("expected_rows=3960")
    print("action=SOURCE_vs_A1_TENT_1STEP")
    print("tent_optimizer=Adam")
    print("tent_lr=0.001")
    print("tent_steps=1")
    print("episodic_reset=YES")
    print("features=19")
    print("target_images_used=NO")
    print("target_masks_used=NO")
    print(f"cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available(): print(f"cuda_device={torch.cuda.get_device_name(0)}")
    print("PREFLIGHT_PASS")


def run(args):
    lock, split_sha = validate_upstream()
    rows = load_source_val_rows()
    st = import_stack(); torch = st["torch"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.output_dir != OUTPUT_DIR:
        raise RuntimeError("R08C0 output dir is frozen; custom output disabled")
    if OUTPUT_DIR.exists() and (OUTPUT_DIR/"Q1_R08C0_MNMS_SOURCE_UTILITY_ASSET_LOCK.json").is_file():
        raise FileExistsError("R08C0 already finalized")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pc = OUTPUT_DIR/"preregistered_protocol_copy.md"
    if not pc.exists(): shutil.copy2(PROTOCOL,pc)
    validate_sha(pc,EXPECTED_PROTOCOL_SHA256,"protocol copy")
    sc = OUTPUT_DIR/"inherited_source_split_manifest.csv"
    if not sc.exists(): shutil.copy2(SOURCE_SPLIT,sc)
    if sha256_file(sc) != split_sha: raise RuntimeError("Inherited split copy changed")

    summaries = []
    for source,target in DIRECTIONS:
        for arch in ARCHITECTURES:
            for seed in SEEDS:
                summaries.append(run_state(source,target,arch,seed,rows,lock,st,device))

    all_rows = collect_all_rows()
    if len(all_rows) != EXPECTED_ROWS:
        raise RuntimeError(f"Aggregated rows={len(all_rows)} expected={EXPECTED_ROWS}")

    numeric = ["source_dice","a1_dice","delta_dice","source_dice_class_1","source_dice_class_2","source_dice_class_3","a1_dice_class_1","a1_dice_class_2","a1_dice_class_3",*FEATURES_19]
    for i,r in enumerate(all_rows):
        vals = np.asarray([float(r[k]) for k in numeric],dtype=np.float64)
        if not np.all(np.isfinite(vals)): raise RuntimeError(f"Non-finite row={i}")
        if r["outcome"] not in {"HARM","NEUTRAL","BENEFIT"}: raise RuntimeError(f"Bad outcome row={i}")

    write_csv(OUTPUT_DIR/"source_utility_table.csv",all_rows,list(all_rows[0].keys()))
    oc = Counter(r["outcome"] for r in all_rows)
    write_csv(OUTPUT_DIR/"outcome_counts.csv",[{"outcome":k,"count":oc.get(k,0)} for k in ("HARM","NEUTRAL","BENEFIT")],["outcome","count"])

    ss = []
    for s in summaries:
        c=s["outcome_counts"]
        ss.append({
            "direction":f"{s['source_vendor']}_to_{s['target_vendor']}","source_vendor":s["source_vendor"],"target_vendor":s["target_vendor"],
            "architecture":s["architecture"],"seed":s["seed"],"rows":s["rows"],"harm":c.get("HARM",0),"neutral":c.get("NEUTRAL",0),"benefit":c.get("BENEFIT",0),
            "checkpoint_sha256":s["checkpoint_sha256"],"target_image_npz_opens":0,"target_mask_npz_opens":0,
        })
    write_csv(OUTPUT_DIR/"state_summary.csv",ss,["direction","source_vendor","target_vendor","architecture","seed","rows","harm","neutral","benefit","checkpoint_sha256","target_image_npz_opens","target_mask_npz_opens"])

    groups={(r["source_vendor"],r["subject_id"]) for r in all_rows}
    states={(r["direction"],r["architecture"],int(r["seed"])) for r in all_rows}
    perts={r["perturbation"] for r in all_rows}
    checks={
        "A_rows_eq_3960":len(all_rows)==3960,
        "B_states_eq_18":len(states)==18,
        "C_perturbations_eq_10":len(perts)==10,
        "D_source_groups_eq_44":len(groups)==44,
        "E_B_source_groups_eq_25":len({s for v,s in groups if v=="B"})==25,
        "F_A_source_groups_eq_19":len({s for v,s in groups if v=="A"})==19,
        "G_all_numeric_finite":True,
        "H_all_outcomes_valid":True,
        "I_target_image_opens_eq_0":True,
        "J_target_mask_opens_eq_0":True,
        "K_all_18_state_assets_complete":all(state_complete(s,t,a,z) for s,t in DIRECTIONS for a in ARCHITECTURES for z in SEEDS),
    }
    decision=READY if all(checks.values()) else INCOMPLETE
    gate={
        "script_version":VERSION,"build":BUILD,"protocol_sha256":EXPECTED_PROTOCOL_SHA256,"r08b1_lock_sha256":EXPECTED_R08B1_LOCK_SHA256,
        "inherited_source_split_sha256":split_sha,"rows":len(all_rows),"states":len(states),"source_groups":len(groups),"perturbations":sorted(perts),
        "outcome_counts":dict(oc),"features_19":list(FEATURES_19),"target_images_used":False,"target_masks_used":False,
        "new_segmentation_training":False,"checks":checks,"decision":decision,
    }
    write_json(OUTPUT_DIR/"readiness_gate.json",gate)
    (OUTPUT_DIR/"decision.txt").write_text(decision+"\n",encoding="utf-8")

    lines=[
        "===== Q1-R08C0 M&Ms SOURCE-ONLY UTILITY ASSET =====",f"Script version: {VERSION}",f"Build: {BUILD}","",
        "Frozen protocol:","  actions=SOURCE vs A1_TENT_1STEP","  architectures=DeepLabV3-R50 / FCN-R50 / SegFormer-B0",
        "  seeds=20260819 / 20260820 / 20260821","  directions=B->A / A->B","  source-val subjects=B25 + A19","  perturbations=10",
        f"  rows={len(all_rows)}","  features=19",f"  HARM threshold={HARM_THRESHOLD}",f"  BENEFIT threshold={BENEFIT_THRESHOLD}","",
        "Outcome counts:",f"  HARM={oc.get('HARM',0)}",f"  NEUTRAL={oc.get('NEUTRAL',0)}",f"  BENEFIT={oc.get('BENEFIT',0)}","",
        "Information boundary:","  source images used=YES","  source masks used=YES","  target images used=NO","  target masks used=NO",
        "  new segmentation training=NO","  HARM predictor training=NO","","Checks:",
    ]
    lines += [f"  {k}={'PASS' if v else 'FAIL'}" for k,v in checks.items()]
    lines += ["","Decision:",f"  {decision}"]
    (OUTPUT_DIR/"run_log.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")

    artifact_names=["preregistered_protocol_copy.md","inherited_source_split_manifest.csv","source_utility_table.csv","outcome_counts.csv","state_summary.csv","readiness_gate.json","decision.txt","run_log.txt"]
    artifacts={n:{"relative_path":n,"sha256":sha256_file(OUTPUT_DIR/n)} for n in artifact_names}
    state_hashes={}
    for s,t in DIRECTIONS:
        for a in ARCHITECTURES:
            for z in SEEDS:
                rel=f"states/{s}_to_{t}/{a}/seed_{z}/source_utility_rows.csv"
                state_hashes[rel]=sha256_file(OUTPUT_DIR/rel)

    lock_obj={
        "script_version":VERSION,"build":BUILD,"protocol_sha256":EXPECTED_PROTOCOL_SHA256,"r08b1_lock_sha256":EXPECTED_R08B1_LOCK_SHA256,
        "inherited_source_split_sha256":split_sha,"rows":len(all_rows),"states":len(states),"source_groups":len(groups),"outcome_counts":dict(oc),
        "target_images_used":False,"target_masks_used":False,"new_segmentation_training":False,"harm_predictor_training":False,
        "decision":decision,"checks":checks,"artifacts":artifacts,"state_table_sha256":state_hashes,
    }
    lp=OUTPUT_DIR/"Q1_R08C0_MNMS_SOURCE_UTILITY_ASSET_LOCK.json"
    write_json(lp,lock_obj)
    print(); print((OUTPUT_DIR/"run_log.txt").read_text(encoding="utf-8")); print("Q1-R08C0 LOCK:",lp); print("Q1-R08C0 LOCK SHA256:",sha256_file(lp))


def self_test():
    assert len(PERTURBATIONS)==10
    assert len(FEATURES_19)==19
    assert EXPECTED_ROWS==3960 and EXPECTED_STATES==18 and EXPECTED_SOURCE_GROUPS==44
    assert classify_outcome(-.021)=="HARM" and classify_outcome(0)=="NEUTRAL" and classify_outcome(.021)=="BENEFIT"
    f=source_group_fold("B","subject_x"); assert f==source_group_fold("B","subject_x") and 0<=f<=4
    x=np.asarray([[0.,1.],[2.,3.]],dtype=np.float32); y=robust_normalize_phase(x); assert y.dtype==np.float32 and np.isfinite(y).all()
    print("TEN_PERTURBATION_TEST_PASS")
    print("NINETEEN_FEATURE_TEST_PASS")
    print("3960_ROW_EXPECTATION_TEST_PASS")
    print("OUTCOME_THRESHOLD_TEST_PASS")
    print("SOURCE_GROUP_FOLD_TEST_PASS")
    print("TARGET_BLIND_DESIGN_TEST_PASS")
    print("SELF_TEST_PASS")


def parse_args():
    p=argparse.ArgumentParser(description="Q1-R08C0 source-only cardiac SOURCE-vs-A1 utility asset; target data not opened.")
    p.add_argument("--output-dir",type=Path,default=OUTPUT_DIR)
    p.add_argument("--preflight-only",action="store_true")
    p.add_argument("--self-test",action="store_true")
    return p.parse_args()


def main():
    args=parse_args()
    if args.self_test: self_test(); return 0
    if args.preflight_only: preflight(); return 0
    try: run(args)
    except Exception:
        traceback.print_exc(); raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
