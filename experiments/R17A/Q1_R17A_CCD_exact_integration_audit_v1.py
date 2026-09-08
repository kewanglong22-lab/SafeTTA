#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q1_R17A_CCD_exact_integration_audit_v1.py

READ-ONLY exact integration audit before generating SicTTA CCD scores.

Purpose
-------
The broad R17A asset audit established that relevant assets exist, but did not
prove exact row alignment or identify the precise SOURCE probability/logit
tensor for each final evaluation condition.

This audit:
1) inventories candidate SOURCE probability/logit/prediction files for
   PolypGen, SUN-SEG, NeoPolyp, Prostate158, and PROMISE12;
2) inspects NPY/NPZ shape/key metadata without loading large arrays fully;
3) finds final/frozen inference scripts and reports SOURCE-forward/softmax/logit
   code snippets;
4) inventories final SOURCE checkpoints;
5) identifies whether CCD can be computed directly from frozen soft
   probabilities or requires GT-free SOURCE re-inference.

No training, inference, GT loading, deletion, or moving is performed.

Output:
  F:\\MEDSEG_SAFETTA\\outputs\\Q1_R17A_CCD_exact_integration_audit_v1
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

VERSION = "2026-09-08-Q1-R17A-CCD-EXACT-INTEGRATION-AUDIT-v1"
DEFAULT_ROOT = Path(r"F:\MEDSEG_SAFETTA")
OUT_REL = Path(r"outputs\Q1_R17A_CCD_exact_integration_audit_v1")

SEARCH_ROOTS = [
    Path("outputs"),
    Path("code"),
    Path("release/SafeTTA_clean_test_v102"),
    Path("release/public_v1_fix4"),
    Path("data"),
    Path("cross_modality_prostate_mri"),
]

DATASET_PATTERNS = {
    "NeoPolyp": ["neopolyp", "neo_polyp"],
    "PolypGen": ["polypgen"],
    "SUNSEG": ["sunseg", "sun_seg", "sun-seg"],
    "Prostate158": ["prostate158", "prostate_158"],
    "PROMISE12": ["promise12"],
}

SOFT_TOKENS = (
    "logit", "logits", "prob", "probs", "probability", "softmax"
)
HARD_TOKENS = (
    "mask", "masks", "packbits", "prediction"
)
CKPT_SUFFIXES = {".pt", ".pth", ".ckpt"}
ARRAY_SUFFIXES = {".npy", ".npz"}
TEXT_SUFFIXES = {".py", ".txt", ".md", ".json", ".csv"}

SCRIPT_ROUTE_TOKENS = (
    "polypgen", "sunseg", "promise12", "prostate158", "neopolyp",
    "r10l3", "r14c", "cm3", "cm4", "cm5", "cm6", "source"
)

CODE_SIGNAL_TOKENS = (
    "softmax(", ".softmax(", "sigmoid(", "logit", "source",
    "checkpoint", "state_dict", "model(", "output"
)


def fmt_bytes(n: int) -> str:
    x = float(n)
    for u in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if x < 1024 or u == "TiB":
            return f"{x:.2f} {u}"
        x /= 1024
    return f"{n} B"


def norm(p: Path, root: Path) -> str:
    return str(p.relative_to(root)).replace("\\", "/")


def dataset_tags(rel: str) -> List[str]:
    low = rel.lower()
    tags = []
    for ds, pats in DATASET_PATTERNS.items():
        if any(x in low for x in pats):
            tags.append(ds)
    # Route aliases that do not always spell dataset name.
    if any(x in low for x in ["q1x_cm3", "q1x_cm4"]):
        if "Prostate158" not in tags:
            tags.append("Prostate158")
    if any(x in low for x in ["q1x_cm5", "q1x_cm6", "q1x_cm7"]):
        if "PROMISE12" not in tags:
            tags.append("PROMISE12")
    if any(x in low for x in ["q1_r14"]):
        if "SUNSEG" not in tags:
            tags.append("SUNSEG")
    if any(x in low for x in ["q1_r10l3", "s06_c"]):
        if "PolypGen" not in tags:
            tags.append("PolypGen")
    return sorted(set(tags))


def npy_meta(path: Path) -> Dict[str, object]:
    try:
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
        return {
            "format": "npy",
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "keys": "",
            "member_meta": "",
            "status": "OK",
        }
    except Exception as e:
        return {
            "format": "npy",
            "shape": "",
            "dtype": "",
            "keys": "",
            "member_meta": "",
            "status": f"ERROR:{type(e).__name__}:{e}",
        }


def npz_meta(path: Path) -> Dict[str, object]:
    """
    Read NPZ key/header metadata. np.load is lazy for npz; individual members
    are opened only to inspect shape/dtype. Avoid materializing large data.
    """
    try:
        z = np.load(path, allow_pickle=False)
        metas = []
        keys = list(z.files)
        for key in keys[:30]:
            try:
                a = z[key]
                metas.append(f"{key}:shape={tuple(a.shape)},dtype={a.dtype}")
                del a
            except Exception as e:
                metas.append(f"{key}:ERROR:{type(e).__name__}")
        z.close()
        return {
            "format": "npz",
            "shape": "",
            "dtype": "",
            "keys": "|".join(keys[:50]),
            "member_meta": " | ".join(metas),
            "status": "OK",
        }
    except Exception as e:
        return {
            "format": "npz",
            "shape": "",
            "dtype": "",
            "keys": "",
            "member_meta": "",
            "status": f"ERROR:{type(e).__name__}:{e}",
        }


def scan_arrays(root: Path) -> List[dict]:
    candidates = []
    for rel_root in SEARCH_ROOTS:
        base = root / rel_root
        if not base.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(base, followlinks=False):
            dp = Path(dirpath)
            dirnames[:] = [
                d for d in dirnames
                if d not in {".git", "__pycache__", ".pytest_cache"}
                and not (dp / d).is_symlink()
            ]
            for name in filenames:
                p = dp / name
                if p.suffix.lower() not in ARRAY_SUFFIXES:
                    continue
                rel = norm(p, root)
                tags = dataset_tags(rel)
                if not tags:
                    continue
                low = rel.lower()
                signal = (
                    "SOFT_CANDIDATE" if any(t in low for t in SOFT_TOKENS)
                    else "HARD_OR_UNKNOWN" if any(t in low for t in HARD_TOKENS)
                    else "UNKNOWN_ARRAY"
                )
                try:
                    size = p.stat().st_size
                except OSError:
                    size = 0
                candidates.append((p, rel, tags, signal, size))

    candidates.sort(key=lambda x: -x[4])
    rows = []
    iterator = candidates
    if tqdm is not None:
        iterator = tqdm(candidates, desc="Inspect NPY/NPZ metadata", unit="file", dynamic_ncols=True)
    for p, rel, tags, signal, size in iterator:
        meta = npy_meta(p) if p.suffix.lower() == ".npy" else npz_meta(p)
        rows.append({
            "path": rel,
            "datasets": "|".join(tags),
            "signal_type": signal,
            "size_bytes": int(size),
            "size_human": fmt_bytes(int(size)),
            **meta,
        })
    return rows


def scan_checkpoints(root: Path) -> List[dict]:
    rows = []
    for rel_root in SEARCH_ROOTS:
        base = root / rel_root
        if not base.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(base, followlinks=False):
            dp = Path(dirpath)
            dirnames[:] = [
                d for d in dirnames
                if d not in {".git", "__pycache__", ".pytest_cache"}
                and not (dp / d).is_symlink()
            ]
            for name in filenames:
                p = dp / name
                if p.suffix.lower() not in CKPT_SUFFIXES:
                    continue
                rel = norm(p, root)
                tags = dataset_tags(rel)
                # Keep source/final model checkpoints even when dataset alias is implicit.
                low = rel.lower()
                if not tags and not any(t in low for t in ["source", "final", "best", "cm3", "cm4"]):
                    continue
                try:
                    size = p.stat().st_size
                except OSError:
                    size = 0
                rows.append({
                    "path": rel,
                    "datasets": "|".join(tags),
                    "size_bytes": int(size),
                    "size_human": fmt_bytes(int(size)),
                })
    rows.sort(key=lambda r: -r["size_bytes"])
    return rows


def scan_inference_code(root: Path) -> List[dict]:
    rows = []
    code_roots = [
        root / "code",
        root / "release/SafeTTA_clean_test_v102/experiments",
        root / "release/SafeTTA_clean_test_v102/method_core",
    ]
    seen = set()
    for base in code_roots:
        if not base.exists():
            continue
        for p in base.rglob("*.py"):
            rel = norm(p, root)
            if rel in seen:
                continue
            seen.add(rel)
            lowrel = rel.lower()
            if not any(t in lowrel for t in SCRIPT_ROUTE_TOKENS):
                continue
            try:
                lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
            except Exception:
                continue
            matches = []
            for i, line in enumerate(lines):
                low = line.lower()
                if any(tok in low for tok in CODE_SIGNAL_TOKENS):
                    start = max(0, i - 2)
                    end = min(len(lines), i + 3)
                    snippet = " || ".join(
                        f"L{j+1}:{lines[j].strip()[:180]}" for j in range(start, end)
                    )
                    matches.append(snippet)
            if matches:
                rows.append({
                    "path": rel,
                    "datasets": "|".join(dataset_tags(rel)),
                    "signal_hit_count": len(matches),
                    "examples": " ### ".join(matches[:8]),
                })
    return rows


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for k in row:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = ap.parse_args()
    root = args.root
    if not root.exists():
        raise FileNotFoundError(root)

    out_dir = root / OUT_REL
    out_dir.mkdir(parents=True, exist_ok=True)

    print("===== R17A CCD EXACT INTEGRATION AUDIT =====")
    print("Version:", VERSION)
    print("Root:", root)
    print("Read-only: YES")
    print("Training: NO")
    print("Inference: NO")
    print("GT loading: NO")
    print("Deletion/move: NO")
    print()
    print("Official SicTTA CCD lock:")
    print("  SOURCE softmax -> [N,C]")
    print("  random pixels=min(200,N)")
    print("  L2 normalize each pixel probability vector")
    print("  G=P^T P")
    print("  softmax(G, dim=1)")
    print("  mean row entropy")
    print("  high CCD = higher HARM-risk direction")
    print("  RNG seed for R17A=20260908")
    print()

    print("[1/4] Inspect candidate probability/logit arrays...")
    arrays = scan_arrays(root)

    print("[2/4] Inventory checkpoints...")
    ckpts = scan_checkpoints(root)

    print("[3/4] Locate exact SOURCE inference code paths...")
    code = scan_inference_code(root)

    print("[4/4] Build dataset readiness...")
    readiness = {}
    for ds in DATASET_PATTERNS:
        ds_arrays = [r for r in arrays if ds in r["datasets"].split("|")]
        soft = [r for r in ds_arrays if r["signal_type"] == "SOFT_CANDIDATE" and r["status"] == "OK"]
        ds_ckpt = [r for r in ckpts if ds in r["datasets"].split("|")]
        ds_code = [r for r in code if ds in r["datasets"].split("|")]
        if soft:
            route = "DIRECT_SOFT_ASSET_CANDIDATE"
        elif ds_ckpt and ds_code:
            route = "GT_FREE_REINFERENCE_CANDIDATE"
        else:
            route = "NEEDS_EXACT_INTEGRATION_REVIEW"
        readiness[ds] = {
            "soft_candidate_count": len(soft),
            "checkpoint_count": len(ds_ckpt),
            "inference_script_count": len(ds_code),
            "route": route,
        }

    write_csv(out_dir / "R17A_CCD_ARRAY_METADATA.csv", arrays)
    write_csv(out_dir / "R17A_CCD_CHECKPOINT_INVENTORY.csv", ckpts)
    write_csv(out_dir / "R17A_CCD_INFERENCE_CODE_AUDIT.csv", code)
    (out_dir / "R17A_CCD_READINESS.json").write_text(
        json.dumps(readiness, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lines = [
        "===== R17A CCD EXACT INTEGRATION AUDIT REPORT =====",
        f"Version: {VERSION}",
        f"Root: {root}",
        "",
        "OFFICIAL-CODE-FAITHFUL CCD:",
        "SOURCE softmax -> reshape [N,C] -> random min(200,N) pixels ->",
        "L2 normalize rows -> G=P^T P -> softmax(G,dim=1) -> mean row entropy.",
        "R17A RNG seed=20260908; lower CCD=more source-friendly; risk direction=CCD.",
        "",
        "DATASET READINESS:",
    ]
    for ds, r in readiness.items():
        lines.append(
            f"{ds}: route={r['route']}; soft={r['soft_candidate_count']}; "
            f"ckpt={r['checkpoint_count']}; inference_scripts={r['inference_script_count']}"
        )
    lines += [
        "",
        f"Array candidates inspected: {len(arrays)}",
        f"Checkpoint candidates: {len(ckpts)}",
        f"Inference-code candidates: {len(code)}",
        "",
        "DECISION:",
        "- DIRECT_SOFT_ASSET_CANDIDATE still requires exact sample/model-state alignment before score generation.",
        "- GT_FREE_REINFERENCE_CANDIDATE is acceptable because CCD is computed from frozen SOURCE inference before GT.",
        "- Hard masks alone are insufficient for a faithful CCD implementation.",
        "- Do not delete R17A-relevant logits/checkpoints while this experiment is open.",
        "",
        "GATE=READY_FOR_R17A_CCD_EXACT_MAPPING_REVIEW",
        "NEXT=FREEZE_DATASET_SPECIFIC_SOURCE_PROBABILITY_MAPPING_AND_RUN_CCD_SCORE_LOCK",
    ]
    report = out_dir / "R17A_CCD_INTEGRATION_AUDIT_REPORT.txt"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print()
    print("GATE=READY_FOR_R17A_CCD_EXACT_MAPPING_REVIEW")
    print("NEXT=FREEZE_DATASET_SPECIFIC_SOURCE_PROBABILITY_MAPPING_AND_RUN_CCD_SCORE_LOCK")
    print("Report:", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
