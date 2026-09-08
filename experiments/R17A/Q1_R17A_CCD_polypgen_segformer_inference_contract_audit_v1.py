#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q1_R17A_CCD_polypgen_segformer_inference_contract_audit_v1.py

SafeTTA R17A — exact SegFormer-B0 inference-contract audit for the remaining
3 PolypGen CCD states.

READ-ONLY
---------
Training: NO
Inference: NO
Target GT loading: NO
HARM/outcome loading: NO
Deletion/move: NO

Why this narrow audit exists
----------------------------
The first 6 PolypGen states already have frozen SOURCE logits and their CCD
scores are locked. The remaining SegFormer-B0 states require GT-free SOURCE
re-inference. We must reuse the project's exact historical SegFormer model
construction, checkpoint-loading, preprocessing, resize/interpolation, and
output-logit semantics rather than guess them.

This script extracts the exact contract from:
  code/Q1_R03_nine_state_heterogeneous_source_utility_asset_v1_fix2.py
  code/Q1_R05D3_neopolyp_source_a1_prediction_lock_v1_fix1.py
  code/Q1_R10L3A_polypgen_source_a1_prediction_lock_fix2.py

It also audits the three frozen SegFormer checkpoints and the corresponding
PolypGen pre-GT hard prediction locks that will later be used for strict parity
validation.

No model forward pass occurs in this script.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import inspect
import json
import os
import re
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

VERSION = "2026-09-08-Q1-R17A-CCD-POLYPGEN-SEGF-INFERENCE-CONTRACT-AUDIT-v1"
DEFAULT_ROOT = Path(r"F:\MEDSEG_SAFETTA")
OUT_REL = Path(r"outputs\Q1_R17A_CCD_polypgen_segformer_inference_contract_audit_v1")

R03 = Path(r"code\Q1_R03_nine_state_heterogeneous_source_utility_asset_v1_fix2.py")
R05 = Path(r"code\Q1_R05D3_neopolyp_source_a1_prediction_lock_v1_fix1.py")
R10 = Path(r"code\Q1_R10L3A_polypgen_source_a1_prediction_lock_fix2.py")

EXPECTED_SCRIPT_SHA = {
    str(R03).replace("\\", "/"): "d737272b756a4d2051f9c74051d7085640eaec9cd8be6dbd951c325af2a17c31",
    str(R05).replace("\\", "/"): "df22d6f3054257743d1f12d9e757323ea9f4a052b7642eb4ab610da46ff21251",
}

CHECKPOINTS = {
    20260820: (
        Path(r"outputs\Q1_R02_segformer_b0_three_seed_source_training_v1_fix7"
             r"\seed_20260820\best_model_state.pt"),
        "9dae2ae907b193ea36c2bccc8e376ddbad1699ba27ae0769cf40bcba13e95605",
    ),
    20260821: (
        Path(r"outputs\Q1_R02_segformer_b0_three_seed_source_training_v1_fix7"
             r"\seed_20260821\best_model_state.pt"),
        "ad75290168eab7d116aa3de61b3eafc1e986f11fc0bbfa4a6108abbf669a258d",
    ),
    20260822: (
        Path(r"outputs\Q1_R02_segformer_b0_three_seed_source_training_v1_fix7"
             r"\seed_20260822\best_model_state.pt"),
        "ea0b3881373e9f966475a082490fabe4b0acae81179b8596c48a06ee2661a34a",
    ),
}

POLYPGEN_STATE_DIR = Path(
    r"outputs\Q1_R10L3A_polypgen_source_a1_prediction_lock_fix2_v1\state_predictions"
)

TARGET_FUNCTIONS = {
    str(R03).replace("\\", "/"): [
        "segformer_logits_and_z",
        "build_segformer_context",
        "load_segformer_state",
        "build_segformer_rows",
    ],
    str(R05).replace("\\", "/"): [
        "run_segformer_state",
    ],
}

KEYWORDS = (
    "segformer", "imageprocessor", "image_processor", "processor",
    "resize", "interpolate", "normalize", "mean", "std",
    "pixel_values", "sigmoid", "softmax", "logits",
    "num_labels", "ignore_mismatched_sizes", "from_pretrained",
    "mit-b0", "nvidia", "352",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path, root: Path) -> str:
    return str(path.relative_to(root)).replace("\\", "/")


def node_signature(node: ast.FunctionDef) -> str:
    parts = []
    args = node.args
    for a in args.posonlyargs:
        parts.append(a.arg)
    if args.posonlyargs:
        parts.append("/")
    for a in args.args:
        parts.append(a.arg)
    if args.vararg:
        parts.append("*" + args.vararg.arg)
    elif args.kwonlyargs:
        parts.append("*")
    for a in args.kwonlyargs:
        parts.append(a.arg)
    if args.kwarg:
        parts.append("**" + args.kwarg.arg)
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    return f"{prefix} {node.name}({', '.join(parts)})"


def extract_function_blocks(path: Path, names: List[str]) -> List[dict]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    tree = ast.parse(text, filename=str(path))

    rows = []
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            found.add(node.name)
            start = int(node.lineno)
            end = int(getattr(node, "end_lineno", node.lineno))
            block = "\n".join(lines[start - 1:end])
            rows.append({
                "function": node.name,
                "signature": node_signature(node),
                "start_line": start,
                "end_line": end,
                "source": block,
            })

    missing = sorted(set(names) - found)
    if missing:
        raise RuntimeError(f"{path}: missing expected functions {missing}")

    rows.sort(key=lambda x: x["start_line"])
    return rows


def extract_keyword_context(path: Path) -> List[dict]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    hits = []
    for i, line in enumerate(lines):
        low = line.lower()
        if any(k in low for k in KEYWORDS):
            start = max(0, i - 2)
            end = min(len(lines), i + 3)
            hits.append({
                "line": i + 1,
                "match": line.strip()[:300],
                "context": " || ".join(
                    f"L{j+1}:{lines[j].strip()[:220]}"
                    for j in range(start, end)
                ),
            })
    return hits


def safe_torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def state_dict_from_checkpoint(obj):
    if isinstance(obj, dict):
        for key in [
            "model_state_dict", "state_dict", "model", "net",
            "best_model_state", "weights",
        ]:
            if key in obj and isinstance(obj[key], dict):
                return obj[key], key
        # Raw state dict heuristic.
        if obj and all(isinstance(k, str) for k in obj.keys()):
            tensor_like = sum(
                1 for v in obj.values()
                if torch.is_tensor(v)
            )
            if tensor_like >= max(1, len(obj) // 2):
                return obj, "RAW_STATE_DICT"
    return None, ""


def checkpoint_audit(root: Path) -> List[dict]:
    rows = []
    iterator = list(CHECKPOINTS.items())
    if tqdm is not None:
        iterator = tqdm(iterator, desc="Inspect SegFormer checkpoints", unit="ckpt", dynamic_ncols=True)

    for seed, (relp, expected_sha) in iterator:
        p = root / relp
        if not p.exists():
            raise FileNotFoundError(p)

        actual_sha = sha256_file(p)
        if actual_sha != expected_sha:
            raise RuntimeError(
                f"CHECKPOINT_SHA_MISMATCH seed={seed}\n"
                f"expected={expected_sha}\nactual={actual_sha}\npath={p}"
            )

        obj = safe_torch_load(p)
        top_type = type(obj).__name__
        top_keys = list(obj.keys())[:100] if isinstance(obj, dict) else []

        sd, sd_key = state_dict_from_checkpoint(obj)
        if sd is None:
            raise RuntimeError(f"Could not identify state_dict in {p}")

        sample_keys = []
        tensor_shapes = []
        for k, v in list(sd.items())[:200]:
            sample_keys.append(str(k))
            if torch.is_tensor(v):
                tensor_shapes.append(f"{k}:{tuple(v.shape)}:{v.dtype}")

        rows.append({
            "seed": seed,
            "path": rel(p, root),
            "sha256": actual_sha,
            "size_bytes": p.stat().st_size,
            "checkpoint_top_type": top_type,
            "checkpoint_top_keys": "|".join(map(str, top_keys)),
            "state_dict_container": sd_key,
            "state_dict_key_count": len(sd),
            "state_dict_key_examples": " | ".join(sample_keys[:80]),
            "state_dict_shape_examples": " | ".join(tensor_shapes[:80]),
        })

        del obj, sd

    return rows


def npz_header_members(path: Path) -> List[dict]:
    rows = []
    with zipfile.ZipFile(path, "r") as zf:
        for member in zf.namelist():
            if not member.lower().endswith(".npy"):
                continue
            with zf.open(member, "r") as f:
                version = np.lib.format.read_magic(f)
                if version == (1, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
                else:
                    shape, fortran, dtype = np.lib.format.read_array_header_2_0(f)
            rows.append({
                "member": member,
                "shape": list(shape),
                "dtype": str(dtype),
            })
    return rows


def polypgen_segformer_lock_audit(root: Path) -> List[dict]:
    rows = []

    for seed in [20260820, 20260821, 20260822]:
        patterns = [
            f"*segformer*{seed}*.npz",
            f"*segformer_b0*{seed}*.npz",
            f"*segformer*{seed}*source*a1*.npz",
        ]

        candidates = []
        for pat in patterns:
            candidates.extend(POLYPGEN_STATE_DIR.joinpath().parent.glob(pat))

        # Correct path root.
        base = root / POLYPGEN_STATE_DIR
        candidates = []
        for pat in patterns:
            candidates.extend(base.glob(pat))

        candidates = sorted(set(candidates))
        if len(candidates) != 1:
            # Fallback to any NPZ whose name contains seed + segformer.
            candidates = sorted(
                p for p in base.glob("*.npz")
                if str(seed) in p.name and "segformer" in p.name.lower()
            )

        if len(candidates) != 1:
            raise RuntimeError(
                f"Expected exactly one PolypGen SegFormer pre-GT NPZ for seed={seed}, "
                f"found={len(candidates)}: {[str(x) for x in candidates]}"
            )

        p = candidates[0]
        members = npz_header_members(p)
        member_names = [m["member"] for m in members]

        source_like = [
            m for m in members
            if "source" in m["member"].lower()
            and ("mask" in m["member"].lower() or "packed" in m["member"].lower())
        ]
        if not source_like:
            raise RuntimeError(
                f"No frozen SOURCE hard-mask member found in {p}; members={member_names}"
            )

        index_csvs = sorted(
            q for q in base.glob("*.csv")
            if str(seed) in q.name and "segformer" in q.name.lower()
        )

        rows.append({
            "seed": seed,
            "npz_path": rel(p, root),
            "npz_sha256": sha256_file(p),
            "npz_members": json.dumps(members, ensure_ascii=False),
            "source_hard_mask_members": "|".join(m["member"] for m in source_like),
            "index_csv_count": len(index_csvs),
            "index_csv_paths": " | ".join(rel(q, root) for q in index_csvs),
        })

    return rows


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for r in rows:
        for k in r:
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

    print("===== R17A POLYPGEN SEGF INFERENCE CONTRACT AUDIT =====")
    print("Version:", VERSION)
    print("Root:", root)
    print("Training: NO")
    print("Inference: NO")
    print("Target GT loading: NO")
    print("HARM/outcome loading: NO")
    print("Deletion/move: NO")
    print()

    script_rows = []
    function_rows = []
    keyword_rows = []

    print("[1/4] Verify and extract exact historical SegFormer code...")
    for relp in [R03, R05, R10]:
        p = root / relp
        if not p.exists():
            raise FileNotFoundError(p)

        actual_sha = sha256_file(p)
        key = str(relp).replace("\\", "/")
        expected = EXPECTED_SCRIPT_SHA.get(key, "")

        if expected and actual_sha != expected:
            raise RuntimeError(
                f"SCRIPT_SHA_MISMATCH\npath={p}\nexpected={expected}\nactual={actual_sha}"
            )

        script_rows.append({
            "path": key,
            "sha256": actual_sha,
            "expected_sha256": expected,
            "sha_match": (not expected) or actual_sha == expected,
        })

        if key in TARGET_FUNCTIONS:
            blocks = extract_function_blocks(p, TARGET_FUNCTIONS[key])
            for b in blocks:
                function_rows.append({
                    "path": key,
                    **b,
                })

        for hit in extract_keyword_context(p):
            keyword_rows.append({
                "path": key,
                **hit,
            })

    print("[2/4] Inspect frozen SegFormer checkpoints...")
    ckpt_rows = checkpoint_audit(root)

    print("[3/4] Inspect PolypGen SegFormer frozen hard-prediction locks...")
    lock_rows = polypgen_segformer_lock_audit(root)

    print("[4/4] Write exact contract report...")
    write_csv(out_dir / "R17A_SEGF_SCRIPT_SHA_AUDIT.csv", script_rows)
    write_csv(out_dir / "R17A_SEGF_FUNCTION_CONTRACT.csv", function_rows)
    write_csv(out_dir / "R17A_SEGF_CODE_KEYWORD_CONTEXT.csv", keyword_rows)
    write_csv(out_dir / "R17A_SEGF_CHECKPOINT_AUDIT.csv", ckpt_rows)
    write_csv(out_dir / "R17A_SEGF_POLYPGEN_HARD_LOCK_AUDIT.csv", lock_rows)

    report_lines = [
        "===== R17A POLYPGEN SEGF INFERENCE CONTRACT REPORT =====",
        f"Version: {VERSION}",
        f"Root: {root}",
        "",
        "INFORMATION BOUNDARY:",
        "Training: NO",
        "Inference: NO",
        "Target GT loading: NO",
        "HARM/outcome loading: NO",
        "",
        "SCRIPT SHA:",
    ]

    for r in script_rows:
        report_lines.append(
            f"{r['path']} | sha256={r['sha256']} | "
            f"expected={r['expected_sha256'] or 'UNLOCKED_IN_THIS_AUDIT'} | "
            f"match={r['sha_match']}"
        )

    report_lines += ["", "EXACT FUNCTION CONTRACTS:"]
    for r in function_rows:
        report_lines.append(
            f"\n--- {r['path']} :: {r['signature']} "
            f"[L{r['start_line']}-L{r['end_line']}] ---\n{r['source']}"
        )

    report_lines += ["", "CHECKPOINTS:"]
    for r in ckpt_rows:
        report_lines.append(
            f"seed={r['seed']} path={r['path']} sha256={r['sha256']} "
            f"container={r['state_dict_container']} keys={r['state_dict_key_count']}"
        )
        report_lines.append(
            f"  TOP_KEYS={r['checkpoint_top_keys'][:1200]}"
        )
        report_lines.append(
            f"  STATE_KEYS={r['state_dict_key_examples'][:3500]}"
        )

    report_lines += ["", "POLYPGEN FROZEN HARD LOCKS:"]
    for r in lock_rows:
        report_lines.append(
            f"seed={r['seed']} npz={r['npz_path']} sha256={r['npz_sha256']}"
        )
        report_lines.append(
            f"  source_hard_mask_members={r['source_hard_mask_members']}"
        )
        report_lines.append(
            f"  index_csvs={r['index_csv_paths']}"
        )
        report_lines.append(
            f"  npz_members={r['npz_members'][:3500]}"
        )

    report_lines += [
        "",
        "IMPLEMENTATION RULE FOR NEXT RUNNER:",
        "- Reuse the exact extracted SegFormer helpers; do not hand-reimplement preprocessing.",
        "- Load only the three SHA-locked SOURCE checkpoints above.",
        "- Use frozen PolypGen no-GT manifest/image paths only.",
        "- Before accepting any soft logits, threshold the re-inferred SOURCE prediction",
        "  using the historical code's exact rule and require parity with the frozen",
        "  pre-GT SegFormer SOURCE hard-mask lock.",
        "- If parity fails, STOP and do not write CCD scores.",
        "- After parity passes, compute CCD from the same pre-threshold SOURCE logits,",
        "  serialize the 3-state score table, and SHA-lock it before any HARM join.",
        "",
        "GATE=PASS_R17A_POLYPGEN_SEGF_INFERENCE_CONTRACT_AUDIT",
        "NEXT=BUILD_BITEXACT_VALIDATED_SEGFOMER_POLYPGEN_CCD_RUNNER",
    ]

    report = out_dir / "R17A_POLYPGEN_SEGF_INFERENCE_CONTRACT_REPORT.txt"
    report.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print("GATE=PASS_R17A_POLYPGEN_SEGF_INFERENCE_CONTRACT_AUDIT")
    print("NEXT=BUILD_BITEXACT_VALIDATED_SEGFOMER_POLYPGEN_CCD_RUNNER")
    print("Report:", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
