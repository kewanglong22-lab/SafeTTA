#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q1_R17A_CCD_blocked_mapping_resolver_v1_fix1.py

SafeTTA R17A — resolve the three CCD datasets that were falsely/ambiguously
blocked by the previous dataset-mapping audit.

IMPORTANT FIX
-------------
The previous audit's GT-path guard was too broad: it tokenized path names and
treated "no_label" and "gt_free" as if they implied GT usage. In this project
those names mean the opposite (pre-GT / label-free assets). This fix therefore:

  - explicitly ALLOWS: no_label, no-label, gt_free, gt-free, preGT, pre_gt
  - excludes only explicit post-GT / outcome assets such as:
      gt_reveal, ground_truth, harm, delta_dice, policy_evaluation, outcome
  - never opens target GT files.

TARGET DATASETS ONLY
--------------------
  PolypGen
  SUNSEG
  PROMISE12

READ-ONLY
---------
  Training: NO
  Inference: NO
  Target GT loading: NO
  Deletion/move: NO

What it resolves
----------------
1) Re-inspects pre-GT NPY/NPZ assets, including `locked_logits.npz`.
2) Reads NPZ member headers only; large arrays are not materialized.
3) Extracts checkpoint paths/SHA256 claims from current lock JSON/CSV manifests.
4) Resolves those checkpoint claims against existing SOURCE checkpoints.
5) Confirms image-only/sample manifests and retained inference code.
6) Emits one dataset decision:
      DIRECT_SOFT_ASSET_READY
      GT_FREE_SOURCE_REINFERENCE_READY
      BLOCKED_AMBIGUOUS_MAPPING

Output:
  F:\\MEDSEG_SAFETTA\\outputs\\Q1_R17A_CCD_blocked_mapping_resolver_v1_fix1
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterable

import numpy as np

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

VERSION = "2026-09-08-Q1-R17A-CCD-BLOCKED-MAPPING-RESOLVER-v1-fix1"
DEFAULT_ROOT = Path(r"F:\MEDSEG_SAFETTA")
OUT_REL = Path(r"outputs\Q1_R17A_CCD_blocked_mapping_resolver_v1_fix1")

TARGETS = ("PolypGen", "SUNSEG", "PROMISE12")

SEARCH_ROOTS = [
    Path("outputs"),
    Path("code"),
    Path("release/SafeTTA_clean_test_v102"),
    Path("release/public_v1_fix4"),
    Path("release/SafeTTA_repro_release_candidate_v1"),
    Path("data"),
    Path("cross_modality_prostate_mri"),
]

CODE_ROOTS = [
    Path("code"),
    Path("release/SafeTTA_clean_test_v102/experiments"),
    Path("release/SafeTTA_clean_test_v102/method_core"),
]

ALIASES = {
    "PolypGen": [
        "polypgen",
        "s06_c_polypgen",
        "q1_r10l3",
    ],
    "SUNSEG": [
        "sunseg", "sun_seg", "sun-seg",
        "q1_r14b", "q1_r14c",
    ],
    "PROMISE12": [
        "promise12",
        "q1x_cm5", "q1x_cm6", "q1x_cm7",
    ],
}

# Strong priority: current/final pre-GT routes first.
PREFERRED = {
    "PolypGen": [
        "s06_c_polypgen_dual_backbone_no_label_lock_v1_fix1",
        "q1_r10l3a_polypgen_source_a1_prediction_lock_fix2",
        "q1_r10l3a_polypgen_source_a1_prediction_lock",
        "q1_r10l3b_polypgen_frozen_safety_score_lock",
    ],
    "SUNSEG": [
        "q1_r14c1_sunseg_source_tent1_pl_prediction_lock_pregt_fix3",
        "q1_r14c2b_sunseg_frozen_safety_score_lock_pregt",
    ],
    "PROMISE12": [
        "q1x_cm5a_promise12_gt_free_prediction_tent1_safety_scoring_lock_fix1",
        "q1x_cm2a_prostate158_official_manifest_t2_resolver_and_slice_preflight_fix3",
        "q1x_cm4d_final_all_source_segmentation_models_lock_fix2",
    ],
}

ARRAY_SUFFIXES = {".npy", ".npz"}
CKPT_SUFFIXES = {".pt", ".pth", ".ckpt"}
TEXT_SUFFIXES = {".json", ".csv", ".txt", ".md", ".py", ".yaml", ".yml"}

SOFT_TOKENS = (
    "logit", "logits", "prob", "probs", "probability", "softmax"
)
HARD_TOKENS = (
    "packbits", "mask", "masks", "binary_mask"
)

# Explicitly pre-GT / label-free language.
ALLOW_PRE_GT_TOKENS = (
    "no_label", "no-label", "nolabel",
    "gt_free", "gt-free", "gtfree",
    "pregt", "pre_gt", "pre-gt",
    "image_only", "image-only",
)

# Only these are treated as clear post-GT/outcome indicators.
BLOCK_POST_GT_TOKENS = (
    "gt_reveal", "gt-reveal",
    "ground_truth", "ground-truth", "groundtruth",
    "delta_dice",
    "policy_evaluation",
    "harm_label", "harmful",
    "outcome_asset", "outcome_table",
)

# Known SOURCE checkpoint families that can support re-inference.
SOURCE_CKPT_ROUTE_HINTS = (
    "s01_b_pranet_source_only",
    "s05_b_deeplabv3_r50_source_only",
    "q1_r02_segformer_b0_three_seed_source_training",
    "q1x_cm3_source_oof_segmentation_training",
    "q1x_cm4d_final_all_source_segmentation_models",
)

STATE_COLUMNS = (
    "model_family", "model_state_id", "training_seed"
)
CHECKPOINT_PATH_COLUMNS = (
    "checkpoint", "checkpoint_path", "checkpoint_file", "model_path"
)
CHECKPOINT_SHA_COLUMNS = (
    "checkpoint_sha256", "checkpoint_sha", "model_sha256"
)


def fmt_bytes(n: int) -> str:
    x = float(n)
    for u in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if x < 1024.0 or u == "TiB":
            return f"{x:.2f} {u}"
        x /= 1024.0
    return f"{n} B"


def norm(p: Path, root: Path) -> str:
    return str(p.relative_to(root)).replace("\\", "/")


def dataset_tags(rel: str) -> List[str]:
    low = rel.lower()
    out = []
    for ds, aliases in ALIASES.items():
        if any(a in low for a in aliases):
            out.append(ds)
    return sorted(set(out))


def priority(ds: str, rel: str) -> int:
    low = rel.lower()
    for i, tok in enumerate(PREFERRED.get(ds, [])):
        if tok in low:
            return 100 - i
    return 0


def is_post_gt_path(rel: str) -> bool:
    low = rel.lower()

    # Critical fix: explicit pre-GT language overrides generic appearances
    # of the substrings "gt" or "label".
    if any(tok in low for tok in ALLOW_PRE_GT_TOKENS):
        # Still block explicit GT-reveal/outcome terms if both occur.
        return any(tok in low for tok in BLOCK_POST_GT_TOKENS)

    return any(tok in low for tok in BLOCK_POST_GT_TOKENS)


def read_npy_header_stream(f) -> Dict[str, object]:
    try:
        version = np.lib.format.read_magic(f)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
        else:
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(f)
        return {
            "shape": list(shape),
            "dtype": str(dtype),
            "fortran_order": bool(fortran),
            "header_status": "OK",
        }
    except Exception as e:
        return {
            "shape": "",
            "dtype": "",
            "fortran_order": "",
            "header_status": f"ERROR:{type(e).__name__}:{e}",
        }


def inspect_array(path: Path) -> List[dict]:
    if path.suffix.lower() == ".npy":
        try:
            with path.open("rb") as f:
                meta = read_npy_header_stream(f)
        except Exception as e:
            meta = {
                "shape": "", "dtype": "", "fortran_order": "",
                "header_status": f"ERROR:{type(e).__name__}:{e}",
            }
        return [{"member": "", **meta}]

    rows = []
    try:
        with zipfile.ZipFile(path, "r") as zf:
            for member in zf.namelist():
                if not member.lower().endswith(".npy"):
                    continue
                with zf.open(member, "r") as f:
                    meta = read_npy_header_stream(f)
                rows.append({"member": member, **meta})
    except Exception as e:
        rows.append({
            "member": "",
            "shape": "",
            "dtype": "",
            "fortran_order": "",
            "header_status": f"ERROR:{type(e).__name__}:{e}",
        })
    return rows


def signal_class(path_rel: str, member: str) -> str:
    text = f"{path_rel} {member}".lower()
    if any(tok in text for tok in SOFT_TOKENS):
        return "SOFT_LOGIT_OR_PROB"
    if any(tok in text for tok in HARD_TOKENS):
        return "HARD_MASK"
    return "UNKNOWN_ARRAY"


def scan_pre_gt_arrays(root: Path) -> List[dict]:
    files = []
    seen = set()

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
                if rel in seen:
                    continue
                seen.add(rel)

                tags = dataset_tags(rel)
                if not tags:
                    continue
                if is_post_gt_path(rel):
                    continue

                try:
                    size = int(p.stat().st_size)
                except OSError:
                    size = 0
                files.append((p, rel, tags, size))

    rows = []
    iterator = sorted(files, key=lambda x: -x[3])
    if tqdm is not None:
        iterator = tqdm(
            iterator,
            desc="Reinspect pre-GT NPY/NPZ assets",
            unit="file",
            dynamic_ncols=True,
        )

    for p, rel, tags, size in iterator:
        metas = inspect_array(p)
        for meta in metas:
            cls = signal_class(rel, str(meta["member"]))
            for ds in tags:
                rows.append({
                    "dataset": ds,
                    "path": rel,
                    "member": meta["member"],
                    "size_bytes": size,
                    "size_human": fmt_bytes(size),
                    "shape": json.dumps(meta["shape"]) if isinstance(meta["shape"], list) else meta["shape"],
                    "dtype": meta["dtype"],
                    "header_status": meta["header_status"],
                    "signal_class": cls,
                    "priority": priority(ds, rel),
                })
    return rows


def recursive_collect_checkpoint_claims(obj, source_path: str, ds: str) -> List[dict]:
    claims = []

    def walk(x, prefix=""):
        if isinstance(x, dict):
            # Capture path/SHA pairs when present in the same dictionary.
            path_val = ""
            sha_val = ""
            family = str(x.get("model_family", "") or x.get("family", ""))
            state = str(x.get("model_state_id", "") or x.get("state_id", ""))
            seed = str(x.get("training_seed", "") or x.get("seed", ""))

            for key in CHECKPOINT_PATH_COLUMNS:
                if key in x and isinstance(x[key], str):
                    path_val = x[key]
                    break
            for key in CHECKPOINT_SHA_COLUMNS:
                if key in x and isinstance(x[key], str):
                    sha_val = x[key]
                    break

            if path_val or sha_val:
                claims.append({
                    "dataset": ds,
                    "source_manifest": source_path,
                    "model_family": family,
                    "model_state_id": state,
                    "training_seed": seed,
                    "checkpoint_claim": path_val,
                    "checkpoint_sha256_claim": sha_val,
                })

            for k, v in x.items():
                walk(v, f"{prefix}.{k}" if prefix else k)

        elif isinstance(x, list):
            for i, v in enumerate(x):
                walk(v, f"{prefix}[{i}]")

    walk(obj)
    return claims


def scan_checkpoint_claims(root: Path) -> List[dict]:
    rows = []
    seen = set()

    for rel_root in [Path("outputs"), Path("release/SafeTTA_clean_test_v102"), Path("release/public_v1_fix4")]:
        base = root / rel_root
        if not base.exists():
            continue

        for dirpath, dirnames, filenames in os.walk(base, followlinks=False):
            dp = Path(dirpath)
            dirnames[:] = [
                d for d in dirnames
                if d not in {".git", "__pycache__"}
                and not (dp / d).is_symlink()
            ]

            for name in filenames:
                p = dp / name
                if p.suffix.lower() not in {".json", ".csv"}:
                    continue
                rel = norm(p, root)
                if rel in seen:
                    continue
                seen.add(rel)

                tags = dataset_tags(rel)
                if not tags:
                    continue
                if is_post_gt_path(rel):
                    continue

                try:
                    if p.stat().st_size > 30 * 1024 * 1024:
                        continue
                except OSError:
                    continue

                if p.suffix.lower() == ".json":
                    try:
                        obj = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
                    except Exception:
                        continue
                    for ds in tags:
                        rows.extend(recursive_collect_checkpoint_claims(obj, rel, ds))

                elif p.suffix.lower() == ".csv":
                    try:
                        with p.open("r", newline="", encoding="utf-8-sig", errors="ignore") as f:
                            reader = csv.DictReader(f)
                            fields = reader.fieldnames or []
                            if not any(c in fields for c in CHECKPOINT_PATH_COLUMNS + CHECKPOINT_SHA_COLUMNS):
                                continue

                            for r in reader:
                                path_val = next((r.get(c, "") for c in CHECKPOINT_PATH_COLUMNS if r.get(c)), "")
                                sha_val = next((r.get(c, "") for c in CHECKPOINT_SHA_COLUMNS if r.get(c)), "")
                                family = r.get("model_family", "") or r.get("family", "")
                                state = r.get("model_state_id", "") or r.get("state_id", "")
                                seed = r.get("training_seed", "") or r.get("seed", "")
                                for ds in tags:
                                    rows.append({
                                        "dataset": ds,
                                        "source_manifest": rel,
                                        "model_family": family,
                                        "model_state_id": state,
                                        "training_seed": seed,
                                        "checkpoint_claim": path_val,
                                        "checkpoint_sha256_claim": sha_val,
                                    })
                    except Exception:
                        continue

    # De-duplicate exact claims.
    unique = []
    seen_key = set()
    for r in rows:
        key = tuple(str(r.get(k, "")) for k in [
            "dataset", "source_manifest", "model_family",
            "model_state_id", "training_seed",
            "checkpoint_claim", "checkpoint_sha256_claim",
        ])
        if key not in seen_key:
            seen_key.add(key)
            unique.append(r)
    return unique


def enumerate_source_checkpoints(root: Path) -> List[dict]:
    rows = []
    for base in [root / "outputs", root / "release/SafeTTA_repro_release_candidate_v1"]:
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if not p.is_file() or p.suffix.lower() not in CKPT_SUFFIXES:
                continue
            rel = norm(p, root)
            low = rel.lower()
            if not any(tok in low for tok in SOURCE_CKPT_ROUTE_HINTS):
                continue
            try:
                size = int(p.stat().st_size)
            except OSError:
                continue
            rows.append({
                "path_obj": p,
                "path": rel,
                "basename": p.name,
                "size_bytes": size,
                "size_human": fmt_bytes(size),
                "sha256": "",  # lazy
            })
    return rows


def resolve_claim_path(root: Path, claim: str) -> Optional[Path]:
    if not claim:
        return None

    raw = claim.strip().strip('"').strip("'")
    p = Path(raw)

    candidates = []
    if p.is_absolute():
        candidates.append(p)
    candidates += [
        root / raw,
        root / raw.replace("/", "\\"),
    ]

    # Some manifests store paths relative to outputs or current working dir.
    candidates += [
        root / "outputs" / raw,
        root / "code" / raw,
    ]

    for c in candidates:
        try:
            if c.exists() and c.is_file():
                return c
        except OSError:
            pass
    return None


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_checkpoint_claims(root: Path, claims: List[dict], inventory: List[dict]) -> List[dict]:
    by_basename = defaultdict(list)
    for r in inventory:
        by_basename[r["basename"]].append(r)

    sha_cache = {}

    def get_sha(path: Path) -> str:
        key = str(path)
        if key not in sha_cache:
            sha_cache[key] = sha256(path)
        return sha_cache[key]

    rows = []
    iterator = claims
    if tqdm is not None:
        iterator = tqdm(
            claims,
            desc="Resolve frozen checkpoint claims",
            unit="claim",
            dynamic_ncols=True,
        )

    for r in iterator:
        claim = str(r.get("checkpoint_claim", "") or "")
        sha_claim = str(r.get("checkpoint_sha256_claim", "") or "").lower()

        resolved = resolve_claim_path(root, claim)
        method = "EXACT_PATH" if resolved else ""

        # If exact path is stale, try basename within known SOURCE checkpoint inventory.
        if resolved is None and claim:
            name = Path(claim.replace("\\", "/")).name
            matches = by_basename.get(name, [])
            if len(matches) == 1:
                resolved = matches[0]["path_obj"]
                method = "UNIQUE_BASENAME"
            elif len(matches) > 1 and sha_claim:
                for item in matches:
                    if get_sha(item["path_obj"]).lower() == sha_claim:
                        resolved = item["path_obj"]
                        method = "BASENAME_PLUS_SHA256"
                        break

        # If we have only SHA, resolve against targeted source-checkpoint inventory.
        if resolved is None and sha_claim:
            for item in inventory:
                if get_sha(item["path_obj"]).lower() == sha_claim:
                    resolved = item["path_obj"]
                    method = "SHA256_MATCH"
                    break

        resolved_sha = ""
        sha_match = ""
        if resolved is not None:
            resolved_sha = get_sha(resolved)
            if sha_claim:
                sha_match = (resolved_sha.lower() == sha_claim)
            else:
                sha_match = "NO_SHA_CLAIM"

        rows.append({
            **r,
            "resolved_path": norm(resolved, root) if resolved is not None else "",
            "resolution_method": method,
            "resolved_sha256": resolved_sha,
            "sha256_match": sha_match,
            "resolved": resolved is not None and (sha_match is True or sha_match == "NO_SHA_CLAIM"),
        })
    return rows


def scan_image_manifests(root: Path) -> List[dict]:
    rows = []

    for rel_root in [Path("outputs"), Path("data"), Path("cross_modality_prostate_mri")]:
        base = root / rel_root
        if not base.exists():
            continue

        for p in base.rglob("*.csv"):
            rel = norm(p, root)
            tags = dataset_tags(rel)
            if not tags or is_post_gt_path(rel):
                continue
            try:
                if p.stat().st_size > 30 * 1024 * 1024:
                    continue
            except OSError:
                continue

            try:
                with p.open("r", newline="", encoding="utf-8-sig", errors="ignore") as f:
                    reader = csv.reader(f)
                    header = next(reader, [])
                    header_low = [x.lower() for x in header]
                    # Image-only/alignment signal.
                    if not any(x in header_low for x in [
                        "sample_id", "image_path", "image_raw",
                        "image_mhd", "frame_member", "case_id",
                    ]):
                        continue
                    n = sum(1 for _ in reader)
            except Exception:
                continue

            for ds in tags:
                rows.append({
                    "dataset": ds,
                    "path": rel,
                    "rows": n,
                    "columns": "|".join(header[:80]),
                    "priority": priority(ds, rel),
                })

    return rows


def scan_inference_code(root: Path) -> List[dict]:
    rows = []
    SIGNALS = (
        "load_state_dict", "torch.sigmoid", "sigmoid(",
        "torch.softmax", "softmax(", "model(",
        "source_logits", "source_prob", "source_prediction",
    )

    for rel_root in CODE_ROOTS:
        base = root / rel_root
        if not base.exists():
            continue

        for p in base.rglob("*.py"):
            rel = norm(p, root)
            tags = dataset_tags(rel)
            if not tags:
                continue
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            lines = text.splitlines()
            hits = []
            for i, line in enumerate(lines):
                low = line.lower()
                if any(sig.lower() in low for sig in SIGNALS):
                    start = max(0, i - 2)
                    end = min(len(lines), i + 3)
                    hits.append(" || ".join(
                        f"L{j+1}:{lines[j].strip()[:160]}"
                        for j in range(start, end)
                    ))
            if hits:
                for ds in tags:
                    rows.append({
                        "dataset": ds,
                        "path": rel,
                        "hit_count": len(hits),
                        "examples": " ### ".join(hits[:8]),
                        "priority": priority(ds, rel),
                    })

    return rows


def exact_state_key(row: dict) -> Tuple[str, str, str]:
    return (
        str(row.get("model_family", "") or ""),
        str(row.get("model_state_id", "") or ""),
        str(row.get("training_seed", "") or ""),
    )


def build_decisions(
    arrays: List[dict],
    resolved_claims: List[dict],
    manifests: List[dict],
    code: List[dict],
) -> Dict[str, dict]:
    decisions = {}

    for ds in TARGETS:
        ds_soft = [
            r for r in arrays
            if r["dataset"] == ds
            and r["signal_class"] == "SOFT_LOGIT_OR_PROB"
            and r["header_status"] == "OK"
        ]
        # Group soft candidate members by actual file path.
        soft_files = sorted(set(r["path"] for r in ds_soft))

        ds_claims = [r for r in resolved_claims if r["dataset"] == ds]
        resolved = [r for r in ds_claims if bool(r["resolved"])]

        # Unique model-state claims with nonempty identity.
        state_claims = {}
        for r in ds_claims:
            key = exact_state_key(r)
            if any(key):
                state_claims[key] = r

        state_resolved = {}
        for r in resolved:
            key = exact_state_key(r)
            if any(key):
                state_resolved[key] = r

        ds_man = [r for r in manifests if r["dataset"] == ds]
        ds_code = [r for r in code if r["dataset"] == ds]

        if soft_files and ds_man:
            decision = "DIRECT_SOFT_ASSET_READY"
            reason = (
                "pre-GT soft/logit file(s) found after corrected no_label/gt_free "
                "handling, with image/sample manifests available"
            )
        elif state_claims and len(state_resolved) == len(state_claims) and ds_man and ds_code:
            decision = "GT_FREE_SOURCE_REINFERENCE_READY"
            reason = (
                "all discovered frozen model-state checkpoint claims resolve, "
                "and image-only/sample manifests plus retained inference code exist"
            )
        elif resolved and ds_man and ds_code:
            decision = "GT_FREE_SOURCE_REINFERENCE_READY"
            reason = (
                "frozen checkpoint mappings, manifests, and retained inference code "
                "exist; exact state count will be frozen by the score-lock script"
            )
        else:
            decision = "BLOCKED_AMBIGUOUS_MAPPING"
            reason = (
                "insufficient direct soft assets and/or unresolved checkpoint/manifests/code"
            )

        decisions[ds] = {
            "decision": decision,
            "reason": reason,
            "soft_file_count": len(soft_files),
            "soft_files": [
                {
                    "path": path,
                    "members": [
                        {
                            "member": r["member"],
                            "shape": r["shape"],
                            "dtype": r["dtype"],
                        }
                        for r in ds_soft if r["path"] == path
                    ][:20],
                }
                for path in sorted(
                    soft_files,
                    key=lambda p: -max(
                        r["priority"] for r in ds_soft if r["path"] == p
                    )
                )[:20]
            ],
            "checkpoint_claim_count": len(ds_claims),
            "resolved_checkpoint_claim_count": len(resolved),
            "unique_state_claim_count": len(state_claims),
            "resolved_unique_state_count": len(state_resolved),
            "image_manifest_count": len(ds_man),
            "inference_code_count": len(ds_code),
            "top_resolved_checkpoints": [
                {
                    "model_family": r.get("model_family", ""),
                    "model_state_id": r.get("model_state_id", ""),
                    "training_seed": r.get("training_seed", ""),
                    "resolved_path": r.get("resolved_path", ""),
                    "resolution_method": r.get("resolution_method", ""),
                    "sha256_match": r.get("sha256_match", ""),
                }
                for r in resolved[:20]
            ],
            "top_image_manifests": [
                {
                    "path": r["path"],
                    "rows": r["rows"],
                    "columns": r["columns"],
                }
                for r in sorted(ds_man, key=lambda x: (-x["priority"], x["path"]))[:12]
            ],
            "top_inference_code": [
                r["path"]
                for r in sorted(ds_code, key=lambda x: (-x["priority"], x["path"]))[:12]
            ],
        }

    return decisions


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

    print("===== R17A CCD BLOCKED-MAPPING RESOLVER v1-fix1 =====")
    print("Version:", VERSION)
    print("Root:", root)
    print("Targets:", ", ".join(TARGETS))
    print("Read-only: YES")
    print("Training: NO")
    print("Inference: NO")
    print("Target GT loading: NO")
    print("Deletion/move: NO")
    print()
    print("Audit fix:")
    print("  no_label / gt_free / preGT are treated as PRE-GT, not GT.")
    print()

    print("[1/6] Reinspect pre-GT soft/logit assets...")
    arrays = scan_pre_gt_arrays(root)

    print("[2/6] Extract frozen checkpoint claims from lock manifests...")
    claims = scan_checkpoint_claims(root)

    print("[3/6] Inventory relevant SOURCE checkpoints...")
    inventory = enumerate_source_checkpoints(root)
    print("Relevant SOURCE checkpoint files:", len(inventory))

    print("[4/6] Resolve checkpoint paths/SHA256...")
    resolved = resolve_checkpoint_claims(root, claims, inventory)

    print("[5/6] Audit image-only/sample manifests and inference code...")
    manifests = scan_image_manifests(root)
    code = scan_inference_code(root)

    print("[6/6] Build final mapping decisions...")
    decisions = build_decisions(arrays, resolved, manifests, code)

    write_csv(out_dir / "R17A_BLOCKED_PREGT_ARRAY_AUDIT.csv", arrays)
    write_csv(out_dir / "R17A_BLOCKED_CHECKPOINT_CLAIMS.csv", claims)
    write_csv(
        out_dir / "R17A_BLOCKED_CHECKPOINT_RESOLUTION.csv",
        [
            {k: v for k, v in r.items() if k != "path_obj"}
            for r in resolved
        ],
    )
    write_csv(
        out_dir / "R17A_BLOCKED_SOURCE_CHECKPOINT_INVENTORY.csv",
        [
            {k: v for k, v in r.items() if k != "path_obj"}
            for r in inventory
        ],
    )
    write_csv(out_dir / "R17A_BLOCKED_IMAGE_MANIFEST_AUDIT.csv", manifests)
    write_csv(out_dir / "R17A_BLOCKED_INFERENCE_CODE_AUDIT.csv", code)

    (out_dir / "R17A_BLOCKED_MAPPING_DECISIONS.json").write_text(
        json.dumps(decisions, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lines = [
        "===== R17A CCD BLOCKED-MAPPING RESOLVER REPORT =====",
        f"Version: {VERSION}",
        f"Root: {root}",
        "",
        "FIX APPLIED:",
        "- no_label / gt_free / preGT are treated as pre-GT assets.",
        "- only explicit gt_reveal / ground_truth / harm / outcome indicators are excluded.",
        "- target GT was not opened.",
        "",
        "DATASET DECISIONS:",
    ]

    for ds in TARGETS:
        r = decisions[ds]
        lines.append(
            f"{ds}: {r['decision']} | "
            f"soft_files={r['soft_file_count']} "
            f"ckpt_claims={r['checkpoint_claim_count']} "
            f"resolved_ckpt={r['resolved_checkpoint_claim_count']} "
            f"unique_states={r['unique_state_claim_count']} "
            f"resolved_states={r['resolved_unique_state_count']} "
            f"image_manifests={r['image_manifest_count']} "
            f"inference_code={r['inference_code_count']}"
        )
        lines.append(f"  REASON: {r['reason']}")
        for sf in r["soft_files"][:8]:
            lines.append(f"  SOFT: {sf['path']}")
            for m in sf["members"][:8]:
                lines.append(
                    f"    member={m['member']} shape={m['shape']} dtype={m['dtype']}"
                )
        for ck in r["top_resolved_checkpoints"][:10]:
            lines.append(
                "  CKPT: "
                f"family={ck['model_family']} state={ck['model_state_id']} "
                f"seed={ck['training_seed']} -> {ck['resolved_path']} "
                f"via={ck['resolution_method']} sha_match={ck['sha256_match']}"
            )
        for man in r["top_image_manifests"][:5]:
            lines.append(
                f"  MANIFEST: {man['path']} :: rows={man['rows']} "
                f"cols={man['columns'][:300]}"
            )
        for cp in r["top_inference_code"][:5]:
            lines.append(f"  CODE: {cp}")

    blocked = [
        ds for ds in TARGETS
        if decisions[ds]["decision"] == "BLOCKED_AMBIGUOUS_MAPPING"
    ]

    lines += [
        "",
        "IMPORTANT:",
        "- DIRECT_SOFT_ASSET_READY means the score-lock script may read only the frozen soft/logit asset plus its sample manifest.",
        "- GT_FREE_SOURCE_REINFERENCE_READY means the score-lock script may run only frozen SOURCE inference and must serialize/hash CCD before joining HARM labels.",
        "- Hard masks are never used to approximate SicTTA CCD.",
        "- No target GT was opened in this resolver.",
        "",
    ]

    if blocked:
        lines.append(
            "GATE=BLOCKED_R17A_CCD_MAPPING_FOR_" + "_".join(blocked)
        )
        lines.append(
            "NEXT=RESOLVE_ONLY_REMAINING_BLOCKED_DATASETS"
        )
    else:
        lines.append("GATE=PASS_R17A_CCD_BLOCKED_MAPPING_RESOLUTION")
        lines.append(
            "NEXT=BUILD_DATASET_SPECIFIC_PRE_GT_CCD_SCORE_LOCK_RUNNER"
        )

    report = out_dir / "R17A_CCD_BLOCKED_MAPPING_RESOLVER_REPORT.txt"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print()
    for line in lines[-10:]:
        print(line)
    print("Report:", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
