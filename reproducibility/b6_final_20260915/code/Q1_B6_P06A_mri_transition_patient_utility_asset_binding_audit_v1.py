#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SafeTTA B6-P0-6A — MRI transition patient-utility asset binding audit.

READ ONLY / NO SCIENTIFIC METRICS.

This stage does NOT:
- load segmentation models;
- run inference or TTA;
- read/decode GT pixels;
- compute Dice, DeltaDice, HARM, BENEFIT, AUROC/AUPRC, or utility;
- fit/refit/calibrate/select score direction;
- choose coverage based on outcomes.

It only discovers and binds the exact already-frozen FinalB/B6/CM assets needed
for the subsequent PROMISE12 patient-level 3D deployment-utility audit.

Success routes:
DIRECT_COUNTS:
    a frozen target outcome table already contains patient/slice identity,
    family, action, SOURCE TP/FP/FN, candidate TP/FP/FN, and needed risk scores.

RECOUNT_FROM_FROZEN_MASKS:
    exact frozen target SOURCE/candidate masks + PROMISE12 GT assets can be bound,
    allowing a later read-only post-reveal recount of TP/FP/FN.

STOP:
    required lineage cannot be established. Do not improvise or rerun actions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Iterable

import numpy as np
import pandas as pd


VERSION = "2026-09-15-B6-P06A-v1"

ROOT = Path(r"F:\MEDSEG_SAFETTA")
CODE = ROOT / "code"
OUT = ROOT / "outputs"

PROTOCOL_PATH = CODE / "B6_P06_MRI_TRANSITION_PATIENT_UTILITY_PROTOCOL_v1.json"
EXPECTED_PROTOCOL_SHA256 = "d6e02e54cc4c4e39c6321428b5c9bfacdf33716acfca3424b98ec90adfc23b15"

OUT_DIR = ROOT / "B6_P06A_mri_transition_patient_utility_asset_binding_audit_v1"
PASS_GATE = "PASS_B6_P06A_MRI_TRANSITION_PATIENT_UTILITY_ASSET_BINDING"

# High-value historical roots. Missing roots are reported, not fabricated.
ROOT_PATTERNS = [
    "FinalB*",
    "B6_P01A*",
    "B6_P03*",
]
OUTPUT_PATTERNS = [
    "Q1X_CM5A*",
    "Q1X_CM6*",
    "Q1X_CM7B*",
]

MAX_CSV_BYTES = 50 * 1024 * 1024
MAX_JSON_BYTES = 10 * 1024 * 1024
MAX_NPZ_BYTES = 2 * 1024 * 1024 * 1024

IDENTITY_ALIASES = {
    "patient": ["case_key", "patient_id", "case_id", "patient", "subject_id"],
    "slice": ["slice_index", "global_index", "sample_id", "row_id"],
    "family": ["family", "model_family"],
    "action": ["action", "heldout_action", "target_action", "candidate_action"],
}

SOURCE_COUNT_ALIASES = {
    "tp": ["source_tp", "src_tp"],
    "fp": ["source_fp", "src_fp"],
    "fn": ["source_fn", "src_fn"],
}
CAND_COUNT_ALIASES = {
    "tp": ["candidate_tp", "action_tp", "adapted_tp", "tent_tp", "pl_tp", "memo_tp"],
    "fp": ["candidate_fp", "action_fp", "adapted_fp", "tent_fp", "pl_fp", "memo_fp"],
    "fn": ["candidate_fn", "action_fn", "adapted_fn", "tent_fn", "pl_fn", "memo_fn"],
}

SCORE_GROUPS = {
    "full_transition": [
        "full_transition_risk",
        "full_transition_score",
        "safettta_risk",
        "safetta_risk",
        "harm_risk",
        "risk_score",
    ],
    "source_state": [
        "source_state_risk",
        "source_risk",
        "q66_risk",
        "source_score",
    ],
    "simple_geometry": [
        "simple_geometry_risk",
        "geometry_risk",
        "mask_change_risk",
        "simple_mask_change_risk",
    ],
}

OUTCOME_HINTS = [
    "harm", "benefit", "delta_dice", "source_dice", "candidate_dice",
    "action_dice", "adapted_dice",
]

MASK_ARRAY_HINTS = [
    "source_masks", "source_masks_packed", "candidate_masks", "candidate_masks_packed",
    "tent", "pl", "memo", "gt", "mask",
]


def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def verify_protocol() -> Dict[str, Any]:
    if not PROTOCOL_PATH.is_file():
        raise FileNotFoundError(PROTOCOL_PATH)
    got = sha256_file(PROTOCOL_PATH)
    if got.lower() != EXPECTED_PROTOCOL_SHA256.lower():
        raise RuntimeError(
            "P06 protocol SHA mismatch.\n"
            f"expected={EXPECTED_PROTOCOL_SHA256}\nobserved={got}"
        )
    p = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if p.get("status") != "FROZEN_POST_REVEAL_ADDITIVE_PROTOCOL_BEFORE_NEW_PATIENT_UTILITY_METRICS":
        raise RuntimeError("P06 protocol status changed.")
    return {"path": str(PROTOCOL_PATH), "sha256": got, "status": p["status"]}


def candidate_roots() -> List[Path]:
    roots: List[Path] = []
    for pat in ROOT_PATTERNS:
        roots.extend(sorted(ROOT.glob(pat)))
    if OUT.is_dir():
        for pat in OUTPUT_PATTERNS:
            roots.extend(sorted(OUT.glob(pat)))

    # De-duplicate while preserving sorted deterministic order.
    uniq = []
    seen = set()
    for p in roots:
        try:
            k = str(p.resolve()).lower()
        except Exception:
            k = str(p).lower()
        if k not in seen and p.exists():
            seen.add(k)
            uniq.append(p)
    return uniq


def iter_candidate_files(roots: Iterable[Path]) -> Iterable[Path]:
    allowed = {".csv", ".json", ".npz", ".npy"}
    for root in roots:
        if root.is_file() and root.suffix.lower() in allowed:
            yield root
            continue
        if not root.is_dir():
            continue
        for p in sorted(root.rglob("*")):
            if not p.is_file():
                continue
            if p.suffix.lower() not in allowed:
                continue
            # Skip temporary/cache noise.
            low = p.name.lower()
            if low.endswith(".tmp") or ".tmp." in low:
                continue
            yield p


def find_first(columns_lower: Dict[str, str], aliases: List[str]) -> str | None:
    for a in aliases:
        if a.lower() in columns_lower:
            return columns_lower[a.lower()]
    return None


def all_count_roles(columns: List[str]) -> Dict[str, Any]:
    lower = {c.lower(): c for c in columns}
    src = {k: find_first(lower, v) for k, v in SOURCE_COUNT_ALIASES.items()}
    cand = {k: find_first(lower, v) for k, v in CAND_COUNT_ALIASES.items()}
    return {
        "source": src,
        "candidate": cand,
        "source_complete": all(src.values()),
        "candidate_complete": all(cand.values()),
    }


def identity_roles(columns: List[str]) -> Dict[str, Any]:
    lower = {c.lower(): c for c in columns}
    roles = {k: find_first(lower, v) for k, v in IDENTITY_ALIASES.items()}
    roles["minimum_complete"] = all(
        roles[k] is not None for k in ["patient", "slice", "family"]
    )
    return roles


def score_roles(columns: List[str]) -> Dict[str, Any]:
    lower = {c.lower(): c for c in columns}
    return {
        group: find_first(lower, aliases)
        for group, aliases in SCORE_GROUPS.items()
    }


def inspect_csv(path: Path) -> Dict[str, Any]:
    size = path.stat().st_size
    if size > MAX_CSV_BYTES:
        return {
            "type": "csv",
            "path": str(path),
            "bytes": size,
            "skipped": "too_large",
        }

    df0 = pd.read_csv(path, nrows=0, low_memory=False)
    cols = list(map(str, df0.columns))
    ident = identity_roles(cols)
    counts = all_count_roles(cols)
    scores = score_roles(cols)

    # Read key columns only, if present, to report cardinality without computing outcomes.
    usecols = [x for x in [
        ident.get("patient"), ident.get("slice"), ident.get("family"), ident.get("action")
    ] if x is not None]
    rows = None
    patients = None
    families = None
    actions = None
    if usecols:
        try:
            d = pd.read_csv(path, usecols=sorted(set(usecols)), low_memory=False)
            rows = int(len(d))
            if ident.get("patient"):
                patients = int(d[ident["patient"]].astype(str).nunique())
            if ident.get("family"):
                families = sorted(d[ident["family"]].astype(str).dropna().unique().tolist())[:20]
            if ident.get("action"):
                actions = sorted(d[ident["action"]].astype(str).dropna().unique().tolist())[:20]
        except Exception as e:
            pass

    outcome_cols = [
        c for c in cols
        if any(h in c.lower() for h in OUTCOME_HINTS)
    ][:80]

    score_hits = {k: v for k, v in scores.items() if v is not None}
    direct_score = 0
    direct_score += 4 if ident["minimum_complete"] else 0
    direct_score += 4 if ident.get("action") else 0
    direct_score += 5 if counts["source_complete"] else 0
    direct_score += 5 if counts["candidate_complete"] else 0
    direct_score += 2 * len(score_hits)
    direct_score += min(3, len(outcome_cols))

    return {
        "type": "csv",
        "path": str(path),
        "name": path.name,
        "bytes": size,
        "sha256": sha256_file(path),
        "columns": cols,
        "identity": ident,
        "counts": counts,
        "scores": scores,
        "score_hits": score_hits,
        "outcome_columns": outcome_cols,
        "rows": rows,
        "unique_patients": patients,
        "families": families,
        "actions": actions,
        "direct_binding_score": direct_score,
    }


def inspect_json(path: Path) -> Dict[str, Any]:
    size = path.stat().st_size
    if size > MAX_JSON_BYTES:
        return {"type": "json", "path": str(path), "bytes": size, "skipped": "too_large"}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        return {"type": "json", "path": str(path), "bytes": size, "error": repr(e)}

    text = json.dumps(obj, ensure_ascii=False)
    keywords = [
        "FinalB", "PROMISE12", "candidate", "action", "outcome",
        "mask", "score", "risk", "TP", "FP", "FN", "patient"
    ]
    hits = [k for k in keywords if k.lower() in text.lower()]

    artifact_paths = []
    def walk(x):
        if isinstance(x, dict):
            for _, v in x.items():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
        elif isinstance(x, str):
            low = x.lower()
            if any(ext in low for ext in [".csv", ".npz", ".npy", ".json"]):
                artifact_paths.append(x)

    walk(obj)
    return {
        "type": "json",
        "path": str(path),
        "name": path.name,
        "bytes": size,
        "sha256": sha256_file(path),
        "keyword_hits": hits,
        "artifact_paths": artifact_paths[:200],
        "status": obj.get("status") if isinstance(obj, dict) else None,
        "gate": obj.get("gate") if isinstance(obj, dict) else None,
        "decision": obj.get("decision") if isinstance(obj, dict) else None,
    }


def inspect_np(path: Path) -> Dict[str, Any]:
    size = path.stat().st_size
    if size > MAX_NPZ_BYTES:
        return {"type": path.suffix.lower()[1:], "path": str(path), "bytes": size, "skipped": "too_large"}

    info = {
        "type": path.suffix.lower()[1:],
        "path": str(path),
        "name": path.name,
        "bytes": size,
        "sha256": sha256_file(path),
    }
    try:
        if path.suffix.lower() == ".npz":
            arrays = {}
            with np.load(path, allow_pickle=False) as z:
                for k in z.files:
                    arr = z[k]
                    arrays[k] = {
                        "shape": list(arr.shape),
                        "dtype": str(arr.dtype),
                    }
            info["arrays"] = arrays
            info["mask_like_arrays"] = [
                k for k in arrays
                if any(h in k.lower() for h in MASK_ARRAY_HINTS)
            ]
        else:
            a = np.load(path, mmap_mode="r", allow_pickle=False)
            info["shape"] = list(a.shape)
            info["dtype"] = str(a.dtype)
            info["mask_like_name"] = any(
                h in path.name.lower() for h in MASK_ARRAY_HINTS
            )
    except Exception as e:
        info["error"] = repr(e)
    return info


def main():
    ap = argparse.ArgumentParser(
        description="SafeTTA P06A static/read-only MRI patient-utility asset binding audit."
    )
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        cols = [
            "case_key","slice_index","family","action",
            "source_tp","source_fp","source_fn",
            "candidate_tp","candidate_fp","candidate_fn",
            "full_transition_risk",
        ]
        assert identity_roles(cols)["minimum_complete"]
        cc = all_count_roles(cols)
        assert cc["source_complete"] and cc["candidate_complete"]
        assert score_roles(cols)["full_transition"] == "full_transition_risk"
        print("SELF_TEST_SCHEMA=PASS")
        print("SELF_TEST=PASS")
        return

    print("=" * 168)
    print("SafeTTA B6-P0-6A — MRI transition patient-utility asset binding audit")
    print("Version             :", VERSION)
    print("Scientific metrics  : NO")
    print("GT pixel read/decode: NO")
    print("Model loading       : NO")
    print("Inference/TTA       : NO")
    print("=" * 168)

    protocol = verify_protocol()
    print("PROTOCOL_SHA256 =", protocol["sha256"])

    roots = candidate_roots()
    print("\n[1/3] Candidate roots")
    for p in roots:
        print(" ", p)

    if not roots:
        raise RuntimeError("No FinalB/B6/CM candidate roots found.")

    print("\n[2/3] Inspect frozen schemas and artifacts")
    records: List[Dict[str, Any]] = []
    for p in iter_candidate_files(roots):
        try:
            if p.suffix.lower() == ".csv":
                r = inspect_csv(p)
            elif p.suffix.lower() == ".json":
                r = inspect_json(p)
            else:
                r = inspect_np(p)
            records.append(r)
        except Exception as e:
            records.append({
                "type": p.suffix.lower().lstrip("."),
                "path": str(p),
                "error": repr(e),
            })

    csvs = [
        r for r in records
        if r.get("type") == "csv" and "direct_binding_score" in r
    ]
    csvs = sorted(
        csvs,
        key=lambda r: (-int(r.get("direct_binding_score", 0)), r.get("path",""))
    )

    print("\nTOP CSV BINDING CANDIDATES")
    for r in csvs[:20]:
        print(
            f"  score={r['direct_binding_score']:02d} "
            f"rows={r.get('rows')} patients={r.get('unique_patients')} "
            f"families={r.get('families')} actions={r.get('actions')}\n"
            f"    {r['path']}\n"
            f"    identity={r['identity']}\n"
            f"    counts={r['counts']}\n"
            f"    scores={r['score_hits']}\n"
            f"    outcomes={r['outcome_columns'][:20]}"
        )

    np_candidates = [
        r for r in records
        if r.get("type") in {"npz","npy"} and (
            r.get("mask_like_arrays") or r.get("mask_like_name")
        )
    ]
    print("\nMASK / PACKED-MASK CANDIDATES")
    for r in np_candidates[:40]:
        print(" ", r["path"])
        if "arrays" in r:
            print("    arrays:", r["arrays"])
        else:
            print("    shape/dtype:", r.get("shape"), r.get("dtype"))

    # Decision: direct if one table has minimum identity + action + all counts.
    direct = [
        r for r in csvs
        if r["identity"]["minimum_complete"]
        and r["identity"].get("action") is not None
        and r["counts"]["source_complete"]
        and r["counts"]["candidate_complete"]
    ]

    score_sources = {
        "full_transition": [
            r for r in csvs if r["scores"].get("full_transition")
        ],
        "source_state": [
            r for r in csvs if r["scores"].get("source_state")
        ],
        "simple_geometry": [
            r for r in csvs if r["scores"].get("simple_geometry")
        ],
    }

    if direct and score_sources["full_transition"]:
        route = "DIRECT_COUNTS"
    elif np_candidates and score_sources["full_transition"]:
        route = "RECOUNT_FROM_FROZEN_MASKS"
    else:
        route = "STOP_NEEDS_EXACT_ASSET_BINDING"

    print("\n[3/3] Binding decision")
    print("ROUTE =", route)
    print("direct outcome/count candidates =", len(direct))
    for k, vals in score_sources.items():
        print(f"{k} score candidates =", len(vals))

    if OUT_DIR.exists():
        raise FileExistsError(
            f"Never overwrite P06A output: {OUT_DIR}\n"
            "Use _fix1 only for implementation repair."
        )
    OUT_DIR.mkdir(parents=True, exist_ok=False)

    # Compact inventories.
    inventory_path = OUT_DIR / "B6_P06A_ASSET_INVENTORY.json"
    inventory_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    decision = {
        "status": "PASS_AUDIT" if route != "STOP_NEEDS_EXACT_ASSET_BINDING" else "STOP",
        "gate": PASS_GATE if route != "STOP_NEEDS_EXACT_ASSET_BINDING" else None,
        "version": VERSION,
        "protocol": protocol,
        "route": route,
        "direct_candidates": direct[:10],
        "score_sources": {k: v[:10] for k, v in score_sources.items()},
        "mask_candidates": np_candidates[:30],
        "next_stage": (
            "Generate exact patient-level utility execution bound only to these assets."
            if route != "STOP_NEEDS_EXACT_ASSET_BINDING"
            else "Do not compute utility. Inspect missing exact outcome/count or mask lineage."
        ),
    }
    decision_path = OUT_DIR / "B6_P06A_BINDING_DECISION.json"
    decision_path.write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\ninventory_json =", inventory_path)
    print("decision_json  =", decision_path)
    print("stage_dir      =", OUT_DIR)
    if route != "STOP_NEEDS_EXACT_ASSET_BINDING":
        print("GATE=" + PASS_GATE)
    else:
        print("GATE=STOP")
    print("=" * 168)


if __name__ == "__main__":
    main()
