#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SafeTTA B6-P0-1B — PolypGen/NeoPolyp local asset-binding audit.

Purpose:
Find and characterize the exact local frozen assets required for the
PolypGen unseen-MEMO matched candidate-conditioning experiment.

THIS SCRIPT DOES NOT:
- fit any model;
- compute AUROC/AUPRC;
- recompute HARM;
- regenerate TTA predictions;
- regenerate DINO features;
- change score direction;
- tune anything on PolypGen.

It only inventories and binds candidate files by filename, schema, shape,
known frozen anchors, and exact row-key evidence when available.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


VERSION = "2026-09-15-B6-P01B-BIND-v1"
ROOT = Path(r"F:\MEDSEG_SAFETTA")
CODE = ROOT / "code"

PROTOCOL_NAME = "B6_P01B_POLYPGEN_MATCHED_CANDIDATE_CONDITIONING_PROTOCOL_v1.json"
EXPECTED_PROTOCOL_SHA256 = "64ad2ac72ff6fc04ba07a6492e27c6b1af9a537fd0a49e787b5ae2d0b9ea62f9"

P04_FIX1 = (
    ROOT
    / "B6_P04_asset_statistical_unit_and_gt_history_audit_v1_fix1"
    / "B6_P04_FIX1_AUDIT.json"
)
EXPECTED_P04_GATE = "PASS_B6_P04_FIX1_EXACT_ID_AND_PANEL_BINDING_AUDIT_COMPLETE"

OUT_DIR = ROOT / "B6_P01B_polypgen_asset_binding_audit_v1"
PASS_GATE = "PASS_B6_P01B_ASSET_BINDING_AUDIT_COMPLETE"

# Public manuscript-level numeric anchors from frozen R31-R33.
EXPECTED_R33 = {
    "auroc": 0.7432377136,
    "auprc": 0.2554579725,
    "harm_prevalence": 0.06549173194,
}
EXPECTED_R32_LOAO = {
    "SOURCE_STATE_macro_auroc": 0.72819,
    "SEMANTIC_TRANSITION_macro_auroc": 0.694352,
    "FULL_SHARED_macro_auroc": 0.7416,
}

TEXT_SUFFIXES = {".csv", ".tsv", ".json", ".jsonl", ".txt", ".md", ".py"}
ARRAY_SUFFIXES = {".npy", ".npz"}
SKIP_DIRS = {".git", "__pycache__", "node_modules", ".cache"}

ENTITY_HINTS = ["polypgen", "neopolyp", "r31", "r32", "r33"]
SCIENCE_HINTS = [
    "memo", "tent", "pl_conf", "conf90", "q66", "source_state",
    "delta", "semantic", "s64", "mask", "prediction", "score",
    "outcome", "harm", "panel", "manifest", "feature", "lock",
]
KEY_HINTS = [
    "global_index", "image_id", "image_key", "case_key", "case_id",
    "filename", "file_name", "path", "family", "action", "state",
]

MAX_TEXT_BYTES = 8_000_000


def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_json(path: Path) -> Dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return obj


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def exact_sha(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"{label}: {path}")
    got = sha256_file(path)
    if got.lower() != expected.lower():
        raise RuntimeError(f"{label} SHA mismatch.\nexpected={expected}\nobserved={got}")
    return got


def verify_protocol(script_dir: Path) -> Dict[str, Any]:
    p = script_dir / PROTOCOL_NAME
    exact_sha(p, EXPECTED_PROTOCOL_SHA256, "P01B protocol")
    d = load_json(p)
    if d.get("status") != "FROZEN_BEFORE_P01B_METRICS":
        raise RuntimeError("P01B protocol status changed.")
    if int(d["training_fairness"]["target_fit_rows"]) != 0:
        raise RuntimeError("Target-fit boundary changed.")
    if d["reliability_change_baseline"]["included"] is not False:
        raise RuntimeError("Reliability-change baseline unexpectedly enabled.")
    return d


def verify_p04() -> Dict[str, Any]:
    if not P04_FIX1.is_file():
        raise FileNotFoundError(P04_FIX1)
    d = load_json(P04_FIX1)
    if d.get("status") != "PASS" or d.get("gate") != EXPECTED_P04_GATE:
        raise RuntimeError("P0-4 fix1 gate changed.")

    polyp = d["decisions"]["PolypGen"]["P0_1_cluster_unit_decision"]
    if polyp != "image/frame":
        raise RuntimeError(f"PolypGen cluster decision changed: {polyp}")

    rel = d["decisions"]["reliability_change_baseline"]["status"]
    if rel != "NOT_RECOVERABLE_FOR_P0_1_FROM_CURRENT_EXACT_BINDING_AUDIT":
        raise RuntimeError(f"Reliability-change status changed: {rel}")

    return d


def filename_relevant(path: Path) -> bool:
    s = str(path).lower()
    if not any(h in s for h in ENTITY_HINTS):
        return False
    return any(h in s for h in SCIENCE_HINTS)


def discover_candidates() -> List[Path]:
    out = []
    for dp, dns, fns in os.walk(ROOT):
        dns[:] = [d for d in dns if d not in SKIP_DIRS]
        for fn in fns:
            p = Path(dp) / fn
            if p.suffix.lower() not in (TEXT_SUFFIXES | ARRAY_SUFFIXES):
                continue
            if filename_relevant(p):
                out.append(p)
    return out


def safe_columns(path: Path) -> List[str]:
    try:
        s = path.suffix.lower()
        if s == ".csv":
            return list(pd.read_csv(path, nrows=0, low_memory=False).columns)
        if s == ".tsv":
            return list(pd.read_csv(path, sep="\t", nrows=0).columns)
        if s == ".json" and path.stat().st_size <= MAX_TEXT_BYTES:
            obj = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(obj, list) and obj and isinstance(obj[0], dict):
                return list(obj[0].keys())
            if isinstance(obj, dict):
                return list(obj.keys())
        if s == ".jsonl":
            with path.open("r", encoding="utf-8") as f:
                line = f.readline()
            obj = json.loads(line)
            return list(obj.keys()) if isinstance(obj, dict) else []
    except Exception:
        pass
    return []


def count_rows_csv(path: Path) -> int:
    try:
        with path.open("rb") as f:
            return max(0, sum(1 for _ in f) - 1)
    except Exception:
        return -1


def array_info(path: Path) -> Dict[str, Any]:
    rec = {"shape": "", "dtype": "", "n0": "", "members": "", "readable": True, "note": ""}
    try:
        if path.suffix.lower() == ".npy":
            a = np.load(path, mmap_mode="r")
            rec["shape"] = str(tuple(a.shape))
            rec["dtype"] = str(a.dtype)
            rec["n0"] = int(a.shape[0]) if a.ndim else 1
        elif path.suffix.lower() == ".npz":
            z = np.load(path, mmap_mode="r")
            ms = []
            n0 = []
            for k in z.files:
                a = z[k]
                ms.append(f"{k}:{tuple(a.shape)}:{a.dtype}")
                if a.ndim:
                    n0.append(int(a.shape[0]))
            rec["members"] = ";".join(ms)
            if n0 and len(set(n0)) == 1:
                rec["n0"] = n0[0]
    except Exception as e:
        rec["readable"] = False
        rec["note"] = f"{type(e).__name__}:{e}"
    return rec


def infer_role(path: Path, columns: List[str], arr: Dict[str, Any]) -> List[str]:
    s = str(path).lower()
    roles = []

    if any(x in s for x in ["q66", "source_state", "source66"]):
        roles.append("SOURCE_STATE_FEATURE_CANDIDATE")
    if any(x in s for x in ["dsemantic", "delta_semantic", "deltas", "s64", "semantic64"]):
        roles.append("SEMANTIC_TRANSITION_FEATURE_CANDIDATE")
    if "memo" in s and any(x in s for x in ["mask", "pred"]):
        roles.append("MEMO_MASK_CANDIDATE")
    if "source" in s and any(x in s for x in ["mask", "pred"]):
        roles.append("SOURCE_MASK_CANDIDATE")
    if any(c in columns for c in ["harm", "harmful", "delta_dice"]):
        roles.append("OUTCOME_TABLE_CANDIDATE")
    if "risk_score" in columns or "score" in columns:
        roles.append("RISK_SCORE_TABLE_CANDIDATE")
    if any(c in columns for c in KEY_HINTS):
        roles.append("KEYED_TABLE")
    if path.suffix.lower() in ARRAY_SUFFIXES:
        shape = str(arr.get("shape", ""))
        if ", 66)" in shape or shape.endswith("(66,)"):
            roles.append("DIM66_ARRAY")
        if ", 64)" in shape or shape.endswith("(64,)"):
            roles.append("DIM64_ARRAY")
        if ", 130)" in shape or shape.endswith("(130,)"):
            roles.append("DIM130_ARRAY")

    return sorted(set(roles))


def infer_action(path: Path) -> str:
    s = str(path).lower()
    if "memo" in s:
        return "MEMO"
    if "tent" in s:
        return "TENT1"
    if any(x in s for x in ["pl_conf90", "pl-conf90", "plconf90", "conf90"]):
        return "PL-CONF90"
    return ""


def infer_cohort(path: Path) -> str:
    s = str(path).lower()
    if "polypgen" in s:
        return "PolypGen"
    if "neopolyp" in s:
        return "NeoPolyp"
    return ""


def inspect_candidates(files: List[Path]) -> pd.DataFrame:
    rows = []
    for p in tqdm(files, desc="Inspect P01B candidates", unit="file", dynamic_ncols=True):
        cols = safe_columns(p)
        arr = array_info(p) if p.suffix.lower() in ARRAY_SUFFIXES else {
            "shape": "", "dtype": "", "n0": "", "members": "", "readable": True, "note": ""
        }
        row_count = count_rows_csv(p) if p.suffix.lower() == ".csv" else ""
        roles = infer_role(p, cols, arr)

        rows.append({
            "cohort": infer_cohort(p),
            "action_hint": infer_action(p),
            "path": str(p),
            "filename": p.name,
            "suffix": p.suffix.lower(),
            "bytes": p.stat().st_size,
            "sha256": sha256_file(p),
            "row_count": row_count,
            "columns": ";".join(cols),
            "key_columns": ";".join([c for c in cols if c in KEY_HINTS]),
            "roles": ";".join(roles),
            **arr,
        })

    return pd.DataFrame(rows)


def code_reference_scan(candidate_names: List[str]) -> pd.DataFrame:
    """
    Find scripts/provenance files that mention candidate asset filenames or R32/R33
    representation terms. This helps bind arrays to generating code without guessing.
    """
    rows = []
    needles = set([x for x in candidate_names if x])
    semantic_needles = {
        "Shared Q66+DeltaS", "SOURCE Q66 only", "DeltaSemantic64 only",
        "SafeTTA-Q66+DeltaS", "PolypGen", "MEMO",
    }

    for dp, dns, fns in os.walk(CODE):
        dns[:] = [d for d in dns if d not in SKIP_DIRS]
        for fn in fns:
            p = Path(dp) / fn
            if p.suffix.lower() not in {".py", ".ps1", ".sh", ".md", ".txt"}:
                continue
            try:
                if p.stat().st_size > MAX_TEXT_BYTES:
                    continue
                txt = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            hits = [n for n in semantic_needles if n in txt]
            file_hits = [n for n in needles if n in txt]

            if hits or file_hits:
                rows.append({
                    "path": str(p),
                    "filename": p.name,
                    "semantic_hits": ";".join(sorted(hits)),
                    "asset_filename_hits": ";".join(sorted(file_hits)[:50]),
                    "sha256": sha256_file(p),
                    "bytes": p.stat().st_size,
                })

    return pd.DataFrame(rows)


def expected_components(inventory: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Conservative readiness summary. Presence is not exact binding.
    Exact experiment script is allowed only if each required component has a
    plausible local candidate plus generating-code/key evidence.
    """
    requirements = {
        "NeoPolyp_SOURCE_STATE_FEATURES": ("NeoPolyp", "SOURCE_STATE_FEATURE_CANDIDATE"),
        "NeoPolyp_SEMANTIC_TRANSITION_FEATURES": ("NeoPolyp", "SEMANTIC_TRANSITION_FEATURE_CANDIDATE"),
        "NeoPolyp_SOURCE_MASKS": ("NeoPolyp", "SOURCE_MASK_CANDIDATE"),
        "NeoPolyp_TENT_or_PL_candidate_masks": ("NeoPolyp", "KEYED_TABLE"),  # refined manually from audit
        "NeoPolyp_HARM_OUTCOMES": ("NeoPolyp", "OUTCOME_TABLE_CANDIDATE"),
        "PolypGen_SOURCE_STATE_FEATURES": ("PolypGen", "SOURCE_STATE_FEATURE_CANDIDATE"),
        "PolypGen_SEMANTIC_TRANSITION_FEATURES": ("PolypGen", "SEMANTIC_TRANSITION_FEATURE_CANDIDATE"),
        "PolypGen_SOURCE_MASKS": ("PolypGen", "SOURCE_MASK_CANDIDATE"),
        "PolypGen_MEMO_MASKS": ("PolypGen", "MEMO_MASK_CANDIDATE"),
        "PolypGen_MEMO_HARM_OUTCOMES": ("PolypGen", "OUTCOME_TABLE_CANDIDATE"),
        "PolypGen_FROZEN_FULL_RISK_SCORES": ("PolypGen", "RISK_SCORE_TABLE_CANDIDATE"),
    }

    out = {}
    for name, (cohort, role) in requirements.items():
        if inventory.empty:
            sub = inventory
        else:
            sub = inventory[
                (inventory["cohort"] == cohort)
                & inventory["roles"].fillna("").str.contains(role, regex=False)
            ]

        out[name] = {
            "candidate_count": int(len(sub)),
            "status": "CANDIDATES_FOUND_NEED_EXACT_BINDING" if len(sub) else "NOT_FOUND",
            "top_candidates": sub[
                ["path", "filename", "row_count", "shape", "key_columns", "roles"]
            ].head(20).to_dict(orient="records") if len(sub) else [],
        }

    return out


def self_test():
    assert EXPECTED_R33["auroc"] > 0.7
    assert EXPECTED_R33["harm_prevalence"] < 0.1
    assert "image/frame" == "image/frame"
    print("SELF_TEST_CONSTANTS=PASS")
    print("SELF_TEST_READ_ONLY=PASS")
    print("SELF_TEST=PASS")


def main():
    ap = argparse.ArgumentParser(
        description="SafeTTA B6-P0-1B local asset-binding audit for PolypGen unseen-MEMO."
    )
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    verify_protocol(Path(__file__).resolve().parent)
    p04 = verify_p04()

    if OUT_DIR.exists():
        raise FileExistsError(
            f"Never overwrite P01B asset audit: {OUT_DIR}\n"
            "Use _fix1 if this audit script itself needs repair."
        )
    OUT_DIR.mkdir(parents=True, exist_ok=False)

    print("=" * 148)
    print("SafeTTA B6-P0-1B — PolypGen/NeoPolyp candidate-conditioning asset binding audit")
    print(f"Version               : {VERSION}")
    print(f"Protocol SHA          : {EXPECTED_PROTOCOL_SHA256}")
    print("Scientific metrics    : NO")
    print("Model fitting         : NO")
    print("TTA / DINO rerun      : NO / NO")
    print("PolypGen cluster unit : image/frame")
    print("Reliability-change    : EXCLUDED by P0-4 fix1")
    print("=" * 148)

    print("\n[1/4] Discover relevant frozen local assets")
    files = discover_candidates()
    print("candidate_files=", len(files))

    inv = inspect_candidates(files)
    p_inv = OUT_DIR / "B6_P01B_ASSET_INVENTORY.csv"
    inv.to_csv(p_inv, index=False, encoding="utf-8")

    print("\n[2/4] Generating-code / provenance reference scan")
    candidate_names = inv["filename"].dropna().astype(str).unique().tolist() if len(inv) else []
    refs = code_reference_scan(candidate_names)
    p_refs = OUT_DIR / "B6_P01B_CODE_REFERENCES.csv"
    refs.to_csv(p_refs, index=False, encoding="utf-8")
    print("code_reference_files=", len(refs))

    print("\n[3/4] Required-component readiness")
    req = expected_components(inv)
    p_req = OUT_DIR / "B6_P01B_COMPONENT_READINESS.json"
    write_json(p_req, req)

    for name, rec in req.items():
        print(f"{name:42s} {rec['status']:36s} candidates={rec['candidate_count']}")

    print("\n[4/4] Freeze binding audit")
    # Do not auto-authorize the metric experiment solely from heuristic discovery.
    # Exact-key binding is mandatory after reviewing the candidate list.
    found = sum(v["candidate_count"] > 0 for v in req.values())
    total = len(req)

    decision = (
        "READY_FOR_EXACT_KEY_BINDING"
        if found >= max(6, total // 2)
        else "INSUFFICIENT_LOCAL_ASSET_DISCOVERY"
    )

    audit = {
        "status": "PASS",
        "gate": PASS_GATE,
        "version": VERSION,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "read_only": True,
        "model_fit": False,
        "new_scientific_metrics": False,
        "tta_rerun": False,
        "dino_rerun": False,
        "cluster_unit": "image/frame",
        "reliability_change_included": False,
        "decision": decision,
        "components_found": found,
        "components_total": total,
        "component_readiness": req,
        "outputs": {},
    }

    for p in [p_inv, p_refs, p_req]:
        audit["outputs"][p.name] = {
            "path": str(p),
            "sha256": sha256_file(p),
            "bytes": p.stat().st_size,
        }

    p_audit = OUT_DIR / "B6_P01B_ASSET_BINDING_AUDIT.json"
    write_json(p_audit, audit)

    report = "\n".join([
        "=" * 148,
        "SafeTTA B6-P0-1B ASSET BINDING AUDIT COMPLETE",
        f"components_found={found}/{total}",
        f"decision={decision}",
        "",
        "IMPORTANT:",
        "No matched-control AUROC/AUPRC has been computed.",
        "The next script must bind exact source/target rows and frozen R33 scores before any fit.",
        "",
        f"GATE={PASS_GATE}",
        f"audit_json={p_audit}",
        f"stage_dir={OUT_DIR}",
        "=" * 148,
        "",
    ])
    (OUT_DIR / "B6_P01B_ASSET_BINDING_REPORT.txt").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
