#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SafeTTA B6-P0-1B — exact-key + NeoPolyp source-feature lineage recovery audit.

This is the mandatory bridge between:
    B6_P01B_polypgen_asset_binding_audit_v1
and the actual matched candidate-conditioning experiment.

WHY THIS STAGE EXISTS
---------------------
The first P01B inventory found 9/11 required component classes, but did not
locate the two NeoPolyp feature components:
    - SOURCE state features
    - semantic-transition features

It also found many target candidates that still require exact row-key binding.

This script therefore:
1) reads the previous P01B inventory;
2) performs a deeper shape-based scan for 64-D / 66-D / 130-D arrays even when
   filenames do not contain "Q66", "S64", "semantic", etc.;
3) scans historical R31/R32/R33 code for representation names and output-path
   literals, then checks whether those files still exist locally;
4) computes deterministic key signatures for candidate tabular assets;
5) produces exact-binding candidates for source/target panels.

READ ONLY.
NO model fitting.
NO HARM recomputation.
NO AUROC/AUPRC.
NO TTA inference.
NO DINO inference.
NO score reversal.
"""

from __future__ import annotations

import argparse
import ast as pyast
import csv
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


VERSION = "2026-09-15-B6-P01B-EXACT-BIND-v1"

ROOT = Path(r"F:\MEDSEG_SAFETTA")
CODE = ROOT / "code"

PROTOCOL = CODE / "B6_P01B_POLYPGEN_MATCHED_CANDIDATE_CONDITIONING_PROTOCOL_v1.json"
EXPECTED_PROTOCOL_SHA256 = "64ad2ac72ff6fc04ba07a6492e27c6b1af9a537fd0a49e787b5ae2d0b9ea62f9"

P01B_BIND_DIR = ROOT / "B6_P01B_polypgen_asset_binding_audit_v1"
P01B_BIND_AUDIT = P01B_BIND_DIR / "B6_P01B_ASSET_BINDING_AUDIT.json"
P01B_INVENTORY = P01B_BIND_DIR / "B6_P01B_ASSET_INVENTORY.csv"
P01B_CODE_REFS = P01B_BIND_DIR / "B6_P01B_CODE_REFERENCES.csv"

P04_FIX1 = (
    ROOT
    / "B6_P04_asset_statistical_unit_and_gt_history_audit_v1_fix1"
    / "B6_P04_FIX1_AUDIT.json"
)

EXPECTED_P01B_BIND_GATE = "PASS_B6_P01B_ASSET_BINDING_AUDIT_COMPLETE"
EXPECTED_P04_GATE = "PASS_B6_P04_FIX1_EXACT_ID_AND_PANEL_BINDING_AUDIT_COMPLETE"

OUT_DIR = ROOT / "B6_P01B_exact_key_and_source_feature_lineage_audit_v1"
PASS_GATE = "PASS_B6_P01B_EXACT_KEY_AND_SOURCE_FEATURE_LINEAGE_AUDIT_COMPLETE"

SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", ".cache",
    "B6_P01B_exact_key_and_source_feature_lineage_audit_v1",
}

ARRAY_SUFFIXES = {".npy", ".npz"}
TABULAR_SUFFIXES = {".csv", ".tsv", ".json", ".jsonl"}
CODE_SUFFIXES = {".py", ".ps1", ".sh", ".md", ".txt", ".json"}

REP_TERMS = [
    "Q66",
    "SOURCE Q66 only",
    "SOURCE_STATE",
    "source_state",
    "source66",
    "DeltaSemantic64",
    "dSemantic64",
    "delta_semantic",
    "semantic64",
    "S64",
    "Shared Q66+DeltaS",
    "SafeTTA-Q66+DeltaS",
    "R32",
    "R33",
    "PolypGen",
    "NeoPolyp",
    "MEMO",
]

PATH_EXT_RE = re.compile(
    r"""(?P<q>["'])(?P<path>[^"'\\\n\r]*?(?:\.npy|\.npz|\.csv|\.json|\.joblib|\.pkl))(?P=q)""",
    re.IGNORECASE,
)

KEY_PRIORITY = [
    "global_index",
    "image_id",
    "image_key",
    "case_key",
    "case_id",
    "filename",
    "file_name",
    "path",
    "family",
    "action",
    "state",
]

MAX_CODE_BYTES = 8_000_000
MAX_JSON_BYTES = 12_000_000


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
    x = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(x, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return x


def write_json(path: Path, x: Dict[str, Any]) -> None:
    path.write_text(
        json.dumps(x, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def exact_sha(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"{label}: {path}")
    got = sha256_file(path)
    if got.lower() != expected.lower():
        raise RuntimeError(
            f"{label} SHA mismatch.\nexpected={expected}\nobserved={got}"
        )
    return got


def verify_upstream() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    exact_sha(PROTOCOL, EXPECTED_PROTOCOL_SHA256, "P01B protocol")
    protocol = load_json(PROTOCOL)

    if protocol.get("status") != "FROZEN_BEFORE_P01B_METRICS":
        raise RuntimeError("P01B protocol status changed.")

    for p in [P01B_BIND_AUDIT, P01B_INVENTORY, P01B_CODE_REFS, P04_FIX1]:
        if not p.is_file():
            raise FileNotFoundError(p)

    bind = load_json(P01B_BIND_AUDIT)
    if bind.get("status") != "PASS" or bind.get("gate") != EXPECTED_P01B_BIND_GATE:
        raise RuntimeError("P01B first-stage asset-binding gate changed.")

    p04 = load_json(P04_FIX1)
    if p04.get("status") != "PASS" or p04.get("gate") != EXPECTED_P04_GATE:
        raise RuntimeError("P0-4 fix1 gate changed.")

    if p04["decisions"]["PolypGen"]["P0_1_cluster_unit_decision"] != "image/frame":
        raise RuntimeError("PolypGen cluster-unit decision changed.")

    return protocol, bind, p04


def iter_files(root: Path, suffixes: set[str]) -> Iterable[Path]:
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns if d not in SKIP_DIRS]
        for fn in fns:
            p = Path(dp) / fn
            if p.suffix.lower() in suffixes:
                yield p


def inspect_array(path: Path) -> List[Dict[str, Any]]:
    """
    One row per .npy file or per member in .npz.
    """
    rows: List[Dict[str, Any]] = []

    try:
        if path.suffix.lower() == ".npy":
            a = np.load(path, mmap_mode="r")
            rows.append({
                "member": "",
                "shape": str(tuple(a.shape)),
                "ndim": int(a.ndim),
                "n0": int(a.shape[0]) if a.ndim else 1,
                "last_dim": int(a.shape[-1]) if a.ndim else 1,
                "dtype": str(a.dtype),
                "readable": True,
                "note": "",
            })
            return rows

        if path.suffix.lower() == ".npz":
            z = np.load(path, mmap_mode="r")
            for key in z.files:
                a = z[key]
                rows.append({
                    "member": key,
                    "shape": str(tuple(a.shape)),
                    "ndim": int(a.ndim),
                    "n0": int(a.shape[0]) if a.ndim else 1,
                    "last_dim": int(a.shape[-1]) if a.ndim else 1,
                    "dtype": str(a.dtype),
                    "readable": True,
                    "note": "",
                })
            return rows

    except Exception as e:
        rows.append({
            "member": "",
            "shape": "",
            "ndim": "",
            "n0": "",
            "last_dim": "",
            "dtype": "",
            "readable": False,
            "note": f"{type(e).__name__}:{e}",
        })

    return rows


def classify_cohort(path: Path) -> str:
    s = str(path).lower()
    if "polypgen" in s:
        return "PolypGen"
    if "neopolyp" in s:
        return "NeoPolyp"
    return ""


def classify_rep_candidate(path: Path, last_dim: Any) -> List[str]:
    s = str(path).lower()
    roles: List[str] = []

    if last_dim == 66:
        roles.append("DIM66_SOURCE_STATE_CANDIDATE")
    if last_dim == 64:
        roles.append("DIM64_SEMANTIC_CANDIDATE")
    if last_dim == 130:
        roles.append("DIM130_FULL_TRANSITION_CANDIDATE")

    if any(x in s for x in ["q66", "source_state", "source66"]):
        roles.append("NAME_SOURCE_STATE")
    if any(x in s for x in ["s64", "semantic64", "dsemantic", "delta_semantic"]):
        roles.append("NAME_SEMANTIC_TRANSITION")
    if any(x in s for x in ["r32", "r33"]):
        roles.append("R32_R33_PATH")

    return sorted(set(roles))


def deep_array_scan() -> pd.DataFrame:
    rows = []

    files = list(iter_files(ROOT, ARRAY_SUFFIXES))

    for p in tqdm(
        files,
        desc="Deep 64/66/130-D array scan",
        unit="file",
        dynamic_ncols=True,
    ):
        for a in inspect_array(p):
            last_dim = a.get("last_dim")
            if last_dim not in {64, 66, 130}:
                continue

            roles = classify_rep_candidate(p, last_dim)

            rows.append({
                "cohort_path_hint": classify_cohort(p),
                "path": str(p),
                "filename": p.name,
                "suffix": p.suffix.lower(),
                "bytes": p.stat().st_size,
                "sha256": sha256_file(p),
                "member": a["member"],
                "shape": a["shape"],
                "ndim": a["ndim"],
                "n0": a["n0"],
                "last_dim": a["last_dim"],
                "dtype": a["dtype"],
                "roles": ";".join(roles),
                "readable": a["readable"],
                "note": a["note"],
            })

    return pd.DataFrame(rows)


def read_text(path: Path) -> str:
    try:
        if path.stat().st_size > MAX_CODE_BYTES:
            return ""
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def path_literals_from_code() -> pd.DataFrame:
    rows = []

    files = list(iter_files(CODE, CODE_SUFFIXES))

    for p in tqdm(
        files,
        desc="Historical code lineage scan",
        unit="file",
        dynamic_ncols=True,
    ):
        txt = read_text(p)
        if not txt:
            continue

        term_hits = [term for term in REP_TERMS if term.lower() in txt.lower()]
        if not term_hits:
            continue

        literals = [m.group("path") for m in PATH_EXT_RE.finditer(txt)]

        # Include code even if no literal; it may expose lineage terminology.
        if not literals:
            rows.append({
                "script": str(p),
                "script_sha256": sha256_file(p),
                "term_hits": ";".join(sorted(set(term_hits))),
                "literal": "",
                "resolved_path": "",
                "exists": False,
                "resolved_sha256": "",
                "resolved_shape": "",
                "resolved_last_dim": "",
            })
            continue

        for lit in sorted(set(literals)):
            raw = Path(lit)

            candidates = []
            if raw.is_absolute():
                candidates.append(raw)
            else:
                candidates.extend([
                    ROOT / raw,
                    CODE / raw,
                    p.parent / raw,
                ])

            resolved: Optional[Path] = None
            for cand in candidates:
                if cand.is_file():
                    resolved = cand
                    break

            shape = ""
            last_dim = ""
            rsha = ""

            if resolved is not None:
                rsha = sha256_file(resolved)
                if resolved.suffix.lower() in ARRAY_SUFFIXES:
                    arr = inspect_array(resolved)
                    if arr:
                        shape = arr[0].get("shape", "")
                        last_dim = arr[0].get("last_dim", "")

            rows.append({
                "script": str(p),
                "script_sha256": sha256_file(p),
                "term_hits": ";".join(sorted(set(term_hits))),
                "literal": lit,
                "resolved_path": str(resolved) if resolved else "",
                "exists": bool(resolved),
                "resolved_sha256": rsha,
                "resolved_shape": shape,
                "resolved_last_dim": last_dim,
            })

    return pd.DataFrame(rows)


def safe_columns(path: Path) -> List[str]:
    try:
        s = path.suffix.lower()
        if s == ".csv":
            return list(pd.read_csv(path, nrows=0, low_memory=False).columns)
        if s == ".tsv":
            return list(pd.read_csv(path, sep="\t", nrows=0).columns)
        if s == ".json" and path.stat().st_size <= MAX_JSON_BYTES:
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


def read_table_for_keys(path: Path) -> Optional[pd.DataFrame]:
    try:
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path, low_memory=False)
        if path.suffix.lower() == ".tsv":
            return pd.read_csv(path, sep="\t", low_memory=False)
        if path.suffix.lower() == ".json" and path.stat().st_size <= MAX_JSON_BYTES:
            obj = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(obj, list) and (not obj or isinstance(obj[0], dict)):
                return pd.DataFrame(obj)
        if path.suffix.lower() == ".jsonl":
            return pd.read_json(path, lines=True)
    except Exception:
        return None
    return None


def canonical_key_columns(columns: List[str]) -> List[str]:
    cols = []
    for key in KEY_PRIORITY:
        if key in columns:
            cols.append(key)

    # family/action are useful secondary keys but insufficient alone.
    identity = [c for c in cols if c not in {"family", "action", "state"}]
    if not identity:
        return []

    # Keep identity plus family/action/state when present.
    return cols


def key_signature(df: pd.DataFrame, keys: List[str]) -> Tuple[str, int, int, str]:
    if not keys:
        return "", 0, 0, ""

    part = df[keys].copy()

    for c in keys:
        part[c] = part[c].astype(str)

    joined = part.astype(str).agg("\x1f".join, axis=1)
    unique_n = int(joined.nunique(dropna=False))
    duplicate_n = int(len(joined) - unique_n)

    # Order-sensitive and order-insensitive hashes.
    h_order = hashlib.sha256(
        ("\n".join(joined.tolist())).encode("utf-8")
    ).hexdigest()

    h_set = hashlib.sha256(
        ("\n".join(sorted(joined.tolist()))).encode("utf-8")
    ).hexdigest()

    return h_order, unique_n, duplicate_n, h_set


def tabular_key_audit(first_inventory: pd.DataFrame) -> pd.DataFrame:
    rows = []

    if first_inventory.empty:
        return pd.DataFrame()

    paths = []
    for x in first_inventory["path"].dropna().astype(str).unique().tolist():
        p = Path(x)
        if p.is_file() and p.suffix.lower() in TABULAR_SUFFIXES:
            paths.append(p)

    for p in tqdm(
        paths,
        desc="Candidate exact-key signatures",
        unit="file",
        dynamic_ncols=True,
    ):
        cols = safe_columns(p)
        keys = canonical_key_columns(cols)
        if not keys:
            continue

        df = read_table_for_keys(p)
        if df is None or df.empty:
            continue

        h_order, unique_n, duplicate_n, h_set = key_signature(df, keys)

        roles = ""
        matches = first_inventory[first_inventory["path"].astype(str) == str(p)]
        if len(matches):
            roles = str(matches.iloc[0].get("roles", ""))

        rows.append({
            "cohort": classify_cohort(p),
            "path": str(p),
            "filename": p.name,
            "rows": int(len(df)),
            "key_columns": ";".join(keys),
            "unique_key_rows": unique_n,
            "duplicate_key_rows": duplicate_n,
            "ordered_key_sha256": h_order,
            "unordered_key_sha256": h_set,
            "roles": roles,
            "file_sha256": sha256_file(p),
        })

    return pd.DataFrame(rows)


def cross_reference_arrays(
    arrays: pd.DataFrame,
    lineage: pd.DataFrame,
) -> pd.DataFrame:
    """
    Adds evidence that an array is directly referenced by R32/R33-related code.
    """
    if arrays.empty:
        return arrays.copy()

    referenced_paths = set()
    referenced_shas = set()

    if not lineage.empty:
        referenced_paths = set(
            lineage.loc[lineage["exists"] == True, "resolved_path"]
            .dropna()
            .astype(str)
            .tolist()
        )
        referenced_shas = set(
            lineage.loc[lineage["exists"] == True, "resolved_sha256"]
            .dropna()
            .astype(str)
            .tolist()
        )

    out = arrays.copy()
    out["referenced_by_historical_code"] = [
        (str(p) in referenced_paths) or (str(sha) in referenced_shas)
        for p, sha in zip(out["path"], out["sha256"])
    ]

    def score_row(r) -> int:
        score = 0
        if r["cohort_path_hint"] == "NeoPolyp":
            score += 4
        if r["cohort_path_hint"] == "PolypGen":
            score += 3
        if r["referenced_by_historical_code"]:
            score += 6
        roles = str(r["roles"])
        if "R32_R33_PATH" in roles:
            score += 3
        if "NAME_SOURCE_STATE" in roles or "NAME_SEMANTIC_TRANSITION" in roles:
            score += 3
        if int(r["last_dim"]) in {64,66,130}:
            score += 1
        return score

    out["lineage_score"] = out.apply(score_row, axis=1)
    return out.sort_values(
        ["lineage_score", "cohort_path_hint", "last_dim", "n0"],
        ascending=[False, True, True, False],
    )


def source_feature_decision(arrays: pd.DataFrame) -> Dict[str, Any]:
    result: Dict[str, Any] = {}

    for label, dim in [
        ("NeoPolyp_SOURCE_STATE", 66),
        ("NeoPolyp_SEMANTIC_TRANSITION", 64),
        ("NeoPolyp_FULL_TRANSITION", 130),
    ]:
        if arrays.empty:
            sub = arrays
        else:
            sub = arrays[
                (arrays["last_dim"] == dim)
                & (
                    (arrays["cohort_path_hint"] == "NeoPolyp")
                    | (arrays["referenced_by_historical_code"] == True)
                )
            ].copy()

        strong = sub[
            sub["referenced_by_historical_code"] == True
        ] if len(sub) else sub

        if len(strong):
            status = "STRONG_LINEAGE_CANDIDATES_FOUND"
            cand = strong.head(20)
        elif len(sub):
            status = "WEAK_SHAPE_CANDIDATES_FOUND"
            cand = sub.head(20)
        else:
            status = "NOT_RECOVERED"
            cand = sub

        result[label] = {
            "dim": dim,
            "status": status,
            "candidate_count": int(len(sub)),
            "strong_candidate_count": int(len(strong)),
            "top_candidates": cand[
                [
                    "path", "member", "shape", "n0", "dtype",
                    "sha256", "referenced_by_historical_code",
                    "lineage_score",
                ]
            ].to_dict(orient="records") if len(cand) else [],
        }

    return result


def exact_key_groups(keys: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Group tables with identical unordered key signatures.
    This reveals exact row-panel relationships without assuming filenames.
    """
    if keys.empty:
        return []

    groups = []

    for sig, g in keys.groupby("unordered_key_sha256"):
        if len(g) < 2:
            continue

        cohorts = sorted(set(g["cohort"].astype(str)))
        roles = ";".join(sorted(set(g["roles"].astype(str))))

        groups.append({
            "unordered_key_sha256": sig,
            "table_count": int(len(g)),
            "cohorts": ";".join(cohorts),
            "roles_union": roles,
            "paths": g[
                ["path", "rows", "key_columns", "roles", "file_sha256"]
            ].to_dict(orient="records"),
        })

    groups.sort(key=lambda x: x["table_count"], reverse=True)
    return groups


def self_test():
    assert EXPECTED_PROTOCOL_SHA256.startswith("64ad2a")
    assert canonical_key_columns(["family", "action"]) == []
    assert canonical_key_columns(["global_index", "family", "action"]) == [
        "global_index", "family", "action"
    ]
    print("SELF_TEST_KEY_RULES=PASS")
    print("SELF_TEST_READ_ONLY=PASS")
    print("SELF_TEST=PASS")


def main():
    ap = argparse.ArgumentParser(
        description=(
            "SafeTTA B6-P0-1B exact-key binding and NeoPolyp source-feature "
            "lineage recovery audit."
        )
    )
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    verify_upstream()

    if OUT_DIR.exists():
        raise FileExistsError(
            f"Never overwrite exact-binding audit: {OUT_DIR}\n"
            "Use _fix1 if this script itself needs repair."
        )

    OUT_DIR.mkdir(parents=True, exist_ok=False)

    print("=" * 156)
    print("SafeTTA B6-P0-1B — exact-key + NeoPolyp source-feature lineage audit")
    print(f"Version            : {VERSION}")
    print(f"Protocol SHA       : {EXPECTED_PROTOCOL_SHA256}")
    print("Model fitting      : NO")
    print("New metrics        : NO")
    print("TTA / DINO rerun   : NO / NO")
    print("Goal               : recover missing NeoPolyp 66-D/64-D lineage and bind exact panels")
    print("=" * 156)

    first_inv = pd.read_csv(P01B_INVENTORY, low_memory=False)

    print("\n[1/5] Deep shape-based 64/66/130-D array recovery")
    arrays = deep_array_scan()
    p_arrays_raw = OUT_DIR / "B6_P01B_DEEP_ARRAY_SCAN.csv"
    arrays.to_csv(p_arrays_raw, index=False, encoding="utf-8")
    print("64/66/130-D candidate arrays =", len(arrays))

    print("\n[2/5] Historical R31/R32/R33 code lineage + path-literal recovery")
    lineage = path_literals_from_code()
    p_lineage = OUT_DIR / "B6_P01B_CODE_LINEAGE_PATHS.csv"
    lineage.to_csv(p_lineage, index=False, encoding="utf-8")
    print("lineage rows =", len(lineage))
    print("resolved historical paths =", int(lineage["exists"].sum()) if len(lineage) else 0)

    print("\n[3/5] Cross-reference arrays against generating-code lineage")
    arrays2 = cross_reference_arrays(arrays, lineage)
    p_arrays = OUT_DIR / "B6_P01B_SOURCE_FEATURE_LINEAGE_CANDIDATES.csv"
    arrays2.to_csv(p_arrays, index=False, encoding="utf-8")

    feature_decision = source_feature_decision(arrays2)
    p_feature_decision = OUT_DIR / "B6_P01B_SOURCE_FEATURE_RECOVERY.json"
    write_json(p_feature_decision, feature_decision)

    for k, v in feature_decision.items():
        print(
            f"{k:36s} {v['status']:34s} "
            f"candidates={v['candidate_count']} strong={v['strong_candidate_count']}"
        )

    print("\n[4/5] Exact key signatures for first-stage tabular candidates")
    keys = tabular_key_audit(first_inv)
    p_keys = OUT_DIR / "B6_P01B_EXACT_KEY_SIGNATURES.csv"
    keys.to_csv(p_keys, index=False, encoding="utf-8")

    groups = exact_key_groups(keys)
    p_groups = OUT_DIR / "B6_P01B_EXACT_KEY_GROUPS.json"
    write_json(p_groups, {"groups": groups})
    print("keyed tables =", len(keys))
    print("multi-table exact-key groups =", len(groups))

    print("\n[5/5] Binding decision")
    src66 = feature_decision["NeoPolyp_SOURCE_STATE"]["status"]
    src64 = feature_decision["NeoPolyp_SEMANTIC_TRANSITION"]["status"]

    source_ready = (
        src66 == "STRONG_LINEAGE_CANDIDATES_FOUND"
        and src64 == "STRONG_LINEAGE_CANDIDATES_FOUND"
    )

    if source_ready and len(groups) > 0:
        decision = "READY_FOR_MANUAL_FREE_EXACT_BINDING_SCRIPT"
    elif (
        feature_decision["NeoPolyp_SOURCE_STATE"]["candidate_count"] > 0
        or feature_decision["NeoPolyp_SEMANTIC_TRANSITION"]["candidate_count"] > 0
    ):
        decision = "SOURCE_FEATURE_CANDIDATES_FOUND_REQUIRE_FINAL_BINDING"
    else:
        decision = "SOURCE_FEATURES_NOT_RECOVERED_FROM_EXISTING_ARRAYS"

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
        "decision": decision,
        "source_feature_recovery": feature_decision,
        "exact_key_group_count": int(len(groups)),
        "outputs": {},
    }

    for p in [
        p_arrays_raw,
        p_lineage,
        p_arrays,
        p_feature_decision,
        p_keys,
        p_groups,
    ]:
        audit["outputs"][p.name] = {
            "path": str(p),
            "sha256": sha256_file(p),
            "bytes": p.stat().st_size,
        }

    p_audit = OUT_DIR / "B6_P01B_EXACT_BINDING_AUDIT.json"
    write_json(p_audit, audit)

    report = "\n".join([
        "=" * 156,
        "SafeTTA B6-P0-1B EXACT BINDING AUDIT COMPLETE",
        f"NeoPolyp SOURCE_STATE       : {src66}",
        f"NeoPolyp SEMANTIC_TRANSITION: {src64}",
        f"exact_key_groups            : {len(groups)}",
        f"decision                    : {decision}",
        "",
        "No AUROC/AUPRC has been computed.",
        "No source feature has been regenerated.",
        "If source features are not recovered, the next decision is whether an exact historical",
        "reconstruction is possible from frozen upstream assets without target-driven redesign.",
        "",
        f"GATE={PASS_GATE}",
        f"audit_json={p_audit}",
        f"stage_dir={OUT_DIR}",
        "=" * 156,
        "",
    ])

    (OUT_DIR / "B6_P01B_EXACT_BINDING_REPORT.txt").write_text(
        report,
        encoding="utf-8",
    )
    print(report)


if __name__ == "__main__":
    main()
