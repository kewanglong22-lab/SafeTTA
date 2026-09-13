#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SafeTTA R33A2A
Exact PolypGen MEMO + Transition Execution-Context Binding

Purpose
-------
Prepare an exact implementation bundle for the NEW R33A2 external
joint domain+action shift execution:

    NeoPolyp TENT1 + PL-CONF90 training
        -> PolypGen MEMO-SEG4-1STEP target

This stage DOES NOT run:
  - SOURCE segmentation inference
  - MEMO inference
  - DINOv2 inference
  - PCA transform
  - SafeTTA predictor fitting
  - GT decode/hash/read
  - HARM evaluation
  - target calibration
  - score direction reversal

Instead it binds and extracts the exact retained implementation context from:
  1) R31B2: exact MEMO prediction + semantic lock implementation
  2) R31B1: exact selected MEMO LR/config lineage
  3) R22A1 fix1: exact candidate-conditioned semantic transition extractor
  4) R05D3 fix1: exact DeepLab SOURCE model/preprocessing conventions
  5) R30A0 fix1: exact QSOURCE66 / DSEM64 schema

It also validates:
  - R33A1 1532 x 3 PolypGen target manifest
  - exact target image/checkpoint mapping
  - absence of GT/HARM outcome columns
  - availability of the three DeepLab checkpoints and RGB images

The resulting TXT bundle is intended to be inspected before authoring R33A2B.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import pandas as pd
from tqdm import tqdm


VERSION = "2026-09-12-R33A2A-v1-fix2"

ROOT = Path(r"F:\MEDSEG_SAFETTA")
CODE = ROOT / "code"
OUTPUTS = ROOT / "outputs"

ORIGINAL_R33A2A_SCRIPT = CODE / "Q1_R33A2A_exact_polypgen_memo_transition_context_bundle_v1.py"
EXPECTED_ORIGINAL_R33A2A_SCRIPT_SHA256 = (
    "9251ddad4aef77c5f2020803db9d2b2c9ef1ef70f7e21a9056e3063fe789c328"
)

FAILED_R33A2A_FIX1_SCRIPT = (
    CODE / "Q1_R33A2A_exact_polypgen_memo_transition_context_bundle_v1_fix1.py"
)
EXPECTED_FAILED_R33A2A_FIX1_SCRIPT_SHA256 = (
    "3a931f58b9320bd1aa48e9e48981d7ae8c2a84bdcbcb12f522614677a3986709"
)

# ---------------------------------------------------------------------
# R33A1 immutable lock
# ---------------------------------------------------------------------

R33A1_SCRIPT = CODE / "Q1_R33A1_external_joint_shift_protocol_lock_v1.py"
EXPECTED_R33A1_SCRIPT_SHA256 = (
    "5ff7367d07724ff36517de8e2bb5b6cfe3251c0bbc4e34e5239e357bba815fc4"
)

R33A1_DIR = ROOT / "R33A1_external_joint_shift_protocol_lock_v1"
R33A1_FINAL = R33A1_DIR / "R33A1_FINAL_LOCK.json"
EXPECTED_R33A1_FINAL_SHA256 = (
    "cff6f482b7b170d3be868d6a1ce4ec26e828d062a3a51501efed4a40a3fa684a"
)
R33A1_PROTOCOL = R33A1_DIR / "R33A1_EXTERNAL_JOINT_SHIFT_PROTOCOL_LOCK.json"
EXPECTED_R33A1_PROTOCOL_SHA256 = (
    "1779be8accd4d316170754eab8aaa642a42d7e4a5999625eda27bc662179dd2c"
)
R33A1_TARGET = R33A1_DIR / "R33A1_POLYPGEN_1532x3_PREGT_TARGET_MANIFEST.csv"

# ---------------------------------------------------------------------
# Historical exact implementations
# ---------------------------------------------------------------------

R31B2_SCRIPT = CODE / "Q1_R31B2_800case_memo_prediction_and_semantic_lock_v1.py"
R31B2_DIR = ROOT / "R31B2_800case_memo_prediction_and_semantic_lock_v1"
EXPECTED_R31B2_FINAL_SHA256 = (
    "f2d5c7b6edbffe596d40cc839e9ec437349f8e1bc127a3c7ab5df0c345a120e8"
)

R31B1_SCRIPT = CODE / "Q1_R31B1_memo_seg4_lr_selection_v1.py"
R31B1_DIR = ROOT / "R31B1_memo_seg4_lr_selection_v1"

R22A1_SCRIPT = CODE / "Q1_R22A1_gtfree_candidate_conditioned_transition_feature_extraction_v1_fix1.py"

R05D3_SCRIPT = CODE / "Q1_R05D3_neopolyp_source_a1_prediction_lock_v1_fix1.py"
EXPECTED_R05D3_SHA256 = (
    "df22d6f3054257743d1f12d9e757323ea9f4a052b7642eb4ab610da46ff21251"
)

R30A0_SCRIPT = CODE / "Q1_R30A0_paired_action_crc_protocol_and_asset_audit_v1_fix1.py"
EXPECTED_R30A0_SHA256 = (
    "9dc5d3a13f3d64608c4c599f811f929f731d7fac05e2aaa66eb344e24ace2298"
)

DEFAULT_OUT = ROOT / "R33A2A_exact_polypgen_memo_transition_context_bundle_v1_fix2"

TARGET_CASES = 1532
TARGET_STATES = 3
EXPECTED_TARGET_ROWS = TARGET_CASES * TARGET_STATES
TARGET_FAMILY = "DeepLabV3-R50"
TARGET_STATE_IDS = {
    "DeepLabV3-R50::20260817",
    "DeepLabV3-R50::20260818",
    "DeepLabV3-R50::20260819",
}

# Broad extraction on purpose: this is an implementation-context bundle.
SYMBOL_TOKENS = (
    "memo",
    "deeplab",
    "dino",
    "semantic",
    "transition",
    "occup",
    "mask",
    "pca",
    "qsource",
    "q_source",
    "dsem",
    "pool",
    "feature",
    "model",
    "checkpoint",
    "image",
    "transform",
    "entropy",
    "augment",
    "flip",
    "predict",
    "infer",
    "source",
    "sentinel",
)

CONSTANT_TOKENS = (
    "MEMO",
    "LR",
    "DINO",
    "PCA",
    "QSOURCE",
    "DSEM",
    "IMAGE",
    "MASK",
    "BATCH",
    "DEVICE",
    "SEED",
    "TRANSFORM",
    "MODEL",
    "CHECKPOINT",
    "HARM",
)

CALL_TOKENS = (
    "memo",
    "dino",
    "pca",
    "semantic",
    "transition",
    "deeplab",
    "model",
    "predict",
    "infer",
    "source",
    "mask",
    "entropy",
)

FORBIDDEN_TARGET_COLUMNS = (
    "harm_label",
    "delta_dice",
    "dice_delta",
    "adapted_dice",
    "source_dice",
    "ground_truth",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require_sha(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"{label}: {path}")
    got = sha256_file(path)
    if got.lower() != expected.lower():
        raise RuntimeError(
            f"{label} SHA mismatch\n"
            f"expected={expected}\nobserved={got}\npath={path}"
        )
    return got


def atomic_json(path: Path, payload: Any):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def atomic_csv(df: pd.DataFrame, path: Path):
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def resolve_file(raw: str) -> Path:
    p = Path(str(raw))
    candidates = [p]
    if not p.is_absolute():
        candidates.extend([
            ROOT / p,
            ROOT / str(raw).replace("/", "\\"),
        ])
    for c in candidates:
        if c.is_file():
            return c.resolve()
    raise FileNotFoundError(raw)


def _candidate_run_dirs(token: str) -> List[Path]:
    """
    Locate only likely historical run/output directories.
    Avoid recursively walking raw image/dataset trees.
    """
    token = token.lower()
    candidates: List[Path] = []

    for base in [ROOT, OUTPUTS]:
        if not base.is_dir():
            continue
        try:
            children = list(base.iterdir())
        except Exception:
            children = []
        for p in children:
            if p.is_dir() and token in p.name.lower():
                candidates.append(p.resolve())

    if token == "r31b2" and R31B2_DIR.is_dir():
        candidates.append(R31B2_DIR.resolve())
    if token == "r31b1" and R31B1_DIR.is_dir():
        candidates.append(R31B1_DIR.resolve())

    out: List[Path] = []
    seen = set()
    for p in candidates:
        key = str(p).lower()
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def discover_json_by_exact_sha(
    *,
    token: str,
    expected_sha256: str,
    label: str,
) -> Path:
    """
    Resolve a historical JSON artifact by frozen SHA256, not filename.
    """
    matches: List[Path] = []
    scanned = 0

    for run_dir in _candidate_run_dirs(token):
        for p in run_dir.rglob("*.json"):
            if not p.is_file():
                continue
            scanned += 1
            try:
                got = sha256_file(p)
            except Exception:
                continue
            if got.lower() == expected_sha256.lower():
                matches.append(p.resolve())

    uniq: List[Path] = []
    seen = set()
    for p in matches:
        key = str(p).lower()
        if key not in seen:
            seen.add(key)
            uniq.append(p)

    if len(uniq) != 1:
        raise RuntimeError(
            f"{label}: expected exactly one JSON with SHA256={expected_sha256}; "
            f"found={len(uniq)} paths={[str(p) for p in uniq]}; "
            f"scanned_json_count={scanned}"
        )
    return uniq[0]


def discover_r31b1_script_candidates() -> List[Path]:
    """
    R31B1 source code is optional provenance only.

    Historical workspaces may have renamed/moved the original R31B1 script.
    Never block R33A2A merely because the exact old filename is absent.
    """
    candidates: List[Path] = []

    # Historical nominal path, if it happens to exist.
    if R31B1_SCRIPT.is_file():
        candidates.append(R31B1_SCRIPT.resolve())

    # Search only code/ and likely historical run directories by R31B1 token.
    for base in [CODE, ROOT, OUTPUTS]:
        if not base.is_dir():
            continue

        if base == CODE:
            iterator = base.glob("*R31B1*.py")
        else:
            iterator = []
            try:
                for child in base.iterdir():
                    if child.is_dir() and "r31b1" in child.name.lower():
                        iterator.extend(child.rglob("*.py"))
            except Exception:
                iterator = []

        for p in iterator:
            if p.is_file():
                candidates.append(p.resolve())

    out: List[Path] = []
    seen = set()
    for p in candidates:
        key = str(p).lower()
        if key not in seen:
            seen.add(key)
            out.append(p)

    out.sort(key=lambda p: str(p).lower())
    return out


def discover_r31b1_metadata_jsons() -> List[Dict[str, Any]]:
    """
    R31B1 JSON metadata is useful provenance but not an R33A2A execution gate.
    R31B2 is the authoritative frozen pre-GT MEMO/semantic lock and already
    carries the selected MEMO configuration lineage.
    """
    rows: List[Dict[str, Any]] = []

    for run_dir in _candidate_run_dirs("r31b1"):
        for p in run_dir.rglob("*.json"):
            if not p.is_file():
                continue
            try:
                obj = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue

            status = str(obj.get("status", ""))
            serialized = json.dumps(obj, ensure_ascii=False).lower()
            relevant = (
                "r31b1" in status.lower()
                or "selected_lr" in obj
                or "memo_lr" in obj
                or "1e-05" in serialized
                or "0.00001" in serialized
            )
            if not relevant:
                continue

            rows.append({
                "path": str(p.resolve()),
                "sha256": sha256_file(p),
                "status": obj.get("status"),
                "selected_lr": obj.get("selected_lr", obj.get("memo_lr")),
                "next": obj.get("next"),
            })

    rows.sort(key=lambda x: str(x["path"]).lower())
    return rows


def verify_r33a1() -> Dict[str, Any]:
    require_sha(R33A1_SCRIPT, EXPECTED_R33A1_SCRIPT_SHA256, "R33A1 script")
    require_sha(R33A1_FINAL, EXPECTED_R33A1_FINAL_SHA256, "R33A1 final")
    require_sha(R33A1_PROTOCOL, EXPECTED_R33A1_PROTOCOL_SHA256, "R33A1 protocol")

    if not R33A1_TARGET.is_file():
        raise FileNotFoundError(R33A1_TARGET)

    final = json.loads(R33A1_FINAL.read_text(encoding="utf-8"))
    if final.get("status") != (
        "PASS_R33A1_EXTERNAL_JOINT_SHIFT_PROTOCOL_LOCK_COMPLETE"
    ):
        raise RuntimeError("R33A1 final status changed.")

    if final.get("target_cases") != TARGET_CASES:
        raise RuntimeError("R33A1 target case count changed.")
    if final.get("target_states") != TARGET_STATES:
        raise RuntimeError("R33A1 target state count changed.")
    if final.get("target_modelcases") != EXPECTED_TARGET_ROWS:
        raise RuntimeError("R33A1 target model-case count changed.")
    if bool(final.get("target_calibration", True)):
        raise RuntimeError("R33A1 target calibration guard changed.")
    if bool(final.get("GT_HARM_evaluation", True)):
        raise RuntimeError("R33A1 GT/HARM guard changed.")
    if bool(final.get("new_inference", True)):
        raise RuntimeError("R33A1 new-inference boundary changed.")

    if sha256_file(R33A1_TARGET) != str(final.get("target_manifest_sha256")):
        raise RuntimeError("R33A1 target manifest SHA changed.")

    return final


def verify_historical_files() -> Dict[str, Any]:
    require_sha(
        ORIGINAL_R33A2A_SCRIPT,
        EXPECTED_ORIGINAL_R33A2A_SCRIPT_SHA256,
        "original failed R33A2A v1 script",
    )
    require_sha(
        FAILED_R33A2A_FIX1_SCRIPT,
        EXPECTED_FAILED_R33A2A_FIX1_SCRIPT_SHA256,
        "failed R33A2A fix1 script",
    )
    require_sha(R05D3_SCRIPT, EXPECTED_R05D3_SHA256, "R05D3 script")
    require_sha(R30A0_SCRIPT, EXPECTED_R30A0_SHA256, "R30A0 schema")

    # Authoritative execution dependencies only.
    for p in [R31B2_SCRIPT, R22A1_SCRIPT]:
        if not p.is_file():
            raise FileNotFoundError(p)

    # R31B1 source is historical provenance only.
    r31b1_scripts = discover_r31b1_script_candidates()

    r31b2_final_path = discover_json_by_exact_sha(
        token="r31b2",
        expected_sha256=EXPECTED_R31B2_FINAL_SHA256,
        label="R31B2 authoritative final lock",
    )
    r31b2 = json.loads(
        r31b2_final_path.read_text(encoding="utf-8")
    )

    if r31b2.get("status") != "PASS_R31B2_800CASE_PRE_GT_LOCK_COMPLETE":
        raise RuntimeError(
            f"Unexpected authoritative R31B2 status={r31b2.get('status')!r}"
        )

    r31b1_metadata = discover_r31b1_metadata_jsons()

    return {
        "R31B2_final": r31b2,
        "R31B2_final_path": str(r31b2_final_path),
        "R31B2_final_sha256": sha256_file(r31b2_final_path),
        "R31B1_script_candidates": [
            {
                "path": str(p),
                "sha256": sha256_file(p),
            }
            for p in r31b1_scripts
        ],
        "R31B1_metadata_candidates": r31b1_metadata,
        "observed_script_sha256": {
            "R31B2": sha256_file(R31B2_SCRIPT),
            "R22A1_fix1": sha256_file(R22A1_SCRIPT),
            "R05D3_fix1": sha256_file(R05D3_SCRIPT),
            "R30A0_fix1": sha256_file(R30A0_SCRIPT),
        },
    }


def target_manifest_audit() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    d = pd.read_csv(
        R33A1_TARGET,
        dtype={
            "sample_id": str,
            "model_family": str,
            "model_state_id": str,
        },
        low_memory=False,
    )

    required = {
        "sample_id",
        "image_path",
        "image_raw_sha256",
        "model_family",
        "model_state_id",
        "training_seed",
        "checkpoint_sha256",
        "ccd_risk",
        "adic_harm_risk",
        "mc_predictive_entropy_risk",
    }
    missing = sorted(required.difference(d.columns))
    if missing:
        raise RuntimeError(f"R33A1 target manifest missing={missing}")

    bad = [
        c for c in d.columns
        if any(tok in str(c).lower() for tok in FORBIDDEN_TARGET_COLUMNS)
    ]
    if bad:
        raise RuntimeError(
            f"R33A1 target manifest contains forbidden outcome columns={bad}"
        )

    if len(d) != EXPECTED_TARGET_ROWS:
        raise RuntimeError(
            f"Target rows={len(d)} expected={EXPECTED_TARGET_ROWS}"
        )
    if d["sample_id"].nunique() != TARGET_CASES:
        raise RuntimeError("Target physical-case count drift.")
    if d["model_state_id"].nunique() != TARGET_STATES:
        raise RuntimeError("Target state count drift.")
    if set(d["model_family"].astype(str)) != {TARGET_FAMILY}:
        raise RuntimeError("Target family drift.")
    if set(d["model_state_id"].astype(str)) != TARGET_STATE_IDS:
        raise RuntimeError("Target state IDs drift.")
    if d.duplicated(["sample_id", "model_state_id"]).any():
        raise RuntimeError("Duplicate target model-case.")

    # Validate exact image identity once per physical sample.
    img = (
        d[
            [
                "sample_id",
                "image_path",
                "image_raw_sha256",
            ]
        ]
        .drop_duplicates()
        .sort_values("sample_id", kind="mergesort")
        .reset_index(drop=True)
    )
    if len(img) != TARGET_CASES:
        raise RuntimeError(
            "Target model states do not share one exact RGB mapping/case."
        )

    image_rows = []
    for r in tqdm(
        img.itertuples(index=False),
        total=len(img),
        desc="R33A2A verify PolypGen RGB/checkpoint assets",
        unit="case",
        dynamic_ncols=True,
    ):
        p = resolve_file(str(r.image_path))
        actual = sha256_file(p)
        expected = str(r.image_raw_sha256).lower()
        if actual.lower() != expected:
            raise RuntimeError(
                f"PolypGen RGB SHA mismatch sample={r.sample_id}\n"
                f"expected={expected}\nactual={actual}\npath={p}"
            )
        image_rows.append({
            "sample_id": str(r.sample_id),
            "image_path": str(p),
            "image_raw_sha256": actual,
            "exists": True,
        })

    # Resolve checkpoints by unique state. If exact checkpoint path is not in
    # target manifest, historical R31B2/R05D3 code context will locate it;
    # here we at least bind the expected SHA per state.
    ck = (
        d[
            [
                "model_state_id",
                "training_seed",
                "checkpoint_sha256",
            ]
        ]
        .drop_duplicates()
        .sort_values("training_seed", kind="mergesort")
        .reset_index(drop=True)
    )
    if len(ck) != TARGET_STATES:
        raise RuntimeError("Target checkpoint identity count drift.")

    info = {
        "physical_cases": TARGET_CASES,
        "states": TARGET_STATES,
        "modelcases": EXPECTED_TARGET_ROWS,
        "image_assets_verified": len(image_rows),
        "state_checkpoint_sha256_values": ck[
            ["model_state_id", "checkpoint_sha256"]
        ].to_dict(orient="records"),
        "GT_HARM_columns_present": False,
    }

    return pd.DataFrame(image_rows), ck, info


def name_relevant(name: str) -> bool:
    low = name.lower()
    return any(tok in low for tok in SYMBOL_TOKENS)


def constant_relevant(name: str) -> bool:
    up = name.upper()
    return any(tok in up for tok in CONSTANT_TOKENS)


def extract_context(path: Path) -> Tuple[Dict[str, Any], str]:
    src = path.read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(src, filename=str(path))
    lines = src.splitlines()

    def snippet(node: ast.AST) -> str:
        a = int(getattr(node, "lineno", 1))
        b = int(getattr(node, "end_lineno", a))
        return "\n".join(lines[a - 1:b])

    funcs = []
    classes = []
    constants = []
    calls = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if name_relevant(node.name):
                funcs.append({
                    "name": node.name,
                    "line": int(node.lineno),
                    "end": int(getattr(node, "end_lineno", node.lineno)),
                    "source": snippet(node),
                })

        elif isinstance(node, ast.ClassDef):
            if name_relevant(node.name):
                classes.append({
                    "name": node.name,
                    "line": int(node.lineno),
                    "end": int(getattr(node, "end_lineno", node.lineno)),
                    "source": snippet(node),
                })

        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            names = []
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        names.append(t.id)
            else:
                if isinstance(node.target, ast.Name):
                    names.append(node.target.id)
            for name in names:
                if constant_relevant(name):
                    constants.append({
                        "name": name,
                        "line": int(node.lineno),
                        "source": snippet(node),
                    })

        elif isinstance(node, ast.Call):
            try:
                expr = ast.unparse(node.func)
            except Exception:
                expr = ""
            if any(tok in expr.lower() for tok in CALL_TOKENS):
                ln = int(getattr(node, "lineno", 1))
                a = max(1, ln - 4)
                b = min(len(lines), int(getattr(node, "end_lineno", ln)) + 4)
                calls.append({
                    "expr": expr,
                    "line": ln,
                    "context_start": a,
                    "context_end": b,
                    "source": "\n".join(lines[a - 1:b]),
                })

    funcs.sort(key=lambda x: (x["line"], x["name"]))
    classes.sort(key=lambda x: (x["line"], x["name"]))
    constants.sort(key=lambda x: (x["line"], x["name"]))
    calls.sort(key=lambda x: (x["line"], x["expr"]))

    meta = {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": int(path.stat().st_size),
        "relevant_functions": len(funcs),
        "relevant_classes": len(classes),
        "relevant_constants": len(constants),
        "relevant_callsites": len(calls),
        "function_names": " | ".join(x["name"] for x in funcs),
        "class_names": " | ".join(x["name"] for x in classes),
        "constant_names": " | ".join(x["name"] for x in constants),
    }

    parts = [
        "=" * 120,
        f"FILE: {path}",
        f"SHA256: {meta['sha256']}",
        "=" * 120,
        "",
        "### RELEVANT CONSTANTS",
        "",
    ]

    for x in constants:
        parts += [
            f"--- {x['name']} [L{x['line']}] ---",
            x["source"],
            "",
        ]

    parts += ["### RELEVANT CLASSES", ""]
    for x in classes:
        parts += [
            f"--- {x['name']} [L{x['line']}-L{x['end']}] ---",
            x["source"],
            "",
        ]

    parts += ["### RELEVANT FUNCTIONS", ""]
    for x in funcs:
        parts += [
            f"--- {x['name']} [L{x['line']}-L{x['end']}] ---",
            x["source"],
            "",
        ]

    parts += ["### RELEVANT CALL SITES", ""]
    seen = set()
    for x in calls:
        key = (x["context_start"], x["context_end"], x["source"])
        if key in seen:
            continue
        seen.add(key)
        parts += [
            f"--- call={x['expr']} around L{x['line']} "
            f"[L{x['context_start']}-L{x['context_end']}] ---",
            x["source"],
            "",
        ]

    return meta, "\n".join(parts)


def extract_final_lock_summary(
    path: Path,
    label: str,
) -> Dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    keep = {}
    for key in [
        "status",
        "version",
        "selected_lr",
        "memo_lr",
        "learning_rate",
        "protocol_sha256",
        "script_sha256",
        "final_lock_sha256",
        "GT_read",
        "GT_hash",
        "GT_decode",
        "next",
    ]:
        if key in obj:
            keep[key] = obj[key]
    return {
        "label": label,
        "path": str(path),
        "sha256": sha256_file(path),
        "fields": keep,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    print("=" * 124)
    print("SafeTTA R33A2A Exact PolypGen MEMO + Transition Execution-Context Binding")
    print("Version                     :", VERSION)
    print("FIX2                        : R31B1 source/json fully optional provenance")
    print("Scientific protocol change  : NO")
    print("SOURCE inference            : NO")
    print("MEMO inference              : NO")
    print("DINO/PCA inference          : NO")
    print("Predictor fitting           : NO")
    print("GT/HARM read/evaluation     : NO")
    print("Target calibration          : NO")
    print("External web/network access : NO")
    print("=" * 124)

    verify_r33a1()
    hist = verify_historical_files()
    image_audit, checkpoint_audit, target_info = target_manifest_audit()

    out = args.output_dir.resolve()
    if out.exists():
        raise RuntimeError(
            f"Output exists; R33A2A will not overwrite: {out}"
        )
    out.mkdir(parents=True, exist_ok=False)

    script_targets = [
        ("R31B2_MEMO_SEMANTIC_LOCK", R31B2_SCRIPT),
        ("R22A1_TRANSITION_FEATURES_FIX1", R22A1_SCRIPT),
        ("R05D3_SOURCE_RUNNERS_FIX1", R05D3_SCRIPT),
        ("R30A0_FEATURE_SCHEMA_FIX1", R30A0_SCRIPT),
    ]

    # Append any discoverable R31B1 source files as optional provenance.
    # Their absence does not affect readiness because the authoritative
    # execution implementation is R31B2 and MEMO LR is already frozen.
    for idx, item in enumerate(
        hist["R31B1_script_candidates"],
        start=1,
    ):
        p = Path(item["path"])
        script_targets.append(
            (f"R31B1_OPTIONAL_PROVENANCE_{idx}", p)
        )

    mandatory_script_count = 4

    metas = []
    bundle_parts = []

    for label, path in script_targets:
        meta, text = extract_context(path)
        meta["label"] = label
        metas.append(meta)
        bundle_parts += [
            "",
            "#" * 120,
            f"# {label}",
            "#" * 120,
            text,
        ]

    r31b2_final_path = Path(hist["R31B2_final_path"])

    lock_summaries = [
        extract_final_lock_summary(
            r31b2_final_path,
            "R31B2_AUTHORITATIVE_BY_SHA",
        ),
        extract_final_lock_summary(R33A1_FINAL, "R33A1"),
    ]

    for idx, item in enumerate(
        hist["R31B1_metadata_candidates"],
        start=1,
    ):
        p = Path(item["path"])
        if p.is_file():
            lock_summaries.append(
                extract_final_lock_summary(
                    p,
                    f"R31B1_METADATA_CANDIDATE_{idx}",
                )
            )

    script_binding_path = out / "R33A2A_IMPLEMENTATION_SCRIPT_BINDINGS.csv"
    image_path = out / "R33A2A_POLYPGEN_RGB_ASSET_AUDIT.csv"
    checkpoint_path = out / "R33A2A_POLYPGEN_DEEPLAB_CHECKPOINT_IDENTITY.csv"
    locks_path = out / "R33A2A_HISTORICAL_LOCK_SUMMARY.json"
    bundle_path = out / "R33A2A_EXACT_EXECUTION_CONTEXT_BUNDLE.txt"

    atomic_csv(pd.DataFrame(metas), script_binding_path)
    atomic_csv(image_audit, image_path)
    atomic_csv(checkpoint_audit, checkpoint_path)
    atomic_json(locks_path, lock_summaries)

    header = [
        "=" * 120,
        "SafeTTA R33A2A EXACT EXECUTION-CONTEXT BUNDLE",
        f"Version: {VERSION}",
        "=" * 120,
        "",
        "INFORMATION BOUNDARY",
        "  SOURCE inference          : NO",
        "  MEMO inference            : NO",
        "  DINO/PCA inference        : NO",
        "  predictor fitting         : NO",
        "  GT/HARM read/evaluation   : NO",
        "  target calibration        : NO",
        "  external web access       : NO",
        "",
        "R33A1 EXTERNAL JOINT-SHIFT TARGET",
        f"  dataset                  : PolypGen",
        f"  physical cases           : {target_info['physical_cases']}",
        f"  DeepLab states           : {target_info['states']}",
        f"  target model-cases       : {target_info['modelcases']}",
        f"  RGB assets verified      : {target_info['image_assets_verified']}",
        f"  GT/HARM columns present  : {target_info['GT_HARM_columns_present']}",
        "",
        "PRIMARY DIRECTION",
        "  NeoPolyp TENT1 + PL-CONF90 -> PolypGen MEMO-SEG4-1STEP",
        "  features: QSOURCE66 + DSEM64",
        "  target calibration: NO",
        "",
        "SCRIPT BINDINGS",
    ]

    for m in metas:
        header += [
            f"  {m['label']}",
            f"    path      : {m['path']}",
            f"    sha256    : {m['sha256']}",
            f"    functions : {m['relevant_functions']}",
            f"    callsites : {m['relevant_callsites']}",
        ]

    header += [
        "",
        "OBSERVED HISTORICAL SCRIPT SHA256",
    ]
    for k, v in hist["observed_script_sha256"].items():
        header.append(f"  {k}: {v}")

    header += [
        "",
        "HISTORICAL FINAL LOCKS",
    ]
    for item in lock_summaries:
        header += [
            f"  {item['label']}",
            f"    path   : {item['path']}",
            f"    sha256 : {item['sha256']}",
            f"    fields : {json.dumps(item['fields'], ensure_ascii=False)}",
        ]

    bundle_path.write_text(
        "\n".join(header + bundle_parts),
        encoding="utf-8",
    )

    mandatory_labels = {
        "R31B2_MEMO_SEMANTIC_LOCK",
        "R22A1_TRANSITION_FEATURES_FIX1",
        "R05D3_SOURCE_RUNNERS_FIX1",
        "R30A0_FEATURE_SCHEMA_FIX1",
    }
    observed_labels = {m["label"] for m in metas}

    ready = (
        mandatory_labels.issubset(observed_labels)
        and len(image_audit) == TARGET_CASES
        and len(checkpoint_audit) == TARGET_STATES
        and target_info["GT_HARM_columns_present"] is False
    )

    audit = {
        "status": (
            "READY_FOR_R33A2B_EXTERNAL_MEMO_TRANSITION_EXECUTION"
            if ready
            else "STOP_R33A2A_CONTEXT_INCOMPLETE"
        ),
        "version": VERSION,
        "implementation_fix": (
            "Mechanical provenance-resolution fix: authoritative execution "
            "depends on R31B2/R22A1/R05D3/R30A0 only; R31B1 source and JSON "
            "are optional provenance because MEMO LR=1e-5 is already frozen "
            "upstream."
        ),
        "original_R33A2A_script_sha256": EXPECTED_ORIGINAL_R33A2A_SCRIPT_SHA256,
        "failed_R33A2A_fix1_script_sha256": (
            EXPECTED_FAILED_R33A2A_FIX1_SCRIPT_SHA256
        ),
        "scientific_protocol_changed": False,
        "R33A1_script_sha256": EXPECTED_R33A1_SCRIPT_SHA256,
        "R33A1_final_sha256": EXPECTED_R33A1_FINAL_SHA256,
        "R33A1_protocol_sha256": EXPECTED_R33A1_PROTOCOL_SHA256,
        "R31B2_final_sha256": EXPECTED_R31B2_FINAL_SHA256,
        "historical_script_sha256": hist["observed_script_sha256"],
        "target": target_info,
        "bundle_sha256": sha256_file(bundle_path),
        "SOURCE_inference": False,
        "MEMO_inference": False,
        "DINO_PCA_inference": False,
        "predictor_fitting": False,
        "GT_HARM_read_evaluation": False,
        "target_calibration": False,
        "external_web_network_access": False,
        "next": (
            "R33A2B_EXTERNAL_MEMO_TRANSITION_SCORE_LOCK"
            if ready
            else "STOP_AND_RESOLVE_R33A2_CONTEXT"
        ),
    }

    audit_path = out / "R33A2A_AUDIT.json"
    atomic_json(audit_path, audit)

    final = {
        "status": (
            "PASS_R33A2A_EXACT_EXECUTION_CONTEXT_BINDING_COMPLETE"
            if ready
            else "STOP_R33A2A_EXACT_EXECUTION_CONTEXT_BINDING_INCOMPLETE"
        ),
        "version": VERSION,
        "implementation_fix_only": True,
        "original_R33A2A_script_sha256": EXPECTED_ORIGINAL_R33A2A_SCRIPT_SHA256,
        "failed_R33A2A_fix1_script_sha256": (
            EXPECTED_FAILED_R33A2A_FIX1_SCRIPT_SHA256
        ),
        "scientific_protocol_changed": False,
        "decision": audit["status"],
        "R33A1_final_sha256": EXPECTED_R33A1_FINAL_SHA256,
        "R31B2_final_sha256": EXPECTED_R31B2_FINAL_SHA256,
        "script_bindings_sha256": sha256_file(script_binding_path),
        "RGB_asset_audit_sha256": sha256_file(image_path),
        "checkpoint_identity_sha256": sha256_file(checkpoint_path),
        "historical_lock_summary_sha256": sha256_file(locks_path),
        "bundle_sha256": sha256_file(bundle_path),
        "audit_sha256": sha256_file(audit_path),
        "new_inference": False,
        "GT_HARM_evaluation": False,
        "target_calibration": False,
        "external_web_network_access": False,
        "next": audit["next"],
    }

    final_path = out / "R33A2A_FINAL_LOCK.json"
    atomic_json(final_path, final)

    print("\nR33A2A TARGET ASSET AUDIT")
    print("  PolypGen physical cases :", target_info["physical_cases"])
    print("  DeepLab states          :", target_info["states"])
    print("  target model-cases      :", target_info["modelcases"])
    print("  RGB SHA verified        :", target_info["image_assets_verified"])
    print("  GT/HARM columns         : NONE")

    print("\nR33A2A IMPLEMENTATION BINDINGS")
    for m in metas:
        print(
            f"  {m['label']}: "
            f"functions={m['relevant_functions']} "
            f"callsites={m['relevant_callsites']} "
            f"sha={m['sha256']}"
        )

    print("\nR33A2A HISTORICAL LOCKS")
    print(
        "  Authoritative R31B2 final path:",
        hist["R31B2_final_path"],
    )
    print(
        "  Authoritative R31B2 final SHA :",
        hist["R31B2_final_sha256"],
    )
    print(
        "  R31B1 source candidates       :",
        len(hist["R31B1_script_candidates"]),
    )
    print(
        "  R31B1 metadata candidates     :",
        len(hist["R31B1_metadata_candidates"]),
    )
    print("  R31B1 required for readiness  : NO")
    for item in lock_summaries:
        print(f"  {item['label']} final SHA: {item['sha256']}")
        print(f"    fields: {item['fields']}")

    print("\nR33A2A DECISION:", audit["status"])
    print("Execution-context bundle :", bundle_path)
    print("Bundle SHA256            :", sha256_file(bundle_path))
    print("\nFINAL STATUS :", final["status"])
    print("Scientific protocol change: NO")
    print("New inference            : NO")
    print("GT/HARM evaluation       : NO")
    print("Target calibration       : NO")
    print("Final lock SHA256        :", sha256_file(final_path))
    print("Output                   :", out)
    print("NEXT                     :", final["next"])
    print("=" * 124)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
