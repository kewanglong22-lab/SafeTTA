#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SafeTTA R32B2A
Exact Published-Baseline Execution-Context Binding Bundle

Goal
----
Prepare an exact, GT-independent execution-context bundle for the next
R32B2B score-generation runner.

This stage DOES NOT run:
  - SicTTA-CCD inference
  - TEGDA-ADIC inference
  - MC-dropout inference
  - model training
  - HARM/outcome evaluation
  - target calibration
  - score reversal

Instead it binds and extracts the exact retained implementation context from:

  CCD:
    Q1_R17A_CCD_polypgen_full9_preGT_score_lock_v1_fix2.py

  ADIC + MC-dropout:
    Q1_R17A_ADIC_MC_deeplab3_preGT_score_lock_v1_fix2.py

  NeoPolyp DeepLab SOURCE runtime:
    Q1_R05D3_neopolyp_source_a1_prediction_lock_v1_fix1.py

It also audits:
  - the frozen R32B1 DeepLabV3-R50 common panel;
  - the 3 DeepLab state/index mappings;
  - available NeoPolyp SOURCE-side caches;
  - exact function/class definitions and relevant call sites;
  - exact script/file SHA256 values.

Why a separate binding step?
----------------------------
R32B is a published-method comparison. We must not approximate old CCD,
ADIC, or MC-dropout semantics from memory. R32B2A records the exact local
source context so R32B2B can import/reuse the real implementation.

No scientific protocol is changed by this split.
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


VERSION = "2026-09-12-R32B2A-v1"

ROOT = Path(r"F:\MEDSEG_SAFETTA")
CODE = ROOT / "code"
OUTPUTS = ROOT / "outputs"

# ---------------------------------------------------------------------
# R32B1 lock
# ---------------------------------------------------------------------

R32B1_SCRIPT = CODE / "Q1_R32B1_faithful_common_baseline_panel_lock_v1.py"
EXPECTED_R32B1_SCRIPT_SHA256 = (
    "f25aea7474d4ba19688414f6512ae904495d61ac1c05569220e8cf3c1e3acf0f"
)

R32B1_DIR = ROOT / "R32B1_faithful_common_baseline_panel_lock_v1"
R32B1_MANIFEST = R32B1_DIR / "R32B1_DEEPLAB3_COMMON_PANEL_MANIFEST.csv"
EXPECTED_R32B1_MANIFEST_SHA256 = (
    "2c7745dc145d6bcea33f3cd37e9c3cd0fbbf34d6c5264c69b64eb8fb48942d24"
)
R32B1_PROTOCOL = R32B1_DIR / "R32B1_COMMON_PANEL_PROTOCOL_LOCK.json"
EXPECTED_R32B1_PROTOCOL_SHA256 = (
    "7623cffddd065e3c7aaaf9c3a3a28fa654279cce42efbf3c413bbeddf721a49b"
)
R32B1_FINAL = R32B1_DIR / "R32B1_FINAL_LOCK.json"
EXPECTED_R32B1_FINAL_SHA256 = (
    "ce305d32f77b2c72e5ca6589043aa6644542e35ee74fa7f41e68db45d4cd31e4"
)

# ---------------------------------------------------------------------
# Exact retained implementation scripts
# ---------------------------------------------------------------------

CCD_SCRIPT = CODE / "Q1_R17A_CCD_polypgen_full9_preGT_score_lock_v1_fix2.py"
ADIC_MC_SCRIPT = CODE / "Q1_R17A_ADIC_MC_deeplab3_preGT_score_lock_v1_fix2.py"
NEOPOLYP_SOURCE_SCRIPT = CODE / "Q1_R05D3_neopolyp_source_a1_prediction_lock_v1_fix1.py"

EXACT_MODEL_AUDIT = (
    OUTPUTS
    / "Q1_R17A_ADIC_matched6_exact_model_instantiation_audit_v1"
    / "R17A_ADIC_EXACT_MODEL_INSTANTIATION.csv"
)
FAMILY_SUPPORT = (
    OUTPUTS
    / "Q1_R17A_ADIC_matched6_exact_model_instantiation_audit_v1"
    / "R17A_ADIC_FAMILY_SUPPORT_DECISION.csv"
)

# Existing NeoPolyp source-index assets.
R05D3_STATE_DIR = (
    OUTPUTS
    / "Q1_R05D3_neopolyp_source_a1_prediction_lock_v1_fix1"
    / "state_predictions"
)

# Older NeoPolyp pre-adaptation/source-side caches found in the R32B0 audit.
R05D2_CACHE_DIR = (
    OUTPUTS
    / "Q1_R05D2_neopolyp_prospective_paot_probability_lock_v1_fix1"
    / "state_cache"
)

DEEP_STATES = (
    ("20260817", "DeepLabV3-R50::20260817"),
    ("20260818", "DeepLabV3-R50::20260818"),
    ("20260819", "DeepLabV3-R50::20260819"),
)

EXPECTED_CASES = 800
EXPECTED_STATES = 3
EXPECTED_ACTIONS = 3
EXPECTED_COMMON_ROWS = EXPECTED_CASES * EXPECTED_STATES * EXPECTED_ACTIONS

DEFAULT_OUT = ROOT / "R32B2A_exact_baseline_execution_context_bundle_v1"

# AST extraction is deliberately broad: exact context is preferable to
# accidentally omitting a dependency needed by R32B2B.
SYMBOL_KEYWORDS = (
    "ccd",
    "adic",
    "dropout",
    "mc",
    "entropy",
    "uncert",
    "risk",
    "score",
    "deeplab",
    "model",
    "checkpoint",
    "preprocess",
    "transform",
    "image",
    "infer",
    "predict",
    "forward",
    "logit",
    "prob",
    "mask",
    "seed",
    "rng",
)

CALL_KEYWORDS = (
    "ccd",
    "adic",
    "dropout",
    "entropy",
    "uncert",
    "score",
    "predict",
    "infer",
    "forward",
    "model",
)

CONSTANT_KEYWORDS = (
    "CCD",
    "ADIC",
    "DROPOUT",
    "MC",
    "ITER",
    "SEED",
    "THRESH",
    "TEMP",
    "ALPHA",
    "BETA",
    "EPS",
    "MODEL",
    "CKPT",
    "CHECKPOINT",
    "IMAGE",
    "SIZE",
    "DEVICE",
)

SCHEMA_SCORE_TOKENS = (
    "score",
    "risk",
    "prob",
    "logit",
    "entropy",
    "uncert",
    "confidence",
    "ccd",
    "adic",
    "dropout",
)

SCHEMA_ID_TOKENS = (
    "sample_id",
    "image",
    "path",
    "model_family",
    "model_state_id",
    "training_seed",
    "checkpoint",
)


# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------

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


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def line_span(source: str, node: ast.AST) -> str:
    lines = source.splitlines()
    start = max(1, int(getattr(node, "lineno", 1)))
    end = int(getattr(node, "end_lineno", start))
    return "\n".join(lines[start - 1:end])


# ---------------------------------------------------------------------
# Upstream verification
# ---------------------------------------------------------------------

def verify_r32b1() -> Dict[str, Any]:
    require_sha(R32B1_SCRIPT, EXPECTED_R32B1_SCRIPT_SHA256, "R32B1 script")
    require_sha(R32B1_MANIFEST, EXPECTED_R32B1_MANIFEST_SHA256, "R32B1 manifest")
    require_sha(R32B1_PROTOCOL, EXPECTED_R32B1_PROTOCOL_SHA256, "R32B1 protocol")
    require_sha(R32B1_FINAL, EXPECTED_R32B1_FINAL_SHA256, "R32B1 final")

    final = json.loads(R32B1_FINAL.read_text(encoding="utf-8"))
    if final.get("status") != (
        "PASS_R32B1_FAITHFUL_COMMON_BASELINE_PANEL_LOCK_COMPLETE"
    ):
        raise RuntimeError("R32B1 final status changed.")

    if final.get("common_family") != "DeepLabV3-R50":
        raise RuntimeError("R32B1 common family changed.")

    if bool(final.get("baseline_inference", True)):
        raise RuntimeError("R32B1 unexpectedly reports baseline inference.")
    if bool(final.get("target_recalibration", True)):
        raise RuntimeError("R32B1 target recalibration guard changed.")
    if bool(final.get("post_hoc_score_reversal", True)):
        raise RuntimeError("R32B1 score-reversal guard changed.")
    if bool(final.get("external_data_access", True)):
        raise RuntimeError("R32B1 external-access guard changed.")

    return final


def verify_common_manifest() -> tuple[pd.DataFrame, pd.DataFrame]:
    d = pd.read_csv(
        R32B1_MANIFEST,
        dtype={
            "sample_id": str,
            "model_state_id": str,
            "model_family": str,
            "action": str,
        },
        low_memory=False,
    )

    if len(d) != EXPECTED_COMMON_ROWS:
        raise RuntimeError(
            f"R32B1 common rows={len(d)} expected={EXPECTED_COMMON_ROWS}"
        )

    if d["sample_id"].nunique() != EXPECTED_CASES:
        raise RuntimeError("R32B1 physical-case count drift.")
    if d["model_state_id"].nunique() != EXPECTED_STATES:
        raise RuntimeError("R32B1 state count drift.")
    if d["action"].nunique() != EXPECTED_ACTIONS:
        raise RuntimeError("R32B1 action count drift.")

    # One SOURCE-side baseline score will later be shared across the 3 actions.
    base = (
        d[
            [
                "sample_id",
                "model_state_id",
                "model_family",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            ["model_state_id", "sample_id"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    expected_base_rows = EXPECTED_CASES * EXPECTED_STATES
    if len(base) != expected_base_rows:
        raise RuntimeError(
            f"Common SOURCE model-cases={len(base)} expected={expected_base_rows}"
        )

    return d, base


# ---------------------------------------------------------------------
# Exact source-code extraction
# ---------------------------------------------------------------------

def target_symbol(name: str) -> bool:
    low = name.lower()
    return any(k in low for k in SYMBOL_KEYWORDS)


def target_constant(name: str) -> bool:
    up = name.upper()
    return any(k in up for k in CONSTANT_KEYWORDS)


def extract_script_context(path: Path) -> tuple[Dict[str, Any], str]:
    if not path.is_file():
        raise FileNotFoundError(path)

    source = read_text(path)
    tree = ast.parse(source, filename=str(path))

    functions: List[Dict[str, Any]] = []
    classes: List[Dict[str, Any]] = []
    constants: List[Dict[str, Any]] = []
    callsites: List[Dict[str, Any]] = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if target_symbol(node.name):
                functions.append({
                    "name": node.name,
                    "lineno": int(node.lineno),
                    "end_lineno": int(getattr(node, "end_lineno", node.lineno)),
                    "source": line_span(source, node),
                })

        elif isinstance(node, ast.ClassDef):
            if target_symbol(node.name):
                classes.append({
                    "name": node.name,
                    "lineno": int(node.lineno),
                    "end_lineno": int(getattr(node, "end_lineno", node.lineno)),
                    "source": line_span(source, node),
                })

        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = []
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        targets.append(t.id)
            else:
                if isinstance(node.target, ast.Name):
                    targets.append(node.target.id)

            for name in targets:
                if target_constant(name):
                    constants.append({
                        "name": name,
                        "lineno": int(node.lineno),
                        "source": line_span(source, node),
                    })

        elif isinstance(node, ast.Call):
            try:
                expr = ast.unparse(node.func)
            except Exception:
                expr = ""

            if any(k in expr.lower() for k in CALL_KEYWORDS):
                start = max(1, int(getattr(node, "lineno", 1)) - 3)
                end = min(
                    len(source.splitlines()),
                    int(getattr(node, "end_lineno", getattr(node, "lineno", 1))) + 3,
                )
                snippet = "\n".join(source.splitlines()[start - 1:end])
                callsites.append({
                    "expr": expr,
                    "lineno": int(getattr(node, "lineno", 1)),
                    "context_start": start,
                    "context_end": end,
                    "context": snippet,
                })

    functions.sort(key=lambda x: (x["lineno"], x["name"]))
    classes.sort(key=lambda x: (x["lineno"], x["name"]))
    constants.sort(key=lambda x: (x["lineno"], x["name"]))
    callsites.sort(key=lambda x: (x["lineno"], x["expr"]))

    meta = {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": int(path.stat().st_size),
        "functions": len(functions),
        "classes": len(classes),
        "constants": len(constants),
        "callsites": len(callsites),
        "function_names": [x["name"] for x in functions],
        "class_names": [x["name"] for x in classes],
        "constant_names": [x["name"] for x in constants],
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
        parts.extend([
            f"--- {x['name']} [L{x['lineno']}] ---",
            x["source"],
            "",
        ])

    parts.extend(["### RELEVANT CLASSES", ""])
    for x in classes:
        parts.extend([
            f"--- {x['name']} [L{x['lineno']}-L{x['end_lineno']}] ---",
            x["source"],
            "",
        ])

    parts.extend(["### RELEVANT FUNCTIONS", ""])
    for x in functions:
        parts.extend([
            f"--- {x['name']} [L{x['lineno']}-L{x['end_lineno']}] ---",
            x["source"],
            "",
        ])

    parts.extend(["### RELEVANT CALL SITES", ""])
    # De-duplicate identical surrounding snippets.
    seen = set()
    for x in callsites:
        key = (x["context_start"], x["context_end"], x["context"])
        if key in seen:
            continue
        seen.add(key)
        parts.extend([
            f"--- call={x['expr']} around L{x['lineno']} "
            f"[L{x['context_start']}-L{x['context_end']}] ---",
            x["context"],
            "",
        ])

    return meta, "\n".join(parts)


# ---------------------------------------------------------------------
# DeepLab family support / model-state audit
# ---------------------------------------------------------------------

def audit_native_dropout_support() -> pd.DataFrame:
    if not EXACT_MODEL_AUDIT.is_file():
        raise FileNotFoundError(EXACT_MODEL_AUDIT)

    d = pd.read_csv(EXACT_MODEL_AUDIT, low_memory=False)

    required = {
        "model_family",
        "model_state_id",
        "exact_nn_dropout_count",
    }
    missing = sorted(required.difference(d.columns))
    if missing:
        raise RuntimeError(
            f"Exact model audit missing columns={missing}"
        )

    g = d[d["model_family"].astype(str) == "DeepLabV3-R50"].copy()

    if len(g) != EXPECTED_STATES:
        raise RuntimeError(
            f"DeepLab exact model rows={len(g)} expected={EXPECTED_STATES}"
        )

    if set(g["model_state_id"].astype(str)) != {
        state_id for _, state_id in DEEP_STATES
    }:
        raise RuntimeError("DeepLab exact model state IDs do not match R32B1.")

    if not (pd.to_numeric(g["exact_nn_dropout_count"]) == 1).all():
        raise RuntimeError(
            "DeepLab exact nn.Dropout support is no longer exactly one/state."
        )

    return g.sort_values("model_state_id", kind="mergesort").reset_index(drop=True)


# ---------------------------------------------------------------------
# NeoPolyp exact case/state mapping and cache schema
# ---------------------------------------------------------------------

def csv_schema(path: Path) -> Dict[str, Any]:
    result = {
        "path": str(path),
        "exists": path.is_file(),
        "sha256": "",
        "rows": None,
        "columns": "",
        "score_like_columns": "",
        "id_like_columns": "",
    }

    if not path.is_file():
        return result

    result["sha256"] = sha256_file(path)

    try:
        d = pd.read_csv(path, low_memory=False)
        cols = [str(c) for c in d.columns]
        lows = [c.lower() for c in cols]

        score_cols = [
            c for c, lc in zip(cols, lows)
            if any(tok in lc for tok in SCHEMA_SCORE_TOKENS)
        ]
        id_cols = [
            c for c, lc in zip(cols, lows)
            if any(tok in lc for tok in SCHEMA_ID_TOKENS)
        ]

        result.update({
            "rows": int(len(d)),
            "columns": " | ".join(cols),
            "score_like_columns": " | ".join(score_cols),
            "id_like_columns": " | ".join(id_cols),
        })
    except Exception as exc:
        result["read_error"] = repr(exc)

    return result


def audit_neopolyp_mapping(
    base_modelcases: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    index_rows = []
    cache_rows = []
    mapping_records = []

    common_ids = set(base_modelcases["sample_id"].astype(str))

    for seed, state_id in DEEP_STATES:
        idx = (
            R05D3_STATE_DIR
            / f"deeplabv3_r50_seed{seed}_index.csv"
        )
        cache = (
            R05D2_CACHE_DIR
            / f"deeplabv3_r50_seed{seed}.csv"
        )

        idx_info = csv_schema(idx)
        idx_info.update({
            "asset_type": "R05D3_SOURCE_INDEX",
            "seed": seed,
            "model_state_id": state_id,
        })
        index_rows.append(idx_info)

        cache_info = csv_schema(cache)
        cache_info.update({
            "asset_type": "R05D2_SOURCE_CACHE",
            "seed": seed,
            "model_state_id": state_id,
        })
        cache_rows.append(cache_info)

        if not idx.is_file():
            raise FileNotFoundError(idx)

        d = pd.read_csv(idx, dtype={"sample_id": str}, low_memory=False)
        if "sample_id" not in d.columns:
            raise RuntimeError(f"R05D3 index lacks sample_id: {idx}")

        mapped = d[d["sample_id"].astype(str).isin(common_ids)].copy()

        if mapped["sample_id"].nunique() != EXPECTED_CASES:
            raise RuntimeError(
                f"{state_id}: R05D3 mapping covers "
                f"{mapped['sample_id'].nunique()}/{EXPECTED_CASES} common cases."
            )

        if len(mapped) != EXPECTED_CASES:
            # Exact one index row per physical case is required.
            raise RuntimeError(
                f"{state_id}: mapped rows={len(mapped)} expected={EXPECTED_CASES}"
            )

        preferred_cols = [
            c for c in [
                "sample_id",
                "image_path",
                "image_raw_sha256",
                "model_family",
                "model_state_id",
                "training_seed",
                "checkpoint",
                "checkpoint_sha256",
                "row_index",
            ]
            if c in mapped.columns
        ]

        if "image_path" not in mapped.columns:
            raise RuntimeError(
                f"{state_id}: R05D3 exact index lacks image_path."
            )

        for _, r in mapped[preferred_cols].iterrows():
            rec = {
                "seed": seed,
                "expected_model_state_id": state_id,
            }
            for c in preferred_cols:
                rec[c] = r[c]
            mapping_records.append(rec)

    mapping = pd.DataFrame(mapping_records)

    # Validate exact 800 rows/state.
    counts = (
        mapping.groupby("expected_model_state_id")["sample_id"]
        .nunique()
        .to_dict()
    )
    for _, state_id in DEEP_STATES:
        if int(counts.get(state_id, 0)) != EXPECTED_CASES:
            raise RuntimeError(
                f"{state_id}: exact common-case mapping incomplete."
            )

    info = {
        "states": EXPECTED_STATES,
        "physical_cases_per_state": EXPECTED_CASES,
        "mapping_rows": int(len(mapping)),
        "all_states_have_R05D3_index": all(
            bool(x["exists"]) for x in index_rows
        ),
        "available_R05D2_caches": int(
            sum(bool(x["exists"]) for x in cache_rows)
        ),
    }

    return (
        pd.DataFrame(index_rows),
        pd.DataFrame(cache_rows),
        info,
    ), mapping


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    print("=" * 124)
    print("SafeTTA R32B2A Exact Published-Baseline Execution-Context Binding")
    print("Version                     :", VERSION)
    print("CCD inference                : NO")
    print("ADIC inference               : NO")
    print("MC-dropout inference         : NO")
    print("GT/HARM evaluation           : NO")
    print("Target calibration           : NO")
    print("Post-hoc score reversal      : NO")
    print("External cohort access       : NO")
    print("=" * 124)

    verify_r32b1()
    _, base_modelcases = verify_common_manifest()

    for label, path in {
        "CCD fix2 implementation": CCD_SCRIPT,
        "ADIC/MC fix2 implementation": ADIC_MC_SCRIPT,
        "NeoPolyp SOURCE implementation": NEOPOLYP_SOURCE_SCRIPT,
        "ADIC exact model audit": EXACT_MODEL_AUDIT,
        "ADIC family support": FAMILY_SUPPORT,
    }.items():
        if not path.is_file():
            raise FileNotFoundError(f"{label}: {path}")

    dropout_support = audit_native_dropout_support()

    script_metas = []
    context_parts = []

    for label, path in [
        ("CCD_FIX2", CCD_SCRIPT),
        ("ADIC_MC_FIX2", ADIC_MC_SCRIPT),
        ("NEOPOLYP_SOURCE_FIX1", NEOPOLYP_SOURCE_SCRIPT),
    ]:
        meta, text = extract_script_context(path)
        meta["label"] = label
        script_metas.append(meta)
        context_parts.extend([
            "\n",
            "#" * 120,
            f"# {label}",
            "#" * 120,
            text,
        ])

    (index_schema, cache_schema, mapping_info), mapping = (
        audit_neopolyp_mapping(base_modelcases)
    )

    out = args.output_dir.resolve()
    if out.exists():
        raise RuntimeError(
            f"Output exists; R32B2A will not overwrite: {out}"
        )
    out.mkdir(parents=True, exist_ok=False)

    script_meta_path = out / "R32B2A_IMPLEMENTATION_SCRIPT_BINDINGS.csv"
    dropout_path = out / "R32B2A_DEEPLAB_NATIVE_DROPOUT_SUPPORT.csv"
    index_path = out / "R32B2A_NEOPOLYP_R05D3_INDEX_SCHEMA.csv"
    cache_path = out / "R32B2A_NEOPOLYP_R05D2_CACHE_SCHEMA.csv"
    mapping_path = out / "R32B2A_NEOPOLYP_800CASE_DEEPLAB_MAPPING.csv"
    bundle_path = out / "R32B2A_EXACT_EXECUTION_CONTEXT_BUNDLE.txt"

    atomic_csv(pd.DataFrame(script_metas), script_meta_path)
    atomic_csv(dropout_support, dropout_path)
    atomic_csv(index_schema, index_path)
    atomic_csv(cache_schema, cache_path)
    atomic_csv(mapping, mapping_path)

    header = [
        "=" * 120,
        "SafeTTA R32B2A EXACT EXECUTION-CONTEXT BUNDLE",
        f"Version: {VERSION}",
        "=" * 120,
        "",
        "INFORMATION BOUNDARY",
        "  CCD inference            : NO",
        "  ADIC inference           : NO",
        "  MC-dropout inference     : NO",
        "  GT/HARM evaluation       : NO",
        "  target calibration       : NO",
        "  post-hoc score reversal  : NO",
        "  external cohort access   : NO",
        "",
        "R32B1 COMMON PANEL",
        f"  physical cases          : {EXPECTED_CASES}",
        f"  DeepLab states          : {EXPECTED_STATES}",
        f"  SOURCE model-cases      : {EXPECTED_CASES * EXPECTED_STATES}",
        "",
        "SCRIPT BINDINGS",
    ]

    for meta in script_metas:
        header.extend([
            f"  {meta['label']}",
            f"    path: {meta['path']}",
            f"    sha : {meta['sha256']}",
            f"    relevant functions: {meta['functions']}",
            f"    relevant classes  : {meta['classes']}",
            f"    relevant constants: {meta['constants']}",
            f"    relevant callsites: {meta['callsites']}",
        ])

    header.extend([
        "",
        "NEOPOLYP MAPPING",
        f"  mapping rows          : {mapping_info['mapping_rows']}",
        f"  R05D3 indexes present : {mapping_info['all_states_have_R05D3_index']}",
        f"  R05D2 caches present  : {mapping_info['available_R05D2_caches']}/3",
        "",
    ])

    bundle_text = "\n".join(header + context_parts)
    bundle_path.write_text(bundle_text, encoding="utf-8")

    # Readiness is intentionally structural only. R32B2B will decide which
    # exact functions/caches to invoke after this bundle is inspected.
    ready = (
        len(script_metas) == 3
        and len(dropout_support) == 3
        and int(mapping_info["mapping_rows"]) == EXPECTED_CASES * EXPECTED_STATES
        and bool(mapping_info["all_states_have_R05D3_index"])
    )

    audit = {
        "status": (
            "READY_FOR_R32B2B_EXACT_SCORE_RUNNER"
            if ready
            else "STOP_R32B2A_BINDING_INCOMPLETE"
        ),
        "version": VERSION,
        "R32B1_script_sha256": EXPECTED_R32B1_SCRIPT_SHA256,
        "R32B1_manifest_sha256": EXPECTED_R32B1_MANIFEST_SHA256,
        "R32B1_protocol_sha256": EXPECTED_R32B1_PROTOCOL_SHA256,
        "R32B1_final_sha256": EXPECTED_R32B1_FINAL_SHA256,
        "script_bindings": {
            x["label"]: {
                "path": x["path"],
                "sha256": x["sha256"],
            }
            for x in script_metas
        },
        "native_dropout_support_rows": int(len(dropout_support)),
        "neopolyp_mapping": mapping_info,
        "execution_context_bundle_sha256": sha256_file(bundle_path),
        "baseline_inference": False,
        "GT_HARM_evaluation": False,
        "target_calibration": False,
        "post_hoc_score_reversal": False,
        "external_data_access": False,
        "next": (
            "R32B2B_EXACT_CCD_ADIC_MC_SCORE_GENERATION"
            if ready
            else "STOP_AND_RESOLVE_EXECUTION_CONTEXT"
        ),
    }

    audit_path = out / "R32B2A_AUDIT.json"
    atomic_json(audit_path, audit)

    final = {
        "status": (
            "PASS_R32B2A_EXACT_EXECUTION_CONTEXT_BINDING_COMPLETE"
            if ready
            else "STOP_R32B2A_EXECUTION_CONTEXT_BINDING_INCOMPLETE"
        ),
        "version": VERSION,
        "decision": audit["status"],
        "R32B1_final_sha256": EXPECTED_R32B1_FINAL_SHA256,
        "implementation_bindings_sha256": sha256_file(script_meta_path),
        "dropout_support_sha256": sha256_file(dropout_path),
        "index_schema_sha256": sha256_file(index_path),
        "cache_schema_sha256": sha256_file(cache_path),
        "mapping_sha256": sha256_file(mapping_path),
        "bundle_sha256": sha256_file(bundle_path),
        "audit_sha256": sha256_file(audit_path),
        "baseline_inference": False,
        "external_data_access": False,
        "next": audit["next"],
    }

    final_path = out / "R32B2A_FINAL_LOCK.json"
    atomic_json(final_path, final)

    print("\nR32B2A SCRIPT BINDINGS")
    for meta in script_metas:
        print(
            f"  {meta['label']}: "
            f"functions={meta['functions']} "
            f"callsites={meta['callsites']} "
            f"sha={meta['sha256']}"
        )

    print("\nR32B2A DEEPLAB NATIVE DROPOUT SUPPORT")
    cols = [
        c for c in [
            "model_family",
            "model_state_id",
            "exact_nn_dropout_count",
            "exact_nn_dropout_p",
            "official_tegda_native_nn_dropout_supported",
        ]
        if c in dropout_support.columns
    ]
    print(dropout_support[cols].to_string(index=False))

    print("\nR32B2A NEOPOLYP SOURCE ASSETS")
    print(index_schema[
        [
            "model_state_id",
            "exists",
            "rows",
            "score_like_columns",
            "id_like_columns",
        ]
    ].to_string(index=False))

    print("\nR32B2A OPTIONAL SOURCE CACHES")
    print(cache_schema[
        [
            "model_state_id",
            "exists",
            "rows",
            "score_like_columns",
            "id_like_columns",
        ]
    ].to_string(index=False))

    print("\nR32B2A DECISION:", audit["status"])
    print("Execution-context bundle:", bundle_path)
    print("Bundle SHA256           :", sha256_file(bundle_path))
    print("\nFINAL STATUS :", final["status"])
    print("Baseline inference      : NO")
    print("GT/HARM evaluation      : NO")
    print("Target calibration      : NO")
    print("Post-hoc score reversal : NO")
    print("External cohort access  : NO")
    print("Final lock SHA256       :", sha256_file(final_path))
    print("Output                  :", out)
    print("NEXT                    :", final["next"])
    print("=" * 124)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
