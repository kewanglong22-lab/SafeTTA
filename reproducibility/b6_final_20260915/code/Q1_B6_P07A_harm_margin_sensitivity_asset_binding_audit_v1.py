#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SafeTTA B6-P0-7A — HARM-margin sensitivity asset-binding audit.

READ ONLY. No AUROC/AUPRC, no bootstrap, no label recomputation.

Binds:
1) MRI PROMISE12 full-transition matched-score panel.
2) PolypGen unseen-MEMO frozen full-transition score + DeltaDice assets.

The scientific sensitivity protocol is already frozen in:
B6_P07_HARM_MARGIN_SENSITIVITY_PROTOCOL_v1.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import numpy as np


VERSION = "2026-09-15-B6-P07A-v1"

ROOT = Path(r"F:\MEDSEG_SAFETTA")
CODE = ROOT / "code"

PROTOCOL = CODE / "B6_P07_HARM_MARGIN_SENSITIVITY_PROTOCOL_v1.json"
EXPECTED_PROTOCOL_SHA256 = "9bed5e361d92db364fdc72eaefcd5494b051464977525e266e4c0aa8bc0d233e"

MRI_SCORE = (
    ROOT / "B6_P01A_mri_matched_representation_attribution_v1"
    / "B6_P01A_PROMISE12_MATCHED_SCORES.csv"
)

OUT_DIR = ROOT / "B6_P07A_harm_margin_sensitivity_asset_binding_audit_v1"
PASS_GATE = "PASS_B6_P07A_HARM_MARGIN_SENSITIVITY_ASSET_BINDING"

POLYPGEN_ROOT_PATTERNS = [
    "B6_P01B_polypgen_geometry_core_experiment_v1_fix5",
    "B6_P02_polypgen_exact50_utility_v1",
    "B6_P02B_polypgen_exact50_absolute_references_v1",
    "R33A2B*",
    "R33*",
]

SCORE_ALIASES = [
    "risk_score",
    "full_transition_risk",
    "full_transition_score",
    "safettta_score",
    "safetta_score",
    "harm_risk_score",
]
DELTA_ALIASES = ["delta_dice", "delta", "dice_delta"]
ID_ALIASES = ["sample_id", "image_id", "case_key", "frame_id"]
STATE_ALIASES = ["state_id", "model_state_id", "state", "model_state", "source_state"]


def sha256_file(path: Path, chunk=8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def first_alias(cols: List[str], aliases: List[str]) -> str | None:
    lower = {c.lower(): c for c in cols}
    for a in aliases:
        if a.lower() in lower:
            return lower[a.lower()]
    return None


def verify_protocol() -> Dict[str, Any]:
    if not PROTOCOL.is_file():
        raise FileNotFoundError(PROTOCOL)
    got = sha256_file(PROTOCOL)
    if got != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError(
            f"P07 protocol SHA mismatch expected={EXPECTED_PROTOCOL_SHA256} observed={got}"
        )
    p = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    expected = "FROZEN_POST_REVEAL_SENSITIVITY_PROTOCOL_BEFORE_MARGIN_SENSITIVITY_METRICS"
    if p.get("status") != expected:
        raise RuntimeError(f"P07 protocol status changed: {p.get('status')}")
    return {"path": str(PROTOCOL), "sha256": got, "status": p["status"]}


def bind_mri() -> Dict[str, Any]:
    if not MRI_SCORE.is_file():
        raise FileNotFoundError(MRI_SCORE)

    d = pd.read_csv(MRI_SCORE, low_memory=False)
    required = {"family","action","case_key","slice_index","risk_score","delta_dice","representation"}
    missing = sorted(required - set(d.columns))
    if missing:
        raise RuntimeError(f"MRI P01A missing columns={missing}")

    g = d[d["representation"].astype(str) == "FULL_TRANSITION_SAFETTA"].copy()
    if len(g) != 8262:
        raise RuntimeError(f"MRI full-transition rows={len(g)} expected=8262")
    if g["case_key"].astype(str).nunique() != 50:
        raise RuntimeError("MRI PROMISE12 patient count drift.")
    cells = g[["family","action"]].drop_duplicates()
    if len(cells) != 6:
        raise RuntimeError(f"MRI cell count={len(cells)} expected=6")

    return {
        "path": str(MRI_SCORE),
        "sha256": sha256_file(MRI_SCORE),
        "rows_full_transition": int(len(g)),
        "patients": int(g["case_key"].astype(str).nunique()),
        "cells": cells.sort_values(["family","action"]).to_dict(orient="records"),
        "columns": list(map(str, d.columns)),
        "representation": "FULL_TRANSITION_SAFETTA",
        "score_col": "risk_score",
        "delta_col": "delta_dice",
        "patient_col": "case_key",
    }


def candidate_roots() -> List[Path]:
    roots = []
    seen = set()
    for pat in POLYPGEN_ROOT_PATTERNS:
        for p in sorted(ROOT.glob(pat)):
            if not p.exists():
                continue
            k = str(p.resolve()).lower()
            if k not in seen:
                seen.add(k)
                roots.append(p)
    return roots


def inspect_csv(path: Path) -> Dict[str, Any] | None:
    try:
        hdr = pd.read_csv(path, nrows=0, low_memory=False)
    except Exception:
        return None
    cols = list(map(str, hdr.columns))
    score = first_alias(cols, SCORE_ALIASES)
    delta = first_alias(cols, DELTA_ALIASES)
    ident = first_alias(cols, ID_ALIASES)
    state = first_alias(cols, STATE_ALIASES)

    name = path.name.lower()
    name_bonus = sum(
        int(tok in name)
        for tok in ["polyp", "memo", "r33", "full", "safetta", "score", "utility"]
    )
    schema_score = (
        5 * int(score is not None)
        + 5 * int(delta is not None)
        + 3 * int(ident is not None)
        + 2 * int(state is not None)
        + name_bonus
    )

    if schema_score == 0:
        return None

    rows = None
    unique_ids = None
    unique_states = None
    try:
        use = [x for x in [ident, state] if x]
        if use:
            d = pd.read_csv(path, usecols=sorted(set(use)), low_memory=False)
            rows = int(len(d))
            if ident:
                unique_ids = int(d[ident].astype(str).nunique())
            if state:
                vals = sorted(d[state].astype(str).dropna().unique().tolist())
                unique_states = vals[:30]
        else:
            # Row count without loading full table.
            rows = sum(1 for _ in path.open("r", encoding="utf-8", errors="ignore")) - 1
    except Exception:
        pass

    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "columns": cols,
        "score_col": score,
        "delta_col": delta,
        "id_col": ident,
        "state_col": state,
        "rows": rows,
        "unique_ids": unique_ids,
        "unique_states": unique_states,
        "binding_score": int(schema_score),
    }


def bind_polypgen_candidates() -> Dict[str, Any]:
    roots = candidate_roots()
    candidates = []

    for root in roots:
        if root.is_file() and root.suffix.lower() == ".csv":
            r = inspect_csv(root)
            if r:
                candidates.append(r)
            continue
        if not root.is_dir():
            continue
        for p in sorted(root.rglob("*.csv")):
            r = inspect_csv(p)
            if r:
                candidates.append(r)

    candidates = sorted(
        candidates,
        key=lambda x: (-x["binding_score"], x["path"]),
    )

    # Direct candidate: one 4596-row table carrying score + delta + physical ID.
    direct = [
        r for r in candidates
        if r["score_col"] and r["delta_col"] and r["id_col"]
        and r.get("rows") == 4596
        and (r.get("unique_ids") in {1532, 4596, None})
    ]

    # Join route: 4596-row score table + 4596-row delta table, both with IDs.
    score_only = [
        r for r in candidates
        if r["score_col"] and r["id_col"] and r.get("rows") == 4596
    ]
    delta_only = [
        r for r in candidates
        if r["delta_col"] and r["id_col"] and r.get("rows") == 4596
    ]

    if direct:
        route = "DIRECT_4596_SCORE_DELTA_TABLE"
    elif score_only and delta_only:
        route = "JOIN_4596_SCORE_AND_DELTA_TABLES"
    else:
        route = "STOP_POLYPGEN_EXACT_ASSET_BINDING_INCOMPLETE"

    return {
        "roots": [str(p) for p in roots],
        "route": route,
        "direct_candidates": direct[:10],
        "score_candidates": score_only[:15],
        "delta_candidates": delta_only[:15],
        "top_candidates": candidates[:30],
    }


def main():
    ap = argparse.ArgumentParser(
        description="SafeTTA P07A read-only HARM-margin asset binding."
    )
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        cols = ["sample_id","model_state_id","risk_score","delta_dice"]
        assert first_alias(cols, SCORE_ALIASES) == "risk_score"
        assert first_alias(cols, DELTA_ALIASES) == "delta_dice"
        assert first_alias(cols, ID_ALIASES) == "sample_id"
        assert first_alias(cols, STATE_ALIASES) == "model_state_id"
        print("SELF_TEST_SCHEMA=PASS")
        print("SELF_TEST=PASS")
        return

    print("=" * 168)
    print("SafeTTA B6-P0-7A — HARM-margin sensitivity asset binding")
    print("Version            :", VERSION)
    print("Sensitivity metrics: NO")
    print("Bootstrap          : NO")
    print("Fitting/calibration: NO")
    print("=" * 168)

    protocol = verify_protocol()
    print("PROTOCOL_SHA256 =", protocol["sha256"])

    print("\n[1/2] MRI full-transition panel")
    mri = bind_mri()
    print("path     =", mri["path"])
    print("rows     =", mri["rows_full_transition"])
    print("patients =", mri["patients"])
    print("cells    =", mri["cells"])
    print("MRI_BINDING=PASS")

    print("\n[2/2] PolypGen unseen-MEMO panel")
    poly = bind_polypgen_candidates()
    print("roots =", poly["roots"])
    print("ROUTE =", poly["route"])

    print("\nTOP CANDIDATES")
    for r in poly["top_candidates"][:20]:
        print(
            f" score={r['binding_score']:02d} rows={r.get('rows')} "
            f"ids={r.get('unique_ids')} state={r.get('state_col')} "
            f"score_col={r.get('score_col')} delta_col={r.get('delta_col')}\n"
            f"   {r['path']}"
        )

    if OUT_DIR.exists():
        raise FileExistsError(
            f"Never overwrite P07A output: {OUT_DIR}\n"
            "Use _fix1 only for implementation repair."
        )
    OUT_DIR.mkdir(parents=True, exist_ok=False)

    decision = {
        "status": (
            "PASS"
            if poly["route"] != "STOP_POLYPGEN_EXACT_ASSET_BINDING_INCOMPLETE"
            else "STOP"
        ),
        "gate": (
            PASS_GATE
            if poly["route"] != "STOP_POLYPGEN_EXACT_ASSET_BINDING_INCOMPLETE"
            else None
        ),
        "version": VERSION,
        "protocol": protocol,
        "MRI": mri,
        "PolypGen": poly,
        "next": (
            "Generate P07B fixed-score margin sensitivity execution."
            if poly["route"] != "STOP_POLYPGEN_EXACT_ASSET_BINDING_INCOMPLETE"
            else "Do not compute sensitivity; inspect exact PolypGen score/outcome join assets."
        ),
    }

    p = OUT_DIR / "B6_P07A_HARM_MARGIN_SENSITIVITY_BINDING.json"
    p.write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\ndecision_json =", p)
    if decision["status"] == "PASS":
        print("GATE=" + PASS_GATE)
    else:
        print("GATE=STOP")
    print("=" * 168)


if __name__ == "__main__":
    main()
