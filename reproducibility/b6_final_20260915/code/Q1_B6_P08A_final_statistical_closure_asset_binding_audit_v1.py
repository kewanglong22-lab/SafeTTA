#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SafeTTA B6-P0-8A — final statistical-closure asset-binding audit.

READ ONLY / NO NEW SCIENTIFIC METRICS.

Finds exact frozen assets for:
1) NeoPolyp three-action LOAO row-level paired score/outcome replay;
2) MRI and PolypGen event-support counting;
3) PolypGen highest-recoverable patient/video grouping;
4) optional already-frozen NeoPolyp simple-change comparator.

No model load, no inference, no refit, no AUROC/AUPRC, no bootstrap.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


VERSION = "2026-09-15-B6-P08A-v1"

ROOT = Path(r"F:\MEDSEG_SAFETTA")
CODE = ROOT / "code"

PROTOCOL = CODE / "B6_P08_FINAL_STATISTICAL_CLOSURE_PROTOCOL_v1.json"
EXPECTED_PROTOCOL_SHA256 = "fc231ec059188f44db277940a321f341b2f322b80902664c10889bfb5fe2a147"

OUT_DIR = ROOT / "B6_P08A_final_statistical_closure_asset_binding_audit_v1"
PASS_GATE = "PASS_B6_P08A_FINAL_STATISTICAL_CLOSURE_ASSET_BINDING"

ROOT_PATTERNS = [
    "R31*",
    "R32*",
    "R33*",
    "B6_P01A*",
    "B6_P06B*",
    "B6_P07B*",
]

# We require conservative semantic discovery. No values are used to choose assets.
ID_ALIASES = [
    "sample_id", "image_id", "case_id", "case_key",
    "physical_image_id", "original_sample_id",
]
ACTION_ALIASES = [
    "heldout_action", "held_out_action", "action", "target_action",
]
HARM_ALIASES = [
    "harm", "harm_label", "memo_harm_label", "is_harm",
]
DELTA_ALIASES = [
    "delta_dice", "memo_delta_dice", "dice_delta",
]
STATE_ALIASES = [
    "model_state_id", "state_id", "model_state", "state",
]
PATIENT_VIDEO_CANDIDATES = [
    "patient_id", "patient", "subject_id", "subject",
    "video_id", "video", "sequence_id", "sequence",
    "case_id", "case", "polyp_id", "lesion_id",
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
    if not PROTOCOL.is_file():
        raise FileNotFoundError(PROTOCOL)
    got = sha256_file(PROTOCOL)
    if got != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError(
            f"P08 protocol SHA mismatch expected={EXPECTED_PROTOCOL_SHA256} observed={got}"
        )
    p = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    expected = "FROZEN_AFTER_P07B_BEFORE_FINAL_STATISTICAL_CLOSURE_METRICS"
    if p.get("status") != expected:
        raise RuntimeError(f"P08 protocol status changed: {p.get('status')}")
    return {"path": str(PROTOCOL), "sha256": got, "status": p["status"]}


def first_alias(cols: List[str], aliases: List[str]) -> str | None:
    lower = {c.lower(): c for c in cols}
    for a in aliases:
        if a.lower() in lower:
            return lower[a.lower()]
    return None


def find_semantic_cols(cols: List[str]) -> Dict[str, Any]:
    lower = {c.lower(): c for c in cols}

    def hits(pred):
        return [c for c in cols if pred(c.lower())]

    source_q66 = hits(
        lambda x: (
            ("source" in x or "q66" in x)
            and ("score" in x or "risk" in x or "prob" in x)
            and not any(t in x for t in ["delta", "full", "shared", "transition"])
        )
    )

    shared_full = hits(
        lambda x: (
            any(t in x for t in ["shared", "full_transition", "q66_deltas", "q66+deltas", "safetta"])
            and ("score" in x or "risk" in x or "prob" in x)
        )
    )

    simple_change = hits(
        lambda x: (
            any(t in x for t in ["simple", "geometry", "mask_change"])
            and ("score" in x or "risk" in x or "prob" in x)
        )
    )

    representation = first_alias(cols, ["representation", "method", "variant", "feature_set"])

    return {
        "source_q66_score_candidates": source_q66,
        "shared_full_score_candidates": shared_full,
        "simple_change_score_candidates": simple_change,
        "representation_col": representation,
    }


def inspect_csv(path: Path) -> Dict[str, Any] | None:
    try:
        hdr = pd.read_csv(path, nrows=0, low_memory=False)
    except Exception:
        return None

    cols = list(map(str, hdr.columns))
    sem = find_semantic_cols(cols)

    id_col = first_alias(cols, ID_ALIASES)
    action_col = first_alias(cols, ACTION_ALIASES)
    harm_col = first_alias(cols, HARM_ALIASES)
    delta_col = first_alias(cols, DELTA_ALIASES)
    state_col = first_alias(cols, STATE_ALIASES)

    grouping_cols = [
        c for c in cols
        if c.lower() in PATIENT_VIDEO_CANDIDATES
        or any(
            token in c.lower()
            for token in ["patient", "subject", "video", "sequence", "polyp", "lesion"]
        )
    ]

    name = path.name.lower()
    score = 0
    score += 4 if id_col else 0
    score += 4 if action_col else 0
    score += 4 if harm_col or delta_col else 0
    score += 4 if sem["source_q66_score_candidates"] else 0
    score += 4 if sem["shared_full_score_candidates"] else 0
    score += 2 if sem["representation_col"] else 0
    score += 2 if state_col else 0
    score += 2 * min(2, len(grouping_cols))
    score += sum(
        int(tok in name)
        for tok in ["r32", "loao", "score", "panel", "outcome", "manifest", "polypgen"]
    )

    if score == 0:
        return None

    rows = None
    unique_ids = None
    unique_actions = None
    unique_states = None
    grouping_cardinality = {}

    use = [x for x in [id_col, action_col, state_col, *grouping_cols] if x]
    try:
        if use:
            d = pd.read_csv(path, usecols=sorted(set(use)), low_memory=False)
            rows = int(len(d))
            if id_col:
                unique_ids = int(d[id_col].astype(str).nunique())
            if action_col:
                unique_actions = sorted(d[action_col].astype(str).dropna().unique().tolist())[:30]
            if state_col:
                unique_states = sorted(d[state_col].astype(str).dropna().unique().tolist())[:30]
            for c in grouping_cols:
                grouping_cardinality[c] = int(d[c].astype(str).nunique())
        else:
            rows = sum(
                1 for _ in path.open("r", encoding="utf-8", errors="ignore")
            ) - 1
    except Exception:
        pass

    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "rows": rows,
        "columns": cols,
        "id_col": id_col,
        "action_col": action_col,
        "harm_col": harm_col,
        "delta_col": delta_col,
        "state_col": state_col,
        "grouping_cols": grouping_cols,
        "grouping_cardinality": grouping_cardinality,
        "unique_ids": unique_ids,
        "unique_actions": unique_actions,
        "unique_states": unique_states,
        "semantic": sem,
        "binding_score": int(score),
    }


def candidate_roots() -> List[Path]:
    roots = []
    seen = set()
    for pat in ROOT_PATTERNS:
        for p in sorted(ROOT.glob(pat)):
            if not p.exists():
                continue
            k = str(p.resolve()).lower()
            if k not in seen:
                seen.add(k)
                roots.append(p)
    return roots


def collect_candidates() -> List[Dict[str, Any]]:
    rows = []
    seen = set()
    for root in candidate_roots():
        if root.is_file() and root.suffix.lower() == ".csv":
            files = [root]
        elif root.is_dir():
            files = sorted(root.rglob("*.csv"))
        else:
            files = []

        for p in files:
            k = str(p.resolve()).lower()
            if k in seen:
                continue
            seen.add(k)
            r = inspect_csv(p)
            if r:
                rows.append(r)
    return sorted(rows, key=lambda x: (-x["binding_score"], x["path"]))


def choose_loao_candidates(cands: List[Dict[str, Any]]) -> Dict[str, Any]:
    direct = []
    long_format = []
    join_score = []
    join_outcome = []

    for r in cands:
        path_low = r["path"].lower()
        neoish = any(t in path_low for t in ["r31", "r32", "neopolyp", "loao"])
        if not neoish:
            continue

        has_id = r["id_col"] is not None
        has_action = r["action_col"] is not None
        has_label = r["harm_col"] is not None or r["delta_col"] is not None
        has_source = bool(r["semantic"]["source_q66_score_candidates"])
        has_full = bool(r["semantic"]["shared_full_score_candidates"])
        has_rep = r["semantic"]["representation_col"] is not None

        if has_id and has_action and has_label and has_source and has_full:
            direct.append(r)
        if has_id and has_action and has_label and has_rep:
            long_format.append(r)
        if has_id and has_action and (has_source or has_full or has_rep):
            join_score.append(r)
        if has_id and has_action and has_label:
            join_outcome.append(r)

    if direct:
        route = "DIRECT_WIDE_LOAO_PANEL"
    elif long_format:
        route = "DIRECT_LONG_LOAO_PANEL"
    elif join_score and join_outcome:
        route = "JOIN_LOAO_SCORE_AND_OUTCOME_PANELS"
    else:
        route = "STOP_LOAO_ROW_LEVEL_BINDING_INCOMPLETE"

    return {
        "route": route,
        "direct_wide": direct[:10],
        "direct_long": long_format[:10],
        "score_candidates": join_score[:15],
        "outcome_candidates": join_outcome[:15],
    }


def choose_event_candidates(cands: List[Dict[str, Any]]) -> Dict[str, Any]:
    mri = []
    poly = []
    for r in cands:
        p = r["path"].lower()

        if (
            ("b6_p01a" in p or "finalb3" in p or "b6_p06b" in p)
            and r["id_col"]
            and (r["harm_col"] or r["delta_col"])
        ):
            mri.append(r)

        if (
            ("r33" in p or "polypgen" in p)
            and r["id_col"]
            and r["state_col"]
            and (r["harm_col"] or r["delta_col"])
        ):
            poly.append(r)

    return {
        "MRI_candidates": mri[:15],
        "PolypGen_candidates": poly[:15],
    }


def grouping_audit(cands: List[Dict[str, Any]]) -> Dict[str, Any]:
    poly = []
    for r in cands:
        p = r["path"].lower()
        if "polypgen" not in p and "r33" not in p:
            continue
        if r["grouping_cols"]:
            poly.append(r)

    exact_higher = []
    for r in poly:
        for c in r["grouping_cols"]:
            lc = c.lower()
            if any(t in lc for t in ["patient", "subject", "video", "sequence"]):
                exact_higher.append({
                    "path": r["path"],
                    "sha256": r["sha256"],
                    "column": c,
                    "unique_groups": r["grouping_cardinality"].get(c),
                    "rows": r["rows"],
                    "id_col": r["id_col"],
                    "unique_ids": r["unique_ids"],
                })

    route = (
        "HIGHER_LEVEL_PATIENT_VIDEO_FIELD_RECOVERABLE"
        if exact_higher
        else "IMAGE_FRAME_IS_HIGHEST_RECOVERABLE_UNIT_IN_CURRENT_ASSETS"
    )

    return {
        "route": route,
        "higher_level_candidates": exact_higher[:30],
        "all_grouping_candidates": poly[:30],
    }


def optional_simple_change(cands: List[Dict[str, Any]]) -> Dict[str, Any]:
    hits = []
    for r in cands:
        p = r["path"].lower()
        if not any(t in p for t in ["r31", "r32", "neopolyp", "loao"]):
            continue
        if r["semantic"]["simple_change_score_candidates"]:
            hits.append(r)

    return {
        "recoverable": bool(hits),
        "candidates": hits[:20],
        "rule": "Do not regenerate if absent.",
    }


def main():
    ap = argparse.ArgumentParser(
        description="SafeTTA P08A read-only final statistical-closure asset audit."
    )
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        cols = [
            "sample_id", "heldout_action", "harm_label",
            "source_q66_risk", "shared_q66_deltas_risk",
            "patient_id",
        ]
        sem = find_semantic_cols(cols)
        assert sem["source_q66_score_candidates"] == ["source_q66_risk"]
        assert sem["shared_full_score_candidates"] == ["shared_q66_deltas_risk"]
        assert first_alias(cols, ID_ALIASES) == "sample_id"
        assert first_alias(cols, ACTION_ALIASES) == "heldout_action"
        assert first_alias(cols, HARM_ALIASES) == "harm_label"
        print("SELF_TEST_SCHEMA=PASS")
        print("SELF_TEST=PASS")
        return

    print("=" * 176)
    print("SafeTTA B6-P0-8A — final statistical-closure asset-binding audit")
    print("Version          :", VERSION)
    print("New metrics      : NO")
    print("Bootstrap        : NO")
    print("Model/inference  : NO")
    print("Refit/calibration: NO")
    print("=" * 176)

    protocol = verify_protocol()
    print("PROTOCOL_SHA256 =", protocol["sha256"])

    print("\n[1/4] Scan frozen R31/R32/R33/B6 tables")
    cands = collect_candidates()
    print("candidate CSVs =", len(cands))
    for r in cands[:30]:
        print(
            f" score={r['binding_score']:02d} rows={r.get('rows')} "
            f"ids={r.get('unique_ids')} actions={r.get('unique_actions')} "
            f"states={r.get('unique_states')}\n"
            f"   {r['path']}\n"
            f"   id={r['id_col']} action={r['action_col']} "
            f"harm={r['harm_col']} delta={r['delta_col']} "
            f"rep={r['semantic']['representation_col']}\n"
            f"   sourceQ66={r['semantic']['source_q66_score_candidates']} "
            f"sharedFull={r['semantic']['shared_full_score_candidates']} "
            f"simple={r['semantic']['simple_change_score_candidates']} "
            f"grouping={r['grouping_cols']}"
        )

    print("\n[2/4] NeoPolyp LOAO paired-increment binding")
    loao = choose_loao_candidates(cands)
    print("LOAO_ROUTE =", loao["route"])
    for label in ["direct_wide", "direct_long", "score_candidates", "outcome_candidates"]:
        print(f"\n{label}:")
        for r in loao[label][:8]:
            print(" ", r["path"])
            print("   columns =", r["columns"])

    print("\n[3/4] Event-support asset binding")
    events = choose_event_candidates(cands)
    for k,v in events.items():
        print(k)
        for r in v[:10]:
            print(" ", r["path"])
            print(
                "   id/action/state/harm/delta =",
                r["id_col"], r["action_col"], r["state_col"], r["harm_col"], r["delta_col"]
            )

    print("\n[4/4] PolypGen higher-level grouping + optional simple-change")
    grouping = grouping_audit(cands)
    optional = optional_simple_change(cands)
    print("POLYPGEN_GROUPING_ROUTE =", grouping["route"])
    print("higher-level candidates =", grouping["higher_level_candidates"])
    print("NEOPOLYP_SIMPLE_CHANGE_RECOVERABLE =", optional["recoverable"])
    for r in optional["candidates"][:10]:
        print(" ", r["path"], r["semantic"]["simple_change_score_candidates"])

    if OUT_DIR.exists():
        raise FileExistsError(
            f"Never overwrite P08A output: {OUT_DIR}\n"
            "Use _fix1 only for implementation repair."
        )
    OUT_DIR.mkdir(parents=True, exist_ok=False)

    status = (
        "PASS"
        if loao["route"] != "STOP_LOAO_ROW_LEVEL_BINDING_INCOMPLETE"
        and events["MRI_candidates"]
        and events["PolypGen_candidates"]
        else "STOP"
    )

    decision = {
        "status": status,
        "gate": PASS_GATE if status == "PASS" else None,
        "version": VERSION,
        "protocol": protocol,
        "LOAO": loao,
        "event_support": events,
        "PolypGen_grouping": grouping,
        "optional_NeoPolyp_simple_change": optional,
        "next": (
            "Generate P08B paired LOAO bootstrap + event-support/statistical-unit closure."
            if status == "PASS"
            else "Do not compute final closure metrics; inspect missing row-level binding."
        ),
    }

    p = OUT_DIR / "B6_P08A_FINAL_STATISTICAL_CLOSURE_BINDING.json"
    p.write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\ndecision_json =", p)
    if status == "PASS":
        print("GATE=" + PASS_GATE)
    else:
        print("GATE=STOP")
    print("=" * 176)


if __name__ == "__main__":
    main()
