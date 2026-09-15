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


VERSION = "2026-09-15-B6-P08A-v1-fix2"

ROOT = Path(r"F:\MEDSEG_SAFETTA")
CODE = ROOT / "code"

PROTOCOL = CODE / "B6_P08_FINAL_STATISTICAL_CLOSURE_PROTOCOL_v1.json"
EXPECTED_PROTOCOL_SHA256 = "fc231ec059188f44db277940a321f341b2f322b80902664c10889bfb5fe2a147"

OUT_DIR = ROOT / "B6_P08A_final_statistical_closure_asset_binding_audit_v1_fix2"
PASS_GATE = "PASS_B6_P08A_FINAL_STATISTICAL_CLOSURE_ASSET_BINDING"

ROOT_PATTERNS = [
    "R31*",
    "R32*",
    "R33*",
    "B6_P01A*",
    "B6_P06B*",
    "B6_P07B*",
]

# NeoPolyp three-action LOAO binding is allowed ONLY from the R31/R32 colonoscopy lineage.
# B6_P01A is MRI and MUST NEVER satisfy the NeoPolyp LOAO route.
NEOPOLYP_LOAO_ALLOWED_PREFIXES = ("R31", "R32")

NEOPOLYP_EXACT_FILES = [
    ROOT / "R32A1_three_action_loao_comparison_execution_v1_fix1"
         / "R32A1_FIX1_OOF_PREDICTIONS.csv",
    ROOT / "R32B3_matched_three_action_harm_evaluation_v1_fix1"
         / "R32B3_MATCHED_EVALUATION_PANEL.csv",
    ROOT / "R31B3_first_800case_gt_reveal_and_three_action_loao_v1"
         / "R31B3_THREE_ACTION_MODELCASE_TABLE.csv",
    ROOT / "R31B3_first_800case_gt_reveal_and_three_action_loao_v1"
         / "R31B3_OOF_PREDICTIONS.csv",
    ROOT / "R31B3_first_800case_gt_reveal_and_three_action_loao_v1"
         / "R31B3_MEMO_800CASE_MODELCASE_OUTCOMES.csv",
]

R32A1_OOF = (
    ROOT / "R32A1_three_action_loao_comparison_execution_v1_fix1"
    / "R32A1_FIX1_OOF_PREDICTIONS.csv"
)
R31B3_OOF = (
    ROOT / "R31B3_first_800case_gt_reveal_and_three_action_loao_v1"
    / "R31B3_OOF_PREDICTIONS.csv"
)
R31B3_MODELCASE = (
    ROOT / "R31B3_first_800case_gt_reveal_and_three_action_loao_v1"
    / "R31B3_THREE_ACTION_MODELCASE_TABLE.csv"
)

# These tokens are used ONLY to surface semantically named frozen variants.
# They never inspect AUROC/AUPRC or choose a variant by outcome performance.
SOURCE_STATE_POSITIVE_TOKENS = (
    "source_state", "source-state", "source state",
    "q66", "source66", "source_66", "state_only", "source_only_state",
)
TRANSITION_POSITIVE_TOKENS = (
    "transition", "delta", "deltas", "dq", "candidate", "change",
)
SHARED_POSITIVE_TOKENS = (
    "shared", "pooled", "common", "joint",
)
EXCLUDE_SOURCE_STATE_TOKENS = (
    "transition", "delta", "deltas", "dq", "candidate",
    "semantic_only", "semantic-only", "separate", "action_cond",
    "action-cond", "action_specific", "action-specific",
)
EXCLUDE_SHARED_FULL_TOKENS = (
    "separate", "action_cond", "action-cond",
    "action_specific", "action-specific",
)


def _norm_token(x: Any) -> str:
    if pd.isna(x):
        return "<NA>"
    return str(x).strip()


def _signature_text(row: Dict[str, Any]) -> str:
    parts = []
    for k in ("model", "evaluation_type", "direction"):
        if k in row:
            parts.append(f"{k}={_norm_token(row[k])}")
    return " | ".join(parts)


def _semantic_signature_candidates(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Identify possible frozen variant labels from their NAMES ONLY.

    No outcome values, AUROC, AUPRC, historical point estimates, or score
    correlation are used here. Ambiguity intentionally produces STOP.
    """
    source_state = []
    shared_transition = []
    all_signatures = []

    for rec in records:
        sig = _signature_text(rec)
        low = sig.lower()
        all_signatures.append(sig)

        source_positive = any(t in low for t in SOURCE_STATE_POSITIVE_TOKENS)
        source_excluded = any(t in low for t in EXCLUDE_SOURCE_STATE_TOKENS)
        if source_positive and not source_excluded:
            source_state.append(sig)

        has_state = any(t in low for t in SOURCE_STATE_POSITIVE_TOKENS)
        has_transition = any(t in low for t in TRANSITION_POSITIVE_TOKENS)
        has_shared = any(t in low for t in SHARED_POSITIVE_TOKENS)
        full_excluded = any(t in low for t in EXCLUDE_SHARED_FULL_TOKENS)

        # "shared" is preferred, but a clearly named source-state+transition
        # variant is also surfaced if it is not action-specific/separate.
        if has_state and has_transition and (has_shared or "full" in low) and not full_excluded:
            shared_transition.append(sig)

    return {
        "all_signatures": sorted(set(all_signatures)),
        "SOURCE_STATE_BINDING_CANDIDATES": sorted(set(source_state)),
        "SHARED_TRANSITION_BINDING_CANDIDATES": sorted(set(shared_transition)),
        "rule": (
            "NAME_ONLY semantic surfacing; no performance-based selection. "
            "Ambiguous/absent candidates require a frozen lineage amendment."
        ),
    }


def _records_from_groupby(df: pd.DataFrame, cols: List[str]) -> List[Dict[str, Any]]:
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return []
    g = (
        df.groupby(cols, dropna=False)
        .size()
        .reset_index(name="rows")
        .sort_values(cols, kind="stable")
    )
    out = []
    for rec in g.to_dict(orient="records"):
        clean = {}
        for k, v in rec.items():
            if pd.isna(v):
                clean[k] = None
            elif hasattr(v, "item"):
                clean[k] = v.item()
            else:
                clean[k] = v
        out.append(clean)
    return out


def _value_counts_records(s: pd.Series, col: str) -> List[Dict[str, Any]]:
    vc = s.fillna("<NA>").astype(str).value_counts(dropna=False, sort=False)
    return [{col: str(k), "rows": int(v)} for k, v in vc.items()]


def _base_key_integrity(
    df: pd.DataFrame,
    signature_cols: List[str],
    key_cols: List[str],
) -> Dict[str, Any]:
    """
    Structural check only:
    - within each immutable predictor signature/split/fold, is there exactly
      one score per [sample_id, model_state_id, action]?
    - are labels invariant for the same physical modelcase-action key?

    This does not compute any scientific performance metric.
    """
    sig = [c for c in signature_cols if c in df.columns]
    key = [c for c in key_cols if c in df.columns]

    if not key or "score" not in df.columns:
        return {"available": False, "reason": "required key/score columns absent"}

    full_key = sig + key
    dup_rows = int(df.duplicated(full_key, keep=False).sum()) if full_key else None

    label_col = "y_harm" if "y_harm" in df.columns else None
    label_invariant = None
    max_label_nunique = None
    if label_col:
        nuniq = df.groupby(key, dropna=False)[label_col].nunique(dropna=False)
        max_label_nunique = int(nuniq.max()) if len(nuniq) else 0
        label_invariant = bool(max_label_nunique <= 1)

    # Count base rows per immutable predictor signature. Expected cardinality
    # is diagnostic only; no "best" signature is chosen.
    per_sig = []
    if sig:
        tmp = (
            df.groupby(sig, dropna=False)
            .agg(
                rows=("score", "size"),
                unique_base_keys=("score", lambda x: 0),  # overwritten below
            )
            .reset_index()
        )
        # nunique over a MultiIndex cannot be expressed cleanly in agg above.
        counts = (
            df[sig + key]
            .drop_duplicates()
            .groupby(sig, dropna=False)
            .size()
            .reset_index(name="unique_base_keys")
        )
        tmp = tmp.drop(columns=["unique_base_keys"]).merge(counts, on=sig, how="left")
        per_sig = [
            {
                **{k: (None if pd.isna(v) else (v.item() if hasattr(v, "item") else v))
                   for k, v in rec.items()}
            }
            for rec in tmp.to_dict(orient="records")
        ]

    return {
        "available": True,
        "signature_columns": sig,
        "base_key_columns": key,
        "duplicate_rows_within_signature_and_key": dup_rows,
        "label_column": label_col,
        "label_invariant_across_repeated_predictions": label_invariant,
        "max_label_nunique_per_base_key": max_label_nunique,
        "per_signature_structure": per_sig,
    }


def inspect_oof_lineage(path: Path, kind: str) -> Dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "exists": False}

    hdr = pd.read_csv(path, nrows=0, low_memory=False)
    cols = list(map(str, hdr.columns))

    wanted = [
        c for c in [
            "split_seed", "fold", "model", "evaluation_type", "direction",
            "sample_id", "model_state_id", "model_family", "action",
            "y_harm", "score",
        ] if c in cols
    ]
    df = pd.read_csv(path, usecols=wanted, low_memory=False)

    value_counts = {}
    for c in ["model", "evaluation_type", "direction", "split_seed", "fold"]:
        if c in df.columns:
            value_counts[c] = _value_counts_records(df[c], c)

    if kind == "R32A1":
        combo_cols = ["model", "evaluation_type", "direction", "split_seed"]
    else:
        combo_cols = ["model", "direction", "split_seed"]

    combo_counts = _records_from_groupby(df, combo_cols)

    # One semantic signature per model/evaluation/direction combination.
    signature_cols = [c for c in ["model", "evaluation_type", "direction"] if c in df.columns]
    signatures = _records_from_groupby(df, signature_cols)
    semantic = _semantic_signature_candidates(signatures)

    integrity = _base_key_integrity(
        df,
        signature_cols=["split_seed", "fold", *signature_cols],
        key_cols=["sample_id", "model_state_id", "action"],
    )

    harm_values = []
    if "y_harm" in df.columns:
        harm_values = _value_counts_records(df["y_harm"], "y_harm")

    return {
        "path": str(path),
        "exists": True,
        "sha256": sha256_file(path),
        "rows": int(len(df)),
        "columns": cols,
        "value_counts": value_counts,
        "combination_counts": combo_counts,
        "semantic_name_binding": semantic,
        "integrity": integrity,
        "y_harm_value_counts": harm_values,
    }


def inspect_modelcase_outcome_consistency() -> Dict[str, Any]:
    if not R31B3_MODELCASE.is_file():
        return {"path": str(R31B3_MODELCASE), "exists": False}

    hdr = pd.read_csv(R31B3_MODELCASE, nrows=0, low_memory=False)
    cols = list(map(str, hdr.columns))
    use = [
        c for c in ["sample_id", "model_state_id", "action", "r31b3_harm_label"]
        if c in cols
    ]
    d = pd.read_csv(R31B3_MODELCASE, usecols=use, low_memory=False)

    key = [c for c in ["sample_id", "model_state_id", "action"] if c in d.columns]
    duplicate_keys = int(d.duplicated(key, keep=False).sum()) if key else None

    return {
        "path": str(R31B3_MODELCASE),
        "exists": True,
        "sha256": sha256_file(R31B3_MODELCASE),
        "rows": int(len(d)),
        "key_columns": key,
        "duplicate_key_rows": duplicate_keys,
        "harm_column": "r31b3_harm_label" if "r31b3_harm_label" in d.columns else None,
        "harm_value_counts": (
            _value_counts_records(d["r31b3_harm_label"], "r31b3_harm_label")
            if "r31b3_harm_label" in d.columns else []
        ),
    }


def _print_count_block(title: str, rows: List[Dict[str, Any]]) -> None:
    print(title)
    if not rows:
        print("  <none>")
        return
    for rec in rows:
        print(" ", rec)


def build_loao_lineage_audit() -> Dict[str, Any]:
    r32 = inspect_oof_lineage(R32A1_OOF, "R32A1")
    r31 = inspect_oof_lineage(R31B3_OOF, "R31B3")
    modelcase = inspect_modelcase_outcome_consistency()

    # R32A1 is the preferred exact frozen comparison execution asset because
    # it explicitly contains evaluation_type in addition to model/direction.
    # R31B3 is retained as provenance corroboration.
    source_candidates = []
    shared_candidates = []
    if r32.get("exists"):
        source_candidates = r32["semantic_name_binding"]["SOURCE_STATE_BINDING_CANDIDATES"]
        shared_candidates = r32["semantic_name_binding"]["SHARED_TRANSITION_BINDING_CANDIDATES"]

    # If R32 labels are not self-describing, do NOT silently choose from R31
    # by performance. Surface R31 candidates separately for a manual/frozen
    # lineage amendment.
    source_r31 = (
        r31.get("semantic_name_binding", {}).get("SOURCE_STATE_BINDING_CANDIDATES", [])
        if r31.get("exists") else []
    )
    shared_r31 = (
        r31.get("semantic_name_binding", {}).get("SHARED_TRANSITION_BINDING_CANDIDATES", [])
        if r31.get("exists") else []
    )

    enough_name_binding = (
        len(source_candidates) == 1
        and len(shared_candidates) == 1
        and source_candidates[0] != shared_candidates[0]
    )

    structural_ok = bool(
        r32.get("exists")
        and r32.get("integrity", {}).get("duplicate_rows_within_signature_and_key") == 0
        and r32.get("integrity", {}).get("label_invariant_across_repeated_predictions") is True
        and modelcase.get("duplicate_key_rows") == 0
    )

    if enough_name_binding and structural_ok:
        route = "READY_FOR_P08_LOAO_BINDING_AMENDMENT"
    else:
        route = "STOP_NEEDS_EXPLICIT_FROZEN_LINEAGE_MAPPING"

    return {
        "route": route,
        "R32A1": r32,
        "R31B3": r31,
        "R31B3_modelcase": modelcase,
        "SOURCE_STATE_BINDING_CANDIDATES": source_candidates,
        "SHARED_TRANSITION_BINDING_CANDIDATES": shared_candidates,
        "R31B3_SOURCE_STATE_BINDING_CANDIDATES": source_r31,
        "R31B3_SHARED_TRANSITION_BINDING_CANDIDATES": shared_r31,
        "name_binding_unique_in_R32A1": enough_name_binding,
        "structural_integrity_pass": structural_ok,
        "selection_rule": (
            "Freeze binding from immutable semantic labels/lineage only. "
            "Never choose a variant because its AUROC/AUPRC matches a historical result."
        ),
    }

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
    "y_harm", "r31b3_harm_label",
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

    representation = first_alias(
        cols,
        ["representation", "method", "variant", "feature_set", "representation_name"]
    )

    generic_score = hits(
        lambda x: (
            ("score" in x or "risk" in x or "prob" in x or "prediction" in x)
            and not any(
                t in x for t in [
                    "auroc", "auprc", "precision", "recall", "threshold",
                    "source_dice", "action_dice", "delta_dice"
                ]
            )
        )
    )

    fold_cols = hits(lambda x: "fold" in x)
    heldout_cols = hits(lambda x: "held" in x and "action" in x)

    return {
        "source_q66_score_candidates": source_q66,
        "shared_full_score_candidates": shared_full,
        "simple_change_score_candidates": simple_change,
        "generic_score_candidates": generic_score,
        "representation_col": representation,
        "fold_candidates": fold_cols,
        "heldout_action_candidates": heldout_cols,
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


def _is_allowed_neopolyp_loao_path(path_str: str) -> bool:
    p = Path(path_str)
    try:
        rel = p.resolve().relative_to(ROOT.resolve())
        top = rel.parts[0] if rel.parts else ""
    except Exception:
        top = p.parts[-2] if len(p.parts) >= 2 else ""
    return top.startswith(NEOPOLYP_LOAO_ALLOWED_PREFIXES)


def choose_loao_candidates(cands: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Bind ONLY NeoPolyp R31/R32 LOAO lineage.

    Important: P08A v1 incorrectly allowed any path containing "loao", which
    admitted B6_P01A MRI SOURCE_LOAO_MATCHED_SCORES.csv. This fix makes such
    a cross-task false-positive impossible.
    """
    direct = []
    long_format = []
    join_score = []
    join_outcome = []

    for r in cands:
        if not _is_allowed_neopolyp_loao_path(r["path"]):
            continue

        path_low = r["path"].lower()
        # Restrict further to known three-action / LOAO development stages.
        stage_ok = any(
            t in path_low
            for t in [
                "r31a_action_conditional",
                "r31b3_first_800case",
                "r32a1_three_action_loao",
                "r32b1_faithful_common_baseline",
                "r32b3_matched_three_action",
            ]
        )
        if not stage_ok:
            continue

        has_id = r["id_col"] is not None
        has_action = r["action_col"] is not None
        has_label = r["harm_col"] is not None or r["delta_col"] is not None
        has_source = bool(r["semantic"]["source_q66_score_candidates"])
        has_full = bool(r["semantic"]["shared_full_score_candidates"])
        has_rep = r["semantic"]["representation_col"] is not None
        has_generic_score = bool(r["semantic"]["generic_score_candidates"])

        if has_id and has_action and has_label and has_source and has_full:
            direct.append(r)
        if has_id and has_action and has_label and has_rep and has_generic_score:
            long_format.append(r)
        if has_id and has_action and (has_source or has_full or (has_rep and has_generic_score)):
            join_score.append(r)
        if has_id and has_action and has_label:
            join_outcome.append(r)

    if direct:
        route = "DIRECT_WIDE_NEOPOLYP_LOAO_PANEL"
    elif long_format:
        route = "DIRECT_LONG_NEOPOLYP_LOAO_PANEL"
    elif join_score and join_outcome:
        route = "JOIN_NEOPOLYP_LOAO_SCORE_AND_OUTCOME_PANELS"
    else:
        route = "STOP_NEOPOLYP_LOAO_ROW_LEVEL_BINDING_INCOMPLETE"

    # Exact known-file diagnostics: always inspect full schemas if present.
    exact_files = []
    by_path = {str(Path(r["path"]).resolve()).lower(): r for r in cands}
    for p in NEOPOLYP_EXACT_FILES:
        key = str(p.resolve()).lower()
        exact_files.append(
            by_path.get(
                key,
                {
                    "path": str(p),
                    "exists": p.is_file(),
                    "not_discovered": True,
                },
            )
        )

    return {
        "route": route,
        "direct_wide": direct[:10],
        "direct_long": long_format[:10],
        "score_candidates": join_score[:20],
        "outcome_candidates": join_outcome[:20],
        "exact_file_diagnostics": exact_files,
        "cross_task_exclusion": {
            "MRI_B6_P01A_allowed": False,
            "rule": "NeoPolyp LOAO may bind only from R31/R32 lineage."
        },
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
        assert first_alias(["sample_id", "y_harm"], HARM_ALIASES) == "y_harm"
        assert first_alias(["sample_id", "r31b3_harm_label"], HARM_ALIASES) == "r31b3_harm_label"
        sem_bind = _semantic_signature_candidates([
            {"model": "SOURCE_STATE_Q66", "evaluation_type": "LOAO", "direction": "HARM"},
            {"model": "SHARED_SOURCE_STATE_FULL_TRANSITION", "evaluation_type": "LOAO", "direction": "HARM"},
            {"model": "SEPARATE_ACTION_SOURCE_STATE_FULL_TRANSITION", "evaluation_type": "REFERENCE", "direction": "HARM"},
        ])
        assert len(sem_bind["SOURCE_STATE_BINDING_CANDIDATES"]) == 1
        assert len(sem_bind["SHARED_TRANSITION_BINDING_CANDIDATES"]) == 1
        assert _is_allowed_neopolyp_loao_path(
            str(ROOT / "R32A1_three_action_loao_comparison_execution_v1_fix1" / "x.csv")
        )
        assert not _is_allowed_neopolyp_loao_path(
            str(ROOT / "B6_P01A_mri_matched_representation_attribution_v1" / "B6_P01A_SOURCE_LOAO_MATCHED_SCORES.csv")
        )
        print("SELF_TEST_CROSS_TASK_EXCLUSION=PASS")
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
            f"genericScore={r['semantic']['generic_score_candidates']} "
            f"rep={r['semantic']['representation_col']} "
            f"fold={r['semantic']['fold_candidates']} "
            f"heldout={r['semantic']['heldout_action_candidates']} "
            f"simple={r['semantic']['simple_change_score_candidates']} "
            f"grouping={r['grouping_cols']}"
        )

    print("\n[2/4] NeoPolyp LOAO paired-increment binding")
    loao = choose_loao_candidates(cands)
    print("LOAO_ROUTE =", loao["route"])
    print("MRI_B6_P01A_ALLOWED =", loao["cross_task_exclusion"]["MRI_B6_P01A_allowed"])

    for label in ["direct_wide", "direct_long", "score_candidates", "outcome_candidates"]:
        print(f"\n{label}:")
        for r in loao[label][:10]:
            print(" ", r["path"])
            print("   rows/ids/actions =", r.get("rows"), r.get("unique_ids"), r.get("unique_actions"))
            print("   columns =", r["columns"])
            print("   semantic =", r["semantic"])

    print("\nEXACT R31/R32 FILE DIAGNOSTICS")
    for r in loao["exact_file_diagnostics"]:
        print("\n ", r.get("path"))
        if r.get("not_discovered"):
            print("   exists =", r.get("exists"), "not_discovered=True")
        else:
            print("   rows/ids/actions/states =",
                  r.get("rows"), r.get("unique_ids"),
                  r.get("unique_actions"), r.get("unique_states"))
            print("   id/action/harm/delta/state =",
                  r.get("id_col"), r.get("action_col"),
                  r.get("harm_col"), r.get("delta_col"), r.get("state_col"))
            print("   semantic =", r.get("semantic"))
            print("   columns =", r.get("columns"))

    print("\n[2b/4] Exact NeoPolyp R31/R32 OOF categorical lineage audit")
    lineage = build_loao_lineage_audit()

    r32 = lineage["R32A1"]
    if r32.get("exists"):
        _print_count_block(
            "R32A1_MODEL_VALUE_COUNTS",
            r32["value_counts"].get("model", []),
        )
        _print_count_block(
            "R32A1_EVALUATION_TYPE_VALUE_COUNTS",
            r32["value_counts"].get("evaluation_type", []),
        )
        _print_count_block(
            "R32A1_DIRECTION_VALUE_COUNTS",
            r32["value_counts"].get("direction", []),
        )
        _print_count_block(
            "R32A1_SPLIT_SEED_VALUE_COUNTS",
            r32["value_counts"].get("split_seed", []),
        )
        _print_count_block(
            "R32A1_FOLD_VALUE_COUNTS",
            r32["value_counts"].get("fold", []),
        )
        _print_count_block(
            "R32A1_MODEL_EVAL_DIRECTION_SEED_COUNTS",
            r32["combination_counts"],
        )
        print("R32A1_INTEGRITY =", r32["integrity"])

    r31 = lineage["R31B3"]
    if r31.get("exists"):
        _print_count_block(
            "R31B3_MODEL_VALUE_COUNTS",
            r31["value_counts"].get("model", []),
        )
        _print_count_block(
            "R31B3_DIRECTION_VALUE_COUNTS",
            r31["value_counts"].get("direction", []),
        )
        _print_count_block(
            "R31B3_SPLIT_SEED_VALUE_COUNTS",
            r31["value_counts"].get("split_seed", []),
        )
        _print_count_block(
            "R31B3_FOLD_VALUE_COUNTS",
            r31["value_counts"].get("fold", []),
        )
        _print_count_block(
            "R31B3_MODEL_DIRECTION_SEED_COUNTS",
            r31["combination_counts"],
        )
        print("R31B3_INTEGRITY =", r31["integrity"])

    print("SOURCE_STATE_BINDING_CANDIDATES =",
          lineage["SOURCE_STATE_BINDING_CANDIDATES"])
    print("SHARED_TRANSITION_BINDING_CANDIDATES =",
          lineage["SHARED_TRANSITION_BINDING_CANDIDATES"])
    print("R31B3_SOURCE_STATE_BINDING_CANDIDATES =",
          lineage["R31B3_SOURCE_STATE_BINDING_CANDIDATES"])
    print("R31B3_SHARED_TRANSITION_BINDING_CANDIDATES =",
          lineage["R31B3_SHARED_TRANSITION_BINDING_CANDIDATES"])
    print("LOAO_BINDING_ROUTE =", lineage["route"])
    print("R31B3_MODELCASE_INTEGRITY =", lineage["R31B3_modelcase"])

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
            "Use _fix3 only for implementation repair."
        )
    OUT_DIR.mkdir(parents=True, exist_ok=False)

    status = (
        "PASS"
        if lineage["route"] == "READY_FOR_P08_LOAO_BINDING_AMENDMENT"
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
        "NeoPolyp_LOAO_lineage_audit": lineage,
        "event_support": events,
        "PolypGen_grouping": grouping,
        "optional_NeoPolyp_simple_change": optional,
        "next": (
            "Freeze a P08 LOAO binding amendment from the exact immutable R32A1/R31B3 labels, then run P08B."
            if status == "PASS"
            else "Do not compute final closure metrics; inspect the printed R32A1/R31B3 categorical lineage and freeze an explicit binding amendment only if provenance supports it."
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
