#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R11B_candidate_action_semantic_lineage_audit_fix1.py

Targeted semantic/lineage audit after R11A.

R11A found three candidates requiring review:
    TENT
    BN_AFFINE
    PL_PSEUDOLABEL

This stage determines whether BN_AFFINE and PL_PSEUDOLABEL are genuinely
distinct, reusable, 3-family TTA actions or merely implementation details /
single-family prototypes.

NO inference.
NO TTA execution.
NO GT access.
NO PolypGen performance/score read.
NO target-performance-based action selection.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from pathlib import Path

import pandas as pd


VERSION = "2026-08-26-Q1-R11B-v1-fix1"
BUILD = "Q1_R11B_CANDIDATE_ACTION_SEMANTIC_LINEAGE_AUDIT_FIX1"

ROOT = Path(r"F:\MEDSEG_SAFETTA")
CODE = ROOT / "code"
OUT = ROOT / "outputs"

R11A_DIR = OUT / "Q1_R11A_tta_action_inventory_and_lineage_audit_fix1_v1"
R11A_LOCK = R11A_DIR / "R11A_ACTION_INVENTORY_LOCK.json"

R05D3 = CODE / "Q1_R05D3_neopolyp_source_a1_prediction_lock_v1_fix1.py"
R03 = CODE / "Q1_R03_nine_state_heterogeneous_source_utility_asset_v1_fix2.py"
S01D1 = CODE / "S01_D1_tta_feasibility_locked_evaluation_v1.py"
S03B = CODE / "S03_B_build_source_side_safettta_gate_v1.py"
S05C = CODE / "S05_C_build_deeplab_source_side_safettta_gate_v1.py"

POLYPGEN_LOCK = (
    OUT
    / "Q1_R10L3C_polypgen_gt_reveal_prospective_external_evaluation_fix1_v1"
    / "R10L3C_POLYPGEN_PROSPECTIVE_EXTERNAL_EVALUATION_LOCK.json"
)
EXPECTED_POLYPGEN_DECISION = (
    "POLYPGEN_PROSPECTIVE_EXTERNAL_EVALUATION_COMPLETE_NO_RETUNING"
)

OUTPUT_DIR = OUT / "Q1_R11B_candidate_action_semantic_lineage_audit_fix1_v1"

FAMILIES = ("PraNet", "DeepLabV3-R50", "SegFormer-B0")
TENT_TOKENS = ("tent", "entropy", "configure_tent", "singleton_safe_tent")
PL_TOKENS = ("pseudo", "self_train", "selftraining", "testfit")


def sha256_file(path: Path, chunk=8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def parse_defs(path: Path):
    text = path.read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(text)
    lines = text.splitlines()
    out = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = int(getattr(node, "lineno", 1))
            end = int(getattr(node, "end_lineno", start))
            body = "\n".join(lines[start - 1:end])
            out.append((node.name, type(node).__name__, start, end, body))
    return out


def detect_families(text: str):
    fams = set()
    if re.search(r"\bpranet\b", text, flags=re.I):
        fams.add("PraNet")
    if re.search(r"deeplab", text, flags=re.I):
        fams.add("DeepLabV3-R50")
    if re.search(r"segformer|mit[-_ ]?b0", text, flags=re.I):
        fams.add("SegFormer-B0")
    return sorted(fams)


def semantic_rows(paths):
    rows = []
    for path in paths:
        for name, kind, start, end, body in parse_defs(path):
            low = (name + "\n" + body).lower()
            tent = any(tok in low for tok in TENT_TOKENS)
            pl = any(tok in low for tok in PL_TOKENS)
            bn = (
                ("batchnorm" in low or "batchnorm2d" in low or "bn" in name.lower())
                and (
                    "requires_grad" in low
                    or ".weight" in low
                    or ".bias" in low
                    or "affine" in low
                )
            )
            if not (tent or pl or bn):
                continue
            rows.append({
                "path": str(path),
                "sha256": sha256_file(path),
                "function_or_class": name,
                "kind": kind,
                "start_line": start,
                "end_line": end,
                "families_in_body": ";".join(detect_families(body)),
                "tent_semantic": int(tent),
                "bn_affine_semantic": int(bn),
                "pseudolabel_semantic": int(pl),
                "target_label_reference": int(
                    bool(re.search(
                        r"\by_true\b|\btarget_label\b|\bground[_ ]?truth\b",
                        body,
                        flags=re.I,
                    ))
                ),
            })
    return pd.DataFrame(rows)


def audit_tent():
    text = R05D3.read_text(encoding="utf-8", errors="replace")
    defs = {x[0] for x in parse_defs(R05D3)}
    required = {
        "run_pranet_state",
        "run_deeplab_state",
        "run_segformer_state",
    }
    all3 = required.issubset(defs)
    hp = (
        re.search(r"TENT_LR\s*=\s*1e-3", text) is not None
        and re.search(r"TENT_WEIGHT_DECAY\s*=\s*0\.0", text) is not None
        and re.search(r"TENT_STEPS\s*=\s*1", text) is not None
    )
    return {
        "action": "TENT",
        "distinct_action": True,
        "all_three_families_explicit": all3,
        "label_free_adaptation_supported": True,
        "hyperparameters_frozen": hp,
        "panel_status": "KEEP_REFERENCE_ACTION" if all3 and hp else "STOP_TENT_LINEAGE_INCOMPLETE",
        "reason": "Authoritative R05D3 implements A1_TENT_1STEP for all three frozen families.",
    }


def audit_bn(df):
    bn = df[df["bn_affine_semantic"] == 1].copy()
    standalone = bn[bn["tent_semantic"] == 0].copy()
    fams = set()
    for x in standalone["families_in_body"].astype(str):
        fams.update([v for v in x.split(";") if v])

    if standalone.empty:
        status = "EXCLUDE_NOT_DISTINCT_FROM_TENT"
        reason = (
            "All discovered BN-affine semantics occur inside TENT-related routines; "
            "BN_AFFINE is not a separate frozen action."
        )
    elif fams != set(FAMILIES):
        status = "EXCLUDE_DISTINCT_IMPLEMENTATION_NOT_3_FAMILY"
        reason = (
            "Standalone BN-affine semantics exist, but explicit reusable coverage "
            "of all three frozen model families is incomplete."
        )
    else:
        status = "CANDIDATE_DISTINCT_REQUIRES_ACTION_LOCK"
        reason = (
            "Standalone BN-affine semantics with apparent 3-family coverage exist; "
            "exact update semantics and hyperparameters still require freezing."
        )

    return {
        "action": "BN_AFFINE",
        "distinct_action": not standalone.empty,
        "all_three_families_explicit": fams == set(FAMILIES),
        "label_free_adaptation_supported": not standalone.empty,
        "hyperparameters_frozen": False,
        "panel_status": status,
        "reason": reason,
    }


def audit_pl(df):
    pl = df[df["pseudolabel_semantic"] == 1].copy()
    fams = set()
    for x in pl["families_in_body"].astype(str):
        fams.update([v for v in x.split(";") if v])

    label_free = int(pl["target_label_reference"].sum()) == 0 if len(pl) else False

    if pl.empty:
        status = "EXCLUDE_NO_IMPLEMENTATION"
        reason = "No runnable pseudolabel/self-training semantics found."
    elif fams != set(FAMILIES):
        status = "EXCLUDE_SINGLE_OR_INCOMPLETE_FAMILY_IMPLEMENTATION"
        reason = (
            "Pseudolabel/self-training is implemented only for a subset of the "
            "three frozen model families; porting now would be a new method branch."
        )
    elif not label_free:
        status = "EXCLUDE_TARGET_LABEL_REFERENCE"
        reason = "Target-label references detected in pseudolabel implementation."
    else:
        status = "CANDIDATE_DISTINCT_REQUIRES_ACTION_LOCK"
        reason = (
            "Pseudolabel implementation appears label-free with 3-family coverage; "
            "pseudo-target rule and hyperparameters still require freezing."
        )

    return {
        "action": "PL_PSEUDOLABEL",
        "distinct_action": True,
        "all_three_families_explicit": fams == set(FAMILIES),
        "label_free_adaptation_supported": label_free,
        "hyperparameters_frozen": False,
        "panel_status": status,
        "reason": reason,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = ap.parse_args()

    print("===== R11B CANDIDATE ACTION SEMANTIC + LINEAGE AUDIT FIX1 =====")
    print("MODEL INFERENCE: NONE")
    print("TTA EXECUTION: NONE")
    print("GT ACCESS: NONE")
    print("POLYPGEN PERFORMANCE READ: NONE")
    print("POLYPGEN SCORE READ: NONE")
    print("TARGET-OUTCOME-BASED ACTION SELECTION: NO")
    print()

    required_paths = [R11A_LOCK, R05D3, R03, S01D1, S03B, S05C, POLYPGEN_LOCK]
    for p in required_paths:
        print("exists:", p.exists(), p)
        if not p.exists():
            raise FileNotFoundError(p)

    r11a = json.loads(R11A_LOCK.read_text(encoding="utf-8"))
    if r11a.get("decision") != "TTA_ACTION_INVENTORY_FROZEN_FOR_R11B_PANEL_SELECTION":
        raise RuntimeError("Unexpected R11A decision.")

    ext = json.loads(POLYPGEN_LOCK.read_text(encoding="utf-8"))
    if ext.get("decision") != EXPECTED_POLYPGEN_DECISION:
        raise RuntimeError("Unexpected PolypGen lock decision.")

    df = semantic_rows([R05D3, R03, S01D1, S03B, S05C])
    if df.empty:
        raise RuntimeError("No candidate semantic definitions found.")

    print("\n===== FUNCTION-LEVEL SEMANTIC EVIDENCE =====")
    print(df.to_string(index=False))

    result_df = pd.DataFrame([
        audit_tent(),
        audit_bn(df),
        audit_pl(df),
    ])

    print("\n===== CANDIDATE ACTION DECISIONS =====")
    print(result_df.to_string(index=False))

    distinct_candidates = result_df[
        result_df["panel_status"] == "CANDIDATE_DISTINCT_REQUIRES_ACTION_LOCK"
    ]["action"].astype(str).tolist()

    excluded = result_df[
        result_df["panel_status"].str.startswith("EXCLUDE")
    ][["action", "panel_status", "reason"]].to_dict("records")

    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)

    semantic_path = args.output_dir / "R11B_function_level_semantic_evidence.csv"
    decision_path = args.output_dir / "R11B_candidate_action_decisions.csv"
    df.to_csv(semantic_path, index=False)
    result_df.to_csv(decision_path, index=False)

    lock = {
        "decision": "R11B_CANDIDATE_ACTION_SEMANTICS_FROZEN",
        "script_version": VERSION,
        "build": BUILD,
        "reference_action_ready": ["TENT"],
        "distinct_candidates_requiring_action_lock": distinct_candidates,
        "actions_not_admitted": excluded,
        "polypgen_used_for_selection": False,
        "polypgen_performance_read": False,
        "target_outcomes_read": False,
        "model_inference_run": False,
        "tta_run": False,
        "next_rule": (
            "Do not execute a cross-action panel unless at least one non-TENT "
            "action passes explicit 3-family label-free semantic lineage."
        ),
    }

    lock_path = args.output_dir / "R11B_CANDIDATE_ACTION_LOCK.json"
    lock_path.write_text(
        json.dumps(lock, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n===== FINAL DECISION =====")
    print("DECISION:", lock["decision"])
    print("Distinct candidates requiring action lock:", distinct_candidates)
    print("Excluded:", [x["action"] for x in excluded])
    print("POLYPGEN USED FOR SELECTION: NO")
    print("TARGET OUTCOMES READ: NO")
    print("MODEL INFERENCE RUN: NO")
    print("TTA RUN: NO")
    print(
        "R11C STATUS:",
        "ACTION-LOCK REQUIRED" if distinct_candidates
        else "NOT AUTHORIZED — no distinct 3-family action has passed lineage.",
    )
    print("PASS")


if __name__ == "__main__":
    main()
