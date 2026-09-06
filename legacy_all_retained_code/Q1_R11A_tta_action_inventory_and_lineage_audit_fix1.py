#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R11A_tta_action_inventory_and_lineage_audit_fix1.py

Purpose
-------
Prepare the next Q1 experiment line:
Cross-TTA-Action Safety Generalization.

This stage does NOT run adaptation. It inventories authoritative TTA action
implementations already present in the project BEFORE any new action panel is
chosen.

Why this audit exists
---------------------
PolypGen is now a locked confirmatory external cohort and must never be used to
choose or tune new actions. Therefore R11A searches only historical project
code/protocol/output artifacts and freezes an action inventory before R11B.

The audit answers:
1) Which label-free TTA actions are actually implemented in this project?
2) Which scripts are authoritative / frozen enough to reuse?
3) Which model families are explicitly covered by each implementation?
4) Which actions expose frozen hyperparameters in code/protocol text?
5) Which candidates are plausible for a fair 3-family cross-action panel?

No model inference.
No GT access.
No PolypGen score read.
No target performance read.
No action selection from target outcomes.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm


VERSION = "2026-08-26-Q1-R11A-v1-fix1"
BUILD = "Q1_R11A_TTA_ACTION_INVENTORY_AND_LINEAGE_AUDIT_FIX1"

ROOT = Path(r"F:\MEDSEG_SAFETTA")
CODE_ROOT = ROOT / "code"
OUTPUT_ROOT = ROOT / "outputs"

OUTPUT_DIR = (
    OUTPUT_ROOT
    / "Q1_R11A_tta_action_inventory_and_lineage_audit_fix1_v1"
)

# Authoritative currently frozen action implementation.
R05D3_SCRIPT = (
    CODE_ROOT
    / "Q1_R05D3_neopolyp_source_a1_prediction_lock_v1_fix1.py"
)

# PolypGen lock is checked only to ensure the external cohort already exists
# and is not used in R11A/R11B development.
POLYPGEN_EXTERNAL_LOCK = (
    OUTPUT_ROOT
    / "Q1_R10L3C_polypgen_gt_reveal_prospective_external_evaluation_fix1_v1"
    / "R10L3C_POLYPGEN_PROSPECTIVE_EXTERNAL_EVALUATION_LOCK.json"
)

EXPECTED_EXTERNAL_DECISION = (
    "POLYPGEN_PROSPECTIVE_EXTERNAL_EVALUATION_COMPLETE_NO_RETUNING"
)

TEXT_EXTS = {
    ".py", ".md", ".txt", ".json", ".yaml", ".yml", ".log"
}

EXCLUDE_DIR_NAMES = {
    "__pycache__", ".git", ".ipynb_checkpoints", "node_modules",
    "Q1_R10L1_polypgen_official_singleframe_asset_audit_fix2_v1",
    "Q1_R10L1A_external_dataset_identity_audit_fix1_v1",
    "Q1_R10L1B_polypgen_unique_static_manifest_lock_fix3_v1",
    "Q1_R10L2A_source_training_lineage_discovery_audit_fix1_v1",
    "Q1_R10L2B_authoritative_s01_source_training_overlap_lock_fix1_v1",
    "Q1_R10L3A_polypgen_source_a1_prediction_lock_fix2_v1",
    "Q1_R10L3B_polypgen_frozen_safety_score_lock_fix1_v1",
    "Q1_R10L3C_polypgen_gt_reveal_prospective_external_evaluation_fix1_v1",
    "Q1_R10L3D_polypgen_external_generalization_diagnostic_freeze_fix1_v1",
}

# Search vocabulary. This does not imply that every keyword corresponds to a
# valid action; R11A only inventories what is really present.
ACTION_PATTERNS = {
    "TENT": [
        r"\btent\b",
        r"test[-_ ]time entropy minim",
        r"entropy minimization",
    ],
    "BN_AFFINE": [
        r"bn[-_ ]?affine",
        r"batchnorm.*affine",
        r"batch norm.*affine",
    ],
    "ADABN_BN_STATS": [
        r"\badabn\b",
        r"bn[-_ ]?stat",
        r"batchnorm.*running",
        r"batch norm.*running",
        r"running_mean",
        r"running_var",
    ],
    "EATA": [
        r"\beata\b",
        r"efficient anti-forgetting",
    ],
    "SAR": [
        r"\bsar\b",
        r"sharpness-aware.*adapt",
    ],
    "COTTA": [
        r"\bcotta\b",
        r"continual test[-_ ]time adaptation",
    ],
    "MEMO": [
        r"\bmemo\b",
        r"marginal entropy minim",
    ],
    "T3A": [
        r"\bt3a\b",
    ],
    "SHOT": [
        r"\bshot\b",
        r"source hypothesis transfer",
    ],
    "PL_PSEUDOLABEL": [
        r"pseudo[-_ ]?label",
        r"self[-_ ]?training",
    ],
    "AUG_CONSISTENCY": [
        r"augmentation.*consisten",
        r"consisten.*augmentation",
        r"test[-_ ]time augment",
    ],
}

MODEL_PATTERNS = {
    "PraNet": [
        r"\bpranet\b",
        r"pra_net",
    ],
    "DeepLabV3-R50": [
        r"deeplabv3",
        r"deeplab",
        r"resnet50",
    ],
    "SegFormer-B0": [
        r"segformer",
        r"mit[-_ ]?b0",
    ],
}

HYPERPARAM_PATTERNS = {
    "learning_rate": [
        r"\blr\b\s*[:=]\s*([0-9.eE+-]+)",
        r"learning[_ ]rate\s*[:=]\s*([0-9.eE+-]+)",
    ],
    "steps": [
        r"\bsteps?\b\s*[:=]\s*([0-9]+)",
        r"adapt[_ ]steps?\s*[:=]\s*([0-9]+)",
    ],
    "weight_decay": [
        r"weight[_ ]decay\s*[:=]\s*([0-9.eE+-]+)",
    ],
    "optimizer": [
        r"optimizer\s*[:=]\s*[\"']?([A-Za-z0-9_]+)",
        r"\b(torch\.optim\.[A-Za-z0-9_]+)",
    ],
}

RISK_PATTERNS = {
    "target_gt_reference": [
        r"\bgt_path\b",
        r"\bground truth\b",
        r"\bharm_label\b",
        r"\bdelta_dice\b",
    ],
    "polypgen_reference": [
        r"\bpolypgen\b",
    ],
    "explicit_tuning_language": [
        r"\bgrid search\b",
        r"\bhyperparameter sweep\b",
        r"\btune\b",
        r"\boptimization on target\b",
        r"\bselect.*target\b",
    ],
}

MAX_TEXT_BYTES = 8 * 1024 * 1024
MAX_MATCH_LINES_PER_FILE = 80


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def safe_text(path: Path):
    try:
        if path.stat().st_size > MAX_TEXT_BYTES:
            return None
    except Exception:
        return None

    for enc in ("utf-8", "utf-8-sig", "gb18030", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except Exception:
            continue
    return None


def iter_candidate_files(root: Path):
    for current_root, dirs, files in os.walk(root):
        dirs[:] = [
            d for d in dirs
            if d not in EXCLUDE_DIR_NAMES
            and not d.startswith(".")
        ]
        for name in files:
            p = Path(current_root) / name
            if p.suffix.lower() in TEXT_EXTS:
                yield p


def line_hits(text: str, patterns):
    compiled = [re.compile(p, flags=re.I) for p in patterns]
    hits = []
    for i, line in enumerate(text.splitlines(), start=1):
        if any(rx.search(line) for rx in compiled):
            hits.append((i, line.strip()))
            if len(hits) >= MAX_MATCH_LINES_PER_FILE:
                break
    return hits


def any_match(text: str, patterns):
    return any(
        re.search(p, text, flags=re.I) is not None
        for p in patterns
    )


def extract_hyperparams(text: str):
    out = {}
    for key, patterns in HYPERPARAM_PATTERNS.items():
        values = []
        for p in patterns:
            for m in re.finditer(p, text, flags=re.I):
                val = m.group(1)
                values.append(val)
                if len(values) >= 20:
                    break
            if len(values) >= 20:
                break
        out[key] = sorted(set(values))
    return out


def python_function_names(path: Path):
    if path.suffix.lower() != ".py":
        return []
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except Exception:
        return []

    names = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            n = node.name
            nl = n.lower()
            if any(
                token in nl
                for token in (
                    "adapt", "tent", "tta", "entropy", "bn",
                    "pseudo", "memo", "eata", "sar", "cotta",
                )
            ):
                names.append(n)
    return sorted(set(names))


def classify_file(path: Path, text: str):
    action_hits = {}
    for action, pats in ACTION_PATTERNS.items():
        if any_match(text, pats):
            action_hits[action] = line_hits(text, pats)

    if not action_hits:
        return None

    families = []
    for fam, pats in MODEL_PATTERNS.items():
        if any_match(text, pats):
            families.append(fam)

    risks = {}
    for risk, pats in RISK_PATTERNS.items():
        risks[risk] = any_match(text, pats)

    hp = extract_hyperparams(text)
    funcs = python_function_names(path)

    lower_name = path.name.lower()
    protocolish = any(
        x in lower_name
        for x in ("protocol", "preregister", "freeze", "lock")
    )
    implementationish = (
        path.suffix.lower() == ".py"
        and bool(funcs)
    )

    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "suffix": path.suffix.lower(),
        "actions": sorted(action_hits),
        "families": sorted(families),
        "functions_or_classes": funcs,
        "hyperparams": hp,
        "risk_flags": risks,
        "protocolish": protocolish,
        "implementationish": implementationish,
        "action_hits": action_hits,
    }


def summarize_action(records, action):
    subset = [r for r in records if action in r["actions"]]
    files = len(subset)
    implementation_files = sum(
        int(r["implementationish"])
        for r in subset
    )
    protocol_files = sum(
        int(r["protocolish"])
        for r in subset
    )

    family_union = sorted({
        fam
        for r in subset
        for fam in r["families"]
    })

    explicit_all3_single_file = any(
        set(r["families"]) == set(MODEL_PATTERNS)
        and r["implementationish"]
        for r in subset
    )

    risk_gt = sum(
        int(r["risk_flags"]["target_gt_reference"])
        for r in subset
    )
    risk_polypgen = sum(
        int(r["risk_flags"]["polypgen_reference"])
        for r in subset
    )
    risk_tuning = sum(
        int(r["risk_flags"]["explicit_tuning_language"])
        for r in subset
    )

    hp_values = defaultdict(set)
    for r in subset:
        for k, vals in r["hyperparams"].items():
            hp_values[k].update(vals)

    # Conservative readiness category. R11A does NOT select the final panel.
    if action == "TENT":
        readiness = "AUTHORITATIVE_EXISTING_ACTION_EXPECTED"
    elif (
        implementation_files > 0
        and set(family_union) == set(MODEL_PATTERNS)
    ):
        readiness = "CANDIDATE_REQUIRES_MANUAL_LINEAGE_REVIEW"
    elif implementation_files > 0:
        readiness = "IMPLEMENTED_BUT_FAMILY_COVERAGE_INCOMPLETE"
    else:
        readiness = "TEXT_REFERENCE_ONLY"

    return {
        "action": action,
        "matched_files": files,
        "implementation_files": implementation_files,
        "protocol_or_lock_files": protocol_files,
        "family_union": ";".join(family_union),
        "explicit_all3_single_implementation_file":
            explicit_all3_single_file,
        "gt_reference_files": risk_gt,
        "polypgen_reference_files": risk_polypgen,
        "tuning_language_files": risk_tuning,
        "learning_rate_values":
            ";".join(sorted(hp_values["learning_rate"])),
        "steps_values":
            ";".join(sorted(hp_values["steps"])),
        "weight_decay_values":
            ";".join(sorted(hp_values["weight_decay"])),
        "optimizer_values":
            ";".join(sorted(hp_values["optimizer"])),
        "readiness": readiness,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = ap.parse_args()

    print("===== R11A TTA ACTION INVENTORY + LINEAGE AUDIT FIX1 =====")
    print("STATUS: PRE-EXPERIMENT ACTION INVENTORY")
    print("MODEL INFERENCE: NONE")
    print("TTA EXECUTION: NONE")
    print("GT ACCESS: NONE")
    print("POLYPGEN SCORE READ: NONE")
    print("TARGET PERFORMANCE READ: NONE")
    print("ACTION SELECTION FROM TARGET OUTCOMES: NO")
    print()

    for label, p in {
        "project root": args.root,
        "code root": CODE_ROOT,
        "output root": OUTPUT_ROOT,
        "authoritative R05D3": R05D3_SCRIPT,
        "locked PolypGen external evaluation": POLYPGEN_EXTERNAL_LOCK,
    }.items():
        print(label, "exists:", p.exists(), p)
        if not p.exists():
            raise FileNotFoundError(p)

    external_lock = json.loads(
        POLYPGEN_EXTERNAL_LOCK.read_text(encoding="utf-8")
    )
    if external_lock.get("decision") != EXPECTED_EXTERNAL_DECISION:
        raise RuntimeError(
            "PolypGen external evaluation is not in the expected frozen state."
        )

    print(
        "PolypGen post-GT method modification authorized:",
        external_lock.get(
            "information_boundary",
            {}
        ).get(
            "target_feature_selection",
            False
        ),
        "(must remain False)",
    )
    print(
        "R05D3 authoritative SHA256:",
        sha256_file(R05D3_SCRIPT),
    )

    scan_roots = [CODE_ROOT, OUTPUT_ROOT]
    files = []
    for root in scan_roots:
        files.extend(iter_candidate_files(root))
    files = sorted(set(files), key=lambda x: str(x).lower())

    print("\nCandidate text/code files:", len(files))

    records = []
    detail_rows = []

    for p in tqdm(
        files,
        desc="Scanning historical TTA artifacts",
        unit="file",
        dynamic_ncols=True,
    ):
        text = safe_text(p)
        if text is None:
            continue

        rec = classify_file(p, text)
        if rec is None:
            continue

        records.append(rec)

        for action, hits in rec["action_hits"].items():
            for line_no, snippet in hits:
                detail_rows.append({
                    "action": action,
                    "path": rec["path"],
                    "sha256": rec["sha256"],
                    "line": line_no,
                    "snippet": snippet[:2000],
                    "families": ";".join(rec["families"]),
                    "functions_or_classes":
                        ";".join(rec["functions_or_classes"]),
                    "protocolish": int(rec["protocolish"]),
                    "implementationish": int(rec["implementationish"]),
                    "gt_reference_file": int(
                        rec["risk_flags"]["target_gt_reference"]
                    ),
                    "polypgen_reference_file": int(
                        rec["risk_flags"]["polypgen_reference"]
                    ),
                    "tuning_language_file": int(
                        rec["risk_flags"]["explicit_tuning_language"]
                    ),
                })

    if not records:
        raise RuntimeError("No TTA-related historical artifacts discovered.")

    action_rows = [
        summarize_action(records, action)
        for action in ACTION_PATTERNS
        if any(action in r["actions"] for r in records)
    ]

    summary_df = pd.DataFrame(action_rows).sort_values(
        ["readiness", "action"],
        kind="mergesort",
    ).reset_index(drop=True)

    detail_df = pd.DataFrame(detail_rows)

    file_rows = []
    for r in records:
        file_rows.append({
            "path": r["path"],
            "sha256": r["sha256"],
            "suffix": r["suffix"],
            "actions": ";".join(r["actions"]),
            "families": ";".join(r["families"]),
            "functions_or_classes":
                ";".join(r["functions_or_classes"]),
            "learning_rate_values":
                ";".join(r["hyperparams"]["learning_rate"]),
            "steps_values":
                ";".join(r["hyperparams"]["steps"]),
            "weight_decay_values":
                ";".join(r["hyperparams"]["weight_decay"]),
            "optimizer_values":
                ";".join(r["hyperparams"]["optimizer"]),
            "protocolish": int(r["protocolish"]),
            "implementationish": int(r["implementationish"]),
            "gt_reference_file": int(
                r["risk_flags"]["target_gt_reference"]
            ),
            "polypgen_reference_file": int(
                r["risk_flags"]["polypgen_reference"]
            ),
            "tuning_language_file": int(
                r["risk_flags"]["explicit_tuning_language"]
            ),
        })
    files_df = pd.DataFrame(file_rows)

    print("\n===== ACTION-LEVEL INVENTORY =====")
    print(summary_df.to_string(index=False))

    print("\n===== HIGH-VALUE IMPLEMENTATION FILES =====")
    hv = files_df[
        files_df["implementationish"] == 1
    ].copy()
    if len(hv):
        cols = [
            "path",
            "actions",
            "families",
            "functions_or_classes",
            "learning_rate_values",
            "steps_values",
            "optimizer_values",
            "gt_reference_file",
            "polypgen_reference_file",
            "tuning_language_file",
        ]
        print(hv[cols].head(120).to_string(index=False))
    else:
        print("NONE")

    # Freeze candidate-selection rules BEFORE R11B.
    selection_rules = {
        "panel_target": "3_to_4_label_free_tta_actions",
        "required_model_families": list(MODEL_PATTERNS.keys()),
        "required_properties": [
            "implementation exists before R11B",
            "no target GT required during adaptation",
            "same 3 frozen source-model families supported or reproducibly portable",
            "hyperparameters fixed without PolypGen outcomes",
            "pre-adaptation safety representation unchanged",
            "PolypGen never used to choose/tune action or hyperparameters",
        ],
        "preferred_action_diversity": [
            "entropy_minimization",
            "normalization_or_batch_statistics",
            "one additional genuinely distinct label-free TTA action if already implemented",
        ],
        "forbidden_selection_basis": [
            "PolypGen Dice",
            "PolypGen DeltaDice",
            "PolypGen HARM labels",
            "PolypGen safety AUROC/AUPRC",
            "PolypGen threshold behavior",
        ],
        "existing_reference_action": {
            "name": "A1_TENT_1STEP",
            "implementation": str(R05D3_SCRIPT),
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "steps": 1,
            "episodic_reset": True,
        },
    }

    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)

    summary_path = args.output_dir / "R11A_action_inventory_summary.csv"
    files_path = args.output_dir / "R11A_tta_artifact_inventory.csv"
    detail_path = args.output_dir / "R11A_action_match_snippets.csv"
    rules_path = args.output_dir / "R11A_ACTION_PANEL_SELECTION_RULES.json"

    summary_df.to_csv(summary_path, index=False)
    files_df.to_csv(files_path, index=False)
    detail_df.to_csv(detail_path, index=False)
    rules_path.write_text(
        json.dumps(selection_rules, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    candidate_actions = summary_df[
        summary_df["readiness"].isin([
            "AUTHORITATIVE_EXISTING_ACTION_EXPECTED",
            "CANDIDATE_REQUIRES_MANUAL_LINEAGE_REVIEW",
        ])
    ]["action"].astype(str).tolist()

    final = {
        "decision": (
            "TTA_ACTION_INVENTORY_FROZEN_FOR_R11B_PANEL_SELECTION"
        ),
        "script_version": VERSION,
        "build": BUILD,
        "candidate_actions_requiring_review": candidate_actions,
        "discovered_action_types": summary_df["action"].astype(str).tolist(),
        "matched_tta_artifact_files": len(files_df),
        "polypgen_used_for_selection": False,
        "target_performance_read": False,
        "model_inference_run": False,
        "tta_run": False,
        "next_stage": (
            "R11B_action_panel_lineage_lock_then_cross_action_generation"
        ),
        "artifacts": {
            summary_path.name: sha256_file(summary_path),
            files_path.name: sha256_file(files_path),
            detail_path.name: sha256_file(detail_path),
            rules_path.name: sha256_file(rules_path),
        },
    }

    final_path = args.output_dir / "R11A_ACTION_INVENTORY_LOCK.json"
    final_path.write_text(
        json.dumps(final, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n===== FINAL DECISION =====")
    print(
        "DECISION:",
        final["decision"],
    )
    print(
        "Candidate actions requiring lineage review:",
        candidate_actions,
    )
    print("POLYPGEN USED FOR SELECTION: NO")
    print("TARGET PERFORMANCE READ: NO")
    print("MODEL INFERENCE RUN: NO")
    print("TTA RUN: NO")
    print("\nOutputs:")
    print(summary_path)
    print(files_path)
    print(detail_path)
    print(rules_path)
    print(final_path)
    print("PASS")


if __name__ == "__main__":
    main()
