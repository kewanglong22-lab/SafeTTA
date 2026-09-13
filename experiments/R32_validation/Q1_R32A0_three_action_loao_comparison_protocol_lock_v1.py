#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SafeTTA R32A0
Three-Action Leave-One-Action-Out Comparison Protocol Lock

Purpose
-------
Freeze the post-R31B3 comparison protocol for the new action-transferable
SafeTTA contribution. No new segmentation/TTA inference is performed here.

Panel
-----
Reuse the exact R31B3 panel:
  800 NeoPolyp physical cases
  9 frozen segmentation states
  3 frozen actions:
    TENT1
    PL-CONF90
    MEMO-SEG4-1STEP

LOAO directions
---------------
  TENT1 + PL-CONF90            -> unseen MEMO
  TENT1 + MEMO                 -> unseen PL
  PL-CONF90 + MEMO             -> unseen TENT1

Exact split rule
----------------
Do NOT regenerate folds.
Recover and freeze the exact (split_seed, sample_id, fold) assignments already
used by R31B3 PRIMARY_LOAO_ACTION_CONDITIONAL OOF predictions.

Metrics
-------
  AUROC
  AUPRC
  AUPRC Lift = AUPRC / HARM prevalence
Report each held-out action + macro across actions + mean/std across 5 seeds.

Frozen LOAO comparison models
-----------------------------
  RANDOM_CONSTANT
  STRUCTURE_ONLY_M2
  SOURCE_SEMANTIC64_ONLY
  SOURCE_Q66_ONLY
  DELTA_SEMANTIC64_ONLY
  SHARED_TRANSITION_Q66_DSEM64          [PRIMARY]
  ACTION_CONDITIONAL_Q66_DSEM64         [diagnostic]

Upper reference
---------------
  SEPARATE_ACTION_REFERENCE
This is same-action training with physical-case holdout. It is NOT LOAO and
must never be presented as unseen-action transfer evidence.

Learned model
-------------
  StandardScaler fit on training fold only
  LogisticRegression(C=1, class_weight='balanced', solver='lbfgs')
  No hyperparameter search.

Published reliability baselines
-------------------------------
SicTTA-CCD / TEGDA-ADIC / MC-dropout are deferred to R32B because they need
additional frozen score generation beyond the current R31B3 feature table.

Scientific status
-----------------
POST-R31B3 METHOD INTERPRETATION / COMPARISON.
Not prospective confirmation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd


VERSION = "2026-09-12-R32A0-v1"

ROOT = Path(r"F:\MEDSEG_SAFETTA")
CODE = ROOT / "code"

R31B3_DIR = ROOT / "R31B3_first_800case_gt_reveal_and_three_action_loao_v1"
R31B3_FINAL = R31B3_DIR / "R31B3_FINAL_LOCK.json"
EXPECTED_R31B3_FINAL_SHA256 = (
    "eb91d991752c50e71b1993e4ee0a50e0b9e9cdb35fffdd58b36553e0cebaaadc"
)
R31B3_TABLE = R31B3_DIR / "R31B3_THREE_ACTION_MODELCASE_TABLE.csv"
R31B3_OOF = R31B3_DIR / "R31B3_OOF_PREDICTIONS.csv"
R31B3_DECISION = R31B3_DIR / "R31B3_DECISION.json"

R30A0_SCRIPT = CODE / "Q1_R30A0_paired_action_crc_protocol_and_asset_audit_v1_fix1.py"
EXPECTED_R30A0_SHA256 = (
    "9dc5d3a13f3d64608c4c599f811f929f731d7fac05e2aaa66eb344e24ace2298"
)

DEFAULT_OUT = ROOT / "R32A0_three_action_loao_comparison_protocol_lock_v1"

EXPECTED_DECISION = "GO_STRONG_ACTION_TRANSFERABLE_SAFETY"

ACTIONS = ("TENT1", "PL-CONF90", "MEMO-SEG4-1STEP")
EXPECTED_CASES = 800
EXPECTED_STATES = 9
EXPECTED_ROWS = EXPECTED_CASES * EXPECTED_STATES * len(ACTIONS)

SPLIT_SEEDS = [20260912, 20260913, 20260914, 20260915, 20260916]
N_FOLDS = 5
HARM_THRESHOLD = -0.02

LOAO_DIRECTIONS = [
    {"train_actions": ["TENT1", "PL-CONF90"], "test_action": "MEMO-SEG4-1STEP"},
    {"train_actions": ["TENT1", "MEMO-SEG4-1STEP"], "test_action": "PL-CONF90"},
    {"train_actions": ["PL-CONF90", "MEMO-SEG4-1STEP"], "test_action": "TENT1"},
]

LOAO_MODELS = [
    {
        "name": "RANDOM_CONSTANT",
        "features": [],
        "fit": "NONE",
        "role": "chance_reference",
    },
    {
        "name": "STRUCTURE_ONLY_M2",
        "features": "QSOURCE[0:2]",
        "fit": "balanced LogisticRegression(C=1)",
        "role": "source_mask_morphology_baseline",
    },
    {
        "name": "SOURCE_SEMANTIC64_ONLY",
        "features": "QSOURCE[2:66]",
        "fit": "balanced LogisticRegression(C=1)",
        "role": "source_prediction_conditioned_semantic_baseline",
    },
    {
        "name": "SOURCE_Q66_ONLY",
        "features": "QSOURCE[0:66]",
        "fit": "balanced LogisticRegression(C=1)",
        "role": "source_state_baseline",
    },
    {
        "name": "DELTA_SEMANTIC64_ONLY",
        "features": "DSEM[0:64]",
        "fit": "balanced LogisticRegression(C=1)",
        "role": "action_transition_only_ablation",
    },
    {
        "name": "SHARED_TRANSITION_Q66_DSEM64",
        "features": "QSOURCE[0:66] + DSEM[0:64]",
        "fit": "balanced LogisticRegression(C=1)",
        "role": "PRIMARY_ACTION_TRANSFERABLE_REPRESENTATION",
    },
    {
        "name": "ACTION_CONDITIONAL_Q66_DSEM64",
        "features": "QSOURCE[0:66] + DSEM[0:64] + ActionID + FamilyActionID",
        "fit": "balanced LogisticRegression(C=1)",
        "role": "diagnostic_action_identity_variant",
    },
]

UPPER_REFERENCE = {
    "name": "SEPARATE_ACTION_REFERENCE",
    "features": "QSOURCE[0:66] + DSEM[0:64]",
    "fit": "balanced LogisticRegression(C=1)",
    "role": "same-action physical-case-heldout reference; NOT LOAO transfer",
}

METRICS = ["AUROC", "AUPRC", "AUPRC_LIFT=AUPRC/HARM_PREVALENCE"]

FORBIDDEN_EXTERNAL_TOKENS = ("endotect", "polypgen", "sun-seg", "sunseg", "promise12")


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
            f"{label} SHA mismatch\nexpected={expected}\nobserved={got}\npath={path}"
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


def reject_external(path: Path):
    low = str(path).lower()
    if any(tok in low for tok in FORBIDDEN_EXTERNAL_TOKENS):
        raise RuntimeError(f"External path forbidden in R32A0: {path}")


def verify_upstream() -> dict:
    require_sha(R31B3_FINAL, EXPECTED_R31B3_FINAL_SHA256, "R31B3 final")
    require_sha(R30A0_SCRIPT, EXPECTED_R30A0_SHA256, "R30 feature schema")

    final = json.loads(R31B3_FINAL.read_text(encoding="utf-8"))
    if final.get("status") != (
        "PASS_R31B3_FIRST_800CASE_GT_REVEAL_AND_THREE_ACTION_LOAO_COMPLETE"
    ):
        raise RuntimeError("R31B3 final status changed.")
    if str(final.get("decision")) != EXPECTED_DECISION:
        raise RuntimeError(
            f"R31B3 decision={final.get('decision')!r}; expected={EXPECTED_DECISION!r}"
        )
    if bool(final.get("external_data_access", True)):
        raise RuntimeError("R31B3 reports external data access.")

    for p in [R31B3_TABLE, R31B3_OOF, R31B3_DECISION]:
        reject_external(p)
        if not p.is_file():
            raise FileNotFoundError(p)

    expected_table = final.get("three_action_table_sha256")
    if expected_table and sha256_file(R31B3_TABLE) != str(expected_table):
        raise RuntimeError("R31B3 table SHA changed.")

    expected_oof = final.get("OOF_predictions_sha256")
    if expected_oof and sha256_file(R31B3_OOF) != str(expected_oof):
        raise RuntimeError("R31B3 OOF SHA changed.")

    return final


def build_exact_split_manifest() -> pd.DataFrame:
    d = pd.read_csv(
        R31B3_OOF,
        usecols=["split_seed", "fold", "model", "sample_id"],
        dtype={"sample_id": str},
        low_memory=False,
    )

    p = d[d["model"] == "PRIMARY_LOAO_ACTION_CONDITIONAL"].copy()
    if len(p) == 0:
        raise RuntimeError("Cannot recover R31B3 primary LOAO fold assignments.")

    manifest = (
        p[["split_seed", "fold", "sample_id"]]
        .drop_duplicates()
        .sort_values(["split_seed", "fold", "sample_id"], kind="mergesort")
        .reset_index(drop=True)
    )

    if set(manifest["split_seed"].astype(int).unique()) != set(SPLIT_SEEDS):
        raise RuntimeError("Recovered split-seed set changed.")

    if manifest.duplicated(["split_seed", "sample_id"]).any():
        raise RuntimeError("Physical case maps to multiple folds in one seed.")

    for seed in SPLIT_SEEDS:
        g = manifest[manifest["split_seed"].astype(int) == seed]
        if g["sample_id"].nunique() != EXPECTED_CASES:
            raise RuntimeError(
                f"seed={seed} cases={g['sample_id'].nunique()} expected={EXPECTED_CASES}"
            )
        if set(g["fold"].astype(int).unique()) != set(range(N_FOLDS)):
            raise RuntimeError(f"seed={seed} fold set changed.")

    expected_rows = len(SPLIT_SEEDS) * EXPECTED_CASES
    if len(manifest) != expected_rows:
        raise RuntimeError(
            f"split manifest rows={len(manifest)} expected={expected_rows}"
        )
    return manifest


def verify_panel() -> dict:
    d = pd.read_csv(
        R31B3_TABLE,
        usecols=[
            "sample_id",
            "model_state_id",
            "model_family",
            "action",
            "r31b3_harm_label",
        ],
        dtype={"sample_id": str},
        low_memory=False,
    )

    if len(d) != EXPECTED_ROWS:
        raise RuntimeError(f"R31B3 panel rows={len(d)} expected={EXPECTED_ROWS}")
    if d["sample_id"].nunique() != EXPECTED_CASES:
        raise RuntimeError("Physical-case count drift.")
    if d["model_state_id"].nunique() != EXPECTED_STATES:
        raise RuntimeError("State count drift.")
    if set(d["action"].astype(str).unique()) != set(ACTIONS):
        raise RuntimeError("Action set drift.")
    if d.duplicated(["sample_id", "model_state_id", "action"]).any():
        raise RuntimeError("Duplicate case/state/action row.")

    harm = (
        d.groupby("action", as_index=False)
        .agg(
            rows=("r31b3_harm_label", "size"),
            harm_count=("r31b3_harm_label", "sum"),
            harm_rate=("r31b3_harm_label", "mean"),
        )
        .sort_values("action", kind="mergesort")
    )

    return {
        "physical_cases": int(d["sample_id"].nunique()),
        "model_states": int(d["model_state_id"].nunique()),
        "rows": int(len(d)),
        "harm_prevalence": harm.to_dict(orient="records"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    print("=" * 124)
    print("SafeTTA R32A0 Three-Action LOAO Comparison Protocol Lock")
    print("Version                       :", VERSION)
    print("New inference                  : NO")
    print("Hyperparameter search          : NO")
    print("Fold regeneration              : NO")
    print("External cohort access         : NO")
    print("Scientific status              : POST-R31B3 METHOD INTERPRETATION")
    print("=" * 124)

    verify_upstream()
    panel_info = verify_panel()
    split_manifest = build_exact_split_manifest()

    out = args.output_dir.resolve()
    if out.exists():
        raise RuntimeError(f"Output exists; protocol lock cannot overwrite: {out}")
    out.mkdir(parents=True, exist_ok=False)

    split_path = out / "R32A0_EXACT_R31B3_LOAO_SPLIT_MANIFEST.csv"
    atomic_csv(split_manifest, split_path)

    protocol = {
        "status": "LOCKED_R32A_THREE_ACTION_LOAO_COMPARISON_PROTOCOL",
        "version": VERSION,
        "scientific_status": "POST_R31B3_METHOD_INTERPRETATION_AND_COMPARISON",
        "prospective_confirmation_claim": False,
        "upstream": {
            "R31B3_final_sha256": EXPECTED_R31B3_FINAL_SHA256,
            "R31B3_decision": EXPECTED_DECISION,
            "R31B3_three_action_table": str(R31B3_TABLE),
            "R31B3_three_action_table_sha256": sha256_file(R31B3_TABLE),
            "R31B3_OOF_predictions": str(R31B3_OOF),
            "R31B3_OOF_predictions_sha256": sha256_file(R31B3_OOF),
            "R31B3_decision_file": str(R31B3_DECISION),
            "R31B3_decision_file_sha256": sha256_file(R31B3_DECISION),
            "R30A0_schema_sha256": EXPECTED_R30A0_SHA256,
        },
        "panel": panel_info,
        "exact_split_manifest": {
            "path": str(split_path),
            "sha256": sha256_file(split_path),
            "source": "R31B3 PRIMARY_LOAO_ACTION_CONDITIONAL OOF rows",
            "regenerated": False,
            "split_seeds": SPLIT_SEEDS,
            "folds": N_FOLDS,
            "group": "physical sample_id",
        },
        "actions": list(ACTIONS),
        "loao_directions": LOAO_DIRECTIONS,
        "harm_definition": f"DeltaDice <= {HARM_THRESHOLD}",
        "metrics": METRICS,
        "learned_model_contract": {
            "scaler": "StandardScaler fit on training fold only",
            "classifier": "LogisticRegression",
            "C": 1.0,
            "class_weight": "balanced",
            "solver": "lbfgs",
            "hyperparameter_search": False,
        },
        "loao_models": LOAO_MODELS,
        "primary_representation": "SHARED_TRANSITION_Q66_DSEM64",
        "action_identity_status": (
            "diagnostic covariate; not assumed necessary for unseen-action transfer"
        ),
        "upper_reference": UPPER_REFERENCE,
        "reporting": {
            "per_heldout_action": True,
            "macro_across_actions": True,
            "mean_std_across_five_split_seeds": True,
            "AUPRC_lift_required": True,
            "separate_action_reference_must_be_labeled_non_LOAO": True,
        },
        "published_reliability_baselines": {
            "included_in_R32A": False,
            "reason": (
                "CCD/ADIC/MC-dropout require additional frozen score generation; "
                "reserved for R32B on the same split manifest."
            ),
            "next_stage_after_R32A": "R32B_PUBLISHED_RELIABILITY_BASELINES",
        },
        "prohibitions": [
            "no new segmentation/TTA inference",
            "no feature engineering after R32A0 lock",
            "no classifier hyperparameter search",
            "no fold regeneration",
            "no outcome-dependent method selection inside R32A1",
            "no external cohort access",
        ],
        "next": "R32A1_THREE_ACTION_LOAO_COMPARISON_EXECUTION",
    }

    protocol_path = out / "R32A0_COMPARISON_PROTOCOL_LOCK.json"
    atomic_json(protocol_path, protocol)

    final_lock = {
        "status": "PASS_R32A0_THREE_ACTION_LOAO_COMPARISON_PROTOCOL_LOCK_COMPLETE",
        "version": VERSION,
        "scientific_status": "POST_R31B3_METHOD_INTERPRETATION_AND_COMPARISON",
        "R31B3_final_sha256": EXPECTED_R31B3_FINAL_SHA256,
        "R31B3_table_sha256": sha256_file(R31B3_TABLE),
        "R31B3_OOF_sha256": sha256_file(R31B3_OOF),
        "split_manifest_sha256": sha256_file(split_path),
        "protocol_sha256": sha256_file(protocol_path),
        "new_inference": False,
        "hyperparameter_search": False,
        "external_data_access": False,
        "next": "R32A1_THREE_ACTION_LOAO_COMPARISON_EXECUTION",
    }
    final_path = out / "R32A0_FINAL_LOCK.json"
    atomic_json(final_path, final_lock)

    print("\nR32A0 FROZEN COMPARISON")
    print("  panel                  : 800 cases x 9 states x 3 actions")
    print("  exact R31B3 folds      : YES")
    print("  split seeds            :", SPLIT_SEEDS)
    print("  LOAO models            :", len(LOAO_MODELS))
    for m in LOAO_MODELS:
        print("    -", m["name"])
    print("  upper reference        :", UPPER_REFERENCE["name"])
    print("  primary representation :", "SHARED_TRANSITION_Q66_DSEM64")
    print("  metrics                : AUROC / AUPRC / AUPRC Lift")
    print("  published baselines    : deferred to R32B")
    print("  new inference          : NO")
    print("  external access        : NO")

    print("\nFINAL STATUS : PASS_R32A0_THREE_ACTION_LOAO_COMPARISON_PROTOCOL_LOCK_COMPLETE")
    print("Split manifest SHA256:", sha256_file(split_path))
    print("Protocol SHA256      :", sha256_file(protocol_path))
    print("Final lock SHA256    :", sha256_file(final_path))
    print("Output               :", out)
    print("NEXT                 : R32A1_THREE_ACTION_LOAO_COMPARISON_EXECUTION")
    print("=" * 124)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
