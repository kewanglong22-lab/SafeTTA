#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SafeTTA B6-P0-8A3 — NeoPolyp three-action LOAO exact binding amendment

Purpose
-------
Freeze the exact, pre-existing R32A1 OOF selectors for the P08 primary
paired increment:
    SHARED_TRANSITION_Q66_DSEM64 - SOURCE_Q66_ONLY

This is a lineage/binding amendment only.
It computes NO AUROC/AUPRC, NO bootstrap, NO model fit, NO inference,
NO score flip, NO thresholding, and NO outcome-driven selection.

Why this amendment exists
-------------------------
P08A-fix2 surfaced three "candidates" for each intended model because the
same immutable model is evaluated in the three pre-specified LOAO held-out
action directions.  Those three directions are not competing predictors.
They jointly define the frozen three-action LOAO panel.

The binding below is selected exclusively from immutable semantic labels:
  model, evaluation_type, direction, split_seed, fold
and never from historical performance anchors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


VERSION = "2026-09-15-B6-P08A3-v1"

ROOT = Path(r"F:\MEDSEG_SAFETTA")
CODE = ROOT / "code"

PROTOCOL = CODE / "B6_P08_FINAL_STATISTICAL_CLOSURE_PROTOCOL_v1.json"
EXPECTED_PROTOCOL_SHA256 = (
    "fc231ec059188f44db277940a321f341b2f322b80902664c10889bfb5fe2a147"
)

P08A_FIX2_DECISION = (
    ROOT / "B6_P08A_final_statistical_closure_asset_binding_audit_v1_fix2"
    / "B6_P08A_FINAL_STATISTICAL_CLOSURE_BINDING.json"
)

R32A1_OOF = (
    ROOT / "R32A1_three_action_loao_comparison_execution_v1_fix1"
    / "R32A1_FIX1_OOF_PREDICTIONS.csv"
)

R31B3_MODELCASE = (
    ROOT / "R31B3_first_800case_gt_reveal_and_three_action_loao_v1"
    / "R31B3_THREE_ACTION_MODELCASE_TABLE.csv"
)

OUT_DIR = ROOT / "B6_P08A3_neopolyp_loao_exact_binding_amendment_v1"
LOCK_JSON = OUT_DIR / "B6_P08A3_NEOPOLYP_LOAO_EXACT_BINDING_AMENDMENT.json"
SELECTOR_CSV = OUT_DIR / "B6_P08A3_LOAO_SELECTOR_MANIFEST.csv"
REPORT_TXT = OUT_DIR / "B6_P08A3_REPORT.txt"

PASS_GATE = "PASS_B6_P08A3_NEOPOLYP_LOAO_EXACT_BINDING_AMENDMENT"

SOURCE_MODEL = "SOURCE_Q66_ONLY"
SHARED_MODEL = "SHARED_TRANSITION_Q66_DSEM64"
EVAL_TYPE = "LOAO"

DIRECTION_TO_ACTION = {
    "PL-CONF90+MEMO-SEG4-1STEP->TENT1": "TENT1",
    "TENT1+MEMO-SEG4-1STEP->PL-CONF90": "PL-CONF90",
    "TENT1+PL-CONF90->MEMO-SEG4-1STEP": "MEMO-SEG4-1STEP",
}
EXPECTED_DIRECTIONS = list(DIRECTION_TO_ACTION.keys())
EXPECTED_ACTIONS = ["MEMO-SEG4-1STEP", "PL-CONF90", "TENT1"]
EXPECTED_SPLIT_SEEDS = [20260912, 20260913, 20260914, 20260915, 20260916]
EXPECTED_FOLDS = [0, 1, 2, 3, 4]
EXPECTED_IMAGES = 800
EXPECTED_STATES = 9
EXPECTED_BASE_ROWS = 21600
EXPECTED_ROWS_PER_MODEL = 108000
EXPECTED_ROWS_PER_MODEL_DIRECTION = 36000
EXPECTED_ROWS_PER_MODEL_DIRECTION_SEED = 7200

REQUIRED_COLUMNS = [
    "split_seed", "fold", "model", "evaluation_type", "direction",
    "sample_id", "model_state_id", "model_family", "action",
    "y_harm", "score",
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


def _clean_ints(values) -> List[int]:
    return sorted(int(x) for x in pd.Series(values).dropna().unique().tolist())


def _clean_strs(values) -> List[str]:
    return sorted(str(x) for x in pd.Series(values).dropna().unique().tolist())


def validate_protocol() -> Dict[str, str]:
    if not PROTOCOL.is_file():
        raise FileNotFoundError(PROTOCOL)
    observed = sha256_file(PROTOCOL)
    if observed != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError(
            "P08 protocol SHA mismatch: "
            f"expected={EXPECTED_PROTOCOL_SHA256} observed={observed}"
        )
    payload = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    expected_status = "FROZEN_AFTER_P07B_BEFORE_FINAL_STATISTICAL_CLOSURE_METRICS"
    if payload.get("status") != expected_status:
        raise RuntimeError(
            f"P08 protocol status changed: {payload.get('status')!r}"
        )
    return {
        "path": str(PROTOCOL),
        "sha256": observed,
        "status": payload["status"],
    }


def load_fix2_context() -> Dict:
    if not P08A_FIX2_DECISION.is_file():
        raise FileNotFoundError(
            f"Required P08A-fix2 decision JSON not found: {P08A_FIX2_DECISION}"
        )
    d = json.loads(P08A_FIX2_DECISION.read_text(encoding="utf-8"))
    lineage = d.get("NeoPolyp_LOAO_lineage_audit", {})
    route = lineage.get("route")

    # The amendment is specifically justified by fix2's structural finding:
    # exact semantic model names exist but the generic rule treated the three
    # required LOAO directions as multiple candidates.
    if route not in {
        "STOP_NEEDS_EXPLICIT_FROZEN_LINEAGE_MAPPING",
        "READY_FOR_P08_LOAO_BINDING_AMENDMENT",
    }:
        raise RuntimeError(
            f"Unexpected P08A-fix2 LOAO lineage route: {route!r}"
        )

    return {
        "path": str(P08A_FIX2_DECISION),
        "sha256": sha256_file(P08A_FIX2_DECISION),
        "route": route,
        "structural_integrity_pass": lineage.get("structural_integrity_pass"),
        "source_candidates": lineage.get("SOURCE_STATE_BINDING_CANDIDATES", []),
        "shared_candidates": lineage.get("SHARED_TRANSITION_BINDING_CANDIDATES", []),
    }


def read_r32() -> pd.DataFrame:
    if not R32A1_OOF.is_file():
        raise FileNotFoundError(R32A1_OOF)

    header = list(pd.read_csv(R32A1_OOF, nrows=0, low_memory=False).columns)
    missing = [c for c in REQUIRED_COLUMNS if c not in header]
    if missing:
        raise RuntimeError(f"R32A1 missing required columns: {missing}")

    df = pd.read_csv(R32A1_OOF, usecols=REQUIRED_COLUMNS, low_memory=False)

    # Normalize only schema types; do not transform scientific values.
    df["split_seed"] = pd.to_numeric(df["split_seed"], errors="raise").astype(int)
    df["fold"] = pd.to_numeric(df["fold"], errors="raise").astype(int)
    df["y_harm"] = pd.to_numeric(df["y_harm"], errors="raise").astype(int)
    df["score"] = pd.to_numeric(df["score"], errors="raise").astype(float)

    if not np.isfinite(df["score"].to_numpy()).all():
        raise RuntimeError("R32A1 score contains non-finite values.")

    return df


def selector(df: pd.DataFrame, model: str) -> pd.DataFrame:
    m = (
        (df["model"].astype(str) == model)
        & (df["evaluation_type"].astype(str) == EVAL_TYPE)
        & (df["direction"].astype(str).isin(EXPECTED_DIRECTIONS))
    )
    return df.loc[m].copy()


def validate_one_model(d: pd.DataFrame, model: str) -> Dict:
    if len(d) != EXPECTED_ROWS_PER_MODEL:
        raise RuntimeError(
            f"{model}: expected {EXPECTED_ROWS_PER_MODEL} rows, got {len(d)}"
        )

    if set(_clean_strs(d["direction"])) != set(EXPECTED_DIRECTIONS):
        raise RuntimeError(f"{model}: direction set mismatch.")
    if set(_clean_strs(d["action"])) != set(EXPECTED_ACTIONS):
        raise RuntimeError(f"{model}: action set mismatch.")
    if _clean_ints(d["split_seed"]) != EXPECTED_SPLIT_SEEDS:
        raise RuntimeError(f"{model}: split_seed set mismatch.")
    if _clean_ints(d["fold"]) != EXPECTED_FOLDS:
        raise RuntimeError(f"{model}: fold set mismatch.")

    n_images = int(d["sample_id"].nunique())
    n_states = int(d["model_state_id"].nunique())
    if n_images != EXPECTED_IMAGES:
        raise RuntimeError(f"{model}: expected {EXPECTED_IMAGES} images, got {n_images}")
    if n_states != EXPECTED_STATES:
        raise RuntimeError(f"{model}: expected {EXPECTED_STATES} states, got {n_states}")

    # Each LOAO direction must contain only its held-out action.
    per_direction = []
    for direction, expected_action in DIRECTION_TO_ACTION.items():
        x = d[d["direction"].astype(str) == direction]
        if len(x) != EXPECTED_ROWS_PER_MODEL_DIRECTION:
            raise RuntimeError(
                f"{model}/{direction}: expected "
                f"{EXPECTED_ROWS_PER_MODEL_DIRECTION} rows, got {len(x)}"
            )
        acts = _clean_strs(x["action"])
        if acts != [expected_action]:
            raise RuntimeError(
                f"{model}/{direction}: held-out action mismatch: {acts}"
            )

        for seed in EXPECTED_SPLIT_SEEDS:
            z = x[x["split_seed"] == seed]
            if len(z) != EXPECTED_ROWS_PER_MODEL_DIRECTION_SEED:
                raise RuntimeError(
                    f"{model}/{direction}/seed={seed}: expected "
                    f"{EXPECTED_ROWS_PER_MODEL_DIRECTION_SEED} rows, got {len(z)}"
                )

            # Across five folds, a given physical modelcase-action must have
            # exactly one OOF prediction for this split seed.
            base_key = ["sample_id", "model_state_id", "action"]
            if z.duplicated(base_key, keep=False).any():
                raise RuntimeError(
                    f"{model}/{direction}/seed={seed}: duplicate OOF base keys."
                )
            if int(z["sample_id"].nunique()) != EXPECTED_IMAGES:
                raise RuntimeError(
                    f"{model}/{direction}/seed={seed}: image count mismatch."
                )
            if int(z["model_state_id"].nunique()) != EXPECTED_STATES:
                raise RuntimeError(
                    f"{model}/{direction}/seed={seed}: state count mismatch."
                )

        per_direction.append({
            "model": model,
            "evaluation_type": EVAL_TYPE,
            "direction": direction,
            "heldout_action": expected_action,
            "rows_all_split_seeds": int(len(x)),
            "rows_per_split_seed": EXPECTED_ROWS_PER_MODEL_DIRECTION_SEED,
            "split_seeds": EXPECTED_SPLIT_SEEDS,
            "folds": EXPECTED_FOLDS,
        })

    return {
        "model": model,
        "rows": int(len(d)),
        "images": n_images,
        "states": n_states,
        "actions": _clean_strs(d["action"]),
        "directions": _clean_strs(d["direction"]),
        "split_seeds": _clean_ints(d["split_seed"]),
        "folds": _clean_ints(d["fold"]),
        "harm_values": _clean_ints(d["y_harm"]),
        "per_direction": per_direction,
    }


def validate_exact_pairing(src: pd.DataFrame, shared: pd.DataFrame) -> Dict:
    pair_key = [
        "split_seed", "fold", "direction",
        "sample_id", "model_state_id", "model_family", "action",
    ]

    s = src[pair_key + ["y_harm"]].rename(columns={"y_harm": "y_src"})
    t = shared[pair_key + ["y_harm"]].rename(columns={"y_harm": "y_shared"})

    if s.duplicated(pair_key, keep=False).any():
        raise RuntimeError("SOURCE selector has duplicate pair keys.")
    if t.duplicated(pair_key, keep=False).any():
        raise RuntimeError("SHARED selector has duplicate pair keys.")

    j = s.merge(t, on=pair_key, how="outer", indicator=True, validate="one_to_one")
    if not (j["_merge"] == "both").all():
        counts = j["_merge"].value_counts().to_dict()
        raise RuntimeError(f"SOURCE/SHARED pairing mismatch: {counts}")

    if not np.array_equal(
        j["y_src"].to_numpy(dtype=int),
        j["y_shared"].to_numpy(dtype=int),
    ):
        raise RuntimeError("SOURCE/SHARED y_harm mismatch on exact paired rows.")

    return {
        "pair_key": pair_key,
        "paired_rows": int(len(j)),
        "exact_row_pairing": True,
        "harm_label_identical": True,
    }


def validate_against_modelcase(src: pd.DataFrame) -> Dict:
    if not R31B3_MODELCASE.is_file():
        raise FileNotFoundError(R31B3_MODELCASE)

    required = ["sample_id", "model_state_id", "action", "r31b3_harm_label"]
    h = list(pd.read_csv(R31B3_MODELCASE, nrows=0, low_memory=False).columns)
    missing = [c for c in required if c not in h]
    if missing:
        raise RuntimeError(f"R31B3 modelcase missing columns: {missing}")

    y = pd.read_csv(R31B3_MODELCASE, usecols=required, low_memory=False)
    y["r31b3_harm_label"] = pd.to_numeric(
        y["r31b3_harm_label"], errors="raise"
    ).astype(int)

    key = ["sample_id", "model_state_id", "action"]
    if len(y) != EXPECTED_BASE_ROWS:
        raise RuntimeError(
            f"R31B3 modelcase expected {EXPECTED_BASE_ROWS} rows, got {len(y)}"
        )
    if y.duplicated(key, keep=False).any():
        raise RuntimeError("R31B3 modelcase has duplicate physical modelcase-action keys.")

    # Check each split seed independently; do not collapse OOF scores.
    checked = 0
    for seed in EXPECTED_SPLIT_SEEDS:
        z = src[src["split_seed"] == seed][key + ["y_harm"]]
        if len(z) != EXPECTED_BASE_ROWS:
            raise RuntimeError(
                f"SOURCE selector seed={seed}: expected {EXPECTED_BASE_ROWS} rows, got {len(z)}"
            )
        if z.duplicated(key, keep=False).any():
            raise RuntimeError(f"SOURCE selector seed={seed}: duplicate base keys.")

        j = z.merge(y, on=key, how="outer", indicator=True, validate="one_to_one")
        if not (j["_merge"] == "both").all():
            raise RuntimeError(f"Seed {seed}: SOURCE vs R31B3 key mismatch.")
        if not np.array_equal(
            j["y_harm"].to_numpy(dtype=int),
            j["r31b3_harm_label"].to_numpy(dtype=int),
        ):
            raise RuntimeError(f"Seed {seed}: SOURCE vs R31B3 HARM label mismatch.")
        checked += len(j)

    return {
        "path": str(R31B3_MODELCASE),
        "sha256": sha256_file(R31B3_MODELCASE),
        "base_rows": int(len(y)),
        "unique_images": int(y["sample_id"].nunique()),
        "unique_states": int(y["model_state_id"].nunique()),
        "actions": _clean_strs(y["action"]),
        "checked_source_oof_rows_across_split_seeds": int(checked),
        "harm_label_parity": True,
    }


def write_outputs(
    protocol: Dict,
    fix2: Dict,
    r32_sha: str,
    src_info: Dict,
    shared_info: Dict,
    pairing: Dict,
    outcome_parity: Dict,
) -> None:
    if OUT_DIR.exists():
        raise FileExistsError(
            f"Never overwrite a prior amendment output: {OUT_DIR}"
        )
    OUT_DIR.mkdir(parents=True, exist_ok=False)

    selector_rows = src_info["per_direction"] + shared_info["per_direction"]
    pd.DataFrame(selector_rows).to_csv(SELECTOR_CSV, index=False)

    payload = {
        "amendment_id": "B6_P08A3_NEOPOLYP_LOAO_EXACT_BINDING_AMENDMENT_v1",
        "version": VERSION,
        "status": "FROZEN_LINEAGE_BINDING_BEFORE_P08B_STATISTICAL_CLOSURE_METRICS",
        "scientific_protocol": protocol,
        "reason_for_amendment": (
            "P08A-fix2 correctly surfaced the exact R32A1 semantic labels but "
            "its generic uniqueness rule counted the three pre-specified LOAO "
            "held-out action directions as three competing candidates. They are "
            "the three required directions of one frozen model binding, not "
            "alternative predictors."
        ),
        "selection_basis": (
            "Immutable semantic lineage only: model/evaluation_type/direction/"
            "split_seed/fold. Historical AUROC/AUPRC anchors are not used to "
            "select, flip, weight, or repair any predictor."
        ),
        "p08a_fix2_context": fix2,
        "authoritative_oof_asset": {
            "path": str(R32A1_OOF),
            "sha256": r32_sha,
            "required_columns": REQUIRED_COLUMNS,
        },
        "frozen_source_state_binding": {
            "model": SOURCE_MODEL,
            "evaluation_type": EVAL_TYPE,
            "directions": DIRECTION_TO_ACTION,
            "split_seeds": EXPECTED_SPLIT_SEEDS,
            "folds": EXPECTED_FOLDS,
            "score_column": "score",
            "label_column": "y_harm",
            "rows": EXPECTED_ROWS_PER_MODEL,
        },
        "frozen_shared_transition_binding": {
            "model": SHARED_MODEL,
            "evaluation_type": EVAL_TYPE,
            "directions": DIRECTION_TO_ACTION,
            "split_seeds": EXPECTED_SPLIT_SEEDS,
            "folds": EXPECTED_FOLDS,
            "score_column": "score",
            "label_column": "y_harm",
            "rows": EXPECTED_ROWS_PER_MODEL,
        },
        "exact_pairing": pairing,
        "outcome_lineage_parity": outcome_parity,
        "p08b_statistical_replay_contract": {
            "cluster_unit": "physical NeoPolyp image = sample_id",
            "bootstrap_reps": 2000,
            "bootstrap_seed": 20260923,
            "paired_comparison": (
                "SHARED_TRANSITION_Q66_DSEM64 - SOURCE_Q66_ONLY"
            ),
            "metric_aggregation": (
                "For each bootstrap replicate, resample the 800 physical "
                "sample_id clusters with replacement while preserving all "
                "nine states, all three held-out actions, and all five frozen "
                "split-seed OOF predictions. Compute AUROC/AUPRC separately "
                "for each split_seed x held-out action; macro-average the "
                "three actions within split seed, then average the five split "
                "seeds. No action weighting or seed selection."
            ),
            "invalid_single_class_rule": (
                "Drop only the affected metric/action replicate and report "
                "valid replicate counts; never impute 0.5."
            ),
            "historical_point_anchors_role": (
                "Parity diagnostic only after this binding. A mismatch must "
                "STOP P08B; it must never trigger selector changes."
            ),
            "forbidden": [
                "refit", "recalibration", "score flip", "action redefinition",
                "representation change", "target-domain use",
                "post-hoc action weighting", "post-hoc seed selection",
                "performance-based selector choice",
            ],
        },
        "gate": PASS_GATE,
    }

    LOCK_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = f"""\
SafeTTA B6-P08A3 — NeoPolyp three-action LOAO exact binding amendment
Version: {VERSION}

P08 protocol SHA256:
  {protocol['sha256']}

Authoritative frozen OOF asset:
  {R32A1_OOF}
  SHA256 = {r32_sha}

Frozen SOURCE-state selector:
  model = {SOURCE_MODEL}
  evaluation_type = {EVAL_TYPE}
  directions = {EXPECTED_DIRECTIONS}
  split_seeds = {EXPECTED_SPLIT_SEEDS}
  folds = {EXPECTED_FOLDS}
  rows = {EXPECTED_ROWS_PER_MODEL}

Frozen shared-transition selector:
  model = {SHARED_MODEL}
  evaluation_type = {EVAL_TYPE}
  directions = {EXPECTED_DIRECTIONS}
  split_seeds = {EXPECTED_SPLIT_SEEDS}
  folds = {EXPECTED_FOLDS}
  rows = {EXPECTED_ROWS_PER_MODEL}

Exact paired rows:
  {pairing['paired_rows']}

R31B3 outcome-label parity:
  {outcome_parity['harm_label_parity']}

IMPORTANT:
  The three LOAO directions are the three required held-out actions of one
  frozen predictor definition. They are not three alternative predictors.

  No performance anchor was used for selector binding.
  No AUROC/AUPRC was computed.
  No bootstrap was run.
  No model was fit.
  No score direction was changed.

GATE={PASS_GATE}
NEXT=P08B paired physical-image clustered statistical closure replay
"""
    REPORT_TXT.write_text(textwrap.dedent(report), encoding="utf-8")


def self_test() -> None:
    assert len(DIRECTION_TO_ACTION) == 3
    assert set(DIRECTION_TO_ACTION.values()) == set(EXPECTED_ACTIONS)
    assert SOURCE_MODEL != SHARED_MODEL
    assert EXPECTED_ROWS_PER_MODEL == (
        len(EXPECTED_DIRECTIONS)
        * len(EXPECTED_SPLIT_SEEDS)
        * EXPECTED_ROWS_PER_MODEL_DIRECTION_SEED
    )
    assert EXPECTED_BASE_ROWS == (
        EXPECTED_IMAGES * EXPECTED_STATES * len(EXPECTED_ACTIONS)
    )
    print("SELF_TEST_BINDING_CARDINALITY=PASS")
    print("SELF_TEST_NO_PERFORMANCE_SELECTION=PASS")
    print("SELF_TEST=PASS")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Freeze the exact NeoPolyp R32A1 SOURCE_Q66_ONLY and "
            "SHARED_TRANSITION_Q66_DSEM64 three-action LOAO bindings."
        )
    )
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    print("=" * 150)
    print("SafeTTA B6-P08A3 — NeoPolyp three-action LOAO exact binding amendment")
    print("Version          :", VERSION)
    print("New metrics      : NO")
    print("Bootstrap        : NO")
    print("Model/inference  : NO")
    print("Refit/calibration: NO")
    print("Performance-based selector choice: NO")
    print("=" * 150)

    protocol = validate_protocol()
    print("PROTOCOL_SHA256 =", protocol["sha256"])

    fix2 = load_fix2_context()
    print("P08A_FIX2_ROUTE =", fix2["route"])

    print("\n[1/4] Load exact frozen R32A1 OOF")
    df = read_r32()
    r32_sha = sha256_file(R32A1_OOF)
    print("R32A1 =", R32A1_OOF)
    print("SHA256 =", r32_sha)
    print("rows =", len(df))

    print("\n[2/4] Freeze semantic selectors")
    src = selector(df, SOURCE_MODEL)
    shared = selector(df, SHARED_MODEL)

    src_info = validate_one_model(src, SOURCE_MODEL)
    shared_info = validate_one_model(shared, SHARED_MODEL)

    print("SOURCE rows =", len(src))
    print("SHARED rows =", len(shared))
    print("directions =", EXPECTED_DIRECTIONS)
    print("split_seeds =", EXPECTED_SPLIT_SEEDS)
    print("folds =", EXPECTED_FOLDS)

    print("\n[3/4] Exact row pairing + frozen outcome-label parity")
    pairing = validate_exact_pairing(src, shared)
    outcome_parity = validate_against_modelcase(src)
    print("paired_rows =", pairing["paired_rows"])
    print("SOURCE/SHARED harm parity =", pairing["harm_label_identical"])
    print("R31B3 harm parity =", outcome_parity["harm_label_parity"])

    print("\n[4/4] Write immutable amendment")
    write_outputs(
        protocol=protocol,
        fix2=fix2,
        r32_sha=r32_sha,
        src_info=src_info,
        shared_info=shared_info,
        pairing=pairing,
        outcome_parity=outcome_parity,
    )

    print("lock_json =", LOCK_JSON)
    print("selector_manifest =", SELECTOR_CSV)
    print("report =", REPORT_TXT)
    print(f"GATE={PASS_GATE}")
    print("NEXT=P08B paired physical-image clustered statistical closure replay")
    print("=" * 150)


if __name__ == "__main__":
    main()
