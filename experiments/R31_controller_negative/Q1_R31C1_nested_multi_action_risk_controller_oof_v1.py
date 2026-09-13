#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SafeTTA R31C1
Nested Multi-Action Risk-Controlled Controller OOF

This script executes the exact R31C0 protocol:

  800 NeoPolyp physical cases
  5 split seeds x 5 outer folds
  480 TRAIN / 160 CALIBRATION / 160 TEST

Shared utility model:
  StandardScaler(train only) + Ridge(alpha=1)
  target = true action DeltaDice

Shared HARM model:
  StandardScaler(train only) +
  LogisticRegression(C=1, class_weight='balanced', solver='lbfgs')
  target = 1[DeltaDice <= -0.02]

Features:
  q_source66 + delta_semantic64 + ActionID + FamilyActionID

CRC:
  alpha_total = 0.020
  alpha_per_action = 0.020 / 3
  corrected risk = n/(n+1)*Rhat + 1/(n+1)
  threshold per action uses CALIBRATION only and chooses the largest tau
  satisfying corrected empirical HARM risk <= alpha_per_action.
  If no threshold qualifies, tau=-inf.

Proposed controller:
  safe(a) iff utility_pred(a)>0 AND p_harm(a)<=tau_a
  if no safe action -> SOURCE
  else choose largest predicted utility
  tie-break -> lower predicted HARM probability -> fixed action order

Frozen baselines:
  SOURCE_ONLY
  ALWAYS_TENT1
  ALWAYS_PL_CONF90
  ALWAYS_MEMO
  UTILITY_ONLY
  CRC_RISK_ONLY
  R31C_RISK_UTILITY_CRC
  ORACLE_BEST_ACTION

R31C0 feasibility gate:
  averaged over 5 complete OOF split-seed panels:
    R31C HARM rate <= 0.020
    mean deployed DeltaDice > 0
    adaptation coverage >= 0.05

No EndoTect or external cohort is accessed.
No controller hyperparameter search is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


VERSION = "2026-09-12-R31C1-v1"

ROOT = Path(r"F:\MEDSEG_SAFETTA")
CODE = ROOT / "code"

# ---------------------------------------------------------------------
# Exact frozen R31C0 protocol
# ---------------------------------------------------------------------

R31C0_SCRIPT = CODE / "Q1_R31C0_multi_action_risk_controller_protocol_lock_v1.py"
EXPECTED_R31C0_SCRIPT_SHA256 = (
    "e87a71357ac5237cf3e12c1ce7a5da2bd581dedf80e5261799a668dda4056aee"
)

R31C0_DIR = ROOT / "R31C0_multi_action_risk_controller_protocol_lock_v1"
R31C0_PROTOCOL = R31C0_DIR / "R31C0_CONTROLLER_PROTOCOL_LOCK.json"
EXPECTED_R31C0_PROTOCOL_SHA256 = (
    "e48de71c33ba7c6b74f41cc4ff846b2e726ab06376d1366f8622e1781fa216d6"
)
R31C0_FINAL = R31C0_DIR / "R31C0_FINAL_LOCK.json"
EXPECTED_R31C0_FINAL_SHA256 = (
    "e0df9dd61199dbe491651a1fffa23b7d4c954cf290b2f14a9d4d519ad566ac7c"
)
R31C0_PARTITIONS = R31C0_DIR / "R31C0_NESTED_480_160_160_PARTITIONS.csv"

# ---------------------------------------------------------------------
# Frozen R31B3 assets
# ---------------------------------------------------------------------

R31B3_DIR = ROOT / "R31B3_first_800case_gt_reveal_and_three_action_loao_v1"
R31B3_FINAL = R31B3_DIR / "R31B3_FINAL_LOCK.json"
EXPECTED_R31B3_FINAL_SHA256 = (
    "eb91d991752c50e71b1993e4ee0a50e0b9e9cdb35fffdd58b36553e0cebaaadc"
)
R31B3_TABLE = R31B3_DIR / "R31B3_THREE_ACTION_MODELCASE_TABLE.csv"
R31B3_MEMO_OUTCOMES = R31B3_DIR / "R31B3_MEMO_800CASE_MODELCASE_OUTCOMES.csv"

R31A_DIR = ROOT / "R31A_action_conditional_harm_predictor_v1"
R31A_LINEAGE = R31A_DIR / "R31A_SOURCE_TABLE_LINEAGE.json"

R30A0_SCRIPT = CODE / "Q1_R30A0_paired_action_crc_protocol_and_asset_audit_v1_fix1.py"
EXPECTED_R30A0_SHA256 = (
    "9dc5d3a13f3d64608c4c599f811f929f731d7fac05e2aaa66eb344e24ace2298"
)

DEFAULT_OUT = ROOT / "R31C1_nested_multi_action_risk_controller_oof_v1"

EXPECTED_DECISION = "GO_STRONG_ACTION_TRANSFERABLE_SAFETY"

EXPECTED_CASES = 800
EXPECTED_STATES = 9
ACTIONS = ("TENT1", "PL-CONF90", "MEMO-SEG4-1STEP")
ACTION_ORDER = {a: i for i, a in enumerate(ACTIONS)}

SPLIT_SEEDS = [20260912, 20260913, 20260914, 20260915, 20260916]
N_FOLDS = 5

TRAIN_CASES = 480
CAL_CASES = 160
TEST_CASES = 160

HARM_THRESHOLD = -0.02
BENEFIT_THRESHOLD = +0.02

ALPHA_TOTAL = 0.020
ALPHA_PER_ACTION = ALPHA_TOTAL / len(ACTIONS)

RIDGE_ALPHA = 1.0
HARM_LR_C = 1.0
HARM_LR_MAX_ITER = 5000

BASELINES = (
    "SOURCE_ONLY",
    "ALWAYS_TENT1",
    "ALWAYS_PL_CONF90",
    "ALWAYS_MEMO",
    "UTILITY_ONLY",
    "CRC_RISK_ONLY",
    "R31C_RISK_UTILITY_CRC",
    "ORACLE_BEST_ACTION",
)

FEASIBILITY_HARM_MAX = 0.020
FEASIBILITY_MEAN_DELTA_MIN_EXCLUSIVE = 0.0
FEASIBILITY_COVERAGE_MIN = 0.05

FORBIDDEN_EXTERNAL_TOKENS = (
    "endotect",
    "polypgen",
    "sun-seg",
    "sunseg",
    "promise12",
)


# ---------------------------------------------------------------------
# Utilities
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
            f"{label} SHA mismatch\nexpected={expected}\n"
            f"observed={got}\npath={path}"
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


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def reject_external(path: Path):
    low = str(path).lower()
    if any(tok in low for tok in FORBIDDEN_EXTERNAL_TOKENS):
        raise RuntimeError(f"External path forbidden in R31C1: {path}")


def norm_action(v: Any) -> str:
    s = str(v).strip().upper().replace("_", "-")
    aliases = {
        "TENT1": "TENT1",
        "TENT-1": "TENT1",
        "A1": "TENT1",
        "A1-TENT-1STEP": "TENT1",
        "A1-TENT1": "TENT1",
        "PL-CONF90": "PL-CONF90",
        "PLCONF90": "PL-CONF90",
        "PL-CONF-90": "PL-CONF90",
        "PL90": "PL-CONF90",
        "MEMO-SEG4-1STEP": "MEMO-SEG4-1STEP",
    }
    return aliases.get(s, str(v).strip())


# ---------------------------------------------------------------------
# Frozen upstream validation
# ---------------------------------------------------------------------

def verify_upstream() -> Dict[str, Any]:
    require_sha(R31C0_SCRIPT, EXPECTED_R31C0_SCRIPT_SHA256, "R31C0 script")
    require_sha(R31C0_PROTOCOL, EXPECTED_R31C0_PROTOCOL_SHA256, "R31C0 protocol")
    require_sha(R31C0_FINAL, EXPECTED_R31C0_FINAL_SHA256, "R31C0 final")
    require_sha(R31B3_FINAL, EXPECTED_R31B3_FINAL_SHA256, "R31B3 final")
    require_sha(R30A0_SCRIPT, EXPECTED_R30A0_SHA256, "R30 feature schema")

    protocol = json.loads(R31C0_PROTOCOL.read_text(encoding="utf-8"))
    final = json.loads(R31C0_FINAL.read_text(encoding="utf-8"))
    b3 = json.loads(R31B3_FINAL.read_text(encoding="utf-8"))

    if final.get("status") != "PASS_R31C0_MULTI_ACTION_CONTROLLER_PROTOCOL_LOCK_COMPLETE":
        raise RuntimeError("R31C0 final status changed.")
    if str(b3.get("decision")) != EXPECTED_DECISION:
        raise RuntimeError(
            f"R31B3 decision={b3.get('decision')!r}; expected={EXPECTED_DECISION!r}"
        )
    if bool(final.get("external_data_access", True)):
        raise RuntimeError("R31C0 reports external data access.")
    if float(final.get("alpha_total")) != ALPHA_TOTAL:
        raise RuntimeError("R31C0 alpha_total changed.")
    if not math.isclose(
        float(final.get("alpha_per_action")),
        ALPHA_PER_ACTION,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise RuntimeError("R31C0 alpha_per_action changed.")

    if not R31C0_PARTITIONS.is_file():
        raise FileNotFoundError(R31C0_PARTITIONS)
    if sha256_file(R31C0_PARTITIONS) != str(final.get("partition_sha256")):
        raise RuntimeError("R31C0 partition SHA changed.")

    if not R31B3_TABLE.is_file():
        raise FileNotFoundError(R31B3_TABLE)
    if sha256_file(R31B3_TABLE) != str(
        protocol["upstream"]["R31B3_three_action_table_sha256"]
    ):
        raise RuntimeError("R31B3 three-action feature table SHA changed.")

    if not R31B3_MEMO_OUTCOMES.is_file():
        raise FileNotFoundError(R31B3_MEMO_OUTCOMES)

    return {
        "protocol": protocol,
        "final": final,
        "r31b3": b3,
    }


# ---------------------------------------------------------------------
# Reconstruct exact frozen 3-action table with DeltaDice
# ---------------------------------------------------------------------

def load_feature_schema() -> Tuple[List[str], List[str]]:
    a0 = import_module(R30A0_SCRIPT, "r31c1_r30a0")
    qnames = list(getattr(a0, "QSOURCE", []))
    dnames = list(getattr(a0, "DSEM", []))
    if len(qnames) != 66 or len(dnames) != 64:
        raise RuntimeError(
            f"Frozen feature schema drift: q={len(qnames)} dsem={len(dnames)}"
        )
    return qnames, dnames


def load_three_action_panel(
    qnames: Sequence[str],
    dnames: Sequence[str],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    base = pd.read_csv(R31B3_TABLE, low_memory=False)
    base["sample_id"] = base["sample_id"].astype(str)
    base["action"] = base["action"].map(norm_action)

    expected_rows = EXPECTED_CASES * EXPECTED_STATES * len(ACTIONS)
    if len(base) != expected_rows:
        raise RuntimeError(f"R31B3 table rows={len(base)} expected={expected_rows}")

    required = {
        "sample_id",
        "model_state_id",
        "model_family",
        "action",
        "r31b3_harm_label",
        *qnames,
        *dnames,
    }
    missing = sorted(required.difference(base.columns))
    if missing:
        raise RuntimeError(f"R31B3 table missing columns={missing[:20]}")

    if set(base["action"].unique()) != set(ACTIONS):
        raise RuntimeError(f"Unexpected action set={sorted(base['action'].unique())}")

    # TENT/PL true DeltaDice comes from the exact source-action table
    # recorded by R31A lineage.
    if not R31A_LINEAGE.is_file():
        raise FileNotFoundError(R31A_LINEAGE)
    lineage = json.loads(R31A_LINEAGE.read_text(encoding="utf-8"))
    src_info = lineage["source_table"]
    src_path = Path(str(src_info["path"]))
    reject_external(src_path)
    if not src_path.is_file():
        raise FileNotFoundError(src_path)
    if sha256_file(src_path) != str(src_info["sha256"]):
        raise RuntimeError("R31A source-action table SHA changed.")

    eval_ids = set(base["sample_id"].astype(str).unique())
    tp = pd.read_csv(src_path, low_memory=False)
    tp["sample_id"] = tp["sample_id"].astype(str)
    tp["action"] = tp["action"].map(norm_action)
    tp = tp[
        tp["sample_id"].isin(eval_ids)
        & tp["action"].isin(["TENT1", "PL-CONF90"])
    ].copy()

    if "delta_dice" not in tp.columns:
        found = None
        for c in [
            "candidate_delta_dice",
            "true_delta_dice",
            "action_delta_dice",
            "utility_target",
        ]:
            if c in tp.columns:
                found = c
                break
        if found is None:
            raise RuntimeError("TENT/PL source table lacks DeltaDice target.")
        tp["delta_dice"] = pd.to_numeric(tp[found], errors="raise")

    # MEMO true DeltaDice from the exact R31B3 first-reveal outcome file.
    memo = pd.read_csv(R31B3_MEMO_OUTCOMES, low_memory=False)
    memo["sample_id"] = memo["sample_id"].astype(str)
    memo["action"] = memo["action"].map(norm_action)
    if "delta_dice" not in memo.columns:
        raise RuntimeError("R31B3 MEMO outcome table lacks delta_dice.")

    keys = ["sample_id", "model_state_id", "model_family", "action"]

    target = pd.concat(
        [
            tp[keys + ["delta_dice"]],
            memo[keys + ["delta_dice"]],
        ],
        ignore_index=True,
    )

    if len(target) != expected_rows:
        raise RuntimeError(
            f"Reconstructed target rows={len(target)} expected={expected_rows}"
        )
    if target.duplicated(keys).any():
        raise RuntimeError("Duplicate target key in 3-action utility table.")

    panel = base.merge(
        target,
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    if len(panel) != expected_rows:
        raise RuntimeError("Feature/DeltaDice merge lost rows.")

    panel["delta_dice"] = pd.to_numeric(panel["delta_dice"], errors="raise")
    if not np.isfinite(panel["delta_dice"].to_numpy(float)).all():
        raise RuntimeError("Non-finite DeltaDice.")

    derived_harm = (panel["delta_dice"].to_numpy(float) <= HARM_THRESHOLD).astype(int)
    stored_harm = panel["r31b3_harm_label"].to_numpy(int)
    if not np.array_equal(derived_harm, stored_harm):
        mismatch = int(np.count_nonzero(derived_harm != stored_harm))
        raise RuntimeError(f"HARM label / DeltaDice mismatch rows={mismatch}")

    panel["benefit_label"] = (
        panel["delta_dice"].to_numpy(float) >= BENEFIT_THRESHOLD
    ).astype(int)

    # One exact row per case/state/action.
    if panel.duplicated(["sample_id", "model_state_id", "action"]).any():
        raise RuntimeError("Duplicate case/state/action row.")

    counts = panel["action"].value_counts().to_dict()
    expected_action_rows = EXPECTED_CASES * EXPECTED_STATES
    for action in ACTIONS:
        if int(counts.get(action, 0)) != expected_action_rows:
            raise RuntimeError(
                f"{action} rows={counts.get(action)} expected={expected_action_rows}"
            )

    return panel, {
        "R31B3_feature_table": str(R31B3_TABLE),
        "R31B3_feature_table_sha256": sha256_file(R31B3_TABLE),
        "R31A_source_action_table": str(src_path),
        "R31A_source_action_table_sha256": sha256_file(src_path),
        "R31B3_MEMO_outcomes": str(R31B3_MEMO_OUTCOMES),
        "R31B3_MEMO_outcomes_sha256": sha256_file(R31B3_MEMO_OUTCOMES),
    }


# ---------------------------------------------------------------------
# Exact feature construction
# ---------------------------------------------------------------------

def global_category_schema(
    panel: pd.DataFrame,
) -> Tuple[List[str], List[str]]:
    action_cats = list(ACTIONS)
    families = sorted(panel["model_family"].astype(str).unique().tolist())
    family_action_cats = [
        f"{fam}::{action}"
        for fam in families
        for action in action_cats
    ]
    return action_cats, family_action_cats


def make_X(
    d: pd.DataFrame,
    qnames: Sequence[str],
    dnames: Sequence[str],
    action_cats: Sequence[str],
    family_action_cats: Sequence[str],
) -> np.ndarray:
    base_names = list(qnames) + list(dnames)
    blocks = [d[base_names].to_numpy(float)]

    action_values = d["action"].astype(str).to_numpy()
    for cat in action_cats:
        blocks.append((action_values == cat).astype(float)[:, None])

    family_action_values = (
        d["model_family"].astype(str)
        + "::"
        + d["action"].astype(str)
    ).to_numpy()
    for cat in family_action_cats:
        blocks.append((family_action_values == cat).astype(float)[:, None])

    X = np.concatenate(blocks, axis=1)
    if not np.isfinite(X).all():
        raise RuntimeError("Non-finite controller features.")
    return X


def fit_models(
    train: pd.DataFrame,
    qnames: Sequence[str],
    dnames: Sequence[str],
    action_cats: Sequence[str],
    family_action_cats: Sequence[str],
):
    X = make_X(
        train,
        qnames,
        dnames,
        action_cats,
        family_action_cats,
    )
    y_u = train["delta_dice"].to_numpy(float)
    y_h = train["r31b3_harm_label"].to_numpy(int)

    if len(np.unique(y_h)) != 2:
        raise RuntimeError("TRAIN HARM target is one-class.")

    utility = Pipeline([
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=RIDGE_ALPHA)),
    ])
    harm = Pipeline([
        ("scale", StandardScaler()),
        (
            "lr",
            LogisticRegression(
                C=HARM_LR_C,
                class_weight="balanced",
                solver="lbfgs",
                max_iter=HARM_LR_MAX_ITER,
                random_state=0,
            ),
        ),
    ])

    utility.fit(X, y_u)
    harm.fit(X, y_h)
    return utility, harm


def predict_models(
    d: pd.DataFrame,
    utility,
    harm,
    qnames: Sequence[str],
    dnames: Sequence[str],
    action_cats: Sequence[str],
    family_action_cats: Sequence[str],
) -> pd.DataFrame:
    X = make_X(
        d,
        qnames,
        dnames,
        action_cats,
        family_action_cats,
    )
    out = d[
        [
            "sample_id",
            "model_state_id",
            "model_family",
            "action",
            "delta_dice",
            "r31b3_harm_label",
            "benefit_label",
        ]
    ].copy()
    out["utility_pred"] = utility.predict(X).astype(float)
    out["harm_prob"] = harm.predict_proba(X)[:, 1].astype(float)

    if not np.isfinite(
        out[["utility_pred", "harm_prob"]].to_numpy(float)
    ).all():
        raise RuntimeError("Non-finite controller predictions.")
    return out


# ---------------------------------------------------------------------
# Exact CRC calibration
# ---------------------------------------------------------------------

def crc_corrected_risk(harm_count: int, n: int) -> float:
    if n <= 0:
        return math.inf
    # n/(n+1)*Rhat + 1/(n+1) = (harm_count+1)/(n+1)
    return float((int(harm_count) + 1) / (int(n) + 1))


def select_crc_threshold(
    action_cal: pd.DataFrame,
) -> Dict[str, Any]:
    if len(action_cal) != CAL_CASES * EXPECTED_STATES:
        raise RuntimeError(
            f"Calibration rows/action={len(action_cal)} "
            f"expected={CAL_CASES * EXPECTED_STATES}"
        )

    g = action_cal.sort_values(
        ["harm_prob", "sample_id", "model_state_id"],
        kind="mergesort",
    ).reset_index(drop=True)

    score = g["harm_prob"].to_numpy(float)
    harm = g["r31b3_harm_label"].to_numpy(int)

    # Evaluate only boundaries after the final row sharing each probability.
    c_harm = np.cumsum(harm)
    valid_candidates = []

    for i in range(len(g)):
        if i + 1 < len(g) and score[i + 1] == score[i]:
            continue
        n = i + 1
        h = int(c_harm[i])
        corrected = crc_corrected_risk(h, n)
        if corrected <= ALPHA_PER_ACTION:
            valid_candidates.append(
                {
                    "tau": float(score[i]),
                    "accepted_n": int(n),
                    "harm_count": h,
                    "empirical_risk": float(h / n),
                    "corrected_risk": corrected,
                }
            )

    if not valid_candidates:
        return {
            "tau": float("-inf"),
            "accepted_n": 0,
            "harm_count": 0,
            "empirical_risk": math.nan,
            "corrected_risk": math.inf,
            "available": False,
        }

    # Protocol: choose largest tau satisfying corrected risk budget.
    best = max(valid_candidates, key=lambda x: x["tau"])
    best["available"] = True
    return best


# ---------------------------------------------------------------------
# Controller decisions
# ---------------------------------------------------------------------

def _fixed_order_min(actions: Sequence[str]) -> str:
    return sorted(actions, key=lambda a: ACTION_ORDER[a])[0]


def decide_one_modelcase(
    three_rows: pd.DataFrame,
    thresholds: Mapping[str, float],
) -> List[Dict[str, Any]]:
    if len(three_rows) != len(ACTIONS):
        raise RuntimeError("Decision unit does not contain exactly 3 actions.")

    by_action = {
        str(r.action): r
        for r in three_rows.itertuples(index=False)
    }
    if set(by_action) != set(ACTIONS):
        raise RuntimeError(f"Decision action set={set(by_action)}")

    def truth(action: str):
        r = by_action[action]
        return {
            "delta": float(r.delta_dice),
            "harm": int(r.r31b3_harm_label),
            "benefit": int(r.benefit_label),
            "utility_pred": float(r.utility_pred),
            "harm_prob": float(r.harm_prob),
        }

    out: List[Dict[str, Any]] = []

    def emit(controller: str, chosen: str):
        if chosen == "SOURCE":
            delta = 0.0
            harm = 0
            benefit = 0
            up = 0.0
            hp = 0.0
        else:
            t = truth(chosen)
            delta = t["delta"]
            harm = t["harm"]
            benefit = t["benefit"]
            up = t["utility_pred"]
            hp = t["harm_prob"]
        out.append({
            "controller": controller,
            "chosen_action": chosen,
            "deployed_delta_dice": float(delta),
            "deployed_harm": int(harm),
            "deployed_benefit": int(benefit),
            "chosen_utility_pred": float(up),
            "chosen_harm_prob": float(hp),
            "adapted": int(chosen != "SOURCE"),
        })

    emit("SOURCE_ONLY", "SOURCE")
    emit("ALWAYS_TENT1", "TENT1")
    emit("ALWAYS_PL_CONF90", "PL-CONF90")
    emit("ALWAYS_MEMO", "MEMO-SEG4-1STEP")

    # Utility only: positive predicted utility required, then max utility.
    positive = [
        a for a in ACTIONS
        if truth(a)["utility_pred"] > 0.0
    ]
    if not positive:
        utility_choice = "SOURCE"
    else:
        best_u = max(truth(a)["utility_pred"] for a in positive)
        tied = [
            a for a in positive
            if truth(a)["utility_pred"] == best_u
        ]
        utility_choice = _fixed_order_min(tied)
    emit("UTILITY_ONLY", utility_choice)

    # CRC risk only: safe-by-risk actions only, choose minimum predicted HARM.
    risk_safe = [
        a for a in ACTIONS
        if truth(a)["harm_prob"] <= float(thresholds[a])
    ]
    if not risk_safe:
        risk_choice = "SOURCE"
    else:
        best_p = min(truth(a)["harm_prob"] for a in risk_safe)
        tied = [
            a for a in risk_safe
            if truth(a)["harm_prob"] == best_p
        ]
        risk_choice = _fixed_order_min(tied)
    emit("CRC_RISK_ONLY", risk_choice)

    # Proposed: utility-positive + CRC-safe, choose max utility.
    safe = [
        a for a in ACTIONS
        if truth(a)["utility_pred"] > 0.0
        and truth(a)["harm_prob"] <= float(thresholds[a])
    ]
    if not safe:
        proposed = "SOURCE"
    else:
        best_u = max(truth(a)["utility_pred"] for a in safe)
        tied_u = [
            a for a in safe
            if truth(a)["utility_pred"] == best_u
        ]
        if len(tied_u) == 1:
            proposed = tied_u[0]
        else:
            best_p = min(truth(a)["harm_prob"] for a in tied_u)
            tied_p = [
                a for a in tied_u
                if truth(a)["harm_prob"] == best_p
            ]
            proposed = _fixed_order_min(tied_p)
    emit("R31C_RISK_UTILITY_CRC", proposed)

    # Oracle analysis upper bound: max true DeltaDice among SOURCE=0 and 3 actions.
    deltas = {a: truth(a)["delta"] for a in ACTIONS}
    best_delta = max([0.0] + list(deltas.values()))
    if best_delta <= 0.0:
        oracle = "SOURCE"
    else:
        tied = [
            a for a in ACTIONS
            if deltas[a] == best_delta
        ]
        oracle = _fixed_order_min(tied)
    emit("ORACLE_BEST_ACTION", oracle)

    return out


def evaluate_test_fold(
    test_pred: pd.DataFrame,
    thresholds: Mapping[str, float],
    split_seed: int,
    outer_fold: int,
) -> pd.DataFrame:
    rows = []

    grouped = test_pred.groupby(
        ["sample_id", "model_state_id", "model_family"],
        sort=False,
    )
    expected_units = TEST_CASES * EXPECTED_STATES
    if grouped.ngroups != expected_units:
        raise RuntimeError(
            f"Test model-cases={grouped.ngroups} expected={expected_units}"
        )

    for (sid, state, family), g in grouped:
        decisions = decide_one_modelcase(g, thresholds)
        for d in decisions:
            rows.append({
                "split_seed": int(split_seed),
                "outer_fold": int(outer_fold),
                "sample_id": str(sid),
                "model_state_id": str(state),
                "model_family": str(family),
                **d,
            })

    out = pd.DataFrame(rows)
    expected_rows = expected_units * len(BASELINES)
    if len(out) != expected_rows:
        raise RuntimeError(
            f"Controller decision rows={len(out)} expected={expected_rows}"
        )
    return out


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def controller_metric_row(
    g: pd.DataFrame,
    controller: str,
    split_seed: int | str,
    model_family: str = "ALL",
) -> Dict[str, Any]:
    n = len(g)
    counts = g["chosen_action"].value_counts().to_dict()

    return {
        "split_seed": split_seed,
        "controller": controller,
        "model_family": model_family,
        "model_cases": int(n),
        "mean_deployed_delta_dice": float(g["deployed_delta_dice"].mean()),
        "harm_rate": float(g["deployed_harm"].mean()),
        "benefit_rate": float(g["deployed_benefit"].mean()),
        "adaptation_coverage": float(g["adapted"].mean()),
        "SOURCE_count": int(counts.get("SOURCE", 0)),
        "TENT1_count": int(counts.get("TENT1", 0)),
        "PL_CONF90_count": int(counts.get("PL-CONF90", 0)),
        "MEMO_count": int(counts.get("MEMO-SEG4-1STEP", 0)),
    }


def per_seed_metrics(decisions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (seed, controller), g in decisions.groupby(
        ["split_seed", "controller"],
        sort=True,
    ):
        rows.append(
            controller_metric_row(g, controller, int(seed), "ALL")
        )
    return pd.DataFrame(rows)


def per_family_metrics(decisions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (seed, controller, family), g in decisions.groupby(
        ["split_seed", "controller", "model_family"],
        sort=True,
    ):
        rows.append(
            controller_metric_row(
                g,
                controller,
                int(seed),
                str(family),
            )
        )
    return pd.DataFrame(rows)


def summarize_seed_metrics(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for controller, g in seed_metrics.groupby("controller", sort=True):
        row = {
            "controller": controller,
            "seeds": int(g["split_seed"].nunique()),
        }

        for metric in [
            "mean_deployed_delta_dice",
            "harm_rate",
            "benefit_rate",
            "adaptation_coverage",
        ]:
            vals = g[metric].to_numpy(float)
            row[f"{metric}_mean"] = float(vals.mean())
            row[f"{metric}_std"] = float(vals.std(ddof=1))
            row[f"{metric}_min"] = float(vals.min())
            row[f"{metric}_max"] = float(vals.max())

        for count_col in [
            "SOURCE_count",
            "TENT1_count",
            "PL_CONF90_count",
            "MEMO_count",
        ]:
            vals = g[count_col].to_numpy(float)
            row[f"{count_col}_mean_per_seed"] = float(vals.mean())
            row[f"{count_col}_std_per_seed"] = float(vals.std(ddof=1))

        rows.append(row)

    return pd.DataFrame(rows)


def prevented_harm_and_retained_benefit(
    decisions: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for seed, s in decisions.groupby("split_seed", sort=True):
        prop = s[
            s["controller"] == "R31C_RISK_UTILITY_CRC"
        ][
            [
                "sample_id",
                "model_state_id",
                "deployed_harm",
                "deployed_benefit",
            ]
        ].rename(
            columns={
                "deployed_harm": "prop_harm",
                "deployed_benefit": "prop_benefit",
            }
        )

        for controller in [
            "ALWAYS_TENT1",
            "ALWAYS_PL_CONF90",
            "ALWAYS_MEMO",
        ]:
            b = s[s["controller"] == controller][
                [
                    "sample_id",
                    "model_state_id",
                    "deployed_harm",
                    "deployed_benefit",
                ]
            ].rename(
                columns={
                    "deployed_harm": "base_harm",
                    "deployed_benefit": "base_benefit",
                }
            )
            m = prop.merge(
                b,
                on=["sample_id", "model_state_id"],
                how="inner",
                validate="one_to_one",
            )

            base_harm_n = int(m["base_harm"].sum())
            prevented_n = int(
                ((m["base_harm"] == 1) & (m["prop_harm"] == 0)).sum()
            )

            base_benefit_n = int(m["base_benefit"].sum())
            retained_n = int(
                ((m["base_benefit"] == 1) & (m["prop_benefit"] == 1)).sum()
            )

            rows.append({
                "split_seed": int(seed),
                "baseline": controller,
                "baseline_harm_count": base_harm_n,
                "prevented_harm_count": prevented_n,
                "prevented_harm_fraction": (
                    float(prevented_n / base_harm_n)
                    if base_harm_n > 0
                    else math.nan
                ),
                "baseline_benefit_count": base_benefit_n,
                "retained_benefit_count": retained_n,
                "retained_benefit_fraction": (
                    float(retained_n / base_benefit_n)
                    if base_benefit_n > 0
                    else math.nan
                ),
            })

    return pd.DataFrame(rows)


def feasibility_decision(summary: pd.DataFrame) -> Dict[str, Any]:
    g = summary[
        summary["controller"] == "R31C_RISK_UTILITY_CRC"
    ]
    if len(g) != 1:
        raise RuntimeError("Missing unique R31C summary row.")
    r = g.iloc[0]

    harm = float(r["harm_rate_mean"])
    delta = float(r["mean_deployed_delta_dice_mean"])
    coverage = float(r["adaptation_coverage_mean"])

    pass_harm = harm <= FEASIBILITY_HARM_MAX
    pass_delta = delta > FEASIBILITY_MEAN_DELTA_MIN_EXCLUSIVE
    pass_coverage = coverage >= FEASIBILITY_COVERAGE_MIN

    if pass_harm and pass_delta and pass_coverage:
        decision = "PASS_CONTROLLER_FEASIBILITY"
        next_stage = "R31C2_CONTROLLER_INTERPRETATION_AND_FROZEN_EXTERNAL_PLAN"
    else:
        decision = "NO_PASS_CONTROLLER_FEASIBILITY"
        next_stage = "STOP_CONTROLLER_CLAIM_AND_INTERPRET"

    return {
        "decision": decision,
        "criteria": {
            "harm_rate_at_most": FEASIBILITY_HARM_MAX,
            "mean_deployed_delta_dice_strictly_greater_than": (
                FEASIBILITY_MEAN_DELTA_MIN_EXCLUSIVE
            ),
            "adaptation_coverage_at_least": FEASIBILITY_COVERAGE_MIN,
        },
        "observed": {
            "harm_rate": harm,
            "mean_deployed_delta_dice": delta,
            "adaptation_coverage": coverage,
        },
        "passes": {
            "harm": bool(pass_harm),
            "mean_delta": bool(pass_delta),
            "coverage": bool(pass_coverage),
        },
        "next": next_stage,
    }


def make_inventory(out: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(out.rglob("*")):
        if p.is_file() and not p.name.endswith(".tmp"):
            rows.append({
                "relative_path": str(p.relative_to(out)),
                "bytes": int(p.stat().st_size),
                "sha256": sha256_file(p),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    print("=" * 124)
    print("SafeTTA R31C1 Nested Multi-Action Risk-Controlled Controller OOF")
    print("Version                  :", VERSION)
    print("Nested partitions        : 5 seeds x 5 folds")
    print("Train / Cal / Test       : 480 / 160 / 160 physical cases")
    print("Candidate actions        :", list(ACTIONS))
    print("Total HARM budget        :", ALPHA_TOTAL)
    print("Per-action HARM budget   :", ALPHA_PER_ACTION)
    print("Controller tuning        : NO")
    print("EndoTect / external      : NO / NO")
    print("=" * 124)

    upstream = verify_upstream()
    qnames, dnames = load_feature_schema()
    panel, lineage = load_three_action_panel(qnames, dnames)

    partitions = pd.read_csv(
        R31C0_PARTITIONS,
        dtype={
            "sample_id": str,
            "role": str,
        },
    )

    expected_partition_rows = len(SPLIT_SEEDS) * N_FOLDS * EXPECTED_CASES
    if len(partitions) != expected_partition_rows:
        raise RuntimeError(
            f"Partition rows={len(partitions)} expected={expected_partition_rows}"
        )

    out = args.output_dir.resolve()
    if out.exists():
        raise RuntimeError(
            f"Output exists; do not overwrite R31C1 results: {out}"
        )
    out.mkdir(parents=True, exist_ok=False)

    # Lock exact input lineage before controller fitting.
    input_lock = {
        "status": "PASS_R31C1_INPUT_LOCK",
        "version": VERSION,
        "R31C0_script_sha256": EXPECTED_R31C0_SCRIPT_SHA256,
        "R31C0_protocol_sha256": EXPECTED_R31C0_PROTOCOL_SHA256,
        "R31C0_final_sha256": EXPECTED_R31C0_FINAL_SHA256,
        "R31C0_partitions_sha256": sha256_file(R31C0_PARTITIONS),
        "R31B3_final_sha256": EXPECTED_R31B3_FINAL_SHA256,
        "feature_schema": {
            "q_source_dim": len(qnames),
            "delta_semantic_dim": len(dnames),
        },
        "lineage": lineage,
        "physical_cases": int(panel["sample_id"].nunique()),
        "model_states": int(panel["model_state_id"].nunique()),
        "actions": panel["action"].value_counts().to_dict(),
        "external_data_access": False,
    }
    input_lock_path = out / "R31C1_INPUT_LOCK.json"
    atomic_json(input_lock_path, input_lock)

    action_cats, family_action_cats = global_category_schema(panel)

    all_decisions = []
    threshold_rows = []
    fold_audit_rows = []

    total_partitions = len(SPLIT_SEEDS) * N_FOLDS
    pbar = tqdm(
        total=total_partitions,
        desc="R31C1 nested controller partitions",
        unit="partition",
        dynamic_ncols=True,
    )

    for seed in SPLIT_SEEDS:
        seed_part = partitions[
            partitions["split_seed"] == seed
        ].copy()

        for outer in range(N_FOLDS):
            part = seed_part[
                seed_part["outer_fold"] == outer
            ].copy()

            train_ids = set(
                part.loc[part["role"] == "TRAIN", "sample_id"].astype(str)
            )
            cal_ids = set(
                part.loc[
                    part["role"] == "CALIBRATION",
                    "sample_id",
                ].astype(str)
            )
            test_ids = set(
                part.loc[part["role"] == "TEST", "sample_id"].astype(str)
            )

            if len(train_ids) != TRAIN_CASES:
                raise RuntimeError("TRAIN physical-case count drift.")
            if len(cal_ids) != CAL_CASES:
                raise RuntimeError("CAL physical-case count drift.")
            if len(test_ids) != TEST_CASES:
                raise RuntimeError("TEST physical-case count drift.")
            if train_ids & cal_ids or train_ids & test_ids or cal_ids & test_ids:
                raise RuntimeError("Nested physical-case role overlap.")

            train = panel[panel["sample_id"].isin(train_ids)].copy()
            cal = panel[panel["sample_id"].isin(cal_ids)].copy()
            test = panel[panel["sample_id"].isin(test_ids)].copy()

            expected_train_rows = TRAIN_CASES * EXPECTED_STATES * len(ACTIONS)
            expected_cal_rows = CAL_CASES * EXPECTED_STATES * len(ACTIONS)
            expected_test_rows = TEST_CASES * EXPECTED_STATES * len(ACTIONS)

            if len(train) != expected_train_rows:
                raise RuntimeError(f"TRAIN rows={len(train)}")
            if len(cal) != expected_cal_rows:
                raise RuntimeError(f"CAL rows={len(cal)}")
            if len(test) != expected_test_rows:
                raise RuntimeError(f"TEST rows={len(test)}")

            utility, harm = fit_models(
                train,
                qnames,
                dnames,
                action_cats,
                family_action_cats,
            )

            cal_pred = predict_models(
                cal,
                utility,
                harm,
                qnames,
                dnames,
                action_cats,
                family_action_cats,
            )
            test_pred = predict_models(
                test,
                utility,
                harm,
                qnames,
                dnames,
                action_cats,
                family_action_cats,
            )

            thresholds: Dict[str, float] = {}

            for action in ACTIONS:
                action_cal = cal_pred[
                    cal_pred["action"] == action
                ].copy()
                info = select_crc_threshold(action_cal)
                thresholds[action] = float(info["tau"])

                threshold_rows.append({
                    "split_seed": int(seed),
                    "outer_fold": int(outer),
                    "action": action,
                    "alpha_per_action": ALPHA_PER_ACTION,
                    **info,
                })

            decisions = evaluate_test_fold(
                test_pred,
                thresholds,
                int(seed),
                int(outer),
            )
            all_decisions.append(decisions)

            fold_audit_rows.append({
                "split_seed": int(seed),
                "outer_fold": int(outer),
                "train_cases": len(train_ids),
                "calibration_cases": len(cal_ids),
                "test_cases": len(test_ids),
                "train_rows": len(train),
                "calibration_rows": len(cal),
                "test_rows": len(test),
                "tau_TENT1": thresholds["TENT1"],
                "tau_PL_CONF90": thresholds["PL-CONF90"],
                "tau_MEMO": thresholds["MEMO-SEG4-1STEP"],
            })

            pbar.set_postfix(
                seed=seed,
                fold=outer,
                tent=f"{thresholds['TENT1']:.3g}",
                pl=f"{thresholds['PL-CONF90']:.3g}",
                memo=f"{thresholds['MEMO-SEG4-1STEP']:.3g}",
            )
            pbar.update(1)

    pbar.close()

    decisions = pd.concat(all_decisions, ignore_index=True)
    thresholds = pd.DataFrame(threshold_rows)
    fold_audit = pd.DataFrame(fold_audit_rows)

    expected_decision_rows = (
        len(SPLIT_SEEDS)
        * EXPECTED_CASES
        * EXPECTED_STATES
        * len(BASELINES)
    )
    if len(decisions) != expected_decision_rows:
        raise RuntimeError(
            f"OOF controller rows={len(decisions)} "
            f"expected={expected_decision_rows}"
        )

    # Each seed must be a complete OOF panel: every case/state once/controller.
    for seed in SPLIT_SEEDS:
        s = decisions[decisions["split_seed"] == seed]
        for controller in BASELINES:
            g = s[s["controller"] == controller]
            expected = EXPECTED_CASES * EXPECTED_STATES
            if len(g) != expected:
                raise RuntimeError(
                    f"seed={seed} controller={controller} rows={len(g)} "
                    f"expected={expected}"
                )
            if g.duplicated(["sample_id", "model_state_id"]).any():
                raise RuntimeError(
                    f"Duplicate OOF unit seed={seed} controller={controller}"
                )

    seed_metrics = per_seed_metrics(decisions)
    family_metrics = per_family_metrics(decisions)
    summary = summarize_seed_metrics(seed_metrics)
    secondary = prevented_harm_and_retained_benefit(decisions)
    feasibility = feasibility_decision(summary)

    # Threshold summary.
    threshold_summary_rows = []
    for action, g in thresholds.groupby("action", sort=True):
        finite = g[np.isfinite(g["tau"].to_numpy(float))]
        threshold_summary_rows.append({
            "action": action,
            "partitions": int(len(g)),
            "available_partitions": int(g["available"].sum()),
            "availability_fraction": float(g["available"].mean()),
            "tau_mean_finite": (
                float(finite["tau"].mean()) if len(finite) else math.nan
            ),
            "tau_min_finite": (
                float(finite["tau"].min()) if len(finite) else math.nan
            ),
            "tau_max_finite": (
                float(finite["tau"].max()) if len(finite) else math.nan
            ),
            "accepted_n_mean": float(g["accepted_n"].mean()),
            "corrected_risk_mean_available": (
                float(
                    g.loc[g["available"], "corrected_risk"].mean()
                )
                if bool(g["available"].any())
                else math.nan
            ),
        })
    threshold_summary = pd.DataFrame(threshold_summary_rows)

    # Save exact artifacts.
    decisions_path = out / "R31C1_OOF_CONTROLLER_DECISIONS.csv"
    threshold_path = out / "R31C1_FOLD_ACTION_CRC_THRESHOLDS.csv"
    threshold_summary_path = out / "R31C1_CRC_THRESHOLD_SUMMARY.csv"
    fold_audit_path = out / "R31C1_NESTED_FOLD_AUDIT.csv"
    seed_metrics_path = out / "R31C1_PER_SEED_CONTROLLER_METRICS.csv"
    family_metrics_path = out / "R31C1_PER_FAMILY_CONTROLLER_METRICS.csv"
    summary_path = out / "R31C1_CONTROLLER_SUMMARY.csv"
    secondary_path = out / "R31C1_PREVENTED_HARM_RETAINED_BENEFIT.csv"
    feasibility_path = out / "R31C1_FEASIBILITY_DECISION.json"

    atomic_csv(decisions, decisions_path)
    atomic_csv(thresholds, threshold_path)
    atomic_csv(threshold_summary, threshold_summary_path)
    atomic_csv(fold_audit, fold_audit_path)
    atomic_csv(seed_metrics, seed_metrics_path)
    atomic_csv(family_metrics, family_metrics_path)
    atomic_csv(summary, summary_path)
    atomic_csv(secondary, secondary_path)

    feasibility.update({
        "version": VERSION,
        "R31C0_protocol_sha256": EXPECTED_R31C0_PROTOCOL_SHA256,
        "R31C0_final_sha256": EXPECTED_R31C0_FINAL_SHA256,
        "complete_OOF_seed_panels": len(SPLIT_SEEDS),
        "nested_partitions": total_partitions,
        "alpha_total": ALPHA_TOTAL,
        "alpha_per_action": ALPHA_PER_ACTION,
        "external_data_access": False,
    })
    atomic_json(feasibility_path, feasibility)

    inventory = make_inventory(out)
    inventory_path = out / "R31C1_SHA256_INVENTORY.csv"
    atomic_csv(inventory, inventory_path)

    final = {
        "status": "PASS_R31C1_NESTED_MULTI_ACTION_RISK_CONTROLLER_OOF_COMPLETE",
        "version": VERSION,
        "decision": feasibility["decision"],
        "R31C0_protocol_sha256": EXPECTED_R31C0_PROTOCOL_SHA256,
        "R31C0_final_sha256": EXPECTED_R31C0_FINAL_SHA256,
        "R31B3_final_sha256": EXPECTED_R31B3_FINAL_SHA256,
        "physical_cases": EXPECTED_CASES,
        "model_states": EXPECTED_STATES,
        "actions": list(ACTIONS),
        "complete_OOF_seed_panels": len(SPLIT_SEEDS),
        "nested_partitions": total_partitions,
        "controller_summary_sha256": sha256_file(summary_path),
        "thresholds_sha256": sha256_file(threshold_path),
        "OOF_decisions_sha256": sha256_file(decisions_path),
        "feasibility_decision_sha256": sha256_file(feasibility_path),
        "controller_hyperparameter_search": False,
        "external_data_access": False,
        "next": feasibility["next"],
    }
    final_path = out / "R31C1_FINAL_LOCK.json"
    atomic_json(final_path, final)

    print("\n" + "=" * 124)
    print("R31C1 CRC THRESHOLD AVAILABILITY")
    print("=" * 124)
    print(threshold_summary.to_string(index=False))

    print("\n" + "=" * 124)
    print("R31C1 CONTROLLER SUMMARY (MEAN +/- STD OVER 5 COMPLETE OOF SEEDS)")
    print("=" * 124)

    show_cols = [
        "controller",
        "mean_deployed_delta_dice_mean",
        "mean_deployed_delta_dice_std",
        "harm_rate_mean",
        "harm_rate_std",
        "benefit_rate_mean",
        "adaptation_coverage_mean",
        "SOURCE_count_mean_per_seed",
        "TENT1_count_mean_per_seed",
        "PL_CONF90_count_mean_per_seed",
        "MEMO_count_mean_per_seed",
    ]
    print(summary[show_cols].to_string(index=False))

    print("\nR31C1 FEASIBILITY:", feasibility["decision"])
    print(
        "  R31C HARM rate              :",
        feasibility["observed"]["harm_rate"],
        "<=",
        FEASIBILITY_HARM_MAX,
        ":",
        feasibility["passes"]["harm"],
    )
    print(
        "  R31C mean deployed DeltaDice:",
        feasibility["observed"]["mean_deployed_delta_dice"],
        "> 0 :",
        feasibility["passes"]["mean_delta"],
    )
    print(
        "  R31C adaptation coverage    :",
        feasibility["observed"]["adaptation_coverage"],
        ">=",
        FEASIBILITY_COVERAGE_MIN,
        ":",
        feasibility["passes"]["coverage"],
    )

    print("\nFINAL STATUS : PASS_R31C1_NESTED_MULTI_ACTION_RISK_CONTROLLER_OOF_COMPLETE")
    print("Controller hyperparameter search : NO")
    print("EndoTect / external access       : NO / NO")
    print("Final lock SHA256                :", sha256_file(final_path))
    print("Output                           :", out)
    print("NEXT                             :", feasibility["next"])
    print("=" * 124)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
