#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SafeTTA R31D1
Nested Utility-First Policy-Level CRC OOF

Executes the exact post-R31C developmental protocol frozen in R31D0.

Nested evaluation:
  5 split seeds x 5 outer folds
  480 TRAIN / 160 CALIBRATION / 160 TEST physical cases

TRAIN:
  shared utility = StandardScaler + Ridge(alpha=1)
  shared HARM    = StandardScaler + balanced LogisticRegression(C=1)

Per model-case candidate rule:
  among TENT1 / PL-CONF90 / MEMO-SEG4-1STEP with predicted utility > 0,
  choose maximum predicted utility;
  tie -> lower predicted HARM probability -> fixed action order;
  if none positive -> SOURCE.

CALIBRATION:
  keep one utility-selected adaptation candidate per model-case when one exists;
  calibrate ONE policy-level CRC threshold tau at alpha=0.020:
      corrected risk = (harm_count + 1) / (n + 1)
  choose largest tau satisfying corrected risk <= 0.020;
  if none qualifies, tau=-inf.

TEST:
  no positive-utility candidate -> SOURCE
  candidate p(HARM) <= tau    -> commit candidate
  candidate p(HARM) > tau     -> SOURCE rollback

Primary feasibility gate (unchanged from R31C0):
  HARM rate <= 0.020
  mean deployed DeltaDice > 0
  adaptation coverage >= 0.05

R31D is explicitly POST-R31C DEVELOPMENTAL. Success here is internal evidence
only and requires a new untouched external confirmation before a prospective
external claim.

No hyperparameter search.
No EndoTect.
No external cohort.
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

VERSION = "2026-09-12-R31D1-v1"

ROOT = Path(r"F:\MEDSEG_SAFETTA")
CODE = ROOT / "code"

# ---------------------------------------------------------------------
# Exact upstream locks
# ---------------------------------------------------------------------

R31D0_SCRIPT = CODE / "Q1_R31D0_policy_level_crc_protocol_lock_v1.py"
EXPECTED_R31D0_SCRIPT_SHA256 = (
    "7cea998f8a64e2e4346782a9448477ad5d200ca89a735f12467c44fa0b685a41"
)

R31D0_DIR = ROOT / "R31D0_policy_level_crc_protocol_lock_v1"
R31D0_PROTOCOL = R31D0_DIR / "R31D0_POLICY_LEVEL_CRC_PROTOCOL_LOCK.json"
EXPECTED_R31D0_PROTOCOL_SHA256 = (
    "78d2c878009f420f301a6afb400e3e6dc4666667f1be4323dee853823aa614e3"
)
R31D0_FINAL = R31D0_DIR / "R31D0_FINAL_LOCK.json"
EXPECTED_R31D0_FINAL_SHA256 = (
    "070af53b502c1135b6a9d1654f782fba94a08282af27270b4e7650db8b775575"
)

R31C1_SCRIPT = CODE / "Q1_R31C1_nested_multi_action_risk_controller_oof_v1.py"
EXPECTED_R31C1_SCRIPT_SHA256 = (
    "29b5b8c3460fa371194f84936afcc67ad2957c61574654dddfbedaf66ce353a1"
)

R31C1_DIR = ROOT / "R31C1_nested_multi_action_risk_controller_oof_v1"
R31C1_FINAL = R31C1_DIR / "R31C1_FINAL_LOCK.json"
EXPECTED_R31C1_FINAL_SHA256 = (
    "ef217393ac5d3f09cbd39273cc1442229feb503c9c0cad7d25ea35ca614bd23e"
)
R31C1_DECISIONS = R31C1_DIR / "R31C1_OOF_CONTROLLER_DECISIONS.csv"

R31C0_DIR = ROOT / "R31C0_multi_action_risk_controller_protocol_lock_v1"
R31C0_PARTITIONS = R31C0_DIR / "R31C0_NESTED_480_160_160_PARTITIONS.csv"

DEFAULT_OUT = ROOT / "R31D1_nested_utility_first_policy_level_crc_oof_v1"

# ---------------------------------------------------------------------
# Frozen R31D0 protocol constants
# ---------------------------------------------------------------------

ACTIONS = ("TENT1", "PL-CONF90", "MEMO-SEG4-1STEP")
ACTION_ORDER = {a: i for i, a in enumerate(ACTIONS)}

SPLIT_SEEDS = [20260912, 20260913, 20260914, 20260915, 20260916]
N_FOLDS = 5
TRAIN_CASES = 480
CAL_CASES = 160
TEST_CASES = 160
EXPECTED_CASES = 800
EXPECTED_STATES = 9

ALPHA_POLICY = 0.020

PRIMARY_CONTROLLER = "R31D_UTILITY_FIRST_POLICY_CRC"
HISTORICAL_R31C = "R31C_RISK_UTILITY_CRC"

BASELINES = (
    "SOURCE_ONLY",
    "ALWAYS_TENT1",
    "ALWAYS_PL_CONF90",
    "ALWAYS_MEMO",
    "UTILITY_ONLY",
    PRIMARY_CONTROLLER,
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
            f"{label} SHA mismatch\nexpected={expected}\n"
            f"observed={got}\npath={path}"
        )
    return got


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import module: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


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
        raise RuntimeError(f"External path forbidden in R31D1: {path}")


# ---------------------------------------------------------------------
# Upstream validation
# ---------------------------------------------------------------------

def verify_upstream() -> Dict[str, Any]:
    require_sha(R31D0_SCRIPT, EXPECTED_R31D0_SCRIPT_SHA256, "R31D0 script")
    require_sha(R31D0_PROTOCOL, EXPECTED_R31D0_PROTOCOL_SHA256, "R31D0 protocol")
    require_sha(R31D0_FINAL, EXPECTED_R31D0_FINAL_SHA256, "R31D0 final")
    require_sha(R31C1_SCRIPT, EXPECTED_R31C1_SCRIPT_SHA256, "R31C1 script")
    require_sha(R31C1_FINAL, EXPECTED_R31C1_FINAL_SHA256, "R31C1 final")

    protocol = json.loads(R31D0_PROTOCOL.read_text(encoding="utf-8"))
    final = json.loads(R31D0_FINAL.read_text(encoding="utf-8"))
    c1 = json.loads(R31C1_FINAL.read_text(encoding="utf-8"))

    if final.get("status") != "PASS_R31D0_POLICY_LEVEL_CRC_PROTOCOL_LOCK_COMPLETE":
        raise RuntimeError("R31D0 final status changed.")
    if str(final.get("scientific_status")) != "POST_R31C_DEVELOPMENTAL":
        raise RuntimeError("R31D0 scientific-status flag changed.")
    if not math.isclose(
        float(final.get("alpha_policy")),
        ALPHA_POLICY,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise RuntimeError("R31D0 alpha_policy changed.")

    if protocol["candidate_selection"]["eligible_action"] != "predicted utility > 0":
        raise RuntimeError("R31D0 utility eligibility rule changed.")
    if float(protocol["policy_level_crc"]["alpha_policy"]) != ALPHA_POLICY:
        raise RuntimeError("R31D0 policy CRC alpha changed.")
    if bool(
        protocol["scientific_status"]["prospective_confirmation_claim_allowed"]
    ):
        raise RuntimeError("R31D0 incorrectly permits prospective claim.")

    partition_sha = str(final.get("partition_sha256", ""))
    if not R31C0_PARTITIONS.is_file():
        raise FileNotFoundError(R31C0_PARTITIONS)
    if sha256_file(R31C0_PARTITIONS) != partition_sha:
        raise RuntimeError("Frozen R31C0 partition SHA changed.")

    if not R31C1_DECISIONS.is_file():
        raise FileNotFoundError(R31C1_DECISIONS)
    if sha256_file(R31C1_DECISIONS) != str(c1.get("OOF_decisions_sha256")):
        raise RuntimeError("Historical R31C1 decision table SHA changed.")

    return {
        "protocol": protocol,
        "final": final,
        "c1_final": c1,
        "partition_sha256": partition_sha,
        "historical_r31c_decisions_sha256": sha256_file(R31C1_DECISIONS),
    }


# ---------------------------------------------------------------------
# Exact R31C1 model/data implementation reuse
# ---------------------------------------------------------------------

def load_r31c1_helpers():
    return import_module(R31C1_SCRIPT, "r31d1_r31c1_exact")


def validate_partitions(partitions: pd.DataFrame):
    required = {"split_seed", "outer_fold", "sample_id", "role"}
    missing = sorted(required.difference(partitions.columns))
    if missing:
        raise RuntimeError(f"Partition missing columns={missing}")

    expected = len(SPLIT_SEEDS) * N_FOLDS * EXPECTED_CASES
    if len(partitions) != expected:
        raise RuntimeError(f"Partition rows={len(partitions)} expected={expected}")

    if set(partitions["split_seed"].astype(int).unique()) != set(SPLIT_SEEDS):
        raise RuntimeError("Partition split-seed set changed.")
    if set(partitions["outer_fold"].astype(int).unique()) != set(range(N_FOLDS)):
        raise RuntimeError("Partition outer-fold set changed.")

    for seed in SPLIT_SEEDS:
        for fold in range(N_FOLDS):
            p = partitions[
                (partitions["split_seed"].astype(int) == seed)
                & (partitions["outer_fold"].astype(int) == fold)
            ]
            counts = p["role"].value_counts().to_dict()
            if int(counts.get("TRAIN", 0)) != TRAIN_CASES:
                raise RuntimeError("TRAIN count changed.")
            if int(counts.get("CALIBRATION", 0)) != CAL_CASES:
                raise RuntimeError("CAL count changed.")
            if int(counts.get("TEST", 0)) != TEST_CASES:
                raise RuntimeError("TEST count changed.")


# ---------------------------------------------------------------------
# Utility-first candidate selection
# ---------------------------------------------------------------------

def choose_candidate(three_rows: pd.DataFrame) -> Dict[str, Any]:
    if len(three_rows) != len(ACTIONS):
        raise RuntimeError("Model-case does not contain exactly 3 actions.")
    if set(three_rows["action"].astype(str)) != set(ACTIONS):
        raise RuntimeError("Model-case action set drift.")

    positive = three_rows[
        three_rows["utility_pred"].to_numpy(float) > 0.0
    ].copy()

    if len(positive) == 0:
        return {
            "candidate_action": "SOURCE",
            "candidate_exists": False,
            "candidate_utility_pred": 0.0,
            "candidate_harm_prob": 0.0,
            "candidate_delta_dice": 0.0,
            "candidate_harm": 0,
            "candidate_benefit": 0,
        }

    max_u = float(positive["utility_pred"].max())
    tied = positive[
        positive["utility_pred"].to_numpy(float) == max_u
    ].copy()

    if len(tied) > 1:
        min_p = float(tied["harm_prob"].min())
        tied = tied[
            tied["harm_prob"].to_numpy(float) == min_p
        ].copy()

    if len(tied) > 1:
        tied = tied.assign(
            __action_order=tied["action"].map(ACTION_ORDER).astype(int)
        ).sort_values("__action_order", kind="mergesort")

    r = tied.iloc[0]
    return {
        "candidate_action": str(r["action"]),
        "candidate_exists": True,
        "candidate_utility_pred": float(r["utility_pred"]),
        "candidate_harm_prob": float(r["harm_prob"]),
        "candidate_delta_dice": float(r["delta_dice"]),
        "candidate_harm": int(r["r31b3_harm_label"]),
        "candidate_benefit": int(r["benefit_label"]),
    }


def build_candidates(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = pred.groupby(
        ["sample_id", "model_state_id", "model_family"],
        sort=False,
    )

    for (sid, state, family), g in grouped:
        c = choose_candidate(g)
        rows.append({
            "sample_id": str(sid),
            "model_state_id": str(state),
            "model_family": str(family),
            **c,
        })

    out = pd.DataFrame(rows)
    expected_units = pred["sample_id"].nunique() * EXPECTED_STATES
    if len(out) != expected_units:
        raise RuntimeError(
            f"Candidate model-cases={len(out)} expected={expected_units}"
        )
    return out


# ---------------------------------------------------------------------
# Policy-level CRC calibration
# ---------------------------------------------------------------------

def corrected_risk(harm_count: int, n: int) -> float:
    if n <= 0:
        return math.inf
    return float((int(harm_count) + 1) / (int(n) + 1))


def select_policy_tau(cal_candidates: pd.DataFrame) -> Dict[str, Any]:
    cand = cal_candidates[
        cal_candidates["candidate_exists"].astype(bool)
    ].copy()

    total_modelcases = int(len(cal_candidates))
    candidate_modelcases = int(len(cand))

    if candidate_modelcases == 0:
        return {
            "tau": float("-inf"),
            "available": False,
            "calibration_modelcases": total_modelcases,
            "positive_utility_candidates": 0,
            "candidate_fraction": 0.0,
            "accepted_n": 0,
            "harm_count": 0,
            "empirical_risk": math.nan,
            "corrected_risk": math.inf,
        }

    g = cand.sort_values(
        [
            "candidate_harm_prob",
            "sample_id",
            "model_state_id",
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    p = g["candidate_harm_prob"].to_numpy(float)
    y = g["candidate_harm"].to_numpy(int)
    cumulative_harm = np.cumsum(y)

    qualifying = []

    for i in range(len(g)):
        # Thresholds are evaluated at the end of equal-probability blocks.
        if i + 1 < len(g) and p[i + 1] == p[i]:
            continue

        n = i + 1
        h = int(cumulative_harm[i])
        risk = corrected_risk(h, n)

        if risk <= ALPHA_POLICY:
            qualifying.append({
                "tau": float(p[i]),
                "accepted_n": int(n),
                "harm_count": h,
                "empirical_risk": float(h / n),
                "corrected_risk": risk,
            })

    if not qualifying:
        return {
            "tau": float("-inf"),
            "available": False,
            "calibration_modelcases": total_modelcases,
            "positive_utility_candidates": candidate_modelcases,
            "candidate_fraction": float(candidate_modelcases / total_modelcases),
            "accepted_n": 0,
            "harm_count": 0,
            "empirical_risk": math.nan,
            "corrected_risk": math.inf,
        }

    # Frozen R31D0 rule: largest tau satisfying policy-level CRC.
    best = max(qualifying, key=lambda x: x["tau"])
    return {
        "available": True,
        "calibration_modelcases": total_modelcases,
        "positive_utility_candidates": candidate_modelcases,
        "candidate_fraction": float(candidate_modelcases / total_modelcases),
        **best,
    }


# ---------------------------------------------------------------------
# Test deployment and baselines
# ---------------------------------------------------------------------

def _oracle_choice(g: pd.DataFrame) -> str:
    deltas = {
        str(r.action): float(r.delta_dice)
        for r in g.itertuples(index=False)
    }
    best = max([0.0] + list(deltas.values()))
    if best <= 0.0:
        return "SOURCE"

    tied = [a for a in ACTIONS if deltas[a] == best]
    return sorted(tied, key=lambda a: ACTION_ORDER[a])[0]


def _emit_truth(
    rows: List[Dict[str, Any]],
    controller: str,
    chosen: str,
    truth_by_action: Mapping[str, Any],
    seed: int,
    fold: int,
    sid: str,
    state: str,
    family: str,
    candidate: Mapping[str, Any] | None = None,
    tau: float | None = None,
):
    if chosen == "SOURCE":
        delta = 0.0
        harm = 0
        benefit = 0
    else:
        r = truth_by_action[chosen]
        delta = float(r.delta_dice)
        harm = int(r.r31b3_harm_label)
        benefit = int(r.benefit_label)

    row = {
        "split_seed": int(seed),
        "outer_fold": int(fold),
        "sample_id": str(sid),
        "model_state_id": str(state),
        "model_family": str(family),
        "controller": controller,
        "chosen_action": chosen,
        "deployed_delta_dice": delta,
        "deployed_harm": harm,
        "deployed_benefit": benefit,
        "adapted": int(chosen != "SOURCE"),
    }

    if candidate is not None:
        row.update({
            "utility_candidate_action": str(candidate["candidate_action"]),
            "utility_candidate_exists": int(bool(candidate["candidate_exists"])),
            "utility_candidate_pred": float(candidate["candidate_utility_pred"]),
            "utility_candidate_harm_prob": float(candidate["candidate_harm_prob"]),
            "policy_tau": float(tau) if tau is not None else math.nan,
        })
    else:
        row.update({
            "utility_candidate_action": "",
            "utility_candidate_exists": 0,
            "utility_candidate_pred": math.nan,
            "utility_candidate_harm_prob": math.nan,
            "policy_tau": math.nan,
        })

    rows.append(row)


def evaluate_test(
    test_pred: pd.DataFrame,
    tau: float,
    seed: int,
    fold: int,
) -> pd.DataFrame:
    candidates = build_candidates(test_pred)
    candidate_map = {
        (str(r.sample_id), str(r.model_state_id)): r._asdict()
        for r in candidates.itertuples(index=False)
    }

    rows: List[Dict[str, Any]] = []
    grouped = test_pred.groupby(
        ["sample_id", "model_state_id", "model_family"],
        sort=False,
    )

    expected_units = TEST_CASES * EXPECTED_STATES
    if grouped.ngroups != expected_units:
        raise RuntimeError(
            f"TEST model-cases={grouped.ngroups} expected={expected_units}"
        )

    for (sid, state, family), g in grouped:
        truth = {
            str(r.action): r
            for r in g.itertuples(index=False)
        }
        if set(truth) != set(ACTIONS):
            raise RuntimeError("TEST action set drift.")

        key = (str(sid), str(state))
        cand = candidate_map[key]

        # Standard baselines.
        _emit_truth(
            rows, "SOURCE_ONLY", "SOURCE", truth,
            seed, fold, sid, state, family,
        )
        _emit_truth(
            rows, "ALWAYS_TENT1", "TENT1", truth,
            seed, fold, sid, state, family,
        )
        _emit_truth(
            rows, "ALWAYS_PL_CONF90", "PL-CONF90", truth,
            seed, fold, sid, state, family,
        )
        _emit_truth(
            rows, "ALWAYS_MEMO", "MEMO-SEG4-1STEP", truth,
            seed, fold, sid, state, family,
        )

        utility_choice = (
            str(cand["candidate_action"])
            if bool(cand["candidate_exists"])
            else "SOURCE"
        )
        _emit_truth(
            rows, "UTILITY_ONLY", utility_choice, truth,
            seed, fold, sid, state, family,
            candidate=cand,
        )

        if (
            bool(cand["candidate_exists"])
            and float(cand["candidate_harm_prob"]) <= float(tau)
        ):
            proposed = str(cand["candidate_action"])
        else:
            proposed = "SOURCE"

        _emit_truth(
            rows, PRIMARY_CONTROLLER, proposed, truth,
            seed, fold, sid, state, family,
            candidate=cand,
            tau=tau,
        )

        oracle = _oracle_choice(g)
        _emit_truth(
            rows, "ORACLE_BEST_ACTION", oracle, truth,
            seed, fold, sid, state, family,
        )

    out = pd.DataFrame(rows)
    expected = expected_units * len(BASELINES)
    if len(out) != expected:
        raise RuntimeError(
            f"TEST decision rows={len(out)} expected={expected}"
        )
    return out


# ---------------------------------------------------------------------
# Historical R31C extraction
# ---------------------------------------------------------------------

def load_historical_r31c() -> pd.DataFrame:
    d = pd.read_csv(R31C1_DECISIONS, low_memory=False)
    required = {
        "split_seed",
        "outer_fold",
        "sample_id",
        "model_state_id",
        "model_family",
        "controller",
        "chosen_action",
        "deployed_delta_dice",
        "deployed_harm",
        "deployed_benefit",
        "adapted",
    }
    missing = sorted(required.difference(d.columns))
    if missing:
        raise RuntimeError(f"Historical R31C decisions missing={missing}")

    h = d[d["controller"] == HISTORICAL_R31C].copy()
    expected = len(SPLIT_SEEDS) * EXPECTED_CASES * EXPECTED_STATES
    if len(h) != expected:
        raise RuntimeError(
            f"Historical R31C rows={len(h)} expected={expected}"
        )

    h["controller"] = "R31C_RISK_UTILITY_CRC_HISTORICAL"
    return h


# ---------------------------------------------------------------------
# Metrics and comparisons
# ---------------------------------------------------------------------

def metric_row(
    g: pd.DataFrame,
    controller: str,
    seed: int | str,
) -> Dict[str, Any]:
    counts = g["chosen_action"].value_counts().to_dict()
    return {
        "split_seed": seed,
        "controller": controller,
        "model_cases": int(len(g)),
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
        rows.append(metric_row(g, str(controller), int(seed)))
    return pd.DataFrame(rows)


def summarize(seed_metrics: pd.DataFrame) -> pd.DataFrame:
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
            v = g[metric].to_numpy(float)
            row[f"{metric}_mean"] = float(v.mean())
            row[f"{metric}_std"] = float(v.std(ddof=1))
            row[f"{metric}_min"] = float(v.min())
            row[f"{metric}_max"] = float(v.max())

        for c in [
            "SOURCE_count",
            "TENT1_count",
            "PL_CONF90_count",
            "MEMO_count",
        ]:
            v = g[c].to_numpy(float)
            row[f"{c}_mean_per_seed"] = float(v.mean())
            row[f"{c}_std_per_seed"] = float(v.std(ddof=1))

        rows.append(row)

    return pd.DataFrame(rows)


def compare_r31d_vs_r31c(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    d = seed_metrics[
        seed_metrics["controller"] == PRIMARY_CONTROLLER
    ].copy()
    c = seed_metrics[
        seed_metrics["controller"]
        == "R31C_RISK_UTILITY_CRC_HISTORICAL"
    ].copy()

    m = d.merge(
        c,
        on="split_seed",
        suffixes=("_r31d", "_r31c"),
        validate="one_to_one",
    )

    rows = []
    for metric in [
        "mean_deployed_delta_dice",
        "harm_rate",
        "adaptation_coverage",
        "benefit_rate",
    ]:
        delta = (
            m[f"{metric}_r31d"].to_numpy(float)
            - m[f"{metric}_r31c"].to_numpy(float)
        )
        rows.append({
            "metric": metric,
            "mean_R31D": float(m[f"{metric}_r31d"].mean()),
            "mean_R31C": float(m[f"{metric}_r31c"].mean()),
            "mean_delta_R31D_minus_R31C": float(delta.mean()),
            "std_delta": float(delta.std(ddof=1)),
            "min_delta": float(delta.min()),
            "max_delta": float(delta.max()),
        })

    return pd.DataFrame(rows)


def prevented_harm_retained_benefit(
    decisions: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for seed, s in decisions.groupby("split_seed", sort=True):
        prop = s[s["controller"] == PRIMARY_CONTROLLER][
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

        for baseline in [
            "ALWAYS_TENT1",
            "ALWAYS_PL_CONF90",
            "ALWAYS_MEMO",
        ]:
            b = s[s["controller"] == baseline][
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
            x = prop.merge(
                b,
                on=["sample_id", "model_state_id"],
                validate="one_to_one",
            )

            bh = int(x["base_harm"].sum())
            ph = int(
                ((x["base_harm"] == 1) & (x["prop_harm"] == 0)).sum()
            )
            bb = int(x["base_benefit"].sum())
            rb = int(
                ((x["base_benefit"] == 1) & (x["prop_benefit"] == 1)).sum()
            )

            rows.append({
                "split_seed": int(seed),
                "baseline": baseline,
                "baseline_harm_count": bh,
                "prevented_harm_count": ph,
                "prevented_harm_fraction": (
                    float(ph / bh) if bh > 0 else math.nan
                ),
                "baseline_benefit_count": bb,
                "retained_benefit_count": rb,
                "retained_benefit_fraction": (
                    float(rb / bb) if bb > 0 else math.nan
                ),
            })

    return pd.DataFrame(rows)


def feasibility(summary: pd.DataFrame) -> Dict[str, Any]:
    g = summary[summary["controller"] == PRIMARY_CONTROLLER]
    if len(g) != 1:
        raise RuntimeError("Missing unique R31D summary row.")

    r = g.iloc[0]
    harm = float(r["harm_rate_mean"])
    delta = float(r["mean_deployed_delta_dice_mean"])
    coverage = float(r["adaptation_coverage_mean"])

    p_harm = harm <= FEASIBILITY_HARM_MAX
    p_delta = delta > FEASIBILITY_MEAN_DELTA_MIN_EXCLUSIVE
    p_cov = coverage >= FEASIBILITY_COVERAGE_MIN

    if p_harm and p_delta and p_cov:
        decision = "PASS_R31D_POLICY_LEVEL_CONTROLLER_FEASIBILITY"
        next_stage = (
            "R31D2_FREEZE_FINAL_POLICY_AND_PLAN_NEW_UNTOUCHED_EXTERNAL_CONFIRMATION"
        )
    else:
        decision = "NO_PASS_R31D_POLICY_LEVEL_CONTROLLER_FEASIBILITY"
        next_stage = (
            "STOP_MULTI_ACTION_CONTROLLER_ENHANCEMENT_AND_RETAIN_R31B_ACTION_TRANSFER_CLAIM"
        )

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
            "harm": bool(p_harm),
            "mean_delta": bool(p_delta),
            "coverage": bool(p_cov),
        },
        "next": next_stage,
    }


def threshold_summary(thresholds: pd.DataFrame) -> pd.DataFrame:
    finite = thresholds[np.isfinite(thresholds["tau"].to_numpy(float))]
    return pd.DataFrame([{
        "partitions": int(len(thresholds)),
        "available_partitions": int(thresholds["available"].sum()),
        "availability_fraction": float(thresholds["available"].mean()),
        "tau_mean_finite": (
            float(finite["tau"].mean()) if len(finite) else math.nan
        ),
        "tau_min_finite": (
            float(finite["tau"].min()) if len(finite) else math.nan
        ),
        "tau_max_finite": (
            float(finite["tau"].max()) if len(finite) else math.nan
        ),
        "positive_utility_candidates_mean": float(
            thresholds["positive_utility_candidates"].mean()
        ),
        "candidate_fraction_mean": float(
            thresholds["candidate_fraction"].mean()
        ),
        "accepted_n_mean": float(thresholds["accepted_n"].mean()),
        "corrected_risk_mean_available": (
            float(
                thresholds.loc[
                    thresholds["available"].astype(bool),
                    "corrected_risk",
                ].mean()
            )
            if bool(thresholds["available"].any())
            else math.nan
        ),
    }])


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
    print("SafeTTA R31D1 Nested Utility-First Policy-Level CRC OOF")
    print("Version                  :", VERSION)
    print("Scientific status        : POST-R31C DEVELOPMENTAL")
    print("Nested partitions        : 5 seeds x 5 folds")
    print("Train / Cal / Test       : 480 / 160 / 160 physical cases")
    print("Policy HARM budget       :", ALPHA_POLICY)
    print("Per-action budget split  : NO")
    print("Hyperparameter search    : NO")
    print("EndoTect / external      : NO / NO")
    print("=" * 124)

    upstream = verify_upstream()
    r31c1 = load_r31c1_helpers()

    qnames, dnames = r31c1.load_feature_schema()
    panel, lineage = r31c1.load_three_action_panel(qnames, dnames)

    partitions = pd.read_csv(
        R31C0_PARTITIONS,
        dtype={"sample_id": str, "role": str},
    )
    validate_partitions(partitions)

    action_cats, family_action_cats = r31c1.global_category_schema(panel)

    out = args.output_dir.resolve()
    if out.exists():
        raise RuntimeError(
            f"Output exists; do not overwrite R31D1 result: {out}"
        )
    out.mkdir(parents=True, exist_ok=False)

    input_lock = {
        "status": "PASS_R31D1_INPUT_LOCK",
        "version": VERSION,
        "scientific_status": "POST_R31C_DEVELOPMENTAL",
        "R31D0_script_sha256": EXPECTED_R31D0_SCRIPT_SHA256,
        "R31D0_protocol_sha256": EXPECTED_R31D0_PROTOCOL_SHA256,
        "R31D0_final_sha256": EXPECTED_R31D0_FINAL_SHA256,
        "R31C1_script_sha256": EXPECTED_R31C1_SCRIPT_SHA256,
        "R31C1_final_sha256": EXPECTED_R31C1_FINAL_SHA256,
        "partition_sha256": upstream["partition_sha256"],
        "historical_R31C_decisions_sha256": (
            upstream["historical_r31c_decisions_sha256"]
        ),
        "lineage": lineage,
        "physical_cases": int(panel["sample_id"].nunique()),
        "model_states": int(panel["model_state_id"].nunique()),
        "actions": panel["action"].value_counts().to_dict(),
        "alpha_policy": ALPHA_POLICY,
        "external_data_access": False,
    }
    input_lock_path = out / "R31D1_INPUT_LOCK.json"
    atomic_json(input_lock_path, input_lock)

    all_new_decisions = []
    threshold_rows = []
    audit_rows = []

    pbar = tqdm(
        total=len(SPLIT_SEEDS) * N_FOLDS,
        desc="R31D1 nested policy-CRC partitions",
        unit="partition",
        dynamic_ncols=True,
    )

    for seed in SPLIT_SEEDS:
        sp = partitions[partitions["split_seed"].astype(int) == seed]

        for fold in range(N_FOLDS):
            pp = sp[sp["outer_fold"].astype(int) == fold]

            train_ids = set(
                pp.loc[pp["role"] == "TRAIN", "sample_id"].astype(str)
            )
            cal_ids = set(
                pp.loc[
                    pp["role"] == "CALIBRATION",
                    "sample_id",
                ].astype(str)
            )
            test_ids = set(
                pp.loc[pp["role"] == "TEST", "sample_id"].astype(str)
            )

            if len(train_ids) != TRAIN_CASES:
                raise RuntimeError("TRAIN case count drift.")
            if len(cal_ids) != CAL_CASES:
                raise RuntimeError("CAL case count drift.")
            if len(test_ids) != TEST_CASES:
                raise RuntimeError("TEST case count drift.")
            if train_ids & cal_ids or train_ids & test_ids or cal_ids & test_ids:
                raise RuntimeError("Nested role overlap.")

            train = panel[panel["sample_id"].isin(train_ids)].copy()
            cal = panel[panel["sample_id"].isin(cal_ids)].copy()
            test = panel[panel["sample_id"].isin(test_ids)].copy()

            utility, harm = r31c1.fit_models(
                train,
                qnames,
                dnames,
                action_cats,
                family_action_cats,
            )

            cal_pred = r31c1.predict_models(
                cal,
                utility,
                harm,
                qnames,
                dnames,
                action_cats,
                family_action_cats,
            )
            test_pred = r31c1.predict_models(
                test,
                utility,
                harm,
                qnames,
                dnames,
                action_cats,
                family_action_cats,
            )

            cal_candidates = build_candidates(cal_pred)
            crc = select_policy_tau(cal_candidates)
            tau = float(crc["tau"])

            test_decisions = evaluate_test(
                test_pred,
                tau,
                seed,
                fold,
            )
            all_new_decisions.append(test_decisions)

            threshold_rows.append({
                "split_seed": int(seed),
                "outer_fold": int(fold),
                "alpha_policy": ALPHA_POLICY,
                **crc,
            })

            audit_rows.append({
                "split_seed": int(seed),
                "outer_fold": int(fold),
                "train_cases": len(train_ids),
                "calibration_cases": len(cal_ids),
                "test_cases": len(test_ids),
                "cal_candidate_modelcases": len(cal_candidates),
                "cal_positive_utility_candidates": int(
                    cal_candidates["candidate_exists"].sum()
                ),
                "policy_tau": tau,
                "policy_tau_available": bool(crc["available"]),
            })

            pbar.set_postfix(
                seed=seed,
                fold=fold,
                tau=f"{tau:.3g}",
                cand=int(crc["positive_utility_candidates"]),
                acc=int(crc["accepted_n"]),
            )
            pbar.update(1)

    pbar.close()

    new_decisions = pd.concat(all_new_decisions, ignore_index=True)

    expected_new_rows = (
        len(SPLIT_SEEDS)
        * EXPECTED_CASES
        * EXPECTED_STATES
        * len(BASELINES)
    )
    if len(new_decisions) != expected_new_rows:
        raise RuntimeError(
            f"New decision rows={len(new_decisions)} "
            f"expected={expected_new_rows}"
        )

    # Every controller must be a complete 800x9 OOF panel within each seed.
    for seed in SPLIT_SEEDS:
        s = new_decisions[new_decisions["split_seed"] == seed]
        for controller in BASELINES:
            g = s[s["controller"] == controller]
            expected = EXPECTED_CASES * EXPECTED_STATES
            if len(g) != expected:
                raise RuntimeError(
                    f"seed={seed} controller={controller} rows={len(g)}"
                )
            if g.duplicated(["sample_id", "model_state_id"]).any():
                raise RuntimeError(
                    f"Duplicate OOF unit seed={seed} controller={controller}"
                )

    historical = load_historical_r31c()

    # Keep common decision fields for joint summary.
    common = [
        "split_seed",
        "outer_fold",
        "sample_id",
        "model_state_id",
        "model_family",
        "controller",
        "chosen_action",
        "deployed_delta_dice",
        "deployed_harm",
        "deployed_benefit",
        "adapted",
    ]
    all_decisions = pd.concat(
        [
            new_decisions[common],
            historical[common],
        ],
        ignore_index=True,
    )

    seed_metrics = per_seed_metrics(all_decisions)
    summary = summarize(seed_metrics)
    comparison = compare_r31d_vs_r31c(seed_metrics)
    secondary = prevented_harm_retained_benefit(new_decisions)

    thresholds = pd.DataFrame(threshold_rows)
    tau_summary = threshold_summary(thresholds)
    audit = pd.DataFrame(audit_rows)
    decision = feasibility(summary)

    # Outputs.
    new_decisions_path = out / "R31D1_OOF_NEW_CONTROLLER_DECISIONS.csv"
    all_decisions_path = out / "R31D1_OOF_WITH_HISTORICAL_R31C.csv"
    thresholds_path = out / "R31D1_POLICY_CRC_THRESHOLDS.csv"
    tau_summary_path = out / "R31D1_POLICY_CRC_THRESHOLD_SUMMARY.csv"
    audit_path = out / "R31D1_NESTED_FOLD_AUDIT.csv"
    seed_metrics_path = out / "R31D1_PER_SEED_CONTROLLER_METRICS.csv"
    summary_path = out / "R31D1_CONTROLLER_SUMMARY.csv"
    comparison_path = out / "R31D1_R31D_VS_R31C_COMPARISON.csv"
    secondary_path = out / "R31D1_PREVENTED_HARM_RETAINED_BENEFIT.csv"
    decision_path = out / "R31D1_FEASIBILITY_DECISION.json"

    atomic_csv(new_decisions, new_decisions_path)
    atomic_csv(all_decisions, all_decisions_path)
    atomic_csv(thresholds, thresholds_path)
    atomic_csv(tau_summary, tau_summary_path)
    atomic_csv(audit, audit_path)
    atomic_csv(seed_metrics, seed_metrics_path)
    atomic_csv(summary, summary_path)
    atomic_csv(comparison, comparison_path)
    atomic_csv(secondary, secondary_path)

    decision.update({
        "version": VERSION,
        "scientific_status": "POST_R31C_DEVELOPMENTAL",
        "R31D0_protocol_sha256": EXPECTED_R31D0_PROTOCOL_SHA256,
        "R31D0_final_sha256": EXPECTED_R31D0_FINAL_SHA256,
        "R31C1_final_sha256": EXPECTED_R31C1_FINAL_SHA256,
        "alpha_policy": ALPHA_POLICY,
        "complete_OOF_seed_panels": len(SPLIT_SEEDS),
        "nested_partitions": len(SPLIT_SEEDS) * N_FOLDS,
        "prospective_confirmation_claim_allowed": False,
        "new_untouched_external_required_if_successful": True,
        "external_data_access": False,
    })
    atomic_json(decision_path, decision)

    inventory = make_inventory(out)
    inventory_path = out / "R31D1_SHA256_INVENTORY.csv"
    atomic_csv(inventory, inventory_path)

    final = {
        "status": "PASS_R31D1_NESTED_POLICY_LEVEL_CRC_OOF_COMPLETE",
        "version": VERSION,
        "scientific_status": "POST_R31C_DEVELOPMENTAL",
        "decision": decision["decision"],
        "R31D0_protocol_sha256": EXPECTED_R31D0_PROTOCOL_SHA256,
        "R31D0_final_sha256": EXPECTED_R31D0_FINAL_SHA256,
        "R31C1_final_sha256": EXPECTED_R31C1_FINAL_SHA256,
        "alpha_policy": ALPHA_POLICY,
        "complete_OOF_seed_panels": len(SPLIT_SEEDS),
        "nested_partitions": len(SPLIT_SEEDS) * N_FOLDS,
        "controller_summary_sha256": sha256_file(summary_path),
        "policy_thresholds_sha256": sha256_file(thresholds_path),
        "new_OOF_decisions_sha256": sha256_file(new_decisions_path),
        "comparison_sha256": sha256_file(comparison_path),
        "feasibility_decision_sha256": sha256_file(decision_path),
        "hyperparameter_search": False,
        "external_data_access": False,
        "prospective_confirmation_claim_allowed": False,
        "next": decision["next"],
    }
    final_path = out / "R31D1_FINAL_LOCK.json"
    atomic_json(final_path, final)

    print("\n" + "=" * 124)
    print("R31D1 POLICY-LEVEL CRC THRESHOLD SUMMARY")
    print("=" * 124)
    print(tau_summary.to_string(index=False))

    print("\n" + "=" * 124)
    print("R31D1 CONTROLLER SUMMARY (MEAN +/- STD OVER 5 COMPLETE OOF SEEDS)")
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

    print("\nR31D1 VS R31C")
    print(comparison.to_string(index=False))

    print("\nR31D1 FEASIBILITY:", decision["decision"])
    print(
        "  R31D HARM rate              :",
        decision["observed"]["harm_rate"],
        "<=",
        FEASIBILITY_HARM_MAX,
        ":",
        decision["passes"]["harm"],
    )
    print(
        "  R31D mean deployed DeltaDice:",
        decision["observed"]["mean_deployed_delta_dice"],
        "> 0 :",
        decision["passes"]["mean_delta"],
    )
    print(
        "  R31D adaptation coverage    :",
        decision["observed"]["adaptation_coverage"],
        ">=",
        FEASIBILITY_COVERAGE_MIN,
        ":",
        decision["passes"]["coverage"],
    )

    print("\nFINAL STATUS : PASS_R31D1_NESTED_POLICY_LEVEL_CRC_OOF_COMPLETE")
    print("Scientific status             : POST-R31C DEVELOPMENTAL")
    print("Prospective confirmation claim: NO")
    print("Hyperparameter search         : NO")
    print("EndoTect / external           : NO / NO")
    print("Final lock SHA256             :", sha256_file(final_path))
    print("Output                        :", out)
    print("NEXT                          :", decision["next"])
    print("=" * 124)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
