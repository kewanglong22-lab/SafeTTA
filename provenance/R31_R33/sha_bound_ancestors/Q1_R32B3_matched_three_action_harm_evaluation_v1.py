#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SafeTTA R32B3
Matched Three-Action Published-Reliability Baseline HARM Evaluation

Primary scientific question
---------------------------
On the exact matched DeepLabV3-R50 panel, how well do frozen SOURCE
reliability scores rank FUTURE HARM under three mechanistically distinct TTA
actions, compared with the action-transferable SafeTTA transition predictor?

Matched panel
-------------
800 physical NeoPolyp cases
x 3 DeepLabV3-R50 states
x 3 future actions:
  TENT1
  PL-CONF90
  MEMO-SEG4-1STEP
= 7200 action-level outcomes.

Methods
-------
Published SOURCE-reliability baselines (one score/source model-case, copied
unchanged to all three future-action endpoints):
  SicTTA-CCD       -> ccd_risk
  TEGDA-ADIC       -> adic_harm_risk = -adic_quality
  MC-dropout       -> mc_predictive_entropy_risk

SafeTTA primary:
  SHARED_TRANSITION_Q66_DSEM64
  exact frozen R32A1 LOAO OOF predictions
  5 pre-specified split seeds.

No score direction reversal, target calibration, threshold tuning,
feature selection, subset selection, model refitting, or new inference.

Primary metrics
---------------
Per action:
  AUROC
  AUPRC
  AUPRC Lift = AUPRC / HARM prevalence

Macro:
  unweighted mean across the three future actions.

SafeTTA reporting:
  mean +/- std across five frozen split seeds.

Published baseline reporting:
  deterministic point metrics (the exact score is independent of split seed).

Paired inference
----------------
2000-replicate physical-case clustered bootstrap.

For every bootstrap replicate:
  1) sample 800 physical sample_id clusters with replacement;
  2) retain all 3 DeepLab states for each sampled case;
  3) use the SAME sampled cases for all three actions;
  4) for SafeTTA, compute the metric separately for all five split seeds,
     then average the five metric values;
  5) subtract the comparator metric computed on the same resample.

Thus significance targets:
  mean-five-seed SafeTTA metric - frozen baseline metric,
without averaging/ensembling the five SafeTTA scores themselves.

This is a POST-R31B3 retrospective matched evaluation. R32B2B froze all
baseline scores before this script reads the HARM outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.metrics import average_precision_score, roc_auc_score


VERSION = "2026-09-12-R32B3-v1"

ROOT = Path(r"F:\MEDSEG_SAFETTA")
CODE = ROOT / "code"

# ---------------------------------------------------------------------
# Exact R32B2B frozen baseline-score bindings
# ---------------------------------------------------------------------

R32B2B_SCRIPT = CODE / "Q1_R32B2B_exact_ccd_adic_mc_score_generation_v1_fix1.py"
EXPECTED_R32B2B_SCRIPT_SHA256 = (
    "c0877d3bcfa46317fd5ab944ca905e0327b09bb3deaee38324bea100606d3383"
)

R32B2B_DIR = ROOT / "R32B2B_exact_ccd_adic_mc_score_generation_v1_fix1"
R32B2B_FINAL = R32B2B_DIR / "R32B2B_FINAL_LOCK.json"
EXPECTED_R32B2B_FINAL_SHA256 = (
    "6b24c34b39f5f270f027521040b5953057170960e9215f7406ece4436a23d112"
)
R32B2B_SCORE_LOCK = R32B2B_DIR / "R32B2B_SCORE_LOCK.json"
EXPECTED_R32B2B_SCORE_LOCK_SHA256 = (
    "2ece10745f483262b39f391d8942424550e5f1ca03068ada7ff4a5df8ee83d55"
)
R32B2B_COMMON = (
    R32B2B_DIR
    / "R32B2B_NEOPOLYP_COMMON800_DEEPLAB_SOURCE_RELIABILITY_SCORE_LOCK.csv"
)
EXPECTED_R32B2B_COMMON_SHA256 = (
    "81f66fe67025793c99c54325705889f209f975672e923681a8a547ebb50f169a"
)
R32B2B_ALL1000 = (
    R32B2B_DIR
    / "R32B2B_NEOPOLYP_ALL1000_DEEPLAB_SOURCE_RELIABILITY_SCORE_LOCK.csv"
)
EXPECTED_R32B2B_ALL1000_SHA256 = (
    "6be9ca204ca0b34bb45cad379f9092210b8f88080d044a0728d753c025ba6640"
)

# ---------------------------------------------------------------------
# Exact common-panel + SafeTTA OOF bindings
# ---------------------------------------------------------------------

R32B1_DIR = ROOT / "R32B1_faithful_common_baseline_panel_lock_v1"
R32B1_MANIFEST = R32B1_DIR / "R32B1_DEEPLAB3_COMMON_PANEL_MANIFEST.csv"
EXPECTED_R32B1_MANIFEST_SHA256 = (
    "2c7745dc145d6bcea33f3cd37e9c3cd0fbbf34d6c5264c69b64eb8fb48942d24"
)
R32B1_FINAL = R32B1_DIR / "R32B1_FINAL_LOCK.json"
EXPECTED_R32B1_FINAL_SHA256 = (
    "ce305d32f77b2c72e5ca6589043aa6644542e35ee74fa7f41e68db45d4cd31e4"
)

R32A1_DIR = ROOT / "R32A1_three_action_loao_comparison_execution_v1_fix1"
R32A1_FINAL = R32A1_DIR / "R32A1_FIX1_FINAL_LOCK.json"
EXPECTED_R32A1_FINAL_SHA256 = (
    "0822cf8c6db035e836dcc6d3f67ae4b0e9d58622e72ba1086edf60fa90e8873b"
)
R32A1_OOF = R32A1_DIR / "R32A1_FIX1_OOF_PREDICTIONS.csv"

DEFAULT_OUT = ROOT / "R32B3_matched_three_action_harm_evaluation_v1"

# ---------------------------------------------------------------------
# Frozen evaluation protocol
# ---------------------------------------------------------------------

FAMILY = "DeepLabV3-R50"
ACTIONS = ("MEMO-SEG4-1STEP", "PL-CONF90", "TENT1")
N_CASES = 800
N_STATES = 3
EXPECTED_ACTION_ROWS = N_CASES * N_STATES * len(ACTIONS)
EXPECTED_SOURCE_ROWS = N_CASES * N_STATES

SAFE_MODEL = "SHARED_TRANSITION_Q66_DSEM64"
SAFE_SPLIT_SEEDS = (20260912, 20260913, 20260914, 20260915, 20260916)

BASELINES = {
    "SicTTA-CCD": "ccd_risk",
    "TEGDA-ADIC": "adic_harm_risk",
    "MC-dropout": "mc_predictive_entropy_risk",
}

BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 20260912

PRIMARY_METRICS = ("AUROC", "AUPRC")
ALL_METRICS = ("AUROC", "AUPRC", "AUPRC_LIFT")

# R32B1 already froze the direction. Do not infer/reverse from outcomes.
ORIENTATION = {
    "SicTTA-CCD": "higher = higher HARM risk",
    "TEGDA-ADIC": "higher adic_harm_risk (-quality) = higher HARM risk",
    "MC-dropout": "higher predictive entropy = higher HARM risk",
    "SafeTTA": "higher predicted score = higher HARM risk",
}


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


def metric_value(
    y: np.ndarray,
    score: np.ndarray,
    metric: str,
) -> float:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)

    if len(np.unique(y)) != 2:
        return math.nan

    if metric == "AUROC":
        return float(roc_auc_score(y, score))
    if metric == "AUPRC":
        return float(average_precision_score(y, score))
    raise ValueError(metric)


def action_metrics(
    y: np.ndarray,
    score: np.ndarray,
) -> Dict[str, float]:
    y = np.asarray(y, dtype=int)
    prevalence = float(np.mean(y))
    auroc = metric_value(y, score, "AUROC")
    auprc = metric_value(y, score, "AUPRC")
    lift = (
        float(auprc / prevalence)
        if prevalence > 0 and math.isfinite(auprc)
        else math.nan
    )
    return {
        "AUROC": auroc,
        "AUPRC": auprc,
        "AUPRC_LIFT": lift,
        "prevalence": prevalence,
        "n": int(len(y)),
    }


# ---------------------------------------------------------------------
# Upstream immutable bindings
# ---------------------------------------------------------------------

def verify_upstream() -> Dict[str, Any]:
    require_sha(R32B2B_SCRIPT, EXPECTED_R32B2B_SCRIPT_SHA256, "R32B2B fix1 script")
    require_sha(R32B2B_FINAL, EXPECTED_R32B2B_FINAL_SHA256, "R32B2B final")
    require_sha(
        R32B2B_SCORE_LOCK,
        EXPECTED_R32B2B_SCORE_LOCK_SHA256,
        "R32B2B score lock",
    )
    require_sha(R32B2B_COMMON, EXPECTED_R32B2B_COMMON_SHA256, "R32B2B common scores")
    require_sha(R32B2B_ALL1000, EXPECTED_R32B2B_ALL1000_SHA256, "R32B2B all1000 scores")

    require_sha(R32B1_MANIFEST, EXPECTED_R32B1_MANIFEST_SHA256, "R32B1 common manifest")
    require_sha(R32B1_FINAL, EXPECTED_R32B1_FINAL_SHA256, "R32B1 final")
    require_sha(R32A1_FINAL, EXPECTED_R32A1_FINAL_SHA256, "R32A1 final")

    b2 = json.loads(R32B2B_FINAL.read_text(encoding="utf-8"))
    if b2.get("status") != (
        "PASS_R32B2B_EXACT_CCD_ADIC_MC_SCORE_GENERATION_COMPLETE"
    ):
        raise RuntimeError("R32B2B final status changed.")
    if not bool(b2.get("R31B3_HARM_GT_read") is False):
        raise RuntimeError("R32B2B no-HARM boundary changed.")
    if bool(b2.get("target_calibration", True)):
        raise RuntimeError("R32B2B target-calibration guard changed.")
    if bool(b2.get("post_hoc_score_reversal", True)):
        raise RuntimeError("R32B2B score-reversal guard changed.")
    if bool(b2.get("external_data_access", True)):
        raise RuntimeError("R32B2B external-access guard changed.")

    a1 = json.loads(R32A1_FINAL.read_text(encoding="utf-8"))
    if a1.get("status") != (
        "PASS_R32A1_FIX1_THREE_ACTION_LOAO_COMPARISON_COMPLETE"
    ):
        raise RuntimeError("R32A1 final status changed.")
    if a1.get("primary_model") != SAFE_MODEL:
        raise RuntimeError(
            f"R32A1 primary model changed: {a1.get('primary_model')}"
        )
    if not R32A1_OOF.is_file():
        raise FileNotFoundError(R32A1_OOF)
    if sha256_file(R32A1_OOF) != str(a1.get("OOF_predictions_sha256")):
        raise RuntimeError("R32A1 OOF SHA changed.")

    return {
        "R32B2B": b2,
        "R32A1": a1,
    }


# ---------------------------------------------------------------------
# Pre-outcome analysis lock
# ---------------------------------------------------------------------

def write_analysis_lock(out: Path) -> Path:
    """
    This is written before R32B1 HARM outcomes are loaded in main().
    It freezes all reporting / bootstrap choices.
    """
    payload = {
        "status": "LOCKED_R32B3_MATCHED_EVALUATION_PROTOCOL",
        "version": VERSION,
        "scientific_status": "POST_R31B3_RETROSPECTIVE_MATCHED_EVALUATION",
        "matched_family": FAMILY,
        "physical_cases": N_CASES,
        "model_states": N_STATES,
        "actions": list(ACTIONS),
        "SafeTTA_model": SAFE_MODEL,
        "SafeTTA_split_seeds": list(SAFE_SPLIT_SEEDS),
        "published_baselines": BASELINES,
        "orientations": ORIENTATION,
        "primary_metrics": list(PRIMARY_METRICS),
        "secondary_metric": "AUPRC_LIFT=AUPRC/HARM_PREVALENCE",
        "macro_definition": "unweighted mean of the three action metrics",
        "SafeTTA_main_reporting": (
            "metric mean +/- sample std across five frozen split seeds"
        ),
        "baseline_main_reporting": (
            "single deterministic point metric from frozen SOURCE score"
        ),
        "bootstrap": {
            "reps": BOOTSTRAP_REPS,
            "seed": BOOTSTRAP_SEED,
            "cluster": "physical sample_id",
            "same_case_resample_across_actions": True,
            "all_3_model_states_retained_per_sampled_case": True,
            "SafeTTA_delta_definition": (
                "within each bootstrap replicate compute metric for each of "
                "five split seeds, average those five metric values, then "
                "subtract comparator metric on the same resampled cases"
            ),
            "score_ensemble_used": False,
            "CI": "percentile 2.5% / 97.5%",
        },
        "prohibitions": [
            "no score direction reversal",
            "no target score calibration",
            "no threshold tuning",
            "no feature selection",
            "no favorable subset selection",
            "no model refitting",
            "no score ensembling across SafeTTA split seeds",
            "no new baseline or TTA inference",
        ],
        "R32B2B_common_score_sha256": EXPECTED_R32B2B_COMMON_SHA256,
        "R32B2B_score_lock_sha256": EXPECTED_R32B2B_SCORE_LOCK_SHA256,
        "R32B1_common_manifest_sha256": EXPECTED_R32B1_MANIFEST_SHA256,
        "R32A1_final_sha256": EXPECTED_R32A1_FINAL_SHA256,
    }

    path = out / "R32B3_PRE_OUTCOME_ANALYSIS_LOCK.json"
    atomic_json(path, payload)
    return path


# ---------------------------------------------------------------------
# Exact matched panel construction
# ---------------------------------------------------------------------

def load_baseline_scores() -> pd.DataFrame:
    d = pd.read_csv(
        R32B2B_COMMON,
        dtype={
            "sample_id": str,
            "model_state_id": str,
            "model_family": str,
        },
        low_memory=False,
    )

    required = {
        "sample_id",
        "model_state_id",
        "model_family",
        *BASELINES.values(),
    }
    missing = sorted(required.difference(d.columns))
    if missing:
        raise RuntimeError(f"R32B2B common score missing={missing}")

    if len(d) != EXPECTED_SOURCE_ROWS:
        raise RuntimeError(
            f"Baseline SOURCE rows={len(d)} expected={EXPECTED_SOURCE_ROWS}"
        )
    if d["sample_id"].nunique() != N_CASES:
        raise RuntimeError("Baseline physical-case count drift.")
    if d["model_state_id"].nunique() != N_STATES:
        raise RuntimeError("Baseline state count drift.")
    if set(d["model_family"].astype(str)) != {FAMILY}:
        raise RuntimeError("Baseline family drift.")
    if d.duplicated(["sample_id", "model_state_id"]).any():
        raise RuntimeError("Duplicate baseline SOURCE score row.")

    for c in BASELINES.values():
        if not np.isfinite(d[c].to_numpy(float)).all():
            raise RuntimeError(f"Non-finite baseline score column={c}")

    # Internal ADIC orientation identity, independent of outcomes.
    if "adic_quality" not in d.columns:
        raise RuntimeError("adic_quality missing from frozen baseline score.")
    if not np.array_equal(
        d["adic_harm_risk"].to_numpy(float),
        -d["adic_quality"].to_numpy(float),
    ):
        raise RuntimeError("Frozen ADIC risk orientation identity changed.")

    return d


def load_outcome_panel() -> pd.DataFrame:
    """
    FIRST outcome read in this script. Called only after analysis lock exists.
    """
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

    required = {
        "sample_id",
        "model_state_id",
        "model_family",
        "action",
        "r31b3_harm_label",
    }
    missing = sorted(required.difference(d.columns))
    if missing:
        raise RuntimeError(f"R32B1 common outcome manifest missing={missing}")

    if len(d) != EXPECTED_ACTION_ROWS:
        raise RuntimeError(
            f"Outcome rows={len(d)} expected={EXPECTED_ACTION_ROWS}"
        )
    if d["sample_id"].nunique() != N_CASES:
        raise RuntimeError("Outcome case count drift.")
    if d["model_state_id"].nunique() != N_STATES:
        raise RuntimeError("Outcome state count drift.")
    if set(d["action"].astype(str)) != set(ACTIONS):
        raise RuntimeError("Outcome action set drift.")
    if set(d["model_family"].astype(str)) != {FAMILY}:
        raise RuntimeError("Outcome family drift.")
    if d.duplicated(["sample_id", "model_state_id", "action"]).any():
        raise RuntimeError("Duplicate outcome unit.")

    y = d["r31b3_harm_label"].to_numpy(int)
    if not set(np.unique(y)).issubset({0, 1}):
        raise RuntimeError("HARM label is not binary.")

    return d


def load_safettta_predictions(
    outcome: pd.DataFrame,
) -> pd.DataFrame:
    usecols = [
        "split_seed",
        "model",
        "evaluation_type",
        "sample_id",
        "model_state_id",
        "model_family",
        "action",
        "y_harm",
        "score",
    ]
    d = pd.read_csv(
        R32A1_OOF,
        usecols=usecols,
        dtype={
            "sample_id": str,
            "model_state_id": str,
            "model_family": str,
            "action": str,
        },
        low_memory=False,
    )

    d = d[
        (d["model"] == SAFE_MODEL)
        & (d["evaluation_type"] == "LOAO")
        & (d["model_family"] == FAMILY)
    ].copy()

    expected = len(SAFE_SPLIT_SEEDS) * EXPECTED_ACTION_ROWS
    if len(d) != expected:
        raise RuntimeError(
            f"SafeTTA matched OOF rows={len(d)} expected={expected}"
        )
    if set(pd.to_numeric(d["split_seed"]).astype(int)) != set(SAFE_SPLIT_SEEDS):
        raise RuntimeError("SafeTTA split-seed set drift.")
    if d.duplicated(
        ["split_seed", "sample_id", "model_state_id", "action"]
    ).any():
        raise RuntimeError("Duplicate SafeTTA matched OOF unit.")
    if not np.isfinite(d["score"].to_numpy(float)).all():
        raise RuntimeError("Non-finite SafeTTA OOF score.")

    truth = outcome[
        [
            "sample_id",
            "model_state_id",
            "action",
            "r31b3_harm_label",
        ]
    ].rename(columns={"r31b3_harm_label": "truth_harm"})

    chk = d.merge(
        truth,
        on=["sample_id", "model_state_id", "action"],
        how="left",
        validate="many_to_one",
    )
    mismatch = int(
        np.count_nonzero(
            chk["y_harm"].to_numpy(int)
            != chk["truth_harm"].to_numpy(int)
        )
    )
    if mismatch:
        raise RuntimeError(
            f"SafeTTA OOF / R32B1 HARM mismatch rows={mismatch}"
        )

    return d


def build_wide_eval_panel(
    baseline: pd.DataFrame,
    outcome: pd.DataFrame,
    safe: pd.DataFrame,
) -> pd.DataFrame:
    base_cols = [
        "sample_id",
        "model_state_id",
        "model_family",
        *BASELINES.values(),
    ]

    panel = outcome.merge(
        baseline[base_cols],
        on=["sample_id", "model_state_id", "model_family"],
        how="left",
        validate="many_to_one",
    )

    if panel[list(BASELINES.values())].isna().any().any():
        raise RuntimeError("Missing baseline score after action-panel join.")

    for seed in SAFE_SPLIT_SEEDS:
        g = safe[pd.to_numeric(safe["split_seed"]).astype(int) == seed][
            ["sample_id", "model_state_id", "action", "score"]
        ].rename(columns={"score": f"safettta_{seed}"})

        panel = panel.merge(
            g,
            on=["sample_id", "model_state_id", "action"],
            how="left",
            validate="one_to_one",
        )

    safe_cols = [f"safettta_{s}" for s in SAFE_SPLIT_SEEDS]
    if panel[safe_cols].isna().any().any():
        raise RuntimeError("Missing SafeTTA score after exact matched join.")

    panel = panel.sort_values(
        ["sample_id", "model_state_id", "action"],
        kind="mergesort",
    ).reset_index(drop=True)

    if len(panel) != EXPECTED_ACTION_ROWS:
        raise RuntimeError("Wide matched panel row count changed.")

    return panel


# ---------------------------------------------------------------------
# Main metrics
# ---------------------------------------------------------------------

def baseline_action_metrics(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for method, col in BASELINES.items():
        for action in ACTIONS:
            g = panel[panel["action"] == action]
            m = action_metrics(
                g["r31b3_harm_label"].to_numpy(int),
                g[col].to_numpy(float),
            )
            rows.append({
                "method": method,
                "action": action,
                "reporting": "deterministic_frozen_point",
                "metric_seed_count": 1,
                "auroc_mean": m["AUROC"],
                "auroc_std": 0.0,
                "auprc_mean": m["AUPRC"],
                "auprc_std": 0.0,
                "auprc_lift_mean": m["AUPRC_LIFT"],
                "auprc_lift_std": 0.0,
                "harm_prevalence": m["prevalence"],
                "rows": m["n"],
            })

    return pd.DataFrame(rows)


def safettta_action_metrics(panel: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    per_seed_rows = []

    for seed in SAFE_SPLIT_SEEDS:
        col = f"safettta_{seed}"
        for action in ACTIONS:
            g = panel[panel["action"] == action]
            m = action_metrics(
                g["r31b3_harm_label"].to_numpy(int),
                g[col].to_numpy(float),
            )
            per_seed_rows.append({
                "split_seed": seed,
                "method": "SafeTTA-Q66+dSemantic64",
                "action": action,
                "AUROC": m["AUROC"],
                "AUPRC": m["AUPRC"],
                "AUPRC_LIFT": m["AUPRC_LIFT"],
                "harm_prevalence": m["prevalence"],
                "rows": m["n"],
            })

    per_seed = pd.DataFrame(per_seed_rows)

    rows = []
    for action, g in per_seed.groupby("action", sort=True):
        rows.append({
            "method": "SafeTTA-Q66+dSemantic64",
            "action": action,
            "reporting": "five_split_seed_mean_std",
            "metric_seed_count": len(SAFE_SPLIT_SEEDS),
            "auroc_mean": float(g["AUROC"].mean()),
            "auroc_std": float(g["AUROC"].std(ddof=1)),
            "auprc_mean": float(g["AUPRC"].mean()),
            "auprc_std": float(g["AUPRC"].std(ddof=1)),
            "auprc_lift_mean": float(g["AUPRC_LIFT"].mean()),
            "auprc_lift_std": float(g["AUPRC_LIFT"].std(ddof=1)),
            "harm_prevalence": float(g["harm_prevalence"].iloc[0]),
            "rows": int(g["rows"].iloc[0]),
        })

    return per_seed, pd.DataFrame(rows)


def macro_summary(
    action_table: pd.DataFrame,
    safe_per_seed: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    safe_macro_seed_rows = []

    for method in BASELINES:
        g = action_table[
            (action_table["method"] == method)
        ]
        if len(g) != len(ACTIONS):
            raise RuntimeError(f"Baseline macro rows incomplete for {method}")
        rows.append({
            "method": method,
            "reporting": "deterministic_frozen_point",
            "metric_seed_count": 1,
            "macro_auroc_mean": float(g["auroc_mean"].mean()),
            "macro_auroc_std": 0.0,
            "macro_auprc_mean": float(g["auprc_mean"].mean()),
            "macro_auprc_std": 0.0,
            "macro_auprc_lift_mean": float(g["auprc_lift_mean"].mean()),
            "macro_auprc_lift_std": 0.0,
        })

    for seed, g in safe_per_seed.groupby("split_seed", sort=True):
        if set(g["action"]) != set(ACTIONS):
            raise RuntimeError(f"SafeTTA macro action set incomplete seed={seed}")
        safe_macro_seed_rows.append({
            "split_seed": int(seed),
            "method": "SafeTTA-Q66+dSemantic64",
            "macro_AUROC": float(g["AUROC"].mean()),
            "macro_AUPRC": float(g["AUPRC"].mean()),
            "macro_AUPRC_LIFT": float(g["AUPRC_LIFT"].mean()),
        })

    safe_macro_seed = pd.DataFrame(safe_macro_seed_rows)
    rows.append({
        "method": "SafeTTA-Q66+dSemantic64",
        "reporting": "five_split_seed_mean_std",
        "metric_seed_count": len(SAFE_SPLIT_SEEDS),
        "macro_auroc_mean": float(safe_macro_seed["macro_AUROC"].mean()),
        "macro_auroc_std": float(safe_macro_seed["macro_AUROC"].std(ddof=1)),
        "macro_auprc_mean": float(safe_macro_seed["macro_AUPRC"].mean()),
        "macro_auprc_std": float(safe_macro_seed["macro_AUPRC"].std(ddof=1)),
        "macro_auprc_lift_mean": float(
            safe_macro_seed["macro_AUPRC_LIFT"].mean()
        ),
        "macro_auprc_lift_std": float(
            safe_macro_seed["macro_AUPRC_LIFT"].std(ddof=1)
        ),
    })

    return safe_macro_seed, pd.DataFrame(rows)


def per_state_secondary(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for state_id, sd in panel.groupby("model_state_id", sort=True):
        for action in ACTIONS:
            g = sd[sd["action"] == action]

            for method, col in BASELINES.items():
                m = action_metrics(
                    g["r31b3_harm_label"].to_numpy(int),
                    g[col].to_numpy(float),
                )
                rows.append({
                    "model_state_id": state_id,
                    "action": action,
                    "method": method,
                    "AUROC": m["AUROC"],
                    "AUPRC": m["AUPRC"],
                    "AUPRC_LIFT": m["AUPRC_LIFT"],
                    "harm_prevalence": m["prevalence"],
                })

            vals = []
            for seed in SAFE_SPLIT_SEEDS:
                m = action_metrics(
                    g["r31b3_harm_label"].to_numpy(int),
                    g[f"safettta_{seed}"].to_numpy(float),
                )
                vals.append(m)

            rows.append({
                "model_state_id": state_id,
                "action": action,
                "method": "SafeTTA-Q66+dSemantic64",
                "AUROC": float(np.mean([x["AUROC"] for x in vals])),
                "AUPRC": float(np.mean([x["AUPRC"] for x in vals])),
                "AUPRC_LIFT": float(np.mean([x["AUPRC_LIFT"] for x in vals])),
                "harm_prevalence": vals[0]["prevalence"],
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Physical-case clustered paired bootstrap
# ---------------------------------------------------------------------

def prepare_action_arrays(
    panel: pd.DataFrame,
) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
    case_ids = sorted(panel["sample_id"].astype(str).unique().tolist())
    if len(case_ids) != N_CASES:
        raise RuntimeError("Bootstrap case count drift.")

    action_data = {}

    for action in ACTIONS:
        g = panel[panel["action"] == action].copy()
        g = g.sort_values(
            ["sample_id", "model_state_id"],
            kind="mergesort",
        ).reset_index(drop=True)

        counts = g.groupby("sample_id").size()
        if not (counts == N_STATES).all():
            raise RuntimeError(
                f"Action={action}: expected exactly {N_STATES} rows/case."
            )

        group_idx = {
            sid: np.flatnonzero(
                g["sample_id"].astype(str).to_numpy() == sid
            )
            for sid in case_ids
        }

        action_data[action] = {
            "df": g,
            "group_idx": group_idx,
            "y": g["r31b3_harm_label"].to_numpy(int),
            "baseline": {
                method: g[col].to_numpy(float)
                for method, col in BASELINES.items()
            },
            "safe": {
                seed: g[f"safettta_{seed}"].to_numpy(float)
                for seed in SAFE_SPLIT_SEEDS
            },
        }

    return case_ids, action_data


def build_resampled_indices(
    sampled_ids: np.ndarray,
    group_idx: Dict[str, np.ndarray],
) -> np.ndarray:
    return np.concatenate([group_idx[str(sid)] for sid in sampled_ids])


def bootstrap_pairwise(
    panel: pd.DataFrame,
) -> pd.DataFrame:
    case_ids, action_data = prepare_action_arrays(panel)
    ids_arr = np.asarray(case_ids, dtype=object)
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    scopes = list(ACTIONS) + ["MACRO"]
    deltas: Dict[Tuple[str, str, str], List[float]] = {
        (method, scope, metric): []
        for method in BASELINES
        for scope in scopes
        for metric in PRIMARY_METRICS
    }

    for _ in tqdm(
        range(BOOTSTRAP_REPS),
        desc="R32B3 physical-case clustered bootstrap",
        unit="rep",
        dynamic_ncols=True,
    ):
        sampled = rng.choice(ids_arr, size=N_CASES, replace=True)

        # Cache all action metrics for this physical-case resample.
        base_metric: Dict[Tuple[str, str, str], float] = {}
        safe_metric: Dict[Tuple[int, str, str], float] = {}

        for action in ACTIONS:
            ad = action_data[action]
            idx = build_resampled_indices(sampled, ad["group_idx"])
            y = ad["y"][idx]

            for method in BASELINES:
                for metric in PRIMARY_METRICS:
                    base_metric[(method, action, metric)] = metric_value(
                        y,
                        ad["baseline"][method][idx],
                        metric,
                    )

            for seed in SAFE_SPLIT_SEEDS:
                for metric in PRIMARY_METRICS:
                    safe_metric[(seed, action, metric)] = metric_value(
                        y,
                        ad["safe"][seed][idx],
                        metric,
                    )

        for method in BASELINES:
            for metric in PRIMARY_METRICS:
                # Action-specific delta.
                for action in ACTIONS:
                    safe_mean = float(
                        np.nanmean([
                            safe_metric[(seed, action, metric)]
                            for seed in SAFE_SPLIT_SEEDS
                        ])
                    )
                    base_val = base_metric[(method, action, metric)]
                    deltas[(method, action, metric)].append(
                        safe_mean - base_val
                    )

                # Macro delta: macro is formed from action metrics, preserving
                # equal action weighting, then averaged over five SafeTTA seeds.
                safe_seed_macros = []
                for seed in SAFE_SPLIT_SEEDS:
                    safe_seed_macros.append(
                        float(np.nanmean([
                            safe_metric[(seed, action, metric)]
                            for action in ACTIONS
                        ]))
                    )
                safe_macro = float(np.nanmean(safe_seed_macros))
                base_macro = float(np.nanmean([
                    base_metric[(method, action, metric)]
                    for action in ACTIONS
                ]))
                deltas[(method, "MACRO", metric)].append(
                    safe_macro - base_macro
                )

    # Observed deltas using the same estimand.
    rows = []
    for method in BASELINES:
        for scope in scopes:
            for metric in PRIMARY_METRICS:
                if scope == "MACRO":
                    safe_obs = float(np.mean([
                        np.mean([
                            action_metrics(
                                action_data[a]["y"],
                                action_data[a]["safe"][seed],
                            )[metric]
                            for a in ACTIONS
                        ])
                        for seed in SAFE_SPLIT_SEEDS
                    ]))
                    base_obs = float(np.mean([
                        action_metrics(
                            action_data[a]["y"],
                            action_data[a]["baseline"][method],
                        )[metric]
                        for a in ACTIONS
                    ]))
                else:
                    safe_obs = float(np.mean([
                        action_metrics(
                            action_data[scope]["y"],
                            action_data[scope]["safe"][seed],
                        )[metric]
                        for seed in SAFE_SPLIT_SEEDS
                    ]))
                    base_obs = action_metrics(
                        action_data[scope]["y"],
                        action_data[scope]["baseline"][method],
                    )[metric]

                vals = np.asarray(
                    deltas[(method, scope, metric)],
                    dtype=float,
                )
                vals = vals[np.isfinite(vals)]
                if len(vals) < int(0.95 * BOOTSTRAP_REPS):
                    raise RuntimeError(
                        f"Too many invalid bootstrap reps: "
                        f"{method}/{scope}/{metric} valid={len(vals)}"
                    )

                lo, hi = np.quantile(vals, [0.025, 0.975])
                observed = safe_obs - base_obs

                rows.append({
                    "comparator": method,
                    "scope": scope,
                    "metric": metric,
                    "SafeTTA_metric": safe_obs,
                    "comparator_metric": base_obs,
                    "delta_SafeTTA_minus_comparator": observed,
                    "bootstrap_reps": int(len(vals)),
                    "ci95_low": float(lo),
                    "ci95_high": float(hi),
                    "SafeTTA_significantly_better_95CI": bool(lo > 0),
                    "comparator_significantly_better_95CI": bool(hi < 0),
                })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Compact paper table
# ---------------------------------------------------------------------

def make_paper_table(
    action_table: pd.DataFrame,
    macro_table: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    method_order = [
        "MC-dropout",
        "TEGDA-ADIC",
        "SicTTA-CCD",
        "SafeTTA-Q66+dSemantic64",
    ]

    for method in method_order:
        mr = macro_table[macro_table["method"] == method]
        if len(mr) != 1:
            raise RuntimeError(f"Missing macro row for {method}")
        mr = mr.iloc[0]

        row = {
            "method": method,
            "macro_AUROC_mean": float(mr["macro_auroc_mean"]),
            "macro_AUROC_std": float(mr["macro_auroc_std"]),
            "macro_AUPRC_mean": float(mr["macro_auprc_mean"]),
            "macro_AUPRC_std": float(mr["macro_auprc_std"]),
            "macro_AUPRC_lift_mean": float(mr["macro_auprc_lift_mean"]),
        }

        for action in ACTIONS:
            g = action_table[
                (action_table["method"] == method)
                & (action_table["action"] == action)
            ]
            if len(g) != 1:
                raise RuntimeError(f"Missing action row {method}/{action}")
            g = g.iloc[0]

            token = (
                "MEMO"
                if action == "MEMO-SEG4-1STEP"
                else "PL"
                if action == "PL-CONF90"
                else "TENT"
            )
            row[f"{token}_AUROC"] = float(g["auroc_mean"])
            row[f"{token}_AUPRC"] = float(g["auprc_mean"])
            row[f"{token}_AUPRC_lift"] = float(g["auprc_lift_mean"])

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    print("=" * 124)
    print("SafeTTA R32B3 Matched Three-Action Published-Baseline HARM Evaluation")
    print("Version                     :", VERSION)
    print("Matched family              :", FAMILY)
    print("Physical cases              :", N_CASES)
    print("Model states                :", N_STATES)
    print("Future actions              :", list(ACTIONS))
    print("SafeTTA                     :", SAFE_MODEL)
    print("SafeTTA score ensemble      : NO")
    print("Bootstrap                   :", f"{BOOTSTRAP_REPS} physical-case clustered reps")
    print("Score direction reversal    : NO")
    print("Target calibration/tuning   : NO")
    print("New inference               : NO")
    print("=" * 124)

    upstream = verify_upstream()

    out = args.output_dir.resolve()
    if out.exists():
        raise RuntimeError(
            f"Output exists; R32B3 will not overwrite: {out}"
        )
    out.mkdir(parents=True, exist_ok=False)

    # Freeze all analysis choices before first HARM-outcome read.
    analysis_lock_path = write_analysis_lock(out)
    print("\nPRE-OUTCOME ANALYSIS LOCK")
    print("  path   :", analysis_lock_path)
    print("  SHA256 :", sha256_file(analysis_lock_path))

    # Baseline scores themselves remain outcome-blind frozen assets.
    baseline = load_baseline_scores()

    # FIRST HARM outcome read occurs here, after analysis lock.
    outcome = load_outcome_panel()
    safe = load_safettta_predictions(outcome)
    panel = build_wide_eval_panel(baseline, outcome, safe)

    # Exact matched join audit.
    preval = (
        panel.groupby("action", as_index=False)
        .agg(
            rows=("r31b3_harm_label", "size"),
            physical_cases=("sample_id", "nunique"),
            harm_count=("r31b3_harm_label", "sum"),
            harm_prevalence=("r31b3_harm_label", "mean"),
        )
        .sort_values("action", kind="mergesort")
    )

    base_action = baseline_action_metrics(panel)
    safe_per_seed, safe_action = safettta_action_metrics(panel)
    action_table = pd.concat(
        [base_action, safe_action],
        ignore_index=True,
    )

    safe_macro_seed, macro_table = macro_summary(
        action_table,
        safe_per_seed,
    )
    state_secondary = per_state_secondary(panel)
    bootstrap = bootstrap_pairwise(panel)
    paper_table = make_paper_table(action_table, macro_table)

    # Save.
    panel_path = out / "R32B3_MATCHED_EVALUATION_PANEL.csv"
    preval_path = out / "R32B3_ACTION_HARM_PREVALENCE.csv"
    safe_seed_path = out / "R32B3_SAFETTA_PER_SEED_ACTION_METRICS.csv"
    action_path = out / "R32B3_MAIN_ACTION_METRICS.csv"
    safe_macro_seed_path = out / "R32B3_SAFETTA_PER_SEED_MACRO_METRICS.csv"
    macro_path = out / "R32B3_MAIN_MACRO_METRICS.csv"
    state_path = out / "R32B3_PER_STATE_SECONDARY_METRICS.csv"
    bootstrap_path = out / "R32B3_PAIRED_CLUSTERED_BOOTSTRAP.csv"
    paper_path = out / "R32B3_PAPER_TABLE.csv"

    atomic_csv(panel, panel_path)
    atomic_csv(preval, preval_path)
    atomic_csv(safe_per_seed, safe_seed_path)
    atomic_csv(action_table, action_path)
    atomic_csv(safe_macro_seed, safe_macro_seed_path)
    atomic_csv(macro_table, macro_path)
    atomic_csv(state_secondary, state_path)
    atomic_csv(bootstrap, bootstrap_path)
    atomic_csv(paper_table, paper_path)

    # Descriptive decision, with no method reselection.
    safe_macro = macro_table[
        macro_table["method"] == "SafeTTA-Q66+dSemantic64"
    ].iloc[0]

    best_baseline_auroc = (
        macro_table[
            macro_table["method"].isin(BASELINES.keys())
        ]
        .sort_values("macro_auroc_mean", ascending=False)
        .iloc[0]
    )
    best_baseline_auprc = (
        macro_table[
            macro_table["method"].isin(BASELINES.keys())
        ]
        .sort_values("macro_auprc_mean", ascending=False)
        .iloc[0]
    )

    readout = {
        "status": "R32B3_DESCRIPTIVE_PUBLISHED_BASELINE_READOUT",
        "version": VERSION,
        "SafeTTA": {
            "macro_AUROC_mean": float(safe_macro["macro_auroc_mean"]),
            "macro_AUROC_std": float(safe_macro["macro_auroc_std"]),
            "macro_AUPRC_mean": float(safe_macro["macro_auprc_mean"]),
            "macro_AUPRC_std": float(safe_macro["macro_auprc_std"]),
        },
        "best_published_baseline_by_macro_AUROC": {
            "method": str(best_baseline_auroc["method"]),
            "macro_AUROC": float(best_baseline_auroc["macro_auroc_mean"]),
            "SafeTTA_minus_baseline": float(
                safe_macro["macro_auroc_mean"]
                - best_baseline_auroc["macro_auroc_mean"]
            ),
        },
        "best_published_baseline_by_macro_AUPRC": {
            "method": str(best_baseline_auprc["method"]),
            "macro_AUPRC": float(best_baseline_auprc["macro_auprc_mean"]),
            "SafeTTA_minus_baseline": float(
                safe_macro["macro_auprc_mean"]
                - best_baseline_auprc["macro_auprc_mean"]
            ),
        },
        "claim_rule": (
            "Do not claim universal superiority. Interpret per-action and macro "
            "AUROC/AUPRC together with paired clustered-bootstrap CIs."
        ),
        "next": "R32B4_PAPER_INTEGRATION_OR_EXTERNAL_JOINT_SHIFT_CONFIRMATION",
    }
    readout_path = out / "R32B3_DESCRIPTIVE_READOUT.json"
    atomic_json(readout_path, readout)

    final = {
        "status": "PASS_R32B3_MATCHED_THREE_ACTION_HARM_EVALUATION_COMPLETE",
        "version": VERSION,
        "scientific_status": "POST_R31B3_RETROSPECTIVE_MATCHED_EVALUATION",
        "analysis_lock_sha256": sha256_file(analysis_lock_path),
        "R32B2B_final_sha256": EXPECTED_R32B2B_FINAL_SHA256,
        "R32B2B_score_lock_sha256": EXPECTED_R32B2B_SCORE_LOCK_SHA256,
        "R32B2B_common_score_sha256": EXPECTED_R32B2B_COMMON_SHA256,
        "R32B1_manifest_sha256": EXPECTED_R32B1_MANIFEST_SHA256,
        "R32A1_final_sha256": EXPECTED_R32A1_FINAL_SHA256,
        "matched_panel_sha256": sha256_file(panel_path),
        "action_metrics_sha256": sha256_file(action_path),
        "macro_metrics_sha256": sha256_file(macro_path),
        "bootstrap_sha256": sha256_file(bootstrap_path),
        "paper_table_sha256": sha256_file(paper_path),
        "descriptive_readout_sha256": sha256_file(readout_path),
        "score_direction_reversal": False,
        "target_calibration": False,
        "threshold_tuning": False,
        "model_refitting": False,
        "score_ensemble_across_SafeTTA_seeds": False,
        "new_inference": False,
        "external_data_access": False,
        "next": "R32B4_PAPER_INTEGRATION_OR_EXTERNAL_JOINT_SHIFT_CONFIRMATION",
    }
    final_path = out / "R32B3_FINAL_LOCK.json"
    atomic_json(final_path, final)

    # Console.
    print("\n" + "=" * 124)
    print("R32B3 ACTION HARM PREVALENCE")
    print("=" * 124)
    print(preval.to_string(index=False))

    print("\n" + "=" * 124)
    print("R32B3 MATCHED ACTION METRICS")
    print("=" * 124)
    print(
        action_table[
            [
                "method",
                "action",
                "auroc_mean",
                "auroc_std",
                "auprc_mean",
                "auprc_std",
                "auprc_lift_mean",
                "harm_prevalence",
            ]
        ].sort_values(
            ["action", "auroc_mean"],
            ascending=[True, False],
            kind="mergesort",
        ).to_string(index=False)
    )

    print("\n" + "=" * 124)
    print("R32B3 MATCHED MACRO METRICS")
    print("=" * 124)
    print(
        macro_table[
            [
                "method",
                "macro_auroc_mean",
                "macro_auroc_std",
                "macro_auprc_mean",
                "macro_auprc_std",
                "macro_auprc_lift_mean",
            ]
        ].sort_values(
            "macro_auroc_mean",
            ascending=False,
            kind="mergesort",
        ).to_string(index=False)
    )

    print("\n" + "=" * 124)
    print("R32B3 PAIRED CLUSTERED BOOTSTRAP (SafeTTA - comparator)")
    print("=" * 124)
    print(
        bootstrap[
            [
                "comparator",
                "scope",
                "metric",
                "delta_SafeTTA_minus_comparator",
                "ci95_low",
                "ci95_high",
                "SafeTTA_significantly_better_95CI",
                "comparator_significantly_better_95CI",
            ]
        ].to_string(index=False)
    )

    print("\nR32B3 DESCRIPTIVE READOUT")
    print(
        "  SafeTTA macro AUROC :",
        f"{readout['SafeTTA']['macro_AUROC_mean']:.9f}",
        "+/-",
        f"{readout['SafeTTA']['macro_AUROC_std']:.9f}",
    )
    print(
        "  SafeTTA macro AUPRC :",
        f"{readout['SafeTTA']['macro_AUPRC_mean']:.9f}",
        "+/-",
        f"{readout['SafeTTA']['macro_AUPRC_std']:.9f}",
    )
    print(
        "  Best baseline AUROC :",
        readout["best_published_baseline_by_macro_AUROC"],
    )
    print(
        "  Best baseline AUPRC :",
        readout["best_published_baseline_by_macro_AUPRC"],
    )

    print(
        "\nFINAL STATUS : "
        "PASS_R32B3_MATCHED_THREE_ACTION_HARM_EVALUATION_COMPLETE"
    )
    print("Score direction reversal      : NO")
    print("Target calibration/tuning     : NO")
    print("SafeTTA score ensemble        : NO")
    print("New inference                 : NO")
    print("External cohort access        : NO")
    print("Final lock SHA256             :", sha256_file(final_path))
    print("Output                        :", out)
    print(
        "NEXT                          : "
        "R32B4_PAPER_INTEGRATION_OR_EXTERNAL_JOINT_SHIFT_CONFIRMATION"
    )
    print("=" * 124)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
