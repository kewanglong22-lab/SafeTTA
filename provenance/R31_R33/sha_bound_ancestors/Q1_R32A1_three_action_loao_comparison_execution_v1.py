#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SafeTTA R32A1
Three-Action Leave-One-Action-Out Comparison Execution

Executes the exact R32A0 comparison protocol on the frozen R31B3 panel.

NO new segmentation/TTA inference.
NO feature engineering.
NO fold regeneration.
NO hyperparameter search.
NO external cohort access.

LOAO directions:
  TENT1 + PL-CONF90              -> unseen MEMO-SEG4-1STEP
  TENT1 + MEMO-SEG4-1STEP       -> unseen PL-CONF90
  PL-CONF90 + MEMO-SEG4-1STEP   -> unseen TENT1

Models:
  RANDOM_CONSTANT
  STRUCTURE_ONLY_M2
  SOURCE_SEMANTIC64_ONLY
  SOURCE_Q66_ONLY
  DELTA_SEMANTIC64_ONLY
  SHARED_TRANSITION_Q66_DSEM64          [PRIMARY]
  ACTION_CONDITIONAL_Q66_DSEM64         [diagnostic]
  SEPARATE_ACTION_REFERENCE             [NON-LOAO upper reference]

Learned models use:
  StandardScaler(train fold only)
  + LogisticRegression(
        C=1,
        class_weight='balanced',
        solver='lbfgs',
        max_iter=5000,
        random_state=0
    )

Metrics:
  AUROC
  AUPRC
  AUPRC Lift = AUPRC / HARM prevalence

A replay sentinel verifies that:
  SHARED_TRANSITION_Q66_DSEM64
  ACTION_CONDITIONAL_Q66_DSEM64
  SEPARATE_ACTION_REFERENCE
reproduce the corresponding frozen R31B3 OOF metrics on the exact same folds.
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

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


VERSION = "2026-09-12-R32A1-v1"

ROOT = Path(r"F:\MEDSEG_SAFETTA")
CODE = ROOT / "code"

# ---------------------------------------------------------------------
# Exact R32A0 locks
# ---------------------------------------------------------------------

R32A0_SCRIPT = CODE / "Q1_R32A0_three_action_loao_comparison_protocol_lock_v1.py"
EXPECTED_R32A0_SCRIPT_SHA256 = (
    "a669a6a6bf20bd6383ed54e07191e5f2395572b11c03a3fedbd84d445353f7c6"
)

R32A0_DIR = ROOT / "R32A0_three_action_loao_comparison_protocol_lock_v1"
R32A0_SPLIT = R32A0_DIR / "R32A0_EXACT_R31B3_LOAO_SPLIT_MANIFEST.csv"
EXPECTED_R32A0_SPLIT_SHA256 = (
    "c28a7ca0421c45a4d44c968336445da7458bb514fb33f4add975b942ed6d254e"
)
R32A0_PROTOCOL = R32A0_DIR / "R32A0_COMPARISON_PROTOCOL_LOCK.json"
EXPECTED_R32A0_PROTOCOL_SHA256 = (
    "c3ae941ceaee4d9d2d4dfe35a930903aa1ae9046157af7d76a0d5d50d9012579"
)
R32A0_FINAL = R32A0_DIR / "R32A0_FINAL_LOCK.json"
EXPECTED_R32A0_FINAL_SHA256 = (
    "e4726a4067db276d7f250b6c1890b66064807911751c3bac64e846b610d8265f"
)

# ---------------------------------------------------------------------
# Frozen R31B3 / feature schema
# ---------------------------------------------------------------------

R31B3_DIR = ROOT / "R31B3_first_800case_gt_reveal_and_three_action_loao_v1"
R31B3_FINAL = R31B3_DIR / "R31B3_FINAL_LOCK.json"
EXPECTED_R31B3_FINAL_SHA256 = (
    "eb91d991752c50e71b1993e4ee0a50e0b9e9cdb35fffdd58b36553e0cebaaadc"
)
R31B3_TABLE = R31B3_DIR / "R31B3_THREE_ACTION_MODELCASE_TABLE.csv"
R31B3_OOF = R31B3_DIR / "R31B3_OOF_PREDICTIONS.csv"

R30A0_SCRIPT = CODE / "Q1_R30A0_paired_action_crc_protocol_and_asset_audit_v1_fix1.py"
EXPECTED_R30A0_SHA256 = (
    "9dc5d3a13f3d64608c4c599f811f929f731d7fac05e2aaa66eb344e24ace2298"
)

DEFAULT_OUT = ROOT / "R32A1_three_action_loao_comparison_execution_v1"

# ---------------------------------------------------------------------
# Frozen protocol constants
# ---------------------------------------------------------------------

ACTIONS = ("TENT1", "PL-CONF90", "MEMO-SEG4-1STEP")
EXPECTED_CASES = 800
EXPECTED_STATES = 9
ROWS_PER_ACTION = EXPECTED_CASES * EXPECTED_STATES
EXPECTED_PANEL_ROWS = ROWS_PER_ACTION * len(ACTIONS)

SPLIT_SEEDS = [20260912, 20260913, 20260914, 20260915, 20260916]
N_FOLDS = 5

LOAO_DIRECTIONS = (
    (("TENT1", "PL-CONF90"), "MEMO-SEG4-1STEP"),
    (("TENT1", "MEMO-SEG4-1STEP"), "PL-CONF90"),
    (("PL-CONF90", "MEMO-SEG4-1STEP"), "TENT1"),
)

LOAO_MODELS = (
    "RANDOM_CONSTANT",
    "STRUCTURE_ONLY_M2",
    "SOURCE_SEMANTIC64_ONLY",
    "SOURCE_Q66_ONLY",
    "DELTA_SEMANTIC64_ONLY",
    "SHARED_TRANSITION_Q66_DSEM64",
    "ACTION_CONDITIONAL_Q66_DSEM64",
)
UPPER_REFERENCE = "SEPARATE_ACTION_REFERENCE"

PRIMARY_MODEL = "SHARED_TRANSITION_Q66_DSEM64"

LR_C = 1.0
LR_MAX_ITER = 5000
REPLAY_TOL = 1e-10

R31B3_EQUIVALENT_MODELS = {
    "SHARED_TRANSITION_Q66_DSEM64": "LOAO_TRANSITION_ONLY",
    "ACTION_CONDITIONAL_Q66_DSEM64": "PRIMARY_LOAO_ACTION_CONDITIONAL",
    "SEPARATE_ACTION_REFERENCE": "SEPARATE_ACTION",
}

FORBIDDEN_EXTERNAL_TOKENS = (
    "endotect",
    "polypgen",
    "sun-seg",
    "sunseg",
    "promise12",
)


# ---------------------------------------------------------------------
# Helpers
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
        raise RuntimeError(f"Cannot import {path}")
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
        raise RuntimeError(f"External path forbidden in R32A1: {path}")


def safe_metrics(y: np.ndarray, score: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)

    prevalence = float(np.mean(y))
    if len(np.unique(y)) != 2:
        return {
            "auroc": math.nan,
            "auprc": math.nan,
            "brier": math.nan,
            "prevalence": prevalence,
            "auprc_lift": math.nan,
            "n": int(len(y)),
        }

    ap = float(average_precision_score(y, score))
    return {
        "auroc": float(roc_auc_score(y, score)),
        "auprc": ap,
        "brier": float(brier_score_loss(y, score)),
        "prevalence": prevalence,
        "auprc_lift": float(ap / prevalence) if prevalence > 0 else math.nan,
        "n": int(len(y)),
    }


# ---------------------------------------------------------------------
# Upstream verification
# ---------------------------------------------------------------------

def verify_upstream() -> Dict[str, Any]:
    require_sha(R32A0_SCRIPT, EXPECTED_R32A0_SCRIPT_SHA256, "R32A0 script")
    require_sha(R32A0_SPLIT, EXPECTED_R32A0_SPLIT_SHA256, "R32A0 split manifest")
    require_sha(R32A0_PROTOCOL, EXPECTED_R32A0_PROTOCOL_SHA256, "R32A0 protocol")
    require_sha(R32A0_FINAL, EXPECTED_R32A0_FINAL_SHA256, "R32A0 final")
    require_sha(R31B3_FINAL, EXPECTED_R31B3_FINAL_SHA256, "R31B3 final")
    require_sha(R30A0_SCRIPT, EXPECTED_R30A0_SHA256, "R30 feature schema")

    a0 = json.loads(R32A0_FINAL.read_text(encoding="utf-8"))
    if a0.get("status") != (
        "PASS_R32A0_THREE_ACTION_LOAO_COMPARISON_PROTOCOL_LOCK_COMPLETE"
    ):
        raise RuntimeError("R32A0 status changed.")
    if bool(a0.get("new_inference", True)):
        raise RuntimeError("R32A0 unexpectedly permits new inference.")
    if bool(a0.get("hyperparameter_search", True)):
        raise RuntimeError("R32A0 unexpectedly permits hyperparameter search.")
    if bool(a0.get("external_data_access", True)):
        raise RuntimeError("R32A0 reports external access.")

    if not R31B3_TABLE.is_file():
        raise FileNotFoundError(R31B3_TABLE)
    if sha256_file(R31B3_TABLE) != str(a0["R31B3_table_sha256"]):
        raise RuntimeError("R31B3 table SHA drift.")

    if not R31B3_OOF.is_file():
        raise FileNotFoundError(R31B3_OOF)
    if sha256_file(R31B3_OOF) != str(a0["R31B3_OOF_sha256"]):
        raise RuntimeError("R31B3 OOF SHA drift.")

    return a0


# ---------------------------------------------------------------------
# Load frozen panel and exact folds
# ---------------------------------------------------------------------

def load_schema() -> Tuple[List[str], List[str]]:
    mod = import_module(R30A0_SCRIPT, "r32a1_r30a0")
    qnames = list(getattr(mod, "QSOURCE", []))
    dnames = list(getattr(mod, "DSEM", []))
    if len(qnames) != 66 or len(dnames) != 64:
        raise RuntimeError(
            f"Frozen schema drift: QSOURCE={len(qnames)} DSEM={len(dnames)}"
        )
    return qnames, dnames


def load_panel(
    qnames: Sequence[str],
    dnames: Sequence[str],
) -> pd.DataFrame:
    reject_external(R31B3_TABLE)
    d = pd.read_csv(R31B3_TABLE, low_memory=False)

    required = {
        "sample_id",
        "model_state_id",
        "model_family",
        "action",
        "r31b3_harm_label",
        *qnames,
        *dnames,
    }
    missing = sorted(required.difference(d.columns))
    if missing:
        raise RuntimeError(f"R31B3 table missing columns={missing[:20]}")

    d["sample_id"] = d["sample_id"].astype(str)
    d["action"] = d["action"].astype(str)

    if len(d) != EXPECTED_PANEL_ROWS:
        raise RuntimeError(
            f"Panel rows={len(d)} expected={EXPECTED_PANEL_ROWS}"
        )
    if d["sample_id"].nunique() != EXPECTED_CASES:
        raise RuntimeError("Physical-case count drift.")
    if d["model_state_id"].nunique() != EXPECTED_STATES:
        raise RuntimeError("Model-state count drift.")
    if set(d["action"].unique()) != set(ACTIONS):
        raise RuntimeError(f"Action set drift={sorted(d['action'].unique())}")
    if d.duplicated(["sample_id", "model_state_id", "action"]).any():
        raise RuntimeError("Duplicate case/state/action rows.")

    y = d["r31b3_harm_label"].to_numpy(int)
    if not set(np.unique(y)).issubset({0, 1}):
        raise RuntimeError("HARM label is not binary.")

    X = d[list(qnames) + list(dnames)].to_numpy(float)
    if not np.isfinite(X).all():
        raise RuntimeError("Non-finite frozen R31B3 feature values.")

    return d


def load_split_manifest() -> pd.DataFrame:
    d = pd.read_csv(
        R32A0_SPLIT,
        dtype={"sample_id": str},
        low_memory=False,
    )

    required = {"split_seed", "fold", "sample_id"}
    missing = sorted(required.difference(d.columns))
    if missing:
        raise RuntimeError(f"Split manifest missing={missing}")

    if len(d) != len(SPLIT_SEEDS) * EXPECTED_CASES:
        raise RuntimeError("Split manifest row-count drift.")

    if d.duplicated(["split_seed", "sample_id"]).any():
        raise RuntimeError("Duplicate split assignment per seed/case.")

    for seed in SPLIT_SEEDS:
        g = d[d["split_seed"].astype(int) == seed]
        if g["sample_id"].nunique() != EXPECTED_CASES:
            raise RuntimeError(f"seed={seed} split case-count drift.")
        if set(g["fold"].astype(int).unique()) != set(range(N_FOLDS)):
            raise RuntimeError(f"seed={seed} split fold-set drift.")

    return d


# ---------------------------------------------------------------------
# Feature construction
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
    model_name: str,
    qnames: Sequence[str],
    dnames: Sequence[str],
    action_cats: Sequence[str],
    family_action_cats: Sequence[str],
) -> np.ndarray:
    qnames = list(qnames)
    dnames = list(dnames)

    if model_name == "STRUCTURE_ONLY_M2":
        X = d[qnames[:2]].to_numpy(float)

    elif model_name == "SOURCE_SEMANTIC64_ONLY":
        X = d[qnames[2:]].to_numpy(float)

    elif model_name == "SOURCE_Q66_ONLY":
        X = d[qnames].to_numpy(float)

    elif model_name == "DELTA_SEMANTIC64_ONLY":
        X = d[dnames].to_numpy(float)

    elif model_name in (
        "SHARED_TRANSITION_Q66_DSEM64",
        "SEPARATE_ACTION_REFERENCE",
    ):
        X = d[qnames + dnames].to_numpy(float)

    elif model_name == "ACTION_CONDITIONAL_Q66_DSEM64":
        blocks = [d[qnames + dnames].to_numpy(float)]

        a = d["action"].astype(str).to_numpy()
        for cat in action_cats:
            blocks.append((a == cat).astype(float)[:, None])

        fa = (
            d["model_family"].astype(str)
            + "::"
            + d["action"].astype(str)
        ).to_numpy()
        for cat in family_action_cats:
            blocks.append((fa == cat).astype(float)[:, None])

        X = np.concatenate(blocks, axis=1)

    else:
        raise ValueError(f"Unsupported model={model_name}")

    if not np.isfinite(X).all():
        raise RuntimeError(f"Non-finite X for {model_name}")
    return X


def fit_model(X: np.ndarray, y: np.ndarray):
    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) != 2:
        raise RuntimeError("Training fold has one HARM class.")

    model = Pipeline([
        ("scale", StandardScaler()),
        (
            "lr",
            LogisticRegression(
                C=LR_C,
                class_weight="balanced",
                solver="lbfgs",
                max_iter=LR_MAX_ITER,
                random_state=0,
            ),
        ),
    ])
    model.fit(X, y)
    return model


# ---------------------------------------------------------------------
# OOF execution
# ---------------------------------------------------------------------

def append_predictions(
    rows: List[Dict[str, Any]],
    test: pd.DataFrame,
    score: np.ndarray,
    seed: int,
    fold: int,
    model_name: str,
    evaluation_type: str,
    direction: str,
):
    y = test["r31b3_harm_label"].to_numpy(int)
    score = np.asarray(score, dtype=float)

    if len(test) != len(score):
        raise RuntimeError("Prediction length mismatch.")

    for r, yy, ss in zip(test.itertuples(index=False), y, score):
        rows.append({
            "split_seed": int(seed),
            "fold": int(fold),
            "model": model_name,
            "evaluation_type": evaluation_type,
            "direction": direction,
            "sample_id": str(r.sample_id),
            "model_state_id": str(r.model_state_id),
            "model_family": str(r.model_family),
            "action": str(r.action),
            "y_harm": int(yy),
            "score": float(ss),
        })


def run_all(
    panel: pd.DataFrame,
    split_manifest: pd.DataFrame,
    qnames: Sequence[str],
    dnames: Sequence[str],
) -> pd.DataFrame:
    action_cats, family_action_cats = global_category_schema(panel)
    pred_rows: List[Dict[str, Any]] = []

    pbar = tqdm(
        total=len(SPLIT_SEEDS) * N_FOLDS,
        desc="R32A1 exact-fold LOAO comparison",
        unit="fold",
        dynamic_ncols=True,
    )

    for seed in SPLIT_SEEDS:
        sm = split_manifest[
            split_manifest["split_seed"].astype(int) == seed
        ][["sample_id", "fold"]].copy()

        work = panel.merge(
            sm,
            on="sample_id",
            how="inner",
            validate="many_to_one",
        )

        if len(work) != EXPECTED_PANEL_ROWS:
            raise RuntimeError(
                f"seed={seed} merged panel rows={len(work)}"
            )

        for fold in range(N_FOLDS):
            train_cases = set(
                sm.loc[sm["fold"].astype(int) != fold, "sample_id"]
            )
            test_cases = set(
                sm.loc[sm["fold"].astype(int) == fold, "sample_id"]
            )

            if train_cases & test_cases:
                raise RuntimeError("Physical-case leakage.")
            if len(train_cases) + len(test_cases) != EXPECTED_CASES:
                raise RuntimeError("Physical-case split incomplete.")

            # ---------------------------------------------------------
            # True leave-one-action-out comparisons.
            # ---------------------------------------------------------
            for train_actions, test_action in LOAO_DIRECTIONS:
                train = work[
                    (work["sample_id"].isin(train_cases))
                    & (work["action"].isin(train_actions))
                ].copy()
                test = work[
                    (work["sample_id"].isin(test_cases))
                    & (work["action"] == test_action)
                ].copy()

                if len(test) != len(test_cases) * EXPECTED_STATES:
                    raise RuntimeError(
                        f"seed={seed} fold={fold} action={test_action} "
                        f"test rows={len(test)}"
                    )

                direction = f"{'+'.join(train_actions)}->{test_action}"

                for model_name in LOAO_MODELS:
                    if model_name == "RANDOM_CONSTANT":
                        score = np.full(len(test), 0.5, dtype=float)
                    else:
                        Xtr = make_X(
                            train,
                            model_name,
                            qnames,
                            dnames,
                            action_cats,
                            family_action_cats,
                        )
                        ytr = train["r31b3_harm_label"].to_numpy(int)
                        Xte = make_X(
                            test,
                            model_name,
                            qnames,
                            dnames,
                            action_cats,
                            family_action_cats,
                        )
                        model = fit_model(Xtr, ytr)
                        score = model.predict_proba(Xte)[:, 1]

                    append_predictions(
                        pred_rows,
                        test,
                        score,
                        seed,
                        fold,
                        model_name,
                        "LOAO",
                        direction,
                    )

            # ---------------------------------------------------------
            # Same-action upper reference.
            # ---------------------------------------------------------
            for action in ACTIONS:
                train = work[
                    (work["sample_id"].isin(train_cases))
                    & (work["action"] == action)
                ].copy()
                test = work[
                    (work["sample_id"].isin(test_cases))
                    & (work["action"] == action)
                ].copy()

                Xtr = make_X(
                    train,
                    UPPER_REFERENCE,
                    qnames,
                    dnames,
                    action_cats,
                    family_action_cats,
                )
                ytr = train["r31b3_harm_label"].to_numpy(int)
                Xte = make_X(
                    test,
                    UPPER_REFERENCE,
                    qnames,
                    dnames,
                    action_cats,
                    family_action_cats,
                )

                model = fit_model(Xtr, ytr)
                score = model.predict_proba(Xte)[:, 1]

                append_predictions(
                    pred_rows,
                    test,
                    score,
                    seed,
                    fold,
                    UPPER_REFERENCE,
                    "SAME_ACTION_REFERENCE",
                    f"{action}->{action}",
                )

            pbar.set_postfix(seed=seed, fold=fold)
            pbar.update(1)

    pbar.close()

    pred = pd.DataFrame(pred_rows)

    expected_per_model = (
        len(SPLIT_SEEDS) * EXPECTED_PANEL_ROWS
    )
    for model_name in list(LOAO_MODELS) + [UPPER_REFERENCE]:
        n = int((pred["model"] == model_name).sum())
        if n != expected_per_model:
            raise RuntimeError(
                f"{model_name} OOF rows={n} expected={expected_per_model}"
            )

    return pred


# ---------------------------------------------------------------------
# Metrics / macro summaries
# ---------------------------------------------------------------------

def per_seed_action_metrics(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (seed, model_name, eval_type, action), g in pred.groupby(
        ["split_seed", "model", "evaluation_type", "action"],
        sort=True,
    ):
        m = safe_metrics(
            g["y_harm"].to_numpy(int),
            g["score"].to_numpy(float),
        )
        rows.append({
            "split_seed": int(seed),
            "model": str(model_name),
            "evaluation_type": str(eval_type),
            "heldout_or_test_action": str(action),
            **m,
        })

    return pd.DataFrame(rows)


def action_summary(per_seed: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (model_name, eval_type, action), g in per_seed.groupby(
        ["model", "evaluation_type", "heldout_or_test_action"],
        sort=True,
    ):
        row = {
            "model": str(model_name),
            "evaluation_type": str(eval_type),
            "heldout_or_test_action": str(action),
            "seeds": int(g["split_seed"].nunique()),
        }

        for metric in [
            "auroc",
            "auprc",
            "brier",
            "prevalence",
            "auprc_lift",
        ]:
            v = g[metric].to_numpy(float)
            row[f"{metric}_mean"] = float(np.nanmean(v))
            row[f"{metric}_std"] = float(np.nanstd(v, ddof=1))
            row[f"{metric}_min"] = float(np.nanmin(v))
            row[f"{metric}_max"] = float(np.nanmax(v))

        rows.append(row)

    return pd.DataFrame(rows)


def per_seed_macro(per_seed: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (seed, model_name, eval_type), g in per_seed.groupby(
        ["split_seed", "model", "evaluation_type"],
        sort=True,
    ):
        if set(g["heldout_or_test_action"]) != set(ACTIONS):
            raise RuntimeError(
                f"Macro action set incomplete: seed={seed} model={model_name}"
            )

        rows.append({
            "split_seed": int(seed),
            "model": str(model_name),
            "evaluation_type": str(eval_type),
            "macro_auroc": float(g["auroc"].mean()),
            "macro_auprc": float(g["auprc"].mean()),
            "macro_auprc_lift": float(g["auprc_lift"].mean()),
            "macro_brier": float(g["brier"].mean()),
        })

    return pd.DataFrame(rows)


def macro_summary(macro_seed: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (model_name, eval_type), g in macro_seed.groupby(
        ["model", "evaluation_type"],
        sort=True,
    ):
        row = {
            "model": str(model_name),
            "evaluation_type": str(eval_type),
            "seeds": int(g["split_seed"].nunique()),
        }

        for metric in [
            "macro_auroc",
            "macro_auprc",
            "macro_auprc_lift",
            "macro_brier",
        ]:
            v = g[metric].to_numpy(float)
            row[f"{metric}_mean"] = float(v.mean())
            row[f"{metric}_std"] = float(v.std(ddof=1))
            row[f"{metric}_min"] = float(v.min())
            row[f"{metric}_max"] = float(v.max())

        rows.append(row)

    return pd.DataFrame(rows)


def primary_deltas(
    action_sum: pd.DataFrame,
    macro_sum: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    primary_action = action_sum[
        (action_sum["model"] == PRIMARY_MODEL)
        & (action_sum["evaluation_type"] == "LOAO")
    ][
        [
            "heldout_or_test_action",
            "auroc_mean",
            "auprc_mean",
            "auprc_lift_mean",
        ]
    ].rename(
        columns={
            "auroc_mean": "primary_auroc",
            "auprc_mean": "primary_auprc",
            "auprc_lift_mean": "primary_auprc_lift",
        }
    )

    compare_action = action_sum[
        action_sum["evaluation_type"] == "LOAO"
    ].merge(
        primary_action,
        on="heldout_or_test_action",
        how="left",
        validate="many_to_one",
    )

    compare_action["delta_auroc_primary_minus_model"] = (
        compare_action["primary_auroc"]
        - compare_action["auroc_mean"]
    )
    compare_action["delta_auprc_primary_minus_model"] = (
        compare_action["primary_auprc"]
        - compare_action["auprc_mean"]
    )
    compare_action["delta_auprc_lift_primary_minus_model"] = (
        compare_action["primary_auprc_lift"]
        - compare_action["auprc_lift_mean"]
    )

    p = macro_sum[
        (macro_sum["model"] == PRIMARY_MODEL)
        & (macro_sum["evaluation_type"] == "LOAO")
    ]
    if len(p) != 1:
        raise RuntimeError("Missing unique primary macro row.")

    p = p.iloc[0]
    compare_macro = macro_sum[
        macro_sum["evaluation_type"] == "LOAO"
    ].copy()

    compare_macro["delta_macro_auroc_primary_minus_model"] = (
        float(p["macro_auroc_mean"])
        - compare_macro["macro_auroc_mean"]
    )
    compare_macro["delta_macro_auprc_primary_minus_model"] = (
        float(p["macro_auprc_mean"])
        - compare_macro["macro_auprc_mean"]
    )
    compare_macro["delta_macro_auprc_lift_primary_minus_model"] = (
        float(p["macro_auprc_lift_mean"])
        - compare_macro["macro_auprc_lift_mean"]
    )

    return compare_action, compare_macro


# ---------------------------------------------------------------------
# Exact R31B3 replay sentinel
# ---------------------------------------------------------------------

def metrics_from_r31b3_oof() -> pd.DataFrame:
    old = pd.read_csv(R31B3_OOF, low_memory=False)

    required = {
        "split_seed",
        "model",
        "action",
        "y_harm",
        "score",
    }
    missing = sorted(required.difference(old.columns))
    if missing:
        raise RuntimeError(f"R31B3 OOF missing={missing}")

    rows = []

    for new_name, old_name in R31B3_EQUIVALENT_MODELS.items():
        g0 = old[old["model"] == old_name].copy()
        if len(g0) == 0:
            raise RuntimeError(f"R31B3 equivalent model absent: {old_name}")

        for (seed, action), g in g0.groupby(
            ["split_seed", "action"],
            sort=True,
        ):
            m = safe_metrics(
                g["y_harm"].to_numpy(int),
                g["score"].to_numpy(float),
            )
            rows.append({
                "split_seed": int(seed),
                "model": new_name,
                "heldout_or_test_action": str(action),
                "auroc_r31b3": m["auroc"],
                "auprc_r31b3": m["auprc"],
            })

    return pd.DataFrame(rows)


def replay_sentinel(
    per_seed: pd.DataFrame,
) -> pd.DataFrame:
    new = per_seed[
        per_seed["model"].isin(R31B3_EQUIVALENT_MODELS.keys())
    ][
        [
            "split_seed",
            "model",
            "heldout_or_test_action",
            "auroc",
            "auprc",
        ]
    ].copy()

    old = metrics_from_r31b3_oof()

    m = new.merge(
        old,
        on=[
            "split_seed",
            "model",
            "heldout_or_test_action",
        ],
        how="inner",
        validate="one_to_one",
    )

    expected = len(SPLIT_SEEDS) * len(ACTIONS) * len(R31B3_EQUIVALENT_MODELS)
    if len(m) != expected:
        raise RuntimeError(
            f"Replay sentinel rows={len(m)} expected={expected}"
        )

    m["abs_diff_auroc"] = np.abs(
        m["auroc"].to_numpy(float)
        - m["auroc_r31b3"].to_numpy(float)
    )
    m["abs_diff_auprc"] = np.abs(
        m["auprc"].to_numpy(float)
        - m["auprc_r31b3"].to_numpy(float)
    )

    max_auroc = float(m["abs_diff_auroc"].max())
    max_auprc = float(m["abs_diff_auprc"].max())

    if max_auroc > REPLAY_TOL or max_auprc > REPLAY_TOL:
        raise RuntimeError(
            "R31B3 replay sentinel failed: "
            f"max AUROC diff={max_auroc}, "
            f"max AUPRC diff={max_auprc}, "
            f"tol={REPLAY_TOL}"
        )

    m["status"] = "PASS"
    return m


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
    print("SafeTTA R32A1 Three-Action LOAO Comparison Execution")
    print("Version                  :", VERSION)
    print("Exact R31B3 folds        : YES")
    print("New segmentation inference: NO")
    print("Feature engineering      : NO")
    print("Hyperparameter search    : NO")
    print("External cohort access   : NO")
    print("Primary model            :", PRIMARY_MODEL)
    print("=" * 124)

    verify_upstream()
    qnames, dnames = load_schema()
    panel = load_panel(qnames, dnames)
    split_manifest = load_split_manifest()

    out = args.output_dir.resolve()
    if out.exists():
        raise RuntimeError(
            f"Output exists; R32A1 will not overwrite: {out}"
        )
    out.mkdir(parents=True, exist_ok=False)

    input_lock = {
        "status": "PASS_R32A1_INPUT_LOCK",
        "version": VERSION,
        "R32A0_script_sha256": EXPECTED_R32A0_SCRIPT_SHA256,
        "R32A0_split_sha256": EXPECTED_R32A0_SPLIT_SHA256,
        "R32A0_protocol_sha256": EXPECTED_R32A0_PROTOCOL_SHA256,
        "R32A0_final_sha256": EXPECTED_R32A0_FINAL_SHA256,
        "R31B3_final_sha256": EXPECTED_R31B3_FINAL_SHA256,
        "R31B3_table_sha256": sha256_file(R31B3_TABLE),
        "R31B3_OOF_sha256": sha256_file(R31B3_OOF),
        "physical_cases": EXPECTED_CASES,
        "model_states": EXPECTED_STATES,
        "actions": list(ACTIONS),
        "split_seeds": SPLIT_SEEDS,
        "folds": N_FOLDS,
        "q_source_dim": len(qnames),
        "delta_semantic_dim": len(dnames),
        "new_inference": False,
        "external_data_access": False,
    }
    input_lock_path = out / "R32A1_INPUT_LOCK.json"
    atomic_json(input_lock_path, input_lock)

    pred = run_all(
        panel,
        split_manifest,
        qnames,
        dnames,
    )

    per_seed = per_seed_action_metrics(pred)
    replay = replay_sentinel(per_seed)

    act_sum = action_summary(per_seed)
    macro_seed = per_seed_macro(per_seed)
    macro_sum = macro_summary(macro_seed)
    delta_action, delta_macro = primary_deltas(act_sum, macro_sum)

    # Output artifacts.
    pred_path = out / "R32A1_OOF_PREDICTIONS.csv"
    per_seed_path = out / "R32A1_PER_SEED_ACTION_METRICS.csv"
    action_sum_path = out / "R32A1_ACTION_SUMMARY.csv"
    macro_seed_path = out / "R32A1_PER_SEED_MACRO_METRICS.csv"
    macro_sum_path = out / "R32A1_MACRO_SUMMARY.csv"
    delta_action_path = out / "R32A1_PRIMARY_ACTION_DELTAS.csv"
    delta_macro_path = out / "R32A1_PRIMARY_MACRO_DELTAS.csv"
    replay_path = out / "R32A1_R31B3_REPLAY_SENTINEL.csv"

    atomic_csv(pred, pred_path)
    atomic_csv(per_seed, per_seed_path)
    atomic_csv(act_sum, action_sum_path)
    atomic_csv(macro_seed, macro_seed_path)
    atomic_csv(macro_sum, macro_sum_path)
    atomic_csv(delta_action, delta_action_path)
    atomic_csv(delta_macro, delta_macro_path)
    atomic_csv(replay, replay_path)

    # Mechanism readout is descriptive only; primary was locked in R32A0.
    primary_macro = macro_sum[
        (macro_sum["model"] == PRIMARY_MODEL)
        & (macro_sum["evaluation_type"] == "LOAO")
    ].iloc[0]

    delta_only_macro = macro_sum[
        (macro_sum["model"] == "DELTA_SEMANTIC64_ONLY")
        & (macro_sum["evaluation_type"] == "LOAO")
    ].iloc[0]

    q_only_macro = macro_sum[
        (macro_sum["model"] == "SOURCE_Q66_ONLY")
        & (macro_sum["evaluation_type"] == "LOAO")
    ].iloc[0]

    action_cond_macro = macro_sum[
        (macro_sum["model"] == "ACTION_CONDITIONAL_Q66_DSEM64")
        & (macro_sum["evaluation_type"] == "LOAO")
    ].iloc[0]

    mechanism = {
        "status": "DESCRIPTIVE_R32A1_MECHANISM_READOUT",
        "version": VERSION,
        "primary_model": PRIMARY_MODEL,
        "primary_macro": {
            "AUROC": float(primary_macro["macro_auroc_mean"]),
            "AUPRC": float(primary_macro["macro_auprc_mean"]),
            "AUPRC_lift": float(primary_macro["macro_auprc_lift_mean"]),
        },
        "delta_semantic_only_macro": {
            "AUROC": float(delta_only_macro["macro_auroc_mean"]),
            "AUPRC": float(delta_only_macro["macro_auprc_mean"]),
        },
        "source_q66_only_macro": {
            "AUROC": float(q_only_macro["macro_auroc_mean"]),
            "AUPRC": float(q_only_macro["macro_auprc_mean"]),
        },
        "action_conditional_macro": {
            "AUROC": float(action_cond_macro["macro_auroc_mean"]),
            "AUPRC": float(action_cond_macro["macro_auprc_mean"]),
        },
        "primary_minus_delta_only_AUROC": float(
            primary_macro["macro_auroc_mean"]
            - delta_only_macro["macro_auroc_mean"]
        ),
        "primary_minus_q66_only_AUROC": float(
            primary_macro["macro_auroc_mean"]
            - q_only_macro["macro_auroc_mean"]
        ),
        "primary_minus_action_conditional_AUROC": float(
            primary_macro["macro_auroc_mean"]
            - action_cond_macro["macro_auroc_mean"]
        ),
        "note": (
            "No post-hoc method reselection is performed here. "
            "SHARED_TRANSITION_Q66_DSEM64 was frozen as primary in R32A0."
        ),
        "next": "R32B_PUBLISHED_RELIABILITY_BASELINES",
    }
    mechanism_path = out / "R32A1_MECHANISM_READOUT.json"
    atomic_json(mechanism_path, mechanism)

    inventory = make_inventory(out)
    inventory_path = out / "R32A1_SHA256_INVENTORY.csv"
    atomic_csv(inventory, inventory_path)

    final = {
        "status": "PASS_R32A1_THREE_ACTION_LOAO_COMPARISON_COMPLETE",
        "version": VERSION,
        "scientific_status": "POST_R31B3_METHOD_INTERPRETATION_AND_COMPARISON",
        "primary_model": PRIMARY_MODEL,
        "R32A0_protocol_sha256": EXPECTED_R32A0_PROTOCOL_SHA256,
        "R32A0_final_sha256": EXPECTED_R32A0_FINAL_SHA256,
        "R31B3_final_sha256": EXPECTED_R31B3_FINAL_SHA256,
        "replay_sentinel": "PASS",
        "replay_max_abs_auroc_diff": float(replay["abs_diff_auroc"].max()),
        "replay_max_abs_auprc_diff": float(replay["abs_diff_auprc"].max()),
        "OOF_predictions_sha256": sha256_file(pred_path),
        "action_summary_sha256": sha256_file(action_sum_path),
        "macro_summary_sha256": sha256_file(macro_sum_path),
        "mechanism_readout_sha256": sha256_file(mechanism_path),
        "new_inference": False,
        "hyperparameter_search": False,
        "external_data_access": False,
        "next": "R32B_PUBLISHED_RELIABILITY_BASELINES",
    }
    final_path = out / "R32A1_FINAL_LOCK.json"
    atomic_json(final_path, final)

    # Console result tables.
    print("\n" + "=" * 124)
    print("R32A1 HELD-OUT ACTION SUMMARY")
    print("=" * 124)

    show_action = act_sum[
        [
            "model",
            "evaluation_type",
            "heldout_or_test_action",
            "auroc_mean",
            "auroc_std",
            "auprc_mean",
            "auprc_std",
            "prevalence_mean",
            "auprc_lift_mean",
        ]
    ]
    print(show_action.to_string(index=False))

    print("\n" + "=" * 124)
    print("R32A1 MACRO SUMMARY")
    print("=" * 124)

    show_macro = macro_sum[
        [
            "model",
            "evaluation_type",
            "macro_auroc_mean",
            "macro_auroc_std",
            "macro_auprc_mean",
            "macro_auprc_std",
            "macro_auprc_lift_mean",
        ]
    ].sort_values(
        ["evaluation_type", "macro_auroc_mean"],
        ascending=[True, False],
        kind="mergesort",
    )
    print(show_macro.to_string(index=False))

    print("\nR32A1 PRIMARY MECHANISM READOUT")
    print(
        "  primary macro AUROC                :",
        mechanism["primary_macro"]["AUROC"],
    )
    print(
        "  primary macro AUPRC                :",
        mechanism["primary_macro"]["AUPRC"],
    )
    print(
        "  primary - delta-only macro AUROC   :",
        mechanism["primary_minus_delta_only_AUROC"],
    )
    print(
        "  primary - Q66-only macro AUROC     :",
        mechanism["primary_minus_q66_only_AUROC"],
    )
    print(
        "  primary - action-cond macro AUROC  :",
        mechanism["primary_minus_action_conditional_AUROC"],
    )

    print("\nR31B3 REPLAY SENTINEL: PASS")
    print(
        "  max abs AUROC diff:",
        float(replay["abs_diff_auroc"].max()),
    )
    print(
        "  max abs AUPRC diff:",
        float(replay["abs_diff_auprc"].max()),
    )

    print("\nFINAL STATUS : PASS_R32A1_THREE_ACTION_LOAO_COMPARISON_COMPLETE")
    print("New segmentation/TTA inference : NO")
    print("Hyperparameter search          : NO")
    print("External cohort access         : NO")
    print("Final lock SHA256              :", sha256_file(final_path))
    print("Output                         :", out)
    print("NEXT                           : R32B_PUBLISHED_RELIABILITY_BASELINES")
    print("=" * 124)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
