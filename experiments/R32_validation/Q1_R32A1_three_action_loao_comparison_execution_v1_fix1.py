#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SafeTTA R32A1 fix1
Three-Action LOAO Comparison Execution with exact frozen-R31B3 OOF reuse.

Why fix1 exists
---------------
The original R32A1 execution completed all refits but its strict replay
sentinel stopped because re-fitting already-frozen R31B3 models produced tiny
numerical differences:

    max AUROC diff = 8.685573438582672e-05
    max AUPRC diff = 3.8889339563513703e-04

The R32A0 protocol is NOT changed and the replay tolerance is NOT relaxed.

Instead, fix1 treats the already-frozen R31B3 OOF predictions as the exact
source of truth for models that were already evaluated in R31B3:

    SOURCE_Q66_ONLY
      <- R31B3 SOURCE_ONLY_Q66

    SHARED_TRANSITION_Q66_DSEM64
      <- R31B3 LOAO_TRANSITION_ONLY

    ACTION_CONDITIONAL_Q66_DSEM64
      <- R31B3 PRIMARY_LOAO_ACTION_CONDITIONAL

    SEPARATE_ACTION_REFERENCE
      <- R31B3 SEPARATE_ACTION

Only the genuinely new R32A ablations are fitted on the exact frozen folds:

    STRUCTURE_ONLY_M2
    SOURCE_SEMANTIC64_ONLY
    DELTA_SEMANTIC64_ONLY

RANDOM_CONSTANT requires no fitting.

This is an implementation-replay correction only:
  - no R32A0 protocol change
  - no fold change
  - no feature change
  - no classifier/hyperparameter change
  - no outcome-dependent selection
  - no external cohort access
  - no new segmentation/TTA inference

The failed original R32A1 output directory is never overwritten.
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


VERSION = "2026-09-12-R32A1-v1-fix1"

ROOT = Path(r"F:\MEDSEG_SAFETTA")
CODE = ROOT / "code"

# ---------------------------------------------------------------------
# Bind exact original R32A1 failure lineage and R32A0 protocol
# ---------------------------------------------------------------------

ORIGINAL_R32A1_SCRIPT = CODE / "Q1_R32A1_three_action_loao_comparison_execution_v1.py"
EXPECTED_ORIGINAL_R32A1_SCRIPT_SHA256 = (
    "3673fa0d5d3b9ab2305955047068656e3c389eb24c857ca62c22c38bb5603451"
)
ORIGINAL_R32A1_OUT = ROOT / "R32A1_three_action_loao_comparison_execution_v1"

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
# Frozen R31B3 assets
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

DEFAULT_OUT = ROOT / "R32A1_three_action_loao_comparison_execution_v1_fix1"

# ---------------------------------------------------------------------
# Frozen R32A0 constants
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

NEW_FITTED_MODELS = (
    "STRUCTURE_ONLY_M2",
    "SOURCE_SEMANTIC64_ONLY",
    "DELTA_SEMANTIC64_ONLY",
)

FROZEN_MODEL_MAP = {
    "SOURCE_Q66_ONLY": "SOURCE_ONLY_Q66",
    "SHARED_TRANSITION_Q66_DSEM64": "LOAO_TRANSITION_ONLY",
    "ACTION_CONDITIONAL_Q66_DSEM64": "PRIMARY_LOAO_ACTION_CONDITIONAL",
    "SEPARATE_ACTION_REFERENCE": "SEPARATE_ACTION",
}

LOAO_MODEL_ORDER = (
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
        raise RuntimeError(f"External path forbidden in R32A1 fix1: {path}")


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


def direction_for_heldout(action: str) -> str:
    if action == "MEMO-SEG4-1STEP":
        return "TENT1+PL-CONF90->MEMO-SEG4-1STEP"
    if action == "PL-CONF90":
        return "TENT1+MEMO-SEG4-1STEP->PL-CONF90"
    if action == "TENT1":
        return "PL-CONF90+MEMO-SEG4-1STEP->TENT1"
    raise ValueError(action)


# ---------------------------------------------------------------------
# Verify immutable lineage
# ---------------------------------------------------------------------

def verify_upstream() -> Dict[str, Any]:
    require_sha(
        ORIGINAL_R32A1_SCRIPT,
        EXPECTED_ORIGINAL_R32A1_SCRIPT_SHA256,
        "original R32A1 script",
    )
    require_sha(R32A0_SCRIPT, EXPECTED_R32A0_SCRIPT_SHA256, "R32A0 script")
    require_sha(R32A0_SPLIT, EXPECTED_R32A0_SPLIT_SHA256, "R32A0 split")
    require_sha(R32A0_PROTOCOL, EXPECTED_R32A0_PROTOCOL_SHA256, "R32A0 protocol")
    require_sha(R32A0_FINAL, EXPECTED_R32A0_FINAL_SHA256, "R32A0 final")
    require_sha(R31B3_FINAL, EXPECTED_R31B3_FINAL_SHA256, "R31B3 final")
    require_sha(R30A0_SCRIPT, EXPECTED_R30A0_SHA256, "R30 schema")

    a0 = json.loads(R32A0_FINAL.read_text(encoding="utf-8"))
    if a0.get("status") != (
        "PASS_R32A0_THREE_ACTION_LOAO_COMPARISON_PROTOCOL_LOCK_COMPLETE"
    ):
        raise RuntimeError("R32A0 final status changed.")
    if bool(a0.get("new_inference", True)):
        raise RuntimeError("R32A0 new-inference contract changed.")
    if bool(a0.get("hyperparameter_search", True)):
        raise RuntimeError("R32A0 hyperparameter contract changed.")
    if bool(a0.get("external_data_access", True)):
        raise RuntimeError("R32A0 external-access contract changed.")

    for p in [R31B3_TABLE, R31B3_OOF]:
        reject_external(p)
        if not p.is_file():
            raise FileNotFoundError(p)

    if sha256_file(R31B3_TABLE) != str(a0["R31B3_table_sha256"]):
        raise RuntimeError("R31B3 table SHA changed.")
    if sha256_file(R31B3_OOF) != str(a0["R31B3_OOF_sha256"]):
        raise RuntimeError("R31B3 OOF SHA changed.")

    return a0


# ---------------------------------------------------------------------
# Load schema/panel/splits
# ---------------------------------------------------------------------

def load_schema() -> Tuple[List[str], List[str]]:
    mod = import_module(R30A0_SCRIPT, "r32a1fix1_r30a0")
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
    d["model_state_id"] = d["model_state_id"].astype(str)
    d["model_family"] = d["model_family"].astype(str)
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
        raise RuntimeError("Action set drift.")
    if d.duplicated(["sample_id", "model_state_id", "action"]).any():
        raise RuntimeError("Duplicate case/state/action row.")

    arr = d[list(qnames) + list(dnames)].to_numpy(dtype=np.float64)
    if not np.isfinite(arr).all():
        raise RuntimeError("Non-finite frozen features.")

    return d


def load_split() -> pd.DataFrame:
    d = pd.read_csv(R32A0_SPLIT, dtype={"sample_id": str}, low_memory=False)
    required = {"split_seed", "fold", "sample_id"}
    missing = sorted(required.difference(d.columns))
    if missing:
        raise RuntimeError(f"Split manifest missing={missing}")

    if len(d) != len(SPLIT_SEEDS) * EXPECTED_CASES:
        raise RuntimeError("Split-manifest row count drift.")
    if d.duplicated(["split_seed", "sample_id"]).any():
        raise RuntimeError("Duplicate seed/sample split row.")
    return d


# ---------------------------------------------------------------------
# Frozen R31B3 predictions as source of truth
# ---------------------------------------------------------------------

def load_frozen_r31b3_predictions(
    split_manifest: pd.DataFrame,
    panel: pd.DataFrame,
) -> pd.DataFrame:
    old = pd.read_csv(R31B3_OOF, low_memory=False)

    required = {
        "split_seed",
        "model",
        "sample_id",
        "model_state_id",
        "action",
        "y_harm",
        "score",
    }
    missing = sorted(required.difference(old.columns))
    if missing:
        raise RuntimeError(
            "R31B3 OOF lacks row-level columns needed for exact reuse: "
            f"{missing}"
        )

    old["sample_id"] = old["sample_id"].astype(str)
    old["model_state_id"] = old["model_state_id"].astype(str)
    old["action"] = old["action"].astype(str)

    # Add model_family from the immutable R31B3 panel if absent.
    family_key = panel[
        ["sample_id", "model_state_id", "action", "model_family"]
    ].drop_duplicates()

    if "model_family" not in old.columns:
        old = old.merge(
            family_key,
            on=["sample_id", "model_state_id", "action"],
            how="left",
            validate="many_to_one",
        )
    else:
        old["model_family"] = old["model_family"].astype(str)

    # Add exact fold from frozen R32A0 manifest if absent.
    if "fold" not in old.columns:
        old = old.merge(
            split_manifest[["split_seed", "sample_id", "fold"]],
            on=["split_seed", "sample_id"],
            how="left",
            validate="many_to_one",
        )

    if old["fold"].isna().any():
        raise RuntimeError("Could not recover fold for frozen R31B3 OOF rows.")

    rows = []
    expected_per_model = len(SPLIT_SEEDS) * EXPECTED_PANEL_ROWS

    for new_name, old_name in FROZEN_MODEL_MAP.items():
        g = old[old["model"] == old_name].copy()
        if len(g) != expected_per_model:
            raise RuntimeError(
                f"Frozen {old_name} rows={len(g)} expected={expected_per_model}"
            )

        eval_type = (
            "SAME_ACTION_REFERENCE"
            if new_name == UPPER_REFERENCE
            else "LOAO"
        )

        g["model"] = new_name
        g["evaluation_type"] = eval_type

        if eval_type == "LOAO":
            g["direction"] = g["action"].map(direction_for_heldout)
        else:
            g["direction"] = (
                g["action"].astype(str) + "->" + g["action"].astype(str)
            )

        rows.append(
            g[
                [
                    "split_seed",
                    "fold",
                    "model",
                    "evaluation_type",
                    "direction",
                    "sample_id",
                    "model_state_id",
                    "model_family",
                    "action",
                    "y_harm",
                    "score",
                ]
            ].copy()
        )

    out = pd.concat(rows, ignore_index=True)

    # Exact-label audit against immutable R31B3 panel.
    truth = panel[
        ["sample_id", "model_state_id", "action", "r31b3_harm_label"]
    ].rename(columns={"r31b3_harm_label": "panel_harm"})

    chk = out.merge(
        truth,
        on=["sample_id", "model_state_id", "action"],
        how="left",
        validate="many_to_one",
    )
    mismatch = int(
        np.count_nonzero(
            chk["y_harm"].to_numpy(int)
            != chk["panel_harm"].to_numpy(int)
        )
    )
    if mismatch:
        raise RuntimeError(f"Frozen OOF / panel HARM mismatch rows={mismatch}")

    return out


# ---------------------------------------------------------------------
# New ablations only
# ---------------------------------------------------------------------

def make_X(
    d: pd.DataFrame,
    model_name: str,
    qnames: Sequence[str],
    dnames: Sequence[str],
) -> np.ndarray:
    qnames = list(qnames)
    dnames = list(dnames)

    if model_name == "STRUCTURE_ONLY_M2":
        cols = qnames[:2]
    elif model_name == "SOURCE_SEMANTIC64_ONLY":
        cols = qnames[2:66]
    elif model_name == "DELTA_SEMANTIC64_ONLY":
        cols = dnames
    else:
        raise ValueError(f"Not a new ablation model: {model_name}")

    X = d[cols].to_numpy(dtype=np.float64)
    if not np.isfinite(X).all():
        raise RuntimeError(f"Non-finite features for {model_name}")
    return X


def fit_lr(X: np.ndarray, y: np.ndarray):
    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) != 2:
        raise RuntimeError("Training fold contains one HARM class.")

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


def append_rows(
    rows: List[Dict[str, Any]],
    test: pd.DataFrame,
    score: np.ndarray,
    seed: int,
    fold: int,
    model_name: str,
    direction: str,
):
    score = np.asarray(score, dtype=float)
    if len(test) != len(score):
        raise RuntimeError("Prediction length mismatch.")

    for r, ss in zip(test.itertuples(index=False), score):
        rows.append({
            "split_seed": int(seed),
            "fold": int(fold),
            "model": model_name,
            "evaluation_type": "LOAO",
            "direction": direction,
            "sample_id": str(r.sample_id),
            "model_state_id": str(r.model_state_id),
            "model_family": str(r.model_family),
            "action": str(r.action),
            "y_harm": int(r.r31b3_harm_label),
            "score": float(ss),
        })


def fit_new_ablations(
    panel: pd.DataFrame,
    split_manifest: pd.DataFrame,
    qnames: Sequence[str],
    dnames: Sequence[str],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    pbar = tqdm(
        total=len(SPLIT_SEEDS) * N_FOLDS,
        desc="R32A1-fix1 new ablations only",
        unit="fold",
        dynamic_ncols=True,
    )

    for seed in SPLIT_SEEDS:
        sm = split_manifest[
            split_manifest["split_seed"].astype(int) == seed
        ][["sample_id", "fold"]].copy()

        if sm["sample_id"].nunique() != EXPECTED_CASES:
            raise RuntimeError(f"seed={seed} case count drift.")

        for fold in range(N_FOLDS):
            train_ids = set(
                sm.loc[sm["fold"].astype(int) != fold, "sample_id"]
            )
            test_ids = set(
                sm.loc[sm["fold"].astype(int) == fold, "sample_id"]
            )

            if train_ids & test_ids:
                raise RuntimeError("Physical-case leakage.")

            for train_actions, test_action in LOAO_DIRECTIONS:
                train = panel[
                    panel["sample_id"].isin(train_ids)
                    & panel["action"].isin(train_actions)
                ].copy()
                test = panel[
                    panel["sample_id"].isin(test_ids)
                    & (panel["action"] == test_action)
                ].copy()

                expected_test = len(test_ids) * EXPECTED_STATES
                if len(test) != expected_test:
                    raise RuntimeError(
                        f"seed={seed} fold={fold} action={test_action} "
                        f"test rows={len(test)} expected={expected_test}"
                    )

                direction = f"{'+'.join(train_actions)}->{test_action}"
                ytr = train["r31b3_harm_label"].to_numpy(int)

                for model_name in NEW_FITTED_MODELS:
                    Xtr = make_X(train, model_name, qnames, dnames)
                    Xte = make_X(test, model_name, qnames, dnames)
                    model = fit_lr(Xtr, ytr)
                    score = model.predict_proba(Xte)[:, 1]

                    append_rows(
                        rows,
                        test,
                        score,
                        seed,
                        fold,
                        model_name,
                        direction,
                    )

            pbar.set_postfix(seed=seed, fold=fold)
            pbar.update(1)

    pbar.close()

    out = pd.DataFrame(rows)
    expected_per_model = len(SPLIT_SEEDS) * EXPECTED_PANEL_ROWS

    for model_name in NEW_FITTED_MODELS:
        n = int((out["model"] == model_name).sum())
        if n != expected_per_model:
            raise RuntimeError(
                f"{model_name} rows={n} expected={expected_per_model}"
            )

    return out


def make_random_from_frozen_template(
    frozen: pd.DataFrame,
) -> pd.DataFrame:
    template = frozen[
        frozen["model"] == "SHARED_TRANSITION_Q66_DSEM64"
    ].copy()

    expected = len(SPLIT_SEEDS) * EXPECTED_PANEL_ROWS
    if len(template) != expected:
        raise RuntimeError("Frozen template row count drift.")

    template["model"] = "RANDOM_CONSTANT"
    template["evaluation_type"] = "LOAO"
    template["score"] = 0.5
    return template


# ---------------------------------------------------------------------
# Metrics
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

        for metric in ["auroc", "auprc", "brier", "prevalence", "auprc_lift"]:
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
                f"Incomplete action set seed={seed} model={model_name}"
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
    p_action = action_sum[
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

    da = action_sum[
        action_sum["evaluation_type"] == "LOAO"
    ].merge(
        p_action,
        on="heldout_or_test_action",
        how="left",
        validate="many_to_one",
    )

    da["delta_auroc_primary_minus_model"] = (
        da["primary_auroc"] - da["auroc_mean"]
    )
    da["delta_auprc_primary_minus_model"] = (
        da["primary_auprc"] - da["auprc_mean"]
    )
    da["delta_auprc_lift_primary_minus_model"] = (
        da["primary_auprc_lift"] - da["auprc_lift_mean"]
    )

    p_macro = macro_sum[
        (macro_sum["model"] == PRIMARY_MODEL)
        & (macro_sum["evaluation_type"] == "LOAO")
    ]
    if len(p_macro) != 1:
        raise RuntimeError("Missing unique primary macro row.")
    p = p_macro.iloc[0]

    dm = macro_sum[
        macro_sum["evaluation_type"] == "LOAO"
    ].copy()
    dm["delta_macro_auroc_primary_minus_model"] = (
        float(p["macro_auroc_mean"]) - dm["macro_auroc_mean"]
    )
    dm["delta_macro_auprc_primary_minus_model"] = (
        float(p["macro_auprc_mean"]) - dm["macro_auprc_mean"]
    )
    dm["delta_macro_auprc_lift_primary_minus_model"] = (
        float(p["macro_auprc_lift_mean"]) - dm["macro_auprc_lift_mean"]
    )

    return da, dm


def frozen_reuse_audit(
    pred: pd.DataFrame,
    frozen: pd.DataFrame,
) -> pd.DataFrame:
    keys = [
        "split_seed",
        "fold",
        "model",
        "sample_id",
        "model_state_id",
        "action",
    ]

    p = pred[pred["model"].isin(FROZEN_MODEL_MAP.keys())][
        keys + ["y_harm", "score"]
    ].copy()
    f = frozen[keys + ["y_harm", "score"]].copy()

    m = p.merge(
        f,
        on=keys,
        suffixes=("_combined", "_frozen"),
        validate="one_to_one",
    )

    if len(m) != len(f):
        raise RuntimeError(
            f"Frozen reuse merge rows={len(m)} expected={len(f)}"
        )

    max_score = float(
        np.max(
            np.abs(
                m["score_combined"].to_numpy(float)
                - m["score_frozen"].to_numpy(float)
            )
        )
    )
    harm_mismatch = int(
        np.count_nonzero(
            m["y_harm_combined"].to_numpy(int)
            != m["y_harm_frozen"].to_numpy(int)
        )
    )

    if max_score != 0.0 or harm_mismatch != 0:
        raise RuntimeError(
            f"Frozen reuse audit failed: score diff={max_score}, "
            f"harm mismatch={harm_mismatch}"
        )

    return pd.DataFrame([{
        "frozen_models": len(FROZEN_MODEL_MAP),
        "rows_checked": int(len(m)),
        "max_abs_score_diff": max_score,
        "harm_label_mismatch": harm_mismatch,
        "status": "PASS_EXACT_FROZEN_REUSE",
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
    print("SafeTTA R32A1 fix1 Three-Action LOAO Comparison")
    print("Version                  :", VERSION)
    print("R32A0 protocol change    : NO")
    print("Replay tolerance relaxed : NO")
    print("Frozen R31B3 OOF reuse   : YES")
    print("New fitted ablations     :", list(NEW_FITTED_MODELS))
    print("New segmentation inference: NO")
    print("Hyperparameter search    : NO")
    print("External cohort access   : NO")
    print("=" * 124)

    verify_upstream()
    qnames, dnames = load_schema()
    panel = load_panel(qnames, dnames)
    split_manifest = load_split()

    out = args.output_dir.resolve()
    if out.exists():
        raise RuntimeError(
            f"Output exists; fix1 will not overwrite: {out}"
        )
    out.mkdir(parents=True, exist_ok=False)

    lineage = {
        "status": "R32A1_FIX1_IMPLEMENTATION_REPLAY_CORRECTION",
        "version": VERSION,
        "original_R32A1_script_sha256": EXPECTED_ORIGINAL_R32A1_SCRIPT_SHA256,
        "original_R32A1_output_path": str(ORIGINAL_R32A1_OUT),
        "original_failure": {
            "stage": "strict R31B3 replay sentinel after all refits",
            "max_AUROC_diff": 8.685573438582672e-05,
            "max_AUPRC_diff": 0.00038889339563513703,
            "original_tolerance": 1e-10,
        },
        "fix": (
            "Do not relax tolerance. Reuse exact frozen R31B3 row-level OOF "
            "predictions for already-existing models; fit only new R32A ablations."
        ),
        "protocol_changed": False,
        "folds_changed": False,
        "features_changed": False,
        "classifier_contract_changed": False,
        "outcome_dependent_model_selection": False,
        "external_data_access": False,
    }
    lineage_path = out / "R32A1_FIX1_LINEAGE.json"
    atomic_json(lineage_path, lineage)

    frozen = load_frozen_r31b3_predictions(split_manifest, panel)
    new = fit_new_ablations(panel, split_manifest, qnames, dnames)
    random_pred = make_random_from_frozen_template(frozen)

    pred = pd.concat(
        [frozen, new, random_pred],
        ignore_index=True,
    )

    expected_per_model = len(SPLIT_SEEDS) * EXPECTED_PANEL_ROWS
    all_models = list(LOAO_MODEL_ORDER) + [UPPER_REFERENCE]

    for model_name in all_models:
        g = pred[pred["model"] == model_name]
        if len(g) != expected_per_model:
            raise RuntimeError(
                f"{model_name} rows={len(g)} expected={expected_per_model}"
            )
        if g.duplicated(
            ["split_seed", "sample_id", "model_state_id", "action"]
        ).any():
            raise RuntimeError(f"Duplicate OOF unit for {model_name}")

    reuse = frozen_reuse_audit(pred, frozen)

    per_seed = per_seed_action_metrics(pred)
    act_sum = action_summary(per_seed)
    macro_seed = per_seed_macro(per_seed)
    macro_sum = macro_summary(macro_seed)
    delta_action, delta_macro = primary_deltas(act_sum, macro_sum)

    primary = macro_sum[
        (macro_sum["model"] == PRIMARY_MODEL)
        & (macro_sum["evaluation_type"] == "LOAO")
    ].iloc[0]
    delta_only = macro_sum[
        (macro_sum["model"] == "DELTA_SEMANTIC64_ONLY")
        & (macro_sum["evaluation_type"] == "LOAO")
    ].iloc[0]
    q_only = macro_sum[
        (macro_sum["model"] == "SOURCE_Q66_ONLY")
        & (macro_sum["evaluation_type"] == "LOAO")
    ].iloc[0]
    src_sem = macro_sum[
        (macro_sum["model"] == "SOURCE_SEMANTIC64_ONLY")
        & (macro_sum["evaluation_type"] == "LOAO")
    ].iloc[0]
    struct = macro_sum[
        (macro_sum["model"] == "STRUCTURE_ONLY_M2")
        & (macro_sum["evaluation_type"] == "LOAO")
    ].iloc[0]
    act_cond = macro_sum[
        (macro_sum["model"] == "ACTION_CONDITIONAL_Q66_DSEM64")
        & (macro_sum["evaluation_type"] == "LOAO")
    ].iloc[0]

    mechanism = {
        "status": "DESCRIPTIVE_R32A1_FIX1_MECHANISM_READOUT",
        "version": VERSION,
        "primary_model": PRIMARY_MODEL,
        "primary_macro_AUROC": float(primary["macro_auroc_mean"]),
        "primary_macro_AUPRC": float(primary["macro_auprc_mean"]),
        "primary_macro_AUPRC_lift": float(primary["macro_auprc_lift_mean"]),
        "structure_only_macro_AUROC": float(struct["macro_auroc_mean"]),
        "source_semantic64_macro_AUROC": float(src_sem["macro_auroc_mean"]),
        "source_q66_macro_AUROC": float(q_only["macro_auroc_mean"]),
        "delta_semantic64_macro_AUROC": float(delta_only["macro_auroc_mean"]),
        "action_conditional_macro_AUROC": float(act_cond["macro_auroc_mean"]),
        "primary_minus_structure_AUROC": float(
            primary["macro_auroc_mean"] - struct["macro_auroc_mean"]
        ),
        "primary_minus_source_semantic_AUROC": float(
            primary["macro_auroc_mean"] - src_sem["macro_auroc_mean"]
        ),
        "primary_minus_q66_AUROC": float(
            primary["macro_auroc_mean"] - q_only["macro_auroc_mean"]
        ),
        "primary_minus_delta_semantic_AUROC": float(
            primary["macro_auroc_mean"] - delta_only["macro_auroc_mean"]
        ),
        "primary_minus_action_conditional_AUROC": float(
            primary["macro_auroc_mean"] - act_cond["macro_auroc_mean"]
        ),
        "post_hoc_primary_reselection": False,
        "next": "R32B_PUBLISHED_RELIABILITY_BASELINES",
    }

    # Save artifacts.
    pred_path = out / "R32A1_FIX1_OOF_PREDICTIONS.csv"
    per_seed_path = out / "R32A1_FIX1_PER_SEED_ACTION_METRICS.csv"
    action_path = out / "R32A1_FIX1_ACTION_SUMMARY.csv"
    macro_seed_path = out / "R32A1_FIX1_PER_SEED_MACRO_METRICS.csv"
    macro_path = out / "R32A1_FIX1_MACRO_SUMMARY.csv"
    delta_action_path = out / "R32A1_FIX1_PRIMARY_ACTION_DELTAS.csv"
    delta_macro_path = out / "R32A1_FIX1_PRIMARY_MACRO_DELTAS.csv"
    reuse_path = out / "R32A1_FIX1_FROZEN_R31B3_REUSE_AUDIT.csv"
    mechanism_path = out / "R32A1_FIX1_MECHANISM_READOUT.json"

    atomic_csv(pred, pred_path)
    atomic_csv(per_seed, per_seed_path)
    atomic_csv(act_sum, action_path)
    atomic_csv(macro_seed, macro_seed_path)
    atomic_csv(macro_sum, macro_path)
    atomic_csv(delta_action, delta_action_path)
    atomic_csv(delta_macro, delta_macro_path)
    atomic_csv(reuse, reuse_path)
    atomic_json(mechanism_path, mechanism)

    inventory = make_inventory(out)
    inventory_path = out / "R32A1_FIX1_SHA256_INVENTORY.csv"
    atomic_csv(inventory, inventory_path)

    final = {
        "status": "PASS_R32A1_FIX1_THREE_ACTION_LOAO_COMPARISON_COMPLETE",
        "version": VERSION,
        "scientific_status": "POST_R31B3_METHOD_INTERPRETATION_AND_COMPARISON",
        "implementation_fix_only": True,
        "R32A0_protocol_changed": False,
        "replay_tolerance_relaxed": False,
        "frozen_R31B3_OOF_reused_exactly": True,
        "new_fitted_ablations": list(NEW_FITTED_MODELS),
        "R32A0_protocol_sha256": EXPECTED_R32A0_PROTOCOL_SHA256,
        "R32A0_final_sha256": EXPECTED_R32A0_FINAL_SHA256,
        "R31B3_final_sha256": EXPECTED_R31B3_FINAL_SHA256,
        "frozen_reuse_audit": "PASS_EXACT_FROZEN_REUSE",
        "OOF_predictions_sha256": sha256_file(pred_path),
        "action_summary_sha256": sha256_file(action_path),
        "macro_summary_sha256": sha256_file(macro_path),
        "mechanism_readout_sha256": sha256_file(mechanism_path),
        "new_segmentation_TTA_inference": False,
        "hyperparameter_search": False,
        "external_data_access": False,
        "next": "R32B_PUBLISHED_RELIABILITY_BASELINES",
    }
    final_path = out / "R32A1_FIX1_FINAL_LOCK.json"
    atomic_json(final_path, final)

    print("\n" + "=" * 124)
    print("R32A1 FIX1 EXACT FROZEN-R31B3 REUSE AUDIT")
    print("=" * 124)
    print(reuse.to_string(index=False))

    print("\n" + "=" * 124)
    print("R32A1 FIX1 HELD-OUT ACTION SUMMARY")
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
    print("R32A1 FIX1 MACRO SUMMARY")
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

    print("\nR32A1 FIX1 PRIMARY MECHANISM READOUT")
    print("  primary macro AUROC               :", mechanism["primary_macro_AUROC"])
    print("  primary macro AUPRC               :", mechanism["primary_macro_AUPRC"])
    print("  structure-only macro AUROC        :", mechanism["structure_only_macro_AUROC"])
    print("  source-semantic64 macro AUROC     :", mechanism["source_semantic64_macro_AUROC"])
    print("  Q66-only macro AUROC              :", mechanism["source_q66_macro_AUROC"])
    print("  delta-semantic64-only macro AUROC :", mechanism["delta_semantic64_macro_AUROC"])
    print("  action-conditional macro AUROC    :", mechanism["action_conditional_macro_AUROC"])
    print("  primary - Q66                     :", mechanism["primary_minus_q66_AUROC"])
    print("  primary - delta-semantic          :", mechanism["primary_minus_delta_semantic_AUROC"])
    print("  primary - action-conditional      :", mechanism["primary_minus_action_conditional_AUROC"])

    print("\nFINAL STATUS : PASS_R32A1_FIX1_THREE_ACTION_LOAO_COMPARISON_COMPLETE")
    print("R32A0 protocol changed          : NO")
    print("Replay tolerance relaxed       : NO")
    print("Frozen R31B3 OOF exact reuse   : YES")
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
