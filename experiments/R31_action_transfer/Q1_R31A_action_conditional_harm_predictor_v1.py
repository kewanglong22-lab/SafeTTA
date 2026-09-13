#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SafeTTA R31A
Unified Action-Conditional HARM Predictor + Leakage-Free Cross-Action Feasibility

This is the first innovation-enhancement experiment after the frozen R30
EndoTect confirmation.

R31A NEVER accesses EndoTect, PolypGen, SUN-SEG, PROMISE12 or any external
target outcome. It reuses only the already-existing NeoPolyp SOURCE-development
q_source66 + delta_semantic64 action table.

Questions
---------
Q1. Can one shared HARM predictor model TENT1 and PL-CONF90 without materially
    losing action-specific discrimination?
Q2. Does q_source66 + delta_semantic64 contain transferable structure such that
    a model trained on one action generalizes to the other action on held-out
    physical images?

Models
------
M0 SOURCE_ONLY_Q66
    q_source66 only; shared-vulnerability reference.

M1 SHARED_TRANSITION
    q_source66 + delta_semantic64; one shared HARM head, no action ID.

M2 SHARED_ACTION_CONDITIONAL
    q_source66 + delta_semantic64 + ActionID + FamilyActionID.

M3 SEPARATE_ACTION
    independent TENT1 and PL-CONF90 HARM heads.

X1 CROSS_ACTION_TENT_TO_PL
    train only TENT1 on training physical images;
    test PL-CONF90 only on held-out physical images.

X2 CROSS_ACTION_PL_TO_TENT
    train only PL-CONF90 on training physical images;
    test TENT1 only on held-out physical images.

Leakage control
---------------
Physical sample_id is split BEFORE action selection. Thus X1/X2 hold out both:
  (a) the test action from fitting and
  (b) the physical images used at test time.

With only two existing actions, a GO result is a feasibility gate for R31B;
it is NOT final proof of generalization to arbitrary unseen TTA algorithms.

No hyperparameter search is performed.

Run
---
python Q1_R31A_action_conditional_harm_predictor_v1.py

Optional explicit table:
python Q1_R31A_action_conditional_harm_predictor_v1.py --source-table <csv>
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
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


VERSION = "2026-09-12-R31A-v1"

ROOT = Path(r"F:\MEDSEG_SAFETTA")
CODE = ROOT / "code"

R30A0_SCRIPT = CODE / "Q1_R30A0_paired_action_crc_protocol_and_asset_audit_v1_fix1.py"
EXPECTED_R30A0_SHA = "9dc5d3a13f3d64608c4c599f811f929f731d7fac05e2aaa66eb344e24ace2298"

R30A3_SCRIPT = CODE / "Q1_R30A3_final_source_train_calibration_freeze_v1.py"
EXPECTED_R30A3_SCRIPT_SHA = "e9645e28ff2010229f79467581a9fa60a59cb56ce94fbd2fc2975c7b674e555c"

R30A3_DIR = ROOT / "R30A3_final_source_train_calibration_freeze_v1"
R30A3_FINAL = R30A3_DIR / "R30A3_FINAL_LOCK.json"
EXPECTED_R30A3_FINAL_SHA = "32a462132df8d5fb1d0102fc67af9ac81ad1fc75c103b28b14a98d9c8d3b63f8"

DEFAULT_OUT = ROOT / "R31A_action_conditional_harm_predictor_v1"

EXPECTED_ACTIONS = ("TENT1", "PL-CONF90")
EXPECTED_CASES = 1000
EXPECTED_STATES = 9
EXPECTED_ROWS_PER_ACTION = EXPECTED_CASES * EXPECTED_STATES
EXPECTED_TOTAL_ROWS = EXPECTED_ROWS_PER_ACTION * 2

HARM_THRESHOLD = -0.02

CV_SEEDS = [20260912, 20260913, 20260914, 20260915, 20260916]
N_FOLDS = 5

LR_C = 1.0
LR_MAX_ITER = 5000

# R&D gate only; these are not paper significance thresholds.
GO_CROSS_MACRO = 0.65
GO_CROSS_MIN_DIRECTION = 0.60
GO_SHARED_GAP = -0.03

STRONG_CROSS_MACRO = 0.70
STRONG_CROSS_MIN_DIRECTION = 0.65
STRONG_SHARED_GAP = -0.02

FORBIDDEN_PATH_TOKENS = (
    "endotect",
    "polypgen",
    "sun-seg",
    "sunseg",
    "promise12",
    "external_confirmation",
    "external_confirm",
)

OUTCOME_CANDIDATES = ["harm_label", "is_harm", "harm"]
DELTA_CANDIDATES = [
    "delta_dice",
    "candidate_delta_dice",
    "true_delta_dice",
    "action_delta_dice",
    "utility_target",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require_exact_sha(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"{label}: {path}")
    got = sha256_file(path)
    if got.lower() != expected.lower():
        raise RuntimeError(
            f"{label} SHA mismatch expected={expected} observed={got} path={path}"
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


def atomic_write_json(path: Path, payload: Any):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def normalize_action(v: Any) -> str:
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
    }
    return aliases.get(s, str(v).strip())


def path_is_external(path: Path) -> bool:
    low = str(path).lower()
    return any(tok in low for tok in FORBIDDEN_PATH_TOKENS)


def recursively_collect_paths(obj: Any) -> List[Path]:
    out: List[Path] = []
    if isinstance(obj, dict):
        for v in obj.values():
            if isinstance(v, str):
                p = Path(v)
                if p.suffix.lower() in {".csv", ".json", ".npz", ".joblib"}:
                    out.append(p)
            else:
                out.extend(recursively_collect_paths(v))
    elif isinstance(obj, list):
        for v in obj:
            out.extend(recursively_collect_paths(v))
    return out


def collect_candidate_csvs(r30a3_mod) -> List[Path]:
    candidates = set()

    if R30A3_FINAL.is_file():
        try:
            lock = json.loads(R30A3_FINAL.read_text(encoding="utf-8"))
            for p in recursively_collect_paths(lock):
                if p.suffix.lower() == ".csv" and p.is_file():
                    candidates.add(p.resolve())
        except Exception:
            pass

    for v in vars(r30a3_mod).values():
        p = None
        if isinstance(v, Path):
            p = v
        elif isinstance(v, str) and v.lower().endswith(".csv"):
            p = Path(v)
        if p is not None and p.is_file():
            candidates.add(p.resolve())

    roots = [R30A3_DIR]
    if ROOT.is_dir():
        for p in ROOT.iterdir():
            if p.is_dir() and p.name.lower().startswith("r30a"):
                roots.append(p)

    out_root = ROOT / "outputs"
    if out_root.is_dir():
        for p in out_root.iterdir():
            if p.is_dir() and "r30" in p.name.lower():
                roots.append(p)

    for root in roots:
        if root.is_dir():
            for p in root.rglob("*.csv"):
                candidates.add(p.resolve())

    return sorted(
        p for p in candidates
        if p.is_file() and not path_is_external(p)
    )


def read_header(path: Path) -> List[str]:
    try:
        return list(pd.read_csv(path, nrows=0).columns)
    except Exception:
        return []


def resolve_harm_column(d: pd.DataFrame) -> Tuple[np.ndarray, str]:
    for c in OUTCOME_CANDIDATES:
        if c in d.columns:
            y = pd.to_numeric(d[c], errors="raise").to_numpy(int)
            if not set(np.unique(y)).issubset({0, 1}):
                raise RuntimeError(f"{c} is not binary.")
            return y, c

    for c in DELTA_CANDIDATES:
        if c in d.columns:
            delta = pd.to_numeric(d[c], errors="raise").to_numpy(float)
            if not np.isfinite(delta).all():
                raise RuntimeError(f"Non-finite {c}.")
            return (delta <= HARM_THRESHOLD).astype(int), f"{c}<=-0.02"

    raise RuntimeError("No reproducible HARM label or DeltaDice column.")


def validate_source_table_structure(
    d: pd.DataFrame,
    qnames: List[str],
    dnames: List[str],
    soft: bool = False,
):
    required = [
        "sample_id", "model_state_id", "model_family", "action"
    ] + qnames + dnames

    missing = [c for c in required if c not in d.columns]
    if missing:
        raise RuntimeError(f"missing columns={missing[:20]}")

    if len(d) != EXPECTED_TOTAL_ROWS:
        raise RuntimeError(f"rows={len(d)} expected={EXPECTED_TOTAL_ROWS}")

    actions = d["action"].map(normalize_action)
    counts = actions.value_counts().to_dict()

    for action in EXPECTED_ACTIONS:
        if int(counts.get(action, 0)) != EXPECTED_ROWS_PER_ACTION:
            raise RuntimeError(
                f"{action} rows={counts.get(action, 0)} "
                f"expected={EXPECTED_ROWS_PER_ACTION}"
            )

    if d["sample_id"].astype(str).nunique() != EXPECTED_CASES:
        raise RuntimeError(
            f"physical cases={d['sample_id'].astype(str).nunique()} "
            f"expected={EXPECTED_CASES}"
        )

    if d["model_state_id"].astype(str).nunique() != EXPECTED_STATES:
        raise RuntimeError(
            f"states={d['model_state_id'].astype(str).nunique()} "
            f"expected={EXPECTED_STATES}"
        )

    tmp = d.copy()
    tmp["__action"] = actions

    if tmp[["sample_id", "model_state_id", "__action"]].duplicated().any():
        raise RuntimeError("Duplicate sample_id/model_state_id/action rows.")

    numeric = tmp[qnames + dnames].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise RuntimeError("Non-finite q_source/delta_semantic features.")

    y, _ = resolve_harm_column(tmp)
    if y.sum() <= 0 or y.sum() >= len(y):
        raise RuntimeError("Degenerate HARM labels.")

    if not soft:
        per_case = tmp.groupby(["sample_id", "__action"]).size()
        if not np.all(per_case.to_numpy(int) == EXPECTED_STATES):
            raise RuntimeError("Each physical case/action must contain 9 states.")


def resolve_source_table(
    explicit: Path | None,
    r30a3_mod,
    qnames: List[str],
    dnames: List[str],
) -> Tuple[Path, pd.DataFrame, Dict[str, Any]]:
    if explicit is not None:
        path = explicit.resolve()
        if path_is_external(path):
            raise RuntimeError(f"External/target table forbidden: {path}")
        if not path.is_file():
            raise FileNotFoundError(path)
        d = pd.read_csv(path, low_memory=False)
        return path, d, {"mode": "explicit"}

    candidates = collect_candidate_csvs(r30a3_mod)
    required_core = {"sample_id", "model_state_id", "model_family", "action"}
    required_features = set(qnames + dnames)

    scored = []
    for p in tqdm(
        candidates,
        desc="R31A locate frozen source action table",
        unit="csv",
        dynamic_ncols=True,
    ):
        cols = read_header(p)
        cset = set(cols)
        core_hits = len(required_core.intersection(cset))
        feat_hits = len(required_features.intersection(cset))
        outcome_hits = int(any(c in cset for c in OUTCOME_CANDIDATES))
        delta_hits = int(any(c in cset for c in DELTA_CANDIDATES))
        score = core_hits * 1000 + feat_hits * 10 + outcome_hits * 5 + delta_hits * 5
        scored.append((score, p, core_hits, feat_hits, outcome_hits, delta_hits))

    scored.sort(key=lambda x: (-x[0], str(x[1])))

    rejected = []
    for score, p, core_hits, feat_hits, outcome_hits, delta_hits in scored:
        if core_hits < len(required_core):
            continue
        if feat_hits < len(required_features):
            continue
        if not (outcome_hits or delta_hits):
            continue
        try:
            d = pd.read_csv(p, low_memory=False)
            validate_source_table_structure(d, qnames, dnames, soft=True)
            return p, d, {
                "mode": "auto_discovery",
                "candidate_count": len(candidates),
                "selected_score": int(score),
                "top_candidates": [
                    {
                        "score": int(x[0]),
                        "path": str(x[1]),
                        "core_hits": int(x[2]),
                        "feature_hits": int(x[3]),
                        "outcome_hits": int(x[4]),
                        "delta_hits": int(x[5]),
                    }
                    for x in scored[:20]
                ],
            }
        except Exception as e:
            rejected.append((str(p), str(e)))

    tops = "\n".join(
        f"  score={x[0]} core={x[2]} feat={x[3]} "
        f"outcome={x[4]} delta={x[5]} path={x[1]}"
        for x in scored[:20]
    )
    rej = "\n".join(f"  {p}: {e}" for p, e in rejected[:20])

    raise RuntimeError(
        "Could not resolve exact frozen 18,000-row SOURCE-development "
        "TENT1+PL table.\nTop candidates:\n"
        + tops
        + "\nRejected:\n"
        + rej
    )


def add_exact_labels_and_ids(
    d: pd.DataFrame,
    qnames: List[str],
) -> pd.DataFrame:
    x = d.copy()
    x["action"] = x["action"].map(normalize_action)

    bad = sorted(set(x["action"]) - set(EXPECTED_ACTIONS))
    if bad:
        raise RuntimeError(f"Unexpected actions={bad}")

    y, y_source = resolve_harm_column(x)
    x["r31a_harm_label"] = y.astype(int)
    x["family_action_id"] = (
        x["model_family"].astype(str) + "::" + x["action"].astype(str)
    )
    x["r31a_row_id"] = np.arange(len(x), dtype=int)

    key = ["sample_id", "model_state_id"]
    tent = (
        x[x["action"] == "TENT1"][key + qnames]
        .sort_values(key, kind="mergesort")
        .reset_index(drop=True)
    )
    pl = (
        x[x["action"] == "PL-CONF90"][key + qnames]
        .sort_values(key, kind="mergesort")
        .reset_index(drop=True)
    )

    if not tent[key].equals(pl[key]):
        raise RuntimeError("TENT/PL model-case keys do not align.")

    tq = tent[qnames].to_numpy(float)
    pq = pl[qnames].to_numpy(float)
    if not np.array_equal(tq, pq):
        raise RuntimeError(
            "q_source differs between paired actions; "
            f"max_abs={float(np.max(np.abs(tq-pq)))}"
        )

    x.attrs["harm_label_source"] = y_source
    return x


def make_group_folds(d: pd.DataFrame, seed: int) -> Dict[str, int]:
    g = (
        d.groupby("sample_id", as_index=False)
        .agg(harm_rate=("r31a_harm_label", "mean"))
        .sort_values("sample_id", kind="mergesort")
        .reset_index(drop=True)
    )

    ranks = g["harm_rate"].rank(method="first", pct=True).to_numpy(float)
    strata = np.minimum((ranks * 10).astype(int), 9)

    vc = pd.Series(strata).value_counts()
    if int(vc.min()) < N_FOLDS:
        strata = np.minimum((ranks * 5).astype(int), 4)

    cv = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=seed,
    )
    dummy = np.zeros(len(g), dtype=int)

    fold_map: Dict[str, int] = {}
    for fold, (_, test_idx) in enumerate(cv.split(dummy, strata)):
        for i in test_idx:
            sid = str(g.iloc[int(i)]["sample_id"])
            if sid in fold_map:
                raise RuntimeError("Physical sample assigned twice.")
            fold_map[sid] = int(fold)

    if len(fold_map) != EXPECTED_CASES:
        raise RuntimeError(
            f"fold cases={len(fold_map)} expected={EXPECTED_CASES}"
        )
    return fold_map


def categorical_matrix(d: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    action_cats = ["TENT1", "PL-CONF90"]
    families = sorted(d["model_family"].astype(str).unique().tolist())
    family_action_cats = [
        f"{fam}::{action}"
        for fam in families
        for action in action_cats
    ]

    blocks = []
    names = []

    a = d["action"].astype(str).to_numpy()
    for cat in action_cats:
        blocks.append((a == cat).astype(float)[:, None])
        names.append("ActionID::" + cat)

    fa = d["family_action_id"].astype(str).to_numpy()
    for cat in family_action_cats:
        blocks.append((fa == cat).astype(float)[:, None])
        names.append("FamilyActionID::" + cat)

    return np.concatenate(blocks, axis=1), names


def make_xy(
    d: pd.DataFrame,
    qnames: List[str],
    dnames: List[str],
    mode: str,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    if mode == "q66":
        names = list(qnames)
        X = d[names].to_numpy(float)

    elif mode == "transition":
        names = list(qnames) + list(dnames)
        X = d[names].to_numpy(float)

    elif mode == "action_conditional":
        names = list(qnames) + list(dnames)
        base = d[names].to_numpy(float)
        cats, cat_names = categorical_matrix(d)
        X = np.concatenate([base, cats], axis=1)
        names = names + cat_names

    else:
        raise ValueError(mode)

    y = d["r31a_harm_label"].to_numpy(int)

    if not np.isfinite(X).all():
        raise RuntimeError(f"Non-finite features for mode={mode}")

    return X, y, names


def fit_lr(X: np.ndarray, y: np.ndarray):
    if len(np.unique(y)) != 2:
        raise RuntimeError("Training fold has one class only.")

    model = Pipeline([
        ("scale", StandardScaler()),
        (
            "lr",
            LogisticRegression(
                C=LR_C,
                class_weight="balanced",
                max_iter=LR_MAX_ITER,
                solver="lbfgs",
                random_state=0,
            ),
        ),
    ])
    model.fit(X, y)
    return model


def safe_metrics(y: np.ndarray, score: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)

    if len(np.unique(y)) != 2:
        return {
            "auroc": math.nan,
            "auprc": math.nan,
            "brier": math.nan,
            "prevalence": float(np.mean(y)),
            "n": int(len(y)),
        }

    return {
        "auroc": float(roc_auc_score(y, score)),
        "auprc": float(average_precision_score(y, score)),
        "brier": float(brier_score_loss(y, score)),
        "prevalence": float(np.mean(y)),
        "n": int(len(y)),
    }


def append_predictions(
    rows: List[Dict[str, Any]],
    test: pd.DataFrame,
    y: np.ndarray,
    score: np.ndarray,
    seed: int,
    fold: int,
    model_name: str,
    direction: str,
):
    for row, yy, ss in zip(test.itertuples(index=False), y, score):
        rows.append({
            "split_seed": int(seed),
            "fold": int(fold),
            "model": model_name,
            "direction": direction,
            "r31a_row_id": int(row.r31a_row_id),
            "sample_id": str(row.sample_id),
            "model_state_id": str(row.model_state_id),
            "model_family": str(row.model_family),
            "action": str(row.action),
            "y_harm": int(yy),
            "score": float(ss),
        })


def run_seed(
    d: pd.DataFrame,
    qnames: List[str],
    dnames: List[str],
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    fold_map = make_group_folds(d, seed)

    work = d.copy()
    work["fold"] = work["sample_id"].astype(str).map(fold_map).astype(int)

    pred_rows: List[Dict[str, Any]] = []

    for fold in range(N_FOLDS):
        train = work[work["fold"] != fold].copy()
        test = work[work["fold"] == fold].copy()

        if set(train["sample_id"]).intersection(set(test["sample_id"])):
            raise RuntimeError("Physical-image leakage across fold.")

        pooled_specs = [
            ("SOURCE_ONLY_Q66", "q66"),
            ("SHARED_TRANSITION", "transition"),
            ("SHARED_ACTION_CONDITIONAL", "action_conditional"),
        ]

        for model_name, mode in pooled_specs:
            Xtr, ytr, _ = make_xy(train, qnames, dnames, mode)
            Xte, yte, _ = make_xy(test, qnames, dnames, mode)
            model = fit_lr(Xtr, ytr)
            score = model.predict_proba(Xte)[:, 1]
            append_predictions(
                pred_rows, test, yte, score,
                seed, fold, model_name, "POOLED"
            )

        for action in EXPECTED_ACTIONS:
            tr = train[train["action"] == action].copy()
            te = test[test["action"] == action].copy()

            Xtr, ytr, _ = make_xy(tr, qnames, dnames, "transition")
            Xte, yte, _ = make_xy(te, qnames, dnames, "transition")
            model = fit_lr(Xtr, ytr)
            score = model.predict_proba(Xte)[:, 1]

            append_predictions(
                pred_rows, te, yte, score,
                seed, fold, "SEPARATE_ACTION", action
            )

        directions = [
            ("TENT1", "PL-CONF90", "CROSS_ACTION_TENT_TO_PL"),
            ("PL-CONF90", "TENT1", "CROSS_ACTION_PL_TO_TENT"),
        ]

        for train_action, test_action, model_name in directions:
            tr = train[train["action"] == train_action].copy()
            te = test[test["action"] == test_action].copy()

            # No action-ID feature here: the target action is deliberately unseen.
            Xtr, ytr, _ = make_xy(tr, qnames, dnames, "transition")
            Xte, yte, _ = make_xy(te, qnames, dnames, "transition")

            model = fit_lr(Xtr, ytr)
            score = model.predict_proba(Xte)[:, 1]

            append_predictions(
                pred_rows, te, yte, score,
                seed, fold, model_name,
                f"{train_action}->{test_action}"
            )

    pred = pd.DataFrame(pred_rows)

    # 3 pooled models + 1 separate-action model = 4 full 18k panels,
    # plus two 9k cross-action directions = one additional 18k panel.
    expected = EXPECTED_TOTAL_ROWS * 5
    if len(pred) != expected:
        raise RuntimeError(f"prediction rows={len(pred)} expected={expected}")

    metrics = []
    for model_name, g in pred.groupby("model", sort=True):
        m = safe_metrics(g["y_harm"].to_numpy(), g["score"].to_numpy())
        metrics.append({
            "split_seed": seed,
            "model": model_name,
            "scope": "POOLED",
            "action": "ALL",
            **m,
        })

        for action, ga in g.groupby("action", sort=True):
            m = safe_metrics(ga["y_harm"].to_numpy(), ga["score"].to_numpy())
            metrics.append({
                "split_seed": seed,
                "model": model_name,
                "scope": "ACTION",
                "action": str(action),
                **m,
            })

    return pred, pd.DataFrame(metrics)


def summarize_metrics(per_seed: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (model, scope, action), g in per_seed.groupby(
        ["model", "scope", "action"],
        sort=True,
    ):
        row = {
            "model": model,
            "scope": scope,
            "action": action,
            "seeds": int(g["split_seed"].nunique()),
        }

        for metric in ["auroc", "auprc", "brier", "prevalence", "n"]:
            vals = pd.to_numeric(g[metric], errors="coerce").to_numpy(float)
            row[f"{metric}_mean"] = float(np.nanmean(vals))
            row[f"{metric}_std"] = (
                float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0
            )
            row[f"{metric}_min"] = float(np.nanmin(vals))
            row[f"{metric}_max"] = float(np.nanmax(vals))

        rows.append(row)

    return pd.DataFrame(rows)


def action_macro_auroc(summary: pd.DataFrame, model: str) -> float:
    g = summary[
        (summary["model"] == model)
        & (summary["scope"] == "ACTION")
        & (summary["action"].isin(EXPECTED_ACTIONS))
    ]
    if len(g) != 2:
        raise RuntimeError(f"Expected two action metrics for {model}; got {len(g)}")
    return float(g["auroc_mean"].mean())


def cross_aurocs(summary: pd.DataFrame) -> Dict[str, float]:
    out = {}
    for model in ["CROSS_ACTION_TENT_TO_PL", "CROSS_ACTION_PL_TO_TENT"]:
        g = summary[
            (summary["model"] == model)
            & (summary["scope"] == "POOLED")
        ]
        if len(g) != 1:
            raise RuntimeError(f"Missing pooled metric for {model}.")
        out[model] = float(g.iloc[0]["auroc_mean"])
    return out


def make_decision(summary: pd.DataFrame) -> Dict[str, Any]:
    q66_macro = action_macro_auroc(summary, "SOURCE_ONLY_Q66")
    trans_macro = action_macro_auroc(summary, "SHARED_TRANSITION")
    shared_macro = action_macro_auroc(summary, "SHARED_ACTION_CONDITIONAL")
    separate_macro = action_macro_auroc(summary, "SEPARATE_ACTION")

    cross = cross_aurocs(summary)
    vals = list(cross.values())
    cross_macro = float(np.mean(vals))
    cross_min = float(np.min(vals))
    shared_gap = float(shared_macro - separate_macro)

    strong = (
        cross_macro >= STRONG_CROSS_MACRO
        and cross_min >= STRONG_CROSS_MIN_DIRECTION
        and shared_gap >= STRONG_SHARED_GAP
    )
    go = (
        cross_macro >= GO_CROSS_MACRO
        and cross_min >= GO_CROSS_MIN_DIRECTION
        and shared_gap >= GO_SHARED_GAP
    )

    if strong:
        decision = "GO_STRONG_TO_R31B_THIRD_ACTION"
    elif go:
        decision = "GO_TO_R31B_THIRD_ACTION"
    else:
        decision = "NO_GO_UNIFIED_ACTION_GENERALIZATION"

    return {
        "decision": decision,
        "scope_note": (
            "Only two actions exist in R31A. GO means proceed to a genuinely "
            "third action; it is not final unseen-action proof."
        ),
        "metrics": {
            "source_only_q66_macro_action_auroc": q66_macro,
            "shared_transition_macro_action_auroc": trans_macro,
            "shared_action_conditional_macro_action_auroc": shared_macro,
            "separate_action_macro_action_auroc": separate_macro,
            "shared_minus_separate_macro_action_auroc": shared_gap,
            "cross_action_macro_auroc": cross_macro,
            "cross_action_min_direction_auroc": cross_min,
            **cross,
        },
        "go_rule": {
            "cross_action_macro_auroc_at_least": GO_CROSS_MACRO,
            "each_cross_direction_auroc_at_least": GO_CROSS_MIN_DIRECTION,
            "shared_minus_separate_macro_auroc_at_least": GO_SHARED_GAP,
        },
        "strong_go_rule": {
            "cross_action_macro_auroc_at_least": STRONG_CROSS_MACRO,
            "each_cross_direction_auroc_at_least": STRONG_CROSS_MIN_DIRECTION,
            "shared_minus_separate_macro_auroc_at_least": STRONG_SHARED_GAP,
        },
        "next_if_go": (
            "R31B: add a genuinely third frozen TTA action and perform "
            "leave-one-action-out safety prediction with an explicit "
            "action descriptor plus risk-controlled commit/rollback."
        ),
        "next_if_no_go": (
            "Do not force an action-generalization claim. Keep R30 "
            "transactional safety action-specific and analyze why "
            "semantic transitions fail to transfer."
        ),
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--source-table",
        type=Path,
        default=None,
        help=(
            "Optional explicit frozen NeoPolyp source action table. "
            "Expected: 18,000 rows = 1000 cases x 9 states x 2 actions."
        ),
    )
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    print("=" * 120)
    print("SafeTTA R31A Unified Action-Conditional HARM Predictor")
    print("Version:", VERSION)
    print("External cohort access       : FORBIDDEN")
    print("EndoTect access              : NO")
    print("Feature extraction/inference : NO")
    print("Uses frozen source features  : YES")
    print("Hyperparameter search        : NO")
    print("Grouped physical-image CV    : YES")
    print("Cross-action case leakage    : NO")
    print("=" * 120)

    if args.source_table is not None and path_is_external(args.source_table):
        raise RuntimeError("Explicit source table points to forbidden external data.")

    require_exact_sha(R30A0_SCRIPT, EXPECTED_R30A0_SHA, "R30A0 frozen schema")
    require_exact_sha(R30A3_SCRIPT, EXPECTED_R30A3_SCRIPT_SHA, "R30A3 script")
    require_exact_sha(R30A3_FINAL, EXPECTED_R30A3_FINAL_SHA, "R30A3 final lock")

    a0 = import_module(R30A0_SCRIPT, "r31a_r30a0")
    r30a3 = import_module(R30A3_SCRIPT, "r31a_r30a3")

    if not hasattr(a0, "QSOURCE") or not hasattr(a0, "DSEM"):
        raise RuntimeError("R30A0 missing QSOURCE/DSEM.")

    qnames = list(a0.QSOURCE)
    dnames = list(a0.DSEM)

    if len(qnames) != 66 or len(dnames) != 64:
        raise RuntimeError(
            f"Frozen feature schema drift QSOURCE={len(qnames)} DSEM={len(dnames)}"
        )

    source_path, raw, discovery = resolve_source_table(
        args.source_table, r30a3, qnames, dnames
    )

    if path_is_external(source_path):
        raise RuntimeError(f"Resolved forbidden external table: {source_path}")

    validate_source_table_structure(raw, qnames, dnames, soft=False)
    d = add_exact_labels_and_ids(raw, qnames)
    source_sha = sha256_file(source_path)

    print("\nFROZEN SOURCE-DEVELOPMENT TABLE")
    print("  path   :", source_path)
    print("  SHA    :", source_sha)
    print("  rows   :", len(d))
    print("  cases  :", d["sample_id"].astype(str).nunique())
    print("  states :", d["model_state_id"].astype(str).nunique())
    print("  actions:", d["action"].value_counts().to_dict())
    print(
        "  HARM   :",
        d.groupby("action")["r31a_harm_label"]
        .agg(["sum", "mean"])
        .to_dict("index"),
    )
    print("  label source:", d.attrs.get("harm_label_source"))

    out = args.output_dir.resolve()
    if out.exists():
        raise RuntimeError(
            f"Output exists; do not overwrite prior run: {out}"
        )
    out.mkdir(parents=True, exist_ok=False)

    lineage = {
        "version": VERSION,
        "source_table": {
            "path": str(source_path),
            "sha256": source_sha,
            "rows": int(len(d)),
            "physical_cases": int(d["sample_id"].astype(str).nunique()),
            "model_states": int(d["model_state_id"].astype(str).nunique()),
            "actions": d["action"].value_counts().to_dict(),
            "harm_label_source": d.attrs.get("harm_label_source"),
        },
        "frozen_schema": {
            "R30A0_script": str(R30A0_SCRIPT),
            "R30A0_sha256": EXPECTED_R30A0_SHA,
            "q_source_dim": len(qnames),
            "delta_semantic_dim": len(dnames),
            "q_source_names": qnames,
            "delta_semantic_names": dnames,
        },
        "information_boundary": {
            "external_cohort_access": False,
            "endotect_access": False,
            "target_outcomes_used": False,
            "feature_reextraction": False,
            "segmentation_inference": False,
            "hyperparameter_search": False,
            "source_development_labels_used": True,
        },
        "discovery": discovery,
    }
    atomic_write_json(out / "R31A_SOURCE_TABLE_LINEAGE.json", lineage)

    pred_all = []
    metric_all = []

    for seed in tqdm(
        CV_SEEDS,
        desc="R31A grouped-CV seeds",
        unit="seed",
        dynamic_ncols=True,
    ):
        pred, metrics = run_seed(d, qnames, dnames, seed)
        pred_all.append(pred)
        metric_all.append(metrics)

    predictions = pd.concat(pred_all, ignore_index=True)
    per_seed = pd.concat(metric_all, ignore_index=True)
    summary = summarize_metrics(per_seed)
    decision = make_decision(summary)

    pred_path = out / "R31A_OOF_PREDICTIONS.csv"
    per_seed_path = out / "R31A_PER_SEED_METRICS.csv"
    summary_path = out / "R31A_METRIC_SUMMARY.csv"
    decision_path = out / "R31A_DECISION.json"

    predictions.to_csv(pred_path, index=False)
    per_seed.to_csv(per_seed_path, index=False)
    summary.to_csv(summary_path, index=False)

    decision.update({
        "version": VERSION,
        "source_table_sha256": source_sha,
        "cv": {
            "physical_group": "sample_id",
            "folds": N_FOLDS,
            "seeds": CV_SEEDS,
            "no_case_overlap": True,
            "cross_action_train_test_case_overlap": False,
        },
        "classifier": {
            "name": "balanced LogisticRegression",
            "C": LR_C,
            "max_iter": LR_MAX_ITER,
            "scaling": "StandardScaler fitted in training fold only",
            "hyperparameter_search": False,
        },
        "artifacts": {
            "predictions": str(pred_path),
            "per_seed_metrics": str(per_seed_path),
            "metric_summary": str(summary_path),
        },
    })
    atomic_write_json(decision_path, decision)

    inventory = make_inventory(out)
    inventory_path = out / "R31A_SHA256_INVENTORY.csv"
    inventory.to_csv(inventory_path, index=False)

    final = {
        "status": "PASS_R31A_ACTION_CONDITIONAL_HARM_FEASIBILITY_COMPLETE",
        "version": VERSION,
        "decision": decision["decision"],
        "source_table_sha256": source_sha,
        "output": str(out),
        "decision_json_sha256": sha256_file(decision_path),
        "metric_summary_sha256": sha256_file(summary_path),
        "predictions_sha256": sha256_file(pred_path),
        "external_data_access": False,
        "next": (
            "R31B_THIRD_ACTION_LEAVE_ONE_ACTION_OUT"
            if decision["decision"].startswith("GO")
            else "STOP_ACTION_GENERALIZATION_AND_INTERPRET"
        ),
    }

    final_path = out / "R31A_FINAL_LOCK.json"
    atomic_write_json(final_path, final)

    print("\n" + "=" * 120)
    print("R31A KEY RESULTS")
    print("=" * 120)

    show = summary[
        summary["model"].isin([
            "SOURCE_ONLY_Q66",
            "SHARED_TRANSITION",
            "SHARED_ACTION_CONDITIONAL",
            "SEPARATE_ACTION",
            "CROSS_ACTION_TENT_TO_PL",
            "CROSS_ACTION_PL_TO_TENT",
        ])
    ][
        [
            "model", "scope", "action",
            "auroc_mean", "auroc_std",
            "auprc_mean", "auprc_std",
            "brier_mean",
        ]
    ]
    print(show.to_string(index=False))

    print("\nR31A DECISION:", decision["decision"])
    print(
        "  shared-action conditional macro AUROC :",
        decision["metrics"]["shared_action_conditional_macro_action_auroc"],
    )
    print(
        "  separate-action macro AUROC           :",
        decision["metrics"]["separate_action_macro_action_auroc"],
    )
    print(
        "  shared - separate AUROC               :",
        decision["metrics"]["shared_minus_separate_macro_action_auroc"],
    )
    print(
        "  cross-action macro AUROC              :",
        decision["metrics"]["cross_action_macro_auroc"],
    )
    print(
        "  cross-action min direction AUROC      :",
        decision["metrics"]["cross_action_min_direction_auroc"],
    )

    print("\nFINAL STATUS : PASS_R31A_ACTION_CONDITIONAL_HARM_FEASIBILITY_COMPLETE")
    print("External cohort access: NO")
    print("EndoTect access        : NO")
    print("Output                 :", out)
    print("Final lock SHA256      :", sha256_file(final_path))
    print("NEXT                   :", final["next"])
    print("=" * 120)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
