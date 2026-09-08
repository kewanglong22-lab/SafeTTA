#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q1_R16A2_crossfitted_flexible_source_quality_control_v1.py

SafeTTA R16-A2 robustness audit
================================

Purpose
-------
R16-A showed that frozen SafeTTA risk adds information beyond a *linear*
SOURCE-Dice covariate on PolypGen. R16-A2 is a stricter post-hoc robustness
audit designed to address two possible reviewer objections:

1) the relationship between current SOURCE Dice and future adaptation HARM may
   be nonlinear;
2) the R16-A descriptive AUROC comparison was fitted/evaluated on the same
   external rows.

R16-A2 therefore uses:
- the exact R16-A fix2 aligned PolypGen analysis table;
- 5-fold physical-image cross-fitting;
- a flexible cubic spline for SOURCE Dice fitted *within each training fold*;
- fixed architecture-family effects;
- a frozen SafeTTA risk covariate added only in the second model;
- paired physical-image clustered bootstrap on the fixed out-of-fold
  predictions.

Models
------
M_quality_flex:
    HARM ~ spline(SOURCE_Dice) + architecture_family

M_quality_flex+risk:
    HARM ~ spline(SOURCE_Dice) + architecture_family + z(SafeTTA_risk)

Important interpretation boundary
---------------------------------
This is a POST-REVEAL explanatory/diagnostic analysis. It is NOT a deployable
target-side calibration model. SOURCE Dice and HARM labels are used only to
test whether the already frozen SafeTTA score carries information not reducible
to current segmentation quality.

No SafeTTA representation/head/threshold is retrained, recalibrated, or changed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.preprocessing import SplineTransformer

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


VERSION = "2026-09-08-Q1-R16A2-CROSSFITTED-FLEXIBLE-SOURCE-QUALITY-CONTROL-v1"

DEFAULT_ROOT = Path(r"F:\MEDSEG_SAFETTA")
DEFAULT_INPUT = DEFAULT_ROOT / r"outputs\Q1_R16A_source_quality_control_polypgen_v1_fix2\R16A_ANALYSIS_TABLE.csv"
DEFAULT_OUT = DEFAULT_ROOT / r"outputs\Q1_R16A2_crossfitted_flexible_source_quality_control_v1"

EXPECTED_ROWS = 13788
EXPECTED_PHYSICAL = 1532
EXPECTED_FAMILIES = 3
EXPECTED_ROWS_PER_PHYSICAL = 9

N_SPLITS = 5
SPLIT_SEED = 20260908
N_KNOTS = 5
SPLINE_DEGREE = 3
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 20260909


def validate_input(df: pd.DataFrame) -> None:
    required = {
        "physical_id", "family", "source_dice", "risk", "harm",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    if len(df) != EXPECTED_ROWS:
        raise RuntimeError(f"Rows {len(df)} != {EXPECTED_ROWS}")

    df["physical_id"] = df["physical_id"].astype(str)
    if df["physical_id"].nunique() != EXPECTED_PHYSICAL:
        raise RuntimeError("Physical-image count != 1532")

    vc = df["physical_id"].value_counts()
    if int(vc.min()) != EXPECTED_ROWS_PER_PHYSICAL or int(vc.max()) != EXPECTED_ROWS_PER_PHYSICAL:
        raise RuntimeError("Each PolypGen physical image must have exactly 9 frozen model-state rows")

    if df["family"].astype(str).nunique() != EXPECTED_FAMILIES:
        raise RuntimeError("Architecture-family count != 3")

    for c in ["source_dice", "risk"]:
        a = pd.to_numeric(df[c], errors="coerce").to_numpy(float)
        if not np.isfinite(a).all():
            raise RuntimeError(f"Non-finite values in {c}")
        if a.min() < -1e-6 or a.max() > 1.000001:
            raise RuntimeError(f"{c} outside [0,1]")

    y = pd.to_numeric(df["harm"], errors="coerce")
    if y.isna().any() or set(y.astype(int).unique().tolist()) != {0, 1}:
        raise RuntimeError("HARM must be binary and contain both classes")


def assign_group_folds(physical_ids: Sequence[str], n_splits: int, seed: int) -> Dict[str, int]:
    ids = np.array(sorted(set(map(str, physical_ids))), dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    fold_map = {}
    for i, pid in enumerate(ids):
        fold_map[str(pid)] = int(i % n_splits)
    return fold_map


def family_matrix(series: pd.Series, levels: Sequence[str]) -> np.ndarray:
    cat = pd.Categorical(series.astype(str), categories=list(levels))
    d = pd.get_dummies(cat, drop_first=True, dtype=float)
    return d.to_numpy(float)


def fit_unpenalized_lr(X: np.ndarray, y: np.ndarray) -> LogisticRegression:
    model = LogisticRegression(
        penalty=None,
        solver="lbfgs",
        max_iter=5000,
        fit_intercept=True,
    )
    model.fit(X, y)
    return model


def build_train_test_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    family_levels: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    # Flexible SOURCE-Dice transform: fit only on training-fold SOURCE Dice.
    spline = SplineTransformer(
        n_knots=N_KNOTS,
        degree=SPLINE_DEGREE,
        knots="quantile",
        include_bias=False,
    )
    q_tr = train[["source_dice"]].to_numpy(float)
    q_te = test[["source_dice"]].to_numpy(float)
    s_tr = spline.fit_transform(q_tr)
    s_te = spline.transform(q_te)

    f_tr = family_matrix(train["family"], family_levels)
    f_te = family_matrix(test["family"], family_levels)

    X0_tr = np.concatenate([s_tr, f_tr], axis=1)
    X0_te = np.concatenate([s_te, f_te], axis=1)

    # Risk scaling fit on training fold only.
    r_mu = float(train["risk"].mean())
    r_sd = float(train["risk"].std(ddof=0))
    if not np.isfinite(r_sd) or r_sd <= 0:
        raise RuntimeError("Training-fold risk has zero/invalid variance")

    r_tr = ((train["risk"].to_numpy(float) - r_mu) / r_sd).reshape(-1, 1)
    r_te = ((test["risk"].to_numpy(float) - r_mu) / r_sd).reshape(-1, 1)

    X1_tr = np.concatenate([X0_tr, r_tr], axis=1)
    X1_te = np.concatenate([X0_te, r_te], axis=1)

    meta = {
        "risk_train_mean": r_mu,
        "risk_train_sd": r_sd,
        "spline_n_features": int(s_tr.shape[1]),
        "family_dummy_features": int(f_tr.shape[1]),
        "quality_model_features": int(X0_tr.shape[1]),
        "quality_plus_risk_features": int(X1_tr.shape[1]),
    }
    return X0_tr, X0_te, X1_tr, X1_te, meta


def crossfit(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict[str, object]]]:
    family_levels = sorted(df["family"].astype(str).unique().tolist())
    fold_map = assign_group_folds(df["physical_id"], N_SPLITS, SPLIT_SEED)

    z = df.copy()
    z["fold"] = z["physical_id"].astype(str).map(fold_map)
    if z["fold"].isna().any():
        raise RuntimeError("Fold assignment missing for at least one physical image")

    outputs = []
    fold_meta = []

    iterator = range(N_SPLITS)
    if tqdm is not None:
        iterator = tqdm(iterator, desc="R16-A2 physical-image cross-fitting", unit="fold", dynamic_ncols=True)

    for fold in iterator:
        tr = z[z["fold"] != fold].copy()
        te = z[z["fold"] == fold].copy()

        train_ids = set(tr["physical_id"].astype(str))
        test_ids = set(te["physical_id"].astype(str))
        if train_ids & test_ids:
            raise RuntimeError(f"Physical-image leakage in fold {fold}")

        y_tr = tr["harm"].to_numpy(int)
        y_te = te["harm"].to_numpy(int)

        if len(np.unique(y_tr)) != 2 or len(np.unique(y_te)) != 2:
            raise RuntimeError(f"Fold {fold} lacks both HARM classes")

        X0_tr, X0_te, X1_tr, X1_te, meta = build_train_test_features(
            tr, te, family_levels
        )
        m0 = fit_unpenalized_lr(X0_tr, y_tr)
        m1 = fit_unpenalized_lr(X1_tr, y_tr)

        p0 = m0.predict_proba(X0_te)[:, 1]
        p1 = m1.predict_proba(X1_te)[:, 1]

        # Last coefficient corresponds to z(risk), since risk is appended last.
        risk_coef = float(m1.coef_.reshape(-1)[-1])

        part = te[[
            "physical_id", "family", "source_dice", "risk", "harm"
        ]].copy()
        part["fold"] = int(fold)
        part["p_quality_flex"] = p0
        part["p_quality_flex_plus_risk"] = p1
        outputs.append(part)

        fold_meta.append({
            "fold": int(fold),
            "n_train_rows": int(len(tr)),
            "n_test_rows": int(len(te)),
            "n_train_physical": int(tr["physical_id"].nunique()),
            "n_test_physical": int(te["physical_id"].nunique()),
            "train_harm_prevalence": float(y_tr.mean()),
            "test_harm_prevalence": float(y_te.mean()),
            "risk_coefficient": risk_coef,
            **meta,
        })

    oof = pd.concat(outputs, ignore_index=True)
    if len(oof) != len(df):
        raise RuntimeError("OOF row count mismatch")
    if oof[["physical_id", "family", "source_dice", "risk", "harm"]].duplicated().all():
        # This guard is intentionally conservative; duplicate scientific rows may
        # exist, but all rows being duplicates would indicate a severe failure.
        raise RuntimeError("Unexpected duplicated OOF construction")

    return oof, fold_meta


def auc(y: np.ndarray, s: np.ndarray) -> float:
    return float(roc_auc_score(y, s))


def auprc(y: np.ndarray, s: np.ndarray) -> float:
    return float(average_precision_score(y, s))


def metrics(oof: pd.DataFrame) -> Dict[str, float]:
    y = oof["harm"].to_numpy(int)
    p0 = oof["p_quality_flex"].to_numpy(float)
    p1 = oof["p_quality_flex_plus_risk"].to_numpy(float)

    return {
        "auroc_quality_flex": auc(y, p0),
        "auroc_quality_flex_plus_risk": auc(y, p1),
        "delta_auroc": auc(y, p1) - auc(y, p0),
        "auprc_quality_flex": auprc(y, p0),
        "auprc_quality_flex_plus_risk": auprc(y, p1),
        "delta_auprc": auprc(y, p1) - auprc(y, p0),
        "logloss_quality_flex": float(log_loss(y, p0)),
        "logloss_quality_flex_plus_risk": float(log_loss(y, p1)),
        "delta_logloss_plus_risk_minus_quality": float(log_loss(y, p1) - log_loss(y, p0)),
    }


def paired_cluster_bootstrap(oof: pd.DataFrame, reps: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ids = oof["physical_id"].astype(str).drop_duplicates().to_numpy()
    groups = {
        pid: g.copy()
        for pid, g in oof.groupby(oof["physical_id"].astype(str), sort=False)
    }

    rows = []
    iterator = range(reps)
    if tqdm is not None:
        iterator = tqdm(
            iterator,
            total=reps,
            desc="R16-A2 paired physical-image bootstrap",
            unit="rep",
            dynamic_ncols=True,
        )

    for rep in iterator:
        sampled = rng.choice(ids, size=len(ids), replace=True)
        b = pd.concat([groups[pid] for pid in sampled], ignore_index=True)
        y = b["harm"].to_numpy(int)
        if len(np.unique(y)) != 2:
            continue
        p0 = b["p_quality_flex"].to_numpy(float)
        p1 = b["p_quality_flex_plus_risk"].to_numpy(float)
        rows.append({
            "rep": int(rep),
            "delta_auroc": auc(y, p1) - auc(y, p0),
            "delta_auprc": auprc(y, p1) - auprc(y, p0),
            "delta_logloss_plus_risk_minus_quality": float(log_loss(y, p1) - log_loss(y, p0)),
        })

    return pd.DataFrame(rows)


def ci(s: pd.Series) -> Tuple[float, float]:
    a = pd.to_numeric(s, errors="coerce").dropna().to_numpy(float)
    if not len(a):
        return float("nan"), float("nan")
    q = np.percentile(a, [2.5, 97.5])
    return float(q[0]), float(q[1])


def family_metrics(oof: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family, g in oof.groupby("family", sort=True):
        y = g["harm"].to_numpy(int)
        p0 = g["p_quality_flex"].to_numpy(float)
        p1 = g["p_quality_flex_plus_risk"].to_numpy(float)
        rows.append({
            "family": str(family),
            "n_rows": int(len(g)),
            "harm_prevalence": float(y.mean()),
            "quality_flex_auroc": auc(y, p0),
            "quality_flex_plus_risk_auroc": auc(y, p1),
            "delta_auroc": auc(y, p1) - auc(y, p0),
            "quality_flex_auprc": auprc(y, p0),
            "quality_flex_plus_risk_auprc": auprc(y, p1),
            "delta_auprc": auprc(y, p1) - auprc(y, p0),
        })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--bootstrap-reps", type=int, default=BOOTSTRAP_REPS)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("===== SAFETTA R16-A2 FLEXIBLE CROSS-FITTED QUALITY CONTROL =====")
    print("Version:", VERSION)
    print("Input:", args.input)
    print("Post-reveal explanatory audit: YES")
    print("SafeTTA score refit/recalibration: NO")
    print("SOURCE Dice deployable input: NO")
    print("Physical-image cross-fitting:", f"{N_SPLITS}-fold")
    print("Flexible quality control:", f"cubic spline, n_knots={N_KNOTS}")
    print("Bootstrap reps:", args.bootstrap_reps)
    print()

    if not args.input.exists():
        raise FileNotFoundError(args.input)

    df = pd.read_csv(args.input, low_memory=False)
    validate_input(df)

    # Keep a clean exact table.
    df = df[["physical_id", "family", "source_dice", "risk", "harm"]].copy()
    df["physical_id"] = df["physical_id"].astype(str)
    df["family"] = df["family"].astype(str)
    df["source_dice"] = pd.to_numeric(df["source_dice"], errors="raise")
    df["risk"] = pd.to_numeric(df["risk"], errors="raise")
    df["harm"] = pd.to_numeric(df["harm"], errors="raise").astype(int)

    oof, fold_meta = crossfit(df)
    main_metrics = metrics(oof)
    fam = family_metrics(oof)
    boot = paired_cluster_bootstrap(oof, args.bootstrap_reps, BOOTSTRAP_SEED)

    da_ci = ci(boot["delta_auroc"]) if len(boot) else (np.nan, np.nan)
    dp_ci = ci(boot["delta_auprc"]) if len(boot) else (np.nan, np.nan)
    dll_ci = ci(boot["delta_logloss_plus_risk_minus_quality"]) if len(boot) else (np.nan, np.nan)

    oof.to_csv(args.out_dir / "R16A2_OOF_PREDICTIONS.csv", index=False)
    pd.DataFrame(fold_meta).to_csv(args.out_dir / "R16A2_FOLD_AUDIT.csv", index=False)
    fam.to_csv(args.out_dir / "R16A2_FAMILY_METRICS.csv", index=False)
    boot.to_csv(args.out_dir / "R16A2_CLUSTER_BOOTSTRAP.csv", index=False)

    supportive_auc = np.isfinite(da_ci[0]) and da_ci[0] > 0
    supportive_ap = np.isfinite(dp_ci[0]) and dp_ci[0] > 0
    supportive_logloss = np.isfinite(dll_ci[1]) and dll_ci[1] < 0

    if supportive_auc or supportive_ap or supportive_logloss:
        gate = "PASS_R16A2_FROZEN_RISK_ADDS_CROSSFITTED_INFORMATION_BEYOND_FLEXIBLE_SOURCE_QUALITY"
    else:
        gate = "R16A2_NO_SUPPORTED_INCREMENT_BEYOND_FLEXIBLE_SOURCE_QUALITY"

    summary = {
        "version": VERSION,
        "input": str(args.input),
        "rows": int(len(df)),
        "physical_images": int(df["physical_id"].nunique()),
        "n_splits": N_SPLITS,
        "split_seed": SPLIT_SEED,
        "spline_n_knots": N_KNOTS,
        "spline_degree": SPLINE_DEGREE,
        **main_metrics,
        "delta_auroc_ci95": list(da_ci),
        "delta_auprc_ci95": list(dp_ci),
        "delta_logloss_ci95": list(dll_ci),
        "bootstrap_requested": int(args.bootstrap_reps),
        "bootstrap_completed": int(len(boot)),
        "gate": gate,
        "interpretation_boundary": "post-reveal cross-fitted explanatory audit; not a deployable target-side calibration model",
    }
    (args.out_dir / "R16A2_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    report = [
        "===== SAFETTA R16-A2 FLEXIBLE CROSS-FITTED QUALITY CONTROL SUMMARY =====",
        f"Version: {VERSION}",
        f"Rows: {len(df)}",
        f"Physical images: {df['physical_id'].nunique()}",
        f"HARM prevalence: {df['harm'].mean():.6f}",
        "",
        f"Flexible quality-only OOF AUROC={main_metrics['auroc_quality_flex']:.6f}",
        f"Flexible quality+risk OOF AUROC={main_metrics['auroc_quality_flex_plus_risk']:.6f}",
        f"Delta AUROC={main_metrics['delta_auroc']:.6f} CI95=[{da_ci[0]:.6f}, {da_ci[1]:.6f}]",
        "",
        f"Flexible quality-only OOF AUPRC={main_metrics['auprc_quality_flex']:.6f}",
        f"Flexible quality+risk OOF AUPRC={main_metrics['auprc_quality_flex_plus_risk']:.6f}",
        f"Delta AUPRC={main_metrics['delta_auprc']:.6f} CI95=[{dp_ci[0]:.6f}, {dp_ci[1]:.6f}]",
        "",
        f"Quality-only OOF log loss={main_metrics['logloss_quality_flex']:.6f}",
        f"Quality+risk OOF log loss={main_metrics['logloss_quality_flex_plus_risk']:.6f}",
        f"Delta log loss (risk-quality)={main_metrics['delta_logloss_plus_risk_minus_quality']:.6f} CI95=[{dll_ci[0]:.6f}, {dll_ci[1]:.6f}]",
        "",
        f"Bootstrap completed: {len(boot)}/{args.bootstrap_reps}",
        "CI scope: physical-image uncertainty conditional on frozen model panel.",
        "All reported AUROC/AUPRC values are grouped out-of-fold, not in-sample.",
        "SOURCE Dice is post-reveal explanatory only and never a deployable SafeTTA input.",
        "",
        f"GATE={gate}",
    ]

    (args.out_dir / "R16A2_REPORT.txt").write_text("\n".join(report) + "\n", encoding="utf-8")

    print("\n".join(report))
    print("\nFAMILY METRICS:")
    print(fam.to_string(index=False))
    print("\nFOLD AUDIT:")
    print(pd.DataFrame(fold_meta).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
