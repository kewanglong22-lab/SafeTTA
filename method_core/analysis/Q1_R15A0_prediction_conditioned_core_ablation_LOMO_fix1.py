#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R15A0_prediction_conditioned_core_ablation_LOMO_fix1.py

Post-freeze explanatory source-side ablation for the already frozen SafeTTA
representation.  This experiment DOES NOT reopen method selection.

Scientific question
-------------------
Which part of the frozen prediction-conditioned representation contributes to
source-side cross-model-family harm ranking?

Existing historical controls are reused from the frozen R10K2B prediction file:
  1) M2_backbone
  2) ImageCLS_PCA64
  3) CondDINO_PCA64               = FG + BG
  4) M2_plus_CondDINO_PCA64       = frozen final method

Only the missing component ablations are newly trained:
  5) FG_PCA64
  6) BG_PCA64
  7) M2_plus_FG_PCA64
  8) M2_plus_BG_PCA64

The exact historical protocol is preserved:
- 3 held-out model families;
- 5 split seeds;
- outer 5-fold StratifiedGroupKFold by sample_id;
- inner 4-fold grouped OOF threshold selection;
- target Recall goal = 0.90 on source-only inner OOF;
- PCA64 randomized + whiten, fit on train only;
- median imputer + StandardScaler + balanced LogisticRegression;
- target-family labels used for evaluation only;
- no external cohort is accessed;
- no final method/threshold is changed from this analysis.

The newly generated prediction rows are merged with the exact historical
R10K2B prediction rows so R15A1 can perform a fully paired case-clustered
bootstrap across all 8 ablation methods.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


VERSION = "2026-09-04-Q1-R15A0-v1-fix1"
BUILD = "Q1_R15A0_PREDICTION_CONDITIONED_CORE_ABLATION_LOMO_FIX1"

ROOT = Path(r"F:\MEDSEG_SAFETTA")
OUT = ROOT / "outputs"

SEEDS = [20260816, 20260817, 20260818, 20260819, 20260820]
OUTER_FOLDS = 5
INNER_FOLDS = 4
TARGET_RECALL = 0.90
PPV_PREVALENCE = 0.01
PCA_DIM = 64

FAMILIES = [
    "DeepLabV3-R50",
    "PraNet",
    "SegFormer-B0",
]

M2 = [
    "morph_fg_fraction",
    "morph_boundary_density",
]

HISTORICAL_METHODS = [
    "M2_backbone",
    "ImageCLS_PCA64",
    "CondDINO_PCA64",
    "M2_plus_CondDINO_PCA64",
]

NEW_METHODS = [
    "FG_PCA64",
    "BG_PCA64",
    "M2_plus_FG_PCA64",
    "M2_plus_BG_PCA64",
]

ALL_METHODS = HISTORICAL_METHODS + NEW_METHODS

FINAL_METHOD = "M2_plus_CondDINO_PCA64"

R10K2A_DIR = (
    OUT
    / "Q1_R10K2A_dinov2_image_mask_representation_lock_fix1_v1"
)
IMAGE_FEATURES = R10K2A_DIR / "R10K2A_image_cls_features.npz"
STATE_FEATURES = R10K2A_DIR / "R10K2A_mask_conditioned_dinov2_features.npz"
STATE_METADATA = R10K2A_DIR / "R10K2A_state_metadata_no_labels.csv"

IMAGE_MANIFEST = (
    OUT
    / "Q1_R05D1_neopolyp_acquisition_pairing_duplicate_qc_v1_fix2"
    / "frozen_confirmatory_manifest.csv"
)

MORPH = (
    OUT
    / "Q1_R10J1_source_mask_morphology_feature_builder_fix1_v1"
    / "source_mask_morphology_features_labeled.csv"
)

R10K2B_DIR = (
    OUT
    / "Q1_R10K2B_dinov2_image_mask_LOMO_evaluation_fix2_v1"
)
HISTORICAL_PREDICTIONS = R10K2B_DIR / "R10K2B_target_predictions.csv"

DEFAULT_OUTPUT = (
    OUT
    / "Q1_R15A0_prediction_conditioned_core_ablation_LOMO_fix1_v1"
)

# Historical macro point-estimate sentinels from the frozen R10K2B result.
# These are audit tolerances, not targets for the new ablation methods.
HISTORICAL_AUROC_SENTINELS = {
    "M2_backbone": 0.805745,
    "ImageCLS_PCA64": 0.728028,
    "CondDINO_PCA64": 0.825508,
    "M2_plus_CondDINO_PCA64": 0.826563,
}
HISTORICAL_AUPRC_SENTINELS = {
    "M2_backbone": 0.656118,
    "ImageCLS_PCA64": 0.558977,
    "CondDINO_PCA64": 0.708001,
    "M2_plus_CondDINO_PCA64": 0.709391,
}
SENTINEL_TOL = 0.002

DECISION = "CORE_ABLATION_COMPLETE_READY_FOR_R15A1_PAIRED_CLUSTER_BOOTSTRAP"


def sha256_file(path: Path, chunk=8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def norm_sid_series(s):
    return (
        s.astype(str)
        .str.strip()
        .str.replace("\\", "/", regex=False)
        .str.lower()
    )


def build_lr(seed):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=seed,
        )),
    ])


def tabular_matrix(df, cols):
    return np.column_stack([
        pd.to_numeric(df[c], errors="coerce").to_numpy(float)
        for c in cols
    ])


def threshold_for_recall(y, p, target):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)

    pos = np.sort(p[y == 1])[::-1]
    if len(pos) == 0:
        raise RuntimeError("No positive samples for threshold selection.")

    k = int(np.ceil(target * len(pos)))
    k = min(max(k, 1), len(pos))
    return float(pos[k - 1])


def operating_metrics(y, p, threshold):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    pred = (p >= threshold).astype(int)

    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))

    recall = tp / (tp + fn) if (tp + fn) else np.nan
    fpr = fp / (fp + tn) if (fp + tn) else np.nan
    empirical_ppv = tp / (tp + fp) if (tp + fp) else np.nan

    pi = PPV_PREVALENCE
    denom = recall * pi + fpr * (1.0 - pi)
    ppv1 = recall * pi / denom if denom > 0 else np.nan

    return {
        "Recall": float(recall),
        "FPR": float(fpr),
        "EmpiricalPPV": float(empirical_ppv),
        "PPV1pct": float(ppv1),
    }


def fit_pca(train, eval_, seed):
    pca = PCA(
        n_components=PCA_DIM,
        svd_solver="randomized",
        whiten=True,
        random_state=seed,
    )
    ztr = pca.fit_transform(train)
    zev = pca.transform(eval_)
    return ztr, zev, float(pca.explained_variance_ratio_.sum())


def build_new_method_matrix(method, df_train, df_eval, fg_train, fg_eval, bg_train, bg_eval, seed):
    """Only the four missing component ablations are implemented here."""
    if method in {"FG_PCA64", "M2_plus_FG_PCA64"}:
        ztr, zev, evr = fit_pca(fg_train, fg_eval, seed)
    elif method in {"BG_PCA64", "M2_plus_BG_PCA64"}:
        ztr, zev, evr = fit_pca(bg_train, bg_eval, seed)
    else:
        raise ValueError(method)

    if method.startswith("M2_plus_"):
        m2tr = tabular_matrix(df_train, M2)
        m2ev = tabular_matrix(df_eval, M2)
        ztr = np.column_stack([m2tr, ztr])
        zev = np.column_stack([m2ev, zev])

    return ztr, zev, evr


def inner_oof_new_method(method, df_outer_train, y, groups, fg_outer_train, bg_outer_train, seed):
    cv = StratifiedGroupKFold(
        n_splits=INNER_FOLDS,
        shuffle=True,
        random_state=seed,
    )

    p = np.full(len(y), np.nan, dtype=float)

    for fold, (tr, va) in enumerate(
        cv.split(np.zeros((len(y), 1)), y, groups)
    ):
        if set(groups[tr]) & set(groups[va]):
            raise RuntimeError("Inner group leakage.")

        Xtr, Xva, _ = build_new_method_matrix(
            method=method,
            df_train=df_outer_train.iloc[tr],
            df_eval=df_outer_train.iloc[va],
            fg_train=fg_outer_train[tr],
            fg_eval=fg_outer_train[va],
            bg_train=bg_outer_train[tr],
            bg_eval=bg_outer_train[va],
            seed=seed + fold,
        )

        clf = build_lr(seed + 100 + fold)
        clf.fit(Xtr, y[tr])
        p[va] = clf.predict_proba(Xva)[:, 1]

    if np.isnan(p).any():
        raise RuntimeError("Incomplete inner OOF probabilities.")

    return p


def load_panel_and_components():
    # ---------- numeric image CLS ----------
    with np.load(IMAGE_FEATURES, allow_pickle=False) as z:
        if "cls" not in z.files:
            raise RuntimeError("R10K2A image CLS numeric key missing.")
        cls = np.asarray(z["cls"], dtype=np.float32)
    if cls.shape != (1000, 768):
        raise RuntimeError(f"Unexpected CLS shape: {cls.shape}")

    image_manifest = pd.read_csv(IMAGE_MANIFEST, low_memory=False)
    if not {"sample_id", "image_path"}.issubset(image_manifest.columns):
        raise RuntimeError("Frozen image manifest schema changed.")
    image_manifest["_sid"] = norm_sid_series(image_manifest["sample_id"])
    image_manifest = image_manifest.sort_values("_sid").reset_index(drop=True)
    if len(image_manifest) != 1000 or image_manifest["_sid"].nunique() != 1000:
        raise RuntimeError("Frozen image manifest cardinality changed.")
    cls_map = {
        sid: cls[i]
        for i, sid in enumerate(image_manifest["_sid"].tolist())
    }

    # ---------- numeric FG/BG ----------
    with np.load(STATE_FEATURES, allow_pickle=False) as z:
        required = {"foreground_patch_mean", "background_patch_mean"}
        if not required.issubset(z.files):
            raise RuntimeError(f"R10K2A state feature keys changed: {z.files}")
        fg = np.asarray(z["foreground_patch_mean"], dtype=np.float32)
        bg = np.asarray(z["background_patch_mean"], dtype=np.float32)

    if fg.shape != (9000, 768) or bg.shape != (9000, 768):
        raise RuntimeError(f"Unexpected FG/BG shapes: {fg.shape}, {bg.shape}")
    if not np.isfinite(fg).all() or not np.isfinite(bg).all():
        raise RuntimeError("Non-finite FG/BG representation.")

    state_meta = pd.read_csv(STATE_METADATA, low_memory=False)
    req_meta = {
        "sample_id", "model_family", "model_state_id",
        "training_seed", "checkpoint_sha256",
    }
    if not req_meta.issubset(state_meta.columns):
        raise RuntimeError("R10K2A state metadata schema changed.")
    state_meta["_sid"] = norm_sid_series(state_meta["sample_id"])
    state_meta["_state_row"] = np.arange(len(state_meta), dtype=int)
    if len(state_meta) != 9000:
        raise RuntimeError("R10K2A state metadata rows !=9000.")
    if state_meta[["_sid", "model_state_id"]].drop_duplicates().shape[0] != 9000:
        raise RuntimeError("R10K2A state key not unique.")

    morph = pd.read_csv(MORPH, low_memory=False)
    req_morph = {
        "sample_id", "model_family", "model_state_id", "harm_label",
        *M2,
    }
    if not req_morph.issubset(morph.columns):
        raise RuntimeError("Morphology table schema changed.")
    morph["_sid"] = norm_sid_series(morph["sample_id"])
    if len(morph) != 9000:
        raise RuntimeError("Morphology rows !=9000.")

    joined = morph.merge(
        state_meta[[
            "_sid", "model_state_id", "model_family",
            "training_seed", "checkpoint_sha256", "_state_row",
        ]],
        on=["_sid", "model_state_id"],
        how="inner",
        validate="one_to_one",
        suffixes=("_morph", "_repr"),
    )

    if len(joined) != 9000:
        raise RuntimeError("Morphology/state metadata join !=9000.")

    if not (
        joined["model_family_morph"].astype(str)
        == joined["model_family_repr"].astype(str)
    ).all():
        raise RuntimeError("Model-family mismatch after explicit-key join.")

    joined = joined.rename(columns={"model_family_morph": "model_family"})
    joined = joined.drop(columns=[
        c for c in (
            "model_family_repr", "training_seed_repr", "checkpoint_sha256_repr"
        )
        if c in joined.columns
    ])

    order = joined["_state_row"].to_numpy(dtype=int)
    fg_aligned = fg[order]
    bg_aligned = bg[order]
    cls_aligned = np.stack([
        cls_map[sid] for sid in joined["_sid"].tolist()
    ]).astype(np.float32)

    return joined.reset_index(drop=True), cls_aligned, fg_aligned, bg_aligned


def load_and_audit_historical_predictions():
    df = pd.read_csv(HISTORICAL_PREDICTIONS, low_memory=False)
    required = {
        "target_family", "seed", "method", "sample_id", "model_state_id",
        "harm_label", "probability", "source_only_threshold",
    }
    if not required.issubset(df.columns):
        raise RuntimeError("Historical R10K2B prediction schema changed.")

    df = df[df["method"].isin(HISTORICAL_METHODS)].copy()
    df["seed"] = pd.to_numeric(df["seed"], errors="raise").astype(int)
    df["harm_label"] = pd.to_numeric(df["harm_label"], errors="raise").astype(int)
    df["probability"] = pd.to_numeric(df["probability"], errors="raise").astype(float)
    df["source_only_threshold"] = pd.to_numeric(
        df["source_only_threshold"], errors="raise"
    ).astype(float)
    df["_sid"] = norm_sid_series(df["sample_id"])

    expected = len(FAMILIES) * len(SEEDS) * len(HISTORICAL_METHODS) * 3000
    if len(df) != expected:
        raise RuntimeError(
            f"Historical prediction rows={len(df)} expected={expected}."
        )
    if set(df["target_family"].unique()) != set(FAMILIES):
        raise RuntimeError("Historical target families changed.")
    if set(df["seed"].unique()) != set(SEEDS):
        raise RuntimeError("Historical seeds changed.")
    if set(df["method"].unique()) != set(HISTORICAL_METHODS):
        raise RuntimeError("Historical methods changed.")

    return df.drop(columns=["_sid"])


def metrics_from_prediction_rows(df):
    rows = []
    for (family, seed, method), g in df.groupby(
        ["target_family", "seed", "method"], sort=False
    ):
        y = g["harm_label"].to_numpy(dtype=int)
        p = g["probability"].to_numpy(dtype=float)
        threshold = g["source_only_threshold"].to_numpy(dtype=float)
        pred = (p >= threshold).astype(int)

        auc = float(roc_auc_score(y, p))
        ap = float(average_precision_score(y, p))

        tp = int(np.sum((pred == 1) & (y == 1)))
        fp = int(np.sum((pred == 1) & (y == 0)))
        tn = int(np.sum((pred == 0) & (y == 0)))
        fn = int(np.sum((pred == 0) & (y == 1)))
        recall = tp / (tp + fn)
        fpr = fp / (fp + tn)

        pi = PPV_PREVALENCE
        denom = recall * pi + fpr * (1 - pi)
        ppv1 = recall * pi / denom if denom > 0 else np.nan

        oracle_thr = threshold_for_recall(y, p, TARGET_RECALL)
        oracle = operating_metrics(y, p, oracle_thr)

        rows.append({
            "target_family": family,
            "seed": int(seed),
            "method": method,
            "target_AUROC": auc,
            "target_AUPRC": ap,
            "transfer_Recall": float(recall),
            "transfer_FPR": float(fpr),
            "transfer_PPV1pct": float(ppv1),
            "oracle_R90_FPR": float(oracle["FPR"]),
        })
    return pd.DataFrame(rows)


def summarize(results):
    by_family = (
        results.groupby(["target_family", "method"], sort=False)
        .agg(
            seeds=("seed", "count"),
            target_AUROC_mean=("target_AUROC", "mean"),
            target_AUROC_std=("target_AUROC", "std"),
            target_AUPRC_mean=("target_AUPRC", "mean"),
            target_AUPRC_std=("target_AUPRC", "std"),
            transfer_Recall_mean=("transfer_Recall", "mean"),
            transfer_Recall_std=("transfer_Recall", "std"),
            transfer_FPR_mean=("transfer_FPR", "mean"),
            transfer_FPR_std=("transfer_FPR", "std"),
            transfer_PPV1pct_mean=("transfer_PPV1pct", "mean"),
            transfer_PPV1pct_std=("transfer_PPV1pct", "std"),
            oracle_R90_FPR_mean=("oracle_R90_FPR", "mean"),
            oracle_R90_FPR_std=("oracle_R90_FPR", "std"),
        )
        .reset_index()
    )

    macro_seed = (
        results.groupby(["method", "seed"], sort=False)
        .agg(
            macro_AUROC=("target_AUROC", "mean"),
            macro_AUPRC=("target_AUPRC", "mean"),
            macro_transfer_Recall=("transfer_Recall", "mean"),
            macro_transfer_FPR=("transfer_FPR", "mean"),
            macro_PPV1pct=("transfer_PPV1pct", "mean"),
            macro_oracle_R90_FPR=("oracle_R90_FPR", "mean"),
        )
        .reset_index()
    )

    macro = (
        macro_seed.groupby("method", sort=False)
        .agg(
            seeds=("seed", "count"),
            macro_AUROC_mean=("macro_AUROC", "mean"),
            macro_AUROC_std=("macro_AUROC", "std"),
            macro_AUPRC_mean=("macro_AUPRC", "mean"),
            macro_AUPRC_std=("macro_AUPRC", "std"),
            macro_transfer_Recall_mean=("macro_transfer_Recall", "mean"),
            macro_transfer_Recall_std=("macro_transfer_Recall", "std"),
            macro_transfer_FPR_mean=("macro_transfer_FPR", "mean"),
            macro_transfer_FPR_std=("macro_transfer_FPR", "std"),
            macro_PPV1pct_mean=("macro_PPV1pct", "mean"),
            macro_PPV1pct_std=("macro_PPV1pct", "std"),
            macro_oracle_R90_FPR_mean=("macro_oracle_R90_FPR", "mean"),
            macro_oracle_R90_FPR_std=("macro_oracle_R90_FPR", "std"),
        )
        .reset_index()
    )
    return by_family, macro_seed, macro


def audit_historical_macro(macro):
    print("\n===== HISTORICAL R10K2B REPRODUCTION SENTINEL =====")
    for method in HISTORICAL_METHODS:
        row = macro[macro["method"] == method]
        if len(row) != 1:
            raise RuntimeError(f"Historical macro row missing: {method}")
        row = row.iloc[0]
        auc = float(row["macro_AUROC_mean"])
        ap = float(row["macro_AUPRC_mean"])
        auc_ref = HISTORICAL_AUROC_SENTINELS[method]
        ap_ref = HISTORICAL_AUPRC_SENTINELS[method]
        print(
            method,
            f"AUROC={auc:.6f} ref={auc_ref:.6f} delta={auc-auc_ref:+.6f}",
            f"AUPRC={ap:.6f} ref={ap_ref:.6f} delta={ap-ap_ref:+.6f}",
        )
        if abs(auc - auc_ref) > SENTINEL_TOL:
            raise RuntimeError(f"Historical AUROC sentinel failed: {method}")
        if abs(ap - ap_ref) > SENTINEL_TOL:
            raise RuntimeError(f"Historical AUPRC sentinel failed: {method}")
    print("HISTORICAL_R10K2B_REPRODUCTION_SENTINEL_PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = ap.parse_args()

    print("===== Q1 R15A0 PREDICTION-CONDITIONED CORE ABLATION =====")
    print("STATUS=POST_FREEZE_EXPLANATORY_SOURCE_SIDE_ABLATION")
    print("FINAL_METHOD_RESELECTION=NO")
    print("EXTERNAL_DATA_ACCESS=NO")
    print("TARGET_FAMILY_LABELS_FOR_EVALUATION_ONLY=YES")
    print("GROUP_KEY=sample_id")
    print("OUTER_FOLDS=", OUTER_FOLDS)
    print("INNER_FOLDS=", INNER_FOLDS)
    print("SEEDS=", SEEDS)
    print("PCA_DIM=", PCA_DIM)
    print("TARGET_RECALL_SOURCE_ONLY=", TARGET_RECALL)
    print("HISTORICAL_METHODS_REUSED=", HISTORICAL_METHODS)
    print("NEW_METHODS=", NEW_METHODS)

    required = [
        IMAGE_FEATURES,
        STATE_FEATURES,
        STATE_METADATA,
        IMAGE_MANIFEST,
        MORPH,
        HISTORICAL_PREDICTIONS,
    ]
    for p in required:
        print("exists:", p.exists(), p)
        if not p.is_file():
            raise FileNotFoundError(p)

    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)

    df, cls_all, fg_all, bg_all = load_panel_and_components()
    historical_preds = load_and_audit_historical_predictions()

    y_all = pd.to_numeric(df["harm_label"], errors="raise").astype(int).to_numpy()
    groups_all = df["_sid"].to_numpy()
    family_all = df["model_family"].astype(str).to_numpy()

    print("\n===== PANEL AUDIT =====")
    print("rows:", len(df))
    print("cases:", df["_sid"].nunique())
    print("states:", df["model_state_id"].nunique())
    print("HARM/NON-HARM:", int(y_all.sum()), int((y_all == 0).sum()))
    print("CLS:", cls_all.shape)
    print("FG:", fg_all.shape)
    print("BG:", bg_all.shape)

    if len(df) != 9000 or df["_sid"].nunique() != 1000:
        raise RuntimeError("Source panel cardinality changed.")
    if set(df["model_family"].astype(str).unique()) != set(FAMILIES):
        raise RuntimeError("Source family set changed.")

    new_prediction_rows = []
    fold_rows = []

    total_outer = len(FAMILIES) * len(SEEDS) * OUTER_FOLDS
    outer_progress = tqdm(total=total_outer, desc="R15A0 outer folds", unit="fold", dynamic_ncols=True)

    for target_family in FAMILIES:
        source_idx = np.flatnonzero(family_all != target_family)
        target_idx = np.flatnonzero(family_all == target_family)

        source = df.iloc[source_idx].reset_index(drop=True)
        target = df.iloc[target_idx].reset_index(drop=True)

        y_source = y_all[source_idx]
        y_target = y_all[target_idx]
        g_source = groups_all[source_idx]
        g_target = groups_all[target_idx]

        fg_source = fg_all[source_idx]
        fg_target = fg_all[target_idx]
        bg_source = bg_all[source_idx]
        bg_target = bg_all[target_idx]

        if set(g_source) != set(g_target):
            raise RuntimeError(f"Source/target case pools differ: {target_family}")

        print("\n============================================================")
        print("TARGET FAMILY:", target_family)
        print("Source rows:", len(source), "Target rows:", len(target))
        print("Target HARM/NON-HARM:", int(y_target.sum()), int((y_target == 0).sum()))
        print("============================================================")

        for seed in SEEDS:
            random.seed(seed)
            np.random.seed(seed)

            outer = StratifiedGroupKFold(
                n_splits=OUTER_FOLDS,
                shuffle=True,
                random_state=seed,
            )
            splits = list(outer.split(np.zeros((len(source), 1)), y_source, g_source))

            probs = {
                method: np.full(len(target), np.nan, dtype=float)
                for method in NEW_METHODS
            }
            thresholds = {
                method: np.full(len(target), np.nan, dtype=float)
                for method in NEW_METHODS
            }

            for fold, (tr_s, te_s) in enumerate(splits):
                train_cases = set(g_source[tr_s])
                test_cases = set(g_source[te_s])
                if train_cases & test_cases:
                    raise RuntimeError("Outer source group leakage.")

                te_t = np.flatnonzero(np.isin(g_target, list(test_cases)))
                if set(g_target[te_t]) != test_cases:
                    raise RuntimeError("Held-out target case set mismatch.")

                df_train = source.iloc[tr_s]
                df_eval = target.iloc[te_t]
                y_train = y_source[tr_s]
                g_train = g_source[tr_s]

                fg_train = fg_source[tr_s]
                fg_eval = fg_target[te_t]
                bg_train = bg_source[tr_s]
                bg_eval = bg_target[te_t]

                for method in NEW_METHODS:
                    inner_p = inner_oof_new_method(
                        method=method,
                        df_outer_train=df_train.reset_index(drop=True),
                        y=y_train,
                        groups=g_train,
                        fg_outer_train=fg_train,
                        bg_outer_train=bg_train,
                        seed=seed + 1000 * fold,
                    )
                    threshold = threshold_for_recall(y_train, inner_p, TARGET_RECALL)

                    Xtr, Xev, evr = build_new_method_matrix(
                        method=method,
                        df_train=df_train,
                        df_eval=df_eval,
                        fg_train=fg_train,
                        fg_eval=fg_eval,
                        bg_train=bg_train,
                        bg_eval=bg_eval,
                        seed=seed + 10000 + fold,
                    )

                    clf = build_lr(seed + 20000 + fold)
                    clf.fit(Xtr, y_train)
                    probs[method][te_t] = clf.predict_proba(Xev)[:, 1]
                    thresholds[method][te_t] = threshold

                    fold_rows.append({
                        "target_family": target_family,
                        "seed": seed,
                        "fold": fold,
                        "method": method,
                        "train_cases": len(train_cases),
                        "test_cases": len(test_cases),
                        "source_train_rows": len(tr_s),
                        "target_test_rows": len(te_t),
                        "pca_components": PCA_DIM,
                        "pca_explained_variance": evr,
                        "target_rows_used_to_fit_pca": 0,
                        "source_inner_threshold_R90": threshold,
                    })

                outer_progress.update(1)
                outer_progress.set_postfix(target=target_family, seed=seed, fold=fold)

            print(f"\n######## TARGET={target_family} SEED={seed} ########")
            for method in NEW_METHODS:
                if np.isnan(probs[method]).any() or np.isnan(thresholds[method]).any():
                    raise RuntimeError(f"Incomplete predictions: {target_family}/{seed}/{method}")

                p = probs[method]
                auc = float(roc_auc_score(y_target, p))
                apv = float(average_precision_score(y_target, p))
                pred = (p >= thresholds[method]).astype(int)
                tp = int(np.sum((pred == 1) & (y_target == 1)))
                fp = int(np.sum((pred == 1) & (y_target == 0)))
                tn = int(np.sum((pred == 0) & (y_target == 0)))
                fn = int(np.sum((pred == 0) & (y_target == 1)))
                recall = tp / (tp + fn)
                fpr = fp / (fp + tn)
                denom = recall * PPV_PREVALENCE + fpr * (1 - PPV_PREVALENCE)
                ppv1 = recall * PPV_PREVALENCE / denom if denom > 0 else np.nan
                oracle_thr = threshold_for_recall(y_target, p, TARGET_RECALL)
                oracle = operating_metrics(y_target, p, oracle_thr)

                print(
                    f"{method:24s} AUROC={auc:.6f} AUPRC={apv:.6f} "
                    f"Recall={recall:.6f} FPR={fpr:.6f} "
                    f"PPV1%={ppv1:.6f} OracleFPR90={oracle['FPR']:.6f}"
                )

                for i in range(len(target)):
                    new_prediction_rows.append({
                        "target_family": target_family,
                        "seed": seed,
                        "method": method,
                        "sample_id": target.iloc[i]["sample_id"],
                        "model_state_id": target.iloc[i]["model_state_id"],
                        "harm_label": int(y_target[i]),
                        "probability": float(p[i]),
                        "source_only_threshold": float(thresholds[method][i]),
                    })

    outer_progress.close()

    new_preds = pd.DataFrame(new_prediction_rows)
    expected_new_rows = len(FAMILIES) * len(SEEDS) * len(NEW_METHODS) * 3000
    if len(new_preds) != expected_new_rows:
        raise RuntimeError(f"New prediction rows={len(new_preds)} expected={expected_new_rows}")

    all_preds = pd.concat([historical_preds, new_preds], ignore_index=True)
    expected_all = len(FAMILIES) * len(SEEDS) * len(ALL_METHODS) * 3000
    if len(all_preds) != expected_all:
        raise RuntimeError(f"All ablation rows={len(all_preds)} expected={expected_all}")

    # Every method must share the same explicit target model-case keys per family/seed.
    for family in FAMILIES:
        for seed in SEEDS:
            ref = None
            for method in ALL_METHODS:
                g = all_preds[
                    (all_preds["target_family"] == family)
                    & (all_preds["seed"] == seed)
                    & (all_preds["method"] == method)
                ]
                keys = set(zip(g["sample_id"].astype(str), g["model_state_id"].astype(str)))
                if len(keys) != 3000:
                    raise RuntimeError(f"Key cardinality failure: {family}/{seed}/{method}")
                if ref is None:
                    ref = keys
                elif keys != ref:
                    raise RuntimeError(f"Paired target key mismatch: {family}/{seed}/{method}")

    results = metrics_from_prediction_rows(all_preds)
    by_family, macro_seed, macro = summarize(results)
    audit_historical_macro(macro)

    print("\n===== R15A0 MACRO CORE ABLATION =====")
    print(macro.to_string(index=False))

    print("\n===== COMPONENT DELTAS (POINT ESTIMATES ONLY) =====")
    mm = macro.set_index("method")
    comparisons = [
        ("CondDINO_PCA64", "ImageCLS_PCA64", "FG+BG conditioning vs image CLS"),
        ("CondDINO_PCA64", "FG_PCA64", "FG+BG vs FG only"),
        ("CondDINO_PCA64", "BG_PCA64", "FG+BG vs BG only"),
        (FINAL_METHOD, "M2_backbone", "final vs morphology only"),
        (FINAL_METHOD, "M2_plus_FG_PCA64", "final vs M2+FG"),
        (FINAL_METHOD, "M2_plus_BG_PCA64", "final vs M2+BG"),
        (FINAL_METHOD, "CondDINO_PCA64", "M2 incremental contribution"),
    ]

    delta_rows = []
    for a, b, label in comparisons:
        row = {
            "comparison": f"{a} minus {b}",
            "interpretation": label,
            "delta_macro_AUROC": float(mm.loc[a, "macro_AUROC_mean"] - mm.loc[b, "macro_AUROC_mean"]),
            "delta_macro_AUPRC": float(mm.loc[a, "macro_AUPRC_mean"] - mm.loc[b, "macro_AUPRC_mean"]),
            "delta_macro_transfer_FPR": float(mm.loc[a, "macro_transfer_FPR_mean"] - mm.loc[b, "macro_transfer_FPR_mean"]),
            "delta_macro_oracle_R90_FPR": float(mm.loc[a, "macro_oracle_R90_FPR_mean"] - mm.loc[b, "macro_oracle_R90_FPR_mean"]),
        }
        delta_rows.append(row)
        print(
            label,
            f"dAUROC={row['delta_macro_AUROC']:+.6f}",
            f"dAUPRC={row['delta_macro_AUPRC']:+.6f}",
            f"dFPR={row['delta_macro_transfer_FPR']:+.6f}",
            f"dOracleFPR90={row['delta_macro_oracle_R90_FPR']:+.6f}",
        )

    all_pred_path = args.output_dir / "R15A0_all_8method_target_predictions.csv"
    new_pred_path = args.output_dir / "R15A0_new_component_ablation_predictions.csv"
    results_path = args.output_dir / "R15A0_family_seed_metrics.csv"
    by_family_path = args.output_dir / "R15A0_by_family_summary.csv"
    macro_seed_path = args.output_dir / "R15A0_macro_by_seed.csv"
    macro_path = args.output_dir / "R15A0_macro_summary.csv"
    delta_path = args.output_dir / "R15A0_component_point_deltas.csv"
    fold_path = args.output_dir / "R15A0_new_method_fold_audit.csv"

    all_preds.to_csv(all_pred_path, index=False)
    new_preds.to_csv(new_pred_path, index=False)
    results.to_csv(results_path, index=False)
    by_family.to_csv(by_family_path, index=False)
    macro_seed.to_csv(macro_seed_path, index=False)
    macro.to_csv(macro_path, index=False)
    pd.DataFrame(delta_rows).to_csv(delta_path, index=False)
    pd.DataFrame(fold_rows).to_csv(fold_path, index=False)

    artifacts = {}
    for p in [
        all_pred_path, new_pred_path, results_path, by_family_path,
        macro_seed_path, macro_path, delta_path, fold_path,
    ]:
        artifacts[p.name] = {
            "sha256": sha256_file(p),
            "bytes": p.stat().st_size,
        }

    lock = {
        "status": "PASS",
        "decision": DECISION,
        "script_version": VERSION,
        "build": BUILD,
        "scientific_role": "post-freeze explanatory source-side component ablation",
        "final_method_reselected": False,
        "external_data_accessed": False,
        "panel": {
            "rows": 9000,
            "cases": 1000,
            "model_states": 9,
            "families": FAMILIES,
        },
        "protocol": {
            "outer_folds": OUTER_FOLDS,
            "inner_folds": INNER_FOLDS,
            "seeds": SEEDS,
            "group_key": "sample_id",
            "pca_dim": PCA_DIM,
            "pca_whiten": True,
            "target_recall_for_source_only_threshold": TARGET_RECALL,
            "target_family_labels_used_for_evaluation_only": True,
            "target_family_features_used_to_fit_pca": False,
            "final_method_or_threshold_changed": False,
        },
        "methods": ALL_METHODS,
        "historical_methods_reused_from_frozen_predictions": HISTORICAL_METHODS,
        "newly_fitted_component_ablation_methods": NEW_METHODS,
        "upstream": {
            "image_features_sha256": sha256_file(IMAGE_FEATURES),
            "state_features_sha256": sha256_file(STATE_FEATURES),
            "state_metadata_sha256": sha256_file(STATE_METADATA),
            "image_manifest_sha256": sha256_file(IMAGE_MANIFEST),
            "morphology_sha256": sha256_file(MORPH),
            "historical_r10k2b_predictions_sha256": sha256_file(HISTORICAL_PREDICTIONS),
        },
        "artifacts": artifacts,
        "next_stage": "R15A1_CASE_CLUSTERED_PAIRED_BOOTSTRAP_FOR_CORE_ABLATION",
    }

    lock_path = args.output_dir / "R15A0_CORE_ABLATION_LOCK.json"
    lock_path.write_text(json.dumps(lock, indent=2), encoding="utf-8")

    print("\n===== R15A0 FINAL =====")
    print("Decision=", DECISION)
    print("LOCK=", lock_path)
    print("LOCK SHA256=", sha256_file(lock_path))
    print("PASS")


if __name__ == "__main__":
    main()
