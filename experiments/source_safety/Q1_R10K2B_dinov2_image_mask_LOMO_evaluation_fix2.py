#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R10K2B_dinov2_image_mask_LOMO_evaluation_fix2.py

Strict leave-one-model-family-out evaluation of the frozen R10K2A
image-conditioned representation.

Frozen representation inputs
----------------------------
- image CLS: 1000 x 768
- foreground-weighted DINOv2 patch mean: 9000 x 768
- background-weighted DINOv2 patch mean: 9000 x 768
- morphology table with frozen M2 backbone and HARM labels

Methods
-------
1) M2_backbone
2) ImageCLS_PCA64
3) CondDINO_PCA64
4) M2_plus_CondDINO_PCA64   [PRIMARY]

Strict source-only fitting
--------------------------
- Outer: 5-fold StratifiedGroupKFold by sample_id
- Target family labels: evaluation only
- PCA is fitted on SOURCE-TRAIN rows only
- For source-only R90 threshold selection, PCA is re-fitted inside each
  INNER source-training split; inner validation features are not used to fit PCA
- No target-family features are used to fit PCA
- No target labels are used for fitting / threshold selection / tuning

Frozen dimensions
-----------------
- Image CLS PCA: 64
- Mask-conditioned FG+BG DINO PCA: 64
- No PCA dimension sweep
"""

from __future__ import annotations

import argparse
from pathlib import Path
import random
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score


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

METHODS = [
    "M2_backbone",
    "ImageCLS_PCA64",
    "CondDINO_PCA64",
    "M2_plus_CondDINO_PCA64",
]

PRIMARY = "M2_plus_CondDINO_PCA64"

DEFAULT_IMAGE_FEATURES = (
    "F:/MEDSEG_SAFETTA/outputs/"
    "Q1_R10K2A_dinov2_image_mask_representation_lock_fix1_v1/"
    "R10K2A_image_cls_features.npz"
)

DEFAULT_STATE_FEATURES = (
    "F:/MEDSEG_SAFETTA/outputs/"
    "Q1_R10K2A_dinov2_image_mask_representation_lock_fix1_v1/"
    "R10K2A_mask_conditioned_dinov2_features.npz"
)

DEFAULT_IMAGE_MANIFEST = (
    "F:/MEDSEG_SAFETTA/outputs/"
    "Q1_R05D1_neopolyp_acquisition_pairing_duplicate_qc_v1_fix2/"
    "frozen_confirmatory_manifest.csv"
)

DEFAULT_STATE_METADATA = (
    "F:/MEDSEG_SAFETTA/outputs/"
    "Q1_R10K2A_dinov2_image_mask_representation_lock_fix1_v1/"
    "R10K2A_state_metadata_no_labels.csv"
)

DEFAULT_MORPH = (
    "F:/MEDSEG_SAFETTA/outputs/"
    "Q1_R10J1_source_mask_morphology_feature_builder_fix1_v1/"
    "source_mask_morphology_features_labeled.csv"
)

DEFAULT_OUTPUT = (
    "F:/MEDSEG_SAFETTA/outputs/"
    "Q1_R10K2B_dinov2_image_mask_LOMO_evaluation_fix2_v1"
)


def norm_sid_value(x):
    return str(x).strip().replace("\\", "/").lower()


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
        raise AssertionError("No positive samples for threshold selection.")

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
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    empirical_ppv = tp / (tp + fp) if (tp + fp) else np.nan

    pi = PPV_PREVALENCE
    denom = recall * pi + fpr * (1.0 - pi)
    ppv1 = recall * pi / denom if denom > 0 else np.nan

    return {
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "Recall": float(recall),
        "FPR": float(fpr),
        "Specificity": float(specificity),
        "EmpiricalPPV": float(empirical_ppv),
        "PPV1pct": float(ppv1),
    }


def make_method_matrices(
    method,
    df_train,
    df_eval,
    cls_train,
    cls_eval,
    cond_train,
    cond_eval,
    seed,
):
    """
    Fit all unsupervised transforms on TRAIN only, then transform EVAL.

    Returns:
      X_train_final, X_eval_final, diagnostics
    """
    diag = {
        "pca_components": 0,
        "pca_explained_variance": np.nan,
    }

    if method == "M2_backbone":
        return (
            tabular_matrix(df_train, M2),
            tabular_matrix(df_eval, M2),
            diag,
        )

    if method == "ImageCLS_PCA64":
        pca = PCA(
            n_components=PCA_DIM,
            svd_solver="randomized",
            whiten=True,
            random_state=seed,
        )
        ztr = pca.fit_transform(cls_train)
        zev = pca.transform(cls_eval)
        diag["pca_components"] = PCA_DIM
        diag["pca_explained_variance"] = float(
            pca.explained_variance_ratio_.sum()
        )
        return ztr, zev, diag

    if method == "CondDINO_PCA64":
        pca = PCA(
            n_components=PCA_DIM,
            svd_solver="randomized",
            whiten=True,
            random_state=seed,
        )
        ztr = pca.fit_transform(cond_train)
        zev = pca.transform(cond_eval)
        diag["pca_components"] = PCA_DIM
        diag["pca_explained_variance"] = float(
            pca.explained_variance_ratio_.sum()
        )
        return ztr, zev, diag

    if method == "M2_plus_CondDINO_PCA64":
        pca = PCA(
            n_components=PCA_DIM,
            svd_solver="randomized",
            whiten=True,
            random_state=seed,
        )
        ztr = pca.fit_transform(cond_train)
        zev = pca.transform(cond_eval)
        m2tr = tabular_matrix(df_train, M2)
        m2ev = tabular_matrix(df_eval, M2)

        diag["pca_components"] = PCA_DIM
        diag["pca_explained_variance"] = float(
            pca.explained_variance_ratio_.sum()
        )

        return (
            np.column_stack([m2tr, ztr]),
            np.column_stack([m2ev, zev]),
            diag,
        )

    raise ValueError(f"Unknown method: {method}")


def inner_oof_probabilities(
    method,
    df_outer_train,
    y,
    groups,
    cls_outer_train,
    cond_outer_train,
    seed,
):
    """
    Strict nested preprocessing:
    each inner fold refits PCA only on inner-train rows.
    """
    cv = StratifiedGroupKFold(
        n_splits=INNER_FOLDS,
        shuffle=True,
        random_state=seed,
    )

    p = np.full(len(y), np.nan, dtype=float)

    for fold, (tr, va) in enumerate(
        cv.split(
            np.zeros((len(y), 1)),
            y,
            groups,
        )
    ):
        if set(groups[tr]) & set(groups[va]):
            raise AssertionError("Inner group leakage.")

        Xtr, Xva, _ = make_method_matrices(
            method=method,
            df_train=df_outer_train.iloc[tr],
            df_eval=df_outer_train.iloc[va],
            cls_train=cls_outer_train[tr],
            cls_eval=cls_outer_train[va],
            cond_train=cond_outer_train[tr],
            cond_eval=cond_outer_train[va],
            seed=seed + fold,
        )

        clf = build_lr(seed + 100 + fold)
        clf.fit(Xtr, y[tr])
        p[va] = clf.predict_proba(Xva)[:, 1]

    if np.isnan(p).any():
        raise AssertionError("Incomplete inner OOF probabilities.")

    return p


def load_locked_features(
    image_feature_path,
    state_feature_path,
    image_manifest_path,
    state_metadata_path,
    morph_path,
):
    """
    Fix2 loader.

    R10K2A NPZ files contain string metadata arrays that NumPy stored with
    object dtype.  Accessing those object arrays under allow_pickle=False
    fails.  Fix2 deliberately DOES NOT enable pickle loading.

    Instead:
    - numeric feature arrays are read from NPZ with allow_pickle=False;
    - image sample_id order is reconstructed from the authoritative R05D1
      frozen image manifest using the exact same normalization+sort used in
      R10K2A;
    - state-row identity is taken from the explicit R10K2A no-label metadata
      CSV, which was written in the same frozen state-row order as the numeric
      conditioned-DINO arrays.

    No scientific representation, split, PCA rule, model, threshold rule, seed,
    or outcome label changes.
    """

    # ---------------- Image CLS numeric array only ----------------
    with np.load(image_feature_path, allow_pickle=False) as z:
        if "cls" not in z.files:
            raise AssertionError(
                f"Image feature file keys={z.files}, required numeric key='cls'"
            )
        image_cls = np.asarray(z["cls"], dtype=np.float32)

    if image_cls.shape != (1000, 768):
        raise AssertionError(
            f"Unexpected image CLS shape: {image_cls.shape}"
        )
    if not np.isfinite(image_cls).all():
        raise AssertionError("Non-finite image CLS features.")

    # Authoritative image row order: exactly the R10K2A construction rule.
    image_manifest = pd.read_csv(
        image_manifest_path,
        low_memory=False,
    )
    required_image_manifest = ["sample_id", "image_path"]
    missing = [
        c for c in required_image_manifest
        if c not in image_manifest.columns
    ]
    if missing:
        raise AssertionError(
            f"Image manifest missing: {missing}"
        )

    image_manifest["_sid"] = norm_sid_series(
        image_manifest["sample_id"]
    )

    if len(image_manifest) != 1000:
        raise AssertionError(
            f"Expected 1000 image-manifest rows, got {len(image_manifest)}."
        )
    if image_manifest["_sid"].nunique() != 1000:
        raise AssertionError(
            "Image manifest sample_id is not one-to-one."
        )

    image_manifest = (
        image_manifest
        .sort_values("_sid")
        .reset_index(drop=True)
    )
    image_sid_norm = image_manifest["_sid"].to_numpy()

    cls_map = {
        sid: image_cls[i]
        for i, sid in enumerate(image_sid_norm)
    }

    # ---------------- State-conditioned DINO numeric arrays only ----------------
    with np.load(state_feature_path, allow_pickle=False) as z:
        required_numeric = {
            "foreground_patch_mean",
            "background_patch_mean",
        }
        if not required_numeric.issubset(z.files):
            raise AssertionError(
                f"State feature file keys={z.files}, "
                f"required numeric keys={required_numeric}"
            )

        fg = np.asarray(
            z["foreground_patch_mean"],
            dtype=np.float32,
        )
        bg = np.asarray(
            z["background_patch_mean"],
            dtype=np.float32,
        )

    if fg.shape != (9000, 768):
        raise AssertionError(f"Unexpected FG shape: {fg.shape}")
    if bg.shape != (9000, 768):
        raise AssertionError(f"Unexpected BG shape: {bg.shape}")
    if not np.isfinite(fg).all():
        raise AssertionError("Non-finite foreground DINO features.")
    if not np.isfinite(bg).all():
        raise AssertionError("Non-finite background DINO features.")

    cond = np.concatenate([fg, bg], axis=1).astype(np.float32)

    # Explicit no-label metadata was written by R10K2A in exactly the same
    # state-row order as fg/bg features.
    state_meta = pd.read_csv(
        state_metadata_path,
        low_memory=False,
    )

    required_state_meta = [
        "sample_id",
        "model_family",
        "model_state_id",
        "training_seed",
        "checkpoint_sha256",
    ]
    missing = [
        c for c in required_state_meta
        if c not in state_meta.columns
    ]
    if missing:
        raise AssertionError(
            f"State metadata missing: {missing}"
        )

    state_meta["_sid"] = norm_sid_series(
        state_meta["sample_id"]
    )
    state_meta["_state_row"] = np.arange(
        len(state_meta),
        dtype=int,
    )

    if len(state_meta) != 9000:
        raise AssertionError(
            f"Expected 9000 state metadata rows, got {len(state_meta)}."
        )
    if state_meta["_sid"].nunique() != 1000:
        raise AssertionError(
            "Expected 1000 cases in state metadata."
        )
    if state_meta["model_state_id"].nunique() != 9:
        raise AssertionError(
            "Expected 9 model states in state metadata."
        )
    if (
        state_meta[["_sid", "model_state_id"]]
        .drop_duplicates()
        .shape[0]
        != 9000
    ):
        raise AssertionError(
            "R10K2A state metadata explicit key is not unique."
        )

    if set(image_sid_norm.tolist()) != set(state_meta["_sid"].tolist()):
        raise AssertionError(
            "Image-manifest and state-metadata case sets differ."
        )

    # ---------------- Frozen labeled morphology ----------------
    morph = pd.read_csv(morph_path, low_memory=False)

    required_morph = [
        "sample_id",
        "model_family",
        "model_state_id",
        "harm_label",
    ] + M2

    missing = [
        c for c in required_morph
        if c not in morph.columns
    ]
    if missing:
        raise AssertionError(
            f"Morphology table missing: {missing}"
        )

    morph["_sid"] = norm_sid_series(morph["sample_id"])

    if len(morph) != 9000:
        raise AssertionError(
            f"Expected 9000 morphology rows, got {len(morph)}."
        )
    if (
        morph[["_sid", "model_state_id"]]
        .drop_duplicates()
        .shape[0]
        != 9000
    ):
        raise AssertionError(
            "Morphology explicit key is not unique."
        )

    # Explicit-key join.  No row-order inference from the morphology table.
    joined = morph.merge(
        state_meta[
            [
                "_sid",
                "model_state_id",
                "model_family",
                "training_seed",
                "checkpoint_sha256",
                "_state_row",
            ]
        ],
        on=["_sid", "model_state_id"],
        how="inner",
        validate="one_to_one",
        suffixes=("_morph", "_repr"),
    )

    if len(joined) != 9000:
        raise AssertionError(
            "Morphology/R10K2A representation join != 9000."
        )

    fam_ok = (
        joined["model_family_morph"].astype(str)
        == joined["model_family_repr"].astype(str)
    )
    if not fam_ok.all():
        raise AssertionError(
            "Model-family mismatch in representation join."
        )

    # Where morphology contains seed/checkpoint, cross-check them too.
    if "training_seed_morph" in joined.columns:
        seed_ok = (
            joined["training_seed_morph"].astype(str)
            == joined["training_seed_repr"].astype(str)
        )
        if not seed_ok.all():
            raise AssertionError(
                "Training-seed mismatch in representation join."
            )

    if "checkpoint_sha256_morph" in joined.columns:
        sha_ok = (
            joined["checkpoint_sha256_morph"].astype(str)
            == joined["checkpoint_sha256_repr"].astype(str)
        )
        if not sha_ok.all():
            raise AssertionError(
                "Checkpoint mismatch in representation join."
            )

    joined = joined.rename(
        columns={"model_family_morph": "model_family"}
    )

    drop_cols = [
        c for c in [
            "model_family_repr",
            "training_seed_repr",
            "checkpoint_sha256_repr",
        ]
        if c in joined.columns
    ]
    joined = joined.drop(columns=drop_cols)

    # Align numeric conditioned-DINO rows by the explicit R10K2A state-row
    # metadata index, not by morphology row order.
    order = joined["_state_row"].to_numpy(int)
    cond_aligned = cond[order]

    # Image CLS is case-level and mapped by exact normalized sample_id.
    cls_aligned = np.stack([
        cls_map[sid]
        for sid in joined["_sid"].tolist()
    ]).astype(np.float32)

    if not np.isfinite(cond_aligned).all():
        raise AssertionError("Non-finite conditioned DINO features.")
    if not np.isfinite(cls_aligned).all():
        raise AssertionError("Non-finite CLS features.")

    return joined.reset_index(drop=True), cls_aligned, cond_aligned

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--image_features",
        default=DEFAULT_IMAGE_FEATURES,
    )
    ap.add_argument(
        "--state_features",
        default=DEFAULT_STATE_FEATURES,
    )
    ap.add_argument(
        "--image_manifest",
        default=DEFAULT_IMAGE_MANIFEST,
    )
    ap.add_argument(
        "--state_metadata",
        default=DEFAULT_STATE_METADATA,
    )
    ap.add_argument(
        "--morphology",
        default=DEFAULT_MORPH,
    )
    ap.add_argument(
        "--output_dir",
        default=DEFAULT_OUTPUT,
    )
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    image_feature_path = Path(args.image_features)
    state_feature_path = Path(args.state_features)
    image_manifest_path = Path(args.image_manifest)
    state_metadata_path = Path(args.state_metadata)
    morph_path = Path(args.morphology)

    print("===== R10K2B DINOV2 IMAGE-MASK LOMO EVALUATION FIX2 =====")
    print("STATUS: FROZEN REPRESENTATION EVALUATION")
    print("NPZ OBJECT-METADATA PICKLE LOADING: NO")
    print("IMAGE ID ORDER SOURCE: frozen R05D1 manifest")
    print("STATE ID ORDER SOURCE: R10K2A no-label metadata CSV")
    print("PRIMARY METHOD:", PRIMARY)
    print("METHODS:", METHODS)
    print("PCA COMPONENTS:", PCA_DIM)
    print("PCA DIMENSION SWEEP: NO")
    print("GROUP KEY: sample_id")
    print("OUTER FOLDS:", OUTER_FOLDS)
    print("INNER FOLDS:", INNER_FOLDS)
    print("SOURCE-ONLY THRESHOLD TARGET RECALL:", TARGET_RECALL)
    print("INNER PCA REFIT PER INNER TRAIN SPLIT: YES")
    print("TARGET FEATURES USED TO FIT PCA: NO")
    print("TARGET LABELS USED FOR OUTER SPLIT: NO")
    print("TARGET LABELS USED FOR FITTING: NO")
    print("TARGET LABELS USED FOR THRESHOLD SELECTION: NO")
    print("TARGET LABELS USED FOR HYPERPARAMETER TUNING: NO")
    print("TARGET LABELS USED FOR EVALUATION ONLY: YES")
    print()

    for p in [
        image_feature_path,
        state_feature_path,
        image_manifest_path,
        state_metadata_path,
        morph_path,
    ]:
        print("exists:", p.exists(), p)
        if not p.exists():
            raise FileNotFoundError(p)

    df, cls_all, cond_all = load_locked_features(
        image_feature_path=image_feature_path,
        state_feature_path=state_feature_path,
        image_manifest_path=image_manifest_path,
        state_metadata_path=state_metadata_path,
        morph_path=morph_path,
    )

    y_all = pd.to_numeric(
        df["harm_label"],
        errors="raise",
    ).astype(int).to_numpy()
    groups_all = df["_sid"].to_numpy()
    family_all = df["model_family"].astype(str).to_numpy()

    print("\n===== PANEL AUDIT =====")
    print("Rows:", len(df))
    print("Cases:", df["_sid"].nunique())
    print("Families:")
    print(df["model_family"].value_counts().to_string())
    print("HARM/NON-HARM:", int(y_all.sum()), int((y_all == 0).sum()))
    print("CLS matrix:", cls_all.shape)
    print("Conditioned DINO matrix:", cond_all.shape)

    if len(df) != 9000:
        raise AssertionError("Expected 9000 rows.")
    if df["_sid"].nunique() != 1000:
        raise AssertionError("Expected 1000 cases.")
    if set(df["model_family"].unique()) != set(FAMILIES):
        raise AssertionError("Unexpected family set.")

    per_case_family = (
        df.groupby(["_sid", "model_family"])
        .size()
        .reset_index(name="n")
    )
    if not (per_case_family["n"] == 3).all():
        raise AssertionError("Expected 3 states/case/family.")

    result_rows = []
    fold_rows = []
    prediction_rows = []

    for target_family in FAMILIES:
        source_idx = np.flatnonzero(
            family_all != target_family
        )
        target_idx = np.flatnonzero(
            family_all == target_family
        )

        source = df.iloc[source_idx].reset_index(drop=True)
        target = df.iloc[target_idx].reset_index(drop=True)

        cls_source = cls_all[source_idx]
        cls_target = cls_all[target_idx]
        cond_source = cond_all[source_idx]
        cond_target = cond_all[target_idx]

        y_source = y_all[source_idx]
        y_target = y_all[target_idx]
        g_source = groups_all[source_idx]
        g_target = groups_all[target_idx]

        if set(g_source) != set(g_target):
            raise AssertionError(
                f"Source/target case pools differ: {target_family}"
            )

        print("\n\n============================================================")
        print("TARGET FAMILY:", target_family)
        print(
            "SOURCE FAMILIES:",
            [f for f in FAMILIES if f != target_family],
        )
        print("Source rows:", len(source))
        print("Target rows:", len(target))
        print(
            "Source HARM/NON-HARM:",
            int(y_source.sum()),
            int((y_source == 0).sum()),
        )
        print(
            "Target HARM/NON-HARM:",
            int(y_target.sum()),
            int((y_target == 0).sum()),
        )
        print("============================================================")

        for seed in SEEDS:
            random.seed(seed)
            np.random.seed(seed)

            outer = StratifiedGroupKFold(
                n_splits=OUTER_FOLDS,
                shuffle=True,
                random_state=seed,
            )

            splits = list(
                outer.split(
                    np.zeros((len(source), 1)),
                    y_source,
                    g_source,
                )
            )

            probs = {
                m: np.full(len(target), np.nan, dtype=float)
                for m in METHODS
            }
            thresholds = {
                m: np.full(len(target), np.nan, dtype=float)
                for m in METHODS
            }

            for fold, (tr_s, te_s) in enumerate(splits):
                train_cases = set(g_source[tr_s])
                test_cases = set(g_source[te_s])

                if train_cases & test_cases:
                    raise AssertionError(
                        "Outer source group leakage."
                    )

                te_t = np.flatnonzero(
                    np.isin(
                        g_target,
                        list(test_cases),
                    )
                )

                if set(g_target[te_t]) != test_cases:
                    raise AssertionError(
                        "Held-out target case set mismatch."
                    )

                df_train = source.iloc[tr_s]
                df_target_test = target.iloc[te_t]

                y_train = y_source[tr_s]
                g_train = g_source[tr_s]

                cls_train = cls_source[tr_s]
                cls_target_test = cls_target[te_t]

                cond_train = cond_source[tr_s]
                cond_target_test = cond_target[te_t]

                for method in METHODS:
                    # Strict nested source-only OOF threshold.
                    inner_p = inner_oof_probabilities(
                        method=method,
                        df_outer_train=df_train.reset_index(drop=True),
                        y=y_train,
                        groups=g_train,
                        cls_outer_train=cls_train,
                        cond_outer_train=cond_train,
                        seed=seed + 1000 * fold,
                    )

                    threshold = threshold_for_recall(
                        y_train,
                        inner_p,
                        TARGET_RECALL,
                    )

                    # Final unsupervised transform: OUTER source-train only.
                    Xtr, Xtarget, diag = make_method_matrices(
                        method=method,
                        df_train=df_train,
                        df_eval=df_target_test,
                        cls_train=cls_train,
                        cls_eval=cls_target_test,
                        cond_train=cond_train,
                        cond_eval=cond_target_test,
                        seed=seed + 10000 + fold,
                    )

                    clf = build_lr(
                        seed + 20000 + fold
                    )
                    clf.fit(Xtr, y_train)

                    probs[method][te_t] = (
                        clf.predict_proba(Xtarget)[:, 1]
                    )
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
                        "pca_components": diag[
                            "pca_components"
                        ],
                        "pca_explained_variance": diag[
                            "pca_explained_variance"
                        ],
                        "target_rows_used_to_fit_pca": 0,
                        "source_inner_threshold_R90": threshold,
                    })

            print(
                f"\n######## TARGET={target_family} SEED={seed} ########"
            )

            for method in METHODS:
                if np.isnan(probs[method]).any():
                    raise AssertionError(
                        f"Incomplete probabilities: "
                        f"{target_family}/{seed}/{method}"
                    )
                if np.isnan(thresholds[method]).any():
                    raise AssertionError(
                        f"Incomplete thresholds: "
                        f"{target_family}/{seed}/{method}"
                    )

                p = probs[method]

                auc = float(
                    roc_auc_score(y_target, p)
                )
                auprc = float(
                    average_precision_score(y_target, p)
                )

                pred = (
                    p >= thresholds[method]
                ).astype(int)

                tp = int(
                    np.sum(
                        (pred == 1)
                        & (y_target == 1)
                    )
                )
                fp = int(
                    np.sum(
                        (pred == 1)
                        & (y_target == 0)
                    )
                )
                tn = int(
                    np.sum(
                        (pred == 0)
                        & (y_target == 0)
                    )
                )
                fn = int(
                    np.sum(
                        (pred == 0)
                        & (y_target == 1)
                    )
                )

                recall = tp / (tp + fn)
                fpr = fp / (fp + tn)
                specificity = tn / (tn + fp)
                empirical_ppv = (
                    tp / (tp + fp)
                    if (tp + fp)
                    else np.nan
                )

                pi = PPV_PREVALENCE
                denom = (
                    recall * pi
                    + fpr * (1 - pi)
                )
                ppv1 = (
                    recall * pi / denom
                    if denom > 0
                    else np.nan
                )

                # Target oracle: ranking diagnostic only.
                oracle_thr = threshold_for_recall(
                    y_target,
                    p,
                    TARGET_RECALL,
                )
                oracle = operating_metrics(
                    y_target,
                    p,
                    oracle_thr,
                )

                print(
                    f"{method:27s} "
                    f"AUROC={auc:.6f} "
                    f"AUPRC={auprc:.6f} "
                    f"TRANSFER_RECALL={recall:.6f} "
                    f"TRANSFER_FPR={fpr:.6f} "
                    f"PPV1%={ppv1:.6f} "
                    f"ORACLE_R90_FPR={oracle['FPR']:.6f}"
                )

                result_rows.append({
                    "target_family": target_family,
                    "seed": seed,
                    "method": method,
                    "target_AUROC": auc,
                    "target_AUPRC": auprc,
                    "transfer_Recall": recall,
                    "transfer_FPR": fpr,
                    "transfer_Specificity": specificity,
                    "transfer_EmpiricalPPV": empirical_ppv,
                    "transfer_PPV1pct": ppv1,
                    "oracle_R90_Recall": oracle["Recall"],
                    "oracle_R90_FPR": oracle["FPR"],
                    "oracle_R90_PPV1pct": oracle["PPV1pct"],
                })

                for i in range(len(target)):
                    prediction_rows.append({
                        "target_family": target_family,
                        "seed": seed,
                        "method": method,
                        "sample_id": target.iloc[i]["sample_id"],
                        "model_state_id": target.iloc[i]["model_state_id"],
                        "harm_label": int(y_target[i]),
                        "probability": float(p[i]),
                        "source_only_threshold": float(
                            thresholds[method][i]
                        ),
                    })

    results = pd.DataFrame(result_rows)
    folds = pd.DataFrame(fold_rows)
    preds = pd.DataFrame(prediction_rows)

    by_family = (
        results.groupby(
            ["target_family", "method"],
            sort=False,
        )
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
        results.groupby(
            ["method", "seed"],
            sort=False,
        )
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

    print("\n\n===== R10K2B BY-FAMILY SUMMARY =====")
    print(by_family.to_string(index=False))

    print("\n===== R10K2B MACRO SUMMARY =====")
    print(macro.to_string(index=False))

    base = macro[
        macro["method"] == "M2_backbone"
    ].iloc[0]
    cand = macro[
        macro["method"] == PRIMARY
    ].iloc[0]

    base_family = (
        by_family[
            by_family["method"] == "M2_backbone"
        ]
        .set_index("target_family")
    )
    cand_family = (
        by_family[
            by_family["method"] == PRIMARY
        ]
        .set_index("target_family")
    )

    delta_auc = float(
        cand["macro_AUROC_mean"]
        - base["macro_AUROC_mean"]
    )
    delta_auprc = float(
        cand["macro_AUPRC_mean"]
        - base["macro_AUPRC_mean"]
    )
    delta_fpr = float(
        cand["macro_transfer_FPR_mean"]
        - base["macro_transfer_FPR_mean"]
    )
    delta_oracle = float(
        cand["macro_oracle_R90_FPR_mean"]
        - base["macro_oracle_R90_FPR_mean"]
    )
    worst_auc = float(
        cand_family["target_AUROC_mean"].min()
    )
    worst_delta_auc = float(
        (
            cand_family["target_AUROC_mean"]
            - base_family["target_AUROC_mean"]
        ).min()
    )

    print("\n===== PRIMARY DELTAS: M2+COND-DINO vs M2 =====")
    print("Delta macro AUROC:", delta_auc)
    print("Delta macro AUPRC:", delta_auprc)
    print("Delta transferred FPR:", delta_fpr)
    print("Delta oracle R90 FPR:", delta_oracle)
    print("Worst-family candidate AUROC:", worst_auc)
    print("Worst-family AUROC delta:", worst_delta_auc)

    if (
        delta_auc >= 0.02
        and delta_fpr <= -0.05
        and delta_oracle <= -0.04
        and float(cand["macro_transfer_Recall_mean"]) >= 0.85
        and worst_auc >= 0.75
        and worst_delta_auc >= -0.01
    ):
        decision = "IMAGE_CONDITIONED_REPRESENTATION_STRONGLY_SUPPORTED"
    elif (
        (
            delta_auc >= 0.01
            or delta_fpr <= -0.03
            or delta_oracle <= -0.03
        )
        and float(cand["macro_transfer_Recall_mean"]) >= 0.85
        and worst_delta_auc >= -0.02
    ):
        decision = "IMAGE_CONDITIONED_REPRESENTATION_PARTIALLY_SUPPORTED"
    else:
        decision = "IMAGE_CONDITIONED_REPRESENTATION_NOT_SUPPORTED"

    print("\n===== FINAL DECISION =====")
    print("DECISION:", decision)

    results.to_csv(
        out / "R10K2B_per_seed_family_metrics.csv",
        index=False,
    )
    by_family.to_csv(
        out / "R10K2B_by_family_summary.csv",
        index=False,
    )
    macro.to_csv(
        out / "R10K2B_macro_summary.csv",
        index=False,
    )
    folds.to_csv(
        out / "R10K2B_fold_audit.csv",
        index=False,
    )
    preds.to_csv(
        out / "R10K2B_target_predictions.csv",
        index=False,
    )

    print("\nOutputs:")
    print(out / "R10K2B_per_seed_family_metrics.csv")
    print(out / "R10K2B_by_family_summary.csv")
    print(out / "R10K2B_macro_summary.csv")
    print(out / "R10K2B_fold_audit.csv")
    print(out / "R10K2B_target_predictions.csv")
    print("PASS")


if __name__ == "__main__":
    main()
