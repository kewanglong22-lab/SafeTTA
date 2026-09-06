#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R10L0_final_source_safety_estimator_lock_fix2.py

Purpose
-------
After R10K2D freezes the confirmed method, fit ONE final source safety
estimator on the complete NeoPolyp development cohort, before any independent
external-cohort outcome is inspected.

Frozen method
-------------
Primary = M2_plus_CondDINO_PCA64

Input per model-case:
- M2 morphology:
    morph_fg_fraction
    morph_boundary_density
- frozen R10K2A conditioned DINO:
    foreground_patch_mean 768-D
    background_patch_mean 768-D
  concatenated -> 1536-D
- PCA64, whiten=True
- concatenate M2 + PCA64 -> 66-D
- median imputation + standard scaling + class-balanced logistic regression

Frozen operating point
----------------------
Five grouped OOF runs, seeds 20260816..20260820:
- group = sample_id
- 5-fold StratifiedGroupKFold
- PCA is fit INSIDE each training fold only
- LR is fit inside training fold only
- one OOF threshold per seed for Recall >= 0.90
- FINAL FROZEN THRESHOLD = median of the five seed-specific OOF thresholds

Then:
- fit final PCA64 on all 9000 NeoPolyp development rows
- fit final safety head on all 9000 rows
- serialize both artifacts

Strict boundary
---------------
- no external cohort is read
- no target/external labels are read
- no method tuning
- no PCA dimension sweep
- no classifier sweep
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score


PCA_DIM = 64
TARGET_RECALL = 0.90
SEEDS = [20260816, 20260817, 20260818, 20260819, 20260820]
FOLDS = 5

M2 = [
    "morph_fg_fraction",
    "morph_boundary_density",
]

DEFAULT_K2D_LOCK = (
    "F:/MEDSEG_SAFETTA/outputs/"
    "Q1_R10K2D_confirmed_method_freeze_fix1_v1/"
    "R10K2D_CONFIRMED_METHOD_LOCK.json"
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

DEFAULT_STATE_FEATURES = (
    "F:/MEDSEG_SAFETTA/outputs/"
    "Q1_R10K2A_dinov2_image_mask_representation_lock_fix1_v1/"
    "R10K2A_mask_conditioned_dinov2_features.npz"
)

DEFAULT_MORPH = (
    "F:/MEDSEG_SAFETTA/outputs/"
    "Q1_R10J1_source_mask_morphology_feature_builder_fix1_v1/"
    "source_mask_morphology_features_labeled.csv"
)

DEFAULT_OUTPUT = (
    "F:/MEDSEG_SAFETTA/outputs/"
    "Q1_R10L0_final_source_safety_estimator_lock_fix2_v1"
)


def norm_sid_series(s):
    return (
        s.astype(str)
        .str.strip()
        .str.replace("\\", "/", regex=False)
        .str.lower()
    )


def build_head(seed):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=seed,
        )),
    ])


def threshold_for_recall(y, p, target):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)

    pos = np.sort(p[y == 1])[::-1]
    if len(pos) == 0:
        raise AssertionError("No positive rows.")

    k = int(np.ceil(target * len(pos)))
    k = min(max(k, 1), len(pos))
    return float(pos[k - 1])


def op_metrics(y, p, threshold):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    pred = (p >= threshold).astype(int)

    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))

    recall = tp / (tp + fn) if tp + fn else np.nan
    fpr = fp / (fp + tn) if fp + tn else np.nan

    return {
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "Recall": float(recall),
        "FPR": float(fpr),
    }


def load_development_panel(
    state_metadata_path,
    state_features_path,
    morph_path,
):
    # Numeric conditioned DINO only; no pickle metadata reads.
    with np.load(state_features_path, allow_pickle=False) as z:
        required = {
            "foreground_patch_mean",
            "background_patch_mean",
        }
        if not required.issubset(z.files):
            raise AssertionError(
                f"Missing numeric DINO keys. Available={z.files}"
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

    cond = np.concatenate([fg, bg], axis=1).astype(np.float32)

    state_meta = pd.read_csv(
        state_metadata_path,
        low_memory=False,
    )
    required_state = [
        "sample_id",
        "model_family",
        "model_state_id",
        "training_seed",
        "checkpoint_sha256",
    ]
    missing = [c for c in required_state if c not in state_meta.columns]
    if missing:
        raise AssertionError(f"State metadata missing: {missing}")

    state_meta["_sid"] = norm_sid_series(
        state_meta["sample_id"]
    )
    state_meta["_state_row"] = np.arange(
        len(state_meta),
        dtype=int,
    )

    if len(state_meta) != 9000:
        raise AssertionError("State metadata rows != 9000.")
    if (
        state_meta[["_sid", "model_state_id"]]
        .drop_duplicates()
        .shape[0]
        != 9000
    ):
        raise AssertionError(
            "State metadata explicit key not unique."
        )

    morph = pd.read_csv(morph_path, low_memory=False)
    required_morph = [
        "sample_id",
        "model_family",
        "model_state_id",
        "harm_label",
    ] + M2
    missing = [c for c in required_morph if c not in morph.columns]
    if missing:
        raise AssertionError(f"Morphology table missing: {missing}")

    morph["_sid"] = norm_sid_series(morph["sample_id"])

    if len(morph) != 9000:
        raise AssertionError("Morphology rows != 9000.")
    if (
        morph[["_sid", "model_state_id"]]
        .drop_duplicates()
        .shape[0]
        != 9000
    ):
        raise AssertionError(
            "Morphology explicit key not unique."
        )

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
        raise AssertionError("Development join != 9000 rows.")

    fam_ok = (
        joined["model_family_morph"].astype(str)
        == joined["model_family_repr"].astype(str)
    )
    if not fam_ok.all():
        raise AssertionError("Model-family mismatch.")

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

    cond_aligned = cond[
        joined["_state_row"].to_numpy(int)
    ]

    if not np.isfinite(cond_aligned).all():
        raise AssertionError("Non-finite DINO features.")

    return joined.reset_index(drop=True), cond_aligned


def make_final_input(m2, z64):
    m2 = np.asarray(m2, dtype=np.float64)
    z64 = np.asarray(z64, dtype=np.float64)
    return np.column_stack([m2, z64])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k2d_lock", default=DEFAULT_K2D_LOCK)
    ap.add_argument(
        "--image_manifest",
        default=DEFAULT_IMAGE_MANIFEST,
    )
    ap.add_argument(
        "--state_metadata",
        default=DEFAULT_STATE_METADATA,
    )
    ap.add_argument(
        "--state_features",
        default=DEFAULT_STATE_FEATURES,
    )
    ap.add_argument("--morphology", default=DEFAULT_MORPH)
    ap.add_argument("--output_dir", default=DEFAULT_OUTPUT)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths = {
        "k2d_lock": Path(args.k2d_lock),
        "image_manifest": Path(args.image_manifest),
        "state_metadata": Path(args.state_metadata),
        "state_features": Path(args.state_features),
        "morphology": Path(args.morphology),
    }

    print("===== R10L0 FINAL SOURCE SAFETY ESTIMATOR LOCK FIX2 =====")
    print("STATUS: FINAL DEVELOPMENT-COHORT ESTIMATOR LOCK")
    print("PRIMARY METHOD: M2_plus_CondDINO_PCA64")
    print("PCA DIMENSION:", PCA_DIM)
    print("PCA DIMENSION SWEEP: NO")
    print("CLASSIFIER SWEEP: NO")
    print("EXTERNAL COHORT READ: NO")
    print("EXTERNAL LABEL READ: NO")
    print("TARGET/EXTERNAL TUNING: NO")
    print("GROUP KEY: sample_id")
    print("OOF SEEDS:", SEEDS)
    print("OOF FOLDS:", FOLDS)
    print("FINAL THRESHOLD RULE: median of 5 seed-specific grouped OOF R90 thresholds")
    print()

    for name, p in paths.items():
        print(name, "exists:", p.exists(), p)
        if not p.exists():
            raise FileNotFoundError(p)

    # K2D method lock check.
    with open(paths["k2d_lock"], "r", encoding="utf-8") as f:
        lock = json.load(f)

    if lock.get("decision") != (
        "R10K2_PRIMARY_METHOD_FROZEN_FOR_EXTERNAL_VALIDATION"
    ):
        raise AssertionError(
            "R10K2D method lock decision is not the expected frozen state."
        )

    # Image manifest is audited only for 1000-case lineage consistency.
    img = pd.read_csv(paths["image_manifest"], low_memory=False)
    if "sample_id" not in img.columns:
        raise AssertionError("Image manifest missing sample_id.")
    img["_sid"] = norm_sid_series(img["sample_id"])
    if len(img) != 1000 or img["_sid"].nunique() != 1000:
        raise AssertionError("Expected 1000 unique development images.")

    df, cond = load_development_panel(
        state_metadata_path=paths["state_metadata"],
        state_features_path=paths["state_features"],
        morph_path=paths["morphology"],
    )

    if set(img["_sid"]) != set(df["_sid"]):
        raise AssertionError(
            "Image-manifest and development-panel case sets differ."
        )

    y = pd.to_numeric(
        df["harm_label"],
        errors="raise",
    ).astype(int).to_numpy()
    groups = df["_sid"].to_numpy()
    m2 = np.column_stack([
        pd.to_numeric(df[c], errors="raise").to_numpy(float)
        for c in M2
    ])

    print("\n===== DEVELOPMENT PANEL =====")
    print("Rows:", len(df))
    print("Cases:", df["_sid"].nunique())
    print("Families:")
    print(df["model_family"].value_counts().to_string())
    print("HARM/NON-HARM:", int(y.sum()), int((y == 0).sum()))
    print("CondDINO:", cond.shape)
    print("M2:", m2.shape)

    # --------------------------------------------------------------
    # Fully nested grouped OOF solely to freeze source operating threshold.
    # --------------------------------------------------------------
    threshold_rows = []
    oof_metric_rows = []

    for seed in SEEDS:
        cv = StratifiedGroupKFold(
            n_splits=FOLDS,
            shuffle=True,
            random_state=seed,
        )

        p_oof = np.full(len(y), np.nan, dtype=float)

        for fold, (tr, va) in enumerate(
            cv.split(
                np.zeros((len(y), 1)),
                y,
                groups,
            )
        ):
            if set(groups[tr]) & set(groups[va]):
                raise AssertionError("Grouped OOF leakage.")

            pca = PCA(
                n_components=PCA_DIM,
                svd_solver="randomized",
                whiten=True,
                random_state=seed + fold,
            )
            ztr = pca.fit_transform(cond[tr])
            zva = pca.transform(cond[va])

            Xtr = make_final_input(m2[tr], ztr)
            Xva = make_final_input(m2[va], zva)

            head = build_head(seed + 1000 + fold)
            head.fit(Xtr, y[tr])

            p_oof[va] = head.predict_proba(Xva)[:, 1]

        if np.isnan(p_oof).any():
            raise AssertionError("Incomplete grouped OOF probabilities.")

        threshold = threshold_for_recall(
            y,
            p_oof,
            TARGET_RECALL,
        )
        op = op_metrics(y, p_oof, threshold)

        auc = float(roc_auc_score(y, p_oof))
        auprc = float(average_precision_score(y, p_oof))

        threshold_rows.append({
            "seed": seed,
            "threshold_R90": threshold,
            "OOF_AUROC": auc,
            "OOF_AUPRC": auprc,
            "OOF_Recall": op["Recall"],
            "OOF_FPR": op["FPR"],
        })
        oof_metric_rows.append({
            "seed": seed,
            "AUROC": auc,
            "AUPRC": auprc,
            "Recall": op["Recall"],
            "FPR": op["FPR"],
        })

        print(
            f"seed={seed} "
            f"threshold={threshold:.12f} "
            f"AUROC={auc:.6f} "
            f"AUPRC={auprc:.6f} "
            f"Recall={op['Recall']:.6f} "
            f"FPR={op['FPR']:.6f}"
        )

    thresholds = pd.DataFrame(threshold_rows)
    final_threshold = float(
        np.median(thresholds["threshold_R90"].to_numpy(float))
    )

    print("\nSeed-specific thresholds:")
    print(thresholds.to_string(index=False))
    print("\nFINAL FROZEN THRESHOLD:", final_threshold)

    # --------------------------------------------------------------
    # Final estimator fit on all 9000 development rows.
    # --------------------------------------------------------------
    final_pca = PCA(
        n_components=PCA_DIM,
        svd_solver="randomized",
        whiten=True,
        random_state=SEEDS[-1],
    )
    z_all = final_pca.fit_transform(cond)

    X_all = make_final_input(m2, z_all)

    final_head = build_head(SEEDS[-1])
    final_head.fit(X_all, y)

    p_all = final_head.predict_proba(X_all)[:, 1]

    print("\n===== FINAL FIT AUDIT =====")
    print("PCA input:", cond.shape)
    print("PCA output:", z_all.shape)
    print(
        "PCA explained variance ratio sum:",
        float(final_pca.explained_variance_ratio_.sum()),
    )
    print("Final safety input:", X_all.shape)
    print("Final train probability finite:", bool(np.isfinite(p_all).all()))

    if z_all.shape != (9000, PCA_DIM):
        raise AssertionError("Final PCA output shape invalid.")
    if X_all.shape != (9000, PCA_DIM + 2):
        raise AssertionError("Final safety input shape invalid.")
    if not np.isfinite(p_all).all():
        raise AssertionError("Non-finite final probabilities.")

    pca_path = out / "R10L0_final_condDINO_PCA64.joblib"
    head_path = out / "R10L0_final_safety_head.joblib"

    joblib.dump(final_pca, pca_path)
    joblib.dump(final_head, head_path)

    # Round-trip artifact verification.
    #
    # IMPORTANT:
    # sklearn PCA.fit_transform(X) and PCA.transform(X) can differ by tiny
    # floating-point amounts for randomized PCA + whitening.  Therefore the
    # correct serialization round-trip reference is the ORIGINAL fitted
    # estimator's transform() path, because that is exactly the path used for
    # every future external cohort.  Comparing reload.transform() against the
    # earlier fit_transform() training matrix is unnecessarily strict and can
    # trigger a false failure without any artifact corruption.
    pca_reload = joblib.load(pca_path)
    head_reload = joblib.load(head_path)

    z_direct = final_pca.transform(cond[:32])
    X_direct = make_final_input(m2[:32], z_direct)
    p_direct = final_head.predict_proba(X_direct)[:, 1]

    z_reload = pca_reload.transform(cond[:32])
    X_reload = make_final_input(m2[:32], z_reload)
    p_reload = head_reload.predict_proba(X_reload)[:, 1]

    roundtrip_max_abs_diff = float(
        np.max(np.abs(p_reload - p_direct))
    )
    fittransform_vs_transform_max_abs_diff = float(
        np.max(np.abs(p_all[:32] - p_direct))
    )

    print(
        "Serialized round-trip probability max abs diff:",
        roundtrip_max_abs_diff,
    )
    print(
        "fit_transform vs transform probability max abs diff:",
        fittransform_vs_transform_max_abs_diff,
    )

    if not np.allclose(
        p_reload,
        p_direct,
        rtol=1e-10,
        atol=1e-12,
    ):
        raise AssertionError(
            "Serialized estimator round-trip mismatch on transform() path."
        )

    threshold_path = out / "R10L0_frozen_operating_threshold.json"
    with open(threshold_path, "w", encoding="utf-8") as f:
        json.dump({
            "target_recall": TARGET_RECALL,
            "seed_specific_thresholds": {
                str(int(r["seed"])): float(r["threshold_R90"])
                for _, r in thresholds.iterrows()
            },
            "aggregation": "median",
            "final_threshold": final_threshold,
            "external_labels_used": False,
        }, f, indent=2)

    thresholds.to_csv(
        out / "R10L0_grouped_OOF_thresholds.csv",
        index=False,
    )

    lock_out = {
        "status": "FROZEN",
        "decision": "FINAL_SOURCE_ESTIMATOR_LOCKED_FOR_INDEPENDENT_EXTERNAL_VALIDATION",
        "development_dataset": "NeoPolyp",
        "development_rows": 9000,
        "development_cases": 1000,
        "model_families": [
            "DeepLabV3-R50",
            "PraNet",
            "SegFormer-B0",
        ],
        "primary_method": "M2_plus_CondDINO_PCA64",
        "representation": {
            "encoder": "facebook/dinov2-base",
            "conditioned_semantics": (
                "foreground/background weighted patch means"
            ),
            "conditioned_dimension": 1536,
            "pca_dimension": 64,
            "m2_features": M2,
            "final_input_dimension": 66,
        },
        "final_fit": {
            "pca_fit_rows": 9000,
            "head_fit_rows": 9000,
            "external_rows_used": 0,
            "external_labels_used": False,
            "serialized_roundtrip_max_abs_probability_diff": (
                roundtrip_max_abs_diff
            ),
            "fittransform_vs_transform_max_abs_probability_diff": (
                fittransform_vs_transform_max_abs_diff
            ),
        },
        "operating_point": {
            "target_recall": TARGET_RECALL,
            "selection": (
                "5-seed 5-fold StratifiedGroupKFold OOF by sample_id; "
                "median seed-specific threshold"
            ),
            "final_threshold": final_threshold,
        },
        "artifacts": {
            "pca": str(pca_path),
            "safety_head": str(head_path),
            "threshold": str(threshold_path),
        },
        "post_lock_rules": [
            "Do not refit PCA on external-cohort features.",
            "Do not refit or recalibrate the safety head on external labels.",
            "Do not alter PCA dimension.",
            "Do not alter M2 features.",
            "Do not change DINOv2 or semantic pooling.",
            "Do not alter the frozen operating threshold after external outcomes are revealed.",
        ],
    }

    lock_path = out / "R10L0_FINAL_SOURCE_ESTIMATOR_LOCK.json"
    with open(lock_path, "w", encoding="utf-8") as f:
        json.dump(lock_out, f, indent=2, ensure_ascii=False)

    print("\n===== FINAL DECISION =====")
    print(
        "DECISION: "
        "FINAL_SOURCE_ESTIMATOR_LOCKED_FOR_INDEPENDENT_EXTERNAL_VALIDATION"
    )
    print("EXTERNAL DATA USED IN FIT: NO")
    print("EXTERNAL LABELS USED: NO")
    print("METHOD CHANGES AFTER THIS POINT: FORBIDDEN")
    print("Frozen threshold:", final_threshold)
    print("\nOutputs:")
    print(pca_path)
    print(head_path)
    print(threshold_path)
    print(lock_path)
    print("PASS")


if __name__ == "__main__":
    main()
