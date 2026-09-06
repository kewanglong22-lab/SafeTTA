#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1X_CM4C_final_source_safety_estimator_and_threshold_lock_fix1.py

SafeTTA Q1 enhancement — CM4C.

Purpose
-------
Freeze the final MRI SOURCE safety estimator and the SOURCE operating point
before any PROMISE12 safety inference/evaluation.

Inputs are SOURCE only:
- CM4B prediction-conditioned DINO+M2 feature asset;
- CM4B patient-grouped OOF Final scores;
- CM4A SOURCE TENT1 HARM labels.

This stage:
1) derives the frozen SOURCE operating point from cross-fitted OOF Final scores;
2) fits the final all-SOURCE prediction-conditioned safety estimator;
3) serializes all source-only preprocessing/classifier components;
4) locks the DINOv2 identity and score semantics for future target scoring.

No PROMISE12 access.
No target calibration.
No target GT.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


VERSION = "2026-09-05-Q1X-CM4C-v1-fix2"
BUILD = "Q1X_CM4C_FINAL_SOURCE_SAFETY_ESTIMATOR_AND_THRESHOLD_LOCK_FIX2"

ROOT = Path(r"F:\MEDSEG_SAFETTA")
OUT = ROOT / "outputs"
CODE = ROOT / "code"

CM4A_DIR = (
    OUT
    / "Q1X_CM4A_source_oof_tent1_outcome_asset_lock_fix4_v1"
)
CM4A_LOCK = CM4A_DIR / "CM4A_SOURCE_TENT1_OUTCOME_LOCK.json"
EXPECTED_CM4A_LOCK_SHA = (
    "7230c3eacf5e25b70e91ebdeb274fe4bc29b86c6e922f5350f69e5818569f4cb"
)

CM4B_DIR = (
    OUT
    / "Q1X_CM4B_mri_source_prediction_conditioned_safety_grouped_oof_fix1_v1"
)
CM4B_LOCK = CM4B_DIR / "CM4B_MRI_SOURCE_SAFETY_OOF_LOCK.json"
EXPECTED_CM4B_LOCK_SHA = (
    "7903f8dd84338be8051883b1a905607daec25e55539205f7c6fb78ca3cc41a7e"
)

CM4B_HELPER = (
    CODE
    / "Q1X_CM4B_mri_source_prediction_conditioned_safety_grouped_oof_fix1.py"
)
EXPECTED_CM4B_HELPER_SHA = (
    "0b894a2db0d58dda6fa9fe26bf0b1482a3da89d25918405e45e39346d132cc40"
)

DEFAULT_OUTPUT = (
    OUT
    / "Q1X_CM4C_final_source_safety_estimator_and_threshold_lock_fix2_v1"
)

FAMILIES = ["UNet", "DeepLabV3_R50", "SegFormer_B0"]
N_SLICES = 3553
N_PATIENTS = 139
EXPECTED_ROWS = N_SLICES * len(FAMILIES)

COND_DIM = 1536
FINAL_DIM = 1538
PCA_COMPONENTS = 64
SEED = 20260904
LR_MAX_ITER = 3000

HARM_RECALL_TARGET = 0.90

# Persistence numerical self-test only. This is NOT a decision-threshold
# tolerance and is not used to alter HARM classification or source operating
# point. The Fix1 observed reload drift was ~2.96e-7.
RELOAD_SCORE_ATOL = 1e-6

PASS_DECISION = (
    "FINAL_MRI_SOURCE_SAFETY_ESTIMATOR_AND_THRESHOLD_LOCKED_"
    "READY_FOR_CM5_PROMISE12_GT_FREE_INFERENCE"
)


def sha256_file(path: Path, chunk=8 * 1024 * 1024):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, obj):
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def verify_lineage():
    for label, path, expected in [
        ("CM4A_LOCK", CM4A_LOCK, EXPECTED_CM4A_LOCK_SHA),
        ("CM4B_LOCK", CM4B_LOCK, EXPECTED_CM4B_LOCK_SHA),
        ("CM4B_HELPER", CM4B_HELPER, EXPECTED_CM4B_HELPER_SHA),
    ]:
        if not path.is_file():
            raise FileNotFoundError(path)
        got = sha256_file(path)
        print(label, got, "PASS" if got == expected else "FAIL")
        if got != expected:
            raise RuntimeError(f"{label} SHA mismatch.")

    cm4a = load_json(CM4A_LOCK)
    cm4b = load_json(CM4B_LOCK)

    if cm4a.get("status") != "PASS":
        raise RuntimeError("CM4A status changed.")
    if cm4b.get("status") != "PASS":
        raise RuntimeError("CM4B status changed.")
    if cm4b.get("decision") != (
        "MRI_SOURCE_HARM_RANKING_SUPPORTED_READY_FOR_CM4C_FINAL_SOURCE_ESTIMATOR_LOCK"
    ):
        raise RuntimeError("Unexpected CM4B decision.")

    if cm4b.get("promises12_access") is not False:
        raise RuntimeError("CM4B information boundary changed.")
    if cm4b.get("threshold_selection") is not False:
        raise RuntimeError("CM4B unexpectedly selected a threshold.")
    if cm4b.get("final_all_source_head_fitting") is not False:
        raise RuntimeError("CM4B unexpectedly fit final source head.")

    return cm4a, cm4b


def load_source_oof_scores(cm4b):
    path = Path(cm4b["artifacts"]["scores"])
    expected_sha = cm4b["artifacts"]["scores_sha256"]

    if not path.is_file():
        raise FileNotFoundError(path)
    if sha256_file(path) != expected_sha:
        raise RuntimeError("CM4B OOF-score SHA mismatch.")

    df = pd.read_csv(path)

    required = {
        "family",
        "fold",
        "global_index",
        "case_key",
        "harmful",
        "delta_dice",
        "score_Final",
    }
    missing = required.difference(df.columns)
    if missing:
        raise RuntimeError(f"Missing OOF score columns: {sorted(missing)}")

    if len(df) != EXPECTED_ROWS:
        raise RuntimeError(f"Unexpected OOF rows: {len(df)}")
    if df["case_key"].nunique() != N_PATIENTS:
        raise RuntimeError("Unexpected SOURCE patient count.")
    if sorted(df["family"].unique().tolist()) != sorted(FAMILIES):
        raise RuntimeError("Unexpected family set in OOF scores.")

    if not np.isfinite(df["score_Final"].to_numpy(dtype=np.float64)).all():
        raise RuntimeError("Nonfinite Final OOF score.")
    if not set(df["harmful"].astype(int).unique()).issubset({0, 1}):
        raise RuntimeError("Invalid HARM labels.")

    return df


def choose_high_recall_threshold(df):
    """
    High score = higher predicted HARM risk.
    Deployment rule:
      score >= tau -> retain SOURCE / skip adaptation
      score <  tau -> adapt

    Choose the largest OOF score threshold whose pooled SOURCE HARM recall
    remains >= 0.90. This maximizes SOURCE adaptation coverage under the
    frozen high-recall constraint on cross-fitted predictions.
    """
    y = df["harmful"].to_numpy(dtype=np.int64)
    s = df["score_Final"].to_numpy(dtype=np.float64)

    if int(y.sum()) <= 0:
        raise RuntimeError("No SOURCE HARM events.")

    candidates = np.unique(s)
    best = None

    for tau in candidates:
        pred_harm = s >= tau
        tp = int(np.sum((y == 1) & pred_harm))
        fn = int(np.sum((y == 1) & (~pred_harm)))
        recall = tp / (tp + fn)

        if recall + 1e-15 >= HARM_RECALL_TARGET:
            best = float(tau)

    if best is None:
        raise RuntimeError("No threshold meets HARM recall target.")

    tau = best
    high = s >= tau
    low = ~high

    tp = int(np.sum((y == 1) & high))
    fn = int(np.sum((y == 1) & low))
    fp = int(np.sum((y == 0) & high))
    tn = int(np.sum((y == 0) & low))

    recall = tp / (tp + fn)
    fpr = fp / (fp + tn)
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    coverage = float(np.mean(low))
    prevented_harm = recall

    per_family = {}
    for family in FAMILIES:
        m = df["family"].to_numpy() == family
        yf = y[m]
        sf = s[m]
        hf = sf >= tau

        tpf = int(np.sum((yf == 1) & hf))
        fnf = int(np.sum((yf == 1) & (~hf)))
        fpf = int(np.sum((yf == 0) & hf))
        tnf = int(np.sum((yf == 0) & (~hf)))

        per_family[family] = {
            "rows": int(m.sum()),
            "harm_count": int(yf.sum()),
            "recall": (
                tpf / (tpf + fnf)
                if (tpf + fnf)
                else float("nan")
            ),
            "fpr": (
                fpf / (fpf + tnf)
                if (fpf + tnf)
                else float("nan")
            ),
            "adaptation_coverage": float(np.mean(~hf)),
        }

    return {
        "threshold": tau,
        "rule": "score >= threshold => retain SOURCE; score < threshold => adapt",
        "target_harm_recall": HARM_RECALL_TARGET,
        "pooled": {
            "tp": tp,
            "fn": fn,
            "fp": fp,
            "tn": tn,
            "harm_recall": recall,
            "fpr": fpr,
            "precision": precision,
            "adaptation_coverage": coverage,
            "prevented_harm_fraction": prevented_harm,
        },
        "per_family": per_family,
        "selection_data": "patient-grouped cross-fitted SOURCE OOF scores only",
    }


def map_oof_operating_point_to_refit_score_scale(
    refit_scores,
    oof_threshold_info,
):
    """
    Transport the already label-derived OOF operating point onto the score
    scale of the final all-SOURCE refit estimator WITHOUT using in-sample
    labels for threshold selection.

    The invariant carried from OOF is the SOURCE adaptation coverage selected
    under the cross-fitted HARM-recall constraint.

    If OOF allowed exactly k low-risk rows to adapt, choose a refit threshold
    between the k-th and (k+1)-th sorted refit scores whenever possible.

    This is source score-scale alignment, not target calibration.
    """
    s = np.asarray(refit_scores, dtype=np.float64)
    if s.ndim != 1 or not np.isfinite(s).all():
        raise RuntimeError("Invalid refit score vector.")

    n = len(s)
    target_coverage = float(
        oof_threshold_info["pooled"]["adaptation_coverage"]
    )
    target_low_count = int(round(target_coverage * n))

    if target_low_count < 0 or target_low_count > n:
        raise RuntimeError("Invalid target low-risk count.")

    ordered = np.sort(s)

    if target_low_count == 0:
        tau = float(ordered[0])
    elif target_low_count == n:
        tau = float(np.nextafter(ordered[-1], np.inf))
    else:
        lo = float(ordered[target_low_count - 1])
        hi = float(ordered[target_low_count])

        if lo < hi:
            tau = float(lo + 0.5 * (hi - lo))
        else:
            # Continuous LR probabilities should very rarely tie exactly.
            # Choose the smallest representable value above the tie and audit
            # the achieved coverage explicitly rather than silently assuming
            # exact coverage.
            tau = float(np.nextafter(hi, np.inf))

    achieved_low = s < tau
    achieved_count = int(achieved_low.sum())
    achieved_coverage = float(np.mean(achieved_low))

    return {
        "threshold": tau,
        "rule": "score >= threshold => retain SOURCE; score < threshold => adapt",
        "mapping": (
            "OOF cross-fitted operating-point coverage -> "
            "all-SOURCE-refit score quantile"
        ),
        "uses_in_sample_labels_for_mapping": False,
        "oof_reference_threshold": float(
            oof_threshold_info["threshold"]
        ),
        "oof_reference_adaptation_coverage": target_coverage,
        "target_low_risk_count": target_low_count,
        "achieved_low_risk_count": achieved_count,
        "achieved_adaptation_coverage": achieved_coverage,
        "coverage_abs_error": abs(
            achieved_coverage - target_coverage
        ),
    }


def load_final_features(cm4b, score_df):
    meta_path = Path(cm4b["artifacts"]["feature_meta"])
    if not meta_path.is_file():
        raise FileNotFoundError(meta_path)
    if sha256_file(meta_path) != cm4b["artifacts"]["feature_meta_sha256"]:
        raise RuntimeError("CM4B feature-meta SHA mismatch.")

    meta = load_json(meta_path)
    feature_path = Path(meta["feature_file"])

    if not feature_path.is_file():
        raise FileNotFoundError(feature_path)
    if sha256_file(feature_path) != meta["feature_sha256"]:
        raise RuntimeError("CM4B feature SHA mismatch.")

    features = np.load(feature_path, mmap_mode="r")
    expected_shape = (len(FAMILIES), N_SLICES, FINAL_DIM)

    if tuple(features.shape) != expected_shape:
        raise RuntimeError(
            f"Unexpected feature shape {features.shape} != {expected_shape}"
        )

    family_to_idx = {f: i for i, f in enumerate(FAMILIES)}
    X = np.empty((len(score_df), FINAL_DIM), dtype=np.float32)

    for family in FAMILIES:
        m = score_df["family"].to_numpy() == family
        row_idx = np.flatnonzero(m)
        gi = score_df.loc[m, "global_index"].to_numpy(dtype=np.int64)

        X[row_idx] = np.asarray(
            features[family_to_idx[family], gi, :],
            dtype=np.float32,
        )

    if not np.isfinite(X).all():
        raise RuntimeError("Nonfinite final SOURCE features.")

    return X, meta, feature_path


def fit_final_source_estimator(X, y):
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    cond_imputer = SimpleImputer(strategy="median")
    X_cond = cond_imputer.fit_transform(X[:, :COND_DIM])

    pca = PCA(
        n_components=PCA_COMPONENTS,
        whiten=True,
        svd_solver="randomized",
        random_state=SEED,
    )
    X_pca = pca.fit_transform(X_cond)

    m2_imputer = SimpleImputer(strategy="median")
    X_m2 = m2_imputer.fit_transform(X[:, COND_DIM:FINAL_DIM])

    X_final = np.concatenate([X_pca, X_m2], axis=1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_final)

    clf = LogisticRegression(
        class_weight="balanced",
        max_iter=LR_MAX_ITER,
        random_state=SEED,
    )
    clf.fit(X_scaled, y)

    bundle = {
        "version": VERSION,
        "representation": "CondDINO_FG768_BG768_PCA64_plus_M2",
        "cond_dim": COND_DIM,
        "final_raw_dim": FINAL_DIM,
        "pca_components": PCA_COMPONENTS,
        "cond_imputer": cond_imputer,
        "pca": pca,
        "m2_imputer": m2_imputer,
        "scaler": scaler,
        "classifier": clf,
        "score_semantics": "predict_proba[:,1] = P(HARM)",
        "harm_definition": "DeltaDice <= -0.02",
        "source_only_fit": True,
        "promises12_access": False,
    }

    return bundle, X_scaled


def score_bundle(bundle, X):
    cond = bundle["cond_imputer"].transform(X[:, :COND_DIM])
    pca = bundle["pca"].transform(cond)

    m2 = bundle["m2_imputer"].transform(X[:, COND_DIM:FINAL_DIM])

    final = np.concatenate([pca, m2], axis=1)
    scaled = bundle["scaler"].transform(final)

    return bundle["classifier"].predict_proba(scaled)[:, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    args = ap.parse_args()

    print("===== Q1X CM4C FIX2 FINAL SOURCE SAFETY ESTIMATOR LOCK =====")
    print("SOURCE_ONLY=YES")
    print("PROMISE12_ACCESS=NO")
    print("TARGET_GT=NO")
    print("TARGET_CALIBRATION=NO")
    print("OOF_THRESHOLD_TARGET_HARM_RECALL=", HARM_RECALL_TARGET)
    print("FINAL_HEAD=all SOURCE family-slice rows")
    print("OOF_THRESHOLD_RAW_SCORE_APPLIED_TO_REFIT=NO")
    print("REFIT_SCORE_SCALE_ALIGNMENT=OOF_SELECTED_SOURCE_COVERAGE_QUANTILE")
    print("RELOAD_SCORE_ATOL=", RELOAD_SCORE_ATOL)
    print("REPRESENTATION=CondDINO_PCA64_plus_M2")
    print("CLASSIFIER=balanced LogisticRegression")

    cm4a, cm4b = verify_lineage()
    score_df = load_source_oof_scores(cm4b)

    print("\n===== SOURCE OOF OPERATING-POINT LOCK =====")
    threshold_info = choose_high_recall_threshold(score_df)

    print("threshold:", threshold_info["threshold"])
    print("pooled:", threshold_info["pooled"])
    print("per-family:")
    for family, vals in threshold_info["per_family"].items():
        print(" ", family, vals)

    if (
        threshold_info["pooled"]["harm_recall"] + 1e-15
        < HARM_RECALL_TARGET
    ):
        raise RuntimeError("Frozen OOF threshold missed recall target.")

    X, feature_meta, feature_path = load_final_features(cm4b, score_df)
    y = score_df["harmful"].to_numpy(dtype=np.int64)

    print("\n===== FIT FINAL ALL-SOURCE SAFETY ESTIMATOR =====")
    print("rows:", len(X))
    print("patients:", score_df["case_key"].nunique())
    print("HARM:", int(y.sum()))
    print("raw feature dim:", X.shape[1])

    bundle, X_scaled = fit_final_source_estimator(X, y)
    final_source_scores = score_bundle(bundle, X)

    if not np.isfinite(final_source_scores).all():
        raise RuntimeError("Nonfinite final all-source safety scores.")

    model_path = args.output_dir / "CM4C_FINAL_SOURCE_SAFETY_ESTIMATOR.joblib"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path, compress=3)

    # Reload-and-score self-test.
    loaded = joblib.load(model_path)
    reloaded_scores = score_bundle(loaded, X)

    max_reload_diff = float(
        np.max(np.abs(final_source_scores - reloaded_scores))
    )
    if max_reload_diff > RELOAD_SCORE_ATOL:
        raise RuntimeError(
            "Final estimator reload numerical drift exceeded tolerance: "
            f"{max_reload_diff} > {RELOAD_SCORE_ATOL}"
        )

    # Important: the OOF threshold lives on cross-fitted model probability
    # scales. The final refit estimator is a different fitted object, so we do
    # not blindly apply the raw OOF probability threshold to it. Instead map
    # the OOF-selected SOURCE adaptation coverage onto the final refit score
    # scale without using in-sample labels for threshold selection.
    refit_threshold_info = map_oof_operating_point_to_refit_score_scale(
        final_source_scores,
        threshold_info,
    )
    refit_threshold = float(refit_threshold_info["threshold"])

    # Persistence must not change any SOURCE deployment decision at the frozen
    # refit threshold.
    before_decision = final_source_scores >= refit_threshold
    after_decision = reloaded_scores >= refit_threshold
    reload_decision_mismatch = int(
        np.count_nonzero(before_decision != after_decision)
    )
    if reload_decision_mismatch != 0:
        raise RuntimeError(
            "Estimator reload changed SOURCE deployment decisions: "
            f"{reload_decision_mismatch}"
        )

    source_score_path = (
        args.output_dir / "CM4C_FINAL_ALL_SOURCE_REFIT_SCORES.csv"
    )
    source_score_df = score_df[
        [
            "family",
            "fold",
            "global_index",
            "case_key",
            "harmful",
            "delta_dice",
            "score_Final",
        ]
    ].copy()
    source_score_df["score_final_refit_all_source"] = final_source_scores
    source_score_df.to_csv(source_score_path, index=False)

    threshold_path = args.output_dir / "CM4C_SOURCE_OPERATING_POINT.json"
    save_json(
        threshold_path,
        {
            "status": "PASS",
            "version": VERSION,
            "oof_reference_operating_point": threshold_info,
            "final_refit_deployment_operating_point": refit_threshold_info,
            "important_scope_note": (
                "HARM recall constraint is selected only on patient-grouped "
                "cross-fitted SOURCE OOF scores. Because the final all-SOURCE "
                "refit is a different fitted estimator, the OOF-selected "
                "SOURCE adaptation coverage is mapped to the refit score scale "
                "using refit SOURCE scores only and no in-sample labels. "
                "No target information is used."
            ),
            "promises12_access": False,
        },
    )

    # Descriptive source-only check of score-scale association after refit.
    from scipy.stats import spearmanr

    score_scale_spearman = float(
        spearmanr(
            score_df["score_Final"].to_numpy(dtype=np.float64),
            final_source_scores,
        ).statistic
    )

    # Descriptive only: quantify what the mapped threshold does on the
    # all-SOURCE refit scores. These labels are NOT used to choose or modify
    # the mapped threshold.
    refit_high = final_source_scores >= refit_threshold
    y_ref = y.astype(np.int64)

    refit_tp = int(np.sum((y_ref == 1) & refit_high))
    refit_fn = int(np.sum((y_ref == 1) & (~refit_high)))
    refit_fp = int(np.sum((y_ref == 0) & refit_high))
    refit_tn = int(np.sum((y_ref == 0) & (~refit_high)))

    refit_descriptive = {
        "selection_role": "DESCRIPTIVE_ONLY_NOT_USED_FOR_THRESHOLD_SELECTION",
        "tp": refit_tp,
        "fn": refit_fn,
        "fp": refit_fp,
        "tn": refit_tn,
        "harm_recall": (
            refit_tp / (refit_tp + refit_fn)
            if (refit_tp + refit_fn) else float("nan")
        ),
        "fpr": (
            refit_fp / (refit_fp + refit_tn)
            if (refit_fp + refit_tn) else float("nan")
        ),
        "adaptation_coverage": float(np.mean(~refit_high)),
    }

    lock = {
        "status": "PASS",
        "decision": PASS_DECISION,
        "version": VERSION,
        "build": BUILD,
        "cm4a_lock_sha256": EXPECTED_CM4A_LOCK_SHA,
        "cm4b_lock_sha256": EXPECTED_CM4B_LOCK_SHA,
        "cm4b_helper_sha256": EXPECTED_CM4B_HELPER_SHA,
        "source": {
            "patients": N_PATIENTS,
            "slices": N_SLICES,
            "family_slice_rows": EXPECTED_ROWS,
            "families": FAMILIES,
            "harm_count": int(y.sum()),
        },
        "frozen_representation": {
            "name": "Prediction-Conditioned DINOv2 + M2",
            "dino_lock": cm4b["dino_lock"],
            "feature_meta_sha256": cm4b["artifacts"]["feature_meta_sha256"],
            "feature_file_sha256": feature_meta["feature_sha256"],
            "cond_dim": COND_DIM,
            "pca_components": PCA_COMPONENTS,
            "m2_dim": 2,
        },
        "frozen_source_operating_point": {
            "oof_reference": threshold_info,
            "final_refit_deployment": refit_threshold_info,
            "final_refit_descriptive_label_audit": refit_descriptive,
        },
        "final_estimator": {
            "path": str(model_path),
            "sha256": sha256_file(model_path),
            "joblib_reload_max_abs_score_diff": max_reload_diff,
            "joblib_reload_score_atol": RELOAD_SCORE_ATOL,
            "joblib_reload_decision_mismatch_at_frozen_threshold": (
                reload_decision_mismatch
            ),
            "all_source_fit": True,
            "classifier": "balanced LogisticRegression",
            "max_iter": LR_MAX_ITER,
            "score_semantics": "predict_proba[:,1] = P(HARM)",
        },
        "score_scale_audit": {
            "oof_vs_all_source_refit_spearman": score_scale_spearman,
            "note": (
                "Descriptive SOURCE-only audit; not used to alter threshold."
            ),
        },
        "information_boundary": {
            "promises12_access": False,
            "target_gt": False,
            "target_calibration": False,
            "target_threshold_tuning": False,
        },
        "artifacts": {
            "source_operating_point": str(threshold_path),
            "source_operating_point_sha256": sha256_file(threshold_path),
            "all_source_refit_scores": str(source_score_path),
            "all_source_refit_scores_sha256": sha256_file(source_score_path),
        },
        "next_stage": (
            "CM5_PROMISE12_GT_FREE_SOURCE_AND_TENT1_PREDICTION_"
            "SAFETY_SCORING_AND_UNLABELED_OPERATING_POINT_TRANSPORT"
        ),
    }

    lock_path = (
        args.output_dir
        / "CM4C_FINAL_SOURCE_SAFETY_ESTIMATOR_LOCK.json"
    )
    save_json(lock_path, lock)

    print("\n===== CM4C FIX2 FINAL =====")
    print("Frozen OOF reference threshold:", threshold_info["threshold"])
    print(
        "Frozen OOF SOURCE HARM recall:",
        threshold_info["pooled"]["harm_recall"],
    )
    print("Frozen OOF SOURCE FPR:", threshold_info["pooled"]["fpr"])
    print(
        "Frozen OOF SOURCE adaptation coverage:",
        threshold_info["pooled"]["adaptation_coverage"],
    )
    print(
        "Final-refit deployment threshold:",
        refit_threshold_info["threshold"],
    )
    print(
        "Final-refit mapped adaptation coverage:",
        refit_threshold_info["achieved_adaptation_coverage"],
    )
    print(
        "Final-refit descriptive HARM recall:",
        refit_descriptive["harm_recall"],
        "(NOT used for threshold selection)",
    )
    print(
        "OOF vs all-source-refit score Spearman:",
        score_scale_spearman,
    )
    print(
        "Estimator reload max abs score diff:",
        max_reload_diff,
    )
    print(
        "Estimator reload decision mismatch:",
        reload_decision_mismatch,
    )
    print("Final estimator:", model_path)
    print("Estimator SHA256:", sha256_file(model_path))
    print("PROMISE12 access: NO")
    print("Target calibration: NO")
    print("Decision=", PASS_DECISION)
    print("LOCK=", lock_path)
    print("LOCK SHA256=", sha256_file(lock_path))
    print("PASS")


if __name__ == "__main__":
    main()
