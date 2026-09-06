#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1X_CM5B_unlabeled_support_aware_operating_point_transport_lock_fix1.py

SafeTTA Q1 enhancement — CM5B.

Purpose
-------
Lock an unlabeled PROMISE12 operating-point transport rule BEFORE GT reveal.

Scientific assumption
---------------------
The method is a covariate-shift transport procedure. It requires approximate
conditional stability in the frozen safety representation z:

    P_target(HARM | z, action) ~= P_source(HARM | z, action)

Unlabeled target data cannot identify arbitrary conditional shift. Therefore
CM5B explicitly audits overlap/support and abstains if the weighted SOURCE HARM
support is inadequate.

Primary transport
-----------------
1) Transform SOURCE and target raw CondDINO+M2 features through the already
   frozen final SOURCE safety preprocessing into z = PCA64+M2 standardized space.
2) Estimate p_target(z)/p_source(z) with patient-cross-fitted logistic domain
   classifiers, separately for each model family.
3) Stabilize source density-ratio weights by mean normalization within family
   and a preregistered maximum normalized weight of 10.
4) On patient-grouped cross-fitted SOURCE safety scores (score_Final), use
   source HARM labels + density-ratio weights to estimate target-weighted HARM
   recall.
5) Use a patient-cluster bootstrap to choose a conservative OOF-space threshold:
   the 5th percentile of bootstrap-specific largest thresholds satisfying
   weighted HARM recall >= 0.90.
6) Convert that threshold into a target adaptation-coverage estimand via the
   weighted SOURCE score distribution.
7) Map the estimated target adaptation coverage to the actual unlabeled target
   final-refit risk-score distribution, yielding tau_transport.

No PROMISE12 GT, DeltaDice, HARM labels, target tuning, or oracle information is
used.

A simple unlabeled quantile baseline is also locked prospectively:
- preserve the original SOURCE adaptation coverage (30.5563%) on target scores.

CM5B does not evaluate either target threshold. That happens only after CM6 GT
reveal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm


VERSION = "2026-09-05-Q1X-CM5B-v1-fix1"
BUILD = "Q1X_CM5B_UNLABELED_SUPPORT_AWARE_OPERATING_POINT_TRANSPORT_LOCK_FIX1"

ROOT = Path(r"F:\MEDSEG_SAFETTA")
OUT = ROOT / "outputs"
CODE = ROOT / "code"

CM4B_LOCK = (
    OUT
    / "Q1X_CM4B_mri_source_prediction_conditioned_safety_grouped_oof_fix1_v1"
    / "CM4B_MRI_SOURCE_SAFETY_OOF_LOCK.json"
)
EXPECTED_CM4B_LOCK_SHA = (
    "7903f8dd84338be8051883b1a905607daec25e55539205f7c6fb78ca3cc41a7e"
)

CM4C_LOCK = (
    OUT
    / "Q1X_CM4C_final_source_safety_estimator_and_threshold_lock_fix2_v1"
    / "CM4C_FINAL_SOURCE_SAFETY_ESTIMATOR_LOCK.json"
)
EXPECTED_CM4C_LOCK_SHA = (
    "5ddf4a6a4b44d19a1e3c8cccc8a2a9cdba71da18850a7df58cbbee83a40527ff"
)

CM5A_LOCK = (
    OUT
    / "Q1X_CM5A_promise12_gt_free_prediction_tent1_safety_scoring_lock_fix1_v1"
    / "CM5A_PROMISE12_GT_FREE_PREDICTION_SAFETY_LOCK.json"
)
EXPECTED_CM5A_LOCK_SHA = (
    "18198bf417ec30434fedcab39f8f50bff270fade676e47fccfa1e397fc2c93fd"
)

CM5A_HELPER = (
    CODE
    / "Q1X_CM5A_promise12_gt_free_prediction_tent1_safety_scoring_lock_fix1.py"
)
EXPECTED_CM5A_HELPER_SHA = (
    "e7373152880068157b88cba31f091258932c9366e53313476692ab81c5c50f3b"
)

DEFAULT_OUTPUT = (
    OUT
    / "Q1X_CM5B_unlabeled_support_aware_operating_point_transport_lock_fix1_v1"
)

FAMILIES = ["UNet", "DeepLabV3_R50", "SegFormer_B0"]

N_SOURCE_PATIENTS = 139
N_SOURCE_SLICES = 3553
N_TARGET_PATIENTS = 50
N_TARGET_SLICES = 1377

RAW_DIM = 1538
COND_DIM = 1536
Z_DIM = 66

N_DOMAIN_FOLDS = 5
DOMAIN_LR_C = 1.0
DOMAIN_LR_MAX_ITER = 3000
DOMAIN_PROB_EPS = 1e-4
MAX_NORMALIZED_WEIGHT = 10.0

HARM_RECALL_TARGET = 0.90
BOOTSTRAP_REPS = 1000
BOOTSTRAP_SEED = 20260905
CONSERVATIVE_BOOTSTRAP_QUANTILE = 0.05

# Support adequacy gates, fixed before any target GT reveal.
MIN_POOLED_HARM_SLICE_ESS = 100.0
MIN_POOLED_HARM_PATIENT_ESS = 20.0
MIN_FAMILY_HARM_PATIENT_ESS = 8.0
MAX_FAMILY_TARGET_Q99_FRACTION = 0.50

PASS_DECISION = (
    "UNLABELED_SUPPORT_AWARE_OPERATING_POINT_TRANSPORT_LOCKED_"
    "READY_FOR_CM6_PROMISE12_GT_REVEAL"
)
ABSTAIN_DECISION = (
    "UNLABELED_OPERATING_POINT_TRANSPORT_ABSTAIN_SUPPORT_INADEQUATE_"
    "READY_FOR_CM6_GT_REVEAL_AS_ABSTENTION_ANALYSIS"
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
        ("CM4B_LOCK", CM4B_LOCK, EXPECTED_CM4B_LOCK_SHA),
        ("CM4C_LOCK", CM4C_LOCK, EXPECTED_CM4C_LOCK_SHA),
        ("CM5A_LOCK", CM5A_LOCK, EXPECTED_CM5A_LOCK_SHA),
        ("CM5A_HELPER", CM5A_HELPER, EXPECTED_CM5A_HELPER_SHA),
    ]:
        if not path.is_file():
            raise FileNotFoundError(path)
        got = sha256_file(path)
        print(label, got, "PASS" if got == expected else "FAIL")
        if got != expected:
            raise RuntimeError(f"{label} SHA mismatch.")

    cm4b = load_json(CM4B_LOCK)
    cm4c = load_json(CM4C_LOCK)
    cm5a = load_json(CM5A_LOCK)

    if cm4b.get("status") != "PASS":
        raise RuntimeError("CM4B status changed.")
    if cm4c.get("status") != "PASS":
        raise RuntimeError("CM4C status changed.")
    if cm5a.get("status") != "PASS":
        raise RuntimeError("CM5A status changed.")

    if cm5a.get("information_boundary", {}).get(
        "promises12_gt_mhd_payload_read"
    ) is not False:
        raise RuntimeError("CM5A GT-MHD boundary changed.")
    if cm5a.get("information_boundary", {}).get(
        "promises12_gt_raw_payload_read"
    ) is not False:
        raise RuntimeError("CM5A GT-RAW boundary changed.")
    if cm5a.get("information_boundary", {}).get(
        "target_harm_labels"
    ) is not False:
        raise RuntimeError("CM5A target HARM boundary changed.")

    return cm4b, cm4c, cm5a


def load_source_assets(cm4b):
    score_path = Path(cm4b["artifacts"]["scores"])
    if sha256_file(score_path) != cm4b["artifacts"]["scores_sha256"]:
        raise RuntimeError("CM4B SOURCE OOF-score SHA mismatch.")

    source_scores = pd.read_csv(score_path)

    required = {
        "family",
        "fold",
        "global_index",
        "case_key",
        "harmful",
        "score_Final",
    }
    missing = required.difference(source_scores.columns)
    if missing:
        raise RuntimeError(
            f"Missing SOURCE score columns: {sorted(missing)}"
        )

    if len(source_scores) != len(FAMILIES) * N_SOURCE_SLICES:
        raise RuntimeError("Unexpected SOURCE score rows.")
    if source_scores["case_key"].nunique() != N_SOURCE_PATIENTS:
        raise RuntimeError("Unexpected SOURCE patient count.")

    meta_path = Path(cm4b["artifacts"]["feature_meta"])
    if sha256_file(meta_path) != cm4b["artifacts"]["feature_meta_sha256"]:
        raise RuntimeError("CM4B SOURCE feature-meta SHA mismatch.")

    meta = load_json(meta_path)
    feature_path = Path(meta["feature_file"])
    if sha256_file(feature_path) != meta["feature_sha256"]:
        raise RuntimeError("CM4B SOURCE feature SHA mismatch.")

    source_features = np.load(feature_path, mmap_mode="r")
    expected_shape = (
        len(FAMILIES),
        N_SOURCE_SLICES,
        RAW_DIM,
    )
    if tuple(source_features.shape) != expected_shape:
        raise RuntimeError(
            f"Unexpected SOURCE feature shape {source_features.shape}"
        )

    return source_scores, source_features, meta


def load_target_assets(cm5a):
    feature_path = Path(cm5a["artifacts"]["target_features"])
    if sha256_file(feature_path) != cm5a["artifacts"]["target_features_sha256"]:
        raise RuntimeError("CM5A target feature SHA mismatch.")

    target_features = np.load(feature_path, mmap_mode="r")
    expected_shape = (
        len(FAMILIES),
        N_TARGET_SLICES,
        RAW_DIM,
    )
    if tuple(target_features.shape) != expected_shape:
        raise RuntimeError(
            f"Unexpected target feature shape {target_features.shape}"
        )

    score_path = Path(cm5a["artifacts"]["target_scores"])
    if sha256_file(score_path) != cm5a["artifacts"]["target_scores_sha256"]:
        raise RuntimeError("CM5A target score SHA mismatch.")

    target_scores = pd.read_csv(score_path)

    required = {
        "family",
        "global_index",
        "case_key",
        "slice_index",
        "risk_score",
        "fixed_source_threshold",
        "fixed_source_gate_adapt",
    }
    missing = required.difference(target_scores.columns)
    if missing:
        raise RuntimeError(
            f"Missing target score columns: {sorted(missing)}"
        )

    forbidden = {
        "harmful",
        "beneficial",
        "delta_dice",
        "source_dice",
        "tent1_dice",
    }
    present_forbidden = forbidden.intersection(target_scores.columns)
    if present_forbidden:
        raise RuntimeError(
            "Target GT-derived columns unexpectedly present: "
            f"{sorted(present_forbidden)}"
        )

    if len(target_scores) != len(FAMILIES) * N_TARGET_SLICES:
        raise RuntimeError("Unexpected target score rows.")
    if target_scores["case_key"].nunique() != N_TARGET_PATIENTS:
        raise RuntimeError("Unexpected target patient count.")

    return target_scores, target_features


def transform_raw_to_z(bundle, X):
    X = np.asarray(X, dtype=np.float32)

    cond = bundle["cond_imputer"].transform(X[:, :COND_DIM])
    pca = bundle["pca"].transform(cond)

    m2 = bundle["m2_imputer"].transform(X[:, COND_DIM:RAW_DIM])

    final = np.concatenate([pca, m2], axis=1)
    z = bundle["scaler"].transform(final)

    if z.shape[1] != Z_DIM:
        raise RuntimeError(f"Unexpected z dim: {z.shape}")
    if not np.isfinite(z).all():
        raise RuntimeError("Nonfinite standardized safety representation.")

    return z.astype(np.float32)


def aligned_family_arrays(
    family,
    family_index,
    source_scores,
    source_features,
    target_scores,
    target_features,
    safety_bundle,
):
    s = (
        source_scores[source_scores["family"] == family]
        .sort_values("global_index")
        .reset_index(drop=True)
    )
    t = (
        target_scores[target_scores["family"] == family]
        .sort_values("global_index")
        .reset_index(drop=True)
    )

    if len(s) != N_SOURCE_SLICES:
        raise RuntimeError(f"{family} SOURCE rows changed.")
    if len(t) != N_TARGET_SLICES:
        raise RuntimeError(f"{family} target rows changed.")

    if not np.array_equal(
        s["global_index"].to_numpy(dtype=np.int64),
        np.arange(N_SOURCE_SLICES, dtype=np.int64),
    ):
        raise RuntimeError(f"{family} SOURCE global_index coverage changed.")
    if not np.array_equal(
        t["global_index"].to_numpy(dtype=np.int64),
        np.arange(N_TARGET_SLICES, dtype=np.int64),
    ):
        raise RuntimeError(f"{family} target global_index coverage changed.")

    Xs = np.asarray(
        source_features[family_index],
        dtype=np.float32,
    )
    Xt = np.asarray(
        target_features[family_index],
        dtype=np.float32,
    )

    zs = transform_raw_to_z(safety_bundle, Xs)
    zt = transform_raw_to_z(safety_bundle, Xt)

    return s, t, zs, zt


def target_fold_from_case_key(case_key):
    text = str(case_key)
    if not text.startswith("Case"):
        raise RuntimeError(f"Unexpected PROMISE case key {text}")
    case_id = int(text[4:])
    if case_id < 0 or case_id >= N_TARGET_PATIENTS:
        raise RuntimeError(f"Unexpected PROMISE case id {case_id}")
    return case_id % N_DOMAIN_FOLDS


def fit_cross_fitted_density_ratio(
    family,
    s,
    t,
    zs,
    zt,
):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    s_fold = s["fold"].to_numpy(dtype=np.int64)
    t_fold = np.asarray(
        [target_fold_from_case_key(v) for v in t["case_key"]],
        dtype=np.int64,
    )

    if sorted(np.unique(s_fold).tolist()) != list(range(N_DOMAIN_FOLDS)):
        raise RuntimeError(f"{family} SOURCE folds changed.")
    if sorted(np.unique(t_fold).tolist()) != list(range(N_DOMAIN_FOLDS)):
        raise RuntimeError(f"{family} target folds changed.")

    q_s = np.full(len(s), np.nan, dtype=np.float64)
    q_t = np.full(len(t), np.nan, dtype=np.float64)

    for fold in range(N_DOMAIN_FOLDS):
        s_train = s_fold != fold
        s_val = s_fold == fold
        t_train = t_fold != fold
        t_val = t_fold == fold

        source_train_patients = set(s.loc[s_train, "case_key"].astype(str))
        source_val_patients = set(s.loc[s_val, "case_key"].astype(str))
        target_train_patients = set(t.loc[t_train, "case_key"].astype(str))
        target_val_patients = set(t.loc[t_val, "case_key"].astype(str))

        if source_train_patients.intersection(source_val_patients):
            raise RuntimeError(
                f"{family} domain fold{fold} SOURCE patient leakage."
            )
        if target_train_patients.intersection(target_val_patients):
            raise RuntimeError(
                f"{family} domain fold{fold} target patient leakage."
            )

        X_train = np.concatenate(
            [zs[s_train], zt[t_train]],
            axis=0,
        )
        y_train = np.concatenate(
            [
                np.zeros(int(s_train.sum()), dtype=np.int64),
                np.ones(int(t_train.sum()), dtype=np.int64),
            ]
        )

        clf = LogisticRegression(
            C=DOMAIN_LR_C,
            class_weight="balanced",
            max_iter=DOMAIN_LR_MAX_ITER,
            solver="lbfgs",
            random_state=20260905 + fold,
        )
        clf.fit(X_train, y_train)

        q_s[s_val] = clf.predict_proba(zs[s_val])[:, 1]
        q_t[t_val] = clf.predict_proba(zt[t_val])[:, 1]

    if not np.isfinite(q_s).all() or not np.isfinite(q_t).all():
        raise RuntimeError(f"{family} incomplete domain OOF probabilities.")

    y_domain = np.concatenate(
        [
            np.zeros(len(q_s), dtype=np.int64),
            np.ones(len(q_t), dtype=np.int64),
        ]
    )
    q_domain = np.concatenate([q_s, q_t])

    domain_auc = float(roc_auc_score(y_domain, q_domain))

    # Equal effective class priors are induced by class_weight='balanced',
    # therefore odds q/(1-q) estimate p_target(z)/p_source(z).
    q_s_clip = np.clip(
        q_s,
        DOMAIN_PROB_EPS,
        1.0 - DOMAIN_PROB_EPS,
    )
    raw_w = q_s_clip / (1.0 - q_s_clip)

    mean_raw = float(raw_w.mean())
    if not np.isfinite(mean_raw) or mean_raw <= 0:
        raise RuntimeError(f"{family} invalid raw density ratio.")

    normalized_w = raw_w / mean_raw
    capped_w = np.minimum(
        normalized_w,
        MAX_NORMALIZED_WEIGHT,
    )

    # Re-normalize to family mean 1 after capping.
    capped_mean = float(capped_w.mean())
    if not np.isfinite(capped_mean) or capped_mean <= 0:
        raise RuntimeError(f"{family} invalid capped density ratio.")

    w = capped_w / capped_mean

    if not np.isfinite(w).all() or np.any(w <= 0):
        raise RuntimeError(f"{family} invalid stabilized weights.")

    cap_fraction = float(
        np.mean(normalized_w > MAX_NORMALIZED_WEIGHT)
    )
    target_q99_fraction = float(np.mean(q_t >= 0.99))

    return {
        "q_source": q_s,
        "q_target": q_t,
        "raw_weight": raw_w,
        "weight": w,
        "domain_auc": domain_auc,
        "weight_cap_fraction": cap_fraction,
        "target_q99_fraction": target_q99_fraction,
        "raw_weight_p50": float(np.quantile(raw_w, 0.50)),
        "raw_weight_p90": float(np.quantile(raw_w, 0.90)),
        "raw_weight_p99": float(np.quantile(raw_w, 0.99)),
        "normalized_weight_max_before_cap": float(normalized_w.max()),
        "stabilized_weight_max": float(w.max()),
    }


def ess(weights):
    w = np.asarray(weights, dtype=np.float64)
    if len(w) == 0:
        return 0.0
    denom = float(np.sum(w * w))
    if denom <= 0:
        return 0.0
    return float((np.sum(w) ** 2) / denom)


def patient_aggregated_ess(case_keys, weights):
    df = pd.DataFrame(
        {
            "case_key": np.asarray(case_keys).astype(str),
            "weight": np.asarray(weights, dtype=np.float64),
        }
    )
    agg = df.groupby("case_key", as_index=False)["weight"].sum()
    return ess(agg["weight"].to_numpy(dtype=np.float64))


def largest_threshold_for_weighted_harm_recall(
    scores,
    harm,
    weights,
    target_recall,
):
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(harm, dtype=np.int64)
    w = np.asarray(weights, dtype=np.float64)

    if len(s) != len(y) or len(s) != len(w):
        raise RuntimeError("Threshold arrays length mismatch.")

    harm_mask = y == 1
    total_harm_weight = float(w[harm_mask].sum())

    if total_harm_weight <= 0:
        raise RuntimeError("No weighted HARM mass.")

    candidates = np.unique(s)
    best = None

    # With only ~10k SOURCE family-slice rows this exact loop is cheap and
    # avoids quantile/tie ambiguity.
    for tau in candidates:
        protected = s >= tau
        recall = float(
            w[harm_mask & protected].sum() / total_harm_weight
        )
        if recall + 1e-15 >= target_recall:
            best = float(tau)

    if best is None:
        raise RuntimeError("No threshold satisfies weighted HARM recall.")

    return best


def weighted_policy_stats(scores, harm, weights, tau):
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(harm, dtype=np.int64)
    w = np.asarray(weights, dtype=np.float64)

    high = s >= tau
    low = ~high

    harm_mask = y == 1
    nonharm = ~harm_mask

    harm_total = float(w[harm_mask].sum())
    nonharm_total = float(w[nonharm].sum())
    total = float(w.sum())

    return {
        "threshold": float(tau),
        "weighted_harm_recall": float(
            w[harm_mask & high].sum() / harm_total
        ),
        "weighted_fpr": float(
            w[nonharm & high].sum() / nonharm_total
        ),
        "estimated_target_adaptation_coverage": float(
            w[low].sum() / total
        ),
    }


def conservative_patient_bootstrap_threshold(
    source_table,
):
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    patients = np.asarray(
        sorted(source_table["case_key"].astype(str).unique())
    )

    if len(patients) != N_SOURCE_PATIENTS:
        raise RuntimeError("Unexpected SOURCE bootstrap patient count.")

    patient_to_rows = {
        p: np.flatnonzero(
            source_table["case_key"].astype(str).to_numpy() == p
        )
        for p in patients
    }

    scores = source_table["score_Final"].to_numpy(dtype=np.float64)
    harm = source_table["harmful"].to_numpy(dtype=np.int64)
    weights = source_table["density_ratio_weight"].to_numpy(
        dtype=np.float64
    )

    taus = []

    for _ in tqdm(
        range(BOOTSTRAP_REPS),
        desc="CM5B SOURCE patient bootstrap",
        unit="rep",
        dynamic_ncols=True,
    ):
        sampled = rng.choice(
            patients,
            size=len(patients),
            replace=True,
        )
        idx = np.concatenate(
            [patient_to_rows[p] for p in sampled]
        )

        tau = largest_threshold_for_weighted_harm_recall(
            scores[idx],
            harm[idx],
            weights[idx],
            HARM_RECALL_TARGET,
        )
        taus.append(tau)

    taus = np.asarray(taus, dtype=np.float64)

    conservative_tau = float(
        np.quantile(
            taus,
            CONSERVATIVE_BOOTSTRAP_QUANTILE,
        )
    )

    return {
        "reps": BOOTSTRAP_REPS,
        "seed": BOOTSTRAP_SEED,
        "target_harm_recall": HARM_RECALL_TARGET,
        "bootstrap_threshold_median": float(np.median(taus)),
        "bootstrap_threshold_q05": float(np.quantile(taus, 0.05)),
        "bootstrap_threshold_q95": float(np.quantile(taus, 0.95)),
        "conservative_quantile": CONSERVATIVE_BOOTSTRAP_QUANTILE,
        "conservative_oof_threshold": conservative_tau,
        "interpretation": (
            "5th percentile of patient-bootstrap largest feasible "
            "weighted-HARM-recall>=0.90 OOF thresholds"
        ),
    }


def threshold_for_target_coverage(scores, target_coverage):
    s = np.asarray(scores, dtype=np.float64)

    if not np.isfinite(s).all():
        raise RuntimeError("Nonfinite target risk scores.")
    if not (0.0 <= target_coverage <= 1.0):
        raise RuntimeError("Invalid target coverage.")

    n = len(s)
    k = int(round(target_coverage * n))
    ordered = np.sort(s)

    if k <= 0:
        tau = float(ordered[0])
    elif k >= n:
        tau = float(np.nextafter(ordered[-1], np.inf))
    else:
        lo = float(ordered[k - 1])
        hi = float(ordered[k])

        if lo < hi:
            tau = float(lo + 0.5 * (hi - lo))
        else:
            tau = float(np.nextafter(hi, np.inf))

    achieved = float(np.mean(s < tau))

    return {
        "threshold": tau,
        "target_adaptation_coverage_requested": float(target_coverage),
        "target_adaptation_coverage_achieved": achieved,
        "target_low_risk_rows": int(np.sum(s < tau)),
        "target_total_rows": int(n),
        "rule": "risk_score >= threshold => retain SOURCE; below => adapt",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    args = ap.parse_args()

    print(
        "===== Q1X CM5B UNLABELED SUPPORT-AWARE OPERATING-POINT TRANSPORT ====="
    )
    print("PROMISE12_GT_ACCESS=NO")
    print("TARGET_HARM_LABELS=NO")
    print("TARGET_DELTA_DICE=NO")
    print("TARGET_THRESHOLD_TUNING_WITH_GT=NO")
    print("DOMAIN_RATIO_MODEL=patient-cross-fitted balanced LogisticRegression")
    print("DOMAIN_SPACE=frozen final SOURCE PCA64+M2 standardized z")
    print("DENSITY_RATIO=domain odds under balanced priors")
    print("MAX_NORMALIZED_WEIGHT=", MAX_NORMALIZED_WEIGHT)
    print("HARM_RECALL_TARGET=", HARM_RECALL_TARGET)
    print("PATIENT_BOOTSTRAP_REPS=", BOOTSTRAP_REPS)
    print("SUPPORT_AWARE_ABSTENTION=YES")

    cm4b, cm4c, cm5a = verify_lineage()

    source_scores, source_features, source_feature_meta = (
        load_source_assets(cm4b)
    )
    target_scores, target_features = load_target_assets(cm5a)

    estimator_path = Path(cm4c["final_estimator"]["path"])
    if sha256_file(estimator_path) != cm4c["final_estimator"]["sha256"]:
        raise RuntimeError("Final SOURCE safety estimator SHA mismatch.")

    safety_bundle = joblib.load(estimator_path)

    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)

    source_parts = []
    target_parts = []
    support_rows = []

    family_to_idx = {family: i for i, family in enumerate(FAMILIES)}

    print("\n===== PATIENT-CROSS-FITTED DOMAIN RATIO ESTIMATION =====")

    for family in FAMILIES:
        fi = family_to_idx[family]

        s, t, zs, zt = aligned_family_arrays(
            family,
            fi,
            source_scores,
            source_features,
            target_scores,
            target_features,
            safety_bundle,
        )

        ratio = fit_cross_fitted_density_ratio(
            family,
            s,
            t,
            zs,
            zt,
        )

        s = s.copy()
        s["domain_q_target_oof"] = ratio["q_source"]
        s["density_ratio_raw"] = ratio["raw_weight"]
        s["density_ratio_weight"] = ratio["weight"]

        t = t.copy()
        t["domain_q_target_oof"] = ratio["q_target"]
        t["domain_fold"] = [
            target_fold_from_case_key(v)
            for v in t["case_key"]
        ]

        harm_mask = s["harmful"].to_numpy(dtype=np.int64) == 1
        harm_weights = s.loc[
            harm_mask,
            "density_ratio_weight",
        ].to_numpy(dtype=np.float64)
        harm_cases = s.loc[
            harm_mask,
            "case_key",
        ].astype(str).to_numpy()

        support = {
            "family": family,
            "domain_oof_auroc": ratio["domain_auc"],
            "source_weight_ess_all_slices": ess(
                s["density_ratio_weight"].to_numpy(dtype=np.float64)
            ),
            "source_weight_ess_harm_slices": ess(harm_weights),
            "source_weight_ess_harm_patients": patient_aggregated_ess(
                harm_cases,
                harm_weights,
            ),
            "source_weight_cap_fraction": ratio["weight_cap_fraction"],
            "target_q_ge_0.99_fraction": ratio["target_q99_fraction"],
            "raw_weight_p50": ratio["raw_weight_p50"],
            "raw_weight_p90": ratio["raw_weight_p90"],
            "raw_weight_p99": ratio["raw_weight_p99"],
            "normalized_weight_max_before_cap": (
                ratio["normalized_weight_max_before_cap"]
            ),
            "stabilized_weight_max": ratio["stabilized_weight_max"],
        }
        support_rows.append(support)

        print(
            f"{family}: "
            f"domain_AUROC={support['domain_oof_auroc']:.6f} "
            f"HARM_slice_ESS={support['source_weight_ess_harm_slices']:.1f} "
            f"HARM_patient_ESS={support['source_weight_ess_harm_patients']:.1f} "
            f"target_q>=.99={support['target_q_ge_0.99_fraction']:.3f} "
            f"cap_fraction={support['source_weight_cap_fraction']:.3f}"
        )

        source_parts.append(s)
        target_parts.append(t)

    source_table = pd.concat(
        source_parts,
        ignore_index=True,
    )
    target_table = pd.concat(
        target_parts,
        ignore_index=True,
    )

    # Family-normalized weights give each family equal total SOURCE mass,
    # matching the equal target family panel.
    all_weights = source_table["density_ratio_weight"].to_numpy(
        dtype=np.float64
    )
    harm_mask = source_table["harmful"].to_numpy(dtype=np.int64) == 1

    pooled_harm_slice_ess = ess(all_weights[harm_mask])
    pooled_harm_patient_ess = patient_aggregated_ess(
        source_table.loc[harm_mask, "case_key"].astype(str).to_numpy(),
        all_weights[harm_mask],
    )

    support_df = pd.DataFrame(support_rows)
    support_path = args.output_dir / "CM5B_SUPPORT_DIAGNOSTICS.csv"
    support_df.to_csv(support_path, index=False)

    support_gate = {
        "pooled_harm_slice_ess": pooled_harm_slice_ess,
        "pooled_harm_patient_ess": pooled_harm_patient_ess,
        "min_family_harm_patient_ess": float(
            support_df["source_weight_ess_harm_patients"].min()
        ),
        "max_family_target_q_ge_0.99_fraction": float(
            support_df["target_q_ge_0.99_fraction"].max()
        ),
        "requirements": {
            "min_pooled_harm_slice_ess": MIN_POOLED_HARM_SLICE_ESS,
            "min_pooled_harm_patient_ess": MIN_POOLED_HARM_PATIENT_ESS,
            "min_family_harm_patient_ess": MIN_FAMILY_HARM_PATIENT_ESS,
            "max_family_target_q_ge_0.99_fraction": (
                MAX_FAMILY_TARGET_Q99_FRACTION
            ),
        },
    }

    support_ok = (
        pooled_harm_slice_ess >= MIN_POOLED_HARM_SLICE_ESS
        and pooled_harm_patient_ess >= MIN_POOLED_HARM_PATIENT_ESS
        and support_gate["min_family_harm_patient_ess"]
        >= MIN_FAMILY_HARM_PATIENT_ESS
        and support_gate["max_family_target_q_ge_0.99_fraction"]
        <= MAX_FAMILY_TARGET_Q99_FRACTION
    )

    support_gate["pass"] = bool(support_ok)

    print("\n===== SUPPORT GATE =====")
    print(json.dumps(support_gate, indent=2))

    source_weight_path = (
        args.output_dir
        / "CM5B_SOURCE_CROSSFIT_DENSITY_RATIO_WEIGHTS.csv"
    )
    source_table.to_csv(source_weight_path, index=False)

    target_domain_path = (
        args.output_dir
        / "CM5B_TARGET_CROSSFIT_DOMAIN_PROPENSITIES.csv"
    )
    target_table.to_csv(target_domain_path, index=False)

    source_oof_reference_coverage = float(
        cm4c["frozen_source_operating_point"]
        ["oof_reference"]["pooled"]["adaptation_coverage"]
    )
    fixed_source_threshold = float(
        cm4c["frozen_source_operating_point"]
        ["final_refit_deployment"]["threshold"]
    )

    target_risk = target_table["risk_score"].to_numpy(dtype=np.float64)

    # Lock the simple unlabeled quantile baseline regardless of whether the
    # proposed support-aware method abstains.
    naive_quantile = threshold_for_target_coverage(
        target_risk,
        source_oof_reference_coverage,
    )

    if support_ok:
        print("\n===== SUPPORT-AWARE TRANSPORT =====")

        bootstrap = conservative_patient_bootstrap_threshold(
            source_table
        )
        tau_oof_transport = float(
            bootstrap["conservative_oof_threshold"]
        )

        weighted_stats = weighted_policy_stats(
            source_table["score_Final"].to_numpy(dtype=np.float64),
            source_table["harmful"].to_numpy(dtype=np.int64),
            source_table["density_ratio_weight"].to_numpy(dtype=np.float64),
            tau_oof_transport,
        )

        estimated_target_coverage = float(
            weighted_stats["estimated_target_adaptation_coverage"]
        )

        transport = threshold_for_target_coverage(
            target_risk,
            estimated_target_coverage,
        )
        transport.update(
            {
                "status": "PASS",
                "method": (
                    "cross-fitted density-ratio weighted SOURCE HARM recall "
                    "+ patient-bootstrap conservative OOF threshold "
                    "+ unlabeled target score-quantile mapping"
                ),
                "covariate_shift_assumption": (
                    "P_target(HARM|z,A1_TENT_1STEP) approximately equals "
                    "P_source(HARM|z,A1_TENT_1STEP)"
                ),
                "oof_transport_threshold": tau_oof_transport,
                "weighted_source_policy_at_oof_transport_threshold": (
                    weighted_stats
                ),
                "patient_bootstrap": bootstrap,
                "support_gate": support_gate,
                "uses_target_gt": False,
                "uses_target_harm_labels": False,
            }
        )

        decision = PASS_DECISION

    else:
        transport = {
            "status": "ABSTAIN",
            "method": (
                "support-aware covariate-shift operating-point transport"
            ),
            "reason": "pre-registered overlap/ESS support gate failed",
            "support_gate": support_gate,
            "threshold": None,
            "uses_target_gt": False,
            "uses_target_harm_labels": False,
        }
        decision = ABSTAIN_DECISION

    # Family-wise target coverages under each already locked threshold are
    # descriptive only; they cannot alter either threshold.
    policy_rows = []

    for family in FAMILIES:
        m = target_table["family"].to_numpy() == family
        scores = target_table.loc[m, "risk_score"].to_numpy(dtype=np.float64)

        row = {
            "family": family,
            "fixed_source_threshold": fixed_source_threshold,
            "fixed_source_adaptation_coverage": float(
                np.mean(scores < fixed_source_threshold)
            ),
            "naive_quantile_threshold": float(
                naive_quantile["threshold"]
            ),
            "naive_quantile_adaptation_coverage": float(
                np.mean(scores < naive_quantile["threshold"])
            ),
        }

        if transport["status"] == "PASS":
            row["support_aware_threshold"] = float(
                transport["threshold"]
            )
            row["support_aware_adaptation_coverage"] = float(
                np.mean(scores < transport["threshold"])
            )
        else:
            row["support_aware_threshold"] = np.nan
            row["support_aware_adaptation_coverage"] = np.nan

        policy_rows.append(row)

    policy_df = pd.DataFrame(policy_rows)
    policy_path = args.output_dir / "CM5B_LOCKED_POLICY_SUMMARY.csv"
    policy_df.to_csv(policy_path, index=False)

    print("\n===== LOCKED UNLABELED POLICY SUMMARY =====")
    print("fixed SOURCE threshold:", fixed_source_threshold)
    print(
        "naive target-quantile threshold:",
        naive_quantile["threshold"],
        "coverage:",
        naive_quantile["target_adaptation_coverage_achieved"],
    )
    if transport["status"] == "PASS":
        print(
            "support-aware target threshold:",
            transport["threshold"],
            "coverage:",
            transport["target_adaptation_coverage_achieved"],
        )
        print(
            "transport weighted SOURCE HARM recall:",
            transport[
                "weighted_source_policy_at_oof_transport_threshold"
            ]["weighted_harm_recall"],
        )
    else:
        print("support-aware transport: ABSTAIN")

    print("\nFamily-wise target coverage (GT-free):")
    print(policy_df.to_string(index=False))

    transport_path = (
        args.output_dir
        / "CM5B_UNLABELED_OPERATING_POINT_TRANSPORT.json"
    )
    save_json(
        transport_path,
        {
            "status": "PASS" if support_ok else "ABSTAIN",
            "version": VERSION,
            "fixed_source_reference": {
                "threshold": fixed_source_threshold,
                "source_oof_adaptation_coverage": (
                    source_oof_reference_coverage
                ),
            },
            "naive_target_quantile_baseline": naive_quantile,
            "support_aware_transport": transport,
            "promises12_gt_access": False,
        },
    )

    lock = {
        "status": "PASS" if support_ok else "ABSTAIN",
        "decision": decision,
        "version": VERSION,
        "build": BUILD,
        "cm4b_lock_sha256": EXPECTED_CM4B_LOCK_SHA,
        "cm4c_lock_sha256": EXPECTED_CM4C_LOCK_SHA,
        "cm5a_lock_sha256": EXPECTED_CM5A_LOCK_SHA,
        "cm5a_helper_sha256": EXPECTED_CM5A_HELPER_SHA,
        "scientific_assumption": (
            "Covariate shift / conditional stability in frozen safety "
            "representation: P_t(HARM|z,a) ~= P_s(HARM|z,a). "
            "No unlabeled method can identify arbitrary conditional shift."
        ),
        "domain_ratio_estimator": {
            "space": "frozen final SOURCE standardized PCA64+M2 z",
            "per_family": True,
            "folds": N_DOMAIN_FOLDS,
            "grouping": "patient",
            "classifier": "balanced LogisticRegression",
            "C": DOMAIN_LR_C,
            "max_iter": DOMAIN_LR_MAX_ITER,
            "density_ratio": "q_target/(1-q_target) under balanced priors",
            "weight_normalization": "mean 1 within family",
            "max_normalized_weight": MAX_NORMALIZED_WEIGHT,
            "cross_fitted_source_weights": True,
            "cross_fitted_target_propensities": True,
        },
        "support_gate": support_gate,
        "fixed_source_reference": {
            "threshold": fixed_source_threshold,
            "source_oof_adaptation_coverage": source_oof_reference_coverage,
        },
        "naive_target_quantile_baseline": naive_quantile,
        "support_aware_transport": transport,
        "information_boundary": {
            "promises12_gt_access": False,
            "target_harm_labels": False,
            "target_delta_dice": False,
            "target_oracle_threshold": False,
            "target_threshold_tuning_with_gt": False,
        },
        "locked_cm6_evaluation_plan": {
            "primary_ranking": (
                "macro-family target HARM AUROC with 2000x patient-cluster "
                "bootstrap 95% CI"
            ),
            "operating_points": [
                "fixed SOURCE deployment threshold",
                "naive unlabeled target-quantile baseline",
                (
                    "support-aware unlabeled transported threshold"
                    if support_ok
                    else "support-aware method ABSTAIN"
                ),
                "oracle target threshold at HARM recall >=0.90 (reference only)",
            ],
            "metrics": [
                "HARM recall",
                "FPR",
                "precision",
                "adaptation coverage",
                "prevented HARM fraction",
                "deployed Dice",
                "SOURCE Dice",
                "TENT1 Dice",
                "Spearman(risk,-DeltaDice)",
            ],
            "family_level_reporting": True,
            "no_post_gt_threshold_changes": True,
        },
        "artifacts": {
            "support_diagnostics": str(support_path),
            "support_diagnostics_sha256": sha256_file(support_path),
            "source_density_ratio_weights": str(source_weight_path),
            "source_density_ratio_weights_sha256": (
                sha256_file(source_weight_path)
            ),
            "target_domain_propensities": str(target_domain_path),
            "target_domain_propensities_sha256": (
                sha256_file(target_domain_path)
            ),
            "policy_summary": str(policy_path),
            "policy_summary_sha256": sha256_file(policy_path),
            "transport": str(transport_path),
            "transport_sha256": sha256_file(transport_path),
        },
        "next_stage": (
            "CM6_PROMISE12_GT_REVEAL_AND_LOCKED_POLICY_EVALUATION"
        ),
    }

    lock_path = (
        args.output_dir
        / "CM5B_UNLABELED_OPERATING_POINT_TRANSPORT_LOCK.json"
    )
    save_json(lock_path, lock)

    print("\n===== CM5B FINAL =====")
    print("Support gate PASS:", support_ok)
    print("PROMISE12 GT access: NO")
    print("Target HARM labels: NO")
    print("Target DeltaDice: NO")
    print("Post-GT threshold changes allowed: NO")
    print("Decision=", decision)
    print("LOCK=", lock_path)
    print("LOCK SHA256=", sha256_file(lock_path))
    print("PASS" if support_ok else "ABSTAIN")


if __name__ == "__main__":
    main()
