#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R13D_pl_conf90_frozen_safety_transfer_evaluation_fix1.py

R13D — Frozen pre-adaptation safety-score transfer to PL-CONF90 harm.

Purpose
-------
Evaluate whether the already-frozen R10L0 safety estimator retains
risk-ranking ability when the downstream TTA algorithm changes from
A1_TENT_1STEP to A4_PL_CONF90_1STEP on the same NeoPolyp development cohort.

Frozen boundaries
-----------------
- training/refit: NO
- recalibration: NO
- threshold reselection: NO
- score reversal: NO
- PolypGen access: NO
- PL hyperparameter tuning: NO
- independent external validation: NO
- post-PolypGen exploratory robustness extension: YES
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm


ROOT = Path(r"F:\MEDSEG_SAFETTA")
CODE = ROOT / "code"
OUT = ROOT / "outputs"

# ---------------- Upstream exact-gate files ----------------
R13C_SCRIPT = CODE / "Q1_R13C_pl_conf90_gt_utility_reveal_and_outcome_lock_fix1.py"
EXPECTED_R13C_SCRIPT_SHA256 = "f2dd94d59e61be0432a2117e60aca1dc7ec4352af15e0aea0a27afdbf293590b"

R13C_DIR = OUT / "Q1_R13C_pl_conf90_gt_utility_reveal_and_outcome_lock_fix1_v1"
R13C_LOCK = R13C_DIR / "R13C_PL_OUTCOME_LOCK.json"
R13C_PANEL = R13C_DIR / "R13C_PL_model_case_outcomes.csv"
EXPECTED_R13C_LOCK_SHA256 = "a676040f2e1db8d118f49efc0e3ec36821b3abe0c96a13e2c4e9054e9cfeb6d9"
EXPECTED_R13C_DECISION = "NEOPOLYP_PL_CONF90_GT_UTILITY_AND_OUTCOMES_LOCKED"

R10L0_SCRIPT = CODE / "Q1_R10L0_final_source_safety_estimator_lock_fix3.py"
EXPECTED_R10L0_SCRIPT_SHA256 = "ed828784d835b2fc77d93e3afb6865b382da7774dae6826add17ca8534cb2bb6"

R10L0_DIR = OUT / "Q1_R10L0_final_source_safety_estimator_lock_fix3_v1"
R10L0_PCA = R10L0_DIR / "R10L0_final_condDINO_PCA64.joblib"
R10L0_HEAD = R10L0_DIR / "R10L0_final_safety_head.joblib"
R10L0_THRESHOLD = R10L0_DIR / "R10L0_frozen_operating_threshold.json"
R10L0_LOCK = R10L0_DIR / "R10L0_FINAL_SOURCE_ESTIMATOR_LOCK.json"
EXPECTED_R10L0_DECISION = (
    "FINAL_SOURCE_ESTIMATOR_LOCKED_FOR_INDEPENDENT_EXTERNAL_VALIDATION"
)
EXPECTED_FROZEN_THRESHOLD = 0.300584763193734

# ---------------- Frozen representation inputs ----------------
IMAGE_FEATURES = OUT / "Q1_R10K2A_dinov2_image_mask_representation_lock_fix1_v1" / "R10K2A_image_cls_features.npz"
STATE_FEATURES = OUT / "Q1_R10K2A_dinov2_image_mask_representation_lock_fix1_v1" / "R10K2A_mask_conditioned_dinov2_features.npz"
IMAGE_MANIFEST = OUT / "Q1_R05D1_neopolyp_acquisition_pairing_duplicate_qc_v1_fix2" / "frozen_confirmatory_manifest.csv"
STATE_METADATA = OUT / "Q1_R10K2A_dinov2_image_mask_representation_lock_fix1_v1" / "R10K2A_state_metadata_no_labels.csv"
MORPHOLOGY = OUT / "Q1_R10J1_source_mask_morphology_feature_builder_fix1_v1" / "source_mask_morphology_features_labeled.csv"

OUTPUT_DIR = OUT / "Q1_R13D_pl_conf90_frozen_safety_transfer_evaluation_fix2_v1"

M2 = [
    "morph_fg_fraction",
    "morph_boundary_density",
]

TARGET_RECALL = 0.90
PPV_PREVALENCE = 0.01
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 20260827

SCENARIO_A1 = "PRIMARY_A1"
SCENARIO_PL = "PRIMARY_PL"
DELTA_PL_MINUS_A1 = "PRIMARY_PL_MINUS_PRIMARY_A1"

DECISION_SUPPORTED = "PL_CONF90_FROZEN_SAFETY_RANKING_TRANSFER_SUPPORTED"
DECISION_SUPPORTED_ATTENUATED = (
    "PL_CONF90_FROZEN_SAFETY_RANKING_TRANSFER_SUPPORTED_WITH_ATTENUATION"
)
DECISION_NOT_SUPPORTED = "PL_CONF90_FROZEN_SAFETY_RANKING_TRANSFER_NOT_SUPPORTED"


def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def norm_sid_series(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.strip()
        .str.replace("\\", "/", regex=False)
        .str.lower()
    )


def tabular_matrix(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return np.column_stack([
        pd.to_numeric(df[c], errors="coerce").to_numpy(float)
        for c in cols
    ]).astype(np.float32)


def threshold_for_recall(y, p, target):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    pos = np.sort(p[y == 1])[::-1]
    if len(pos) == 0:
        return np.nan
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
    denom = recall * pi + fpr * (1.0 - pi) if np.isfinite(recall) and np.isfinite(fpr) else np.nan
    ppv1pct = recall * pi / denom if np.isfinite(denom) and denom > 0 else np.nan

    return {
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "Recall": float(recall),
        "FPR": float(fpr),
        "EmpiricalPPV": float(empirical_ppv),
        "PPV1pct": float(ppv1pct),
    }


def ece_10bin(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    bins = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    n = len(y)
    for lo, hi in zip(bins[:-1], bins[1:]):
        if hi < 1.0:
            mask = (p >= lo) & (p < hi)
        else:
            mask = (p >= lo) & (p <= hi)
        if not np.any(mask):
            continue
        conf = float(np.mean(p[mask]))
        acc = float(np.mean(y[mask]))
        ece += (np.sum(mask) / n) * abs(acc - conf)
    return float(ece)


def scenario_metrics(df: pd.DataFrame, label_col: str, threshold: float):
    y = df[label_col].to_numpy(int)
    p = df["frozen_probability"].to_numpy(float)

    if len(np.unique(y)) < 2:
        raise RuntimeError(f"Single-class target for {label_col}.")

    auc = float(roc_auc_score(y, p))
    ap = float(average_precision_score(y, p))
    op = operating_metrics(y, p, threshold)
    oracle_thr = threshold_for_recall(y, p, TARGET_RECALL)
    oracle = operating_metrics(y, p, oracle_thr) if np.isfinite(oracle_thr) else {"FPR": np.nan}
    brier = float(np.mean((p - y) ** 2))
    ece = ece_10bin(y, p)

    return {
        "AUROC": auc,
        "AUPRC": ap,
        "Recall": op["Recall"],
        "FPR": op["FPR"],
        "EmpiricalPPV": op["EmpiricalPPV"],
        "PPV1pct": op["PPV1pct"],
        "OracleFPRatR90": oracle["FPR"],
        "Brier": brier,
        "ECE10": ece,
    }


def family_macro_metrics(df: pd.DataFrame, label_col: str, threshold: float):
    fam_rows = []
    for fam, g in df.groupby("model_family", sort=True):
        met = scenario_metrics(g, label_col, threshold)
        fam_rows.append({
            "scenario": label_col,
            "model_family": fam,
            "rows": int(len(g)),
            "cases": int(g["sample_id"].nunique()),
            **met,
        })
    fam_df = pd.DataFrame(fam_rows)
    macro = {
        "AUROC": float(fam_df["AUROC"].mean()),
        "AUPRC": float(fam_df["AUPRC"].mean()),
        "Recall": float(fam_df["Recall"].mean()),
        "FPR": float(fam_df["FPR"].mean()),
        "EmpiricalPPV": float(fam_df["EmpiricalPPV"].mean()),
        "PPV1pct": float(fam_df["PPV1pct"].mean()),
        "OracleFPRatR90": float(fam_df["OracleFPRatR90"].mean()),
        "Brier": float(fam_df["Brier"].mean()),
        "ECE10": float(fam_df["ECE10"].mean()),
    }
    return fam_df, macro


def verify_r13c():
    for p in (R13C_SCRIPT, R13C_LOCK, R13C_PANEL):
        if not p.exists():
            raise FileNotFoundError(p)

    if sha256_file(R13C_SCRIPT) != EXPECTED_R13C_SCRIPT_SHA256:
        raise RuntimeError("R13C script SHA mismatch.")

    lock_sha = sha256_file(R13C_LOCK)
    if lock_sha != EXPECTED_R13C_LOCK_SHA256:
        raise RuntimeError(
            f"R13C lock SHA mismatch: expected={EXPECTED_R13C_LOCK_SHA256} actual={lock_sha}"
        )

    lock = json.loads(R13C_LOCK.read_text(encoding="utf-8"))
    if lock.get("decision") != EXPECTED_R13C_DECISION:
        raise RuntimeError(f"Unexpected R13C decision: {lock.get('decision')}")
    if not bool(lock.get("r13d_transfer_evaluable", False)):
        raise RuntimeError("R13C says R13D not evaluable.")
    if int(lock.get("pl_harm_unique_classes", -1)) != 2:
        raise RuntimeError("R13C target is not binary evaluable.")
    if int(lock.get("target_cases", -1)) != 1000:
        raise RuntimeError("R13C case count mismatch.")
    if int(lock.get("model_states", -1)) != 9:
        raise RuntimeError("R13C state count mismatch.")
    if int(lock.get("model_case_rows", -1)) != 9000:
        raise RuntimeError("R13C row count mismatch.")

    panel = pd.read_csv(R13C_PANEL, low_memory=False)
    required = [
        "sample_id",
        "model_family",
        "model_state_id",
        "training_seed",
        "checkpoint_sha256",
        "source_dice",
        "tent1_dice",
        "tent1_delta_dice",
        "tent1_adaptation_outcome",
        "tent1_harm_label",
        "tent1_benefit_label",
        "pl_dice",
        "pl_delta_dice",
        "pl_adaptation_outcome",
        "pl_harm_label",
        "pl_benefit_label",
    ]
    missing = [c for c in required if c not in panel.columns]
    if missing:
        raise RuntimeError(f"R13C panel missing: {missing}")

    if len(panel) != 9000:
        raise RuntimeError(f"R13C panel rows={len(panel)}")
    if panel[["sample_id", "model_state_id"]].drop_duplicates().shape[0] != 9000:
        raise RuntimeError("R13C sample+state key is not unique.")

    return lock, panel, lock_sha


def verify_r10l0():
    for p in (R10L0_SCRIPT, R10L0_PCA, R10L0_HEAD, R10L0_THRESHOLD, R10L0_LOCK):
        if not p.exists():
            raise FileNotFoundError(p)

    if sha256_file(R10L0_SCRIPT) != EXPECTED_R10L0_SCRIPT_SHA256:
        raise RuntimeError("R10L0 script SHA mismatch.")

    lock = json.loads(R10L0_LOCK.read_text(encoding="utf-8"))
    if lock.get("decision") != EXPECTED_R10L0_DECISION:
        raise RuntimeError(f"Unexpected R10L0 decision: {lock.get('decision')}")

    th = json.loads(R10L0_THRESHOLD.read_text(encoding="utf-8"))
    frozen_threshold = float(th["final_threshold"])
    if abs(frozen_threshold - EXPECTED_FROZEN_THRESHOLD) > 1e-15:
        raise RuntimeError(
            f"Frozen threshold changed: expected={EXPECTED_FROZEN_THRESHOLD} actual={frozen_threshold}"
        )

    return lock, th, frozen_threshold


def load_locked_features(
    image_feature_path: Path,
    state_feature_path: Path,
    image_manifest_path: Path,
    state_metadata_path: Path,
    morph_path: Path,
):
    # ---- image CLS: verify case set/order exactly as R10K2A/K2B ----
    with np.load(image_feature_path, allow_pickle=False) as z:
        if "cls" not in z.files:
            raise AssertionError(f"Image feature file keys={z.files}, required key='cls'")
        image_cls = np.asarray(z["cls"], dtype=np.float32)

    if image_cls.shape != (1000, 768):
        raise AssertionError(f"Unexpected image CLS shape: {image_cls.shape}")
    if not np.isfinite(image_cls).all():
        raise AssertionError("Non-finite image CLS features.")

    image_manifest = pd.read_csv(image_manifest_path, low_memory=False)
    required_image_manifest = ["sample_id", "image_path"]
    missing = [c for c in required_image_manifest if c not in image_manifest.columns]
    if missing:
        raise AssertionError(f"Image manifest missing: {missing}")
    image_manifest["_sid"] = norm_sid_series(image_manifest["sample_id"])

    if len(image_manifest) != 1000:
        raise AssertionError(f"Expected 1000 image-manifest rows, got {len(image_manifest)}")
    if image_manifest["_sid"].nunique() != 1000:
        raise AssertionError("Image manifest sample_id is not one-to-one.")

    image_manifest = image_manifest.sort_values("_sid").reset_index(drop=True)
    image_sid_norm = image_manifest["_sid"].to_numpy()

    # ---- state conditioned DINO numeric arrays only ----
    with np.load(state_feature_path, allow_pickle=False) as z:
        required_numeric = {"foreground_patch_mean", "background_patch_mean"}
        if not required_numeric.issubset(z.files):
            raise AssertionError(
                f"State feature file keys={z.files}, required keys={required_numeric}"
            )
        fg = np.asarray(z["foreground_patch_mean"], dtype=np.float32)
        bg = np.asarray(z["background_patch_mean"], dtype=np.float32)

    if fg.shape != (9000, 768):
        raise AssertionError(f"Unexpected FG shape: {fg.shape}")
    if bg.shape != (9000, 768):
        raise AssertionError(f"Unexpected BG shape: {bg.shape}")
    if not np.isfinite(fg).all():
        raise AssertionError("Non-finite foreground DINO features.")
    if not np.isfinite(bg).all():
        raise AssertionError("Non-finite background DINO features.")

    cond = np.concatenate([fg, bg], axis=1).astype(np.float32)

    state_meta = pd.read_csv(state_metadata_path, low_memory=False)
    required_state_meta = [
        "sample_id",
        "model_family",
        "model_state_id",
        "training_seed",
        "checkpoint_sha256",
    ]
    missing = [c for c in required_state_meta if c not in state_meta.columns]
    if missing:
        raise AssertionError(f"State metadata missing: {missing}")

    state_meta["_sid"] = norm_sid_series(state_meta["sample_id"])
    state_meta["_state_row"] = np.arange(len(state_meta), dtype=int)

    if len(state_meta) != 9000:
        raise AssertionError(f"Expected 9000 state metadata rows, got {len(state_meta)}")
    if state_meta["_sid"].nunique() != 1000:
        raise AssertionError("Expected 1000 cases in state metadata.")
    if state_meta["model_state_id"].nunique() != 9:
        raise AssertionError("Expected 9 model states in state metadata.")
    if state_meta[["_sid", "model_state_id"]].drop_duplicates().shape[0] != 9000:
        raise AssertionError("R10K2A state metadata explicit key is not unique.")
    if set(image_sid_norm.tolist()) != set(state_meta["_sid"].tolist()):
        raise AssertionError("Image-manifest and state-metadata case sets differ.")

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
        raise AssertionError(f"Expected 9000 morphology rows, got {len(morph)}")
    if morph[["_sid", "model_state_id"]].drop_duplicates().shape[0] != 9000:
        raise AssertionError("Morphology explicit key is not unique.")

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
        raise AssertionError("Morphology/R10K2A representation join != 9000.")

    fam_ok = joined["model_family_morph"].astype(str) == joined["model_family_repr"].astype(str)
    if not fam_ok.all():
        raise AssertionError("Model-family mismatch in representation join.")

    if "training_seed_morph" in joined.columns:
        if "training_seed_repr" not in joined.columns:
            raise AssertionError(
                "training_seed_morph exists but training_seed_repr is missing."
            )
        seed_ok = (
            joined["training_seed_morph"].astype(str)
            == joined["training_seed_repr"].astype(str)
        )
        if not seed_ok.all():
            raise AssertionError("Training-seed mismatch in representation join.")

    if "checkpoint_sha256_morph" in joined.columns:
        if "checkpoint_sha256_repr" not in joined.columns:
            raise AssertionError(
                "checkpoint_sha256_morph exists but checkpoint_sha256_repr is missing."
            )
        sha_ok = (
            joined["checkpoint_sha256_morph"].astype(str)
            == joined["checkpoint_sha256_repr"].astype(str)
        )
        if not sha_ok.all():
            raise AssertionError("Checkpoint mismatch in representation join.")

    joined = joined.rename(columns={"model_family_morph": "model_family"})

    # Canonicalize metadata from authoritative R10K2A state metadata.
    # This is a schema correction only; numeric representations are untouched.
    if "training_seed_repr" in joined.columns:
        joined["training_seed"] = joined["training_seed_repr"]
    elif "training_seed" not in joined.columns:
        raise AssertionError(
            "No canonical training_seed available after representation join."
        )

    if "checkpoint_sha256_repr" in joined.columns:
        joined["checkpoint_sha256"] = joined["checkpoint_sha256_repr"]
    elif "checkpoint_sha256" not in joined.columns:
        raise AssertionError(
            "No canonical checkpoint_sha256 available after representation join."
        )

    drop_cols = [
        c for c in [
            "model_family_repr",
            "training_seed_repr",
            "training_seed_morph",
            "checkpoint_sha256_repr",
            "checkpoint_sha256_morph",
        ] if c in joined.columns
    ]
    joined = joined.drop(columns=drop_cols)

    required_canonical = [
        "sample_id",
        "model_family",
        "model_state_id",
        "training_seed",
        "checkpoint_sha256",
    ]
    missing_canonical = [
        c for c in required_canonical
        if c not in joined.columns
    ]
    if missing_canonical:
        raise AssertionError(
            f"Canonical metadata missing after representation join: "
            f"{missing_canonical}"
        )

    order = joined["_state_row"].to_numpy(int)
    cond_aligned = cond[order].astype(np.float32)

    if not np.isfinite(cond_aligned).all():
        raise AssertionError("Non-finite conditioned DINO features.")

    return joined.reset_index(drop=True), cond_aligned


def build_score_panel(r13c_panel: pd.DataFrame, frozen_threshold: float):
    feat_df, cond = load_locked_features(
        IMAGE_FEATURES,
        STATE_FEATURES,
        IMAGE_MANIFEST,
        STATE_METADATA,
        MORPHOLOGY,
    )

    if len(feat_df) != 9000 or cond.shape != (9000, 1536):
        raise RuntimeError("Feature cardinality mismatch.")

    pca = joblib.load(R10L0_PCA)
    head = joblib.load(R10L0_HEAD)

    z = pca.transform(cond)
    if z.shape != (9000, 64):
        raise RuntimeError(f"PCA transform output shape={z.shape}")

    X = np.concatenate([tabular_matrix(feat_df, M2), np.asarray(z, dtype=np.float32)], axis=1)
    if X.shape != (9000, 66):
        raise RuntimeError(f"Final safety input shape={X.shape}")

    p = head.predict_proba(X)[:, 1]
    if p.shape != (9000,):
        raise RuntimeError(f"Probability shape={p.shape}")
    if not np.isfinite(p).all():
        raise RuntimeError("Non-finite frozen probabilities.")

    score_df = feat_df.copy()
    score_df["frozen_probability"] = p.astype(float)
    score_df["frozen_threshold"] = float(frozen_threshold)
    score_df["frozen_flag"] = (score_df["frozen_probability"] >= frozen_threshold).astype(int)

    merged = score_df.merge(
        r13c_panel,
        on=["sample_id", "model_family", "model_state_id"],
        how="inner",
        validate="one_to_one",
        suffixes=("_score", "_r13c"),
    )

    if len(merged) != 9000:
        raise RuntimeError(f"Score/outcome merge rows={len(merged)}")

    for c in ("training_seed", "checkpoint_sha256"):
        score_col = f"{c}_score"
        r13c_col = f"{c}_r13c"
        if score_col not in merged.columns or r13c_col not in merged.columns:
            raise RuntimeError(
                f"Expected canonical metadata columns missing after merge: "
                f"{score_col}, {r13c_col}; "
                f"available={list(merged.columns)}"
            )
        a = merged[score_col].astype(str)
        b = merged[r13c_col].astype(str)
        if not a.equals(b):
            raise RuntimeError(f"Metadata mismatch after merge: {c}")
        merged[c] = merged[score_col]

    drop_cols = [
        "training_seed_score",
        "training_seed_r13c",
        "checkpoint_sha256_score",
        "checkpoint_sha256_r13c",
        "_sid",
        "_state_row",
    ]
    drop_cols = [c for c in drop_cols if c in merged.columns]
    merged = merged.drop(columns=drop_cols)

    return merged.reset_index(drop=True), z, X, p


def bootstrap_macro(panel: pd.DataFrame, reps: int, seed: int, threshold: float):
    rng = np.random.default_rng(seed)

    cases = np.sort(panel["sample_id"].astype(str).unique())
    if len(cases) != 1000:
        raise RuntimeError(f"Expected 1000 unique cases, got {len(cases)}")

    grouped = {
        sid: g.reset_index(drop=True)
        for sid, g in panel.groupby("sample_id", sort=False)
    }

    scenario_rows = []
    delta_rows = []

    for rep in tqdm(
        range(reps),
        total=reps,
        desc="R13D case-clustered bootstrap",
        unit="rep",
        dynamic_ncols=True,
    ):
        sampled = rng.choice(cases, size=len(cases), replace=True)
        boot = pd.concat([grouped[sid] for sid in sampled], axis=0, ignore_index=True)

        _, m_a1 = family_macro_metrics(boot, "tent1_harm_label", threshold)
        _, m_pl = family_macro_metrics(boot, "pl_harm_label", threshold)

        for metric, value in m_a1.items():
            scenario_rows.append({
                "level": "SCENARIO",
                "group": SCENARIO_A1,
                "metric": metric,
                "rep": rep,
                "value": float(value),
            })
        for metric, value in m_pl.items():
            scenario_rows.append({
                "level": "SCENARIO",
                "group": SCENARIO_PL,
                "metric": metric,
                "rep": rep,
                "value": float(value),
            })
        for metric in m_pl.keys():
            delta_rows.append({
                "level": "PAIRED_DELTA",
                "group": DELTA_PL_MINUS_A1,
                "metric": metric,
                "rep": rep,
                "value": float(m_pl[metric] - m_a1[metric]),
            })

    boot_long = pd.DataFrame(scenario_rows + delta_rows)
    boot_ci = (
        boot_long
        .groupby(["level", "group", "metric"], as_index=False)["value"]
        .agg(
            mean="mean",
            ci95_low=lambda s: float(np.quantile(s, 0.025)),
            ci95_high=lambda s: float(np.quantile(s, 0.975)),
        )
    )
    return boot_long, boot_ci


def decide(pl_macro: dict, a1_macro: dict, boot_ci: pd.DataFrame, prevalence: float):
    def ci_row(level, group, metric):
        x = boot_ci[
            (boot_ci["level"] == level)
            & (boot_ci["group"] == group)
            & (boot_ci["metric"] == metric)
        ]
        if len(x) != 1:
            raise RuntimeError(f"Missing bootstrap CI: {level}/{group}/{metric}")
        return x.iloc[0]

    auroc_pl = ci_row("SCENARIO", SCENARIO_PL, "AUROC")
    auprc_delta = ci_row("PAIRED_DELTA", DELTA_PL_MINUS_A1, "AUPRC")
    auroc_delta = ci_row("PAIRED_DELTA", DELTA_PL_MINUS_A1, "AUROC")

    supported = (float(auroc_pl["ci95_low"]) > 0.5) and (float(pl_macro["AUPRC"]) > float(prevalence))
    attenuated = (
        float(auroc_delta["ci95_high"]) < 0.0
        and float(auprc_delta["ci95_high"]) < 0.0
    )

    if supported and attenuated:
        return DECISION_SUPPORTED_ATTENUATED
    if supported:
        return DECISION_SUPPORTED
    return DECISION_NOT_SUPPORTED


def self_test():
    assert abs(threshold_for_recall([1, 1, 0], [0.9, 0.2, 0.1], 0.9) - 0.2) < 1e-12
    op = operating_metrics([1, 0, 1, 0], [0.9, 0.6, 0.4, 0.1], 0.5)
    assert op["Recall"] == 0.5
    assert op["FPR"] == 0.5
    assert op["EmpiricalPPV"] == 0.5
    assert op["PPV1pct"] > 0
    e = ece_10bin([0, 1], [0.1, 0.9])
    assert 0 <= e <= 1

    left = pd.DataFrame({
        "sample_id": ["a"],
        "model_family": ["f"],
        "model_state_id": ["s"],
        "training_seed": [1],
        "checkpoint_sha256": ["x"],
    })
    right = left.copy()
    mm = left.merge(
        right,
        on=["sample_id", "model_family", "model_state_id"],
        suffixes=("_score", "_r13c"),
        validate="one_to_one",
    )
    assert "training_seed_score" in mm.columns
    assert "training_seed_r13c" in mm.columns
    assert "checkpoint_sha256_score" in mm.columns
    assert "checkpoint_sha256_r13c" in mm.columns

    print("CANONICAL_METADATA_SUFFIX_TEST_PASS")
    print("THRESHOLD_TEST_PASS")
    print("OPERATING_METRICS_TEST_PASS")
    print("ECE_TEST_PASS")
    print("SELF_TEST_PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    ap.add_argument("--bootstrap-reps", type=int, default=BOOTSTRAP_REPS)
    ap.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    print("===== R13D PL-CONF90 FROZEN SAFETY TRANSFER EVALUATION FIX2 =====")
    print("cross-TTA-algorithm transfer=YES")
    print("independent external validation=NO")
    print("post-PolypGen exploratory robustness extension=YES")
    print("training/refit=NO")
    print("threshold reselection=NO")
    print("score reversal=NO")
    print("PolypGen access=NO")
    print("Fix1 failure=metadata suffix/schema assumption only")
    print("Fix2 scientific protocol change=NO")
    print()

    _, r13c_panel, r13c_lock_sha = verify_r13c()
    print("R13C exact script+lock+panel gate=PASS")
    print("R13C LOCK SHA256:", r13c_lock_sha)

    _, threshold_json, frozen_threshold = verify_r10l0()
    print("R10L0 exact script+artifacts gate=PASS")
    print("Frozen threshold:", frozen_threshold)

    panel, z, X, p = build_score_panel(r13c_panel, frozen_threshold)

    sentinel = float(np.max(np.abs(np.asarray(z[:, 0], dtype=np.float32) - np.asarray(z[:, 0], dtype=np.float16).astype(np.float32))))
    print()
    print("===== FROZEN R10L0 ESTIMATOR =====")
    print("PCA transform output:", z.shape)
    print("Safety input:", X.shape)
    print("Probability finite:", bool(np.isfinite(p).all()))
    print("Frozen-threshold flagged:", int(np.sum(p >= frozen_threshold)), "/", len(p))
    print("DINO transform sentinel max abs:", sentinel)

    fam_a1, macro_a1 = family_macro_metrics(panel, "tent1_harm_label", frozen_threshold)
    fam_pl, macro_pl = family_macro_metrics(panel, "pl_harm_label", frozen_threshold)

    prevalence_pl = float(panel["pl_harm_label"].mean())
    prevalence_a1 = float(panel["tent1_harm_label"].mean())

    boot_long, boot_ci = bootstrap_macro(
        panel,
        reps=args.bootstrap_reps,
        seed=args.bootstrap_seed,
        threshold=frozen_threshold,
    )

    decision = decide(macro_pl, macro_a1, boot_ci, prevalence_pl)

    print("\n===== PAPER-FACING LOCKED FACTS =====")
    print("A1 harm prevalence:", prevalence_a1)
    print("PL harm prevalence:", prevalence_pl)
    print("Primary PL Macro AUROC:", macro_pl["AUROC"])
    print("Primary PL Macro AUPRC:", macro_pl["AUPRC"])
    print("Primary PL AUPRC - prevalence:", macro_pl["AUPRC"] - prevalence_pl)
    print("Primary PL AUPRC / prevalence:", macro_pl["AUPRC"] / prevalence_pl if prevalence_pl > 0 else np.nan)
    print("Frozen-threshold Recall:", macro_pl["Recall"])
    print("Frozen-threshold FPR:", macro_pl["FPR"])
    print("Frozen-threshold Empirical PPV:", macro_pl["EmpiricalPPV"])
    print("Frozen-threshold PPV@1%:", macro_pl["PPV1pct"])
    print("Oracle target FPR@R90:", macro_pl["OracleFPRatR90"])
    print("Brier score diagnostic:", macro_pl["Brier"])
    print("10-bin ECE diagnostic:", macro_pl["ECE10"])

    print("\n===== FAMILY POINT METRICS: PRIMARY_PL =====")
    print(fam_pl.to_string(index=False))

    print("\n===== FAMILY POINT METRICS: PRIMARY_A1 =====")
    print(fam_a1.to_string(index=False))

    print("\n==== CLUSTERED BOOTSTRAP =====")
    printable = boot_ci.copy()
    printable["mean"] = printable["mean"].astype(float)
    printable["ci95_low"] = printable["ci95_low"].astype(float)
    printable["ci95_high"] = printable["ci95_high"].astype(float)
    print(printable.to_string(index=False))

    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)

    point_rows = [
        {"level": "SCENARIO", "group": SCENARIO_A1, **macro_a1},
        {"level": "SCENARIO", "group": SCENARIO_PL, **macro_pl},
        {
            "level": "PAIRED_DELTA",
            "group": DELTA_PL_MINUS_A1,
            **{k: float(macro_pl[k] - macro_a1[k]) for k in macro_pl.keys()},
        },
    ]
    point_df = pd.DataFrame(point_rows)

    eval_panel = panel[
        [
            "sample_id",
            "model_family",
            "model_state_id",
            "training_seed",
            "checkpoint_sha256",
            "morph_fg_fraction",
            "morph_boundary_density",
            "frozen_probability",
            "frozen_threshold",
            "frozen_flag",
            "source_dice",
            "tent1_dice",
            "tent1_delta_dice",
            "tent1_adaptation_outcome",
            "tent1_harm_label",
            "tent1_benefit_label",
            "pl_dice",
            "pl_delta_dice",
            "pl_adaptation_outcome",
            "pl_harm_label",
            "pl_benefit_label",
        ]
    ].copy()

    point_path = args.output_dir / "R13D_point_metrics_macro.csv"
    fam_pl_path = args.output_dir / "R13D_family_point_metrics_primary_pl.csv"
    fam_a1_path = args.output_dir / "R13D_family_point_metrics_primary_a1.csv"
    boot_path = args.output_dir / "R13D_clustered_bootstrap_ci.csv"
    point_long_path = args.output_dir / "R13D_clustered_bootstrap_long.csv"
    panel_path = args.output_dir / "R13D_locked_score_pl_evaluation_panel.csv"

    point_df.to_csv(point_path, index=False)
    fam_pl.to_csv(fam_pl_path, index=False)
    fam_a1.to_csv(fam_a1_path, index=False)
    boot_ci.to_csv(boot_path, index=False)
    boot_long.to_csv(point_long_path, index=False)
    eval_panel.to_csv(panel_path, index=False)

    summary = {
        "status": "FROZEN",
        "decision": decision,
        "r13c_script_sha256": sha256_file(R13C_SCRIPT),
        "r13c_lock_sha256": sha256_file(R13C_LOCK),
        "r10l0_script_sha256": sha256_file(R10L0_SCRIPT),
        "r10l0_lock_sha256": sha256_file(R10L0_LOCK),
        "target_cases": int(panel["sample_id"].nunique()),
        "model_states": int(panel["model_state_id"].nunique()),
        "model_case_rows": int(len(panel)),
        "frozen_threshold": float(frozen_threshold),
        "prevalence": {
            "primary_a1": float(prevalence_a1),
            "primary_pl": float(prevalence_pl),
        },
        "primary_a1_macro": macro_a1,
        "primary_pl_macro": macro_pl,
        "paired_delta_primary_pl_minus_primary_a1": {
            k: float(macro_pl[k] - macro_a1[k])
            for k in macro_pl.keys()
        },
        "bootstrap_reps": int(args.bootstrap_reps),
        "bootstrap_seed": int(args.bootstrap_seed),
        "bootstrap_cluster": "sample_id",
        "interpretation": {
            "cross_tta_algorithm_transfer": "YES",
            "independent_external_validation": "NO",
            "post_polypgen_extension": "YES",
            "auroc_ci_entirely_gt_0_5": bool(
                float(
                    boot_ci[
                        (boot_ci["level"] == "SCENARIO")
                        & (boot_ci["group"] == SCENARIO_PL)
                        & (boot_ci["metric"] == "AUROC")
                    ]["ci95_low"].iloc[0]
                ) > 0.5
            ),
            "auprc_gt_prevalence": bool(macro_pl["AUPRC"] > prevalence_pl),
            "attenuated_vs_primary_a1": bool(
                decision == DECISION_SUPPORTED_ATTENUATED
            ),
            "training_or_refit": False,
            "threshold_reselection": False,
            "score_reversal": False,
            "polypgen_access": False,
            "fix2_change": (
                "metadata-column canonicalization only; "
                "no representation/model/threshold/evaluation change"
            ),
        },
        "artifacts": {
            point_path.name: sha256_file(point_path),
            fam_pl_path.name: sha256_file(fam_pl_path),
            fam_a1_path.name: sha256_file(fam_a1_path),
            boot_path.name: sha256_file(boot_path),
            point_long_path.name: sha256_file(point_long_path),
            panel_path.name: sha256_file(panel_path),
        },
    }

    lock_path = args.output_dir / "R13D_PL_SAFETY_TRANSFER_EVALUATION_LOCK.json"
    lock_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n===== FINAL DECISION =====")
    print("Decision:", decision)
    print("Primary PL AUROC:", macro_pl["AUROC"])
    auroc_row = boot_ci[
        (boot_ci["level"] == "SCENARIO")
        & (boot_ci["group"] == SCENARIO_PL)
        & (boot_ci["metric"] == "AUROC")
    ].iloc[0]
    print("Primary PL AUROC CI95:", [float(auroc_row["ci95_low"]), float(auroc_row["ci95_high"])])
    print("Primary PL AUPRC:", macro_pl["AUPRC"])
    print("Primary PL Recall:", macro_pl["Recall"])
    print("Primary PL FPR:", macro_pl["FPR"])
    print("Primary PL Oracle FPR@R90:", macro_pl["OracleFPRatR90"])
    print("R13D LOCK:", lock_path)
    print("R13D LOCK SHA256:", sha256_file(lock_path))
    print("PASS")


if __name__ == "__main__":
    main()
