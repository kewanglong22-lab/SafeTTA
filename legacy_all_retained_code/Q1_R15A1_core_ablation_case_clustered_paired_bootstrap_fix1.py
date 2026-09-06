#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R15A1_core_ablation_case_clustered_paired_bootstrap_fix1.py

Post-freeze inferential audit for R15A0.

This script reuses the exact historical R10K2C inferential structure:
- cluster unit = sample_id;
- each sampled case contributes all 3 target-family model-state rows;
- the SAME 1000-case bootstrap draw is used across all methods, families,
  and split seeds;
- 2000 bootstrap replicates;
- bootstrap seed = 20260826;
- family metrics are first averaged over the 5 split seeds;
- macro metrics are then averaged over the 3 target families.

No model fitting, PCA fitting, threshold selection, feature change, external
data access, or method reselection is performed.

Primary post-freeze explanatory comparisons:
1) CondDINO vs ImageCLS
2) CondDINO vs FG-only
3) CondDINO vs BG-only
4) Frozen final vs M2
5) Frozen final vs M2+FG
6) Frozen final vs M2+BG
7) Frozen final vs CondDINO

The historical PRIMARY-vs-M2 bootstrap rows are reproduced exactly against
the already-frozen R10K2C result as a lineage/inference sentinel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm


VERSION = "2026-09-04-Q1-R15A1-v1-fix1"
BUILD = "Q1_R15A1_CORE_ABLATION_CASE_CLUSTERED_PAIRED_BOOTSTRAP_FIX1"

ROOT = Path(r"F:\MEDSEG_SAFETTA")
OUT = ROOT / "outputs"

R15A0_DIR = (
    OUT
    / "Q1_R15A0_prediction_conditioned_core_ablation_LOMO_fix1_v1"
)
R15A0_LOCK = R15A0_DIR / "R15A0_CORE_ABLATION_LOCK.json"
PREDICTIONS = R15A0_DIR / "R15A0_all_8method_target_predictions.csv"
R15A0_MACRO = R15A0_DIR / "R15A0_macro_summary.csv"

EXPECTED_R15A0_LOCK_SHA = (
    "a28835ad18cd31e01bf0af0a1696788f6a3aecaf241c083e8399230ca7ccb30a"
)
EXPECTED_R15A0_DECISION = (
    "CORE_ABLATION_COMPLETE_READY_FOR_R15A1_PAIRED_CLUSTER_BOOTSTRAP"
)

# Exact historical inferential reference.
R10K2C_DIR = (
    OUT
    / "Q1_R10K2C_case_clustered_paired_bootstrap_fix1_v1"
)
R10K2C_DELTA = R10K2C_DIR / "R10K2C_paired_macro_delta_ci.csv"
R10K2C_SUMMARY = R10K2C_DIR / "R10K2C_summary.json"
EXPECTED_R10K2C_DECISION = (
    "PAIRED_CLUSTER_BOOTSTRAP_CONFIRMS_MACRO_PRIMARY_BENEFIT"
)

DEFAULT_OUTPUT = (
    OUT
    / "Q1_R15A1_core_ablation_case_clustered_paired_bootstrap_fix1_v1"
)

M2 = "M2_backbone"
CLS = "ImageCLS_PCA64"
FG = "FG_PCA64"
BG = "BG_PCA64"
COND = "CondDINO_PCA64"
M2_FG = "M2_plus_FG_PCA64"
M2_BG = "M2_plus_BG_PCA64"
FINAL = "M2_plus_CondDINO_PCA64"

METHODS = [
    M2,
    CLS,
    FG,
    BG,
    COND,
    M2_FG,
    M2_BG,
    FINAL,
]

FAMILIES = [
    "DeepLabV3-R50",
    "PraNet",
    "SegFormer-B0",
]

SEEDS = [
    20260816,
    20260817,
    20260818,
    20260819,
    20260820,
]

TARGET_RECALL = 0.90
PPV_PREVALENCE = 0.01

# Exact historical R10K2C bootstrap settings.
BOOTSTRAP_SEED = 20260826
DEFAULT_BOOTSTRAPS = 2000

METRICS = [
    "AUROC",
    "AUPRC",
    "Recall",
    "FPR",
    "PPV1pct",
    "OracleR90FPR",
]

COMPARISONS = [
    ("COND_vs_CLS", COND, CLS),
    ("COND_vs_FG", COND, FG),
    ("COND_vs_BG", COND, BG),
    ("FINAL_vs_M2", FINAL, M2),
    ("FINAL_vs_M2_FG", FINAL, M2_FG),
    ("FINAL_vs_M2_BG", FINAL, M2_BG),
    ("FINAL_vs_COND", FINAL, COND),
]

DECISION = (
    "CORE_ABLATION_PAIRED_CLUSTER_INFERENCE_COMPLETE_"
    "FINAL_METHOD_REMAINS_FROZEN"
)


def sha256_file(path: Path, chunk=8 * 1024 * 1024) -> str:
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


def norm_sid_series(s):
    return (
        s.astype(str)
        .str.strip()
        .str.replace("\\", "/", regex=False)
        .str.lower()
    )


def threshold_for_recall(y, p, target):
    # Exact historical R10K2C rule.
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)

    pos = np.sort(p[y == 1])[::-1]
    if len(pos) == 0:
        return np.nan

    k = int(np.ceil(target * len(pos)))
    k = min(max(k, 1), len(pos))
    return float(pos[k - 1])


def metrics(y, p, thresholds):
    # Exact historical R10K2C metric semantics.
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)

    if len(np.unique(y)) != 2:
        return {
            "AUROC": np.nan,
            "AUPRC": np.nan,
            "Recall": np.nan,
            "FPR": np.nan,
            "PPV1pct": np.nan,
            "OracleR90FPR": np.nan,
        }

    auc = float(roc_auc_score(y, p))
    auprc = float(average_precision_score(y, p))

    pred = (p >= thresholds).astype(int)

    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))

    recall = tp / (tp + fn) if tp + fn else np.nan
    fpr = fp / (fp + tn) if fp + tn else np.nan

    pi = PPV_PREVALENCE
    denom = recall * pi + fpr * (1 - pi)
    ppv1 = recall * pi / denom if denom > 0 else np.nan

    oracle_thr = threshold_for_recall(y, p, TARGET_RECALL)
    oracle_pred = (p >= oracle_thr).astype(int)

    ofp = int(np.sum((oracle_pred == 1) & (y == 0)))
    otn = int(np.sum((oracle_pred == 0) & (y == 0)))
    oracle_fpr = ofp / (ofp + otn) if ofp + otn else np.nan

    return {
        "AUROC": auc,
        "AUPRC": auprc,
        "Recall": float(recall),
        "FPR": float(fpr),
        "PPV1pct": float(ppv1),
        "OracleR90FPR": float(oracle_fpr),
    }


def percentile_ci(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    return (
        float(np.mean(x)),
        float(np.percentile(x, 2.5)),
        float(np.percentile(x, 97.5)),
    )


def verify_r15a0():
    print("===== R15A0 FROZEN INPUT GATE =====")

    for p in (R15A0_LOCK, PREDICTIONS, R15A0_MACRO):
        print("exists:", p.exists(), p)
        if not p.is_file():
            raise FileNotFoundError(p)

    got = sha256_file(R15A0_LOCK)
    if got != EXPECTED_R15A0_LOCK_SHA:
        raise RuntimeError(
            f"R15A0 lock SHA changed: {got}"
        )

    lock = load_json(R15A0_LOCK)
    if lock.get("status") != "PASS":
        raise RuntimeError("R15A0 status changed.")
    if lock.get("decision") != EXPECTED_R15A0_DECISION:
        raise RuntimeError("R15A0 decision changed.")
    if bool(lock.get("final_method_reselected", True)):
        raise RuntimeError("R15A0 reports method reselection.")
    if bool(lock.get("external_data_accessed", True)):
        raise RuntimeError("R15A0 reports external-data access.")

    pred_meta = lock.get("artifacts", {}).get(PREDICTIONS.name)
    macro_meta = lock.get("artifacts", {}).get(R15A0_MACRO.name)

    for path, meta in (
        (PREDICTIONS, pred_meta),
        (R15A0_MACRO, macro_meta),
    ):
        if not isinstance(meta, dict):
            raise RuntimeError(
                f"R15A0 lock lacks artifact metadata: {path.name}"
            )
        digest = sha256_file(path)
        if digest != str(meta.get("sha256", "")):
            raise RuntimeError(
                f"R15A0 artifact SHA mismatch: {path.name}"
            )

    print("R15A0 lock SHA:", got)
    print("prediction SHA:", sha256_file(PREDICTIONS))
    print("macro SHA:", sha256_file(R15A0_MACRO))
    print("PASS")

    return lock


def verify_historical_r10k2c():
    print("\n===== HISTORICAL R10K2C INFERENCE REFERENCE =====")

    for p in (R10K2C_DELTA, R10K2C_SUMMARY):
        print("exists:", p.exists(), p)
        if not p.is_file():
            raise FileNotFoundError(p)

    summary = load_json(R10K2C_SUMMARY)
    if summary.get("decision") != EXPECTED_R10K2C_DECISION:
        raise RuntimeError(
            "Historical R10K2C decision changed."
        )
    if summary.get("cluster_unit") != "sample_id":
        raise RuntimeError("Historical bootstrap cluster changed.")
    if int(summary.get("cases", -1)) != 1000:
        raise RuntimeError("Historical R10K2C case count changed.")
    if int(summary.get("states_per_target_family_case", -1)) != 3:
        raise RuntimeError("Historical R10K2C state count changed.")
    if int(summary.get("bootstrap_replicates", -1)) != DEFAULT_BOOTSTRAPS:
        raise RuntimeError("Historical bootstrap replicate count changed.")
    if int(summary.get("bootstrap_seed", -1)) != BOOTSTRAP_SEED:
        raise RuntimeError("Historical bootstrap seed changed.")

    hist = pd.read_csv(R10K2C_DELTA, low_memory=False)
    required = {
        "comparison",
        "metric",
        "bootstrap_mean_delta",
        "ci95_low",
        "ci95_high",
    }
    if not required.issubset(hist.columns):
        raise RuntimeError("Historical R10K2C delta schema changed.")

    hist_primary = hist[
        hist["comparison"].astype(str) == "PRIMARY_vs_M2"
    ].copy()

    if set(hist_primary["metric"].astype(str)) != set(METRICS):
        raise RuntimeError(
            "Historical PRIMARY-vs-M2 metric set changed."
        )

    print("historical R10K2C summary SHA:", sha256_file(R10K2C_SUMMARY))
    print("historical R10K2C delta SHA:", sha256_file(R10K2C_DELTA))
    print("bootstrap seed:", BOOTSTRAP_SEED)
    print("bootstrap replicates:", DEFAULT_BOOTSTRAPS)
    print("PASS")

    return hist_primary


def load_predictions():
    df = pd.read_csv(PREDICTIONS, low_memory=False)

    required = [
        "target_family",
        "seed",
        "method",
        "sample_id",
        "model_state_id",
        "harm_label",
        "probability",
        "source_only_threshold",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"Missing prediction columns: {missing}"
        )

    df = df[df["method"].isin(METHODS)].copy()
    df["_sid"] = norm_sid_series(df["sample_id"])

    df["seed"] = pd.to_numeric(
        df["seed"], errors="raise"
    ).astype(int)
    df["harm_label"] = pd.to_numeric(
        df["harm_label"], errors="raise"
    ).astype(int)
    df["probability"] = pd.to_numeric(
        df["probability"], errors="raise"
    ).astype(float)
    df["source_only_threshold"] = pd.to_numeric(
        df["source_only_threshold"], errors="raise"
    ).astype(float)

    expected_rows = (
        len(FAMILIES)
        * len(SEEDS)
        * len(METHODS)
        * 3000
    )
    print("\n===== R15A1 PREDICTION PANEL AUDIT =====")
    print("rows:", len(df))
    print("expected:", expected_rows)

    if len(df) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} rows, got {len(df)}."
        )
    if set(df["target_family"].unique()) != set(FAMILIES):
        raise RuntimeError("Unexpected target-family set.")
    if set(df["seed"].unique()) != set(SEEDS):
        raise RuntimeError("Unexpected split-seed set.")
    if set(df["method"].unique()) != set(METHODS):
        raise RuntimeError("Unexpected method set.")

    if not np.isfinite(df["probability"].to_numpy(float)).all():
        raise RuntimeError("Non-finite probabilities.")
    if not np.isfinite(
        df["source_only_threshold"].to_numpy(float)
    ).all():
        raise RuntimeError("Non-finite thresholds.")

    case_sets = []
    for family in FAMILIES:
        s = set(
            df.loc[
                df["target_family"] == family,
                "_sid",
            ].unique()
        )
        case_sets.append(s)

    if not all(s == case_sets[0] for s in case_sets[1:]):
        raise RuntimeError(
            "Target-family case sets are not exactly equal."
        )

    all_sids = sorted(case_sets[0])
    if len(all_sids) != 1000:
        raise RuntimeError(
            f"Expected 1000 unique cases, got {len(all_sids)}."
        )

    # Exact explicit key parity across all methods/families/seeds.
    for family in FAMILIES:
        for seed in SEEDS:
            ref = None
            for method in METHODS:
                sub = df[
                    (df["target_family"] == family)
                    & (df["seed"] == seed)
                    & (df["method"] == method)
                ]
                if len(sub) != 3000:
                    raise RuntimeError(
                        f"{family}/{seed}/{method}: rows !=3000."
                    )

                keys = set(
                    zip(
                        sub["_sid"].astype(str),
                        sub["model_state_id"].astype(str),
                    )
                )
                if len(keys) != 3000:
                    raise RuntimeError(
                        f"{family}/{seed}/{method}: key count !=3000."
                    )

                if ref is None:
                    ref = keys
                elif keys != ref:
                    raise RuntimeError(
                        f"Paired model-case key mismatch: "
                        f"{family}/{seed}/{method}"
                    )

    print("unique cases:", len(all_sids))
    print("methods:", len(METHODS))
    print("target families:", len(FAMILIES))
    print("split seeds:", len(SEEDS))
    print("paired explicit-key audit: PASS")

    return df, all_sids


def build_case_index(df):
    """
    Historical R10K2C structure:
      index[(family, seed, method)][sid] -> 3 row indices.
    """
    result = {}

    for family in FAMILIES:
        for seed in SEEDS:
            for method in METHODS:
                sub = df[
                    (df["target_family"] == family)
                    & (df["seed"] == seed)
                    & (df["method"] == method)
                ]

                groups = {}
                for sid, g in sub.groupby("_sid", sort=False):
                    if len(g) != 3:
                        raise RuntimeError(
                            f"{family}/{seed}/{method}/{sid}: "
                            f"expected 3 model-state rows, got {len(g)}"
                        )
                    groups[sid] = g.index.to_numpy(dtype=int)

                if len(groups) != 1000:
                    raise RuntimeError(
                        f"{family}/{seed}/{method}: "
                        f"expected 1000 cases, got {len(groups)}"
                    )

                result[(family, seed, method)] = groups

    return result


def compute_seed_family_metrics(
    df,
    case_index,
    sampled_sids,
):
    """
    Exact historical R10K2C resampling semantics.

    sampled_sids may contain duplicates. Each selected case contributes all
    three target-family state rows. The same sampled_sids list is shared by
    every method, family, and seed.
    """
    out = {}

    for family in FAMILIES:
        for seed in SEEDS:
            for method in METHODS:
                groups = case_index[(family, seed, method)]
                idx = np.concatenate([
                    groups[sid]
                    for sid in sampled_sids
                ])

                sub = df.loc[idx]

                out[(family, seed, method)] = metrics(
                    sub["harm_label"].to_numpy(int),
                    sub["probability"].to_numpy(float),
                    sub["source_only_threshold"].to_numpy(float),
                )

    return out


def aggregate_over_seeds_and_families(metric_map):
    """
    Exact historical aggregation:
      1) average 5 split-seed metrics within each target family;
      2) macro-average the 3 target-family means.
    """
    family_method = {}

    for family in FAMILIES:
        for method in METHODS:
            vals = {}
            for metric_name in METRICS:
                x = [
                    metric_map[(family, seed, method)][metric_name]
                    for seed in SEEDS
                ]
                vals[metric_name] = float(np.nanmean(x))
            family_method[(family, method)] = vals

    macro = {}
    for method in METHODS:
        vals = {}
        for metric_name in METRICS:
            x = [
                family_method[(family, method)][metric_name]
                for family in FAMILIES
            ]
            vals[metric_name] = float(np.nanmean(x))
        macro[method] = vals

    return family_method, macro


def compute_point_estimates(df, all_sids):
    case_index = build_case_index(df)
    metric_map = compute_seed_family_metrics(
        df,
        case_index,
        all_sids,
    )
    family_method, macro = aggregate_over_seeds_and_families(
        metric_map
    )

    family_rows = []
    for (family, method), vals in family_method.items():
        row = {
            "target_family": family,
            "method": method,
            **vals,
        }
        family_rows.append(row)

    macro_rows = []
    for method, vals in macro.items():
        macro_rows.append({
            "method": method,
            **vals,
        })

    return (
        pd.DataFrame(family_rows),
        pd.DataFrame(macro_rows),
        case_index,
    )


def audit_point_estimates_against_r15a0(macro_point_df):
    historical = pd.read_csv(R15A0_MACRO, low_memory=False)

    required = {
        "method",
        "macro_AUROC_mean",
        "macro_AUPRC_mean",
        "macro_transfer_Recall_mean",
        "macro_transfer_FPR_mean",
        "macro_PPV1pct_mean",
        "macro_oracle_R90_FPR_mean",
    }
    if not required.issubset(historical.columns):
        raise RuntimeError(
            "R15A0 macro summary schema changed."
        )

    hist = historical.set_index("method")
    new = macro_point_df.set_index("method")

    mapping = {
        "AUROC": "macro_AUROC_mean",
        "AUPRC": "macro_AUPRC_mean",
        "Recall": "macro_transfer_Recall_mean",
        "FPR": "macro_transfer_FPR_mean",
        "PPV1pct": "macro_PPV1pct_mean",
        "OracleR90FPR": "macro_oracle_R90_FPR_mean",
    }

    print("\n===== POINT-ESTIMATE REPRODUCTION SENTINEL =====")
    for method in METHODS:
        for metric, col in mapping.items():
            a = float(new.loc[method, metric])
            b = float(hist.loc[method, col])
            if abs(a - b) > 1e-12:
                raise RuntimeError(
                    f"Point-estimate mismatch: "
                    f"{method}/{metric}: {a} vs {b}"
                )
        print(method, "PASS")

    print("R15A0_POINT_ESTIMATE_REPRODUCTION_PASS")


def bootstrap(df, all_sids, case_index, n_bootstrap):
    rng = np.random.RandomState(BOOTSTRAP_SEED)

    bootstrap_rows = []
    family_comparison_rows = []

    for b in tqdm(
        range(n_bootstrap),
        desc="R15A1 paired case bootstrap",
        unit="rep",
        dynamic_ncols=True,
    ):
        sampled_sids = rng.choice(
            np.asarray(all_sids, dtype=object),
            size=len(all_sids),
            replace=True,
        )

        metric_map = compute_seed_family_metrics(
            df,
            case_index,
            sampled_sids,
        )
        fam, macro = aggregate_over_seeds_and_families(
            metric_map
        )

        for method in METHODS:
            bootstrap_rows.append({
                "bootstrap": b,
                "method": method,
                **macro[method],
            })

        # Supplemental family-level paired deltas.
        for comp_name, a, bmethod in COMPARISONS:
            for family in FAMILIES:
                aa = fam[(family, a)]
                bb = fam[(family, bmethod)]
                row = {
                    "bootstrap": b,
                    "target_family": family,
                    "comparison": comp_name,
                }
                for metric_name in METRICS:
                    row[f"delta_{metric_name}"] = (
                        aa[metric_name] - bb[metric_name]
                    )
                family_comparison_rows.append(row)

    return (
        pd.DataFrame(bootstrap_rows),
        pd.DataFrame(family_comparison_rows),
    )


def summarize_macro_bootstrap(boot):
    rows = []
    for method in METHODS:
        sub = boot[boot["method"] == method]
        for metric_name in METRICS:
            mean, lo, hi = percentile_ci(
                sub[metric_name]
            )
            rows.append({
                "level": "macro",
                "target_family": "MACRO",
                "method": method,
                "metric": metric_name,
                "bootstrap_mean": mean,
                "ci95_low": lo,
                "ci95_high": hi,
            })
    return pd.DataFrame(rows)


def summarize_paired_macro_deltas(boot):
    wide = boot.pivot(
        index="bootstrap",
        columns="method",
        values=METRICS,
    )

    rows = []

    for comp_name, a, b in COMPARISONS:
        for metric_name in METRICS:
            delta = (
                wide[(metric_name, a)]
                - wide[(metric_name, b)]
            ).to_numpy(float)

            mean, lo, hi = percentile_ci(delta)

            if metric_name in ("FPR", "OracleR90FPR"):
                favorable_fraction = float(
                    np.mean(delta < 0)
                )
                direction = (
                    "A_BETTER"
                    if hi < 0
                    else "A_WORSE"
                    if lo > 0
                    else "NO_95CI_DIFFERENCE"
                )
            else:
                favorable_fraction = float(
                    np.mean(delta > 0)
                )
                direction = (
                    "A_BETTER"
                    if lo > 0
                    else "A_WORSE"
                    if hi < 0
                    else "NO_95CI_DIFFERENCE"
                )

            rows.append({
                "comparison": comp_name,
                "method_A": a,
                "method_B": b,
                "metric": metric_name,
                "bootstrap_mean_delta": mean,
                "ci95_low": lo,
                "ci95_high": hi,
                "favorable_bootstrap_fraction": favorable_fraction,
                "inference": direction,
            })

    return pd.DataFrame(rows)


def summarize_family_deltas(family_boot):
    rows = []

    for comp_name, _, _ in COMPARISONS:
        for family in FAMILIES:
            sub = family_boot[
                (family_boot["comparison"] == comp_name)
                & (family_boot["target_family"] == family)
            ]
            for metric_name in METRICS:
                col = f"delta_{metric_name}"
                mean, lo, hi = percentile_ci(sub[col])

                if metric_name in ("FPR", "OracleR90FPR"):
                    inference = (
                        "A_BETTER"
                        if hi < 0
                        else "A_WORSE"
                        if lo > 0
                        else "NO_95CI_DIFFERENCE"
                    )
                else:
                    inference = (
                        "A_BETTER"
                        if lo > 0
                        else "A_WORSE"
                        if hi < 0
                        else "NO_95CI_DIFFERENCE"
                    )

                rows.append({
                    "target_family": family,
                    "comparison": comp_name,
                    "metric": metric_name,
                    "bootstrap_mean_delta": mean,
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "inference": inference,
                })

    return pd.DataFrame(rows)


def audit_historical_primary_bootstrap(delta_df, hist_primary):
    """
    Because R15A0 reuses the exact frozen R10K2B rows for FINAL and M2, and
    R15A1 reuses the exact R10K2C resampling seed/protocol, FINAL-vs-M2 must
    exactly reproduce the old PRIMARY-vs-M2 bootstrap table.
    """
    print("\n===== HISTORICAL PRIMARY-vs-M2 BOOTSTRAP SENTINEL =====")

    new = delta_df[
        delta_df["comparison"] == "FINAL_vs_M2"
    ].set_index("metric")
    old = hist_primary.set_index("metric")

    for metric_name in METRICS:
        for new_col, old_col in (
            ("bootstrap_mean_delta", "bootstrap_mean_delta"),
            ("ci95_low", "ci95_low"),
            ("ci95_high", "ci95_high"),
        ):
            a = float(new.loc[metric_name, new_col])
            b = float(old.loc[metric_name, old_col])
            if abs(a - b) > 1e-12:
                raise RuntimeError(
                    "Historical bootstrap reproduction failed: "
                    f"{metric_name}/{new_col}: {a} vs {b}"
                )

        print(
            metric_name,
            f"mean={float(new.loc[metric_name, 'bootstrap_mean_delta']):+.6f}",
            f"CI=[{float(new.loc[metric_name, 'ci95_low']):+.6f},"
            f"{float(new.loc[metric_name, 'ci95_high']):+.6f}]",
            "PASS",
        )

    print("HISTORICAL_R10K2C_PRIMARY_VS_M2_REPRODUCTION_PASS")


def build_claim_audit(delta_df):
    """
    Freeze only what the paired CIs support. This is explanatory inference,
    not method reselection.
    """
    d = delta_df.set_index(["comparison", "metric"])

    def inf(comp, metric):
        return str(d.loc[(comp, metric), "inference"])

    claims = {
        "prediction_conditioning_beats_image_only_CLS_by_AUROC": (
            inf("COND_vs_CLS", "AUROC") == "A_BETTER"
        ),
        "prediction_conditioning_beats_image_only_CLS_by_AUPRC": (
            inf("COND_vs_CLS", "AUPRC") == "A_BETTER"
        ),
        "FG_plus_BG_beats_FG_only_by_AUROC": (
            inf("COND_vs_FG", "AUROC") == "A_BETTER"
        ),
        "FG_plus_BG_beats_FG_only_by_AUPRC": (
            inf("COND_vs_FG", "AUPRC") == "A_BETTER"
        ),
        "FG_plus_BG_improves_oracle_R90_FPR_vs_FG_only": (
            inf("COND_vs_FG", "OracleR90FPR") == "A_BETTER"
        ),
        "FG_plus_BG_beats_BG_only_by_AUROC": (
            inf("COND_vs_BG", "AUROC") == "A_BETTER"
        ),
        "frozen_final_beats_M2_by_AUROC": (
            inf("FINAL_vs_M2", "AUROC") == "A_BETTER"
        ),
        "frozen_final_beats_M2_plus_FG_by_AUROC": (
            inf("FINAL_vs_M2_FG", "AUROC") == "A_BETTER"
        ),
        "frozen_final_improves_oracle_R90_FPR_vs_M2_plus_FG": (
            inf("FINAL_vs_M2_FG", "OracleR90FPR") == "A_BETTER"
        ),
        "M2_addition_beats_CondDINO_by_AUROC": (
            inf("FINAL_vs_COND", "AUROC") == "A_BETTER"
        ),
        "M2_addition_beats_CondDINO_by_AUPRC": (
            inf("FINAL_vs_COND", "AUPRC") == "A_BETTER"
        ),
    }

    return claims


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    ap.add_argument(
        "--n-bootstrap",
        type=int,
        default=DEFAULT_BOOTSTRAPS,
    )
    args = ap.parse_args()

    print("===== Q1 R15A1 CORE ABLATION PAIRED CASE BOOTSTRAP =====")
    print("STATUS=POST_FREEZE_EXPLANATORY_INFERENCE")
    print("MODEL_FITTING=NO")
    print("PCA_FITTING=NO")
    print("THRESHOLD_SELECTION=NO")
    print("FEATURE_CHANGES=NO")
    print("EXTERNAL_DATA_ACCESS=NO")
    print("FINAL_METHOD_RESELECTION=NO")
    print("CLUSTER_UNIT=sample_id")
    print("ROWS_PER_TARGET_FAMILY_CASE=3 model states")
    print("SAME_BOOTSTRAP_CASES_ACROSS_ALL_METHODS_FAMILIES_SEEDS=YES")
    print("BOOTSTRAPS=", args.n_bootstrap)
    print("BOOTSTRAP_SEED=", BOOTSTRAP_SEED)
    print("METHODS=", METHODS)

    if args.n_bootstrap != DEFAULT_BOOTSTRAPS:
        raise RuntimeError(
            "R15A1 is frozen to the historical 2000 bootstrap replicates."
        )

    verify_r15a0()
    hist_primary = verify_historical_r10k2c()

    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)

    df, all_sids = load_predictions()
    (
        point_df,
        macro_point_df,
        case_index,
    ) = compute_point_estimates(
        df,
        all_sids,
    )

    audit_point_estimates_against_r15a0(
        macro_point_df
    )

    print("\n===== R15A1 POINT ESTIMATES =====")
    print(macro_point_df.to_string(index=False))

    boot, family_boot = bootstrap(
        df,
        all_sids,
        case_index,
        args.n_bootstrap,
    )

    ci_df = summarize_macro_bootstrap(boot)
    delta_df = summarize_paired_macro_deltas(boot)
    family_delta_df = summarize_family_deltas(
        family_boot
    )

    audit_historical_primary_bootstrap(
        delta_df,
        hist_primary,
    )

    print("\n===== PAIRED MACRO CORE-ABLATION DELTA 95% CI =====")
    display_cols = [
        "comparison",
        "metric",
        "bootstrap_mean_delta",
        "ci95_low",
        "ci95_high",
        "favorable_bootstrap_fraction",
        "inference",
    ]
    print(delta_df[display_cols].to_string(index=False))

    print("\n===== PAPER-CRITICAL COMPARISONS =====")
    for comp in [
        "COND_vs_CLS",
        "COND_vs_FG",
        "COND_vs_BG",
        "FINAL_vs_M2",
        "FINAL_vs_M2_FG",
        "FINAL_vs_M2_BG",
        "FINAL_vs_COND",
    ]:
        sub = delta_df[
            (delta_df["comparison"] == comp)
            & (
                delta_df["metric"].isin(
                    ["AUROC", "AUPRC", "FPR", "OracleR90FPR"]
                )
            )
        ]
        print("\n", comp)
        print(
            sub[
                [
                    "metric",
                    "bootstrap_mean_delta",
                    "ci95_low",
                    "ci95_high",
                    "inference",
                ]
            ].to_string(index=False)
        )

    claims = build_claim_audit(delta_df)

    print("\n===== CLAIM AUDIT =====")
    for key, value in claims.items():
        print(key, "=", value)

    # Save only summary-level bootstrap products. Raw 2000x8 macro rows are
    # useful for audit and small enough to retain; family bootstrap raw rows
    # are not needed after the family CI table has been created.
    point_path = (
        args.output_dir
        / "R15A1_point_estimates_by_family.csv"
    )
    macro_point_path = (
        args.output_dir
        / "R15A1_point_estimates_macro.csv"
    )
    boot_path = (
        args.output_dir
        / "R15A1_macro_bootstrap_replicates.csv"
    )
    ci_path = (
        args.output_dir
        / "R15A1_macro_bootstrap_ci.csv"
    )
    delta_path = (
        args.output_dir
        / "R15A1_paired_macro_component_delta_ci.csv"
    )
    family_delta_path = (
        args.output_dir
        / "R15A1_family_component_delta_ci.csv"
    )
    claim_path = (
        args.output_dir
        / "R15A1_component_claim_audit.json"
    )

    point_df.to_csv(point_path, index=False)
    macro_point_df.to_csv(macro_point_path, index=False)
    boot.to_csv(boot_path, index=False)
    ci_df.to_csv(ci_path, index=False)
    delta_df.to_csv(delta_path, index=False)
    family_delta_df.to_csv(
        family_delta_path,
        index=False,
    )
    claim_path.write_text(
        json.dumps(claims, indent=2),
        encoding="utf-8",
    )

    artifacts = {}
    for p in (
        point_path,
        macro_point_path,
        boot_path,
        ci_path,
        delta_path,
        family_delta_path,
        claim_path,
    ):
        artifacts[p.name] = {
            "sha256": sha256_file(p),
            "bytes": p.stat().st_size,
        }

    lock = {
        "status": "PASS",
        "decision": DECISION,
        "script_version": VERSION,
        "build": BUILD,
        "scientific_role": (
            "post-freeze explanatory paired case-clustered "
            "inference for representation components"
        ),
        "final_method_reselected": False,
        "external_data_accessed": False,
        "bootstrap": {
            "cluster_unit": "sample_id",
            "cases": 1000,
            "rows_per_target_family_case": 3,
            "target_families": FAMILIES,
            "split_seeds": SEEDS,
            "replicates": args.n_bootstrap,
            "seed": BOOTSTRAP_SEED,
            "same_case_draw_across_methods_families_seeds": True,
            "aggregation": (
                "mean over 5 split seeds within family, "
                "then macro mean over 3 target families"
            ),
        },
        "methods": METHODS,
        "comparisons": [
            {
                "name": name,
                "method_A": a,
                "method_B": b,
            }
            for name, a, b in COMPARISONS
        ],
        "historical_r10k2c_primary_vs_m2_reproduced_exactly": True,
        "claim_audit": claims,
        "interpretation_boundary": {
            "component_ablation_only": True,
            "frozen_final_method_changed": False,
            "new_external_evidence_claimed": False,
            "non_significant_component_differences_must_not_be_overclaimed": True,
        },
        "upstream": {
            "r15a0_lock_sha256": EXPECTED_R15A0_LOCK_SHA,
            "r15a0_predictions_sha256": sha256_file(PREDICTIONS),
            "historical_r10k2c_delta_sha256": sha256_file(R10K2C_DELTA),
            "historical_r10k2c_summary_sha256": sha256_file(R10K2C_SUMMARY),
        },
        "artifacts": artifacts,
        "next_stage": (
            "R15B_SELECTIVE_ADAPTATION_RISK_COVERAGE_UTILITY_"
            "USING_ALREADY_FROZEN_EXTERNAL_SCORES"
        ),
    }

    lock_path = (
        args.output_dir
        / "R15A1_CORE_ABLATION_PAIRED_BOOTSTRAP_LOCK.json"
    )
    lock_path.write_text(
        json.dumps(lock, indent=2),
        encoding="utf-8",
    )

    print("\n===== R15A1 FINAL =====")
    print("Decision=", DECISION)
    print("FINAL_METHOD_RESELECTION=NO")
    print("LOCK=", lock_path)
    print("LOCK SHA256=", sha256_file(lock_path))
    print("PASS")


if __name__ == "__main__":
    main()
