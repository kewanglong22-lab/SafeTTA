#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R10K2C_case_clustered_paired_bootstrap_fix1.py

Confirmatory inference for the already-frozen R10K2B predictions.

Purpose
-------
R10K2B strongly supports the frozen primary representation:
    M2_plus_CondDINO_PCA64

R10K2C does NOT retrain anything. It quantifies case-level uncertainty while
respecting the repeated-measures structure:

    1000 unique sample_id cases
      x 3 held-out model families
      x 3 model states/family
      x 5 split seeds

The same sampled case IDs are used for all families, methods, and seeds within
each bootstrap replicate, preserving paired comparisons.

Primary comparison
------------------
    M2_plus_CondDINO_PCA64  vs  M2_backbone

Secondary decomposition
-----------------------
    CondDINO_PCA64          vs  M2_backbone
    M2_plus_CondDINO_PCA64  vs  CondDINO_PCA64

No fitting, threshold selection, or method tuning occurs here.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.metrics import roc_auc_score, average_precision_score


DEFAULT_PREDICTIONS = (
    "F:/MEDSEG_SAFETTA/outputs/"
    "Q1_R10K2B_dinov2_image_mask_LOMO_evaluation_fix2_v1/"
    "R10K2B_target_predictions.csv"
)

DEFAULT_OUTPUT = (
    "F:/MEDSEG_SAFETTA/outputs/"
    "Q1_R10K2C_case_clustered_paired_bootstrap_fix1_v1"
)

PRIMARY = "M2_plus_CondDINO_PCA64"
BASELINE = "M2_backbone"
COND = "CondDINO_PCA64"

METHODS = [BASELINE, COND, PRIMARY]
FAMILIES = ["DeepLabV3-R50", "PraNet", "SegFormer-B0"]
SEEDS = [20260816, 20260817, 20260818, 20260819, 20260820]

TARGET_RECALL = 0.90
PPV_PREVALENCE = 0.01
BOOTSTRAP_SEED = 20260826
DEFAULT_BOOTSTRAPS = 2000


def norm_sid_series(s):
    return (
        s.astype(str)
        .str.strip()
        .str.replace("\\", "/", regex=False)
        .str.lower()
    )


def threshold_for_recall(y, p, target):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)

    pos = np.sort(p[y == 1])[::-1]
    if len(pos) == 0:
        return np.nan

    k = int(np.ceil(target * len(pos)))
    k = min(max(k, 1), len(pos))
    return float(pos[k - 1])


def metrics(y, p, thresholds):
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


def build_case_index(df):
    """
    Return nested mapping:
      index[(family, seed, method)][sid] -> row indices
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

                if len(sub) != 3000:
                    raise AssertionError(
                        f"{family}/{seed}/{method}: expected 3000 rows, "
                        f"got {len(sub)}"
                    )

                groups = {}
                for sid, g in sub.groupby("_sid", sort=False):
                    if len(g) != 3:
                        raise AssertionError(
                            f"{family}/{seed}/{method}/{sid}: "
                            f"expected 3 model-state rows, got {len(g)}"
                        )
                    groups[sid] = g.index.to_numpy(dtype=int)

                if len(groups) != 1000:
                    raise AssertionError(
                        f"{family}/{seed}/{method}: "
                        f"expected 1000 cases, got {len(groups)}"
                    )

                result[(family, seed, method)] = groups

    return result


def compute_seed_family_metrics(df, case_index, sampled_sids):
    """
    sampled_sids can contain duplicates. Each selected case contributes all
    three state rows in the target family.
    """
    out = {}

    for family in FAMILIES:
        for seed in SEEDS:
            for method in METHODS:
                groups = case_index[(family, seed, method)]
                idx = np.concatenate([
                    groups[sid] for sid in sampled_sids
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
    For each method:
      1) average metric over the 5 split seeds within each target family;
      2) macro-average the three target-family means.
    """
    family_method = {}

    for family in FAMILIES:
        for method in METHODS:
            vals = {}
            for metric_name in [
                "AUROC",
                "AUPRC",
                "Recall",
                "FPR",
                "PPV1pct",
                "OracleR90FPR",
            ]:
                x = [
                    metric_map[(family, seed, method)][metric_name]
                    for seed in SEEDS
                ]
                vals[metric_name] = float(np.nanmean(x))
            family_method[(family, method)] = vals

    macro = {}
    for method in METHODS:
        vals = {}
        for metric_name in [
            "AUROC",
            "AUPRC",
            "Recall",
            "FPR",
            "PPV1pct",
            "OracleR90FPR",
        ]:
            x = [
                family_method[(family, method)][metric_name]
                for family in FAMILIES
            ]
            vals[metric_name] = float(np.nanmean(x))
        macro[method] = vals

    return family_method, macro


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--predictions",
        default=DEFAULT_PREDICTIONS,
    )
    ap.add_argument(
        "--output_dir",
        default=DEFAULT_OUTPUT,
    )
    ap.add_argument(
        "--n_bootstrap",
        type=int,
        default=DEFAULT_BOOTSTRAPS,
    )
    ap.add_argument(
        "--bootstrap_seed",
        type=int,
        default=BOOTSTRAP_SEED,
    )
    args = ap.parse_args()

    pred_path = Path(args.predictions)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("===== R10K2C CASE-CLUSTERED PAIRED BOOTSTRAP FIX1 =====")
    print("STATUS: CONFIRMATORY INFERENCE ON FROZEN PREDICTIONS")
    print("MODEL FITTING: NONE")
    print("PCA FITTING: NONE")
    print("THRESHOLD SELECTION: NONE")
    print("METHOD TUNING: NONE")
    print("CLUSTER UNIT: sample_id")
    print("ROWS KEPT TOGETHER PER TARGET FAMILY/CASE: 3 model states")
    print("SAME RESAMPLED CASE IDs USED ACROSS METHODS/FAMILIES/SEEDS: YES")
    print("PRIMARY:", PRIMARY, "vs", BASELINE)
    print("BOOTSTRAPS:", args.n_bootstrap)
    print("BOOTSTRAP SEED:", args.bootstrap_seed)
    print()

    print("predictions exists:", pred_path.exists(), pred_path)
    if not pred_path.exists():
        raise FileNotFoundError(pred_path)

    df = pd.read_csv(pred_path, low_memory=False)

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
        raise AssertionError(f"Missing prediction columns: {missing}")

    df = df[df["method"].isin(METHODS)].copy()
    df["_sid"] = norm_sid_series(df["sample_id"])
    df["seed"] = pd.to_numeric(df["seed"], errors="raise").astype(int)
    df["harm_label"] = pd.to_numeric(
        df["harm_label"], errors="raise"
    ).astype(int)
    df["probability"] = pd.to_numeric(
        df["probability"], errors="raise"
    ).astype(float)
    df["source_only_threshold"] = pd.to_numeric(
        df["source_only_threshold"], errors="raise"
    ).astype(float)

    expected_rows = 3 * 5 * 3 * 3000
    print("Rows after method filter:", len(df))
    print("Expected rows:", expected_rows)
    if len(df) != expected_rows:
        raise AssertionError(
            f"Expected {expected_rows} rows, got {len(df)}."
        )

    if set(df["target_family"].unique()) != set(FAMILIES):
        raise AssertionError("Unexpected target-family set.")
    if set(df["seed"].unique()) != set(SEEDS):
        raise AssertionError("Unexpected split-seed set.")
    if set(df["method"].unique()) != set(METHODS):
        raise AssertionError("Unexpected method set.")

    case_sets = []
    for family in FAMILIES:
        s = set(df.loc[
            df["target_family"] == family,
            "_sid",
        ].unique())
        case_sets.append(s)

    if not all(s == case_sets[0] for s in case_sets[1:]):
        raise AssertionError(
            "Target-family case sets are not exactly equal."
        )

    all_sids = sorted(case_sets[0])
    if len(all_sids) != 1000:
        raise AssertionError(
            f"Expected 1000 unique cases, got {len(all_sids)}."
        )

    print("Unique cases:", len(all_sids))

    case_index = build_case_index(df)

    # --------------------------------------------------------------
    # Point estimates on the original 1000 cases.
    # --------------------------------------------------------------
    point_map = compute_seed_family_metrics(
        df,
        case_index,
        all_sids,
    )
    point_family, point_macro = aggregate_over_seeds_and_families(
        point_map
    )

    print("\n===== ORIGINAL CASE-LEVEL POINT ESTIMATES =====")
    point_rows = []
    for family in FAMILIES:
        for method in METHODS:
            row = {
                "target_family": family,
                "method": method,
                **point_family[(family, method)],
            }
            point_rows.append(row)

    point_df = pd.DataFrame(point_rows)
    print(point_df.to_string(index=False))

    print("\nMACRO:")
    macro_point_df = pd.DataFrame([
        {"method": method, **point_macro[method]}
        for method in METHODS
    ])
    print(macro_point_df.to_string(index=False))

    # --------------------------------------------------------------
    # Paired case-clustered bootstrap.
    # --------------------------------------------------------------
    rng = np.random.default_rng(args.bootstrap_seed)

    bootstrap_rows = []
    family_delta_rows = []

    for b in tqdm(
        range(args.n_bootstrap),
        desc="Case-clustered bootstrap",
        unit="rep",
        dynamic_ncols=True,
    ):
        sampled = rng.choice(
            all_sids,
            size=len(all_sids),
            replace=True,
        ).tolist()

        metric_map = compute_seed_family_metrics(
            df,
            case_index,
            sampled,
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

        # Family-level primary paired deltas.
        for family in FAMILIES:
            p = fam[(family, PRIMARY)]
            q = fam[(family, BASELINE)]
            family_delta_rows.append({
                "bootstrap": b,
                "target_family": family,
                "delta_AUROC": p["AUROC"] - q["AUROC"],
                "delta_AUPRC": p["AUPRC"] - q["AUPRC"],
                "delta_Recall": p["Recall"] - q["Recall"],
                "delta_FPR": p["FPR"] - q["FPR"],
                "delta_PPV1pct": p["PPV1pct"] - q["PPV1pct"],
                "delta_OracleR90FPR": (
                    p["OracleR90FPR"] - q["OracleR90FPR"]
                ),
            })

    boot = pd.DataFrame(bootstrap_rows)
    fam_delta = pd.DataFrame(family_delta_rows)

    # --------------------------------------------------------------
    # Macro method CIs.
    # --------------------------------------------------------------
    ci_rows = []
    for method in METHODS:
        sub = boot[boot["method"] == method]
        for metric_name in [
            "AUROC",
            "AUPRC",
            "Recall",
            "FPR",
            "PPV1pct",
            "OracleR90FPR",
        ]:
            mean, lo, hi = percentile_ci(sub[metric_name])
            ci_rows.append({
                "level": "macro",
                "target_family": "MACRO",
                "method": method,
                "metric": metric_name,
                "bootstrap_mean": mean,
                "ci95_low": lo,
                "ci95_high": hi,
            })

    ci_df = pd.DataFrame(ci_rows)

    # --------------------------------------------------------------
    # Paired macro deltas from the same bootstrap replicate.
    # --------------------------------------------------------------
    wide = boot.pivot(
        index="bootstrap",
        columns="method",
        values=[
            "AUROC",
            "AUPRC",
            "Recall",
            "FPR",
            "PPV1pct",
            "OracleR90FPR",
        ],
    )

    comparisons = [
        ("PRIMARY_vs_M2", PRIMARY, BASELINE),
        ("COND_vs_M2", COND, BASELINE),
        ("PRIMARY_vs_COND", PRIMARY, COND),
    ]

    delta_rows = []

    for comp_name, a, b in comparisons:
        for metric_name in [
            "AUROC",
            "AUPRC",
            "Recall",
            "FPR",
            "PPV1pct",
            "OracleR90FPR",
        ]:
            delta = (
                wide[(metric_name, a)]
                - wide[(metric_name, b)]
            ).to_numpy(float)

            mean, lo, hi = percentile_ci(delta)

            if metric_name in ["FPR", "OracleR90FPR"]:
                favorable_fraction = float(np.mean(delta < 0))
            else:
                favorable_fraction = float(np.mean(delta > 0))

            delta_rows.append({
                "comparison": comp_name,
                "metric": metric_name,
                "bootstrap_mean_delta": mean,
                "ci95_low": lo,
                "ci95_high": hi,
                "favorable_bootstrap_fraction": favorable_fraction,
            })

    delta_df = pd.DataFrame(delta_rows)

    # --------------------------------------------------------------
    # Family-level primary paired delta CIs.
    # --------------------------------------------------------------
    fam_ci_rows = []
    for family in FAMILIES:
        sub = fam_delta[
            fam_delta["target_family"] == family
        ]
        for metric_col in [
            "delta_AUROC",
            "delta_AUPRC",
            "delta_Recall",
            "delta_FPR",
            "delta_PPV1pct",
            "delta_OracleR90FPR",
        ]:
            mean, lo, hi = percentile_ci(sub[metric_col])
            fam_ci_rows.append({
                "target_family": family,
                "comparison": "PRIMARY_vs_M2",
                "metric": metric_col.replace("delta_", ""),
                "bootstrap_mean_delta": mean,
                "ci95_low": lo,
                "ci95_high": hi,
            })

    fam_ci_df = pd.DataFrame(fam_ci_rows)

    print("\n===== MACRO BOOTSTRAP 95% CI =====")
    print(ci_df.to_string(index=False))

    print("\n===== PAIRED MACRO DELTA 95% CI =====")
    print(delta_df.to_string(index=False))

    print("\n===== FAMILY PRIMARY-vs-M2 DELTA 95% CI =====")
    print(fam_ci_df.to_string(index=False))

    # --------------------------------------------------------------
    # Frozen inference decision.
    # --------------------------------------------------------------
    primary_delta = delta_df[
        delta_df["comparison"] == "PRIMARY_vs_M2"
    ].set_index("metric")

    macro_auc_confirm = (
        primary_delta.loc["AUROC", "ci95_low"] > 0
    )
    macro_auprc_confirm = (
        primary_delta.loc["AUPRC", "ci95_low"] > 0
    )
    macro_fpr_confirm = (
        primary_delta.loc["FPR", "ci95_high"] < 0
    )
    macro_oracle_confirm = (
        primary_delta.loc["OracleR90FPR", "ci95_high"] < 0
    )

    fam_auc = fam_ci_df[
        fam_ci_df["metric"] == "AUROC"
    ]
    all_family_auc_positive = bool(
        (fam_auc["ci95_low"] > 0).all()
    )

    if (
        macro_auc_confirm
        and macro_auprc_confirm
        and macro_fpr_confirm
        and macro_oracle_confirm
        and all_family_auc_positive
    ):
        decision = "PAIRED_CLUSTER_BOOTSTRAP_STRONGLY_CONFIRMS_PRIMARY"
    elif (
        macro_auc_confirm
        and macro_fpr_confirm
        and macro_oracle_confirm
    ):
        decision = "PAIRED_CLUSTER_BOOTSTRAP_CONFIRMS_MACRO_PRIMARY_BENEFIT"
    else:
        decision = "PAIRED_CLUSTER_BOOTSTRAP_DOES_NOT_CONFIRM_PRIMARY"

    print("\n===== FINAL INFERENCE DECISION =====")
    print("Macro AUROC delta CI > 0:", macro_auc_confirm)
    print("Macro AUPRC delta CI > 0:", macro_auprc_confirm)
    print("Macro transferred FPR delta CI < 0:", macro_fpr_confirm)
    print("Macro oracle R90 FPR delta CI < 0:", macro_oracle_confirm)
    print("All family AUROC delta CIs > 0:", all_family_auc_positive)
    print("DECISION:", decision)

    point_df.to_csv(
        out / "R10K2C_point_estimates_by_family.csv",
        index=False,
    )
    macro_point_df.to_csv(
        out / "R10K2C_point_estimates_macro.csv",
        index=False,
    )
    ci_df.to_csv(
        out / "R10K2C_macro_bootstrap_ci.csv",
        index=False,
    )
    delta_df.to_csv(
        out / "R10K2C_paired_macro_delta_ci.csv",
        index=False,
    )
    fam_ci_df.to_csv(
        out / "R10K2C_family_primary_delta_ci.csv",
        index=False,
    )

    summary = {
        "decision": decision,
        "cluster_unit": "sample_id",
        "cases": 1000,
        "states_per_target_family_case": 3,
        "target_families": FAMILIES,
        "split_seeds": SEEDS,
        "bootstrap_replicates": args.n_bootstrap,
        "bootstrap_seed": args.bootstrap_seed,
        "primary": PRIMARY,
        "baseline": BASELINE,
        "macro_auc_delta_ci_low_gt_0": bool(macro_auc_confirm),
        "macro_auprc_delta_ci_low_gt_0": bool(macro_auprc_confirm),
        "macro_fpr_delta_ci_high_lt_0": bool(macro_fpr_confirm),
        "macro_oracle_fpr_delta_ci_high_lt_0": bool(macro_oracle_confirm),
        "all_family_auc_delta_ci_low_gt_0": bool(
            all_family_auc_positive
        ),
    }

    with open(
        out / "R10K2C_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nOutputs:")
    print(out / "R10K2C_point_estimates_by_family.csv")
    print(out / "R10K2C_point_estimates_macro.csv")
    print(out / "R10K2C_macro_bootstrap_ci.csv")
    print(out / "R10K2C_paired_macro_delta_ci.csv")
    print(out / "R10K2C_family_primary_delta_ci.csv")
    print(out / "R10K2C_summary.json")
    print("PASS")


if __name__ == "__main__":
    main()
