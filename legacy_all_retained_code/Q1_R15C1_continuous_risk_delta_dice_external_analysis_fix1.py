#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R15C1_continuous_risk_delta_dice_external_analysis_fix1.py

Threshold-free post-freeze external analysis of whether the already-frozen
pre-adaptation safety score tracks continuous adaptation damage.

Primary continuous endpoint:
    Spearman( frozen safety probability, -DeltaDice )

Higher score means higher predicted HARM risk, so a positive correlation means
that higher pre-adaptation risk is associated with more negative adaptation
utility.

Settings:
  1) PolypGen / TENT1
  2) SUN-SEG / TENT1
  3) SUN-SEG / PL_CONF90

Primary statistic:
- compute Spearman rho within each of the 3 model families;
- macro-average the 3 family rhos.

Secondary:
- pooled model-frame Spearman rho;
- label-free frozen score quintiles (Q1=lowest risk, Q5=highest risk);
- mean/median DeltaDice, HARM prevalence and BENEFIT prevalence per quintile;
- Q5 minus Q1 difference in mean DeltaDice and HARM prevalence.

Bootstrap:
- PolypGen cluster = sample_id;
- SUN cluster = physical case/video cluster_id;
- 2000 replicates;
- identical SUN cluster draws for TENT1 and PL;
- quintile membership is frozen once from score ranking before outcome analysis.

No model fitting, score recomputation, threshold tuning, target calibration,
coverage selection, feature changes, or case exclusion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


VERSION = "2026-09-04-Q1-R15C1-v1-fix1"
BUILD = "Q1_R15C1_CONTINUOUS_RISK_DELTA_DICE_EXTERNAL_ANALYSIS_FIX1"

ROOT = Path(r"F:\MEDSEG_SAFETTA")
OUT = ROOT / "outputs"

R15B0_DIR = (
    OUT
    / "Q1_R15B0_selective_adaptation_external_utility_preflight_fix1_v1"
)
R15B0_LOCK = (
    R15B0_DIR
    / "R15B0_SELECTIVE_ADAPTATION_UTILITY_INPUT_LOCK.json"
)
EXPECTED_R15B0_LOCK_SHA = (
    "49bdf197919059c115156a44eaa57a9304c5c1f11545f9e2841f657856af4713"
)

R15B1_DIR = (
    OUT
    / "Q1_R15B1_selective_adaptation_risk_coverage_utility_fix1_v1"
)
R15B1_LOCK = (
    R15B1_DIR
    / "R15B1_SELECTIVE_ADAPTATION_UTILITY_LOCK.json"
)
EXPECTED_R15B1_LOCK_SHA = (
    "4f0ce882a5e040a7e4f19dd140daab9b2bef3fe3be22fb63a9c62748dd64f981"
)
EXPECTED_R15B1_DECISION = (
    "EXTERNAL_SELECTIVE_ADAPTATION_UTILITY_COMPLETE_NO_TARGET_POLICY_RETUNING"
)

SETTINGS = [
    "PolypGen_TENT1",
    "SUN_TENT1",
    "SUN_PL_CONF90",
]

N_QUINTILES = 5
BOOTSTRAPS = 2000
POLYPGEN_BOOTSTRAP_SEED = 20260906
SUN_SHARED_BOOTSTRAP_SEED = 20260907

DEFAULT_OUTPUT = (
    OUT
    / "Q1_R15C1_continuous_risk_delta_dice_external_analysis_fix1_v1"
)

DECISION = (
    "CONTINUOUS_EXTERNAL_RISK_SEVERITY_ANALYSIS_COMPLETE_"
    "NO_TARGET_RETUNING"
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


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def percentile_ci(values):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return math.nan, math.nan, math.nan
    return (
        float(np.mean(x)),
        float(np.quantile(x, 0.025)),
        float(np.quantile(x, 0.975)),
    )


def spearman_rho(x, y):
    """
    Standard Spearman rho = Pearson correlation of average ranks.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    keep = np.isfinite(x) & np.isfinite(y)
    x = x[keep]
    y = y[keep]

    if len(x) < 3:
        return math.nan
    if np.all(x == x[0]) or np.all(y == y[0]):
        return math.nan

    rx = pd.Series(x).rank(method="average").to_numpy(float)
    ry = pd.Series(y).rank(method="average").to_numpy(float)

    return float(np.corrcoef(rx, ry)[0, 1])


def weighted_mean(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    keep = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not keep.any():
        return math.nan
    return float(
        np.sum(values[keep] * weights[keep])
        / np.sum(weights[keep])
    )


def weighted_median(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    keep = np.isfinite(values) & np.isfinite(weights) & (weights > 0)

    if not keep.any():
        return math.nan

    values = values[keep]
    weights = weights[keep]

    order = np.argsort(values, kind="mergesort")
    values = values[order]
    weights = weights[order]

    cutoff = 0.5 * float(weights.sum())
    idx = int(
        np.searchsorted(
            np.cumsum(weights),
            cutoff,
            side="left",
        )
    )
    idx = min(idx, len(values) - 1)
    return float(values[idx])


def verify_upstream():
    print("===== R15C1 FROZEN UPSTREAM GATES =====")

    if not R15B0_LOCK.is_file():
        raise FileNotFoundError(R15B0_LOCK)
    if not R15B1_LOCK.is_file():
        raise FileNotFoundError(R15B1_LOCK)

    b0_sha = sha256_file(R15B0_LOCK)
    b1_sha = sha256_file(R15B1_LOCK)

    if b0_sha != EXPECTED_R15B0_LOCK_SHA:
        raise RuntimeError(
            f"R15B0 lock SHA changed: {b0_sha}"
        )
    if b1_sha != EXPECTED_R15B1_LOCK_SHA:
        raise RuntimeError(
            f"R15B1 lock SHA changed: {b1_sha}"
        )

    b0 = load_json(R15B0_LOCK)
    b1 = load_json(R15B1_LOCK)

    if b0.get("status") != "PASS":
        raise RuntimeError("R15B0 status changed.")
    if b1.get("status") != "PASS":
        raise RuntimeError("R15B1 status changed.")
    if b1.get("decision") != EXPECTED_R15B1_DECISION:
        raise RuntimeError("R15B1 decision changed.")

    boundary = b1.get("information_boundary", {})
    forbidden_true = (
        "model_fitting",
        "pca_fitting",
        "score_recomputation",
        "target_calibration",
        "threshold_tuning",
        "coverage_selection_by_gt",
        "feature_change",
        "case_exclusion",
        "action_specific_safety_retraining",
        "best_coverage_selected_or_deployed",
    )
    for key in forbidden_true:
        if bool(boundary.get(key, True)):
            raise RuntimeError(
                f"R15B1 information-boundary failure: {key}"
            )

    print("R15B0 lock SHA:", b0_sha)
    print("R15B1 lock SHA:", b1_sha)
    print("PASS")

    return b0, b1


def load_setting(setting, meta):
    path = Path(meta["panel_path"])
    if not path.is_file():
        raise FileNotFoundError(path)

    actual_sha = sha256_file(path)
    if actual_sha != str(meta["panel_sha256"]):
        raise RuntimeError(
            f"{setting}: frozen panel SHA changed."
        )

    raw = pd.read_csv(path, low_memory=False)

    score_col = str(meta["score_column"])
    delta_col = str(meta["delta_column"])
    harm_col = str(meta["harm_column"])
    benefit_col = str(meta["benefit_column"])
    cluster_col = str(meta["cluster_unit"])

    required = {
        "sample_id",
        "model_family",
        "model_state_id",
        score_col,
        delta_col,
        harm_col,
        benefit_col,
        cluster_col,
    }
    missing = sorted(required - set(raw.columns))
    if missing:
        raise RuntimeError(
            f"{setting}: missing columns {missing}"
        )

    df = pd.DataFrame({
        "sample_id": raw["sample_id"].astype(str),
        "model_family": raw["model_family"].astype(str),
        "model_state_id": raw["model_state_id"].astype(str),
        "cluster_id": raw[cluster_col].astype(str),
        "score": pd.to_numeric(
            raw[score_col], errors="raise"
        ).astype(float),
        "delta_dice": pd.to_numeric(
            raw[delta_col], errors="raise"
        ).astype(float),
        "harm_label": pd.to_numeric(
            raw[harm_col], errors="raise"
        ).astype(int),
        "benefit_label": pd.to_numeric(
            raw[benefit_col], errors="raise"
        ).astype(int),
    })

    if not np.isfinite(
        df[["score", "delta_dice"]].to_numpy(float)
    ).all():
        raise RuntimeError(
            f"{setting}: non-finite score/delta."
        )

    if (
        df[["sample_id", "model_state_id"]]
        .drop_duplicates()
        .shape[0]
        != len(df)
    ):
        raise RuntimeError(
            f"{setting}: model-frame key not unique."
        )

    families = sorted(df["model_family"].unique())
    if len(families) != 3:
        raise RuntimeError(
            f"{setting}: expected 3 model families, got {families}"
        )

    # Freeze equal-count score quintiles WITHOUT using outcome labels.
    # Stable sort ensures deterministic membership under score ties.
    order = (
        df.reset_index()
        .sort_values(
            [
                "score",
                "model_family",
                "model_state_id",
                "sample_id",
                "index",
            ],
            kind="mergesort",
        )
        ["index"]
        .to_numpy(dtype=int)
    )

    quintile = np.empty(len(df), dtype=int)
    for rank_pos, row_idx in enumerate(order):
        q = int(
            np.floor(
                rank_pos * N_QUINTILES / len(df)
            )
        ) + 1
        q = min(max(q, 1), N_QUINTILES)
        quintile[row_idx] = q

    df["risk_quintile"] = quintile

    counts = (
        df["risk_quintile"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    print("\n=====", setting, "=====")
    print("panel:", path)
    print("SHA256:", actual_sha)
    print("rows:", len(df))
    print("clusters:", df["cluster_id"].nunique())
    print("families:", families)
    print("score quintile counts:", counts)
    print("quintiles defined from score only: YES")
    print("PASS")

    return df, families, path


def family_macro_spearman(df, expanded_idx=None):
    if expanded_idx is None:
        work = df
    else:
        work = df.iloc[expanded_idx]

    values = []
    family_rows = []

    for family, g in work.groupby(
        "model_family",
        sort=True,
    ):
        rho = spearman_rho(
            g["score"].to_numpy(float),
            -g["delta_dice"].to_numpy(float),
        )
        values.append(rho)
        family_rows.append({
            "model_family": family,
            "spearman_score_vs_negative_delta": rho,
        })

    return float(np.nanmean(values)), family_rows


def point_analysis(setting, df):
    macro_rho, family_rows = family_macro_spearman(df)

    pooled_rho = spearman_rho(
        df["score"].to_numpy(float),
        -df["delta_dice"].to_numpy(float),
    )

    quintile_rows = []
    for q in range(1, N_QUINTILES + 1):
        g = df[df["risk_quintile"] == q]

        quintile_rows.append({
            "setting": setting,
            "risk_quintile": q,
            "rows": len(g),
            "score_mean": float(g["score"].mean()),
            "score_median": float(g["score"].median()),
            "delta_dice_mean": float(g["delta_dice"].mean()),
            "delta_dice_median": float(g["delta_dice"].median()),
            "harm_prevalence": float(g["harm_label"].mean()),
            "benefit_prevalence": float(g["benefit_label"].mean()),
        })

    qdf = pd.DataFrame(quintile_rows)
    q1 = qdf[qdf["risk_quintile"] == 1].iloc[0]
    q5 = qdf[qdf["risk_quintile"] == 5].iloc[0]

    contrast = {
        "setting": setting,
        "q5_minus_q1_delta_dice_mean": float(
            q5["delta_dice_mean"]
            - q1["delta_dice_mean"]
        ),
        "q5_minus_q1_harm_prevalence": float(
            q5["harm_prevalence"]
            - q1["harm_prevalence"]
        ),
        "q5_minus_q1_benefit_prevalence": float(
            q5["benefit_prevalence"]
            - q1["benefit_prevalence"]
        ),
    }

    return {
        "setting": setting,
        "macro_family_spearman_score_vs_negative_delta": macro_rho,
        "pooled_spearman_score_vs_negative_delta": pooled_rho,
    }, family_rows, qdf, contrast


def make_bootstrap_counts(clusters, seed):
    rng = np.random.default_rng(seed)
    n = len(clusters)

    counts = np.zeros(
        (BOOTSTRAPS, n),
        dtype=np.int16,
    )

    for b in range(BOOTSTRAPS):
        draw = rng.integers(
            0,
            n,
            size=n,
        )
        counts[b] = np.bincount(
            draw,
            minlength=n,
        ).astype(np.int16)

    return counts


def row_cluster_positions(df, cluster_order):
    pos = {
        cluster: i
        for i, cluster in enumerate(cluster_order)
    }
    return np.asarray(
        [
            pos[str(x)]
            for x in df["cluster_id"].astype(str)
        ],
        dtype=np.int64,
    )


def bootstrap_setting(
    setting,
    df,
    cluster_counts,
    row_cluster_pos,
):
    rows = []

    score = df["score"].to_numpy(float)
    delta = df["delta_dice"].to_numpy(float)
    harm = df["harm_label"].to_numpy(float)
    benefit = df["benefit_label"].to_numpy(float)
    quintile = df["risk_quintile"].to_numpy(int)

    for b in tqdm(
        range(BOOTSTRAPS),
        desc=f"R15C1 bootstrap {setting}",
        unit="rep",
        dynamic_ncols=True,
    ):
        cluster_w = cluster_counts[b].astype(np.int64)
        row_w = cluster_w[row_cluster_pos]

        expanded_idx = np.repeat(
            np.arange(len(df), dtype=np.int64),
            row_w,
        )

        if len(expanded_idx) == 0:
            raise RuntimeError(
                f"{setting}: zero bootstrap rows."
            )

        macro_rho, _ = family_macro_spearman(
            df,
            expanded_idx,
        )
        pooled_rho = spearman_rho(
            score[expanded_idx],
            -delta[expanded_idx],
        )

        row = {
            "setting": setting,
            "bootstrap": b,
            "macro_family_spearman_score_vs_negative_delta": macro_rho,
            "pooled_spearman_score_vs_negative_delta": pooled_rho,
        }

        for q in range(1, N_QUINTILES + 1):
            mask = quintile == q
            qw = row_w * mask.astype(np.int64)

            row[f"Q{q}_delta_dice_mean"] = weighted_mean(
                delta,
                qw,
            )
            row[f"Q{q}_harm_prevalence"] = weighted_mean(
                harm,
                qw,
            )
            row[f"Q{q}_benefit_prevalence"] = weighted_mean(
                benefit,
                qw,
            )

        row["Q5_minus_Q1_delta_dice_mean"] = (
            row["Q5_delta_dice_mean"]
            - row["Q1_delta_dice_mean"]
        )
        row["Q5_minus_Q1_harm_prevalence"] = (
            row["Q5_harm_prevalence"]
            - row["Q1_harm_prevalence"]
        )
        row["Q5_minus_Q1_benefit_prevalence"] = (
            row["Q5_benefit_prevalence"]
            - row["Q1_benefit_prevalence"]
        )

        rows.append(row)

    return pd.DataFrame(rows)


def summarize_bootstrap(boot):
    metrics = [
        "macro_family_spearman_score_vs_negative_delta",
        "pooled_spearman_score_vs_negative_delta",
        "Q5_minus_Q1_delta_dice_mean",
        "Q5_minus_Q1_harm_prevalence",
        "Q5_minus_Q1_benefit_prevalence",
    ]

    rows = []

    for setting, g in boot.groupby(
        "setting",
        sort=False,
    ):
        for metric in metrics:
            mean, lo, hi = percentile_ci(
                g[metric].to_numpy(float)
            )
            rows.append({
                "setting": setting,
                "metric": metric,
                "bootstrap_valid": int(
                    np.isfinite(
                        g[metric].to_numpy(float)
                    ).sum()
                ),
                "bootstrap_mean": mean,
                "ci95_low": lo,
                "ci95_high": hi,
            })

    return pd.DataFrame(rows)


def summarize_quintile_bootstrap(boot):
    rows = []

    for setting, g in boot.groupby(
        "setting",
        sort=False,
    ):
        for q in range(1, N_QUINTILES + 1):
            for suffix in (
                "delta_dice_mean",
                "harm_prevalence",
                "benefit_prevalence",
            ):
                col = f"Q{q}_{suffix}"
                mean, lo, hi = percentile_ci(
                    g[col].to_numpy(float)
                )
                rows.append({
                    "setting": setting,
                    "risk_quintile": q,
                    "metric": suffix,
                    "bootstrap_valid": int(
                        np.isfinite(
                            g[col].to_numpy(float)
                        ).sum()
                    ),
                    "bootstrap_mean": mean,
                    "ci95_low": lo,
                    "ci95_high": hi,
                })

    return pd.DataFrame(rows)


def claim_audit(summary):
    s = summary.set_index(["setting", "metric"])

    claims = {}

    for setting in SETTINGS:
        macro = s.loc[
            (
                setting,
                "macro_family_spearman_score_vs_negative_delta",
            )
        ]
        qdelta = s.loc[
            (
                setting,
                "Q5_minus_Q1_delta_dice_mean",
            )
        ]
        qharm = s.loc[
            (
                setting,
                "Q5_minus_Q1_harm_prevalence",
            )
        ]

        claims[setting] = {
            "continuous_macro_severity_relation_supported": (
                float(macro["ci95_low"]) > 0
            ),
            "highest_risk_quintile_has_lower_mean_delta_than_lowest": (
                float(qdelta["ci95_high"]) < 0
            ),
            "highest_risk_quintile_has_higher_harm_prevalence": (
                float(qharm["ci95_low"]) > 0
            ),
        }

    return claims


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    args = parser.parse_args()

    print(
        "===== Q1 R15C1 CONTINUOUS RISK vs DELTADICE "
        "EXTERNAL ANALYSIS ====="
    )
    print("STATUS=POST_FREEZE_THRESHOLD_FREE_EXTERNAL_ANALYSIS")
    print("MODEL_FITTING=NO")
    print("SCORE_RECOMPUTATION=NO")
    print("TARGET_CALIBRATION=NO")
    print("THRESHOLD_TUNING=NO")
    print("FEATURE_CHANGE=NO")
    print("CASE_EXCLUSION=NO")
    print("OUTCOME_THRESHOLD_USED_FOR_PRIMARY_SPEARMAN=NO")
    print(
        "PRIMARY=macro-family Spearman("
        "frozen_score, -DeltaDice)"
    )
    print("QUINTILES=5 equal-count score-only strata")
    print("BOOTSTRAPS=", BOOTSTRAPS)
    print(
        "PolypGen bootstrap seed=",
        POLYPGEN_BOOTSTRAP_SEED,
    )
    print(
        "SUN shared bootstrap seed=",
        SUN_SHARED_BOOTSTRAP_SEED,
    )

    b0, _ = verify_upstream()

    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    inputs = b0.get("inputs", {})

    data = {}
    paths = {}
    family_sets = {}

    for setting in SETTINGS:
        if setting not in inputs:
            raise RuntimeError(
                f"R15B0 missing setting: {setting}"
            )

        df, families, path = load_setting(
            setting,
            inputs[setting],
        )
        data[setting] = df
        family_sets[setting] = families
        paths[setting] = path

    point_rows = []
    family_rows = []
    quintile_frames = []
    contrast_rows = []

    for setting in SETTINGS:
        point, fam, qdf, contrast = point_analysis(
            setting,
            data[setting],
        )
        point_rows.append(point)

        for row in fam:
            family_rows.append({
                "setting": setting,
                **row,
            })

        quintile_frames.append(qdf)
        contrast_rows.append(contrast)

    point_df = pd.DataFrame(point_rows)
    family_df = pd.DataFrame(family_rows)
    quintile_df = pd.concat(
        quintile_frames,
        ignore_index=True,
    )
    contrast_df = pd.DataFrame(contrast_rows)

    print("\n===== POINT CONTINUOUS RELATION =====")
    print(point_df.to_string(index=False))

    print("\n===== SCORE-ONLY RISK QUINTILES =====")
    print(
        quintile_df[
            [
                "setting",
                "risk_quintile",
                "score_mean",
                "delta_dice_mean",
                "delta_dice_median",
                "harm_prevalence",
                "benefit_prevalence",
            ]
        ].to_string(index=False)
    )

    print("\n===== Q5 MINUS Q1 CONTRAST =====")
    print(contrast_df.to_string(index=False))

    # Bootstrap cluster sequences.
    pg_clusters = sorted(
        data["PolypGen_TENT1"]["cluster_id"]
        .astype(str)
        .unique()
    )
    sun_tent_clusters = sorted(
        data["SUN_TENT1"]["cluster_id"]
        .astype(str)
        .unique()
    )
    sun_pl_clusters = sorted(
        data["SUN_PL_CONF90"]["cluster_id"]
        .astype(str)
        .unique()
    )

    if sun_tent_clusters != sun_pl_clusters:
        raise RuntimeError(
            "SUN TENT1/PL cluster sets differ."
        )

    pg_counts = make_bootstrap_counts(
        pg_clusters,
        POLYPGEN_BOOTSTRAP_SEED,
    )
    sun_counts = make_bootstrap_counts(
        sun_tent_clusters,
        SUN_SHARED_BOOTSTRAP_SEED,
    )

    pg_row_pos = row_cluster_positions(
        data["PolypGen_TENT1"],
        pg_clusters,
    )
    sun_tent_row_pos = row_cluster_positions(
        data["SUN_TENT1"],
        sun_tent_clusters,
    )
    sun_pl_row_pos = row_cluster_positions(
        data["SUN_PL_CONF90"],
        sun_pl_clusters,
    )

    boot_frames = []

    boot_frames.append(
        bootstrap_setting(
            "PolypGen_TENT1",
            data["PolypGen_TENT1"],
            pg_counts,
            pg_row_pos,
        )
    )
    boot_frames.append(
        bootstrap_setting(
            "SUN_TENT1",
            data["SUN_TENT1"],
            sun_counts,
            sun_tent_row_pos,
        )
    )
    boot_frames.append(
        bootstrap_setting(
            "SUN_PL_CONF90",
            data["SUN_PL_CONF90"],
            sun_counts,
            sun_pl_row_pos,
        )
    )

    boot = pd.concat(
        boot_frames,
        ignore_index=True,
    )

    summary = summarize_bootstrap(boot)
    quintile_ci = summarize_quintile_bootstrap(
        boot
    )
    claims = claim_audit(summary)

    print("\n===== PHYSICAL-CASE CLUSTERED BOOTSTRAP =====")
    print(
        summary[
            [
                "setting",
                "metric",
                "bootstrap_mean",
                "ci95_low",
                "ci95_high",
            ]
        ].to_string(index=False)
    )

    print("\n===== CLAIM AUDIT =====")
    for setting, row in claims.items():
        print(setting)
        for key, value in row.items():
            print(" ", key, "=", value)

    point_path = (
        args.output_dir
        / "R15C1_continuous_risk_point_metrics.csv"
    )
    family_path = (
        args.output_dir
        / "R15C1_family_spearman_point_metrics.csv"
    )
    quintile_path = (
        args.output_dir
        / "R15C1_score_quintile_point_metrics.csv"
    )
    contrast_path = (
        args.output_dir
        / "R15C1_Q5_minus_Q1_point_contrast.csv"
    )
    summary_path = (
        args.output_dir
        / "R15C1_continuous_risk_cluster_bootstrap_ci.csv"
    )
    quintile_ci_path = (
        args.output_dir
        / "R15C1_score_quintile_cluster_bootstrap_ci.csv"
    )
    claim_path = (
        args.output_dir
        / "R15C1_CONTINUOUS_RISK_CLAIM_AUDIT.json"
    )

    point_df.to_csv(
        point_path,
        index=False,
    )
    family_df.to_csv(
        family_path,
        index=False,
    )
    quintile_df.to_csv(
        quintile_path,
        index=False,
    )
    contrast_df.to_csv(
        contrast_path,
        index=False,
    )
    summary.to_csv(
        summary_path,
        index=False,
    )
    quintile_ci.to_csv(
        quintile_ci_path,
        index=False,
    )
    write_json(
        claim_path,
        claims,
    )

    artifacts = {}
    for p in (
        point_path,
        family_path,
        quintile_path,
        contrast_path,
        summary_path,
        quintile_ci_path,
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
            "post-freeze threshold-free continuous external "
            "risk-severity analysis"
        ),
        "primary_endpoint": (
            "macro-family Spearman("
            "frozen safety probability, -DeltaDice)"
        ),
        "secondary_endpoints": [
            "pooled Spearman(frozen score, -DeltaDice)",
            "score-only equal-count risk quintiles",
            "Q5 minus Q1 mean DeltaDice",
            "Q5 minus Q1 HARM prevalence",
        ],
        "risk_quintiles": {
            "count": N_QUINTILES,
            "defined_from": "frozen safety score only",
            "outcome_labels_used_to_define_quintiles": False,
            "membership_frozen_before_outcome_summary": True,
        },
        "bootstrap": {
            "replicates": BOOTSTRAPS,
            "PolypGen_cluster": "sample_id",
            "PolypGen_seed": POLYPGEN_BOOTSTRAP_SEED,
            "SUN_cluster": "cluster_id",
            "SUN_shared_seed": SUN_SHARED_BOOTSTRAP_SEED,
            "same_SUN_cluster_draw_for_TENT1_and_PL": True,
        },
        "claim_audit": claims,
        "information_boundary": {
            "model_fitting": False,
            "score_recomputation": False,
            "target_calibration": False,
            "threshold_tuning": False,
            "feature_change": False,
            "case_exclusion": False,
            "outcome_threshold_used_for_primary_spearman": False,
            "method_reselection": False,
        },
        "upstream": {
            "r15b0_lock_sha256": EXPECTED_R15B0_LOCK_SHA,
            "r15b1_lock_sha256": EXPECTED_R15B1_LOCK_SHA,
            "panels": {
                setting: {
                    "path": str(paths[setting]),
                    "sha256": sha256_file(
                        paths[setting]
                    ),
                }
                for setting in SETTINGS
            },
        },
        "artifacts": artifacts,
        "next_stage": (
            "R15D_RUNTIME_AND_COMPUTATIONAL_OVERHEAD_AUDIT"
        ),
    }

    lock_path = (
        args.output_dir
        / "R15C1_CONTINUOUS_RISK_EXTERNAL_ANALYSIS_LOCK.json"
    )
    write_json(
        lock_path,
        lock,
    )

    print("\n===== R15C1 FINAL =====")
    print("Decision=", DECISION)
    print("TARGET_RETUNING=NO")
    print("LOCK=", lock_path)
    print("LOCK SHA256=", sha256_file(lock_path))
    print("PASS")


if __name__ == "__main__":
    main()
