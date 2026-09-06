#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R15B1_selective_adaptation_risk_coverage_utility_fix1.py

Post-freeze external selective-adaptation / risk-coverage utility analysis.

Inputs are locked by R15B0:
  1) PolypGen / TENT1
  2) SUN-SEG / TENT1
  3) SUN-SEG / PL_CONF90

Policy is label-free and fixed before this analysis:
- higher frozen safety probability = higher predicted HARM risk;
- for each pre-specified coverage c, adapt the lowest-risk c fraction of
  model-frame rows and retain SOURCE for the highest-risk 1-c fraction;
- separately evaluate the already-frozen deployment threshold:
      adapt iff frozen_safety_probability < 0.300584763193734

No score fitting/recomputation, target calibration, threshold tuning,
coverage selection by GT, feature change, case exclusion, or model change.

The random-selection reference is the exact expectation at the same coverage
under uniform row-level selection, so no extra Monte-Carlo policy randomness is
introduced.

Bootstrap:
- PolypGen: sample_id cluster
- SUN: physical case/video cluster_id
- 2000 replicates
- same sampled-cluster multiplicities across every coverage within a setting
- SUN TENT1 and SUN PL reuse the same SUN cluster-bootstrap draws
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


VERSION = "2026-09-04-Q1-R15B1-v1-fix1"
BUILD = "Q1_R15B1_SELECTIVE_ADAPTATION_RISK_COVERAGE_UTILITY_FIX1"

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
EXPECTED_R15B0_DECISION = (
    "EXTERNAL_SELECTIVE_ADAPTATION_UTILITY_INPUTS_LOCKED_READY_FOR_R15B1"
)

DEFAULT_OUTPUT = (
    OUT
    / "Q1_R15B1_selective_adaptation_risk_coverage_utility_fix1_v1"
)

FROZEN_THRESHOLD = 0.300584763193734
COVERAGE_GRID = [
    1.00,
    0.90,
    0.80,
    0.70,
    0.60,
    0.50,
    0.40,
    0.30,
    0.20,
    0.10,
    0.00,
]

BOOTSTRAPS = 2000
POLYPGEN_BOOTSTRAP_SEED = 20260904
SUN_SHARED_BOOTSTRAP_SEED = 20260905

SETTING_ORDER = [
    "PolypGen_TENT1",
    "SUN_TENT1",
    "SUN_PL_CONF90",
]

UTILITY_METRICS = [
    "harm_rate_among_adapted",
    "benefit_rate_among_adapted",
    "prevented_harm_fraction",
    "withheld_benefit_fraction",
    "mean_deployed_dice",
    "median_deployed_dice",
    "mean_delta_vs_source",
    "mean_delta_vs_adapt_all",
]

RANDOM_COMPARABLE_METRICS = [
    "harm_rate_among_adapted",
    "benefit_rate_among_adapted",
    "prevented_harm_fraction",
    "withheld_benefit_fraction",
    "mean_deployed_dice",
    "mean_delta_vs_source",
    "mean_delta_vs_adapt_all",
]

DECISION = (
    "EXTERNAL_SELECTIVE_ADAPTATION_UTILITY_COMPLETE_"
    "NO_TARGET_POLICY_RETUNING"
)


def sha256_file(path: Path, chunk=8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
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
    if x.size == 0:
        return math.nan, math.nan, math.nan
    return (
        float(np.mean(x)),
        float(np.quantile(x, 0.025)),
        float(np.quantile(x, 0.975)),
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
    idx = int(np.searchsorted(np.cumsum(weights), cutoff, side="left"))
    idx = min(idx, len(values) - 1)
    return float(values[idx])


def coverage_count(n, coverage):
    """
    Deterministic nearest-integer row coverage.
    Avoid Python bankers-rounding.
    """
    if coverage <= 0:
        return 0
    if coverage >= 1:
        return int(n)
    return int(np.floor(float(coverage) * int(n) + 0.5))


def verify_r15b0():
    print("===== R15B0 FROZEN INPUT GATE =====")

    if not R15B0_LOCK.is_file():
        raise FileNotFoundError(R15B0_LOCK)

    got = sha256_file(R15B0_LOCK)
    if got != EXPECTED_R15B0_LOCK_SHA:
        raise RuntimeError(
            f"R15B0 lock SHA changed: {got}"
        )

    lock = load_json(R15B0_LOCK)
    if lock.get("status") != "PASS":
        raise RuntimeError("R15B0 status changed.")
    if lock.get("decision") != EXPECTED_R15B0_DECISION:
        raise RuntimeError("R15B0 decision changed.")

    if float(lock.get("frozen_threshold", math.nan)) != FROZEN_THRESHOLD:
        raise RuntimeError("R15B0 frozen threshold changed.")

    if list(lock.get("coverage_grid", [])) != COVERAGE_GRID:
        raise RuntimeError("R15B0 coverage grid changed.")

    policy = lock.get("policy", {})
    if bool(policy.get("coverage_selected_by_target_labels", True)):
        raise RuntimeError("R15B0 policy reports GT-selected coverage.")
    if bool(policy.get("threshold_changed", True)):
        raise RuntimeError("R15B0 threshold changed.")
    if bool(policy.get("score_changed", True)):
        raise RuntimeError("R15B0 score changed.")
    if bool(policy.get("model_changed", True)):
        raise RuntimeError("R15B0 model changed.")

    print("R15B0 lock SHA:", got)
    print("coverage grid:", COVERAGE_GRID)
    print("frozen threshold:", FROZEN_THRESHOLD)
    print("PASS")

    return lock


def resolve_locked_input(meta):
    path = Path(meta["panel_path"])
    if not path.is_file():
        raise FileNotFoundError(path)

    actual_sha = sha256_file(path)
    expected_sha = str(meta["panel_sha256"])
    if actual_sha.lower() != expected_sha.lower():
        raise RuntimeError(
            f"Frozen panel SHA mismatch: {path}"
        )

    return path


def load_setting(setting_name, meta):
    panel_path = resolve_locked_input(meta)
    df = pd.read_csv(panel_path, low_memory=False)

    source_col = str(meta["source_dice_column"])
    adapted_col = str(meta["adapted_dice_column"])
    delta_col = str(meta["delta_column"])
    harm_col = str(meta["harm_column"])
    benefit_col = str(meta["benefit_column"])
    score_col = str(meta["score_column"])
    cluster_col = str(meta["cluster_unit"])

    required = {
        "sample_id",
        "model_family",
        "model_state_id",
        source_col,
        adapted_col,
        delta_col,
        harm_col,
        benefit_col,
        score_col,
        "frozen_operating_threshold",
    }
    if cluster_col not in required:
        required.add(cluster_col)

    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(
            f"{setting_name}: panel missing columns {missing}"
        )

    out = pd.DataFrame({
        "sample_id": df["sample_id"].astype(str),
        "model_family": df["model_family"].astype(str),
        "model_state_id": df["model_state_id"].astype(str),
        "cluster_id": df[cluster_col].astype(str),
        "source_dice": pd.to_numeric(
            df[source_col], errors="raise"
        ).astype(float),
        "adapted_dice": pd.to_numeric(
            df[adapted_col], errors="raise"
        ).astype(float),
        "delta_dice": pd.to_numeric(
            df[delta_col], errors="raise"
        ).astype(float),
        "harm_label": pd.to_numeric(
            df[harm_col], errors="raise"
        ).astype(int),
        "benefit_label": pd.to_numeric(
            df[benefit_col], errors="raise"
        ).astype(int),
        "score": pd.to_numeric(
            df[score_col], errors="raise"
        ).astype(float),
        "frozen_operating_threshold": pd.to_numeric(
            df["frozen_operating_threshold"],
            errors="raise",
        ).astype(float),
    })

    if not np.isfinite(
        out[
            [
                "source_dice",
                "adapted_dice",
                "delta_dice",
                "score",
                "frozen_operating_threshold",
            ]
        ].to_numpy(float)
    ).all():
        raise RuntimeError(
            f"{setting_name}: non-finite numeric values."
        )

    if not np.allclose(
        out["adapted_dice"].to_numpy(float)
        - out["source_dice"].to_numpy(float),
        out["delta_dice"].to_numpy(float),
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError(
            f"{setting_name}: delta_dice does not reproduce adapted-source."
        )

    if not set(out["harm_label"].unique()).issubset({0, 1}):
        raise RuntimeError(
            f"{setting_name}: harm label not binary."
        )
    if not set(out["benefit_label"].unique()).issubset({0, 1}):
        raise RuntimeError(
            f"{setting_name}: benefit label not binary."
        )

    if not np.allclose(
        out["frozen_operating_threshold"].to_numpy(float),
        FROZEN_THRESHOLD,
        rtol=0.0,
        atol=1e-15,
    ):
        raise RuntimeError(
            f"{setting_name}: frozen threshold changed."
        )

    # Deterministic tie-breaking for risk ranking.
    out["_rank_order"] = np.arange(len(out), dtype=int)
    out = out.sort_values(
        [
            "score",
            "model_family",
            "model_state_id",
            "sample_id",
            "_rank_order",
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    if out[["sample_id", "model_state_id"]].drop_duplicates().shape[0] != len(out):
        raise RuntimeError(
            f"{setting_name}: model-frame/model-case key not unique."
        )

    print("\n=====", setting_name, "PANEL =====")
    print("path:", panel_path)
    print("SHA256:", sha256_file(panel_path))
    print("rows:", len(out))
    print("clusters:", out["cluster_id"].nunique())
    print("states:", out["model_state_id"].nunique())
    print("HARM prevalence:", float(out["harm_label"].mean()))
    print("BENEFIT prevalence:", float(out["benefit_label"].mean()))
    print(
        "SOURCE Dice mean/median:",
        float(out["source_dice"].mean()),
        float(out["source_dice"].median()),
    )
    print(
        "adapt-all Dice mean/median:",
        float(out["adapted_dice"].mean()),
        float(out["adapted_dice"].median()),
    )
    print(
        "score min/median/max:",
        float(out["score"].min()),
        float(out["score"].median()),
        float(out["score"].max()),
    )
    print("PASS")

    return out, panel_path


def selected_weights_by_coverage(row_weights, coverage):
    """
    Rows are already sorted from lowest to highest risk.

    A bootstrap cluster may appear multiple times, so each original row has an
    integer multiplicity. If the coverage boundary cuts through duplicated
    copies of one row, only the required number of identical copies is adapted.
    """
    row_weights = np.asarray(row_weights, dtype=np.int64)
    total = int(row_weights.sum())
    k = coverage_count(total, coverage)

    selected = np.zeros_like(row_weights)
    remaining = k

    if remaining <= 0:
        return selected, k, total

    for i, w in enumerate(row_weights):
        if remaining <= 0:
            break
        if w <= 0:
            continue
        take = min(int(w), remaining)
        selected[i] = take
        remaining -= take

    if remaining != 0:
        raise RuntimeError(
            "Coverage selection did not consume expected adapted rows."
        )

    return selected, k, total


def random_expected_metrics(
    row_weights,
    source,
    adapted,
    harm,
    benefit,
    selected_count,
):
    """
    Exact expectation under uniform row-level selection at the same coverage.
    No Monte Carlo random policy is needed.
    """
    w = np.asarray(row_weights, dtype=float)
    total = float(w.sum())
    if total <= 0:
        raise RuntimeError("Zero bootstrap weight.")

    q = float(selected_count) / total

    source_mean = float(np.sum(w * source) / total)
    adapted_mean = float(np.sum(w * adapted) / total)
    delta_mean = adapted_mean - source_mean

    total_harm = float(np.sum(w * harm))
    total_benefit = float(np.sum(w * benefit))

    harm_prev = total_harm / total if total_harm > 0 else 0.0
    benefit_prev = total_benefit / total if total_benefit > 0 else 0.0

    mean_deployed = source_mean + q * delta_mean

    return {
        "adaptation_coverage": q,
        "harm_rate_among_adapted": (
            harm_prev if selected_count > 0 else math.nan
        ),
        "benefit_rate_among_adapted": (
            benefit_prev if selected_count > 0 else math.nan
        ),
        "prevented_harm_fraction": (
            1.0 - q if total_harm > 0 else math.nan
        ),
        "withheld_benefit_fraction": (
            1.0 - q if total_benefit > 0 else math.nan
        ),
        "mean_deployed_dice": mean_deployed,
        "median_deployed_dice": math.nan,
        "mean_delta_vs_source": q * delta_mean,
        "mean_delta_vs_adapt_all": (
            mean_deployed - adapted_mean
        ),
    }


def policy_metrics(
    df,
    row_weights,
    selected_weights,
):
    w = np.asarray(row_weights, dtype=float)
    sel = np.asarray(selected_weights, dtype=float)

    if w.shape != sel.shape:
        raise RuntimeError("Weight shape mismatch.")
    if np.any(sel < 0) or np.any(sel > w):
        raise RuntimeError("Invalid selected multiplicities.")

    total = float(w.sum())
    adapted_n = float(sel.sum())

    source = df["source_dice"].to_numpy(float)
    adapted = df["adapted_dice"].to_numpy(float)
    harm = df["harm_label"].to_numpy(float)
    benefit = df["benefit_label"].to_numpy(float)

    withheld = w - sel

    source_mean = float(np.sum(w * source) / total)
    adapt_all_mean = float(np.sum(w * adapted) / total)

    deployed_sum = float(
        np.sum(sel * adapted)
        + np.sum(withheld * source)
    )
    mean_deployed = deployed_sum / total

    # Exact weighted median of the deployed population.
    deployed_values = np.concatenate([adapted, source])
    deployed_weights = np.concatenate([sel, withheld])
    median_deployed = weighted_median(
        deployed_values,
        deployed_weights,
    )

    total_harm = float(np.sum(w * harm))
    adapted_harm = float(np.sum(sel * harm))
    withheld_harm = total_harm - adapted_harm

    total_benefit = float(np.sum(w * benefit))
    adapted_benefit = float(np.sum(sel * benefit))
    withheld_benefit = total_benefit - adapted_benefit

    return {
        "adaptation_coverage": adapted_n / total,
        "adapted_rows": int(adapted_n),
        "total_rows": int(total),
        "harm_rate_among_adapted": (
            adapted_harm / adapted_n
            if adapted_n > 0
            else math.nan
        ),
        "benefit_rate_among_adapted": (
            adapted_benefit / adapted_n
            if adapted_n > 0
            else math.nan
        ),
        "prevented_harm_fraction": (
            withheld_harm / total_harm
            if total_harm > 0
            else math.nan
        ),
        "withheld_benefit_fraction": (
            withheld_benefit / total_benefit
            if total_benefit > 0
            else math.nan
        ),
        "mean_deployed_dice": mean_deployed,
        "median_deployed_dice": median_deployed,
        "mean_delta_vs_source": (
            mean_deployed - source_mean
        ),
        "mean_delta_vs_adapt_all": (
            mean_deployed - adapt_all_mean
        ),
        "source_mean_dice": source_mean,
        "adapt_all_mean_dice": adapt_all_mean,
        "total_harm_rows": int(total_harm),
        "total_benefit_rows": int(total_benefit),
        "adapted_harm_rows": int(adapted_harm),
        "adapted_benefit_rows": int(adapted_benefit),
    }


def fixed_threshold_selected_weights(df, row_weights):
    w = np.asarray(row_weights, dtype=np.int64)
    eligible = (
        df["score"].to_numpy(float)
        < FROZEN_THRESHOLD
    ).astype(np.int64)
    return w * eligible


def make_cluster_row_map(df):
    clusters = sorted(df["cluster_id"].astype(str).unique())
    index = {}

    cluster_to_pos = {
        cluster: i
        for i, cluster in enumerate(clusters)
    }

    row_cluster_pos = np.asarray([
        cluster_to_pos[str(x)]
        for x in df["cluster_id"].astype(str)
    ], dtype=np.int64)

    for cluster in clusters:
        index[cluster] = np.flatnonzero(
            df["cluster_id"].astype(str).to_numpy()
            == cluster
        )

    return clusters, row_cluster_pos, index


def point_curve(setting_name, df):
    w = np.ones(len(df), dtype=np.int64)

    rows = []
    delta_rows = []

    source = df["source_dice"].to_numpy(float)
    adapted = df["adapted_dice"].to_numpy(float)
    harm = df["harm_label"].to_numpy(float)
    benefit = df["benefit_label"].to_numpy(float)

    for coverage in COVERAGE_GRID:
        sel, k, total = selected_weights_by_coverage(
            w,
            coverage,
        )
        selective = policy_metrics(df, w, sel)
        random_ref = random_expected_metrics(
            w,
            source,
            adapted,
            harm,
            benefit,
            k,
        )

        row = {
            "setting": setting_name,
            "policy": "RISK_RANKED",
            "requested_coverage": coverage,
            **selective,
        }
        rows.append(row)

        for metric in RANDOM_COMPARABLE_METRICS:
            sval = selective[metric]
            rval = random_ref[metric]
            delta_rows.append({
                "setting": setting_name,
                "policy": "RISK_RANKED_MINUS_RANDOM_EXPECTATION",
                "requested_coverage": coverage,
                "metric": metric,
                "selective_value": sval,
                "random_expected_value": rval,
                "delta": (
                    float(sval - rval)
                    if np.isfinite(sval) and np.isfinite(rval)
                    else math.nan
                ),
            })

    fixed_sel = fixed_threshold_selected_weights(df, w)
    fixed = policy_metrics(
        df,
        w,
        fixed_sel,
    )
    fixed_random = random_expected_metrics(
        w,
        source,
        adapted,
        harm,
        benefit,
        int(fixed_sel.sum()),
    )

    fixed_row = {
        "setting": setting_name,
        "policy": "FROZEN_THRESHOLD",
        "requested_coverage": math.nan,
        "threshold": FROZEN_THRESHOLD,
        **fixed,
    }

    fixed_delta_rows = []
    for metric in RANDOM_COMPARABLE_METRICS:
        sval = fixed[metric]
        rval = fixed_random[metric]
        fixed_delta_rows.append({
            "setting": setting_name,
            "policy": "FROZEN_THRESHOLD_MINUS_RANDOM_EXPECTATION",
            "requested_coverage": math.nan,
            "threshold": FROZEN_THRESHOLD,
            "metric": metric,
            "selective_value": sval,
            "random_expected_value": rval,
            "delta": (
                float(sval - rval)
                if np.isfinite(sval) and np.isfinite(rval)
                else math.nan
            ),
        })

    return (
        pd.DataFrame(rows),
        pd.DataFrame(delta_rows),
        pd.DataFrame([fixed_row]),
        pd.DataFrame(fixed_delta_rows),
    )


def generate_bootstrap_cluster_counts(
    clusters,
    n_bootstrap,
    seed,
):
    """
    Return multiplicity matrix [replicate, n_clusters].
    """
    n_clusters = len(clusters)
    rng = np.random.default_rng(seed)

    counts = np.zeros(
        (n_bootstrap, n_clusters),
        dtype=np.int16,
    )

    for b in range(n_bootstrap):
        draw = rng.integers(
            0,
            n_clusters,
            size=n_clusters,
        )
        counts[b] = np.bincount(
            draw,
            minlength=n_clusters,
        ).astype(np.int16)

    return counts


def bootstrap_setting(
    setting_name,
    df,
    cluster_counts,
    row_cluster_pos,
):
    """
    Compute all pre-specified coverages and fixed-threshold policy using the
    same cluster multiplicity draw for every coverage.
    """
    source = df["source_dice"].to_numpy(float)
    adapted = df["adapted_dice"].to_numpy(float)
    harm = df["harm_label"].to_numpy(float)
    benefit = df["benefit_label"].to_numpy(float)

    metric_rows = []
    delta_rows = []
    fixed_metric_rows = []
    fixed_delta_rows = []

    for b in tqdm(
        range(cluster_counts.shape[0]),
        desc=f"R15B1 bootstrap {setting_name}",
        unit="rep",
        dynamic_ncols=True,
    ):
        cluster_w = cluster_counts[b].astype(np.int64)
        row_w = cluster_w[row_cluster_pos]

        if int(row_w.sum()) <= 0:
            raise RuntimeError(
                f"{setting_name}: zero bootstrap population."
            )

        for coverage in COVERAGE_GRID:
            sel, k, total = selected_weights_by_coverage(
                row_w,
                coverage,
            )
            selective = policy_metrics(
                df,
                row_w,
                sel,
            )
            random_ref = random_expected_metrics(
                row_w,
                source,
                adapted,
                harm,
                benefit,
                k,
            )

            for metric in UTILITY_METRICS:
                metric_rows.append({
                    "setting": setting_name,
                    "bootstrap": b,
                    "policy": "RISK_RANKED",
                    "requested_coverage": coverage,
                    "metric": metric,
                    "value": selective[metric],
                })

            for metric in RANDOM_COMPARABLE_METRICS:
                sval = selective[metric]
                rval = random_ref[metric]
                delta_rows.append({
                    "setting": setting_name,
                    "bootstrap": b,
                    "policy": "RISK_RANKED_MINUS_RANDOM_EXPECTATION",
                    "requested_coverage": coverage,
                    "metric": metric,
                    "delta": (
                        float(sval - rval)
                        if np.isfinite(sval) and np.isfinite(rval)
                        else math.nan
                    ),
                })

        fixed_sel = fixed_threshold_selected_weights(
            df,
            row_w,
        )
        fixed = policy_metrics(
            df,
            row_w,
            fixed_sel,
        )
        fixed_random = random_expected_metrics(
            row_w,
            source,
            adapted,
            harm,
            benefit,
            int(fixed_sel.sum()),
        )

        for metric in UTILITY_METRICS:
            fixed_metric_rows.append({
                "setting": setting_name,
                "bootstrap": b,
                "policy": "FROZEN_THRESHOLD",
                "threshold": FROZEN_THRESHOLD,
                "metric": metric,
                "value": fixed[metric],
            })

        for metric in RANDOM_COMPARABLE_METRICS:
            sval = fixed[metric]
            rval = fixed_random[metric]
            fixed_delta_rows.append({
                "setting": setting_name,
                "bootstrap": b,
                "policy": "FROZEN_THRESHOLD_MINUS_RANDOM_EXPECTATION",
                "threshold": FROZEN_THRESHOLD,
                "metric": metric,
                "delta": (
                    float(sval - rval)
                    if np.isfinite(sval) and np.isfinite(rval)
                    else math.nan
                ),
            })

    return (
        pd.DataFrame(metric_rows),
        pd.DataFrame(delta_rows),
        pd.DataFrame(fixed_metric_rows),
        pd.DataFrame(fixed_delta_rows),
    )


def summarize_curve_bootstrap(boot_df):
    rows = []

    for (
        setting,
        coverage,
        metric,
    ), g in boot_df.groupby(
        ["setting", "requested_coverage", "metric"],
        sort=False,
    ):
        mean, lo, hi = percentile_ci(
            g["value"].to_numpy(float)
        )
        rows.append({
            "setting": setting,
            "policy": "RISK_RANKED",
            "requested_coverage": float(coverage),
            "metric": metric,
            "bootstrap_valid": int(
                np.isfinite(
                    g["value"].to_numpy(float)
                ).sum()
            ),
            "bootstrap_mean": mean,
            "ci95_low": lo,
            "ci95_high": hi,
        })

    return pd.DataFrame(rows)


def summarize_delta_bootstrap(delta_df):
    rows = []

    for (
        setting,
        coverage,
        metric,
    ), g in delta_df.groupby(
        ["setting", "requested_coverage", "metric"],
        sort=False,
    ):
        mean, lo, hi = percentile_ci(
            g["delta"].to_numpy(float)
        )

        # Favorable direction:
        # lower adapted harm rate and withheld benefit are not both "better";
        # interpret raw delta, do not collapse all metrics to a GO flag.
        rows.append({
            "setting": setting,
            "policy": "RISK_RANKED_MINUS_RANDOM_EXPECTATION",
            "requested_coverage": float(coverage),
            "metric": metric,
            "bootstrap_valid": int(
                np.isfinite(
                    g["delta"].to_numpy(float)
                ).sum()
            ),
            "bootstrap_mean_delta": mean,
            "ci95_low": lo,
            "ci95_high": hi,
        })

    return pd.DataFrame(rows)


def summarize_fixed_bootstrap(fixed_df):
    rows = []

    for (setting, metric), g in fixed_df.groupby(
        ["setting", "metric"],
        sort=False,
    ):
        mean, lo, hi = percentile_ci(
            g["value"].to_numpy(float)
        )
        rows.append({
            "setting": setting,
            "policy": "FROZEN_THRESHOLD",
            "threshold": FROZEN_THRESHOLD,
            "metric": metric,
            "bootstrap_valid": int(
                np.isfinite(
                    g["value"].to_numpy(float)
                ).sum()
            ),
            "bootstrap_mean": mean,
            "ci95_low": lo,
            "ci95_high": hi,
        })

    return pd.DataFrame(rows)


def summarize_fixed_delta_bootstrap(delta_df):
    rows = []

    for (setting, metric), g in delta_df.groupby(
        ["setting", "metric"],
        sort=False,
    ):
        mean, lo, hi = percentile_ci(
            g["delta"].to_numpy(float)
        )
        rows.append({
            "setting": setting,
            "policy": "FROZEN_THRESHOLD_MINUS_RANDOM_EXPECTATION",
            "threshold": FROZEN_THRESHOLD,
            "metric": metric,
            "bootstrap_valid": int(
                np.isfinite(
                    g["delta"].to_numpy(float)
                ).sum()
            ),
            "bootstrap_mean_delta": mean,
            "ci95_low": lo,
            "ci95_high": hi,
        })

    return pd.DataFrame(rows)


def print_key_point_results(
    point_curve_df,
    point_delta_df,
    fixed_point_df,
    fixed_delta_df,
):
    print("\n===== R15B1 POINT UTILITY CURVES =====")

    for setting in SETTING_ORDER:
        print("\n---", setting, "---")
        sub = point_curve_df[
            point_curve_df["setting"] == setting
        ]

        display = sub[
            [
                "requested_coverage",
                "adaptation_coverage",
                "harm_rate_among_adapted",
                "prevented_harm_fraction",
                "withheld_benefit_fraction",
                "mean_deployed_dice",
                "mean_delta_vs_source",
                "mean_delta_vs_adapt_all",
            ]
        ]
        print(display.to_string(index=False))

        print("\nSelective - random expectation:")
        d = point_delta_df[
            (point_delta_df["setting"] == setting)
            & (
                point_delta_df["metric"].isin(
                    [
                        "harm_rate_among_adapted",
                        "prevented_harm_fraction",
                        "mean_deployed_dice",
                    ]
                )
            )
        ]
        print(
            d[
                [
                    "requested_coverage",
                    "metric",
                    "delta",
                ]
            ].to_string(index=False)
        )

        print("\nFrozen threshold point:")
        f = fixed_point_df[
            fixed_point_df["setting"] == setting
        ]
        print(
            f[
                [
                    "adaptation_coverage",
                    "harm_rate_among_adapted",
                    "prevented_harm_fraction",
                    "withheld_benefit_fraction",
                    "mean_deployed_dice",
                    "mean_delta_vs_source",
                    "mean_delta_vs_adapt_all",
                ]
            ].to_string(index=False)
        )

        print("\nFrozen threshold - random expectation:")
        fd = fixed_delta_df[
            (fixed_delta_df["setting"] == setting)
            & (
                fixed_delta_df["metric"].isin(
                    [
                        "harm_rate_among_adapted",
                        "prevented_harm_fraction",
                        "mean_deployed_dice",
                    ]
                )
            )
        ]
        print(
            fd[
                [
                    "metric",
                    "delta",
                ]
            ].to_string(index=False)
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    args = parser.parse_args()

    print(
        "===== Q1 R15B1 SELECTIVE-ADAPTATION "
        "RISK-COVERAGE UTILITY ====="
    )
    print("STATUS=POST_FREEZE_EXTERNAL_UTILITY_ANALYSIS")
    print("MODEL_FITTING=NO")
    print("PCA_FITTING=NO")
    print("SCORE_RECOMPUTATION=NO")
    print("TARGET_CALIBRATION=NO")
    print("THRESHOLD_TUNING=NO")
    print("COVERAGE_SELECTION_BY_GT=NO")
    print("FEATURE_CHANGE=NO")
    print("CASE_EXCLUSION=NO")
    print("FROZEN_THRESHOLD=", FROZEN_THRESHOLD)
    print("COVERAGE_GRID=", COVERAGE_GRID)
    print("BOOTSTRAPS=", BOOTSTRAPS)
    print(
        "RANDOM_REFERENCE="
        "EXACT_UNIFORM_ROW_SELECTION_EXPECTATION"
    )

    r15b0 = verify_r15b0()

    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    inputs = r15b0.get("inputs", {})

    settings = {}
    panel_paths = {}

    for setting_name in SETTING_ORDER:
        if setting_name not in inputs:
            raise RuntimeError(
                f"R15B0 missing setting: {setting_name}"
            )
        settings[setting_name], panel_paths[setting_name] = (
            load_setting(
                setting_name,
                inputs[setting_name],
            )
        )

    # Point curves.
    point_curves = []
    point_deltas = []
    fixed_points = []
    fixed_deltas = []

    for setting_name in SETTING_ORDER:
        (
            p,
            d,
            f,
            fd,
        ) = point_curve(
            setting_name,
            settings[setting_name],
        )
        point_curves.append(p)
        point_deltas.append(d)
        fixed_points.append(f)
        fixed_deltas.append(fd)

    point_curve_df = pd.concat(
        point_curves,
        ignore_index=True,
    )
    point_delta_df = pd.concat(
        point_deltas,
        ignore_index=True,
    )
    fixed_point_df = pd.concat(
        fixed_points,
        ignore_index=True,
    )
    fixed_delta_df = pd.concat(
        fixed_deltas,
        ignore_index=True,
    )

    print_key_point_results(
        point_curve_df,
        point_delta_df,
        fixed_point_df,
        fixed_delta_df,
    )

    # Build cluster maps.
    pg_clusters, pg_row_pos, _ = make_cluster_row_map(
        settings["PolypGen_TENT1"]
    )
    sun_clusters_t1, sun_row_pos_t1, _ = make_cluster_row_map(
        settings["SUN_TENT1"]
    )
    sun_clusters_pl, sun_row_pos_pl, _ = make_cluster_row_map(
        settings["SUN_PL_CONF90"]
    )

    if sun_clusters_t1 != sun_clusters_pl:
        raise RuntimeError(
            "SUN TENT1/PL physical cluster sets differ."
        )

    # Shared bootstrap multiplicities.
    pg_counts = generate_bootstrap_cluster_counts(
        pg_clusters,
        BOOTSTRAPS,
        POLYPGEN_BOOTSTRAP_SEED,
    )
    sun_counts = generate_bootstrap_cluster_counts(
        sun_clusters_t1,
        BOOTSTRAPS,
        SUN_SHARED_BOOTSTRAP_SEED,
    )

    bootstrap_metric_frames = []
    bootstrap_delta_frames = []
    fixed_metric_frames = []
    fixed_delta_frames = []

    for setting_name in SETTING_ORDER:
        if setting_name == "PolypGen_TENT1":
            counts = pg_counts
            row_pos = pg_row_pos
        elif setting_name == "SUN_TENT1":
            counts = sun_counts
            row_pos = sun_row_pos_t1
        elif setting_name == "SUN_PL_CONF90":
            counts = sun_counts
            row_pos = sun_row_pos_pl
        else:
            raise RuntimeError(setting_name)

        (
            bm,
            bd,
            fm,
            fd,
        ) = bootstrap_setting(
            setting_name,
            settings[setting_name],
            counts,
            row_pos,
        )

        bootstrap_metric_frames.append(bm)
        bootstrap_delta_frames.append(bd)
        fixed_metric_frames.append(fm)
        fixed_delta_frames.append(fd)

    boot_metrics = pd.concat(
        bootstrap_metric_frames,
        ignore_index=True,
    )
    boot_deltas = pd.concat(
        bootstrap_delta_frames,
        ignore_index=True,
    )
    fixed_boot_metrics = pd.concat(
        fixed_metric_frames,
        ignore_index=True,
    )
    fixed_boot_deltas = pd.concat(
        fixed_delta_frames,
        ignore_index=True,
    )

    curve_ci = summarize_curve_bootstrap(
        boot_metrics
    )
    delta_ci = summarize_delta_bootstrap(
        boot_deltas
    )
    fixed_ci = summarize_fixed_bootstrap(
        fixed_boot_metrics
    )
    fixed_delta_ci = summarize_fixed_delta_bootstrap(
        fixed_boot_deltas
    )

    print("\n===== SELECTIVE-vs-RANDOM 95% CI: KEY METRICS =====")
    key_delta = delta_ci[
        delta_ci["metric"].isin(
            [
                "harm_rate_among_adapted",
                "prevented_harm_fraction",
                "mean_deployed_dice",
            ]
        )
    ]
    print(
        key_delta[
            [
                "setting",
                "requested_coverage",
                "metric",
                "bootstrap_mean_delta",
                "ci95_low",
                "ci95_high",
            ]
        ].to_string(index=False)
    )

    print("\n===== FROZEN-THRESHOLD SELECTIVE-vs-RANDOM 95% CI =====")
    key_fixed_delta = fixed_delta_ci[
        fixed_delta_ci["metric"].isin(
            [
                "harm_rate_among_adapted",
                "prevented_harm_fraction",
                "mean_deployed_dice",
            ]
        )
    ]
    print(
        key_fixed_delta[
            [
                "setting",
                "metric",
                "bootstrap_mean_delta",
                "ci95_low",
                "ci95_high",
            ]
        ].to_string(index=False)
    )

    # Save compact products, not raw bootstrap rows.
    point_curve_path = (
        args.output_dir
        / "R15B1_selective_utility_point_curve.csv"
    )
    point_delta_path = (
        args.output_dir
        / "R15B1_selective_vs_random_point_delta.csv"
    )
    fixed_point_path = (
        args.output_dir
        / "R15B1_frozen_threshold_policy_point.csv"
    )
    fixed_delta_point_path = (
        args.output_dir
        / "R15B1_frozen_threshold_vs_random_point_delta.csv"
    )
    curve_ci_path = (
        args.output_dir
        / "R15B1_selective_utility_cluster_bootstrap_ci.csv"
    )
    delta_ci_path = (
        args.output_dir
        / "R15B1_selective_vs_random_cluster_bootstrap_delta_ci.csv"
    )
    fixed_ci_path = (
        args.output_dir
        / "R15B1_frozen_threshold_cluster_bootstrap_ci.csv"
    )
    fixed_delta_ci_path = (
        args.output_dir
        / "R15B1_frozen_threshold_vs_random_cluster_bootstrap_delta_ci.csv"
    )

    point_curve_df.to_csv(
        point_curve_path,
        index=False,
    )
    point_delta_df.to_csv(
        point_delta_path,
        index=False,
    )
    fixed_point_df.to_csv(
        fixed_point_path,
        index=False,
    )
    fixed_delta_df.to_csv(
        fixed_delta_point_path,
        index=False,
    )
    curve_ci.to_csv(
        curve_ci_path,
        index=False,
    )
    delta_ci.to_csv(
        delta_ci_path,
        index=False,
    )
    fixed_ci.to_csv(
        fixed_ci_path,
        index=False,
    )
    fixed_delta_ci.to_csv(
        fixed_delta_ci_path,
        index=False,
    )

    artifacts = {}
    for p in (
        point_curve_path,
        point_delta_path,
        fixed_point_path,
        fixed_delta_point_path,
        curve_ci_path,
        delta_ci_path,
        fixed_ci_path,
        fixed_delta_ci_path,
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
            "post-freeze external selective-adaptation "
            "risk-coverage utility analysis"
        ),
        "frozen_policy": {
            "score_orientation": (
                "higher frozen safety probability = higher HARM risk"
            ),
            "coverage_grid": COVERAGE_GRID,
            "ranking_rule": (
                "adapt lowest-risk c fraction of model-frame rows; "
                "retain SOURCE for highest-risk 1-c fraction"
            ),
            "coverage_count_rule": (
                "nearest integer via floor(c*N + 0.5)"
            ),
            "tie_break": (
                "score, model_family, model_state_id, sample_id, "
                "original row order"
            ),
            "frozen_threshold": FROZEN_THRESHOLD,
            "fixed_threshold_rule": (
                "adapt iff score < frozen_threshold"
            ),
            "random_reference": (
                "exact expectation under uniform row-level selection "
                "at identical coverage"
            ),
        },
        "bootstrap": {
            "replicates": BOOTSTRAPS,
            "PolypGen_cluster": "sample_id",
            "PolypGen_seed": POLYPGEN_BOOTSTRAP_SEED,
            "SUN_cluster": "cluster_id",
            "SUN_shared_seed": SUN_SHARED_BOOTSTRAP_SEED,
            "same_cluster_draw_across_coverages": True,
            "same_SUN_cluster_draw_for_TENT1_and_PL": True,
            "duplicate_clusters_preserved_by_integer_multiplicity": True,
        },
        "information_boundary": {
            "model_fitting": False,
            "pca_fitting": False,
            "score_recomputation": False,
            "target_calibration": False,
            "threshold_tuning": False,
            "coverage_selection_by_gt": False,
            "feature_change": False,
            "case_exclusion": False,
            "action_specific_safety_retraining": False,
            "best_coverage_selected_or_deployed": False,
        },
        "upstream": {
            "r15b0_lock_sha256": EXPECTED_R15B0_LOCK_SHA,
            "panels": {
                name: {
                    "path": str(panel_paths[name]),
                    "sha256": sha256_file(
                        panel_paths[name]
                    ),
                }
                for name in SETTING_ORDER
            },
        },
        "artifacts": artifacts,
        "next_stage": (
            "R15C_CONTINUOUS_RISK_VS_DELTA_DICE_ANALYSIS"
        ),
    }

    lock_path = (
        args.output_dir
        / "R15B1_SELECTIVE_ADAPTATION_UTILITY_LOCK.json"
    )
    write_json(
        lock_path,
        lock,
    )

    print("\n===== R15B1 FINAL =====")
    print("Decision=", DECISION)
    print("BEST_COVERAGE_SELECTED_OR_DEPLOYED=NO")
    print("TARGET_POLICY_RETUNING=NO")
    print("LOCK=", lock_path)
    print("LOCK SHA256=", sha256_file(lock_path))
    print("PASS")


if __name__ == "__main__":
    main()
