#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R10L3D_polypgen_external_generalization_diagnostic_freeze_fix1.py

Paper-facing diagnostic consolidation AFTER the locked prospective PolypGen
evaluation (R10L3C).

NO model training.
NO score fitting.
NO recalibration.
NO threshold change.
NO score reversal.
NO feature selection.
NO case/center/state exclusion.

Purpose:
- summarize the already-locked independent external result;
- quantify how far the ranking signal is above chance / prevalence baselines;
- characterize family and center heterogeneity;
- separate ranking quality from high-recall operating-point weakness;
- freeze a paper claim boundary without changing the method.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(r"F:\MEDSEG_SAFETTA")

R10L3C_DIR = (
    ROOT / "outputs"
    / "Q1_R10L3C_polypgen_gt_reveal_prospective_external_evaluation_fix1_v1"
)
R10L3C_LOCK = (
    R10L3C_DIR
    / "R10L3C_POLYPGEN_PROSPECTIVE_EXTERNAL_EVALUATION_LOCK.json"
)
EXPECTED_R10L3C_LOCK_SHA256 = (
    "cf9e42ea7f0f6c1d53844a2358b05007b46d2a99124b585014626704245d3861"
)
EXPECTED_R10L3C_DECISION = (
    "POLYPGEN_PROSPECTIVE_EXTERNAL_EVALUATION_COMPLETE_NO_RETUNING"
)

PANEL = R10L3C_DIR / "R10L3C_LOCKED_SCORE_GT_EVALUATION_PANEL.csv"
POINT = R10L3C_DIR / "R10L3C_external_point_metrics.csv"
BOOT = R10L3C_DIR / "R10L3C_clustered_bootstrap_ci.csv"

OUTPUT_DIR = (
    ROOT / "outputs"
    / "Q1_R10L3D_polypgen_external_generalization_diagnostic_freeze_fix1_v1"
)

FROZEN_THRESHOLD = 0.300584763193734
EXPECTED_ROWS = 13788
EXPECTED_CASES = 1532
EXPECTED_STATES = 9

FAMILIES = ("DeepLabV3-R50", "PraNet", "SegFormer-B0")
CENTERS = ("C1", "C2", "C3", "C4", "C5", "C6")

DECISION = (
    "INDEPENDENT_EXTERNAL_RANKING_SIGNAL_PRESENT_"
    "BUT_HIGH_RECALL_TRANSFER_WEAK"
)


def sha256_file(path: Path, chunk=16 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def write_json(path: Path, obj):
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def verify():
    for label, p in {
        "R10L3C lock": R10L3C_LOCK,
        "evaluation panel": PANEL,
        "point metrics": POINT,
        "bootstrap CI": BOOT,
    }.items():
        print(label, "exists:", p.exists(), p)
        if not p.exists():
            raise FileNotFoundError(p)

    actual = sha256_file(R10L3C_LOCK)
    if actual.lower() != EXPECTED_R10L3C_LOCK_SHA256.lower():
        raise RuntimeError(
            f"R10L3C lock SHA mismatch: expected "
            f"{EXPECTED_R10L3C_LOCK_SHA256}, actual {actual}"
        )

    lock = json.loads(R10L3C_LOCK.read_text(encoding="utf-8"))
    if lock.get("decision") != EXPECTED_R10L3C_DECISION:
        raise RuntimeError(
            f"Unexpected R10L3C decision: {lock.get('decision')}"
        )

    if int(lock.get("target_cases", -1)) != EXPECTED_CASES:
        raise RuntimeError("R10L3C case count changed.")
    if int(lock.get("model_states", -1)) != EXPECTED_STATES:
        raise RuntimeError("R10L3C state count changed.")
    if int(lock.get("model_case_rows", -1)) != EXPECTED_ROWS:
        raise RuntimeError("R10L3C row count changed.")

    return lock, actual


def metric_row(df):
    y = df["harm_label"].to_numpy(dtype=int)
    p = df["frozen_safety_probability"].to_numpy(dtype=float)
    pred = p >= FROZEN_THRESHOLD

    n = len(df)
    harm = int(y.sum())
    nonharm = n - harm

    if harm == 0 or nonharm == 0:
        return {
            "rows": n,
            "harm": harm,
            "nonharm": nonharm,
            "harm_prevalence": harm / n if n else math.nan,
            "auroc": math.nan,
            "auprc": math.nan,
            "auprc_minus_prevalence": math.nan,
            "auprc_over_prevalence": math.nan,
            "recall": math.nan,
            "fpr": math.nan,
            "flagged_fraction": float(pred.mean()) if n else math.nan,
        }

    tp = int(np.count_nonzero(pred & (y == 1)))
    fp = int(np.count_nonzero(pred & (y == 0)))
    fn = int(np.count_nonzero((~pred) & (y == 1)))
    tn = int(np.count_nonzero((~pred) & (y == 0)))

    recall = tp / (tp + fn)
    fpr = fp / (fp + tn)
    prev = harm / n

    auroc = float(roc_auc_score(y, p))
    auprc = float(average_precision_score(y, p))

    return {
        "rows": n,
        "harm": harm,
        "nonharm": nonharm,
        "harm_prevalence": prev,
        "auroc": auroc,
        "auprc": auprc,
        "auprc_minus_prevalence": auprc - prev,
        "auprc_over_prevalence": auprc / prev,
        "recall": recall,
        "fpr": fpr,
        "flagged_fraction": float(pred.mean()),
    }


def fixed_calibration_diagnostic(df, bins=10):
    y = df["harm_label"].to_numpy(dtype=float)
    p = df["frozen_safety_probability"].to_numpy(dtype=float)

    brier = float(np.mean((p - y) ** 2))

    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    ece = 0.0

    for i in range(bins):
        lo = edges[i]
        hi = edges[i + 1]
        if i < bins - 1:
            mask = (p >= lo) & (p < hi)
        else:
            mask = (p >= lo) & (p <= hi)

        n = int(mask.sum())
        if n == 0:
            rows.append({
                "bin": i,
                "lower": lo,
                "upper": hi,
                "rows": 0,
                "mean_probability": math.nan,
                "observed_harm_rate": math.nan,
                "absolute_gap": math.nan,
            })
            continue

        mp = float(p[mask].mean())
        obs = float(y[mask].mean())
        gap = abs(mp - obs)
        ece += (n / len(df)) * gap

        rows.append({
            "bin": i,
            "lower": lo,
            "upper": hi,
            "rows": n,
            "mean_probability": mp,
            "observed_harm_rate": obs,
            "absolute_gap": gap,
        })

    return rows, brier, float(ece)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = ap.parse_args()

    print("===== R10L3D POLYPGEN EXTERNAL GENERALIZATION DIAGNOSTIC FREEZE FIX1 =====")
    print("MODEL TRAINING: NO")
    print("SCORE FITTING: NO")
    print("RECALIBRATION: NO")
    print("THRESHOLD CHANGE: NO")
    print("SCORE REVERSAL: NO")
    print("FEATURE SELECTION: NO")
    print("TARGET SUBSET SELECTION: NO")
    print()

    lock, lock_sha = verify()

    panel = pd.read_csv(PANEL, low_memory=False)
    point = pd.read_csv(POINT, low_memory=False)
    boot = pd.read_csv(BOOT, low_memory=False)

    if len(panel) != EXPECTED_ROWS:
        raise RuntimeError(f"Panel rows={len(panel)}")
    if panel["sample_id"].nunique() != EXPECTED_CASES:
        raise RuntimeError("Panel case count mismatch.")
    if panel["model_state_id"].nunique() != EXPECTED_STATES:
        raise RuntimeError("Panel state count mismatch.")

    required = [
        "sample_id", "center", "model_family", "model_state_id",
        "harm_label", "adaptation_outcome",
        "frozen_safety_probability", "frozen_risk_flag",
    ]
    missing = [c for c in required if c not in panel.columns]
    if missing:
        raise RuntimeError(f"Panel missing: {missing}")

    primary = point[
        point["level"].astype(str) == "PRIMARY_MACRO_FAMILY"
    ].copy()
    if len(primary) != 1:
        raise RuntimeError("Expected exactly one primary macro row.")
    primary_row = primary.iloc[0].to_dict()

    primary_ci = boot[
        boot["level"].astype(str) == "PRIMARY_MACRO_FAMILY"
    ].copy()

    needed_ci_metrics = {
        "auroc", "auprc", "recall", "fpr",
        "empirical_ppv", "ppv_at_1pct",
    }
    if set(primary_ci["metric"].astype(str)) != needed_ci_metrics:
        raise RuntimeError("Primary bootstrap metric set changed.")

    grid_rows = []
    for fam in FAMILIES:
        for center in CENTERS:
            sub = panel[
                (panel["model_family"].astype(str) == fam)
                & (panel["center"].astype(str) == center)
            ]
            row = metric_row(sub)
            row["model_family"] = fam
            row["center"] = center
            grid_rows.append(row)

    grid = pd.DataFrame(grid_rows)

    dist_rows = []
    for outcome in ["HARM", "NEUTRAL", "BENEFIT"]:
        sub = panel[
            panel["adaptation_outcome"].astype(str) == outcome
        ]["frozen_safety_probability"].to_numpy(dtype=float)

        dist_rows.append({
            "adaptation_outcome": outcome,
            "rows": len(sub),
            "mean": float(np.mean(sub)),
            "q10": float(np.quantile(sub, 0.10)),
            "q25": float(np.quantile(sub, 0.25)),
            "median": float(np.quantile(sub, 0.50)),
            "q75": float(np.quantile(sub, 0.75)),
            "q90": float(np.quantile(sub, 0.90)),
        })

    score_dist = pd.DataFrame(dist_rows)

    calibration_rows, brier, ece = fixed_calibration_diagnostic(panel)

    macro_auroc = float(primary_row["auroc"])
    macro_auprc = float(primary_row["auprc"])
    macro_prev = float(primary_row["harm_prevalence"])
    macro_recall = float(primary_row["recall"])
    macro_fpr = float(primary_row["fpr"])
    oracle_fpr90 = float(
        primary_row["oracle_target_fpr_at_r90_diagnostic"]
    )

    auroc_ci = primary_ci[
        primary_ci["metric"] == "auroc"
    ].iloc[0]
    auprc_ci = primary_ci[
        primary_ci["metric"] == "auprc"
    ].iloc[0]

    auroc_ci_low = float(auroc_ci["ci_low_2p5"])
    auroc_ci_high = float(auroc_ci["ci_high_97p5"])
    auprc_ci_low = float(auprc_ci["ci_low_2p5"])
    auprc_ci_high = float(auprc_ci["ci_high_97p5"])

    ranking_above_chance = auroc_ci_low > 0.5
    auprc_above_prevalence = macro_auprc > macro_prev
    high_recall_weak = oracle_fpr90 > 0.80

    print("===== PAPER-FACING LOCKED EXTERNAL FACTS =====")
    print("HARM prevalence:", f"{macro_prev:.6f}")
    print(
        "Macro AUROC:",
        f"{macro_auroc:.6f}",
        f"CI95=[{auroc_ci_low:.6f}, {auroc_ci_high:.6f}]",
    )
    print(
        "Macro AUPRC:",
        f"{macro_auprc:.6f}",
        f"CI95=[{auprc_ci_low:.6f}, {auprc_ci_high:.6f}]",
    )
    print(
        "AUPRC - harm prevalence:",
        f"{macro_auprc - macro_prev:.6f}",
    )
    print(
        "AUPRC / harm prevalence:",
        f"{macro_auprc / macro_prev:.3f}x",
    )
    print("Frozen-threshold Recall:", f"{macro_recall:.6f}")
    print("Frozen-threshold FPR:", f"{macro_fpr:.6f}")
    print("Oracle target FPR@R90:", f"{oracle_fpr90:.6f}")
    print("AUROC CI entirely >0.5:", ranking_above_chance)
    print("AUPRC > prevalence:", auprc_above_prevalence)
    print("High-recall operating point weak:", high_recall_weak)
    print("Brier score diagnostic:", f"{brier:.6f}")
    print("10-bin ECE diagnostic:", f"{ece:.6f}")

    print("\n===== INTERPRETATION FREEZE =====")
    print("Independent external ranking signal: PRESENT")
    print("Strength: MODERATE, NOT STRONG")
    print("Frozen operating-point transfer: WEAK")
    print("High-recall discrimination: WEAK")
    print("Automatic universal safety gate claim: NOT SUPPORTED")
    print("Independent multicenter risk-ranking claim: SUPPORTED WITH QUALIFICATION")
    print("Post-GT method modification authorized: NO")

    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)

    grid.to_csv(
        args.output_dir / "R10L3D_family_center_metric_grid.csv",
        index=False,
    )
    score_dist.to_csv(
        args.output_dir / "R10L3D_score_distribution_by_outcome.csv",
        index=False,
    )
    pd.DataFrame(calibration_rows).to_csv(
        args.output_dir / "R10L3D_fixed10bin_calibration_diagnostic.csv",
        index=False,
    )

    primary_ci.to_csv(
        args.output_dir / "R10L3D_primary_bootstrap_ci.csv",
        index=False,
    )

    summary = {
        "decision": DECISION,
        "r10l3c_lock_sha256": lock_sha,
        "target_cases": EXPECTED_CASES,
        "model_case_rows": EXPECTED_ROWS,
        "harm_prevalence": macro_prev,
        "primary_macro_auroc": macro_auroc,
        "primary_macro_auroc_ci95": [auroc_ci_low, auroc_ci_high],
        "primary_macro_auprc": macro_auprc,
        "primary_macro_auprc_ci95": [auprc_ci_low, auprc_ci_high],
        "auprc_minus_harm_prevalence": macro_auprc - macro_prev,
        "auprc_over_harm_prevalence": macro_auprc / macro_prev,
        "frozen_threshold_recall": macro_recall,
        "frozen_threshold_fpr": macro_fpr,
        "oracle_target_fpr_at_r90_diagnostic": oracle_fpr90,
        "brier_score_diagnostic": brier,
        "ece_fixed10bin_diagnostic": ece,
        "ranking_signal_above_chance_by_clustered_ci": ranking_above_chance,
        "auprc_above_prevalence": auprc_above_prevalence,
        "high_recall_operating_point_weak": high_recall_weak,
        "claim_boundary": {
            "independent_multicenter_risk_ranking": "SUPPORTED_WITH_QUALIFICATION",
            "strong_cross_dataset_ranking": "NOT_SUPPORTED",
            "uniform_family_superiority": "NOT_ESTABLISHED",
            "automatic_universal_safety_gate": "NOT_SUPPORTED",
            "high_recall_operating_point_transfer": "NOT_SUPPORTED",
            "post_gt_retuning": "FORBIDDEN",
        },
        "method_changes_after_gt_reveal": False,
    }

    write_json(
        args.output_dir / "R10L3D_EXTERNAL_GENERALIZATION_CLAIM_LOCK.json",
        summary,
    )

    print("\n===== FINAL DECISION =====")
    print("DECISION:", DECISION)
    print("METHOD CHANGES AFTER GT REVEAL: FORBIDDEN")
    print("PASS")


if __name__ == "__main__":
    main()
