#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R10K2D_confirmed_method_freeze_fix1.py

Purpose
-------
Freeze the R10K2A/R10K2B method after R10K2C confirmatory clustered inference.

This script DOES NOT:
- retrain models
- refit PCA
- change features
- change thresholds
- tune hyperparameters
- inspect new datasets

It verifies the already-produced K2B/K2C summary files and writes a single
paper-facing method lock containing:
- representation definition
- evaluation protocol
- confirmed macro effects
- family-specific caveats
- claim boundary
- next required evidence: independent external cohort

The goal is to prevent post-hoc method drift before external validation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import pandas as pd


DEFAULT_K2B_MACRO = (
    "F:/MEDSEG_SAFETTA/outputs/"
    "Q1_R10K2B_dinov2_image_mask_LOMO_evaluation_fix2_v1/"
    "R10K2B_macro_summary.csv"
)

DEFAULT_K2B_FAMILY = (
    "F:/MEDSEG_SAFETTA/outputs/"
    "Q1_R10K2B_dinov2_image_mask_LOMO_evaluation_fix2_v1/"
    "R10K2B_by_family_summary.csv"
)

DEFAULT_K2C_DELTA = (
    "F:/MEDSEG_SAFETTA/outputs/"
    "Q1_R10K2C_case_clustered_paired_bootstrap_fix1_v1/"
    "R10K2C_paired_macro_delta_ci.csv"
)

DEFAULT_K2C_FAMILY_DELTA = (
    "F:/MEDSEG_SAFETTA/outputs/"
    "Q1_R10K2C_case_clustered_paired_bootstrap_fix1_v1/"
    "R10K2C_family_primary_delta_ci.csv"
)

DEFAULT_K2C_SUMMARY = (
    "F:/MEDSEG_SAFETTA/outputs/"
    "Q1_R10K2C_case_clustered_paired_bootstrap_fix1_v1/"
    "R10K2C_summary.json"
)

DEFAULT_OUTPUT = (
    "F:/MEDSEG_SAFETTA/outputs/"
    "Q1_R10K2D_confirmed_method_freeze_fix1_v1"
)

PRIMARY = "M2_plus_CondDINO_PCA64"
BASELINE = "M2_backbone"
COND = "CondDINO_PCA64"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k2b_macro", default=DEFAULT_K2B_MACRO)
    ap.add_argument("--k2b_family", default=DEFAULT_K2B_FAMILY)
    ap.add_argument("--k2c_delta", default=DEFAULT_K2C_DELTA)
    ap.add_argument(
        "--k2c_family_delta",
        default=DEFAULT_K2C_FAMILY_DELTA,
    )
    ap.add_argument(
        "--k2c_summary",
        default=DEFAULT_K2C_SUMMARY,
    )
    ap.add_argument(
        "--output_dir",
        default=DEFAULT_OUTPUT,
    )
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths = {
        "k2b_macro": Path(args.k2b_macro),
        "k2b_family": Path(args.k2b_family),
        "k2c_delta": Path(args.k2c_delta),
        "k2c_family_delta": Path(args.k2c_family_delta),
        "k2c_summary": Path(args.k2c_summary),
    }

    print("===== R10K2D CONFIRMED METHOD FREEZE FIX1 =====")
    print("STATUS: METHOD / CLAIM FREEZE")
    print("MODEL FITTING: NONE")
    print("PCA FITTING: NONE")
    print("FEATURE CHANGES: NONE")
    print("THRESHOLD CHANGES: NONE")
    print("HYPERPARAMETER TUNING: NONE")
    print("NEW DATASET INSPECTION: NONE")
    print()

    for name, p in paths.items():
        print(name, "exists:", p.exists(), p)
        if not p.exists():
            raise FileNotFoundError(p)

    macro = pd.read_csv(paths["k2b_macro"])
    family = pd.read_csv(paths["k2b_family"])
    delta = pd.read_csv(paths["k2c_delta"])
    family_delta = pd.read_csv(paths["k2c_family_delta"])

    with open(paths["k2c_summary"], "r", encoding="utf-8") as f:
        summary = json.load(f)

    expected_decision = (
        "PAIRED_CLUSTER_BOOTSTRAP_CONFIRMS_MACRO_PRIMARY_BENEFIT"
    )
    actual_decision = summary.get("decision")

    print("\nK2C decision:", actual_decision)
    if actual_decision != expected_decision:
        raise AssertionError(
            f"Expected K2C decision {expected_decision}, "
            f"got {actual_decision}"
        )

    if PRIMARY not in set(macro["method"].astype(str)):
        raise AssertionError("Primary method missing from K2B macro table.")
    if BASELINE not in set(macro["method"].astype(str)):
        raise AssertionError("Baseline missing from K2B macro table.")

    p = macro[macro["method"] == PRIMARY].iloc[0]
    b = macro[macro["method"] == BASELINE].iloc[0]
    c = macro[macro["method"] == COND].iloc[0]

    pdlt = delta[
        delta["comparison"] == "PRIMARY_vs_M2"
    ].set_index("metric")

    required_metrics = [
        "AUROC",
        "AUPRC",
        "FPR",
        "PPV1pct",
        "OracleR90FPR",
    ]
    for m in required_metrics:
        if m not in pdlt.index:
            raise AssertionError(
                f"Missing primary paired delta metric: {m}"
            )

    # Family-specific primary AUROC/FPR intervals.
    fam_auc = family_delta[
        family_delta["metric"] == "AUROC"
    ].copy()
    fam_fpr = family_delta[
        family_delta["metric"] == "FPR"
    ].copy()

    if set(fam_auc["target_family"]) != {
        "DeepLabV3-R50",
        "PraNet",
        "SegFormer-B0",
    }:
        raise AssertionError("Unexpected family set in K2C AUROC CIs.")

    # Frozen method definition.
    method_lock = {
        "status": "FROZEN",
        "decision": "R10K2_PRIMARY_METHOD_FROZEN_FOR_EXTERNAL_VALIDATION",
        "primary_method": {
            "name": PRIMARY,
            "input_rgb": True,
            "pre_adaptation_source_mask": True,
            "encoder": "facebook/dinov2-base",
            "encoder_trainable": False,
            "image_resize": "224x224 direct bicubic",
            "dinov2_patch_grid": "16x16",
            "source_mask_alignment": (
                "352x352 -> exact non-overlapping 22x22 occupancy -> 16x16"
            ),
            "semantic_pooling": [
                "foreground-weighted patch mean 768-D",
                "background-weighted patch mean 768-D",
            ],
            "semantic_concat_dimension": 1536,
            "pca_dimension": 64,
            "pca_fit_scope": "source-training rows only",
            "structural_auxiliary": [
                "morph_fg_fraction",
                "morph_boundary_density",
            ],
            "safety_head": (
                "median imputation + standard scaling + "
                "class-balanced logistic regression"
            ),
            "operating_point": (
                "source-only inner grouped OOF threshold for Recall>=0.90"
            ),
        },
        "frozen_evaluation_protocol": {
            "generalization_axis": (
                "leave-one-model-family-out within NeoPolyp"
            ),
            "target_families": [
                "DeepLabV3-R50",
                "PraNet",
                "SegFormer-B0",
            ],
            "group_key": "sample_id",
            "outer_folds": 5,
            "inner_folds": 4,
            "split_seeds": [
                20260816,
                20260817,
                20260818,
                20260819,
                20260820,
            ],
            "target_labels_fit": False,
            "target_features_fit_pca": False,
            "clustered_bootstrap_unit": "sample_id",
            "clustered_bootstrap_replicates": 2000,
        },
        "confirmed_point_estimates": {
            "baseline_macro_AUROC": float(b["macro_AUROC_mean"]),
            "primary_macro_AUROC": float(p["macro_AUROC_mean"]),
            "baseline_macro_AUPRC": float(b["macro_AUPRC_mean"]),
            "primary_macro_AUPRC": float(p["macro_AUPRC_mean"]),
            "baseline_macro_transferred_FPR": float(
                b["macro_transfer_FPR_mean"]
            ),
            "primary_macro_transferred_FPR": float(
                p["macro_transfer_FPR_mean"]
            ),
            "baseline_macro_PPV1pct": float(
                b["macro_PPV1pct_mean"]
            ),
            "primary_macro_PPV1pct": float(
                p["macro_PPV1pct_mean"]
            ),
            "baseline_macro_oracle_R90_FPR": float(
                b["macro_oracle_R90_FPR_mean"]
            ),
            "primary_macro_oracle_R90_FPR": float(
                p["macro_oracle_R90_FPR_mean"]
            ),
            "cond_only_macro_AUROC": float(c["macro_AUROC_mean"]),
            "cond_only_macro_AUPRC": float(c["macro_AUPRC_mean"]),
        },
        "confirmed_paired_macro_delta_ci95": {
            m: {
                "mean": float(pdlt.loc[m, "bootstrap_mean_delta"]),
                "low": float(pdlt.loc[m, "ci95_low"]),
                "high": float(pdlt.loc[m, "ci95_high"]),
            }
            for m in required_metrics
        },
        "family_primary_AUROC_delta_ci95": {
            r["target_family"]: {
                "mean": float(r["bootstrap_mean_delta"]),
                "low": float(r["ci95_low"]),
                "high": float(r["ci95_high"]),
            }
            for _, r in fam_auc.iterrows()
        },
        "family_primary_FPR_delta_ci95": {
            r["target_family"]: {
                "mean": float(r["bootstrap_mean_delta"]),
                "low": float(r["ci95_low"]),
                "high": float(r["ci95_high"]),
            }
            for _, r in fam_fpr.iterrows()
        },
        "claim_boundary": {
            "supported": (
                "cross-model-family TTA safety-risk ranking within NeoPolyp, "
                "with macro clustered inferential support"
            ),
            "not_supported": [
                "independent external-cohort validation",
                "uniform per-family AUROC superiority",
                "uniformly improved transferred high-recall operating point",
                "automatic clinically reliable safety gate",
            ],
            "known_family_caveat": (
                "SegFormer-B0 transferred FPR increases despite improved "
                "ranking/oracle-R90 FPR; source-threshold calibration remains "
                "architecture dependent."
            ),
        },
        "post_freeze_rules": [
            "Do not change DINOv2 model.",
            "Do not change image resize.",
            "Do not change source-mask alignment.",
            "Do not change foreground/background pooling.",
            "Do not sweep PCA dimension.",
            "Do not change safety classifier.",
            "Do not select features using external-cohort labels.",
            "Do not retune the source-only Recall>=0.90 threshold protocol.",
        ],
        "next_required_evidence": (
            "independent external-cohort validation with the frozen method"
        ),
    }

    lock_path = out / "R10K2D_CONFIRMED_METHOD_LOCK.json"
    with open(lock_path, "w", encoding="utf-8") as f:
        json.dump(method_lock, f, indent=2, ensure_ascii=False)

    # Paper-facing concise summary.
    lines = []
    lines.append("# R10K2 Confirmed Method Freeze")
    lines.append("")
    lines.append(
        "**Frozen primary:** `M2_plus_CondDINO_PCA64`"
    )
    lines.append("")
    lines.append(
        f"- Macro AUROC: {b['macro_AUROC_mean']:.6f} -> "
        f"{p['macro_AUROC_mean']:.6f}"
    )
    lines.append(
        f"- Macro AUPRC: {b['macro_AUPRC_mean']:.6f} -> "
        f"{p['macro_AUPRC_mean']:.6f}"
    )
    lines.append(
        f"- Transferred macro FPR: "
        f"{b['macro_transfer_FPR_mean']:.6f} -> "
        f"{p['macro_transfer_FPR_mean']:.6f}"
    )
    lines.append(
        f"- Macro PPV@1%: {b['macro_PPV1pct_mean']:.6f} -> "
        f"{p['macro_PPV1pct_mean']:.6f}"
    )
    lines.append(
        f"- Oracle R90 macro FPR: "
        f"{b['macro_oracle_R90_FPR_mean']:.6f} -> "
        f"{p['macro_oracle_R90_FPR_mean']:.6f}"
    )
    lines.append("")
    lines.append("## Paired case-clustered bootstrap")
    for m in required_metrics:
        row = pdlt.loc[m]
        lines.append(
            f"- Delta {m}: {row['bootstrap_mean_delta']:.6f}, "
            f"95% CI [{row['ci95_low']:.6f}, "
            f"{row['ci95_high']:.6f}]"
        )
    lines.append("")
    lines.append("## Frozen claim")
    lines.append(
        "The current evidence supports cross-model-family TTA safety-risk "
        "ranking within NeoPolyp with macro case-clustered inferential support."
    )
    lines.append("")
    lines.append(
        "It does not yet support independent external-cohort generalization "
        "or a uniformly transferable automatic high-recall safety gate."
    )
    lines.append("")
    lines.append(
        "Next required evidence: independent external-cohort validation with "
        "the method frozen exactly as above."
    )

    md_path = out / "R10K2D_paper_facing_freeze_summary.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print("\n===== CONFIRMED PRIMARY =====")
    print(
        "Macro AUROC:",
        float(b["macro_AUROC_mean"]),
        "->",
        float(p["macro_AUROC_mean"]),
    )
    print(
        "Macro AUPRC:",
        float(b["macro_AUPRC_mean"]),
        "->",
        float(p["macro_AUPRC_mean"]),
    )
    print(
        "Transferred macro FPR:",
        float(b["macro_transfer_FPR_mean"]),
        "->",
        float(p["macro_transfer_FPR_mean"]),
    )
    print(
        "Macro PPV@1%:",
        float(b["macro_PPV1pct_mean"]),
        "->",
        float(p["macro_PPV1pct_mean"]),
    )
    print(
        "Oracle R90 macro FPR:",
        float(b["macro_oracle_R90_FPR_mean"]),
        "->",
        float(p["macro_oracle_R90_FPR_mean"]),
    )

    print("\n===== FAMILY CAVEAT =====")
    print(
        fam_fpr[
            ["target_family", "bootstrap_mean_delta", "ci95_low", "ci95_high"]
        ].to_string(index=False)
    )

    print("\n===== FINAL DECISION =====")
    print("DECISION: R10K2_PRIMARY_METHOD_FROZEN_FOR_EXTERNAL_VALIDATION")
    print("METHOD CHANGES AFTER THIS POINT: FORBIDDEN")
    print("NEXT REQUIRED EVIDENCE: INDEPENDENT EXTERNAL COHORT")
    print("\nOutputs:")
    print(lock_path)
    print(md_path)
    print("PASS")


if __name__ == "__main__":
    main()
