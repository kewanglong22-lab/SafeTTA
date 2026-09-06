#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R15E0_final_experiment_evidence_synthesis_fix1.py

Read-only final experiment evidence synthesis for the SafeTTA paper.

This stage performs:
- no model inference;
- no fitting/refitting;
- no score recomputation;
- no threshold tuning;
- no target calibration;
- no GT modification;
- no method reselection.

It audits the frozen R15A/B/C/D result locks and assembles paper-ready
summary tables from already generated CSV/JSON artifacts.

Outputs:
1) Table_R15A_core_ablation.csv
2) Table_R15A_paired_component_inference.csv
3) Table_R15B_selective_utility.csv
4) Table_R15C_continuous_severity.csv
5) Table_R15D_runtime_overhead.csv
6) R15E0_PAPER_READY_RESULTS.md
7) R15E0_FINAL_EXPERIMENT_EVIDENCE_LOCK.json
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


VERSION = "2026-09-04-Q1-R15E0-v1-fix1"
BUILD = "Q1_R15E0_FINAL_EXPERIMENT_EVIDENCE_SYNTHESIS_FIX1"

ROOT = Path(r"F:\MEDSEG_SAFETTA")
OUT = ROOT / "outputs"

R15A1_DIR = (
    OUT
    / "Q1_R15A1_core_ablation_case_clustered_paired_bootstrap_fix2_v1"
)
R15A1_LOCK = (
    R15A1_DIR
    / "R15A1_CORE_ABLATION_PAIRED_BOOTSTRAP_LOCK.json"
)
EXPECTED_R15A1_LOCK_SHA = (
    "d7ab63c9f643e39b237e17cfd7bf761730519ccb8aea53f3ad4809c63e0d6a45"
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

R15C1_DIR = (
    OUT
    / "Q1_R15C1_continuous_risk_delta_dice_external_analysis_fix1_v1"
)
R15C1_LOCK = (
    R15C1_DIR
    / "R15C1_CONTINUOUS_RISK_EXTERNAL_ANALYSIS_LOCK.json"
)
EXPECTED_R15C1_LOCK_SHA = (
    "5aa2299b541df31f488d252ed3e59505f3b646d58141b74c863b2c569d697e17"
)

R15D1_DIR = (
    OUT
    / "Q1_R15D1_runtime_computational_overhead_benchmark_fix1_v1"
)
R15D1_LOCK = (
    R15D1_DIR
    / "R15D1_RUNTIME_COMPUTATIONAL_OVERHEAD_LOCK.json"
)
EXPECTED_R15D1_LOCK_SHA = (
    "687265698d416b5cf92714af682ae1c4779dc41c6e3d5c41a515bf278fecc9d9"
)

DEFAULT_OUTPUT = (
    OUT
    / "Q1_R15E0_final_experiment_evidence_synthesis_fix1_v1"
)

DECISION = (
    "FINAL_EXPERIMENT_EVIDENCE_SYNTHESIZED_READY_FOR_MANUSCRIPT"
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


def assert_lock(path: Path, expected_sha: str, label: str):
    if not path.is_file():
        raise FileNotFoundError(path)

    got = sha256_file(path)
    print(label, got, "PASS" if got == expected_sha else "FAIL")

    if got != expected_sha:
        raise RuntimeError(
            f"{label} SHA mismatch: {got}"
        )

    lock = load_json(path)
    if lock.get("status") != "PASS":
        raise RuntimeError(
            f"{label} status is not PASS."
        )

    return lock


def require(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def fmt_ci(mean, lo, hi, nd=4):
    return f"{mean:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


def main():
    print("===== Q1 R15E0 FINAL EXPERIMENT EVIDENCE SYNTHESIS =====")
    print("STATUS=READ_ONLY_FINAL_SYNTHESIS")
    print("MODEL_INFERENCE=NO")
    print("MODEL_FITTING=NO")
    print("SCORE_RECOMPUTATION=NO")
    print("TARGET_CALIBRATION=NO")
    print("THRESHOLD_TUNING=NO")
    print("METHOD_RESELECTION=NO")

    a1 = assert_lock(
        R15A1_LOCK,
        EXPECTED_R15A1_LOCK_SHA,
        "R15A1_LOCK",
    )
    b1 = assert_lock(
        R15B1_LOCK,
        EXPECTED_R15B1_LOCK_SHA,
        "R15B1_LOCK",
    )
    c1 = assert_lock(
        R15C1_LOCK,
        EXPECTED_R15C1_LOCK_SHA,
        "R15C1_LOCK",
    )
    d1 = assert_lock(
        R15D1_LOCK,
        EXPECTED_R15D1_LOCK_SHA,
        "R15D1_LOCK",
    )

    if a1.get("final_method_reselected", True):
        raise RuntimeError("R15A1 reports method reselection.")
    if b1.get("information_boundary", {}).get(
        "best_coverage_selected_or_deployed",
        True,
    ):
        raise RuntimeError("R15B1 reports best-coverage deployment.")
    if c1.get("information_boundary", {}).get(
        "method_reselection",
        True,
    ):
        raise RuntimeError("R15C1 reports method reselection.")
    if d1.get("information_boundary", {}).get(
        "method_change",
        True,
    ):
        raise RuntimeError("R15D1 reports method change.")

    # ------------------------------------------------------------------
    # R15A
    # ------------------------------------------------------------------
    a_point = pd.read_csv(
        require(
            R15A1_DIR
            / "R15A1_point_estimates_macro.csv"
        ),
        low_memory=False,
    )

    a_delta = pd.read_csv(
        require(
            R15A1_DIR
            / "R15A1_paired_macro_component_delta_ci.csv"
        ),
        low_memory=False,
    )

    a_point = a_point[
        [
            "method",
            "AUROC",
            "AUPRC",
            "Recall",
            "FPR",
            "PPV1pct",
            "OracleR90FPR",
        ]
    ].copy()

    critical_comps = [
        "COND_vs_CLS",
        "COND_vs_FG",
        "COND_vs_BG",
        "FINAL_vs_M2",
        "FINAL_vs_M2_FG",
        "FINAL_vs_M2_BG",
        "FINAL_vs_COND",
    ]

    a_delta = a_delta[
        a_delta["comparison"].isin(critical_comps)
        & a_delta["metric"].isin(
            ["AUROC", "AUPRC", "FPR", "OracleR90FPR"]
        )
    ].copy()

    # ------------------------------------------------------------------
    # R15B
    # ------------------------------------------------------------------
    b_fixed = pd.read_csv(
        require(
            R15B1_DIR
            / "R15B1_frozen_threshold_policy_point.csv"
        ),
        low_memory=False,
    )

    b_fixed_delta_ci = pd.read_csv(
        require(
            R15B1_DIR
            / "R15B1_frozen_threshold_vs_random_cluster_bootstrap_delta_ci.csv"
        ),
        low_memory=False,
    )

    key_b_metrics = [
        "harm_rate_among_adapted",
        "prevented_harm_fraction",
        "mean_deployed_dice",
    ]

    b_fixed_delta_ci = b_fixed_delta_ci[
        b_fixed_delta_ci["metric"].isin(key_b_metrics)
    ].copy()

    # ------------------------------------------------------------------
    # R15C
    # ------------------------------------------------------------------
    c_point = pd.read_csv(
        require(
            R15C1_DIR
            / "R15C1_continuous_risk_point_metrics.csv"
        ),
        low_memory=False,
    )

    c_ci = pd.read_csv(
        require(
            R15C1_DIR
            / "R15C1_continuous_risk_cluster_bootstrap_ci.csv"
        ),
        low_memory=False,
    )

    c_key_metrics = [
        "macro_family_spearman_score_vs_negative_delta",
        "Q5_minus_Q1_delta_dice_mean",
        "Q5_minus_Q1_harm_prevalence",
    ]
    c_ci = c_ci[
        c_ci["metric"].isin(c_key_metrics)
    ].copy()

    # ------------------------------------------------------------------
    # R15D
    # ------------------------------------------------------------------
    d_total = pd.read_csv(
        require(
            R15D1_DIR
            / "R15D1_total_safety_overhead.csv"
        ),
        low_memory=False,
    )

    d_downstream = pd.read_csv(
        require(
            R15D1_DIR
            / "R15D1_downstream_safety_runtime.csv"
        ),
        low_memory=False,
    )

    d_complexity = load_json(
        require(
            R15D1_DIR
            / "R15D1_model_complexity_and_artifact_size.json"
        )
    )

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    if DEFAULT_OUTPUT.exists():
        raise FileExistsError(DEFAULT_OUTPUT)
    DEFAULT_OUTPUT.mkdir(
        parents=True,
        exist_ok=False,
    )

    p_a = DEFAULT_OUTPUT / "Table_R15A_core_ablation.csv"
    p_ai = DEFAULT_OUTPUT / "Table_R15A_paired_component_inference.csv"
    p_b = DEFAULT_OUTPUT / "Table_R15B_selective_utility.csv"
    p_c = DEFAULT_OUTPUT / "Table_R15C_continuous_severity.csv"
    p_d = DEFAULT_OUTPUT / "Table_R15D_runtime_overhead.csv"

    a_point.to_csv(p_a, index=False)
    a_delta.to_csv(p_ai, index=False)

    # One row per setting with fixed-threshold point + bootstrap-vs-random.
    b_rows = []
    for setting in [
        "PolypGen_TENT1",
        "SUN_TENT1",
        "SUN_PL_CONF90",
    ]:
        p = b_fixed[
            b_fixed["setting"] == setting
        ].iloc[0]

        row = {
            "setting": setting,
            "adaptation_coverage": float(
                p["adaptation_coverage"]
            ),
            "harm_rate_among_adapted": float(
                p["harm_rate_among_adapted"]
            ),
            "prevented_harm_fraction": float(
                p["prevented_harm_fraction"]
            ),
            "withheld_benefit_fraction": float(
                p["withheld_benefit_fraction"]
            ),
            "mean_deployed_dice": float(
                p["mean_deployed_dice"]
            ),
            "mean_delta_vs_source": float(
                p["mean_delta_vs_source"]
            ),
            "mean_delta_vs_adapt_all": float(
                p["mean_delta_vs_adapt_all"]
            ),
        }

        for metric in key_b_metrics:
            g = b_fixed_delta_ci[
                (b_fixed_delta_ci["setting"] == setting)
                & (b_fixed_delta_ci["metric"] == metric)
            ].iloc[0]
            row[f"{metric}_vs_random_mean_delta"] = float(
                g["bootstrap_mean_delta"]
            )
            row[f"{metric}_vs_random_ci95_low"] = float(
                g["ci95_low"]
            )
            row[f"{metric}_vs_random_ci95_high"] = float(
                g["ci95_high"]
            )

        b_rows.append(row)

    b_table = pd.DataFrame(b_rows)
    b_table.to_csv(p_b, index=False)

    c_rows = []
    for setting in [
        "PolypGen_TENT1",
        "SUN_TENT1",
        "SUN_PL_CONF90",
    ]:
        p = c_point[
            c_point["setting"] == setting
        ].iloc[0]

        row = {
            "setting": setting,
            "macro_family_spearman_point": float(
                p[
                    "macro_family_spearman_score_vs_negative_delta"
                ]
            ),
            "pooled_spearman_point": float(
                p[
                    "pooled_spearman_score_vs_negative_delta"
                ]
            ),
        }

        for metric in c_key_metrics:
            g = c_ci[
                (c_ci["setting"] == setting)
                & (c_ci["metric"] == metric)
            ].iloc[0]
            row[f"{metric}_bootstrap_mean"] = float(
                g["bootstrap_mean"]
            )
            row[f"{metric}_ci95_low"] = float(
                g["ci95_low"]
            )
            row[f"{metric}_ci95_high"] = float(
                g["ci95_high"]
            )

        c_rows.append(row)

    c_table = pd.DataFrame(c_rows)
    c_table.to_csv(p_c, index=False)

    d_table = d_total.copy()
    d_table["dino_parameters"] = int(
        d_complexity["dino_parameters"]
    )
    d_table[
        "non_dino_safety_artifact_MiB_pca_plus_head"
    ] = float(
        d_complexity[
            "non_dino_safety_artifact_MiB_pca_plus_head"
        ]
    )
    d_table.to_csv(p_d, index=False)

    # ------------------------------------------------------------------
    # Paper-ready markdown
    # ------------------------------------------------------------------
    md = []
    md.append("# SafeTTA Final Experiment Evidence Synthesis")
    md.append("")
    md.append("## 1. Core representation ablation")
    md.append("")
    md.append(a_point.to_markdown(index=False))
    md.append("")
    md.append(
        "**Frozen interpretation:** prediction-conditioned semantics "
        "are strongly better than image-only CLS; foreground-conditioned "
        "semantics carry the dominant ranking signal. FG+BG is not "
        "significantly better than FG-only in AUROC/AUPRC. Morphology "
        "does not significantly improve CondDINO ranking, although the "
        "frozen final method retains a small FPR advantage."
    )
    md.append("")

    md.append("## 2. Selective-adaptation utility at the frozen threshold")
    md.append("")
    md.append(b_table.to_markdown(index=False))
    md.append("")
    md.append(
        "**Frozen interpretation:** the safety ranking reduces HARM burden "
        "relative to random selection at matched coverage in all three "
        "external settings. Mean deployed Dice improves over random "
        "selection for PolypGen/TENT1 and SUN/TENT1, but not for SUN/PL, "
        "showing action-dependent downstream utility."
    )
    md.append("")

    md.append("## 3. Threshold-free continuous severity")
    md.append("")
    md.append(c_table.to_markdown(index=False))
    md.append("")
    md.append(
        "**Frozen interpretation:** continuous severity tracking is "
        "supported for TENT1 on PolypGen and SUN, but not for PL. "
        "Under PL, the score retains binary HARM stratification without "
        "a stable continuous DeltaDice severity gradient."
    )
    md.append("")

    md.append("## 4. Runtime and computational overhead")
    md.append("")
    md.append(d_table.to_markdown(index=False))
    md.append("")
    md.append(
        f"DINOv2-base parameters: "
        f"{int(d_complexity['dino_parameters']):,}. "
        f"Approximate FP32 parameter memory: "
        f"{d_complexity['approx_fp32_parameter_memory_MiB']:.2f} MiB. "
        f"Non-DINO PCA+head serialized artifacts: "
        f"{d_complexity['non_dino_safety_artifact_MiB_pca_plus_head']:.3f} MiB."
    )
    md.append("")
    md.append(
        "**Runtime interpretation:** on the RTX 4060 Laptop GPU, the "
        "incremental safety module requires about 20–24 ms/image depending "
        "on DINO batch size, while the post-DINO safety computation itself "
        "is only about 1.37 ms/model-frame. The dominant cost is the frozen "
        "DINO semantic encoder."
    )
    md.append("")

    md.append("## 5. Final experiment-level conclusion")
    md.append("")
    md.append(
        "> The frozen pre-adaptation safety score generalizes as a harm-ranking "
        "signal across domain and adaptation-action shifts. Prediction-conditioned "
        "foreground semantics provide the dominant representation signal. "
        "The ranking can reduce HARM under selective adaptation, but both "
        "continuous severity mapping and downstream utility remain action-dependent. "
        "Reliable operating-point transfer remains unresolved."
    )
    md.append("")

    md_path = (
        DEFAULT_OUTPUT
        / "R15E0_PAPER_READY_RESULTS.md"
    )
    md_path.write_text(
        "\n".join(md),
        encoding="utf-8",
    )

    artifacts = {}
    for p in [
        p_a,
        p_ai,
        p_b,
        p_c,
        p_d,
        md_path,
    ]:
        artifacts[p.name] = {
            "sha256": sha256_file(p),
            "bytes": int(p.stat().st_size),
        }

    lock = {
        "status": "PASS",
        "decision": DECISION,
        "script_version": VERSION,
        "build": BUILD,
        "scientific_role": (
            "read-only final experiment evidence synthesis"
        ),
        "upstream_locks": {
            "R15A1": EXPECTED_R15A1_LOCK_SHA,
            "R15B1": EXPECTED_R15B1_LOCK_SHA,
            "R15C1": EXPECTED_R15C1_LOCK_SHA,
            "R15D1": EXPECTED_R15D1_LOCK_SHA,
        },
        "information_boundary": {
            "model_inference": False,
            "model_fitting": False,
            "score_recomputation": False,
            "target_calibration": False,
            "threshold_tuning": False,
            "method_reselection": False,
        },
        "artifacts": artifacts,
        "next_stage": (
            "WRITE_MANUSCRIPT_RESULTS_METHODS_AND_FINAL_TABLES"
        ),
    }

    lock_path = (
        DEFAULT_OUTPUT
        / "R15E0_FINAL_EXPERIMENT_EVIDENCE_LOCK.json"
    )
    lock_path.write_text(
        json.dumps(
            lock,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n===== R15E0 FINAL =====")
    print("Decision=", DECISION)
    print("MARKDOWN=", md_path)
    print("LOCK=", lock_path)
    print("LOCK SHA256=", sha256_file(lock_path))
    print("PASS")


if __name__ == "__main__":
    main()
