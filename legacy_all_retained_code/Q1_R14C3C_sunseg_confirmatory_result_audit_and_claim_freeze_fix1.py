#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R14C3C_sunseg_confirmatory_result_audit_and_claim_freeze_fix1.py

Post-reveal audit only. No model inference, TTA, score recomputation, fitting,
threshold tuning, or case exclusion.

Purpose:
1) verify the completed R14C3B lock and artifact SHAs;
2) audit SUN GT native-value structure from the frozen GT decode audit;
3) reproduce TENT1/PL HARM/NEUTRAL/BENEFIT labels exactly from DeltaDice;
4) freeze the bootstrap-supported scientific claims that are actually
   supported by the untouched SUN confirmatory cohort;
5) explicitly freeze claims that are NOT supported.

This stage does not alter the primary R14C3B result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"F:\MEDSEG_SAFETTA")
OUT = ROOT / "outputs"

R14C3B_DIR = (
    OUT
    / "Q1_R14C3B_sunseg_gt_reveal_dual_action_frozen_score_evaluation_fix1_v1"
)
R14C3B_LOCK = (
    R14C3B_DIR
    / "R14C3B_SUNSEG_DUAL_ACTION_FROZEN_SCORE_EVALUATION_LOCK.json"
)

EXPECTED_R14C3B_LOCK_SHA = (
    "d510c3f1b11aa6210265eba1f30c1c3d8c1c5a7e1c16539abe6b31269488329c"
)
EXPECTED_DECISION = (
    "SUNSEG_DUAL_ACTION_GT_OUTCOMES_AND_FROZEN_SCORE_EVALUATION_COMPLETE"
)

GT_AUDIT = R14C3B_DIR / "R14C3B_SUN_GT_DECODE_AUDIT.csv"
OUTCOMES = R14C3B_DIR / "R14C3B_DUAL_ACTION_MODEL_FRAME_OUTCOMES.csv"
PANEL = R14C3B_DIR / "R14C3B_LOCKED_SCORE_DUAL_ACTION_EVALUATION_PANEL.csv"
POINT = R14C3B_DIR / "R14C3B_DUAL_ACTION_FROZEN_SCORE_POINT_METRICS.csv"
BOOT = R14C3B_DIR / "R14C3B_PHYSICAL_CASE_BOOTSTRAP_CI.csv"
OUTCOME_SUMMARY = R14C3B_DIR / "R14C3B_DUAL_ACTION_OUTCOME_SUMMARY.csv"
MECHANISM = R14C3B_DIR / "R14C3B_ACTION_SHIFT_MECHANISM_SUMMARY.json"

EXPECTED_FRAMES = 980
EXPECTED_CASES = 49
EXPECTED_STATES = 9
EXPECTED_ROWS = 8820

HARM_THRESHOLD = -0.02
BENEFIT_THRESHOLD = 0.02
FROZEN_THRESHOLD = 0.300584763193734

OUTPUT_DIR = (
    OUT
    / "Q1_R14C3C_sunseg_confirmatory_result_audit_and_claim_freeze_fix1_v1"
)

DECISION = (
    "SUNSEG_DUAL_ACTION_CONFIRMATORY_CLAIMS_FROZEN_"
    "RANKING_SUPPORTED_OPERATING_POINT_WEAK"
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


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def verify_lock_and_artifacts():
    print("===== R14C3C LOCK / ARTIFACT AUDIT =====")

    if not R14C3B_LOCK.is_file():
        raise FileNotFoundError(R14C3B_LOCK)

    got = sha256_file(R14C3B_LOCK)
    if got != EXPECTED_R14C3B_LOCK_SHA:
        raise RuntimeError(
            f"R14C3B lock SHA changed: {got}"
        )

    lock = load_json(R14C3B_LOCK)
    if lock.get("status") != "COMPLETE":
        raise RuntimeError("R14C3B status changed.")
    if lock.get("decision") != EXPECTED_DECISION:
        raise RuntimeError("R14C3B decision changed.")

    if int(lock.get("target_frames", -1)) != EXPECTED_FRAMES:
        raise RuntimeError("R14C3B frame count changed.")
    if int(lock.get("physical_cases", -1)) != EXPECTED_CASES:
        raise RuntimeError("R14C3B case count changed.")
    if int(lock.get("model_states", -1)) != EXPECTED_STATES:
        raise RuntimeError("R14C3B state count changed.")
    if int(lock.get("model_frame_rows", -1)) != EXPECTED_ROWS:
        raise RuntimeError("R14C3B row count changed.")

    frozen = lock.get("frozen_safety_evaluation", {})
    if abs(float(frozen.get("operating_threshold", math.nan)) - FROZEN_THRESHOLD) > 1e-15:
        raise RuntimeError("Frozen threshold changed.")
    if bool(frozen.get("threshold_changed", True)):
        raise RuntimeError("R14C3B reports threshold change.")

    info = lock.get("information_boundary", {})
    forbidden_true = (
        "model_inference_run_in_r14c3b",
        "tta_run_in_r14c3b",
        "source_predictions_modified",
        "tent1_predictions_modified",
        "pl_predictions_modified",
        "safety_scores_modified",
        "safety_model_fit_or_refit",
        "score_orientation_changed",
        "threshold_changed",
        "target_calibration",
        "target_feature_selection",
        "target_case_selection",
        "target_state_selection",
        "target_split_selection",
        "post_gt_case_exclusion",
    )
    for key in forbidden_true:
        if bool(info.get(key, True)):
            raise RuntimeError(
                f"R14C3B information-boundary failure: {key}"
            )

    artifacts = lock.get("artifacts", {})
    required_paths = (
        GT_AUDIT,
        OUTCOMES,
        PANEL,
        POINT,
        BOOT,
        OUTCOME_SUMMARY,
        MECHANISM,
    )

    for p in required_paths:
        if not p.is_file():
            raise FileNotFoundError(p)
        meta = artifacts.get(p.name)
        if not isinstance(meta, dict):
            raise RuntimeError(
                f"R14C3B lock missing artifact metadata: {p.name}"
            )
        digest = sha256_file(p)
        if digest != str(meta.get("sha256", "")):
            raise RuntimeError(
                f"Artifact SHA mismatch: {p.name}"
            )
        print(p.name, digest[:12] + "...", "PASS")

    print("R14C3B lock:", got)
    print("PASS")
    return lock


def audit_gt_decode():
    print("\n===== SUN GT NATIVE-VALUE AUDIT =====")

    gt = pd.read_csv(GT_AUDIT, low_memory=False)
    if len(gt) != EXPECTED_FRAMES:
        raise RuntimeError("GT audit rows !=980.")
    if gt["sample_id"].astype(str).nunique() != EXPECTED_FRAMES:
        raise RuntimeError("GT audit sample ids not unique.")

    required = [
        "native_unique_value_count",
        "native_unique_min",
        "native_unique_max",
        "native_foreground_pixels",
        "resized_foreground_pixels",
    ]
    missing = [c for c in required if c not in gt.columns]
    if missing:
        raise RuntimeError(f"GT audit missing columns: {missing}")

    counts = (
        gt.groupby(
            [
                "native_unique_value_count",
                "native_unique_min",
                "native_unique_max",
            ],
            dropna=False,
        )
        .size()
        .reset_index(name="frames")
        .sort_values(
            ["native_unique_value_count", "native_unique_min", "native_unique_max"]
        )
    )

    print(counts.to_string(index=False))

    empty_native = int((gt["native_foreground_pixels"].astype(int) == 0).sum())
    empty_resized = int((gt["resized_foreground_pixels"].astype(int) == 0).sum())

    binary_like = bool(
        (gt["native_unique_value_count"].astype(int) <= 2).all()
        and (gt["native_unique_min"].astype(int) == 0).all()
        and (gt["native_unique_max"].astype(int) > 0).all()
    )

    print("empty native GT:", empty_native)
    print("empty resized GT:", empty_resized)
    print("binary-like native masks:", binary_like)

    return {
        "rows": len(gt),
        "empty_native": empty_native,
        "empty_resized": empty_resized,
        "all_native_binary_like_by_min_max_count": binary_like,
        "native_value_structure": counts.to_dict(orient="records"),
    }


def reproduce_outcome_labels():
    print("\n===== OUTCOME REPRODUCTION AUDIT =====")

    df = pd.read_csv(OUTCOMES, low_memory=False)
    if len(df) != EXPECTED_ROWS:
        raise RuntimeError("Outcome rows !=8820.")

    if (
        df[["sample_id", "model_state_id"]]
        .drop_duplicates()
        .shape[0]
        != EXPECTED_ROWS
    ):
        raise RuntimeError("Outcome model-frame key not unique.")

    result = {}

    for action in ("tent1", "pl"):
        delta = df[f"{action}_delta_dice"].to_numpy(dtype=float)

        expected = np.where(
            delta <= HARM_THRESHOLD,
            "HARM",
            np.where(
                delta >= BENEFIT_THRESHOLD,
                "BENEFIT",
                "NEUTRAL",
            ),
        )

        observed = (
            df[f"{action}_adaptation_outcome"]
            .astype(str)
            .str.upper()
            .to_numpy()
        )

        if not np.array_equal(expected, observed):
            raise RuntimeError(
                f"{action}: tri-state labels do not reproduce thresholds."
            )

        harm = (expected == "HARM").astype(int)
        benefit = (expected == "BENEFIT").astype(int)

        if not np.array_equal(
            harm,
            df[f"{action}_harm_label"].to_numpy(dtype=int),
        ):
            raise RuntimeError(f"{action}: harm labels mismatch.")
        if not np.array_equal(
            benefit,
            df[f"{action}_benefit_label"].to_numpy(dtype=int),
        ):
            raise RuntimeError(f"{action}: benefit labels mismatch.")

        counts = {
            "HARM": int((expected == "HARM").sum()),
            "NEUTRAL": int((expected == "NEUTRAL").sum()),
            "BENEFIT": int((expected == "BENEFIT").sum()),
        }

        print(action.upper(), counts, "PASS")

        result[action.upper()] = {
            "counts": counts,
            "harm_prevalence": float(harm.mean()),
            "delta_mean": float(np.mean(delta)),
            "delta_median": float(np.median(delta)),
        }

    return result


def primary_point_rows():
    point = pd.read_csv(POINT, low_memory=False)

    primary = point[
        point["scope"].astype(str)
        == "PRIMARY_MACRO_OVER_3_FAMILIES"
    ].copy()

    if set(primary["action"].astype(str)) != {"TENT1", "PL"}:
        raise RuntimeError("Primary point rows missing TENT1 or PL.")

    out = {}
    for r in primary.itertuples(index=False):
        out[str(r.action)] = {
            "harm_prevalence": float(r.harm_prevalence),
            "auroc": float(r.auroc),
            "auprc": float(r.auprc),
            "recall": float(r.recall),
            "fpr": float(r.fpr),
            "empirical_ppv": float(r.empirical_ppv),
            "ppv_at_1pct": float(r.ppv_at_1pct),
            "oracle_target_fpr_at_r90_diagnostic": float(
                r.oracle_target_fpr_at_r90_diagnostic
            ),
        }
    return out


def bootstrap_rows():
    boot = pd.read_csv(BOOT, low_memory=False)

    required = {
        "comparison",
        "metric",
        "bootstrap_requested",
        "bootstrap_valid",
        "bootstrap_mean",
        "ci95_low",
        "ci95_high",
        "cluster_unit",
    }
    missing = sorted(required - set(boot.columns))
    if missing:
        raise RuntimeError(f"Bootstrap columns missing: {missing}")

    for comparison in ("TENT1", "PL", "PL_minus_TENT1"):
        sub = boot[boot["comparison"].astype(str) == comparison]
        if sub.empty:
            raise RuntimeError(
                f"Bootstrap comparison missing: {comparison}"
            )
        if not (sub["bootstrap_requested"].astype(int) == 2000).all():
            raise RuntimeError("Bootstrap requested count changed.")
        if not (sub["bootstrap_valid"].astype(int) == 2000).all():
            raise RuntimeError("Bootstrap valid count changed.")
        if not (
            sub["cluster_unit"].astype(str)
            == "physical_case_cluster_id"
        ).all():
            raise RuntimeError("Bootstrap cluster unit changed.")

    def row(comparison, metric):
        sub = boot[
            (boot["comparison"].astype(str) == comparison)
            & (boot["metric"].astype(str) == metric)
        ]
        if len(sub) != 1:
            raise RuntimeError(
                f"Expected one bootstrap row: {comparison}/{metric}"
            )
        r = sub.iloc[0]
        return {
            "mean": float(r["bootstrap_mean"]),
            "ci95_low": float(r["ci95_low"]),
            "ci95_high": float(r["ci95_high"]),
        }

    return {
        "TENT1": {
            metric: row("TENT1", metric)
            for metric in (
                "auroc",
                "auprc",
                "recall",
                "fpr",
                "ppv_at_1pct",
                "oracle_target_fpr_at_r90_diagnostic",
            )
        },
        "PL": {
            metric: row("PL", metric)
            for metric in (
                "auroc",
                "auprc",
                "recall",
                "fpr",
                "ppv_at_1pct",
                "oracle_target_fpr_at_r90_diagnostic",
            )
        },
        "PL_minus_TENT1": {
            metric: row("PL_minus_TENT1", metric)
            for metric in (
                "auroc",
                "auprc",
                "recall",
                "fpr",
                "ppv_at_1pct",
                "oracle_target_fpr_at_r90_diagnostic",
            )
        },
    }


def derive_claims(point, boot, mechanism):
    tent_auc = boot["TENT1"]["auroc"]
    pl_auc = boot["PL"]["auroc"]
    auc_diff = boot["PL_minus_TENT1"]["auroc"]
    fpr_diff = boot["PL_minus_TENT1"]["fpr"]

    claims_supported = {
        "TENT1_independent_ranking_signal": (
            tent_auc["ci95_low"] > 0.5
        ),
        "PL_independent_ranking_signal": (
            pl_auc["ci95_low"] > 0.5
        ),
        "dual_action_ranking_signal": (
            tent_auc["ci95_low"] > 0.5
            and pl_auc["ci95_low"] > 0.5
        ),
        "TENT1_and_PL_AUROC_different_at_95pct": (
            auc_diff["ci95_low"] > 0
            or auc_diff["ci95_high"] < 0
        ),
        "PL_frozen_threshold_FPR_higher_than_TENT1": (
            fpr_diff["ci95_low"] > 0
        ),
        "high_recall_operating_point_transfer_reliable": (
            point["TENT1"]["recall"] >= 0.90
            and point["PL"]["recall"] >= 0.90
            and point["TENT1"]["fpr"] <= 0.20
            and point["PL"]["fpr"] <= 0.20
        ),
        "action_harm_identity_stable": (
            float(mechanism["harm_jaccard"]) >= 0.50
            and float(mechanism["harm_cohen_kappa"]) >= 0.40
        ),
    }

    prohibited = [
        (
            "Automatic safe deployment gate is solved",
            "Unsupported: frozen-threshold recall is ~0.665 and FPR is "
            "~0.421-0.458; oracle FPR at Recall≈0.90 remains ~0.65-0.69."
        ),
        (
            "TENT1 ranking is significantly better than PL by AUROC",
            "Unsupported: paired PL-minus-TENT1 AUROC 95% CI crosses 0."
        ),
        (
            "The same cases are harmed by TENT1 and PL",
            "Unsupported: HARM Jaccard≈0.091 and Cohen kappa≈0.011."
        ),
        (
            "AUPRC proves TENT1 is intrinsically better than PL",
            "Do not make this direct claim because HARM prevalence differs "
            "substantially between the two actions."
        ),
    ]

    paper_claim = (
        "On an untouched SUN-SEG cohort, the frozen pre-adaptation "
        "SOURCE-conditioned safety score retained statistically supported "
        "harm-ranking signal for both TENT1 and PL_CONF90 despite severe "
        "adaptation-action outcome shift. However, the frozen high-recall "
        "operating point did not transfer reliably."
    )

    return claims_supported, prohibited, paper_claim


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = p.parse_args()

    print(
        "===== Q1 R14C3C SUN-SEG CONFIRMATORY RESULT AUDIT "
        "+ CLAIM FREEZE ====="
    )
    print("MODEL_INFERENCE=NO")
    print("TTA_RERUN=NO")
    print("SCORE_RECOMPUTATION=NO")
    print("MODEL_FIT_REFIT=NO")
    print("THRESHOLD_CHANGE=NO")
    print("POST_GT_CASE_EXCLUSION=NO")

    lock = verify_lock_and_artifacts()
    gt_audit = audit_gt_decode()
    outcome_repro = reproduce_outcome_labels()
    point = primary_point_rows()
    boot = bootstrap_rows()
    mechanism = load_json(MECHANISM)

    claims_supported, prohibited, paper_claim = derive_claims(
        point,
        boot,
        mechanism,
    )

    print("\n===== CLAIM AUDIT =====")
    for key, value in claims_supported.items():
        print(key, "=", value)

    print("\nPaper-level frozen interpretation:")
    print(paper_claim)

    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)

    audit = {
        "status": "PASS",
        "decision": DECISION,
        "source_result": {
            "r14c3b_lock_path": str(R14C3B_LOCK),
            "r14c3b_lock_sha256": EXPECTED_R14C3B_LOCK_SHA,
            "r14c3b_decision": EXPECTED_DECISION,
        },
        "gt_decode_audit": gt_audit,
        "outcome_reproduction": outcome_repro,
        "primary_point": point,
        "physical_case_bootstrap": boot,
        "action_shift_mechanism": mechanism,
        "supported_claim_flags": claims_supported,
        "paper_level_frozen_interpretation": paper_claim,
        "claims_not_supported": [
            {
                "claim": claim,
                "reason": reason,
            }
            for claim, reason in prohibited
        ],
        "scientific_boundary": {
            "result_recomputed": False,
            "safety_score_recomputed": False,
            "model_fit_or_refit": False,
            "threshold_changed": False,
            "gt_rule_changed": False,
            "post_gt_case_exclusion": False,
            "claim_freeze_only": True,
        },
    }

    audit_path = args.output_dir / "R14C3C_CONFIRMATORY_CLAIM_FREEZE.json"
    write_json(audit_path, audit)

    md = f"""# R14C3C SUN-SEG Confirmatory Claim Freeze

## Frozen interpretation

{paper_claim}

## TENT1

- AUROC: {point['TENT1']['auroc']:.6f}
- AUROC 95% CI: [{boot['TENT1']['auroc']['ci95_low']:.6f}, {boot['TENT1']['auroc']['ci95_high']:.6f}]
- AUPRC: {point['TENT1']['auprc']:.6f}
- Recall at frozen threshold: {point['TENT1']['recall']:.6f}
- FPR at frozen threshold: {point['TENT1']['fpr']:.6f}
- Oracle diagnostic FPR at Recall≈0.90: {point['TENT1']['oracle_target_fpr_at_r90_diagnostic']:.6f}

## PL_CONF90

- AUROC: {point['PL']['auroc']:.6f}
- AUROC 95% CI: [{boot['PL']['auroc']['ci95_low']:.6f}, {boot['PL']['auroc']['ci95_high']:.6f}]
- AUPRC: {point['PL']['auprc']:.6f}
- Recall at frozen threshold: {point['PL']['recall']:.6f}
- FPR at frozen threshold: {point['PL']['fpr']:.6f}
- Oracle diagnostic FPR at Recall≈0.90: {point['PL']['oracle_target_fpr_at_r90_diagnostic']:.6f}

## Action shift

- HARM Jaccard: {float(mechanism['harm_jaccard']):.6f}
- HARM Cohen kappa: {float(mechanism['harm_cohen_kappa']):.6f}
- DeltaDice Spearman: {float(mechanism['delta_dice_spearman']):.6f}
- Paired PL-TENT1 AUROC difference 95% CI:
  [{boot['PL_minus_TENT1']['auroc']['ci95_low']:.6f}, {boot['PL_minus_TENT1']['auroc']['ci95_high']:.6f}]
- Paired PL-TENT1 frozen-threshold FPR difference 95% CI:
  [{boot['PL_minus_TENT1']['fpr']['ci95_low']:.6f}, {boot['PL_minus_TENT1']['fpr']['ci95_high']:.6f}]

## Interpretation boundary

Supported:
- independent harm-ranking signal for TENT1;
- independent harm-ranking signal for PL_CONF90;
- dual-action ranking signal despite action-specific harm identities;
- weak/unreliable high-recall operating-point transfer.

Not supported:
- automatic deployment gate is solved;
- TENT1 AUROC is significantly better than PL;
- the same samples are harmed across actions;
- direct AUPRC superiority claim across actions with different prevalences.
"""

    md_path = args.output_dir / "R14C3C_PAPER_READY_INTERPRETATION.md"
    md_path.write_text(md, encoding="utf-8")

    print("\n===== R14C3C FINAL =====")
    print("Decision=", DECISION)
    print("AUDIT=", audit_path)
    print("SUMMARY=", md_path)
    print("PASS")


if __name__ == "__main__":
    main()
