#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q1_SAFETTA_repro_chain_resolver_v2.py

Read-only resolver for the exact paper-relevant SafeTTA reproducibility chain.

This script fixes the main limitation of repro_inventory_v1: v1 intentionally
filtered candidate code using a narrow token list, so several paper-critical
colonoscopy scripts (R10L*, R14C1-3, R15A-C, etc.) were not counted as
candidate_final_scripts even though their output locks were present.

v2 scans ALL Python files under F:\\MEDSEG_SAFETTA\\code, resolves the
manuscript-facing stages below, checks required output locks/artifacts, hashes
selected scripts and required artifacts, and emits an explicit PASS/BLOCKED
reproducibility gate.

It does NOT execute any experiment script, does NOT train, does NOT access GT,
and does NOT modify existing outputs/checkpoints.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Dict

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

VERSION = "2026-09-06-Q1-SAFETTA-REPRO-CHAIN-RESOLVER-v2"
DEFAULT_ROOT = Path(r"F:\MEDSEG_SAFETTA")


@dataclass
class Stage:
    order: int
    branch: str
    stage_id: str
    role: str
    script: str
    output_dir: str
    required_globs: List[str]
    required_for_exact_repro: bool = True


# This mapping is paper-facing, not a history dump.
# Selected fixes are the retained final/frozen variants indicated by the
# manuscript, output locks, and the implementation audit.
STAGES: List[Stage] = [
    # ---------------- Colonoscopy: SOURCE representation / estimator ----------------
    Stage(10, "colonoscopy", "R10K2A", "DINOv2 + SOURCE-mask representation lock",
          "Q1_R10K2A_dinov2_image_mask_representation_lock_fix1.py",
          "Q1_R10K2A_dinov2_image_mask_representation_lock_fix1_v1",
          ["R10K2A_REPRESENTATION_LOCK.json", "R10K2A_mask_conditioned_dinov2_features.npz"]),
    Stage(11, "colonoscopy", "R10K2B", "Grouped SOURCE LOMO representation evaluation",
          "Q1_R10K2B_dinov2_image_mask_LOMO_evaluation_fix2.py",
          "Q1_R10K2B_dinov2_image_mask_LOMO_evaluation_fix2_v1",
          ["R10K2B_by_family_summary.csv", "R10K2B_macro_summary.csv"]),
    Stage(12, "colonoscopy", "R10K2C", "Case-clustered paired bootstrap for representation",
          "Q1_R10K2C_case_clustered_paired_bootstrap_fix1.py",
          "Q1_R10K2C_case_clustered_paired_bootstrap_fix1_v1",
          ["R10K2C_summary.json"]),
    Stage(13, "colonoscopy", "R10K2D", "Confirmed method freeze",
          "Q1_R10K2D_confirmed_method_freeze_fix1.py",
          "Q1_R10K2D_confirmed_method_freeze_fix1_v1",
          ["R10K2D_CONFIRMED_METHOD_LOCK.json"]),
    Stage(14, "colonoscopy", "R10L0", "Final SOURCE safety estimator + threshold lock",
          "Q1_R10L0_final_source_safety_estimator_lock_fix3.py",
          "Q1_R10L0_final_source_safety_estimator_lock_fix3_v1",
          ["R10L0_FINAL_SOURCE_ESTIMATOR_LOCK.json", "*.joblib"]),

    # ---------------- PolypGen external lock-before-GT chain ----------------
    Stage(20, "polypgen", "R10L1", "Official single-frame asset audit",
          "Q1_R10L1_polypgen_official_singleframe_asset_audit_fix2.py",
          "Q1_R10L1_polypgen_official_singleframe_asset_audit_fix2_v1",
          ["R10L1_polypgen_official_singleframe_manifest.csv", "R10L1_summary.json"]),
    Stage(21, "polypgen", "R10L1A", "External dataset identity audit",
          "Q1_R10L1A_external_dataset_identity_audit_fix1.py",
          "Q1_R10L1A_external_dataset_identity_audit_fix1_v1",
          ["R10L1A_summary.json"]),
    Stage(22, "polypgen", "R10L1B", "Frozen unique-static PolypGen manifest",
          "Q1_R10L1B_polypgen_unique_static_manifest_lock_fix3.py",
          "Q1_R10L1B_polypgen_unique_static_manifest_lock_fix3_v1",
          ["R10L1B_MANIFEST_LOCK.json", "R10L1B_POLYPGEN_UNIQUE_STATIC_MANIFEST.csv"]),
    Stage(23, "polypgen", "R10L2B", "SOURCE-training overlap/independence lock",
          "Q1_R10L2B_authoritative_s01_source_training_overlap_lock_fix1.py",
          "Q1_R10L2B_authoritative_s01_source_training_overlap_lock_fix1_v1",
          ["R10L2B_INDEPENDENCE_GATE_LOCK.json", "R10L2B_per_state_independence_audit.csv"]),
    Stage(24, "polypgen", "R10L3A", "GT-free SOURCE + TENT1 prediction lock",
          "Q1_R10L3A_polypgen_source_a1_prediction_lock_fix2.py",
          "Q1_R10L3A_polypgen_source_a1_prediction_lock_fix2_v1",
          ["R10L3A_POLYPGEN_SOURCE_A1_PREDICTION_LOCK.json", "model_case_prediction_manifest.csv"]),
    Stage(25, "polypgen", "R10L3B", "Frozen safety score lock before GT",
          "Q1_R10L3B_polypgen_frozen_safety_score_lock_fix1.py",
          "Q1_R10L3B_polypgen_frozen_safety_score_lock_fix1_v1",
          ["R10L3B_POLYPGEN_FROZEN_SAFETY_SCORE_LOCK.json"]),
    Stage(26, "polypgen", "R10L3C", "GT-reveal external evaluation",
          "Q1_R10L3C_polypgen_gt_reveal_prospective_external_evaluation_fix1.py",
          "Q1_R10L3C_polypgen_gt_reveal_prospective_external_evaluation_fix1_v1",
          ["R10L3C_LOCKED_SCORE_GT_EVALUATION_PANEL.csv", "R10L3C_POLYPGEN_PROSPECTIVE_EXTERNAL_EVALUATION_LOCK.json"]),
    Stage(27, "polypgen", "R10L3D", "External generalization claim freeze",
          "Q1_R10L3D_polypgen_external_generalization_diagnostic_freeze_fix1.py",
          "Q1_R10L3D_polypgen_external_generalization_diagnostic_freeze_fix1_v1",
          ["R10L3D_EXTERNAL_GENERALIZATION_CLAIM_LOCK.json"]),

    # ---------------- SUN-SEG action-shift chain ----------------
    Stage(30, "sunseg", "R14B3", "Untouched unseen case-balanced candidate manifest",
          "Q1_R14B3_sunseg_unseen_case_balanced_candidate_lock_fix1.py",
          "Q1_R14B3_sunseg_unseen_case_balanced_candidate_lock_fix1_v1",
          ["R14B3_SUNSEG_PRECONTAMINATION_CANDIDATE_MANIFEST.csv", "R14B3_SUNSEG_PRECONTAMINATION_CANDIDATE_LOCK.json"]),
    Stage(31, "sunseg", "R14B4B", "Final contamination audit + confirmatory manifest",
          "Q1_R14B4B_sunseg_selected_rgb_contamination_audit_fix3.py",
          "Q1_R14B4B_sunseg_selected_rgb_contamination_audit_fix3_v1",
          ["R14B4B_FIX3_FINAL_CONFIRMATORY_MANIFEST.csv", "R14B4B_FIX3_SUNSEG_CONTAMINATION_AUDIT_LOCK.json"]),
    Stage(32, "sunseg", "R14C0", "Pre-GT runtime/lineage preflight",
          "Q1_R14C0_sunseg_preGT_prediction_runtime_lineage_preflight_fix4.py",
          "Q1_R14C0_sunseg_preGT_prediction_runtime_lineage_preflight_fix4_v1",
          ["R14C0_SUNSEG_PREGT_RUNTIME_LINEAGE_LOCK.json", "R14C0_FROZEN_9_STATE_CHECKPOINT_LINEAGE.csv"]),
    Stage(33, "sunseg", "R14C1", "GT-free SOURCE/TENT1/PL-CONF90 prediction lock",
          "Q1_R14C1_sunseg_source_tent1_pl_prediction_lock_preGT_fix3.py",
          "Q1_R14C1_sunseg_source_tent1_pl_prediction_lock_preGT_fix3_v1",
          ["R14C1_SUNSEG_SOURCE_TENT1_PL_PREDICTION_LOCK.json", "R14C1_SUNSEG_MODEL_FRAME_PREDICTION_MANIFEST.csv"]),
    Stage(34, "sunseg", "R14C2A", "Frozen safety-estimator lineage/schema preflight",
          "Q1_R14C2A_frozen_safety_estimator_lineage_schema_preflight_fix2.py",
          "Q1_R14C2A_frozen_safety_estimator_lineage_schema_preflight_fix2_v1",
          ["R14C2A_FROZEN_SAFETY_ESTIMATOR_LINEAGE_SCHEMA_AUDIT.json"]),
    Stage(35, "sunseg", "R14C2B", "Frozen safety-score lock before GT",
          "Q1_R14C2B_sunseg_frozen_safety_score_lock_preGT_fix1.py",
          "Q1_R14C2B_sunseg_frozen_safety_score_lock_preGT_fix1_v1",
          ["R14C2B_SUNSEG_FROZEN_SAFETY_SCORE_LOCK.json"]),
    Stage(36, "sunseg", "R14C3A", "GT-reveal lineage preflight",
          "Q1_R14C3A_sunseg_gt_reveal_outcome_lineage_preflight_fix1.py",
          "Q1_R14C3A_sunseg_gt_reveal_outcome_lineage_preflight_fix1_v1",
          ["R14C3A_SUNSEG_GT_REVEAL_OUTCOME_LINEAGE_AUDIT.json"]),
    Stage(37, "sunseg", "R14C3B", "Dual-action GT-reveal evaluation",
          "Q1_R14C3B_sunseg_gt_reveal_dual_action_frozen_score_evaluation_fix1.py",
          "Q1_R14C3B_sunseg_gt_reveal_dual_action_frozen_score_evaluation_fix1_v1",
          ["R14C3B_LOCKED_SCORE_DUAL_ACTION_EVALUATION_PANEL.csv", "R14C3B_ACTION_SHIFT_MECHANISM_SUMMARY.json", "R14C3B_SUNSEG_DUAL_ACTION_FROZEN_SCORE_EVALUATION_LOCK.json"]),

    # ---------------- Paper-facing colonoscopy analyses ----------------
    Stage(40, "paper_analysis", "R15A0", "Core representation ablation",
          "Q1_R15A0_prediction_conditioned_core_ablation_LOMO_fix1.py",
          "Q1_R15A0_prediction_conditioned_core_ablation_LOMO_fix1_v1",
          ["R15A0_CORE_ABLATION_LOCK.json", "R15A0_macro_summary.csv"]),
    Stage(41, "paper_analysis", "R15A1", "Core ablation paired bootstrap",
          "Q1_R15A1_core_ablation_case_clustered_paired_bootstrap_fix2.py",
          "Q1_R15A1_core_ablation_case_clustered_paired_bootstrap_fix2_v1",
          ["R15A1_CORE_ABLATION_PAIRED_BOOTSTRAP_LOCK.json", "R15A1_component_claim_audit.json"]),
    Stage(42, "paper_analysis", "R15B0", "Selective-adaptation utility input preflight",
          "Q1_R15B0_selective_adaptation_external_utility_preflight_fix1.py",
          "Q1_R15B0_selective_adaptation_external_utility_preflight_fix1_v1",
          ["R15B0_SELECTIVE_ADAPTATION_UTILITY_INPUT_LOCK.json"]),
    Stage(43, "paper_analysis", "R15B1", "Selective-adaptation risk/coverage utility",
          "Q1_R15B1_selective_adaptation_risk_coverage_utility_fix1.py",
          "Q1_R15B1_selective_adaptation_risk_coverage_utility_fix1_v1",
          ["R15B1_SELECTIVE_ADAPTATION_UTILITY_LOCK.json"]),
    Stage(44, "paper_analysis", "R15C1", "Continuous risk vs delta-Dice analysis",
          "Q1_R15C1_continuous_risk_delta_dice_external_analysis_fix1.py",
          "Q1_R15C1_continuous_risk_delta_dice_external_analysis_fix1_v1",
          ["R15C1_CONTINUOUS_RISK_EXTERNAL_ANALYSIS_LOCK.json", "R15C1_CONTINUOUS_RISK_CLAIM_AUDIT.json"]),
    Stage(45, "paper_analysis", "R15D0", "Runtime cohort/protocol lock",
          "Q1_R15D0_runtime_computational_overhead_preflight_fix2.py",
          "Q1_R15D0_runtime_computational_overhead_preflight_fix2_v1",
          ["R15D0_RUNTIME_OVERHEAD_PROTOCOL_LOCK.json", "R15D0_RUNTIME_COHORT_256.csv"]),
    Stage(46, "paper_analysis", "R15D1", "Runtime benchmark",
          "Q1_R15D1_runtime_computational_overhead_benchmark_fix1.py",
          "Q1_R15D1_runtime_computational_overhead_benchmark_fix1_v1",
          ["R15D1_RUNTIME_COMPUTATIONAL_OVERHEAD_LOCK.json", "R15D1_dino_runtime_by_batch.csv", "R15D1_total_safety_overhead.csv"]),
    Stage(47, "paper_analysis", "HARM_MARGIN", "Fixed-score HARM margin sensitivity",
          "Q1_SAFETTA_harm_margin_sensitivity_v1.py",
          "Q1_SAFETTA_harm_margin_sensitivity_v1",
          ["harm_margin_sensitivity_ranked.csv"]),

    # ---------------- MRI replication: Prostate158 -> PROMISE12 ----------------
    Stage(50, "mri", "CM0", "Dataset acquisition/integrity audit",
          "Q1X_CM0_prostate_mri_data_acquisition_integrity_audit_fix1.py",
          "Q1X_CM0_prostate_mri_data_acquisition_integrity_audit_fix1_v1",
          ["CM0_PROSTATE_MRI_DATASET_LOCK.json"]),
    Stage(51, "mri", "CM1", "MRI schema/source-label audit",
          "Q1X_CM1_prostate_mri_schema_and_source_label_audit_fix3.py",
          "Q1X_CM1_prostate_mri_schema_and_source_label_audit_fix3_v1",
          ["CM1_PROSTATE_MRI_SCHEMA_LOCK.json"]),
    Stage(52, "mri", "CM2A", "Official source manifest + target image-only preflight",
          "Q1X_CM2A_prostate158_official_manifest_t2_resolver_and_slice_preflight_fix3.py",
          "Q1X_CM2A_prostate158_official_manifest_t2_resolver_and_slice_preflight_fix3_v1",
          ["CM2A_PROSTATE_MRI_SAMPLE_UNIT_LOCK.json", "CM2A_PROSTATE158_OFFICIAL_T2_CASE_MANIFEST.csv", "CM2A_PROMISE12_IMAGE_ONLY_SLICE_MANIFEST.csv"]),
    Stage(53, "mri", "CM2B", "Preprocessing/model protocol lock",
          "Q1X_CM2B_prostate_mri_preprocessing_and_model_protocol_lock_fix1.py",
          "Q1X_CM2B_prostate_mri_preprocessing_and_model_protocol_lock_fix1_v1",
          ["CM2B_PROTOCOL_LOCK.json", "CM2B_FROZEN_PROTOCOL.json", "CM2B_SOURCE_PATIENT_5FOLD_LOCK.csv"]),
    Stage(54, "mri", "CM3", "SOURCE OOF segmentation training/prediction lock",
          "Q1X_CM3_source_oof_segmentation_training_and_prediction_lock_fix3.py",
          "Q1X_CM3_source_oof_segmentation_training_and_prediction_lock_fix3_v1",
          ["CM3_SOURCE_OOF_SEGMENTATION_PANEL_LOCK.json", "CM3_SOURCE_OOF_MODEL_PANEL_SUMMARY.csv"]),
    Stage(55, "mri", "CM4A", "SOURCE OOF TENT1 outcome lock",
          "Q1X_CM4A_source_oof_tent1_outcome_asset_lock_fix4.py",
          "Q1X_CM4A_source_oof_tent1_outcome_asset_lock_fix4_v1",
          ["CM4A_SOURCE_TENT1_OUTCOME_LOCK.json", "CM4A_SOURCE_TENT1_OUTCOME_TABLE.csv"]),
    Stage(56, "mri", "CM4B", "SOURCE prediction-conditioned safety OOF",
          "Q1X_CM4B_mri_source_prediction_conditioned_safety_grouped_oof_fix1.py",
          "Q1X_CM4B_mri_source_prediction_conditioned_safety_grouped_oof_fix1_v1",
          ["CM4B_MRI_SOURCE_SAFETY_OOF_LOCK.json", "CM4B_SOURCE_SAFETY_OOF_METRICS.csv", "CM4B_SOURCE_SAFETY_OOF_SCORES.csv"]),
    Stage(57, "mri", "CM4C", "Final SOURCE safety estimator + threshold lock",
          "Q1X_CM4C_final_source_safety_estimator_and_threshold_lock_fix2.py",
          "Q1X_CM4C_final_source_safety_estimator_and_threshold_lock_fix2_v1",
          ["CM4C_FINAL_SOURCE_SAFETY_ESTIMATOR_LOCK.json", "CM4C_SOURCE_OPERATING_POINT.json", "CM4C_FINAL_SOURCE_SAFETY_ESTIMATOR.joblib"]),
    Stage(58, "mri", "CM4D", "Final all-SOURCE segmentation models",
          "Q1X_CM4D_final_all_source_segmentation_models_lock_fix2.py",
          "Q1X_CM4D_final_all_source_segmentation_models_lock_fix2_v1",
          ["CM4D_FINAL_ALL_SOURCE_SEGMENTATION_MODELS_LOCK.json", "CM4D_FINAL_MODEL_SUMMARY.csv", "models/*/final.pt"]),
    Stage(59, "mri", "CM5A", "PROMISE12 GT-free prediction/TENT1/safety score lock",
          "Q1X_CM5A_promise12_gt_free_prediction_tent1_safety_scoring_lock_fix1.py",
          "Q1X_CM5A_promise12_gt_free_prediction_tent1_safety_scoring_lock_fix1_v1",
          ["CM5A_PROMISE12_GT_FREE_PREDICTION_SAFETY_LOCK.json", "CM5A_PROMISE12_GT_FREE_SAFETY_SCORES.csv"]),
    Stage(60, "mri", "CM5B", "Unlabeled support-aware operating-point transport",
          "Q1X_CM5B_unlabeled_support_aware_operating_point_transport_lock_fix2.py",
          "Q1X_CM5B_unlabeled_support_aware_operating_point_transport_lock_fix2_v1",
          ["CM5B_UNLABELED_OPERATING_POINT_TRANSPORT_LOCK.json", "CM5B_LOCKED_POLICY_SUMMARY.csv"]),
    Stage(61, "mri", "CM6", "PROMISE12 GT-reveal locked-policy evaluation",
          "Q1X_CM6_promise12_gt_reveal_locked_policy_evaluation_fix1.py",
          "Q1X_CM6_promise12_gt_reveal_locked_policy_evaluation_fix1_v1",
          ["CM6_PROMISE12_GT_REVEAL_LOCKED_EVALUATION_LOCK.json", "CM6_LOCKED_POLICY_EVALUATION.csv", "CM6_TARGET_HARM_RANKING.csv"]),
    Stage(62, "mri", "CM7A", "PROMISE12 external ranking baseline audit",
          "Q1X_CM7A_promise12_external_ranking_baseline_audit_fix2.py",
          "Q1X_CM7A_promise12_external_ranking_baseline_audit_fix2_v1",
          ["CM7A_PROMISE12_EXTERNAL_RANKING_BASELINE_AUDIT_LOCK.json", "CM7A_PROMISE12_BASELINE_RANKING_METRICS.csv", "CM7A_FINAL_VS_BASELINE_PAIRED_DELTA_AUROC_BOOTSTRAP.csv"]),
    Stage(63, "mri", "CM7B", "PROMISE12 matched-coverage utility audit",
          "Q1X_CM7B_promise12_matched_coverage_utility_audit_fix1.py",
          "Q1X_CM7B_promise12_matched_coverage_utility_audit_fix1_v1",
          ["CM7B_PROMISE12_MATCHED_COVERAGE_UTILITY_AUDIT_LOCK.json", "CM7B_PAIRED_PATIENT_BOOTSTRAP_SUMMARY.csv", "CM7B_POINT_UTILITY_SUMMARY.json"]),
]


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def py_syntax_ok(path: Path) -> bool:
    try:
        ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        return True
    except Exception:
        return False


def find_glob(base: Path, pattern: str) -> List[Path]:
    return sorted([p for p in base.glob(pattern) if p.is_file()])


def get_environment() -> Dict[str, str]:
    env = {
        "python_executable": sys.executable,
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
    }
    for modname in ["torch", "torchvision", "numpy", "pandas", "scipy", "sklearn", "transformers", "joblib"]:
        try:
            m = __import__(modname)
            env[modname] = getattr(m, "__version__", "UNKNOWN")
        except Exception as e:
            env[modname] = f"NOT_IMPORTABLE: {e}"
    return env


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--no-hash-large", action="store_true", help="Skip SHA256 for large required artifacts")
    args = ap.parse_args()

    root = args.root
    code_dir = root / "code"
    outputs_dir = root / "outputs"
    out = args.output_dir or (outputs_dir / "Q1_SAFETTA_repro_chain_resolver_v2")
    out.mkdir(parents=True, exist_ok=True)

    if not code_dir.exists():
        raise FileNotFoundError(code_dir)
    if not outputs_dir.exists():
        raise FileNotFoundError(outputs_dir)

    print("===== SAFETTA EXACT PAPER REPRO CHAIN RESOLVER =====")
    print("Version:", VERSION)
    print("Root:", root)
    print("Stages:", len(STAGES))
    print("Experiment execution: NONE")
    print("Read-only audit: YES")
    print()

    rows = []
    required_artifact_rows = []
    iterator = STAGES
    if tqdm is not None:
        iterator = tqdm(STAGES, desc="Resolve paper chain", unit="stage", dynamic_ncols=True)

    for s in iterator:
        script_path = code_dir / s.script
        output_path = outputs_dir / s.output_dir
        script_exists = script_path.is_file()
        syntax_ok = py_syntax_ok(script_path) if script_exists else False
        script_sha = sha256_file(script_path) if script_exists else None
        output_exists = output_path.is_dir()

        missing_patterns = []
        all_required_matches = []
        for pat in s.required_globs:
            matches = find_glob(output_path, pat) if output_exists else []
            if not matches:
                missing_patterns.append(pat)
            else:
                all_required_matches.extend(matches)

        # de-duplicate matched files across globs
        uniq = []
        seen = set()
        for p in all_required_matches:
            rp = str(p.resolve()).lower()
            if rp not in seen:
                seen.add(rp)
                uniq.append(p)

        for p in uniq:
            size = p.stat().st_size
            digest = None
            if (not args.no_hash_large) or size <= 50 * 1024 * 1024:
                digest = sha256_file(p)
            required_artifact_rows.append({
                "order": s.order,
                "branch": s.branch,
                "stage_id": s.stage_id,
                "artifact_relpath": str(p.relative_to(root)),
                "size_bytes": size,
                "sha256": digest,
            })

        status = "PASS" if (script_exists and syntax_ok and output_exists and not missing_patterns) else "BLOCKED"
        rows.append({
            "order": s.order,
            "branch": s.branch,
            "stage_id": s.stage_id,
            "role": s.role,
            "script_relpath": str(Path("code") / s.script),
            "script_exists": script_exists,
            "script_syntax_ok": syntax_ok,
            "script_sha256": script_sha,
            "output_relpath": str(Path("outputs") / s.output_dir),
            "output_exists": output_exists,
            "required_patterns": ";".join(s.required_globs),
            "missing_patterns": ";".join(missing_patterns),
            "required_artifacts_found": len(uniq),
            "required_for_exact_repro": s.required_for_exact_repro,
            "status": status,
        })

    # Duplicate code audit: same SHA among selected scripts should be visible, not hidden.
    sha_to_scripts: Dict[str, List[str]] = {}
    for r in rows:
        if r["script_sha256"]:
            sha_to_scripts.setdefault(r["script_sha256"], []).append(r["script_relpath"])
    duplicate_selected = {k: v for k, v in sha_to_scripts.items() if len(v) > 1}

    # Also scan ALL .py files so paper-relevant scripts omitted by inventory_v1 are no longer invisible.
    all_py = sorted(code_dir.glob("*.py"))
    all_code_rows = []
    for p in all_py:
        all_code_rows.append({
            "relpath": str(p.relative_to(root)),
            "size_bytes": p.stat().st_size,
            "syntax_ok": py_syntax_ok(p),
            "sha256": sha256_file(p),
        })

    # write CSVs
    chain_csv = out / "paper_repro_chain_manifest.csv"
    with chain_csv.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    art_csv = out / "paper_repro_required_artifacts.csv"
    fields = ["order","branch","stage_id","artifact_relpath","size_bytes","sha256"]
    with art_csv.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(required_artifact_rows)

    all_code_csv = out / "all_code_sha256.csv"
    with all_code_csv.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["relpath","size_bytes","syntax_ok","sha256"])
        w.writeheader(); w.writerows(all_code_rows)

    blocked = [r for r in rows if r["required_for_exact_repro"] and r["status"] != "PASS"]
    env = get_environment()

    gate = {
        "version": VERSION,
        "root": str(root),
        "selected_stage_count": len(rows),
        "selected_stage_pass": len(rows) - len(blocked),
        "selected_stage_blocked": len(blocked),
        "all_python_file_count": len(all_code_rows),
        "all_python_syntax_failures": [r["relpath"] for r in all_code_rows if not r["syntax_ok"]],
        "duplicate_selected_script_sha256": duplicate_selected,
        "environment": env,
        "gate": "PASS_CHAIN_ASSET_RESOLUTION" if not blocked else "BLOCKED_CHAIN_ASSET_RESOLUTION",
        "blocked_stages": blocked,
        "important_note": (
            "PASS here means the exact retained paper-chain scripts and required frozen artifacts are present and hashed. "
            "It does NOT yet mean paper numeric results were recomputed. Numeric reproduction is the next gate."
        ),
    }
    (out / "paper_repro_chain_gate.json").write_text(json.dumps(gate, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = []
    summary.append("===== SAFETTA EXACT PAPER REPRO CHAIN =====")
    summary.append(f"Version: {VERSION}")
    summary.append(f"Selected stages: {len(rows)}")
    summary.append(f"PASS: {len(rows)-len(blocked)}")
    summary.append(f"BLOCKED: {len(blocked)}")
    summary.append(f"All code/*.py files inventoried: {len(all_code_rows)}")
    summary.append(f"All-code syntax failures: {len(gate['all_python_syntax_failures'])}")
    summary.append(f"Gate: {gate['gate']}")
    summary.append("")
    if blocked:
        summary.append("BLOCKED STAGES:")
        for r in blocked:
            summary.append(f"  {r['stage_id']}: script_exists={r['script_exists']} output_exists={r['output_exists']} missing={r['missing_patterns']}")
    else:
        summary.append("All selected frozen paper-chain stages resolved.")
    summary.append("")
    summary.append("NEXT GATE:")
    summary.append("Build paper-value assertions from the locked outputs and rerun the analysis-only stages to reproduce Tables 2-7 / Figures 2-5 without changing frozen artifacts.")
    (out / "PAPER_REPRO_CHAIN_SUMMARY.txt").write_text("\n".join(summary), encoding="utf-8")

    print("\n===== FINAL =====")
    print("PASS stages:", len(rows)-len(blocked))
    print("BLOCKED stages:", len(blocked))
    print("All Python files inventoried:", len(all_code_rows))
    print("Gate:", gate["gate"])
    print("Output:", out)
    return 0 if not blocked else 2


if __name__ == "__main__":
    raise SystemExit(main())
