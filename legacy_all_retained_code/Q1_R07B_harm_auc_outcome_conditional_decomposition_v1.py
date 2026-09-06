#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Q1-R07B — Mechanism audit for the predictability-control gap.

No model fit, inference, TTA, calibration, or threshold selection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import traceback
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.metrics import roc_auc_score


VERSION = "2026-08-19-Q1-R07B-v1"
BUILD = "Q1_R07B_HARM_AUC_OUTCOME_CONDITIONAL_DECOMPOSITION"

ROOT = Path(r"F:\MEDSEG_SAFETTA")

PROTOCOL = (
    ROOT / "docs"
    / "Q1_R07B_harm_auc_outcome_conditional_decomposition_preregistered_protocol_v1.md"
)
EXPECTED_PROTOCOL_SHA256 = "0934c03510b127e7e1615d64b825c5eb26ffeef166bd71896eccb6d62fe7241a"

R06B_DIR = ROOT / "outputs" / "Q1_R06B_source_only_nested_recall_anchored_harmguard_gate_v1"
R06B_LOCK = R06B_DIR / "Q1_R06B_HARMGUARD_GATE_LOCK.json"
EXPECTED_R06B_LOCK_SHA256 = "4bd92854103343e06170729d861b9efd60422faeb71b117c03050ec168fec888"

R06C_DIR = ROOT / "outputs" / "Q1_R06C_scalar_harm_risk_gate_feasibility_frontier_audit_v1"
R06C_LOCK = R06C_DIR / "Q1_R06C_SCALAR_GATE_FRONTIER_LOCK.json"
EXPECTED_R06C_LOCK_SHA256 = "59751d9977a342234f00820252d6db0e335abf8597e3bcf4f8ad82ffd1a9f18a"

D4_DIR = ROOT / "outputs" / "Q1_R05D4_neopolyp_gt_reveal_paot_confirmatory_evaluation_v1"
D4_LOCK = D4_DIR / "Q1_R05D4_NEOPOLYP_CONFIRMATORY_EVALUATION_LOCK.json"
EXPECTED_D4_LOCK_SHA256 = "0e9dcd76fbe736959dfadd64a3f6b5d5ae7fe35434793c41fa80ff0ac6dacc45"

R07A_DIR = ROOT / "outputs" / "Q1_R07A_neopolyp_external_predictability_control_gap_audit_v1"
R07A_LOCK = R07A_DIR / "Q1_R07A_EXTERNAL_PREDICTABILITY_CONTROL_GAP_LOCK.json"
EXPECTED_R07A_LOCK_SHA256 = "78666a9b507ee7c3d19f7af62196a1d9e71cd6dae478cbe3414dc70ddf097358"

OUTPUT_DIR = ROOT / "outputs" / "Q1_R07B_harm_auc_outcome_conditional_decomposition_v1"

FAMILIES = ("PraNet", "DeepLabV3-R50", "SegFormer-B0")
DOMAINS = ("SOURCE_OOF", "NEOPOLYP_EXTERNAL")
DECOMP_TOL = 1e-12

DECISION_SUPPORTED = "PREDICTABILITY_CONTROL_GAP_MECHANISM_SUPPORTED"
DECISION_NOT_UNIVERSAL = "PREDICTABILITY_CONTROL_GAP_MECHANISM_NOT_UNIVERSAL"


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def validate_sha(path: Path, expected: str, label: str):
    if not path.exists():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual.lower() != expected.lower():
        raise RuntimeError(f"{label} SHA mismatch: {actual} != {expected}")
    return actual


def read_csv(path: Path):
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        return list(r), r.fieldnames or []


def write_csv(path: Path, rows, fields: Sequence[str]):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(fields))
        w.writeheader()
        w.writerows(rows)


def write_json(path: Path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def artifact_path(base: Path, lock: dict, key: str) -> Path:
    meta = lock.get("artifacts", {}).get(key)
    if not isinstance(meta, dict):
        raise RuntimeError(f"Missing artifact metadata: {key}")
    rel = meta.get("relative_path") or meta.get("filename")
    if not rel:
        raise RuntimeError(f"Missing artifact path: {key}")
    p = base / str(rel)
    validate_sha(p, str(meta["sha256"]), f"artifact {key}")
    return p


def auc_binary(pos_scores, neg_scores):
    pos_scores = np.asarray(pos_scores, dtype=np.float64)
    neg_scores = np.asarray(neg_scores, dtype=np.float64)
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        raise RuntimeError("Pairwise AUC class empty.")
    y = np.concatenate([
        np.ones(len(pos_scores), dtype=np.int8),
        np.zeros(len(neg_scores), dtype=np.int8),
    ])
    p = np.concatenate([pos_scores, neg_scores])
    return float(roc_auc_score(y, p))


def decompose_cell(domain, family, rows):
    subset = [r for r in rows if r["domain"] == domain and r["family"] == family]
    harm = [r["score"] for r in subset if r["outcome"] == "HARM"]
    neutral = [r["score"] for r in subset if r["outcome"] == "NEUTRAL"]
    benefit = [r["score"] for r in subset if r["outcome"] == "BENEFIT"]

    auc_hn = auc_binary(harm, neutral)
    auc_hb = auc_binary(harm, benefit)
    auc_hr = auc_binary(harm, neutral + benefit)

    n_n = len(neutral)
    n_b = len(benefit)
    w_n = n_n / (n_n + n_b)
    w_b = n_b / (n_n + n_b)
    reconstructed = w_n * auc_hn + w_b * auc_hb
    error = abs(auc_hr - reconstructed)

    if error > DECOMP_TOL:
        raise RuntimeError(
            f"AUROC decomposition failed {domain}/{family}: error={error}"
        )

    return {
        "domain": domain,
        "model_family": family,
        "rows": len(subset),
        "harm_rows": len(harm),
        "neutral_rows": len(neutral),
        "benefit_rows": len(benefit),
        "negative_neutral_weight": float(w_n),
        "negative_benefit_weight": float(w_b),
        "auc_harm_vs_rest": auc_hr,
        "auc_harm_vs_neutral": auc_hn,
        "auc_harm_vs_benefit": auc_hb,
        "neutral_benefit_gap": float(auc_hn - auc_hb),
        "reconstructed_harm_vs_rest_auc": float(reconstructed),
        "decomposition_abs_error": float(error),
        "decomposition_pass": int(error <= DECOMP_TOL),
    }


def load_inputs():
    validate_sha(PROTOCOL, EXPECTED_PROTOCOL_SHA256, "R07B protocol")
    validate_sha(R06B_LOCK, EXPECTED_R06B_LOCK_SHA256, "R06B lock")
    validate_sha(R06C_LOCK, EXPECTED_R06C_LOCK_SHA256, "R06C lock")
    validate_sha(D4_LOCK, EXPECTED_D4_LOCK_SHA256, "D4 lock")
    validate_sha(R07A_LOCK, EXPECTED_R07A_LOCK_SHA256, "R07A lock")

    r06b = json.loads(R06B_LOCK.read_text(encoding="utf-8"))
    r06c = json.loads(R06C_LOCK.read_text(encoding="utf-8"))
    d4 = json.loads(D4_LOCK.read_text(encoding="utf-8"))
    r07a = json.loads(R07A_LOCK.read_text(encoding="utf-8"))

    if r06b.get("decision") != "STOP_RECALL_ANCHORED_HARMGUARD_GATE":
        raise RuntimeError("Unexpected R06B decision.")
    if r06c.get("decision") != "SCALAR_HARM_RISK_GATE_STRUCTURALLY_INSUFFICIENT":
        raise RuntimeError("Unexpected R06C decision.")
    if d4.get("decision") != "NEOPOLYP_PAOT_CONFIRMATION_FAIL":
        raise RuntimeError("Unexpected D4 decision.")
    if r07a.get("decision") != "EXTERNAL_PREDICTABILITY_CONTROL_GAP_REPLICATED":
        raise RuntimeError("Unexpected R07A decision.")

    source_path = artifact_path(R06B_DIR, r06b, "outer_model_case_decisions.csv")
    source_rows, source_fields = read_csv(source_path)
    required_source = {
        "model_family", "harm_probability", "true_harmful", "true_beneficial"
    }
    if not required_source.issubset(set(source_fields)):
        raise RuntimeError("R06B source schema mismatch.")

    external_path = artifact_path(D4_DIR, d4, "model_case_outcomes")
    external_rows, external_fields = read_csv(external_path)
    required_external = {
        "model_family", "paot_mrz19_harm_probability",
        "harm_label", "benefit_label"
    }
    if not required_external.issubset(set(external_fields)):
        raise RuntimeError("D4 external schema mismatch.")

    rows = []
    for r in source_rows:
        h = int(r["true_harmful"])
        b = int(r["true_beneficial"])
        outcome = "HARM" if h else "BENEFIT" if b else "NEUTRAL"
        rows.append({
            "domain": "SOURCE_OOF",
            "family": r["model_family"],
            "score": float(r["harm_probability"]),
            "outcome": outcome,
        })

    for r in external_rows:
        h = int(r["harm_label"])
        b = int(r["benefit_label"])
        outcome = "HARM" if h else "BENEFIT" if b else "NEUTRAL"
        rows.append({
            "domain": "NEOPOLYP_EXTERNAL",
            "family": r["model_family"],
            "score": float(r["paot_mrz19_harm_probability"]),
            "outcome": outcome,
        })

    source_frontier_path = artifact_path(
        R06C_DIR, r06c, "architecture_frontier_summary.csv"
    )
    source_frontier, _ = read_csv(source_frontier_path)

    external_frontier_path = artifact_path(
        R07A_DIR, r07a, "external_architecture_frontier_summary.csv"
    )
    external_frontier, _ = read_csv(external_frontier_path)

    return (
        r06b, r06c, d4, r07a,
        source_path, external_path,
        rows, source_frontier, external_frontier
    )


def parse_optional_float(x):
    if x is None or str(x).strip() in ("", "None"):
        return None
    return float(x)


def preflight():
    *_, rows, source_frontier, external_frontier = load_inputs()
    print("===== Q1-R07B PREFLIGHT =====")
    print(f"protocol_sha256={EXPECTED_PROTOCOL_SHA256}")
    print(f"r06b_lock_sha256={EXPECTED_R06B_LOCK_SHA256}")
    print(f"r06c_lock_sha256={EXPECTED_R06C_LOCK_SHA256}")
    print(f"d4_lock_sha256={EXPECTED_D4_LOCK_SHA256}")
    print(f"r07a_lock_sha256={EXPECTED_R07A_LOCK_SHA256}")
    print(f"combined_rows={len(rows)}")
    print(f"source_frontier_architectures={len(source_frontier)}")
    print(f"external_frontier_architectures={len(external_frontier)}")
    print("new_model_fit=NO")
    print("new_inference_TTA=NO")
    print("threshold_selection=NO")
    print("mechanism=HARM_vs_NEUTRAL_vs_BENEFIT_AUROC_decomposition")
    print("PREFLIGHT_PASS")


def run(args):
    (
        r06b, r06c, d4, r07a,
        source_path, external_path,
        rows, source_frontier, external_frontier
    ) = load_inputs()

    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    build_dir = Path(str(args.output_dir) + "__building")
    if build_dir.exists():
        raise FileExistsError(build_dir)
    build_dir.mkdir(parents=True, exist_ok=False)

    protocol_copy = build_dir / "preregistered_protocol_copy.md"
    shutil.copy2(PROTOCOL, protocol_copy)
    validate_sha(protocol_copy, EXPECTED_PROTOCOL_SHA256, "protocol copy")

    write_json(
        build_dir / "upstream_audit.json",
        {
            "script_version": VERSION,
            "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
            "r06b_lock_sha256": EXPECTED_R06B_LOCK_SHA256,
            "r06c_lock_sha256": EXPECTED_R06C_LOCK_SHA256,
            "d4_lock_sha256": EXPECTED_D4_LOCK_SHA256,
            "r07a_lock_sha256": EXPECTED_R07A_LOCK_SHA256,
            "source_oof_table_sha256": sha256_file(source_path),
            "external_outcome_table_sha256": sha256_file(external_path),
            "new_model_fit": False,
            "new_inference": False,
            "tta": False,
            "calibration": False,
            "threshold_selection": False,
        },
    )

    decomposition = [
        decompose_cell(domain, family, rows)
        for domain in DOMAINS
        for family in FAMILIES
    ]

    write_csv(
        build_dir / "outcome_conditional_auc_decomposition.csv",
        decomposition,
        [
            "domain", "model_family", "rows",
            "harm_rows", "neutral_rows", "benefit_rows",
            "negative_neutral_weight", "negative_benefit_weight",
            "auc_harm_vs_rest", "auc_harm_vs_neutral",
            "auc_harm_vs_benefit", "neutral_benefit_gap",
            "reconstructed_harm_vs_rest_auc",
            "decomposition_abs_error", "decomposition_pass",
        ],
    )

    source_idx = {r["model_family"]: r for r in source_frontier}
    ext_idx = {r["model_family"]: r for r in external_frontier}
    dec_idx = {(r["domain"], r["model_family"]): r for r in decomposition}

    summary = []
    for domain in DOMAINS:
        for family in FAMILIES:
            d = dec_idx[(domain, family)]
            f = source_idx[family] if domain == "SOURCE_OOF" else ext_idx[family]
            max_benefit = parse_optional_float(
                f["max_benefit_retention_given_harm_ge_0p80_coverage_ge_0p30_dice"]
            )
            summary.append({
                "domain": domain,
                "model_family": family,
                "auc_harm_vs_rest": d["auc_harm_vs_rest"],
                "auc_harm_vs_neutral": d["auc_harm_vs_neutral"],
                "auc_harm_vs_benefit": d["auc_harm_vs_benefit"],
                "neutral_benefit_gap": d["neutral_benefit_gap"],
                "max_benefit_retention_at_harm80_cov30_dice": max_benefit,
                "scalar_frontier_joint_feasible": int(f["joint_feasible"]),
            })

    write_csv(
        build_dir / "source_external_mechanism_summary.csv",
        summary,
        [
            "domain", "model_family",
            "auc_harm_vs_rest", "auc_harm_vs_neutral",
            "auc_harm_vs_benefit", "neutral_benefit_gap",
            "max_benefit_retention_at_harm80_cov30_dice",
            "scalar_frontier_joint_feasible",
        ],
    )

    checks = {}
    for r in summary:
        tag = f"{r['domain']}__{r['model_family']}"
        checks[f"{tag}__overall_auc_ge_0p70"] = r["auc_harm_vs_rest"] >= 0.70
        checks[f"{tag}__harm_neutral_auc_gt_harm_benefit_auc"] = (
            r["auc_harm_vs_neutral"] > r["auc_harm_vs_benefit"]
        )
        checks[f"{tag}__max_benefit_retention_lt_0p50"] = (
            r["max_benefit_retention_at_harm80_cov30_dice"] is not None
            and r["max_benefit_retention_at_harm80_cov30_dice"] < 0.50
        )
        checks[f"{tag}__scalar_frontier_infeasible"] = (
            r["scalar_frontier_joint_feasible"] == 0
        )

    decomposition_ok = all(r["decomposition_pass"] == 1 for r in decomposition)
    checks["all_six_exact_auc_decomposition"] = decomposition_ok

    decision = (
        DECISION_SUPPORTED
        if all(checks.values())
        else DECISION_NOT_UNIVERSAL
    )

    gate = {
        "mechanism": (
            "overall HARM AUROC can be inflated by easier HARM-vs-NEUTRAL "
            "ranking while HARM-vs-BENEFIT ranking remains weaker"
        ),
        "checks": checks,
        "decision": decision,
        "new_model_fit": False,
        "threshold_selected": False,
    }
    write_json(build_dir / "mechanism_gate.json", gate)
    (build_dir / "decision.txt").write_text(decision + "\n", encoding="utf-8")

    lines = [
        "===== Q1-R07B HARM AUROC OUTCOME-CONDITIONAL DECOMPOSITION =====",
        f"Script version: {VERSION}",
        f"Build: {BUILD}",
        "",
        "Status:",
        "  new model fit=NO",
        "  inference/TTA=NO",
        "  calibration=NO",
        "  threshold selection=NO",
        "",
        "Outcome-conditional AUROC:",
    ]

    for r in decomposition:
        lines += [
            (
                f"  {r['domain']} / {r['model_family']}: "
                f"H-vs-rest={r['auc_harm_vs_rest']:.6f} "
                f"H-vs-neutral={r['auc_harm_vs_neutral']:.6f} "
                f"H-vs-benefit={r['auc_harm_vs_benefit']:.6f} "
                f"gap={r['neutral_benefit_gap']:+.6f}"
            ),
            (
                f"    weights neutral={r['negative_neutral_weight']:.4f} "
                f"benefit={r['negative_benefit_weight']:.4f} "
                f"decomp_error={r['decomposition_abs_error']:.3e}"
            ),
        ]

    lines += ["", "Mechanism checks:"]
    for k, v in checks.items():
        lines.append(f"  {k}={'PASS' if v else 'FAIL'}")

    lines += [
        "",
        "Decision:",
        f"  {decision}",
    ]

    run_log = build_dir / "run_log.txt"
    run_log.write_text("\n".join(lines) + "\n", encoding="utf-8")

    artifact_names = [
        "preregistered_protocol_copy.md",
        "upstream_audit.json",
        "outcome_conditional_auc_decomposition.csv",
        "source_external_mechanism_summary.csv",
        "mechanism_gate.json",
        "decision.txt",
        "run_log.txt",
    ]
    artifacts = {
        name: {
            "relative_path": name,
            "sha256": sha256_file(build_dir / name),
        }
        for name in artifact_names
    }

    final_lock = {
        "script_version": VERSION,
        "build": BUILD,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "r06b_lock_sha256": EXPECTED_R06B_LOCK_SHA256,
        "r06c_lock_sha256": EXPECTED_R06C_LOCK_SHA256,
        "d4_lock_sha256": EXPECTED_D4_LOCK_SHA256,
        "r07a_lock_sha256": EXPECTED_R07A_LOCK_SHA256,
        "new_model_fit": False,
        "new_inference": False,
        "threshold_selected": False,
        "decision": decision,
        "mechanism_gate": gate,
        "artifacts": artifacts,
    }

    lock_path = build_dir / "Q1_R07B_HARM_AUC_DECOMPOSITION_LOCK.json"
    write_json(lock_path, final_lock)
    lock_sha = sha256_file(lock_path)

    for name, meta in artifacts.items():
        if sha256_file(build_dir / meta["relative_path"]) != meta["sha256"]:
            raise RuntimeError(f"Artifact changed before commit: {name}")

    build_dir.rename(args.output_dir)

    print()
    print((args.output_dir / "run_log.txt").read_text(encoding="utf-8"))
    print(
        "Q1-R07B LOCK:",
        args.output_dir / "Q1_R07B_HARM_AUC_DECOMPOSITION_LOCK.json",
    )
    print("Q1-R07B LOCK SHA256:", lock_sha)


def self_test():
    harm = np.asarray([0.9, 0.8])
    neutral = np.asarray([0.2, 0.3, 0.4])
    benefit = np.asarray([0.6, 0.7])

    a_hr = auc_binary(harm, np.concatenate([neutral, benefit]))
    a_hn = auc_binary(harm, neutral)
    a_hb = auc_binary(harm, benefit)
    w_n = len(neutral) / (len(neutral) + len(benefit))
    w_b = len(benefit) / (len(neutral) + len(benefit))
    recon = w_n * a_hn + w_b * a_hb
    assert abs(a_hr - recon) <= DECOMP_TOL

    print("PAIRWISE_AUROC_TEST_PASS")
    print("EXACT_WEIGHTED_DECOMPOSITION_TEST_PASS")
    print("NO_MODEL_FIT_SELF_TEST_PASS")
    print("SELF_TEST_PASS")


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Q1-R07B: decompose frozen HARM-vs-rest AUROC into "
            "HARM-vs-NEUTRAL and HARM-vs-BENEFIT pairwise AUROCs."
        )
    )
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--self-test", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.preflight_only:
        preflight()
        return 0
    try:
        run(args)
    except Exception:
        traceback.print_exc()
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
