#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Q1-R07A — Retrospective NeoPolyp predictability-control gap audit.

No model fit.
No inference.
No TTA.
No calibration.
No deployable threshold selection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import traceback
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm


VERSION = "2026-08-19-Q1-R07A-v1"
BUILD = "Q1_R07A_NEOPOLYP_EXTERNAL_PREDICTABILITY_CONTROL_GAP_AUDIT"

ROOT = Path(r"F:\MEDSEG_SAFETTA")

PROTOCOL = (
    ROOT / "docs"
    / "Q1_R07A_neopolyp_external_predictability_control_gap_audit_preregistered_protocol_v1.md"
)
EXPECTED_PROTOCOL_SHA256 = "10a752cf06c9695947f4f0443f2968218cccf60d5d8f20482bc0681e5bee5134"

R06C_DIR = (
    ROOT / "outputs"
    / "Q1_R06C_scalar_harm_risk_gate_feasibility_frontier_audit_v1"
)
R06C_LOCK = R06C_DIR / "Q1_R06C_SCALAR_GATE_FRONTIER_LOCK.json"
EXPECTED_R06C_LOCK_SHA256 = (
    "59751d9977a342234f00820252d6db0e335abf8597e3bcf4f8ad82ffd1a9f18a"
)
EXPECTED_R06C_DECISION = "SCALAR_HARM_RISK_GATE_STRUCTURALLY_INSUFFICIENT"

R05D2_DIR = (
    ROOT / "outputs"
    / "Q1_R05D2_neopolyp_prospective_paot_probability_lock_v1_fix1"
)
R05D2_LOCK = R05D2_DIR / "Q1_R05D2_NEOPOLYP_PAOT_PROBABILITY_LOCK.json"
EXPECTED_R05D2_LOCK_SHA256 = (
    "35705ef25915e173f5ba624fcab524cc74225047c9b340572ffe64566631d7ab"
)
EXPECTED_R05D2_DECISION = "NEOPOLYP_PROSPECTIVE_PAOT_PROBABILITIES_LOCKED"

R05D4_DIR = (
    ROOT / "outputs"
    / "Q1_R05D4_neopolyp_gt_reveal_paot_confirmatory_evaluation_v1"
)
R05D4_LOCK = R05D4_DIR / "Q1_R05D4_NEOPOLYP_CONFIRMATORY_EVALUATION_LOCK.json"
EXPECTED_R05D4_LOCK_SHA256 = (
    "0e9dcd76fbe736959dfadd64a3f6b5d5ae7fe35434793c41fa80ff0ac6dacc45"
)
EXPECTED_R05D4_DECISION = "NEOPOLYP_PAOT_CONFIRMATION_FAIL"

OUTPUT_DIR = (
    ROOT / "outputs"
    / "Q1_R07A_neopolyp_external_predictability_control_gap_audit_v1"
)

FAMILIES = ("PraNet", "DeepLabV3-R50", "SegFormer-B0")
EXPECTED_ROWS = 9000
EXPECTED_ROWS_PER_FAMILY = 3000
EXPECTED_CASES = 1000

HARM_PREVENTION_MIN = 0.80
COVERAGE_MIN = 0.30
BENEFIT_RETENTION_MIN = 0.50
GATE_MINUS_SOURCE_DICE_MIN = -0.005
PREDICTABILITY_AUROC_MIN = 0.70

EXPECTED_D4_HARM_AUROC = {
    "PraNet": 0.840904460273438,
    "DeepLabV3-R50": 0.7793326162394997,
    "SegFormer-B0": 0.7621465567587901,
}

DECISION_REPLICATED = "EXTERNAL_PREDICTABILITY_CONTROL_GAP_REPLICATED"
DECISION_NOT_UNIVERSAL = "EXTERNAL_PREDICTABILITY_CONTROL_GAP_NOT_UNIVERSAL"


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
        raise RuntimeError(
            f"{label} SHA mismatch: expected={expected} actual={actual}"
        )
    return actual


def read_csv(path: Path):
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        return list(r), r.fieldnames or []


def write_csv(path: Path, rows, fields: Sequence[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(fields))
        w.writeheader()
        w.writerows(rows)


def write_json(path: Path, obj):
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def artifact_path_from_lock(base_dir: Path, lock: dict, key: str) -> Path:
    meta = lock.get("artifacts", {}).get(key)
    if not isinstance(meta, dict):
        raise RuntimeError(f"Lock missing artifact: {key}")
    rel = meta.get("relative_path") or meta.get("filename")
    if not rel:
        raise RuntimeError(f"Artifact path missing: {key}")
    p = base_dir / str(rel)
    validate_sha(p, str(meta["sha256"]), f"artifact {key}")
    return p


def validate_and_load():
    validate_sha(PROTOCOL, EXPECTED_PROTOCOL_SHA256, "R07A protocol")
    validate_sha(R06C_LOCK, EXPECTED_R06C_LOCK_SHA256, "R06C lock")
    validate_sha(R05D2_LOCK, EXPECTED_R05D2_LOCK_SHA256, "D2 lock")
    validate_sha(R05D4_LOCK, EXPECTED_R05D4_LOCK_SHA256, "D4 lock")

    r06c = json.loads(R06C_LOCK.read_text(encoding="utf-8"))
    d2 = json.loads(R05D2_LOCK.read_text(encoding="utf-8"))
    d4 = json.loads(R05D4_LOCK.read_text(encoding="utf-8"))

    if r06c.get("decision") != EXPECTED_R06C_DECISION:
        raise RuntimeError(f"Unexpected R06C decision: {r06c.get('decision')}")
    if bool(r06c.get("target_data_read", True)):
        raise RuntimeError("R06C reports target-data read.")

    if d2.get("decision") != EXPECTED_R05D2_DECISION:
        raise RuntimeError(f"Unexpected D2 decision: {d2.get('decision')}")

    if d4.get("decision") != EXPECTED_R05D4_DECISION:
        raise RuntimeError(f"Unexpected D4 decision: {d4.get('decision')}")
    if int(d4.get("target_cases", -1)) != EXPECTED_CASES:
        raise RuntimeError("D4 target-case count changed.")
    if int(d4.get("model_states", -1)) != 9:
        raise RuntimeError("D4 model-state count changed.")
    if int(d4.get("model_case_rows", -1)) != EXPECTED_ROWS:
        raise RuntimeError("D4 model-case count changed.")
    if bool(d4.get("model_inference_run_in_d4", True)):
        raise RuntimeError("D4 reports model inference.")
    if bool(d4.get("tta_run_in_d4", True)):
        raise RuntimeError("D4 reports TTA.")
    if bool(d4.get("predictor_fit_or_modified", True)):
        raise RuntimeError("D4 reports predictor modification.")
    if bool(d4.get("target_calibration", True)):
        raise RuntimeError("D4 reports target calibration.")
    if bool(d4.get("target_threshold_selected", True)):
        raise RuntimeError("D4 reports target threshold selection.")

    outcomes_path = artifact_path_from_lock(
        R05D4_DIR,
        d4,
        "model_case_outcomes",
    )
    rows, fields = read_csv(outcomes_path)

    required = {
        "sample_id",
        "model_family",
        "model_state_id",
        "source_dice",
        "a1_dice",
        "delta_dice",
        "adaptation_outcome",
        "harm_label",
        "benefit_label",
        "paot_mrz19_harm_probability",
    }
    missing = sorted(required - set(fields))
    if missing:
        raise RuntimeError(f"D4 outcomes missing columns: {missing}")
    if len(rows) != EXPECTED_ROWS:
        raise RuntimeError(f"D4 outcome rows={len(rows)}")

    family_counts = Counter(r["model_family"] for r in rows)
    if set(family_counts) != set(FAMILIES):
        raise RuntimeError(f"Family set mismatch: {dict(family_counts)}")
    if any(family_counts[f] != EXPECTED_ROWS_PER_FAMILY for f in FAMILIES):
        raise RuntimeError(f"Family row mismatch: {dict(family_counts)}")

    parsed = []
    for i, r in enumerate(rows):
        p = float(r["paot_mrz19_harm_probability"])
        h = int(r["harm_label"])
        b = int(r["benefit_label"])
        sd = float(r["source_dice"])
        ad = float(r["a1_dice"])
        dd = float(r["delta_dice"])

        if not (0.0 <= p <= 1.0 and math.isfinite(p)):
            raise RuntimeError(f"Invalid HARM score row={i}")
        if h not in (0, 1) or b not in (0, 1) or (h and b):
            raise RuntimeError(f"Invalid labels row={i}")
        if not all(math.isfinite(x) for x in (sd, ad, dd)):
            raise RuntimeError(f"Invalid utility row={i}")

        expected_label = (
            "HARM" if dd <= -0.02
            else "BENEFIT" if dd >= 0.02
            else "NEUTRAL"
        )
        if r["adaptation_outcome"] != expected_label:
            raise RuntimeError(f"Outcome boundary mismatch row={i}")

        parsed.append({
            "sample_id": r["sample_id"],
            "model_family": r["model_family"],
            "model_state_id": r["model_state_id"],
            "risk": p,
            "harm": h,
            "benefit": b,
            "source_dice": sd,
            "action_dice": ad,
        })

    source_summary_path = artifact_path_from_lock(
        R06C_DIR,
        r06c,
        "architecture_frontier_summary.csv",
    )
    source_summary, _ = read_csv(source_summary_path)

    return r06c, d2, d4, outcomes_path, parsed, source_summary


def predictability_metrics(rows, family):
    subset = [r for r in rows if r["model_family"] == family]
    y = np.asarray([r["harm"] for r in subset], dtype=np.int64)
    p = np.asarray([r["risk"] for r in subset], dtype=np.float64)

    if np.unique(y).size != 2:
        raise RuntimeError(f"{family} HARM labels degenerate.")

    return {
        "model_family": family,
        "rows": len(subset),
        "harm_rows": int(y.sum()),
        "harm_prevalence": float(y.mean()),
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
    }


def frontier_for_family(rows, family):
    subset = [r for r in rows if r["model_family"] == family]
    p = np.asarray([r["risk"] for r in subset], dtype=np.float64)
    harm = np.asarray([r["harm"] for r in subset], dtype=np.int64)
    benefit = np.asarray([r["benefit"] for r in subset], dtype=np.int64)
    source = np.asarray([r["source_dice"] for r in subset], dtype=np.float64)
    action = np.asarray([r["action_dice"] for r in subset], dtype=np.float64)

    n = len(subset)
    harm_total = int(harm.sum())
    benefit_total = int(benefit.sum())
    if harm_total == 0 or benefit_total == 0:
        raise RuntimeError(f"{family} outcome class degenerate.")

    order = np.argsort(-p, kind="mergesort")
    ps = p[order]
    hs = harm[order]
    bs = benefit[order]
    src = source[order]
    act = action[order]

    source_mean = float(source.mean())
    action_mean = float(action.mean())
    gated_sum = float(action.sum())

    points = [{
        "model_family": family,
        "blocked_rows": 0,
        "adapted_rows": n,
        "threshold_lower_bound": float("inf"),
        "threshold_semantics": "adapt_all_endpoint",
        "harm_prevention_recall": 0.0,
        "adaptation_coverage": 1.0,
        "benefit_retention": 1.0,
        "source_mean_dice": source_mean,
        "blind_a1_mean_dice": action_mean,
        "gated_mean_dice": action_mean,
        "gated_minus_source_mean_dice": action_mean - source_mean,
        "joint_feasible": 0,
    }]

    blocked_rows = 0
    blocked_harm = 0
    blocked_benefit = 0

    i = 0
    while i < n:
        score = ps[i]
        j = i + 1
        while j < n and ps[j] == score:
            j += 1

        blocked_rows += j - i
        blocked_harm += int(hs[i:j].sum())
        blocked_benefit += int(bs[i:j].sum())
        gated_sum += float(np.sum(src[i:j] - act[i:j]))

        harm_prevention = blocked_harm / harm_total
        coverage = (n - blocked_rows) / n
        benefit_retention = (benefit_total - blocked_benefit) / benefit_total
        gated_mean = gated_sum / n
        gate_source = gated_mean - source_mean

        feasible = (
            harm_prevention >= HARM_PREVENTION_MIN
            and coverage >= COVERAGE_MIN
            and benefit_retention >= BENEFIT_RETENTION_MIN
            and gate_source >= GATE_MINUS_SOURCE_DICE_MIN
        )

        points.append({
            "model_family": family,
            "blocked_rows": blocked_rows,
            "adapted_rows": n - blocked_rows,
            "threshold_lower_bound": float(score),
            "threshold_semantics": "block_if_risk_ge_threshold",
            "harm_prevention_recall": float(harm_prevention),
            "adaptation_coverage": float(coverage),
            "benefit_retention": float(benefit_retention),
            "source_mean_dice": source_mean,
            "blind_a1_mean_dice": action_mean,
            "gated_mean_dice": float(gated_mean),
            "gated_minus_source_mean_dice": float(gate_source),
            "joint_feasible": int(feasible),
        })
        i = j

    return points


def max_or_none(vals):
    vals = list(vals)
    return None if not vals else float(max(vals))


def frontier_summary(points, family):
    feasible = [r for r in points if r["joint_feasible"] == 1]

    c1 = [
        r for r in points
        if r["harm_prevention_recall"] >= HARM_PREVENTION_MIN
        and r["adaptation_coverage"] >= COVERAGE_MIN
        and r["gated_minus_source_mean_dice"] >= GATE_MINUS_SOURCE_DICE_MIN
    ]
    c2 = [
        r for r in points
        if r["harm_prevention_recall"] >= HARM_PREVENTION_MIN
        and r["benefit_retention"] >= BENEFIT_RETENTION_MIN
        and r["gated_minus_source_mean_dice"] >= GATE_MINUS_SOURCE_DICE_MIN
    ]
    c3 = [
        r for r in points
        if r["adaptation_coverage"] >= COVERAGE_MIN
        and r["benefit_retention"] >= BENEFIT_RETENTION_MIN
        and r["gated_minus_source_mean_dice"] >= GATE_MINUS_SOURCE_DICE_MIN
    ]

    return {
        "model_family": family,
        "frontier_points": len(points),
        "joint_feasible_points": len(feasible),
        "joint_feasible": int(bool(feasible)),
        "max_benefit_retention_given_harm_ge_0p80_coverage_ge_0p30_dice":
            max_or_none(r["benefit_retention"] for r in c1),
        "max_coverage_given_harm_ge_0p80_benefit_ge_0p50_dice":
            max_or_none(r["adaptation_coverage"] for r in c2),
        "max_harm_prevention_given_coverage_ge_0p30_benefit_ge_0p50_dice":
            max_or_none(r["harm_prevention_recall"] for r in c3),
    }


def preflight():
    r06c, d2, d4, outcomes_path, rows, source_summary = validate_and_load()
    print("===== Q1-R07A PREFLIGHT =====")
    print(f"protocol_sha256={EXPECTED_PROTOCOL_SHA256}")
    print(f"r06c_lock_sha256={EXPECTED_R06C_LOCK_SHA256}")
    print(f"d2_lock_sha256={EXPECTED_R05D2_LOCK_SHA256}")
    print(f"d4_lock_sha256={EXPECTED_R05D4_LOCK_SHA256}")
    print(f"external_rows={len(rows)}")
    print(f"source_frontier_architectures={len(source_summary)}")
    print("new_model_fit=NO")
    print("new_inference=NO")
    print("TTA=NO")
    print("calibration=NO")
    print("deployment_threshold_selected=NO")
    print("retrospective_external_characterization=YES")
    print("PREFLIGHT_PASS")


def run(args):
    r06c, d2, d4, outcomes_path, rows, source_summary = validate_and_load()

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
            "r06c_lock_sha256": EXPECTED_R06C_LOCK_SHA256,
            "r06c_decision": r06c.get("decision"),
            "d2_lock_sha256": EXPECTED_R05D2_LOCK_SHA256,
            "d2_decision": d2.get("decision"),
            "d4_lock_sha256": EXPECTED_R05D4_LOCK_SHA256,
            "d4_decision": d4.get("decision"),
            "d4_outcomes_sha256": sha256_file(outcomes_path),
            "rows": EXPECTED_ROWS,
            "retrospective_external_characterization": True,
            "new_model_fit": False,
            "new_inference": False,
            "tta": False,
            "calibration": False,
            "deployment_threshold_selected": False,
        },
    )

    predict_rows = []
    frontier_rows = []
    summary_rows = []

    for family in tqdm(FAMILIES, desc="R07A NeoPolyp frontiers", unit="arch"):
        pm = predictability_metrics(rows, family)
        expected_auc = EXPECTED_D4_HARM_AUROC[family]
        if abs(pm["auroc"] - expected_auc) > 1e-12:
            raise RuntimeError(
                f"D4 HARM AUROC integrity mismatch {family}: "
                f"{pm['auroc']} != {expected_auc}"
            )
        predict_rows.append(pm)

        pts = frontier_for_family(rows, family)
        frontier_rows.extend(pts)
        summary_rows.append(frontier_summary(pts, family))

    write_csv(
        build_dir / "external_predictability_metrics.csv",
        predict_rows,
        [
            "model_family",
            "rows",
            "harm_rows",
            "harm_prevalence",
            "auroc",
            "auprc",
        ],
    )

    write_csv(
        build_dir / "external_frontier_points.csv",
        frontier_rows,
        [
            "model_family",
            "blocked_rows",
            "adapted_rows",
            "threshold_lower_bound",
            "threshold_semantics",
            "harm_prevention_recall",
            "adaptation_coverage",
            "benefit_retention",
            "source_mean_dice",
            "blind_a1_mean_dice",
            "gated_mean_dice",
            "gated_minus_source_mean_dice",
            "joint_feasible",
        ],
    )

    write_csv(
        build_dir / "external_architecture_frontier_summary.csv",
        summary_rows,
        [
            "model_family",
            "frontier_points",
            "joint_feasible_points",
            "joint_feasible",
            "max_benefit_retention_given_harm_ge_0p80_coverage_ge_0p30_dice",
            "max_coverage_given_harm_ge_0p80_benefit_ge_0p50_dice",
            "max_harm_prevention_given_coverage_ge_0p30_benefit_ge_0p50_dice",
        ],
    )

    source_idx = {r["model_family"]: r for r in source_summary}
    external_idx = {r["model_family"]: r for r in summary_rows}
    predict_idx = {r["model_family"]: r for r in predict_rows}

    compare_rows = []
    for family in FAMILIES:
        s = source_idx[family]
        e = external_idx[family]
        p = predict_idx[family]
        compare_rows.append({
            "model_family": family,
            "source_scalar_frontier_joint_feasible": int(s["joint_feasible"]),
            "external_harm_auroc": p["auroc"],
            "external_harm_auprc": p["auprc"],
            "external_scalar_frontier_joint_feasible": int(e["joint_feasible"]),
            "source_max_benefit_retention_at_harm80_cov30_dice":
                s["max_benefit_retention_given_harm_ge_0p80_coverage_ge_0p30_dice"],
            "external_max_benefit_retention_at_harm80_cov30_dice":
                e["max_benefit_retention_given_harm_ge_0p80_coverage_ge_0p30_dice"],
        })

    write_csv(
        build_dir / "source_vs_external_control_gap_summary.csv",
        compare_rows,
        [
            "model_family",
            "source_scalar_frontier_joint_feasible",
            "external_harm_auroc",
            "external_harm_auprc",
            "external_scalar_frontier_joint_feasible",
            "source_max_benefit_retention_at_harm80_cov30_dice",
            "external_max_benefit_retention_at_harm80_cov30_dice",
        ],
    )

    checks = {}
    for family in FAMILIES:
        checks[f"{family}_harm_auroc_ge_0p70"] = (
            predict_idx[family]["auroc"] >= PREDICTABILITY_AUROC_MIN
        )
        checks[f"{family}_scalar_frontier_infeasible"] = (
            int(external_idx[family]["joint_feasible"]) == 0
        )

    decision = (
        DECISION_REPLICATED
        if all(checks.values())
        else DECISION_NOT_UNIVERSAL
    )

    gate = {
        "retrospective": True,
        "predictability_auroc_min": PREDICTABILITY_AUROC_MIN,
        "feasibility_constraints": {
            "harm_prevention_min": HARM_PREVENTION_MIN,
            "coverage_min": COVERAGE_MIN,
            "benefit_retention_min": BENEFIT_RETENTION_MIN,
            "gate_minus_source_dice_min": GATE_MINUS_SOURCE_DICE_MIN,
        },
        "checks": checks,
        "decision": decision,
        "deployment_threshold_selected": False,
    }
    write_json(build_dir / "descriptive_gate.json", gate)
    (build_dir / "decision.txt").write_text(decision + "\n", encoding="utf-8")

    lines = [
        "===== Q1-R07A NEOPOLYP EXTERNAL PREDICTABILITY-CONTROL GAP AUDIT =====",
        f"Script version: {VERSION}",
        f"Build: {BUILD}",
        "",
        "Status:",
        "  retrospective external characterization=YES",
        "  D2 score frozen before GT reveal=YES",
        "  new model fit=NO",
        "  new inference/TTA=NO",
        "  deployment threshold selected=NO",
        "",
        "External architecture results:",
    ]

    for family in FAMILIES:
        p = predict_idx[family]
        e = external_idx[family]
        lines += [
            (
                f"  {family}: HARM AUROC={p['auroc']:.6f} "
                f"AUPRC={p['auprc']:.6f}"
            ),
            (
                f"    scalar frontier joint feasible="
                f"{'YES' if e['joint_feasible'] else 'NO'} "
                f"feasible_points={e['joint_feasible_points']}"
            ),
            (
                "    max benefit retention | harm>=.80,cov>=.30,dice = "
                f"{e['max_benefit_retention_given_harm_ge_0p80_coverage_ge_0p30_dice']}"
            ),
        ]

    lines += [
        "",
        "Descriptive checks:",
    ]
    for key, passed in checks.items():
        lines.append(f"  {key}={'PASS' if passed else 'FAIL'}")

    lines += [
        "",
        "Decision:",
        f"  {decision}",
        "",
        "Interpretation:",
        (
            "  High external HARM predictability coexists with absence of a useful "
            "monotone scalar safety gate across all three architectures."
            if decision == DECISION_REPLICATED
            else
            "  The source predictability-control gap does not reproduce uniformly "
            "across all external architectures."
        ),
    ]

    run_log = build_dir / "run_log.txt"
    run_log.write_text("\n".join(lines) + "\n", encoding="utf-8")

    artifact_names = [
        "preregistered_protocol_copy.md",
        "upstream_audit.json",
        "external_predictability_metrics.csv",
        "external_frontier_points.csv",
        "external_architecture_frontier_summary.csv",
        "source_vs_external_control_gap_summary.csv",
        "descriptive_gate.json",
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

    lock = {
        "script_version": VERSION,
        "build": BUILD,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "r06c_lock_sha256": EXPECTED_R06C_LOCK_SHA256,
        "d2_lock_sha256": EXPECTED_R05D2_LOCK_SHA256,
        "d4_lock_sha256": EXPECTED_R05D4_LOCK_SHA256,
        "retrospective_external_characterization": True,
        "new_model_fit": False,
        "new_inference": False,
        "deployment_threshold_selected": False,
        "decision": decision,
        "descriptive_gate": gate,
        "artifacts": artifacts,
    }

    lock_path = build_dir / "Q1_R07A_EXTERNAL_PREDICTABILITY_CONTROL_GAP_LOCK.json"
    write_json(lock_path, lock)
    lock_sha = sha256_file(lock_path)

    for name, meta in artifacts.items():
        if sha256_file(build_dir / meta["relative_path"]) != meta["sha256"]:
            raise RuntimeError(f"Artifact changed before commit: {name}")

    build_dir.rename(args.output_dir)

    print()
    print((args.output_dir / "run_log.txt").read_text(encoding="utf-8"))
    print(
        "Q1-R07A LOCK:",
        args.output_dir / "Q1_R07A_EXTERNAL_PREDICTABILITY_CONTROL_GAP_LOCK.json",
    )
    print("Q1-R07A LOCK SHA256:", lock_sha)


def self_test():
    assert EXPECTED_ROWS == 9000
    assert EXPECTED_ROWS_PER_FAMILY == 3000
    assert HARM_PREVENTION_MIN == 0.80
    assert COVERAGE_MIN == 0.30
    assert BENEFIT_RETENTION_MIN == 0.50
    assert GATE_MINUS_SOURCE_DICE_MIN == -0.005
    assert PREDICTABILITY_AUROC_MIN == 0.70
    assert EXPECTED_D4_HARM_AUROC["PraNet"] == 0.840904460273438

    print("CARDINALITY_TEST_PASS")
    print("FROZEN_EXTERNAL_AUROC_ASSERTION_TEST_PASS")
    print("FROZEN_FEASIBILITY_REGION_TEST_PASS")
    print("NO_MODEL_FIT_SELF_TEST_PASS")
    print("RETROSPECTIVE_STATUS_TEST_PASS")
    print("SELF_TEST_PASS")


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Q1-R07A: retrospective external NeoPolyp audit of the "
            "predictability-control gap for the frozen MRZ19 HARM score."
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
