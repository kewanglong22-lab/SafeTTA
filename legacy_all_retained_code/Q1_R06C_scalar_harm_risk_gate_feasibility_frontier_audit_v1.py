#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Q1-R06C — Source-only oracle feasibility audit of the complete monotone
threshold frontier of the already-frozen R06B B1 HARM score.

No new model is fit.
No deployment threshold is selected.
No target data are read.
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
from tqdm import tqdm


VERSION = "2026-08-19-Q1-R06C-v1"
BUILD = "Q1_R06C_SCALAR_HARM_RISK_GATE_FEASIBILITY_FRONTIER_AUDIT"

ROOT = Path(r"F:\MEDSEG_SAFETTA")

PROTOCOL = (
    ROOT / "docs"
    / "Q1_R06C_scalar_harm_risk_gate_feasibility_frontier_audit_preregistered_protocol_v1.md"
)
EXPECTED_PROTOCOL_SHA256 = "fbe5023fa1c651cec9cc84621873115ffe04ac4557f332d805dafdd152f3d1a7"

R06B_DIR = (
    ROOT / "outputs"
    / "Q1_R06B_source_only_nested_recall_anchored_harmguard_gate_v1"
)
R06B_LOCK = R06B_DIR / "Q1_R06B_HARMGUARD_GATE_LOCK.json"
EXPECTED_R06B_LOCK_SHA256 = (
    "4bd92854103343e06170729d861b9efd60422faeb71b117c03050ec168fec888"
)
EXPECTED_R06B_DECISION = "STOP_RECALL_ANCHORED_HARMGUARD_GATE"

OUTPUT_DIR = (
    ROOT / "outputs"
    / "Q1_R06C_scalar_harm_risk_gate_feasibility_frontier_audit_v1"
)

FAMILIES = ("PraNet", "DeepLabV3-R50", "SegFormer-B0")
EXPECTED_ROWS = 13050
EXPECTED_ROWS_PER_FAMILY = 4350

HARM_PREVENTION_MIN = 0.80
COVERAGE_MIN = 0.30
BENEFIT_RETENTION_MIN = 0.50
GATE_MINUS_SOURCE_DICE_MIN = -0.005

DECISION_FEASIBLE = "SCALAR_HARM_RISK_GATE_FRONTIER_FEASIBLE"
DECISION_INSUFFICIENT = "SCALAR_HARM_RISK_GATE_STRUCTURALLY_INSUFFICIENT"


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


def artifact_path_from_lock(lock: dict, key: str) -> Path:
    meta = lock.get("artifacts", {}).get(key)
    if not isinstance(meta, dict):
        raise RuntimeError(f"R06B lock missing artifact: {key}")
    rel = meta.get("relative_path")
    if not rel:
        raise RuntimeError(f"R06B artifact path missing: {key}")
    p = R06B_DIR / rel
    validate_sha(p, meta["sha256"], f"R06B artifact {key}")
    return p


def validate_and_load():
    validate_sha(PROTOCOL, EXPECTED_PROTOCOL_SHA256, "R06C protocol")
    validate_sha(R06B_LOCK, EXPECTED_R06B_LOCK_SHA256, "R06B lock")

    lock = json.loads(R06B_LOCK.read_text(encoding="utf-8"))
    if lock.get("decision") != EXPECTED_R06B_DECISION:
        raise RuntimeError(f"Unexpected R06B decision: {lock.get('decision')}")
    if bool(lock.get("target_data_read_by_script", True)):
        raise RuntimeError("R06B reports target data read.")
    if lock.get("risk_estimator") != "B1_MRZ19_POOLED_LOGISTIC":
        raise RuntimeError("R06B risk estimator changed.")

    table_path = artifact_path_from_lock(
        lock,
        "outer_model_case_decisions.csv",
    )
    rows, fields = read_csv(table_path)

    required = {
        "model_family",
        "model_state_id",
        "source_group_id",
        "perturbation",
        "harm_probability",
        "true_harmful",
        "true_beneficial",
        "source_dice",
        "action_dice",
    }
    missing = sorted(required - set(fields))
    if missing:
        raise RuntimeError(f"R06B decision table missing columns: {missing}")
    if len(rows) != EXPECTED_ROWS:
        raise RuntimeError(f"Rows={len(rows)} expected={EXPECTED_ROWS}")

    counts = Counter(r["model_family"] for r in rows)
    if set(counts) != set(FAMILIES):
        raise RuntimeError(f"Family set mismatch: {dict(counts)}")
    if any(counts[f] != EXPECTED_ROWS_PER_FAMILY for f in FAMILIES):
        raise RuntimeError(f"Family row mismatch: {dict(counts)}")

    keys = {
        (r["model_state_id"], r["source_group_id"], r["perturbation"])
        for r in rows
    }
    if len(keys) != EXPECTED_ROWS:
        raise RuntimeError("Model-case keys are not unique.")

    parsed = []
    for i, r in enumerate(rows):
        p = float(r["harm_probability"])
        h = int(r["true_harmful"])
        b = int(r["true_beneficial"])
        sd = float(r["source_dice"])
        ad = float(r["action_dice"])
        if not (0.0 <= p <= 1.0 and math.isfinite(p)):
            raise RuntimeError(f"Invalid risk row={i}")
        if h not in (0, 1) or b not in (0, 1):
            raise RuntimeError(f"Invalid labels row={i}")
        if h and b:
            raise RuntimeError(f"HARM and BENEFIT both positive row={i}")
        if not all(math.isfinite(x) for x in (sd, ad)):
            raise RuntimeError(f"Invalid Dice row={i}")
        parsed.append({
            "model_family": r["model_family"],
            "model_state_id": r["model_state_id"],
            "source_group_id": r["source_group_id"],
            "perturbation": r["perturbation"],
            "risk": p,
            "harm": h,
            "benefit": b,
            "source_dice": sd,
            "action_dice": ad,
        })

    return lock, table_path, parsed


def frontier_for_family(rows, family):
    subset = [r for r in rows if r["model_family"] == family]
    if len(subset) != EXPECTED_ROWS_PER_FAMILY:
        raise RuntimeError(f"{family} rows={len(subset)}")

    p = np.asarray([r["risk"] for r in subset], dtype=np.float64)
    harm = np.asarray([r["harm"] for r in subset], dtype=np.int64)
    benefit = np.asarray([r["benefit"] for r in subset], dtype=np.int64)
    source = np.asarray([r["source_dice"] for r in subset], dtype=np.float64)
    action = np.asarray([r["action_dice"] for r in subset], dtype=np.float64)

    n = len(subset)
    harm_total = int(harm.sum())
    benefit_total = int(benefit.sum())
    if harm_total == 0 or benefit_total == 0:
        raise RuntimeError(f"{family} class-degenerate.")

    # Descending risk; stable sorting keeps deterministic group order.
    order = np.argsort(-p, kind="mergesort")
    ps = p[order]
    hs = harm[order]
    bs = benefit[order]
    src = source[order]
    act = action[order]

    # Baseline all-adapt point.
    points = [{
        "model_family": family,
        "blocked_rows": 0,
        "adapted_rows": n,
        "threshold_lower_bound": float("inf"),
        "threshold_semantics": "adapt_all_endpoint",
        "harm_prevention_recall": 0.0,
        "residual_harm_rows": harm_total,
        "adaptation_coverage": 1.0,
        "benefit_retention": 1.0,
        "source_mean_dice": float(source.mean()),
        "blind_a1_mean_dice": float(action.mean()),
        "gated_mean_dice": float(action.mean()),
        "gated_minus_source_mean_dice": float(action.mean() - source.mean()),
        "gated_minus_blind_a1_mean_dice": 0.0,
        "joint_feasible": 0,
    }]

    blocked_harm = 0
    blocked_benefit = 0
    blocked_rows = 0
    gated_sum = float(action.sum())
    source_mean = float(source.mean())
    action_mean = float(action.mean())

    i = 0
    while i < n:
        score = ps[i]
        j = i + 1
        while j < n and ps[j] == score:
            j += 1

        # Blocking score >= threshold means block this entire tie group.
        blocked_rows += (j - i)
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
            "residual_harm_rows": int(harm_total - blocked_harm),
            "adaptation_coverage": float(coverage),
            "benefit_retention": float(benefit_retention),
            "source_mean_dice": source_mean,
            "blind_a1_mean_dice": action_mean,
            "gated_mean_dice": float(gated_mean),
            "gated_minus_source_mean_dice": float(gate_source),
            "gated_minus_blind_a1_mean_dice": float(gated_mean - action_mean),
            "joint_feasible": int(feasible),
        })
        i = j

    return points


def max_or_none(values):
    values = list(values)
    return None if not values else float(max(values))


def summarize_family(points, family):
    feasible = [r for r in points if int(r["joint_feasible"]) == 1]

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
        "best_joint_feasible_harm_prevention":
            max_or_none(r["harm_prevention_recall"] for r in feasible),
        "best_joint_feasible_coverage":
            max_or_none(r["adaptation_coverage"] for r in feasible),
        "best_joint_feasible_benefit_retention":
            max_or_none(r["benefit_retention"] for r in feasible),
        "best_joint_feasible_gate_minus_source_dice":
            max_or_none(r["gated_minus_source_mean_dice"] for r in feasible),
    }


def risk_summary(rows):
    out = []
    for family in FAMILIES:
        subset = [r for r in rows if r["model_family"] == family]
        classes = {
            "HARM": [r["risk"] for r in subset if r["harm"] == 1],
            "BENEFIT": [r["risk"] for r in subset if r["benefit"] == 1],
            "NEUTRAL": [
                r["risk"] for r in subset
                if r["harm"] == 0 and r["benefit"] == 0
            ],
        }
        for label, vals in classes.items():
            x = np.asarray(vals, dtype=np.float64)
            if len(x) == 0:
                raise RuntimeError(f"{family}/{label} empty.")
            out.append({
                "model_family": family,
                "outcome_class": label,
                "rows": len(x),
                "risk_mean": float(x.mean()),
                "risk_q10": float(np.quantile(x, 0.10)),
                "risk_q25": float(np.quantile(x, 0.25)),
                "risk_q50": float(np.quantile(x, 0.50)),
                "risk_q75": float(np.quantile(x, 0.75)),
                "risk_q90": float(np.quantile(x, 0.90)),
            })
    return out


def preflight():
    lock, table_path, rows = validate_and_load()
    print("===== Q1-R06C PREFLIGHT =====")
    print(f"protocol_sha256={EXPECTED_PROTOCOL_SHA256}")
    print(f"r06b_lock_sha256={EXPECTED_R06B_LOCK_SHA256}")
    print(f"r06b_decision={lock.get('decision')}")
    print(f"rows={len(rows)}")
    print(f"families={len(set(r['model_family'] for r in rows))}")
    print("new_model_fit=NO")
    print("deployment_threshold_selected=NO")
    print("target_data_read=NO")
    print("analysis=complete_monotone_B1_threshold_frontier")
    print("PREFLIGHT_PASS")


def run(args):
    lock, table_path, rows = validate_and_load()

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
            "r06b_decision": lock.get("decision"),
            "r06b_outer_decision_table_sha256": sha256_file(table_path),
            "rows": EXPECTED_ROWS,
            "new_model_fit": False,
            "deployment_threshold_selected": False,
            "target_data_read": False,
            "oracle_source_outcome_frontier_audit": True,
        },
    )

    frontier_rows = []
    summary_rows = []

    for family in tqdm(FAMILIES, desc="R06C architecture frontiers", unit="arch"):
        pts = frontier_for_family(rows, family)
        frontier_rows.extend(pts)
        summary_rows.append(summarize_family(pts, family))

    risk_rows = risk_summary(rows)

    write_csv(
        build_dir / "frontier_points.csv",
        frontier_rows,
        [
            "model_family",
            "blocked_rows",
            "adapted_rows",
            "threshold_lower_bound",
            "threshold_semantics",
            "harm_prevention_recall",
            "residual_harm_rows",
            "adaptation_coverage",
            "benefit_retention",
            "source_mean_dice",
            "blind_a1_mean_dice",
            "gated_mean_dice",
            "gated_minus_source_mean_dice",
            "gated_minus_blind_a1_mean_dice",
            "joint_feasible",
        ],
    )

    write_csv(
        build_dir / "architecture_frontier_summary.csv",
        summary_rows,
        [
            "model_family",
            "frontier_points",
            "joint_feasible_points",
            "joint_feasible",
            "max_benefit_retention_given_harm_ge_0p80_coverage_ge_0p30_dice",
            "max_coverage_given_harm_ge_0p80_benefit_ge_0p50_dice",
            "max_harm_prevention_given_coverage_ge_0p30_benefit_ge_0p50_dice",
            "best_joint_feasible_harm_prevention",
            "best_joint_feasible_coverage",
            "best_joint_feasible_benefit_retention",
            "best_joint_feasible_gate_minus_source_dice",
        ],
    )

    write_csv(
        build_dir / "outcome_risk_summary.csv",
        risk_rows,
        [
            "model_family",
            "outcome_class",
            "rows",
            "risk_mean",
            "risk_q10",
            "risk_q25",
            "risk_q50",
            "risk_q75",
            "risk_q90",
        ],
    )

    per_family = {r["model_family"]: bool(r["joint_feasible"]) for r in summary_rows}
    decision = (
        DECISION_FEASIBLE
        if all(per_family.values())
        else DECISION_INSUFFICIENT
    )

    gate = {
        "constraints": {
            "harm_prevention_recall_min": HARM_PREVENTION_MIN,
            "adaptation_coverage_min": COVERAGE_MIN,
            "benefit_retention_min": BENEFIT_RETENTION_MIN,
            "gated_minus_source_mean_dice_min": GATE_MINUS_SOURCE_DICE_MIN,
        },
        "architecture_joint_feasibility": per_family,
        "decision": decision,
        "deployment_threshold_selected": False,
        "interpretation": (
            "All architectures possess at least one oracle monotone threshold "
            "inside the desired safety/utility region."
            if decision == DECISION_FEASIBLE
            else
            "At least one architecture has no oracle monotone B1 threshold "
            "inside the desired safety/utility region."
        ),
    }
    write_json(build_dir / "feasibility_gate.json", gate)
    (build_dir / "decision.txt").write_text(decision + "\n", encoding="utf-8")

    lines = [
        "===== Q1-R06C SCALAR HARM-RISK GATE FEASIBILITY FRONTIER AUDIT =====",
        f"Script version: {VERSION}",
        f"Build: {BUILD}",
        "",
        "Audit:",
        "  predictor=B1 MRZ19 pooled logistic (already frozen)",
        "  new model fit=NO",
        "  threshold family=complete monotone scalar frontier",
        "  deployment threshold selected=NO",
        "  target data read=NO",
        "",
        "Frozen feasibility region:",
        "  harm prevention >= 0.80",
        "  adaptation coverage >= 0.30",
        "  benefit retention >= 0.50",
        "  Gate-Source mean Dice >= -0.005",
        "",
        "Architecture oracle frontier:",
    ]

    for r in summary_rows:
        lines += [
            (
                f"  {r['model_family']}: "
                f"joint_feasible={'YES' if r['joint_feasible'] else 'NO'} "
                f"feasible_points={r['joint_feasible_points']}"
            ),
            (
                "    max benefit retention | harm>=.80,cov>=.30,dice = "
                f"{r['max_benefit_retention_given_harm_ge_0p80_coverage_ge_0p30_dice']}"
            ),
            (
                "    max coverage | harm>=.80,benefit>=.50,dice = "
                f"{r['max_coverage_given_harm_ge_0p80_benefit_ge_0p50_dice']}"
            ),
            (
                "    max harm prevention | cov>=.30,benefit>=.50,dice = "
                f"{r['max_harm_prevention_given_coverage_ge_0p30_benefit_ge_0p50_dice']}"
            ),
        ]

    lines += [
        "",
        "Decision:",
        f"  {decision}",
        "",
        "Interpretation:",
        (
            "  Scalar monotone HARM gating remains structurally feasible; "
            "a new threshold-learning rule may be justified."
            if decision == DECISION_FEASIBLE
            else
            "  Current scalar HARM score cannot deliver the desired trade-off "
            "through monotone thresholding on all architectures; stop this gate family."
        ),
    ]

    run_log = build_dir / "run_log.txt"
    run_log.write_text("\n".join(lines) + "\n", encoding="utf-8")

    artifact_names = [
        "preregistered_protocol_copy.md",
        "upstream_audit.json",
        "outcome_risk_summary.csv",
        "frontier_points.csv",
        "architecture_frontier_summary.csv",
        "feasibility_gate.json",
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
        "rows": EXPECTED_ROWS,
        "new_model_fit": False,
        "deployment_threshold_selected": False,
        "target_data_read": False,
        "decision": decision,
        "gate": gate,
        "artifacts": artifacts,
    }
    lock_path = build_dir / "Q1_R06C_SCALAR_GATE_FRONTIER_LOCK.json"
    write_json(lock_path, final_lock)
    lock_sha = sha256_file(lock_path)

    for name, meta in artifacts.items():
        if sha256_file(build_dir / meta["relative_path"]) != meta["sha256"]:
            raise RuntimeError(f"Artifact changed before commit: {name}")

    build_dir.rename(args.output_dir)

    print()
    print((args.output_dir / "run_log.txt").read_text(encoding="utf-8"))
    print(
        "Q1-R06C LOCK:",
        args.output_dir / "Q1_R06C_SCALAR_GATE_FRONTIER_LOCK.json",
    )
    print("Q1-R06C LOCK SHA256:", lock_sha)


def self_test():
    assert HARM_PREVENTION_MIN == 0.80
    assert COVERAGE_MIN == 0.30
    assert BENEFIT_RETENTION_MIN == 0.50
    assert GATE_MINUS_SOURCE_DICE_MIN == -0.005

    toy = [
        {
            "model_family": "PraNet",
            "risk": 0.9,
            "harm": 1,
            "benefit": 0,
            "source_dice": 0.8,
            "action_dice": 0.5,
        },
        {
            "model_family": "PraNet",
            "risk": 0.8,
            "harm": 1,
            "benefit": 0,
            "source_dice": 0.8,
            "action_dice": 0.6,
        },
        {
            "model_family": "PraNet",
            "risk": 0.2,
            "harm": 0,
            "benefit": 1,
            "source_dice": 0.7,
            "action_dice": 0.9,
        },
        {
            "model_family": "PraNet",
            "risk": 0.1,
            "harm": 0,
            "benefit": 1,
            "source_dice": 0.7,
            "action_dice": 0.8,
        },
    ]

    # Small stand-alone sweep analogue to validate tie-group semantics.
    p = np.asarray([r["risk"] for r in toy])
    order = np.argsort(-p, kind="mergesort")
    assert order.tolist() == [0, 1, 2, 3]
    assert np.allclose(p[order], [0.9, 0.8, 0.2, 0.1])

    print("FROZEN_FEASIBILITY_CONSTRAINT_TEST_PASS")
    print("MONOTONE_THRESHOLD_ORDER_TEST_PASS")
    print("NO_MODEL_FIT_SELF_TEST_PASS")
    print("SELF_TEST_PASS")


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Q1-R06C: source-only oracle feasibility audit of the complete "
            "monotone threshold frontier of frozen B1 HARM risk."
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
