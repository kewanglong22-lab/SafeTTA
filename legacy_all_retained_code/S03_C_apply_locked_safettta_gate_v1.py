#!/usr/bin/env python
# -*- coding: utf-8 -*-

r"""
S03-C: Apply the already-locked SOURCE-SIDE SafeTTA-v1 gate to target
no-label features, lock decisions, and only then merge frozen D1 DeltaDice.

NO model fitting occurs in this script.
NO target GT mask file is opened by this script.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


VERSION = "2026-08-17-S03-C-v1"
BUILD = "S03_C_LOCK_GATE_DECISIONS_BEFORE_TARGET_LABEL_MERGE"

ROOT = Path(r"F:\MEDSEG_SAFETTA")

PROTOCOL = ROOT / "docs" / "S03_C_locked_target_application_protocol_v1.md"
EXPECTED_PROTOCOL_SHA256 = "b9d841d8fffbce8898396a1a26378bc1809972d5a80ae88f79407afb7646e751"

GATE_DIR = ROOT / "outputs" / "S03_B_source_side_safettta_gate_v1"
GATE_ARTIFACT = GATE_DIR / "SafeTTA_v1_SOURCE_LOCKED_GATE.json"
EXPECTED_GATE_SHA256 = "a71696fd3730dae19bda7a66405a5314022c1eb11acca6604ddcb7ccd1b82e2f"

S02_DIR = ROOT / "outputs" / "S02_B_unlabeled_harm_signal_feasibility_v1"
S02_FEATURE_LOCK = S02_DIR / "NO_LABEL_FEATURE_LOCK.json"
EXPECTED_S02_FEATURE_LOCK_SHA256 = "dba040a0b35b6f227cb5ebb03be7cad996bab088f693402d2abbfe53032b08c9"
S02_FEATURE_TABLE = S02_DIR / "NO_LABEL_FEATURES_LOCKED.csv"
EXPECTED_S02_FEATURE_TABLE_SHA256 = "70d55e55243fb3aa1ed7c114f2ab614f034d725b1365421f1d7c629a0bfa0158"

D1_DIR = ROOT / "outputs" / "S01_D1_tta_feasibility_v1"
D1_DELTA = D1_DIR / "per_image_delta_dice.csv"
D1_LOCK = D1_DIR / "NO_LABEL_TTA_LOCK.json"
EXPECTED_D1_LOCK_SHA256 = "cafb77c03bbc8d3a2a5bc19ce69231b55daf9c73234f8d12431c99c179b743ae"

OUTPUT_DIR = ROOT / "outputs" / "S03_C_locked_safettta_target_application_v1"

SEEDS = (20260817, 20260818, 20260819)
UNSEEN_DOMAINS = ("CVC-ColonDB", "CVC-300", "ETIS-LaribPolypDB")

EXPECTED_UNSEEN = {
    "CVC-ColonDB": 380 * 3,
    "CVC-300": 60 * 3,
    "ETIS-LaribPolypDB": 196 * 3,
}
EXPECTED_POOLED_UNSEEN = 1908

HARM_THRESHOLD = -0.02
BENEFIT_THRESHOLD = +0.02

PASS_HARM_PREVENTION = 0.50
PASS_BENEFIT_RETENTION = 0.30
PASS_ACCEPTANCE_MIN = 0.10
PASS_ACCEPTANCE_MAX = 0.90


def file_sha256(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def read_csv(path: Path):
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return list(reader), reader.fieldnames or []


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def stable_sigmoid_scalar(z: float) -> float:
    if z >= 0:
        return float(1.0 / (1.0 + math.exp(-z)))
    ez = math.exp(z)
    return float(ez / (1.0 + ez))


def validate_preflight():
    checks = [
        (PROTOCOL, EXPECTED_PROTOCOL_SHA256, "S03-C protocol"),
        (GATE_ARTIFACT, EXPECTED_GATE_SHA256, "S03-B gate"),
        (S02_FEATURE_LOCK, EXPECTED_S02_FEATURE_LOCK_SHA256, "S02 feature lock"),
        (S02_FEATURE_TABLE, EXPECTED_S02_FEATURE_TABLE_SHA256, "S02 feature table"),
        (D1_LOCK, EXPECTED_D1_LOCK_SHA256, "D1 no-label TTA lock"),
    ]
    actual = {}
    for path, expected, label in checks:
        if not path.exists():
            raise FileNotFoundError(path)
        sha = file_sha256(path)
        if sha.lower() != expected.lower():
            raise RuntimeError(
                f"{label} SHA mismatch.\nExpected: {expected}\nActual: {sha}"
            )
        actual[label] = sha
    return actual


def load_gate():
    gate = json.loads(GATE_ARTIFACT.read_text(encoding="utf-8"))

    required = {
        "feature_names",
        "scaler_mean",
        "scaler_scale",
        "logistic_coef",
        "logistic_intercept",
        "harm_threshold_tau",
        "decision_rule",
    }
    missing = sorted(required - set(gate))
    if missing:
        raise RuntimeError(f"Gate artifact missing fields: {missing}")

    names = list(gate["feature_names"])
    mean = np.asarray(gate["scaler_mean"], dtype=np.float64)
    scale = np.asarray(gate["scaler_scale"], dtype=np.float64)
    coef = np.asarray(gate["logistic_coef"], dtype=np.float64)
    intercept = float(gate["logistic_intercept"])
    tau = float(gate["harm_threshold_tau"])

    n = len(names)
    if not (n == 24 == len(mean) == len(scale) == len(coef)):
        raise RuntimeError(
            f"Gate dimension mismatch: names={n}, mean={len(mean)}, "
            f"scale={len(scale)}, coef={len(coef)}"
        )
    if np.any(~np.isfinite(mean)) or np.any(~np.isfinite(scale)):
        raise RuntimeError("Non-finite scaler parameters.")
    if np.any(scale <= 0):
        raise RuntimeError("Scaler scale must be positive.")
    if np.any(~np.isfinite(coef)) or not np.isfinite(intercept):
        raise RuntimeError("Non-finite logistic parameters.")
    if not (0.0 <= tau <= 1.0):
        raise RuntimeError(f"Invalid gate threshold tau={tau}")

    return gate, names, mean, scale, coef, intercept, tau


def apply_gate_to_no_label_features(
    feature_rows,
    feature_names,
    mean,
    scale,
    coef,
    intercept,
    tau,
):
    decisions = []

    required_meta = {"seed", "sample_id", "role", "dataset"}
    for idx, r in enumerate(feature_rows):
        if not required_meta.issubset(r):
            raise RuntimeError(f"Feature metadata missing on row {idx}")

        try:
            x = np.asarray([float(r[f]) for f in feature_names], dtype=np.float64)
        except KeyError as exc:
            raise RuntimeError(f"Frozen feature missing: {exc}") from exc

        if np.any(~np.isfinite(x)):
            raise RuntimeError(
                f"Non-finite target feature seed={r['seed']} sample={r['sample_id']}"
            )

        xs = (x - mean) / scale
        logit = float(np.dot(coef, xs) + intercept)
        p_harm = stable_sigmoid_scalar(logit)
        accepted = int(p_harm < tau)

        decisions.append({
            "seed": int(r["seed"]),
            "sample_id": r["sample_id"],
            "role": r["role"],
            "dataset": r["dataset"],
            "p_harm": f"{p_harm:.12f}",
            "tau": f"{tau:.12f}",
            "tent_accepted": accepted,
            "decision": "ACCEPT_TENT" if accepted else "REJECT_TENT_USE_SOURCE",
        })

    keys = [(r["seed"], r["sample_id"]) for r in decisions]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Duplicate target gate-decision keys.")

    return decisions


def load_frozen_tent_delta():
    """
    PHASE B only.

    Reads the already-frozen D1 result table. This script never opens GT masks.
    """
    if not D1_DELTA.exists():
        raise FileNotFoundError(D1_DELTA)

    rows, fields = read_csv(D1_DELTA)
    required = {
        "method", "seed", "sample_id", "role", "dataset",
        "source_dice", "tta_dice", "delta_dice",
    }
    missing = sorted(required - set(fields))
    if missing:
        raise RuntimeError(f"D1 DeltaDice table missing fields: {missing}")

    out = {}
    for r in rows:
        if r["method"] != "TENT":
            continue

        seed = int(r["seed"])
        key = (seed, r["sample_id"])
        if key in out:
            raise RuntimeError(f"Duplicate D1 TENT row: {key}")

        source_dice = float(r["source_dice"])
        tent_dice = float(r["tta_dice"])
        delta = float(r["delta_dice"])

        if abs((tent_dice - source_dice) - delta) > 2e-8:
            raise RuntimeError(f"D1 DeltaDice arithmetic mismatch: {key}")

        out[key] = {
            "role": r["role"],
            "dataset": r["dataset"],
            "source_dice": source_dice,
            "tent_dice": tent_dice,
            "delta_dice": delta,
        }

    if len(out) != 798 * len(SEEDS):
        raise RuntimeError(
            f"D1 TENT row count mismatch: expected={798*len(SEEDS)}, actual={len(out)}"
        )

    return out


def merge_and_evaluate(decisions, labels):
    rows = []

    for d in decisions:
        key = (int(d["seed"]), d["sample_id"])
        lab = labels.get(key)
        if lab is None:
            raise RuntimeError(f"Missing D1 TENT label row for {key}")
        if lab["role"] != d["role"] or lab["dataset"] != d["dataset"]:
            raise RuntimeError(f"Decision/label metadata mismatch for {key}")

        accepted = int(d["tent_accepted"]) == 1
        source = lab["source_dice"]
        tent = lab["tent_dice"]
        tent_delta = lab["delta_dice"]

        safe = tent if accepted else source
        safe_delta = safe - source
        oracle = max(source, tent)

        tent_harmful = int(tent_delta <= HARM_THRESHOLD)
        tent_beneficial = int(tent_delta >= BENEFIT_THRESHOLD)
        safe_harmful = int(safe_delta <= HARM_THRESHOLD)
        safe_beneficial = int(safe_delta >= BENEFIT_THRESHOLD)

        rows.append({
            **d,
            "source_dice": f"{source:.10f}",
            "tent_dice": f"{tent:.10f}",
            "safettta_dice": f"{safe:.10f}",
            "oracle_dice": f"{oracle:.10f}",
            "tent_delta_dice": f"{tent_delta:.10f}",
            "safettta_delta_dice": f"{safe_delta:.10f}",
            "tent_harmful": tent_harmful,
            "tent_beneficial": tent_beneficial,
            "safettta_harmful": safe_harmful,
            "safettta_beneficial": safe_beneficial,
            "harm_prevented": int(tent_harmful and not accepted),
            "benefit_retained": int(tent_beneficial and accepted),
        })

    return rows


def summarize_group(group, label):
    n = len(group)
    if n == 0:
        raise RuntimeError(f"Empty summary group: {label}")

    source = np.asarray([float(r["source_dice"]) for r in group])
    tent = np.asarray([float(r["tent_dice"]) for r in group])
    safe = np.asarray([float(r["safettta_dice"]) for r in group])
    oracle = np.asarray([float(r["oracle_dice"]) for r in group])

    accepted = np.asarray([int(r["tent_accepted"]) for r in group], dtype=np.int64)
    tent_harm = np.asarray([int(r["tent_harmful"]) for r in group], dtype=np.int64)
    tent_benefit = np.asarray([int(r["tent_beneficial"]) for r in group], dtype=np.int64)
    safe_harm = np.asarray([int(r["safettta_harmful"]) for r in group], dtype=np.int64)

    harm_n = int(tent_harm.sum())
    benefit_n = int(tent_benefit.sum())

    harm_prevention = (
        float(np.sum((tent_harm == 1) & (accepted == 0))) / harm_n
        if harm_n > 0 else float("nan")
    )
    benefit_retention = (
        float(np.sum((tent_benefit == 1) & (accepted == 1))) / benefit_n
        if benefit_n > 0 else float("nan")
    )

    return {
        "group": label,
        "n": n,
        "source_mean_dice": float(source.mean()),
        "tent_mean_dice": float(tent.mean()),
        "safettta_mean_dice": float(safe.mean()),
        "oracle_mean_dice": float(oracle.mean()),
        "tent_mean_delta": float((tent - source).mean()),
        "safettta_mean_delta": float((safe - source).mean()),
        "tent_harmful_n": harm_n,
        "tent_harmful_fraction": float(tent_harm.mean()),
        "safettta_harmful_n": int(safe_harm.sum()),
        "safettta_harmful_fraction": float(safe_harm.mean()),
        "harm_prevention_rate": harm_prevention,
        "tent_beneficial_n": benefit_n,
        "benefit_retention_rate": benefit_retention,
        "tent_acceptance_rate": float(accepted.mean()),
        "oracle_gap": float(oracle.mean() - safe.mean()),
    }


def build_summaries(rows):
    unseen = [r for r in rows if r["role"] == "unseen_locked"]
    if len(unseen) != EXPECTED_POOLED_UNSEEN:
        raise RuntimeError(
            f"Pooled unseen count mismatch: expected={EXPECTED_POOLED_UNSEEN}, "
            f"actual={len(unseen)}"
        )

    summaries = [summarize_group(unseen, "POOLED_UNSEEN")]

    for domain in UNSEEN_DOMAINS:
        group = [
            r for r in unseen
            if r["dataset"] == domain
        ]
        expected = EXPECTED_UNSEEN[domain]
        if len(group) != expected:
            raise RuntimeError(
                f"Domain count mismatch {domain}: expected={expected}, actual={len(group)}"
            )
        summaries.append(summarize_group(group, domain))

    for seed in SEEDS:
        group = [
            r for r in unseen
            if int(r["seed"]) == seed
        ]
        if len(group) != 636:
            raise RuntimeError(
                f"Seed unseen count mismatch {seed}: {len(group)}"
            )
        summaries.append(summarize_group(group, f"SEED_{seed}"))

    return summaries


def evaluate_primary_gate(pooled):
    pass_mean = (
        pooled["safettta_mean_dice"] + 1e-12
        >= pooled["source_mean_dice"]
    )
    pass_harm = (
        pooled["harm_prevention_rate"] + 1e-12
        >= PASS_HARM_PREVENTION
    )
    pass_benefit = (
        pooled["benefit_retention_rate"] + 1e-12
        >= PASS_BENEFIT_RETENTION
    )
    pass_acceptance = (
        pooled["tent_acceptance_rate"] + 1e-12 >= PASS_ACCEPTANCE_MIN
        and pooled["tent_acceptance_rate"] - 1e-12 <= PASS_ACCEPTANCE_MAX
    )

    passed = pass_mean and pass_harm and pass_benefit and pass_acceptance

    return {
        "mean_non_degradation": pass_mean,
        "harm_prevention": pass_harm,
        "benefit_retention": pass_benefit,
        "nontrivial_acceptance": pass_acceptance,
        "all_primary_criteria": passed,
        "decision": (
            "S03_C_STRONG_TARGET_TRANSFER_PASS"
            if passed
            else "S03_C_STRONG_TARGET_TRANSFER_NOT_MET"
        ),
    }


def run(args):
    preflight = validate_preflight()
    gate, feature_names, mean, scale, coef, intercept, tau = load_gate()

    if args.output_dir.exists():
        raise FileExistsError(
            f"Final S03-C result already exists: {args.output_dir}"
        )

    build_dir = Path(str(args.output_dir) + "__building")
    if build_dir.exists():
        if not args.technical_rerun:
            raise FileExistsError(
                f"Partial S03-C build exists: {build_dir}. "
                "Use --technical-rerun only after a technical failure."
            )
        shutil.rmtree(build_dir)

    build_dir.mkdir(parents=True, exist_ok=False)

    print(f"[BUILD] {VERSION} | {BUILD}")
    print(f"Gate artifact SHA256: {EXPECTED_GATE_SHA256}")
    print(f"Frozen tau: {tau:.10f}")
    print("Phase A target DeltaDice read=NO")
    print("Phase A target GT masks opened=NO")
    print()

    # ----------------------------------------------------------
    # PHASE A: no-label gate decisions.
    # ----------------------------------------------------------
    feature_rows, feature_fields = read_csv(S02_FEATURE_TABLE)

    required_feature_fields = (
        {"seed", "sample_id", "role", "dataset"} | set(feature_names)
    )
    missing = sorted(required_feature_fields - set(feature_fields))
    if missing:
        raise RuntimeError(f"S02 no-label feature table missing fields: {missing}")

    decisions = apply_gate_to_no_label_features(
        feature_rows=feature_rows,
        feature_names=feature_names,
        mean=mean,
        scale=scale,
        coef=coef,
        intercept=intercept,
        tau=tau,
    )

    if len(decisions) != 2394:
        raise RuntimeError(
            f"Gate-decision row count mismatch: expected=2394, actual={len(decisions)}"
        )

    decision_csv = build_dir / "TARGET_NO_LABEL_GATE_DECISIONS_LOCKED.csv"
    decision_fields = [
        "seed", "sample_id", "role", "dataset",
        "p_harm", "tau", "tent_accepted", "decision",
    ]
    write_csv(decision_csv, decisions, decision_fields)
    decision_csv_sha = file_sha256(decision_csv)

    decision_lock = {
        "script_version": VERSION,
        "build": BUILD,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "gate_artifact_sha256": EXPECTED_GATE_SHA256,
        "s02_feature_lock_sha256": EXPECTED_S02_FEATURE_LOCK_SHA256,
        "s02_feature_table_sha256": EXPECTED_S02_FEATURE_TABLE_SHA256,
        "gate_decision_table_sha256": decision_csv_sha,
        "gate_decision_rows": len(decisions),
        "feature_names": feature_names,
        "harm_threshold_tau": tau,
        "target_delta_dice_read_before_gate_decision_lock": False,
        "target_gt_masks_opened": False,
        "model_refit": False,
        "threshold_recalibrated": False,
    }

    decision_lock_path = build_dir / "TARGET_NO_LABEL_GATE_DECISION_LOCK.json"
    decision_lock_path.write_text(
        json.dumps(decision_lock, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    decision_lock_sha = file_sha256(decision_lock_path)

    print("===== S03-C TARGET NO-LABEL GATE DECISION LOCK =====")
    print(f"Rows: {len(decisions)}")
    print(f"Decision CSV SHA256: {decision_csv_sha}")
    print(f"Decision lock SHA256: {decision_lock_sha}")
    print("Target DeltaDice read before lock: NO")
    print("Target GT masks opened: NO")
    print("Model refit: NO")
    print("Threshold recalibrated: NO")
    print()

    # Re-verify before Phase B.
    if file_sha256(decision_csv) != decision_csv_sha:
        raise RuntimeError("Gate decision CSV changed before label merge.")
    if file_sha256(decision_lock_path) != decision_lock_sha:
        raise RuntimeError("Gate decision lock changed before label merge.")

    # ----------------------------------------------------------
    # PHASE B: merge already-frozen D1 label-derived metrics.
    # ----------------------------------------------------------
    print("===== BEGIN FROZEN D1 DELTADICE MERGE =====")
    d1_sha = file_sha256(D1_DELTA)
    labels = load_frozen_tent_delta()

    merged = merge_and_evaluate(decisions, labels)
    summaries = build_summaries(merged)

    pooled = next(s for s in summaries if s["group"] == "POOLED_UNSEEN")
    primary = evaluate_primary_gate(pooled)

    merged_fields = decision_fields + [
        "source_dice", "tent_dice", "safettta_dice", "oracle_dice",
        "tent_delta_dice", "safettta_delta_dice",
        "tent_harmful", "tent_beneficial",
        "safettta_harmful", "safettta_beneficial",
        "harm_prevented", "benefit_retained",
    ]
    write_csv(
        build_dir / "per_image_seed_target_evaluation.csv",
        merged,
        merged_fields,
    )

    summary_fields = [
        "group", "n",
        "source_mean_dice", "tent_mean_dice",
        "safettta_mean_dice", "oracle_mean_dice",
        "tent_mean_delta", "safettta_mean_delta",
        "tent_harmful_n", "tent_harmful_fraction",
        "safettta_harmful_n", "safettta_harmful_fraction",
        "harm_prevention_rate",
        "tent_beneficial_n", "benefit_retention_rate",
        "tent_acceptance_rate", "oracle_gap",
    ]
    write_csv(
        build_dir / "target_summary.csv",
        summaries,
        summary_fields,
    )

    result = {
        "script_version": VERSION,
        "build": BUILD,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "gate_artifact_sha256": EXPECTED_GATE_SHA256,
        "target_no_label_gate_decision_lock_sha256": decision_lock_sha,
        "d1_delta_table_sha256_at_merge": d1_sha,
        "model_refit": False,
        "threshold_recalibrated": False,
        "target_gt_masks_opened_by_s03c": False,
        "primary_pooled_unseen": pooled,
        "primary_criteria": {
            "safettta_mean_dice_ge_source": True,
            "harm_prevention_ge": PASS_HARM_PREVENTION,
            "benefit_retention_ge": PASS_BENEFIT_RETENTION,
            "acceptance_range": [
                PASS_ACCEPTANCE_MIN, PASS_ACCEPTANCE_MAX
            ],
        },
        "criterion_results": primary,
        "decision": primary["decision"],
    }
    (build_dir / "decision.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "===== S03-C LOCKED SAFETTA TARGET APPLICATION =====",
        f"Script version: {VERSION}",
        f"Build: {BUILD}",
        "",
        "No-label deployment lock:",
        f"  gate artifact SHA256={EXPECTED_GATE_SHA256}",
        f"  tau={tau:.10f}",
        f"  gate decision lock SHA256={decision_lock_sha}",
        "  model refit=NO",
        "  threshold recalibration=NO",
        "  target DeltaDice read before gate lock=NO",
        "  target GT masks opened by S03-C=NO",
        "",
        "Primary pooled unseen:",
        f"  N={pooled['n']}",
        f"  Source-Only mean Dice={pooled['source_mean_dice']:.6f}",
        f"  Always-TENT mean Dice={pooled['tent_mean_dice']:.6f}",
        f"  SafeTTA-v1 mean Dice={pooled['safettta_mean_dice']:.6f}",
        f"  Oracle mean Dice={pooled['oracle_mean_dice']:.6f}",
        f"  SafeTTA mean DeltaDice={pooled['safettta_mean_delta']:+.6f}",
        f"  Always-TENT harmful fraction={pooled['tent_harmful_fraction']:.6f}",
        f"  SafeTTA harmful fraction={pooled['safettta_harmful_fraction']:.6f}",
        f"  harm prevention rate={pooled['harm_prevention_rate']:.6f}",
        f"  TENT beneficial cases={pooled['tent_beneficial_n']}",
        f"  benefit retention rate={pooled['benefit_retention_rate']:.6f}",
        f"  TENT acceptance rate={pooled['tent_acceptance_rate']:.6f}",
        f"  Oracle gap={pooled['oracle_gap']:.6f}",
        "",
        "Frozen primary criteria:",
        f"  A mean Dice >= Source-Only: {primary['mean_non_degradation']}",
        f"  B harm prevention >= {PASS_HARM_PREVENTION:.2f}: "
        f"{primary['harm_prevention']}",
        f"  C benefit retention >= {PASS_BENEFIT_RETENTION:.2f}: "
        f"{primary['benefit_retention']}",
        f"  D acceptance in [{PASS_ACCEPTANCE_MIN:.2f}, "
        f"{PASS_ACCEPTANCE_MAX:.2f}]: {primary['nontrivial_acceptance']}",
        "",
        "Unseen domain summaries:",
    ]

    for s in summaries:
        if s["group"] in UNSEEN_DOMAINS:
            lines.append(
                f"  {s['group']:22s} "
                f"Src={s['source_mean_dice']:.6f} "
                f"TENT={s['tent_mean_dice']:.6f} "
                f"Safe={s['safettta_mean_dice']:.6f} "
                f"HarmPrev={s['harm_prevention_rate']:.3f} "
                f"BenefitRet={s['benefit_retention_rate']:.3f} "
                f"Accept={s['tent_acceptance_rate']:.3f}"
            )

    lines += [
        "",
        "Seed summaries:",
    ]
    for s in summaries:
        if s["group"].startswith("SEED_"):
            lines.append(
                f"  {s['group']:14s} "
                f"Src={s['source_mean_dice']:.6f} "
                f"TENT={s['tent_mean_dice']:.6f} "
                f"Safe={s['safettta_mean_dice']:.6f} "
                f"HarmPrev={s['harm_prevention_rate']:.3f} "
                f"BenefitRet={s['benefit_retention_rate']:.3f} "
                f"Accept={s['tent_acceptance_rate']:.3f}"
            )

    lines += [
        "",
        f"Decision: {primary['decision']}",
    ]

    (build_dir / "summary.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    build_dir.rename(args.output_dir)

    print()
    print((args.output_dir / "summary.txt").read_text(encoding="utf-8"))
    print(f"[OK] Outputs: {args.output_dir}")


def self_test():
    # Manual exported-logistic computation.
    mean = np.zeros(3)
    scale = np.ones(3)
    coef = np.array([1.0, -0.5, 0.25])
    x = np.array([1.0, 2.0, 0.0])
    intercept = 0.1
    z = float(np.dot(coef, (x - mean) / scale) + intercept)
    p = stable_sigmoid_scalar(z)
    assert 0.0 < p < 1.0

    # Merge/safety logic.
    decisions = [{
        "seed": 20260817,
        "sample_id": "a",
        "role": "unseen_locked",
        "dataset": "CVC-ColonDB",
        "p_harm": "0.9",
        "tau": "0.46",
        "tent_accepted": 0,
        "decision": "REJECT_TENT_USE_SOURCE",
    }]
    labels = {
        (20260817, "a"): {
            "role": "unseen_locked",
            "dataset": "CVC-ColonDB",
            "source_dice": 0.8,
            "tent_dice": 0.6,
            "delta_dice": -0.2,
        }
    }
    merged = merge_and_evaluate(decisions, labels)
    assert float(merged[0]["safettta_dice"]) == 0.8
    assert int(merged[0]["harm_prevented"]) == 1
    assert int(merged[0]["safettta_harmful"]) == 0

    # Acceptance of beneficial case retains TENT.
    decisions2 = [{
        "seed": 20260817,
        "sample_id": "b",
        "role": "unseen_locked",
        "dataset": "CVC-300",
        "p_harm": "0.1",
        "tau": "0.46",
        "tent_accepted": 1,
        "decision": "ACCEPT_TENT",
    }]
    labels2 = {
        (20260817, "b"): {
            "role": "unseen_locked",
            "dataset": "CVC-300",
            "source_dice": 0.7,
            "tent_dice": 0.8,
            "delta_dice": 0.1,
        }
    }
    merged2 = merge_and_evaluate(decisions2, labels2)
    assert float(merged2[0]["safettta_dice"]) == 0.8
    assert int(merged2[0]["benefit_retained"]) == 1

    assert PASS_HARM_PREVENTION == 0.50
    assert PASS_BENEFIT_RETENTION == 0.30
    assert PASS_ACCEPTANCE_MIN == 0.10
    assert PASS_ACCEPTANCE_MAX == 0.90

    print("EXPORTED_LOGISTIC_TEST_PASS")
    print("HARM_REJECTION_LOGIC_TEST_PASS")
    print("BENEFIT_RETENTION_LOGIC_TEST_PASS")
    print("FROZEN_TARGET_GATE_TEST_PASS")
    print("SELF_TEST_PASS")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Apply locked source-side SafeTTA gate to locked target "
            "no-label features, lock decisions, then merge frozen D1 DeltaDice."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--technical-rerun",
        action="store_true",
        help="Only after a technical failure before final S03-C output exists.",
    )
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
