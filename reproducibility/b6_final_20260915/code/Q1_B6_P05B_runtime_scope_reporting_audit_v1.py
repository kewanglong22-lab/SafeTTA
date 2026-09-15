#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SafeTTA B6-P0-5B — Runtime scope reporting audit.

READ ONLY. No inference, no model loading, no CUDA timing, no GT, no science metrics.

Purpose
-------
P0-5A intentionally timed both:
  (a) the frozen semantic full-transition SafeTTA path, and
  (b) an auxiliary simple-geometry comparator.

Therefore its wall-clock `end_to_end_precommit_ms` includes the auxiliary
`simple_geometry_ms` computation. This audit derives paper-facing runtime scopes
row-by-row from the already frozen P0-5A timing table without rerunning anything.

Primary reporting scopes:
1) pipeline_total_with_aux_geometry_ms
2) semantic_full_transition_precommit_ms
   = end_to_end_precommit_ms - simple_geometry_ms
3) candidate_aware_incremental_after_source_available_ms
   = restore_before_candidate
     + candidate_update
     + candidate_reinference
     + restore_after_candidate
     + dino_conditioning_pca
     + safety_head
4) post_candidate_risk_scoring_ms
   = dino_conditioning_pca + safety_head
5) candidate_action_transaction_ms
   = restore_before_candidate
     + candidate_update
     + candidate_reinference
     + restore_after_candidate

This is a reporting-scope correction only. It does NOT change P0-5A measurements.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd


VERSION = "2026-09-15-B6-P05B-v1"

ROOT = Path(r"F:\MEDSEG_SAFETTA")
P05A_DIR = ROOT / "B6_P05A_memo_full_transition_runtime_benchmark_v1_fix3"

P05A_AUDIT = P05A_DIR / "B6_P05A_MEMO_FULL_TRANSITION_RUNTIME_AUDIT.json"
P05A_ROWS = P05A_DIR / "B6_P05A_MEMO_FULL_TRANSITION_RUNTIME_ROWS.csv"

EXPECTED_P05A_GATE = "PASS_B6_P05A_MEMO_FULL_TRANSITION_RUNTIME_BENCHMARK"
EXPECTED_MEASURED = 100
EXPECTED_WARMUP = 10

OUT_DIR = ROOT / "B6_P05B_runtime_scope_reporting_audit_v1"
PASS_GATE = "PASS_B6_P05B_RUNTIME_SCOPE_REPORTING_AUDIT"

REQUIRED_COLUMNS = [
    "sample_id",
    "measured",
    "image_load_preprocess_ms",
    "restore_before_source_ms",
    "source_inference_ms",
    "restore_before_candidate_ms",
    "candidate_update_ms",
    "candidate_reinference_ms",
    "restore_after_candidate_ms",
    "dino_conditioning_pca_ms",
    "safety_head_ms",
    "simple_geometry_ms",
    "end_to_end_precommit_ms",
    "baseline_cuda_allocated_MB",
    "peak_cuda_allocated_MB",
    "peak_cuda_reserved_MB",
    "peak_incremental_allocated_MB",
]


def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def stats(a: np.ndarray) -> Dict[str, float]:
    a = np.asarray(a, dtype=np.float64)
    return {
        "mean": float(a.mean()),
        "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "median": float(np.median(a)),
        "p95": float(np.quantile(a, 0.95)),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def load_and_verify() -> tuple[pd.DataFrame, Dict[str, Any]]:
    if not P05A_AUDIT.is_file():
        raise FileNotFoundError(P05A_AUDIT)
    if not P05A_ROWS.is_file():
        raise FileNotFoundError(P05A_ROWS)

    audit = json.loads(P05A_AUDIT.read_text(encoding="utf-8"))
    if audit.get("gate") != EXPECTED_P05A_GATE:
        raise RuntimeError(f"P05A gate changed: {audit.get('gate')}")

    d = pd.read_csv(P05A_ROWS, low_memory=False)
    missing = sorted(set(REQUIRED_COLUMNS) - set(d.columns))
    if missing:
        raise RuntimeError(f"P05A timing rows missing columns={missing}")

    if len(d) != EXPECTED_WARMUP + EXPECTED_MEASURED:
        raise RuntimeError(
            f"P05A row count={len(d)} expected={EXPECTED_WARMUP + EXPECTED_MEASURED}"
        )

    measured = d[d["measured"].astype(bool)].copy().reset_index(drop=True)
    if len(measured) != EXPECTED_MEASURED:
        raise RuntimeError(
            f"Measured rows={len(measured)} expected={EXPECTED_MEASURED}"
        )

    numeric_cols = [c for c in REQUIRED_COLUMNS if c not in {"sample_id", "measured"}]
    for c in numeric_cols:
        measured[c] = pd.to_numeric(measured[c], errors="raise")
        if not np.isfinite(measured[c].to_numpy(dtype=float)).all():
            raise RuntimeError(f"Non-finite timing/memory column={c}")

    return measured, audit


def derive_scopes(d: pd.DataFrame) -> pd.DataFrame:
    x = d.copy()

    x["pipeline_total_with_aux_geometry_ms"] = x["end_to_end_precommit_ms"]

    x["semantic_full_transition_precommit_ms"] = (
        x["end_to_end_precommit_ms"] - x["simple_geometry_ms"]
    )

    x["candidate_action_transaction_ms"] = (
        x["restore_before_candidate_ms"]
        + x["candidate_update_ms"]
        + x["candidate_reinference_ms"]
        + x["restore_after_candidate_ms"]
    )

    x["post_candidate_risk_scoring_ms"] = (
        x["dino_conditioning_pca_ms"] + x["safety_head_ms"]
    )

    x["candidate_aware_incremental_after_source_available_ms"] = (
        x["candidate_action_transaction_ms"]
        + x["post_candidate_risk_scoring_ms"]
    )

    instrumented = (
        x["image_load_preprocess_ms"]
        + x["restore_before_source_ms"]
        + x["source_inference_ms"]
        + x["candidate_action_transaction_ms"]
        + x["post_candidate_risk_scoring_ms"]
        + x["simple_geometry_ms"]
    )
    x["unattributed_orchestration_residual_ms"] = (
        x["end_to_end_precommit_ms"] - instrumented
    )

    # Simple descriptive throughput equivalents; not additional runtime measurements.
    x["semantic_full_transition_cases_per_s"] = (
        1000.0 / x["semantic_full_transition_precommit_ms"]
    )
    x["candidate_incremental_cases_per_s"] = (
        1000.0 / x["candidate_aware_incremental_after_source_available_ms"]
    )

    return x


def build_summary(x: pd.DataFrame) -> pd.DataFrame:
    fields = [
        "pipeline_total_with_aux_geometry_ms",
        "semantic_full_transition_precommit_ms",
        "candidate_action_transaction_ms",
        "post_candidate_risk_scoring_ms",
        "candidate_aware_incremental_after_source_available_ms",
        "simple_geometry_ms",
        "unattributed_orchestration_residual_ms",
        "semantic_full_transition_cases_per_s",
        "candidate_incremental_cases_per_s",
        "baseline_cuda_allocated_MB",
        "peak_cuda_allocated_MB",
        "peak_cuda_reserved_MB",
        "peak_incremental_allocated_MB",
    ]

    rows: List[Dict[str, Any]] = []
    for c in fields:
        rows.append({"scope": c, **stats(x[c].to_numpy(dtype=float))})
    return pd.DataFrame(rows)


def self_test():
    toy = pd.DataFrame([{
        "end_to_end_precommit_ms": 100.0,
        "simple_geometry_ms": 1.0,
        "restore_before_candidate_ms": 2.0,
        "candidate_update_ms": 60.0,
        "candidate_reinference_ms": 10.0,
        "restore_after_candidate_ms": 2.0,
        "dino_conditioning_pca_ms": 10.0,
        "safety_head_ms": 0.1,
        "image_load_preprocess_ms": 5.0,
        "restore_before_source_ms": 2.0,
        "source_inference_ms": 7.0,
    }])
    z = derive_scopes(toy)
    assert abs(float(z.loc[0, "semantic_full_transition_precommit_ms"]) - 99.0) < 1e-12
    assert abs(float(z.loc[0, "candidate_action_transaction_ms"]) - 74.0) < 1e-12
    assert abs(float(z.loc[0, "post_candidate_risk_scoring_ms"]) - 10.1) < 1e-12
    assert abs(float(z.loc[0, "candidate_aware_incremental_after_source_available_ms"]) - 84.1) < 1e-12
    print("SELF_TEST_SCOPE_FORMULAS=PASS")
    print("SELF_TEST=PASS")


def main():
    ap = argparse.ArgumentParser(
        description="SafeTTA P05B read-only runtime scope reporting audit."
    )
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    print("=" * 156)
    print("SafeTTA B6-P0-5B — runtime scope reporting audit")
    print("Version                :", VERSION)
    print("New runtime measurement: NO")
    print("Model loading          : NO")
    print("Inference              : NO")
    print("GT/science metrics     : NO")
    print("=" * 156)

    d, p05a_audit = load_and_verify()
    x = derive_scopes(d)
    summary = build_summary(x)

    print("\nP05A_GATE =", p05a_audit["gate"])
    print("measured rows =", len(x))
    print("\nRUNTIME SCOPE SUMMARY")
    print(summary.to_string(index=False))

    if OUT_DIR.exists():
        raise FileExistsError(
            f"Never overwrite P05B output: {OUT_DIR}\n"
            "Use _fix1 only for implementation repair."
        )
    OUT_DIR.mkdir(parents=True, exist_ok=False)

    p_rows = OUT_DIR / "B6_P05B_RUNTIME_SCOPE_ROWS.csv"
    p_summary = OUT_DIR / "B6_P05B_RUNTIME_SCOPE_SUMMARY.csv"
    p_audit = OUT_DIR / "B6_P05B_RUNTIME_SCOPE_AUDIT.json"
    p_report = OUT_DIR / "B6_P05B_RUNTIME_SCOPE_REPORT.txt"

    x.to_csv(p_rows, index=False)
    summary.to_csv(p_summary, index=False)

    ix = summary.set_index("scope")
    full = ix.loc["semantic_full_transition_precommit_ms"]
    cand = ix.loc["candidate_aware_incremental_after_source_available_ms"]
    action = ix.loc["candidate_action_transaction_ms"]
    risk = ix.loc["post_candidate_risk_scoring_ms"]
    geom = ix.loc["simple_geometry_ms"]
    peak = ix.loc["peak_cuda_allocated_MB"]

    payload = {
        "status": "PASS",
        "gate": PASS_GATE,
        "version": VERSION,
        "source_P05A": {
            "audit_path": str(P05A_AUDIT),
            "audit_sha256": sha256_file(P05A_AUDIT),
            "rows_path": str(P05A_ROWS),
            "rows_sha256": sha256_file(P05A_ROWS),
            "gate": p05a_audit["gate"],
        },
        "reporting_scope_correction": {
            "P05A_end_to_end_included_aux_simple_geometry": True,
            "runtime_measurement_changed": False,
            "semantic_full_transition_definition": (
                "end_to_end_precommit_ms - simple_geometry_ms"
            ),
        },
        "headline": {
            "semantic_full_transition_mean_ms": float(full["mean"]),
            "semantic_full_transition_median_ms": float(full["median"]),
            "semantic_full_transition_p95_ms": float(full["p95"]),
            "semantic_full_transition_mean_cases_per_s": float(
                ix.loc["semantic_full_transition_cases_per_s", "mean"]
            ),
            "candidate_aware_incremental_after_source_available_mean_ms": float(
                cand["mean"]
            ),
            "candidate_action_transaction_mean_ms": float(action["mean"]),
            "post_candidate_risk_scoring_mean_ms": float(risk["mean"]),
            "aux_simple_geometry_mean_ms": float(geom["mean"]),
            "peak_cuda_allocated_mean_MB": float(peak["mean"]),
            "peak_cuda_allocated_max_MB": float(peak["max"]),
        },
        "claim_boundary": {
            "primary_runtime": "semantic_full_transition_precommit_ms",
            "candidate_incremental_is_secondary_descriptive_scope": True,
            "P05A_raw_pipeline_total_preserved": True,
            "no_new_measurement": True,
            "no_runtime_rescue": True,
        },
    }

    p_audit.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    report = "\n".join([
        "=" * 156,
        "SafeTTA B6-P0-5B RUNTIME SCOPE REPORTING AUDIT COMPLETE",
        "",
        "Primary paper-facing runtime:",
        f"  semantic full-transition mean   = {full['mean']:.3f} ms/case",
        f"  semantic full-transition median = {full['median']:.3f} ms/case",
        f"  semantic full-transition p95    = {full['p95']:.3f} ms/case",
        "",
        "Secondary decomposition:",
        f"  candidate action transaction    = {action['mean']:.3f} ms/case",
        f"  post-candidate risk scoring     = {risk['mean']:.3f} ms/case",
        f"  incremental after SOURCE ready  = {cand['mean']:.3f} ms/case",
        f"  auxiliary geometry comparator   = {geom['mean']:.3f} ms/case",
        f"  peak CUDA allocated mean/max    = {peak['mean']:.1f} / {peak['max']:.1f} MB",
        "",
        "GATE=" + PASS_GATE,
        "audit_json=" + str(p_audit),
        "stage_dir=" + str(OUT_DIR),
        "=" * 156,
        "",
    ])
    p_report.write_text(report, encoding="utf-8")
    print("\n" + report)


if __name__ == "__main__":
    main()
