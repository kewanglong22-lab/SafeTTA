#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SafeTTA B7-M01B — reporting completion

NO new scientific experiment.
NO new bootstrap draws.
NO model fitting / inference / calibration / rebinning.

Uses only the already-frozen B7-M01 fix2 outputs to:
1) distinguish raw point deltas from bootstrap mean deltas;
2) derive paired Full-vs-Simple and Full-vs-SOURCE CIs from the exact existing
   paired bootstrap replicates;
3) report HARM/Benefit model-case and physical-image support.

The original four-stratum B7-M01 primary remains NOT EVALUABLE.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd


VERSION = "2026-09-15-B7-M01B-v1"

ROOT = Path(r"F:\MEDSEG_SAFETTA")
CODE = ROOT / "code"

PROTOCOL = CODE / "B7_M01B_REPORTING_COMPLETION_PROTOCOL_v1.json"
EXPECTED_PROTOCOL_SHA256 = "cf42ea73ad66f3b267987c103221ea22b9f1ec7e2091f2389867354a73f27a3d"

PARENT_DIR = ROOT / "B7_M01_change_magnitude_vs_harm_direction_v1_fix2"
PARENT_AUDIT = PARENT_DIR / "B7_M01_CHANGE_MAGNITUDE_VS_HARM_DIRECTION_AUDIT.json"
OVERALL_CSV = PARENT_DIR / "B7_M01_HARM_VS_BENEFIT_OVERALL.csv"
REPS_CSV = PARENT_DIR / "B7_M01_BOOTSTRAP_REPLICATES.csv"
BOUND_CSV = PARENT_DIR / "B7_M01_BOUND_DIRECTION_PANEL.csv"

EXPECTED_PARENT_GATE = "PASS_B7_M01_FIX2_EXECUTION_COMPLETE_PRIMARY_SUPPORT_FAILED"
EXPECTED_PRIMARY_STATUS = "NOT_EVALUABLE_INSUFFICIENT_FROZEN_STRATUM_SUPPORT"

OUT_DIR = ROOT / "B7_M01B_reporting_completion_v1"
PASS_GATE = "PASS_B7_M01B_REPORTING_COMPLETION"

FULL = "FROZEN_FULL_TRANSITION_SAFETTA"
SIMPLE = "SOURCE_PLUS_SIMPLE_MASK_CHANGE"
SOURCE = "SOURCE_STATE"
MAG = "CHANGE_MAGNITUDE_ONLY"

EXPECTED_REPS = 2000


def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def verify_protocol() -> Dict[str, Any]:
    if not PROTOCOL.is_file():
        raise FileNotFoundError(PROTOCOL)
    got = sha256_file(PROTOCOL)
    if got != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError(
            f"M01B protocol SHA mismatch expected={EXPECTED_PROTOCOL_SHA256} observed={got}"
        )
    d = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    expected = "FROZEN_AFTER_B7_M01_FIX2_BEFORE_ADDITIONAL_REPORTING_CONTRASTS"
    if d.get("status") != expected:
        raise RuntimeError(f"M01B protocol status changed: {d.get('status')}")
    return {"path": str(PROTOCOL), "sha256": got, "status": d["status"]}


def verify_parent() -> Dict[str, Any]:
    for p in [PARENT_AUDIT, OVERALL_CSV, REPS_CSV, BOUND_CSV]:
        if not p.is_file():
            raise FileNotFoundError(p)

    a = json.loads(PARENT_AUDIT.read_text(encoding="utf-8"))
    if a.get("gate") != EXPECTED_PARENT_GATE:
        raise RuntimeError(f"Parent gate changed: {a.get('gate')}")
    if a.get("primary_status") != EXPECTED_PRIMARY_STATUS:
        raise RuntimeError(f"Parent primary status changed: {a.get('primary_status')}")

    return {
        "audit_path": str(PARENT_AUDIT),
        "audit_sha256": sha256_file(PARENT_AUDIT),
        "gate": a["gate"],
        "primary_status": a["primary_status"],
    }


def load_overall() -> pd.DataFrame:
    d = pd.read_csv(OVERALL_CSV, low_memory=False)
    required = {"score", "n", "harm_n", "benefit_n", "auroc", "auprc", "prevalence"}
    missing = sorted(required - set(d.columns))
    if missing:
        raise RuntimeError(f"Overall table missing columns: {missing}")

    for score in [FULL, SIMPLE, SOURCE, MAG]:
        if int((d["score"] == score).sum()) != 1:
            raise RuntimeError(f"Expected one overall row for {score}")

    counts = d[["n", "harm_n", "benefit_n"]].drop_duplicates()
    if len(counts) != 1:
        raise RuntimeError("Overall score rows disagree on event counts.")
    r = counts.iloc[0]
    if int(r["n"]) != int(r["harm_n"]) + int(r["benefit_n"]):
        raise RuntimeError("HARM+BENEFIT model-case count mismatch.")

    expected_prev = int(r["harm_n"]) / int(r["n"])
    prevs = d["prevalence"].astype(float).to_numpy()
    if not np.allclose(prevs, expected_prev, rtol=0, atol=1e-12):
        raise RuntimeError("Conditional HARM prevalence mismatch.")

    return d


def raw_point_deltas(overall: pd.DataFrame) -> pd.DataFrame:
    m = overall.set_index("score")
    rows = []
    for ref in [MAG, SIMPLE, SOURCE]:
        for metric in ["auroc", "auprc"]:
            rows.append({
                "comparison": f"{FULL} - {ref}",
                "metric": metric.upper(),
                "full_point": float(m.loc[FULL, metric]),
                "reference_point": float(m.loc[ref, metric]),
                "raw_point_delta": float(m.loc[FULL, metric] - m.loc[ref, metric]),
            })
    return pd.DataFrame(rows)


def paired_bootstrap_deltas() -> pd.DataFrame:
    r = pd.read_csv(REPS_CSV, low_memory=False)
    required = {"replicate", "scope", "score", "auroc", "auprc"}
    missing = sorted(required - set(r.columns))
    if missing:
        raise RuntimeError(f"Bootstrap replicate table missing columns: {missing}")

    r = r[r["scope"] == "OVERALL"].copy()
    if r.empty:
        raise RuntimeError("No OVERALL bootstrap rows.")

    rows = []
    for metric in ["auroc", "auprc"]:
        p = r.pivot(index="replicate", columns="score", values=metric)
        for ref in [MAG, SIMPLE, SOURCE]:
            needed = [FULL, ref]
            if not set(needed).issubset(p.columns):
                raise RuntimeError(f"Missing paired bootstrap scores for {needed}")
            z = p[needed].dropna()
            if len(z) != EXPECTED_REPS:
                raise RuntimeError(
                    f"Paired replicate count for {FULL}-{ref}/{metric} = {len(z)}, "
                    f"expected {EXPECTED_REPS}"
                )
            delta = (z[FULL] - z[ref]).to_numpy(dtype=float)
            lo = float(np.quantile(delta, 0.025))
            hi = float(np.quantile(delta, 0.975))
            rows.append({
                "comparison": f"{FULL} - {ref}",
                "metric": metric.upper(),
                "valid_reps": int(len(delta)),
                "bootstrap_mean_delta": float(delta.mean()),
                "ci95_low": lo,
                "ci95_high": hi,
                "ci_excludes_zero": bool(lo > 0 or hi < 0),
            })

    return pd.DataFrame(rows)


def event_support() -> tuple[pd.DataFrame, pd.DataFrame]:
    d = pd.read_csv(BOUND_CSV, low_memory=False)
    required = {"sample_id", "model_state_id", "harm", "benefit"}
    missing = sorted(required - set(d.columns))
    if missing:
        raise RuntimeError(f"Bound panel missing columns: {missing}")

    harm = pd.to_numeric(d["harm"], errors="raise").astype(int)
    benefit = pd.to_numeric(d["benefit"], errors="raise").astype(int)
    if np.any((harm == 1) & (benefit == 1)):
        raise RuntimeError("HARM/BENEFIT overlap at model-case level.")

    x = d[(harm == 1) | (benefit == 1)].copy()
    x["sample_id"] = x["sample_id"].astype(str).str.strip().str.lower()

    # Model-case support: this is the population on which AUROC/AUPRC are computed.
    n = len(x)
    hn = int(pd.to_numeric(x["harm"]).sum())
    bn = int(pd.to_numeric(x["benefit"]).sum())
    if n != hn + bn:
        raise RuntimeError("Direction subset model-case count mismatch.")

    harm_ids = set(x.loc[pd.to_numeric(x["harm"]) == 1, "sample_id"])
    benefit_ids = set(x.loc[pd.to_numeric(x["benefit"]) == 1, "sample_id"])
    union_ids = harm_ids | benefit_ids
    overlap_ids = harm_ids & benefit_ids

    summary = pd.DataFrame([{
        "model_cases_total": n,
        "harm_model_cases": hn,
        "benefit_model_cases": bn,
        "conditional_harm_prevalence": hn / n,
        "physical_images_union": len(union_ids),
        "physical_images_with_harm": len(harm_ids),
        "physical_images_with_benefit": len(benefit_ids),
        "physical_images_with_both_across_states": len(overlap_ids),
        "physical_images_harm_only": len(harm_ids - benefit_ids),
        "physical_images_benefit_only": len(benefit_ids - harm_ids),
    }])

    per_state = []
    for state, g in x.groupby("model_state_id", sort=True):
        hs = set(g.loc[pd.to_numeric(g["harm"]) == 1, "sample_id"])
        bs = set(g.loc[pd.to_numeric(g["benefit"]) == 1, "sample_id"])
        per_state.append({
            "model_state_id": state,
            "model_cases_total": len(g),
            "harm_model_cases": int(pd.to_numeric(g["harm"]).sum()),
            "benefit_model_cases": int(pd.to_numeric(g["benefit"]).sum()),
            "physical_images_union": len(set(g["sample_id"])),
            "physical_images_with_harm": len(hs),
            "physical_images_with_benefit": len(bs),
            "physical_images_with_both": len(hs & bs),
        })

    return summary, pd.DataFrame(per_state)


def self_test() -> None:
    assert EXPECTED_REPS == 2000
    assert FULL != SIMPLE != SOURCE != MAG
    assert "REPORTING_COMPLETION" in PASS_GATE
    print("SELF_TEST_REGISTRY=PASS")
    print("SELF_TEST=PASS")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Complete B7-M01 reporting from existing frozen outputs only."
    )
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    print("=" * 168)
    print("SafeTTA B7-M01B — Reporting completion")
    print("Version                  :", VERSION)
    print("New bootstrap draws      : NO")
    print("New model/inference/TTA  : NO")
    print("Primary redefinition     : NO")
    print("Target rebinning         : NO")
    print("=" * 168)

    protocol = verify_protocol()
    parent = verify_parent()

    overall = load_overall()
    point = raw_point_deltas(overall)
    boot = paired_bootstrap_deltas()
    support, per_state = event_support()

    # Join point and bootstrap reporting for final reviewer-facing table.
    final = point.merge(
        boot,
        on=["comparison", "metric"],
        how="left",
        validate="one_to_one",
    )
    if final.isna().any().any():
        raise RuntimeError("Final paired reporting table contains missing values.")

    if OUT_DIR.exists():
        raise FileExistsError(
            f"Never overwrite M01B reporting outputs: {OUT_DIR}\n"
            "Use _fix1 only for implementation repair."
        )
    OUT_DIR.mkdir(parents=True, exist_ok=False)

    p_final = OUT_DIR / "B7_M01B_PAIRED_REPORTING_COMPLETION.csv"
    p_support = OUT_DIR / "B7_M01B_EVENT_SUPPORT.csv"
    p_state = OUT_DIR / "B7_M01B_EVENT_SUPPORT_BY_STATE.csv"

    final.to_csv(p_final, index=False)
    support.to_csv(p_support, index=False)
    per_state.to_csv(p_state, index=False)

    print("\nRAW POINT + PAIRED BOOTSTRAP REPORTING")
    print(final.to_string(index=False))

    print("\nHARM-vs-BENEFIT EVENT SUPPORT")
    print(support.to_string(index=False))

    print("\nEVENT SUPPORT BY MODEL STATE")
    print(per_state.to_string(index=False))

    audit = {
        "status": "PASS",
        "gate": PASS_GATE,
        "version": VERSION,
        "protocol": protocol,
        "parent": parent,
        "new_bootstrap_draws": False,
        "new_model_fit": False,
        "new_inference_tta": False,
        "target_rebinning": False,
        "primary_redefinition": False,
        "primary_status_preserved": EXPECTED_PRIMARY_STATUS,
        "outputs": {},
    }

    for p in [p_final, p_support, p_state]:
        audit["outputs"][p.name] = {
            "path": str(p),
            "bytes": p.stat().st_size,
            "sha256": sha256_file(p),
        }

    p_audit = OUT_DIR / "B7_M01B_REPORTING_COMPLETION_AUDIT.json"
    p_audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    report = "\n".join([
        "=" * 168,
        "SafeTTA B7-M01B REPORTING COMPLETION",
        "",
        "RAW POINT + PAIRED BOOTSTRAP:",
        final.to_string(index=False),
        "",
        "EVENT SUPPORT:",
        support.to_string(index=False),
        "",
        "EVENT SUPPORT BY STATE:",
        per_state.to_string(index=False),
        "",
        f"PRIMARY_STATUS_PRESERVED={EXPECTED_PRIMARY_STATUS}",
        "NEW_BOOTSTRAP_DRAWS=NO",
        "PRIMARY_REDEFINITION=NO",
        f"GATE={PASS_GATE}",
        f"audit_json={p_audit}",
        "=" * 168,
        "",
    ])
    p_report = OUT_DIR / "B7_M01B_REPORTING_COMPLETION_REPORT.txt"
    p_report.write_text(report, encoding="utf-8")
    print("\n" + report)


if __name__ == "__main__":
    main()
