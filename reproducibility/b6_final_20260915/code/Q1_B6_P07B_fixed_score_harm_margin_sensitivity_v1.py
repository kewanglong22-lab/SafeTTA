#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SafeTTA B6-P0-7B — fixed-score HARM-margin sensitivity.

Frozen sensitivity margins:
  HARM iff DeltaDice <= -0.01
  HARM iff DeltaDice <= -0.02  (original primary)
  HARM iff DeltaDice <= -0.05

Panels:
1) MRI PROMISE12: frozen B6-P01A FULL_TRANSITION_SAFETTA scores,
   2 families x 3 actions = 6 cells, 8262 rows, 50 patients.
2) PolypGen unseen MEMO: frozen R33A2B pre-HARM SafeTTA score joined to
   frozen R33A3 first-HARM-reveal DeltaDice, 4596 rows, 1532 physical images,
   3 model states.

NO fitting/refitting.
NO target calibration.
NO score flip.
NO representation change.
NO margin selection.
NO action tuning.

The -0.02 endpoint remains the primary/original definition.
Sensitivity margins are robustness analyses only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score


VERSION = "2026-09-15-B6-P07B-v1"

ROOT = Path(r"F:\MEDSEG_SAFETTA")
CODE = ROOT / "code"

PROTOCOL = CODE / "B6_P07_HARM_MARGIN_SENSITIVITY_PROTOCOL_v1.json"
EXPECTED_PROTOCOL_SHA256 = (
    "9bed5e361d92db364fdc72eaefcd5494b051464977525e266e4c0aa8bc0d233e"
)

BINDING_AMENDMENT = CODE / "B6_P07_POLYPGEN_SCORE_BINDING_AMENDMENT_v1.json"
EXPECTED_BINDING_AMENDMENT_SHA256 = "5e89b2be98fd05b113b9b85d1062074557703f2d33ff7898e0df0f904b36ac2b"

P07A_DECISION = (
    ROOT / "B6_P07A_harm_margin_sensitivity_asset_binding_audit_v1_fix1"
    / "B6_P07A_HARM_MARGIN_SENSITIVITY_BINDING.json"
)
EXPECTED_P07A_GATE = "PASS_B6_P07A_HARM_MARGIN_SENSITIVITY_ASSET_BINDING"

MRI_SCORE = (
    ROOT / "B6_P01A_mri_matched_representation_attribution_v1"
    / "B6_P01A_PROMISE12_MATCHED_SCORES.csv"
)

POLY_SCORE = (
    ROOT / "R33A2B_external_polypgen_memo_transition_score_lock_v1_fix1"
    / "R33A2B_POLYPGEN_MEMO_EXTERNAL_SAFETTA_SCORE_PRE_HARM_LOCK.csv"
)
POLY_OUTCOME = (
    ROOT / "R33A3_first_polypgen_memo_harm_reveal_external_joint_shift_evaluation_v1"
    / "R33A3_POLYPGEN_MEMO_MODELCASE_OUTCOMES.csv"
)

OUT_DIR = ROOT / "B6_P07B_fixed_score_harm_margin_sensitivity_v1"

MARGINS = [-0.01, -0.02, -0.05]
PRIMARY_MARGIN = -0.02

MRI_ROWS = 8262
MRI_PATIENTS = 50
MRI_CELLS = 6
MRI_BOOT_REPS = 2000
MRI_BOOT_SEED = 20260921

POLY_ROWS = 4596
POLY_IMAGES = 1532
POLY_STATES = 3
POLY_BOOT_REPS = 2000
POLY_BOOT_SEED = 20260922

PASS_GATE = "PASS_B6_P07B_FIXED_SCORE_HARM_MARGIN_SENSITIVITY_COMPLETE"


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


def verify_lineage() -> Dict[str, Any]:
    for label, path, expected in [
        ("P07_PROTOCOL", PROTOCOL, EXPECTED_PROTOCOL_SHA256),
        ("P07_BINDING_AMENDMENT", BINDING_AMENDMENT, EXPECTED_BINDING_AMENDMENT_SHA256),
    ]:
        if not path.is_file():
            raise FileNotFoundError(path)
        got = sha256_file(path)
        if got != expected:
            raise RuntimeError(f"{label} SHA mismatch expected={expected} observed={got}")

    p = load_json(PROTOCOL)
    if p.get("status") != (
        "FROZEN_POST_REVEAL_SENSITIVITY_PROTOCOL_BEFORE_MARGIN_SENSITIVITY_METRICS"
    ):
        raise RuntimeError("P07 protocol status changed.")

    a = load_json(BINDING_AMENDMENT)
    if a.get("status") != (
        "FROZEN_AFTER_P07A_SCHEMA_DISCOVERY_BEFORE_P07B_SENSITIVITY_METRICS"
    ):
        raise RuntimeError("P07 binding amendment status changed.")

    if not P07A_DECISION.is_file():
        raise FileNotFoundError(P07A_DECISION)
    d = load_json(P07A_DECISION)
    if d.get("gate") != EXPECTED_P07A_GATE:
        raise RuntimeError(f"P07A gate changed: {d.get('gate')}")
    if d.get("PolypGen", {}).get("route") != "DIRECT_4596_SCORE_DELTA_TABLE":
        raise RuntimeError(
            "P07A PolypGen route changed; expected DIRECT_4596_SCORE_DELTA_TABLE."
        )

    return {
        "protocol_path": str(PROTOCOL),
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "binding_amendment_path": str(BINDING_AMENDMENT),
        "binding_amendment_sha256": EXPECTED_BINDING_AMENDMENT_SHA256,
        "p07a_decision_path": str(P07A_DECISION),
        "p07a_decision_sha256": sha256_file(P07A_DECISION),
    }


def metrics_binary(y: np.ndarray, score: np.ndarray) -> Tuple[float,float]:
    y = np.asarray(y, dtype=np.int8)
    score = np.asarray(score, dtype=np.float64)
    if len(np.unique(y)) != 2:
        return np.nan, np.nan
    return float(roc_auc_score(y, score)), float(average_precision_score(y, score))


def load_mri() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if not MRI_SCORE.is_file():
        raise FileNotFoundError(MRI_SCORE)
    d = pd.read_csv(MRI_SCORE, low_memory=False)

    required = {
        "family","action","case_key","slice_index",
        "risk_score","delta_dice","representation","harm"
    }
    missing = sorted(required - set(d.columns))
    if missing:
        raise RuntimeError(f"MRI matched score table missing columns={missing}")

    d = d[d["representation"].astype(str) == "FULL_TRANSITION_SAFETTA"].copy()

    if len(d) != MRI_ROWS:
        raise RuntimeError(f"MRI rows={len(d)} expected={MRI_ROWS}")
    if d["case_key"].astype(str).nunique() != MRI_PATIENTS:
        raise RuntimeError("MRI patient count changed.")
    if len(d[["family","action"]].drop_duplicates()) != MRI_CELLS:
        raise RuntimeError("MRI cell count changed.")

    for c in ["risk_score","delta_dice"]:
        d[c] = pd.to_numeric(d[c], errors="raise")
        if not np.isfinite(d[c].to_numpy(float)).all():
            raise RuntimeError(f"MRI non-finite {c}")

    # Primary margin label parity to the frozen existing label.
    y_primary = (d["delta_dice"].to_numpy(float) <= PRIMARY_MARGIN).astype(np.int8)
    y_frozen = pd.to_numeric(d["harm"], errors="raise").to_numpy(np.int8)
    if not np.array_equal(y_primary, y_frozen):
        raise RuntimeError("MRI primary -0.02 HARM label parity failed.")

    d["cluster_id"] = d["case_key"].astype(str)
    d["cell"] = d["family"].astype(str) + "||" + d["action"].astype(str)

    return d.reset_index(drop=True), {
        "path": str(MRI_SCORE),
        "sha256": sha256_file(MRI_SCORE),
        "rows": int(len(d)),
        "patients": int(d["cluster_id"].nunique()),
        "cells": sorted(d["cell"].unique().tolist()),
        "primary_harm_count": int(y_primary.sum()),
        "primary_harm_prevalence": float(y_primary.mean()),
        "primary_label_parity": True,
    }


def load_polypgen() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    for p in [POLY_SCORE, POLY_OUTCOME]:
        if not p.is_file():
            raise FileNotFoundError(p)

    s = pd.read_csv(POLY_SCORE, low_memory=False)
    o = pd.read_csv(POLY_OUTCOME, low_memory=False)

    score_col = "safettta_external_harm_risk"
    delta_col = "memo_delta_dice"
    primary_label_col = "memo_harm_label"
    keys = [
        "sample_id",
        "model_family",
        "model_state_id",
        "training_seed",
        "checkpoint_sha256",
    ]

    for c in keys + [score_col]:
        if c not in s.columns:
            raise RuntimeError(f"PolypGen score table missing {c}")
    for c in keys + [delta_col, primary_label_col]:
        if c not in o.columns:
            raise RuntimeError(f"PolypGen outcome table missing {c}")

    if len(s) != POLY_ROWS or len(o) != POLY_ROWS:
        raise RuntimeError(f"PolypGen row drift score/outcome={len(s)}/{len(o)}")
    if s.duplicated(keys).any() or o.duplicated(keys).any():
        raise RuntimeError("PolypGen join keys are not unique.")

    keep_s = keys + [score_col]
    keep_o = keys + [delta_col, primary_label_col]
    d = s[keep_s].merge(o[keep_o], on=keys, how="inner", validate="one_to_one")

    if len(d) != POLY_ROWS:
        raise RuntimeError(f"PolypGen exact join rows={len(d)} expected={POLY_ROWS}")

    if d["sample_id"].astype(str).nunique() != POLY_IMAGES:
        raise RuntimeError("PolypGen physical image count changed.")
    if d["model_state_id"].astype(str).nunique() != POLY_STATES:
        raise RuntimeError("PolypGen state count changed.")

    for c in [score_col, delta_col]:
        d[c] = pd.to_numeric(d[c], errors="raise")
        if not np.isfinite(d[c].to_numpy(float)).all():
            raise RuntimeError(f"PolypGen non-finite {c}")

    y_primary = (d[delta_col].to_numpy(float) <= PRIMARY_MARGIN).astype(np.int8)
    y_frozen = pd.to_numeric(d[primary_label_col], errors="raise").to_numpy(np.int8)
    if not np.array_equal(y_primary, y_frozen):
        raise RuntimeError("PolypGen primary -0.02 HARM label parity failed.")

    d = d.rename(
        columns={
            score_col: "risk_score",
            delta_col: "delta_dice",
        }
    )
    d["cluster_id"] = d["sample_id"].astype(str)
    d["cell"] = d["model_state_id"].astype(str)

    return d.reset_index(drop=True), {
        "score_path": str(POLY_SCORE),
        "score_sha256": sha256_file(POLY_SCORE),
        "outcome_path": str(POLY_OUTCOME),
        "outcome_sha256": sha256_file(POLY_OUTCOME),
        "rows": int(len(d)),
        "physical_images": int(d["cluster_id"].nunique()),
        "states": sorted(d["cell"].unique().tolist()),
        "score_column": score_col,
        "delta_column": delta_col,
        "primary_harm_count": int(y_primary.sum()),
        "primary_harm_prevalence": float(y_primary.mean()),
        "primary_label_parity": True,
    }


def point_metrics(
    d: pd.DataFrame,
    panel: str,
    expected_cells: int,
) -> pd.DataFrame:
    rows = []
    score = d["risk_score"].to_numpy(float)

    for margin in MARGINS:
        y = (d["delta_dice"].to_numpy(float) <= margin).astype(np.int8)
        pauc, pap = metrics_binary(y, score)
        if not np.isfinite(pauc):
            raise RuntimeError(f"{panel} pooled margin {margin}: single class.")

        cell_aucs = []
        cell_aps = []
        for cell, g in d.groupby("cell", sort=True):
            yy = (g["delta_dice"].to_numpy(float) <= margin).astype(np.int8)
            ss = g["risk_score"].to_numpy(float)
            a, p = metrics_binary(yy, ss)
            if not (np.isfinite(a) and np.isfinite(p)):
                raise RuntimeError(
                    f"{panel} cell={cell} margin={margin}: single class."
                )
            cell_aucs.append(a)
            cell_aps.append(p)

        if len(cell_aucs) != expected_cells:
            raise RuntimeError(f"{panel}: cell count drift.")

        rows.append({
            "panel": panel,
            "margin": margin,
            "is_original_primary_margin": bool(abs(margin - PRIMARY_MARGIN) < 1e-12),
            "rows": int(len(d)),
            "clusters": int(d["cluster_id"].nunique()),
            "harm_n": int(y.sum()),
            "harm_prevalence": float(y.mean()),
            "pooled_auroc": pauc,
            "pooled_auprc": pap,
            "macro_cell_auroc": float(np.mean(cell_aucs)),
            "macro_cell_auprc": float(np.mean(cell_aps)),
        })

    return pd.DataFrame(rows)


def clustered_bootstrap(
    d: pd.DataFrame,
    panel: str,
    expected_clusters: int,
    expected_cells: int,
    reps: int,
    seed: int,
) -> Tuple[pd.DataFrame,pd.DataFrame]:
    rng = np.random.default_rng(seed)

    clusters = np.asarray(sorted(d["cluster_id"].astype(str).unique()))
    if len(clusters) != expected_clusters:
        raise RuntimeError(f"{panel}: cluster count drift.")

    cluster_arr = d["cluster_id"].astype(str).to_numpy()
    cluster_to_rows = {c: np.flatnonzero(cluster_arr == c) for c in clusters}

    rep_rows = []

    for rep in tqdm(
        range(reps),
        desc=f"P07B {panel} clustered bootstrap",
        unit="rep",
        dynamic_ncols=True,
    ):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        idx = np.concatenate([cluster_to_rows[c] for c in sampled])

        b = d.iloc[idx]

        for margin in MARGINS:
            y = (b["delta_dice"].to_numpy(float) <= margin).astype(np.int8)
            s = b["risk_score"].to_numpy(float)
            pa, pp = metrics_binary(y, s)

            cell_aucs = []
            cell_aps = []
            cell_valid = True
            for _, g in b.groupby("cell", sort=True):
                yy = (g["delta_dice"].to_numpy(float) <= margin).astype(np.int8)
                ss = g["risk_score"].to_numpy(float)
                a, p = metrics_binary(yy, ss)
                if not (np.isfinite(a) and np.isfinite(p)):
                    cell_valid = False
                    break
                cell_aucs.append(a)
                cell_aps.append(p)

            if len(cell_aucs) != expected_cells:
                cell_valid = False

            rep_rows.append({
                "panel": panel,
                "rep": rep,
                "margin": margin,
                "pooled_auroc": pa,
                "pooled_auprc": pp,
                "macro_cell_auroc": (
                    float(np.mean(cell_aucs)) if cell_valid else np.nan
                ),
                "macro_cell_auprc": (
                    float(np.mean(cell_aps)) if cell_valid else np.nan
                ),
            })

    reps_df = pd.DataFrame(rep_rows)

    summary = []
    for margin in MARGINS:
        g = reps_df[reps_df["margin"] == margin]
        for metric in [
            "pooled_auroc",
            "pooled_auprc",
            "macro_cell_auroc",
            "macro_cell_auprc",
        ]:
            vals = g[metric].to_numpy(float)
            vals = vals[np.isfinite(vals)]
            if len(vals) < int(0.90 * reps):
                raise RuntimeError(
                    f"{panel} margin={margin} metric={metric}: "
                    f"too few valid bootstrap reps {len(vals)}/{reps}"
                )
            summary.append({
                "panel": panel,
                "margin": margin,
                "metric": metric,
                "valid_reps": int(len(vals)),
                "requested_reps": int(reps),
                "bootstrap_mean": float(vals.mean()),
                "ci95_low": float(np.quantile(vals, 0.025)),
                "ci95_high": float(np.quantile(vals, 0.975)),
            })

    return reps_df, pd.DataFrame(summary)


def main():
    ap = argparse.ArgumentParser(
        description="SafeTTA fixed-score HARM-margin sensitivity."
    )
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--preflight-only", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        y = np.asarray([0,0,1,1], dtype=np.int8)
        s = np.asarray([0.1,0.2,0.8,0.9], dtype=float)
        a,p = metrics_binary(y,s)
        assert abs(a - 1.0) < 1e-12
        assert abs(p - 1.0) < 1e-12
        assert MARGINS == [-0.01,-0.02,-0.05]
        print("SELF_TEST_METRICS=PASS")
        print("SELF_TEST=PASS")
        return

    print("=" * 172)
    print("SafeTTA B6-P0-7B — fixed-score HARM-margin sensitivity")
    print("Version           :", VERSION)
    print("Margins           :", MARGINS)
    print("Original primary  :", PRIMARY_MARGIN)
    print("Model fitting     : NO")
    print("Target calibration: NO")
    print("Score flip        : NO")
    print("Margin selection  : NO")
    print("=" * 172)

    print("\n[1/5] Lineage")
    lineage = verify_lineage()
    print("PROTOCOL_SHA256 =", lineage["protocol_sha256"])
    print("BINDING_SHA256  =", lineage["binding_amendment_sha256"])
    print("P07A_GATE       =", EXPECTED_P07A_GATE)
    print("LINEAGE=PASS")

    print("\n[2/5] Bind frozen panels + primary-label parity")
    mri, mri_meta = load_mri()
    poly, poly_meta = load_polypgen()

    print("MRI:", mri_meta)
    print("POLYPGEN:", poly_meta)
    print("PRIMARY_-0.02_LABEL_PARITY=PASS")

    if args.preflight_only:
        print("\nP07B_PREFLIGHT=PASS")
        print("SENSITIVITY_METRICS=NOT_RUN")
        print("BOOTSTRAP=NOT_RUN")
        return

    if OUT_DIR.exists():
        raise FileExistsError(
            f"Never overwrite P07B output: {OUT_DIR}\n"
            "Use _fix1 only for implementation repair."
        )
    OUT_DIR.mkdir(parents=True, exist_ok=False)

    print("\n[3/5] Point sensitivity metrics")
    mri_point = point_metrics(mri, "MRI_PROMISE12", MRI_CELLS)
    poly_point = point_metrics(poly, "POLYPGEN_MEMO", POLY_STATES)
    points = pd.concat([mri_point, poly_point], ignore_index=True)
    print(points.to_string(index=False))

    print("\n[4/5] MRI patient-cluster bootstrap")
    mri_rep, mri_boot = clustered_bootstrap(
        mri, "MRI_PROMISE12",
        MRI_PATIENTS, MRI_CELLS,
        MRI_BOOT_REPS, MRI_BOOT_SEED,
    )

    print("\n[5/5] PolypGen physical-image clustered bootstrap")
    poly_rep, poly_boot = clustered_bootstrap(
        poly, "POLYPGEN_MEMO",
        POLY_IMAGES, POLY_STATES,
        POLY_BOOT_REPS, POLY_BOOT_SEED,
    )

    boot = pd.concat([mri_boot, poly_boot], ignore_index=True)

    print("\nBOOTSTRAP SUMMARY")
    print(boot.to_string(index=False))

    p_points = OUT_DIR / "B6_P07B_HARM_MARGIN_POINT_METRICS.csv"
    p_boot = OUT_DIR / "B6_P07B_HARM_MARGIN_BOOTSTRAP_SUMMARY.csv"
    p_mri_rep = OUT_DIR / "B6_P07B_MRI_PATIENT_BOOTSTRAP_REPLICATES.csv"
    p_poly_rep = OUT_DIR / "B6_P07B_POLYPGEN_IMAGE_BOOTSTRAP_REPLICATES.csv"

    points.to_csv(p_points, index=False)
    boot.to_csv(p_boot, index=False)
    mri_rep.to_csv(p_mri_rep, index=False)
    poly_rep.to_csv(p_poly_rep, index=False)

    audit = {
        "status": "PASS",
        "gate": PASS_GATE,
        "version": VERSION,
        "lineage": lineage,
        "frozen_constraints": {
            "original_primary_margin": PRIMARY_MARGIN,
            "sensitivity_margins": MARGINS,
            "model_refit": False,
            "target_calibration": False,
            "score_flip": False,
            "margin_selection": False,
        },
        "MRI": mri_meta,
        "PolypGen": poly_meta,
        "point_metrics": points.to_dict(orient="records"),
        "bootstrap_summary": boot.to_dict(orient="records"),
        "artifacts": {},
    }

    for p in [p_points,p_boot,p_mri_rep,p_poly_rep]:
        audit["artifacts"][p.name] = {
            "path": str(p),
            "sha256": sha256_file(p),
            "bytes": int(p.stat().st_size),
        }

    p_audit = OUT_DIR / "B6_P07B_HARM_MARGIN_SENSITIVITY_AUDIT.json"
    p_audit.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    report = "\n".join([
        "=" * 172,
        "SafeTTA B6-P0-7B FIXED-SCORE HARM-MARGIN SENSITIVITY COMPLETE",
        "",
        "POINT METRICS:",
        points.to_string(index=False),
        "",
        "BOOTSTRAP SUMMARY:",
        boot.to_string(index=False),
        "",
        "Original -0.02 endpoint remains primary: YES",
        "No refit/calibration/score flip/margin selection: YES",
        "",
        "GATE=" + PASS_GATE,
        "audit_json=" + str(p_audit),
        "stage_dir=" + str(OUT_DIR),
        "=" * 172,
        "",
    ])
    p_report = OUT_DIR / "B6_P07B_HARM_MARGIN_SENSITIVITY_REPORT.txt"
    p_report.write_text(report, encoding="utf-8")
    print("\n" + report)


if __name__ == "__main__":
    main()
