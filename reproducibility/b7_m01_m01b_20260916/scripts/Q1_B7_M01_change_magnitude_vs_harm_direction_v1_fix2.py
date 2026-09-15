#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SafeTTA B7-M01 — Change magnitude versus HARM direction

Post-B6 explanatory analysis only.

Question:
    Do frozen candidate-aware risk scores distinguish HARM from BENEFIT
    beyond merely detecting large SOURCE-to-candidate mask changes?

Frozen design:
- Primary panel: PolypGen unseen MEMO, exact B6 common support.
- HARM:    DeltaDice <= -0.02.
- BENEFIT: DeltaDice >= +0.02.
- NEUTRAL excluded for the directionality analysis.
- Primary change magnitude: 1 - prediction Dice(SOURCE mask, candidate mask).
- Source-defined magnitude strata: quartiles from frozen NeoPolyp TENT1+PL-CONF90
  candidate-mask geometry, without using source outcomes for the cutpoints.
- Frozen risk scores only; NO model fit, calibration, score flip, or target tuning.
- Physical-image clustered bootstrap, 2000 reps.

This script reuses exact P01B asset-binding functions so that no new candidate-mask
adjudication is introduced.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm


VERSION = "2026-09-15-B7-M01-v1-fix2"

ROOT = Path(r"F:\MEDSEG_SAFETTA")
CODE = ROOT / "code"

PROTOCOL = CODE / "B7_M01_CHANGE_MAGNITUDE_VS_HARM_DIRECTION_PROTOCOL_v1.json"
EXPECTED_PROTOCOL_SHA256 = "5f71f47062b0630a3f7e391a9906d541593297e0f7a55f8954b6993a52d0316b"

P01B_SCRIPT = CODE / "Q1_B6_P01B_polypgen_geometry_core_experiment_v1_fix5.py"
P01B_DIR = ROOT / "B6_P01B_polypgen_geometry_core_experiment_v1_fix5"
P01B_AUDIT = P01B_DIR / "B6_P01B_POLYPGEN_GEOMETRY_CORE_AUDIT.json"
P01B_SCORES = P01B_DIR / "B6_P01B_POLYPGEN_GEOMETRY_CORE_SCORES.csv"

TARGET_PANEL = (
    ROOT
    / "R33A3_first_polypgen_memo_harm_reveal_external_joint_shift_evaluation_v1"
    / "R33A3_POLYPGEN_MEMO_MATCHED_SCORE_OUTCOME_PANEL.csv"
)

OUT_DIR = ROOT / "B7_M01_change_magnitude_vs_harm_direction_v1_fix2"

EXPECTED_P01B_GATE = "PASS_B6_P01B_POLYPGEN_GEOMETRY_CORE_EXPERIMENT_COMPLETE"

N_TARGET_ROWS = 4596
N_TARGET_IMAGES = 1532
N_TARGET_STATES = 3

BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 20260924
MIN_SUPPORT_PER_CLASS = 10

SCORE_COLS = [
    "SOURCE_STATE",
    "SOURCE_PLUS_SIMPLE_MASK_CHANGE",
    "FROZEN_FULL_TRANSITION_SAFETTA",
]

PRIMARY_FULL = "FROZEN_FULL_TRANSITION_SAFETTA"
PRIMARY_MAG = "CHANGE_MAGNITUDE_ONLY"
SENS_MAG = "CHANGE_MAGNITUDE_IOU_ONLY"

PASS_GATE = "PASS_B7_M01_FIX2_EXECUTION_COMPLETE_PRIMARY_SUPPORT_FAILED"


def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_protocol() -> Dict[str, Any]:
    if not PROTOCOL.is_file():
        raise FileNotFoundError(PROTOCOL)
    got = sha256_file(PROTOCOL)
    if got != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError(
            f"M01 protocol SHA mismatch expected={EXPECTED_PROTOCOL_SHA256} observed={got}"
        )
    p = load_json(PROTOCOL)
    expected = "FROZEN_POST_B6_BEFORE_NEW_EXPLANATORY_METRICS"
    if p.get("status") != expected:
        raise RuntimeError(f"M01 protocol status changed: {p.get('status')}")
    return {"path": str(PROTOCOL), "sha256": got, "status": p["status"]}


def load_p01b_module():
    if not P01B_SCRIPT.is_file():
        raise FileNotFoundError(P01B_SCRIPT)
    spec = importlib.util.spec_from_file_location("b6_p01b_geom", P01B_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load frozen P01B implementation.")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def verify_p01b_outputs() -> Dict[str, Any]:
    if not P01B_AUDIT.is_file():
        raise FileNotFoundError(P01B_AUDIT)
    if not P01B_SCORES.is_file():
        raise FileNotFoundError(P01B_SCORES)

    a = load_json(P01B_AUDIT)
    if a.get("gate") != EXPECTED_P01B_GATE:
        raise RuntimeError(f"P01B gate changed: {a.get('gate')}")

    outputs = a.get("outputs", {})
    meta = outputs.get(P01B_SCORES.name)
    if not isinstance(meta, dict):
        raise RuntimeError("P01B audit does not bind the frozen score file.")
    observed = sha256_file(P01B_SCORES)
    if observed != meta.get("sha256"):
        raise RuntimeError(
            f"P01B frozen score SHA drift manifest={meta.get('sha256')} observed={observed}"
        )

    return {
        "audit_path": str(P01B_AUDIT),
        "audit_sha256": sha256_file(P01B_AUDIT),
        "score_path": str(P01B_SCORES),
        "score_sha256": observed,
        "p01b_gate": a.get("gate"),
        "p01b_decision": a.get("decision"),
    }


def norm_id(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.strip()
        .str.replace("\\", "/", regex=False)
        .str.lower()
    )


def load_target_outcomes_and_scores() -> pd.DataFrame:
    if not TARGET_PANEL.is_file():
        raise FileNotFoundError(TARGET_PANEL)

    t = pd.read_csv(TARGET_PANEL, low_memory=False)
    required_t = {
        "sample_id", "model_state_id", "memo_delta_dice",
        "memo_harm_label", "memo_benefit_label",
    }
    missing = sorted(required_t - set(t.columns))
    if missing:
        raise RuntimeError(f"Target outcome panel missing columns: {missing}")
    if len(t) != N_TARGET_ROWS:
        raise RuntimeError(f"Target rows={len(t)} != {N_TARGET_ROWS}")
    if t["sample_id"].astype(str).nunique() != N_TARGET_IMAGES:
        raise RuntimeError("Target physical-image count changed.")
    if t["model_state_id"].astype(str).nunique() != N_TARGET_STATES:
        raise RuntimeError("Target state count changed.")

    d = pd.DataFrame({
        "sample_id": norm_id(t["sample_id"]),
        "model_state_id": t["model_state_id"].astype(str).str.strip(),
        "delta_dice": pd.to_numeric(t["memo_delta_dice"], errors="raise").astype(float),
        "harm": pd.to_numeric(t["memo_harm_label"], errors="raise").astype(int),
        "benefit": pd.to_numeric(t["memo_benefit_label"], errors="raise").astype(int),
    })

    expected_h = (d["delta_dice"].to_numpy() <= -0.02).astype(int)
    expected_b = (d["delta_dice"].to_numpy() >= +0.02).astype(int)
    if not np.array_equal(expected_h, d["harm"].to_numpy(dtype=int)):
        raise RuntimeError("Target HARM rule reproduction failed.")
    if not np.array_equal(expected_b, d["benefit"].to_numpy(dtype=int)):
        raise RuntimeError("Target BENEFIT rule reproduction failed.")
    if np.any((d["harm"] == 1) & (d["benefit"] == 1)):
        raise RuntimeError("HARM/BENEFIT overlap detected.")

    s = pd.read_csv(P01B_SCORES, low_memory=False)
    required_s = {"sample_id", "model_state_id", *SCORE_COLS}
    miss2 = sorted(required_s - set(s.columns))
    if miss2:
        raise RuntimeError(f"P01B score file missing columns: {miss2}")
    if len(s) != N_TARGET_ROWS:
        raise RuntimeError("P01B score row count changed.")

    ss = pd.DataFrame({
        "sample_id": norm_id(s["sample_id"]),
        "model_state_id": s["model_state_id"].astype(str).str.strip(),
    })
    for c in SCORE_COLS:
        ss[c] = pd.to_numeric(s[c], errors="raise").astype(float)

    out = d.merge(ss, on=["sample_id", "model_state_id"], how="left", validate="one_to_one")
    if out[SCORE_COLS].isna().any().any():
        raise RuntimeError("Frozen scores do not cover target outcome panel.")
    if not np.isfinite(out[SCORE_COLS].to_numpy(dtype=float)).all():
        raise RuntimeError("Frozen scores contain non-finite values.")

    return out


def build_source_and_target_geometry(mod, target: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    # Reuse the frozen P01B lineage and exact mask adjudication.
    _, inv = mod.verify_lineage(CODE)
    source_outcomes, source_outcome_meta = mod.choose_source_outcomes(inv)
    target_outcomes, target_outcome_meta = mod.choose_target_outcomes(inv)

    mod.verify_authoritative_mask_adjudication()

    source_parts = []
    source_geom_meta = {}
    for action in mod.TRAIN_ACTIONS:
        req = source_outcomes[source_outcomes["action"] == action][mod.KEY_COLS].copy()
        g, meta = mod.bind_r38a1a_neopolyp_geometry(req, action)
        part = req.reset_index(drop=True)
        part[["pred_dice", "pred_iou", "area_delta", "boundary_delta"]] = g
        part["action"] = action
        source_parts.append(part)
        source_geom_meta[action] = meta

    source_geom = pd.concat(source_parts, ignore_index=True)
    if len(source_geom) != mod.SOURCE_TOTAL_ROWS:
        raise RuntimeError("Source geometry row count changed.")

    tg, target_geom_meta = mod.bind_r33_polypgen_memo_geometry(target_outcomes[mod.KEY_COLS])
    target_geom = target_outcomes[mod.KEY_COLS].copy().reset_index(drop=True)
    target_geom[["pred_dice", "pred_iou", "area_delta", "boundary_delta"]] = tg

    bound = target.merge(
        target_geom,
        on=["sample_id", "model_state_id"],
        how="left",
        validate="one_to_one",
    )
    if bound[["pred_dice", "pred_iou"]].isna().any().any():
        raise RuntimeError("Target geometry binding incomplete.")

    meta = {
        "source_outcomes": source_outcome_meta,
        "target_outcomes": target_outcome_meta,
        "source_geometry": source_geom_meta,
        "target_geometry": target_geom_meta,
    }
    return source_geom, bound, meta


def frozen_source_quartiles(source_geom: pd.DataFrame) -> np.ndarray:
    mag = 1.0 - source_geom["pred_dice"].to_numpy(dtype=float)
    if not np.isfinite(mag).all():
        raise RuntimeError("Non-finite source change magnitude.")
    q = np.quantile(mag, [0.25, 0.50, 0.75])
    if not (q[0] <= q[1] <= q[2]):
        raise RuntimeError("Source quartiles invalid.")
    return q.astype(float)


def assign_strata(x: np.ndarray, q: np.ndarray) -> np.ndarray:
    # Q1: <=q25; Q2: (q25,q50]; Q3: (q50,q75]; Q4: >q75
    return np.digitize(x, q, right=True).astype(int) + 1


def binary_metrics(y: np.ndarray, score: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    if len(y) == 0:
        return {"auroc": math.nan, "auprc": math.nan, "prevalence": math.nan}
    if len(np.unique(y)) != 2:
        return {"auroc": math.nan, "auprc": math.nan, "prevalence": float(y.mean())}
    return {
        "auroc": float(roc_auc_score(y, score)),
        "auprc": float(average_precision_score(y, score)),
        "prevalence": float(y.mean()),
    }


def point_analysis(d: pd.DataFrame, q: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Direction subset only: HARM vs BENEFIT.
    x = d[(d["harm"] == 1) | (d["benefit"] == 1)].copy()
    x["direction_y"] = x["harm"].astype(int)
    x[PRIMARY_MAG] = 1.0 - x["pred_dice"].astype(float)
    x[SENS_MAG] = 1.0 - x["pred_iou"].astype(float)
    x["magnitude_stratum"] = assign_strata(x[PRIMARY_MAG].to_numpy(dtype=float), q)

    overall_rows = []
    for score_col in SCORE_COLS + [PRIMARY_MAG, SENS_MAG]:
        m = binary_metrics(x["direction_y"].to_numpy(), x[score_col].to_numpy())
        overall_rows.append({
            "analysis": "HARM_VS_BENEFIT_OVERALL",
            "score": score_col,
            "n": len(x),
            "harm_n": int(x["direction_y"].sum()),
            "benefit_n": int((1 - x["direction_y"]).sum()),
            **m,
        })

    stratum_rows = []
    for k in [1, 2, 3, 4]:
        g = x[x["magnitude_stratum"] == k]
        hn = int(g["direction_y"].sum())
        bn = int((1 - g["direction_y"]).sum())
        for score_col in SCORE_COLS + [PRIMARY_MAG]:
            m = binary_metrics(g["direction_y"].to_numpy(), g[score_col].to_numpy())
            stratum_rows.append({
                "stratum": k,
                "score": score_col,
                "n": len(g),
                "harm_n": hn,
                "benefit_n": bn,
                "support_gate": bool(hn >= MIN_SUPPORT_PER_CLASS and bn >= MIN_SUPPORT_PER_CLASS),
                **m,
            })

    strata_df = pd.DataFrame(stratum_rows)

    macro_rows = []
    for score_col in SCORE_COLS + [PRIMARY_MAG]:
        g = strata_df[
            (strata_df["score"] == score_col)
            & (strata_df["support_gate"])
            & np.isfinite(strata_df["auroc"])
        ]
        macro_rows.append({
            "score": score_col,
            "valid_strata": int(len(g)),
            "macro_stratified_auroc": float(g["auroc"].mean()) if len(g) else math.nan,
        })

    return pd.DataFrame(overall_rows), strata_df, pd.DataFrame(macro_rows)


def clustered_bootstrap(d: pd.DataFrame, q: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame]:
    x = d[(d["harm"] == 1) | (d["benefit"] == 1)].copy()
    x["direction_y"] = x["harm"].astype(int)
    x[PRIMARY_MAG] = 1.0 - x["pred_dice"].astype(float)
    x[SENS_MAG] = 1.0 - x["pred_iou"].astype(float)
    x["magnitude_stratum"] = assign_strata(x[PRIMARY_MAG].to_numpy(dtype=float), q)

    # Cluster bootstrap must resample the frozen analysis population itself
    # (HARM-or-BENEFIT rows), not physical images that contribute only NEUTRAL
    # rows and are absent from this conditional directionality endpoint.
    samples = np.asarray(sorted(x["sample_id"].unique()))
    idx_map = {
        s: np.flatnonzero(x["sample_id"].to_numpy() == s)
        for s in samples
    }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    print("direction_subset_cluster_images =", len(samples))

    reps = []
    deltas = []

    for b in tqdm(
        range(BOOTSTRAP_REPS),
        desc="B7-M01 physical-image bootstrap",
        unit="rep",
        dynamic_ncols=True,
    ):
        draw = rng.choice(samples, size=len(samples), replace=True)
        chunks = [idx_map[s] for s in draw if len(idx_map[s])]
        if not chunks:
            continue
        idx = np.concatenate(chunks)
        g = x.iloc[idx].copy()

        yy = g["direction_y"].to_numpy(dtype=int)
        if len(np.unique(yy)) != 2:
            continue

        overall = {}
        for score_col in SCORE_COLS + [PRIMARY_MAG, SENS_MAG]:
            m = binary_metrics(yy, g[score_col].to_numpy(dtype=float))
            overall[score_col] = m
            reps.append({
                "replicate": b,
                "scope": "OVERALL",
                "score": score_col,
                "auroc": m["auroc"],
                "auprc": m["auprc"],
                "valid_strata": np.nan,
            })

        # Fixed source-defined strata. Require all 4 strata to have both classes.
        macro_auc = {}
        for score_col in SCORE_COLS + [PRIMARY_MAG]:
            vals = []
            all_valid = True
            for k in [1, 2, 3, 4]:
                z = g[g["magnitude_stratum"] == k]
                yk = z["direction_y"].to_numpy(dtype=int)
                if len(np.unique(yk)) != 2:
                    all_valid = False
                    break
                vals.append(float(roc_auc_score(yk, z[score_col].to_numpy(dtype=float))))
            if all_valid:
                macro_auc[score_col] = float(np.mean(vals))
                reps.append({
                    "replicate": b,
                    "scope": "SOURCE_QUARTILE_MACRO",
                    "score": score_col,
                    "auroc": macro_auc[score_col],
                    "auprc": np.nan,
                    "valid_strata": 4,
                })

        # Paired primary/secondary macro contrasts.
        for score_col in [PRIMARY_FULL, "SOURCE_PLUS_SIMPLE_MASK_CHANGE", "SOURCE_STATE"]:
            if score_col in macro_auc and PRIMARY_MAG in macro_auc:
                deltas.append({
                    "replicate": b,
                    "scope": "SOURCE_QUARTILE_MACRO",
                    "comparison": f"{score_col} - {PRIMARY_MAG}",
                    "metric": "AUROC",
                    "delta": macro_auc[score_col] - macro_auc[PRIMARY_MAG],
                })

        # Overall contrasts against magnitude-only.
        for score_col in [PRIMARY_FULL, "SOURCE_PLUS_SIMPLE_MASK_CHANGE", "SOURCE_STATE"]:
            for metric in ["auroc", "auprc"]:
                if np.isfinite(overall[score_col][metric]) and np.isfinite(overall[PRIMARY_MAG][metric]):
                    deltas.append({
                        "replicate": b,
                        "scope": "OVERALL",
                        "comparison": f"{score_col} - {PRIMARY_MAG}",
                        "metric": metric.upper(),
                        "delta": overall[score_col][metric] - overall[PRIMARY_MAG][metric],
                    })

    return pd.DataFrame(reps), pd.DataFrame(deltas)


def summarize_bootstrap(reps: pd.DataFrame, deltas: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ci_rows = []
    for (scope, score, metric), g in reps.melt(
        id_vars=["replicate", "scope", "score", "valid_strata"],
        value_vars=["auroc", "auprc"],
        var_name="metric",
        value_name="value",
    ).dropna(subset=["value"]).groupby(["scope", "score", "metric"]):
        a = g["value"].to_numpy(dtype=float)
        ci_rows.append({
            "scope": scope,
            "score": score,
            "metric": metric.upper(),
            "valid_reps": len(a),
            "bootstrap_mean": float(a.mean()),
            "ci95_low": float(np.quantile(a, 0.025)),
            "ci95_high": float(np.quantile(a, 0.975)),
        })

    delta_rows = []
    for (scope, comp, metric), g in deltas.groupby(["scope", "comparison", "metric"]):
        a = g["delta"].to_numpy(dtype=float)
        lo = float(np.quantile(a, 0.025))
        hi = float(np.quantile(a, 0.975))
        delta_rows.append({
            "scope": scope,
            "comparison": comp,
            "metric": metric,
            "valid_reps": len(a),
            "bootstrap_mean_delta": float(a.mean()),
            "ci95_low": lo,
            "ci95_high": hi,
            "ci_excludes_zero": bool(lo > 0 or hi < 0),
        })

    return pd.DataFrame(ci_rows), pd.DataFrame(delta_rows)


def self_test() -> None:
    q = np.array([0.1, 0.2, 0.3])
    x = np.array([0.0, 0.1, 0.15, 0.25, 0.5])
    s = assign_strata(x, q)
    assert s.tolist() == [1, 1, 2, 3, 4]
    assert BOOTSTRAP_REPS == 2000
    assert PRIMARY_FULL in SCORE_COLS
    print("SELF_TEST_STRATA=PASS")
    print("SELF_TEST_FROZEN_SCORE_REGISTRY=PASS")
    print("SELF_TEST=PASS")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="SafeTTA B7-M01 change magnitude versus HARM direction."
    )
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument(
        "--preflight-only",
        action="store_true",
        help="Verify all frozen bindings and source-defined strata; do not compute target scientific metrics.",
    )
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    print("=" * 168)
    print("SafeTTA B7-M01 — Change magnitude versus HARM direction")
    print("Version                    :", VERSION)
    print("New model fit              : NO")
    print("New inference/TTA          : NO")
    print("Target calibration/flip    : NO / NO")
    print("Target-defined bins        : NO")
    print("Primary magnitude          : 1 - prediction Dice(SOURCE,candidate)")
    print("Direction subset           : HARM vs BENEFIT; NEUTRAL excluded")
    print("Bootstrap                  : 2000 physical-image clusters")
    print("=" * 168)

    protocol = verify_protocol()
    p01b_meta = verify_p01b_outputs()
    mod = load_p01b_module()

    print("\n[1/5] Bind frozen target outcomes and scores")
    target = load_target_outcomes_and_scores()
    print("target_rows =", len(target))
    print("target_images =", target["sample_id"].nunique())
    print("target_states =", target["model_state_id"].nunique())
    print("harm_n =", int(target["harm"].sum()))
    print("benefit_n =", int(target["benefit"].sum()))
    print("FROZEN_SCORE_BINDING=PASS")

    print("\n[2/5] Reuse exact frozen candidate-mask lineage")
    source_geom, target, geom_meta = build_source_and_target_geometry(mod, target)
    q = frozen_source_quartiles(source_geom)
    print("source_magnitude_q25_q50_q75 =", [float(v) for v in q])
    print("SOURCE_DEFINED_STRATA=PASS")
    print("TARGET_LABELS_USED_FOR_STRATA=NO")

    if args.preflight_only:
        print("\nM01_PREFLIGHT=PASS")
        print("SCIENTIFIC_METRICS=NOT_RUN")
        print("NEXT=run without --preflight-only")
        return

    if OUT_DIR.exists():
        raise FileExistsError(
            f"Never overwrite B7-M01 results: {OUT_DIR}\n"
            "Use _fix1 only for implementation repair."
        )
    OUT_DIR.mkdir(parents=True, exist_ok=False)

    print("\n[3/5] HARM-versus-BENEFIT directionality analysis")
    overall, strata, macro = point_analysis(target, q)
    print("\nOVERALL HARM vs BENEFIT")
    print(overall.to_string(index=False))
    print("\nSOURCE-DEFINED MAGNITUDE STRATA")
    print(strata.to_string(index=False))
    print("\nMACRO STRATIFIED AUROC")
    print(macro.to_string(index=False))

    # Frozen support gate for the original four-stratum primary endpoint.
    # IMPORTANT: if support fails, we DO NOT change bins or redefine the primary.
    # We still execute the already-prespecified overall HARM-vs-BENEFIT secondary
    # bootstrap, because it was frozen in the protocol before target metrics.
    support = (
        strata[strata["score"] == PRIMARY_FULL]
        .set_index("stratum")["support_gate"]
        .to_dict()
    )
    all_four_supported = all(bool(support.get(k, False)) for k in [1, 2, 3, 4])
    if all_four_supported:
        primary_status = "EVALUABLE"
        print("PRIMARY_FOUR_STRATUM_SUPPORT=PASS")
    else:
        primary_status = "NOT_EVALUABLE_INSUFFICIENT_FROZEN_STRATUM_SUPPORT"
        print("PRIMARY_FOUR_STRATUM_SUPPORT=FAIL")
        print("PRIMARY_ENDPOINT_STATUS=" + primary_status)
        print("NO_REBINNING_OR_PRIMARY_REDEFINITION=PASS")

    print("\n[4/5] Paired physical-image clustered bootstrap")
    reps, deltas = clustered_bootstrap(target, q)
    ci, delta_ci = summarize_bootstrap(reps, deltas)

    primary_rows = delta_ci[
        (delta_ci["scope"] == "SOURCE_QUARTILE_MACRO")
        & (delta_ci["comparison"] == f"{PRIMARY_FULL} - {PRIMARY_MAG}")
        & (delta_ci["metric"] == "AUROC")
    ]

    primary_result = None
    if all_four_supported:
        if len(primary_rows) != 1:
            raise RuntimeError("Primary M01 paired delta row missing despite support PASS.")
        pr = primary_rows.iloc[0]
        primary_result = {
            "bootstrap_mean_delta": float(pr["bootstrap_mean_delta"]),
            "ci95_low": float(pr["ci95_low"]),
            "ci95_high": float(pr["ci95_high"]),
            "valid_reps": int(pr["valid_reps"]),
        }
        primary_supported = bool(float(pr["ci95_low"]) > 0)
        scientific_decision = (
            "FULL_TRANSITION_BEYOND_MAGNITUDE_SUPPORTED"
            if primary_supported
            else "FULL_TRANSITION_BEYOND_MAGNITUDE_NOT_SUPPORTED"
        )
    else:
        scientific_decision = (
            "PRIMARY_NOT_EVALUABLE__REPORT_PRESPECIFIED_OVERALL_SECONDARY_ONLY"
        )

    # Prespecified secondary: overall HARM-vs-BENEFIT paired contrast.
    secondary = delta_ci[
        (delta_ci["scope"] == "OVERALL")
        & (delta_ci["comparison"] == f"{PRIMARY_FULL} - {PRIMARY_MAG}")
        & (delta_ci["metric"].isin(["AUROC", "AUPRC"]))
    ].copy()
    if set(secondary["metric"]) != {"AUROC", "AUPRC"}:
        raise RuntimeError("Prespecified overall secondary full-vs-magnitude rows missing.")

    print("\nPAIRED BOOTSTRAP DELTAS")
    print(delta_ci.to_string(index=False))
    print("\nPRIMARY_ENDPOINT_STATUS =", primary_status)
    print("PRESPECIFIED_OVERALL_SECONDARY_FULL_VS_MAGNITUDE")
    print(secondary.to_string(index=False))
    print("\nSCIENTIFIC_DECISION =", scientific_decision)

    print("\n[5/5] Freeze post-B6 explanatory outputs")
    p_overall = OUT_DIR / "B7_M01_HARM_VS_BENEFIT_OVERALL.csv"
    p_strata = OUT_DIR / "B7_M01_SOURCE_DEFINED_MAGNITUDE_STRATA.csv"
    p_macro = OUT_DIR / "B7_M01_STRATIFIED_MACRO_POINT.csv"
    p_ci = OUT_DIR / "B7_M01_BOOTSTRAP_CI.csv"
    p_delta = OUT_DIR / "B7_M01_PAIRED_DELTAS.csv"
    p_reps = OUT_DIR / "B7_M01_BOOTSTRAP_REPLICATES.csv"
    p_bound = OUT_DIR / "B7_M01_BOUND_DIRECTION_PANEL.csv"

    overall.to_csv(p_overall, index=False)
    strata.to_csv(p_strata, index=False)
    macro.to_csv(p_macro, index=False)
    ci.to_csv(p_ci, index=False)
    delta_ci.to_csv(p_delta, index=False)
    reps.to_csv(p_reps, index=False)

    bound = target.copy()
    bound[PRIMARY_MAG] = 1.0 - bound["pred_dice"].astype(float)
    bound[SENS_MAG] = 1.0 - bound["pred_iou"].astype(float)
    bound["magnitude_stratum"] = assign_strata(bound[PRIMARY_MAG].to_numpy(dtype=float), q)
    bound.to_csv(p_bound, index=False)

    audit = {
        "status": "PASS",
        "gate": PASS_GATE,
        "version": VERSION,
        "protocol": protocol,
        "p01b": p01b_meta,
        "source_magnitude_quartiles": [float(v) for v in q],
        "source_cutpoints_use_outcomes": False,
        "target_labels_used_for_cutpoints": False,
        "new_model_fit": False,
        "new_inference_tta": False,
        "target_calibration": False,
        "score_flip": False,
        "direction_population": {
            "harm_n": int(((target["harm"] == 1) & (target["benefit"] == 0)).sum()),
            "benefit_n": int(((target["benefit"] == 1) & (target["harm"] == 0)).sum()),
        },
        "primary_endpoint": (
            f"macro source-quartile AUROC delta: {PRIMARY_FULL} - {PRIMARY_MAG}"
        ),
        "primary_support_by_stratum": {str(k): bool(support.get(k, False)) for k in [1, 2, 3, 4]},
        "primary_status": primary_status,
        "primary_result": primary_result,
        "prespecified_overall_secondary_full_vs_magnitude": (
            secondary.to_dict(orient="records")
        ),
        "scientific_decision": scientific_decision,
        "geometry_lineage": geom_meta,
        "outputs": {},
    }

    for p in [p_overall, p_strata, p_macro, p_ci, p_delta, p_reps, p_bound]:
        audit["outputs"][p.name] = {
            "path": str(p),
            "sha256": sha256_file(p),
            "bytes": p.stat().st_size,
        }

    p_audit = OUT_DIR / "B7_M01_CHANGE_MAGNITUDE_VS_HARM_DIRECTION_AUDIT.json"
    p_audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    report = "\n".join([
        "=" * 168,
        "SafeTTA B7-M01 CHANGE MAGNITUDE VS HARM DIRECTION COMPLETE",
        "",
        "Source magnitude quartiles:",
        str([float(v) for v in q]),
        "",
        "OVERALL HARM vs BENEFIT:",
        overall.to_string(index=False),
        "",
        "STRATIFIED MACRO POINT:",
        macro.to_string(index=False),
        "",
        "PAIRED BOOTSTRAP DELTAS:",
        delta_ci.to_string(index=False),
        "",
        f"PRIMARY_ENDPOINT_STATUS={primary_status}",
        "PRESPECIFIED_OVERALL_SECONDARY_FULL_VS_MAGNITUDE:",
        secondary.to_string(index=False),
        "",
        f"SCIENTIFIC_DECISION={scientific_decision}",
        f"GATE={PASS_GATE}",
        f"audit_json={p_audit}",
        "=" * 168,
        "",
    ])
    p_report = OUT_DIR / "B7_M01_CHANGE_MAGNITUDE_VS_HARM_DIRECTION_REPORT.txt"
    p_report.write_text(report, encoding="utf-8")
    print("\n" + report)


if __name__ == "__main__":
    main()
