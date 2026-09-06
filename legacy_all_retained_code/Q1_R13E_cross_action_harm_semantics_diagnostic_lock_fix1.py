#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R13E_cross_action_harm_semantics_diagnostic_lock_fix1.py

Post-hoc descriptive audit only.

Purpose:
Explain WHY the frozen TENT-developed safety score transfers to PL-CONF90 only
with strong attenuation, without changing or refitting any method.

Reads only the already-locked R13D evaluation panel and R13D lock.
No model inference, no GT decoding, no fitting, no threshold change, no
PolypGen access.

Outputs:
- TENT1 vs PL HARM 2x2 transition
- overlap/agreement/Jaccard/Cohen-kappa
- P(PL harm | TENT harm) and P(TENT harm | PL harm)
- four cross-action strata score summaries
- family-wise overlap diagnostics
- case-level 9-state harm-burden correlation
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score


ROOT = Path(r"F:\MEDSEG_SAFETTA")
OUT = ROOT / "outputs"

R13D_DIR = OUT / "Q1_R13D_pl_conf90_frozen_safety_transfer_evaluation_fix2_v1"
R13D_LOCK = R13D_DIR / "R13D_PL_SAFETY_TRANSFER_EVALUATION_LOCK.json"
R13D_PANEL = R13D_DIR / "R13D_locked_score_pl_evaluation_panel.csv"

EXPECTED_R13D_LOCK_SHA = (
    "5f9a73008c3747759deb5da7c47e015f6cad130478e903cc8acd4c71ec6e2ce7"
)
EXPECTED_R13D_DECISIONS = {
    "PL_CONF90_FROZEN_SAFETY_RANKING_TRANSFER_SUPPORTED",
    "PL_CONF90_FROZEN_SAFETY_RANKING_TRANSFER_SUPPORTED_WITH_ATTENUATION",
    "PL_CONF90_FROZEN_SAFETY_RANKING_TRANSFER_NOT_SUPPORTED",
}

OUTPUT_DIR = (
    OUT / "Q1_R13E_cross_action_harm_semantics_diagnostic_lock_fix1_v1"
)

DECISION = "CROSS_ACTION_HARM_SEMANTICS_DIAGNOSTIC_LOCKED_NO_METHOD_CHANGE"


def sha256_file(path: Path, chunk=8 * 1024 * 1024):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def verify():
    for p in (R13D_LOCK, R13D_PANEL):
        if not p.exists():
            raise FileNotFoundError(p)

    actual = sha256_file(R13D_LOCK)
    if actual != EXPECTED_R13D_LOCK_SHA:
        raise RuntimeError(
            f"R13D lock SHA mismatch: expected={EXPECTED_R13D_LOCK_SHA} actual={actual}"
        )

    lock = json.loads(R13D_LOCK.read_text(encoding="utf-8"))
    if lock.get("decision") not in EXPECTED_R13D_DECISIONS:
        raise RuntimeError(f"Unexpected R13D decision: {lock.get('decision')}")

    expected_panel_sha = lock.get("artifacts", {}).get(R13D_PANEL.name)
    if not expected_panel_sha:
        raise RuntimeError("R13D lock missing evaluation-panel SHA.")
    if sha256_file(R13D_PANEL) != expected_panel_sha:
        raise RuntimeError("R13D evaluation-panel SHA mismatch.")

    return lock, actual


def overlap_metrics(y_a1, y_pl):
    y_a1 = np.asarray(y_a1, dtype=int)
    y_pl = np.asarray(y_pl, dtype=int)

    both = int(np.sum((y_a1 == 1) & (y_pl == 1)))
    a1_only = int(np.sum((y_a1 == 1) & (y_pl == 0)))
    pl_only = int(np.sum((y_a1 == 0) & (y_pl == 1)))
    neither = int(np.sum((y_a1 == 0) & (y_pl == 0)))

    union = both + a1_only + pl_only
    a1_pos = both + a1_only
    pl_pos = both + pl_only

    return {
        "rows": int(len(y_a1)),
        "both_harm": both,
        "a1_only_harm": a1_only,
        "pl_only_harm": pl_only,
        "neither_harm": neither,
        "agreement": float(np.mean(y_a1 == y_pl)),
        "jaccard_positive": float(both / union) if union else np.nan,
        "cohen_kappa": float(cohen_kappa_score(y_a1, y_pl)),
        "p_pl_harm_given_a1_harm": float(both / a1_pos) if a1_pos else np.nan,
        "p_a1_harm_given_pl_harm": float(both / pl_pos) if pl_pos else np.nan,
    }


def assign_stratum(df):
    a = df["tent1_harm_label"].to_numpy(int)
    p = df["pl_harm_label"].to_numpy(int)
    out = np.empty(len(df), dtype=object)
    out[(a == 1) & (p == 1)] = "BOTH_HARM"
    out[(a == 1) & (p == 0)] = "A1_ONLY_HARM"
    out[(a == 0) & (p == 1)] = "PL_ONLY_HARM"
    out[(a == 0) & (p == 0)] = "NEITHER_HARM"
    return out


def score_strata(df):
    x = df.copy()
    x["cross_action_stratum"] = assign_stratum(x)

    order = [
        "BOTH_HARM",
        "A1_ONLY_HARM",
        "PL_ONLY_HARM",
        "NEITHER_HARM",
    ]

    rows = []
    for s in order:
        g = x[x["cross_action_stratum"] == s]
        q = g["frozen_probability"].quantile([0.25, 0.5, 0.75])
        rows.append({
            "cross_action_stratum": s,
            "rows": int(len(g)),
            "cases": int(g["sample_id"].nunique()),
            "score_mean": float(g["frozen_probability"].mean()),
            "score_q25": float(q.loc[0.25]),
            "score_median": float(q.loc[0.5]),
            "score_q75": float(q.loc[0.75]),
            "frozen_flag_rate": float(g["frozen_flag"].mean()),
            "tent1_delta_dice_mean": float(g["tent1_delta_dice"].mean()),
            "pl_delta_dice_mean": float(g["pl_delta_dice"].mean()),
        })
    return pd.DataFrame(rows)


def family_overlap(df):
    rows = []
    for fam, g in df.groupby("model_family", sort=True):
        m = overlap_metrics(
            g["tent1_harm_label"].to_numpy(int),
            g["pl_harm_label"].to_numpy(int),
        )
        rows.append({"model_family": fam, **m})
    return pd.DataFrame(rows)


def case_burden(df):
    # 9 model states per physical case.
    case = (
        df.groupby("sample_id", as_index=False)
        .agg(
            tent1_harm_state_count=("tent1_harm_label", "sum"),
            pl_harm_state_count=("pl_harm_label", "sum"),
            mean_frozen_probability=("frozen_probability", "mean"),
        )
    )
    if len(case) != 1000:
        raise RuntimeError(f"Expected 1000 physical cases, got {len(case)}")

    rho, pval = spearmanr(
        case["tent1_harm_state_count"].to_numpy(float),
        case["pl_harm_state_count"].to_numpy(float),
    )
    return case, float(rho), float(pval)


def self_test():
    m = overlap_metrics([1, 1, 0, 0], [1, 0, 1, 0])
    assert m["both_harm"] == 1
    assert m["a1_only_harm"] == 1
    assert m["pl_only_harm"] == 1
    assert m["neither_harm"] == 1
    assert abs(m["agreement"] - 0.5) < 1e-12
    print("OVERLAP_METRICS_TEST_PASS")
    print("SELF_TEST_PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    print("===== R13E CROSS-ACTION HARM SEMANTICS DIAGNOSTIC FIX1 =====")
    print("descriptive post-hoc audit=YES")
    print("model fit/refit=NO")
    print("threshold change=NO")
    print("score reversal=NO")
    print("PolypGen access=NO")
    print("method selection/tuning=NO")
    print()

    r13d_lock, lock_sha = verify()
    df = pd.read_csv(R13D_PANEL, low_memory=False)

    required = [
        "sample_id",
        "model_family",
        "model_state_id",
        "frozen_probability",
        "frozen_flag",
        "tent1_delta_dice",
        "tent1_harm_label",
        "pl_delta_dice",
        "pl_harm_label",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"R13D panel missing: {missing}")

    if len(df) != 9000:
        raise RuntimeError(f"Rows={len(df)}")
    if df["sample_id"].nunique() != 1000:
        raise RuntimeError("Physical-case count !=1000.")
    if df["model_state_id"].nunique() != 9:
        raise RuntimeError("State count !=9.")

    global_overlap = overlap_metrics(
        df["tent1_harm_label"].to_numpy(int),
        df["pl_harm_label"].to_numpy(int),
    )
    fam = family_overlap(df)
    strata = score_strata(df)
    case, burden_rho, burden_p = case_burden(df)

    transition = pd.crosstab(
        df["tent1_harm_label"],
        df["pl_harm_label"],
        rownames=["TENT1_harm"],
        colnames=["PL_harm"],
        dropna=False,
    )

    delta_rho, delta_p = spearmanr(
        df["tent1_delta_dice"].to_numpy(float),
        df["pl_delta_dice"].to_numpy(float),
    )

    print("===== GLOBAL HARM OVERLAP =====")
    for k, v in global_overlap.items():
        print(f"{k}: {v}")

    print("\n===== TENT1 -> PL HARM TRANSITION =====")
    print(transition.to_string())

    print("\n===== FROZEN SCORE BY CROSS-ACTION STRATUM =====")
    print(strata.to_string(index=False))

    print("\n===== FAMILY OVERLAP =====")
    print(fam.to_string(index=False))

    print("\n===== CASE-LEVEL ACTION BURDEN =====")
    print("Spearman TENT1-harm-state-count vs PL-harm-state-count:", burden_rho)
    print("p-value:", burden_p)
    print("Model-case DeltaDice Spearman:", float(delta_rho))
    print("p-value:", float(delta_p))

    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)

    transition_path = args.output_dir / "R13E_TENT1_to_PL_harm_transition.csv"
    strata_path = args.output_dir / "R13E_score_by_cross_action_stratum.csv"
    family_path = args.output_dir / "R13E_family_harm_overlap.csv"
    case_path = args.output_dir / "R13E_case_level_harm_burden.csv"

    transition.to_csv(transition_path)
    strata.to_csv(strata_path, index=False)
    fam.to_csv(family_path, index=False)
    case.to_csv(case_path, index=False)

    summary = {
        "status": "FROZEN",
        "decision": DECISION,
        "r13d_lock_sha256": lock_sha,
        "r13d_decision": r13d_lock.get("decision"),
        "global_overlap": global_overlap,
        "model_case_delta_dice_spearman": float(delta_rho),
        "model_case_delta_dice_spearman_p": float(delta_p),
        "case_level_harm_burden_spearman": burden_rho,
        "case_level_harm_burden_spearman_p": burden_p,
        "information_boundary": {
            "descriptive_posthoc_only": True,
            "model_fit_or_refit": False,
            "threshold_change": False,
            "score_reversal": False,
            "polypgen_access": False,
            "method_selection_or_tuning": False,
        },
        "paper_interpretation": (
            "Cross-TTA safety ranking transfer is statistically present but "
            "strongly attenuated because downstream harm semantics are action-"
            "dependent; TENT1 and PL-CONF90 define substantially different "
            "harm subsets."
        ),
        "artifacts": {
            transition_path.name: sha256_file(transition_path),
            strata_path.name: sha256_file(strata_path),
            family_path.name: sha256_file(family_path),
            case_path.name: sha256_file(case_path),
        },
    }

    lock_path = args.output_dir / "R13E_CROSS_ACTION_HARM_SEMANTICS_LOCK.json"
    lock_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nDecision=", DECISION)
    print("R13E LOCK:", lock_path)
    print("R13E LOCK SHA256:", sha256_file(lock_path))
    print("PASS")


if __name__ == "__main__":
    main()
