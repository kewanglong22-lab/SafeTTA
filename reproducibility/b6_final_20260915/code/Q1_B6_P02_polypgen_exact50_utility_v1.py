#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SafeTTA B6-P0-2 — PolypGen exact-50 matched-coverage deployment utility.

Frozen before P0-2 utility metrics:
- no new model fit;
- no target calibration;
- no threshold/coverage tuning;
- higher score = higher HARM risk;
- exactly 766 / 1532 lowest-risk images selected within each of 3 states;
- MEMO is committed on selected rows; SOURCE is retained otherwise;
- physical-image clustered bootstrap, 2000 reps, seed 20260918;
- selection is fixed on the full panel before bootstrap.

Primary endpoint:
  mean_deployed_dice

Primary comparison:
  SOURCE_PLUS_SIMPLE_MASK_CHANGE - SOURCE_STATE

This script consumes already-frozen P0-1B scores. It does NOT refit the
SOURCE-state or geometry models and does NOT recompute candidate masks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


VERSION = "2026-09-15-B6-P02-v1"

ROOT = Path(r"F:\MEDSEG_SAFETTA")
CODE = ROOT / "code"

PROTOCOL_PATH = CODE / "B6_P02_POLYPGEN_EXACT50_UTILITY_PROTOCOL_v1.json"
EXPECTED_PROTOCOL_SHA256 = "f7551127d6960f3323ec52b5554b5c4cdf16b700315344786afa972ff5410d12"

P01B_DIR = ROOT / "B6_P01B_polypgen_geometry_core_experiment_v1_fix5"
P01B_AUDIT = P01B_DIR / "B6_P01B_POLYPGEN_GEOMETRY_CORE_AUDIT.json"
P01B_SCORES = P01B_DIR / "B6_P01B_POLYPGEN_GEOMETRY_CORE_SCORES.csv"

TARGET_PANEL = (
    ROOT
    / "R33A3_first_polypgen_memo_harm_reveal_external_joint_shift_evaluation_v1"
    / "R33A3_POLYPGEN_MEMO_MATCHED_SCORE_OUTCOME_PANEL.csv"
)

OUT_DIR = ROOT / "B6_P02_polypgen_exact50_utility_v1"
PASS_GATE = "PASS_B6_P02_POLYPGEN_EXACT50_UTILITY_COMPLETE"
PREFLIGHT_GATE = "PASS_B6_P02_POLYPGEN_EXACT50_UTILITY_PREFLIGHT"

EXPECTED_P01B_GATE = "PASS_B6_P01B_POLYPGEN_GEOMETRY_CORE_EXPERIMENT_COMPLETE"

N_IMAGES = 1532
N_STATES = 3
N_ROWS = 4596
SELECT_PER_STATE = 766
TOTAL_SELECTED = 2298
COVERAGE = 0.5

BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 20260918

KEY_COLS = ["sample_id", "model_state_id"]

METHODS = [
    "SOURCE_STATE",
    "SOURCE_PLUS_SIMPLE_MASK_CHANGE",
    "FROZEN_FULL_TRANSITION_SAFETTA",
]

PAIRWISE = [
    ("SOURCE_PLUS_SIMPLE_MASK_CHANGE", "SOURCE_STATE"),
    ("FROZEN_FULL_TRANSITION_SAFETTA", "SOURCE_STATE"),
    ("SOURCE_PLUS_SIMPLE_MASK_CHANGE", "FROZEN_FULL_TRANSITION_SAFETTA"),
]

HIGHER_IS_BETTER_METRICS = [
    "mean_deployed_dice",
    "mean_deployed_delta_vs_source",
    "prevented_harm_fraction",
    "benefit_capture_fraction",
]

ALL_UTILITY_METRICS = HIGHER_IS_BETTER_METRICS + [
    "committed_harm_rate",
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


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def norm_id(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.strip()
        .str.replace("\\", "/", regex=False)
        .str.lower()
    )


def verify_protocol() -> Dict[str, Any]:
    if not PROTOCOL_PATH.is_file():
        raise FileNotFoundError(PROTOCOL_PATH)

    observed = sha256_file(PROTOCOL_PATH)
    if observed.lower() != EXPECTED_PROTOCOL_SHA256.lower():
        raise RuntimeError(
            "P0-2 protocol SHA mismatch.\n"
            f"expected={EXPECTED_PROTOCOL_SHA256}\n"
            f"observed={observed}"
        )

    p = load_json(PROTOCOL_PATH)
    if p.get("status") != "FROZEN_BEFORE_P02_UTILITY_METRICS":
        raise RuntimeError("P0-2 protocol status changed.")
    if float(p["deployment_policy"]["coverage"]) != COVERAGE:
        raise RuntimeError("P0-2 coverage changed.")
    if int(p["deployment_policy"]["selected_per_state"]) != SELECT_PER_STATE:
        raise RuntimeError("P0-2 selected_per_state changed.")

    return {
        "path": str(PROTOCOL_PATH),
        "sha256": observed,
        "status": p.get("status"),
    }


def verify_p01b() -> Dict[str, Any]:
    if not P01B_AUDIT.is_file():
        raise FileNotFoundError(P01B_AUDIT)
    if not P01B_SCORES.is_file():
        raise FileNotFoundError(P01B_SCORES)

    a = load_json(P01B_AUDIT)
    if a.get("gate") != EXPECTED_P01B_GATE:
        raise RuntimeError(
            f"P0-1B gate changed: {a.get('gate')}"
        )
    if a.get("decision") != "CANDIDATE_CONDITIONING_INCREMENT_SUPPORTED":
        raise RuntimeError(
            f"P0-1B decision changed: {a.get('decision')}"
        )

    return {
        "audit_path": str(P01B_AUDIT),
        "audit_sha256": sha256_file(P01B_AUDIT),
        "scores_path": str(P01B_SCORES),
        "scores_sha256": sha256_file(P01B_SCORES),
        "p01b_gate": a.get("gate"),
        "p01b_decision": a.get("decision"),
    }


def load_exact_panel() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if not TARGET_PANEL.is_file():
        raise FileNotFoundError(TARGET_PANEL)

    t = pd.read_csv(TARGET_PANEL, low_memory=False)
    required = {
        "sample_id",
        "model_state_id",
        "action",
        "source_dice",
        "memo_dice",
        "memo_delta_dice",
        "memo_harm_label",
        "memo_benefit_label",
    }
    missing = sorted(required - set(t.columns))
    if missing:
        raise RuntimeError(
            f"Target outcome panel missing columns: {missing}"
        )

    if len(t) != N_ROWS:
        raise RuntimeError(f"Target rows={len(t)} != {N_ROWS}")
    if t["sample_id"].astype(str).nunique() != N_IMAGES:
        raise RuntimeError("Target physical-image count changed.")
    if t["model_state_id"].astype(str).nunique() != N_STATES:
        raise RuntimeError("Target state count changed.")
    if not t["action"].astype(str).str.contains("MEMO", case=False, regex=False).all():
        raise RuntimeError("Target action changed from MEMO.")

    panel = pd.DataFrame({
        "sample_id": norm_id(t["sample_id"]),
        "model_state_id": t["model_state_id"].astype(str).str.strip(),
        "source_dice": pd.to_numeric(t["source_dice"], errors="raise").astype(float),
        "memo_dice": pd.to_numeric(t["memo_dice"], errors="raise").astype(float),
        "memo_delta_dice": pd.to_numeric(t["memo_delta_dice"], errors="raise").astype(float),
        "harm_label": pd.to_numeric(t["memo_harm_label"], errors="raise").astype(int),
        "benefit_label": pd.to_numeric(t["memo_benefit_label"], errors="raise").astype(int),
    })

    if panel[KEY_COLS].drop_duplicates().shape[0] != N_ROWS:
        raise RuntimeError("Target sample/state key not unique.")

    expected_harm = (panel["memo_delta_dice"].to_numpy() <= -0.02).astype(int)
    expected_benefit = (panel["memo_delta_dice"].to_numpy() >= +0.02).astype(int)

    if not np.array_equal(expected_harm, panel["harm_label"].to_numpy(dtype=int)):
        raise RuntimeError("Target HARM rule reproduction failed.")
    if not np.array_equal(expected_benefit, panel["benefit_label"].to_numpy(dtype=int)):
        raise RuntimeError("Target BENEFIT rule reproduction failed.")

    dice_delta = panel["memo_dice"].to_numpy() - panel["source_dice"].to_numpy()
    if not np.allclose(
        dice_delta,
        panel["memo_delta_dice"].to_numpy(),
        atol=1e-10,
        rtol=0.0,
        equal_nan=False,
    ):
        max_abs = float(np.max(np.abs(dice_delta - panel["memo_delta_dice"].to_numpy())))
        raise RuntimeError(
            f"memo_dice - source_dice != memo_delta_dice; max_abs={max_abs}"
        )

    s = pd.read_csv(P01B_SCORES, low_memory=False)
    required_scores = set(KEY_COLS + ["harm_label"] + METHODS)
    miss2 = sorted(required_scores - set(s.columns))
    if miss2:
        raise RuntimeError(f"P0-1B score file missing columns: {miss2}")
    if len(s) != N_ROWS:
        raise RuntimeError("P0-1B score row count changed.")

    scores = pd.DataFrame({
        "sample_id": norm_id(s["sample_id"]),
        "model_state_id": s["model_state_id"].astype(str).str.strip(),
        "harm_label_p01b": pd.to_numeric(s["harm_label"], errors="raise").astype(int),
    })
    for m in METHODS:
        scores[m] = pd.to_numeric(s[m], errors="raise").astype(float)

    if scores[KEY_COLS].drop_duplicates().shape[0] != N_ROWS:
        raise RuntimeError("P0-1B score key not unique.")

    joined = panel.merge(
        scores,
        on=KEY_COLS,
        how="left",
        validate="one_to_one",
    )
    if joined[METHODS].isna().any().any():
        raise RuntimeError("P0-1B scores do not cover exact target keys.")
    if not np.array_equal(
        joined["harm_label"].to_numpy(dtype=int),
        joined["harm_label_p01b"].to_numpy(dtype=int),
    ):
        raise RuntimeError("P0-1B HARM labels disagree with R33A3.")

    if not np.isfinite(joined[METHODS].to_numpy(dtype=float)).all():
        raise RuntimeError("Non-finite P0-1B score.")

    joined = joined.drop(columns=["harm_label_p01b"])

    return joined, {
        "target_path": str(TARGET_PANEL),
        "target_sha256": sha256_file(TARGET_PANEL),
        "scores_path": str(P01B_SCORES),
        "scores_sha256": sha256_file(P01B_SCORES),
        "rows": N_ROWS,
        "images": N_IMAGES,
        "states": N_STATES,
        "harm_n": int(joined["harm_label"].sum()),
        "benefit_n": int(joined["benefit_label"].sum()),
        "harm_prevalence": float(joined["harm_label"].mean()),
        "benefit_prevalence": float(joined["benefit_label"].mean()),
        "harm_rule_reproduction": "PASS",
        "benefit_rule_reproduction": "PASS",
        "dice_delta_identity": "PASS",
    }


def freeze_exact50_selection(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()

    for method in METHODS:
        col = f"selected_{method}"
        out[col] = 0

        total = 0
        state_counts = {}

        for state_id, g in out.groupby("model_state_id", sort=True):
            if len(g) != N_IMAGES:
                raise RuntimeError(
                    f"State {state_id} rows={len(g)} != {N_IMAGES}"
                )

            ranked = g.sort_values(
                [method, "sample_id"],
                ascending=[True, True],
                kind="mergesort",
            )
            chosen = ranked.index[:SELECT_PER_STATE]
            out.loc[chosen, col] = 1

            n = int(out.loc[g.index, col].sum())
            if n != SELECT_PER_STATE:
                raise RuntimeError(
                    f"{method}/{state_id} selected={n} != {SELECT_PER_STATE}"
                )
            state_counts[state_id] = n
            total += n

        if total != TOTAL_SELECTED:
            raise RuntimeError(
                f"{method} total selected={total} != {TOTAL_SELECTED}"
            )

    return out


def method_metrics(panel: pd.DataFrame, method: str) -> Dict[str, Any]:
    sel = panel[f"selected_{method}"].to_numpy(dtype=int)
    src = panel["source_dice"].to_numpy(dtype=float)
    memo = panel["memo_dice"].to_numpy(dtype=float)
    delta = panel["memo_delta_dice"].to_numpy(dtype=float)
    harm = panel["harm_label"].to_numpy(dtype=int)
    benefit = panel["benefit_label"].to_numpy(dtype=int)

    deployed_dice = np.where(sel == 1, memo, src)
    deployed_delta = np.where(sel == 1, delta, 0.0)

    harm_n = int(harm.sum())
    benefit_n = int(benefit.sum())
    selected_n = int(sel.sum())
    selected_harm = int((sel * harm).sum())
    prevented_harm = int(((1 - sel) * harm).sum())
    selected_benefit = int((sel * benefit).sum())

    return {
        "representation": method,
        "coverage": selected_n / len(panel),
        "selected_n": selected_n,
        "mean_deployed_dice": float(deployed_dice.mean()),
        "mean_deployed_delta_vs_source": float(deployed_delta.mean()),
        "prevented_harm_n": prevented_harm,
        "prevented_harm_fraction": float(prevented_harm / harm_n),
        "committed_harm_n": selected_harm,
        "committed_harm_rate": float(selected_harm / selected_n),
        "captured_benefit_n": selected_benefit,
        "benefit_capture_fraction": float(selected_benefit / benefit_n),
        "mean_selected_delta_dice": float(delta[sel == 1].mean()),
        "source_mean_dice": float(src.mean()),
        "deploy_all_memo_mean_dice": float(memo.mean()),
        "deploy_all_mean_delta_vs_source": float(delta.mean()),
        "harm_n": harm_n,
        "benefit_n": benefit_n,
    }


def make_contribution_table(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for method in METHODS:
        sel = panel[f"selected_{method}"].to_numpy(dtype=int)
        src = panel["source_dice"].to_numpy(dtype=float)
        memo = panel["memo_dice"].to_numpy(dtype=float)
        delta = panel["memo_delta_dice"].to_numpy(dtype=float)
        harm = panel["harm_label"].to_numpy(dtype=int)
        benefit = panel["benefit_label"].to_numpy(dtype=int)

        x = pd.DataFrame({
            "sample_id": panel["sample_id"].to_numpy(),
            "method": method,
            "row_n": 1,
            "selected_n": sel,
            "deployed_dice_sum": np.where(sel == 1, memo, src),
            "deployed_delta_sum": np.where(sel == 1, delta, 0.0),
            "harm_n": harm,
            "prevented_harm_n": (1 - sel) * harm,
            "committed_harm_n": sel * harm,
            "benefit_n": benefit,
            "captured_benefit_n": sel * benefit,
        })
        rows.append(x)

    long = pd.concat(rows, ignore_index=True)
    agg = (
        long.groupby(["sample_id", "method"], as_index=False)
        .sum(numeric_only=True)
    )

    # Exactly 3 state rows per image per method.
    if not (agg["row_n"] == N_STATES).all():
        raise RuntimeError("Bootstrap physical-image cluster does not contain exactly 3 states.")

    return agg


def metrics_from_cluster_rows(x: pd.DataFrame) -> Dict[str, float]:
    row_n = float(x["row_n"].sum())
    selected_n = float(x["selected_n"].sum())
    harm_n = float(x["harm_n"].sum())
    benefit_n = float(x["benefit_n"].sum())

    return {
        "mean_deployed_dice": float(x["deployed_dice_sum"].sum() / row_n),
        "mean_deployed_delta_vs_source": float(x["deployed_delta_sum"].sum() / row_n),
        "prevented_harm_fraction": (
            float(x["prevented_harm_n"].sum() / harm_n)
            if harm_n > 0 else np.nan
        ),
        "benefit_capture_fraction": (
            float(x["captured_benefit_n"].sum() / benefit_n)
            if benefit_n > 0 else np.nan
        ),
        "committed_harm_rate": (
            float(x["committed_harm_n"].sum() / selected_n)
            if selected_n > 0 else np.nan
        ),
    }


def bootstrap_utility(panel: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    contrib = make_contribution_table(panel)

    ids = sorted(panel["sample_id"].unique().tolist())
    if len(ids) != N_IMAGES:
        raise RuntimeError("Bootstrap image count changed.")

    per_method = {}
    for method in METHODS:
        x = contrib[contrib["method"] == method].copy()
        x = x.set_index("sample_id").loc[ids].reset_index()
        per_method[method] = x

    rng = np.random.default_rng(BOOTSTRAP_SEED)

    rep_rows = []
    delta_rows = []

    for rep in tqdm(
        range(BOOTSTRAP_REPS),
        desc="P02 exact50 physical-image bootstrap",
        unit="rep",
        dynamic_ncols=True,
    ):
        draw = rng.integers(0, N_IMAGES, size=N_IMAGES)

        vals = {}
        for method in METHODS:
            x = per_method[method].iloc[draw]
            vals[method] = metrics_from_cluster_rows(x)

            rep_rows.append({
                "rep": rep,
                "representation": method,
                **vals[method],
            })

        for a, b in PAIRWISE:
            for metric in ALL_UTILITY_METRICS:
                va = vals[a][metric]
                vb = vals[b][metric]
                delta_rows.append({
                    "rep": rep,
                    "comparison": f"{a} - {b}",
                    "metric": metric,
                    "delta": va - vb,
                })

    return pd.DataFrame(rep_rows), pd.DataFrame(delta_rows)


def summarize_bootstrap(
    reps: pd.DataFrame,
    deltas: pd.DataFrame,
    point_metrics: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ci_rows = []
    for (method, metric), g in (
        reps.melt(
            id_vars=["rep", "representation"],
            value_vars=ALL_UTILITY_METRICS,
            var_name="metric",
            value_name="value",
        )
        .groupby(["representation", "metric"], sort=False)
    ):
        vals = g["value"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        ci_rows.append({
            "representation": method,
            "metric": metric,
            "valid_reps": len(vals),
            "bootstrap_mean": float(np.mean(vals)),
            "ci95_low": float(np.quantile(vals, 0.025)),
            "ci95_high": float(np.quantile(vals, 0.975)),
        })

    point = point_metrics.set_index("representation")

    delta_rows = []
    for (comparison, metric), g in deltas.groupby(
        ["comparison", "metric"],
        sort=False,
    ):
        vals = g["delta"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]

        a, b = comparison.split(" - ")
        point_delta = float(point.loc[a, metric] - point.loc[b, metric])

        lo = float(np.quantile(vals, 0.025))
        hi = float(np.quantile(vals, 0.975))

        delta_rows.append({
            "comparison": comparison,
            "metric": metric,
            "valid_reps": len(vals),
            "point_delta": point_delta,
            "bootstrap_mean_delta": float(np.mean(vals)),
            "ci95_low": lo,
            "ci95_high": hi,
            "ci_excludes_zero": bool(lo > 0 or hi < 0),
        })

    return pd.DataFrame(ci_rows), pd.DataFrame(delta_rows)


def preflight(panel: pd.DataFrame) -> Dict[str, Any]:
    selected = freeze_exact50_selection(panel)

    counts = {}
    for m in METHODS:
        per_state = (
            selected.groupby("model_state_id")[f"selected_{m}"]
            .sum()
            .astype(int)
            .to_dict()
        )
        counts[m] = per_state
        if any(v != SELECT_PER_STATE for v in per_state.values()):
            raise RuntimeError(f"{m} exact50 state counts changed: {per_state}")
        if sum(per_state.values()) != TOTAL_SELECTED:
            raise RuntimeError(f"{m} exact50 total changed.")

    return {
        "rows": len(selected),
        "images": selected["sample_id"].nunique(),
        "states": selected["model_state_id"].nunique(),
        "selection_counts": counts,
        "coverage": COVERAGE,
    }


def self_test() -> None:
    assert N_IMAGES * N_STATES == N_ROWS
    assert SELECT_PER_STATE * N_STATES == TOTAL_SELECTED
    assert SELECT_PER_STATE / N_IMAGES == COVERAGE
    assert BOOTSTRAP_REPS == 2000
    print("SELF_TEST_COUNTS=PASS")
    print("SELF_TEST_COVERAGE=PASS")
    print("SELF_TEST=PASS")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="SafeTTA P0-2 PolypGen exact50 matched-coverage utility."
    )
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--preflight-only", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    print("=" * 164)
    print("SafeTTA B6-P0-2 — PolypGen exact-50 matched-coverage deployment utility")
    print(f"Version              : {VERSION}")
    print("New model fitting    : NO")
    print("Target calibration   : NO")
    print("Coverage tuning      : NO")
    print("Coverage             : 0.50 within each state")
    print("Selection            : 766 lowest-risk / 1532 per state")
    print("Bootstrap            : physical-image clustered, 2000 reps")
    print("=" * 164)

    print("\n[1/5] Verify frozen P0-2 protocol and completed P0-1B scores")
    protocol_meta = verify_protocol()
    p01b_meta = verify_p01b()
    print("PROTOCOL_SHA256 =", protocol_meta["sha256"])
    print("P01B_GATE       =", p01b_meta["p01b_gate"])
    print("P01B_DECISION   =", p01b_meta["p01b_decision"])

    print("\n[2/5] Exact R33A3 outcome + P0-1B score binding")
    panel, panel_meta = load_exact_panel()
    print("rows      =", len(panel))
    print("images    =", panel["sample_id"].nunique())
    print("states    =", panel["model_state_id"].nunique())
    print("harm_n    =", int(panel["harm_label"].sum()))
    print("benefit_n =", int(panel["benefit_label"].sum()))
    print("EXACT_PANEL_BINDING=PASS")

    print("\n[3/5] Freeze exact-50 selections")
    pf = preflight(panel)
    selected = freeze_exact50_selection(panel)

    for m in METHODS:
        print(m)
        print(
            selected.groupby("model_state_id")[f"selected_{m}"]
            .sum()
            .astype(int)
            .to_string()
        )
    print("EXACT50_SELECTION=PASS")

    if args.preflight_only:
        print("\nP02_EXACT50_PREFLIGHT=PASS")
        print("UTILITY_METRICS=NOT_RUN")
        print(f"GATE={PREFLIGHT_GATE}")
        return

    if OUT_DIR.exists():
        raise FileExistsError(
            f"Never overwrite P0-2 stage: {OUT_DIR}\n"
            "Use _fix1 only for implementation repair."
        )
    OUT_DIR.mkdir(parents=True, exist_ok=False)

    print("\n[4/5] Point deployment utility")
    point_rows = [method_metrics(selected, m) for m in METHODS]
    metrics_df = pd.DataFrame(point_rows)

    print(metrics_df[[
        "representation",
        "coverage",
        "mean_deployed_dice",
        "mean_deployed_delta_vs_source",
        "prevented_harm_fraction",
        "committed_harm_rate",
        "benefit_capture_fraction",
    ]].to_string(index=False))

    print("\n[5/5] Paired physical-image bootstrap")
    reps, deltas = bootstrap_utility(selected)
    ci_df, delta_df = summarize_bootstrap(reps, deltas, metrics_df)

    print("\nPAIRED DELTAS")
    print(delta_df.to_string(index=False))

    primary = delta_df[
        (delta_df["comparison"] == "SOURCE_PLUS_SIMPLE_MASK_CHANGE - SOURCE_STATE")
        & (delta_df["metric"] == "mean_deployed_dice")
    ]
    safety = delta_df[
        (delta_df["comparison"] == "SOURCE_PLUS_SIMPLE_MASK_CHANGE - SOURCE_STATE")
        & (delta_df["metric"] == "prevented_harm_fraction")
    ]

    if len(primary) != 1 or len(safety) != 1:
        raise RuntimeError("Primary/safety decision row missing.")

    pr = primary.iloc[0]
    sr = safety.iloc[0]

    utility_supported = bool(
        float(pr["point_delta"]) > 0
        and float(pr["ci95_low"]) > 0
    )
    safety_supported = bool(
        float(sr["point_delta"]) > 0
        and float(sr["ci95_low"]) > 0
    )

    if utility_supported and safety_supported:
        decision = "CANDIDATE_CONDITIONING_EXACT50_UTILITY_AND_SAFETY_SUPPORTED"
    elif utility_supported:
        decision = "CANDIDATE_CONDITIONING_EXACT50_UTILITY_SUPPORTED_SAFETY_NOT_CONFIRMED"
    elif safety_supported:
        decision = "CANDIDATE_CONDITIONING_EXACT50_SAFETY_SUPPORTED_UTILITY_NOT_CONFIRMED"
    else:
        decision = "CANDIDATE_CONDITIONING_EXACT50_INCREMENT_NOT_SUPPORTED"

    # Row-level deployment table.
    row_out = selected.copy()
    for m in METHODS:
        sel = row_out[f"selected_{m}"].to_numpy(dtype=int)
        row_out[f"deployed_dice_{m}"] = np.where(
            sel == 1,
            row_out["memo_dice"].to_numpy(dtype=float),
            row_out["source_dice"].to_numpy(dtype=float),
        )
        row_out[f"deployed_delta_{m}"] = np.where(
            sel == 1,
            row_out["memo_delta_dice"].to_numpy(dtype=float),
            0.0,
        )

    p_metrics = OUT_DIR / "B6_P02_POLYPGEN_EXACT50_UTILITY_METRICS.csv"
    p_ci = OUT_DIR / "B6_P02_POLYPGEN_EXACT50_BOOTSTRAP_CI.csv"
    p_delta = OUT_DIR / "B6_P02_POLYPGEN_EXACT50_PAIRED_DELTAS.csv"
    p_reps = OUT_DIR / "B6_P02_POLYPGEN_EXACT50_BOOTSTRAP_REPLICATES.csv"
    p_rows = OUT_DIR / "B6_P02_POLYPGEN_EXACT50_SELECTIONS_AND_DEPLOYMENT.csv"

    metrics_df.to_csv(p_metrics, index=False)
    ci_df.to_csv(p_ci, index=False)
    delta_df.to_csv(p_delta, index=False)
    reps.to_csv(p_reps, index=False)
    row_out.to_csv(p_rows, index=False)

    audit = {
        "status": "PASS",
        "gate": PASS_GATE,
        "version": VERSION,
        "decision": decision,
        "utility_supported": utility_supported,
        "safety_supported": safety_supported,
        "protocol": protocol_meta,
        "p01b": p01b_meta,
        "panel": panel_meta,
        "selection": pf,
        "primary": {
            "comparison": "SOURCE_PLUS_SIMPLE_MASK_CHANGE - SOURCE_STATE",
            "metric": "mean_deployed_dice",
            "point_delta": float(pr["point_delta"]),
            "ci95_low": float(pr["ci95_low"]),
            "ci95_high": float(pr["ci95_high"]),
        },
        "safety": {
            "comparison": "SOURCE_PLUS_SIMPLE_MASK_CHANGE - SOURCE_STATE",
            "metric": "prevented_harm_fraction",
            "point_delta": float(sr["point_delta"]),
            "ci95_low": float(sr["ci95_low"]),
            "ci95_high": float(sr["ci95_high"]),
        },
        "outputs": {},
    }

    for p in [p_metrics, p_ci, p_delta, p_reps, p_rows]:
        audit["outputs"][p.name] = {
            "path": str(p),
            "sha256": sha256_file(p),
            "bytes": p.stat().st_size,
        }

    p_audit = OUT_DIR / "B6_P02_POLYPGEN_EXACT50_UTILITY_AUDIT.json"
    write_json(p_audit, audit)

    report = "\n".join([
        "=" * 164,
        "SafeTTA B6-P0-2 POLYPGEN EXACT50 UTILITY COMPLETE",
        "",
        metrics_df[[
            "representation",
            "coverage",
            "mean_deployed_dice",
            "mean_deployed_delta_vs_source",
            "prevented_harm_fraction",
            "committed_harm_rate",
            "benefit_capture_fraction",
        ]].to_string(index=False),
        "",
        "PAIRED DELTAS:",
        delta_df.to_string(index=False),
        "",
        f"UTILITY_SUPPORTED={utility_supported}",
        f"SAFETY_SUPPORTED={safety_supported}",
        f"DECISION={decision}",
        f"GATE={PASS_GATE}",
        f"audit_json={p_audit}",
        f"stage_dir={OUT_DIR}",
        "=" * 164,
        "",
    ])
    (OUT_DIR / "B6_P02_POLYPGEN_EXACT50_UTILITY_REPORT.txt").write_text(
        report,
        encoding="utf-8",
    )

    print("\n" + report)


if __name__ == "__main__":
    main()
